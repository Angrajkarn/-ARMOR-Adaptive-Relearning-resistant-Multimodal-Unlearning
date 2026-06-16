"""
armor/eval/mia.py
=================
Membership Inference Attack (MIA) — Verifiable Unlearning Audit Score.

Why MIA?
--------
After unlearning, we need to formally verify that the model has actually
forgotten the target data — not just declined to answer. Membership
Inference Attacks attempt to distinguish "seen during training" from
"not seen during training" samples.

If unlearning was successful, the model should treat forget-set samples
as if they were never in the training set → MIA AUROC ≈ 0.5 (random chance).
If AUROC >> 0.5, the model still "remembers" those samples.

Method: Min-K% Prob (Shi et al., 2024)
---------------------------------------
One of the best-performing black-box MIA methods for LLMs.

For each sample x:
  1. Compute per-token log-probabilities: log P(x_t | x_{<t})
  2. Take the K% tokens with the LOWEST log-prob (most surprising tokens)
  3. Average those K% log-probs → MIA score s(x)

Intuition: Members (training data) tend to have higher minimum log-probs
because the model has memorised them. Non-members have lower "floor" scores.

Evaluation:
  • Score the forget set (should look like non-members → low scores)
  • Score a reference "non-member" set (typically the retain set)
  • Compute AUROC of (non-member label, score)
  • AUROC → 0.5 means forget set is indistinguishable from non-members ✓

References
----------
  • Shi et al., "Detecting Pretraining Data from Large Language Models"
    (2024) [arXiv:2310.16789] — Min-K% Prob method
  • Carlini et al., "Membership Inference Attacks from First Principles"
    (2022) — foundational MIA paper
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from tqdm import tqdm

from armor.config import ARMORConfig
from armor.data import TOFUSample


@dataclass
class MIAResult:
    """Result of the MIA audit."""
    method: str
    auroc: float            # Key metric: 0.5 = perfect unlearning, 1.0 = not unlearned
    forget_scores: list     # MIA scores for forget set (should be low)
    nonmember_scores: list  # MIA scores for non-member (retain) set
    k_percent: float        # The K% used in Min-K% Prob

    @property
    def audit_verdict(self) -> str:
        """
        Informal verdict based on AUROC.
        AUROC close to 0.5 → unlearning verifiable.
        """
        if self.auroc <= 0.55:
            return "✓ VERIFIED — forget set indistinguishable from non-members"
        elif self.auroc <= 0.65:
            return "⚠ PARTIAL — some residual membership signal detected"
        else:
            return "✗ FAILED  — model still memorises forget set (AUROC > 0.65)"

    def summary(self):
        print(f"\n[MIA] Method: {self.method}")
        print(f"[MIA] Min-K% k={self.k_percent:.0%} | AUROC = {self.auroc:.4f}")
        print(f"[MIA] Verdict: {self.audit_verdict}")
        print(f"[MIA] Forget  avg score : {np.mean(self.forget_scores):.4f}")
        print(f"[MIA] NonMem  avg score : {np.mean(self.nonmember_scores):.4f}")


class MembershipInferenceAuditor:
    """
    Membership Inference Attack auditor using the Min-K% Prob method.

    This provides a formal, quantitative audit score for the verifiable
    unlearning component of ARMOR.

    Usage
    -----
    auditor = MembershipInferenceAuditor(model, cfg)
    result  = auditor.audit(forget_samples, nonmember_samples, method_name="NPO+SAM")
    result.summary()
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer,
        cfg: ARMORConfig,
        k_percent: float = 0.20,   # Use 20% lowest-prob tokens
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.device    = cfg.device
        self.k_percent = k_percent

    @torch.no_grad()
    def _min_k_score(self, input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     labels: torch.Tensor) -> list[float]:
        """
        Compute Min-K% Prob scores for a batch of samples.

        Parameters
        ----------
        input_ids, attention_mask, labels : standard tokenizer outputs

        Returns
        -------
        scores : list of floats, one per sample in the batch
        """
        self.model.eval()

        input_ids  = input_ids.to(self.device)
        attn_mask  = attention_mask.to(self.device)
        labels_dev = labels.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
        )
        logits = outputs.logits  # (B, T, V)

        # Shift for causal LM alignment
        shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
        shift_labels = labels_dev[:, 1:].contiguous()   # (B, T-1)

        log_probs_all = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)

        batch_scores = []
        for i in range(input_ids.size(0)):
            # Get per-token log-probs for the actual label token
            valid_mask = (shift_labels[i] != -100)
            valid_labels = shift_labels[i][valid_mask]

            if valid_labels.numel() == 0:
                batch_scores.append(0.0)
                continue

            token_log_probs = log_probs_all[i][valid_mask]  # (n_valid, V)
            # Gather log-prob of actual token
            actual_log_probs = token_log_probs.gather(
                dim=-1, index=valid_labels.unsqueeze(-1)
            ).squeeze(-1)  # (n_valid,)

            # Min-K%: take the K% lowest-prob tokens
            k = max(1, int(self.k_percent * actual_log_probs.numel()))
            # Sort ascending (lowest first) and take top-k
            min_k_probs = actual_log_probs.sort().values[:k]

            # Score = mean of the K% lowest log-probs
            score = min_k_probs.mean().item()
            batch_scores.append(score)

        return batch_scores

    def _score_dataset(self, loader: DataLoader, desc: str) -> list[float]:
        """Score all samples in a DataLoader and return a list of MIA scores."""
        all_scores = []
        for batch in tqdm(loader, desc=f"  [MIA] {desc}", leave=False):
            scores = self._min_k_score(
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )
            all_scores.extend(scores)
        return all_scores

    def audit(
        self,
        forget_loader: DataLoader,
        nonmember_loader: DataLoader,
        method_name: str = "Unknown",
    ) -> MIAResult:
        """
        Run the MIA audit.

        AUROC interpretation:
          0.5  → perfect unlearning (forget ≡ non-member in model's eyes)
          1.0  → no unlearning (forget set perfectly distinguishable)

        Parameters
        ----------
        forget_loader    : DataLoader for forget set (should look like non-members)
        nonmember_loader : DataLoader for genuine non-members (retain set)
        method_name      : Label for logging

        Returns
        -------
        MIAResult with AUROC and raw scores
        """
        print(f"\n[MIA] Auditing '{method_name}' with Min-K% (k={self.k_percent:.0%})")

        forget_scores    = self._score_dataset(forget_loader,    "forget set")
        nonmember_scores = self._score_dataset(nonmember_loader, "non-members")

        # Labels: 0 = non-member, 1 = member (forget set = suspected member)
        # A high MIA score → model thinks it's a member
        # After unlearning: forget scores should be similar to nonmember scores
        n_forget    = len(forget_scores)
        n_nonmember = len(nonmember_scores)

        y_true  = [1] * n_forget + [0] * n_nonmember
        y_score = forget_scores + nonmember_scores

        # AUROC of "is this a member?"
        # Perfect unlearning → AUROC = 0.5 (can't distinguish)
        try:
            auroc = roc_auc_score(y_true, y_score)
        except ValueError:
            auroc = 0.5  # All same class edge case

        result = MIAResult(
            method=method_name,
            auroc=auroc,
            forget_scores=forget_scores,
            nonmember_scores=nonmember_scores,
            k_percent=self.k_percent,
        )
        result.summary()
        return result
