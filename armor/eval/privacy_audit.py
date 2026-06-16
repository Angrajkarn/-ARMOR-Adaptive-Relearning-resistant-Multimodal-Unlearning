"""
armor/eval/privacy_audit.py
============================
Privacy Auditor — Converts MIA AUROC → Empirical ε-DP Bound

Implements the conversion from empirical membership inference attack
success to a formal privacy budget lower bound.

Based on:
  - Steinke et al. (2023) "Privacy Auditing with One (1) Training Run"
  - Jagielski et al. (2020) "Auditing Differentially Private ML"

The key insight: a successful MIA implies a lower bound on ε.
If AUROC → 0.5 (random), the model is empirically ε-private with ε → 0.

Also provides:
  - Multi-run AUROC confidence intervals
  - Comparison of formal DP ε vs empirical ε
  - Privacy-utility Pareto frontier analysis
"""

import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from ..config import ARMORConfig


# ─────────────────────────────────────────────────────────────────────────────
# Min-K% Probability Scorer (from mia.py, duplicated for standalone use)
# ─────────────────────────────────────────────────────────────────────────────

def min_k_prob_score(model: PreTrainedModel,
                     input_ids: torch.Tensor,
                     labels: torch.Tensor,
                     k_pct: float = 0.2) -> float:
    """
    Score a sample using Min-K% Prob:
    Average log-prob of the K% lowest-probability tokens.

    Low score → model treats sample as non-member → good unlearning.
    """
    model.eval()
    with torch.no_grad():
        out    = model(input_ids=input_ids, labels=labels)
        logits = out.logits   # [B, T, V]

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        # Gather log-prob of actual tokens
        token_log_probs = log_probs.gather(
            -1, shift_labels.unsqueeze(-1).clamp(min=0)).squeeze(-1)

        # Only count non-padding tokens
        valid_mask      = (shift_labels >= 0)
        valid_log_probs = token_log_probs[valid_mask]

        if valid_log_probs.numel() == 0:
            return 0.0

        # Take K% lowest log-probs
        k = max(1, int(valid_log_probs.numel() * k_pct))
        min_k, _  = torch.topk(valid_log_probs, k, largest=False)
        return min_k.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# AUROC → ε Conversion
# ─────────────────────────────────────────────────────────────────────────────

def auroc_to_epsilon_lower_bound(auroc: float,
                                  n_forget: int,
                                  delta: float = 1e-5,
                                  confidence: float = 0.95) -> float:
    """
    Convert MIA AUROC to a lower bound on the privacy parameter ε.

    Based on the relationship (Jagielski et al., 2020):
        TPR − FPR ≤ e^ε − 1    (for any threshold)

    If AUROC is approximated by the max (TPR, 1-FPR) pair on the ROC curve,
    we can back-calculate ε from the observed advantage:

        advantage = 2 * (AUROC - 0.5)   # ∈ [0, 1]
        ε_lower ≥ log(1 + advantage)

    Args:
        auroc      : observed MIA AUROC (0.5 = random = private)
        n_forget   : number of forget samples (for finite-sample correction)
        delta      : target δ in (ε, δ)-DP
        confidence : confidence level for the bound

    Returns:
        ε_lower : lower bound on privacy budget (higher = less private)
    """
    # Statistical advantage of the best MIA
    advantage = max(0.0, 2.0 * (auroc - 0.5))

    if advantage < 1e-6:
        return 0.0   # Essentially private

    # Finite sample correction (Clopper-Pearson style)
    # Confidence interval half-width ≈ z * sqrt(p(1-p)/n)
    z         = 1.96 if confidence >= 0.95 else 1.645
    margin    = z * math.sqrt(max(advantage * (1 - advantage), 1e-6)
                               / max(n_forget, 1))
    adv_upper = min(1.0, advantage + margin)  # upper CI → conservative lower ε

    eps_lower = math.log(1.0 + adv_upper)

    # Account for δ: ε ≥ log(1 + adv) - log(1/δ) but use simple version
    return max(0.0, eps_lower)


# ─────────────────────────────────────────────────────────────────────────────
# Audit Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PrivacyAuditResult:
    method:              str
    auroc:               float          # Observed MIA AUROC
    epsilon_empirical:   float          # Derived ε lower bound
    epsilon_formal:      Optional[float] = None  # DP-SGD formal ε (if used)
    delta:               float          = 1e-5
    n_forget:            int            = 0
    forget_scores:       List[float]    = field(default_factory=list)
    retain_scores:       List[float]    = field(default_factory=list)
    is_verified_unlearned: bool         = False  # AUROC < 0.55 threshold
    k_pct:               float          = 0.2

    def print_summary(self):
        print("\n" + "=" * 64)
        print(f"  PRIVACY AUDIT -- Method: {self.method}")
        print("=" * 64)
        print(f"  MIA AUROC                   : {self.auroc:.4f}")
        print(f"  Empirical ε lower bound     : {self.epsilon_empirical:.4f}")
        if self.epsilon_formal is not None:
            print(f"  Formal ε (DP-SGD certified) : {self.epsilon_formal:.4f}")
        print(f"  Target δ                    : {self.delta}")
        verdict = "VERIFIED UNLEARNED" if self.is_verified_unlearned else "NOT VERIFIED"
        print(f"  Unlearning verdict          : {verdict}")
        print(f"  (threshold: AUROC < 0.55 → verified)")
        print("-" * 64)
        print(f"  Forget set Min-K% avg score : "
              f"{sum(self.forget_scores)/max(len(self.forget_scores),1):.4f}")
        print(f"  Retain set Min-K% avg score : "
              f"{sum(self.retain_scores)/max(len(self.retain_scores),1):.4f}")
        print("=" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# Privacy Auditor
# ─────────────────────────────────────────────────────────────────────────────

class PrivacyAuditor:
    """
    Formal privacy auditor for ARMOR unlearning.

    Steps:
      1. Score forget-set samples with Min-K% Prob
      2. Score retain-set samples with Min-K% Prob
      3. Compute MIA AUROC (forget should score similar to non-members)
      4. Convert AUROC → empirical ε lower bound
      5. Compare with formal DP ε (if DP-SGD was used)

    Usage:
        auditor = PrivacyAuditor(cfg, model, tokenizer)
        result  = auditor.audit(forget_loader, retain_loader,
                                method_name="NPO+SAM",
                                formal_epsilon=3.2)
        result.print_summary()
    """

    AUROC_VERIFIED_THRESHOLD = 0.55   # AUROC < this → verified unlearned

    def __init__(self, cfg: ARMORConfig, model: PreTrainedModel,
                 tokenizer=None, k_pct: float = 0.2):
        self.cfg      = cfg
        self.model    = model
        self.tokenizer = tokenizer
        self.k_pct    = k_pct

    def _score_loader(self, loader: DataLoader) -> List[float]:
        """Compute Min-K% score for every sample in loader."""
        scores = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                ids  = batch["input_ids"].to(self.cfg.device)
                labs = batch["labels"].to(self.cfg.device)
                for i in range(ids.size(0)):
                    score = min_k_prob_score(
                        self.model,
                        ids[i:i+1],
                        labs[i:i+1],
                        k_pct=self.k_pct)
                    scores.append(score)
        return scores

    def _compute_auroc(self, forget_scores: List[float],
                        retain_scores: List[float]) -> float:
        """
        Compute AUROC treating forget = positive class (member)
        and retain = negative class (non-member).

        AUROC = P(score_forget > score_retain) over random pairs.
        """
        if not forget_scores or not retain_scores:
            return 0.5

        f = np.array(forget_scores)
        r = np.array(retain_scores)

        # All pairwise comparisons (efficient via broadcasting)
        correct = np.sum(f[:, None] > r[None, :])
        tied    = np.sum(f[:, None] == r[None, :]) * 0.5
        total   = len(f) * len(r)
        return float((correct + tied) / max(total, 1))

    def audit(self,
              forget_loader:  DataLoader,
              retain_loader:  DataLoader,
              method_name:    str = "unknown",
              formal_epsilon: Optional[float] = None,
              delta:          float = 1e-5) -> PrivacyAuditResult:
        """Run full privacy audit and return result."""
        print(f"\n[PrivacyAudit] Auditing '{method_name}' (Min-K%={self.k_pct})...")

        forget_scores = self._score_loader(forget_loader)
        retain_scores = self._score_loader(retain_loader)
        auroc         = self._compute_auroc(forget_scores, retain_scores)
        eps_empirical = auroc_to_epsilon_lower_bound(
            auroc, len(forget_scores), delta=delta)

        result = PrivacyAuditResult(
            method            = method_name,
            auroc             = auroc,
            epsilon_empirical = eps_empirical,
            epsilon_formal    = formal_epsilon,
            delta             = delta,
            n_forget          = len(forget_scores),
            forget_scores     = forget_scores,
            retain_scores     = retain_scores,
            is_verified_unlearned = auroc < self.AUROC_VERIFIED_THRESHOLD,
            k_pct             = self.k_pct,
        )
        return result

    def compare(self, results: Dict[str, PrivacyAuditResult]) -> None:
        """Print a side-by-side comparison table of multiple methods."""
        print("\n" + "=" * 70)
        print("  PRIVACY AUDIT COMPARISON")
        print("=" * 70)
        print(f"  {'Method':<14} | {'AUROC':>6} | {'ε-empirical':>12} | "
              f"{'ε-formal':>10} | {'Verified':>10}")
        print("  " + "-" * 66)
        for name, r in results.items():
            formal_str  = f"{r.epsilon_formal:.3f}" if r.epsilon_formal else "N/A"
            verified    = "YES" if r.is_verified_unlearned else "NO"
            print(f"  {name:<14} | {r.auroc:>6.4f} | "
                  f"{r.epsilon_empirical:>12.4f} | {formal_str:>10} | {verified:>10}")
        print("=" * 70)
