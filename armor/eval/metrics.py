"""
armor/eval/metrics.py
=====================
Evaluation metrics for LLM unlearning.

Metrics measured
----------------

1. Forget Quality (FQ)
   How well has the model forgotten the forget set?
   FQ = 1 - token_accuracy(model, forget_set)
   → Higher is better (0 = no forgetting, 1 = complete forgetting)

2. Retain Accuracy (RA)
   Has the model preserved its knowledge of the retain set?
   RA = token_accuracy(model, retain_set)
   → Higher is better (1 = no catastrophic forgetting)

3. ROUGE Score (ROUGE-1 / ROUGE-L)
   How similar are generated answers to ground-truth answers?
   Measured on both forget and retain sets.
   → Forget-ROUGE should be LOW (model generates wrong/gibberish answers)
   → Retain-ROUGE should be HIGH (model still answers correctly)

4. Model Utility (MU)
   Synonym for retain accuracy — measures overall capability preservation.

Evaluation modes
----------------
  A) Teacher-forced accuracy: Compute loss on gold labels (fast, no generation)
  B) Generative ROUGE: Actually generate answers and score them (slower)

Both modes are implemented. Use teacher-forced for quick epoch-level monitoring
and generative ROUGE for final paper-quality evaluation.

Cross-modal NOTE (Step 2):
  Add a VQA accuracy metric here for LLaVA evaluation.
  The infrastructure (EvaluationResult, print_table) stays unchanged.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from rouge_score import rouge_scorer
from tqdm import tqdm

from armor.config import ARMORConfig
from armor.data import TOFUSample


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Holds all evaluation metrics for one model checkpoint."""
    method: str = "Unknown"

    # Forget set metrics
    forget_quality: float   = 0.0   # 1 - forget_accuracy (higher = better)
    forget_accuracy: float  = 0.0   # Token accuracy on forget set (lower = better)
    forget_rouge1: float    = 0.0   # ROUGE-1 on forget set (lower = better)
    forget_rougeL: float    = 0.0   # ROUGE-L on forget set (lower = better)

    # Retain set metrics
    retain_accuracy: float  = 0.0   # Token accuracy on retain set (higher = better)
    retain_rouge1: float    = 0.0   # ROUGE-1 on retain set (higher = better)
    retain_rougeL: float    = 0.0   # ROUGE-L on retain set (higher = better)

    # Composite scores
    model_utility: float    = 0.0   # Alias for retain_accuracy

    # MIA audit score (set externally by MembershipInferenceAuditor)
    mia_auroc: float        = -1.0  # AUROC from MIA (-1 = not computed)

    def print_table(self):
        """Pretty-print the evaluation results."""
        print("\n" + "=" * 60)
        print(f"  ARMOR Evaluation -- Method: {self.method}")
        print("=" * 60)
        print(f"  {'Metric':<30} {'Value':>10}")
        print("  " + "-" * 40)
        print(f"  {'Forget Quality (higher=better)':<30} {self.forget_quality:>10.4f}")
        print(f"  {'Forget Accuracy (lower=better)':<30} {self.forget_accuracy:>10.4f}")
        print(f"  {'Forget ROUGE-1 (lower=better)':<30} {self.forget_rouge1:>10.4f}")
        print(f"  {'Forget ROUGE-L (lower=better)':<30} {self.forget_rougeL:>10.4f}")
        print("  " + "-" * 40)
        print(f"  {'Retain Accuracy (higher=better)':<30} {self.retain_accuracy:>10.4f}")
        print(f"  {'Retain ROUGE-1 (higher=better)':<30} {self.retain_rouge1:>10.4f}")
        print(f"  {'Retain ROUGE-L (higher=better)':<30} {self.retain_rougeL:>10.4f}")
        print("  " + "-" * 40)
        if self.mia_auroc >= 0:
            print(f"  {'MIA AUROC (audit, 0.5=best)':<30} {self.mia_auroc:>10.4f}")
        print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────────────────────────

class UnlearningEvaluator:
    """
    Comprehensive evaluator for unlearned LLMs.

    Usage
    -----
    evaluator = UnlearningEvaluator(model, tokenizer, cfg)
    result    = evaluator.evaluate(forget_samples, retain_samples,
                                   method_name="GA")
    result.print_table()
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cfg: ARMORConfig,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.device    = cfg.device

        self._rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rougeL"], use_stemmer=True
        )

    # ── Teacher-forced accuracy ────────────────────────────────────────────────

    @torch.no_grad()
    def _compute_token_accuracy(
        self, loader: DataLoader
    ) -> float:
        """
        Compute token-level accuracy (% of label tokens predicted correctly).

        This is the fastest way to measure model behaviour on a set without
        full text generation.
        """
        self.model.eval()
        total_correct = 0
        total_tokens  = 0

        for batch in tqdm(loader, desc="  [eval] token accuracy", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)
            labels    = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
            )
            logits = outputs.logits  # (B, T, V)

            # Shift: prediction at t → label at t+1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            predictions = shift_logits.argmax(dim=-1)   # (B, T-1)

            # Only evaluate on non-padding positions
            mask    = (shift_labels != -100)
            correct = (predictions == shift_labels) & mask

            total_correct += correct.sum().item()
            total_tokens  += mask.sum().item()

        accuracy = total_correct / max(total_tokens, 1)
        return accuracy

    # ── Generative ROUGE ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _compute_rouge(
        self,
        samples: list[TOFUSample],
        n_samples: Optional[int] = None,
    ) -> tuple[float, float]:
        """
        Generate answers for each sample and compute ROUGE against ground truth.

        Parameters
        ----------
        samples   : List of TOFUSample to evaluate
        n_samples : If set, only evaluate the first n_samples (for speed)

        Returns
        -------
        (rouge1_f, rougeL_f) — macro-averaged F1 scores
        """
        self.model.eval()
        if n_samples is not None:
            samples = samples[:n_samples]

        rouge1_scores = []
        rougeL_scores = []

        for sample in tqdm(samples, desc="  [eval] ROUGE generation", leave=False):
            # Format the question as a prompt (without the answer)
            prompt = f"Question: {sample.question}\nAnswer:"
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.cfg.max_seq_len // 2,
            ).to(self.device)

            # Generate
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,       # Greedy for reproducibility
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Decode only the new tokens (not the prompt)
            n_prompt_tokens = inputs["input_ids"].shape[-1]
            generated_ids   = output_ids[0, n_prompt_tokens:]
            generated_text  = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()

            # Score against ground truth
            scores = self._rouge_scorer.score(sample.answer, generated_text)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)

        return (
            sum(rouge1_scores) / max(len(rouge1_scores), 1),
            sum(rougeL_scores) / max(len(rougeL_scores), 1),
        )

    # ── Main evaluation entry point ────────────────────────────────────────────

    def evaluate(
        self,
        forget_samples: list[TOFUSample],
        retain_samples: list[TOFUSample],
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        method_name: str = "Unknown",
        run_rouge: bool = True,
        max_rouge_samples: int = 50,
    ) -> EvaluationResult:
        """
        Run full evaluation on forget and retain sets.

        Parameters
        ----------
        forget_samples   : Raw TOFUSample list (for ROUGE generation)
        retain_samples   : Raw TOFUSample list (for ROUGE generation)
        forget_loader    : DataLoader (for teacher-forced accuracy)
        retain_loader    : DataLoader (for teacher-forced accuracy)
        method_name      : Label for the results table
        run_rouge        : If True, run generative ROUGE (slower)
        max_rouge_samples: Cap the number of samples for ROUGE (speed)

        Returns
        -------
        EvaluationResult — all metrics populated
        """
        print(f"\n[eval] Evaluating method: {method_name}")

        # ── Token accuracy ─────────────────────────────────────────────────────
        print("[eval] Computing forget token accuracy...")
        forget_acc = self._compute_token_accuracy(forget_loader)

        print("[eval] Computing retain token accuracy...")
        retain_acc = self._compute_token_accuracy(retain_loader)

        result = EvaluationResult(
            method=method_name,
            forget_accuracy=forget_acc,
            forget_quality=1.0 - forget_acc,   # Higher = more forgotten
            retain_accuracy=retain_acc,
            model_utility=retain_acc,
        )

        # ── Generative ROUGE ───────────────────────────────────────────────────
        if run_rouge:
            print("[eval] Computing forget ROUGE (generative)...")
            f_rouge1, f_rougeL = self._compute_rouge(
                forget_samples, n_samples=max_rouge_samples
            )
            result.forget_rouge1 = f_rouge1
            result.forget_rougeL = f_rougeL

            print("[eval] Computing retain ROUGE (generative)...")
            r_rouge1, r_rougeL = self._compute_rouge(
                retain_samples, n_samples=max_rouge_samples
            )
            result.retain_rouge1 = r_rouge1
            result.retain_rougeL = r_rougeL

        return result
