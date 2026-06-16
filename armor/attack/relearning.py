"""
armor/attack/relearning.py
===========================
Relearning Attack simulation for evaluating unlearning robustness.

Threat Model
------------
An adversary has access to N forget-set samples (e.g. via data poisoning
or a memorisation oracle). They fine-tune the unlearned model on those
samples and measure how quickly the model "relearns" the forgotten data.

This is the standard robustness evaluation protocol in the unlearning
literature (Hu et al., 2024; Maini et al., 2024).

Attack Protocol
---------------
1. Load the unlearned model checkpoint
2. Fine-tune for R epochs on n_samples forget-set samples (n=50 default)
3. After each epoch, measure:
   - forget_accuracy : how much knowledge has been recovered?
   - retain_accuracy : is the retain set still intact?
4. Plot accuracy recovery curves (see notebooks/results_analysis.ipynb)

Key comparison metric: "accuracy recovery after K steps"
  GA     : typically recovers quickly (sharp minima → easy to relearn)
  NPO    : recovers more slowly (flatter minima)
  NPO+SAM: slowest recovery (SAM explicitly flattens the loss landscape)

This module is designed to be called AFTER unlearning is complete.
It uses the same GradientAscentUnlearner infrastructure internally,
but with normal gradient *descent* (not ascent) on the forget set.
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from armor.config import ARMORConfig
from armor.data import TOFUSample, get_relearning_subset, make_dataloader
from armor.eval.metrics import UnlearningEvaluator, EvaluationResult


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RelearningResult:
    """
    Stores the per-epoch accuracy recovery trajectory during a relearning attack.

    Attributes
    ----------
    method             : Name of the unlearning method that was attacked
    pre_attack_acc     : Forget accuracy before the attack (baseline)
    recovery_trajectory: [(epoch, forget_acc, retain_acc)] over attack epochs
    final_forget_acc   : Forget accuracy after all attack epochs
    recovery_pct       : How much accuracy was recovered relative to pre-unlearning
    elapsed_sec        : Attack wall-clock time
    """
    method: str
    pre_attack_forget_acc: float
    original_forget_acc: float          # Accuracy BEFORE any unlearning
    recovery_trajectory: list = field(default_factory=list)  # [(ep, f_acc, r_acc)]
    final_forget_acc: float = 0.0
    recovery_pct: float = 0.0
    elapsed_sec: float = 0.0

    def compute_recovery_pct(self):
        """
        Raw accuracy jump after the relearning attack.

        Simple metric: how many accuracy points did the attack recover?
            jump = final_forget_acc - pre_attack_forget_acc

        This is the most robust measure — avoids division edge-cases when
        the model starts near-random (debug mode) or NPO slightly raises acc.

        Also computes normalised recovery % where meaningful:
            recovery = jump / (original_acc - pre_attack_acc)
        Clamped to [0, 100]% and set to N/A if denominator <= 0.
        """
        self.acc_jump = self.final_forget_acc - self.pre_attack_forget_acc
        numerator     = self.acc_jump
        denominator   = self.original_forget_acc - self.pre_attack_forget_acc
        if denominator <= 1e-4:
            # Post-unlearn acc >= original → model never properly forgot
            # Recovery % is not meaningful; use raw jump instead
            self.recovery_pct = float("nan")
        else:
            self.recovery_pct = max(0.0, min(100.0,
                                  (numerator / denominator) * 100.0))

    def print_summary(self):
        print("\n" + "=" * 60)
        print(f"  RELEARNING ATTACK -- Method: {self.method}")
        print("=" * 60)
        print(f"  Original forget accuracy (pre-unlearn) : {self.original_forget_acc:.4f}")
        print(f"  Post-unlearn forget accuracy            : {self.pre_attack_forget_acc:.4f}")
        print(f"  Post-attack  forget accuracy            : {self.final_forget_acc:.4f}")
        print(f"  Knowledge recovery                      : {self.recovery_pct:.1f}%")
        print(f"  Attack wall time                        : {self.elapsed_sec:.1f}s")
        print("=" * 60)
        print("\n  Epoch | Forget Acc | Retain Acc")
        print("  " + "-" * 30)
        for ep, f_acc, r_acc in self.recovery_trajectory:
            print(f"  {ep:5d} | {f_acc:10.4f} | {r_acc:10.4f}")
        print()


# ──────────────────────────────────────────────────────────────────────────────
# Relearning Attacker
# ──────────────────────────────────────────────────────────────────────────────

class RelearningAttack:
    """
    Simulates a relearning attack on an unlearned model.

    The attacker fine-tunes the unlearned model on a small subset of
    the forget set and measures how much forgotten knowledge is recovered.

    Usage
    -----
    attacker = RelearningAttack(
        unlearned_model, tokenizer, cfg,
        forget_samples=forget_samples,
        retain_samples=retain_samples,
        method_name="NPO+SAM",
        original_forget_acc=0.85,   # Accuracy before unlearning
    )
    result = attacker.run()
    result.print_summary()
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cfg: ARMORConfig,
        forget_samples: list[TOFUSample],
        retain_samples: list[TOFUSample],
        method_name: str = "Unknown",
        original_forget_acc: float = 1.0,
    ):
        self.cfg       = cfg
        self.tokenizer = tokenizer
        self.device    = cfg.device
        self.forget_samples  = forget_samples
        self.retain_samples  = retain_samples
        self.method_name     = method_name
        self.original_forget_acc = original_forget_acc

        # Work on a copy so the original unlearned model stays intact
        print(f"[attack] Cloning unlearned model for relearning attack...")
        self.model = copy.deepcopy(model)
        self.model.to(self.device)

    def _compute_forget_loss(self, batch: dict) -> torch.Tensor:
        """Standard cross-entropy (gradient DESCENT — this is relearning)."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs.loss

    def run(self) -> RelearningResult:
        """
        Execute the relearning attack.

        Steps:
        1. Sample cfg.relearn_n_samples from forget_set
        2. Fine-tune the cloned model for cfg.relearn_epochs epochs
        3. Measure forget accuracy at each epoch
        4. Compute recovery percentage

        Returns
        -------
        RelearningResult with full recovery trajectory
        """
        cfg = self.cfg
        print(f"\n[attack] === Relearning Attack on '{self.method_name}' ===")
        print(f"[attack] Attack samples : {cfg.relearn_n_samples}")
        print(f"[attack] Attack epochs  : {cfg.relearn_epochs}")
        print(f"[attack] Attack LR      : {cfg.relearn_lr}")

        # ── Sample attack subset from forget set ───────────────────────────────
        attack_samples = get_relearning_subset(
            self.forget_samples, n=cfg.relearn_n_samples
        )
        attack_loader = make_dataloader(
            attack_samples, self.tokenizer, cfg,
            include_rephrases=False,   # Attacker only has original phrasing
            shuffle=True,
        )

        # Evaluation loaders (full sets for accurate measurement)
        eval_forget_loader = make_dataloader(
            self.forget_samples, self.tokenizer, cfg, shuffle=False
        )
        eval_retain_loader = make_dataloader(
            self.retain_samples, self.tokenizer, cfg, shuffle=False
        )

        evaluator = UnlearningEvaluator(self.model, self.tokenizer, cfg)

        # Measure pre-attack accuracy (state after unlearning)
        print("[attack] Measuring pre-attack accuracy...")
        pre_forget_acc = evaluator._compute_token_accuracy(eval_forget_loader)
        pre_retain_acc = evaluator._compute_token_accuracy(eval_retain_loader)
        print(f"[attack] Pre-attack forget acc : {pre_forget_acc:.4f}")
        print(f"[attack] Pre-attack retain acc : {pre_retain_acc:.4f}")

        result = RelearningResult(
            method=self.method_name,
            pre_attack_forget_acc=pre_forget_acc,
            original_forget_acc=self.original_forget_acc,
        )

        # ── Fine-tuning loop (relearning) ──────────────────────────────────────
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.relearn_lr,
        )
        self.model.train()
        t0 = time.time()
        trajectory = []

        for epoch in range(cfg.relearn_epochs):
            epoch_loss = 0.0
            n_batches  = 0

            for batch in tqdm(attack_loader,
                              desc=f"[attack] Epoch {epoch+1}/{cfg.relearn_epochs}",
                              leave=False):
                optimizer.zero_grad()
                loss = self._compute_forget_loss(batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            # Evaluate after each epoch
            self.model.eval()
            f_acc = evaluator._compute_token_accuracy(eval_forget_loader)
            r_acc = evaluator._compute_token_accuracy(eval_retain_loader)
            self.model.train()

            trajectory.append((epoch + 1, f_acc, r_acc))
            print(f"[attack] Epoch {epoch+1:02d} | "
                  f"loss={epoch_loss/max(n_batches,1):.4f} | "
                  f"forget_acc={f_acc:.4f} | retain_acc={r_acc:.4f}")

        result.recovery_trajectory = trajectory
        result.final_forget_acc    = trajectory[-1][1] if trajectory else pre_forget_acc
        result.elapsed_sec         = time.time() - t0
        result.compute_recovery_pct()

        return result
