"""
scripts/run_npo_sam.py
=======================
Run NPO + SAM (Sharpness-Aware Minimization) unlearning — ARMOR's core method.

This is the relearning-resistant version. SAM finds flat minima in the loss
landscape, making the unlearned model robust to relearning attacks.

Usage
-----
  python scripts/run_npo_sam.py --debug
  python scripts/run_npo_sam.py --model mistral-7b --qlora --sam-rho 0.05

How it works
------------
  SAM wraps the AdamW optimizer used inside NPOUnlearner.
  The two-phase update:
    Phase 1: Perturb θ → θ + ε̂ (find sharpest direction)
    Phase 2: Update using gradient at θ + ε̂ (descend from the worst-case point)
  This forces the algorithm to converge to flat minima.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from armor.config import ARMORConfig
from armor.data   import load_tofu_splits, make_dataloader
from armor.model  import get_model_and_tokenizer, get_frozen_reference_model, save_checkpoint
from armor.unlearn.npo         import NPOUnlearner, compute_token_log_probs
from armor.unlearn.sam_wrapper import SAMOptimizer, sam_training_step
from armor.unlearn.gradient_ascent import UnlearningResult, _infinite_iter
from armor.eval.metrics  import UnlearningEvaluator
from armor.eval.mia      import MembershipInferenceAuditor


def parse_args():
    p = argparse.ArgumentParser(description="ARMOR — NPO + SAM")
    p.add_argument("--debug",      action="store_true")
    p.add_argument("--model",      default="debug",
                   choices=["debug", "mistral-7b", "llama2-7b"])
    p.add_argument("--qlora",      action="store_true")
    p.add_argument("--hf-token",   default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--npo-beta",   type=float, default=None)
    p.add_argument("--sam-rho",    type=float, default=None,
                   help="SAM neighbourhood radius ρ (default: 0.05)")
    p.add_argument("--sam-adaptive", action="store_true",
                   help="Use Adaptive SAM (ASAM)")
    p.add_argument("--output-dir", default="outputs/npo_sam")
    p.add_argument("--no-rouge",   action="store_true")
    p.add_argument("--run-mia",    action="store_true")
    p.add_argument("--no-save",    action="store_true",
                   help="Skip saving checkpoint (for smoke tests)")
    return p.parse_args()


class NPOSAMUnlearner:
    """
    NPO unlearning with SAM optimizer for relearning-resistant flat minima.

    Extends NPOUnlearner by replacing AdamW with SAMOptimizer(AdamW).
    The SAM two-phase update (first_step + second_step) requires two
    forward passes per batch — this doubles compute but greatly improves
    robustness against relearning attacks.
    """

    def __init__(self, model, ref_model, cfg: ARMORConfig, sam_rho: float = 0.05):
        self.model     = model
        self.ref_model = ref_model
        self.cfg       = cfg
        self.device    = cfg.device
        self.sam_rho   = sam_rho

        # Freeze reference model
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()

        # SAM wraps AdamW
        base_optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.unlearn_lr,
            weight_decay=cfg.weight_decay,
        )
        self.sam_optimizer = SAMOptimizer(
            base_optimizer,
            model=model,
            rho=sam_rho,
            adaptive=cfg.sam_adaptive,
        )

    def _compute_npo_loss(self, forget_batch: dict,
                           retain_batch=None) -> torch.Tensor:
        """NPO + retain loss computation (same as NPOUnlearner)."""
        import torch.nn.functional as F

        input_ids = forget_batch["input_ids"].to(self.device)
        attn_mask = forget_batch["attention_mask"].to(self.device)
        labels    = forget_batch["labels"].to(self.device)

        policy_log_probs = compute_token_log_probs(
            self.model, input_ids, attn_mask, labels
        )
        with torch.no_grad():
            ref_log_probs = compute_token_log_probs(
                self.ref_model, input_ids, attn_mask, labels
            )

        log_ratio = policy_log_probs - ref_log_probs
        npo_loss  = -F.logsigmoid(self.cfg.npo_beta * log_ratio).mean()

        if retain_batch is not None:
            retain_batch = {k: v.to(self.device) for k, v in retain_batch.items()}
            retain_out = self.model(**retain_batch)
            npo_loss   = npo_loss + self.cfg.npo_retain_coeff * retain_out.loss

        return npo_loss

    def run(self, forget_loader, retain_loader=None) -> UnlearningResult:
        """Run NPO+SAM unlearning with the two-phase SAM update per batch."""
        cfg   = self.cfg
        model = self.model
        model.train()

        retain_iter = _infinite_iter(retain_loader) if retain_loader else None

        epoch_losses, forget_losses, retain_losses = [], [], []
        total_steps = 0
        t0 = time.time()

        for epoch in range(cfg.unlearn_epochs):
            epoch_total  = 0.0
            n_batches    = 0

            pbar = tqdm(forget_loader,
                        desc=f"[NPO+SAM] Epoch {epoch+1}/{cfg.unlearn_epochs}",
                        leave=False)

            for step, forget_batch in enumerate(pbar):
                retain_batch = next(retain_iter) if retain_iter else None

                # ── SAM two-phase update ───────────────────────────────────────
                # We need to call the loss function twice.
                # Use a closure that captures the current batch.
                def loss_closure():
                    return self._compute_npo_loss(forget_batch, retain_batch)

                # Phase 1: compute loss + grad, perturb weights
                self.sam_optimizer.zero_grad()
                loss = loss_closure()
                loss.backward()
                self.sam_optimizer.first_step(zero_grad=True)

                # Phase 2: compute loss at perturbed point + update
                loss2 = loss_closure()
                loss2.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                self.sam_optimizer.second_step(zero_grad=True)

                total_steps += 1
                epoch_total += loss.item()
                n_batches   += 1

                pbar.set_postfix({
                    "loss1": f"{loss.item():.3f}",
                    "loss2": f"{loss2.item():.3f}",
                })

            avg = epoch_total / max(n_batches, 1)
            epoch_losses.append((epoch + 1, avg))
            forget_losses.append((epoch + 1, avg))
            retain_losses.append((epoch + 1, 0.0))

            print(f"[NPO+SAM] Epoch {epoch+1:02d} | loss={avg:.4f}")

        elapsed = time.time() - t0
        print(f"[NPO+SAM] Training complete in {elapsed:.1f}s ({total_steps} steps)")

        return UnlearningResult(
            method="NPO+SAM",
            epoch_losses=epoch_losses,
            forget_losses=forget_losses,
            retain_losses=retain_losses,
            total_steps=total_steps,
            elapsed_sec=elapsed,
        )


def main():
    args = parse_args()

    cfg = ARMORConfig(
        debug=args.debug,
        model_key=args.model if not args.debug else "debug",
        use_qlora=args.qlora,
        hf_token=args.hf_token,
        output_dir=args.output_dir,
    )
    if args.epochs:   cfg.unlearn_epochs = args.epochs
    if args.lr:       cfg.unlearn_lr     = args.lr
    if args.npo_beta: cfg.npo_beta       = args.npo_beta
    if args.sam_rho:  cfg.sam_rho        = args.sam_rho
    if args.sam_adaptive: cfg.sam_adaptive = True

    print("=" * 60)
    print(f"  ARMOR — NPO + SAM (Relearning-Resistant)")
    print(f"  Model   : {cfg.model_name}")
    print(f"  Device  : {cfg.device}")
    print(f"  SAM ρ   : {cfg.sam_rho}")
    print(f"  NPO β   : {cfg.npo_beta}")
    print("=" * 60)

    forget_samples, retain_samples = load_tofu_splits(cfg)
    model, tokenizer = get_model_and_tokenizer(cfg)
    ref_model        = get_frozen_reference_model(model, cfg)

    eval_forget_loader = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    eval_retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)

    evaluator = UnlearningEvaluator(model, tokenizer, cfg)
    print("\n[main] Pre-unlearning evaluation:")
    pre_result = evaluator.evaluate(
        forget_samples, retain_samples,
        eval_forget_loader, eval_retain_loader,
        method_name="Pre-unlearning",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50,
    )
    pre_result.print_table()

    forget_loader = make_dataloader(
        forget_samples, tokenizer, cfg,
        include_rephrases=cfg.use_rephrase_augmentation,
        shuffle=True,
    )
    retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=True)

    print(f"\n[main] Starting NPO+SAM unlearning (ρ={cfg.sam_rho})...")
    unlearner = NPOSAMUnlearner(model, ref_model, cfg, sam_rho=cfg.sam_rho)
    train_result = unlearner.run(forget_loader, retain_loader)

    print("\n[main] Post-unlearning evaluation:")
    post_result = evaluator.evaluate(
        forget_samples, retain_samples,
        eval_forget_loader, eval_retain_loader,
        method_name="NPO+SAM",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50,
    )
    post_result.print_table()

    if args.run_mia:
        auditor   = MembershipInferenceAuditor(model, tokenizer, cfg)
        mia_result = auditor.audit(eval_forget_loader, eval_retain_loader, "NPO+SAM")
        post_result.mia_auroc = mia_result.auroc
        post_result.print_table()

    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        save_checkpoint(model, tokenizer,
                        os.path.join(args.output_dir, "npo_sam_unlearned"), cfg)

    print(f"\n[main] NPO+SAM complete.")
    print(f"  Forget quality : {post_result.forget_quality:.4f}")
    print(f"  Retain accuracy: {post_result.retain_accuracy:.4f}")


if __name__ == "__main__":
    main()
