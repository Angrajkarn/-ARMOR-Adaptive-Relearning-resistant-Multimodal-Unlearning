"""
scripts/run_continual_unlearn.py
================================
Run Lifelong (Continual) unlearning pipeline on TOFU.
Simulates sequential unlearning requests (cohorts) arriving over time.

Usage
-----
  python scripts/run_continual_unlearn.py --debug
  python scripts/run_continual_unlearn.py --model mistral-7b --qlora --run-mia
"""

import argparse
import os
import sys
import warnings

# Set UTF-8 encoding for stdout and stderr on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Make armor importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore", message=".*symlink.*")

import torch
from armor.config import ARMORConfig
from armor.data   import load_tofu_splits, make_dataloader
from armor.model  import get_model_and_tokenizer, save_checkpoint
from armor.unlearn.continual_unlearner import ContinualUnlearner
from armor.eval.metrics  import UnlearningEvaluator
from armor.eval.mia      import MembershipInferenceAuditor


def parse_args():
    p = argparse.ArgumentParser(description="ARMOR — Continual/Lifelong Unlearning")
    p.add_argument("--debug",      action="store_true")
    p.add_argument("--model",      default="debug", choices=["debug","mistral-7b","llama2-7b"])
    p.add_argument("--qlora",      action="store_true")
    p.add_argument("--hf-token",   default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--output-dir", default="outputs/continual")
    p.add_argument("--no-rouge",   action="store_true")
    p.add_argument("--run-mia",    action="store_true")
    p.add_argument("--no-save",    action="store_true")
    p.add_argument("--num-cohorts",type=int, default=3, help="Split forget set into sequential cohorts")
    p.add_argument("--use-fim",    action="store_true", help="Enable FIM-based parameter masking")
    p.add_argument("--fim-topk",   type=float, default=0.30)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ARMORConfig(
        debug=args.debug,
        model_key=args.model if not args.debug else "debug",
        use_qlora=args.qlora,
        hf_token=args.hf_token,
        output_dir=args.output_dir,
    )
    if args.epochs: cfg.unlearn_epochs = args.epochs
    if args.lr:     cfg.unlearn_lr     = args.lr
    cfg.continual_use_fim_mask = args.use_fim
    cfg.continual_fim_topk = args.fim_topk

    print("=" * 60)
    print(f"  ARMOR — Continual/Lifelong Machine Unlearning")
    print(f"  Model       : {cfg.model_name}")
    print(f"  FIM Masking : {cfg.continual_use_fim_mask} (topk={cfg.continual_fim_topk:.0%})")
    print(f"  Cohorts     : {args.num_cohorts}")
    print("=" * 60)

    # ── Load Model and Data ───────────────────────────────────────────────────
    forget_samples, retain_samples = load_tofu_splits(cfg)
    model, tokenizer = get_model_and_tokenizer(cfg)

    # Split the forget set into sequential cohorts to simulate lifelong unlearning
    # (e.g. GDPR requests coming in over time)
    n_forget = len(forget_samples)
    cohort_size = max(1, n_forget // args.num_cohorts)
    cohorts = [
        forget_samples[i : i + cohort_size]
        for i in range(0, n_forget, cohort_size)
    ]
    # Restrict to num_cohorts in case of rounding
    cohorts = cohorts[:args.num_cohorts]

    print(f"[main] Loaded {n_forget} forget samples, split into {len(cohorts)} cohorts.")
    for idx, c in enumerate(cohorts):
        print(f"  Cohort #{idx+1}: {len(c)} samples")

    evaluator = UnlearningEvaluator(model, tokenizer, cfg)

    # ── Initialize Continual Unlearner ────────────────────────────────────────
    unlearner = ContinualUnlearner(model, cfg, tokenizer)

    # Single global loader for retain set (used for replay / masking)
    retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=True)
    eval_retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)

    # ── Sequential Unlearning Loop ────────────────────────────────────────────
    for idx, cohort in enumerate(cohorts):
        cohort_id = idx + 1
        print(f"\n[main] ────────────────── Processing Cohort #{cohort_id} ──────────────────")

        # Create dataloader for this cohort
        forget_loader = make_dataloader(
            cohort, tokenizer, cfg,
            include_rephrases=cfg.use_rephrase_augmentation,
            shuffle=True
        )
        eval_forget_loader = make_dataloader(cohort, tokenizer, cfg, shuffle=False)

        # ── Pre-cohort evaluation ─────────────────────────────────────────────
        print(f"[main] Pre-cohort #{cohort_id} evaluation:")
        pre_res = evaluator.evaluate(
            cohort, retain_samples, eval_forget_loader, eval_retain_loader,
            method_name=f"Pre-Cohort #{cohort_id}",
            run_rouge=not args.no_rouge,
            max_rouge_samples=10 if cfg.debug else 30
        )
        pre_res.print_table()

        # ── Unlearn this cohort ───────────────────────────────────────────────
        unlearner.unlearn(forget_loader, retain_loader, request_id=cohort_id)

        # ── Post-cohort evaluation ────────────────────────────────────────────
        print(f"[main] Post-cohort #{cohort_id} evaluation:")
        post_res = evaluator.evaluate(
            cohort, retain_samples, eval_forget_loader, eval_retain_loader,
            method_name=f"Post-Cohort #{cohort_id}",
            run_rouge=not args.no_rouge,
            max_rouge_samples=10 if cfg.debug else 30
        )
        post_res.print_table()

        if args.run_mia:
            print(f"[main] Auditing Cohort #{cohort_id} with MIA...")
            auditor = MembershipInferenceAuditor(model, tokenizer, cfg)
            auditor.audit(eval_forget_loader, eval_retain_loader, method_name=f"Continual-Cohort#{cohort_id}")

    # ── Save Final Model ──────────────────────────────────────────────────────
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_path = os.path.join(args.output_dir, "continual_unlearned")
        save_checkpoint(model, tokenizer, ckpt_path, cfg)
        print(f"\n[main] Done. Final checkpoint saved to: {ckpt_path}")
    else:
        print("\n[main] Done. (--no-save: checkpoint skipped)")


if __name__ == "__main__":
    main()
