"""
scripts/run_baseline_ga.py
===========================
Run Gradient Ascent (GA) unlearning baseline on TOFU.

Usage
-----
  # Quick CPU test (~2 min):
  python scripts/run_baseline_ga.py --debug

  # Full run on GPU with Mistral-7B:
  python scripts/run_baseline_ga.py --model mistral-7b --qlora

  # Full run with LLaMA-2-7B (requires HF token):
  python scripts/run_baseline_ga.py --model llama2-7b --hf-token YOUR_TOKEN
"""

import argparse
import os
import sys

# Make armor importable from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from armor.config import ARMORConfig
from armor.data   import load_tofu_splits, make_dataloader
from armor.model  import get_model_and_tokenizer, save_checkpoint
from armor.unlearn.gradient_ascent import GradientAscentUnlearner
from armor.eval.metrics  import UnlearningEvaluator
from armor.eval.mia      import MembershipInferenceAuditor


def parse_args():
    p = argparse.ArgumentParser(description="ARMOR — Gradient Ascent Baseline")
    p.add_argument("--debug",      action="store_true",
                   help="Use tiny model + tiny data for fast CPU testing")
    p.add_argument("--model",      default="debug", choices=["debug","mistral-7b","llama2-7b"],
                   help="Model to use (default: debug = opt-125m)")
    p.add_argument("--qlora",      action="store_true", help="Use 4-bit QLoRA (GPU)")
    p.add_argument("--hf-token",   default=None, help="HuggingFace token for gated models")
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--output-dir", default="outputs/ga")
    p.add_argument("--no-rouge",   action="store_true",
                   help="Skip slow generative ROUGE evaluation")
    p.add_argument("--run-mia",    action="store_true",
                   help="Run Membership Inference Attack audit")
    p.add_argument("--no-save",    action="store_true",
                   help="Skip saving checkpoint (for smoke tests)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Config ──────────────────────────────────────────────────────────────────
    cfg = ARMORConfig(
        debug=args.debug,
        model_key=args.model if not args.debug else "debug",
        use_qlora=args.qlora,
        hf_token=args.hf_token,
        output_dir=args.output_dir,
    )
    if args.epochs: cfg.unlearn_epochs = args.epochs
    if args.lr:     cfg.unlearn_lr     = args.lr

    print("=" * 60)
    print(f"  ARMOR — Gradient Ascent Baseline")
    print(f"  Model  : {cfg.model_name}")
    print(f"  Device : {cfg.device}")
    print(f"  Debug  : {cfg.debug}")
    print("=" * 60)

    # ── Data ────────────────────────────────────────────────────────────────────
    forget_samples, retain_samples = load_tofu_splits(cfg)

    # ── Model ───────────────────────────────────────────────────────────────────
    model, tokenizer = get_model_and_tokenizer(cfg)

    # ── Pre-unlearning evaluation ────────────────────────────────────────────────
    eval_forget_loader = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    eval_retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)

    evaluator = UnlearningEvaluator(model, tokenizer, cfg)
    print("\n[main] Pre-unlearning evaluation:")
    pre_result = evaluator.evaluate(
        forget_samples, retain_samples,
        eval_forget_loader, eval_retain_loader,
        method_name="Pre-unlearning (no GA)",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50,
    )
    pre_result.print_table()
    original_forget_acc = pre_result.forget_accuracy

    # ── Unlearning data loaders ───────────────────────────────────────────────────
    # include_rephrases=True for rephrase-invariant gradient ascent
    forget_loader = make_dataloader(
        forget_samples, tokenizer, cfg,
        include_rephrases=cfg.use_rephrase_augmentation,
        shuffle=True,
    )
    retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=True)

    # ── Gradient Ascent Unlearning ────────────────────────────────────────────────
    print("\n[main] Starting Gradient Ascent unlearning...")
    unlearner = GradientAscentUnlearner(model, cfg)
    train_result = unlearner.run(forget_loader, retain_loader)

    # ── Post-unlearning evaluation ────────────────────────────────────────────────
    print("\n[main] Post-unlearning evaluation:")
    post_result = evaluator.evaluate(
        forget_samples, retain_samples,
        eval_forget_loader, eval_retain_loader,
        method_name="Gradient Ascent",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50,
    )
    post_result.print_table()

    # ── Optional MIA audit ────────────────────────────────────────────────────────
    if args.run_mia:
        print("\n[main] Running MIA audit...")
        auditor = MembershipInferenceAuditor(model, tokenizer, cfg)
        mia_result = auditor.audit(
            eval_forget_loader, eval_retain_loader,
            method_name="Gradient Ascent"
        )
        post_result.mia_auroc = mia_result.auroc
        post_result.print_table()

    # ── Save checkpoint ───────────────────────────────────────────────────────────
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_path = os.path.join(args.output_dir, "ga_unlearned")
        save_checkpoint(model, tokenizer, ckpt_path, cfg)
        print("\n[main] Done. Checkpoint saved to:", ckpt_path)
    else:
        print("\n[main] Done. (--no-save: checkpoint skipped)")

    print(f"[main] Original forget accuracy : {original_forget_acc:.4f}")
    print(f"[main] Post-GA forget accuracy  : {post_result.forget_accuracy:.4f}")
    print(f"[main] Post-GA forget quality   : {post_result.forget_quality:.4f}")
    print(f"[main] Post-GA retain accuracy  : {post_result.retain_accuracy:.4f}")


if __name__ == "__main__":
    main()
