"""
scripts/run_lora_unlearn.py
==========================
Run Modular LoRA Unlearning (Negative Adapter weight arithmetic) on TOFU.

Usage
-----
  python scripts/run_lora_unlearn.py --debug
  python scripts/run_lora_unlearn.py --model mistral-7b --qlora
"""

import argparse
import os
import sys
import warnings

# Make armor importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore", message=".*symlink.*")

import torch
from armor.config import ARMORConfig
from armor.data   import load_tofu_splits, make_dataloader
from armor.model  import get_model_and_tokenizer, save_checkpoint
from armor.unlearn.lora_unlearner import LoRAUnlearner
from armor.eval.metrics  import UnlearningEvaluator
from armor.eval.mia      import MembershipInferenceAuditor


def parse_args():
    p = argparse.ArgumentParser(description="ARMOR — Modular LoRA Unlearning")
    p.add_argument("--debug",      action="store_true")
    p.add_argument("--model",      default="debug", choices=["debug","mistral-7b","llama2-7b"])
    p.add_argument("--qlora",      action="store_true")
    p.add_argument("--hf-token",   default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--output-dir", default="outputs/lora")
    p.add_argument("--no-rouge",   action="store_true")
    p.add_argument("--run-mia",    action="store_true")
    p.add_argument("--no-save",    action="store_true")
    p.add_argument("--unlearn-r",  type=int, default=16)
    p.add_argument("--unlearn-alpha", type=int, default=32)
    p.add_argument("--unlearn-scale", type=float, default=1.0)
    p.add_argument("--retain-r",   type=int, default=8, help="Rank of retain LoRA (0 to disable)")
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
    cfg.lora_unlearn_r = args.unlearn_r
    cfg.lora_unlearn_alpha = args.unlearn_alpha
    cfg.lora_unlearn_scale = args.unlearn_scale
    cfg.lora_retain_r = args.retain_r

    print("=" * 60)
    print(f"  ARMOR — Modular LoRA Unlearning")
    print(f"  Model        : {cfg.model_name}")
    print(f"  Unlearn Rank : {cfg.lora_unlearn_r} (alpha={cfg.lora_unlearn_alpha})")
    print(f"  Unlearn Scale: {cfg.lora_unlearn_scale}")
    print(f"  Retain Rank  : {cfg.lora_retain_r}")
    print("=" * 60)

    # ── Load Model and Data ───────────────────────────────────────────────────
    forget_samples, retain_samples = load_tofu_splits(cfg)
    model, tokenizer = get_model_and_tokenizer(cfg)

    eval_forget_loader = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    eval_retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)

    evaluator = UnlearningEvaluator(model, tokenizer, cfg)

    # ── Pre-unlearning Evaluation ─────────────────────────────────────────────
    print("\n[main] Pre-unlearning evaluation:")
    pre_result = evaluator.evaluate(
        forget_samples, retain_samples,
        eval_forget_loader, eval_retain_loader,
        method_name="Pre-unlearning (LoRA)",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50,
    )
    pre_result.print_table()

    # ── Initialize LoRA Unlearner ─────────────────────────────────────────────
    forget_loader = make_dataloader(forget_samples, tokenizer, cfg, include_rephrases=False, shuffle=True)
    retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=True)

    unlearner = LoRAUnlearner(model, cfg)
    unlearner.unlearn(forget_loader, retain_loader)

    # ── Post-unlearning Evaluation ────────────────────────────────────────────
    print("\n[main] Post-unlearning evaluation:")
    post_result = evaluator.evaluate(
        forget_samples, retain_samples,
        eval_forget_loader, eval_retain_loader,
        method_name="Negative LoRA Arithmetic",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50,
    )
    post_result.print_table()

    if args.run_mia:
        print("\n[main] Running MIA audit...")
        auditor = MembershipInferenceAuditor(model, tokenizer, cfg)
        mia_result = auditor.audit(eval_forget_loader, eval_retain_loader, method_name="Negative-LoRA")
        post_result.mia_auroc = mia_result.auroc
        post_result.print_table()

    # ── Save Checkpoint ───────────────────────────────────────────────────────
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_path = os.path.join(args.output_dir, "lora_unlearned")
        save_checkpoint(model, tokenizer, ckpt_path, cfg)
        print(f"\n[main] Done. Checkpoint saved to: {ckpt_path}")
    else:
        print("\n[main] Done. (--no-save: checkpoint skipped)")


if __name__ == "__main__":
    main()
