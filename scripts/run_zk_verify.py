"""
scripts/run_zk_verify.py
========================
Run ZK-style Commit-Reveal Influence auditing on TOFU unlearning.

Usage
-----
  python scripts/run_zk_verify.py --debug
  python scripts/run_zk_verify.py --model mistral-7b --qlora
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
from armor.model  import get_model_and_tokenizer
from armor.unlearn.gradient_ascent import GradientAscentUnlearner
from armor.eval.zk_verify import ZKVerifier


def parse_args():
    p = argparse.ArgumentParser(description="ARMOR — ZK Unlearning Verification")
    p.add_argument("--debug",      action="store_true")
    p.add_argument("--model",      default="debug", choices=["debug","mistral-7b","llama2-7b"])
    p.add_argument("--qlora",      action="store_true")
    p.add_argument("--hf-token",   default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--output-dir", default="outputs/zk")
    p.add_argument("--threshold",  type=float, default=0.01)
    p.add_argument("--damping",    type=float, default=5e-3)
    p.add_argument("--probe-samples", type=int, default=16)
    p.add_argument("--no-save",    action="store_true", help="Skip saving checkpoint")
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
    cfg.zk_influence_damping = args.damping
    cfg.zk_n_probe_samples = args.probe_samples
    cfg.zk_influence_threshold = args.threshold

    print("=" * 60)
    print(f"  ARMOR — ZK Unlearning Verification (Commit-Reveal)")
    print(f"  Model       : {cfg.model_name}")
    print(f"  Threshold   : {cfg.zk_influence_threshold}")
    print(f"  Damping (λ) : {cfg.zk_influence_damping}")
    print(f"  Probe Size  : {cfg.zk_n_probe_samples}")
    print("=" * 60)

    # ── Load Model and Data ───────────────────────────────────────────────────
    forget_samples, retain_samples = load_tofu_splits(cfg)
    model, tokenizer = get_model_and_tokenizer(cfg)

    eval_forget_loader = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    eval_retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)

    # ── Initialize ZK Verifier ────────────────────────────────────────────────
    verifier = ZKVerifier(cfg)

    # Phase 1: Pre-unlearning commitment & influence estimation
    verifier.commit_pre(model, eval_forget_loader, eval_retain_loader, method="GradientAscent")

    # ── Run Base Unlearning (Gradient Ascent) ─────────────────────────────────
    print("\n[main] Running base unlearning (Gradient Ascent)...")
    forget_loader = make_dataloader(
        forget_samples, tokenizer, cfg,
        include_rephrases=cfg.use_rephrase_augmentation,
        shuffle=True
    )
    retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=True)

    unlearner = GradientAscentUnlearner(model, cfg)
    unlearner.run(forget_loader, retain_loader)

    # ── Phase 2: Post-unlearning Verification & Report ────────────────────────
    report = verifier.verify_post(model, eval_forget_loader, eval_retain_loader)

    # Save verification report
    report_path = os.path.join(args.output_dir, "zk_audit_report.json")
    verifier.save_report(report, report_path)


if __name__ == "__main__":
    main()
