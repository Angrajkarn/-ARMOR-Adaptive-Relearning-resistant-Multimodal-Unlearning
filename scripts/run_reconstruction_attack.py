"""
scripts/run_reconstruction_attack.py
====================================
Adversarial Auditing: Benchmark Text Reconstruction & Model Inversion.

Usage:
  python scripts/run_reconstruction_attack.py --debug
"""

import argparse
import os
import sys
import warnings
import numpy as np

# Make armor importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore", message=".*symlink.*")

import torch
from armor.config import ARMORConfig
from armor.data   import load_tofu_splits, make_dataloader
from armor.model  import get_model_and_tokenizer
from armor.attack.reconstruction import TextReconstructionAttack
from armor.unlearn.gradient_ascent import GradientAscentUnlearner


def parse_args():
    p = argparse.ArgumentParser(description="ARMOR — Text Reconstruction Attack Audit")
    p.add_argument("--debug",      action="store_true")
    p.add_argument("--model",      default="debug", choices=["debug", "mistral-7b", "llama2-7b"])
    p.add_argument("--qlora",      action="store_true")
    p.add_argument("--hf-token",   default=None)
    p.add_argument("--output-dir", default="outputs/attack")
    p.add_argument("--threshold",  type=float, default=0.5,
                   help="ROUGE-L threshold to count as successful leakage")
    p.add_argument("--no-save",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    
    # Set up stdout/stderr encoding wrapper for Windows to prevent charmap errors
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    cfg = ARMORConfig(
        debug=args.debug,
        model_key=args.model if not args.debug else "debug",
        use_qlora=args.qlora,
        hf_token=args.hf_token,
        output_dir=args.output_dir,
    )

    print("=" * 64)
    print("  ARMOR — Text Reconstruction Attack (Model Inversion)")
    print(f"  Model       : {cfg.model_name}")
    print(f"  Threshold   : {args.threshold}")
    print("=" * 64)

    # ── Load Model and Data ───────────────────────────────────────────────────
    forget_samples, retain_samples = load_tofu_splits(cfg)
    model, tokenizer = get_model_and_tokenizer(cfg)

    # Instantiate the attack engine
    attacker = TextReconstructionAttack(cfg, model, tokenizer)

    # ── Phase 1: Attack the Base Model (Pre-Unlearn) ──────────────────────────
    print("\n[Attack Phase 1] Attacking the base model (before unlearning)...")
    pre_result = attacker.run_reconstruction_attack(
        forget_samples=forget_samples,
        method_name="Pre-unlearn",
        threshold=args.threshold
    )
    pre_result.print_summary()

    # ── Run Unlearning (Gradient Ascent) ──────────────────────────────────────
    print("\n[Unlearn Phase] Running Gradient Ascent unlearning...")
    forget_loader = make_dataloader(
        forget_samples, tokenizer, cfg,
        include_rephrases=cfg.use_rephrase_augmentation,
        shuffle=True
    )
    retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=True)
    unlearner = GradientAscentUnlearner(model, cfg)
    unlearner.run(forget_loader, retain_loader)

    # ── Phase 2: Attack the Unlearned Model (Post-Unlearn) ────────────────────
    print("\n[Attack Phase 2] Attacking the model after unlearning...")
    post_result = attacker.run_reconstruction_attack(
        forget_samples=forget_samples,
        method_name="GradientAscent (Post-Unlearn)",
        threshold=args.threshold
    )
    post_result.print_summary()

    # ── Comparison Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  ATTACK BENCHMARK COMPARISON")
    print("=" * 66)
    print(f"  Metric                     | {'Pre-Unlearn':<15} | {'Post-Unlearn':<15}")
    print("  " + "-" * 62)
    print(f"  Greedy ROUGE-L             | {pre_result.avg_greedy_rougeL:>15.4f} | {post_result.avg_greedy_rougeL:>15.4f}")
    print(f"  Beam Search ROUGE-L        | {pre_result.avg_beam_rougeL:>15.4f} | {post_result.avg_beam_rougeL:>15.4f}")
    print(f"  Prefix Tree Search ROUGE-L | {pre_result.avg_tree_rougeL:>15.4f} | {post_result.avg_tree_rougeL:>15.4f}")
    print(f"  Avg Suffix Log-Prob        | {pre_result.avg_target_logprob:>15.4f} | {post_result.avg_target_logprob:>15.4f}")
    print(f"  Leakage Rate               | {pre_result.leakage_rate:>15.2%} | {post_result.leakage_rate:>15.2%}")
    print("=" * 66 + "\n")

    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save a JSON file detailing the benchmark
    summary_path = os.path.join(args.output_dir, "reconstruction_attack_summary.json")
    import json
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "pre_unlearn": {
                "greedy_rouge_l": pre_result.avg_greedy_rougeL,
                "beam_rouge_l": pre_result.avg_beam_rougeL,
                "tree_rouge_l": pre_result.avg_tree_rougeL,
                "avg_logprob": pre_result.avg_target_logprob,
                "leakage_rate": pre_result.leakage_rate
            },
            "post_unlearn": {
                "greedy_rouge_l": post_result.avg_greedy_rougeL,
                "beam_rouge_l": post_result.avg_beam_rougeL,
                "tree_rouge_l": post_result.avg_tree_rougeL,
                "avg_logprob": post_result.avg_target_logprob,
                "leakage_rate": post_result.leakage_rate
            }
        }, f, indent=2)
    print(f"[Done] Attack summary report saved → {summary_path}")


if __name__ == "__main__":
    main()
