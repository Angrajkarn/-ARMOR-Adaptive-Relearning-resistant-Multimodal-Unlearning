"""
scripts/run_cas_unlearn.py
==========================
Run Causal Attention Severing (CAS) Unlearning on TOFU.

Usage
-----
  python scripts/run_cas_unlearn.py --debug
"""

import argparse
import os
import sys
import warnings

# Make armor importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore", message=".*symlink.*")

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

import torch
from armor.config import ARMORConfig
from armor.data   import load_tofu_splits, make_dataloader
from armor.model  import get_model_and_tokenizer, save_checkpoint
from armor.unlearn.cas import CausalAttentionSevering
from armor.eval.metrics  import UnlearningEvaluator

def parse_args():
    p = argparse.ArgumentParser(description="ARMOR Vanguard — CAS Unlearning")
    p.add_argument("--debug",      action="store_true")
    p.add_argument("--model",      default="debug", choices=["debug","mistral-7b","llama2-7b"])
    p.add_argument("--qlora",      action="store_true")
    p.add_argument("--hf-token",   default=None)
    p.add_argument("--output-dir", default="outputs/cas")
    p.add_argument("--no-rouge",   action="store_true")
    p.add_argument("--no-save",    action="store_true")
    p.add_argument("--gamma",      type=float, default=1.5)
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
    
    cfg.cas_gamma = args.gamma
    
    print("=" * 60)
    print(f"  ARMOR Vanguard — Causal Attention Severing (CAS)")
    print(f"  Model       : {cfg.model_name}")
    print(f"  Gamma       : {cfg.cas_gamma}")
    print("=" * 60)

    # ── Load Model and Data ───────────────────────────────────────────────────
    forget_samples, retain_samples = load_tofu_splits(cfg)
    model, tokenizer = get_model_and_tokenizer(cfg)

    eval_forget_loader = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    eval_retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)

    evaluator = UnlearningEvaluator(model, tokenizer, cfg)

    # ── Pre-unlearning Evaluation ─────────────────────────────────────────────
    print("\n[main] Pre-unlearning evaluation (Base Model):")
    pre_result = evaluator.evaluate(
        forget_samples, retain_samples,
        eval_forget_loader, eval_retain_loader,
        method_name="Base Model",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50,
    )
    pre_result.print_table()

    # ── Zero-Shot Unlearning via CAS ──────────────────────────────────────────
    cas = CausalAttentionSevering(model, cfg)
    cas.unlearn(eval_forget_loader, eval_retain_loader)

    # ── Post-unlearning Evaluation ────────────────────────────────────────────
    print("\n[main] Post-unlearning evaluation:")
    post_result = evaluator.evaluate(
        forget_samples, retain_samples,
        eval_forget_loader, eval_retain_loader,
        method_name="CAS Zero-Shot",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50,
    )
    post_result.print_table()
    
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_path = os.path.join(args.output_dir, "cas_unlearned")
        save_checkpoint(model, tokenizer, ckpt_path, cfg)
        print(f"\n[main] Done. Checkpoint saved to: {ckpt_path}")
    else:
        print("\n[main] Done. (--no-save: checkpoint skipped)")

    # Save evaluation results to JSON
    import json
    res_dict = {
        "forget_quality": post_result.forget_quality,
        "forget_accuracy": post_result.forget_accuracy,
        "retain_accuracy": post_result.retain_accuracy,
        "mia_auroc": getattr(post_result, "mia_auroc", -1.0)
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(res_dict, f, indent=2)


if __name__ == "__main__":
    main()