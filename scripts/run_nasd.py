"""
scripts/run_nasd.py
===================
Run Neuro-Apoptotic Subnetwork Decay (NASD) on TOFU.

Usage
-----
  python scripts/run_nasd.py --debug
  python scripts/run_nasd.py --model mistral-7b --qlora
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
from armor.model  import get_model_and_tokenizer
from armor.unlearn.nasd import NeuroApoptoticDecay
from armor.eval.metrics  import UnlearningEvaluator

def parse_args():
    p = argparse.ArgumentParser(description="ARMOR Vanguard — NASD Unlearning")
    p.add_argument("--debug",      action="store_true")
    p.add_argument("--model",      default="debug", choices=["debug","mistral-7b","llama2-7b"])
    p.add_argument("--qlora",      action="store_true")
    p.add_argument("--hf-token",   default=None)
    p.add_argument("--output-dir", default="outputs/nasd")
    p.add_argument("--no-rouge",   action="store_true")
    p.add_argument("--decay-steps", type=int, default=50)
    p.add_argument("--decay-rate",  type=float, default=0.90)
    p.add_argument("--topk",        type=float, default=0.05)
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
    
    cfg.nasd_decay_steps = args.decay_steps
    cfg.nasd_decay_rate = 0.99
    cfg.nasd_topk_fraction = 0.001
    cfg.nasd_lambda_retain = 100.0

    print("=" * 60)
    print(f"  ARMOR Vanguard — Neuro-Apoptotic Subnetwork Decay (NASD)")
    print(f"  Model       : {cfg.model_name}")
    print(f"  Decay Steps : {cfg.nasd_decay_steps}")
    print(f"  Decay Rate  : {cfg.nasd_decay_rate}")
    print(f"  Target Top-K: {cfg.nasd_topk_fraction:.2%}")
    print("=" * 60)

    # ── Load Model and Data ───────────────────────────────────────────────────
    forget_samples, retain_samples = load_tofu_splits(cfg)
    model, tokenizer = get_model_and_tokenizer(cfg)

    eval_forget_loader = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    eval_retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)

    evaluator = UnlearningEvaluator(model, tokenizer, cfg)

    # ── Pre-unlearning Evaluation ─────────────────────────────────────────────
    print("\n[main] Pre-infection evaluation (Base Model):")
    pre_result = evaluator.evaluate(
        forget_samples, retain_samples,
        eval_forget_loader, eval_retain_loader,
        method_name="Base Model",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50,
    )
    pre_result.print_table()

    # ── Initialize NASD Unlearner ─────────────────────────────────────────────
    # We use non-shuffled loaders for Fisher to be consistent
    nasd = NeuroApoptoticDecay(model, cfg)
    nasd.infect(eval_forget_loader, eval_retain_loader)

    # ── Live Inference Decay Simulation ───────────────────────────────────────
    # Since the evaluation ITSELF uses the forward pass, the model will naturally
    # decay as we evaluate it. We'll run evaluation multiple times.
    
    eval_rounds = 3
    for rnd in range(eval_rounds):
        # We peek at the hook step from the first hook (if any exist)
        current_step = nasd.hooks[0].hook.step if hasattr(nasd.hooks[0], 'hook') else "?" 
        print(f"\n[main] Live Inference Decay Round {rnd+1}/{eval_rounds} | Hook Step ≈ {current_step}")
        
        post_result = evaluator.evaluate(
            forget_samples, retain_samples,
            eval_forget_loader, eval_retain_loader,
            method_name=f"NASD Decay Phase {rnd+1}",
            run_rouge=not args.no_rouge,
            max_rouge_samples=20 if cfg.debug else 50,
        )
        post_result.print_table()
        
    print("\n[main] Infection lifecycle complete.")

if __name__ == "__main__":
    main()
