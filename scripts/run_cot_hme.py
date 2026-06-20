"""
scripts/run_cot_hme.py
=======================
CoT-HME: Chain-of-Thought Hidden Memory Erasure Runner
=======================================================

Runs the full CoT-HME pipeline:
  1. Probe the forget set for CoT leakage (pre-training)
  2. Train CoT-HME (NPO + CoT entropy suppression loss)
  3. Re-probe the forget set for CoT leakage (post-training)
  4. Evaluate forget/retain accuracy
  5. Save probe reports

Usage
-----
  # Debug mode (CPU, distilgpt2)
  python scripts/run_cot_hme.py --debug

  # With custom CoT loss weight
  python scripts/run_cot_hme.py --debug --cot-coeff 0.5

  # Full GPU run
  python scripts/run_cot_hme.py --model mistral-7b --qlora --cot-coeff 0.3
"""

import argparse
import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armor.config import ARMORConfig
from armor.data   import load_tofu_splits, make_dataloader
from armor.model  import get_model_and_tokenizer, get_frozen_reference_model
from armor.unlearn.cot_hme import CoTHMEUnlearner
from armor.attack.cot_leakage_probe import CoTLeakageProbe
from armor.eval.metrics import UnlearningEvaluator


# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CoT-HME: Chain-of-Thought Hidden Memory Erasure"
    )
    p.add_argument("--debug",          action="store_true")
    p.add_argument("--model",          default="debug")
    p.add_argument("--qlora",          action="store_true")
    p.add_argument("--cot-coeff",      type=float, default=0.3,
                   help="Weight of CoT entropy loss (default: 0.3)")
    p.add_argument("--cot-threshold",  type=float, default=0.3,
                   help="Leakage threshold for step classification")
    p.add_argument("--cot-max-tokens", type=int,   default=64,
                   help="Max new tokens for CoT trace generation")
    p.add_argument("--no-rouge",       action="store_true")
    p.add_argument("--no-save",        action="store_true")
    p.add_argument("--output-dir",     default="outputs/cot_hme")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    cfg = ARMORConfig(
        debug=args.debug,
        model_key="debug" if args.debug else args.model,
        use_qlora=args.qlora,
    )

    print("\n" + "=" * 72)
    print("  CoT-HME: CHAIN-OF-THOUGHT HIDDEN MEMORY ERASURE")
    print("=" * 72)
    print(f"  Model          : {cfg.model_name}")
    print(f"  Device         : {cfg.device}")
    print(f"  CoT loss coeff : {args.cot_coeff}")
    print(f"  Debug          : {cfg.debug}")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    print("[CoT-HME] Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(cfg)
    ref_model        = get_frozen_reference_model(model, cfg)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[CoT-HME] Loading TOFU dataset...")
    forget_samples, retain_samples = load_tofu_splits(cfg)

    forget_loader = make_dataloader(
        forget_samples, tokenizer, cfg,
        include_rephrases=True, shuffle=True,
    )
    retain_loader = make_dataloader(
        retain_samples, tokenizer, cfg,
        shuffle=True,
    )

    # Extract (Q, A) pairs for CoT probing — small subset for speed
    n_probe = min(4 if cfg.debug else 20, len(forget_samples))
    qa_pairs = [(s.question, s.answer) for s in forget_samples[:n_probe]]
    print(f"[CoT-HME] Using {len(qa_pairs)} (Q,A) pairs for CoT probing")

    # ── Pre-training CoT probe ────────────────────────────────────────────────
    print("\n[CoT-HME] === PRE-TRAINING CoT PROBE ===")
    cot_max_tokens = args.cot_max_tokens if not cfg.debug else 24
    probe = CoTLeakageProbe(
        model=model, tokenizer=tokenizer, cfg=cfg,
        leakage_threshold=args.cot_threshold,
        max_new_tokens=cot_max_tokens,
        few_shot=False,
    )

    pre_report = probe.probe_dataset(
        qa_pairs=qa_pairs,
        method_name=f"PRE-{cfg.model_key}",
    )

    # ── CoT-HME Training ──────────────────────────────────────────────────────
    print("\n[CoT-HME] === STARTING CoT-HME TRAINING ===")
    unlearner = CoTHMEUnlearner(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        cfg=cfg,
        qa_forget_pairs=qa_pairs,
        cot_loss_coeff=args.cot_coeff,
        cot_leak_threshold=args.cot_threshold,
        cot_max_new_tokens=cot_max_tokens,
        cot_probe_batch=n_probe,
    )

    result = unlearner.run(forget_loader, retain_loader)

    # ── Post-training CoT probe ───────────────────────────────────────────────
    print("\n[CoT-HME] === POST-TRAINING CoT PROBE ===")
    post_report = probe.probe_dataset(
        qa_pairs=qa_pairs,
        method_name=f"POST-{cfg.model_key}",
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n[CoT-HME] === EVALUATION ===")
    eval_forget_loader = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    eval_retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)

    evaluator = UnlearningEvaluator(model, tokenizer, cfg)
    eval_result = evaluator.evaluate(
        forget_samples=forget_samples,
        retain_samples=retain_samples,
        forget_loader=eval_forget_loader,
        retain_loader=eval_retain_loader,
        method_name="CoT-HME",
        run_rouge=not args.no_rouge,
    )
    eval_result.print_table()

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\n[CoT-HME] === CoT LEAKAGE COMPARISON ===")
    print(f"  Pre-training  leakage rate : {pre_report.trace_leakage_rate:.4f}")
    print(f"  Post-training leakage rate : {post_report.trace_leakage_rate:.4f}")
    improvement = pre_report.trace_leakage_rate - post_report.trace_leakage_rate
    print(f"  Improvement               : {improvement:+.4f}")
    if post_report.cot_erased:
        print("  CoT hidden memory ERASED")
    else:
        print("  CoT leakage still detected — consider higher cot_coeff")

    # ── Save outputs ──────────────────────────────────────────────────────────
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        ts = int(time.time())
        pre_path  = os.path.join(args.output_dir, f"pre_probe_{ts}.json")
        post_path = os.path.join(args.output_dir, f"post_probe_{ts}.json")
        probe.save_report(pre_report,  pre_path,  save_html=True)
        probe.save_report(post_report, post_path, save_html=True)
        print(f"\n[CoT-HME] Reports saved to {args.output_dir}")

    print("\n[CoT-HME] Done.")
    sys.exit(0)


if __name__ == "__main__":
    main()
