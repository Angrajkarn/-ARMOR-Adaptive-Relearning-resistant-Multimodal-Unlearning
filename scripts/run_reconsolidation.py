"""
scripts/run_reconsolidation.py
==============================
NRU: Neural Reconsolidation Unlearning Runner
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
from armor.data import load_tofu_splits, make_dataloader
from armor.model import get_model_and_tokenizer, get_frozen_reference_model
from armor.unlearn.neural_reconsolidation import NRUUnlearner
from armor.eval.metrics import UnlearningEvaluator

def parse_args():
    p = argparse.ArgumentParser(description="NRU: Neural Reconsolidation Unlearning")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--model", default="debug")
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--recall-lr", type=float, default=5e-5)
    p.add_argument("--no-rouge", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--output-dir", default="outputs/reconsolidation")
    return p.parse_args()

def main():
    args = parse_args()

    cfg = ARMORConfig(
        debug=args.debug,
        model_key="debug" if args.debug else args.model,
        use_qlora=args.qlora,
    )

    # Override defaults with CLI args if specified
    cfg.nru_recall_lr = args.recall_lr

    print("\n" + "=" * 72)
    print("  NRU: NEURAL RECONSOLIDATION UNLEARNING")
    print("=" * 72)
    print(f"  Model          : {cfg.model_name}")
    print(f"  Device         : {cfg.device}")
    print(f"  Recall LR      : {cfg.nru_recall_lr}")
    print(f"  SAM radius     : {cfg.sam_rho}")
    print(f"  Debug          : {cfg.debug}")
    print()

    # Load model and reference model
    print("[NRU] Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(cfg)
    ref_model = get_frozen_reference_model(model, cfg)

    # Load TOFU dataset
    print("[NRU] Loading TOFU dataset...")
    forget_samples, retain_samples = load_tofu_splits(cfg)

    forget_loader = make_dataloader(
        forget_samples, tokenizer, cfg,
        include_rephrases=True, shuffle=True,
    )
    retain_loader = make_dataloader(
        retain_samples, tokenizer, cfg,
        shuffle=True,
    )

    # Pre-evaluation
    print("\n[NRU] === PRE-UNLEARN EVALUATION ===")
    eval_fl = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    eval_rl = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)
    evaluator = UnlearningEvaluator(model, tokenizer, cfg)
    pre_result = evaluator.evaluate(
        forget_samples=forget_samples,
        retain_samples=retain_samples,
        forget_loader=eval_fl,
        retain_loader=eval_rl,
        method_name="Pre-unlearn",
        run_rouge=not args.no_rouge,
    )
    pre_result.print_table()

    # NRU Training
    print("\n[NRU] === STARTING NRU UNLEARNING ===")
    unlearner = NRUUnlearner(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        cfg=cfg,
    )

    result = unlearner.run(
        forget_loader=forget_loader,
        retain_loader=retain_loader,
    )

    # Post-evaluation
    print("\n[NRU] === POST-UNLEARN EVALUATION ===")
    post_evaluator = UnlearningEvaluator(model, tokenizer, cfg)
    post_result = post_evaluator.evaluate(
        forget_samples=forget_samples,
        retain_samples=retain_samples,
        forget_loader=eval_fl,
        retain_loader=eval_rl,
        method_name="NRU",
        run_rouge=not args.no_rouge,
    )
    post_result.print_table()

    print("\n[NRU] Done.")

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

    sys.exit(0)

if __name__ == "__main__":
    main()
