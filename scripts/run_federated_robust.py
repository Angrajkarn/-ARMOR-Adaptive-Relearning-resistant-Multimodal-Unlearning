"""
scripts/run_federated_robust.py
===============================
BRFU: Byzantine-Robust Federated Unlearning Runner
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
from armor.unlearn.federated_robust import BRFUUnlearner
from armor.eval.metrics import UnlearningEvaluator

def parse_args():
    p = argparse.ArgumentParser(description="BRFU: Byzantine-Robust Federated Unlearning")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--model", default="debug")
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--num-clients", type=int, default=3)
    p.add_argument("--byzantine-frac", type=float, default=0.33)
    p.add_argument("--aggregation", default="krum", choices=["krum", "trimmed_mean", "mean"])
    p.add_argument("--hf-token", default=None)
    p.add_argument("--no-rouge", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--output-dir", default="outputs/federated_robust")
    return p.parse_args()

def main():
    args = parse_args()

    cfg = ARMORConfig(
        debug=args.debug,
        model_key="debug" if args.debug else args.model,
        use_qlora=args.qlora,
        hf_token=args.hf_token,
    )

    # Override defaults with CLI args if specified
    cfg.brfu_num_clients = args.num_clients
    cfg.brfu_byzantine_frac = args.byzantine_frac
    cfg.brfu_aggregation = args.aggregation

    print("\n" + "=" * 72)
    print("  BRFU: BYZANTINE-ROBUST FEDERATED UNLEARNING")
    print("=" * 72)
    print(f"  Model          : {cfg.model_name}")
    print(f"  Device         : {cfg.device}")
    print(f"  Total Clients  : {cfg.brfu_num_clients}")
    print(f"  Byzantine Frac : {cfg.brfu_byzantine_frac} ({int(cfg.brfu_num_clients * cfg.brfu_byzantine_frac)} Byzantine)")
    print(f"  Aggregation    : {cfg.brfu_aggregation}")
    print(f"  Debug          : {cfg.debug}")
    print()

    # Load model and reference model
    print("[BRFU] Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(cfg)
    ref_model = get_frozen_reference_model(model, cfg)

    # Load TOFU dataset
    print("[BRFU] Loading TOFU dataset...")
    forget_samples, retain_samples = load_tofu_splits(cfg)

    # Pre-evaluation
    print("\n[BRFU] === PRE-UNLEARN EVALUATION ===")
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

    # BRFU Training
    print("\n[BRFU] === STARTING FEDERATED UNLEARNING ===")
    unlearner = BRFUUnlearner(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        cfg=cfg,
    )

    result = unlearner.run(
        forget_dataset=forget_samples,
        retain_dataset=retain_samples,
        tokenizer=tokenizer,
    )

    # Post-evaluation
    print("\n[BRFU] === POST-UNLEARN EVALUATION ===")
    post_evaluator = UnlearningEvaluator(model, tokenizer, cfg)
    post_result = post_evaluator.evaluate(
        forget_samples=forget_samples,
        retain_samples=retain_samples,
        forget_loader=eval_fl,
        retain_loader=eval_rl,
        method_name="BRFU",
        run_rouge=not args.no_rouge,
    )
    post_result.print_table()

    print("\n[BRFU] Done.")

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
