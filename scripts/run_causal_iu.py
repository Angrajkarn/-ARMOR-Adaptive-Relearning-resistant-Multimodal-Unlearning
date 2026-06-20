"""
scripts/run_causal_iu.py
========================
CIU: Causal Interventional Unlearning Runner
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
from armor.unlearn.causal_iu import CausalUnlearner
from armor.eval.metrics import UnlearningEvaluator

def parse_args():
    p = argparse.ArgumentParser(description="CIU: Causal Interventional Unlearning via Do-Calculus")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--model", default="debug")
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--num-nodes", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--no-rouge", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--output-dir", default="outputs/causal_iu")
    return p.parse_args()

def main():
    args = parse_args()

    cfg = ARMORConfig(
        debug=args.debug,
        model_key="debug" if args.debug else args.model,
        use_qlora=args.qlora,
    )

    # Override defaults with CLI args if specified
    cfg.ciu_num_nodes = args.num_nodes
    cfg.ciu_threshold = args.threshold

    print("\n" + "=" * 72)
    print("  CIU: CAUSAL INTERVENTIONAL UNLEARNING VIA DO-CALCULUS")
    print("=" * 72)
    print(f"  Model          : {cfg.model_name}")
    print(f"  Device         : {cfg.device}")
    print(f"  Num layers     : {cfg.ciu_num_nodes}")
    print(f"  ACE threshold  : {cfg.ciu_threshold}")
    print(f"  Debug          : {cfg.debug}")
    print()

    # Load model and reference model
    print("[CIU] Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(cfg)
    ref_model = get_frozen_reference_model(model, cfg)

    # Load TOFU dataset
    print("[CIU] Loading TOFU dataset...")
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
    print("\n[CIU] === PRE-UNLEARN EVALUATION ===")
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

    # CIU Training
    print("\n[CIU] === STARTING CAUSAL UNLEARNING ===")
    unlearner = CausalUnlearner(
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
    print("\n[CIU] === POST-UNLEARN EVALUATION ===")
    post_evaluator = UnlearningEvaluator(model, tokenizer, cfg)
    post_result = post_evaluator.evaluate(
        forget_samples=forget_samples,
        retain_samples=retain_samples,
        forget_loader=eval_fl,
        retain_loader=eval_rl,
        method_name="CIU",
        run_rouge=not args.no_rouge,
    )
    post_result.print_table()

    print("\n[CIU] Done.")
    sys.exit(0)

if __name__ == "__main__":
    main()
