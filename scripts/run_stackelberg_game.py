"""
scripts/run_stackelberg_game.py
===============================
SAUG: Stackelberg Adversarial Unlearning Game Runner
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
from armor.unlearn.stackelberg_game import SAUGUnlearner
from armor.eval.metrics import UnlearningEvaluator

def parse_args():
    p = argparse.ArgumentParser(description="SAUG: Stackelberg Adversarial Unlearning Game")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--model", default="debug")
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--adv-steps", type=int, default=2)
    p.add_argument("--adv-lr", type=float, default=5e-5)
    p.add_argument("--saug-coeff", type=float, default=0.5)
    p.add_argument("--no-rouge", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--output-dir", default="outputs/saug")
    return p.parse_args()

def main():
    args = parse_args()

    cfg = ARMORConfig(
        debug=args.debug,
        model_key="debug" if args.debug else args.model,
        use_qlora=args.qlora,
    )

    # Override defaults with CLI args if specified
    cfg.saug_adv_steps = args.adv_steps
    cfg.saug_adv_lr = args.adv_lr
    cfg.saug_coeff = args.saug_coeff

    print("\n" + "=" * 72)
    print("  SAUG: STACKELBERG ADVERSARIAL UNLEARNING GAME")
    print("=" * 72)
    print(f"  Model          : {cfg.model_name}")
    print(f"  Device         : {cfg.device}")
    print(f"  Auditor steps  : {cfg.saug_adv_steps}")
    print(f"  Auditor LR     : {cfg.saug_adv_lr}")
    print(f"  SAUG Coeff     : {cfg.saug_coeff}")
    print(f"  Debug          : {cfg.debug}")
    print()

    # Load model and reference model
    print("[SAUG] Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(cfg)
    ref_model = get_frozen_reference_model(model, cfg)

    # Load TOFU dataset
    print("[SAUG] Loading TOFU dataset...")
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
    print("\n[SAUG] === PRE-UNLEARN EVALUATION ===")
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

    # SAUG Training
    print("\n[SAUG] === STARTING SAUG UNLEARNING ===")
    unlearner = SAUGUnlearner(
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
    print("\n[SAUG] === POST-UNLEARN EVALUATION ===")
    post_evaluator = UnlearningEvaluator(model, tokenizer, cfg)
    post_result = post_evaluator.evaluate(
        forget_samples=forget_samples,
        retain_samples=retain_samples,
        forget_loader=eval_fl,
        retain_loader=eval_rl,
        method_name="SAUG",
        run_rouge=not args.no_rouge,
    )
    post_result.print_table()

    print("\n[SAUG] Done.")
    sys.exit(0)

if __name__ == "__main__":
    main()
