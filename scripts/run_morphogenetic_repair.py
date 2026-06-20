"""
scripts/run_morphogenetic_repair.py
===================================
MWRP: Morphogenetic Weight Regeneration Runner
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
from armor.unlearn.npo import NPOUnlearner
from armor.unlearn.morphogenetic_repair import MWRPRepairer
from armor.eval.metrics import UnlearningEvaluator

def parse_args():
    p = argparse.ArgumentParser(description="MWRP: Morphogenetic Weight Regeneration Post-Unlearning")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--model", default="debug")
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--damage-threshold", type=float, default=0.01)
    p.add_argument("--repair-epochs", type=int, default=2)
    p.add_argument("--no-rouge", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--output-dir", default="outputs/morphogenetic_repair")
    return p.parse_args()

def main():
    args = parse_args()

    cfg = ARMORConfig(
        debug=args.debug,
        model_key="debug" if args.debug else args.model,
        use_qlora=args.qlora,
    )

    # Set parameters
    cfg.mwrp_damage_threshold = args.damage_threshold
    cfg.mwrp_repair_epochs = args.repair_epochs

    print("\n" + "=" * 72)
    print("  MWRP: MORPHOGENETIC WEIGHT REGENERATION POST-UNLEARNING")
    print("=" * 72)
    print(f"  Model             : {cfg.model_name}")
    print(f"  Device            : {cfg.device}")
    print(f"  Damage threshold  : {cfg.mwrp_damage_threshold}")
    print(f"  Repair epochs     : {cfg.mwrp_repair_epochs}")
    print(f"  Debug             : {cfg.debug}")
    print()

    # Load model and pre-model reference
    print("[MWRP] Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(cfg)
    pre_model = get_frozen_reference_model(model, cfg)

    # Load TOFU dataset
    print("[MWRP] Loading TOFU dataset...")
    forget_samples, retain_samples = load_tofu_splits(cfg)

    forget_loader = make_dataloader(
        forget_samples, tokenizer, cfg,
        include_rephrases=True, shuffle=True,
    )
    retain_loader = make_dataloader(
        retain_samples, tokenizer, cfg,
        shuffle=True,
    )

    # Pre-evaluation (before any unlearning/repair)
    print("\n[MWRP] === PRE-UNLEARN EVALUATION ===")
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

    # Step 1: Perform unlearning (use NPO as baseline unlearning) to introduce "damage"
    print("\n[MWRP] === INTRODUCING UNLEARNING DAMAGE (NPO) ===")
    unlearner = NPOUnlearner(
        model=model,
        ref_model=pre_model,
        cfg=cfg,
    )
    unlearner.run(forget_loader, retain_loader)

    print("\n[MWRP] === POST-UNLEARN (DAMAGED) EVALUATION ===")
    post_unlearn_eval = UnlearningEvaluator(model, tokenizer, cfg)
    damaged_result = post_unlearn_eval.evaluate(
        forget_samples=forget_samples,
        retain_samples=retain_samples,
        forget_loader=eval_fl,
        retain_loader=eval_rl,
        method_name="Damaged (Post-NPO)",
        run_rouge=not args.no_rouge,
    )
    damaged_result.print_table()

    # Step 2: Perform morphogenetic repair on the unlearned model
    print("\n[MWRP] === STARTING MORPHOGENETIC REPAIR ===")
    repairer = MWRPRepairer(
        model=model,
        pre_model=pre_model,
        tokenizer=tokenizer,
        cfg=cfg,
    )

    # Run repair distillation on the retain set
    repair_result = repairer.run(retain_loader)

    # Post-repair evaluation
    print("\n[MWRP] === POST-REPAIR EVALUATION ===")
    post_repair_eval = UnlearningEvaluator(model, tokenizer, cfg)
    repaired_result = post_repair_eval.evaluate(
        forget_samples=forget_samples,
        retain_samples=retain_samples,
        forget_loader=eval_fl,
        retain_loader=eval_rl,
        method_name="Repaired (MWRP)",
        run_rouge=not args.no_rouge,
    )
    repaired_result.print_table()

    print("\n[MWRP] Done.")

    # Save evaluation results to JSON
    import json
    res_dict = {
        "forget_quality": repaired_result.forget_quality,
        "forget_accuracy": repaired_result.forget_accuracy,
        "retain_accuracy": repaired_result.retain_accuracy,
        "mia_auroc": getattr(repaired_result, "mia_auroc", -1.0)
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(res_dict, f, indent=2)

    sys.exit(0)

if __name__ == "__main__":
    main()
