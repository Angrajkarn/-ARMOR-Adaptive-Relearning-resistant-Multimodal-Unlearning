"""
scripts/run_lcage.py
====================
LCAGE: Latent Concept Association Graph Erasure Runner
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
from armor.unlearn.lcage import LCAGEUnlearner
from armor.eval.metrics import UnlearningEvaluator

def parse_args():
    p = argparse.ArgumentParser(description="LCAGE: Latent Concept Association Graph Erasure")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--model", default="debug")
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--lcage-coeff", type=float, default=0.3)
    p.add_argument("--lcage-threshold", type=float, default=0.5)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--no-rouge", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--output-dir", default="outputs/lcage")
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
    cfg.lcage_coeff = args.lcage_coeff
    cfg.lcage_pmi_threshold = args.lcage_threshold

    print("\n" + "=" * 72)
    print("  LCAGE: LATENT CONCEPT ASSOCIATION GRAPH ERASURE")
    print("=" * 72)
    print(f"  Model          : {cfg.model_name}")
    print(f"  Device         : {cfg.device}")
    print(f"  LCAGE coeff    : {cfg.lcage_coeff}")
    print(f"  PMI threshold  : {cfg.lcage_pmi_threshold}")
    print(f"  Debug          : {cfg.debug}")
    print()

    # Load model and reference model
    print("[LCAGE] Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(cfg)
    ref_model = get_frozen_reference_model(model, cfg)

    # Load TOFU dataset
    print("[LCAGE] Loading TOFU dataset...")
    forget_samples, retain_samples = load_tofu_splits(cfg)

    forget_loader = make_dataloader(
        forget_samples, tokenizer, cfg,
        include_rephrases=True, shuffle=True,
    )
    retain_loader = make_dataloader(
        retain_samples, tokenizer, cfg,
        shuffle=True,
    )

    # Prepare QA pairs for graph construction
    qa_forget_pairs = [(s.question, s.answer) for s in forget_samples]
    forget_questions = [s.question for s in forget_samples for _ in range(3 if cfg.use_rephrase_augmentation else 1)]

    # Pre-evaluation
    print("\n[LCAGE] === PRE-UNLEARN EVALUATION ===")
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

    # LCAGE Training
    print("\n[LCAGE] === STARTING LCAGE UNLEARNING ===")
    unlearner = LCAGEUnlearner(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        cfg=cfg,
        qa_forget_pairs=qa_forget_pairs,
    )

    result = unlearner.run(
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        forget_questions=forget_questions,
    )

    # Post-evaluation
    print("\n[LCAGE] === POST-UNLEARN EVALUATION ===")
    post_evaluator = UnlearningEvaluator(model, tokenizer, cfg)
    post_result = post_evaluator.evaluate(
        forget_samples=forget_samples,
        retain_samples=retain_samples,
        forget_loader=eval_fl,
        retain_loader=eval_rl,
        method_name="LCAGE",
        run_rouge=not args.no_rouge,
    )
    post_result.print_table()

    print("\n[LCAGE] Done.")

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
