"""
scripts/run_rlace_rmu.py
========================
Run Advanced RMU with RLACE Concept Erasure on TOFU.

Usage
-----
  python scripts/run_rlace_rmu.py --debug
  python scripts/run_rlace_rmu.py --model mistral-7b --qlora
"""

import argparse
import os
import sys
import warnings

# Make armor importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore", message=".*symlink.*")

import torch
from armor.config import ARMORConfig
from armor.data   import load_tofu_splits, make_dataloader
from armor.model  import get_model_and_tokenizer, save_checkpoint, get_frozen_reference_model
from armor.unlearn.rlace_rmu import RLACERMUUnlearner
from armor.eval.metrics  import UnlearningEvaluator
from armor.eval.mia      import MembershipInferenceAuditor


def parse_args():
    p = argparse.ArgumentParser(description="ARMOR — RLACE RMU Unlearning")
    p.add_argument("--debug",      action="store_true")
    p.add_argument("--model",      default="debug", choices=["debug","mistral-7b","llama2-7b"])
    p.add_argument("--qlora",      action="store_true")
    p.add_argument("--hf-token",   default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--output-dir", default="outputs/rlace_rmu")
    p.add_argument("--no-rouge",   action="store_true")
    p.add_argument("--run-mia",    action="store_true")
    p.add_argument("--no-save",    action="store_true")
    p.add_argument("--alpha",      type=float, default=1200.0)
    p.add_argument("--beta",       type=float, default=6.5)
    p.add_argument("--rlace-layers", type=int, default=3)
    p.add_argument("--rlace-epochs", type=int, default=10)
    p.add_argument("--rlace-iters",  type=int, default=300)
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
    if args.epochs: cfg.unlearn_epochs = args.epochs
    if args.lr:     cfg.unlearn_lr     = args.lr
    cfg.rlace_n_layers = args.rlace_layers
    cfg.rlace_probe_epochs = args.rlace_epochs
    cfg.rlace_whittle_iters = args.rlace_iters

    print("=" * 60)
    print(f"  ARMOR — RLACE RMU Unlearning")
    print(f"  Model        : {cfg.model_name}")
    print(f"  Layers       : {cfg.rlace_n_layers}")
    print(f"  Probe Epochs : {cfg.rlace_probe_epochs}")
    print(f"  Whittle Iter : {cfg.rlace_whittle_iters}")
    print("=" * 60)

    # ── Load Model and Data ───────────────────────────────────────────────────
    forget_samples, retain_samples = load_tofu_splits(cfg)
    model, tokenizer = get_model_and_tokenizer(cfg)
    ref_model = get_frozen_reference_model(model, cfg)

    # Note: prepare/fit needs loaders with batch size = 1 (or small) and no shuffle for extracting representations,
    # but RLACERMUUnlearner._collect_hidden_states works on any loader. Let's use clean eval loaders for prepare phase,
    # and training loaders for train loop.
    eval_forget_loader = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    eval_retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)

    evaluator = UnlearningEvaluator(model, tokenizer, cfg)

    # ── Pre-unlearning Evaluation ─────────────────────────────────────────────
    print("\n[main] Pre-unlearning evaluation:")
    pre_result = evaluator.evaluate(
        forget_samples, retain_samples,
        eval_forget_loader, eval_retain_loader,
        method_name="Pre-unlearning (RLACE RMU)",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50,
    )
    pre_result.print_table()

    # ── Initialize RLACE RMU Unlearner ────────────────────────────────────────
    unlearner = RLACERMUUnlearner(
        cfg=cfg,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        alpha=args.alpha,
        beta=args.beta
    )

    # ── Training Loader Setup ─────────────────────────────────────────────────
    forget_loader = make_dataloader(forget_samples, tokenizer, cfg, include_rephrases=False, shuffle=True)
    retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=True)

    # Prepare and Train
    unlearner.train(forget_loader, retain_loader)

    # ── Post-unlearning Evaluation ────────────────────────────────────────────
    print("\n[main] Post-unlearning evaluation:")
    post_result = evaluator.evaluate(
        forget_samples, retain_samples,
        eval_forget_loader, eval_retain_loader,
        method_name="RLACE-RMU",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50,
    )
    post_result.print_table()

    if args.run_mia:
        print("\n[main] Running MIA audit...")
        auditor = MembershipInferenceAuditor(model, tokenizer, cfg)
        mia_result = auditor.audit(eval_forget_loader, eval_retain_loader, method_name="RLACE-RMU")
        post_result.mia_auroc = mia_result.auroc
        post_result.print_table()

    # ── Save Checkpoint ───────────────────────────────────────────────────────
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_path = os.path.join(args.output_dir, "rlace_rmu_unlearned")
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