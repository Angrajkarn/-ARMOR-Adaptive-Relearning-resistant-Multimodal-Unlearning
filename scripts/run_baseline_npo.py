"""
scripts/run_baseline_npo.py
============================
Run NPO (Negative Preference Optimization) unlearning baseline on TOFU.

Usage
-----
  python scripts/run_baseline_npo.py --debug
  python scripts/run_baseline_npo.py --model mistral-7b --qlora
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armor.config import ARMORConfig
from armor.data   import load_tofu_splits, make_dataloader
from armor.model  import get_model_and_tokenizer, get_frozen_reference_model, save_checkpoint
from armor.unlearn.npo   import NPOUnlearner
from armor.eval.metrics  import UnlearningEvaluator
from armor.eval.mia      import MembershipInferenceAuditor


def parse_args():
    p = argparse.ArgumentParser(description="ARMOR — NPO Baseline")
    p.add_argument("--debug",      action="store_true")
    p.add_argument("--model",      default="debug",
                   choices=["debug", "mistral-7b", "llama2-7b"])
    p.add_argument("--qlora",      action="store_true")
    p.add_argument("--hf-token",   default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--npo-beta",   type=float, default=None,
                   help="NPO temperature β (default from config: 0.1)")
    p.add_argument("--output-dir", default="outputs/npo")
    p.add_argument("--no-rouge",   action="store_true")
    p.add_argument("--run-mia",    action="store_true")
    p.add_argument("--no-save",    action="store_true",
                   help="Skip saving checkpoint (for smoke tests)")
    # ── Speed flags (for Kaggle / T4 GPU) ─────────────────────────────────────
    p.add_argument("--fast",       action="store_true",
                   help="Kaggle speed preset: cap retain=200, fp16, skip pre-eval ROUGE, "
                        "max_rouge_samples=10. Reduces 6-7hr runs to ~20-30min.")
    p.add_argument("--max-retain", type=int, default=None,
                   help="Cap retain set size (e.g. 200). 0 = full retain99 (3960 samples).")
    p.add_argument("--fp16",       action="store_true",
                   help="Enable fp16 autocast during training (T4/V100).")
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
    if args.epochs:   cfg.unlearn_epochs = args.epochs
    if args.lr:       cfg.unlearn_lr     = args.lr
    if args.npo_beta: cfg.npo_beta       = args.npo_beta

    # ── Apply --fast / speed flags ───────────────────────────────────────────────
    if args.fast:
        cfg.max_retain_samples   = 200
        cfg.use_fp16             = True
        cfg.rouge_max_new_tokens = 32
        print("[fast] Speed preset active: retain=200, fp16=True, rouge_tokens=32")
    if args.max_retain is not None:
        cfg.max_retain_samples = args.max_retain
    if args.fp16:
        cfg.use_fp16 = True

    print("=" * 60)
    print(f"  ARMOR — NPO Baseline")
    print(f"  Model  : {cfg.model_name}")
    print(f"  Device : {cfg.device}")
    print(f"  β      : {cfg.npo_beta}")
    print(f"  fp16   : {cfg.use_fp16}")
    print(f"  Retain : {cfg.max_retain_samples if cfg.max_retain_samples > 0 else 'full (retain99)'}")
    print("=" * 60)

    # ── Data ────────────────────────────────────────────────────────────────────
    forget_samples, retain_samples = load_tofu_splits(cfg)

    # ── Model + Reference ────────────────────────────────────────────────────────
    # NPO requires a frozen copy of the original model (π_ref)
    model, tokenizer = get_model_and_tokenizer(cfg)
    ref_model        = get_frozen_reference_model(model, cfg)

    # ── Pre-unlearning baseline ────────────────────────────────────────────────────
    eval_forget_loader = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    eval_retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)

    evaluator = UnlearningEvaluator(model, tokenizer, cfg)
    # In --fast mode, skip pre-eval ROUGE to save 5-10 min on 7B models
    run_pre_rouge = (not args.no_rouge) and (not args.fast)
    max_rouge     = 10 if args.fast else (20 if cfg.debug else 50)
    print("\n[main] Pre-unlearning evaluation:")
    pre_result = evaluator.evaluate(
        forget_samples, retain_samples,
        eval_forget_loader, eval_retain_loader,
        method_name="Pre-unlearning (no NPO)",
        run_rouge=run_pre_rouge,
        max_rouge_samples=max_rouge,
    )
    pre_result.print_table()
    original_forget_acc = pre_result.forget_accuracy

    # ── Unlearning ────────────────────────────────────────────────────────────────
    forget_loader = make_dataloader(
        forget_samples, tokenizer, cfg,
        include_rephrases=cfg.use_rephrase_augmentation,
        shuffle=True,
    )
    retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=True)

    print("\n[main] Starting NPO unlearning...")
    unlearner = NPOUnlearner(model, ref_model, cfg)
    train_result = unlearner.run(forget_loader, retain_loader)

    # ── Post-unlearning evaluation ────────────────────────────────────────────────────
    print("\n[main] Post-unlearning evaluation:")
    post_result = evaluator.evaluate(
        forget_samples, retain_samples,
        eval_forget_loader, eval_retain_loader,
        method_name="NPO",
        run_rouge=not args.no_rouge,
        max_rouge_samples=max_rouge,
    )
    post_result.print_table()

    # ── Optional MIA audit ────────────────────────────────────────────────────────
    if args.run_mia:
        auditor   = MembershipInferenceAuditor(model, tokenizer, cfg)
        mia_result = auditor.audit(eval_forget_loader, eval_retain_loader, "NPO")
        post_result.mia_auroc = mia_result.auroc
        post_result.print_table()

    # ── Save ───────────────────────────────────────────────────────────
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        save_checkpoint(model, tokenizer, os.path.join(args.output_dir, "npo_unlearned"), cfg)

    print(f"\n[main] NPO complete.")
    print(f"  Forget quality : {post_result.forget_quality:.4f}")
    print(f"  Retain accuracy: {post_result.retain_accuracy:.4f}")

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