"""
scripts/run_rmu.py
==================
Run RMU (Representation Misdirection Unlearning) on TOFU benchmark.

Usage:
    python scripts/run_rmu.py --debug --no-rouge
    python scripts/run_rmu.py --model mistral-7b --qlora --run-mia
    python scripts/run_rmu.py --debug --layer 3 --alpha 800 --beta 4.0
"""

import argparse, os, sys, warnings, copy
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore", message=".*symlink.*")
warnings.filterwarnings("ignore", message=".*Xet.*")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armor.config import ARMORConfig
from armor.data   import load_tofu_splits, make_dataloader
from armor.model  import get_model_and_tokenizer, save_checkpoint, get_frozen_reference_model
from armor.eval.metrics import UnlearningEvaluator
from armor.eval.mia     import MembershipInferenceAuditor
from armor.unlearn.rmu  import RMUUnlearner


def parse_args():
    p = argparse.ArgumentParser(description="ARMOR: RMU Unlearning")
    p.add_argument("--debug",    action="store_true")
    p.add_argument("--model",    default="debug",
                   choices=["debug", "mistral-7b", "llama2-7b"])
    p.add_argument("--qlora",    action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--no-rouge", action="store_true")
    p.add_argument("--run-mia",  action="store_true")
    p.add_argument("--no-save",  action="store_true",
                   help="Skip saving checkpoint (for smoke tests)")
    p.add_argument("--layer",    type=int,   default=None)
    p.add_argument("--alpha",    type=float, default=1200.0)
    p.add_argument("--beta",     type=float, default=6.5)
    p.add_argument("--c-scale",  type=float, default=20.0)
    p.add_argument("--output-dir", default="outputs/rmu")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = ARMORConfig(
        debug      = args.debug,
        model_key  = "debug" if args.debug else args.model,
        use_qlora  = args.qlora,
        hf_token   = args.hf_token,
        output_dir = args.output_dir,
    )

    print("=" * 58)
    print("  ARMOR -- RMU Unlearning")
    print("=" * 58)
    print(f"  Model : {cfg.model_name}  |  alpha={args.alpha}  beta={args.beta}")
    print("=" * 58)

    # ── Data & Model ──────────────────────────────────────────────────────────
    forget_samples, retain_samples = load_tofu_splits(cfg)
    model, tokenizer               = get_model_and_tokenizer(cfg)
    ref_model                      = get_frozen_reference_model(model, cfg)

    forget_loader = make_dataloader(forget_samples, tokenizer, cfg,
                                    include_rephrases=cfg.use_rephrase_augmentation,
                                    shuffle=True)
    retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=True)
    eval_fl       = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    eval_rl       = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)

    # ── Pre-eval ──────────────────────────────────────────────────────────────
    evaluator   = UnlearningEvaluator(model, tokenizer, cfg)
    pre_result  = evaluator.evaluate(
        forget_samples, retain_samples, eval_fl, eval_rl,
        method_name="Pre-unlearn",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50)
    pre_result.print_table()

    # ── RMU Training ──────────────────────────────────────────────────────────
    unlearner = RMUUnlearner(
        cfg       = cfg,
        model     = model,
        ref_model = ref_model,
        tokenizer = tokenizer,
        layer_idx = args.layer,
        alpha     = args.alpha,
        beta      = args.beta,
        c_scale   = args.c_scale)
    unlearner.train(forget_loader, retain_loader)

    # ── Post-eval ─────────────────────────────────────────────────────────────
    post_result = evaluator.evaluate(
        forget_samples, retain_samples, eval_fl, eval_rl,
        method_name="RMU",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50)
    post_result.print_table()

    if args.run_mia:
        auditor = MembershipInferenceAuditor(model, tokenizer, cfg)
        mia_res = auditor.audit(eval_fl, eval_rl, method_name="RMU")
        post_result.mia_auroc = mia_res.auroc

    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt = os.path.join(args.output_dir, "rmu_unlearned")
        save_checkpoint(model, tokenizer, ckpt, cfg)
        print(f"\n[done] Checkpoint saved to: {ckpt}")
    else:
        print("\n[done] (--no-save: checkpoint skipped)")
    print(f"  forget_quality : {post_result.forget_quality:.4f}")
    print(f"  retain_accuracy: {post_result.retain_accuracy:.4f}")

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