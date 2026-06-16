"""
scripts/run_dp_armor.py
========================
DP-NPO+SAM: full ARMOR stack with Differential Privacy certificate.

Usage:
    python scripts/run_dp_armor.py --debug --no-rouge
    python scripts/run_dp_armor.py --model mistral-7b --qlora --epsilon 8.0 --run-mia
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
from armor.eval.privacy_audit import PrivacyAuditor
from armor.unlearn.dp_npo_sam import DPNPOSAMUnlearner


def parse_args():
    p = argparse.ArgumentParser(description="ARMOR: DP-NPO+SAM")
    p.add_argument("--debug",     action="store_true")
    p.add_argument("--model",     default="debug",
                   choices=["debug", "mistral-7b", "llama2-7b"])
    p.add_argument("--qlora",     action="store_true")
    p.add_argument("--hf-token",  default=None)
    p.add_argument("--no-rouge",  action="store_true")
    p.add_argument("--run-mia",   action="store_true")
    p.add_argument("--epsilon",   type=float, default=8.0,
                   help="Target epsilon (training stops when reached)")
    p.add_argument("--delta",     type=float, default=1e-5)
    p.add_argument("--noise",     type=float, default=1.0,
                   help="DP-SGD noise multiplier sigma")
    p.add_argument("--clip",      type=float, default=1.0,
                   help="DP-SGD gradient clip norm C")
    p.add_argument("--sam-rho",   type=float, default=0.05)
    p.add_argument("--beta-npo",  type=float, default=0.1)
    p.add_argument("--n-samples", type=int,   default=1000,
                   help="Total dataset size for privacy accounting")
    p.add_argument("--output-dir", default="outputs/dp_npo_sam")
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

    print("=" * 62)
    print("  ARMOR -- DP-NPO+SAM (Full Privacy Stack)")
    print("=" * 62)
    print(f"  Model   : {cfg.model_name}")
    print(f"  Target  : epsilon={args.epsilon}  delta={args.delta}")
    print(f"  DP-SGD  : noise={args.noise}  clip={args.clip}")
    print(f"  SAM rho : {args.sam_rho}")
    print("=" * 62)

    forget_samples, retain_samples = load_tofu_splits(cfg)
    model, tokenizer               = get_model_and_tokenizer(cfg)
    ref_model                      = get_frozen_reference_model(model, cfg)

    eval_fl       = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    eval_rl       = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)
    forget_loader = make_dataloader(
        forget_samples, tokenizer, cfg,
        include_rephrases=cfg.use_rephrase_augmentation, shuffle=True)
    retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=True)

    evaluator  = UnlearningEvaluator(model, tokenizer, cfg)
    pre_result = evaluator.evaluate(
        forget_samples, retain_samples, eval_fl, eval_rl,
        method_name="Pre-unlearn",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50)
    pre_result.print_table()

    unlearner = DPNPOSAMUnlearner(
        cfg              = cfg,
        model            = model,
        ref_model        = ref_model,
        tokenizer        = tokenizer,
        noise_multiplier = args.noise,
        max_grad_norm    = args.clip,
        sam_rho          = args.sam_rho,
        beta_npo         = args.beta_npo,
        target_epsilon   = args.epsilon,
        target_delta     = args.delta)

    history = unlearner.train(forget_loader, retain_loader,
                               n_samples=args.n_samples)

    final_eps = history["epsilon"][-1] if history["epsilon"] else float("nan")

    post_result = evaluator.evaluate(
        forget_samples, retain_samples, eval_fl, eval_rl,
        method_name="DP-NPO+SAM",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50)
    post_result.print_table()

    print(f"\n[privacy] Formal epsilon achieved: {final_eps:.4f}  "
          f"(delta={args.delta})")

    if args.run_mia:
        auditor = PrivacyAuditor(cfg, model, tokenizer)
        result  = auditor.audit(eval_fl, eval_rl,
                                method_name="DP-NPO+SAM",
                                formal_epsilon=final_eps,
                                delta=args.delta)
        result.print_summary()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt = os.path.join(args.output_dir, "dp_npo_sam_unlearned")
    save_checkpoint(model, tokenizer, ckpt, cfg)
    print(f"\n[done] epsilon={final_eps:.3f} | Saved -> {ckpt}")
    print(f"  forget_quality : {post_result.forget_quality:.4f}")
    print(f"  retain_accuracy: {post_result.retain_accuracy:.4f}")


if __name__ == "__main__":
    main()
