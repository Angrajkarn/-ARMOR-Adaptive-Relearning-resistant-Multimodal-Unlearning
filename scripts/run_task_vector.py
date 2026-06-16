"""
scripts/run_task_vector.py
==========================
Task Vector Unlearning: negate the forget fine-tune direction.

Usage:
    python scripts/run_task_vector.py --debug --no-rouge
    python scripts/run_task_vector.py --model mistral-7b --qlora --lam 1.5
"""

import argparse, os, sys, warnings
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore", message=".*symlink.*")
warnings.filterwarnings("ignore", message=".*Xet.*")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armor.config import ARMORConfig
from armor.data   import load_tofu_splits, make_dataloader
from armor.model  import get_model_and_tokenizer, save_checkpoint
from armor.eval.metrics import UnlearningEvaluator
from armor.eval.mia     import MembershipInferenceAuditor
from armor.unlearn.task_vector import TaskVectorUnlearner


def parse_args():
    p = argparse.ArgumentParser(description="ARMOR: Task Vector Unlearning")
    p.add_argument("--debug",     action="store_true")
    p.add_argument("--model",     default="debug",
                   choices=["debug", "mistral-7b", "llama2-7b"])
    p.add_argument("--qlora",     action="store_true")
    p.add_argument("--hf-token",  default=None)
    p.add_argument("--no-rouge",  action="store_true")
    p.add_argument("--run-mia",   action="store_true")
    p.add_argument("--no-save",   action="store_true",
                   help="Skip saving checkpoint (for smoke tests)")
    p.add_argument("--lam",       type=float, default=1.0,
                   help="Negation scale lambda (higher = stronger forget)")
    p.add_argument("--ft-epochs", type=int,   default=3)
    p.add_argument("--ft-lr",     type=float, default=5e-5)
    p.add_argument("--output-dir", default="outputs/task_vector")
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
    print("  ARMOR -- Task Vector Unlearning")
    print("=" * 58)
    print(f"  Model : {cfg.model_name}  |  lambda={args.lam}")
    print("=" * 58)

    forget_samples, retain_samples = load_tofu_splits(cfg)
    model, tokenizer               = get_model_and_tokenizer(cfg)

    eval_fl = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    eval_rl = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)
    forget_loader = make_dataloader(forget_samples, tokenizer, cfg, shuffle=True)
    retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=True)

    evaluator  = UnlearningEvaluator(model, tokenizer, cfg)
    pre_result = evaluator.evaluate(
        forget_samples, retain_samples, eval_fl, eval_rl,
        method_name="Pre-unlearn",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50)
    pre_result.print_table()

    # Task Vector returns a NEW model (weight arithmetic)
    unlearner = TaskVectorUnlearner(
        cfg             = cfg,
        model           = model,
        tokenizer       = tokenizer,
        forget_scale    = args.lam,
        finetune_epochs = args.ft_epochs,
        finetune_lr     = args.ft_lr)
    unlearned_model = unlearner.run(forget_loader, retain_loader)

    # Evaluate with the returned model
    post_eval = UnlearningEvaluator(unlearned_model, tokenizer, cfg)
    # Rebuild loaders (same data, new model)
    post_result = post_eval.evaluate(
        forget_samples, retain_samples, eval_fl, eval_rl,
        method_name="Task-Vector",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50)
    post_result.print_table()

    if args.run_mia:
        auditor = MembershipInferenceAuditor(unlearned_model, tokenizer, cfg)
        auditor.audit(eval_fl, eval_rl, method_name="Task-Vector")

    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt = os.path.join(args.output_dir, "task_vector_unlearned")
        save_checkpoint(unlearned_model, tokenizer, ckpt, cfg)
        print(f"\n[done] Saved to: {ckpt}")
    else:
        print("\n[done] (--no-save: checkpoint skipped)")
    print(f"  forget_quality : {post_result.forget_quality:.4f}")
    print(f"  retain_accuracy: {post_result.retain_accuracy:.4f}")


if __name__ == "__main__":
    main()
