"""
scripts/run_multitask_unlearn.py
=================================
Multi-task NPO: forget K topics simultaneously with orthogonal gradients.

Usage:
    python scripts/run_multitask_unlearn.py --debug --n-tasks 2 --no-rouge
    python scripts/run_multitask_unlearn.py --model mistral-7b --qlora --n-tasks 3
"""

import argparse, os, sys, warnings, copy
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore", message=".*symlink.*")
warnings.filterwarnings("ignore", message=".*Xet.*")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch.utils.data import DataLoader, Subset

from armor.config import ARMORConfig
from armor.data   import load_tofu_splits, make_dataloader
from armor.model  import get_model_and_tokenizer, save_checkpoint, get_frozen_reference_model
from armor.eval.metrics import UnlearningEvaluator
from armor.eval.mia     import MembershipInferenceAuditor
from armor.unlearn.multitask_npo import MultiTaskNPOUnlearner


def parse_args():
    p = argparse.ArgumentParser(description="ARMOR: Multi-Task NPO Unlearning")
    p.add_argument("--debug",       action="store_true")
    p.add_argument("--model",       default="debug",
                   choices=["debug", "mistral-7b", "llama2-7b"])
    p.add_argument("--qlora",       action="store_true")
    p.add_argument("--hf-token",    default=None)
    p.add_argument("--no-rouge",    action="store_true")
    p.add_argument("--run-mia",     action="store_true")
    p.add_argument("--n-tasks",     type=int,   default=2,
                   help="Split forget set into N tasks")
    p.add_argument("--beta-npo",    type=float, default=0.1)
    p.add_argument("--beta-retain", type=float, default=1.0)
    p.add_argument("--output-dir",  default="outputs/multitask_npo")
    return p.parse_args()


def split_into_tasks(samples: list, n_tasks: int) -> list:
    """Split sample list into N roughly equal sub-lists."""
    chunk = max(1, len(samples) // n_tasks)
    parts = []
    for k in range(n_tasks):
        start = k * chunk
        end   = start + chunk if k < n_tasks - 1 else len(samples)
        parts.append(samples[start:end])
    return parts


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
    print("  ARMOR -- Multi-Task NPO Unlearning")
    print("=" * 58)
    print(f"  Model   : {cfg.model_name}  |  N-tasks: {args.n_tasks}")
    print("=" * 58)

    forget_samples, retain_samples = load_tofu_splits(cfg)
    model, tokenizer               = get_model_and_tokenizer(cfg)
    ref_model                      = get_frozen_reference_model(model, cfg)

    eval_fl = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    eval_rl = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)

    # Split forget into per-task loaders
    task_sample_lists = split_into_tasks(forget_samples, args.n_tasks)
    task_names        = [f"Author_{chr(65+k)}" for k in range(args.n_tasks)]
    task_loaders      = [
        make_dataloader(s, tokenizer, cfg, shuffle=True)
        for s in task_sample_lists
    ]
    retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=True)

    print(f"[multitask] Tasks: {task_names}")
    for n, tl in zip(task_names, task_loaders):
        print(f"  {n}: {len(tl.dataset)} samples")

    evaluator  = UnlearningEvaluator(model, tokenizer, cfg)
    pre_result = evaluator.evaluate(
        forget_samples, retain_samples, eval_fl, eval_rl,
        method_name="Pre-unlearn",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50)
    pre_result.print_table()

    unlearner = MultiTaskNPOUnlearner(
        cfg         = cfg,
        model       = model,
        ref_model   = ref_model,
        tokenizer   = tokenizer,
        task_names  = task_names,
        beta_npo    = args.beta_npo,
        beta_retain = args.beta_retain)
    unlearner.train(task_loaders, retain_loader)

    post_result = evaluator.evaluate(
        forget_samples, retain_samples, eval_fl, eval_rl,
        method_name="MultiTask-NPO",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50)
    post_result.print_table()

    if args.run_mia:
        MembershipInferenceAuditor(model, tokenizer, cfg).audit(
            eval_fl, eval_rl, method_name="MultiTask-NPO")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt = os.path.join(args.output_dir, "multitask_unlearned")
    save_checkpoint(model, tokenizer, ckpt, cfg)
    print(f"\n[done] Saved to: {ckpt}")
    print(f"  forget_quality : {post_result.forget_quality:.4f}")
    print(f"  retain_accuracy: {post_result.retain_accuracy:.4f}")


if __name__ == "__main__":
    main()
