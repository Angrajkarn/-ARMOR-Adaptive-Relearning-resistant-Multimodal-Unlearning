"""
scripts/run_muse_benchmark.py
==============================
Run any ARMOR unlearning method on the MUSE benchmark.

Domains  : books | news | github | biomedical
Methods  : ga | npo | npo_sam | rmu | task_vector | who | eul

Usage:
    python scripts/run_muse_benchmark.py --debug --domain books --method npo_sam
    python scripts/run_muse_benchmark.py --model mistral-7b --qlora --domain news --method rmu
"""

import argparse, os, sys, warnings, copy
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore", message=".*symlink.*")
warnings.filterwarnings("ignore", message=".*Xet.*")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armor.config    import ARMORConfig
from armor.data_muse import MUSEDataLoader, MUSE_DOMAINS
from armor.data      import make_dataloader
from armor.model     import get_model_and_tokenizer, save_checkpoint, get_frozen_reference_model
from armor.eval.metrics import UnlearningEvaluator
from armor.eval.mia     import MembershipInferenceAuditor
from armor.eval.privacy_audit import PrivacyAuditor


def parse_args():
    p = argparse.ArgumentParser(description="ARMOR on MUSE Benchmark")
    p.add_argument("--debug",    action="store_true")
    p.add_argument("--model",    default="debug",
                   choices=["debug", "mistral-7b", "llama2-7b"])
    p.add_argument("--qlora",    action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--no-rouge", action="store_true")
    p.add_argument("--run-mia",  action="store_true")
    p.add_argument("--domain",   default="books",
                   choices=list(MUSE_DOMAINS.keys()))
    p.add_argument("--method",   default="npo_sam",
                   choices=["ga", "npo", "npo_sam", "rmu",
                            "task_vector", "who", "eul"])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-save",  action="store_true",
                   help="Skip saving checkpoint (for smoke tests)")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or f"outputs/muse_{args.domain}_{args.method}"
    cfg = ARMORConfig(
        debug      = args.debug,
        model_key  = "debug" if args.debug else args.model,
        use_qlora  = args.qlora,
        hf_token   = args.hf_token,
        output_dir = output_dir,
    )

    print("=" * 60)
    print("  ARMOR -- MUSE Benchmark")
    print("=" * 60)
    print(f"  Domain : {args.domain}  |  Method: {args.method}")
    print(f"  Model  : {cfg.model_name}")
    print("=" * 60)

    model, tokenizer = get_model_and_tokenizer(cfg)
    ref_model        = get_frozen_reference_model(model, cfg)

    # MUSE data (returns DataLoaders directly)
    muse = MUSEDataLoader(cfg, tokenizer, domain=args.domain)
    forget_loader, retain_loader, _ = muse.get_loaders()

    # For evaluation we also need sample-lists — wrap loader contents into lists
    # MUSE uses text (not QA pairs), so we pass loaders directly to evaluator
    # using a simplified accuracy measure
    print(f"\n[muse] Loaded {args.domain}: "
          f"forget={len(forget_loader.dataset)} | "
          f"retain={len(retain_loader.dataset)} samples")

    # ── Unlearning ─────────────────────────────────────────────────────────────
    if args.method == "ga":
        from armor.unlearn.gradient_ascent import GradientAscentUnlearner
        unlearner = GradientAscentUnlearner(model, cfg)
        unlearner.run(forget_loader, retain_loader)

    elif args.method == "npo":
        from armor.unlearn.npo import NPOUnlearner
        NPOUnlearner(model, ref_model, cfg).run(forget_loader, retain_loader)

    elif args.method == "npo_sam":
        from armor.unlearn.npo import NPOUnlearner
        from armor.unlearn.sam_wrapper import SAMOptimizer
        import torch
        base_opt = torch.optim.AdamW(model.parameters(), lr=cfg.unlearn_lr)
        sam_opt  = SAMOptimizer(base_opt, model, rho=cfg.sam_rho)
        NPOUnlearner(model, ref_model, cfg, optimizer=sam_opt).run(
            forget_loader, retain_loader)

    elif args.method == "rmu":
        from armor.unlearn.rmu import RMUUnlearner
        RMUUnlearner(cfg, model, ref_model, tokenizer).train(
            forget_loader, retain_loader)

    elif args.method == "task_vector":
        from armor.unlearn.task_vector import TaskVectorUnlearner
        model = TaskVectorUnlearner(cfg, model, tokenizer).run(
            forget_loader, retain_loader)

    elif args.method == "who":
        from armor.unlearn.who import WHOUnlearner
        WHOUnlearner(cfg, model, tokenizer).train(forget_loader, retain_loader)

    elif args.method == "eul":
        from armor.unlearn.eul import EULUnlearner
        EULUnlearner(cfg, model, tokenizer).train(forget_loader, retain_loader)

    print(f"\n[{args.method}] Unlearning complete on MUSE-{args.domain}")

    if args.run_mia:
        print("\n[privacy] Running privacy audit...")
        auditor = PrivacyAuditor(cfg, model, tokenizer)
        result  = auditor.audit(forget_loader, retain_loader,
                                method_name=args.method)
        result.print_summary()

    if not args.no_save:
        os.makedirs(output_dir, exist_ok=True)
        ckpt = os.path.join(output_dir, "unlearned")
        save_checkpoint(model, tokenizer, ckpt, cfg)
        print(f"\n[done] MUSE-{args.domain}/{args.method} -> {ckpt}")
    else:
        print(f"\n[done] MUSE-{args.domain}/{args.method} (--no-save: checkpoint skipped)")


if __name__ == "__main__":
    main()
