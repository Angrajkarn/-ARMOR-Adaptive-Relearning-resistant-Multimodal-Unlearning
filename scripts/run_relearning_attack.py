"""
scripts/run_relearning_attack.py
=================================
Simulate a relearning attack on a previously unlearned model checkpoint.

This script:
1. Loads a saved checkpoint from one of the unlearning scripts
2. Applies the relearning attack (fine-tune on N forget samples)
3. Tracks accuracy recovery over epochs
4. Compares across methods if multiple checkpoints are provided

Usage
-----
  # Attack a single checkpoint:
  python scripts/run_relearning_attack.py \\
      --checkpoint outputs/npo_sam/npo_sam_unlearned \\
      --method "NPO+SAM" \\
      --debug

  # Compare all three methods side by side:
  python scripts/run_relearning_attack.py \\
      --compare \\
      --ga-checkpoint   outputs/ga/ga_unlearned \\
      --npo-checkpoint  outputs/npo/npo_unlearned \\
      --sam-checkpoint  outputs/npo_sam/npo_sam_unlearned
"""

import argparse
import os
import sys

# Set UTF-8 encoding for stdout and stderr on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armor.config import ARMORConfig
from armor.data   import load_tofu_splits
from armor.model  import load_checkpoint
from armor.attack.relearning import RelearningAttack


def parse_args():
    p = argparse.ArgumentParser(description="ARMOR — Relearning Attack")
    p.add_argument("--debug",         action="store_true")
    p.add_argument("--model",         default="debug",
                   choices=["debug", "mistral-7b", "llama2-7b"])
    p.add_argument("--qlora",         action="store_true")
    p.add_argument("--hf-token",      default=None)

    # Single checkpoint mode
    p.add_argument("--checkpoint",    default=None,
                   help="Path to unlearned model checkpoint")
    p.add_argument("--method",        default="Unknown",
                   help="Name of the unlearning method (for display)")
    p.add_argument("--original-acc",  type=float, default=1.0,
                   help="Forget accuracy BEFORE unlearning (for recovery % calc)")

    # Comparison mode
    p.add_argument("--compare",       action="store_true",
                   help="Compare GA vs NPO vs NPO+SAM attack recovery")
    p.add_argument("--ga-checkpoint",  default="outputs/ga/ga_unlearned")
    p.add_argument("--npo-checkpoint", default="outputs/npo/npo_unlearned")
    p.add_argument("--sam-checkpoint", default="outputs/npo_sam/npo_sam_unlearned")

    p.add_argument("--n-samples",     type=int, default=None,
                   help="Number of forget samples for attack (default from config)")
    p.add_argument("--epochs",        type=int, default=None)
    p.add_argument("--no-save",       action="store_true",
                   help="No-op: relearning attack never saves checkpoints")
    return p.parse_args()


def run_attack_on_checkpoint(ckpt_path: str, method_name: str,
                              forget_samples, retain_samples,
                              cfg: ARMORConfig,
                              original_forget_acc: float = 1.0):
    """Load checkpoint and run a relearning attack on it."""
    if not os.path.exists(ckpt_path):
        print(f"[attack] Checkpoint not found: {ckpt_path}")
        print(f"[attack] Run the unlearning script first, then attack.")
        return None

    model, tokenizer = load_checkpoint(ckpt_path, cfg)

    attacker = RelearningAttack(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        forget_samples=forget_samples,
        retain_samples=retain_samples,
        method_name=method_name,
        original_forget_acc=original_forget_acc,
    )
    result = attacker.run()
    result.print_summary()
    return result


def main():
    args = parse_args()

    cfg = ARMORConfig(
        debug=args.debug,
        model_key=args.model if not args.debug else "debug",
        use_qlora=args.qlora,
        hf_token=args.hf_token,
    )
    if args.n_samples: cfg.relearn_n_samples = args.n_samples
    if args.epochs:    cfg.relearn_epochs    = args.epochs

    # Load data (same splits as training)
    forget_samples, retain_samples = load_tofu_splits(cfg)

    if args.compare:
        # ── Side-by-side comparison ─────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  ARMOR — Relearning Attack Comparison: GA vs NPO vs NPO+SAM")
        print("=" * 70)

        results = {}
        for name, ckpt in [("GA",       args.ga_checkpoint),
                            ("NPO",      args.npo_checkpoint),
                            ("NPO+SAM",  args.sam_checkpoint)]:
            r = run_attack_on_checkpoint(
                ckpt, name, forget_samples, retain_samples, cfg,
                original_forget_acc=args.original_acc,
            )
            if r:
                results[name] = r

        # Print comparison table
        if results:
            print("\n" + "=" * 72)
            print("  COMPARISON SUMMARY")
            print("=" * 72)
            print(f"  {'Method':<12} | {'Post-unlearn':>13} | "
                  f"{'Post-attack':>12} | {'Acc Jump':>10} | {'Recovery%':>10}")
            print("  " + "-" * 68)
            for name, r in results.items():
                jump = r.final_forget_acc - r.pre_attack_forget_acc
                rec  = f"{r.recovery_pct:.1f}%" if not (isinstance(r.recovery_pct, float) and r.recovery_pct != r.recovery_pct) else "N/A*"
                print(f"  {name:<12} | {r.pre_attack_forget_acc:>13.4f} | "
                      f"{r.final_forget_acc:>12.4f} | {jump:>+10.4f} | {rec:>10}")
            print("=" * 72)
            print("  * Recovery% = N/A when post-unlearn acc >= original acc")
            print("    (model did not forget below baseline; use Acc Jump instead)")

    elif args.checkpoint:
        # ── Single checkpoint attack ────────────────────────────────────────────
        run_attack_on_checkpoint(
            args.checkpoint,
            args.method,
            forget_samples, retain_samples,
            cfg,
            original_forget_acc=args.original_acc,
        )
    else:
        print("[error] Specify --checkpoint or --compare. See --help.")
        sys.exit(1)


if __name__ == "__main__":
    main()
