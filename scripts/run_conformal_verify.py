"""
scripts/run_conformal_verify.py
================================
CU-AR: Conformal Unlearning Verification Runner
================================================

Applies conformal prediction to verify unlearning quality for an
already-unlearned model, giving distribution-free statistical guarantees.

Usage
-----
  # Debug mode (CPU, distilgpt2)
  python scripts/run_conformal_verify.py --debug

  # Custom alpha
  python scripts/run_conformal_verify.py --debug --alpha 0.10

  # Full GPU run after unlearning
  python scripts/run_conformal_verify.py --model mistral-7b --qlora --alpha 0.05

  # Save HTML report
  python scripts/run_conformal_verify.py --debug --save-html
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
from armor.data   import load_tofu_splits, make_dataloader
from armor.model  import get_model_and_tokenizer
from armor.eval.conformal_verify import ConformalUnlearningVerifier


# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CU-AR: Conformal Unlearning Verification"
    )
    p.add_argument("--debug",    action="store_true", help="CPU debug mode with distilgpt2")
    p.add_argument("--model",    default="debug",     help="Model key (debug / mistral-7b)")
    p.add_argument("--qlora",    action="store_true", help="Use 4-bit QLoRA (GPU)")
    p.add_argument("--alpha",    type=float, default=0.05,
                   help="Miscoverage rate alpha (default: 0.05 = 5%%)")
    p.add_argument("--retain-check-n", type=int, default=50,
                   help="Number of retain samples for sanity coverage check")
    p.add_argument("--save-html",  action="store_true", help="Save HTML report")
    p.add_argument("--no-save",    action="store_true", help="Skip saving outputs")
    p.add_argument("--output-dir", default="outputs/conformal")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    cfg = ARMORConfig(
        debug=args.debug,
        model_key="debug" if args.debug else args.model,
        use_qlora=args.qlora,
    )

    print("\n" + "=" * 72)
    print("  CU-AR: CONFORMAL UNLEARNING VERIFICATION")
    print("=" * 72)
    print(f"  Model     : {cfg.model_name}")
    print(f"  Device    : {cfg.device}")
    print(f"  Alpha (a) : {args.alpha}")
    print(f"  Debug     : {cfg.debug}")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    print("[CU-AR] Loading model...")
    model, tokenizer = get_model_and_tokenizer(cfg)
    model.eval()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[CU-AR] Loading TOFU dataset...")
    forget_samples, retain_samples = load_tofu_splits(cfg)

    forget_loader = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)

    # ── Run conformal verification ────────────────────────────────────────────
    verifier = ConformalUnlearningVerifier(model, tokenizer, cfg)

    report = verifier.verify(
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        method_name=f"CU-AR ({cfg.model_key})",
        alpha=args.alpha,
        retain_check_n=min(args.retain_check_n, len(retain_samples)),
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        ts        = int(time.time())
        json_path = os.path.join(args.output_dir, f"conformal_report_{ts}.json")
        verifier.save_report(report, json_path, save_html=True)

    # ── Exit ──────────────────────────────────────────────────────────────────
    verdict = "CERTIFIED" if report.unlearning_certified else "NOT CERTIFIED"
    print(f"\n[CU-AR] Result: {verdict}")
    print(f"[CU-AR] Forget coverage: {report.forget_coverage_rate:.4f} "
          f"(alpha={args.alpha:.3f})")
    sys.exit(0)


if __name__ == "__main__":
    main()
