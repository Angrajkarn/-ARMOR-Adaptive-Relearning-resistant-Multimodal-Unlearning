"""
scripts/run_temporal_unlearn.py
================================
TKDU: Temporal Knowledge Decay Unlearning Runner
=================================================

Runs the full TKDU pipeline:
  1. Create a knowledge registry with temporal validity metadata
  2. Display the unlearning schedule (which facts expire when)
  3. Run temporally-weighted unlearning
  4. Generate a GDPR compliance certificate
  5. Save reports and certificates

Usage
-----
  # Debug mode (CPU, distilgpt2, demo timestamps)
  python scripts/run_temporal_unlearn.py --debug

  # With custom half-life
  python scripts/run_temporal_unlearn.py --debug --halflife-days 7

  # Load knowledge registry from JSON
  python scripts/run_temporal_unlearn.py --debug --knowledge-registry path/to/registry.json

  # Full GPU run
  python scripts/run_temporal_unlearn.py --model mistral-7b --qlora --halflife-days 30

  # Print schedule only (no training)
  python scripts/run_temporal_unlearn.py --debug --schedule-only
"""

import argparse
import json
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
from armor.model  import get_model_and_tokenizer, get_frozen_reference_model
from armor.unlearn.temporal_decay import (
    TKDUUnlearner,
    TemporalUnlearningScheduler,
    TemporalValidityScorer,
    create_demo_knowledge_registry,
    load_knowledge_registry_from_json,
    save_knowledge_registry_to_json,
)
from armor.eval.temporal_certificate import TemporalCertificateGenerator
from armor.eval.metrics import UnlearningEvaluator


# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="TKDU: Temporal Knowledge Decay Unlearning"
    )
    p.add_argument("--debug",              action="store_true")
    p.add_argument("--model",              default="debug")
    p.add_argument("--qlora",              action="store_true")
    p.add_argument("--halflife-days",      type=float, default=30.0,
                   help="Half-life for temporal decay (days, default: 30)")
    p.add_argument("--expired-threshold",  type=float, default=0.1,
                   help="Validity score below which a fact is expired")
    p.add_argument("--expiry-buffer-days", type=float, default=0.0,
                   help="Trigger unlearning this many days before expiry")
    p.add_argument("--knowledge-registry", default=None,
                   help="Path to JSON knowledge registry file")
    p.add_argument("--schedule-only",      action="store_true",
                   help="Print schedule and exit without training")
    p.add_argument("--save-registry",      action="store_true",
                   help="Save the generated knowledge registry to JSON")
    p.add_argument("--no-rouge",           action="store_true")
    p.add_argument("--no-save",            action="store_true")
    p.add_argument("--output-dir",         default="outputs/temporal")
    p.add_argument("--data-controller",    default="ARMOR Framework")
    p.add_argument("--dpo-contact",        default="dpo@armor-framework.ai")
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
    print("  TKDU: TEMPORAL KNOWLEDGE DECAY UNLEARNING")
    print("=" * 72)
    print(f"  Model           : {cfg.model_name}")
    print(f"  Device          : {cfg.device}")
    print(f"  Half-life (days): {args.halflife_days}")
    print(f"  Debug           : {cfg.debug}")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    print("[TKDU] Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(cfg)
    ref_model        = get_frozen_reference_model(model, cfg)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[TKDU] Loading TOFU dataset...")
    forget_samples, retain_samples = load_tofu_splits(cfg)

    # Extract (Q, A) pairs for knowledge registry
    n_forget = len(forget_samples)
    qa_pairs = [(s.question, s.answer) for s in forget_samples]
    print(f"[TKDU] Loaded {len(qa_pairs)} forget-set (Q,A) pairs")

    # ── Build knowledge registry ───────────────────────────────────────────────
    if args.knowledge_registry and os.path.isfile(args.knowledge_registry):
        print(f"[TKDU] Loading knowledge registry: {args.knowledge_registry}")
        knowledge_items = load_knowledge_registry_from_json(args.knowledge_registry)
        knowledge_items = knowledge_items[:n_forget]  # align with forget set
    else:
        print("[TKDU] Creating demo knowledge registry with synthetic timestamps...")
        knowledge_items = create_demo_knowledge_registry(qa_pairs)

    # Score all items
    scorer = TemporalValidityScorer(
        halflife_days=args.halflife_days,
        expired_threshold=args.expired_threshold,
    )
    scorer.score_all(knowledge_items)

    n_expired = sum(1 for k in knowledge_items if k.is_expired)
    n_near    = sum(1 for k in knowledge_items
                   if not k.is_expired and k.current_validity < 0.5)
    n_valid   = len(knowledge_items) - n_expired - n_near

    print(f"[TKDU] Registry: total={len(knowledge_items)} | "
          f"expired={n_expired} | near_expiry={n_near} | valid={n_valid}")

    # Optionally save registry
    if args.save_registry and not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        reg_path = os.path.join(args.output_dir, "knowledge_registry.json")
        save_knowledge_registry_to_json(knowledge_items, reg_path)

    # ── Display schedule ──────────────────────────────────────────────────────
    scheduler = TemporalUnlearningScheduler(
        knowledge_registry=knowledge_items,
        check_interval_days=1.0,
        expiry_buffer_days=args.expiry_buffer_days,
        halflife_days=args.halflife_days,
    )
    scheduler.print_schedule()
    due_items = scheduler.get_due_for_unlearning()
    print(f"[TKDU] Items due for unlearning now: {len(due_items)}")

    if args.schedule_only:
        print("[TKDU] Schedule-only mode — exiting without training.")
        sys.exit(0)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    forget_loader = make_dataloader(
        forget_samples, tokenizer, cfg,
        include_rephrases=True, shuffle=True,
    )
    retain_loader = make_dataloader(
        retain_samples, tokenizer, cfg,
        shuffle=True,
    )

    # ── Run TKDU unlearning ───────────────────────────────────────────────────
    print("\n[TKDU] === STARTING TEMPORAL UNLEARNING ===")
    unlearner = TKDUUnlearner(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        cfg=cfg,
        knowledge_items=knowledge_items,
        halflife_days=args.halflife_days,
        expired_threshold=args.expired_threshold,
    )

    result = unlearner.run(forget_loader, retain_loader)
    scheduler.mark_triggered(due_items)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n[TKDU] === EVALUATION ===")
    eval_forget_loader = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    eval_retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)

    evaluator = UnlearningEvaluator(model, tokenizer, cfg)
    eval_result = evaluator.evaluate(
        forget_samples=forget_samples,
        retain_samples=retain_samples,
        forget_loader=eval_forget_loader,
        retain_loader=eval_retain_loader,
        method_name="TKDU",
        run_rouge=not args.no_rouge,
    )
    eval_result.print_table()

    # ── Generate compliance certificate ───────────────────────────────────────
    print("\n[TKDU] === GENERATING GDPR COMPLIANCE CERTIFICATE ===")
    cert_gen = TemporalCertificateGenerator()
    cert = cert_gen.generate(
        unlearning_result=result,
        knowledge_items=knowledge_items,
        data_controller=args.data_controller,
        dpo_contact=args.dpo_contact,
    )

    # ── Save all outputs ──────────────────────────────────────────────────────
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        ts = int(time.time())

        # Certificate
        cert_path = os.path.join(args.output_dir, f"temporal_cert_{ts}.json")
        cert_gen.save(cert, cert_path, save_html=True)

        # Unlearning result
        result_path = os.path.join(args.output_dir, f"tkdu_result_{ts}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"[TKDU] Result saved -> {result_path}")

        print(f"\n[TKDU] All outputs saved to: {args.output_dir}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  TKDU COMPLETE")
    print("=" * 72)
    print(f"  Expired items unlearned  : {result.n_expired}")
    print(f"  Mean validity score      : {result.mean_validity_score:.4f}")
    gdpr_ok = "YES" if cert.retention_period_compliant else "NO"
    print(f"  GDPR Compliant           : {gdpr_ok}")
    print(f"  Certificate ID           : {cert.certificate_id}")
    print("=" * 72 + "\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
