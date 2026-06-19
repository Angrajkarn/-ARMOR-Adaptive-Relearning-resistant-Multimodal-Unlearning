"""
scripts/run_audit_gen.py
========================
Compliance Pipeline: Run Unlearning, Collect Metrics, and Generate Signed Certificate.

Usage:
  python scripts/run_audit_gen.py --debug
"""

import argparse
import os
import sys
import warnings

# Make armor importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore", message=".*symlink.*")

import torch
from armor.config import ARMORConfig
from armor.data   import load_tofu_splits, make_dataloader
from armor.model  import get_model_and_tokenizer
from armor.unlearn.gradient_ascent import GradientAscentUnlearner
from armor.eval.zk_verify import ZKVerifier, UnlearningCommitment
from armor.eval.privacy_audit import PrivacyAuditor
from armor.attack.reconstruction import TextReconstructionAttack
from armor.eval.certificate import AuditCertificateGenerator


def parse_args():
    p = argparse.ArgumentParser(description="ARMOR — Compliance Audit Certificate Generator")
    p.add_argument("--debug",      action="store_true")
    p.add_argument("--model",      default="debug", choices=["debug", "mistral-7b", "llama2-7b"])
    p.add_argument("--qlora",      action="store_true")
    p.add_argument("--hf-token",   default=None)
    p.add_argument("--output-dir", default="outputs/audit")
    p.add_argument("--threshold-zk", type=float, default=0.01)
    p.add_argument("--probe-samples", type=int, default=16)
    p.add_argument("--signing-key",  default="ARMOR_ENTERPRISE_ROOT_KEY_2026")
    return p.parse_args()


def main():
    args = parse_args()

    # Set up stdout/stderr encoding wrapper for Windows to prevent charmap errors
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    cfg = ARMORConfig(
        debug=args.debug,
        model_key=args.model if not args.debug else "debug",
        use_qlora=args.qlora,
        hf_token=args.hf_token,
        output_dir=args.output_dir,
    )
    cfg.zk_n_probe_samples = args.probe_samples
    cfg.zk_influence_threshold = args.threshold_zk

    print("=" * 64)
    print("  ARMOR Compliance Audit & Verification Pipeline")
    print(f"  Model : {cfg.model_name}")
    print("=" * 64)

    # ── Load Model and Data ───────────────────────────────────────────────────
    forget_samples, retain_samples = load_tofu_splits(cfg)
    model, tokenizer = get_model_and_tokenizer(cfg)

    eval_forget_loader = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    eval_retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)

    # ── Pre-Unlearning Commitments ──────────────────────────────────────────
    print("\n[Audit Pipeline] Computing pre-unlearning weight and dataset hashes...")
    pre_model_hash = UnlearningCommitment.hash_model_weights(model)
    forget_set_hash = UnlearningCommitment.hash_forget_set(eval_forget_loader)
    print(f"  Pre-model weight hash : {pre_model_hash}")
    print(f"  Forget dataset hash   : {forget_set_hash}")

    # Initialize ZK Verifier
    verifier = ZKVerifier(cfg)
    # Phase 1: Pre-unlearning commitment & influence estimation
    verifier.commit_pre(model, eval_forget_loader, eval_retain_loader, method="GradientAscent")

    # ── Run Unlearning (Gradient Ascent) ──────────────────────────────────────
    print("\n[Audit Pipeline] Running Gradient Ascent unlearning...")
    forget_loader = make_dataloader(
        forget_samples, tokenizer, cfg,
        include_rephrases=cfg.use_rephrase_augmentation,
        shuffle=True
    )
    retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=True)
    unlearner = GradientAscentUnlearner(model, cfg)
    unlearner.run(forget_loader, retain_loader)

    # ── Post-Unlearning Hashes ────────────────────────────────────────────────
    print("\n[Audit Pipeline] Computing post-unlearning weight hash...")
    post_model_hash = UnlearningCommitment.hash_model_weights(model)
    print(f"  Post-model weight hash: {post_model_hash}")

    # ── Collect Metrics ───────────────────────────────────────────────────────
    
    # 1. ZK Influence Verification Report
    print("\n[Audit Pipeline] Executing ZK influence verification (Phase 2)...")
    zk_report = verifier.verify_post(model, eval_forget_loader, eval_retain_loader)

    # 2. Privacy & MIA Audit Report
    print("\n[Audit Pipeline] Executing Privacy & Membership Inference Attack (MIA) audit...")
    auditor = PrivacyAuditor(cfg, model, tokenizer)
    privacy_report = auditor.audit(
        forget_loader=eval_forget_loader,
        retain_loader=eval_retain_loader,
        method_name="GradientAscent"
    )
    privacy_report.print_summary()

    # 3. Adversarial Reconstruction Attack Report
    print("\n[Audit Pipeline] Running text reconstruction model inversion attack...")
    attacker = TextReconstructionAttack(cfg, model, tokenizer)
    reconstruction_report = attacker.run_reconstruction_attack(
        forget_samples=forget_samples[:cfg.zk_n_probe_samples],
        method_name="GradientAscent",
        threshold=0.5
    )
    reconstruction_report.print_summary()

    # ── Generate Certificate ──────────────────────────────────────────────────
    print("\n[Audit Pipeline] Compiling, signing, and generating compliance certificate...")
    cert_gen = AuditCertificateGenerator(cfg, private_signing_key=args.signing_key)
    
    certificate, html_path = cert_gen.generate_certificate(
        model_name=cfg.model_name,
        method_name="GradientAscent",
        pre_model_hash=pre_model_hash,
        post_model_hash=post_model_hash,
        forget_set_hash=forget_set_hash,
        zk_report=zk_report,
        privacy_report=privacy_report,
        reconstruction_report=reconstruction_report,
        output_dir=args.output_dir
    )

    print("\n" + "=" * 64)
    print("  COMPLIANCE CERTIFICATE GENERATION COMPLETE")
    print("=" * 64)
    print(f"  Certificate JSON : {os.path.abspath(os.path.join(args.output_dir, 'audit_certificate.json'))}")
    print(f"  Certificate HTML : {os.path.abspath(html_path)}")
    print(f"  Digital Signature: {certificate['signature']['signature_hash']}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
