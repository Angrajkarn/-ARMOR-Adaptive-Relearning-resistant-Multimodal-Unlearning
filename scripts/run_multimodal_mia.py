"""
scripts/run_multimodal_mia.py
=============================
Run Multimodal MIA Audit and Contrastive Unlearning on TOFU.

Usage
-----
  python scripts/run_multimodal_mia.py --debug
  python scripts/run_multimodal_mia.py --model mistral-7b --qlora
"""

import argparse
import os
import sys
import warnings

# Make armor importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore", message=".*symlink.*")

import torch
from armor.config import ARMORConfig
from armor.data   import load_tofu_splits, make_dataloader
from armor.model  import get_model_and_tokenizer
from armor.unlearn.gradient_ascent import GradientAscentUnlearner
from armor.eval.multimodal_mia import MultimodalMIAEvaluator, ContrastiveUnlearningAugmentation


def parse_args():
    p = argparse.ArgumentParser(description="ARMOR — Multimodal MIA Audit")
    p.add_argument("--debug",      action="store_true")
    p.add_argument("--model",      default="debug", choices=["debug","mistral-7b","llama2-7b"])
    p.add_argument("--qlora",      action="store_true")
    p.add_argument("--hf-token",   default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--output-dir", default="outputs/multimodal_mia")
    p.add_argument("--run-mia",    action="store_true", default=True)
    p.add_argument("--no-save",    action="store_true")
    p.add_argument("--contrastive-temp",  type=float, default=0.07)
    p.add_argument("--contrastive-coeff", type=float, default=1.0)
    p.add_argument("--sim-threshold",     type=float, default=0.50)
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
    if args.epochs: cfg.unlearn_epochs = args.epochs
    if args.lr:     cfg.unlearn_lr     = args.lr
    cfg.mm_mia_contrastive_temp = args.contrastive_temp
    cfg.mm_mia_contrastive_coeff = args.contrastive_coeff
    cfg.mm_mia_similarity_threshold = args.sim_threshold

    print("=" * 60)
    print(f"  ARMOR — Multimodal MIA Audit & Contrastive Unlearning")
    print(f"  Model       : {cfg.model_name}")
    print(f"  Temp        : {cfg.mm_mia_contrastive_temp}")
    print(f"  Coeff       : {cfg.mm_mia_contrastive_coeff}")
    print(f"  Threshold   : {cfg.mm_mia_similarity_threshold}")
    print("=" * 60)

    # ── Load Model and Data ───────────────────────────────────────────────────
    forget_samples, retain_samples = load_tofu_splits(cfg)
    model, tokenizer = get_model_and_tokenizer(cfg)

    eval_forget_loader = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    eval_retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)

    # ── Initialize Multimodal MIA Auditor ─────────────────────────────────────
    auditor = MultimodalMIAEvaluator(model, cfg, tokenizer)

    # Pre-unlearning audit
    pre_scores = auditor.run(eval_forget_loader, eval_retain_loader, label="Pre-unlearning")

    # ── Run Unlearning with Contrastive Loss Augmentation ─────────────────────
    # (Since this is a mixin/augmentation, we can augment standard Gradient Ascent training loop)
    print("\n[main] Starting unlearning with Contrastive Unlearning Loss...")
    forget_loader = make_dataloader(
        forget_samples, tokenizer, cfg,
        include_rephrases=cfg.use_rephrase_augmentation,
        shuffle=True
    )
    retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=True)

    # Instantiate the contrastive augmentation
    mm_aug = ContrastiveUnlearningAugmentation(model, cfg)

    # Simple training loop incorporating the contrastive loss
    device = cfg.device
    model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.unlearn_lr,
        weight_decay=cfg.weight_decay
    )

    from tqdm import tqdm
    from armor.unlearn.gradient_ascent import _infinite_iter
    retain_iter = _infinite_iter(retain_loader)

    for epoch in range(1, cfg.unlearn_epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(forget_loader, desc=f"Training Epoch {epoch}/{cfg.unlearn_epochs}", leave=False)
        for f_batch in pbar:
            # Forget base GA loss
            f_ids    = f_batch["input_ids"].to(device)
            f_mask   = f_batch.get("attention_mask", torch.ones_like(f_ids)).to(device)
            f_labels = f_batch.get("labels", f_ids).to(device)
            out_f    = model(input_ids=f_ids, attention_mask=f_mask, labels=f_labels)
            base_loss = -cfg.ga_forget_coeff * out_f.loss

            # Retain base CE loss
            r_batch  = next(retain_iter)
            r_ids    = r_batch["input_ids"].to(device)
            r_mask   = r_batch.get("attention_mask", torch.ones_like(r_ids)).to(device)
            r_labels = r_batch.get("labels", r_ids).to(device)
            out_r    = model(input_ids=r_ids, attention_mask=r_mask, labels=r_labels)
            retain_loss = cfg.ga_retain_coeff * out_r.loss

            # Contrastive unlearning loss
            f_pixels = f_batch.get("pixel_values", None)
            r_pixels = r_batch.get("pixel_values", None)
            contrastive_loss = mm_aug.compute(
                forget_pixels=f_pixels, forget_ids=f_ids,
                retain_pixels=r_pixels, retain_ids=r_ids
            )

            total_loss = base_loss + retain_loss + contrastive_loss
            optimizer.zero_grad()
            (total_loss / cfg.gradient_accumulation_steps).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            epoch_loss += total_loss.item()
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}", "mm_loss": f"{contrastive_loss.item():.4f}"})
        print(f"[main] Epoch {epoch:02d} | loss={epoch_loss / len(forget_loader):.4f}")

    # Post-unlearning audit
    post_scores = auditor.run(eval_forget_loader, eval_retain_loader, label="Post-unlearning")

    # Compare pre and post
    report = auditor.compare(pre_scores, post_scores)
    report_path = os.path.join(args.output_dir, "mm_mia_audit_report.json")
    auditor.save_report(report, report_path)


if __name__ == "__main__":
    main()
