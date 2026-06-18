"""
scripts/run_llava_unlearn.py
=============================
Cross-Modal Unlearning: NPO+SAM extended to LLaVA (vision + language).

In --text-only / --debug mode, runs on standard TOFU text data with distilgpt2.
In full mode, wraps text batches with synthetic pixel_values to test the
cross-modal forward pass architecture (replace with real VQA data for research).

Usage:
    python scripts/run_llava_unlearn.py --debug --text-only --no-rouge
    python scripts/run_llava_unlearn.py --model mistral-7b --qlora --text-only
"""

import argparse, os, sys, warnings, copy
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore", message=".*symlink.*")
warnings.filterwarnings("ignore", message=".*Xet.*")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from armor.config import ARMORConfig
from armor.data   import load_tofu_splits, make_dataloader
from armor.model  import get_model_and_tokenizer, save_checkpoint, get_frozen_reference_model
from armor.eval.metrics import UnlearningEvaluator
from armor.eval.mia     import MembershipInferenceAuditor
from armor.eval.privacy_audit import PrivacyAuditor


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Multimodal Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalWrapperDataset(Dataset):
    """
    Wraps any Dataset and injects random pixel_values for architecture testing.
    In production, replace with real image-text pairs from a VQA dataset.
    """
    def __init__(self, base_dataset, image_size: int = 336):
        self.base       = base_dataset
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx) -> dict:
        sample = dict(self.base[idx])
        sample["pixel_values"] = torch.randn(3, self.image_size, self.image_size)
        return sample


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Modal NPO + SAM Unlearner
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalNPOSAMUnlearner:
    """
    Extends NPO+SAM to multimodal forward passes.
    pixel_values are forwarded if present in batch (cross-modal mode).
    """

    def __init__(self, cfg, model, ref_model, tokenizer,
                 sam_rho=0.05, beta_npo=0.1, beta_retain=1.0,
                 multimodal=False):
        self.cfg         = cfg
        self.model       = model
        self.ref_model   = ref_model
        self.tokenizer   = tokenizer
        self.sam_rho     = sam_rho
        self.beta_npo    = beta_npo
        self.beta_retain = beta_retain
        self.multimodal  = multimodal

    def _forward(self, model, batch):
        ids  = batch["input_ids"].to(self.cfg.device)
        labs = batch["labels"].to(self.cfg.device)
        mask = batch.get("attention_mask",
                         torch.ones_like(ids)).to(self.cfg.device)
        kwargs = dict(input_ids=ids, attention_mask=mask, labels=labs)
        if self.multimodal and "pixel_values" in batch:
            kwargs["pixel_values"] = batch["pixel_values"].to(self.cfg.device)
        return model(**kwargs)

    def _npo_loss(self, f_batch):
        out_cur = self._forward(self.model, f_batch)
        with torch.no_grad():
            out_ref = self._forward(self.ref_model, f_batch)
        log_ratio = out_cur.loss - out_ref.loss
        return -F.logsigmoid(-self.beta_npo * log_ratio).mean()

    def _sam_perturb(self):
        norms = [p.grad.norm().cpu() for p in self.model.parameters()
                 if p.grad is not None]
        if not norms:
            return
        g_norm = torch.norm(torch.stack(norms)).item()
        scale  = self.sam_rho / (g_norm + 1e-12)
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    e = scale * p.grad
                    p.data.add_(e)
                    p._sam_e = e

    def _sam_restore(self):
        with torch.no_grad():
            for p in self.model.parameters():
                if hasattr(p, "_sam_e"):
                    p.data.sub_(p._sam_e)
                    del p._sam_e

    def run(self, forget_loader, retain_loader):
        from tqdm import tqdm
        import time

        self.model.train()
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.unlearn_lr)

        mode = "cross-modal" if self.multimodal else "text-only"
        print(f"\n[LLaVA-NPO+SAM] Mode: {mode} | "
              f"rho={self.sam_rho} | beta_npo={self.beta_npo}")

        for epoch in range(1, self.cfg.unlearn_epochs + 1):
            ep_npo = ep_ret = 0.0
            n = 0
            pbar = tqdm(zip(forget_loader, retain_loader),
                        total=min(len(forget_loader), len(retain_loader)),
                        desc=f"[LLaVA] Epoch {epoch}/{self.cfg.unlearn_epochs}")

            for f_batch, r_batch in pbar:
                # SAM Pass 1
                optimizer.zero_grad()
                npo  = self._npo_loss(f_batch)
                ret1 = self._forward(self.model, r_batch)
                (npo + self.beta_retain * ret1.loss).backward()
                self._sam_perturb()

                # SAM Pass 2
                optimizer.zero_grad()
                npo2 = self._npo_loss(f_batch)
                ret2 = self._forward(self.model, r_batch)
                (npo2 + self.beta_retain * ret2.loss).backward()
                self._sam_restore()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                ep_npo += npo.item()
                ep_ret += ret2.loss.item()
                n += 1
                pbar.set_postfix(npo=f"{npo.item():.3f}",
                                  ret=f"{ret2.loss.item():.3f}")

            s = max(n, 1)
            print(f"[LLaVA] Epoch {epoch:02d} [{mode}] | "
                  f"npo={ep_npo/s:.4f} | retain={ep_ret/s:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ARMOR: LLaVA Cross-Modal Unlearning")
    p.add_argument("--debug",      action="store_true")
    p.add_argument("--model",      default="debug",
                   choices=["debug", "mistral-7b", "llama2-7b"])
    p.add_argument("--qlora",      action="store_true")
    p.add_argument("--hf-token",   default=None)
    p.add_argument("--no-rouge",   action="store_true")
    p.add_argument("--run-mia",    action="store_true")
    p.add_argument("--text-only",  action="store_true",
                   help="Skip pixel_values injection (text-only mode)")
    p.add_argument("--sam-rho",    type=float, default=0.05)
    p.add_argument("--beta-npo",   type=float, default=0.1)
    p.add_argument("--image-size", type=int,   default=336)
    p.add_argument("--output-dir", default="outputs/llava_npo_sam")
    p.add_argument("--no-save",    action="store_true",
                   help="Skip saving checkpoint (for smoke tests)")
    return p.parse_args()


def main():
    args = parse_args()
    # In debug mode always use text-only (no vision encoder)
    multimodal = not args.text_only and not args.debug

    cfg = ARMORConfig(
        debug      = args.debug,
        model_key  = "debug" if args.debug else args.model,
        use_qlora  = args.qlora,
        hf_token   = args.hf_token,
        output_dir = args.output_dir,
    )

    print("=" * 62)
    print("  ARMOR -- LLaVA Cross-Modal NPO+SAM Unlearning")
    print("=" * 62)
    mode_str = "cross-modal (text+vision)" if multimodal else "text-only"
    print(f"  Model  : {cfg.model_name}  |  Mode: {mode_str}")
    print(f"  SAM rho: {args.sam_rho}  |  Beta-NPO: {args.beta_npo}")
    print("=" * 62)

    forget_samples, retain_samples = load_tofu_splits(cfg)
    model, tokenizer               = get_model_and_tokenizer(cfg)
    ref_model                      = get_frozen_reference_model(model, cfg)

    eval_fl = make_dataloader(forget_samples, tokenizer, cfg, shuffle=False)
    eval_rl = make_dataloader(retain_samples, tokenizer, cfg, shuffle=False)

    forget_loader = make_dataloader(
        forget_samples, tokenizer, cfg,
        include_rephrases=cfg.use_rephrase_augmentation, shuffle=True)
    retain_loader = make_dataloader(retain_samples, tokenizer, cfg, shuffle=True)

    # Optionally wrap with synthetic pixel_values
    if multimodal:
        forget_loader = DataLoader(
            MultimodalWrapperDataset(forget_loader.dataset, args.image_size),
            batch_size=cfg.batch_size, shuffle=True)
        retain_loader = DataLoader(
            MultimodalWrapperDataset(retain_loader.dataset, args.image_size),
            batch_size=cfg.batch_size, shuffle=True)
        print(f"[llava] Injected synthetic pixel_values "
              f"(3x{args.image_size}x{args.image_size})")

    evaluator  = UnlearningEvaluator(model, tokenizer, cfg)
    pre_result = evaluator.evaluate(
        forget_samples, retain_samples, eval_fl, eval_rl,
        method_name="Pre-unlearn",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50)
    pre_result.print_table()

    unlearner = CrossModalNPOSAMUnlearner(
        cfg        = cfg,
        model      = model,
        ref_model  = ref_model,
        tokenizer  = tokenizer,
        sam_rho    = args.sam_rho,
        beta_npo   = args.beta_npo,
        multimodal = multimodal)
    unlearner.run(forget_loader, retain_loader)

    post_result = evaluator.evaluate(
        forget_samples, retain_samples, eval_fl, eval_rl,
        method_name="LLaVA-NPO+SAM",
        run_rouge=not args.no_rouge,
        max_rouge_samples=20 if cfg.debug else 50)
    post_result.print_table()

    if args.run_mia:
        PrivacyAuditor(cfg, model, tokenizer).audit(
            eval_fl, eval_rl, method_name="LLaVA-NPO+SAM").print_summary()

    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt = os.path.join(args.output_dir, "llava_unlearned")
        save_checkpoint(model, tokenizer, ckpt, cfg)
        print(f"\n[done] Saved -> {ckpt}")
    else:
        print("\n[done] (--no-save: checkpoint skipped)")
    print(f"  forget_quality : {post_result.forget_quality:.4f}")
    print(f"  retain_accuracy: {post_result.retain_accuracy:.4f}")


if __name__ == "__main__":
    main()
