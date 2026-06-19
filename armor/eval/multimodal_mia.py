"""
armor/eval/multimodal_mia.py
=============================
Module 5 — Multimodal Membership Inference Attack (MIA) Audit
           with Cross-Modal Contrastive Unlearning Loss

Problem
-------
The existing MIA in `armor/eval/mia.py` operates on text-only models using
the Min-K% token probability test. This fails for vision-language models (VLMs)
because:
  • Forget data includes (image, caption) pairs — the image modality is ignored
  • A model can "forget" the text while retaining the image → membership
    can still be inferred from visual similarity
  • Cross-modal association (image ↔ text binding) can survive text-only erasure

Solution: Visual MIA + Contrastive Unlearning Loss
---------------------------------------------------
1. VISUAL MIA (evaluation)
   ─────────────────────────
   For each forget (image, caption) pair:
     • Compute image embedding I = vision_encoder(image) — before and after
     • Compute text embedding T  = text_encoder(caption) — before and after
     • Measure cosine similarity: sim(I, T)
   A strong cosine similarity drop post-unlearning indicates successful erasure
   of the cross-modal association.

2. CONTRASTIVE UNLEARNING LOSS (training augmentation)
   ─────────────────────────────────────────────────────
   Add a SimCLR-style contrastive loss to ANY unlearner's training loop:
     L_contrastive = -log( exp(sim(I_f, T_f) / τ) / Σ_j exp(sim(I_f, T_j) / τ) )
   For forget pairs: MINIMISE similarity (push apart)
   For retain pairs: MAXIMISE similarity (pull together)

   This is added as an auxiliary loss:
     L_total = L_base_unlearn + λ_mm · L_contrastive_forget - λ_mm · L_contrastive_retain

3. AUDIT REPORT
   ─────────────
   Precision/recall of forget-set detection at multiple similarity thresholds.
   Outputs a JSON report with per-sample membership scores.

Architecture Compatibility
--------------------------
Works with:
  • LLaVA-1.5 (via `run_llava_unlearn.py`)
  • CLIP / SigLIP (standalone vision encoders)
  • Any model exposing `.vision_tower` and `.language_model` attributes

For text-only models (e.g., distilGPT2 in debug mode), the module falls back
to the standard Min-K% MIA from `mia.py` with a warning.

References
----------
  • Chen et al., "A Simple Framework for Contrastive Learning of Visual
    Representations." ICML 2020. (SimCLR)
  • Radford et al., "Learning Transferable Visual Models From Natural
    Language Supervision." ICML 2021. (CLIP)
  • Carlini et al., "Membership Inference Attacks from First Principles." 2022.
"""

import time
import json
import os
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import ARMORConfig
from .mia import MembershipInferenceAuditor   # base text-only MIA


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Modal Contrastive Unlearning Loss
# ─────────────────────────────────────────────────────────────────────────────

class ContrastiveUnlearningLoss(nn.Module):
    """
    CLIP-style contrastive loss adapted for unlearning.

    For forget (image, caption) pairs: push embeddings APART (low similarity).
    For retain (image, caption) pairs: pull embeddings TOGETHER (high similarity).

    The loss for a batch of forget pairs is:
        L_forget = mean_i [ sim(I_i, T_i) / τ ]   → MAXIMISE (positive sign → invert)

    This encourages the model to dissociate forget images from their captions.

    Usage
    -----
        loss_fn = ContrastiveUnlearningLoss(temperature=0.07)
        loss = loss_fn(image_embeds, text_embeds, mode="forget")
        loss = loss_fn(image_embeds, text_embeds, mode="retain")
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(self,
                image_embeds: torch.Tensor,
                text_embeds:  torch.Tensor,
                mode:         str = "forget") -> torch.Tensor:
        """
        Parameters
        ----------
        image_embeds : [B, D] L2-normalised image embeddings
        text_embeds  : [B, D] L2-normalised text embeddings
        mode         : "forget" → minimise sim (push apart)
                       "retain" → maximise sim (pull together)

        Returns
        -------
        scalar loss
        """
        # Cosine similarity matrix [B, B]
        sim = image_embeds @ text_embeds.T / self.tau   # [B, B]

        # Symmetric InfoNCE (CLIP loss)
        B      = sim.shape[0]
        labels = torch.arange(B, device=sim.device)

        loss_i2t = F.cross_entropy(sim,   labels)   # image→text
        loss_t2i = F.cross_entropy(sim.T, labels)   # text→image
        info_nce = (loss_i2t + loss_t2i) / 2.0

        if mode == "forget":
            # For forget pairs: maximise the loss (push apart)
            # We negate InfoNCE so gradient ascent on this term
            # pushes paired forget embeddings apart
            return -info_nce
        else:
            # For retain pairs: minimise the loss (pull together)
            return info_nce


# ─────────────────────────────────────────────────────────────────────────────
# Vision Encoder Extractor — extract image embeddings from VLM
# ─────────────────────────────────────────────────────────────────────────────

class VisionEncoderExtractor:
    """
    Extracts image embeddings from a vision-language model.

    Supports:
      • LLaVA: model.vision_tower(pixel_values) → [B, N, D]
      • CLIP:  model.vision_model(pixel_values) → pooled [B, D]
      • Generic: any model with .encode_image() or .visual()

    For text-only models, returns None.
    """

    def __init__(self, model: nn.Module):
        self.model       = model
        self._vision_enc = self._find_vision_encoder()

    def _find_vision_encoder(self) -> Optional[nn.Module]:
        """Try known attribute names for the vision tower."""
        for attr in ["vision_tower", "visual", "vision_model",
                     "image_encoder", "encode_image"]:
            if hasattr(self.model, attr):
                enc = getattr(self.model, attr)
                if callable(enc) or isinstance(enc, nn.Module):
                    return enc
        return None   # text-only model

    @property
    def is_multimodal(self) -> bool:
        return self._vision_enc is not None

    @torch.no_grad()
    def extract(self, pixel_values: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract and mean-pool image embeddings.

        Parameters
        ----------
        pixel_values : [B, C, H, W]

        Returns
        -------
        embeddings : [B, D] L2-normalised, or None if text-only
        """
        if not self.is_multimodal:
            return None

        enc = self._vision_enc
        if callable(enc) and not isinstance(enc, nn.Module):
            out = enc(pixel_values)
        else:
            out = enc(pixel_values)

        # Handle different output formats
        if isinstance(out, tuple):
            h = out[0]   # (last_hidden_state, ...)
        elif hasattr(out, "last_hidden_state"):
            h = out.last_hidden_state
        elif hasattr(out, "pooler_output"):
            h = out.pooler_output
        else:
            h = out

        # Mean pool if sequence output [B, N, D] → [B, D]
        if h.dim() == 3:
            h = h.mean(dim=1)

        return F.normalize(h.float(), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Visual Membership Test — cosine similarity based MIA
# ─────────────────────────────────────────────────────────────────────────────

class VisualMembershipTest:
    """
    Infers forget-set membership from cross-modal similarity changes.

    A sample is flagged as "still memorised" if:
        sim(I, T) > threshold   (post-unlearning)

    A good unlearner should push sim(I_forget, T_forget) below threshold.
    """

    def __init__(self,
                 model:              nn.Module,
                 cfg:                ARMORConfig,
                 similarity_threshold: float = 0.5):
        self.model     = model
        self.cfg       = cfg
        self.threshold = similarity_threshold
        self.extractor = VisionEncoderExtractor(model)

    @torch.no_grad()
    def evaluate(self,
                 loader:    DataLoader,
                 set_name:  str = "forget") -> Dict[str, Any]:
        """
        Evaluate visual membership inference on a DataLoader.

        Each batch must contain:
          • "input_ids"    : text tokens
          • "pixel_values" : preprocessed images (if available)
          • "attention_mask"

        Returns
        -------
        {
            "similarities": [float, ...],
            "membership_flags": [bool, ...],
            "mean_similarity": float,
            "membership_rate": float  # fraction flagged as members
        }
        """
        if not self.extractor.is_multimodal:
            print(f"[MM-MIA] Text-only model — visual test skipped for {set_name}")
            return {"similarities": [], "membership_flags": [],
                    "mean_similarity": 0.0, "membership_rate": 0.0}

        device = self.cfg.device
        self.model.eval()
        similarities     = []
        membership_flags = []

        for batch in tqdm(loader, desc=f"[MM-MIA] {set_name}", leave=False):
            if "pixel_values" not in batch:
                continue   # skip text-only batches

            px   = batch["pixel_values"].to(device)
            ids  = batch["input_ids"].to(device)
            mask = batch.get("attention_mask",
                             torch.ones_like(ids)).to(device)

            # Vision embedding
            img_emb = self.extractor.extract(px)   # [B, D]

            # Text embedding — mean pool hidden states from language head
            out     = self.model(input_ids=ids, attention_mask=mask,
                                 output_hidden_states=True)
            h_last  = out.hidden_states[-1]   # [B, T, D]
            txt_emb = h_last.mean(dim=1)      # [B, D]
            txt_emb = F.normalize(txt_emb.float(), dim=-1)

            # Cosine similarity per sample
            sim = (img_emb * txt_emb).sum(dim=-1).tolist()   # [B]
            similarities.extend(sim)
            membership_flags.extend([s > self.threshold for s in sim])

        mean_sim         = sum(similarities) / max(len(similarities), 1)
        membership_rate  = sum(membership_flags) / max(len(membership_flags), 1)

        print(f"[MM-MIA] {set_name.upper()} | "
              f"mean_sim={mean_sim:.4f} | "
              f"membership_rate={membership_rate:.2%} "
              f"(threshold={self.threshold})")

        return {
            "similarities":    similarities,
            "membership_flags":membership_flags,
            "mean_similarity": mean_sim,
            "membership_rate": membership_rate,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Multimodal MIA Evaluator — full audit pipeline
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalMIAEvaluator:
    """
    Full multimodal membership inference audit.

    Wraps both:
      1. Text-only MIA (from mia.py) — for language model heads
      2. Visual MIA — for cross-modal embedding similarity

    Usage
    -----
        evaluator = MultimodalMIAEvaluator(model, cfg)

        pre_scores  = evaluator.run(forget_loader, retain_loader, label="pre")
        # ... run unlearning ...
        post_scores = evaluator.run(forget_loader, retain_loader, label="post")

        report = evaluator.compare(pre_scores, post_scores)
        evaluator.save_report(report, "outputs/mm_mia_report.json")
    """

    def __init__(self, model: nn.Module, cfg: ARMORConfig, tokenizer=None):
        self.model        = model
        self.cfg          = cfg
        self.visual_test  = VisualMembershipTest(
            model, cfg,
            similarity_threshold=cfg.mm_mia_similarity_threshold)
        self.text_mia     = MembershipInferenceAuditor(model, tokenizer, cfg)

    def run(self,
            forget_loader: DataLoader,
            retain_loader: DataLoader,
            label:         str = "") -> Dict[str, Any]:
        """
        Run both text MIA and visual MIA on forget + retain sets.
        """
        print(f"\n[MM-MIA] ═══ Running MIA Audit {label} ═══")

        # Text-only MIA
        text_results_obj = self.text_mia.audit(forget_loader, retain_loader, method_name=label)
        text_results = {
            "method": text_results_obj.method,
            "auroc": text_results_obj.auroc,
            "k_percent": text_results_obj.k_percent,
            "verdict": text_results_obj.audit_verdict
        }

        # Visual MIA
        vis_forget = self.visual_test.evaluate(forget_loader, "forget")
        vis_retain = self.visual_test.evaluate(retain_loader, "retain")

        return {
            "label":       label,
            "text_mia":    text_results,
            "visual_mia": {
                "forget": vis_forget,
                "retain": vis_retain,
            },
        }

    @staticmethod
    def compare(pre: Dict[str, Any],
                post: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare pre- and post-unlearning MIA scores and generate a verdict.
        """
        def _delta(key, pre_d, post_d):
            a = pre_d.get(key, 0.0)
            b = post_d.get(key, 0.0)
            return round(a - b, 4), round(a, 4), round(b, 4)

        # Visual similarity on forget set should DROP post-unlearning
        vis_pre  = pre.get("visual_mia",  {}).get("forget", {})
        vis_post = post.get("visual_mia", {}).get("forget", {})

        sim_delta, sim_pre_val, sim_post_val = _delta(
            "mean_similarity", vis_pre, vis_post)
        mr_delta, mr_pre_val, mr_post_val    = _delta(
            "membership_rate", vis_pre, vis_post)

        report = {
            "visual_similarity_drop": {
                "pre":   sim_pre_val,
                "post":  sim_post_val,
                "delta": sim_delta,
                "verdict": "ERASED ✓" if sim_delta > 0 else "NOT ERASED ✗",
            },
            "membership_rate_drop": {
                "pre":   mr_pre_val,
                "post":  mr_post_val,
                "delta": mr_delta,
                "verdict": "REDUCED ✓" if mr_delta > 0 else "NOT REDUCED ✗",
            },
            "text_mia_pre":  pre.get("text_mia", {}),
            "text_mia_post": post.get("text_mia", {}),
            "generated_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return report

    @staticmethod
    def save_report(report: Dict[str, Any], path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[MM-MIA] Report saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Contrastive Unlearning Augmentation — mixin for any unlearner
# ─────────────────────────────────────────────────────────────────────────────

class ContrastiveUnlearningAugmentation:
    """
    Adds the cross-modal contrastive unlearning loss to any existing unlearner.

    Usage (in a training loop)
    --------------------------
        aug = ContrastiveUnlearningAugmentation(model, cfg)

        # Inside training step with multimodal batch:
        extra_loss = aug.compute(
            forget_pixel_values, forget_input_ids,
            retain_pixel_values, retain_input_ids,
        )
        total_loss = base_loss + extra_loss
    """

    def __init__(self, model: nn.Module, cfg: ARMORConfig):
        self.model     = model
        self.cfg       = cfg
        self.device    = cfg.device
        self.extractor = VisionEncoderExtractor(model)
        self.loss_fn   = ContrastiveUnlearningLoss(
            temperature=cfg.mm_mia_contrastive_temp)

    def compute(self,
                forget_pixels:   Optional[torch.Tensor],
                forget_ids:      torch.Tensor,
                retain_pixels:   Optional[torch.Tensor] = None,
                retain_ids:      Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the contrastive unlearning auxiliary loss.

        Returns 0.0 for text-only models or when pixel_values not provided.
        """
        if not self.extractor.is_multimodal or forget_pixels is None:
            return torch.tensor(0.0, device=self.device)

        λ = self.cfg.mm_mia_contrastive_coeff

        # Forget pair embeddings
        img_f   = self.extractor.extract(forget_pixels)
        mask_f  = torch.ones_like(forget_ids)
        out_f   = self.model(input_ids=forget_ids, attention_mask=mask_f,
                              output_hidden_states=True)
        txt_f   = F.normalize(out_f.hidden_states[-1].mean(dim=1).float(), dim=-1)

        # Forget contrastive: push apart
        loss_forget = λ * self.loss_fn(img_f, txt_f, mode="forget")

        # Retain contrastive: pull together (optional)
        loss_retain = torch.tensor(0.0, device=self.device)
        if retain_pixels is not None and retain_ids is not None:
            img_r  = self.extractor.extract(retain_pixels)
            mask_r = torch.ones_like(retain_ids)
            out_r  = self.model(input_ids=retain_ids, attention_mask=mask_r,
                                output_hidden_states=True)
            txt_r  = F.normalize(out_r.hidden_states[-1].mean(dim=1).float(), dim=-1)
            loss_retain = λ * self.loss_fn(img_r, txt_r, mode="retain")

        return loss_forget + loss_retain
