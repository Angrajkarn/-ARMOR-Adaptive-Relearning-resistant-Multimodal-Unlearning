"""
armor/unlearn/npo.py
====================
Negative Preference Optimization (NPO) for LLM unlearning.

Algorithm
---------
NPO adapts DPO's contrastive training objective for unlearning.
The key intuition: we want the current model π_θ to assign LOWER
log-probability to forget-set outputs than the original reference π_ref.

Loss (per batch):
    L_NPO = -log σ( β · (log π_θ(y|x) - log π_ref(y|x)) )
          + λ · L_retain(θ)

Where:
    π_θ     = current (trainable) model
    π_ref   = frozen original model (reference)
    β       = temperature controlling sharpness of the log-ratio
    λ       = retain regularisation coefficient
    σ(·)    = sigmoid function
    y|x     = answer tokens given question context

Advantages over plain GA:
  ✓ Bounded update (sigmoid keeps the ratio well-scaled)
  ✓ Self-referential: as π_θ diverges from π_ref, loss naturally decreases
  ✓ Better preserves retain set performance (empirically)

Implementation notes:
  • Reference model shares quantized base weights when using QLoRA
    (only adapter weights differ) — see model.py:get_frozen_reference_model()
  • Token-level log-probs are averaged over non-padding positions
  • Gradient accumulation is handled the same way as GA

Cross-modal NOTE (Step 2):
  The log-prob computation is modality-agnostic — just feed multimodal
  inputs to both π_θ and π_ref and the loss formula stays identical.

References
----------
  • Zhang et al., "Negative Preference Optimization: How to Make LLMs
    Forget" (2024) [arXiv:2404.05868]
  • Rafailov et al., "Direct Preference Optimization" (2023)
"""

import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, get_linear_schedule_with_warmup
from tqdm import tqdm

from armor.config import ARMORConfig
from armor.unlearn.gradient_ascent import UnlearningResult, _infinite_iter


# ──────────────────────────────────────────────────────────────────────────────
# Log-probability computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_token_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-sample average log-probability of label tokens.

    Parameters
    ----------
    model         : The language model (π_θ or π_ref)
    input_ids     : [B, T] token IDs
    attention_mask: [B, T] attention mask
    labels        : [B, T] label IDs (-100 for positions to ignore)

    Returns
    -------
    log_probs : [B] — mean log-prob per sample, averaged over non-padding tokens
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    # outputs.logits: [B, T, V]
    logits = outputs.logits  # (B, T, V)

    # Shift: logits at position t predict token at position t+1
    shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()        # (B, T-1)

    # Per-token log-softmax
    log_probs_all = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)

    # Gather log-prob of the actual label token
    shift_labels_clamped = shift_labels.clamp(min=0)     # Avoid -100 indexing error
    token_log_probs = log_probs_all.gather(
        dim=-1, index=shift_labels_clamped.unsqueeze(-1)
    ).squeeze(-1)  # (B, T-1)

    # Mask out padding (-100 labels)
    mask = (shift_labels != -100).float()                # (B, T-1)
    # Average over non-padding positions → (B,)
    sample_log_probs = (token_log_probs * mask).sum(-1) / mask.sum(-1).clamp(min=1)

    return sample_log_probs  # (B,) — one scalar per sample


# ──────────────────────────────────────────────────────────────────────────────
# NPO Unlearner
# ──────────────────────────────────────────────────────────────────────────────

class NPOUnlearner:
    """
    Implements Negative Preference Optimization unlearning.

    The reference model (π_ref) is a frozen copy of the original model.
    NPO pushes the trainable model away from the forget set by minimising
    the log-ratio loss, while a retain term prevents catastrophic forgetting.

    Usage
    -----
    ref_model = get_frozen_reference_model(model, cfg)
    unlearner = NPOUnlearner(model, ref_model, cfg)
    result    = unlearner.run(forget_loader, retain_loader)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        cfg: ARMORConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model     = model
        self.ref_model = ref_model
        self.cfg       = cfg
        self.device    = cfg.device

        # Ensure reference model is fully frozen + eval
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()

        if optimizer is None:
            self.optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.unlearn_lr,
                weight_decay=cfg.weight_decay,
            )
        else:
            self.optimizer = optimizer

    # ── Loss computation ───────────────────────────────────────────────────────

    def _npo_forget_loss(self, batch: dict) -> torch.Tensor:
        """
        Compute the NPO contrastive loss on the forget batch.

            L_NPO = -mean[ log σ( β · (log π_θ(y|x) - log π_ref(y|x)) ) ]

        Returns
        -------
        loss : scalar tensor
        """
        input_ids = batch["input_ids"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)
        labels    = batch["labels"].to(self.device)

        # Current model log-probs (trainable)
        policy_log_probs = compute_token_log_probs(
            self.model, input_ids, attn_mask, labels
        )

        # Reference model log-probs (frozen, no grad)
        with torch.no_grad():
            ref_log_probs = compute_token_log_probs(
                self.ref_model, input_ids, attn_mask, labels
            )

        # Log-ratio: how much more/less probable is the forget data?
        log_ratio = policy_log_probs - ref_log_probs  # (B,)

        # NPO objective: push log_ratio negative (policy < reference)
        # -log σ(β · log_ratio) — minimise this = make log_ratio negative
        npo_loss = -F.logsigmoid(self.cfg.npo_beta * log_ratio).mean()

        return npo_loss

    def _retain_loss(self, batch: dict) -> torch.Tensor:
        """Standard cross-entropy on retain batch to prevent forgetting."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs.loss

    # ── Main training loop ─────────────────────────────────────────────────────

    def run(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        scheduler=None,
    ) -> UnlearningResult:
        """
        Run the full NPO unlearning loop.

        Parameters
        ----------
        forget_loader : DataLoader — forget set (optionally with rephrases)
        retain_loader : DataLoader — retain set for regularisation
        scheduler     : Optional LR scheduler

        Returns
        -------
        UnlearningResult with loss history
        """
        cfg   = self.cfg
        model = self.model
        model.train()

        retain_iter = _infinite_iter(retain_loader) if retain_loader else None

        # LR scheduler
        if scheduler is None and retain_loader is not None:
            total_steps  = len(forget_loader) * cfg.unlearn_epochs
            warmup_steps = max(1, total_steps // 10)
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

        epoch_losses, forget_losses, retain_losses = [], [], []
        total_steps = 0
        t0 = time.time()

        for epoch in range(cfg.unlearn_epochs):
            epoch_total = 0.0
            epoch_forget = 0.0
            epoch_retain = 0.0
            n_batches = 0

            pbar = tqdm(forget_loader,
                        desc=f"[NPO] Epoch {epoch+1}/{cfg.unlearn_epochs}",
                        leave=False)

            for step, forget_batch in enumerate(pbar):
                retain_batch = next(retain_iter) if retain_iter else None

                # ── NPO forget loss ────────────────────────────────────────────
                npo_loss = self._npo_forget_loss(forget_batch)

                # ── Retain loss (optional) ────────────────────────────────────
                if retain_batch is not None:
                    r_loss  = self._retain_loss(retain_batch)
                    total_loss = npo_loss + cfg.npo_retain_coeff * r_loss
                else:
                    r_loss = torch.tensor(0.0)
                    total_loss = npo_loss

                # ── Backward + gradient accumulation ──────────────────────────
                scaled = total_loss / cfg.gradient_accumulation_steps
                scaled.backward()

                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    self.optimizer.step()
                    if scheduler:
                        scheduler.step()
                    self.optimizer.zero_grad()
                    total_steps += 1

                epoch_total  += total_loss.item()
                epoch_forget += npo_loss.item()
                epoch_retain += r_loss.item() if hasattr(r_loss, 'item') else 0.0
                n_batches    += 1

                pbar.set_postfix({
                    "npo↓":    f"{npo_loss.item():.3f}",
                    "retain↓": f"{r_loss.item():.3f}" if hasattr(r_loss, 'item') else "N/A",
                    "total":   f"{total_loss.item():.3f}",
                })

            avg_total  = epoch_total  / max(n_batches, 1)
            avg_forget = epoch_forget / max(n_batches, 1)
            avg_retain = epoch_retain / max(n_batches, 1)

            epoch_losses.append((epoch + 1, avg_total))
            forget_losses.append((epoch + 1, avg_forget))
            retain_losses.append((epoch + 1, avg_retain))

            print(f"[NPO] Epoch {epoch+1:02d} | "
                  f"npo_loss={avg_forget:.4f} | "
                  f"retain_loss={avg_retain:.4f} | "
                  f"total={avg_total:.4f}")

        elapsed = time.time() - t0
        print(f"[NPO] Training complete in {elapsed:.1f}s ({total_steps} steps)")

        return UnlearningResult(
            method="NPO",
            epoch_losses=epoch_losses,
            forget_losses=forget_losses,
            retain_losses=retain_losses,
            total_steps=total_steps,
            elapsed_sec=elapsed,
        )
