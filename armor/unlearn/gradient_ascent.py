"""
armor/unlearn/gradient_ascent.py
=================================
Standard Gradient Ascent (GA) unlearning for LLMs.

Algorithm
---------
For each training step:

    loss = -α * L_forget(θ) + β * L_retain(θ)

Where:
    L_forget = cross-entropy loss on forget set (we MAXIMISE this via negation)
    L_retain = cross-entropy loss on retain set (we MINIMISE this normally)
    α, β     = cfg.ga_forget_coeff, cfg.ga_retain_coeff

The forget loss is negated (ascent) to push model weights away from
memorising the forget set. The retain term prevents catastrophic forgetting.

Rephrase-invariant extension:
  If forget DataLoader was built with include_rephrases=True, the ascent
  gradient averages over multiple phrasings of each forget question,
  making unlearning robust to prompt-rephrasing attacks.

Cross-modal NOTE (Step 2):
  Replace model(**batch) with a multimodal forward pass that processes
  both image and text inputs. The loss computation stays identical.

References
----------
  • Yao et al., "Large Language Model Unlearning" (2023)
  • Maini et al., "TOFU: A Task of Fictitious Unlearning" (2024)
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, get_linear_schedule_with_warmup
from tqdm import tqdm

from armor.config import ARMORConfig


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class UnlearningResult:
    """Stores per-epoch training statistics for later analysis."""
    method: str
    epoch_losses: list      # List of (epoch, avg_total_loss)
    forget_losses: list     # List of (epoch, avg_forget_loss)
    retain_losses: list     # List of (epoch, avg_retain_loss)
    total_steps: int
    elapsed_sec: float


# ──────────────────────────────────────────────────────────────────────────────
# Gradient Ascent Unlearner
# ──────────────────────────────────────────────────────────────────────────────

class GradientAscentUnlearner:
    """
    Implements gradient ascent unlearning with retain-set regularisation.

    Usage
    -----
    unlearner = GradientAscentUnlearner(model, cfg)
    result    = unlearner.run(forget_loader, retain_loader)
    """

    def __init__(self, model: PreTrainedModel, cfg: ARMORConfig,
                 optimizer: Optional[torch.optim.Optimizer] = None):
        self.model  = model
        self.cfg    = cfg
        self.device = cfg.device

        # Allow injecting a custom optimizer (e.g. SAMOptimizer from sam_wrapper)
        if optimizer is None:
            self.optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.unlearn_lr,
                weight_decay=cfg.weight_decay,
            )
        else:
            self.optimizer = optimizer

    # ── Forward pass helpers ───────────────────────────────────────────────────

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """Standard cross-entropy forward pass. Moves batch to device."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs.loss

    def _ga_loss(self, forget_batch: dict,
                 retain_batch: Optional[dict]) -> torch.Tensor:
        """
        Compute the combined GA objective:
            total_loss = -α * forget_loss + β * retain_loss

        The negative sign on forget_loss converts minimisation to maximisation
        (gradient *ascent* on the forget set).
        """
        forget_loss = self._compute_loss(forget_batch)

        # Negate → gradient ascent on forget set
        total_loss = -self.cfg.ga_forget_coeff * forget_loss

        if retain_batch is not None:
            retain_loss = self._compute_loss(retain_batch)
            total_loss += self.cfg.ga_retain_coeff * retain_loss
        else:
            retain_loss = torch.tensor(0.0)

        return total_loss, forget_loss.detach(), retain_loss.detach()

    # ── Main training loop ─────────────────────────────────────────────────────

    def run(self,
            forget_loader: DataLoader,
            retain_loader: Optional[DataLoader] = None,
            scheduler=None) -> UnlearningResult:
        """
        Run the full gradient ascent unlearning loop.

        Parameters
        ----------
        forget_loader : DataLoader for the forget set (optionally with rephrases)
        retain_loader : DataLoader for the retain set (None = no retain term)
        scheduler     : Optional LR scheduler

        Returns
        -------
        UnlearningResult with per-epoch loss history
        """
        cfg   = self.cfg
        model = self.model
        model.train()

        # Create an infinite retain iterator (retain set may be much larger)
        retain_iter = _infinite_iter(retain_loader) if retain_loader else None

        # ── Optional LR scheduler ──────────────────────────────────────────────
        if scheduler is None and retain_loader is not None:
            total_steps = len(forget_loader) * cfg.unlearn_epochs
            warmup_steps = max(1, total_steps // 10)
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

        # ── Results tracking ───────────────────────────────────────────────────
        epoch_losses, forget_losses, retain_losses = [], [], []
        total_steps = 0
        t0 = time.time()

        for epoch in range(cfg.unlearn_epochs):
            epoch_total = 0.0
            epoch_forget = 0.0
            epoch_retain = 0.0
            n_batches = 0

            pbar = tqdm(forget_loader,
                        desc=f"[GA] Epoch {epoch+1}/{cfg.unlearn_epochs}",
                        leave=False)

            for step, forget_batch in enumerate(pbar):
                # Get a retain batch (if retain regularisation is enabled)
                retain_batch = next(retain_iter) if retain_iter else None

                # ── Forward ───────────────────────────────────────────────────
                total_loss, f_loss, r_loss = self._ga_loss(
                    forget_batch, retain_batch
                )

                # Scale for gradient accumulation
                scaled_loss = total_loss / cfg.gradient_accumulation_steps
                scaled_loss.backward()

                # ── Optimizer step (every N accumulation steps) ────────────────
                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.max_grad_norm
                    )
                    self.optimizer.step()
                    if scheduler:
                        scheduler.step()
                    self.optimizer.zero_grad()
                    total_steps += 1

                # Track
                epoch_total  += total_loss.item()
                epoch_forget += f_loss.item()
                epoch_retain += r_loss.item() if hasattr(r_loss, 'item') else r_loss
                n_batches    += 1

                pbar.set_postfix({
                    "forget↑": f"{f_loss.item():.3f}",
                    "retain↓": f"{r_loss.item():.3f}" if hasattr(r_loss, 'item') else "N/A",
                    "total":   f"{total_loss.item():.3f}",
                })

            # Epoch averages
            avg_total  = epoch_total  / max(n_batches, 1)
            avg_forget = epoch_forget / max(n_batches, 1)
            avg_retain = epoch_retain / max(n_batches, 1)

            epoch_losses.append((epoch + 1, avg_total))
            forget_losses.append((epoch + 1, avg_forget))
            retain_losses.append((epoch + 1, avg_retain))

            print(f"[GA] Epoch {epoch+1:02d} | "
                  f"forget_loss={avg_forget:.4f} | "
                  f"retain_loss={avg_retain:.4f} | "
                  f"total={avg_total:.4f}")

        elapsed = time.time() - t0
        print(f"[GA] Training complete in {elapsed:.1f}s ({total_steps} steps)")

        return UnlearningResult(
            method="GradientAscent",
            epoch_losses=epoch_losses,
            forget_losses=forget_losses,
            retain_losses=retain_losses,
            total_steps=total_steps,
            elapsed_sec=elapsed,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _infinite_iter(loader: DataLoader):
    """Cycle through a DataLoader indefinitely (for retain set sampling)."""
    while True:
        for batch in loader:
            yield batch
