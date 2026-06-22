"""
armor/unlearn/who.py
====================
WHO — Weights Harmonization Objective for Unlearning

Decomposes the weight-update space into:
  - Forget subspace  : directions that increase forget-set loss (unlearn)
  - Retain subspace  : directions that preserve retain-set loss (keep)

Uses gradient-based subspace identification:
    g_f = gradient of forget loss  (ascend this)
    g_r = gradient of retain loss  (preserve this)
    g_ortho = component of g_f orthogonal to g_r

This ensures the forget update does NOT interfere with retain directions.

Loss per step:
    θ ← θ + α · g_ortho   (ascend forget, orthogonal to retain)
    θ ← θ − β · g_r       (descend retain, preserve utility)
"""

import time
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
from typing import Dict, Any

from ..config import ARMORConfig


def orthogonal_component(v: torch.Tensor, u: torch.Tensor,
                          eps: float = 1e-8) -> torch.Tensor:
    """
    Return the component of v orthogonal to u:
        v_ortho = v - (v · u / ‖u‖²) · u
    Works on flat tensors.
    """
    dot   = (v * u).sum()
    norm2 = (u * u).sum().clamp(min=eps)
    return v - (dot / norm2) * u


class WHOUnlearner:
    """
    Weights Harmonization Objective unlearner.

    Computes orthogonal forget gradient per step, ensuring forget updates
    don't degrade the retain set. More precise than plain GA.

    Usage:
        unlearner = WHOUnlearner(cfg, model, tokenizer)
        unlearner.train(forget_loader, retain_loader)
    """

    def __init__(self,
                 cfg:       ARMORConfig,
                 model:     PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 alpha:     float = 1.0,   # forget ascent weight
                 beta:      float = 1.0):  # retain descent weight
        self.cfg       = cfg
        self.model     = model
        self.tokenizer = tokenizer
        self.alpha     = alpha
        self.beta      = beta

    def train(self, forget_loader: DataLoader,
              retain_loader: DataLoader) -> Dict[str, Any]:
        self.model.train()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.unlearn_lr,
            weight_decay=self.cfg.weight_decay)

        history = {"forget_loss": [], "retain_loss": [], "ortho_norm": []}
        t0      = time.time()
        n_epochs = self.cfg.unlearn_epochs

        for epoch in range(1, n_epochs + 1):
            ep_f = ep_r = ep_on = 0.0
            n_steps = 0

            pbar = tqdm(zip(forget_loader, retain_loader),
                        total=min(len(forget_loader), len(retain_loader)),
                        desc=f"[WHO] Epoch {epoch}/{n_epochs}")

            for f_batch, r_batch in pbar:
                # ── Forget gradient ────────────────────────────────────────
                f_ids  = f_batch["input_ids"].to(self.cfg.device)
                f_labs = f_batch["labels"].to(self.cfg.device)
                f_mask = f_batch.get("attention_mask",
                                     torch.ones_like(f_ids)).to(self.cfg.device)

                optimizer.zero_grad()
                f_out  = self.model(input_ids=f_ids,
                                    attention_mask=f_mask,
                                    labels=f_labs)
                f_loss = f_out.loss
                f_loss.backward()

                # Capture flat forget gradient
                target_device = next(self.model.parameters()).device
                g_f = torch.cat([
                    p.grad.view(-1).to(target_device) if p.grad is not None
                    else torch.zeros(p.numel(), device=target_device)
                    for p in self.model.parameters()
                ])
                optimizer.zero_grad()

                # ── Retain gradient ────────────────────────────────────────
                r_ids  = r_batch["input_ids"].to(self.cfg.device)
                r_labs = r_batch["labels"].to(self.cfg.device)
                r_mask = r_batch.get("attention_mask",
                                     torch.ones_like(r_ids)).to(self.cfg.device)

                r_out  = self.model(input_ids=r_ids,
                                    attention_mask=r_mask,
                                    labels=r_labs)
                r_loss = r_out.loss
                r_loss.backward()

                g_r = torch.cat([
                    p.grad.view(-1).to(target_device) if p.grad is not None
                    else torch.zeros(p.numel(), device=target_device)
                    for p in self.model.parameters()
                ])
                optimizer.zero_grad()

                # ── Orthogonal forget component ────────────────────────────
                g_ortho = orthogonal_component(g_f, g_r)
                ortho_norm = g_ortho.norm().item()

                # ── Apply combined gradient ────────────────────────────────
                # Net update = −α·g_ortho (ascend forget) − β·g_r (descend retain)
                # We negate because optimizer does gradient DESCENT
                idx = 0
                with torch.no_grad():
                    for p in self.model.parameters():
                        n = p.numel()
                        go_chunk = g_ortho[idx:idx+n].view_as(p).to(p.device)
                        gr_chunk = g_r[idx:idx+n].view_as(p).to(p.device)
                        # Set synthetic gradient: retain descent - forget ascent
                        p.grad = self.beta * gr_chunk - self.alpha * go_chunk
                        idx += n

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                ep_f  += f_loss.item()
                ep_r  += r_loss.item()
                ep_on += ortho_norm
                n_steps += 1

                pbar.set_postfix(forget=f"{f_loss.item():.3f}",
                                  retain=f"{r_loss.item():.3f}",
                                  ortho=f"{ortho_norm:.3f}")

            s = max(n_steps, 1)
            history["forget_loss"].append(ep_f/s)
            history["retain_loss"].append(ep_r/s)
            history["ortho_norm"].append(ep_on/s)
            print(f"[WHO] Epoch {epoch:02d} | "
                  f"forget={ep_f/s:.4f} | retain={ep_r/s:.4f} | "
                  f"ortho_norm={ep_on/s:.4f}")

        print(f"[WHO] Done in {time.time()-t0:.1f}s")
        return history
