"""
armor/unlearn/eul.py
====================
EUL — Exact Unlearning via Influence Functions (approximation)

Approximates the leave-one-out gradient update that would have resulted
from training *without* the forget set, using influence functions.

Theoretical basis (Koh & Liang, 2017):
    θ_unlearn ≈ θ − H⁻¹ · ∇L_forget(θ)

Where H = Hessian of the retain loss (approximated using LiSSA or
identity scaling for efficiency on large models).

For practical LLM scale, we use the following approximation:
    θ_unlearn ≈ θ − (1/n_retain) · Σ_{x∈D_f} ∇L(x;θ) / (λ + ε)

This reduces to a damped gradient step on the forget loss.

Note: True EUL with Hessian inversion is O(d³) — prohibitive for 7B.
This implementation uses the diagonal Fisher approximation (fast, scalable).
"""

import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
from typing import Dict, Any, Optional

from ..config import ARMORConfig


class EULUnlearner:
    """
    Exact Unlearning via Influence Functions (diagonal Fisher approximation).

    Two modes:
        mode="diagonal_fisher"  : Scale gradient by inverse diagonal Fisher
        mode="gradient_descent" : Simple damped forget-gradient subtraction

    Usage:
        unlearner = EULUnlearner(cfg, model, tokenizer)
        unlearner.train(forget_loader, retain_loader)
    """

    def __init__(self,
                 cfg:         ARMORConfig,
                 model:       PreTrainedModel,
                 tokenizer:   PreTrainedTokenizer,
                 mode:        str   = "gradient_descent",
                 damping:     float = 0.001,   # λ in H⁻¹ approximation
                 n_fisher:    int   = 50):       # samples for Fisher diagonal
        self.cfg      = cfg
        self.model    = model
        self.tokenizer = tokenizer
        self.mode     = mode
        self.damping  = damping
        self.n_fisher = n_fisher

    def _compute_diagonal_fisher(self,
                                  retain_loader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Compute diagonal of Fisher Information Matrix using retain set.
        F_ii ≈ E[( ∂ log p(y|x) / ∂θ_i )²]
        """
        print("[EUL] Computing diagonal Fisher (retain set)...")
        self.model.eval()
        fisher = {n: torch.zeros_like(p)
                  for n, p in self.model.named_parameters()
                  if p.requires_grad}
        n_samples = 0

        pbar = tqdm(retain_loader, desc="[EUL] Fisher", total=self.n_fisher)
        for batch in pbar:
            if n_samples >= self.n_fisher:
                break
            ids  = batch["input_ids"].to(self.cfg.device)
            labs = batch["labels"].to(self.cfg.device)
            mask = batch.get("attention_mask",
                             torch.ones_like(ids)).to(self.cfg.device)

            self.model.zero_grad()
            out  = self.model(input_ids=ids, attention_mask=mask, labels=labs)
            out.loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)

            n_samples += ids.size(0)

        # Normalise
        for n in fisher:
            fisher[n] /= max(n_samples, 1)

        self.model.train()
        return fisher

    def train(self, forget_loader: DataLoader,
              retain_loader: DataLoader) -> Dict[str, Any]:
        """Run EUL and return history."""
        t0      = time.time()
        history = {"forget_loss": [], "retain_loss": []}
        n_epochs = self.cfg.unlearn_epochs

        # Optionally compute Fisher diagonal for scaling
        fisher = None
        if self.mode == "diagonal_fisher":
            fisher = self._compute_diagonal_fisher(retain_loader)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.unlearn_lr,
            weight_decay=self.cfg.weight_decay)
        self.model.train()

        for epoch in range(1, n_epochs + 1):
            ep_f = ep_r = 0.0
            n_steps = 0

            pbar = tqdm(zip(forget_loader, retain_loader),
                        total=min(len(forget_loader), len(retain_loader)),
                        desc=f"[EUL] Epoch {epoch}/{n_epochs}")

            for f_batch, r_batch in pbar:
                # ── Compute forget gradient ────────────────────────────────
                f_ids  = f_batch["input_ids"].to(self.cfg.device)
                f_labs = f_batch["labels"].to(self.cfg.device)
                f_mask = f_batch.get("attention_mask",
                                     torch.ones_like(f_ids)).to(self.cfg.device)

                optimizer.zero_grad()
                f_out  = self.model(input_ids=f_ids, attention_mask=f_mask,
                                    labels=f_labs)
                f_loss = f_out.loss
                f_loss.backward()

                if fisher is not None:
                    # Scale gradient by inverse diagonal Fisher: g → g / (F + λ)
                    for n, p in self.model.named_parameters():
                        if p.grad is not None and n in fisher:
                            scaling = 1.0 / (fisher[n] + self.damping)
                            p.grad.mul_(scaling)

                # Negate: we want to ASCEND the forget loss
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.neg_()

                # ── Add retain gradient ────────────────────────────────────
                r_ids  = r_batch["input_ids"].to(self.cfg.device)
                r_labs = r_batch["labels"].to(self.cfg.device)
                r_mask = r_batch.get("attention_mask",
                                     torch.ones_like(r_ids)).to(self.cfg.device)
                r_out  = self.model(input_ids=r_ids, attention_mask=r_mask,
                                    labels=r_labs)
                r_loss = r_out.loss
                r_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                ep_f += f_loss.item()
                ep_r += r_loss.item()
                n_steps += 1
                pbar.set_postfix(forget=f"{f_loss.item():.3f}",
                                  retain=f"{r_loss.item():.3f}")

            s = max(n_steps, 1)
            history["forget_loss"].append(ep_f/s)
            history["retain_loss"].append(ep_r/s)
            print(f"[EUL] Epoch {epoch:02d} | "
                  f"forget={ep_f/s:.4f} | retain={ep_r/s:.4f}")

        print(f"[EUL] Done in {time.time()-t0:.1f}s")
        return history
