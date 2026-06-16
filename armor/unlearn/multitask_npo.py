"""
armor/unlearn/multitask_npo.py
==============================
Multi-Task NPO — simultaneously forget K disjoint topics.

Challenge: naive multi-task unlearning causes gradient interference:
    ∇L_forget_A ⊥̸ ∇L_forget_B → one task's update degrades the other.

Solution: per-task gradient projection onto orthogonal subspaces.

Algorithm per step:
    For each task k ∈ {1..K}:
        g_k = ∇ L_NPO(D_forget^k)         # task k forget gradient
        g_k_ortho = g_k − Σ_{j<k} proj(g_k, g_j)  # project out previous tasks

    Combined update:
        θ ← θ − η · (Σ_k w_k · g_k_ortho + β · ∇L_retain)

Use cases:
    - Forget multiple authors simultaneously
    - Forget multiple sensitive topics (e.g., harmful instructions + PII)
    - Multi-domain unlearning (MUSE: books + news)
"""

import time
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from ..config import ARMORConfig


def project_out(v: torch.Tensor, basis: List[torch.Tensor],
                eps: float = 1e-8) -> torch.Tensor:
    """
    Remove components of v along all vectors in basis (Gram-Schmidt step).
    v_ortho = v − Σ_u (v·u / ‖u‖²) · u
    """
    result = v.clone()
    for u in basis:
        norm2 = (u * u).sum().clamp(min=eps)
        result = result - ((result * u).sum() / norm2) * u
    return result


class MultiTaskNPOUnlearner:
    """
    Multi-Task NPO with orthogonal gradient projection.

    Usage:
        forget_loaders = [loader_topic_A, loader_topic_B, loader_topic_C]
        unlearner = MultiTaskNPOUnlearner(cfg, model, ref_model, tokenizer,
                                           task_names=["A", "B", "C"])
        unlearner.train(forget_loaders, retain_loader)
    """

    def __init__(self,
                 cfg:          ARMORConfig,
                 model:        PreTrainedModel,
                 ref_model:    PreTrainedModel,
                 tokenizer:    PreTrainedTokenizer,
                 task_names:   Optional[List[str]] = None,
                 beta_npo:     float = 0.1,    # NPO log-ratio strength
                 beta_retain:  float = 1.0,    # Retain loss weight
                 task_weights: Optional[List[float]] = None):
        self.cfg          = cfg
        self.model        = model
        self.ref_model    = ref_model
        self.tokenizer    = tokenizer
        self.beta_npo     = beta_npo
        self.beta_retain  = beta_retain
        self.task_names   = task_names
        self.task_weights = task_weights

    def _npo_loss(self, batch: Dict, ref_model: PreTrainedModel) -> torch.Tensor:
        """Compute NPO loss for a single batch."""
        ids  = batch["input_ids"].to(self.cfg.device)
        labs = batch["labels"].to(self.cfg.device)
        mask = batch.get("attention_mask",
                         torch.ones_like(ids)).to(self.cfg.device)

        # Current model log-probs
        out_cur = self.model(input_ids=ids, attention_mask=mask, labels=labs)

        # Reference model log-probs (frozen)
        with torch.no_grad():
            out_ref = ref_model(input_ids=ids, attention_mask=mask, labels=labs)

        log_ratio = out_cur.loss - out_ref.loss   # positive = model knows more
        loss_npo  = -F.logsigmoid(-self.beta_npo * log_ratio).mean()
        return loss_npo

    def _flat_grad(self) -> torch.Tensor:
        """Flatten all parameter gradients into a single vector."""
        return torch.cat([
            p.grad.view(-1) if p.grad is not None
            else torch.zeros(p.numel(), device=self.cfg.device)
            for p in self.model.parameters()
        ])

    def _set_grad_from_flat(self, flat_grad: torch.Tensor):
        """Write a flat gradient vector back to model.parameters()."""
        idx = 0
        for p in self.model.parameters():
            n = p.numel()
            p.grad = flat_grad[idx:idx+n].view_as(p).clone()
            idx += n

    def train(self,
              forget_loaders: List[DataLoader],
              retain_loader:  DataLoader) -> Dict[str, Any]:
        """
        Run multi-task NPO with orthogonal gradient projection.

        Args:
            forget_loaders: One DataLoader per topic to forget (K loaders)
            retain_loader:  Single retain DataLoader
        """
        K = len(forget_loaders)
        names = self.task_names or [f"Task_{k}" for k in range(K)]
        weights = self.task_weights or [1.0 / K] * K

        self.model.train()
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.unlearn_lr,
            weight_decay=self.cfg.weight_decay)

        print(f"\n[MultiTask-NPO] Unlearning {K} tasks: {names}")
        print(f"[MultiTask-NPO] Task weights: {weights}")

        history: Dict[str, List] = {f"task_{n}": [] for n in names}
        history["retain"] = []
        t0 = time.time()

        # Zip all loaders together (iterate until shortest is exhausted)
        n_steps_per_epoch = min(len(ld) for ld in forget_loaders)

        for epoch in range(1, self.cfg.unlearn_epochs + 1):
            epoch_losses = {n: 0.0 for n in names}
            epoch_retain = 0.0
            n_steps = 0

            iters = [iter(ld) for ld in forget_loaders]
            retain_iter = iter(retain_loader)

            pbar = tqdm(range(n_steps_per_epoch),
                        desc=f"[MultiTask-NPO] Epoch {epoch}/{self.cfg.unlearn_epochs}")

            for _ in pbar:
                optimizer.zero_grad()
                task_grads = []   # store flat grad per task

                # ── Per-task NPO gradient ──────────────────────────────────
                for k, (it, name) in enumerate(zip(iters, names)):
                    try:
                        batch = next(it)
                    except StopIteration:
                        iters[k] = iter(forget_loaders[k])
                        batch = next(iters[k])

                    loss_k = self._npo_loss(batch, self.ref_model)
                    loss_k.backward()

                    g_k = self._flat_grad()
                    task_grads.append(g_k)
                    epoch_losses[name] += loss_k.item()
                    optimizer.zero_grad()

                # ── Orthogonal projection ──────────────────────────────────
                # Remove interference between task gradients
                ortho_grads = []
                basis = []
                for k, g in enumerate(task_grads):
                    g_ortho = project_out(g, basis)
                    ortho_grads.append(g_ortho)
                    if g_ortho.norm() > 1e-6:
                        basis.append(g_ortho / g_ortho.norm())

                # ── Weighted sum of orthogonal gradients ───────────────────
                combined = torch.zeros_like(ortho_grads[0])
                for w, g in zip(weights, ortho_grads):
                    combined += w * g

                # ── Retain gradient ────────────────────────────────────────
                try:
                    r_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    r_batch = next(retain_iter)

                r_ids  = r_batch["input_ids"].to(self.cfg.device)
                r_labs = r_batch["labels"].to(self.cfg.device)
                r_mask = r_batch.get("attention_mask",
                                     torch.ones_like(r_ids)).to(self.cfg.device)
                r_out  = self.model(input_ids=r_ids, attention_mask=r_mask,
                                    labels=r_labs)
                r_out.loss.backward()
                g_retain = self._flat_grad()
                epoch_retain += r_out.loss.item()
                optimizer.zero_grad()

                # ── Final update: negate forget + add retain ───────────────
                final = self.beta_retain * g_retain - combined
                self._set_grad_from_flat(final)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                n_steps += 1

                pbar.set_postfix(
                    **{n[:6]: f"{epoch_losses[n]/max(n_steps,1):.3f}" for n in names},
                    retain=f"{epoch_retain/max(n_steps,1):.3f}")

            s = max(n_steps, 1)
            for n in names:
                history[f"task_{n}"].append(epoch_losses[n] / s)
            history["retain"].append(epoch_retain / s)

            loss_str = " | ".join(f"{n}={epoch_losses[n]/s:.4f}" for n in names)
            print(f"[MultiTask-NPO] Epoch {epoch:02d} | {loss_str} | "
                  f"retain={epoch_retain/s:.4f}")

        print(f"[MultiTask-NPO] Done in {time.time()-t0:.1f}s | {K} tasks")
        return history
