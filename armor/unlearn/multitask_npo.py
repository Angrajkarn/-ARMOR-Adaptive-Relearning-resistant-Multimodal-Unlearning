"""
armor/unlearn/multitask_npo.py
==============================
Multi-Task NPO — simultaneously forget K disjoint topics.

Challenge: naive multi-task unlearning causes gradient interference:
    nabla L_forget_A is not perpendicular to nabla L_forget_B

Solution: per-task gradient projection onto orthogonal subspaces.

Algorithm per step:
    For each task k in {1..K}:
        g_k = nabla L_NPO(D_forget^k)
        g_k_ortho = g_k - sum_{j<k} proj(g_k, g_j)

    Combined update:
        theta <- theta - eta * (sum_k w_k * g_k_ortho + beta * nabla L_retain)

Use cases:
    - Forget multiple authors simultaneously
    - Forget multiple sensitive topics (e.g., harmful instructions + PII)
    - Multi-domain unlearning (MUSE: books + news)

Memory note:
    Gradients are stored as per-parameter lists (not a single flat tensor).
    Flattening 81.9M params into a 330MB tensor 3x per step caused OOM/segfault
    on Windows. Per-param storage avoids this entirely.
"""

import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from ..config import ARMORConfig


# ──────────────────────────────────────────────────────────────────────────────
# Per-parameter Gram-Schmidt helpers (no flat tensor allocation)
# ──────────────────────────────────────────────────────────────────────────────

def _dot_per_param(grads_a: List[torch.Tensor],
                   grads_b: List[torch.Tensor]) -> float:
    """Dot product across per-parameter gradient lists."""
    return sum((a * b).sum().item() for a, b in zip(grads_a, grads_b))


def _norm_sq_per_param(grads: List[torch.Tensor]) -> float:
    """Squared norm across per-parameter gradient lists."""
    return sum((g * g).sum().item() for g in grads)


def project_out_per_param(v: List[torch.Tensor],
                           basis: List[List[torch.Tensor]],
                           eps: float = 1e-8) -> List[torch.Tensor]:
    """
    Remove components of v along all gradient lists in basis (Gram-Schmidt).
    Works entirely in per-parameter space — no flat tensor allocation.
    """
    result = [g.clone() for g in v]
    for u in basis:
        norm2 = _norm_sq_per_param(u)
        if norm2 < eps:
            continue
        dot = _dot_per_param(result, u)
        scale = dot / norm2
        result = [r - scale * ui for r, ui in zip(result, u)]
    return result


# ──────────────────────────────────────────────────────────────────────────────
# MultiTaskNPOUnlearner
# ──────────────────────────────────────────────────────────────────────────────

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
                 beta_npo:     float = 0.1,
                 beta_retain:  float = 1.0,
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

        out_cur = self.model(input_ids=ids, attention_mask=mask, labels=labs)

        with torch.no_grad():
            if ref_model is self.model:
                with self.model.disable_adapter():
                    out_ref = self.model(input_ids=ids, attention_mask=mask, labels=labs)
            else:
                out_ref = ref_model(input_ids=ids, attention_mask=mask, labels=labs)

        log_ratio = out_cur.loss - out_ref.loss
        loss_npo  = -F.logsigmoid(-self.beta_npo * log_ratio).mean()
        return loss_npo

    def _capture_grads(self) -> List[torch.Tensor]:
        """
        Capture per-parameter gradients as a list of cloned tensors.
        Only processes trainable parameters to avoid allocating huge zero tensors for frozen weights.
        """
        grads = []
        for p in self.model.parameters():
            if p.requires_grad:
                if p.grad is not None:
                    grads.append(p.grad.detach().clone())
                else:
                    grads.append(torch.zeros_like(p))
        return grads

    def _apply_grads(self, grads: List[torch.Tensor]):
        """Write per-parameter gradient list back to model in-place."""
        idx = 0
        for p in self.model.parameters():
            if p.requires_grad:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.copy_(grads[idx])
                idx += 1

    def train(self,
              forget_loaders: List[DataLoader],
              retain_loader:  DataLoader) -> Dict[str, Any]:
        """
        Run multi-task NPO with per-parameter orthogonal gradient projection.

        Gradient storage uses per-parameter lists (not flat tensors) to
        avoid memory exhaustion crashes on Windows with large models.
        """
        K = len(forget_loaders)
        names   = self.task_names   or [f"Task_{k}" for k in range(K)]
        weights = self.task_weights or [1.0 / K] * K

        self.model.train()
        if self.ref_model is not self.model:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.unlearn_lr,
            weight_decay=self.cfg.weight_decay)

        print(f"\n[MultiTask-NPO] Unlearning {K} tasks: {names}")
        print(f"[MultiTask-NPO] Task weights: {weights}")

        history: Dict[str, List] = {f"task_{n}": [] for n in names}
        history["retain"] = []
        t0 = time.time()

        n_steps_per_epoch = min(len(ld) for ld in forget_loaders)

        for epoch in range(1, self.cfg.unlearn_epochs + 1):
            epoch_losses = {n: 0.0 for n in names}
            epoch_retain = 0.0
            n_steps = 0

            iters       = [iter(ld) for ld in forget_loaders]
            retain_iter = iter(retain_loader)

            pbar = tqdm(range(n_steps_per_epoch),
                        desc=f"[MultiTask-NPO] Epoch {epoch}/{self.cfg.unlearn_epochs}")

            for _ in pbar:
                optimizer.zero_grad()
                task_grads: List[List[torch.Tensor]] = []

                # ── Per-task NPO gradient ──────────────────────────────────
                for k, (it, name) in enumerate(zip(iters, names)):
                    try:
                        batch = next(it)
                    except StopIteration:
                        iters[k] = iter(forget_loaders[k])
                        batch = next(iters[k])

                    loss_k = self._npo_loss(batch, self.ref_model)
                    loss_k.backward()

                    g_k = self._capture_grads()   # per-param list, no flat tensor
                    task_grads.append(g_k)
                    epoch_losses[name] += loss_k.item()
                    optimizer.zero_grad()

                # ── Orthogonal projection (per-parameter Gram-Schmidt) ──────
                ortho_grads: List[List[torch.Tensor]] = []
                basis: List[List[torch.Tensor]] = []
                for g in task_grads:
                    g_ortho = project_out_per_param(g, basis)
                    ortho_grads.append(g_ortho)
                    norm_sq = _norm_sq_per_param(g_ortho)
                    if norm_sq > 1e-12:
                        scale = norm_sq ** 0.5
                        basis.append([gi / scale for gi in g_ortho])

                # ── Weighted sum of orthogonal gradients ───────────────────
                n_params = len([p for p in self.model.parameters() if p.requires_grad])
                combined: List[torch.Tensor] = [
                    torch.zeros_like(p) for p in self.model.parameters() if p.requires_grad
                ]
                for w, g in zip(weights, ortho_grads):
                    for i, gi in enumerate(g):
                        combined[i].add_(gi, alpha=w)

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
                g_retain = self._capture_grads()
                epoch_retain += r_out.loss.item()
                optimizer.zero_grad()

                # ── Final gradient: retain descent − forget ascent ─────────
                final: List[torch.Tensor] = [
                    gr.mul_(self.beta_retain).sub_(cf)
                    for gr, cf in zip(g_retain, combined)
                ]
                self._apply_grads(final)

                # Explicitly free temporary grad lists to help GC
                del task_grads, ortho_grads, basis, combined, g_retain, final

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
