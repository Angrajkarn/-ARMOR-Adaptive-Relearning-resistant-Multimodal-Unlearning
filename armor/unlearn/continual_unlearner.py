"""
armor/unlearn/continual_unlearner.py
=====================================
Module 1 — Lifelong (Continual) Machine Unlearning

Solves the core challenge of **sequential** "right to be forgotten" requests:
each new forget cohort risks catastrophic forgetting of the retain set and
progressive degradation of the base model's general reasoning abilities.

Two complementary defences are implemented:

1. Experience Replay Buffer
   ─────────────────────────
   Maintains a circular buffer of compressed retain-set exemplars using
   reservoir sampling (Vitter, 1985). On each new forget request, a small
   replay mini-batch is mixed into every gradient step alongside the new
   retain batch, anchoring the model near its previous retain-set behaviour.

   Why reservoir sampling?
     • Guarantees uniform coverage of the entire retain stream seen so far
     • O(1) insertion / O(K) memory — no need to store the full retain history

2. Fisher Information Matrix (FIM) Subspace Masking (optional)
   ──────────────────────────────────────────────────────────────
   Computes the *diagonal* FIM over the retain set (a cheap approximation of
   the true Hessian, similar to EWC — Kirkpatrick et al., 2017). Builds a
   binary mask protecting the top-p% most important parameters. Unlearning
   gradients are then restricted to the low-importance subspace, preventing
   catastrophic forgetting of high-value parameters.

   Why diagonal FIM?
     • Full FIM is O(P²) — infeasible for 7B+ param models
     • Diagonal FIM is O(P), computed in a single forward-backward pass
     • Empirically, diagonal EWC retains 85–92% of full-FIM effectiveness

References
──────────
  • Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks"
    PNAS 2017. (EWC / diagonal FIM)
  • Vitter, "Random sampling with a reservoir." ACM TOCS 1985.
  • Maini et al., "TOFU: A Task of Fictitious Unlearning." 2024.
"""

import random
import time
from collections import deque
from copy import deepcopy
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..config import ARMORConfig
from .gradient_ascent import GradientAscentUnlearner, _infinite_iter


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer — reservoir sampling over retain exemplars
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Fixed-size reservoir-sampled buffer of retain-set batches.

    Guarantees that every sample seen so far has an equal probability of
    occupying each slot in the buffer (uniform coverage).

    Usage
    -----
        buf = ReplayBuffer(capacity=200)
        buf.add_batch(batch)          # call after each retain step
        replay_batch = buf.sample()   # returns a random stored batch
    """

    def __init__(self, capacity: int = 200):
        self.capacity   = capacity
        self._storage: List[Dict[str, torch.Tensor]] = []
        self._n_seen    = 0   # total samples ever offered (for reservoir algo)

    def add_batch(self, batch: Dict[str, torch.Tensor]):
        """Add a batch of retain samples using reservoir sampling."""
        batch_cpu = {k: v.detach().cpu() for k, v in batch.items()}
        # Iterate sample-by-sample for proper Vitter reservoir sampling
        bs = next(iter(batch_cpu.values())).shape[0]
        for i in range(bs):
            sample = {k: v[i:i+1] for k, v in batch_cpu.items()}
            self._n_seen += 1
            if len(self._storage) < self.capacity:
                self._storage.append(sample)
            else:
                # Replace a random slot with probability capacity / n_seen
                j = random.randint(0, self._n_seen - 1)
                if j < self.capacity:
                    self._storage[j] = sample

    def sample(self, n: int = 4) -> Optional[Dict[str, torch.Tensor]]:
        """
        Sample `n` exemplars from the buffer and collate into a batch.
        Returns None if the buffer is empty.
        """
        if not self._storage:
            return None
        chosen = random.choices(self._storage, k=min(n, len(self._storage)))
        return {k: torch.cat([s[k] for s in chosen], dim=0) for k in chosen[0]}

    def __len__(self) -> int:
        return len(self._storage)

    def __repr__(self) -> str:
        return (f"ReplayBuffer(capacity={self.capacity}, "
                f"stored={len(self._storage)}, seen={self._n_seen})")


# ─────────────────────────────────────────────────────────────────────────────
# FIM Subspace Mask — protect high-importance parameters
# ─────────────────────────────────────────────────────────────────────────────

class FIMSubspaceMask:
    """
    Computes a binary importance mask from the diagonal Fisher Information Matrix.

    Algorithm
    ---------
    1. Run a forward-backward pass over the retain DataLoader.
    2. Accumulate squared gradients per parameter → diagonal FIM estimate.
    3. Flatten, rank-order, take the top `topk_fraction` as "important."
    4. Return a per-parameter binary mask (True = protected, False = updateable).

    Usage
    -----
        fim = FIMSubspaceMask(model, retain_loader, device, topk_fraction=0.3)
        mask = fim.compute()          # Dict[param_name, bool Tensor]
        fim.apply_mask(grad_dict)     # zeros out protected-param gradients
    """

    def __init__(self,
                 model:          nn.Module,
                 retain_loader:  DataLoader,
                 device:         str,
                 topk_fraction:  float = 0.30):
        self.model          = model
        self.retain_loader  = retain_loader
        self.device         = device
        self.topk_fraction  = topk_fraction
        self._mask: Optional[Dict[str, torch.Tensor]] = None

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Compute the diagonal FIM mask.  Runs one full pass over retain_loader.
        Returns a dict mapping parameter name → binary BoolTensor (same shape).
        """
        print(f"[FIM] Computing diagonal Fisher Information Matrix "
              f"(topk={self.topk_fraction:.0%}) ...")
        self.model.eval()

        # Accumulate squared gradients
        fim_diag: Dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                fim_diag[name] = torch.zeros_like(p.data)

        n_batches = 0
        for batch in tqdm(self.retain_loader, desc="[FIM] retain pass", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch.get("attention_mask",
                                  torch.ones_like(input_ids)).to(self.device)
            labels    = batch.get("labels", input_ids).to(self.device)

            self.model.zero_grad()
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attn_mask,
                                 labels=labels)
            outputs.loss.backward()

            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fim_diag[name] += p.grad.detach().pow(2)

            n_batches += 1

        # Normalize
        for name in fim_diag:
            fim_diag[name] /= max(n_batches, 1)

        # Rank and threshold
        all_vals = torch.cat([v.flatten() for v in fim_diag.values()])
        k        = max(1, int(len(all_vals) * self.topk_fraction))
        threshold, _ = torch.topk(all_vals, k)
        threshold    = threshold[-1].item()   # k-th largest value

        mask: Dict[str, torch.Tensor] = {}
        n_protected = 0
        n_total     = 0
        for name, v in fim_diag.items():
            m = (v >= threshold)
            mask[name] = m
            n_protected += m.sum().item()
            n_total     += m.numel()

        self._mask = mask
        self.model.train()
        print(f"[FIM] Protected {n_protected:,} / {n_total:,} params "
              f"({n_protected/n_total:.1%})")
        return mask

    def zero_protected_grads(self):
        """
        Zero out gradients for protected parameters (call after loss.backward()).
        Must call compute() first.
        """
        if self._mask is None:
            raise RuntimeError("Call compute() before zero_protected_grads()")
        for name, p in self.model.named_parameters():
            if p.grad is not None and name in self._mask:
                p.grad.data[self._mask[name]] = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Continual Unlearner — sequential forget requests with replay + optional FIM
# ─────────────────────────────────────────────────────────────────────────────

class ContinualUnlearner:
    """
    Lifelong Machine Unlearning.

    Wraps `GradientAscentUnlearner` to handle *sequential* forget requests
    while preserving retain-set performance via:
      1. Experience replay buffer (always active)
      2. FIM-based subspace masking (optional, `cfg.continual_use_fim_mask`)

    Usage
    -----
        unlearner = ContinualUnlearner(model, cfg, tokenizer)

        # First forget cohort
        unlearner.unlearn(forget_loader_1, retain_loader_1, request_id=1)

        # Second forget cohort — buffer already populated, no extra args needed
        unlearner.unlearn(forget_loader_2, retain_loader_2, request_id=2)

    The replay buffer persists across `unlearn()` calls — it must NOT be
    recreated between requests.
    """

    def __init__(self,
                 model:     nn.Module,
                 cfg:       ARMORConfig,
                 tokenizer=None):
        self.model     = model
        self.cfg       = cfg
        self.tokenizer = tokenizer
        self.device    = cfg.device

        # Persistent replay buffer — survives across sequential requests
        self.buffer = ReplayBuffer(capacity=cfg.continual_buffer_size)

        # FIM mask (computed lazily on first call if enabled)
        self._fim: Optional[FIMSubspaceMask] = None

        self._request_count = 0
        print(f"[Continual] Initialized | buffer_capacity={cfg.continual_buffer_size} | "
              f"fim_mask={'enabled' if cfg.continual_use_fim_mask else 'disabled'}")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _populate_buffer(self, retain_loader: DataLoader, n_warmup: int = 2):
        """Run n_warmup passes over retain_loader to seed the buffer."""
        print(f"[Continual] Seeding replay buffer from retain_loader ...")
        for _ in range(n_warmup):
            for batch in retain_loader:
                self.buffer.add_batch(batch)
        print(f"[Continual] Buffer: {self.buffer}")

    def _compute_fim(self, retain_loader: DataLoader):
        """Compute (or refresh) the FIM mask from the current retain_loader."""
        self._fim = FIMSubspaceMask(
            model=self.model,
            retain_loader=retain_loader,
            device=self.device,
            topk_fraction=self.cfg.continual_fim_topk,
        )
        self._fim.compute()

    # ── Main API ───────────────────────────────────────────────────────────────

    def unlearn(self,
                forget_loader: DataLoader,
                retain_loader: DataLoader,
                request_id:    int = 0) -> Dict[str, Any]:
        """
        Process one forget request while preserving retain-set knowledge.

        Parameters
        ----------
        forget_loader : DataLoader for the new forget cohort
        retain_loader : DataLoader for the (current) retain set
        request_id    : sequential index for logging

        Returns
        -------
        history dict with loss curves
        """
        self._request_count += 1
        print(f"\n[Continual] ═══ Request #{self._request_count} "
              f"(request_id={request_id}) ═══")

        # Step 1 — Seed replay buffer on first request; update on subsequent
        self._populate_buffer(retain_loader)

        # Step 2 — Compute FIM mask (refresh every request if enabled)
        if self.cfg.continual_use_fim_mask:
            self._compute_fim(retain_loader)

        # Step 3 — Run unlearning with augmented retain stream
        history = self._run_unlearning(forget_loader, retain_loader)

        return history

    def _run_unlearning(self,
                        forget_loader: DataLoader,
                        retain_loader: DataLoader) -> Dict[str, Any]:
        """Core GA loop augmented with replay + optional FIM masking."""
        cfg    = self.cfg
        model  = self.model
        device = self.device
        model.train()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.unlearn_lr,
            weight_decay=cfg.weight_decay,
        )

        retain_iter  = _infinite_iter(retain_loader)
        history      = {"forget_loss": [], "retain_loss": [],
                        "replay_loss": [], "total_loss": []}
        t0           = time.time()
        n_epochs     = cfg.unlearn_epochs

        for epoch in range(1, n_epochs + 1):
            e_forget = e_retain = e_replay = e_total = 0.0
            n_steps  = 0

            pbar = tqdm(forget_loader,
                        desc=f"[Continual] Epoch {epoch}/{n_epochs}",
                        leave=False)

            for f_batch in pbar:
                # ── Forget loss (gradient ascent) ──────────────────────────
                f_ids   = f_batch["input_ids"].to(device)
                f_mask  = f_batch.get("attention_mask",
                                      torch.ones_like(f_ids)).to(device)
                f_labels = f_batch.get("labels", f_ids).to(device)
                out_f   = model(input_ids=f_ids, attention_mask=f_mask,
                                labels=f_labels)
                forget_loss = -cfg.ga_forget_coeff * out_f.loss   # ascent

                # ── Retain loss (current batch) ────────────────────────────
                r_batch  = next(retain_iter)
                r_ids    = r_batch["input_ids"].to(device)
                r_mask   = r_batch.get("attention_mask",
                                       torch.ones_like(r_ids)).to(device)
                r_labels = r_batch.get("labels", r_ids).to(device)
                out_r    = model(input_ids=r_ids, attention_mask=r_mask,
                                 labels=r_labels)
                retain_loss = cfg.ga_retain_coeff * out_r.loss

                # ── Replay loss (from buffer) ──────────────────────────────
                replay_batch = self.buffer.sample(n=cfg.batch_size)
                replay_loss  = torch.tensor(0.0, device=device)
                if replay_batch is not None:
                    rp_ids   = replay_batch["input_ids"].to(device)
                    rp_mask  = replay_batch.get(
                        "attention_mask", torch.ones_like(rp_ids)).to(device)
                    rp_labels = replay_batch.get("labels", rp_ids).to(device)
                    out_rp   = model(input_ids=rp_ids, attention_mask=rp_mask,
                                     labels=rp_labels)
                    # Replay weighted same as retain
                    replay_loss = cfg.ga_retain_coeff * out_rp.loss

                # ── Total loss + backward ──────────────────────────────────
                total_loss = forget_loss + retain_loss + replay_loss
                optimizer.zero_grad()
                (total_loss / cfg.gradient_accumulation_steps).backward()

                # Apply FIM mask: zero gradients on protected parameters
                if self.cfg.continual_use_fim_mask and self._fim is not None:
                    self._fim.zero_protected_grads()

                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                # ── Update replay buffer with this retain batch ────────────
                self.buffer.add_batch(r_batch)

                # ── Track ──────────────────────────────────────────────────
                e_forget += forget_loss.item()
                e_retain += retain_loss.item()
                e_replay += replay_loss.item() \
                    if hasattr(replay_loss, 'item') else 0.0
                e_total  += total_loss.item()
                n_steps  += 1

                pbar.set_postfix({
                    "forget↑": f"{forget_loss.item():.3f}",
                    "retain↓": f"{retain_loss.item():.3f}",
                    "replay":  f"{replay_loss.item():.3f}"
                    if hasattr(replay_loss, 'item') else "0",
                })

            # Epoch summary
            n = max(n_steps, 1)
            history["forget_loss"].append(e_forget / n)
            history["retain_loss"].append(e_retain / n)
            history["replay_loss"].append(e_replay / n)
            history["total_loss"].append(e_total / n)
            print(f"[Continual] Epoch {epoch:02d} | "
                  f"forget={e_forget/n:.4f} | retain={e_retain/n:.4f} | "
                  f"replay={e_replay/n:.4f} | total={e_total/n:.4f} | "
                  f"buf={len(self.buffer)}")

        elapsed = time.time() - t0
        print(f"[Continual] Request complete in {elapsed:.1f}s | "
              f"buffer size: {len(self.buffer)}")
        return history
