"""
armor/unlearn/rlace_rmu.py
===========================
Module 3 — Advanced RMU with RLACE (Relaxed Linear Adversarial Concept Erasure)

Problem with vanilla RMU
------------------------
The original RMU (rmu.py) pushes hidden states toward a *random* unit vector u.
This is conceptually simple but leaves a key vulnerability: a determined
adversary can train a linear probe on the misdirected representations and
*still* recover forget-set membership, because:
  • The misdirection direction is arbitrary — it doesn't account for the
    actual geometry of forget-set information in representation space.
  • Representations are corrupted in one direction; forget-set info may
    leak through orthogonal dimensions.

RLACE Solution
--------------
Replace the random u with a **data-driven** erasure direction computed via
Relaxed Linear Adversarial Concept Erasure (Ravfogel et al., 2022):

1. Collect hidden states at layer L for forget set (class=1) and retain set
   (class=0).
2. Solve a minimax game to find the projection P = I − v·vᵀ that:
     • Minimises linear probe accuracy on the projected representations
     • Preserves maximum variance in non-membership dimensions
3. The misdirection target becomes: P · h_L(x_f) instead of c · u

Multi-Layer Extension
---------------------
Apply RLACE independently at N equally-spaced layers. This forces the model
to erase forget-set membership from multiple levels of the representation
hierarchy, making the erasure resistant to probes at any layer depth.

Practical Implementation
------------------------
We implement RLACE via **projected gradient descent** on the minimax objective
(Ravfogel et al., Algorithm 1). No external library needed — pure PyTorch.

The projection matrix P is recomputed ONCE before training begins, then used
as a fixed target during the RMU training loop (similar to generating u).

References
----------
  • Ravfogel et al., "Linear Adversarial Concept Erasure." ICML 2022.
  • Li et al., "The WMDP Benchmark." (RMU baseline) NeurIPS 2024.
  • Belrose et al., "Eliciting Latent Predictions from Transformers with
    the Tuned Lens." 2023. (linear probing methodology)
"""

import time
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import ARMORConfig
from .rmu import HiddenStateExtractor


# ─────────────────────────────────────────────────────────────────────────────
# Linear Membership Probe — logistic regression on hidden states
# ─────────────────────────────────────────────────────────────────────────────

class LinearMembershipProbe(nn.Module):
    """
    Single-layer linear probe: h → {forget, retain}.
    Trained for a few epochs to find the most linearly-separable direction.

    Used by RLACEEraser to define the concept direction to erase.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 2)   # binary: forget vs retain

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [N, H] → logits [N, 2]."""
        return self.linear(h)


# ─────────────────────────────────────────────────────────────────────────────
# RLACE Eraser — compute the null-space projection matrix
# ─────────────────────────────────────────────────────────────────────────────

class RLACEEraser:
    """
    Computes the RLACE orthogonal projection matrix P = I − v·vᵀ
    that nullifies the forget-set membership direction in hidden-state space.

    Algorithm (simplified Ravfogel et al., 2022 — single-direction removal)
    ─────────────────────────────────────────────────────────────────────────
    1. Collect mean-pooled hidden states for forget (y=1) and retain (y=0).
    2. Run projected gradient descent to find v = argmax_{|v|=1} accuracy(probe)
       constrained to a unit sphere, where the probe is trained on P·h = (I-vvᵀ)h.
    3. The final P = I − v·vᵀ is the erasure projector.

    Usage
    -----
        eraser = RLACEEraser(hidden_size=768, n_iters=300, probe_epochs=10)
        P = eraser.fit(forget_h, retain_h)   # [H, H] float32 CPU tensor
        target = eraser.project(h_forget)    # [B, T, H]
    """

    def __init__(self,
                 hidden_size:  int,
                 n_iters:      int   = 300,
                 probe_epochs: int   = 10,
                 device:       str   = "cpu"):
        self.hidden_size  = hidden_size
        self.n_iters      = n_iters
        self.probe_epochs = probe_epochs
        self.device       = device
        self.P: Optional[torch.Tensor] = None   # [H, H] projection matrix

    def fit(self,
            forget_h: torch.Tensor,
            retain_h: torch.Tensor) -> torch.Tensor:
        """
        Fit the erasure projector.

        Parameters
        ----------
        forget_h : [N_f, H] mean-pooled hidden states for forget set
        retain_h : [N_r, H] mean-pooled hidden states for retain set

        Returns
        -------
        P : [H, H] orthogonal projection matrix (on CPU)
        """
        H = self.hidden_size
        dev = self.device

        # Combine into supervised dataset
        X = torch.cat([forget_h, retain_h], dim=0).float().to(dev)   # [N, H]
        y = torch.cat([
            torch.ones(len(forget_h), dtype=torch.long),
            torch.zeros(len(retain_h), dtype=torch.long)
        ]).to(dev)

        # Initialise concept direction v (unit vector)
        v = F.normalize(torch.randn(H, device=dev), dim=0)
        v = v.requires_grad_(True)

        lr  = 1e-2
        opt = torch.optim.SGD([v], lr=lr, momentum=0.9)

        # Projected gradient descent on unit sphere
        # Outer loop: find v that maximises probe accuracy after projection
        for it in range(self.n_iters):
            # Project X onto the space orthogonal to v: X_proj = X - (X·v)v
            v_unit  = F.normalize(v.detach(), dim=0)
            Xv      = (X @ v_unit).unsqueeze(1) * v_unit.unsqueeze(0)  # [N, H]
            X_proj  = (X - Xv).detach()                                 # [N, H]

            # Train a fresh linear probe on projected reps
            probe   = LinearMembershipProbe(H).to(dev)
            p_opt   = torch.optim.Adam(probe.parameters(), lr=1e-2)
            for _ in range(self.probe_epochs):
                p_opt.zero_grad()
                logits = probe(X_proj)
                loss   = F.cross_entropy(logits, y)
                loss.backward()
                p_opt.step()

            # Compute probe accuracy on projected reps (for monitoring)
            with torch.no_grad():
                acc = (probe(X_proj).argmax(dim=1) == y).float().mean().item()

            # Adversarial: update v to MAXIMISE probe loss (maximize separability
            # — we want the direction that most separates forget/retain so we can
            # then PROJECT it away)
            opt.zero_grad()
            v_unit2 = F.normalize(v, dim=0)
            Xv2     = (X @ v_unit2).unsqueeze(1) * v_unit2.unsqueeze(0)
            X_proj2 = X - Xv2
            logits2 = probe(X_proj2)
            adv_loss = -F.cross_entropy(logits2, y)   # maximise CE = find best v
            adv_loss.backward()
            opt.step()

            # Project v back to unit sphere (Riemannian retraction)
            with torch.no_grad():
                v.data = F.normalize(v.data, dim=0)

            if (it + 1) % 50 == 0 or it == 0:
                print(f"  [RLACE] iter {it+1:3d}/{self.n_iters} | "
                      f"probe_acc={acc:.3f} | target→0.5 (chance)")

        # Final projection matrix: P = I − v·vᵀ
        v_final = F.normalize(v.detach(), dim=0).cpu()   # [H]
        I        = torch.eye(H)
        self.P   = I - torch.outer(v_final, v_final)     # [H, H]

        print(f"  [RLACE] P computed | rank={H-1} "
              f"(1 direction erased) | shape={self.P.shape}")
        return self.P

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply the erasure projection P to hidden states.

        Parameters
        ----------
        h : [B, T, H]

        Returns
        -------
        P·h : [B, T, H] with the membership direction nullified
        """
        if self.P is None:
            raise RuntimeError("Call fit() before project()")
        P = self.P.to(device=h.device, dtype=h.dtype)   # [H, H]
        # Efficient batch matmul: h @ P.T (P is symmetric, so P.T = P)
        return torch.einsum("bth,hd->btd", h, P)


# ─────────────────────────────────────────────────────────────────────────────
# RLACE-RMU Unlearner — multi-layer concept erasure
# ─────────────────────────────────────────────────────────────────────────────

class RLACERMUUnlearner:
    """
    Advanced RMU with RLACE misdirection targets.

    Instead of pushing h_L(x_f) → c·u (random), we push:
        h_L(x_f) → P_L · h_L(x_f)   for each chosen layer L

    where P_L = I − v_L·v_L^T is computed by fitting a linear probe
    on the forget vs retain hidden states AT that layer.

    This erases the exact axis that a linear classifier uses to
    identify forget-set samples — provably harder to reverse than
    random misdirection.

    Usage
    -----
        unlearner = RLACERMUUnlearner(cfg, model, ref_model, tokenizer)
        history   = unlearner.train(forget_loader, retain_loader)
    """

    def __init__(self,
                 cfg:       ARMORConfig,
                 model:     nn.Module,
                 ref_model: nn.Module,
                 tokenizer=None,
                 alpha:     float = 1200.0,
                 beta:      float = 6.5):
        self.cfg       = cfg
        self.model     = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.alpha     = alpha
        self.beta      = beta
        self.device    = cfg.device

        # Choose N equally-spaced layers for multi-layer erasure
        n_layers   = self._count_layers(model)
        spacing    = max(1, n_layers // (cfg.rlace_n_layers + 1))
        self.target_layers = [
            spacing * (i + 1) for i in range(cfg.rlace_n_layers)
            if spacing * (i + 1) < n_layers
        ]
        print(f"[RLACE-RMU] Target layers: {self.target_layers} "
              f"(of {n_layers} total)")

        # Projection matrices — computed in prepare()
        self.projectors: Dict[int, RLACEEraser] = {}

    def _count_layers(self, model: nn.Module) -> int:
        for attr in ["layers", "h", "blocks", "transformer.h",
                     "model.layers", "gpt_neox.layers"]:
            try:
                obj = model
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return len(obj)
            except AttributeError:
                continue
        return 12

    @torch.no_grad()
    def _collect_hidden_states(self,
                                loader:    DataLoader,
                                layer_idx: int) -> torch.Tensor:
        """
        Run the model over `loader` and collect mean-pooled hidden states
        at `layer_idx`. Returns [N, H] CPU tensor.
        """
        extractor = HiddenStateExtractor(self.model, layer_idx)
        all_h = []
        self.model.eval()
        for batch in loader:
            ids  = batch["input_ids"].to(self.device)
            mask = batch.get("attention_mask",
                             torch.ones_like(ids)).to(self.device)
            self.model(input_ids=ids, attention_mask=mask)
            h = extractor.hidden_state   # [B, T, H]
            # Mean pool over sequence (ignore padding)
            m = mask.unsqueeze(-1).float()
            h_mean = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            all_h.append(h_mean.float().cpu())
        extractor.remove()
        self.model.train()
        return torch.cat(all_h, dim=0)

    def prepare(self,
                forget_loader: DataLoader,
                retain_loader: DataLoader):
        """
        Compute RLACE projectors for all target layers.
        Must be called BEFORE train().
        """
        print(f"[RLACE-RMU] Computing erasure projectors for "
              f"{len(self.target_layers)} layer(s) ...")

        for layer_idx in self.target_layers:
            print(f"\n[RLACE-RMU] Layer {layer_idx}:")
            forget_h = self._collect_hidden_states(forget_loader, layer_idx)
            retain_h = self._collect_hidden_states(retain_loader, layer_idx)
            print(f"  Forget states: {forget_h.shape} | "
                  f"Retain states: {retain_h.shape}")

            eraser = RLACEEraser(
                hidden_size  = forget_h.shape[1],
                n_iters      = self.cfg.rlace_whittle_iters,
                probe_epochs = self.cfg.rlace_probe_epochs,
                device       = self.device,
            )
            eraser.fit(forget_h, retain_h)
            self.projectors[layer_idx] = eraser

        print(f"\n[RLACE-RMU] All projectors ready.")

    def train(self,
              forget_loader: DataLoader,
              retain_loader: DataLoader) -> Dict[str, Any]:
        """
        Run RLACE-RMU training.
        Calls prepare() automatically if projectors not yet computed.
        """
        if not self.projectors:
            self.prepare(forget_loader, retain_loader)

        cfg    = self.cfg
        model  = self.model
        device = self.device
        model.train()
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.unlearn_lr,
            weight_decay=cfg.weight_decay)

        # Attach hooks for all target layers (model + ref_model)
        model_extractors = {
            L: HiddenStateExtractor(model, L) for L in self.target_layers}
        ref_extractors   = {
            L: HiddenStateExtractor(self.ref_model, L) for L in self.target_layers}

        history  = {"forget_loss": [], "retain_loss": [], "total_loss": []}
        t0       = time.time()
        n_epochs = cfg.unlearn_epochs

        retain_iter = iter([])
        try:
            while True:
                next(retain_iter)
        except StopIteration:
            pass

        from .gradient_ascent import _infinite_iter
        retain_iter = _infinite_iter(retain_loader)

        for epoch in range(1, n_epochs + 1):
            e_forget = e_retain = e_total = 0.0
            n_steps  = 0
            pbar = tqdm(zip(forget_loader, retain_loader),
                        total=min(len(forget_loader), len(retain_loader)),
                        desc=f"[RLACE-RMU] Epoch {epoch}/{n_epochs}",
                        leave=False)

            for f_batch, r_batch in pbar:
                # ── Forget: push h_L → P_L·h_L (erase membership direction) ─
                f_ids  = f_batch["input_ids"].to(device)
                f_mask = f_batch.get("attention_mask",
                                     torch.ones_like(f_ids)).to(device)
                model(input_ids=f_ids, attention_mask=f_mask)

                forget_loss = torch.tensor(0.0, device=device)
                for L, extractor in model_extractors.items():
                    h_forget = extractor.hidden_state   # [B, T, H]
                    # RLACE target: project h onto null-space of membership direction
                    target = self.projectors[L].project(h_forget).detach()
                    forget_loss = forget_loss + F.mse_loss(h_forget, target)
                forget_loss = self.alpha * forget_loss / max(len(self.target_layers), 1)

                # ── Retain: keep h_L(x_r) ≈ h_L_ref(x_r) ────────────────────
                r_ids  = r_batch["input_ids"].to(device)
                r_mask = r_batch.get("attention_mask",
                                     torch.ones_like(r_ids)).to(device)
                model(input_ids=r_ids, attention_mask=r_mask)

                retain_loss = torch.tensor(0.0, device=device)
                with torch.no_grad():
                    self.ref_model(input_ids=r_ids, attention_mask=r_mask)

                for L in self.target_layers:
                    h_retain = model_extractors[L].hidden_state
                    h_ref    = ref_extractors[L].hidden_state.detach()
                    retain_loss = retain_loss + F.mse_loss(
                        h_retain,
                        h_ref.to(device=h_retain.device, dtype=h_retain.dtype))
                retain_loss = self.beta * retain_loss / max(len(self.target_layers), 1)

                total_loss = forget_loss + retain_loss
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                e_forget += forget_loss.item()
                e_retain += retain_loss.item()
                e_total  += total_loss.item()
                n_steps  += 1
                pbar.set_postfix({
                    "rlace_forget": f"{forget_loss.item():.4f}",
                    "retain":       f"{retain_loss.item():.4f}"
                })

            n = max(n_steps, 1)
            history["forget_loss"].append(e_forget / n)
            history["retain_loss"].append(e_retain / n)
            history["total_loss"].append(e_total / n)
            print(f"[RLACE-RMU] Epoch {epoch:02d} | "
                  f"forget={e_forget/n:.4f} | retain={e_retain/n:.4f} | "
                  f"total={e_total/n:.4f}")

        # Cleanup
        for e in model_extractors.values():
            e.remove()
        for e in ref_extractors.values():
            e.remove()

        elapsed = time.time() - t0
        print(f"[RLACE-RMU] Training complete in {elapsed:.1f}s")
        return history
