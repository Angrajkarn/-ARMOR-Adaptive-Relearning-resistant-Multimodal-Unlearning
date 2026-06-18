"""
armor/unlearn/sam_wrapper.py
==============================
Sharpness-Aware Minimization (SAM) optimizer wrapper.

Why SAM for unlearning?
-----------------------
Standard GA/NPO find a loss minimum, but the minimum may lie in a sharp
valley of the loss landscape. An adversary who fine-tunes on even a small
number of forget samples can quickly "relearn" forgotten knowledge by
stepping down the other side of that valley.

SAM explicitly seeks FLAT minima by performing a two-step update:

  Step 1 (ascent):  ε̂ = ρ · ∇L / ‖∇L‖   (perturb weights towards steep direction)
  Step 2 (descent): θ ← θ - lr · ∇L(θ + ε̂)  (update at the perturbed point)

By minimising the loss at the worst-case neighbour within radius ρ, SAM
ensures the resulting minimum is flat → small fine-tuning perturbations
stay in the same loss basin → relearning resistance.

Integration
-----------
SAMOptimizer wraps any base optimizer (AdamW, SGD, etc.) and exposes the
same interface. It can wrap both GA and NPO:

    base_opt = AdamW(model.parameters(), lr=1e-5)
    sam_opt  = SAMOptimizer(base_opt, model, rho=0.05)

    # In training loop:
    # First forward+backward (computes perturbation):
    loss.backward()
    sam_opt.first_step(zero_grad=True)
    # Second forward+backward (update at perturbed point):
    loss2.backward()
    sam_opt.second_step(zero_grad=True)

References
----------
  • Foret et al., "Sharpness-Aware Minimization for Efficiently Improving
    Generalization" (ICLR 2021) [arXiv:2010.01412]
  • Kwon et al., "ASAM: Adaptive Sharpness-Aware Minimization" (ICML 2021)
  • Jia et al., "Model Sparsity Can Simplify Machine Unlearning" (2023)
    — demonstrates SAM's effectiveness for relearning resistance
"""

from typing import Callable, Optional

import torch
from torch.optim import Optimizer


class SAMOptimizer(Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.

    Wraps a base optimizer with a two-phase update. The perturbation radius
    ρ (rho) controls how far SAM looks for sharp directions.

    Parameters
    ----------
    base_optimizer : Optimizer — the inner optimizer (e.g. AdamW)
    model          : nn.Module — the model being optimised
    rho            : float — neighbourhood radius (default 0.05)
    adaptive       : bool — if True, uses ASAM (per-parameter scaling)

    Usage
    -----
    optimizer = SAMOptimizer(
        AdamW(model.parameters(), lr=1e-5),
        model=model, rho=0.05
    )

    # Training step:
    def closure():
        optimizer.zero_grad()
        loss = compute_loss(...)
        loss.backward()
        return loss

    loss = closure()
    optimizer.first_step(zero_grad=True)   # Perturb θ → θ + ε̂
    compute_loss(...).backward()            # Compute grad at perturbed point
    optimizer.second_step(zero_grad=True)  # Update + restore θ
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        model: torch.nn.Module,
        rho: float = 0.05,
        adaptive: bool = False,
    ):
        if rho < 0:
            raise ValueError(f"SAM rho must be non-negative, got {rho}")

        # We need param_groups from the base optimizer
        defaults = dict(rho=rho, adaptive=adaptive)
        super().__init__(base_optimizer.param_groups, defaults)

        self.base_optimizer = base_optimizer
        self.model          = model
        self.rho            = rho
        self.adaptive       = adaptive

        # Share param groups with base optimizer for lr/wd access
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """
        Step 1: Perturb weights towards the sharpest direction.

        Saves a copy of the original weights, then applies:
            θ ← θ + ε̂,  where  ε̂ = ρ · ∇L / ‖∇L‖

        After calling this, run another forward+backward pass to get
        gradients at the perturbed point, then call second_step().
        """
        grad_norm = self._grad_norm().item()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # ASAM: scale by |θ|² for adaptive neighbourhood
                if group.get("adaptive", False):
                    scale_p = scale * (p.abs() ** 2)
                else:
                    scale_p = scale

                # ε̂ = scale * ∇L (unit gradient scaled by ρ/‖∇L‖)
                e_w = p.grad * scale_p
                p.add_(e_w)                  # θ ← θ + ε̂

                # Save perturbation for restoration in second_step
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """
        Step 2: Restore original weights and apply base optimizer update.

        After calling first_step() + another backward():
            1. Restore θ ← θ - ε̂  (undo perturbation)
            2. Apply base_optimizer.step() using perturbed-point gradients
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Restore: θ ← θ - ε̂
                p.sub_(self.state[p].get("e_w", 0.0))

        # Apply the actual optimizer step with perturbed-point gradients
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def step(self, closure: Optional[Callable] = None):
        """
        Convenience method for use with standard PyTorch training loops.
        NOTE: For full SAM, prefer calling first_step + second_step manually.
        This method performs only a single-step SAM approximation.
        """
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.first_step(zero_grad=False)
        if closure is not None:
            with torch.enable_grad():
                closure()
        self.second_step()

        if closure is not None:
            return loss

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        """Compute the global ℓ2 norm of all gradients."""
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if group.get("adaptive", False):
                        norms.append((p.abs() * p.grad).norm(p=2).to(shared_device))
                    else:
                        norms.append(p.grad.norm(p=2).to(shared_device))
        return torch.norm(torch.stack(norms), p=2)

    # ── Delegate state dict management to base optimizer ──────────────────────

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)


# ──────────────────────────────────────────────────────────────────────────────
# SAM-aware training loop helper
# ──────────────────────────────────────────────────────────────────────────────

def sam_training_step(
    sam_optimizer: SAMOptimizer,
    loss_fn: Callable[[], torch.Tensor],
    max_grad_norm: float = 1.0,
) -> torch.Tensor:
    """
    Execute one full SAM two-phase update.

    Parameters
    ----------
    sam_optimizer : SAMOptimizer
    loss_fn       : Callable that computes and returns the loss.
                    Must be callable twice (first and second forward passes).
    max_grad_norm : Gradient clipping threshold

    Returns
    -------
    loss : The loss value from the first forward pass (for logging)

    Usage in training loop
    ----------------------
    loss = sam_training_step(
        sam_optimizer,
        loss_fn=lambda: compute_npo_loss(forget_batch, retain_batch),
    )
    """
    # ── Phase 1: Compute gradients at current θ ────────────────────────────────
    sam_optimizer.zero_grad()
    loss = loss_fn()
    loss.backward()

    # Perturb: θ → θ + ε̂
    sam_optimizer.first_step(zero_grad=True)

    # ── Phase 2: Compute gradients at perturbed θ + ε̂ ─────────────────────────
    loss2 = loss_fn()
    loss2.backward()

    # Clip + apply base optimizer update + restore θ
    torch.nn.utils.clip_grad_norm_(
        sam_optimizer.model.parameters(), max_grad_norm
    )
    sam_optimizer.second_step(zero_grad=True)

    return loss.detach()
