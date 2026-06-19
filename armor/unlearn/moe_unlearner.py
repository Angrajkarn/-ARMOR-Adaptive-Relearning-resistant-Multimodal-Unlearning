"""
armor/unlearn/moe_unlearner.py
================================
Module 2 — Mixture-of-Experts (MoE) Targeted Unlearning

Problem
-------
MoE models (Mixtral 8x7B, DeepSeek-MoE, Switch Transformer) route each token
through a sparse subset of experts via a learned gating network. Standard
gradient-ascent acts on ALL parameters, including experts that never process
forget-set tokens — wasting computation and risking collateral damage to
unrelated knowledge.

Solution
--------
Two targeted mechanisms:

1. Router-Diversion Loss
   ─────────────────────
   Hook into the MoE gating layer and add a differentiable loss term that
   pushes forget-set token routing *away* from currently-active experts and
   toward the least-used ("dummy") experts:

       L_route = -Σ_i log P(gate_i → dummy_expert | x_forget)

   This degrades the forget path at the routing stage — before expert FFNs
   even compute anything — making unlearning structurally robust to
   relearning attacks that only fine-tune FFN weights.

2. Expert-Selective Magnitude Pruning (optional)
   ───────────────────────────────────────────────
   After identifying which expert indices activate most on the forget set
   (via a tally forward pass), selectively zeroes the smallest-magnitude
   weights within those experts' FFN matrices. Non-targeted experts are
   completely untouched.

Fallback for Dense Models
--------------------------
If no MoE gating layer is detected (e.g., distilGPT2), the module falls
back to standard Gradient Ascent with a warning — the smoke test uses this
path since we don't load Mixtral in debug mode.

References
----------
  • Jiang et al., "Mixtral of Experts." 2024.
  • Fedus et al., "Switch Transformers." JMLR 2022.
  • Zuo et al., "Taming Sparsely Activated Transformer with Stochastic Experts."
    ICLR 2022. (routing regularisation)
"""

import time
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import ARMORConfig
from .gradient_ascent import GradientAscentUnlearner, _infinite_iter


# ─────────────────────────────────────────────────────────────────────────────
# MoE Gating Hook — intercept routing probabilities
# ─────────────────────────────────────────────────────────────────────────────

class MoERouterHook:
    """
    Attaches forward hooks to all MoE gating (router) layers in the model.

    Supports:
      • Mixtral  — `model.model.layers[i].block_sparse_moe.gate`
      • Switch   — `encoder.block[i].layer[1].mlp.router.classifier`
      • Generic  — any `nn.Linear` named 'gate' or 'router' with out_features ≥ 4

    After a forward pass, `self.router_logits` contains a list of
    (batch×seq, n_experts) tensors — one per MoE layer.
    """

    def __init__(self, model: nn.Module):
        self.model          = model
        self.router_logits: List[torch.Tensor] = []
        self._hooks:        List               = []
        self._n_experts:    int                = 0
        self._found_moe     = False

        self._attach_hooks()

    def _attach_hooks(self):
        """Walk the module tree and hook every detected gating layer."""
        for name, module in self.model.named_modules():
            is_gate = (
                isinstance(module, nn.Linear)
                and any(k in name.lower() for k in ("gate", "router"))
                and module.out_features >= 4          # at least 4 experts
            )
            if is_gate:
                self._hooks.append(
                    module.register_forward_hook(self._make_hook(name))
                )
                self._n_experts = max(self._n_experts, module.out_features)
                self._found_moe = True

        if self._found_moe:
            print(f"[MoE] Detected {len(self._hooks)} router(s) | "
                  f"n_experts={self._n_experts}")
        else:
            print("[MoE] No MoE gating layers detected — "
                  "falling back to dense GA unlearning.")

    def _make_hook(self, name: str):
        def _hook(module, input, output):
            # output is the raw routing logits: [B*T, n_experts]
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            self.router_logits.append(logits)
        return _hook

    def clear(self):
        """Reset accumulated logits between forward passes."""
        self.router_logits.clear()

    def remove(self):
        """Detach all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @property
    def is_moe(self) -> bool:
        return self._found_moe


# ─────────────────────────────────────────────────────────────────────────────
# Expert Usage Tallier — identify which experts activate on forget set
# ─────────────────────────────────────────────────────────────────────────────

class ExpertUsageTallier:
    """
    Runs a single forward pass over the forget set and tallies expert
    activation frequency per layer.

    Returns
    -------
    tally : Dict[layer_idx, Tensor(n_experts)]  — activation counts
    """

    @staticmethod
    @torch.no_grad()
    def tally(model: nn.Module,
              forget_loader: DataLoader,
              router_hook: MoERouterHook,
              device: str) -> Dict[int, torch.Tensor]:
        model.eval()
        layer_counts: Dict[int, torch.Tensor] = {}

        for batch in forget_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch.get("attention_mask", torch.ones_like(ids)).to(device)

            router_hook.clear()
            model(input_ids=ids, attention_mask=mask)

            for layer_idx, logits in enumerate(router_hook.router_logits):
                # logits: [B*T, n_experts]
                probs    = F.softmax(logits, dim=-1)
                top_k    = probs.argmax(dim=-1)   # shape [B*T]
                counts   = torch.bincount(top_k.cpu(),
                                          minlength=router_hook._n_experts).float()
                if layer_idx not in layer_counts:
                    layer_counts[layer_idx] = counts
                else:
                    layer_counts[layer_idx] += counts

        model.train()
        return layer_counts


# ─────────────────────────────────────────────────────────────────────────────
# Expert Magnitude Pruner — zero low-magnitude weights in hot experts
# ─────────────────────────────────────────────────────────────────────────────

class ExpertMagnitudePruner:
    """
    Identifies the `top_k` most-activated experts per layer and zeroes
    the lowest-magnitude fraction of their FFN weights.

    Only expert FFN parameters are modified; embedding/attention layers
    are left completely untouched.
    """

    def __init__(self,
                 model:          nn.Module,
                 prune_fraction: float = 0.10,
                 top_k_experts:  int   = 2):
        self.model          = model
        self.prune_fraction = prune_fraction
        self.top_k_experts  = top_k_experts

    def prune(self, tally: Dict[int, torch.Tensor]) -> int:
        """
        Prune weights.  Returns the number of parameters zeroed.

        tally : dict from ExpertUsageTallier.tally()
        """
        n_zeroed = 0

        # Collect expert FFN modules — Mixtral naming convention:
        # model.model.layers[L].block_sparse_moe.experts[E].w1/w2/w3
        for layer_idx, counts in tally.items():
            hot_experts = counts.argsort(descending=True)[:self.top_k_experts]

            for expert_idx in hot_experts.tolist():
                pattern = f"experts.{expert_idx}"
                for name, param in self.model.named_parameters():
                    if pattern in name and "weight" in name:
                        with torch.no_grad():
                            flat    = param.data.abs().flatten()
                            k       = max(1, int(len(flat) * self.prune_fraction))
                            threshold, _ = torch.kthvalue(flat, k)
                            mask    = param.data.abs() <= threshold.item()
                            param.data[mask] = 0.0
                            n_zeroed += mask.sum().item()

        print(f"[MoE Pruner] Zeroed {n_zeroed:,} expert-FFN parameters "
              f"(fraction={self.prune_fraction:.0%})")
        return n_zeroed


# ─────────────────────────────────────────────────────────────────────────────
# MoE Router-Diversion Loss
# ─────────────────────────────────────────────────────────────────────────────

def router_diversion_loss(router_logits: List[torch.Tensor],
                           n_experts:     int) -> torch.Tensor:
    """
    Encourages forget-set tokens to be routed away from *any single* expert
    by maximising the entropy of the routing distribution (uniform diversion).

    A perfectly diverted model routes forget tokens uniformly across all
    experts, effectively spreading — and thus diluting — the forget memory.

    Alternative: push toward the *least-used* expert (index=-1 of tally).
    We use entropy maximisation here as it requires no tally-pass dependency.

    Loss = -mean_over_layers( H(softmax(logits)) )
         = mean_over_layers( mean_over_tokens( sum_e p_e * log(p_e) ) )
    """
    if not router_logits:
        return torch.tensor(0.0)

    total = torch.tensor(0.0, device=router_logits[0].device)
    for logits in router_logits:
        probs   = F.softmax(logits, dim=-1)           # [B*T, E]
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
        total   = total + entropy

    # Negate: we want to MAXIMISE entropy (maximise routing uncertainty)
    return -(total / len(router_logits))


# ─────────────────────────────────────────────────────────────────────────────
# MoE Unlearner — main class
# ─────────────────────────────────────────────────────────────────────────────

class MoEUnlearner:
    """
    Mixture-of-Experts Targeted Unlearning.

    For MoE models: adds router-diversion loss + optional expert pruning.
    For dense models: falls back to standard Gradient Ascent.

    Usage
    -----
        unlearner = MoEUnlearner(model, cfg)
        history   = unlearner.unlearn(forget_loader, retain_loader)
    """

    def __init__(self, model: nn.Module, cfg: ARMORConfig):
        self.model   = model
        self.cfg     = cfg
        self.device  = cfg.device

        # Attach router hooks (will auto-detect MoE vs dense)
        self.router_hook = MoERouterHook(model)

    def unlearn(self,
                forget_loader: DataLoader,
                retain_loader: DataLoader) -> Dict[str, Any]:
        """Run MoE-targeted unlearning."""

        # ── Optional: expert magnitude pruning (one-shot, before training) ──
        if self.cfg.moe_prune_experts and self.router_hook.is_moe:
            print("[MoE] Running expert usage tally for pruning ...")
            tally = ExpertUsageTallier.tally(
                self.model, forget_loader, self.router_hook, self.device)
            pruner = ExpertMagnitudePruner(
                self.model,
                prune_fraction=self.cfg.moe_prune_fraction)
            pruner.prune(tally)

        if not self.router_hook.is_moe:
            # Dense model fallback — plain GA
            print("[MoE] Falling back to standard Gradient Ascent ...")
            ga = GradientAscentUnlearner(self.model, self.cfg)
            result = ga.run(forget_loader, retain_loader)
            return {
                "forget_loss":  [x[1] for x in result.forget_losses],
                "retain_loss":  [x[1] for x in result.retain_losses],
                "router_loss":  [],
                "total_loss":   [x[1] for x in result.epoch_losses],
            }

        # ── MoE training loop with router-diversion ──────────────────────────
        return self._moe_train(forget_loader, retain_loader)

    def _moe_train(self,
                   forget_loader: DataLoader,
                   retain_loader: DataLoader) -> Dict[str, Any]:
        cfg    = self.cfg
        model  = self.model
        device = self.device
        model.train()

        optimizer   = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.unlearn_lr, weight_decay=cfg.weight_decay)
        retain_iter = _infinite_iter(retain_loader)
        history     = {"forget_loss": [], "retain_loss": [],
                       "router_loss": [], "total_loss": []}
        t0          = time.time()
        n_epochs    = cfg.unlearn_epochs

        for epoch in range(1, n_epochs + 1):
            e_forget = e_retain = e_router = e_total = 0.0
            n_steps  = 0
            pbar = tqdm(forget_loader,
                        desc=f"[MoE] Epoch {epoch}/{n_epochs}", leave=False)

            for f_batch in pbar:
                # ── Forget: GA loss + router-diversion ────────────────────
                f_ids    = f_batch["input_ids"].to(device)
                f_mask   = f_batch.get("attention_mask",
                                       torch.ones_like(f_ids)).to(device)
                f_labels = f_batch.get("labels", f_ids).to(device)

                self.router_hook.clear()
                out_f       = model(input_ids=f_ids, attention_mask=f_mask,
                                    labels=f_labels)
                lm_forget   = -cfg.ga_forget_coeff * out_f.loss   # ascent
                route_loss  = cfg.moe_router_loss_coeff * router_diversion_loss(
                    self.router_hook.router_logits,
                    self.router_hook._n_experts)

                # ── Retain: standard CE ───────────────────────────────────
                r_batch  = next(retain_iter)
                r_ids    = r_batch["input_ids"].to(device)
                r_mask   = r_batch.get("attention_mask",
                                       torch.ones_like(r_ids)).to(device)
                r_labels = r_batch.get("labels", r_ids).to(device)
                out_r    = model(input_ids=r_ids, attention_mask=r_mask,
                                 labels=r_labels)
                retain_loss = cfg.ga_retain_coeff * out_r.loss

                total_loss = lm_forget + route_loss + retain_loss
                optimizer.zero_grad()
                (total_loss / cfg.gradient_accumulation_steps).backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                e_forget += lm_forget.item()
                e_retain += retain_loss.item()
                e_router += route_loss.item()
                e_total  += total_loss.item()
                n_steps  += 1

                pbar.set_postfix({
                    "lm↑":    f"{lm_forget.item():.3f}",
                    "route":  f"{route_loss.item():.3f}",
                    "retain↓":f"{retain_loss.item():.3f}",
                })

            n = max(n_steps, 1)
            history["forget_loss"].append(e_forget / n)
            history["retain_loss"].append(e_retain / n)
            history["router_loss"].append(e_router / n)
            history["total_loss"].append(e_total / n)
            print(f"[MoE] Epoch {epoch:02d} | "
                  f"forget={e_forget/n:.4f} | router={e_router/n:.4f} | "
                  f"retain={e_retain/n:.4f} | total={e_total/n:.4f}")

        self.router_hook.remove()
        elapsed = time.time() - t0
        print(f"[MoE] Training complete in {elapsed:.1f}s")
        return history
