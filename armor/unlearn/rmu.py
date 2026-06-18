"""
armor/unlearn/rmu.py
====================
RMU — Representation Misdirection for Unlearning (Li et al., 2024)

Core idea: instead of manipulating output logits (like GA/NPO), RMU
corrupts the INTERNAL REPRESENTATIONS of forget-set inputs at a chosen
intermediate layer, pushing them toward a random "misdirection" vector.

Loss:
    L_RMU = α · ‖h_L(x_f) − c·u‖²  +  β · ‖h_L(x_r) − h_L_ref(x_r)‖²

Where:
    h_L(x)     = hidden state at layer L for input x
    u          = fixed random unit vector (misdirection direction)
    c          = scaling constant (controls how far to push)
    h_L_ref    = reference model hidden states (frozen)
    α, β       = loss weights

Advantages over GA / NPO:
  - Works at representation level → harder to reverse via output fine-tuning
  - No instability from gradient ascent
  - No reference model log-ratio (simpler than NPO)
  - Fast convergence: single forward pass per step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
from typing import Optional, Dict, Any
import time

from ..config import ARMORConfig


# ─────────────────────────────────────────────────────────────────────────────
# Hook Utilities — extract hidden states at a specific layer
# ─────────────────────────────────────────────────────────────────────────────

class HiddenStateExtractor:
    """
    Uses PyTorch forward hooks to capture hidden states at layer `layer_idx`.
    Works with any HuggingFace transformer that exposes `.layers` or `.h`.
    """

    def __init__(self, model: PreTrainedModel, layer_idx: int):
        self.hidden_state: Optional[torch.Tensor] = None
        self._hook = self._register_hook(model, layer_idx)

    def _register_hook(self, model: PreTrainedModel, layer_idx: int):
        """Find the right transformer block and attach a hook."""
        # Try different architectures
        layers = None
        for attr in ["layers", "h", "blocks", "transformer.h",
                     "model.layers", "gpt_neox.layers"]:
            try:
                obj = model
                for part in attr.split("."):
                    obj = getattr(obj, part)
                if hasattr(obj, "__len__") and len(obj) > layer_idx:
                    layers = obj
                    break
            except AttributeError:
                continue

        if layers is None:
            raise RuntimeError(
                "[RMU] Could not find transformer layers. "
                "Supported: GPT-2, OPT, LLaMA, Mistral, Falcon.")

        def _hook_fn(module, input, output):
            # output is typically (hidden_state, ...) or just hidden_state
            if isinstance(output, tuple):
                self.hidden_state = output[0]
            else:
                self.hidden_state = output

        return layers[layer_idx].register_forward_hook(_hook_fn)

    def remove(self):
        self._hook.remove()


# ─────────────────────────────────────────────────────────────────────────────
# RMU Unlearner
# ─────────────────────────────────────────────────────────────────────────────

class RMUUnlearner:
    """
    Representation Misdirection for Unlearning.

    Usage:
        unlearner = RMUUnlearner(cfg, model, ref_model, tokenizer)
        unlearner.train(forget_loader, retain_loader)
    """

    def __init__(self,
                 cfg:        ARMORConfig,
                 model:      PreTrainedModel,
                 ref_model:  PreTrainedModel,
                 tokenizer:  PreTrainedTokenizer,
                 layer_idx:  Optional[int]  = None,
                 alpha:      float           = 1200.0,
                 beta:       float           = 6.5,
                 c_scale:    float           = 20.0):
        """
        Args:
            layer_idx : which transformer layer to hook (default: ~70% depth)
            alpha     : forget misdirection loss weight
            beta      : retain representation preservation weight
            c_scale   : magnitude of misdirection vector target
        """
        self.cfg       = cfg
        self.model     = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.alpha     = alpha
        self.beta      = beta
        self.c_scale   = c_scale

        # Auto-detect number of layers and pick ~70% depth
        n_layers = self._count_layers(model)
        self.layer_idx = layer_idx if layer_idx is not None else int(n_layers * 0.7)
        print(f"[RMU] Using layer {self.layer_idx} / {n_layers} for misdirection")

        # Hidden size for misdirection vector
        hidden_size = model.config.hidden_size
        # Random unit misdirection vector (fixed throughout training)
        raw = torch.randn(hidden_size, device=cfg.device)
        self.u = F.normalize(raw, dim=0)   # unit vector
        print(f"[RMU] Misdirection vector shape: {self.u.shape} | "
              f"target magnitude: {c_scale}")

    def _count_layers(self, model: PreTrainedModel) -> int:
        for attr in ["layers", "h", "blocks", "transformer.h",
                     "model.layers", "gpt_neox.layers"]:
            try:
                obj = model
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return len(obj)
            except AttributeError:
                continue
        return 12  # safe fallback

    def train(self,
              forget_loader: DataLoader,
              retain_loader: DataLoader) -> Dict[str, Any]:
        """Run RMU training and return loss history."""
        self.model.train()
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        # ── Optimiser ──────────────────────────────────────────────────────
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.unlearn_lr,
            weight_decay=self.cfg.weight_decay)

        # ── Hook hidden states at target layer ─────────────────────────────
        model_extractor = HiddenStateExtractor(self.model,     self.layer_idx)
        ref_extractor   = HiddenStateExtractor(self.ref_model, self.layer_idx)

        history = {"forget_loss": [], "retain_loss": [], "total_loss": []}
        t0      = time.time()
        n_epochs = self.cfg.unlearn_epochs

        for epoch in range(1, n_epochs + 1):
            epoch_f = epoch_r = epoch_t = 0.0
            n_steps = 0

            pbar = tqdm(zip(forget_loader, retain_loader),
                        total=min(len(forget_loader), len(retain_loader)),
                        desc=f"[RMU] Epoch {epoch}/{n_epochs}")

            for f_batch, r_batch in pbar:
                # ── Forget: push h_L(x_f) → c · u ─────────────────────────
                f_ids  = f_batch["input_ids"].to(self.cfg.device)
                f_mask = f_batch.get("attention_mask",
                                     torch.ones_like(f_ids)).to(self.cfg.device)

                self.model(input_ids=f_ids, attention_mask=f_mask)
                h_forget = model_extractor.hidden_state  # [B, T, H]

                # Target: c * u, broadcast to [B, T, H]
                target = self.c_scale * self.u.unsqueeze(0).unsqueeze(0)
                target = target.expand_as(h_forget).to(device=h_forget.device, dtype=h_forget.dtype)
                forget_loss = F.mse_loss(h_forget, target)

                # ── Retain: keep h_L(x_r) ≈ h_L_ref(x_r) ─────────────────
                r_ids  = r_batch["input_ids"].to(self.cfg.device)
                r_mask = r_batch.get("attention_mask",
                                     torch.ones_like(r_ids)).to(self.cfg.device)

                self.model(input_ids=r_ids, attention_mask=r_mask)
                h_retain = model_extractor.hidden_state

                with torch.no_grad():
                    self.ref_model(input_ids=r_ids, attention_mask=r_mask)
                    h_ref = ref_extractor.hidden_state.detach()

                retain_loss = F.mse_loss(h_retain, h_ref.to(device=h_retain.device, dtype=h_retain.dtype))

                # ── Total loss ─────────────────────────────────────────────
                loss = self.alpha * forget_loss + self.beta * retain_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_f += forget_loss.item()
                epoch_r += retain_loss.item()
                epoch_t += loss.item()
                n_steps += 1

                pbar.set_postfix({
                    "forget": f"{forget_loss.item():.4f}",
                    "retain": f"{retain_loss.item():.4f}"})

            # ── Epoch summary ──────────────────────────────────────────────
            f_avg = epoch_f / max(n_steps, 1)
            r_avg = epoch_r / max(n_steps, 1)
            t_avg = epoch_t / max(n_steps, 1)
            history["forget_loss"].append(f_avg)
            history["retain_loss"].append(r_avg)
            history["total_loss"].append(t_avg)
            print(f"[RMU] Epoch {epoch:02d} | "
                  f"forget={f_avg:.4f} | retain={r_avg:.4f} | "
                  f"total={t_avg:.4f}")

        elapsed = time.time() - t0
        n_total = n_epochs * n_steps
        print(f"[RMU] Training complete in {elapsed:.1f}s ({n_total} steps)")

        # Clean up hooks
        model_extractor.remove()
        ref_extractor.remove()

        return history
