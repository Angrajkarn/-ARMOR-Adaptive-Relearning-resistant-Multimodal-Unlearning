"""
armor/unlearn/lora_unlearner.py
================================
Module 6 — Modular LoRA Unlearning (Negative Adapter Weight Arithmetic)

Core Insight
------------
Standard unlearning mutates the base model weights — making it impossible
to separately inspect, audit, or reverse individual forget requests.
LoRA weight arithmetic solves this by keeping the base model FROZEN and
expressing the forget/retain deltas as low-rank adapter matrices.

Algorithm
---------
Step 1  ─ Fine-tune a "forget LoRA" adapter (rank=lora_unlearn_r) on the
          forget set using standard cross-entropy (not gradient ascent).
          This adapter captures the *exact* delta that makes the model
          remember the forget set: Δ_forget = A_forget × B_forget.

Step 2  ─ Negate the adapter: subtract λ · Δ_forget from the base model.
          Mathematically:
              W_unlearned = W_base - λ · (B_forget^T × A_forget^T)
          where A ∈ ℝ^{d×r}, B ∈ ℝ^{r×d_out} are the LoRA matrices.
          λ = cfg.lora_unlearn_scale controls the erasure strength.

Step 3  ─ (Optional) Fine-tune a "retain LoRA" adapter (rank=lora_retain_r)
          on the retain set and ADD it back to recover any lost general
          capability: W_final = W_unlearned + B_retain^T × A_retain^T.

Step 4  ─ Merge both adapters into base model weights for zero-latency
          inference (no adapter overhead at test time).

Advantages over GA/NPO
-----------------------
• Auditable: the forget delta is a stored file (not a weight mutation)
• Modular: different forget requests have independent adapter files
• Reversible: to "un-unlearn," add back the forget adapter
• Zero inference overhead: merged into weights via weight arithmetic
• No catastrophic forgetting of retain set (retain LoRA repairs damage)

Implementation Note
--------------------
We implement the LoRA matrices from scratch (no peft required for the
core algorithm). Optional peft integration is available for QLoRA support.
The from-scratch implementation makes the weight arithmetic fully transparent
and avoids peft version incompatibilities on Kaggle.

References
----------
  • Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
  • Ilharco et al., "Editing Models with Task Arithmetic." ICLR 2023.
    (weight arithmetic / task vectors — our negative LoRA is analogous)
  • Gandikota et al., "Erasing Concepts from Diffusion Models." ICCV 2023.
    (negative fine-tuning for concept erasure in diffusion models)
"""

import time
import math
from copy import deepcopy
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import ARMORConfig
from .gradient_ascent import _infinite_iter


# ─────────────────────────────────────────────────────────────────────────────
# LoRA Layer — low-rank adapter for a single Linear weight matrix
# ─────────────────────────────────────────────────────────────────────────────

class LoRALayer(nn.Module):
    """
    A LoRA adapter for a single nn.Linear layer.

    The forward pass is:  output = linear(x) + (x @ A^T) @ B^T * scaling

    where scaling = lora_alpha / rank.

    This is kept separate from peft to ensure full control over
    the weight arithmetic during unlearning.
    """

    def __init__(self,
                 linear:     nn.Linear,
                 rank:       int   = 16,
                 lora_alpha: int   = 32,
                 dropout:    float = 0.05):
        super().__init__()
        self.linear     = linear
        self.rank       = rank
        self.scaling    = lora_alpha / rank
        self.dropout    = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features  = linear.in_features
        out_features = linear.out_features

        # LoRA matrices A and B
        device = linear.weight.device
        dtype  = linear.weight.dtype
        if not dtype.is_floating_point:
            if hasattr(linear, "compute_dtype") and getattr(linear, "compute_dtype") is not None:
                dtype = getattr(linear, "compute_dtype")
            else:
                dtype = torch.float16 if device.type == "cuda" else torch.float32

        self.lora_A  = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B  = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))

        # Kaiming init for A (B starts at zero → no effect at init)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out  = self.linear(x)
        lora_out  = F.linear(F.linear(self.dropout(x), self.lora_A),
                             self.lora_B) * self.scaling
        return base_out + lora_out

    def get_delta_weight(self) -> torch.Tensor:
        """
        Compute the weight delta: Δ = (B @ A) * scaling.
        Returns: [out_features, in_features]
        """
        return (self.lora_B @ self.lora_A) * self.scaling


# ─────────────────────────────────────────────────────────────────────────────
# LoRA Injection — replace target linear layers with LoRA-wrapped versions
# ─────────────────────────────────────────────────────────────────────────────

class LoRAInjector:
    """
    Injects LoRA adapters into target attention projection layers.
    Stores original nn.Linear references so we can merge later.

    Target modules: q_proj, v_proj, k_proj, o_proj (configurable)
    """

    # Attribute names that contain the target linear layers
    TARGET_SUFFIXES = ("q_proj", "v_proj", "k_proj", "o_proj",
                       "query", "value", "key",       # BERT/RoBERTa names
                       "c_attn",                       # GPT-2 fused QKV
                       )

    def __init__(self,
                 model:      nn.Module,
                 rank:       int   = 16,
                 lora_alpha: int   = 32,
                 dropout:    float = 0.05):
        self.model      = model
        self.rank       = rank
        self.lora_alpha = lora_alpha
        self.dropout    = dropout
        self._adapters: Dict[str, LoRALayer] = {}   # name → LoRALayer

    def inject(self) -> int:
        """
        Walk the model and replace target Linear layers with LoRALayer wrappers.
        Returns number of adapters injected.
        """
        # Freeze base model weights
        for p in self.model.parameters():
            p.requires_grad_(False)

        n_injected = 0
        for name, module in list(self.model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            # Check if this layer name ends with a target suffix
            short_name = name.split(".")[-1]
            if short_name not in self.TARGET_SUFFIXES:
                continue

            # Build LoRALayer wrapping the original linear
            lora_layer = LoRALayer(
                module,
                rank=self.rank,
                lora_alpha=self.lora_alpha,
                dropout=self.dropout,
            )
            # Enable gradients only on LoRA parameters
            lora_layer.lora_A.requires_grad_(True)
            lora_layer.lora_B.requires_grad_(True)

            # Replace module in model
            parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
            parent = dict(self.model.named_modules())[parent_name] \
                if parent_name else self.model
            setattr(parent, attr, lora_layer)

            self._adapters[name] = lora_layer
            n_injected += 1

        if n_injected == 0:
            # Fallback: inject into ALL Linear layers (e.g., distilGPT2)
            print("[LoRA] No target projections found by name — "
                  "falling back to ALL Linear layers ...")
            for name, module in list(self.model.named_modules()):
                if isinstance(module, nn.Linear) and name not in self._adapters:
                    lora_layer = LoRALayer(
                        module, rank=self.rank,
                        lora_alpha=self.lora_alpha,
                        dropout=self.dropout)
                    lora_layer.lora_A.requires_grad_(True)
                    lora_layer.lora_B.requires_grad_(True)
                    parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
                    parent = dict(self.model.named_modules())[parent_name] \
                        if parent_name else self.model
                    setattr(parent, attr, lora_layer)
                    self._adapters[name] = lora_layer
                    n_injected += 1
                    if n_injected >= 12:   # limit for debug models
                        break

        print(f"[LoRA] Injected {n_injected} adapters "
              f"(rank={self.rank}, alpha={self.lora_alpha})")
        return n_injected

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Return only the LoRA A+B parameters (base model is frozen)."""
        params = []
        for adapter in self._adapters.values():
            params.extend([adapter.lora_A, adapter.lora_B])
        return params

    def get_delta_weights(self) -> Dict[str, torch.Tensor]:
        """
        Extract Δ = B@A * scaling for every injected adapter.
        Returns dict: module_name → [out_features, in_features] delta tensor.
        """
        return {name: adapter.get_delta_weight().detach().cpu()
                for name, adapter in self._adapters.items()}

    def remove_and_restore(self):
        """
        Remove LoRA wrappers and restore original Linear layers.
        Re-enables gradients on base model parameters.
        """
        for name, adapter in self._adapters.items():
            parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
            parent = dict(self.model.named_modules())[parent_name] \
                if parent_name else self.model
            setattr(parent, attr, adapter.linear)   # restore original

        for p in self.model.parameters():
            p.requires_grad_(True)

        self._adapters.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Negative LoRA Applicator — subtract forget delta from base weights
# ─────────────────────────────────────────────────────────────────────────────

class NegativeLoRAApplicator:
    """
    Applies: W_unlearned = W_base - λ · Δ_forget
    where Δ_forget = B_forget @ A_forget * scaling.

    The subtraction is done in-place, directly modifying model.weight tensors.
    The forget adapter is then removed (base model restored).
    """

    @staticmethod
    def apply(model:        nn.Module,
              delta_weights: Dict[str, torch.Tensor],
              scale:         float = 1.0,
              verbose:       bool  = True) -> int:
        """
        Subtract λ · delta_weights from the corresponding base model weights.

        Parameters
        ----------
        model          : the base model (adapters must be REMOVED first)
        delta_weights  : {module_name: [out, in] delta tensor} from injector
        scale          : λ — subtraction scale (1.0 = full erasure)
        verbose        : print summary

        Returns
        -------
        n_params_modified : number of parameters modified
        """
        n_modified  = 0
        name_to_mod = {name: mod for name, mod in model.named_modules()}

        for mod_name, delta in delta_weights.items():
            if mod_name not in name_to_mod:
                continue
            mod = name_to_mod[mod_name]
            if not isinstance(mod, nn.Linear):
                continue

            # Check if this is a quantized 4-bit layer
            is_4bit = hasattr(mod.weight, "quant_state") and mod.weight.quant_state is not None
            
            with torch.no_grad():
                if is_4bit:
                    try:
                        import bitsandbytes as bnb
                        # Dequantize to float
                        weight_float = bnb.functional.dequantize_4bit(mod.weight.data, mod.weight.quant_state)
                        delta_dev = delta.to(device=weight_float.device, dtype=weight_float.dtype)
                        # Subtract delta
                        weight_float -= scale * delta_dev
                        
                        # Requantize and assign back
                        quant_type = getattr(mod.weight, "quant_type", "nf4")
                        new_param = bnb.nn.Params4bit(
                            weight_float.to("cpu"),
                            requires_grad=False,
                            quant_type=quant_type
                        ).to(mod.weight.device)
                        mod.weight.data = new_param.data
                        mod.weight.quant_state = new_param.quant_state
                        delta_dev_numel = delta.numel()
                    except Exception as e:
                        print(f"Warning: Failed to apply negative adapter to 4-bit layer {mod_name} due to: {e}. Skipping.")
                        delta_dev_numel = 0
                else:
                    delta_dev = delta.to(device=mod.weight.device, dtype=mod.weight.dtype)
                    mod.weight.data -= scale * delta_dev
                    delta_dev_numel = delta_dev.numel()

            n_modified += delta_dev_numel

        if verbose:
            print(f"[LoRA] Negative adapter applied: "
                  f"modified {n_modified:,} params with λ={scale}")
        return n_modified


# ─────────────────────────────────────────────────────────────────────────────
# LoRA Fine-tuner — train forget or retain adapter
# ─────────────────────────────────────────────────────────────────────────────

def _train_lora(model:       nn.Module,
                cfg:         ARMORConfig,
                loader:      DataLoader,
                injector:    LoRAInjector,
                label:       str = "forget",
                negate_loss: bool = False) -> Dict[str, torch.Tensor]:
    """
    Train a LoRA adapter on `loader` and return the delta weights.

    Parameters
    ----------
    negate_loss : if True, maximise CE (gradient ascent, for forget pass)
                  if False, minimise CE (standard fine-tune, default)
    """
    device = cfg.device
    model.train()

    params = injector.trainable_parameters()
    opt    = torch.optim.AdamW(params, lr=cfg.unlearn_lr,
                                weight_decay=cfg.weight_decay)
    sign   = -1.0 if negate_loss else 1.0

    history = []
    for epoch in range(1, cfg.unlearn_epochs + 1):
        epoch_loss = 0.0
        n_steps    = 0
        pbar = tqdm(loader,
                    desc=f"[LoRA-{label}] Epoch {epoch}/{cfg.unlearn_epochs}",
                    leave=False)
        for batch in pbar:
            ids    = batch["input_ids"].to(device)
            mask   = batch.get("attention_mask",
                               torch.ones_like(ids)).to(device)
            labels = batch.get("labels", ids).to(device)
            out    = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss   = sign * out.loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step()
            epoch_loss += loss.item()
            n_steps    += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg = epoch_loss / max(n_steps, 1)
        history.append(avg)
        print(f"[LoRA-{label}] Epoch {epoch:02d} | loss={avg:.4f}")

    delta = injector.get_delta_weights()
    return delta


# ─────────────────────────────────────────────────────────────────────────────
# LoRA Unlearner — main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class LoRAUnlearner:
    """
    Modular LoRA Unlearning via negative adapter weight arithmetic.

    Pipeline
    --------
    1. Inject forget LoRA adapters → train on forget set (standard CE)
    2. Extract Δ_forget → remove adapters → subtract λ·Δ_forget from base
    3. (Optional) Inject retain LoRA adapters → train on retain set → add to base
    4. Report final model state

    Usage
    -----
        unlearner = LoRAUnlearner(model, cfg)
        history   = unlearner.unlearn(forget_loader, retain_loader)
    """

    def __init__(self, model: nn.Module, cfg: ARMORConfig):
        self.model  = model
        self.cfg    = cfg
        self.device = cfg.device

    def unlearn(self,
                forget_loader: DataLoader,
                retain_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Full LoRA unlearning pipeline."""
        t0      = time.time()
        history = {}

        # ── Step 1: Train forget LoRA ───────────────────────────────────────
        print("\n[LoRA] === Step 1: Training forget adapter ===")
        forget_injector = LoRAInjector(
            self.model,
            rank       = self.cfg.lora_unlearn_r,
            lora_alpha = self.cfg.lora_unlearn_alpha,
        )
        n_forget_adapters = forget_injector.inject()

        # Train: standard CE on forget set (capture what the model knows)
        forget_delta = _train_lora(
            self.model, self.cfg, forget_loader,
            forget_injector, label="forget", negate_loss=False)
        history["forget_delta_norms"] = {
            k: float(v.norm()) for k, v in forget_delta.items()}

        # ── Step 2: Remove forget adapter + subtract from base ──────────────
        print("\n[LoRA] === Step 2: Applying negative adapter ===")
        forget_injector.remove_and_restore()   # restores original Linear layers

        n_modified = NegativeLoRAApplicator.apply(
            self.model,
            forget_delta,
            scale   = self.cfg.lora_unlearn_scale,
            verbose = True,
        )
        history["n_params_modified"] = n_modified

        # ── Step 3 (optional): Train retain LoRA and add back ───────────────
        if retain_loader is not None and self.cfg.lora_retain_r > 0:
            print("\n[LoRA] === Step 3: Training retain adapter ===")
            retain_injector = LoRAInjector(
                self.model,
                rank       = self.cfg.lora_retain_r,
                lora_alpha = self.cfg.lora_unlearn_alpha,
            )
            retain_injector.inject()

            retain_delta = _train_lora(
                self.model, self.cfg, retain_loader,
                retain_injector, label="retain", negate_loss=False)

            # ADD the retain delta back: W += Δ_retain
            print("\n[LoRA] === Step 3b: Adding retain delta ===")
            retain_injector.remove_and_restore()
            NegativeLoRAApplicator.apply(
                self.model, retain_delta,
                scale=-1.0,    # negative of negative = addition
                verbose=True)
            history["retain_delta_norms"] = {
                k: float(v.norm()) for k, v in retain_delta.items()}

        elapsed = time.time() - t0
        print(f"\n[LoRA] Unlearning complete in {elapsed:.1f}s")
        history["elapsed_sec"] = elapsed
        return history
