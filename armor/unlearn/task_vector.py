"""
armor/unlearn/task_vector.py
============================
Task Vector Unlearning — weight-space arithmetic, no training loop needed.

Based on "Editing Models with Task Arithmetic" (Ilharco et al., 2023).
Adapted for unlearning: negate the fine-tune direction on the forget set.

Algorithm:
    1. θ_pretrained = original model weights
    2. θ_forget     = fine-tune θ_pretrained on forget set for E epochs
    3. τ_forget     = θ_forget − θ_pretrained   (forget task vector)
    4. θ_unlearned  = θ_pretrained − λ · τ_forget

Advantages:
    - No adversarial training loop → very fast (~seconds)
    - No reference model needed
    - Interpretable: task vector is explicit and inspectable

Limitations:
    - Less precise than NPO/RMU
    - λ needs tuning: too large → catastrophic forgetting of retain set
    - Vulnerable to relearning (does not target flat minima)
"""

import copy
import time
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
from typing import Dict, Optional

from ..config import ARMORConfig


# ─────────────────────────────────────────────────────────────────────────────
# Task Vector
# ─────────────────────────────────────────────────────────────────────────────

class TaskVector:
    """Stores the difference between two sets of model weights."""

    def __init__(self, model_a: PreTrainedModel, model_b: PreTrainedModel):
        """Compute τ = θ_b − θ_a  (element-wise for all parameters)."""
        self.vector: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            params_a = dict(model_a.named_parameters())
            params_b = dict(model_b.named_parameters())
            for name in params_a:
                if name in params_b:
                    self.vector[name] = (params_b[name].float()
                                         - params_a[name].float()).cpu()

    @property
    def norm(self) -> float:
        """L2 norm of the full task vector (useful for diagnostics)."""
        total = sum(v.pow(2).sum().item() for v in self.vector.values())
        return total ** 0.5

    def apply(self, base_model: PreTrainedModel,
              scale: float = 1.0) -> PreTrainedModel:
        """
        Return new model: θ_new = θ_base + scale · τ
        Positive scale  → task addition
        Negative scale  → task negation (unlearning direction)
        """
        result = copy.deepcopy(base_model)
        with torch.no_grad():
            for name, param in result.named_parameters():
                if name in self.vector:
                    delta = self.vector[name].to(param.device, param.dtype)
                    param.data.add_(scale * delta)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Task Vector Unlearner
# ─────────────────────────────────────────────────────────────────────────────

class TaskVectorUnlearner:
    """
    Unlearn by negating the forget task vector.

    Usage:
        unlearner = TaskVectorUnlearner(cfg, model, tokenizer)
        unlearned_model = unlearner.run(forget_loader, retain_loader)
    """

    def __init__(self,
                 cfg:            ARMORConfig,
                 model:          PreTrainedModel,
                 tokenizer:      PreTrainedTokenizer,
                 forget_scale:   float = 1.0,   # λ — negation strength
                 finetune_epochs: int  = 3,
                 finetune_lr:    float = 5e-5):
        self.cfg             = cfg
        self.model           = model
        self.tokenizer       = tokenizer
        self.forget_scale    = forget_scale
        self.finetune_epochs = finetune_epochs
        self.finetune_lr     = finetune_lr

    def _finetune_on_forget(self, forget_loader: DataLoader) -> PreTrainedModel:
        """Step 2: Fine-tune a copy of the model on the forget set."""
        ft_model = copy.deepcopy(self.model)
        ft_model.train()
        optimizer = torch.optim.AdamW(ft_model.parameters(),
                                       lr=self.finetune_lr)
        print(f"[TaskVec] Fine-tuning on forget set "
              f"({self.finetune_epochs} epochs)...")
        for epoch in range(1, self.finetune_epochs + 1):
            total_loss = 0.0
            pbar = tqdm(forget_loader,
                        desc=f"  [TaskVec] FT Epoch {epoch}/{self.finetune_epochs}")
            for batch in pbar:
                ids  = batch["input_ids"].to(self.cfg.device)
                labs = batch["labels"].to(self.cfg.device)
                mask = batch.get("attention_mask",
                                 torch.ones_like(ids)).to(self.cfg.device)
                optimizer.zero_grad()
                out  = ft_model(input_ids=ids, attention_mask=mask, labels=labs)
                out.loss.backward()
                optimizer.step()
                total_loss += out.loss.item()
                pbar.set_postfix(loss=f"{out.loss.item():.4f}")
            avg = total_loss / max(len(forget_loader), 1)
            print(f"  [TaskVec] Epoch {epoch:02d} avg loss: {avg:.4f}")
        return ft_model

    def run(self,
            forget_loader: DataLoader,
            retain_loader: Optional[DataLoader] = None) -> PreTrainedModel:
        """
        Full pipeline:
            1. Fine-tune on forget set → θ_forget
            2. Compute task vector τ = θ_forget − θ_original
            3. Apply negation: θ_unlearned = θ_original − λ · τ
        """
        t0 = time.time()
        print(f"\n[TaskVec] Starting task vector unlearning "
              f"(λ={self.forget_scale})...")

        # Save original weights
        original_model = copy.deepcopy(self.model)

        # Step 2: fine-tune on forget set
        ft_model = self._finetune_on_forget(forget_loader)

        # Step 3: compute forget task vector
        forget_tv = TaskVector(original_model, ft_model)
        print(f"[TaskVec] Forget task vector L2 norm: {forget_tv.norm:.4f}")

        # Step 4: negate → unlearned model
        # scale = −λ means we subtract the forget direction
        unlearned = forget_tv.apply(original_model, scale=-self.forget_scale)
        unlearned.to(self.cfg.device)

        elapsed = time.time() - t0
        print(f"[TaskVec] Done in {elapsed:.1f}s  "
              f"(negation scale λ={self.forget_scale})")
        return unlearned
