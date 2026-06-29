"""
armor/unlearn/morphogenetic_repair.py
=====================================
MWRP: Morphogenetic Weight Regeneration Post-Unlearning

This module implements the post-unlearning weight repair algorithm that heals
latent utility degradation by selectively refitting surgically altered weights.
"""

import time
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from armor.config import ARMORConfig
from armor.unlearn.gradient_ascent import UnlearningResult

class MWRPRepairer:
    """
    MWRP: Morphogenetic Weight Regeneration Post-Unlearning.

    Identifies which parameters were surgically altered (damaged) during unlearning,
    creates a parameter-wise gradient mask, and runs a selective retain-set distillation
    finetuning loop updating ONLY the damaged parameters to restore baseline utility.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        pre_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cfg: ARMORConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        pre_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.model = model
        self.pre_model = pre_model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = cfg.device
        self.pre_weights = pre_weights

        # Freeze pre-model if it's a separate model instance
        if pre_model is not model:
            for p in pre_model.parameters():
                p.requires_grad_(False)
            pre_model.eval()

        self.damage_threshold = cfg.mwrp_damage_threshold
        self.repair_epochs = cfg.mwrp_repair_epochs
        self.repair_lr = cfg.mwrp_repair_lr

        # Base optimizer for selective finetuning
        if optimizer is None:
            self.optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.repair_lr,
                weight_decay=cfg.weight_decay,
            )
        else:
            self.optimizer = optimizer

        # Construct gradient masks
        self.masks: Dict[str, torch.Tensor] = {}
        self._construct_damage_masks()

    def _construct_damage_masks(self):
        """Identifies surgically altered weights and builds binary gradient masks."""
        print(f"[MWRP] Constructing damage masks (threshold={self.damage_threshold})...")
        total_params = 0
        damaged_params = 0

        # Create dictionaries of model parameters
        if self.pre_weights is not None:
            pre_params = self.pre_weights
        else:
            pre_params = {name: param for name, param in self.pre_model.named_parameters()}

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in pre_params:
                with torch.no_grad():
                    # Calculate absolute weight difference
                    diff = torch.abs(param - pre_params[name])
                    
                    # Create binary mask (1 if changed, 0 otherwise)
                    mask = (diff > self.damage_threshold).float().to(self.device)
                    
                    self.masks[name] = mask
                    
                    total_params += param.numel()
                    damaged_params += mask.sum().item()

        pct = (damaged_params / max(total_params, 1)) * 100
        print(f"[MWRP] Damage assessment: {damaged_params:.0f}/{total_params} parameters altered ({pct:.4f}%).")

    def _apply_masks_to_gradients(self):
        """Zeroes out gradients for all undamaged parameters, forcing selective repair."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None and name in self.masks:
                    # Element-wise multiply gradient by the binary mask
                    param.grad.data.mul_(self.masks[name])

    def run(self, retain_loader: DataLoader) -> UnlearningResult:
        """Runs the MWRP selective utility repair loop."""
        cfg = self.cfg
        model = self.model
        model.train()

        total_steps_count = len(retain_loader) * self.repair_epochs
        warmup_steps = max(1, total_steps_count // 10)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps_count,
        )

        epoch_losses, retain_losses = [], []
        total_optimizer_steps = 0
        t0 = time.time()

        print(f"\n[MWRP] Starting morphogenetic repair: {self.repair_epochs} epochs")
        print(f"       Repair LR: {self.repair_lr}")

        for epoch in range(self.repair_epochs):
            e_total = 0.0
            n_batches = 0

            pbar = tqdm(
                retain_loader,
                desc=f"[MWRP] Repair Epoch {epoch+1}/{self.repair_epochs}",
                leave=False,
            )

            for step, batch in enumerate(pbar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss

                # Backward pass
                loss.backward()

                # Apply gradient mask to restrict updates strictly to damaged weights
                self._apply_masks_to_gradients()

                # Optimizer step
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

                total_optimizer_steps += 1
                e_total += loss.item()
                n_batches += 1

                pbar.set_postfix({"repair_loss": f"{loss.item():.4f}"})

            avg_l = e_total / max(n_batches, 1)
            epoch_losses.append((epoch + 1, avg_l))
            retain_losses.append((epoch + 1, avg_l))

            print(f"[MWRP] Epoch {epoch+1:02d} | repair_loss={avg_l:.4f}")

        elapsed = time.time() - t0
        print(f"[MWRP] Repair complete in {elapsed:.1f}s")

        return UnlearningResult(
            method="MWRP",
            epoch_losses=epoch_losses,
            forget_losses=[(i, 0.0) for i in range(1, self.repair_epochs + 1)],
            retain_losses=retain_losses,
            total_steps=total_optimizer_steps,
            elapsed_sec=elapsed,
        )
