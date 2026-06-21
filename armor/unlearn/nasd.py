"""
armor/unlearn/nasd.py
======================
Module 7 — Neuro-Apoptotic Subnetwork Decay (NASD) Unlearning

Simulates biological programmed cell death. Instead of instantaneous weight
updates, NASD identifies a "forget subnetwork" using Fisher Information and
attaches a decay hook. During live inference, those specific weights decay 
gracefully over N steps, simulating organic forgetting without catastrophic
gradient shock.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader

from armor.config import ARMORConfig

class ApoptosisHook:
    """
    A PyTorch forward pre-hook that autonomously decays specific parameters
    during the forward pass, simulating live cell death (forgetting).
    """
    def __init__(self, param_name: str, mask: torch.Tensor, decay_rate: float, max_steps: int):
        self.param_name = param_name
        self.mask = mask
        self.decay_rate = decay_rate
        self.max_steps = max_steps
        self.step = 0

    def __call__(self, module, inputs):
        if self.step < self.max_steps:
            with torch.no_grad():
                param = getattr(module, self.param_name)
                # Apply decay only to the masked parameters
                param.copy_(torch.where(self.mask, param * self.decay_rate, param))
            self.step += 1
        return inputs


class NeuroApoptoticDecay:
    def __init__(self, model: nn.Module, cfg: ARMORConfig):
        self.model = model
        self.cfg = cfg
        self.device = cfg.device
        
        # Hyperparameters (read from config or defaults)
        self.decay_steps   = getattr(cfg, 'nasd_decay_steps', 50)
        self.decay_rate    = getattr(cfg, 'nasd_decay_rate', 0.90)
        self.topk_fraction = getattr(cfg, 'nasd_topk_fraction', 0.05)
        self.lambda_retain = getattr(cfg, 'nasd_lambda_retain', 1.0)
        
        self.hooks = []

    def compute_fisher(self, loader: DataLoader) -> Dict[str, torch.Tensor]:
        """Compute diagonal Fisher Information Matrix for the given loader."""
        self.model.eval()
        fim = {name: torch.zeros_like(param) 
               for name, param in self.model.named_parameters() if param.requires_grad}
        
        n_samples = 0
        for batch in tqdm(loader, desc="[NASD] Computing Fisher", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            if loss is None:
                continue
                
            loss.backward()
            
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fim[name] += param.grad.pow(2)
            
            n_samples += input_ids.size(0)
            
        self.model.zero_grad()
        if n_samples > 0:
            for name in fim:
                fim[name] /= n_samples
                
        return fim

    def infect(self, forget_loader: DataLoader, retain_loader: DataLoader):
        """
        Identify the apoptotic subnetwork and attach autonomous decay hooks.
        """
        print("\n[NASD] === Phase 1: Identifying Apoptotic Subnetwork ===")
        
        print("[NASD] Analyzing Forget Set...")
        fim_forget = self.compute_fisher(forget_loader)
        
        print("[NASD] Analyzing Retain Set...")
        fim_retain = self.compute_fisher(retain_loader)
        
        # Calculate Apoptosis Score: F_forget - lambda * F_retain
        apoptosis_scores = {}
        all_scores = []
        for name in fim_forget:
            score = fim_forget[name] - self.lambda_retain * fim_retain[name]
            apoptosis_scores[name] = score
            all_scores.append(score.view(-1))
            
        # Determine global threshold for top-k%
        all_scores_flat = torch.cat(all_scores)
        k_idx = int((1.0 - self.topk_fraction) * all_scores_flat.numel())
        # Use kthvalue for thresholding
        threshold = torch.kthvalue(all_scores_flat, max(1, k_idx)).values.item()
        
        # Create binary masks for parameters to decay
        masks = {}
        total_params = 0
        decay_params = 0
        for name, score in apoptosis_scores.items():
            mask = score > threshold
            masks[name] = mask
            total_params += mask.numel()
            decay_params += mask.sum().item()
            
        print(f"[NASD] Subnetwork identified: {decay_params:,} / {total_params:,} parameters "
              f"({decay_params/max(1,total_params):.2%}) marked for apoptosis.")
              
        print("\n[NASD] === Phase 2: Injecting Decay Hooks ===")
        self._attach_hooks(masks)
        
    def _attach_hooks(self, masks: Dict[str, torch.Tensor]):
        """
        Attach PyTorch forward_pre_hooks to modules to decay the masked weights.
        """
        module_dict = dict(self.model.named_modules())
        
        hook_count = 0
        for name, param in self.model.named_parameters():
            if name not in masks:
                continue
                
            mask = masks[name]
            if not mask.any():
                continue
                
            # Find the parent module and the attribute name
            parent_name = name.rsplit('.', 1)[0]
            attr_name = name.rsplit('.', 1)[1]
            
            if parent_name in module_dict:
                parent_module = module_dict[parent_name]
                hook = ApoptosisHook(attr_name, mask, self.decay_rate, self.decay_steps)
                handle = parent_module.register_forward_pre_hook(hook)
                self.hooks.append(handle)
                hook_count += 1
                
        print(f"[NASD] Injected {hook_count} apoptosis hooks into the model architecture.")
        print(f"[NASD] Infection complete. The subnetwork will autonomously decay over {self.decay_steps} forward passes.")

    def cure(self):
        """Remove all apoptosis hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        print("[NASD] All apoptosis hooks removed.")
