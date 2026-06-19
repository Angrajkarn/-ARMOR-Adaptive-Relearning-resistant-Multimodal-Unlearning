"""
armor/unlearn/hdi.py
=====================
Module 8 — Holographic Destructive Interference (HDI) Unlearning

A zero-shot, gradient-free unlearning method.
It extracts the "eigen-memory" of the forget set from the activation space,
orthogonalizes it against the retain set, and applies a destructive wave-interference 
projection matrix directly onto the model weights algebraically.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader

from armor.config import ARMORConfig

class HolographicInterference:
    def __init__(self, model: nn.Module, cfg: ARMORConfig):
        self.model = model
        self.cfg = cfg
        self.device = cfg.device
        
        # Hyperparameters
        # We target the output projections of MLPs and Attention layers
        self.target_layers = getattr(cfg, 'hdi_target_layers', ['c_proj', 'out_proj', 'down_proj'])
        
    def _find_target_modules(self) -> Dict[str, nn.Module]:
        targets = {}
        for name, module in self.model.named_modules():
            if any(t in name for t in self.target_layers):
                if isinstance(module, nn.Linear) or type(module).__name__ == "Conv1D":
                    targets[name] = module
        return targets

    def get_activations(self, loader: DataLoader, targets: Dict[str, nn.Module]) -> Dict[str, torch.Tensor]:
        """Runs a forward pass and captures the input activations to target layers."""
        activations = {name: [] for name in targets}
        hooks = []
        
        def get_hook(name):
            def hook(module, inp):
                x = inp[0].detach()
                x = x.reshape(-1, x.size(-1))
                if x.size(0) > 1000:
                    indices = torch.randperm(x.size(0))[:1000]
                    x = x[indices]
                activations[name].append(x)
            return hook
            
        for name, module in targets.items():
            hooks.append(module.register_forward_pre_hook(get_hook(name)))
            
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="[HDI] Capturing activations", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                self.model(input_ids=input_ids, attention_mask=attention_mask)
                
        for h in hooks:
            h.remove()
            
        for name in targets:
            if activations[name]:
                activations[name] = torch.cat(activations[name], dim=0)
            else:
                activations[name] = torch.empty(0, device=self.device)
                
        return activations

    def _get_eigen_memory(self, acts: torch.Tensor) -> torch.Tensor:
        """Compute the primary principal component of the activations using Covariance/Eigendecomposition."""
        if acts.size(0) == 0:
            return None
        acts = acts - acts.mean(dim=0, keepdim=True)
        cov = torch.mm(acts.t(), acts) / (acts.size(0) - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        return eigenvectors[:, -1]

    def unlearn(self, forget_loader: DataLoader, retain_loader: DataLoader):
        """
        Executes the Holographic Destructive Interference protocol.
        """
        print("\n[HDI] ═══ Holographic Destructive Interference ═══")
        
        targets = self._find_target_modules()
        print(f"[HDI] Identified {len(targets)} target layers for projection.")
        
        print("[HDI] Phase 1: Capturing Forget Set Hologram...")
        forget_acts = self.get_activations(forget_loader, targets)
        
        print("[HDI] Phase 2: Capturing Retain Set Hologram...")
        retain_acts = self.get_activations(retain_loader, targets)
        
        print("[HDI] Phase 3: Constructing Anti-Phase Projection Matrices...")
        modified_count = 0
        for name, module in tqdm(targets.items(), desc="[HDI] Injecting Interference"):
            f_acts = forget_acts[name]
            r_acts = retain_acts[name]
            
            if f_acts.size(0) == 0:
                continue
                
            v_f = self._get_eigen_memory(f_acts)
            v_r = self._get_eigen_memory(r_acts)
            
            if v_f is None or v_r is None:
                continue
                
            # Orthogonalize the forget eigen-memory against the retain eigen-memory
            overlap = torch.dot(v_f, v_r)
            v_target = v_f - overlap * v_r
            
            norm = torch.norm(v_target)
            if norm < 1e-8:
                continue
                
            v_target = v_target / norm
            
            D = v_target.size(0)
            P = torch.eye(D, device=self.device) - torch.outer(v_target, v_target)
            
            # Apply projection algebraically
            with torch.no_grad():
                if isinstance(module, nn.Linear):
                    # W_new = W P
                    module.weight.copy_(torch.mm(module.weight, P))
                elif type(module).__name__ == "Conv1D":
                    # Conv1D weights are (in_features, out_features), so W_new = P W
                    module.weight.copy_(torch.mm(P, module.weight))
            
            modified_count += 1
            
        print(f"[HDI] Zero-shot unlearning complete. Projected {modified_count} layers.")
