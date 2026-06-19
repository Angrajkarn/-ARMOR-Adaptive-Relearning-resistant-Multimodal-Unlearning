"""
armor/unlearn/cas.py
=====================
Module 9 — Causal Attention Severing (CAS) Unlearning

Identifies the specific attention heads that fire disproportionately for the 
forget set compared to the retain set. It then surgically severs those heads 
by zeroing out their corresponding input weights in the attention output 
projection layer (c_proj / o_proj).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader

from armor.config import ARMORConfig

class CausalAttentionSevering:
    def __init__(self, model: nn.Module, cfg: ARMORConfig):
        self.model = model
        self.cfg = cfg
        self.device = cfg.device
        
        self.n_head = getattr(model.config, 'n_head', getattr(model.config, 'num_attention_heads', 12))
        self.hidden_size = getattr(model.config, 'n_embd', getattr(model.config, 'hidden_size', 768))
        self.head_dim = self.hidden_size // self.n_head
        
        # Hyperparameters
        self.target_layers = getattr(cfg, 'cas_target_layers', ['attn.c_proj', 'self_attn.o_proj'])
        self.gamma = getattr(cfg, 'cas_gamma', 3.0) # Forget importance must be > gamma * Retain importance
        
    def _find_target_modules(self) -> Dict[str, nn.Module]:
        targets = {}
        for name, module in self.model.named_modules():
            if any(name.endswith(t) for t in self.target_layers):
                targets[name] = module
        return targets

    def get_head_importances(self, loader: DataLoader, targets: Dict[str, nn.Module]) -> Dict[str, torch.Tensor]:
        """Runs a forward pass and captures the L2 norm of each head's output before projection."""
        importances = {name: torch.zeros(self.n_head, device=self.device) for name in targets}
        counts = {name: 0 for name in targets}
        hooks = []
        
        def get_hook(name):
            def hook(module, inp):
                x = inp[0].detach() # (batch, seq, hidden_size)
                batch_size = x.size(0)
                seq_len = x.size(1)
                x = x.view(batch_size, seq_len, self.n_head, self.head_dim)
                
                # L2 norm over head_dim
                head_norms = torch.norm(x, p=2, dim=-1) # (batch, seq, n_head)
                # Mean over batch and seq
                mean_norms = head_norms.mean(dim=(0, 1)) # (n_head,)
                
                importances[name] += mean_norms
                counts[name] += 1
            return hook
            
        for name, module in targets.items():
            hooks.append(module.register_forward_pre_hook(get_hook(name)))
            
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="[CAS] Tracing head activations", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                self.model(input_ids=input_ids, attention_mask=attention_mask)
                
        for h in hooks:
            h.remove()
            
        for name in targets:
            if counts[name] > 0:
                importances[name] /= counts[name]
                
        return importances

    def unlearn(self, forget_loader: DataLoader, retain_loader: DataLoader):
        """
        Executes the Causal Attention Severing protocol.
        """
        print("\n[CAS] ═══ Causal Attention Severing ═══")
        
        targets = self._find_target_modules()
        print(f"[CAS] Identified {len(targets)} attention projection layers.")
        
        print("[CAS] Phase 1: Tracing Forget Set Attention Graph...")
        f_imp = self.get_head_importances(forget_loader, targets)
        
        print("[CAS] Phase 2: Tracing Retain Set Attention Graph...")
        r_imp = self.get_head_importances(retain_loader, targets)
        
        print("[CAS] Phase 3: Surgically Severing Forbidden Attention Heads...")
        severed_count = 0
        
        for name, module in targets.items():
            forget_norms = f_imp[name]
            retain_norms = r_imp[name]
            
            # Find heads where forget activation is disproportionately high
            forbidden_heads = []
            max_ratio = 0.0
            for h in range(self.n_head):
                if retain_norms[h] > 1e-6:
                    ratio = forget_norms[h] / retain_norms[h]
                    max_ratio = max(max_ratio, ratio.item())
                if forget_norms[h] > self.gamma * retain_norms[h] and forget_norms[h] > 1e-3:
                    forbidden_heads.append(h)
                    
            print(f"  > [CAS] Layer {name}: Max Forget/Retain Ratio = {max_ratio:.3f}")
            if not forbidden_heads:
                continue
                
            print(f"  > [CAS] Layer {name}: Severing {len(forbidden_heads)} heads: {forbidden_heads}")
            
            # Surgically zero out the specific slices of the projection matrix
            with torch.no_grad():
                for h in forbidden_heads:
                    start_idx = h * self.head_dim
                    end_idx = (h + 1) * self.head_dim
                    
                    if isinstance(module, nn.Linear):
                        # Weight is (out_features, in_features)
                        module.weight[:, start_idx:end_idx].zero_()
                    elif type(module).__name__ == "Conv1D":
                        # Conv1D weight is (in_features, out_features)
                        module.weight[start_idx:end_idx, :].zero_()
                        
                    severed_count += 1
                    
        print(f"[CAS] Attention Graph Blockade complete. Severed {severed_count} causal pathways.")
