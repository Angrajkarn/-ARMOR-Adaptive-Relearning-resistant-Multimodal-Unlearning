"""
armor/unlearn/causal_iu.py
==========================
CIU: Causal Interventional Unlearning via Do-Calculus

This module implements Pearl's causal do-calculus on transformer weight spaces.
It computes the Average Causal Effect (ACE) of each transformer layer on forget-set predictions,
and performs surgical do-interventions (targeted unlearning) restricted to the causal sub-network.
"""

import time
from typing import Optional, List, Dict, Any, Tuple, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from armor.config import ARMORConfig
from armor.unlearn.gradient_ascent import UnlearningResult
from armor.unlearn.npo import compute_token_log_probs

class CausalUnlearner:
    """
    CausalUnlearner: Performs Causal Interventional Unlearning (CIU).

    1. Measures Average Causal Effect (ACE) of each transformer block on forget predictions
       by simulating a do(Layer = 0) intervention.
    2. Identifies top-K layers with the highest ACE.
    3. Freezes all other layers and performs targeted weight surgery (unlearning)
       exclusively on the high-causal modules to preserve utility.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cfg: ARMORConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = cfg.device

        # Freeze reference model
        if ref_model is not model:
            for p in ref_model.parameters():
                p.requires_grad_(False)
            ref_model.eval()

        self.num_nodes = cfg.ciu_num_nodes
        self.threshold = cfg.ciu_threshold

        # Find transformer blocks
        self.blocks = self._get_transformer_blocks()
        if self.blocks is None:
            raise ValueError("[CIU] Could not dynamically locate transformer blocks in the model.")
        print(f"[CIU] Dynamically identified {len(self.blocks)} transformer layers.")

    def _get_transformer_blocks(self) -> Optional[nn.ModuleList]:
        """Dynamically identifies the list of transformer blocks."""
        # Try common Hugging Face names
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        
        # Generic module list search
        for name, module in self.model.named_modules():
            if name.endswith("layers") or name.endswith(".h") or name.endswith("layer"):
                if isinstance(module, nn.ModuleList):
                    return module
        return None

    def _compute_forget_loss(self, batch: dict) -> torch.Tensor:
        """Computes the cross-entropy loss on a forget set batch."""
        input_ids = batch["input_ids"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels,
        )
        return outputs.loss

    def _compute_causal_effects(self, forget_loader: DataLoader) -> List[Tuple[int, float]]:
        """
        Computes the Average Causal Effect (ACE) of each transformer layer
        by performing activation patching do-interventions.
        """
        print("[CIU] Computing Average Causal Effect (ACE) of each layer on forget set...")
        self.model.eval()

        # Get a single calibration batch from forget set
        calibration_batch = next(iter(forget_loader))
        calibration_batch = {k: v.to(self.device) for k, v in calibration_batch.items()}

        # 1. Base loss without interventions
        with torch.no_grad():
            base_loss = self._compute_forget_loss(calibration_batch).item()
        
        ace_scores = []

        # 2. Iterate through layers, apply do(Layer_i = 0) hook, and measure loss shift
        for idx in range(len(self.blocks)):
            hook_handle = None
            
            # Hook function to set activations to zero
            def zero_activation_hook(module, inputs, outputs):
                if isinstance(outputs, tuple):
                    hidden = outputs[0]
                    # do(Layer = 0)
                    patched = torch.zeros_like(hidden)
                    return (patched,) + outputs[1:]
                else:
                    return torch.zeros_like(outputs)

            # Register hook on block idx
            hook_handle = self.blocks[idx].register_forward_hook(zero_activation_hook)

            try:
                with torch.no_grad():
                    intervened_loss = self._compute_forget_loss(calibration_batch).item()
                # ACE is the difference in loss when the causal node is zeroed out
                ace = intervened_loss - base_loss
                ace_scores.append((idx, ace))
            finally:
                # Remove hook
                hook_handle.remove()

        # Sort layers by absolute causal effect
        ace_scores = sorted(ace_scores, key=lambda x: abs(x[1]), reverse=True)
        
        print("[CIU] Causal effects (ACE) per layer:")
        for idx, ace in ace_scores:
            print(f"  Layer {idx:2d} | ACE: {ace:+.4f}")

        return ace_scores

    def _npo_forget_loss(self, forget_batch: dict) -> torch.Tensor:
        """Standard NPO loss on the forget batch."""
        input_ids = forget_batch["input_ids"].to(self.device)
        attn_mask = forget_batch["attention_mask"].to(self.device)
        labels = forget_batch["labels"].to(self.device)

        policy_lp = compute_token_log_probs(self.model, input_ids, attn_mask, labels)
        with torch.no_grad():
            if self.ref_model is self.model:
                with self.model.disable_adapter():
                    ref_lp = compute_token_log_probs(self.model, input_ids, attn_mask, labels)
            else:
                ref_lp = compute_token_log_probs(self.ref_model, input_ids, attn_mask, labels)

        log_ratio = policy_lp - ref_lp
        return -F.logsigmoid(self.cfg.npo_beta * log_ratio).mean()

    def _retain_loss(self, retain_batch: dict) -> torch.Tensor:
        """Standard cross-entropy retain loss."""
        batch = {k: v.to(self.device) for k, v in retain_batch.items()}
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs.loss

    @staticmethod
    def _infinite_iter(loader: Optional[DataLoader]):
        if loader is None:
            return None
        while True:
            yield from loader

    def run(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
    ) -> UnlearningResult:
        """Runs the Causal Interventional Unlearning process."""
        # 1. Compute causal layers
        ace_scores = self._compute_causal_effects(forget_loader)
        
        # Select top-K causal layers
        top_k_indices = {idx for idx, ace in ace_scores[:self.num_nodes]}
        print(f"[CIU] Selecting top-{self.num_nodes} causal layers for targeted surgery: {sorted(list(top_k_indices))}")

        # Save original requires_grad states
        orig_requires_grad = {name: param.requires_grad for name, param in self.model.named_parameters()}

        # 2. Freeze all parameters except the targeted causal blocks
        for name, param in self.model.named_parameters():
            is_causal = False
            for idx in top_k_indices:
                # Check if param belongs to the selected transformer block
                if f".h.{idx}." in name or f".layers.{idx}." in name or name.startswith(f"transformer.h.{idx}.") or name.startswith(f"model.layers.{idx}."):
                    is_causal = True
                    break
            
            if is_causal:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 3. Setup optimizer for the surgical parameters only
        active_params = [p for p in self.model.parameters() if p.requires_grad]
        active_names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        print(f"[CIU] Performing surgery on {len(active_params)} parameters (frozen rest of network).")

        optimizer = AdamW(
            active_params,
            lr=self.cfg.unlearn_lr,
            weight_decay=self.cfg.weight_decay,
        )

        cfg = self.cfg
        self.model.train()

        retain_iter = self._infinite_iter(retain_loader)
        total_steps_count = len(forget_loader) * cfg.unlearn_epochs
        warmup_steps = max(1, total_steps_count // 10)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps_count,
        )

        epoch_losses, forget_losses, retain_losses = [], [], []
        total_optimizer_steps = 0
        t0 = time.time()

        print(f"\n[CIU] Starting Interventional Unlearning: {cfg.unlearn_epochs} epochs")

        for epoch in range(cfg.unlearn_epochs):
            e_total = e_npo = e_retain = 0.0
            n_batches = 0

            pbar = tqdm(
                forget_loader,
                desc=f"[CIU] Epoch {epoch+1}/{cfg.unlearn_epochs}",
                leave=False,
            )

            for step, forget_batch in enumerate(pbar):
                retain_batch = next(retain_iter) if retain_iter else None

                optimizer.zero_grad()

                # NPO forget loss
                npo_loss = self._npo_forget_loss(forget_batch)

                # Retain loss
                if retain_batch is not None:
                    r_loss = self._retain_loss(retain_batch)
                else:
                    r_loss = torch.tensor(0.0, device=self.device)

                total_loss = npo_loss + cfg.npo_retain_coeff * r_loss

                # Backward
                scaled = total_loss / cfg.gradient_accumulation_steps
                scaled.backward()

                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(active_params, cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    total_optimizer_steps += 1

                e_total  += total_loss.item()
                e_npo    += npo_loss.item()
                e_retain += r_loss.item() if hasattr(r_loss, "item") else 0.0
                n_batches += 1

                pbar.set_postfix({
                    "npo": f"{npo_loss.item():.3f}",
                    "retain": f"{r_loss.item():.3f}" if hasattr(r_loss, "item") else "0",
                })

            avg_t = e_total / max(n_batches, 1)
            avg_n = e_npo / max(n_batches, 1)
            avg_r = e_retain / max(n_batches, 1)

            epoch_losses.append((epoch + 1, avg_t))
            forget_losses.append((epoch + 1, avg_n))
            retain_losses.append((epoch + 1, avg_r))

            print(f"[CIU] Epoch {epoch+1:02d} | "
                  f"npo={avg_n:.4f} | retain={avg_r:.4f} | total={avg_t:.4f}")

        # Restore original requires_grad states
        for name, param in self.model.named_parameters():
            if name in orig_requires_grad:
                param.requires_grad = orig_requires_grad[name]

        elapsed = time.time() - t0
        print(f"[CIU] Interventional surgery complete in {elapsed:.1f}s")

        return UnlearningResult(
            method="CIU",
            epoch_losses=epoch_losses,
            forget_losses=forget_losses,
            retain_losses=retain_losses,
            total_steps=total_optimizer_steps,
            elapsed_sec=elapsed,
        )
