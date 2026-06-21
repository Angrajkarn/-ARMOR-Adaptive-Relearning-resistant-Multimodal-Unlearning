"""
armor/unlearn/neural_reconsolidation.py
=======================================
NRU: Neural Reconsolidation Unlearning

This module implements a neuroscience-inspired unlearning algorithm based on the concept of memory reconsolidation.
"""

import time
from typing import Optional, List, Dict, Any, Tuple

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
from armor.unlearn.sam_wrapper import SAMOptimizer

class NRUUnlearner:
    """
    NRU: Neural Reconsolidation Unlearning.

    Three-phase optimization:
      1. Recall Activation: Maximizes likelihood of forget samples to make the memory trace labile.
      2. Amnestic Erasure: Performs unlearning (NPO) on labile weights.
      3. Stabilization: Performs SAM optimization on retain samples to lock the erased weights in a flat minimum.
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

        # We construct the base optimizer and wrap it with SAM for the stabilization phase
        if optimizer is None:
            self.base_optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.unlearn_lr,
                weight_decay=cfg.weight_decay,
            )
        else:
            self.base_optimizer = optimizer

        # SAM Optimizer wrapper for Phase 3 (stabilization)
        self.sam_optimizer = SAMOptimizer(
            self.base_optimizer,
            model=self.model,
            rho=cfg.sam_rho,
            adaptive=cfg.sam_adaptive,
        )

        # Learning rates and factors
        self.recall_lr = cfg.nru_recall_lr
        self.stabilize_coeff = cfg.nru_stabilize_coeff

    def _recall_step(self, forget_batch: dict):
        """Phase 1: Recall-activation step (gradient ascent to make weights labile)"""
        input_ids = forget_batch["input_ids"].to(self.device)
        attn_mask = forget_batch["attention_mask"].to(self.device)
        labels = forget_batch["labels"].to(self.device)

        # Temporary learning rate adjustment for recall step
        for g in self.base_optimizer.param_groups:
            g["lr"] = self.recall_lr

        self.base_optimizer.zero_grad()
        # Compute MLE loss: -log P(y|x)
        policy_lp = compute_token_log_probs(self.model, input_ids, attn_mask, labels)
        mle_loss = -policy_lp.mean()

        mle_loss.backward()
        # Optimization step moves weights in direction of gradient descent of negative logprob (ascent on logprob)
        self.base_optimizer.step()
        self.base_optimizer.zero_grad()

        # Restore original learning rate
        for g in self.base_optimizer.param_groups:
            g["lr"] = self.cfg.unlearn_lr

    def _erasure_step(self, forget_batch: dict) -> torch.Tensor:
        """Phase 2: Amnestic erasure step (NPO unlearning)"""
        input_ids = forget_batch["input_ids"].to(self.device)
        attn_mask = forget_batch["attention_mask"].to(self.device)
        labels = forget_batch["labels"].to(self.device)

        self.base_optimizer.zero_grad()
        # Compute log-probs on current model
        policy_lp = compute_token_log_probs(self.model, input_ids, attn_mask, labels)
        
        # Log-probs on ref model
        with torch.no_grad():
            if self.ref_model is self.model:
                with self.model.disable_adapter():
                    ref_lp = compute_token_log_probs(self.model, input_ids, attn_mask, labels)
            else:
                ref_lp = compute_token_log_probs(self.ref_model, input_ids, attn_mask, labels)

        log_ratio = policy_lp - ref_lp
        npo_loss = -F.logsigmoid(self.cfg.npo_beta * log_ratio).mean()

        npo_loss.backward()
        self.base_optimizer.step()
        self.base_optimizer.zero_grad()
        return npo_loss

    def _stabilization_step(self, retain_batch: dict) -> torch.Tensor:
        """Phase 3: Stabilization step (SAM optimization on retain set)"""
        input_ids = retain_batch["input_ids"].to(self.device)
        attn_mask = retain_batch["attention_mask"].to(self.device)
        labels = retain_batch["labels"].to(self.device)

        # SAM Step 1
        def compute_loss():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels,
            )
            return outputs.loss

        self.sam_optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        self.sam_optimizer.first_step(zero_grad=True)

        # SAM Step 2
        loss2 = compute_loss()
        loss2.backward()
        self.sam_optimizer.second_step(zero_grad=True)

        return loss

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
        """Runs the NRU unlearning loop."""
        cfg = self.cfg
        model = self.model
        model.train()

        retain_iter = self._infinite_iter(retain_loader)
        total_steps_count = len(forget_loader) * cfg.unlearn_epochs
        warmup_steps = max(1, total_steps_count // 10)
        scheduler = get_linear_schedule_with_warmup(
            self.base_optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps_count,
        )

        epoch_losses, forget_losses, retain_losses = [], [], []
        total_optimizer_steps = 0
        t0 = time.time()

        print(f"\n[NRU] Starting training: {cfg.unlearn_epochs} epochs")
        print(f"      NRU Recall LR: {self.recall_lr}")
        print(f"      SAM Neighborhood radius: {cfg.sam_rho}")

        for epoch in range(cfg.unlearn_epochs):
            e_total = e_npo = e_retain = 0.0
            n_batches = 0

            pbar = tqdm(
                forget_loader,
                desc=f"[NRU] Epoch {epoch+1}/{cfg.unlearn_epochs}",
                leave=False,
            )

            for step, forget_batch in enumerate(pbar):
                retain_batch = next(retain_iter) if retain_iter else None

                # 1. Phase 1: Recall-activation (make labile)
                self._recall_step(forget_batch)

                # 2. Phase 2: Amnestic erasure
                npo_loss = self._erasure_step(forget_batch)

                # 3. Phase 3: SAM stabilization on retain
                if retain_batch is not None:
                    r_loss = self._stabilization_step(retain_batch)
                else:
                    r_loss = torch.tensor(0.0, device=self.device)

                scheduler.step()
                total_optimizer_steps += 1

                e_npo    += npo_loss.item()
                e_retain += r_loss.item() if hasattr(r_loss, "item") else 0.0
                e_total  += npo_loss.item() + cfg.npo_retain_coeff * (r_loss.item() if hasattr(r_loss, "item") else 0.0)
                n_batches += 1

                pbar.set_postfix({
                    "npo_erased": f"{npo_loss.item():.3f}",
                    "retain_sam": f"{r_loss.item():.3f}" if hasattr(r_loss, "item") else "0",
                })

            avg_t = e_total / max(n_batches, 1)
            avg_n = e_npo / max(n_batches, 1)
            avg_r = e_retain / max(n_batches, 1)

            epoch_losses.append((epoch + 1, avg_t))
            forget_losses.append((epoch + 1, avg_n))
            retain_losses.append((epoch + 1, avg_r))

            print(f"[NRU] Epoch {epoch+1:02d} | "
                  f"npo_erased={avg_n:.4f} | retain_sam={avg_r:.4f} | total={avg_t:.4f}")

        elapsed = time.time() - t0
        print(f"[NRU] Training complete in {elapsed:.1f}s")

        return UnlearningResult(
            method="NRU",
            epoch_losses=epoch_losses,
            forget_losses=forget_losses,
            retain_losses=retain_losses,
            total_steps=total_optimizer_steps,
            elapsed_sec=elapsed,
        )
