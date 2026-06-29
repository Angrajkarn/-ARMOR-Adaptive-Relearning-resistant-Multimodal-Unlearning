"""
armor/unlearn/stackelberg_game.py
==================================
SAUG: Stackelberg Adversarial Unlearning Game

This module implements the Stackelberg minimax game unlearning framework where the
unlearner and a relearning auditor are co-trained to produce a model that is robust
against downstream relearning / recovery attacks.
"""

import time
import copy
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

class SAUGUnlearner:
    """
    SAUG: Stackelberg Adversarial Unlearning Game.

    The unlearner (leader) minimizes unlearning loss + retain loss, while anticipating
    an auditor (follower) who tries to recover the forgotten knowledge.
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

        if optimizer is None:
            self.optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.unlearn_lr,
                weight_decay=cfg.weight_decay,
            )
        else:
            self.optimizer = optimizer

        # Instantiate the adversarial auditor (cloned from target model structure)
        print("[SAUG] Cloning model to initialize the adversarial auditor...")
        self.auditor = copy.deepcopy(model)
        
        # Hyperparameters
        self.adv_steps = cfg.saug_adv_steps
        self.adv_lr = cfg.saug_adv_lr
        self.saug_coeff = cfg.saug_coeff

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

    def _run_auditor_inner_loop(self, forget_batch: dict) -> torch.Tensor:
        """
        Syncs the auditor with current model weights, runs inner-loop gradient steps
        to maximize recovery of the forget set, and returns the auditor's final loss.
        """
        forget_batch = {k: v.to(self.device) for k, v in forget_batch.items()}

        # 1. Sync auditor weights with the current model
        self.auditor.load_state_dict(self.model.state_dict(), strict=False)
        self.auditor.train()

        # 2. Setup standard AdamW optimizer for the auditor
        auditor_opt = AdamW(self.auditor.parameters(), lr=self.adv_lr)

        # 3. Perform inner loop gradient steps to relearn/recover
        for _ in range(self.adv_steps):
            auditor_opt.zero_grad()
            outputs = self.auditor(
                input_ids=forget_batch["input_ids"],
                attention_mask=forget_batch["attention_mask"],
                labels=forget_batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            auditor_opt.step()

        # 4. Compute final relearn/recovery loss of the auditor
        self.auditor.eval()
        with torch.no_grad():
            outputs_final = self.auditor(
                input_ids=forget_batch["input_ids"],
                attention_mask=forget_batch["attention_mask"],
                labels=forget_batch["labels"],
            )
            relearn_loss = outputs_final.loss

        return relearn_loss

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
        """Runs the Stackelberg adversarial unlearning game loop."""
        cfg = self.cfg
        model = self.model
        model.train()

        retain_iter = self._infinite_iter(retain_loader)
        total_steps_count = len(forget_loader) * cfg.unlearn_epochs
        warmup_steps = max(1, total_steps_count // 10)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps_count,
        )

        epoch_losses, forget_losses, retain_losses = [], [], []
        total_optimizer_steps = 0
        t0 = time.time()

        print(f"\n[SAUG] Starting Stackelberg Co-Training: {cfg.unlearn_epochs} epochs")
        print(f"       Auditor LR: {self.adv_lr:.2e} | Steps: {self.adv_steps}")
        print(f"       SAUG Penalty Coeff: {self.saug_coeff:.3f}")

        for epoch in range(cfg.unlearn_epochs):
            e_total = e_npo = e_retain = e_relearn = 0.0
            n_batches = 0

            pbar = tqdm(
                forget_loader,
                desc=f"[SAUG] Epoch {epoch+1}/{cfg.unlearn_epochs}",
                leave=False,
            )

            for step, forget_batch in enumerate(pbar):
                retain_batch = next(retain_iter) if retain_iter else None

                # 1. Run follower (auditor) relearning inner loop
                relearn_loss = self._run_auditor_inner_loop(forget_batch)

                # 2. Compute main model losses
                npo_loss = self._npo_forget_loss(forget_batch)

                if retain_batch is not None:
                    r_loss = self._retain_loss(retain_batch)
                else:
                    r_loss = torch.tensor(0.0, device=self.device)

                # Unlearner objective: minimize NPO forget loss + retain loss,
                # and maximize the auditor's final post-relearning loss (resistance to relearning).
                # Therefore, we subtract the relearn loss.
                total_loss = (
                    npo_loss
                    + cfg.npo_retain_coeff * r_loss
                    - self.saug_coeff * relearn_loss
                )

                # Backward
                self.optimizer.zero_grad()
                scaled = total_loss / cfg.gradient_accumulation_steps
                scaled.backward()

                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    total_optimizer_steps += 1

                e_total   += total_loss.item()
                e_npo     += npo_loss.item()
                e_retain  += r_loss.item() if hasattr(r_loss, "item") else 0.0
                e_relearn += relearn_loss.item()
                n_batches += 1

                pbar.set_postfix({
                    "npo": f"{npo_loss.item():.3f}",
                    "retain": f"{r_loss.item():.3f}" if hasattr(r_loss, "item") else "0",
                    "relearn": f"{relearn_loss.item():.3f}",
                })

            avg_t = e_total / max(n_batches, 1)
            avg_n = e_npo / max(n_batches, 1)
            avg_r = e_retain / max(n_batches, 1)
            avg_re = e_relearn / max(n_batches, 1)

            epoch_losses.append((epoch + 1, avg_t))
            forget_losses.append((epoch + 1, avg_n))
            retain_losses.append((epoch + 1, avg_r))

            print(f"[SAUG] Epoch {epoch+1:02d} | "
                  f"npo={avg_n:.4f} | retain={avg_r:.4f} | "
                  f"relearn_loss={avg_re:.4f} | total={avg_t:.4f}")

        elapsed = time.time() - t0
        print(f"[SAUG] Co-training complete in {elapsed:.1f}s")

        return UnlearningResult(
            method="SAUG",
            epoch_losses=epoch_losses,
            forget_losses=forget_losses,
            retain_losses=retain_losses,
            total_steps=total_optimizer_steps,
            elapsed_sec=elapsed,
        )
