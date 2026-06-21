"""
armor/unlearn/cot_hme.py
========================
CoT-HME: Chain-of-Thought Hidden Memory Erasure — Unlearning Module
====================================================================

This module implements the CoT-HME unlearning loss that suppresses
*reasoning-level* leakage of forgotten knowledge in addition to the
standard output-level unlearning (NPO/GA).

Background
----------
Standard unlearning methods (NPO, GA, RMU) optimize the model to NOT
produce the memorized answer as a final output token sequence.  However,
the model's chain-of-thought (CoT) reasoning traces can still reconstruct
the forbidden knowledge via multi-step inference, even when the final
answer token is suppressed.

CoT-HME adds a **reasoning trace erasure loss** that:
  1. Forces the model to generate CoT traces for forget-set questions
  2. Identifies trace tokens that "leak" forbidden knowledge (via step scores)
  3. Applies a KL-divergence loss to push leaked reasoning steps toward a
     uniform / low-confidence distribution
  4. Combines this CoT loss with the standard NPO loss

Loss formulation:
    L_CoT-HME = L_NPO(θ, forget)                   # standard output suppression
              + λ_retain · L_CE(θ, retain)           # retain preservation
              + λ_cot    · L_CoT(θ, forget, cot)     # CoT trace suppression

Where:
    L_CoT(θ, forget, cot) = Σ_{t∈leaked_steps} leakage_score_t
                            · KL(P_θ(·|context_t) || Uniform)
                          = Σ_{t∈leaked_steps} leakage_score_t
                            · (-H[P_θ(·|context_t)] + log|V|)
                          ≡ Σ_{t∈leaked_steps} leakage_score_t
                            · (-entropy of the distribution at t)

Maximizing entropy at leaked positions = pushing the model toward maximum
uncertainty at those exact reasoning steps, preventing the model from
"reasoning through" to the forbidden answer.

Integration
-----------
CoTHMEUnlearner is designed as a drop-in wrapper around any base unlearner
(NPO, GA, etc.).  It augments the base loss with the CoT suppression term
without modifying the base unlearner's logic.
"""

import time
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel, PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from armor.config import ARMORConfig
from armor.unlearn.gradient_ascent import UnlearningResult
from armor.unlearn.npo import NPOUnlearner, compute_token_log_probs
from armor.attack.cot_leakage_probe import (
    CoTLeakageProbe, CoTLeakageResult, CoTStep,
    build_cot_prompt, segment_cot_trace,
)


# ──────────────────────────────────────────────────────────────────────────────
# CoT Entropy Loss
# ──────────────────────────────────────────────────────────────────────────────

def compute_cot_entropy_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    leaked_steps: List[CoTStep],
    question: str,
    device: str,
    max_step_tokens: int = 48,
) -> torch.Tensor:
    """
    Compute the CoT entropy suppression loss for leaked reasoning steps.

    For each leaked reasoning step, this loss:
      - Encodes the full context up to that step
      - Computes the model's output distribution at the start of the step
      - Maximizes entropy of that distribution (pushes toward uniform)

    This prevents the model from "thinking through" to the forbidden answer
    by making it maximally uncertain at the leaked reasoning positions.

    Parameters
    ----------
    model        : The trainable model
    tokenizer    : The tokenizer
    leaked_steps : Steps identified as leaking forbidden knowledge
    question     : The forget-set question (for context building)
    device       : Compute device
    max_step_tokens : Max tokens per step to process

    Returns
    -------
    loss : Scalar tensor (mean entropy maximization loss over leaked steps)
    """
    if not leaked_steps:
        return torch.tensor(0.0, device=device, requires_grad=True)

    model.train()
    step_losses: List[torch.Tensor] = []

    # Build the CoT context prefix up to the first leaked step
    prefix = build_cot_prompt(question, few_shot=False)

    for step in leaked_steps:
        step_text = step.text[:max_step_tokens * 4]  # rough char limit

        # Build context: question prefix + step text
        context = prefix + step_text
        enc = tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        input_ids = enc["input_ids"].to(device)

        if input_ids.shape[-1] < 2:
            continue

        # Forward pass
        outputs = model(input_ids=input_ids)
        logits  = outputs.logits  # (1, T, V)

        # Take the logits at the last position (where the step begins)
        last_logits = logits[0, -1, :]  # (V,)

        # Maximize entropy = minimize negative entropy = minimize -H[P]
        # H[P] = -Σ p_i log p_i
        log_probs = F.log_softmax(last_logits, dim=-1)
        probs     = log_probs.exp()
        entropy   = -(probs * log_probs).sum()  # scalar

        # Weight by leakage score (more leaked = more suppression)
        weighted_loss = step.leakage_score * (-entropy)  # maximize entropy = minimize negative entropy
        step_losses.append(weighted_loss)

    if not step_losses:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return torch.stack(step_losses).mean()


# ──────────────────────────────────────────────────────────────────────────────
# CoT-HME Sample Bank
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CoTForgetSample:
    """A forget-set sample with pre-probed CoT leakage information."""
    question: str
    answer: str
    leaked_steps: List[CoTStep] = field(default_factory=list)
    trace_leakage_score: float = 0.0
    probed: bool = False


class CoTSampleBank:
    """
    Maintains a bank of forget-set samples with their CoT leakage information.

    Pre-probes the forget set before training to identify which samples
    have leaked CoT traces, then provides them during the training loop.

    The bank also supports lazy re-probing: if a sample's leakage score
    drops below a threshold, it is re-probed to get fresh leakage signals.
    """

    def __init__(
        self,
        probe: CoTLeakageProbe,
        reprobe_threshold: float = 0.1,
        reprobe_interval: int = 2,
    ):
        self.probe             = probe
        self.reprobe_threshold = reprobe_threshold
        self.reprobe_interval  = reprobe_interval

        self._samples: List[CoTForgetSample] = []
        self._epoch = 0

    def load_qa_pairs(self, qa_pairs: List[Tuple[str, str]]) -> None:
        """Load (question, answer) pairs into the bank."""
        self._samples = [
            CoTForgetSample(question=q, answer=a)
            for q, a in qa_pairs
        ]

    def probe_all(self, max_samples: Optional[int] = None) -> None:
        """Pre-probe all samples in the bank."""
        samples = self._samples[:max_samples] if max_samples else self._samples
        print(f"[CoT-HME] Pre-probing {len(samples)} forget-set samples...")

        for sample in tqdm(samples, desc="  [CoT-HME] Pre-probing", leave=False):
            result = self.probe.probe_sample(sample.question, sample.answer)
            sample.leaked_steps       = [s for s in result.steps if s.is_leaked]
            sample.trace_leakage_score = result.trace_level_score
            sample.probed              = True

        n_leaked = sum(1 for s in samples if s.trace_leakage_score > 0.1)
        print(f"[CoT-HME] Pre-probe complete: "
              f"{n_leaked}/{len(samples)} samples have CoT leakage")

    def on_epoch_end(self, epoch: int, reprobe_n: int = 4) -> None:
        """Re-probe a subset of samples at epoch end to refresh leakage signals."""
        self._epoch = epoch
        if epoch % self.reprobe_interval == 0 and self._samples:
            # Re-probe random subset
            subset = random.sample(
                self._samples, min(reprobe_n, len(self._samples))
            )
            for sample in subset:
                result = self.probe.probe_sample(sample.question, sample.answer)
                sample.leaked_steps        = [s for s in result.steps if s.is_leaked]
                sample.trace_leakage_score = result.trace_level_score

    def get_leaked_samples(self) -> List[CoTForgetSample]:
        """Return all samples with at least one leaked CoT step."""
        return [s for s in self._samples if s.leaked_steps]

    def __len__(self) -> int:
        return len(self._samples)


# ──────────────────────────────────────────────────────────────────────────────
# CoT-HME Unlearner
# ──────────────────────────────────────────────────────────────────────────────

class CoTHMEUnlearner:
    """
    CoT-HME: Chain-of-Thought Hidden Memory Erasure Unlearner.

    Wraps NPO unlearning with an additional CoT entropy suppression loss
    that targets reasoning-level leakage of forgotten knowledge.

    Loss:
        L = L_NPO(forget) + λ_retain · L_CE(retain) + λ_cot · L_CoT(cot_leaked)

    Architecture:
        - Base unlearner: NPO (configurable)
        - CoT probe: CoTLeakageProbe (runs during training for live leakage detection)
        - CoT loss: entropy maximization at leaked reasoning positions
        - Sample bank: pre-computed leakage info, periodically refreshed

    Usage
    -----
    ref_model = get_frozen_reference_model(model, cfg)
    unlearner = CoTHMEUnlearner(
        model, ref_model, tokenizer, cfg,
        qa_forget_pairs=[("Q1", "A1"), ("Q2", "A2")],
    )
    result = unlearner.run(forget_loader, retain_loader)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cfg: ARMORConfig,
        qa_forget_pairs: Optional[List[Tuple[str, str]]] = None,
        cot_loss_coeff: float = 0.3,
        cot_leak_threshold: float = 0.3,
        cot_max_new_tokens: int = 64,
        cot_probe_batch: int = 4,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model     = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.device    = cfg.device

        self.cot_coeff     = cot_loss_coeff
        self.cot_threshold = cot_leak_threshold

        # Ensure reference model is frozen
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

        # Build CoT probe
        self._probe = CoTLeakageProbe(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            leakage_threshold=cot_leak_threshold,
            max_new_tokens=cot_max_new_tokens,
            few_shot=False,  # Disable few-shot in debug mode for speed
        )

        # Build sample bank
        self._bank = CoTSampleBank(self._probe, reprobe_threshold=0.05)
        if qa_forget_pairs:
            self._bank.load_qa_pairs(qa_forget_pairs)
            # Pre-probe up to cot_probe_batch samples per epoch in debug
            n_probe = cot_probe_batch if cfg.debug else len(qa_forget_pairs)
            self._bank.probe_all(max_samples=n_probe)

    # ── Loss computation ───────────────────────────────────────────────────────

    def _npo_forget_loss(self, forget_batch: dict) -> torch.Tensor:
        """Standard NPO loss on the forget batch."""
        input_ids = forget_batch["input_ids"].to(self.device)
        attn_mask = forget_batch["attention_mask"].to(self.device)
        labels    = forget_batch["labels"].to(self.device)

        policy_lp = compute_token_log_probs(
            self.model, input_ids, attn_mask, labels
        )
        with torch.no_grad():
            if self.ref_model is self.model:
                with self.model.disable_adapter():
                    ref_lp = compute_token_log_probs(
                        self.model, input_ids, attn_mask, labels
                    )
            else:
                ref_lp = compute_token_log_probs(
                    self.ref_model, input_ids, attn_mask, labels
                )

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

    def _cot_loss(self, step: int) -> torch.Tensor:
        """
        Compute CoT entropy suppression loss over pre-probed leaked samples.

        Selects a random subset of leaked samples for efficiency, then
        computes the entropy maximization loss at leaked reasoning positions.
        """
        leaked_samples = self._bank.get_leaked_samples()
        if not leaked_samples:
            return torch.tensor(0.0, device=self.device)

        # Pick a random sample to compute CoT loss on this step
        sample = random.choice(leaked_samples)

        if not sample.leaked_steps:
            return torch.tensor(0.0, device=self.device)

        return compute_cot_entropy_loss(
            model=self.model,
            tokenizer=self.tokenizer,
            leaked_steps=sample.leaked_steps,
            question=sample.question,
            device=self.device,
        )

    # ── Infinite iterator helper ───────────────────────────────────────────────

    @staticmethod
    def _infinite_iter(loader: Optional[DataLoader]):
        if loader is None:
            return None
        while True:
            yield from loader

    # ── Main training loop ─────────────────────────────────────────────────────

    def run(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
    ) -> UnlearningResult:
        """
        Run the full CoT-HME unlearning loop.

        Parameters
        ----------
        forget_loader : DataLoader — forget set batches
        retain_loader : DataLoader — retain set batches (optional but recommended)

        Returns
        -------
        UnlearningResult with full loss history
        """
        cfg   = self.cfg
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

        epoch_losses, forget_losses, retain_losses, cot_losses = [], [], [], []
        total_optimizer_steps = 0
        t0 = time.time()

        print(f"\n[CoT-HME] Starting training: {cfg.unlearn_epochs} epochs")
        print(f"          CoT loss coefficient: {self.cot_coeff:.3f}")
        print(f"          Pre-probed samples with leakage: {len(self._bank.get_leaked_samples())}")

        for epoch in range(cfg.unlearn_epochs):
            e_total = e_npo = e_retain = e_cot = 0.0
            n_batches = 0

            pbar = tqdm(
                forget_loader,
                desc=f"[CoT-HME] Epoch {epoch+1}/{cfg.unlearn_epochs}",
                leave=False,
            )

            for step, forget_batch in enumerate(pbar):
                retain_batch = next(retain_iter) if retain_iter else None

                # 1. NPO forget loss
                npo_loss = self._npo_forget_loss(forget_batch)

                # 2. Retain loss
                if retain_batch is not None:
                    r_loss = self._retain_loss(retain_batch)
                else:
                    r_loss = torch.tensor(0.0, device=self.device)

                # 3. CoT entropy suppression loss
                cot_loss = self._cot_loss(step)

                # Combined loss
                total_loss = (
                    npo_loss
                    + cfg.npo_retain_coeff * r_loss
                    + self.cot_coeff * cot_loss
                )

                # Backward + gradient accumulation
                scaled = total_loss / cfg.gradient_accumulation_steps
                scaled.backward()

                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.max_grad_norm
                    )
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    total_optimizer_steps += 1

                e_total  += total_loss.item()
                e_npo    += npo_loss.item()
                e_retain += r_loss.item() if hasattr(r_loss, "item") else 0.0
                e_cot    += cot_loss.item() if hasattr(cot_loss, "item") else 0.0
                n_batches += 1

                pbar.set_postfix({
                    "npo":    f"{npo_loss.item():.3f}",
                    "retain": f"{r_loss.item():.3f}" if hasattr(r_loss, "item") else "0",
                    "cot":    f"{cot_loss.item():.3f}" if hasattr(cot_loss, "item") else "0",
                })

            avg_t = e_total  / max(n_batches, 1)
            avg_n = e_npo    / max(n_batches, 1)
            avg_r = e_retain / max(n_batches, 1)
            avg_c = e_cot    / max(n_batches, 1)

            epoch_losses.append((epoch + 1, avg_t))
            forget_losses.append((epoch + 1, avg_n))
            retain_losses.append((epoch + 1, avg_r))
            cot_losses.append((epoch + 1, avg_c))

            print(f"[CoT-HME] Epoch {epoch+1:02d} | "
                  f"npo={avg_n:.4f} | retain={avg_r:.4f} | "
                  f"cot={avg_c:.4f} | total={avg_t:.4f}")

            # Re-probe subset at end of each epoch
            self._bank.on_epoch_end(epoch, reprobe_n=2)

        elapsed = time.time() - t0
        print(f"[CoT-HME] Training complete in {elapsed:.1f}s "
              f"({total_optimizer_steps} optimizer steps)")

        return UnlearningResult(
            method="CoT-HME",
            epoch_losses=epoch_losses,
            forget_losses=forget_losses,
            retain_losses=retain_losses,
            total_steps=total_optimizer_steps,
            elapsed_sec=elapsed,
        )
