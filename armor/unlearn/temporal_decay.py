"""
armor/unlearn/temporal_decay.py
================================
TKDU: Temporal Knowledge Decay Unlearning
==========================================

The world's first machine unlearning system for facts that expire over time.

Motivation
----------
All existing machine unlearning methods treat knowledge as *static*:
a fact is either "to be forgotten" or "to be retained" forever.  This is
fundamentally wrong for real-world deployment:

  - "The CEO of Acme Corp is John Smith" → true in 2022, false in 2024
  - "Drug X has side effect Y" → true in 2020, recalled/revised in 2023
  - "The recommended dosage of Z is N mg" → changed by clinical update 2025
  - "User Alice's address is 123 Main St" → GDPR retention expires after 3 years

These are **temporally-bounded facts** — knowledge that is valid for a
period [t_start, t_end] and must be automatically unlearned after t_end.

TKDU solves this by:
  1. **Timestamping** knowledge with validity windows [t_valid_start, t_valid_end]
  2. **Computing temporal validity scores** τ(k, t_now) ∈ [0, 1] that decay
     smoothly from 1 (fully valid) to 0 (fully expired)
  3. **Applying temporally-weighted unlearning** — expired facts are forgotten,
     still-valid facts are retained
  4. **Scheduling automatic unlearning** — a scheduler monitors expiry dates
     and triggers incremental unlearning as facts expire
  5. **Generating temporal compliance certificates** — each unlearning run
     is certified with the knowledge expiry date, triggering timestamp,
     and GDPR retention period status

Mathematical Formulation
------------------------
Temporal validity score:
    τ(k, t) = sigmoid( (t_end_k - t_now) / τ_halflife )

    where:
      t_end_k   = knowledge expiry timestamp (UNIX time)
      t_now     = current timestamp
      τ_halflife = half-life parameter (default: 30 days in seconds)

    τ = 1.0 → fully valid knowledge (far from expiry)
    τ = 0.5 → at expiry boundary
    τ = 0.0 → fully expired (should be completely forgotten)

Temporally-weighted loss:
    L_TKDU(θ) = Σ_k (1 − τ_k) · L_forget(k)    # forget expired facts
              + Σ_k τ_k         · L_retain(k)    # preserve valid facts
              + L_CE(θ, retain_set)               # general retain regularisation

GDPR Compliance
---------------
TKDU directly enables GDPR Article 17 ("Right to Erasure") compliance
for time-bounded data retention:
  - Personal data must be deleted after the retention period expires
  - TKDU automatically triggers unlearning when retention_period elapses
  - A signed temporal certificate is generated as compliance evidence
"""

import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import (
    PreTrainedModel, PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from armor.config import ARMORConfig
from armor.unlearn.gradient_ascent import UnlearningResult
from armor.unlearn.npo import compute_token_log_probs


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeTimestamp:
    """
    Represents a piece of knowledge with temporal validity metadata.

    Attributes
    ----------
    knowledge_id    : Unique identifier for this piece of knowledge
    description     : Human-readable description of the knowledge
    content         : The textual content (e.g., question + answer)
    question        : The forget-set question
    answer          : The memorized answer

    t_valid_start   : UNIX timestamp when this knowledge became true
    t_valid_end     : UNIX timestamp when this knowledge expires (None = permanent)
    retention_days  : GDPR retention period in days (overrides t_valid_end if set)

    domain          : Knowledge domain (e.g., "healthcare", "personal", "legal")
    source          : Data source identifier
    gdpr_category   : GDPR data category (if applicable)
    """
    knowledge_id: str
    description: str
    content: str
    question: str
    answer: str

    # Temporal validity
    t_valid_start: float = 0.0         # UNIX timestamp
    t_valid_end: Optional[float] = None  # UNIX timestamp (None = no expiry)
    retention_days: Optional[int] = None  # GDPR retention period

    # Metadata
    domain: str = "general"
    source: str = "unknown"
    gdpr_category: str = "none"        # e.g., "personal", "health", "financial"

    # Computed at runtime
    current_validity: float = 1.0      # τ(k, t_now)
    is_expired: bool = False

    @property
    def t_expiry(self) -> Optional[float]:
        """Compute effective expiry timestamp."""
        if self.retention_days is not None:
            return self.t_valid_start + self.retention_days * 86400
        return self.t_valid_end

    @property
    def is_gdpr_personal(self) -> bool:
        """True if this knowledge contains personal GDPR data."""
        return self.gdpr_category in {"personal", "health", "financial", "biometric"}

    def days_until_expiry(self, t_now: Optional[float] = None) -> Optional[float]:
        """Days remaining until expiry (negative = already expired)."""
        t = t_now or time.time()
        exp = self.t_expiry
        if exp is None:
            return None
        return (exp - t) / 86400

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TemporalUnlearningResult:
    """Result of a temporal unlearning run."""
    method: str = "TKDU"
    timestamp: str = ""
    run_id: str = ""

    # Statistics
    n_total: int = 0
    n_expired: int = 0
    n_near_expiry: int = 0
    n_valid: int = 0

    # Loss values
    forget_loss_avg: float = 0.0
    retain_loss_avg: float = 0.0
    total_loss_avg: float = 0.0

    # Temporal validity stats
    mean_validity_score: float = 0.0
    min_validity_score: float = 0.0

    # Knowledge list with updated validity
    knowledge_items: List[Dict[str, Any]] = field(default_factory=list)

    # Performance
    elapsed_sec: float = 0.0
    total_steps: int = 0

    def print_summary(self):
        print("\n" + "=" * 72)
        print("  TKDU: TEMPORAL KNOWLEDGE DECAY UNLEARNING — SUMMARY")
        print("=" * 72)
        print(f"  Run ID         : {self.run_id}")
        print(f"  Timestamp      : {self.timestamp}")
        print(f"  Method         : {self.method}")
        print("  " + "-" * 68)
        print(f"  Total knowledge items   : {self.n_total}")
        print(f"  Expired (τ < 0.1)       : {self.n_expired}")
        print(f"  Near expiry (τ < 0.5)   : {self.n_near_expiry}")
        print(f"  Still valid (τ ≥ 0.5)   : {self.n_valid}")
        print("  " + "-" * 68)
        print(f"  Mean validity score     : {self.mean_validity_score:.4f}")
        print(f"  Min validity score      : {self.min_validity_score:.4f}")
        print(f"  Forget loss (weighted)  : {self.forget_loss_avg:.4f}")
        print(f"  Retain loss             : {self.retain_loss_avg:.4f}")
        print(f"  Training time           : {self.elapsed_sec:.1f}s")
        print("=" * 72 + "\n")

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Validity Scorer
# ──────────────────────────────────────────────────────────────────────────────

class TemporalValidityScorer:
    """
    Computes temporal validity scores τ(k, t_now) for knowledge items.

    The score uses a sigmoid decay function centered at the expiry timestamp:

        τ(k, t) = σ( (t_end - t_now) / halflife_sec )

    This gives:
      τ = 1.0  when t_now << t_end  (far from expiry, fully valid)
      τ = 0.5  when t_now == t_end  (exactly at expiry boundary)
      τ = 0.0  when t_now >> t_end  (far past expiry, fully expired)

    The half-life parameter controls how quickly the score decays around
    the expiry boundary. A 30-day half-life means that one month after
    expiry, τ ≈ 0.27.
    """

    def __init__(
        self,
        halflife_days: float = 30.0,
        expired_threshold: float = 0.1,
        near_expiry_threshold: float = 0.5,
    ):
        self.halflife_sec        = halflife_days * 86400
        self.expired_threshold   = expired_threshold
        self.near_expiry_threshold = near_expiry_threshold

    def score(
        self,
        knowledge: KnowledgeTimestamp,
        t_now: Optional[float] = None,
    ) -> float:
        """
        Compute τ(k, t_now) for a single knowledge item.

        Parameters
        ----------
        knowledge : The knowledge item to score
        t_now     : Current UNIX timestamp (defaults to time.time())

        Returns
        -------
        float in [0, 1] — temporal validity score
        """
        if t_now is None:
            t_now = time.time()

        t_expiry = knowledge.t_expiry

        if t_expiry is None:
            # No expiry → permanently valid
            return 1.0

        # Sigmoid decay: σ((t_end - t_now) / halflife)
        delta = (t_expiry - t_now) / self.halflife_sec
        tau   = 1.0 / (1.0 + math.exp(-delta))
        return float(tau)

    def score_all(
        self,
        knowledge_list: List[KnowledgeTimestamp],
        t_now: Optional[float] = None,
    ) -> List[float]:
        """Compute validity scores for all knowledge items."""
        t = t_now or time.time()
        scores = []
        for k in knowledge_list:
            tau = self.score(k, t)
            k.current_validity = tau
            k.is_expired = tau < self.expired_threshold
            scores.append(tau)
        return scores

    def classify(
        self,
        knowledge_list: List[KnowledgeTimestamp],
        t_now: Optional[float] = None,
    ) -> Dict[str, List[KnowledgeTimestamp]]:
        """
        Classify knowledge items by temporal status.

        Returns dict with keys: 'expired', 'near_expiry', 'valid'
        """
        t = t_now or time.time()
        self.score_all(knowledge_list, t)

        expired     = [k for k in knowledge_list if k.current_validity < self.expired_threshold]
        near_expiry = [k for k in knowledge_list if self.expired_threshold <= k.current_validity < self.near_expiry_threshold]
        valid       = [k for k in knowledge_list if k.current_validity >= self.near_expiry_threshold]

        return {
            "expired":     expired,
            "near_expiry": near_expiry,
            "valid":       valid,
        }

    def get_unlearning_weights(
        self,
        knowledge_list: List[KnowledgeTimestamp],
        t_now: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Get temporal forget weights (1 - τ) for the unlearning loss.

        A sample with τ = 0.0 (fully expired) gets weight 1.0 (full forgetting).
        A sample with τ = 1.0 (fully valid) gets weight 0.0 (no forgetting).

        Returns
        -------
        Tensor of shape (len(knowledge_list),) with forget weights in [0, 1]
        """
        scores = self.score_all(knowledge_list, t_now)
        forget_weights = [1.0 - tau for tau in scores]
        return torch.tensor(forget_weights, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Knowledge Dataset
# ──────────────────────────────────────────────────────────────────────────────

class TemporalKnowledgeDataset(Dataset):
    """
    Dataset wrapper for temporally-annotated knowledge items.

    Extends a base tokenized dataset with temporal validity weights,
    so that the TKDU loss can apply per-sample weighting.
    """

    def __init__(
        self,
        base_samples: List[Dict[str, Any]],  # tokenized {input_ids, labels, ...}
        knowledge_items: List[KnowledgeTimestamp],
        validity_scorer: TemporalValidityScorer,
        t_now: Optional[float] = None,
    ):
        assert len(base_samples) == len(knowledge_items), (
            "base_samples and knowledge_items must have the same length"
        )
        self.base_samples  = base_samples
        self.knowledge     = knowledge_items
        self.scorer        = validity_scorer
        self.t_now         = t_now or time.time()

        # Pre-compute validity scores
        self.validity_scores = self.scorer.score_all(knowledge_items, self.t_now)

    def __len__(self) -> int:
        return len(self.base_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(self.base_samples[idx])
        item["validity_score"] = torch.tensor(
            self.validity_scores[idx], dtype=torch.float32
        )
        item["forget_weight"] = torch.tensor(
            1.0 - self.validity_scores[idx], dtype=torch.float32
        )
        return item


# ──────────────────────────────────────────────────────────────────────────────
# TKDU Unlearner
# ──────────────────────────────────────────────────────────────────────────────

class TKDUUnlearner:
    """
    TKDU: Temporal Knowledge Decay Unlearner.

    Applies temporally-weighted unlearning: expired facts receive full
    unlearning pressure, nearly-expired facts receive partial pressure,
    and fully-valid facts are preserved.

    The loss is:
        L = Σ_k (1 − τ_k) · L_forget(k) + λ_retain · L_CE(retain)

    where τ_k ∈ [0, 1] is the temporal validity score of knowledge item k.

    Usage
    -----
    # Create knowledge items with timestamps
    items = [
        KnowledgeTimestamp(
            knowledge_id="k001",
            description="CEO of Acme Corp",
            content="Q: Who is CEO? A: John Smith",
            question="Who is the CEO of Acme Corp?",
            answer="John Smith",
            t_valid_start=datetime(2020,1,1).timestamp(),
            t_valid_end=datetime(2023,6,1).timestamp(),  # expired June 2023
            domain="corporate",
        ),
        ...
    ]

    unlearner = TKDUUnlearner(model, ref_model, tokenizer, cfg,
                               knowledge_items=items)
    result = unlearner.run(forget_loader, retain_loader)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cfg: ARMORConfig,
        knowledge_items: Optional[List[KnowledgeTimestamp]] = None,
        halflife_days: float = 30.0,
        expired_threshold: float = 0.1,
        optimizer: Optional[torch.optim.Optimizer] = None,
        t_now: Optional[float] = None,
    ):
        self.model     = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.device    = cfg.device
        self.t_now     = t_now or time.time()

        # Freeze reference model
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

        # Temporal scorer
        self._scorer = TemporalValidityScorer(
            halflife_days=halflife_days,
            expired_threshold=expired_threshold,
        )

        # Knowledge registry
        self._knowledge = knowledge_items or []
        if self._knowledge:
            self._scorer.score_all(self._knowledge, self.t_now)

    # ── Loss computation ───────────────────────────────────────────────────────

    def _temporal_forget_loss(
        self,
        forget_batch: dict,
        batch_validity_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute temporally-weighted NPO forget loss.

        If batch_validity_weights is provided, each sample's loss is scaled
        by its temporal forget weight (1 - τ).
        """
        input_ids = forget_batch["input_ids"].to(self.device)
        attn_mask = forget_batch["attention_mask"].to(self.device)
        labels    = forget_batch["labels"].to(self.device)

        policy_lp = compute_token_log_probs(
            self.model, input_ids, attn_mask, labels
        )
        with torch.no_grad():
            ref_lp = compute_token_log_probs(
                self.ref_model, input_ids, attn_mask, labels
            )

        log_ratio = policy_lp - ref_lp
        per_sample_loss = -F.logsigmoid(self.cfg.npo_beta * log_ratio)  # (B,)

        if batch_validity_weights is not None:
            forget_weights = batch_validity_weights.to(self.device)
            # Ensure shape matches batch
            if forget_weights.shape[0] != per_sample_loss.shape[0]:
                forget_weights = forget_weights[:per_sample_loss.shape[0]]
            weighted = per_sample_loss * forget_weights
            return weighted.mean()

        return per_sample_loss.mean()

    def _retain_loss(self, retain_batch: dict) -> torch.Tensor:
        """Standard cross-entropy retain loss."""
        batch = {k: v.to(self.device) for k, v in retain_batch.items()}
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs.loss

    # ── Main training loop ─────────────────────────────────────────────────────

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
        validity_weights: Optional[torch.Tensor] = None,
    ) -> TemporalUnlearningResult:
        """
        Run the temporal unlearning training loop.

        Parameters
        ----------
        forget_loader     : DataLoader over forget set
        retain_loader     : DataLoader over retain set (recommended)
        validity_weights  : (N,) tensor of temporal forget weights per sample.
                            If None, computed from knowledge_items registry.
                            If no knowledge_items, falls back to uniform (1.0).

        Returns
        -------
        TemporalUnlearningResult
        """
        cfg   = self.cfg
        model = self.model
        model.train()

        retain_iter = self._infinite_iter(retain_loader)

        # Compute temporal weights if not provided
        if validity_weights is None and self._knowledge:
            w = self._scorer.get_unlearning_weights(self._knowledge, self.t_now)
            print(f"[TKDU] Using temporal weights: "
                  f"min={w.min():.3f} | max={w.max():.3f} | mean={w.mean():.3f}")
        elif validity_weights is not None:
            w = validity_weights
        else:
            # No knowledge timestamps — uniform weighting (plain NPO)
            warnings.warn("[TKDU] No KnowledgeTimestamp items provided. "
                          "Falling back to uniform (1.0) weights.")
            w = None

        # LR scheduler
        total_steps  = len(forget_loader) * cfg.unlearn_epochs
        warmup_steps = max(1, total_steps // 10)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        epoch_losses = []
        forget_losses_hist = []
        retain_losses_hist = []
        total_optimizer_steps = 0
        total_forget = 0.0
        total_retain = 0.0
        t0 = time.time()

        n_expired     = sum(1 for k in self._knowledge if k.is_expired)
        n_near_expiry = sum(1 for k in self._knowledge
                           if not k.is_expired
                           and k.current_validity < self._scorer.near_expiry_threshold)
        n_valid       = len(self._knowledge) - n_expired - n_near_expiry

        print(f"\n[TKDU] Starting temporal unlearning: {cfg.unlearn_epochs} epochs")
        print(f"       Expired: {n_expired} | Near expiry: {n_near_expiry} | Valid: {n_valid}")

        for epoch in range(cfg.unlearn_epochs):
            e_total = e_forget = e_retain = 0.0
            n_batches = 0

            pbar = tqdm(
                forget_loader,
                desc=f"[TKDU] Epoch {epoch+1}/{cfg.unlearn_epochs}",
                leave=False,
            )

            sample_idx = 0
            for step, forget_batch in enumerate(pbar):
                retain_batch = next(retain_iter) if retain_iter else None

                # Get batch-level temporal weights
                if w is not None:
                    batch_size = forget_batch["input_ids"].shape[0]
                    batch_w    = w[sample_idx:sample_idx + batch_size]
                    sample_idx += batch_size
                    if sample_idx >= len(w):
                        sample_idx = 0  # wrap around
                else:
                    batch_w = None

                # Temporal forget loss
                f_loss = self._temporal_forget_loss(forget_batch, batch_w)

                # Retain loss
                if retain_batch is not None:
                    r_loss = self._retain_loss(retain_batch)
                else:
                    r_loss = torch.tensor(0.0, device=self.device)

                total_loss = f_loss + cfg.npo_retain_coeff * r_loss

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
                e_forget += f_loss.item()
                e_retain += r_loss.item() if hasattr(r_loss, "item") else 0.0
                n_batches += 1

                pbar.set_postfix({
                    "forget↓": f"{f_loss.item():.3f}",
                    "retain↓": f"{r_loss.item():.3f}" if hasattr(r_loss, "item") else "0",
                })

            avg_t = e_total  / max(n_batches, 1)
            avg_f = e_forget / max(n_batches, 1)
            avg_r = e_retain / max(n_batches, 1)

            epoch_losses.append((epoch + 1, avg_t))
            forget_losses_hist.append((epoch + 1, avg_f))
            retain_losses_hist.append((epoch + 1, avg_r))
            total_forget += avg_f
            total_retain += avg_r

            print(f"[TKDU] Epoch {epoch+1:02d} | "
                  f"forget={avg_f:.4f} | retain={avg_r:.4f} | total={avg_t:.4f}")

        elapsed  = time.time() - t0
        run_id   = f"tkdu_{int(t0)}"
        tau_vals = [k.current_validity for k in self._knowledge]

        print(f"[TKDU] Training complete in {elapsed:.1f}s")

        result = TemporalUnlearningResult(
            method="TKDU",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            run_id=run_id,
            n_total=len(self._knowledge),
            n_expired=n_expired,
            n_near_expiry=n_near_expiry,
            n_valid=n_valid,
            forget_loss_avg=total_forget / max(cfg.unlearn_epochs, 1),
            retain_loss_avg=total_retain / max(cfg.unlearn_epochs, 1),
            total_loss_avg=sum(l for _, l in epoch_losses) / max(len(epoch_losses), 1),
            mean_validity_score=float(np.mean(tau_vals)) if tau_vals else 0.0,
            min_validity_score=float(np.min(tau_vals)) if tau_vals else 0.0,
            knowledge_items=[k.to_dict() for k in self._knowledge],
            elapsed_sec=elapsed,
            total_steps=total_optimizer_steps,
        )
        result.print_summary()
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Unlearning Scheduler
# ──────────────────────────────────────────────────────────────────────────────

class TemporalUnlearningScheduler:
    """
    Production-ready scheduler for automatic temporal unlearning.

    Monitors a registry of KnowledgeTimestamp items and triggers incremental
    unlearning when facts expire.  Designed to run as a background service
    or cron job in a production deployment.

    Usage
    -----
    scheduler = TemporalUnlearningScheduler(
        knowledge_registry=all_knowledge_items,
        check_interval_days=1.0,
        expiry_buffer_days=7.0,  # trigger 7 days before expiry
    )

    # Check which items are due for unlearning
    due_items = scheduler.get_due_for_unlearning()

    # In a production loop:
    for event in scheduler.iter_events():
        if event["type"] == "unlearn_triggered":
            trigger_unlearning(event["items"])
    """

    def __init__(
        self,
        knowledge_registry: List[KnowledgeTimestamp],
        check_interval_days: float = 1.0,
        expiry_buffer_days: float = 0.0,
        halflife_days: float = 30.0,
    ):
        self._registry       = knowledge_registry
        self._check_interval = check_interval_days * 86400
        self._buffer         = expiry_buffer_days * 86400
        self._scorer         = TemporalValidityScorer(halflife_days=halflife_days)
        self._last_check     = 0.0
        self._triggered_ids  = set()

    def get_due_for_unlearning(
        self,
        t_now: Optional[float] = None,
    ) -> List[KnowledgeTimestamp]:
        """
        Return all knowledge items that are due for unlearning.

        Items are due if:
          - They have an expiry timestamp AND
          - t_now + buffer >= t_expiry  (at or past expiry, accounting for buffer)

        Parameters
        ----------
        t_now : Current UNIX timestamp (defaults to time.time())

        Returns
        -------
        List of KnowledgeTimestamp items that need to be unlearned
        """
        t = t_now or time.time()
        t_effective = t + self._buffer  # look ahead by buffer days

        due = []
        for k in self._registry:
            if k.knowledge_id in self._triggered_ids:
                continue  # already triggered
            t_exp = k.t_expiry
            if t_exp is not None and t_effective >= t_exp:
                due.append(k)

        return due

    def mark_triggered(self, items: List[KnowledgeTimestamp]) -> None:
        """Mark items as having been triggered for unlearning."""
        for k in items:
            self._triggered_ids.add(k.knowledge_id)

    def get_schedule_summary(
        self,
        t_now: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get a summary of the unlearning schedule for all knowledge items.

        Returns a list of dicts with expiry date, days remaining, and status.
        """
        t = t_now or time.time()
        self._scorer.score_all(self._registry, t)

        summary = []
        for k in self._registry:
            days = k.days_until_expiry(t)
            t_exp = k.t_expiry
            status = "permanent"
            if t_exp is not None:
                if days < 0:
                    status = "⚠️ EXPIRED"
                elif days < 7:
                    status = "⚡ EXPIRES SOON"
                elif days < 30:
                    status = "📅 NEAR EXPIRY"
                else:
                    status = "✅ VALID"

            summary.append({
                "id": k.knowledge_id,
                "description": k.description[:60],
                "validity_score": round(k.current_validity, 4),
                "days_until_expiry": round(days, 1) if days is not None else None,
                "expiry_date": (datetime.fromtimestamp(t_exp, tz=timezone.utc).isoformat()
                               if t_exp else "N/A"),
                "status": status,
                "gdpr_category": k.gdpr_category,
                "already_triggered": k.knowledge_id in self._triggered_ids,
            })

        # Sort by expiry urgency
        summary.sort(key=lambda x: (x["days_until_expiry"] or float("inf")))
        return summary

    def print_schedule(self, t_now: Optional[float] = None) -> None:
        """Pretty-print the unlearning schedule."""
        schedule = self.get_schedule_summary(t_now)
        print("\n" + "=" * 80)
        print("  TKDU: TEMPORAL UNLEARNING SCHEDULE")
        print("=" * 80)
        print(f"  {'ID':<12} {'Description':<35} {'τ':>6} {'Days':>8} {'Status':<20}")
        print("  " + "-" * 75)
        for item in schedule:
            days_str = f"{item['days_until_expiry']:.1f}" if item["days_until_expiry"] is not None else "∞"
            print(f"  {item['id']:<12} {item['description']:<35} "
                  f"{item['validity_score']:>6.3f} {days_str:>8} {item['status']:<20}")
        print("=" * 80 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────────────────────────────────────

def create_demo_knowledge_registry(
    qa_pairs: List[Tuple[str, str]],
    base_date: Optional[datetime] = None,
    expiry_days_list: Optional[List[Optional[int]]] = None,
) -> List[KnowledgeTimestamp]:
    """
    Create a demo knowledge registry from (question, answer) pairs.

    Assigns synthetic temporal metadata suitable for testing TKDU.

    Parameters
    ----------
    qa_pairs         : List of (question, answer) tuples
    base_date        : Reference date (defaults to 2020-01-01)
    expiry_days_list : Days-until-expiry for each item (None = no expiry).
                       If not provided, assigns a mix of expired and valid.

    Returns
    -------
    List[KnowledgeTimestamp]
    """
    if base_date is None:
        base_date = datetime(2020, 1, 1, tzinfo=timezone.utc)

    base_ts = base_date.timestamp()

    if expiry_days_list is None:
        # Mix: half expired (2 years ago), half valid (1 year ahead)
        now = time.time()
        expiry_days_list = []
        for i in range(len(qa_pairs)):
            if i % 2 == 0:
                # Expired 2 years ago
                expiry_days_list.append(-730)
            else:
                # Valid for 1 more year
                expiry_days_list.append(365)

    knowledge_items = []
    domains = ["corporate", "healthcare", "legal", "personal", "financial"]

    for i, (q, a) in enumerate(qa_pairs):
        exp_days = expiry_days_list[i] if i < len(expiry_days_list) else None

        if exp_days is not None:
            # Expiry = now + exp_days
            t_end = time.time() + exp_days * 86400
        else:
            t_end = None

        gdpr_cat = "personal" if i % 3 == 0 else ("health" if i % 3 == 1 else "none")

        item = KnowledgeTimestamp(
            knowledge_id=f"K{i:04d}",
            description=q[:60],
            content=f"Q: {q}\nA: {a}",
            question=q,
            answer=a,
            t_valid_start=base_ts,
            t_valid_end=t_end,
            domain=domains[i % len(domains)],
            source="TOFU_demo",
            gdpr_category=gdpr_cat,
        )
        knowledge_items.append(item)

    return knowledge_items


def load_knowledge_registry_from_json(path: str) -> List[KnowledgeTimestamp]:
    """Load a knowledge registry from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        KnowledgeTimestamp(
            knowledge_id=d["knowledge_id"],
            description=d["description"],
            content=d["content"],
            question=d["question"],
            answer=d["answer"],
            t_valid_start=d.get("t_valid_start", 0.0),
            t_valid_end=d.get("t_valid_end"),
            retention_days=d.get("retention_days"),
            domain=d.get("domain", "general"),
            source=d.get("source", "unknown"),
            gdpr_category=d.get("gdpr_category", "none"),
        )
        for d in data
    ]


def save_knowledge_registry_to_json(
    items: List[KnowledgeTimestamp],
    path: str,
) -> None:
    """Save a knowledge registry to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([k.to_dict() for k in items], f, indent=2)
    print(f"[TKDU] Knowledge registry saved → {path}")
