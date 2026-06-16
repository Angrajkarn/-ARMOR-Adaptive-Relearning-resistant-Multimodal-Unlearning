"""
armor/attack/federated_attack.py
=================================
Federated Relearning Attack — distributed adversarial recovery.

The strongest attack: K independent clients each hold a fraction of the
forget set, fine-tune locally, and aggregate via FedAvg.

This simulates a realistic threat where:
  - No single attacker has the full forget set
  - Aggregated gradients are harder to detect / filter
  - Tests whether ARMOR's flat minima survive gradient averaging

Protocol:
  1. Split forget set into K equal partitions (one per client)
  2. Each client fine-tunes the unlearned model for E_local epochs
  3. Collect per-client weight deltas: Δθ_k = θ_k - θ_init
  4. FedAvg: θ_agg = θ_init + (1/K) * Σ_k Δθ_k
  5. Evaluate forget accuracy on aggregated model
"""

import copy
import time
import torch
from dataclasses import dataclass, field
from typing import List, Tuple
from torch.utils.data import DataLoader, Subset
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..config import ARMORConfig
from ..eval.metrics import compute_token_accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FederatedAttackResult:
    method:               str
    n_clients:            int
    original_forget_acc:  float
    pre_attack_forget_acc: float = 0.0
    final_forget_acc:     float = 0.0
    acc_jump:             float = 0.0
    recovery_pct:         float = float("nan")
    elapsed_sec:          float = 0.0
    per_client_acc: List[float] = field(default_factory=list)

    def compute(self):
        self.acc_jump = self.final_forget_acc - self.pre_attack_forget_acc
        denom = self.original_forget_acc - self.pre_attack_forget_acc
        if denom > 1e-4:
            self.recovery_pct = max(0.0, min(100.0,
                                  (self.acc_jump / denom) * 100.0))

    def print_summary(self):
        print("\n" + "=" * 64)
        print(f"  FEDERATED ATTACK -- Method: {self.method} | Clients: {self.n_clients}")
        print("=" * 64)
        print(f"  Pre-attack forget accuracy  : {self.pre_attack_forget_acc:.4f}")
        print(f"  Post-FedAvg forget accuracy : {self.final_forget_acc:.4f}")
        print(f"  Accuracy jump               : {self.acc_jump:+.4f}")
        rstr = f"{self.recovery_pct:.1f}%" if self.recovery_pct == self.recovery_pct \
               else "N/A"
        print(f"  Knowledge recovery          : {rstr}")
        print(f"  Wall time                   : {self.elapsed_sec:.1f}s")
        print("-" * 64)
        print("  Per-client forget acc after local training:")
        for i, acc in enumerate(self.per_client_acc, 1):
            print(f"    Client {i}: {acc:.4f}")
        print("=" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# FedAvg Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def fedavg(global_model: PreTrainedModel,
           client_models: List[PreTrainedModel]) -> PreTrainedModel:
    """
    Aggregate client models using FedAvg:
        θ_agg = (1/K) * Σ_k θ_k
    Modifies global_model in-place and returns it.
    """
    n = len(client_models)
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            # Average weights across all clients
            avg_weight = torch.stack(
                [dict(m.named_parameters())[name].data for m in client_models]
            ).mean(dim=0)
            param.data.copy_(avg_weight)
    return global_model


# ─────────────────────────────────────────────────────────────────────────────
# Main Attack Class
# ─────────────────────────────────────────────────────────────────────────────

class FederatedRelearningAttack:
    """
    Simulates a federated learning-based relearning attack.

    Usage:
        attack = FederatedRelearningAttack(cfg, model, tokenizer,
                                            forget_dataset, retain_loader,
                                            n_clients=3)
        result = attack.run(method_name="NPO+SAM", original_acc=0.85)
    """

    def __init__(self,
                 cfg:            ARMORConfig,
                 model:          PreTrainedModel,
                 tokenizer:      PreTrainedTokenizer,
                 forget_dataset,               # torch Dataset (not DataLoader)
                 retain_loader:  DataLoader,
                 n_clients:      int   = 3,
                 local_epochs:   int   = 2,
                 local_lr:       float = 2e-5):
        self.cfg            = cfg
        self.model          = model
        self.tokenizer      = tokenizer
        self.forget_dataset = forget_dataset
        self.retain_loader  = retain_loader
        self.n_clients      = n_clients
        self.local_epochs   = local_epochs
        self.local_lr       = local_lr

    def _make_client_loaders(self) -> List[DataLoader]:
        """Split forget dataset into K equal partitions."""
        n     = len(self.forget_dataset)
        chunk = max(1, n // self.n_clients)
        loaders = []
        for i in range(self.n_clients):
            start   = i * chunk
            end     = start + chunk if i < self.n_clients - 1 else n
            indices = list(range(start, end))
            subset  = Subset(self.forget_dataset, indices)
            loaders.append(DataLoader(subset,
                                       batch_size=self.cfg.batch_size,
                                       shuffle=True))
        return loaders

    def _local_train(self, client_model: PreTrainedModel,
                     client_loader: DataLoader) -> PreTrainedModel:
        """Fine-tune a single client on its local forget partition."""
        client_model.train()
        optimizer = torch.optim.AdamW(
            client_model.parameters(), lr=self.local_lr)

        for _ in range(self.local_epochs):
            for batch in client_loader:
                input_ids = batch["input_ids"].to(self.cfg.device)
                labels    = batch["labels"].to(self.cfg.device)
                mask      = batch.get("attention_mask",
                                      torch.ones_like(input_ids)).to(self.cfg.device)
                optimizer.zero_grad()
                out  = client_model(input_ids=input_ids,
                                    attention_mask=mask,
                                    labels=labels)
                out.loss.backward()
                optimizer.step()
        return client_model

    def run(self, method_name: str = "unknown",
            original_acc: float = 0.5) -> FederatedAttackResult:
        result = FederatedAttackResult(
            method=method_name,
            n_clients=self.n_clients,
            original_forget_acc=original_acc)

        # ── Full forget loader for evaluation ──────────────────────────────
        full_forget_loader = DataLoader(
            self.forget_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False)

        # ── Pre-attack baseline ────────────────────────────────────────────
        self.model.eval()
        result.pre_attack_forget_acc = compute_token_accuracy(
            self.model, full_forget_loader, self.cfg.device)
        print(f"\n[Fed Attack] Pre-attack forget acc : {result.pre_attack_forget_acc:.4f}")
        print(f"[Fed Attack] Simulating {self.n_clients} clients, "
              f"{self.local_epochs} local epochs each...")

        t0            = time.time()
        client_loaders = self._make_client_loaders()
        client_models  = []

        # ── Local training ─────────────────────────────────────────────────
        for k, loader in enumerate(client_loaders, 1):
            client_model = copy.deepcopy(self.model)
            client_model = self._local_train(client_model, loader)
            client_model.eval()

            # Measure this client's local forget acc
            local_acc = compute_token_accuracy(
                client_model, full_forget_loader, self.cfg.device)
            result.per_client_acc.append(local_acc)
            client_models.append(client_model)
            print(f"[Fed Attack] Client {k}/{self.n_clients} "
                  f"| local forget acc={local_acc:.4f}")

        # ── FedAvg aggregation ─────────────────────────────────────────────
        print("[Fed Attack] Running FedAvg aggregation...")
        aggregated = copy.deepcopy(self.model)
        aggregated = fedavg(aggregated, client_models)
        aggregated.eval()

        result.final_forget_acc = compute_token_accuracy(
            aggregated, full_forget_loader, self.cfg.device)
        result.elapsed_sec = time.time() - t0
        result.compute()
        return result
