"""
armor/unlearn/federated_robust.py
=================================
BRFU: Byzantine-Robust Federated Unlearning

This module implements a simulated federated learning system for machine unlearning
where multiple clients perform local unlearning, and a server aggregates their updates
using Byzantine-robust protocols (Krum or Coordinate-wise Trimmed Mean) to withstand
malicious gradient poisoning attacks.
"""

import time
import copy
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import PreTrainedModel, PreTrainedTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from armor.config import ARMORConfig
from armor.unlearn.gradient_ascent import UnlearningResult
from armor.unlearn.npo import compute_token_log_probs

class BRFUUnlearner:
    """
    BRFUUnlearner: Simulates Byzantine-Robust Federated Unlearning.

    - Partitions the forget and retain datasets among multiple clients.
    - Simulates Byzantine clients that inject poisoned gradients.
    - Aggregates updates at the server using robust algorithms (Krum, Trimmed Mean).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cfg: ARMORConfig,
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

        # Federated parameters
        self.num_clients = cfg.brfu_num_clients
        self.byzantine_frac = cfg.brfu_byzantine_frac
        self.num_byzantine = int(round(self.num_clients * self.byzantine_frac))
        self.aggregation = cfg.brfu_aggregation.lower()

        # Server optimizer
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.unlearn_lr,
            weight_decay=cfg.weight_decay,
        )

    def _partition_dataset(self, dataset: Any, num_partitions: int) -> List[Subset]:
        """Partitions a dataset equally among clients."""
        total_len = len(dataset)
        indices = list(range(total_len))
        partition_size = total_len // num_partitions
        
        partitions = []
        for i in range(num_partitions):
            start = i * partition_size
            end = total_len if i == num_partitions - 1 else (i + 1) * partition_size
            partitions.append(Subset(dataset, indices[start:end]))
        return partitions

    def _compute_local_gradient(
        self,
        client_model: PreTrainedModel,
        forget_batch: dict,
        retain_batch: Optional[dict],
    ) -> Dict[str, torch.Tensor]:
        """Computes gradients for a client on a single batch step."""
        client_model.zero_grad()

        # 1. Forget set NPO loss
        f_input_ids = forget_batch["input_ids"].to(self.device)
        f_attn_mask = forget_batch["attention_mask"].to(self.device)
        f_labels = forget_batch["labels"].to(self.device)

        policy_lp = compute_token_log_probs(client_model, f_input_ids, f_attn_mask, f_labels)
        with torch.no_grad():
            if self.ref_model is self.model:
                with self.model.disable_adapter():
                    ref_lp = compute_token_log_probs(self.model, f_input_ids, f_attn_mask, f_labels)
            else:
                ref_lp = compute_token_log_probs(self.ref_model, f_input_ids, f_attn_mask, f_labels)

        log_ratio = policy_lp - ref_lp
        npo_loss = -F.logsigmoid(self.cfg.npo_beta * log_ratio).mean()

        # 2. Retain set loss
        if retain_batch is not None:
            r_batch = {k: v.to(self.device) for k, v in retain_batch.items()}
            outputs = client_model(
                input_ids=r_batch["input_ids"],
                attention_mask=r_batch["attention_mask"],
                labels=r_batch["labels"],
            )
            r_loss = outputs.loss
        else:
            r_loss = torch.tensor(0.0, device=self.device)

        total_loss = npo_loss + self.cfg.npo_retain_coeff * r_loss
        total_loss.backward()

        # Extract gradients
        grads = {}
        for name, param in client_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads[name] = param.grad.clone().detach()
        return grads

    def _apply_byzantine_attack(self, grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Simulates a Byzantine client injecting poisoned gradients to degrade the global model."""
        poisoned_grads = {}
        for name, g in grads.items():
            # Standard sign inversion + scale amplification + high variance Gaussian noise
            poisoned_grads[name] = g * -8.0 + torch.randn_like(g) * 0.15
        return poisoned_grads

    def _aggregate_krum(self, client_grads_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Krum Byzantine-Robust Aggregation.
        Selects the client gradient vector that minimizes the sum of Euclidean distances
        to its nearest neighbors.
        """
        M = len(client_grads_list)
        f = self.num_byzantine
        # Krum requires at least M - f - 2 neighbors
        k = max(1, M - f - 2)

        # Flatten gradients into 1D vectors for distance calculation
        flat_vectors = []
        param_names = sorted(client_grads_list[0].keys())

        for grads in client_grads_list:
            flat = torch.cat([grads[name].flatten() for name in param_names])
            flat_vectors.append(flat)
        
        flat_vectors = torch.stack(flat_vectors) # (M, D)

        # Compute pairwise distance matrix
        dists = torch.cdist(flat_vectors, flat_vectors, p=2) # (M, M)

        # For each client, sum the distances to its k-nearest neighbors (excluding itself)
        scores = []
        for i in range(M):
            client_dists = dists[i]
            # Sort distances
            sorted_dists, _ = torch.sort(client_dists)
            # Take index 1 to k+1 (excluding index 0 which is distance to itself, i.e., 0)
            score = sorted_dists[1:k+1].sum().item()
            scores.append(score)

        # Find client with the lowest score
        best_client_idx = scores.index(min(scores))
        return client_grads_list[best_client_idx]

    def _aggregate_trimmed_mean(self, client_grads_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Coordinate-wise Trimmed Mean Byzantine-Robust Aggregation.
        Sorts the gradients coordinate-wise across all clients and removes the top and bottom f values.
        """
        M = len(client_grads_list)
        f = self.num_byzantine
        param_names = client_grads_list[0].keys()

        aggregated_grads = {}
        for name in param_names:
            # Stack along client dimension
            stacked = torch.stack([grads[name] for grads in client_grads_list], dim=0) # (M, ...)
            
            # Sort along client dimension
            sorted_stacked, _ = torch.sort(stacked, dim=0)
            
            # Trim top and bottom f elements
            # If f is 0, we take the entire range
            if f > 0 and M > 2 * f:
                trimmed = sorted_stacked[f:-f]
            else:
                trimmed = sorted_stacked
            
            # Average the remaining
            aggregated_grads[name] = trimmed.mean(dim=0)

        return aggregated_grads

    @staticmethod
    def _infinite_iter(loader):
        while True:
            for batch in loader:
                yield batch

    def run(
        self,
        forget_dataset: Any,
        retain_dataset: Optional[Any] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> UnlearningResult:
        """Runs the Byzantine-Robust Federated Unlearning simulation loop."""
        cfg = self.cfg
        model = self.model
        
        # Partition datasets
        forget_partitions = self._partition_dataset(forget_dataset, self.num_clients)
        if retain_dataset is not None:
            retain_partitions = self._partition_dataset(retain_dataset, self.num_clients)
        else:
            retain_partitions = [None] * self.num_clients

        # Build dataloaders for clients
        from armor.data import make_dataloader
        forget_loaders = [
            make_dataloader(sub, tokenizer or self.tokenizer, cfg, include_rephrases=True, shuffle=True)
            for sub in forget_partitions
        ]
        retain_loaders = [
            make_dataloader(sub, tokenizer or self.tokenizer, cfg, shuffle=True) if sub is not None else None
            for sub in retain_partitions
        ]

        # Determine step count based on the smallest forget loader
        steps_per_epoch = min(len(loader) for loader in forget_loaders)
        total_steps_count = steps_per_epoch * cfg.unlearn_epochs
        warmup_steps = max(1, total_steps_count // 10)
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps_count,
        )

        epoch_losses, forget_losses, retain_losses = [], [], []
        total_optimizer_steps = 0
        t0 = time.time()

        print(f"\n[BRFU] Starting Byzantine-Robust Federated Unlearning: {cfg.unlearn_epochs} epochs")
        print(f"       Clients: {self.num_clients} (Byzantine: {self.num_byzantine})")
        print(f"       Server Aggregation: {self.aggregation}")

        # Instantiate a client-side temporary model clone to perform local gradient computation
        client_model = copy.deepcopy(model)

        for epoch in range(cfg.unlearn_epochs):
            e_total = 0.0
            n_batches = 0

            # Make iterators for each loader (wrapped to cycle infinitely)
            f_iters = [self._infinite_iter(loader) for loader in forget_loaders]
            r_iters = [self._infinite_iter(loader) if loader is not None else None for loader in retain_loaders]

            pbar = tqdm(
                range(steps_per_epoch),
                desc=f"[BRFU] Epoch {epoch+1}/{cfg.unlearn_epochs}",
                leave=False,
            )

            for step in pbar:
                # Sync client model with the global model
                client_model.load_state_dict(model.state_dict())
                client_model.train()

                client_grads_list = []

                # Compute gradients for each client
                for client_idx in range(self.num_clients):
                    f_batch = next(f_iters[client_idx])
                    r_batch = next(r_iters[client_idx]) if r_iters[client_idx] is not None else None

                    # Compute honest local gradient
                    grads = self._compute_local_gradient(client_model, f_batch, r_batch)

                    # Simulate Byzantine client injection for the first `num_byzantine` clients
                    if client_idx < self.num_byzantine:
                        grads = self._apply_byzantine_attack(grads)

                    client_grads_list.append(grads)

                # Aggregation at the server
                if self.aggregation == "krum":
                    aggregated_grads = self._aggregate_krum(client_grads_list)
                elif self.aggregation == "trimmed_mean":
                    aggregated_grads = self._aggregate_trimmed_mean(client_grads_list)
                else:
                    # Fallback to standard mean (non-robust baseline)
                    aggregated_grads = {}
                    for name in client_grads_list[0].keys():
                        aggregated_grads[name] = torch.stack([cg[name] for cg in client_grads_list]).mean(dim=0)

                # Apply aggregated gradients to the global server model
                self.optimizer.zero_grad()
                for name, param in model.named_parameters():
                    if name in aggregated_grads:
                        param.grad = aggregated_grads[name].to(self.device)

                # Server optimizer step
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()
                
                total_optimizer_steps += 1
                n_batches += 1

            avg_t = e_total / max(n_batches, 1)
            epoch_losses.append((epoch + 1, avg_t))

            print(f"[BRFU] Epoch {epoch+1:02d} | Completed Aggregations")

        elapsed = time.time() - t0
        print(f"[BRFU] Federated unlearning simulation complete in {elapsed:.1f}s")

        return UnlearningResult(
            method=f"BRFU-{self.aggregation.upper()}",
            epoch_losses=epoch_losses,
            forget_losses=[(i, 0.0) for i in range(1, cfg.unlearn_epochs + 1)],
            retain_losses=[(i, 0.0) for i in range(1, cfg.unlearn_epochs + 1)],
            total_steps=total_optimizer_steps,
            elapsed_sec=elapsed,
        )
