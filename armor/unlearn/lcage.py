"""
armor/unlearn/lcage.py
======================
LCAGE: Latent Concept Association Graph Erasure

This module implements the LCAGE unlearning algorithm that erases the implicit
associative network of a forgotten concept, rather than just the direct question-answer mapping.
"""

import time
import math
import random
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

class LCAGEUnlearner:
    """
    LCAGE: Latent Concept Association Graph Erasure Unlearner.

    Extracts an association graph of related tokens/concepts in the model's
    embedding space for the forget set, computes the conceptual closure,
    and applies a weighted association suppression loss during unlearning.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cfg: ARMORConfig,
        qa_forget_pairs: Optional[List[Tuple[str, str]]] = None,
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

        # Extracted concept association parameters
        self.pmi_threshold = cfg.lcage_pmi_threshold
        self.lcage_coeff = cfg.lcage_coeff
        self.max_hops = cfg.lcage_max_hops

        # Build Latent Concept Association Graph
        self.association_closure_tokens: Dict[str, List[Tuple[int, float]]] = {}
        if qa_forget_pairs:
            self._build_concept_association_graph(qa_forget_pairs)

    def _build_concept_association_graph(self, qa_pairs: List[Tuple[str, str]]):
        """
        Builds the associative concept graph using cosine similarity
        in the word embedding space of the model.
        """
        print("[LCAGE] Extracting latent concept associations from embedding space...")
        embeds = self.model.get_input_embeddings().weight.data # (V, D)
        
        # Helper to get clean noun/keyword token IDs from text
        def extract_keywords(text: str) -> Set[int]:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            keywords = set()
            for t in tokens:
                s = self.tokenizer.decode([t]).strip().lower()
                # Simple heuristic: keep words with length > 4 that are not common punctuation
                if len(s) > 4 and s.isalnum():
                    keywords.add(t)
            return keywords

        for q, a in qa_pairs:
            keywords = extract_keywords(q) | extract_keywords(a)
            if not keywords:
                continue

            # Compute average embedding vector for forget keywords
            kw_ids = torch.tensor(list(keywords), device=self.device)
            with torch.no_grad():
                kw_embeds = embeds[kw_ids] # (N, D)
                mean_embed = kw_embeds.mean(dim=0, keepdim=True) # (1, D)
                mean_embed = F.normalize(mean_embed, p=2, dim=-1) # (1, D)

                # Compute cosine similarity with ALL tokens in vocab to find associations
                norm_embeds = F.normalize(embeds, p=2, dim=-1) # (V, D)
                sims = torch.matmul(norm_embeds, mean_embed.squeeze(0)) # (V,)

                # Get tokens exceeding similarity threshold (PMI proxy)
                mask = sims > self.pmi_threshold
                assoc_indices = torch.nonzero(mask).squeeze(-1).tolist()
                assoc_sims = sims[mask].tolist()

            # Store the associations
            closure = []
            for idx, val in zip(assoc_indices, assoc_sims):
                if idx not in keywords: # don't include direct keywords
                    closure.append((idx, float(val)))

            # Sort by similarity and keep top 20 to prevent training slowdown
            closure = sorted(closure, key=lambda x: x[1], reverse=True)[:20]
            self.association_closure_tokens[q] = closure

        total_assoc = sum(len(v) for v in self.association_closure_tokens.values())
        print(f"[LCAGE] Concept graph built. Total associations: {total_assoc} across {len(self.association_closure_tokens)} concepts.")

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

    def _association_suppression_loss(self, forget_batch: dict, questions: List[str]) -> torch.Tensor:
        """
        Suppresses the generation probability of conceptually associated tokens
        given the forget prompts.
        """
        loss_val = torch.tensor(0.0, device=self.device)
        count = 0

        # We inspect each sample in the batch
        input_ids = forget_batch["input_ids"].to(self.device)
        
        # Forward pass on the model to get logits
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits # (B, T, V)

        for i, q in enumerate(questions):
            # Find the index of the last non-padding token for this batch item
            seq_len = (forget_batch["attention_mask"][i] == 1).sum().item()
            if seq_len < 2 or q not in self.association_closure_tokens:
                continue

            associations = self.association_closure_tokens[q]
            if not associations:
                continue

            # Log probabilities at the final prompt token (predicting continuation)
            last_logits = logits[i, seq_len - 2, :] # (V,)
            log_probs = F.log_softmax(last_logits, dim=-1)

            # Penalize the log-likelihood of associated concepts, weighted by their PMI similarity
            sample_loss = torch.tensor(0.0, device=self.device)
            for token_id, sim in associations:
                # Minimize log probability = maximize negative log probability
                sample_loss += sim * (-log_probs[token_id])

            loss_val += sample_loss / len(associations)
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return loss_val / count

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
        forget_questions: Optional[List[str]] = None,
    ) -> UnlearningResult:
        """Runs the LCAGE unlearning loop."""
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

        epoch_losses, forget_losses, retain_losses, graph_losses = [], [], [], []
        total_optimizer_steps = 0
        t0 = time.time()

        print(f"\n[LCAGE] Starting training: {cfg.unlearn_epochs} epochs")
        print(f"        LCAGE Loss Coeff: {self.lcage_coeff:.3f}")

        # Keep track of questions per batch if not provided
        questions_pool = forget_questions or [""] * len(forget_loader.dataset)

        for epoch in range(cfg.unlearn_epochs):
            e_total = e_npo = e_retain = e_graph = 0.0
            n_batches = 0

            pbar = tqdm(
                forget_loader,
                desc=f"[LCAGE] Epoch {epoch+1}/{cfg.unlearn_epochs}",
                leave=False,
            )

            for step, forget_batch in enumerate(pbar):
                retain_batch = next(retain_iter) if retain_iter else None

                # Extract questions for this batch
                batch_start = step * forget_loader.batch_size
                batch_end = batch_start + forget_loader.batch_size
                batch_questions = questions_pool[batch_start:batch_end]

                # Ensure length matches batch size
                if len(batch_questions) < forget_batch["input_ids"].shape[0]:
                    batch_questions += [""] * (forget_batch["input_ids"].shape[0] - len(batch_questions))

                # 1. NPO forget loss
                npo_loss = self._npo_forget_loss(forget_batch)

                # 2. Retain loss
                if retain_batch is not None:
                    r_loss = self._retain_loss(retain_batch)
                else:
                    r_loss = torch.tensor(0.0, device=self.device)

                # 3. LCAGE graph closure suppression loss
                g_loss = self._association_suppression_loss(forget_batch, batch_questions)

                # Combined loss
                total_loss = (
                    npo_loss
                    + cfg.npo_retain_coeff * r_loss
                    + self.lcage_coeff * g_loss
                )

                # Backward
                scaled = total_loss / cfg.gradient_accumulation_steps
                scaled.backward()

                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    total_optimizer_steps += 1

                e_total  += total_loss.item()
                e_npo    += npo_loss.item()
                e_retain += r_loss.item() if hasattr(r_loss, "item") else 0.0
                e_graph  += g_loss.item() if hasattr(g_loss, "item") else 0.0
                n_batches += 1

                pbar.set_postfix({
                    "npo": f"{npo_loss.item():.3f}",
                    "retain": f"{r_loss.item():.3f}" if hasattr(r_loss, "item") else "0",
                    "graph": f"{g_loss.item():.3f}" if hasattr(g_loss, "item") else "0",
                })

            avg_t = e_total / max(n_batches, 1)
            avg_n = e_npo / max(n_batches, 1)
            avg_r = e_retain / max(n_batches, 1)
            avg_g = e_graph / max(n_batches, 1)

            epoch_losses.append((epoch + 1, avg_t))
            forget_losses.append((epoch + 1, avg_n))
            retain_losses.append((epoch + 1, avg_r))
            graph_losses.append((epoch + 1, avg_g))

            print(f"[LCAGE] Epoch {epoch+1:02d} | "
                  f"npo={avg_n:.4f} | retain={avg_r:.4f} | "
                  f"graph={avg_g:.4f} | total={avg_t:.4f}")

        elapsed = time.time() - t0
        print(f"[LCAGE] Training complete in {elapsed:.1f}s")

        return UnlearningResult(
            method="LCAGE",
            epoch_losses=epoch_losses,
            forget_losses=forget_losses,
            retain_losses=retain_losses,
            total_steps=total_optimizer_steps,
            elapsed_sec=elapsed,
        )
