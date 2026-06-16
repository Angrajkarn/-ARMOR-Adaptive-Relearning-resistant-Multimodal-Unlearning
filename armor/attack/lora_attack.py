"""
armor/attack/lora_attack.py
===========================
LoRA-based Relearning Attack — a parameter-efficient adversarial attack.

Instead of full fine-tuning (which is expensive and unrealistic), the
attacker inserts a tiny rank-r LoRA adapter on top of the unlearned model
and trains ONLY the adapter on the forget set.

This is a STRONGER threat model than full fine-tuning because:
  - Much cheaper: adapter params << full model params
  - More realistic: attacker may only have API-style access + gradient
  - Harder to defend: any non-flat minimum is exploitable

Protocol:
  1. Freeze all base model weights
  2. Inject LoRA adapters (rank=4, alpha=16) into Q, V projections
  3. Fine-tune adapter on N forget samples x E epochs
  4. Measure recovery of forget accuracy
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Tuple, Optional
from tqdm import tqdm

from ..config import ARMORConfig
from ..eval.metrics import compute_token_accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Minimal LoRA implementation (no peft dependency required for debug)
# ─────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Replaces a nn.Linear with W + (alpha/r) * B @ A.
    Only A and B are trainable; W is frozen.
    """
    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 16.0):
        super().__init__()
        self.linear   = linear
        self.rank     = rank
        self.scaling  = alpha / rank

        in_features   = linear.in_features
        out_features  = linear.out_features

        # LoRA decomposition: ΔW = B @ A
        self.lora_A = nn.Parameter(torch.randn(rank, in_features)  * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Freeze the base weight
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x):
        base_out = self.linear(x)
        # LoRA delta: x @ A^T @ B^T * scaling
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out


def inject_lora(model: PreTrainedModel, rank: int = 4, alpha: float = 16.0,
                target_modules: Optional[List[str]] = None) -> PreTrainedModel:
    """
    Inject LoRA adapters into target projection layers.
    Default targets: 'q_proj', 'v_proj' (or 'c_attn' for GPT-2 style).
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "c_attn", "query", "value"]

    injected = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if any target string matches the layer name
            if any(t in name for t in target_modules):
                # Replace with LoRA wrapper via parent module
                parts   = name.split(".")
                parent  = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, parts[-1], lora_layer)
                injected += 1

    # Freeze everything that's not a LoRA parameter
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] Injected {injected} adapter(s) | "
          f"Trainable: {trainable:,} / {total:,} params "
          f"({100*trainable/total:.2f}%)")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# LoRA Attack Result
# ─────────────────────────────────────────────────────────────────────────────

class LoRAAttackResult:
    def __init__(self, method: str, original_acc: float):
        self.method               = method
        self.original_forget_acc  = original_acc
        self.pre_attack_forget_acc= 0.0
        self.final_forget_acc     = 0.0
        self.acc_jump             = 0.0
        self.recovery_pct         = float("nan")
        self.adapter_params       = 0
        self.elapsed_sec          = 0.0
        self.trajectory: List[Tuple[int, float, float]] = []

    def compute(self):
        self.acc_jump = self.final_forget_acc - self.pre_attack_forget_acc
        denom = self.original_forget_acc - self.pre_attack_forget_acc
        if denom > 1e-4:
            self.recovery_pct = max(0.0, min(100.0,
                                  (self.acc_jump / denom) * 100.0))

    def print_summary(self):
        print("\n" + "=" * 62)
        print(f"  LORA ATTACK RESULT -- Method: {self.method}")
        print("=" * 62)
        print(f"  Adapter trainable params       : {self.adapter_params:,}")
        print(f"  Pre-attack forget accuracy     : {self.pre_attack_forget_acc:.4f}")
        print(f"  Post-attack forget accuracy    : {self.final_forget_acc:.4f}")
        print(f"  Accuracy jump (+delta)         : {self.acc_jump:+.4f}")
        rstr = f"{self.recovery_pct:.1f}%" if self.recovery_pct == self.recovery_pct \
               else "N/A (model did not forget below baseline)"
        print(f"  Knowledge recovery             : {rstr}")
        print(f"  Attack wall time               : {self.elapsed_sec:.1f}s")
        print("=" * 62)
        print("\n  Epoch | Forget Acc | Retain Acc")
        print("  " + "-" * 32)
        for ep, f_acc, r_acc in self.trajectory:
            print(f"  {ep:5d} | {f_acc:10.4f} | {r_acc:10.4f}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main Attack Class
# ─────────────────────────────────────────────────────────────────────────────

class LoRARelearningAttack:
    """
    LoRA-based relearning attack against an unlearned model.

    Usage:
        attack = LoRARelearningAttack(cfg, model, tokenizer,
                                       forget_loader, retain_loader)
        result = attack.run(original_acc=0.85)
        result.print_summary()
    """

    def __init__(self,
                 cfg:            ARMORConfig,
                 model:          PreTrainedModel,
                 tokenizer:      PreTrainedTokenizer,
                 forget_loader:  DataLoader,
                 retain_loader:  DataLoader,
                 lora_rank:      int   = 4,
                 lora_alpha:     float = 16.0):
        self.cfg           = cfg
        self.model         = model
        self.tokenizer     = tokenizer
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader
        self.lora_rank     = lora_rank
        self.lora_alpha    = lora_alpha

    def run(self, method_name: str = "unknown",
            original_acc: float = 0.5) -> LoRAAttackResult:
        """
        Clone the unlearned model, inject LoRA, attack, and return results.
        """
        import copy
        result = LoRAAttackResult(method=method_name,
                                  original_acc=original_acc)

        # ── Clone so we don't mutate the original ──────────────────────────
        print(f"\n[LoRA Attack] Cloning model '{method_name}'...")
        attack_model = copy.deepcopy(self.model)
        attack_model.train()

        # ── Measure pre-attack accuracy ────────────────────────────────────
        attack_model.eval()
        result.pre_attack_forget_acc = compute_token_accuracy(
            attack_model, self.forget_loader, self.cfg.device)
        retain_acc_pre = compute_token_accuracy(
            attack_model, self.retain_loader, self.cfg.device)
        print(f"[LoRA Attack] Pre-attack forget acc : {result.pre_attack_forget_acc:.4f}")
        print(f"[LoRA Attack] Pre-attack retain acc : {retain_acc_pre:.4f}")

        # ── Inject LoRA adapters ───────────────────────────────────────────
        attack_model = inject_lora(attack_model,
                                    rank=self.lora_rank,
                                    alpha=self.lora_alpha)
        result.adapter_params = sum(
            p.numel() for p in attack_model.parameters()
            if p.requires_grad)

        # ── Fine-tune LoRA on forget set ───────────────────────────────────
        optimizer = torch.optim.AdamW(
            [p for p in attack_model.parameters() if p.requires_grad],
            lr=self.cfg.relearn_lr * 2)   # 2x LR since fewer params

        n_epochs = self.cfg.relearn_epochs
        t0 = time.time()

        for epoch in range(1, n_epochs + 1):
            attack_model.train()
            epoch_loss = 0.0
            for batch in self.forget_loader:
                input_ids = batch["input_ids"].to(self.cfg.device)
                labels    = batch["labels"].to(self.cfg.device)
                mask      = batch.get("attention_mask",
                                      torch.ones_like(input_ids)).to(self.cfg.device)
                optimizer.zero_grad()
                out  = attack_model(input_ids=input_ids,
                                    attention_mask=mask,
                                    labels=labels)
                loss = out.loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            attack_model.eval()
            f_acc = compute_token_accuracy(
                attack_model, self.forget_loader, self.cfg.device)
            r_acc = compute_token_accuracy(
                attack_model, self.retain_loader, self.cfg.device)
            result.trajectory.append((epoch, f_acc, r_acc))
            print(f"[LoRA Attack] Epoch {epoch:02d} | "
                  f"forget_acc={f_acc:.4f} | retain_acc={r_acc:.4f}")

        result.elapsed_sec     = time.time() - t0
        result.final_forget_acc = result.trajectory[-1][1]
        result.compute()
        return result
