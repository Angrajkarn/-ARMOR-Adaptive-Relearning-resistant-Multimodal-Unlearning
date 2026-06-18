"""
armor/model.py
==============
Model and tokenizer loader for ARMOR experiments.

Supports:
  • CPU debugging     : opt-125m, no quantization
  • Single GPU (≥16GB): Mistral-7B / LLaMA-2-7B with 4-bit QLoRA
  • Multi-GPU         : device_map="auto" distributes layers

Cross-modal NOTE (Step 2):
  For LLaVA extension, replace `get_model_and_tokenizer()` with
  `get_llava_model_and_processor()` which also returns a CLIPImageProcessor.
  All unlearning modules operate on model.language_model, keeping the
  interface consistent.
"""

# Suppress noisy but harmless Windows warnings before any imports
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import warnings
warnings.filterwarnings("ignore", message=".*hf_xet.*")

from copy import deepcopy
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
)

from armor.config import ARMORConfig


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_bnb_config() -> "BitsAndBytesConfig":
    """4-bit NF4 quantization config for QLoRA (GPU only)."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,    # Nested quantization saves ~0.4 GB
    )


def _apply_lora(model: PreTrainedModel, cfg: ARMORConfig) -> PreTrainedModel:
    """
    Wrap the model with LoRA adapters using PEFT.

    Only the LoRA adapter weights are trained — base model stays frozen.
    This allows sharing the base quantized weights between the trainable
    model and the frozen reference model (NPO) without doubling VRAM.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError(
            "peft is required for LoRA. Install with: pip install peft"
        )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Main loader
# ──────────────────────────────────────────────────────────────────────────────

def get_model_and_tokenizer(
    cfg: ARMORConfig,
    verbose: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load the base model and tokenizer specified by cfg.model_name.

    Returns
    -------
    model     : Wrapped with LoRA if cfg.use_qlora=True; otherwise full model
    tokenizer : With pad_token set (required for batched training)

    Example
    -------
    cfg   = ARMORConfig(debug=True)
    model, tok = get_model_and_tokenizer(cfg)
    """
    model_name = cfg.model_name
    device     = cfg.device

    if verbose:
        print(f"[model] Loading '{model_name}' on device='{device}' "
              f"(qlora={cfg.use_qlora})")

    # Clean up stale HF lock files if any exist (prevents infinite hangs after aborted/killed runs)
    try:
        import pathlib
        cache_dir = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
        if cache_dir.exists():
            for lock_file in cache_dir.glob("**/*.lock"):
                try:
                    lock_file.unlink()
                except Exception:
                    pass
    except Exception:
        pass

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        token=cfg.hf_token,
        # Padding side left is standard for causal generation, right for training
        padding_side="right",
    )
    # Ensure a pad token exists (LLaMA-2 / Mistral use EOS as pad by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Model loading kwargs ───────────────────────────────────────────────────
    load_kwargs = {
        "token": cfg.hf_token,
        "low_cpu_mem_usage": True,
    }

    if cfg.model_key == "debug":
        # Debug model: load on single device, no device_map distribution needed
        load_kwargs["dtype"] = torch.float32
    elif cfg.use_qlora and device == "cuda":
        # 4-bit QLoRA: quantize base model, train only adapter weights
        load_kwargs["quantization_config"] = _build_bnb_config()
        load_kwargs["device_map"] = "auto"     # Spread across all GPUs
    elif device == "cuda":
        load_kwargs["dtype"] = torch.bfloat16
        load_kwargs["device_map"] = "auto"
    else:
        # CPU: load in float32 (bfloat16 unsupported on most CPUs)
        load_kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Move to device if not handled by device_map
    if "device_map" not in load_kwargs:
        model = model.to(device)

    # ── LoRA wrapping ──────────────────────────────────────────────────────────
    if cfg.use_qlora:
        model = _apply_lora(model, cfg)
        # Required after 4-bit loading + LoRA wrapping
        model.config.use_cache = False
        model.enable_input_require_grads()

    # ── Gradient checkpointing (saves ~40% memory, small speed cost) ───────────
    if device == "cuda":
        model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"[model] Total params     : {total_params/1e6:.1f}M")
        print(f"[model] Trainable params : {trainable/1e6:.1f}M "
              f"({100*trainable/total_params:.1f}%)")

    return model, tokenizer


def get_frozen_reference_model(
    model: PreTrainedModel,
    cfg: ARMORConfig,
) -> PreTrainedModel:
    """
    Create a frozen copy of the model for NPO's reference (π_ref).

    Memory-efficient: if using QLoRA, the base quantized weights are
    shared — only adapter weights are duplicated.

    Parameters
    ----------
    model : The trainable model (already wrapped with LoRA if applicable)

    Returns
    -------
    ref_model : Frozen copy — no gradients, eval mode
    """
    if cfg.use_qlora:
        # For QLoRA: disable LoRA adapters to get base model predictions
        # This avoids copying 7B parameters
        try:
            from peft import get_peft_model_state_dict
            ref_model = model
            # We'll call model.disable_adapter_layers() during NPO forward pass
            # See npo.py for usage
            print("[model] NPO reference: using base weights (LoRA-disabled mode)")
        except ImportError:
            ref_model = deepcopy(model)
    else:
        # Full precision: deepcopy the model
        ref_model = deepcopy(model)

    # Freeze all parameters
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    return ref_model


def save_checkpoint(model: PreTrainedModel,
                    tokenizer: PreTrainedTokenizer,
                    path: str,
                    cfg: ARMORConfig) -> None:
    """Save model + tokenizer checkpoint to disk."""
    os.makedirs(path, exist_ok=True)
    if cfg.use_qlora:
        # Save only LoRA adapter weights (much smaller)
        model.save_pretrained(path)
    else:
        model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"[model] Checkpoint saved → {path}")


def load_checkpoint(path: str,
                    cfg: ARMORConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a previously saved checkpoint for evaluation or attack."""
    print(f"[model] Loading checkpoint from {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.use_qlora:
        from peft import PeftModel
        base_model, _ = get_model_and_tokenizer(cfg, verbose=False)
        model = PeftModel.from_pretrained(base_model, path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=torch.float32 if cfg.device == "cpu" else torch.bfloat16,
        ).to(cfg.device)

    return model, tokenizer
