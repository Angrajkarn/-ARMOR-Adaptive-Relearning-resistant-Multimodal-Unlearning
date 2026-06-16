"""
armor/config.py
===============
Central configuration for ARMOR experiments.

All hyperparameters live here — never scatter magic numbers across files.
Change MODEL_NAME + DEBUG_MODE to switch between local CPU testing and
full-scale GPU training.
"""

from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# Switch MODEL_NAME here to change the backbone. The loader in model.py
# handles the differences automatically.
# ──────────────────────────────────────────────────────────────────────────────
SUPPORTED_MODELS = {
    # Tiny model — runs on CPU in seconds; only ~82MB download
    # distilgpt2 is a 6-layer GPT-2 distilled model — identical causal LM interface
    "debug": "distilgpt2",

    # Slightly larger CPU option (~500MB) — better representations
    # "debug": "facebook/opt-125m",

    # Full open-weight models — require GPU (≥16 GB VRAM with 4-bit QLoRA)
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "llama2-7b":  "meta-llama/Llama-2-7b-hf",     # needs HF token + gated access

    # Placeholder for LLaVA (Step 2 — cross-modal extension)
    # "llava-7b": "llava-hf/llava-1.5-7b-hf",
}


@dataclass
class ARMORConfig:
    """
    Master config dataclass. Pass one instance through the entire pipeline.

    Usage
    -----
    cfg = ARMORConfig(debug=True)          # CPU / opt-125m
    cfg = ARMORConfig(model_key="mistral-7b", use_qlora=True)   # GPU
    """

    # ── Model ─────────────────────────────────────────────────────────────────
    model_key: str = "debug"                   # Key into SUPPORTED_MODELS
    hf_token: Optional[str] = None             # HF token (needed for LLaMA-2)
    use_qlora: bool = False                    # 4-bit QLoRA (GPU only)
    lora_r: int = 8                            # LoRA rank
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(         # Attention projections to adapt
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # ── TOFU Dataset ──────────────────────────────────────────────────────────
    tofu_forget_split: str = "forget10"        # forget01 / forget05 / forget10
    tofu_retain_split: str = "retain90"        # complement of forget split
    max_seq_len: int = 256                     # Truncate to save memory on CPU

    # ── Unlearning Training ───────────────────────────────────────────────────
    unlearn_epochs: int = 5
    unlearn_lr: float = 1e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 4      # Effective batch = 8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # ── GA-specific ───────────────────────────────────────────────────────────
    ga_forget_coeff: float = 1.0              # α — weight on forget loss
    ga_retain_coeff: float = 0.5             # β — weight on retain loss

    # ── NPO-specific ──────────────────────────────────────────────────────────
    npo_beta: float = 0.1                     # β in DPO-style log-ratio loss
    npo_retain_coeff: float = 0.5

    # ── SAM-specific ──────────────────────────────────────────────────────────
    sam_rho: float = 0.05                     # Neighbourhood radius ρ
    sam_adaptive: bool = False                # Adaptive SAM (ASAM variant)

    # ── Rephrase augmentation (relearning-resistant) ───────────────────────────
    use_rephrase_augmentation: bool = True
    num_rephrases: int = 3                    # Extra paraphrases per sample

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_batch_size: int = 2
    rouge_n: int = 1                          # ROUGE-1 (also reports ROUGE-L)
    mia_n_neighbors: int = 5                  # Min-K% neighbours for MIA

    # ── Relearning Attack ─────────────────────────────────────────────────────
    relearn_n_samples: int = 50               # Forget samples for attack
    relearn_epochs: int = 10
    relearn_lr: float = 2e-5

    # ── Paths ─────────────────────────────────────────────────────────────────
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # ── Debug / CPU mode ──────────────────────────────────────────────────────
    debug: bool = False                        # --debug flag in scripts
    debug_n_samples: int = 16                 # Tiny subset for quick testing

    def __post_init__(self):
        if self.debug:
            # Override to tiny model + minimal data for fast CPU smoke test
            self.model_key         = "debug"
            self.unlearn_epochs    = 2
            self.batch_size        = 2
            self.max_seq_len       = 48        # Very short sequences → fast CPU
            self.debug_n_samples   = 8         # 8 samples is enough to verify the loop
            self.relearn_n_samples = 6
            self.relearn_epochs    = 2
            self.use_qlora         = False
            self.num_rephrases     = 2         # Fewer rephrases in debug

    @property
    def model_name(self) -> str:
        """Resolve model key → HuggingFace model ID."""
        if self.model_key not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model_key '{self.model_key}'. "
                f"Choose from: {list(SUPPORTED_MODELS.keys())}"
            )
        return SUPPORTED_MODELS[self.model_key]

    @property
    def device(self) -> str:
        """Auto-detect best available device."""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
