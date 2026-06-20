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

    # ══════════════════════════════════════════════════════════════════════════
    # RESEARCH EXPANSION MODULES (added 2026-06)
    # ══════════════════════════════════════════════════════════════════════════

    # ── Module 1 — Lifelong (Continual) Unlearning ───────────────────────────
    continual_buffer_size: int = 200          # retain exemplars in replay buffer
    continual_fim_topk: float = 0.30          # fraction of params protected by FIM mask
    continual_use_fim_mask: bool = False      # enable FIM-based subspace masking

    # ── Module 2 — MoE Targeted Unlearning ───────────────────────────────────
    moe_router_loss_coeff: float = 0.50       # weight of router-diversion loss
    moe_prune_experts: bool = False           # enable expert magnitude pruning
    moe_prune_fraction: float = 0.10          # fraction of expert weights to zero

    # ── Module 3 — Advanced RMU / RLACE ──────────────────────────────────────
    rlace_n_layers: int = 3                   # number of layers for concept erasure
    rlace_probe_epochs: int = 10              # epochs to train linear membership probe
    rlace_whittle_iters: int = 300            # projected gradient descent steps

    # ── Module 4 — Zero-Knowledge Unlearning Verification ────────────────────
    zk_influence_damping: float = 5e-3        # Tikhonov damping for Hessian approx
    zk_n_probe_samples: int = 50              # samples for influence estimation
    zk_influence_threshold: float = 0.01     # min influence gap to declare verified

    # ── Module 5 — Multimodal MIA Audit ──────────────────────────────────────
    mm_mia_contrastive_temp: float = 0.07     # softmax temperature for contrastive loss
    mm_mia_contrastive_coeff: float = 1.0    # weight of contrastive unlearning loss
    mm_mia_similarity_threshold: float = 0.50  # cosine sim threshold for MIA detection

    # ── Module 6 — Modular LoRA Unlearning ───────────────────────────────────
    lora_unlearn_r: int = 16                  # rank of forget LoRA adapter
    lora_unlearn_alpha: int = 32              # LoRA alpha for forget adapter
    lora_unlearn_scale: float = 1.0           # λ — subtraction scale factor
    lora_retain_r: int = 8                    # rank of retain LoRA (0 = disabled)
    lora_merge_final: bool = True             # merge adapters into base weights at end

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 RESEARCH MODULES (added 2026-06)
    # ══════════════════════════════════════════════════════════════════════════

    # ── CU-AR — Conformal Unlearning Verification ─────────────────────────────
    conformal_alpha: float = 0.05             # max miscoverage rate (5%)
    conformal_halflife_adj: bool = True       # use finite-sample adjusted quantile
    conformal_retain_check_n: int = 100       # retain samples for sanity check

    # ── CoT-HME — Chain-of-Thought Hidden Memory Erasure ──────────────────────
    cot_loss_coeff: float = 0.30              # weight of CoT entropy loss
    cot_leak_threshold: float = 0.30         # leakage threshold for step classification
    cot_max_new_tokens: int = 128            # max tokens for CoT trace generation
    cot_probe_batch: int = 20               # max samples probed per epoch
    cot_keyword_weight: float = 0.60        # keyword score weight in leakage scorer
    cot_semantic_weight: float = 0.40       # semantic score weight in leakage scorer
    cot_reprobe_interval: int = 2           # re-probe every N epochs

    # ── TKDU — Temporal Knowledge Decay Unlearning ────────────────────────────
    tkdu_halflife_days: float = 30.0         # temporal decay half-life
    tkdu_expired_threshold: float = 0.10    # validity below which fact is expired
    tkdu_near_expiry_threshold: float = 0.50  # validity boundary for "near expiry"
    tkdu_expiry_buffer_days: float = 0.0    # trigger unlearning before expiry

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
            try:
                # Test if CUDA is actually working (e.g., checks against incompatible sm_60 GPUs)
                _ = torch.ones(1, device="cuda")
                return "cuda"
            except Exception:
                return "cpu"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
