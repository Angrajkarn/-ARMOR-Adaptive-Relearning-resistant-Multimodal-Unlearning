# ARMOR — Adaptive Relearning-resistant Multimodal Unlearning

A research codebase for **machine unlearning** in large language models with formal verifiability.

## Architecture

```
armor/
├── config.py                # Central hyperparameter dataclass
├── data.py                  # TOFU dataset + rephrase augmentation
├── model.py                 # Model loader (opt-125m → Mistral-7B → LLaVA)
├── unlearn/
│   ├── gradient_ascent.py   # Baseline GA: -α*L_forget + β*L_retain
│   ├── npo.py               # NPO: -log σ(β·(log π_θ - log π_ref))
│   └── sam_wrapper.py       # SAM: flat-minima optimizer for relearning resistance
├── eval/
│   ├── metrics.py           # Forget quality, retain acc, ROUGE-1/L
│   └── mia.py               # Min-K% Prob MIA → formal audit AUROC
└── attack/
    └── relearning.py        # Fine-tune on N forget samples → recovery %
```

## Quick Start (CPU Debug)

```bash
pip install -r requirements.txt

# 1. Gradient Ascent baseline (~2 min on CPU)
python scripts/run_baseline_ga.py --debug

# 2. NPO baseline
python scripts/run_baseline_npo.py --debug

# 3. NPO + SAM (relearning-resistant)
python scripts/run_npo_sam.py --debug

# 4. Relearning attack comparison
python scripts/run_relearning_attack.py --debug --compare
```

## Full GPU Run (Mistral-7B + QLoRA)

```bash
# Uncomment bitsandbytes in requirements.txt first
pip install bitsandbytes>=0.43.0

python scripts/run_baseline_ga.py   --model mistral-7b --qlora --run-mia
python scripts/run_baseline_npo.py  --model mistral-7b --qlora --run-mia
python scripts/run_npo_sam.py       --model mistral-7b --qlora --run-mia --sam-rho 0.05

python scripts/run_relearning_attack.py --model mistral-7b --qlora --compare \
    --original-acc 0.85
```

## Metrics

| Metric | Description | Direction |
|---|---|---|
| **Forget Quality** | 1 − forget accuracy | ↑ Higher = better |
| **Retain Accuracy** | Token accuracy on retain set | ↑ Higher = better |
| **Forget ROUGE-L** | ROUGE of generated vs. GT answers on forget set | ↓ Lower = better |
| **Retain ROUGE-L** | ROUGE on retain set (utility) | ↑ Higher = better |
| **MIA AUROC** | Min-K% Prob membership inference audit score | → 0.5 = verified |

## Extending to LLaVA (Step 2)

1. In `armor/model.py`, add `get_llava_model_and_processor()`
2. In `armor/data.py`, add `MultimodalTOFUDataset` with image paths
3. In `armor/unlearn/npo.py`, pass `pixel_values` alongside `input_ids`
4. In `armor/eval/metrics.py`, add VQA accuracy metric

All module interfaces are designed to be backward compatible.

## References

- Maini et al., "TOFU: A Task of Fictitious Unlearning" (2024)
- Zhang et al., "Negative Preference Optimization" (2024) [arXiv:2404.05868]
- Foret et al., "Sharpness-Aware Minimization" (ICLR 2021)
- Shi et al., "Detecting Pretraining Data via Min-K% Prob" (2024)
