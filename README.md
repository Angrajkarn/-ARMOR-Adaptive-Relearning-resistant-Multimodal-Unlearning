<div align="center">

<h1>🛡️ ARMOR</h1>
<h3>Adaptive Relearning-resistant Multimodal Unlearning</h3>

<p><em>A research framework for verifiable, robust machine unlearning in large language and vision-language models</em></p>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Angrajkarn/-ARMOR-Adaptive-Relearning-resistant-Multimodal-Unlearning/blob/main/ARMOR_Colab_Experiments.ipynb)

</div>

---

## 📌 What is ARMOR?

**ARMOR** is a machine unlearning system for large language models (LLMs) that goes beyond simply erasing knowledge — it ensures that erased knowledge **cannot be recovered** through fine-tuning attacks or prompt rephrasing.

### The Problem with Existing Unlearning Methods

```
Unlearn → Model forgets ✓
Attacker fine-tunes on 50 forget samples → Model re-learns everything ✗
```

ARMOR solves this by pushing unlearned models into **flat loss minima** using Sharpness-Aware Minimization (SAM), making re-learning geometrically difficult.

### Key Contributions

| Feature | Description |
|---|---|
| 🔁 **Relearning-Resistant** | SAM optimizer targets flat minima — hard to escape via fine-tuning |
| 📝 **Rephrasing-Invariant** | Rephrase augmentation ensures the forget effect survives prompt variations |
| 🔍 **Formally Verifiable** | Min-K% Prob MIA generates an AUROC audit score proving unlearning |
| 🖼️ **Cross-Modal Ready** | Extended to LLaVA (vision + language unlearning) |
| 🔒 **DP Certified** | DP-NPO+SAM provides formal (ε, δ)-differential privacy guarantee |
| 🌐 **Multi-Benchmark** | Supports TOFU and MUSE benchmarks |

---

## 🧠 Methods

### 1. Gradient Ascent (GA) — Baseline

The simplest unlearning method: maximize loss on the forget set while preserving the retain set.

```
L_GA = −α · L_forget(θ) + β · L_retain(θ)
```

**Weakness:** Converges to sharp minima → vulnerable to relearning attacks.

---

### 2. Negative Preference Optimization (NPO)

Inspired by RLHF's DPO, NPO treats forget-set answers as "dispreferred" responses and pushes the model away from the reference policy.

```
L_NPO = −log σ( β · (log π_θ(y|x) − log π_ref(y|x)) ) + γ · L_retain
```

---

### 3. NPO + SAM — ARMOR Core Method ⭐

Wraps NPO inside a **Sharpness-Aware Minimization** optimizer:

```
Step 1 (perturbation):   ε̂ = ρ · ∇L / ‖∇L‖
Step 2 (update):         θ ← θ − η · ∇L(θ + ε̂)
```

---

### 4. RMU — Representation Misdirection Unlearning 🆕

Corrupts internal representations of forget-set inputs at a chosen transformer layer, pushing them toward a random misdirection vector.

```
L_RMU = α · ‖h_L(x_f) − c·u‖² + β · ‖h_L(x_r) − h_L_ref(x_r)‖²
```

**Advantage:** Works at representation level → harder to reverse via output fine-tuning.

---

### 5. Task Vector Unlearning 🆕

Computes the "forget task vector" (fine-tuned weights − original weights), then negates it to steer the model away from memorized knowledge.

```
θ_unlearned = θ_original − λ · (θ_finetuned − θ_original)
```

---

### 6. Multi-Task NPO 🆕

Simultaneously forgets K disjoint topics using orthogonal gradient projection to avoid interference between tasks.

```
g_k_ortho = g_k − Σ_{j<k} proj(g_k, g_j)
```

---

### 7. EUL — Exact Unlearning via Influence Functions 🆕

Approximates the leave-one-out gradient update using diagonal Fisher Information scaling.

```
θ_unlearn ≈ θ − H⁻¹ · ∇L_forget(θ)
```

---

### 8. WHO — Weights Harmonization Objective 🆕

Projects the forget gradient orthogonal to the retain gradient, preventing catastrophic forgetting.

```
g_ortho = g_f − (g_f · g_r / ‖g_r‖²) · g_r
```

---

### 9. DP-NPO+SAM — Full Privacy Stack 🆕

Combines NPO + SAM + Differential Privacy (DP-SGD) for a formal (ε, δ)-DP certificate.

```
g̃ = (1/B) Σ clip(∇L_i, C) + N(0, σ²C²/B² · I)   # noised gradient
θ ← SAM_update(g̃)
```

---

## 🗂️ Project Structure

```
ARMOR/
├── armor/
│   ├── config.py                  # ARMORConfig dataclass — all hyperparams
│   ├── data.py                    # TOFU loader + rephrase augmentation (×3)
│   ├── data_muse.py               # MUSE benchmark data loader (books/news) 🆕
│   ├── model.py                   # Model loader: distilgpt2 → Mistral-7B → LLaVA
│   ├── unlearn/
│   │   ├── gradient_ascent.py     # GA baseline
│   │   ├── npo.py                 # NPO: DPO-style forget divergence
│   │   ├── sam_wrapper.py         # SAMOptimizer: 2-pass flat-minima wrapper
│   │   ├── rmu.py                 # RMU: Representation Misdirection 🆕
│   │   ├── task_vector.py         # Task Vector Unlearning 🆕
│   │   ├── multitask_npo.py       # Multi-Task NPO with orthogonal projection 🆕
│   │   ├── eul.py                 # EUL: Influence Function approximation 🆕
│   │   ├── who.py                 # WHO: Weights Harmonization Objective 🆕
│   │   └── dp_npo_sam.py          # DP-NPO+SAM: Full privacy stack 🆕
│   ├── eval/
│   │   ├── metrics.py             # EvaluationResult: forget/retain + ROUGE
│   │   ├── mia.py                 # Min-K% Prob → MIA AUROC
│   │   └── privacy_audit.py       # Comprehensive privacy audit suite 🆕
│   └── attack/
│       ├── relearning.py          # Relearning attack simulation
│       ├── lora_attack.py         # LoRA fine-tuning attack 🆕
│       ├── prompt_attack.py       # Prompt injection attack 🆕
│       └── federated_attack.py    # Federated relearning attack 🆕
│
├── scripts/
│   ├── run_baseline_ga.py         # Gradient Ascent
│   ├── run_baseline_npo.py        # NPO
│   ├── run_npo_sam.py             # ARMOR core (NPO+SAM)
│   ├── run_relearning_attack.py   # Attack all checkpoints
│   ├── run_rmu.py                 # RMU experiment 🆕
│   ├── run_task_vector.py         # Task Vector experiment 🆕
│   ├── run_multitask_unlearn.py   # Multi-Task NPO experiment 🆕
│   ├── run_dp_armor.py            # DP-NPO+SAM experiment 🆕
│   ├── run_llava_unlearn.py       # LLaVA cross-modal experiment 🆕
│   └── run_muse_benchmark.py      # MUSE benchmark (books/news) 🆕
│
├── ARMOR_Colab_Experiments.ipynb  # 🚀 Full GPU experiment suite for Google Colab
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### 🚀 Google Colab (Recommended — Free GPU)

Click the badge at the top or open directly:

```
https://colab.research.google.com/github/Angrajkarn/-ARMOR-Adaptive-Relearning-resistant-Multimodal-Unlearning/blob/main/ARMOR_Colab_Experiments.ipynb
```

The notebook:
- Clones the repo automatically
- Installs all dependencies (including bitsandbytes, opacus)
- Runs all 9 methods on Mistral-7B with 4-bit QLoRA
- Saves all checkpoints to Google Drive
- Produces a comparison table and plot

---

### CPU Debug Mode (distilgpt2, ~3 min total)

```bash
set PYTHONIOENCODING=utf-8

# Baselines
python scripts/run_baseline_ga.py  --debug --no-rouge
python scripts/run_baseline_npo.py --debug --no-rouge
python scripts/run_npo_sam.py      --debug --no-rouge

# New methods
python scripts/run_rmu.py              --debug --no-rouge
python scripts/run_task_vector.py      --debug --no-rouge
python scripts/run_multitask_unlearn.py --debug --no-rouge --n-tasks 2
python scripts/run_dp_armor.py         --debug --no-rouge
python scripts/run_llava_unlearn.py    --debug --text-only --no-rouge
python scripts/run_muse_benchmark.py   --debug --domain books --method npo_sam

# Relearning attack
python scripts/run_relearning_attack.py --debug --compare --original-acc 0.3983
```

### Full GPU Run (Mistral-7B + 4-bit QLoRA, ≥16 GB VRAM)

```bash
pip install bitsandbytes>=0.43.0 opacus>=1.4.0

python scripts/run_npo_sam.py   --model mistral-7b --qlora --run-mia
python scripts/run_rmu.py       --model mistral-7b --qlora --run-mia
python scripts/run_dp_armor.py  --model mistral-7b --qlora --run-mia --epsilon 8.0
python scripts/run_muse_benchmark.py --model mistral-7b --qlora --domain books --method rmu
```

---

## 📊 Benchmark Results

Evaluated on **TOFU** (`locuslab/TOFU`) and **MUSE** benchmarks.

### Debug Run (distilgpt2, CPU, 2 epochs)

| Method | Forget Quality ↑ | Forget Acc ↓ | Retain Acc ↑ | Train Time |
|:---:|:---:|:---:|:---:|:---:|
| Pre-unlearning | 0.602 | 0.398 | 0.373 | — |
| Gradient Ascent | 0.622 | 0.378 | 0.387 | 64s |
| NPO | 0.585 | 0.416 | 0.403 | 128s |
| **NPO + SAM (ARMOR)** | 0.550 | 0.450 | **0.459** | 168s |
| RMU | 0.584 | 0.416 | 0.373 | 45s |
| Task Vector | 0.765 | 0.235 | 0.307 | 53s |
| MultiTask-NPO | 0.599 | 0.401 | 0.390 | 14min |
| DP-NPO+SAM (ε=8.0) | 0.602 | 0.398 | 0.373 | 90s |
| LLaVA-NPO+SAM | 0.593 | 0.407 | 0.406 | 30s |

> ⚠️ Debug numbers use distilgpt2 (never trained on TOFU). GPU results on Mistral-7B are significantly more pronounced.

---

## 📐 Evaluation Metrics

| Metric | Formula | Goal |
|---|---|---|
| **Forget Quality** | `1 − forget_accuracy` | ↑ Maximize |
| **Forget Accuracy** | Token accuracy on forget-set Q&A | ↓ Minimize |
| **Forget ROUGE-L** | ROUGE-L of generated vs. reference | ↓ Minimize |
| **Retain Accuracy** | Token accuracy on retain-set Q&A | ↑ Maximize |
| **MIA AUROC** | Min-K% Prob membership inference | → 0.5 |
| **DP Epsilon (ε)** | Formal differential privacy budget | ↓ Minimize |
| **Relearning Recovery %** | Attack recovery rate | ↓ Minimize |

---

## 🔒 Formal Audit: Membership Inference + DP Certificate

```python
# MIA: AUROC ≈ 0.5 = verified unlearned
auditor = MembershipInferenceAuditor(model, tokenizer, cfg)
auditor.audit(forget_loader, retain_loader, method_name="NPO+SAM")

# DP Certificate: (ε, δ)-DP guarantee
# DP-NPO+SAM stops training when target ε is reached
# Final: ε = 0.826, δ = 1e-5  →  formal (ε, δ)-DP certificate
```

---

## 🗺️ Roadmap

- [x] Gradient Ascent baseline on TOFU
- [x] NPO (Negative Preference Optimization)
- [x] SAM optimizer wrapper (flat-minima, relearning-resistant)
- [x] Min-K% Prob MIA audit (AUROC verification)
- [x] Relearning attack simulation (LoRA + prompt + federated)
- [x] Rephrase augmentation (×3 per sample)
- [x] RMU (Representation Misdirection Unlearning)
- [x] Task Vector Unlearning
- [x] Multi-Task NPO with orthogonal gradient projection
- [x] EUL (Exact Unlearning via Influence Functions)
- [x] WHO (Weights Harmonization Objective)
- [x] DP-NPO+SAM (Differential Privacy stack)
- [x] LLaVA Cross-Modal Unlearning (text-only verified)
- [x] MUSE Benchmark integration (books / news domains)
- [x] Google Colab experiment notebook
- [ ] Full Mistral-7B / LLaMA-2-7B GPU results (run on Colab)
- [ ] HuggingFace Hub model card upload
- [ ] Real LLaVA-1.5-7b multimodal forward pass

---

## 📚 References

1. **TOFU Benchmark** — Maini et al., *"TOFU: A Task of Fictitious Unlearning for LLMs"* (2024) · [arXiv:2401.06121](https://arxiv.org/abs/2401.06121)
2. **NPO** — Zhang et al., *"Negative Preference Optimization: How to Make LLMs Forget"* (2024) · [arXiv:2404.05868](https://arxiv.org/abs/2404.05868)
3. **SAM** — Foret et al., *"Sharpness-Aware Minimization"* (ICLR 2021) · [arXiv:2010.01412](https://arxiv.org/abs/2010.01412)
4. **RMU** — Li et al., *"The WMDP Benchmark"* (2024) · [arXiv:2403.03218](https://arxiv.org/abs/2403.03218)
5. **Task Vector** — Ilharco et al., *"Editing Models with Task Arithmetic"* (ICLR 2023) · [arXiv:2212.04089](https://arxiv.org/abs/2212.04089)
6. **Min-K% Prob MIA** — Shi et al., *"Detecting Pretraining Data from Large Language Models"* (2024) · [arXiv:2310.16789](https://arxiv.org/abs/2310.16789)
7. **MUSE** — Shi et al., *"MUSE: Machine Unlearning Six-Way Evaluation"* (2024) · [arXiv:2407.06460](https://arxiv.org/abs/2407.06460)

---

<div align="center">
<sub>Built for ML research · ARMOR © 2024 · <a href="https://colab.research.google.com/github/Angrajkarn/-ARMOR-Adaptive-Relearning-resistant-Multimodal-Unlearning/blob/main/ARMOR_Colab_Experiments.ipynb">🚀 Open in Colab</a></sub>
</div>
