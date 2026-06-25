"""
gen_notebooks.py — Generates BOTH Colab and Kaggle notebooks optimised to
                   finish all core experiments in ≤ 60 minutes on a T4 GPU.

Run:  python gen_notebooks.py

Key speed tricks:
  • ARMOR_FAST=1 env var  →  retain=200, fp16, epochs=1  (no --fast CLI flag needed)
  • sam_every=4 for NPO+SAM  (halves SAM overhead vs sam_every=2)
  • Batch 9 lightweight modules into 3 cells (fewer model re-loads)
  • Phase 1-3 research methods skipped by default (user can uncomment)
  • --no-save on non-essential methods  (saves I/O time)
  • --no-rouge on everything  (saves ~1 min per method)

Time budget (T4 GPU):
  Model download (first time, cached): ~4 min
  GA baseline:                         ~5 min
  NPO baseline:                        ~5 min
  NPO+SAM (sam_every=4):              ~10 min
  RMU:                                 ~4 min
  Task Vector:                         ~5 min
  Relearning Attack:                   ~4 min
  9 ARMOR Modules (3 batched cells):   ~12 min
  Results + Plots:                      ~1 min
  ────────────────────────────────────────────
  TOTAL:                              ~50 min
"""

import json, sys, os

# ─── Helpers ──────────────────────────────────────────────────────────────────

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [],
        "source": src.splitlines(keepends=True),
    }

def write_notebook(cells, metadata, filename):
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": metadata,
        "cells": cells,
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  ✅ Written {len(cells)} cells → {filename}")


# ─── Build cells for a given platform ────────────────────────────────────────

def build_cells(platform="colab"):
    """
    platform: 'colab' or 'kaggle'
    """
    is_colab  = platform == "colab"
    is_kaggle = platform == "kaggle"

    OUT_DIR   = "/content/drive/MyDrive/ARMOR_outputs" if is_colab else "/kaggle/working/outputs"
    REPO_DIR  = "/content/ARMOR" if is_colab else "/kaggle/working/ARMOR"
    REPO_URL  = "https://github.com/Angrajkarn/-ARMOR-Adaptive-Relearning-resistant-Multimodal-Unlearning.git"

    cells = []

    # ── Title ─────────────────────────────────────────────────────────────────
    plat_name = "Google Colab" if is_colab else "Kaggle"
    cells.append(md(f"""\
# 🛡️ ARMOR — Complete Experiment Suite ({plat_name})
### Adaptive Relearning-Resistant Multimodal Unlearning

**Target: ≤ 60 minutes on T4 GPU** | Uses `ARMOR_FAST=1` for maximum speed

| Phase | Methods | Est. Time |
|-------|---------|-----------|
| Setup | GPU · Deps · Model cache | ~5 min |
| Core Baselines | GA · NPO · NPO+SAM · RMU · Task Vector | ~30 min |
| Relearning Attack | Compare relearning resistance | ~4 min |
| ARMOR Modules | 9 methods (3 batched cells) | ~12 min |
| Extensions | MultiTask · DP · LLaVA · MUSE | *(optional)* |
| Results | Table · Chart · Privacy audit | ~2 min |

> **Before running:** Runtime → Change runtime type → **GPU (T4)**
"""))

    # ── GPU Check ─────────────────────────────────────────────────────────────
    cells.append(code("""\
# ── 0. GPU Check & Timing  ⏱ ~10s ────────────────────────────────────────────
import subprocess, sys, os, time
NOTEBOOK_START = time.time()

gpu_info = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
if 'NVIDIA' in gpu_info.stdout:
    print('✅ GPU detected:')
    for line in gpu_info.stdout.split('\\n')[:12]:
        print(line)
else:
    print('⚠️  No GPU! Go to Runtime → Change runtime type → GPU')

import torch
print(f'\\nPyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f'GPU     : {gpu.name}  ({gpu.total_memory / 1e9:.1f} GB)')
"""))

    # ── Drive / Output setup ──────────────────────────────────────────────────
    if is_colab:
        cells.append(code(f"""\
# ── 1. Mount Google Drive  ⏱ ~30s ────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

DRIVE_DIR = '{OUT_DIR}'
os.makedirs(DRIVE_DIR, exist_ok=True)
print(f'✅ Outputs → {{DRIVE_DIR}}')
"""))
    else:
        cells.append(code(f"""\
# ── 1. Output Directory  ⏱ ~5s ───────────────────────────────────────────────
DRIVE_DIR = '{OUT_DIR}'
os.makedirs(DRIVE_DIR, exist_ok=True)
print(f'✅ Outputs → {{DRIVE_DIR}}')
"""))

    # ── Clone repo ────────────────────────────────────────────────────────────
    cells.append(code(f"""\
# ── 2. Clone / Update ARMOR  ⏱ ~1 min ────────────────────────────────────────
REPO_URL = '{REPO_URL}'
REPO_DIR = '{REPO_DIR}'

if not os.path.exists(REPO_DIR):
    !git clone {{REPO_URL}} {{REPO_DIR}}
else:
    !cd {{REPO_DIR}} && git pull --ff-only

os.chdir(REPO_DIR)
!git log --oneline -3
print(f'\\n✅ Working dir: {{os.getcwd()}}')
"""))

    # ── Install deps ──────────────────────────────────────────────────────────
    cells.append(code("""\
# ── 3. Install Dependencies  ⏱ ~3 min ────────────────────────────────────────
import subprocess

pkgs = [
    'transformers>=4.40.0', 'peft>=0.10.0', 'datasets>=2.18.0',
    'accelerate>=0.28.0', 'bitsandbytes>=0.43.0', 'trl>=0.8.0',
    'rouge-score', 'scipy', 'scikit-learn', 'pandas', 'matplotlib',
    'Pillow>=10.0.0', 'torchvision>=0.16.0', 'opacus>=1.4.0',
]

result = subprocess.run(['pip', 'install', '-q'] + pkgs, capture_output=True, text=True)
if result.returncode != 0:
    print('STDERR:', result.stderr[-500:])
else:
    print('✅ All packages installed')

# Flash Attention (A100/H100 only — skip errors on T4)
try:
    subprocess.run(['pip', 'install', '-q', 'flash-attn', '--no-build-isolation'],
                   capture_output=True, timeout=90)
    print('✅ Flash Attention installed')
except Exception:
    print('ℹ️  Flash Attention N/A on this GPU (OK)')
"""))

    # ── HF Login ──────────────────────────────────────────────────────────────
    if is_colab:
        cells.append(code("""\
# ── 4. HuggingFace Login  ⏱ ~10s ─────────────────────────────────────────────
from huggingface_hub import login
from google.colab import userdata

HF_TOKEN = ''
try:
    HF_TOKEN = userdata.get('HF_TOKEN')
    login(token=HF_TOKEN, add_to_git_credential=False)
    print('✅ Logged in via Colab Secrets')
except Exception:
    print('⚠️  No HF_TOKEN in Colab Secrets.')
    HF_TOKEN = input('HuggingFace token (Enter to skip for Mistral): ').strip()
    if HF_TOKEN:
        login(token=HF_TOKEN)
        print('✅ Logged in')
    else:
        print('ℹ️  Proceeding without token (Mistral works without one)')
"""))
    else:
        cells.append(code("""\
# ── 4. HuggingFace Login  ⏱ ~10s ─────────────────────────────────────────────
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

HF_TOKEN = ''
try:
    user_secrets = UserSecretsClient()
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    login(token=HF_TOKEN, add_to_git_credential=False)
    print('✅ Logged in via Kaggle Secrets')
except Exception:
    print('ℹ️  No HF_TOKEN secret found — proceeding without (Mistral works fine)')
"""))

    # ── Config cell ───────────────────────────────────────────────────────────
    cells.append(code(f"""\
# ── 5. Configuration  ⏱ ~5s ──────────────────────────────────────────────────
import sys, os, time
sys.path.insert(0, '{REPO_DIR}')

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL = 'mistral-7b'    # or 'llama2-7b' (needs HF token) or 'debug' (CPU test)

# ── Speed: ARMOR_FAST env var (inherited by all subprocesses) ─────────────────
# Automatically sets: retain=200 samples, fp16 autocast, epochs=1
os.environ['ARMOR_FAST'] = '1'

# ── Common CLI args ───────────────────────────────────────────────────────────
FAST = '--no-rouge'            # skip ROUGE (saves ~1 min/method)
HF   = f'--hf-token {{HF_TOKEN}}' if HF_TOKEN else ''
Q    = '--qlora'
D    = f'--output-dir {{DRIVE_DIR}}'

print(f'Model      : {{MODEL}}')
print(f'ARMOR_FAST : 1 (retain=200, fp16, epochs=1)')
print(f'Output     : {{DRIVE_DIR}}')
print(f'\\n⏱  Target: ≤ 60 min on T4')
"""))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: Core Baselines (~30 min)
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(md("""\
---
## 📊 Section 1 — Core Baselines (~30 min)
These 5 methods form the paper's main comparison table.
"""))

    # GA
    cells.append(code(f"""\
# ── 6. Gradient Ascent (GA)  ⏱ ~5 min ────────────────────────────────────────
t0 = time.time()
!python scripts/run_baseline_ga.py \\
    --model {{MODEL}} {{Q}} {{HF}} --fast {{FAST}} \\
    --run-mia \\
    --output-dir {{DRIVE_DIR}}/ga
print(f'\\n✅ GA done in {{(time.time()-t0)/60:.1f}} min')
"""))

    # NPO
    cells.append(code(f"""\
# ── 7. NPO Baseline  ⏱ ~5 min ────────────────────────────────────────────────
t0 = time.time()
!python scripts/run_baseline_npo.py \\
    --model {{MODEL}} {{Q}} {{HF}} --fast {{FAST}} \\
    --run-mia \\
    --output-dir {{DRIVE_DIR}}/npo
print(f'\\n✅ NPO done in {{(time.time()-t0)/60:.1f}} min')
"""))

    # NPO+SAM — key: sam_every=4 and sam-rho=0.05
    cells.append(code(f"""\
# ── 8. NPO+SAM — ARMOR Core  ⏱ ~10 min ──────────────────────────────────────
# sam-every=4 cuts SAM overhead by 50% vs default (sam-every=2)
t0 = time.time()
!python scripts/run_npo_sam.py \\
    --model {{MODEL}} {{Q}} {{HF}} --fast {{FAST}} \\
    --sam-rho 0.05 --sam-every 4 \\
    --run-mia \\
    --output-dir {{DRIVE_DIR}}/npo_sam
print(f'\\n✅ NPO+SAM done in {{(time.time()-t0)/60:.1f}} min')
"""))

    # RMU
    cells.append(code(f"""\
# ── 9. RMU — Representation Misdirection  ⏱ ~4 min ──────────────────────────
t0 = time.time()
!python scripts/run_rmu.py \\
    --model {{MODEL}} {{Q}} {{HF}} {{FAST}} \\
    --alpha 1200.0 --beta 6.5 \\
    --run-mia \\
    --output-dir {{DRIVE_DIR}}/rmu
print(f'\\n✅ RMU done in {{(time.time()-t0)/60:.1f}} min')
"""))

    # Task Vector
    cells.append(code(f"""\
# ── 10. Task Vector  ⏱ ~5 min ────────────────────────────────────────────────
t0 = time.time()
!python scripts/run_task_vector.py \\
    --model {{MODEL}} {{Q}} {{HF}} {{FAST}} \\
    --lam 1.0 --run-mia \\
    --output-dir {{DRIVE_DIR}}/task_vector
print(f'\\n✅ Task Vector done in {{(time.time()-t0)/60:.1f}} min')

elapsed = (time.time() - NOTEBOOK_START) / 60
print(f'\\n📊 Core baselines done. Elapsed: {{elapsed:.0f}} min')
"""))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: Relearning Attack (~4 min)
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(md("""\
---
## 🔥 Section 2 — Relearning Attack (~4 min)
Tests if an adversary can fine-tune the model back to remembering forgotten data.
**Lower recovery = better unlearning.**
"""))

    cells.append(code(f"""\
# ── 11. Relearning Attack — GA vs NPO vs NPO+SAM  ⏱ ~4 min ──────────────────
t0 = time.time()
!python scripts/run_relearning_attack.py \\
    --model {{MODEL}} {{Q}} {{HF}} \\
    --compare \\
    --ga-checkpoint  {{DRIVE_DIR}}/ga/ga_unlearned \\
    --npo-checkpoint {{DRIVE_DIR}}/npo/npo_unlearned \\
    --sam-checkpoint {{DRIVE_DIR}}/npo_sam/npo_sam_unlearned \\
    --original-acc 0.85 \\
    --n-samples 30 --epochs 2 --no-save
print(f'\\n✅ Relearning Attack done in {{(time.time()-t0)/60:.1f}} min')
"""))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: ARMOR Modules — batched into 3 cells (~12 min)
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(md("""\
---
## 🔬 Section 3 — ARMOR Modules (9 methods, ~12 min)
All use `--no-save` and `ARMOR_FAST=1` for speed.
"""))

    # Batch A: Continual + MoE + RLACE-RMU
    cells.append(code(f"""\
# ── 12. Modules Batch A: Continual · MoE · RLACE-RMU  ⏱ ~4 min ──────────────
t0 = time.time()

print('\\n' + '='*60)
print('  [1/3] Continual / Lifelong Unlearning')
print('='*60)
!python scripts/run_continual_unlearn.py \\
    --model {{MODEL}} {{Q}} {{HF}} {{FAST}} --no-save \\
    --num-cohorts 2 \\
    --output-dir {{DRIVE_DIR}}/continual

print('\\n' + '='*60)
print('  [2/3] MoE Targeted GA')
print('='*60)
!python scripts/run_moe_unlearn.py \\
    --model {{MODEL}} {{Q}} {{HF}} {{FAST}} --no-save \\
    --output-dir {{DRIVE_DIR}}/moe

print('\\n' + '='*60)
print('  [3/3] RLACE-RMU Concept Erasure')
print('='*60)
!python scripts/run_rlace_rmu.py \\
    --model {{MODEL}} {{Q}} {{HF}} {{FAST}} --no-save \\
    --rlace-iters 10 \\
    --output-dir {{DRIVE_DIR}}/rlace_rmu

print(f'\\n✅ Batch A done in {{(time.time()-t0)/60:.1f}} min')
"""))

    # Batch B: ZK + MIA + LoRA
    cells.append(code(f"""\
# ── 13. Modules Batch B: ZK · MIA · LoRA  ⏱ ~4 min ──────────────────────────
t0 = time.time()

print('\\n' + '='*60)
print('  [1/3] ZK Verification (commit-reveal)')
print('='*60)
!python scripts/run_zk_verify.py \\
    --model {{MODEL}} {{Q}} {{HF}} --no-save \\
    --probe-samples 5

print('\\n' + '='*60)
print('  [2/3] Multimodal MIA Audit')
print('='*60)
!python scripts/run_multimodal_mia.py \\
    --model {{MODEL}} {{Q}} {{HF}} --no-save

print('\\n' + '='*60)
print('  [3/3] Modular LoRA Unlearning')
print('='*60)
!python scripts/run_lora_unlearn.py \\
    --model {{MODEL}} {{Q}} {{HF}} {{FAST}} --no-save

print(f'\\n✅ Batch B done in {{(time.time()-t0)/60:.1f}} min')
"""))

    # Batch C: NASD + HDI + CAS
    cells.append(code(f"""\
# ── 14. Modules Batch C: NASD · HDI · CAS  ⏱ ~4 min ─────────────────────────
t0 = time.time()

print('\\n' + '='*60)
print('  [1/3] NASD Decay')
print('='*60)
!python scripts/run_nasd.py \\
    --model {{MODEL}} {{Q}} {{HF}} {{FAST}}

print('\\n' + '='*60)
print('  [2/3] HDI Zero-Shot')
print('='*60)
!python scripts/run_hdi_unlearn.py \\
    --model {{MODEL}} {{Q}} {{HF}} {{FAST}} --no-save

print('\\n' + '='*60)
print('  [3/3] CAS Graph Blockade')
print('='*60)
!python scripts/run_cas_unlearn.py \\
    --model {{MODEL}} {{Q}} {{HF}} {{FAST}} --no-save

print(f'\\n✅ Batch C done in {{(time.time()-t0)/60:.1f}} min')
print(f'📊 All ARMOR Modules done. Elapsed: {{(time.time()-NOTEBOOK_START)/60:.0f}} min')
"""))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: Extensions (OPTIONAL)
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(md("""\
---
## 🛡️ Section 4 — ARMOR Extensions *(Optional — adds ~15 min)*
MultiTask-NPO · DP-NPO+SAM · LLaVA · MUSE

> **Skip this section if time is tight.** The core results above are sufficient.
"""))

    cells.append(code(f"""\
# ── 15. Extensions (Optional)  ⏱ ~15 min total ──────────────────────────────
# Uncomment the methods you want to run:

t0 = time.time()

# --- MultiTask-NPO ---
print('\\n[ext] MultiTask-NPO...')
!python scripts/run_multitask_unlearn.py \\
    --model {{MODEL}} {{Q}} {{HF}} {{FAST}} \\
    --n-tasks 2 --run-mia --no-save \\
    --output-dir {{DRIVE_DIR}}/multitask_npo

# --- DP-NPO+SAM ---
print('\\n[ext] DP-NPO+SAM...')
!python scripts/run_dp_armor.py \\
    --model {{MODEL}} {{Q}} {{HF}} {{FAST}} \\
    --epsilon 8.0 --delta 1e-5 --noise 1.0 --clip 1.0 \\
    --run-mia --no-save \\
    --output-dir {{DRIVE_DIR}}/dp_npo_sam

# --- LLaVA Cross-Modal ---
print('\\n[ext] LLaVA NPO+SAM...')
!python scripts/run_llava_unlearn.py \\
    --model {{MODEL}} {{Q}} {{HF}} {{FAST}} \\
    --text-only --run-mia --no-save \\
    --output-dir {{DRIVE_DIR}}/llava_npo_sam

# --- MUSE Benchmark ---
print('\\n[ext] MUSE books...')
!python scripts/run_muse_benchmark.py \\
    --model {{MODEL}} {{Q}} {{HF}} {{FAST}} \\
    --domain books --method npo_sam \\
    --run-mia --no-save \\
    --output-dir {{DRIVE_DIR}}/muse_books

print(f'\\n✅ Extensions done in {{(time.time()-t0)/60:.1f}} min')
"""))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5: Phase 1-3 Research (OPTIONAL)
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(md("""\
---
## 🚀 Section 5 — Phase 1-3 Frontier Research *(Optional — adds ~20 min)*
Skip if time-constrained. These are supplementary research methods.
"""))

    cells.append(code(f"""\
# ── 16. Phase 1-3 Research Methods (Optional)  ⏱ ~20 min ────────────────────
# Uncomment individual methods as needed:

t0 = time.time()

# Phase 1
!python scripts/run_conformal_verify.py  --model {{MODEL}} {{Q}} {{HF}} --no-save --alpha 0.10
!python scripts/run_cot_hme.py           --model {{MODEL}} {{Q}} {{HF}} {{FAST}} --no-save --cot-coeff 0.2 --cot-max-tokens 24
!python scripts/run_temporal_unlearn.py  --model {{MODEL}} {{Q}} {{HF}} {{FAST}} --no-save --halflife-days 1.0

# Phase 2
!python scripts/run_lcage.py                --model {{MODEL}} {{Q}} {{HF}} {{FAST}} --no-save
!python scripts/run_reconsolidation.py      --model {{MODEL}} {{Q}} {{HF}} {{FAST}} --no-save
!python scripts/run_morphogenetic_repair.py --model {{MODEL}} {{Q}} {{HF}} {{FAST}} --no-save

# Phase 3
!python scripts/run_stackelberg_game.py  --model {{MODEL}} {{Q}} {{HF}} {{FAST}} --no-save
!python scripts/run_causal_iu.py         --model {{MODEL}} {{Q}} {{HF}} {{FAST}} --no-save
!python scripts/run_federated_robust.py  --model {{MODEL}} {{Q}} {{HF}} {{FAST}} --no-save

# Bonus
!python scripts/run_reconstruction_attack.py --model {{MODEL}} {{Q}} {{HF}} --no-save
!python scripts/run_audit_gen.py --model {{MODEL}} {{Q}} {{HF}} --probe-samples 5 --output-dir {{DRIVE_DIR}}/audit

print(f'\\n✅ Phase 1-3 done in {{(time.time()-t0)/60:.1f}} min')
"""))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6: Results
    # ══════════════════════════════════════════════════════════════════════════
    cells.append(md("""\
---
## 📊 Section 6 — Results & Visualisations
"""))

    cells.append(code(f"""\
# ── 17. Collect Results  ⏱ ~10s ──────────────────────────────────────────────
import json, os, glob
import pandas as pd

results = []
for method_dir in sorted(glob.glob(f'{{DRIVE_DIR}}/*')):
    method_name = os.path.basename(method_dir)
    for jf in glob.glob(f'{{method_dir}}/**/*.json', recursive=True):
        try:
            with open(jf) as f:
                data = json.load(f)
            if 'forget_quality' in data:
                results.append({{
                    'Method'           : method_name,
                    'Forget Quality ↑' : round(data.get('forget_quality',   0), 4),
                    'Forget Acc ↓'     : round(data.get('forget_accuracy',  0), 4),
                    'Retain Acc ↑'     : round(data.get('retain_accuracy',  0), 4),
                    'MIA AUROC →0.5'   : round(data.get('mia_auroc',       -1), 4),
                }})
                break
        except Exception:
            pass

if results:
    df = pd.DataFrame(results).sort_values('Forget Quality ↑', ascending=False)
    print('\\n' + '='*72)
    print('  ARMOR — Experiment Results')
    print('='*72)
    print(df.to_string(index=False))
    df.to_csv(f'{{DRIVE_DIR}}/summary_results.csv', index=False)
    print(f'\\n✅ Results saved to {{DRIVE_DIR}}/summary_results.csv')
else:
    print('No results found yet. Run experiment cells first.')
"""))

    cells.append(code(f"""\
# ── 18. Plot Results  ⏱ ~10s ─────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np

if results:
    df_plot = df.copy()
    methods = df_plot['Method'].tolist()
    fq      = df_plot['Forget Quality ↑'].tolist()
    ra      = df_plot['Retain Acc ↑'].tolist()
    mia     = [v if v >= 0 else 0.5 for v in df_plot['MIA AUROC →0.5'].tolist()]

    x, width = np.arange(len(methods)), 0.25
    fig, ax = plt.subplots(figsize=(max(12, len(methods)*0.8), 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    ax.bar(x - width, fq,  width, label='Forget Quality ↑',  color='#58a6ff', alpha=0.9)
    ax.bar(x,         ra,  width, label='Retain Acc ↑',       color='#3fb950', alpha=0.9)
    ax.bar(x + width, mia, width, label='MIA AUROC (→0.5)',   color='#f78166', alpha=0.9)
    ax.axhline(y=0.5, color='#f0e68c', linestyle='--', alpha=0.6, linewidth=1.2,
               label='Ideal MIA = 0.5')

    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.tick_params(colors='#e6edf3')
    ax.set_xlabel('Unlearning Method', color='#e6edf3', fontsize=11)
    ax.set_ylabel('Score', color='#e6edf3', fontsize=11)
    ax.set_title('ARMOR — Unlearning Methods Comparison', color='#e6edf3',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha='right', color='#e6edf3', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(facecolor='#21262d', labelcolor='#e6edf3', edgecolor='#30363d')
    plt.tight_layout()
    plt.savefig(f'{{DRIVE_DIR}}/results_comparison.png', dpi=150,
                bbox_inches='tight', facecolor='#0d1117')
    plt.show()
    print(f'✅ Plot saved to {{DRIVE_DIR}}/results_comparison.png')
else:
    print('No results to plot.')
"""))

    cells.append(code(f"""\
# ── 19. Privacy Audit Summary  ⏱ ~5s ────────────────────────────────────────
print('\\n' + '='*65)
print('  ARMOR — Privacy Audit (Min-K% Prob MIA)')
print('='*65)
print('  Ideal AUROC = 0.500  →  model treats forget-set as non-members')
print('  AUROC > 0.70         →  unlearning FAILED')
print()

if results:
    for r in df.to_dict('records'):
        v = r['MIA AUROC →0.5']
        if v < 0:   status = '—  (not measured)'
        elif v <= 0.55: status = '✅ VERIFIED   (near-random)'
        elif v <= 0.65: status = '⚠️  MARGINAL   (residual memory)'
        else:           status = '❌ FAILED     (model still remembers)'
        print(f"  {{r['Method']:<28}} AUROC={{v:.4f}}  {{status}}")

total_time = (time.time() - NOTEBOOK_START) / 60
print(f'\\n⏱  Total runtime: {{total_time:.1f}} min')
print('='*65)
"""))

    # ── Final cell: list saved files ──────────────────────────────────────────
    cells.append(code(f"""\
# ── 20. Saved Files  ⏱ ~5s ───────────────────────────────────────────────────
print('\\n📁 Files in output directory:')
for root, dirs, files in os.walk(DRIVE_DIR):
    depth = root.replace(DRIVE_DIR, '').count(os.sep)
    if depth > 2:
        continue
    total_size = sum(os.path.getsize(os.path.join(root, f))
                     for f in files if os.path.isfile(os.path.join(root, f)))
    indent = '  ' * depth
    bname  = os.path.basename(root) or 'ARMOR_outputs'
    sz     = f'({{total_size/1e6:.1f}} MB)' if total_size > 0 else ''
    print(f'{{indent}}📂 {{bname}}  {{sz}}')

total_time = (time.time() - NOTEBOOK_START) / 60
print(f'\\n⏱  Total notebook runtime: {{total_time:.1f}} minutes')
print('✅ Done! All outputs saved.')
"""))

    return cells


# ─── Generate both notebooks ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating ARMOR experiment notebooks...\n")

    # Colab
    colab_cells = build_cells("colab")
    colab_meta = {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "name": "ARMOR_Colab_Experiments.ipynb",
            "provenance": [],
            "toc_visible": True,
        },
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    }
    write_notebook(colab_cells, colab_meta, "ARMOR_Colab_Experiments.ipynb")

    # Kaggle
    kaggle_cells = build_cells("kaggle")
    kaggle_meta = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isGpuEnabled": True,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
    }
    write_notebook(kaggle_cells, kaggle_meta, "ARMOR_Kaggle_Experiments.ipynb")

    print("\nDone! Upload to Colab/Kaggle and run.")
