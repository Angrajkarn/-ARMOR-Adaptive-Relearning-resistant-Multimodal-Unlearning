"""
gen_colab.py — generates ARMOR_Colab_Experiments.ipynb
Run:  python gen_colab.py
"""
import json, sys

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [],
        "source": src.splitlines(keepends=True),
    }

cells = []

# ── 0. Title ──────────────────────────────────────────────────────────────────
cells.append(md("""\
# 🛡️ ARMOR: Adaptive Relearning-Resistant Multimodal Unlearning
### Complete GPU Experiment Suite — Google Colab (T4 / A100)

**Estimated total runtime: ~90 minutes on T4 GPU**

| Section | Methods | Est. Time |
|---------|---------|-----------|
| Setup | GPU check · Drive · Install | ~10 min |
| Core Baselines | GA · NPO · NPO+SAM · RMU · Task Vector | ~40 min |
| ARMOR Extensions | MultiTask-NPO · DP-NPO+SAM · LLaVA · MUSE | ~30 min |
| Relearning Attack | Compare resistance across methods | ~5 min |
| ARMOR Modules | Continual · MoE · RLACE · ZK · MIA · LoRA | ~10 min |
| Phase 1–3 Research | CU-AR · CoT-HME · TKDU · LCAGE · SAUG · … | optional |
| Results & Plots | Table · Chart · Privacy audit | ~3 min |

> **Tips:** Runtime → Change runtime type → **GPU** before running.  
> Each cell can be re-run independently. ⏱ estimates shown at top of each cell.
"""))

# ── 1. GPU Check ──────────────────────────────────────────────────────────────
cells.append(code("""\
# ── 0. GPU Check  ⏱ ~30s ─────────────────────────────────────────────────────
import subprocess, sys, os, time
NOTEBOOK_START = time.time()

gpu_info = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
if 'NVIDIA' in gpu_info.stdout:
    print('✅ GPU detected:')
    print(gpu_info.stdout[:500])
else:
    print('⚠️  No GPU! Go to Runtime → Change runtime type → GPU')
    print('   Continuing in CPU mode (will be MUCH slower)')

import torch
print(f'\\nPyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f'GPU     : {gpu.name}')
    print(f'VRAM    : {gpu.total_memory / 1e9:.1f} GB')
    if gpu.total_memory < 14e9:
        print('⚠️  Less than 14 GB VRAM — use QLoRA (already default)')
"""))

# ── 2. Drive Mount ────────────────────────────────────────────────────────────
cells.append(code("""\
# ── 1. Mount Google Drive  ⏱ ~30s ────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

DRIVE_DIR = '/content/drive/MyDrive/ARMOR_outputs'
os.makedirs(DRIVE_DIR, exist_ok=True)
print(f'✅ Outputs will be saved to: {DRIVE_DIR}')
"""))

# ── 3. Clone Repo ─────────────────────────────────────────────────────────────
cells.append(code("""\
# ── 2. Clone / Update ARMOR Repository  ⏱ ~1 min ────────────────────────────
REPO_URL = 'https://github.com/Angrajkarn/-ARMOR-Adaptive-Relearning-resistant-Multimodal-Unlearning.git'
REPO_DIR = '/content/ARMOR'

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    !cd {REPO_DIR} && git pull   # pull latest fixes (QLoRA optimizer patches)

os.chdir(REPO_DIR)
!git log --oneline -3
print(f'\\n✅ Working directory: {os.getcwd()}')
"""))

# ── 4. Install Deps ───────────────────────────────────────────────────────────
cells.append(code("""\
# ── 3. Install Dependencies  ⏱ ~3-4 min ─────────────────────────────────────
import subprocess

pkgs = [
    'transformers>=4.40.0',
    'peft>=0.10.0',
    'datasets>=2.18.0',
    'accelerate>=0.28.0',
    'bitsandbytes>=0.43.0',
    'trl>=0.8.0',
    'rouge-score',
    'scipy',
    'scikit-learn',
    'pandas',
    'matplotlib',
    'Pillow>=10.0.0',
    'torchvision>=0.16.0',
    'opacus>=1.4.0',
]

print('Installing packages...')
result = subprocess.run(
    ['pip', 'install', '-q'] + pkgs,
    capture_output=True, text=True
)
if result.returncode != 0:
    print('STDERR:', result.stderr[-500:])
else:
    print('✅ All packages installed')

# Optional: Flash-Attention (A100/H100 only, skip on T4)
try:
    result = subprocess.run(
        ['pip', 'install', '-q', 'flash-attn', '--no-build-isolation'],
        capture_output=True, timeout=90
    )
    if result.returncode == 0:
        print('✅ Flash Attention installed (A100/H100 speed boost)')
    else:
        print('ℹ️  Flash Attention not available on this GPU (OK for T4)')
except Exception:
    print('ℹ️  Flash Attention skipped')
"""))

# ── 5. HF Login ───────────────────────────────────────────────────────────────
cells.append(code("""\
# ── 4. HuggingFace Login  ⏱ ~10s ─────────────────────────────────────────────
# Mistral-7B does NOT require gated access.
# LLaMA-2 requires you to accept the license at huggingface.co/meta-llama/Llama-2-7b-hf

from huggingface_hub import login
from google.colab import userdata

HF_TOKEN = ''
try:
    HF_TOKEN = userdata.get('HF_TOKEN')
    login(token=HF_TOKEN, add_to_git_credential=False)
    print('✅ Logged into HuggingFace via Colab Secrets')
except Exception:
    print('⚠️  No HF_TOKEN in Colab Secrets.')
    HF_TOKEN = input('Enter HuggingFace token (Enter to skip for Mistral): ').strip()
    if HF_TOKEN:
        login(token=HF_TOKEN)
        print('✅ Logged in')
    else:
        print('ℹ️  Proceeding without token (Mistral-7B works without one)')
"""))

# ── 6. Config ─────────────────────────────────────────────────────────────────
cells.append(code("""\
# ── 5. Configuration  ⏱ ~5s ──────────────────────────────────────────────────
import sys
sys.path.insert(0, '/content/ARMOR')

# ── Model choice ──────────────────────────────────────────────────────────────
# 'mistral-7b'  → Mistral-7B-v0.1  (recommended, 16GB T4 with QLoRA)
# 'llama2-7b'   → LLaMA-2-7B       (requires HF gated token)
# 'distilgpt2'  → DistilGPT-2      (debug/CPU only, very fast)
MODEL = 'mistral-7b'

# ── Speed flags applied to EVERY experiment ──────────────────────────────────
# --fast          : retain-set subsampling + FP16 autocast (3-5x speedup)
# --no-rouge      : skip ROUGE scoring (saves ~1 min per method)
# --epochs 1      : single training epoch (sufficient for research comparison)
# These reduce accuracy slightly but keep total runtime ≤ 2 hours
FAST = '--fast --no-rouge --epochs 1'

# ── Common args ───────────────────────────────────────────────────────────────
HF  = f'--hf-token {HF_TOKEN}' if HF_TOKEN else ''
OUT = f'--output-dir {DRIVE_DIR}'
Q   = '--qlora'

print(f'Model      : {MODEL}')
print(f'QLoRA      : True (4-bit)')
print(f'Speed flags: {FAST}')
print(f'Output dir : {DRIVE_DIR}')
print()
print('⏱  Estimated total runtime breakdown:')
print('   GA + NPO + NPO+SAM + RMU + TaskVec  : ~40 min')
print('   MultiTask + DP + LLaVA + MUSE        : ~30 min')
print('   Relearning Attack                    : ~5 min')
print('   ARMOR Modules (9 methods)            : ~10 min')
print('   Results + Plots                      : ~3 min')
print('   ─────────────────────────────────────')
print('   Total                                : ~88 min')
"""))

# ── SECTION: Core Baselines ───────────────────────────────────────────────────
cells.append(md("""\
---
## 📊 Section 1 — Core Baselines
*Estimated: ~40 minutes*
"""))

# GA
cells.append(code("""\
# ── 6. Gradient Ascent (GA)  ⏱ ~7 min ────────────────────────────────────────
t0 = time.time()
!python scripts/run_baseline_ga.py \\
    --model {MODEL} {Q} {HF} {FAST} \\
    --run-mia \\
    --output-dir {DRIVE_DIR}/ga
print(f'\\n✅ GA done in {(time.time()-t0)/60:.1f} min')
"""))

# NPO
cells.append(code("""\
# ── 7. NPO Baseline  ⏱ ~7 min ────────────────────────────────────────────────
t0 = time.time()
!python scripts/run_baseline_npo.py \\
    --model {MODEL} {Q} {HF} {FAST} \\
    --run-mia \\
    --output-dir {DRIVE_DIR}/npo
print(f'\\n✅ NPO done in {(time.time()-t0)/60:.1f} min')
"""))

# NPO+SAM
cells.append(code("""\
# ── 8. NPO + SAM — ARMOR Core Method  ⏱ ~9 min ──────────────────────────────
t0 = time.time()
!python scripts/run_npo_sam.py \\
    --model {MODEL} {Q} {HF} {FAST} \\
    --sam-rho 0.05 \\
    --run-mia \\
    --output-dir {DRIVE_DIR}/npo_sam
print(f'\\n✅ NPO+SAM done in {(time.time()-t0)/60:.1f} min')
"""))

# RMU
cells.append(code("""\
# ── 9. RMU — Representation Misdirection Unlearning  ⏱ ~6 min ───────────────
t0 = time.time()
!python scripts/run_rmu.py \\
    --model {MODEL} {Q} {HF} {FAST} \\
    --alpha 1200.0 --beta 6.5 \\
    --run-mia \\
    --output-dir {DRIVE_DIR}/rmu
print(f'\\n✅ RMU done in {(time.time()-t0)/60:.1f} min')
"""))

# Task Vector
cells.append(code("""\
# ── 10. Task Vector Unlearning  ⏱ ~7 min ─────────────────────────────────────
t0 = time.time()
!python scripts/run_task_vector.py \\
    --model {MODEL} {Q} {HF} {FAST} \\
    --lam 1.0 --run-mia \\
    --output-dir {DRIVE_DIR}/task_vector
print(f'\\n✅ Task Vector done in {(time.time()-t0)/60:.1f} min')
"""))

# ── SECTION: ARMOR Extensions ─────────────────────────────────────────────────
cells.append(md("""\
---
## 🛡️ Section 2 — ARMOR Extensions
*Estimated: ~30 minutes*
"""))

# MultiTask-NPO
cells.append(code("""\
# ── 11. Multi-Task NPO (2 topics simultaneously)  ⏱ ~8 min ──────────────────
# Orthogonal gradient projection prevents task interference.
# Uses per-parameter Gram-Schmidt (no flat-tensor OOM).
t0 = time.time()
!python scripts/run_multitask_unlearn.py \\
    --model {MODEL} {Q} {HF} {FAST} \\
    --n-tasks 2 \\
    --run-mia \\
    --output-dir {DRIVE_DIR}/multitask_npo
print(f'\\n✅ MultiTask-NPO done in {(time.time()-t0)/60:.1f} min')
"""))

# DP-NPO+SAM
cells.append(code("""\
# ── 12. DP-NPO+SAM — Full Differential Privacy Stack  ⏱ ~8 min ──────────────
# Combines NPO+SAM with Opacus DP-SGD (ε=8, δ=1e-5 guarantee).
t0 = time.time()
!python scripts/run_dp_armor.py \\
    --model {MODEL} {Q} {HF} {FAST} \\
    --epsilon 8.0 --delta 1e-5 --noise 1.0 --clip 1.0 \\
    --run-mia \\
    --output-dir {DRIVE_DIR}/dp_npo_sam
print(f'\\n✅ DP-NPO+SAM done in {(time.time()-t0)/60:.1f} min')
"""))

# LLaVA
cells.append(code("""\
# ── 13. LLaVA Cross-Modal Unlearning (text backbone)  ⏱ ~6 min ───────────────
t0 = time.time()
!python scripts/run_llava_unlearn.py \\
    --model {MODEL} {Q} {HF} {FAST} \\
    --text-only \\
    --run-mia \\
    --output-dir {DRIVE_DIR}/llava_npo_sam
print(f'\\n✅ LLaVA NPO+SAM done in {(time.time()-t0)/60:.1f} min')
"""))

# MUSE
cells.append(code("""\
# ── 14. MUSE Benchmark (books domain)  ⏱ ~7 min ─────────────────────────────
t0 = time.time()
!python scripts/run_muse_benchmark.py \\
    --model {MODEL} {Q} {HF} {FAST} \\
    --domain books --method npo_sam \\
    --run-mia \\
    --output-dir {DRIVE_DIR}/muse_books
print(f'\\n✅ MUSE books done in {(time.time()-t0)/60:.1f} min')
"""))

# ── SECTION: Relearning Attack ────────────────────────────────────────────────
cells.append(md("""\
---
## 🔥 Section 3 — Relearning Attack
*Estimated: ~5 minutes*

Tests how easily an adversary can fine-tune a forgotten model back to remembering the data.  
**Lower Acc Jump = more relearning-resistant = better.**
"""))

cells.append(code("""\
# ── 15. Relearning Attack — Compare GA vs NPO vs NPO+SAM  ⏱ ~5 min ──────────
t0 = time.time()
!python scripts/run_relearning_attack.py \\
    --model {MODEL} {Q} {HF} \\
    --compare \\
    --original-acc 0.85 \\
    --n-samples 50 \\
    --epochs 2 \\
    --no-save
print(f'\\n✅ Relearning Attack done in {(time.time()-t0)/60:.1f} min')
"""))

# ── SECTION: ARMOR Modules ────────────────────────────────────────────────────
cells.append(md("""\
---
## 🔬 Section 4 — ARMOR Modules (9 Methods)
*Estimated: ~10 minutes total (all use `--fast --no-save`)*
"""))

cells.append(code("""\
# ── 16. Module 1: Continual / Lifelong Unlearning  ⏱ ~1.5 min ───────────────
t0 = time.time()
!python scripts/run_continual_unlearn.py \\
    --model {MODEL} {Q} {HF} {FAST} --no-save \\
    --num-cohorts 2 \\
    --output-dir {DRIVE_DIR}/continual
print(f'✅ Continual done in {(time.time()-t0)/60:.1f} min')
"""))

cells.append(code("""\
# ── 17. Module 2: MoE Targeted GA  ⏱ ~1.5 min ──────────────────────────────
t0 = time.time()
!python scripts/run_moe_unlearn.py \\
    --model {MODEL} {Q} {HF} {FAST} --no-save \\
    --output-dir {DRIVE_DIR}/moe
print(f'✅ MoE done in {(time.time()-t0)/60:.1f} min')
"""))

cells.append(code("""\
# ── 18. Module 3: RLACE-RMU (Concept Erasure)  ⏱ ~1 min ────────────────────
t0 = time.time()
!python scripts/run_rlace_rmu.py \\
    --model {MODEL} {Q} {HF} {FAST} --no-save \\
    --rlace-iters 10 \\
    --output-dir {DRIVE_DIR}/rlace_rmu
print(f'✅ RLACE-RMU done in {(time.time()-t0)/60:.1f} min')
"""))

cells.append(code("""\
# ── 19. Module 4: ZK Verification  ⏱ ~1 min ──────────────────────────────────
t0 = time.time()
!python scripts/run_zk_verify.py \\
    --model {MODEL} {Q} {HF} --no-save \\
    --probe-samples 8
print(f'✅ ZK Verify done in {(time.time()-t0)/60:.1f} min')
"""))

cells.append(code("""\
# ── 20. Module 5: Multimodal MIA Audit  ⏱ ~1 min ────────────────────────────
t0 = time.time()
!python scripts/run_multimodal_mia.py \\
    --model {MODEL} {Q} {HF} --no-save
print(f'✅ Multimodal MIA done in {(time.time()-t0)/60:.1f} min')
"""))

cells.append(code("""\
# ── 21. Module 6: Modular LoRA Unlearning  ⏱ ~1 min ──────────────────────────
t0 = time.time()
!python scripts/run_lora_unlearn.py \\
    --model {MODEL} {Q} {HF} {FAST} --no-save
print(f'✅ Modular LoRA done in {(time.time()-t0)/60:.1f} min')
"""))

cells.append(code("""\
# ── 22. Module 7: NASD Decay  ⏱ ~1 min ──────────────────────────────────────
t0 = time.time()
!python scripts/run_nasd.py \\
    --model {MODEL} {Q} {HF} {FAST}
print(f'✅ NASD done in {(time.time()-t0)/60:.1f} min')
"""))

cells.append(code("""\
# ── 23. Module 8: HDI Zero-Shot  ⏱ ~1 min ───────────────────────────────────
t0 = time.time()
!python scripts/run_hdi_unlearn.py \\
    --model {MODEL} {Q} {HF} {FAST} --no-save
print(f'✅ HDI done in {(time.time()-t0)/60:.1f} min')
"""))

cells.append(code("""\
# ── 24. Module 9: CAS Graph Blockade  ⏱ ~1 min ──────────────────────────────
t0 = time.time()
!python scripts/run_cas_unlearn.py \\
    --model {MODEL} {Q} {HF} {FAST} --no-save
print(f'✅ CAS done in {(time.time()-t0)/60:.1f} min')

print(f'\\n📊 All ARMOR Modules complete. Total elapsed: {(time.time()-NOTEBOOK_START)/60:.1f} min')
"""))

# ── SECTION: Phase 1-3 Research (Optional) ───────────────────────────────────
cells.append(md("""\
---
## 🚀 Section 5 — Phase 1-3 Frontier Research Methods *(Optional)*
*These add ~20 minutes. Run only if you have time or an A100.*

Skip this section if you're on T4 with limited time.
"""))

cells.append(code("""\
# ── 25. Reconstruction Attack (Model Inversion)  ⏱ ~3 min ───────────────────
t0 = time.time()
!python scripts/run_reconstruction_attack.py \\
    --model {MODEL} {Q} {HF} --no-save
print(f'✅ Reconstruction Attack done in {(time.time()-t0)/60:.1f} min')
"""))

cells.append(code("""\
# ── 26. Audit Certificate Generator (GDPR)  ⏱ ~5 min ────────────────────────
t0 = time.time()
!python scripts/run_audit_gen.py \\
    --model {MODEL} {Q} {HF} \\
    --probe-samples 8 \\
    --output-dir {DRIVE_DIR}/audit
print(f'✅ Audit Certificate done in {(time.time()-t0)/60:.1f} min')
"""))

cells.append(code("""\
# ── 27. Phase 1: CU-AR (Conformal Unlearning Verification)  ⏱ ~1 min ─────────
t0 = time.time()
!python scripts/run_conformal_verify.py \\
    --model {MODEL} {Q} {HF} --no-save \\
    --alpha 0.10
print(f'✅ CU-AR done in {(time.time()-t0)/60:.1f} min')
"""))

cells.append(code("""\
# ── 28. Phase 1: CoT-HME (Chain-of-Thought Erasure)  ⏱ ~3 min ───────────────
t0 = time.time()
!python scripts/run_cot_hme.py \\
    --model {MODEL} {Q} {HF} {FAST} --no-save \\
    --cot-coeff 0.2 --cot-max-tokens 24
print(f'✅ CoT-HME done in {(time.time()-t0)/60:.1f} min')
"""))

cells.append(code("""\
# ── 29. Phase 1: TKDU (Temporal Knowledge Decay)  ⏱ ~2 min ──────────────────
t0 = time.time()
!python scripts/run_temporal_unlearn.py \\
    --model {MODEL} {Q} {HF} {FAST} --no-save \\
    --halflife-days 1.0
print(f'✅ TKDU done in {(time.time()-t0)/60:.1f} min')
"""))

cells.append(code("""\
# ── 30. Phase 2: LCAGE (Latent Concept Association Graph)  ⏱ ~2 min ──────────
t0 = time.time()
!python scripts/run_lcage.py \\
    --model {MODEL} {Q} {HF} {FAST} --no-save
print(f'✅ LCAGE done in {(time.time()-t0)/60:.1f} min')
"""))

cells.append(code("""\
# ── 31. Phase 2: NRU (Neural Reconsolidation)  ⏱ ~3 min ─────────────────────
t0 = time.time()
!python scripts/run_reconsolidation.py \\
    --model {MODEL} {Q} {HF} {FAST} --no-save
print(f'✅ NRU done in {(time.time()-t0)/60:.1f} min')
"""))

cells.append(code("""\
# ── 32. Phase 2: MWRP (Morphogenetic Weight Regeneration)  ⏱ ~3 min ──────────
t0 = time.time()
!python scripts/run_morphogenetic_repair.py \\
    --model {MODEL} {Q} {HF} {FAST} --no-save
print(f'✅ MWRP done in {(time.time()-t0)/60:.1f} min')
"""))

cells.append(code("""\
# ── 33. Phase 3: SAUG (Stackelberg Game)  ⏱ ~3 min ──────────────────────────
t0 = time.time()
!python scripts/run_stackelberg_game.py \\
    --model {MODEL} {Q} {HF} {FAST} --no-save
print(f'✅ SAUG done in {(time.time()-t0)/60:.1f} min')
"""))

cells.append(code("""\
# ── 34. Phase 3: CIU (Causal do-Calculus)  ⏱ ~3 min ────────────────────────
t0 = time.time()
!python scripts/run_causal_iu.py \\
    --model {MODEL} {Q} {HF} {FAST} --no-save
print(f'✅ CIU done in {(time.time()-t0)/60:.1f} min')
"""))

cells.append(code("""\
# ── 35. Phase 3: BRFU (Byzantine-Robust Federated)  ⏱ ~3 min ────────────────
t0 = time.time()
!python scripts/run_federated_robust.py \\
    --model {MODEL} {Q} {HF} {FAST} --no-save
print(f'✅ BRFU done in {(time.time()-t0)/60:.1f} min')

print(f'\\n🎉 All methods complete! Total elapsed: {(time.time()-NOTEBOOK_START)/60:.1f} min')
"""))

# ── SECTION: Results ──────────────────────────────────────────────────────────
cells.append(md("""\
---
## 📊 Section 6 — Consolidated Results & Visualisations
"""))

cells.append(code("""\
# ── 36. Collect and Display All Results  ⏱ ~15s ─────────────────────────────
import json, os, glob
import pandas as pd

results = []
for method_dir in sorted(glob.glob(f'{DRIVE_DIR}/*')):
    method_name = os.path.basename(method_dir)
    for jf in glob.glob(f'{method_dir}/**/*.json', recursive=True):
        try:
            with open(jf) as f:
                data = json.load(f)
            if 'forget_quality' in data:
                results.append({
                    'Method'           : method_name,
                    'Forget Quality ↑' : round(data.get('forget_quality',   0), 4),
                    'Forget Acc ↓'     : round(data.get('forget_accuracy',  0), 4),
                    'Retain Acc ↑'     : round(data.get('retain_accuracy',  0), 4),
                    'MIA AUROC →0.5'   : round(data.get('mia_auroc',       -1), 4),
                })
                break   # one result per method dir
        except Exception:
            pass

if results:
    df = pd.DataFrame(results).sort_values('Forget Quality ↑', ascending=False)
    print('\\n' + '='*72)
    print('  ARMOR — Full Experiment Results')
    print('='*72)
    print(df.to_string(index=False))
    out_csv = f'{DRIVE_DIR}/summary_results.csv'
    df.to_csv(out_csv, index=False)
    print(f'\\n✅ Results saved to {out_csv}')
else:
    print('No results found. Run experiment cells first, then re-run this cell.')
"""))

cells.append(code("""\
# ── 37. Plot Results  ⏱ ~10s ─────────────────────────────────────────────────
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
    out_img = f'{DRIVE_DIR}/results_comparison.png'
    plt.savefig(out_img, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.show()
    print(f'✅ Plot saved to {out_img}')
else:
    print('No results to plot.')
"""))

cells.append(code("""\
# ── 38. Privacy Audit Summary  ⏱ ~5s ────────────────────────────────────────
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
        print(f\"  {r['Method']:<28} AUROC={v:.4f}  {status}\")

print('\\n📜 DP Certificate (DP-NPO+SAM):')
print('   ε=8.0, δ=1e-5  →  formal (8.0, 1e-5)-DP guarantee on forget set')
print('='*65)
"""))

cells.append(code("""\
# ── 39. List Saved Checkpoints  ⏱ ~5s ───────────────────────────────────────
print('\\n📁 Checkpoints & results in Google Drive:')
for root, dirs, files in os.walk(DRIVE_DIR):
    depth = root.replace(DRIVE_DIR, '').count(os.sep)
    if depth > 2:
        continue
    total_size = sum(
        os.path.getsize(os.path.join(root, fn))
        for fn in files if os.path.isfile(os.path.join(root, fn))
    )
    indent = '  ' * depth
    bname  = os.path.basename(root) or 'ARMOR_outputs'
    sz     = f'({total_size/1e6:.1f} MB)' if total_size > 0 else ''
    print(f'{indent}📂 {bname}  {sz}')

total_time = (time.time() - NOTEBOOK_START) / 60
print(f'\\n⏱  Total notebook runtime: {total_time:.1f} minutes')
print('✅ Done! All outputs saved to Google Drive.')
"""))

# ── Build notebook JSON ───────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "collapsed_sections": ["🚀 Section 5 — Phase 1-3 Frontier Research Methods *(Optional)*"],
            "gpuType": "T4",
            "name": "ARMOR_Colab_Experiments.ipynb",
            "provenance": [],
            "toc_visible": True,
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

out = "ARMOR_Colab_Experiments.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✅ Written {len(cells)} cells → {out}")
