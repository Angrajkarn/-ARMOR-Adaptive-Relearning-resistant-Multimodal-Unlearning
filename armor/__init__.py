"""
ARMOR — Multimodal LLM Unlearning System
=========================================
Package root. Import submodules explicitly to avoid circular imports.

Architecture overview (designed for LLaVA extension):
    armor/
    ├── config.py       — Central hyperparameter dataclass
    ├── data.py         — TOFU dataset + rephrase augmentation
    ├── model.py        — Model/tokenizer loader (text → multimodal ready)
    ├── unlearn/        — Unlearning algorithms (GA, NPO, SAM)
    ├── eval/           — Evaluation metrics + MIA audit
    └── attack/         — Relearning attack simulation
"""

__version__ = "0.1.0"
__author__  = "ARMOR Research"

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

