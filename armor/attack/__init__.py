"""armor/attack/__init__.py"""
from armor.attack.relearning       import RelearningAttack
from armor.attack.lora_attack      import LoRALinear, LoRARelearningAttack
from armor.attack.prompt_attack    import PromptInjectionAttack, PromptAttackResult
from armor.attack.federated_attack import FederatedRelearningAttack, FederatedAttackResult
from armor.attack.reconstruction   import TextReconstructionAttack, ReconstructionAttackResult

# ── Phase 1 Research (2026-06) ────────────────────────────────────────────────
from armor.attack.cot_leakage_probe import (  # CoT-HME — Leakage Detection
    CoTLeakageProbe,
    CoTLeakageScorer,
    CoTLeakageResult,
    CoTLeakageReport,
    CoTStep,
    build_cot_prompt,
    segment_cot_trace,
)

__all__ = [
    "RelearningAttack",
    "LoRALinear",
    "LoRARelearningAttack",
    "PromptInjectionAttack",
    "PromptAttackResult",
    "FederatedRelearningAttack",
    "FederatedAttackResult",
    "TextReconstructionAttack",
    "ReconstructionAttackResult",
    # Phase 1
    "CoTLeakageProbe",
    "CoTLeakageScorer",
    "CoTLeakageResult",
    "CoTLeakageReport",
    "CoTStep",
    "build_cot_prompt",
    "segment_cot_trace",
]
