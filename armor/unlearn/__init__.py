"""armor/unlearn/__init__.py"""
# ── Baseline Modules ──────────────────────────────────────────────────────────
from armor.unlearn.gradient_ascent import GradientAscentUnlearner
from armor.unlearn.npo             import NPOUnlearner
from armor.unlearn.sam_wrapper     import SAMOptimizer
from armor.unlearn.rmu             import RMUUnlearner
from armor.unlearn.task_vector     import TaskVectorUnlearner, TaskVector
from armor.unlearn.who             import WHOUnlearner
from armor.unlearn.eul             import EULUnlearner
from armor.unlearn.multitask_npo   import MultiTaskNPOUnlearner
from armor.unlearn.dp_npo_sam      import DPNPOSAMUnlearner

# ── Research Expansion Modules (2026-06) ──────────────────────────────────────
from armor.unlearn.continual_unlearner import (  # Module 1 — Lifelong Unlearning
    ContinualUnlearner,
    ReplayBuffer,
    FIMSubspaceMask,
)
from armor.unlearn.moe_unlearner import (        # Module 2 — MoE Targeted Unlearning
    MoEUnlearner,
    MoERouterHook,
    ExpertUsageTallier,
    ExpertMagnitudePruner,
)
from armor.unlearn.rlace_rmu import (            # Module 3 — Advanced RMU / RLACE
    RLACERMUUnlearner,
    RLACEEraser,
    LinearMembershipProbe,
)
from armor.unlearn.lora_unlearner import (       # Module 6 — Modular LoRA Unlearning
    LoRAUnlearner,
    LoRALayer,
    LoRAInjector,
    NegativeLoRAApplicator,
)

# ── Phase 1 Research (2026-06) ────────────────────────────────────────────────
from armor.unlearn.cot_hme import (             # CoT-HME — CoT Erasure
    CoTHMEUnlearner,
    CoTSampleBank,
    CoTForgetSample,
    compute_cot_entropy_loss,
)
from armor.unlearn.temporal_decay import (      # TKDU — Temporal Unlearning
    TKDUUnlearner,
    KnowledgeTimestamp,
    TemporalValidityScorer,
    TemporalUnlearningScheduler,
    TemporalKnowledgeDataset,
    create_demo_knowledge_registry,
    load_knowledge_registry_from_json,
    save_knowledge_registry_to_json,
)

__all__ = [
    # Baseline
    "GradientAscentUnlearner",
    "NPOUnlearner",
    "SAMOptimizer",
    "RMUUnlearner",
    "TaskVectorUnlearner",
    "TaskVector",
    "WHOUnlearner",
    "EULUnlearner",
    "MultiTaskNPOUnlearner",
    "DPNPOSAMUnlearner",
    # Research expansions
    "ContinualUnlearner",
    "ReplayBuffer",
    "FIMSubspaceMask",
    "MoEUnlearner",
    "MoERouterHook",
    "ExpertUsageTallier",
    "ExpertMagnitudePruner",
    "RLACERMUUnlearner",
    "RLACEEraser",
    "LinearMembershipProbe",
    "LoRAUnlearner",
    "LoRALayer",
    "LoRAInjector",
    "NegativeLoRAApplicator",
    # Phase 1
    "CoTHMEUnlearner",
    "CoTSampleBank",
    "CoTForgetSample",
    "compute_cot_entropy_loss",
    "TKDUUnlearner",
    "KnowledgeTimestamp",
    "TemporalValidityScorer",
    "TemporalUnlearningScheduler",
    "TemporalKnowledgeDataset",
    "create_demo_knowledge_registry",
    "load_knowledge_registry_from_json",
    "save_knowledge_registry_to_json",
]
