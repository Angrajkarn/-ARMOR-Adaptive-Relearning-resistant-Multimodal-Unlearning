"""armor/eval/__init__.py"""
# ── Baseline ──────────────────────────────────────────────────────────────────
from armor.eval.metrics import UnlearningEvaluator
from armor.eval.mia     import MembershipInferenceAuditor

# ── Research Expansions (2026-06) ─────────────────────────────────────────────
from armor.eval.zk_verify import (          # Module 4 — ZK Verification
    ZKVerifier,
    UnlearningCommitment,
    InfluenceEstimator,
)
from armor.eval.multimodal_mia import (     # Module 5 — Multimodal MIA
    MultimodalMIAEvaluator,
    ContrastiveUnlearningLoss,
    VisualMembershipTest,
)
from armor.eval.certificate import (         # Enterprise Expansion
    AuditCertificateGenerator,
)

__all__ = [
    "UnlearningEvaluator",
    "MembershipInferenceAuditor",
    "ZKVerifier",
    "UnlearningCommitment",
    "InfluenceEstimator",
    "MultimodalMIAEvaluator",
    "ContrastiveUnlearningLoss",
    "VisualMembershipTest",
    "AuditCertificateGenerator",
]
