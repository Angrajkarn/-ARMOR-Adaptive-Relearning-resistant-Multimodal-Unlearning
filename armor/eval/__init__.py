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

# ── Phase 1 Research (2026-06) ────────────────────────────────────────────────
from armor.eval.conformal_verify import (   # CU-AR — Conformal Unlearning
    ConformalUnlearningVerifier,
    NonconformityScorer,
    ConformalCalibrator,
    ConformalUnlearningReport,
    ConformalCalibrationResult,
    NonconformityResult,
    conformal_unlearning_test,
)
from armor.eval.temporal_certificate import (  # TKDU — Temporal Certificates
    TemporalComplianceCertificate,
    TemporalCertificateGenerator,
)

__all__ = [
    # Baseline
    "UnlearningEvaluator",
    "MembershipInferenceAuditor",
    # Research Expansions
    "ZKVerifier",
    "UnlearningCommitment",
    "InfluenceEstimator",
    "MultimodalMIAEvaluator",
    "ContrastiveUnlearningLoss",
    "VisualMembershipTest",
    "AuditCertificateGenerator",
    # Phase 1
    "ConformalUnlearningVerifier",
    "NonconformityScorer",
    "ConformalCalibrator",
    "ConformalUnlearningReport",
    "ConformalCalibrationResult",
    "NonconformityResult",
    "conformal_unlearning_test",
    "TemporalComplianceCertificate",
    "TemporalCertificateGenerator",
]
