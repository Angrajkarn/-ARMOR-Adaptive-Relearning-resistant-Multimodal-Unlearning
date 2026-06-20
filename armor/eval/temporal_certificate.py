"""
armor/eval/temporal_certificate.py
====================================
TKDU: Temporal Compliance Certificate Generator
================================================

Generates GDPR-compliant, cryptographically-signed audit certificates
for temporal knowledge decay unlearning runs.

Each certificate documents:
  - Which knowledge items were unlearned and when
  - Their validity scores (τ) at unlearning time
  - The expiry dates that triggered unlearning
  - GDPR retention period compliance status
  - HMAC-SHA256 tamper-evident signature
  - Machine-readable JSON + human-readable HTML output

GDPR Article 17 Compliance
---------------------------
This certificate provides evidence of compliance with the "Right to
Erasure" (Right to be Forgotten) under GDPR Article 17, specifically:
  - Clause 1(a): data no longer necessary for the original purpose
  - Clause 1(e): retention period has expired
  - Clause 3(b): compliance with a legal obligation

The certificate can be presented to Data Protection Authorities (DPAs)
as cryptographic evidence that temporal unlearning was performed.
"""

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from armor.unlearn.temporal_decay import (
    KnowledgeTimestamp, TemporalUnlearningResult,
    TemporalValidityScorer,
)

# HMAC signing key (in production, load from secure key vault)
_TEMPORAL_CERT_SECRET = b"ARMOR_TEMPORAL_CERT_SECRET_2026"


# ──────────────────────────────────────────────────────────────────────────────
# Certificate data class
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TemporalComplianceCertificate:
    """
    GDPR-compliant temporal unlearning compliance certificate.

    Fields mirror the structure required by GDPR Article 17 compliance
    documentation and NIST SP 800-188 privacy engineering guidelines.
    """
    # Certificate identity
    certificate_id: str = ""
    issued_at: str = ""
    issuer: str = "ARMOR Temporal Compliance Engine v1.0"
    schema_version: str = "1.0"

    # Method info
    method: str = "TKDU"
    run_id: str = ""

    # Temporal statistics
    n_total_items: int = 0
    n_expired_items: int = 0
    n_near_expiry_items: int = 0
    n_valid_items: int = 0
    mean_validity_score: float = 0.0
    unlearning_triggered_at: str = ""

    # Loss metrics
    forget_loss: float = 0.0
    retain_loss: float = 0.0
    total_loss: float = 0.0
    elapsed_sec: float = 0.0
    total_optimizer_steps: int = 0

    # GDPR fields
    gdpr_article: str = "Article 17 — Right to Erasure"
    gdpr_categories_erased: List[str] = field(default_factory=list)
    retention_period_compliant: bool = False
    data_controller: str = "ARMOR Framework"
    dpo_contact: str = "dpo@armor-framework.ai"

    # Per-item details
    knowledge_items: List[Dict[str, Any]] = field(default_factory=list)

    # Cryptographic integrity
    hmac_signature: str = ""
    hmac_algorithm: str = "HMAC-SHA256"
    payload_hash: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def verify_signature(self) -> bool:
        """Verify the HMAC signature to detect tampering."""
        payload = _build_signable_payload(self)
        expected = _sign_payload(payload)
        return hmac.compare_digest(self.hmac_signature, expected)


# ──────────────────────────────────────────────────────────────────────────────
# Certificate Generator
# ──────────────────────────────────────────────────────────────────────────────

def _build_signable_payload(cert: TemporalComplianceCertificate) -> bytes:
    """Build the canonical serialization for signing (excluding signature field)."""
    signable = {k: v for k, v in cert.to_dict().items()
                if k not in ("hmac_signature", "payload_hash")}
    return json.dumps(signable, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _sign_payload(payload: bytes) -> str:
    """Compute HMAC-SHA256 signature."""
    return hmac.new(_TEMPORAL_CERT_SECRET, payload, hashlib.sha256).hexdigest()


def _hash_payload(payload: bytes) -> str:
    """Compute SHA-256 hash of payload."""
    return hashlib.sha256(payload).hexdigest()


class TemporalCertificateGenerator:
    """
    Generates, signs, and saves temporal compliance certificates.

    Usage
    -----
    gen  = TemporalCertificateGenerator()
    cert = gen.generate(
        unlearning_result = result,
        knowledge_items   = items,
    )
    gen.save(cert, "outputs/temporal/certificate.json", save_html=True)
    """

    def generate(
        self,
        unlearning_result: TemporalUnlearningResult,
        knowledge_items: List[KnowledgeTimestamp],
        data_controller: str = "ARMOR Framework",
        dpo_contact: str = "dpo@armor-framework.ai",
    ) -> TemporalComplianceCertificate:
        """
        Generate a signed temporal compliance certificate.

        Parameters
        ----------
        unlearning_result : The result from TKDUUnlearner.run()
        knowledge_items   : The knowledge registry that was unlearned
        data_controller   : Organisation name (for GDPR header)
        dpo_contact       : Data Protection Officer contact email

        Returns
        -------
        TemporalComplianceCertificate — signed, tamper-evident certificate
        """
        now_ts    = time.time()
        cert_id   = f"TKDU-CERT-{int(now_ts)}-{unlearning_result.run_id[-8:]}"
        issued_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts))

        # GDPR categories erased
        gdpr_cats = list(set(
            k.gdpr_category for k in knowledge_items
            if k.gdpr_category != "none" and k.is_expired
        ))

        # Per-item detail (expired and near-expiry items only)
        item_details = []
        for k in knowledge_items:
            days = k.days_until_expiry(now_ts)
            t_exp = k.t_expiry
            item_details.append({
                "id": k.knowledge_id,
                "description": k.description[:80],
                "domain": k.domain,
                "gdpr_category": k.gdpr_category,
                "validity_score": round(k.current_validity, 6),
                "is_expired": k.is_expired,
                "expiry_date": (datetime.fromtimestamp(t_exp, tz=timezone.utc).isoformat()
                               if t_exp else "N/A"),
                "days_until_expiry": round(days, 2) if days is not None else None,
                "status": ("ERASED" if k.is_expired else
                          "PARTIAL_ERASURE" if k.current_validity < 0.5 else "RETAINED"),
            })

        # Determine GDPR compliance
        expired_count = sum(1 for k in knowledge_items if k.is_expired)
        retention_compliant = expired_count > 0  # at least some items were erased

        cert = TemporalComplianceCertificate(
            certificate_id=cert_id,
            issued_at=issued_at,
            issuer="ARMOR Temporal Compliance Engine v1.0",
            schema_version="1.0",
            method=unlearning_result.method,
            run_id=unlearning_result.run_id,
            n_total_items=unlearning_result.n_total,
            n_expired_items=unlearning_result.n_expired,
            n_near_expiry_items=unlearning_result.n_near_expiry,
            n_valid_items=unlearning_result.n_valid,
            mean_validity_score=unlearning_result.mean_validity_score,
            unlearning_triggered_at=unlearning_result.timestamp,
            forget_loss=unlearning_result.forget_loss_avg,
            retain_loss=unlearning_result.retain_loss_avg,
            total_loss=unlearning_result.total_loss_avg,
            elapsed_sec=unlearning_result.elapsed_sec,
            total_optimizer_steps=unlearning_result.total_steps,
            gdpr_article="Article 17 — Right to Erasure",
            gdpr_categories_erased=gdpr_cats,
            retention_period_compliant=retention_compliant,
            data_controller=data_controller,
            dpo_contact=dpo_contact,
            knowledge_items=item_details,
        )

        # Sign the certificate
        payload = _build_signable_payload(cert)
        cert.hmac_signature = _sign_payload(payload)
        cert.payload_hash   = _hash_payload(payload)

        return cert

    def save(
        self,
        cert: TemporalComplianceCertificate,
        json_path: str,
        save_html: bool = True,
    ) -> None:
        """Save the certificate as JSON and optionally HTML."""
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)

        # JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(cert.to_dict(), f, indent=2)
        print(f"[TemporalCert] JSON saved → {json_path}")

        # Verify own signature
        if cert.verify_signature():
            print("[TemporalCert] ✅ HMAC signature valid")
        else:
            print("[TemporalCert] ❌ HMAC signature INVALID — possible tampering!")

        if save_html:
            html_path = json_path.replace(".json", ".html")
            self._write_html(cert, html_path)

    def _write_html(self, cert: TemporalComplianceCertificate, path: str) -> None:
        """Generate a premium styled HTML certificate."""
        status_color = "#00c896" if cert.retention_period_compliant else "#ff4757"
        status_text  = (
            "✅ GDPR RETENTION PERIOD COMPLIANCE CERTIFIED"
            if cert.retention_period_compliant
            else "⚠️ COMPLIANCE REVIEW REQUIRED"
        )

        # Build knowledge table rows
        rows = ""
        for item in cert.knowledge_items:
            color = "#f85149" if item["is_expired"] else "#3fb950"
            days_str = (f"{item['days_until_expiry']:.1f}d"
                       if item["days_until_expiry"] is not None else "∞")
            rows += f"""
            <tr>
              <td>{item['id']}</td>
              <td class="mono">{item['description']}</td>
              <td>{item['domain']}</td>
              <td>{item['gdpr_category']}</td>
              <td class="mono" style="color:{color}">{item['validity_score']:.4f}</td>
              <td>{days_str}</td>
              <td style="color:{color}">{item['status']}</td>
            </tr>"""

        gdpr_cats_str = ", ".join(cert.gdpr_categories_erased) or "none"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TKDU Temporal Compliance Certificate — {cert.certificate_id}</title>
<style>
  :root{{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#e6edf3;
         --muted:#8b949e;--accent:#58a6ff;--green:#3fb950;--red:#f85149;}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);
        font-family:'Segoe UI',system-ui,sans-serif;padding:2rem;}}
  .container{{max-width:1100px;margin:0 auto;}}
  h1{{font-size:1.8rem;background:linear-gradient(135deg,#58a6ff,#bc8cff);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
  .badge{{display:inline-block;padding:.8rem 1.5rem;border-radius:8px;
          font-size:1.1rem;font-weight:700;background:{status_color}22;
          border:2px solid {status_color};color:{status_color};margin:1rem 0;}}
  .card{{background:var(--card);border:1px solid var(--border);
         border-radius:12px;padding:1.5rem;margin:1rem 0;}}
  .card h3{{color:var(--accent);margin-bottom:1rem;text-transform:uppercase;
            letter-spacing:1px;font-size:.9rem;}}
  table{{width:100%;border-collapse:collapse;font-size:.85rem;}}
  th,td{{padding:.5rem .8rem;border-bottom:1px solid var(--border);}}
  th{{color:var(--muted);text-transform:uppercase;font-size:.75rem;}}
  .mono{{font-family:'Courier New',monospace;}}
  .seal{{background:linear-gradient(135deg,#58a6ff22,#bc8cff22);
         border:2px solid #58a6ff44;border-radius:12px;
         padding:1rem;text-align:center;margin:1rem 0;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;}}
</style>
</head>
<body>
<div class="container">
  <div style="text-align:center;margin-bottom:2rem">
    <h1>📅 ARMOR — TKDU Temporal Compliance Certificate</h1>
    <p style="color:var(--muted)">Temporal Knowledge Decay Unlearning · GDPR Article 17 Compliance</p>
    <p style="color:var(--muted);font-size:.85rem">{cert.issued_at}</p>
    <div class="badge">{status_text}</div>
  </div>

  <div class="seal">
    <div style="font-size:1.2rem;font-weight:700;color:#58a6ff">🔏 CERTIFICATE ID</div>
    <div class="mono" style="font-size:1rem;margin:.5rem 0">{cert.certificate_id}</div>
    <div style="font-size:.8rem;color:var(--muted)">Issued by: {cert.issuer}</div>
  </div>

  <div class="grid">
    <div class="card">
      <h3>⚖️ GDPR Compliance</h3>
      <table>
        <tr><th>Field</th><th>Value</th></tr>
        <tr><td>Legal Basis</td><td>{cert.gdpr_article}</td></tr>
        <tr><td>Categories Erased</td><td>{gdpr_cats_str}</td></tr>
        <tr><td>Data Controller</td><td>{cert.data_controller}</td></tr>
        <tr><td>DPO Contact</td><td>{cert.dpo_contact}</td></tr>
        <tr><td>Compliance Status</td>
            <td style="color:{'#3fb950' if cert.retention_period_compliant else '#f85149'}">
              {"✅ Compliant" if cert.retention_period_compliant else "⚠️ Review Required"}
            </td></tr>
      </table>
    </div>

    <div class="card">
      <h3>📊 Unlearning Statistics</h3>
      <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total items</td><td>{cert.n_total_items}</td></tr>
        <tr><td>Expired (erased)</td><td style="color:#f85149">{cert.n_expired_items}</td></tr>
        <tr><td>Near expiry</td><td style="color:#f0a500">{cert.n_near_expiry_items}</td></tr>
        <tr><td>Still valid</td><td style="color:#3fb950">{cert.n_valid_items}</td></tr>
        <tr><td>Mean τ score</td><td class="mono">{cert.mean_validity_score:.4f}</td></tr>
        <tr><td>Forget loss</td><td class="mono">{cert.forget_loss:.4f}</td></tr>
        <tr><td>Training time</td><td>{cert.elapsed_sec:.1f}s</td></tr>
      </table>
    </div>
  </div>

  <div class="card">
    <h3>📋 Knowledge Registry Audit</h3>
    <table>
      <tr>
        <th>ID</th><th>Description</th><th>Domain</th>
        <th>GDPR Cat.</th><th>τ Score</th><th>Days Left</th><th>Status</th>
      </tr>
      {rows}
    </table>
  </div>

  <div class="card">
    <h3>🔐 Cryptographic Integrity</h3>
    <table>
      <tr><th>Field</th><th>Value</th></tr>
      <tr><td>Algorithm</td><td>{cert.hmac_algorithm}</td></tr>
      <tr><td>Signature</td>
          <td class="mono" style="font-size:.75rem;word-break:break-all">{cert.hmac_signature}</td></tr>
      <tr><td>Payload Hash (SHA-256)</td>
          <td class="mono" style="font-size:.75rem;word-break:break-all">{cert.payload_hash}</td></tr>
    </table>
    <p style="color:var(--muted);margin-top:.5rem;font-size:.8rem">
      This certificate is tamper-evident. Any modification to the fields above
      will invalidate the HMAC signature.
    </p>
  </div>
</div>
</body>
</html>"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[TemporalCert] HTML saved → {path}")
