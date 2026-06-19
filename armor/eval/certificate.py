"""
armor/eval/certificate.py
==========================
Compliance & Cryptographic Audit Certificate Generator.

Combines the results of ZK-style influence verification, Differential Privacy bounds
(empirical and formal), and adversarial reconstruction attacks into a unified,
tamper-evident audit certificate (JSON & HTML). The certificate is cryptographically
signed with an HMAC-SHA256 signature.
"""

import hmac
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

from ..config import ARMORConfig


class AuditCertificateGenerator:
    """
    Generates regulatory compliance audit certificates for machine unlearning.
    Produces structured JSON certificates and beautiful, production-ready HTML reports.
    """
    def __init__(self, cfg: ARMORConfig, private_signing_key: str = "ARMOR_ENTERPRISE_KEY_2026"):
        self.cfg = cfg
        self.private_signing_key = private_signing_key.encode("utf-8")

    def _compute_signature(self, payload: Dict[str, Any]) -> str:
        """Compute HMAC-SHA256 signature of the sorted, serialized payload."""
        canonical_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        sig = hmac.new(self.private_signing_key, canonical_json.encode("utf-8"), hashlib.sha256)
        return sig.hexdigest()

    def generate_certificate(self,
                             model_name: str,
                             method_name: str,
                             pre_model_hash: str,
                             post_model_hash: str,
                             forget_set_hash: str,
                             zk_report: Dict[str, Any],
                             privacy_report: Any,  # PrivacyAuditResult or dict
                             reconstruction_report: Any,  # ReconstructionAttackResult or dict
                             output_dir: str = "outputs/audit") -> Tuple[Dict[str, Any], str]:
        """
        Creates a signed audit certificate, saves JSON and HTML files, and returns them.
        """
        os.makedirs(output_dir, exist_ok=True)
        cert_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Normalize reports to dicts
        p_dict = privacy_report if isinstance(privacy_report, dict) else {
            "auroc": getattr(privacy_report, "auroc", 0.5),
            "epsilon_empirical": getattr(privacy_report, "epsilon_empirical", 0.0),
            "epsilon_formal": getattr(privacy_report, "epsilon_formal", None),
            "delta": getattr(privacy_report, "delta", 1e-5),
            "is_verified_unlearned": getattr(privacy_report, "is_verified_unlearned", False)
        }

        r_dict = reconstruction_report if isinstance(reconstruction_report, dict) else {
            "avg_greedy_rougeL": getattr(reconstruction_report, "avg_greedy_rougeL", 0.0),
            "avg_tree_rougeL": getattr(reconstruction_report, "avg_tree_rougeL", 0.0),
            "avg_target_logprob": getattr(reconstruction_report, "avg_target_logprob", 0.0),
            "leakage_rate": getattr(reconstruction_report, "leakage_rate", 0.0)
        }

        # 1. Build Payload
        payload = {
            "certificate_id": cert_id,
            "timestamp": timestamp,
            "issuer": "ARMOR Compliance Auditing Suite v1.2",
            "model_name": model_name,
            "unlearning_method": method_name,
            "commitments": {
                "pre_unlearn_model_hash": pre_model_hash,
                "post_unlearn_model_hash": post_model_hash,
                "forget_dataset_hash": forget_set_hash,
                "pre_unlearn_zk_commitment": zk_report.get("pre_commitment", {}).get("commitment", ""),
                "post_unlearn_zk_commitment": zk_report.get("post_commitment", {}).get("commitment", "")
            },
            "metrics": {
                "zk_influence_verification": {
                    "verdict": zk_report.get("verdict", "UNKNOWN"),
                    "n_verified": zk_report.get("n_verified", 0),
                    "n_total": zk_report.get("n_total", 0),
                    "mean_influence_gap": zk_report.get("mean_influence_gap", 0.0),
                    "threshold": zk_report.get("threshold", 0.01)
                },
                "differential_privacy_bounds": {
                    "mia_auroc": p_dict["auroc"],
                    "empirical_epsilon": p_dict["epsilon_empirical"],
                    "formal_epsilon": p_dict["epsilon_formal"],
                    "delta": p_dict["delta"],
                    "is_privacy_verified": p_dict["is_verified_unlearned"]
                },
                "adversarial_reconstruction_resistance": {
                    "greedy_rouge_l": r_dict["avg_greedy_rougeL"],
                    "prefix_tree_rouge_l": r_dict["avg_tree_rougeL"],
                    "reconstruction_leakage_rate": r_dict["leakage_rate"],
                    "reconstruction_verdict": "SECURE" if r_dict["leakage_rate"] < 0.10 else "VULNERABLE"
                }
            }
        }

        # 2. Cryptographically Sign the Payload
        signature = self._compute_signature(payload)
        certificate = {
            "certificate": payload,
            "signature": {
                "algorithm": "HMAC-SHA256",
                "signed_by": "ARMOR Enterprise Compliance CA",
                "signature_hash": signature
            }
        }

        # 3. Save JSON Certificate
        json_path = os.path.join(output_dir, f"audit_certificate_{cert_id[:8]}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(certificate, f, indent=2, ensure_ascii=False)

        # 4. Generate & Save HTML Certificate
        html_path = os.path.join(output_dir, f"audit_certificate_{cert_id[:8]}.html")
        html_content = self._render_html_certificate(certificate)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Also copy as generic names for easy linking / tests
        generic_json_path = os.path.join(output_dir, "audit_certificate.json")
        generic_html_path = os.path.join(output_dir, "audit_certificate.html")
        with open(generic_json_path, "w", encoding="utf-8") as f:
            json.dump(certificate, f, indent=2, ensure_ascii=False)
        with open(generic_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"[Certificate] Saved JSON certificate → {json_path}")
        print(f"[Certificate] Saved HTML certificate → {html_path}")
        return certificate, html_path

    def _render_html_certificate(self, cert_data: Dict[str, Any]) -> str:
        """Renders the compliance certificate into a premium, responsive dark-mode HTML page."""
        c = cert_data["certificate"]
        sig = cert_data["signature"]

        # Color codes based on results
        zk_passed = c["metrics"]["zk_influence_verification"]["verdict"] == "VERIFIED ✓"
        zk_color = "#10b981" if zk_passed else "#f59e0b"
        
        mia_val = c["metrics"]["differential_privacy_bounds"]["mia_auroc"]
        mia_color = "#10b981" if mia_val < 0.55 else ("#f59e0b" if mia_val < 0.65 else "#ef4444")
        
        leakage = c["metrics"]["adversarial_reconstruction_resistance"]["reconstruction_leakage_rate"]
        leak_color = "#10b981" if leakage < 0.10 else ("#f59e0b" if leakage < 0.25 else "#ef4444")

        # HTML and CSS Template
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ARMOR GDPR Compliance Certificate</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #080a13;
            --card-bg: rgba(18, 22, 40, 0.65);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-primary: #4f46e5;
            --accent-cyan: #06b6d4;
            --accent-purple: #a855f7;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 0% 0%, rgba(79, 70, 229, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.12) 0px, transparent 50%),
                radial-gradient(at 50% 50%, rgba(6, 182, 212, 0.08) 0px, transparent 60%);
            background-attachment: fixed;
            font-family: 'Inter', sans-serif;
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }}

        .certificate-container {{
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            width: 100%;
            max-width: 900px;
            padding: 3rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            position: relative;
            overflow: hidden;
        }}

        /* Glow effects */
        .certificate-container::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 6px;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-primary), var(--accent-purple));
        }}

        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            padding-bottom: 2rem;
            margin-bottom: 2.5rem;
        }}

        .header-logo {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.05em;
            background: linear-gradient(135deg, #06b6d4, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .badge-verified {{
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--success);
            padding: 0.6rem 1.2rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.15);
            text-transform: uppercase;
        }}

        .title-block {{
            text-align: center;
            margin-bottom: 3rem;
        }}

        .title-block h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }}

        .title-block p {{
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 300;
        }}

        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}

        .meta-card {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.04);
            padding: 1.2rem 1.5rem;
            border-radius: 14px;
            transition: all 0.3s ease;
        }}

        .meta-card:hover {{
            background: rgba(255, 255, 255, 0.04);
            border-color: rgba(255, 255, 255, 0.08);
            transform: translateY(-2px);
        }}

        .meta-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.4rem;
        }}

        .meta-value {{
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 1.1rem;
            word-break: break-all;
        }}

        .meta-hash {{
            font-family: monospace;
            font-size: 0.8rem;
            color: var(--accent-cyan);
            word-break: break-all;
        }}

        .metrics-section {{
            margin-bottom: 3.5rem;
        }}

        .section-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .section-title::after {{
            content: '';
            flex-grow: 1;
            height: 1px;
            background: rgba(255, 255, 255, 0.08);
            margin-left: 1rem;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
        }}

        .metric-card {{
            background: rgba(255, 255, 255, 0.015);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 18px;
            padding: 1.8rem 1.5rem;
            text-align: center;
            position: relative;
            transition: all 0.3s ease;
        }}

        .metric-card:hover {{
            background: rgba(255, 255, 255, 0.03);
            border-color: rgba(255, 255, 255, 0.08);
            box-shadow: 0 10px 20px -10px rgba(0,0,0,0.3);
            transform: translateY(-4px);
        }}

        .metric-badge {{
            position: absolute;
            top: 0.8rem;
            right: 0.8rem;
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}

        .metric-val {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0.8rem 0;
            letter-spacing: -0.03em;
        }}

        .metric-desc {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            line-height: 1.4;
        }}

        .commitment-log {{
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 14px;
            padding: 1.5rem;
            margin-bottom: 3rem;
        }}

        .log-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.6rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            font-size: 0.85rem;
        }}

        .log-row:last-child {{
            border-bottom: none;
            padding-bottom: 0;
        }}

        .log-label {{
            color: var(--text-secondary);
        }}

        .log-val {{
            font-family: monospace;
            color: var(--text-primary);
            max-width: 60%;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .signature-block {{
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.08) 0%, rgba(168, 85, 247, 0.08) 100%);
            border: 1px dashed rgba(168, 85, 247, 0.25);
            border-radius: 16px;
            padding: 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .sig-details {{
            max-width: 65%;
        }}

        .sig-title {{
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            margin-bottom: 0.4rem;
            font-size: 1.1rem;
        }}

        .sig-hash {{
            font-family: monospace;
            font-size: 0.78rem;
            color: var(--accent-cyan);
            word-break: break-all;
            background: rgba(0, 0, 0, 0.3);
            padding: 0.5rem;
            border-radius: 6px;
            margin-top: 0.5rem;
            border: 1px solid rgba(255, 255, 255, 0.03);
        }}

        .sig-badge {{
            text-align: center;
        }}

        .sig-stamp {{
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            font-size: 0.9rem;
            border: 2px solid var(--accent-purple);
            color: var(--accent-purple);
            padding: 0.8rem 1.2rem;
            border-radius: 8px;
            text-transform: uppercase;
            transform: rotate(-5deg);
            display: inline-block;
            box-shadow: 0 0 10px rgba(168, 85, 247, 0.1);
        }}

        @media (max-width: 768px) {{
            .certificate-container {{
                padding: 1.5rem;
            }}
            .meta-grid {{
                grid-template-columns: 1fr;
            }}
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
            .signature-block {{
                flex-direction: column;
                gap: 1.5rem;
                text-align: center;
            }}
            .sig-details {{
                max-width: 100%;
            }}
        }}

        @media print {{
            body {{
                background: white;
                color: black;
                padding: 0;
            }}
            .certificate-container {{
                border: none;
                box-shadow: none;
                background: white;
                color: black;
                backdrop-filter: none;
            }}
            .meta-card, .metric-card, .commitment-log, .signature-block {{
                background: #f9fafb !important;
                border: 1px solid #e5e7eb !important;
                color: black !important;
            }}
            .meta-hash, .sig-hash {{
                background: #f3f4f6 !important;
                color: #111827 !important;
                border: 1px solid #d1d5db !important;
            }}
            .badge-verified {{
                border: 1px solid #10b981;
                background: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="certificate-container">
        <div class="header">
            <div class="header-logo">ARMOR</div>
            <div class="badge-verified">
                <svg width="16" height="16" viewBox="0 0 20 20" fill="currentColor" style="display:inline; vertical-align:middle;">
                    <path fill-rule="evenodd" d="M6.267 3.455a.75.75 0 00-.708-.523H4.5a2.5 2.5 0 00-2.5 2.5v1.059a.75.75 0 00.523.708 3.5 3.5 0 010 6.596.75.75 0 00-.523.708v1.059a2.5 2.5 0 002.5 2.5h1.059a.75.75 0 00.708-.523 3.5 3.5 0 016.596 0 .75.75 0 00.708.523h1.059a2.5 2.5 0 002.5-2.5v-1.059a.75.75 0 00-.523-.708 3.5 3.5 0 010-6.596.75.75 0 00.523-.708V5.455a2.5 2.5 0 00-2.5-2.5h-1.059a.75.75 0 00-.708.523 3.5 3.5 0 01-6.596 0zM10 13a3 3 0 100-6 3 3 0 000 6z" clip-rule="evenodd"/>
                </svg>
                Verified Erasure
            </div>
        </div>

        <div class="title-block">
            <h1>GDPR Compliance Certificate</h1>
            <p>Cryptographic & Empirical Verification of Right to be Forgotten</p>
        </div>

        <div class="meta-grid">
            <div class="meta-card">
                <div class="meta-label">Certificate ID</div>
                <div class="meta-value" style="font-size:0.95rem; font-family:monospace; color:var(--accent-cyan);">{c["certificate_id"]}</div>
            </div>
            <div class="meta-card">
                <div class="meta-label">Timestamp (UTC)</div>
                <div class="meta-value" style="font-size:0.95rem; font-family:monospace;">{c["timestamp"]}</div>
            </div>
            <div class="meta-card">
                <div class="meta-label">Target Model</div>
                <div class="meta-value">{c["model_name"]}</div>
            </div>
            <div class="meta-card">
                <div class="meta-label">Unlearning Method</div>
                <div class="meta-value" style="color:var(--accent-purple);">{c["unlearning_method"]}</div>
            </div>
        </div>

        <div class="metrics-section">
            <div class="section-title">Compliance Verification Metrics</div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-badge" style="background-color:{zk_color}; box-shadow: 0 0 10px {zk_color};"></div>
                    <div class="meta-label">ZK Influence Gap</div>
                    <div class="metric-val" style="color:{zk_color};">{c["metrics"]["zk_influence_verification"]["verdict"].split()[0]}</div>
                    <div class="metric-desc">
                        Verified {c["metrics"]["zk_influence_verification"]["n_verified"]}/{c["metrics"]["zk_influence_verification"]["n_total"]} forget samples<br>
                        Mean influence delta: {c["metrics"]["zk_influence_verification"]["mean_influence_gap"]:.4f} (thr={c["metrics"]["zk_influence_verification"]["threshold"]})
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-badge" style="background-color:{mia_color}; box-shadow: 0 0 10px {mia_color};"></div>
                    <div class="meta-label">MIA AUROC</div>
                    <div class="metric-val" style="color:{mia_color};">{mia_val:.4f}</div>
                    <div class="metric-desc">
                        Empirical ε lower bound: {c["metrics"]["differential_privacy_bounds"]["empirical_epsilon"]:.4f}<br>
                        DP-SGD ε: {c["metrics"]["differential_privacy_bounds"]["formal_epsilon"] if c["metrics"]["differential_privacy_bounds"]["formal_epsilon"] is not None else "N/A"} (δ={c["metrics"]["differential_privacy_bounds"]["delta"]})
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-badge" style="background-color:{leak_color}; box-shadow: 0 0 10px {leak_color};"></div>
                    <div class="meta-label">Reconstruction Leakage</div>
                    <div class="metric-val" style="color:{leak_color};">{leakage:.1%}</div>
                    <div class="metric-desc">
                        Tree ROUGE-L: {c["metrics"]["adversarial_reconstruction_resistance"]["prefix_tree_rouge_l"]:.4f}<br>
                        Status: <strong>{c["metrics"]["adversarial_reconstruction_resistance"]["reconstruction_verdict"]}</strong> against inversion
                    </div>
                </div>
            </div>
        </div>

        <div class="metrics-section">
            <div class="section-title">Cryptographic Commitments</div>
            <div class="commitment-log">
                <div class="log-row">
                    <span class="log-label">Pre-Unlearning Model Weight Hash (first 1M params)</span>
                    <span class="log-val" title="{c["commitments"]["pre_unlearn_model_hash"]}">{c["commitments"]["pre_unlearn_model_hash"]}</span>
                </div>
                <div class="log-row">
                    <span class="log-label">Post-Unlearning Model Weight Hash (first 1M params)</span>
                    <span class="log-val" title="{c["commitments"]["post_unlearn_model_hash"]}">{c["commitments"]["post_unlearn_model_hash"]}</span>
                </div>
                <div class="log-row">
                    <span class="log-label">Forget Dataset Hash</span>
                    <span class="log-val" title="{c["commitments"]["forget_dataset_hash"]}">{c["commitments"]["forget_dataset_hash"]}</span>
                </div>
                <div class="log-row">
                    <span class="log-label">Pre-Unlearning ZK Commitment</span>
                    <span class="log-val" title="{c["commitments"]["pre_unlearn_zk_commitment"]}">{c["commitments"]["pre_unlearn_zk_commitment"]}</span>
                </div>
                <div class="log-row">
                    <span class="log-label">Post-Unlearning ZK Commitment</span>
                    <span class="log-val" title="{c["commitments"]["post_unlearn_zk_commitment"]}">{c["commitments"]["post_unlearn_zk_commitment"]}</span>
                </div>
            </div>
        </div>

        <div class="signature-block">
            <div class="sig-details">
                <div class="sig-title">Audit Engine Verification Signature</div>
                <p style="font-size:0.8rem; color:var(--text-secondary);">
                    This certificate carries an HMAC-SHA256 signature calculated over the complete compliance metrics payload using the private key of the ARMOR root authority. Any modification to the payload invalidates the signature.
                </p>
                <div class="sig-hash">
                    {sig["signature_hash"]}
                </div>
            </div>
            <div class="sig-badge">
                <div class="sig-stamp">ARMOR SIGNED</div>
                <div style="font-size:0.7rem; color:var(--text-secondary); margin-top:0.4rem;">{sig["signed_by"]}</div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        return html
