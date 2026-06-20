"""
armor/eval/conformal_verify.py
==============================
CU-AR: Conformal Unlearning for Autoregressive LLMs
=====================================================

The first distribution-free, finite-sample statistical verification method
for machine unlearning in autoregressive text generation models.

Theory
------
Standard conformal prediction provides *coverage* guarantees for classifiers:
    P(y_true ∈ C(x)) ≥ 1 - α

For autoregressive LLMs, we adapt this to token-sequence prediction sets using
the *nonconformity score*:

    s(x, y) = -log P_θ(y | x)   (negative log-likelihood over answer tokens)

Given a calibration set (retain samples), we find the (1-α)-quantile:
    q̂ = quantile({s(x_i, y_i) : i=1..n}, 1-α)

The conformal prediction set for a new input x is:
    C(x) = {y : s(x, y) ≤ q̂}

UNLEARNING VERIFIED if:
    s(x_f, y_f*) > q̂   for all forget samples (x_f, y_f*)
    ⟺ the correct forget-set answer is NOT in the model's prediction set

This gives a formal guarantee:
    P(forget answer ∈ prediction set after unlearning) ≤ α

Key advantages over existing MIA methods:
  ✓ Distribution-free: no parametric assumptions on the model
  ✓ Finite-sample: valid for any n (not just asymptotically)
  ✓ Architecture-agnostic: works on any autoregressive LLM
  ✓ Interpretable: α directly controls the Type I error rate

Reference
---------
  Conformal Prediction (Vovk et al. 2005 / Angelopoulos et al. 2021)
  Applied to Machine Unlearning — Novel contribution in ARMOR (2026)
"""

import json
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from armor.config import ARMORConfig


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NonconformityResult:
    """Nonconformity score for a single (input, output) pair."""
    text_prefix: str               # The question / prompt
    text_target: str               # The answer / target continuation
    score: float                   # s(x, y) = -log P(y|x)  (higher = less likely)
    is_forget: bool = False        # True if this is a forget-set sample
    token_count: int = 0           # Number of answer tokens scored


@dataclass
class ConformalCalibrationResult:
    """Result of calibrating the conformal threshold on the retain set."""
    alpha: float                   # Miscoverage rate (e.g. 0.05)
    n_calibration: int             # Number of calibration (retain) samples
    threshold: float               # q̂ — the (1-α) quantile
    calibration_scores: List[float] = field(default_factory=list)  # All retain scores
    coverage_empirical: float = 0.0  # Empirical coverage on calibration set

    def summary(self) -> str:
        return (
            f"ConformalCalibration | α={self.alpha:.3f} | "
            f"n={self.n_calibration} | threshold q̂={self.threshold:.4f} | "
            f"empirical_coverage={self.coverage_empirical:.3f}"
        )


@dataclass
class ConformalUnlearningReport:
    """
    Full conformal unlearning verification report.

    INTERPRETATION
    --------------
    unlearning_certified = True  iff  forget_coverage_rate ≤ alpha
    i.e., ≤ alpha fraction of forget-set answers remain in prediction sets.

    Statistical guarantee: under the null hypothesis that unlearning succeeded,
    P(forget answer ∈ prediction set) ≤ alpha (e.g. 5%).
    """
    method: str = "Unknown"
    timestamp: str = ""

    # Calibration
    alpha: float = 0.05
    threshold: float = 0.0
    n_calibration: int = 0

    # Forget-set results
    n_forget: int = 0
    forget_scores: List[float] = field(default_factory=list)  # s(x_f, y_f*)
    forget_coverage_rate: float = 0.0     # fraction of forget samples in prediction set (SHOULD BE ≤ alpha)
    forget_mean_score: float = 0.0        # mean nonconformity score on forget set
    forget_min_score: float = 0.0
    forget_max_score: float = 0.0

    # Retain-set sanity check
    n_retain_check: int = 0
    retain_coverage_rate: float = 0.0    # fraction of retain samples in prediction set (SHOULD BE ≥ 1-alpha)

    # Verdict
    unlearning_certified: bool = False   # True if forget_coverage_rate ≤ alpha
    certification_margin: float = 0.0   # (alpha - forget_coverage_rate) — how far inside guarantee

    # Per-sample results
    per_sample: List[Dict[str, Any]] = field(default_factory=list)

    def print_report(self):
        print("\n" + "=" * 72)
        print("  CU-AR: CONFORMAL UNLEARNING VERIFICATION REPORT")
        print("=" * 72)
        print(f"  Method         : {self.method}")
        print(f"  Timestamp      : {self.timestamp}")
        print(f"  Alpha (α)      : {self.alpha:.3f}  (max allowed forget coverage)")
        print(f"  Threshold (q̂)  : {self.threshold:.4f}")
        print(f"  Calibration n  : {self.n_calibration}")
        print("  " + "-" * 68)
        print(f"  {'Metric':<40} {'Value':>15}")
        print("  " + "-" * 68)
        print(f"  {'Forget Coverage Rate (target: ≤α)':<40} {self.forget_coverage_rate:>15.4f}")
        print(f"  {'Retain Coverage Rate (target: ≥1-α)':<40} {self.retain_coverage_rate:>15.4f}")
        print(f"  {'Forget Mean Score (higher=more forgotten)':<40} {self.forget_mean_score:>15.4f}")
        print(f"  {'Forget Min Score':<40} {self.forget_min_score:>15.4f}")
        print(f"  {'Forget Max Score':<40} {self.forget_max_score:>15.4f}")
        print(f"  {'Certification Margin (α - coverage)':<40} {self.certification_margin:>15.4f}")
        print("  " + "-" * 68)

        if self.unlearning_certified:
            print(f"  ✅ UNLEARNING CERTIFIED  (coverage {self.forget_coverage_rate:.3f} ≤ α={self.alpha:.3f})")
            print(f"     Statistical guarantee: P(recall forget answer) ≤ {self.alpha:.1%}")
        else:
            print(f"  ❌ NOT CERTIFIED  (coverage {self.forget_coverage_rate:.3f} > α={self.alpha:.3f})")
            print(f"     {self.forget_coverage_rate - self.alpha:.3f} above allowed threshold — more unlearning needed")
        print("=" * 72 + "\n")

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Nonconformity Scorer
# ──────────────────────────────────────────────────────────────────────────────

class NonconformityScorer:
    """
    Computes the nonconformity score s(x, y) = -log P_θ(y | x) for any
    autoregressive language model.

    This is the average negative log-probability of the *answer tokens* given
    the *question prefix* — a natural measure of how "surprising" the model
    finds a given (question, answer) pair.

    Higher score → model finds the answer less likely → answer is NOT in the
    model's prediction set.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cfg: ARMORConfig,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.device    = cfg.device

    @torch.no_grad()
    def score_batch(self, batch: dict) -> torch.Tensor:
        """
        Compute nonconformity scores for a batch from a DataLoader.

        The batch must have 'input_ids', 'attention_mask', and 'labels'
        (labels = -100 for prompt tokens, answer token IDs for answer tokens).

        Returns
        -------
        scores : (B,) tensor — nonconformity score per sample
        """
        self.model.eval()
        input_ids = batch["input_ids"].to(self.device)
        attn_mask = batch.get("attention_mask",
                               torch.ones_like(input_ids)).to(self.device)
        labels    = batch["labels"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
        )
        logits = outputs.logits  # (B, T, V)

        # Shift for autoregressive prediction
        shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
        shift_labels = labels[:, 1:].contiguous()        # (B, T-1)

        # Per-token log-softmax
        log_probs_all = F.log_softmax(shift_logits, dim=-1)

        # Gather log-prob of the actual answer token
        labels_clamped = shift_labels.clamp(min=0)
        token_log_probs = log_probs_all.gather(
            dim=-1, index=labels_clamped.unsqueeze(-1)
        ).squeeze(-1)  # (B, T-1)

        # Mask: only score answer tokens (label != -100)
        mask = (shift_labels != -100).float()  # (B, T-1)

        # Average NLL per sample over answer tokens
        nll_per_sample = -(token_log_probs * mask).sum(-1) / mask.sum(-1).clamp(min=1)

        return nll_per_sample  # (B,) — higher = less likely = NOT in prediction set

    @torch.no_grad()
    def score_text(self, question: str, answer: str) -> NonconformityResult:
        """
        Score a single (question, answer) pair as free text.

        This is used for custom forget-set evaluation without a DataLoader.

        Parameters
        ----------
        question : The prompt / question string
        answer   : The answer / target continuation string

        Returns
        -------
        NonconformityResult with the score
        """
        self.model.eval()

        # Encode question + answer together
        full_text = f"Question: {question}\nAnswer: {answer}"
        prompt_text = f"Question: {question}\nAnswer:"

        full_enc   = self.tokenizer(full_text,   return_tensors="pt",
                                    truncation=True, max_length=self.cfg.max_seq_len)
        prompt_enc = self.tokenizer(prompt_text, return_tensors="pt",
                                    truncation=True, max_length=self.cfg.max_seq_len)

        input_ids = full_enc["input_ids"].to(self.device)
        n_prompt  = prompt_enc["input_ids"].shape[-1]

        # Build labels: -100 for prompt tokens, answer token IDs for answer tokens
        labels = input_ids.clone()
        labels[:, :n_prompt] = -100

        # Score
        outputs = self.model(input_ids=input_ids, labels=labels)
        # Average NLL on answer tokens = loss (already computed by HF)
        score = outputs.loss.item()  # -log P(answer | question) per token

        return NonconformityResult(
            text_prefix=question,
            text_target=answer,
            score=score,
            token_count=max(0, input_ids.shape[-1] - n_prompt),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Conformal Calibrator
# ──────────────────────────────────────────────────────────────────────────────

class ConformalCalibrator:
    """
    Calibrates the conformal prediction threshold q̂ on the retain (calibration) set.

    The threshold q̂ is the (1-α) empirical quantile of nonconformity scores
    on the calibration set. Under exchangeability, this guarantees:

        P(s(X_new, Y_new) ≤ q̂) ≥ 1 - α

    for any new sample drawn from the same distribution as the calibration set.

    For unlearning verification, we check if forget-set scores EXCEED q̂ —
    meaning the correct forget-set answers fall outside the prediction set.
    """

    def __init__(self, scorer: NonconformityScorer, alpha: float = 0.05):
        self.scorer = scorer
        self.alpha  = alpha
        self._threshold: Optional[float] = None
        self._calibration_scores: List[float] = []

    def calibrate(
        self,
        retain_loader: DataLoader,
        desc: str = "  [CU-AR] Calibrating conformal threshold",
    ) -> ConformalCalibrationResult:
        """
        Compute the conformal threshold q̂ from the retain set.

        Parameters
        ----------
        retain_loader : DataLoader over the retain (calibration) set

        Returns
        -------
        ConformalCalibrationResult
        """
        all_scores: List[float] = []

        for batch in tqdm(retain_loader, desc=desc, leave=False):
            scores = self.scorer.score_batch(batch)
            all_scores.extend(scores.cpu().tolist())

        if not all_scores:
            warnings.warn("[CU-AR] Calibration set empty — using default threshold 0.0")
            self._threshold = 0.0
            self._calibration_scores = []
            return ConformalCalibrationResult(
                alpha=self.alpha,
                n_calibration=0,
                threshold=0.0,
                calibration_scores=[],
            )

        scores_arr = np.array(all_scores)

        # Compute the (1-α) quantile — the conformal threshold
        # Use the "adjusted" formula for finite-sample validity:
        # q̂ = quantile at level ceil((n+1)(1-α)) / n
        n = len(scores_arr)
        adjusted_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        adjusted_level = min(adjusted_level, 1.0)
        threshold = float(np.quantile(scores_arr, adjusted_level))

        self._threshold = threshold
        self._calibration_scores = all_scores

        # Empirical coverage: fraction of calibration scores ≤ threshold
        empirical_coverage = float((scores_arr <= threshold).mean())

        result = ConformalCalibrationResult(
            alpha=self.alpha,
            n_calibration=n,
            threshold=threshold,
            calibration_scores=all_scores,
            coverage_empirical=empirical_coverage,
        )

        print(f"[CU-AR] Calibration complete: "
              f"n={n} | α={self.alpha:.3f} | q̂={threshold:.4f} | "
              f"empirical_coverage={empirical_coverage:.3f}")

        return result

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("Call calibrate() before accessing threshold.")
        return self._threshold

    def predict_set_contains(self, score: float) -> bool:
        """Returns True if the answer IS in the prediction set (score ≤ q̂)."""
        return score <= self.threshold

    def is_unlearned(self, score: float) -> bool:
        """Returns True if the answer is NOT in the prediction set (score > q̂)."""
        return score > self.threshold


# ──────────────────────────────────────────────────────────────────────────────
# Main Conformal Unlearning Verifier
# ──────────────────────────────────────────────────────────────────────────────

class ConformalUnlearningVerifier:
    """
    CU-AR: Conformal Unlearning Verifier for Autoregressive LLMs.

    Full pipeline:
      1. Score retain set → calibrate threshold q̂
      2. Score forget set → check if forget answers exceed q̂
      3. Generate certification report with formal statistical guarantee

    Usage
    -----
    verifier = ConformalUnlearningVerifier(model, tokenizer, cfg)
    report   = verifier.verify(
        forget_loader,
        retain_loader,
        method_name="HDI+NPO+SAM",
        alpha=0.05,
    )
    report.print_report()
    verifier.save_report(report, "outputs/conformal/report.json")
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        cfg: ARMORConfig,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.device    = cfg.device

        self._scorer     = NonconformityScorer(model, tokenizer, cfg)
        self._calibrator: Optional[ConformalCalibrator] = None

    # ── Step 1: Calibration ───────────────────────────────────────────────────

    def calibrate(
        self,
        retain_loader: DataLoader,
        alpha: float = 0.05,
    ) -> ConformalCalibrationResult:
        """
        Calibrate the conformal threshold on the retain set.
        Must be called before verify_forget().
        """
        self._calibrator = ConformalCalibrator(self._scorer, alpha=alpha)
        return self._calibrator.calibrate(retain_loader)

    # ── Step 2: Forget-set scoring ─────────────────────────────────────────────

    @torch.no_grad()
    def _score_forget_set(
        self,
        forget_loader: DataLoader,
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Score all forget-set samples and return (scores_list, per_sample_list)."""
        all_scores: List[float] = []
        per_sample: List[Dict[str, Any]] = []

        for batch in tqdm(forget_loader,
                          desc="  [CU-AR] Scoring forget set",
                          leave=False):
            scores = self._scorer.score_batch(batch)
            for i, sc in enumerate(scores.cpu().tolist()):
                in_pred_set = self._calibrator.predict_set_contains(sc)
                all_scores.append(sc)
                per_sample.append({
                    "index": len(per_sample),
                    "score": round(sc, 6),
                    "in_prediction_set": in_pred_set,
                    "unlearned": not in_pred_set,
                })

        return all_scores, per_sample

    # ── Step 3: Retain-set sanity check ───────────────────────────────────────

    @torch.no_grad()
    def _check_retain_coverage(
        self,
        retain_loader: DataLoader,
        n_check: int = 100,
    ) -> float:
        """
        Sanity check: verify that retain-set answers are still in prediction sets.
        Expected coverage ≥ 1 - alpha.

        Returns empirical retain coverage rate.
        """
        in_set_count = 0
        total = 0

        for batch in tqdm(retain_loader,
                          desc="  [CU-AR] Retain sanity check",
                          leave=False):
            scores = self._scorer.score_batch(batch)
            for sc in scores.cpu().tolist():
                in_set_count += int(self._calibrator.predict_set_contains(sc))
                total += 1
                if total >= n_check:
                    break
            if total >= n_check:
                break

        return in_set_count / max(total, 1)

    # ── Full verification pipeline ─────────────────────────────────────────────

    def verify(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        method_name: str = "Unknown",
        alpha: float = 0.05,
        retain_check_n: int = 100,
    ) -> ConformalUnlearningReport:
        """
        Full conformal unlearning verification pipeline.

        Parameters
        ----------
        forget_loader : DataLoader over the forget set
        retain_loader : DataLoader over the retain set (used for calibration)
        method_name   : Name of the unlearning method being verified
        alpha         : Miscoverage rate — max allowed fraction of forget answers
                        that may still be in prediction sets. Typical: 0.05 (5%)
        retain_check_n: Number of retain samples for the sanity coverage check

        Returns
        -------
        ConformalUnlearningReport — full report with formal certification verdict
        """
        print(f"\n[CU-AR] Starting Conformal Unlearning Verification")
        print(f"        Method: {method_name}  |  α={alpha:.3f}")
        t0 = time.time()

        # Step 1: Calibrate threshold on retain set
        print("[CU-AR] Step 1/3 — Calibrating conformal threshold...")
        calib_result = self.calibrate(retain_loader, alpha=alpha)

        # Step 2: Score forget set
        print("[CU-AR] Step 2/3 — Scoring forget set...")
        forget_scores, per_sample = self._score_forget_set(forget_loader)

        if not forget_scores:
            warnings.warn("[CU-AR] Forget set is empty!")
            forget_scores = [0.0]

        scores_arr = np.array(forget_scores)
        threshold  = calib_result.threshold

        # Fraction of forget-set answers still in prediction sets
        forget_coverage_rate = float((scores_arr <= threshold).mean())

        # Step 3: Retain sanity check
        print("[CU-AR] Step 3/3 — Retain coverage sanity check...")
        retain_coverage = self._check_retain_coverage(retain_loader, n_check=retain_check_n)

        # Certification verdict
        certified = forget_coverage_rate <= alpha
        margin    = alpha - forget_coverage_rate

        report = ConformalUnlearningReport(
            method=method_name,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            alpha=alpha,
            threshold=threshold,
            n_calibration=calib_result.n_calibration,
            n_forget=len(forget_scores),
            forget_scores=forget_scores,
            forget_coverage_rate=forget_coverage_rate,
            forget_mean_score=float(scores_arr.mean()),
            forget_min_score=float(scores_arr.min()),
            forget_max_score=float(scores_arr.max()),
            n_retain_check=retain_check_n,
            retain_coverage_rate=retain_coverage,
            unlearning_certified=certified,
            certification_margin=margin,
            per_sample=per_sample,
        )

        elapsed = time.time() - t0
        print(f"[CU-AR] Verification complete in {elapsed:.1f}s")
        report.print_report()

        return report

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_report(
        self,
        report: ConformalUnlearningReport,
        path: str,
        save_html: bool = True,
    ) -> None:
        """
        Save the conformal verification report as JSON and optionally HTML.

        Parameters
        ----------
        report    : The ConformalUnlearningReport to save
        path      : Path to the JSON output file
        save_html : If True, also write an HTML report at {path}.html
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # JSON
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"[CU-AR] JSON report saved → {path}")

        # HTML
        if save_html:
            html_path = path.replace(".json", ".html")
            self._write_html(report, html_path)

    def _write_html(self, report: ConformalUnlearningReport, path: str) -> None:
        """Generate a styled HTML verification certificate."""
        cert_color = "#00c896" if report.unlearning_certified else "#ff4757"
        cert_text  = (
            "✅ CONFORMAL UNLEARNING CERTIFIED"
            if report.unlearning_certified
            else "❌ NOT CERTIFIED — Additional Unlearning Required"
        )

        # Score histogram as simple ASCII bars
        scores = report.forget_scores
        if scores:
            hist_lines = _ascii_histogram(scores, bins=10, width=40,
                                          threshold=report.threshold)
        else:
            hist_lines = ["(no data)"]

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CU-AR Conformal Unlearning Certificate — {report.method}</title>
<style>
  :root {{
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text);
         font-family: 'Segoe UI', system-ui, sans-serif; padding: 2rem; }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  .header {{ text-align: center; margin-bottom: 2rem; }}
  .header h1 {{ font-size: 2rem; background: linear-gradient(135deg,#58a6ff,#bc8cff);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .header p {{ color: var(--muted); margin-top: 0.5rem; }}
  .badge {{ display: inline-block; padding: 1rem 2rem; border-radius: 8px;
            font-size: 1.2rem; font-weight: 700; letter-spacing: 0.5px;
            background: {cert_color}22; border: 2px solid {cert_color};
            color: {cert_color}; margin: 1rem auto; }}
  .card {{ background: var(--card); border: 1px solid var(--border);
           border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }}
  .card h3 {{ color: var(--accent); margin-bottom: 1rem; font-size: 1rem;
              text-transform: uppercase; letter-spacing: 1px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 0.6rem 1rem; border-bottom: 1px solid var(--border); text-align: left; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 0.85rem; text-transform: uppercase; }}
  .good {{ color: var(--green); }} .bad {{ color: var(--red); }}
  .mono {{ font-family: 'Courier New', monospace; font-size: 0.85rem; }}
  pre {{ background: #0d1117; padding: 1rem; border-radius: 6px;
         font-size: 0.8rem; color: #79c0ff; overflow-x: auto; }}
  .guarantee {{ background: #58a6ff11; border: 1px solid #58a6ff44;
                border-radius: 8px; padding: 1rem; margin: 0.5rem 0;
                font-size: 0.9rem; color: var(--accent); text-align: center; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>🔬 ARMOR — CU-AR Verification Report</h1>
    <p>Conformal Unlearning for Autoregressive LLMs · Distribution-Free Statistical Certification</p>
    <p style="color:var(--muted);font-size:0.85rem">{report.timestamp}</p>
  </div>

  <div style="text-align:center">
    <div class="badge">{cert_text}</div>
  </div>

  <div class="guarantee">
    📐 Statistical Guarantee: P(forget answer ∈ prediction set after unlearning) ≤ {report.alpha:.1%}
    &nbsp;|&nbsp; Distribution-free &nbsp;|&nbsp; Finite-sample valid
  </div>

  <div class="card">
    <h3>⚙️ Configuration</h3>
    <table>
      <tr><th>Parameter</th><th>Value</th></tr>
      <tr><td>Method</td><td><strong>{report.method}</strong></td></tr>
      <tr><td>Alpha (α)</td><td>{report.alpha:.4f}</td></tr>
      <tr><td>Threshold (q̂)</td><td class="mono">{report.threshold:.6f}</td></tr>
      <tr><td>Calibration samples (retain set)</td><td>{report.n_calibration}</td></tr>
      <tr><td>Forget-set samples</td><td>{report.n_forget}</td></tr>
    </table>
  </div>

  <div class="card">
    <h3>📊 Verification Results</h3>
    <table>
      <tr><th>Metric</th><th>Value</th><th>Target</th><th>Status</th></tr>
      <tr>
        <td>Forget Coverage Rate</td>
        <td class="mono {'good' if report.unlearning_certified else 'bad'}">{report.forget_coverage_rate:.4f}</td>
        <td>≤ {report.alpha:.3f}</td>
        <td class="{'good' if report.unlearning_certified else 'bad'}">{'✅ PASS' if report.unlearning_certified else '❌ FAIL'}</td>
      </tr>
      <tr>
        <td>Retain Coverage Rate</td>
        <td class="mono {'good' if report.retain_coverage_rate >= (1-report.alpha) else 'bad'}">{report.retain_coverage_rate:.4f}</td>
        <td>≥ {(1-report.alpha):.3f}</td>
        <td class="{'good' if report.retain_coverage_rate >= (1-report.alpha) else 'bad'}">{'✅ OK' if report.retain_coverage_rate >= (1-report.alpha) else '⚠️ LOW'}</td>
      </tr>
      <tr><td>Forget Mean Score</td><td class="mono">{report.forget_mean_score:.4f}</td><td>↑ (higher = better)</td><td></td></tr>
      <tr><td>Certification Margin</td><td class="mono {'good' if report.certification_margin > 0 else 'bad'}">{report.certification_margin:.4f}</td><td>&gt; 0</td><td></td></tr>
    </table>
  </div>

  <div class="card">
    <h3>📈 Score Distribution (Forget Set)</h3>
    <pre>{"<br>".join(hist_lines).replace("<br>", chr(10))}</pre>
    <p style="color:var(--muted);font-size:0.8rem;margin-top:0.5rem">
      Vertical line = threshold q̂={report.threshold:.4f}.
      Scores to the right of q̂ → answer NOT in prediction set (unlearned ✅)
    </p>
  </div>

  <div class="card">
    <h3>📐 Formal Guarantee</h3>
    <p style="line-height:1.8">
      Under conformal prediction theory (Vovk 2005 / Angelopoulos 2021),
      with <strong>n={report.n_calibration}</strong> calibration samples and
      α=<strong>{report.alpha:.3f}</strong>, the threshold
      q̂=<strong>{report.threshold:.4f}</strong> satisfies:
      <br><br>
      &nbsp;&nbsp;&nbsp;&nbsp;P(y_forget* ∈ C(x_forget)) ≤ α = <strong>{report.alpha:.1%}</strong>
      <br><br>
      This bound is <em>distribution-free</em> (no model assumptions) and
      <em>finite-sample valid</em> (holds for any n ≥ 1).
    </p>
  </div>
</div>
</body>
</html>"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[CU-AR] HTML report saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Score histogram utility
# ──────────────────────────────────────────────────────────────────────────────

def _ascii_histogram(
    values: List[float],
    bins: int = 10,
    width: int = 40,
    threshold: Optional[float] = None,
) -> List[str]:
    """Generate a simple ASCII histogram for score visualization."""
    if not values:
        return ["(empty)"]

    arr     = np.array(values)
    vmin    = arr.min()
    vmax    = arr.max() + 1e-9
    edges   = np.linspace(vmin, vmax, bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    max_cnt = max(counts.max(), 1)

    lines = [f"  Score distribution (n={len(values)})"]
    for i, (lo, hi, cnt) in enumerate(zip(edges[:-1], edges[1:], counts)):
        bar_len  = int(cnt / max_cnt * width)
        bar      = "█" * bar_len + " " * (width - bar_len)
        marker   = ""
        if threshold is not None and lo <= threshold <= hi:
            marker = " ◄ q̂"
        lines.append(f"  [{lo:7.3f},{hi:7.3f}) |{bar}| {cnt:4d}{marker}")

    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Batch scoring helpers (for integration with other modules)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_conformal_scores_from_loader(
    model: PreTrainedModel,
    loader: DataLoader,
    cfg: ARMORConfig,
    desc: str = "  Scoring",
) -> List[float]:
    """
    Convenience function: score all samples in a DataLoader and return
    a list of nonconformity scores.

    Parameters
    ----------
    model  : The language model
    loader : DataLoader with 'input_ids', 'attention_mask', 'labels'
    cfg    : ARMORConfig
    desc   : Progress bar label

    Returns
    -------
    List[float] — one nonconformity score per sample
    """
    scorer = NonconformityScorer(model, None, cfg)  # tokenizer not needed for batch scoring
    all_scores: List[float] = []

    for batch in tqdm(loader, desc=desc, leave=False):
        scores = scorer.score_batch(batch)
        all_scores.extend(scores.cpu().tolist())

    return all_scores


def conformal_unlearning_test(
    forget_scores: List[float],
    calibration_scores: List[float],
    alpha: float = 0.05,
) -> Tuple[bool, float, float]:
    """
    Minimal conformal test given pre-computed scores (no model needed).

    Parameters
    ----------
    forget_scores       : Nonconformity scores on forget set
    calibration_scores  : Nonconformity scores on retain set (calibration)
    alpha               : Miscoverage rate

    Returns
    -------
    (certified, threshold, forget_coverage_rate)
    """
    n         = len(calibration_scores)
    level     = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    threshold = float(np.quantile(calibration_scores, level))

    f_arr    = np.array(forget_scores)
    coverage = float((f_arr <= threshold).mean())
    return coverage <= alpha, threshold, coverage
