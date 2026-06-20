"""
scripts/run_smoke_tests.py
==========================
Comprehensive smoke test runner for all ARMOR unlearning methods.

Runs every script in --debug mode, checks for success, and prints
a final pass/fail summary table.

Usage:
    python scripts/run_smoke_tests.py
    python scripts/run_smoke_tests.py --stop-on-fail
"""

import argparse
import subprocess
import sys
import time
import os

# Set UTF-8 encoding for stdout and stderr on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ─── Test suite definition ────────────────────────────────────────────────────
TESTS = [
    {
        "name"   : "Gradient Ascent (GA)",
        "cmd"    : ["python", "scripts/run_baseline_ga.py",
                    "--debug", "--no-rouge", "--no-save"],
    },
    {
        "name"   : "NPO",
        "cmd"    : ["python", "scripts/run_baseline_npo.py",
                    "--debug", "--no-rouge", "--no-save"],
    },
    {
        "name"   : "NPO + SAM  (ARMOR core)",
        "cmd"    : ["python", "scripts/run_npo_sam.py",
                    "--debug", "--no-rouge", "--no-save"],
    },
    {
        "name"   : "RMU",
        "cmd"    : ["python", "scripts/run_rmu.py",
                    "--debug", "--no-rouge", "--no-save"],
    },
    {
        "name"   : "Task Vector",
        "cmd"    : ["python", "scripts/run_task_vector.py",
                    "--debug", "--no-rouge", "--no-save"],
    },
    {
        "name"   : "MultiTask-NPO",
        "cmd"    : ["python", "scripts/run_multitask_unlearn.py",
                    "--debug", "--no-rouge", "--no-save", "--n-tasks", "2"],
    },
    {
        "name"   : "DP-NPO+SAM",
        "cmd"    : ["python", "scripts/run_dp_armor.py",
                    "--debug", "--no-rouge", "--no-save"],
    },
    {
        "name"   : "LLaVA NPO+SAM (text-only)",
        "cmd"    : ["python", "scripts/run_llava_unlearn.py",
                    "--debug", "--text-only", "--no-rouge", "--no-save"],
    },
    {
        "name"   : "MUSE Benchmark (books)",
        "cmd"    : ["python", "scripts/run_muse_benchmark.py",
                    "--debug", "--domain", "books", "--method", "npo_sam",
                    "--no-save"],
    },
    {
        "name"   : "Relearning Attack",
        "cmd"    : ["python", "scripts/run_relearning_attack.py",
                    "--debug", "--compare", "--original-acc", "0.3983",
                    "--no-save"],
    },
    {
        "name"   : "Module 1: Lifelong / Continual",
        "cmd"    : ["python", "scripts/run_continual_unlearn.py",
                    "--debug", "--no-rouge", "--no-save", "--num-cohorts", "2"],
    },
    {
        "name"   : "Module 2: MoE Targeted GA",
        "cmd"    : ["python", "scripts/run_moe_unlearn.py",
                    "--debug", "--no-rouge", "--no-save"],
    },
    {
        "name"   : "Module 3: RLACE RMU",
        "cmd"    : ["python", "scripts/run_rlace_rmu.py",
                    "--debug", "--no-rouge", "--no-save", "--rlace-iters", "10"],
    },
    {
        "name"   : "Module 4: ZK Verification",
        "cmd"    : ["python", "scripts/run_zk_verify.py",
                    "--debug", "--no-save", "--probe-samples", "4"],
    },
    {
        "name"   : "Module 5: Multimodal MIA Audit",
        "cmd"    : ["python", "scripts/run_multimodal_mia.py",
                    "--debug", "--no-save"],
    },
    {
        "name"   : "Module 6: Modular LoRA",
        "cmd"    : ["python", "scripts/run_lora_unlearn.py",
                    "--debug", "--no-save", "--no-rouge"],
    },
    {
        "name"   : "Module 7: NASD Decay",
        "cmd"    : ["python", "scripts/run_nasd.py",
                    "--debug", "--no-rouge"],
    },
    {
        "name"   : "Module 8: HDI Zero-Shot",
        "cmd"    : ["python", "scripts/run_hdi_unlearn.py",
                    "--debug", "--no-rouge", "--no-save"],
    },
    {
        "name"   : "Module 9: CAS Graph Blockade",
        "cmd"    : ["python", "scripts/run_cas_unlearn.py",
                    "--debug", "--no-rouge", "--no-save"],
    },
    {
        "name"   : "Reconstruction Attack (Model Inversion)",
        "cmd"    : ["python", "scripts/run_reconstruction_attack.py",
                    "--debug", "--no-save"],
    },
    {
        "name"   : "Audit Certificate Generator (GDPR Compliance)",
        "cmd"    : ["python", "scripts/run_audit_gen.py",
                    "--debug", "--probe-samples", "4"],
    },
    # ── Phase 1: New Frontier Research Methods ────────────────────────────────
    {
        "name"   : "Phase 1: CU-AR (Conformal Unlearning Verification)",
        "cmd"    : ["python", "scripts/run_conformal_verify.py",
                    "--debug", "--no-save", "--alpha", "0.10"],
    },
    {
        "name"   : "Phase 1: CoT-HME (Chain-of-Thought Erasure)",
        "cmd"    : ["python", "scripts/run_cot_hme.py",
                    "--debug", "--no-rouge", "--no-save",
                    "--cot-coeff", "0.2", "--cot-max-tokens", "24"],
    },
    {
        "name"   : "Phase 1: TKDU (Temporal Knowledge Decay)",
        "cmd"    : ["python", "scripts/run_temporal_unlearn.py",
                    "--debug", "--no-rouge", "--no-save",
                    "--halflife-days", "1.0"],
    },
    # ── Phase 2: New Frontier Research Methods ────────────────────────────────
    {
        "name"   : "Phase 2: LCAGE (Latent Concept Association Graph)",
        "cmd"    : ["python", "scripts/run_lcage.py",
                    "--debug", "--no-rouge", "--no-save"],
    },
    {
        "name"   : "Phase 2: NRU (Neural Reconsolidation)",
        "cmd"    : ["python", "scripts/run_reconsolidation.py",
                    "--debug", "--no-rouge", "--no-save"],
    },
    {
        "name"   : "Phase 2: MWRP (Morphogenetic Weight Regeneration)",
        "cmd"    : ["python", "scripts/run_morphogenetic_repair.py",
                    "--debug", "--no-rouge", "--no-save"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────

def run_test(test: dict, stop_on_fail: bool) -> dict:
    """Run a single test and return a result dict."""
    name = test["name"]
    cmd  = test["cmd"]

    print(f"\n{'='*64}")
    print(f"  SMOKE TEST: {name}")
    print(f"  CMD:        {' '.join(cmd)}")
    print(f"{'='*64}")

    t0 = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    result = subprocess.run(
        cmd,
        capture_output=False,   # stream output live
        text=True,
        env=env,
    )

    elapsed = time.time() - t0
    passed  = result.returncode == 0

    return {
        "name"   : name,
        "passed" : passed,
        "elapsed": elapsed,
        "code"   : result.returncode,
    }


def print_summary(results: list):
    print("\n" + "=" * 64)
    print("  ARMOR SMOKE TEST SUMMARY")
    print("=" * 64)
    print(f"  {'Method':<35} {'Status':<10} {'Time':>8}")
    print("  " + "-" * 58)

    n_pass = n_fail = 0
    for r in results:
        icon  = "✅ PASS" if r["passed"] else "❌ FAIL"
        n_pass += r["passed"]
        n_fail += (not r["passed"])
        print(f"  {r['name']:<35} {icon:<10} {r['elapsed']:>6.1f}s")

    print("  " + "-" * 58)
    total = n_pass + n_fail
    print(f"  Result: {n_pass}/{total} passed")
    print("=" * 64 + "\n")

    return n_fail == 0


def main():
    parser = argparse.ArgumentParser(description="ARMOR smoke test suite")
    parser.add_argument("--stop-on-fail", action="store_true",
                        help="Stop immediately on first failure")
    parser.add_argument("--tests", nargs="*", default=None,
                        help="Names of tests to run (substring match). "
                             "Runs all if not specified.")
    args = parser.parse_args()

    # Filter tests if requested
    tests = TESTS
    if args.tests:
        tests = [t for t in TESTS
                 if any(sel.lower() in t["name"].lower()
                        for sel in args.tests)]
        if not tests:
            print(f"No tests matched: {args.tests}")
            sys.exit(1)

    print(f"\n🛡️  ARMOR Smoke Test Suite  ({len(tests)} tests)")
    print(f"   Model: distilgpt2 (CPU debug)\n")

    results = []
    for test in tests:
        result = run_test(test, args.stop_on_fail)
        results.append(result)
        if not result["passed"] and args.stop_on_fail:
            print(f"\n❌ STOPPED: '{result['name']}' failed "
                  f"(exit code {result['code']})")
            print_summary(results)
            sys.exit(1)

    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
