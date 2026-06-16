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
