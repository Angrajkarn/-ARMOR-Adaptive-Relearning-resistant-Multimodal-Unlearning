"""
patch_fast.py  –  Add --fast / --epochs CLI flags to every run_*.py script.

Strategy:
  1. Add '--fast' and '--epochs' to argparse in each script.
  2. At the very top of main(), pre-parse those two flags and set
     ARMOR_FAST / ARMOR_EPOCHS env vars BEFORE ARMORConfig is created.
  3. ARMORConfig.__post_init__ already reads these env vars.
"""

import os, re, sys, textwrap

SCRIPTS_DIR = "scripts"
# These already have full --fast handling
SKIP = {
    "run_baseline_ga.py",
    "run_baseline_npo.py",
    "run_npo_sam.py",
    "run_smoke_tests.py",
    "start_api_server.py",
    "test_api_client.py",
}

fixed, skipped, failed = [], [], []

for fn in sorted(os.listdir(SCRIPTS_DIR)):
    if not fn.startswith("run_") or not fn.endswith(".py") or fn in SKIP:
        continue
    path = os.path.join(SCRIPTS_DIR, fn)
    txt = open(path, encoding="utf-8").read()

    # Already fully patched?
    if "'--fast'" in txt and "ARMOR_FAST" in txt:
        skipped.append(fn)
        continue

    # ── Step 0: Undo any broken earlier patches ─────────────────────────
    # Remove stray "import os\n" prepended before docstring
    if txt.startswith('import os\n"""'):
        txt = txt[len("import os\n"):]
    # Remove any half-injected blocks from the earlier attempt
    txt = re.sub(
        r"    parser\.add_argument\('--fast'.*?help='Override unlearn_epochs'\)\n",
        "", txt, flags=re.DOTALL,
    )
    txt = re.sub(
        r"    p\.add_argument\('--fast'.*?help='Override unlearn_epochs'\)\n",
        "", txt, flags=re.DOTALL,
    )
    txt = re.sub(
        r"\n    # ── Fast preset \(Colab.*?os\.environ\['ARMOR_EPOCHS'\].*?\n",
        "\n", txt, flags=re.DOTALL,
    )
    txt = re.sub(
        r"\n    # Apply --fast env vars.*?os\.environ\['ARMOR_EPOCHS'\].*?\n\n",
        "\n", txt, flags=re.DOTALL,
    )

    # ── Step 1: Find the ArgumentParser variable name ───────────────────
    ap = re.search(r"(\w+)\s*=\s*argparse\.ArgumentParser", txt)
    if not ap:
        failed.append(f"{fn}: no ArgumentParser")
        continue
    pvar = ap.group(1)  # e.g. 'p' or 'parser'

    # ── Step 2: Find parse_args() call ──────────────────────────────────
    pa = re.search(
        rf"(return\s+{pvar}|args\s*=\s*{pvar})\.parse_args\(\)", txt
    )
    if not pa:
        failed.append(f"{fn}: no parse_args()")
        continue

    # ── Step 3: Insert --fast/--epochs args before parse_args ───────────
    if "'--fast'" not in txt:
        new_args = (
            f"    {pvar}.add_argument('--fast', action='store_true',\n"
            f"                        help='Speed preset: retain=200, fp16, 1 epoch')\n"
            f"    {pvar}.add_argument('--epochs', type=int, default=None,\n"
            f"                        help='Override training epochs')\n"
        )
        # But check if --epochs already exists
        if "'--epochs'" in txt:
            new_args = (
                f"    {pvar}.add_argument('--fast', action='store_true',\n"
                f"                        help='Speed preset: retain=200, fp16, 1 epoch')\n"
            )
        txt = txt[: pa.start()] + new_args + txt[pa.start():]

    # ── Step 4: Insert ARMOR_FAST env-var setter at top of main() ───────
    if "ARMOR_FAST" not in txt:
        main_m = re.search(r"def main\(\):\s*\n", txt)
        if main_m:
            insert_at = main_m.end()
            env_block = textwrap.dedent("""\
                # ── Apply --fast / --epochs env vars before config creation ────────
                import argparse as _ap
                _pre = _ap.ArgumentParser(add_help=False)
                _pre.add_argument('--fast', action='store_true')
                _pre.add_argument('--epochs', type=int, default=None)
                _known, _ = _pre.parse_known_args()
                if _known.fast:
                    os.environ['ARMOR_FAST'] = '1'
                if _known.epochs:
                    os.environ['ARMOR_EPOCHS'] = str(_known.epochs)

            """)
            # Indent the block
            indented = "".join("    " + line + "\n" for line in env_block.strip().splitlines()) + "\n"
            txt = txt[:insert_at] + indented + txt[insert_at:]

    # ── Step 5: Ensure 'import os' exists ───────────────────────────────
    if "import os" not in txt.split("def ")[0]:
        # Add after the first import statement
        m = re.search(r"^(import \w+)", txt, re.MULTILINE)
        if m:
            txt = txt[: m.end()] + ", os" + txt[m.end():]

    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    fixed.append(fn)

print(f"FIXED  ({len(fixed)}): {fixed}")
print(f"SKIPPED({len(skipped)}): {skipped}")
print(f"FAILED ({len(failed)}): {failed}")
