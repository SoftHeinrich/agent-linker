#!/usr/bin/env python3
"""Regenerate the RQ *data* CSVs into a temp dir and diff them against the repo.

A no-overwrite "data walkthrough": runs the three metric-computing generators
(rq12.py, rq34.py, rq34_rq2.py) with their outputs redirected to a scratch dir,
then byte-compares every produced file against the committed copy under the eval
repo's reports/. Your working tree is never written.

    python3 mini-src/gen_csv_to_temp.py                  # default temp dir, diff vs repo
    python3 mini-src/gen_csv_to_temp.py --out /some/dir    # choose the temp dir
    python3 mini-src/gen_csv_to_temp.py --no-diff          # generate only, skip the compare
    EVAL_ROOT=/path/to/transarc-emp python3 mini-src/gen_csv_to_temp.py

The two reshapers (rq_tables.py / csv_to_tex.py) have hardcoded output paths and
do no metric math, so they are intentionally skipped here. For a fully isolated
run that also includes them, use a git worktree of the eval repo.

Exit 0 = every generated CSV reproduces the repo copy; 1 = some file differs.
stdlib only.
"""
from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Default to the repo this script lives in (mini-src/ -> repo root); EVAL_ROOT overrides.
DEFAULT_EVAL = Path(os.environ.get("EVAL_ROOT") or Path(__file__).resolve().parent.parent)


def jobs(eval_root: Path, out: Path):
    """(label, argv, temp_output_base, repo_output_base) for each data generator."""
    py = sys.executable
    return [
        ("rq12  (RQ1 + RQ2)",
         [py, "mini-src/rq12.py",
          "--csv",           str(out / "rq12" / "RQ12_BIGTABLE.csv"),
          "--rq2-csv",       str(out / "rq12" / "RQ2_PANEL.csv"),
          "--perproject-csv", str(out / "rq12" / "RQ12_PERPROJECT.csv")],
         out / "rq12", eval_root / "reports"),
        ("rq34  (RQ3 + RQ4)",
         [py, "mini-rq34/rq34.py", "--csv-root", str(out / "rq34")],
         out / "rq34", eval_root / "mini-rq34" / "reports"),
        ("rq34_rq2  (RQ3/RQ4 size-aware)",
         [py, "mini-rq34/rq34_rq2.py", "--csv-root", str(out / "rq34_rq2")],
         out / "rq34_rq2", eval_root / "mini-rq34" / "reports"),
    ]


def diff_tree(temp_base: Path, repo_base: Path):
    """Compare every file generated under temp_base to its repo_base counterpart."""
    rows = []
    for path in sorted(temp_base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(temp_base)
        repo = repo_base / rel
        if not repo.is_file():
            rows.append((rel, "new"))           # produced now, not tracked in repo
        elif filecmp.cmp(path, repo, shallow=False):
            rows.append((rel, "identical"))
        else:
            rows.append((rel, "DIFFERS"))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL,
                    help=f"transarc-emp root (default: {DEFAULT_EVAL})")
    ap.add_argument("--out", type=Path,
                    default=Path(tempfile.gettempdir()) / "rq-csv-walkthrough",
                    help="temp output dir (wiped and recreated each run)")
    ap.add_argument("--no-diff", action="store_true",
                    help="generate only; skip the byte-compare against the repo")
    args = ap.parse_args(argv)

    eval_root = args.eval_root.resolve()
    out = args.out.resolve()
    if not (eval_root / "mini-src" / "rq12.py").is_file():
        sys.exit(f"ERROR: {eval_root} does not look like the transarc-emp repo")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    print(f"eval repo : {eval_root}")
    print(f"temp out  : {out}  (repo working tree is never written)\n")

    differs = failed = 0
    for label, cmd, temp_base, repo_base in jobs(eval_root, out):
        temp_base.mkdir(parents=True, exist_ok=True)
        print(f"=== {label} ===")
        print("  $ " + " ".join(cmd))
        if subprocess.run(cmd, cwd=eval_root).returncode != 0:
            print("  generator FAILED\n")
            failed += 1
            continue
        if args.no_diff:
            print()
            continue
        for rel, status in diff_tree(temp_base, repo_base):
            print(f"  [{status:9}] {rel}")
            if status == "DIFFERS":
                differs += 1
        print()

    if args.no_diff:
        print(f"Generated CSVs under {out} (repo untouched). Diff skipped.")
        return 0
    if failed or differs:
        print(f"RESULT: {failed} generator(s) failed, {differs} file(s) differ from the repo.")
        return 1
    print("RESULT: all generated CSVs reproduce the committed repo copies. Repo untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
