"""Reconcile manual No-Knowledge hand-runs into the canonical 51-04 sweep layout.

The No-Knowledge cells were produced by `run_s20union_once.sh -k` under
`results/manual/<slug>__repN/` (CSVs flat at the rep root, phase_cache under
`phase_cache/s_linker20_union/<subdir>/<project>/`). The 51-04 deliverable and the
51-05 extractor's NOKNOW_MATRIX expect the canonical sweep layout:

    <canon_root>/run<i>/<project>/s_linker20_union_noknow_<project>_links.csv
    <canon_root>/run<i>/phase_cache/s_linker20_union/<subdir>/<project>/{layer1..4,final}.pkl
    <canon_root>/run<i>/<project>/.done

This copies (never moves — manual dirs stay as raw provenance) each rep into the
matching run<i>, nesting the flat CSVs under per-project dirs. Idempotent.

Usage:
    python scripts/reconcile_noknow_manual_to_canonical.py \
        --glob 'results/manual/s20union-noknow_openai_gpt-5.4__*' \
        --canon-root results/v2.6.6_s20union_noknow/gpt
    python scripts/reconcile_noknow_manual_to_canonical.py \
        --glob 'results/manual/s20union-noknow_claude_sonnet__*__rep1' \
        --canon-root results/v2.6.6_s20union_noknow_sonnet
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
from pathlib import Path

PROJECTS = ["bigbluebutton", "jabref", "mediastore", "teammates", "teastore"]
CSV_PREFIX = "s_linker20_union_noknow"


def rep_index(path: str) -> int:
    m = re.search(r"__rep(\d+)", os.path.basename(path))
    if not m:
        raise SystemExit(f"cannot derive rep index from {path!r} (no __repN tag)")
    return int(m.group(1))


def reconcile(src_rep: str, run_dir: str) -> dict:
    """Copy one manual rep dir into a canonical run<i> dir. Returns a summary."""
    out = {"run_dir": run_dir, "csvs": 0, "phase_cache": False, "projects": []}
    os.makedirs(run_dir, exist_ok=True)

    # 1) phase_cache (same internal structure) -> run<i>/phase_cache
    src_pc = os.path.join(src_rep, "phase_cache")
    dst_pc = os.path.join(run_dir, "phase_cache")
    if os.path.isdir(src_pc):
        if os.path.isdir(dst_pc):
            shutil.rmtree(dst_pc)
        shutil.copytree(src_pc, dst_pc)
        out["phase_cache"] = True

    # 2) flat CSVs -> run<i>/<project>/<csv>, plus a .done marker
    for proj in PROJECTS:
        src_csv = os.path.join(src_rep, f"{CSV_PREFIX}_{proj}_links.csv")
        if not os.path.isfile(src_csv):
            continue
        pdir = os.path.join(run_dir, proj)
        os.makedirs(pdir, exist_ok=True)
        shutil.copy2(src_csv, os.path.join(pdir, os.path.basename(src_csv)))
        # carry the ablation json if a per-project one is identifiable (best-effort)
        Path(os.path.join(pdir, ".done")).touch()
        out["csvs"] += 1
        out["projects"].append(proj)

    # 3) provenance note
    with open(os.path.join(run_dir, "PROVENANCE.txt"), "w") as f:
        f.write(f"reconciled-from: {src_rep}\n")
        f.write("tool: scripts/reconcile_noknow_manual_to_canonical.py\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob for manual rep dirs (must carry __repN)")
    ap.add_argument("--canon-root", required=True, help="canonical root, e.g. results/v2.6.6_s20union_noknow/gpt")
    args = ap.parse_args()

    reps = sorted(d for d in glob.glob(args.glob) if os.path.isdir(d) and "FAILED" not in d)
    if not reps:
        raise SystemExit(f"no manual rep dirs matched {args.glob!r}")

    print(f"canonical root: {args.canon_root}")
    total_csvs = 0
    for src in reps:
        i = rep_index(src)
        run_dir = os.path.join(args.canon_root, f"run{i}")
        res = reconcile(src, run_dir)
        total_csvs += res["csvs"]
        print(f"  rep{i}: {os.path.basename(src)}")
        print(f"     -> {run_dir}  (phase_cache={res['phase_cache']}, csvs={res['csvs']}: {','.join(res['projects'])})")

    # report canonical cell count
    n_cells = len(glob.glob(os.path.join(args.canon_root, "run*", "*", f"{CSV_PREFIX}_*_links.csv")))
    print(f"\ncanonical cells now present under {args.canon_root}: {n_cells}")
    print(f"copied this pass: {total_csvs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
