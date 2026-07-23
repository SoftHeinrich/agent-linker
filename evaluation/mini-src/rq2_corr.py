#!/usr/bin/env python3
"""RQ2 per-(system, project) cell panel + rank-correlation of the size-aware
metrics against link-level (file) F1.

``rq12.py`` macro-averages each system's per-project ``metrics.compute_sad_code``
vectors into ONE row per system; the per-project cells are computed and thrown
away. This driver keeps the cells: it emits the per-(system, project) panel behind the
per-project worst-component claims in ``results.tex`` sec:results:rq2 (those figures now
appear in the appendix ``tab:detailed-perproject``; the old standalone
``appendix/rq2-tail.tex`` was archived).

NOTE the rank correlation is computed and written (RQ2_CORR.csv) only as a
DIAGNOSTIC: the paper does NOT report it. Under the canonical s21 variant the
size-aware metrics correlate too strongly with file F1 (coverage rho .90,
worst-component .67, harmonic .70) to read as "adds information file F1 does not",
so RQ2 was reframed around the limitation a looks-good standard score hides rather
than around a low correlation.

Cells = the three systems shown in the RQ2 body floats (fig:rq2-profile) on the
doc-to-code task, GPT-5.4 backend, x the five projects:
  TransArC, Artemis (GPT-5.4), approach (GPT-5.4; canonical Full s_linker21, N=3 mean).
=> 15 cells. The approach cell is the mean of run1/run2/run3; the two baselines
are single-shot.

For each cell: file F1, sentence coverage, worst-component F1, harmonic-mean
component F1 -- the same four ``RQ2_COLS`` as ``RQ2_PANEL.csv``. We then report
Spearman rho between file F1 and each of the other three across the 15 cells.

No new metric code: every cell comes from ``metrics.compute_sad_code`` (the sole
impl, pinned by ``check.py``) loaded through ``rq12``'s roster + SOTA dump, so
the cell means re-aggregate exactly to the ``RQ2_PANEL.csv`` system rows.
Stdlib-only Spearman (tie-aware average ranks), matching ``metrics.py``.

Usage
-----
    python3 mini-src/rq2_corr.py                       # stdout + reports/RQ2_CELLS.csv, RQ2_CORR.csv
    python3 mini-src/rq2_corr.py --cells /tmp/c.csv --corr /tmp/r.csv
    python3 mini-src/rq2_corr.py --backend claude      # Claude cells instead of GPT-5.4
"""

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics as m   # noqa: E402  (sole metric impl + loaders)
import rq12           # noqa: E402  (ROSTER + SOTA_LINKS resolution)

# RQ2 cell systems, by backend. Labels must match rq12.ROSTER. Canonical Full =
# s_linker21 (S21); rq12.ROSTER's plain "approach (...)" labels resolve to the S21
# dump (gpt-5.4_s21 / sonnet_s21), matching the body RQ2 floats.
SYSTEMS = {
    "gpt-5.4": ["TransArC", "Artemis (GPT-5.4)", "approach (GPT-5.4)"],
    "claude":  ["TransArC", "Artemis (GPT-5.4)", "approach (Claude)"],
}
# Short, paper-facing project names for the table.
PROJ_NAME = {"mediastore": "MediaStore", "teastore": "TeaStore",
             "teammates": "Teammates", "bigbluebutton": "BigBlueButton",
             "jabref": "JabRef"}
# size-aware metrics correlated against file F1 (the reference link-level metric).
SIZE_AWARE = [
    ("sentence_coverage", "sentence coverage"),
    ("worst_component_f1", "worst-component F1"),
    ("harmonic_component_f1", "harmonic-component F1"),
]


def system_by_label(label):
    for s in rq12.ROSTER:
        if s["label"] == label:
            return s
    raise SystemExit(f"unknown system label: {label}")


def cell_vector(system, project):
    """Per-project doc-to-code vector for one system, mean over its runs."""
    pattern = system["sad-code"]
    runs = system["runs"] or [None]
    vecs = []
    for run in runs:
        rel = pattern.format(run=run, project=project) if run else pattern.format(project=project)
        path = rq12.SOTA_LINKS / rel
        if not path.exists():
            raise SystemExit(f"missing doc-to-code result for {system['label']} "
                             f"{run or 'single'} {project}: {path}")
        res = m.load_result(path, "sad-code")
        if not res:
            raise SystemExit(f"empty/unparseable doc-to-code result for "
                             f"{system['label']} {run or 'single'} {project}: {path}")
        vecs.append(m.compute_sad_code(project, res))
    cols = m.PANELS["sad-code"]
    return {c: sum(v[c] for v in vecs) / len(vecs) for c in cols}


def build_cells(labels):
    cells = []
    for label in labels:
        system = system_by_label(label)
        for proj in m.PROJECTS:
            v = cell_vector(system, proj)
            cells.append({
                "system": label,
                "project": PROJ_NAME[proj],
                "file_f1": v["file_f1"],
                "sentence_coverage": v["sentence_coverage"],
                "worst_component_f1": v["worst_component_f1"],
                "harmonic_component_f1": v["harmonic_component_f1"],
            })
    return cells


def _avg_ranks(xs):
    """1-based ranks with ties resolved to the average of the tied positions."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # mean of 0-based positions i..j, made 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman's rho (Pearson on average ranks; tie-aware). Stdlib only."""
    n = len(xs)
    rx, ry = _avg_ranks(xs), _avg_ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx * vy) ** 0.5


def correlations(cells):
    f1 = [c["file_f1"] for c in cells]
    return [{"metric": key, "label": name, "rho": spearman(f1, [c[key] for c in cells]),
             "n": len(cells)} for key, name in SIZE_AWARE]


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def print_cells(cells):
    cols = ["file_f1", "sentence_coverage", "worst_component_f1", "harmonic_component_f1"]
    w_sys = max(len(c["system"]) for c in cells) + 1
    w_proj = max(len(c["project"]) for c in cells) + 1
    head = "system".ljust(w_sys) + "project".ljust(w_proj) + "".join(c.rjust(24) for c in cols)
    print(head)
    print("-" * len(head))
    for c in cells:
        print(c["system"].ljust(w_sys) + c["project"].ljust(w_proj)
              + "".join(fmt(c[k]).rjust(24) for k in cols))


def print_corr(corr, n):
    print(f"\nSpearman rho vs file F1 across {n} (system, project) cells:")
    for r in corr:
        print(f"  file F1 vs {r['label']:<24} rho = {r['rho']:+.3f}")


def write_cells_csv(cells, path):
    fields = ["system", "project", "file_f1", "sentence_coverage",
              "worst_component_f1", "harmonic_component_f1"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(fields)
        for c in cells:
            w.writerow([fmt(c[k]) for k in fields])


def write_corr_csv(corr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["reference", "metric", "metric_label", "spearman_rho", "n_cells"])
        for r in corr:
            w.writerow(["file_f1", r["metric"], r["label"], f"{r['rho']:.4f}", r["n"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=sorted(SYSTEMS), default="gpt-5.4",
                    help="approach variant for the approach cell (default: gpt-5.4 = S21, the RQ2 body variant)")
    ap.add_argument("--cells", default=None, help="per-cell CSV path (default: reports/RQ2_CELLS.csv)")
    ap.add_argument("--corr", default=None, help="correlation CSV path (default: reports/RQ2_CORR.csv)")
    args = ap.parse_args()

    cells = build_cells(SYSTEMS[args.backend])
    corr = correlations(cells)
    print_cells(cells)
    print_corr(corr, len(cells))
    print(f"\nProvenance: {rq12.SOTA_LINKS}  (approach cell = run1/run2/run3 mean; backend={args.backend})")

    reports = m._ARDOCO_HOME / "transarc-emp/reports"
    cells_out = Path(args.cells) if args.cells else reports / "RQ2_CELLS.csv"
    corr_out = Path(args.corr) if args.corr else reports / "RQ2_CORR.csv"
    write_cells_csv(cells, cells_out)
    write_corr_csv(corr, corr_out)
    print(f"\n[rq2_corr] wrote {cells_out}\n[rq2_corr] wrote {corr_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
