#!/usr/bin/env python3
"""Evaluate RQ3/RQ4 variants with the RQ2 doc-to-code metric suite.

RQ3/RQ4 are native SAD-SAM (doc-to-model) analyses. This companion asks whether
their conclusions survive the RQ2 doc-to-code lens: compose each phase-cache
doc-to-model link set through the recovered SAM-CODE links, then score the
resulting doc-to-code links with the RQ2 panel from ``mini-src/metrics.py``.

Outputs:
    reports/rq34_rq2_variants.csv   RQ3 variants: Full / validators removed
    reports/rq34_rq2_linkers.csv    RQ4 linker sets: Full / EntityOnly / CorefOnly
    reports/RQ34_RQ2_INVESTIGATION.md

No metric code lives here: ``mini-src/metrics.py`` is the sole implementation
(pinned by ``mini-src/check.py``) and this module only renames its panel keys to
the paper's doc-to-code column names. The phase-cache reader is reused from this
mini-study's own ``rq34.py``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "mini-src"))
import metrics as m  # noqa: E402  (shared core: benchmark layout + the RQ2 panel)
import rq34 as rq  # noqa: E402  (same mini-study; phase-cache reader)


LinkKey = Tuple[int, str]
DocCodeLink = Tuple[str, str]

# The RQ2 doc-to-code panel under the paper's column names -> the metric key in
# ``metrics.PANELS["sad-code"]``. Adding a metric to the suite means adding it
# here, not implementing it here.
PANEL_KEY = {
    "doc_to_code_file_precision": "file_p",
    "doc_to_code_file_recall": "file_r",
    "doc_to_code_file_f1": "file_f1",
    "doc_to_code_file_f2": "file_f2",
    "doc_to_code_component_micro_f1": "component_f1",
    "doc_to_code_component_micro_f2": "component_f2",
    "doc_to_code_worst_component_f1": "worst_component_f1",
    "doc_to_code_worst_component_f2": "worst_component_f2",
    "doc_to_code_harmonic_component_f1": "harmonic_component_f1",
    "doc_to_code_harmonic_component_f2": "harmonic_component_f2",
}
PANEL = list(PANEL_KEY)
write_csv = m.write_dict_csv   # the tree's one dict-row CSV writer
DELTA_COLS = [
    "doc_to_code_file_f1",
    "doc_to_code_file_f2",
    "doc_to_code_worst_component_f1",
    "doc_to_code_worst_component_f2",
    "doc_to_code_harmonic_component_f1",
    "doc_to_code_harmonic_component_f2",
]


def compute_sad_code(project: str, res: Set[DocCodeLink]) -> Dict[str, float]:
    """The RQ2 doc-to-code panel for one composed link set, keyed by paper column.

    ``metrics.compute_sad_code`` does the work (enrolment, the per-component
    grouping, the D-12 interface drop, the worst/harmonic tail); this is a
    rename, so RQ2 here and RQ2 in ``rq12.py`` cannot drift apart.
    """
    row = m.compute_sad_code(project, res)
    return {name: row[key] for name, key in PANEL_KEY.items()}


# --------------------------------------------------------------------------- #
# Composition: SAD-SAM phase-cache links -> recovered SAD-CODE links.
# --------------------------------------------------------------------------- #
def load_recovered_sam_code(project: str) -> Dict[str, Set[str]]:
    by_comp = defaultdict(set)
    path = m.DEFAULT_RESULTS / project / "sam-code" / f"samCodeTlr_{project}.csv"
    if not path.exists():
        raise SystemExit(f"missing required recovered sam-code file: {path}")
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            comp = (r.get("sentenceID") or r.get("modelElementID") or "").strip()
            code = (r.get("codeID") or "").strip()
            if comp and code:
                by_comp[comp].add(m.normalize_path(code))
    return by_comp


def compose_doc_code(project: str, sad_sam: Set[LinkKey]) -> Set[DocCodeLink]:
    sam_code = load_recovered_sam_code(project)
    return {(str(sentence), fp) for sentence, comp in sad_sam for fp in sam_code.get(comp, ())}


def score_project_sets(project: str, sets: Dict[str, Set[LinkKey]]) -> Dict[str, Dict[str, float]]:
    return {name: compute_sad_code(project, compose_doc_code(project, links))
            for name, links in sets.items()}


def macro(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {c: sum(r[c] for r in rows) / len(rows) for c in PANEL}


def add_average(rows: List[Dict[str, str]], keys: List[str],
                per_project: bool = False) -> List[Dict[str, str]]:
    """Append an ``run=average`` row per group, averaging over the three runs.

    Groups are keyed by ``keys`` (plus ``project`` when ``per_project``); a group
    that does not have all three runs is left without an average rather than
    averaged over a partial set. The per-project rows carry no delta columns, so
    the deltas are averaged only for the aggregate rows.
    """
    group_keys = keys + (["project"] if per_project else [])
    out = list(rows)
    for group in sorted({tuple(r[k] for k in group_keys) for r in rows}):
        subset = [r for r in rows
                  if tuple(r[k] for k in group_keys) == group and r["run"] in rq.RUNS]
        if len(subset) != len(rq.RUNS):
            continue
        avg = dict(zip(group_keys, group))
        avg["run"] = "average"
        for c in PANEL:
            avg[c] = f"{sum(float(r[c]) for r in subset) / len(subset):.6f}"
        if not per_project:
            for c in DELTA_COLS:
                dc = f"delta_{c}_vs_full"
                avg[dc] = f"{sum(float(r[dc]) for r in subset) / len(subset):+.6f}"
        out.append(avg)
    return out


# RQ4 doc-to-code set names, in display order: the pipeline output then each linker alone.
LINKER_SET_NAMES = ["Full"] + [f"{ph['linker']}Only" for ph in rq.PHASES]


def build_rows(backends: List[str], runs: List[str]):
    """Returns (variant macro, linker macro, variant per-project, linker per-project)."""
    variant_rows: List[Dict[str, str]] = []
    linker_rows: List[Dict[str, str]] = []
    variant_pp: List[Dict[str, str]] = []
    linker_pp: List[Dict[str, str]] = []

    for backend in backends:
        slot = rq.SLOTS.get(backend, Path("/nonexistent"))   # s92 resolves per-run dirs
        for run in runs:
            variant_project_rows = defaultdict(list)
            linker_project_rows = defaultdict(list)
            for project in rq.PROJECTS:
                rq.require_phase_files(slot, run, backend, project)
                cell = rq.compute_cell(slot, run, backend, project)

                variant_scores = score_project_sets(project, rq.rq3_variant_sets(cell))
                for name, score in variant_scores.items():
                    variant_project_rows[name].append(score)
                    variant_pp.append({"backend": backend, "run": run, "variant": name,
                                       "project": project,
                                       **{c: f"{score[c]:.6f}" for c in PANEL}})

                linker_sets = {"Full": cell.final}
                linker_sets.update({f"{ph['linker']}Only": cell.kept[ph["key"]]
                                    for ph in rq.PHASES})
                linker_scores = score_project_sets(project, linker_sets)
                for name, score in linker_scores.items():
                    linker_project_rows[name].append(score)
                    linker_pp.append({"backend": backend, "run": run, "linker_set": name,
                                      "project": project,
                                      **{c: f"{score[c]:.6f}" for c in PANEL}})

            full_variant = macro(variant_project_rows["Full"])
            for name in rq.RQ3_VARIANTS:
                score = macro(variant_project_rows[name])
                row = {"backend": backend, "run": run, "variant": name}
                row.update({c: f"{score[c]:.6f}" for c in PANEL})
                row.update({f"delta_{c}_vs_full": f"{score[c] - full_variant[c]:+.6f}"
                            for c in DELTA_COLS})
                variant_rows.append(row)

            full_linker = macro(linker_project_rows["Full"])
            for name in LINKER_SET_NAMES:
                score = macro(linker_project_rows[name])
                row = {"backend": backend, "run": run, "linker_set": name}
                row.update({c: f"{score[c]:.6f}" for c in PANEL})
                row.update({f"delta_{c}_vs_full": f"{score[c] - full_linker[c]:+.6f}"
                            for c in DELTA_COLS})
                linker_rows.append(row)

    return (add_average(variant_rows, ["backend", "variant"]),
            add_average(linker_rows, ["backend", "linker_set"]),
            add_average(variant_pp, ["backend", "variant"], per_project=True),
            add_average(linker_pp, ["backend", "linker_set"], per_project=True))


def row_by(rows: List[Dict[str, str]], **query) -> Dict[str, str]:
    return next(r for r in rows if all(r[k] == v for k, v in query.items()))


def write_summary(path: Path, variant_rows: List[Dict[str, str]], linker_rows: List[Dict[str, str]]) -> None:
    lines = [
        "# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens",
        "",
        "Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, "
        "then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.",
        "",
        "## RQ3 Validator Counterfactuals",
        "",
    ]
    for backend in sorted({r["backend"] for r in variant_rows}):
        no_all = row_by(variant_rows, backend=backend, run="average", variant="NoValidator")
        lines.append(f"- **{backend} NoValidator vs Full:** "
                     f"file-F1 {no_all['delta_doc_to_code_file_f1_vs_full']}, "
                     f"file-F2 {no_all['delta_doc_to_code_file_f2_vs_full']}, "
                     f"worst-component F1 {no_all['delta_doc_to_code_worst_component_f1_vs_full']}, "
                     f"harmonic-component F1 {no_all['delta_doc_to_code_harmonic_component_f1_vs_full']}.")
        for ph in rq.PHASES:
            r = row_by(variant_rows, backend=backend, run="average", variant=ph["variant"])
            lines.append(f"- **{backend} {ph['variant']} (judge off) vs Full:** "
                         f"file-F1 {r['delta_doc_to_code_file_f1_vs_full']}, "
                         f"file-F2 {r['delta_doc_to_code_file_f2_vs_full']}.")
    lines += ["", "## RQ4 Linker Sets", ""]
    for backend in sorted({r["backend"] for r in linker_rows}):
        for name in LINKER_SET_NAMES[1:]:
            r = row_by(linker_rows, backend=backend, run="average", linker_set=name)
            lines.append(f"- **{backend} {name} vs Full:** "
                         f"file-F1 {r['delta_doc_to_code_file_f1_vs_full']}, "
                         f"file-F2 {r['delta_doc_to_code_file_f2_vs_full']}, "
                         f"worst-component F1 {r['delta_doc_to_code_worst_component_f1_vs_full']}.")
    lines += [
        "",
        "Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-root", type=Path, default=HERE / "reports",
                    help="output root (default: mini-rq34/reports)")
    ap.add_argument("--backends", nargs="+", default=list(rq.BACKENDS),
                    choices=list(rq.BACKENDS),
                    help=f"backends of the {rq.ARM} arm (default: all)")
    ap.add_argument("--runs", nargs="+", default=list(rq.RUNS), choices=rq.RUNS,
                    help="runs to score (default: run1 run2 run3)")
    args = ap.parse_args()

    rq.install_unpickler()
    variant_rows, linker_rows, variant_pp, linker_pp = build_rows(args.backends, args.runs)

    delta_cols = [f"delta_{c}_vs_full" for c in DELTA_COLS]
    write_csv(args.csv_root / "rq34_rq2_variants.csv",
              ["backend", "run", "variant"] + PANEL + delta_cols, variant_rows)
    write_csv(args.csv_root / "rq34_rq2_linkers.csv",
              ["backend", "run", "linker_set"] + PANEL + delta_cols, linker_rows)
    write_csv(args.csv_root / "rq34_rq2_variants_perproject.csv",
              ["backend", "run", "variant", "project"] + PANEL, variant_pp)
    write_csv(args.csv_root / "rq34_rq2_linkers_perproject.csv",
              ["backend", "run", "linker_set", "project"] + PANEL, linker_pp)
    write_summary(args.csv_root / "RQ34_RQ2_INVESTIGATION.md", variant_rows, linker_rows)

    print(f"[rq34-rq2] wrote {args.csv_root / 'rq34_rq2_variants.csv'}")
    print(f"[rq34-rq2] wrote {args.csv_root / 'rq34_rq2_linkers.csv'}")
    print(f"[rq34-rq2] wrote {args.csv_root / 'rq34_rq2_variants_perproject.csv'}")
    print(f"[rq34-rq2] wrote {args.csv_root / 'rq34_rq2_linkers_perproject.csv'}")
    print(f"[rq34-rq2] wrote {args.csv_root / 'RQ34_RQ2_INVESTIGATION.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
