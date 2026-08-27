#!/usr/bin/env python3
"""Evaluate RQ3/RQ4 variants with the RQ2 doc-to-code metric suite.

RQ3/RQ4 are native SAD-SAM (doc-to-model) analyses. This companion asks whether
their conclusions survive the RQ2 doc-to-code lens: compose each phase-cache
doc-to-model link set through the recovered SAM-CODE links, then score the
resulting doc-to-code links with the RQ2 panel copied from ``mini-src/metrics.py``.

Outputs:
    reports/rq34_rq2_variants.csv   RQ3 variants: Full / validators removed
    reports/rq34_rq2_linkers.csv    RQ4 linker sets: Full / EntityOnly / CorefOnly
    reports/RQ34_RQ2_INVESTIGATION.md

No imports from ``mini-src``: the RQ2 metric primitives/loaders are copied here.
The phase-cache reader is reused from this mini-study's own ``rq34.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rq34 as rq  # noqa: E402  (same mini-study; phase-cache reader)


# --------------------------------------------------------------------------- #
# Roots and benchmark layout copied from mini-src/metrics.py.
# --------------------------------------------------------------------------- #
_ARDOCO_HOME = HERE.parents[1]
BENCHMARK = Path(os.environ.get(
    "TRANSARC_BENCHMARK",
    _ARDOCO_HOME / "ardoco/core/tests-base/src/main/resources/benchmark",
))
DEFAULT_RESULTS = Path(os.environ.get(
    "TRANSARC_RESULTS_DIR",
    _ARDOCO_HOME / "transarc-emp/mini-data",
))

GS_SAD_CODE = {
    "mediastore":    "mediastore/goldstandards/goldstandard_sad_2016-code_2016.csv",
    "teastore":      "teastore/goldstandards/goldstandard_sad_2020-code_2022.csv",
    "teammates":     "teammates/goldstandards/goldstandard_sad_2021-code_2023.csv",
    "bigbluebutton": "bigbluebutton/goldstandards/goldstandard_sad_2021-code_2023.csv",
    "jabref":        "jabref/goldstandards/goldstandard_sad_2021-code_2023.csv",
}
GS_SAM_CODE = {
    "mediastore":    "mediastore/goldstandards/goldstandard_sam_2016-code_2016.csv",
    "teastore":      "teastore/goldstandards/goldstandard_sam_2020-code_2022.csv",
    "teammates":     "teammates/goldstandards/goldstandard_sam_2021-code_2023.csv",
    "bigbluebutton": "bigbluebutton/goldstandards/goldstandard_sam_2021-code_2023.csv",
    "jabref":        "jabref/goldstandards/goldstandard_sam_2021-code_2023.csv",
}
ACM_FILES = {
    "mediastore":    "mediastore/model_2016/code/codeModel.acm",
    "teastore":      "teastore/model_2022/code/codeModel.acm",
    "teammates":     "teammates/model_2023/code/codeModel.acm",
    "bigbluebutton": "bigbluebutton/model_2023/code/codeModel.acm",
    "jabref":        "jabref/model_2023/code/codeModel.acm",
}

LinkKey = Tuple[int, str]
DocCodeLink = Tuple[str, str]

PANEL = [
    "doc_to_code_file_precision",
    "doc_to_code_file_recall",
    "doc_to_code_file_f1",
    "doc_to_code_file_f2",
    "doc_to_code_component_micro_f1",
    "doc_to_code_component_micro_f2",
    "doc_to_code_worst_component_f1",
    "doc_to_code_worst_component_f2",
    "doc_to_code_harmonic_component_f1",
    "doc_to_code_harmonic_component_f2",
]
DELTA_COLS = [
    "doc_to_code_file_f1",
    "doc_to_code_file_f2",
    "doc_to_code_worst_component_f1",
    "doc_to_code_worst_component_f2",
    "doc_to_code_harmonic_component_f1",
    "doc_to_code_harmonic_component_f2",
]


# --------------------------------------------------------------------------- #
# RQ2 doc-to-code metric implementation copied from mini-src/metrics.py.
# --------------------------------------------------------------------------- #
def normalize_path(path: str) -> str:
    prefix = "Implementation/"
    return path[len(prefix):] if path.startswith(prefix) else path


def enroll(gold: Iterable[DocCodeLink], code_files: Set[str]) -> Set[DocCodeLink]:
    enrolled = set()
    for gid, gpath in gold:
        if gpath.endswith("/"):
            for fp in code_files:
                if fp.startswith(gpath):
                    enrolled.add((gid, fp))
        else:
            enrolled.add((gid, gpath))
    return enrolled


def prf(gold: Set[DocCodeLink], res: Set[DocCodeLink]) -> Tuple[float, float, float]:
    if not res:
        return 0.0, 0.0, 0.0
    tp = len(gold & res)
    precision = tp / len(res)
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def fbeta(precision: float, recall: float, beta: float = 2.0) -> float:
    """Recall-weighted F-beta (beta=2 = the paper's \\ftwo); mirrors mini-src/metrics.py."""
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def load_code_model_files(project: str) -> Set[str]:
    files = set()
    with open(BENCHMARK / ACM_FILES[project], encoding="utf-8") as f:
        data = json.load(f)
    repo = data.get("codeItemRepository", {}).get("repository", {})
    for item in repo.values():
        if item.get("type") != "CodeCompilationUnit":
            continue
        parts = item.get("pathElements", [])
        name = item.get("name", "")
        ext = item.get("extension", "")
        if parts and name:
            full = "/".join(parts) + "/" + name + (f".{ext}" if ext else "")
            files.add(normalize_path(full))
    return files


def load_gs_sad_code_raw(project: str) -> Set[DocCodeLink]:
    with open(BENCHMARK / GS_SAD_CODE[project], encoding="utf-8") as f:
        return {(r["sentenceID"], normalize_path(r["codeID"])) for r in csv.DictReader(f)}


def load_file_to_comps(project: str, code_files: Set[str]) -> Dict[str, Set[str]]:
    names, raw = {}, set()
    with open(BENCHMARK / GS_SAM_CODE[project], encoding="utf-8") as f:
        for r in csv.DictReader(f):
            names[r["ae_id"]] = r["ae_name"]
            raw.add((r["ae_id"], normalize_path(r.get("ce_ids") or r.get("ce_id"))))
    file_to_comps = defaultdict(set)
    for ae, fp in enroll(raw, code_files):
        name = names.get(ae, ae)
        if name.startswith("Interface:"):
            continue
        file_to_comps[fp].add(ae)
    return file_to_comps


def compute_sad_code(project: str, res: Set[DocCodeLink]) -> Dict[str, float]:
    code_files = load_code_model_files(project)
    gold = enroll(load_gs_sad_code_raw(project), code_files)
    file_to_comps = load_file_to_comps(project, code_files)

    fp_, fr_, ff1 = prf(gold, res)

    def to_comp(pairs):
        out = set()
        for s, c in pairs:
            for comp in file_to_comps.get(c, ()):
                out.add((s, comp))
        return out

    gold_c, res_c = to_comp(gold), to_comp(res)
    comp_p, comp_r, comp_f1 = prf(gold_c, res_c)

    # Per-component slices at the LINK grain (metric.tex eq:worst / eq:harm); mirrors
    # mini-src/metrics.py compute_sad_code -- keep the two in step.
    gold_by_c, res_by_c = defaultdict(set), defaultdict(set)
    for s, f in gold:
        for comp in file_to_comps.get(f, ()):
            gold_by_c[comp].add((s, f))
    for s, f in res:
        for comp in file_to_comps.get(f, ()):
            res_by_c[comp].add((s, f))

    def comp_score(c):
        """(F1, F2) for one gold component, over the links whose target belongs to c."""
        p, rec, f1 = prf(gold_by_c.get(c, set()), res_by_c.get(c, set()))
        return f1, fbeta(p, rec)

    def tail(scores):
        """(worst, harmonic mean) of the per-component scores; 0 if any is 0."""
        if not scores:
            return 0.0, 0.0
        harmonic = (len(scores) / sum(1.0 / x for x in scores)
                    if all(x > 0 for x in scores) else 0.0)
        return min(scores), harmonic

    per_gold = [comp_score(c) for c in gold_by_c]
    worst_f1, harmonic_f1 = tail([f1 for f1, _ in per_gold])
    worst_f2, harmonic_f2 = tail([f2 for _, f2 in per_gold])

    return {
        "doc_to_code_file_precision": fp_,
        "doc_to_code_file_recall": fr_,
        "doc_to_code_file_f1": ff1,
        "doc_to_code_file_f2": fbeta(fp_, fr_),
        "doc_to_code_component_micro_f1": comp_f1,
        "doc_to_code_component_micro_f2": fbeta(comp_p, comp_r),
        "doc_to_code_worst_component_f1": worst_f1,
        "doc_to_code_worst_component_f2": worst_f2,
        "doc_to_code_harmonic_component_f1": harmonic_f1,
        "doc_to_code_harmonic_component_f2": harmonic_f2,
    }


# --------------------------------------------------------------------------- #
# Composition: SAD-SAM phase-cache links -> recovered SAD-CODE links.
# --------------------------------------------------------------------------- #
def load_recovered_sam_code(project: str) -> Dict[str, Set[str]]:
    by_comp = defaultdict(set)
    path = DEFAULT_RESULTS / project / "sam-code" / f"samCodeTlr_{project}.csv"
    if not path.exists():
        raise SystemExit(f"missing required recovered sam-code file: {path}")
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            comp = (r.get("sentenceID") or r.get("modelElementID") or "").strip()
            code = (r.get("codeID") or "").strip()
            if comp and code:
                by_comp[comp].add(normalize_path(code))
    return by_comp


def compose_doc_code(project: str, sad_sam: Set[LinkKey]) -> Set[DocCodeLink]:
    sam_code = load_recovered_sam_code(project)
    return {(str(sentence), fp) for sentence, comp in sad_sam for fp in sam_code.get(comp, ())}


def score_project_sets(project: str, sets: Dict[str, Set[LinkKey]]) -> Dict[str, Dict[str, float]]:
    return {name: compute_sad_code(project, compose_doc_code(project, links))
            for name, links in sets.items()}


def macro(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {c: sum(r[c] for r in rows) / len(rows) for c in PANEL}


def add_average(rows: List[Dict[str, str]], keys: List[str]) -> List[Dict[str, str]]:
    out = list(rows)
    groups = sorted({tuple(r[k] for k in keys) for r in rows})
    for group in groups:
        subset = [r for r in rows if tuple(r[k] for k in keys) == group and r["run"] in rq.RUNS]
        if len(subset) != len(rq.RUNS):
            continue
        avg = {k: v for k, v in zip(keys, group)}
        avg["run"] = "average"
        for c in PANEL:
            avg[c] = f"{sum(float(r[c]) for r in subset) / len(subset):.6f}"
        for c in DELTA_COLS:
            dc = f"delta_{c}_vs_full"
            avg[dc] = f"{sum(float(r[dc]) for r in subset) / len(subset):+.6f}"
        out.append(avg)
    return out


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def add_average_perproject(rows: List[Dict[str, str]], keys: List[str]) -> List[Dict[str, str]]:
    """Average per-project rows over runs, per (keys..., project)."""
    out = list(rows)
    groups = sorted({tuple(r[k] for k in keys) + (r["project"],) for r in rows})
    for group in groups:
        keyvals, project = group[:-1], group[-1]
        subset = [r for r in rows
                  if tuple(r[k] for k in keys) == keyvals
                  and r["project"] == project and r["run"] in rq.RUNS]
        if len(subset) != len(rq.RUNS):
            continue
        avg = {k: v for k, v in zip(keys, keyvals)}
        avg["run"] = "average"
        avg["project"] = project
        for c in PANEL:
            avg[c] = f"{sum(float(r[c]) for r in subset) / len(subset):.6f}"
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
            add_average_perproject(variant_pp, ["backend", "variant"]),
            add_average_perproject(linker_pp, ["backend", "linker_set"]))


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
