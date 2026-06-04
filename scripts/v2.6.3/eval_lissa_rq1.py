#!/usr/bin/env python3
"""Re-evaluate LiSSA's shipped gpt-5-mini tracelinks against the TransArc gold
standards via transarc-emp's metric API, so RQ1's LiSSA columns are
apples-to-apples with TransArc / SWATTR / AALinker.

Reads LiSSA's per-project tracelink CSVs from the lissa-replication clone,
converts each into the result-set shape `compute_sad_sam_metrics` /
`compute_sad_code_metrics` expect, and writes the full metric suite (the same
schema as transarc-emp's `reports/metrics_*.csv`) to
`evaluation/reports/lissa_metrics_{sad-sam,sad-code}.csv`.

d2m covers all 5 projects; d2c only mediastore/teastore/bigbluebutton (LiSSA
never ran the other two). The missing d2c projects are emitted as an em-dash
row so the downstream table-filling step can distinguish them from a zero
score.
"""

import csv
import sys
from pathlib import Path

# transarc-emp exposes its API via src/lib; reuse compute_*_metrics + loaders.
LIB = Path("/mnt/hostshare/ardoco-home/transarc-emp/src/lib")
sys.path.insert(0, str(LIB))
sys.path.insert(0, str(LIB.parent / "paper"))

from metrics_api import (  # noqa: E402
    SCHEMA, NA, compute_sad_sam_metrics, compute_sad_code_metrics,
)
from transarc_error_analysis import (  # noqa: E402
    normalize_path, calc_metrics, load_gs_sad_sam,
    load_code_model_files, load_gs_sad_code_enrolled,
)

LISSA_ROOT = Path("/mnt/hostshare/ardoco-home/sota/lissa-replication"
                  "/results/tracelinks")
MODEL = "gpt-5-mini"

D2M_PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
D2C_PROJECTS = ["mediastore", "teastore", "bigbluebutton"]
D2C_MISSING = ["teammates", "jabref"]

REPORTS = Path("/mnt/hostshare/ardoco-home/transarc-emp/reports")


def load_lissa_d2m(project: str) -> set[tuple[str, str]]:
    """LiSSA d2m CSV → set[(modelElementID, sentence_str)] (same shape as
    `load_gs_sad_sam`). LiSSA writes `sentenceId,modelElementId`; flip to
    `(modelElementID, sentence)`."""
    path = LISSA_ROOT / "d2m" / f"{project}-{MODEL}.csv"
    links = set()
    with open(path) as f:
        for row in csv.reader(f):
            sentence_id, model_element_id = row[0], row[1]
            links.add((model_element_id, sentence_id))
    return links


def load_lissa_d2c(project: str) -> set[tuple[str, str]]:
    """LiSSA d2c CSV → set[(sentence_str, normalized_code_path)] (same shape
    as `load_result_sad_code`). LiSSA writes `sentenceId,codeFilePath` with
    an `Implementation/` prefix that `normalize_path` strips."""
    path = LISSA_ROOT / "d2c" / f"{project}-{MODEL}.csv"
    links = set()
    with open(path) as f:
        for row in csv.reader(f):
            sentence_id, code_path = row[0], row[1]
            links.add((sentence_id, normalize_path(code_path)))
    return links


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, NA) for k in SCHEMA})


RQ1_SCHEMA = ["project", "precision", "recall", "f1", "tp", "fp", "fn", "n_pred"]


def fmt_prf(row: dict) -> str:
    return "  ".join(
        f"{row[k]:.3f}" if isinstance(row[k], float) else str(row[k])
        for k in ("project", "precision", "recall", "f1")
    )


def run_sad_sam() -> tuple[list[dict], list[dict]]:
    """Returns (full-metric rows for lissa_metrics_sad-sam.csv,
                RQ1 P/R/F1 rows for lissa_rq1_d2m.csv)."""
    full_rows, rq1_rows = [], []
    print(f"=== LiSSA d2m ({MODEL}) ===")
    for proj in D2M_PROJECTS:
        res = load_lissa_d2m(proj)
        # Full unified suite (link/sentence/component/MCC/HUS/MAP).
        full_rows.append(compute_sad_sam_metrics(proj, res))
        # RQ1 P/R/F1 via calc_metrics against load_gs_sad_sam (link-level).
        gold = load_gs_sad_sam(proj)
        p, r, f1, tp, fp, fn = calc_metrics(gold, res)
        rq1_rows.append({
            "project": proj, "precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "n_pred": len(res),
        })
        print(f"  {fmt_prf(rq1_rows[-1])}  (tp={tp} fp={fp} fn={fn})")
    return full_rows, rq1_rows


def run_sad_code() -> tuple[list[dict], list[dict]]:
    full_rows, rq1_rows = [], []
    print(f"=== LiSSA d2c ({MODEL}) ===")
    for proj in D2C_PROJECTS:
        res = load_lissa_d2c(proj)
        full_rows.append(compute_sad_code_metrics(proj, res))
        # RQ1 P/R/F1 = file-level metrics against enrolled gold (the file_f1
        # `compute_sad_code_metrics` reports is calc_metrics(enrolled, res)[2]).
        code_model = load_code_model_files(proj)
        enrolled_gold = load_gs_sad_code_enrolled(proj, code_model)
        p, r, f1, tp, fp, fn = calc_metrics(enrolled_gold, res)
        rq1_rows.append({
            "project": proj, "precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "n_pred": len(res),
        })
        print(f"  {fmt_prf(rq1_rows[-1])}  (tp={tp} fp={fp} fn={fn})")
    for proj in D2C_MISSING:
        full_rows.append({**{k: NA for k in SCHEMA}, "project": proj})
        rq1_rows.append({**{k: NA for k in RQ1_SCHEMA}, "project": proj})
        print(f"  {proj}  ---  (LiSSA did not run d2c)")
    return full_rows, rq1_rows


def write_rq1_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RQ1_SCHEMA)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, NA) for k in RQ1_SCHEMA})


def main() -> None:
    sad_sam_full, sad_sam_rq1 = run_sad_sam()
    sad_code_full, sad_code_rq1 = run_sad_code()

    sam_path = REPORTS / "lissa_metrics_sad-sam.csv"
    code_path = REPORTS / "lissa_metrics_sad-code.csv"
    rq1_d2m_path = REPORTS / "lissa_rq1_d2m.csv"
    rq1_d2c_path = REPORTS / "lissa_rq1_d2c.csv"
    write_csv(sam_path, sad_sam_full)
    write_csv(code_path, sad_code_full)
    write_rq1_csv(rq1_d2m_path, sad_sam_rq1)
    write_rq1_csv(rq1_d2c_path, sad_code_rq1)
    print(f"\nWrote {sam_path}")
    print(f"Wrote {code_path}")
    print(f"Wrote {rq1_d2m_path}")
    print(f"Wrote {rq1_d2c_path}")


if __name__ == "__main__":
    main()
