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
TABLES = Path("/mnt/hostshare/ardoco-home/agent-linker/writing/gen/table")

# Per-project display names used as the leading cell in the RQ1 tables.
PROJECT_DISPLAY = {
    "mediastore":    "MediaStore",
    "teastore":      "TeaStore",
    "teammates":     "Teammates",
    "bigbluebutton": "BigBlueButton",
    "jabref":        "JabRef",
}


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


# ── RQ1 table rendering ───────────────────────────────────────────────────────
# Each RQ1 booktabs table has the structure:
#   <project> & <P_sys1> & <R_sys1> & <F1_sys1>     <- systems = (SWATTR, LiSSA,
#             & <P_sys2> & <R_sys2> & <F1_sys2>        AALinker) for d2m and
#             & <P_sys3> & <R_sys3> & <F1_sys3> \\     (TransArc, LiSSA, AALinker)
#                                                      for d2c.
# LiSSA is system index 1 (the middle 3 cells, 1-indexed parts[4:7]).
# `render_lissa_row` rewrites only those three cells, preserving the systems
# on either side and any \textbf{} decoration they carry.

LISSA_COLS = (4, 5, 6)  # parts indices for LiSSA P, R, F1 after split("&")


def _fmt_cell(v) -> str:
    """Render a CSV cell as a fixed-width 3-decimal LaTeX literal, or '---'.

    The CSV writer serialises floats with full precision and em-dashes for N/A
    rows. The table cells are right-aligned to a 6-char field (matching the
    existing manual formatting), so columns line up in raw .tex too.
    """
    if v in (NA, "—", "---", "", None):
        return f"{'---':>6}"
    return f"{float(v):.3f}"


def _read_rq1_csv(path: Path) -> dict[str, dict]:
    """project -> row dict. Numeric fields stay as strings (we just reformat)."""
    with open(path) as f:
        return {row["project"]: row for row in csv.DictReader(f)}


def _row_lookup(rows: dict[str, dict], display_name: str) -> dict | None:
    """Map a table's display name back to the CSV project key."""
    for key, disp in PROJECT_DISPLAY.items():
        if disp == display_name and key in rows:
            return rows[key]
    return None


def _macro_row(rows: dict[str, dict], projects: list[str]) -> dict:
    """Compute the macro-average row across `projects`. Returns em-dash cells
    if any selected project has a non-numeric value, matching the table
    convention 'macro = mean across five projects' (no partial mixing)."""
    out = {"precision": NA, "recall": NA, "f1": NA}
    for key in ("precision", "recall", "f1"):
        try:
            vals = [float(rows[p][key]) for p in projects]
        except (KeyError, ValueError):
            return out
        out[key] = sum(vals) / len(vals)
    return out


def _rewrite_lissa(line: str, lissa_row: dict | None) -> str:
    """Return `line` with the LiSSA P/R/F1 cells replaced by `lissa_row`'s
    values. If the line doesn't look like a data row (no '&' or trailing
    '\\\\'), or `lissa_row` is None, return it unchanged."""
    if lissa_row is None or "&" not in line or "\\\\" not in line:
        return line
    # Preserve the trailing whitespace/newline by splitting at `\\` first.
    body, _, tail = line.rpartition("\\\\")
    parts = body.split("&")
    if len(parts) < 10:
        return line  # not a 10-cell row (project + 9 numeric cols)
    parts[LISSA_COLS[0]] = f" {_fmt_cell(lissa_row['precision'])} "
    parts[LISSA_COLS[1]] = f" {_fmt_cell(lissa_row['recall'])} "
    parts[LISSA_COLS[2]] = f" {_fmt_cell(lissa_row['f1'])} "
    return "&".join(parts) + "\\\\" + tail


def render_rq1_table(table_path: Path, csv_path: Path,
                     macro_projects: list[str]) -> None:
    """In-place rewrite the LiSSA cells of `table_path` from `csv_path`."""
    rows = _read_rq1_csv(csv_path)
    macro = _macro_row(rows, macro_projects)
    macro_disp = {**macro, "project": "Macro average"}
    rows["Macro average"] = macro_disp

    out_lines = []
    for line in table_path.read_text().splitlines(keepends=True):
        stripped = line.lstrip()
        leading = stripped.split("&", 1)[0].strip()
        lissa = None
        if leading == "Macro average":
            lissa = macro_disp
        elif leading in PROJECT_DISPLAY.values():
            lissa = _row_lookup(rows, leading)
        out_lines.append(_rewrite_lissa(line, lissa))
    table_path.write_text("".join(out_lines))
    print(f"Rewrote LiSSA cells in {table_path}")


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

    render_rq1_table(TABLES / "rq1-doc-to-model.tex", rq1_d2m_path,
                     macro_projects=D2M_PROJECTS)
    render_rq1_table(TABLES / "rq1-doc-to-code.tex", rq1_d2c_path,
                     macro_projects=D2C_PROJECTS + D2C_MISSING)


if __name__ == "__main__":
    main()
