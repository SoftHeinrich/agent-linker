#!/usr/bin/env python3
"""Compute apples-to-apples RQ1 baseline cells (SWATTR + TransArc + LiSSA) and
print them as paste-ready LaTeX text for writing/gen/table/rq1-doc-to-{model,code}.tex.

Single source of truth for the three baseline systems against the TransArc
gold standards (sad-sam for d2m, enrolled sad-code for d2c). For each system,
the script:

  1. loads the system's predicted links via the appropriate loader,
  2. calls `calc_metrics(gold, res)` to get P/R/F1 + TP/FP/FN at the same
     granularity AALinker is evaluated at (link-level for d2m, file-level
     for d2c),
  3. writes the per-project + macro row to a CSV under
     `transarc-emp/reports/<system>_rq1_<task>.csv`,
  4. prints a paste-ready block of LaTeX cells per row, so a human can
     copy each row's `& P & R & F1` slice into the right column of the
     matching .tex table.

The script does NOT rewrite the .tex files — the user pastes the cells
manually.

Systems and slices:

  rq1-doc-to-model.tex (d2m):
    SWATTR    -> parts[1:4]  (cols 2-4)
    LiSSA     -> parts[4:7]  (cols 5-7)
    AALinker  -> parts[7:10] (cols 8-10) — not produced by this script

  rq1-doc-to-code.tex (d2c):
    TransArc  -> parts[1:4]
    LiSSA     -> parts[4:7]
    AALinker  -> parts[7:10] — not produced by this script
"""

import argparse
import csv
import sys
from pathlib import Path

# transarc-emp exposes its API via src/lib; reuse loaders + calc_metrics.
LIB = Path("/mnt/hostshare/ardoco-home/transarc-emp/src/lib")
sys.path.insert(0, str(LIB))
sys.path.insert(0, str(LIB.parent / "paper"))

from metrics_api import (  # noqa: E402
    SCHEMA, NA, compute_sad_sam_metrics, compute_sad_code_metrics,
)
from transarc_error_analysis import (  # noqa: E402
    normalize_path, calc_metrics,
    load_gs_sad_sam, load_code_model_files, load_gs_sad_code_enrolled,
    load_result_sad_sam_standalone, load_result_sad_code,
)

LISSA_ROOT = Path("/mnt/hostshare/ardoco-home/sota/lissa-replication"
                  "/results/tracelinks")
LISSA_MODEL = "gpt-5-mini"

D2M_PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
D2C_PROJECTS = ["mediastore", "teastore", "bigbluebutton"]
D2C_MISSING = ["teammates", "jabref"]

REPORTS = Path("/mnt/hostshare/ardoco-home/transarc-emp/reports")
TABLES = Path("/mnt/hostshare/ardoco-home/agent-linker/writing/gen/table")

# Per-project display names match the leading `Project` cell in the .tex tables.
PROJECT_DISPLAY = {
    "mediastore":    "MediaStore",
    "teastore":      "TeaStore",
    "teammates":     "Teammates",
    "bigbluebutton": "BigBlueButton",
    "jabref":        "JabRef",
}

RQ1_SCHEMA = ["project", "precision", "recall", "f1", "tp", "fp", "fn", "n_pred"]


# ── Predicted-link loaders ────────────────────────────────────────────────────

def load_lissa_d2m(project: str) -> set[tuple[str, str]]:
    """LiSSA d2m CSV → set[(modelElementID, sentence_str)] (matches
    `load_gs_sad_sam`). LiSSA's columns are `sentenceId,modelElementId`; flip
    them so the result-set shape matches the gold."""
    path = LISSA_ROOT / "d2m" / f"{project}-{LISSA_MODEL}.csv"
    links = set()
    with open(path) as f:
        for row in csv.reader(f):
            sentence_id, model_element_id = row[0], row[1]
            links.add((model_element_id, sentence_id))
    return links


def load_lissa_d2c(project: str) -> set[tuple[str, str]]:
    """LiSSA d2c CSV → set[(sentence_str, normalized_code_path)] (matches
    `load_result_sad_code`). LiSSA paths carry an `Implementation/` prefix
    that `normalize_path` strips."""
    path = LISSA_ROOT / "d2c" / f"{project}-{LISSA_MODEL}.csv"
    links = set()
    with open(path) as f:
        for row in csv.reader(f):
            sentence_id, code_path = row[0], row[1]
            links.add((sentence_id, normalize_path(code_path)))
    return links


# ── P/R/F1 computation ────────────────────────────────────────────────────────

def prf_d2m(loader, project: str) -> dict:
    """Link-level P/R/F1 of `loader(project)` against the sad-sam gold."""
    res = loader(project)
    gold = load_gs_sad_sam(project)
    p, r, f1, tp, fp, fn = calc_metrics(gold, res)
    return {
        "project": project, "precision": p, "recall": r, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "n_pred": len(res),
    }


def prf_d2c(loader, project: str) -> dict:
    """File-level P/R/F1 of `loader(project)` against the enrolled sad-code
    gold (the same granularity TransArc itself is reported at)."""
    res = loader(project)
    code_model = load_code_model_files(project)
    enrolled = load_gs_sad_code_enrolled(project, code_model)
    p, r, f1, tp, fp, fn = calc_metrics(enrolled, res)
    return {
        "project": project, "precision": p, "recall": r, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "n_pred": len(res),
    }


def macro_row(rows: list[dict]) -> dict:
    """Macro = arithmetic mean of P, R, F1 across all rows. Returns em-dash
    cells if any row contains non-numeric values, so a partial set never
    silently averages into a 5-project column."""
    out = {"project": "Macro average", "precision": NA, "recall": NA, "f1": NA}
    for key in ("precision", "recall", "f1"):
        try:
            vals = [float(r[key]) for r in rows]
        except (TypeError, ValueError):
            return out
        out[key] = sum(vals) / len(vals)
    return out


def na_row(project: str) -> dict:
    return {**{k: NA for k in RQ1_SCHEMA}, "project": project}


# ── CSV emit ──────────────────────────────────────────────────────────────────

def write_rq1_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RQ1_SCHEMA)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, NA) for k in RQ1_SCHEMA})


def write_full_csv(path: Path, rows: list[dict]) -> None:
    """The full unified metric suite (SCHEMA) used by the LiSSA reports."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, NA) for k in SCHEMA})


# ── Paste-ready printout ──────────────────────────────────────────────────────

CELL_NA = "  ---"  # 5-char em-dash slot (matches 5-char "0.000" placeholder)


def _fmt_cell(v) -> str:
    if v in (NA, "—", "---", "", None):
        return CELL_NA
    return f"{float(v):.3f}"


def _row_display(row: dict) -> str:
    name = PROJECT_DISPLAY.get(row["project"], row["project"])
    return (f"  {name:<14} :  & {_fmt_cell(row['precision'])}"
            f" & {_fmt_cell(row['recall'])}"
            f" & {_fmt_cell(row['f1'])}")


def print_block(title: str, target_table: str, slot_label: str,
                rows: list[dict]) -> None:
    bar = "─" * 70
    print(f"\n{bar}")
    print(f"{title}")
    print(f"  paste into: {target_table}")
    print(f"  slot      : {slot_label}")
    print(bar)
    for row in rows:
        print(_row_display(row))


# ── In-place .tex paste (opt-in via --write-tex) ──────────────────────────────
# Each RQ1 table row has 10 cells after splitting on '&':
#   parts[0] = "Project       "  ·  parts[1:4] = system 1 (SWATTR or TransArc)
#   parts[4:7] = LiSSA            ·  parts[7:10] = AALinker (untouched by us)
# `paste_systems` rewrites only the slices we own, preserving any \textbf{}
# decoration on AALinker cells.

def _name_to_row(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for r in rows:
        proj = r["project"]
        out[PROJECT_DISPLAY.get(proj, proj)] = r
    return out


def paste_systems(table_path: Path,
                  slice_to_rows: dict[tuple[int, int], list[dict]]) -> None:
    """In-place rewrite of `table_path`. `slice_to_rows` maps a parts-slice
    (start, end) onto the list of CSV-derived rows whose P/R/F1 should fill
    the cells at that slice for every data row of the table."""
    lookups = {sl: _name_to_row(rows) for sl, rows in slice_to_rows.items()}
    valid_leads = set(PROJECT_DISPLAY.values()) | {"Macro average"}
    out_lines = []
    for line in table_path.read_text().splitlines(keepends=True):
        stripped = line.lstrip()
        leading = stripped.split("&", 1)[0].strip()
        if leading not in valid_leads or "\\\\" not in line:
            out_lines.append(line)
            continue
        body, _, tail = line.rpartition("\\\\")
        parts = body.split("&")
        if len(parts) < 10:
            out_lines.append(line)
            continue
        for (start, end), name_to_row in lookups.items():
            row = name_to_row.get(leading)
            if row is None:
                continue
            cells = [_fmt_cell(row["precision"]),
                     _fmt_cell(row["recall"]),
                     _fmt_cell(row["f1"])]
            for i, cell in enumerate(cells):
                parts[start + i] = f" {cell} "
        out_lines.append("&".join(parts) + "\\\\" + tail)
    table_path.write_text("".join(out_lines))
    print(f"Pasted cells into {table_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write-tex", action="store_true",
                    help="In addition to printing paste-ready cells, write "
                         "them into writing/gen/table/rq1-doc-to-*.tex in "
                         "place (SWATTR + LiSSA for d2m, TransArc + LiSSA "
                         "for d2c). AALinker cells are left untouched.")
    args = ap.parse_args()

    # ----- d2m (rq1-doc-to-model.tex) -----
    swattr_rows = [prf_d2m(load_result_sad_sam_standalone, p)
                   for p in D2M_PROJECTS]
    swattr_rows.append(macro_row(swattr_rows))

    lissa_d2m_rows = [prf_d2m(load_lissa_d2m, p) for p in D2M_PROJECTS]
    lissa_d2m_rows.append(macro_row(lissa_d2m_rows))

    # ----- d2c (rq1-doc-to-code.tex) -----
    transarc_rows = [prf_d2c(load_result_sad_code, p) for p in D2M_PROJECTS]
    transarc_rows.append(macro_row(transarc_rows))

    lissa_d2c_rows = [prf_d2c(load_lissa_d2c, p) for p in D2C_PROJECTS]
    lissa_d2c_rows.extend(na_row(p) for p in D2C_MISSING)
    lissa_d2c_rows.append(macro_row(lissa_d2c_rows))

    # ----- LiSSA-only: also emit the full unified metric suite -----
    lissa_d2m_full = [compute_sad_sam_metrics(p, load_lissa_d2m(p))
                      for p in D2M_PROJECTS]
    lissa_d2c_full = ([compute_sad_code_metrics(p, load_lissa_d2c(p))
                       for p in D2C_PROJECTS]
                      + [{**{k: NA for k in SCHEMA}, "project": p}
                         for p in D2C_MISSING])

    # ----- CSV outputs -----
    write_rq1_csv(REPORTS / "swattr_rq1_d2m.csv",    swattr_rows)
    write_rq1_csv(REPORTS / "lissa_rq1_d2m.csv",     lissa_d2m_rows)
    write_rq1_csv(REPORTS / "transarc_rq1_d2c.csv",  transarc_rows)
    write_rq1_csv(REPORTS / "lissa_rq1_d2c.csv",     lissa_d2c_rows)
    write_full_csv(REPORTS / "lissa_metrics_sad-sam.csv",  lissa_d2m_full)
    write_full_csv(REPORTS / "lissa_metrics_sad-code.csv", lissa_d2c_full)

    # ----- Paste-ready printout -----
    print_block("SWATTR (TransArc sad-sam standalone)",
                "writing/gen/table/rq1-doc-to-model.tex",
                "SWATTR column — parts[1:4] after splitting on '&'",
                swattr_rows)
    print_block(f"LiSSA d2m ({LISSA_MODEL})",
                "writing/gen/table/rq1-doc-to-model.tex",
                "LiSSA column — parts[4:7] after splitting on '&'",
                lissa_d2m_rows)
    print_block("TransArc (sad-code final)",
                "writing/gen/table/rq1-doc-to-code.tex",
                "TransArc column — parts[1:4] after splitting on '&'",
                transarc_rows)
    print_block(f"LiSSA d2c ({LISSA_MODEL})",
                "writing/gen/table/rq1-doc-to-code.tex",
                "LiSSA column — parts[4:7] after splitting on '&'",
                lissa_d2c_rows)

    print("\nCSVs written:")
    for name in ("swattr_rq1_d2m", "lissa_rq1_d2m",
                 "transarc_rq1_d2c", "lissa_rq1_d2c",
                 "lissa_metrics_sad-sam", "lissa_metrics_sad-code"):
        print(f"  {REPORTS / (name + '.csv')}")

    if args.write_tex:
        print()
        paste_systems(TABLES / "rq1-doc-to-model.tex", {
            (1, 4): swattr_rows,    # SWATTR slice
            (4, 7): lissa_d2m_rows,  # LiSSA slice
        })
        paste_systems(TABLES / "rq1-doc-to-code.tex", {
            (1, 4): transarc_rows,   # TransArc slice
            (4, 7): lissa_d2c_rows,  # LiSSA slice
        })


if __name__ == "__main__":
    main()
