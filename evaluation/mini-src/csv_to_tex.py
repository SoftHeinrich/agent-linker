#!/usr/bin/env python3
"""Render the per-table CSVs from ``rq_tables.py`` into booktabs ``.tex`` tables.

The "script to tex table" half of the pipeline: a declarative spec registry
(``SPECS``) maps each ``reports/tex_src/*.csv`` to one ``reports/tex/*.tex``
file, choosing which columns to show, their headers and number precision, the
\\multicolumn group bands, the bolding rule, and the caption/label. Output drops
straight into the paper (copy ``reports/tex/*.tex`` into ``working/table`` and
``working/appendix``; see HOWTO-REGENERATE-RQ.md).

stdlib only; deterministic (re-running yields byte-identical .tex).

    python3 mini-src/rq_tables.py        # build the CSVs first
    python3 mini-src/csv_to_tex.py       # -> reports/tex/*.tex
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
# Arm-scoped like rq_tables.py: the incumbent arm keeps the historical directories (the
# paper syncs against them), a candidate arm renders beside it. Keep $ALINKER_ARM's default
# in step with rq12.py / rq_tables.py.
import os

DEFAULT_ARM = "s126"
ARM = os.environ.get("ALINKER_ARM", DEFAULT_ARM)
ARM_SUFFIX = "" if ARM == DEFAULT_ARM else f"_{ARM}"

TEX_SRC = EVAL / "reports" / f"tex_src{ARM_SUFFIX}"
TEX_OUT = EVAL / "reports" / f"tex{ARM_SUFFIX}"


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def fmt(val, kind):
    if val in ("", None):
        return "--"
    if kind == "int":
        return str(round(float(val)))
    if kind == "num":                     # integer if whole, else one decimal (mixed run + averaged counts)
        f = float(val)
        return str(round(f)) if abs(f - round(f)) < 1e-9 else f"{f:.1f}"
    if kind == "f1":                      # one decimal, keep the leading zero (averaged counts)
        return f"{float(val):.1f}"
    if kind == "signed":                  # one decimal WITH its sign (delta columns)
        f = float(val)
        # A delta that rounds to nothing is printed unsigned: `+0.0` reads as a gain
        # too small to show, and the judge whose CMR does not move made no gain.
        return "0.0" if abs(f) < 0.05 else f"{f:+.1f}"
    dp = 3 if kind == "f3" else 2
    s = f"{float(val):.{dp}f}"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s


def _num(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def extrema(rows, cols):
    """{field: best_value} for every field flagged bold max/min (over non-empty cells).

    Compared on the PRINTED value, like ``row_winners``: two systems whose raw values
    differ below the last shown decimal print the same string and are both marked.
    """
    best = {}
    for c in cols:
        for field, kind, mode in bold_specs(c):
            vals = [v for v in (_shown_num(r.get(field, ""), kind) for r in rows)
                    if v is not None]
            if vals:
                best[field] = max(vals) if mode == "max" else min(vals)
    return best


def _shown_num(val, kind):
    """The cell's value as the reader sees it, or None when it prints as ``--``.

    Comparing the rounded number, not the raw one, keeps the bold consistent with
    the printed digits: two systems whose values differ below the last shown
    decimal print the same string and are both marked, instead of one of them
    carrying a lead the table does not display.
    """
    s = fmt(val, kind)
    if s == "--":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def col_fields(col):
    """[(field, kind), ...] in the order that column prints them.

    A ``lines`` entry may carry a third element, its column-wise bold mode; it is not
    part of the (field, kind) shape every caller here compares, so it is dropped.
    """
    if "lines" in col:
        return [pair[:2] for line in col["lines"] for pair in line]
    if "fields" in col:
        kinds = col.get("kinds") or [col["kind"]] * len(col["fields"])
        return list(zip(col["fields"], kinds))
    return [(col["field"], col["kind"])]


def bold_specs(col):
    """[(field, kind, mode), ...] for every field of this column bolded down its column.

    A single-field column says ``"bold": "max"``; a compact cell that prints several
    numbers (``lines``) says it per number, as the third element of the tuple -- so
    folding precision and recall into one cell does not silently drop the column-wise
    bold they had as columns of their own.
    """
    if "lines" in col:
        return [(f, k, mode) for line in col["lines"]
                for f, k, *rest in line for mode in rest]
    if col.get("bold"):
        return [(f, k, col["bold"]) for f, k in col_fields(col)]
    return []


def position_groups(cols, mode="max"):
    """Row-wise comparison groups read off the spec's columns, one per metric.

    In a systems-as-columns table every column prints the same tuple of metrics in
    the same order, so the fields sharing a position are one metric's competitors:
    position 0 is precision against precision, position 1 recall against recall,
    and so on. Deriving the groups from the columns keeps the comparison tied to
    what the table actually prints -- add or drop a system column and the groups
    follow, with no metric or system list restated by hand.
    """
    shapes = [col_fields(c) for c in cols]
    widths = {len(sh) for sh in shapes}
    if len(widths) != 1:
        raise ValueError(f"row_bold by position needs columns of one shape, got {sorted(widths)}")
    groups = []
    for pos in zip(*shapes):                      # one (field, kind) per column
        kinds = {kind for _, kind in pos}
        if len(kinds) != 1:
            raise ValueError(
                f"row_bold by position compares one precision per metric, got {sorted(kinds)}")
        groups.append({"fields": [f for f, _ in pos], "kind": kinds.pop(), "mode": mode})
    return groups


def row_bold_groups(spec, cols):
    """Resolve a spec's ``row_bold`` into comparison groups (None when unset).

    ``"by_position"`` (or ``{"by": "position", "mode": ...}``) derives them from the
    columns; an explicit list of ``{"fields": [...], "kind":, "mode":}`` is taken as
    written, for a table whose competitors are not positionally aligned.
    """
    rb = spec.get("row_bold")
    if not rb:
        return None
    if rb == "by_position":
        return position_groups(cols)
    if isinstance(rb, dict):
        if rb.get("by") != "position":
            raise ValueError(f"unknown row_bold spec: {rb}")
        return position_groups(cols, rb.get("mode", "max"))
    return rb


def row_winners(row, groups):
    """Fields that hold the best value in their row-wise comparison group.

    ``extrema`` bolds down a column (best row per metric); this bolds across a
    row (best system per metric) for tables whose systems are the columns. Every
    field tied on the printed value is returned, so a tie marks both cells.
    """
    winners = set()
    for group in groups:
        mode = group.get("mode", "max")
        kind = group.get("kind", "f3")
        vals = {f: _shown_num(row.get(f, ""), kind) for f in group["fields"]}
        present = [v for v in vals.values() if v is not None]
        if not present:
            continue
        target = max(present) if mode == "max" else min(present)
        winners |= {f for f, v in vals.items() if v is not None and v == target}
    return winners


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render(spec):
    rows = list(csv.DictReader((TEX_SRC / spec["csv"]).open(encoding="utf-8")))
    labels = spec["labels"]
    cols = spec["cols"]
    nlab = len(labels)
    best = {} if spec.get("no_bold") else extrema(rows, cols)
    rb_groups = row_bold_groups(spec, cols)

    if "colspec" in spec:
        colspec = spec["colspec"]
    else:
        colspec = "@{}" + "l" * nlab + " " + "r" * len(cols) + "@{}"

    out = []
    out.append(f"% GENERATED by transarc-emp/mini-src/csv_to_tex.py from reports/tex_src/{spec['csv']}.")
    out.append("% Do not edit by hand: rerun rq_tables.py + csv_to_tex.py, then re-copy into the paper.")
    out.append("\\begin{table*}[t]" if spec.get("star") else "\\begin{table}[t]")
    out.append(f"\\caption{{{spec['caption']}}}")
    out.append(f"\\label{{{spec['label']}}}")
    out.append("\\centering" + spec.get("size", "\\small"))
    if spec.get("colsep"):
        out.append(f"\\setlength{{\\tabcolsep}}{{{spec['colsep']}}}")
    if spec.get("fit"):
        width = "\\textwidth" if spec.get("star") else "\\columnwidth"
        out.append("\\adjustbox{max width=" + width + "}{")
    tabularx = spec.get("tabularx")       # e.g. "\\columnwidth": stretch to that width
    if tabularx:
        out.append(f"\\begin{{tabularx}}{{{tabularx}}}{{{colspec}}}")
    else:
        out.append(f"\\begin{{tabular}}{{{colspec}}}")
    out.append("\\toprule")

    # group band + cmidrules
    if spec.get("groups"):
        cells = [""] * nlab + [f"\\multicolumn{{{span}}}{{c}}{{{lab}}}" for lab, span in spec["groups"]]
        out.append(" & ".join(cells) + " \\\\")
        rules, start = [], nlab + 1
        for lab, span in spec["groups"]:
            if lab:
                rules.append(f"\\cmidrule(lr){{{start}-{start + span - 1}}}")
            start += span
        out.append("".join(rules))

    # column headers
    head = [lab["header"] for lab in labels] + [c["header"] for c in cols]
    out.append(" & ".join(head) + " \\\\")
    if spec.get("subheaders"):
        out.append(" & ".join([""] * nlab + spec["subheaders"]) + r" \\")
    out.append("\\midrule")

    # body. Each label flagged group_by blanks when its value repeats the row above.
    # \addlinespace separates blocks: the tuple of fields in ``block_by`` if given,
    # else the group_by fields. ``summary`` (e.g. project == "Average") bolds that row.
    gb_fields = [lab["field"] for lab in labels if lab.get("group_by")]
    block_fields = spec.get("block_by") or gb_fields
    summary = spec.get("summary")
    prev_block = None
    prev_vals = {}
    for ri, r in enumerate(rows):
        if spec.get("midrule_before_last") and ri == len(rows) - 1:
            out.append("\\midrule")
        block_key = tuple(r[f] for f in block_fields) if block_fields else None
        if block_key is not None and prev_block is not None and block_key != prev_block:
            out.append("\\addlinespace[2pt]")
        prev_block = block_key
        is_summary = summary is not None and r.get(summary["field"]) == summary["value"]
        # Row-wise winners (systems as columns): bold the best cell of each metric
        # in this row. A spec that uses it keeps the summary row's numbers unbolded
        # -- see ``summary_bold_values`` -- so the winner marks stay readable there.
        winners = row_winners(r, rb_groups) if rb_groups else set()
        bold_summary_values = spec.get("summary_bold_values", True)
        # summary_label: force this label to show (bold) on the summary row even when
        # its group_by value repeats — it names the block the summary belongs to.
        summary_label = spec.get("summary_label")
        cells = []
        for lab in labels:
            v = r[lab["field"]]
            shown = lab.get("map", {}).get(v, v)
            repeats = lab.get("group_by") and prev_vals.get(lab["field"]) == v
            if repeats and not (is_summary and lab["field"] == summary_label):
                shown = ""
            if is_summary and shown:
                shown = f"\\textbf{{{shown}}}"
            cells.append(shown)
        def value(field, kind):
            """One number, bolded when it wins its row-wise group or leads its column."""
            shown = fmt(r.get(field, ""), kind)
            if shown == "--":
                return shown
            n = _shown_num(r.get(field, ""), kind)
            wins = field in winners or (field in best and n is not None
                                        and abs(n - best[field]) < 1e-9)
            return f"\\textbf{{{shown}}}" if wins else shown

        for c in cols:
            # Pair F1/F2 in one cell when a table needs many per-project rows.
            if "lines" in c:
                parts = [
                    "/".join(value(field, kind) for field, kind, *_ in line)
                    for line in c["lines"]]
                if c.get("multiline"):
                    linebreak = r"\\[-1pt]"
                    s = f"\\makecell[c]{{{linebreak.join(parts)}}}"
                else:
                    s = c.get("line_separator", " ").join(parts)
                if is_summary and bold_summary_values:
                    s = f"\\textbf{{{s}}}"
            elif "fields" in c:
                kinds = c.get("kinds") or [c["kind"]] * len(c["fields"])
                s = "/".join(value(field, kind)
                             for field, kind in zip(c["fields"], kinds))
                if is_summary and bold_summary_values and s != "--/--":
                    s = f"\\textbf{{{s}}}"
            else:
                # ``value`` already marks a row-winner and a column leader; only the
                # summary row's blanket bold is left to apply here.
                s = value(c["field"], c["kind"])
                if (not s.startswith("\\textbf")
                        and is_summary and bold_summary_values and s != "--"):
                    s = f"\\textbf{{{s}}}"
            cells.append(s)
        out.append(" & ".join(cells) + " \\\\")
        for lab in labels:
            prev_vals[lab["field"]] = r[lab["field"]]

    out.append("\\bottomrule")
    out.append("\\end{tabularx}" if tabularx else "\\end{tabular}")
    if spec.get("fit"):
        out.append("}")
    if spec.get("footnote"):
        out.append(f"\\par\\smallskip\\footnotesize {spec['footnote']}")
    out.append("\\end{table*}" if spec.get("star") else "\\end{table}")

    text = "\n".join(out) + "\n"
    TEX_OUT.mkdir(parents=True, exist_ok=True)
    (TEX_OUT / spec["out"]).write_text(text, encoding="utf-8")
    print(f"[csv2tex] wrote {TEX_OUT / spec['out']}")


# --------------------------------------------------------------------------- #
# Column / label building blocks
# --------------------------------------------------------------------------- #
SYS_MAP = {"approach": "\\approach{}", "Artemis": "\\Artemis{}",
           "SWATTR": "SWATTR$^{\\dagger}$", "TransArC": "\\TransArc{}"}
BIGSYS_MAP = {"approach (GPT-5.6-terra)": "\\approach{} (GPT-5.6-terra)",
              "approach (GPT-5.6-luna)": "\\approach{} (GPT-5.6-luna)",
              "Artemis (GPT-5.6-terra)": "\\Artemis{} (GPT-5.6-terra)",
              "Artemis (GPT-5.4)": "\\Artemis{} (GPT-5.4)", "TransArC": "\\TransArc{}$^{\\dagger}$"}

#: How many judges this arm has, in words -- the `all_combined` row and the RQ3
#: caption both name it, and s120 has two where every arm before it had three.
JUDGE_COUNT_WORD = {"s120": "both", "s126": "both"}.get(ARM, "all three")
JUDGE_MAP = {"full_name": "\\entValidator{}", "partial_name": "\\partValidator{}",
             "name": "\\nameValidator{}",
             "coref": "\\corefValidator{}", "all_combined": JUDGE_COUNT_WORD}
VAR_MAP = {"Full": "Full", "Name": "\\linkerN{} only", "Coref": "\\linkerC{} only",
           "FullName": "\\linkerB{} only", "PartialName": "\\linkerD{} only",
           "No knowledge": "No knowledge"}
#: RQ4's knowledge x judge grid, keyed by whether the judging layer is ON. The grid
#: varies that layer as a whole against the knowledge layer as a whole -- every linker
#: of the full pipeline is in place in all four cells -- so it has exactly two rows.
JUDGES_ON_MAP = {"both": "on", "none": "off"}

BACKEND_MAP = {"terra": "GPT-5.6-terra", "luna": "GPT-5.6-luna"}
RUN_MAP = {"run1": "Run 1", "run2": "Run 2", "run3": "Run 3", "average": "Avg", "single": "--"}

# the curated "whole suite" shown in the big tables (link P/R/F1 + file P/R/F1 + size-aware)
SUITE9 = [
    {"field": "doc_to_model_link_precision", "header": "P", "kind": "f2", "bold": "max"},
    {"field": "doc_to_model_link_recall", "header": "R", "kind": "f2", "bold": "max"},
    {"field": "doc_to_model_link_f1", "header": "\\fone", "kind": "f3", "bold": "max"},
    {"field": "doc_to_model_link_f2", "header": "\\ftwo", "kind": "f3", "bold": "max"},
    {"field": "doc_to_model_component_miss_rate", "header": "CMR", "kind": "f1", "bold": "min"},
    {"field": "doc_to_code_file_precision", "header": "P", "kind": "f2", "bold": "max"},
    {"field": "doc_to_code_file_recall", "header": "R", "kind": "f2", "bold": "max"},
    {"field": "doc_to_code_file_f1", "header": "\\fone", "kind": "f3", "bold": "max"},
    {"field": "doc_to_code_file_f2", "header": "\\ftwo", "kind": "f3", "bold": "max"},
    {"field": "doc_to_code_worst_component_f1", "header": "\\fone", "kind": "f2", "bold": "max"},
    {"field": "doc_to_code_worst_component_f2", "header": "\\ftwo", "kind": "f2", "bold": "max"},
    {"field": "doc_to_code_harmonic_component_f1", "header": "\\fone", "kind": "f2", "bold": "max"},
    {"field": "doc_to_code_harmonic_component_f2", "header": "\\ftwo", "kind": "f2", "bold": "max"},
]
# The size-aware pair gets one band each rather than a single 4-wide band: the
# column headers are then just \fone/\ftwo, exactly as under the P/R bands, and
# the band name says which component statistic they summarise.
SUITE9_GROUPS = [("doc-model (link)", 5), ("doc-code (file)", 4),
                 ("worst comp.", 2), ("harm. comp.", 2)]

# Same suite without the doc-model CMR column: the per-run big table reports CMR in the
# body RQ2 + per-project detailed tables instead, so it is omitted here (co-author design).
SUITE_NOCMR = [c for c in SUITE9 if c["field"] != "doc_to_model_component_miss_rate"]
SUITE_NOCMR_GROUPS = [("doc-model (link)", 4), ("doc-code (file)", 4),
                      ("worst comp.", 2), ("harm. comp.", 2)]


def compact(p, r, f1, f2, kind="f3", mode="max", header="Prec./Rec.; \\fone/\\ftwo"):
    r"""One cell printing ``P/R\,;\,F1/F2``, each number still bolded down its column.

    The table-2 shape: folding four numbers into one cell is what lets a task band fit
    beside its size-aware companions without the table shrinking to illegibility.
    """
    return {"header": header, "line_separator": "\\,;\\,",
            "lines": [[(p, kind, mode), (r, kind, mode)],
                      [(f1, kind, mode), (f2, kind, mode)]]}


def pair(f1, f2, header, kind="f2", mode="max"):
    """One cell printing ``F1/F2`` -- the size-aware bands, now inside the doc-code rule."""
    return {"header": header, "lines": [[(f1, kind, mode), (f2, kind, mode)]]}


# --------------------------------------------------------------------------- #
# Spec registry
# --------------------------------------------------------------------------- #
SPECS = [
    # ---- RQ1 body: primary-backend comparison, transposed by project ----
    {"csv": "rq1_transposed.csv", "out": "rq1-results.tex", "label": "tab:rq1",
     "star": True, "size": "\\scriptsize", "colsep": "3pt", "no_bold": True,
     # systems are the columns here: compare each metric across the row (argmax over
     # the printed value, computed per row at render time), not down the column.
     "row_bold": "by_position", "summary_bold_values": False,
     "summary": {"field": "project", "value": "Average"}, "block_by": ["project"],
     "colspec": "@{}llccc@{}",
     "caption": "RQ1 precision, recall, \\fone, and \\ftwo per project on GPT-5.6-terra. "
                "\\textbf{Bold} marks the best system for that metric in that row; each of "
                "precision, recall, \\fone, and \\ftwo is compared separately, per project "
                "and on the average. "
                "All values use three decimal places. LLM results are means of three runs; "
                "the deterministic SWATTR$\\rightarrow$\\TransArc{} pipeline supplies the corresponding doc-model and doc-code stages.",
     "labels": [{"field": "project", "header": "Project", "group_by": True},
                {"field": "task", "header": "Task", "map": {"DM": "doc-model", "DC": "doc-code"}}],
     "subheaders": ["Prec./Rec.; \\fone/\\ftwo", "Prec./Rec.; \\fone/\\ftwo", "Prec./Rec.; \\fone/\\ftwo"],
     "cols": [
         {"header": "\\approach{}", "line_separator": "\\,;\\,", "lines": [
             [("approach_p", "f3"), ("approach_r", "f3")],
             [("approach_f1", "f3"), ("approach_f2", "f3")]]},
         {"header": "\\Artemis{}", "line_separator": "\\,;\\,", "lines": [
             [("Artemis_p", "f3"), ("Artemis_r", "f3")],
             [("Artemis_f1", "f3"), ("Artemis_f2", "f3")]]},
         {"header": "SWATTR$\\rightarrow$\\TransArc{}", "line_separator": "\\,;\\,", "lines": [
             [("pipeline_p", "f3"), ("pipeline_r", "f3")],
             [("pipeline_f1", "f3"), ("pipeline_f2", "f3")]]},
     ],
     "footnote": "SWATTR is the deterministic doc-model stage of \\TransArc{}; \\TransArc{} has no "
                 "standalone doc-model output."},

    # ---- RQ2 body (size-aware macro suite) ----
    # Plain `c` columns, not tabularx `Z`: `Z` is an equal-width X, which would pad the
    # CMR column out to the width of the four-number reference cells. Natural widths
    # inside \\adjustbox is what makes the compact shape actually compact (as in RQ1).
    {"csv": "rq2.csv", "out": "rq2-results.tex", "label": "tab:rq2", "colsep": "3pt",
     "colspec": "@{}l cc @{\\hskip 0.8em} ccc@{}", "fit": True,
     "caption": "RQ2 size-aware suite, both tasks: reference precision, recall, \\fone\\ and "
                "\\ftwo\\ beside the size-aware metric of each task -- the \\cmrname{} (CMR) on "
                "doc-model, the worst- and harmonic-component \\fone/\\ftwo\\ on doc-code. "
                "Each cell reads Prec./Rec.\\,;\\,\\fone/\\ftwo. "
                "Both \\ac{LLM} systems run on GPT-5.6-terra and are reported as the mean of "
                "three runs; \\TransArc{} is deterministic.",
     "labels": [{"field": "system", "header": "System", "map": SYS_MAP}],
     "groups": [("doc-model", 2), ("doc-code", 3)],
     "cols": [
         compact("dm_link_p", "dm_link_r", "dm_link_f1", "dm_link_f2"),
         {"field": "component_miss_rate", "header": "CMR\\%", "kind": "f1", "bold": "min"},
         compact("dc_file_p", "dc_file_r", "dc_file_f1", "dc_file_f2"),
         pair("worst_component_f1", "worst_component_f2", "worst\\ \\fone/\\ftwo"),
         pair("harmonic_component_f1", "harmonic_component_f2", "harm.\\ \\fone/\\ftwo"),
     ],
     "footnote": "$^{\\dagger}$SWATTR is the deterministic doc-model stage of \\TransArc{}; "
                 "\\TransArc{} has no standalone doc-model output."},

    # ---- RQ3 body confusion matrix (mean of 3 runs) ----
    {"csv": "rq3.csv", "out": "rq3-confusion.tex", "label": "tab:rq3-confusion",
     "size": "\\footnotesize", "colsep": "3pt", "fit": True,
     "colspec": "@{}l cc @{\\hskip 1em} cc @{\\hskip 1em} ccccc@{}",
     "caption": "RQ3 per judge on the GPT-5.6-terra backend, averaged over the three runs: "
                "the links it rejects and keeps, and what the pipeline loses on each "
                "doc-model metric when it is switched off (percentage points, judge off "
                "minus \\fullVariant{}; CMR is a miss rate, so a negative $\\Delta$CMR is the "
                "one delta that favours switching the judge off). REJ-TP counts only true "
                "links no other linker "
                f"recovers; the \\emph{{{JUDGE_COUNT_WORD}}} row is measured on the union, "
                "not summed.",
     "labels": [{"field": "judge", "header": "Judge", "map": JUDGE_MAP}],
     "groups": [("rejects", 2), ("keeps", 2), ("judge off (pp)", 5)],
     "cols": [
         {"field": "rej_fp", "header": "FP", "kind": "f1", "bold": "max"},
         {"field": "rej_tp", "header": "TP", "kind": "f1", "bold": "min"},
         {"field": "keep_tp", "header": "TP", "kind": "f1"},
         {"field": "keep_fp", "header": "FP", "kind": "f1"},
         # Signed throughout: the judges buy precision and F-score at the price of
         # recall, so the block carries both signs and an unsigned +3.2 would read wrong.
         {"field": "d_p", "header": "$\\Delta$P", "kind": "signed"},
         {"field": "d_r", "header": "$\\Delta$R", "kind": "signed"},
         {"field": "d_f1", "header": "$\\Delta$\\fone", "kind": "signed", "bold": "min"},
         {"field": "d_f2", "header": "$\\Delta$\\ftwo", "kind": "signed", "bold": "min"},
         {"field": "d_cmr", "header": "$\\Delta$CMR", "kind": "signed"},
     ]},

    # ---- RQ3 confusion, both backends, each run + the average in ONE table (appendix) ----
    {"csv": "rq3_runs.csv", "out": "rq3-runs.tex", "label": "tab:rq3-runs", "no_bold": True,
     "colspec": "@{}lll cc @{\\hskip 1.4em} cc @{\\hskip 1.4em} ccccc@{}", "fit": True,
     "block_by": ["backend", "run"], "summary": {"field": "run", "value": "average"},
     "caption": "RQ3 per judge, per run and averaged, both backends "
                "(the runs behind body \\autoref{tab:rq3-confusion}).",
     "labels": [{"field": "backend", "header": "Backend", "map": BACKEND_MAP, "group_by": True},
                {"field": "run", "header": "Run", "map": RUN_MAP, "group_by": True},
                {"field": "judge", "header": "Judge", "map": JUDGE_MAP}],
     "groups": [("rejects", 2), ("keeps", 2), ("judge off (pp)", 5)],
     "cols": [
         {"field": "rej_fp", "header": "FP", "kind": "num"},
         {"field": "rej_tp", "header": "TP", "kind": "num"},
         {"field": "keep_tp", "header": "TP", "kind": "num"},
         {"field": "keep_fp", "header": "FP", "kind": "num"},
         {"field": "d_p", "header": "$\\Delta$P", "kind": "signed"},
         {"field": "d_r", "header": "$\\Delta$R", "kind": "signed"},
         {"field": "d_f1", "header": "$\\Delta$\\fone", "kind": "signed"},
         {"field": "d_f2", "header": "$\\Delta$\\ftwo", "kind": "signed"},
         {"field": "d_cmr", "header": "$\\Delta$CMR", "kind": "signed"},
     ]},

    # ---- RQ4 body (was fig:rq4-ablation) ----
    # Same shape as RQ2: compact Prec./Rec.\\,;\\,F1/F2 cells, the CMR inside the doc-model
    # rule and the two component bands inside the doc-code one, on plain `c` columns.
    {"csv": "rq4.csv", "out": "rq4-results.tex", "label": "tab:rq4", "colsep": "3pt",
     "colspec": "@{}l cc @{\\hskip 0.8em} ccc@{}", "fit": True,
     "caption": "RQ4 module ablation on the GPT-5.6-terra backend, mean of three runs: each "
                "variant on the reference suite of both tasks beside that task's size-aware "
                "metric -- the \\cmrname{} (CMR) on doc-model, the worst- and "
                "harmonic-component \\fone/\\ftwo\\ on doc-code. Each cell reads "
                "Prec./Rec.\\,;\\,\\fone/\\ftwo. The linker rows keep one linker and drop the "
                "other; \\emph{No knowledge} keeps both and withholds the knowledge the "
                "pipeline discovers before linking.",
     "labels": [{"field": "variant", "header": "Variant", "map": VAR_MAP}],
     "groups": [("doc-model", 2), ("doc-code", 3)],
     "cols": [
         compact("doc_to_model_macro_precision", "doc_to_model_macro_recall",
                 "doc_to_model_macro_f1", "doc_to_model_macro_f2"),
         {"field": "doc_to_model_component_miss_rate", "header": "CMR\\%", "kind": "f1",
          "bold": "min"},
         compact("dc_file_precision", "dc_file_recall", "dc_file_f1", "dc_file_f2"),
         pair("dc_worst_component_f1", "dc_worst_component_f2", "worst\\ \\fone/\\ftwo"),
         pair("dc_harmonic_component_f1", "dc_harmonic_component_f2", "harm.\\ \\fone/\\ftwo"),
     ]},

    # ---- RQ5 body: the judges crossed with the discovered knowledge ----
    # Both halves come from the same cache: the knowledge-on half re-prices the s126 phase
    # states with judges switched off, the knowledge-off half does the same on the
    # no-knowledge sweep. No new LLM calls, which is why the cross is affordable at all.
    {"csv": "rq5.csv", "out": "rq5-knowledge-judges.tex", "label": "tab:judges-knowledge",
     "colsep": "3pt", "colspec": "@{}l ccc @{\\hskip 0.8em} ccc@{}", "fit": True,
     "caption": "Judging $\\times$ knowledge on the GPT-5.6-terra backend, mean of three runs: "
                "the judging layer switched on and off, each with the discovered knowledge "
                "in place and with it withheld. Every linker is in place in all four cells; "
                "the per-judge and per-linker breakdowns are \\autoref{tab:rq3-confusion} and "
                "\\autoref{tab:rq4}. Cells read Prec./Rec.\\,;\\,\\fone/\\ftwo, CMR is the "
                "\\cmrname{}. Both halves are re-scored from the cached phase states of the "
                "two sweeps, so the grid costs no additional \\ac{LLM} calls.",
     "labels": [{"field": "judges", "header": "Judges", "map": JUDGES_ON_MAP}],
     "groups": [("knowledge on", 3), ("knowledge off", 3)],
     "cols": [
         compact("kn_dm_p", "kn_dm_r", "kn_dm_f1", "kn_dm_f2", header="doc-model"),
         {"field": "kn_cmr", "header": "CMR\\%", "kind": "f1", "bold": "min"},
         compact("kn_dc_p", "kn_dc_r", "kn_dc_f1", "kn_dc_f2", header="doc-code"),
         compact("nk_dm_p", "nk_dm_r", "nk_dm_f1", "nk_dm_f2", header="doc-model"),
         {"field": "nk_cmr", "header": "CMR\\%", "kind": "f1", "bold": "min"},
         compact("nk_dc_p", "nk_dc_r", "nk_dc_f1", "nk_dc_f2", header="doc-code"),
     ]},

    # ---- RQ4 floor: the workflow against one linking call (body backend) ----
    {"csv": "rq4_floor.csv", "out": "rq4-floor.tex", "label": "tab:rq4-floor",
     "colsep": "4pt", "fit": True, "tabularx": "\\columnwidth",
     "colspec": "@{}l ZZ @{\\hskip 1em} ZZ @{\\hskip 1em} Z@{}",
     "summary": {"field": "project", "value": "Average"}, "no_bold": True,
     "caption": "RQ4: \\approach{} against one linking call on the GPT-5.6-terra backend, "
                "mean of three runs. The floor receives the document, the component list and "
                "the discovered alias table and returns the link set directly, carrying "
                "\\approach{}'s rubrics verbatim -- what it removes is the arrangement, not "
                "the guidance. Reported per project because the loss is not monotone in "
                "document length (sentence counts in \\autoref{tab:gold_concentration}).",
     "labels": [{"field": "project", "header": "Project"}],
     "groups": [("\\approach{}", 2), ("one call", 2), ("", 1)],
     "cols": [
         {"field": "full_f1", "header": "\\fone", "kind": "f3"},
         {"field": "full_f2", "header": "\\ftwo", "kind": "f3"},
         {"field": "floor_f1", "header": "\\fone", "kind": "f3"},
         {"field": "floor_f2", "header": "\\ftwo", "kind": "f3"},
         {"field": "d_f1", "header": "$\\Delta$\\fone", "kind": "signed"},
     ]},

    # ---- RQ1+RQ2 big table: per project + per-system Average row, both backends ----
    {"csv": "bigtable_rq12_perproject.csv", "out": "big-table-perproject.tex",
     "label": "tab:detailed-perproject", "star": True, "size": "\\footnotesize", "no_bold": True, "fit": True, "colsep": "3pt",
     "summary": {"field": "project", "value": "Average"},
     "caption": "Full comparison per project, with the five-project Average per system, both backends.",
     "labels": [{"field": "system", "header": "System", "map": BIGSYS_MAP, "group_by": True},
                {"field": "project", "header": "Project"}],
     "groups": SUITE9_GROUPS,
     "cols": SUITE9,
     "footnote": "$^{\\dagger}$The doc-model columns for \\TransArc{} are SWATTR, its deterministic "
                 "doc-model stage (\\TransArc{} has no standalone doc-model system). The size-aware "
                 "(doc-code) suite is the worst- and harmonic-component bands, each as "
                 "\\fone/\\ftwo; the doc-model Component Miss Rate (CMR) sits with the doc-model "
                 "columns."},

    # ---- RQ1+RQ2 big table: per run + the average, both backends (CMR omitted here) ----
    {"csv": "bigtable_rq12_perrun.csv", "out": "big-table-perrun.tex",
     "label": "tab:detailed-perrun", "star": True, "size": "\\footnotesize", "no_bold": True, "fit": True, "colsep": "3pt",
     "summary": {"field": "run", "value": "average"},
     "caption": "Full comparison per run for \\approach{} (stochastic; the baselines are "
                "deterministic, one run), both backends, five-project average.",
     "labels": [{"field": "system", "header": "System", "map": BIGSYS_MAP, "group_by": True},
                {"field": "run", "header": "Run", "map": RUN_MAP}],
     "groups": SUITE_NOCMR_GROUPS,
     "cols": SUITE_NOCMR,
     "footnote": "$^{\\dagger}$The doc-model columns for \\TransArc{} are SWATTR, its deterministic "
                 "doc-model stage (\\TransArc{} has no standalone doc-model system). The size-aware "
                 "columns shown here are the doc-code component tail (worst and harmonic, each as "
                 "\\fone/\\ftwo); CMR is the doc-model member, reported in \\autoref{tab:rq2} and "
                 "\\autoref{tab:detailed-perproject}."},

    # ---- RQ4 big table: per project + per-variant Average row, both backends ----
    {"csv": "bigtable_rq4_perproject.csv", "out": "rq4-bigtable-perproject.tex",
     "label": "tab:rq4-perproject", "star": True, "size": "\\footnotesize", "no_bold": True, "fit": True, "colsep": "3pt",
     "block_by": ["backend", "variant"], "summary": {"field": "project", "value": "Average"},
     "summary_label": "variant",
     "caption": "RQ4 module ablation per project, with the per-variant average, both backends.",
     "labels": [{"field": "backend", "header": "Backend", "map": BACKEND_MAP, "group_by": True},
                {"field": "variant", "header": "Variant", "map": VAR_MAP, "group_by": True},
                {"field": "project", "header": "Project"}],
     "groups": [("doc-model", 5), ("doc-code (file)", 4),
                ("worst comp.", 2), ("harm. comp.", 2)],
     "cols": [
         {"field": "dm_link_precision", "header": "P", "kind": "f2"},
         {"field": "dm_link_recall", "header": "R", "kind": "f2"},
         {"field": "dm_link_f1", "header": "\\fone", "kind": "f3"},
         {"field": "dm_link_f2", "header": "\\ftwo", "kind": "f3"},
         {"field": "dm_component_miss_rate", "header": "CMR\\%", "kind": "f1"},
         {"field": "dc_file_precision", "header": "P", "kind": "f2"},
         {"field": "dc_file_recall", "header": "R", "kind": "f2"},
         {"field": "dc_file_f1", "header": "\\fone", "kind": "f3"},
         {"field": "dc_file_f2", "header": "\\ftwo", "kind": "f3"},
         {"field": "dc_worst_component_f1", "header": "\\fone", "kind": "f2"},
         {"field": "dc_worst_component_f2", "header": "\\ftwo", "kind": "f2"},
         {"field": "dc_harmonic_component_f1", "header": "\\fone", "kind": "f2"},
         {"field": "dc_harmonic_component_f2", "header": "\\ftwo", "kind": "f2"},
     ]},
]


# ---- RQ4 per-run aggregate tables: one per run + the average (both backends x variants) ----
_RQ4_RUN_GROUPS = [("doc-model", 5), ("doc-code (file)", 4),
                   ("worst comp.", 2), ("harm. comp.", 2)]
_RQ4_RUN_COLS = [
    {"field": "doc_to_model_macro_precision", "header": "P", "kind": "f2", "bold": "max"},
    {"field": "doc_to_model_macro_recall", "header": "R", "kind": "f2", "bold": "max"},
    {"field": "doc_to_model_macro_f1", "header": "\\fone", "kind": "f3", "bold": "max"},
    {"field": "doc_to_model_macro_f2", "header": "\\ftwo", "kind": "f3", "bold": "max"},
    {"field": "doc_to_model_component_miss_rate", "header": "CMR\\%", "kind": "f1",
     "bold": "min"},
    {"field": "dc_file_precision", "header": "P", "kind": "f2", "bold": "max"},
    {"field": "dc_file_recall", "header": "R", "kind": "f2", "bold": "max"},
    {"field": "dc_file_f1", "header": "\\fone", "kind": "f3", "bold": "max"},
    {"field": "dc_file_f2", "header": "\\ftwo", "kind": "f3", "bold": "max"},
    {"field": "dc_worst_component_f1", "header": "\\fone", "kind": "f2", "bold": "max"},
    {"field": "dc_worst_component_f2", "header": "\\ftwo", "kind": "f2", "bold": "max"},
    {"field": "dc_harmonic_component_f1", "header": "\\fone", "kind": "f2", "bold": "max"},
    {"field": "dc_harmonic_component_f2", "header": "\\ftwo", "kind": "f2", "bold": "max"},
]
SPECS += [
    {"csv": csv, "out": out, "label": label, "star": True, "caption": caption,
     "labels": [{"field": "backend", "header": "Backend", "map": BACKEND_MAP, "group_by": True},
                {"field": "variant", "header": "Variant", "map": VAR_MAP}],
     "groups": _RQ4_RUN_GROUPS, "cols": _RQ4_RUN_COLS}
    for csv, out, label, caption in [
        ("rq4_run1.csv", "rq4-run1.tex", "tab:rq4-run1", "RQ4 module ablation, run 1, both backends."),
        ("rq4_run2.csv", "rq4-run2.tex", "tab:rq4-run2", "RQ4 module ablation, run 2, both backends."),
        ("rq4_run3.csv", "rq4-run3.tex", "tab:rq4-run3", "RQ4 module ablation, run 3, both backends."),
        ("rq4_runavg.csv", "rq4-runavg.tex", "tab:rq4-runavg",
         "RQ4 module ablation, average of the three runs, both backends."),
    ]
]

def check_specs():
    """Assert every spec's \\multicolumn bands cover exactly its columns.

    A band row that is one span short of the column count is not a TeX error -- it
    renders, shifted, and the wrong header sits over each number. Adding a column
    without widening its band is the easy mistake (it happened when the size-aware
    band split into worst/harmonic), so the registry checks itself on import.
    """
    for spec in SPECS:
        ncol = len(spec["cols"])
        if spec.get("groups"):
            span = sum(n for _, n in spec["groups"])
            assert span == ncol, (
                f"{spec['out']}: bands cover {span} columns but there are {ncol}")
        z = spec.get("colspec", "").count("Z")
        assert not z or z == ncol, (
            f"{spec['out']}: colspec has {z} Z columns but there are {ncol}")
        # Resolve row_bold on import: a by-position spec surfaces a ragged column
        # set here, and an explicit group that names a field the table does not
        # print would bold nothing and fail silently.
        groups = row_bold_groups(spec, spec["cols"])
        if groups:
            shown = {field for c in spec["cols"] for field, _ in col_fields(c)}
            for group in groups:
                missing = [f for f in group["fields"] if f not in shown]
                assert not missing, (
                    f"{spec['out']}: row_bold names unrendered field(s) {missing}")


check_specs()



def main():
    # A table whose source CSV this arm does not have is SKIPPED and reported, not
    # rendered from another arm's data: `rq_tables.py` drops the one-call floor for an
    # arm with no floor sweep, exactly as it drops the no-knowledge row.
    written = skipped = 0
    for spec in SPECS:
        if not (TEX_SRC / spec["csv"]).is_file():
            print(f"[csv2tex] {spec['csv']} absent for arm {ARM}: "
                  f"{spec['out']} not written")
            skipped += 1
            continue
        render(spec)
        written += 1
    print(f"\n[csv2tex] {written} tables written under {TEX_OUT}"
          + (f", {skipped} skipped (no source CSV for arm {ARM})" if skipped else ""))


if __name__ == "__main__":
    main()
