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

# One font size for every generated table, so the paper reads uniformly instead of
# table-by-table (\small here, \footnotesize there). \scriptsize is the size the
# widest tables need anyway: above it the 12-16 column tables overrun the measure and
# `fit` rubber-scales them, which prints a size nobody declared and thins the rules.
# A spec may still override with "size", but prefer moving this constant.
TABLE_SIZE = "\\footnotesize"

# Decimal places for every score cell (the `f2`/`f3` kinds -- precision, recall, F1, F2,
# the component pairs). Counts (`f1`, `num`), percentages (CMR) and pp deltas (`signed`)
# keep their own one-decimal grain: they are not 0..1 scores. Two places is also what
# makes the wide tables fit at TABLE_SIZE without `fit` rubber-scaling them.
SCORE_DP = 2


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
    if kind == "sd_score":
        return f"{float(val):.3f}"
    if kind == "sd":
        f = float(val) * 100
        return f"{f:.1f}" if abs(f) >= 0.05 else "0.0"
    dp = SCORE_DP                     # f2/f3 alike: one precision for every score cell
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
    out.append("\\centering" + spec.get("size", TABLE_SIZE))
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
                def cell_value(field, kind):
                    shown = value(field, kind)
                    if field in c.get("sd_fields", {}):
                        shown += r"$\pm$" + fmt(r[c["sd_fields"][field]], "sd_score")
                    return shown

                parts = [
                    "/".join(cell_value(field, kind) for field, kind, *_ in line)
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
# Paired-panel rendering
# --------------------------------------------------------------------------- #
def _panel_band(cells_per_panel, rule=None, lead="", panel_lead=None):
    """One header band, printed once per panel: the left panel, then the right one.

    The two panels carry the same bands, so each band is built once and emitted
    twice. A spanning cell at the left panel's right edge closes on the vertical
    rule that separates the panels; the right panel's does not. ``rule`` draws the
    \\cmidrule under each cell of the band -- the row that names the metric groups
    draws them, the rows that subdivide a group or head one column do not.
    ``panel_lead`` is the cell over each panel's own project column, which sits left
    of that panel's bands and therefore shifts every span one column right.
    """
    lines, rules, column = [], [], 2
    for panel, last_panel in ((0, False), (1, True)):
        cells = []
        if panel_lead is not None:
            cells.append(panel_lead)
            column += 1
        for i, (label, span) in enumerate(cells_per_panel):
            last_cell = i == len(cells_per_panel) - 1
            # A cell that spans one column is printed as itself: \multicolumn would
            # override the column spec, which is what carries the panel rule and the
            # right alignment of the numbers underneath.
            align = "c|" if last_cell and not last_panel else "c"
            cells.append(f"\\multicolumn{{{span}}}{{{align}}}{{{label}}}" if span > 1 else label)
            if rule:
                trim = "l" if last_cell and last_panel else "lr"
                rules.append(f"\\cmidrule({trim}){{{column}-{column + span - 1}}}")
            column += span
        lines.append(" & ".join(cells))
    out = [f"{lead} & " + lines[0] if lead else "& " + lines[0], "& " + lines[1] + r" \\"]
    if rule:
        out.append("".join(rules))
    return out


def render_panels(spec):
    """Render a two-panel per-project table: one row per system, two projects per row.

    ``render`` prints one metric block per row under a single band row. This layout
    repeats the same metric block twice across the page -- two projects side by side,
    separated by a vertical rule, each panel naming its project in a column of its
    own (abbreviated; the gold table spells the names out) -- so it needs two
    stacked band rows, and is rendered here. The registry entry still carries the csv/out/label
    triple, so ``sync_paper.py`` and ``main`` treat it like any other table.
    """
    rows = list(csv.DictReader((TEX_SRC / spec["csv"]).open(encoding="utf-8")))
    metrics = spec["metrics"]                       # [(column suffix, kind), ...] per panel
    width = len(metrics)
    label = spec["label_column"]
    proj = spec["project_column"]                   # each panel's own leading column

    out = []
    out.append(f"% GENERATED by transarc-emp/mini-src/csv_to_tex.py from reports/tex_src/{spec['csv']}.")
    out.append("% Do not edit by hand: rerun rq_tables.py + csv_to_tex.py, then re-copy into the paper.")
    out.append("\\begin{table}[t]")
    out.append(f"\\caption{{{spec['caption']}}}")
    out.append(f"\\label{{{spec['label']}}}")
    out.append("\\centering" + spec.get("size", TABLE_SIZE))
    out.append(f"\\setlength{{\\tabcolsep}}{{{spec['colsep']}}}")
    out.append(f"\\renewcommand{{\\arraystretch}}{{{spec['arraystretch']}}}")
    out.append(f"\\begin{{tabular*}}{{\\linewidth}}{{@{{}}l@{{\\extracolsep{{\\fill}}}}"
               f"l*{{{width}}}{{r}}|l*{{{width}}}{{r}}@{{}}}}")
    out.append("\\toprule")
    out += _panel_band(spec["groups"], rule=True, panel_lead="")
    out += _panel_band(spec["subgroups"], lead=label["header"], panel_lead=proj["header"])
    out += _panel_band([(header, 1) for _, header in
                        zip(metrics, spec["headers"])], panel_lead="")
    out.append("\\midrule")

    previous = None
    for row in rows:
        block = (row["left_project"], row["right_project"])
        opens = block != previous
        if opens and previous is not None:
            out.append(spec["block_skip"])
        name = row[label["field"]]
        cells = [label.get("map", {}).get(name, name)]
        for side, project in zip(("left", "right"), block):
            # The project names its panel from a column, printed on the block's first
            # row and blank underneath -- the same idiom as the grouped label columns
            # in the other tables. It used to be a row spanning the whole panel.
            cells.append(proj["map"].get(project, project) if opens else "")
            cells += [fmt(row[f"{side}_{column}"], kind) for column, kind in metrics]
        out.append(" & ".join(cells) + r" \\")
        previous = block

    out.append("\\bottomrule")
    out.append("\\end{tabular*}")
    if spec.get("footnote"):
        out.append(f"\\par\\smallskip\\footnotesize {spec['footnote']}")
    out.append("\\end{table}")

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

#: How many judges this arm has, in words -- the RQ3 caption names it, and s120 has two
#: where every arm before it had three.
JUDGE_COUNT_WORD = {"s120": "both", "s126": "both"}.get(ARM, "all three")
#: The RQ3 rows are configurations. The last row has no active judge.
JUDGE_MAP = {"full_on": "Full",
             "full_name": "\\entValidator{}", "partial_name": "\\partValidator{}",
             "name": "\\nameValidator{}",
             "coref": "\\corefValidator{}", "no_judge": "No judge"}
VAR_MAP = {"Full": "Full", "Name": "\\linkerN{} only", "Coref": "\\linkerC{} only",
           "FullName": "\\linkerB{} only", "PartialName": "\\linkerD{} only",
           "No knowledge": "No knowledge"}
#: The RQ2 panel rows. A per-project panel is one row per system, so the bundled
#: pipeline keeps one row and its legend names the stage behind each task half.
RQ2_PANEL_SYS_MAP = {"approach": "AL", "Artemis": "AT", "TransArC": "S/T"}
#: Project identifiers as the benchmark stores them -> as the paper prints them.
PROJECT_MAP = {"mediastore": "MediaStore", "teastore": "TeaStore",
               "teammates": "Teammates", "bigbluebutton": "BigBlueButton",
               "jabref": "JabRef", "Average": "Average"}
#: The short form used where a project has to fit inside a column.
PROJECT_ABBR = {"mediastore": "MS", "teastore": "TS", "teammates": "TM",
                "bigbluebutton": "BBB", "jabref": "JR", "Average": "Avg"}
#: Full name + its short form for tables that have room to show both.
PROJECT_LONG_MAP = {k: (v if k == "Average" else f"{v} ({PROJECT_ABBR[k]})")
                    for k, v in PROJECT_MAP.items()}
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
    # ---- RQ1 body: transposed, two rows per project (DM / DC) ----
    {"csv": "rq1_transposed.csv", "out": "rq1-results.tex", "label": "tab:rq1",
     "star": True, "colsep": "3pt", "no_bold": True,
     "row_bold": [{"fields": [f"{system}_{metric}"
                              for system in ("approach", "Artemis", "pipeline")],
                   "kind": "f2", "mode": "max"}
                  for metric in ("p", "r", "f1", "f2")],
     "summary_bold_values": False,
     "summary": {"field": "project", "value": "Average"},
     "summary_label": "project",
     "block_by": ["project"],
     "colspec": "@{}llccc@{}",
     "caption": "RQ1 link metrics by project on GPT-5.6-terra.",
     "labels": [{"field": "project", "header": "Proj.", "map": PROJECT_ABBR, "group_by": True},
                {"field": "task", "header": "Task"}],
     "cols": [dict(compact(f"{system}_p", f"{system}_r", f"{system}_f1", f"{system}_f2",
                           header=header), multiline=True,
                   sd_fields={f"{system}_{metric}": f"{system}_{metric}_sd" for metric in ("p", "r")}
                             if system != "pipeline" else {})
              for system, header in (("approach", "\\approach{}"), ("Artemis", "\\Artemis{}"),
                                     ("pipeline", "SWATTR / \\TransArc{}"))],
     "footnote": "Cells show P/R; \\fone/\\ftwo. P and R include sample SD across three runs "
                 "on the score scale; Average SD uses the three per-run project means. "
                 "SWATTR supplies deterministic DM results and \\TransArc{} deterministic DC results."},

    {"csv": "inference_cost.csv", "out": "inference-cost.tex", "label": "tab:inference-cost",
     "star": True, "colsep": "4pt", "no_bold": True,
     "caption": "Recorded inference usage per project, averaged over three runs.",
     "labels": [{"field": "project", "header": "Project", "map": PROJECT_ABBR}],
     "groups": [("\\approach{} (GPT-5.6-terra)", 2), ("\\Artemis{} (GPT-5.6-luna)", 2)],
     "cols": [{"field": f"{system}_{metric}", "header": header, "kind": "f1"}
              for system in ("approach", "Artemis")
              for metric, header in (("input_k", "Input (k)"), ("output_k", "Output (k)"))],
     "summary": {"field": "project", "value": "Total"},
     "footnote": "Tokens are in thousands. Total sums project means. "
                 "Models and collection dates differ; these are unpaired usage observations. "
                 "SWATTR and \\TransArc{} consume no LLM tokens."},

    # ---- RQ2 body (size-aware suite, per project) ----
    # Two project panels side by side, one row per system: the per-project shape the
    # macro table could not show, at the same vertical cost as the macro one. Rendered
    # by ``render_panels`` -- the band rows repeat once per panel, which the column
    # registry of ``render`` does not express (the macro RQ2 layout it did express,
    # with a Prec./Rec.; F1/F2 cell per task, was retired for this one on 2026-09-21).
    {"csv": "rq2.csv", "out": "rq2-results.tex", "label": "tab:rq2",
     "render": "panels", "colsep": "1pt", "arraystretch": "0.96",
     "caption": "RQ2 size-aware metrics by project on GPT-5.6-terra.",
     "label_column": {"field": "system", "header": "Approach", "map": RQ2_PANEL_SYS_MAP},
     "project_column": {"header": "Proj.", "map": PROJECT_ABBR},
     "groups": [("doc-model", 3), ("doc-code", 6)],
     "subgroups": [("Link", 2), ("CMR", 1), ("Link", 2), ("Worst", 2), ("Harm.", 2)],
     "headers": ["\\fone", "\\ftwo", "\\%", "\\fone", "\\ftwo",
                 "\\fone", "\\ftwo", "\\fone", "\\ftwo"],
     "metrics": [("dm_link_f1", "f2"), ("dm_link_f2", "f2"), ("dm_cmr", "f1"),
                 ("dc_file_f1", "f2"), ("dc_file_f2", "f2"),
                 ("dc_worst_f1", "f2"), ("dc_worst_f2", "f2"),
                 ("dc_harm_f1", "f2"), ("dc_harm_f2", "f2")],
     "block_skip": "\\addlinespace[1.5pt]",
     "footnote": "AL = \\approach{}; AT = \\Artemis{}; S/T = SWATTR for doc-model and "
                 "\\TransArc{} for doc-code."},

    # ---- RQ3 body: the judging configurations (mean of 3 runs) ----
    # Rows are configurations, not judges, and the metric block prints what each one
    # actually scores on both tasks. Deltas were the earlier shape; they hid the level
    # the pipeline operates at and could not be compared against tab:rq4, whose rows
    # print the same Prec./Rec.\\,;\\,F1/F2 cell (tab:rq2 printed it too until it
    # moved to the per-project panel layout).
    # The counts stay at the JUDGE grain (each off-row is that judge's own distinct set,
    # what it kills and keeps while it is on); the metrics beside them are per
    # configuration. They are NOT bolded: the all-off row rejects nothing by
    # construction, which is not a win.
    {"csv": "rq3.csv", "out": "rq3-confusion.tex", "label": "tab:rq3-confusion",
     # At \footnotesize these 12 columns overran \columnwidth and \adjustbox
     # rubber-scaled the box by 0.94 (8pt printed at 7.5pt, hairlines thinned with it);
     # at TABLE_SIZE it fits natively, so `fit` is only a guard.
     "colsep": "3pt", "fit": True,
     "colspec": "@{}l cc @{\\hskip 1em} cc @{\\hskip 1em} cc @{\\hskip 0.8em} ccc@{}",
     "caption": "RQ3 judging configurations on GPT-5.6-terra, averaged across three runs.",
     "labels": [{"field": "judge", "header": "Judges", "map": JUDGE_MAP}],
     "groups": [("rejects", 2), ("keeps", 2), ("doc-model", 2), ("doc-code", 3)],
     "cols": [
         {"field": "rej_fp", "header": "FP", "kind": "f1"},
         {"field": "rej_tp", "header": "TP", "kind": "f1"},
         {"field": "keep_tp", "header": "TP", "kind": "f1"},
         {"field": "keep_fp", "header": "FP", "kind": "f1"},
         compact("dm_p", "dm_r", "dm_f1", "dm_f2"),
         {"field": "cmr", "header": "CMR\\%", "kind": "f1", "bold": "min"},
         compact("dc_p", "dc_r", "dc_f1", "dc_f2"),
         pair("dc_worst_f1", "dc_worst_f2", "worst\\ \\fone/\\ftwo"),
         pair("dc_harm_f1", "dc_harm_f2", "harm.\\ \\fone/\\ftwo"),
     ]},

    # ---- RQ3 configurations, both backends, each run + the average in ONE table (appendix) ----
    {"csv": "rq3_runs.csv", "out": "rq3-runs.tex", "label": "tab:rq3-runs", "no_bold": True,
     "colspec": "@{}lll cc @{\\hskip 1.4em} cc @{\\hskip 1.4em} cc @{\\hskip 0.8em} ccc@{}",
     "colsep": "3pt", "fit": True,
     "block_by": ["backend", "run"], "summary": {"field": "run", "value": "average"},
     "caption": "RQ3 judging configurations by run and backend.",
     "labels": [{"field": "backend", "header": "Backend", "map": BACKEND_MAP, "group_by": True},
                {"field": "run", "header": "Run", "map": RUN_MAP, "group_by": True},
                {"field": "judge", "header": "Judges", "map": JUDGE_MAP}],
     "groups": [("rejects", 2), ("keeps", 2), ("doc-model", 2), ("doc-code", 3)],
     "cols": [
         {"field": "rej_fp", "header": "FP", "kind": "num"},
         {"field": "rej_tp", "header": "TP", "kind": "num"},
         {"field": "keep_tp", "header": "TP", "kind": "num"},
         {"field": "keep_fp", "header": "FP", "kind": "num"},
         compact("dm_p", "dm_r", "dm_f1", "dm_f2"),
         {"field": "cmr", "header": "CMR\\%", "kind": "f1"},
         compact("dc_p", "dc_r", "dc_f1", "dc_f2"),
         pair("dc_worst_f1", "dc_worst_f2", "worst\\ \\fone/\\ftwo"),
         pair("dc_harm_f1", "dc_harm_f2", "harm.\\ \\fone/\\ftwo"),
     ]},

    # ---- RQ4 body (was fig:rq4-ablation) ----
    # Same shape as RQ2: compact Prec./Rec.\\,;\\,F1/F2 cells, the CMR inside the doc-model
    # rule and the two component bands inside the doc-code one, on plain `c` columns.
    {"csv": "rq4.csv", "out": "rq4-results.tex", "label": "tab:rq4", "colsep": "3pt",
     "colspec": "@{}l cc @{\\hskip 0.8em} ccc@{}", "fit": True,
     "caption": "RQ4 module ablation on GPT-5.6-terra, averaged across three runs.",
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

    # ---- RQ4 floor: the workflow against one linking call (body backend) ----
    {"csv": "rq4_floor.csv", "out": "rq4-floor.tex", "label": "tab:rq4-floor",
     "colsep": "4pt", "fit": True, "tabularx": "\\columnwidth",
     "colspec": "@{}l ZZ @{\\hskip 1em} ZZ @{\\hskip 1em} Z@{}",
     "summary": {"field": "project", "value": "Average"}, "no_bold": True,
     "caption": "RQ4 compares \\approach{} with one linking call on GPT-5.6-terra.",
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
     "label": "tab:detailed-perproject", "star": True, "no_bold": True, "fit": True, "colsep": "3pt",
     "summary": {"field": "project", "value": "Average"},
     "caption": "Detailed per-project comparison across both backends.",
     "labels": [{"field": "system", "header": "System", "map": BIGSYS_MAP, "group_by": True},
                {"field": "project", "header": "Project", "map": PROJECT_ABBR}],
     "groups": SUITE9_GROUPS,
     "cols": SUITE9,
     "footnote": "$^{\\dagger}$The doc-model columns for \\TransArc{} are SWATTR, its deterministic "
                 "doc-model stage (\\TransArc{} has no standalone doc-model system). The size-aware "
                 "(doc-code) suite is the worst- and harmonic-component bands, each as "
                 "\\fone/\\ftwo; the doc-model Component Miss Rate (CMR) sits with the doc-model "
                 "columns."},

    # ---- RQ1+RQ2 big table: per run + the average, both backends (CMR omitted here) ----
    {"csv": "bigtable_rq12_perrun.csv", "out": "big-table-perrun.tex",
     "label": "tab:detailed-perrun", "star": True, "no_bold": True, "fit": True, "colsep": "3pt",
     "summary": {"field": "run", "value": "average"},
     "caption": "Detailed per-run comparison across both backends.",
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
     "label": "tab:rq4-perproject", "star": True, "no_bold": True, "fit": True, "colsep": "3pt",
     "block_by": ["backend", "variant"], "summary": {"field": "project", "value": "Average"},
     "summary_label": "variant",
     "caption": "RQ4 ablation by project across both backends.",
     "labels": [{"field": "backend", "header": "Backend", "map": BACKEND_MAP, "group_by": True},
                {"field": "variant", "header": "Variant", "map": VAR_MAP, "group_by": True},
                {"field": "project", "header": "Project", "map": PROJECT_ABBR}],
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
        if spec.get("render") == "panels":
            # A panel spec has no column registry: its bands repeat per panel, so
            # each band row must cover the metrics of ONE panel exactly.
            nmetric = len(spec["metrics"])
            for key in ("groups", "subgroups"):
                span = sum(n for _, n in spec[key])
                assert span == nmetric, (
                    f"{spec['out']}: {key} cover {span} columns but a panel has {nmetric}")
            assert len(spec["headers"]) == nmetric, (
                f"{spec['out']}: {len(spec['headers'])} headers for {nmetric} panel columns")
            continue
        ncol = len(spec["cols"])
        if spec.get("groups"):
            span = sum(n for _, n in spec["groups"])
            assert span == ncol, (
                f"{spec['out']}: bands cover {span} columns but there are {ncol}")
        if spec.get("subheaders"):
            assert len(spec["subheaders"]) == ncol, (
                f"{spec['out']}: {len(spec['subheaders'])} subheaders for {ncol} columns")
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
        render_panels(spec) if spec.get("render") == "panels" else render(spec)
        written += 1
    print(f"\n[csv2tex] {written} tables written under {TEX_OUT}"
          + (f", {skipped} skipped (no source CSV for arm {ARM})" if skipped else ""))


if __name__ == "__main__":
    main()
