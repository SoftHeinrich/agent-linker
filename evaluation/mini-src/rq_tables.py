#!/usr/bin/env python3
"""Consolidate the canonical RQ CSVs into per-table "this is the table" CSVs.

The numbers behind the paper's research questions are produced by several
engines (``rq12.py`` for RQ1/RQ2, ``rq34.py`` + ``rq34_rq2.py`` + ``rq4_floor.py`` for RQ3/RQ4)
and land in wide, machine-oriented CSVs. This driver is the *reshape* layer: it
selects the exact rows/columns each paper float needs and writes one small,
human-readable CSV per table under ``reports/tex_src/``. ``csv_to_tex.py`` then
renders each of those into a booktabs ``.tex`` table — so the CSV is reviewable
on its own and the TeX step stays dumb.

It performs NO metric computation; every cell is copied from an upstream CSV.
Run the upstream generators first (see HOWTO-REGENERATE-RQ.md):

    python3 mini-src/rq12.py            # RQ12_BIGTABLE.csv, RQ12_PERPROJECT.csv
    python3 mini-src/rq34.py            # rq3_validators.csv, rq4_variants.csv, rq4_linkers.csv, runs_summary
    python3 mini-src/rq34_rq2.py        # rq34_rq2_linkers.csv (+ _perproject); FULL slots
    python3 mini-src/rq4_floor.py       # rq4_floor.csv (RQ4's one-call floor)
    #   + the two no-knowledge rq34_rq2 runs (see HOWTO §4) for the RQ4 "No knowledge" row

Outputs (reports/tex_src/):
    rq1.csv  rq2.csv  rq3.csv  rq4.csv             -- the BODY tables (body backend; rq3 = mean of 3 runs)
    rq3_runs.csv                   -- RQ3 appendix: both backends, each run + avg in ONE table
    bigtable_rq12_perproject.csv   -- RQ1+RQ2 appendix: per-project + Average row, both backends
    bigtable_rq12_perrun.csv       -- RQ1+RQ2 appendix: per-run + avg (approach), both backends
    bigtable_rq4_perproject.csv    -- RQ4 appendix: per-project + Average row, both backends
    rq4_run{1,2,3}.csv  rq4_runavg.csv  -- RQ4 appendix: four per-run aggregate tables (both backends)
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import metrics as m   # same directory: the shared core (project list, CSV writer)

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent                                  # .../evaluation
REPORTS = EVAL / "reports"
# Every input below is arm-scoped, so a candidate arm is reshaped by setting one knob
# instead of editing four paths. $ALINKER_ARM selects the arm (default below is the arm
# the paper reports; check.py asserts every generator declares the same DEFAULT_ARM).
# $RQ34_REPORTS still names the RQ3/RQ4 directory outright, for one named off-pattern.
DEFAULT_ARM = "s126"
ARM = os.environ.get("ALINKER_ARM", DEFAULT_ARM)
ARM_SUFFIX = "" if ARM == DEFAULT_ARM else f"_{ARM}"   # matches rq12.py's output naming

RQ34 = m.RQ34_REPORTS / os.environ.get("RQ34_REPORTS", ARM)
RQ34_FLOOR = m.RQ34_REPORTS / f"{ARM}_floor"        # rq4_floor.py's output
RQ34_NOKNOW = {                                     # backend -> no-knowledge rq34_rq2 report dir
    "terra": m.RQ34_REPORTS / f"{ARM}_noknow",
    "luna": m.RQ34_REPORTS / f"{ARM}_noknow_luna",
}
# rq12.py writes the incumbent arm to the unsuffixed name and any candidate beside it.
RQ12_BIGTABLE = REPORTS / f"RQ12_BIGTABLE{ARM_SUFFIX}.csv"
RQ12_PERPROJECT = REPORTS / f"RQ12_PERPROJECT{ARM_SUFFIX}.csv"
TEX_SRC = REPORTS / f"tex_src{ARM_SUFFIX}"    # csv_to_tex.py derives the same path

PROJECTS = m.PROJECTS

# The reported arm: body backend first, mirror second (rq34.py's BACKENDS for this arm).
BODY_BACKEND = "terra"
BACKENDS = ["terra", "luna"]
BODY_SYSTEM = "approach (GPT-5.6-terra)"
MIRROR_SYSTEM = "approach (GPT-5.6-luna)"
# ArTEMiS on the body backend is the baseline the body tables compare against; the
# released GPT-5.4 arm stays in the appendix big tables (see BIG_SYSTEMS).
BASELINE_SYSTEM = "Artemis (GPT-5.6-terra)"
BASELINE_RELEASED = "Artemis (GPT-5.4)"
# The re-run baseline is stochastic exactly like \approach, so it is scored the same way:
# three runs, and the tables read their mean. The released GPT-5.4 arm is a single
# recorded run ("single") -- there is no second one to average.
BASELINE_RUN = "average"

# The judges this arm records, in pipeline order. Keys match rq34.py's PHASES, which
# is per-arm: every arm through s110 has one judge per reference form, and s120 unions
# the two name judges, so RQ3 has two rows there and RQ4 still has three.
JUDGE_SETS = {
    "s110": ["full_name", "partial_name", "coref"],
    "s120": ["name", "coref"],
    "s126": ["name", "coref"],
}
JUDGES = JUDGE_SETS.get(ARM, JUDGE_SETS["s110"])
# The linkers RQ4 prices -- one row per judge, the same list as JUDGES. s120/s126 once
# split their single name phase into a full-name and a partial-name row by reading the
# stage label each link carries; that split was retired on 2026-09-19 (the arm ships one
# name linker, so a standalone partial-name row prices a component that does not exist).
FORM_KEY_SETS = {
    "s110": ["full_name", "partial_name", "coref"],
    "s120": ["name", "coref"],
    "s126": ["name", "coref"],
}
FORM_KEYS = FORM_KEY_SETS.get(ARM, FORM_KEY_SETS["s110"])   # rq34.py's FORMS keys
# rq4_linkers.csv's linker column, and the rq34_rq2 doc-code set names (``<label>Only``).
LINKER_LABELS = [k.title().replace("_", "") for k in FORM_KEYS]
RQ4_VARIANTS = ["Full"] + LINKER_LABELS + ["No knowledge"]

# The no-knowledge sweep is measured per backend and lands in its own report dir.
# A backend whose slot is absent has its "No knowledge" row dropped rather than
# filled from another arm -- and the absence is printed, so a missing row is never
# mistaken for a measured zero.
def noknow_available(backend):
    return (RQ34_NOKNOW[backend] / "rq4_variants.csv").is_file()

# Whole doc-to-code suite, in display order (matches rq34_rq2 PANEL / RQ12 columns).
DC_SUITE = ["file_precision", "file_recall", "file_f1", "file_f2",
            "component_micro_f1", "component_micro_f2",
            "worst_component_f1", "worst_component_f2",
            "harmonic_component_f1", "harmonic_component_f2"]


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def read_csv(path: Path):
    if not path.exists():
        raise SystemExit(f"[rq_tables] missing required input CSV: {path}\n"
                         f"  run the upstream generator first (see HOWTO-REGENERATE-RQ.md).")
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def index(rows, *keys):
    return {tuple(r[k] for k in keys): r for r in rows}


def write_csv(name, fieldnames, rows):
    path = TEX_SRC / name
    m.write_dict_csv(path, fieldnames, rows)         # the tree's one dict-row writer
    print(f"[rq_tables] wrote {path}")


def i(v):
    """Round a possibly-fractional count string to an integer for display."""
    return str(round(float(v))) if v not in ("", None) else ""


# --------------------------------------------------------------------------- #
# RQ1 / RQ2 body tables (body backend)
# --------------------------------------------------------------------------- #
def build_rq1(big):
    """One row per display system; SWATTR/TransArC split the bundled TransArc row."""
    ap = big[(BODY_SYSTEM, "average")]
    ar = big[(BASELINE_SYSTEM, BASELINE_RUN)]
    tx = big[("TransArC", "single")]
    cols = ["dm_p", "dm_r", "dm_f1", "dm_f2", "dc_p", "dc_r", "dc_f1", "dc_f2"]

    def row(label, src, dm=True, dc=True):
        return {
            "system": label,
            "dm_p": src["doc_to_model_link_precision"] if dm else "",
            "dm_r": src["doc_to_model_link_recall"] if dm else "",
            "dm_f1": src["doc_to_model_link_f1"] if dm else "",
            "dm_f2": src["doc_to_model_link_f2"] if dm else "",
            "dc_p": src["doc_to_code_file_precision"] if dc else "",
            "dc_r": src["doc_to_code_file_recall"] if dc else "",
            "dc_f1": src["doc_to_code_file_f1"] if dc else "",
            "dc_f2": src["doc_to_code_file_f2"] if dc else "",
        }

    rows = [
        row("approach", ap),
        row("Artemis", ar),
        row("SWATTR", tx, dm=True, dc=False),       # TransArc's deterministic doc-to-model stage
        row("TransArC", tx, dm=False, dc=True),     # TransArc proper = doc-to-code only
    ]
    write_csv("rq1.csv", ["system"] + cols, rows)


RQ2_PANEL_SYSTEMS = [  # (row label, per-project system name, (system, run) average key)
    ("approach", BODY_SYSTEM, (BODY_SYSTEM, "average")),
    ("Artemis", BASELINE_SYSTEM, (BASELINE_SYSTEM, BASELINE_RUN)),
    # The bundled pipeline row: SWATTR supplies the doc-model half, TransArc proper the
    # doc-code one. Unlike the earlier macro RQ2 shape, it is NOT split into two rows --
    # a per-project panel has one row per system, and each half of this row is scored by
    # the stage that produces it. The table's legend names both stages.
    ("TransArC", "TransArC", ("TransArC", "single")),
]
#: display column -> the RQ12 column it copies, in the order each panel prints them.
RQ2_PANEL_METRIC_OF = {
    "dm_link_f1": "doc_to_model_link_f1",
    "dm_link_f2": "doc_to_model_link_f2",
    "dm_cmr": "doc_to_model_component_miss_rate",
    "dc_file_f1": "doc_to_code_file_f1",
    "dc_file_f2": "doc_to_code_file_f2",
    "dc_worst_f1": "doc_to_code_worst_component_f1",
    "dc_worst_f2": "doc_to_code_worst_component_f2",
    "dc_harm_f1": "doc_to_code_harmonic_component_f1",
    "dc_harm_f2": "doc_to_code_harmonic_component_f2",
}


def build_rq2(big):
    """RQ2 size-aware suite per project, as the two panels the float prints side by side.

    One row per (project pair, system): the size-aware suite of both tasks for the
    left panel's project, then the same suite for the right panel's project. The
    projects are the five benchmark projects plus the five-project macro average,
    split in halves so both panels carry the same number of project blocks; the
    pairing is written here, rather than in the renderer, so the CSV stays row-for-row
    what the table prints. Body backend.
    """
    per_project = index(read_csv(RQ12_PERPROJECT), "system", "project")
    columns = [*PROJECTS, "Average"]
    if len(columns) % 2:
        raise SystemExit(f"[rq_tables] RQ2 panels need an even number of columns, "
                         f"got {len(columns)}: {columns}")
    half = len(columns) // 2
    pairs = list(zip(columns[:half], columns[half:]))

    def values(system, average_key, project):
        src = big[average_key] if project == "Average" else per_project[(system, project)]
        return {short: src[column] for short, column in RQ2_PANEL_METRIC_OF.items()}

    rows = []
    for left, right in pairs:
        for label, system, average_key in RQ2_PANEL_SYSTEMS:
            row = {"left_project": left, "system": label, "right_project": right}
            for side, project in (("left", left), ("right", right)):
                for short, value in values(system, average_key, project).items():
                    row[f"{side}_{short}"] = value
            rows.append(row)

    shorts = list(RQ2_PANEL_METRIC_OF)
    write_csv("rq2.csv",
              ["left_project", "system"] + [f"left_{s}" for s in shorts]
              + ["right_project"] + [f"right_{s}" for s in shorts],
              rows)


def build_rq1_transposed(big):
    """Expanded body RQ1 table, transposed for project-wise comparison."""
    per_project = index(read_csv(RQ12_PERPROJECT), "system", "project")
    systems = [
        ("approach", BODY_SYSTEM, (BODY_SYSTEM, "average")),
        ("Artemis", BASELINE_SYSTEM, (BASELINE_SYSTEM, BASELINE_RUN)),
        ("pipeline", "TransArC", ("TransArC", "single")),
    ]
    task_columns = {
        "DM": ("doc_to_model_link_precision", "doc_to_model_link_recall",
               "doc_to_model_link_f1", "doc_to_model_link_f2"),
        "DC": ("doc_to_code_file_precision", "doc_to_code_file_recall",
               "doc_to_code_file_f1", "doc_to_code_file_f2"),
    }
    rows = []
    for project in [*PROJECTS, "Average"]:
        for task, columns in task_columns.items():
            row = {"project": project, "task": task}
            for label, source, average_key in systems:
                values = big[average_key] if project == "Average" else per_project[(source, project)]
                for short, column in zip(("p", "r", "f1", "f2"), columns):
                    row[f"{label}_{short}"] = values[column]
            rows.append(row)
    fields = ["project", "task"] + [f"{label}_{short}"
             for label, _, _ in systems for short in ("p", "r", "f1", "f2")]
    write_csv("rq1_transposed.csv", fields, rows)


# --------------------------------------------------------------------------- #
# RQ3 judging layer (one row per configuration): mean over the three runs for the body
# table and the mirror backend, plus a per-run breakdown for the appendix. Both backends.
# --------------------------------------------------------------------------- #
RQ3_RUNS = ["run1", "run2", "run3"]
# One row per judging configuration -- the shipped pipeline first, then each judge
# switched off, then the whole layer off -- carrying the kills/keeps of the judge(s) that
# row is about and the metrics the pipeline actually scores in that configuration.
RQ3_COLS = ["rej_fp", "rej_tp", "keep_tp", "keep_fp",
            "dm_p", "dm_r", "dm_f1", "dm_f2", "cmr",
            "dc_p", "dc_r", "dc_f1", "dc_f2",
            "dc_worst_f1", "dc_worst_f2", "dc_harm_f1", "dc_harm_f2"]
#: output column -> the rq3_validators.csv (audit) column it copies. ``rej_tp`` is the
#: *unique* rejected true positives -- the ones no other linker recovers -- so it is the
#: recall that judge costs outright, not the raw count it dropped.
RQ3_COUNT_OF = {"rej_fp": "rejected_fp", "rej_tp": "unique_rejected_tp",
                "keep_tp": "kept_tp", "keep_fp": "kept_fp"}
RQ3_DM_OF = {"dm_p": "macro_precision", "dm_r": "macro_recall", "dm_f1": "macro_f1",
             "dm_f2": "macro_f2", "cmr": "component_miss_rate"}
#: output column -> the rq34_rq2_variants.csv (doc-code) column it copies. The judging
#: layer is priced on both tasks: it rejects doc-model links, and every doc-code link is
#: composed through one, so a rejection there propagates.
RQ3_DC_OF = {"dc_p": "doc_to_code_file_precision", "dc_r": "doc_to_code_file_recall",
             "dc_f1": "doc_to_code_file_f1", "dc_f2": "doc_to_code_file_f2",
             "dc_worst_f1": "doc_to_code_worst_component_f1",
             "dc_worst_f2": "doc_to_code_worst_component_f2",
             "dc_harm_f1": "doc_to_code_harmonic_component_f1",
             "dc_harm_f2": "doc_to_code_harmonic_component_f2"}
#: The full pipeline: the reference row, every judge on.
RQ3_FULL_ROW = "full_on"
RQ3_ROW_ORDER = [RQ3_FULL_ROW] + JUDGES + ["all_combined"]
# row key -> the rq3_validators.csv row its counts come from. The counts stay per JUDGE:
# each judge-off row prints that judge's own distinct kills and keeps (what it does while
# it is on), the \fullVariant{} row prints the judges together (the union, not the sum --
# two judges can reject the same link), and the all-off row prints the no-judge audit:
# nothing rejected, the whole candidate pool kept. The metrics beside them are per
# CONFIGURATION, which is the point of the table -- what each judging setup costs.
RQ3_AUDIT_ROW = {RQ3_FULL_ROW: "all_combined",
                 "full_name": "full_name", "partial_name": "partial_name",
                 "name": "name", "coref": "coref",
                 "all_combined": "none"}
# row key -> the rq3_variants / rq34_rq2_variants row that scores it. Scoped to THIS
# arm's judges: the map is read by `.values()` in two places, so carrying another arm's
# keys asks rq3_variants.csv for a row it does not have.
_OFF_VARIANT = {RQ3_FULL_ROW: "Full",
                "full_name": "NoFullNameValid", "partial_name": "NoPartialNameValid",
                "name": "NoNameValid",
                "coref": "NoCitation", "all_combined": "NoValidator"}
RQ3_OFF_VARIANT = {key: _OFF_VARIANT[key] for key in RQ3_ROW_ORDER}


def _rq3_rows(audits, variants, dc_variants, extra=None):
    """One row per judging configuration: whose kills it prints, and what it then scores.

    ``audits`` maps rq3_validators.csv row name -> that row; ``variants`` and
    ``dc_variants`` map RQ3 variant name -> its rq3_variants (doc-model) /
    rq34_rq2_variants (doc-code) row. Counts and metrics are read at their own grain:
    the counts are the judge-level audit picked by ``RQ3_AUDIT_ROW`` (each off-row keeps
    the distinct set of that judge, \fullVariant{} the union over the judges, all-off the
    empty judge set), while the metrics are the pipeline's own scores in that
    configuration on both tasks -- not deltas, with the doc-code half carrying its
    size-aware pair (worst and harmonic component) beside the file-level reference.
    ``extra`` prepends fixed columns.
    """
    rows = []
    for j in RQ3_ROW_ORDER:
        a = audits[RQ3_AUDIT_ROW[j]]
        var = variants[RQ3_OFF_VARIANT[j]]
        dc = dc_variants[RQ3_OFF_VARIANT[j]]
        rows.append({**(extra or {}), "judge": j,
                     **{o: a[c] for o, c in RQ3_COUNT_OF.items()},
                     **{o: var[c] for o, c in RQ3_DM_OF.items()},
                     **{o: dc[c] for o, c in RQ3_DC_OF.items()}})
    return rows


def _rq3_sources():
    """The three CSVs an RQ3 row reads: per-judge audit, doc-model, doc-code."""
    return (index(read_csv(RQ34 / "rq3_validators.csv"), "backend", "run", "validator"),
            index(read_csv(RQ34 / "rq3_variants.csv"), "backend", "run", "variant"),
            index(read_csv(RQ34 / "rq34_rq2_variants.csv"), "backend", "run", "variant"))


def _rq3_slice(indexed, keys, backend, run):
    return {k: indexed[(backend, run, k)] for k in keys}


def build_rq3(backend, out):
    """Per-configuration table for one backend, averaged over the three runs (body table)."""
    val, var, dcvar = _rq3_sources()
    audits = {RQ3_AUDIT_ROW[j] for j in RQ3_ROW_ORDER}
    variants = list(RQ3_OFF_VARIANT.values())
    rows = _rq3_rows(_rq3_slice(val, audits, backend, "average"),
                     _rq3_slice(var, variants, backend, "average"),
                     _rq3_slice(dcvar, variants, backend, "average"))
    write_csv(out, ["judge"] + RQ3_COLS, rows)


def build_rq3_runs(out):
    """The same table, both backends, every run plus the average in ONE table."""
    val, var, dcvar = _rq3_sources()
    audits = {RQ3_AUDIT_ROW[j] for j in RQ3_ROW_ORDER}
    variants = list(RQ3_OFF_VARIANT.values())
    rows = []
    for backend in BACKENDS:
        for run in RQ3_RUNS + ["average"]:
            rows += _rq3_rows(_rq3_slice(val, audits, backend, run),
                              _rq3_slice(var, variants, backend, run),
                              _rq3_slice(dcvar, variants, backend, run),
                              extra={"backend": backend, "run": run})
    write_csv(out, ["backend", "run", "judge"] + RQ3_COLS, rows)


# --------------------------------------------------------------------------- #
# RQ4 body table (body backend): the ablation variants on the size-aware suite
# --------------------------------------------------------------------------- #
def _rq4_variant_cells(backend, run, dm_full, dm_noknow, size_link, size_noknow, uniq):
    """Assemble the RQ4 variant rows for one backend and run ('average' or runN):
    Full, then each linker alone, then No knowledge when that slot exists.

    dm_full/dm_noknow: rq4_variants.csv (linker_set -> the DM_SUITE_RQ4 vector) for the
    full / no-knowledge slot.
    size_link: rq34_rq2_linkers.csv rows (linker_set Full + one per linker).
    size_noknow: no-knowledge rq34_rq2_variants.csv 'Full' row (all linkers, knowledge off).
    uniq: rq4_linkers.csv rows (per-linker unique_tps) -- a diagnostic, never displayed,
    so it stays on the run-average slot.
    """
    def panel(src):
        return {f"dc_{c}": src[f"doc_to_code_{c}"] for c in DC_SUITE}

    rows = [{"variant": "Full", **dm_full["full"],
             **panel(size_link[(backend, run, "Full")]), "unique_tps": ""}]
    for label, key in zip(LINKER_LABELS, FORM_KEYS):
        rows.append({"variant": label, **dm_full[f"{key}_only"],
                     **panel(size_link[(backend, run, f"{label}Only")]),
                     "unique_tps": i(uniq[(backend, "average", label)]["unique_tps"])})
    if dm_noknow and size_noknow:
        rows.append({"variant": "No knowledge", **dm_noknow["full"],
                     **panel(size_noknow), "unique_tps": ""})
    return rows


#: The doc-model columns the RQ4 tables carry, as (output name, rq4_variants.csv column).
#: ``macro_`` names the five-project mean, matching the RQ3 table's vocabulary.
DM_SUITE_RQ4 = [("doc_to_model_macro_precision", "macro_precision"),
                ("doc_to_model_macro_recall", "macro_recall"),
                ("doc_to_model_macro_f1", "macro_f1"),
                ("doc_to_model_macro_f2", "macro_f2"),
                ("doc_to_model_component_miss_rate", "component_miss_rate")]
DM_RQ4_FIELDS = [name for name, _ in DM_SUITE_RQ4]


def _dm_vector(row):
    return {name: row[col] for name, col in DM_SUITE_RQ4}


def _load_rq4_sources(backend, run="average"):
    dm_full = {r["linker_set"]: _dm_vector(r)
               for r in read_csv(RQ34 / "rq4_variants.csv")
               if r["backend"] == backend and r["run"] == run}
    size_link = index(read_csv(RQ34 / "rq34_rq2_linkers.csv"), "backend", "run", "linker_set")
    uniq = index(read_csv(RQ34 / "rq4_linkers.csv"), "backend", "run", "linker")
    if not noknow_available(backend):
        print(f"[rq_tables] NOTE: no no-knowledge run for the reported arm on {backend} "
              f"({RQ34_NOKNOW[backend]}); the RQ4 'No knowledge' row is omitted.")
        return dm_full, {}, size_link, None, uniq
    dm_noknow = {r["linker_set"]: _dm_vector(r)
                 for r in read_csv(RQ34_NOKNOW[backend] / "rq4_variants.csv")
                 if r["backend"] == backend and r["run"] == run}
    size_noknow = index(read_csv(RQ34_NOKNOW[backend] / "rq34_rq2_variants.csv"),
                        "backend", "run", "variant")[(backend, run, "Full")]
    return dm_full, dm_noknow, size_link, size_noknow, uniq


def build_rq4():
    dm_full, dm_noknow, size_link, size_noknow, uniq = _load_rq4_sources(BODY_BACKEND)
    rows = _rq4_variant_cells(BODY_BACKEND, "average", dm_full, dm_noknow, size_link,
                              size_noknow, uniq)
    fields = (["variant"] + DM_RQ4_FIELDS
              + [f"dc_{c}" for c in ("file_precision", "file_recall", "file_f1", "file_f2",
                                     "worst_component_f1", "worst_component_f2",
                                     "harmonic_component_f1", "harmonic_component_f2")]
              + ["unique_tps"])
    # Body table shows only the headline tail metrics (each as \fone + \ftwo).
    rows = [{k: r[k] for k in fields} for r in rows]
    write_csv("rq4.csv", fields, rows)


def floor_available():
    """Has this arm's one-call floor been measured?

    Same rule as the no-knowledge row: a sweep an arm does not have is DROPPED and the
    absence is printed, never filled from another arm. The floor's control is the arm
    itself, so borrowing s110's would compare one arm's workflow against another arm's
    one-call reply.
    """
    return (RQ34_FLOOR / "rq4_floor.csv").is_file()


def build_rq4_floor(backend, out):
    """RQ4's floor for one backend: the workflow against one linking call, per project.

    Per project and not only the average, because the whole point of the row order is
    that the loss is NOT monotone in document length -- teastore (43 sentences) is the
    worst project while teammates (198) is milder, which is what refuses the
    document-length explanation. Sentence counts are in tab:gold_concentration; they
    are deliberately not duplicated here.
    """
    floor = index(read_csv(RQ34_FLOOR / "rq4_floor.csv"), "backend", "run", "arm", "project")
    rows = []
    for project in PROJECTS + ["Average"]:
        full = floor[(backend, "average", "Full", project)]
        one = floor[(backend, "average", "OneCall", project)]
        rows.append({
            "project": project,
            "full_f1": full["f1"], "full_f2": full["f2"],
            "floor_f1": one["f1"], "floor_f2": one["f2"],
            "d_f1": round((float(one["f1"]) - float(full["f1"])) * 100, 1),
        })
    write_csv(out, ["project", "full_f1", "full_f2", "floor_f1", "floor_f2", "d_f1"], rows)


# --------------------------------------------------------------------------- #
# RQ1+RQ2 big tables (whole suite, both backends): average + per-project
# --------------------------------------------------------------------------- #
SUITE_COLS = (["doc_to_model_link_precision", "doc_to_model_link_recall", "doc_to_model_link_f1",
               "doc_to_model_link_f2", "doc_to_model_component_miss_rate"]
              + [f"doc_to_code_{c}" for c in DC_SUITE])

BIG_SYSTEMS = [  # (display label, (system, run) key into RQ12_BIGTABLE)
    (BODY_SYSTEM,          (BODY_SYSTEM, "average")),
    (MIRROR_SYSTEM,        (MIRROR_SYSTEM, "average")),
    (BASELINE_SYSTEM,      (BASELINE_SYSTEM, BASELINE_RUN)),
    (BASELINE_RELEASED,    (BASELINE_RELEASED, "single")),
    ("TransArC",           ("TransArC", "single")),
]


def build_bigtable_rq12_perproject(big):
    """Per-project suite for every system, both backends, with a per-system ``Average``
    summary row carrying the five-project aggregate (the former standalone avg table)."""
    pp = index(read_csv(RQ12_PERPROJECT), "system", "project")
    rows = []
    for label, key in BIG_SYSTEMS:
        for proj in PROJECTS:
            s = pp[(label, proj)]
            rows.append({"system": label, "project": proj, **{c: s[c] for c in SUITE_COLS}})
        avg = big[key]
        rows.append({"system": label, "project": "Average", **{c: avg[c] for c in SUITE_COLS}})
    write_csv("bigtable_rq12_perproject.csv", ["system", "project"] + SUITE_COLS, rows)


# (display label, key, runs) -- \approach and the re-run \Artemis{} baseline are both
# stochastic and run three times; TransArC and the released GPT-5.4 arm are single runs.
PERRUN_SYSTEMS = [
    (BODY_SYSTEM,          BODY_SYSTEM,          ["run1", "run2", "run3", "average"]),
    (MIRROR_SYSTEM,        MIRROR_SYSTEM,        ["run1", "run2", "run3", "average"]),
    (BASELINE_SYSTEM,      BASELINE_SYSTEM,      ["run1", "run2", "run3", "average"]),
    (BASELINE_RELEASED,    BASELINE_RELEASED,    ["single"]),
    ("TransArC",           "TransArC",           ["single"]),
]


def build_bigtable_rq12_perrun(big):
    """Whole suite per run for the stochastic systems (the approach on both backends and
    the re-run \\Artemis{} baseline), each with its mean, plus the single-run baselines. Aggregate over the five projects."""
    rows = []
    for label, sys_key, runs in PERRUN_SYSTEMS:
        for run in runs:
            s = big[(sys_key, run)]
            rows.append({"system": label, "run": run, **{c: s[c] for c in SUITE_COLS}})
    write_csv("bigtable_rq12_perrun.csv", ["system", "run"] + SUITE_COLS, rows)


# --------------------------------------------------------------------------- #
# RQ4 big tables (whole suite, both backends): average + per-project
# --------------------------------------------------------------------------- #
RQ4_DISPLAY = [(v, v) for v in RQ4_VARIANTS]

DM_SUITE = ["link_precision", "link_recall", "link_f1", "link_f2",
            "component_miss_rate"]


def build_bigtable_rq4_perproject():
    """Doc-model link P/R/F1 + doc-code suite per (backend, variant, project), plus a
    per-(backend, variant) ``Average`` summary row. The per-project doc-model link F1
    means reproduce the variant macro F1 exactly, so the Average row's doc-model cells
    are the across-project mean of those P/R/F1; the dc cells come from the run-avg
    aggregate (the former standalone avg table, now folded in here)."""
    link_pp = index(read_csv(RQ34 / "rq34_rq2_linkers_perproject.csv"),
                    "backend", "run", "linker_set", "project")
    dm_pp = index(read_csv(RQ34 / "rq4_variants_perproject.csv"),
                  "backend", "run", "linker_set", "project")
    fields = ["backend", "variant", "project"] + [f"dm_{c}" for c in DM_SUITE] \
        + [f"dc_{c}" for c in DC_SUITE]
    setmap = {"Full": "Full", **{l: f"{l}Only" for l in LINKER_LABELS}}
    dm_setmap = {"Full": "full",
                 **{l: f"{k}_only" for l, k in zip(LINKER_LABELS, FORM_KEYS)}}
    rows = []
    for backend in BACKENDS:
        has_noknow = noknow_available(backend)
        noknow_pp = index(read_csv(RQ34_NOKNOW[backend] / "rq34_rq2_variants_perproject.csv"),
                          "backend", "run", "variant", "project") if has_noknow else {}
        noknow_dm = index(read_csv(RQ34_NOKNOW[backend] / "rq4_variants_perproject.csv"),
                          "backend", "run", "linker_set", "project") if has_noknow else {}
        avg = {r["variant"]: r
               for r in _rq4_variant_cells(backend, "average", *_load_rq4_sources(backend))}
        for variant, _ in RQ4_DISPLAY:
            if variant not in avg:
                continue
            dm_acc = {c: [] for c in DM_SUITE}
            for proj in PROJECTS:
                if variant == "No knowledge":
                    s = noknow_pp[(backend, "average", "Full", proj)]
                    dm = noknow_dm[(backend, "average", "full", proj)]
                else:
                    s = link_pp[(backend, "average", setmap[variant], proj)]
                    dm = dm_pp[(backend, "average", dm_setmap[variant], proj)]
                for c in DM_SUITE:
                    dm_acc[c].append(float(dm[f"doc_to_model_{c}"]))
                rows.append({"backend": backend, "variant": variant, "project": proj,
                             **{f"dm_{c}": dm[f"doc_to_model_{c}"] for c in DM_SUITE},
                             **{f"dc_{c}": s[f"doc_to_code_{c}"] for c in DC_SUITE}})
            a = avg[variant]
            rows.append({"backend": backend, "variant": variant, "project": "Average",
                         **{f"dm_{c}": f"{sum(dm_acc[c]) / len(dm_acc[c]):.6f}" for c in DM_SUITE},
                         **{f"dc_{c}": a[f"dc_{c}"] for c in DC_SUITE}})
    write_csv("bigtable_rq4_perproject.csv", fields, rows)


# RQ4 per-run aggregate: one CSV per run (+ the mean), each = both backends x four variants.
RQ4_PERRUN = [("run1", "rq4_run1.csv"), ("run2", "rq4_run2.csv"),
              ("run3", "rq4_run3.csv"), ("average", "rq4_runavg.csv")]
RQ4_RUN_DC = ["file_precision", "file_recall", "file_f1", "file_f2",
              "worst_component_f1", "worst_component_f2",
              "harmonic_component_f1", "harmonic_component_f2"]


def build_rq4_perrun():
    fields = (["backend", "variant"] + DM_RQ4_FIELDS
              + [f"dc_{c}" for c in RQ4_RUN_DC])
    for run, out in RQ4_PERRUN:
        rows = []
        for backend in BACKENDS:
            cells = _rq4_variant_cells(backend, run, *_load_rq4_sources(backend, run))
            for r in cells:
                rows.append({"backend": backend, "variant": r["variant"],
                             **{f: r[f] for f in DM_RQ4_FIELDS},
                             **{f"dc_{c}": r[f"dc_{c}"] for c in RQ4_RUN_DC}})
        write_csv(out, fields, rows)


# --------------------------------------------------------------------------- #
def main():
    big = index(read_csv(RQ12_BIGTABLE), "system", "run")
    build_rq1(big)
    build_rq2(big)
    build_rq1_transposed(big)
    if floor_available():
        build_rq4_floor(BODY_BACKEND, "rq4_floor.csv")
    else:
        # Remove a previous arm's rendering as well as skipping this one: a table left
        # behind in the shared output directory is indistinguishable from a current one,
        # and csv_to_tex.py would render it under this arm's name.
        stale = TEX_SRC / "rq4_floor.csv"
        note = "" if not stale.exists() else " (a previous arm's copy removed)"
        stale.unlink(missing_ok=True)
        print(f"[rq_tables] no one-call floor for arm {ARM} "
              f"({RQ34_FLOOR / 'rq4_floor.csv'} absent): rq4_floor.csv not written"
              + note, file=sys.stderr)
    build_rq3(BODY_BACKEND, "rq3.csv")             # body confusion (body backend, mean of 3)
    build_rq3_runs("rq3_runs.csv")                  # appendix: both backends, each run + avg in one table
    build_rq4()
    build_bigtable_rq12_perproject(big)             # RQ1/RQ2 per-project + Average (both backends)
    build_bigtable_rq12_perrun(big)                 # RQ1/RQ2 per-run + avg (both backends)
    build_bigtable_rq4_perproject()                 # RQ4 per-project + Average (both backends)
    build_rq4_perrun()                              # RQ4 four per-run tables (run1/2/3 + avg)
    print(f"\n[rq_tables] table CSVs written under {TEX_SRC}", file=sys.stderr)


if __name__ == "__main__":
    main()
