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
    rq1.csv  rq2.csv  rq3.csv  rq4.csv                 -- the four BODY tables (body backend; rq3 = mean of 3 runs)
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
DEFAULT_ARM = "s110"
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

# The judges/linkers this arm records, in pipeline order. Keys match rq34.py PHASES.
JUDGES = ["full_name", "partial_name", "coref"]
LINKER_LABELS = ["FullName", "PartialName", "Coref"]   # rq4_linkers.csv linker column
                                                       # (also the rq34_rq2 doc-code set names)
RQ4_VARIANTS = ["Full", "FullName", "PartialName", "Coref", "No knowledge"]

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


def build_rq2(big):
    """RQ2 size-aware suite clustered by task: doc-model (link \\fone/\\ftwo + CMR)
    and doc-code (file \\fone/\\ftwo + the doc-code size-aware metrics, each also as
    an \\ftwo), body backend."""
    rows = []

    def row(label, s, dm=True, dc=True):
        """One display row. ``dm``/``dc`` blank the task the system does not do.

        The ``TransArC`` entry of RQ12_BIGTABLE bundles two systems -- SWATTR supplies
        its doc-model stage, TransArc proper the doc-code one -- so printing it whole
        would credit TransArc with SWATTR's doc-model \fone/\ftwo and CMR. ``build_rq1``
        splits it for exactly this reason; RQ2 splits it the same way, and results.tex
        attributes the CMR to SWATTR in prose.
        """
        return {"system": label,
                "dm_link_f1": s["doc_to_model_link_f1"] if dm else "",
                "dm_link_f2": s["doc_to_model_link_f2"] if dm else "",
                "component_miss_rate": s["doc_to_model_component_miss_rate"] if dm else "",
                "dc_file_f1": s["doc_to_code_file_f1"] if dc else "",
                "dc_file_f2": s["doc_to_code_file_f2"] if dc else "",
                "worst_component_f1": s["doc_to_code_worst_component_f1"] if dc else "",
                "worst_component_f2": s["doc_to_code_worst_component_f2"] if dc else "",
                "harmonic_component_f1": s["doc_to_code_harmonic_component_f1"] if dc else "",
                "harmonic_component_f2": s["doc_to_code_harmonic_component_f2"] if dc else ""}

    tx = big[("TransArC", "single")]
    rows = [
        row("approach", big[(BODY_SYSTEM, "average")]),
        row("Artemis", big[(BASELINE_SYSTEM, BASELINE_RUN)]),
        row("SWATTR", tx, dm=True, dc=False),       # TransArc's deterministic doc-model stage
        row("TransArC", tx, dm=False, dc=True),     # TransArc proper = doc-code only
    ]
    write_csv("rq2.csv",
              ["system", "dm_link_f1", "dm_link_f2", "component_miss_rate",
               "dc_file_f1", "dc_file_f2",
               "worst_component_f1", "worst_component_f2",
               "harmonic_component_f1", "harmonic_component_f2"],
              rows)


# --------------------------------------------------------------------------- #
# RQ3 confusion matrix (per-judge): mean over the three runs for the body table
# and the mirror backend, plus a per-run breakdown for the appendix. Both backends.
# --------------------------------------------------------------------------- #
RQ3_RUNS = ["run1", "run2", "run3"]
# One row per judge (plus the whole stack); counts then the \fone/\ftwo the pipeline
# loses when that judge is switched off.
RQ3_COLS = ["rej_fp", "rej_tp", "keep_tp", "keep_fp", "d_f1", "d_f2"]
# judge key -> (display key, the rq3_variants row that switches it off)
RQ3_ROW_ORDER = JUDGES + ["all_combined"]
RQ3_OFF_VARIANT = {"full_name": "NoFullNameValid", "partial_name": "NoPartialNameValid",
                   "coref": "NoCitation", "all_combined": "NoValidator"}


def _rq3_rows(audits, variants, extra=None):
    """One row per judge: what it rejected and kept, and what switching it off costs.

    ``audits`` maps judge key -> its rq3_validators row; ``variants`` maps RQ3 variant
    name -> its rq3_variants row. ``rej_tp`` is the *unique* rejected true positives --
    the ones no other linker recovers -- so it is the recall the judge costs outright,
    and the all_combined row is measured on the union, not summed (two judges can reject
    the same link). ``d_f1``/``d_f2`` are percentage points against \fullVariant{}:
    negative means the judge is worth that much. ``extra`` prepends fixed columns.
    """
    full = variants["Full"]
    rows = []
    for j in RQ3_ROW_ORDER:
        a = audits[j]
        off = variants[RQ3_OFF_VARIANT[j]]
        rows.append({**(extra or {}), "judge": j,
                     "rej_fp": a["rejected_fp"], "rej_tp": a["unique_rejected_tp"],
                     "keep_tp": a["kept_tp"], "keep_fp": a["kept_fp"],
                     "d_f1": 100 * (float(off["macro_f1"]) - float(full["macro_f1"])),
                     "d_f2": 100 * (float(off["macro_f2"]) - float(full["macro_f2"]))})
    return rows


def build_rq3(backend, out):
    """Per-judge table for one backend, averaged over the three runs (the body table)."""
    val = index(read_csv(RQ34 / "rq3_validators.csv"), "backend", "run", "validator")
    var = index(read_csv(RQ34 / "rq3_variants.csv"), "backend", "run", "variant")
    rows = _rq3_rows({j: val[(backend, "average", j)] for j in RQ3_ROW_ORDER},
                     {v: var[(backend, "average", v)] for v in ["Full"] + list(RQ3_OFF_VARIANT.values())})
    write_csv(out, ["judge"] + RQ3_COLS, rows)


def build_rq3_runs(out):
    """The same table, both backends, every run plus the average in ONE table."""
    val = index(read_csv(RQ34 / "rq3_validators.csv"), "backend", "run", "validator")
    var = index(read_csv(RQ34 / "rq3_variants.csv"), "backend", "run", "variant")
    rows = []
    for backend in BACKENDS:
        for run in RQ3_RUNS + ["average"]:
            rows += _rq3_rows(
                {j: val[(backend, run, j)] for j in RQ3_ROW_ORDER},
                {v: var[(backend, run, v)] for v in ["Full"] + list(RQ3_OFF_VARIANT.values())},
                extra={"backend": backend, "run": run})
    write_csv(out, ["backend", "run", "judge"] + RQ3_COLS, rows)


# --------------------------------------------------------------------------- #
# RQ4 body table (body backend): the ablation variants on the size-aware suite
# --------------------------------------------------------------------------- #
def _rq4_variant_cells(backend, run, dm_full, dm_noknow, size_link, size_noknow, uniq):
    """Assemble the RQ4 variant rows for one backend and run ('average' or runN):
    Full, then each linker alone, then No knowledge when that slot exists.

    dm_full/dm_noknow: rq4_variants.csv (linker_set->macro_f1) for full / no-knowledge slot.
    size_link: rq34_rq2_linkers.csv rows (linker_set Full + one per linker).
    size_noknow: no-knowledge rq34_rq2_variants.csv 'Full' row (all linkers, knowledge off).
    uniq: rq4_linkers.csv rows (per-linker unique_tps) -- a diagnostic, never displayed,
    so it stays on the run-average slot.
    """
    def panel(src):
        return {f"dc_{c}": src[f"doc_to_code_{c}"] for c in DC_SUITE}

    rows = [{"variant": "Full", "doc_to_model_macro_f1": dm_full["full"][0],
             "doc_to_model_macro_f2": dm_full["full"][1],
             **panel(size_link[(backend, run, "Full")]), "unique_tps": ""}]
    for label, key in zip(LINKER_LABELS, JUDGES):
        rows.append({"variant": label,
                     "doc_to_model_macro_f1": dm_full[f"{key}_only"][0],
                     "doc_to_model_macro_f2": dm_full[f"{key}_only"][1],
                     **panel(size_link[(backend, run, f"{label}Only")]),
                     "unique_tps": i(uniq[(backend, "average", label)]["unique_tps"])})
    if dm_noknow and size_noknow:
        rows.append({"variant": "No knowledge",
                     "doc_to_model_macro_f1": dm_noknow["full"][0],
                     "doc_to_model_macro_f2": dm_noknow["full"][1],
                     **panel(size_noknow), "unique_tps": ""})
    return rows


def _load_rq4_sources(backend, run="average"):
    dm_full = {r["linker_set"]: (r["macro_f1"], r["macro_f2"])
               for r in read_csv(RQ34 / "rq4_variants.csv")
               if r["backend"] == backend and r["run"] == run}
    size_link = index(read_csv(RQ34 / "rq34_rq2_linkers.csv"), "backend", "run", "linker_set")
    uniq = index(read_csv(RQ34 / "rq4_linkers.csv"), "backend", "run", "linker")
    if not noknow_available(backend):
        print(f"[rq_tables] NOTE: no no-knowledge run for the reported arm on {backend} "
              f"({RQ34_NOKNOW[backend]}); the RQ4 'No knowledge' row is omitted.")
        return dm_full, {}, size_link, None, uniq
    dm_noknow = {r["linker_set"]: (r["macro_f1"], r["macro_f2"])
                 for r in read_csv(RQ34_NOKNOW[backend] / "rq4_variants.csv")
                 if r["backend"] == backend and r["run"] == run}
    size_noknow = index(read_csv(RQ34_NOKNOW[backend] / "rq34_rq2_variants.csv"),
                        "backend", "run", "variant")[(backend, run, "Full")]
    return dm_full, dm_noknow, size_link, size_noknow, uniq


def build_rq4():
    dm_full, dm_noknow, size_link, size_noknow, uniq = _load_rq4_sources(BODY_BACKEND)
    rows = _rq4_variant_cells(BODY_BACKEND, "average", dm_full, dm_noknow, size_link,
                              size_noknow, uniq)
    fields = (["variant", "doc_to_model_macro_f1", "doc_to_model_macro_f2"]
              + [f"dc_{c}" for c in ("file_f1", "file_f2",
                                     "worst_component_f1", "worst_component_f2",
                                     "harmonic_component_f1", "harmonic_component_f2")]
              + ["unique_tps"])
    # Body table shows only the headline tail metrics (each as \fone + \ftwo).
    rows = [{k: r[k] for k in fields} for r in rows]
    write_csv("rq4.csv", fields, rows)


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

DM_SUITE = ["link_precision", "link_recall", "link_f1", "link_f2"]


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
    dm_setmap = {"Full": "full", **{l: f"{k}_only" for l, k in zip(LINKER_LABELS, JUDGES)}}
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
    fields = (["backend", "variant", "doc_to_model_macro_f1", "doc_to_model_macro_f2"]
              + [f"dc_{c}" for c in RQ4_RUN_DC])
    for run, out in RQ4_PERRUN:
        rows = []
        for backend in BACKENDS:
            cells = _rq4_variant_cells(backend, run, *_load_rq4_sources(backend, run))
            for r in cells:
                rows.append({"backend": backend, "variant": r["variant"],
                             "doc_to_model_macro_f1": r["doc_to_model_macro_f1"],
                             "doc_to_model_macro_f2": r["doc_to_model_macro_f2"],
                             **{f"dc_{c}": r[f"dc_{c}"] for c in RQ4_RUN_DC}})
        write_csv(out, fields, rows)


# --------------------------------------------------------------------------- #
def main():
    big = index(read_csv(RQ12_BIGTABLE), "system", "run")
    build_rq1(big)
    build_rq2(big)
    build_rq4_floor(BODY_BACKEND, "rq4_floor.csv")
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
