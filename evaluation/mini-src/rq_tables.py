#!/usr/bin/env python3
"""Consolidate the canonical RQ CSVs into per-table "this is the table" CSVs.

The numbers behind the paper's research questions are produced by several
engines (``rq12.py`` for RQ1/RQ2, ``rq34.py`` + ``rq34_rq2.py`` for RQ3/RQ4)
and land in wide, machine-oriented CSVs. This driver is the *reshape* layer: it
selects the exact rows/columns each paper float needs and writes one small,
human-readable CSV per table under ``reports/tex_src/``. ``csv_to_tex.py`` then
renders each of those into a booktabs ``.tex`` table — so the CSV is reviewable
on its own and the TeX step stays dumb.

It performs NO metric computation; every cell is copied from an upstream CSV.
Run the upstream generators first (see HOWTO-REGENERATE-RQ.md):

    python3 mini-src/rq12.py            # RQ12_BIGTABLE.csv, RQ12_PERPROJECT.csv, RQ2_PANEL.csv
    python3 mini-rq34/rq34.py           # rq3_validators.csv, rq4_variants.csv, rq4_linkers.csv, runs_summary
    python3 mini-rq34/rq34_rq2.py       # rq34_rq2_linkers.csv (+ _perproject); FULL slots
    #   + the two no-knowledge rq34_rq2 runs (see HOWTO §4) for the RQ4 "No knowledge" row

Outputs (reports/tex_src/):
    rq1.csv  rq2.csv  rq3.csv  rq4.csv                 -- the four BODY tables (GPT-5.4; rq3 = mean of 3 runs)
    rq3_runs.csv                   -- RQ3 appendix: both backends, each run + avg in ONE table
    bigtable_rq12_perproject.csv   -- RQ1+RQ2 appendix: per-project + Average row, both backends
    bigtable_rq12_perrun.csv       -- RQ1+RQ2 appendix: per-run + avg (approach), both backends
    bigtable_rq4_perproject.csv    -- RQ4 appendix: per-project + Average row, both backends
    rq4_run{1,2,3}.csv  rq4_runavg.csv  -- RQ4 appendix: four per-run aggregate tables (both backends)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent                                  # .../transarc-emp
REPORTS = EVAL / "reports"
RQ34 = EVAL / "mini-rq34" / "reports"
RQ34_NOKNOW = {                                     # backend -> no-knowledge rq34_rq2 report dir
    "openai": EVAL / "mini-rq34" / "reports_s21_noknow",
    "claude": EVAL / "mini-rq34" / "reports_s21_noknow_sonnet",
}
TEX_SRC = REPORTS / "tex_src"

PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]

# Whole doc-to-code suite, in display order (matches rq34_rq2 PANEL / RQ12 columns).
DC_SUITE = ["file_precision", "file_recall", "file_f1", "component_micro_f1",
            "worst_component_f1", "harmonic_component_f1", "sentence_coverage", "noise_rate"]


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
    TEX_SRC.mkdir(parents=True, exist_ok=True)
    path = TEX_SRC / name
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[rq_tables] wrote {path}")


def i(v):
    """Round a possibly-fractional count string to an integer for display."""
    return str(round(float(v))) if v not in ("", None) else ""


# --------------------------------------------------------------------------- #
# RQ1 / RQ2 body tables (GPT-5.4)
# --------------------------------------------------------------------------- #
def build_rq1(big):
    """One row per display system; SWATTR/TransArC split the bundled TransArc row."""
    ap = big[("approach (GPT-5.4)", "average")]
    ar = big[("Artemis (GPT-5.4)", "single")]
    tx = big[("TransArC", "single")]
    cols = ["dm_p", "dm_r", "dm_f1", "dc_p", "dc_r", "dc_f1"]

    def row(label, src, dm=True, dc=True):
        return {
            "system": label,
            "dm_p": src["doc_to_model_link_precision"] if dm else "",
            "dm_r": src["doc_to_model_link_recall"] if dm else "",
            "dm_f1": src["doc_to_model_link_f1"] if dm else "",
            "dc_p": src["doc_to_code_file_precision"] if dc else "",
            "dc_r": src["doc_to_code_file_recall"] if dc else "",
            "dc_f1": src["doc_to_code_file_f1"] if dc else "",
        }

    rows = [
        row("approach", ap),
        row("Artemis", ar),
        row("SWATTR", tx, dm=True, dc=False),       # TransArc's deterministic doc-to-model stage
        row("TransArC", tx, dm=False, dc=True),     # TransArc proper = doc-to-code only
    ]
    write_csv("rq1.csv", ["system"] + cols, rows)


def build_rq2(big):
    """RQ2 size-aware suite clustered by task: doc-model (link \\fone + sent cov + SFM) and
    doc-code (file \\fone + the doc-code size-aware metrics), GPT-5.4."""
    rows = []
    for label, key in (("approach", ("approach (GPT-5.4)", "average")),
                       ("Artemis", ("Artemis (GPT-5.4)", "single")),
                       ("TransArC", ("TransArC", "single"))):
        s = big[key]
        rows.append({"system": label,
                     "dm_link_f1": s["doc_to_model_link_f1"],
                     "dm_sentence_coverage": s["doc_to_model_sentence_coverage"],
                     "silent_failure_mass": s["doc_to_model_silent_failure_mass"],
                     "dc_file_f1": s["doc_to_code_file_f1"],
                     "dc_sentence_coverage": s["doc_to_code_sentence_coverage"],
                     "worst_component_f1": s["doc_to_code_worst_component_f1"],
                     "harmonic_component_f1": s["doc_to_code_harmonic_component_f1"]})
    write_csv("rq2.csv",
              ["system", "dm_link_f1", "dm_sentence_coverage", "silent_failure_mass",
               "dc_file_f1", "dc_sentence_coverage",
               "worst_component_f1", "harmonic_component_f1"],
              rows)


# --------------------------------------------------------------------------- #
# RQ3 confusion matrix (per-judge): mean over the three runs for the body table
# and the Claude mirror, plus a per-run breakdown for the appendix. Both backends.
# --------------------------------------------------------------------------- #
PROJ_DISPLAY = {"mediastore": "MediaStore", "teastore": "TeaStore", "teammates": "Teammates",
                "bigbluebutton": "BigBlueButton", "jabref": "JabRef"}

RQ3_RUNS = ["run1", "run2", "run3"]
RQ3_COLS = ["ent_reject", "ent_keep", "coref_reject", "coref_keep"]


def _rq3_matrix(ent, cor, extra=None):
    """The two-row FP/TP confusion matrix shared by the mean and per-run tables.

    rows = true class, cols = judge x {REJECT, KEEP}. The TP-REJECT cell reports the
    *unique* rejected true positives (those this judge rejects that the other judge would
    keep) — the recall cost attributable to it alone. Raw counts pass through unrounded;
    csv_to_tex rounds per column kind (per-run = integer, mean = one decimal). ``extra``
    prepends fixed columns (e.g. the run label) to every row.
    """
    base = extra or {}
    return [
        {**base, "true_class": "False positive (FP)",
         "ent_reject": ent["rejected_fp"], "ent_keep": ent["kept_fp"],
         "coref_reject": cor["rejected_fp"], "coref_keep": cor["kept_fp"]},
        {**base, "true_class": "True positive (TP)",
         "ent_reject": ent["unique_rejected_tp"], "ent_keep": ent["kept_tp"],
         "coref_reject": cor["unique_rejected_tp"], "coref_keep": cor["kept_tp"]},
    ]


def build_rq3(backend, out):
    """Per-judge confusion matrix for one backend, averaged over the three runs
    (the body table; called for GPT-5.4 only)."""
    val = index(read_csv(RQ34 / "rq3_validators.csv"), "backend", "run", "validator")
    rows = _rq3_matrix(val[(backend, "average", "entity")], val[(backend, "average", "coref")])
    write_csv(out, ["true_class"] + RQ3_COLS, rows)


def build_rq3_runs(out):
    """Per-judge confusion matrix, both backends, every run plus the average in ONE table
    (grouped by backend, then run = Run 1/2/3 then the mean)."""
    val = index(read_csv(RQ34 / "rq3_validators.csv"), "backend", "run", "validator")
    rows = []
    for backend in ("openai", "claude"):
        for run in RQ3_RUNS + ["average"]:
            rows += _rq3_matrix(val[(backend, run, "entity")], val[(backend, run, "coref")],
                                extra={"backend": backend, "run": run})
    write_csv(out, ["backend", "run", "true_class"] + RQ3_COLS, rows)


# --------------------------------------------------------------------------- #
# RQ4 body table (GPT-5.4): the four ablation variants on the size-aware suite
# --------------------------------------------------------------------------- #
def _rq4_variant_cells(backend, run, dm_full, dm_noknow, size_link, size_noknow, uniq):
    """Assemble the four RQ4 variant rows for one backend and run ('average' or runN).

    dm_full/dm_noknow: rq4_variants.csv (linker_set->macro_f1) for full / no-knowledge slot.
    size_link: rq34_rq2_linkers.csv rows (linker_set Full/EntityOnly/CorefOnly).
    size_noknow: no-knowledge rq34_rq2_variants.csv 'Full' row (both linkers, knowledge off).
    uniq: rq4_linkers.csv rows (Entity/Coref unique_tps) -- a diagnostic, never displayed,
    so it stays on the run-average slot.
    """
    def panel(src):
        return {f"dc_{c}": src[f"doc_to_code_{c}"] for c in DC_SUITE}

    return [
        {"variant": "Full", "doc_to_model_macro_f1": dm_full["full"],
         **panel(size_link[(backend, run, "Full")]), "unique_tps": ""},
        {"variant": "Direct", "doc_to_model_macro_f1": dm_full["entity_only"],
         **panel(size_link[(backend, run, "EntityOnly")]),
         "unique_tps": i(uniq[(backend, "average", "Entity")]["unique_tps"])},
        {"variant": "Indirect", "doc_to_model_macro_f1": dm_full["coref_only"],
         **panel(size_link[(backend, run, "CorefOnly")]),
         "unique_tps": i(uniq[(backend, "average", "Coref")]["unique_tps"])},
        {"variant": "No knowledge", "doc_to_model_macro_f1": dm_noknow["full"],
         **panel(size_noknow), "unique_tps": ""},
    ]


def _load_rq4_sources(backend, run="average"):
    dm_full = {r["linker_set"]: r["macro_f1"]
               for r in read_csv(RQ34 / "rq4_variants.csv")
               if r["backend"] == backend and r["run"] == run}
    dm_noknow = {r["linker_set"]: r["macro_f1"]
                 for r in read_csv(RQ34_NOKNOW[backend] / "rq4_variants.csv")
                 if r["backend"] == backend and r["run"] == run}
    size_link = index(read_csv(RQ34 / "rq34_rq2_linkers.csv"), "backend", "run", "linker_set")
    size_noknow = index(read_csv(RQ34_NOKNOW[backend] / "rq34_rq2_variants.csv"),
                        "backend", "run", "variant")[(backend, run, "Full")]
    uniq = index(read_csv(RQ34 / "rq4_linkers.csv"), "backend", "run", "linker")
    return dm_full, dm_noknow, size_link, size_noknow, uniq


def build_rq4():
    dm_full, dm_noknow, size_link, size_noknow, uniq = _load_rq4_sources("openai")
    rows = _rq4_variant_cells("openai", "average", dm_full, dm_noknow, size_link, size_noknow, uniq)
    fields = (["variant", "doc_to_model_macro_f1"]
              + [f"dc_{c}" for c in ("file_f1", "sentence_coverage",
                                     "worst_component_f1", "harmonic_component_f1")]
              + ["unique_tps"])
    # Body table shows only the headline tail metrics; keep the four key ones.
    rows = [{k: r[k] for k in fields} for r in rows]
    write_csv("rq4.csv", fields, rows)


# --------------------------------------------------------------------------- #
# RQ1+RQ2 big tables (whole suite, both backends): average + per-project
# --------------------------------------------------------------------------- #
SUITE_COLS = (["doc_to_model_link_precision", "doc_to_model_link_recall", "doc_to_model_link_f1",
               "doc_to_model_sentence_coverage", "doc_to_model_noise_rate",
               "doc_to_model_silent_failure_mass"]
              + [f"doc_to_code_{c}" for c in DC_SUITE])

BIG_SYSTEMS = [  # (display label, (system, run) key into RQ12_BIGTABLE)
    ("approach (GPT-5.4)", ("approach (GPT-5.4)", "average")),
    ("approach (Claude)",  ("approach (Claude)", "average")),
    ("Artemis (GPT-5.4)",  ("Artemis (GPT-5.4)", "single")),
    ("TransArC",           ("TransArC", "single")),
]


def build_bigtable_rq12_perproject(big):
    """Per-project suite for every system, both backends, with a per-system ``Average``
    summary row carrying the five-project aggregate (the former standalone avg table)."""
    pp = index(read_csv(REPORTS / "RQ12_PERPROJECT.csv"), "system", "project")
    rows = []
    for label, key in BIG_SYSTEMS:
        for proj in PROJECTS:
            s = pp[(label, proj)]
            rows.append({"system": label, "project": proj, **{c: s[c] for c in SUITE_COLS}})
        avg = big[key]
        rows.append({"system": label, "project": "Average", **{c: avg[c] for c in SUITE_COLS}})
    write_csv("bigtable_rq12_perproject.csv", ["system", "project"] + SUITE_COLS, rows)


# (display label, key, runs) -- the approach is run three times; the baselines are deterministic.
PERRUN_SYSTEMS = [
    ("approach (GPT-5.4)", "approach (GPT-5.4)", ["run1", "run2", "run3", "average"]),
    ("approach (Claude)",  "approach (Claude)",  ["run1", "run2", "run3", "average"]),
    ("Artemis (GPT-5.4)",  "Artemis (GPT-5.4)",  ["single"]),
    ("TransArC",           "TransArC",           ["single"]),
]


def build_bigtable_rq12_perrun(big):
    """Whole suite per run for the (stochastic) approach on both backends, with the mean,
    plus the deterministic baselines once. Aggregate over the five projects."""
    rows = []
    for label, sys_key, runs in PERRUN_SYSTEMS:
        for run in runs:
            s = big[(sys_key, run)]
            rows.append({"system": label, "run": run, **{c: s[c] for c in SUITE_COLS}})
    write_csv("bigtable_rq12_perrun.csv", ["system", "run"] + SUITE_COLS, rows)


# --------------------------------------------------------------------------- #
# RQ4 big tables (whole suite, both backends): average + per-project
# --------------------------------------------------------------------------- #
RQ4_DISPLAY = [("Full", "Full"), ("Direct", "Direct"),
               ("Indirect", "Indirect"), ("No knowledge", "No knowledge")]


DM_SUITE = ["link_precision", "link_recall", "link_f1"]


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
    setmap = {"Full": "Full", "Direct": "EntityOnly", "Indirect": "CorefOnly"}
    dm_setmap = {"Full": "full", "Direct": "entity_only", "Indirect": "coref_only"}
    rows = []
    for backend in ("openai", "claude"):
        noknow_pp = index(read_csv(RQ34_NOKNOW[backend] / "rq34_rq2_variants_perproject.csv"),
                          "backend", "run", "variant", "project")
        noknow_dm = index(read_csv(RQ34_NOKNOW[backend] / "rq4_variants_perproject.csv"),
                          "backend", "run", "linker_set", "project")
        avg = {r["variant"]: r
               for r in _rq4_variant_cells(backend, "average", *_load_rq4_sources(backend))}
        for variant, _ in RQ4_DISPLAY:
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
RQ4_RUN_DC = ["file_precision", "file_recall", "file_f1",
              "sentence_coverage", "worst_component_f1", "harmonic_component_f1"]


def build_rq4_perrun():
    fields = ["backend", "variant", "doc_to_model_macro_f1"] + [f"dc_{c}" for c in RQ4_RUN_DC]
    for run, out in RQ4_PERRUN:
        rows = []
        for backend in ("openai", "claude"):
            cells = _rq4_variant_cells(backend, run, *_load_rq4_sources(backend, run))
            for r in cells:
                rows.append({"backend": backend, "variant": r["variant"],
                             "doc_to_model_macro_f1": r["doc_to_model_macro_f1"],
                             **{f"dc_{c}": r[f"dc_{c}"] for c in RQ4_RUN_DC}})
        write_csv(out, fields, rows)


# --------------------------------------------------------------------------- #
def main():
    big = index(read_csv(REPORTS / "RQ12_BIGTABLE.csv"), "system", "run")
    build_rq1(big)
    build_rq2(big)
    build_rq3("openai", "rq3.csv")                  # body confusion (GPT-5.4, mean of 3)
    build_rq3_runs("rq3_runs.csv")                  # appendix: both backends, each run + avg in one table
    build_rq4()
    build_bigtable_rq12_perproject(big)             # RQ1/RQ2 per-project + Average (both backends)
    build_bigtable_rq12_perrun(big)                 # RQ1/RQ2 per-run + avg (both backends)
    build_bigtable_rq4_perproject()                 # RQ4 per-project + Average (both backends)
    build_rq4_perrun()                              # RQ4 four per-run tables (run1/2/3 + avg)
    print(f"\n[rq_tables] table CSVs written under {TEX_SRC}", file=sys.stderr)


if __name__ == "__main__":
    main()
