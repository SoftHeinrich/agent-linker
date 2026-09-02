#!/usr/bin/env python3
"""RQ1/RQ2 cross-system "big table" generator (CSV).

``metrics.py`` is a per-*system* calculator whose output axis is *projects*: it
scores one result set at a time and prints a per-project panel + macro average.
The paper's RQ1/RQ2 tables have *systems* on the row axis, the approach averaged
over three runs, and a per-task layout. This driver closes that gap: it sweeps
the system roster, scores each through ``metrics.py`` (the sole metric impl),
macro-averages over the five projects, emits each approach run plus its average,
and writes ONE wide CSV whose columns are the union of both tasks.

That single big table is a superset of every RQ1/RQ2 cell:
  * RQ1 doc-to-model (tab:rq1-sadsam) = columns
    ``doc_to_model_link_precision``, ``doc_to_model_link_recall``,
    ``doc_to_model_link_f1``/``_f2``. The doc-model group also carries the size-aware
    Component Miss Rate: ``doc_to_model_component_miss_rate`` (%) (added
    2026-06-30, doc-model only; the doc-code suite keeps worst/harmonic and gets
    NO CMR column). The ``..._component_miss_count`` twin was dropped 2026-09-01
    unread -- CSVs written before that date still carry the column, and every
    reader here selects columns by name, so old and new dumps interoperate.
  * RQ1 doc-to-code  (tab:rq1-sadcode) = columns
    ``doc_to_code_file_precision``, ``doc_to_code_file_recall``,
    ``doc_to_code_file_f1``/``_f2``.
  * RQ2 size-aware panel (tab:rq2) =
    ``doc_to_code_file_f1``, ``doc_to_code_worst_component_f1``,
    ``doc_to_code_harmonic_component_f1``.

Every F1 column above has an ``_f2`` twin beside it (recall-weighted \ftwo, the
paper reports the pair everywhere): file, link, component-micro, worst-component
and harmonic-component. Both flavours come out of the same
``metrics.compute_sad_{code,sam}`` call, so nothing here re-derives an F.

It generates only the numbers (CSVs); it does not write .tex. No new metric
code lives here — every cell comes from ``metrics.compute_sad_{code,sam}``, so
``check.py``'s frozen goldens still pin the arithmetic.

Inputs (the normalized SOTA dump; ``sentence_id,target_id`` dialect, which
``metrics.load_result`` auto-detects):
    <ardoco-home>/sota/recovered-links/
      model-doc/   doc-to-model links  (SWATTR/Artemis/LiSSA + aalinker/<be>/run*)
      doc-code/    doc-to-code  links  (TransArc/Artemis/LiSSA + aalinker-composed/<be>/run*)
Override the root via ``$SOTA_LINKS``.

Usage
-----
    python3 mini-src/rq12.py                      # -> reports/RQ12_BIGTABLE.csv (+ stdout)
    python3 mini-src/rq12.py --csv /tmp/big.csv

Every root defaults to the in-repo layout, so a bare run (or an IDE Run button)
needs no environment variables; $SOTA_LINKS / $TRANSARC_BENCHMARK still override.

Provenance note (worst/harmonic, the approach rows): these are recomputed here
from the recorded three-run ``aalinker-composed`` dump (mean of the three runs),
not copied from any earlier table — the tail metrics are run-sensitive, so they
are only meaningful next to the run set they came from.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics as m   # noqa: E402  (mini-src/metrics.py — sole metric impl + loaders)

SOTA_LINKS = Path(os.environ.get("SOTA_LINKS", m.REPO / "sota-links"))
REPORTS = m.REPO / "evaluation" / "reports"      # where the committed CSVs live

# The \approach arm whose dump slots the roster reads. The paper reports s110; a
# candidate arm is scored by pointing this at its slot and writing the output beside the
# incumbent's, then diffing the two with ../studies/compare_arms.py (whose --base is the
# arm s110 replaced, s92a). Only the \approach rows move -- the baselines are
# arm-independent and stay pinned.
# Precedence: --arm > $ALINKER_ARM > DEFAULT_ARM.
DEFAULT_ARM = "s110"
ARM = os.environ.get("ALINKER_ARM", DEFAULT_ARM)


def set_arm(arm):
    """Repoint the roster's \approach rows at `arm`'s dump slots (paths only, labels fixed).

    Labels stay ``approach (GPT-5.6-<backend>)`` across arms on purpose: rq_tables.py
    matches on them, so an arm swap must not rename a row. The arm is identified by the
    output file it is written to and by the provenance line, not by the row label."""
    global ARM
    ARM = arm
    for system in ROSTER:
        for task, template in system.get("_arm_paths", {}).items():
            system[task] = template.format(arm=arm, run="{run}", project="{project}")

# ── System roster ─────────────────────────────────────────────────────────────
# Each entry resolves a per-(run,)project file path for both tasks. `runs=None`
# means single-shot (deterministic / SOTA); a run list means mean-of-runs.
# `aalinker`/`aalinker-composed` are the recovered doc-to-model / composed
# doc-to-code dumps. SWATTR is TransArC's deterministic doc-to-model stage
# (TransArC has no standalone doc-to-model system).
#
# CANONICAL = s_linker92a (entity extraction as a scan), N=3, on two GPT-5.6
# backends: terra = paper body, luna = the second-backend mirror. Their sota slots
# are built by mini-src/build_alinker_extracts.py (run CSVs -> neutral extracts)
# piped into mini-src/build_dump.py.
#
# The retired arms (s_linker21 `*_s21`, s_linker20_union `*_full`) were dropped
# from the roster on 2026-08-26: nothing downstream read their rows once the paper
# rebased onto s92a. Their slots are still in the sota dump, so scoring them again
# is a matter of restoring the entries — see this file's git history. s92a's own slots
# stay built for a different reason: ../studies/compare_arms.py --base s92a reads them,
# and the promotion decision is only reproducible while they exist.
ROSTER = [
    # `_arm_paths` is the arm-templated source of `sad-sam`/`sad-code`, rendered by
    # set_arm() -- which is now called at import (below the roster), so these literals are
    # overwritten before anything can read them. They stayed on s92a through the s110
    # promotion and were the last reader of the retired arm: check.py's guard reads
    # DEFAULT_ARM, which had moved, so nothing caught them. They are pinned to
    # DEFAULT_ARM and kept only so the shape of an entry is readable here.
    {"label": "approach (GPT-5.6-terra)", "backend": "gpt-5.6-terra",
     "runs": ["run1", "run2", "run3"],
     "_arm_paths": {
         "sad-sam":  "model-doc/aalinker/terra_{arm}/{run}/{project}.csv",
         "sad-code": "doc-code/aalinker-composed/terra_{arm}/{run}/{project}.csv"},
     "sad-sam":  "model-doc/aalinker/terra_s110/{run}/{project}.csv",
     "sad-code": "doc-code/aalinker-composed/terra_s110/{run}/{project}.csv"},
    {"label": "approach (GPT-5.6-luna)", "backend": "gpt-5.6-luna",
     "runs": ["run1", "run2", "run3"],
     "_arm_paths": {
         "sad-sam":  "model-doc/aalinker/luna_{arm}/{run}/{project}.csv",
         "sad-code": "doc-code/aalinker-composed/luna_{arm}/{run}/{project}.csv"},
     "sad-sam":  "model-doc/aalinker/luna_s110/{run}/{project}.csv",
     "sad-code": "doc-code/aalinker-composed/luna_s110/{run}/{project}.csv"},
    # ArTEMiS twice: once on the backend \approach uses, once at the authors' released
    # configuration. The matched-backend row is the one the body reports -- comparing a
    # GPT-5.6 approach against a GPT-5.4 baseline confounds the workflow with the model.
    # The released row stays in the roster so the appendix keeps the published arm and a
    # reader can see what the backend change alone costs the baseline (it trades precision
    # for recall: F1 down, F2 up). The matched-backend arm is run three times like the
    # approach rows -- each with its own LLM cache dir, or the runs replay each other --
    # while the released GPT-5.4 arm is the authors' single published run.
    {"label": "Artemis (GPT-5.6-terra)", "backend": "gpt-5.6-terra",
     "runs": ["run1", "run2", "run3"],
     "sad-sam":  "model-doc/artemis/terra_5.6/{run}/{project}.csv",
     "sad-code": "doc-code/artemis/terra_5.6/{run}/{project}.csv"},
    {"label": "Artemis (GPT-5.4)",  "backend": "gpt-5.4", "runs": None,
     "sad-sam":  "model-doc/artemis-{project}-gpt-5.4.csv",
     "sad-code": "doc-code/artemis-{project}-gpt-5.4.csv"},
    {"label": "TransArC",           "backend": "deterministic", "runs": None,
     "sad-sam":  "model-doc/swattr-{project}.csv",        # SWATTR = TransArC doc-model
     "sad-code": "doc-code/transarc-{project}.csv"},
]

# The arm the paper's body tables report, and the mirror backend beside it.
# Render the roster now, so an importer that never reaches main() still reads the
# reported arm. Before this call the roster's literals were the only thing in the
# pipeline still resolving to s92a after the s110 promotion.
set_arm(ARM)

BODY_ARM = "approach (GPT-5.6-terra)"
MIRROR_ARM = "approach (GPT-5.6-luna)"
# The baseline the Delta rows subtract: ArTEMiS on the SAME backend as BODY_ARM.
BASELINE_ARM = "Artemis (GPT-5.6-terra)"

# Combined big-table column layout: friendly name -> (task, metric key in PANELS).
SS = "sad-sam"
SC = "sad-code"
COLUMNS = [
    ("doc_to_model_link_precision", SS, "link_p"),
    ("doc_to_model_link_recall", SS, "link_r"),
    ("doc_to_model_link_f1", SS, "link_f1"),
    ("doc_to_model_link_f2", SS, "link_f2"),
    ("doc_to_model_component_miss_rate", SS, "component_miss_rate"),
    ("doc_to_code_file_precision", SC, "file_p"),
    ("doc_to_code_file_recall", SC, "file_r"),
    ("doc_to_code_file_f1", SC, "file_f1"),
    ("doc_to_code_file_f2", SC, "file_f2"),
    ("doc_to_code_component_micro_f1", SC, "component_f1"),
    ("doc_to_code_component_micro_f2", SC, "component_f2"),
    ("doc_to_code_worst_component_f1", SC, "worst_component_f1"),
    ("doc_to_code_worst_component_f2", SC, "worst_component_f2"),
    ("doc_to_code_harmonic_component_f1", SC, "harmonic_component_f1"),
    ("doc_to_code_harmonic_component_f2", SC, "harmonic_component_f2"),
]

# The two big tables carry the same metric columns and differ only in their
# leading row-axis columns: (system, backend, run) vs (system, backend, project).
METRIC_FIELDS = [name for name, _task, _key in COLUMNS]
BIGTABLE_FIELDS = ["system", "backend", "run",
                   "doc_to_model_projects", "doc_to_code_projects"] + METRIC_FIELDS
PERPROJECT_FIELDS = ["system", "backend", "project"] + METRIC_FIELDS


def score_cells(system, task):
    """Every (run, project) cell of one system on one task, scored.

    The single loader behind both aggregation axes below. Scores each project
    through ``metrics.compute_*``, failing loud if a required result file is
    absent or empty -- so a panel is always complete by construction, never
    silently short a project. Returns ``[(run_label, {project: vector}), ...]``
    in run order; single-shot systems yield one ``single`` entry.
    """
    compute = m.compute_sad_code if task == SC else m.compute_sad_sam
    pattern = system[task]

    cells = []
    for run in (system["runs"] or [None]):
        by_project = {}
        for proj in m.PROJECTS:
            rel = pattern.format(run=run, project=proj) if run else pattern.format(project=proj)
            path = SOTA_LINKS / rel
            cell = f"{system['label']} {run or 'single'} {proj}: {path}"
            if not path.exists():
                raise SystemExit(f"missing required {task} result for {cell}")
            res = m.load_result(path, task)
            if not res:
                raise SystemExit(f"empty/unparseable required {task} result for {cell}")
            by_project[proj] = compute(proj, res)
        cells.append((run or "single", by_project))
    return cells


def mean_vector(vectors, cols):
    """Element-wise mean of metric vectors, over `cols`."""
    vectors = list(vectors)
    mean = {}
    for col in cols:
        mean[col] = sum(v[col] for v in vectors) / len(vectors)
    return mean


def macro_by_run(system, task):
    """One vector per run, macro-averaged over the five projects.

    The big table's row axis: ``[(run_label, vector), ...]`` in run order.
    """
    cols = m.PANELS[task]
    panels = []
    for run, by_project in score_cells(system, task):
        macro = mean_vector(by_project.values(), cols)
        panels.append((run, macro))
    return panels


def mean_by_project(system, task):
    """One vector per project, averaged over the system's runs.

    The orthogonal aggregation to ``macro_by_run``: the project axis is kept and
    the run axis collapses. Feeds the per-project table. Single-shot systems
    contribute their one run unchanged.
    """
    cols = m.PANELS[task]
    cells = score_cells(system, task)
    panels = {}
    for proj in m.PROJECTS:
        over_runs = [by_project[proj] for _run, by_project in cells]
        panels[proj] = mean_vector(over_runs, cols)
    return panels


def metric_columns(dm_vec, dc_vec):
    """The COLUMNS projection: column name -> value from the right task's vector."""
    cells = {}
    for name, task, key in COLUMNS:
        vector = dm_vec if task == SS else dc_vec
        cells[name] = vector[key]
    return cells


def build_rows(system):
    """Big-table rows for one system: one per run, plus `average` if it has runs."""
    dm = macro_by_run(system, SS)
    dc = macro_by_run(system, SC)
    dm_runs = [run for run, _vec in dm]
    dc_runs = [run for run, _vec in dc]
    if dm_runs != dc_runs:
        raise SystemExit(f"run labels differ between tasks for {system['label']}")

    # (row label, doc-model vector, doc-code vector), in printed order. A
    # single-shot system yields one "single" panel and has nothing to average.
    panels = []
    for (run, dm_vec), (_run, dc_vec) in zip(dm, dc):
        panels.append((run, dm_vec, dc_vec))
    if system["runs"] is not None:
        dm_average = mean_vector([vec for _run, vec in dm], m.PANELS[SS])
        dc_average = mean_vector([vec for _run, vec in dc], m.PANELS[SC])
        panels.append(("average", dm_average, dc_average))

    rows = []
    for label, dm_vec, dc_vec in panels:
        row = {"system": system["label"], "backend": system["backend"], "run": label,
               "doc_to_model_projects": len(m.PROJECTS),
               "doc_to_code_projects": len(m.PROJECTS)}
        row.update(metric_columns(dm_vec, dc_vec))
        rows.append(row)
    return rows


def build_perproject_rows(system):
    """One row per (system, project): the full suite, mean over the system's runs."""
    dm = mean_by_project(system, SS)
    dc = mean_by_project(system, SC)
    rows = []
    for proj in m.PROJECTS:
        row = {"system": system["label"], "backend": system["backend"], "project": proj}
        row.update(metric_columns(dm[proj], dc[proj]))
        rows.append(row)
    return rows


def delta_row(rows, arm_label, baseline_label):
    """The Δ (arm − baseline) row: the RQ2 panel's Δ = approach − Artemis column.

    Each operand is that system's summary row -- ``average`` for a multi-run
    system, ``single`` for a single-shot one. None if either label is absent.
    """
    summary = {}
    for row in rows:
        if row["run"] in ("average", "single"):
            summary[row["system"]] = row
    arm = summary.get(arm_label)
    baseline = summary.get(baseline_label)
    if not arm or not baseline:
        return None

    delta = {"system": f"Delta ({arm_label} - {baseline_label})",
             "backend": "", "run": "delta",
             "doc_to_model_projects": "", "doc_to_code_projects": ""}
    for name in METRIC_FIELDS:
        delta[name] = arm[name] - baseline[name]
    return delta


def fmt(v):
    return "" if v is None or v == "" else (f"{v:.4f}" if isinstance(v, float) else str(v))


# Short headers + task band for the stdout table ONLY. The CSVs keep the long
# machine names of COLUMNS -- nothing downstream reads this print, and printing a
# 35-character column name over a 6-character number made the table unreadable.
SHORT = {
    "doc_to_model_link_precision": "P",
    "doc_to_model_link_recall": "R",
    "doc_to_model_link_f1": "F1",
    "doc_to_model_link_f2": "F2",
    "doc_to_model_component_miss_rate": "CMR%",
    "doc_to_code_file_precision": "P",
    "doc_to_code_file_recall": "R",
    "doc_to_code_file_f1": "F1",
    "doc_to_code_file_f2": "F2",
    "doc_to_code_component_micro_f1": "cF1",
    "doc_to_code_component_micro_f2": "cF2",
    "doc_to_code_worst_component_f1": "wF1",
    "doc_to_code_worst_component_f2": "wF2",
    "doc_to_code_harmonic_component_f1": "hF1",
    "doc_to_code_harmonic_component_f2": "hF2",
}
BAND = {SS: "doc-model (link + CMR%)", SC: "doc-code (file + per-component)"}


def print_table(rows):
    """The stdout view: two header lines -- a task band over short metric names.

    Same rows, same values, same order as the CSV; only the header shortens, and
    P/R/F1/F2 repeat per task because the band above says which task they belong
    to. The doc-model block comes first because COLUMNS orders it first.
    """
    names = METRIC_FIELDS
    widths = []
    for name in names:
        widths.append(max(len(SHORT[name]), 7) + 1)
    label_w = max(len(row["system"]) for row in rows) + 2
    run_w = max(len(row["run"]) for row in rows) + 2

    # Line 1: one band per task, centred over that task's columns.
    band = " " * (label_w + run_w)
    for task in (SS, SC):
        span = 0
        for (_name, column_task, _key), width in zip(COLUMNS, widths):
            if column_task == task:
                span += width
        band += BAND[task].center(span)

    # Line 2: the short metric names.
    head = "system".ljust(label_w) + "run".rjust(run_w)
    for name, width in zip(names, widths):
        head += SHORT[name].rjust(width)

    print(band.rstrip())
    print(head)
    print("-" * len(head))
    for row in rows:
        line = row["system"].ljust(label_w) + row["run"].rjust(run_w)
        for name, width in zip(names, widths):
            line += fmt(row[name]).rjust(width)
        print(line)


def write_csv(rows, fields, path):
    """Write `rows` as `fields`, every cell through ``fmt`` (LF line endings)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(fields)
        for row in rows:
            cells = []
            for field in fields:
                cells.append(fmt(row[field]))
            writer.writerow(cells)


def arm_slots(arm):
    """The dump directories `arm` must provide, one per \approach row and task."""
    slots = []
    for system in ROSTER:
        for template in system.get("_arm_paths", {}).values():
            slots.append(SOTA_LINKS / template.format(arm=arm, run="", project="").split("//")[0])
    return slots


def check_arm(arm):
    """Fail fast, and by name, when an arm has no measured run set yet.

    Without this an unbuilt arm surfaces as five per-project "missing file" errors that
    read like a corrupt dump rather than "this arm was never run"."""
    missing = [d for d in dict.fromkeys(arm_slots(arm)) if not d.is_dir()]
    if missing:
        raise SystemExit(
            f"[rq12] arm {arm!r} has no dump slots:\n" +
            "\n".join(f"  missing: {d}" for d in missing) +
            f"\n[rq12] build the arm first (build_alinker_extracts.py -> build_dump.py), "
            f"or pick a built arm: --arm {DEFAULT_ARM}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default=ARM,
                    help=f"\\approach arm to score, i.e. the dump slot suffix "
                         f"(default: $ALINKER_ARM or {DEFAULT_ARM}). Baselines are "
                         f"arm-independent and are unaffected.")
    ap.add_argument("--csv", default=None,
                    help="output CSV path (default: <evaluation>/reports/RQ12_BIGTABLE.csv)")
    ap.add_argument("--perproject-csv", default=None,
                    help="per-project full-suite CSV path "
                         "(default: reports/RQ12_PERPROJECT.csv, or next to --csv)")
    args = ap.parse_args()
    check_arm(args.arm)
    set_arm(args.arm)

    rows = []
    for system in ROSTER:
        rows.extend(build_rows(system))

    big = list(rows)
    for arm in (BODY_ARM, MIRROR_ARM):
        delta = delta_row(rows, arm, BASELINE_ARM)
        if delta:
            big.append(delta)
    print_table(big)
    print(f"\nProvenance: {SOTA_LINKS}  arm={ARM}  "
          f"(approach rows = run1/run2/run3 plus average)")
    print("Columns: P/R/F1/F2 = link (doc-model) resp. file (doc-code) scores; "
          "CMR% = component miss rate;")
    print("         cF1/cF2 = per-component micro, wF1/wF2 = worst component, "
          "hF1/hF2 = harmonic-mean component.")
    print("RQ1 = the two P/R/F1/F2 blocks. RQ2 size-aware = CMR% (doc-model) and "
          "w*/h* + file F1/F2 (doc-code).")

    # Both CSVs land in one directory: REPORTS by default, or beside --csv.
    # The default arm keeps the historical filenames (the paper syncs against them);
    # any other arm is suffixed so scoring a candidate cannot clobber the incumbent.
    suffix = "" if args.arm == DEFAULT_ARM else f"_{args.arm}"
    if args.csv:
        big_csv = Path(args.csv)
    else:
        big_csv = REPORTS / f"RQ12_BIGTABLE{suffix}.csv"
    if args.perproject_csv:
        perproject_csv = Path(args.perproject_csv)
    else:
        perproject_csv = big_csv.parent / f"RQ12_PERPROJECT{suffix}.csv"

    perproject_rows = []
    for system in ROSTER:
        perproject_rows.extend(build_perproject_rows(system))

    write_csv(big, BIGTABLE_FIELDS, big_csv)
    write_csv(perproject_rows, PERPROJECT_FIELDS, perproject_csv)
    print(f"\n[rq12] wrote {big_csv}\n[rq12] wrote {perproject_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
