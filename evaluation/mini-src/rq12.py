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
    Component Miss Rate: ``doc_to_model_component_miss_rate`` (%) +
    ``doc_to_model_component_miss_count`` (added 2026-06-30, doc-model only;
    the doc-code suite keeps worst/harmonic and gets NO CMR column).
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

SOTA_LINKS = Path(os.environ.get("SOTA_LINKS", m._ARDOCO_HOME / "sota/recovered-links"))

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
# is a matter of restoring the entries — see this file's git history.
ROSTER = [
    {"label": "approach (GPT-5.6-terra)", "backend": "gpt-5.6-terra",
     "runs": ["run1", "run2", "run3"],
     "sad-sam":  "model-doc/aalinker/terra_s92a/{run}/{project}.csv",
     "sad-code": "doc-code/aalinker-composed/terra_s92a/{run}/{project}.csv"},
    {"label": "approach (GPT-5.6-luna)", "backend": "gpt-5.6-luna",
     "runs": ["run1", "run2", "run3"],
     "sad-sam":  "model-doc/aalinker/luna_s92a/{run}/{project}.csv",
     "sad-code": "doc-code/aalinker-composed/luna_s92a/{run}/{project}.csv"},
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
    ("doc_to_model_component_miss_count", SS, "component_miss_count"),
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


def _mean(vectors, cols):
    return {c: sum(v[c] for v in vectors) / len(vectors) for c in cols}


def run_panels(system, task):
    """Per-run vectors, macro-averaged over the five projects.

    ``[(run_label, vector_dict), ...]``; the row axis of the big table.
    """
    cols = m.PANELS[task]
    return [(run, _mean(list(by_project.values()), cols))
            for run, by_project in score_cells(system, task)]


def average_vec(run_vectors, task):
    return _mean([v for _run, v in run_vectors], m.PANELS[task])


def metric_cells(ss_vec, sc_vec):
    """The COLUMNS projection: column name -> value from the right task's vector."""
    return {name: (ss_vec if task == SS else sc_vec)[key] for name, task, key in COLUMNS}


def build_row(system, run_label, ss_vec, sc_vec):
    return {"system": system["label"], "backend": system["backend"], "run": run_label,
            "doc_to_model_projects": len(m.PROJECTS),
            "doc_to_code_projects": len(m.PROJECTS),
            **metric_cells(ss_vec, sc_vec)}


def build_rows(system):
    """Big-table rows for one system: per-run rows plus average for multi-run systems."""
    ss_runs = run_panels(system, SS)
    sc_runs = run_panels(system, SC)
    if [r for r, _v in ss_runs] != [r for r, _v in sc_runs]:
        raise SystemExit(f"run labels differ between tasks for {system['label']}")
    if system["runs"] is None:
        return [build_row(system, "single", ss_runs[0][1], sc_runs[0][1])]

    rows = [build_row(system, run, ss_vec, sc_vec)
            for (run, ss_vec), (_run2, sc_vec) in zip(ss_runs, sc_runs)]
    rows.append(build_row(system, "average", average_vec(ss_runs, SS), average_vec(sc_runs, SC)))
    return rows


def project_panels(system, task):
    """Per-*project* metric vectors for one system on one task, averaged over runs.

    The orthogonal aggregation to ``run_panels`` (which macro-averages over
    projects per run): here the project axis is kept and each project's vector is
    averaged over the system's runs. Feeds the per-project big table. Single-shot
    systems contribute their one run unchanged.
    """
    cols = m.PANELS[task]
    cells = score_cells(system, task)
    return {proj: _mean([by_project[proj] for _run, by_project in cells], cols)
            for proj in m.PROJECTS}


def build_perproject_rows(system):
    """One row per (system, project): the full suite, mean over the system's runs."""
    ss = project_panels(system, SS)
    sc = project_panels(system, SC)
    return [{"system": system["label"], "backend": system["backend"], "project": proj,
             **metric_cells(ss[proj], sc[proj])}
            for proj in m.PROJECTS]


def summary_row(rows, label):
    return next((r for r in rows if r["system"] == label and r["run"] in ("average", "single")), None)



def delta_row(rows, a_label, b_label):
    """Δ row (a − b), per the RQ2 panel's Δ = approach − Artemis column."""
    a = summary_row(rows, a_label)
    b = summary_row(rows, b_label)
    if not a or not b:
        return None
    out = {"system": f"Delta ({a_label} - {b_label})", "backend": "", "run": "delta",
           "doc_to_model_projects": "", "doc_to_code_projects": ""}
    for name, _, _ in COLUMNS:
        out[name] = (a[name] - b[name]) if (a[name] is not None and b[name] is not None) else None
    return out


def fmt(v):
    return "" if v is None or v == "" else (f"{v:.4f}" if isinstance(v, float) else str(v))


def print_table(rows):
    names = [c[0] for c in COLUMNS]
    w = max(len(r["system"]) for r in rows) + 1
    widths = [max(len(n), 8) + 2 for n in names]
    head = "system".ljust(w) + "run".rjust(9) + "".join(n.rjust(width) for n, width in zip(names, widths))
    print(head)
    print("-" * len(head))
    for r in rows:
        print(r["system"].ljust(w) + r["run"].rjust(9)
              + "".join(fmt(r[n]).rjust(width) for n, width in zip(names, widths)))


# The two big tables differ only in their leading row-axis columns.
METRIC_FIELDS = [c[0] for c in COLUMNS]
BIGTABLE_FIELDS = ["system", "backend", "run",
                   "doc_to_model_projects", "doc_to_code_projects"] + METRIC_FIELDS
PERPROJECT_FIELDS = ["system", "backend", "project"] + METRIC_FIELDS


def write_csv(rows, fields, path):
    """Write `rows` as `fields`, every cell through ``fmt`` (LF line endings)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(fields)
        for r in rows:
            w.writerow([fmt(r[k]) for k in fields])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None,
                    help="output CSV path (default: <evaluation>/reports/RQ12_BIGTABLE.csv)")
    ap.add_argument("--perproject-csv", default=None,
                    help="per-project full-suite CSV path "
                         "(default: reports/RQ12_PERPROJECT.csv, or next to --csv)")
    args = ap.parse_args()

    rows = [row for system in ROSTER for row in build_rows(system)]

    big = list(rows)
    for arm in (BODY_ARM, MIRROR_ARM):
        d = delta_row(rows, arm, BASELINE_ARM)
        if d:
            big.append(d)
    print_table(big)
    print(f"\nProvenance: {SOTA_LINKS}  (approach rows = run1/run2/run3 plus average)")
    print("RQ1 doc-to-model = doc_to_model_link_precision/recall/f1/f2; "
          "RQ1 doc-to-code = doc_to_code_file_precision/recall/f1/f2.")
    print("RQ2 size-aware = doc_to_code_file_f1/f2, "
          "doc_to_code_{worst,harmonic}_component_f1/f2, "
          "doc_to_model_component_miss_rate.")

    reports = m._ARDOCO_HOME / "transarc-emp/reports"
    out = Path(args.csv) if args.csv else reports / "RQ12_BIGTABLE.csv"
    write_csv(big, BIGTABLE_FIELDS, out)

    pp_rows = [row for system in ROSTER for row in build_perproject_rows(system)]
    out2 = Path(args.perproject_csv) if args.perproject_csv else (
        out.parent / "RQ12_PERPROJECT.csv" if args.csv else reports / "RQ12_PERPROJECT.csv")
    write_csv(pp_rows, PERPROJECT_FIELDS, out2)
    print(f"\n[rq12] wrote {out}\n[rq12] wrote {out2}", file=sys.stderr)


if __name__ == "__main__":
    main()
