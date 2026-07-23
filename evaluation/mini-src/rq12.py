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
    ``doc_to_model_link_f1``. The doc-model group also carries the size-aware
    Silent-Failure Mass: ``doc_to_model_silent_failure_mass`` (%) +
    ``doc_to_model_silent_failure_count`` (added 2026-06-30, doc-model only;
    the doc-code suite keeps worst/harmonic and gets NO SFM column).
  * RQ1 doc-to-code  (tab:rq1-sadcode) = columns
    ``doc_to_code_file_precision``, ``doc_to_code_file_recall``,
    ``doc_to_code_file_f1``.
  * RQ2 size-aware panel (tab:rq2-summary) =
    ``doc_to_code_file_f1``, ``doc_to_code_sentence_coverage``,
    ``doc_to_code_worst_component_f1``, ``doc_to_code_harmonic_component_f1``.

A second, focused CSV (``reports/RQ2_PANEL.csv``) is also emitted for RQ2: it
lists those four columns for BOTH approach backends (GPT-5.4 **and Claude/sonnet**,
the latter not in the current paper table), the deltas vs the strongest baseline,
and a row flagged ``OUTDATED`` carrying the paper's archived approach numbers
(.62/.71) — produced from a deleted run set, kept only so the stale values are
visibly marked against the reproducible ones.

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
    python3 mini-src/rq12.py --lissa              # also include the LiSSA row(s)

Provenance note (worst/harmonic, the approach rows): these are recomputed here
from the bundled three-run ``aalinker-composed`` dump (mean of the three runs).
The paper's RQ2 worst/harmonic for the approach (.62/.71) were produced by a
now-deleted script over a now-deleted run set (``v2.6.5_s20union_gpt_re_medium``);
the file-F1 still matches (~.874), but the run-sensitive tail does not. The
values emitted here are the reproducible ones from surviving data.
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

# ── System roster (matches tables/cross_system.tex) ───────────────────────────
# Each entry resolves a per-(run,)project file path for both tasks. `runs=None`
# means single-shot (deterministic / SOTA); a run list means mean-of-runs.
# `aalinker`/`aalinker-composed` are the recovered doc-to-model / composed
# doc-to-code dumps. SWATTR is TransArC's deterministic doc-to-model stage
# (TransArC has no standalone doc-to-model system).
#
# CANONICAL = S21 (s_linker21, layered no-reasoning validator), N=3. The bare
# "approach (...)" rows are S21 — the configuration the paper reports (GPT-5.4 =
# main body, Claude = appendix mirror); the `*_s21` aalinker slots are built by
# build_s21_dump.py. The prior canonical s_linker20_union (the `*_full` slots,
# the NO-REASONING extract from agent-linker/results/v2.6.6_extracts) is kept as
# the "approach s20union (...)" rows for side-by-side comparison only. The paper's
# RQ2 numbers came from a now-deleted reasoning=medium slot — see PAPER_RQ2_OUTDATED.
ROSTER = [
    # Canonical: s_linker21 (S21), N=3 — bare "approach" = S21 everywhere downstream.
    {"label": "approach (GPT-5.4)", "backend": "gpt-5.4", "runs": ["run1", "run2", "run3"],
     "sad-sam":  "model-doc/aalinker/gpt-5.4_s21/{run}/{project}.csv",
     "sad-code": "doc-code/aalinker-composed/gpt-5.4_s21/{run}/{project}.csv"},
    {"label": "approach (Claude)",  "backend": "claude",  "runs": ["run1", "run2", "run3"],
     "sad-sam":  "model-doc/aalinker/sonnet_s21/{run}/{project}.csv",
     "sad-code": "doc-code/aalinker-composed/sonnet_s21/{run}/{project}.csv"},
    # Prior canonical, kept for side-by-side comparison: s_linker20_union (`*_full`).
    {"label": "approach s20union (GPT-5.4)", "backend": "gpt-5.4", "runs": ["run1", "run2", "run3"],
     "sad-sam":  "model-doc/aalinker/gpt-5.4_full/{run}/{project}.csv",
     "sad-code": "doc-code/aalinker-composed/gpt-5.4_full/{run}/{project}.csv"},
    {"label": "approach s20union (Claude)", "backend": "claude", "runs": ["run1", "run2", "run3"],
     "sad-sam":  "model-doc/aalinker/sonnet_full/{run}/{project}.csv",
     "sad-code": "doc-code/aalinker-composed/sonnet_full/{run}/{project}.csv"},
    {"label": "Artemis (GPT-5.4)",  "backend": "gpt-5.4", "runs": None,
     "sad-sam":  "model-doc/artemis-{project}-gpt-5.4.csv",
     "sad-code": "doc-code/artemis-{project}-gpt-5.4.csv"},
    {"label": "TransArC",           "backend": "deterministic", "runs": None,
     "sad-sam":  "model-doc/swattr-{project}.csv",        # SWATTR = TransArC doc-model
     "sad-code": "doc-code/transarc-{project}.csv"},
]
LISSA = {"label": "LiSSA (gpt-5-mini)", "backend": "gpt-5-mini", "runs": None,
         "sad-sam":  "model-doc/lissa-{project}-gpt-5-mini.csv",
         "sad-code": "doc-code/lissa-{project}-gpt-5-mini.csv"}

# The paper's RQ2 approach numbers (working/table/rq2-results.tex) came from the
# WRONG-CONFIG run: the deleted v2.6.5_s20union_gpt_re_medium slot, i.e. gpt-5.4
# with reasoning=medium (run_s20union_gpt_re_medium_n3.sh), scored by the deleted
# /tmp/v265.py. The correct input is the newest NO-REASONING run (v2.6.6_extracts
# /gpt; see the roster note above). file-F1 is config-robust so it still matches
# (~.87), but the run-sensitive tail (worst/harmonic) does not. Carried here ONLY
# to flag these as OUTDATED next to the reproducible no-reasoning values.
PAPER_RQ2_OUTDATED = {  # approach, gpt-5.4 reasoning=medium, doc-to-code
    "doc_to_code_file_f1": 0.87,
    "doc_to_code_sentence_coverage": 0.80,
    "doc_to_code_worst_component_f1": 0.62,
    "doc_to_code_harmonic_component_f1": 0.71,
}

# RQ2 size-aware panel: doc-to-code, both approach backends + the two baselines.
RQ2_SYSTEMS = ["TransArC", "Artemis (GPT-5.4)", "approach (GPT-5.4)", "approach (Claude)", "approach s20union (GPT-5.4)", "approach s20union (Claude)"]
RQ2_COLS = [
    "doc_to_code_file_f1",
    "doc_to_code_sentence_coverage",
    "doc_to_code_worst_component_f1",
    "doc_to_code_harmonic_component_f1",
]

# Combined big-table column layout: friendly name -> (task, metric key in PANELS).
SS = "sad-sam"
SC = "sad-code"
COLUMNS = [
    ("doc_to_model_link_precision", SS, "link_p"),
    ("doc_to_model_link_recall", SS, "link_r"),
    ("doc_to_model_link_f1", SS, "link_f1"),
    ("doc_to_model_sentence_coverage", SS, "sentence_coverage"),
    ("doc_to_model_noise_rate", SS, "noise_rate"),
    ("doc_to_model_silent_failure_mass", SS, "silent_failure_mass"),
    ("doc_to_model_silent_failure_count", SS, "silent_failure_count"),
    ("doc_to_code_file_precision", SC, "file_p"),
    ("doc_to_code_file_recall", SC, "file_r"),
    ("doc_to_code_file_f1", SC, "file_f1"),
    ("doc_to_code_component_micro_f1", SC, "component_f1"),
    ("doc_to_code_worst_component_f1", SC, "worst_component_f1"),
    ("doc_to_code_harmonic_component_f1", SC, "harmonic_component_f1"),
    ("doc_to_code_sentence_coverage", SC, "sentence_coverage"),
    ("doc_to_code_noise_rate", SC, "noise_rate"),
]


def run_panels(system, task):
    """Per-run macro-averaged metric vectors for one system on one task.

    Scores every project through ``metrics.compute_*`` (failing if any required
    result file is absent or empty), then macro-averages over projects. Returns
    ``[(run_label, vector_dict), ...]``. Single-shot systems use ``run=single``.
    """
    cols = m.PANELS[task]
    compute = m.compute_sad_code if task == SC else m.compute_sad_sam
    pattern = system[task]
    runs = system["runs"] or [None]

    run_vectors = []
    for run in runs:
        rows = []
        for proj in m.PROJECTS:
            rel = pattern.format(run=run, project=proj) if run else pattern.format(project=proj)
            path = SOTA_LINKS / rel
            if not path.exists():
                raise SystemExit(f"missing required {task} result for {system['label']} "
                                 f"{run or 'single'} {proj}: {path}")
            res = m.load_result(path, task)
            if not res:
                raise SystemExit(f"empty/unparseable required {task} result for "
                                 f"{system['label']} {run or 'single'} {proj}: {path}")
            rows.append(compute(proj, res))
        if len(rows) != len(m.PROJECTS):
            raise SystemExit(f"incomplete {task} panel for {system['label']} "
                             f"{run or 'single'}: {len(rows)}/{len(m.PROJECTS)} projects")
        run_vectors.append((run or "single", {c: sum(r[c] for r in rows) / len(rows) for c in cols}))
    if len(run_vectors) != len(runs):
        raise SystemExit(f"incomplete {task} run set for {system['label']}: "
                         f"{len(run_vectors)}/{len(runs)} runs")
    return run_vectors


def average_vec(run_vectors, task):
    cols = m.PANELS[task]
    return {c: sum(v[c] for _run, v in run_vectors) / len(run_vectors) for c in cols}


def build_row(system, run_label, ss_vec, sc_vec):
    row = {"system": system["label"], "backend": system["backend"], "run": run_label,
           "doc_to_model_projects": len(m.PROJECTS), "doc_to_code_projects": len(m.PROJECTS)}
    for name, task, key in COLUMNS:
        vec = ss_vec if task == SS else sc_vec
        row[name] = vec[key]
    return row


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
    projects per run): here we keep the project axis and average each project's
    vector over the system's runs. Feeds the per-project big table. Single-shot
    systems contribute their one run unchanged.
    """
    cols = m.PANELS[task]
    compute = m.compute_sad_code if task == SC else m.compute_sad_sam
    pattern = system[task]
    runs = system["runs"] or [None]

    per_proj = {proj: [] for proj in m.PROJECTS}
    for run in runs:
        for proj in m.PROJECTS:
            rel = pattern.format(run=run, project=proj) if run else pattern.format(project=proj)
            path = SOTA_LINKS / rel
            if not path.exists():
                raise SystemExit(f"missing required {task} result for {system['label']} "
                                 f"{run or 'single'} {proj}: {path}")
            res = m.load_result(path, task)
            if not res:
                raise SystemExit(f"empty/unparseable required {task} result for "
                                 f"{system['label']} {run or 'single'} {proj}: {path}")
            per_proj[proj].append(compute(proj, res))
    return {proj: {c: sum(r[c] for r in vecs) / len(vecs) for c in cols}
            for proj, vecs in per_proj.items()}


def build_perproject_rows(system):
    """One row per (system, project): the full suite, mean over the system's runs."""
    ss = project_panels(system, SS)
    sc = project_panels(system, SC)
    rows = []
    for proj in m.PROJECTS:
        row = {"system": system["label"], "backend": system["backend"], "project": proj}
        for name, task, key in COLUMNS:
            vec = ss[proj] if task == SS else sc[proj]
            row[name] = vec[key]
        rows.append(row)
    return rows


def write_perproject_csv(rows, path):
    fields = ["system", "backend", "project"] + [c[0] for c in COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(fields)
        for r in rows:
            w.writerow([fmt(r[k]) for k in fields])


def summary_row(rows, label):
    return next((r for r in rows if r["system"] == label and r["run"] in ("average", "single")), None)


def row_for(rows, label, run):
    return next((r for r in rows if r["system"] == label and r["run"] == run), None)


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


def write_csv(rows, path):
    fields = ["system", "backend", "run", "doc_to_model_projects", "doc_to_code_projects"] + [c[0] for c in COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(fields)
        for r in rows:
            w.writerow([fmt(r[k]) for k in fields])


def build_rq2_panel(rows):
    """RQ2 size-aware rows (doc-to-code), approach runs + averages + deltas."""
    panel = []
    for label in ("TransArC", "Artemis (GPT-5.4)"):
        r = row_for(rows, label, "single")
        if r is None:
            continue
        panel.append({"system": label, "run": "single",
                      **{c: r[c] for c in RQ2_COLS}, "note": ""})

    approach_runs = ["run1", "run2", "run3", "average"]
    for label in ("approach (GPT-5.4)", "approach (Claude)", "approach s20union (GPT-5.4)", "approach s20union (Claude)"):
        for run in approach_runs:
            r = row_for(rows, label, run)
            if r is not None:
                panel.append({"system": label, "run": run,
                              **{c: r[c] for c in RQ2_COLS}, "note": ""})

    art = row_for(rows, "Artemis (GPT-5.4)", "single")
    if art is not None:
        for label in ("approach (GPT-5.4)", "approach (Claude)", "approach s20union (GPT-5.4)", "approach s20union (Claude)"):
            for run in approach_runs:
                r = row_for(rows, label, run)
                if r is None:
                    continue
                panel.append({"system": f"Delta ({label} - Artemis)",
                              "run": run,
                              **{c: r[c] - art[c] for c in RQ2_COLS}, "note": ""})
    panel.append({"system": "paper approach (GPT-5.4)", "run": "paper", **PAPER_RQ2_OUTDATED,
                  "note": "OUTDATED: gpt reasoning=medium (deleted v2.6.5_s20union_gpt_re_medium); live rows = no-reasoning v2.6.6"})
    return panel


def print_rq2(panel):
    print("\nRQ2 size-aware panel (doc-to-code) -- now incl. the Claude/sonnet row")
    w = max(len(r["system"]) for r in panel) + 1
    widths = [max(len(c), 8) + 2 for c in RQ2_COLS]
    head = "system".ljust(w) + "run".rjust(9) + "".join(c.rjust(width) for c, width in zip(RQ2_COLS, widths)) + "  note"
    print(head)
    print("-" * (len(head)))
    for r in panel:
        print(r["system"].ljust(w) + r["run"].rjust(9)
              + "".join(fmt(r[c]).rjust(width) for c, width in zip(RQ2_COLS, widths))
              + ("  " + r["note"] if r["note"] else ""))


def write_rq2_csv(panel, path):
    fields = ["system", "run"] + RQ2_COLS + ["note"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(fields)
        for r in panel:
            w.writerow([fmt(r[k]) for k in fields])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None,
                    help="output CSV path (default: <evaluation>/reports/RQ12_BIGTABLE.csv)")
    ap.add_argument("--rq2-csv", default=None,
                    help="RQ2 panel CSV path (default: reports/RQ2_PANEL.csv, or next to --csv)")
    ap.add_argument("--perproject-csv", default=None,
                    help="per-project full-suite CSV path "
                         "(default: reports/RQ12_PERPROJECT.csv, or next to --csv)")
    ap.add_argument("--lissa", action="store_true",
                    help="also include LiSSA; aborts unless all required project files are present")
    args = ap.parse_args()

    roster = ROSTER + ([LISSA] if args.lissa else [])
    rows = [row for system in roster for row in build_rows(system)]

    big = list(rows)
    d = delta_row(rows, "approach (GPT-5.4)", "Artemis (GPT-5.4)")
    if d:
        big.append(d)
    print_table(big)
    print(f"\nProvenance: {SOTA_LINKS}  (approach rows = run1/run2/run3 plus average)")
    print("RQ1 doc-to-model = doc_to_model_link_precision/recall/f1; "
          "RQ1 doc-to-code = doc_to_code_file_precision/recall/f1.")
    print("RQ2 panel = doc_to_code_file_f1, doc_to_code_sentence_coverage, "
          "doc_to_code_worst_component_f1, doc_to_code_harmonic_component_f1.")

    panel = build_rq2_panel(rows)
    print_rq2(panel)

    reports = m._ARDOCO_HOME / "transarc-emp/reports"
    out = Path(args.csv) if args.csv else reports / "RQ12_BIGTABLE.csv"
    write_csv(big, out)
    out2 = Path(args.rq2_csv) if args.rq2_csv else (
        out.parent / "RQ2_PANEL.csv" if args.csv else reports / "RQ2_PANEL.csv")
    write_rq2_csv(panel, out2)

    pp_rows = [row for system in roster for row in build_perproject_rows(system)]
    out3 = Path(args.perproject_csv) if args.perproject_csv else (
        out.parent / "RQ12_PERPROJECT.csv" if args.csv else reports / "RQ12_PERPROJECT.csv")
    write_perproject_csv(pp_rows, out3)
    print(f"\n[rq12] wrote {out}\n[rq12] wrote {out2}\n[rq12] wrote {out3}", file=sys.stderr)


if __name__ == "__main__":
    main()
