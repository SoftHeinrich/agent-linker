#!/usr/bin/env python3
"""RQ4's total floor: the workflow against one linking call.

`s_linker110_onecall` is given the document, the component list and the discovered
alias table, and returns the final link set -- no scan, no window, no evidence bundle,
no antecedent shortlist, no judge, no union. The head's rubrics are rendered verbatim,
so what the arm removes is the arrangement and not the guidance.

The arm has no per-linker phases, so `rq34.py` cannot read it: there are no stages to
attribute. This engine therefore scores both arms end to end, straight off the
predicted-link CSVs each run writes, using the tree's shared confusion matrix
(`metrics.prf_counts` / `metrics.fbeta`) so no F-measure is re-derived here.

Doc-model grain only. A link is `(modelElementID, sentence)`, which is exactly the key
`metrics.load_gs_sad_sam` returns, so gold and prediction are compared without a
normalisation step.

No document-length column: `s_linker27`'s length effect is the first thing a reader of a
floor number asks about, and the sentence counts already live in
`paper/table/gold_concentration.csv`. One source, cited from the caption, rather than a
second that can drift from it.

Output (`--csv-root`, default `reports_<arm>_floor/`):

    rq4_floor.csv   backend x run x arm x project, plus a `project=Average` row per
                    run and a `run=average` row per project -- counts, P, R, F1, F2.

Both run families are named by template so a re-run only changes a knob:

    RQ4_FLOOR_HEAD_TMPL   default noevidence_e2e_{model}_r{i}_20260902
    RQ4_FLOOR_ARM_TMPL    default onecall_e2e_{model}_r{i}_20260902
    RQ4_FLOOR_HEAD_KEY    default s_linker110
    RQ4_FLOOR_ARM_KEY     default s_linker110_onecall

**The control is CROSS-SET by decision.** The head runs come from a different
invocation than the arm's, which the branch normally forbids because absolute levels
drift: `s_linker110` on terra read macro F1 93.85 in one of the 2026-09-02 sets and
92.90 in another. Roughly 1 F1 of that band sits on every delta this engine reports.
The header of the emitted CSV says so; keep it there.

    python3 mini-rq34/rq4_floor.py --backends terra luna
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "mini-src"))
import metrics as m  # noqa: E402  (shared core: benchmark layout, gold, F-measures)

_ARDOCO_HOME = _HERE.parent.parent
RESULTS = Path(os.environ.get(
    "ALINKER_RESULTS",
    _ARDOCO_HOME / "results" if (_ARDOCO_HOME / "results").is_dir()
    else _ARDOCO_HOME / "agent-linker/results"))

HEAD_TMPL = os.environ.get("RQ4_FLOOR_HEAD_TMPL", "noevidence_e2e_{model}_r{i}_20260902")
ARM_TMPL = os.environ.get("RQ4_FLOOR_ARM_TMPL", "onecall_e2e_{model}_r{i}_20260902")
HEAD_KEY = os.environ.get("RQ4_FLOOR_HEAD_KEY", "s_linker110")
ARM_KEY = os.environ.get("RQ4_FLOOR_ARM_KEY", "s_linker110_onecall")

ARMS = (("Full", HEAD_TMPL, HEAD_KEY), ("OneCall", ARM_TMPL, ARM_KEY))
BACKENDS = ("terra", "luna")
RUNS = (1, 2, 3)

FIELDS = ["backend", "run", "arm", "project",
          "tp", "fp", "fn", "precision", "recall", "f1", "f2"]


def read_links(run_dir: Path, key: str, project: str):
    """The run's predicted doc-model links, keyed as `load_gs_sad_sam` keys gold.

    Returns None when the file is absent, so a partial sweep is reported as missing
    rather than silently scored as an empty prediction -- which `prf_counts` would
    score 0 and which would look like a result.
    """
    path = run_dir / f"{key}_{project}_links.csv"
    if not path.exists():
        return None
    with path.open() as f:
        return {(r["component_id"], r["sentence"]) for r in csv.DictReader(f)}


def score(gold, pred, project, backend, run, arm):
    tp, fp, fn, precision, recall, f1 = m.prf_counts(gold, pred)
    return {"backend": backend, "run": run, "arm": arm, "project": project,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "f2": round(m.fbeta(precision, recall), 4)}


def averaged(rows, over, keep):
    """Macro-average `rows` over the `over` field, keeping `keep` fixed.

    Macro, not micro: the paper's link-level figures average per project and per run,
    and RQ2 exists because volume-weighting hides the small components.
    """
    if not rows:
        return None
    out = dict(rows[0])
    out.update(keep)
    for field in ("tp", "fp", "fn"):
        out[field] = round(sum(r[field] for r in rows) / len(rows), 1)
    for field in ("precision", "recall", "f1", "f2"):
        out[field] = round(sum(r[field] for r in rows) / len(rows), 4)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv-root", type=Path, default=None,
                    help="output dir (default reports_<arm>_floor/)")
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS),
                    choices=list(BACKENDS))
    args = ap.parse_args()
    csv_root = args.csv_root or _HERE / "reports_s110_floor"

    gold = {p: m.load_gs_sad_sam(p) for p in m.PROJECTS}
    rows, missing = [], []

    for backend in args.backends:
        for arm, tmpl, key in ARMS:
            per_run = []
            for i in RUNS:
                run_dir = RESULTS / tmpl.format(model=backend, i=i)
                scored = []
                for project in m.PROJECTS:
                    pred = read_links(run_dir, key, project)
                    if pred is None:
                        missing.append(f"{backend}/{arm}/run{i}/{project}")
                        continue
                    scored.append(score(gold[project], pred, project, backend, i, arm))
                if not scored:
                    continue
                rows.extend(scored)
                macro = averaged(scored, "project", {"project": "Average"})
                rows.append(macro)
                per_run.append(scored)

            # one `run=average` row per project, and one for the macro
            for project in m.PROJECTS:
                cells = [r for run in per_run for r in run if r["project"] == project]
                if cells:
                    rows.append(averaged(cells, "run", {"run": "average"}))
            macros = [averaged(run, "project", {"project": "Average"}) for run in per_run]
            if macros:
                rows.append(averaged(macros, "run",
                                     {"run": "average", "project": "Average"}))

    m.write_dict_csv(csv_root / "rq4_floor.csv", FIELDS, rows)
    print(f"[rq4-floor] wrote {csv_root / 'rq4_floor.csv'} ({len(rows)} rows)")
    if missing:
        print(f"[rq4-floor] NOTE: {len(missing)} missing run/project slots, "
              f"not scored: {', '.join(missing[:6])}"
              + (" ..." if len(missing) > 6 else ""))
    print("[rq4-floor] NOTE: the control is CROSS-SET; ~1 macro F1 of invocation "
          "drift sits on every delta derived from this file.")


if __name__ == "__main__":
    main()
