"""Score whole five-project runs and permutation-test two variants against each other.

Every end-to-end comparison in this branch has been assembled by hand from the
per-run `ablation_*.json` files. This does it in one place: read the predicted-link
CSVs of a set of run directories, score them against the gold standard, and run the
exact permutation test in `ab_stats.py` on the pooled link sets, so both the
composition statistic and the quality statistics (TP, FP, macro F1, macro F2) come
from the same code.

Links are keyed `(project, sentence, component_id)`, so one run is one set across all
five projects and the macro scores can be recomputed inside a quality callable for
every relabelling.

    ../.venv/bin/python pilot/score_runs.py \
        --arm s_linker25 ../results/s25_simplified_e2e_r*_20260810 \
        --arm s_linker42 ../results/s42_threevalue_e2e_r*_20260812
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from ab_stats import permutation_report                             # noqa: E402
from design_audit import PROJECTS, load_gold                         # noqa: E402


def gold_all():
    out = set()
    for project in PROJECTS:
        out |= {(project, snum, cid) for snum, cid in load_gold(project)}
    return out


GOLD = gold_all()
GOLD_BY_PROJECT = {p: {k for k in GOLD if k[0] == p} for p in PROJECTS}


def load_run(run: Path, variant: str):
    """Every predicted link of one run, keyed (project, sentence, component_id)."""
    links = set()
    found = 0
    for project in PROJECTS:
        path = run / f"{variant}_{project}_links.csv"
        if not path.exists():
            continue
        found += 1
        with path.open() as handle:
            for row in csv.DictReader(handle):
                links.add((project, int(row["sentence"]), row["component_id"]))
    return links if found == len(PROJECTS) else None


def scores(links):
    """TP, FP, macro F1, macro F2 of one run's link set."""
    tp = len(links & GOLD)
    fp = len(links - GOLD)
    f1s, f2s = [], []
    for project, gold in GOLD_BY_PROJECT.items():
        got = {k for k in links if k[0] == project}
        hit = len(got & gold)
        precision = hit / len(got) if got else 0.0
        recall = hit / len(gold) if gold else 0.0
        f1s.append(0.0 if not (precision + recall) else
                   2 * precision * recall / (precision + recall))
        f2s.append(0.0 if not (4 * precision + recall) else
                   5 * precision * recall / (4 * precision + recall))
    return {"TP": tp, "FP": fp,
            "macro F1": 100 * sum(f1s) / len(f1s),
            "macro F2": 100 * sum(f2s) / len(f2s)}


QUALITY = {name: (lambda links, key=name: scores(links)[key])
           for name in ("TP", "FP", "macro F1", "macro F2")}


def calls_of(run: Path, variant: str):
    total = 0
    for path in (run / "llm_logs").glob(f"{variant}_openai_*_calls.json"):
        with path.open() as handle:
            total += len(json.load(handle))
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", nargs="+", action="append", required=True,
                        metavar=("VARIANT", "RUN"),
                        help="variant name followed by its run directories")
    args = parser.parse_args()

    arms = {}
    for arm in args.arm:
        variant, run_paths = arm[0], [Path(p) for p in arm[1:]]
        runs = []
        print(f"\n{variant}")
        for run in run_paths:
            links = load_run(run, variant)
            if links is None:
                print(f"  {run.name:44s} incomplete — skipped")
                continue
            runs.append(links)
            s = scores(links)
            print(f"  {run.name:44s} TP {s['TP']:3d} FP {s['FP']:3d} "
                  f"F1 {s['macro F1']:6.2f} F2 {s['macro F2']:6.2f} "
                  f"calls {calls_of(run, variant)}")
        if not runs:
            continue
        arms[variant] = runs
        mean = {k: sum(scores(r)[k] for r in runs) / len(runs) for k in QUALITY}
        spread = max(scores(r)["macro F1"] for r in runs) - min(
            scores(r)["macro F1"] for r in runs)
        print(f"  {'MEAN over ' + str(len(runs)) + ' runs':44s} "
              f"TP {mean['TP']:5.1f} FP {mean['FP']:5.1f} "
              f"F1 {mean['macro F1']:6.2f} F2 {mean['macro F2']:6.2f} "
              f"(F1 range {spread:.2f})")

    # With more than two arms every later arm is tested against the first, which is
    # the baseline by convention. One invocation can therefore carry a baseline and
    # several candidates and still produce properly paired tests.
    names = list(arms)
    for other in names[1:]:
        pair = {names[0]: arms[names[0]], other: arms[other]}
        if min(len(v) for v in pair.values()) < 2:
            continue
        if len(pair[names[0]]) != len(pair[other]):
            keep = min(len(v) for v in pair.values())
            print(f"\ntruncating both arms to {keep} runs for the paired test")
            pair = {k: v[:keep] for k, v in pair.items()}
        permutation_report(pair, quality=QUALITY,
                           title=f"{other} minus {names[0]}")


if __name__ == "__main__":
    main()
