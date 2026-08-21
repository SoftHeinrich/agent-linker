"""Paired statistics for the static round's paraphrase arms.

Reads the `kept_<group>_<model>_<arm>.json` dumps `pilot/static_pilots.py`
writes, composes each arm's kept pairs with the recorded stages it does not touch,
and runs the branch's own sign-flip permutation test (`pilot/ab_stats.py`) of every
arm against the first one. Two readings per arm, because they answer different
questions:

  stage     gold and spurious pairs at the stage the arm changes -- the only surface
            the change can reach, which is what this branch reads an arm on first.
  composed  macro F1 / F2 / TP / FP of the arm's stage unioned with the same run's
            recorded other two stages. Exact, not projected: no prompt in any arm
            reaches those stages.

The p floor at n=3 is 0.10 and every p below is reported against it.

Usage (from approach/):
    ../.venv/bin/python pilot/static_round_stats.py --group qual1 --model terra
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "src")

from ab_stats import permutation_report                              # noqa: E402
from static_pilots import (                                    # noqa: E402
    OTHER_STAGES, PROJECTS, RECORDED, scores,
)

BASE = "/mnt/hostshare/ardoco-home/alinker-replication-package"


def gold(path):
    with open(os.path.join(BASE, "benchmark", path)) as fh:
        return {(int(r["sentence"]), r["modelElementID"]) for r in csv.DictReader(fh)}


GOLD = {proj: gold(g) for proj, (_t, _m, g) in PROJECTS.items()}


def recorded_dirs(model):
    pattern, variant = RECORDED[model]
    return sorted(glob.glob(os.path.join(BASE, "results", pattern,
                                         "phase_states"))), variant


def other_stage_links(run_dir, variant, proj, stages):
    links = set()
    for stage in stages:
        fn = os.path.join(run_dir, variant, "openai", proj, f"linker_{stage}.pkl")
        if os.path.exists(fn):
            links |= {(l.sentence_number, l.component_id)
                      for l in pickle.load(open(fn, "rb"))["links"]}
    return links


def macro(links_by_project, key):
    per = []
    for proj, links in links_by_project.items():
        g = GOLD[proj]
        tp = len(links & g)
        per.append(scores(tp, len(links) - tp, len(g) - tp))
    return st.mean(x[0 if key == "f1" else 1] for x in per)


def flat(links_by_project):
    return {(proj, s, c) for proj, links in links_by_project.items()
            for (s, c) in links}


def load(group, model, arm, out_dir):
    return json.load(open(Path(out_dir) / f"kept_{group}_{model}_{arm}.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--out", default="../results/static_round")
    args = ap.parse_args()

    dumps = sorted(glob.glob(str(Path(args.out) /
                                 f"kept_{args.group}_{args.model}_*.json")))
    arms = args.arms or [Path(d).stem.split("_")[-1] for d in dumps]
    # the control is the arm named ctl, first
    arms = ["ctl"] + [a for a in arms if a != "ctl"]
    run_dirs, variant = recorded_dirs(args.model)

    stage_sets, composed_sets = {}, {}
    for arm in arms:
        kept = load(args.group, args.model, arm, args.out)
        s_runs, c_runs = [], []
        for i, run_key in enumerate(sorted(kept)):
            run_dir = run_dirs[i % len(run_dirs)]
            stage_by_proj, comp_by_proj = {}, {}
            for proj in PROJECTS:
                pairs = {tuple(x) for x in kept[run_key].get(proj, [])}
                stage_by_proj[proj] = pairs
                comp_by_proj[proj] = pairs | other_stage_links(
                    run_dir, variant, proj, OTHER_STAGES[args.group])
            s_runs.append(flat(stage_by_proj))
            c_runs.append(flat(comp_by_proj))
        stage_sets[arm] = s_runs
        composed_sets[arm] = c_runs

    def tp(flat_set):
        return sum(1 for (proj, s, c) in flat_set if (s, c) in GOLD[proj])

    def fp(flat_set):
        return len(flat_set) - tp(flat_set)

    def by_project(flat_set):
        out = {proj: set() for proj in PROJECTS}
        for proj, s, c in flat_set:
            out[proj].add((s, c))
        return out

    quality_stage = {"gold": tp, "spurious": fp}
    quality_comp = {
        "TP": tp, "FP": fp,
        "macro F1": lambda s: macro(by_project(s), "f1"),
        "macro F2": lambda s: macro(by_project(s), "f2"),
    }

    n = len(stage_sets["ctl"])
    from math import comb
    print(f"\n{args.group} on {args.model}: {n} runs a side, "
          f"p floor {2 / comb(2 * n, n):.2f}\n")
    for arm in arms:
        print(f"  {arm:<13} stage gold {st.mean(map(tp, stage_sets[arm])):6.1f}  "
              f"spurious {st.mean(map(fp, stage_sets[arm])):6.1f}   "
              f"composed macroF1 {macro_mean(composed_sets[arm], 'f1', by_project):6.2f}  "
              f"macroF2 {macro_mean(composed_sets[arm], 'f2', by_project):6.2f}  "
              f"TP {st.mean(map(tp, composed_sets[arm])):6.1f}  "
              f"FP {st.mean(map(fp, composed_sets[arm])):6.1f}")
    for arm in arms[1:]:
        permutation_report({"ctl": stage_sets["ctl"], arm: stage_sets[arm]},
                           quality=quality_stage,
                           title=f"{args.group}/{args.model}: {arm} vs ctl "
                                 f"(stage only)")
        permutation_report({"ctl": composed_sets["ctl"], arm: composed_sets[arm]},
                           quality=quality_comp,
                           title=f"{args.group}/{args.model}: {arm} vs ctl "
                                 f"(composed with recorded "
                                 f"{'+'.join(OTHER_STAGES[args.group])})")


def macro_mean(runs, key, by_project):
    return st.mean(macro(by_project(r), key) for r in runs)


if __name__ == "__main__":
    main()
