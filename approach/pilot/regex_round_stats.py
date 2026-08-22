"""The regex round's statistics: the stage arm's composed link sets, permutation-tested.

`pilot/regex_proposer_pilots.py` writes each arm's kept full-name pairs per run.
This composes them with the same recorded run's untouched partial-name and
coreference stages -- the exact pipeline link set, not a projection -- and runs
`ab_stats.permutation_report` on the pooled sets, so composition and the four quality
statistics come from one place.

    ../.venv/bin/python pilot/regex_round_stats.py --model terra
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from ab_stats import permutation_report                              # noqa: E402
import regex_proposer_pilots as PILOT                                # noqa: E402

ROUND = Path("../results/regex_round")


def gold_all():
    return {(project, snum, cid)
            for project, (_t, _m, path) in PILOT.PROJECTS.items()
            for snum, cid in PILOT.gold(path)}


GOLD = gold_all()
GOLD_BY_PROJECT = {p: {k for k in GOLD if k[0] == p} for p in PILOT.PROJECTS}


def scores(links):
    f1s, f2s = [], []
    for project, project_gold in GOLD_BY_PROJECT.items():
        got = {k for k in links if k[0] == project}
        hit = len(got & project_gold)
        f1, f2 = PILOT.scores(hit, len(got) - hit, len(project_gold) - hit)
        f1s.append(f1)
        f2s.append(f2)
    return {"TP": len(links & GOLD), "FP": len(links - GOLD),
            "macro F1": sum(f1s) / len(f1s), "macro F2": sum(f2s) / len(f2s)}


QUALITY = {name: (lambda links, key=name: scores(links)[key])
           for name in ("TP", "FP", "macro F1", "macro F2")}


def composed(model, arm, runs):
    """One set per run: the arm's full-name pairs plus the run's other two stages."""
    kept = json.loads((ROUND / f"kept_{model}_{arm}.json").read_text())
    run_dirs, variant = PILOT.recorded_runs(model)
    out = []
    for index in range(runs):
        run_dir = run_dirs[index % len(run_dirs)]
        links = set()
        for project in PILOT.PROJECTS:
            links |= {(project, snum, cid)
                      for snum, cid in kept[f"run{index + 1}"].get(project, [])}
            for stage in PILOT.OTHER_STAGES:
                recorded = PILOT.state(run_dir, variant, project, f"linker_{stage}")
                if recorded:
                    links |= {(project, link.sentence_number, link.component_id)
                              for link in recorded["links"]}
        out.append(links)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(PILOT.RECORDED), required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--arms", nargs="*", default=list(PILOT.ARMS))
    args = parser.parse_args()

    arms = {arm: composed(args.model, arm, args.runs) for arm in args.arms}
    for arm, sets in arms.items():
        rows = [scores(s) for s in sets]
        print(f"{arm:<5} " + "  ".join(
            f"{k} {sum(r[k] for r in rows) / len(rows):7.2f}" for k in QUALITY))
    control = list(arms)[0]
    for arm in list(arms)[1:]:
        print()
        permutation_report({control: arms[control], arm: arms[arm]},
                           quality=QUALITY,
                           title=f"{arm} vs {control}, {args.model}, n={args.runs}")


if __name__ == "__main__":
    main()
