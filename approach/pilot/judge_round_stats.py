"""Paired statistics for the judge round's stage dumps.

`pilot/nextgen_pilots.py --dump` writes the pairs each arm KEPT, per sample and
project. A stage arm screens candidates and does not decide them, so the two numbers
that order arms here are gold kept and spurious kept -- not F1, which would treat a
gate's own recall denominator as the whole task's.

The single ordering statistic is `3*gold - spurious`, the F2 derivative at the head's
operating point written out: one recovered gold link is worth about three avoided false
positives, so an arm that trades fewer than three spurious for one gold is a loss under
this branch's budget even when it looks like a precision win.

Every arm is compared to `control` from the SAME invocation, by the round's own
permutation test (`pilot/ab_stats.py`), pooling the five projects per sample.

    ../.venv/bin/python pilot/judge_round_stats.py ../results/judge_round/dump_sortal_terra.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ab_stats import permutation_report  # noqa: E402
from reading_pilots import BENCH, DATASETS, gold_pairs  # noqa: E402

#: One gold link is worth this many avoided false positives under F2 at the head.
F2_RATIO = 3.0


def load(path: Path):
    """{arm: [pooled kept-set per sample]} and {arm: {project: [set per sample]}}."""
    raw = json.load(open(path))
    pooled = defaultdict(list)
    per_project = defaultdict(lambda: defaultdict(list))
    for sample in sorted(raw):
        by_arm = defaultdict(set)
        for project, arms in raw[sample].items():
            for arm, pairs in arms.items():
                keys = {(project, int(s), c) for s, c in pairs}
                by_arm[arm] |= keys
                per_project[arm][project].append(keys)
        for arm, keys in by_arm.items():
            pooled[arm].append(keys)
    return pooled, per_project


def gold_keys():
    keys = set()
    for project, (_t, _m, gold_path) in DATASETS.items():
        keys |= {(project, s, c) for s, c in gold_pairs(BENCH / gold_path)}
    return keys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    parser.add_argument("--control", default="control")
    args = parser.parse_args()

    pooled, per_project = load(args.dump)
    gold = gold_keys()
    tp = lambda s: len(s & gold)            # noqa: E731
    fp = lambda s: len(s - gold)            # noqa: E731
    net = lambda s: F2_RATIO * tp(s) - fp(s)  # noqa: E731

    arms = [a for a in pooled if a != args.control]
    print(f"{args.dump.name}: {len(pooled[args.control])} samples, "
          f"arms {[args.control] + arms}\n")
    print(f"  {'arm':<10}{'gold':>8}{'spurious':>10}{'precision':>11}"
          f"{'3*gold-spurious':>18}")
    for arm in [args.control] + arms:
        runs = pooled[arm]
        g = sum(tp(s) for s in runs) / len(runs)
        f = sum(fp(s) for s in runs) / len(runs)
        print(f"  {arm:<10}{g:>8.1f}{f:>10.1f}"
              f"{g / (g + f) if g + f else 0:>11.3f}{F2_RATIO * g - f:>18.1f}")

    for arm in arms:
        print(f"\n--- {args.control} vs {arm} ---")
        report = permutation_report(
            {args.control: pooled[args.control], arm: pooled[arm]},
            quality={"gold": tp, "spurious": fp, "3*gold-spurious": net},
            title=f"{args.control} vs {arm}",
        )
        print(report if isinstance(report, str) else json.dumps(report, indent=1))
        print("  per project (gold / spurious, mean over samples):")
        for project in sorted(DATASETS):
            base = per_project[args.control].get(project)
            other = per_project[arm].get(project)
            if not base or not other:
                continue
            bg = sum(tp(s) for s in base) / len(base)
            bf = sum(fp(s) for s in base) / len(base)
            og = sum(tp(s) for s in other) / len(other)
            of = sum(fp(s) for s in other) / len(other)
            print(f"    {project:<15}{bg:5.1f} / {bf:4.1f}   ->{og:6.1f} / {of:4.1f}"
                  f"   net {F2_RATIO * (og - bg) - (of - bf):+6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
