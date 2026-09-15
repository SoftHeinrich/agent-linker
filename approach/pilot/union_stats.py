"""Paired statistics for the union pilot's dumps. No LLM calls.

`pilot/union_pilots.py --dump` writes each arm's kept pairs per sample per project.
This reads one or more dumps and reports, per model:

  * per (sample, project) paired deltas — 15 units at three samples on five projects;
  * the mean delta in gold, spurious and `net = 3*gold - spurious` (the branch's F2
    exchange rate at the head's operating point);
  * a **two-sided sign-flip permutation test** over those units, which is the test the
    branch reads deltas with, because the run-to-run band is wide enough that a mean
    on its own says little;
  * the same split by `naming` row, since the rows have base rates 0.98 / 0.49 / 0.31
    and a union that trades them against each other reads neutral in a sum.

    ../.venv/bin/python pilot/union_stats.py ../results/union_round/dump_terra_v13.json
    ../.venv/bin/python pilot/union_stats.py <dump> --arms control v13
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from union_diff import project_context                               # noqa: E402
from union_pilots import DEFAULT_RUN, ROWS                           # noqa: E402

F2_RATIO = 3.0


def sign_flip(deltas, trials=20000):
    """Two-sided sign-flip permutation p for a paired mean, exact when it can be."""
    observed = abs(statistics.fmean(deltas))
    n = len(deltas)
    if n == 0:
        return 1.0
    if n <= 18:                                   # exact: 2^n <= 262144
        extreme = sum(
            1 for signs in itertools.product((1, -1), repeat=n)
            if abs(statistics.fmean([s * d for s, d in zip(signs, deltas)]))
            >= observed - 1e-12)
        return extreme / 2 ** n
    import random
    rng = random.Random(20260911)
    extreme = sum(
        1 for _ in range(trials)
        if abs(statistics.fmean([d * rng.choice((1, -1)) for d in deltas]))
        >= observed - 1e-12)
    return extreme / trials


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dumps", nargs="+")
    parser.add_argument("--arms", nargs=2, default=["control", "union"])
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    left, right = args.arms

    for path in args.dumps:
        with open(path) as handle:
            dump = json.load(handle)
        contexts: dict = {}
        units = []
        row_units = defaultdict(list)
        for sample in sorted(dump):
            for project, arms in sorted(dump[sample].items()):
                if left not in arms or right not in arms:
                    continue
                if project not in contexts:
                    contexts[project] = project_context(project, args.run)
                context = contexts[project]
                gold = context["gold"]
                row_of = {pair: evidence["naming"]
                          for pair, evidence in context["evidence"].items()}
                kept = {arm: {tuple(p) for p in arms[arm]} for arm in (left, right)}
                score = {arm: (len(kept[arm] & gold), len(kept[arm] - gold))
                         for arm in (left, right)}
                units.append((sample, project,
                              score[right][0] - score[left][0],
                              score[right][1] - score[left][1]))
                for row in ROWS:
                    members = {pair for pair, value in row_of.items() if value == row}
                    row_units[row].append((
                        len(kept[right] & members & gold)
                        - len(kept[left] & members & gold),
                        len((kept[right] & members) - gold)
                        - len((kept[left] & members) - gold)))

        print(f"\n{Path(path).name}: `{right}` minus `{left}`, {len(units)} paired "
              f"(sample, project) units")
        golds = [unit[2] for unit in units]
        spurious = [unit[3] for unit in units]
        nets = [F2_RATIO * g - s for g, s in zip(golds, spurious)]
        print(f"  {'measure':<10}{'mean/unit':>11}{'per 5-project run':>19}{'p':>8}")
        for label, series, scale in (("gold", golds, 5), ("spurious", spurious, 5),
                                     ("net", nets, 5)):
            print(f"  {label:<10}{statistics.fmean(series):>11.2f}"
                  f"{statistics.fmean(series) * scale:>19.1f}"
                  f"{sign_flip(series):>8.3f}")
        wins = sum(1 for g, s in zip(golds, spurious) if F2_RATIO * g - s > 0)
        ties = sum(1 for g, s in zip(golds, spurious) if F2_RATIO * g - s == 0)
        print(f"  units where the union is ahead on net: {wins}, tied {ties}, "
              f"behind {len(units) - wins - ties}")

        print(f"\n  by naming row (mean per unit, p over the same units):")
        print(f"  {'row':<12}{'gold':>8}{'p':>8}{'spurious':>10}{'p':>8}")
        for row in ROWS:
            series = row_units[row]
            if not series:
                continue
            row_gold = [g for g, _ in series]
            row_spurious = [s for _, s in series]
            print(f"  {row:<12}{statistics.fmean(row_gold):>8.2f}"
                  f"{sign_flip(row_gold):>8.3f}"
                  f"{statistics.fmean(row_spurious):>10.2f}"
                  f"{sign_flip(row_spurious):>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
