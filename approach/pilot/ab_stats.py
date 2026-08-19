"""Permutation test for A/B pilots on a stochastic LLM pipeline.

A "min distance vs max noise band" rule is not a test: it passes automatically
whenever the noise band is wide, which is exactly when the design has no power.
Instead relabel the runs every possible way and ask how extreme the true A|B
labelling is among all labellings.

Two statistics, because they answer different questions:

  * composition -- mean between-arm set distance minus mean within-arm distance.
    Does the change alter WHICH items the pipeline produces?
  * quality -- any per-run scalar (TP count, FP count, ...). Does it alter how
    many are right?

Composition can shift while quality does not. That combination is a real result
and a single pass/fail verdict cannot express it.

Note the p floor: with n runs per arm there are only C(2n, n)/2 distinct splits,
so p cannot fall below 2/C(2n, n). At n=3 the floor is 0.10 — an effect can be
the most extreme of all labellings and still not reach a conventional
threshold. Always report the floor next to the p.
"""
from __future__ import annotations

from itertools import combinations


def permutation_report(arm_sets, quality=None, title="permutation test"):
    """arm_sets: {arm_name: [set, ...]} with equal run counts per arm.

    quality: {label: callable(set) -> number}, compared as arm2-mean minus
    arm1-mean, two-sided.
    """
    names = list(arm_sets)
    if len(names) != 2:
        raise ValueError("exactly two arms")
    a_runs, b_runs = arm_sets[names[0]], arm_sets[names[1]]
    if len(a_runs) != len(b_runs):
        raise ValueError("arms must have equal run counts")
    runs = list(a_runs) + list(b_runs)
    n, half = len(runs), len(a_runs)

    def spread(group):
        left = [runs[i] for i in group]
        right = [runs[i] for i in range(n) if i not in group]
        within = [len(x ^ y) for side in (left, right)
                  for i, x in enumerate(side) for y in side[i + 1:]]
        between = [len(x ^ y) for x in left for y in right]
        if not within or not between:
            raise ValueError("need >=2 runs per arm")
        return sum(between) / len(between) - sum(within) / len(within)

    def delta(group, key):
        vals = [key(s) for s in runs]
        left = [vals[i] for i in group]
        right = [vals[i] for i in range(n) if i not in group]
        return sum(right) / len(right) - sum(left) / len(left)

    truth = tuple(range(half))
    # Halve by symmetry: a split and its complement give the same statistic.
    splits = [s for s in combinations(range(n), half) if 0 in s]
    floor = 1 / len(splits)

    observed = spread(truth)
    null = sorted((spread(s) for s in splits), reverse=True)
    p_composition = sum(1 for v in null if v >= observed) / len(splits)

    quality_out = {}
    for label, key in (quality or {}).items():
        obs = delta(truth, key)
        p = sum(1 for s in splits if abs(delta(s, key)) >= abs(obs)) / len(splits)
        quality_out[label] = {
            f"{names[0]}_mean": round(sum(key(s) for s in a_runs) / half, 1),
            f"{names[1]}_mean": round(sum(key(s) for s in b_runs) / half, 1),
            "delta": round(obs, 1),
            "p": round(p, 2),
        }

    flat = all(v["p"] > 0.2 for v in quality_out.values()) if quality_out else None
    if flat is None:
        verdict = f"composition p={p_composition:.2f} (floor {floor:.2f})"
    elif flat:
        verdict = (f"QUALITY-NEUTRAL ("
                   + ", ".join(f"{k} p={v['p']}" for k, v in quality_out.items())
                   + f"); composition p={p_composition:.2f}, floor {floor:.2f}")
    else:
        verdict = ("QUALITY-CHANGING ("
                   + ", ".join(f"{k} p={v['p']}" for k, v in quality_out.items())
                   + ")")

    stats = {
        "arms": names,
        "runs_per_arm": half,
        "sizes": {names[0]: [len(s) for s in a_runs],
                  names[1]: [len(s) for s in b_runs]},
        "within_A": [len(x ^ y) for i, x in enumerate(a_runs) for y in a_runs[i + 1:]],
        "within_B": [len(x ^ y) for i, x in enumerate(b_runs) for y in b_runs[i + 1:]],
        "between": [len(x ^ y) for x in a_runs for y in b_runs],
        "composition_stat": round(observed, 1),
        "composition_null": [round(v, 1) for v in null],
        "p_composition": round(p_composition, 2),
        "p_floor": round(floor, 2),
        "quality": quality_out,
        "verdict": verdict,
    }

    print(f"\n{title}")
    print(f"  sizes   {names[0]}: {stats['sizes'][names[0]]}   "
          f"{names[1]}: {stats['sizes'][names[1]]}")
    print(f"  within  {names[0]}: {stats['within_A']}   "
          f"{names[1]}: {stats['within_B']}")
    print(f"  between min {min(stats['between'])} / mean "
          f"{sum(stats['between'])/len(stats['between']):.1f} / "
          f"max {max(stats['between'])}")
    print(f"  composition (mean between - mean within): {observed:+.1f}")
    print(f"    null {stats['composition_null']}")
    print(f"    p = {p_composition:.2f}  (floor {floor:.2f})")
    for label, item in quality_out.items():
        print(f"  {label}: {names[0]} {item[names[0] + '_mean']}  "
              f"{names[1]} {item[names[1] + '_mean']}  "
              f"delta {item['delta']:+.1f}  two-sided p {item['p']:.2f}")
    print(f"  verdict: {verdict}")
    return stats
