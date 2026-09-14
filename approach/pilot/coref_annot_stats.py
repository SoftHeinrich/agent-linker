"""Paired per-sample statistics for the shortlist-annotation arms.

The round's tables report means. This reports the PAIRED deltas -- both arms scored on
the same sample, which is how this branch reads a delta -- with an exact sign-flip
permutation test over the 2^n assignments, and the link-level accounting that explains
the sign.

    $PY pilot/coref_annot_stats.py ../results/coref_annot_terra_20260914/dump.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from score_runs import GOLD, scores  # noqa: E402

METRICS = ("TP", "FP", "macro F1", "macro F2")


def sign_flip_p(deltas):
    """Two-sided exact sign-flip permutation p for the mean of paired deltas.

    With n paired samples there are 2^n sign assignments; the p is the share whose
    mean absolute value reaches the observed one. At n = 3 the floor is 0.25, which is
    why a mixed-sign arm here cannot read below 0.75 however large the mean looks.
    """
    n = len(deltas)
    observed = abs(sum(deltas) / n)
    hits = sum(1 for signs in itertools.product((1, -1), repeat=n)
               if abs(sum(s * d for s, d in zip(signs, deltas)) / n) >= observed - 1e-12)
    return hits / 2 ** n


def sets_of(data, sample, arm):
    return {(project, pair[0], pair[1])
            for project, arms in data[sample].items() for pair in arms[arm]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    parser.add_argument("--base", default="head")
    args = parser.parse_args()

    data = json.load(open(args.dump))
    samples = sorted(data)
    arms = [a for a in next(iter(data[samples[0]].values())) if a != args.base]

    print(f"=== paired per-sample deltas against `{args.base}`, "
          f"n = {len(samples)} ===")
    for arm in arms:
        print(f"\n  {arm}")
        rows = []
        for sample in samples:
            base = scores(sets_of(data, sample, args.base))
            other = scores(sets_of(data, sample, arm))
            rows.append({k: other[k] - base[k] for k in METRICS})
            print("    " + sample + "  " + "  ".join(
                f"{k} {rows[-1][k]:+7.2f}" for k in METRICS))
        print("    " + "-" * 62)
        print("    mean  " + "  ".join(
            f"{k} {sum(r[k] for r in rows) / len(rows):+7.2f}" for k in METRICS))
        print("    p     " + "  ".join(
            f"{k} {sign_flip_p([r[k] for r in rows]):7.2f}" for k in METRICS))
        print("    signs " + "  ".join(
            f"{k} {sum(1 for r in rows if r[k] > 0)}+/"
            f"{sum(1 for r in rows if r[k] < 0)}-/"
            f"{sum(1 for r in rows if r[k] == 0)}=" for k in METRICS))

    print(f"\n=== the exchange rate: what each arm adds and drops, per run ===")
    print(f"  {'arm':<14}{'+gold':>8}{'+FP':>7}{'-gold':>8}{'-FP':>7}"
          f"{'net gold':>10}{'net FP':>8}")
    for arm in arms:
        add_g = add_f = drop_g = drop_f = 0
        for sample in samples:
            base = sets_of(data, sample, args.base)
            other = sets_of(data, sample, arm)
            for pair in other - base:
                if pair in GOLD:
                    add_g += 1
                else:
                    add_f += 1
            for pair in base - other:
                if pair in GOLD:
                    drop_g += 1
                else:
                    drop_f += 1
        n = len(samples)
        print(f"  {arm:<14}{add_g / n:>8.2f}{add_f / n:>7.2f}{drop_g / n:>8.2f}"
              f"{drop_f / n:>7.2f}{(add_g - drop_g) / n:>10.2f}"
              f"{(add_f - drop_f) / n:>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
