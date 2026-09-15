"""Where the annotation's few changed links land: the composed dumps, differenced.

The stage table says the arms move 30% of the resolver's proposals and nothing of its
net contribution. This says which links DID move, per arm and per sample, with the gold
label on each -- the check that the near-null is a near-null everywhere rather than two
large effects cancelling.

    $PY pilot/coref_annot_diff.py ../results/coref_annot_terra_20260914/dump.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from score_runs import GOLD  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    parser.add_argument("--base", default="head")
    args = parser.parse_args()

    data = json.load(open(args.dump))
    samples = sorted(data)
    arms = [a for a in next(iter(data[samples[0]].values())) if a != args.base]
    tally = collections.defaultdict(collections.Counter)

    print(f"=== every link the arm adds or drops against `{args.base}` ===")
    for arm in arms:
        for sample in samples:
            for project in sorted(data[sample]):
                rows = data[sample][project]
                base = {(project, p[0], p[1]) for p in rows[args.base]}
                other = {(project, p[0], p[1]) for p in rows[arm]}
                for pair in sorted(other - base):
                    mark = "GOLD" if pair in GOLD else "FP  "
                    tally[arm][f"+{mark.strip()}"] += 1
                    print(f"  {arm:<13} {sample} {project:<14} +S{pair[1]:<4} {mark}")
                for pair in sorted(base - other):
                    mark = "GOLD" if pair in GOLD else "FP  "
                    tally[arm][f"-{mark.strip()}"] += 1
                    print(f"  {arm:<13} {sample} {project:<14} -S{pair[1]:<4} {mark}")

    print(f"\n=== totals over {len(samples)} samples ===")
    print(f"  {'arm':<13}{'+gold':>8}{'+FP':>7}{'-gold':>8}{'-FP':>7}")
    for arm in arms:
        print(f"  {arm:<13}{tally[arm]['+GOLD']:>8}{tally[arm]['+FP']:>7}"
              f"{tally[arm]['-GOLD']:>8}{tally[arm]['-FP']:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
