"""Level-1 census of all three judges at the head. No LLM calls.

The branch's design law sets a judge's default polarity from the base rate of the
stream that routes cases to it -- lenient in front of a stream that proposes mostly
gold, strict in front of one that does not. That is a claim about numbers, and the
numbers are in every recorded run: each gate's own decisions, split by gold.

Per gate and run this reports what the gate was handed (`cases`, and the `base rate`
of gold among them), what it kept (`TP`, `FP`), and what it refused (`killed` spurious,
`lost` gold that no later stage recovers). `lost` is the gate's recall headroom and
`FP` its precision headroom; under an F2 budget one point of `lost` is worth about
three of `FP`, so the two columns are not comparable at face value and the report
prints the F2-weighted total that makes them so.

    ../.venv/bin/python pilot/judge_census.py [--variant s_linker110]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from chooser_audit import runs_of  # noqa: E402
from consolidation_audit import load_projects, model_of, stage_of  # noqa: E402
from design_audit import PROJECTS  # noqa: E402

GATES = ("full_name", "partial_name", "coreference")

#: One recovered gold link is worth this many avoided false positives under F2, at
#: the head's operating point (dF2/dTP over dF2/dFP, evaluated at P .90 / R .97).
F2_RATIO = 3.0


def census(variant: str):
    projects = load_projects()
    rows = []
    for base in runs_of(variant):
        for gate in GATES:
            row = {"run": base.parts[2], "model": model_of(base), "gate": gate}
            counts = defaultdict(int)
            for project, data in projects.items():
                gold = data["gold"]
                stage = stage_of(base, project, f"linker_{gate}")
                final = {(l.sentence_number, l.component_id)
                         for l in stage_of(base, project, "final")["final"]}
                for decision in stage["feedback"].get("judge_decisions", []):
                    pair = (decision["sentence"], decision["component_id"])
                    is_gold = pair in gold
                    counts["cases"] += 1
                    counts["gold in"] += is_gold
                    if decision.get("approved"):
                        counts["TP" if is_gold else "FP"] += 1
                    elif is_gold:
                        counts["lost"] += pair not in final
                        counts["rescued"] += pair in final
                    else:
                        counts["killed"] += 1
            row.update(counts)
            rows.append(row)
    return rows


def report(rows):
    if not rows:
        print("no recorded runs")
        return
    keys = ["cases", "TP", "FP", "killed", "lost", "rescued"]
    for model in sorted({r["model"] for r in rows}):
        print(f"\n{model}, mean per five-project run "
              f"({len({r['run'] for r in rows if r['model'] == model})} runs)")
        print(f"  {'gate':<14}{'cases':>7}{'base rate':>11}{'TP':>7}{'FP':>7}"
              f"{'killed':>8}{'lost':>7}{'rescued':>9}{'F2-weighted headroom':>22}")
        for gate in GATES:
            group = [r for r in rows if r["model"] == model and r["gate"] == gate]
            if not group:
                continue
            mean = {k: statistics.mean(r.get(k, 0) for r in group) for k in keys}
            rate = mean["gold in"] if "gold in" in mean else statistics.mean(
                r.get("gold in", 0) for r in group)
            base_rate = rate / mean["cases"] if mean["cases"] else 0.0
            headroom = F2_RATIO * mean["lost"] + mean["FP"]
            print(f"  {gate:<14}{mean['cases']:>7.1f}{base_rate:>11.2f}"
                  f"{mean['TP']:>7.1f}{mean['FP']:>7.1f}{mean['killed']:>8.1f}"
                  f"{mean['lost']:>7.1f}{mean['rescued']:>9.1f}{headroom:>22.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="s_linker110")
    parser.add_argument("--json")
    args = parser.parse_args()
    rows = census(args.variant)
    report(rows)
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)
