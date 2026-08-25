"""Level-1 audit for the contrastive chooser, on the adopted head. No LLM calls.

The consolidation round priced a chooser over the partial-name gate's decisions and
did not build it (`results/consolidation_round/README.md`, question 4). That pricing
was read off `s_linker92a`, the base the round then replaced: `s_linker109` refuses 12
sibling candidates a run before any judge sees them, so the population a chooser would
be asked about is smaller at the head than it was where it was priced. This recomputes
the ceiling on `s_linker110`'s own checkpoints, and adds the two numbers the pricing
did not carry -- how many questions have no gold answer at all, and how much gold sits
inside a group a wrong chooser could destroy.

A *group instance* is the chooser's unit: one sentence, one sibling family (catalog
names sharing a signature word), two or more of whose members the scan proposed there.
One member cannot be chosen between, so a family with a single proposal in a sentence
is not a question and is not counted.

    ../.venv/bin/python pilot/chooser_audit.py [--variant s_linker110]
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

from consolidation_audit import (  # noqa: E402
    RESULTS, load_projects, model_of, siblings_of, stage_of,
)
from design_audit import PROJECTS  # noqa: E402


def runs_of(variant: str):
    for base in sorted(RESULTS.glob(f"*/phase_states/{variant}/openai")):
        if all((base / p / "linker_partial_name.pkl").exists() for p in PROJECTS):
            yield base


def group_instances(data, decisions):
    """(sentence, family) -> [decision, ...] for families with >1 proposal there."""
    names = list(data["name_to_id"])
    family_of = {n: frozenset([n, *siblings_of(n, names)]) for n in names}
    buckets = defaultdict(list)
    for decision in decisions:
        name = data["id_to_name"].get(decision["component_id"], "")
        family = family_of.get(name)
        if family and len(family) > 1:
            buckets[(decision["sentence"], family)].append(decision)
    return {key: rows for key, rows in buckets.items() if len(rows) > 1}


def audit(variant: str):
    projects = load_projects()
    rows = []
    for base in runs_of(variant):
        row = {"run": base.parts[2], "model": model_of(base)}
        counts = defaultdict(int)
        for project, data in projects.items():
            gold = data["gold"]
            stage = stage_of(base, project, "linker_partial_name")
            final = {(l.sentence_number, l.component_id)
                     for l in stage_of(base, project, "final")["final"]}
            decisions = stage["feedback"].get("judge_decisions", [])
            counts["gate cases"] += len(decisions)
            for (_, _family), members in group_instances(data, decisions).items():
                counts["questions"] += 1
                counts["options"] += len(members)
                golds = [d for d in members
                         if (d["sentence"], d["component_id"]) in gold]
                counts["no gold answer"] += not golds
                for decision in members:
                    pair = (decision["sentence"], decision["component_id"])
                    approved = bool(decision.get("approved"))
                    if approved and pair not in gold:
                        counts["removable FP"] += 1
                    if approved and pair in gold:
                        counts["TP at risk"] += 1
                    if not approved and pair in gold and pair not in final:
                        counts["recoverable FN"] += 1
        row.update(counts)
        rows.append(row)
    return rows


def report(rows):
    if not rows:
        print("no recorded runs")
        return
    keys = ["gate cases", "questions", "options", "no gold answer",
            "removable FP", "TP at risk", "recoverable FN"]
    print(f"{'run':<34}{'model':<7}" + "".join(f"{k:>16}" for k in keys))
    for row in rows:
        print(f"{row['run']:<34}{row['model']:<7}"
              + "".join(f"{row.get(k, 0):>16}" for k in keys))
    for model in sorted({r["model"] for r in rows}):
        group = [r for r in rows if r["model"] == model]
        print(f"\n{model}, mean of {len(group)} five-project runs:")
        for key in keys:
            print(f"  {key:<18}{statistics.mean(r.get(key, 0) for r in group):8.1f}")
        opts = statistics.mean(r.get("options", 0) for r in group)
        qs = statistics.mean(r.get("questions", 0) for r in group) or 1
        print(f"  {'options a question':<18}{opts / qs:8.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="s_linker110")
    parser.add_argument("--json")
    args = parser.parse_args()
    rows = audit(args.variant)
    report(rows)
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)
