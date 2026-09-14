"""Which gold links the no-anchor arm loses, and what those cases looked like. No calls.

`score_runs.py` says `s_linker122` costs luna 10.3 true links a run and 8.0 of them are
teammates. This reads the two arms' per-project link CSVs out of the E2E runs and asks
what the lost pairs have in common — which linker found them in the control, what the
match computed for them, and whether the anchor block was the thing that carried them.

A pair is counted by how many of the runs lost it, so a pair lost in three runs of three
is separated from one lost in one: the first is the change, the second is sampling.

    ../.venv/bin/python pilot/noanchor_fn.py luna
    ../.venv/bin/python pilot/noanchor_fn.py terra --project teammates
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import PROJECTS, load_gold  # noqa: E402
from s121_ablations import DEFAULT_RUN, load, probe  # noqa: E402

CONTROL, ARM = "s_linker121", "s_linker122"


def links_of(path):
    """{(sentence, component id): source} out of one arm-project link CSV."""
    out = {}
    if not path.exists():
        return out
    with path.open() as handle:
        for row in csv.DictReader(handle):
            out[(int(row["sentence"]), row["component_id"].strip())] = row["source"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--stamp", default="20260914")
    parser.add_argument("--project", nargs="+", default=sorted(PROJECTS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--knowledge-run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()

    runs = [ROOT.parent / f"results/noanchor_e2e_{args.model}_r{r}_{args.stamp}"
            for r in range(1, args.runs + 1)]

    for project in args.project:
        gold = load_gold(project)
        lost = collections.Counter()
        gained = collections.Counter()
        source_of = {}
        for run in runs:
            control = links_of(run / f"{CONTROL}_{project}_links.csv")
            arm = links_of(run / f"{ARM}_{project}_links.csv")
            if not control or not arm:
                continue
            for pair in (set(control) & gold) - set(arm):
                lost[pair] += 1
                source_of.setdefault(pair, control[pair])
            for pair in (set(arm) & gold) - set(control):
                gained[pair] += 1
                source_of.setdefault(pair, arm[pair])
        if not lost and not gained:
            continue

        data = load(project, args.knowledge_run)
        linker = probe("head", data)
        cases = {(c.sentence_number, c.component_id): c
                 for c in linker._name_candidates(
                     data["sentences"], data["components"],
                     data["name_to_id"], data["sent_map"])}

        print(f"\n=== {project} ({args.model}): {len(lost)} gold pairs lost, "
              f"{len(gained)} gained, over {len(runs)} runs ===")
        rows = collections.Counter()
        for pair, count in sorted(lost.items(), key=lambda kv: -kv[1]):
            candidate = cases.get(pair)
            source = source_of[pair]
            if candidate is None:
                print(f"  -{count}/{len(runs)} S{pair[0]} [{source}] "
                      f"(not a name case -- coreference only)")
                rows[f"{source} / no name case"] += count
                continue
            evidence = linker._union_evidence(
                candidate, data["components"], data["sent_map"])
            span = evidence["span"]
            start = candidate.sentence_text.find(span)
            qualified = start >= 0 and linker._in_dotted_path(
                candidate.sentence_text, start, start + len(span))
            rows[f"{source} / {evidence['naming']}"
                 f"{' / dotted' if qualified else ''}"] += count
            print(f"  -{count}/{len(runs)} S{pair[0]} \"{span}\" -> "
                  f"{candidate.component_name} [{source}] "
                  f"writes={evidence['naming']}, anchors={len(evidence['anchors'])}"
                  f"{', DOTTED' if qualified else ''}")
            print(f"        {candidate.sentence_text[:130]}")
        for pair, count in sorted(gained.items(), key=lambda kv: -kv[1]):
            candidate = cases.get(pair)
            name = candidate.component_name if candidate else "?"
            print(f"  +{count}/{len(runs)} S{pair[0]} -> {name} "
                  f"[{source_of[pair]}] (gained)")
        print(f"\n  lost, by the control's linker and the match "
              f"(run-weighted): {dict(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
