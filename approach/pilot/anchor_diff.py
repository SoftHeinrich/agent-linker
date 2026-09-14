"""What the `anchors` block restrains, case by case. No LLM calls.

`pilot/s121_ablations.py --dump` records each arm's kept pairs per sample per project.
This reads two arms out of one dump and prints the cases they disagree on, with the
evidence the match computed for each, so "anchors are worth ~12 spurious a run" becomes
a list of sentences a reader can check.

The question it exists to answer: **is the anchor block carrying a fact the judge cannot
get anywhere else, or is it patching a rule that does not say enough?** Those have
different repairs — the first is not removable, the second is a clause.

    ../.venv/bin/python pilot/anchor_diff.py ../results/s121_ablations/dump_terra.json
    ../.venv/bin/python pilot/anchor_diff.py <dump> --arms head noanchor
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

from s121_ablations import DEFAULT_RUN, load, probe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump")
    parser.add_argument("--arms", nargs=2, default=["head", "noanchor"])
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--limit", type=int, default=8,
                        help="cases printed per project per direction")
    args = parser.parse_args()
    left, right = args.arms

    with open(args.dump) as handle:
        dump = json.load(handle)

    #: How often each pair is kept by each arm, over the dump's samples.
    kept = collections.defaultdict(collections.Counter)
    samples = collections.Counter()
    for sample in dump:
        for project, arms in dump[sample].items():
            if left not in arms or right not in arms:
                continue
            samples[project] += 1
            for arm in (left, right):
                for pair in arms[arm]:
                    kept[(project, arm)][tuple(pair)] += 1

    totals = collections.Counter()
    by_field = collections.defaultdict(collections.Counter)

    for project in sorted(samples):
        data = load(project, args.run)
        linker = probe("head", data)
        sent_map = data["sent_map"]
        cases = {(c.sentence_number, c.component_id): c
                 for c in linker._name_candidates(
                     data["sentences"], data["components"],
                     data["name_to_id"], sent_map)}
        runs = samples[project]
        mine, theirs = kept[(project, left)], kept[(project, right)]

        #: A pair the right arm keeps more often than the left. Counted in samples, not
        #: in pairs, because a judge that keeps a pair in one sample of three has not
        #: made the same error as one that keeps it in three.
        deltas = []
        for pair in set(mine) | set(theirs):
            gap = theirs[pair] - mine[pair]
            if gap:
                deltas.append((gap, pair))
        added = sorted((d for d in deltas if d[0] > 0), reverse=True)
        removed = sorted(d for d in deltas if d[0] < 0)

        print(f"\n=== {project}: {runs} samples, {len(added)} pairs {right} keeps more "
              f"often, {len(removed)} it keeps less often ===")
        for direction, rows in (("+", added), ("-", removed)):
            for gap, pair in rows[:args.limit]:
                candidate = cases.get(pair)
                if candidate is None:
                    print(f"  {direction}{abs(gap)}/{runs} {pair} (not a head case)")
                    continue
                evidence = linker._union_evidence(
                    candidate, data["components"], sent_map)
                mark = "GOLD" if tuple(pair) in {tuple(p) for p in data["gold"]} else "    "
                anchors = len(evidence["anchors"])
                totals[f"{direction}{mark.strip() or 'SPURIOUS'}"] += abs(gap)
                if direction == "+":
                    by_field[evidence["naming"]][mark.strip() or "SPURIOUS"] += abs(gap)
                    by_field["anchors shown"][
                        "some" if anchors else "none"] += abs(gap)
                print(f"  {direction}{abs(gap)}/{runs} {mark} "
                      f'"{evidence["span"]}" -> {candidate.component_name} '
                      f'(writes={evidence["naming"]}, '
                      f'alternatives={evidence["alternatives"] or "-"}, '
                      f'anchors={anchors})')
                print(f"        {candidate.sentence_text[:150]}")

    print(f"\n{right} minus {left}, summed over every sample of every project "
          f"(sample-counts, not pairs):")
    for key in sorted(totals):
        print(f"  {key:<12} {totals[key]}")
    print(f"\nwhat {right} keeps more often, by the evidence of the case:")
    for field in sorted(by_field):
        counts = ", ".join(f"{k} {v}" for k, v in sorted(by_field[field].items()))
        print(f"  {field:<16} {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
