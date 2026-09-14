"""Why the anchor arms split by model: the flips, profiled against their population.

`anchor_diff.py` lists the cases two arms disagree on. This asks what distinguishes them
from the cases both arms agree on, over the properties an anchor block could plausibly
act through:

  anchors      how many naming sentences the case carries
  anchor_chars how much text that block is (the crowding hypothesis: a long block is
               competing with the case's own sentence for the judge's attention)
  qualified    whether the surface sits inside a dotted identifier, which is the
               population `QUALIFIED_CLAUSE` speaks about
  first        whether this case PRINTS the block or back-references an earlier case
               ("as shown in Case N"), since only the first case in a batch pays for it

A flip population that looks like its base population says the property is not the
mechanism. No LLM calls.

    ../.venv/bin/python pilot/anchor_why.py ../results/s121_ablations/dump_terra_anchor2.json \\
        --arms head anchor_count
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from s121_ablations import DEFAULT_RUN, load, probe  # noqa: E402


def profile(linker, candidate, evidence, data, in_qualified):
    return {
        "anchors": len(evidence["anchors"]),
        "anchor_chars": sum(len(a) for a in evidence["anchors"]),
        "qualified": in_qualified,
        "row": evidence["naming"],
    }


def describe(rows, label):
    if not rows:
        print(f"  {label:<26} (none)")
        return
    anchors = statistics.fmean(r["anchors"] for r in rows)
    chars = statistics.fmean(r["anchor_chars"] for r in rows)
    qualified = sum(1 for r in rows if r["qualified"]) / len(rows)
    by_row = collections.Counter(r["row"] for r in rows)
    print(f"  {label:<26} n={len(rows):3d}  anchors {anchors:4.1f}  "
          f"anchor_chars {chars:6.0f}  qualified {qualified:5.1%}  "
          f"{dict(by_row)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump")
    parser.add_argument("--arms", nargs=2, default=["head", "anchor_count"])
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    left, right = args.arms

    with open(args.dump) as handle:
        dump = json.load(handle)

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

    buckets = collections.defaultdict(list)
    for project in sorted(samples):
        data = load(project, args.run)
        linker = probe("head", data)
        gold = {tuple(p) for p in data["gold"]}
        candidates = linker._name_candidates(
            data["sentences"], data["components"], data["name_to_id"],
            data["sent_map"])
        mine, theirs = kept[(project, left)], kept[(project, right)]
        for candidate in candidates:
            pair = (candidate.sentence_number, candidate.component_id)
            evidence = linker._union_evidence(
                candidate, data["components"], data["sent_map"])
            span = evidence["span"]
            start = candidate.sentence_text.find(span)
            in_qualified = start >= 0 and linker._in_dotted_path(
                candidate.sentence_text, start, start + len(span))
            row = profile(linker, candidate, evidence, data, in_qualified)
            row["gold"] = pair in gold
            gap = theirs[pair] - mine[pair]
            if gap > 0:
                buckets[f"{right} keeps MORE / gold" if row["gold"]
                        else f"{right} keeps MORE / spurious"].append(row)
            elif gap < 0:
                buckets[f"{right} keeps LESS / gold" if row["gold"]
                        else f"{right} keeps LESS / spurious"].append(row)
            else:
                buckets["agreed"].append(row)

    print(f"\n{Path(args.dump).name}: {right} against {left}, "
          f"every case of every project, profiled\n")
    for label in sorted(buckets):
        describe(buckets[label], label)

    #: The crowding hypothesis in one number: if a long anchor block is what the arm
    #: relieves, the cases it changes carry more anchor text than the cases it does not.
    changed = [r for label, rows in buckets.items() if label != "agreed" for r in rows]
    agreed = buckets["agreed"]
    if changed and agreed:
        print(f"\n  anchor_chars, changed {statistics.fmean(r['anchor_chars'] for r in changed):.0f} "
              f"vs agreed {statistics.fmean(r['anchor_chars'] for r in agreed):.0f}")
        print(f"  qualified,    changed "
              f"{sum(1 for r in changed if r['qualified']) / len(changed):.1%} "
              f"vs agreed {sum(1 for r in agreed if r['qualified']) / len(agreed):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
