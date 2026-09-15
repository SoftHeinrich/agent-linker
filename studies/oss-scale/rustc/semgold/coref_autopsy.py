#!/usr/bin/env python3
"""Where do the coreference stage's links die? Reads the run's own phase state.

The stage records what its resolver proposed and what its judge approved, so the loss can be
split into the two places it can happen: proposals never made, and proposals thrown away.
Compared against the judge in `topic_probe.py`, which sees the anchor sentence.

usage: coref_autopsy.py <run_dir> [--probe out/topic_probe_k3.csv]
"""
from __future__ import annotations

import argparse
import collections
import csv
import pickle
from pathlib import Path

from common import OUT


def gold_set(tiers=("gold", "gold_plus_only")):
    rows = list(csv.DictReader(open(OUT / "semantic_labels.csv")))
    return ({(int(r["sentence"]), r["crate"]) for r in rows if r["tier"] in tiers},
            {(int(r["sentence"]), r["crate"]) for r in rows if r["tier"] == "refers"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--probe", default=str(OUT / "topic_probe_k3.csv"))
    ap.add_argument("--sentences", type=int, default=1762)
    args = ap.parse_args()
    gold, refers = gold_set()
    state = pickle.load(open(Path(args.run) /
                             "phase_states/s_linker110/openai/sentences/linker_coreference.pkl", "rb"))
    fb = state["feedback"]
    cand = {(c["sentence"], c["component"]) for c in fb["candidates"]}
    acc = {(a["sentence"], a["component"]) for a in fb["accepted"]}
    rej = cand - acc
    print(f"resolver proposed {len(cand)} pairs on {len({s for s, _ in cand})} of {args.sentences} sentences "
          f"({len({s for s, _ in cand})/args.sentences:.3f})")
    print(f"  by component: {collections.Counter(c for _, c in cand).most_common(6)}")
    print(f"  correct among proposals: {len(cand & gold)} ({len(cand & gold)/len(cand):.3f}); "
          f"REFERS {len(cand & refers)}")
    print(f"judge approved {len(acc)} -> correct {len(acc & gold)} ({len(acc & gold)/max(1,len(acc)):.3f})")
    print(f"judge rejected {len(rej)} -> correct {len(rej & gold)} ({len(rej & gold)/max(1,len(rej)):.3f})")
    print(f"  correct links the judge threw away: {len(rej & gold)}")
    print("  -> the judge keeps the same hit rate it was given: it removes volume, not error"
          if abs(len(acc & gold)/max(1, len(acc)) - len(cand & gold)/len(cand)) < 0.05 else "")

    try:
        rows = list(csv.DictReader(open(args.probe)))
    except FileNotFoundError:
        return
    n = len(rows)
    g = sum(int(r["in_gold"]) for r in rows)
    a = [r for r in rows if int(r["approved"])]
    ag = sum(int(r["in_gold"]) for r in a)
    r_ = [r for r in rows if not int(r["approved"])]
    rg = sum(int(r["in_gold"]) for r in r_)
    print(f"\nfor contrast, the topic-propagation judge (same model, anchor sentence in the prompt):")
    print(f"  proposals {n}, correct {g} ({g/n:.3f})")
    print(f"  approved {len(a)} -> correct {ag} ({ag/len(a):.3f}); rejected {len(r_)} -> correct {rg} ({rg/len(r_):.3f})")
    print(f"  lift over its own base rate: {(ag/len(a))/(g/n):.1f}x; it keeps {ag/g:.3f} of the correct proposals")


if __name__ == "__main__":
    main()
