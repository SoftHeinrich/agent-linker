"""How much of the semantic gold is reachable by coreference, and by what window?

The s110 coreference stage resolves a referring expression to a component that an
earlier sentence *names*.  This measures the ceiling of that design on the semantic
gold: for every implicit gold pair (the sentence does not name the crate), how far
back is the nearest sentence that does?  Anything with no preceding mention inside the
chapter is unreachable by any antecedent-based resolver, however good.

usage: coref_headroom.py [--gold gold_plus|gold|3way] [--links links.csv]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re

from common import DATA, OUT, load_components, load_sentences

ANAPHOR = re.compile(r"^(it|this|these|those|they|them|their|its|such|that|here|the same)\b", re.I)


def prf(pred: set, gold: set):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return tp, p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="gold_plus")
    ap.add_argument("--links", default="")
    args = ap.parse_args()

    labels = list(csv.DictReader(open(OUT / "semantic_labels.csv")))
    if args.gold in ("gold", "gold_plus"):
        tiers = {"gold": {"gold"}, "gold_plus": {"gold", "gold_plus_only"}}[args.gold]
        gold = {(int(l["sentence"]), l["crate"]) for l in labels if l["tier"] in tiers}
    else:
        gold = {(int(x["sentence"]), x["modelElementID"])
                for x in csv.DictReader(open(OUT / "gold_semantic_3way.csv"))}

    rows = load_sentences()
    by_num = {r["number"]: r for r in rows}
    crates = load_components()
    order = [r["number"] for r in rows]
    chapter = {r["number"]: r["chapter"] for r in rows}
    # position of each sentence inside its chapter, for window arithmetic
    pos, seen = {}, collections.Counter()
    for n in order:
        seen[chapter[n]] += 1
        pos[n] = seen[chapter[n]]

    names = {c: re.compile(rf"\b{re.escape(c)}\b") for c in crates}
    mentions = collections.defaultdict(list)  # crate -> [sentence numbers naming it]
    for n in order:
        text = by_num[n]["text"]
        for c, rx in names.items():
            if rx.search(text):
                mentions[c].append(n)

    explicit = {(n, c) for (n, c) in gold if names[c].search(by_num[n]["text"])}
    implicit = gold - explicit

    print(f"gold {args.gold}: {len(gold)} pairs, explicit {len(explicit)}, implicit {len(implicit)}")

    # ---- 1. distance from an implicit gold sentence back to the nearest naming sentence
    dist = {}
    for (n, c) in implicit:
        prev = [m for m in mentions[c] if m < n and chapter[m] == chapter[n]]
        dist[(n, c)] = (pos[n] - pos[prev[-1]]) if prev else None
    hist = collections.Counter(dist.values())
    print("\n== implicit gold: sentences back to nearest preceding mention of the same crate (same chapter)")
    cum = 0
    for k in [1, 2, 3, 5, 10, 20, 50]:
        cum = sum(v for d, v in hist.items() if d is not None and d <= k)
        print(f"  within {k:2d} sentences: {cum:5d}  ceiling recall on implicit {cum/len(implicit):.3f}"
              f"  on all gold {(cum+len(explicit))/len(gold):.3f}")
    anywhere = sum(v for d, v in hist.items() if d is not None)
    print(f"  anywhere earlier in chapter: {anywhere:5d}  ({anywhere/len(implicit):.3f} of implicit)")
    print(f"  NO preceding mention in chapter: {hist[None]:5d}  ({hist[None]/len(implicit):.3f}) "
          f"-> unreachable by any antecedent-based coreference")
    # forward-only mention (crate named later in the chapter but not before)
    fwd = sum(1 for (n, c) in implicit if dist[(n, c)] is None
              and any(m > n and chapter[m] == chapter[n] for m in mentions[c]))
    never = sum(1 for (n, c) in implicit
                if not any(chapter[m] == chapter[n] for m in mentions[c]))
    print(f"    of those: named only later in the chapter {fwd}, never named in the chapter {never}")

    # ---- 2. surface: does the implicit gold sentence even open with a referring expression?
    ana = collections.Counter()
    for (n, c) in implicit:
        ana[bool(ANAPHOR.match(by_num[n]["text"].strip()))] += 1
    reach = {(n, c) for (n, c) in implicit if dist[(n, c)] is not None and dist[(n, c)] <= 3}
    ana_reach = sum(1 for (n, c) in reach if ANAPHOR.match(by_num[n]["text"].strip()))
    print(f"\n== surface of implicit gold sentences")
    print(f"  opens with a pronoun/demonstrative: {ana[True]} ({ana[True]/len(implicit):.3f})")
    print(f"  of the {len(reach)} pairs reachable within 3 sentences, {ana_reach} open with one "
          f"({ana_reach/len(reach):.3f})")

    # ---- 3. sticky-topic oracle: propagate every naming sentence forward K sentences
    print(f"\n== sticky-topic baseline: every sentence that names a crate propagates it forward K sentences")
    print(f"  {'K':>4s} {'links':>7s} {'TP':>6s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    for k in [0, 1, 2, 3, 5, 10]:
        pred = set()
        for c, ms in mentions.items():
            for m in ms:
                for n in order:
                    if chapter[n] == chapter[m] and 0 <= pos[n] - pos[m] <= k:
                        pred.add((n, c))
        tp, p, r, f = prf(pred, gold)
        print(f"  {k:>4d} {len(pred):>7d} {tp:>6d} {p:>6.3f} {r:>6.3f} {f:>6.3f}")
    # chapter-level topic slot: a crate named anywhere in the chapter is assigned to all of it
    pred = {(n, c) for c, ms in mentions.items() for m in ms
            for n in order if chapter[n] == chapter[m]}
    tp, p, r, f = prf(pred, gold)
    print(f"  chapter {len(pred):>7d} {tp:>6d} {p:>6.3f} {r:>6.3f} {f:>6.3f}")

    # ---- 4. what the linker's coreference stage actually produced
    if args.links:
        coref = set()
        for rec in csv.DictReader(open(args.links)):
            if rec["source"] == "coreference":
                coref.add((int(rec["sentence"]), rec["component_id"]))
        tp, p, r, f = prf(coref, gold)
        new = {pair for pair in coref}
        print(f"\n== linker coreference stage: {len(coref)} links, TP {tp}, P {p:.3f}, R {r:.3f}")
        cov = sum(1 for pair in coref if pair in implicit)
        print(f"  of them on implicit gold pairs: {cov}")
        ceiling3 = len(reach)
        print(f"  it takes {tp} of the {ceiling3} pairs a 3-sentence-window antecedent resolver could take "
              f"({tp/ceiling3:.3f})")

    # ---- 5. how the two annotators justified implicit gold with no antecedent
    json.dump({"gold": args.gold, "n_gold": len(gold), "explicit": len(explicit),
               "implicit": len(implicit),
               "no_preceding_mention": hist[None],
               "reachable_within_3": len(reach),
               "reachable_in_chapter": anywhere},
              open(OUT / f"coref_headroom_{args.gold}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
