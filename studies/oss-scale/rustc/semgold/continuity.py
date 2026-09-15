#!/usr/bin/env python3
"""Is the answer key actually continuous? Does a sentence with no name in it belong to the
same component as its neighbours, and how well-structured is the document really?

Measured on the gold itself, with no linker and no LLM involved:
  neighbour agreement  for each gold pair, does the previous / next sentence carry it too
  runs                 how long the stretches of consecutive sentences on one component are
  isolation            gold pairs whose component appears on neither neighbour
  chapter shape        how concentrated each chapter is on one component
"""
from __future__ import annotations

import argparse
import collections
import csv
import json

from common import OUT, load_components, load_sentences
from surface import strata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="gold_plus")
    ap.add_argument("--links", default="", help="also run the chapter-topic baselines against this links.csv")
    args = ap.parse_args()
    labels = list(csv.DictReader(open(OUT / "semantic_labels.csv")))
    tiers = {"gold": {"gold"}, "gold_plus": {"gold", "gold_plus_only"}}[args.gold]
    gold = {(int(l["sentence"]), l["crate"]) for l in labels if l["tier"] in tiers}
    rows = load_sentences()
    crates = load_components()
    strat = strata(gold, rows, crates)
    order = [r["number"] for r in rows]
    chapter = {r["number"]: r["chapter"] for r in rows}
    idx = {n: i for i, n in enumerate(order)}
    per_sent = collections.defaultdict(set)
    for n, c in gold:
        per_sent[n].add(c)

    def neighbour(n, delta):
        i = idx[n] + delta
        if 0 <= i < len(order) and chapter[order[i]] == chapter[n]:
            return per_sent[order[i]]
        return None

    print(f"gold {args.gold}: {len(gold)} pairs on {len(per_sent)} sentences\n")
    print(f"{'stratum':12s} {'pairs':>6s} {'prev has it':>12s} {'next has it':>12s} "
          f"{'either':>8s} {'neither':>8s} {'prev unlabelled':>16s}")
    for st in ("verbatim", "name-echo", "no-surface", "ALL"):
        sel = [p for p in gold if st == "ALL" or strat[p] == st]
        prev = nxt = both = nei = blank = 0
        for (n, c) in sel:
            p, x = neighbour(n, -1), neighbour(n, +1)
            hp, hx = (p is not None and c in p), (x is not None and c in x)
            prev += hp
            nxt += hx
            both += hp or hx
            nei += not (hp or hx)
            blank += p is not None and not p
        k = len(sel)
        print(f"{st:12s} {k:>6d} {prev/k:>12.3f} {nxt/k:>12.3f} {both/k:>8.3f} "
              f"{nei/k:>8.3f} {blank/k:>16.3f}")

    # runs of consecutive sentences carrying the same component
    runs = collections.Counter()
    for c in {c for _, c in gold}:
        cur = 0
        prev_i = None
        for n in order:
            if (n, c) in gold and (prev_i is None or idx[n] == prev_i + 1):
                cur += 1
            elif (n, c) in gold:
                runs[cur] += 1 if cur else 0
                cur = 1
            else:
                if cur:
                    runs[cur] += 1
                cur = 0
            if (n, c) in gold:
                prev_i = idx[n]
        if cur:
            runs[cur] += 1
    tot_runs = sum(runs.values())
    covered = sum(k * v for k, v in runs.items())
    print(f"\nstretches of consecutive sentences on one component: {tot_runs} stretches "
          f"covering {covered} pairs; mean length {covered/tot_runs:.2f}")
    print("  length: " + "  ".join(f"{k}:{runs[k]}" for k in sorted(runs)[:8])
          + f"   >=8: {sum(v for k, v in runs.items() if k >= 8)}")
    singles = runs[1]
    print(f"  stretches of length 1 (an isolated sentence): {singles} "
          f"({singles/tot_runs:.3f} of stretches, {singles/covered:.3f} of pairs)")

    # how concentrated is each chapter
    print(f"\n{'chapter':34s} {'sents':>6s} {'pairs':>6s} {'top component':>26s} {'its share':>10s}")
    shares = []
    for chap in dict.fromkeys(chapter[n] for n in order):
        ns = [n for n in order if chapter[n] == chap]
        cnt = collections.Counter(c for n in ns for c in per_sent[n])
        if not cnt:
            continue
        top, k = cnt.most_common(1)[0]
        share = k / sum(cnt.values())
        shares.append(share)
        print(f"{chap[-34:]:34s} {len(ns):>6d} {sum(cnt.values()):>6d} {top[:26]:>26s} {share:>10.3f}")
    print(f"\nmean share of a chapter's pairs held by its top component: {sum(shares)/len(shares):.3f}")
    json.dump({"gold": args.gold, "runs": dict(runs), "mean_top_share": sum(shares)/len(shares)},
              open(OUT / f"continuity_{args.gold}.json", "w"), indent=1)
    print()
    chapter_topic_baselines(args.links)


def chapter_topic_baselines(links: str = ""):
    """Same question from the other side: if the document is that well-structured, how far
    does "one topic per chapter" alone get you? ORACLE takes each chapter's top component
    from the answer key; the automatic version takes it from the linker's own full-name
    links, so it uses no gold."""
    rows = load_sentences()
    order = [r["number"] for r in rows]
    chap = {r["number"]: r["chapter"] for r in rows}
    labels = list(csv.DictReader(open(OUT / "semantic_labels.csv")))
    gold = {(int(l["sentence"]), l["crate"]) for l in labels
            if l["tier"] in {"gold", "gold_plus_only"}}
    strat = strata(gold, rows, load_components())
    nosurf = {p for p in gold if strat[p] == "no-surface"}
    per = collections.defaultdict(set)
    for n, c in gold:
        per[n].add(c)

    def show(pred, name):
        tp = len(pred & gold)
        print(f"{name:56s} {len(pred):>5d} links  correct {tp/len(pred):.3f}  "
              f"found {tp/len(gold):.3f}  no-name found {len(pred & nosurf)/len(nosurf):.3f}")

    def by_chapter(counts):
        top = {ch: c.most_common(1)[0][0] for ch, c in counts.items() if c}
        return {(n, top[chap[n]]) for n in order if chap[n] in top}

    if links:
        src = collections.defaultdict(set)
        for r in csv.DictReader(open(links)):
            src[r["source"]].add((int(r["sentence"]), r["component_id"]))
        counts = collections.defaultdict(collections.Counter)
        for (n, c) in src["full_name"]:
            if c != "rustc":            # the umbrella entry is never a chapter topic
                counts[chap[n]][c] += 1
        auto = by_chapter(counts)
        show(auto, "chapter topic from the linker's own full-name links")
        show(auto | src["full_name"], "  the same, unioned with the full-name links")
        show(set().union(*src.values()), "the linker, all stages")
    counts = collections.defaultdict(collections.Counter)
    for n in order:
        for c in per[n]:
            counts[chap[n]][c] += 1
    show(by_chapter(counts), "ORACLE chapter topic (taken from the answer key)")


if __name__ == "__main__":
    main()
