#!/usr/bin/env python3
"""Does the semantic-gold recipe recover the human MAINTAINERS assignment?

Per sentence: is the file's owning subsystem among the ABOUT labels?
Per document: is it the most-voted subsystem?
Baseline: how often a *non-owner* candidate gets ABOUT, which is what a labeller that says
yes to everything would score.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="", help="audit directory holding out/ (default: this one)")
    ap.add_argument("--tag", default="terra")
    args = ap.parse_args()
    global OUT
    if args.dir:
        OUT = Path(args.dir).resolve() / "out"
    docs = {d["path"]: d for d in json.load(open(OUT / "dataset.json"))}
    ann = json.load(open(OUT / f"annotations_{args.tag}.json"))
    labels = ann["labels"]
    ranks = [r["owner_rank"] for r in ann["retrieval"]]
    inK = sum(1 for r in ranks if r is not None and r < 12)
    subs = json.load(open(OUT / "subsystems.json"))
    print(f"BM25 retrieval of the human owner (universe {len(subs)} components): "
          f"top-12 {inK}/{len(ranks)} = {inK/len(ranks):.3f}; "
          f"top-1 {sum(1 for r in ranks if r == 0)/len(ranks):.3f}; "
          f"median rank {sorted(r if r is not None else 9999 for r in ranks)[len(ranks)//2]}")

    dr = ann.get("doc_retrieval", [])
    if dr:
        d12 = sum(1 for r in dr if r["owner_rank"] is not None and r["owner_rank"] < 12)
        print(f"  same query pooled over the whole document: top-12 {d12}/{len(dr)} = {d12/len(dr):.3f}"
              f"  (ranks {[r['owner_rank'] for r in dr]})")

    per_doc = collections.defaultdict(lambda: collections.Counter())
    hit = miss = refers = 0
    empty = 0
    for key, lab in labels.items():
        path, i = key.rsplit("#", 1)
        owner = docs[path]["owner"]
        about = set(lab["about"])
        per_doc[path].update(about)
        if not about:
            empty += 1
        if owner in about:
            hit += 1
        elif owner in set(lab["refers"]):
            refers += 1
            miss += 1
        else:
            miss += 1
    n = hit + miss
    print(f"\nsentences labelled: {n} (no ABOUT at all: {empty})")
    print(f"  owner among ABOUT:   {hit}/{n} = {hit/n:.3f}")
    print(f"  owner only REFERS:   {refers}")
    print(f"  owner absent:        {miss - refers}")
    print(f"  of the sentences that got any ABOUT: {hit}/{n-empty} = {hit/(n-empty):.3f}")

    # what a yes-to-everything labeller would get: mean ABOUT rate of a non-owner candidate
    tot_about = sum(len(l["about"]) for l in labels.values())
    print(f"  ABOUT labels per sentence: {tot_about/n:.2f} (candidates shown: 12)")

    print(f"\n{'document':52s} {'owner-ABOUT':>11s} {'top voted subsystem':>34s}")
    correct_doc = 0
    for path, cnt in per_doc.items():
        owner = docs[path]["owner"]
        sents = [k for k in labels if k.rsplit("#", 1)[0] == path]
        h = sum(1 for k in sents if owner in labels[k]["about"])
        top = cnt.most_common(1)[0] if cnt else ("-", 0)
        correct_doc += top[0] == owner
        print(f"{path[-52:]:52s} {h:>4d}/{len(sents):<6d} {top[0][:30]:>30s} {top[1]:>3d}")
    print(f"\ndocuments whose most-voted subsystem is the human owner: {correct_doc}/{len(per_doc)}")
    json.dump({"sentences": n, "owner_about": hit, "owner_refers": refers,
               "no_about": empty, "docs": len(per_doc), "docs_top_owner": correct_doc,
               "retrieval_top12": inK / len(ranks)},
              open(OUT / f"score_{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
