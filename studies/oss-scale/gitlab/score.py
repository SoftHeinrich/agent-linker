#!/usr/bin/env python3
"""Score linker links against a gold file for the GitLab document.

  score.py <links.csv>... [--gold out/gold_semantic.csv] [--show-fp N]

links.csv is the runner's format (sentence,component_id,component_name,confidence,source).
Several links files = several runs; each is scored and the mean is reported.  Views:
all stages, each stage, all minus partial-name.  Recall is split by whether the sentence
names the component verbatim (explicit) or not (implicit), and a per-component table is
printed for the mean run.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
import os as _os
DIR = Path(_os.environ.get("OSS_DIR", HERE)).resolve()
DATA = DIR / "data"


def load_links(path):
    by_src = collections.defaultdict(set)
    with open(path) as h:
        for r in csv.DictReader(h):
            by_src[r.get("source", "")].add((int(r["sentence"]), r["component_id"]))
    return by_src


def load_gold(path):
    with open(path) as h:
        return {(int(r["sentence"]), r["modelElementID"]) for r in csv.DictReader(h)}


def prf(pred, gold):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return tp, p, r, f


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("links", nargs="+")
    ap.add_argument("--gold", default=str(DIR / "out" / "gold_semantic.csv"))
    ap.add_argument("--refers", default=str(DIR / "out" / "semantic_labels.csv"), help="labels csv; REFERS pairs not counted as FP in lenient P")
    ap.add_argument("--show-fp", type=int, default=0)
    ap.add_argument("--show-fn", type=int, default=0)
    args = ap.parse_args()

    sents = DATA.joinpath("sentences.txt").read_text().splitlines()
    comps = {c["id"]: c["name"] for c in json.load(open(DATA / "components.json"))}
    gold = load_gold(args.gold)
    explicit = {(s, c) for s, c in gold if comps.get(c, "").lower() in sents[s - 1].lower()}
    implicit = gold - explicit
    refers = set()
    if Path(args.refers).exists():
        with open(args.refers) as h:
            refers = {(int(r["sentence"]), r["component"]) for r in csv.DictReader(h) if r["tier"] == "refers"}
    print(f"gold {args.gold}: {len(gold)} pairs on {len({s for s, _ in gold})} sentences, "
          f"{len(explicit)} explicit / {len(implicit)} implicit; refers {len(refers)}")

    views = collections.OrderedDict()
    per_run = []
    for path in args.links:
        by_src = load_links(path)
        allp = set().union(*by_src.values()) if by_src else set()
        v = collections.OrderedDict()
        v["all stages"] = allp
        for s in sorted(by_src):
            v[f"stage {s}"] = by_src[s]
        if "partial_name" in by_src:
            v["minus partial_name"] = allp - by_src["partial_name"]
        per_run.append(v)
        for k in v:
            views.setdefault(k, [])
    print(f"\n{'view':22s} {'links':>6s} {'TP':>5s} {'P':>6s} {'R':>6s} {'F1':>6s} {'P len':>6s} {'R expl':>6s} {'R impl':>6s}   (mean over {len(per_run)} run(s))")
    for k in views:
        rows = []
        for v in per_run:
            pred = v.get(k, set())
            tp, p, r, f = prf(pred, gold)
            plen = len(pred & (gold | refers)) / len(pred) if pred else 0.0
            re_ = len(pred & explicit) / len(explicit) if explicit else 0.0
            ri = len(pred & implicit) / len(implicit) if implicit else 0.0
            rows.append((len(pred), tp, p, r, f, plen, re_, ri))
        m = [sum(x[i] for x in rows) / len(rows) for i in range(8)]
        spread = f"  [F1 {min(x[4] for x in rows):.3f}–{max(x[4] for x in rows):.3f}]" if len(rows) > 1 else ""
        print(f"{k:22s} {m[0]:6.1f} {m[1]:5.1f} {m[2]:6.3f} {m[3]:6.3f} {m[4]:6.3f} {m[5]:6.3f} {m[6]:6.3f} {m[7]:6.3f}{spread}")

    # per-component on the first run, all stages
    v = per_run[0]["all stages"]
    print("\nper component (run 1, all stages): gold / links / TP")
    gc = collections.Counter(c for _, c in gold)
    lc = collections.Counter(c for _, c in v)
    tc = collections.Counter(c for _, c in v & gold)
    for c in sorted(comps, key=lambda c: -gc[c]):
        if gc[c] or lc[c]:
            print(f"  {c:30s} {gc[c]:4d} {lc[c]:4d} {tc[c]:4d}")
    if args.show_fp:
        print("\nfalse positives (run 1, all stages):")
        for s, c in sorted(v - gold)[: args.show_fp]:
            print(f"  S{s} -> {c}{' [REFERS]' if (s, c) in refers else ''}: {sents[s - 1][:150]}")
    if args.show_fn:
        print("\nfalse negatives (run 1, all stages):")
        for s, c in sorted(gold - v)[: args.show_fn]:
            print(f"  S{s} -> {c}{' [explicit]' if (s, c) in explicit else ''}: {sents[s - 1][:150]}")


if __name__ == "__main__":
    main()
