"""Split the semantic gold by what surface of the crate name the sentence actually carries.

Three strata, from a generic morphological rule (no project word lists):
  verbatim   the crate id appears literally (`rustc_borrowck`)
  name-echo  a name token echoes: some crate-id token (minus the vendor prefix shared by
             every component) shares a >=4-character prefix with a sentence token, so a
             matcher working on the name alone can still fire ("borrow" ~ borrowck,
             "MIR" ~ mir_transform, "resolution" ~ resolve)
  no-surface nothing of the name is in the sentence; only meaning connects the two
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re

from common import OUT, load_components, load_sentences, tokens

MIN_PREFIX = 4


def vendor_prefix(crates: list[str]) -> str:
    """The leading id token every component shares, if there is one (here: rustc)."""
    firsts = {c.split("_")[0] for c in crates}
    return firsts.pop() if len(firsts) == 1 else ""


def echo(crate_tokens: list[str], sent_tokens: set[str]) -> bool:
    for t in crate_tokens:
        for w in sent_tokens:
            a, b = t[:MIN_PREFIX], w[:MIN_PREFIX]
            if len(t) >= MIN_PREFIX and len(w) >= MIN_PREFIX and a == b:
                return True
            if t == w:
                return True
    return False


def strata(gold, rows, crates):
    by_num = {r["number"]: r for r in rows}
    vp = vendor_prefix(crates)
    ctok = {c: [t for t in c.split("_") if t != vp] or [c] for c in crates}
    stok = {n: set(tokens(r["text"])) for n, r in by_num.items()}
    out = {}
    for (n, c) in gold:
        if re.search(rf"\b{re.escape(c)}\b", by_num[n]["text"]):
            out[(n, c)] = "verbatim"
        elif echo(ctok[c], stok[n]):
            out[(n, c)] = "name-echo"
        else:
            out[(n, c)] = "no-surface"
    return out


def prf(pred, gold):
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
    tiers = {"gold": {"gold"}, "gold_plus": {"gold", "gold_plus_only"}}[args.gold]
    gold = {(int(l["sentence"]), l["crate"]) for l in labels if l["tier"] in tiers}
    rows = load_sentences()
    crates = load_components()
    strat = strata(gold, rows, crates)
    counts = collections.Counter(strat.values())
    print(f"gold {args.gold}: {len(gold)} pairs")
    for k in ("verbatim", "name-echo", "no-surface"):
        print(f"  {k:11s} {counts[k]:5d}  {counts[k]/len(gold):.3f}")

    if args.links:
        by_src = collections.defaultdict(set)
        for r in csv.DictReader(open(args.links)):
            by_src[r["source"]].add((int(r["sentence"]), r["component_id"]))
        allp = set().union(*by_src.values())
        views = [("all stages", allp)] + [(f"stage {s}", by_src[s]) for s in sorted(by_src)]
        views.append(("minus partial_name", allp - by_src.get("partial_name", set())))
        print(f"\n{'view':22s} {'links':>6s} {'TP':>5s} {'P':>6s} {'R':>6s} "
              f"{'R_verb':>7s} {'R_echo':>7s} {'R_nosurf':>9s}")
        for name, pred in views:
            tp, p, r, f = prf(pred, gold)
            cells = []
            for k in ("verbatim", "name-echo", "no-surface"):
                g = {x for x in gold if strat[x] == k}
                cells.append(len(pred & g) / len(g) if g else 0.0)
            print(f"{name:22s} {len(pred):>6d} {tp:>5d} {p:>6.3f} {r:>6.3f} "
                  f"{cells[0]:>7.3f} {cells[1]:>7.3f} {cells[2]:>9.3f}")
        # what the links themselves look like, by stratum of the predicted pair
        print("\nstratum of predicted links (whether or not correct):")
        pstrat = strata(allp, rows, crates)
        for name, pred in views:
            c = collections.Counter(pstrat[x] for x in pred)
            print(f"  {name:22s} verbatim {c['verbatim']:5d}  name-echo {c['name-echo']:5d}  "
                  f"no-surface {c['no-surface']:5d}")
    json.dump({k: counts[k] for k in ("verbatim", "name-echo", "no-surface")},
              open(OUT / f"surface_strata_{args.gold}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
