"""Score linker output against the semantic gold tiers.
usage: rescore.py <links.csv> [--gold gold_plus|gold] [--source full_name,coreference]
Reports strict P/R/F1, lenient P (REFERS predictions not counted as FP), recall split by
whether the sentence names the crate verbatim (explicit) or not (implicit tail), per-stage.
"""
from __future__ import annotations

import argparse
import collections
import csv
import re

from common import OUT, load_components, load_sentences


def load_links(path: str, sources: set[str] | None):
    links = collections.defaultdict(set)
    with open(path) as h:
        for r in csv.DictReader(h):
            src = r.get("source", "")
            if sources and src not in sources:
                continue
            links[src].add((int(r["sentence"]), r["component_id"]))
    return links


def prf(pred: set, gold: set):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return tp, p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("links")
    ap.add_argument("--gold", default="gold_plus")
    ap.add_argument("--sources", default="")
    ap.add_argument("--show-fp", type=int, default=0, help="sample N false positives of the full_name stage with annotator rationale")
    ap.add_argument("--show-implicit", type=int, default=0, help="sample N implicit gold pairs")
    args = ap.parse_args()
    labels = list(csv.DictReader(open(OUT / "semantic_labels.csv")))
    if args.gold in ("gold", "gold_plus"):
        tiers = {"gold": {"gold"}, "gold_plus": {"gold", "gold_plus_only"}}[args.gold]
        gold = {(int(l["sentence"]), l["crate"]) for l in labels if l["tier"] in tiers}
    else:  # 3way | a2only | anchor
        path = {"3way": OUT / "gold_semantic_3way.csv", "a2only": OUT / "gold_semantic_a2only.csv",
                "anchor": OUT.parent.parent / "data" / "core" / "gold.csv"}[args.gold]
        gold = {(int(x["sentence"]), x["modelElementID"]) for x in csv.DictReader(open(path))}
    refers = {(int(l["sentence"]), l["crate"]) for l in labels if l["tier"] == "refers"}
    silver = {(int(l["sentence"]), l["crate"]) for l in labels if l["tier"] == "silver"}
    rows = {r["number"]: r for r in load_sentences()}
    explicit = {(n, c) for (n, c) in gold if re.search(rf"\b{re.escape(c)}\b", rows[n]["text"])}
    implicit = gold - explicit
    sources = set(args.sources.split(",")) if args.sources else None
    by_src = load_links(args.links, sources)
    allp = set().union(*by_src.values()) if by_src else set()
    print(f"gold tier {args.gold}: {len(gold)} pairs on {len({n for n,_ in gold})} sentences; explicit {len(explicit)} implicit {len(implicit)}; refers {len(refers)} silver {len(silver)}")
    print(f"{'view':34s} {'links':>6s} {'TP':>5s} {'P':>6s} {'R':>6s} {'F1':>6s} {'P_len':>6s} {'R_expl':>6s} {'R_impl':>6s} {'FP->silver':>10s}")
    views = [("all stages", allp)] + [(f"stage {s}", p) for s, p in sorted(by_src.items())]
    if "partial_name" in by_src:
        views.append(("minus partial-name stage", allp - by_src["partial_name"]))
    for name, pred in views:
        tp, p, r, f = prf(pred, gold)
        fp = pred - gold
        lenient = tp / max(1, len(pred - refers)) if pred else 0.0
        re_ = len(pred & explicit) / max(1, len(explicit))
        ri = len(pred & implicit) / max(1, len(implicit))
        print(f"{name:34s} {len(pred):6d} {tp:5d} {p:6.3f} {r:6.3f} {f:6.3f} {lenient:6.3f} {re_:6.3f} {ri:6.3f} {len(fp & silver):10d}")
    import json, random
    random.seed(3)
    why = {}
    for pth in OUT.glob("annotations_*.json"):
        if "crateview" in pth.name or "_r" in pth.stem[-3:]:
            continue
        for n, v in json.loads(pth.read_text())["labels"].items():
            why.setdefault(int(n), []).append(f"{pth.stem.split('_')[1][:6]}: about={v['about']} refers={v['refers']} | {v.get('why','')[:90]}")
    if args.show_fp and "full_name" in by_src:
        fps = sorted(by_src["full_name"] - gold - refers - silver)
        print(f"\n== {args.show_fp} of {len(fps)} full_name links that no annotator labelled (hard FPs)")
        for n, c in random.sample(fps, min(args.show_fp, len(fps))):
            print(f"[{n}] -> {c} :: {rows[n]['text'][:160]}")
            for w in why.get(n, []):
                print("      ", w)
    if args.show_implicit:
        imp = sorted(implicit - allp)
        print(f"\n== {args.show_implicit} of {len(imp)} implicit gold pairs the linker missed")
        for n, c in random.sample(imp, min(args.show_implicit, len(imp))):
            print(f"[{n}] {c} :: {rows[n]['text'][:170]}")


if __name__ == "__main__":
    main()
