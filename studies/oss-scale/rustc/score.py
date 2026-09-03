#!/usr/bin/env python3
"""Score a linker run on the rustc dataset against the anchor gold.

Usage: score.py <links.csv> <data dir> [--show N]

The gold is partial by construction: only sentences carrying an anchor have gold,
and an anchored sentence may describe more crates than it anchors.  So besides the
naive strict P/R/F1 the script reports the numbers that the gold licenses:
  recall            over all gold pairs, and split by anchor kind (link-only =
                    the crate is invisible in the sentence text; verbatim = named)
  anchored P        precision of predicted links on anchored sentences only
  unanchored preds  count + a sample to spot-check by hand (the gold says nothing
                    about them)
  per-crate         TP/FP/FN for the crates carrying >= 5 gold pairs
Stdlib only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from pathlib import Path


def read_links(path: Path) -> set[tuple[int, str]]:
    out = set()
    with open(path) as handle:
        for row in csv.DictReader(handle):
            cid = (row.get("modelElementID") or row.get("component_id") or "").strip()
            snum = (row.get("sentence") or row.get("sentence_number") or "").strip()
            if cid and snum:
                out.add((int(snum), cid))
    return out


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("links", type=Path)
    ap.add_argument("data", type=Path)
    ap.add_argument("--show", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    pred = read_links(args.links)
    gold = read_links(args.data / "gold.csv")
    meta = {m["number"]: m for m in json.loads((args.data / "meta.json").read_text())}
    sentences = (args.data / "sentences.txt").read_text().splitlines()
    anchored = {n for n, m in meta.items() if m["link"] or m["verbatim"]}

    tp = pred & gold
    p, r, f = prf(len(tp), len(pred - gold), len(gold - pred))
    print(f"strict     TP {len(tp)} FP {len(pred - gold)} FN {len(gold - pred)}  P {p:.3f} R {r:.3f} F1 {f:.3f}")

    link_only = {(n, c) for n, c in gold if c in meta[n]["link"] and c not in meta[n]["verbatim"]}
    verbatim = gold - link_only
    for name, subset in (("link-only", link_only), ("verbatim", verbatim)):
        hit = len(subset & pred)
        print(f"recall {name:10s} {hit}/{len(subset)} = {hit / len(subset):.3f}" if subset else f"recall {name}: no pairs")

    pred_anch = {(n, c) for n, c in pred if n in anchored}
    pa = len(pred_anch & gold) / len(pred_anch) if pred_anch else 0.0
    print(f"anchored P {len(pred_anch & gold)}/{len(pred_anch)} = {pa:.3f}   (links on anchored sentences)")
    pred_un = sorted((n, c) for n, c in pred if n not in anchored)
    print(f"unanchored predictions: {len(pred_un)} on {len({n for n, _ in pred_un})} sentences (no gold; spot-check below)")
    print(f"sentences linked: {len({n for n, _ in pred})}/{len(sentences)}; crates linked: {len({c for _, c in pred})}")

    per = collections.defaultdict(lambda: [0, 0, 0])
    for n, c in pred:
        per[c][0 if (n, c) in gold else 1] += 1
    for n, c in gold - pred:
        per[c][2] += 1
    gold_count = collections.Counter(c for _, c in gold)
    print("\nper-crate (gold >= 5): crate TP FP FN")
    for c, cnt in gold_count.most_common():
        if cnt < 5:
            break
        t, fpc, fnc = per[c]
        print(f"  {c:24s} {t:3d} {fpc:3d} {fnc:3d}")
    noisy = sorted(((v[1], c) for c, v in per.items() if gold_count[c] < 5 and v[1]), reverse=True)[:8]
    if noisy:
        print("  most FP among crates with < 5 gold:", ", ".join(f"{c}({fpc})" for fpc, c in noisy))

    rng = random.Random(args.seed)
    print(f"\nFN sample (gold missed), {args.show}:")
    for n, c in rng.sample(sorted(gold - pred), min(args.show, len(gold - pred))):
        kind = "link" if c in meta[n]["link"] else "verb"
        print(f"  [{n}] {c} ({kind}) :: {sentences[n - 1][:160]}")
    print(f"\nFP-on-anchored sample, {args.show}:")
    fp_anch = sorted(pred_anch - gold)
    for n, c in rng.sample(fp_anch, min(args.show, len(fp_anch))):
        print(f"  [{n}] {c} (gold {meta[n]['link'] + meta[n]['verbatim']}) :: {sentences[n - 1][:160]}")
    print(f"\nunanchored-prediction sample, {args.show}:")
    for n, c in rng.sample(pred_un, min(args.show, len(pred_un))):
        print(f"  [{n}] {c} :: {sentences[n - 1][:160]}")


if __name__ == "__main__":
    main()
