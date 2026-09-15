#!/usr/bin/env python3
"""Replace the antecedent-gated coreference stage with topic propagation, and judge it.

s110's coreference stage only fires when an earlier sentence *names* the component, and
§9.2 shows that gate costs it almost everything: 22 links, 7 correct.  This probe keeps the
same idea but widens the antecedent to "a nearby sentence the linker already linked", then
puts every proposal in front of the same kind of judge.

  propose   for every link (n, C) the linker made with the full-name stage, propose
            (n+1..n+K, C) inside the same chapter, minus pairs the linker already has
  judge     one LLM call per batch of proposals: is the later sentence still making a claim
            about C, given the sentence that established it?

Scored against the semantic gold, alone and unioned with the run it extends.
usage: topic_probe.py <links.csv> [--k 3] [--gold gold_plus] [--dry]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json

from common import OUT, load_sentences
from llm import call_many, extract_json

JUDGE = """You decide whether a sentence continues to make a claim about a component.

You are given, for each case: the component's name, the sentence that established it as the
topic (the ANCHOR), the sentences in between, and the sentence in question (the TARGET).

Answer yes when the TARGET states something about that component: what it does, how it
behaves, what it holds, what it produces. The component does not have to be named in the
TARGET — a pronoun, a definite noun phrase, or plain topical continuation all count.

Answer no when the topic has moved on, when the TARGET is about a different component, a
data structure, a process or the system as a whole, or when it is an instruction to the
reader, an example, or a general remark.

Return JSON only: {"<case number>": {"answer": "yes"|"no", "why": "one clause"}}
"""


def prf(pred, gold):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return tp, p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("links")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gold", default="gold_plus")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--salt", default="", help="cache salt: a fresh sample of the same judge")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    rows = load_sentences()
    by = {r["number"]: r for r in rows}
    order = [r["number"] for r in rows]
    chapter = {r["number"]: r["chapter"] for r in rows}
    pos, seen = {}, collections.Counter()
    for n in order:
        seen[chapter[n]] += 1
        pos[n] = seen[chapter[n]]
    nxt = collections.defaultdict(list)
    for n in order:
        nxt[chapter[n]].append(n)

    by_src = collections.defaultdict(set)
    for r in csv.DictReader(open(args.links)):
        by_src[r["source"]].add((int(r["sentence"]), r["component_id"]))
    have = set().union(*by_src.values())
    anchors = by_src["full_name"]

    proposals = {}
    for (n, c) in sorted(anchors):
        chap = nxt[chapter[n]]
        i = chap.index(n)
        for m in chap[i + 1: i + 1 + args.k]:
            if (m, c) in have or (m, c) in proposals:
                continue
            proposals[(m, c)] = n
    print(f"{len(anchors)} anchors -> {len(proposals)} proposals (k={args.k})")

    items = sorted(proposals.items())
    prompts, keys = [], []
    for start in range(0, len(items), args.batch):
        chunk = items[start:start + args.batch]
        blocks = []
        for j, ((m, c), a) in enumerate(chunk, 1):
            between = [f"S{x}: {by[x]['text']}" for x in range(a + 1, m)][:3]
            blocks.append(
                f"--- Case {j} ---\nCOMPONENT: {c}\n"
                f"ANCHOR S{a}: {by[a]['text']}\n"
                + ("BETWEEN:\n  " + "\n  ".join(between) + "\n" if between else "")
                + f"TARGET S{m}: {by[m]['text']}")
        prompts.append(JUDGE + "\n" + "\n".join(blocks) + "\n\nJSON only:")
        keys.append(chunk)

    if args.dry:
        print(prompts[0][:2000])
        return
    res = call_many(args.backend, args.model, prompts, workers=args.workers, progress="topic", salt=args.salt)
    approved, unparsed = set(), 0
    for chunk, r in zip(keys, res):
        data = extract_json(r.get("text", ""))
        if not data:
            unparsed += 1
            continue
        for k, v in data.items():
            try:
                j = int("".join(ch for ch in str(k) if ch.isdigit()))
            except ValueError:
                continue
            if 1 <= j <= len(chunk) and str(v.get("answer", "")).lower().startswith("y"):
                approved.add(chunk[j - 1][0])
    print(f"approved {len(approved)} / {len(proposals)} proposals (unparsed batches {unparsed})")

    labels = list(csv.DictReader(open(OUT / "semantic_labels.csv")))
    tiers = {"gold": {"gold"}, "gold_plus": {"gold", "gold_plus_only"}}[args.gold]
    gold = {(int(l["sentence"]), l["crate"]) for l in labels if l["tier"] in tiers}
    print(f"\ngold {args.gold}: {len(gold)}")
    print(f"{'view':40s} {'links':>6s} {'TP':>5s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    for name, pred in [
            ("s110 coreference stage", by_src["coreference"]),
            ("topic propagation, judged", approved),
            ("full_name only", anchors),
            ("full_name + topic propagation", anchors | approved),
            ("all s110 stages", have),
            ("all s110 stages + topic propagation", have | approved)]:
        tp, p, r, f = prf(pred, gold)
        print(f"{name:40s} {len(pred):>6d} {tp:>5d} {p:>6.3f} {r:>6.3f} {f:>6.3f}")

    with open(OUT / f"topic_probe_k{args.k}{args.salt}.csv", "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["sentence", "component_id", "anchor_sentence", "approved", "in_gold"])
        for (m, c), a in items:
            w.writerow([m, c, a, int((m, c) in approved), int((m, c) in gold)])


if __name__ == "__main__":
    main()
