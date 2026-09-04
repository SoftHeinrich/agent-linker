"""Validate the semantic labels against the independent signals and produce the
human-check sheet.
  1. Anchor gold (hyperlink item / concept, verbatim): how does each annotator label the
     anchored pair? Item links should come out REFERS-heavy, verbatim ABOUT-heavy.
  2. Co-change pairs: coverage by gold_plus.
  3. Stratified 120-pair sheet for a human: sentence, crate, tier, blank verdict column.
"""
from __future__ import annotations

import collections
import csv
import json
import random

from common import DATA, OUT, load_anchor_gold, load_sentences, write_csv


def main() -> None:
    rows = {r["number"]: r for r in load_sentences()}
    labels = list(csv.DictReader(open(OUT / "semantic_labels.csv")))
    annotators = [k for k in labels[0].keys() if k not in ("sentence", "crate", "tier", "consistency", "crateview", "sym", "anchor", "cochange")]
    by_pair = {(int(l["sentence"]), l["crate"]): l for l in labels}
    gold_plus = {p for p, l in by_pair.items() if l["tier"] in ("gold", "gold_plus_only")}
    refers = {p for p, l in by_pair.items() if l["tier"] == "refers"}

    # 1. anchors by kind
    kinds = collections.defaultdict(list)
    for n, r in rows.items():
        for c in r["link"]:
            kinds["link_" + r["link_kind"].get(c, "?")].append((n, c))
        for c in r["verbatim"]:
            if c not in r["link"]:
                kinds["verbatim"].append((n, c))
    out = {}
    for kind, pairs in sorted(kinds.items()):
        stat = collections.Counter()
        for p in pairs:
            l = by_pair.get(p)
            for a in annotators:
                stat[a + ":" + (l[a] if l and l[a] else "-")] += 1
            stat["tier:" + (l["tier"] if l else "none")] += 1
        out[kind] = {"n": len(pairs), **{k: v for k, v in sorted(stat.items())}}
    # 2. co-change coverage
    cc = OUT / "cochange_pairs.csv"
    if cc.exists():
        ccp = {(int(r["sentence"]), r["crate"]) for r in csv.DictReader(open(cc))}
        out["cochange"] = {"pairs": len(ccp), "in_gold_plus": len(ccp & gold_plus), "in_refers": len(ccp & refers),
                           "unlabelled": len(ccp - gold_plus - refers)}
    (OUT / "validation.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

    # 3. human sheet, stratified by tier x evidence pattern (design 1 of LITERATURE.md), no-source stratum over-sampled
    random.seed(11)
    sheet = []
    def pattern(l):
        det = int(l["sym"]) + int(l["anchor"]) + int(l["cochange"])
        return "llm-only" if det == 0 else ("multi-source" if det >= 2 else ("anchor" if int(l["anchor"]) else ("symbol" if int(l["sym"]) else "cochange")))
    quota = {("gold", "llm-only"): 60, ("gold", "anchor"): 20, ("gold", "symbol"): 20, ("gold", "multi-source"): 10,
             ("gold_plus_only", "anchor"): 15, ("gold_plus_only", "symbol"): 15, ("gold_plus_only", "multi-source"): 5,
             ("silver", "llm-only"): 40, ("refers", "llm-only"): 15, ("refers", "symbol"): 15}
    for (tier, pat), k in quota.items():
        pool = [l for l in labels if l["tier"] == tier and pattern(l) == pat]
        for l in random.sample(pool, min(k, len(pool))):
            sheet.append({"sentence": l["sentence"], "crate": l["crate"], "tier": tier, "pattern": pat, "text": rows[int(l["sentence"])]["text"],
                          "chapter": rows[int(l["sentence"])]["chapter"], "verdict(A/R/N)": ""})
    ev = {e["number"]: e for e in json.loads((OUT / "evidence.json").read_text())}
    labelled_s = {int(l["sentence"]) for l in labels}
    neg_pool = [n for n in rows if n not in labelled_s and ev[n]["bm25_top"]]
    for n in random.sample(neg_pool, min(40, len(neg_pool))):
        sheet.append({"sentence": n, "crate": ev[n]["bm25_top"][0], "tier": "none", "pattern": "bm25-top1", "text": rows[n]["text"],
                      "chapter": rows[n]["chapter"], "verdict(A/R/N)": ""})
    random.shuffle(sheet)
    write_csv(OUT / "human_check_sheet.csv", sheet, ["sentence", "crate", "tier", "pattern", "chapter", "text", "verdict(A/R/N)"])
    print("human sheet rows:", len(sheet))


if __name__ == "__main__":
    main()
