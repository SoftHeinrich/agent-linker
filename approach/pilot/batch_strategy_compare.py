"""Empirical comparison of batching STRATEGIES for the grounded proposer.

Question: can we read a whole document in one / a few LLM calls WITHOUT losing the
per-sentence extraction recall? (Per-sentence = one call per sentence, which is not
acceptable.) A flat numbered list ("plain") degrades — the model skims a long list —
so we test structural fixes, all defined in `proposer.py` (this script imports them,
so what is measured is exactly what ships):

  plain     — flat numbered sentences in one call (the degrading baseline).
  forced    — plain + instruction to process each sentence independently.
  coverage  — REQUIRE one output row per sentence (forces the model to walk each).
  blocks    — each sentence rendered as its own item with its prev-sentence context.

Metric: gold recall = |grounded (sentence,component) ∩ gold| / |gold|, plus number
of LLM calls. Winner = fewest calls whose recall matches a small-batch reference.
Calls cached per (dataset, strategy, batch, sentence-range) so a flaky API resumes
for free. Run:

    python pilot/batch_strategy_compare.py --dataset bigbluebutton \
        --configs plain:20 plain:999 coverage:999 blocks:999 forced:999
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.proposer import (
    build_batch_prompt, _parse_batch, make_client,
)

BASE = Path("../ardoco/core/tests-base/src/main/resources/benchmark")
DS = {
    "mediastore": ("mediastore/model_2016/pcm/ms.repository", "mediastore/text_2016/mediastore.txt", "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"),
    "teastore": ("teastore/model_2020/pcm/teastore.repository", "teastore/text_2020/teastore.txt", "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"),
    "teammates": ("teammates/model_2021/pcm/teammates.repository", "teammates/text_2021/teammates.txt", "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "bigbluebutton": ("bigbluebutton/model_2021/pcm/bbb.repository", "bigbluebutton/text_2021/bigbluebutton.txt", "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "jabref": ("jabref/model_2021/pcm/jabref.repository", "jabref/text_2021/jabref.txt", "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
}
CACHE = Path("pilot/cache/batch_strategy_cache.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bigbluebutton")
    ap.add_argument("--configs", nargs="+",
                    default=["plain:20", "plain:999", "coverage:999", "blocks:999", "forced:999"])
    args = ap.parse_args()

    repo, text, gcsv = (BASE / p for p in DS[args.dataset])
    comps = parse_pcm_repository(str(repo))
    names = [c.name for c in comps]
    id2name = {c.id: c.name for c in comps}
    sents = load_sentences(str(text))
    sent_map = build_sent_map(sents)
    prev_of = {s.number: (sent_map.get(s.number - 1).text if sent_map.get(s.number - 1) else "")
               for s in sents}
    lut = {n.lower(): n for n in names}
    gold = set()
    for row in csv.DictReader(open(gcsv)):
        nm = id2name.get(row["modelElementID"])
        if nm:
            gold.add((int(row["sentence"]), nm))

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    client = None
    print(f"dataset={args.dataset} sentences={len(sents)} components={len(names)} gold={len(gold)}\n")
    print(f"{'strategy':<10}{'batch':>6}{'calls':>7}{'cand':>6}{'gold_hit':>9}{'recall':>8}")
    ref, rows = None, []
    for cfg in args.configs:
        strategy, B = cfg.split(":"); B = int(B)
        surfaced, calls = set(), 0
        for i in range(0, len(sents), B):
            chunk = sents[i:i + B]
            calls += 1
            ck = f"{args.dataset}|{strategy}|{B}|{chunk[0].number}-{chunk[-1].number}"
            if ck in cache:
                raw = cache[ck]
            else:
                if client is None:
                    client = make_client()
                prompt = build_batch_prompt(chunk, names, None, strategy=strategy, prev_of=prev_of)
                resp = client.query(prompt, timeout=300)
                raw = _parse_batch(resp.text if resp.success else "", strategy)
                cache[ck] = raw
                CACHE.write_text(json.dumps(cache))
            for r in raw:
                c = lut.get(r["component"].lower())
                if c:
                    surfaced.add((int(r["sentence"]), c))
        gh = surfaced & gold
        rows.append((strategy, B, calls, gh))
        if ref is None:
            ref = gh
        print(f"{strategy:<10}{B:>6}{calls:>7}{len(surfaced):>6}{len(gh):>9}{len(gh)/len(gold):>8.3f}")

    print(f"\nreference (first config) gold set size = {len(ref)}")
    for strategy, B, calls, gh in rows[1:]:
        print(f"  {strategy}:{B} ({calls} calls) lost {len(ref - gh)} / gained {len(gh - ref)} gold vs reference")
    print(f"\ncache: {CACHE}")


if __name__ == "__main__":
    main()
