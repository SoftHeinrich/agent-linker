"""Empirical: can the batched `blocks` proposer REPLACE s21's Phase-2 Framing-C
extraction? Compares gold-candidate recall (and candidate volume) of:

  * s21 Framing-C  — the existing extractor (2-pass UNION, flat batches of 50,
    alias-informed by Phase-1 knowledge). We run the REAL method and capture its
    candidate set, stopping right after Phase 2 (no validation/coref).
  * blocks:20      — the new batched proposer (from pilot/batch_strategy_compare.py
    cache), grounded to catalog names.

Both candidate sets are pre-validation, so the metric is the recall CEILING each
hands to the downstream gate. Run (default tier avoids flex 500s):

    OPENAI_SERVICE_TIER=default python pilot/extraction_replace_compare.py --dataset teammates
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker21 import SLinker21

BASE = Path("../ardoco/core/tests-base/src/main/resources/benchmark")
DS = {
    "mediastore": ("mediastore/model_2016/pcm/ms.repository", "mediastore/text_2016/mediastore.txt", "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"),
    "teastore": ("teastore/model_2020/pcm/teastore.repository", "teastore/text_2020/teastore.txt", "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"),
    "teammates": ("teammates/model_2021/pcm/teammates.repository", "teammates/text_2021/teammates.txt", "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "bigbluebutton": ("bigbluebutton/model_2021/pcm/bbb.repository", "bigbluebutton/text_2021/bigbluebutton.txt", "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "jabref": ("jabref/model_2021/pcm/jabref.repository", "jabref/text_2021/jabref.txt", "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
}
STRAT_CACHE = Path("pilot/cache/batch_strategy_cache.json")


class _Stop(Exception):
    def __init__(self, candidates):
        self.candidates = candidates


class _CaptureS21(SLinker21):
    """Run s21 through Phase 2 only, then hand back the Framing-C candidate set."""
    def _run_framing_c(self, sentences, components, name_to_id, sent_map):
        cands = super()._run_framing_c(sentences, components, name_to_id, sent_map)
        raise _Stop(cands)


def s21_framing_c_keys(repo, text):
    linker = _CaptureS21(backend=LLMBackend.OPENAI, model="gpt-5.4")
    try:
        linker.link(str(text), str(repo))
    except _Stop as s:
        # candidates: dict keyed (sentence_number, component_id)
        return set(s.candidates.keys())
    raise RuntimeError("Phase 2 did not fire")


def blocks_keys(dataset, name_to_id):
    if not STRAT_CACHE.exists():
        return None
    cache = json.loads(STRAT_CACHE.read_text())
    keys = set()
    found = False
    for k, refs in cache.items():
        ds, strat, B, _rng = k.split("|")
        if ds == dataset and strat == "blocks" and B == "20":
            found = True
            for r in refs:
                cid = name_to_id.get(r["component"])
                if cid:
                    keys.add((int(r["sentence"]), cid))
    return keys if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="teammates")
    args = ap.parse_args()
    repo, text, gcsv = (BASE / p for p in DS[args.dataset])

    comps = parse_pcm_repository(str(repo))
    name_to_id = {c.name: c.id for c in comps}
    gold = set()
    for row in csv.DictReader(open(gcsv)):
        gold.add((int(row["sentence"]), row["modelElementID"]))
    gold = {(s, c) for (s, c) in gold if c in name_to_id.values()}

    s21 = s21_framing_c_keys(repo, text)
    blk = blocks_keys(args.dataset, name_to_id)
    if blk is None:
        print("No cached blocks:20 for this dataset — run batch_strategy_compare first.")
        return

    def rec(keys):
        return len(keys & gold) / len(gold)

    print(f"\n=== extraction-replacement check: {args.dataset} (gold={len(gold)}) ===")
    print(f"{'extractor':<16}{'cands':>7}{'gold_hit':>9}{'recall':>8}")
    print(f"{'s21 Framing-C':<16}{len(s21):>7}{len(s21 & gold):>9}{rec(s21):>8.3f}")
    print(f"{'blocks:20':<16}{len(blk):>7}{len(blk & gold):>9}{rec(blk):>8.3f}")

    gs, gb = s21 & gold, blk & gold
    print(f"\ngold caught by BOTH:            {len(gs & gb)}")
    print(f"gold caught by blocks NOT s21:  {len(gb - gs)}   <- blocks adds these")
    print(f"gold caught by s21 NOT blocks:  {len(gs - gb)}   <- blocks would LOSE these if it replaced s21")
    print(f"union (blocks + s21):           {len(gs | gb)}   recall {len(gs | gb)/len(gold):.3f}")


if __name__ == "__main__":
    main()
