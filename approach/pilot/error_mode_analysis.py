"""Systematic error-mode analysis of the replace/union e2e predictions vs s21.
Reads the predicted-link CSVs already written by run_ablation (results/ablation_results/
<variant>_<dataset>_links.csv) and the gold standard — NO new API calls. For each
variant it dumps every FALSE POSITIVE (with its source stage + sentence text) and every
FALSE NEGATIVE (with sentence text, and whether the OTHER variant caught it), so error
MODES can be read and categorized before designing any fix.

    python pilot/error_mode_analysis.py --dataset teammates --variants s_linker21 s_linker23_union
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from llm_sad_sam.core.document_loader_v2 import load_sentences
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository

BASE = Path("../ardoco/core/tests-base/src/main/resources/benchmark")
RES = Path("results/ablation_results")
DS = {
    "mediastore": ("mediastore/model_2016/pcm/ms.repository", "mediastore/text_2016/mediastore.txt", "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"),
    "teastore": ("teastore/model_2020/pcm/teastore.repository", "teastore/text_2020/teastore.txt", "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"),
    "teammates": ("teammates/model_2021/pcm/teammates.repository", "teammates/text_2021/teammates.txt", "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "bigbluebutton": ("bigbluebutton/model_2021/pcm/bbb.repository", "bigbluebutton/text_2021/bigbluebutton.txt", "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "jabref": ("jabref/model_2021/pcm/jabref.repository", "jabref/text_2021/jabref.txt", "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
}


def load_pred(variant, dataset):
    f = RES / f"{variant}_{dataset}_links.csv"
    if not f.exists():
        return None
    out = {}
    for r in csv.DictReader(open(f)):
        out[(int(r["sentence"]), r["component_id"])] = (r["component_name"], r.get("source", ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="teammates")
    ap.add_argument("--variants", nargs="+",
                    default=["s_linker21", "s_linker23_replace", "s_linker23_union"])
    args = ap.parse_args()
    repo, text, gcsv = (BASE / p for p in DS[args.dataset])

    comps = parse_pcm_repository(str(repo))
    id_to_name = {c.id: c.name for c in comps}
    valid_ids = set(id_to_name)
    sent = {s.number: s.text for s in load_sentences(str(text))}
    gold = {(int(r["sentence"]), r["modelElementID"]) for r in csv.DictReader(open(gcsv))}
    gold = {k for k in gold if k[1] in valid_ids}

    preds = {v: load_pred(v, args.dataset) for v in args.variants}
    preds = {v: p for v, p in preds.items() if p is not None}

    print(f"\n############ {args.dataset} (gold={len(gold)}) ############")
    for v, pred in preds.items():
        keys = set(pred)
        tp, fp, fn = keys & gold, keys - gold, gold - keys
        print(f"\n===== {v}: TP={len(tp)} FP={len(fp)} FN={len(fn)} =====")

        print(f"  -- FALSE POSITIVES ({len(fp)}) [pred not in gold] --")
        for (sn, cid) in sorted(fp):
            name, src = pred[(sn, cid)]
            other = [ov for ov, op in preds.items() if ov != v and (sn, cid) in (op or {})]
            print(f"   FP S{sn} [{src:11}] -> {name}")
            print(f"       \"{sent.get(sn,'?')}\"")
            if other:
                print(f"       (also predicted by: {', '.join(other)})")

        print(f"  -- FALSE NEGATIVES ({len(fn)}) [gold not predicted] --")
        for (sn, cid) in sorted(fn):
            name = id_to_name.get(cid, cid)
            other = [ov for ov, op in preds.items() if ov != v and (sn, cid) in (op or {})]
            tag = f"  (CAUGHT by: {', '.join(other)})" if other else "  (missed by ALL)"
            print(f"   FN S{sn} -> {name}{tag}")
            print(f"       \"{sent.get(sn,'?')}\"")


if __name__ == "__main__":
    main()
