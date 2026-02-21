#!/usr/bin/env python3
"""Detailed TransArc baseline vs gold standard analysis for all 5 datasets."""

import csv
import sys
from collections import defaultdict

sys.path.insert(0, "/mnt/hostshare/ardoco-home/llm-sad-sam-v45/src")
from llm_sad_sam.pcm_parser import parse_pcm_repository

BENCHMARK = "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
CLI_RESULTS = "/mnt/hostshare/ardoco-home/cli-results"

DATASETS = {
    "mediastore": {
        "gold": f"{BENCHMARK}/mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
        "transarc": f"{CLI_RESULTS}/mediastore-sad-sam/sadSamTlr_mediastore.csv",
        "pcm": f"{BENCHMARK}/mediastore/model_2016/pcm/ms.repository",
        "text": f"{BENCHMARK}/mediastore/text_2016/mediastore.txt",
    },
    "teastore": {
        "gold": f"{BENCHMARK}/teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
        "transarc": f"{CLI_RESULTS}/teastore-sad-sam/sadSamTlr_teastore.csv",
        "pcm": f"{BENCHMARK}/teastore/model_2020/pcm/teastore.repository",
        "text": f"{BENCHMARK}/teastore/text_2020/teastore.txt",
    },
    "teammates": {
        "gold": f"{BENCHMARK}/teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc": f"{CLI_RESULTS}/teammates-sad-sam/sadSamTlr_teammates.csv",
        "pcm": f"{BENCHMARK}/teammates/model_2021/pcm/teammates.repository",
        "text": f"{BENCHMARK}/teammates/text_2021/teammates.txt",
    },
    "bigbluebutton": {
        "gold": f"{BENCHMARK}/bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc": f"{CLI_RESULTS}/bigbluebutton-sad-sam/sadSamTlr_bigbluebutton.csv",
        "pcm": f"{BENCHMARK}/bigbluebutton/model_2021/pcm/bbb.repository",
        "text": f"{BENCHMARK}/bigbluebutton/text_2021/bigbluebutton.txt",
    },
    "jabref": {
        "gold": f"{BENCHMARK}/jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc": f"{CLI_RESULTS}/jabref-sad-sam/sadSamTlr_jabref.csv",
        "pcm": f"{BENCHMARK}/jabref/model_2021/pcm/jabref.repository",
        "text": f"{BENCHMARK}/jabref/text_2021/jabref.txt",
    },
}


def load_csv(path):
    links = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["modelElementID"].strip()
            sent = int(row["sentence"].strip())
            links.add((mid, sent))
    return links


def load_sentences(path):
    sentences = {}
    with open(path) as f:
        for i, line in enumerate(f, 1):
            sentences[i] = line.strip()
    return sentences


def load_id_to_name(pcm_path):
    components = parse_pcm_repository(pcm_path)
    id_to_name = {}
    for c in components:
        id_to_name[c.id] = c.name
    return id_to_name


def fmt_pct(val):
    return f"{val*100:.1f}%"


def analyze_dataset(name, paths):
    print(f"\n{'='*100}")
    print(f"  DATASET: {name.upper()}")
    print(f"{'='*100}")

    gold = load_csv(paths["gold"])
    transarc = load_csv(paths["transarc"])
    sentences = load_sentences(paths["text"])
    id_to_name = load_id_to_name(paths["pcm"])

    tp = gold & transarc
    fp = transarc - gold
    fn = gold - transarc

    precision = len(tp) / len(transarc) if transarc else 0
    recall = len(tp) / len(gold) if gold else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  Overall Statistics:")
    print(f"  {'Gold standard links:':<30} {len(gold):>4}")
    print(f"  {'TransArc links:':<30} {len(transarc):>4}")
    print(f"  {'True Positives (TP):':<30} {len(tp):>4}")
    print(f"  {'False Positives (FP):':<30} {len(fp):>4}")
    print(f"  {'False Negatives (FN):':<30} {len(fn):>4}")
    print(f"  {'Precision:':<30} {fmt_pct(precision):>8}")
    print(f"  {'Recall:':<30} {fmt_pct(recall):>8}")
    print(f"  {'F1:':<30} {fmt_pct(f1):>8}")

    # Collect all component IDs
    all_ids = set()
    for mid, _ in gold | transarc:
        all_ids.add(mid)

    # Per-component breakdown
    comp_stats = {}
    for mid in all_ids:
        cname = id_to_name.get(mid, f"UNKNOWN({mid[:20]}...)")
        gold_links = {(m, s) for m, s in gold if m == mid}
        trans_links = {(m, s) for m, s in transarc if m == mid}
        c_tp = gold_links & trans_links
        c_fp = trans_links - gold_links
        c_fn = gold_links - trans_links
        c_prec = len(c_tp) / len(trans_links) if trans_links else 0
        c_rec = len(c_tp) / len(gold_links) if gold_links else 0
        c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0
        comp_stats[mid] = {
            "name": cname,
            "gold": len(gold_links),
            "transarc": len(trans_links),
            "tp": len(c_tp),
            "fp": len(c_fp),
            "fn": len(c_fn),
            "precision": c_prec,
            "recall": c_rec,
            "f1": c_f1,
            "fn_links": c_fn,
            "fp_links": c_fp,
        }

    # Print per-component table
    print(f"\n  Per-Component Breakdown:")
    print(f"  {'Component':<30} {'Gold':>5} {'TArc':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"  {'-'*30} {'-'*5} {'-'*5} {'-'*4} {'-'*4} {'-'*4} {'-'*7} {'-'*7} {'-'*7}")

    for mid in sorted(comp_stats, key=lambda x: comp_stats[x]["name"]):
        s = comp_stats[mid]
        print(f"  {s['name']:<30} {s['gold']:>5} {s['transarc']:>5} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} {fmt_pct(s['precision']):>7} {fmt_pct(s['recall']):>7} {fmt_pct(s['f1']):>7}")

    # Components with 0% recall (completely missed)
    missed = [mid for mid, s in comp_stats.items() if s["gold"] > 0 and s["recall"] == 0]
    if missed:
        print(f"\n  Components COMPLETELY MISSED by TransArc (0% recall):")
        for mid in missed:
            s = comp_stats[mid]
            print(f"    - {s['name']} ({s['gold']} gold links)")

    # Components with 100% recall
    perfect = [mid for mid, s in comp_stats.items() if s["gold"] > 0 and s["recall"] == 1.0]
    if perfect:
        print(f"\n  Components with PERFECT recall (100%):")
        for mid in perfect:
            s = comp_stats[mid]
            extra = f" (+{s['fp']} FP)" if s["fp"] > 0 else ""
            print(f"    - {s['name']} ({s['tp']}/{s['gold']} links){extra}")

    # Components with only FP (no gold links)
    fp_only = [mid for mid, s in comp_stats.items() if s["gold"] == 0 and s["fp"] > 0]
    if fp_only:
        print(f"\n  Components with ONLY false positives (no gold links):")
        for mid in fp_only:
            s = comp_stats[mid]
            print(f"    - {s['name']} ({s['fp']} FP links)")

    # FN analysis with sentence text
    if fn:
        print(f"\n  False Negative Analysis (links TransArc misses):")
        print(f"  {'Component':<30} {'Sent#':>5}  Sentence Text")
        print(f"  {'-'*30} {'-'*5}  {'-'*60}")
        fn_sorted = sorted(fn, key=lambda x: (id_to_name.get(x[0], x[0]), x[1]))
        for mid, sent in fn_sorted:
            cname = id_to_name.get(mid, mid[:20])
            text = sentences.get(sent, "<sentence not found>")
            if len(text) > 80:
                text = text[:77] + "..."
            print(f"  {cname:<30} S{sent:<4}  {text}")

    # FP analysis with sentence text
    if fp:
        print(f"\n  False Positive Analysis (incorrect TransArc links):")
        print(f"  {'Component':<30} {'Sent#':>5}  Sentence Text")
        print(f"  {'-'*30} {'-'*5}  {'-'*60}")
        fp_sorted = sorted(fp, key=lambda x: (id_to_name.get(x[0], x[0]), x[1]))
        for mid, sent in fp_sorted:
            cname = id_to_name.get(mid, mid[:20])
            text = sentences.get(sent, "<sentence not found>")
            if len(text) > 80:
                text = text[:77] + "..."
            print(f"  {cname:<30} S{sent:<4}  {text}")

    return {
        "gold": len(gold),
        "transarc": len(transarc),
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    all_stats = {}
    for name, paths in DATASETS.items():
        all_stats[name] = analyze_dataset(name, paths)

    # Summary table
    print(f"\n{'='*100}")
    print(f"  SUMMARY ACROSS ALL DATASETS")
    print(f"{'='*100}")
    print(f"\n  {'Dataset':<20} {'Gold':>5} {'TArc':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print(f"  {'-'*20} {'-'*5} {'-'*5} {'-'*4} {'-'*4} {'-'*4} {'-'*8} {'-'*8} {'-'*8}")

    total_gold = total_ta = total_tp = total_fp = total_fn = 0
    f1_sum = 0
    for name in DATASETS:
        s = all_stats[name]
        print(f"  {name:<20} {s['gold']:>5} {s['transarc']:>5} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} {fmt_pct(s['precision']):>8} {fmt_pct(s['recall']):>8} {fmt_pct(s['f1']):>8}")
        total_gold += s["gold"]
        total_ta += s["transarc"]
        total_tp += s["tp"]
        total_fp += s["fp"]
        total_fn += s["fn"]
        f1_sum += s["f1"]

    micro_p = total_tp / total_ta if total_ta else 0
    micro_r = total_tp / total_gold if total_gold else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0
    macro_f1 = f1_sum / len(DATASETS)

    print(f"  {'-'*20} {'-'*5} {'-'*5} {'-'*4} {'-'*4} {'-'*4} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'MICRO TOTAL':<20} {total_gold:>5} {total_ta:>5} {total_tp:>4} {total_fp:>4} {total_fn:>4} {fmt_pct(micro_p):>8} {fmt_pct(micro_r):>8} {fmt_pct(micro_f1):>8}")
    print(f"  {'MACRO AVERAGE':<20} {'':>5} {'':>5} {'':>4} {'':>4} {'':>4} {'':>8} {'':>8} {fmt_pct(macro_f1):>8}")
    print()


if __name__ == "__main__":
    main()
