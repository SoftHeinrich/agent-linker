#!/usr/bin/env python3
"""
Offline analysis: What happens if TransArc immunity is removed from V39a Phase 9 judge?

TransArc immunity means links with source="transarc" skip the judge entirely (auto-approved).
This analysis loads pre_judge checkpoints and cross-references against gold standards
to determine how many TransArc TPs would be at risk vs how many TransArc FPs could be caught.

No LLM calls — purely offline checkpoint + gold standard analysis.
"""

import csv
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

# ── Gold standard and checkpoint paths ──────────────────────────────────────

BENCHMARK_DIR = Path("/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark")
CHECKPOINT_DIR = Path("/mnt/hostshare/ardoco-home/llm-sad-sam-v45/results/phase_cache/v39a")

# Dataset name -> gold standard file (excluding UME and code files)
GOLD_STANDARD_FILES = {
    "mediastore": BENCHMARK_DIR / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    "teastore": BENCHMARK_DIR / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    "teammates": BENCHMARK_DIR / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "bigbluebutton": BENCHMARK_DIR / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "jabref": BENCHMARK_DIR / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
}

DATASETS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]


def load_gold_standard(dataset: str) -> set[tuple[int, str]]:
    """Load gold standard as set of (sentence_number, component_id) pairs."""
    gs_path = GOLD_STANDARD_FILES[dataset]
    gold = set()
    with open(gs_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sent = int(row["sentence"])
            comp_id = row["modelElementID"]
            gold.add((sent, comp_id))
    return gold


def load_pre_judge_checkpoint(dataset: str) -> dict:
    """Load the pre_judge checkpoint for a dataset."""
    pkl_path = CHECKPOINT_DIR / dataset / "pre_judge.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"No pre_judge checkpoint for {dataset}: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_final_checkpoint(dataset: str) -> dict:
    """Load the final checkpoint to see post-judge results."""
    pkl_path = CHECKPOINT_DIR / dataset / "final.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"No final checkpoint for {dataset}: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def analyze_dataset(dataset: str):
    """Analyze TransArc immunity impact for one dataset."""
    gold = load_gold_standard(dataset)
    checkpoint = load_pre_judge_checkpoint(dataset)
    final_checkpoint = load_final_checkpoint(dataset)

    preliminary_links = checkpoint["preliminary"]
    transarc_set = checkpoint["transarc_set"]
    final_links = final_checkpoint["final"]

    # Partition preliminary links by source
    transarc_links = []
    non_transarc_links = []
    for link in preliminary_links:
        key = (link.sentence_number, link.component_id)
        if link.source == "transarc":
            transarc_links.append(link)
        else:
            non_transarc_links.append(link)

    # Classify TransArc links as TP or FP against gold standard
    transarc_tp = []
    transarc_fp = []
    for link in transarc_links:
        key = (link.sentence_number, link.component_id)
        if key in gold:
            transarc_tp.append(link)
        else:
            transarc_fp.append(link)

    # Classify non-TransArc links
    non_transarc_tp = []
    non_transarc_fp = []
    for link in non_transarc_links:
        key = (link.sentence_number, link.component_id)
        if key in gold:
            non_transarc_tp.append(link)
        else:
            non_transarc_fp.append(link)

    # Check which links survived Phase 9 judge (final output)
    final_set = {(l.sentence_number, l.component_id) for l in final_links}

    # Among TransArc links, how many were actually kept vs rejected by judge?
    # (In V39a, TransArc links get immunity OR go to ta_review if ambiguous)
    transarc_kept = [l for l in transarc_links if (l.sentence_number, l.component_id) in final_set]
    transarc_rejected = [l for l in transarc_links if (l.sentence_number, l.component_id) not in final_set]

    # Among non-TransArc links, how many survived?
    non_transarc_kept = [l for l in non_transarc_links if (l.sentence_number, l.component_id) in final_set]
    non_transarc_rejected = [l for l in non_transarc_links if (l.sentence_number, l.component_id) not in final_set]

    # Gold standard links NOT in our output = FN
    all_final_set = final_set
    fn_links = gold - all_final_set

    # Calculate overall metrics for final output
    total_tp = sum(1 for l in final_links if (l.sentence_number, l.component_id) in gold)
    total_fp = len(final_links) - total_tp
    total_fn = len(gold) - total_tp
    precision = total_tp / len(final_links) if final_links else 0
    recall = total_tp / len(gold) if gold else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "dataset": dataset,
        "gold_count": len(gold),
        "preliminary_count": len(preliminary_links),
        "final_count": len(final_links),
        # TransArc breakdown
        "transarc_total": len(transarc_links),
        "transarc_tp": transarc_tp,
        "transarc_fp": transarc_fp,
        "transarc_kept": len(transarc_kept),
        "transarc_rejected": len(transarc_rejected),
        # Non-TransArc breakdown
        "non_transarc_total": len(non_transarc_links),
        "non_transarc_tp": non_transarc_tp,
        "non_transarc_fp": non_transarc_fp,
        "non_transarc_kept": len(non_transarc_kept),
        "non_transarc_rejected": len(non_transarc_rejected),
        # Overall metrics
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_separator(char="=", width=80):
    print(char * width)


def main():
    print_separator()
    print("TransArc Immunity Impact Analysis — V39a Pipeline")
    print("What happens if TransArc immunity is removed from Phase 9 judge?")
    print_separator()

    all_results = []
    totals = defaultdict(int)

    for dataset in DATASETS:
        checkpoint_path = CHECKPOINT_DIR / dataset / "pre_judge.pkl"
        if not checkpoint_path.exists():
            print(f"\n  SKIP {dataset}: no pre_judge.pkl checkpoint")
            continue

        result = analyze_dataset(dataset)
        all_results.append(result)

        print(f"\n{'─' * 80}")
        print(f"  Dataset: {dataset.upper()}")
        print(f"{'─' * 80}")
        print(f"  Gold standard links:  {result['gold_count']}")
        print(f"  Preliminary links:    {result['preliminary_count']}")
        print(f"  Final links (post-9): {result['final_count']}")
        print(f"  Current metrics:      P={result['precision']:.1%}  R={result['recall']:.1%}  F1={result['f1']:.1%}")
        print(f"  Current:              {result['total_tp']} TP, {result['total_fp']} FP, {result['total_fn']} FN")

        print(f"\n  TransArc links: {result['transarc_total']}")
        print(f"    TPs (at risk if immunity removed):  {len(result['transarc_tp'])}")
        print(f"    FPs (could be caught):              {len(result['transarc_fp'])}")
        print(f"    Currently kept by judge:             {result['transarc_kept']}")
        print(f"    Currently rejected by judge:         {result['transarc_rejected']}")

        if result["transarc_fp"]:
            print(f"\n    TransArc FPs (immunity protects these false positives):")
            for link in result["transarc_fp"]:
                kept_marker = "KEPT" if (link.sentence_number, link.component_id) in {(l.sentence_number, l.component_id) for l in all_results[-1]["transarc_fp"]} else "REJ"
                print(f"      sent={link.sentence_number:3d}  {link.component_name:<30s}  conf={link.confidence:.2f}")

        if result["transarc_tp"]:
            print(f"\n    TransArc TPs (immunity protects these true positives):")
            for link in result["transarc_tp"]:
                print(f"      sent={link.sentence_number:3d}  {link.component_name:<30s}  conf={link.confidence:.2f}")

        print(f"\n  Non-TransArc links: {result['non_transarc_total']}")
        print(f"    TPs: {len(result['non_transarc_tp'])}")
        print(f"    FPs: {len(result['non_transarc_fp'])}")
        print(f"    Kept after judge:    {result['non_transarc_kept']}")
        print(f"    Rejected by judge:   {result['non_transarc_rejected']}")

        if result["non_transarc_fp"]:
            print(f"\n    Non-TransArc FPs (already subject to judge):")
            for link in result["non_transarc_fp"]:
                in_final = (link.sentence_number, link.component_id) in {(l.sentence_number, l.component_id) for l in all_results[-1]["non_transarc_fp"] if True}
                status = "SURVIVED" if (link.sentence_number, link.component_id) in {(l.sentence_number, l.component_id) for l in result.get("_final_links", []) or []} else ""
                print(f"      sent={link.sentence_number:3d}  {link.component_name:<30s}  src={link.source:<16s}  conf={link.confidence:.2f}")

        # Accumulate totals
        totals["transarc_total"] += result["transarc_total"]
        totals["transarc_tp"] += len(result["transarc_tp"])
        totals["transarc_fp"] += len(result["transarc_fp"])
        totals["non_transarc_total"] += result["non_transarc_total"]
        totals["non_transarc_tp"] += len(result["non_transarc_tp"])
        totals["non_transarc_fp"] += len(result["non_transarc_fp"])
        totals["total_tp"] += result["total_tp"]
        totals["total_fp"] += result["total_fp"]
        totals["total_fn"] += result["total_fn"]
        totals["gold_count"] += result["gold_count"]
        totals["final_count"] += result["final_count"]
        totals["transarc_kept"] += result["transarc_kept"]
        totals["transarc_rejected"] += result["transarc_rejected"]

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("AGGREGATE SUMMARY")
    print(f"{'=' * 80}")

    print(f"\n  Total TransArc links across all datasets:  {totals['transarc_total']}")
    print(f"    True Positives (AT RISK if immunity removed): {totals['transarc_tp']}")
    print(f"    False Positives (COULD BE caught):            {totals['transarc_fp']}")
    print(f"    Currently kept by judge:                      {totals['transarc_kept']}")
    print(f"    Currently rejected by judge:                  {totals['transarc_rejected']}")

    print(f"\n  Total non-TransArc links:                      {totals['non_transarc_total']}")
    print(f"    True Positives:                               {totals['non_transarc_tp']}")
    print(f"    False Positives:                              {totals['non_transarc_fp']}")

    print(f"\n  Current overall: {totals['total_tp']} TP, {totals['total_fp']} FP, {totals['total_fn']} FN")
    overall_p = totals['total_tp'] / totals['final_count'] if totals['final_count'] else 0
    overall_r = totals['total_tp'] / totals['gold_count'] if totals['gold_count'] else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
    print(f"  Current micro-avg: P={overall_p:.1%}  R={overall_r:.1%}  F1={overall_f1:.1%}")

    # ── Risk assessment ─────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("RISK ASSESSMENT: Removing TransArc Immunity")
    print(f"{'─' * 80}")

    ta_tp = totals["transarc_tp"]
    ta_fp = totals["transarc_fp"]
    ratio = ta_tp / ta_fp if ta_fp > 0 else float('inf')

    print(f"\n  Potential gain:  Up to {ta_fp} FPs could be caught by judge")
    print(f"  Potential risk:  Up to {ta_tp} TPs could be killed by judge")
    print(f"  TP:FP ratio:     {ratio:.1f}:1 (for every FP catchable, {ratio:.1f} TPs at risk)")

    print(f"\n  WORST CASE (judge kills ALL TransArc links):")
    worst_tp = totals["total_tp"] - ta_tp
    worst_fp = totals["total_fp"] - ta_fp
    worst_fn = totals["total_fn"] + ta_tp
    worst_p = worst_tp / (worst_tp + worst_fp) if (worst_tp + worst_fp) else 0
    worst_r = worst_tp / totals["gold_count"] if totals["gold_count"] else 0
    worst_f1 = 2 * worst_p * worst_r / (worst_p + worst_r) if (worst_p + worst_r) > 0 else 0
    print(f"    TP={worst_tp}, FP={worst_fp}, FN={worst_fn}")
    print(f"    P={worst_p:.1%}  R={worst_r:.1%}  F1={worst_f1:.1%}")

    print(f"\n  BEST CASE (judge catches ALL FPs, kills ZERO TPs):")
    best_tp = totals["total_tp"]
    best_fp = totals["total_fp"] - ta_fp
    best_fn = totals["total_fn"]
    best_p = best_tp / (best_tp + best_fp) if (best_tp + best_fp) else 0
    best_r = best_tp / totals["gold_count"] if totals["gold_count"] else 0
    best_f1 = 2 * best_p * best_r / (best_p + best_r) if (best_p + best_r) > 0 else 0
    print(f"    TP={best_tp}, FP={best_fp}, FN={best_fn}")
    print(f"    P={best_p:.1%}  R={best_r:.1%}  F1={best_f1:.1%}")

    # ── Per-dataset summary table ───────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("PER-DATASET SUMMARY TABLE")
    print(f"{'─' * 80}")
    print(f"  {'Dataset':<15s} {'TA Total':>8s} {'TA TP':>6s} {'TA FP':>6s} {'Non-TA':>7s} {'F1':>7s}")
    print(f"  {'─'*15} {'─'*8} {'─'*6} {'─'*6} {'─'*7} {'─'*7}")

    macro_f1_sum = 0
    for r in all_results:
        print(f"  {r['dataset']:<15s} {r['transarc_total']:>8d} {len(r['transarc_tp']):>6d} {len(r['transarc_fp']):>6d} {r['non_transarc_total']:>7d} {r['f1']:>6.1%}")
        macro_f1_sum += r["f1"]

    macro_f1 = macro_f1_sum / len(all_results) if all_results else 0
    print(f"\n  Macro-avg F1 (current): {macro_f1:.1%}")
    print(f"\n  Key insight: {ta_tp} TransArc TPs at risk to catch just {ta_fp} TransArc FPs.")
    print(f"  V29 experiment confirmed: removing immunity caused catastrophic TP loss (77.7% F1).")
    print(f"  The judge is not calibrated to handle TransArc links safely.")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
