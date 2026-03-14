#!/usr/bin/env python3
"""Analyze V39 Phase 9 Judge impact: TPs killed, FPs caught, FPs survived."""

import csv
import glob
import pickle
import sys

BENCHMARK_BASE = "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"

DATASETS = {
    "mediastore": {
        "text": f"{BENCHMARK_BASE}/mediastore/text_2016/mediastore.txt",
        "gold": f"{BENCHMARK_BASE}/mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    },
    "teastore": {
        "text": f"{BENCHMARK_BASE}/teastore/text_2020/teastore.txt",
        "gold": f"{BENCHMARK_BASE}/teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    },
    "teammates": {
        "text": f"{BENCHMARK_BASE}/teammates/text_2021/teammates.txt",
        "gold": f"{BENCHMARK_BASE}/teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "bigbluebutton": {
        "text": f"{BENCHMARK_BASE}/bigbluebutton/text_2021/bigbluebutton.txt",
        "gold": f"{BENCHMARK_BASE}/bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "jabref": {
        "text": f"{BENCHMARK_BASE}/jabref/text_2021/jabref.txt",
        "gold": f"{BENCHMARK_BASE}/jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
}


def load_gold(gold_path):
    """Load gold standard as set of (sentence_number, model_element_id)."""
    gold = set()
    with open(gold_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sent = int(row["sentence"])
            eid = row["modelElementID"]
            gold.add((sent, eid))
    return gold


def load_text(text_path):
    """Load text file, return dict of sentence_number -> text."""
    sentences = {}
    with open(text_path) as f:
        for i, line in enumerate(f, 1):
            sentences[i] = line.strip()
    return sentences


def links_to_set(links):
    """Convert list of SadSamLink to set of (sentence_number, component_id)."""
    return {(l.sentence_number, l.component_id) for l in links}


def links_to_dict(links):
    """Convert list of SadSamLink to dict keyed by (sentence_number, component_id)."""
    d = {}
    for l in links:
        d[(l.sentence_number, l.component_id)] = l
    return d


def classify(link_set, gold):
    """Split link set into TP and FP sets."""
    tp = link_set & gold
    fp = link_set - gold
    return tp, fp


def main():
    totals = {
        "pre_count": 0, "pre_tp": 0, "pre_fp": 0,
        "post_count": 0, "post_tp": 0, "post_fp": 0,
        "killed_tp": 0, "killed_fp": 0,
        "survived_fp": 0,
        "gold_count": 0,
    }

    for dataset_name in ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]:
        ds = DATASETS[dataset_name]
        pre_path = f"results/phase_cache/v39/{dataset_name}/pre_judge.pkl"
        final_path = f"results/phase_cache/v39/{dataset_name}/final.pkl"

        # Load data
        with open(pre_path, "rb") as f:
            pre_data = pickle.load(f)
        with open(final_path, "rb") as f:
            final_data = pickle.load(f)

        gold = load_gold(ds["gold"])
        sentences = load_text(ds["text"])

        preliminary = pre_data["preliminary"]
        final_links = final_data["final"]
        reviewed = final_data.get("reviewed", [])
        rejected = final_data.get("rejected", [])

        pre_set = links_to_set(preliminary)
        pre_dict = links_to_dict(preliminary)
        final_set = links_to_set(final_links)
        final_dict = links_to_dict(final_links)

        # Classify before/after
        pre_tp, pre_fp = classify(pre_set, gold)
        post_tp, post_fp = classify(final_set, gold)

        # Killed links = in pre but not in final
        killed_set = pre_set - final_set
        killed_tp = killed_set & gold
        killed_fp = killed_set - gold

        # Added links = in final but not in pre (shouldn't happen for judge, but check)
        added_set = final_set - pre_set

        # Surviving FPs
        survived_fp = post_fp

        # Pre-judge F1
        pre_p = len(pre_tp) / len(pre_set) * 100 if pre_set else 0
        pre_r = len(pre_tp) / len(gold) * 100 if gold else 0
        pre_f1 = 2 * pre_p * pre_r / (pre_p + pre_r) if (pre_p + pre_r) else 0

        # Post-judge F1
        post_p = len(post_tp) / len(final_set) * 100 if final_set else 0
        post_r = len(post_tp) / len(gold) * 100 if gold else 0
        post_f1 = 2 * post_p * post_r / (post_p + post_r) if (post_p + post_r) else 0

        # Update totals
        totals["pre_count"] += len(pre_set)
        totals["pre_tp"] += len(pre_tp)
        totals["pre_fp"] += len(pre_fp)
        totals["post_count"] += len(final_set)
        totals["post_tp"] += len(post_tp)
        totals["post_fp"] += len(post_fp)
        totals["killed_tp"] += len(killed_tp)
        totals["killed_fp"] += len(killed_fp)
        totals["survived_fp"] += len(survived_fp)
        totals["gold_count"] += len(gold)

        # Print header
        print("=" * 100)
        print(f"  {dataset_name.upper()}")
        print("=" * 100)
        print(f"  Gold: {len(gold)} links")
        print(f"  Pre-judge:  {len(pre_set):3d} links  (TP={len(pre_tp):2d}, FP={len(pre_fp):2d})  P={pre_p:5.1f}%  R={pre_r:5.1f}%  F1={pre_f1:5.1f}%")
        print(f"  Post-judge: {len(final_set):3d} links  (TP={len(post_tp):2d}, FP={len(post_fp):2d})  P={post_p:5.1f}%  R={post_r:5.1f}%  F1={post_f1:5.1f}%")
        print(f"  Reviewed: {len(reviewed)}, Rejected: {len(rejected)}")
        print(f"  Judge killed: {len(killed_set)} links  (TP killed={len(killed_tp)}, FP killed={len(killed_fp)})")
        if added_set:
            print(f"  Judge ADDED: {len(added_set)} links (unexpected!)")
            for key in added_set:
                link = final_dict[key]
                is_tp = key in gold
                label = "TP_ADDED" if is_tp else "FP_ADDED"
                stext = sentences.get(key[0], "???")[:80]
                print(f"    [{label}] s{key[0]:3d} {link.component_name:<30s} src={link.source:<16s} | {stext}")
        print(f"  Delta: F1 {pre_f1:5.1f}% -> {post_f1:5.1f}% ({post_f1 - pre_f1:+.1f}pp)")
        print()

        # Print killed links
        if killed_set:
            print("  --- KILLED BY JUDGE ---")
            for key in sorted(killed_set, key=lambda k: k[0]):
                link = pre_dict[key]
                is_tp = key in gold
                label = "TP_KILLED" if is_tp else "FP_KILLED"
                stext = sentences.get(key[0], "???")[:80]
                reasoning = ""
                # Check if this link appears in rejected list
                for rl in rejected:
                    if rl.sentence_number == key[0] and rl.component_id == key[1]:
                        if rl.reasoning:
                            reasoning = f"\n                 reason: {rl.reasoning[:120]}"
                        break
                print(f"    [{label}] s{key[0]:3d} {link.component_name:<30s} src={link.source:<16s} conf={link.confidence:.2f} | {stext}{reasoning}")
            print()

        # Print surviving FPs
        if survived_fp:
            print("  --- FPs SURVIVING JUDGE ---")
            for key in sorted(survived_fp, key=lambda k: k[0]):
                link = final_dict[key]
                stext = sentences.get(key[0], "???")[:80]
                print(f"    [FP_SURVIVED] s{key[0]:3d} {link.component_name:<30s} src={link.source:<16s} conf={link.confidence:.2f} | {stext}")
            print()

        # Print FNs (in gold but not in final)
        fn = gold - final_set
        if fn:
            print(f"  --- FALSE NEGATIVES ({len(fn)}) ---")
            for key in sorted(fn, key=lambda k: k[0]):
                stext = sentences.get(key[0], "???")[:80]
                # Check if this was in pre but killed
                was_killed = key in killed_tp
                tag = "KILLED_BY_JUDGE" if was_killed else "NEVER_FOUND"
                print(f"    [{tag}] s{key[0]:3d} id={key[1][:30]:<32s} | {stext}")
            print()

        # Breakdown by source
        source_counts = {}
        for key in killed_set:
            link = pre_dict[key]
            src = link.source
            is_tp = key in gold
            if src not in source_counts:
                source_counts[src] = {"tp_killed": 0, "fp_killed": 0}
            if is_tp:
                source_counts[src]["tp_killed"] += 1
            else:
                source_counts[src]["fp_killed"] += 1

        if source_counts:
            print("  --- KILLED BY SOURCE ---")
            for src, counts in sorted(source_counts.items()):
                print(f"    {src:<20s}: TP_killed={counts['tp_killed']}, FP_killed={counts['fp_killed']}")
            print()

    # Summary table
    print()
    print("=" * 100)
    print("  SUMMARY TABLE")
    print("=" * 100)
    print()
    print(f"  {'Dataset':<16s} {'Gold':>5s} {'Pre':>5s} {'Pre_TP':>7s} {'Pre_FP':>7s} {'Post':>5s} {'Post_TP':>8s} {'Post_FP':>8s} {'Kill_TP':>8s} {'Kill_FP':>8s} {'Pre_F1':>7s} {'Post_F1':>8s} {'Delta':>7s}")
    print("  " + "-" * 95)

    macro_pre_f1 = 0
    macro_post_f1 = 0

    for dataset_name in ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]:
        ds = DATASETS[dataset_name]
        pre_path = f"results/phase_cache/v39/{dataset_name}/pre_judge.pkl"
        final_path = f"results/phase_cache/v39/{dataset_name}/final.pkl"

        with open(pre_path, "rb") as f:
            pre_data = pickle.load(f)
        with open(final_path, "rb") as f:
            final_data = pickle.load(f)

        gold = load_gold(ds["gold"])
        preliminary = pre_data["preliminary"]
        final_links = final_data["final"]

        pre_set = links_to_set(preliminary)
        final_set = links_to_set(final_links)

        pre_tp, pre_fp = classify(pre_set, gold)
        post_tp, post_fp = classify(final_set, gold)

        killed_set = pre_set - final_set
        killed_tp = killed_set & gold
        killed_fp = killed_set - gold

        pre_p = len(pre_tp) / len(pre_set) * 100 if pre_set else 0
        pre_r = len(pre_tp) / len(gold) * 100 if gold else 0
        pre_f1 = 2 * pre_p * pre_r / (pre_p + pre_r) if (pre_p + pre_r) else 0

        post_p = len(post_tp) / len(final_set) * 100 if final_set else 0
        post_r = len(post_tp) / len(gold) * 100 if gold else 0
        post_f1 = 2 * post_p * post_r / (post_p + post_r) if (post_p + post_r) else 0

        macro_pre_f1 += pre_f1
        macro_post_f1 += post_f1

        print(f"  {dataset_name:<16s} {len(gold):5d} {len(pre_set):5d} {len(pre_tp):7d} {len(pre_fp):7d} {len(final_set):5d} {len(post_tp):8d} {len(post_fp):8d} {len(killed_tp):8d} {len(killed_fp):8d} {pre_f1:6.1f}% {post_f1:7.1f}% {post_f1-pre_f1:+6.1f}%")

    macro_pre_f1 /= 5
    macro_post_f1 /= 5

    print("  " + "-" * 95)
    print(f"  {'TOTAL':<16s} {totals['gold_count']:5d} {totals['pre_count']:5d} {totals['pre_tp']:7d} {totals['pre_fp']:7d} {totals['post_count']:5d} {totals['post_tp']:8d} {totals['post_fp']:8d} {totals['killed_tp']:8d} {totals['killed_fp']:8d}")
    print()
    print(f"  Macro-avg Pre-judge  F1: {macro_pre_f1:.1f}%")
    print(f"  Macro-avg Post-judge F1: {macro_post_f1:.1f}%")
    print(f"  Judge delta:             {macro_post_f1 - macro_pre_f1:+.1f}pp")
    print()
    print(f"  Total links killed: {totals['killed_tp'] + totals['killed_fp']}")
    print(f"    TP killed (BAD):  {totals['killed_tp']}")
    print(f"    FP killed (GOOD): {totals['killed_fp']}")
    print(f"  Surviving FPs:      {totals['survived_fp']}")
    print()

    if totals['killed_tp'] + totals['killed_fp'] > 0:
        precision_of_kills = totals['killed_fp'] / (totals['killed_tp'] + totals['killed_fp']) * 100
        print(f"  Judge kill precision (FP/(TP+FP) killed): {precision_of_kills:.1f}%")
    print()


if __name__ == "__main__":
    main()
