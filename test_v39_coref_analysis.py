#!/usr/bin/env python3
"""Analyze V39 Phase 7 coref links: which are TPs and which are FPs."""

import csv
import glob
import pickle
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from llm_sad_sam.core.document_loader import DocumentLoader

BENCHMARK = "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
CACHE = "results/phase_cache/v39"

DATASETS = {
    "mediastore": {
        "text": f"{BENCHMARK}/mediastore/text_2016/mediastore.txt",
        "gold_pattern": f"{BENCHMARK}/mediastore/goldstandards/goldstandard_sad_*-sam_*.csv",
    },
    "teastore": {
        "text": f"{BENCHMARK}/teastore/text_2020/teastore.txt",
        "gold_pattern": f"{BENCHMARK}/teastore/goldstandards/goldstandard_sad_*-sam_*.csv",
    },
    "teammates": {
        "text": f"{BENCHMARK}/teammates/text_2021/teammates.txt",
        "gold_pattern": f"{BENCHMARK}/teammates/goldstandards/goldstandard_sad_*-sam_*.csv",
    },
    "bigbluebutton": {
        "text": f"{BENCHMARK}/bigbluebutton/text_2021/bigbluebutton.txt",
        "gold_pattern": f"{BENCHMARK}/bigbluebutton/goldstandards/goldstandard_sad_*-sam_*.csv",
    },
    "jabref": {
        "text": f"{BENCHMARK}/jabref/text_2021/jabref.txt",
        "gold_pattern": f"{BENCHMARK}/jabref/goldstandards/goldstandard_sad_*-sam_*.csv",
    },
}


def load_gold_standard(pattern: str) -> set[tuple[str, int]]:
    """Load gold standard as set of (modelElementID, sentence_number) tuples."""
    gold = set()
    for path in glob.glob(pattern):
        if "UME" in path or "MME" in path:
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_id = row["modelElementID"]
                sent = int(row["sentence"])
                gold.add((model_id, sent))
    return gold


def main():
    total_tp = 0
    total_fp = 0
    total_coref = 0

    for ds_name in ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]:
        ds = DATASETS[ds_name]

        # Load phase0 for is_complex
        with open(f"{CACHE}/{ds_name}/phase0.pkl", "rb") as f:
            phase0 = pickle.load(f)
        is_complex = phase0["is_complex"]
        mode = "debate" if is_complex else "discourse"

        # Load phase6 for validated count
        with open(f"{CACHE}/{ds_name}/phase6.pkl", "rb") as f:
            phase6 = pickle.load(f)
        validated_count = len(phase6["validated"])

        # Load phase7 coref links
        with open(f"{CACHE}/{ds_name}/phase7.pkl", "rb") as f:
            phase7 = pickle.load(f)
        coref_links = phase7["coref_links"]

        # Load gold standard
        gold = load_gold_standard(ds["gold_pattern"])

        # Load sentences for text lookup
        sentences = DocumentLoader.load_sentences(ds["text"])

        # Classify each coref link
        tp_links = []
        fp_links = []
        for link in coref_links:
            key = (link.component_id, link.sentence_number)
            if key in gold:
                tp_links.append(link)
            else:
                fp_links.append(link)

        tp_count = len(tp_links)
        fp_count = len(fp_links)
        total_tp += tp_count
        total_fp += fp_count
        total_coref += len(coref_links)

        # Print header
        print("=" * 100)
        print(f"DATASET: {ds_name}")
        print(f"  is_complex: {is_complex}  |  coref mode: {mode}")
        print(f"  validated links (Phase 6): {validated_count}")
        print(f"  coref links (Phase 7):     {len(coref_links)}  (TP: {tp_count}, FP: {fp_count})")
        print()

        # Build sentence lookup by number
        sent_by_num = {s.number: s.text for s in sentences}

        # Print each coref link with details
        for link in sorted(coref_links, key=lambda l: l.sentence_number):
            key = (link.component_id, link.sentence_number)
            label = "TP" if key in gold else "FP"
            # Get sentence text
            sent_text = sent_by_num.get(link.sentence_number, "<not found>")

            # Truncate long sentences
            if len(sent_text) > 120:
                sent_text = sent_text[:117] + "..."

            confidence_str = f"{link.confidence:.2f}" if link.confidence is not None else "N/A"
            reasoning_str = ""
            if link.reasoning:
                r = link.reasoning
                if len(r) > 80:
                    r = r[:77] + "..."
                reasoning_str = f"  reason: {r}"

            print(f"  [{label}] S{link.sentence_number:3d} -> {link.component_name:<30s} "
                  f"(conf={confidence_str}){reasoning_str}")
            print(f"         text: {sent_text}")
            print()

    print("=" * 100)
    print(f"TOTALS: {total_coref} coref links  |  TP: {total_tp}  |  FP: {total_fp}")
    print(f"  Coref precision: {total_tp / total_coref * 100:.1f}%" if total_coref > 0 else "  No coref links")


if __name__ == "__main__":
    main()
