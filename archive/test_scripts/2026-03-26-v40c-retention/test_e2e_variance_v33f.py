#!/usr/bin/env python3
"""Run V33f E2E 3 times per dataset, diff finals, trace variant links to source phase.

Same as test_e2e_variance.py but uses V33f (forward coref ×2 union).
"""

import csv
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

os.environ["CLAUDE_MODEL"] = "sonnet"

from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMBackend

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)

DATASETS = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
        "gold": BENCHMARK_BASE / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
        "model": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
        "gold": BENCHMARK_BASE / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
        "gold": BENCHMARK_BASE / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
        "gold": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
        "gold": BENCHMARK_BASE / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
}

N_RUNS = 3


def load_gold(gold_path):
    links = set()
    with open(gold_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def run_v33f_e2e(text_path, model_path, run_id):
    """Run full V33f pipeline, return final links with source info."""
    os.environ["PHASE_CACHE_DIR"] = f"./results/phase_cache_variance_v33f/run{run_id}"

    from llm_sad_sam.linkers.experimental.ilinker2_v33f import ILinker2V33f
    linker = ILinker2V33f(backend=LLMBackend.CLAUDE)
    final_links = linker.link(text_path, model_path)
    return final_links


def main():
    selected = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())

    # Cross-dataset accumulators
    total_variant_by_source = Counter()
    total_variant_tp_by_source = Counter()
    total_variant_fp_by_source = Counter()

    for ds_name in selected:
        paths = DATASETS[ds_name]
        gold = load_gold(str(paths["gold"]))
        components = parse_pcm_repository(str(paths["model"]))
        id_to_name = {c.id: c.name for c in components}
        text_path = str(paths["text"])
        model_path = str(paths["model"])

        print(f"\n{'='*100}")
        print(f"DATASET: {ds_name} (gold={len(gold)} links)")
        print(f"{'='*100}")

        runs = []  # list of (link_set, source_map)
        for i in range(N_RUNS):
            print(f"\n  ╔══ Run {i+1}/{N_RUNS} ══╗")
            t0 = time.time()
            final_links = run_v33f_e2e(text_path, model_path, i)
            elapsed = time.time() - t0

            link_set = {(l.sentence_number, l.component_id) for l in final_links}
            source_map = {(l.sentence_number, l.component_id): l.source for l in final_links}

            tp = len(link_set & gold)
            fp = len(link_set - gold)
            fn = len(gold - link_set)
            p = tp / (tp + fp) if (tp + fp) else 0
            r = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * p * r / (p + r) if (p + r) else 0

            runs.append((link_set, source_map))

            # Source breakdown
            src_counts = Counter(l.source for l in final_links)
            print(f"  Run {i+1}: {len(link_set)} links, {tp} TP, {fp} FP, "
                  f"P={p:.1%} R={r:.1%} F1={f1:.1%} ({elapsed:.0f}s)")
            print(f"    Sources: {dict(src_counts)}")

        # ── Variance analysis ──
        all_sets = [r[0] for r in runs]
        all_sources = [r[1] for r in runs]

        union = set()
        inter = all_sets[0].copy()
        for s in all_sets:
            union |= s
            inter &= s

        variant = union - inter
        stable = inter

        link_counts = Counter()
        for s in all_sets:
            for key in s:
                link_counts[key] += 1

        print(f"\n  FINAL VARIANCE:")
        print(f"    Stable (3/3):  {len(stable)} links")
        print(f"    Variant:       {len(variant)} links")
        print(f"    Total union:   {len(union)} links")

        # ── Trace variant links to source phase ──
        variant_by_source = Counter()  # source -> count
        variant_tp_by_source = Counter()
        variant_fp_by_source = Counter()

        if variant:
            print(f"\n  VARIANT LINKS BY SOURCE PHASE:")
            for s, c in sorted(variant):
                is_tp = (s, c) in gold
                count = link_counts[(s, c)]
                # Find source from whichever run has it
                source = "?"
                for src_map in all_sources:
                    if (s, c) in src_map:
                        source = src_map[(s, c)]
                        break
                variant_by_source[source] += 1
                if is_tp:
                    variant_tp_by_source[source] += 1
                else:
                    variant_fp_by_source[source] += 1
                print(f"    {'TP' if is_tp else 'FP'}: S{s} -> {id_to_name.get(c, c)} "
                      f"[{source}] ({count}/{N_RUNS} runs)")

            print(f"\n  VARIANCE BY SOURCE:")
            for src in sorted(variant_by_source.keys()):
                total = variant_by_source[src]
                tp = variant_tp_by_source[src]
                fp = variant_fp_by_source[src]
                print(f"    {src:20s}: {total:2d} variant links ({tp} TP, {fp} FP)")

            total_variant_by_source += variant_by_source
            total_variant_tp_by_source += variant_tp_by_source
            total_variant_fp_by_source += variant_fp_by_source

        # ── Stable links by source ──
        stable_by_source = Counter()
        for key in stable:
            for src_map in all_sources:
                if key in src_map:
                    stable_by_source[src_map[key]] += 1
                    break

        print(f"\n  STABLE LINKS BY SOURCE:")
        for src in sorted(stable_by_source.keys()):
            print(f"    {src:20s}: {stable_by_source[src]:3d} stable links")

    # ── Cross-dataset summary ──
    print(f"\n{'='*100}")
    print(f"CROSS-DATASET VARIANCE SUMMARY (V33f: forward coref ×2 union)")
    print(f"{'='*100}")
    if total_variant_by_source:
        for src in sorted(total_variant_by_source.keys()):
            total = total_variant_by_source[src]
            tp = total_variant_tp_by_source[src]
            fp = total_variant_fp_by_source[src]
            print(f"  {src:20s}: {total:2d} variant links ({tp} TP, {fp} FP)")
        grand_total = sum(total_variant_by_source.values())
        print(f"  {'TOTAL':20s}: {grand_total:2d} variant links")
    else:
        print(f"  No variance detected!")


if __name__ == "__main__":
    main()
