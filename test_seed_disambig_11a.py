#!/usr/bin/env python3
"""Unit test: S-Linker11a seed disambiguation on S-Linker11 checkpoint data.

Loads S-Linker11 layer1+layer2 checkpoints, instantiates SLinker11a,
injects knowledge state, and runs only the seed disambiguation step.
Compares results against gold standards.

This is NOT a full e2e run — it tests only the seed disambiguation
method using pre-computed knowledge from S-Linker11 checkpoints.
"""

import csv
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
os.environ["CLAUDE_MODEL"] = "sonnet"

from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker11a import SLinker11a

# ─────────────────────────────────────────────────────────────────────────────
DATASETS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
CACHE_DIR = Path("results/phase_cache/s_linker11")
BENCHMARK = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)


def load_gold(dataset: str) -> set[tuple[int, str]]:
    gs_dir = BENCHMARK / dataset / "goldstandards"
    candidates = [
        f for f in gs_dir.glob("goldstandard_sad*sam*.csv")
        if "UME" not in f.name and "MME" not in f.name
    ]
    assert candidates, f"No gold standard in {gs_dir}"
    links = set()
    with open(candidates[0]) as f:
        for row in csv.DictReader(f):
            sent = int(row.get("sentence", row.get("sentenceNo", 0)))
            links.add((sent, row["modelElementID"]))
    return links


def main():
    linker = SLinker11a()

    print("=" * 70)
    print("UNIT TEST: S-Linker11a seed disambiguation")
    print("=" * 70)

    results = []

    for dataset in DATASETS:
        print(f"\n{'─' * 60}")
        print(f"  {dataset}")
        print(f"{'─' * 60}")

        # Load S-Linker11 checkpoints
        l1 = pickle.load(open(CACHE_DIR / dataset / "layer1.pkl", "rb"))
        l2 = pickle.load(open(CACHE_DIR / dataset / "layer2.pkl", "rb"))

        raw_seeds = l1["raw_seed_links"]
        linker.model_knowledge = l1["model_knowledge"]
        linker.doc_knowledge = l2["doc_knowledge"]

        # Load components and sentences
        model_path = list((BENCHMARK / dataset).glob("model_*/pcm/*.repository"))[0]
        components = parse_pcm_repository(str(model_path))
        text_path = list((BENCHMARK / dataset).glob(f"text_*/{dataset}.txt"))[0]
        sentences = load_sentences(str(text_path))
        sent_map = build_sent_map(sentences)

        gold = load_gold(dataset)

        # Classify raw seeds
        raw_tp = sum(1 for s in raw_seeds if (s.sentence_number, s.component_id) in gold)
        raw_fp = len(raw_seeds) - raw_tp

        print(f"  Raw seeds: {len(raw_seeds)} ({raw_tp} TP, {raw_fp} FP)")

        # Run ONLY the seed disambiguation (the method under test)
        t0 = time.time()
        disambiguated = linker._run_seed_validation(raw_seeds, components, sent_map)
        elapsed = time.time() - t0

        # Evaluate
        dis_set = {(s.sentence_number, s.component_id) for s in disambiguated}
        raw_set = {(s.sentence_number, s.component_id) for s in raw_seeds}
        killed = raw_set - dis_set

        tp_killed = fp_caught = 0
        for sl in raw_seeds:
            key = (sl.sentence_number, sl.component_id)
            if key in killed:
                sent = sent_map.get(sl.sentence_number)
                txt = f'"{sent.text[:70]}"' if sent else "?"
                if key in gold:
                    tp_killed += 1
                    print(f"  ⚠ TP KILLED: S{sl.sentence_number} -> {sl.component_name}: {txt}")
                else:
                    fp_caught += 1
                    print(f"  ✓ FP caught:  S{sl.sentence_number} -> {sl.component_name}: {txt}")

        dis_tp = raw_tp - tp_killed
        dis_fp = raw_fp - fp_caught

        results.append({
            "ds": dataset,
            "raw": len(raw_seeds), "raw_tp": raw_tp, "raw_fp": raw_fp,
            "dis": len(disambiguated), "dis_tp": dis_tp, "dis_fp": dis_fp,
            "kill_tp": tp_killed, "kill_fp": fp_caught,
            "time": elapsed,
        })
        print(f"  → {len(disambiguated)}/{len(raw_seeds)} kept "
              f"({dis_tp} TP, {dis_fp} FP) in {elapsed:.1f}s")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    hdr = f"{'Dataset':<15} {'Raw':>4} {'TP':>4} {'FP':>4} │ {'Dis':>4} {'TP':>4} {'FP':>4} │ {'K.TP':>5} {'K.FP':>5}"
    print(hdr)
    print("─" * len(hdr))
    for r in results:
        print(f"{r['ds']:<15} {r['raw']:>4} {r['raw_tp']:>4} {r['raw_fp']:>4} │ "
              f"{r['dis']:>4} {r['dis_tp']:>4} {r['dis_fp']:>4} │ "
              f"{r['kill_tp']:>5} {r['kill_fp']:>5}")

    t = {k: sum(r[k] for r in results)
         for k in ["raw", "raw_tp", "raw_fp", "dis", "dis_tp", "dis_fp", "kill_tp", "kill_fp"]}
    print("─" * len(hdr))
    print(f"{'TOTAL':<15} {t['raw']:>4} {t['raw_tp']:>4} {t['raw_fp']:>4} │ "
          f"{t['dis']:>4} {t['dis_tp']:>4} {t['dis_fp']:>4} │ "
          f"{t['kill_tp']:>5} {t['kill_fp']:>5}")

    print(f"\n  TP preservation: {t['dis_tp']}/{t['raw_tp']} "
          f"({100 * t['dis_tp'] / t['raw_tp']:.1f}%)")
    print(f"  FP rejection:    {t['kill_fp']}/{t['raw_fp']} "
          f"({100 * t['kill_fp'] / max(t['raw_fp'], 1):.1f}%)")

    # Assertions
    assert t["kill_tp"] == 0, f"FAIL: {t['kill_tp']} TPs killed!"
    assert t["kill_fp"] >= 2, f"FAIL: only {t['kill_fp']} FPs caught (expected ≥2)"
    print(f"\n  ✓ PASS: zero TP kills, {t['kill_fp']} FPs caught")


if __name__ == "__main__":
    main()
