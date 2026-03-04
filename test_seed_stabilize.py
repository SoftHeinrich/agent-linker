#!/usr/bin/env python3
"""Test two seed stabilization strategies for V33 ILinker2 variance.

Strategy 1 — "seed_majority_3":
  Run ILinker2 (Phase 4) three times with fixed Phase 1/3 inputs.
  Keep links found in ≥2/3 runs (majority vote).
  Cost: 3× Phase 4 only.

Strategy 2 — "seed_union_2":
  Run ILinker2 twice with fixed Phase 1/3 inputs.
  Take union of both runs (let downstream Phases 5-9 filter noise).
  Cost: 2× Phase 4 only.

Both strategies reuse the SAME Phase 1/3 checkpoint — no rerun of those phases.
Compares against single-run baseline from checkpoint.
"""

import csv
import os
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

os.environ["CLAUDE_MODEL"] = "sonnet"

from llm_sad_sam.core.data_types import SadSamLink
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


def load_gold(gold_path):
    links = set()
    with open(gold_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def load_checkpoint(ds_name, phase):
    path = Path(f"./results/phase_cache/v33/{ds_name}/{phase}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def run_ilinker2_once(text_path, model_path):
    """Run ILinker2 once and return seed set of (sentence_number, component_id)."""
    from llm_sad_sam.linkers.experimental.ilinker2 import ILinker2
    ilinker = ILinker2(backend=LLMBackend.CLAUDE)
    raw_links = ilinker.link(text_path, model_path)
    return {(lk.sentence_number, lk.component_id) for lk in raw_links}


def setup_linker_state(ds_name, text_path, model_path):
    """Load Phase 1/3 checkpoints into a V33 linker (for context, not re-run)."""
    p1 = load_checkpoint(ds_name, "phase1")
    p3 = load_checkpoint(ds_name, "phase3")
    return p1, p3


def eval_seed(seed_set, gold):
    tp = len(seed_set & gold)
    fp = len(seed_set - gold)
    fn = len(gold - seed_set)
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return tp, fp, fn, p, r, f1


def main():
    selected = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())

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

        # Load baseline from checkpoint
        p4 = load_checkpoint(ds_name, "phase4")
        baseline_set = {(l.sentence_number, l.component_id) for l in p4["transarc_links"]}
        b_tp, b_fp, b_fn, b_p, b_r, b_f1 = eval_seed(baseline_set, gold)
        print(f"\n  BASELINE (checkpoint): {len(baseline_set)} links, {b_tp} TP, {b_fp} FP, P={b_p:.1%} R={b_r:.1%} F1={b_f1:.1%}")

        # Run ILinker2 3 times (reuses same Phase 1/3 — ILinker2 is independent)
        runs = []
        for i in range(3):
            print(f"\n  --- ILinker2 run {i+1}/3 ---")
            t0 = time.time()
            seed = run_ilinker2_once(text_path, model_path)
            elapsed = time.time() - t0
            runs.append(seed)
            tp, fp, fn, p, r, f1 = eval_seed(seed, gold)
            print(f"  {len(seed)} links, {tp} TP, {fp} FP, P={p:.1%} R={r:.1%} F1={f1:.1%} ({elapsed:.0f}s)")

        # ── Strategy 1: seed_majority_3 ──
        link_counts = Counter()
        for s in runs:
            for key in s:
                link_counts[key] += 1
        seed_majority_3 = {k for k, c in link_counts.items() if c >= 2}

        # ── Strategy 2: seed_union_2 ──
        # Use first 2 runs only
        seed_union_2 = runs[0] | runs[1]

        # ── Also compute other combos for reference ──
        seed_intersect_3 = runs[0] & runs[1] & runs[2]
        seed_intersect_2 = runs[0] & runs[1]

        # ── Results ──
        print(f"\n  STRATEGIES:")
        for label, s in [
            ("baseline (ckpt)", baseline_set),
            ("run1 (fresh)", runs[0]),
            ("run2 (fresh)", runs[1]),
            ("run3 (fresh)", runs[2]),
            ("seed_majority_3", seed_majority_3),
            ("seed_union_2", seed_union_2),
            ("intersect_3", seed_intersect_3),
            ("intersect_2", seed_intersect_2),
        ]:
            tp, fp, fn, p, r, f1 = eval_seed(s, gold)
            print(f"    {label:20s}: {len(s):3d} links, {tp:2d} TP, {fp:2d} FP, P={p:.1%} R={r:.1%} F1={f1:.1%}")

        # ── Variant links detail ──
        all_union = runs[0] | runs[1] | runs[2]
        all_intersect = runs[0] & runs[1] & runs[2]
        variant = all_union - all_intersect
        if variant:
            print(f"\n  VARIANT LINKS ({len(variant)}):")
            for s, c in sorted(variant):
                is_tp = (s, c) in gold
                count = link_counts[(s, c)]
                print(f"    {'TP' if is_tp else 'FP'}: S{s} -> {id_to_name.get(c, c)} ({count}/3 runs)")

    # ── Cross-dataset summary ──
    print(f"\n{'='*100}")
    print(f"SUMMARY: seed_majority_3 uses 3× ILinker2, seed_union_2 uses 2× ILinker2")
    print(f"Both reuse Phase 1/3 from checkpoint (no rerun)")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
