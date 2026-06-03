#!/usr/bin/env python3
"""Test whether Framing C extraction consensus (L3) is necessary given Phase 4 validation.

For each dataset with a completed s_linker17f checkpoint:
  - Loads layer2.pkl  → framing_c_pass1, framing_c_pass2
  - Loads layer3.pkl  → Phase 4 decisions (to check if Phase 4 would have saved L3 TPs)
  - Loads gold standard
  - Reports: what L3 rejects, how many are TPs vs FPs, and whether Phase 4 was
    already rejecting the FPs (making L3 redundant) or providing independent signal.

Usage:
    cd /path/to/agent-linker
    python scripts/test_l3_contribution.py [dataset ...]

    # all available datasets:
    python scripts/test_l3_contribution.py

    # specific dataset:
    python scripts/test_l3_contribution.py mediastore jabref
"""

import csv
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

CACHE_DIR = ROOT / "results" / "phase_cache" / "s_linker17f"
BENCHMARK_BASE = ROOT / "../ardoco/core/tests-base/src/main/resources/benchmark"

DATASETS = {
    "mediastore":    BENCHMARK_BASE / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    "teastore":      BENCHMARK_BASE / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    "teammates":     BENCHMARK_BASE / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "bigbluebutton": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "jabref":        BENCHMARK_BASE / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
}


def load_gold(path: Path) -> set[tuple[int, str]]:
    gold = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            comp_id = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if comp_id and snum:
                gold.add((int(snum), comp_id))
    return gold


def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def analyse(dataset: str, gold_path: Path) -> None:
    cache = CACHE_DIR / dataset
    layer2_path = cache / "layer2.pkl"
    layer3_path = cache / "layer3.pkl"

    if not layer2_path.exists():
        print(f"  [{dataset}] SKIP — no layer2.pkl (run s_linker17f first)")
        return

    layer2 = load_pkl(layer2_path)
    pass1: dict = layer2.get("framing_c_pass1") or {}
    pass2: dict = layer2.get("framing_c_pass2") or {}

    if not pass1 and not pass2:
        print(f"  [{dataset}] SKIP — framing_c_pass1/pass2 not in layer2 "
              "(run with updated s_linker17f that saves passes)")
        return

    intersection = set(pass1) & set(pass2)
    union = set(pass1) | set(pass2)
    l3_rejects = union - intersection  # candidates L3 dropped (single-pass-only)

    if not gold_path.exists():
        print(f"  [{dataset}] SKIP — gold not found at {gold_path}")
        return
    gold = load_gold(gold_path)

    # Classify L3 rejects against gold
    l3_tp_killed = {k for k in l3_rejects if k in gold}
    l3_fp_killed = {k for k in l3_rejects if k not in gold}

    # Check Phase 4 decisions for candidates that DID pass L3
    # to estimate: "would Phase 4 have rejected the FPs anyway?"
    phase4_rejects: set = set()
    phase4_approves: set = set()
    if layer3_path.exists():
        layer3 = load_pkl(layer3_path)
        decisions: dict = layer3.get("decisions", {})
        for key, dec in decisions.items():
            if not dec.get("approved", True):
                phase4_rejects.add(key)
            else:
                phase4_approves.add(key)
        # Phase 4 FP rejection rate on intersection candidates (proxy for what it'd do to L3 rejects)
        intersection_fps = {k for k in intersection if k not in gold}
        intersection_fps_rejected_by_p4 = intersection_fps & phase4_rejects
        p4_fp_rate = (len(intersection_fps_rejected_by_p4) / len(intersection_fps)
                      if intersection_fps else None)
    else:
        p4_fp_rate = None

    # Summary
    print(f"\n[{dataset}]")
    print(f"  Framing C: pass1={len(pass1)}, pass2={len(pass2)}, "
          f"intersection={len(intersection)}, union={len(union)}")
    print(f"  L3 rejects (single-pass-only): {len(l3_rejects)}")
    if l3_rejects:
        print(f"    TPs killed by L3: {len(l3_tp_killed)}  "
              f"({100*len(l3_tp_killed)/len(l3_rejects):.0f}%)")
        print(f"    FPs killed by L3: {len(l3_fp_killed)}  "
              f"({100*len(l3_fp_killed)/len(l3_rejects):.0f}%)")
        if l3_tp_killed:
            print(f"    Recall cost: {len(l3_tp_killed)} true links lost")
        if p4_fp_rate is not None:
            print(f"  Phase 4 FP rejection rate on intersection: {100*p4_fp_rate:.0f}% "
                  f"(proxy: would Phase 4 have caught L3's FPs?)")
            if p4_fp_rate > 0.7:
                print("    → Phase 4 likely redundant with L3 for FP suppression")
            else:
                print("    → L3 provides independent FP suppression beyond Phase 4")
    else:
        print("    Both passes agreed on everything — L3 had no effect on this dataset.")


def main():
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS)
    unknown = [d for d in requested if d not in DATASETS]
    if unknown:
        print(f"Unknown datasets: {unknown}. Available: {list(DATASETS)}")
        sys.exit(1)

    print("L3 (Framing C consensus) contribution analysis — s_linker17f")
    print("=" * 60)
    print("Verdict guide:")
    print("  L3 necessary   → many FPs killed, few TPs lost, Phase 4 misses them")
    print("  L3 redundant   → FPs that L3 kills are also killed by Phase 4")
    print("  L3 hurts recall → meaningful TPs lost, few FPs gained")
    print("=" * 60)

    for dataset in requested:
        analyse(dataset, DATASETS[dataset])

    print()


if __name__ == "__main__":
    main()
