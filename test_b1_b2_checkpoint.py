#!/usr/bin/env python3
"""B1/B2 checkpoint-based analysis for S-Linker11 ICSE review.

B1 — Coref validation: Can routing coref links through _validate_intersect
     improve results? Loads tier2 checkpoints and checks coref TP/FP status.
     If all coref are TPs, validation can only hurt → keep source-adapted.

B2 — Seed/entity redundancy: How much do ILinker2 seed and LLM entity
     extraction overlap? Quantifies unique contributions of each strategy.

Uses S-Linker10 checkpoints (no LLM calls needed).
"""

import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
CACHE_DIR = Path("./results/phase_cache/s_linker10")

DATASETS = {
    "mediastore": "goldstandard_sad_2016-sam_2016.csv",
    "teastore": "goldstandard_sad_2020-sam_2020.csv",
    "teammates": "goldstandard_sad_2021-sam_2021.csv",
    "bigbluebutton": "goldstandard_sad_2021-sam_2021.csv",
    "jabref": "goldstandard_sad_2021-sam_2021.csv",
}


def load_gold(ds):
    gs_path = BENCHMARK_BASE / ds / "goldstandards" / DATASETS[ds]
    gold = set()
    with open(gs_path) as f:
        for row in csv.DictReader(f):
            gold.add((int(row["sentence"]), row["modelElementID"]))
    return gold


def load_checkpoints(ds):
    with open(CACHE_DIR / ds / "tier1.pkl", "rb") as f:
        t1 = pickle.load(f)
    with open(CACHE_DIR / ds / "tier2.pkl", "rb") as f:
        t2 = pickle.load(f)
    return t1, t2


def test_b1_coref_validation():
    """B1: Check if coref links would benefit from _validate_intersect."""
    print("=" * 70)
    print("B1: Coref Validation Analysis")
    print("Q: Would routing coref links through 2-pass validation help?")
    print("=" * 70)

    total_tp = total_fp = 0
    total_unique = 0

    for ds in DATASETS:
        gold = load_gold(ds)
        t1, t2 = load_checkpoints(ds)

        coref_links = t2["coref_links"]
        seed_set = t1["seed_set"]
        entity_set = {(v.sentence_number, v.component_id) for v in t2["validated"]}

        tp = fp = unique = 0
        for l in coref_links:
            t = (l.sentence_number, l.component_id)
            if t in gold:
                tp += 1
            else:
                fp += 1
            if t not in seed_set and t not in entity_set:
                unique += 1

        total_tp += tp
        total_fp += fp
        total_unique += unique

        print(f"\n  {ds}: {len(coref_links)} coref links — {tp} TP, {fp} FP, {unique} unique")
        for l in coref_links:
            t = (l.sentence_number, l.component_id)
            status = "TP" if t in gold else "FP"
            in_other = []
            if t in seed_set:
                in_other.append("seed")
            if t in entity_set:
                in_other.append("entity")
            overlap = f" (also: {'+'.join(in_other)})" if in_other else " UNIQUE"
            print(f"    S{l.sentence_number} -> {l.component_name}: {status}{overlap}")

    print(f"\n  TOTAL: {total_tp} TP, {total_fp} FP across {total_tp + total_fp} coref links")
    print(f"  Unique (not in seed or entity): {total_unique}")

    if total_fp == 0:
        print("\n  VERDICT: All coref links are TPs. Validation can only KILL TPs.")
        print("  -> Keep source-adapted verification (no coref validation).")
    else:
        print(f"\n  VERDICT: {total_fp} FPs found. Validation COULD help.")
        print("  -> Create variant with coref through _validate_intersect and test.")


def test_b2_seed_entity_redundancy():
    """B2: Quantify seed vs entity overlap and unique contributions."""
    print("\n" + "=" * 70)
    print("B2: Seed/Entity Redundancy Analysis")
    print("Q: Are seed and entity strategies redundant?")
    print("=" * 70)

    totals = defaultdict(int)

    for ds in DATASETS:
        gold = load_gold(ds)
        t1, t2 = load_checkpoints(ds)

        seed_set = t1["seed_set"]
        entity_set = {(v.sentence_number, v.component_id) for v in t2["validated"]}
        coref_set = {(l.sentence_number, l.component_id) for l in t2["coref_links"]}

        overlap = seed_set & entity_set
        seed_only = seed_set - entity_set
        entity_only = entity_set - seed_set
        coref_unique = coref_set - seed_set - entity_set

        seed_only_tp = len(seed_only & gold)
        entity_only_tp = len(entity_only & gold)
        overlap_tp = len(overlap & gold)

        totals["seed"] += len(seed_set)
        totals["entity"] += len(entity_set)
        totals["overlap"] += len(overlap)
        totals["seed_only"] += len(seed_only)
        totals["entity_only"] += len(entity_only)
        totals["seed_only_tp"] += seed_only_tp
        totals["entity_only_tp"] += entity_only_tp
        totals["seed_only_fp"] += len(seed_only) - seed_only_tp
        totals["entity_only_fp"] += len(entity_only) - entity_only_tp
        totals["overlap_tp"] += overlap_tp
        totals["coref_unique"] += len(coref_unique)
        totals["coref_unique_tp"] += len(coref_unique & gold)

        print(f"\n  {ds} (gold={len(gold)}):")
        print(f"    Seed: {len(seed_set)} | Entity: {len(entity_set)} | Overlap: {len(overlap)}")
        print(f"    Seed-only: {len(seed_only)} ({seed_only_tp} TP, {len(seed_only)-seed_only_tp} FP)")
        print(f"    Entity-only: {len(entity_only)} ({entity_only_tp} TP, {len(entity_only)-entity_only_tp} FP)")
        print(f"    Coref-unique: {len(coref_unique)} ({len(coref_unique & gold)} TP)")

    print(f"\n  TOTALS:")
    print(f"    Seed: {totals['seed']} | Entity: {totals['entity']} | Overlap: {totals['overlap']}")
    print(f"    Overlap rate: {totals['overlap']/max(1,totals['entity'])*100:.0f}% of entity also in seed")
    print(f"    Seed-only: {totals['seed_only']} ({totals['seed_only_tp']} TP, {totals['seed_only_fp']} FP)")
    print(f"    Entity-only: {totals['entity_only']} ({totals['entity_only_tp']} TP, {totals['entity_only_fp']} FP)")
    print(f"    Coref-unique: {totals['coref_unique']} ({totals['coref_unique_tp']} TP)")

    if totals["entity_only_tp"] > 0 and totals["seed_only_tp"] > 0:
        print(f"\n  VERDICT: Both strategies contribute unique TPs.")
        print(f"    Entity adds {totals['entity_only_tp']} TPs not in seed.")
        print(f"    Seed adds {totals['seed_only_tp']} TPs not in entity.")
        print(f"    -> Neither is redundant. Both needed.")
    elif totals["entity_only_tp"] == 0:
        print(f"\n  VERDICT: Entity adds 0 unique TPs. Potentially redundant with seed.")
    elif totals["seed_only_tp"] == 0:
        print(f"\n  VERDICT: Seed adds 0 unique TPs. Entity subsumes seed.")


if __name__ == "__main__":
    test_b1_coref_validation()
    test_b2_seed_entity_redundancy()
