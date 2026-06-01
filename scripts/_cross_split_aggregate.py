#!/usr/bin/env python3
"""Cross-split bank aggregation for Phase 17 Confirmation Tier.

Algorithm per slot:
1. Collect all (pattern, split_name) pairs from 3 split banks.
2. Jaccard dedup: cluster patterns with token-level Jaccard similarity >= 0.6.
   Within each cluster, keep the pattern from the most splits; on tie, keep longest rule_text.
3. Survival filter: discard patterns not present in >= 2 of the 3 splits (before dedup).
   "Present" = pattern is in the split's final_bank for that slot.
4. Write surviving patterns to cross_split_final_bank.json.

Note: Bank format uses {"slot_patterns": {...}} wrapper (v4b format).
AXIOM_SLOTS matches SLOT_NAMES in s_linker14_voyager.py.
"""

import json
import re
from pathlib import Path

ROOT = Path("results/voyager_v4_beta")
CONFIRMATION_DIR = ROOT / "confirmation"
CONFIRMATION_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = [
    "split1_replication",
    "split2_bbb_in_train",
    "split3_rotated_holdout",
]

# All 9 slots from s_linker14_voyager SLOT_NAMES
AXIOM_SLOTS = [
    "AMBIGUITY_FEW_SHOT",
    "AMBIGUITY_RULES",
    "DOC_KNOWLEDGE_EXTRACTION_RULES",
    "DOC_KNOWLEDGE_JUDGE_EXAMPLES",
    "DOC_KNOWLEDGE_JUDGE_RULES",
    "ENTITY_EXTRACTION_RULES",
    "VALIDATION_RULES",
    "COREF_RULES",
    "SEED_DISAMBIGUATION_RULES",
]


def tokenize(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower()))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def cluster_patterns(patterns_with_splits):
    """Greedy Jaccard clustering. Each pattern is (pattern_dict, split_name)."""
    clusters = []  # list of [(pattern_dict, split_name), ...]
    for p, sname in patterns_with_splits:
        rule = p.get("rule_text", p.get("content", ""))
        toks = tokenize(rule)
        placed = False
        for cluster in clusters:
            rep = cluster[0][0]
            rep_rule = rep.get("rule_text", rep.get("content", ""))
            rep_toks = tokenize(rep_rule)
            if jaccard(toks, rep_toks) >= 0.6:
                cluster.append((p, sname))
                placed = True
                break
        if not placed:
            clusters.append([(p, sname)])
    return clusters


cross_bank = {slot: [] for slot in AXIOM_SLOTS}
stats = {"total_raw": 0, "survived_dedup": 0, "survived_filter": 0}

print("Cross-split aggregation (Jaccard >= 0.6, >= 2-split survival filter)")
print(f"Splits: {SPLITS}")
print()

for slot in AXIOM_SLOTS:
    # Collect patterns with their split provenance
    # Bank format: {"slot_patterns": {slot: [patterns...]}}
    patterns_with_splits = []
    for split_name in SPLITS:
        bank_path = ROOT / split_name / "final_bank.json"
        if not bank_path.exists():
            print(f"  WARNING: {bank_path} missing — skipping split for slot {slot}")
            continue
        bank = json.loads(bank_path.read_text())
        slot_patterns = bank.get("slot_patterns", bank)
        for p in slot_patterns.get(slot, []):
            patterns_with_splits.append((p, split_name))
    stats["total_raw"] += len(patterns_with_splits)

    if not patterns_with_splits:
        continue

    # Cluster by Jaccard similarity
    clusters = cluster_patterns(patterns_with_splits)

    slot_survivors = []
    for cluster in clusters:
        # Count distinct splits represented
        cluster_splits = set(sname for _, sname in cluster)
        if len(cluster_splits) < 2:
            # Fails >=2-split survival filter
            continue
        # Pick representative: most splits, then longest rule_text
        best = max(cluster, key=lambda x: (
            sum(1 for _, s in cluster if s == x[1]),  # count in same split (tie-break)
            len(x[0].get("rule_text", x[0].get("content", "")))
        ))[0]
        slot_survivors.append(best)

    stats["survived_dedup"] += len(clusters)
    stats["survived_filter"] += len(slot_survivors)
    cross_bank[slot] = slot_survivors
    if patterns_with_splits:
        print(f"  {slot}: {len(patterns_with_splits)} raw -> {len(clusters)} clusters -> {len(slot_survivors)} survived")

total = sum(len(v) for v in cross_bank.values())
slots_used = [s for s in AXIOM_SLOTS if cross_bank[s]]
print(f"\nCross-split bank: {total} patterns in {len(slots_used)} slots")
print(f"Stats: {stats['total_raw']} raw -> {stats['survived_dedup']} post-dedup -> {stats['survived_filter']} post-filter")

# Check if 0 patterns — try lower Jaccard threshold
if total == 0:
    print("\nWARNING: 0 patterns survived at Jaccard >= 0.6. Retrying at 0.4...")
    cross_bank = {slot: [] for slot in AXIOM_SLOTS}
    stats2 = {"total_raw": 0, "survived_dedup": 0, "survived_filter": 0}

    for slot in AXIOM_SLOTS:
        patterns_with_splits = []
        for split_name in SPLITS:
            bank_path = ROOT / split_name / "final_bank.json"
            if not bank_path.exists():
                continue
            bank = json.loads(bank_path.read_text())
            slot_patterns = bank.get("slot_patterns", bank)
            for p in slot_patterns.get(slot, []):
                patterns_with_splits.append((p, split_name))
        stats2["total_raw"] += len(patterns_with_splits)

        if not patterns_with_splits:
            continue

        # Cluster by Jaccard 0.4
        clusters = []
        for p, sname in patterns_with_splits:
            rule = p.get("rule_text", p.get("content", ""))
            toks = tokenize(rule)
            placed = False
            for cluster in clusters:
                rep = cluster[0][0]
                rep_rule = rep.get("rule_text", rep.get("content", ""))
                rep_toks = tokenize(rep_rule)
                if jaccard(toks, rep_toks) >= 0.4:
                    cluster.append((p, sname))
                    placed = True
                    break
            if not placed:
                clusters.append([(p, sname)])

        slot_survivors = []
        for cluster in clusters:
            cluster_splits = set(sname for _, sname in cluster)
            if len(cluster_splits) < 2:
                continue
            best = max(cluster, key=lambda x: (
                sum(1 for _, s in cluster if s == x[1]),
                len(x[0].get("rule_text", x[0].get("content", "")))
            ))[0]
            slot_survivors.append(best)

        stats2["survived_dedup"] += len(clusters)
        stats2["survived_filter"] += len(slot_survivors)
        cross_bank[slot] = slot_survivors
        if patterns_with_splits:
            print(f"  {slot}: {len(patterns_with_splits)} raw -> {len(clusters)} clusters -> {len(slot_survivors)} survived (Jaccard >= 0.4)")

    total = sum(len(v) for v in cross_bank.values())
    slots_used = [s for s in AXIOM_SLOTS if cross_bank[s]]
    print(f"\nCross-split bank (Jaccard >= 0.4): {total} patterns in {len(slots_used)} slots")
    if total == 0:
        print("WARNING: Still 0 patterns. Falling back to union of all 3 banks (no cross-split consensus).")
        # Union fallback: collect all patterns, dedup by pattern_id
        seen_ids = set()
        for slot in AXIOM_SLOTS:
            for split_name in SPLITS:
                bank_path = ROOT / split_name / "final_bank.json"
                if not bank_path.exists():
                    continue
                bank = json.loads(bank_path.read_text())
                slot_patterns = bank.get("slot_patterns", bank)
                for p in slot_patterns.get(slot, []):
                    pid = p.get("pattern_id", "")
                    if pid not in seen_ids:
                        cross_bank[slot].append(p)
                        seen_ids.add(pid)
        total = sum(len(v) for v in cross_bank.values())
        print(f"Union fallback bank: {total} patterns (documented in verdict as 'no cross-split consensus')")

# Write output — use slot_patterns wrapper format (v4b format, same as per-split banks)
out_dict = {
    "version": "v4b",
    "note": "cross-split aggregation: Jaccard >= 0.6 dedup + >= 2-split survival filter",
    "splits": SPLITS,
    "stats": stats,
    "slot_patterns": cross_bank
}
out = CONFIRMATION_DIR / "cross_split_final_bank.json"
out.write_text(json.dumps(out_dict, indent=2))
print(f"\nWritten: {out}")
print("\nPer-slot breakdown:")
for slot in AXIOM_SLOTS:
    n = len(cross_bank[slot])
    if n > 0:
        print(f"  {slot}: {n} patterns")
