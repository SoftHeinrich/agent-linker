#!/usr/bin/env python3
"""Check: when syn-safe is removed on BBB, which of the 22 judge-killed links are TPs?"""

import csv
import glob
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
V30C_CACHE = Path("./results/phase_cache/v30c")
V30D_CACHE = Path("./results/phase_cache/v30d")


def load_gold(dataset):
    gold_path = BENCHMARK_BASE / dataset
    pattern = str(gold_path / "**" / "goldstandard_sad_*-sam_*.csv")
    files = [f for f in glob.glob(pattern, recursive=True) if "UME" not in f and "code" not in f]
    gold = set()
    for f in files:
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                cid = row.get("modelElementID", "")
                sid = row.get("sentence", "")
                if sid and cid:
                    gold.add((int(sid), cid))
    return gold


for ds in ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]:
    # V30c baseline (with syn-safe)
    baseline_path = V30C_CACHE / ds / "final.pkl"
    if not baseline_path.exists():
        continue
    with open(baseline_path, "rb") as f:
        baseline_data = f.read()
    baseline = pickle.loads(baseline_data)
    baseline_final = baseline["final"]
    baseline_set = {(l.sentence_number, l.component_id) for l in baseline_final}

    # V30d no-synsafe (saved by single-phase 9 run)
    nosyn_path = V30D_CACHE / ds / "final.pkl"
    if not nosyn_path.exists():
        print(f"{ds}: no v30d final checkpoint")
        continue
    with open(nosyn_path, "rb") as f:
        nosyn_data = f.read()
    nosyn = pickle.loads(nosyn_data)
    nosyn_final = nosyn["final"]
    nosyn_set = {(l.sentence_number, l.component_id) for l in nosyn_final}

    gold = load_gold(ds)

    # Links in baseline but not in no-synsafe = killed by judge when syn-safe removed
    killed = baseline_set - nosyn_set
    # Links in no-synsafe but not baseline = somehow new (shouldn't happen with same pre_judge)
    added = nosyn_set - baseline_set

    if not killed and not added:
        print(f"\n{ds}: NO DIFFERENCE between syn-safe and no-syn-safe")
        continue

    print(f"\n{'=' * 70}")
    print(f"  {ds}: {len(killed)} links KILLED by removing syn-safe")
    print(f"{'=' * 70}")

    # Map back to link objects for details
    baseline_map = {(l.sentence_number, l.component_id): l for l in baseline_final}
    tp_killed = 0
    fp_killed = 0
    for key in sorted(killed):
        lk = baseline_map.get(key)
        is_tp = key in gold
        if is_tp:
            tp_killed += 1
        else:
            fp_killed += 1
        label = "TP LOST!" if is_tp else "FP removed"
        name = lk.component_name if lk else "?"
        src = lk.source if lk else "?"
        print(f"  S{key[0]}->{name} [{label}] src={src}")

    if added:
        print(f"\n  {len(added)} links ADDED (unexpected):")
        nosyn_map = {(l.sentence_number, l.component_id): l for l in nosyn_final}
        for key in sorted(added):
            lk = nosyn_map.get(key)
            is_tp = key in gold
            print(f"  S{key[0]}->{lk.component_name if lk else '?'} [{'TP' if is_tp else 'FP'}] src={lk.source if lk else '?'}")

    print(f"\n  IMPACT: {tp_killed} TPs lost, {fp_killed} FPs removed")
    print(f"  Net: {'WORSE' if tp_killed > fp_killed else 'BETTER' if fp_killed > tp_killed else 'NEUTRAL'} "
          f"(precision {'up' if fp_killed > 0 else 'same'}, recall {'down' if tp_killed > 0 else 'same'})")
