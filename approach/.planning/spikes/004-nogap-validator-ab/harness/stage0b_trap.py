#!/usr/bin/env python3
"""Spike 004 — Stage 0b: rule-based Mode 2 trap rejecter on cached nothink links ($0).

Removal-only post-filter on the frozen nothink final links (no LLM). For each trap
(and the combined set) measure: recovered macro-F1, FPs removed (good), and the
guardrail — TPs removed, split by implicit (name_in_text=False).

Run from repo root:
  python .planning/spikes/004-nogap-validator-ab/harness/stage0b_trap.py
"""
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_io as C
from traps import TRAPS, trap_hits


def macro(per_cell_f1):
    by_ds = defaultdict(list)
    for (run, ds), f1 in per_cell_f1.items():
        by_ds[ds].append(f1)
    return statistics.mean(statistics.mean(v) for v in by_ds.values()), \
        {ds: statistics.mean(v) for ds, v in by_ds.items()}


def run_config(cells, reject_fn):
    """reject_fn(ctx) -> bool. Returns per_cell_f1 + removal accounting."""
    per_cell_f1 = {}
    rem_fp = rem_tp = rem_tp_implicit = 0
    rem_by_ds = defaultdict(lambda: [0, 0])  # ds -> [fp_removed, tp_removed]
    for (run, ds), ctxs in cells.items():
        gold = C.load_benchmark(ds)["gold"]
        kept = []
        for c in ctxs:
            if reject_fn(c):
                if c.in_gold:
                    rem_tp += 1
                    rem_by_ds[ds][1] += 1
                    if not c.name_in_text:
                        rem_tp_implicit += 1
                else:
                    rem_fp += 1
                    rem_by_ds[ds][0] += 1
            else:
                kept.append((c.sentence_number, c.component_id))
        per_cell_f1[(run, ds)] = C.score_pairs(kept, gold)["F1"]
    return per_cell_f1, rem_fp, rem_tp, rem_tp_implicit, rem_by_ds


def main():
    # Load all nothink cells once.
    cells = {}
    for run in C.RUNS:
        for ds in C.DATASETS:
            cell = C.load_cell(C.NOTHINK_ROOT, run, ds)
            cells[(run, ds)] = C.build_contexts(cell, ds)

    print("=" * 78)
    print("SPIKE 004 — STAGE 0b: rule-based trap rejecter on cached nothink links ($0)")
    print("=" * 78)

    base_f1, *_ = run_config(cells, lambda c: False)
    bmac, bds = macro(base_f1)
    print(f"\nbaseline nothink macro-F1 = {bmac*100:.1f}  (target thinking-on = 92.8)")
    print(f"  per-dataset: " + "  ".join(f"{ds}={bds[ds]*100:.1f}" for ds in C.DATASETS))

    # Ceiling: remove every FP (perfect precision at current recall).
    ceil_f1, *_ = run_config(cells, lambda c: not c.in_gold)
    cmac, _ = macro(ceil_f1)
    print(f"precision-perfect ceiling macro-F1 = {cmac*100:.1f}  (remove ALL FPs)")

    print("\n%-18s %7s %8s %8s %8s %10s" %
          ("config", "macroF1", "dF1", "FP_rem", "TP_rem", "TP_impl"))
    print("-" * 78)

    def show(name, reject_fn):
        f1, fp, tp, tpi, rem_by_ds = run_config(cells, reject_fn)
        m, mds = macro(f1)
        print("%-18s %7.1f %+8.1f %8d %8d %10d" %
              (name, m*100, (m-bmac)*100, fp, tp, tpi))
        return m, mds, rem_by_ds

    per_trap_ds = {}
    for name, fn in TRAPS.items():
        m, mds, rem = show(name, fn)
        per_trap_ds[name] = (mds, rem)

    # Combined: reject if ANY trap fires.
    print("-" * 78)
    cm, cmds, crem = show("ALL_TRAPS", lambda c: bool(trap_hits(c)))

    # teammates focus (the dominant regression)
    print("\nteammates detail (FP removed / TP removed by config):")
    for name in list(TRAPS) + ["ALL_TRAPS"]:
        rem = per_trap_ds[name][1] if name in per_trap_ds else crem
        fp_rm, tp_rm = rem["teammates"]
        print(f"  {name:18s} FP_removed={fp_rm:2d}  TP_removed={tp_rm}")

    print("\nGuardrail read: any TP_impl > 0 means a trap killed an implicit true link.")


if __name__ == "__main__":
    main()
