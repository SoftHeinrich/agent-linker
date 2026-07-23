#!/usr/bin/env python3
"""Spike 004 — Stage 0: reproduce the note's A/B baselines from cached ablation JSONs.

Aggregates the per-cell ablation_*.json emitted by run_ablation.py for the two
sweeps already on disk:
  - thinking-on : results/v2.6.5_s20union_sonnet                  (note: macro 92.8)
  - nothink     : results/v2.6.5_s20union_sonnet_nothink_20260627 (note: macro 89.4)

Goal: confirm our scoring matches the note (macro-F1 and FP-by-source split) BEFORE
spending anything on LLM modes. Pure read of cached JSON — no LLM, no network.

Run from repo root:  python .planning/spikes/004-nogap-validator-ab/harness/stage0_reproduce.py
"""
import glob
import json
import os
import statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", "..", ".."))

SWEEPS = {
    "thinking_on": "results/v2.6.5_s20union_sonnet",
    "nothink": "results/v2.6.5_s20union_sonnet_nothink_20260627",
}
DATASETS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]


def load_sweep(rel_dir):
    """Return {(run, dataset): variant_metrics_dict} for every ablation JSON found."""
    cells = {}
    base = os.path.join(REPO, rel_dir)
    for path in sorted(glob.glob(os.path.join(base, "run*", "*", "ablation_*.json"))):
        dataset = os.path.basename(os.path.dirname(path))
        run = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if dataset not in DATASETS:
            continue
        d = json.load(open(path))
        # schema: {dataset: {variant: {...}}}
        ds_block = d.get(dataset) or next(iter(d.values()))
        variant_block = next(iter(ds_block.values()))
        cells[(run, dataset)] = variant_block
    return cells


def macro_f1(cells):
    """Macro = mean over datasets of (mean over runs of F1). Also report pooled."""
    by_ds = defaultdict(list)
    for (run, ds), m in cells.items():
        by_ds[ds].append(m["F1"])
    per_ds_mean = {ds: statistics.mean(v) for ds, v in by_ds.items()}
    macro = statistics.mean(per_ds_mean.values())
    pooled = statistics.mean([m["F1"] for m in cells.values()])
    return macro, pooled, per_ds_mean, by_ds


def fp_totals(cells):
    """Sum FP-by-source across all cells; also per-dataset FP totals."""
    src = defaultdict(int)
    per_ds = defaultdict(int)
    for (run, ds), m in cells.items():
        for s, n in (m.get("fp_by_source") or {}).items():
            src[s] += n
        per_ds[ds] += m.get("fp", 0)
    return dict(src), dict(per_ds)


def main():
    results = {name: load_sweep(rel) for name, rel in SWEEPS.items()}

    print("=" * 72)
    print("SPIKE 004 — STAGE 0: reproduce A/B baselines from cached ablation JSONs")
    print("=" * 72)

    for name in ("thinking_on", "nothink"):
        cells = results[name]
        macro, pooled, per_ds_mean, by_ds = macro_f1(cells)
        src, per_ds_fp = fp_totals(cells)
        print(f"\n### {name}  ({len(cells)} cells)")
        print(f"  macro-F1 (mean of per-dataset means) = {macro*100:.1f}")
        print(f"  pooled-F1 (mean over all cells)       = {pooled*100:.1f}")
        print("  per-dataset mean F1:")
        for ds in DATASETS:
            if ds in per_ds_mean:
                print(f"    {ds:14s} F1={per_ds_mean[ds]*100:5.1f}   FP(sum over runs)={per_ds_fp.get(ds,0)}")
        print(f"  FP-by-source (summed over all cells): {src}")

    # Delta view
    t = results["thinking_on"]
    n = results["nothink"]
    tmac, _, tds, _ = macro_f1(t)
    nmac, _, nds, _ = macro_f1(n)
    tsrc, tfp = fp_totals(t)
    nsrc, nfp = fp_totals(n)
    print("\n" + "=" * 72)
    print("DELTA  thinking_on -> nothink")
    print("=" * 72)
    print(f"  macro-F1: {tmac*100:.1f} -> {nmac*100:.1f}   (drop {(tmac-nmac)*100:.1f})")
    print("  per-dataset F1 drop:")
    for ds in DATASETS:
        print(f"    {ds:14s} {tds.get(ds,0)*100:5.1f} -> {nds.get(ds,0)*100:5.1f}   "
              f"(d {(nds.get(ds,0)-tds.get(ds,0))*100:+.1f})")
    print(f"  FP-by-source: entity {tsrc.get('entity',0)} -> {nsrc.get('entity',0)}   "
          f"coref {tsrc.get('coreference',0)} -> {nsrc.get('coreference',0)}")
    print("  teammates FP (sum over runs): "
          f"{tfp.get('teammates',0)} -> {nfp.get('teammates',0)}")
    print("\nNote claims: macro 92.8 -> 89.4; coref FP 7->27; entity 25->35; teammates FP 16->41")


if __name__ == "__main__":
    main()
