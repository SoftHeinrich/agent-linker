#!/usr/bin/env python3
"""Spike 004 — aggregate a replay label and A/B it vs cached baselines.

Reads per-cell result JSONs written by replay.py under results/<label>/run*/<ds>.json,
computes macro-F1 + FP-by-source + token cost + the implicit-FN guardrail, and prints a
side-by-side against the two cached sweeps (nothink 89.7, thinking-on 92.8).

Usage:  python .../harness/summarize.py --label layered_offthink [--label2 ...]
"""
import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_io as C
from stage0_reproduce import load_sweep, macro_f1, fp_totals, SWEEPS

RESULTS = os.path.join(REPO_SPIKE := os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")


def load_label(label):
    """Return {(run,dataset): cell_json} for a replay label."""
    cells = {}
    for path in glob.glob(os.path.join(RESULTS, label, "run*", "*.json")):
        d = json.load(open(path))
        cells[(d["run"], d["dataset"])] = d
    return cells


def macro_label(cells):
    by_ds = defaultdict(list)
    for (run, ds), d in cells.items():
        by_ds[ds].append(d["F1"])
    per_ds = {ds: statistics.mean(v) for ds, v in by_ds.items()}
    macro = statistics.mean(per_ds.values()) if per_ds else 0.0
    return macro, per_ds


def implicit_fn(cells):
    """Count FN with name_in_text == False, summed across cells."""
    n = 0
    for d in cells.values():
        n += sum(1 for f in d.get("fn_details", []) if not f.get("name_in_text"))
    return n


def fp_src_label(cells):
    s = defaultdict(int)
    for d in cells.values():
        for k, v in d.get("fp_by_source", {}).items():
            s[k] += v
    return dict(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    args = ap.parse_args()

    # cached baselines (only over the datasets/runs present in the label, for fairness)
    cached = {name: load_sweep(rel) for name, rel in SWEEPS.items()}

    print("=" * 80)
    print("SPIKE 004 — Stage 1/2 summary: replay labels vs cached baselines")
    print("=" * 80)

    for label in args.labels:
        cells = load_label(label)
        if not cells:
            print(f"\n[{label}] no cells found under {os.path.join(RESULTS, label)}")
            continue
        keys = set(cells.keys())
        macro, per_ds = macro_label(cells)
        fps = fp_src_label(cells)
        toks = sum(d["tokens"].get("completion", 0) for d in cells.values())
        calls = sum(d["n_llm_calls"] for d in cells.values())
        ifn = implicit_fn(cells)

        # restrict cached baselines to the SAME (run,dataset) keys for apples-to-apples
        def restrict(sweep):
            return {k: v for k, v in sweep.items() if k in keys}
        nt = restrict(cached["nothink"])
        to = restrict(cached["thinking_on"])

        print(f"\n### label = {label}   ({len(cells)} cells: "
              f"{sorted(set(ds for _, ds in keys))} x {sorted(set(r for r, _ in keys))})")
        print(f"  {'config':16s} {'macroF1':>8s} {'entFP':>6s} {'corFP':>6s} {'implFN':>7s}")
        for nm, cc in [("layered", None), ("nothink(cache)", nt), ("thinking(cache)", to)]:
            if cc is None:
                print(f"  {nm:16s} {macro*100:8.1f} {fps.get('entity',0):6d} "
                      f"{fps.get('coreference',0):6d} {ifn:7d}")
            else:
                mac, _, _, _ = macro_f1(cc)
                src, _ = fp_totals(cc)
                # implicit FN from cached ablation fn_details
                cifn = 0
                # cached cells store fn_details under variant block
                for k, v in cc.items():
                    cifn += sum(1 for f in v.get("fn_details", []) if not f.get("name_in_text"))
                print(f"  {nm:16s} {mac*100:8.1f} {src.get('entity',0):6d} "
                      f"{src.get('coreference',0):6d} {cifn:7d}")
        print(f"  per-dataset (layered): " +
              "  ".join(f"{ds}={per_ds[ds]*100:.1f}" for ds in sorted(per_ds)))
        print(f"  cost: {calls} LLM calls, {toks} completion tokens "
              f"({toks/max(1,len(cells)):.0f} tok/cell)")


if __name__ == "__main__":
    main()
