#!/usr/bin/env python3
"""Spike 005 — Step 1 ($0): decompose the upstream recall gap from candidate pools.

For each (run, dataset) compute the candidate POOL (entity candidates ∪ coref raw, before
validation) for the nothink and thinking-on Sonnet sweeps, and intersect with gold. A
validator can only approve pool members, so |gold ∩ pool| is that pool's recall ceiling.

Decompose every gold link:
  - in nothink pool      -> validator-recoverable (spike 004's lever)
  - in thinkon pool only -> extraction-bound (this spike's lever)
  - in neither           -> unreachable by either backend's candidates

Run from repo root:
  python .planning/spikes/005-upstream-candidate-gap/harness/candidate_gap.py
"""
import os
import statistics
import sys
from collections import defaultdict

H004 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "004-nogap-validator-ab", "harness")
sys.path.insert(0, os.path.abspath(H004))
import cache_io as C


def pool(cell):
    """Candidate pool = entity candidates ∪ coref raw, as (sentence, component_id) set."""
    ent = {(c.sentence_number, c.component_id) for c in cell["layer3"]["candidates"]}
    cor = {(lk.sentence_number, lk.component_id) for lk in cell["layer4"]["coref_raw"]}
    return ent | cor


def main():
    rows = []
    bucket = defaultdict(int)  # recoverable / extraction_bound / unreachable (over all cells)
    per_ds = defaultdict(lambda: defaultdict(list))
    extraction_bound_links = defaultdict(set)  # ds -> {(s, comp_name)} thinkon-only true links

    for run in C.RUNS:
        for ds in C.DATASETS:
            nt = C.load_cell(C.NOTHINK_ROOT, run, ds)
            to = C.load_cell(C.THINKING_ROOT, run, ds)
            bench = C.load_benchmark(ds)
            gold, id2n = bench["gold"], bench["id_to_name"]
            ntp, top = pool(nt), pool(to)

            nt_ceil = len(gold & ntp) / len(gold)
            to_ceil = len(gold & top) / len(gold)
            per_ds[ds]["nt_ceil"].append(nt_ceil)
            per_ds[ds]["to_ceil"].append(to_ceil)

            for g in gold:
                if g in ntp:
                    bucket["recoverable"] += 1
                elif g in top:
                    bucket["extraction_bound"] += 1
                    extraction_bound_links[ds].add((g[0], id2n.get(g[1], g[1])))
                else:
                    bucket["unreachable"] += 1
            rows.append((run, ds, nt_ceil, to_ceil))

    print("=" * 74)
    print("SPIKE 005 — candidate-pool recall ceilings (nothink vs thinking-on)")
    print("=" * 74)
    print(f"\n  {'dataset':14s} {'nt_ceil':>8s} {'to_ceil':>8s} {'gap(pts)':>9s}")
    nt_all, to_all = [], []
    for ds in C.DATASETS:
        ntc = statistics.mean(per_ds[ds]["nt_ceil"]) * 100
        toc = statistics.mean(per_ds[ds]["to_ceil"]) * 100
        nt_all.append(ntc); to_all.append(toc)
        print(f"  {ds:14s} {ntc:8.1f} {toc:8.1f} {toc-ntc:9.1f}")
    print(f"  {'MACRO':14s} {statistics.mean(nt_all):8.1f} {statistics.mean(to_all):8.1f} "
          f"{statistics.mean(to_all)-statistics.mean(nt_all):9.1f}")

    tot = sum(bucket.values())
    print(f"\n  Gold-link decomposition (summed over all 15 cells, total gold instances={tot}):")
    print(f"    validator-recoverable (in nothink pool) : {bucket['recoverable']:4d}"
          f"  ({100*bucket['recoverable']/tot:.1f}%)")
    print(f"    extraction-bound (thinkon pool only)     : {bucket['extraction_bound']:4d}"
          f"  ({100*bucket['extraction_bound']/tot:.1f}%)")
    print(f"    unreachable (neither pool)               : {bucket['unreachable']:4d}"
          f"  ({100*bucket['unreachable']/tot:.1f}%)")

    print("\n  Extraction-bound true links (thinking-on extracts, nothink never proposes),")
    print("  distinct (sentence, component) per dataset:")
    for ds in C.DATASETS:
        n = len(extraction_bound_links[ds])
        sample = sorted(extraction_bound_links[ds])[:5]
        print(f"    {ds:14s} {n:3d}  e.g. {sample}")


if __name__ == "__main__":
    main()
