"""Offline feasibility pilot for the s23_verify simplifications (S1-S4).

The tier-signal caches (pilot/cache/tier_signals_<ds>.json) hold, per candidate in
the Framing-C UNION blocks-proposer population: source x match x p1 x p2 x gold.
Crucially they were captured ROUTER-LESS (tiered_ranking.capture runs the evidence
gate directly), so every keep-rule below is the *gate* decision with NO router in
front. That makes this the exact test bed for the GATE simplifications:

  BASE  s23_verify gate         : keep  <=>  p1 AND p2   (two-pass evidence gate)
  S1    single evidence pass    : keep  <=>  p1
  S2    EXACT-shortcut + gate    : EXACT-standalone auto-kept, others gated (p1&p2)
  S4    source x match x votes    : tiered_ranking.assign_tier != REJECT (F1 cut)

S3 (drop the router agent) is NOT projectable here — the cache already has no
router — so S3 is measured e2e only; the numbers here are its router-less ceiling.

    python pilot/simplify_verify_pilot.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

from tiered_ranking import assign_tier  # S4 tier rule (match x votes x source)


def load():
    per_ds, pool = {}, []
    for f in sorted(glob.glob("pilot/cache/tier_signals_*.json")):
        ds = os.path.basename(f)[len("tier_signals_"):-5]
        d = json.load(open(f))
        for r in d["rows"]:
            r["ds"] = ds
        per_ds[ds] = (d["rows"], d["n_gold"])
        pool += d["rows"]
    return per_ds, pool


def prf(kept, gold_n):
    tp = sum(1 for r in kept if r["gold"]); fp = len(kept) - tp; fn = gold_n - tp
    P = tp / (tp + fp) if tp + fp else 1.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    F2 = 5 * P * R / (4 * P + R) if 4 * P + R else 0.0
    return tp, fp, fn, P, R, F1, F2


# ── keep-rules under test (each maps a row -> kept? bool) ──────────────────────
def keep_base(r): return bool(r["p1"] and r["p2"])
def keep_s1(r):   return bool(r["p1"])
def keep_s2(r):   return True if r["match"] == "EXACT" else bool(r["p1"] and r["p2"])
def keep_s4(r):   return assign_tier(r) != "REJECT"

# composites the pilot should also surface
def keep_s1s2(r): return True if r["match"] == "EXACT" else bool(r["p1"])

RULES = {
    "BASE p1&p2":        keep_base,
    "S1  p1 only":       keep_s1,
    "S2  EXACT+ gate":    keep_s2,
    "S1+S2 EXACT+ p1":    keep_s1s2,
    "S4  tier(!REJECT)": keep_s4,
}


def report(title, rows, gold_n):
    print(f"\n=== {title} (gold={gold_n}, cands={len(rows)}) ===")
    print(f"{'rule':<20}{'keep':>5}{'TP':>5}{'FP':>5}{'P':>7}{'R':>7}{'F1':>7}{'F2':>7}  dF1")
    base_f1 = None
    for name, rule in RULES.items():
        kept = [r for r in rows if rule(r)]
        tp, fp, fn, P, R, F1, F2 = prf(kept, gold_n)
        if base_f1 is None:
            base_f1 = F1
        d = "" if name.startswith("BASE") else f"{F1 - base_f1:+.3f}"
        print(f"{name:<20}{len(kept):>5}{tp:>5}{fp:>5}{P:>7.3f}{R:>7.3f}{F1:>7.3f}{F2:>7.3f}  {d}")


def feasibility_s2(pool):
    print("\n=== S2 FEASIBILITY: is EXACT safe to auto-accept? (purity by votes) ===")
    cells = defaultdict(lambda: [0, 0])
    for r in pool:
        if r["match"] == "EXACT":
            v = r["p1"] + r["p2"]
            cells[v][0] += 1; cells[v][1] += r["gold"]
    print(f"{'EXACT@votes':<12}{'n':>5}{'gold':>6}{'purity':>8}")
    for v in sorted(cells):
        n, g = cells[v]
        print(f"{'v'+str(v):<12}{n:>5}{g:>6}{g/n:>8.2f}")
    # what auto-accepting ALL exact would cost vs gating them
    exact = [r for r in pool if r["match"] == "EXACT"]
    leaked = [r for r in exact if not keep_base(r) and not r["gold"]]  # gate rejected, nongold, would leak
    lost_recovered = [r for r in exact if not keep_base(r) and r["gold"]]  # gate rejected, gold, would recover
    print(f"auto-accept ALL EXACT vs gate: n_exact={len(exact)}  "
          f"would LEAK (gate-rejected nongold)={len(leaked)}  "
          f"would RECOVER (gate-rejected gold)={len(lost_recovered)}")


def main():
    per_ds, pool = load()
    total_gold = sum(g for _, g in per_ds.values())
    feasibility_s2(pool)
    report("POOLED (5 datasets)", pool, total_gold)
    for ds, (rows, gn) in per_ds.items():
        report(ds, rows, gn)


if __name__ == "__main__":
    main()
