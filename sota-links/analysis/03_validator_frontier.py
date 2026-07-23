#!/usr/bin/env python3
"""03 — A more general validator.  The entity gate is a two-pass AND
(approve iff p1 AND p2).  Using the cached per-pass verdicts, re-combine them
into a family of validator policies and score each against gold WITHOUT any new
LLM calls.

  * Pass decomposition  — marginal FP-removed vs gold-TP-cost of each pass.
  * Combination frontier — AND (current) / P2-only / P1-only / OR (consensus).
  * Self-consistency     — approve if a policy holds in >=k of the 3 runs.

Runs for both backends (gpt under .../openai, sonnet under .../claude).

Usage:  python3 analysis/03_validator_frontier.py
"""
import os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

pickle = C.load_phase_cache()
BACKENDS = [("gpt", "openai"), ("sonnet", "claude")]
GOLD = {p: C.gold(p) for p in C.PROJECTS}
NGOLD = sum(len(v) for v in GOLD.values())


def load_decisions(backend):
    """List over runs; each run is {(proj,sid,cid): (p1,p2)}."""
    cache = os.path.join(C.RL, "phase-cache-s21", f"v2.6.6_s21_{backend[0]}")
    runs = []
    for run in (1, 2, 3):
        dec = {}
        for p in C.PROJECTS:
            l3 = pickle.load(open(os.path.join(
                cache, f"run{run}", "phase_cache", "s_linker21", backend[1], p, "layer3.pkl"), "rb"))
            for (sid, cid), d in l3["decisions"].items():
                dec[(p, sid, cid)] = (d["p1"], d["p2"])
        runs.append(dec)
    return runs


def is_gold(key):
    p, sid, cid = key
    return (sid, cid) in GOLD[p]


def metrics(approved):
    tp = sum(1 for k in approved if is_gold(k))
    fp = len(approved) - tp
    R = tp / NGOLD
    P = tp / max(1, len(approved))
    F = 2 * P * R / max(1e-9, P + R)
    return tp, fp, 100 * P, 100 * R, 100 * F


POLICIES = {
    "AND  (current: p1 & p2)":          lambda p1, p2: p1 and p2,
    "P2-only (drop P1)":                lambda p1, p2: p2,
    "P1-only (drop P2)":                lambda p1, p2: p1,
    "OR   (reject iff BOTH reject)":    lambda p1, p2: p1 or p2,
}


def main():
    for backend in BACKENDS:
        runs = load_decisions(backend)
        print(f"\n################  backend={backend[0]}  (entity stage; gold={NGOLD})  ################")

        # -- pass decomposition --
        c = Counter()
        for d in runs:
            for k, (p1, p2) in d.items():
                c[(p1, p2, is_gold(k))] += 1
        p1_ufp, p1_utp = c[(False, True, False)], c[(False, True, True)]
        p2_ufp, p2_utp = c[(True, False, False)], c[(True, False, True)]
        both_fp, both_tp = c[(False, False, False)], c[(False, False, True)]
        print("Pass decomposition (3 runs):")
        print(f"  P1 unique veto: removes {p1_ufp} FP, costs {p1_utp} gold TP")
        print(f"  P2 unique veto: removes {p2_ufp} FP, costs {p2_utp} gold TP")
        print(f"  consensus veto: removes {both_fp} FP, costs {both_tp} gold TP")

        # -- combination frontier (per-run avg) --
        print("A. Combination rule (avg over 3 runs):")
        print(f"   {'policy':32}{'TP':>5}{'FP':>5}{'P%':>7}{'R%':>7}{'F1%':>7}")
        for name, fn in POLICIES.items():
            agg = [0.0] * 5
            for d in runs:
                m = metrics([k for k, (p1, p2) in d.items() if fn(p1, p2)])
                agg = [a + b for a, b in zip(agg, m)]
            tp, fp, P, R, F = [a / 3 for a in agg]
            print(f"   {name:32}{tp:5.0f}{fp:5.0f}{P:7.1f}{R:7.1f}{F:7.1f}")

        # -- self-consistency ensemble (union candidate pool) --
        print("B. Self-consistency: approve if policy holds in >=k of 3 runs:")
        allkeys = set().union(*[set(d) for d in runs])
        for name, fn in [("AND", POLICIES["AND  (current: p1 & p2)"]),
                         ("P2-only", POLICIES["P2-only (drop P1)"]),
                         ("OR", POLICIES["OR   (reject iff BOTH reject)"])]:
            for k in (1, 2, 3):
                appr = [key for key in allkeys
                        if sum(1 for d in runs if key in d and fn(*d[key])) >= k]
                tp, fp, P, R, F = metrics(appr)
                print(f"   {name+' >=%d/3' % k:32}{tp:5.0f}{fp:5.0f}{P:7.1f}{R:7.1f}{F:7.1f}")


if __name__ == "__main__":
    main()
