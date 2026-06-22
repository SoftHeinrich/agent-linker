#!/usr/bin/env python3
"""Score the s_linker19U N=3 sweeps and lay them head-to-head against s_linker20_union.

s19U = full un-minimized prompts (incl. few-shots) + Framing C UNION.
s20_union = minimized prompts (few-shots emptied) + Framing C UNION.
They differ ONLY by the 12 Phase-46 prompt cuts — so the macro-F1 delta isolates
"keep the full prompts/few-shots" vs "minimized prompts" under identical union logic.
"""
import json, glob, statistics as st, os

DS = ["mediastore", "teastore", "jabref", "bigbluebutton", "teammates"]
VAR = "s_linker19U"


def f1(run_dir, ds):
    fs = sorted(glob.glob(f"{run_dir}/{ds}/ablation_*.json"))
    if not fs:
        return None
    j = json.load(open(fs[-1]))
    if ds in j and isinstance(j[ds], dict):
        for _, vv in j[ds].items():
            if isinstance(vv, dict) and "F1" in vv:
                return vv["F1"]
    return None


def score_slot(slot, runs=("run1", "run2", "run3")):
    permac, perds = [], {d: [] for d in DS}
    for r in runs:
        rd = f"{slot}/{r}"
        if not os.path.isdir(rd):
            continue
        vals = [f1(rd, d) for d in DS]
        if all(v is not None for v in vals):
            permac.append(st.mean(vals))
            for d, v in zip(DS, vals):
                perds[d].append(v)
    return permac, perds


# s_linker20_union reference (fresh N=3, from results/v2.6.5_s20union/README.md)
S20U = {
    "sonnet": {"macro": 0.9276, "ds": {"mediastore": 0.9670, "teastore": 0.9590,
                                       "jabref": 0.9910, "bigbluebutton": 0.8143, "teammates": 0.9064}},
    "gpt":    {"macro": 0.8939, "ds": {"mediastore": 0.9561, "teastore": 0.9811,
                                       "jabref": 0.9322, "bigbluebutton": 0.7634, "teammates": 0.8366}},
}

SLOTS = {"sonnet": "results/v2.6.5_s19U_sonnet", "gpt": "results/v2.6.5_s19U/gpt"}

for backend, slot in SLOTS.items():
    permac, perds = score_slot(slot)
    print(f"\n{'='*72}\nBACKEND: {backend}   (s19U slot: {slot})\n{'='*72}")
    if not permac:
        print("  (no complete runs scored yet)")
        continue
    s19u_macro = st.mean(permac)
    s20u_macro = S20U[backend]["macro"]
    sd = st.pstdev(permac) if len(permac) > 1 else 0.0
    print(f"  s19U  macro-F1 = {s19u_macro:.4f}  (N={len(permac)}, sd={sd:.4f}, per-run={[round(x,4) for x in permac]})")
    print(f"  s20U  macro-F1 = {s20u_macro:.4f}  (reference fresh N=3)")
    print(f"  Δ (s19U − s20U) = {s19u_macro - s20u_macro:+.4f}   "
          f"-> {'s19U (full prompts/few-shots) wins' if s19u_macro > s20u_macro else 's20U (minimized) wins or ties'}")
    print(f"\n  {'dataset':14s} {'s19U':>8s} {'s20U':>8s} {'Δ':>9s}")
    for d in DS:
        if perds[d]:
            a = st.mean(perds[d]); b = S20U[backend]["ds"][d]
            print(f"  {d:14s} {a:8.4f} {b:8.4f} {a-b:+9.4f}")
