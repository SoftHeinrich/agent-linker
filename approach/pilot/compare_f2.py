"""Project-level F2 (primary) and F1 (secondary) for the variants in a run dir.

Recall-led reading: F2 first, F1 second, and recall shown alongside because a
verifier can only remove -- proposal recall bounds everything downstream.

    ../.venv/bin/python pilot/compare_f2.py ../results/reading_e2e_terra_r1 [more dirs]
"""
from __future__ import annotations
import glob, json, statistics as st, sys
from pathlib import Path

PROJ = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]

def load(d: Path):
    runs = sorted(glob.glob(str(d / "ablation_*.json")))
    if not runs:
        return None
    return json.load(open(runs[-1]))

def main(dirs):
    per_dir = []
    for d in dirs:
        data = load(Path(d))
        if not data:
            print(f"  (no ablation json yet in {d})"); continue
        variants = sorted({v for p in data.values() for v in p})
        print(f"\n########## {Path(d).name} ##########")
        for proj in PROJ:
            if proj not in data: continue
            print(f"\n  {proj}")
            print(f"    {'variant':<14}{'P':>7}{'R':>7}{'F1':>7}{'F2':>7}{'calls':>7}")
            for v in variants:
                r = data[proj].get(v)
                if not r: continue
                print(f"    {v:<14}{r['P']*100:>7.1f}{r['R']*100:>7.1f}"
                      f"{r['F1']*100:>7.1f}{r['F2']*100:>7.1f}{r.get('llm_calls',0):>7}")
        print(f"\n  MACRO over projects   (F2 primary, F1 secondary)")
        print(f"    {'variant':<14}{'P':>7}{'R':>7}{'F1':>7}{'F2':>7}{'calls':>7}")
        base = None
        for v in variants:
            rows = [data[p][v] for p in PROJ if p in data and v in data[p]]
            if not rows: continue
            m = {k: st.mean(r[k] for r in rows) for k in ("P", "R", "F1", "F2")}
            calls = st.mean(r.get("llm_calls", 0) for r in rows)
            if base is None: base = m
            d1, d2 = (m["F1"] - base["F1"]) * 100, (m["F2"] - base["F2"]) * 100
            tag = "" if v == variants[0] else f"   dF2 {d2:+.2f}  dF1 {d1:+.2f}"
            print(f"    {v:<14}{m['P']*100:>7.1f}{m['R']*100:>7.1f}"
                  f"{m['F1']*100:>7.1f}{m['F2']*100:>7.1f}{calls:>7.1f}{tag}")
        per_dir.append((d, data))
    return per_dir

if __name__ == "__main__":
    main(sys.argv[1:] or ["../results/reading_e2e_terra_r1"])
