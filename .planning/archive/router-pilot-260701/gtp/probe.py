#!/usr/bin/env python3
"""EMPIRICAL PROBE — does GTP surface the never-proposed FN without flooding distractors?

The existing pilot (`fn_judge/router_ceiling.py`) fed the GOLD component in and
measured the judge/router *assuming a perfect proposer*. This probe measures the one
thing that assumption hides: **proposer precision**. GTP is handed only the sentence
+ previous sentence + the full component catalog (gold NOT leaked) and must choose.

Target set: the 14 sentences whose gold link s21 NEVER proposed (NP-FN, 16 gold
pairs) plus their 42 curated sibling distractors (NP-CTRL) — the exact ceiling set
from `fn_judge`. Scoring is against the real SAD-SAM gold (benchmark CSVs).

Two catalog modes test the PROPOSAL's "context must constrain, not enrich" caution
on the *proposer* side (fn_judge found role profiles backfire for the JUDGE; the
proposer is untested):
  name  — catalog is component names only.
  role  — catalog is names + one-line role descriptions (profiles.json).

Run:  python3 probe.py            # both modes, cached
      python3 probe.py name       # one mode
Cheap (~14 sentences x N modes, cached in proposer_cache.json).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FN = HERE.parent / "fn_judge"
sys.path.insert(0, str(FN))
sys.path.insert(0, str(HERE))

import build_cases as BC              # load_gold, _fill_names_from_model, BENCH
from proposer import GroundedTypedProposer, MODES

CASES = json.loads((FN / "cases.json").read_text())
PROFILES = json.loads((FN / "profiles.json").read_text())
PCACHE = HERE / "proposer_cache.json"


# ── target set + gold ────────────────────────────────────────────────────────

def target_sentences():
    """14 never-proposed sentences -> {(proj,s): {sent, prev, npfn:set, ctrl:set}}."""
    t = {}
    for c in CASES:
        if c["label"] not in ("NP-FN", "NP-CTRL"):
            continue
        k = (c["project"], c["sentence_num"])
        e = t.setdefault(k, {"sent": c["sentence"], "prev": c.get("preceding") or "",
                             "npfn": set(), "ctrl": set()})
        (e["npfn"] if c["label"] == "NP-FN" else e["ctrl"]).add(c["component"])
    return t


def gold_names_by_project(project):
    """{sentence_num: {gold component names}} from the real SAD-SAM gold."""
    gold = BC.load_gold(project)
    id2name = {eid: None for (_s, eid) in gold}
    BC._fill_names_from_model(project, id2name)
    by_s = defaultdict(set)
    for (s, eid) in gold:
        nm = id2name.get(eid)
        if nm:
            by_s[s].add(nm)
    return by_s


# ── run one catalog mode ─────────────────────────────────────────────────────

def run_mode(mode, targets, gold_by_proj):
    gtp = GroundedTypedProposer(cache_path=PCACHE, catalog_mode=mode)
    per = []
    for (proj, s), e in sorted(targets.items()):
        names = list(PROFILES[proj].keys())
        roles = PROFILES[proj]
        key = f"{proj}|{s}"
        proposed = {r["component"]: r["mode"]
                    for r in gtp.propose(key, e["sent"], e["prev"], names, roles)}
        gold = gold_by_proj[proj].get(s, set())
        pset = set(proposed)
        per.append({
            "proj": proj, "s": s, "sent": e["sent"],
            "proposed": proposed, "gold": gold,
            "npfn": e["npfn"], "ctrl": e["ctrl"],
            "npfn_hit": e["npfn"] & pset,           # never-proposed gold recovered
            "ctrl_hit": e["ctrl"] & pset,           # curated sibling distractors emitted
            "tp": pset & gold, "fp": pset - gold,    # vs full gold
        })
    return per, gtp.dropped_total


def report(mode, per, dropped):
    npfn_tot = sum(len(r["npfn"]) for r in per)
    ctrl_tot = sum(len(r["ctrl"]) for r in per)
    npfn_hit = sum(len(r["npfn_hit"]) for r in per)
    ctrl_hit = sum(len(r["ctrl_hit"]) for r in per)
    tp = sum(len(r["tp"]) for r in per)
    fp = sum(len(r["fp"]) for r in per)
    prop = sum(len(r["proposed"]) for r in per)
    gold = sum(len(r["gold"]) for r in per)
    print("=" * 90)
    print(f"GTP PROPOSER PROBE — catalog_mode={mode!r} — gpt-5.4, reasoning-off, gold NOT leaked")
    print("=" * 90)
    print(f"never-proposed gold RECOVERED : {npfn_hit}/{npfn_tot} "
          f"({100*npfn_hit/npfn_tot:.0f}%)   <- proposer recall (the missing number)")
    print(f"curated sibling distractors   : {ctrl_hit}/{ctrl_tot} "
          f"({100*ctrl_hit/ctrl_tot:.0f}%)   <- proposer over-proposal (lower=better)")
    print(f"vs full SAD-SAM gold on these 14 sentences:")
    print(f"   proposals={prop}  gold={gold}  TP={tp}  FP={fp}  "
          f"P={tp/prop if prop else 0:.3f}  R={tp/gold if gold else 0:.3f}")
    print(f"   ungrounded refs dropped (hallucinated names) = {dropped}")
    print("\nper-sentence (G=gold-recovered  C=sibling  •=other):")
    for r in sorted(per, key=lambda r: (r["proj"], r["s"])):
        tags = []
        for comp, md in sorted(r["proposed"].items()):
            mark = ("G" if comp in r["npfn"] else
                    "C" if comp in r["ctrl"] else
                    ("g" if comp in r["gold"] else "•"))
            tags.append(f"{mark}{comp}[{md[:4]}]")
        miss = r["npfn"] - r["npfn_hit"]
        print(f"  {r['proj'][:8]+' s'+str(r['s']):<16} want={sorted(r['npfn'])}"
              f"{'  MISS='+str(sorted(miss)) if miss else ''}")
        print(f"       -> {'  '.join(tags) if tags else '(nothing proposed)'}")
    return {"mode": mode, "npfn_hit": npfn_hit, "npfn_tot": npfn_tot,
            "ctrl_hit": ctrl_hit, "ctrl_tot": ctrl_tot, "tp": tp, "fp": fp,
            "prop": prop, "gold": gold, "dropped": dropped}


def main():
    modes = sys.argv[1:] or ["name", "role"]
    targets = target_sentences()
    gold_by_proj = {p: gold_names_by_project(p)
                    for p in {proj for (proj, _s) in targets}}
    summary = []
    for mode in modes:
        per, dropped = run_mode(mode, targets, gold_by_proj)
        summary.append(report(mode, per, dropped))
        print()
    if len(summary) > 1:
        print("=" * 90)
        print("COMPARISON (proposer recall vs sibling over-proposal)")
        print("=" * 90)
        print(f"  {'mode':<8}{'FN recall':>12}{'sibling over':>14}{'full-gold P':>13}{'dropped':>9}")
        for s in summary:
            print(f"  {s['mode']:<8}{s['npfn_hit']}/{s['npfn_tot']:>10}"
                  f"{s['ctrl_hit']}/{s['ctrl_tot']:>12}"
                  f"{s['tp']/s['prop'] if s['prop'] else 0:>13.3f}{s['dropped']:>9}")
    (HERE / "probe_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
