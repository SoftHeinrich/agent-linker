#!/usr/bin/env python3
"""END-TO-END — GTP proposer -> mode router -> specialized judge, on the ceiling set.

`probe.py` showed the PROPOSER surfaces 11/16 never-proposed FN at 10% sibling
over-proposal. This closes the loop: each GTP proposal is routed by the mode GTP
itself emitted to the matching specialized judge (the same judges as
`fn_judge/router_judge.py`), with the real anchor sentences fed to the context
judge. We then measure what SURVIVES — the deployable recall (NP-FN kept) at the
realized precision (siblings/distractors kept).

Baseline for comparison: `fn_judge/router_ceiling.py` fed the GOLD component to the
router and recovered 43% (25/58) of never-proposed FN. Here a REAL proposer chooses
the candidates first, so this is the honest, non-oracle number.

Reasoning-off throughout. Cheap + cached (proposals reuse proposer_cache.json;
judge verdicts cached in e2e_cache.json).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FN = HERE.parent / "fn_judge"
sys.path.insert(0, str(FN))
sys.path.insert(0, str(HERE))

import build_cases as BC
import run_judges as RJ
import router_judge as RJU
from proposer import GroundedTypedProposer

CASES = json.loads((FN / "cases.json").read_text())
PROFILES = json.loads((FN / "profiles.json").read_text())
PCACHE = HERE / "proposer_cache.json"
ECACHE = HERE / "e2e_cache.json"

# per-(proj,s,component) side info from the labelled cases (ambiguity, coref, label)
_INFO = {(c["project"], c["sentence_num"], c["component"]): c for c in CASES}


def targets():
    t = {}
    for c in CASES:
        if c["label"] not in ("NP-FN", "NP-CTRL"):
            continue
        k = (c["project"], c["sentence_num"])
        e = t.setdefault(k, {"sent": c["sentence"], "prev": c.get("preceding") or "",
                             "npfn": set(), "ctrl": set()})
        (e["npfn"] if c["label"] == "NP-FN" else e["ctrl"]).add(c["component"])
    return t


def gold_by_project(project):
    gold = BC.load_gold(project)
    id2name = {eid: None for (_s, eid) in gold}
    BC._fill_names_from_model(project, id2name)
    by_s = defaultdict(set)
    for (s, eid) in gold:
        nm = id2name.get(eid)
        if nm:
            by_s[s].add(nm)
    return by_s


def anchors_for(project, sent_num, comp, sents):
    """Recompute anchors the way build_cases did: other sentences naming this comp."""
    out = []
    for i in sorted(sents):
        if i == sent_num:
            continue
        if BC.standalone(comp, sents[i]):
            out.append(f"S{i}: {sents[i]}")
        if len(out) >= 4:
            break
    return out


def make_case(project, s, comp, quote, sent, prev, sents):
    info = _INFO.get((project, s, comp), {})
    return {
        "id": f"gtp|{project}|{s}|{comp}",
        "project": project, "sentence_num": s, "component": comp,
        "sentence": sent, "preceding": prev,
        "matched_text": quote or comp,
        "mention_type": info.get("mention_type"),
        "is_ambiguous": info.get("is_ambiguous", False),
        "coref": info.get("coref"),
        "anchors": info.get("anchors") or anchors_for(project, s, comp, sents),
    }


def dispatch_one(mode, case, client, cache):
    """Run the routed judge for a single case; return keep bool (cached)."""
    ck = f"{mode}|{case['id']}"
    if ck in cache:
        return cache[ck]
    m = mode if mode in RJU.ROUTES else "AFFIRMATIVE"
    if m == "AFFIRMATIVE":
        p1 = RJ.run_batches([case], lambda b: RJ.prompt_entity_pass(b, RJ.P1_FOCUS, True), client, "AFF.P1")
        p2 = RJ.run_batches([case], lambda b: RJ.prompt_entity_pass(b, RJ.P2_FOCUS, True), client, "AFF.P2")
        keep = bool(p1.get(case["id"]) and p2.get(case["id"]))
    elif m == "CONTRAST":
        keep = bool(RJ.run_batches([case], RJU.prompt_contrast, client, "CON").get(case["id"]))
    elif m == "IMPLICIT":
        keep = bool(RJ.run_batches([case], RJU.prompt_context, client, "IMP").get(case["id"]))
    elif m == "ANAPHORA":
        keep = bool(RJ.run_batches([case], RJ.prompt_coref_pass, client, "ANA").get(case["id"]))
    else:                                   # CODEPATH / ABSENT -> reject
        keep = False
    cache[ck] = keep
    ECACHE.write_text(json.dumps(cache, indent=1))
    return keep


def main():
    mode_cat = sys.argv[1] if len(sys.argv) > 1 else "name"
    T = targets()
    gold_p = {p: gold_by_project(p) for p in {proj for (proj, _s) in T}}
    sents_p = {p: BC.sentences(p) for p in {proj for (proj, _s) in T}}
    gtp = GroundedTypedProposer(cache_path=PCACHE, catalog_mode=mode_cat)
    cache = json.loads(ECACHE.read_text()) if ECACHE.exists() else {}
    client = RJ._client()

    rows = []
    for (proj, s), e in sorted(T.items()):
        names = list(PROFILES[proj].keys())
        proposed = gtp.propose(f"{proj}|{s}", e["sent"], e["prev"], names, PROFILES[proj])
        for r in proposed:
            comp, md = r["component"], r["mode"]
            case = make_case(proj, s, comp, r.get("quote"), e["sent"], e["prev"], sents_p[proj])
            keep = dispatch_one(md, case, client, cache)
            gold = gold_p[proj].get(s, set())
            rows.append({"proj": proj, "s": s, "comp": comp, "mode": md, "keep": keep,
                         "is_npfn": comp in e["npfn"], "is_ctrl": comp in e["ctrl"],
                         "is_gold": comp in gold})

    npfn_tot = sum(len(e["npfn"]) for e in T.values())
    ctrl_tot = sum(len(e["ctrl"]) for e in T.values())
    kept = [r for r in rows if r["keep"]]
    npfn_kept = len({(r["proj"], r["s"], r["comp"]) for r in kept if r["is_npfn"]})
    ctrl_kept = len({(r["proj"], r["s"], r["comp"]) for r in kept if r["is_ctrl"]})
    tp = sum(1 for r in kept if r["is_gold"])
    fp = sum(1 for r in kept if not r["is_gold"])

    print("=" * 90)
    print(f"GTP END-TO-END (propose -> route -> judge) — catalog={mode_cat!r}, gpt-5.4, reasoning-off")
    print("=" * 90)
    print(f"never-proposed FN kept after judge : {npfn_kept}/{npfn_tot} "
          f"({100*npfn_kept/npfn_tot:.0f}%)   <- DEPLOYABLE recall on the ceiling set")
    print(f"   (proposer surfaced 11/16; the judge is the second gate)")
    print(f"curated siblings kept (leak)       : {ctrl_kept}/{ctrl_tot} ({100*ctrl_kept/ctrl_tot:.0f}%)")
    print(f"kept vs full gold : TP={tp} FP={fp} P={tp/(tp+fp) if kept else 0:.3f}")
    print(f"\nbaseline (router_ceiling, GOLD fed in): 43% of never-proposed FN, 0% sibling over-link")
    print("\nper proposal (mode | verdict):")
    for r in sorted(rows, key=lambda r: (r["proj"], r["s"], r["comp"])):
        tag = ("NP-FN" if r["is_npfn"] else "SIBLING" if r["is_ctrl"]
               else "gold" if r["is_gold"] else "other")
        print(f"  {r['proj'][:8]+' s'+str(r['s']):<16}{r['comp'][:16]:<17}"
              f"{r['mode'][:5]:<6}{'KEEP' if r['keep'] else 'drop':<5}{tag}")
    (HERE / "e2e_summary.json").write_text(json.dumps(
        {"catalog": mode_cat, "npfn_kept": npfn_kept, "npfn_tot": npfn_tot,
         "ctrl_kept": ctrl_kept, "ctrl_tot": ctrl_tot, "tp": tp, "fp": fp}, indent=1))


if __name__ == "__main__":
    main()
