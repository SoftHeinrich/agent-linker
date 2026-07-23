#!/usr/bin/env python3
"""DESIGN SPACE — proposer aggressiveness x judge firmness (the combinatory idea).

The proposer and the judge are not a fixed pipeline: a firmer judge lets the
proposer run at lower precision (the judge removes its FPs), while a more permissive
routed judge needs a more precise proposer. This sweeps the grid on the SAME cached
GTP proposals (corpus_proposer_cache.json), re-judging under each config, and scores
corpus model-doc macro-F1 (sentence, component) vs SAD-SAM gold, 5 proj x 3 runs.

AXIS 1 — proposer (filter cached proposals by the mode GTP emitted):
  affirm  AFFIRMATIVE only              (conservative — the name is present)
  named   AFFIRMATIVE + CONTRAST        (name present, incl. contrast/negation)
  all     every mode                    (aggressive — incl. IMPLICIT / ANAPHORA)

AXIS 2 — judge (the verifier for the NEW candidates):
  none    accept all proposals          (raw proposer; shows why a judge is needed)
  strict  reuse s21 two-pass gate on EVERY candidate   (firm; kills FPs)
  routed  mode-calibrated judge-router  (permissive on IMPLICIT/CONTRAST/ANAPHORA)

Reuses live_run.py loaders + the shared judge cache. `strict` adds new gpt-5.4 calls
(s21 gate on all candidates); `routed` reuses the live-run verdicts. Reasoning-off.

Run:  python3 design_space.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import live_run as LR
import build_cases as BC
import run_judges as RJ
import router_judge as RJU
import proposer as PR

PROJECTS, RUNS = LR.PROJECTS, LR.RUNS
JCACHE = HERE / "corpus_judge_cache.json"

PROPOSER_VARIANTS = {
    "affirm": {"AFFIRMATIVE"},
    "named": {"AFFIRMATIVE", "CONTRAST"},
    "all": {"AFFIRMATIVE", "CONTRAST", "IMPLICIT", "ANAPHORA", "CODEPATH"},
}
JUDGES = ["none", "strict", "routed"]


def load_proposals(catalog_mode="name"):
    """project -> [(s, name, id, mode)], deduped to one mode per (s,id)."""
    cache = json.loads(LR.CORPUS_PCACHE.read_text())
    out = defaultdict(list)
    for project in PROJECTS:
        rost = LR.roster(project)
        seen = set()
        for s in sorted(BC.sentences(project)):
            raw = cache.get(f"{catalog_mode}|{project}|{s}", [])
            grounded, _ = PR.ground(raw, list(rost.keys()))
            for r in grounded:
                cid = rost[r["component"]]
                if (s, cid) in seen:
                    continue
                seen.add((s, cid))
                out[project].append((s, r["component"], cid, r["mode"]))
    return out


def make_case(project, s, name, quote, sents, amb):
    return {"id": f"{project}|{s}|{LR.roster(project)[name]}", "project": project,
            "sentence_num": s, "component": name, "sentence": sents[s],
            "preceding": sents.get(s - 1, ""), "matched_text": name,
            "mention_type": None, "is_ambiguous": name in amb, "coref": None,
            "anchors": LR._anchors(project, s, name, sents)}


def judge_candidates(project, cands, client, jcache):
    """cands: [(s, name, id, mode)]. Fills jcache with STRICT|id and mode|id verdicts."""
    sents = BC.sentences(project)
    amb = LR.ambiguous_names(project)
    cases = {(s, cid): (make_case(project, s, name, None, sents, amb), mode)
             for (s, name, cid, mode) in cands}

    # STRICT: s21 two-pass on ALL candidates
    need = [(k, c) for k, (c, _m) in cases.items() if f"STRICT|{c['id']}" not in jcache]
    if need:
        p1 = LR._run(need, lambda b: RJ.prompt_entity_pass(b, RJ.P1_FOCUS, True), client, RJ)
        p2 = LR._run(need, lambda b: RJ.prompt_entity_pass(b, RJ.P2_FOCUS, True), client, RJ)
        for (k, c) in need:
            jcache[f"STRICT|{c['id']}"] = bool(p1.get(c["id"]) and p2.get(c["id"]))

    # ROUTED: mode-calibrated (per-mode judges), cache key mode|id
    by_mode = defaultdict(list)
    for k, (c, mode) in cases.items():
        m = mode if mode in RJU.ROUTES else "AFFIRMATIVE"
        if f"{m}|{c['id']}" not in jcache:
            by_mode[m].append((k, c))
    for m, group in by_mode.items():
        if m == "AFFIRMATIVE":
            p1 = LR._run(group, lambda b: RJ.prompt_entity_pass(b, RJ.P1_FOCUS, True), client, RJ)
            p2 = LR._run(group, lambda b: RJ.prompt_entity_pass(b, RJ.P2_FOCUS, True), client, RJ)
            for (k, c) in group:
                jcache[f"{m}|{c['id']}"] = bool(p1.get(c["id"]) and p2.get(c["id"]))
        else:
            build = {"CONTRAST": RJU.prompt_contrast, "IMPLICIT": RJU.prompt_context,
                     "ANAPHORA": RJ.prompt_coref_pass}.get(m)
            if build is None:                    # CODEPATH
                for (k, c) in group:
                    jcache[f"{m}|{c['id']}"] = False
            else:
                r = LR._run(group, build, client, RJ)
                for (k, c) in group:
                    jcache[f"{m}|{c['id']}"] = bool(r.get(c["id"]))
    return cases


def keep(judge, cid_key, mode, jcache):
    if judge == "none":
        return True
    if judge == "strict":
        return jcache.get(f"STRICT|{cid_key}", False)
    m = mode if mode in RJU.ROUTES else "AFFIRMATIVE"
    return jcache.get(f"{m}|{cid_key}", False)


def main():
    proposals = load_proposals("name")
    golds = {p: LR.gold_ids(p) for p in PROJECTS}

    # judge EVERY unique proposed (s,id) under BOTH judges (cached). Clean per-run
    # scoring below decides add/skip; no cross-run auto-keep confound.
    jcache = json.loads(JCACHE.read_text()) if JCACHE.exists() else {}
    client = RJ._client()
    for project in PROJECTS:
        inter = set.intersection(*(LR.s21_final(project, r) for r in RUNS))
        # candidates in EVERY run's final are always present -> never need a verdict
        cands = [(s, name, cid, mode) for (s, name, cid, mode) in proposals[project]
                 if (s, cid) not in inter]
        judge_candidates(project, cands, client, jcache)
        print(f"  judged {project}: {len(cands)} candidates (not in all-run intersection)",
              file=sys.stderr)
    JCACHE.write_text(json.dumps(jcache, indent=1))

    base_per = LR.macro_over_runs(LR.s21_final, golds)
    bF = LR._avg(base_per, 2); bP = LR._avg(base_per, 0); bR = LR._avg(base_per, 1)

    print("=" * 92)
    print("DESIGN SPACE — corpus model-doc macro-F1 (sentence,component), gpt-5.4, reasoning-off")
    print("=" * 92)
    print("per-run marginal: add a judged-keep proposal only to runs whose s21 final lacks it.")
    print(f"baseline s21:  P={bP:.4f}  R={bR:.4f}  F1={bF:.4f}\n")
    print(f"  {'proposer':<9}{'judge':<8}{'P':>8}{'R':>8}{'F1':>8}{'dF1':>8}"
          f"{'addTP':>7}{'addFP':>7}")
    grid = []
    for pv, modes in PROPOSER_VARIANTS.items():
        for jv in JUDGES:
            def aug(project, run, _modes=modes, _jv=jv):
                links = set(LR.s21_final(project, run))
                for (s, name, cid, mode) in proposals[project]:
                    if mode not in _modes or (s, cid) in links:
                        continue
                    if keep(_jv, f"{project}|{s}|{cid}", mode, jcache):
                        links.add((s, cid))
                return links

            per = LR.macro_over_runs(aug, golds)
            P, R, F1 = LR._avg(per, 0), LR._avg(per, 1), LR._avg(per, 2)
            # added-link tallies vs gold (union over runs, illustrative)
            added = {p: set() for p in PROJECTS}
            for project in PROJECTS:
                base_all = set().union(*(LR.s21_final(project, r) for r in RUNS))
                for (s, name, cid, mode) in proposals[project]:
                    if mode in modes and (s, cid) not in base_all \
                            and keep(jv, f"{project}|{s}|{cid}", mode, jcache):
                        added[project].add((s, cid))
            atp = sum(len(added[p] & golds[p]) for p in PROJECTS)
            afp = sum(len(added[p] - golds[p]) for p in PROJECTS)
            print(f"  {pv:<9}{jv:<8}{P:>8.4f}{R:>8.4f}{F1:>8.4f}{F1-bF:>+8.4f}{atp:>7}{afp:>7}")
            grid.append((pv, jv, P, R, F1, atp, afp))
        print()

    best = max(grid, key=lambda g: g[4])
    print(f"BEST F1: proposer={best[0]} judge={best[1]} -> F1={best[4]:.4f} "
          f"(dF1 {best[4]-bF:+.4f}, +{best[5]}TP/+{best[6]}FP)")
    (HERE / "design_space_summary.json").write_text(json.dumps(
        {"baseline_F1": bF, "grid": [
            {"proposer": g[0], "judge": g[1], "P": g[2], "R": g[3], "F1": g[4],
             "newTP": g[5], "newFP": g[6]} for g in grid]}, indent=1))
    print("wrote design_space_summary.json")


if __name__ == "__main__":
    main()
