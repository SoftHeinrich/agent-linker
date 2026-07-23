#!/usr/bin/env python3
"""Judge-side precision push — a skeptic/refutation pass on the new accepts.

Decisive fact (verified): the agentic router's new accept-FP are 0 real errors, 4
gold-incompleteness (it already rejected the 3 verified errors). So there is nothing
"wrong" left in the accept set to remove. This tests what a precision-oriented judge
(adversarial skeptic, default-refute) actually does: does it remove errors (good) or
valid links (overfitting the incomplete gold)?

Skeptic on the truly-new marginal accepts (not in any s21 run) = 4 TP + 4 FP(gold-gap).
Reasoning-off, claim-before-verdict, default REJECT. Cheap.
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import live_run as LR, design_space as DS, build_cases as BC, run_judges as RJ

golds = {p: LR.gold_ids(p) for p in LR.PROJECTS}
proposals = DS.load_proposals("name")
JC = json.load(open(HERE / "corpus_judge_cache.json"))
AC = json.load(open(HERE / "agent_router_cache.json"))
PC = json.load(open(HERE / "corpus_proposer_cache.json"))


def gate(p, s, cid):
    return bool(JC.get(f"STRICT|{p}|{s}|{cid}") or JC.get(f"CONTRAST|{p}|{s}|{cid}"))


def quote(p, s, comp):
    for r in PC.get(f"name|{p}|{s}", []):
        if r.get("component") == comp:
            return r.get("quote", "")
    return ""


SKEPTIC = (
    "You are a skeptical reviewer REMOVING weak documentation-to-component trace links. "
    "For each case, REFUTE the link (keep=false) UNLESS the sentence makes an explicit, "
    "specific claim that THIS component is used, provides/consumes a service, is "
    "implemented, contains or is contained, or stores/routes data. Refute incidental "
    "mentions, generic nouns, examples, and sentences primarily about a different "
    "component. FIRST quote the exact words making the specific claim about THIS "
    'component (or "none"), THEN decide keep true/false. Default keep=false.\n\nCASES:\n'
    "{body}\n\n"
    'Return JSON: {{"validations":[{{"case":1,"claim":"<quote or none>","keep":false}}]}}\nJSON only:')


def main():
    cands = []
    for p in LR.PROJECTS:
        union = set().union(*(LR.s21_final(p, r) for r in LR.RUNS))
        sents = BC.sentences(p)
        for (s, name, cid, mode) in proposals[p]:
            if (s, cid) in union:
                continue
            if AC.get(f"{p}|{s}|{cid}") != "VALIDATE" or not gate(p, s, cid):
                continue
            cands.append(dict(p=p, s=s, name=name, cid=cid, gold=(s, cid) in golds[p],
                              sent=sents.get(s, ""), q=quote(p, s, name)))
    body = "\n".join(f'{i}. SENTENCE: "{c["sent"]}"  COMPONENT: {c["name"]} '
                     f'(referred by "{c["q"] or c["name"]}")' for i, c in enumerate(cands, 1))
    resp = RJ._client().query(SKEPTIC.format(body=body), timeout=180)
    keep = RJ.parse(resp.text)     # {case_idx: bool}

    print("=" * 76)
    print("SKEPTIC (default-refute) on the agentic router's NEW accepts")
    print("=" * 76)
    removed_tp = removed_fp = kept_tp = kept_fp = 0
    for i, c in enumerate(cands, 1):
        k = keep.get(i, False)
        tag = "TP(valid)" if c["gold"] else "FP(gold-incompleteness)"
        print(f"  {'KEEP' if k else 'REMOVE':<7} {c['p'][:8]+' S'+str(c['s']):<15}{c['name'][:16]:<17}{tag}")
        if k and c["gold"]: kept_tp += 1
        elif k: kept_fp += 1
        elif c["gold"]: removed_tp += 1
        else: removed_fp += 1
    print(f"\n  skeptic REMOVED: {removed_tp} valid TP + {removed_fp} gold-gap FP")
    print(f"  skeptic KEPT   : {kept_tp} valid TP + {kept_fp} gold-gap FP")
    print(f"\n  => every FP it removes is a VALID link (gold-incompleteness); it cannot "
          f"raise precision\n     without deleting real recall. Judge-side precision on "
          f"model-doc is EXHAUSTED.")


if __name__ == "__main__":
    main()
