#!/usr/bin/env python3
"""AGENT PROBE — can a scratchpad-deliberating router pick richer ACTIONS?

Workflow today: proposer emits mode -> fixed dispatch -> binary keep/reject. The routed
judge KEEPS all 48 implicit/anaphora candidates it sees (13 gold + 35 "FP"), because its
only move is keep/reject and it (correctly) finds them all link-like.

Agent step: give ONE reasoning-off LLM a scratchpad (deliberation = answer tokens, NOT
thinking) and a richer ACTION set, and let it DECIDE per candidate:
  ARCH   - a real architecture-level trace link -> accept
  CODE   - the sentence names a code element (class/package/file) -> route to the
           doc->code linker (different target space), not the arch gate
  REJECT - the component is not really the referent -> drop
Evidence given = the constrain-not-enrich kind: prev sentence + anchor sentences that
name the component (pins the referent). No regex, no rubric — the LLM triages.

We score the agent's actions against the gold-verified categories (FINDINGS §8):
CODE-structure(10) should route CODE; gold(13) + gold-gap(~22) should stay ARCH; the 3
hand-verified errors should REJECT. This measures whether AGENCY over the action set
recovers the routing the fixed workflow cannot.

Run:  python3 agent_probe.py     (cached in agent_cache.json)
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import live_run as LR, design_space as DS, build_cases as BC, run_judges as RJ

jc = json.load(open(HERE / "corpus_judge_cache.json"))
pcache = json.load(open(HERE / "corpus_proposer_cache.json"))
ACACHE = HERE / "agent_cache.json"
golds = {p: LR.gold_ids(p) for p in LR.PROJECTS}
proposals = DS.load_proposals("name")
CODE = re.compile(r'[A-Z][a-z]+(?:Servlet|Factory|Action|Adapter)|web\.xml|index\.html|\bx\.[a-z]+|\bDb classes\b')
ERRORS = {("teastore", 31, "Auth"), ("teammates", 36, "UI"), ("teammates", 165, "UI")}


def anchors(p, s, name, sents):
    out = [f"S{i}: {sents[i]}" for i in sorted(sents) if i != s and BC.standalone(name, sents[i])]
    return out[:4]


def quote(p, s, comp):
    for r in pcache.get(f"name|{p}|{s}", []):
        if r.get("component") == comp:
            return r.get("quote", "")
    return ""


def build_candidates():
    cands = []
    for p in LR.PROJECTS:
        sents = BC.sentences(p)
        for (s, name, cid, mode) in proposals[p]:
            if mode not in ("IMPLICIT", "ANAPHORA") or not jc.get(f"{mode}|{p}|{s}|{cid}"):
                continue
            gold = (s, cid) in golds[p]
            truth = ("GOLD" if gold else "CODE" if CODE.search(sents.get(s, ""))
                     else "ERROR" if (p, s, name) in ERRORS else "GAP")
            cands.append(dict(p=p, s=s, name=name, mode=mode, gold=gold, truth=truth,
                              sent=sents.get(s, ""), prev=sents.get(s - 1, ""),
                              quote=quote(p, s, name), anchors=anchors(p, s, name, sents)))
    return cands


PROMPT = (
    "You triage candidate trace links between a documentation sentence and a named software "
    "component. For EACH case: (1) write a one-line NOTE — how the component is referenced, "
    "and whether the referring words name a CODE element (a class/package/file identifier) or "
    "describe an ARCHITECTURE role; use the anchors only to pin the referent, not to justify. "
    "(2) choose ACTION:\n"
    "  ARCH   - a genuine architecture-level reference to this component.\n"
    "  CODE   - the words name a concrete code element (class/package/file); this belongs to a "
    "code-level linker, not the architecture link set.\n"
    "  REJECT - this component is not actually the referent (incidental word, wrong entity).\n\n"
    "CASES:\n{body}\n\n"
    'Return JSON: {{"decisions":[{{"case":1,"note":"...","action":"ARCH|CODE|REJECT"}}]}}\nJSON only:')


def fmt(c, i):
    lines = [f'Case {i}: component "{c["name"]}" referenced by "{c["quote"] or c["name"]}"',
             f'  PREV: {c["prev"]}', f'  SENTENCE: {c["sent"]}']
    if c["anchors"]:
        lines.append("  anchors (where the component is named): " + " | ".join(c["anchors"][:3]))
    return "\n".join(lines)


def parse(txt):
    a, b = txt.find("{"), txt.rfind("}")
    try:
        obj = json.loads(txt[a:b + 1])
    except Exception:
        return {}
    out = {}
    for d in obj.get("decisions", []):
        try:
            out[int(d["case"])] = str(d.get("action", "")).upper().strip()
        except Exception:
            pass
    return out


def main():
    cands = build_candidates()
    cache = json.loads(ACACHE.read_text()) if ACACHE.exists() else {}
    client = RJ._client()
    B = 8
    for k in range(0, len(cands), B):
        sub = cands[k:k + B]
        if all(f"{c['p']}|{c['s']}|{c['name']}" in cache for c in sub):
            continue
        body = "\n".join(fmt(c, i) for i, c in enumerate(sub, 1))
        verd = parse(client.query(PROMPT.format(body=body), timeout=180).text)
        for i, c in enumerate(sub, 1):
            cache[f"{c['p']}|{c['s']}|{c['name']}"] = verd.get(i, "ARCH")
        ACACHE.write_text(json.dumps(cache, indent=1))
        print(f"  {min(k+B,len(cands))}/{len(cands)}", file=sys.stderr)

    # score: agent ACTION vs gold-verified truth
    conf = defaultdict(Counter)
    for c in cands:
        conf[c["truth"]][cache[f"{c['p']}|{c['s']}|{c['name']}"]] += 1
    print("\n=== agent ACTION vs verified truth (rows=truth, cols=agent action) ===")
    print(f"  {'truth':<8}{'n':>4}{'ARCH':>7}{'CODE':>7}{'REJECT':>8}")
    for t in ("GOLD", "GAP", "CODE", "ERROR"):
        r = conf[t]; n = sum(r.values())
        print(f"  {t:<8}{n:>4}{r['ARCH']:>7}{r['CODE']:>7}{r['REJECT']:>8}")
    # headline outcomes
    code_routed = conf["CODE"]["CODE"]; code_n = sum(conf["CODE"].values())
    gold_kept = conf["GOLD"]["ARCH"]; gold_n = sum(conf["GOLD"].values())
    err_rej = conf["ERROR"]["REJECT"]; err_n = sum(conf["ERROR"].values())
    print(f"\n  code-structure routed to CODE : {code_routed}/{code_n}")
    print(f"  gold kept as ARCH (no loss)   : {gold_kept}/{gold_n}")
    print(f"  genuine errors REJECTed       : {err_rej}/{err_n}")
    print("\n  per-case (truth -> agent):")
    for c in cands:
        a = cache[f"{c['p']}|{c['s']}|{c['name']}"]
        flag = "" if (c["truth"], a) in {("GOLD","ARCH"),("GAP","ARCH"),("CODE","CODE"),("ERROR","REJECT")} else "  <-diff"
        print(f"    {c['p'][:8]+' S'+str(c['s']):<15}{c['name'][:16]:<17}{c['truth']:<6}-> {a}{flag}")


if __name__ == "__main__":
    main()
