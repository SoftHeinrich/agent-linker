#!/usr/bin/env python3
"""BOUNDED-AUTONOMY agentic router — keep the value, drop the heuristic dispatch.

Goal (per direction): NOT to improve F1, but to (a) KEEP the measured value
(named+routed = F1 0.9506, precision-safe), (b) gain small, CONTROLLED autonomy so the
method is honestly agentic, and (c) replace the hard-coded mode->judge dispatch with an
LLM decision (router less heuristic).

The safety construction (why autonomy can't cost value):
  - DEFAULT path = the trusted s21 gate. Doing nothing special == named+routed.
  - The agent may only DIVERT a candidate (route-to-code / reject / flag-gold-gap) or
    send it to a gate (arch/contrast). It can NEVER add a link the gate rejects.
  - Every model-doc ACCEPT is floored: accept == agent-sent-to-gate AND gate-approves.
  => accept set is a SUBSET of gate-approved; precision cannot regress below the gate,
     and recall only drops if the agent actively diverts a good candidate (measured).

So the agent has real autonomy (it decides the action, no mode->table lookup), but it is
BOUNDED by the unchanged validator. Reasoning-off; the NOTE is externalized answer tokens.

Run:  python3 agent_router.py     (cached in agent_router_cache.json)
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import live_run as LR, design_space as DS, build_cases as BC, run_judges as RJ

JC = json.load(open(HERE / "corpus_judge_cache.json"))     # trusted gate verdicts (STRICT|, CONTRAST|, ...)
PCACHE = json.load(open(HERE / "corpus_proposer_cache.json"))
RCACHE = HERE / "agent_router_cache.json"
golds = {p: LR.gold_ids(p) for p in LR.PROJECTS}
proposals = DS.load_proposals("name")


def gate_approve(p, s, cid):
    """Trusted floor: the unchanged s21 strict gate OR the contrast gate approves."""
    return bool(JC.get(f"STRICT|{p}|{s}|{cid}") or JC.get(f"CONTRAST|{p}|{s}|{cid}"))


def anchors(p, s, name, sents):
    out = [f"S{i}: {sents[i]}" for i in sorted(sents) if i != s and BC.standalone(name, sents[i])]
    return out[:3]


def quote(p, s, comp):
    for r in PCACHE.get(f"name|{p}|{s}", []):
        if r.get("component") == comp:
            return r.get("quote", "")
    return ""


# ── the agent (bounded action set; NOTE = externalized deliberation) ──────────

PROMPT = (
    "You are the routing step of a documentation-to-architecture trace-link recovery "
    "system. A candidate names a software COMPONENT possibly referenced by a SENTENCE. "
    "For EACH case write a one-line NOTE (how the component is referenced; use anchors "
    "only to pin the referent, never to justify), then choose ACTION:\n"
    "  VALIDATE - a plausible architecture-level reference; send it to the validator "
    "(the validator makes the final keep/reject — when unsure, prefer VALIDATE).\n"
    "  CODE     - the words name a concrete code element (class/package/file); it belongs "
    "to the code-level linker, not the architecture set.\n"
    "  REJECT   - the component is clearly not the referent (incidental word, other entity).\n"
    "Default to VALIDATE unless CODE or REJECT is clear.\n\nCASES:\n{body}\n\n"
    'Return JSON: {{"decisions":[{{"case":1,"note":"...","action":"VALIDATE|CODE|REJECT"}}]}}\nJSON only:')


def fmt(c, i):
    ls = [f'Case {i}: component "{c["name"]}" referenced by "{c["q"] or c["name"]}"',
          f'  SENTENCE: {c["sent"]}']
    if c["anch"]:
        ls.append("  anchors: " + " | ".join(c["anch"]))
    return "\n".join(ls)


def parse(txt):
    a, b = txt.find("{"), txt.rfind("}")
    try:
        obj = json.loads(txt[a:b + 1])
    except Exception:
        return {}
    return {int(d["case"]): str(d.get("action", "VALIDATE")).upper().strip()
            for d in obj.get("decisions", []) if "case" in d}


def run_agent(cands, cache):
    client = RJ._client(); B = 8
    todo = [c for c in cands if c["id"] not in cache]
    for k in range(0, len(todo), B):
        sub = todo[k:k + B]
        v = parse(client.query(PROMPT.format(body="\n".join(fmt(c, i) for i, c in enumerate(sub, 1))),
                                timeout=180).text)
        for i, c in enumerate(sub, 1):
            cache[c["id"]] = v.get(i, "VALIDATE")       # safe default
        RCACHE.write_text(json.dumps(cache, indent=1))
        print(f"  agent {min(k+B,len(todo))}/{len(todo)}", file=sys.stderr)


def main():
    # marginal candidates = GTP proposals not in every run's s21 final
    cands = []
    for p in LR.PROJECTS:
        sents = BC.sentences(p); inter = set.intersection(*(LR.s21_final(p, r) for r in LR.RUNS))
        for (s, name, cid, mode) in proposals[p]:
            if (s, cid) in inter:
                continue
            cands.append(dict(id=f"{p}|{s}|{cid}", p=p, s=s, name=name, cid=cid, mode=mode,
                              sent=sents.get(s, ""), q=quote(p, s, name),
                              anch=anchors(p, s, name, sents)))
    cache = json.loads(RCACHE.read_text()) if RCACHE.exists() else {}
    run_agent(cands, cache)

    acts = Counter(cache[c["id"]] for c in cands)
    # bounded accept: agent said VALIDATE AND the trusted gate approves
    accept = defaultdict(set); routed_code = defaultdict(set); rejected = defaultdict(set)
    for c in cands:
        a = cache[c["id"]]
        if a == "CODE":     routed_code[c["p"]].add((c["s"], c["cid"]))
        elif a == "REJECT": rejected[c["p"]].add((c["s"], c["cid"]))
        elif gate_approve(c["p"], c["s"], c["cid"]):      # VALIDATE ∩ gate  == the FLOOR
            accept[c["p"]].add((c["s"], c["cid"]))

    def aug(p, run): return LR.s21_final(p, run) | accept[p]
    base = LR.macro_over_runs(LR.s21_final, golds); per = LR.macro_over_runs(aug, golds)
    bF = LR._avg(base, 2)

    print("=" * 78)
    print("BOUNDED-AUTONOMY AGENTIC ROUTER — keep value, LLM-decided routing, gate-floored")
    print("=" * 78)
    print(f"agent actions over {len(cands)} marginal candidates: {dict(acts)}")
    print(f"\n  {'config':<26}{'P':>8}{'R':>8}{'F1':>8}")
    print(f"  {'baseline s21':<26}{LR._avg(base,0):>8.4f}{LR._avg(base,1):>8.4f}{bF:>8.4f}")
    print(f"  {'named+routed (target)':<26}{0.9897:>8.4f}{0.9173:>8.4f}{0.9506:>8.4f}")
    print(f"  {'agentic (this)':<26}{LR._avg(per,0):>8.4f}{LR._avg(per,1):>8.4f}{LR._avg(per,2):>8.4f}")
    # value-preservation + autonomy audit
    atp = sum(len(accept[p] & golds[p]) for p in LR.PROJECTS)
    afp = sum(len(accept[p] - golds[p]) for p in LR.PROJECTS)
    code_n = sum(len(routed_code[p]) for p in LR.PROJECTS)
    rej_n = sum(len(rejected[p]) for p in LR.PROJECTS)
    # downside-bound check: is every accept gate-approved?
    floored = all(gate_approve(p, s, cid) for p in LR.PROJECTS for (s, cid) in accept[p])
    print(f"\n  value: F1 {LR._avg(per,2):.4f} vs target 0.9506  (Δ {LR._avg(per,2)-0.9506:+.4f})")
    print(f"  gate-floor holds (every accept gate-approved): {floored}")
    print(f"  autonomy: {code_n} routed to CODE-linker, {rej_n} rejected, "
          f"accepts +{atp}TP/+{afp}FP")
    (HERE / "agent_router_summary.json").write_text(json.dumps(
        {"F1": LR._avg(per, 2), "target": 0.9506, "actions": dict(acts),
         "code_routed": code_n, "rejected": rej_n, "accept_TP": atp, "accept_FP": afp,
         "gate_floor_holds": floored}, indent=1))


if __name__ == "__main__":
    main()
