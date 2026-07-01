#!/usr/bin/env python3
"""Judge-router v2 — recall routes + a reasoning-off SKEPTIC verifier (propose -> verify).

v1 (`router_judge.py`) decoupled recall from precision but still leaked 8 distractors, almost
all in ONE mode: the IMPLICIT route resolving a generic ACTOR word to an ambiguous component
("user"/"browser" -> UI, "gui", "model") — semantic association, not a real reference. The real
IMPLICIT recoveries are LEXICAL ("logic" -> Logic, "datastore" -> GAE Datastore).

Elegant fix (Mode-4 adversarial verify, reasoning-off): let the lenient routes PROPOSE, then a
single SKEPTIC pass verifies every lenient-route approval. The skeptic rejects unless the
referring words LEXICALLY denote THIS component (its name, a lowercased/terminal form, or a
documented alias) rather than a generic actor/object merely associated with it. AFFIRMATIVE
(strict) approvals and all rejects pass through unchanged. Goal: keep the recovered FN, drop the
generic-association FP -> no precision regress.

Run:  python3 router_v2.py            (uses router_cache.json v1 routing+verdict; caches skeptic)
"""
import json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_judges as RJ
import router_judge as R1

HERE = Path(__file__).resolve().parent
RCACHE = HERE / "router_cache.json"
CASES = RJ.CASES
LENIENT = {"CONTRAST", "IMPLICIT", "ANAPHORA"}

SKEPTIC_RULES = (
    "Each case was tentatively accepted as a trace link. Your job is to REJECT it unless the "
    "sentence gives UNAMBIGUOUS evidence that this SPECIFIC named component is the referent. "
    "Reject when the referring words are a generic ACTOR or OBJECT merely associated with the "
    "component (a user, a browser, a request, a page) rather than the component itself. Approve "
    "ONLY when the referring words LEXICALLY denote this component -- its proper name, that name "
    "lowercased or a terminal word of it, a documented abbreviation/alias, or (for a pronoun) an "
    "antecedent that itself names the component. If the component name is an ordinary English "
    "word, demand this lexical anchor strictly. When the evidence is only topical/semantic "
    "association, REJECT.")


def prompt_skeptic(batch):
    body = "\n".join(RJ.ctx_block(c, i, anchors=True, antecedent=True) for i, c in batch)
    return ("You are a strict verifier removing weak trace links. " + SKEPTIC_RULES +
            "\n\nFor each case, FIRST quote the exact referring words and say whether they name "
            "the component lexically or only associate with it, THEN decide keep true/false.\n\n"
            "CASES:\n" + body +
            '\n\nReturn JSON: {"validations":[{"case":1,"claim":"<quote; lexical|associative>",'
            '"approve":true}]}\nJSON only:')


def main():
    cache = json.loads(RCACHE.read_text())
    routing, v1 = cache["routing"], cache["verdict"]
    # candidates to verify: lenient-route approvals
    todo = [c for c in CASES if routing[c["id"]] in LENIENT and v1.get(c["id"])]
    skept = cache.get("skeptic", {})
    pending = [c for c in todo if c["id"] not in skept or "--refresh" in sys.argv]
    if pending:
        cl = RJ._client()
        print(f"== skeptic verify on {len(pending)} lenient-route approvals ==", file=sys.stderr)
        res = RJ.run_batches(pending, prompt_skeptic, cl, "skeptic")
        for c in pending:
            skept[c["id"]] = bool(res.get(c["id"]))
        cache["skeptic"] = skept
        RCACHE.write_text(json.dumps(cache, indent=1))

    # v2 verdict = v1, but lenient approvals must survive the skeptic
    v2 = {}
    for c in CASES:
        cid = c["id"]
        if routing[cid] in LENIENT and v1.get(cid):
            v2[cid] = bool(skept.get(cid, False))
        else:
            v2[cid] = bool(v1.get(cid))
    cache["verdict_v2"] = v2
    RCACHE.write_text(json.dumps(cache, indent=1))

    # score by label
    by = defaultdict(list)
    for c in CASES:
        by[c["label"]].append(c)
    print("=" * 78)
    print("JUDGE-ROUTER v2 (recall routes + skeptic verify) — gpt-5.4, reasoning-off")
    print("=" * 78)
    print(f"\n{'label':<9}{'v1 approve':>14}{'v2 approve':>14}   (R-TP/NP-FN=recall, R-TN/NP-CTRL=precision)")
    for lab in ("R-TP", "R-TN", "NP-FN", "NP-CTRL"):
        sub = by[lab]
        k1 = sum(1 for c in sub if v1.get(c["id"]))
        k2 = sum(1 for c in sub if v2[c["id"]])
        print(f"{lab:<9}{k1:>8}/{len(sub):<4}{k2:>8}/{len(sub):<4}")

    import report as REP
    consistent, _, _ = REP.fn_status()
    cons, seen = [], set()
    for c in CASES:
        k = (c["project"], c["sentence_num"], c["component_id"])
        if k in consistent and k not in seen:
            seen.add(k); cons.append(c)
    k2 = sum(1 for c in cons if v2[c["id"]])
    print(f"\nremaining consistent FN recovered: v2 {k2}/{len(cons)}")


if __name__ == "__main__":
    main()
