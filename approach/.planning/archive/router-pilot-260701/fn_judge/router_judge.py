#!/usr/bin/env python3
"""Elegant recall path: an LLM JUDGE-ROUTER over general signals + specialized judges.

Instead of one global judge (which couples recall and precision through a single leniency
knob), a reasoning-off LLM router reads GENERAL, taboo-safe linguistic signals and dispatches
each candidate to a specialized judge tuned for that mention MODE:

  AFFIRMATIVE   the component is named plainly and asserted to participate
                -> STRICT entity gate (s21 two-pass P1 and P2). precision route.
  CONTRAST      the component is named inside a negation / contrast / exclusion
                ("not X", "other than X", "unlike X", "rather than X")
                -> CONTRAST judge: approve if the sentence still asserts a fact ABOUT this
                   component's role (comparison / alternative / exclusion), else reject.
  IMPLICIT      a generic/lowercase common-noun or an example ("the client", "a file server",
                "e.g. a database") whose referent is this component
                -> CONTEXT judge with anchor sentences: approve only if the anchors + context
                   pin the referent to THIS component.
  ANAPHORA      a pronoun / role phrase ("it", "the service") refers back to the component
                -> COREF gate (s21 coref rubric + antecedent context).
  CODEPATH/ABSENT  reference is only a code path, or the component is genuinely absent
                -> REJECT (no judge).

Only the routed mode's judge runs, so distractors that route to AFFIRMATIVE/CODEPATH/ABSENT
never get the lenient treatment — the router, not a global rubric, is the precision gate.

ALL LLM calls reasoning-off. Router signals are general linguistic categories (no benchmark
terms); the component name is a runtime input, exactly as s21 passes it to the validator.

Run:  python3 router_judge.py         (routes + judges + scores; cached in router_cache.json)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_judges as RJ   # shares CASES, parse, ctx_block, run_batches, _client, rubric consts

HERE = Path(__file__).resolve().parent
RCACHE = HERE / "router_cache.json"
CASES = RJ.CASES

# ── router ───────────────────────────────────────────────────────────────────

ROUTER_RULES = (
    "Classify HOW the component is referenced in the SENTENCE, using only general language "
    "signals (not domain knowledge). Choose exactly one MODE:\n"
    "  AFFIRMATIVE - the component's name (or a plain alias) appears and the sentence states "
    "it does / provides / contains / is something, with no negation or contrast.\n"
    "  CONTRAST - the component's name appears inside a negation or contrast or exclusion: "
    "\"not X\", \"no X\", \"other than X\", \"unlike X\", \"rather than X\", \"instead of X\".\n"
    "  IMPLICIT - the component is referred to WITHOUT its proper name: by a generic/lowercase "
    "common noun, a role phrase, or as a concrete example (\"e.g. a ...\", \"such as a ...\"), "
    "and you must use context to tell which component it is.\n"
    "  ANAPHORA - the reference is a pronoun or bare role phrase (\"it\", \"they\", \"the "
    "service\", \"the module\") pointing back to something named earlier.\n"
    "  CODEPATH - the only occurrence is inside a dotted code/package path (x.y.z).\n"
    "  ABSENT - the component is not referenced in the sentence at all.")


def prompt_router(batch):
    body = "\n".join(RJ.ctx_block(c, i) for i, c in batch)
    return ("You route candidate documentation-to-component trace links to the right checker. "
            "For each case, decide the reference MODE.\n\n" + ROUTER_RULES +
            "\n\nCASES:\n" + body +
            '\n\nReturn JSON: {"routes":[{"case":1,"mode":"AFFIRMATIVE"}]}\nJSON only:')


def parse_routes(txt):
    a, b = txt.find("{"), txt.rfind("}")
    if a < 0 or b < 0:
        return {}
    try:
        obj = json.loads(txt[a:b + 1])
    except Exception:
        return {}
    out = {}
    for r in obj.get("routes", []):
        try:
            out[int(r["case"])] = str(r["mode"]).strip().upper()
        except Exception:
            pass
    return out


def run_router(client):
    out = {}
    for k in range(0, len(CASES), RJ.BATCH):
        sub = list(enumerate(CASES[k:k + RJ.BATCH], start=1))
        resp = client.query(prompt_router(sub), timeout=180)
        routes = parse_routes(resp.text if resp.success else "")
        for idx, c in sub:
            out[c["id"]] = routes.get(idx, "AFFIRMATIVE")   # default to the strict route
        print(f"    router: {min(k+RJ.BATCH,len(CASES))}/{len(CASES)}", file=sys.stderr)
    return out


# ── specialized judges (reasoning-off) ───────────────────────────────────────

CONTRAST_RULES = (
    "The component appears inside a negation, contrast, or exclusion. Approve when the sentence "
    "still asserts a fact ABOUT THIS component's role in the system -- it is compared against, "
    "excluded from, or offered as an alternative to something (\"provided by Y, not X\", "
    "\"systems other than X\", \"unlike X\"). All of these record a real relationship to X. "
    "Reject only when the sentence denies that X is part of the system at all, or the token is a "
    "different entity / a product-brand name.")

CONTEXT_RULES = (
    "The component is referred to WITHOUT its proper name -- by a generic common noun, a role "
    "phrase, or a concrete example. Use the anchor sentences (where the component IS named) plus "
    "the local context to decide the referent. Approve ONLY when the context makes it clear the "
    "generic/example phrase denotes THIS specific component, and the sentence makes an "
    "architectural claim about it. Reject when the referent is ambiguous between components, or "
    "the phrase is ordinary vocabulary with no specific referent.")


def prompt_contrast(batch):
    body = "\n".join(RJ.ctx_block(c, i, anchors=True) for i, c in batch)
    return ("You validate trace links where the component is named in a CONTRAST or NEGATION.\n\n"
            + CONTRAST_RULES + "\n\nFor each case, FIRST quote the exact contrast/negation words, "
            "THEN decide approve true/false.\n\nCASES:\n" + body +
            '\n\nReturn JSON: {"validations":[{"case":1,"claim":"<quote>","approve":true}]}\nJSON only:')


def prompt_context(batch):
    body = "\n".join(RJ.ctx_block(c, i, anchors=True, antecedent=True) for i, c in batch)
    return ("You validate trace links where the component is referenced WITHOUT its proper "
            "name.\n\n" + CONTEXT_RULES + "\n\nFor each case, FIRST quote the referring words and "
            "the anchor that fixes the referent (or \"none\"), THEN decide approve true/false.\n\n"
            "CASES:\n" + body +
            '\n\nReturn JSON: {"validations":[{"case":1,"claim":"<quote or none>","approve":true}]}\nJSON only:')


ROUTES = ("AFFIRMATIVE", "CONTRAST", "IMPLICIT", "ANAPHORA", "CODEPATH", "ABSENT")


def dispatch(routing, client, cache):
    """Run the specialized judge for each route; write final verdict per case into cache."""
    groups = defaultdict(list)
    for c in CASES:
        groups[routing[c["id"]]].append(c)

    verdict = {}
    # AFFIRMATIVE -> strict s21 entity two-pass (P1 and P2)
    aff = groups.get("AFFIRMATIVE", [])
    if aff:
        p1 = RJ.run_batches(aff, lambda b: RJ.prompt_entity_pass(b, RJ.P1_FOCUS, True), client, "route.AFFIRM.P1")
        p2 = RJ.run_batches(aff, lambda b: RJ.prompt_entity_pass(b, RJ.P2_FOCUS, True), client, "route.AFFIRM.P2")
        for c in aff:
            verdict[c["id"]] = bool(p1.get(c["id"]) and p2.get(c["id"]))
    # CONTRAST
    con = groups.get("CONTRAST", [])
    if con:
        r = RJ.run_batches(con, prompt_contrast, client, "route.CONTRAST")
        for c in con:
            verdict[c["id"]] = bool(r.get(c["id"]))
    # IMPLICIT -> context judge with anchors
    imp = groups.get("IMPLICIT", [])
    if imp:
        r = RJ.run_batches(imp, prompt_context, client, "route.IMPLICIT")
        for c in imp:
            verdict[c["id"]] = bool(r.get(c["id"]))
    # ANAPHORA -> coref gate
    ana = groups.get("ANAPHORA", [])
    if ana:
        r = RJ.run_batches(ana, RJ.prompt_coref_pass, client, "route.ANAPHORA")
        for c in ana:
            verdict[c["id"]] = bool(r.get(c["id"]))
    # CODEPATH / ABSENT -> reject
    for m in ("CODEPATH", "ABSENT"):
        for c in groups.get(m, []):
            verdict[c["id"]] = False

    cache["routing"] = routing
    cache["verdict"] = verdict
    return verdict


# ── scoring ──────────────────────────────────────────────────────────────────

def score(routing, verdict):
    by_label = defaultdict(list)
    for c in CASES:
        by_label[c["label"]].append(c)

    print("=" * 84)
    print("JUDGE-ROUTER (LLM router over general signals -> specialized judges) — gpt-5.4, reasoning-off")
    print("=" * 84)
    print("\nRouting mix by label (how each label's cases were routed):")
    print(f"  {'label':<9}" + "".join(f"{m[:7]:>9}" for m in ROUTES))
    for lab in ("R-TP", "R-TN", "NP-FN", "NP-CTRL"):
        cnt = Counter(routing[c["id"]] for c in by_label[lab])
        print(f"  {lab:<9}" + "".join(f"{cnt.get(m,0):>9}" for m in ROUTES))

    print("\nApproval rate per label (HIGH on R-TP/NP-FN = recall; LOW on R-TN/NP-CTRL = precision):")
    for lab in ("R-TP", "R-TN", "NP-FN", "NP-CTRL"):
        sub = by_label[lab]
        k = sum(1 for c in sub if verdict[c["id"]])
        print(f"  {lab:<9} {k:>3}/{len(sub):<3} = {100*k/len(sub) if sub else 0:>3.0f}%")

    # consistent-FN slice
    import report as REP
    consistent, perrun, tot = REP.fn_status()
    cons_cases, seen = [], set()
    for c in CASES:
        k = (c["project"], c["sentence_num"], c["component_id"])
        if k in consistent and k not in seen:
            seen.add(k); cons_cases.append(c)
    kk = sum(1 for c in cons_cases if verdict[c["id"]])
    print(f"\nREMAINING-FN SLICE: router approves {kk}/{len(cons_cases)} consistent FN "
          f"({100*kk/len(cons_cases):.0f}%)")
    print("  per-FN (mode | verdict):")
    for c in sorted(cons_cases, key=lambda c: (c["project"], c["sentence_num"])):
        print(f"    {c['project'][:8]+' s'+str(c['sentence_num']):<16}{str(c['component'])[:14]:<15}"
              f"{routing[c['id']]:<12}{'APPROVE' if verdict[c['id']] else 'reject'}")


def main():
    cache = json.loads(RCACHE.read_text()) if RCACHE.exists() else {}
    if "routing" in cache and "verdict" in cache and "--refresh" not in sys.argv:
        routing, verdict = cache["routing"], cache["verdict"]
    else:
        cl = RJ._client()
        print("== routing ==", file=sys.stderr)
        routing = run_router(cl)
        print("== dispatch to specialized judges ==", file=sys.stderr)
        verdict = dispatch(routing, cl, cache)
        RCACHE.write_text(json.dumps(cache, indent=1))
    score(routing, verdict)


if __name__ == "__main__":
    main()
