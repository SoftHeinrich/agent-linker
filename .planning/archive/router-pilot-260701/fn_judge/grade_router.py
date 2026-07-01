"""Router + EVIDENCE-TYPED judge (elegant, LLM-driven, no-regress attempt).

Diagnosis: the router's residual leaks are cases where a generic ACTOR word is only topically
associated with a component ("the user" -> UI). The real recoveries are NAMED or RESOLVED
(a proper name / documented alias / a pronoun with a naming antecedent). So instead of a
stricter reject rubric, have the lenient-route judge EMIT THE EVIDENCE TYPE it used, and deploy
only the trustworthy types. This is a calibration STRUCTURE, not a regex.

The judge tags each approval:
  NAMED       the component's proper name / a documented alias / a lowercased-or-terminal form
              of its name appears in the sentence.
  RESOLVED    a pronoun or generic phrase whose antecedent (or the doc's alias for it) NAMES the
              component -- the reference is pinned, not guessed.
  ASSOCIATIVE the words are a generic actor/object only topically related to the component
              (a user, a browser, a request) -- NOT the component itself.
Deploy rule: keep NAMED + RESOLVED; drop ASSOCIATIVE.

Applied to the router's lenient-routed cases (CONTRAST/IMPLICIT/ANAPHORA); AFFIRMATIVE stays
strict two-pass, CODEPATH/ABSENT reject. Reasoning-off.
Run: python3 grade_router.py
"""
import json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_judges as RJ

HERE = Path(__file__).resolve().parent
CASES = RJ.CASES
LENIENT = {"CONTRAST", "IMPLICIT", "ANAPHORA"}
GCACHE = HERE / "grade_cache.json"

GRADE_RULES = (
    "For each case decide whether the sentence refers to the CANDIDATE component, and if so tag "
    "the EVIDENCE TYPE:\n"
    "  NAMED - the component's proper name, a documented alias, or a lowercased/terminal word of "
    "its name appears in the sentence (including inside a negation or contrast, or as a concrete "
    "example).\n"
    "  RESOLVED - a pronoun or generic phrase whose antecedent in the context NAMES the "
    "component, so the referent is pinned (not guessed).\n"
    "  ASSOCIATIVE - the words are a generic actor or object only topically related to the "
    "component (a user, a browser, a request, a page, a file) rather than the component itself.\n"
    "  NONE - the component is not referred to at all.\n"
    "Pick the single best tag. Only NAMED and RESOLVED are real trace links.")


def prompt_grade(batch):
    body = "\n".join(RJ.ctx_block(c, i, anchors=True, antecedent=True) for i, c in batch)
    return ("You type the evidence for candidate trace links between a sentence and a named "
            "component.\n\n" + GRADE_RULES +
            "\n\nFor each case, FIRST quote the referring words, THEN give the tag.\n\nCASES:\n"
            + body +
            '\n\nReturn JSON: {"validations":[{"case":1,"claim":"<quote>","tag":"NAMED"}]}\nJSON only:')


def parse_tags(txt):
    a, b = txt.find("{"), txt.rfind("}")
    if a < 0 or b < 0:
        return {}
    try:
        obj = json.loads(txt[a:b + 1])
    except Exception:
        return {}
    out = {}
    for v in obj.get("validations", []):
        try:
            out[int(v["case"])] = str(v.get("tag", "NONE")).strip().upper()
        except Exception:
            pass
    return out


def main():
    rc = json.loads((HERE / "router_cache.json").read_text())
    routing, v1 = rc["routing"], rc["verdict"]
    lenient_cases = [c for c in CASES if routing[c["id"]] in LENIENT]

    cache = json.loads(GCACHE.read_text()) if GCACHE.exists() else {}
    tags = cache.get("tags", {})
    pending = [c for c in lenient_cases if c["id"] not in tags or "--refresh" in sys.argv]
    if pending:
        cl = RJ._client()
        print(f"== evidence-typing {len(pending)} lenient-routed cases ==", file=sys.stderr)
        for k in range(0, len(pending), RJ.BATCH):
            sub = list(enumerate(pending[k:k + RJ.BATCH], start=1))
            resp = cl.query(prompt_grade(sub), timeout=180)
            res = parse_tags(resp.text if resp.success else "")
            for idx, c in sub:
                tags[c["id"]] = res.get(idx, "NONE")
            print(f"    grade: {min(k+RJ.BATCH,len(pending))}/{len(pending)}", file=sys.stderr)
        cache["tags"] = tags
        GCACHE.write_text(json.dumps(cache, indent=1))

    KEEP = {"NAMED", "RESOLVED"}
    verdict = {}
    for c in CASES:
        cid = c["id"]
        if routing[cid] in LENIENT:
            verdict[cid] = tags.get(cid, "NONE") in KEEP
        else:
            verdict[cid] = bool(v1.get(cid))
    cache["verdict_grade"] = verdict
    GCACHE.write_text(json.dumps(cache, indent=1))

    # tag distribution by label on lenient routes
    print("\nEvidence-tag distribution on lenient-routed cases (by gold label):")
    tl = defaultdict(lambda: defaultdict(int))
    for c in lenient_cases:
        tl[c["label"]][tags.get(c["id"], "NONE")] += 1
    for lab in ("R-TP", "R-TN", "NP-FN", "NP-CTRL"):
        print(f"  {lab:<8} " + "  ".join(f"{t}={tl[lab][t]}" for t in ("NAMED", "RESOLVED", "ASSOCIATIVE", "NONE")))

    by = defaultdict(list)
    for c in CASES:
        by[c["label"]].append(c)
    import report as REP
    consistent, _, _ = REP.fn_status()
    seen, cons = set(), []
    for c in CASES:
        k = (c["project"], c["sentence_num"], c["component_id"])
        if k in consistent and k not in seen:
            seen.add(k); cons.append(c)
    print("\nGRADE-ROUTER (keep NAMED+RESOLVED) vs router v1:")
    for name, vd in (("router v1", v1), ("grade-router", verdict)):
        line = "  " + f"{name:<14}"
        for lab in ("R-TP", "R-TN", "NP-FN", "NP-CTRL"):
            sub = by[lab]; k = sum(1 for c in sub if vd.get(c["id"]))
            line += f"{lab} {k:>2}/{len(sub):<3}({100*k//len(sub) if sub else 0:>3}%)  "
        fn = sum(1 for c in cons if vd.get(c["id"]))
        line += f"FN {fn}/{len(cons)}"
        print(line)


if __name__ == "__main__":
    main()
