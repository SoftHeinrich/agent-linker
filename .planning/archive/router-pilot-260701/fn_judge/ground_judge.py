"""Roster/profile-grounded DISCRIMINATIVE judge (context augmentation, LLM-driven, no regex).

The precision leak in the router was actor-vs-component confusion ("the user"/"the browser" ->
UI). Fix by AUGMENTING CONTEXT: show the judge the full component roster with one-line role
profiles (from build_profiles.py), and ask a DISCRIMINATIVE question — is THIS component the
specific referent, versus a different component in the roster, an external actor/object, or
nothing? Seeing the alternatives + roles lets the model reject "the user"->UI while keeping
"the datastore"->GAE Datastore. Reasoning-off. Prompt/context/structure only.

Configs:
  G_all     grounded judge applied to every case (no router)
  G_router  router modes, but the lenient routes (CONTRAST/IMPLICIT/ANAPHORA) use the grounded
            judge; AFFIRMATIVE stays strict two-pass, CODEPATH/ABSENT reject (router + grounding)
Run: python3 ground_judge.py            (caches grounded verdicts; scores both configs)
"""
import json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_judges as RJ
import router_judge as R1

HERE = Path(__file__).resolve().parent
CASES = RJ.CASES
PROFILES = json.loads((HERE / "profiles.json").read_text())
GCACHE = HERE / "ground_cache.json"
LENIENT = {"CONTRAST", "IMPLICIT", "ANAPHORA"}


def roster_block(project, target):
    lines = []
    for comp, role in sorted(PROFILES.get(project, {}).items()):
        mark = "  <- CANDIDATE" if comp == target else ""
        lines.append(f'    - {comp}: {role}{mark}')
    return "COMPONENT ROSTER (the only components that exist in this system):\n" + "\n".join(lines)


GROUND_RULES = (
    "Decide whether the SENTENCE refers to the CANDIDATE component specifically, as a "
    "participant in an architectural claim. Use the roster: every real component is listed with "
    "its role. Approve when the referring words (a name, an alias, a lowercased/terminal form, a "
    "pronoun with a clear antecedent, an example, or a contrastive mention) denote the CANDIDATE "
    "and the sentence says something about its role. REJECT when the referring words instead "
    "denote: (a) an external actor or object that is NOT a component (a user, a browser, a "
    "request, a file on disk, a person), (b) a DIFFERENT component in the roster than the "
    "candidate, or (c) nothing specific. A contrast/negation ('provided by Y, not the "
    "candidate') still refers to the candidate — approve it. When the referring words are an "
    "ordinary English word that matches the candidate's name, approve only if the roster role "
    "(not the ordinary meaning) is what the sentence is talking about.")


def prompt_ground(project, batch):
    parts = []
    for i, c in batch:
        blk = RJ.ctx_block(c, i, anchors=True, antecedent=True)
        parts.append(roster_block(project, c["component"]) + "\n" + blk)
    body = "\n\n".join(parts)
    return ("You are a grounded trace-link judge. Each case gives the full component roster with "
            "roles, then a candidate link from a sentence to one CANDIDATE component.\n\n"
            + GROUND_RULES +
            "\n\nFor each case, FIRST quote the referring words and name what they denote "
            "(the candidate / a different component / an external actor / nothing), THEN decide "
            "approve true/false.\n\nCASES:\n" + body +
            '\n\nReturn JSON: {"validations":[{"case":1,"claim":"<quote; denotes ...>",'
            '"approve":true}]}\nJSON only:')


def run_grounded(cache):
    """grounded verdict for every case, batched within project (roster is per-project)."""
    cl = RJ._client()
    byproj = defaultdict(list)
    for c in CASES:
        byproj[c["project"]].append(c)
    verd = cache.get("grounded", {})
    for proj, cs in byproj.items():
        pending = [c for c in cs if c["id"] not in verd or "--refresh" in sys.argv]
        for k in range(0, len(pending), RJ.BATCH):
            sub = list(enumerate(pending[k:k + RJ.BATCH], start=1))
            resp = cl.query(prompt_ground(proj, sub), timeout=180)
            res = RJ.parse(resp.text if resp.success else "")
            for idx, c in sub:
                verd[c["id"]] = bool(res.get(idx, False))
            print(f"    ground:{proj}: {min(k+RJ.BATCH,len(pending))}/{len(pending)}", file=sys.stderr)
    cache["grounded"] = verd
    GCACHE.write_text(json.dumps(cache, indent=1))
    return verd


def score(name, verdict):
    by = defaultdict(list)
    for c in CASES:
        by[c["label"]].append(c)
    row = {}
    for lab in ("R-TP", "R-TN", "NP-FN", "NP-CTRL"):
        sub = by[lab]
        row[lab] = (sum(1 for c in sub if verdict[c["id"]]), len(sub))
    import report as REP
    consistent, _, _ = REP.fn_status()
    seen, cons = set(), []
    for c in CASES:
        k = (c["project"], c["sentence_num"], c["component_id"])
        if k in consistent and k not in seen:
            seen.add(k); cons.append(c)
    fn = sum(1 for c in cons if verdict[c["id"]])
    print(f"  {name:<10} " + "  ".join(
        f"{lab} {row[lab][0]:>2}/{row[lab][1]:<3}({100*row[lab][0]//row[lab][1] if row[lab][1] else 0:>3}%)"
        for lab in ("R-TP", "R-TN", "NP-FN", "NP-CTRL")) + f"   FN {fn}/{len(cons)}")
    return verdict


def main():
    cache = json.loads(GCACHE.read_text()) if GCACHE.exists() else {}
    grounded = run_grounded(cache)

    # G_all
    v_all = {c["id"]: grounded[c["id"]] for c in CASES}

    # G_router: router modes; lenient routes use grounded verdict, else v1's route verdict
    rc = json.loads((HERE / "router_cache.json").read_text())
    routing, v1 = rc["routing"], rc["verdict"]
    v_router = {}
    for c in CASES:
        cid = c["id"]
        if routing[cid] in LENIENT:
            v_router[cid] = bool(grounded[cid])
        else:
            v_router[cid] = bool(v1.get(cid))
    cache["verdict_G_router"] = v_router
    cache["verdict_G_all"] = v_all
    GCACHE.write_text(json.dumps(cache, indent=1))

    print("=" * 92)
    print("GROUNDED DISCRIMINATIVE JUDGE (roster + role profiles as context) — gpt-5.4, reasoning-off")
    print("  want HIGH R-TP/NP-FN (recall), LOW R-TN/NP-CTRL (precision), HIGH FN")
    print("=" * 92)
    score("G_all", v_all)
    score("G_router", v_router)


if __name__ == "__main__":
    main()
