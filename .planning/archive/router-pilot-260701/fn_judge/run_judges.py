#!/usr/bin/env python3
"""Feed every FN-experiment case through a set of judge STRUCTURES and cache verdicts.

ALL judges are REASONING-OFF (never set OPENAI_REASONING_EFFORT). s21 is a no-reasoning
config by design; a recall fix that needs reasoning is not deployable here. The only
"reasoning" is the Mode-5 claim-before-verdict OUTPUT field (billed as answer tokens), which
IS the s21 design. Structures differ only in rubric, context, and pass structure.

Judges (all gpt-5.4, reasoning-off):
  J0_s21     s21 replica: layered rubric, entity two-pass (P1 and P2) / coref single-pass,
             claim-before-verdict, bare context (preceding + mention_type). = "what s21 does".
  J0_amb     J0 + s21's actual evidence bundle (ambiguity flag + anchor sentences) on the
             entity pass — isolates whether that context is what rejects the FN.
  J1_soft    pilot DirectLinkJudge (softened): approve concrete examples, reject only
             exclusion/negation/product-name; single pass, claim-before-verdict.
  J2_recover recall rubric: topic/participant framing that approves contrastive/negated/
             example/generic/anaphoric mentions when the sentence conveys the component's
             role; richer context (preceding + anchors + coref antecedent). single pass.
  J3_vote    self-consistency: J2 rubric sampled K=3 at temperature=1.0, approve if >=2
             (majority) — also records ANY(>=1). Reasoning-off structural variance.

Verdicts cached in verdicts.json keyed judge|case_id (J3 stores the 3 samples).
Run:  python3 run_judges.py [J0_s21 J0_amb J1_soft J2_recover J3_vote]   (default: all)
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
# load OPENAI_API_KEY from approach/.env if present
_envf = Path(__file__).resolve().parents[2] / ".env"
if _envf.exists():
    for ln in _envf.read_text().splitlines():
        if "=" in ln and not ln.strip().startswith("#"):
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from llm_sad_sam.llm_client import LLMClient, LLMBackend
from llm_sad_sam.linkers.experimental.s_linker21 import (
    LAYERED_ENTITY_RULES, LAYERED_COREF_RULES, P1_FOCUS, P2_FOCUS,
    COREF_VALIDATION_FOCUS,
)

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "cases.json").read_text())
VCACHE = HERE / "verdicts.json"
BATCH = 8
MODEL = "gpt-5.4"


def _named(case):
    name = case.get("component")
    if name and re.search(rf'(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])',
                          case["sentence"], re.IGNORECASE):
        return True
    mt = case.get("matched_text")
    if mt and mt.lower() in case["sentence"].lower():
        return True
    return False


def mode(case):
    return "entity" if _named(case) else "coref"


def ctx_block(case, i, *, anchors=False, antecedent=False, ambiguity=False):
    lines = [f'Case {i}: "{case.get("matched_text") or case["component"]}" -> {case["component"]}']
    if case.get("preceding"):
        lines.append(f'  [prev: "{case["preceding"]}"]')
    lines.append(f'  SENTENCE: "{case["sentence"]}"')
    if case.get("mention_type"):
        lines.append(f'  mention_type: {case["mention_type"]}')
    if ambiguity and case.get("is_ambiguous"):
        lines.append('  note: this component name is AMBIGUOUS (often used as an ordinary word)')
    if antecedent and case.get("coref"):
        co = case["coref"]
        if co.get("antecedent_text"):
            lines.append(f'  [antecedent S{co.get("antecedent_sentence")}: '
                         f'"{co.get("antecedent_text")}"; reference: "{co.get("reference")}"]')
    if anchors and case.get("anchors"):
        lines.append("  Other sentences that name this component:")
        for a in case["anchors"][:4]:
            lines.append(f"    {a}")
    return "\n".join(lines)


def parse(txt):
    a, b = txt.find("{"), txt.rfind("}")
    if a < 0 or b < 0:
        return {}
    try:
        obj = json.loads(txt[a:b + 1])
    except Exception:
        # tolerate trailing junk: try array-only
        try:
            aa, bb = txt.find("["), txt.rfind("]")
            obj = {"validations": json.loads(txt[aa:bb + 1])}
        except Exception:
            return {}
    res = {}
    for v in obj.get("validations", []):
        try:
            k = int(v["case"])
            val = v.get("approve", v.get("keep"))
            res[k] = (val is True) or (isinstance(val, str) and val.strip().lower() == "true")
        except Exception:
            pass
    return res


# ── judge prompt builders (return one prompt for a batch of (idx,case)) ──────────

def prompt_entity_pass(batch, focus, faithful=False):
    body = "\n".join(ctx_block(c, i, anchors=faithful, ambiguity=faithful)
                     for i, c in batch)
    return (f"Validate components in a document. {focus}\n\n{LAYERED_ENTITY_RULES}\n\n"
            "For each case, first quote the EXACT words from the sentence that state the "
            "architectural claim about the component (or \"none\"), then decide approve "
            "true/false based on that claim.\n\nCASES:\n" + body +
            '\n\nReturn JSON: {"validations":[{"case":1,"claim":"<quote or none>","approve":true}]}\nJSON only:')


def prompt_coref_pass(batch):
    body = "\n".join(ctx_block(c, i, antecedent=True) for i, c in batch)
    return (f"Validate components in a document. {COREF_VALIDATION_FOCUS}\n\n{LAYERED_COREF_RULES}\n\n"
            "For each case, first quote the EXACT words that state the architectural claim "
            "(or \"none\"), then decide approve true/false.\n\nCASES:\n" + body +
            '\n\nReturn JSON: {"validations":[{"case":1,"claim":"<quote or none>","approve":true}]}\nJSON only:')


SOFT_RULES = (
    "A link is VALID if the sentence states the component is used, provides or consumes a "
    "service, is implemented, contains or is contained, stores/routes data, or is described "
    "-- INCLUDING when it is named as a concrete example or instance (\"such as X\", \"e.g. "
    "X\", \"including X\"). A link is INVALID only if the component appears in an exclusion -- "
    "a negation (\"not X\", \"no X\") or a contrast/counter-example (\"other than X\", "
    "\"unlike X\", \"rather than X\") -- OR the token is used as a product/brand/system name "
    "rather than a reference to that architectural component, OR it appears only in a "
    "code/package path (x.y.z).")


def prompt_soft(batch):
    body = "\n".join(ctx_block(c, i) for i, c in batch)
    return ("You validate candidate trace links between a documentation sentence and a named "
            f"architecture component.\n\n{SOFT_RULES}\n\nFor each case, FIRST quote the exact "
            "words that assert (or exclude) the link, or \"none\", THEN decide approve "
            "true/false based only on that quote.\n\nCASES:\n" + body +
            '\n\nReturn JSON: {"validations":[{"case":1,"claim":"<quote or none>","approve":true}]}\nJSON only:')


RECOVER_RULES = (
    "Decide whether the SENTENCE conveys information about THIS specific component's role in "
    "the system -- i.e. whether a reader would record a trace link from the sentence to the "
    "component. Approve when the component participates in, is compared against, is excluded "
    "by, exemplifies, or is the referent of the architectural claim the sentence makes. This "
    "INCLUDES harder-but-valid cases:\n"
    "  - CONTRAST/NEGATION: \"systems other than X\", \"not provided by X, but by Y\" -- the "
    "sentence still asserts a fact ABOUT X's role, so approve X (and any Y).\n"
    "  - EXAMPLE/INSTANCE: \"a specific location (e.g. a file server or database)\" -- approve "
    "the component the example denotes.\n"
    "  - GENERIC/LOWERCASE: a common-noun form (\"the logic\", \"the datastore\") that, in "
    "this document, names this component -- approve when context makes the referent this "
    "component.\n"
    "  - ANAPHORA: a pronoun or role phrase (\"it\", \"the service\") whose antecedent is this "
    "component -- approve when the antecedent is unambiguous.\n"
    "Reject only when: the component is genuinely absent/unrelated, the match is a different "
    "entity, or the reference is purely a code/package path with no architectural claim. When "
    "the sentence plausibly concerns this component, prefer APPROVE (recall-oriented gate).")


def prompt_recover(batch):
    body = "\n".join(ctx_block(c, i, anchors=True, antecedent=True) for i, c in batch)
    return ("You are a recall-oriented trace-link judge. A documentation sentence is a "
            "candidate trace link to a named architecture component.\n\n" + RECOVER_RULES +
            "\n\nFor each case, FIRST quote the exact words that concern the component (or "
            "\"none\"), THEN decide approve true/false.\n\nCASES:\n" + body +
            '\n\nReturn JSON: {"validations":[{"case":1,"claim":"<quote or none>","approve":true}]}\nJSON only:')


# ── runner ───────────────────────────────────────────────────────────────────

def _client(temperature=0.1):
    """ALWAYS reasoning-off (never set OPENAI_REASONING_EFFORT). Only temperature varies,
    for self-consistency sampling. s21 is a no-reasoning config by design."""
    os.environ.pop("OPENAI_REASONING_EFFORT", None)
    os.environ["OPENAI_MODEL_NAME"] = MODEL
    return LLMClient(backend=LLMBackend.OPENAI, model=MODEL,
                     temperature=temperature, enable_logging=False)


def run_batches(cases, build_prompt, client, tag):
    """Return {case_id: bool} for a single pass over `cases`."""
    out = {}
    for k in range(0, len(cases), BATCH):
        sub = list(enumerate(cases[k:k + BATCH], start=1))  # (idx1based, case)
        prompt = build_prompt(sub)
        resp = client.query(prompt, timeout=180)
        verd = parse(resp.text if resp.success else "")
        for idx, c in sub:
            out[c["id"]] = bool(verd.get(idx, False))
        print(f"    {tag}: {min(k+BATCH,len(cases))}/{len(cases)}", file=sys.stderr)
    return out


def judge_J0(cache, key="J0_s21", faithful=False):
    """s21 replica, reasoning-off. faithful=True adds s21's evidence bundle
    (ambiguity flag + anchor sentences) to the entity pass, matching s21's actual
    in-run context."""
    ent = [c for c in CASES if mode(c) == "entity"]
    cor = [c for c in CASES if mode(c) == "coref"]
    cl = _client()
    p1 = run_batches(ent, lambda b: prompt_entity_pass(b, P1_FOCUS, faithful), cl, f"{key}.P1")
    p2 = run_batches(ent, lambda b: prompt_entity_pass(b, P2_FOCUS, faithful), cl, f"{key}.P2")
    cp = run_batches(cor, prompt_coref_pass, cl, f"{key}.coref")
    for c in ent:
        cache[f"{key}|{c['id']}"] = bool(p1.get(c["id"]) and p2.get(c["id"]))
    for c in cor:
        cache[f"{key}|{c['id']}"] = bool(cp.get(c["id"]))


def judge_single(cache, key, build_prompt):
    cl = _client()
    res = run_batches(CASES, build_prompt, cl, key)
    for c in CASES:
        cache[f"{key}|{c['id']}"] = bool(res.get(c["id"]))


def judge_J3(cache):
    """Self-consistency K=3 of the recall rubric, reasoning-OFF, temperature=1.0 for
    sampling variance. approve = majority (>=2); 'any' (>=1) recorded too."""
    cl = _client(temperature=1.0)
    samples = []
    for s in range(3):
        res = run_batches(CASES, prompt_recover, cl, f"J3.sample{s+1}")
        samples.append(res)
    for c in CASES:
        votes = [int(bool(samples[s].get(c["id"]))) for s in range(3)]
        cache[f"J3_vote|{c['id']}"] = {"votes": votes, "any": sum(votes) >= 1,
                                       "majority": sum(votes) >= 2}


def main():
    which = sys.argv[1:] or ["J0_s21", "J0_amb", "J1_soft", "J2_recover", "J3_vote"]
    cache = json.loads(VCACHE.read_text()) if VCACHE.exists() else {}
    if "J0_s21" in which:
        print("== J0_s21 (replica, bare context) ==", file=sys.stderr); judge_J0(cache, "J0_s21", faithful=False); VCACHE.write_text(json.dumps(cache, indent=1))
    if "J0_amb" in which:
        print("== J0_amb (replica + s21 evidence bundle: ambiguity+anchors) ==", file=sys.stderr); judge_J0(cache, "J0_amb", faithful=True); VCACHE.write_text(json.dumps(cache, indent=1))
    if "J1_soft" in which:
        print("== J1_soft ==", file=sys.stderr); judge_single(cache, "J1_soft", prompt_soft); VCACHE.write_text(json.dumps(cache, indent=1))
    if "J2_recover" in which:
        print("== J2_recover (recall rubric, reasoning-OFF) ==", file=sys.stderr); judge_single(cache, "J2_recover", prompt_recover); VCACHE.write_text(json.dumps(cache, indent=1))
    if "J3_vote" in which:
        print("== J3_vote (self-consistency K=3, reasoning-OFF, temp=1.0) ==", file=sys.stderr); judge_J3(cache); VCACHE.write_text(json.dumps(cache, indent=1))
    print("wrote", VCACHE, file=sys.stderr)


if __name__ == "__main__":
    main()
