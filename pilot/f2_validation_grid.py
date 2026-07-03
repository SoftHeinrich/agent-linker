#!/usr/bin/env python3
"""F2 validation grid over cached typed extraction proposals.

This script does not run new extraction calls. It consumes the live caches from
`s21_prompt_router_live.py`, applies cheap structural filters and cached judge
verdicts, and ranks policies by macro F2.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import s21_prompt_router_live as P
from llm_sad_sam.linkers.experimental.s_linker21 import (
    COREF_VALIDATION_FOCUS,
    LAYERED_COREF_RULES,
)


VARIANT = "typed_all_filter_named"
OUT = P.CACHE_DIR / "f2_validation_grid_summary.json"
EXTRA_JUDGE_CACHE = P.CACHE_DIR / "f2_extra_mode_judge_cache.json"


def load_proposals() -> dict[str, list[dict]]:
    cache = json.loads((P.CACHE_DIR / f"{VARIANT}_extract_cache.json").read_text())
    out: dict[str, list[dict]] = {}
    for project in P.PROJECTS:
        names_to_ids = P.roster(project)
        sents = P.sentences(project)
        refs = []
        for key, raw in cache.items():
            parts = key.split("|")
            if parts[0] == project:
                refs.extend(P.normalize_refs(raw, names_to_ids, set(sents)))
        out[project] = refs
    return out


def case_for(project: str, ref: dict) -> dict:
    return P.build_case(project, ref, P.sentences(project), P.ambiguous_names(project))


def judge_keep(case: dict, cache: dict) -> bool:
    if case["mode"] == "AFFIRMATIVE":
        return bool(cache.get(f"p1|{case['id']}") and cache.get(f"p2|{case['id']}"))
    if case["mode"] == "CONTRAST":
        return bool(cache.get(f"contrast|{case['id']}"))
    return False


CODE_HINT = re.compile(
    r"(^|[^A-Za-z0-9])([a-z][a-z0-9_]*(?:\.[a-zA-Z_][A-Za-z0-9_]*)+|[A-Za-z_][A-Za-z0-9_]*(?:Servlet|Factory|Action|Controller|Util|Test|Tests|Socket|Layer)|[A-Za-z0-9]+\s+tests?)($|[^A-Za-z0-9])",
    re.IGNORECASE,
)
LOWER_GENERIC = {
    "client", "server", "service", "component", "logic", "storage", "preferences",
    "slope one", "nearest-neighbor approach", "order-based nearest-neighbor approach",
}


def quote_has_component(ref: dict) -> bool:
    quote = ref.get("quote", "")
    component = ref.get("component", "")
    return bool(quote and component and re.search(
        rf"(?<![A-Za-z0-9]){re.escape(component)}(?![A-Za-z0-9])",
        quote,
        re.IGNORECASE,
    ))


def quote_in_sentence(ref: dict, sent: str) -> bool:
    quote = ref.get("quote", "").strip()
    return bool(quote and quote.lower() in sent.lower())


def terminal_quote(ref: dict) -> bool:
    quote = ref.get("quote", "").strip().lower()
    quote = re.sub(r"^(the|a|an)\s+", "", quote)
    tokens = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", ref.get("component", ""))
    if not tokens:
        return False
    terminal = tokens[-1].lower()
    return quote == terminal or quote.endswith(" " + terminal)


def sentence_has_component(ref: dict, sent: str) -> bool:
    component = ref.get("component", "")
    return bool(component and re.search(
        rf"(?<![A-Za-z0-9]){re.escape(component)}(?![A-Za-z0-9])",
        sent,
        re.IGNORECASE,
    ))


def is_code_like(ref: dict, sent: str) -> bool:
    quote = ref.get("quote", "")
    return bool(CODE_HINT.search(quote) or CODE_HINT.search(sent[:120]))


def is_lower_generic_quote(ref: dict) -> bool:
    quote = ref.get("quote", "").strip().lower()
    return quote in LOWER_GENERIC or (quote.startswith("the ") and quote[4:] in LOWER_GENERIC)


def has_anchor(ref: dict, project: str) -> bool:
    return bool(P.anchors(project, ref["sentence"], ref["component"], P.sentences(project)))


def policy_modes(ref: dict, project: str, base_union: set[tuple[int, str]]) -> bool:
    return ref["mode"] in {"AFFIRMATIVE", "CONTRAST"}


def policy_no_code(ref: dict, project: str, base_union: set[tuple[int, str]]) -> bool:
    if ref["mode"] not in {"AFFIRMATIVE", "CONTRAST"}:
        return False
    return not is_code_like(ref, P.sentences(project).get(ref["sentence"], ""))


def policy_named_quote(ref: dict, project: str, base_union: set[tuple[int, str]]) -> bool:
    if ref["mode"] == "CONTRAST":
        return True
    if ref["mode"] != "AFFIRMATIVE":
        return False
    sent = P.sentences(project).get(ref["sentence"], "")
    return quote_has_component(ref) or sentence_has_component(ref, sent)


def policy_named_quote_no_code(ref: dict, project: str, base_union: set[tuple[int, str]]) -> bool:
    return policy_named_quote(ref, project, base_union) and policy_no_code(ref, project, base_union)


def policy_drop_lower_generic(ref: dict, project: str, base_union: set[tuple[int, str]]) -> bool:
    if not policy_named_quote_no_code(ref, project, base_union):
        return False
    if ref["mode"] == "AFFIRMATIVE" and is_lower_generic_quote(ref):
        return False
    return True


def policy_quote_present_no_code(ref: dict, project: str, base_union: set[tuple[int, str]]) -> bool:
    if not policy_no_code(ref, project, base_union):
        return False
    return quote_in_sentence(ref, P.sentences(project).get(ref["sentence"], ""))


def policy_exact_or_terminal_no_code(ref: dict, project: str, base_union: set[tuple[int, str]]) -> bool:
    if ref["mode"] == "CONTRAST":
        return policy_no_code(ref, project, base_union)
    if ref["mode"] != "AFFIRMATIVE":
        return False
    sent = P.sentences(project).get(ref["sentence"], "")
    if not policy_no_code(ref, project, base_union) or not quote_in_sentence(ref, sent):
        return False
    return quote_has_component(ref) or sentence_has_component(ref, sent) or terminal_quote(ref)


def policy_contrast_plus_subjectish(ref: dict, project: str, base_union: set[tuple[int, str]]) -> bool:
    if ref["mode"] == "CONTRAST":
        return True
    if ref["mode"] != "AFFIRMATIVE":
        return False
    sent = P.sentences(project).get(ref["sentence"], "")
    component = re.escape(ref["component"])
    subjectish = re.search(rf"^\s*(The\s+)?{component}\b|[.;]\s*(The\s+)?{component}\b", sent, re.I)
    return bool(subjectish) and not is_code_like(ref, sent)


def policy_f2_implicit_anchor(ref: dict, project: str, base_union: set[tuple[int, str]]) -> bool:
    if policy_named_quote_no_code(ref, project, base_union):
        return True
    if ref["mode"] != "IMPLICIT":
        return False
    if is_code_like(ref, P.sentences(project).get(ref["sentence"], "")):
        return False
    # F2-biased: let anchored implicit proposals through to the same strict gate if
    # the proposer did not also emit many siblings for the sentence.
    return has_anchor(ref, project)


def policy_context_modes_no_code(ref: dict, project: str, base_union: set[tuple[int, str]]) -> bool:
    if policy_no_code(ref, project, base_union):
        return True
    if ref["mode"] in {"IMPLICIT", "ANAPHORA"}:
        return has_anchor(ref, project) and not is_code_like(ref, P.sentences(project).get(ref["sentence"], ""))
    return False


POLICIES = {
    "current_modes": policy_modes,
    "no_code": policy_no_code,
    "named_quote": policy_named_quote,
    "named_quote_no_code": policy_named_quote_no_code,
    "drop_lower_generic": policy_drop_lower_generic,
    "quote_present_no_code": policy_quote_present_no_code,
    "exact_or_terminal_no_code": policy_exact_or_terminal_no_code,
    "contrast_plus_subjectish": policy_contrast_plus_subjectish,
    "f2_implicit_anchor": policy_f2_implicit_anchor,
    "context_modes_no_code": policy_context_modes_no_code,
}


def context_prompt(cases: list[tuple[int, dict]]) -> str:
    body = "\n".join(P.case_block(case, i, anchors_on=True, ambiguity_on=True)
                     for i, case in cases)
    return """Validate trace links where the component is referenced WITHOUT its proper name.

The component is referenced by a generic common noun, role phrase, concrete
example, or local phrase. Use anchor sentences where the component IS named,
plus local context, to decide the referent. Approve only when the context makes
it clear the phrase denotes THIS specific component and the sentence makes an
architectural claim about it. Reject ordinary vocabulary, ambiguous siblings,
and code/package paths.

For each case, FIRST quote the referring words and the anchor that fixes the
referent (or "none"), THEN decide approve true/false.

CASES:
""" + body + """

Return JSON:
{"validations":[{"case":1,"claim":"<quote or none>","approve":true}]}
JSON only:"""


def anaphora_prompt(cases: list[tuple[int, dict]]) -> str:
    body = "\n".join(P.case_block(case, i, anchors_on=True, ambiguity_on=True)
                     for i, case in cases)
    return f"""Validate components in a document. {COREF_VALIDATION_FOCUS}

{LAYERED_COREF_RULES}

The candidate reference is an anaphoric phrase or pronoun. Use the previous
sentence and anchors only to resolve the referent. For each case, first quote
the exact referring words and architectural claim (or "none"), then decide
approve true/false.

CASES:
{body}

Return JSON:
{{"validations":[{{"case":1,"claim":"<quote or none>","approve":true}}]}}
JSON only:"""


def ensure_extra_judges(cases: list[dict]) -> dict:
    cache = json.loads(EXTRA_JUDGE_CACHE.read_text()) if EXTRA_JUDGE_CACHE.exists() else {}
    cl = P.client()
    for mode, prompt_builder in (("IMPLICIT", context_prompt), ("ANAPHORA", anaphora_prompt)):
        need = [case for case in cases if case["mode"] == mode and f"{mode}|{case['id']}" not in cache]
        for start in range(0, len(need), 8):
            batch = need[start:start + 8]
            indexed = [(i, case) for i, case in enumerate(batch, 1)]
            resp = cl.query(prompt_builder(indexed), timeout=180)
            parsed = P.parse_validations(resp.text if resp.success else "")
            for i, case in enumerate(batch, 1):
                cache[f"{mode}|{case['id']}"] = bool(parsed.get(i, False))
            EXTRA_JUDGE_CACHE.write_text(json.dumps(cache, indent=1))
            print(f"  extra judge {mode}: {min(start + 8, len(need))}/{len(need)}")
    return cache


def evaluate_policy(name: str, proposals: dict[str, list[dict]], jcache: dict) -> dict:
    variant_name = f"grid_{name}"
    P.VARIANTS.setdefault(variant_name, {
        "description": f"F2 validation grid policy: {name}",
        "builder": None,
        "keep_modes": {"AFFIRMATIVE", "CONTRAST"},
    })
    golds = {project: P.gold(project) for project in P.PROJECTS}
    base_union = {
        project: set().union(*(P.s21_final(project, run) for run in P.RUNS))
        for project in P.PROJECTS
    }
    kept_new: dict[str, set[tuple[int, str]]] = defaultdict(set)
    kept_details = []
    proposed_by_sentence = {
        project: Counter(ref["sentence"] for ref in refs)
        for project, refs in proposals.items()
    }

    eligible_extra_cases = []
    for project, refs in proposals.items():
        base = base_union[project]
        seen = set()
        for ref in refs:
            key = (ref["sentence"], ref["component_id"])
            if key in base or key in seen:
                continue
            if not POLICIES[name](ref, project, base):
                continue
            seen.add(key)
            if ref["mode"] in {"IMPLICIT", "ANAPHORA"}:
                eligible_extra_cases.append(case_for(project, ref))
    extra_cache = ensure_extra_judges(eligible_extra_cases) if eligible_extra_cases else {}

    for project, refs in proposals.items():
        seen = set()
        for ref in refs:
            key = (ref["sentence"], ref["component_id"])
            if key in base_union[project] or key in seen:
                continue
            if not POLICIES[name](ref, project, base_union[project]):
                continue
            seen.add(key)
            case = case_for(project, ref)
            if ref["mode"] in {"AFFIRMATIVE", "CONTRAST"}:
                ok = judge_keep(case, jcache)
            elif ref["mode"] in {"IMPLICIT", "ANAPHORA"}:
                ok = bool(extra_cache.get(f"{ref['mode']}|{case['id']}"))
            else:
                ok = False
            if not ok:
                continue
            kept_new[project].add(key)
            kept_details.append({
                "project": project,
                "sentence": ref["sentence"],
                "component": ref["component"],
                "component_id": ref["component_id"],
                "mode": ref["mode"],
                "quote": ref.get("quote", ""),
                "gold": key in golds[project],
                "sentence_text": P.sentences(project).get(ref["sentence"], ""),
                "proposal_count_same_sentence": proposed_by_sentence[project][ref["sentence"]],
            })

    summary = P.score_variant(variant_name, proposals, kept_new)
    summary["policy"] = name
    summary["kept_new_links"] = kept_details
    summary["fp_categories"] = categorize_fps(kept_details)
    return summary


def categorize_fps(links: list[dict]) -> dict[str, int]:
    counts = Counter()
    for link in links:
        if link["gold"]:
            continue
        sent = link["sentence_text"]
        quote = link.get("quote", "")
        if is_code_like(link, sent):
            counts["code_like"] += 1
        elif is_lower_generic_quote(link):
            counts["lower_generic"] += 1
        elif link["proposal_count_same_sentence"] > 3:
            counts["crowded_sentence"] += 1
        else:
            counts["other"] += 1
    return dict(counts)


def main() -> None:
    proposals = load_proposals()
    jcache = json.loads((P.CACHE_DIR / f"{VARIANT}_judge_cache.json").read_text())
    rows = [evaluate_policy(name, proposals, jcache) for name in POLICIES]
    rows.sort(key=lambda r: (r["augmented"]["F2"], r["augmented"]["F1"]), reverse=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"{'policy':<24}{'P':>8}{'R':>8}{'F1':>8}{'F2':>8}{'dF2':>8}{'TP':>5}{'FP':>5}")
    for row in rows:
        a, d = row["augmented"], row["delta"]
        print(f"{row['policy']:<24}{a['P']:>8.4f}{a['R']:>8.4f}{a['F1']:>8.4f}"
              f"{a['F2']:>8.4f}{d['F2']:>+8.4f}{row['new_TP']:>5}{row['new_FP']:>5}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
