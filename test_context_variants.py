#!/usr/bin/env python3
"""Test two design variants for handling generic partial references.

Variant 1: PHASE 3 FILTER — skip single-word generic partials entirely.
  Links matched only by a single generic word get NO syn-safe protection.
  They fall through to the normal judge (4-rule or advocate-prosecutor).

Variant 2: TWO-STAGE JUDGE — decomposed reasoning.
  Stage 1: "Is the word used as a NAME or generically?" (no architecture reasoning)
  Stage 2: Only if Stage 1 says NAME, run the normal 3-step evaluation.
  Key: Stage 1 is ISOLATED — it must not consider whether the sentence is
  architecturally relevant, only how the word is used linguistically.

Both variants tested on BBB syn-review links from V38 checkpoint.
"""
import csv
import glob
import os
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.stdout.reconfigure(line_buffering=True)

from llm_sad_sam.llm_client import LLMClient, LLMBackend

os.environ.setdefault("CLAUDE_MODEL", "sonnet")

BENCHMARK = Path("/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark")


def load_gold(dataset):
    files = [f for f in glob.glob(str(BENCHMARK / dataset / "**" / "goldstandard_sad_*-sam_*.csv"), recursive=True)
             if "UME" not in f and "code" not in f]
    gold = set()
    for f in files:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                sid, cid = row.get("sentence", ""), row.get("modelElementID", "")
                if sid and cid:
                    gold.add((int(sid), cid))
    return gold


def build_ctx(dk, comp_name):
    parts = []
    for syn, target in dk.synonyms.items():
        if target == comp_name:
            parts.append(f'"{syn}" is a known synonym for {comp_name}')
    for partial, target in dk.partial_references.items():
        if target == comp_name:
            parts.append(f'"{partial}" is a known short name for {comp_name}')
    return "; ".join(parts) if parts else ""


def find_alias(dk, comp_name, sent_text):
    text_lower = sent_text.lower()
    for partial, target in dk.partial_references.items():
        if target == comp_name:
            if re.search(rf'\b{re.escape(partial.lower())}\b', text_lower):
                return partial, "partial"
    for syn, target in dk.synonyms.items():
        if target == comp_name:
            if re.search(rf'\b{re.escape(syn.lower())}\b', text_lower):
                return syn, "synonym"
    return None, None


def is_single_word_generic(alias, alias_type):
    """True if the alias is a single common word (not CamelCase, not multi-word)."""
    if alias_type != "partial" or not alias:
        return False
    if ' ' in alias:
        return False
    if re.search(r'[a-z][A-Z]', alias):
        return False
    return True


# ─── Variant 1: Phase 3 filter ───────────────────────────────────────

def variant1_decision(alias, alias_type):
    """Variant 1: single-word generic partials get NO syn-safe protection.
    Returns: 'safe' (auto-approve) or 'no_protection' (falls to normal judge)."""
    if is_single_word_generic(alias, alias_type):
        return "no_protection"
    return "safe"


# ─── Variant 2: Two-stage judge ──────────────────────────────────────

def variant2_stage1(llm, snum, comp_name, sent_text, alias):
    """Stage 1: Is the word used as a NAME or generically?
    ISOLATED — no architecture reasoning allowed."""
    prompt = f"""LINGUISTIC ANALYSIS: How is the word "{alias}" used in this sentence?

SENTENCE: {sent_text}

CONTEXT: In this document, "{alias}" can be a short name for an architecture component
called "{comp_name}".

QUESTION: Is the author using "{alias}" in this sentence as a NAME for that specific
component, or is the author using it as an ordinary English word?

NAMING USE: The word identifies or refers to a specific named entity in the system.
  Examples: "the {alias} connects to...", "HTML5 {alias}", "{comp_name} {alias.lower()}"

GENERIC USE: The word describes a general concept, activity, or type — it could appear
  in any technical document regardless of the system's component names.
  Examples: "{alias.lower()} process", "CPU {alias.lower()}", "SVG {alias.lower()}"

Answer ONLY based on how the word is used linguistically. Do NOT consider whether
the sentence is architecturally relevant — that is a separate question.

Return JSON: {{"usage": "naming" or "generic", "reason": "brief explanation"}}
JSON only:"""
    data = llm.extract_json(llm.query(prompt, timeout=60))
    if data:
        return data.get("usage", "naming") == "naming", data.get("reason", "")
    return True, "parse failure"


def variant2_stage2(llm, snum, comp_name, sent_text, comp_names_str, context_str):
    """Stage 2: Standard 3-step evaluation (only reached if Stage 1 says 'naming')."""
    prompt = f"""JUDGE: Should sentence S{snum} be linked to component "{comp_name}"?

DOCUMENT ANALYSIS CONTEXT:
{context_str}
These mappings were confirmed by prior analysis of this document's terminology.

SENTENCE: {sent_text}
ALL COMPONENTS: {comp_names_str}

IMPORTANT: A synonym/short name appearing in the sentence is NECESSARY but NOT SUFFICIENT.

Step 1 — REFERENCE CHECK: Does the component name or a confirmed synonym/short name appear
in the sentence as a reference to the component? REJECT if:
  - It is a hyphenated modifier with no standalone usage (e.g., "client-side communication")
  - It is part of a different proper name not in the synonym list
  - It is only in a transitional/navigational phrase (e.g., "below the X diagram")
  NOTE: "the client", "the server", "on the client side" DO count as references when the
  word is used as a noun referring to a system participant, even in a prepositional phrase.

Step 2 — ARCHITECTURAL RELEVANCE: Does the sentence describe what this component does,
how it interacts with other components, or its role in the system? A sentence that merely
mentions the component as a location or recipient without describing its behavior is still
relevant if the interaction itself is architectural.

Step 3 — NOT PURELY INCIDENTAL: The component should be a meaningful participant in what
the sentence describes, not just a passing mention in a list or aside.

Return JSON: {{"approve": true/false, "reason": "brief explanation"}}
JSON only:"""
    data = llm.extract_json(llm.query(prompt, timeout=60))
    if data:
        return data.get("approve", True), data.get("reason", "")
    return True, "parse failure"


def main():
    from llm_sad_sam.core.document_loader import DocumentLoader

    with open("results/phase_cache/v38/bigbluebutton/pre_judge.pkl", "rb") as f:
        pj = pickle.load(f)
    with open("results/phase_cache/v38/bigbluebutton/phase3.pkl", "rb") as f:
        p3 = pickle.load(f)

    dk = p3['doc_knowledge']
    prelim = pj['preliminary']
    gold = load_gold("bigbluebutton")

    text_files = glob.glob(str(BENCHMARK / "bigbluebutton" / "**" / "bigbluebutton.txt"), recursive=True)
    loader = DocumentLoader()
    sentences = loader.load_sentences(text_files[0])
    sent_map = {s.number: s for s in sentences}
    comp_names = sorted(set(l.component_name for l in prelim))
    comp_names_str = ', '.join(comp_names)

    # Find ALL syn-review links
    syn_review = []
    for l in prelim:
        sent = sent_map.get(l.sentence_number)
        if not sent:
            continue
        alias, alias_type = find_alias(dk, l.component_name, sent.text)
        if alias:
            is_tp = (l.sentence_number, l.component_id) in gold
            is_generic = is_single_word_generic(alias, alias_type)
            syn_review.append((l, alias, alias_type, is_generic, is_tp))

    print(f"BBB syn-review: {len(syn_review)} links")
    tp_count = sum(1 for x in syn_review if x[4])
    fp_count = sum(1 for x in syn_review if not x[4])
    generic_count = sum(1 for x in syn_review if x[3])
    print(f"  TPs: {tp_count}, FPs: {fp_count}, Generic-partial: {generic_count}")

    # ═══════════════════════════════════════════════════════════════════
    # VARIANT 1: Phase 3 filter
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"VARIANT 1: PHASE 3 FILTER (no syn-safe for single-word generic partials)")
    print(f"{'='*80}")

    v1 = {"tp_safe": 0, "tp_noprot": 0, "fp_safe": 0, "fp_noprot": 0}
    for l, alias, alias_type, is_generic, is_tp in sorted(syn_review, key=lambda x: x[0].sentence_number):
        decision = variant1_decision(alias, alias_type)
        label = "TP" if is_tp else "FP"
        tag = " [GENERIC→no_protection]" if decision == "no_protection" else " [safe]"

        if is_tp:
            if decision == "safe": v1["tp_safe"] += 1
            else: v1["tp_noprot"] += 1
        else:
            if decision == "safe": v1["fp_safe"] += 1
            else: v1["fp_noprot"] += 1

        print(f"  [{label}] S{l.sentence_number} -> {l.component_name} alias=\"{alias}\"{tag}")

    print(f"\nVariant 1 Results:")
    print(f"  TPs safe (auto-approved):     {v1['tp_safe']}")
    print(f"  TPs no-protection (→ judge):  {v1['tp_noprot']}  ← these go to normal judge, may survive")
    print(f"  FPs safe (auto-approved):     {v1['fp_safe']}  ← leaked FPs")
    print(f"  FPs no-protection (→ judge):  {v1['fp_noprot']}  ← judge may catch these")
    print(f"  NET: {v1['fp_noprot']} FPs sent to judge (currently {v1['fp_safe']} would leak)")

    # ═══════════════════════════════════════════════════════════════════
    # VARIANT 2: Two-stage judge
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"VARIANT 2: TWO-STAGE JUDGE (Stage 1: naming vs generic, Stage 2: architecture)")
    print(f"{'='*80}")

    llm = LLMClient(backend=LLMBackend.CLAUDE)
    v2 = {"tp_kept": 0, "tp_killed": 0, "fp_kept": 0, "fp_killed": 0}

    # Only test generic-partial links (the ones the guide targets)
    generic_links = [(l, alias, is_tp) for l, alias, alias_type, is_generic, is_tp in syn_review if is_generic]
    non_generic = [(l, alias, is_tp) for l, alias, alias_type, is_generic, is_tp in syn_review if not is_generic]

    print(f"\n  Non-generic syn-review: {len(non_generic)} links (auto-approved, same as V32)")
    for l, alias, is_tp in non_generic:
        label = "TP" if is_tp else "FP"
        if is_tp: v2["tp_kept"] += 1
        else: v2["fp_kept"] += 1
        print(f"    [{label}] S{l.sentence_number} -> {l.component_name} alias=\"{alias}\" → SAFE")

    print(f"\n  Generic-partial links: {len(generic_links)} (two-stage judge)")
    for l, alias, is_tp in sorted(generic_links, key=lambda x: x[0].sentence_number):
        sent = sent_map.get(l.sentence_number)
        label = "TP" if is_tp else "FP"
        ctx_str = build_ctx(dk, l.component_name)

        # Stage 1: naming vs generic (single pass)
        is_naming, s1_reason = variant2_stage1(llm, l.sentence_number, l.component_name, sent.text, alias)

        if not is_naming:
            # Stage 1 says generic → REJECT
            if is_tp: v2["tp_killed"] += 1
            else: v2["fp_killed"] += 1
            print(f"    [{label}] S{l.sentence_number} -> {l.component_name} alias=\"{alias}\"")
            print(f"      Stage 1: GENERIC → REJECT. {s1_reason}")
            continue

        # Stage 1 says naming → proceed to Stage 2
        approve, s2_reason = variant2_stage2(llm, l.sentence_number, l.component_name,
                                              sent.text, comp_names_str, ctx_str)

        if approve:
            if is_tp: v2["tp_kept"] += 1
            else: v2["fp_kept"] += 1
        else:
            if is_tp: v2["tp_killed"] += 1
            else: v2["fp_killed"] += 1

        status = "APPROVE" if approve else "REJECT"
        correct = "correct" if (is_tp == approve) else "WRONG"
        print(f"    [{label}] S{l.sentence_number} -> {l.component_name} alias=\"{alias}\"")
        print(f"      Stage 1: NAMING. {s1_reason}")
        print(f"      Stage 2: {status} ({correct}). {s2_reason}")

    print(f"\n{'='*80}")
    print(f"VARIANT 2 RESULTS")
    print(f"{'='*80}")
    print(f"  TPs kept:   {v2['tp_kept']}")
    print(f"  TPs killed: {v2['tp_killed']}  ← must be 0")
    print(f"  FPs killed: {v2['fp_killed']}  ← want max")
    print(f"  FPs kept:   {v2['fp_kept']}")
    accuracy = v2['tp_kept'] + v2['fp_killed']
    total = v2['tp_kept'] + v2['tp_killed'] + v2['fp_kept'] + v2['fp_killed']
    print(f"  Accuracy:   {accuracy}/{total}")


if __name__ == "__main__":
    main()
