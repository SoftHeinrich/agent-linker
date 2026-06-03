#!/usr/bin/env python3
"""Quick unit test: does the semantic guide help the context judge reject generic-partial FPs?

Loads V38 BBB checkpoint, runs the context judge on just the 7 FP syn-review links
to see if the guide catches them without killing TPs.
"""
import csv
import glob
import os
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

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


def context_judge_pass(llm, snum, comp_name, sent_text, comp_names_str, context_str, alias, is_generic_partial):
    """Semantic guide version of context judge."""
    generic_guide = ""
    if is_generic_partial and alias:
        generic_guide = f"""
CRITICAL — NAMING vs GENERIC USE:
The matching short name "{alias}" is a common English word. Apply the SUBSTITUTION
TEST before anything else:

  Replace every occurrence of "{alias}" in the sentence with the full component name
  "{comp_name}". Read the result aloud. Does the sentence STILL MAKE SENSE and
  MEAN THE SAME THING?

  YES → the author is referring to the component. Proceed to Steps 1-3.
  NO  → the author is using the word generically. REJECT immediately.

This test is DECISIVE. Do not override it. A sentence can describe topics closely
related to what a component does without actually referring to that component by name.
"""

    prompt = f"""JUDGE: Should sentence S{snum} be linked to component "{comp_name}"?

DOCUMENT ANALYSIS CONTEXT:
{context_str}
These mappings were confirmed by prior analysis of this document's terminology.

SENTENCE: {sent_text}
ALL COMPONENTS: {comp_names_str}

IMPORTANT: A synonym/short name appearing in the sentence is NECESSARY but NOT SUFFICIENT.
{generic_guide}
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

    # Find syn-review links (same logic as V38 _judge_review)
    syn_review = []
    for l in prelim:
        sent = sent_map.get(l.sentence_number)
        if not sent:
            continue
        text_lower = sent.text.lower()
        alias = None
        alias_type = None
        for partial, target in dk.partial_references.items():
            if target == l.component_name:
                if re.search(rf'\b{re.escape(partial.lower())}\b', text_lower):
                    alias, alias_type = partial, "partial"
                    break
        if not alias:
            for syn, target in dk.synonyms.items():
                if target == l.component_name:
                    if re.search(rf'\b{re.escape(syn.lower())}\b', text_lower):
                        alias, alias_type = syn, "synonym"
                        break
        if alias:
            is_tp = (l.sentence_number, l.component_id) in gold
            is_generic = (alias_type == "partial" and ' ' not in alias
                          and not re.search(r'[a-z][A-Z]', alias))
            syn_review.append((l, alias, alias_type, is_generic, is_tp))

    print(f"BBB syn-review: {len(syn_review)} links")
    tp_count = sum(1 for x in syn_review if x[4])
    fp_count = sum(1 for x in syn_review if not x[4])
    print(f"  TPs: {tp_count}, FPs: {fp_count}")

    # Build context strings
    def build_ctx(comp_name):
        parts = []
        for syn, target in dk.synonyms.items():
            if target == comp_name:
                parts.append(f'"{syn}" is a known synonym for {comp_name}')
        for partial, target in dk.partial_references.items():
            if target == comp_name:
                parts.append(f'"{partial}" is a known short name for {comp_name}')
        return "; ".join(parts) if parts else ""

    llm = LLMClient(backend=LLMBackend.CLAUDE)

    print(f"\n{'='*80}")
    print(f"CONTEXT JUDGE WITH SEMANTIC GUIDE (single pass, no union)")
    print(f"{'='*80}")

    results = {"tp_kept": 0, "tp_killed": 0, "fp_kept": 0, "fp_killed": 0}

    for l, alias, alias_type, is_generic, is_tp in sorted(syn_review, key=lambda x: x[0].sentence_number):
        sent = sent_map.get(l.sentence_number)
        label = "TP" if is_tp else "FP"
        ctx_str = build_ctx(l.component_name)

        approve, reason = context_judge_pass(
            llm, l.sentence_number, l.component_name, sent.text,
            comp_names_str, ctx_str, alias, is_generic)

        if is_tp:
            if approve: results["tp_kept"] += 1
            else: results["tp_killed"] += 1
        else:
            if approve: results["fp_kept"] += 1
            else: results["fp_killed"] += 1

        status = "APPROVE" if approve else "REJECT"
        correct = "correct" if (is_tp == approve) else "WRONG"
        generic_tag = " [GENERIC]" if is_generic else ""
        print(f"  [{label}] S{l.sentence_number} -> {l.component_name} alias=\"{alias}\"{generic_tag}: {status} ({correct})")
        print(f"       {reason}")

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"  TPs kept:   {results['tp_kept']}")
    print(f"  TPs killed: {results['tp_killed']}  ← must be 0")
    print(f"  FPs killed: {results['fp_killed']}  ← want 7/7")
    print(f"  FPs kept:   {results['fp_kept']}")
    accuracy = results['tp_kept'] + results['fp_killed']
    print(f"  Accuracy:   {accuracy}/{len(syn_review)}")


if __name__ == "__main__":
    main()
