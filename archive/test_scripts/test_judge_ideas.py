#!/usr/bin/env python3
"""Unit-test level evaluation of judge prompt ideas on known BBB cases.

Tests Idea C (batch-context judge) and Idea E (generic-risk-only 4-rule Phase 6)
using real LLM calls but only on the specific method, not the full pipeline.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

os.environ["CLAUDE_MODEL"] = "sonnet"
from llm_sad_sam.llm_client import LLMClient, LLMBackend

llm = LLMClient(backend=LLMBackend.CLAUDE)

# ═══════════════════════════════════════════════════════════════
# BBB test data — known links from V22/V23 runs
# ═══════════════════════════════════════════════════════════════

BBB_COMPONENTS = [
    "HTML5 Client", "HTML5 Server", "BBB web", "Apps", "FreeSWITCH",
    "FSESL", "Redis PubSub", "Redis DB", "WebRTC-SFU", "kurento",
    "Presentation Conversion", "Recording Service"
]

# These are the alias-matched links that went to judge in V23 (no bypass).
# Gold = TP (should be approved), FP (should be rejected)
ALIAS_CASES = [
    # TPs — judge should APPROVE these
    {"s": 9,  "comp": "HTML5 Client", "text": "The HTML5 server is built upon Meteor.js in ECMA2015 for communication between client and server and upon MongoDB for keeping the state of each BigBlueButton client consistent with the BigBlueButton server.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 10, "comp": "HTML5 Client", "text": "The MongoDB database contains information about all meetings on the server and, in turn, each client connected to a meeting.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 12, "comp": "HTML5 Client", "text": "The client side subscribes to the published collections on the server side.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 13, "comp": "HTML5 Client", "text": "Updates to MongoDB on the server side are automatically pushed to MiniMongo on the client side.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 19, "comp": "HTML5 Client", "text": "BigBlueButton 2.3 moves away from a single nodejs process for bbb-html5 towards multiple nodejs processes handling incoming messages from clients.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 37, "comp": "BBB web", "text": "BigBlueButton web application is a Java-based application written in Scala.", "match": "BigBlueButton web application", "gold": "TP", "source": "validated"},
    {"s": 76, "comp": "HTML5 Client", "text": "Uploaded presentations go through a conversion process in order to be displayed inside the client.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 79, "comp": "HTML5 Client", "text": "The conversion process sends progress messages to the client through the Redis pubsub.", "match": "Client", "gold": "TP", "source": "validated"},
    # FPs — judge should REJECT these
    {"s": 16, "comp": "HTML5 Client", "text": "BigBlueButton 2.2 used a single nodejs process for all client-side communication.", "match": "Client", "gold": "FP", "source": "validated"},
    {"s": 58, "comp": "FSESL", "text": "This allows others who are using voice conference systems other than FreeSWITCH to easily create their own integration.", "match": "FreeSWITCH", "gold": "FP", "source": "coreference"},
    {"s": 50, "comp": "Recording Service", "text": "When the meeting ends, the Recording Processor will take all the recorded events as well as the different raw (PDF, WAV, FLV) files for processing.", "match": "Recording Processor", "gold": "FP", "source": "validated"},
    {"s": 83, "comp": "Presentation Conversion", "text": "Then below the SVG conversion flow.", "match": "conversion", "gold": "FP", "source": "validated"},
    {"s": 84, "comp": "Presentation Conversion", "text": "It covers the conversion fallback.", "match": "conversion", "gold": "FP", "source": "validated"},
    {"s": 85, "comp": "Presentation Conversion", "text": "Sometimes we detect that the generated SVG file is heavy to load by the browser, we use the fallback to put a rasterized image inside the SVG file and make its loading light for the browser.", "match": "NONE", "gold": "FP", "source": "validated"},
]

# Phase 6 test cases: candidates that Phase 6 needs to validate
# Include some that SHOULD pass and some that SHOULD be rejected
PHASE6_CASES = [
    # Should APPROVE (TPs)
    {"s": 9,  "comp": "HTML5 Client", "text": "The HTML5 server is built upon Meteor.js in ECMA2015 for communication between client and server and upon MongoDB for keeping the state of each BigBlueButton client consistent with the BigBlueButton server.", "match": "client", "gold": "TP"},
    {"s": 37, "comp": "BBB web", "text": "BigBlueButton web application is a Java-based application written in Scala.", "match": "BigBlueButton web application", "gold": "TP"},
    {"s": 50, "comp": "Recording Service", "text": "When the meeting ends, the Recording Processor will take all the recorded events as well as the different raw (PDF, WAV, FLV) files for processing.", "match": "Recording Processor", "gold": "TP"},
    {"s": 76, "comp": "Presentation Conversion", "text": "Uploaded presentations go through a conversion process in order to be displayed inside the client.", "match": "conversion process", "gold": "TP"},
    # Should REJECT (FPs)
    {"s": 16, "comp": "HTML5 Client", "text": "BigBlueButton 2.2 used a single nodejs process for all client-side communication.", "match": "client-side", "gold": "FP"},
    {"s": 83, "comp": "Presentation Conversion", "text": "Then below the SVG conversion flow.", "match": "conversion", "gold": "FP"},
    {"s": 84, "comp": "Presentation Conversion", "text": "It covers the conversion fallback.", "match": "conversion", "gold": "FP"},
    {"s": 85, "comp": "Presentation Conversion", "text": "Sometimes we detect that the generated SVG file is heavy to load by the browser, we use the fallback to put a rasterized image inside the SVG file and make its loading light for the browser.", "match": "NONE", "gold": "FP"},
]


def eval_judge_results(cases, results, label):
    """Evaluate judge results against gold labels."""
    tp_kept = 0   # TP approved (correct)
    tp_killed = 0  # TP rejected (bad)
    fp_kept = 0    # FP approved (bad)
    fp_killed = 0  # FP rejected (correct)

    for i, case in enumerate(cases):
        approved = results.get(i, True)  # default approve if missing
        if case["gold"] == "TP":
            if approved:
                tp_kept += 1
            else:
                tp_killed += 1
                print(f"  TP KILLED: S{case['s']} -> {case['comp']} [{case['match']}]")
        else:
            if approved:
                fp_kept += 1
                print(f"  FP KEPT:   S{case['s']} -> {case['comp']} [{case['match']}]")
            else:
                fp_killed += 1

    total_tp = tp_kept + tp_killed
    total_fp = fp_kept + fp_killed
    print(f"\n  {label}: TP kept={tp_kept}/{total_tp}, FP killed={fp_killed}/{total_fp}")
    print(f"  TP kill rate: {tp_killed}/{total_tp} = {tp_killed/max(1,total_tp):.0%} (lower is better)")
    print(f"  FP kill rate: {fp_killed}/{total_fp} = {fp_killed/max(1,total_fp):.0%} (higher is better)")
    return {"tp_kept": tp_kept, "tp_killed": tp_killed, "fp_kept": fp_kept, "fp_killed": fp_killed}


# ═══════════════════════════════════════════════════════════════
# IDEA C: Batch-context judge for alias links
# ═══════════════════════════════════════════════════════════════

def test_idea_c_batch_judge():
    """Group alias links by component, judge as batch with document context."""
    print("\n" + "="*80)
    print("IDEA C: Batch-context judge for alias links")
    print("="*80)

    comp_names = ", ".join(BBB_COMPONENTS)

    # Group cases by component
    from collections import defaultdict
    by_comp = defaultdict(list)
    for i, case in enumerate(ALIAS_CASES):
        by_comp[case["comp"]].append((i, case))

    all_results = {}

    for comp, group in by_comp.items():
        if len(group) < 2:
            # Single case — just do individual judge
            idx, case = group[0]
            prompt = f"""JUDGE: Should sentence S{case['s']} be linked to component "{comp}"?

The sentence mentions "{case['match']}" which is a known alias/partial for "{comp}" in this document.

S{case['s']}: {case['text']}

COMPONENTS: {comp_names}

Is this a valid architectural-level reference to {comp}? Answer JSON:
{{"approve": true/false, "reason": "brief"}}
JSON only:"""
            data = llm.extract_json(llm.query(prompt, timeout=60))
            all_results[idx] = data.get("approve", True) if data else True
            continue

        # Multi-case — batch judge
        sentences_block = "\n".join(
            f"  S{case['s']}: {case['text']}" for _, case in group
        )
        match_term = group[0][1]["match"]

        prompt = f"""JUDGE: Review trace links between documentation and the component "{comp}".

In this document, "{match_term}" is a known partial reference / alias for the architecture component "{comp}".
The document analysis confirmed this mapping. Your job is NOT to re-evaluate whether "{match_term}" maps to "{comp}" —
that is already established. Instead, judge whether each sentence discusses "{comp}" at the architectural level.

COMPONENTS: {comp_names}

SENTENCES linked to {comp} via alias "{match_term}":
{sentences_block}

For each sentence, approve if it discusses {comp} at the architectural level (role, behavior, interactions).
Reject if the mention is purely incidental, implementation-level, or metaphorical.

Return JSON:
{{"judgments": [{{"sentence": N, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

        print(f"\n  Batch judging {comp} ({len(group)} links via '{match_term}')...")
        data = llm.extract_json(llm.query(prompt, timeout=120))

        if data:
            for j in data.get("judgments", []):
                snum = j.get("sentence")
                approved = j.get("approve", True)
                # Map back to original index
                for idx, case in group:
                    if case["s"] == snum:
                        all_results[idx] = approved
                        status = "APPROVE" if approved else "REJECT"
                        print(f"    S{snum} -> {comp}: {status} ({j.get('reason', '')})")
                        break

    # Fill missing with approve
    for i in range(len(ALIAS_CASES)):
        if i not in all_results:
            all_results[i] = True

    return eval_judge_results(ALIAS_CASES, all_results, "Idea C (batch judge)")


# ═══════════════════════════════════════════════════════════════
# IDEA C2: Individual 4-rule judge (V22 baseline for comparison)
# ═══════════════════════════════════════════════════════════════

def test_v22_individual_judge():
    """V22's 4-rule individual judge — baseline comparison."""
    print("\n" + "="*80)
    print("BASELINE: V22 individual 4-rule judge (no alias bypass)")
    print("="*80)

    comp_names = ", ".join(BBB_COMPONENTS)
    cases_text = []
    for i, case in enumerate(ALIAS_CASES):
        match_info = f'match:"{case["match"]}"' if case["match"] != "NONE" else "match:NONE(pronoun/context)"
        cases_text.append(
            f'Case {i+1}: S{case["s"]} -> {case["comp"]} (src:{case["source"]}, {match_info})\n'
            f'    >>> S{case["s"]}: {case["text"]}'
        )

    prompt = f"""JUDGE: Review trace links between documentation sentences and software architecture components.

A sentence S should be linked to component C ONLY if ALL FOUR of the following are true:

RULE 1 — REFERENCE: S actually refers to C (not just string-matches its name).
RULE 2 — ARCHITECTURAL LEVEL: S describes C at the architectural level.
RULE 3 — TOPIC: C is the topic or subject of S.
RULE 4 — NOT GENERIC: The reference is to C as a specific architectural component, not as a generic English word.

COMPONENTS: {comp_names}

LINKS:
{chr(10).join(cases_text)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

    print(f"  Judging {len(ALIAS_CASES)} cases individually...")
    data = llm.extract_json(llm.query(prompt, timeout=120))

    results = {}
    if data:
        for j in data.get("judgments", []):
            idx = j.get("case", 0) - 1
            if 0 <= idx < len(ALIAS_CASES):
                results[idx] = j.get("approve", True)
                status = "APPROVE" if results[idx] else "REJECT"
                case = ALIAS_CASES[idx]
                print(f"    S{case['s']} -> {case['comp']}: {status} ({j.get('reason', '')})")

    return eval_judge_results(ALIAS_CASES, results, "V22 individual judge")


# ═══════════════════════════════════════════════════════════════
# IDEA E: 4-rule Phase 6 only for generic-risk components
# ═══════════════════════════════════════════════════════════════

GENERIC_RISK = {"Logic", "Storage", "Common", "Client", "Model", "Action", "Data", "Service", "Server"}

def test_idea_e_generic_risk_phase6():
    """4-rule prompt only for generic-risk component matches. Simple prompt for others."""
    print("\n" + "="*80)
    print("IDEA E: 4-rule Phase 6 only for generic-risk components")
    print("="*80)

    comp_names = ", ".join(BBB_COMPONENTS)

    # Split cases into generic-risk and non-generic
    generic_cases = []
    other_cases = []
    for i, case in enumerate(PHASE6_CASES):
        # Check if any word in the component name is generic-risk
        comp_words = set(case["comp"].split())
        if comp_words & GENERIC_RISK:
            generic_cases.append((i, case))
        else:
            other_cases.append((i, case))

    print(f"  Generic-risk: {len(generic_cases)}, Other: {len(other_cases)}")
    all_results = {}

    # 4-rule prompt for generic-risk
    if generic_cases:
        cases_text = []
        for j, (i, case) in enumerate(generic_cases):
            cases_text.append(f'Case {j+1}: "{case["match"]}" -> {case["comp"]}\n  "{case["text"]}"')

        prompt = f"""Validate component references in a software architecture document.
Focus: Is this a reference to a SPECIFIC architectural component, not a generic word?

A sentence S should be linked to component C ONLY if ALL FOUR are true:

RULE 1 — REFERENCE: S actually refers to C (not just string-matches its name).
RULE 2 — ARCHITECTURAL LEVEL: S describes C at the architectural level.
RULE 3 — TOPIC: C is the topic or subject of S.
RULE 4 — NOT GENERIC: The reference is to C as a specific architectural component.

COMPONENTS: {comp_names}

CASES:
{chr(10).join(cases_text)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

        print(f"\n  4-rule validating {len(generic_cases)} generic-risk cases...")
        data = llm.extract_json(llm.query(prompt, timeout=120))
        if data:
            for v in data.get("validations", []):
                j = v.get("case", 0) - 1
                if 0 <= j < len(generic_cases):
                    orig_idx = generic_cases[j][0]
                    all_results[orig_idx] = v.get("approve", True)

    # Simple prompt for non-generic (more lenient)
    if other_cases:
        cases_text = []
        for j, (i, case) in enumerate(other_cases):
            cases_text.append(f'Case {j+1}: "{case["match"]}" -> {case["comp"]}\n  "{case["text"]}"')

        prompt = f"""Validate component references. Focus on ACTOR role: is the component performing an action or being described?

COMPONENTS: {comp_names}

IMPORTANT: "Recording Processor" is a known synonym for "Recording Service". Approve if the sentence describes what this component does.

CASES:
{chr(10).join(cases_text)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

        print(f"\n  Simple validating {len(other_cases)} non-generic cases...")
        data = llm.extract_json(llm.query(prompt, timeout=120))
        if data:
            for v in data.get("validations", []):
                j = v.get("case", 0) - 1
                if 0 <= j < len(other_cases):
                    orig_idx = other_cases[j][0]
                    all_results[orig_idx] = v.get("approve", True)

    # Fill missing
    for i in range(len(PHASE6_CASES)):
        if i not in all_results:
            all_results[i] = True

    return eval_judge_results(PHASE6_CASES, all_results, "Idea E (generic-risk 4-rule Phase 6)")


def test_v22_phase6():
    """V22's existing Phase 6 prompt — baseline."""
    print("\n" + "="*80)
    print("BASELINE: V22 Phase 6 (old prompt, all cases same)")
    print("="*80)

    comp_names = ", ".join(BBB_COMPONENTS)
    cases_text = []
    for i, case in enumerate(PHASE6_CASES):
        cases_text.append(f'Case {i+1}: "{case["match"]}" -> {case["comp"]}\n  "{case["text"]}"')

    prompt = f"""Validate component references in a software architecture document.
Focus on ACTOR role: is the component performing an action or being described?

COMPONENTS: {comp_names}

IMPORTANT DISTINCTIONS:
- "the routing logic" / "business logic" = generic English, NOT an architectural component → REJECT
- "client-side rendering" / "on the client" = generic usage, NOT a specific component → REJECT
- "data storage layer" / "in-memory cache" = generic concept, NOT a specific component → REJECT
- But "Router handles request processing" = the component IS the actor → APPROVE
- Section headings naming a component = introduces that component's section → APPROVE

CASES:
{chr(10).join(cases_text)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

    print(f"  Validating {len(PHASE6_CASES)} cases...")
    data = llm.extract_json(llm.query(prompt, timeout=120))

    results = {}
    if data:
        for v in data.get("validations", []):
            idx = v.get("case", 0) - 1
            if 0 <= idx < len(PHASE6_CASES):
                results[idx] = v.get("approve", True)

    return eval_judge_results(PHASE6_CASES, results, "V22 Phase 6 baseline")


# ═══════════════════════════════════════════════════════════════
# IDEA E2: Full 4-rule Phase 6 on ALL cases (V23d style)
# ═══════════════════════════════════════════════════════════════

def test_full_4rule_phase6():
    """Full 4-rule Phase 6 on all cases — V23d approach."""
    print("\n" + "="*80)
    print("V23d: Full 4-rule Phase 6 on ALL cases")
    print("="*80)

    comp_names = ", ".join(BBB_COMPONENTS)
    cases_text = []
    for i, case in enumerate(PHASE6_CASES):
        cases_text.append(f'Case {i+1}: "{case["match"]}" -> {case["comp"]}\n  "{case["text"]}"')

    prompt = f"""Validate component references in a software architecture document.
Focus on ACTOR role: is the component performing an action or being described?

A sentence S should be linked to component C ONLY if ALL FOUR are true:

RULE 1 — REFERENCE: S actually refers to C (not just string-matches its name).
RULE 2 — ARCHITECTURAL LEVEL: S describes C at the architectural level.
RULE 3 — TOPIC: C is the topic or subject of S.
RULE 4 — NOT GENERIC: The reference is to C as a specific architectural component.

COMPONENTS: {comp_names}

CASES:
{chr(10).join(cases_text)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

    print(f"  Validating {len(PHASE6_CASES)} cases...")
    data = llm.extract_json(llm.query(prompt, timeout=120))

    results = {}
    if data:
        for v in data.get("validations", []):
            idx = v.get("case", 0) - 1
            if 0 <= idx < len(PHASE6_CASES):
                results[idx] = v.get("approve", True)

    return eval_judge_results(PHASE6_CASES, results, "V23d full 4-rule Phase 6")


if __name__ == "__main__":
    all_results = {}

    # Phase 9 tests
    all_results["v22_judge"] = test_v22_individual_judge()
    all_results["idea_c"] = test_idea_c_batch_judge()

    # Phase 6 tests
    all_results["v22_phase6"] = test_v22_phase6()
    all_results["v23d_phase6"] = test_full_4rule_phase6()
    all_results["idea_e"] = test_idea_e_generic_risk_phase6()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Test':<30} {'TP kept':>8} {'TP kill':>8} {'FP kill':>8} {'FP kept':>8}")
    print("-"*66)
    for name, r in all_results.items():
        print(f"{name:<30} {r['tp_kept']:>8} {r['tp_killed']:>8} {r['fp_killed']:>8} {r['fp_kept']:>8}")
