#!/usr/bin/env python3
"""Unit-test level evaluation of MORE judge prompt ideas on known BBB cases.

Round 2: trying to get the best of batch (TP preservation) and individual (FP killing).
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

os.environ["CLAUDE_MODEL"] = "sonnet"
from llm_sad_sam.llm_client import LLMClient, LLMBackend

llm = LLMClient(backend=LLMBackend.CLAUDE)

BBB_COMPONENTS = [
    "HTML5 Client", "HTML5 Server", "BBB web", "Apps", "FreeSWITCH",
    "FSESL", "Redis PubSub", "Redis DB", "WebRTC-SFU", "kurento",
    "Presentation Conversion", "Recording Service"
]

# All alias-matched links that reach the judge (without bypass)
# Gold = TP (should approve), FP (should reject)
ALIAS_CASES = [
    # TPs
    {"s": 9,  "comp": "HTML5 Client", "text": "The HTML5 server is built upon Meteor.js in ECMA2015 for communication between client and server and upon MongoDB for keeping the state of each BigBlueButton client consistent with the BigBlueButton server.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 10, "comp": "HTML5 Client", "text": "The MongoDB database contains information about all meetings on the server and, in turn, each client connected to a meeting.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 12, "comp": "HTML5 Client", "text": "The client side subscribes to the published collections on the server side.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 13, "comp": "HTML5 Client", "text": "Updates to MongoDB on the server side are automatically pushed to MiniMongo on the client side.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 19, "comp": "HTML5 Client", "text": "BigBlueButton 2.3 moves away from a single nodejs process for bbb-html5 towards multiple nodejs processes handling incoming messages from clients.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 37, "comp": "BBB web", "text": "BigBlueButton web application is a Java-based application written in Scala.", "match": "BigBlueButton web application", "gold": "TP", "source": "validated"},
    {"s": 76, "comp": "HTML5 Client", "text": "Uploaded presentations go through a conversion process in order to be displayed inside the client.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 79, "comp": "HTML5 Client", "text": "The conversion process sends progress messages to the client through the Redis pubsub.", "match": "Client", "gold": "TP", "source": "validated"},
    # FPs
    {"s": 16, "comp": "HTML5 Client", "text": "BigBlueButton 2.2 used a single nodejs process for all client-side communication.", "match": "Client", "gold": "FP", "source": "validated"},
    {"s": 58, "comp": "FSESL", "text": "This allows others who are using voice conference systems other than FreeSWITCH to easily create their own integration.", "match": "FreeSWITCH", "gold": "FP", "source": "coreference"},
    {"s": 50, "comp": "Recording Service", "text": "When the meeting ends, the Recording Processor will take all the recorded events as well as the different raw (PDF, WAV, FLV) files for processing.", "match": "Recording Processor", "gold": "FP", "source": "validated"},
    {"s": 83, "comp": "Presentation Conversion", "text": "Then below the SVG conversion flow.", "match": "conversion", "gold": "FP", "source": "validated"},
    {"s": 84, "comp": "Presentation Conversion", "text": "It covers the conversion fallback.", "match": "conversion", "gold": "FP", "source": "validated"},
    {"s": 85, "comp": "Presentation Conversion", "text": "Sometimes we detect that the generated SVG file is heavy to load by the browser, we use the fallback to put a rasterized image inside the SVG file and make its loading light for the browser.", "match": "NONE", "gold": "FP", "source": "validated"},
]


def eval_judge_results(cases, results, label):
    tp_kept = tp_killed = fp_kept = fp_killed = 0
    for i, case in enumerate(cases):
        approved = results.get(i, True)
        if case["gold"] == "TP":
            if approved: tp_kept += 1
            else:
                tp_killed += 1
                print(f"  TP KILLED: S{case['s']} -> {case['comp']} [{case['match']}]")
        else:
            if approved:
                fp_kept += 1
                print(f"  FP KEPT:   S{case['s']} -> {case['comp']} [{case['match']}]")
            else: fp_killed += 1
    total_tp = tp_kept + tp_killed
    total_fp = fp_kept + fp_killed
    print(f"\n  {label}: TP kept={tp_kept}/{total_tp}, FP killed={fp_killed}/{total_fp}")
    print(f"  TP kill rate: {tp_killed}/{total_tp} = {tp_killed/max(1,total_tp):.0%}")
    print(f"  FP kill rate: {fp_killed}/{total_fp} = {fp_killed/max(1,total_fp):.0%}")
    score = tp_kept * 2 - tp_killed * 3 + fp_killed * 2 - fp_kept * 1  # weighted score
    print(f"  Weighted score: {score} (higher is better)")
    return {"tp_kept": tp_kept, "tp_killed": tp_killed, "fp_kept": fp_kept, "fp_killed": fp_killed, "score": score}


# ═══════════════════════════════════════════════════════════════
# IDEA F: Two-pass appeal — individual judge, then batch appeal
# for rejected alias links
# ═══════════════════════════════════════════════════════════════

def test_idea_f_appeal():
    """Individual 4-rule judge first, then batch appeal for rejected alias links."""
    print("\n" + "="*80)
    print("IDEA F: Two-pass appeal (individual judge → batch appeal for rejected)")
    print("="*80)

    comp_names = ", ".join(BBB_COMPONENTS)

    # Pass 1: Individual 4-rule judge
    cases_text = []
    for i, case in enumerate(ALIAS_CASES):
        match_info = f'match:"{case["match"]}"' if case["match"] != "NONE" else "match:NONE"
        cases_text.append(f'Case {i+1}: S{case["s"]} -> {case["comp"]} ({match_info})\n    >>> {case["text"]}')

    prompt1 = f"""JUDGE: Review trace links. Reject if the sentence doesn't discuss the component at the architectural level.

RULE 1 — REFERENCE: S refers to C (not just string match).
RULE 2 — ARCHITECTURAL LEVEL: S describes C's role/behavior/interactions.
RULE 3 — TOPIC: C is the subject of S.
RULE 4 — NOT GENERIC: Reference is to C as a component, not generic English.

COMPONENTS: {comp_names}

LINKS:
{chr(10).join(cases_text)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

    print("  Pass 1: Individual 4-rule judge...")
    data1 = llm.extract_json(llm.query(prompt1, timeout=120))

    pass1_results = {}
    rejected_indices = []
    if data1:
        for j in data1.get("judgments", []):
            idx = j.get("case", 0) - 1
            if 0 <= idx < len(ALIAS_CASES):
                approved = j.get("approve", True)
                pass1_results[idx] = approved
                if not approved:
                    rejected_indices.append(idx)
                    print(f"    REJECTED: S{ALIAS_CASES[idx]['s']} -> {ALIAS_CASES[idx]['comp']}")

    # Pass 2: Batch appeal for rejected alias links, grouped by component
    print(f"\n  Pass 2: Batch appeal for {len(rejected_indices)} rejected links...")
    by_comp = defaultdict(list)
    for idx in rejected_indices:
        by_comp[ALIAS_CASES[idx]["comp"]].append(idx)

    for comp, indices in by_comp.items():
        if len(indices) < 2:
            continue  # Single rejection — no batch context advantage

        match_term = ALIAS_CASES[indices[0]]["match"]
        sents = "\n".join(f"  S{ALIAS_CASES[idx]['s']}: {ALIAS_CASES[idx]['text']}" for idx in indices)

        prompt2 = f"""APPEAL: These links to "{comp}" were initially rejected. Reconsider with batch context.

In this document, "{match_term}" is a confirmed partial reference for the component "{comp}".
This mapping was verified during document analysis. The question is whether each sentence
discusses "{comp}" at the architectural level.

COMPONENTS: {comp_names}

REJECTED SENTENCES for {comp}:
{sents}

For each sentence, reconsider: does it describe the architectural behavior, role, or interactions
of {comp}? A sentence about how "{comp}" communicates, receives data, or interacts with other
components IS architectural even if "{match_term}" appears in a common phrase like "client side".

Return JSON:
{{"appeals": [{{"sentence": N, "overturn": true/false, "reason": "brief"}}]}}
JSON only:"""

        data2 = llm.extract_json(llm.query(prompt2, timeout=120))
        if data2:
            for a in data2.get("appeals", []):
                snum = a.get("sentence")
                if a.get("overturn", False):
                    for idx in indices:
                        if ALIAS_CASES[idx]["s"] == snum:
                            pass1_results[idx] = True  # Overturn rejection
                            print(f"    OVERTURNED: S{snum} -> {comp}")
                            break

    return eval_judge_results(ALIAS_CASES, pass1_results, "Idea F (appeal)")


# ═══════════════════════════════════════════════════════════════
# IDEA G: Validated + alias = immune from judge
# If Phase 6 already validated AND alias matches, skip judge
# ═══════════════════════════════════════════════════════════════

def test_idea_g_validated_immune():
    """Links that passed Phase 6 validation AND have alias match are immune."""
    print("\n" + "="*80)
    print("IDEA G: Validated + alias match = immune (no judge needed)")
    print("="*80)

    results = {}
    immune_count = 0
    review_cases = []

    for i, case in enumerate(ALIAS_CASES):
        # Immune if: source is "validated" AND match is not NONE
        if case["source"] == "validated" and case["match"] != "NONE":
            results[i] = True  # Immune — auto-approve
            immune_count += 1
        else:
            review_cases.append(i)

    print(f"  Immune: {immune_count}, Need review: {len(review_cases)}")

    # Judge only non-immune cases
    if review_cases:
        comp_names = ", ".join(BBB_COMPONENTS)
        cases_text = []
        for j, idx in enumerate(review_cases):
            case = ALIAS_CASES[idx]
            match_info = f'match:"{case["match"]}"' if case["match"] != "NONE" else "match:NONE"
            cases_text.append(f'Case {j+1}: S{case["s"]} -> {case["comp"]} (src:{case["source"]}, {match_info})\n    >>> {case["text"]}')

        prompt = f"""JUDGE: Review these trace links strictly.

RULE 1 — REFERENCE: S refers to C.
RULE 2 — ARCHITECTURAL LEVEL: S describes C's role/behavior.
RULE 3 — TOPIC: C is the subject of S.
RULE 4 — NOT GENERIC.

COMPONENTS: {comp_names}

LINKS:
{chr(10).join(cases_text)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

        data = llm.extract_json(llm.query(prompt, timeout=60))
        if data:
            for j_data in data.get("judgments", []):
                j = j_data.get("case", 0) - 1
                if 0 <= j < len(review_cases):
                    results[review_cases[j]] = j_data.get("approve", True)

    return eval_judge_results(ALIAS_CASES, results, "Idea G (validated+alias immune)")


# ═══════════════════════════════════════════════════════════════
# IDEA H: Batch judge with strict architectural filter
# Same as C but with explicit rejection criteria
# ═══════════════════════════════════════════════════════════════

def test_idea_h_batch_strict():
    """Batch judge with explicit rejection criteria for weak matches."""
    print("\n" + "="*80)
    print("IDEA H: Batch judge with strict architectural filter")
    print("="*80)

    comp_names = ", ".join(BBB_COMPONENTS)
    by_comp = defaultdict(list)
    for i, case in enumerate(ALIAS_CASES):
        by_comp[case["comp"]].append((i, case))

    all_results = {}

    for comp, group in by_comp.items():
        if len(group) < 2:
            idx, case = group[0]
            # Single case — individual judge
            prompt = f"""Should S{case['s']} be linked to "{comp}"? Match: "{case['match']}"
S{case['s']}: {case['text']}
COMPONENTS: {comp_names}
Answer JSON: {{"approve": true/false, "reason": "brief"}}
JSON only:"""
            data = llm.extract_json(llm.query(prompt, timeout=60))
            all_results[idx] = data.get("approve", True) if data else True
            continue

        match_term = group[0][1]["match"]
        sentences_block = "\n".join(f"  S{case['s']}: {case['text']}" for _, case in group)

        prompt = f"""JUDGE: Review links to "{comp}" via alias "{match_term}".

"{match_term}" is a confirmed alias for "{comp}" in this document.

APPROVE if the sentence describes {comp}'s architectural role, behavior, or interactions with other components.

REJECT if:
- The sentence is about a DIFFERENT component and "{match_term}" is incidental
- The sentence describes implementation details (internal classes, code structure) not architectural behavior
- The sentence is a fragment or heading without architectural content
- The component name appears only in a compound modifier (e.g., "client-side" as adjective)

COMPONENTS: {comp_names}

SENTENCES:
{sentences_block}

Return JSON:
{{"judgments": [{{"sentence": N, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

        print(f"\n  Batch-strict judging {comp} ({len(group)} links)...")
        data = llm.extract_json(llm.query(prompt, timeout=120))
        if data:
            for j in data.get("judgments", []):
                snum = j.get("sentence")
                approved = j.get("approve", True)
                for idx, case in group:
                    if case["s"] == snum:
                        all_results[idx] = approved
                        status = "APPROVE" if approved else "REJECT"
                        print(f"    S{snum} -> {comp}: {status} ({j.get('reason', '')})")
                        break

    for i in range(len(ALIAS_CASES)):
        if i not in all_results:
            all_results[i] = True

    return eval_judge_results(ALIAS_CASES, all_results, "Idea H (batch-strict)")


# ═══════════════════════════════════════════════════════════════
# IDEA I: Hybrid — batch for multi-link components, individual for singles
# with union voting on the individual pass
# ═══════════════════════════════════════════════════════════════

def test_idea_i_hybrid():
    """Batch for components with 3+ alias links. Individual 4-rule for rest. Union vote."""
    print("\n" + "="*80)
    print("IDEA I: Hybrid (batch for 3+ links, individual for rest, union vote)")
    print("="*80)

    comp_names = ", ".join(BBB_COMPONENTS)
    by_comp = defaultdict(list)
    for i, case in enumerate(ALIAS_CASES):
        by_comp[case["comp"]].append((i, case))

    all_results = {}

    # Batch judge for components with 3+ links
    for comp, group in by_comp.items():
        if len(group) >= 3:
            match_term = group[0][1]["match"]
            sentences_block = "\n".join(f"  S{case['s']}: {case['text']}" for _, case in group)

            prompt = f"""JUDGE: "{match_term}" is a confirmed alias for "{comp}" in this document.

Review each sentence: does it discuss "{comp}" at the architectural level?

APPROVE: Describes {comp}'s role, behavior, data flow, or interaction with other components.
REJECT: {comp} is incidental, or sentence is about implementation details, or is a fragment.

COMPONENTS: {comp_names}

SENTENCES linked to {comp}:
{sentences_block}

Return JSON:
{{"judgments": [{{"sentence": N, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

            print(f"\n  Batch judging {comp} ({len(group)} links)...")
            data = llm.extract_json(llm.query(prompt, timeout=120))
            if data:
                for j in data.get("judgments", []):
                    snum = j.get("sentence")
                    for idx, case in group:
                        if case["s"] == snum:
                            all_results[idx] = j.get("approve", True)
                            status = "APPROVE" if all_results[idx] else "REJECT"
                            print(f"    S{snum}: {status} ({j.get('reason','')})")
                            break
        else:
            # Individual links — two-pass union vote (reject only if both reject)
            for idx, case in group:
                match_info = f'match:"{case["match"]}"' if case["match"] != "NONE" else "match:NONE"
                prompt = f"""Should S{case['s']} be linked to "{case['comp']}"? ({match_info}, src:{case['source']})

S{case['s']}: {case['text']}

COMPONENTS: {comp_names}
RULE: Approve if S discusses {case['comp']} at the architectural level.

Answer JSON: {{"approve": true/false, "reason": "brief"}}
JSON only:"""

                d1 = llm.extract_json(llm.query(prompt, timeout=60))
                d2 = llm.extract_json(llm.query(prompt, timeout=60))
                r1 = d1.get("approve", True) if d1 else True
                r2 = d2.get("approve", True) if d2 else True
                all_results[idx] = r1 or r2  # Union: reject only if BOTH reject
                status = "APPROVE" if all_results[idx] else "REJECT"
                print(f"  Individual (union): S{case['s']} -> {case['comp']}: {status} (pass1={r1}, pass2={r2})")

    for i in range(len(ALIAS_CASES)):
        if i not in all_results:
            all_results[i] = True

    return eval_judge_results(ALIAS_CASES, all_results, "Idea I (hybrid)")


# ═══════════════════════════════════════════════════════════════
# IDEA J: Sentence-role judge — ask what component each sentence is about
# instead of yes/no per link
# ═══════════════════════════════════════════════════════════════

def test_idea_j_sentence_role():
    """Instead of judging links, ask: what component is this sentence about?"""
    print("\n" + "="*80)
    print("IDEA J: Sentence-role judge (ask what component each sentence is about)")
    print("="*80)

    comp_names = ", ".join(BBB_COMPONENTS)

    # Unique sentences
    unique_sents = {}
    for case in ALIAS_CASES:
        if case["s"] not in unique_sents:
            unique_sents[case["s"]] = case["text"]

    sents_block = "\n".join(f"S{snum}: {text}" for snum, text in sorted(unique_sents.items()))

    prompt = f"""For each sentence, identify which architecture component(s) it primarily discusses.
A sentence's PRIMARY component is the one whose architectural role/behavior/interaction is being described.

COMPONENTS: {comp_names}

KNOWN ALIASES:
- "Client" / "client" = HTML5 Client (in this BigBlueButton document)
- "Server" / "server" = HTML5 Server
- "BigBlueButton web application" = BBB web
- "Recording Processor" = Recording Service
- "conversion" may refer to Presentation Conversion

SENTENCES:
{sents_block}

For each sentence, list the primary component(s) it discusses. If a component is only mentioned
incidentally (not the topic), do NOT include it.

Return JSON:
{{"sentences": [{{"sentence": N, "primary_components": ["CompA", "CompB"]}}]}}
JSON only:"""

    print(f"  Asking about {len(unique_sents)} unique sentences...")
    data = llm.extract_json(llm.query(prompt, timeout=120))

    # Map results back
    sent_comps = {}
    if data:
        for s in data.get("sentences", []):
            snum = s.get("sentence")
            comps = set(s.get("primary_components", []))
            sent_comps[snum] = comps
            print(f"    S{snum}: {comps}")

    results = {}
    for i, case in enumerate(ALIAS_CASES):
        primary = sent_comps.get(case["s"], set())
        results[i] = case["comp"] in primary

    return eval_judge_results(ALIAS_CASES, results, "Idea J (sentence-role)")


# ═══════════════════════════════════════════════════════════════
# IDEA K: Component-frequency trust — if alias matches 5+ times,
# trust it as a real reference pattern
# ═══════════════════════════════════════════════════════════════

def test_idea_k_frequency_trust():
    """Deterministic: if a partial appears 5+ times for a component, auto-approve all.
    Judge only low-frequency alias links."""
    print("\n" + "="*80)
    print("IDEA K: Frequency trust (5+ alias matches = auto-approve, rest judged)")
    print("="*80)

    comp_names = ", ".join(BBB_COMPONENTS)

    # Count alias frequency per component
    freq = defaultdict(int)
    for case in ALIAS_CASES:
        freq[case["comp"]] += 1

    print(f"  Frequencies: {dict(freq)}")

    results = {}
    review_cases = []

    for i, case in enumerate(ALIAS_CASES):
        if freq[case["comp"]] >= 5:
            results[i] = True  # High-frequency — trust it
            print(f"  Auto-approve (freq={freq[case['comp']]}): S{case['s']} -> {case['comp']}")
        else:
            review_cases.append(i)

    # Judge low-frequency ones individually
    if review_cases:
        cases_text = []
        for j, idx in enumerate(review_cases):
            case = ALIAS_CASES[idx]
            match_info = f'match:"{case["match"]}"' if case["match"] != "NONE" else "match:NONE"
            cases_text.append(f'Case {j+1}: S{case["s"]} -> {case["comp"]} ({match_info})\n    >>> {case["text"]}')

        prompt = f"""JUDGE: Review these trace links strictly.

RULE 1 — REFERENCE: S refers to C.
RULE 2 — ARCHITECTURAL LEVEL: S describes C at arch level.
RULE 3 — TOPIC: C is subject of S.
RULE 4 — NOT GENERIC.

COMPONENTS: {comp_names}

LINKS:
{chr(10).join(cases_text)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

        print(f"\n  Judging {len(review_cases)} low-frequency cases...")
        data = llm.extract_json(llm.query(prompt, timeout=60))
        if data:
            for j in data.get("judgments", []):
                j_idx = j.get("case", 0) - 1
                if 0 <= j_idx < len(review_cases):
                    results[review_cases[j_idx]] = j.get("approve", True)

    for i in range(len(ALIAS_CASES)):
        if i not in results:
            results[i] = True

    return eval_judge_results(ALIAS_CASES, results, "Idea K (frequency trust)")


if __name__ == "__main__":
    all_results = {}

    all_results["idea_f"] = test_idea_f_appeal()
    all_results["idea_g"] = test_idea_g_validated_immune()
    all_results["idea_h"] = test_idea_h_batch_strict()
    all_results["idea_i"] = test_idea_i_hybrid()
    all_results["idea_j"] = test_idea_j_sentence_role()
    all_results["idea_k"] = test_idea_k_frequency_trust()

    # Include round 1 results for comparison
    print("\n" + "="*80)
    print("SUMMARY (all ideas)")
    print("="*80)
    print(f"{'Test':<35} {'TP kept':>8} {'TP kill':>8} {'FP kill':>8} {'FP kept':>8} {'Score':>6}")
    print("-"*75)

    # Round 1 results (hardcoded from previous run)
    print(f"{'[R1] v22_individual_judge':<35} {'3':>8} {'5':>8} {'5':>8} {'1':>8} {'8':>6}")
    print(f"{'[R1] idea_c_batch':<35} {'7':>8} {'1':>8} {'2':>8} {'4':>8} {'11':>6}")

    for name, r in all_results.items():
        print(f"{name:<35} {r['tp_kept']:>8} {r['tp_killed']:>8} {r['fp_killed']:>8} {r['fp_kept']:>8} {r['score']:>6}")
