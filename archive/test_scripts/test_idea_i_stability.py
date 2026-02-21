#!/usr/bin/env python3
"""Stability test: Run Idea I (hybrid batch+union judge) 5 times on the same BBB cases."""
import json
import os
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

ALIAS_CASES = [
    {"s": 9,  "comp": "HTML5 Client", "text": "The HTML5 server is built upon Meteor.js in ECMA2015 for communication between client and server and upon MongoDB for keeping the state of each BigBlueButton client consistent with the BigBlueButton server.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 10, "comp": "HTML5 Client", "text": "The MongoDB database contains information about all meetings on the server and, in turn, each client connected to a meeting.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 12, "comp": "HTML5 Client", "text": "The client side subscribes to the published collections on the server side.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 13, "comp": "HTML5 Client", "text": "Updates to MongoDB on the server side are automatically pushed to MiniMongo on the client side.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 19, "comp": "HTML5 Client", "text": "BigBlueButton 2.3 moves away from a single nodejs process for bbb-html5 towards multiple nodejs processes handling incoming messages from clients.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 37, "comp": "BBB web", "text": "BigBlueButton web application is a Java-based application written in Scala.", "match": "BigBlueButton web application", "gold": "TP", "source": "validated"},
    {"s": 76, "comp": "HTML5 Client", "text": "Uploaded presentations go through a conversion process in order to be displayed inside the client.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 79, "comp": "HTML5 Client", "text": "The conversion process sends progress messages to the client through the Redis pubsub.", "match": "Client", "gold": "TP", "source": "validated"},
    {"s": 16, "comp": "HTML5 Client", "text": "BigBlueButton 2.2 used a single nodejs process for all client-side communication.", "match": "Client", "gold": "FP", "source": "validated"},
    {"s": 58, "comp": "FSESL", "text": "This allows others who are using voice conference systems other than FreeSWITCH to easily create their own integration.", "match": "FreeSWITCH", "gold": "FP", "source": "coreference"},
    {"s": 50, "comp": "Recording Service", "text": "When the meeting ends, the Recording Processor will take all the recorded events as well as the different raw (PDF, WAV, FLV) files for processing.", "match": "Recording Processor", "gold": "FP", "source": "validated"},
    {"s": 83, "comp": "Presentation Conversion", "text": "Then below the SVG conversion flow.", "match": "conversion", "gold": "FP", "source": "validated"},
    {"s": 84, "comp": "Presentation Conversion", "text": "It covers the conversion fallback.", "match": "conversion", "gold": "FP", "source": "validated"},
    {"s": 85, "comp": "Presentation Conversion", "text": "Sometimes we detect that the generated SVG file is heavy to load by the browser, we use the fallback to put a rasterized image inside the SVG file and make its loading light for the browser.", "match": "NONE", "gold": "FP", "source": "validated"},
]


def run_idea_i(run_id):
    """Run Idea I once, return per-case results."""
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

            data = llm.extract_json(llm.query(prompt, timeout=120))
            if data:
                for j in data.get("judgments", []):
                    snum = j.get("sentence")
                    for idx, case in group:
                        if case["s"] == snum:
                            all_results[idx] = j.get("approve", True)
                            break
        else:
            # Individual links — two-pass union vote
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
                all_results[idx] = r1 or r2  # Union

    # Fill missing
    for i in range(len(ALIAS_CASES)):
        if i not in all_results:
            all_results[i] = True

    return all_results


def main():
    N_RUNS = 5
    all_runs = []

    for run in range(N_RUNS):
        print(f"\n{'='*80}")
        print(f"RUN {run+1}/{N_RUNS}")
        print(f"{'='*80}")
        t0 = time.time()
        results = run_idea_i(run)
        elapsed = time.time() - t0

        tp_kept = tp_killed = fp_kept = fp_killed = 0
        for i, case in enumerate(ALIAS_CASES):
            approved = results.get(i, True)
            label = "TP" if case["gold"] == "TP" else "FP"
            status = "APPROVE" if approved else "REJECT"
            correct = (case["gold"] == "TP" and approved) or (case["gold"] == "FP" and not approved)
            marker = "✓" if correct else "✗"
            print(f"  {marker} S{case['s']:>2} -> {case['comp']:<24} [{label}] {status}")
            if case["gold"] == "TP":
                if approved: tp_kept += 1
                else: tp_killed += 1
            else:
                if approved: fp_kept += 1
                else: fp_killed += 1

        print(f"\n  Run {run+1}: TP kept={tp_kept}/8, FP killed={fp_killed}/6 ({elapsed:.0f}s)")
        all_runs.append({"results": dict(results), "tp_kept": tp_kept, "tp_killed": tp_killed,
                         "fp_kept": fp_kept, "fp_killed": fp_killed})

    # Stability analysis
    print(f"\n{'='*80}")
    print("STABILITY ANALYSIS")
    print(f"{'='*80}")

    # Per-case stability
    print(f"\n{'Case':<45} ", end="")
    for r in range(N_RUNS):
        print(f"R{r+1:>2} ", end="")
    print("  Stable?")
    print("-" * (45 + N_RUNS * 4 + 10))

    unstable = []
    for i, case in enumerate(ALIAS_CASES):
        label = case["gold"]
        decisions = [all_runs[r]["results"].get(i, True) for r in range(N_RUNS)]
        approves = sum(1 for d in decisions if d)
        rejects = N_RUNS - approves

        case_str = f"S{case['s']:>2} -> {case['comp']:<22} [{label}]"
        print(f"{case_str:<45} ", end="")
        for d in decisions:
            print(f"{'A':>3} " if d else f"{'R':>3} ", end="")

        if approves == N_RUNS or rejects == N_RUNS:
            print("  YES")
        else:
            print(f"  NO ({approves}A/{rejects}R)")
            unstable.append((i, case, approves, rejects))

    # Summary
    print(f"\n{'Run':<8} {'TP kept':>8} {'TP kill':>8} {'FP kill':>8} {'FP kept':>8} {'Score':>6}")
    print("-" * 46)
    scores = []
    for r, run in enumerate(all_runs):
        score = run["tp_kept"] * 2 - run["tp_killed"] * 3 + run["fp_killed"] * 2 - run["fp_kept"] * 1
        scores.append(score)
        print(f"Run {r+1:<4} {run['tp_kept']:>8} {run['tp_killed']:>8} {run['fp_killed']:>8} {run['fp_kept']:>8} {score:>6}")

    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)
    print(f"\nScore: avg={avg_score:.1f}, min={min_score}, max={max_score}, range={max_score-min_score}")
    print(f"Unstable cases: {len(unstable)}/{len(ALIAS_CASES)}")
    if unstable:
        for i, case, a, r in unstable:
            print(f"  S{case['s']} -> {case['comp']} [{case['gold']}]: {a}A/{r}R")


if __name__ == "__main__":
    main()
