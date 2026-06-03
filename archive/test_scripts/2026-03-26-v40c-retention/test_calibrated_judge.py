#!/usr/bin/env python3
"""Test: Calibrated Elimination Judge for generic partial references.

For each single-word partial (e.g., "Server" → HTML5 Server), batch all
sentences that match, find full-name anchor sentences, and ask the LLM
to classify each partial-only sentence as naming vs generic — calibrated
by the full-name examples from the same document.

One LLM call per partial name, not per link.
"""
import csv
import glob
import os
import pickle
import re
import sys
from collections import defaultdict
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

    # Group syn-review links by (partial, component)
    # Only for single-word generic partials
    partial_groups = defaultdict(list)  # (partial, comp_name) -> [(link, is_tp)]
    other_syn = []  # non-generic syn-review links (auto-approve)

    for l in prelim:
        sent = sent_map.get(l.sentence_number)
        if not sent:
            continue
        text_lower = sent.text.lower()

        # Check partial refs first
        matched_partial = None
        for partial, target in dk.partial_references.items():
            if target == l.component_name:
                if re.search(rf'\b{re.escape(partial.lower())}\b', text_lower):
                    matched_partial = partial
                    break

        if matched_partial:
            is_single_generic = (' ' not in matched_partial
                                 and not re.search(r'[a-z][A-Z]', matched_partial))
            is_tp = (l.sentence_number, l.component_id) in gold
            if is_single_generic:
                partial_groups[(matched_partial, l.component_name)].append((l, is_tp))
            else:
                other_syn.append((l, is_tp))
            continue

        # Check synonyms
        matched_syn = None
        for syn, target in dk.synonyms.items():
            if target == l.component_name:
                if re.search(rf'\b{re.escape(syn.lower())}\b', text_lower):
                    matched_syn = syn
                    break

        if matched_syn:
            is_tp = (l.sentence_number, l.component_id) in gold
            other_syn.append((l, is_tp))

    total_generic = sum(len(v) for v in partial_groups.values())
    print(f"BBB syn-review breakdown:")
    print(f"  Non-generic (auto-approve): {len(other_syn)}")
    print(f"  Generic partials (calibrated judge): {total_generic}")
    for (partial, comp), links in sorted(partial_groups.items()):
        tps = sum(1 for _, is_tp in links if is_tp)
        fps = sum(1 for _, is_tp in links if not is_tp)
        print(f"    \"{partial}\" → {comp}: {len(links)} links ({tps} TP, {fps} FP)")

    # Find full-name anchor sentences for each component
    # These are sentences where the FULL component name appears
    print(f"\nFinding full-name anchors...")
    anchors = {}  # comp_name -> [(snum, text)]
    for (partial, comp), links in partial_groups.items():
        if comp in anchors:
            continue
        comp_anchors = []
        for s in sentences:
            if re.search(rf'\b{re.escape(comp)}\b', s.text, re.IGNORECASE):
                comp_anchors.append((s.number, s.text))
        anchors[comp] = comp_anchors[:5]  # max 5 anchors
        print(f"  {comp}: {len(comp_anchors)} full-name sentences (using {len(anchors[comp])})")
        for snum, text in anchors[comp]:
            print(f"    S{snum}: {text[:80]}...")

    # Run calibrated judge — one call per (partial, component)
    llm = LLMClient(backend=LLMBackend.CLAUDE)

    print(f"\n{'='*80}")
    print(f"CALIBRATED ELIMINATION JUDGE")
    print(f"{'='*80}")

    results = {"tp_kept": 0, "tp_killed": 0, "fp_kept": 0, "fp_killed": 0}

    # Auto-approve non-generic
    for l, is_tp in other_syn:
        if is_tp:
            results["tp_kept"] += 1
        else:
            results["fp_kept"] += 1

    for (partial, comp), links in sorted(partial_groups.items()):
        comp_anchors = anchors.get(comp, [])

        # Build anchor section
        if comp_anchors:
            anchor_lines = "\n".join(
                f"  S{snum}: {text}" for snum, text in comp_anchors
            )
            anchor_section = f"""FULL-NAME REFERENCES (calibration — these definitely refer to {comp}):
{anchor_lines}

In these sentences, the author uses the full name "{comp}". This shows how the author
refers to this component when being explicit."""
        else:
            anchor_section = f"(No full-name references found for {comp} in this document.)"

        # Build cases to judge
        case_lines = []
        for i, (l, is_tp) in enumerate(sorted(links, key=lambda x: x[0].sentence_number)):
            sent = sent_map.get(l.sentence_number)
            case_lines.append(f"  Case {i+1} (S{l.sentence_number}): {sent.text}")

        cases_section = "\n".join(case_lines)

        prompt = f"""BATCH CLASSIFICATION: For each sentence below, determine whether "{partial}"
refers to the architecture component "{comp}", or is used as an ordinary English word.

{anchor_section}

SENTENCES TO CLASSIFY (only the short name "{partial}" appears, not the full name):
{cases_section}

For each case, apply this test:
Does "{partial}" in that sentence follow the SAME usage pattern as the full-name
references above — identifying the same system component as a participant?
Or is "{partial}" used in its ordinary English sense — describing a general concept,
activity, or type that happens to share the word?

A sentence can describe activities RELATED to what {comp} does without actually
referring to {comp} by name. If "{partial}" is part of a descriptive phrase like
"{partial.lower()} process", "SVG {partial.lower()}", or "{partial.lower()} fallback" — where
the word describes a generic activity or type — that is ordinary English, not a
component reference.

Return JSON:
{{"classifications": [
  {{"case": 1, "refers_to_component": true/false, "reason": "brief"}},
  ...
]}}
JSON only:"""

        print(f"\n--- Partial \"{partial}\" → {comp} ({len(links)} links) ---")
        print(f"    Anchors: {len(comp_anchors)} full-name sentences")

        # Two passes, union voting
        all_pass_results = []
        for pass_num in range(2):
            data = llm.extract_json(llm.query(prompt, timeout=120))
            pass_results = {}
            if data:
                for c in data.get("classifications", []):
                    idx = c.get("case", 0) - 1
                    refers = c.get("refers_to_component", True)
                    reason = c.get("reason", "")
                    pass_results[idx] = (refers, reason)
            all_pass_results.append(pass_results)

        # Union voting: reject only if BOTH passes reject
        for i, (l, is_tp) in enumerate(sorted(links, key=lambda x: x[0].sentence_number)):
            label = "TP" if is_tp else "FP"

            p1_refers, p1_reason = all_pass_results[0].get(i, (True, "missing"))
            p2_refers, p2_reason = all_pass_results[1].get(i, (True, "missing"))

            approved = p1_refers or p2_refers  # union: approve if either says yes

            if approved:
                if is_tp: results["tp_kept"] += 1
                else: results["fp_kept"] += 1
            else:
                if is_tp: results["tp_killed"] += 1
                else: results["fp_killed"] += 1

            status = "APPROVE" if approved else "REJECT"
            correct = "correct" if (is_tp == approved) else "WRONG"
            p1_tag = "ref" if p1_refers else "generic"
            p2_tag = "ref" if p2_refers else "generic"
            print(f"    [{label}] S{l.sentence_number} -> {comp}: {status} ({correct}) [P1:{p1_tag}, P2:{p2_tag}]")
            # Show reason from first pass that drove the decision
            reason = p1_reason if not approved else (p1_reason if p1_refers else p2_reason)
            print(f"         {reason}")

    print(f"\n{'='*80}")
    print(f"RESULTS (with union voting)")
    print(f"{'='*80}")
    print(f"  TPs kept:   {results['tp_kept']}")
    print(f"  TPs killed: {results['tp_killed']}  ← must be 0")
    print(f"  FPs killed: {results['fp_killed']}  ← want max")
    print(f"  FPs kept:   {results['fp_kept']}")
    accuracy = results['tp_kept'] + results['fp_killed']
    total = sum(results.values())
    print(f"  Accuracy:   {accuracy}/{total}")
    print(f"  LLM calls:  {len(partial_groups) * 2} (2 passes × {len(partial_groups)} partials)")


if __name__ == "__main__":
    main()
