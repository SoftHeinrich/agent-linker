#!/usr/bin/env python3
"""Test: V39 calibrated batch judge for ORDINARY partials.

Loads V39 BBB checkpoint (phase3 + pre_judge), identifies ORDINARY partials
from Phase 3c classification, then runs the calibrated batch judge on those
links. Verifies:
  - Conversion TPs (S80, S81) are kept (LINK)
  - Conversion FPs (S76, S79, S82, S83, S84) are rejected (NO_LINK)
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
from llm_sad_sam.core.document_loader import DocumentLoader

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
    # Load V39 BBB checkpoints
    with open("results/phase_cache/v39/bigbluebutton/phase3.pkl", "rb") as f:
        p3 = pickle.load(f)
    with open("results/phase_cache/v39/bigbluebutton/pre_judge.pkl", "rb") as f:
        pj = pickle.load(f)

    dk = p3['doc_knowledge']
    prelim = pj['preliminary']
    gold = load_gold("bigbluebutton")

    # Load sentences
    text_files = glob.glob(str(BENCHMARK / "bigbluebutton" / "**" / "bigbluebutton.txt"), recursive=True)
    loader = DocumentLoader()
    sentences = loader.load_sentences(text_files[0])
    sent_map = {s.number: s for s in sentences}

    # ── Step 1: Reproduce Phase 3c classification ──
    # Find single-word generic partials (same logic as V39._classify_partial_usage)
    generic_partials = {}
    for partial, comp_name in dk.partial_references.items():
        if ' ' in partial:
            continue
        if re.search(r'[a-z][A-Z]', partial):  # CamelCase = always entity
            continue
        generic_partials[partial] = comp_name

    print(f"Generic partials to classify: {generic_partials}")

    # Run classification (same as V39 Phase 3c)
    llm = LLMClient(backend=LLMBackend.CLAUDE)
    activity_partials = set()

    for partial, comp_name in sorted(generic_partials.items()):
        partial_lower = partial.lower()
        comp_lower = comp_name.lower()
        partial_sentences = []
        full_name_sentences = []

        for s in sentences:
            text_lower = s.text.lower()
            has_partial = re.search(rf'\b{re.escape(partial_lower)}\b', text_lower)
            has_full = re.search(rf'\b{re.escape(comp_lower)}\b', text_lower)
            if has_full:
                full_name_sentences.append(s)
            elif has_partial:
                partial_sentences.append(s)

        if not partial_sentences:
            continue

        sent_lines = [f"  S{s.number}: {s.text}" for s in partial_sentences[:15]]
        sent_block = "\n".join(sent_lines)

        calibration = ""
        if full_name_sentences:
            fn_lines = [f"  S{s.number}: {s.text}" for s in full_name_sentences[:5]]
            fn_block = "\n".join(fn_lines)
            calibration = f'For reference, these sentences use the FULL component name "{comp_name}":\n{fn_block}\n'

        prompt = f"""WORD USAGE CLASSIFICATION

In this document, the word "{partial}" could be a short name for an architecture
component called "{comp_name}".

{calibration}Below are ALL sentences where "{partial}" appears WITHOUT the full name "{comp_name}".
Analyze how the word "{partial}" is used across these sentences:

{sent_block}

QUESTION: Is "{partial}" used as a standalone entity reference in ANY of these sentences?

Classify as NAME if the word appears as a standalone noun phrase referring to a specific
system entity in at least SOME sentences — even if other sentences use it generically.

Classify as ORDINARY only if EVERY occurrence uses the word as part of a compound phrase,
modifier, or generic descriptor — never as a standalone entity.

The threshold is: if even ONE sentence uses "{partial}" as a standalone entity reference,
classify as NAME. Only classify as ORDINARY when you see ZERO standalone entity uses.

Return JSON: {{"classification": "name" or "ordinary", "reason": "brief explanation"}}
JSON only:"""

        data = llm.extract_json(llm.query(prompt, timeout=60))
        if data:
            classification = data.get("classification", "name")
            reason = data.get("reason", "")
            if classification == "ordinary":
                activity_partials.add(partial)
                print(f"  \"{partial}\" → {comp_name}: ORDINARY (no syn-safe). {reason}")
            else:
                print(f"  \"{partial}\" → {comp_name}: NAME (keep syn-safe). {reason}")
        else:
            print(f"  \"{partial}\" → {comp_name}: PARSE FAILURE (keep syn-safe)")

    print(f"\nActivity-type partials: {sorted(activity_partials)}")

    # ── Step 2: Find ORDINARY-partial links in pre_judge ──
    partial_groups = defaultdict(list)  # (partial, comp_name) -> [(link, is_tp)]
    other_syn = []  # Non-activity syn-review links (auto-approve)

    for l in prelim:
        sent = sent_map.get(l.sentence_number)
        if not sent:
            continue
        text_lower = sent.text.lower()

        # Check partial refs
        matched_partial = None
        for partial, target in dk.partial_references.items():
            if target == l.component_name:
                if re.search(rf'\b{re.escape(partial.lower())}\b', text_lower):
                    matched_partial = partial
                    break

        if matched_partial and matched_partial in activity_partials:
            is_single_generic = (' ' not in matched_partial
                                 and not re.search(r'[a-z][A-Z]', matched_partial))
            is_tp = (l.sentence_number, l.component_id) in gold
            if is_single_generic:
                partial_groups[(matched_partial, l.component_name)].append((l, is_tp))
                continue

        # Check if has alias mention (syn or partial)
        has_alias = False
        for syn, target in dk.synonyms.items():
            if target == l.component_name:
                if re.search(rf'\b{re.escape(syn.lower())}\b', text_lower):
                    has_alias = True
                    break
        if not has_alias:
            for partial, target in dk.partial_references.items():
                if target == l.component_name:
                    if re.search(rf'\b{re.escape(partial.lower())}\b', text_lower):
                        has_alias = True
                        break

        if has_alias:
            is_tp = (l.sentence_number, l.component_id) in gold
            other_syn.append((l, is_tp))

    total_activity = sum(len(v) for v in partial_groups.values())
    print(f"\n{'='*60}")
    print(f"ACTIVITY-PARTIAL LINKS TO JUDGE")
    print(f"{'='*60}")
    print(f"  Other syn-review (auto-approve): {len(other_syn)}")
    print(f"  Activity-partial (calibrated judge): {total_activity}")
    for (partial, comp), links in sorted(partial_groups.items()):
        tps = sum(1 for _, is_tp in links if is_tp)
        fps = sum(1 for _, is_tp in links if not is_tp)
        print(f"    \"{partial}\" → {comp}: {len(links)} links ({tps} TP, {fps} FP)")
        for l, is_tp in sorted(links, key=lambda x: x[0].sentence_number):
            sent = sent_map.get(l.sentence_number)
            tag = "TP" if is_tp else "FP"
            print(f"      [{tag}] S{l.sentence_number}: {sent.text[:80]}...")

    if not partial_groups:
        print("\nNo activity-partial links found. Nothing to judge.")
        return

    # ── Step 3: Run calibrated batch judge ──
    print(f"\n{'='*60}")
    print(f"CALIBRATED BATCH JUDGE (union voting, 2 passes)")
    print(f"{'='*60}")

    # Find full-name anchors
    anchors = {}
    for (partial, comp), links in partial_groups.items():
        if comp in anchors:
            continue
        comp_anchors = []
        for s in sentences:
            if re.search(rf'\b{re.escape(comp)}\b', s.text, re.IGNORECASE):
                comp_anchors.append((s.number, s.text))
        anchors[comp] = comp_anchors[:5]
        print(f"\n  Anchors for {comp}: {len(comp_anchors)} full-name sentences (using {len(anchors[comp])})")
        for snum, text in anchors[comp]:
            print(f"    S{snum}: {text[:80]}...")

    results = {"tp_kept": 0, "tp_killed": 0, "fp_kept": 0, "fp_killed": 0}

    for (partial, comp), links in sorted(partial_groups.items()):
        comp_anchors = anchors.get(comp, [])

        # Build anchor section
        if comp_anchors:
            anchor_lines = "\n".join(
                f"  S{snum}: {text}" for snum, text in comp_anchors
            )
            anchor_section = (
                f'FULL-NAME REFERENCES (calibration — these definitely refer to {comp}):\n'
                f'{anchor_lines}\n\n'
                f'In these sentences, the author uses the full name "{comp}". This shows how\n'
                f'the author refers to this component when being explicit.'
            )
        else:
            anchor_section = f"(No full-name references found for {comp} in this document.)"

        # Build cases
        sorted_links = sorted(links, key=lambda x: x[0].sentence_number)
        case_lines = []
        for i, (l, _) in enumerate(sorted_links):
            sent = sent_map.get(l.sentence_number)
            text = sent.text if sent else "(no text)"
            case_lines.append(f"  Case {i+1} (S{l.sentence_number}): {text}")
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

        print(f"\n--- Partial \"{partial}\" → {comp} ({len(sorted_links)} links) ---")

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
            print(f"  Pass {pass_num+1}: {sum(1 for v, _ in pass_results.values() if v)} approve, "
                  f"{sum(1 for v, _ in pass_results.values() if not v)} reject")

        for i, (l, is_tp) in enumerate(sorted_links):
            label = "TP" if is_tp else "FP"
            p1_refers, p1_reason = all_pass_results[0].get(i, (True, "missing"))
            p2_refers, p2_reason = all_pass_results[1].get(i, (True, "missing"))
            union_approved = p1_refers or p2_refers

            if union_approved:
                if is_tp: results["tp_kept"] += 1
                else: results["fp_kept"] += 1
            else:
                if is_tp: results["tp_killed"] += 1
                else: results["fp_killed"] += 1

            status = "APPROVE" if union_approved else "REJECT"
            correct = "correct" if (is_tp == union_approved) else "WRONG"
            p1_tag = "ref" if p1_refers else "generic"
            p2_tag = "ref" if p2_refers else "generic"
            reason = p1_reason if not union_approved else (p1_reason if p1_refers else p2_reason)
            print(f"    [{label}] S{l.sentence_number} -> {comp}: {status} ({correct}) [P1:{p1_tag}, P2:{p2_tag}]")
            print(f"         {reason}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  TPs kept:   {results['tp_kept']}")
    print(f"  TPs killed: {results['tp_killed']}  ← must be 0")
    print(f"  FPs killed: {results['fp_killed']}  ← want max (5 ideal)")
    print(f"  FPs kept:   {results['fp_kept']}")
    accuracy = results['tp_kept'] + results['fp_killed']
    total = sum(results.values())
    print(f"  Accuracy:   {accuracy}/{total}")
    print(f"  LLM calls:  {3 + len(partial_groups) * 2} "
          f"({len(generic_partials)} classify + {len(partial_groups)} partials × 2 passes)")

    # Assertions
    if results['tp_killed'] > 0:
        print(f"\n  WARNING: {results['tp_killed']} TPs killed!")
    if results['fp_killed'] >= 4:
        print(f"\n  GOOD: {results['fp_killed']}/5 FPs killed")
    else:
        print(f"\n  WARNING: Only {results['fp_killed']}/5 FPs killed")


if __name__ == "__main__":
    main()
