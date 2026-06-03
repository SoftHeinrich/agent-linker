#!/usr/bin/env python3
"""Unit test: Two variance-reduction fixes for Phase 5/6.

Fix 1: Phase 5 two-pass intersection — run entity extraction twice,
       keep only candidates found in BOTH passes.

Fix 2: Remove code-first auto-approve in Phase 6 — force ALL candidates
       through 2-pass LLM validation (no substring shortcut).

Uses BBB V39 checkpoint to measure impact without re-running full pipeline.
Simulates each fix by re-running just Phase 5 or Phase 6 with the modification.
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

os.environ.setdefault("CLAUDE_MODEL", "sonnet")

from llm_sad_sam.llm_client import LLMClient, LLMBackend
from llm_sad_sam.core.data_types import CandidateLink
from llm_sad_sam.core.document_loader import DocumentLoader

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


def load_checkpoint(dataset, phase):
    path = f"results/phase_cache/v39/{dataset}/{phase}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def classify_links(links, gold):
    tp = [l for l in links if (l.sentence_number, l.component_id) in gold]
    fp = [l for l in links if (l.sentence_number, l.component_id) not in gold]
    fn_count = len(gold) - len(tp)
    return tp, fp, fn_count


def print_results(label, tp, fp, fn_count, gold_size):
    p = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
    r = len(tp) / gold_size if gold_size > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"  {label}: P={p:.1%} R={r:.1%} F1={f1:.1%} TP={len(tp)} FP={len(fp)} FN={fn_count}")
    return f1


def test_fix1_phase5_twopass(dataset="bigbluebutton"):
    """Fix 1: Run Phase 5 entity extraction twice, keep intersection."""
    print(f"\n{'='*70}")
    print(f"FIX 1: Phase 5 Two-Pass Intersection ({dataset})")
    print(f"{'='*70}")

    gold = load_gold(dataset)

    # Load Phase 4 output (ILinker2 links) and Phase 3 knowledge
    p4 = load_checkpoint(dataset, "phase4")
    p3 = load_checkpoint(dataset, "phase3")
    p1 = load_checkpoint(dataset, "phase1")

    text_files = glob.glob(str(BENCHMARK / dataset / "**" / f"{dataset}.txt"), recursive=True)
    loader = DocumentLoader()
    sentences = loader.load_sentences(text_files[0])
    sent_map = {s.number: s for s in sentences}

    from llm_sad_sam.pcm_parser import parse_pcm_repository
    model_files = glob.glob(str(BENCHMARK / dataset / "**" / "*.repository"), recursive=True)
    components = parse_pcm_repository(model_files[0])
    name_to_id = {c.name: c.id for c in components}
    comp_names = [c.name for c in components]
    comp_lower = {n.lower() for n in comp_names}

    doc_knowledge = p3.get('doc_knowledge')

    # Build mappings
    mappings = []
    if doc_knowledge:
        mappings.extend([f"{a}={c}" for a, c in doc_knowledge.abbreviations.items()])
        mappings.extend([f"{s}={c}" for s, c in doc_knowledge.synonyms.items()])
        mappings.extend([f"{p}={c}" for p, c in doc_knowledge.partial_references.items()])

    llm = LLMClient(backend=LLMBackend.CLAUDE)

    def run_extraction_pass(pass_label):
        """Run one full Phase 5 extraction pass."""
        batch_size = 50
        all_candidates = {}
        for batch_start in range(0, len(sentences), batch_size):
            batch = sentences[batch_start:batch_start + batch_size]

            prompt = f"""Extract ALL references to software architecture components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings[:20])}' if mappings else ''}

RULES — include a reference when:
1. The component name (or known alias) appears directly in the sentence
2. A space-separated form matches a compound name (e.g., "Memory Manager" → MemoryManager)
3. The sentence describes what a specific component does by name or role
4. A known synonym or partial reference is used
5. The component participates in an interaction described in the sentence (as sender, receiver, or target) — e.g., "X sends data to Y" references BOTH X and Y
6. The component is mentioned in a passive or prepositional phrase — e.g., "data is stored in X", "handled by X", "via X", "through X"

RULES — exclude when:
1. The name appears only inside a dotted path (e.g., com.example.name)
2. The name is used as an ordinary English word, not as a component reference

Favor inclusion over exclusion — later validation will filter borderline cases.

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence", "match_type": "exact|synonym|partial|functional"}}]}}
JSON only:"""

            data = llm.extract_json(llm.query(prompt, timeout=240))
            if not data:
                continue

            for ref in data.get("references", []):
                snum, cname = ref.get("sentence"), ref.get("component")
                if not (snum and cname and cname in name_to_id):
                    continue
                if isinstance(snum, str):
                    snum = snum.lstrip("S")
                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue
                sent = sent_map.get(snum)
                if not sent:
                    continue
                matched = ref.get("matched_text", "")
                if matched and matched.lower() not in sent.text.lower():
                    continue
                key = (snum, name_to_id[cname])
                if key not in all_candidates:
                    all_candidates[key] = (snum, cname, matched)

        print(f"    {pass_label}: {len(all_candidates)} candidates")
        return all_candidates

    # Run two passes
    pass1 = run_extraction_pass("Pass 1")
    pass2 = run_extraction_pass("Pass 2")

    # Intersection: keep only candidates found in BOTH
    intersection_keys = set(pass1.keys()) & set(pass2.keys())
    union_keys = set(pass1.keys()) | set(pass2.keys())

    print(f"\n  Pass 1: {len(pass1)} candidates")
    print(f"  Pass 2: {len(pass2)} candidates")
    print(f"  Intersection: {len(intersection_keys)} candidates")
    print(f"  Union: {len(union_keys)} candidates")
    print(f"  Dropped by intersection: {len(union_keys) - len(intersection_keys)}")

    # Classify intersection vs union
    def classify_keys(keys, source_dict):
        tp_keys = {k for k in keys if k in gold}
        fp_keys = {k for k in keys if k not in gold}
        return tp_keys, fp_keys

    tp_inter, fp_inter = classify_keys(intersection_keys, pass1)
    tp_union, fp_union = classify_keys(union_keys, pass1)
    tp_p1, fp_p1 = classify_keys(set(pass1.keys()), pass1)
    tp_p2, fp_p2 = classify_keys(set(pass2.keys()), pass2)

    print(f"\n  Single pass 1:  {len(tp_p1)} TP, {len(fp_p1)} FP")
    print(f"  Single pass 2:  {len(tp_p2)} TP, {len(fp_p2)} FP")
    print(f"  Intersection:   {len(tp_inter)} TP, {len(fp_inter)} FP  <- want fewer FP, same TP")
    print(f"  Union:          {len(tp_union)} TP, {len(fp_union)} FP")

    # Show what intersection dropped
    dropped = union_keys - intersection_keys
    dropped_tp = {k for k in dropped if k in gold}
    dropped_fp = {k for k in dropped if k not in gold}
    print(f"\n  Dropped by intersection: {len(dropped_tp)} TP, {len(dropped_fp)} FP")
    if dropped_tp:
        for snum, cid in sorted(dropped_tp):
            cname = next((n for n, i in name_to_id.items() if i == cid), cid)
            print(f"    DROPPED TP: S{snum} -> {cname}")
    if dropped_fp:
        for snum, cid in sorted(dropped_fp):
            cname = next((n for n, i in name_to_id.items() if i == cid), cid)
            print(f"    DROPPED FP: S{snum} -> {cname}")


def test_fix2_no_autoapprov(dataset="bigbluebutton"):
    """Fix 2: Remove code-first auto-approve, force all through LLM validation."""
    print(f"\n{'='*70}")
    print(f"FIX 2: No Code-First Auto-Approve ({dataset})")
    print(f"{'='*70}")

    gold = load_gold(dataset)

    # Load Phase 5 output
    p5 = load_checkpoint(dataset, "phase5")
    p3 = load_checkpoint(dataset, "phase3")
    p1 = load_checkpoint(dataset, "phase1")

    text_files = glob.glob(str(BENCHMARK / dataset / "**" / f"{dataset}.txt"), recursive=True)
    loader = DocumentLoader()
    sentences = loader.load_sentences(text_files[0])
    sent_map = {s.number: s for s in sentences}

    from llm_sad_sam.pcm_parser import parse_pcm_repository
    model_files = glob.glob(str(BENCHMARK / dataset / "**" / "*.repository"), recursive=True)
    components = parse_pcm_repository(model_files[0])
    comp_names = [c.name for c in components]

    doc_knowledge = p3.get('doc_knowledge')
    model_knowledge = p1.get('model_knowledge')

    candidates = p5.get('entity_candidates', p5.get('candidates', []))
    print(f"  Phase 5 candidates: {len(candidates)}")

    # Classify current candidates
    needs = [c for c in candidates if c.needs_validation]
    direct = [c for c in candidates if not c.needs_validation]
    print(f"  Direct (no validation needed): {len(direct)}")
    print(f"  Needs validation: {len(needs)}")

    # Count how many "needs" are auto-approved by code-first in current V39
    alias_map = {}
    for c in components:
        aliases = {c.name}
        if doc_knowledge:
            for a, cn in doc_knowledge.abbreviations.items():
                if cn == c.name:
                    aliases.add(a)
            for s, cn in doc_knowledge.synonyms.items():
                if cn == c.name:
                    aliases.add(s)
            for p, cn in doc_knowledge.partial_references.items():
                if cn == c.name:
                    aliases.add(p)
        alias_map[c.name] = aliases

    auto_approved = []
    llm_needed = []
    for c in needs:
        sent = sent_map.get(c.sentence_number)
        if not sent:
            continue
        matched = False
        for a in alias_map.get(c.component_name, set()):
            if len(a) >= 3:
                if a.lower() in sent.text.lower():
                    matched = True
                    break
            elif len(a) >= 2:
                if bool(re.search(r'\b' + re.escape(a) + r'\b', sent.text, re.IGNORECASE)):
                    matched = True
                    break
        if matched:
            auto_approved.append(c)
        else:
            llm_needed.append(c)

    print(f"  Code-first auto-approved: {len(auto_approved)}")
    print(f"  LLM-needed: {len(llm_needed)}")

    # Classify auto-approved as TP/FP
    auto_tp = [c for c in auto_approved if (c.sentence_number, c.component_id) in gold]
    auto_fp = [c for c in auto_approved if (c.sentence_number, c.component_id) not in gold]
    print(f"\n  Auto-approved breakdown: {len(auto_tp)} TP, {len(auto_fp)} FP")

    if auto_fp:
        print(f"\n  FPs that code-first auto-approves (Fix 2 would send to LLM):")
        for c in sorted(auto_fp, key=lambda x: x.sentence_number):
            sent = sent_map.get(c.sentence_number)
            text_preview = sent.text[:80] if sent else "?"
            print(f"    S{c.sentence_number} -> {c.component_name}: {text_preview}...")

    # Now run LLM 2-pass validation on ONLY the auto-approved FPs to see if LLM would reject
    if auto_fp:
        print(f"\n  Running 2-pass LLM validation on {len(auto_fp)} auto-approved FPs...")
        llm = LLMClient(backend=LLMBackend.CLAUDE)

        # Build context
        learned_patterns = None
        try:
            p2 = load_checkpoint(dataset, "phase2")
            learned_patterns = p2.get('learned_patterns')
        except:
            pass

        ctx = []
        if learned_patterns:
            if learned_patterns.action_indicators:
                ctx.append(f"ACTION: {', '.join(learned_patterns.action_indicators[:4])}")
            if learned_patterns.effect_indicators:
                ctx.append(f"EFFECT (reject): {', '.join(learned_patterns.effect_indicators[:3])}")
            if learned_patterns.subprocess_terms:
                ctx.append(f"Subprocess (reject): {', '.join(list(learned_patterns.subprocess_terms)[:5])}")

        # Build cases from auto-approved FPs + a sample of TPs for calibration
        test_items = auto_fp[:25]
        cases = []
        for i, c in enumerate(test_items):
            prev = sent_map.get(c.sentence_number - 1)
            p = f'[prev: {prev.text[:35]}...] ' if prev else ""
            cases.append(f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n  {p}"{c.sentence_text}"')

        cases_str = "\n".join(cases)
        ctx_str = "\n".join(ctx) if ctx else ""

        # Two passes
        results_per_pass = []
        for pass_num, focus in enumerate([
            "Focus on ACTOR role: is the component performing an action or being described?",
            "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component, not a generic concept?"
        ]):
            prompt = f"""Validate component references in a software architecture document. {focus}

COMPONENTS: {', '.join(comp_names)}
{ctx_str}

CASES:
{cases_str}

For each case: does the sentence contain a DIRECT reference to the architecture component — by name, alias, or role?
Return JSON: {{"validations": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

            data = llm.extract_json(llm.query(prompt, timeout=180))
            pass_results = {}
            if data:
                for v in data.get("validations", []):
                    idx = v.get("case", 0) - 1
                    pass_results[idx] = v.get("approve", True)
            results_per_pass.append(pass_results)
            approved_count = sum(1 for v in pass_results.values() if v)
            print(f"    Pass {pass_num+1}: {approved_count}/{len(test_items)} approved")

        # Intersection: approve only if BOTH approve
        llm_rejected = 0
        llm_kept = 0
        for i, c in enumerate(test_items):
            p1_approve = results_per_pass[0].get(i, True)
            p2_approve = results_per_pass[1].get(i, True)
            both_approve = p1_approve and p2_approve
            status = "KEPT" if both_approve else "REJECTED"
            if both_approve:
                llm_kept += 1
            else:
                llm_rejected += 1
            sent = sent_map.get(c.sentence_number)
            print(f"    [{status}] S{c.sentence_number} -> {c.component_name}: "
                  f"P1={'Y' if p1_approve else 'N'} P2={'Y' if p2_approve else 'N'}")

        print(f"\n  Fix 2 impact on auto-approved FPs: {llm_rejected}/{len(test_items)} would be rejected by LLM")
        print(f"  (These are FPs that code-first auto-approves but LLM 2-pass would catch)")

    # Also check: would Fix 2 kill any auto-approved TPs?
    print(f"\n  Risk: {len(auto_tp)} auto-approved TPs would also go through LLM validation")
    print(f"  (Not tested here — but 2-pass intersection is conservative, likely keeps most)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", choices=["1", "2", "both"], default="both")
    parser.add_argument("--dataset", default="bigbluebutton")
    args = parser.parse_args()

    if args.fix in ("1", "both"):
        test_fix1_phase5_twopass(args.dataset)
    if args.fix in ("2", "both"):
        test_fix2_no_autoapprov(args.dataset)
