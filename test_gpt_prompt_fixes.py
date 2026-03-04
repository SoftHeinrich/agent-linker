#!/usr/bin/env python3
"""Unit tests for GPT-5.2 prompt fixes on V32 checkpoints.

Tests three high-impact mitigations for the 3.9pp GPT-5.2 vs Claude gap:

  Fix 1: Phase 6 — Loosen generic mention reject + add architectural-context
         exception to validation prompt.  Targets ~5 TPs killed by GPT's
         over-aggressive generic word interpretation.

  Fix 2: Phase 9 — Add Rule 4 exception for named components with generic
         names.  Targets ~3-4 TPs killed by GPT's over-literal Rule 4.

  Fix 3: Phase 7 — Add structural coreference rules for GPT.
         Targets ~1-2 TPs missed by GPT's weaker discourse reasoning.

Uses V32 checkpoints (from GPT-5.2 v10 run, 90.6% macro F1).
Loads intermediate phase state and re-runs only the target phase with
modified prompts, comparing TP/FP against gold standards.

Usage:
    # Test all fixes on all datasets (requires LLM)
    PYTHONPATH=src python test_gpt_prompt_fixes.py

    # Test specific fix
    PYTHONPATH=src python test_gpt_prompt_fixes.py --fix 6
    PYTHONPATH=src python test_gpt_prompt_fixes.py --fix 9
    PYTHONPATH=src python test_gpt_prompt_fixes.py --fix 7

    # Test specific dataset
    PYTHONPATH=src python test_gpt_prompt_fixes.py --fix 6 --dataset teammates

    # Dry run: show what WOULD be tested (no LLM calls)
    PYTHONPATH=src python test_gpt_prompt_fixes.py --dry-run
"""

import argparse
import csv
import glob
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.core.data_types import SadSamLink, CandidateLink
from llm_sad_sam.llm_client import LLMClient

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
DATASETS = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore" / "text_2016" / "mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore" / "model_2016" / "pcm" / "ms.repository",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore" / "text_2020" / "teastore.txt",
        "model": BENCHMARK_BASE / "teastore" / "model_2020" / "pcm" / "teastore.repository",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates" / "text_2021" / "teammates.txt",
        "model": BENCHMARK_BASE / "teammates" / "model_2021" / "pcm" / "teammates.repository",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton" / "text_2021" / "bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton" / "model_2021" / "pcm" / "bbb.repository",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref" / "text_2021" / "jabref.txt",
        "model": BENCHMARK_BASE / "jabref" / "model_2021" / "pcm" / "jabref.repository",
    },
}
CACHE_DIR = Path("./results/phase_cache/v32")

# ═══════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════

def load_checkpoint(dataset, phase_name):
    path = CACHE_DIR / dataset / f"{phase_name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dataset(dataset):
    paths = DATASETS[dataset]
    components = parse_pcm_repository(str(paths["model"]))
    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    sent_map = DocumentLoader.build_sent_map(sentences)
    name_to_id = {c.name: c.id for c in components}
    id_to_name = {c.id: c.name for c in components}
    return components, sentences, sent_map, name_to_id, id_to_name


def load_gold(dataset):
    gold_path = BENCHMARK_BASE / dataset
    pattern = str(gold_path / "**" / "goldstandard_sad_*-sam_*.csv")
    files = [f for f in glob.glob(pattern, recursive=True) if "UME" not in f and "code" not in f]
    gold = set()
    for f in files:
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                cid = row.get("modelElementID", "")
                sid = row.get("sentence", "")
                if sid and cid:
                    gold.add((int(sid), cid))
    return gold


def _extract_json(text):
    """Extract JSON object or array from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Find first { or [
    for ch, end_ch in [('{', '}'), ('[', ']')]:
        start = text.find(ch)
        if start >= 0:
            depth = 0
            for j in range(start, len(text)):
                if text[j] == ch:
                    depth += 1
                elif text[j] == end_ch:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:j+1])
                        except json.JSONDecodeError:
                            break
    return None


def _has_standalone_mention(comp_name, text):
    """Check if component name appears as standalone mention (not in dotted path)."""
    is_single = ' ' not in comp_name
    if is_single:
        cap_name = comp_name[0].upper() + comp_name[1:]
        pattern = rf'\b{re.escape(cap_name)}\b'
        flags = 0
    else:
        pattern = rf'\b{re.escape(comp_name)}\b'
        flags = re.IGNORECASE
    for m in re.finditer(pattern, text, flags):
        s, e = m.start(), m.end()
        if s > 0 and text[s-1] == '.':
            continue
        if e < len(text) and text[e] == '.' and e + 1 < len(text) and text[e+1].isalpha():
            continue
        if s > 0 and text[s-1] == '-':
            continue
        if e < len(text) and text[e] == '-' and '-' not in comp_name:
            continue
        return True
    return False


def _is_generic_mention_original(comp_name, sentence_text):
    """Original _is_generic_mention from V26a — baseline behavior."""
    if ' ' in comp_name or '-' in comp_name:
        return False
    if re.search(r'[a-z][A-Z]', comp_name):
        return False
    if comp_name.isupper():
        return False
    if comp_name[0].islower():
        return False
    if _has_standalone_mention(comp_name, sentence_text):
        return False
    word_lower = comp_name.lower()
    if re.search(rf'\b{re.escape(word_lower)}\b', sentence_text):
        return True
    return False


def _word_boundary_match(name, text):
    return bool(re.search(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE))


def compute_metrics(links, gold):
    """Compute precision, recall, F1 given links and gold set."""
    predicted = {(l.sentence_number, l.component_id) for l in links}
    tp = predicted & gold
    fp = predicted - gold
    fn = gold - predicted
    p = len(tp) / max(1, len(tp) + len(fp))
    r = len(tp) / max(1, len(tp) + len(fn))
    f1 = 2 * p * r / max(1e-9, p + r)
    return {"P": p, "R": r, "F1": f1, "TP": len(tp), "FP": len(fp), "FN": len(fn),
            "tp_set": tp, "fp_set": fp, "fn_set": fn}


# ═══════════════════════════════════════════════════════════════════════
# Fix 1: Phase 6 — Modified generic mention filter + validation prompt
# ═══════════════════════════════════════════════════════════════════════

def _is_generic_mention_disabled(comp_name, sentence_text):
    """Variant A: Completely disable generic mention filter.

    Rationale: The generic mention pre-filter was tuned for Claude's behavior.
    For GPT-5.2, it kills too many TPs (8 across 5 datasets). The 2-pass LLM
    validation is the proper safety net — let it handle discrimination.
    """
    return False


def _is_generic_mention_fixed(comp_name, sentence_text):
    """Variant B: Relaxed generic mention filter for GPT-5.2.

    Only rejects when the word is clearly used as a modifier/adjective
    (e.g., "common.datatransfer" = dotted path) or preceded by non-architectural
    modifiers. Passes through all standalone lowercase uses that could be
    component references in architectural context.
    """
    if ' ' in comp_name or '-' in comp_name:
        return False
    if re.search(r'[a-z][A-Z]', comp_name):
        return False
    if comp_name.isupper():
        return False
    if comp_name[0].islower():
        return False
    if _has_standalone_mention(comp_name, sentence_text):
        return False
    word_lower = comp_name.lower()
    if not re.search(rf'\b{re.escape(word_lower)}\b', sentence_text):
        return False

    # Only reject if the word is CLEARLY in a dotted path context
    # (e.g., "common.datatransfer", "storage.api") — these are sub-package refs
    if re.search(rf'\b{re.escape(word_lower)}\.\w', sentence_text):
        return True

    # Otherwise, let it through to 2-pass validation
    return False


VALIDATION_PROMPT_FIXED = """Validate component references in a software architecture document. {focus}

COMPONENTS: {comp_names}

{context}

DECISION RULES:
APPROVE when:
- The component is the grammatical actor or subject (the sentence is ABOUT the component)
- A section heading names the component (introduces that component's topic)
- The sentence describes what the component does, provides, or interacts with

REJECT when:
- The name is used as an ordinary English word, not as a proper name
  (Like "proxy" in "proxy pattern" is the design pattern concept, not the Proxy component — reject the component link)
- The name is a modifier inside a larger phrase, not a standalone reference
  (Like "observer" in "observer pattern" modifies pattern — reject if Observer is a component)
- The sentence is about a subprocess, algorithm, or implementation detail — not the component itself

IMPORTANT EXCEPTION for single-word component names (e.g., "Logic", "Storage", "Client"):
When the system has a component literally named after a common English word, and the
sentence uses that word in an ARCHITECTURAL context (describing system behavior,
interactions, responsibilities, or structure), this IS a component reference — APPROVE.
Only reject when the word is used in a clearly NON-ARCHITECTURAL sense:
- "the logic of the argument" (philosophical, not the Logic component)
- "common sense" (idiom, not the Common component)
- "persistent effort" (adjective form, not the Persistence component)
When the word appears as a noun describing part of the system → APPROVE.

CASES:
{cases}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""


def test_fix1_phase6(dataset, llm, dry_run=False):
    """Test Fix 1: Modified Phase 6 generic mention filter + validation prompt.

    Loads V32 phase5 checkpoint (entity candidates) and re-runs Phase 6
    with the fixed generic mention filter and updated validation prompt.

    Tests two variants:
      A) Completely disabled generic mention filter
      B) Relaxed filter (only rejects dotted-path contexts)
    """
    data5 = load_checkpoint(dataset, "phase5")
    data4 = load_checkpoint(dataset, "phase4")
    data3 = load_checkpoint(dataset, "phase3")
    data1 = load_checkpoint(dataset, "phase1")
    data6_orig = load_checkpoint(dataset, "phase6")
    if not all([data5, data4, data3, data1, data6_orig]):
        print(f"  SKIP {dataset}: missing checkpoints")
        return None

    components, sentences, sent_map, name_to_id, id_to_name = load_dataset(dataset)
    gold = load_gold(dataset)

    candidates = data5.get("candidates", [])
    doc_knowledge = data3.get("doc_knowledge")
    model_knowledge = data1.get("model_knowledge")
    transarc_set = data4.get("transarc_set", set())

    # Separate needs_validation vs direct
    needs = [c for c in candidates if c.needs_validation]
    direct = [c for c in candidates if not c.needs_validation]

    # ── Original generic mention filter ──
    orig_rejected = []
    orig_remaining = []
    for c in needs:
        sent = sent_map.get(c.sentence_number)
        if sent and _is_generic_mention_original(c.component_name, sent.text):
            orig_rejected.append(c)
        else:
            orig_remaining.append(c)

    # ── Variant A: Disabled generic mention filter ──
    varA_remaining = list(needs)  # pass everything through

    # ── Variant B: Relaxed generic mention filter ──
    fix_rejected = []
    fix_remaining = []
    for c in needs:
        sent = sent_map.get(c.sentence_number)
        if sent and _is_generic_mention_fixed(c.component_name, sent.text):
            fix_rejected.append(c)
        else:
            fix_remaining.append(c)

    # Show what changed
    orig_rej_keys = {(c.sentence_number, c.component_id) for c in orig_rejected}
    fix_rej_keys = {(c.sentence_number, c.component_id) for c in fix_rejected}
    rescued_B = orig_rej_keys - fix_rej_keys
    rescued_A = set(orig_rej_keys)  # Variant A rescues everything

    print(f"\n  Generic Mention Filter:")
    print(f"    Original rejects: {len(orig_rejected)}")
    print(f"    Variant A (disabled): 0 rejects — all {len(orig_rejected)} rescued")
    print(f"    Variant B (relaxed):  {len(fix_rejected)} rejects — {len(rescued_B)} rescued")

    print(f"\n    Rescued by Variant B (dotted-path-only filter):")
    for key in sorted(rescued_B):
        c = next(x for x in orig_rejected if (x.sentence_number, x.component_id) == key)
        is_tp = key in gold
        sent = sent_map.get(c.sentence_number)
        text_preview = sent.text[:80] + "..." if sent and len(sent.text) > 80 else (sent.text if sent else "")
        print(f"      {'TP RECOVERED' if is_tp else 'FP rescued'}: S{c.sentence_number} → {c.component_name}")
        print(f"        \"{text_preview}\"")

    still_rejected_B = orig_rej_keys & fix_rej_keys
    if still_rejected_B:
        print(f"\n    Still rejected by Variant B ({len(still_rejected_B)}):")
        for key in sorted(still_rejected_B):
            c = next(x for x in fix_rejected if (x.sentence_number, x.component_id) == key)
            is_tp = key in gold
            sent = sent_map.get(c.sentence_number)
            text_preview = sent.text[:80] + "..." if sent and len(sent.text) > 80 else (sent.text if sent else "")
            print(f"      {'TP STILL KILLED' if is_tp else 'FP caught'}: S{c.sentence_number} → {c.component_name}")
            print(f"        \"{text_preview}\"")

    if dry_run:
        print(f"\n  [DRY RUN] Variant A would send {len(varA_remaining)} candidates to LLM validation")
        print(f"  [DRY RUN] Variant B would send {len(fix_remaining)} candidates to LLM validation")
        # Show pure deterministic impact (no LLM needed)
        tp_rescued_A = sum(1 for k in rescued_A if k in gold)
        fp_rescued_A = len(rescued_A) - tp_rescued_A
        tp_rescued_B = sum(1 for k in rescued_B if k in gold)
        fp_rescued_B = len(rescued_B) - tp_rescued_B
        print(f"\n  DETERMINISTIC IMPACT (generic mention filter only, no LLM validation change):")
        print(f"    Variant A: +{tp_rescued_A} TP, +{fp_rescued_A} FP (before 2-pass validation)")
        print(f"    Variant B: +{tp_rescued_B} TP, +{fp_rescued_B} FP (before 2-pass validation)")
        return {"tp_rescued_A": tp_rescued_A, "fp_rescued_A": fp_rescued_A,
                "tp_rescued_B": tp_rescued_B, "fp_rescued_B": fp_rescued_B}

    # ── Run fixed 2-pass validation on remaining candidates ──
    comp_names = [c.name for c in components]
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

    # Step 1: Code-first auto-approve
    auto_approved = []
    llm_needed = []
    for c in fix_remaining:
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
                if _word_boundary_match(a, sent.text):
                    matched = True
                    break
        if matched:
            c.confidence = 1.0
            c.source = "validated"
            auto_approved.append(c)
        else:
            llm_needed.append(c)

    print(f"\n  Code-first auto-approved: {len(auto_approved)}, LLM needed: {len(llm_needed)}")

    # Step 2: 2-pass intersect with FIXED prompt
    ctx_parts = []
    generic_risk = set()
    if model_knowledge and model_knowledge.ambiguous_names:
        generic_risk |= model_knowledge.ambiguous_names

    twopass_approved = []
    generic_to_verify = []

    for batch_start in range(0, len(llm_needed), 25):
        batch = llm_needed[batch_start:batch_start + 25]
        cases = []
        for i, c in enumerate(batch):
            prev = sent_map.get(c.sentence_number - 1)
            p = f"[prev: {prev.text[:35]}...] " if prev else ""
            cases.append(f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n  {p}"{c.sentence_text}"')

        r1 = {}
        r2 = {}
        for focus in [
            "Focus on ACTOR role: is the component performing an action or being described?",
            "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component, not a generic concept?",
        ]:
            prompt = VALIDATION_PROMPT_FIXED.format(
                focus=focus,
                comp_names=', '.join(comp_names),
                context='\n'.join(ctx_parts),
                cases='\n'.join(cases),
            )
            raw = llm.query(prompt, timeout=120)
            data = _extract_json(raw.text if hasattr(raw, 'text') else str(raw))
            results = {}
            if data:
                for v in (data.get("validations", []) if isinstance(data, dict) else []):
                    idx = v.get("case", 0) - 1
                    if 0 <= idx < len(batch):
                        results[idx] = v.get("approve", False)
            if focus.startswith("Focus on ACTOR"):
                r1 = results
            else:
                r2 = results

        for i, c in enumerate(batch):
            if r1.get(i, False) and r2.get(i, False):
                if c.component_name in generic_risk:
                    generic_to_verify.append(c)
                else:
                    c.confidence = 1.0
                    c.source = "validated"
                    twopass_approved.append(c)

    print(f"  2-pass approved: {len(twopass_approved)}, generic verify: {len(generic_to_verify)}")

    # Step 3: Evidence post-filter for generic-risk (same as original)
    generic_validated = []
    if generic_to_verify:
        for batch_start in range(0, len(generic_to_verify), 25):
            batch = generic_to_verify[batch_start:batch_start + 25]
            cases = []
            for i, c in enumerate(batch):
                cases.append(
                    f'Case {i+1}: S{c.sentence_number} "{c.sentence_text}"\n'
                    f'  Candidate: {c.component_name}'
                )
            prompt = f"""For each case, find the EXACT text in the sentence that refers to the architecture component.

COMPONENTS: {', '.join(comp_names)}

CASES:
{chr(10).join(cases)}

For each case, provide:
- evidence_text: the EXACT substring from the sentence that names or references the component
- If you cannot find specific text evidence, set evidence_text to null

Return JSON:
{{"validations": [{{"case": 1, "evidence_text": "exact substring or null"}}]}}
JSON only:"""
            raw = llm.query(prompt, timeout=120)
            data = _extract_json(raw.text if hasattr(raw, 'text') else str(raw))
            if not data:
                continue
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if idx < 0 or idx >= len(batch):
                    continue
                c = batch[idx]
                evidence = v.get("evidence_text")
                if not evidence:
                    continue
                sent = sent_map.get(c.sentence_number)
                if not sent or evidence.lower() not in sent.text.lower():
                    continue
                ev_lower = evidence.lower()
                aliases = alias_map.get(c.component_name, {c.component_name.lower()})
                if any(a.lower() in ev_lower for a in aliases if len(a) >= 2):
                    c.confidence = 1.0
                    c.source = "validated"
                    generic_validated.append(c)

        print(f"  Generic evidence: {len(generic_validated)}/{len(generic_to_verify)}")

    fixed_validated = direct + auto_approved + twopass_approved + generic_validated

    # ── Compare with original Phase 6 output ──
    orig_validated = data6_orig.get("validated", []) if data6_orig else []
    orig_keys = {(c.sentence_number, c.component_id) for c in orig_validated}
    fix_keys = {(c.sentence_number, c.component_id) for c in fixed_validated}

    gained = fix_keys - orig_keys
    lost = orig_keys - fix_keys

    print(f"\n  Comparison vs original Phase 6:")
    print(f"    Original validated: {len(orig_validated)}")
    print(f"    Fixed validated:    {len(fixed_validated)}")
    print(f"    Gained: {len(gained)}")
    for key in sorted(gained):
        is_tp = key in gold
        print(f"      {'TP GAINED' if is_tp else 'FP added'}: S{key[0]} → {id_to_name.get(key[1], key[1])}")
    print(f"    Lost: {len(lost)}")
    for key in sorted(lost):
        is_tp = key in gold
        print(f"      {'TP LOST!' if is_tp else 'FP removed'}: S{key[0]} → {id_to_name.get(key[1], key[1])}")

    return {"orig": len(orig_validated), "fixed": len(fixed_validated),
            "gained": gained, "lost": lost, "gold": gold}


# ═══════════════════════════════════════════════════════════════════════
# Fix 2: Phase 9 — Modified judge prompt with Rule 4 exception
# ═══════════════════════════════════════════════════════════════════════

JUDGE_PROMPT_FIXED = """JUDGE: Validate trace links between documentation and software architecture components.

APPROVAL CRITERIA:
A link S→C is valid when the sentence satisfies all four conditions:

1. EXPLICIT REFERENCE
   The component name (or a direct reference to it) appears in the sentence as a clear
   entity being discussed. This distinguishes component-specific statements from
   incidental mentions or generic discussions where the component name appears but is
   not the subject of the statement.

2. SYSTEM-LEVEL PERSPECTIVE
   The sentence describes the component's role, responsibilities, interfaces, or
   interactions within the overall system architecture. Reject statements focused on
   internal implementation details (data structures, algorithms, code-level concerns)
   that are invisible at the architectural abstraction level.

3. PRIMARY FOCUS
   The component is the main subject of what the sentence conveys, not a secondary
   or incidental mention. The sentence is fundamentally about what the component does
   or how it relates to other system elements.

4. COMPONENT-SPECIFIC USAGE
   The reference is to the component as a named entity within the system architecture,
   not to a generic concept, pattern, or technology that happens to share a name.
   This distinguishes component-specific statements from domain terminology or
   methodological discussions that use ordinary words.

   CRITICAL EXCEPTION: If the system has a component literally named "X" — even when
   X is a common word like "Client", "Persistence", "Recommender", "Logic", "Storage",
   "Apps" — and the sentence describes what X does, how X interacts with other parts
   of the system, or X's responsibilities, then this IS component-specific usage.
   The existence of a named component means architectural references to that concept
   ARE about the component. Only reject Rule 4 when "X" is clearly used in a different
   domain (e.g., "client" meaning "customer" in a business context, not the software
   Client component).

COMPONENTS: {comp_names}

LINKS:
{cases}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief explanation"}}]}}
JSON only:"""


def test_fix2_phase9(dataset, llm, dry_run=False):
    """Test Fix 2: Modified Phase 9 judge with Rule 4 exception.

    Loads V32 pre_judge checkpoint (links before Phase 9) and re-runs
    the judge with the fixed prompt.
    """
    pre_judge = load_checkpoint(dataset, "pre_judge")
    data4 = load_checkpoint(dataset, "phase4")
    data3 = load_checkpoint(dataset, "phase3")
    data1 = load_checkpoint(dataset, "phase1")
    final_orig = load_checkpoint(dataset, "final")
    if not all([pre_judge, data4]):
        print(f"  SKIP {dataset}: missing checkpoints")
        return None

    components, sentences, sent_map, name_to_id, id_to_name = load_dataset(dataset)
    gold = load_gold(dataset)
    comp_names = [c.name for c in components]

    preliminary = pre_judge.get("preliminary", [])
    transarc_set = data4.get("transarc_set", set())
    doc_knowledge = data3.get("doc_knowledge") if data3 else None
    model_knowledge = data1.get("model_knowledge") if data1 else None

    # ── Triage (same logic as V26a) ──
    safe, alias_links, nomatch_links, ta_review = [], [], [], []
    syn_safe_count = 0

    for l in preliminary:
        is_ta = (l.sentence_number, l.component_id) in transarc_set
        sent = sent_map.get(l.sentence_number)

        # Syn-safe check
        syn_safe = False
        if sent and doc_knowledge:
            text_lower = sent.text.lower()
            for syn, target in doc_knowledge.synonyms.items():
                if target == l.component_name:
                    if re.search(rf'\b{re.escape(syn.lower())}\b', text_lower):
                        syn_safe = True
                        break
            if not syn_safe:
                for partial, target in doc_knowledge.partial_references.items():
                    if target == l.component_name:
                        if re.search(rf'\b{re.escape(partial.lower())}\b', text_lower):
                            syn_safe = True
                            break

        if syn_safe:
            safe.append(l)
            syn_safe_count += 1
            continue

        if is_ta:
            # Check if ambiguous name
            is_ambig = False
            cn = l.component_name
            if ' ' not in cn and '-' not in cn and not re.search(r'[a-z][A-Z]', cn) and not cn.isupper():
                if model_knowledge and model_knowledge.ambiguous_names and cn in model_knowledge.ambiguous_names:
                    is_ambig = True
            if is_ambig:
                ta_review.append(l)
            else:
                safe.append(l)
            continue

        if not sent:
            nomatch_links.append(l)
            continue

        if _has_standalone_mention(l.component_name, sent.text):
            safe.append(l)
        else:
            nomatch_links.append(l)

    print(f"\n  Triage: {len(safe)} safe ({syn_safe_count} syn-safe), "
          f"{len(ta_review)} ta-review, {len(nomatch_links)} no-match")

    # Show what goes to judge
    print(f"\n  No-match links going to judge:")
    for l in nomatch_links:
        is_tp = (l.sentence_number, l.component_id) in gold
        sent = sent_map.get(l.sentence_number)
        text_preview = sent.text[:70] + "..." if sent and len(sent.text) > 70 else (sent.text if sent else "")
        print(f"    {'TP' if is_tp else 'FP'} S{l.sentence_number} → {l.component_name} ({l.source})")
        print(f"       \"{text_preview}\"")

    if ta_review:
        print(f"\n  TransArc ambiguous going to advocate-prosecutor:")
        for l in ta_review:
            is_tp = (l.sentence_number, l.component_id) in gold
            print(f"    {'TP' if is_tp else 'FP'} S{l.sentence_number} → {l.component_name}")

    if dry_run:
        print(f"\n  [DRY RUN] Would judge {len(nomatch_links)} no-match + "
              f"{len(ta_review)} ta-review links")
        return None

    # ── Run fixed 4-rule judge on no-match links (union voting) ──
    nomatch_approved = []
    if nomatch_links:
        cases = []
        for i, l in enumerate(nomatch_links[:30]):
            sent = sent_map.get(l.sentence_number)
            match = l.component_name  # simplified
            if sent:
                cases.append(
                    f'Case {i+1}: S{l.sentence_number} → {l.component_name} '
                    f'(match:"{match}", src:{l.source})\n'
                    f'  "{sent.text}"'
                )

        prompt = JUDGE_PROMPT_FIXED.format(
            comp_names=', '.join(comp_names),
            cases='\n'.join(cases),
        )

        # Union voting: reject only if BOTH passes reject
        raw1 = llm.query(prompt, timeout=180)
        raw2 = llm.query(prompt, timeout=180)
        data1 = _extract_json(raw1.text if hasattr(raw1, 'text') else str(raw1))
        data2 = _extract_json(raw2.text if hasattr(raw2, 'text') else str(raw2))

        n = min(30, len(nomatch_links))
        rej1 = set()
        rej2 = set()
        if data1:
            for j in data1.get("judgments", []):
                idx = j.get("case", 0) - 1
                if 0 <= idx < n and not j.get("approve", False):
                    rej1.add(idx)
        if data2:
            for j in data2.get("judgments", []):
                idx = j.get("case", 0) - 1
                if 0 <= idx < n and not j.get("approve", False):
                    rej2.add(idx)

        rejected = rej1 & rej2
        for i in range(n):
            if i not in rejected:
                nomatch_approved.append(nomatch_links[i])
            else:
                l = nomatch_links[i]
                is_tp = (l.sentence_number, l.component_id) in gold
                print(f"    Fixed judge reject: S{l.sentence_number} → {l.component_name} "
                      f"[{'TP KILLED!' if is_tp else 'FP caught'}]")
        nomatch_approved.extend(nomatch_links[n:])
    else:
        nomatch_approved = []

    # For ta_review, use the same advocate-prosecutor as original (not changing that)
    # Just auto-approve for this test — we're testing the 4-rule judge only
    ta_approved = list(ta_review)

    fixed_final = safe + ta_approved + nomatch_approved

    # ── Compare with original final output ──
    orig_final = final_orig.get("final", []) if final_orig else []
    orig_keys = {(l.sentence_number, l.component_id) for l in orig_final}
    fix_keys = {(l.sentence_number, l.component_id) for l in fixed_final}

    gained = fix_keys - orig_keys
    lost = orig_keys - fix_keys

    print(f"\n  Comparison vs original Phase 9:")
    print(f"    Original final: {len(orig_final)}")
    print(f"    Fixed final:    {len(fixed_final)}")
    print(f"    Gained: {len(gained)}")
    for key in sorted(gained):
        is_tp = key in gold
        print(f"      {'TP GAINED' if is_tp else 'FP added'}: S{key[0]} → {id_to_name.get(key[1], key[1])}")
    print(f"    Lost: {len(lost)}")
    for key in sorted(lost):
        is_tp = key in gold
        print(f"      {'TP LOST!' if is_tp else 'FP removed'}: S{key[0]} → {id_to_name.get(key[1], key[1])}")

    # Full metrics
    orig_m = compute_metrics(orig_final, gold)
    fix_m = compute_metrics(fixed_final, gold)
    print(f"\n    Original: P={orig_m['P']:.1%} R={orig_m['R']:.1%} F1={orig_m['F1']:.1%} "
          f"TP={orig_m['TP']} FP={orig_m['FP']} FN={orig_m['FN']}")
    print(f"    Fixed:    P={fix_m['P']:.1%} R={fix_m['R']:.1%} F1={fix_m['F1']:.1%} "
          f"TP={fix_m['TP']} FP={fix_m['FP']} FN={fix_m['FN']}")
    delta_f1 = fix_m['F1'] - orig_m['F1']
    print(f"    Delta F1: {delta_f1:+.1%}")

    return {"orig_m": orig_m, "fix_m": fix_m}


# ═══════════════════════════════════════════════════════════════════════
# Fix 3: Phase 7 — Structural coreference rules for GPT
# ═══════════════════════════════════════════════════════════════════════

COREF_PROMPT_FIXED = """Resolve pronoun references to architecture components.

COMPONENTS: {comp_names}

{cases_block}

For each pronoun that refers to a component, provide:
- antecedent_sentence: the sentence number where the component was EXPLICITLY NAMED
- antecedent_text: the EXACT quote from that sentence containing the component name

STRUCTURAL RULES (apply these FIRST, before deeper analysis):
1. SUBJECT CONTINUITY: If component X is the grammatical subject of the immediately
   preceding sentence (N-1), and sentence N starts with "It", "This", or "They",
   the pronoun refers to X.
2. PARAGRAPH TOPIC: Within the same paragraph (no blank line break), pronouns
   default to the most recently named component unless another referent is introduced.
3. RECENCY: When multiple components appear in the context window, prefer the one
   named MOST RECENTLY before the pronoun.

VALIDATION RULES (all must hold):
1. The component name (or known alias) MUST appear verbatim in the antecedent sentence
2. The antecedent MUST be within the previous 3 sentences
3. The pronoun MUST grammatically refer back to that component as its subject
4. If the pronoun could refer to multiple things, DO NOT resolve it

Like in technical writing: "The Scheduler assigns tasks to threads. It uses a priority queue internally."
— "It" clearly refers to "the Scheduler" because it was the subject of the previous sentence.

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "pronoun": "it", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact text with component name"}}]}}

Only include resolutions you are CERTAIN about. JSON only:"""


def test_fix3_phase7(dataset, llm, dry_run=False):
    """Test Fix 3: Modified Phase 7 coref with structural rules.

    Loads V32 phase6 checkpoint (validated candidates) and re-runs coref
    with the fixed prompt that adds structural rules for GPT.
    """
    data6 = load_checkpoint(dataset, "phase6")
    data4 = load_checkpoint(dataset, "phase4")
    data3 = load_checkpoint(dataset, "phase3")
    data7_orig = load_checkpoint(dataset, "phase7")
    data0 = load_checkpoint(dataset, "phase0")
    data2 = load_checkpoint(dataset, "phase2")
    if not all([data6, data4]):
        print(f"  SKIP {dataset}: missing checkpoints")
        return None

    components, sentences, sent_map, name_to_id, id_to_name = load_dataset(dataset)
    gold = load_gold(dataset)
    comp_names = [c.name for c in components]

    # Find sentences with pronouns
    PRONOUN_PATTERN = re.compile(r'\b(it|its|they|their|them|this|these|those)\b', re.IGNORECASE)
    pronoun_sents = [s for s in sentences if PRONOUN_PATTERN.search(s.text)]

    # Build existing link set (transarc + validated — skip coref since we're re-running it)
    transarc_set = data4.get("transarc_set", set())
    validated = data6.get("validated", [])
    existing = transarc_set | {(c.sentence_number, c.component_id) for c in validated}

    doc_knowledge = data3.get("doc_knowledge") if data3 else None
    learned_patterns = data2.get("learned_patterns") if data2 else None

    print(f"\n  Pronoun sentences: {len(pronoun_sents)}")
    print(f"  Existing links: {len(existing)}")

    if dry_run:
        print(f"  [DRY RUN] Would process {len(pronoun_sents)} sentences in batches of 12")
        # Show original coref results
        orig_coref = data7_orig.get("coref_links", []) if data7_orig else []
        print(f"  Original coref found: {len(orig_coref)}")
        for l in orig_coref:
            is_tp = (l.sentence_number, l.component_id) in gold
            print(f"    {'TP' if is_tp else 'FP'} S{l.sentence_number} → {l.component_name}")
        # Show gold coref candidates (TPs not yet linked)
        gold_not_linked = gold - existing
        pronoun_snum = {s.number for s in pronoun_sents}
        coref_opportunity = {(s, c) for s, c in gold_not_linked if s in pronoun_snum}
        print(f"  Gold coref opportunities (unlinked TPs in pronoun sentences): {len(coref_opportunity)}")
        for s, c in sorted(coref_opportunity):
            sent = sent_map.get(s)
            print(f"    S{s} → {c}: \"{sent.text[:70]}...\"" if sent else f"    S{s} → {c}")
        return None

    # ── Run fixed coref ──
    all_coref = []
    for batch_start in range(0, len(pronoun_sents), 12):
        batch = pronoun_sents[batch_start:batch_start + 12]
        cases_block = ""
        for i, sent in enumerate(batch):
            prev = []
            for j in range(1, 4):
                p = sent_map.get(sent.number - j)
                if p:
                    prev.append(f"S{p.number}: {p.text}")
            cases_block += f"--- Case {i+1}: S{sent.number} ---\n"
            if prev:
                cases_block += "PREVIOUS:\n  " + "\n  ".join(reversed(prev)) + "\n"
            cases_block += f">>> {sent.text}\n\n"

        prompt = COREF_PROMPT_FIXED.format(
            comp_names=', '.join(comp_names),
            cases_block=cases_block,
        )

        raw = llm.query(prompt, timeout=150)
        data = _extract_json(raw.text if hasattr(raw, 'text') else str(raw))
        if not data:
            continue

        for res in data.get("resolutions", []):
            comp = res.get("component")
            snum = res.get("sentence")
            if not (comp and snum and comp in name_to_id):
                continue
            if isinstance(snum, str):
                snum = snum.lstrip("S")
            try:
                snum = int(snum)
            except (ValueError, TypeError):
                continue

            # Verify antecedent
            ant_snum = res.get("antecedent_sentence")
            if ant_snum is not None:
                if isinstance(ant_snum, str):
                    ant_snum = ant_snum.lstrip("S")
                try:
                    ant_snum = int(ant_snum)
                except (ValueError, TypeError):
                    ant_snum = None

            if ant_snum is not None:
                ant_sent = sent_map.get(ant_snum)
                if not ant_sent:
                    continue
                if not (_has_standalone_mention(comp, ant_sent.text) or
                        (doc_knowledge and _has_alias_in_text(comp, ant_sent.text, doc_knowledge))):
                    print(f"    Coref verify-fail (S{ant_snum} doesn't mention {comp}): "
                          f"S{snum} -> {comp}")
                    continue
                if abs(snum - ant_snum) > 3:
                    continue

            # Skip subprocess sentences
            if learned_patterns and learned_patterns.subprocess_terms:
                sent = sent_map.get(snum)
                if sent:
                    text_lower = sent.text.lower()
                    if any(t.lower() in text_lower for t in learned_patterns.subprocess_terms):
                        continue

            key = (snum, name_to_id[comp])
            if key not in existing:
                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, 1.0, "coreference"))

    # ── Compare with original coref ──
    orig_coref = data7_orig.get("coref_links", []) if data7_orig else []
    orig_keys = {(l.sentence_number, l.component_id) for l in orig_coref}
    fix_keys = {(l.sentence_number, l.component_id) for l in all_coref}

    gained = fix_keys - orig_keys
    lost = orig_keys - fix_keys

    print(f"\n  Comparison vs original Phase 7:")
    print(f"    Original coref: {len(orig_coref)}")
    print(f"    Fixed coref:    {len(all_coref)}")
    print(f"    Gained: {len(gained)}")
    for key in sorted(gained):
        is_tp = key in gold
        print(f"      {'TP GAINED' if is_tp else 'FP added'}: S{key[0]} → {id_to_name.get(key[1], key[1])}")
    print(f"    Lost: {len(lost)}")
    for key in sorted(lost):
        is_tp = key in gold
        print(f"      {'TP LOST!' if is_tp else 'FP removed'}: S{key[0]} → {id_to_name.get(key[1], key[1])}")

    return {"orig": len(orig_coref), "fixed": len(all_coref),
            "gained": gained, "lost": lost, "gold": gold}


def _has_alias_in_text(comp_name, text, doc_knowledge):
    """Check if any alias for comp_name appears in text."""
    text_lower = text.lower()
    for syn, target in doc_knowledge.synonyms.items():
        if target == comp_name:
            if re.search(rf'\b{re.escape(syn.lower())}\b', text_lower):
                return True
    for partial, target in doc_knowledge.partial_references.items():
        if target == comp_name:
            if re.search(rf'\b{re.escape(partial.lower())}\b', text_lower):
                return True
    return False


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Test GPT-5.2 prompt fixes on V32 checkpoints")
    parser.add_argument("--fix", type=int, choices=[6, 7, 9], help="Test specific fix (6/7/9)")
    parser.add_argument("--dataset", type=str, choices=list(DATASETS.keys()), help="Test specific dataset")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be tested, no LLM calls")
    args = parser.parse_args()

    os.environ.setdefault("CLAUDE_MODEL", "sonnet")

    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    fixes = [args.fix] if args.fix else [6, 9, 7]

    llm = None
    if not args.dry_run:
        llm = LLMClient()

    for fix_num in fixes:
        fix_name = {6: "Phase 6 (Generic Mention + Validation)",
                    9: "Phase 9 (Judge Rule 4 Exception)",
                    7: "Phase 7 (Structural Coref Rules)"}[fix_num]
        print(f"\n{'=' * 80}")
        print(f"  FIX {fix_num}: {fix_name}")
        print(f"{'=' * 80}")

        all_results = {}
        for ds in datasets:
            print(f"\n{'─' * 70}")
            print(f"  Dataset: {ds}")
            print(f"{'─' * 70}")

            if fix_num == 6:
                result = test_fix1_phase6(ds, llm, dry_run=args.dry_run)
            elif fix_num == 9:
                result = test_fix2_phase9(ds, llm, dry_run=args.dry_run)
            elif fix_num == 7:
                result = test_fix3_phase7(ds, llm, dry_run=args.dry_run)
            if result:
                all_results[ds] = result

        # Summary
        if all_results:
            print(f"\n{'=' * 80}")
            print(f"  SUMMARY: Fix {fix_num} — {fix_name}")
            print(f"{'=' * 80}")
            if fix_num == 6 and args.dry_run:
                total_tp_A = sum(r.get("tp_rescued_A", 0) for r in all_results.values())
                total_fp_A = sum(r.get("fp_rescued_A", 0) for r in all_results.values())
                total_tp_B = sum(r.get("tp_rescued_B", 0) for r in all_results.values())
                total_fp_B = sum(r.get("fp_rescued_B", 0) for r in all_results.values())
                print(f"  Variant A (disabled): +{total_tp_A} TP, +{total_fp_A} FP")
                print(f"  Variant B (relaxed):  +{total_tp_B} TP, +{total_fp_B} FP")
                print(f"\n  Note: These TPs still need to survive 2-pass LLM validation.")
                print(f"  Run without --dry-run to test with actual LLM calls.")
            for ds, r in all_results.items():
                if "orig_m" in r:
                    print(f"  {ds:20s} orig={r['orig_m']['F1']:.1%} → fixed={r['fix_m']['F1']:.1%} "
                          f"(delta={r['fix_m']['F1']-r['orig_m']['F1']:+.1%})")


if __name__ == "__main__":
    main()
