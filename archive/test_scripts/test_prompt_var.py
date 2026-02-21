#!/usr/bin/env python3
"""Test prompt engineering changes for variance reduction.

Injects V18 intermediate state, runs ONLY the target phase N times,
and measures output variance + gold metrics.

Usage:
    python test_prompt_var.py --phase 5 --dataset teastore --runs 3
    python test_prompt_var.py --phase 7 --dataset teammates --runs 3
    python test_prompt_var.py --phase 6 --dataset teammates --runs 3
"""

import copy
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.core import (
    DocumentLoader, SadSamLink, CandidateLink,
    DocumentProfile, LearnedThresholds, ModelKnowledge,
    DocumentKnowledge, LearnedPatterns, DiscourseContext,
)

os.environ["CLAUDE_MODEL"] = "sonnet"

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
CLI_RESULTS = Path("/mnt/hostshare/ardoco-home/cli-results")

DATASETS = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
        "gold_sam": BENCHMARK_BASE / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
        "transarc_sam": CLI_RESULTS / "mediastore-sad-sam/sadSamTlr_mediastore.csv",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
        "model": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
        "gold_sam": BENCHMARK_BASE / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
        "transarc_sam": CLI_RESULTS / "teastore-sad-sam/sadSamTlr_teastore.csv",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
        "gold_sam": BENCHMARK_BASE / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc_sam": CLI_RESULTS / "teammates-sad-sam/sadSamTlr_teammates.csv",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
        "gold_sam": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc_sam": CLI_RESULTS / "bigbluebutton-sad-sam/sadSamTlr_bigbluebutton.csv",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
        "gold_sam": BENCHMARK_BASE / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc_sam": CLI_RESULTS / "jabref-sad-sam/sadSamTlr_jabref.csv",
    },
}


def load_gold_sam(gold_path):
    links = set()
    with open(gold_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def inject_v18_state(linker, dataset_name, paths, sentences, components, sent_map, name_to_id, id_to_name):
    """Inject V18 intermediate state (Phases 0-4) into linker without LLM calls."""
    # Phase 0: Document profile (no LLM)
    linker.doc_profile = linker._learn_document_profile(sentences, components)
    linker._is_complex = linker._structural_complexity(sentences, components)
    linker.thresholds = LearnedThresholds(0, 0, 0, 0, "qualitative", 0)
    linker._cached_sent_map = sent_map

    # Phase 1: Use LAST V18 log for model knowledge (avoid LLM call)
    linker.model_knowledge = linker._analyze_model(components)

    # Phase 2: Pattern learning (LLM — needed for subprocess terms)
    linker.learned_patterns = linker._learn_patterns_with_debate(sentences, components)

    # Phase 3: Doc knowledge (LLM — needed for alias info)
    linker.doc_knowledge = linker._learn_document_knowledge_enriched(sentences, components)

    # Phase 3b: Multiword partials (no LLM)
    linker._enrich_multiword_partials(sentences, components)

    # Phase 4: TransArc (no LLM)
    transarc_links = linker._process_transarc(
        str(paths["transarc_sam"]), id_to_name, sent_map, name_to_id
    )
    transarc_set = {(l.sentence_number, l.component_id) for l in transarc_links}

    return transarc_links, transarc_set


# ═══════════════════════════════════════════════════════════════════════
# NEW PHASE 5: Tighter extraction prompt
# ═══════════════════════════════════════════════════════════════════════

def phase5_new(linker, sentences, components, name_to_id, sent_map):
    """Phase 5 v2: Balanced — keep recall, add matched_text verification + retry."""
    comp_names = linker._get_comp_names(components)
    comp_lower = {n.lower() for n in comp_names}

    mappings = []
    if linker.doc_knowledge:
        mappings.extend([f"{a}={c}" for a, c in linker.doc_knowledge.abbreviations.items()])
        mappings.extend([f"{s}={c}" for s, c in linker.doc_knowledge.synonyms.items()])
        mappings.extend([f"{p}={c}" for p, c in linker.doc_knowledge.partial_references.items()])

    batch_size = 100
    all_candidates = {}

    for batch_start in range(0, len(sentences), batch_size):
        batch = sentences[batch_start:batch_start + batch_size]

        if len(sentences) > batch_size:
            print(f"    Entity batch {batch_start//batch_size + 1}: "
                  f"S{batch[0].number}-S{batch[-1].number} ({len(batch)} sents)")

        prompt = f"""Extract references to software architecture components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings[:20])}' if mappings else ''}

WHAT TO LOOK FOR:
- DIRECT MENTIONS: The component name or a known alias appears in the sentence (exact or case-insensitive)
- CamelCase splits: "Data Manager" in text may refer to component "DataManager"
- COMPONENT AS ACTOR: The sentence describes what a SPECIFIC component does by name or alias
  GOOD: "The persistence layer stores entity data" → refers to a persistence component
  BAD: "Data is persisted to disk" → passive, no component named

For each match, provide the "matched_text" — the EXACT substring from the sentence that triggered the match.

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N, "component": "Name", "matched_text": "exact substring from sentence", "match_type": "exact|synonym|partial|functional"}}]}}
JSON only:"""

        # Try up to 2 times on empty response
        for attempt in range(2):
            data = linker.llm.extract_json(linker.llm.query(prompt, timeout=150))
            if data and data.get("references"):
                break
            if attempt == 0:
                print(f"    Empty response, retrying batch...")

        if not data:
            continue

        for ref in data.get("references", []):
            snum, cname = ref.get("sentence"), ref.get("component")
            if not (snum and cname and cname in name_to_id):
                continue
            sent = sent_map.get(snum)
            if not sent or linker._in_dotted_path(sent.text, cname):
                continue

            # Verify matched_text is actually in sentence (reject hallucinations)
            matched = ref.get("matched_text", "")
            if matched and matched.lower() not in sent.text.lower():
                continue

            matched_lower = matched.lower() if matched else ""
            is_exact = matched_lower in comp_lower or cname.lower() in matched_lower
            is_generic_here = linker._is_generic_mention(cname, sent.text)
            needs_val = not is_exact or ref.get("match_type") != "exact" or is_generic_here

            key = (snum, name_to_id[cname])
            if key not in all_candidates:
                all_candidates[key] = CandidateLink(snum, sent.text, cname, name_to_id[cname],
                                           matched, 0.85, "entity",
                                           ref.get("match_type", "exact"), needs_val)

    return list(all_candidates.values())


# ═══════════════════════════════════════════════════════════════════════
# NEW PHASE 7: Require antecedent citation
# ═══════════════════════════════════════════════════════════════════════

def phase7_new_discourse(linker, sentences, components, name_to_id, sent_map, discourse_model):
    """Phase 7 discourse mode with required antecedent citation."""
    comp_names = linker._get_comp_names(components)
    all_coref = []
    PRONOUN_PATTERN = re.compile(r'\b(it|they|this|these|that|those|its|their)\b', re.IGNORECASE)
    pronoun_sents = [s for s in sentences if PRONOUN_PATTERN.search(s.text)]

    for batch_start in range(0, len(pronoun_sents), 12):
        batch = pronoun_sents[batch_start:batch_start + 12]
        cases = []
        for sent in batch:
            ctx = discourse_model.get(sent.number, DiscourseContext())
            prev = []
            for i in range(1, 4):
                p = sent_map.get(sent.number - i)
                if p:
                    prev.append(f"S{p.number}: {p.text}")
            cases.append({"sent": sent, "ctx": ctx, "prev": prev})

        prompt = f"""Resolve pronoun references to architecture components.

COMPONENTS: {', '.join(comp_names)}

"""
        for i, case in enumerate(cases):
            prompt += f"--- Case {i+1}: S{case['sent'].number} ---\n"
            if case["prev"]:
                prompt += "PREVIOUS:\n  " + "\n  ".join(reversed(case["prev"])) + "\n"
            prompt += f">>> {case['sent'].text}\n\n"

        prompt += """For each pronoun that refers to a component, you MUST provide:
- The antecedent_sentence number where the component was EXPLICITLY NAMED
- The antecedent_text: the EXACT text from that sentence containing the component name

STRICT RULES:
- The component name (or known alias) MUST appear in the antecedent sentence
- The antecedent sentence MUST be within the previous 2 sentences
- The pronoun MUST be the grammatical subject referring back to that component
- If unsure, DO NOT include the resolution

Return JSON:
{"resolutions": [{"case": 1, "sentence": N, "pronoun": "it", "component": "Name", "antecedent_sentence": M, "antecedent_text": "exact text with component name"}]}

Only include resolutions you are CERTAIN about. JSON only:"""

        data = linker.llm.extract_json(linker.llm.query(prompt, timeout=150))
        if not data:
            continue

        for res in data.get("resolutions", []):
            comp = res.get("component")
            snum = res.get("sentence")
            if not (comp and snum and comp in name_to_id):
                continue
            try:
                snum = int(snum)
            except (ValueError, TypeError):
                continue

            # Verify antecedent citation
            ant_snum = res.get("antecedent_sentence")
            ant_text = res.get("antecedent_text", "")
            if ant_snum is not None:
                try:
                    ant_snum = int(ant_snum)
                except (ValueError, TypeError):
                    ant_snum = None

            if ant_snum is not None:
                ant_sent = sent_map.get(ant_snum)
                if not ant_sent:
                    print(f"    Coref skip (bad antecedent S{ant_snum}): S{snum} -> {comp}")
                    continue
                # Verify component actually appears in antecedent
                if not (linker._has_standalone_mention(comp, ant_sent.text) or
                        linker._has_alias_mention(comp, ant_sent.text)):
                    print(f"    Coref skip (comp not in antecedent S{ant_snum}): S{snum} -> {comp}")
                    continue
                # Verify distance
                if abs(snum - ant_snum) > 3:
                    print(f"    Coref skip (antecedent too far S{ant_snum}): S{snum} -> {comp}")
                    continue

            sent = sent_map.get(snum)
            if sent and linker.learned_patterns and linker.learned_patterns.is_subprocess(sent.text):
                continue
            all_coref.append(SadSamLink(snum, name_to_id[comp], comp, 1.0, "coreference"))

    return all_coref


def phase7_new_debate(linker, sentences, components, name_to_id, sent_map):
    """Phase 7 debate mode with required antecedent citation."""
    comp_names = linker._get_comp_names(components)
    all_coref = []
    CONTEXT_WINDOW = 3

    ctx = []
    if linker.learned_patterns and linker.learned_patterns.subprocess_terms:
        ctx.append(f"Subprocesses (don't link): {', '.join(list(linker.learned_patterns.subprocess_terms)[:5])}")

    for batch_start in range(0, len(sentences), 20):
        batch = sentences[batch_start:min(batch_start + 20, len(sentences))]
        ctx_start = max(0, batch_start - CONTEXT_WINDOW)
        ctx_sents = sentences[ctx_start:batch_start + 20]
        doc_lines = [
            f"{'*' if s.number >= batch[0].number else ' '}S{s.number}: {s.text}"
            for s in ctx_sents
        ]

        prompt1 = f"""Resolve pronoun references to architecture components.

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

DOCUMENT (* = analyze these sentences):
{chr(10).join(doc_lines)}

Find pronouns (it, they, this, these) in starred sentences that refer to a component.

STRICT RULES:
- You MUST cite the antecedent_sentence where the component was EXPLICITLY NAMED
- The component name (or known alias) MUST appear verbatim in the antecedent sentence
- The antecedent MUST be within the previous 2 sentences
- Do NOT resolve pronouns in sentences about subprocesses or implementation details

Return JSON:
{{"resolutions": [{{"sentence": N, "pronoun": "it", "component": "Name", "antecedent_sentence": M, "antecedent_text": "exact quote with component name"}}]}}

Only include resolutions you are CERTAIN about. JSON only:"""

        data1 = linker.llm.extract_json(linker.llm.query(prompt1, timeout=100))
        if not data1:
            continue
        proposed = data1.get("resolutions", [])
        if not proposed:
            continue

        # Verify antecedent citations in code
        verified = []
        for res in proposed:
            comp = res.get("component")
            snum = res.get("sentence")
            if not (comp and snum and comp in name_to_id):
                continue
            try:
                snum = int(snum)
            except (ValueError, TypeError):
                continue

            ant_snum = res.get("antecedent_sentence")
            if ant_snum is not None:
                try:
                    ant_snum = int(ant_snum)
                except (ValueError, TypeError):
                    ant_snum = None

            if ant_snum is not None:
                ant_sent = sent_map.get(ant_snum)
                if not ant_sent:
                    continue
                if not (linker._has_standalone_mention(comp, ant_sent.text) or
                        linker._has_alias_mention(comp, ant_sent.text)):
                    print(f"    Coref verify-fail (S{ant_snum} doesn't mention {comp}): S{snum} -> {comp}")
                    continue
                if abs(snum - ant_snum) > 3:
                    continue

            sent = sent_map.get(snum)
            if sent and linker.learned_patterns and linker.learned_patterns.is_subprocess(sent.text):
                continue
            verified.append(res)

        for res in verified:
            snum = int(res["sentence"])
            comp = res["component"]
            all_coref.append(SadSamLink(snum, name_to_id[comp], comp, 1.0, "coreference"))

    return all_coref


# ═══════════════════════════════════════════════════════════════════════
# PHASE 6 VARIANTS
# ═══════════════════════════════════════════════════════════════════════

def _phase6_pre_filter(linker, candidates, sent_map):
    """Shared pre-filtering: separate direct vs needs-validation, reject generic mentions."""
    needs = [c for c in candidates if c.needs_validation]
    direct = [c for c in candidates if not c.needs_validation]
    remaining = []
    for c in needs:
        sent = sent_map.get(c.sentence_number)
        if sent and linker._is_generic_mention(c.component_name, sent.text):
            print(f"    Generic mention reject: S{c.sentence_number} -> {c.component_name}")
        else:
            remaining.append(c)
    return direct, remaining


def phase6_new(linker, candidates, components, sent_map):
    """Phase 6 variant A: Evidence-citation validation.

    LLM cites exact text evidence for each candidate. Code verifies the
    evidence actually appears in the sentence. No approve/reject — code decides.
    """
    if not candidates:
        return []

    comp_names = linker._get_comp_names(components)
    direct, needs = _phase6_pre_filter(linker, candidates, sent_map)
    if not needs:
        return candidates

    # Build alias lookup for code verification
    alias_map = {}  # component_name -> set of valid names/aliases
    for c in components:
        aliases = {c.name, c.name.lower()}
        if linker.doc_knowledge:
            for a, cn in linker.doc_knowledge.abbreviations.items():
                if cn == c.name:
                    aliases.add(a.lower())
            for s, cn in linker.doc_knowledge.synonyms.items():
                if cn == c.name:
                    aliases.add(s.lower())
            for p, cn in linker.doc_knowledge.partial_references.items():
                if cn == c.name:
                    aliases.add(p.lower())
        alias_map[c.name] = aliases

    validated = []
    for batch_start in range(0, len(needs), 25):
        batch = needs[batch_start:batch_start + 25]
        cases = []
        for i, c in enumerate(batch):
            cases.append(f'Case {i+1}: S{c.sentence_number} "{c.sentence_text}"\n  Candidate: {c.component_name}')

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

        data = linker.llm.extract_json(linker.llm.query(prompt, timeout=120))
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
            if not sent:
                continue
            # Code verification 1: evidence actually in sentence
            if evidence.lower() not in sent.text.lower():
                continue
            # Code verification 2: evidence contains component name or alias
            ev_lower = evidence.lower()
            aliases = alias_map.get(c.component_name, {c.component_name.lower()})
            if any(a in ev_lower for a in aliases if len(a) >= 2):
                c.confidence = 1.0
                c.source = "validated"
                validated.append(c)

    return direct + validated


def phase6_var_b(linker, candidates, components, sent_map):
    """Phase 6 variant B: Code-first + LLM-fallback.

    Auto-approve when component name/alias appears as substring in sentence text.
    Only use LLM for remaining ambiguous candidates.
    """
    if not candidates:
        return []

    comp_names = linker._get_comp_names(components)
    direct, needs = _phase6_pre_filter(linker, candidates, sent_map)
    if not needs:
        return candidates

    # Build alias lookup
    alias_map = {}
    for c in components:
        aliases = {c.name.lower()}
        if linker.doc_knowledge:
            for a, cn in linker.doc_knowledge.abbreviations.items():
                if cn == c.name and len(a) >= 3:
                    aliases.add(a.lower())
            for s, cn in linker.doc_knowledge.synonyms.items():
                if cn == c.name:
                    aliases.add(s.lower())
            for p, cn in linker.doc_knowledge.partial_references.items():
                if cn == c.name and len(p) >= 3:
                    aliases.add(p.lower())
        alias_map[c.name] = aliases

    # Step 1: Code-first — auto-approve if name/alias in sentence
    auto_approved = []
    llm_needed = []
    for c in needs:
        sent = sent_map.get(c.sentence_number)
        if not sent:
            continue
        text_lower = sent.text.lower()
        aliases = alias_map.get(c.component_name, set())
        if any(a in text_lower for a in aliases):
            c.confidence = 1.0
            c.source = "validated"
            auto_approved.append(c)
        else:
            llm_needed.append(c)

    print(f"    Code-first auto-approved: {len(auto_approved)}, LLM needed: {len(llm_needed)}")

    # Step 2: LLM validation for remaining (functional/implicit references)
    llm_validated = []
    if llm_needed:
        ctx = []
        if linker.learned_patterns:
            if linker.learned_patterns.action_indicators:
                ctx.append(f"ACTION: {', '.join(linker.learned_patterns.action_indicators[:4])}")
            if linker.learned_patterns.effect_indicators:
                ctx.append(f"EFFECT (reject): {', '.join(linker.learned_patterns.effect_indicators[:3])}")
            if linker.learned_patterns.subprocess_terms:
                ctx.append(f"Subprocess (reject): {', '.join(list(linker.learned_patterns.subprocess_terms)[:5])}")

        for batch_start in range(0, len(llm_needed), 25):
            batch = llm_needed[batch_start:batch_start + 25]
            cases = []
            for i, c in enumerate(batch):
                prev = sent_map.get(c.sentence_number - 1)
                p = f"[prev: {prev.text[:35]}...] " if prev else ""
                cases.append(f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n  {p}"{c.sentence_text}"')

            # Use V18's 2-pass intersect for LLM-needed cases
            r1 = _validation_pass(linker, comp_names, ctx, cases,
                "Focus on ACTOR role: is the component performing an action or being described?")
            r2 = _validation_pass(linker, comp_names, ctx, cases,
                "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component, not a generic concept?")

            for i, c in enumerate(batch):
                if r1.get(i, False) and r2.get(i, False):
                    c.confidence = 1.0
                    c.source = "validated"
                    llm_validated.append(c)

    return direct + auto_approved + llm_validated


def phase6_var_c(linker, candidates, components, sent_map):
    """Phase 6 variant C: Negative-framing validation.

    Ask for rejection REASONS instead of approve/reject. If no valid reason → approve.
    Single pass, deterministic framing.
    """
    if not candidates:
        return []

    comp_names = linker._get_comp_names(components)
    direct, needs = _phase6_pre_filter(linker, candidates, sent_map)
    if not needs:
        return candidates

    validated = []
    for batch_start in range(0, len(needs), 25):
        batch = needs[batch_start:batch_start + 25]
        cases = []
        for i, c in enumerate(batch):
            cases.append(f'Case {i+1}: S{c.sentence_number} "{c.sentence_text}"\n  Claim: refers to {c.component_name} (matched: "{c.matched_text}")')

        prompt = f"""Review claims that sentences reference architecture components. Your job is to find REJECTION reasons.

COMPONENTS: {', '.join(comp_names)}

REJECTION REASONS:
A. GENERIC_WORD: The matched text is a generic English word used in its ordinary sense, not naming the architecture component
   (e.g., "the routing logic" where "logic" is generic, NOT the Logic component)
B. WRONG_REFERENT: The matched text refers to something else that shares the name
   (e.g., "client-side rendering" refers to browser behavior, NOT a Client component)
C. NO_REFERENCE: The sentence does not actually describe, mention, or reference this component at all

If NONE of these reasons apply, the claim is valid.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "reject_reason": "A" or "B" or "C" or null}}]}}
Set reject_reason to null if the claim is valid. JSON only:"""

        data = linker.llm.extract_json(linker.llm.query(prompt, timeout=120))
        if not data:
            continue

        for v in data.get("validations", []):
            idx = v.get("case", 0) - 1
            if idx < 0 or idx >= len(batch):
                continue
            reason = v.get("reject_reason")
            if not reason:  # null = approved
                c = batch[idx]
                c.confidence = 1.0
                c.source = "validated"
                validated.append(c)

    return direct + validated


def phase6_var_d(linker, candidates, components, sent_map):
    """Phase 6 variant D: Code-first + evidence-citation fallback.

    Hybrid of B (code-first) + A (evidence-citation for LLM part).
    Auto-approve when name/alias in text. For remaining, ask LLM to cite evidence.
    """
    if not candidates:
        return []

    comp_names = linker._get_comp_names(components)
    direct, needs = _phase6_pre_filter(linker, candidates, sent_map)
    if not needs:
        return candidates

    # Build alias lookup
    alias_map = {}
    for c in components:
        aliases = {c.name.lower()}
        if linker.doc_knowledge:
            for a, cn in linker.doc_knowledge.abbreviations.items():
                if cn == c.name and len(a) >= 3:
                    aliases.add(a.lower())
            for s, cn in linker.doc_knowledge.synonyms.items():
                if cn == c.name:
                    aliases.add(s.lower())
            for p, cn in linker.doc_knowledge.partial_references.items():
                if cn == c.name and len(p) >= 3:
                    aliases.add(p.lower())
        alias_map[c.name] = aliases

    # Step 1: Code-first auto-approve
    auto_approved = []
    llm_needed = []
    for c in needs:
        sent = sent_map.get(c.sentence_number)
        if not sent:
            continue
        text_lower = sent.text.lower()
        aliases = alias_map.get(c.component_name, set())
        if any(a in text_lower for a in aliases):
            c.confidence = 1.0
            c.source = "validated"
            auto_approved.append(c)
        else:
            llm_needed.append(c)

    print(f"    Code-first auto-approved: {len(auto_approved)}, LLM needed: {len(llm_needed)}")

    # Step 2: Evidence-citation for remaining (like variant A)
    evidence_validated = []
    if llm_needed:
        for batch_start in range(0, len(llm_needed), 25):
            batch = llm_needed[batch_start:batch_start + 25]
            cases = []
            for i, c in enumerate(batch):
                cases.append(f'Case {i+1}: S{c.sentence_number} "{c.sentence_text}"\n  Candidate: {c.component_name}')

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

            data = linker.llm.extract_json(linker.llm.query(prompt, timeout=120))
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
                if not sent:
                    continue
                if evidence.lower() not in sent.text.lower():
                    continue
                ev_lower = evidence.lower()
                aliases = alias_map.get(c.component_name, {c.component_name.lower()})
                if any(a in ev_lower for a in aliases if len(a) >= 2):
                    c.confidence = 1.0
                    c.source = "validated"
                    evidence_validated.append(c)

    return direct + auto_approved + evidence_validated


def phase6_var_g(linker, candidates, components, sent_map):
    """Phase 6 variant G: Code-first + strict evidence + single YES/NO fallback.

    Three tiers:
    1. Code-first: name/alias in sentence → auto-approve (deterministic)
    2. Evidence-citation: LLM cites evidence, code verifies name in evidence (deterministic)
    3. Single YES/NO pass: for remaining, simple binary question (minimal LLM variance)
    """
    if not candidates:
        return []
    comp_names = linker._get_comp_names(components)
    direct, auto_approved, llm_needed, alias_map = _code_first_split(linker, candidates, components, sent_map)
    if not llm_needed:
        return candidates if not auto_approved else direct + auto_approved

    # Tier 2: Evidence-citation with strict name check
    evidence_validated = []
    still_needed = []
    for batch_start in range(0, len(llm_needed), 25):
        batch = llm_needed[batch_start:batch_start + 25]
        cases = []
        for i, c in enumerate(batch):
            cases.append(f'Case {i+1}: S{c.sentence_number} "{c.sentence_text}"\n  Candidate: {c.component_name}')

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

        data = linker.llm.extract_json(linker.llm.query(prompt, timeout=120))
        evidence_results = {}
        if data:
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(batch):
                    evidence_results[idx] = v.get("evidence_text")

        for i, c in enumerate(batch):
            evidence = evidence_results.get(i)
            if not evidence:
                still_needed.append(c)
                continue
            sent = sent_map.get(c.sentence_number)
            if not sent:
                continue
            if evidence.lower() not in sent.text.lower():
                still_needed.append(c)
                continue
            # Strict: name/alias in evidence
            ev_lower = evidence.lower()
            aliases = alias_map.get(c.component_name, {c.component_name.lower()})
            if any(a in ev_lower for a in aliases if len(a) >= 2):
                c.confidence = 1.0
                c.source = "validated"
                evidence_validated.append(c)
            else:
                still_needed.append(c)

    print(f"    Evidence validated: {len(evidence_validated)}, still need YES/NO: {len(still_needed)}")

    # Tier 3: Single YES/NO pass for remaining
    yesno_validated = []
    if still_needed:
        for batch_start in range(0, len(still_needed), 25):
            batch = still_needed[batch_start:batch_start + 25]
            cases = []
            for i, c in enumerate(batch):
                cases.append(
                    f'Case {i+1}: Does sentence S{c.sentence_number} describe what {c.component_name} does?\n'
                    f'  Sentence: "{c.sentence_text}"'
                )

            prompt = f"""Answer YES or NO for each case. A sentence "describes what a component does" ONLY if:
- The component is the SUBJECT or ACTOR performing a specific action
- The sentence is about THIS component specifically, not a generic concept
- Generic English words used in their ordinary sense do NOT count

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(cases)}

Return JSON:
{{"answers": [{{"case": 1, "answer": "YES" or "NO"}}]}}
JSON only:"""

            data = linker.llm.extract_json(linker.llm.query(prompt, timeout=120))
            if not data:
                continue

            for v in data.get("answers", []):
                idx = v.get("case", 0) - 1
                if idx < 0 or idx >= len(batch):
                    continue
                if v.get("answer", "").upper() == "YES":
                    c = batch[idx]
                    c.confidence = 0.9
                    c.source = "validated"
                    yesno_validated.append(c)

    print(f"    YES/NO validated: {len(yesno_validated)}")
    return direct + auto_approved + evidence_validated + yesno_validated


def _word_boundary_match(name, text):
    """Check if name appears as standalone word in text (word-boundary match)."""
    return bool(re.search(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE))


def _code_first_split_v2(linker, candidates, components, sent_map):
    """Enhanced code-first: word-boundary matching (catches short names like UI, DB)."""
    direct, needs = _phase6_pre_filter(linker, candidates, sent_map)
    if not needs:
        return direct, [], [], {}

    alias_map = {}
    for c in components:
        # Include ALL aliases (even short ones) for word-boundary matching
        aliases = {c.name}
        if linker.doc_knowledge:
            for a, cn in linker.doc_knowledge.abbreviations.items():
                if cn == c.name:
                    aliases.add(a)
            for s, cn in linker.doc_knowledge.synonyms.items():
                if cn == c.name:
                    aliases.add(s)
            for p, cn in linker.doc_knowledge.partial_references.items():
                if cn == c.name:
                    aliases.add(p)
        alias_map[c.name] = aliases

    auto_approved = []
    llm_needed = []
    for c in needs:
        sent = sent_map.get(c.sentence_number)
        if not sent:
            continue
        aliases = alias_map.get(c.component_name, set())
        matched = False
        for a in aliases:
            if len(a) >= 3:
                # Substring match OK for longer names
                if a.lower() in sent.text.lower():
                    matched = True
                    break
            elif len(a) >= 2:
                # Word-boundary match for short names (UI, DB)
                if _word_boundary_match(a, sent.text):
                    matched = True
                    break
        if matched:
            c.confidence = 1.0
            c.source = "validated"
            auto_approved.append(c)
        else:
            llm_needed.append(c)

    print(f"    Code-first v2 auto-approved: {len(auto_approved)}, LLM needed: {len(llm_needed)}")
    return direct, auto_approved, llm_needed, alias_map


def phase6_var_h(linker, candidates, components, sent_map):
    """Phase 6 variant H: Enhanced code-first (word-boundary) + strict evidence.

    Better code-first matching catches short names (UI, DB) as standalone words.
    Then strict evidence-citation for remaining.
    """
    if not candidates:
        return []
    comp_names = linker._get_comp_names(components)
    direct, auto_approved, llm_needed, alias_map = _code_first_split_v2(linker, candidates, components, sent_map)
    if not llm_needed:
        return candidates if not auto_approved else direct + auto_approved

    # Strict evidence-citation for remaining (same as D)
    evidence_validated = []
    for batch_start in range(0, len(llm_needed), 25):
        batch = llm_needed[batch_start:batch_start + 25]
        cases = []
        for i, c in enumerate(batch):
            cases.append(f'Case {i+1}: S{c.sentence_number} "{c.sentence_text}"\n  Candidate: {c.component_name}')

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

        data = linker.llm.extract_json(linker.llm.query(prompt, timeout=120))
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
            if not sent:
                continue
            if evidence.lower() not in sent.text.lower():
                continue
            ev_lower = evidence.lower()
            aliases = alias_map.get(c.component_name, {c.component_name.lower()})
            if any(a.lower() in ev_lower for a in aliases if len(a) >= 2):
                c.confidence = 1.0
                c.source = "validated"
                evidence_validated.append(c)

    return direct + auto_approved + evidence_validated


def phase6_var_i(linker, candidates, components, sent_map):
    """Phase 6 variant I: Word-boundary code-first + 2-pass intersect for LLM-needed.

    Best of H (stable code-first) + B (high-recall 2-pass intersect).
    More auto-approved → fewer LLM calls → less variance than B.
    """
    if not candidates:
        return []
    comp_names = linker._get_comp_names(components)
    direct, auto_approved, llm_needed, alias_map = _code_first_split_v2(linker, candidates, components, sent_map)
    if not llm_needed:
        return candidates if not auto_approved else direct + auto_approved

    # 2-pass intersect for remaining (same as V18/variant B)
    ctx = []
    if linker.learned_patterns:
        if linker.learned_patterns.action_indicators:
            ctx.append(f"ACTION: {', '.join(linker.learned_patterns.action_indicators[:4])}")
        if linker.learned_patterns.effect_indicators:
            ctx.append(f"EFFECT (reject): {', '.join(linker.learned_patterns.effect_indicators[:3])}")
        if linker.learned_patterns.subprocess_terms:
            ctx.append(f"Subprocess (reject): {', '.join(list(linker.learned_patterns.subprocess_terms)[:5])}")

    llm_validated = []
    for batch_start in range(0, len(llm_needed), 25):
        batch = llm_needed[batch_start:batch_start + 25]
        cases = []
        for i, c in enumerate(batch):
            prev = sent_map.get(c.sentence_number - 1)
            p = f"[prev: {prev.text[:35]}...] " if prev else ""
            cases.append(f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n  {p}"{c.sentence_text}"')

        r1 = _validation_pass(linker, comp_names, ctx, cases,
            "Focus on ACTOR role: is the component performing an action or being described?")
        r2 = _validation_pass(linker, comp_names, ctx, cases,
            "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component, not a generic concept?")

        for i, c in enumerate(batch):
            if r1.get(i, False) and r2.get(i, False):
                c.confidence = 1.0
                c.source = "validated"
                llm_validated.append(c)

    print(f"    LLM 2-pass validated: {len(llm_validated)} of {len(llm_needed)}")
    return direct + auto_approved + llm_validated


def phase6_var_j(linker, candidates, components, sent_map):
    """Phase 6 variant J: Word-boundary code-first + negative-framing for LLM-needed.

    Code-first deterministic approval + single-pass rejection-reason for remaining.
    Negative-framing was too permissive in variant C (7 FP on TS), but with code-first
    handling most candidates, fewer go to LLM.
    """
    if not candidates:
        return []
    comp_names = linker._get_comp_names(components)
    direct, auto_approved, llm_needed, alias_map = _code_first_split_v2(linker, candidates, components, sent_map)
    if not llm_needed:
        return candidates if not auto_approved else direct + auto_approved

    # Single negative-framing pass for remaining
    llm_validated = []
    for batch_start in range(0, len(llm_needed), 25):
        batch = llm_needed[batch_start:batch_start + 25]
        cases = []
        for i, c in enumerate(batch):
            cases.append(f'Case {i+1}: S{c.sentence_number} "{c.sentence_text}"\n  Claim: refers to {c.component_name} (matched: "{c.matched_text}")')

        prompt = f"""Review claims that sentences reference architecture components. Find REJECTION reasons.

COMPONENTS: {', '.join(comp_names)}

REJECTION REASONS:
A. GENERIC_WORD: The matched text is a generic English word used in its ordinary sense, NOT naming the architecture component
   (e.g., "the routing logic" where "logic" is generic, NOT the Logic component)
B. WRONG_REFERENT: The matched text refers to something else that shares the name
   (e.g., "client-side rendering" refers to browser behavior, NOT a Client component)
C. NO_REFERENCE: The sentence does not describe, mention, or reference this component at all
   — the sentence must describe what this SPECIFIC component does or is

If NONE of these reasons apply, the claim is valid.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "reject_reason": "A" or "B" or "C" or null}}]}}
Set reject_reason to null if the claim is valid. JSON only:"""

        data = linker.llm.extract_json(linker.llm.query(prompt, timeout=120))
        if not data:
            continue

        for v in data.get("validations", []):
            idx = v.get("case", 0) - 1
            if idx < 0 or idx >= len(batch):
                continue
            reason = v.get("reject_reason")
            if not reason:
                c = batch[idx]
                c.confidence = 1.0
                c.source = "validated"
                llm_validated.append(c)

    print(f"    Negative-framing validated: {len(llm_validated)} of {len(llm_needed)}")
    return direct + auto_approved + llm_validated


def phase6_var_k(linker, candidates, components, sent_map):
    """Phase 6 variant K: Word-boundary code-first + evidence with functional accept.

    Like H but relaxed evidence: accept if evidence contains name/alias OR
    if evidence is a substantial functional description (>=15 chars, no name needed).
    """
    if not candidates:
        return []
    comp_names = linker._get_comp_names(components)
    direct, auto_approved, llm_needed, alias_map = _code_first_split_v2(linker, candidates, components, sent_map)
    if not llm_needed:
        return candidates if not auto_approved else direct + auto_approved

    evidence_validated = []
    for batch_start in range(0, len(llm_needed), 25):
        batch = llm_needed[batch_start:batch_start + 25]
        cases = []
        for i, c in enumerate(batch):
            cases.append(f'Case {i+1}: S{c.sentence_number} "{c.sentence_text}"\n  Candidate: {c.component_name}')

        prompt = f"""For each case, determine if the sentence references the architecture component.

COMPONENTS: {', '.join(comp_names)}

CASES:
{chr(10).join(cases)}

For each case provide:
- reference_type: "naming" if the component name/alias appears in the text,
  "functional" if the sentence describes what this SPECIFIC component does without naming it,
  or null if no reference exists
- evidence_text: the EXACT substring from the sentence (for naming: the name/alias, for functional: the description)

IMPORTANT: "functional" means the component is the ACTOR performing a specific action described in the sentence.
Generic actions like "handles data" or "processes requests" do NOT count unless they are specific to this component.

Return JSON:
{{"validations": [{{"case": 1, "reference_type": "naming"|"functional"|null, "evidence_text": "exact substring or null"}}]}}
JSON only:"""

        data = linker.llm.extract_json(linker.llm.query(prompt, timeout=120))
        if not data:
            continue

        for v in data.get("validations", []):
            idx = v.get("case", 0) - 1
            if idx < 0 or idx >= len(batch):
                continue
            c = batch[idx]
            ref_type = v.get("reference_type")
            evidence = v.get("evidence_text")
            if not ref_type or not evidence:
                continue
            sent = sent_map.get(c.sentence_number)
            if not sent:
                continue
            # Anti-hallucination: evidence must exist in sentence
            if evidence.lower() not in sent.text.lower():
                continue

            if ref_type == "naming":
                # Strict: name/alias must appear in evidence
                ev_lower = evidence.lower()
                aliases = alias_map.get(c.component_name, {c.component_name.lower()})
                if any(a.lower() in ev_lower for a in aliases if len(a) >= 2):
                    c.confidence = 1.0
                    c.source = "validated"
                    evidence_validated.append(c)
            elif ref_type == "functional":
                # Accept if evidence is substantial (>=15 chars = real description, not just a word)
                if len(evidence) >= 15:
                    c.confidence = 0.9
                    c.source = "validated"
                    evidence_validated.append(c)

    print(f"    Evidence validated: {len(evidence_validated)} of {len(llm_needed)}")
    return direct + auto_approved + evidence_validated


def phase6_var_l(linker, candidates, components, sent_map):
    """Phase 6 variant L: Word-boundary code-first + split LLM by generic risk.

    For LLM-needed candidates:
    - Non-generic components → 2-pass intersect (high recall, stable for specific names)
    - Generic-risk components → strict evidence-citation (deterministic, avoids FPs)
    """
    if not candidates:
        return []
    comp_names = linker._get_comp_names(components)
    direct, auto_approved, llm_needed, alias_map = _code_first_split_v2(linker, candidates, components, sent_map)
    if not llm_needed:
        return candidates if not auto_approved else direct + auto_approved

    # Classify generic-risk: single word, common English, or in ambiguous_names
    GENERIC_WORDS = {"logic", "storage", "common", "client", "model", "action", "data", "service", "server"}
    generic_risk = set()
    if linker.model_knowledge and linker.model_knowledge.ambiguous_names:
        generic_risk |= linker.model_knowledge.ambiguous_names
    for c in components:
        if c.name.lower() in GENERIC_WORDS:
            generic_risk.add(c.name)

    llm_specific = [c for c in llm_needed if c.component_name not in generic_risk]
    llm_generic = [c for c in llm_needed if c.component_name in generic_risk]
    print(f"    LLM split: {len(llm_specific)} specific, {len(llm_generic)} generic-risk")

    # Path A: 2-pass intersect for specific names
    specific_validated = []
    if llm_specific:
        ctx = []
        if linker.learned_patterns:
            if linker.learned_patterns.action_indicators:
                ctx.append(f"ACTION: {', '.join(linker.learned_patterns.action_indicators[:4])}")
            if linker.learned_patterns.effect_indicators:
                ctx.append(f"EFFECT (reject): {', '.join(linker.learned_patterns.effect_indicators[:3])}")
            if linker.learned_patterns.subprocess_terms:
                ctx.append(f"Subprocess (reject): {', '.join(list(linker.learned_patterns.subprocess_terms)[:5])}")

        for batch_start in range(0, len(llm_specific), 25):
            batch = llm_specific[batch_start:batch_start + 25]
            cases = []
            for i, c in enumerate(batch):
                prev = sent_map.get(c.sentence_number - 1)
                p = f"[prev: {prev.text[:35]}...] " if prev else ""
                cases.append(f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n  {p}"{c.sentence_text}"')

            r1 = _validation_pass(linker, comp_names, ctx, cases,
                "Focus on ACTOR role: is the component performing an action or being described?")
            r2 = _validation_pass(linker, comp_names, ctx, cases,
                "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component, not a generic concept?")

            for i, c in enumerate(batch):
                if r1.get(i, False) and r2.get(i, False):
                    c.confidence = 1.0
                    c.source = "validated"
                    specific_validated.append(c)

    # Path B: Strict evidence-citation for generic-risk names
    generic_validated = []
    if llm_generic:
        for batch_start in range(0, len(llm_generic), 25):
            batch = llm_generic[batch_start:batch_start + 25]
            cases = []
            for i, c in enumerate(batch):
                cases.append(f'Case {i+1}: S{c.sentence_number} "{c.sentence_text}"\n  Candidate: {c.component_name}')

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

            data = linker.llm.extract_json(linker.llm.query(prompt, timeout=120))
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
                if not sent:
                    continue
                if evidence.lower() not in sent.text.lower():
                    continue
                ev_lower = evidence.lower()
                aliases = alias_map.get(c.component_name, {c.component_name.lower()})
                if any(a.lower() in ev_lower for a in aliases if len(a) >= 2):
                    c.confidence = 1.0
                    c.source = "validated"
                    generic_validated.append(c)

    print(f"    Specific 2-pass: {len(specific_validated)}/{len(llm_specific)}, "
          f"Generic evidence: {len(generic_validated)}/{len(llm_generic)}")
    return direct + auto_approved + specific_validated + generic_validated


def phase6_var_m(linker, candidates, components, sent_map):
    """Phase 6 variant M: Word-boundary code-first + evidence split by generic risk.

    For LLM-needed candidates, all go to same evidence-citation prompt. But:
    - Non-generic components: accept if evidence exists in sentence (relaxed, high recall)
    - Generic-risk components: accept only if evidence contains name/alias (strict, low FP)
    """
    if not candidates:
        return []
    comp_names = linker._get_comp_names(components)
    direct, auto_approved, llm_needed, alias_map = _code_first_split_v2(linker, candidates, components, sent_map)
    if not llm_needed:
        return candidates if not auto_approved else direct + auto_approved

    # Classify generic-risk
    GENERIC_WORDS = {"logic", "storage", "common", "client", "model", "action", "data", "service", "server"}
    generic_risk = set()
    if linker.model_knowledge and linker.model_knowledge.ambiguous_names:
        generic_risk |= linker.model_knowledge.ambiguous_names
    for c in components:
        if c.name.lower() in GENERIC_WORDS:
            generic_risk.add(c.name)

    n_generic = sum(1 for c in llm_needed if c.component_name in generic_risk)
    n_specific = len(llm_needed) - n_generic
    print(f"    LLM needed: {n_specific} specific, {n_generic} generic-risk")

    evidence_validated = []
    for batch_start in range(0, len(llm_needed), 25):
        batch = llm_needed[batch_start:batch_start + 25]
        cases = []
        for i, c in enumerate(batch):
            cases.append(f'Case {i+1}: S{c.sentence_number} "{c.sentence_text}"\n  Candidate: {c.component_name}')

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

        data = linker.llm.extract_json(linker.llm.query(prompt, timeout=120))
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
            if not sent:
                continue
            # Anti-hallucination: evidence must exist in sentence
            if evidence.lower() not in sent.text.lower():
                continue

            if c.component_name in generic_risk:
                # Strict: name/alias must appear in evidence
                ev_lower = evidence.lower()
                aliases = alias_map.get(c.component_name, {c.component_name.lower()})
                if any(a.lower() in ev_lower for a in aliases if len(a) >= 2):
                    c.confidence = 1.0
                    c.source = "validated"
                    evidence_validated.append(c)
            else:
                # Relaxed: evidence exists in sentence (anti-hallucination passed)
                c.confidence = 0.9
                c.source = "validated"
                evidence_validated.append(c)

    print(f"    Evidence validated: {len(evidence_validated)} of {len(llm_needed)}")
    return direct + auto_approved + evidence_validated


def phase6_var_n(linker, candidates, components, sent_map):
    """Phase 6 variant N: code-first v2 + 2-pass intersect + evidence post-filter for generic names.

    Like variant I but adds evidence post-check for generic-risk candidates that 2-pass approved.
    TS: no generic-risk → identical to I (96-98%)
    TM: 2-pass approves Logic etc, then evidence filter catches FPs
    """
    if not candidates:
        return []
    comp_names = linker._get_comp_names(components)
    direct, auto_approved, llm_needed, alias_map = _code_first_split_v2(linker, candidates, components, sent_map)
    if not llm_needed:
        return candidates if not auto_approved else direct + auto_approved

    # Classify generic-risk
    GENERIC_WORDS = {"logic", "storage", "common", "client", "model", "action", "data", "service", "server"}
    generic_risk = set()
    if linker.model_knowledge and linker.model_knowledge.ambiguous_names:
        generic_risk |= linker.model_knowledge.ambiguous_names
    for c in components:
        if c.name.lower() in GENERIC_WORDS:
            generic_risk.add(c.name)

    # Step 1: 2-pass intersect for ALL LLM-needed (like variant I)
    ctx = []
    if linker.learned_patterns:
        if linker.learned_patterns.action_indicators:
            ctx.append(f"ACTION: {', '.join(linker.learned_patterns.action_indicators[:4])}")
        if linker.learned_patterns.effect_indicators:
            ctx.append(f"EFFECT (reject): {', '.join(linker.learned_patterns.effect_indicators[:3])}")
        if linker.learned_patterns.subprocess_terms:
            ctx.append(f"Subprocess (reject): {', '.join(list(linker.learned_patterns.subprocess_terms)[:5])}")

    twopass_approved = []
    generic_to_verify = []
    for batch_start in range(0, len(llm_needed), 25):
        batch = llm_needed[batch_start:batch_start + 25]
        cases = []
        for i, c in enumerate(batch):
            prev = sent_map.get(c.sentence_number - 1)
            p = f"[prev: {prev.text[:35]}...] " if prev else ""
            cases.append(f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n  {p}"{c.sentence_text}"')

        r1 = _validation_pass(linker, comp_names, ctx, cases,
            "Focus on ACTOR role: is the component performing an action or being described?")
        r2 = _validation_pass(linker, comp_names, ctx, cases,
            "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component, not a generic concept?")

        for i, c in enumerate(batch):
            if r1.get(i, False) and r2.get(i, False):
                if c.component_name in generic_risk:
                    generic_to_verify.append(c)
                else:
                    c.confidence = 1.0
                    c.source = "validated"
                    twopass_approved.append(c)

    print(f"    2-pass approved: {len(twopass_approved)} specific, {len(generic_to_verify)} generic need evidence")

    # Step 2: Evidence post-filter for generic-risk that passed 2-pass
    generic_validated = []
    if generic_to_verify:
        for batch_start in range(0, len(generic_to_verify), 25):
            batch = generic_to_verify[batch_start:batch_start + 25]
            cases = []
            for i, c in enumerate(batch):
                cases.append(f'Case {i+1}: S{c.sentence_number} "{c.sentence_text}"\n  Candidate: {c.component_name}')

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

            data = linker.llm.extract_json(linker.llm.query(prompt, timeout=120))
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
                if not sent:
                    continue
                if evidence.lower() not in sent.text.lower():
                    continue
                ev_lower = evidence.lower()
                aliases = alias_map.get(c.component_name, {c.component_name.lower()})
                if any(a.lower() in ev_lower for a in aliases if len(a) >= 2):
                    c.confidence = 1.0
                    c.source = "validated"
                    generic_validated.append(c)

        print(f"    Generic evidence: {len(generic_validated)}/{len(generic_to_verify)}")

    return direct + auto_approved + twopass_approved + generic_validated


def _code_first_split(linker, candidates, components, sent_map):
    """Shared code-first logic: auto-approve when name/alias in text, return (direct, auto, llm_needed)."""
    direct, needs = _phase6_pre_filter(linker, candidates, sent_map)
    if not needs:
        return direct, [], [], {}

    alias_map = {}
    for c in components:
        aliases = {c.name.lower()}
        if linker.doc_knowledge:
            for a, cn in linker.doc_knowledge.abbreviations.items():
                if cn == c.name and len(a) >= 3:
                    aliases.add(a.lower())
            for s, cn in linker.doc_knowledge.synonyms.items():
                if cn == c.name:
                    aliases.add(s.lower())
            for p, cn in linker.doc_knowledge.partial_references.items():
                if cn == c.name and len(p) >= 3:
                    aliases.add(p.lower())
        alias_map[c.name] = aliases

    auto_approved = []
    llm_needed = []
    for c in needs:
        sent = sent_map.get(c.sentence_number)
        if not sent:
            continue
        text_lower = sent.text.lower()
        aliases = alias_map.get(c.component_name, set())
        if any(a in text_lower for a in aliases):
            c.confidence = 1.0
            c.source = "validated"
            auto_approved.append(c)
        else:
            llm_needed.append(c)

    print(f"    Code-first auto-approved: {len(auto_approved)}, LLM needed: {len(llm_needed)}")
    return direct, auto_approved, llm_needed, alias_map


def phase6_var_e(linker, candidates, components, sent_map):
    """Phase 6 variant E: Code-first + relaxed evidence.

    Like D but relaxed: evidence just needs to exist in sentence (anti-hallucination).
    Does NOT require component name to appear in evidence text.
    """
    if not candidates:
        return []
    comp_names = linker._get_comp_names(components)
    direct, auto_approved, llm_needed, alias_map = _code_first_split(linker, candidates, components, sent_map)
    if not llm_needed:
        return candidates if not auto_approved else direct + auto_approved

    evidence_validated = []
    for batch_start in range(0, len(llm_needed), 25):
        batch = llm_needed[batch_start:batch_start + 25]
        cases = []
        for i, c in enumerate(batch):
            cases.append(f'Case {i+1}: S{c.sentence_number} "{c.sentence_text}"\n  Candidate: {c.component_name}')

        prompt = f"""For each case, find the EXACT text in the sentence that refers to the architecture component.

COMPONENTS: {', '.join(comp_names)}

CASES:
{chr(10).join(cases)}

For each case, provide:
- evidence_text: the EXACT substring from the sentence that references or describes the component
- If the sentence does NOT reference this component at all, set evidence_text to null

Return JSON:
{{"validations": [{{"case": 1, "evidence_text": "exact substring or null"}}]}}
JSON only:"""

        data = linker.llm.extract_json(linker.llm.query(prompt, timeout=120))
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
            if not sent:
                continue
            # Only anti-hallucination: evidence must exist in sentence
            if evidence.lower() not in sent.text.lower():
                continue
            c.confidence = 1.0
            c.source = "validated"
            evidence_validated.append(c)

    return direct + auto_approved + evidence_validated


def phase6_var_f(linker, candidates, components, sent_map):
    """Phase 6 variant F: Code-first + structured evidence with type.

    LLM classifies each reference as "direct" (name in text) or "functional"
    (describes what component does). For "direct", code verifies name in evidence.
    For "functional", requires LLM to explain what function links to the component.
    """
    if not candidates:
        return []
    comp_names = linker._get_comp_names(components)
    direct, auto_approved, llm_needed, alias_map = _code_first_split(linker, candidates, components, sent_map)
    if not llm_needed:
        return candidates if not auto_approved else direct + auto_approved

    evidence_validated = []
    for batch_start in range(0, len(llm_needed), 25):
        batch = llm_needed[batch_start:batch_start + 25]
        cases = []
        for i, c in enumerate(batch):
            cases.append(f'Case {i+1}: S{c.sentence_number} "{c.sentence_text}"\n  Candidate: {c.component_name}')

        prompt = f"""For each case, determine if the sentence references the architecture component.

COMPONENTS: {', '.join(comp_names)}

CASES:
{chr(10).join(cases)}

For each case provide:
- reference_type: "direct" if the component name/alias appears in text, "functional" if the sentence describes what the component does without naming it, or null if no reference
- evidence_text: the EXACT substring from the sentence (for direct: the name/alias, for functional: the description of the component's role)

IMPORTANT: "functional" means the sentence describes a SPECIFIC action that ONLY this component performs.
Generic descriptions like "handles data" or "processes requests" are NOT functional references.

Return JSON:
{{"validations": [{{"case": 1, "reference_type": "direct"|"functional"|null, "evidence_text": "exact substring or null"}}]}}
JSON only:"""

        data = linker.llm.extract_json(linker.llm.query(prompt, timeout=120))
        if not data:
            continue

        for v in data.get("validations", []):
            idx = v.get("case", 0) - 1
            if idx < 0 or idx >= len(batch):
                continue
            c = batch[idx]
            ref_type = v.get("reference_type")
            evidence = v.get("evidence_text")
            if not ref_type or not evidence:
                continue
            sent = sent_map.get(c.sentence_number)
            if not sent:
                continue
            # Anti-hallucination
            if evidence.lower() not in sent.text.lower():
                continue

            if ref_type == "direct":
                # Code verifies name/alias in evidence
                ev_lower = evidence.lower()
                aliases = alias_map.get(c.component_name, {c.component_name.lower()})
                if any(a in ev_lower for a in aliases if len(a) >= 2):
                    c.confidence = 1.0
                    c.source = "validated"
                    evidence_validated.append(c)
            elif ref_type == "functional":
                # Trust functional if evidence is substantial (>10 chars = not just a word)
                if len(evidence) >= 10:
                    c.confidence = 0.9
                    c.source = "validated"
                    evidence_validated.append(c)

    return direct + auto_approved + evidence_validated


def _validation_pass(linker, comp_names, ctx, cases, focus):
    prompt = f"""Validate component references in a software architecture document. {focus}

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

IMPORTANT DISTINCTIONS:
- "the routing logic" / "business logic" = generic English, NOT an architectural component → REJECT
- "client-side rendering" / "on the client" = generic usage, NOT a specific component → REJECT
- "data storage layer" / "in-memory cache" = generic concept, NOT a specific component → REJECT
- But "Router handles request processing" = the component IS the actor → APPROVE
- Section headings naming a component = introduces that component's section → APPROVE

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

    data = linker.llm.extract_json(linker.llm.query(prompt, timeout=120))
    results = {}
    if data:
        for v in data.get("validations", []):
            idx = v.get("case", 0) - 1
            if 0 <= idx < len(cases):
                results[idx] = v.get("approve", False)
    return results


# ═══════════════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════════════

def _cache_path(dataset_name):
    return Path(f"/tmp/phase_var_cache_{dataset_name}.pkl")


def test_phase(phase_num, dataset_name, n_runs, use_new, variant="", use_cache=False):
    import pickle

    paths = DATASETS[dataset_name]
    gold = load_gold_sam(str(paths["gold_sam"]))

    components = parse_pcm_repository(str(paths["model"]))
    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    name_to_id = {c.name: c.id for c in components}
    id_to_name = {c.id: c.name for c in components}
    sent_map = DocumentLoader.build_sent_map(sentences)

    # Create V18 linker
    from llm_sad_sam.linkers.experimental.agent_linker_v18 import AgentLinkerV18
    linker = AgentLinkerV18(backend=LLMBackend.CLAUDE)

    print(f"\n{'='*80}")
    label = variant.upper() if variant else ("NEW" if use_new else "OLD")
    print(f"PHASE {phase_num} VARIANCE TEST ({label}): {dataset_name}, {n_runs} runs")
    print(f"{'='*80}")

    cache_file = _cache_path(dataset_name)
    if use_cache and cache_file.exists():
        print("\n[Setup] Loading cached Phase 0-4 state...")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        linker.doc_profile = cached["doc_profile"]
        linker._is_complex = cached["is_complex"]
        linker.thresholds = cached["thresholds"]
        linker._cached_sent_map = sent_map
        linker.model_knowledge = cached["model_knowledge"]
        linker.learned_patterns = cached["learned_patterns"]
        linker.doc_knowledge = cached["doc_knowledge"]
        transarc_links = cached["transarc_links"]
        transarc_set = cached["transarc_set"]
    else:
        # Run phases 0-3b + 4 (deterministic-ish setup)
        print("\n[Setup] Running Phases 0-4...")
        transarc_links, transarc_set = inject_v18_state(
            linker, dataset_name, paths, sentences, components, sent_map, name_to_id, id_to_name
        )
        # Save cache
        with open(cache_file, "wb") as f:
            pickle.dump({
                "doc_profile": linker.doc_profile,
                "is_complex": linker._is_complex,
                "thresholds": linker.thresholds,
                "model_knowledge": linker.model_knowledge,
                "learned_patterns": linker.learned_patterns,
                "doc_knowledge": linker.doc_knowledge,
                "transarc_links": transarc_links,
                "transarc_set": transarc_set,
            }, f)
        print(f"  [Cached to {cache_file}]")

    print(f"  TransArc: {len(transarc_links)} links")
    print(f"  Doc knowledge: abbrev={len(linker.doc_knowledge.abbreviations)}, "
          f"syn={len(linker.doc_knowledge.synonyms)}, "
          f"partial={len(linker.doc_knowledge.partial_references)}")

    all_run_links = []
    all_run_counts = []

    for run_idx in range(n_runs):
        print(f"\n{'─'*60}")
        print(f"  RUN {run_idx + 1}/{n_runs}")
        print(f"{'─'*60}")

        if phase_num == 5:
            if use_new:
                candidates = phase5_new(linker, sentences, components, name_to_id, sent_map)
            else:
                candidates = linker._extract_entities_enriched(sentences, components, name_to_id, sent_map)

            # Apply abbrev guard
            candidates = linker._apply_abbreviation_guard_to_candidates(candidates, sent_map)
            link_set = {(c.sentence_number, c.component_id) for c in candidates}
            print(f"  Candidates: {len(candidates)}")
            all_run_counts.append(len(candidates))

            # Eval as transarc + raw entity
            entity_links = [SadSamLink(c.sentence_number, c.component_id, c.component_name, 0.85, "entity")
                           for c in candidates]
            combined = {(l.sentence_number, l.component_id) for l in transarc_links + entity_links}
            tp = len(combined & gold)
            fp = len(combined - gold)
            fn = len(gold - combined)
            p = tp / (tp + fp) if (tp + fp) else 0
            r = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * p * r / (p + r) if (p + r) else 0
            print(f"  TransArc+Entity: P={p:.1%} R={r:.1%} F1={f1:.1%} (TP={tp} FP={fp} FN={fn})")
            all_run_links.append(link_set)

        elif phase_num == 7:
            # Need Phase 5+6 first — run V18's Phase 5+6 once (shared across coref runs)
            if run_idx == 0:
                print("  [Pre-req] Running Phase 5+6 once...")
                shared_candidates = linker._extract_entities_enriched(sentences, components, name_to_id, sent_map)
                shared_candidates = linker._apply_abbreviation_guard_to_candidates(shared_candidates, sent_map)
                shared_validated = linker._validate_intersect(shared_candidates, components, sent_map)
                print(f"  Shared: {len(shared_candidates)} candidates -> {len(shared_validated)} validated")

            if use_new:
                if linker._is_complex:
                    coref_links = phase7_new_debate(linker, sentences, components, name_to_id, sent_map)
                else:
                    discourse_model = linker._build_discourse_model(sentences, components, name_to_id)
                    coref_links = phase7_new_discourse(linker, sentences, components, name_to_id, sent_map, discourse_model)
            else:
                if linker._is_complex:
                    coref_links = linker._coref_debate(sentences, components, name_to_id, sent_map)
                else:
                    discourse_model = linker._build_discourse_model(sentences, components, name_to_id)
                    coref_links = linker._coref_discourse(sentences, components, name_to_id, sent_map, discourse_model)

            # Apply antecedent filter (V18 original)
            before = len(coref_links)
            coref_links = linker._filter_generic_coref(coref_links, sent_map)
            if len(coref_links) < before:
                print(f"  After antecedent filter: {len(coref_links)} (-{before - len(coref_links)})")

            link_set = {(l.sentence_number, l.component_id) for l in coref_links}
            print(f"  Coref links: {len(coref_links)}")
            all_run_counts.append(len(coref_links))

            # Eval
            entity_links = [SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
                           for c in shared_validated]
            combined = {(l.sentence_number, l.component_id)
                       for l in transarc_links + entity_links + coref_links}
            tp = len(combined & gold)
            fp = len(combined - gold)
            fn = len(gold - combined)
            p = tp / (tp + fp) if (tp + fp) else 0
            r = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * p * r / (p + r) if (p + r) else 0
            print(f"  TransArc+Valid+Coref: P={p:.1%} R={r:.1%} F1={f1:.1%} (TP={tp} FP={fp} FN={fn})")
            all_run_links.append(link_set)

        elif phase_num == 6:
            # Need Phase 5 first — run once
            if run_idx == 0:
                print("  [Pre-req] Running Phase 5 once...")
                shared_candidates = linker._extract_entities_enriched(sentences, components, name_to_id, sent_map)
                shared_candidates = linker._apply_abbreviation_guard_to_candidates(shared_candidates, sent_map)
                print(f"  Shared: {len(shared_candidates)} candidates")

            if variant == "a":
                validated = phase6_new(linker, copy.deepcopy(shared_candidates), components, sent_map)
            elif variant == "b":
                validated = phase6_var_b(linker, copy.deepcopy(shared_candidates), components, sent_map)
            elif variant == "c":
                validated = phase6_var_c(linker, copy.deepcopy(shared_candidates), components, sent_map)
            elif variant == "d":
                validated = phase6_var_d(linker, copy.deepcopy(shared_candidates), components, sent_map)
            elif variant == "e":
                validated = phase6_var_e(linker, copy.deepcopy(shared_candidates), components, sent_map)
            elif variant == "f":
                validated = phase6_var_f(linker, copy.deepcopy(shared_candidates), components, sent_map)
            elif variant == "g":
                validated = phase6_var_g(linker, copy.deepcopy(shared_candidates), components, sent_map)
            elif variant == "h":
                validated = phase6_var_h(linker, copy.deepcopy(shared_candidates), components, sent_map)
            elif variant == "i":
                validated = phase6_var_i(linker, copy.deepcopy(shared_candidates), components, sent_map)
            elif variant == "j":
                validated = phase6_var_j(linker, copy.deepcopy(shared_candidates), components, sent_map)
            elif variant == "k":
                validated = phase6_var_k(linker, copy.deepcopy(shared_candidates), components, sent_map)
            elif variant == "l":
                validated = phase6_var_l(linker, copy.deepcopy(shared_candidates), components, sent_map)
            elif variant == "m":
                validated = phase6_var_m(linker, copy.deepcopy(shared_candidates), components, sent_map)
            elif variant == "n":
                validated = phase6_var_n(linker, copy.deepcopy(shared_candidates), components, sent_map)
            elif use_new:
                validated = phase6_new(linker, copy.deepcopy(shared_candidates), components, sent_map)
            else:
                validated = linker._validate_intersect(copy.deepcopy(shared_candidates), components, sent_map)

            link_set = {(c.sentence_number, c.component_id) for c in validated}
            print(f"  Validated: {len(validated)} of {len(shared_candidates)}")
            all_run_counts.append(len(validated))

            entity_links = [SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
                           for c in validated]
            combined = {(l.sentence_number, l.component_id) for l in transarc_links + entity_links}
            tp = len(combined & gold)
            fp = len(combined - gold)
            fn = len(gold - combined)
            p = tp / (tp + fp) if (tp + fp) else 0
            r = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * p * r / (p + r) if (p + r) else 0
            print(f"  TransArc+Valid: P={p:.1%} R={r:.1%} F1={f1:.1%} (TP={tp} FP={fp} FN={fn})")
            all_run_links.append(link_set)

    # ── Variance analysis ──
    print(f"\n{'='*80}")
    print(f"VARIANCE SUMMARY: Phase {phase_num} ({label}), {dataset_name}, {n_runs} runs")
    print(f"{'='*80}")

    print(f"\nCounts: {all_run_counts}")
    if all_run_counts:
        print(f"  Range: {min(all_run_counts)}-{max(all_run_counts)} (spread: {max(all_run_counts)-min(all_run_counts)})")

    # Link stability
    if all_run_links:
        union = set()
        intersection = all_run_links[0].copy()
        for s in all_run_links:
            union |= s
            intersection &= s
        print(f"\nLink stability:")
        print(f"  Union: {len(union)}")
        print(f"  Intersection: {len(intersection)}")
        print(f"  Stability: {len(intersection)}/{len(union)} = {len(intersection)/max(1,len(union)):.0%}")

        # Per-link frequency
        freq = defaultdict(int)
        for s in all_run_links:
            for key in s:
                freq[key] += 1
        varying = {k: v for k, v in freq.items() if v < n_runs}
        if varying:
            print(f"\n  Varying links ({len(varying)}):")
            for (snum, cid), count in sorted(varying.items(), key=lambda x: -x[1]):
                cname = id_to_name.get(cid, cid[:15])
                is_gold = (snum, cid) in gold
                status = "GOLD" if is_gold else "non-gold"
                print(f"    S{snum:3d} -> {cname:18s} {count}/{n_runs} [{status}]")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase prompt variance tester")
    parser.add_argument("--phase", type=int, required=True, choices=[5, 6, 7])
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--new", action="store_true", help="Use new prompt (default: old V18 prompt)")
    parser.add_argument("--variant", default="", help="Phase 6 variant: a-n")
    parser.add_argument("--cache", action="store_true", help="Use cached Phase 0-4 state (skip LLM setup)")
    args = parser.parse_args()

    test_phase(args.phase, args.dataset, args.runs, args.new, args.variant, args.cache)


if __name__ == "__main__":
    main()
