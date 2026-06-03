#!/usr/bin/env python3
"""Test coref debate variants that aim to work for BOTH simple and complex docs.

Loads V39 checkpoints (phase2 for learned_patterns, phase3 for doc_knowledge,
phase1 for model_knowledge), runs each coref variant with real LLM calls,
and compares against gold standard.

Usage:
    python test_coref_debate_variants.py                    # all datasets, all variants
    python test_coref_debate_variants.py --datasets mediastore teastore
    python test_coref_debate_variants.py --variants A B
    python test_coref_debate_variants.py --datasets mediastore --variants A B C D E
"""

import csv
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.core.data_types import SadSamLink, DocumentKnowledge, ModelKnowledge, LearnedPatterns
from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend

os.environ.setdefault("CLAUDE_MODEL", "sonnet")

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
DATASETS = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
        "gold": BENCHMARK_BASE / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
        "model": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
        "gold": BENCHMARK_BASE / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
        "gold": BENCHMARK_BASE / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
        "gold": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
        "gold": BENCHMARK_BASE / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
}
CACHE_DIR = Path("./results/phase_cache/v39")

PRONOUN_PATTERN = re.compile(
    r'\b(it|they|this|these|that|those|its|their|the component|the service)\b',
    re.IGNORECASE
)


# ── Helpers ─────────────────────────────────────────────────────────────

def load_checkpoint(dataset, phase_name):
    path = CACHE_DIR / dataset / f"{phase_name}.pkl"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_gold(gold_path):
    links = set()
    with open(gold_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def load_dataset(name):
    paths = DATASETS[name]
    components = parse_pcm_repository(str(paths["model"]))
    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    sent_map = DocumentLoader.build_sent_map(sentences)
    name_to_id = {c.name: c.id for c in components}
    gold = load_gold(paths["gold"])
    return components, sentences, sent_map, name_to_id, gold


def has_standalone_mention(comp_name, text, doc_knowledge=None):
    """Check if component name appears standalone (not in dotted path)."""
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


def has_alias_mention(comp_name, sentence_text, doc_knowledge):
    """Check if any known synonym or partial reference appears."""
    if not doc_knowledge:
        return False
    text_lower = sentence_text.lower()
    for syn, target in doc_knowledge.synonyms.items():
        if target == comp_name:
            if re.search(rf'\b{re.escape(syn.lower())}\b', text_lower):
                return True
    for partial, target in doc_knowledge.partial_references.items():
        if target == comp_name:
            if re.search(rf'\b{re.escape(partial.lower())}\b', text_lower):
                return True
    return False


def verify_antecedent(comp, snum, ant_snum, sent_map, doc_knowledge):
    """Verify antecedent citation: component must appear in antecedent sentence, within 3 sents."""
    if ant_snum is None:
        return True  # No antecedent provided, pass through (code verifies elsewhere)
    if isinstance(ant_snum, str):
        ant_snum = ant_snum.lstrip("S")
    try:
        ant_snum = int(ant_snum)
    except (ValueError, TypeError):
        return False
    ant_sent = sent_map.get(ant_snum)
    if not ant_sent:
        return False
    if not (has_standalone_mention(comp, ant_sent.text) or
            has_alias_mention(comp, ant_sent.text, doc_knowledge)):
        return False
    if abs(snum - ant_snum) > 3:
        return False
    return True


def parse_resolutions(data, name_to_id, sent_map, doc_knowledge, learned_patterns):
    """Parse LLM resolution JSON into SadSamLink list with antecedent verification."""
    links = []
    if not data:
        return links
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
            if not verify_antecedent(comp, snum, ant_snum, sent_map, doc_knowledge):
                continue
        # Skip subprocess sentences
        sent = sent_map.get(snum)
        if sent and learned_patterns and learned_patterns.is_subprocess(sent.text):
            continue
        links.append(SadSamLink(snum, name_to_id[comp], comp, 1.0, "coreference"))
    return links


def get_comp_names(components, model_knowledge):
    """Get non-implementation component names."""
    return [c.name for c in components
            if not (model_knowledge and model_knowledge.is_implementation(c.name))]


# ══════════════════════════════════════════════════════════════════════════
# VARIANT A: Filtered Debate
# Debate window structure, but only *-mark sentences with pronouns.
# Hypothesis: LLM knows exactly which sentences to analyze, gets full
# paragraph context, but doesn't hallucinate refs in pronoun-free sentences.
# ══════════════════════════════════════════════════════════════════════════

def coref_variant_A(llm, sentences, components, name_to_id, sent_map,
                    model_knowledge, doc_knowledge, learned_patterns):
    """Filtered debate: full document window, only pronoun sentences starred."""
    comp_names = get_comp_names(components, model_knowledge)
    all_coref = []

    ctx = []
    if learned_patterns and learned_patterns.subprocess_terms:
        ctx.append(f"Subprocesses (don't link): {', '.join(list(learned_patterns.subprocess_terms)[:5])}")

    # Find which sentences have pronouns
    pronoun_snums = {s.number for s in sentences if PRONOUN_PATTERN.search(s.text)}

    for batch_start in range(0, len(sentences), 20):
        batch = sentences[batch_start:min(batch_start + 20, len(sentences))]
        ctx_start = max(0, batch_start - 3)
        ctx_sents = sentences[ctx_start:batch_start + 20]

        # Only star sentences that are in the batch AND contain pronouns
        starred_in_batch = [s for s in batch if s.number in pronoun_snums]
        if not starred_in_batch:
            continue  # Skip batches with no pronoun sentences

        doc_lines = []
        for s in ctx_sents:
            in_batch = s.number >= batch[0].number
            has_pronoun = s.number in pronoun_snums
            marker = '*' if (in_batch and has_pronoun) else ' '
            doc_lines.append(f"{marker}S{s.number}: {s.text}")

        prompt = f"""Resolve pronoun references to architecture components.

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

DOCUMENT (* = sentences to analyze for pronoun references):
{chr(10).join(doc_lines)}

Find pronouns (it, they, this, these) in starred (*) sentences that refer to a component.
Non-starred sentences are context only — do NOT analyze them.

RULES (all must hold):
1. You MUST cite the antecedent_sentence where the component was EXPLICITLY NAMED
2. The component name (or known alias) MUST appear verbatim in the antecedent sentence
3. The antecedent MUST be within the previous 3 sentences
4. Do NOT resolve pronouns in sentences about subprocesses or implementation details
5. If the pronoun could refer to multiple components, do NOT resolve it

Return JSON:
{{"resolutions": [{{"sentence": N_INTEGER, "pronoun": "it", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

Only include resolutions you are CERTAIN about. JSON only:"""

        data = llm.extract_json(llm.query(prompt, timeout=120))
        links = parse_resolutions(data, name_to_id, sent_map, doc_knowledge, learned_patterns)
        all_coref.extend(links)

    return all_coref


# ══════════════════════════════════════════════════════════════════════════
# VARIANT B: Intersect Debate
# Run debate twice independently, keep only intersection.
# Hypothesis: hallucinated resolutions are random and don't survive voting.
# ══════════════════════════════════════════════════════════════════════════

def coref_variant_B(llm, sentences, components, name_to_id, sent_map,
                    model_knowledge, doc_knowledge, learned_patterns):
    """Intersect debate: two independent passes, keep intersection."""
    comp_names = get_comp_names(components, model_knowledge)

    ctx = []
    if learned_patterns and learned_patterns.subprocess_terms:
        ctx.append(f"Subprocesses (don't link): {', '.join(list(learned_patterns.subprocess_terms)[:5])}")

    def run_pass(pass_label):
        all_links = []
        for batch_start in range(0, len(sentences), 20):
            batch = sentences[batch_start:min(batch_start + 20, len(sentences))]
            ctx_start = max(0, batch_start - 3)
            ctx_sents = sentences[ctx_start:batch_start + 20]
            doc_lines = [
                f"{'*' if s.number >= batch[0].number else ' '}S{s.number}: {s.text}"
                for s in ctx_sents
            ]

            prompt = f"""[Pass {pass_label}] Resolve pronoun references to architecture components.

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

DOCUMENT (* = analyze these sentences):
{chr(10).join(doc_lines)}

Find pronouns (it, they, this, these) in starred sentences that refer to a component.

RULES (all must hold):
1. You MUST cite the antecedent_sentence where the component was EXPLICITLY NAMED
2. The component name (or known alias) MUST appear verbatim in the antecedent sentence
3. The antecedent MUST be within the previous 3 sentences
4. Do NOT resolve pronouns in sentences about subprocesses or implementation details
5. If the pronoun could refer to multiple components, do NOT resolve it

Return JSON:
{{"resolutions": [{{"sentence": N_INTEGER, "pronoun": "it", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

Only include resolutions you are CERTAIN about. JSON only:"""

            data = llm.extract_json(llm.query(prompt, timeout=120))
            links = parse_resolutions(data, name_to_id, sent_map, doc_knowledge, learned_patterns)
            all_links.extend(links)
        return {(l.sentence_number, l.component_id) for l in all_links}, all_links

    set_a, links_a = run_pass("A")
    set_b, links_b = run_pass("B")

    # Keep intersection
    keep = set_a & set_b
    # Use links from pass A for the actual objects
    result = [l for l in links_a if (l.sentence_number, l.component_id) in keep]
    print(f"    Intersect: pass_A={len(set_a)}, pass_B={len(set_b)}, intersection={len(keep)}")
    return result


# ══════════════════════════════════════════════════════════════════════════
# VARIANT C: Small Batch Debate
# Batch size 10 instead of 20 — less context, less hallucination.
# Hypothesis: 10-sentence windows are enough for pronoun resolution
# (antecedent must be within 3 sentences anyway).
# ══════════════════════════════════════════════════════════════════════════

def coref_variant_C(llm, sentences, components, name_to_id, sent_map,
                    model_knowledge, doc_knowledge, learned_patterns):
    """Small batch debate: batch size 10 instead of 20."""
    comp_names = get_comp_names(components, model_knowledge)
    all_coref = []

    ctx = []
    if learned_patterns and learned_patterns.subprocess_terms:
        ctx.append(f"Subprocesses (don't link): {', '.join(list(learned_patterns.subprocess_terms)[:5])}")

    BATCH_SIZE = 10
    for batch_start in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[batch_start:min(batch_start + BATCH_SIZE, len(sentences))]
        ctx_start = max(0, batch_start - 3)
        ctx_sents = sentences[ctx_start:batch_start + BATCH_SIZE]
        doc_lines = [
            f"{'*' if s.number >= batch[0].number else ' '}S{s.number}: {s.text}"
            for s in ctx_sents
        ]

        prompt = f"""Resolve pronoun references to architecture components.

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

DOCUMENT (* = analyze these sentences):
{chr(10).join(doc_lines)}

Find pronouns (it, they, this, these) in starred sentences that refer to a component.

RULES (all must hold):
1. You MUST cite the antecedent_sentence where the component was EXPLICITLY NAMED
2. The component name (or known alias) MUST appear verbatim in the antecedent sentence
3. The antecedent MUST be within the previous 3 sentences
4. Do NOT resolve pronouns in sentences about subprocesses or implementation details
5. If the pronoun could refer to multiple components, do NOT resolve it

Return JSON:
{{"resolutions": [{{"sentence": N_INTEGER, "pronoun": "it", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

Only include resolutions you are CERTAIN about. JSON only:"""

        data = llm.extract_json(llm.query(prompt, timeout=100))
        links = parse_resolutions(data, name_to_id, sent_map, doc_knowledge, learned_patterns)
        all_coref.extend(links)

    return all_coref


# ══════════════════════════════════════════════════════════════════════════
# VARIANT D: Strict Prompt Debate
# Same debate structure, but prompt explicitly forbids implicit references.
# Hypothesis: debate's FPs come from the LLM inferring implicit component
# mentions beyond pronoun resolution. Telling it "ONLY pronouns" fixes this.
# ══════════════════════════════════════════════════════════════════════════

def coref_variant_D(llm, sentences, components, name_to_id, sent_map,
                    model_knowledge, doc_knowledge, learned_patterns):
    """Strict prompt debate: explicit pronoun-only instructions."""
    comp_names = get_comp_names(components, model_knowledge)
    all_coref = []

    ctx = []
    if learned_patterns and learned_patterns.subprocess_terms:
        ctx.append(f"Subprocesses (don't link): {', '.join(list(learned_patterns.subprocess_terms)[:5])}")

    for batch_start in range(0, len(sentences), 20):
        batch = sentences[batch_start:min(batch_start + 20, len(sentences))]
        ctx_start = max(0, batch_start - 3)
        ctx_sents = sentences[ctx_start:batch_start + 20]
        doc_lines = [
            f"{'*' if s.number >= batch[0].number else ' '}S{s.number}: {s.text}"
            for s in ctx_sents
        ]

        prompt = f"""TASK: Resolve PRONOUN references ONLY. Do NOT find implicit or inferred component mentions.

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

DOCUMENT (* = analyze these sentences):
{chr(10).join(doc_lines)}

Your ONLY job: find pronouns (it, they, this, these, those, its, their) in starred sentences
where the pronoun grammatically refers back to a NAMED component in a recent sentence.

STRICT RULES:
1. ONLY resolve literal pronouns — never resolve definite descriptions ("the system"),
   synonyms, or implied references
2. You MUST cite the antecedent_sentence where the component was EXPLICITLY NAMED
3. The component name MUST appear verbatim in the antecedent sentence
4. The antecedent MUST be within the previous 3 sentences
5. The pronoun MUST be the grammatical subject OR object that refers back to that component
6. Do NOT resolve if: the pronoun refers to a concept/process (not a component),
   the pronoun could refer to multiple things, or the sentence is about implementation details

IMPORTANT: If a sentence does NOT contain a pronoun, produce NO resolution for it.
When in doubt, do NOT resolve. False negatives are acceptable; false positives are not.

Return JSON:
{{"resolutions": [{{"sentence": N_INTEGER, "pronoun": "it", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

JSON only:"""

        data = llm.extract_json(llm.query(prompt, timeout=120))
        links = parse_resolutions(data, name_to_id, sent_map, doc_knowledge, learned_patterns)
        all_coref.extend(links)

    return all_coref


# ══════════════════════════════════════════════════════════════════════════
# VARIANT E: Cases-in-Context
# Discourse-style per-case presentation, but with full paragraph context.
# Each pronoun sentence gets its own "case" block with a wider context window
# (±5 sentences instead of discourse's 3-before-only).
# Hypothesis: per-case reasoning is more precise, and wider bidirectional
# context helps the LLM understand discourse structure.
# ══════════════════════════════════════════════════════════════════════════

def coref_variant_E(llm, sentences, components, name_to_id, sent_map,
                    model_knowledge, doc_knowledge, learned_patterns):
    """Cases-in-context: per-case presentation with ±5 sentence window."""
    comp_names = get_comp_names(components, model_knowledge)
    all_coref = []
    pronoun_sents = [s for s in sentences if PRONOUN_PATTERN.search(s.text)]

    ctx_info = []
    if learned_patterns and learned_patterns.subprocess_terms:
        ctx_info.append(f"Subprocesses (don't link): {', '.join(list(learned_patterns.subprocess_terms)[:5])}")

    for batch_start in range(0, len(pronoun_sents), 10):
        batch = pronoun_sents[batch_start:batch_start + 10]
        cases = []
        for sent in batch:
            # ±5 sentence context window (bidirectional)
            context = []
            for i in range(max(1, sent.number - 5), sent.number + 6):
                s = sent_map.get(i)
                if s:
                    marker = ">>>" if s.number == sent.number else "   "
                    context.append(f"{marker} S{s.number}: {s.text}")
            cases.append({"sent": sent, "context": context})

        prompt = f"""Resolve pronoun references to architecture components.

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx_info)}

"""
        for i, case in enumerate(cases):
            prompt += f"--- Case {i+1}: S{case['sent'].number} ---\n"
            prompt += "CONTEXT:\n" + "\n".join(case["context"]) + "\n"
            prompt += f"TARGET: S{case['sent'].number} (marked with >>>)\n\n"

        prompt += """For each case, determine if any pronoun in the TARGET sentence refers to a component.

RULES (all must hold):
1. The component name (or known alias) MUST appear verbatim in the antecedent sentence
2. The antecedent MUST be within the previous 3 sentences
3. The pronoun MUST grammatically refer back to that component as its subject
4. If the pronoun could refer to multiple things, DO NOT resolve it
5. Do NOT resolve pronouns about subprocesses or implementation details

Like in technical writing: "The Scheduler assigns tasks to threads. It uses a priority queue internally."
— "It" clearly refers to "the Scheduler" because it was the subject of the previous sentence.

Return JSON:
{"resolutions": [{"case": 1, "sentence": N_INTEGER, "pronoun": "it", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}]}

Only include resolutions you are CERTAIN about. JSON only:"""

        data = llm.extract_json(llm.query(prompt, timeout=150))
        links = parse_resolutions(data, name_to_id, sent_map, doc_knowledge, learned_patterns)
        all_coref.extend(links)

    return all_coref


# ══════════════════════════════════════════════════════════════════════════
# Baseline: V39's original discourse and debate (for comparison)
# ══════════════════════════════════════════════════════════════════════════

def coref_baseline_discourse(llm, sentences, components, name_to_id, sent_map,
                             model_knowledge, doc_knowledge, learned_patterns):
    """V39's original discourse mode (for simple docs)."""
    comp_names = get_comp_names(components, model_knowledge)
    all_coref = []
    pronoun_sents = [s for s in sentences if PRONOUN_PATTERN.search(s.text)]

    for batch_start in range(0, len(pronoun_sents), 12):
        batch = pronoun_sents[batch_start:batch_start + 12]
        cases = []
        for sent in batch:
            prev = []
            for i in range(1, 4):
                p = sent_map.get(sent.number - i)
                if p:
                    prev.append(f"S{p.number}: {p.text}")
            cases.append({"sent": sent, "prev": prev})

        prompt = f"""Resolve pronoun references to architecture components.

COMPONENTS: {', '.join(comp_names)}

"""
        for i, case in enumerate(cases):
            prompt += f"--- Case {i+1}: S{case['sent'].number} ---\n"
            if case["prev"]:
                prompt += "PREVIOUS:\n  " + "\n  ".join(reversed(case["prev"])) + "\n"
            prompt += f">>> {case['sent'].text}\n\n"

        prompt += """For each pronoun that refers to a component, provide:
- antecedent_sentence: the sentence number where the component was EXPLICITLY NAMED
- antecedent_text: the EXACT quote from that sentence containing the component name

RULES (all must hold):
1. The component name (or known alias) MUST appear verbatim in the antecedent sentence
2. The antecedent MUST be within the previous 3 sentences
3. The pronoun MUST grammatically refer back to that component as its subject
4. If the pronoun could refer to multiple things, DO NOT resolve it

Like in technical writing: "The Scheduler assigns tasks to threads. It uses a priority queue internally."
— "It" clearly refers to "the Scheduler" because it was the subject of the previous sentence.

Return JSON:
{"resolutions": [{"case": 1, "sentence": N_INTEGER, "pronoun": "it", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact text with component name"}]}

Only include resolutions you are CERTAIN about. JSON only:"""

        data = llm.extract_json(llm.query(prompt, timeout=150))
        links = parse_resolutions(data, name_to_id, sent_map, doc_knowledge, learned_patterns)
        all_coref.extend(links)

    return all_coref


def coref_baseline_debate(llm, sentences, components, name_to_id, sent_map,
                          model_knowledge, doc_knowledge, learned_patterns):
    """V39's original debate mode (for complex docs)."""
    comp_names = get_comp_names(components, model_knowledge)
    all_coref = []

    ctx = []
    if learned_patterns and learned_patterns.subprocess_terms:
        ctx.append(f"Subprocesses (don't link): {', '.join(list(learned_patterns.subprocess_terms)[:5])}")

    for batch_start in range(0, len(sentences), 20):
        batch = sentences[batch_start:min(batch_start + 20, len(sentences))]
        ctx_start = max(0, batch_start - 3)
        ctx_sents = sentences[ctx_start:batch_start + 20]
        doc_lines = [
            f"{'*' if s.number >= batch[0].number else ' '}S{s.number}: {s.text}"
            for s in ctx_sents
        ]

        prompt = f"""Resolve pronoun references to architecture components.

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

DOCUMENT (* = analyze these sentences):
{chr(10).join(doc_lines)}

Find pronouns (it, they, this, these) in starred sentences that refer to a component.

RULES (all must hold):
1. You MUST cite the antecedent_sentence where the component was EXPLICITLY NAMED
2. The component name (or known alias) MUST appear verbatim in the antecedent sentence
3. The antecedent MUST be within the previous 3 sentences
4. Do NOT resolve pronouns in sentences about subprocesses or implementation details
5. If the pronoun could refer to multiple components, do NOT resolve it

Return JSON:
{{"resolutions": [{{"sentence": N_INTEGER, "pronoun": "it", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

Only include resolutions you are CERTAIN about. JSON only:"""

        data = llm.extract_json(llm.query(prompt, timeout=100))
        links = parse_resolutions(data, name_to_id, sent_map, doc_knowledge, learned_patterns)
        all_coref.extend(links)

    return all_coref


# ══════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════

ALL_VARIANTS = {
    "discourse": ("Baseline: discourse (V39 simple)", coref_baseline_discourse),
    "debate":    ("Baseline: debate (V39 complex)", coref_baseline_debate),
    "A":         ("Filtered debate (only * pronoun sents)", coref_variant_A),
    "B":         ("Intersect debate (2-pass voting)", coref_variant_B),
    "C":         ("Small batch debate (batch=10)", coref_variant_C),
    "D":         ("Strict prompt debate (pronoun-only)", coref_variant_D),
    "E":         ("Cases-in-context (per-case ±5 window)", coref_variant_E),
}


def run_test(dataset_names, variant_names, backend=LLMBackend.CLAUDE):
    llm = LLMClient(backend=backend)
    print(f"Backend: {backend.value}")
    results = {}  # (dataset, variant) -> {tp, fp, fn, links}

    for ds_name in dataset_names:
        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*70}")

        components, sentences, sent_map, name_to_id, gold = load_dataset(ds_name)

        # Load V39 checkpoints for learned context
        p0 = load_checkpoint(ds_name, "phase0")
        p1 = load_checkpoint(ds_name, "phase1")
        p2 = load_checkpoint(ds_name, "phase2")
        p3 = load_checkpoint(ds_name, "phase3")

        model_knowledge = p1["model_knowledge"] if p1 else None
        learned_patterns = p2["learned_patterns"] if p2 else None
        doc_knowledge = p3["doc_knowledge"] if p3 else None
        is_complex = p0["is_complex"] if p0 else False

        print(f"  Sentences: {len(sentences)}, Components: {len(components)}, "
              f"is_complex: {is_complex}")
        print(f"  V39 used: {'debate' if is_complex else 'discourse'}")

        pronoun_count = sum(1 for s in sentences if PRONOUN_PATTERN.search(s.text))
        print(f"  Pronoun sentences: {pronoun_count}/{len(sentences)}")

        for var_name in variant_names:
            if var_name not in ALL_VARIANTS:
                print(f"  Unknown variant: {var_name}")
                continue

            desc, func = ALL_VARIANTS[var_name]
            print(f"\n  --- Variant {var_name}: {desc} ---")

            t0 = time.time()
            coref_links = func(llm, sentences, components, name_to_id, sent_map,
                               model_knowledge, doc_knowledge, learned_patterns)
            elapsed = time.time() - t0

            # Deduplicate
            seen = set()
            deduped = []
            for l in coref_links:
                key = (l.sentence_number, l.component_id)
                if key not in seen:
                    seen.add(key)
                    deduped.append(l)
            coref_links = deduped

            # Evaluate
            predicted = {(l.sentence_number, l.component_id) for l in coref_links}
            tp = len(predicted & gold)
            fp = len(predicted - gold)
            # FN is relative to what coref COULD recover — compare to gold links
            # in pronoun-bearing sentences only
            pronoun_gold = {(sn, cid) for (sn, cid) in gold
                           if sent_map.get(sn) and PRONOUN_PATTERN.search(sent_map[sn].text)}
            fn_coref = len(pronoun_gold - predicted)

            p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            r = tp / (tp + fn_coref) if (tp + fn_coref) > 0 else 1.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

            print(f"  Links: {len(coref_links)} | TP={tp} FP={fp} FN(coref)={fn_coref} | "
                  f"P={p:.1%} R={r:.1%} F1={f1:.1%} | {elapsed:.1f}s")

            # Show individual links
            for l in sorted(coref_links, key=lambda x: x.sentence_number):
                is_tp = (l.sentence_number, l.component_id) in gold
                sent = sent_map.get(l.sentence_number)
                text = sent.text[:80] if sent else "?"
                label = "TP" if is_tp else "FP"
                print(f"    [{label}] S{l.sentence_number} -> {l.component_name}: {text}")

            # Show missed gold links in pronoun sentences
            missed = pronoun_gold - predicted
            if missed:
                print(f"  Missed ({len(missed)}):")
                for sn, cid in sorted(missed):
                    sent = sent_map.get(sn)
                    # Find component name
                    cname = next((c.name for c in components if c.id == cid), cid)
                    text = sent.text[:80] if sent else "?"
                    print(f"    [FN] S{sn} -> {cname}: {text}")

            results[(ds_name, var_name)] = {
                "tp": tp, "fp": fp, "fn_coref": fn_coref,
                "total": len(coref_links), "time": elapsed,
                "p": p, "r": r, "f1": f1,
            }

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")

    # Header
    header = f"{'Variant':<12}"
    for ds in dataset_names:
        header += f" | {ds:>14}"
    header += " | {'MACRO':>8}"
    print(header)
    print("-" * len(header))

    for var_name in variant_names:
        if var_name not in ALL_VARIANTS:
            continue
        row = f"{var_name:<12}"
        f1_vals = []
        for ds in dataset_names:
            key = (ds, var_name)
            if key in results:
                r = results[key]
                row += f" | {r['tp']}tp {r['fp']}fp {r['f1']:.0%}"
                f1_vals.append(r["f1"])
            else:
                row += f" | {'N/A':>14}"
        if f1_vals:
            macro = sum(f1_vals) / len(f1_vals)
            row += f" |  {macro:.1%}"
        print(row)

    # Also print a clean TP/FP-only summary
    print(f"\n{'Variant':<12} | {'Total TP':>8} | {'Total FP':>8} | {'Net':>6}")
    print("-" * 50)
    for var_name in variant_names:
        if var_name not in ALL_VARIANTS:
            continue
        total_tp = sum(results.get((ds, var_name), {}).get("tp", 0) for ds in dataset_names)
        total_fp = sum(results.get((ds, var_name), {}).get("fp", 0) for ds in dataset_names)
        print(f"{var_name:<12} | {total_tp:>8} | {total_fp:>8} | {total_tp - total_fp:>+6}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"])
    parser.add_argument("--variants", nargs="+",
                        default=["discourse", "debate", "A", "B", "C", "D", "E"])
    parser.add_argument("--backend", choices=["claude", "openai"], default="claude",
                        help="LLM backend: claude (default) or openai (GPT-5.2)")
    args = parser.parse_args()

    backend = LLMBackend.OPENAI if args.backend == "openai" else LLMBackend.CLAUDE
    if backend == LLMBackend.OPENAI:
        # Load API key from .env if not already set
        if not os.environ.get("OPENAI_API_KEY"):
            env_path = Path("/mnt/hostshare/ardoco-home/.env")
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("OPENAI_API_KEY="):
                        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
                        break

    run_test(args.datasets, args.variants, backend=backend)
