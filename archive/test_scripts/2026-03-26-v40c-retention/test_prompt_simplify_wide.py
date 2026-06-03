#!/usr/bin/env python3
"""Wide prompt simplification tests — all major prompts in the pipeline.

Tests simplified variants of each prompt by running on real data and comparing
output against V33 cached results and gold standard. Each test is independent
so phases can be run individually.

Usage:
    python test_prompt_simplify_wide.py --phase 1          # Phase 1 classification
    python test_prompt_simplify_wide.py --phase 3          # Phase 3 doc knowledge
    python test_prompt_simplify_wide.py --phase 5          # Phase 5 extraction
    python test_prompt_simplify_wide.py --phase 6          # Phase 6 validation
    python test_prompt_simplify_wide.py --phase 7          # Phase 7 coreference
    python test_prompt_simplify_wide.py --phase 8c         # Phase 8c convention
    python test_prompt_simplify_wide.py --phase 9          # Phase 9 judge
    python test_prompt_simplify_wide.py --phase ilinker2   # ILinker2 seed
    python test_prompt_simplify_wide.py --phase all        # Everything
    python test_prompt_simplify_wide.py --datasets teammates --phase 1
"""

import argparse
import csv
import glob
import json
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.core.data_types import SadSamLink, DocumentKnowledge, ModelKnowledge, DiscourseContext
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient

os.environ.setdefault("CLAUDE_MODEL", "sonnet")

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
CACHE_DIR = Path("./results/phase_cache/v33")


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
    return components, sentences, sent_map, name_to_id


def load_gold(dataset):
    gold_path = BENCHMARK_BASE / dataset
    pattern = str(gold_path / "**" / "goldstandard_sad_*-sam_*.csv")
    files = [f for f in glob.glob(pattern, recursive=True) if "UME" not in f and "code" not in f]
    gold = set()
    for f in files:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                cid = row.get("modelElementID", "")
                sid = row.get("sentence", "")
                if sid and cid:
                    gold.add((int(sid), cid))
    return gold


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Component Classification
# ═══════════════════════════════════════════════════════════════════════════

FEW_SHOT_ORIGINAL = """
EXAMPLE 1:
NAMES: Lexer, Parser, CodeGenerator, Optimizer, Core, Util, AST, SymbolTable, Base
→ architectural: ["Lexer", "Parser", "CodeGenerator", "Optimizer", "AST", "SymbolTable"]
→ ambiguous: ["Core", "Util", "Base"]
Reasoning: Lexer/Parser/Optimizer name specific compilation roles. Core/Util/Base are
organizational labels that tell you nothing about what the component does.

EXAMPLE 2:
NAMES: Scheduler, Dispatcher, MemoryManager, Monitor, Pool, Helper, ProcessTable
→ architectural: ["Scheduler", "Dispatcher", "MemoryManager", "ProcessTable"]
→ ambiguous: ["Monitor", "Pool", "Helper"]
Reasoning: Scheduler/Dispatcher name specific OS roles. Monitor and Pool are ordinary
English words regularly used generically ("monitor performance", "thread pool").
Helper is an organizational label.

EXAMPLE 3:
NAMES: RenderEngine, SceneGraph, Pipeline, Layer, Proxy, Socket, Router
→ architectural: ["RenderEngine", "SceneGraph", "Socket", "Router"]
→ ambiguous: ["Pipeline", "Layer", "Proxy"]
Reasoning: RenderEngine/SceneGraph are CamelCase compounds — always architectural.
Socket/Router name specific networking roles. Pipeline/Layer/Proxy are ordinary words
used generically in documentation ("processing pipeline", "network layer", "behind a proxy").

EXAMPLE 4:
NAMES: PaymentGateway, OrderProcessor, Connector, Controller, Adapter, Worker, Agent
→ architectural: ["PaymentGateway", "OrderProcessor", "Worker"]
→ ambiguous: ["Connector", "Controller", "Adapter", "Agent"]
Reasoning: PaymentGateway/OrderProcessor are CamelCase compounds naming specific roles.
Worker names a specific concurrency mechanism. But Connector/Controller/Adapter/Agent
seem functional yet are GENERIC categories writers use without referring to any specific
component: "a database connector", "the main controller", "a protocol adapter", "a
background agent". They describe WHAT KIND of thing it is, not WHICH specific mechanism
— so they are ambiguous.""".strip()

# SLIM: Drop to 2 examples, keep the critical distinction
FEW_SHOT_SLIM = """
EXAMPLE 1:
NAMES: Lexer, Parser, CodeGenerator, Core, Util, AST, SymbolTable, Base
→ architectural: ["Lexer", "Parser", "CodeGenerator", "AST", "SymbolTable"]
→ ambiguous: ["Core", "Util", "Base"]

EXAMPLE 2:
NAMES: PaymentGateway, OrderProcessor, Connector, Controller, Adapter, Worker
→ architectural: ["PaymentGateway", "OrderProcessor", "Worker"]
→ ambiguous: ["Connector", "Controller", "Adapter"]
Key: "Connector/Controller/Adapter" describe WHAT KIND, not WHICH mechanism → ambiguous.
CamelCase compounds and specific mechanism names → always architectural.""".strip()

CLASSIFY_RULES_ORIGINAL = """RULES:
1. ARCHITECTURAL: Names that refer to a specific role or responsibility. If the name tells you
   WHAT the component does (scheduling, parsing, rendering, storing data, managing users), it is
   architectural — even if the word also exists in a dictionary.
   Multi-word names, CamelCase compounds, and abbreviations (API, TCP, RPC) → always architectural.

2. AMBIGUOUS: Single words that writers regularly use generically in software documentation.
   This includes TWO categories:
   Category A — Organizational labels: core, util, base, helper (tell you nothing about function)
   Category B — Generic functional categories: connector, controller, adapter, agent
   (describe WHAT KIND of thing, not WHICH specific mechanism)
   The test: "Could a technical writer naturally write this word in a sentence about ANY system
   without referring to a specific component?" If yes → ambiguous.
   Key: Scheduler/Router describe HOW (specific mechanism) → ARCHITECTURAL.
         Connector/Controller/Adapter describe WHAT KIND (generic category) → AMBIGUOUS."""

CLASSIFY_RULES_SLIM = """RULES:
1. ARCHITECTURAL: Names that describe a specific role or responsibility.
   Multi-word, CamelCase, and abbreviations → always architectural.
2. AMBIGUOUS: Single words used generically in documentation — organizational labels
   (core, util, base) or generic categories (connector, controller, adapter).
   Test: "Could a writer use this word in ANY system without meaning a specific component?" → ambiguous."""


def test_phase1(datasets, llm):
    """Test Phase 1 component classification with original vs simplified prompts."""
    print("\n" + "=" * 80)
    print("  PHASE 1 TEST: Component Classification Prompt Variants")
    print("=" * 80)

    for ds in datasets:
        components, sentences, sent_map, name_to_id = load_dataset(ds)
        cached = load_checkpoint(ds, "phase1")
        if not cached:
            print(f"\n  {ds}: SKIP (no cache)")
            continue

        cached_mk = cached["model_knowledge"]
        cached_ambig = cached_mk.ambiguous_names if cached_mk else set()
        cached_arch = cached_mk.architectural_names if cached_mk else set()

        names = [c.name for c in components]
        print(f"\n{'─' * 70}")
        print(f"  {ds}: {len(names)} components")
        print(f"  V33 cached: arch={sorted(cached_arch)}, ambig={sorted(cached_ambig)}")
        print(f"{'─' * 70}")

        variants = [
            ("ORIG", FEW_SHOT_ORIGINAL, CLASSIFY_RULES_ORIGINAL),
            ("SLIM_FS", FEW_SHOT_SLIM, CLASSIFY_RULES_ORIGINAL),
            ("SLIM_RULES", FEW_SHOT_ORIGINAL, CLASSIFY_RULES_SLIM),
            ("SLIM_BOTH", FEW_SHOT_SLIM, CLASSIFY_RULES_SLIM),
        ]

        for vname, few_shot, rules in variants:
            prompt = f"""Classify these software architecture component names.

NAMES: {', '.join(names)}

{few_shot}

NOW CLASSIFY THE NAMES ABOVE.

Return JSON:
{{
  "architectural": ["names that identify specific components"],
  "ambiguous": ["names that could easily be used as ordinary words in documentation"]
}}

{rules}

JSON only:"""

            t0 = time.time()
            data = llm.extract_json(llm.query(prompt, timeout=100))
            elapsed = time.time() - t0

            if data:
                valid = set(names)
                arch = set(data.get("architectural", [])) & valid
                raw_ambig = set(data.get("ambiguous", [])) & valid
                # Apply structural guard (same as V33)
                ambig = set()
                for n in raw_ambig:
                    if len(n.split()) > 1:
                        continue
                    if re.search(r'[a-z][A-Z]', n):
                        continue
                    if n.isupper():
                        continue
                    ambig.add(n)
            else:
                arch, ambig = set(), set()

            match_arch = arch == cached_arch
            match_ambig = ambig == cached_ambig
            diff_arch = (arch - cached_arch, cached_arch - arch)
            diff_ambig = (ambig - cached_ambig, cached_ambig - ambig)

            status = "MATCH" if match_arch and match_ambig else "DIFF"
            print(f"\n  [{vname}] {elapsed:.0f}s — {status}")
            print(f"    arch={sorted(arch)}")
            print(f"    ambig={sorted(ambig)}")
            if not match_arch:
                print(f"    arch diff: +{diff_arch[0]} -{diff_arch[1]}")
            if not match_ambig:
                print(f"    ambig diff: +{diff_ambig[0]} -{diff_ambig[1]}")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Document Knowledge
# ═══════════════════════════════════════════════════════════════════════════

P3_EXTRACT_ORIGINAL = """Find all alternative names used for these components in the document.

COMPONENTS: {comp_names}

WHAT TO FIND:
1. ABBREVIATIONS: Short forms explicitly introduced in the document.
   Rule: The abbreviation must be defined in the text, e.g., "Full Name (FN)" introduces FN.
   Like "Abstract Syntax Tree (AST)" defines AST — look for the same parenthetical pattern.

2. SYNONYMS: Alternative names that SPECIFICALLY refer to one component.
   Rule: The alternative name must unambiguously identify exactly ONE component.
   APPROVE: A proper name, role title, or technical alias used interchangeably with the component
   REJECT: A generic description that could apply to anything (like "the system" or "the process")

3. PARTIAL REFERENCES: A shorter form of a multi-word component name used alone.
   Rule: A trailing word from a multi-word name that, in this document, consistently means the full name.
   APPROVE: Only if the short form is unambiguous — no other component shares this word
   REJECT: Ordinary words that have plain English meanings beyond the component

DOCUMENT:
{doc_lines}

Return JSON:
{{
  "abbreviations": {{"short_form": "FullComponent"}},
  "synonyms": {{"specific_alternative_name": "FullComponent"}},
  "partial_references": {{"partial_name": "FullComponent"}}
}}
JSON only:"""

P3_EXTRACT_SLIM = """Find alternative names for these components in the document.

COMPONENTS: {comp_names}

Find:
1. ABBREVIATIONS: Short forms introduced with parentheses, e.g., "Full Name (FN)"
2. SYNONYMS: Alternative names that unambiguously refer to exactly one component.
   Reject generic terms like "the system" or "the process".
3. PARTIAL REFERENCES: Trailing words of multi-word names used alone (if unambiguous).
   Reject ordinary English words with meanings beyond the component.

DOCUMENT:
{doc_lines}

Return JSON:
{{"abbreviations": {{"short": "Full"}}, "synonyms": {{"alt": "Full"}}, "partial_references": {{"partial": "Full"}}}}
JSON only:"""


def test_phase3(datasets, llm):
    """Test Phase 3 document knowledge extraction with simplified prompts."""
    print("\n" + "=" * 80)
    print("  PHASE 3 TEST: Document Knowledge Extraction")
    print("=" * 80)

    for ds in datasets:
        components, sentences, sent_map, name_to_id = load_dataset(ds)
        cached = load_checkpoint(ds, "phase3")
        if not cached:
            print(f"\n  {ds}: SKIP")
            continue

        dk = cached["doc_knowledge"]
        comp_names = ', '.join(c.name for c in components)
        doc_lines = chr(10).join(s.text for s in sentences[:150])

        print(f"\n{'─' * 70}")
        print(f"  {ds}: {len(components)} components, {len(sentences)} sentences")
        print(f"  V33: abbrev={dict(dk.abbreviations)}, syn={dict(dk.synonyms)}, partial={dict(dk.partial_references)}")
        print(f"{'─' * 70}")

        for vname, template in [("ORIG", P3_EXTRACT_ORIGINAL), ("SLIM", P3_EXTRACT_SLIM)]:
            prompt = template.format(comp_names=comp_names, doc_lines=doc_lines)

            t0 = time.time()
            data = llm.extract_json(llm.query(prompt, timeout=150))
            elapsed = time.time() - t0

            comp_name_set = {c.name for c in components}
            abbrevs, syns, partials = {}, {}, {}
            if data:
                for k, v in data.get("abbreviations", {}).items():
                    if v in comp_name_set:
                        abbrevs[k] = v
                for k, v in data.get("synonyms", {}).items():
                    if v in comp_name_set:
                        syns[k] = v
                for k, v in data.get("partial_references", {}).items():
                    if v in comp_name_set:
                        partials[k] = v

            # Compare with cached
            cached_all = set()
            for d in [dk.abbreviations, dk.synonyms, dk.partial_references]:
                cached_all.update(d.items())
            new_all = set()
            for d in [abbrevs, syns, partials]:
                new_all.update(d.items())

            common = cached_all & new_all
            only_cached = cached_all - new_all
            only_new = new_all - cached_all

            print(f"\n  [{vname}] {elapsed:.0f}s")
            print(f"    abbrev={abbrevs}, syn={syns}, partial={partials}")
            print(f"    vs V33: {len(common)} match, {len(only_cached)} lost, {len(only_new)} new")
            if only_cached:
                print(f"    LOST: {only_cached}")
            if only_new:
                print(f"    NEW:  {only_new}")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: Entity Extraction
# ═══════════════════════════════════════════════════════════════════════════

P5_ORIGINAL = """Extract ALL references to software architecture components from this document.

COMPONENTS: {comp_names}
{alias_line}

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
{doc_block}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence", "match_type": "exact|synonym|partial|functional"}}]}}
JSON only:"""

P5_SLIM = """Extract references to architecture components from this document.

COMPONENTS: {comp_names}
{alias_line}

Include when: component name/alias appears in sentence, describes component's role,
or component participates in an interaction. Also match passive/prepositional phrases.
Exclude: names inside dotted paths, ordinary English word usage.
Favor inclusion — later stages filter borderline cases.

DOCUMENT:
{doc_block}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "evidence", "match_type": "exact|synonym|partial|functional"}}]}}
JSON only:"""


def test_phase5(datasets, llm):
    """Test Phase 5 extraction with simplified prompt on first batch only."""
    print("\n" + "=" * 80)
    print("  PHASE 5 TEST: Entity Extraction (first batch only)")
    print("=" * 80)

    for ds in datasets:
        components, sentences, sent_map, name_to_id = load_dataset(ds)
        cached = load_checkpoint(ds, "phase5")
        cached3 = load_checkpoint(ds, "phase3")
        if not cached:
            print(f"\n  {ds}: SKIP")
            continue

        cached_candidates = cached.get("candidates", [])
        cached_pairs = {(c.sentence_number, c.component_id) for c in cached_candidates}

        comp_names = ', '.join(c.name for c in components)
        gold = load_gold(ds)

        mappings = []
        if cached3 and cached3.get("doc_knowledge"):
            dk = cached3["doc_knowledge"]
            mappings.extend([f"{a}={c}" for a, c in dk.abbreviations.items()])
            mappings.extend([f"{s}={c}" for s, c in dk.synonyms.items()])
            mappings.extend([f"{p}={c}" for p, c in dk.partial_references.items()])
        alias_line = f'KNOWN ALIASES: {", ".join(mappings[:20])}' if mappings else ''

        # First batch only (50 sentences)
        batch = sentences[:50]
        doc_block = chr(10).join([f"S{s.number}: {s.text}" for s in batch])

        print(f"\n{'─' * 70}")
        print(f"  {ds}: first 50 sentences, V33 found {len(cached_candidates)} total candidates")
        print(f"{'─' * 70}")

        for vname, template in [("ORIG", P5_ORIGINAL), ("SLIM", P5_SLIM)]:
            prompt = template.format(comp_names=comp_names, alias_line=alias_line, doc_block=doc_block)

            t0 = time.time()
            data = llm.extract_json(llm.query(prompt, timeout=240))
            elapsed = time.time() - t0

            pairs = set()
            if data:
                for ref in data.get("references", []):
                    snum = ref.get("sentence")
                    cname = ref.get("component")
                    if not (snum and cname):
                        continue
                    if isinstance(snum, str):
                        snum = snum.lstrip("S")
                    try:
                        snum = int(snum)
                    except (ValueError, TypeError):
                        continue
                    cid = name_to_id.get(cname)
                    if cid:
                        pairs.add((snum, cid))

            # Compare with cached (filter to first batch sentences)
            batch_snums = {s.number for s in batch}
            cached_batch = {(s, c) for s, c in cached_pairs if s in batch_snums}
            gold_batch = {(s, c) for s, c in gold if s in batch_snums}

            common = pairs & cached_batch
            only_new = pairs - cached_batch
            only_cached = cached_batch - pairs

            tp_new = len(only_new & gold_batch)
            fp_new = len(only_new - gold_batch)
            tp_lost = len(only_cached & gold_batch)
            fp_lost = len(only_cached - gold_batch)

            print(f"\n  [{vname}] {elapsed:.0f}s: {len(pairs)} refs (batch 1)")
            print(f"    vs V33: {len(common)} match, {len(only_cached)} lost ({tp_lost} TP, {fp_lost} FP), "
                  f"{len(only_new)} new ({tp_new} TP, {fp_new} FP)")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: Validation Prompt
# ═══════════════════════════════════════════════════════════════════════════

P6_ORIGINAL = """Validate component references in a software architecture document. {focus}

COMPONENTS: {comp_names}

{ctx}

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

SINGLE-WORD COMPONENT NAMES (important for names like single common English words):
When the system has a component with a single-word name that is also an ordinary English
word, apply these rules:
- APPROVE when the word is used as a NOUN referring to a part of the system in an
  architectural context (describing system behavior, interactions, responsibilities).
  Even if lowercase, the word refers to the component when the sentence discusses
  the system's architecture.
- REJECT when the word appears inside a DOTTED PACKAGE PATH or QUALIFIED NAME
  (e.g., "x.utils", "x.api", "x.datatransfer"). The dotted path refers to a
  sub-package inside the component, not to the component itself.
- REJECT when the word is used as a plain English adjective, verb, or in an idiom
  unrelated to the system (e.g., "common ground", "persistent effort").

CASES:
{cases}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

P6_SLIM = """Validate component references. {focus}

COMPONENTS: {comp_names}

{ctx}

APPROVE when: component is actor/subject, heading names it, or sentence describes what it does.
REJECT when: name used as ordinary English word, modifier in a phrase, or about implementation details.
For single-word names: APPROVE as architectural noun. REJECT in dotted paths (x.utils) or as plain adjective/verb.

CASES:
{cases}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""


def test_phase6(datasets, llm):
    """Test Phase 6 validation on a sample of links from cached candidates."""
    print("\n" + "=" * 80)
    print("  PHASE 6 TEST: Validation Prompt Variants")
    print("=" * 80)

    for ds in datasets:
        components, sentences, sent_map, name_to_id = load_dataset(ds)
        cached5 = load_checkpoint(ds, "phase5")
        cached6 = load_checkpoint(ds, "phase6")
        if not cached5 or not cached6:
            print(f"\n  {ds}: SKIP")
            continue

        gold = load_gold(ds)
        comp_names = ', '.join(c.name for c in components)

        # Get candidates that needed LLM validation (not auto-approved)
        candidates = cached5.get("candidates", [])
        validated = cached6.get("validated", [])
        val_pairs = {(v.sentence_number, v.component_id) for v in validated}

        # Build test cases from the first 15 candidates
        test_cands = candidates[:15]
        if not test_cands:
            print(f"\n  {ds}: no candidates")
            continue

        cases_lines = []
        for i, c in enumerate(test_cands):
            sent = sent_map.get(c.sentence_number)
            text = sent.text if sent else "?"
            cases_lines.append(f'Case {i+1}: S{c.sentence_number} "{text}"\n  Candidate: {c.component_name}')

        print(f"\n{'─' * 70}")
        print(f"  {ds}: testing {len(test_cands)} candidates")
        print(f"{'─' * 70}")

        for vname, template in [("ORIG", P6_ORIGINAL), ("SLIM", P6_SLIM)]:
            prompt = template.format(
                focus="Focus on ACTOR role: is the component performing an action or being described?",
                comp_names=comp_names,
                ctx="",
                cases=chr(10).join(cases_lines)
            )

            t0 = time.time()
            data = llm.extract_json(llm.query(prompt, timeout=120))
            elapsed = time.time() - t0

            approved_idx = set()
            if data:
                for v in data.get("validations", []):
                    idx = v.get("case", 0) - 1
                    if 0 <= idx < len(test_cands) and v.get("approve", False):
                        approved_idx.add(idx)

            # Compare each with cached validation and gold
            results = []
            for i, c in enumerate(test_cands):
                was_validated = (c.sentence_number, c.component_id) in val_pairs
                is_gold = (c.sentence_number, c.component_id) in gold
                now_approved = i in approved_idx
                results.append((c, was_validated, is_gold, now_approved))

            match_count = sum(1 for _, wv, _, na in results if wv == na)
            tp_saved = sum(1 for _, wv, ig, na in results if not wv and na and ig)
            tp_lost = sum(1 for _, wv, ig, na in results if wv and not na and ig)
            fp_saved = sum(1 for _, wv, ig, na in results if wv and not na and not ig)
            fp_added = sum(1 for _, wv, ig, na in results if not wv and na and not ig)

            print(f"\n  [{vname}] {elapsed:.0f}s: {len(approved_idx)} approved, "
                  f"{match_count}/{len(test_cands)} match V33")
            if tp_saved:
                print(f"    +{tp_saved} TP recovered")
            if tp_lost:
                print(f"    -{tp_lost} TP lost")
            if fp_saved:
                print(f"    -{fp_saved} FP removed")
            if fp_added:
                print(f"    +{fp_added} FP added")

            for c, wv, ig, na in results:
                if wv != na:
                    label = "TP" if ig else "FP"
                    change = "NEW_APPROVE" if na else "NEW_REJECT"
                    print(f"    [{label}] {change}: S{c.sentence_number}→{c.component_name}")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7: Coreference
# ═══════════════════════════════════════════════════════════════════════════

COREF_RULES_ORIGINAL = """RULES (all must hold):
1. The component name (or known alias) MUST appear verbatim in the antecedent sentence
2. The antecedent MUST be within the previous 3 sentences
3. The pronoun MUST grammatically refer back to that component as its subject
4. If the pronoun could refer to multiple things, DO NOT resolve it"""

COREF_RULES_SLIM = """RULES: Component name must appear in antecedent (within 3 prior sentences).
Pronoun must grammatically refer to it. Skip ambiguous references."""


def test_phase7(datasets, llm):
    """Test Phase 7 coreference with simplified rules on a sample batch."""
    print("\n" + "=" * 80)
    print("  PHASE 7 TEST: Coreference Rules (debate mode, first batch)")
    print("=" * 80)

    for ds in datasets:
        components, sentences, sent_map, name_to_id = load_dataset(ds)
        cached7 = load_checkpoint(ds, "phase7")
        if not cached7:
            print(f"\n  {ds}: SKIP")
            continue

        gold = load_gold(ds)
        cached_coref = cached7.get("coref_links", [])
        cached_pairs = {(l.sentence_number, l.component_id) for l in cached_coref}
        comp_names = ', '.join(c.name for c in components)

        # First batch of 20 sentences (debate mode)
        batch = sentences[:20]
        ctx_sents = batch
        doc_lines = [f"*S{s.number}: {s.text}" for s in ctx_sents]

        print(f"\n{'─' * 70}")
        print(f"  {ds}: V33 found {len(cached_coref)} coref links total")
        print(f"{'─' * 70}")

        for vname, rules in [("ORIG", COREF_RULES_ORIGINAL), ("SLIM", COREF_RULES_SLIM)]:
            prompt = f"""Resolve pronoun references to architecture components.

COMPONENTS: {comp_names}

DOCUMENT (* = analyze these sentences):
{chr(10).join(doc_lines)}

Find pronouns (it, they, this, these) in starred sentences that refer to a component.

{rules}

Return JSON:
{{"resolutions": [{{"sentence": N_INTEGER, "pronoun": "it", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

Only include resolutions you are CERTAIN about. JSON only:"""

            t0 = time.time()
            data = llm.extract_json(llm.query(prompt, timeout=100))
            elapsed = time.time() - t0

            pairs = set()
            if data:
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
                    pairs.add((snum, name_to_id[comp]))

            batch_snums = {s.number for s in batch}
            cached_batch = {(s, c) for s, c in cached_pairs if s in batch_snums}

            common = pairs & cached_batch
            only_new = pairs - cached_batch
            only_cached = cached_batch - pairs

            tp_new = sum(1 for s, c in only_new if (s, c) in gold)
            tp_lost = sum(1 for s, c in only_cached if (s, c) in gold)

            print(f"\n  [{vname}] {elapsed:.0f}s: {len(pairs)} resolutions (batch 1)")
            print(f"    vs V33: {len(common)} match, {len(only_cached)} lost ({tp_lost} TP), "
                  f"{len(only_new)} new ({tp_new} TP)")


# ═══════════════════════════════════════════════════════════════════════════
# ILinker2: Pass A/B seed prompts
# ═══════════════════════════════════════════════════════════════════════════

PASS_A_ORIGINAL = """You are a software architecture traceability expert.

ARCHITECTURE COMPONENTS:
{comp_block}

DOCUMENT:
{doc_block}

TASK: For each sentence, identify which architecture components are EXPLICITLY mentioned or referenced.

A valid reference is:
- Exact name: the component name appears verbatim in the sentence
- Synonym: a well-known alternative name for the component (e.g., "code generator" → "CodeGenerator")
- Abbreviation: a shortened form (e.g., "AST" → "AbstractSyntaxTree")
- Partial name: a distinctive sub-phrase of the component name that unambiguously identifies it (e.g., "the scheduler" → "TaskScheduler")

NOT a valid reference:
- A component name that only appears inside a dotted path (e.g., "renderer.utils.config" does NOT reference "Renderer")
- A generic English word used in its ordinary sense (e.g., "optimized code" does NOT reference "Optimizer")
- A sentence that merely describes related functionality without naming or clearly referring to the component

Return ONLY valid JSON:
{{"links": [{{"s": N_INTEGER, "c": "ComponentName", "text": "matched text", "type": "exact|synonym|partial"}}]}}

Precision is critical — only include links with clear textual evidence."""

PASS_A_SLIM = """ARCHITECTURE COMPONENTS:
{comp_block}

DOCUMENT:
{doc_block}

TASK: For each sentence, find architecture components EXPLICITLY mentioned or referenced.

Valid: exact name, synonym, abbreviation, or unambiguous partial name in the sentence text.
Invalid: names inside dotted paths, generic English words, or no clear textual evidence.

Return JSON:
{{"links": [{{"s": N_INTEGER, "c": "ComponentName", "text": "matched text", "type": "exact|synonym|partial"}}]}}
Precision is critical."""

PASS_B_ORIGINAL = """You are a software architecture traceability expert performing an independent review.

ARCHITECTURE COMPONENTS:
{comp_block}

DOCUMENT:
{doc_block}

TASK: For each sentence, determine which architecture components are ARCHITECTURALLY RELEVANT — meaning the sentence describes their role, behavior, interactions, or responsibilities.

"Architecturally relevant" means ANY of these:
- The component PERFORMS an action described in the sentence
- The component RECEIVES an action or is acted upon
- The component INTERACTS WITH another component described in the sentence
- The component's role, responsibility, or behavior is described

IMPORTANT: A sentence like "X connects to Y" is relevant to BOTH X and Y.
Do not limit to just the grammatical subject — report ALL named components that participate.

CAUTION with single-word component names (e.g., "Logic", "Storage", "Client"):
- These words have ordinary English meanings beyond the component
- Only report them when the sentence SPECIFICALLY discusses that component's architectural role
- "the system logic" or "client request" uses the word generically → do NOT report
- "the Logic component handles requests" or "Client connects to Server" → report

Rules:
- Only report components that are explicitly named, abbreviated, or identified by a clear synonym/partial name IN THE SENTENCE TEXT.
- Do NOT report pronoun-only references (e.g., "It does X" — skip these).
- Do NOT match component names inside dotted package paths (e.g., "renderer.utils.config" does NOT reference "Renderer").
- Do NOT match generic English words used in their ordinary sense (e.g., "optimized code" does NOT reference "Optimizer").

Return ONLY valid JSON:
{{"links": [{{"s": N_INTEGER, "c": "ComponentName", "text": "evidence", "type": "exact|synonym|partial"}}]}}

Report ALL architecturally relevant components per sentence, not just the primary subject."""

PASS_B_SLIM = """ARCHITECTURE COMPONENTS:
{comp_block}

DOCUMENT:
{doc_block}

TASK: For each sentence, find components that are ARCHITECTURALLY RELEVANT — the sentence
describes their role, behavior, interactions, or responsibilities.

Report ALL participating components (not just grammatical subject). "X connects to Y" → both X and Y.

CAUTION with single-word names (e.g., "Logic", "Storage"): only report when the sentence
discusses that component's architectural role, not generic English usage.

Rules: Must be explicitly named/abbreviated in text. Skip pronouns. Skip dotted paths. Skip generic word usage.

Return JSON:
{{"links": [{{"s": N_INTEGER, "c": "ComponentName", "text": "evidence", "type": "exact|synonym|partial"}}]}}"""


def test_ilinker2(datasets, llm):
    """Test ILinker2 Pass A/B with simplified prompts on first batch."""
    print("\n" + "=" * 80)
    print("  ILINKER2 TEST: Pass A/B Prompt Variants (first batch)")
    print("=" * 80)

    for ds in datasets:
        components, sentences, sent_map, name_to_id = load_dataset(ds)
        cached4 = load_checkpoint(ds, "phase4")
        if not cached4:
            print(f"\n  {ds}: SKIP")
            continue

        gold = load_gold(ds)
        cached_links = cached4.get("transarc_links", [])
        cached_pairs = {(l.sentence_number, l.component_id) for l in cached_links}

        comp_block = "\n".join(f"- {c.name} (id: {c.id})" for c in components)
        batch = sentences[:50]
        doc_block = "\n".join(f"S{s.number}: {s.text}" for s in batch)

        print(f"\n{'─' * 70}")
        print(f"  {ds}: V33 ILinker2 found {len(cached_links)} total links")
        print(f"{'─' * 70}")

        for pass_name, orig, slim in [("PASS_A", PASS_A_ORIGINAL, PASS_A_SLIM),
                                       ("PASS_B", PASS_B_ORIGINAL, PASS_B_SLIM)]:
            for vname, template in [("ORIG", orig), ("SLIM", slim)]:
                prompt = template.format(comp_block=comp_block, doc_block=doc_block)

                t0 = time.time()
                resp = llm.query(prompt, timeout=300)
                data = llm.extract_json(resp)
                elapsed = time.time() - t0

                pairs = set()
                if data and "links" in data:
                    for lk in data["links"]:
                        snum = lk.get("s")
                        cname = lk.get("c")
                        if not (snum and cname):
                            continue
                        if isinstance(snum, str):
                            snum = snum.lstrip("S")
                        try:
                            snum = int(snum)
                        except (ValueError, TypeError):
                            continue
                        cid = name_to_id.get(cname)
                        if cid:
                            pairs.add((snum, cid))

                batch_snums = {s.number for s in batch}
                cached_batch = {(s, c) for s, c in cached_pairs if s in batch_snums}
                gold_batch = {(s, c) for s, c in gold if s in batch_snums}

                tp = len(pairs & gold_batch)
                fp = len(pairs - gold_batch)
                common = len(pairs & cached_batch)

                print(f"  [{pass_name} {vname}] {elapsed:.0f}s: {len(pairs)} links "
                      f"(TP={tp}, FP={fp}, match_v33={common})")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    parser.add_argument("--phase", default="all",
                        choices=["1", "3", "5", "6", "7", "8c", "9", "ilinker2", "all"])
    args = parser.parse_args()

    datasets = [d for d in args.datasets if d in DATASETS]
    llm = LLMClient()

    print(f"Datasets: {', '.join(datasets)}")
    print(f"Phase: {args.phase}")

    if args.phase in ("1", "all"):
        test_phase1(datasets, llm)
    if args.phase in ("3", "all"):
        test_phase3(datasets, llm)
    if args.phase in ("5", "all"):
        test_phase5(datasets, llm)
    if args.phase in ("6", "all"):
        test_phase6(datasets, llm)
    if args.phase in ("7", "all"):
        test_phase7(datasets, llm)
    if args.phase in ("8c", "all"):
        # Import from the other test file
        from test_prompt_simplify import (
            GUIDE_ORIGINAL, GUIDE_SLIM, GUIDE_SLIM_V2,
            reconstruct_pre8c_links, run_convention_filter,
            links_to_pairs, eval_metrics
        )
        print("\n" + "=" * 80)
        print("  PHASE 8c: See test_prompt_simplify.py --phase 8c")
        print("=" * 80)
    if args.phase in ("9", "all"):
        print("\n" + "=" * 80)
        print("  PHASE 9: See test_prompt_simplify.py --phase 9")
        print("=" * 80)
    if args.phase in ("ilinker2", "all"):
        test_ilinker2(datasets, llm)


if __name__ == "__main__":
    main()
