"""S-Linker: Standalone DAG-based SAD-SAM trace link recovery.

Derived from ILinker2V39. Key changes vs V39:
- DAG workflow architecture matching the paper's 3-tier fan-out/fan-in design
- Concurrent execution within tiers via ThreadPoolExecutor
- Dead code removed: unused imports (Sentence, EntityMention), dead method
  (_build_context_string), dead parameter (implicit_set), dead doc_profile fields
- Bug fix: duplicate return in _judge_syn_safe when partial_groups is empty
- Simplified document profiling (Phase 0 merged into _compute_complexity)

Architecture (from paper §3):
  Tier 1: Knowledge Acquisition
    Concurrent: model analysis, document profiling, document knowledge,
                seed extraction (all independent)
    Then: pattern learning + multiword enrichment (need model analysis),
          then partial usage classification
  Tier 2: Link Recovery
    Concurrent: entity pipeline (extract→guard→recover→validate)
                ∥ coreference resolution
    Then: partial reference injection (needs both)
  Tier 3: Merge and Judicial Review (sequential)
    Priority dedup → parent-overlap guard → convention-aware boundary
    filter → triage-based judicial review
"""

import json
import os
import pickle
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from llm_sad_sam.core.data_types import (
    SadSamLink, CandidateLink,
    ModelKnowledge, DocumentKnowledge, LearnedPatterns,
)
from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.linkers.experimental.ilinker2 import ILinker2
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend

# ── 3-step reasoning guide for convention-aware boundary filtering ────
# All examples use abstract placeholders (X, Y) or safe SE textbook domains.
# Audited against BENCHMARK_TABOO.md — zero overlap with benchmark vocabulary.
CONVENTION_GUIDE = """### STEP 1 — Hierarchical name reference (not about the component itself)?

The most common reason for NO_LINK: the sentence mentions the component name only
as part of a HIERARCHICAL/QUALIFIED NAME (dotted path, namespace, module path) that
refers to a nested sub-unit, not the component's own architectural role.

Software documentation commonly uses hierarchical naming (e.g., "X.utils", "X/handlers",
"X::impl") to refer to parts inside a component. The component name appears only as
a prefix, not as the subject.

Recognize these patterns — all are NO_LINK for component X:
- "X.utils provides helper functions" — dotted sub-unit reference
- "X.handlers, X.mappers, X.adapters follow a pipeline" — listing sub-packages of X
- "Classes in the X.impl package are not exported" — even with
  architectural language, if the subject is X's sub-unit → NO_LINK
- Bare name mixed with qualified paths: "X, Y.adapters, Y.transformers follow
  a pipeline design" — treat ALL as hierarchical references → NO_LINK

KEY DISTINCTION: Sentences that describe what X DOES or HOW X INTERACTS with other
components are LINK, even if they mention implementation details (e.g., "X uses Y
technology for Z" → LINK for X, because it describes X's behavior).

EXCEPTION: If the sentence also explicitly names the target component AS A PROPER
NOUN with the word "component" (e.g., "for the X component") → LINK.

Cross-reference rule: A sub-unit sentence mentioning a DIFFERENT component
in an architectural role is LINK for that other component.

### STEP 2 — Component name confused with a different entity?

**2a. Technology / methodology confusion:**

CRITICAL RULE: If a component IS NAMED AFTER a technology (e.g., the architecture
has a component called "Kafka Broker" or "Nginx Proxy" or "Zookeeper"), then ANY sentence
describing that technology's capabilities, role, or behavior IS about the component → LINK.
This rule applies because architecture components are often named after the technology they wrap.

NO_LINK ONLY when:
- The sentence describes a technology that is NOT one of our components
- The name appears in a compound entity unrelated to the component
  ("X Protocol specification" → NO_LINK for "X" if X is not about that protocol)
- Uses the name as part of a METHODOLOGY ("X testing in CI" → NO_LINK for "X")

LINK when:
- The technology IS one of our architecture components (always LINK)
- Components INTERACT with or connect THROUGH the technology

**2b. Generic word collision:**
NO_LINK — narrow, non-architectural sense:
- Process/activity modifier: "throttle X", "batch X", "polling X"
- Hardware/deployment: "a physical rack-mounted appliance", "multi-socket machine"
- Possessive/personal: "her bookmarks", "their account"

LINK — system-level architectural sense:
- System name + word: "the [System] gateway"
- Architectural role: "the orchestrator routes jobs to the gateway"

### STEP 3 — Default: LINK.
If neither Step 1 nor Step 2 applies → LINK.

### IMPORTANT GUARDRAILS:
- Multi-word component names (e.g., "Kafka Broker", "Nginx Proxy") are NEVER generic words → LINK
- CamelCase identifiers are NEVER generic words → LINK
- Sentences describing how components interact, connect, or communicate → LINK for ALL components involved (not just the grammatical subject). "X connects to Y" is LINK for both X and Y.
- Sentences about what a component does, provides, or handles → LINK
- A component does NOT need to be the grammatical subject to be relevant. If a sentence says "X sends data to Y", both X and Y get LINK.
- Only use NO_LINK when you are CONFIDENT the name is NOT used as a component reference

### Priority:
Be AGGRESSIVE with NO_LINK on sub-package descriptions (Step 1).
For Step 2, only NO_LINK when confident. Default to LINK."""


class SLinker:
    """DAG-based SAD-SAM trace link recovery (standalone, no inheritance)."""

    CONTEXT_WINDOW = 3
    PRONOUN_PATTERN = re.compile(
        r'\b(it|they|this|these|that|those|its|their|the component|the service)\b',
        re.IGNORECASE
    )
    SOURCE_PRIORITY = {
        "seed": 5, "validated": 4, "entity": 3,
        "coreference": 2, "partial_inject": 1,
    }

    # Few-shot examples for model ambiguity classification (safe textbook domains)
    _FEW_SHOT = """
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

    def __init__(self, backend: Optional[LLMBackend] = None):
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        self.llm = LLMClient(backend=backend or LLMBackend.CLAUDE)
        self.model_knowledge: Optional[ModelKnowledge] = None
        self.doc_knowledge: Optional[DocumentKnowledge] = None
        self.learned_patterns: Optional[LearnedPatterns] = None
        self._is_complex: Optional[bool] = None
        self._phase_log = []
        self._ilinker2 = ILinker2(backend=self.llm.backend)
        self._activity_partials: set = set()
        self._components = []
        self.GENERIC_COMPONENT_WORDS: set = set()
        self.GENERIC_PARTIALS: set = set()
        print(f"SLinker (DAG-based standalone)")
        print(f"  Backend: {self.llm.backend.value}, Model: {os.environ.get('CLAUDE_MODEL', 'default')}")

    # ═══════════════════════════════════════════════════════════════════════
    # DAG Infrastructure
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _run_parallel(tasks):
        """Run named tasks concurrently, wait for all. Returns {name: result}.

        On first failure, cancels remaining futures and re-raises.
        """
        if len(tasks) == 1:
            name, fn = next(iter(tasks.items()))
            return {name: fn()}
        results = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(fn): name for name, fn in tasks.items()}
            try:
                for fut in as_completed(futures):
                    name = futures[fut]
                    results[name] = fut.result()
            except Exception:
                for other in futures:
                    other.cancel()
                raise
        return results

    # ═══════════════════════════════════════════════════════════════════════
    # Main Entry Point — DAG Orchestration
    # ═══════════════════════════════════════════════════════════════════════

    def link(self, text_path, model_path, transarc_csv=None):
        """Recover trace links between SAD and SAM via 3-tier DAG pipeline.

        Args:
            text_path: Path to documentation text file (one sentence per line).
            model_path: Path to PCM .repository file.
            transarc_csv: Accepted for API compatibility; unused (seed is ILinker2).

        Returns:
            list[SadSamLink]: Recovered trace links.
        """
        if transarc_csv:
            print("  WARNING: transarc_csv provided but SLinker uses ILinker2 seed; ignoring.")
        self._phase_log = []
        t0 = time.time()

        # Load raw data
        components = parse_pcm_repository(model_path)
        sentences = DocumentLoader.load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        sent_map = DocumentLoader.build_sent_map(sentences)
        self._components = components

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        # ═══ TIER 1: Knowledge Acquisition (all independent) ═══
        print("\n[Tier 1] Knowledge Acquisition (parallel)")
        t1 = self._run_parallel({
            "model": lambda: self._analyze_model(components),
            "complexity": lambda: self._compute_complexity(sentences, components),
            "doc_knowledge": lambda: self._learn_document_knowledge_enriched(sentences, components),
            "seed": lambda: self._run_seed(text_path, model_path),
        })

        self.model_knowledge = t1["model"]
        self._is_complex = t1["complexity"]
        self.doc_knowledge = t1["doc_knowledge"]
        seed_links = t1["seed"]
        seed_set = {(l.sentence_number, l.component_id) for l in seed_links}

        # Derive generic word sets from model analysis
        self._compute_generic_sets(components)

        ambig = self.model_knowledge.ambiguous_names
        print(f"  Model: {len(ambig)} ambiguous (of {len(components)} components)")
        print(f"  Complexity: complex={self._is_complex}")
        print(f"  Doc knowledge: {len(self.doc_knowledge.abbreviations)} abbrev, "
              f"{len(self.doc_knowledge.synonyms)} syn, "
              f"{len(self.doc_knowledge.partial_references)} partial")
        print(f"  Seed: {len(seed_links)} links")
        print(f"  Generic words: {sorted(self.GENERIC_COMPONENT_WORDS)}")
        print(f"  Generic partials: {sorted(self.GENERIC_PARTIALS)}")

        self._log("tier1", {"sents": len(sentences), "comps": len(components)},
                  {"ambig": len(ambig), "seed": len(seed_links),
                   "abbrev": len(self.doc_knowledge.abbreviations)})

        self._save_phase(text_path, "tier1", {
            "model_knowledge": self.model_knowledge,
            "is_complex": self._is_complex,
            "doc_knowledge": self.doc_knowledge,
            "seed_links": seed_links,
            "seed_set": seed_set,
            "generic_component_words": self.GENERIC_COMPONENT_WORDS,
            "generic_partials": self.GENERIC_PARTIALS,
        })

        # ═══ TIER 1.5: Knowledge Enrichment (needs Tier 1) ═══
        print("\n[Tier 1.5] Knowledge Enrichment (parallel)")
        # Thread safety: _enrich_multiword_partials mutates self.doc_knowledge.partial_references
        # in-place. This is safe because _learn_patterns_with_debate does NOT read doc_knowledge.
        # Do not add doc_knowledge reads to pattern learning without removing the parallelism.
        t1_5 = self._run_parallel({
            "patterns": lambda: self._learn_patterns_with_debate(sentences, components),
            "enrichment": lambda: self._enrich_multiword_partials(sentences, components),
        })

        self.learned_patterns = t1_5["patterns"]

        print(f"  Subprocess terms: {len(self.learned_patterns.subprocess_terms)}")

        # Partial usage classification (needs enriched doc_knowledge)
        self._activity_partials = self._classify_partial_usage(sentences)
        print(f"  Activity-type partials (no syn-safe): {sorted(self._activity_partials)}")

        self._save_phase(text_path, "tier1_5", {
            "learned_patterns": self.learned_patterns,
            "doc_knowledge": self.doc_knowledge,
            "activity_partials": self._activity_partials,
        })

        # ═══ TIER 2: Link Recovery (entity pipeline ∥ coreference) ═══
        print("\n[Tier 2] Link Recovery (parallel)")
        t2 = self._run_parallel({
            "entity": lambda: self._run_entity_pipeline(
                sentences, components, name_to_id, sent_map, seed_links),
            "coref": lambda: self._run_coreference(
                sentences, components, name_to_id, sent_map),
        })

        validated = t2["entity"]
        coref_links = t2["coref"]
        print(f"  Entity pipeline: {len(validated)} validated")
        print(f"  Coreference: {len(coref_links)} links")

        # Partial injection (needs validated + coref + seed sets)
        partial_links = self._inject_partial_references(
            sentences, components, name_to_id, seed_set,
            {(c.sentence_number, c.component_id) for c in validated},
            {(l.sentence_number, l.component_id) for l in coref_links},
        )
        if partial_links:
            print(f"  Partial injection: {len(partial_links)} links")

        self._save_phase(text_path, "tier2", {
            "validated": validated,
            "coref_links": coref_links,
            "partial_links": partial_links,
        })

        # ═══ TIER 3: Merge + Filter + Judge (sequential) ═══
        print("\n[Tier 3] Merge + Boundary Filter + Judicial Review")

        # Priority-based deduplication
        entity_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
            for c in validated
        ]
        all_links = seed_links + entity_links + coref_links + partial_links
        link_map = {}
        for lk in all_links:
            key = (lk.sentence_number, lk.component_id)
            if key not in link_map:
                link_map[key] = lk
            else:
                old_p = self.SOURCE_PRIORITY.get(link_map[key].source, 0)
                new_p = self.SOURCE_PRIORITY.get(lk.source, 0)
                if new_p > old_p:
                    link_map[key] = lk
        preliminary = list(link_map.values())
        print(f"  After dedup: {len(preliminary)} (from {len(all_links)} raw)")

        # Parent-overlap guard
        if self.model_knowledge and self.model_knowledge.impl_to_abstract:
            child_to_parent = self.model_knowledge.impl_to_abstract
            sent_comps = defaultdict(set)
            for lk in preliminary:
                sent_comps[lk.sentence_number].add(lk.component_name)
            before_po = len(preliminary)
            filtered_po = []
            for lk in preliminary:
                parent = child_to_parent.get(lk.component_name)
                if parent and parent in sent_comps[lk.sentence_number]:
                    print(f"    Parent-overlap drop: S{lk.sentence_number} -> {lk.component_name} "
                          f"(parent: {parent})")
                else:
                    filtered_po.append(lk)
            if len(filtered_po) < before_po:
                print(f"  Parent-overlap guard: dropped {before_po - len(filtered_po)}")
            preliminary = filtered_po

        # Boundary filter
        preliminary, boundary_rejected = self._apply_boundary_filters(
            preliminary, sent_map, seed_set
        )
        if boundary_rejected:
            print(f"  Boundary filter: rejected {len(boundary_rejected)}")
            self._log("boundary", {},
                      {"rejected": len(boundary_rejected),
                       "details": [(lk.component_name, reason) for lk, reason in boundary_rejected]})

        self._save_phase(text_path, "pre_judge", {
            "preliminary": preliminary,
            "seed_set": seed_set,
        })

        # Judge review
        reviewed = self._judge_review(preliminary, sentences, components, sent_map, seed_set)
        rejected = [l for l in preliminary if (l.sentence_number, l.component_id)
                    not in {(r.sentence_number, r.component_id) for r in reviewed}]
        print(f"  Judge: approved {len(reviewed)}, rejected {len(rejected)}")
        self._log("judge", {"input": len(preliminary)},
                  {"approved": len(reviewed), "rejected": len(rejected)}, reviewed)

        final = reviewed

        # Save log + final checkpoint
        self._log("summary", {"total_time_s": round(time.time() - t0, 1)},
                  {"final": len(final)}, final)
        self._save_log(text_path)

        self._save_phase(text_path, "final", {
            "final": final,
            "reviewed": reviewed,
            "rejected": rejected,
        })

        print(f"\nFinal: {len(final)} links ({time.time() - t0:.0f}s)")
        return final

    # ═══════════════════════════════════════════════════════════════════════
    # Tier 1: Knowledge Acquisition
    # ═══════════════════════════════════════════════════════════════════════

    def _analyze_model(self, components):
        """Analyze model structure: parent-child, shared vocab, classify names."""
        names = [c.name for c in components]
        knowledge = ModelKnowledge()

        for name in names:
            for other in names:
                if other != name and len(other) >= 3 and other in name:
                    idx = name.find(other)
                    prefix, suffix = name[:idx], name[idx + len(other):]
                    if prefix and len(prefix) >= 2:
                        knowledge.impl_indicators.append(prefix)
                        knowledge.impl_to_abstract[name] = other
                    if suffix and len(suffix) >= 2:
                        knowledge.impl_indicators.append(suffix)
                        knowledge.impl_to_abstract[name] = other

        knowledge.impl_indicators = list(set(knowledge.impl_indicators))

        self._classify_components(names, knowledge)

        return knowledge

    @staticmethod
    def _is_structurally_unambiguous(name):
        """CamelCase, multi-word, or all-caps → always architectural."""
        if ' ' in name or '-' in name:
            return True
        if re.search(r'[a-z][A-Z]', name):
            return True
        if name.isupper():
            return True
        return False

    def _classify_components(self, names, knowledge):
        """Classify components using few-shot prompt + structural code guard."""
        prompt = f"""Classify these software architecture component names.

NAMES: {', '.join(names)}

{self._FEW_SHOT}

NOW CLASSIFY THE NAMES ABOVE.

Return JSON:
{{
  "architectural": ["names that identify specific components"],
  "ambiguous": ["names that could easily be used as ordinary words in documentation"]
}}

RULES:
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
         Connector/Controller/Adapter describe WHAT KIND (generic category) → AMBIGUOUS.

JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=100))
        if data:
            valid = set(names)
            raw_ambiguous = set(data.get("ambiguous", [])) & valid
            knowledge.ambiguous_names = {
                n for n in raw_ambiguous
                if len(n.split()) == 1 and not self._is_structurally_unambiguous(n)
            }

    def _compute_complexity(self, sentences, components):
        """Compute document complexity flag.

        A document is complex when explicit mention coverage is below 50%
        and the sentence-to-component ratio exceeds 4.
        """
        comp_names = [c.name for c in components]
        mention_count = sum(1 for sent in sentences
                           if any(cn.lower() in sent.text.lower() for cn in comp_names))
        uncovered_ratio = 1.0 - (mention_count / max(1, len(sentences)))
        spc = len(sentences) / max(1, len(components))
        return uncovered_ratio > 0.5 and spc > 4

    def _compute_generic_sets(self, components):
        """Derive generic word sets from model analysis results."""
        ambig = self.model_knowledge.ambiguous_names if self.model_knowledge else set()

        self.GENERIC_COMPONENT_WORDS = set()
        for name in ambig:
            if ' ' not in name and not name.isupper():
                self.GENERIC_COMPONENT_WORDS.add(name.lower())

        self.GENERIC_PARTIALS = set()
        for comp in components:
            parts = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', comp.name)
            for part in parts:
                p_lower = part.lower()
                if part.isupper():
                    continue
                if len(p_lower) >= 3 and (p_lower in ambig or any(
                    p_lower == a.lower() for a in ambig
                )):
                    self.GENERIC_PARTIALS.add(p_lower)
        for name in ambig:
            if ' ' not in name and not name.isupper():
                self.GENERIC_PARTIALS.add(name.lower())

    def _learn_document_knowledge_enriched(self, sentences, components):
        """Extract abbreviations, synonyms, partial references via few-shot calibrated judge."""
        comp_names = [c.name for c in components]
        doc_lines = [s.text for s in sentences[:150]]

        prompt1 = f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

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
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": {{"short_form": "FullComponent"}},
  "synonyms": {{"specific_alternative_name": "FullComponent"}},
  "partial_references": {{"partial_name": "FullComponent"}}
}}
JSON only:"""

        data1 = self.llm.extract_json(self.llm.query(prompt1, timeout=150))

        all_mappings = {}
        if data1:
            for short, full in data1.get("abbreviations", {}).items():
                if full in comp_names:
                    all_mappings[short] = ("abbrev", full)
            for syn, full in data1.get("synonyms", {}).items():
                if full in comp_names:
                    all_mappings[syn] = ("synonym", full)
            for partial, full in data1.get("partial_references", {}).items():
                if full in comp_names:
                    all_mappings[partial] = ("partial", full)

        if all_mappings:
            mapping_list = [f"'{k}' -> {v[1]} ({v[0]})" for k, v in list(all_mappings.items())[:25]]

            prompt2 = f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

EXAMPLES — study these to calibrate your judgment:

Example 1 — APPROVE (abbreviation from component name):
  'AST' -> AbstractSyntaxTree (abbrev)
  Verdict: APPROVE. "AST" is the initials of "AbstractSyntaxTree". Abbreviations
  formed from the component name's words are always valid.

Example 2 — APPROVE (trailing word of multi-word name):
  'Dispatcher' -> EventDispatcher (partial)
  Verdict: APPROVE. "Dispatcher" is the last word of "EventDispatcher".
  If no other component ends in "Dispatcher", this partial is unambiguous.

Example 3 — APPROVE (CamelCase identifier):
  'RenderEngine' -> GameRenderEngine (synonym)
  Verdict: APPROVE. CamelCase is a constructed identifier — always a proper name.

Example 4 — APPROVE (trailing word of multi-word name):
  'Table' -> SymbolTable (partial)
  Verdict: APPROVE. "Table" is the trailing word of "SymbolTable" and
  likely refers to this specific component when no other component uses "Table".

Example 5 — REJECT (ordinary English verb/noun):
  'process' -> OrderProcessor (partial)
  Verdict: REJECT. "process" is an ordinary English verb/noun used generically
  in many contexts ("process requests", "the process").

Example 6 — REJECT (refers to whole system):
  'system' -> PaymentSystem (partial)
  Verdict: REJECT. "system" is too generic — it could refer to the overall system.

DECISION RULES (apply in order):

1. AUTO-APPROVE these — they are always valid mappings:
   - Abbreviations formed from the component name's initials or words
   - Trailing words of multi-word component names (if no other component shares that word)
   - CamelCase identifiers
   - Multi-word phrases that contain the component name

2. APPROVE if the term plausibly refers to exactly one component and is NOT
   a generic word like "system", "process", "service", "component", "module".

3. REJECT only if the term is clearly generic and could refer to anything,
   or clearly refers to a different component or the system as a whole.

IMPORTANT: When in doubt, APPROVE. False approvals are filtered by later
pipeline stages; false rejections cause permanent recall loss.

Return JSON:
{{
  "approved": ["term1", "term2"],
  "generic_rejected": ["generic_term1"]
}}
JSON only:"""

            data2 = self.llm.extract_json(self.llm.query(prompt2, timeout=120))
            approved = set(data2.get("approved", [])) if data2 else set(all_mappings.keys())
            generic_terms = set(data2.get("generic_rejected", [])) if data2 else set()
        else:
            approved = set()
            generic_terms = set()

        # CamelCase rescue: constructed identifiers are never generic
        for term in list(generic_terms):
            if re.search(r'[a-z][A-Z]', term) and term in all_mappings:
                generic_terms.discard(term)
                approved.add(term)
                print(f"    CamelCase override (rescued): {term}")

        knowledge = DocumentKnowledge()

        for term, (typ, comp) in all_mappings.items():
            if term in approved:
                if typ == "abbrev":
                    knowledge.abbreviations[term] = comp
                    print(f"    Abbrev: {term} -> {comp}")
                elif typ == "synonym":
                    knowledge.synonyms[term] = comp
                    print(f"    Syn: {term} -> {comp}")
                else:
                    knowledge.partial_references[term] = comp
                    print(f"    Partial: {term} -> {comp}")

        # Deterministic CamelCase-split synonym injection
        for comp in [c.name for c in components]:
            split = re.sub(r'([a-z])([A-Z])', r'\1 \2', comp)
            split = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', split)
            if split != comp and split not in knowledge.synonyms:
                knowledge.synonyms[split] = comp
                print(f"    CamelCase syn: {split} -> {comp}")

        return knowledge

    def _run_seed(self, text_path, model_path):
        """Run ILinker2 seed extractor (independent of knowledge phases)."""
        raw = self._ilinker2.link(text_path, model_path)
        return [SadSamLink(l.sentence_number, l.component_id, l.component_name,
                           l.confidence, "seed") for l in raw]

    # ═══════════════════════════════════════════════════════════════════════
    # Tier 1.5: Knowledge Enrichment
    # ═══════════════════════════════════════════════════════════════════════

    def _learn_patterns_with_debate(self, sentences, components):
        """Learn subprocess terms, action/effect indicators via debate."""
        comp_names = self._get_comp_names(components)
        sample = [f"S{s.number}: {s.text}" for s in sentences[:70]]

        prompt1 = f"""Find terms that refer to INTERNAL PARTS of components (subprocesses).

COMPONENTS: {', '.join(comp_names)}

DOCUMENT:
{chr(10).join(sample)}

Return JSON:
{{
  "subprocess_terms": ["term1", "term2"],
  "reasoning": {{"term": "why"}}
}}
JSON only:"""

        data1 = self.llm.extract_json(self.llm.query(prompt1, timeout=120))
        proposed = data1.get("subprocess_terms", []) if data1 else []
        reasonings = data1.get("reasoning", {}) if data1 else {}

        if proposed:
            prompt2 = f"""DEBATE: Validate these subprocess terms.

COMPONENTS: {', '.join(comp_names)}

PROPOSED:
{chr(10).join([f"- {t}: {reasonings.get(t, '')}" for t in proposed[:15]])}

SAMPLE:
{chr(10).join(sample[:30])}

Return JSON:
{{
  "validated": ["terms that ARE subprocesses"],
  "rejected": ["terms that might be valid component references"]
}}
JSON only:"""

            data2 = self.llm.extract_json(self.llm.query(prompt2, timeout=120))
            validated_terms = set(data2.get("validated", [])) if data2 else set(proposed)
        else:
            validated_terms = set()

        prompt3 = f"""Find linguistic patterns.

COMPONENTS: {', '.join(comp_names)}

DOCUMENT:
{chr(10).join(sample[:40])}

Return JSON:
{{
  "action_indicators": ["verbs when component DOES something"],
  "effect_indicators": ["verbs for RESULTS"]
}}
JSON only:"""

        data3 = self.llm.extract_json(self.llm.query(prompt3, timeout=100))

        patterns = LearnedPatterns()
        patterns.subprocess_terms = validated_terms
        if data3:
            patterns.action_indicators = data3.get("action_indicators", [])
            patterns.effect_indicators = data3.get("effect_indicators", [])

        for t in list(validated_terms)[:8]:
            print(f"    Subprocess: '{t}'")

        return patterns

    def _enrich_multiword_partials(self, sentences, components):
        """Auto-discover multi-word partial references from usage patterns."""
        if not self.doc_knowledge:
            return

        added = []
        for comp in components:
            parts = comp.name.split()
            if len(parts) < 2:
                continue
            last_word = parts[-1]
            if len(last_word) < 4:
                continue
            last_lower = last_word.lower()

            other_match = any(
                c.name != comp.name and c.name.lower().endswith(last_lower)
                for c in components
            )
            if other_match:
                continue
            if last_lower in {s.lower() for s in self.doc_knowledge.synonyms}:
                continue
            if last_lower in {p.lower() for p in self.doc_knowledge.partial_references}:
                continue

            is_generic_word = last_lower in self.GENERIC_PARTIALS
            full_lower = comp.name.lower()
            mention_count = 0
            for sent in sentences:
                sl = sent.text.lower()
                if last_lower in sl and full_lower not in sl:
                    if is_generic_word:
                        cap_word = last_word[0].upper() + last_word[1:]
                        if re.search(rf'\b{re.escape(cap_word)}\b', sent.text):
                            mention_count += 1
                    else:
                        if re.search(rf'\b{re.escape(last_word)}\b', sent.text, re.IGNORECASE):
                            mention_count += 1

            if mention_count >= 3:
                self.doc_knowledge.partial_references[last_word] = comp.name
                added.append(f"{last_word} -> {comp.name} ({mention_count} mentions)")

        if added:
            print(f"  [Enrichment] Multi-word partials:")
            for a in added:
                print(f"    Auto-partial: {a}")

    def _classify_partial_usage(self, sentences):
        """Classify single-word generic partials as NAME or ORDINARY.

        For each partial, shows the LLM all sentences where the partial appears
        (without the full component name) and asks whether it's used as a
        standalone entity reference or as an ordinary English word.

        Returns set of ORDINARY-classified partial names (these lose syn-safe).
        """
        if not self.doc_knowledge or not self.doc_knowledge.partial_references:
            return set()

        # Find single-word generic partials (CamelCase = always entity)
        generic_partials = {}
        for partial, comp_name in self.doc_knowledge.partial_references.items():
            if ' ' in partial:
                continue
            if re.search(r'[a-z][A-Z]', partial):
                continue
            generic_partials[partial] = comp_name

        if not generic_partials:
            return set()

        print(f"  [Partial Classification] {len(generic_partials)} generic partials")

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

            sent_lines = []
            for s in partial_sentences[:15]:
                sent_lines.append(f"  S{s.number}: {s.text}")
            sent_block = "\n".join(sent_lines)

            if full_name_sentences:
                fn_lines = []
                for s in full_name_sentences[:5]:
                    fn_lines.append(f"  S{s.number}: {s.text}")
                fn_block = "\n".join(fn_lines)
                calibration = f"""For reference, these sentences use the FULL component name "{comp_name}":
{fn_block}
"""
            else:
                calibration = ""

            prompt = f"""WORD USAGE CLASSIFICATION

In this document, the word "{partial}" could be a short name for an architecture
component called "{comp_name}".

{calibration}Below are ALL sentences where "{partial}" appears WITHOUT the full name "{comp_name}".
Analyze how the word "{partial}" is used across these sentences:

{sent_block}

QUESTION: Is "{partial}" used as a standalone entity reference in ANY of these sentences?

Classify as NAME if the word appears as a standalone noun phrase referring to a specific
system entity in at least SOME sentences — even if other sentences use it generically.
Examples of entity reference: "the {partial.lower()} connects to...", "sends data to the
{partial.lower()}", "the {partial.lower()} handles...", "on the {partial.lower()}"

Classify as ORDINARY only if EVERY occurrence uses the word as part of a compound phrase,
modifier, or generic descriptor — never as a standalone entity.
Examples of purely ordinary: "{partial.lower()} process", "automated {partial.lower()}",
"{partial.lower()} strategy", "{partial.lower()}-based"

The threshold is: if even ONE sentence uses "{partial}" as a standalone entity reference,
classify as NAME. Only classify as ORDINARY when you see ZERO standalone entity uses.

Return JSON: {{"classification": "name" or "ordinary", "reason": "brief explanation"}}
JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=60))
            if data:
                classification = data.get("classification", "name")
                reason = data.get("reason", "")
                if classification == "ordinary":
                    activity_partials.add(partial)
                    print(f"    \"{partial}\" -> {comp_name}: ORDINARY. {reason}")
                else:
                    print(f"    \"{partial}\" -> {comp_name}: NAME (keep syn-safe). {reason}")
            else:
                print(f"    \"{partial}\" -> {comp_name}: PARSE FAILURE (keep syn-safe)")

        return activity_partials

    # ═══════════════════════════════════════════════════════════════════════
    # Tier 2: Link Recovery
    # ═══════════════════════════════════════════════════════════════════════

    def _run_entity_pipeline(self, sentences, components, name_to_id, sent_map, seed_links):
        """Entity extraction → abbreviation guard → targeted recovery → validation.

        This is a sequential chain within Tier 2, running concurrently with coreference.
        """
        # Extract
        candidates = self._extract_entities_enriched(sentences, components, name_to_id, sent_map)
        print(f"    Entity extraction: {len(candidates)} candidates")

        # Abbreviation guard
        before_guard = len(candidates)
        candidates = self._apply_abbreviation_guard_to_candidates(candidates, sent_map)
        if len(candidates) < before_guard:
            print(f"    After abbrev guard: {len(candidates)} (-{before_guard - len(candidates)})")

        # Targeted recovery for unlinked components
        entity_comps = {c.component_name for c in candidates}
        seed_comps = {l.component_name for l in seed_links}
        covered_comps = entity_comps | seed_comps
        unlinked = [c for c in components if c.name not in covered_comps]

        if unlinked:
            print(f"    Targeted recovery: {len(unlinked)} unlinked components")
            extra = self._targeted_recovery(unlinked, sentences, name_to_id, sent_map,
                                              components=components, seed_links=seed_links,
                                              entity_candidates=candidates)
            if extra:
                print(f"    Targeted found: {len(extra)} additional")
                candidates.extend(extra)

        # Validation
        validated = self._validate_intersect(candidates, components, sent_map)
        print(f"    Validation: {len(validated)} / {len(candidates)}")
        return validated

    def _run_coreference(self, sentences, components, name_to_id, sent_map):
        """Coreference resolution — mode selected by document complexity."""
        if self._is_complex:
            print(f"    Coreference: debate mode ({len(sentences)} sents)")
            return self._coref_debate(sentences, components, name_to_id, sent_map)
        else:
            print(f"    Coreference: discourse mode ({len(sentences)} sents)")
            return self._coref_discourse(sentences, components, name_to_id, sent_map)

    def _extract_entities_enriched(self, sentences, components, name_to_id, sent_map):
        """Batched entity extraction with retry on empty response."""
        comp_names = self._get_comp_names(components)
        comp_lower = {n.lower() for n in comp_names}

        mappings = []
        if self.doc_knowledge:
            mappings.extend([f"{a}={c}" for a, c in self.doc_knowledge.abbreviations.items()])
            mappings.extend([f"{s}={c}" for s, c in self.doc_knowledge.synonyms.items()])
            mappings.extend([f"{p}={c}" for p, c in self.doc_knowledge.partial_references.items()])

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

            for attempt in range(2):
                data = self.llm.extract_json(self.llm.query(prompt, timeout=240))
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

                matched_lower = matched.lower() if matched else ""
                is_exact = matched_lower in comp_lower or cname.lower() in matched_lower
                is_generic_here = self._is_generic_mention(cname, sent.text)
                needs_val = not is_exact or ref.get("match_type") != "exact" or is_generic_here

                key = (snum, name_to_id[cname])
                if key not in all_candidates:
                    all_candidates[key] = CandidateLink(snum, sent.text, cname, name_to_id[cname],
                                               matched, 0.85, "entity",
                                               ref.get("match_type", "exact"), needs_val)

        return list(all_candidates.values())

    def _apply_abbreviation_guard_to_candidates(self, candidates, sent_map):
        """Filter candidates where abbreviation match is contextually invalid."""
        if not self.doc_knowledge:
            return candidates
        abbrev_to_comp = {}
        comp_to_abbrevs = {}
        for abbr, comp in self.doc_knowledge.abbreviations.items():
            abbrev_to_comp[abbr.lower()] = comp
            comp_to_abbrevs.setdefault(comp, []).append(abbr)

        filtered = []
        for c in candidates:
            matched_lower = c.matched_text.lower() if c.matched_text else ""
            comp = c.component_name
            sent = sent_map.get(c.sentence_number)
            if matched_lower in abbrev_to_comp and abbrev_to_comp[matched_lower] == comp:
                if sent and not self._abbreviation_match_is_valid(c.matched_text, comp, sent.text):
                    print(f"    Abbrev guard: rejected S{c.sentence_number} {c.matched_text} -> {comp}")
                    continue
            if sent and comp in comp_to_abbrevs and ' ' in comp:
                full_in_text = re.search(rf'\b{re.escape(comp)}\b', sent.text, re.IGNORECASE)
                if not full_in_text:
                    rejected = False
                    for abbr in comp_to_abbrevs[comp]:
                        if re.search(rf'\b{re.escape(abbr)}\b', sent.text, re.IGNORECASE):
                            if not self._abbreviation_match_is_valid(abbr, comp, sent.text):
                                print(f"    Abbrev guard (inferred): rejected S{c.sentence_number} {abbr} -> {comp}")
                                rejected = True
                                break
                    if rejected:
                        continue
            filtered.append(c)
        return filtered

    def _targeted_recovery(self, unlinked_components, sentences, name_to_id, sent_map,
                              components=None, seed_links=None, entity_candidates=None):
        """Single-component LLM prompts for unlinked components."""
        if not unlinked_components:
            return []

        parent_map = {}
        existing_sent_comp = defaultdict(set)
        if components and seed_links:
            all_comp_names = {c.name for c in components}
            for comp in unlinked_components:
                parents = set()
                for other_name in all_comp_names:
                    if other_name != comp.name and len(other_name) >= 3 and other_name in comp.name:
                        parents.add(other_name)
                if parents:
                    parent_map[comp.name] = parents
            for lk in seed_links:
                existing_sent_comp[lk.sentence_number].add(lk.component_name)
            if entity_candidates:
                for c in entity_candidates:
                    existing_sent_comp[c.sentence_number].add(c.component_name)

        all_extra = []
        doc_text = "\n".join([f"S{s.number}: {s.text}" for s in sentences])
        for comp in unlinked_components:

            aliases = []
            if self.doc_knowledge:
                for a, c in self.doc_knowledge.abbreviations.items():
                    if c == comp.name: aliases.append(a)
                for s, c in self.doc_knowledge.synonyms.items():
                    if c == comp.name: aliases.append(s)
                for p, c in self.doc_knowledge.partial_references.items():
                    if c == comp.name: aliases.append(p)

            alias_str = f"\nKNOWN ALIASES: {', '.join(aliases)}" if aliases else ""

            prompt = f"""Find ALL sentences that discuss the software component "{comp.name}".
{alias_str}

Look for:
- Direct mentions of "{comp.name}" or any alias
- Descriptions of what {comp.name} does (functional descriptions)
- References to {comp.name}'s role in the architecture

Exclude:
- Names that appear only inside a dotted package path (e.g., com.example.name does NOT count as a reference to "name")

DOCUMENT:
{doc_text}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "matched_text": "text found", "reason": "why this refers to {comp.name}"}}]}}

Be thorough — find ALL sentences that discuss this component.
JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
            if not data:
                continue

            for ref in data.get("references", []):
                snum = ref.get("sentence")
                if not snum:
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
                cid = name_to_id.get(comp.name)
                if not cid:
                    continue

                if comp.name in parent_map:
                    parents_here = parent_map[comp.name] & existing_sent_comp.get(snum, set())
                    if parents_here:
                        continue

                matched = ref.get("matched_text", comp.name)
                all_extra.append(CandidateLink(
                    snum, sent.text, comp.name, cid,
                    matched, 0.85, "entity", "targeted", True
                ))

        return all_extra

    def _validate_intersect(self, candidates, components, sent_map):
        """Code-first auto-approval + 2-pass intersection + evidence post-filter."""
        if not candidates:
            return []

        comp_names = self._get_comp_names(components)
        needs = [c for c in candidates if c.needs_validation]
        direct = [c for c in candidates if not c.needs_validation]

        if not needs:
            return candidates

        # Pre-check: reject generic mentions
        remaining = []
        for c in needs:
            sent = sent_map.get(c.sentence_number)
            if sent and self._is_generic_mention(c.component_name, sent.text):
                pass
            else:
                remaining.append(c)
        needs = remaining

        # Build alias lookup
        alias_map = {}
        for c in components:
            aliases = {c.name}
            if self.doc_knowledge:
                for a, cn in self.doc_knowledge.abbreviations.items():
                    if cn == c.name:
                        aliases.add(a)
                for s, cn in self.doc_knowledge.synonyms.items():
                    if cn == c.name:
                        aliases.add(s)
                for p, cn in self.doc_knowledge.partial_references.items():
                    if cn == c.name:
                        aliases.add(p)
            alias_map[c.name] = aliases

        # Step 1: Word-boundary code-first
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
                    if self._word_boundary_match(a, sent.text):
                        matched = True
                        break
            if matched:
                c.confidence = 1.0
                c.source = "validated"
                auto_approved.append(c)
            else:
                llm_needed.append(c)

        # Classify generic-risk components
        generic_risk = set()
        if self.model_knowledge and self.model_knowledge.ambiguous_names:
            generic_risk |= self.model_knowledge.ambiguous_names
        for c in components:
            if c.name.lower() in self.GENERIC_COMPONENT_WORDS:
                generic_risk.add(c.name)

        # Step 2: 2-pass intersect for LLM-needed
        ctx = []
        if self.learned_patterns:
            if self.learned_patterns.action_indicators:
                ctx.append(f"ACTION: {', '.join(self.learned_patterns.action_indicators[:4])}")
            if self.learned_patterns.effect_indicators:
                ctx.append(f"EFFECT (reject): {', '.join(self.learned_patterns.effect_indicators[:3])}")
            if self.learned_patterns.subprocess_terms:
                ctx.append(f"Subprocess (reject): {', '.join(list(self.learned_patterns.subprocess_terms)[:5])}")

        twopass_approved = []
        generic_to_verify = []
        for batch_start in range(0, len(llm_needed), 25):
            batch = llm_needed[batch_start:batch_start + 25]
            cases = []
            for i, c in enumerate(batch):
                prev = sent_map.get(c.sentence_number - 1)
                p = f"[prev: {prev.text[:35]}...] " if prev else ""
                cases.append(f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n  {p}"{c.sentence_text}"')

            r1 = self._qual_validation_pass(comp_names, ctx, cases,
                "Focus on ACTOR role: is the component performing an action or being described?")
            r2 = self._qual_validation_pass(comp_names, ctx, cases,
                "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component, not a generic concept?")

            for i, c in enumerate(batch):
                if r1.get(i, False) and r2.get(i, False):
                    if c.component_name in generic_risk:
                        generic_to_verify.append(c)
                    else:
                        c.confidence = 1.0
                        c.source = "validated"
                        twopass_approved.append(c)

        # Step 3: Evidence post-filter for generic-risk
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

                data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
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

        return direct + auto_approved + twopass_approved + generic_validated

    def _qual_validation_pass(self, comp_names, ctx, cases, focus):
        """Single validation pass for 2-pass intersection."""
        prompt = f"""Validate component references in a software architecture document. {focus}

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

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

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
        results = {}
        if data:
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    results[idx] = v.get("approve", False)
        return results

    def _coref_discourse(self, sentences, components, name_to_id, sent_map):
        """Discourse-mode coreference with required antecedent citation."""
        comp_names = self._get_comp_names(components)
        all_coref = []
        pronoun_sents = [s for s in sentences if self.PRONOUN_PATTERN.search(s.text)]

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

            data = self.llm.extract_json(self.llm.query(prompt, timeout=150))
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
                    if not (self._has_standalone_mention(comp, ant_sent.text) or
                            self._has_alias_mention(comp, ant_sent.text)):
                        continue
                    if abs(snum - ant_snum) > 3:
                        continue

                sent = sent_map.get(snum)
                if sent and self.learned_patterns and self.learned_patterns.is_subprocess(sent.text):
                    continue
                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, 1.0, "coreference"))

        return all_coref

    def _coref_debate(self, sentences, components, name_to_id, sent_map):
        """Debate-mode coreference with required antecedent citation."""
        comp_names = self._get_comp_names(components)
        all_coref = []

        ctx = []
        if self.learned_patterns and self.learned_patterns.subprocess_terms:
            ctx.append(f"Subprocesses (don't link): {', '.join(list(self.learned_patterns.subprocess_terms)[:5])}")

        for batch_start in range(0, len(sentences), 20):
            batch = sentences[batch_start:min(batch_start + 20, len(sentences))]
            ctx_start = max(0, batch_start - self.CONTEXT_WINDOW)
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

RULES (all must hold):
1. You MUST cite the antecedent_sentence where the component was EXPLICITLY NAMED
2. The component name (or known alias) MUST appear verbatim in the antecedent sentence
3. The antecedent MUST be within the previous 3 sentences
4. Do NOT resolve pronouns in sentences about subprocesses or implementation details
5. If the pronoun could refer to multiple components, do NOT resolve it

Return JSON:
{{"resolutions": [{{"sentence": N_INTEGER, "pronoun": "it", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

Only include resolutions you are CERTAIN about. JSON only:"""

            data1 = self.llm.extract_json(self.llm.query(prompt1, timeout=100))
            if not data1:
                continue
            proposed = data1.get("resolutions", [])
            if not proposed:
                continue

            for res in proposed:
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
                    if not (self._has_standalone_mention(comp, ant_sent.text) or
                            self._has_alias_mention(comp, ant_sent.text)):
                        continue
                    if abs(snum - ant_snum) > 3:
                        continue

                sent = sent_map.get(snum)
                if sent and self.learned_patterns and self.learned_patterns.is_subprocess(sent.text):
                    continue
                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, 1.0, "coreference"))

        return all_coref

    def _inject_partial_references(self, sentences, components, name_to_id,
                                    seed_set, validated_set, coref_set):
        """Deterministic partial-reference injection for word-boundary matches."""
        if not self.doc_knowledge or not self.doc_knowledge.partial_references:
            return []

        existing = seed_set | validated_set | coref_set
        injected = []

        for partial, comp_name in self.doc_knowledge.partial_references.items():
            if comp_name not in name_to_id:
                continue
            comp_id = name_to_id[comp_name]
            for sent in sentences:
                key = (sent.number, comp_id)
                if key in existing:
                    continue
                if self._has_clean_mention(partial, sent.text):
                    injected.append(SadSamLink(
                        sent.number, comp_id, comp_name, 0.8, "partial_inject"
                    ))
                    existing.add(key)

        return injected

    # ═══════════════════════════════════════════════════════════════════════
    # Tier 3: Merge + Filter + Judge
    # ═══════════════════════════════════════════════════════════════════════

    def _apply_boundary_filters(self, links, sent_map, seed_set):
        """LLM convention filter using 3-step reasoning guide.

        Seed links are immune (handled by judge).
        All other links (including partial_inject) are reviewed.
        """
        comp_names = self._get_comp_names(self._components)

        safe = []
        to_review = []
        for lk in links:
            is_seed = (lk.sentence_number, lk.component_id) in seed_set
            if is_seed:
                safe.append(lk)
            else:
                to_review.append(lk)

        if not to_review:
            return safe, []

        # Build alias context (exclude ORDINARY-type partials)
        alias_context = ""
        if self.doc_knowledge:
            alias_lines = []
            for partial, comp in self.doc_knowledge.partial_references.items():
                if partial not in self._activity_partials:
                    alias_lines.append(f'  "{partial}" is a confirmed short name for {comp}')
            for syn, comp in self.doc_knowledge.synonyms.items():
                alias_lines.append(f'  "{syn}" is a confirmed synonym for {comp}')
            for abbr, comp in self.doc_knowledge.abbreviations.items():
                alias_lines.append(f'  "{abbr}" is a confirmed abbreviation for {comp}')
            if alias_lines:
                alias_context = (
                    "CONFIRMED ALIASES (from document analysis):\n"
                    + "\n".join(alias_lines)
                    + "\n\nIMPORTANT: When a confirmed alias appears in a sentence, it IS a reference "
                    "to that component — even inside compound phrases. For example, if \"Svc\" is a "
                    "confirmed short name for BackendSvc, then \"Svc handler\" in a sentence IS about BackendSvc.\n"
                )

        items = []
        for i, lk in enumerate(to_review):
            sent = sent_map.get(lk.sentence_number)
            text = sent.text if sent else "(no text)"
            items.append(
                f'{i+1}. S{lk.sentence_number}: "{text}"\n'
                f'   Component: "{lk.component_name}"'
            )

        batch_size = 25
        all_verdicts = {}

        for batch_start in range(0, len(items), batch_size):
            batch_items = items[batch_start:batch_start + batch_size]

            prompt = f"""Validate trace links between architecture documentation and components.

ARCHITECTURE COMPONENTS: {', '.join(comp_names)}

{alias_context}{CONVENTION_GUIDE}

---

For each sentence-component pair, apply the 3-step reasoning guide.
Decide LINK (keep the trace link) or NO_LINK (reject it).

{chr(10).join(batch_items)}

Return JSON array:
[{{"id": N, "step": "1|2a|2b|3", "verdict": "LINK" or "NO_LINK", "reason": "brief"}}]
JSON only:"""

            raw = self.llm.query(prompt, timeout=180)
            data = self._extract_json_array(raw.text if hasattr(raw, 'text') else str(raw))
            if data:
                for item in data:
                    vid = item.get("id")
                    verdict = item.get("verdict", "LINK").upper().strip()
                    step = item.get("step", "3")
                    reason = item.get("reason", "")
                    if vid is not None:
                        all_verdicts[vid] = (verdict, step, reason)

        kept = list(safe)
        rejected = []
        for i, lk in enumerate(to_review):
            verdict, step, reason = all_verdicts.get(i + 1, ("LINK", "3", "default"))
            if "NO" in verdict:
                rejected.append((lk, f"convention_step{step}"))
                print(f"    Convention filter [step {step}]: S{lk.sentence_number} -> "
                      f"{lk.component_name} ({lk.source}) — {reason}")
            else:
                kept.append(lk)

        return kept, rejected

    def _judge_review(self, links, sentences, components, sent_map, seed_set):
        """Triage-based judge: immune/synonym-safe/mention-safe → advocate-prosecutor → standard."""
        if len(links) < 5:
            return links

        comp_names = self._get_comp_names(components)

        safe, syn_review, standard_links, seed_review = [], [], [], []
        for l in links:
            is_seed = (l.sentence_number, l.component_id) in seed_set
            sent = sent_map.get(l.sentence_number)
            if sent and self._has_alias_mention(l.component_name, sent.text):
                alias, alias_type = self._find_matching_alias(l.component_name, sent.text)
                if alias and alias in self._activity_partials:
                    syn_review.append(l)
                else:
                    safe.append(l)
                continue
            if is_seed:
                if self._is_ambiguous_name_component(l.component_name):
                    seed_review.append(l)
                else:
                    safe.append(l)
                continue
            if not sent:
                standard_links.append(l)
                continue
            if self._has_standalone_mention(l.component_name, sent.text):
                safe.append(l)
            else:
                standard_links.append(l)

        print(f"  Triage: {len(safe)} immune/syn-safe/mention-safe, {len(syn_review)} activity-partial, "
              f"{len(seed_review)} advocate-prosecutor, {len(standard_links)} standard-review")

        syn_approved = self._judge_syn_safe(syn_review, sent_map)
        seed_approved = self._deliberate_seed(seed_review, comp_names, sent_map)
        standard_approved = self._judge_standard(standard_links, comp_names, sent_map)

        return safe + syn_approved + seed_approved + standard_approved

    def _judge_syn_safe(self, syn_links, sent_map):
        """Calibrated batch judge for activity-partial links.

        Groups by (partial, component), finds full-name anchors, union voting (2 passes).
        Non-generic aliases auto-approved.
        """
        if not syn_links:
            return []

        print(f"  Judging {len(syn_links)} activity-partial links")

        partial_groups = defaultdict(list)
        auto_approve = []

        for l in syn_links:
            sent = sent_map.get(l.sentence_number)
            if not sent:
                auto_approve.append(l)
                continue

            alias, alias_type = self._find_matching_alias(l.component_name, sent.text)
            is_generic_partial = (alias_type == "partial" and alias
                                  and ' ' not in alias
                                  and not re.search(r'[a-z][A-Z]', alias))

            if is_generic_partial:
                partial_groups[(alias, l.component_name)].append(l)
            else:
                auto_approve.append(l)

        if not partial_groups:
            return auto_approve  # BUG FIX: V39 returned auto_approve + syn_links (duplicates)

        # Find full-name anchor sentences
        all_sents = list(sent_map.values())
        anchors = {}
        for (partial, comp), links in partial_groups.items():
            if comp in anchors:
                continue
            comp_anchors = []
            for s in all_sents:
                if re.search(rf'\b{re.escape(comp)}\b', s.text, re.IGNORECASE):
                    comp_anchors.append((s.number, s.text))
            anchors[comp] = comp_anchors[:5]

        approved = list(auto_approve)

        for (partial, comp), links in sorted(partial_groups.items()):
            comp_anchors = anchors.get(comp, [])

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

            sorted_links = sorted(links, key=lambda x: x.sentence_number)
            case_lines = []
            for i, l in enumerate(sorted_links):
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
"{partial.lower()} process", "automated {partial.lower()}", or "{partial.lower()} fallback" — where
the word describes a generic activity or type — that is ordinary English, not a
component reference.

Return JSON:
{{"classifications": [
  {{"case": 1, "refers_to_component": true/false, "reason": "brief"}},
  ...
]}}
JSON only:"""

            # Union voting: 2 passes, reject only if BOTH reject
            all_pass_results = []
            for _ in range(2):
                data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
                pass_results = {}
                if data:
                    for c in data.get("classifications", []):
                        idx = c.get("case", 0) - 1
                        refers = c.get("refers_to_component", True)
                        pass_results[idx] = refers
                all_pass_results.append(pass_results)

            for i, l in enumerate(sorted_links):
                p1_refers = all_pass_results[0].get(i, True)
                p2_refers = all_pass_results[1].get(i, True)
                union_approved = p1_refers or p2_refers

                if union_approved:
                    approved.append(l)
                    if not p1_refers or not p2_refers:
                        print(f"    Context union-save: S{l.sentence_number} -> {l.component_name}")
                else:
                    print(f"    Context reject: S{l.sentence_number} -> {l.component_name}")

        return approved

    def _judge_standard(self, standard_links, comp_names, sent_map):
        """Standard 4-rule judge for non-alias links, with union voting."""
        if not standard_links:
            return []

        cases = self._build_judge_cases(standard_links, sent_map)
        prompt = self._build_judge_prompt(comp_names, cases)
        n = min(30, len(standard_links))

        data1 = self.llm.extract_json(self.llm.query(prompt, timeout=180))
        data2 = self.llm.extract_json(self.llm.query(prompt, timeout=180))

        rej1 = self._parse_rejections(data1, n)
        rej2 = self._parse_rejections(data2, n)
        rejected = rej1 & rej2

        result = []
        for i in range(n):
            if i not in rejected:
                result.append(standard_links[i])
            else:
                print(f"    4-rule reject: S{standard_links[i].sentence_number} -> {standard_links[i].component_name}")
        if len(standard_links) > n:
            print(f"    WARNING: {len(standard_links) - n} nomatch links skipped judge (cap={n})")
        result.extend(standard_links[n:])
        return result

    def _build_judge_prompt(self, comp_names, cases):
        """Generalized 4-rule judge prompt."""
        return f"""JUDGE: Validate trace links between documentation and software architecture components.

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

COMPONENTS: {', '.join(comp_names)}

LINKS:
{chr(10).join(cases)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief explanation"}}]}}
JSON only:"""

    def _build_judge_cases(self, review, sent_map):
        """Format cases for judge prompt with adaptive context."""
        use_full_ctx = not self._is_complex

        cases = []
        for i, l in enumerate(review[:30]):
            sent = sent_map.get(l.sentence_number)
            ctx_lines = []
            if l.source == "coreference":
                p2 = sent_map.get(l.sentence_number - 2)
                if p2:
                    txt = p2.text if use_full_ctx else f"{p2.text[:45]}..."
                    ctx_lines.append(f"    PREV2: {txt}")
            p1 = sent_map.get(l.sentence_number - 1)
            if p1:
                txt = p1.text if use_full_ctx else f"{p1.text[:45]}..."
                ctx_lines.append(f"    PREV: {txt}")
            ctx_lines.append(f"    >>> S{l.sentence_number}: {sent.text if sent else '?'}")

            src_info = f"src:{l.source}"
            if sent:
                match = self._find_match_text(l.component_name, sent.text)
                if match and match.lower() != l.component_name.lower():
                    src_info += f', match:"{match}"'
                elif not match:
                    src_info += ', match:NONE(pronoun/context)'

            cases.append(f"Case {i+1}: S{l.sentence_number} -> {l.component_name} ({src_info})\n"
                         + chr(10).join(ctx_lines))
        return cases

    def _parse_rejections(self, data, n):
        """Extract rejection indices from judge JSON."""
        rejected = set()
        if data:
            for j in data.get("judgments", []):
                idx = j.get("case", 0) - 1
                if 0 <= idx < n and not j.get("approve", False):
                    rejected.add(idx)
        return rejected

    def _deliberate_seed(self, links, comp_names, sent_map):
        """Advocate-Prosecutor deliberation for ambiguous-name seed links."""
        if not links:
            return []

        print(f"  Deliberating {len(links)} ambiguous-name seed links")

        approved = []
        for l in links:
            sent = sent_map.get(l.sentence_number)
            if not sent:
                approved.append(l)
                continue

            verdicts = []
            for _ in range(2):
                verdict = self._single_advocate_prosecutor_pass(
                    l.sentence_number, l.component_name, sent.text, comp_names)
                verdicts.append(verdict)

            if verdicts[0] or verdicts[1]:
                approved.append(l)
                if not verdicts[0] or not verdicts[1]:
                    print(f"    Deliberation union-save: S{l.sentence_number} -> {l.component_name}")
            else:
                print(f"    Deliberation reject: S{l.sentence_number} -> {l.component_name}")

        return approved

    def _single_advocate_prosecutor_pass(self, snum, comp_name, sent_text, comp_names):
        """Run one pass of advocate-prosecutor-jury. Returns True=approve."""
        comp_names_str = ', '.join(comp_names)

        advocate_prompt = f"""You are the ADVOCATE for linking sentence S{snum} to component "{comp_name}".

The system has a component literally named "{comp_name}". Your job is to defend valid links.

SENTENCE: {sent_text}
ALL COMPONENTS: {comp_names_str}

Your job: Find the STRONGEST evidence that this sentence discusses "{comp_name}" at the architectural level. Consider:
- Does the sentence describe {comp_name}'s role, behavior, interactions, or testing?
- Is "{comp_name.lower()}" used as a standalone noun/noun-phrase referring to a layer or part of the system?
- In architecture docs, even generic words like "the {comp_name.lower()} of the application" typically
  refer to the named component when such a component exists.

Provide your argument in 2-3 sentences. Then give your verdict.
Return JSON: {{"argument": "your argument", "verdict": "APPROVE" or "REJECT"}}
JSON only:"""

        prosecutor_prompt = f"""You are the PROSECUTOR arguing AGAINST linking sentence S{snum} to component "{comp_name}".

You should only argue REJECT when there is CLEAR evidence the match is spurious — not just because the word has generic meanings.

SENTENCE: {sent_text}
ALL COMPONENTS: {comp_names_str}

Your job: Find CLEAR evidence that this is a SPURIOUS match. Only these patterns warrant rejection:
1. "{comp_name.lower()}" is used as a modifier/adjective in a compound phrase (e.g., "throttle {comp_name.lower()}", "minimal {comp_name.lower()}") — NOT as a standalone noun referring to the component.
2. The sentence is primarily about a DIFFERENT component, and "{comp_name.lower()}" is purely incidental.
3. "{comp_name.lower()}" refers to a technology/protocol/tool, not the architecture component.
4. This is a package listing or dotted path (like x.foo.bar) where the match is coincidental.

If "{comp_name.lower()}" appears as a standalone noun describing a layer/part of the system, that is NOT spurious.

Provide your argument in 2-3 sentences. Then give your verdict.
Return JSON: {{"argument": "your argument", "verdict": "APPROVE" or "REJECT"}}
JSON only:"""

        adv_data = self.llm.extract_json(self.llm.query(advocate_prompt, timeout=60))
        pros_data = self.llm.extract_json(self.llm.query(prosecutor_prompt, timeout=60))

        adv_arg = adv_data.get("argument", "") if adv_data else ""
        pros_arg = pros_data.get("argument", "") if pros_data else ""

        jury_prompt = f"""JURY: Decide if sentence S{snum} should be linked to component "{comp_name}".

The system has a component literally named "{comp_name}". REJECT only with clear evidence — when in doubt, APPROVE.

SENTENCE: {sent_text}

ADVOCATE argues: {adv_arg}

PROSECUTOR argues: {pros_arg}

Rule: APPROVE when "{comp_name.lower()}" is used as a standalone noun referring to a layer/part
of the system (its role, behavior, interactions, or testing). REJECT only when "{comp_name.lower()}"
is clearly used as a modifier in a compound phrase ("throttle {comp_name.lower()}"), a technology name,
or the sentence is entirely about a different component.

Return JSON: {{"verdict": "APPROVE" or "REJECT", "reason": "brief explanation"}}
JSON only:"""

        jury_data = self.llm.extract_json(self.llm.query(jury_prompt, timeout=60))
        if jury_data:
            return jury_data.get("verdict", "APPROVE").upper() == "APPROVE"
        return True

    # ═══════════════════════════════════════════════════════════════════════
    # Shared Helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _is_generic_mention(self, comp_name, sentence_text):
        """True if component appears only in lowercase (generic use)."""
        if not comp_name:
            return False
        if ' ' in comp_name or '-' in comp_name:
            return False
        if re.search(r'[a-z][A-Z]', comp_name):
            return False
        if comp_name.isupper():
            return False
        if comp_name[0].islower():
            return False
        if self._has_standalone_mention(comp_name, sentence_text):
            return False
        word_lower = comp_name.lower()
        if re.search(rf'\b{re.escape(word_lower)}\b', sentence_text):
            return True
        return False

    def _has_clean_mention(self, term, text):
        """Check if term appears cleanly (not in dotted path or hyphenated compound)."""
        pattern = rf'\b{re.escape(term)}\b'
        for m in re.finditer(pattern, text, re.IGNORECASE):
            s, e = m.start(), m.end()
            if s > 0 and text[s-1] == '.':
                continue
            if e < len(text) and text[e] == '.' and e + 1 < len(text) and text[e+1].isalpha():
                continue
            if (s > 0 and text[s-1] == '-') or (e < len(text) and text[e] == '-'):
                continue
            return True
        return False

    def _word_boundary_match(self, name, text):
        """Check if name appears as standalone word in text."""
        return bool(re.search(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE))

    def _has_standalone_mention(self, comp_name, text):
        """Check for non-generic, clean standalone mention of component name."""
        if not comp_name:
            return False
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

    def _has_alias_mention(self, comp_name, sentence_text):
        """Check if any known synonym or partial reference appears in the text."""
        if not self.doc_knowledge:
            return False
        text_lower = sentence_text.lower()
        for syn, target in self.doc_knowledge.synonyms.items():
            if target == comp_name:
                if re.search(rf'\b{re.escape(syn.lower())}\b', text_lower):
                    return True
        for partial, target in self.doc_knowledge.partial_references.items():
            if target == comp_name:
                if re.search(rf'\b{re.escape(partial.lower())}\b', text_lower):
                    return True
        return False

    def _is_ambiguous_name_component(self, comp_name):
        """True if single-word, non-CamelCase, non-uppercase, classified ambiguous."""
        if ' ' in comp_name or '-' in comp_name:
            return False
        if re.search(r'[a-z][A-Z]', comp_name):
            return False
        if comp_name.isupper():
            return False
        if not self.model_knowledge or not self.model_knowledge.ambiguous_names:
            return False
        return comp_name in self.model_knowledge.ambiguous_names

    def _find_matching_alias(self, comp_name, sent_text):
        """Find which alias triggered the match (synonym or partial)."""
        if not self.doc_knowledge:
            return None, None
        text_lower = sent_text.lower()
        for syn, target in self.doc_knowledge.synonyms.items():
            if target == comp_name:
                if re.search(rf'\b{re.escape(syn.lower())}\b', text_lower):
                    return syn, "synonym"
        for partial, target in self.doc_knowledge.partial_references.items():
            if target == comp_name:
                if re.search(rf'\b{re.escape(partial.lower())}\b', text_lower):
                    return partial, "partial"
        return None, None

    def _find_match_text(self, comp_name, sent_text):
        """Find what text in the sentence triggered the match to this component."""
        if not sent_text:
            return None
        if re.search(rf'\b{re.escape(comp_name)}\b', sent_text, re.IGNORECASE):
            return comp_name
        if self.doc_knowledge:
            for alias, comp in self.doc_knowledge.synonyms.items():
                if comp == comp_name and re.search(rf'\b{re.escape(alias)}\b', sent_text, re.IGNORECASE):
                    return alias
            for alias, comp in self.doc_knowledge.abbreviations.items():
                if comp == comp_name and re.search(rf'\b{re.escape(alias)}\b', sent_text, re.IGNORECASE):
                    return alias
            for partial, comp in self.doc_knowledge.partial_references.items():
                if comp == comp_name and partial.lower() in sent_text.lower():
                    return partial
        return None

    def _abbreviation_match_is_valid(self, abbrev, comp_name, sentence_text):
        """Context-aware validation for abbreviation matches."""
        comp_parts = comp_name.split()
        if len(comp_parts) < 2:
            return True
        if not comp_name.upper().startswith(abbrev.upper()):
            return True
        pattern = rf'\b{re.escape(abbrev)}\b'
        full_rest = comp_name[len(abbrev):].strip()
        found_valid = False
        for m in re.finditer(pattern, sentence_text, re.IGNORECASE):
            end = m.end()
            rest = sentence_text[end:].lstrip()
            if not rest:
                found_valid = True
                break
            if rest.lower().startswith(full_rest.lower()):
                found_valid = True
                break
            next_word_m = re.match(r'(\w+)', rest)
            if next_word_m:
                next_word = next_word_m.group(1).lower()
                expected_next = full_rest.split()[0].lower() if full_rest else ""
                if next_word != expected_next and next_word.isalpha():
                    continue
            found_valid = True
            break
        return found_valid

    def _get_comp_names(self, components) -> list[str]:
        """Get non-implementation component names."""
        return [c.name for c in components
                if not (self.model_knowledge and self.model_knowledge.is_implementation(c.name))]

    @staticmethod
    def _extract_json_array(text):
        """Extract a JSON array from LLM text that may have markdown fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        start = text.find("[")
        if start >= 0:
            depth = 0
            for j in range(start, len(text)):
                if text[j] == '[':
                    depth += 1
                elif text[j] == ']':
                    depth -= 1
                    if depth == 0:
                        try:
                            result = json.loads(text[start:j+1])
                            if isinstance(result, list):
                                return result
                        except json.JSONDecodeError:
                            continue
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # Checkpoint & Logging
    # ═══════════════════════════════════════════════════════════════════════

    def _checkpoint_dir(self, text_path):
        cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
        ds = os.path.splitext(os.path.basename(text_path))[0]
        d = os.path.join(cache_dir, "s_linker", ds)
        os.makedirs(d, exist_ok=True)
        return d

    def _save_phase(self, text_path, phase_name, state):
        d = self._checkpoint_dir(text_path)
        path = os.path.join(d, f"{phase_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"  Checkpoint: {phase_name} saved")

    def _log(self, phase, input_summary, output_summary, links=None):
        entry = {"phase": phase, "ts": time.time(), "in": input_summary, "out": output_summary}
        if links is not None:
            entry["links"] = [
                {"s": l.sentence_number, "c": l.component_name, "src": l.source}
                for l in links
            ]
        self._phase_log.append(entry)

    def _save_log(self, text_path):
        log_dir = os.environ.get("LLM_LOG_DIR", "./results/llm_logs")
        os.makedirs(log_dir, exist_ok=True)
        ds = os.path.splitext(os.path.basename(text_path))[0]
        path = os.path.join(log_dir, f"s_linker_{ds}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w") as f:
            json.dump(self._phase_log, f, indent=2, default=str)
        print(f"  Phase log saved: {path}")
