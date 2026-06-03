"""ILinker2 V32 standalone — V26a pipeline + ILinker2 seed + few-shot Phase 3 + CamelCase rescue + convention filter + checkpoints.

Fully self-contained: no inheritance from AgentLinker or AgentLinkerV26a.
All methods inlined from the parent classes.

Changes vs V31:
- CONVENTION_GUIDE: replaced all benchmark-derived examples with safe abstractions
  (cascade→throttle, dedicated hardware node→physical rack-mounted appliance,
   32-core server→multi-socket machine, preferences→bookmarks, conversion→rendering)
- Convention filter: removed partial_inject immunity. Test showed the convention
  filter catches 8/9 FPs with 0 TPs killed (test_partial_inject_unprotected.py,
  scenario A). The judge is harmful to partial_inject TPs but the convention filter
  is safe — it uses structural patterns (dotted paths, word modifiers) not semantics.
- Syn-safe judge bypass: KEPT for partial_inject (judge kills TPs, tested).
- Checkpoint dir: v32.
"""

import json
import os
import pickle
import re
import time
from collections import defaultdict
from typing import Optional

from llm_sad_sam.core.data_types import (
    SadSamLink, CandidateLink, DocumentProfile,
    ModelKnowledge, DocumentKnowledge, LearnedPatterns, EntityMention,
)
from llm_sad_sam.core.document_loader import DocumentLoader, Sentence
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


class ILinker2V35c:
    """V32 standalone: V26a pipeline + ILinker2 seed + few-shot Phase 3 + CamelCase rescue + convention filter + checkpoints."""

    CONTEXT_WINDOW = 3
    PRONOUN_PATTERN = re.compile(
        r'\b(it|they|this|these|that|those|its|their|the component|the service)\b',
        re.IGNORECASE
    )
    SOURCE_PRIORITY = {
        "transarc": 5, "validated": 4, "entity": 3,
        "coreference": 2, "partial_inject": 1,
    }

    # Few-shot examples (safe textbook domains only)
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
        self.doc_profile: Optional[DocumentProfile] = None
        self._phase_log = []
        self._is_complex = None
        self._ilinker2 = ILinker2(backend=self.llm.backend)
        print(f"ILinker2V35c standalone")
        print(f"  Backend: {self.llm.backend.value}, Model: {os.environ.get('CLAUDE_MODEL', 'default')}")

    # ── Checkpoint helpers ───────────────────────────────────────────────

    def _checkpoint_dir(self, text_path):
        cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
        ds = os.path.splitext(os.path.basename(text_path))[0]
        d = os.path.join(cache_dir, "v35c", ds)
        os.makedirs(d, exist_ok=True)
        return d

    def _save_phase(self, text_path, phase_name, state):
        d = self._checkpoint_dir(text_path)
        path = os.path.join(d, f"{phase_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"  Checkpoint: {phase_name} saved")

    # ── Logging ──────────────────────────────────────────────────────────

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
        path = os.path.join(log_dir, f"v25_{ds}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w") as f:
            json.dump(self._phase_log, f, indent=2, default=str)
        print(f"  Phase log saved: {path}")

    # ═════════════════════════════════════════════════════════════════════
    # Per-mention generic check
    # ═════════════════════════════════════════════════════════════════════

    def _is_generic_mention(self, comp_name, sentence_text):
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

    # ═════════════════════════════════════════════════════════════════════
    # Main pipeline with per-phase checkpoints
    # ═════════════════════════════════════════════════════════════════════

    def link(self, text_path, model_path, transarc_csv=None):
        self._cached_text_path = text_path
        self._cached_model_path = model_path
        self._phase_log = []
        t0 = time.time()

        components = parse_pcm_repository(model_path)
        sentences = DocumentLoader.load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        sent_map = DocumentLoader.build_sent_map(sentences)
        self._cached_components = components

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        # ── Phase 0 ─────────────────────────────────────────────────────
        print("\n[Phase 0] Document Profile")
        self.doc_profile = self._learn_document_profile(sentences, components)
        self._is_complex = self._structural_complexity(sentences, components)
        spc = len(sentences) / max(1, len(components))
        print(f"  Stats: {spc:.1f} sents/comp, {self.doc_profile.pronoun_ratio:.0%} pronouns")
        print(f"  Complex: {self._is_complex}")
        self._log("phase_0", {"sents": len(sentences), "comps": len(components)},
                  {"spc": spc, "complex": self._is_complex})

        self._save_phase(text_path, "phase0", {
            "doc_profile": self.doc_profile,
            "is_complex": self._is_complex,
        })

        # ── Phase 1 ─────────────────────────────────────────────────────
        print("\n[Phase 1] Model Structure")
        self.model_knowledge = self._analyze_model(components)
        arch = self.model_knowledge.architectural_names
        ambig = self.model_knowledge.ambiguous_names

        self.GENERIC_COMPONENT_WORDS = set()
        for name in ambig:
            if ' ' not in name and not name.isupper():
                self.GENERIC_COMPONENT_WORDS.add(name.lower())
        print(f"  Architectural: {len(arch)}, Ambiguous: {sorted(ambig)}")
        print(f"  Discovered generic component words: {sorted(self.GENERIC_COMPONENT_WORDS)}")

        self.GENERIC_PARTIALS = set()
        for comp in components:
            parts = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', comp.name)
            for part in parts:
                p_lower = part.lower()
                if part.isupper():
                    continue
                if len(p_lower) >= 3 and p_lower in ambig or any(
                    p_lower == a.lower() for a in ambig
                ):
                    self.GENERIC_PARTIALS.add(p_lower)
        for name in ambig:
            if ' ' not in name and not name.isupper():
                self.GENERIC_PARTIALS.add(name.lower())
        print(f"  Discovered generic partials: {sorted(self.GENERIC_PARTIALS)}")

        self._log("phase_1", {"components": [c.name for c in components]},
                  {"architectural": sorted(arch), "ambiguous": sorted(ambig),
                   "generic_words": sorted(self.GENERIC_COMPONENT_WORDS),
                   "generic_partials": sorted(self.GENERIC_PARTIALS)})

        self._save_phase(text_path, "phase1", {
            "model_knowledge": self.model_knowledge,
            "generic_component_words": self.GENERIC_COMPONENT_WORDS,
            "generic_partials": self.GENERIC_PARTIALS,
        })

        # ── Phase 2 ─────────────────────────────────────────────────────
        print("\n[Phase 2] Pattern Learning")
        self.learned_patterns = self._learn_patterns_with_debate(sentences, components)
        print(f"  Subprocess terms: {len(self.learned_patterns.subprocess_terms)}")
        self._log("phase_2", {}, {"subprocess": sorted(self.learned_patterns.subprocess_terms)})

        self._save_phase(text_path, "phase2", {
            "learned_patterns": self.learned_patterns,
        })

        # ── Phase 3 ─────────────────────────────────────────────────────
        print("\n[Phase 3] Document Knowledge")
        self.doc_knowledge = self._learn_document_knowledge_enriched(sentences, components)
        print(f"  Abbrev: {len(self.doc_knowledge.abbreviations)}, "
              f"Syn: {len(self.doc_knowledge.synonyms)}, "
              f"Generic: {len(self.doc_knowledge.generic_terms)}")
        self._log("phase_3", {}, {
            "abbreviations": self.doc_knowledge.abbreviations,
            "synonyms": self.doc_knowledge.synonyms,
            "generic": list(self.doc_knowledge.generic_terms),
        })

        # ── Phase 3b ────────────────────────────────────────────────────
        self._enrich_multiword_partials(sentences, components)

        self._save_phase(text_path, "phase3", {
            "doc_knowledge": self.doc_knowledge,
        })

        # ── Phase 4 ─────────────────────────────────────────────────────
        print("\n[Phase 4] TransArc")
        transarc_links = self._process_transarc()
        transarc_set = {(l.sentence_number, l.component_id) for l in transarc_links}
        print(f"  Links: {len(transarc_links)}")
        self._log("phase_4", {"csv": transarc_csv}, {"count": len(transarc_links)}, transarc_links)

        self._save_phase(text_path, "phase4", {
            "transarc_links": transarc_links,
            "transarc_set": transarc_set,
        })

        # ── Phase 5 ─────────────────────────────────────────────────────
        print("\n[Phase 5] Entity Extraction")
        candidates = self._extract_entities_enriched(sentences, components, name_to_id, sent_map)
        print(f"  Candidates: {len(candidates)}")

        before_guard = len(candidates)
        candidates = self._apply_abbreviation_guard_to_candidates(candidates, sent_map)
        if len(candidates) < before_guard:
            print(f"  After abbrev guard: {len(candidates)} (-{before_guard - len(candidates)})")
        self._log("phase_5", {}, {"count": len(candidates)})

        # ── Phase 5b ────────────────────────────────────────────────────
        entity_comps = {c.component_name for c in candidates}
        transarc_comps = {l.component_name for l in transarc_links}
        covered_comps = entity_comps | transarc_comps
        unlinked = [c for c in components if c.name not in covered_comps]

        if unlinked:
            print(f"\n[Phase 5b] Targeted Recovery ({len(unlinked)} unlinked components)")
            extra = self._targeted_extraction(unlinked, sentences, name_to_id, sent_map,
                                              components=components, transarc_links=transarc_links,
                                              entity_candidates=candidates)
            if extra:
                print(f"  Found: {len(extra)} additional candidates")
                candidates.extend(extra)
            else:
                print(f"  Found: 0 additional candidates")
            self._log("phase_5b", {"unlinked": [c.name for c in unlinked]},
                      {"found": len(extra) if extra else 0})

        self._save_phase(text_path, "phase5", {
            "candidates": candidates,
        })

        # ── Phase 6 ─────────────────────────────────────────────────────
        print("\n[Phase 6] Validation")
        validated = self._validate_intersect(candidates, components, sent_map)
        print(f"  Validated: {len(validated)} (of {len(candidates)})")
        self._log("phase_6", {"candidates": len(candidates)}, {"validated": len(validated)})

        self._save_phase(text_path, "phase6", {
            "validated": validated,
        })

        # ── Phase 7 ─────────────────────────────────────────────────────
        print("\n[Phase 7] Coreference")
        if self._is_complex:
            print(f"  Mode: debate (complex, {len(sentences)} sents)")
            coref_links = self._coref_debate(sentences, components, name_to_id, sent_map)
        else:
            print(f"  Mode: discourse ({len(sentences)} sents)")
            coref_links = self._coref_discourse(sentences, components, name_to_id, sent_map)

        print(f"  Coref links: {len(coref_links)}")
        self._log("phase_7", {"method": "debate" if self._is_complex else "discourse"},
                  {"count": len(coref_links)}, coref_links)

        self._save_phase(text_path, "phase7", {
            "coref_links": coref_links,
        })

        # ── Phase 8b ────────────────────────────────────────────────────
        partial_links = self._inject_partial_references(
            sentences, components, name_to_id, transarc_set,
            {(c.sentence_number, c.component_id) for c in validated},
            {(l.sentence_number, l.component_id) for l in coref_links},
            set(),
        )
        if partial_links:
            print(f"\n[Phase 8b] Partial Injection")
            print(f"  Injected: {len(partial_links)} candidates")
            self._log("phase_8b", {}, {"count": len(partial_links)}, partial_links)

        # ── Combine + deduplicate ────────────────────────────────────────
        entity_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
            for c in validated
        ]
        all_links = transarc_links + entity_links + coref_links + partial_links
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
        self._log("dedup", {"raw": len(all_links)}, {"deduped": len(preliminary)}, preliminary)

        # ── Parent-overlap guard ─────────────────────────────────────────
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

        # ── Phase 8c ─────────────────────────────────────────────────────
        print("\n[Phase 8c] Boundary Filters (non-TransArc)")
        preliminary, boundary_rejected = self._apply_boundary_filters(
            preliminary, sent_map, transarc_set
        )
        if boundary_rejected:
            print(f"  Rejected: {len(boundary_rejected)}")
            self._log("phase_8c", {},
                      {"rejected": len(boundary_rejected),
                       "details": [(lk.component_name, reason) for lk, reason in boundary_rejected]})

        self._save_phase(text_path, "pre_judge", {
            "preliminary": preliminary,
            "transarc_set": transarc_set,
        })

        # ── Phase 9 ─────────────────────────────────────────────────────
        print("\n[Phase 9] Judge Review (TransArc immune)")
        reviewed = self._judge_review(preliminary, sentences, components, sent_map, transarc_set)
        rejected = [l for l in preliminary if (l.sentence_number, l.component_id)
                    not in {(r.sentence_number, r.component_id) for r in reviewed}]
        print(f"  Approved: {len(reviewed)} (rejected {len(rejected)})")
        self._log("phase_9", {"input": len(preliminary)},
                  {"approved": len(reviewed), "rejected": len(rejected)}, reviewed)
        if rejected:
            self._log("phase_9_rejected", {}, {"count": len(rejected)}, rejected)

        final = reviewed

        # ── Save log + final checkpoint ──────────────────────────────────
        self._log("summary", {"total_time_s": round(time.time() - t0, 1)},
                  {"final": len(final)}, final)
        self._save_log(text_path)

        self._save_phase(text_path, "final", {
            "final": final,
            "reviewed": reviewed,
            "rejected": rejected,
        })

        print(f"\nFinal: {len(final)} links")
        return final

    # ── Phase 4: ILinker2 seed ───────────────────────────────────────────

    def _process_transarc(self):
        """Run ILinker2 for seed link generation."""
        raw_links = self._ilinker2.link(self._cached_text_path, self._cached_model_path)

        result = []
        for lk in raw_links:
            result.append(SadSamLink(
                sentence_number=lk.sentence_number,
                component_id=lk.component_id,
                component_name=lk.component_name,
                confidence=lk.confidence,
                source="transarc",
            ))
        return result

    # ── Phase 3: Few-shot calibrated judge (from V30b) ───────────────────

    def _learn_document_knowledge_enriched(self, sentences, components):
        """Phase 3: Few-shot calibrated judge — no code-level overrides."""
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

        # CamelCase rescue: terms with CamelCase transitions are constructed
        # identifiers, never generic English — force-approve if judge rejected
        for term in list(generic_terms):
            if re.search(r'[a-z][A-Z]', term) and term in all_mappings:
                generic_terms.discard(term)
                approved.add(term)
                print(f"    CamelCase override (rescued): {term}")

        knowledge = DocumentKnowledge()
        knowledge.generic_terms = generic_terms

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

        if generic_terms:
            print(f"    Generic (rejected): {', '.join(list(generic_terms)[:5])}")

        # Deterministic CamelCase-split synonym injection (universal SE convention)
        for comp in [c.name for c in components]:
            split = re.sub(r'([a-z])([A-Z])', r'\1 \2', comp)
            split = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', split)
            if split != comp and split not in knowledge.synonyms:
                knowledge.synonyms[split] = comp
                print(f"    CamelCase syn: {split} -> {comp}")

        return knowledge

    # ── Convention-aware boundary filter (replaces regex P8c) ──────────

    def _apply_boundary_filters(self, links, sent_map, transarc_set):
        """LLM convention filter using 3-step reasoning guide.

        V32 change: partial_inject links are NO LONGER immune. The convention
        filter uses structural patterns (dotted paths, word modifiers) that
        safely distinguish generic uses from component references without
        killing TPs (tested: 8/9 FP caught, 0 TP killed).

        TransArc links remain immune (handled by Phase 9 judge).
        """
        comp_names = self._get_comp_names(self._cached_components) if hasattr(self, '_cached_components') else []

        safe = []
        to_review = []
        for lk in links:
            is_ta = (lk.sentence_number, lk.component_id) in transarc_set
            if is_ta:
                safe.append(lk)
            else:
                to_review.append(lk)

        if not to_review:
            return safe, []

        # Build items for LLM
        items = []
        for i, lk in enumerate(to_review):
            sent = sent_map.get(lk.sentence_number)
            text = sent.text if sent else "(no text)"
            items.append(
                f'{i+1}. S{lk.sentence_number}: "{text}"\n'
                f'   Component: "{lk.component_name}"'
            )

        # Process in batches
        batch_size = 25
        all_verdicts = {}

        for batch_start in range(0, len(items), batch_size):
            batch_items = items[batch_start:batch_start + batch_size]

            prompt = f"""Validate trace links between architecture documentation and components.

ARCHITECTURE COMPONENTS: {', '.join(comp_names)}

{CONVENTION_GUIDE}

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
                print(f"    Convention filter [step {step}]: S{lk.sentence_number} → "
                      f"{lk.component_name} ({lk.source}) — {reason}")
            else:
                kept.append(lk)

        return kept, rejected

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
                            break
        return None

    # ── Phase 5b: Targeted extraction with dotted-path prompt ────────────

    def _targeted_extraction(self, unlinked_components, sentences, name_to_id, sent_map,
                              components=None, transarc_links=None, entity_candidates=None):
        """Adds dotted-path exclusion instruction to the prompt."""
        if not unlinked_components:
            return []

        parent_map = {}
        existing_sent_comp = defaultdict(set)
        if components and transarc_links:
            all_comp_names = {c.name for c in components}
            for comp in unlinked_components:
                parents = set()
                for other_name in all_comp_names:
                    if other_name != comp.name and len(other_name) >= 3 and other_name in comp.name:
                        parents.add(other_name)
                if parents:
                    parent_map[comp.name] = parents
            for lk in transarc_links:
                existing_sent_comp[lk.sentence_number].add(lk.component_name)
            if entity_candidates:
                for c in entity_candidates:
                    existing_sent_comp[c.sentence_number].add(c.component_name)

        all_extra = []
        for comp in unlinked_components:
            doc_sents = sentences

            doc_text = "\n".join([f"S{s.number}: {s.text}" for s in doc_sents])

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
                        print(f"    Targeted skip (parent overlap): S{snum} -> {comp.name} "
                              f"(parent: {', '.join(parents_here)})")
                        continue

                matched = ref.get("matched_text", comp.name)
                all_extra.append(CandidateLink(
                    snum, sent.text, comp.name, cid,
                    matched, 0.85, "entity", "targeted", True
                ))
                print(f"    Targeted: S{snum} -> {comp.name} ({matched})")

        return all_extra

    def _build_judge_prompt(self, comp_names, cases):
        """Generalized 4-rule judge with reframed criteria."""
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

    # ═════════════════════════════════════════════════════════════════════
    # Phase 0: Document profile + structural complexity
    # ═════════════════════════════════════════════════════════════════════

    def _learn_document_profile(self, sentences, components):
        texts = [s.text for s in sentences]
        comp_names = [c.name for c in components]
        pron = r'\b(it|they|this|these|that|those|its|their)\b'
        pronoun_ratio = sum(1 for t in texts if re.search(pron, t.lower())) / max(1, len(sentences))
        mentions = sum(1 for t in texts for c in comp_names if c.lower() in t.lower())
        density = mentions / max(1, len(sentences))
        spc = len(sentences) / max(1, len(components))
        return DocumentProfile(
            sentence_count=len(sentences), component_count=len(components),
            pronoun_ratio=pronoun_ratio, technical_density=density,
            component_mention_density=density,
            complexity_score=min(1.0, spc / 20),
            recommended_strictness="balanced",
        )

    def _structural_complexity(self, sentences, components):
        """Dynamic threshold based on document characteristics, not hardcoded spc_min."""
        comp_names = [c.name for c in components]
        mention_count = sum(1 for sent in sentences
                          if any(cn.lower() in sent.text.lower() for cn in comp_names))
        uncovered_ratio = 1.0 - (mention_count / max(1, len(sentences)))
        spc = len(sentences) / max(1, len(components))
        result = uncovered_ratio > 0.5 and spc > 4
        print(f"  Structural complexity: uncovered={uncovered_ratio:.1%}, spc={spc:.1f} -> {result}")
        return result

    # ═════════════════════════════════════════════════════════════════════
    # Phase 1: Model Structure Analysis
    # ═════════════════════════════════════════════════════════════════════

    def _analyze_model(self, components):
        """Analyze model structure."""
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

        word_to_comps = {}
        for name in names:
            for word in re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', name):
                w = word.lower()
                if len(w) >= 3:
                    word_to_comps.setdefault(w, []).append(name)
        knowledge.shared_vocabulary = {w: list(set(c)) for w, c in word_to_comps.items() if len(set(c)) > 1}

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
            knowledge.architectural_names = set(data.get("architectural", [])) & valid
            raw_ambiguous = set(data.get("ambiguous", [])) & valid
            knowledge.ambiguous_names = {
                n for n in raw_ambiguous
                if len(n.split()) == 1 and not self._is_structurally_unambiguous(n)
            }

    # ═════════════════════════════════════════════════════════════════════
    # Phase 3b: Multi-word partial enrichment
    # ═════════════════════════════════════════════════════════════════════

    def _enrich_multiword_partials(self, sentences, components):
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
                added.append(f"{last_word} -> {comp.name} ({mention_count} caps mentions)")

        if added:
            print(f"\n[Phase 3b] Multi-word Enrichment")
            for a in added:
                print(f"    Auto-partial: {a}")
            self._log("phase_3b", {}, {"added": added})

    # ═════════════════════════════════════════════════════════════════════
    # Phase 5: Entity extraction (enriched prompt)
    # ═════════════════════════════════════════════════════════════════════

    def _extract_entities_enriched(self, sentences, components, name_to_id, sent_map):
        """Phase 5 v2: Balanced prompt + retry on empty response."""
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

            if len(sentences) > batch_size:
                print(f"    Entity batch {batch_start//batch_size + 1}: "
                      f"S{batch[0].number}-S{batch[-1].number} ({len(batch)} sents)")

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

Return JSON (example output):
{{"references": [{{"sentence": 7, "component": "Scheduler", "matched_text": "the scheduler", "match_type": "partial"}}, {{"sentence": 12, "component": "MemoryManager", "matched_text": "Memory Manager", "match_type": "exact"}}]}}
JSON only:"""

            # Retry once on empty response
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
                # Handle "S101" format from some backends (strip "S" prefix)
                if isinstance(snum, str):
                    snum = snum.lstrip("S")
                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue
                sent = sent_map.get(snum)
                if not sent:
                    continue

                # Verify matched_text is actually in sentence
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

    # ═════════════════════════════════════════════════════════════════════
    # Phase 6: Validation — Code-first + LLM-fallback
    # ═════════════════════════════════════════════════════════════════════

    def _word_boundary_match(self, name, text):
        """Check if name appears as standalone word in text (word-boundary match)."""
        return bool(re.search(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE))

    def _validate_intersect(self, candidates, components, sent_map):
        """Phase 6: Code-first v2 + 2-pass intersect + evidence post-filter for generic names."""
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
                print(f"    Generic mention reject: S{c.sentence_number} -> {c.component_name}")
            else:
                remaining.append(c)
        needs = remaining

        # Build alias lookup — include ALL aliases for word-boundary matching
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

        # Step 1: Word-boundary code-first — handles short names (UI, DB) correctly
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

        print(f"    Code-first v2 auto-approved: {len(auto_approved)}, LLM needed: {len(llm_needed)}")

        # Classify generic-risk components
        generic_risk = set()
        if self.model_knowledge and self.model_knowledge.ambiguous_names:
            generic_risk |= self.model_knowledge.ambiguous_names
        for c in components:
            if c.name.lower() in self.GENERIC_COMPONENT_WORDS:
                generic_risk.add(c.name)

        # Step 2: 2-pass intersect for ALL LLM-needed
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

        print(f"    2-pass approved: {len(twopass_approved)} specific, "
              f"{len(generic_to_verify)} generic need evidence")

        # Step 3: Evidence post-filter for generic-risk that passed 2-pass
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

            print(f"    Generic evidence: {len(generic_validated)}/{len(generic_to_verify)}")

        return direct + auto_approved + twopass_approved + generic_validated

    def _qual_validation_pass(self, comp_names, ctx, cases, focus):
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

    # ═════════════════════════════════════════════════════════════════════
    # Phase 7: Coreference
    # ═════════════════════════════════════════════════════════════════════

    def _coref_discourse(self, sentences, components, name_to_id, sent_map):
        """Phase 7 discourse mode with required antecedent citation."""
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
                # Handle "S6" format from some backends (strip "S" prefix)
                if isinstance(snum, str):
                    snum = snum.lstrip("S")
                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue

                # Verify antecedent citation
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
                        print(f"    Coref skip (bad antecedent S{ant_snum}): S{snum} -> {comp}")
                        continue
                    if not (self._has_standalone_mention(comp, ant_sent.text) or
                            self._has_alias_mention(comp, ant_sent.text)):
                        print(f"    Coref skip (comp not in antecedent S{ant_snum}): S{snum} -> {comp}")
                        continue
                    if abs(snum - ant_snum) > 3:
                        print(f"    Coref skip (antecedent too far S{ant_snum}): S{snum} -> {comp}")
                        continue

                sent = sent_map.get(snum)
                if sent and self.learned_patterns and self.learned_patterns.is_subprocess(sent.text):
                    continue
                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, 1.0, "coreference"))

        return all_coref

    def _coref_debate(self, sentences, components, name_to_id, sent_map):
        """Phase 7 debate mode with required antecedent citation."""
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

            # Verify antecedent citations in code
            for res in proposed:
                comp = res.get("component")
                snum = res.get("sentence")
                if not (comp and snum and comp in name_to_id):
                    continue
                # Handle "S6" format from some backends (strip "S" prefix)
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
                        print(f"    Coref verify-fail (S{ant_snum} doesn't mention {comp}): S{snum} -> {comp}")
                        continue
                    if abs(snum - ant_snum) > 3:
                        continue

                sent = sent_map.get(snum)
                if sent and self.learned_patterns and self.learned_patterns.is_subprocess(sent.text):
                    continue
                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, 1.0, "coreference"))

        return all_coref

    # ═════════════════════════════════════════════════════════════════════
    # Partial reference helpers
    # ═════════════════════════════════════════════════════════════════════

    def _has_clean_mention(self, term, text):
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

    def _inject_partial_references(self, sentences, components, name_to_id,
                                    transarc_set, validated_set, coref_set, implicit_set):
        if not self.doc_knowledge or not self.doc_knowledge.partial_references:
            return []

        existing = transarc_set | validated_set | coref_set | implicit_set
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

    # ═════════════════════════════════════════════════════════════════════
    # Abbreviation guard
    # ═════════════════════════════════════════════════════════════════════

    def _abbreviation_match_is_valid(self, abbrev, comp_name, sentence_text):
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

    def _apply_abbreviation_guard_to_candidates(self, candidates, sent_map):
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

    # ═════════════════════════════════════════════════════════════════════
    # Phase 9: Judge — TransArc immune, adaptive ctx, show match
    # ═════════════════════════════════════════════════════════════════════

    def _has_standalone_mention(self, comp_name, text):
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
        """Check if any known synonym or partial reference for this component appears in the text."""
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
        """True if component name is single-word, not CamelCase, not all-uppercase,
        and was classified as ambiguous by Phase 1."""
        if ' ' in comp_name or '-' in comp_name:
            return False
        if re.search(r'[a-z][A-Z]', comp_name):
            return False
        if comp_name.isupper():
            return False
        if not self.model_knowledge or not self.model_knowledge.ambiguous_names:
            return False
        return comp_name in self.model_knowledge.ambiguous_names

    def _judge_review(self, links, sentences, components, sent_map, transarc_set):
        if len(links) < 5:
            return links

        comp_names = self._get_comp_names(components)

        # Triage: safe (TransArc/standalone/synonym-backed), alias-matched, no-match, ta-review
        safe, nomatch_links, ta_review = [], [], []
        syn_safe_count = 0
        for l in links:
            is_ta = (l.sentence_number, l.component_id) in transarc_set
            sent = sent_map.get(l.sentence_number)
            # Synonym-safe: if a Phase 3 synonym/partial appears in the sentence,
            # trust the doc_knowledge and skip judge review entirely
            if sent and self._has_alias_mention(l.component_name, sent.text):
                safe.append(l)
                syn_safe_count += 1
                continue
            if is_ta:
                if self._is_ambiguous_name_component(l.component_name):
                    ta_review.append(l)
                else:
                    safe.append(l)
                continue
            if not sent:
                nomatch_links.append(l)
                continue
            if self._has_standalone_mention(l.component_name, sent.text):
                safe.append(l)
            else:
                nomatch_links.append(l)

        print(f"  Triage: {len(safe)} safe ({syn_safe_count} syn-safe), {len(ta_review)} ta-review, "
              f"{len(nomatch_links)} no-match")

        # ── Advocate-Prosecutor deliberation for ambiguous TransArc links ──
        ta_approved = self._deliberate_transarc(ta_review, comp_names, sent_map)

        # ── Standard 4-rule judge for no-match links ──
        nomatch_approved = self._judge_nomatch(nomatch_links, comp_names, sent_map)

        return safe + ta_approved + nomatch_approved

    def _judge_nomatch(self, nomatch_links, comp_names, sent_map):
        """Standard 4-rule judge for non-alias links, with union voting."""
        if not nomatch_links:
            return []

        cases = self._build_judge_cases(nomatch_links, sent_map)
        prompt = self._build_judge_prompt(comp_names, cases)
        n = min(30, len(nomatch_links))

        # Union voting: reject only if BOTH passes reject
        data1 = self.llm.extract_json(self.llm.query(prompt, timeout=180))
        data2 = self.llm.extract_json(self.llm.query(prompt, timeout=180))

        rej1 = self._parse_rejections(data1, n)
        rej2 = self._parse_rejections(data2, n)
        rejected = rej1 & rej2

        result = []
        for i in range(n):
            if i not in rejected:
                result.append(nomatch_links[i])
            else:
                print(f"    4-rule reject: S{nomatch_links[i].sentence_number} -> {nomatch_links[i].component_name}")
        result.extend(nomatch_links[n:])
        return result

    def _parse_rejections(self, data, n):
        rejected = set()
        if data:
            for j in data.get("judgments", []):
                idx = j.get("case", 0) - 1
                if 0 <= idx < n and not j.get("approve", False):
                    rejected.add(idx)
        return rejected

    def _find_match_text(self, comp_name, sent_text):
        """Find what text in the sentence triggered the match to this component."""
        if not sent_text:
            return None
        text_lower = sent_text.lower()
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
                if comp == comp_name and partial.lower() in text_lower:
                    return partial
        return None

    def _build_judge_cases(self, review, sent_map):
        # Adaptive context: full for simple docs, truncated for complex
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

            # Show match text (J6)
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

    # ── Advocate-Prosecutor Deliberation for TransArc links ──────────

    def _deliberate_transarc(self, links, comp_names, sent_map):
        """Advocate-Prosecutor deliberation for ambiguous-name TransArc links."""
        if not links:
            return []

        print(f"  Deliberating {len(links)} ambiguous-name TransArc links (advocate-prosecutor)")

        approved = []
        for l in links:
            sent = sent_map.get(l.sentence_number)
            if not sent:
                approved.append(l)
                continue

            # Union voting: two independent deliberation passes
            verdicts = []
            for pass_num in range(2):
                verdict = self._single_advocate_prosecutor_pass(
                    l.sentence_number, l.component_name, sent.text, comp_names)
                verdicts.append(verdict)

            if verdicts[0] or verdicts[1]:  # Union: approve if either pass approves
                approved.append(l)
                if not verdicts[0] or not verdicts[1]:
                    print(f"    Deliberation union-save: S{l.sentence_number} -> {l.component_name}")
            else:
                print(f"    Deliberation reject: S{l.sentence_number} -> {l.component_name}")

        return approved

    def _single_advocate_prosecutor_pass(self, snum, comp_name, sent_text, comp_names):
        """Run one pass of advocate-prosecutor-jury. Returns True=approve, False=reject."""
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

        # Run advocate and prosecutor independently
        adv_data = self.llm.extract_json(self.llm.query(advocate_prompt, timeout=60))
        pros_data = self.llm.extract_json(self.llm.query(prosecutor_prompt, timeout=60))

        adv_arg = adv_data.get("argument", "") if adv_data else ""
        pros_arg = pros_data.get("argument", "") if pros_data else ""

        # Jury sees both arguments
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
        return True  # Default approve on failure

    # ═════════════════════════════════════════════════════════════════════
    # Helper methods from AgentLinker (grandparent)
    # ═════════════════════════════════════════════════════════════════════

    def _get_comp_names(self, components) -> list[str]:
        """Get non-implementation component names."""
        return [c.name for c in components
                if not (self.model_knowledge and self.model_knowledge.is_implementation(c.name))]

    # ═════════════════════════════════════════════════════════════════════
    # Phase 2: Pattern learning with debate
    # ═════════════════════════════════════════════════════════════════════

    def _learn_patterns_with_debate(self, sentences, components) -> LearnedPatterns:
        """Learn patterns using agent debate."""
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
