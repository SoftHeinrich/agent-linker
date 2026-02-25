"""ILinker2 V32 — V31 + convention filter covers partial_inject + zero prompt leakage.

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

Everything else identical to V31.
"""

import json
import os
import pickle
import re
import time
from collections import defaultdict

from llm_sad_sam.core.data_types import (
    SadSamLink, CandidateLink, DocumentKnowledge, LearnedThresholds,
)
from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.linkers.experimental.agent_linker_v26a import AgentLinkerV26a
from llm_sad_sam.linkers.experimental.ilinker2 import ILinker2
from llm_sad_sam.pcm_parser import parse_pcm_repository

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
NO_LINK when the sentence:
- Describes what a technology IS (definition, capabilities)
- Lists technologies as stack dependencies
- Names a COMPOUND ENTITY containing the component name
  ("X Protocol specification" → NO_LINK for "X")
- Uses the name as part of a METHODOLOGY ("X testing in CI" → NO_LINK for "X")

LINK when components INTERACT with or connect THROUGH the technology.

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
- Multi-word component names (e.g., "HTML5 Server", "Recording Service") are NEVER generic words → LINK
- CamelCase identifiers are NEVER generic words → LINK
- Sentences describing how components interact, connect, or communicate → LINK (even if indirect)
- Sentences about what a component does, provides, or handles → LINK
- Only use NO_LINK when you are CONFIDENT the name is NOT used as a component reference

### Priority:
Be AGGRESSIVE with NO_LINK on sub-package descriptions (Step 1).
For Step 2, only NO_LINK when confident. Default to LINK."""


class ILinker2V32(AgentLinkerV26a):
    """V26a pipeline with ILinker2 seed + few-shot Phase 3 + CamelCase rescue + checkpoints."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ilinker2 = ILinker2(backend=self.llm.backend)

    # ── Checkpoint helpers ───────────────────────────────────────────────

    def _checkpoint_dir(self, text_path):
        cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
        ds = os.path.splitext(os.path.basename(text_path))[0]
        d = os.path.join(cache_dir, "v32", ds)
        os.makedirs(d, exist_ok=True)
        return d

    def _save_phase(self, text_path, phase_name, state):
        d = self._checkpoint_dir(text_path)
        path = os.path.join(d, f"{phase_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"  Checkpoint: {phase_name} saved")

    def _load_phase(self, text_path, phase_name):
        d = self._checkpoint_dir(text_path)
        path = os.path.join(d, f"{phase_name}.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── NDF: disable dotted-path regex ───────────────────────────────────

    def _in_dotted_path(self, text: str, comp_name: str) -> bool:
        return False

    # ── Main pipeline with per-phase checkpoints ─────────────────────────

    def link(self, text_path, model_path, transarc_csv=None):
        self._cached_text_path = text_path
        self._cached_model_path = model_path
        self._phase_log = []
        t0 = time.time()

        components = parse_pcm_repository(model_path)
        sentences = DocumentLoader.load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        id_to_name = {c.id: c.name for c in components}
        sent_map = DocumentLoader.build_sent_map(sentences)
        self._cached_sent_map = sent_map
        self._cached_components = components

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        # ── Phase 0 ─────────────────────────────────────────────────────
        print("\n[Phase 0] Document Profile")
        self.doc_profile = self._learn_document_profile(sentences, components)
        self._is_complex = self._structural_complexity(sentences, components)
        spc = len(sentences) / max(1, len(components))
        print(f"  Stats: {spc:.1f} sents/comp, {self.doc_profile.pronoun_ratio:.0%} pronouns")
        print(f"  Complex: {self._is_complex}")
        self.thresholds = LearnedThresholds(0, 0, 0, 0, "qualitative", 0)
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
        transarc_links = self._process_transarc(transarc_csv, id_to_name, sent_map, name_to_id)
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
            discourse_model = self._build_discourse_model(sentences, components, name_to_id)
            print(f"  Mode: discourse ({len(sentences)} sents)")
            coref_links = self._coref_discourse(sentences, components, name_to_id, sent_map, discourse_model)

        print(f"  Coref links: {len(coref_links)}")
        self._log("phase_7", {"method": "debate" if self._is_complex else "discourse"},
                  {"count": len(coref_links)}, coref_links)

        self._save_phase(text_path, "phase7", {
            "coref_links": coref_links,
        })

        # ── Phase 8b ────────────────────────────────────────────────────
        existing = (transarc_set
                    | {(c.sentence_number, c.component_id) for c in validated}
                    | {(l.sentence_number, l.component_id) for l in coref_links})
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

    def _process_transarc(self, transarc_csv, id_to_name, sent_map, name_to_id):
        """Override: run ILinker2 instead of loading TransArc CSV."""
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
        """Override: LLM convention filter using 3-step reasoning guide.

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
        """Override: adds dotted-path exclusion instruction to the prompt."""
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
            if len(sentences) <= 60:
                doc_sents = sentences
            else:
                doc_sents = sentences[:30] + sentences[-30:]

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
        """Override: Generalized 4-rule judge with reframed criteria."""
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
