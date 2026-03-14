"""S-Linker3: DAG-based SAD-SAM trace link recovery.

Derived from S-Linker2. Key changes vs S-Linker2:
- Unified coreference resolution: cases-in-context (Variant E) replaces the
  complexity-gated discourse/debate split. Per-case presentation with ±5
  bidirectional context window. Cross-model Pareto winner (0 FP on both
  Claude Sonnet and GPT-5.2). No complexity gate needed.
- Judge replaced with keep_coref filter: drops no-match links (where component
  name is absent from the sentence) instead of running LLM judge. Coref links
  kept (antecedent verification is their built-in context protection). Matches
  or beats judge at +0.1pp macro F1 with 0 LLM calls.

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
  Tier 3: Merge and Deterministic Filter (sequential)
    Priority dedup → parent-overlap guard → convention-aware boundary
    filter → keep_coref filter (drop no-match, no LLM judge)
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
from llm_sad_sam.linkers.experimental.prompts import (
    CONVENTION_GUIDE, AMBIGUITY_FEW_SHOT, AMBIGUITY_RULES,
    DOC_KNOWLEDGE_JUDGE_EXAMPLES, DOC_KNOWLEDGE_JUDGE_RULES,
    DOC_KNOWLEDGE_EXTRACTION_RULES,
    ENTITY_EXTRACTION_RULES, VALIDATION_RULES, COREF_RULES,
)
from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.linkers.experimental.ilinker2 import ILinker2
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend

class SLinker3:
    """DAG-based SAD-SAM trace link recovery (standalone). Unified coref + no judge."""

    CONTEXT_WINDOW = 3
    PRONOUN_PATTERN = re.compile(
        r'\b(it|they|this|these|that|those|its|their|the component|the service)\b',
        re.IGNORECASE
    )
    SOURCE_PRIORITY = {
        "seed": 5, "validated": 4, "entity": 3,
        "coreference": 2, "partial_inject": 1,
    }

    _FEW_SHOT = AMBIGUITY_FEW_SHOT

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
        print(f"SLinker3 (unified coref + no judge)")
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
            print("  WARNING: transarc_csv provided but SLinker3 uses ILinker2 seed; ignoring.")
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

        self._save_phase(text_path, "pre_filter", {
            "preliminary": preliminary,
            "seed_set": seed_set,
        })

        # Keep-coref deterministic filter (replaces LLM judge)
        # Keep links that have: standalone mention, alias mention, seed source, or coref source
        # Drop no-match links (component name absent from sentence) — these lack context protection
        kept, dropped = [], []
        for l in preliminary:
            if l.source == "coreference":
                kept.append(l)
                continue
            is_seed = (l.sentence_number, l.component_id) in seed_set
            if is_seed:
                kept.append(l)
                continue
            sent = sent_map.get(l.sentence_number)
            if not sent:
                dropped.append(l)
                continue
            if self._has_standalone_mention(l.component_name, sent.text):
                kept.append(l)
            elif self._has_alias_mention(l.component_name, sent.text):
                kept.append(l)
            else:
                dropped.append(l)

        print(f"  Keep-coref filter: kept {len(kept)}, dropped {len(dropped)}")
        self._log("keep_coref", {"input": len(preliminary)},
                  {"kept": len(kept), "dropped": len(dropped),
                   "drop_details": [(l.component_name, l.sentence_number, l.source) for l in dropped]})

        final = kept

        # Save log + final checkpoint
        self._log("summary", {"total_time_s": round(time.time() - t0, 1)},
                  {"final": len(final)}, final)
        self._save_log(text_path)

        self._save_phase(text_path, "final", {
            "final": final,
            "kept": kept,
            "dropped": dropped,
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

{AMBIGUITY_RULES}

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

{DOC_KNOWLEDGE_EXTRACTION_RULES}

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

{DOC_KNOWLEDGE_JUDGE_EXAMPLES}

{DOC_KNOWLEDGE_JUDGE_RULES}

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
        """Unified coreference: cases-in-context (Variant E).

        Per-case presentation with ±5 bidirectional context window.
        No complexity gate. Cross-model Pareto winner (0 FP on both Claude and GPT-5.2).
        """
        pronoun_count = sum(1 for s in sentences if self.PRONOUN_PATTERN.search(s.text))
        print(f"    Coreference: cases-in-context ({pronoun_count} pronoun sents / {len(sentences)} total)")
        return self._coref_cases_in_context(sentences, components, name_to_id, sent_map)

    def _run_single_extraction_pass(self, sentences, comp_names, comp_lower, mappings,
                                     name_to_id, sent_map, pass_label=""):
        """Run one pass of entity extraction over all batches. Returns dict of (snum, cid) -> CandidateLink."""
        batch_size = 50
        candidates = {}

        for batch_start in range(0, len(sentences), batch_size):
            batch = sentences[batch_start:batch_start + batch_size]

            if len(sentences) > batch_size:
                print(f"    {pass_label}Entity batch {batch_start//batch_size + 1}: "
                      f"S{batch[0].number}-S{batch[-1].number} ({len(batch)} sents)")

            prompt = f"""Extract ALL references to software architecture components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings[:20])}' if mappings else ''}

{ENTITY_EXTRACTION_RULES}

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
                    print(f"    {pass_label}Empty response, retrying batch...")

            if not data:
                continue

            for ref in data.get("references", []):
                snum, cname = ref.get("sentence"), ref.get("component")
                if snum is None or not cname or cname not in name_to_id:
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
                if key not in candidates:
                    candidates[key] = CandidateLink(snum, sent.text, cname, name_to_id[cname],
                                               matched, 0.85, "entity",
                                               ref.get("match_type", "exact"), needs_val)

        return candidates

    def _extract_entities_enriched(self, sentences, components, name_to_id, sent_map):
        """Phase 5: Two-pass intersection for variance reduction.

        Runs entity extraction twice independently, keeps only candidates found
        in BOTH passes. This eliminates variance-driven FPs where one pass
        over-extracts candidates the other doesn't find.
        """
        comp_names = self._get_comp_names(components)
        comp_lower = {n.lower() for n in comp_names}

        mappings = []
        if self.doc_knowledge:
            mappings.extend([f"{a}={c}" for a, c in self.doc_knowledge.abbreviations.items()])
            mappings.extend([f"{s}={c}" for s, c in self.doc_knowledge.synonyms.items()])
            mappings.extend([f"{p}={c}" for p, c in self.doc_knowledge.partial_references.items()])

        # Pass 1
        print("    Phase 5 Pass 1:")
        pass1 = self._run_single_extraction_pass(
            sentences, comp_names, comp_lower, mappings, name_to_id, sent_map, pass_label="[P1] ")

        # Pass 2
        print("    Phase 5 Pass 2:")
        pass2 = self._run_single_extraction_pass(
            sentences, comp_names, comp_lower, mappings, name_to_id, sent_map, pass_label="[P2] ")

        # Intersection: keep only candidates found in BOTH passes
        intersected = {key: pass1[key] for key in pass1 if key in pass2}

        print(f"    Phase 5 intersection: Pass1={len(pass1)}, Pass2={len(pass2)}, "
              f"Intersect={len(intersected)} (dropped {len(pass1) + len(pass2) - 2*len(intersected)} unique-to-one-pass)")

        return list(intersected.values())

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
                if snum is None:
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
        """Code-first auto-approval + LLM generic detection + 2-pass intersect + evidence."""
        if not candidates:
            return []

        comp_names = self._get_comp_names(components)
        needs = [c for c in candidates if c.needs_validation]
        direct = [c for c in candidates if not c.needs_validation]

        if not needs:
            return candidates

        # Pre-check: LLM-based contextual generic mention detection
        # Group candidates by component for batch LLM calls
        generic_candidates = {}  # comp_name -> [candidate]
        non_generic = []
        for c in needs:
            sent = sent_map.get(c.sentence_number)
            if not sent:
                non_generic.append(c)
                continue
            # Only check single-word ambiguous names that appear in lowercase
            comp_lower = c.component_name.lower()
            has_exact_case = self._has_standalone_mention(c.component_name, sent.text)
            has_lowercase = (not has_exact_case and
                             re.search(rf'\b{re.escape(comp_lower)}\b', sent.text))
            # Also check partial references that appear in lowercase
            if not has_lowercase and self.doc_knowledge:
                for partial, target in self.doc_knowledge.partial_references.items():
                    if target == c.component_name:
                        partial_lower = partial.lower()
                        if (re.search(rf'\b{re.escape(partial_lower)}\b', sent.text.lower())
                                and not re.search(rf'\b{re.escape(partial)}\b', sent.text)):
                            has_lowercase = True
                            break
            if has_lowercase and self._is_ambiguous_name_component(c.component_name):
                generic_candidates.setdefault(c.component_name, []).append(c)
            else:
                non_generic.append(c)

        # For each ambiguous component with lowercase-only mentions, ask LLM
        remaining = list(non_generic)
        for comp_name, cands in generic_candidates.items():
            # Find anchor sentences where full name appears (calibration)
            anchor_lines = []
            for s in sent_map.values():
                if self._has_standalone_mention(comp_name, s.text):
                    anchor_lines.append(f"  S{s.number}: {s.text}")
                    if len(anchor_lines) >= 5:
                        break

            # Build cases
            case_lines = []
            for i, c in enumerate(cands):
                s = sent_map.get(c.sentence_number)
                prev = sent_map.get(c.sentence_number - 1)
                prev_text = f" [prev: {prev.text[:60]}]" if prev else ""
                case_lines.append(f"  Case {i+1} (S{c.sentence_number}): {s.text}{prev_text}")

            anchor_section = ""
            if anchor_lines:
                anchor_section = (
                    f'FULL-NAME REFERENCES (these definitely refer to the {comp_name} component):\n'
                    + '\n'.join(anchor_lines) + '\n\n'
                )

            prompt = f"""CONTEXTUAL WORD USAGE: Does the word refer to the architecture component "{comp_name}", or is it used as an ordinary English word?

{anchor_section}SENTENCES TO CHECK (the component name appears only in lowercase or as part of a compound phrase):
{chr(10).join(case_lines)}

For each case, determine:
- COMPONENT: The word refers to the specific "{comp_name}" component as a system entity
  (e.g., "the {comp_name.lower()} handles requests" = component reference)
- GENERIC: The word is used as ordinary English describing a general concept, activity, or modifier
  (e.g., "provides {comp_name.lower()} access" or "{comp_name.lower()} operations" = generic usage)

Key distinction: A component reference names a specific system entity as a participant.
A generic use describes a type of activity or quality that happens to share the word.

Return JSON:
{{"results": [{{"case": 1, "usage": "component" or "generic", "reason": "brief"}}]}}
JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
            if not data:
                remaining.extend(cands)  # On failure, keep all (safe default)
                continue

            results_map = {}
            for r in data.get("results", []):
                idx = r.get("case", 0) - 1
                results_map[idx] = r.get("usage", "component")

            for i, c in enumerate(cands):
                usage = results_map.get(i, "component")
                if usage == "generic":
                    print(f"    LLM generic reject: S{c.sentence_number} -> {c.component_name} "
                          f"({data.get('results', [{}])[i].get('reason', '') if i < len(data.get('results', [])) else ''})")
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

{VALIDATION_RULES}

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

    def _coref_cases_in_context(self, sentences, components, name_to_id, sent_map):
        """Unified coreference: per-case presentation with ±5 bidirectional context.

        Cross-model Pareto winner (0 FP on both Claude and GPT-5.2).
        No complexity gate needed.
        """
        comp_names = self._get_comp_names(components)
        all_coref = []
        pronoun_sents = [s for s in sentences if self.PRONOUN_PATTERN.search(s.text)]

        ctx_info = []
        if self.learned_patterns and self.learned_patterns.subprocess_terms:
            ctx_info.append(f"Subprocesses (don't link): {', '.join(list(self.learned_patterns.subprocess_terms)[:5])}")

        for batch_start in range(0, len(pronoun_sents), 10):
            batch = pronoun_sents[batch_start:batch_start + 10]
            cases = []
            for sent in batch:
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

            prompt += f"""{COREF_RULES}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "pronoun": "it", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

Only include resolutions you are CERTAIN about. JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=150))
            if not data:
                continue

            for res in data.get("resolutions", []):
                comp = res.get("component")
                snum = res.get("sentence")
                if snum is None or not comp or comp not in name_to_id:
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

                if ant_snum is None:
                    print(f"    Coref skip (no antecedent): S{snum} -> {comp}")
                    continue

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
                        try:
                            vid = int(vid)
                        except (ValueError, TypeError):
                            continue
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
            for j in range(len(text) - 1, start, -1):
                if text[j] == ']':
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
        d = os.path.join(cache_dir, "s_linker3", ds)
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
        path = os.path.join(log_dir, f"s_linker3_{ds}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w") as f:
            json.dump(self._phase_log, f, indent=2, default=str)
        print(f"  Phase log saved: {path}")
