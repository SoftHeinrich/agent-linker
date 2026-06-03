"""AgentLinker V27g: GoT sub-judges with source-aware branch weighting.

Changes from V26d:
- Phase 9 sub-judges use Graph-of-Thoughts: instead of same-prompt 2-pass voting,
  each sub-judge uses 3 different-perspective branches (Reference Quality, Sentence Topic,
  Component Fit) that evaluate independently, then aggregate with source-aware thresholds.
- Source-aware aggregation: transarc=union(any branch), coref=intersection(all branches),
  partial=majority(2/3), other=majority(2/3).
- Error decorrelation: different prompts have independent failure modes.
"""

import json
import os
import re
import time
from collections import defaultdict
from typing import Optional

from ...core import (
    SadSamLink,
    CandidateLink,
    DocumentProfile,
    LearnedThresholds,
    ModelKnowledge,
    DocumentKnowledge,
    LearnedPatterns,
    EntityMention,
    DiscourseContext,
    DocumentLoader,
    Sentence,
)
from ...pcm_parser import parse_pcm_repository
from ...llm_client import LLMClient, LLMBackend
from .agent_linker import AgentLinker


class AgentLinkerV27g(AgentLinker):
    """V27g: GoT sub-judges with source-aware branch weighting."""

    SOURCE_PRIORITY = {
        "transarc": 5, "validated": 4, "entity": 3,
        "coreference": 2, "partial_inject": 1, "recovered": 0,
    }

    # REMOVED: GENERIC_PARTIALS was hardcoded {"conversion","data","process",...}
    # Now discovered at runtime in Phase 1 and stored in self._discovered_generic_partials
    GENERIC_PARTIALS = set()  # populated dynamically in _analyze_model

    NON_MODIFIERS = {
        "the", "a", "an", "this", "that", "these", "those", "its", "their",
        "our", "your", "my", "his", "her", "some", "any", "all", "each",
        "every", "no", "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did", "will", "would", "can",
        "could", "shall", "should", "may", "might", "must",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "and", "or", "but", "not", "if", "then", "than", "as",
        "about", "into", "through", "during", "before", "after",
        "above", "below", "between", "under", "over",
    }

    def __init__(self, backend: Optional[LLMBackend] = None):
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        super().__init__(backend=backend or LLMBackend.CLAUDE)
        self._phase_log = []
        self._is_complex = None
        self._discovered_generic_partials = set()
        print(f"AgentLinkerV27g: GoT sub-judges + source-aware weighting")
        print(f"  Backend: {self.llm.backend.value}, Model: {os.environ.get('CLAUDE_MODEL', 'default')}")

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
        path = os.path.join(log_dir, f"v27g_{ds}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w") as f:
            json.dump(self._phase_log, f, indent=2, default=str)
        print(f"  Phase log saved: {path}")

    # ── Phase checkpoint save/load ────────────────────────────────────

    def _checkpoint_dir(self, text_path):
        cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
        ds = os.path.splitext(os.path.basename(text_path))[0]
        d = os.path.join(cache_dir, ds)
        os.makedirs(d, exist_ok=True)
        return d

    def _save_checkpoint(self, text_path, phase, state):
        """Save intermediate phase output as JSON checkpoint."""
        d = self._checkpoint_dir(text_path)
        path = os.path.join(d, f"phase_{phase}.json")
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=self._serialize)
        return path

    def _load_checkpoint(self, text_path, phase):
        """Load a phase checkpoint if it exists. Returns None if not found."""
        d = self._checkpoint_dir(text_path)
        path = os.path.join(d, f"phase_{phase}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _serialize(obj):
        if isinstance(obj, set):
            return sorted(obj)
        if hasattr(obj, '__dataclass_fields__'):
            from dataclasses import asdict
            return asdict(obj)
        return str(obj)

    def _links_to_json(self, links):
        return [{"s": l.sentence_number, "cid": l.component_id,
                 "c": l.component_name, "conf": l.confidence, "src": l.source}
                for l in links]

    def _links_from_json(self, data):
        return [SadSamLink(d["s"], d["cid"], d["c"], d.get("conf", 1.0), d["src"])
                for d in data]

    def _candidates_to_json(self, candidates):
        return [{"s": c.sentence_number, "st": c.sentence_text, "c": c.component_name,
                 "cid": c.component_id, "mt": c.matched_text, "conf": c.confidence,
                 "src": c.source, "mtype": c.match_type, "nv": c.needs_validation,
                 "ctx": c.context_sentences}
                for c in candidates]

    def _candidates_from_json(self, data):
        return [CandidateLink(d["s"], d["st"], d["c"], d["cid"], d["mt"],
                               d.get("conf", 0.85), d["src"], d["mtype"],
                               d.get("nv", True), d.get("ctx", []))
                for d in data]

    def _resume_from(self, cp, phase, text_path, components, sentences, sent_map,
                     name_to_id, id_to_name, t0):
        """Resume pipeline from a saved checkpoint."""
        # Restore state from checkpoint
        self.model_knowledge = ModelKnowledge(
            impl_indicators=cp["model_knowledge"].get("impl_indicators", []),
            impl_to_abstract=cp["model_knowledge"].get("impl_to_abstract", {}),
            architectural_names=set(cp["model_knowledge"].get("architectural_names", [])),
            ambiguous_names=set(cp["model_knowledge"].get("ambiguous_names", [])),
            shared_vocabulary=cp["model_knowledge"].get("shared_vocabulary", {}),
        )
        self.doc_knowledge = DocumentKnowledge(
            abbreviations=cp["doc_knowledge"].get("abbreviations", {}),
            synonyms=cp["doc_knowledge"].get("synonyms", {}),
            partial_references=cp["doc_knowledge"].get("partial_references", {}),
            generic_terms=set(cp["doc_knowledge"].get("generic_terms", [])),
        )
        self._is_complex = cp.get("is_complex", False)
        self.GENERIC_COMPONENT_WORDS = set(cp.get("generic_component_words", []))
        self.GENERIC_PARTIALS = set(cp.get("generic_partials", []))
        self.thresholds = LearnedThresholds(0, 0, 0, 0, "qualitative", 0)
        self.doc_profile = DocumentProfile(len(sentences), len(components), 0, 0, 0, 0, "balanced")

        preliminary = self._links_from_json(cp["preliminary"])
        transarc_set = {(s, c) for s, c in cp["transarc_set"]}
        print(f"  Restored: {len(preliminary)} links, {len(transarc_set)} transarc, "
              f"complex={self._is_complex}")

        if phase == 9:
            # Run Phase 9 judge review
            print("\n[Phase 9] Judge Review (TransArc immune) — RESUMED")
            reviewed = self._judge_review(preliminary, sentences, components, sent_map, transarc_set)
            rejected = [l for l in preliminary if (l.sentence_number, l.component_id)
                        not in {(r.sentence_number, r.component_id) for r in reviewed}]
            print(f"  Approved: {len(reviewed)} (rejected {len(rejected)})")
            self._log("phase_9", {"input": len(preliminary)},
                      {"approved": len(reviewed), "rejected": len(rejected)}, reviewed)

            print("\n[Phase 10] FN Recovery — SKIPPED (dead weight)")
            final = reviewed
        else:
            print(f"  ** Resume from phase {phase} not yet supported, running from phase 9 **")
            final = preliminary

        self._save_log(text_path)
        print(f"\nFinal: {len(final)} links")
        return [SadSamLink(l.sentence_number, l.component_id, l.component_name,
                           l.confidence, l.source) for l in final]

    # ═════════════════════════════════════════════════════════════════════
    # Per-mention generic check (from V16)
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

    def _needs_antecedent_check(self, comp_name):
        if ' ' in comp_name or '-' in comp_name:
            return False
        if re.search(r'[a-z][A-Z]', comp_name):
            return False
        if comp_name.isupper():
            return False
        if comp_name[0].islower():
            return False
        return True

    # ═════════════════════════════════════════════════════════════════════
    # Main pipeline
    # ═════════════════════════════════════════════════════════════════════

    def link(self, text_path, model_path, transarc_csv=None, resume_from_phase=None):
        self._phase_log = []
        t0 = time.time()

        components = parse_pcm_repository(model_path)
        sentences = DocumentLoader.load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        id_to_name = {c.id: c.name for c in components}
        sent_map = DocumentLoader.build_sent_map(sentences)
        self._cached_sent_map = sent_map

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        # ── Resume from checkpoint? ───────────────────────────────────
        if resume_from_phase is not None:
            cp = self._load_checkpoint(text_path, f"pre{resume_from_phase}")
            if cp:
                print(f"\n  ** Resuming from phase {resume_from_phase} checkpoint **")
                return self._resume_from(cp, resume_from_phase, text_path, components,
                                          sentences, sent_map, name_to_id, id_to_name, t0)
            else:
                print(f"\n  ** No checkpoint for phase {resume_from_phase}, running full pipeline **")

        # ── Phase 0: Document profile ──────────────────────────────────
        print("\n[Phase 0] Document Profile")
        self.doc_profile = self._learn_document_profile(sentences, components)
        self._is_complex = self._structural_complexity(sentences, components)
        spc = len(sentences) / max(1, len(components))
        print(f"  Stats: {spc:.1f} sents/comp, {self.doc_profile.pronoun_ratio:.0%} pronouns")
        print(f"  Complex: {self._is_complex}")
        self.thresholds = LearnedThresholds(0, 0, 0, 0, "qualitative", 0)
        self._log("phase_0", {"sents": len(sentences), "comps": len(components)},
                  {"spc": spc, "complex": self._is_complex})

        # ── Phase 1: Model Structure ───────────────────────────────────
        print("\n[Phase 1] Model Structure")
        self.model_knowledge = self._analyze_model(components)
        arch = self.model_knowledge.architectural_names
        ambig = self.model_knowledge.ambiguous_names

        # V24: Populate GENERIC_COMPONENT_WORDS from LLM ambiguity detection
        # Exempt all-uppercase names (DB, UI, API) — these are abbreviations,
        # not generic English words, and should be treated as architectural.
        self.GENERIC_COMPONENT_WORDS = set()
        for name in ambig:
            if ' ' not in name and not name.isupper():  # single-word, non-abbreviation
                self.GENERIC_COMPONENT_WORDS.add(name.lower())
        print(f"  Architectural: {len(arch)}, Ambiguous: {sorted(ambig)}")
        print(f"  Discovered generic component words: {sorted(self.GENERIC_COMPONENT_WORDS)}")

        # V24: Populate GENERIC_PARTIALS from multi-word component name fragments
        # that are common English words (discovered from the model, not hardcoded)
        self.GENERIC_PARTIALS = set()
        for comp in components:
            parts = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', comp.name)
            for part in parts:
                p_lower = part.lower()
                # Skip all-uppercase parts (abbreviations like DB, UI, API)
                if part.isupper():
                    continue
                if len(p_lower) >= 3 and any(
                    p_lower == a.lower() for a in ambig
                ):
                    self.GENERIC_PARTIALS.add(p_lower)
        # Also add any single-word ambiguous component names as generic partials
        # (but not all-uppercase abbreviations like DB, UI)
        for name in ambig:
            if ' ' not in name and not name.isupper():
                self.GENERIC_PARTIALS.add(name.lower())
        print(f"  Discovered generic partials: {sorted(self.GENERIC_PARTIALS)}")

        self._log("phase_1", {"components": [c.name for c in components]},
                  {"architectural": sorted(arch), "ambiguous": sorted(ambig),
                   "generic_words": sorted(self.GENERIC_COMPONENT_WORDS),
                   "generic_partials": sorted(self.GENERIC_PARTIALS)})

        # ── Phase 2: Pattern learning ─────────────────────────────────
        print("\n[Phase 2] Pattern Learning")
        self.learned_patterns = self._learn_patterns_with_debate(sentences, components)
        print(f"  Subprocess terms: {len(self.learned_patterns.subprocess_terms)}")
        self._log("phase_2", {}, {"subprocess": sorted(self.learned_patterns.subprocess_terms)})

        # ── Phase 3: Document knowledge (enriched prompt) ──────────────
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

        # ── Phase 3b: Multi-word partial enrichment
        self._enrich_multiword_partials(sentences, components)

        # ── Phase 4: TransArc baseline ────────────────────────────────
        print("\n[Phase 4] TransArc")
        transarc_links = self._process_transarc(transarc_csv, id_to_name, sent_map, name_to_id)
        transarc_set = {(l.sentence_number, l.component_id) for l in transarc_links}
        print(f"  Links: {len(transarc_links)}")
        self._log("phase_4", {"csv": transarc_csv}, {"count": len(transarc_links)}, transarc_links)

        # ── Phase 5: Entity extraction (enriched prompt) ──────────────
        print("\n[Phase 5] Entity Extraction")
        candidates = self._extract_entities_enriched(sentences, components, name_to_id, sent_map)
        print(f"  Candidates: {len(candidates)}")

        # Abbreviation guard
        before_guard = len(candidates)
        candidates = self._apply_abbreviation_guard_to_candidates(candidates, sent_map)
        if len(candidates) < before_guard:
            print(f"  After abbrev guard: {len(candidates)} (-{before_guard - len(candidates)})")
        self._log("phase_5", {}, {"count": len(candidates)})

        # ── Phase 5b: Targeted recovery for unlinked components ───────
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

        # ── Phase 6: Validation (INTERSECT) ───────────────────────────
        print("\n[Phase 6] Validation")
        validated = self._validate_intersect(candidates, components, sent_map)
        print(f"  Validated: {len(validated)} (of {len(candidates)})")
        self._log("phase_6", {"candidates": len(candidates)}, {"validated": len(validated)})

        # ── Phase 7: Coreference ──────────────────────────────────────
        print("\n[Phase 7] Coreference")
        if self._is_complex:
            print(f"  Mode: debate (complex, {len(sentences)} sents)")
            coref_links = self._coref_debate(sentences, components, name_to_id, sent_map)
        else:
            discourse_model = self._build_discourse_model(sentences, components, name_to_id)
            print(f"  Mode: discourse ({len(sentences)} sents)")
            coref_links = self._coref_discourse(sentences, components, name_to_id, sent_map, discourse_model)

        before_coref = len(coref_links)
        coref_links = self._filter_generic_coref(coref_links, sent_map)
        if len(coref_links) < before_coref:
            print(f"  After generic filter: {len(coref_links)} (-{before_coref - len(coref_links)})")

        # Deterministic pronoun resolution (no LLM, eliminates variance)
        coref_set = {(l.sentence_number, l.component_id) for l in coref_links}
        pronoun_links = self._deterministic_pronoun_coref(
            sentences, components, name_to_id, sent_map,
            transarc_set | {(c.sentence_number, c.component_id) for c in validated} | coref_set)
        if pronoun_links:
            coref_links.extend(pronoun_links)
            print(f"  Deterministic pronoun coref: +{len(pronoun_links)}")

        print(f"  Coref links: {len(coref_links)}")
        self._log("phase_7", {"method": "debate" if self._is_complex else "discourse"},
                  {"count": len(coref_links)}, coref_links)

        # ── Phase 8: Implicit References — SKIPPED ───────────────────
        reason = "complex doc" if self._is_complex else "dead weight"
        print(f"\n[Phase 8] Implicit References — SKIPPED ({reason})")
        implicit_links = []

        # ── Phase 8b: Partial-reference injection ─────────────────────
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

        # ── Combine + deduplicate ─────────────────────────────────────
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

        # ── Parent-overlap guard (V25): drop child if parent linked to same sentence
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

        # ── Phase 8c: Boundary filters (NON-TRANSARC ONLY) ───────────
        print("\n[Phase 8c] Boundary Filters (non-TransArc only)")
        preliminary, boundary_rejected = self._apply_boundary_filters(
            preliminary, sent_map, transarc_set
        )
        if boundary_rejected:
            print(f"  Rejected: {len(boundary_rejected)}")
            self._log("phase_8c", {},
                      {"rejected": len(boundary_rejected),
                       "details": [(lk.component_name, reason) for lk, reason in boundary_rejected]})

        # ── Save pre-Phase 9 checkpoint ───────────────────────────────
        self._save_checkpoint(text_path, "pre9", {
            "preliminary": self._links_to_json(preliminary),
            "transarc_set": [[s, c] for s, c in transarc_set],
            "model_knowledge": {
                "architectural_names": sorted(self.model_knowledge.architectural_names),
                "ambiguous_names": sorted(self.model_knowledge.ambiguous_names),
                "impl_indicators": self.model_knowledge.impl_indicators,
                "impl_to_abstract": self.model_knowledge.impl_to_abstract,
                "shared_vocabulary": self.model_knowledge.shared_vocabulary,
            },
            "doc_knowledge": {
                "abbreviations": self.doc_knowledge.abbreviations if self.doc_knowledge else {},
                "synonyms": self.doc_knowledge.synonyms if self.doc_knowledge else {},
                "partial_references": self.doc_knowledge.partial_references if self.doc_knowledge else {},
                "generic_terms": sorted(self.doc_knowledge.generic_terms) if self.doc_knowledge else [],
            },
            "is_complex": self._is_complex,
            "generic_component_words": sorted(self.GENERIC_COMPONENT_WORDS),
            "generic_partials": sorted(self.GENERIC_PARTIALS),
        })
        print(f"  Checkpoint saved: pre-Phase 9 ({len(preliminary)} links)")

        # ── Phase 9: Judge Review (TransArc immune, adaptive ctx, show match)
        print("\n[Phase 9] Judge Review (TransArc immune)")
        reviewed = self._judge_review(preliminary, sentences, components, sent_map, transarc_set)
        rejected = [l for l in preliminary if (l.sentence_number, l.component_id)
                    not in {(r.sentence_number, r.component_id) for r in reviewed}]
        print(f"  Approved: {len(reviewed)} (rejected {len(rejected)})")
        self._log("phase_9", {"input": len(preliminary)},
                  {"approved": len(reviewed), "rejected": len(rejected)}, reviewed)
        if rejected:
            self._log("phase_9_rejected", {}, {"count": len(rejected)}, rejected)

        # ── Phase 10: FN Recovery — SKIPPED ──────────────────────────
        print("\n[Phase 10] FN Recovery — SKIPPED (dead weight)")
        final = reviewed

        # ── Save log ──────────────────────────────────────────────────
        self._log("summary", {"total_time_s": round(time.time() - t0, 1)},
                  {"final": len(final)}, final)
        self._save_log(text_path)

        print(f"\nFinal: {len(final)} links")
        return final

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
        """V24: Dynamic threshold based on document characteristics, not hardcoded spc_min."""
        comp_names = [c.name for c in components]
        mention_count = sum(1 for sent in sentences
                          if any(cn.lower() in sent.text.lower() for cn in comp_names))
        uncovered_ratio = 1.0 - (mention_count / max(1, len(sentences)))
        spc = len(sentences) / max(1, len(components))
        # Dynamic: complex if >50% uncovered AND more sentences than components can cover
        # (i.e., many sentences have no direct component mention)
        result = uncovered_ratio > 0.5 and spc > 4
        print(f"  Structural complexity: uncovered={uncovered_ratio:.1%}, spc={spc:.1f} -> {result}")
        return result

    # ═════════════════════════════════════════════════════════════════════
    # Phase 1: Model Structure Analysis (V24 override with improved prompt)
    # ═════════════════════════════════════════════════════════════════════

    def _analyze_model(self, components):
        """Analyze model structure with improved prompt using rules instead of examples."""
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

        word_to_comps = {}
        for name in names:
            for word in re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', name):
                w = word.lower()
                if len(w) >= 3:
                    word_to_comps.setdefault(w, []).append(name)
        knowledge.shared_vocabulary = {w: list(set(c)) for w, c in word_to_comps.items() if len(set(c)) > 1}

        prompt = f"""Classify these component names into two categories.

NAMES: {', '.join(names)}

RULES:
- "architectural": Names that are INVENTED proper names — they would not appear in a dictionary.
  Multi-word compounds, CamelCase names, and domain-specific terms are architectural.
- "ambiguous": Names that are ALSO common English words you would find in a dictionary.
  A single word that has everyday meaning beyond this system belongs here.
  Example: In a compiler, "Optimizer" could be the component name OR the general concept of optimization.
  In an OS kernel, "Scheduler" is architectural, but "Monitor" could be the component or the generic concept.

Return JSON:
{{
  "architectural": ["names that are clearly invented/technical"],
  "ambiguous": ["names that double as ordinary English words"]
}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=100))
        if data:
            knowledge.architectural_names = set(data.get("architectural", [])) & set(names)
            knowledge.ambiguous_names = set(data.get("ambiguous", [])) & set(names)

        return knowledge

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
    # ENRICHED Phase 3: Document knowledge with examples
    # ═════════════════════════════════════════════════════════════════════

    def _learn_document_knowledge_enriched(self, sentences, components):
        comp_names = self._get_comp_names(components)
        doc_lines = [f"S{s.number}: {s.text}" for s in sentences[:100]]

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
   REJECT: Common words that have ordinary English meanings beyond the component

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

Apply these rules:

REJECT if ANY of these are true:
- The term is used in its ordinary English sense, NOT as a name for the component
  (e.g., "the scheduler runs every minute" uses "scheduler" as a generic concept, not as a named component)
- The term refers to a different component or to the system as a whole
- The mapping cannot be verified from the actual document text

APPROVE if ALL of these are true:
- The term is used AS A NAME for the component in context (even if the word exists in a dictionary)
  (e.g., "The Dispatcher routes incoming requests" uses "Dispatcher" as a proper name for a specific component)
- The term appears in the document in a context that makes the reference clear
- The term unambiguously identifies exactly one component

Return JSON:
{{
  "approved": ["term1", "term2"],
  "generic_rejected": ["generic_term1"]
}}
JSON only:"""

            data2 = self.llm.extract_json(self.llm.query(prompt2, timeout=120))
            approved = set(data2.get("approved", [])) if data2 else set(all_mappings.keys())
            generic_terms = set(data2.get("generic_rejected", [])) if data2 else set()

            # Fix A: CamelCase override — CamelCase terms are constructed proper names, not generic
            for term in list(generic_terms):
                if re.search(r'[a-z][A-Z]', term):
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    CamelCase override (rescued): {term}")

            # Fix B: Uppercase override — short all-uppercase terms are acronyms, not generic
            for term in list(generic_terms):
                if term.isupper() and len(term) <= 4 and term in all_mappings:
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    Uppercase override (rescued): {term}")

            # V24: Deterministic overrides for LLM-rejected terms.
            # The LLM's generic_rejected judgment is trusted for truly generic words,
            # but overridden when document evidence shows proper-name usage.
            for term in list(generic_terms):
                if term not in all_mappings:
                    continue
                # Exact component name match always wins (case-insensitive)
                if any(term.lower() == cn.lower() for cn in comp_names):
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    Exact-component override (rescued): {term}")
                # Capitalized multi-word terms are likely proper names, not generic
                elif term[0].isupper() and ' ' in term:
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    Multi-word proper-name override (rescued): {term}")

            # Fix C: Capitalized synonym-to-component override.
            # A capitalized word that maps to a real component and appears capitalized
            # mid-sentence in the document is being used as a proper name.
            for term in list(generic_terms):
                if term not in all_mappings:
                    continue
                _, target_comp = all_mappings[term]
                if ' ' in term or not term[0].isupper() or target_comp not in comp_names:
                    continue
                # Check if term appears capitalized not only at sentence start
                for s in sentences[:100]:
                    # Find the term mid-sentence (not at position 0)
                    for m in re.finditer(rf'\b{re.escape(term)}\b', s.text):
                        if m.start() > 0:
                            generic_terms.discard(term)
                            approved.add(term)
                            print(f"    Capitalized-synonym override (rescued): {term}")
                            break
                    if term in approved:
                        break
        else:
            approved = set()
            generic_terms = set()

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

        # Deterministic CamelCase-split synonym injection
        # e.g., "AbcDef" → "Abc Def" (space-separated form of CamelCase name)
        for comp in [c.name for c in components]:
            split = re.sub(r'([a-z])([A-Z])', r'\1 \2', comp)
            split = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', split)
            if split != comp and split not in knowledge.synonyms:
                knowledge.synonyms[split] = comp
                print(f"    CamelCase syn: {split} -> {comp}")

        return knowledge

    # ═════════════════════════════════════════════════════════════════════
    # ENRICHED Phase 5: Entity extraction
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
2. A space-separated form matches a compound name (e.g., "Load Balancer" → LoadBalancer)
3. The sentence describes what a specific component does by name or role
4. A known synonym or partial reference is used

RULES — exclude when:
1. The name appears only inside a dotted path (e.g., com.example.name)
2. The name is used as an ordinary English word, not as a component reference

Favor inclusion over exclusion — later validation will filter borderline cases.

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N, "component": "Name", "matched_text": "text found in sentence", "match_type": "exact|synonym|partial|functional"}}]}}
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
                sent = sent_map.get(snum)
                if not sent or self._in_dotted_path(sent.text, cname):
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
    # Phase 5b: Targeted extraction for unlinked components
    # ═════════════════════════════════════════════════════════════════════

    def _targeted_extraction(self, unlinked_components, sentences, name_to_id, sent_map,
                              components=None, transarc_links=None, entity_candidates=None):
        if not unlinked_components:
            return []

        # Build parent-overlap data (Fix C from V20c)
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

DOCUMENT:
{doc_text}

Return JSON:
{{"references": [{{"sentence": N, "matched_text": "text found", "reason": "why this refers to {comp.name}"}}]}}

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
                if not sent or self._in_dotted_path(sent.text, comp.name):
                    continue
                cid = name_to_id.get(comp.name)
                if not cid:
                    continue

                # Parent-overlap guard: skip if parent component already linked here
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

    # ═════════════════════════════════════════════════════════════════════
    # Phase 6: Validation — Code-first + LLM-fallback
    # ═════════════════════════════════════════════════════════════════════

    # REMOVED: GENERIC_COMPONENT_WORDS was hardcoded {"logic","storage","common",...}
    # Now discovered dynamically: single-word component names that are common English
    # words are detected in Phase 1 via LLM ambiguity analysis.
    GENERIC_COMPONENT_WORDS = set()  # populated dynamically in _analyze_model

    def _word_boundary_match(self, name, text):
        """Check if name appears as standalone word in text (word-boundary match)."""
        return bool(re.search(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE))

    def _validate_intersect(self, candidates, components, sent_map):
        """Phase 6: Code-first v2 + 2-pass intersect + evidence post-filter for generic names.

        Step 1: Word-boundary code-first — auto-approve when name/alias appears in text (deterministic).
        Step 2: 2-pass intersect for remaining candidates.
        Step 3: For generic-risk candidates that passed 2-pass, require evidence-citation confirmation.
        """
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
                    # Substring match for longer names
                    if a.lower() in sent.text.lower():
                        matched = True
                        break
                elif len(a) >= 2:
                    # Word-boundary match for short names (UI, DB)
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

    def _coref_discourse(self, sentences, components, name_to_id, sent_map, discourse_model):
        """Phase 7 discourse mode with required antecedent citation."""
        comp_names = self._get_comp_names(components)
        all_coref = []
        pronoun_sents = [s for s in sentences if self.PRONOUN_PATTERN.search(s.text)]

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
{"resolutions": [{"case": 1, "sentence": N, "pronoun": "it", "component": "Name", "antecedent_sentence": M, "antecedent_text": "exact text with component name"}]}

Only include resolutions you are CERTAIN about. JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=150))
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
{{"resolutions": [{{"sentence": N, "pronoun": "it", "component": "Name", "antecedent_sentence": M, "antecedent_text": "exact quote with component name"}}]}}

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
    # Generic coreference filter
    # ═════════════════════════════════════════════════════════════════════

    def _filter_generic_coref(self, coref_links, sent_map):
        kept = []
        for lk in coref_links:
            if not self._needs_antecedent_check(lk.component_name):
                kept.append(lk)
                continue
            found_recent = False
            for offset in range(1, 3):
                prev = sent_map.get(lk.sentence_number - offset)
                if not prev:
                    continue
                if self._has_standalone_mention(lk.component_name, prev.text):
                    found_recent = True
                    break
                # Also accept case-insensitive match for single-word components
                if ' ' not in lk.component_name:
                    if re.search(rf'\b{re.escape(lk.component_name)}\b', prev.text, re.IGNORECASE):
                        found_recent = True
                        break
                # Also accept alias mentions (synonyms/partials from doc_knowledge)
                if self._has_alias_mention(lk.component_name, prev.text):
                    found_recent = True
                    break
            if found_recent:
                kept.append(lk)
            else:
                print(f"    Coref antecedent missing: S{lk.sentence_number} -> {lk.component_name}")
        return kept

    def _deterministic_pronoun_coref(self, sentences, components, name_to_id, sent_map, existing):
        """Resolve obvious pronoun-continuation patterns without LLM.

        If a sentence starts with "It " or "As such, it " and the preceding
        1-3 sentences mention exactly one component, create a coreference link.
        This eliminates LLM variance for the most common pronoun pattern.
        """
        comp_names = self._get_comp_names(components)
        results = []
        for sent in sentences:
            # Match sentences starting with pronoun continuation
            if not re.match(r'^(It|As such, it)\b', sent.text):
                continue
            # Look back up to 3 sentences for unambiguous component mention
            resolved = None
            for offset in range(1, 4):
                prev = sent_map.get(sent.number - offset)
                if not prev:
                    continue
                mentioned = set()
                for cn in comp_names:
                    if self._has_standalone_mention(cn, prev.text):
                        mentioned.add(cn)
                    elif ' ' not in cn and re.search(rf'\b{re.escape(cn)}\b', prev.text, re.IGNORECASE):
                        mentioned.add(cn)
                    elif self._has_alias_mention(cn, prev.text):
                        mentioned.add(cn)
                if len(mentioned) == 1:
                    resolved = mentioned.pop()
                    break
                elif len(mentioned) > 1:
                    break  # Ambiguous, skip this sentence
            if resolved and (sent.number, name_to_id[resolved]) not in existing:
                results.append(SadSamLink(
                    sent.number, name_to_id[resolved], resolved, 1.0, "coreference"))
                print(f"    Pronoun coref: S{sent.number} '{sent.text[:40]}...' -> {resolved}")
        return results

    # ═════════════════════════════════════════════════════════════════════
    # Partial reference injection
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

    # _partial_is_compound_noun removed — Phase 9 judge handles partial_inject filtering

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
    # Boundary filters (NON-TRANSARC ONLY)
    # ═════════════════════════════════════════════════════════════════════

    def _is_in_package_path(self, comp_name, sentence_text):
        words = comp_name.lower().split()
        text_lower = sentence_text.lower()
        for word in words:
            if len(word) < 3:
                continue
            dot_pattern = rf'[\w]+\.{re.escape(word)}|{re.escape(word)}\.[\w]+'
            if re.search(dot_pattern, text_lower):
                cleaned = re.sub(r'[\w]+\.[\w]+(?:\.[\w]+)*', '', text_lower)
                if re.search(rf'\b{re.escape(word)}\b', cleaned):
                    continue
                return True
        return False

    def _is_generic_word_usage(self, comp_name, sentence_text):
        words = comp_name.split()
        if len(words) != 1:
            return False
        word = words[0]
        if re.search(rf'\b{re.escape(word)}\b', sentence_text):
            return False
        pattern = rf'\b(\w+)\s+{re.escape(word.lower())}\b'
        for m in re.finditer(pattern, sentence_text.lower()):
            preceding = m.group(1).lower()
            if preceding not in self.NON_MODIFIERS:
                return True
        return False

    def _is_weak_partial_match(self, link, sent_map):
        if link.source != "partial_inject":
            return False
        sent = sent_map.get(link.sentence_number)
        if not sent:
            return False
        if re.search(rf'\b{re.escape(link.component_name)}\b', sent.text, re.IGNORECASE):
            return False
        if not self.doc_knowledge or not self.doc_knowledge.partial_references:
            return False
        for partial, comp in self.doc_knowledge.partial_references.items():
            if comp == link.component_name:
                if partial.lower() in self.GENERIC_PARTIALS:
                    if re.search(rf'\b{re.escape(partial)}\b', sent.text, re.IGNORECASE):
                        return True
        return False

    # ── Abbreviation guard ────────────────────────────────────────────

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

    def _abbreviation_guard_for_link(self, lk, sent_map):
        if not self.doc_knowledge:
            return True
        sent = sent_map.get(lk.sentence_number)
        if not sent:
            return True
        if re.search(rf'\b{re.escape(lk.component_name)}\b', sent.text, re.IGNORECASE):
            return True
        for abbr, comp in self.doc_knowledge.abbreviations.items():
            if comp == lk.component_name:
                if abbr.lower() in sent.text.lower():
                    if not self._abbreviation_match_is_valid(abbr, comp, sent.text):
                        return False
        return True

    def _apply_boundary_filters(self, links, sent_map, transarc_set):
        kept = []
        rejected = []
        for lk in links:
            sent = sent_map.get(lk.sentence_number)
            if not sent:
                kept.append(lk)
                continue
            is_ta = (lk.sentence_number, lk.component_id) in transarc_set
            if is_ta:
                kept.append(lk)
                continue
            reason = None
            if lk.source in ("validated", "entity"):
                if self._is_in_package_path(lk.component_name, sent.text):
                    reason = "package_path"
            if not reason and lk.source in ("validated", "entity", "coreference"):
                if self._is_generic_word_usage(lk.component_name, sent.text):
                    reason = "generic_word"
            if not reason:
                if self._is_weak_partial_match(lk, sent_map):
                    reason = "weak_partial"
            if not reason:
                if not self._abbreviation_guard_for_link(lk, sent_map):
                    reason = "abbrev_guard"
            if reason:
                rejected.append((lk, reason))
                print(f"    Boundary filter [{reason}]: S{lk.sentence_number} -> {lk.component_name} ({lk.source})")
            else:
                kept.append(lk)
        return kept, rejected

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

        # Triage: safe, ta-review, and source-specific buckets for non-TransArc
        safe, ta_review = [], []
        coref_review, partial_review, other_review = [], [], []
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
                # Route by source even for no-sentence (safety: approve)
                other_review.append(l)
                continue
            if self._has_standalone_mention(l.component_name, sent.text):
                safe.append(l)
            elif l.source == "coreference":
                coref_review.append(l)
            elif l.source == "partial_inject":
                partial_review.append(l)
            else:
                other_review.append(l)

        n_review = len(coref_review) + len(partial_review) + len(other_review)
        print(f"  Triage: {len(safe)} safe ({syn_safe_count} syn-safe), {len(ta_review)} ta-review, "
              f"{len(coref_review)} coref, {len(partial_review)} partial, {len(other_review)} other")

        # ── Advocate-Prosecutor deliberation for ambiguous TransArc links ──
        ta_approved = self._deliberate_transarc(ta_review, comp_names, sent_map)

        # ── Source-specific judges ──
        coref_approved = self._judge_coreference(coref_review, comp_names, sent_map)
        partial_approved = self._judge_partial(partial_review, comp_names, sent_map)
        other_approved = self._judge_other(other_review, comp_names, sent_map)

        return safe + ta_approved + coref_approved + partial_approved + other_approved

    # ── Source-specific judges ──────────────────────────────────────

    def _judge_coreference(self, links, comp_names, sent_map):
        """GoT coreference judge: 3 different-perspective branches with intersection voting.

        Coreference links are created by pronoun/discourse resolution. Instead of running
        the same prompt twice, we run 3 prompts that each evaluate from a different angle:
        Branch 1 (Reference Quality), Branch 2 (Sentence Topic), Branch 3 (Component Fit).
        Intersection voting: approve only if ALL 3 branches approve (strictest).
        """
        if not links:
            return []

        use_full_ctx = not self._is_complex

        # Build cases — every link gets a case
        cases = []
        for i, l in enumerate(links):
            sent = sent_map.get(l.sentence_number)
            if not sent:
                cases.append((l, None))
                continue

            ctx_lines = []
            for offset in [3, 2, 1]:
                prev = sent_map.get(l.sentence_number - offset)
                if prev:
                    txt = prev.text if use_full_ctx else f"{prev.text[:60]}..."
                    ctx_lines.append(f"    S{l.sentence_number - offset}: {txt}")
            ctx_lines.append(f"    >>> S{l.sentence_number}: {sent.text}")
            case_text = (f"Case {i+1}: S{l.sentence_number} -> {l.component_name}\n" +
                         chr(10).join(ctx_lines))
            cases.append((l, case_text))

        reviewable = [(i, l, ct) for i, (l, ct) in enumerate(cases) if ct is not None]
        if not reviewable:
            return [l for l, _ in cases]

        cases_block = chr(10).join(ct for _, _, ct in reviewable)
        n = len(reviewable)

        # Branch 1: Reference Quality — is the pronoun genuinely referring to this component?
        prompt_ref = f"""REFERENCE QUALITY CHECK: Are these coreference links genuine references?

Each link was created by pronoun resolution — a pronoun ("it", "they", "its") or implicit subject
was resolved to the claimed component based on preceding context. Your job: verify the reference chain.

COMPONENTS: {', '.join(comp_names)}

For each case, determine:
- Does the target sentence (>>>) contain a pronoun or implicit subject?
- Looking at the preceding sentences, which component was MOST RECENTLY discussed?
- Does the pronoun genuinely refer back to the claimed component, or to something else entirely?

APPROVE only if the pronoun/reference clearly resolves to the claimed component.
REJECT if it more likely refers to a different component or to a non-component concept.

CASES:
{cases_block}

Return JSON: {{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

        # Branch 2: Sentence Topic — is the sentence actually about this component?
        prompt_topic = f"""SENTENCE TOPIC CHECK: What are these sentences primarily about?

Each link claims a sentence discusses a specific component via pronoun reference. Your job:
determine whether the sentence is actually about this component's functionality or behavior.

COMPONENTS: {', '.join(comp_names)}

For each case, determine:
- What is the target sentence (>>>) primarily about? What functionality or concept does it describe?
- Is the sentence continuing a discussion about the claimed component?
- Or has the topic shifted to something else (a different component, a general concept, user behavior)?

APPROVE only if the sentence is genuinely continuing a discussion about the claimed component.
REJECT if the sentence has moved on to a different topic, even if the claimed component was discussed nearby.

CASES:
{cases_block}

Return JSON: {{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

        # Branch 3: Component Fit — does what's described match the component's architectural role?
        prompt_fit = f"""COMPONENT FIT CHECK: Do these links match the component's architectural role?

Each link claims a pronoun refers to a specific architecture component. Your job: determine whether
what the sentence describes is consistent with that component's role in the system.

COMPONENTS: {', '.join(comp_names)}

For each case, determine:
- What action, property, or behavior does the target sentence (>>>) describe?
- Does this action/property/behavior fit the architectural role of the claimed component?
- Or does it describe something that belongs to a different component or layer of the system?

APPROVE only if the described behavior plausibly belongs to the claimed component.
REJECT if the described behavior is inconsistent with the component's architectural role.

CASES:
{cases_block}

Return JSON: {{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

        # Run all 3 branches
        data1 = self.llm.extract_json(self.llm.query(prompt_ref, timeout=120))
        data2 = self.llm.extract_json(self.llm.query(prompt_topic, timeout=120))
        data3 = self.llm.extract_json(self.llm.query(prompt_fit, timeout=120))

        # Parse approvals from each branch (safe default = approve)
        app1 = self._parse_approvals(data1, n)
        app2 = self._parse_approvals(data2, n)
        app3 = self._parse_approvals(data3, n)

        # Majority: approve if >=2 of 3 branches approve
        approved_idx = set()
        for i in range(n):
            votes = sum(1 for s in [app1, app2, app3] if i in s)
            if votes >= 2:
                approved_idx.add(i)

        result = []
        for i, (l, case_text) in enumerate(cases):
            if case_text is None:
                result.append(l)  # no sentence → auto-approve
            elif i in approved_idx:
                result.append(l)
            else:
                print(f"    GoT coref reject: S{l.sentence_number} -> {l.component_name}")
        return result

    def _judge_partial(self, links, comp_names, sent_map):
        """GoT partial-inject judge: 3 different-perspective branches with majority voting.

        Partial_inject links exist because a partial name (e.g., "Conversion") matched a
        multi-word component (e.g., "Presentation Conversion"). Instead of same-prompt 2-pass,
        we run 3 prompts from different angles and use majority voting (approve if >=2/3).
        """
        if not links:
            return []

        # Build cases — every link gets a case
        cases = []
        for i, l in enumerate(links):
            sent = sent_map.get(l.sentence_number)
            if not sent:
                cases.append((l, None))
                continue

            match = self._find_match_text(l.component_name, sent.text)
            partial_word = match if match else l.component_name
            case_text = (
                f'Case {i+1}: S{l.sentence_number} -> {l.component_name} '
                f'(partial match: "{partial_word}")\n'
                f'    >>> S{l.sentence_number}: {sent.text}')
            cases.append((l, case_text))

        reviewable = [(i, l, ct) for i, (l, ct) in enumerate(cases) if ct is not None]
        if not reviewable:
            return [l for l, _ in cases]

        cases_block = chr(10).join(ct for _, _, ct in reviewable)
        n = len(reviewable)

        # Branch 1: Reference Quality — is the partial word a genuine component reference?
        prompt_ref = f"""REFERENCE QUALITY CHECK: Are these partial-name matches genuine component references?

Each link exists because a PARTIAL word from a multi-word component name appeared in the sentence.
Your job: determine if the partial word is being used as a reference to the architecture component,
or if it is just a coincidental occurrence of a common English word.

COMPONENTS: {', '.join(comp_names)}

For each case, determine:
- Is the partial word being used as a NAME for the component (referring to it as an entity)?
- Or is it being used as an ordinary English word (e.g., "conversion" meaning any transformation,
  "client" meaning any user, "common" meaning shared/typical)?

APPROVE only if the partial word is clearly being used to refer to the named component.
REJECT if the word is used in its generic English dictionary sense.

CASES:
{cases_block}

Return JSON: {{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

        # Branch 2: Sentence Topic — is the sentence about this component?
        prompt_topic = f"""SENTENCE TOPIC CHECK: What are these sentences primarily about?

Each link claims a sentence discusses a specific component because a partial name match was found.
Your job: determine whether the sentence is actually about the claimed component.

COMPONENTS: {', '.join(comp_names)}

For each case, determine:
- What is the sentence primarily about? What entity or concept is its main subject?
- Is the sentence describing the claimed component's behavior, role, or interactions?
- Or is the sentence about something else, with the partial word appearing incidentally?

APPROVE only if the sentence is genuinely about the claimed component or its functionality.
REJECT if the sentence is about a different topic and the partial word is incidental.

CASES:
{cases_block}

Return JSON: {{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

        # Branch 3: Component Fit — does what's described match the component's role?
        prompt_fit = f"""COMPONENT FIT CHECK: Do these links match the component's architectural role?

Each link claims a sentence is about a specific component based on a partial name match.
Your job: determine whether what the sentence describes is consistent with the component's
architectural role and responsibilities in the system.

COMPONENTS: {', '.join(comp_names)}

For each case, determine:
- What action, property, or behavior does the sentence describe?
- Does this fit the architectural role of the claimed component (the partial match)?
- Or does the sentence describe something that belongs to a different part of the system?

APPROVE only if the described functionality plausibly belongs to the claimed component.
REJECT if the described behavior is unrelated to the component's architectural responsibilities.

CASES:
{cases_block}

Return JSON: {{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

        # Run all 3 branches
        data1 = self.llm.extract_json(self.llm.query(prompt_ref, timeout=120))
        data2 = self.llm.extract_json(self.llm.query(prompt_topic, timeout=120))
        data3 = self.llm.extract_json(self.llm.query(prompt_fit, timeout=120))

        # Parse approvals from each branch (safe default = approve)
        app1 = self._parse_approvals(data1, n)
        app2 = self._parse_approvals(data2, n)
        app3 = self._parse_approvals(data3, n)

        # Majority: approve if >=2 of 3 branches approve
        approved_idx = set()
        for i in range(n):
            votes = sum(1 for s in [app1, app2, app3] if i in s)
            if votes >= 2:
                approved_idx.add(i)

        result = []
        for i, (l, case_text) in enumerate(cases):
            if case_text is None:
                result.append(l)
            elif i in approved_idx:
                result.append(l)
            else:
                print(f"    GoT partial reject: S{l.sentence_number} -> {l.component_name}")
        return result

    def _judge_other(self, links, comp_names, sent_map):
        """GoT judge for validated/entity/other links: 3 different-perspective branches with majority voting."""
        if not links:
            return []

        cases_text = self._build_judge_cases(links, sent_map)
        n = min(30, len(links))
        cases_block = chr(10).join(cases_text)

        # Branch 1: Reference Quality — is this a genuine reference, not coincidental overlap?
        prompt_ref = f"""REFERENCE QUALITY CHECK: Are these trace links genuine component references?

Each link claims a sentence references a specific architecture component. Your job: determine
whether the reference is genuine (the sentence actually mentions or discusses the component)
or coincidental (a word happens to match but is used in its ordinary English sense).

COMPONENTS: {', '.join(comp_names)}

For each case, determine:
- Is the matched text being used as a NAME for the architecture component?
- Or is the word used in its generic English sense (e.g., "pool" as a collection, "store" as a verb)?
- For pronoun/context links (match:NONE): does the context clearly establish a reference chain?

APPROVE only if the sentence genuinely references the claimed component.
REJECT if the match is coincidental string overlap or generic word usage.

LINKS:
{cases_block}

Return JSON: {{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

        # Branch 2: Sentence Topic — is the sentence about this component?
        prompt_topic = f"""SENTENCE TOPIC CHECK: What are these sentences primarily about?

Each link claims a sentence is about a specific architecture component. Your job: determine
whether the component is what the sentence is primarily discussing, or just a passing mention.

COMPONENTS: {', '.join(comp_names)}

For each case, determine:
- What is the sentence (>>>) primarily about? What is its main subject?
- Is the claimed component the main topic, or is it mentioned only in passing?
- For context-based links: is the sentence continuing a discussion about this component?

APPROVE only if the component is the primary topic or a significant subject of the sentence.
REJECT if the component is only mentioned in passing while the sentence discusses something else.

LINKS:
{cases_block}

Return JSON: {{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

        # Branch 3: Component Fit — does what's described match the component's architectural role?
        prompt_fit = f"""COMPONENT FIT CHECK: Do these links match the component's architectural role?

Each link claims a sentence is about a specific architecture component. Your job: determine
whether what the sentence describes is consistent with the component's role and responsibilities
in the system architecture.

COMPONENTS: {', '.join(comp_names)}

For each case, determine:
- What action, property, or behavior does the sentence describe?
- Does this fit the architectural role of the claimed component?
- Is this at the architecture level (component roles, interactions, interfaces) rather than
  pure implementation detail?

APPROVE only if the described content plausibly relates to the component's architectural role.
REJECT if the content is unrelated to the component or describes internal implementation details.

LINKS:
{cases_block}

Return JSON: {{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

        # Run all 3 branches
        data1 = self.llm.extract_json(self.llm.query(prompt_ref, timeout=180))
        data2 = self.llm.extract_json(self.llm.query(prompt_topic, timeout=180))
        data3 = self.llm.extract_json(self.llm.query(prompt_fit, timeout=180))

        # Parse approvals from each branch (safe default = approve)
        app1 = self._parse_approvals(data1, n)
        app2 = self._parse_approvals(data2, n)
        app3 = self._parse_approvals(data3, n)

        # Majority: approve if >=2 of 3 branches approve
        approved_idx = set()
        for i in range(n):
            votes = sum(1 for s in [app1, app2, app3] if i in s)
            if votes >= 2:
                approved_idx.add(i)

        result = []
        for i in range(n):
            if i in approved_idx:
                result.append(links[i])
            else:
                print(f"    GoT other reject: S{links[i].sentence_number} -> {links[i].component_name}")
        result.extend(links[n:])
        return result

    def _parse_rejections(self, data, n):
        """Parse rejected case indices. Default: not rejected (safe for union voting)."""
        rejected = set()
        if data:
            for j in data.get("judgments", []):
                idx = j.get("case", 0) - 1
                if 0 <= idx < n and not j.get("approve", False):
                    rejected.add(idx)
        return rejected

    def _parse_approvals(self, data, n):
        """Parse approved case indices. Default: approved (safe for intersection voting).
        Cases not mentioned in LLM response default to approved to avoid false rejections."""
        approved = set(range(n))  # default: all approved
        if data:
            mentioned = set()
            for j in data.get("judgments", []):
                idx = j.get("case", 0) - 1
                if 0 <= idx < n:
                    mentioned.add(idx)
                    if not j.get("approve", True):  # default True = safe
                        approved.discard(idx)
        return approved

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
            if l.source in ("implicit", "coreference"):
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

    def _build_judge_prompt(self, comp_names, cases):
        return f"""JUDGE: Review trace links between documentation sentences and architecture components.

APPROVE a link S→C only when ALL FOUR rules pass:

RULE 1 — REFERENCE: S genuinely refers to C, not just coincidental string overlap.
  Like "Parser" in "The Parser validates token sequences" refers to the architecture component,
  but "parser" in "we use a recursive descent parser technique" is a generic concept.
  REJECT when the name appears in its ordinary English sense, not as a component name.

RULE 2 — ARCHITECTURAL LEVEL: S describes C's role, behavior, or interactions — the "what" and
  "why" of the component, not the "how" of its internals.
  Like "The Dispatcher routes requests to handlers" (architectural) vs "The hash map uses open addressing" (implementation).
  REJECT when S describes internal implementation details invisible at the architecture level.

RULE 3 — TOPIC: C is what S is primarily about, not a passing mention.
  Like "The Lexer tokenizes input" is about the Lexer, but "tokens from the Lexer feed into..." mentions Lexer incidentally.
  REJECT when C is mentioned only in passing while S discusses something else.

RULE 4 — NOT GENERIC: The reference is to C as a named entity, not a common English word.
  Like "Broker" in "The Broker mediates between publishers and subscribers" is the component,
  but "broker" in "a message broker pattern" is the generic concept.
  REJECT when the word is used in its dictionary sense rather than as the component's name.

COMPONENTS: {', '.join(comp_names)}

MATCH TEXT GUIDANCE:
- If match text differs from component name: is this text a DIRECT REFERENCE to the component,
  or just a generic term that happens to match?
- If match:NONE: the link was inferred from pronouns/context — verify the reference chain.

LINKS:
{chr(10).join(cases)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

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
1. "{comp_name.lower()}" is used as a modifier/adjective in a compound phrase (e.g., "cascade {comp_name.lower()}", "minimal {comp_name.lower()}") — NOT as a standalone noun referring to the component.
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
is clearly used as a modifier in a compound phrase ("cascade {comp_name.lower()}"), a technology name,
or the sentence is entirely about a different component.

Return JSON: {{"verdict": "APPROVE" or "REJECT", "reason": "brief explanation"}}
JSON only:"""

        jury_data = self.llm.extract_json(self.llm.query(jury_prompt, timeout=60))
        if jury_data:
            return jury_data.get("verdict", "APPROVE").upper() == "APPROVE"
        return True  # Default approve on failure
