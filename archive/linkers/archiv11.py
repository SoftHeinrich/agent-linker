"""AgentLinker V6B: V6 + abbreviation guard (Feature B from V9T).

Only change from V6: adds code-level abbreviation disambiguation.
When an abbreviation like "GAE" maps to a multi-word component ("GAE Datastore"),
rejects matches where the abbreviation is followed by a DIFFERENT noun
(e.g. "GAE server" should NOT match "GAE Datastore").

Applied at three points:
1. After Phase 5 entity extraction (filter candidates)
2. After Phase 7 coreference (filter coref links)
3. After Phase 8b partial injection (filter partial links)
"""

import os
import re
import time
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
from .agent_linker_v6 import AgentLinkerV6


class AgentLinkerV6B(AgentLinkerV6):
    """V6 + abbreviation guard for multi-word components."""

    def __init__(self, backend: Optional[LLMBackend] = None,
                 post_filter: str = "none"):
        super().__init__(backend=backend, post_filter=post_filter)
        print("  + Feature B: abbreviation guard")

    # ═══════════════════════════════════════════════════════════════════
    # Abbreviation guard helpers
    # ═══════════════════════════════════════════════════════════════════

    def _abbreviation_match_is_valid(self, abbrev, comp_name, sentence_text):
        """Check if abbreviation usage validly refers to the multi-word component.

        When abbreviation (e.g. "GAE") is a prefix of a multi-word component
        ("GAE Datastore"), reject if followed by a DIFFERENT noun ("GAE server").
        Accept standalone abbreviation or full component name.
        """
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
        """Apply abbreviation guard to entity extraction candidates.

        Two checks:
        1. If matched_text is a known abbreviation, verify it's valid.
        2. If component is multi-word and full name NOT in text but
           abbreviation IS, apply guard (catches LLM hallucinating
           full names from abbreviation-only sentences).
        """
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

            # Check 1: matched_text is an abbreviation
            if matched_lower in abbrev_to_comp and abbrev_to_comp[matched_lower] == comp:
                if sent and not self._abbreviation_match_is_valid(c.matched_text, comp, sent.text):
                    print(f"    Abbrev guard: rejected S{c.sentence_number} {c.matched_text} -> {comp}")
                    continue

            # Check 2: multi-word component, full name NOT in text, abbreviation IS
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
        """Check if a link's component was matched via abbreviation; if so, guard it.

        Skips guard if full component name appears standalone in text.
        """
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

    # ═══════════════════════════════════════════════════════════════════
    # Override link() to insert abbreviation guard at 3 points
    # ═══════════════════════════════════════════════════════════════════

    def link(self, text_path: str, model_path: str,
             transarc_csv: str = None) -> list[SadSamLink]:

        self._phase_log = []
        t0 = time.time()

        components = parse_pcm_repository(model_path)
        sentences = DocumentLoader.load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        id_to_name = {c.id: c.name for c in components}
        sent_map = DocumentLoader.build_sent_map(sentences)
        self._cached_sent_map = sent_map

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        # ── Semantic filter init ───────────────────────────────────────
        if self.post_filter != "none":
            print(f"\n[Semantic Filter Init] {self.post_filter}")
            from .agent_linker_v6 import _EmbeddingFilter
            self._semantic_filter = _EmbeddingFilter(components, sentences)

        # ── Phase 0: Document profile ──────────────────────────────────
        print("\n[Phase 0] Document Profile")
        self.doc_profile = self._learn_document_profile(sentences, components)
        self._is_complex = self._structural_complexity(sentences, components)
        spc = len(sentences) / max(1, len(components))
        print(f"  Stats: {spc:.1f} sents/comp, {self.doc_profile.pronoun_ratio:.0%} pronouns")
        print(f"  Complex: {self._is_complex}")
        self.thresholds = LearnedThresholds(0, 0, 0, 0, "qualitative", 0)
        self._log("phase_0", {"sents": len(sentences), "comps": len(components)},
                  {"spc": spc, "pronoun_ratio": self.doc_profile.pronoun_ratio,
                   "complex": self._is_complex})

        # ── Phase 1: Model analysis ────────────────────────────────────
        print("\n[Phase 1] Model Structure")
        self.model_knowledge = self._analyze_model(components)
        print(f"  Architectural: {len(self.model_knowledge.architectural_names)}")
        print(f"  Ambiguous: {self.model_knowledge.ambiguous_names}")
        self._log("phase_1", {"components": [c.name for c in components]},
                  {"architectural": sorted(self.model_knowledge.architectural_names),
                   "ambiguous": sorted(self.model_knowledge.ambiguous_names)})

        # ── Phase 2: Pattern learning (debate) ─────────────────────────
        print("\n[Phase 2] Pattern Learning")
        self.learned_patterns = self._learn_patterns_with_debate(sentences, components)
        print(f"  Subprocess terms: {len(self.learned_patterns.subprocess_terms)}")
        self._log("phase_2", {}, {"subprocess": sorted(self.learned_patterns.subprocess_terms)})

        # ── Phase 3: Document knowledge (judge) ────────────────────────
        print("\n[Phase 3] Document Knowledge")
        self.doc_knowledge = self._learn_document_knowledge_with_judge(sentences, components)
        print(f"  Abbrev: {len(self.doc_knowledge.abbreviations)}, "
              f"Syn: {len(self.doc_knowledge.synonyms)}, "
              f"Generic: {len(self.doc_knowledge.generic_terms)}")
        self._log("phase_3", {}, {
            "abbreviations": self.doc_knowledge.abbreviations,
            "synonyms": self.doc_knowledge.synonyms,
            "generic": list(self.doc_knowledge.generic_terms),
        })

        # ── Phase 3b: Multi-word component enrichment ────────────────
        self._enrich_multiword_partials(sentences, components)

        # ── Phase 4: TransArc baseline ─────────────────────────────────
        print("\n[Phase 4] TransArc")
        transarc_links = self._process_transarc(transarc_csv, id_to_name, sent_map, name_to_id)
        transarc_set = {(l.sentence_number, l.component_id) for l in transarc_links}
        print(f"  Links: {len(transarc_links)}")
        self._log("phase_4", {"csv": transarc_csv}, {"count": len(transarc_links)}, transarc_links)

        # ── Calibrate embedding filter ─────────────────────────────────
        if self._semantic_filter is not None:
            print("\n[Semantic Filter Calibration]")
            self._semantic_filter.calibrate_from_known_links(transarc_links)

        # ── Phase 5: Entity extraction ─────────────────────────────────
        print("\n[Phase 5] Entity Extraction")
        candidates = self._extract_entities(sentences, components, name_to_id, sent_map)
        print(f"  Candidates: {len(candidates)}")

        # *** FEATURE B: abbreviation guard on candidates ***
        before_guard = len(candidates)
        candidates = self._apply_abbreviation_guard_to_candidates(candidates, sent_map)
        if len(candidates) < before_guard:
            print(f"  After abbrev guard: {len(candidates)} (-{before_guard - len(candidates)})")

        self._log("phase_5", {}, {"count": len(candidates)},
                  [SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source) for c in candidates])

        # ── Phase 6: Self-consistency validation ───────────────────────
        print("\n[Phase 6] Validation")
        validated = self._validate_with_self_consistency(candidates, components, sent_map)
        print(f"  Validated: {len(validated)} (of {len(candidates)})")
        self._log("phase_6", {"candidates": len(candidates)},
                  {"validated": len(validated), "rejected": len(candidates) - len(validated)},
                  [SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source) for c in validated])

        # ── Phase 7: Coreference (adaptive: discourse vs debate) ──────
        print("\n[Phase 7] Coreference")
        if self._is_complex:
            print(f"  Mode: debate (complex, {len(sentences)} sents)")
            discourse_model = None
            coref_links = self._coref_debate(sentences, components, name_to_id, sent_map)
        else:
            discourse_model = self._build_discourse_model(sentences, components, name_to_id)
            print(f"  Mode: discourse ({len(sentences)} sents)")
            coref_links = self._coref_discourse(sentences, components, name_to_id, sent_map, discourse_model)
        print(f"  Coref links: {len(coref_links)}")

        # *** FEATURE B: abbreviation guard on coref links ***
        before_guard = len(coref_links)
        coref_links = [lk for lk in coref_links if self._abbreviation_guard_for_link(lk, sent_map)]
        if len(coref_links) < before_guard:
            print(f"  After abbrev guard: {len(coref_links)} (-{before_guard - len(coref_links)})")

        self._log("phase_7", {"method": "debate" if self._is_complex else "discourse"},
                  {"count": len(coref_links)}, coref_links)

        # ── Phase 8: Implicit references (skip for complex docs) ──────
        print("\n[Phase 8] Implicit References")
        existing = (transarc_set
                    | {(c.sentence_number, c.component_id) for c in validated}
                    | {(l.sentence_number, l.component_id) for l in coref_links})

        if self._is_complex:
            print("  SKIPPED (complex doc)")
            implicit_links = []
        else:
            implicit_links = self._detect_implicit_references(
                sentences, components, name_to_id, sent_map, discourse_model, existing
            )
            print(f"  Detected: {len(implicit_links)}")
            if implicit_links:
                implicit_links = self._validate_implicit_links(
                    implicit_links, sentences, components, sent_map
                )
                print(f"  After validation: {len(implicit_links)}")
        self._log("phase_8", {"existing": len(existing), "complex": self._is_complex},
                  {"count": len(implicit_links)}, implicit_links)

        # ── Phase 8b: Partial-reference injection ─────────────────────
        partial_links = self._inject_partial_references(
            sentences, components, name_to_id, transarc_set,
            {(c.sentence_number, c.component_id) for c in validated},
            {(l.sentence_number, l.component_id) for l in coref_links},
            {(l.sentence_number, l.component_id) for l in implicit_links},
        )
        if partial_links:
            print(f"\n[Phase 8b] Partial Injection")
            print(f"  Injected: {len(partial_links)} candidates")

            # *** FEATURE B: abbreviation guard on partial links ***
            before_guard = len(partial_links)
            partial_links = [lk for lk in partial_links if self._abbreviation_guard_for_link(lk, sent_map)]
            if len(partial_links) < before_guard:
                print(f"  After abbrev guard: {len(partial_links)} (-{before_guard - len(partial_links)})")

            self._log("phase_8b", {}, {"count": len(partial_links)}, partial_links)

        # ── Combine + deduplicate by source priority ───────────────────
        entity_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
            for c in validated
        ]
        all_links = transarc_links + entity_links + coref_links + implicit_links + partial_links
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

        # ── Phase 9: Judge review ──────────────────────────────────────
        print("\n[Phase 9] Judge Review")
        reviewed = self._judge_review(preliminary, sentences, components, sent_map, transarc_set)
        rejected = [l for l in preliminary if (l.sentence_number, l.component_id)
                    not in {(r.sentence_number, r.component_id) for r in reviewed}]
        print(f"  Approved: {len(reviewed)} (rejected {len(rejected)})")
        self._log("phase_9", {"input": len(preliminary)},
                  {"approved": len(reviewed), "rejected": len(rejected)}, reviewed)
        if rejected:
            self._log("phase_9_rejected", {}, {"count": len(rejected)}, rejected)

        # ── Phase 10: FN recovery with judge confirmation ──────────────
        print("\n[Phase 10] FN Recovery")
        final = self._fn_recovery(reviewed, sentences, components, name_to_id, sent_map)
        recovered = [l for l in final if l.source == "recovered"]
        print(f"  Final: {len(final)} (+{len(recovered)} recovered)")
        self._log("phase_10", {"ambiguous_skipped": sorted(
            self.model_knowledge.ambiguous_names if self.model_knowledge else [])},
            {"final": len(final), "recovered": len(recovered)}, recovered or None)

        # ── Post-filter: selective embedding ────────────────────────────
        if self._semantic_filter is not None and self._semantic_filter.threshold is not None:
            before = len(final)
            final = self._apply_selective_filter(final, sent_map)
            removed = before - len(final)
            print(f"  Semantic filter ({self.post_filter}): removed {removed}")
            self._log("post_filter", {"before": before, "type": self.post_filter,
                                       "threshold": self._semantic_filter.threshold},
                      {"after": len(final), "removed": removed}, final)

        # ── Save log ───────────────────────────────────────────────────
        self._log("summary", {"total_time_s": round(time.time() - t0, 1)},
                  {"final": len(final)}, final)
        self._save_log(text_path)

        print(f"\nFinal: {len(final)} links")
        return final
