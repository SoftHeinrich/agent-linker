"""AgentLinker V9T: Error-driven features on top of V5.

6 independently-toggleable code-level features targeting specific FP/FN patterns:
  A: fix_dot_filter      — Port V6 dot/hyphen fix for _has_standalone_mention/_has_clean_mention
  B: abbreviation_guard  — Reject abbreviation+different-noun matches (e.g. "GAE server" != GAE Datastore)
  C: coref_distance_filter — Post-Phase 7 proximity guard (component must be nearby)
  D: generic_word_filter — Reject lowercase single-word names in compound phrases pre-judge
  E: ambiguous_recovery  — Controlled FN recovery for multi-word ambiguous components
  F: section_heading_safe — Auto-accept section headings that name a component
"""

import re
import os
from typing import Optional

from ...core import SadSamLink
from ...llm_client import LLMBackend
from .agent_linker_v5 import AgentLinkerV5


class AgentLinkerV9T(AgentLinkerV5):
    """V5 + 6 error-driven features, each independently toggleable."""

    def __init__(self, backend: Optional[LLMBackend] = None,
                 post_filter: str = "none",
                 fix_dot_filter: bool = True,
                 abbreviation_guard: bool = True,
                 coref_distance_filter: bool = True,
                 generic_word_filter: bool = True,
                 ambiguous_recovery: bool = True,
                 section_heading_safe: bool = True):
        super().__init__(backend=backend, post_filter=post_filter)
        self.fix_dot_filter = fix_dot_filter
        self.abbreviation_guard = abbreviation_guard
        self.coref_distance_filter = coref_distance_filter
        self.generic_word_filter = generic_word_filter
        self.ambiguous_recovery = ambiguous_recovery
        self.section_heading_safe = section_heading_safe

        features = []
        if fix_dot_filter: features.append("A:dot_filter")
        if abbreviation_guard: features.append("B:abbrev_guard")
        if coref_distance_filter: features.append("C:coref_dist")
        if generic_word_filter: features.append("D:generic_word")
        if ambiguous_recovery: features.append("E:ambig_recovery")
        if section_heading_safe: features.append("F:heading_safe")
        print(f"AgentLinkerV9T features: {', '.join(features) or 'NONE'}")

    # ═══════════════════════════════════════════════════════════════════
    # Feature A: Fix dot/hyphen filter (ported from V6)
    # ═══════════════════════════════════════════════════════════════════

    def _has_standalone_mention(self, comp_name, text):
        if not self.fix_dot_filter:
            return super()._has_standalone_mention(comp_name, text)

        # V6 version: allow sentence-ending periods, allow hyphens in comp names
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
            # Skip package paths: word.Name (but NOT sentence-ending periods)
            if s > 0 and text[s-1] == '.':
                continue
            if e < len(text) and text[e] == '.' and e + 1 < len(text) and text[e+1].isalpha():
                continue  # Mid-text dot (package path like "logic.core")
            # Skip hyphenated compounds (but NOT component names containing hyphens)
            if s > 0 and text[s-1] == '-':
                continue
            if e < len(text) and text[e] == '-' and '-' not in comp_name:
                continue
            return True
        return False

    def _has_clean_mention(self, term, text):
        if not self.fix_dot_filter:
            return super()._has_clean_mention(term, text)

        # V6 version: allow sentence-ending periods
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

    # ═══════════════════════════════════════════════════════════════════
    # Feature B: Abbreviation guard
    # ═══════════════════════════════════════════════════════════════════

    def _abbreviation_match_is_valid(self, abbrev, comp_name, sentence_text):
        """Check if abbreviation usage in sentence validly refers to the component.

        When abbreviation (e.g. "GAE") is a prefix of a multi-word component
        ("GAE Datastore"), reject if the abbreviation is followed by a DIFFERENT
        noun in the sentence (e.g. "GAE server"). Accept standalone abbreviation
        or full component name.
        """
        if not self.abbreviation_guard:
            return True

        comp_parts = comp_name.split()
        if len(comp_parts) < 2:
            return True  # Single-word component, no ambiguity

        # Check if abbreviation is a prefix of the component name
        if not comp_name.upper().startswith(abbrev.upper()):
            return True  # Not a prefix abbreviation, skip guard

        # Find all occurrences of the abbreviation in the sentence
        pattern = rf'\b{re.escape(abbrev)}\b'
        full_rest = comp_name[len(abbrev):].strip()
        found_valid = False
        for m in re.finditer(pattern, sentence_text, re.IGNORECASE):
            end = m.end()
            rest = sentence_text[end:].lstrip()
            if not rest:
                found_valid = True
                break  # Standalone at end of sentence

            # If full component name follows, it's valid
            if rest.lower().startswith(full_rest.lower()):
                found_valid = True
                break

            # Check if followed by a different noun
            next_word_m = re.match(r'(\w+)', rest)
            if next_word_m:
                next_word = next_word_m.group(1).lower()
                expected_next = full_rest.split()[0].lower() if full_rest else ""
                if next_word != expected_next and next_word.isalpha():
                    continue  # This occurrence is invalid, try next
            # No problematic word follows (punctuation, etc.)
            found_valid = True
            break

        return found_valid

    # ═══════════════════════════════════════════════════════════════════
    # Feature C: Coref distance filter
    # ═══════════════════════════════════════════════════════════════════

    def _filter_coref_by_proximity(self, coref_links, sent_map, components, name_to_id):
        """Filter coreference links by proximity: component must have been
        mentioned (by name, synonym, or partial) within 3 previous sentences."""
        if not self.coref_distance_filter:
            return coref_links

        # Build reverse lookup for synonyms/abbreviations/partials
        alt_to_comp = {}
        if self.doc_knowledge:
            for k, v in self.doc_knowledge.synonyms.items():
                alt_to_comp[k.lower()] = v
            for k, v in self.doc_knowledge.abbreviations.items():
                alt_to_comp[k.lower()] = v
            for k, v in self.doc_knowledge.partial_references.items():
                alt_to_comp[k.lower()] = v

        max_dist = 3
        filtered = []
        for lk in coref_links:
            comp = lk.component_name
            found_nearby = False
            for offset in range(1, max_dist + 1):
                prev = sent_map.get(lk.sentence_number - offset)
                if not prev:
                    continue
                tl = prev.text
                # Check direct name mention (word boundary)
                if re.search(rf'\b{re.escape(comp)}\b', tl, re.IGNORECASE):
                    found_nearby = True
                    break
                # Check synonym/abbreviation/partial mentions (word boundary)
                for alt, target in alt_to_comp.items():
                    if target == comp and re.search(rf'\b{re.escape(alt)}\b', tl, re.IGNORECASE):
                        found_nearby = True
                        break
                if found_nearby:
                    break
            if found_nearby:
                filtered.append(lk)
            else:
                print(f"    Coref proximity filter: rejected S{lk.sentence_number} -> {comp}")
        return filtered

    # ═══════════════════════════════════════════════════════════════════
    # Feature D: Generic word filter
    # ═══════════════════════════════════════════════════════════════════

    def _is_generic_word_usage(self, comp_name, sentence_text):
        """Check if a single-word component name appears ONLY in lowercase
        within compound phrases (e.g. "cascade logic", "processing logic").

        Returns True if usage is generic (should be rejected).
        """
        if not self.generic_word_filter:
            return False

        if ' ' in comp_name:
            return False  # Multi-word names not affected

        # Check if the component name appears capitalized (proper noun)
        cap_name = comp_name[0].upper() + comp_name[1:]
        cap_pattern = rf'\b{re.escape(cap_name)}\b'
        if re.search(cap_pattern, sentence_text):
            return False  # Has capitalized occurrence — not generic

        # Check if it appears only lowercase
        lower_pattern = rf'\b{re.escape(comp_name.lower())}\b'
        matches = list(re.finditer(lower_pattern, sentence_text.lower()))
        if not matches:
            return False

        # Check if every occurrence is preceded by a content-word modifier
        # (adjective, noun used as modifier — NOT articles/prepositions/determiners)
        NON_MODIFIERS = {
            'the', 'a', 'an', 'this', 'that', 'these', 'those', 'its', 'their',
            'our', 'your', 'my', 'his', 'her', 'some', 'any', 'all', 'each',
            'every', 'no', 'of', 'in', 'on', 'at', 'to', 'for', 'by', 'with',
            'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'has', 'have', 'had', 'and', 'or', 'but', 'not', 'if', 'when',
        }
        all_in_compound = True
        for m in re.finditer(lower_pattern, sentence_text, re.IGNORECASE):
            s = m.start()
            before = sentence_text[:s].rstrip()
            if before:
                prev_word_m = re.search(r'(\w+)\s*$', before)
                if prev_word_m:
                    prev_word = prev_word_m.group(1)
                    # Only treat as compound if preceded by a content word
                    # (not articles, prepositions, determiners, etc.)
                    if prev_word.lower() not in NON_MODIFIERS and prev_word[0].islower():
                        continue
            all_in_compound = False
            break

        return all_in_compound

    # ═══════════════════════════════════════════════════════════════════
    # Feature E: Ambiguous recovery
    # ═══════════════════════════════════════════════════════════════════

    def _recover_multiword_ambiguous(self, current_links, sentences, components, name_to_id, sent_map):
        """Recovery for multi-word ambiguous components when FULL name appears."""
        if not self.ambiguous_recovery:
            return []

        ambiguous = self.model_knowledge.ambiguous_names if self.model_knowledge else set()
        if not ambiguous:
            return []

        # Only multi-word ambiguous names
        multiword_ambig = {n for n in ambiguous if ' ' in n and n in name_to_id}
        if not multiword_ambig:
            return []

        covered = {(l.sentence_number, l.component_id) for l in current_links}
        comp_names = self._get_comp_names(components)
        candidates = []

        for comp_name in multiword_ambig:
            comp_id = name_to_id[comp_name]
            pattern = rf'\b{re.escape(comp_name)}\b'
            for sent in sentences:
                key = (sent.number, comp_id)
                if key in covered:
                    continue
                if re.search(pattern, sent.text, re.IGNORECASE):
                    candidates.append((sent.number, sent.text, comp_name))

        if not candidates:
            return []

        print(f"    Ambiguous recovery candidates: {len(candidates)} for {sorted(multiword_ambig)}")

        # Extra-strict two-pass judge
        recovered = []
        for batch_start in range(0, len(candidates), 12):
            batch = candidates[batch_start:batch_start + 12]
            cases = [f"Case {i+1}: S{sn}: \"{txt[:70]}...\" -> {cn}?"
                     for i, (sn, txt, cn) in enumerate(batch)]

            r1 = self._qual_recovery_pass(comp_names, cases, "Is component the ACTOR?")
            r2 = self._qual_recovery_pass(comp_names, cases, "Is this a valid architectural reference?")

            for i, (snum, _, cname) in enumerate(batch):
                if r1.get(i, False) and r2.get(i, False) and cname in name_to_id:
                    key = (snum, name_to_id[cname])
                    if key not in covered:
                        recovered.append(SadSamLink(snum, name_to_id[cname], cname, 1.0, "recovered"))
                        covered.add(key)

        # Strict judge confirmation
        if recovered:
            recovered = self._judge_recovered(recovered, comp_names, sent_map)
            for lk in recovered:
                print(f"    Ambig recovered: S{lk.sentence_number} -> {lk.component_name}")

        return recovered

    # ═══════════════════════════════════════════════════════════════════
    # Feature F: Section heading safe
    # ═══════════════════════════════════════════════════════════════════

    def _is_section_heading(self, sentence_text, comp_name):
        """Check if sentence is essentially a section heading naming a component.

        A section heading is very short and essentially equals a component name
        (with optional punctuation, numbering, articles).
        """
        if not self.section_heading_safe:
            return False

        # Strip punctuation, numbering, articles
        cleaned = re.sub(r'^[\d\.\s]+', '', sentence_text)  # leading numbers
        cleaned = re.sub(r'[.:;,!?\s]+$', '', cleaned)  # trailing punctuation
        cleaned = re.sub(r'^(the|a|an)\s+', '', cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()

        if not cleaned:
            return False

        # Check if cleaned text essentially equals component name
        if cleaned.lower() == comp_name.lower():
            return True

        # Allow "Component Name" with minor variations (word-boundary match)
        if len(cleaned) <= len(comp_name) + 10 and re.search(
                rf'\b{re.escape(comp_name)}\b', cleaned, re.IGNORECASE):
            words_outside = re.sub(rf'\b{re.escape(comp_name)}\b', '', cleaned,
                                   flags=re.IGNORECASE).strip()
            if len(words_outside) <= 15:
                return True

        return False

    # ═══════════════════════════════════════════════════════════════════
    # Override: Main pipeline (inject features C, D, F at right points)
    # ═══════════════════════════════════════════════════════════════════

    def link(self, text_path: str, model_path: str,
             transarc_csv: str = None) -> list[SadSamLink]:

        # Run V5 pipeline up to and including Phase 7 by reusing parent
        # We need to intercept at specific points, so we replicate the
        # pipeline with feature insertions.

        import json
        import time
        from ...core import (
            DocumentLoader, LearnedThresholds,
        )
        from ...pcm_parser import parse_pcm_repository

        self._phase_log = []
        t0 = time.time()

        components = parse_pcm_repository(model_path)
        sentences = DocumentLoader.load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        id_to_name = {c.id: c.name for c in components}
        sent_map = DocumentLoader.build_sent_map(sentences)
        self._cached_sent_map = sent_map

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        # Semantic filter init
        if self.post_filter != "none":
            from .agent_linker_v5 import _EmbeddingFilter
            print(f"\n[Semantic Filter Init] {self.post_filter}")
            self._semantic_filter = _EmbeddingFilter(components, sentences)

        # Phase 0
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

        # Phase 1
        print("\n[Phase 1] Model Structure")
        self.model_knowledge = self._analyze_model(components)
        print(f"  Architectural: {len(self.model_knowledge.architectural_names)}")
        print(f"  Ambiguous: {self.model_knowledge.ambiguous_names}")
        self._log("phase_1", {"components": [c.name for c in components]},
                  {"architectural": sorted(self.model_knowledge.architectural_names),
                   "ambiguous": sorted(self.model_knowledge.ambiguous_names)})

        # Phase 2
        print("\n[Phase 2] Pattern Learning")
        self.learned_patterns = self._learn_patterns_with_debate(sentences, components)
        print(f"  Subprocess terms: {len(self.learned_patterns.subprocess_terms)}")
        self._log("phase_2", {}, {"subprocess": sorted(self.learned_patterns.subprocess_terms)})

        # Phase 3
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

        # Phase 3b
        self._enrich_multiword_partials(sentences, components)

        # Phase 4
        print("\n[Phase 4] TransArc")
        transarc_links = self._process_transarc(transarc_csv, id_to_name, sent_map, name_to_id)
        transarc_set = {(l.sentence_number, l.component_id) for l in transarc_links}
        print(f"  Links: {len(transarc_links)}")
        self._log("phase_4", {"csv": transarc_csv}, {"count": len(transarc_links)}, transarc_links)

        # Calibrate embedding filter
        if self._semantic_filter is not None:
            print("\n[Semantic Filter Calibration]")
            self._semantic_filter.calibrate_from_known_links(transarc_links)

        # Phase 5: Entity extraction
        # Feature B: Apply abbreviation guard to entity candidates
        print("\n[Phase 5] Entity Extraction")
        candidates = self._extract_entities(sentences, components, name_to_id, sent_map)
        if self.abbreviation_guard and self.doc_knowledge:
            before = len(candidates)
            candidates = self._apply_abbreviation_guard_to_candidates(candidates, sent_map)
            if len(candidates) < before:
                print(f"  Abbreviation guard removed {before - len(candidates)} candidates")
        print(f"  Candidates: {len(candidates)}")
        self._log("phase_5", {}, {"count": len(candidates)},
                  [SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source) for c in candidates])

        # Phase 6
        print("\n[Phase 6] Validation")
        validated = self._validate_with_self_consistency(candidates, components, sent_map)
        print(f"  Validated: {len(validated)} (of {len(candidates)})")
        self._log("phase_6", {"candidates": len(candidates)},
                  {"validated": len(validated), "rejected": len(candidates) - len(validated)},
                  [SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source) for c in validated])

        # Phase 7: Coreference
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

        # Feature C: Coref proximity filter
        if self.coref_distance_filter and coref_links:
            before_coref = len(coref_links)
            coref_links = self._filter_coref_by_proximity(coref_links, sent_map, components, name_to_id)
            if len(coref_links) < before_coref:
                print(f"  After proximity filter: {len(coref_links)} (removed {before_coref - len(coref_links)})")

        self._log("phase_7", {"method": "debate" if self._is_complex else "discourse"},
                  {"count": len(coref_links)}, coref_links)

        # Phase 8: Implicit references
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

        # Phase 8b: Partial injection
        # Feature B: Apply abbreviation guard to partial injection
        partial_links = self._inject_partial_references(
            sentences, components, name_to_id, transarc_set,
            {(c.sentence_number, c.component_id) for c in validated},
            {(l.sentence_number, l.component_id) for l in coref_links},
            {(l.sentence_number, l.component_id) for l in implicit_links},
        )
        if self.abbreviation_guard and partial_links:
            before_partial = len(partial_links)
            partial_links = [lk for lk in partial_links
                           if self._abbreviation_guard_for_link(lk, sent_map)]
            if len(partial_links) < before_partial:
                print(f"  Abbreviation guard removed {before_partial - len(partial_links)} partial links")
        if partial_links:
            print(f"\n[Phase 8b] Partial Injection")
            print(f"  Injected: {len(partial_links)} candidates")
            self._log("phase_8b", {}, {"count": len(partial_links)}, partial_links)

        # Combine + deduplicate
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

        # Feature D: Pre-Phase 9 generic word filter
        if self.generic_word_filter:
            before_generic = len(preliminary)
            filtered_prelim = []
            for lk in preliminary:
                sent = sent_map.get(lk.sentence_number)
                if sent and self._is_generic_word_usage(lk.component_name, sent.text):
                    print(f"    Generic word filter: rejected S{lk.sentence_number} -> {lk.component_name} (src:{lk.source})")
                else:
                    filtered_prelim.append(lk)
            if len(filtered_prelim) < before_generic:
                print(f"  Generic word filter removed {before_generic - len(filtered_prelim)} links")
            preliminary = filtered_prelim

        # Phase 9: Judge review
        # Feature F: Section heading safe (auto-accept before judge)
        print("\n[Phase 9] Judge Review")
        reviewed = self._judge_review_v9t(preliminary, sentences, components, sent_map, transarc_set)
        rejected = [l for l in preliminary if (l.sentence_number, l.component_id)
                    not in {(r.sentence_number, r.component_id) for r in reviewed}]
        print(f"  Approved: {len(reviewed)} (rejected {len(rejected)})")
        self._log("phase_9", {"input": len(preliminary)},
                  {"approved": len(reviewed), "rejected": len(rejected)}, reviewed)
        if rejected:
            self._log("phase_9_rejected", {}, {"count": len(rejected)}, rejected)

        # Phase 10: FN recovery
        print("\n[Phase 10] FN Recovery")
        final = self._fn_recovery_v9t(reviewed, sentences, components, name_to_id, sent_map)
        recovered = [l for l in final if l.source == "recovered"]
        print(f"  Final: {len(final)} (+{len(recovered)} recovered)")
        self._log("phase_10", {"ambiguous_skipped": sorted(
            self.model_knowledge.ambiguous_names if self.model_knowledge else [])},
            {"final": len(final), "recovered": len(recovered)}, recovered or None)

        # Post-filter
        if self._semantic_filter is not None and self._semantic_filter.threshold is not None:
            before = len(final)
            final = self._apply_selective_filter(final, sent_map)
            removed = before - len(final)
            print(f"  Semantic filter ({self.post_filter}): removed {removed}")
            self._log("post_filter", {"before": before, "type": self.post_filter,
                                       "threshold": self._semantic_filter.threshold},
                      {"after": len(final), "removed": removed}, final)

        # Save log
        self._log("summary", {"total_time_s": round(time.time() - t0, 1)},
                  {"final": len(final)}, final)
        self._save_log(text_path)

        print(f"\nFinal: {len(final)} links")
        return final

    # ═══════════════════════════════════════════════════════════════════
    # Feature B helpers for entity candidates and partial links
    # ═══════════════════════════════════════════════════════════════════

    def _apply_abbreviation_guard_to_candidates(self, candidates, sent_map):
        """Apply abbreviation guard to entity extraction candidates.

        Two checks:
        1. If matched_text is an abbreviation, verify it's valid.
        2. If component is multi-word and full name is NOT in text but an
           abbreviation IS, apply the guard (catches LLM hallucinating
           full component names from abbreviation-only sentences).
        """
        if not self.doc_knowledge:
            return candidates

        # Build abbreviation -> component mapping
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

            # Check 2: component is multi-word, full name NOT in text,
            # but abbreviation IS in text — LLM may have hallucinated
            if sent and comp in comp_to_abbrevs and ' ' in comp:
                full_in_text = re.search(rf'\b{re.escape(comp)}\b', sent.text, re.IGNORECASE)
                if not full_in_text:
                    for abbr in comp_to_abbrevs[comp]:
                        if re.search(rf'\b{re.escape(abbr)}\b', sent.text, re.IGNORECASE):
                            if not self._abbreviation_match_is_valid(abbr, comp, sent.text):
                                print(f"    Abbrev guard (inferred): rejected S{c.sentence_number} {abbr} -> {comp}")
                                break
                    else:
                        filtered.append(c)
                        continue
                    continue  # rejected by abbreviation guard

            filtered.append(c)
        return filtered

    def _abbreviation_guard_for_link(self, lk, sent_map):
        """Check if a link's component was matched via abbreviation and if so, guard it.

        Skips the guard if the full component name appears as a standalone mention
        (link was likely made via full name, not abbreviation).
        """
        if not self.doc_knowledge:
            return True

        sent = sent_map.get(lk.sentence_number)
        if not sent:
            return True

        # If the full component name appears standalone, don't apply abbreviation guard
        if re.search(rf'\b{re.escape(lk.component_name)}\b', sent.text, re.IGNORECASE):
            return True

        # Check if any abbreviation maps to this component
        for abbr, comp in self.doc_knowledge.abbreviations.items():
            if comp == lk.component_name:
                if abbr.lower() in sent.text.lower():
                    if not self._abbreviation_match_is_valid(abbr, comp, sent.text):
                        return False
        return True

    # ═══════════════════════════════════════════════════════════════════
    # Phase 9 with Feature F (section heading safe)
    # ═══════════════════════════════════════════════════════════════════

    def _judge_review_v9t(self, links, sentences, components, sent_map, transarc_set):
        """Judge review with Feature F: auto-accept section headings."""
        # Feature F: Pull out section headings before judge (runs even for small sets)
        heading_safe = []
        rest = list(links)
        if self.section_heading_safe:
            heading_safe = []
            rest = []
            for l in links:
                sent = sent_map.get(l.sentence_number)
                if sent and self._is_section_heading(sent.text, l.component_name):
                    heading_safe.append(l)
                else:
                    rest.append(l)
            if heading_safe:
                print(f"  Section heading auto-accept: {len(heading_safe)} links")
                for lk in heading_safe:
                    print(f"    S{lk.sentence_number} -> {lk.component_name}")

        # Run normal judge on remaining
        reviewed = self._judge_review(rest, sentences, components, sent_map, transarc_set)
        return heading_safe + reviewed

    # ═══════════════════════════════════════════════════════════════════
    # Phase 10 with Features B and E
    # ═══════════════════════════════════════════════════════════════════

    def _fn_recovery_v9t(self, current_links, sentences, components, name_to_id, sent_map):
        """FN recovery with abbreviation guard (B) and ambiguous recovery (E)."""
        # Standard V5 recovery (with abbreviation guard applied to results)
        final = self._fn_recovery(current_links, sentences, components, name_to_id, sent_map)

        # Feature B: Filter recovered links through abbreviation guard
        if self.abbreviation_guard:
            before_guard = len(final)
            final = [lk for lk in final
                     if lk.source != "recovered" or self._abbreviation_guard_for_link(lk, sent_map)]
            removed = before_guard - len(final)
            if removed:
                print(f"    Abbreviation guard removed {removed} recovered links")

        # Feature E: Recover multi-word ambiguous components
        if self.ambiguous_recovery:
            ambig_recovered = self._recover_multiword_ambiguous(
                final, sentences, components, name_to_id, sent_map
            )
            if ambig_recovered:
                final = final + ambig_recovered

        return final

    def _save_log(self, text_path: str):
        import json
        import time
        log_dir = os.environ.get("LLM_LOG_DIR", "./results/llm_logs")
        os.makedirs(log_dir, exist_ok=True)
        ds = os.path.splitext(os.path.basename(text_path))[0]
        path = os.path.join(log_dir, f"v9t_{ds}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w") as f:
            json.dump(self._phase_log, f, indent=2, default=str)
        print(f"  Phase log saved: {path}")
