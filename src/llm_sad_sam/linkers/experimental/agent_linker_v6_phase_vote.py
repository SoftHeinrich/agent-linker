"""AgentLinker V6 Phase Vote: Phase-level voting.

Strategy 2: Run deterministic phases (0-4) once, then run noisy phases
(5-10) N times and majority-vote on the results. ~2x cost instead of 3x
for full pipeline voting.

Deterministic phases: 0 (profile), 1 (model), 2 (patterns), 3 (doc knowledge),
                      3b (multi-word), 4 (TransArc)
Noisy phases: 5 (entity extraction), 6 (validation), 7 (coreference),
              8 (implicit), 8b (partial inject), 9 (judge), 10 (recovery)
"""

import math
import os
import time
from collections import Counter, defaultdict
from typing import Optional

from ...core import (
    SadSamLink,
    CandidateLink,
    DocumentProfile,
    LearnedThresholds,
    ModelKnowledge,
    DocumentKnowledge,
    LearnedPatterns,
    DiscourseContext,
    DocumentLoader,
    Sentence,
)
from ...pcm_parser import parse_pcm_repository
from ...llm_client import LLMBackend
from .agent_linker_v6 import AgentLinkerV6


class AgentLinkerV6PhaseVote(AgentLinkerV6):
    """Phase-level voting: run deterministic phases once, noisy phases N times."""

    def __init__(self, backend: Optional[LLMBackend] = None,
                 post_filter: str = "none", n_runs: int = 3):
        super().__init__(backend=backend, post_filter=post_filter)
        self.n_runs = n_runs
        self.vote_threshold = math.ceil(n_runs / 2)
        print(f"  Phase-vote mode: {n_runs} noisy runs, threshold={self.vote_threshold}")

    def link(self, text_path: str, model_path: str,
             transarc_csv: str = None) -> list[SadSamLink]:

        self._phase_log = []
        t0 = time.time()

        # ══════════════════════════════════════════════════════════════════
        # DETERMINISTIC PHASES (run once)
        # ══════════════════════════════════════════════════════════════════

        components = parse_pcm_repository(model_path)
        sentences = DocumentLoader.load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        id_to_name = {c.id: c.name for c in components}
        sent_map = DocumentLoader.build_sent_map(sentences)
        self._cached_sent_map = sent_map

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        # Phase 0
        print("\n[Phase 0] Document Profile")
        self.doc_profile = self._learn_document_profile(sentences, components)
        self._is_complex = self._structural_complexity(sentences, components)
        spc = len(sentences) / max(1, len(components))
        print(f"  Stats: {spc:.1f} sents/comp, {self.doc_profile.pronoun_ratio:.0%} pronouns")
        print(f"  Complex: {self._is_complex}")
        self.thresholds = LearnedThresholds(0, 0, 0, 0, "qualitative", 0)

        # Phase 1
        print("\n[Phase 1] Model Structure")
        self.model_knowledge = self._analyze_model(components)
        print(f"  Architectural: {len(self.model_knowledge.architectural_names)}")
        print(f"  Ambiguous: {self.model_knowledge.ambiguous_names}")

        # Phase 2
        print("\n[Phase 2] Pattern Learning")
        self.learned_patterns = self._learn_patterns_with_debate(sentences, components)
        print(f"  Subprocess terms: {len(self.learned_patterns.subprocess_terms)}")

        # Phase 3
        print("\n[Phase 3] Document Knowledge")
        self.doc_knowledge = self._learn_document_knowledge_with_judge(sentences, components)
        print(f"  Abbrev: {len(self.doc_knowledge.abbreviations)}, "
              f"Syn: {len(self.doc_knowledge.synonyms)}, "
              f"Generic: {len(self.doc_knowledge.generic_terms)}")

        # Phase 3b
        self._enrich_multiword_partials(sentences, components)

        # Phase 4
        print("\n[Phase 4] TransArc")
        transarc_links = self._process_transarc(transarc_csv, id_to_name, sent_map, name_to_id)
        transarc_set = {(l.sentence_number, l.component_id) for l in transarc_links}
        print(f"  Links: {len(transarc_links)}")

        # Calibrate embedding filter
        if self._semantic_filter is not None:
            print("\n[Semantic Filter Calibration]")
            self._semantic_filter.calibrate_from_known_links(transarc_links)

        print(f"\n{'='*60}")
        print(f"DETERMINISTIC PHASES COMPLETE. Running {self.n_runs} noisy passes...")
        print(f"{'='*60}")

        # ══════════════════════════════════════════════════════════════════
        # NOISY PHASES (run N times, vote)
        # ══════════════════════════════════════════════════════════════════

        all_run_links = []

        for run_idx in range(self.n_runs):
            print(f"\n{'─'*60}")
            print(f"NOISY PASS {run_idx + 1}/{self.n_runs}")
            print(f"{'─'*60}")

            run_links = self._run_noisy_phases(
                sentences, components, name_to_id, id_to_name,
                sent_map, transarc_links, transarc_set
            )
            link_set = {(l.sentence_number, l.component_id): l for l in run_links}
            all_run_links.append(link_set)
            print(f"  Pass {run_idx + 1}: {len(run_links)} links")

        # ══════════════════════════════════════════════════════════════════
        # MAJORITY VOTE
        # ══════════════════════════════════════════════════════════════════

        vote_counts = Counter()
        link_pool = {}
        for link_set in all_run_links:
            for key, link in link_set.items():
                vote_counts[key] += 1
                if key not in link_pool:
                    link_pool[key] = link

        final = []
        for key, count in vote_counts.items():
            if count >= self.vote_threshold:
                link = link_pool[key]
                final.append(SadSamLink(
                    link.sentence_number, link.component_id,
                    link.component_name, count / self.n_runs, link.source,
                ))

        accepted = len(final)
        rejected = len(vote_counts) - accepted
        print(f"\n{'='*60}")
        print(f"PHASE VOTE RESULT: {accepted} accepted, {rejected} rejected")
        print(f"  Union: {len(vote_counts)}, threshold: {self.vote_threshold}/{self.n_runs}")
        print(f"{'='*60}")

        # Save log
        self._log("summary", {"total_time_s": round(time.time() - t0, 1)},
                  {"final": len(final), "n_runs": self.n_runs}, final)
        self._save_log(text_path)

        print(f"\nFinal: {len(final)} links")
        return final

    def _run_noisy_phases(self, sentences, components, name_to_id, id_to_name,
                          sent_map, transarc_links, transarc_set):
        """Run phases 5-10 once, return the resulting link list."""

        # Phase 5: Entity extraction
        candidates = self._extract_entities(sentences, components, name_to_id, sent_map)
        print(f"  [P5] Entities: {len(candidates)}")

        # Phase 6: Validation
        validated = self._validate_with_self_consistency(candidates, components, sent_map)
        print(f"  [P6] Validated: {len(validated)}")

        # Phase 7: Coreference
        if self._is_complex:
            coref_links = self._coref_debate(sentences, components, name_to_id, sent_map)
        else:
            discourse_model = self._build_discourse_model(sentences, components, name_to_id)
            coref_links = self._coref_discourse(sentences, components, name_to_id, sent_map, discourse_model)
        print(f"  [P7] Coref: {len(coref_links)}")

        # Phase 8: Implicit references
        existing = (transarc_set
                    | {(c.sentence_number, c.component_id) for c in validated}
                    | {(l.sentence_number, l.component_id) for l in coref_links})

        if self._is_complex:
            implicit_links = []
        else:
            discourse_model = self._build_discourse_model(sentences, components, name_to_id)
            implicit_links = self._detect_implicit_references(
                sentences, components, name_to_id, sent_map, discourse_model, existing
            )
            if implicit_links:
                implicit_links = self._validate_implicit_links(
                    implicit_links, sentences, components, sent_map
                )
        print(f"  [P8] Implicit: {len(implicit_links)}")

        # Phase 8b: Partial injection
        partial_links = self._inject_partial_references(
            sentences, components, name_to_id, transarc_set,
            {(c.sentence_number, c.component_id) for c in validated},
            {(l.sentence_number, l.component_id) for l in coref_links},
            {(l.sentence_number, l.component_id) for l in implicit_links},
        )
        print(f"  [P8b] Partial: {len(partial_links)}")

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

        # Phase 9: Judge review
        reviewed = self._judge_review(preliminary, sentences, components, sent_map, transarc_set)
        print(f"  [P9] Judge: {len(reviewed)} approved")

        # Phase 10: FN recovery
        final = self._fn_recovery(reviewed, sentences, components, name_to_id, sent_map)
        recovered = len(final) - len(reviewed)
        print(f"  [P10] Final: {len(final)} (+{recovered} recovered)")

        return final
