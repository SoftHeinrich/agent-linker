"""AgentLinker V11: Generic-name-aware pipeline.

Changes from V6:
1. Generic-risk classification after Phase 1 — deterministic set of single-word
   component names that are common English words (Logic, Common, Storage, etc.)
2. Phase 3b stoplist — block auto-partials for generic words
3. Generic-aware entity validation — require capitalized standalone mention
4. Generic-aware coreference — reject coref to generic components without recent mention
5. Remove dead-weight phases (8 implicit, 10 recovery) — 0 TP across all datasets
6. V6C boundary filters (package-path, generic-word, weak-partial, abbreviation guard)
7. Generic-informed judge prompt
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


# ═════════════════════════════════════════════════════════════════════════
# Embedding Filter (name-only, threshold derived from transarc links)
# ═════════════════════════════════════════════════════════════════════════

class _EmbeddingFilter:
    """Sentence-transformer cosine similarity filter with data-derived threshold."""

    def __init__(self, components, sentences, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self._np = np

        print(f"    Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)

        self._comp_emb = {}
        comp_name_list = [c.name for c in components]
        if comp_name_list:
            name_embs = self._model.encode(comp_name_list)
            for c, emb in zip(components, name_embs):
                self._comp_emb[c.name] = emb

        self._sent_emb = {}
        if sentences:
            all_emb = self._model.encode([s.text for s in sentences])
            for s, e in zip(sentences, all_emb):
                self._sent_emb[s.number] = e

        self.threshold = None

    def similarity(self, snum: int, comp_name: str) -> float:
        if comp_name not in self._comp_emb or snum not in self._sent_emb:
            return 0.0
        a, b = self._sent_emb[snum], self._comp_emb[comp_name]
        return float(self._np.dot(a, b) / (self._np.linalg.norm(a) * self._np.linalg.norm(b) + 1e-8))

    def calibrate_from_known_links(self, links):
        import numpy as np
        sims = [self.similarity(lk.sentence_number, lk.component_name)
                for lk in links]
        sims = [s for s in sims if s > 0.0]

        if len(sims) >= 3:
            self.threshold = float(np.mean(sims) - np.std(sims))
            print(f"    Calibrated threshold: {self.threshold:.3f} "
                  f"(mean={np.mean(sims):.3f}, std={np.std(sims):.3f}, n={len(sims)})")
        else:
            self.threshold = 0.0
            print(f"    Not enough transarc links for calibration (n={len(sims)}), threshold=0.0")


# ═════════════════════════════════════════════════════════════════════════
# AgentLinkerV11
# ═════════════════════════════════════════════════════════════════════════

class AgentLinkerV11(AgentLinker):
    """V11: Generic-name-aware pipeline with dead-weight removal."""

    SOURCE_PRIORITY = {
        "transarc": 5, "validated": 4, "entity": 3,
        "coreference": 2, "implicit": 1, "partial_inject": 1, "recovered": 0,
    }

    # ── Generic-risk configuration ────────────────────────────────────

    GENERIC_RISK_WORDS = {
        "logic", "common", "storage", "client", "server", "web",
        "test", "driver", "data", "service", "api", "model",
        "view", "controller", "apps", "core", "base",
        "conversion", "process", "system", "registry", "persistence",
    }

    # Words that are too generic to be reliable partial matches
    GENERIC_PARTIALS = {
        "conversion", "server", "client", "web", "driver", "test",
        "common", "logic", "storage", "data", "service", "api",
        "model", "view", "controller", "manager", "handler",
        "process", "system", "core", "base", "app", "application",
    }

    # Non-modifier words (articles, prepositions, etc.)
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

    def __init__(self, backend: Optional[LLMBackend] = None,
                 post_filter: str = "none"):
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        super().__init__(backend=backend or LLMBackend.CLAUDE)
        self.post_filter = post_filter
        self._phase_log = []
        self._semantic_filter = None
        self._is_complex = None
        self._generic_risk_names = set()
        print("AgentLinkerV11: Generic-name-aware pipeline")
        print(f"  Backend: {self.llm.backend.value}, Model: {os.environ.get('CLAUDE_MODEL', 'default')}")
        if post_filter != "none":
            print(f"  Post-filter: {post_filter}")

    # ── Logging ──────────────────────────────────────────────────────────

    def _log(self, phase: str, input_summary: dict, output_summary: dict, links: list = None):
        entry = {"phase": phase, "ts": time.time(), "in": input_summary, "out": output_summary}
        if links is not None:
            entry["links"] = [
                {"s": l.sentence_number, "c": l.component_name, "src": l.source}
                for l in links
            ]
        self._phase_log.append(entry)

    def _save_log(self, text_path: str):
        log_dir = os.environ.get("LLM_LOG_DIR", "./results/llm_logs")
        os.makedirs(log_dir, exist_ok=True)
        ds = os.path.splitext(os.path.basename(text_path))[0]
        path = os.path.join(log_dir, f"v11_{ds}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w") as f:
            json.dump(self._phase_log, f, indent=2, default=str)
        print(f"  Phase log saved: {path}")

    # ── Override parent O(n²) paragraph boundary ────────────────────────

    def _is_paragraph_boundary(self, sentences, sent_num):
        if sent_num <= 1:
            return True
        curr = self._cached_sent_map.get(sent_num) if hasattr(self, '_cached_sent_map') else None
        if not curr:
            return False
        transitions = ['however', 'furthermore', 'additionally', 'in addition',
                       'moreover', 'on the other hand', 'the following']
        curr_lower = curr.text.lower()
        return any(curr_lower.startswith(t) for t in transitions)

    # ═════════════════════════════════════════════════════════════════════
    # Change 1: Generic-risk classification
    # ═════════════════════════════════════════════════════════════════════

    def _classify_generic_risk(self, components):
        """Deterministic classification of generic-risk component names.

        A component is generic_risk if it's a single word AND that word
        is in GENERIC_RISK_WORDS. Multi-word components (HTML5 Client,
        GAE Datastore, Test Driver) are NOT generic_risk.
        """
        generic = set()
        for c in components:
            if ' ' not in c.name and c.name.lower() in self.GENERIC_RISK_WORDS:
                generic.add(c.name)
        return generic

    # ═════════════════════════════════════════════════════════════════════
    # Main pipeline
    # ═════════════════════════════════════════════════════════════════════

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

        # ── Change 1: Generic-risk classification ──────────────────────
        self._generic_risk_names = self._classify_generic_risk(components)
        if self._generic_risk_names:
            print(f"  Generic-risk: {sorted(self._generic_risk_names)}")
        self._log("phase_1", {"components": [c.name for c in components]},
                  {"architectural": sorted(self.model_knowledge.architectural_names),
                   "ambiguous": sorted(self.model_knowledge.ambiguous_names),
                   "generic_risk": sorted(self._generic_risk_names)})

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

        # ── Phase 3b: Multi-word component enrichment (with stoplist) ──
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

        # Abbreviation guard on candidates (Change 6)
        before_guard = len(candidates)
        candidates = self._apply_abbreviation_guard_to_candidates(candidates, sent_map)
        if len(candidates) < before_guard:
            print(f"  After abbrev guard: {len(candidates)} (-{before_guard - len(candidates)})")

        self._log("phase_5", {}, {"count": len(candidates)},
                  [SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source) for c in candidates])

        # ── Phase 6: Self-consistency validation (generic-aware) ───────
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

        # Change 4: Filter generic-risk coreference
        before_coref = len(coref_links)
        coref_links = self._filter_generic_coref(coref_links, sent_map)
        if len(coref_links) < before_coref:
            print(f"  After generic filter: {len(coref_links)} (-{before_coref - len(coref_links)})")
        print(f"  Coref links: {len(coref_links)}")
        self._log("phase_7", {"method": "debate" if self._is_complex else "discourse"},
                  {"count": len(coref_links), "generic_filtered": before_coref - len(coref_links)},
                  coref_links)

        # ── Phase 8: SKIPPED (dead weight — 0 TP across all datasets) ─
        print("\n[Phase 8] Implicit References — SKIPPED (dead weight)")
        implicit_links = []

        # ── Phase 8b: Partial-reference injection ─────────────────────
        existing = (transarc_set
                    | {(c.sentence_number, c.component_id) for c in validated}
                    | {(l.sentence_number, l.component_id) for l in coref_links})
        partial_links = self._inject_partial_references(
            sentences, components, name_to_id, transarc_set,
            {(c.sentence_number, c.component_id) for c in validated},
            {(l.sentence_number, l.component_id) for l in coref_links},
            set(),  # no implicit links
        )
        if partial_links:
            print(f"\n[Phase 8b] Partial Injection")
            print(f"  Injected: {len(partial_links)} candidates")
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

        # ── Phase 8c: Deterministic boundary filters (Change 6) ───────
        print("\n[Phase 8c] Boundary Filters")
        preliminary, boundary_rejected = self._apply_boundary_filters(
            preliminary, sent_map, transarc_set
        )
        if boundary_rejected:
            print(f"  Rejected: {len(boundary_rejected)}")
            self._log("phase_8c", {},
                      {"rejected": len(boundary_rejected),
                       "details": [(lk.component_name, reason) for lk, reason in boundary_rejected]})

        # ── Phase 9: Judge review (generic-informed) ──────────────────
        print("\n[Phase 9] Judge Review")
        reviewed = self._judge_review(preliminary, sentences, components, sent_map, transarc_set)
        rejected = [l for l in preliminary if (l.sentence_number, l.component_id)
                    not in {(r.sentence_number, r.component_id) for r in reviewed}]
        print(f"  Approved: {len(reviewed)} (rejected {len(rejected)})")
        self._log("phase_9", {"input": len(preliminary)},
                  {"approved": len(reviewed), "rejected": len(rejected)}, reviewed)
        if rejected:
            self._log("phase_9_rejected", {}, {"count": len(rejected)}, rejected)

        # ── Phase 10: SKIPPED (dead weight — 0% recovery rate) ────────
        print("\n[Phase 10] FN Recovery — SKIPPED (dead weight)")
        final = reviewed

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

    # ═════════════════════════════════════════════════════════════════════
    # Phase 0: Document profile + structural complexity
    # ═════════════════════════════════════════════════════════════════════

    def _learn_document_profile(self, sentences, components) -> DocumentProfile:
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

    def _structural_complexity(self, sentences, components, spc_min: float = 5) -> bool:
        comp_names = [c.name for c in components]
        mention_count = sum(1 for sent in sentences
                          if any(cn.lower() in sent.text.lower() for cn in comp_names))
        uncovered_ratio = 1.0 - (mention_count / max(1, len(sentences)))
        spc = len(sentences) / max(1, len(components))
        result = uncovered_ratio > 0.5 and spc > spc_min
        print(f"  Structural complexity: uncovered={uncovered_ratio:.1%}, spc={spc:.1f}, min_spc={spc_min} -> {result}")
        return result

    # ═════════════════════════════════════════════════════════════════════
    # Change 2: Phase 3b with stoplist
    # ═════════════════════════════════════════════════════════════════════

    def _enrich_multiword_partials(self, sentences, components):
        """For multi-word components, check if the last word alone consistently
        refers to that component. BLOCKS generic words via stoplist."""
        if not self.doc_knowledge:
            return

        ambiguous = self.model_knowledge.ambiguous_names if self.model_knowledge else set()
        added = []

        for comp in components:
            parts = comp.name.split()
            if len(parts) < 2:
                continue
            if comp.name in ambiguous:
                continue

            last_word = parts[-1]
            if len(last_word) < 4:
                continue

            last_lower = last_word.lower()

            # Change 2: STOPLIST — block generic words as auto-partials
            if last_lower in self.GENERIC_RISK_WORDS:
                continue

            # Check: does any OTHER component also end with this word?
            other_match = any(
                c.name != comp.name and c.name.lower().endswith(last_lower)
                for c in components
            )
            if other_match:
                continue

            # Already a known partial/synonym?
            if last_lower in {s.lower() for s in self.doc_knowledge.synonyms}:
                continue
            if last_lower in {p.lower() for p in self.doc_knowledge.partial_references}:
                continue

            # Count standalone mentions
            full_lower = comp.name.lower()
            mention_count = 0
            for sent in sentences:
                sl = sent.text.lower()
                if last_lower in sl and full_lower not in sl:
                    if re.search(rf'\b{re.escape(last_word)}\b', sent.text, re.IGNORECASE):
                        mention_count += 1

            if mention_count >= 3:
                self.doc_knowledge.partial_references[last_word] = comp.name
                added.append(f"{last_word} -> {comp.name} ({mention_count} mentions)")

        if added:
            print(f"\n[Phase 3b] Multi-word Enrichment")
            for a in added:
                print(f"    Auto-partial: {a}")
            self._log("phase_3b", {}, {"added": added})

    # ═════════════════════════════════════════════════════════════════════
    # Change 3: Generic-aware entity validation
    # ═════════════════════════════════════════════════════════════════════

    def _validate_with_self_consistency(self, candidates: list[CandidateLink],
                                        components, sent_map) -> list[CandidateLink]:
        if not candidates:
            return []

        comp_names = self._get_comp_names(components)
        needs = [c for c in candidates if c.needs_validation]
        direct = [c for c in candidates if not c.needs_validation]

        # Change 3: Force validation for generic-risk components
        reclassified = []
        for c in direct:
            if c.component_name in self._generic_risk_names:
                c.needs_validation = True
                needs.append(c)
            else:
                reclassified.append(c)
        direct = reclassified

        if not needs:
            return candidates

        # Change 3: Deterministic pre-check for generic-risk candidates
        pre_approved = []
        pre_rejected = []
        remaining = []
        for c in needs:
            if c.component_name in self._generic_risk_names:
                sent = sent_map.get(c.sentence_number)
                if sent and self._has_standalone_mention(c.component_name, sent.text):
                    remaining.append(c)  # Has capitalized mention → let LLM decide
                else:
                    pre_rejected.append(c)  # No capitalized mention → auto-reject
                    print(f"    Generic pre-reject: S{c.sentence_number} -> {c.component_name}")
            else:
                remaining.append(c)

        if pre_rejected:
            print(f"    Generic pre-rejected: {len(pre_rejected)} candidates")

        needs = remaining

        ctx = []
        if self.learned_patterns:
            if self.learned_patterns.action_indicators:
                ctx.append(f"ACTION: {', '.join(self.learned_patterns.action_indicators[:4])}")
            if self.learned_patterns.effect_indicators:
                ctx.append(f"EFFECT (reject): {', '.join(self.learned_patterns.effect_indicators[:3])}")
            if self.learned_patterns.subprocess_terms:
                ctx.append(f"Subprocess (reject): {', '.join(list(self.learned_patterns.subprocess_terms)[:5])}")

        validated = []
        for batch_start in range(0, len(needs), 25):
            batch = needs[batch_start:batch_start + 25]
            cases = []
            for i, c in enumerate(batch):
                prev = sent_map.get(c.sentence_number - 1)
                p = f"[prev: {prev.text[:35]}...] " if prev else ""
                cases.append(f"Case {i+1}: \"{c.matched_text}\" -> {c.component_name}\n  {p}\"{c.sentence_text}\"")

            r1 = self._qual_validation_pass(comp_names, ctx, cases, "Focus on ACTOR")
            r2 = self._qual_validation_pass(comp_names, ctx, cases, "Focus on DIRECT reference")

            for i, c in enumerate(batch):
                if r1.get(i, False) and r2.get(i, False):
                    c.confidence = 1.0
                    c.source = "validated"
                    validated.append(c)

        return direct + validated

    def _qual_validation_pass(self, comp_names, ctx, cases, focus):
        prompt = f"""Validate component references. {focus}.

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

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
    # Phase 7: Adaptive coreference (copied from V6)
    # ═════════════════════════════════════════════════════════════════════

    def _coref_discourse(self, sentences, components, name_to_id, sent_map, discourse_model):
        comp_names = self._get_comp_names(components)
        all_coref = []
        pronoun_sents = [s for s in sentences if self.PRONOUN_PATTERN.search(s.text)]

        for batch_start in range(0, len(pronoun_sents), 12):
            batch = pronoun_sents[batch_start:batch_start + 12]
            cases = []
            for sent in batch:
                ctx = discourse_model.get(sent.number, DiscourseContext())
                prev = []
                for i in range(1, self.CONTEXT_WINDOW + 1):
                    p = sent_map.get(sent.number - i)
                    if p:
                        prev.append(f"S{p.number}: {p.text[:70]}")
                cases.append({"sent": sent, "ctx": ctx, "prev": prev,
                              "likely": ctx.get_likely_referent()})

            prompt = f"""Resolve pronoun references using discourse context.

COMPONENTS: {', '.join(comp_names)}

"""
            for i, case in enumerate(cases):
                ctx = case["ctx"]
                prompt += f"--- Case {i+1}: S{case['sent'].number} ---\n"
                prompt += f"DISCOURSE:\n"
                prompt += f"  Recent mentions: {ctx.get_context_summary(case['sent'].number)}\n"
                prompt += f"  Paragraph topic: {ctx.paragraph_topic or 'None'}\n"
                prompt += f"  Likely referent: {case['likely'] or 'Unknown'}\n"
                if case["prev"]:
                    prompt += f"PREVIOUS:\n  " + "\n  ".join(reversed(case["prev"])) + "\n"
                prompt += f">>> SENTENCE: {case['sent'].text}\n\n"

            prompt += """Return JSON:
{"resolutions": [{"case": 1, "sentence": N, "pronoun": "it", "component": "Name", "certainty": "high or low", "reasoning": "why"}]}

Guidelines:
- "high" certainty: component was the explicit grammatical subject of the previous 1-2 sentences AND the pronoun unambiguously refers to it
- "low" certainty: pronoun could refer to multiple things, component was not the subject, or mentioned long ago
- Only include resolutions where certainty is "high"
- When in doubt, mark as "low" — false positives are worse than missed coreferences

JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=150))
            if not data:
                continue

            for res in data.get("resolutions", []):
                comp = res.get("component")
                snum = res.get("sentence")
                if str(res.get("certainty", "low")).lower() != "high":
                    continue
                if not (comp and snum and comp in name_to_id):
                    continue
                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue
                sent = sent_map.get(snum)
                if sent and self.learned_patterns and self.learned_patterns.is_subprocess(sent.text):
                    continue
                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, 1.0, "coreference"))

        return all_coref

    def _coref_debate(self, sentences, components, name_to_id, sent_map):
        comp_names = self._get_comp_names(components)
        all_coref = []

        ctx = []
        if self.learned_patterns and self.learned_patterns.action_indicators:
            ctx.append(f"ACTION: {', '.join(self.learned_patterns.action_indicators[:3])}")
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

            prompt1 = f"""Resolve pronoun references to components.

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

DOCUMENT (* = analyze):
{chr(10).join(doc_lines)}

Find pronouns (it, they, this) that refer to components.

Return JSON:
{{"resolutions": [{{"sentence": N, "pronoun": "it", "component": "Name", "certainty": "high or low", "reasoning": "why"}}]}}

Only include resolutions where certainty is "high".
JSON only:"""

            data1 = self.llm.extract_json(self.llm.query(prompt1, timeout=100))
            if not data1:
                continue
            proposed = data1.get("resolutions", [])
            if not proposed:
                continue

            judge_batch = proposed[:12]
            ptxt = [f"Case {i+1}: S{r.get('sentence')}: \"{r.get('pronoun', '?')}\" -> {r.get('component')} ({(r.get('reasoning') or '')[:40]})"
                    for i, r in enumerate(judge_batch)]

            prompt2 = f"""JUDGE: Validate these pronoun-to-component resolutions.

COMPONENTS: {', '.join(comp_names)}

PROPOSALS:
{chr(10).join(ptxt)}

CONTEXT:
{chr(10).join(doc_lines[:15])}

Reject if pronoun might refer to something else, or reference is to a subprocess.

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

            data2 = self.llm.extract_json(self.llm.query(prompt2, timeout=100))
            rejected_indices = set()
            if data2:
                for j in data2.get("judgments", []):
                    idx = j.get("case", 0) - 1
                    if 0 <= idx < len(judge_batch) and not j.get("approve", False):
                        rejected_indices.add(idx)

            approved = set()
            for i, res in enumerate(judge_batch):
                if i not in rejected_indices:
                    snum = res.get("sentence")
                    if snum is not None:
                        approved.add(snum)

            for res in judge_batch:
                snum, comp = res.get("sentence"), res.get("component")
                if str(res.get("certainty", "low")).lower() != "high":
                    continue
                if not (snum and comp and comp in name_to_id):
                    continue
                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue
                if snum not in approved:
                    continue
                sent = sent_map.get(snum)
                if sent and self.learned_patterns and self.learned_patterns.is_subprocess(sent.text):
                    continue
                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, 1.0, "coreference"))

        return all_coref

    # ═════════════════════════════════════════════════════════════════════
    # Change 4: Generic-aware coreference filter
    # ═════════════════════════════════════════════════════════════════════

    def _filter_generic_coref(self, coref_links, sent_map):
        """Reject coreference resolutions to generic-risk components unless
        the component was explicitly mentioned (capitalized) in previous 2 sentences."""
        if not self._generic_risk_names:
            return coref_links

        kept = []
        for lk in coref_links:
            if lk.component_name not in self._generic_risk_names:
                kept.append(lk)
                continue

            # Check if component was explicitly mentioned in previous 2 sentences
            found_recent = False
            for offset in range(1, 3):
                prev = sent_map.get(lk.sentence_number - offset)
                if prev and self._has_standalone_mention(lk.component_name, prev.text):
                    found_recent = True
                    break

            if found_recent:
                kept.append(lk)
            else:
                print(f"    Generic coref rejected: S{lk.sentence_number} -> {lk.component_name}")

        return kept

    # ═════════════════════════════════════════════════════════════════════
    # Partial reference injection (copied from V6)
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

        ambiguous = self.model_knowledge.ambiguous_names if self.model_knowledge else set()
        existing = transarc_set | validated_set | coref_set | implicit_set
        injected = []

        for partial, comp_name in self.doc_knowledge.partial_references.items():
            if comp_name in ambiguous or comp_name not in name_to_id:
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
    # Change 6: Boundary filters (from V6C)
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
            return False  # Has capitalized/exact-case match

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

            reason = None

            if lk.source in ("transarc", "validated", "entity"):
                if self._is_in_package_path(lk.component_name, sent.text):
                    reason = "package_path"

            if not reason and lk.source in ("transarc", "validated", "entity", "coreference"):
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
    # Phase 9: Judge review (Change 7: generic-informed)
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

    def _judge_review(self, links, sentences, components, sent_map, transarc_set):
        if len(links) < 5:
            return links

        comp_names = self._get_comp_names(components)
        ambiguous = self.model_knowledge.ambiguous_names if self.model_knowledge else set()

        safe, review = [], []
        for l in links:
            is_ta = (l.sentence_number, l.component_id) in transarc_set
            sent = sent_map.get(l.sentence_number)
            if not sent or l.component_name in ambiguous:
                review.append(l)
                continue

            has_standalone = self._has_standalone_mention(l.component_name, sent.text)

            if has_standalone:
                safe.append(l)
            elif is_ta:
                has_any = re.search(rf'\b{re.escape(l.component_name)}\b',
                                    sent.text, re.IGNORECASE)
                if has_any:
                    review.append(l)
                else:
                    safe.append(l)
            else:
                review.append(l)

        if not review:
            return safe

        cases = self._build_judge_cases(review, sent_map)
        prompt = self._build_judge_prompt(comp_names, cases)
        n = min(30, len(review))

        data1 = self.llm.extract_json(self.llm.query(prompt, timeout=180))
        data2 = self.llm.extract_json(self.llm.query(prompt, timeout=180))

        rej1 = self._parse_rejections(data1, n)
        rej2 = self._parse_rejections(data2, n)
        rejected = rej1 & rej2

        result = safe.copy()
        for i in range(n):
            if i not in rejected:
                result.append(review[i])
        result.extend(review[n:])
        return result

    def _parse_rejections(self, data, n):
        rejected = set()
        if data:
            for j in data.get("judgments", []):
                idx = j.get("case", 0) - 1
                if 0 <= idx < n and not j.get("approve", False):
                    rejected.add(idx)
        return rejected

    def _build_judge_cases(self, review, sent_map):
        cases = []
        for i, l in enumerate(review[:30]):
            sent = sent_map.get(l.sentence_number)
            ctx_lines = []
            if l.source in ("implicit", "coreference"):
                p2 = sent_map.get(l.sentence_number - 2)
                if p2: ctx_lines.append(f"    PREV2: {p2.text[:45]}...")
            p1 = sent_map.get(l.sentence_number - 1)
            if p1: ctx_lines.append(f"    PREV: {p1.text[:45]}...")
            ctx_lines.append(f"    >>> S{l.sentence_number}: {sent.text if sent else '?'}")
            cases.append(f"Case {i+1}: S{l.sentence_number} -> {l.component_name} (src:{l.source})\n"
                         + chr(10).join(ctx_lines))
        return cases

    def _build_judge_prompt(self, comp_names, cases):
        # Change 7: Add generic-risk info to judge prompt
        generic_list = sorted(self._generic_risk_names) if self._generic_risk_names else []
        generic_section = ""
        if generic_list:
            generic_section = f"\nGENERIC-RISK COMPONENTS (require capitalized proper-noun usage): {', '.join(generic_list)}\nFor these components, reject unless the name clearly refers to the SPECIFIC ARCHITECTURAL COMPONENT (capitalized, proper noun), not the generic English concept.\n"

        return f"""JUDGE: Review trace links. Accept references where the sentence is ABOUT the component, reject false matches.

COMPONENTS: {', '.join(comp_names)}
{generic_section}
VALID IF: Sentence mentions or relates to the component as actor, target, location, object, or participant in an architectural interaction. Section headings that name the component are VALID (they introduce a section about that component).
REJECT IF:
- Component name used as a generic English word, not an architecture reference. Common patterns: "the business logic" / "cascade logic" / "processing logic" when Logic is a component; "client-side rendering" / "on the client side" when Client is a component; "data storage" / "in-memory storage" when Storage is a component. The word must refer to the SPECIFIC ARCHITECTURAL COMPONENT, not the general concept.
- Sentence describes package/directory/module structure using dot notation (e.g. "pkg.subpkg contains classes for...")
- Subprocess/sub-task of the component, not the component itself
- Component name appears only as a technology reference, not referring to the actual architectural component

SOURCE-SPECIFIC RULES:
- "transarc": Check carefully — is the component name an ARCHITECTURE REFERENCE or a GENERIC WORD / PACKAGE PATH in this sentence?
- "implicit": Be skeptical — accept only if component is clearly the topic of discussion.
- "coreference": Verify the pronoun clearly refers to the claimed component.

LINKS:
{chr(10).join(cases)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

    # ═════════════════════════════════════════════════════════════════════
    # Selective embedding filter (copied from V6)
    # ═════════════════════════════════════════════════════════════════════

    def _apply_selective_filter(self, links, sent_map):
        if self._semantic_filter is None or self._semantic_filter.threshold is None:
            return links

        threshold = self._semantic_filter.threshold
        result = []
        for lk in links:
            if lk.source == "transarc":
                result.append(lk)
                continue
            sim = self._semantic_filter.similarity(lk.sentence_number, lk.component_name)
            if sim >= threshold:
                result.append(lk)
            else:
                print(f"    Embed rejected: S{lk.sentence_number} -> {lk.component_name} "
                      f"(sim={sim:.3f} < {threshold:.3f}, src={lk.source})")
        return result
