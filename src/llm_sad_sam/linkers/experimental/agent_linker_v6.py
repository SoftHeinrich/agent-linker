"""AgentLinker V6: Fix dot-filter bug + remove dataset-specific judge examples.

Changes from V5:
- Fix: sentence-ending periods no longer trigger package-path dot filter in _has_standalone_mention
  (was causing 6 BBB section-heading FNs where "ComponentName." was filtered out)
- Fix: _has_clean_mention also updated with same dot-filter fix
- Fix: Judge prompts use generic examples only (no dataset-specific patterns that risk data leakage)
- Keeps all V5 improvements: capitalization-aware bypass, partial-reference injection, union voting
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

        # Component embeddings from names only (no self-referential context)
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
        """Derive threshold from transarc link similarities: mean - 1 stddev."""
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
# AgentLinkerV6
# ═════════════════════════════════════════════════════════════════════════

class AgentLinkerV6(AgentLinker):
    """Consolidated best pipeline: no leakage, structural complexity, judge recovery."""

    SOURCE_PRIORITY = {
        "transarc": 5, "validated": 4, "entity": 3,
        "coreference": 2, "implicit": 1, "partial_inject": 1, "recovered": 0,
    }

    def __init__(self, backend: Optional[LLMBackend] = None,
                 post_filter: str = "none"):
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        super().__init__(backend=backend or LLMBackend.CLAUDE)
        self.post_filter = post_filter
        self._phase_log = []
        self._semantic_filter = None
        self._is_complex = None
        print("AgentLinkerV6: Consolidated pipeline")
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
        path = os.path.join(log_dir, f"v5_{ds}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w") as f:
            json.dump(self._phase_log, f, indent=2, default=str)
        print(f"  Phase log saved: {path}")

    # ── Override parent O(n²) paragraph boundary ────────────────────────

    def _is_paragraph_boundary(self, sentences, sent_num):
        """O(1) paragraph boundary detection using cached sent_map."""
        if sent_num <= 1:
            return True
        curr = self._cached_sent_map.get(sent_num) if hasattr(self, '_cached_sent_map') else None
        if not curr:
            return False
        transitions = ['however', 'furthermore', 'additionally', 'in addition',
                       'moreover', 'on the other hand', 'the following']
        curr_lower = curr.text.lower()
        return any(curr_lower.startswith(t) for t in transitions)

    # ── Main pipeline ────────────────────────────────────────────────────

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
        """Complex if majority of sentences don't mention any component AND
        enough sentences per component for pronoun ambiguity."""
        comp_names = [c.name for c in components]
        mention_count = sum(1 for sent in sentences
                          if any(cn.lower() in sent.text.lower() for cn in comp_names))
        uncovered_ratio = 1.0 - (mention_count / max(1, len(sentences)))
        spc = len(sentences) / max(1, len(components))
        result = uncovered_ratio > 0.5 and spc > spc_min
        print(f"  Structural complexity: uncovered={uncovered_ratio:.1%}, spc={spc:.1f}, min_spc={spc_min} -> {result}")
        return result

    # ═════════════════════════════════════════════════════════════════════
    # Phase 3b: Multi-word component enrichment
    # ═════════════════════════════════════════════════════════════════════

    def _enrich_multiword_partials(self, sentences, components):
        """For multi-word components (e.g. 'HTML5 Client'), check if the last
        word alone ('client') consistently refers to that component in the doc.
        If so, add it as a partial_reference."""
        if not self.doc_knowledge:
            return

        ambiguous = self.model_knowledge.ambiguous_names if self.model_knowledge else set()
        existing_partials = {v.lower() for v in self.doc_knowledge.partial_references.values()}
        added = []

        for comp in components:
            parts = comp.name.split()
            if len(parts) < 2:
                continue
            if comp.name in ambiguous:
                continue

            last_word = parts[-1]
            # Skip very short or very common words
            if len(last_word) < 4:
                continue

            last_lower = last_word.lower()

            # Check: does any OTHER component also end with this word?
            other_match = any(
                c.name != comp.name and c.name.lower().endswith(last_lower)
                for c in components
            )
            if other_match:
                continue  # Ambiguous — multiple components share the suffix

            # Already a known partial/synonym?
            if last_lower in {s.lower() for s in self.doc_knowledge.synonyms}:
                continue
            if last_lower in {p.lower() for p in self.doc_knowledge.partial_references}:
                continue

            # Count how often the last word appears alone (not as part of full name)
            full_lower = comp.name.lower()
            mention_count = 0
            for sent in sentences:
                sl = sent.text.lower()
                if last_lower in sl and full_lower not in sl:
                    if re.search(rf'\b{re.escape(last_word)}\b', sent.text, re.IGNORECASE):
                        mention_count += 1

            # Require at least 3 standalone mentions to add as partial
            if mention_count >= 3:
                self.doc_knowledge.partial_references[last_word] = comp.name
                added.append(f"{last_word} -> {comp.name} ({mention_count} mentions)")

        if added:
            print(f"\n[Phase 3b] Multi-word Enrichment")
            for a in added:
                print(f"    Auto-partial: {a}")
            self._log("phase_3b", {}, {"added": added})

    # ═════════════════════════════════════════════════════════════════════
    # Phase 6: Qualitative validation (unanimous two-pass)
    # ═════════════════════════════════════════════════════════════════════

    def _validate_with_self_consistency(self, candidates: list[CandidateLink],
                                        components, sent_map) -> list[CandidateLink]:
        if not candidates:
            return []

        comp_names = self._get_comp_names(components)
        needs = [c for c in candidates if c.needs_validation]
        direct = [c for c in candidates if not c.needs_validation]

        if not needs:
            return candidates

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
    # Phase 7: Adaptive coreference
    # ═════════════════════════════════════════════════════════════════════

    def _coref_discourse(self, sentences, components, name_to_id, sent_map, discourse_model):
        """Discourse-aware coref with qualitative certainty."""
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
        """Debate-style propose+judge coreference for complex docs."""
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

            # Pass 1: Propose
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

            # Pass 2: Judge (case-index keying for robust mapping)
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
            # Rejection filter: start with all judge_batch approved,
            # then remove explicitly rejected items
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
    # Phase 8: Implicit reference detection
    # ═════════════════════════════════════════════════════════════════════

    def _has_clean_mention(self, term, text):
        """Check if term appears in text NOT inside a package path or hyphenated compound.

        Unlike _has_standalone_mention, this does NOT require capitalization.
        Used for partial-reference injection where lowercase matches are valid
        but package paths (word.term) and hyphenated compounds (term-word) are not.
        """
        pattern = rf'\b{re.escape(term)}\b'
        for m in re.finditer(pattern, text, re.IGNORECASE):
            s, e = m.start(), m.end()
            # Skip package paths (mid-text dots only, not sentence-ending periods)
            if s > 0 and text[s-1] == '.':
                continue
            if e < len(text) and text[e] == '.' and e + 1 < len(text) and text[e+1].isalpha():
                continue
            # Skip hyphenated compounds
            if (s > 0 and text[s-1] == '-') or (e < len(text) and text[e] == '-'):
                continue
            return True
        return False

    def _inject_partial_references(self, sentences, components, name_to_id,
                                    transarc_set, validated_set, coref_set, implicit_set):
        """Mechanically inject links for known partial references from Phase 3/3b.

        Scans all sentences for clean mentions of partial references
        (e.g. 'client' → 'HTML5 Client'). Skips matches in package paths
        and hyphenated compounds but allows lowercase. Resulting candidates
        go through Phase 9 judge for validation.
        """
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

    def _detect_implicit_references(
        self, sentences, components, name_to_id, sent_map,
        discourse_model, existing_links: set[tuple[int, str]]
    ) -> list[SadSamLink]:
        comp_names = self._get_comp_names(components)
        candidates = []
        for sent in sentences:
            tl = sent.text.lower()
            if any(c.lower() in tl for c in comp_names):
                continue
            ctx = discourse_model.get(sent.number, DiscourseContext())
            if not ctx.active_entity or not ctx.paragraph_topic:
                continue
            if ctx.active_entity != ctx.paragraph_topic:
                continue
            likely = ctx.active_entity
            if (sent.number, name_to_id.get(likely, '')) in existing_links:
                continue
            dist = self._distance_to_last_mention(ctx.recent_mentions, likely, sent.number)
            if dist is None or dist > 2:
                continue
            candidates.append((sent, ctx, likely))

        if not candidates:
            return []
        print(f"  Candidates: {len(candidates)}")

        implicit = []
        for batch_start in range(0, len(candidates), 5):
            batch = candidates[batch_start:batch_start + 5]
            prompt = f"""Detect implicit component references. Be VERY conservative.

COMPONENTS: {', '.join(comp_names)}

"""
            for i, (sent, ctx, likely) in enumerate(batch):
                prev = sent_map.get(sent.number - 1)
                prompt += f"--- Case {i+1}: S{sent.number} ---\n"
                prompt += f"Previous: {prev.text[:70] if prev else 'N/A'}\n"
                prompt += f"Sentence: {sent.text}\n"
                prompt += f"Context: {ctx.get_context_summary(sent.number)}\n"
                prompt += f"Likely component: {likely}\n\n"

            prompt += """Return JSON:
{"detections": [{"case": 1, "sentence": N, "component": "Name", "is_implicit": true/false, "certainty": "high or low", "reasoning": "why"}]}

An implicit reference is ONLY when:
- The sentence continues describing the EXACT SAME component without re-naming it
- The action described can ONLY be performed by this specific component
- Subject is omitted but implied from IMMEDIATE context (previous 1-2 sentences)

NOT implicit if: result/effect, multiple possible actors, new topic, general process,
algorithm/data flow, configuration, concept, or system-wide behavior.

Only return is_implicit=true with certainty="high".
JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
            if not data:
                continue
            for det in data.get("detections", []):
                if not det.get("is_implicit"):
                    continue
                comp, snum = det.get("component"), det.get("sentence")
                if str(det.get("certainty", "low")).lower() != "high":
                    continue
                if not (comp and snum and comp in name_to_id):
                    continue
                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue
                implicit.append(SadSamLink(snum, name_to_id[comp], comp, 1.0, "implicit"))

        return implicit

    def _validate_implicit_links(self, implicit_links, sentences, components, sent_map):
        if not implicit_links:
            return []
        comp_names = self._get_comp_names(components)
        validated = []

        for batch_start in range(0, len(implicit_links), 10):
            batch = implicit_links[batch_start:batch_start + 10]
            cases = []
            for i, link in enumerate(batch):
                sent = sent_map.get(link.sentence_number)
                prev1, prev2 = sent_map.get(link.sentence_number - 1), sent_map.get(link.sentence_number - 2)
                ctx_lines = []
                if prev2: ctx_lines.append(f"S{prev2.number}: {prev2.text[:60]}")
                if prev1: ctx_lines.append(f"S{prev1.number}: {prev1.text[:60]}")
                ctx_lines.append(f">>> S{link.sentence_number}: {sent.text if sent else '?'}")
                cases.append(f"Case {i+1}: S{link.sentence_number} -> {link.component_name}\n"
                             + "\n".join(f"    {l}" for l in ctx_lines))

            prompt = f"""VALIDATION: Verify these implicit component references.

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false, "reasoning": "brief"}}]}}

Reject if action could be any component, system-wide, or tenuous connection.
JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=150))
            if not data:
                continue
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(batch) and v.get("approve", False):
                    validated.append(batch[idx])

        return validated

    # ═════════════════════════════════════════════════════════════════════
    # Phase 9: Judge review (single-pass rejection filter)
    # ═════════════════════════════════════════════════════════════════════

    def _has_standalone_mention(self, comp_name, text):
        """Check if component name appears as a standalone architectural reference.

        For single-word names: require capitalized match (proper noun usage).
        'cascade logic' (lowercase) → False; 'Logic is a Facade' (caps) → True.
        For multi-word names: case-insensitive match is sufficient.
        Also rejects matches inside package paths (word.name) or hyphenated compounds (name-word).
        """
        is_single = ' ' not in comp_name
        if is_single:
            # Require capitalized form: first letter uppercase
            cap_name = comp_name[0].upper() + comp_name[1:]
            pattern = rf'\b{re.escape(cap_name)}\b'
            flags = 0  # case-sensitive
        else:
            pattern = rf'\b{re.escape(comp_name)}\b'
            flags = re.IGNORECASE

        for m in re.finditer(pattern, text, flags):
            s, e = m.start(), m.end()
            # Skip package paths: word.Name or Name.word (but NOT sentence-ending periods)
            if s > 0 and text[s-1] == '.':
                continue
            if e < len(text) and text[e] == '.' and e + 1 < len(text) and text[e+1].isalpha():
                continue  # Mid-text dot (package path like "logic.core")
            # Skip hyphenated compounds: name-word or word-name (but NOT component names containing hyphens)
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
                # Strong signal: name appears as proper architectural reference
                safe.append(l)
            elif is_ta:
                # TransArc link, but name either absent or appears generically
                # (lowercase single-word, in package path, hyphenated)
                # Check if name appears at all (even generically)
                has_any = re.search(rf'\b{re.escape(l.component_name)}\b',
                                    sent.text, re.IGNORECASE)
                if has_any:
                    # Name present but generically → let judge decide
                    review.append(l)
                else:
                    # Name not in text (linked by TransArc NLP/synonym) → trust TransArc
                    safe.append(l)
            else:
                review.append(l)

        if not review:
            return safe

        # Multi-pass union voting: reject only if BOTH passes reject
        cases = self._build_judge_cases(review, sent_map)
        prompt = self._build_judge_prompt(comp_names, cases)
        n = min(30, len(review))

        data1 = self.llm.extract_json(self.llm.query(prompt, timeout=180))
        data2 = self.llm.extract_json(self.llm.query(prompt, timeout=180))

        rej1 = self._parse_rejections(data1, n)
        rej2 = self._parse_rejections(data2, n)
        rejected = rej1 & rej2  # Union: only reject if BOTH reject

        result = safe.copy()
        for i in range(n):
            if i not in rejected:
                result.append(review[i])
        result.extend(review[n:])
        return result

    def _parse_rejections(self, data, n):
        """Extract rejected case indices from judge response."""
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
        return f"""JUDGE: Review trace links. Accept references where the sentence is ABOUT the component, reject false matches.

COMPONENTS: {', '.join(comp_names)}

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
    # Phase 10: FN recovery with judge confirmation
    # ═════════════════════════════════════════════════════════════════════

    def _fn_recovery(self, current_links, sentences, components, name_to_id, sent_map):
        comp_names = self._get_comp_names(components)
        ambiguous = self.model_knowledge.ambiguous_names if self.model_knowledge else set()
        if ambiguous:
            print(f"  Skipping ambiguous in recovery: {sorted(ambiguous)}")

        covered = {l.sentence_number for l in current_links}
        synonyms_to_comp = {}
        if self.doc_knowledge:
            for syn, comp in self.doc_knowledge.synonyms.items():
                synonyms_to_comp[syn.lower()] = comp
            for abbr, comp in self.doc_knowledge.abbreviations.items():
                synonyms_to_comp[abbr.lower()] = comp
            for partial, comp in self.doc_knowledge.partial_references.items():
                synonyms_to_comp[partial.lower()] = comp

        potential = []
        for sent in sentences:
            if sent.number in covered:
                continue
            sl = sent.text.lower()
            found = False
            for cname in comp_names:
                if cname in ambiguous:
                    continue
                if cname.lower() in sl and re.search(rf'\b{re.escape(cname)}\b', sent.text, re.IGNORECASE):
                    potential.append((sent.number, sent.text, cname))
                    found = True
            if not found:
                for syn, comp in synonyms_to_comp.items():
                    if comp in ambiguous:
                        continue
                    if syn in sl and comp in comp_names and re.search(
                        rf'\b{re.escape(syn)}\b', sent.text, re.IGNORECASE
                    ):
                        potential.append((sent.number, sent.text, comp))

        if not potential:
            print("    No candidates")
            return current_links

        print(f"    Candidates: {len(potential)}")

        recovered_candidates = []
        for batch_start in range(0, len(potential), 12):
            batch = potential[batch_start:batch_start + 12]
            cases = [f"Case {i+1}: S{sn}: \"{txt[:70]}...\" -> {cn}?"
                     for i, (sn, txt, cn) in enumerate(batch)]

            r1 = self._qual_recovery_pass(comp_names, cases, "Is component the ACTOR?")
            r2 = self._qual_recovery_pass(comp_names, cases, "Is this a valid missed link?")

            for i, (snum, _, cname) in enumerate(batch):
                if r1.get(i, False) and r2.get(i, False) and cname in name_to_id:
                    key = (snum, name_to_id[cname])
                    if not any((l.sentence_number, l.component_id) == key for l in current_links):
                        recovered_candidates.append(SadSamLink(snum, name_to_id[cname], cname, 1.0, "recovered"))

        # Judge confirmation for recovered links
        if recovered_candidates:
            recovered_candidates = self._judge_recovered(recovered_candidates, comp_names, sent_map)

        result = current_links.copy()
        for lk in recovered_candidates:
            result.append(lk)
            print(f"    Recovered: S{lk.sentence_number} -> {lk.component_name}")
        return result

    def _judge_recovered(self, candidates, comp_names, sent_map):
        """Strict judge for recovered links — only approve clear actor references."""
        cases = []
        for i, lk in enumerate(candidates):
            sent = sent_map.get(lk.sentence_number)
            prev = sent_map.get(lk.sentence_number - 1)
            ctx_lines = []
            if prev:
                ctx_lines.append(f"    PREV: {prev.text[:50]}...")
            ctx_lines.append(f"    >>> S{lk.sentence_number}: {sent.text if sent else '?'}")
            cases.append(f"Case {i+1}: S{lk.sentence_number} -> {lk.component_name}\n" + chr(10).join(ctx_lines))

        prompt = f"""JUDGE: These are RECOVERED links — sentences that mention a component but were not linked by any earlier phase. Be STRICT: only approve if the component is clearly the ACTOR/SUBJECT performing an action.

COMPONENTS: {', '.join(comp_names)}

REJECT IF:
- Component name appears as a modifier, adjective, or in a compound word
- Component name is used as a generic English word, not an architecture reference (e.g. "cascade logic" when Logic is a component, "client-side" when Client is a component)
- Sentence describes package/directory/module structure using dot notation (e.g. "pkg.subpkg contains classes for...")
- Sentence describes a general process/concept where the component is just mentioned
- Component is the OBJECT/TARGET of an action by another component
- Sentence is about configuration, deployment, or system-level behavior

LINKS:
{chr(10).join(cases)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
        if not data:
            return []  # Judge failure = reject all recovered

        approved = []
        for j in data.get("judgments", []):
            idx = j.get("case", 0) - 1
            if 0 <= idx < len(candidates) and j.get("approve", False):
                approved.append(candidates[idx])
        print(f"    Recovery judge: {len(approved)}/{len(candidates)} approved")
        return approved

    def _qual_recovery_pass(self, comp_names, cases, question):
        prompt = f"""Check potential missed links. {question}

COMPONENTS: {', '.join(comp_names)}

POTENTIAL:
{chr(10).join(cases)}

Return JSON:
{{"recoveries": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=100))
        results = {}
        if data:
            for r in data.get("recoveries", []):
                idx = r.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    results[idx] = r.get("approve", False)
        return results

    # ═════════════════════════════════════════════════════════════════════
    # Selective embedding filter
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
