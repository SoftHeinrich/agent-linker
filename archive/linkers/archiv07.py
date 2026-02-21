"""AgentLinker V3: Qualitative pipeline with semantic filtering and full logging.

Self-contained linker consolidating V2 improvements:
- All qualitative (no numeric confidence thresholds)
- Source-priority deduplication
- Adaptive coref: discourse (small docs) vs debate (complex docs)
- Adaptive implicit: enabled (small docs) vs skipped (complex docs)
- Ambiguous-aware FN recovery
- Post-hoc semantic filter options: embedding, tfidf, lexical
- Per-phase JSON logging for analysis
- Phase 9 reviews ambiguous TransArc links through judge

Constructor options:
    post_filter: "none" (default), "embedding", "tfidf", "lexical",
                 "selective" (embedding on ambiguous non-transarc links),
                 "selective_all" (embedding on ALL non-transarc links)
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


# ─── Common English words that may also be component names ───────────────
_COMMON_WORDS = {
    "common", "logic", "storage", "client", "server", "test", "driver",
    "core", "web", "app", "apps", "ui", "db", "api", "model", "view",
    "controller", "service", "registry", "gateway", "proxy", "cache",
    "queue", "bus", "hub", "bridge", "adapter", "facade", "factory",
    "builder", "observer", "listener", "handler", "manager", "provider",
    "consumer", "producer", "worker", "scheduler", "monitor", "logger",
    "parser", "validator", "converter", "mapper", "router", "filter",
    "interceptor", "middleware", "plugin", "extension", "module",
    "component", "entity", "repository", "store", "persistence",
    "preferences", "recommender",
}


# ═════════════════════════════════════════════════════════════════════════
# Semantic Filters (no training required)
# ═════════════════════════════════════════════════════════════════════════

class _LexicalFilter:
    """Rule-based word-sense disambiguation via capitalization and syntax."""

    GENERIC_PREFIXES = {
        "cascade", "business", "application", "core", "main",
        "internal", "external", "basic", "minimal", "simple",
        "complex", "general", "common", "shared", "global",
        "access", "data", "file", "user", "system",
    }
    ARCH_SUFFIXES = {
        "component", "module", "service", "server", "client", "layer",
        "subsystem", "package", "handles", "processes", "manages",
        "provides", "receives", "sends", "stores", "retrieves",
    }

    def __init__(self, components):
        self._ambiguous = set()
        for c in components:
            if ' ' not in c.name and '-' not in c.name and len(c.name) <= 10:
                if c.name.lower() in _COMMON_WORDS:
                    self._ambiguous.add(c.name)

    def is_architectural(self, text: str, comp_name: str) -> bool:
        if comp_name not in self._ambiguous:
            return True
        # Capitalized occurrence?
        if re.search(rf'\b{re.escape(comp_name)}\b', text):
            m = re.search(rf'(\w+)\s+{re.escape(comp_name)}\b', text)
            if m and m.group(1).lower() in self.GENERIC_PREFIXES:
                return False
            return True
        # Only lowercase → check for arch suffix
        m = re.search(rf'\b{re.escape(comp_name.lower())}\s+(\w+)', text.lower())
        if m and m.group(1) in self.ARCH_SUFFIXES:
            return True
        return False


class _TfidfFilter:
    """TF-IDF context similarity between sentence and component profile."""

    def __init__(self, components, sentences):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        self._cosine = cosine_similarity

        comp_texts = defaultdict(list)
        for sent in sentences:
            tl = sent.text.lower()
            for c in components:
                if c.name.lower() in tl and re.search(rf'\b{re.escape(c.name)}\b', sent.text, re.IGNORECASE):
                    comp_texts[c.name].append(sent.text)

        docs, self._idx = [], {}
        for cname, texts in comp_texts.items():
            self._idx[cname] = len(docs)
            docs.append(" ".join(texts))

        self._ok = bool(docs)
        if self._ok:
            self._vec = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
            self._mat = self._vec.fit_transform(docs)

    def similarity(self, text: str, comp_name: str) -> float:
        if not self._ok or comp_name not in self._idx:
            return 0.5
        sv = self._vec.transform([text])
        return float(self._cosine(sv, self._mat[self._idx[comp_name]])[0, 0])


class _EmbeddingFilter:
    """Sentence-transformer cosine similarity filter."""

    def __init__(self, components, sentences, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self._np = np

        print(f"    Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)

        comp_texts = defaultdict(list)
        for sent in sentences:
            tl = sent.text.lower()
            for c in components:
                if c.name.lower() in tl and re.search(rf'\b{re.escape(c.name)}\b', sent.text, re.IGNORECASE):
                    comp_texts[c.name].append(sent.text)

        self._comp_emb = {}
        for cname, texts in comp_texts.items():
            if texts:
                self._comp_emb[cname] = np.mean(self._model.encode(texts), axis=0)

        self._sent_emb = {}
        if sentences:
            all_emb = self._model.encode([s.text for s in sentences])
            for s, e in zip(sentences, all_emb):
                self._sent_emb[s.number] = e

    def similarity(self, snum: int, comp_name: str) -> float:
        if comp_name not in self._comp_emb or snum not in self._sent_emb:
            return 0.5
        a, b = self._sent_emb[snum], self._comp_emb[comp_name]
        return float(self._np.dot(a, b) / (self._np.linalg.norm(a) * self._np.linalg.norm(b) + 1e-8))


# ═════════════════════════════════════════════════════════════════════════
# AgentLinkerV3
# ═════════════════════════════════════════════════════════════════════════

class AgentLinkerV3(AgentLinker):
    """Qualitative pipeline with semantic filtering and full phase logging."""

    SOURCE_PRIORITY = {
        "transarc": 5, "validated": 4, "entity": 3,
        "coreference": 2, "implicit": 1, "recovered": 0,
    }

    def __init__(self, backend: Optional[LLMBackend] = None, post_filter: str = "none"):
        # Default to Claude backend with sonnet model
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        super().__init__(backend=backend or LLMBackend.CLAUDE)
        self.post_filter = post_filter
        self._phase_log = []
        self._semantic_filter = None
        print("AgentLinkerV3: Qualitative + semantic filters + full logging")
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
        path = os.path.join(log_dir, f"v3_{ds}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w") as f:
            json.dump(self._phase_log, f, indent=2, default=str)
        print(f"  Phase log saved: {path}")

    # ── Complexity check ─────────────────────────────────────────────────

    def _is_complex_doc(self) -> bool:
        """High sentences-per-component → more ambiguity in coref/implicit."""
        if not self.doc_profile:
            return False
        return self.doc_profile.sentence_count / max(1, self.doc_profile.component_count) > 10

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

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        # ── Semantic filter init (before pipeline, uses raw data) ────────
        if self.post_filter != "none":
            print(f"\n[Semantic Filter Init] {self.post_filter}")
            self._init_filter(components, sentences)

        # ── Phase 0: Document profile ────────────────────────────────────
        print("\n[Phase 0] Document Profile")
        self.doc_profile = self._learn_document_profile(sentences, components)
        spc = len(sentences) / max(1, len(components))
        print(f"  Stats: {spc:.1f} sents/comp, {self.doc_profile.pronoun_ratio:.0%} pronouns")
        print(f"  Complex: {self._is_complex_doc()}")
        self.thresholds = LearnedThresholds(0, 0, 0, 0, "qualitative", 0)
        self._log("phase_0", {"sents": len(sentences), "comps": len(components)},
                  {"spc": spc, "pronoun_ratio": self.doc_profile.pronoun_ratio,
                   "complex": self._is_complex_doc()})

        # ── Phase 1: Model analysis ──────────────────────────────────────
        print("\n[Phase 1] Model Structure")
        self.model_knowledge = self._analyze_model(components)
        print(f"  Architectural: {len(self.model_knowledge.architectural_names)}")
        print(f"  Ambiguous: {self.model_knowledge.ambiguous_names}")
        self._log("phase_1", {"components": [c.name for c in components]},
                  {"architectural": sorted(self.model_knowledge.architectural_names),
                   "ambiguous": sorted(self.model_knowledge.ambiguous_names)})

        # ── Phase 2: Pattern learning (debate) ───────────────────────────
        print("\n[Phase 2] Pattern Learning")
        self.learned_patterns = self._learn_patterns_with_debate(sentences, components)
        print(f"  Subprocess terms: {len(self.learned_patterns.subprocess_terms)}")
        self._log("phase_2", {}, {"subprocess": sorted(self.learned_patterns.subprocess_terms)})

        # ── Phase 3: Document knowledge (judge) ──────────────────────────
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

        # ── Phase 4: TransArc baseline ───────────────────────────────────
        print("\n[Phase 4] TransArc")
        transarc_links = self._process_transarc(transarc_csv, id_to_name, sent_map, name_to_id)
        transarc_set = {(l.sentence_number, l.component_id) for l in transarc_links}
        print(f"  Links: {len(transarc_links)}")
        self._log("phase_4", {"csv": transarc_csv}, {"count": len(transarc_links)}, transarc_links)

        # ── Phase 5: Entity extraction ───────────────────────────────────
        print("\n[Phase 5] Entity Extraction")
        candidates = self._extract_entities(sentences, components, name_to_id, sent_map)
        print(f"  Candidates: {len(candidates)}")
        self._log("phase_5", {}, {"count": len(candidates)},
                  [SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source) for c in candidates])

        # ── Phase 6: Self-consistency validation ─────────────────────────
        print("\n[Phase 6] Validation")
        validated = self._validate_with_self_consistency(candidates, components, sent_map)
        print(f"  Validated: {len(validated)} (of {len(candidates)})")
        self._log("phase_6", {"candidates": len(candidates)},
                  {"validated": len(validated), "rejected": len(candidates) - len(validated)},
                  [SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source) for c in validated])

        # ── Phase 7: Coreference (adaptive: discourse vs debate) ────────
        print("\n[Phase 7] Coreference")
        discourse_model = self._build_discourse_model(sentences, components, name_to_id)
        coref_links = self._resolve_coreferences_with_discourse(
            sentences, components, name_to_id, sent_map, discourse_model
        )
        print(f"  Coref links: {len(coref_links)}")
        self._log("phase_7", {"method": "debate" if self._is_complex_doc() else "discourse"},
                  {"count": len(coref_links)}, coref_links)

        # ── Phase 8: Implicit references (adaptive: skip for complex) ───
        print("\n[Phase 8] Implicit References")
        existing = (transarc_set
                    | {(c.sentence_number, c.component_id) for c in validated}
                    | {(l.sentence_number, l.component_id) for l in coref_links})

        if self._is_complex_doc():
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
        self._log("phase_8", {"existing": len(existing), "complex": self._is_complex_doc()},
                  {"count": len(implicit_links)}, implicit_links)

        # ── Combine + deduplicate by source priority ─────────────────────
        entity_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
            for c in validated
        ]
        all_links = transarc_links + entity_links + coref_links + implicit_links
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

        # ── Phase 9: Judge review (ambiguous TransArc reviewed too) ──────
        print("\n[Phase 9] Judge Review")
        reviewed = self._agent_judge_review(preliminary, sentences, components, sent_map, transarc_set)
        rejected = [l for l in preliminary if (l.sentence_number, l.component_id)
                    not in {(r.sentence_number, r.component_id) for r in reviewed}]
        print(f"  Approved: {len(reviewed)} (rejected {len(rejected)})")
        self._log("phase_9", {"input": len(preliminary)},
                  {"approved": len(reviewed), "rejected": len(rejected)}, reviewed)
        if rejected:
            self._log("phase_9_rejected", {}, {"count": len(rejected)}, rejected)

        # ── Phase 10: FN recovery (skip ambiguous names) ─────────────────
        print("\n[Phase 10] FN Recovery")
        final = self._adaptive_fn_recovery(reviewed, sentences, components, name_to_id, sent_map)
        recovered = [l for l in final if l.source == "recovered"]
        print(f"  Final: {len(final)} (+{len(recovered)} recovered)")
        self._log("phase_10", {"ambiguous_skipped": sorted(
            self.model_knowledge.ambiguous_names if self.model_knowledge else [])},
            {"final": len(final), "recovered": len(recovered)}, recovered or None)

        # ── Post-filter: semantic ────────────────────────────────────────
        if self._semantic_filter is not None:
            before = len(final)
            final = self._apply_filter(final, sent_map)
            removed = before - len(final)
            print(f"  Semantic filter ({self.post_filter}): removed {removed}")
            self._log("post_filter", {"before": before, "type": self.post_filter},
                      {"after": len(final), "removed": removed}, final)

        # ── Save log ─────────────────────────────────────────────────────
        self._log("summary", {"total_time_s": round(time.time() - t0, 1)},
                  {"final": len(final)}, final)
        self._save_log(text_path)

        print(f"\nFinal: {len(final)} links")
        return final

    # ═════════════════════════════════════════════════════════════════════
    # Phase 0: Document profile (no LLM, no strictness)
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

        cases = []
        for i, c in enumerate(needs[:25]):
            prev = sent_map.get(c.sentence_number - 1)
            p = f"[prev: {prev.text[:35]}...] " if prev else ""
            cases.append(f"Case {i+1}: \"{c.matched_text}\" -> {c.component_name}\n  {p}\"{c.sentence_text}\"")

        r1 = self._qual_validation_pass(comp_names, ctx, cases, "Focus on ACTOR")
        r2 = self._qual_validation_pass(comp_names, ctx, cases, "Focus on DIRECT reference")

        validated = []
        for i, c in enumerate(needs[:25]):
            if r1.get(i, False) and r2.get(i, False):  # Unanimous
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
    # Phase 7: Adaptive coreference (discourse for small, debate for complex)
    # ═════════════════════════════════════════════════════════════════════

    def _resolve_coreferences_with_discourse(
        self, sentences: list[Sentence], components: list, name_to_id: dict,
        sent_map: dict, discourse_model: dict
    ) -> list[SadSamLink]:

        if self._is_complex_doc():
            print(f"  Mode: debate (complex, {len(sentences)} sents)")
            return self._coref_debate(sentences, components, name_to_id, sent_map)
        else:
            print(f"  Mode: discourse ({len(sentences)} sents)")
            return self._coref_discourse(sentences, components, name_to_id, sent_map, discourse_model)

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

            # Pass 2: Judge
            ptxt = [f"S{r.get('sentence')}: \"{r.get('pronoun', '?')}\" -> {r.get('component')} ({r.get('reasoning', '')[:40]})"
                    for r in proposed[:12]]

            prompt2 = f"""JUDGE: Validate these pronoun-to-component resolutions.

COMPONENTS: {', '.join(comp_names)}

PROPOSALS:
{chr(10).join(ptxt)}

CONTEXT:
{chr(10).join(doc_lines[:15])}

Reject if pronoun might refer to something else, or reference is to a subprocess.

Return JSON:
{{"judgments": [{{"sentence": N, "approve": true/false}}]}}
JSON only:"""

            data2 = self.llm.extract_json(self.llm.query(prompt2, timeout=100))
            judgments = {}
            if data2:
                for j in data2.get("judgments", []):
                    judgments[j.get("sentence")] = j.get("approve", False)

            for res in proposed:
                snum, comp = res.get("sentence"), res.get("component")
                if str(res.get("certainty", "low")).lower() != "high":
                    continue
                if not (snum and comp and comp in name_to_id):
                    continue
                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue
                if snum in judgments and not judgments[snum]:
                    continue  # Judge rejected
                sent = sent_map.get(snum)
                if sent and self.learned_patterns and self.learned_patterns.is_subprocess(sent.text):
                    continue
                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, 1.0, "coreference"))

        return all_coref

    # ═════════════════════════════════════════════════════════════════════
    # Phase 8: Qualitative implicit reference detection
    # ═════════════════════════════════════════════════════════════════════

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
        cases = []
        for i, link in enumerate(implicit_links):
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
            return []
        validated = []
        for v in data.get("validations", []):
            idx = v.get("case", 0) - 1
            if 0 <= idx < len(implicit_links) and v.get("approve", False):
                validated.append(implicit_links[idx])
        return validated

    # ═════════════════════════════════════════════════════════════════════
    # Phase 9: Judge review (ambiguous TransArc links reviewed too)
    # ═════════════════════════════════════════════════════════════════════

    def _agent_judge_review(self, links, sentences, components, sent_map, transarc_set):
        if len(links) < 5:
            return links

        comp_names = self._get_comp_names(components)
        ambiguous = self.model_knowledge.ambiguous_names if self.model_knowledge else set()

        # Non-ambiguous TransArc auto-approved; ambiguous + all others reviewed
        safe, review = [], []
        for l in links:
            is_ta = (l.sentence_number, l.component_id) in transarc_set
            if is_ta and l.component_name not in ambiguous:
                safe.append(l)
            else:
                review.append(l)

        if not review:
            return safe

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

        prompt = f"""JUDGE: Review trace links. Accept clear references, reject ambiguous ones.

COMPONENTS: {', '.join(comp_names)}

VALID IF: Component is the ACTOR/SUBJECT performing the action described
REJECT IF: Effect/result, subprocess, package structure, general concept

SOURCE-SPECIFIC RULES:
- "transarc" with ambiguous names: Is it an ARCHITECTURE REFERENCE or GENERIC ENGLISH WORD? "cascade logic" = generic; "the Logic component handles..." = architectural.
- "implicit": EXTRA SKEPTICISM. Reject unless UNAMBIGUOUSLY the subject.
- "coreference": Verify the pronoun CLEARLY refers to the claimed component.

LINKS:
{chr(10).join(cases)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=180))
        result = safe.copy()
        n = min(30, len(review))
        if data:
            judged = set()
            for j in data.get("judgments", []):
                idx = j.get("case", 0) - 1
                if 0 <= idx < n:
                    judged.add(idx)
                    if j.get("approve", False):
                        result.append(review[idx])
            for i in range(n):
                if i not in judged:
                    result.append(review[i])
            result.extend(review[n:])
        else:
            result.extend(review)
        return result

    # ═════════════════════════════════════════════════════════════════════
    # Phase 10: FN recovery (skip ambiguous names)
    # ═════════════════════════════════════════════════════════════════════

    def _adaptive_fn_recovery(self, current_links, sentences, components, name_to_id, sent_map):
        comp_names = self._get_comp_names(components)
        ambiguous = self.model_knowledge.ambiguous_names if self.model_knowledge else set()
        # Also skip common English words not caught by Phase 1
        extra_ambig = {cn for cn in comp_names if cn.lower() in _COMMON_WORDS}
        skip_names = ambiguous | extra_ambig
        if skip_names:
            print(f"  Skipping ambiguous in recovery: {sorted(skip_names)}")

        covered = {l.sentence_number for l in current_links}
        synonyms_to_comp = {}
        if self.doc_knowledge:
            for syn, comp in self.doc_knowledge.synonyms.items():
                synonyms_to_comp[syn.lower()] = comp
            for abbr, comp in self.doc_knowledge.abbreviations.items():
                synonyms_to_comp[abbr.lower()] = comp

        potential = []
        for sent in sentences:
            if sent.number in covered:
                continue
            sl = sent.text.lower()
            found = False
            for cname in comp_names:
                if cname in skip_names:
                    continue
                if cname.lower() in sl and re.search(rf'\b{re.escape(cname)}\b', sent.text, re.IGNORECASE):
                    potential.append((sent.number, sent.text, cname))
                    found = True
            if not found:
                for syn, comp in synonyms_to_comp.items():
                    if comp in skip_names:
                        continue
                    if syn in sl and comp in comp_names:
                        potential.append((sent.number, sent.text, comp))

        if not potential:
            print("    No candidates")
            return current_links

        print(f"    Candidates: {len(potential)}")
        cases = [f"Case {i+1}: S{sn}: \"{txt[:70]}...\" -> {cn}?"
                 for i, (sn, txt, cn) in enumerate(potential[:12])]

        r1 = self._qual_recovery_pass(comp_names, cases, "Is component the ACTOR?")
        r2 = self._qual_recovery_pass(comp_names, cases, "Is this a valid missed link?")

        result = current_links.copy()
        for i, (snum, _, cname) in enumerate(potential[:12]):
            if r1.get(i, False) and r2.get(i, False) and cname in name_to_id:
                key = (snum, name_to_id[cname])
                if not any((l.sentence_number, l.component_id) == key for l in result):
                    result.append(SadSamLink(snum, name_to_id[cname], cname, 1.0, "recovered"))
                    print(f"    Recovered: S{snum} -> {cname}")
        return result

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
    # Semantic filter
    # ═════════════════════════════════════════════════════════════════════

    def _init_filter(self, components, sentences):
        if self.post_filter in ("embedding", "selective", "selective_all"):
            self._semantic_filter = _EmbeddingFilter(components, sentences)
        elif self.post_filter == "tfidf":
            self._semantic_filter = _TfidfFilter(components, sentences)
        elif self.post_filter == "lexical":
            self._semantic_filter = _LexicalFilter(components)

    def _apply_filter(self, links, sent_map):
        ambiguous = self.model_knowledge.ambiguous_names if self.model_knowledge else set()
        check_names = ambiguous | {l.component_name for l in links if l.component_name.lower() in _COMMON_WORDS}

        # selective_all: check ALL non-transarc links (not just ambiguous names)
        selective_all = self.post_filter == "selective_all"
        # selective: check only ambiguous non-transarc links
        selective = self.post_filter == "selective"
        # Sources trusted unconditionally
        trusted_sources = {"transarc"}

        if not selective_all and not check_names:
            return links

        result = []
        for lk in links:
            # Trust transarc links in selective modes
            if (selective or selective_all) and lk.source in trusted_sources:
                result.append(lk)
                continue

            # In non-selective_all mode, skip non-ambiguous components
            if not selective_all and lk.component_name not in check_names:
                result.append(lk)
                continue

            keep = True
            if isinstance(self._semantic_filter, _EmbeddingFilter):
                sim = self._semantic_filter.similarity(lk.sentence_number, lk.component_name)
                keep = sim >= 0.3
                if not keep:
                    print(f"    Embed rejected: S{lk.sentence_number} -> {lk.component_name} (sim={sim:.3f}, src={lk.source})")
            elif isinstance(self._semantic_filter, _TfidfFilter):
                sent = sent_map.get(lk.sentence_number)
                if sent:
                    sim = self._semantic_filter.similarity(sent.text, lk.component_name)
                    keep = sim >= 0.1
                    if not keep:
                        print(f"    TF-IDF rejected: S{lk.sentence_number} -> {lk.component_name} (sim={sim:.3f}, src={lk.source})")
            elif isinstance(self._semantic_filter, _LexicalFilter):
                sent = sent_map.get(lk.sentence_number)
                if sent:
                    keep = self._semantic_filter.is_architectural(sent.text, lk.component_name)
                    if not keep:
                        print(f"    Lexical rejected: S{lk.sentence_number} -> {lk.component_name} (src={lk.source})")
            if keep:
                result.append(lk)
        return result
