"""AgentLinker V2: Qualitative judgments, no numeric thresholds.

Extends AgentLinker with a key design change: all accept/reject decisions
use qualitative LLM judgments (approve/reject, certainty: high/low) and
vote agreement instead of numeric confidence thresholds.

Changes from AgentLinker:
- Phase 0: No threshold learning (saves 1 LLM call)
- Phase 6: Two-pass validation uses vote count only (≥1 approve → accept)
- Phase 7: Coreference uses "certainty: high/low" instead of numeric confidence
- Phase 8: Implicit detection uses "certainty: high/low"
- Phase 8 validation: Uses approve/reject only
- Phase 9: Also reviews TransArc links for ambiguous component names (not auto-approved)
- Phase 10: Two-pass recovery requires unanimous approval (both passes agree)
"""

import json
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
from .agent_linker import AgentLinker


class AgentLinkerV2(AgentLinker):
    """AgentLinker with qualitative judgments instead of numeric thresholds."""

    # Source priority for deduplication (higher = preferred)
    SOURCE_PRIORITY = {
        "transarc": 5, "validated": 4, "entity": 3,
        "coreference": 2, "implicit": 1, "recovered": 0,
    }

    def __init__(self, backend: Optional[LLMBackend] = None):
        super().__init__(backend=backend)
        self._phase_log = []
        print("AgentLinkerV2: Qualitative judgments, no numeric thresholds")

    def _log_phase(self, phase: str, input_summary: dict, output_summary: dict, links: list = None):
        """Log phase input/output for analysis."""
        entry = {
            "phase": phase,
            "timestamp": time.time(),
            "input": input_summary,
            "output": output_summary,
        }
        if links is not None:
            entry["links"] = [
                {"sentence": l.sentence_number, "component": l.component_name, "source": l.source}
                for l in links
            ]
        self._phase_log.append(entry)

    def _save_phase_log(self, text_path: str):
        """Save phase log to JSON file."""
        log_dir = os.environ.get("LLM_LOG_DIR", "./results/llm_logs")
        os.makedirs(log_dir, exist_ok=True)
        ds_name = os.path.splitext(os.path.basename(text_path))[0]
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"v2_phases_{ds_name}_{ts}.json")
        with open(log_path, "w") as f:
            json.dump(self._phase_log, f, indent=2, default=str)
        print(f"  Phase log: {log_path}")

    def link(self, text_path: str, model_path: str,
             transarc_csv: str = None) -> list[SadSamLink]:

        self._phase_log = []
        t_start = time.time()

        # Load data
        components = parse_pcm_repository(model_path)
        sentences = DocumentLoader.load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        id_to_name = {c.id: c.name for c in components}
        sent_map = DocumentLoader.build_sent_map(sentences)

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        # Phase 0: Document Analysis (no threshold learning)
        print("\n[Phase 0] Document Analysis")
        self.doc_profile = self._learn_document_profile(sentences, components)
        print(f"  Complexity: {self.doc_profile.complexity_score:.2f}")
        # Dummy thresholds for any code that reads them (e.g. ablation class)
        self.thresholds = LearnedThresholds(0, 0, 0, 0, "qualitative", 0)
        self._log_phase("phase_0_profile", {
            "sentences": len(sentences), "components": len(components),
        }, {
            "sents_per_comp": len(sentences) / max(1, len(components)),
            "pronoun_ratio": self.doc_profile.pronoun_ratio,
            "complexity": self.doc_profile.complexity_score,
        })

        # Phase 1: Model Structure Analysis
        print("\n[Phase 1] Model Structure Analysis")
        self.model_knowledge = self._analyze_model(components)
        print(f"  Architectural: {len(self.model_knowledge.architectural_names)}")
        print(f"  Ambiguous: {self.model_knowledge.ambiguous_names}")
        self._log_phase("phase_1_model", {
            "components": [c.name for c in components],
        }, {
            "architectural": sorted(self.model_knowledge.architectural_names),
            "ambiguous": sorted(self.model_knowledge.ambiguous_names),
        })

        # Phase 2: Pattern Learning (with Debate)
        print("\n[Phase 2] Pattern Learning (with Debate)")
        self.learned_patterns = self._learn_patterns_with_debate(sentences, components)
        print(f"  Subprocess terms: {len(self.learned_patterns.subprocess_terms)}")
        self._log_phase("phase_2_patterns", {}, {
            "subprocess_terms": sorted(self.learned_patterns.subprocess_terms),
            "action_indicators": self.learned_patterns.action_indicators[:5],
        })

        # Phase 3: Document Knowledge (Judge-Validated)
        print("\n[Phase 3] Document Knowledge (Judge-Validated)")
        self.doc_knowledge = self._learn_document_knowledge_with_judge(sentences, components)
        print(f"  Abbreviations: {len(self.doc_knowledge.abbreviations)}")
        print(f"  Synonyms: {len(self.doc_knowledge.synonyms)}")
        print(f"  Generic terms: {len(self.doc_knowledge.generic_terms)}")
        self._log_phase("phase_3_doc_knowledge", {}, {
            "abbreviations": self.doc_knowledge.abbreviations,
            "synonyms": self.doc_knowledge.synonyms,
            "generic_terms": list(self.doc_knowledge.generic_terms),
        })

        # Phase 4: TransArc Processing
        print("\n[Phase 4] TransArc Processing")
        transarc_links = self._process_transarc(transarc_csv, id_to_name, sent_map, name_to_id)
        transarc_set = {(l.sentence_number, l.component_id) for l in transarc_links}
        print(f"  TransArc links: {len(transarc_links)}")
        self._log_phase("phase_4_transarc", {
            "csv": transarc_csv,
        }, {"count": len(transarc_links)}, transarc_links)

        # Phase 5: Entity Extraction
        print("\n[Phase 5] Entity Extraction")
        entity_candidates = self._extract_entities(sentences, components, name_to_id, sent_map)
        print(f"  Candidates: {len(entity_candidates)}")
        self._log_phase("phase_5_entities", {}, {
            "count": len(entity_candidates),
        }, [SadSamLink(c.sentence_number, c.component_id, c.component_name, c.confidence, c.source) for c in entity_candidates])

        # Phase 6: Self-Consistency Validation (qualitative)
        print("\n[Phase 6] Self-Consistency Validation")
        validated = self._validate_with_self_consistency(entity_candidates, components, sent_map)
        print(f"  Validated: {len(validated)}")
        self._log_phase("phase_6_validation", {
            "candidates": len(entity_candidates),
        }, {
            "validated": len(validated),
            "rejected": len(entity_candidates) - len(validated),
        }, [SadSamLink(c.sentence_number, c.component_id, c.component_name, c.confidence, c.source) for c in validated])

        # Phase 7: Discourse-Aware Coreference (qualitative)
        print("\n[Phase 7] Discourse-Aware Coreference")
        discourse_model = self._build_discourse_model(sentences, components, name_to_id)
        coref_links = self._resolve_coreferences_with_discourse(
            sentences, components, name_to_id, sent_map, discourse_model
        )
        print(f"  Coreference: {len(coref_links)}")
        self._log_phase("phase_7_coreference", {
            "pronoun_sentences": sum(1 for s in sentences if self.PRONOUN_PATTERN.search(s.text)),
        }, {"count": len(coref_links)}, coref_links)

        # Phase 8: Conservative Implicit Reference Detection
        print("\n[Phase 8] Conservative Implicit Reference Detection")
        # Pass ALL existing links (not just TransArc) to avoid re-linking covered sentences
        existing_links = (transarc_set
                          | {(c.sentence_number, c.component_id) for c in validated}
                          | {(l.sentence_number, l.component_id) for l in coref_links})
        implicit_links = self._detect_implicit_references(
            sentences, components, name_to_id, sent_map, discourse_model, existing_links
        )
        print(f"  Implicit (before validation): {len(implicit_links)}")

        implicit_before = len(implicit_links)
        if implicit_links:
            implicit_links = self._validate_implicit_links(
                implicit_links, sentences, components, sent_map
            )
            print(f"  Implicit (after validation): {len(implicit_links)}")
        self._log_phase("phase_8_implicit", {
            "existing_links": len(existing_links),
        }, {
            "before_validation": implicit_before,
            "after_validation": len(implicit_links),
        }, implicit_links)

        # Convert candidates to links
        entity_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
            for c in validated
        ]

        # Combine all links — deduplicate by source priority (no numeric comparison)
        all_links = transarc_links + entity_links + coref_links + implicit_links
        link_map = {}
        for link in all_links:
            key = (link.sentence_number, link.component_id)
            if key not in link_map:
                link_map[key] = link
            else:
                existing_prio = self.SOURCE_PRIORITY.get(link_map[key].source, 0)
                new_prio = self.SOURCE_PRIORITY.get(link.source, 0)
                if new_prio > existing_prio:
                    link_map[key] = link

        preliminary = list(link_map.values())
        self._log_phase("dedup", {
            "all_links": len(all_links),
        }, {"after_dedup": len(preliminary)}, preliminary)

        # Phase 9: Source-Aware Agent-as-Judge Review
        print("\n[Phase 9] Source-Aware Agent-as-Judge Review")
        reviewed = self._agent_judge_review(preliminary, sentences, components, sent_map, transarc_set)
        rejected_p9 = [l for l in preliminary if (l.sentence_number, l.component_id) not in
                        {(r.sentence_number, r.component_id) for r in reviewed}]
        print(f"  After review: {len(reviewed)} (was {len(preliminary)})")
        self._log_phase("phase_9_judge", {
            "preliminary": len(preliminary),
        }, {
            "approved": len(reviewed),
            "rejected": len(rejected_p9),
        }, reviewed)
        if rejected_p9:
            self._log_phase("phase_9_rejected", {}, {
                "count": len(rejected_p9),
            }, rejected_p9)

        # Phase 10: FN Recovery (always runs — gated by unanimous vote, not strictness)
        print("\n[Phase 10] FN Recovery")
        final = self._adaptive_fn_recovery(reviewed, sentences, components, name_to_id, sent_map)
        recovered = [l for l in final if l.source == "recovered"]
        print(f"  After recovery: {len(final)} (was {len(reviewed)})")
        self._log_phase("phase_10_recovery", {
            "reviewed": len(reviewed),
            "ambiguous_skipped": sorted(self.model_knowledge.ambiguous_names) if self.model_knowledge else [],
        }, {
            "final": len(final),
            "recovered": len(recovered),
        }, recovered if recovered else None)

        # Save phase log
        self._log_phase("summary", {
            "total_time": time.time() - t_start,
        }, {
            "final_count": len(final),
            "by_source": {},
        }, final)
        self._save_phase_log(text_path)

        print(f"\nFinal: {len(final)} links")
        return final

    # =========================================================================
    # Phase 0: Document statistics only (no strictness categories)
    # =========================================================================

    def _learn_document_profile(self, sentences, components) -> DocumentProfile:
        """Compute document statistics without LLM call or strictness categories.

        V2 does not use strictness to differentiate behavior — all documents
        are processed with the same pipeline settings. Statistics are computed
        for informational purposes only.
        """
        import re as _re
        texts = [s.text for s in sentences]
        comp_names = [c.name for c in components]

        pronoun_pattern = r'\b(it|they|this|these|that|those|its|their)\b'
        pronoun_sentences = sum(1 for t in texts if _re.search(pronoun_pattern, t.lower()))
        pronoun_ratio = pronoun_sentences / len(sentences) if sentences else 0

        comp_mentions = sum(1 for t in texts for c in comp_names if c.lower() in t.lower())
        mention_density = comp_mentions / len(sentences) if sentences else 0

        sents_per_comp = len(sentences) / max(1, len(components))
        print(f"  Stats: {sents_per_comp:.1f} sents/comp, {pronoun_ratio:.0%} pronoun ratio")

        return DocumentProfile(
            sentence_count=len(sentences),
            component_count=len(components),
            pronoun_ratio=pronoun_ratio,
            technical_density=mention_density,
            component_mention_density=mention_density,
            complexity_score=min(1.0, sents_per_comp / 20),
            recommended_strictness="balanced",  # unused — V2 uses uniform behavior
        )

    # =========================================================================
    # Phase 7: Qualitative Coreference Resolution
    # =========================================================================

    def _resolve_coreferences_with_discourse(
        self, sentences: list[Sentence], components: list, name_to_id: dict,
        sent_map: dict, discourse_model: dict
    ) -> list[SadSamLink]:
        """Resolve coreferences using qualitative certainty instead of numeric confidence."""
        comp_names = self._get_comp_names(components)
        all_coref = []

        pronoun_sents = [s for s in sentences if self.PRONOUN_PATTERN.search(s.text)]

        batch_size = 12
        for batch_start in range(0, len(pronoun_sents), batch_size):
            batch = pronoun_sents[batch_start:batch_start + batch_size]

            cases = []
            for sent in batch:
                ctx = discourse_model.get(sent.number, DiscourseContext())

                prev_context = []
                for i in range(1, self.CONTEXT_WINDOW + 1):
                    prev = sent_map.get(sent.number - i)
                    if prev:
                        prev_context.append(f"S{prev.number}: {prev.text[:70]}")

                cases.append({
                    "sent": sent,
                    "ctx": ctx,
                    "prev": prev_context,
                    "likely": ctx.get_likely_referent()
                })

            prompt = f"""Resolve pronoun references using discourse context.

COMPONENTS: {', '.join(comp_names)}

For each sentence, identify what pronouns (it, they, this, etc.) refer to.
Use the discourse context to make accurate resolutions.

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
                certainty = str(res.get("certainty", "low")).lower()

                if not (comp and snum and comp in name_to_id):
                    continue

                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue

                # Qualitative filter: only accept "high" certainty
                if certainty != "high":
                    continue

                # Reject if subprocess
                sent = sent_map.get(snum)
                if sent and self.learned_patterns and self.learned_patterns.is_subprocess(sent.text):
                    continue

                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, 1.0, "coreference"))

        return all_coref

    # =========================================================================
    # Phase 8: Qualitative Implicit Reference Detection
    # =========================================================================

    def _detect_implicit_references(
        self, sentences: list[Sentence], components: list, name_to_id: dict,
        sent_map: dict, discourse_model: dict, existing_links: set[tuple[int, str]]
    ) -> list[SadSamLink]:
        """Detect implicit references with qualitative certainty."""
        comp_names = self._get_comp_names(components)
        implicit_links = []

        # Same pre-filters as AgentLinker (structural, not numeric)
        candidates = []
        for sent in sentences:
            text_lower = sent.text.lower()
            has_explicit = any(c.lower() in text_lower for c in comp_names)
            if has_explicit:
                continue

            ctx = discourse_model.get(sent.number, DiscourseContext())

            if not ctx.active_entity or not ctx.paragraph_topic:
                continue
            if ctx.active_entity != ctx.paragraph_topic:
                continue

            likely_comp = ctx.active_entity

            if (sent.number, name_to_id.get(likely_comp, '')) in existing_links:
                continue

            last_mention_dist = self._distance_to_last_mention(
                ctx.recent_mentions, likely_comp, sent.number
            )
            if last_mention_dist is None or last_mention_dist > 2:
                continue

            candidates.append((sent, ctx, likely_comp))

        if not candidates:
            return []

        print(f"    Candidates after filtering: {len(candidates)}")

        batch_size = 5
        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start:batch_start + batch_size]

            prompt = f"""Detect implicit component references. Be VERY conservative.

COMPONENTS: {', '.join(comp_names)}

For each sentence, determine if it implicitly describes the suggested component
without explicitly naming it. Precision is more important than recall.

"""
            for i, (sent, ctx, likely) in enumerate(batch):
                prev = sent_map.get(sent.number - 1)
                prev_text = prev.text[:70] if prev else "N/A"

                prompt += f"--- Case {i+1}: S{sent.number} ---\n"
                prompt += f"Previous: {prev_text}\n"
                prompt += f"Sentence: {sent.text}\n"
                prompt += f"Context: {ctx.get_context_summary(sent.number)}\n"
                prompt += f"Likely component: {likely}\n\n"

            prompt += """Return JSON:
{"detections": [{"case": 1, "sentence": N, "component": "Name", "is_implicit": true/false, "certainty": "high or low", "reasoning": "why"}]}

An implicit reference is ONLY when:
- The sentence continues describing the EXACT SAME component without re-naming it
- The action described can ONLY be performed by this specific component
- Subject is omitted but implied from IMMEDIATE context (previous 1-2 sentences)

NOT implicit (DO NOT link) if:
- Sentence describes a result/effect, not component action
- Action could be performed by multiple components
- Sentence introduces a new topic
- Sentence describes general processes or workflows
- Sentence describes algorithms, data flows, or transformations
- Sentence describes configuration, parameters, or settings
- Sentence describes concepts (hashing, scheduling, caching, routing, etc.)
- Sentence describes system-wide behavior not specific to one component

Only return is_implicit=true with certainty="high". When in doubt, set is_implicit to false.

JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
            if not data:
                continue

            for det in data.get("detections", []):
                if not det.get("is_implicit"):
                    continue

                comp = det.get("component")
                snum = det.get("sentence")
                certainty = str(det.get("certainty", "low")).lower()

                if not (comp and snum and comp in name_to_id):
                    continue

                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue

                # Qualitative: only accept "high" certainty
                if certainty != "high":
                    continue

                implicit_links.append(SadSamLink(snum, name_to_id[comp], comp, 1.0, "implicit"))

        return implicit_links

    # =========================================================================
    # Phase 8 Validation: Qualitative
    # =========================================================================

    def _validate_implicit_links(
        self, implicit_links: list[SadSamLink], sentences: list[Sentence],
        components: list, sent_map: dict
    ) -> list[SadSamLink]:
        """Validate implicit links with qualitative approve/reject."""
        if not implicit_links:
            return []

        comp_names = self._get_comp_names(components)

        cases = []
        for i, link in enumerate(implicit_links):
            sent = sent_map.get(link.sentence_number)
            prev1 = sent_map.get(link.sentence_number - 1)
            prev2 = sent_map.get(link.sentence_number - 2)
            context_lines = []
            if prev2:
                context_lines.append(f"S{prev2.number}: {prev2.text[:60]}")
            if prev1:
                context_lines.append(f"S{prev1.number}: {prev1.text[:60]}")
            context_lines.append(f">>> S{link.sentence_number}: {sent.text if sent else '?'}")
            cases.append(
                f"Case {i+1}: S{link.sentence_number} -> {link.component_name}\n"
                + "\n".join(f"    {line}" for line in context_lines)
            )

        prompt = f"""VALIDATION: Independently verify these implicit component references.

COMPONENTS: {', '.join(comp_names)}

For each case, the sentence was flagged as implicitly referring to the named component.
Verify this is correct. Be VERY strict — only approve if the sentence UNAMBIGUOUSLY
continues describing that specific component's behavior.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false, "reasoning": "brief"}}]}}

Reject if:
- The action could reasonably be performed by any component
- The sentence is about system behavior in general
- The connection to the component is tenuous
- You are not highly confident

JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=150))
        if not data:
            print("    Implicit validation failed — rejecting all implicit links")
            return []

        validated = []
        for v in data.get("validations", []):
            idx = v.get("case", 0) - 1
            if 0 <= idx < len(implicit_links):
                if v.get("approve", False):
                    validated.append(implicit_links[idx])

        return validated

    # =========================================================================
    # Phase 6: Qualitative Self-Consistency Validation
    # =========================================================================

    def _validate_with_self_consistency(self, candidates: list[CandidateLink], components, sent_map) -> list[CandidateLink]:
        if not candidates:
            return []

        comp_names = self._get_comp_names(components)
        needs_validation = [c for c in candidates if c.needs_validation]
        direct = [c for c in candidates if not c.needs_validation]

        if not needs_validation:
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
        for i, c in enumerate(needs_validation[:25]):
            prev = sent_map.get(c.sentence_number - 1)
            p = f"[prev: {prev.text[:35]}...] " if prev else ""
            cases.append(f"Case {i+1}: \"{c.matched_text}\" -> {c.component_name}\n  {p}\"{c.sentence_text}\"")

        results_1 = self._run_validation_pass_qualitative(comp_names, ctx, cases, "Focus on ACTOR")
        results_2 = self._run_validation_pass_qualitative(comp_names, ctx, cases, "Focus on DIRECT reference")

        # Larger docs need stricter agreement (more noise, more partial matches)
        # Always require unanimous agreement (both passes approve)
        min_votes = 2

        validated = []
        for i, c in enumerate(needs_validation[:25]):
            approvals = 0
            if i in results_1 and results_1[i]:
                approvals += 1
            if i in results_2 and results_2[i]:
                approvals += 1

            if approvals >= min_votes:
                c.confidence = 1.0
                c.source = "validated"
                validated.append(c)

        return direct + validated

    def _run_validation_pass_qualitative(self, comp_names, ctx, cases, focus):
        """Validation pass returning approve/reject only."""
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

    # =========================================================================
    # Phase 10: Qualitative FN Recovery
    # =========================================================================

    def _adaptive_fn_recovery(self, current_links, sentences, components, name_to_id, sent_map) -> list[SadSamLink]:
        comp_names = self._get_comp_names(components)
        ambiguous = self.model_knowledge.ambiguous_names if self.model_knowledge else set()
        covered_sents = {l.sentence_number for l in current_links}

        synonyms_to_comp = {}
        if self.doc_knowledge:
            for syn, comp in self.doc_knowledge.synonyms.items():
                synonyms_to_comp[syn.lower()] = comp
            for abbr, comp in self.doc_knowledge.abbreviations.items():
                synonyms_to_comp[abbr.lower()] = comp

        potential_fns = []
        for sent in sentences:
            if sent.number in covered_sents:
                continue
            sent_lower = sent.text.lower()
            found = False

            # Check all matching components (not just first), skip ambiguous names
            for cname in comp_names:
                if cname in ambiguous:
                    continue
                if cname.lower() in sent_lower:
                    pattern = rf'\b{re.escape(cname)}\b'
                    if re.search(pattern, sent.text, re.IGNORECASE):
                        potential_fns.append((sent.number, sent.text, cname))
                        found = True

            if not found:
                for syn, comp in synonyms_to_comp.items():
                    if comp in ambiguous:
                        continue
                    if syn in sent_lower and comp in comp_names:
                        potential_fns.append((sent.number, sent.text, comp))
                        # Don't break — check all synonyms too

        if not potential_fns:
            print("    No potential FNs found")
            return current_links

        print(f"    Checking {len(potential_fns)} potential FNs")

        cases = [f"Case {i+1}: S{sn}: \"{txt[:70]}...\" -> {cn}?" for i, (sn, txt, cn) in enumerate(potential_fns[:12])]

        results_1 = self._run_recovery_pass_qualitative(comp_names, cases, "Is component the ACTOR?")
        results_2 = self._run_recovery_pass_qualitative(comp_names, cases, "Is this a valid missed link?")

        result = current_links.copy()

        for i, (snum, txt, cname) in enumerate(potential_fns[:12]):
            # Require BOTH passes to approve (unanimous = high confidence)
            approved_1 = results_1.get(i, False)
            approved_2 = results_2.get(i, False)

            if approved_1 and approved_2 and cname in name_to_id:
                key = (snum, name_to_id[cname])
                if not any((l.sentence_number, l.component_id) == key for l in result):
                    result.append(SadSamLink(snum, name_to_id[cname], cname, 1.0, "recovered"))
                    print(f"    Recovered: S{snum} -> {cname}")

        return result

    # =========================================================================
    # Phase 9: Judge reviews ambiguous TransArc links too
    # =========================================================================

    def _agent_judge_review(self, links, sentences, components, sent_map, transarc_set) -> list[SadSamLink]:
        """Override: also review TransArc links for ambiguous component names."""
        if len(links) < 5:
            return links

        comp_names = self._get_comp_names(components)
        ambiguous = self.model_knowledge.ambiguous_names if self.model_knowledge else set()

        # Split TransArc: safe (non-ambiguous) auto-approve, ambiguous get reviewed
        safe_transarc = []
        review_links = []
        for l in links:
            is_transarc = (l.sentence_number, l.component_id) in transarc_set
            if is_transarc and l.component_name not in ambiguous:
                safe_transarc.append(l)
            else:
                review_links.append(l)

        if not review_links:
            return safe_transarc

        cases = []
        for i, l in enumerate(review_links[:30]):
            sent = sent_map.get(l.sentence_number)
            context = []
            if l.source in ("implicit", "coreference"):
                prev2 = sent_map.get(l.sentence_number - 2)
                if prev2:
                    context.append(f"    PREV2: {prev2.text[:45]}...")
            prev = sent_map.get(l.sentence_number - 1)
            if prev:
                context.append(f"    PREV: {prev.text[:45]}...")
            context.append(f"    >>> S{l.sentence_number}: {sent.text if sent else '?'}")
            cases.append(f"Case {i+1}: S{l.sentence_number} -> {l.component_name} (src:{l.source})\n{chr(10).join(context)}")

        prompt = f"""JUDGE: Review trace links. Accept clear references, reject ambiguous ones.

COMPONENTS: {', '.join(comp_names)}

VALID IF: Component is the ACTOR/SUBJECT performing the action described
REJECT IF: Effect/result, subprocess, package structure, general concept

SOURCE-SPECIFIC RULES:
- For "transarc" links with ambiguous names: Check if the component name is used as an ARCHITECTURE REFERENCE (the actual component) vs as a GENERIC ENGLISH WORD or concept. "cascade logic" uses "logic" generically; "the Logic component handles..." uses it architecturally.
- For "implicit" links: Apply EXTRA SKEPTICISM. Reject unless the component is UNAMBIGUOUSLY the subject.
- For "coreference" links: Verify the pronoun CLEARLY refers to the claimed component.
- For "entity" and "validated" links: Standard review applies.

LINKS:
{chr(10).join(cases)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=180))
        result = safe_transarc.copy()

        num_judged = min(30, len(review_links))
        if data:
            judged = set()
            for j in data.get("judgments", []):
                idx = j.get("case", 0) - 1
                if 0 <= idx < num_judged:
                    judged.add(idx)
                    if j.get("approve", False):
                        result.append(review_links[idx])
            for i in range(num_judged):
                if i not in judged:
                    result.append(review_links[i])
            for i in range(num_judged, len(review_links)):
                result.append(review_links[i])
        else:
            result.extend(review_links)

        return result

    def _run_recovery_pass_qualitative(self, comp_names, cases, question):
        """Recovery pass returning approve/reject only."""
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
