"""AgentLinker: Conservative fork of V45 with restricted implicit references.

Based on V45 error analysis showing Phase 8 (implicit references) generating
massive FPs (83/105 FPs on Sonnet, 26/43 on Codex). LLM models have become
more aggressive at implicit link generation.

Key changes from V45:
1. Phase 8: Five stacked restrictions on implicit references
   - Require BOTH active_entity AND paragraph_topic to agree
   - Proximity limit: max 2 sentences from last explicit mention
   - Higher threshold: 0.85 minimum (dedicated implicit_threshold)
   - Restrictive prompt with expanded NOT-implicit list
   - Two-pass validation of implicit links
2. Phase 9: Source-aware strictness in Agent-as-Judge
   - Extra skepticism for implicit and coreference links
   - More context (2 previous sentences) for implicit/coref links
"""

import re
import csv
import os
from dataclasses import dataclass, field
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


class AgentLinker:
    """AgentLinker: Conservative implicit refs, source-aware judge."""

    CONTEXT_WINDOW = 3
    PRONOUN_PATTERN = re.compile(
        r'\b(it|they|this|these|that|those|its|their|the component|the service)\b',
        re.IGNORECASE
    )

    def __init__(self, backend: Optional[LLMBackend] = None):
        self.llm = LLMClient(backend=backend)
        self.model_knowledge: Optional[ModelKnowledge] = None
        self.doc_knowledge: Optional[DocumentKnowledge] = None
        self.learned_patterns: Optional[LearnedPatterns] = None
        self.thresholds: Optional[LearnedThresholds] = None
        self.doc_profile: Optional[DocumentProfile] = None

        print(f"AgentLinker using LLM backend: {self.llm.backend.value}")
        print("AgentLinker: Conservative implicit refs, source-aware judge")

    def link(self, text_path: str, model_path: str,
             transarc_csv: str = None) -> list[SadSamLink]:

        # Load data using core utilities
        components = parse_pcm_repository(model_path)
        sentences = DocumentLoader.load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        id_to_name = {c.id: c.name for c in components}
        sent_map = DocumentLoader.build_sent_map(sentences)

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        # Phase 0: Document Analysis & Threshold Learning
        print("\n[Phase 0] Document Analysis & Threshold Learning")
        self.doc_profile = self._learn_document_profile(sentences, components)
        self.thresholds = self._learn_thresholds(sentences, components, self.doc_profile)
        print(f"  Complexity: {self.doc_profile.complexity_score:.2f} ({self.doc_profile.recommended_strictness})")
        print(f"  Coref: {self.thresholds.coref_threshold:.2f}, Valid: {self.thresholds.validation_threshold:.2f}")
        print(f"  Implicit: {self.thresholds.implicit_threshold:.2f}")

        # Phase 1: Model Structure Analysis
        print("\n[Phase 1] Model Structure Analysis")
        self.model_knowledge = self._analyze_model(components)
        print(f"  Architectural: {len(self.model_knowledge.architectural_names)}")
        print(f"  Ambiguous: {self.model_knowledge.ambiguous_names}")

        # Phase 2: Pattern Learning (with Debate)
        print("\n[Phase 2] Pattern Learning (with Debate)")
        self.learned_patterns = self._learn_patterns_with_debate(sentences, components)
        print(f"  Subprocess terms: {len(self.learned_patterns.subprocess_terms)}")

        # Phase 3: Document Knowledge (Judge-Validated)
        print("\n[Phase 3] Document Knowledge (Judge-Validated)")
        self.doc_knowledge = self._learn_document_knowledge_with_judge(sentences, components)
        print(f"  Abbreviations: {len(self.doc_knowledge.abbreviations)}")
        print(f"  Synonyms: {len(self.doc_knowledge.synonyms)}")
        print(f"  Generic terms: {len(self.doc_knowledge.generic_terms)}")

        # Phase 4: TransArc Processing
        print("\n[Phase 4] TransArc Processing")
        transarc_links = self._process_transarc(transarc_csv, id_to_name, sent_map, name_to_id)
        transarc_set = {(l.sentence_number, l.component_id) for l in transarc_links}
        print(f"  TransArc links: {len(transarc_links)}")

        # Phase 5: Entity Extraction
        print("\n[Phase 5] Entity Extraction")
        entity_candidates = self._extract_entities(sentences, components, name_to_id, sent_map)
        print(f"  Candidates: {len(entity_candidates)}")

        # Phase 6: Self-Consistency Validation
        print("\n[Phase 6] Self-Consistency Validation")
        validated = self._validate_with_self_consistency(entity_candidates, components, sent_map)
        print(f"  Validated: {len(validated)}")

        # Phase 7: Discourse-Aware Coreference
        print("\n[Phase 7] Discourse-Aware Coreference")
        discourse_model = self._build_discourse_model(sentences, components, name_to_id)
        coref_links = self._resolve_coreferences_with_discourse(
            sentences, components, name_to_id, sent_map, discourse_model
        )
        print(f"  Coreference: {len(coref_links)}")

        # Phase 8: Conservative Implicit Reference Detection
        print("\n[Phase 8] Conservative Implicit Reference Detection")
        implicit_links = self._detect_implicit_references(
            sentences, components, name_to_id, sent_map, discourse_model, transarc_set
        )
        print(f"  Implicit (before validation): {len(implicit_links)}")

        # Two-pass validation of implicit links
        if implicit_links:
            implicit_links = self._validate_implicit_links(
                implicit_links, sentences, components, sent_map
            )
            print(f"  Implicit (after validation): {len(implicit_links)}")

        # Convert candidates to links
        entity_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name,
                      min(1.0, c.confidence + (0.05 if c.component_name in self.model_knowledge.architectural_names else 0)),
                      c.source)
            for c in validated
        ]

        # Combine all links
        all_links = transarc_links + entity_links + coref_links + implicit_links
        link_map = {}
        for link in all_links:
            key = (link.sentence_number, link.component_id)
            if key not in link_map or link.source == "transarc" or link.confidence > link_map[key].confidence:
                link_map[key] = link

        preliminary = list(link_map.values())

        # Phase 9: Source-Aware Agent-as-Judge Review
        print("\n[Phase 9] Source-Aware Agent-as-Judge Review")
        reviewed = self._agent_judge_review(preliminary, sentences, components, sent_map, transarc_set)
        print(f"  After review: {len(reviewed)} (was {len(preliminary)})")

        # Phase 10: Adaptive FN Recovery (only if relaxed)
        if self.doc_profile.recommended_strictness == "relaxed":
            print("\n[Phase 10] Adaptive FN Recovery")
            final = self._adaptive_fn_recovery(reviewed, sentences, components, name_to_id, sent_map)
            print(f"  After recovery: {len(final)} (was {len(reviewed)})")
        else:
            print(f"\n[Phase 10] FN Recovery SKIPPED (strictness={self.doc_profile.recommended_strictness})")
            final = reviewed

        print(f"\nFinal: {len(final)} links")
        return final

    # =========================================================================
    # Discourse Model Building
    # =========================================================================

    def _build_discourse_model(
        self, sentences: list[Sentence], components: list, name_to_id: dict
    ) -> dict[int, DiscourseContext]:
        """Build discourse context for each sentence."""
        comp_names = self._get_comp_names(components)
        discourse_map = {}
        context = DiscourseContext()
        para_mentions: dict[str, int] = {}

        for sent in sentences:
            # Detect paragraph boundary
            if self._is_paragraph_boundary(sentences, sent.number):
                context.start_new_paragraph(sent.number)
                para_mentions = {}

            # Find explicit mentions in this sentence
            text_lower = sent.text.lower()
            for comp in comp_names:
                if comp.lower() in text_lower:
                    if self._in_dotted_path(sent.text, comp):
                        continue

                    is_subject = self._is_subject(sent.text, comp)
                    mention = EntityMention(
                        sentence_number=sent.number,
                        component_name=comp,
                        component_id=name_to_id.get(comp, ''),
                        mention_text=comp,
                        is_subject=is_subject
                    )
                    context.add_mention(mention)
                    para_mentions[comp] = para_mentions.get(comp, 0) + 1

            # Update paragraph topic
            if para_mentions:
                context.paragraph_topic = max(para_mentions.keys(), key=lambda k: para_mentions[k])

            # Store snapshot for this sentence
            discourse_map[sent.number] = DiscourseContext(
                recent_mentions=list(context.recent_mentions),
                paragraph_topic=context.paragraph_topic,
                paragraph_start=context.paragraph_start,
                active_entity=context.active_entity
            )

        return discourse_map

    def _is_paragraph_boundary(self, sentences: list[Sentence], sent_num: int) -> bool:
        """Detect paragraph boundaries."""
        if sent_num <= 1:
            return True

        sent_map = {s.number: s for s in sentences}
        curr = sent_map.get(sent_num)
        if not curr:
            return False

        transitions = ['however', 'furthermore', 'additionally', 'in addition',
                      'moreover', 'on the other hand', 'the following']
        curr_lower = curr.text.lower()
        if any(curr_lower.startswith(t) for t in transitions):
            return True

        return False

    def _is_subject(self, sentence: str, component: str) -> bool:
        """Check if component is the grammatical subject."""
        sent_lower = sentence.lower()
        comp_lower = component.lower()
        comp_pos = sent_lower.find(comp_lower)

        if comp_pos == -1:
            return False

        if comp_pos < 60:
            verbs = ['is', 'are', 'does', 'do', 'has', 'have', 'provides', 'handles',
                    'manages', 'stores', 'sends', 'receives', 'creates', 'processes']
            for verb in verbs:
                verb_pos = sent_lower.find(f' {verb} ')
                if verb_pos > comp_pos:
                    return True
        return False

    # =========================================================================
    # Discourse-Aware Coreference Resolution
    # =========================================================================

    def _resolve_coreferences_with_discourse(
        self, sentences: list[Sentence], components: list, name_to_id: dict,
        sent_map: dict, discourse_model: dict
    ) -> list[SadSamLink]:
        """Resolve coreferences using discourse context."""
        comp_names = self._get_comp_names(components)
        threshold = self.thresholds.coref_threshold
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
{"resolutions": [{"case": 1, "sentence": N, "pronoun": "it", "component": "Name", "confidence": 0.0-1.0, "reasoning": "why"}]}

Guidelines:
- Use discourse context (recent mentions, paragraph topic) to resolve ambiguity
- Higher confidence if component was explicit subject in previous sentence
- Lower confidence if pronoun could refer to multiple things
- Skip if uncertain

JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=150))
            if not data:
                continue

            for res in data.get("resolutions", []):
                comp = res.get("component")
                snum = res.get("sentence")
                conf = res.get("confidence", 0.8)

                if not (comp and snum and comp in name_to_id):
                    continue

                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue

                if conf < threshold:
                    continue

                sent = sent_map.get(snum)
                if sent and self.learned_patterns and self.learned_patterns.is_subprocess(sent.text):
                    conf -= 0.05
                    if conf < threshold:
                        continue

                if comp in self.model_knowledge.architectural_names:
                    conf = min(1.0, conf + 0.02)

                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, conf, "coreference"))

        return all_coref

    # =========================================================================
    # CONSERVATIVE Implicit Reference Detection (Phase 8 overhaul)
    # =========================================================================

    def _detect_implicit_references(
        self, sentences: list[Sentence], components: list, name_to_id: dict,
        sent_map: dict, discourse_model: dict, existing_links: set
    ) -> list[SadSamLink]:
        """Detect implicit references with conservative restrictions.

        Five stacked restrictions vs V45:
        a) Require BOTH active_entity AND paragraph_topic to agree
        b) Proximity limit: max 2 sentences from last explicit mention
        c) Higher threshold: 0.85 minimum (implicit_threshold)
        d) Restrictive prompt
        e) Two-pass validation (called after this method)
        """
        comp_names = self._get_comp_names(components)
        implicit_links = []
        threshold = self.thresholds.implicit_threshold  # (c) 0.85 instead of validation_threshold

        # Find sentences without explicit mention but with likely implicit ref
        candidates = []
        for sent in sentences:
            # Skip if has explicit mention
            text_lower = sent.text.lower()
            has_explicit = any(c.lower() in text_lower for c in comp_names)
            if has_explicit:
                continue

            ctx = discourse_model.get(sent.number, DiscourseContext())

            # (a) Require BOTH active_entity AND paragraph_topic, and they must agree
            if not ctx.active_entity or not ctx.paragraph_topic:
                continue
            if ctx.active_entity != ctx.paragraph_topic:
                continue

            likely_comp = ctx.active_entity

            # Skip if already linked
            if (sent.number, name_to_id.get(likely_comp, '')) in existing_links:
                continue

            # (b) Proximity limit: max 2 sentences from last explicit mention
            last_mention_dist = self._distance_to_last_mention(
                ctx.recent_mentions, likely_comp, sent.number
            )
            if last_mention_dist is None or last_mention_dist > 2:
                continue

            candidates.append((sent, ctx, likely_comp))

        if not candidates:
            return []

        print(f"    Candidates after filtering: {len(candidates)}")

        # Process in smaller batches (reduced from 10 to 5)
        batch_size = 5
        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start:batch_start + batch_size]

            # (d) Restrictive prompt
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
{"detections": [{"case": 1, "sentence": N, "component": "Name", "is_implicit": true/false, "confidence": 0.0-1.0, "reasoning": "why"}]}

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

Precision is more important than recall. When in doubt, set is_implicit to false.

JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
            if not data:
                continue

            for det in data.get("detections", []):
                if not det.get("is_implicit"):
                    continue

                comp = det.get("component")
                snum = det.get("sentence")
                conf = det.get("confidence", 0.75)

                if not (comp and snum and comp in name_to_id):
                    continue

                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue

                # (c) Higher threshold
                if conf < threshold:
                    continue

                # Lower confidence cap (0.85 instead of 0.88)
                conf = min(conf, 0.85)

                implicit_links.append(SadSamLink(snum, name_to_id[comp], comp, conf, "implicit"))

        return implicit_links

    def _distance_to_last_mention(
        self, recent_mentions: list[EntityMention], component: str, current_sent: int
    ) -> Optional[int]:
        """Find distance (in sentences) to last explicit mention of component."""
        for mention in reversed(recent_mentions):
            if mention.component_name == component:
                return current_sent - mention.sentence_number
        return None

    # =========================================================================
    # (e) Two-pass validation of implicit links
    # =========================================================================

    def _validate_implicit_links(
        self, implicit_links: list[SadSamLink], sentences: list[Sentence],
        components: list, sent_map: dict
    ) -> list[SadSamLink]:
        """Independent second-pass validation of implicit links.

        If second pass fails entirely, reject ALL implicit links (conservative default).
        Each link must reach 0.85 confidence in second pass to survive.
        """
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
{{"validations": [{{"case": 1, "valid": true/false, "confidence": 0.0-1.0, "reasoning": "brief"}}]}}

Reject if:
- The action could reasonably be performed by any component
- The sentence is about system behavior in general
- The connection to the component is tenuous
- You are not highly confident

JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=150))
        if not data:
            # Conservative: if validation fails entirely, reject all
            print("    Implicit validation failed — rejecting all implicit links")
            return []

        threshold = self.thresholds.implicit_threshold
        validated = []
        for v in data.get("validations", []):
            idx = v.get("case", 0) - 1
            if 0 <= idx < len(implicit_links):
                if v.get("valid", False) and v.get("confidence", 0) >= threshold:
                    validated.append(implicit_links[idx])

        return validated

    # =========================================================================
    # Source-Aware Agent-as-Judge (Phase 9 overhaul)
    # =========================================================================

    def _agent_judge_review(self, links, sentences, components, sent_map, transarc_set) -> list[SadSamLink]:
        if len(links) < 5:
            return links

        comp_names = self._get_comp_names(components)
        transarc = [l for l in links if (l.sentence_number, l.component_id) in transarc_set]
        non_transarc = [l for l in links if (l.sentence_number, l.component_id) not in transarc_set]

        if not non_transarc:
            return transarc

        cases = []
        for i, l in enumerate(non_transarc[:30]):
            sent = sent_map.get(l.sentence_number)
            # (b) More context for implicit/coref: show 2 previous sentences
            context = []
            if l.source in ("implicit", "coreference"):
                prev2 = sent_map.get(l.sentence_number - 2)
                if prev2:
                    context.append(f"    PREV2: {prev2.text[:45]}...")
            prev = sent_map.get(l.sentence_number - 1)
            if prev:
                context.append(f"    PREV: {prev.text[:45]}...")
            context.append(f"    >>> S{l.sentence_number}: {sent.text if sent else '?'}")
            cases.append(f"Case {i+1}: S{l.sentence_number} -> {l.component_name} (src:{l.source}, conf:{l.confidence:.2f})\n{chr(10).join(context)}")

        strictness_instruction = {
            "relaxed": "Be INCLUSIVE - accept reasonable links.",
            "balanced": "Be BALANCED - accept clear references.",
            "strict": "Be STRICT - only approve confident links."
        }.get(self.doc_profile.recommended_strictness, "Be balanced.")

        # (a) Source-aware prompt with extra rules for implicit/coref
        prompt = f"""JUDGE: Review trace links. {strictness_instruction}

COMPONENTS: {', '.join(comp_names)}

VALID IF: Component is ACTOR/SUBJECT
REJECT IF: Effect/result, subprocess, package structure

SOURCE-SPECIFIC RULES:
- For "implicit" links: Apply EXTRA SKEPTICISM. Reject unless the component is UNAMBIGUOUSLY the subject of the sentence. If the sentence could describe general system behavior, reject it.
- For "coreference" links: Verify the pronoun CLEARLY refers to the claimed component. If the pronoun could refer to something else, reject.
- For "entity" and "validated" links: Standard review applies.

LINKS:
{chr(10).join(cases)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=180))
        result = transarc.copy()

        num_judged = min(30, len(non_transarc))
        if data:
            judged = set()
            for j in data.get("judgments", []):
                idx = j.get("case", 0) - 1
                if 0 <= idx < num_judged:
                    judged.add(idx)
                    if j.get("approve", False):
                        result.append(non_transarc[idx])
            # Keep unjudged links (judge didn't return a verdict — don't silently drop)
            for i in range(num_judged):
                if i not in judged:
                    result.append(non_transarc[i])
            # Keep links beyond the judge's scope
            for i in range(num_judged, len(non_transarc)):
                result.append(non_transarc[i])
        else:
            result.extend(non_transarc)

        return result

    # =========================================================================
    # Existing methods (unchanged from V45)
    # =========================================================================

    def _get_comp_names(self, components) -> list[str]:
        """Get non-implementation component names."""
        return [c.name for c in components
                if not (self.model_knowledge and self.model_knowledge.is_implementation(c.name))]

    def _in_dotted_path(self, text: str, comp_name: str) -> bool:
        """Check if component name appears ONLY in dotted paths (not standalone with original case)."""
        cn = comp_name.lower()
        in_dotted = False
        for path in re.findall(r'\b\w+(?:\.\w+)+\b', text.lower()):
            if cn in path.split('.'):
                in_dotted = True
        if not in_dotted:
            return False
        # Check if the name also appears standalone WITH ORIGINAL CASE
        # (case-sensitive match indicates architectural reference, not generic word)
        clean = re.sub(r'\b\w+(?:\.\w+)+\b', '', text)
        if re.search(rf'\b{re.escape(comp_name)}\b', clean):
            return False  # Name appears standalone with original case — don't filter
        return True

    def _learn_document_profile(self, sentences, components) -> DocumentProfile:
        """Learn document characteristics via LLM analysis."""
        texts = [s.text for s in sentences]
        comp_names = [c.name for c in components]

        pronoun_pattern = r'\b(it|they|this|these|that|those|its|their)\b'
        pronoun_sentences = sum(1 for t in texts if re.search(pronoun_pattern, t.lower()))
        pronoun_ratio = pronoun_sentences / len(sentences) if sentences else 0

        comp_mentions = sum(1 for t in texts for c in comp_names if c.lower() in t.lower())
        mention_density = comp_mentions / len(sentences) if sentences else 0

        sample = [f"S{s.number}: {s.text}" for s in sentences[:50]]

        prompt = f"""Analyze this software architecture document.

STATISTICS:
- {len(sentences)} sentences, {len(components)} components
- {pronoun_ratio:.0%} sentences contain pronouns
- ~{mention_density:.1f} component mentions per sentence

COMPONENTS: {', '.join(comp_names)}

SAMPLE:
{chr(10).join(sample)}

Return JSON:
{{
  "technical_density": 0.0-1.0,
  "complexity_score": 0.0-1.0,
  "recommended_strictness": "relaxed|balanced|strict",
  "reasoning": "why"
}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=120))

        if data:
            tech_density = min(1.0, max(0.0, data.get("technical_density", 0.3)))
            complexity = min(1.0, max(0.0, data.get("complexity_score", 0.5)))
            strictness = data.get("recommended_strictness", "balanced")
            if strictness not in ["relaxed", "balanced", "strict"]:
                strictness = "balanced"
        else:
            tech_density = 0.3 if len(sentences) < 50 else 0.6
            complexity = min(1.0, len(sentences) / 200)
            strictness = "relaxed" if len(sentences) < 50 else ("strict" if len(sentences) > 150 else "balanced")

        return DocumentProfile(
            sentence_count=len(sentences),
            component_count=len(components),
            pronoun_ratio=pronoun_ratio,
            technical_density=tech_density,
            component_mention_density=mention_density,
            complexity_score=complexity,
            recommended_strictness=strictness
        )

    def _learn_thresholds(self, sentences, components, profile: DocumentProfile) -> LearnedThresholds:
        """Learn thresholds from document."""
        comp_names = [c.name for c in components]
        sample = [f"S{s.number}: {s.text}" for s in sentences[:40]]

        prompt = f"""Recommend confidence thresholds for trace link recovery.

DOCUMENT PROFILE:
- {profile.sentence_count} sentences, {profile.component_count} components
- Pronoun ratio: {profile.pronoun_ratio:.0%}
- Complexity: {profile.complexity_score:.2f}
- Strictness: {profile.recommended_strictness}

COMPONENTS: {', '.join(comp_names)}

SAMPLE:
{chr(10).join(sample)}

Return JSON:
{{
  "coref_threshold": 0.XX,
  "validation_threshold": 0.XX,
  "fn_recovery_threshold": 0.XX,
  "disambiguation_threshold": 0.XX,
  "reasoning": "why"
}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=120))

        if data:
            return LearnedThresholds(
                coref_threshold=min(0.95, max(0.75, data.get("coref_threshold", 0.85))),
                validation_threshold=min(0.92, max(0.72, data.get("validation_threshold", 0.80))),
                fn_recovery_threshold=min(0.95, max(0.78, data.get("fn_recovery_threshold", 0.85))),
                disambiguation_threshold=min(0.92, max(0.70, data.get("disambiguation_threshold", 0.80))),
                reasoning=data.get("reasoning", "learned"),
                implicit_threshold=0.85,  # Fixed, not LLM-learned
            )

        base = 0.80 if profile.recommended_strictness == "relaxed" else (0.88 if profile.recommended_strictness == "strict" else 0.84)
        return LearnedThresholds(base + 0.02, base, base + 0.03, base - 0.02, "derived", 0.85)

    def _analyze_model(self, components) -> ModelKnowledge:
        """Analyze model structure."""
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

        prompt = f"""Classify these component names.

NAMES: {', '.join(names)}

Return JSON:
{{
  "architectural": ["names clearly representing architecture components"],
  "ambiguous": ["names that could be common English words"]
}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=100))
        if data:
            knowledge.architectural_names = set(data.get("architectural", [])) & set(names)
            knowledge.ambiguous_names = set(data.get("ambiguous", [])) & set(names)

        return knowledge

    def _learn_patterns_with_debate(self, sentences, components) -> LearnedPatterns:
        """Learn patterns using agent debate."""
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

    def _learn_document_knowledge_with_judge(self, sentences, components) -> DocumentKnowledge:
        """Learn abbreviations/synonyms with judge validation."""
        comp_names = self._get_comp_names(components)
        doc_lines = [f"S{s.number}: {s.text}" for s in sentences[:100]]

        prompt1 = f"""Find SPECIFIC alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

DOCUMENT:
{chr(10).join(doc_lines)}

Rules:
- ONLY include terms that are SPECIFIC NAMES for a component, not generic descriptions
- Abbreviations: shortened forms explicitly defined in the text (e.g. "KMS" for "Kurento Media Server")
- Synonyms: proper names or specific technical terms used as interchangeable names (e.g. "ImageProvider service" for "ImageProvider")
- DO NOT include generic English phrases like "business logic", "utility code", "front-end", "helper classes", "storage layer", "web pages" — these describe concepts, not specific components
- Partial references: shortened but distinctive references (e.g. "BigBlueButton client" for "HTML5 Client")

Return JSON:
{{
  "abbreviations": {{"short": "FullComponent"}},
  "synonyms": {{"alternative": "FullComponent"}},
  "partial_references": {{"partial": "FullComponent"}}
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

            prompt2 = f"""JUDGE: Review these proposed component name mappings. Be STRICT.

COMPONENTS: {', '.join(comp_names)}

PROPOSED:
{chr(10).join(mapping_list)}

REJECT any mapping where the term is:
- A generic English phrase describing a concept rather than naming the specific component (e.g. "business logic", "utility code", "helper classes", "storage layer", "front-end", "web pages", "data transfer objects")
- A phrase that could apply to many systems, not specifically to this component
- A common programming concept rather than a proper name

APPROVE only mappings where the term is a SPECIFIC NAME used in this document to refer to exactly that component.

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

        return knowledge

    def _process_transarc(self, transarc_csv, id_to_name, sent_map, name_to_id):
        links = []
        if not transarc_csv or not os.path.exists(transarc_csv):
            return links

        with open(transarc_csv) as f:
            for row in csv.DictReader(f):
                cid = row.get('modelElementID', '')
                try:
                    snum = int(row.get('sentence', 0))
                except (ValueError, TypeError):
                    continue
                cname = id_to_name.get(cid, "")
                sent = sent_map.get(snum)
                if not sent or self._in_dotted_path(sent.text, cname):
                    continue
                if self.model_knowledge and cname in self.model_knowledge.impl_to_abstract:
                    cname = self.model_knowledge.impl_to_abstract[cname]
                    cid = name_to_id.get(cname, cid)
                conf = 0.92 if self.model_knowledge and cname in self.model_knowledge.architectural_names else 0.90
                links.append(SadSamLink(snum, cid, cname, conf, "transarc"))
        return links

    def _extract_entities(self, sentences, components, name_to_id, sent_map) -> list[CandidateLink]:
        comp_names = self._get_comp_names(components)
        comp_lower = {n.lower() for n in comp_names}

        mappings = []
        if self.doc_knowledge:
            mappings.extend([f"{a}={c}" for a, c in self.doc_knowledge.abbreviations.items()])
            mappings.extend([f"{s}={c}" for s, c in self.doc_knowledge.synonyms.items()])
            mappings.extend([f"{p}={c}" for p, c in self.doc_knowledge.partial_references.items()])

        batch_size = 100
        all_candidates = {}  # (snum, cid) -> CandidateLink for dedup

        for batch_start in range(0, len(sentences), batch_size):
            batch = sentences[batch_start:batch_start + batch_size]

            if len(sentences) > batch_size:
                print(f"    Entity batch {batch_start//batch_size + 1}: "
                      f"S{batch[0].number}-S{batch[-1].number} ({len(batch)} sents)")

            prompt = f"""Extract component references.

COMPONENTS: {', '.join(comp_names)}
{f'ALIASES: {", ".join(mappings[:20])}' if mappings else ''}

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N, "component": "Name", "matched_text": "text", "match_type": "exact|synonym|partial"}}]}}
JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=150))
            if not data:
                continue

            for ref in data.get("references", []):
                snum, cname = ref.get("sentence"), ref.get("component")
                if not (snum and cname and cname in name_to_id):
                    continue
                sent = sent_map.get(snum)
                if not sent or self._in_dotted_path(sent.text, cname):
                    continue

                matched = ref.get("matched_text", "").lower()
                is_exact = matched in comp_lower or cname.lower() in matched
                needs_val = not is_exact or ref.get("match_type") != "exact" or \
                           (self.model_knowledge and cname in self.model_knowledge.ambiguous_names)

                key = (snum, name_to_id[cname])
                if key not in all_candidates:
                    all_candidates[key] = CandidateLink(snum, sent.text, cname, name_to_id[cname],
                                               ref.get("matched_text", ""), 0.85, "entity",
                                               ref.get("match_type", "exact"), needs_val)

        return list(all_candidates.values())

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

        results_1 = self._run_validation_pass(comp_names, ctx, cases, "Focus on ACTOR")
        results_2 = self._run_validation_pass(comp_names, ctx, cases, "Focus on DIRECT reference")

        validated = []
        threshold = self.thresholds.validation_threshold

        for i, c in enumerate(needs_validation[:25]):
            votes = []
            if i in results_1 and results_1[i][0]:
                votes.append(results_1[i][1])
            if i in results_2 and results_2[i][0]:
                votes.append(results_2[i][1])

            if len(votes) >= 1:
                avg_conf = sum(votes) / len(votes)
                if avg_conf >= threshold:
                    c.confidence = avg_conf
                    c.source = "validated"
                    validated.append(c)

        return direct + validated

    def _run_validation_pass(self, comp_names, ctx, cases, focus):
        prompt = f"""Validate component references. {focus}.

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "valid": true/false, "confidence": 0.0-1.0}}]}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
        results = {}
        if data:
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    results[idx] = (v.get("valid", False), v.get("confidence", 0.8))
        return results

    def _adaptive_fn_recovery(self, current_links, sentences, components, name_to_id, sent_map) -> list[SadSamLink]:
        comp_names = self._get_comp_names(components)
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

            for cname in comp_names:
                if cname.lower() in sent_lower:
                    pattern = rf'\b{re.escape(cname)}\b'
                    if re.search(pattern, sent.text, re.IGNORECASE):
                        potential_fns.append((sent.number, sent.text, cname))
                        break
            else:
                for syn, comp in synonyms_to_comp.items():
                    if syn in sent_lower and comp in comp_names:
                        potential_fns.append((sent.number, sent.text, comp))
                        break

        if not potential_fns:
            print("    No potential FNs found")
            return current_links

        print(f"    Checking {len(potential_fns)} potential FNs")

        cases = [f"Case {i+1}: S{sn}: \"{txt[:70]}...\" -> {cn}?" for i, (sn, txt, cn) in enumerate(potential_fns[:12])]

        results_1 = self._run_recovery_pass(comp_names, cases, "Is component the ACTOR?")
        results_2 = self._run_recovery_pass(comp_names, cases, "Is this a valid missed link?")

        result = current_links.copy()
        threshold = self.thresholds.fn_recovery_threshold

        for i, (snum, txt, cname) in enumerate(potential_fns[:12]):
            votes = []
            if i in results_1 and results_1[i][0]:
                votes.append(results_1[i][1])
            if i in results_2 and results_2[i][0]:
                votes.append(results_2[i][1])

            if len(votes) >= 2:
                avg_conf = sum(votes) / len(votes)
                if avg_conf >= threshold and cname in name_to_id:
                    key = (snum, name_to_id[cname])
                    if not any((l.sentence_number, l.component_id) == key for l in result):
                        result.append(SadSamLink(snum, name_to_id[cname], cname, avg_conf, "recovered"))
                        print(f"    Recovered: S{snum} -> {cname}")

        return result

    def _run_recovery_pass(self, comp_names, cases, question):
        prompt = f"""Check potential missed links. {question}

COMPONENTS: {', '.join(comp_names)}

POTENTIAL:
{chr(10).join(cases)}

Return JSON:
{{"recoveries": [{{"case": 1, "valid": true/false, "confidence": 0.0-1.0}}]}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=100))
        results = {}
        if data:
            for r in data.get("recoveries", []):
                idx = r.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    results[idx] = (r.get("valid", False), r.get("confidence", 0.8))
        return results


def export_links_csv(links, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "component_id", "component_name", "confidence", "source"])
        for link in sorted(links, key=lambda x: x.sentence_number):
            writer.writerow([link.sentence_number, link.component_id,
                           link.component_name, f"{link.confidence:.2f}", link.source])
