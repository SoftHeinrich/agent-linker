"""TransArc + Meta-Learning V44.

V44: Best of V41 + Multi-Component Extraction

Key insight from V42/V43: The Component Wiki approach added complexity but
didn't improve over V41's simpler document knowledge learning with judge
validation. Multi-component extraction is valuable but needs strict validation.

V44 combines:
1. V41's document knowledge learning with judge (accurate synonyms/abbreviations)
2. Multi-component extraction (addresses 38% of FNs in error analysis)
3. Stricter word-as-concept filtering for ambiguous component names
4. V41's coreference with debate (more accurate than V43's "strict" version)

NO component wiki - it added complexity without clear benefit.

Adapted to use v45 core module (shared data types and document loader).
"""

import re
import csv
import os
from typing import Optional

from ...core import (
    SadSamLink,
    CandidateLink,
    DocumentProfile,
    LearnedThresholds,
    ModelKnowledge,
    DocumentKnowledge,
    LearnedPatterns,
    DocumentLoader,
    Sentence,
)
from ...pcm_parser import parse_pcm_repository
from ...llm_client import LLMClient, LLMBackend


class TransArcRefinedLinkerV44:
    """V44: V41 base + multi-component extraction. NO wiki overhead."""

    CONTEXT_WINDOW = 3

    def __init__(self, backend: Optional[LLMBackend] = None):
        self.llm_client = LLMClient(backend=backend)
        self.model_knowledge = None
        self.doc_knowledge = None
        self.learned_patterns = None
        self.thresholds = None
        self.doc_profile = None

        print(f"V44 using LLM backend: {self.llm_client.backend.value}")
        print("V44: V41 base + multi-component extraction")

    def link(self, text_path: str, model_path: str,
             transarc_csv: str = None) -> list[SadSamLink]:

        components = parse_pcm_repository(model_path)
        sentences = DocumentLoader.load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        id_to_name = {c.id: c.name for c in components}
        sent_map = {s.number: s for s in sentences}

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        # Phase 0: Document Analysis & Threshold Learning
        print("\n[Phase 0] Document Analysis & Threshold Learning")
        self.doc_profile = self._learn_document_profile(sentences, components)
        self.thresholds = self._learn_thresholds(sentences, components, self.doc_profile)
        print(f"  Complexity: {self.doc_profile.complexity_score:.2f} ({self.doc_profile.recommended_strictness})")
        print(f"  Coref: {self.thresholds.coref_threshold:.2f}, Valid: {self.thresholds.validation_threshold:.2f}")

        # Phase 1: Model Structure Analysis
        print("\n[Phase 1] Model Structure Analysis")
        self.model_knowledge = self._analyze_model(components)
        print(f"  Architectural: {len(self.model_knowledge.architectural_names)}")
        print(f"  Ambiguous: {self.model_knowledge.ambiguous_names}")

        # Phase 2: Pattern Learning (with Debate)
        print("\n[Phase 2] Pattern Learning (with Debate)")
        self.learned_patterns = self._learn_patterns_with_debate(sentences, components)
        print(f"  Subprocess terms: {len(self.learned_patterns.subprocess_terms)}")

        # Phase 3: Document Knowledge (Judge-Validated) - from V41
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

        # Phase 5: Multi-Component Entity Extraction (NEW in V44)
        print("\n[Phase 5] Multi-Component Entity Extraction")
        entity_candidates = self._extract_multi_component_entities(sentences, components, name_to_id, sent_map)
        print(f"  Candidates: {len(entity_candidates)}")

        # Phase 6: Word-as-Concept Filtering (for ambiguous names only)
        print("\n[Phase 6] Word-as-Concept Filtering")
        filtered_candidates = self._filter_word_as_concept(entity_candidates, sent_map)
        print(f"  After filtering: {len(filtered_candidates)} (removed {len(entity_candidates) - len(filtered_candidates)})")

        # Phase 7: Self-Consistency Validation
        print("\n[Phase 7] Self-Consistency Validation")
        validated = self._validate_with_self_consistency(filtered_candidates, components, sent_map)
        print(f"  Validated: {len(validated)}")

        # Phase 8: Coreference (Debate-Validated) - V41 approach
        print("\n[Phase 8] Coreference (Debate-Validated)")
        coref_links = self._resolve_coreferences_with_debate(sentences, components, name_to_id, sent_map)
        print(f"  Coreference: {len(coref_links)}")

        # Convert to links
        entity_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name,
                      min(1.0, c.confidence + (0.05 if c.component_name in self.model_knowledge.architectural_names else 0)),
                      c.source)
            for c in validated
        ]

        # Combine
        all_links = transarc_links + entity_links + coref_links
        link_map = {}
        for link in all_links:
            key = (link.sentence_number, link.component_id)
            if key not in link_map or link.source == "transarc" or link.confidence > link_map[key].confidence:
                link_map[key] = link

        preliminary = list(link_map.values())

        # Phase 9: Agent-as-Judge Review
        print("\n[Phase 9] Agent-as-Judge Review")
        reviewed = self._agent_judge_review(preliminary, sentences, components, sent_map, transarc_set)
        print(f"  After review: {len(reviewed)} (was {len(preliminary)})")

        # Phase 10: Adaptive FN Recovery
        if self.doc_profile.recommended_strictness == "relaxed":
            print("\n[Phase 10] Adaptive FN Recovery")
            final = self._adaptive_fn_recovery(reviewed, sentences, components, name_to_id, sent_map)
            print(f"  After recovery: {len(final)} (was {len(reviewed)})")
        else:
            print(f"\n[Phase 10] FN Recovery SKIPPED (strictness={self.doc_profile.recommended_strictness})")
            final = reviewed

        print(f"\nFinal: {len(final)} links")
        return final

    def _extract_multi_component_entities(self, sentences, components, name_to_id, sent_map) -> list[CandidateLink]:
        """Extract ALL component references from sentences - multi-component aware."""
        comp_names = [c.name for c in components if c.name not in self.model_knowledge.impl_to_abstract]
        comp_lower = {n.lower() for n in comp_names}

        # Build alias info from learned knowledge
        mappings = []
        if self.doc_knowledge:
            mappings.extend([f"{a}={c}" for a, c in self.doc_knowledge.abbreviations.items()])
            mappings.extend([f"{s}={c}" for s, c in self.doc_knowledge.synonyms.items()])
            mappings.extend([f"{p}={c}" for p, c in self.doc_knowledge.partial_references.items()])

        prompt = f"""Extract ALL component references from this document.

IMPORTANT: A sentence can reference MULTIPLE components. Extract ALL of them.

COMPONENTS: {', '.join(comp_names)}
{f'ALIASES: {", ".join(mappings[:20])}' if mappings else ''}

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in sentences[:100]])}

For EACH sentence, extract ALL component references (not just the first one).
Look for:
- Direct component name mentions
- Alias/synonym references
- Multiple components in relationship sentences ("A communicates with B")

Return JSON:
{{"references": [
  {{"sentence": N, "component": "Name", "matched_text": "text", "match_type": "exact|synonym|partial"}}
]}}

Example multi-component: "The Dispatcher forwards tasks to the Executor"
-> [{{"sentence": 5, "component": "Dispatcher", ...}}, {{"sentence": 5, "component": "Executor", ...}}]

Exclude dotted paths (package.class.method).
JSON only:"""

        data = self.llm_client.extract_json(self.llm_client.query(prompt, timeout=180))
        if not data:
            return []

        candidates = []
        for ref in data.get("references", []):
            snum, cname = ref.get("sentence"), ref.get("component")
            if not (snum and cname and cname in name_to_id):
                continue
            sent = sent_map.get(snum)
            if not sent or self._component_in_dotted_path(sent.text, cname):
                continue

            matched = ref.get("matched_text", "").lower()
            is_exact = matched in comp_lower or cname.lower() in matched
            needs_val = not is_exact or ref.get("match_type") != "exact" or cname in self.model_knowledge.ambiguous_names

            candidates.append(CandidateLink(snum, sent.text, cname, name_to_id[cname],
                                           ref.get("matched_text", ""), 0.85, "entity",
                                           ref.get("match_type", "exact"), needs_val))

        # Track multi-component sentences
        multi_comp_sents = {}
        for c in candidates:
            multi_comp_sents[c.sentence_number] = multi_comp_sents.get(c.sentence_number, 0) + 1

        multi_count = sum(1 for v in multi_comp_sents.values() if v > 1)
        if multi_count > 0:
            print(f"    Multi-component sentences: {multi_count}")

        return candidates

    def _filter_word_as_concept(self, candidates: list[CandidateLink], sent_map) -> list[CandidateLink]:
        """Filter when ambiguous component names are used as generic concepts."""
        filtered = []
        for c in candidates:
            # Only filter for ambiguous names (common English words)
            if c.component_name not in self.model_knowledge.ambiguous_names:
                filtered.append(c)
                continue

            sent_lower = c.sentence_text.lower()
            name_lower = c.component_name.lower()

            # Check for strong generic usage patterns (common English, not component ref)
            generic_patterns = [
                rf'\bcontains? (some |minimal |the )?{name_lower}\b',  # "contains storage"
                rf'\bminimal {name_lower}\b',  # "minimal overhead"
                rf'\bsome {name_lower}\b',  # "some processing"
                rf'\bhis {name_lower}\b',  # "his config"
                rf'\btheir {name_lower}\b',  # "their config"
                rf'\bbusiness {name_lower}\b',  # "business model"
            ]

            is_generic = any(re.search(p, sent_lower) for p in generic_patterns)

            if is_generic:
                print(f"    Filtered: S{c.sentence_number} '{c.component_name}' (generic concept)")
                continue

            filtered.append(c)

        return filtered

    # ===== V41 methods (unchanged) =====

    def _learn_document_profile(self, sentences, components) -> DocumentProfile:
        texts = [s.text for s in sentences]
        comp_names = [c.name for c in components]

        pronoun_pattern = r'\b(it|they|this|these|that|those|its|their)\b'
        pronoun_sentences = sum(1 for t in texts if re.search(pronoun_pattern, t.lower()))
        pronoun_ratio = pronoun_sentences / len(sentences) if sentences else 0

        comp_mentions = sum(1 for t in texts for c in comp_names if c.lower() in t.lower())
        mention_density = comp_mentions / len(sentences) if sentences else 0

        sample = [f"S{s.number}: {s.text}" for s in sentences[:50]]

        prompt = f"""Analyze this software architecture document's characteristics.

STATISTICS:
- {len(sentences)} sentences, {len(components)} components
- {pronoun_ratio:.0%} sentences contain pronouns
- ~{mention_density:.1f} component mentions per sentence

COMPONENTS: {', '.join(comp_names)}

SAMPLE:
{chr(10).join(sample)}

Analyze and return JSON:
{{
  "technical_density": 0.0-1.0,
  "complexity_score": 0.0-1.0,
  "recommended_strictness": "relaxed|balanced|strict",
  "reasoning": "why this classification"
}}
JSON only:"""

        data = self.llm_client.extract_json(self.llm_client.query(prompt, timeout=120))

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
        comp_names = [c.name for c in components]
        sample = [f"S{s.number}: {s.text}" for s in sentences[:40]]

        prompt = f"""Recommend confidence thresholds for trace link recovery on THIS document.

DOCUMENT PROFILE:
- {profile.sentence_count} sentences, {profile.component_count} components
- Pronoun ratio: {profile.pronoun_ratio:.0%}
- Technical density: {profile.technical_density:.2f}
- Complexity: {profile.complexity_score:.2f}
- Recommended strictness: {profile.recommended_strictness}

COMPONENTS: {', '.join(comp_names)}

SAMPLE:
{chr(10).join(sample)}

Return JSON:
{{
  "coref_threshold": 0.XX,
  "validation_threshold": 0.XX,
  "fn_recovery_threshold": 0.XX,
  "disambiguation_threshold": 0.XX,
  "reasoning": "why these values"
}}
JSON only:"""

        data = self.llm_client.extract_json(self.llm_client.query(prompt, timeout=120))

        if data:
            return LearnedThresholds(
                coref_threshold=min(0.95, max(0.75, data.get("coref_threshold", 0.85))),
                validation_threshold=min(0.92, max(0.72, data.get("validation_threshold", 0.80))),
                fn_recovery_threshold=min(0.95, max(0.78, data.get("fn_recovery_threshold", 0.85))),
                disambiguation_threshold=min(0.92, max(0.70, data.get("disambiguation_threshold", 0.80))),
                reasoning=data.get("reasoning", "learned")
            )

        base = 0.80 if profile.recommended_strictness == "relaxed" else (0.88 if profile.recommended_strictness == "strict" else 0.84)
        return LearnedThresholds(
            coref_threshold=base + 0.02,
            validation_threshold=base,
            fn_recovery_threshold=base + 0.03,
            disambiguation_threshold=base - 0.02,
            reasoning="derived from profile"
        )

    def _analyze_model(self, components) -> ModelKnowledge:
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

        prompt = f"""Classify these software component names.

NAMES: {', '.join(names)}

Return JSON:
{{
  "architectural": ["names clearly representing architecture components"],
  "ambiguous": ["names that could be common English words"]
}}
JSON only:"""

        data = self.llm_client.extract_json(self.llm_client.query(prompt, timeout=100))
        if data:
            knowledge.architectural_names = set(data.get("architectural", [])) & set(names)
            knowledge.ambiguous_names = set(data.get("ambiguous", [])) & set(names)

        return knowledge

    def _learn_patterns_with_debate(self, sentences, components) -> LearnedPatterns:
        comp_names = [c.name for c in components if c.name not in self.model_knowledge.impl_to_abstract]
        sample = [f"S{s.number}: {s.text}" for s in sentences[:70]]

        prompt1 = f"""Find terms in this document that refer to INTERNAL PARTS of components.

COMPONENTS: {', '.join(comp_names)}

DOCUMENT:
{chr(10).join(sample)}

Sub-processes are internal workers, threads, handlers, or instances.

Return JSON:
{{
  "subprocess_terms": ["term1", "term2"],
  "reasoning": {{"term": "why it's a subprocess"}}
}}
JSON only:"""

        data1 = self.llm_client.extract_json(self.llm_client.query(prompt1, timeout=120))

        proposed = data1.get("subprocess_terms", []) if data1 else []
        reasonings = data1.get("reasoning", {}) if data1 else {}

        if proposed:
            prompt2 = f"""DEBATE: Another agent proposed these as subprocess terms. Validate each.

COMPONENTS: {', '.join(comp_names)}

PROPOSED SUBPROCESS TERMS:
{chr(10).join([f"- {t}: {reasonings.get(t, 'no reason given')}" for t in proposed[:15]])}

DOCUMENT SAMPLE:
{chr(10).join(sample[:30])}

Return JSON:
{{
  "validated": ["terms that ARE subprocesses"],
  "rejected": ["terms that might be valid component references"]
}}
JSON only:"""

            data2 = self.llm_client.extract_json(self.llm_client.query(prompt2, timeout=120))
            validated_terms = set(data2.get("validated", [])) if data2 else set(proposed)
        else:
            validated_terms = set()

        prompt3 = f"""Find linguistic patterns in this document.

COMPONENTS: {', '.join(comp_names)}

DOCUMENT:
{chr(10).join(sample[:40])}

Return JSON:
{{
  "action_indicators": ["verbs/phrases when component DOES something"],
  "effect_indicators": ["verbs/phrases for RESULTS (not component action)"]
}}
JSON only:"""

        data3 = self.llm_client.extract_json(self.llm_client.query(prompt3, timeout=100))

        patterns = LearnedPatterns()
        patterns.subprocess_terms = validated_terms
        if data3:
            patterns.action_indicators = data3.get("action_indicators", [])
            patterns.effect_indicators = data3.get("effect_indicators", [])

        for t in list(validated_terms)[:8]:
            print(f"    Subprocess: '{t}'")

        return patterns

    def _learn_document_knowledge_with_judge(self, sentences, components) -> DocumentKnowledge:
        """V41's approach: Learn abbreviations/synonyms with judge validation."""
        comp_names = [c.name for c in components if c.name not in self.model_knowledge.impl_to_abstract]
        doc_lines = [f"S{s.number}: {s.text}" for s in sentences[:100]]

        prompt1 = f"""Find all ways components are referred to in this document.

COMPONENTS: {', '.join(comp_names)}

DOCUMENT:
{chr(10).join(doc_lines)}

Find:
1. Abbreviations (short forms)
2. Synonyms (alternative names)
3. Partial references

Return JSON:
{{
  "abbreviations": {{"short": "FullComponent"}},
  "synonyms": {{"alternative": "FullComponent"}},
  "partial_references": {{"partial": "FullComponent"}}
}}
JSON only:"""

        data1 = self.llm_client.extract_json(self.llm_client.query(prompt1, timeout=150))

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

            prompt2 = f"""JUDGE: Review these term-to-component mappings.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

Identify terms that are TOO GENERIC to map reliably:
- Common English words
- Terms that could refer to multiple components
- Internal implementation details

Return JSON:
{{
  "approved": ["term1", "term2"],
  "generic_rejected": ["generic_term1"]
}}
JSON only:"""

            data2 = self.llm_client.extract_json(self.llm_client.query(prompt2, timeout=120))

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

    def _is_impl(self, name: str) -> bool:
        return self.model_knowledge and any(i in name for i in self.model_knowledge.impl_indicators)

    def _component_in_dotted_path(self, text: str, comp_name: str) -> bool:
        for path in re.findall(r'\b\w+(?:\.\w+)+\b', text.lower()):
            if comp_name.lower() in path.split('.'):
                return True
        return False

    def _process_transarc(self, transarc_csv, id_to_name, sent_map, name_to_id):
        links = []
        if not transarc_csv or not os.path.exists(transarc_csv):
            return links

        with open(transarc_csv) as f:
            for row in csv.DictReader(f):
                cid, snum = row.get('modelElementID', ''), int(row.get('sentence', 0))
                cname = id_to_name.get(cid, "")
                sent = sent_map.get(snum)
                if not sent or self._component_in_dotted_path(sent.text, cname):
                    continue
                if cname in self.model_knowledge.impl_to_abstract:
                    cname = self.model_knowledge.impl_to_abstract[cname]
                    cid = name_to_id.get(cname, cid)
                conf = 0.92 if cname in self.model_knowledge.architectural_names else 0.90
                links.append(SadSamLink(snum, cid, cname, conf, "transarc"))
        return links

    def _validate_with_self_consistency(self, candidates: list[CandidateLink], components, sent_map) -> list[CandidateLink]:
        if not candidates:
            return []

        comp_names = [c.name for c in components if not self._is_impl(c.name)]
        needs_validation = [c for c in candidates if c.needs_validation]
        direct = [c for c in candidates if not c.needs_validation]

        if not needs_validation:
            return candidates

        ctx = []
        if self.learned_patterns.action_indicators:
            ctx.append(f"ACTION patterns: {', '.join(self.learned_patterns.action_indicators[:4])}")
        if self.learned_patterns.effect_indicators:
            ctx.append(f"EFFECT patterns (reject): {', '.join(self.learned_patterns.effect_indicators[:3])}")
        if self.learned_patterns.subprocess_terms:
            ctx.append(f"Subprocess terms (reject): {', '.join(list(self.learned_patterns.subprocess_terms)[:5])}")

        cases = []
        for i, c in enumerate(needs_validation[:25]):
            prev = sent_map.get(c.sentence_number - 1)
            p = f"[prev: {prev.text[:35]}...] " if prev else ""
            cases.append(f"Case {i+1}: \"{c.matched_text}\" -> {c.component_name}\n  {p}\"{c.sentence_text}\"")

        results_1 = self._run_validation_pass(comp_names, ctx, cases, "Focus on whether component is the ACTOR")
        results_2 = self._run_validation_pass(comp_names, ctx, cases, "Focus on whether component is DIRECTLY referenced")

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

        data = self.llm_client.extract_json(self.llm_client.query(prompt, timeout=120))
        results = {}
        if data:
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    results[idx] = (v.get("valid", False), v.get("confidence", 0.8))
        return results

    def _resolve_coreferences_with_debate(self, sentences, components, name_to_id, sent_map) -> list[SadSamLink]:
        """V41's coreference with debate validation."""
        comp_names = [c.name for c in components if not self._is_impl(c.name)]
        threshold = self.thresholds.coref_threshold

        ctx = []
        if self.learned_patterns.action_indicators:
            ctx.append(f"ACTION: {', '.join(self.learned_patterns.action_indicators[:3])}")
        if self.learned_patterns.subprocess_terms:
            ctx.append(f"Subprocesses (don't link): {', '.join(list(self.learned_patterns.subprocess_terms)[:5])}")

        all_coref = []
        for batch_start in range(0, len(sentences), 20):
            batch = sentences[batch_start:min(batch_start + 20, len(sentences))]
            ctx_start = max(0, batch_start - self.CONTEXT_WINDOW)
            ctx_sents = sentences[ctx_start:batch_start + 20]

            doc_lines = [f"{'*' if s.number >= batch[0].number else ' '}S{s.number}: {s.text}" for s in ctx_sents]

            prompt1 = f"""Resolve pronoun references to components.

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

DOCUMENT (* = analyze):
{chr(10).join(doc_lines)}

Find pronouns (it, they, this) that refer to components.

Return JSON:
{{"resolutions": [{{"sentence": N, "pronoun": "it", "component": "Name", "confidence": 0.0-1.0, "reasoning": "why"}}]}}
JSON only:"""

            data1 = self.llm_client.extract_json(self.llm_client.query(prompt1, timeout=100))
            if not data1:
                continue

            proposed = data1.get("resolutions", [])
            if not proposed:
                continue

            proposal_text = [f"S{r['sentence']}: \"{r.get('pronoun','?')}\" -> {r['component']} ({r.get('reasoning','')[:40]})"
                           for r in proposed[:12]]

            prompt2 = f"""JUDGE: Validate these pronoun-to-component resolutions.

COMPONENTS: {', '.join(comp_names)}

PROPOSALS:
{chr(10).join(proposal_text)}

CONTEXT:
{chr(10).join(doc_lines[:15])}

Reject if:
- Pronoun might refer to something else
- Reference is to a subprocess
- Too far from actual component mention

Return JSON:
{{"judgments": [{{"sentence": N, "approve": true/false, "adjusted_confidence": 0.0-1.0}}]}}
JSON only:"""

            data2 = self.llm_client.extract_json(self.llm_client.query(prompt2, timeout=100))

            judgments = {}
            if data2:
                for j in data2.get("judgments", []):
                    judgments[j.get("sentence")] = (j.get("approve", False), j.get("adjusted_confidence", 0.8))

            for res in proposed:
                snum, comp = res.get("sentence"), res.get("component")
                if not (snum and comp and comp in name_to_id):
                    continue

                if snum in judgments:
                    approved, conf = judgments[snum]
                    if not approved or conf < threshold:
                        continue
                else:
                    conf = res.get("confidence", 0.85)
                    if conf < threshold:
                        continue

                sent = sent_map.get(snum)
                if sent and self.learned_patterns.subprocess_terms:
                    if any(t.lower() in sent.text.lower() for t in self.learned_patterns.subprocess_terms):
                        conf -= 0.05
                        if conf < threshold:
                            continue

                if comp in self.model_knowledge.architectural_names:
                    conf = min(1.0, conf + 0.02)

                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, conf, "coreference"))

        return all_coref

    def _agent_judge_review(self, links, sentences, components, sent_map, transarc_set) -> list[SadSamLink]:
        if len(links) < 5:
            return links

        comp_names = [c.name for c in components if not self._is_impl(c.name)]

        transarc = [l for l in links if (l.sentence_number, l.component_id) in transarc_set]
        non_transarc = [l for l in links if (l.sentence_number, l.component_id) not in transarc_set]

        if not non_transarc:
            return transarc

        cases = []
        for i, l in enumerate(non_transarc[:30]):
            sent = sent_map.get(l.sentence_number)
            prev = sent_map.get(l.sentence_number - 1)
            context = []
            if prev:
                context.append(f"    PREV: {prev.text[:45]}...")
            context.append(f"    >>> S{l.sentence_number}: {sent.text if sent else '?'}")
            cases.append(f"Case {i+1}: S{l.sentence_number} -> {l.component_name} (src:{l.source}, conf:{l.confidence:.2f})\n{chr(10).join(context)}")

        strictness_instruction = {
            "relaxed": "Be INCLUSIVE - accept reasonable links, only reject clear errors.",
            "balanced": "Be BALANCED - accept clear references, reject ambiguous ones.",
            "strict": "Be STRICT - only approve very confident links, reject when in doubt."
        }.get(self.doc_profile.recommended_strictness, "Be balanced.")

        prompt = f"""JUDGE: Review these trace links. {strictness_instruction}

COMPONENTS: {', '.join(comp_names)}

CRITERIA FOR VALID LINKS:
- Component is SUBJECT/ACTOR of the sentence
- Pronouns clearly refer to the component
- About component behavior, not code structure

REJECT IF:
- Sentence describes EFFECT/RESULT, not component action
- Reference is to sub-part, not main component
- Describes package structure or imports

PROPOSED LINKS:
{chr(10).join(cases)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

        data = self.llm_client.extract_json(self.llm_client.query(prompt, timeout=180))
        result = transarc.copy()

        if data:
            judged = set()
            for j in data.get("judgments", []):
                idx = j.get("case", 0) - 1
                if 0 <= idx < len(non_transarc):
                    judged.add(idx)
                    if j.get("approve", False):
                        result.append(non_transarc[idx])
            for i, l in enumerate(non_transarc):
                if i not in judged and i >= 30:
                    result.append(l)
        else:
            result.extend(non_transarc)

        return result

    def _adaptive_fn_recovery(self, current_links, sentences, components, name_to_id, sent_map) -> list[SadSamLink]:
        comp_names = [c.name for c in components if not self._is_impl(c.name)]
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

        cases = [f"S{sn}: \"{txt[:70]}...\" -> {cn}?" for sn, txt, cn in potential_fns[:12]]

        results_1 = self._run_recovery_pass(comp_names, cases, "Is the component the ACTOR in this sentence?")
        results_2 = self._run_recovery_pass(comp_names, cases, "Is this a valid trace link that was missed?")

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
                        print(f"    Recovered: S{snum} -> {cname} (conf={avg_conf:.2f})")

        return result

    def _run_recovery_pass(self, comp_names, cases, question):
        prompt = f"""Check potential missed trace links. {question}

COMPONENTS: {', '.join(comp_names)}

POTENTIAL LINKS:
{chr(10).join(cases)}

Return JSON:
{{"recoveries": [{{"case": 0, "valid": true/false, "confidence": 0.0-1.0}}]}}

Use case index (0-based).
JSON only:"""

        data = self.llm_client.extract_json(self.llm_client.query(prompt, timeout=100))
        results = {}
        if data:
            for r in data.get("recoveries", []):
                idx = r.get("case", -1)
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
