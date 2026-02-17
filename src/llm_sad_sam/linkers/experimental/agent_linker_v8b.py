"""AgentLinker V8b: Soft learned hints for judge.

Key insight from V7 experiments:
- V7's learned confusion patterns caused recall regression (89.2% vs 91.4%)
- Problem: patterns were used as REJECTION RULES, causing over-filtering
- Solution: use patterns as soft AWARENESS HINTS — the judge sees extra context
  about which component names are ambiguous, but the approval criteria stay the same

Approach:
- Same Phase 2b confusion learning as V7
- But inject patterns into the judge as "CONTEXT NOTES" rather than "REJECT IF"
- Judge still uses standard V6 approval/rejection criteria
- Patterns just help the judge be more informed about document-specific risks
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
from .agent_linker_v6 import AgentLinkerV6, _EmbeddingFilter


class AgentLinkerV8b(AgentLinkerV6):
    """V6 + soft learned hints (confusion patterns as context, not rules)."""

    def __init__(self, backend: Optional[LLMBackend] = None,
                 post_filter: str = "none"):
        super().__init__(backend=backend, post_filter=post_filter)
        self._confusion_patterns = []
        print("AgentLinkerV8b: Soft learned hints")

    # ═════════════════════════════════════════════════════════════════════
    # Phase 2b: Confusion Pattern Learning (same as V7)
    # ═════════════════════════════════════════════════════════════════════

    def _learn_confusion_patterns(self, sentences, components):
        """Analyze document to identify component names with confusion risk."""
        comp_names = self._get_comp_names(components)

        evidence = defaultdict(list)
        for sent in sentences:
            for cname in comp_names:
                if re.search(rf'\b{re.escape(cname)}\b', sent.text, re.IGNORECASE):
                    evidence[cname].append(f"S{sent.number}: {sent.text[:100]}")

        evidence_text = []
        for cname in comp_names:
            examples = evidence.get(cname, [])
            if examples:
                evidence_text.append(f"\n{cname} ({len(examples)} mentions):")
                for ex in examples[:5]:
                    evidence_text.append(f"  {ex}")
            else:
                evidence_text.append(f"\n{cname}: no mentions in text")

        prompt = f"""Analyze these software architecture component names and the document sentences where they appear. Identify CONFUSION RISKS — cases where the component name might be matched incorrectly.

COMPONENTS: {', '.join(comp_names)}

MENTIONS IN DOCUMENT:
{''.join(evidence_text)}

For each component, analyze:
1. Is the name also a common English word? (e.g. "Logic", "Client", "Storage", "Common")
2. Does the name appear in package/module dot-notation paths? (e.g. "logic.api", "e2e.util")
3. Does the name appear as a technology/tool reference rather than the architectural component? (e.g. "WebRTC" as technology vs "WebRTC-SFU" as component)
4. Does the name appear as part of compound expressions where it means something different? (e.g. "client-side", "cascade logic", "voice conference")
5. If the component is multi-word, does a single word from it appear alone in misleading contexts? (e.g. "Server" alone when "HTML5 Server" is the component, "Conversion" alone when "Presentation Conversion" is the component)

Return JSON:
{{"confusion_patterns": [
  {{
    "component": "ComponentName",
    "risk_type": "generic_word|package_path|technology_ref|compound_expression|partial_mislead",
    "pattern": "the specific word/phrase pattern to watch for",
    "explanation": "why this is misleading — when does it NOT refer to the architectural component?"
  }}
]}}

Only report patterns where there is REAL risk based on the document evidence.
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
        patterns = []
        if data:
            for p in data.get("confusion_patterns", []):
                comp = p.get("component", "")
                if comp in comp_names:
                    patterns.append({
                        "component": comp,
                        "risk_type": p.get("risk_type", "unknown"),
                        "pattern": p.get("pattern", ""),
                        "explanation": p.get("explanation", ""),
                    })
        return patterns

    # ═════════════════════════════════════════════════════════════════════
    # Override link() to add Phase 2b
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

        if self.post_filter != "none":
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

        # Phase 1
        print("\n[Phase 1] Model Structure")
        self.model_knowledge = self._analyze_model(components)
        print(f"  Architectural: {len(self.model_knowledge.architectural_names)}")
        print(f"  Ambiguous: {self.model_knowledge.ambiguous_names}")

        # Phase 2
        print("\n[Phase 2] Pattern Learning")
        self.learned_patterns = self._learn_patterns_with_debate(sentences, components)
        print(f"  Subprocess terms: {len(self.learned_patterns.subprocess_terms)}")

        # Phase 2b: Confusion pattern learning
        print("\n[Phase 2b] Confusion Pattern Learning")
        self._confusion_patterns = self._learn_confusion_patterns(sentences, components)
        by_type = defaultdict(int)
        for p in self._confusion_patterns:
            by_type[p["risk_type"]] += 1
        print(f"  Learned {len(self._confusion_patterns)} patterns: {dict(by_type)}")
        for p in self._confusion_patterns:
            print(f"    {p['component']}: {p['risk_type']} — \"{p['pattern']}\"")
        self._log("phase_2b", {}, {"count": len(self._confusion_patterns),
                                    "patterns": self._confusion_patterns})

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

        if self._semantic_filter is not None:
            print("\n[Semantic Filter Calibration]")
            self._semantic_filter.calibrate_from_known_links(transarc_links)

        # Phase 5
        print("\n[Phase 5] Entity Extraction")
        candidates = self._extract_entities(sentences, components, name_to_id, sent_map)
        print(f"  Candidates: {len(candidates)}")

        # Phase 6
        print("\n[Phase 6] Validation")
        validated = self._validate_with_self_consistency(candidates, components, sent_map)
        print(f"  Validated: {len(validated)} (of {len(candidates)})")

        # Phase 7
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

        # Phase 8
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

        # Phase 8b
        partial_links = self._inject_partial_references(
            sentences, components, name_to_id, transarc_set,
            {(c.sentence_number, c.component_id) for c in validated},
            {(l.sentence_number, l.component_id) for l in coref_links},
            {(l.sentence_number, l.component_id) for l in implicit_links},
        )
        if partial_links:
            print(f"\n[Phase 8b] Partial Injection")
            print(f"  Injected: {len(partial_links)} candidates")

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

        # Phase 9: Judge review (with soft hints)
        print("\n[Phase 9] Judge Review")
        reviewed = self._judge_review(preliminary, sentences, components, sent_map, transarc_set)
        rejected = [l for l in preliminary if (l.sentence_number, l.component_id)
                    not in {(r.sentence_number, r.component_id) for r in reviewed}]
        print(f"  Approved: {len(reviewed)} (rejected {len(rejected)})")

        # Phase 10
        print("\n[Phase 10] FN Recovery")
        final = self._fn_recovery(reviewed, sentences, components, name_to_id, sent_map)
        recovered = [l for l in final if l.source == "recovered"]
        print(f"  Final: {len(final)} (+{len(recovered)} recovered)")

        if self._semantic_filter is not None and self._semantic_filter.threshold is not None:
            before = len(final)
            final = self._apply_selective_filter(final, sent_map)
            print(f"  Semantic filter: removed {before - len(final)}")

        self._log("summary", {"total_time_s": round(time.time() - t0, 1)},
                  {"final": len(final)}, final)
        self._save_log(text_path)

        print(f"\nFinal: {len(final)} links")
        return final

    # ═════════════════════════════════════════════════════════════════════
    # Override judge prompt: SOFT HINTS instead of hard rejection rules
    # ═════════════════════════════════════════════════════════════════════

    def _build_judge_prompt(self, comp_names, cases):
        """Judge prompt with confusion patterns as AWARENESS CONTEXT, not rules."""

        # Build awareness notes from learned patterns
        awareness_notes = []
        seen_components = set()
        for p in self._confusion_patterns:
            comp = p["component"]
            if comp in seen_components:
                continue
            seen_components.add(comp)
            notes = [cp for cp in self._confusion_patterns if cp["component"] == comp]
            risks = []
            for n in notes[:3]:  # Max 3 per component
                risks.append(f'"{n["pattern"]}" ({n["risk_type"]})')
            awareness_notes.append(f"  - {comp}: watch for {', '.join(risks)}")

        awareness_section = ""
        if awareness_notes:
            awareness_section = f"""
DOCUMENT-SPECIFIC AWARENESS (component names that could be confused):
{chr(10).join(awareness_notes)}
NOTE: These are just awareness hints. The component name CAN still be a valid architectural reference even when it matches a pattern above. Judge each case on its own merits.
"""

        return f"""JUDGE: Review trace links. Accept references where the sentence is ABOUT the component, reject false matches.

COMPONENTS: {', '.join(comp_names)}
{awareness_section}
VALID IF: Sentence mentions or relates to the component as actor, target, location, object, or participant in an architectural interaction. Section headings that name the component are VALID (they introduce a section about that component).
REJECT IF:
- Component name used as a generic English word, not an architecture reference. The word must refer to the SPECIFIC ARCHITECTURAL COMPONENT, not the general concept.
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
