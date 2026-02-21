"""AgentLinker V7: LLM-learned rejection patterns for judge.

Changes from V6:
- NEW Phase 2b: "Confusion Pattern Learning" — LLM analyzes the document to identify
  specific patterns that could cause false positives:
  * Component names that are also common English words (e.g. "Logic", "Client", "Storage")
  * Component names that appear in package/module paths
  * Component names that appear as technology references (not architectural)
  * Partial matches that are too generic (e.g. "Conversion" matching any conversion mention)
- Judge prompt is dynamically built using learned confusion patterns instead of generic examples
- All other V6 improvements preserved
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


class AgentLinkerV7(AgentLinkerV6):
    """V6 + LLM-learned confusion patterns for dynamic judge prompts."""

    def __init__(self, backend: Optional[LLMBackend] = None,
                 post_filter: str = "none"):
        super().__init__(backend=backend, post_filter=post_filter)
        self._confusion_patterns = []  # Learned in Phase 2b
        print("AgentLinkerV7: Learned rejection patterns")

    # ═════════════════════════════════════════════════════════════════════
    # Phase 2b: Confusion Pattern Learning
    # ═════════════════════════════════════════════════════════════════════

    def _learn_confusion_patterns(self, sentences, components):
        """Ask the LLM to analyze the document and identify component names
        that could be confused with generic English words, technology names,
        package paths, or other non-architectural usages.

        Returns a list of learned confusion patterns that will be injected
        into the judge prompt.
        """
        comp_names = self._get_comp_names(components)

        # Gather evidence: find sentences where component names appear
        evidence = defaultdict(list)
        for sent in sentences:
            for cname in comp_names:
                if re.search(rf'\b{re.escape(cname)}\b', sent.text, re.IGNORECASE):
                    evidence[cname].append(f"S{sent.number}: {sent.text[:100]}")

        # Build evidence summary (limit to 5 examples per component)
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

        # Semantic filter init
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
        self._log("phase_0", {"sents": len(sentences), "comps": len(components)},
                  {"spc": spc, "complex": self._is_complex})

        # Phase 1
        print("\n[Phase 1] Model Structure")
        self.model_knowledge = self._analyze_model(components)
        print(f"  Architectural: {len(self.model_knowledge.architectural_names)}")
        print(f"  Ambiguous: {self.model_knowledge.ambiguous_names}")
        self._log("phase_1", {}, {"ambiguous": sorted(self.model_knowledge.ambiguous_names)})

        # Phase 2
        print("\n[Phase 2] Pattern Learning")
        self.learned_patterns = self._learn_patterns_with_debate(sentences, components)
        print(f"  Subprocess terms: {len(self.learned_patterns.subprocess_terms)}")
        self._log("phase_2", {}, {"subprocess": sorted(self.learned_patterns.subprocess_terms)})

        # Phase 2b: Confusion pattern learning (NEW)
        print("\n[Phase 2b] Confusion Pattern Learning")
        self._confusion_patterns = self._learn_confusion_patterns(sentences, components)
        by_type = defaultdict(int)
        for p in self._confusion_patterns:
            by_type[p["risk_type"]] += 1
        print(f"  Learned {len(self._confusion_patterns)} patterns: {dict(by_type)}")
        for p in self._confusion_patterns:
            print(f"    {p['component']}: {p['risk_type']} — \"{p['pattern']}\" ({p['explanation'][:60]})")
        self._log("phase_2b", {}, {
            "count": len(self._confusion_patterns),
            "patterns": self._confusion_patterns,
        })

        # Phase 3
        print("\n[Phase 3] Document Knowledge")
        self.doc_knowledge = self._learn_document_knowledge_with_judge(sentences, components)
        print(f"  Abbrev: {len(self.doc_knowledge.abbreviations)}, "
              f"Syn: {len(self.doc_knowledge.synonyms)}, "
              f"Generic: {len(self.doc_knowledge.generic_terms)}")
        self._log("phase_3", {}, {
            "abbreviations": self.doc_knowledge.abbreviations,
            "synonyms": self.doc_knowledge.synonyms,
        })

        # Phase 3b
        self._enrich_multiword_partials(sentences, components)

        # Phase 4
        print("\n[Phase 4] TransArc")
        transarc_links = self._process_transarc(transarc_csv, id_to_name, sent_map, name_to_id)
        transarc_set = {(l.sentence_number, l.component_id) for l in transarc_links}
        print(f"  Links: {len(transarc_links)}")
        self._log("phase_4", {}, {"count": len(transarc_links)}, transarc_links)

        # Calibrate embedding filter
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

        # Phase 9: Judge review (with learned confusion patterns)
        print("\n[Phase 9] Judge Review")
        reviewed = self._judge_review(preliminary, sentences, components, sent_map, transarc_set)
        rejected = [l for l in preliminary if (l.sentence_number, l.component_id)
                    not in {(r.sentence_number, r.component_id) for r in reviewed}]
        print(f"  Approved: {len(reviewed)} (rejected {len(rejected)})")

        # Phase 10: FN recovery
        print("\n[Phase 10] FN Recovery")
        final = self._fn_recovery(reviewed, sentences, components, name_to_id, sent_map)
        recovered = [l for l in final if l.source == "recovered"]
        print(f"  Final: {len(final)} (+{len(recovered)} recovered)")

        # Post-filter
        if self._semantic_filter is not None and self._semantic_filter.threshold is not None:
            before = len(final)
            final = self._apply_selective_filter(final, sent_map)
            print(f"  Semantic filter: removed {before - len(final)}")

        # Save log
        self._log("summary", {"total_time_s": round(time.time() - t0, 1)},
                  {"final": len(final)}, final)
        self._save_log(text_path)

        print(f"\nFinal: {len(final)} links")
        return final

    # ═════════════════════════════════════════════════════════════════════
    # Override judge prompt to use learned confusion patterns
    # ═════════════════════════════════════════════════════════════════════

    def _build_judge_prompt(self, comp_names, cases):
        """Build judge prompt with document-specific confusion patterns."""

        # Group confusion patterns by type for readable output
        generic_word = []
        package_path = []
        tech_ref = []
        compound_expr = []
        partial_mislead = []

        for p in self._confusion_patterns:
            entry = f'"{p["pattern"]}" for {p["component"]} ({p["explanation"][:80]})'
            rt = p["risk_type"]
            if rt == "generic_word":
                generic_word.append(entry)
            elif rt == "package_path":
                package_path.append(entry)
            elif rt == "technology_ref":
                tech_ref.append(entry)
            elif rt == "compound_expression":
                compound_expr.append(entry)
            elif rt == "partial_mislead":
                partial_mislead.append(entry)

        # Build dynamic REJECT section
        reject_rules = []

        reject_rules.append("- Component name used as a generic English word, not an architecture reference. "
                          "The word must refer to the SPECIFIC ARCHITECTURAL COMPONENT, not the general concept.")
        if generic_word:
            reject_rules.append(f"  WATCH FOR in this document: {'; '.join(generic_word[:5])}")

        reject_rules.append("- Sentence describes package/directory/module structure using dot notation.")
        if package_path:
            reject_rules.append(f"  WATCH FOR in this document: {'; '.join(package_path[:5])}")

        reject_rules.append("- Component name appears only as a technology reference, not referring to the actual architectural component.")
        if tech_ref:
            reject_rules.append(f"  WATCH FOR in this document: {'; '.join(tech_ref[:5])}")

        reject_rules.append("- Component name appears in a compound expression with a different meaning.")
        if compound_expr:
            reject_rules.append(f"  WATCH FOR in this document: {'; '.join(compound_expr[:5])}")

        reject_rules.append("- A partial match of a multi-word component name that actually refers to something else.")
        if partial_mislead:
            reject_rules.append(f"  WATCH FOR in this document: {'; '.join(partial_mislead[:5])}")

        reject_rules.append("- Subprocess/sub-task of the component, not the component itself.")

        reject_section = "\n".join(reject_rules)

        return f"""JUDGE: Review trace links. Accept references where the sentence is ABOUT the component, reject false matches.

COMPONENTS: {', '.join(comp_names)}

VALID IF: Sentence mentions or relates to the component as actor, target, location, object, or participant in an architectural interaction. Section headings that name the component are VALID (they introduce a section about that component).

REJECT IF:
{reject_section}

SOURCE-SPECIFIC RULES:
- "transarc": Check carefully — is the component name an ARCHITECTURE REFERENCE or a GENERIC WORD / PACKAGE PATH in this sentence?
- "implicit": Be skeptical — accept only if component is clearly the topic of discussion.
- "coreference": Verify the pronoun clearly refers to the claimed component.

LINKS:
{chr(10).join(cases)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

    def _judge_recovered(self, candidates, comp_names, sent_map):
        """Strict judge for recovered links — uses learned confusion patterns."""
        cases = []
        for i, lk in enumerate(candidates):
            sent = sent_map.get(lk.sentence_number)
            prev = sent_map.get(lk.sentence_number - 1)
            ctx_lines = []
            if prev:
                ctx_lines.append(f"    PREV: {prev.text[:50]}...")
            ctx_lines.append(f"    >>> S{lk.sentence_number}: {sent.text if sent else '?'}")
            cases.append(f"Case {i+1}: S{lk.sentence_number} -> {lk.component_name}\n" + chr(10).join(ctx_lines))

        # Build dynamic rejection patterns from learned confusion patterns
        watch_for = []
        for p in self._confusion_patterns:
            watch_for.append(f'- "{p["pattern"]}" for {p["component"]}: {p["explanation"][:60]}')

        watch_section = ""
        if watch_for:
            watch_section = "\nKNOWN CONFUSION PATTERNS FOR THIS DOCUMENT:\n" + "\n".join(watch_for[:10]) + "\n"

        prompt = f"""JUDGE: These are RECOVERED links — sentences that mention a component but were not linked by any earlier phase. Be STRICT: only approve if the component is clearly the ACTOR/SUBJECT performing an action.

COMPONENTS: {', '.join(comp_names)}
{watch_section}
REJECT IF:
- Component name appears as a modifier, adjective, or in a compound word
- Component name is used as a generic English word, not an architecture reference
- Sentence describes package/directory/module structure using dot notation
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
            return []

        approved = []
        for j in data.get("judgments", []):
            idx = j.get("case", 0) - 1
            if 0 <= idx < len(candidates) and j.get("approve", False):
                approved.append(candidates[idx])
        print(f"    Recovery judge: {len(approved)}/{len(candidates)} approved")
        return approved
