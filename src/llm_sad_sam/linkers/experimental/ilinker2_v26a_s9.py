"""S9: Design C — Document-frequency gating.

Replace length thresholds with frequency-based filtering.
If an alias matches more than 20% of document sentences, it's suspicious —
demote from auto-approved to LLM-needed (Phase 6) or skip injection (Phase 8b).

Fix B/C: only rescue if the term's document frequency is below 20%.
No hardcoded length thresholds.
"""

import re
from llm_sad_sam.core.data_types import DocumentKnowledge, SadSamLink
from llm_sad_sam.linkers.experimental.ilinker2_v26a import ILinker2V26a


class ILinker2V26aS9(ILinker2V26a):
    """V26a + document-frequency gating (no length thresholds)."""

    # Cache for document frequency of aliases
    _alias_doc_freq = None
    _total_sentences = 0
    MAX_FREQ_RATIO = 0.20  # aliases matching > 20% of sentences are suspicious

    def _compute_alias_doc_freq(self, sentences):
        """Compute document frequency for all known aliases."""
        if self._alias_doc_freq is not None:
            return
        self._alias_doc_freq = {}
        self._total_sentences = len(sentences)
        if not self.doc_knowledge:
            return
        all_aliases = {}
        for a, c in self.doc_knowledge.abbreviations.items():
            all_aliases[a] = c
        for s, c in self.doc_knowledge.synonyms.items():
            all_aliases[s] = c
        for p, c in self.doc_knowledge.partial_references.items():
            all_aliases[p] = c
        for alias in all_aliases:
            count = 0
            pattern = rf'\b{re.escape(alias)}\b'
            for sent in sentences:
                if re.search(pattern, sent.text, re.IGNORECASE):
                    count += 1
            self._alias_doc_freq[alias] = count
            if self._total_sentences > 0:
                ratio = count / self._total_sentences
                if ratio > self.MAX_FREQ_RATIO:
                    print(f"    High-freq alias: '{alias}' matches {count}/{self._total_sentences} "
                          f"({ratio:.0%}) sentences")

    def _is_high_freq_alias(self, alias):
        """Check if alias has high document frequency."""
        if not self._alias_doc_freq or self._total_sentences == 0:
            return False
        count = self._alias_doc_freq.get(alias, 0)
        return (count / self._total_sentences) > self.MAX_FREQ_RATIO

    def _learn_document_knowledge_enriched(self, sentences, components):
        """Override: Fix B/C guarded by document frequency instead of length."""
        comp_names = [c.name for c in components]
        doc_lines = [s.text for s in sentences[:150]]

        prompt1 = f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

WHAT TO FIND:
1. ABBREVIATIONS: Short forms explicitly introduced in the document.
   Rule: The abbreviation must be defined in the text, e.g., "Full Name (FN)" introduces FN.
   Like "Abstract Syntax Tree (AST)" defines AST — look for the same parenthetical pattern.

2. SYNONYMS: Alternative names that SPECIFICALLY refer to one component.
   Rule: The alternative name must unambiguously identify exactly ONE component.
   APPROVE: A proper name, role title, or technical alias used interchangeably with the component
   REJECT: A generic description that could apply to anything (like "the system" or "the process")

3. PARTIAL REFERENCES: A shorter form of a multi-word component name used alone.
   Rule: A trailing word from a multi-word name that, in this document, consistently means the full name.
   APPROVE: Only if the short form is unambiguous — no other component shares this word
   REJECT: Common words that have ordinary English meanings beyond the component

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": {{"short_form": "FullComponent"}},
  "synonyms": {{"specific_alternative_name": "FullComponent"}},
  "partial_references": {{"partial_name": "FullComponent"}}
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

        # Pre-compute document frequency for proposed aliases
        alias_freq = {}
        for alias in all_mappings:
            count = 0
            pattern = rf'\b{re.escape(alias)}\b'
            for s in sentences:
                if re.search(pattern, s.text, re.IGNORECASE):
                    count += 1
            alias_freq[alias] = count
            total = len(sentences)
            if total > 0 and (count / total) > self.MAX_FREQ_RATIO:
                print(f"    High-freq proposed: '{alias}' matches {count}/{total} "
                      f"({count/total:.0%}) sentences")

        if all_mappings:
            mapping_list = [f"'{k}' -> {v[1]} ({v[0]})" for k, v in list(all_mappings.items())[:25]]

            prompt2 = f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

Apply these rules:

REJECT if ANY of these are true:
- The term is used in its ordinary English sense, NOT as a name for the component
  (e.g., "the scheduler runs every minute" uses "scheduler" as a generic concept, not as a named component)
- The term refers to a different component or to the system as a whole
- The mapping cannot be verified from the actual document text

APPROVE if ALL of these are true:
- The term is used AS A NAME for the component in context (even if the word exists in a dictionary)
  (e.g., "The Dispatcher routes incoming requests" uses "Dispatcher" as a proper name for a specific component)
- The term appears in the document in a context that makes the reference clear
- The term unambiguously identifies exactly one component

Return JSON:
{{
  "approved": ["term1", "term2"],
  "generic_rejected": ["generic_term1"]
}}
JSON only:"""

            data2 = self.llm.extract_json(self.llm.query(prompt2, timeout=120))
            approved = set(data2.get("approved", [])) if data2 else set(all_mappings.keys())
            generic_terms = set(data2.get("generic_rejected", [])) if data2 else set()

            total = len(sentences)

            # Fix A: CamelCase override (structural — always valid)
            for term in list(generic_terms):
                if re.search(r'[a-z][A-Z]', term):
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    CamelCase override (rescued): {term}")

            # Fix B MODIFIED: Only rescue if document frequency is low
            for term in list(generic_terms):
                if term.isupper() and len(term) <= 4 and term in all_mappings:
                    freq = alias_freq.get(term, 0)
                    ratio = freq / total if total > 0 else 0
                    if ratio <= self.MAX_FREQ_RATIO:
                        generic_terms.discard(term)
                        approved.add(term)
                        print(f"    Uppercase override (rescued, freq={freq}/{total}): {term}")
                    else:
                        print(f"    Uppercase override BLOCKED (high freq {freq}/{total}={ratio:.0%}): {term}")

            # V24: Deterministic overrides
            for term in list(generic_terms):
                if term not in all_mappings:
                    continue
                if any(term.lower() == cn.lower() for cn in comp_names):
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    Exact-component override (rescued): {term}")
                elif term[0].isupper() and ' ' in term:
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    Multi-word proper-name override (rescued): {term}")

            # Fix C MODIFIED: Only rescue if document frequency is low
            for term in list(generic_terms):
                if term not in all_mappings:
                    continue
                _, target_comp = all_mappings[term]
                if ' ' in term or not term[0].isupper() or target_comp not in comp_names:
                    continue
                freq = alias_freq.get(term, 0)
                ratio = freq / total if total > 0 else 0
                if ratio > self.MAX_FREQ_RATIO:
                    print(f"    Fix C BLOCKED (high freq {freq}/{total}={ratio:.0%}): {term}")
                    continue
                for s in sentences[:100]:
                    for m in re.finditer(rf'\b{re.escape(term)}\b', s.text):
                        if m.start() > 0:
                            generic_terms.discard(term)
                            approved.add(term)
                            print(f"    Capitalized-synonym override (rescued, freq={freq}/{total}): {term}")
                            break
                    if term in approved:
                        break
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

        # Deterministic CamelCase-split synonym injection
        for comp in [c.name for c in components]:
            split = re.sub(r'([a-z])([A-Z])', r'\1 \2', comp)
            split = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', split)
            if split != comp and split not in knowledge.synonyms:
                knowledge.synonyms[split] = comp
                print(f"    CamelCase syn: {split} -> {comp}")

        return knowledge

    def _validate_intersect(self, candidates, components, sent_map):
        """Override: frequency-gated code-first instead of length-gated."""
        if not candidates:
            return []

        # Compute doc frequency for all aliases
        all_sents = list(sent_map.values())
        self._compute_alias_doc_freq(all_sents)

        comp_names = self._get_comp_names(components)
        needs = [c for c in candidates if c.needs_validation]
        direct = [c for c in candidates if not c.needs_validation]

        if not needs:
            return candidates

        # Pre-check: reject generic mentions
        remaining = []
        for c in needs:
            sent = sent_map.get(c.sentence_number)
            if sent and self._is_generic_mention(c.component_name, sent.text):
                print(f"    Generic mention reject: S{c.sentence_number} -> {c.component_name}")
            else:
                remaining.append(c)
        needs = remaining

        # Build alias lookup
        alias_map = {}
        for c in components:
            aliases = {c.name}
            if self.doc_knowledge:
                for a, cn in self.doc_knowledge.abbreviations.items():
                    if cn == c.name:
                        aliases.add(a)
                for s, cn in self.doc_knowledge.synonyms.items():
                    if cn == c.name:
                        aliases.add(s)
                for p, cn in self.doc_knowledge.partial_references.items():
                    if cn == c.name:
                        aliases.add(p)
            alias_map[c.name] = aliases

        # Step 1: Code-first — frequency-gated instead of length-gated
        auto_approved = []
        llm_needed = []
        for c in needs:
            sent = sent_map.get(c.sentence_number)
            if not sent:
                continue
            matched = False
            for a in alias_map.get(c.component_name, set()):
                # High-frequency aliases get demoted to LLM validation
                if self._is_high_freq_alias(a):
                    continue
                if len(a) >= 3:
                    if a.lower() in sent.text.lower():
                        matched = True
                        break
                elif len(a) >= 2:
                    if self._word_boundary_match(a, sent.text):
                        matched = True
                        break
            if matched:
                c.confidence = 1.0
                c.source = "validated"
                auto_approved.append(c)
            else:
                llm_needed.append(c)

        print(f"    Code-first freq-gated auto-approved: {len(auto_approved)}, LLM needed: {len(llm_needed)}")

        # Classify generic-risk components
        generic_risk = set()
        if self.model_knowledge and self.model_knowledge.ambiguous_names:
            generic_risk |= self.model_knowledge.ambiguous_names
        for c in components:
            if c.name.lower() in self.GENERIC_COMPONENT_WORDS:
                generic_risk.add(c.name)

        # Step 2: 2-pass intersect for ALL LLM-needed
        ctx = []
        if self.learned_patterns:
            if self.learned_patterns.action_indicators:
                ctx.append(f"ACTION: {', '.join(self.learned_patterns.action_indicators[:4])}")
            if self.learned_patterns.effect_indicators:
                ctx.append(f"EFFECT (reject): {', '.join(self.learned_patterns.effect_indicators[:3])}")
            if self.learned_patterns.subprocess_terms:
                ctx.append(f"Subprocess (reject): {', '.join(list(self.learned_patterns.subprocess_terms)[:5])}")

        twopass_approved = []
        generic_to_verify = []
        for batch_start in range(0, len(llm_needed), 25):
            batch = llm_needed[batch_start:batch_start + 25]
            cases = []
            for i, c in enumerate(batch):
                prev = sent_map.get(c.sentence_number - 1)
                p = f"[prev: {prev.text[:35]}...] " if prev else ""
                cases.append(f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n  {p}"{c.sentence_text}"')

            r1 = self._qual_validation_pass(comp_names, ctx, cases,
                "Focus on ACTOR role: is the component performing an action or being described?")
            r2 = self._qual_validation_pass(comp_names, ctx, cases,
                "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component, not a generic concept?")

            for i, c in enumerate(batch):
                if r1.get(i, False) and r2.get(i, False):
                    if c.component_name in generic_risk:
                        generic_to_verify.append(c)
                    else:
                        c.confidence = 1.0
                        c.source = "validated"
                        twopass_approved.append(c)

        print(f"    2-pass approved: {len(twopass_approved)} specific, "
              f"{len(generic_to_verify)} generic need evidence")

        # Step 3: Evidence post-filter for generic-risk that passed 2-pass
        generic_validated = []
        if generic_to_verify:
            for batch_start in range(0, len(generic_to_verify), 25):
                batch = generic_to_verify[batch_start:batch_start + 25]
                cases = []
                for i, c in enumerate(batch):
                    cases.append(
                        f'Case {i+1}: S{c.sentence_number} "{c.sentence_text}"\n'
                        f'  Candidate: {c.component_name}'
                    )

                prompt = f"""For each case, find the EXACT text in the sentence that refers to the architecture component.

COMPONENTS: {', '.join(comp_names)}

CASES:
{chr(10).join(cases)}

For each case, provide:
- evidence_text: the EXACT substring from the sentence that names or references the component
- If you cannot find specific text evidence, set evidence_text to null

Return JSON:
{{"validations": [{{"case": 1, "evidence_text": "exact substring or null"}}]}}
JSON only:"""

                data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
                if not data:
                    continue

                for v in data.get("validations", []):
                    idx = v.get("case", 0) - 1
                    if idx < 0 or idx >= len(batch):
                        continue
                    c = batch[idx]
                    evidence = v.get("evidence_text")
                    if not evidence:
                        continue
                    sent = sent_map.get(c.sentence_number)
                    if not sent:
                        continue
                    if evidence.lower() not in sent.text.lower():
                        continue
                    ev_lower = evidence.lower()
                    aliases = alias_map.get(c.component_name, {c.component_name.lower()})
                    if any(a.lower() in ev_lower for a in aliases if len(a) >= 2):
                        c.confidence = 1.0
                        c.source = "validated"
                        generic_validated.append(c)

            print(f"    Generic evidence: {len(generic_validated)}/{len(generic_to_verify)}")

        return direct + auto_approved + twopass_approved + generic_validated

    def _inject_partial_references(self, sentences, components, name_to_id,
                                    transarc_set, validated_set, coref_set, implicit_set):
        """Override: skip high-frequency partial references."""
        if not self.doc_knowledge or not self.doc_knowledge.partial_references:
            return []

        # Ensure doc frequency is computed
        self._compute_alias_doc_freq(sentences)

        existing = transarc_set | validated_set | coref_set | implicit_set
        injected = []

        for partial, comp_name in self.doc_knowledge.partial_references.items():
            if self._is_high_freq_alias(partial):
                print(f"    Partial inject SKIPPED (high freq): '{partial}' -> {comp_name}")
                continue
            if comp_name not in name_to_id:
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
