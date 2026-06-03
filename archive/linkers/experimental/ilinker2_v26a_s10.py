"""S10: Design A+B — Prompt-hardened + Trust-the-judge combined.

Enhanced prompts (from S7/Design A) + minimal overrides (from S8/Design B).
- Extraction prompt warns about ambiguous short aliases
- Judge prompt requires formal definition evidence for short aliases
- Fix B removed entirely
- Fix C requires definitional pattern (not just mid-sentence capitalization)
No code-level length thresholds.
"""

import re
from llm_sad_sam.core.data_types import DocumentKnowledge
from llm_sad_sam.linkers.experimental.ilinker2_v26a import ILinker2V26a


class ILinker2V26aS10(ILinker2V26a):
    """V26a + prompt-hardened + trust-the-judge (A+B combined)."""

    def _learn_document_knowledge_enriched(self, sentences, components):
        """Override: Enhanced prompts + no Fix B + definitional Fix C."""
        comp_names = [c.name for c in components]
        doc_lines = [s.text for s in sentences[:150]]

        # Design A: Enhanced extraction prompt
        prompt1 = f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

WHAT TO FIND:
1. ABBREVIATIONS: Short forms explicitly introduced in the document.
   Rule: The abbreviation must be FORMALLY DEFINED in the text with a parenthetical pattern,
   e.g., "Full Name (FN)" introduces FN, or "FN (Full Name)" introduces FN.
   Only propose abbreviations you can point to a specific definitional sentence for.
   Do NOT propose abbreviations that merely "seem likely" — require explicit textual evidence.

2. SYNONYMS: Alternative names that SPECIFICALLY refer to one component.
   Rule: The alternative name must unambiguously identify exactly ONE component.
   APPROVE: A proper name, role title, or technical alias used interchangeably with the component
   REJECT: A generic description that could apply to anything (like "the system" or "the process")

3. PARTIAL REFERENCES: A shorter form of a multi-word component name used alone.
   Rule: A trailing word from a multi-word name that, in this document, consistently means the full name.
   APPROVE: Only if the short form is DISTINCTIVE — it would not be confused with any other concept
   REJECT: Common abbreviations or words that have well-known meanings beyond this specific component
   REJECT: Any partial that is also a widely-used acronym or abbreviation in computing
   (e.g., if a component is "MemoryManager", do not propose "MM" as a partial — "MM" is too
   generic to be a distinctive reference to that specific component)

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

        if all_mappings:
            mapping_list = [f"'{k}' -> {v[1]} ({v[0]})" for k, v in list(all_mappings.items())[:25]]

            # Design A: Enhanced judge prompt
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
- The term is a widely-used computing abbreviation or acronym (like API, OS, IO, VM, CPU)
  that is NOT formally defined in this document as referring to the specific component.
  A formal definition requires explicit text like "ComponentName (Abbrev)" or equivalent.

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

            # Fix A: CamelCase override (structural — always valid)
            for term in list(generic_terms):
                if re.search(r'[a-z][A-Z]', term):
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    CamelCase override (rescued): {term}")

            # Fix B: REMOVED (Design B — trust the judge)

            # V24: Deterministic overrides (structural evidence)
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

            # Fix C: Design B — definitional patterns only
            for term in list(generic_terms):
                if term not in all_mappings:
                    continue
                _, target_comp = all_mappings[term]
                if ' ' in term or not term[0].isupper() or target_comp not in comp_names:
                    continue
                found_def = False
                for s in sentences[:100]:
                    if re.search(rf'\b{re.escape(target_comp)}\b\s*\(\s*{re.escape(term)}\s*\)', s.text):
                        found_def = True
                        break
                    if re.search(rf'\b{re.escape(term)}\b\s*\(\s*{re.escape(target_comp)}\s*\)', s.text):
                        found_def = True
                        break
                    if re.search(rf'\b{re.escape(term)}\b.*(?:also known as|also called|i\.e\.,?|a\.k\.a\.)\s*{re.escape(target_comp)}\b', s.text, re.IGNORECASE):
                        found_def = True
                        break
                    if re.search(rf'\b{re.escape(target_comp)}\b.*(?:also known as|also called|i\.e\.,?|a\.k\.a\.)\s*{re.escape(term)}\b', s.text, re.IGNORECASE):
                        found_def = True
                        break
                if found_def:
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    Definitional override (rescued): {term}")
                else:
                    print(f"    Fix C SKIPPED (no definitional pattern): {term}")
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
