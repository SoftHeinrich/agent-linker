"""S4: Multi-Run Voting — run extraction 3x, keep majority mappings.

Runs the Phase 3 extraction LLM call 3 times, keeps only mappings proposed
in >= 2/3 runs. Judge runs once on the filtered set. Adds ~30s per dataset.
Effect: Eliminates extraction variance. Low-frequency proposals like "registry"
(8% of runs) are eliminated.
"""

import re
from collections import Counter
from llm_sad_sam.core.data_types import DocumentKnowledge
from llm_sad_sam.linkers.experimental.ilinker2_v26a import ILinker2V26a


class ILinker2V26aS4(ILinker2V26a):
    """V26a + 3x extraction voting for Phase 3 stability."""

    def _learn_document_knowledge_enriched(self, sentences, components):
        """Override: run extraction 3 times, keep majority-voted mappings."""
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

        # Run extraction 3 times, keep majority
        category_to_type = {
            "abbreviations": "abbrev",
            "synonyms": "synonym",
            "partial_references": "partial",
        }
        term_counts = Counter()
        term_data = {}  # last seen (typ, comp) for each term

        n_runs = 3
        for run_idx in range(n_runs):
            data1 = self.llm.extract_json(self.llm.query(prompt1, timeout=150))
            if not data1:
                print(f"    Voting run {run_idx+1}: empty extraction")
                continue
            run_terms = set()
            for category, typ in category_to_type.items():
                for term, comp in data1.get(category, {}).items():
                    if comp in comp_names and term not in run_terms:
                        run_terms.add(term)
                        term_counts[term] += 1
                        term_data[term] = (typ, comp)
            print(f"    Voting run {run_idx+1}: {len(run_terms)} terms extracted")

        # Filter: keep only terms in >= 2/3 runs
        all_mappings = {}
        for term, count in term_counts.items():
            if count >= 2:
                all_mappings[term] = term_data[term]
            else:
                print(f"    Voting filtered out (1/{n_runs}): {term}")

        print(f"    After voting: {len(all_mappings)} mappings (from {len(term_counts)} unique)")

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

            # Fix A: CamelCase override
            for term in list(generic_terms):
                if re.search(r'[a-z][A-Z]', term):
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    CamelCase override (rescued): {term}")

            # Fix B: Uppercase override (original, not restricted)
            for term in list(generic_terms):
                if term.isupper() and len(term) <= 4 and term in all_mappings:
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    Uppercase override (rescued): {term}")

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

            # Fix C: Capitalized synonym-to-component override
            for term in list(generic_terms):
                if term not in all_mappings:
                    continue
                _, target_comp = all_mappings[term]
                if ' ' in term or not term[0].isupper() or target_comp not in comp_names:
                    continue
                for s in sentences[:100]:
                    for m in re.finditer(rf'\b{re.escape(term)}\b', s.text):
                        if m.start() > 0:
                            generic_terms.discard(term)
                            approved.add(term)
                            print(f"    Capitalized-synonym override (rescued): {term}")
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
