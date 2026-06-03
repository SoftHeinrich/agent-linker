"""ILinker2 + V26a: Replace TransArc baseline with ILinker2 explicit extraction.

Subclasses V26a and overrides Phase 4 (_process_transarc) to use ILinker2
instead of reading a TransArc CSV. V26a's remaining phases (coref, validation,
judge, etc.) run unchanged on top of ILinker2's high-precision explicit links.

When enable_da=True, runs Document Analysis (from CNRLinker) before Phase 3
to warm-start abbreviation/synonym discovery — same approach as CNR-DK-V26a.
"""

import re

from llm_sad_sam.core.data_types import SadSamLink, DocumentKnowledge
from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.linkers.experimental.agent_linker_v26a import AgentLinkerV26a
from llm_sad_sam.linkers.experimental.cnr_linker import CNRLinker, DocumentAnalysis
from llm_sad_sam.linkers.experimental.ilinker2 import ILinker2


class ILinker2V26a(AgentLinkerV26a):
    """V26a pipeline with ILinker2 replacing TransArc as Phase 4 seed."""

    def __init__(self, enable_da: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._ilinker2 = ILinker2(backend=self.llm.backend)
        self._enable_da = enable_da
        self._doc_analysis: DocumentAnalysis | None = None
        if enable_da:
            print("ILinker2V26a: DA warm-start enabled")

    def _process_transarc(self, transarc_csv, id_to_name, sent_map, name_to_id):
        """Override: run ILinker2 instead of loading TransArc CSV.

        Returns links with source="transarc" so downstream phases
        (TransArc immunity, deliberation judge, boundary filter bypass)
        treat them identically to real TransArc links.
        """
        # ILinker2 needs text_path and model_path — retrieve from cached state
        text_path = self._cached_text_path
        model_path = self._cached_model_path

        raw_links = self._ilinker2.link(text_path, model_path)

        # Re-label source as "transarc" so V26a's downstream logic applies
        # the same immunity/priority as real TransArc links
        result = []
        for lk in raw_links:
            # Filter out any that are in dotted paths (ILinker2 should handle
            # this already, but double-check against V26a's logic)
            sent = sent_map.get(lk.sentence_number)
            if sent and self._in_dotted_path(sent.text, lk.component_name):
                continue
            result.append(SadSamLink(
                sentence_number=lk.sentence_number,
                component_id=lk.component_id,
                component_name=lk.component_name,
                confidence=lk.confidence,
                source="transarc",
            ))
        return result

    def link(self, text_path, model_path, transarc_csv=None):
        """Cache paths for _process_transarc override, then run V26a pipeline."""
        self._cached_text_path = text_path
        self._cached_model_path = model_path

        # Run DA before V26a pipeline if enabled
        if self._enable_da:
            sentences = DocumentLoader.load_sentences(text_path)
            print(f"\n[Phase DA] Document Analysis")
            cnr_tmp = CNRLinker(backend=self.llm.backend, enable_da=False)
            self._doc_analysis = cnr_tmp._document_analysis(sentences)
            # _compute_da_stats is already called inside _document_analysis,
            # but call explicitly in case it wasn't
            if not self._doc_analysis.term_frequency:
                CNRLinker._compute_da_stats(self._doc_analysis, sentences)
            print(f"  Entities: {len(self._doc_analysis.named_entities)}")
            print(f"  Synonym groups: {len(self._doc_analysis.synonym_groups)}")
            print(f"  Abbreviations: {len(self._doc_analysis.abbreviations)}")

        return super().link(text_path, model_path, transarc_csv=transarc_csv)

    # ── Phase 3 DA warm-start ─────────────────────────────────────────

    def _learn_document_knowledge_enriched(self, sentences, components):
        """Override Phase 3 to warm-start from DA results if available."""
        da = self._doc_analysis
        if not da or not self._enable_da:
            return super()._learn_document_knowledge_enriched(sentences, components)

        comp_names = self._get_comp_names(components)
        comp_lower = {n.lower() for n in comp_names}

        # Build known mappings from DA
        known_abbrevs = {}
        known_synonyms = {}
        for abbr, full in da.abbreviations.items():
            if full in comp_names:
                known_abbrevs[abbr] = full
            elif full.lower() in comp_lower:
                for cn in comp_names:
                    if cn.lower() == full.lower():
                        known_abbrevs[abbr] = cn
                        break

        for group in da.synonym_groups:
            comp_members = []
            non_comp_members = []
            for member in group:
                if not member:
                    continue
                if member in comp_names:
                    comp_members.append(member)
                else:
                    matched = False
                    for cn in comp_names:
                        if cn.lower() == member.lower():
                            comp_members.append(cn)
                            matched = True
                            break
                    if not matched:
                        non_comp_members.append(member)
            if len(comp_members) == 1 and non_comp_members:
                for syn in non_comp_members:
                    known_synonyms[syn] = comp_members[0]

        # Build warm-start block for the Phase 3 prompt
        warm_start_lines = []
        if known_abbrevs:
            warm_start_lines.append("KNOWN ABBREVIATIONS (from document analysis):")
            for abbr, comp in known_abbrevs.items():
                warm_start_lines.append(f"  {abbr} = {comp}")
        if known_synonyms:
            warm_start_lines.append("KNOWN SYNONYMS (from document analysis):")
            for syn, comp in known_synonyms.items():
                warm_start_lines.append(f"  {syn} = {comp}")
        warm_start_block = "\n".join(warm_start_lines) if warm_start_lines else ""

        if warm_start_block:
            print(f"    DA warm-start: {len(known_abbrevs)} abbrevs, {len(known_synonyms)} synonyms")

        # Run Phase 3 with warm-start injected into prompt
        doc_lines = [f"S{s.number}: {s.text}" for s in sentences[:100]]

        warm_inject = ""
        if warm_start_block:
            warm_inject = f"""
{warm_start_block}

These were discovered by document analysis. Include them if they are valid, and find any
additional mappings not listed above.
"""

        prompt1 = f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}
{warm_inject}
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
        # Pre-seed DA abbreviations (definitional, safe to trust)
        for abbr, comp in known_abbrevs.items():
            all_mappings[abbr] = ("abbrev", comp)
        # DA synonyms are NOT pre-seeded — they're too noisy (e.g. platform
        # names conflated with component names). They're injected into the
        # prompt as hints only; the LLM must independently confirm them.

        if data1:
            for short, full in data1.get("abbreviations", {}).items():
                if full in comp_names and short not in all_mappings:
                    all_mappings[short] = ("abbrev", full)
            for syn, full in data1.get("synonyms", {}).items():
                if full in comp_names and syn not in all_mappings:
                    all_mappings[syn] = ("synonym", full)
            for partial, full in data1.get("partial_references", {}).items():
                if full in comp_names and partial not in all_mappings:
                    all_mappings[partial] = ("partial", full)

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

            # Deterministic overrides (same as V26a/CNR-V26a)
            for term in list(generic_terms):
                if not term:
                    generic_terms.discard(term)
                    continue
                if re.search(r'[a-z][A-Z]', term):
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    CamelCase override (rescued): {term}")
            for term in list(generic_terms):
                if not term:
                    generic_terms.discard(term)
                    continue
                if term.isupper() and len(term) <= 4 and term in all_mappings:
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    Uppercase override (rescued): {term}")
            for term in list(generic_terms):
                if not term or term not in all_mappings:
                    continue
                if any(term.lower() == cn.lower() for cn in comp_names):
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    Exact-component override (rescued): {term}")
                elif term[0].isupper() and ' ' in term:
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    Multi-word proper-name override (rescued): {term}")
            for term in list(generic_terms):
                if not term or term not in all_mappings:
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
