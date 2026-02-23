"""ILinker2 V30b: ILinker2 + V26a pipeline with few-shot calibrated Phase 3 judge.

Based on V30a (zero code overrides) but adds positive few-shot examples to the
judge prompt so the LLM learns the approve/reject boundary from demonstrations
rather than abstract rules. The extraction prompt (prompt1) uses V26a's original
wording (not the over-hardened V30 version).

Key insight: V30/V30a failed because the judge had only negative examples (what
to reject) and overly strict rules. The LLM became trigger-happy and rejected
legitimate component references. Positive few-shots teach the boundary.
"""

import re

from llm_sad_sam.core.data_types import SadSamLink, DocumentKnowledge
from llm_sad_sam.linkers.experimental.agent_linker_v26a import AgentLinkerV26a
from llm_sad_sam.linkers.experimental.ilinker2 import ILinker2


class ILinker2V30b(AgentLinkerV26a):
    """V26a pipeline with ILinker2 seed + few-shot calibrated Phase 3 judge."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ilinker2 = ILinker2(backend=self.llm.backend)

    def link(self, text_path, model_path, transarc_csv=None):
        self._cached_text_path = text_path
        self._cached_model_path = model_path
        return super().link(text_path, model_path, transarc_csv=transarc_csv)

    def _process_transarc(self, transarc_csv, id_to_name, sent_map, name_to_id):
        """Override: run ILinker2 instead of loading TransArc CSV."""
        raw_links = self._ilinker2.link(self._cached_text_path, self._cached_model_path)

        result = []
        for lk in raw_links:
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

    def _learn_document_knowledge_enriched(self, sentences, components):
        """Phase 3: Few-shot calibrated judge — no code-level overrides."""
        comp_names = [c.name for c in components]
        doc_lines = [s.text for s in sentences[:150]]

        # Extraction prompt — use V26a's original wording (not hardened V30)
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

        if all_mappings:
            mapping_list = [f"'{k}' -> {v[1]} ({v[0]})" for k, v in list(all_mappings.items())[:25]]

            prompt2 = f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

EXAMPLES — study these to calibrate your judgment:

Example 1 — APPROVE (proper name in context):
  'Scheduler' -> TaskScheduler (partial)
  Document says: "The Scheduler assigns threads to available cores."
  Verdict: APPROVE. "Scheduler" is capitalized mid-sentence and used as a proper
  name for the TaskScheduler component, not as a generic concept.

Example 2 — APPROVE (CamelCase identifier):
  'RenderEngine' -> GameRenderEngine (synonym)
  Document says: "The RenderEngine processes draw calls each frame."
  Verdict: APPROVE. CamelCase is a constructed identifier — it is a proper name,
  not a generic English word.

Example 3 — APPROVE (abbreviation with document evidence):
  'AST' -> AbstractSyntaxTree (abbrev)
  Document says: "The Abstract Syntax Tree (AST) represents the parsed program."
  Verdict: APPROVE. Explicitly defined in the document with parenthetical pattern.

Example 4 — REJECT (generic concept, not a component name):
  'process' -> OrderProcessor (partial)
  Document says: "The system will process incoming requests."
  Verdict: REJECT. "process" is used as a verb in its ordinary English sense,
  not as a name for OrderProcessor.

Example 5 — APPROVE (distinctive partial used as proper name):
  'Dispatcher' -> EventDispatcher (partial)
  Document says: "When an event arrives, the Dispatcher routes it to handlers."
  Verdict: APPROVE. "Dispatcher" is capitalized mid-sentence and refers
  specifically to EventDispatcher — it is a distinctive term in this document.

Example 6 — REJECT (ambiguous, refers to the whole system):
  'system' -> PaymentSystem (partial)
  Document says: "The system handles all transactions."
  Verdict: REJECT. "system" refers to the overall system, not specifically
  to PaymentSystem.

NOW JUDGE THE PROPOSED MAPPINGS. Apply these rules:

REJECT if ANY of these are true:
- The term is used in its ordinary English sense, NOT as a name for the component
- The term refers to a different component or to the system as a whole
- The mapping cannot be verified from the actual document text

APPROVE if ANY of these are true:
- The term is a CamelCase identifier (mixed lower-then-upper like "PaymentGateway")
  — CamelCase terms are constructed proper names, not generic English
- The term appears capitalized mid-sentence in the document — this signals
  proper name usage (e.g., "The Optimizer runs" uses Optimizer as a name)
- The term is used AS A NAME for the component in context, refers to exactly
  one component, and this can be verified from the document

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

        # Deterministic CamelCase-split synonym injection
        for comp in [c.name for c in components]:
            split = re.sub(r'([a-z])([A-Z])', r'\1 \2', comp)
            split = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', split)
            if split != comp and split not in knowledge.synonyms:
                knowledge.synonyms[split] = comp
                print(f"    CamelCase syn: {split} -> {comp}")

        return knowledge
