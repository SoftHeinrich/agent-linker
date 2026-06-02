"""S-Linker13 Trim5 Doc-Knowledge-Extraction Runtime Clean — Phase 12 EXTENSION (Plan 12-08).

REMOVED_FROM: prompts_v2.DOC_KNOWLEDGE_EXTRACTION_RULES (the static rules
              telling the alias extractor what to look for).
REPLACED_BY: inference-time rubric builder. A small rubric-builder LLM call
             receives a GENERIC SE-textbook seed example, the project
             document, and the component list; it emits 4-6 extraction
             criteria tailored to the document.
PRESERVED: extraction prompt structure (component list, ALIAS_SCOPE_SCHEMA,
           document text, JSON template). The judge step is left unchanged.

NO STATIC FALLBACK: RuntimeError on empty rubric.

GATE-06: compiler-style seed example, consistent with sibling variants.

Variant priority: FIFTH — Tier 1 alias extraction (proposer side, recall-
sensitive). Higher risk because over-conservative criteria collapse alias
recall and cascade into entity extraction + coref via the alias dictionary.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker13_clean import (
    SLinker13Clean,
    AliasEntry,
    ALIAS_SCOPE_SCHEMA,
)
from llm_sad_sam.core.data_types_v2 import DocumentKnowledge
from llm_sad_sam.linkers.experimental.prompts_v3 import (
    DOC_KNOWLEDGE_JUDGE_EXAMPLES,
    DOC_KNOWLEDGE_JUDGE_RULES,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

EXTRACTION_RUBRIC_BUILDER_SEED_EXAMPLE = (
    "EXAMPLE (a generic compiler-style system, for reference shape only — NOT "
    "the project you will analyze):\n"
    "Components: Lexer, Parser, CodeGenerator, SymbolTable, Optimizer.\n"
    "A good 5-item extraction rubric for this kind of document would be:\n"
    "  - Capture abbreviations explicitly introduced in parenthesis, e.g.\n"
    "    \"Abstract Syntax Tree (AST)\" introduces AST as an abbreviation\n"
    "  - Capture trailing words of multi-word component names when used alone\n"
    "    to refer back to the full name (\"the Generator\" -> CodeGenerator)\n"
    "  - Capture role titles or technical aliases used interchangeably with\n"
    "    the component name in the same paragraph\n"
    "  - Reject generic descriptions that could apply to anything (\"the system\",\n"
    "    \"the process\") or plain English uses of ordinary words\n"
    "  - When in doubt, FAVOR EXTRACTION — the downstream judge will filter\n"
    "    invalid aliases\n"
    "The rubric above is illustrative; build YOUR rubric from the project document below."
)


EXTRACTION_RUBRIC_BUILDER_PROMPT = """You are building a 4-6 item rubric for discovering alternative names (abbreviations and synonyms) used for the architecture components in this specific document.

{seed_example}

PROJECT DOCUMENT:
{document_text}

PROJECT COMPONENTS:
{component_list}

Produce a 4-6 item rubric grounded in patterns the document actually uses. Cover both criteria for WHAT TO CAPTURE (parenthetical abbreviations, trailing-word synonyms, role-title aliases) and WHAT TO REJECT (generic English, system-wide references). Do NOT pre-extract any specific alias — the rubric is the criteria, not the verdicts.

Return JSON:
{{"rubric": ["item 1", "item 2", "item 3", "item 4", "item 5"]}}
JSON only:"""


# ---------------------------------------------------------------------------
# Variant class
# ---------------------------------------------------------------------------


class SLinker13Trim5ExtractionRuntimeClean(SLinker13Clean):
    """Phase 12 EXTENSION (Plan 12-08): runtime-built extraction rubric."""

    _VARIANT_NAME = "s_linker13_trim5_extraction_runtime_clean"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trim5_rubric_calls = 0

    def _build_extraction_rubric(self, components, sentences):
        """Build the extraction rubric. Raise on empty."""
        comp_names = [c.name for c in components]
        component_list = "\n".join(f"- {n}" for n in comp_names)
        doc_lines = [s.text for s in sentences]

        prompt = EXTRACTION_RUBRIC_BUILDER_PROMPT.format(
            seed_example=EXTRACTION_RUBRIC_BUILDER_SEED_EXAMPLE,
            document_text=chr(10).join(doc_lines),
            component_list=component_list,
        )

        rubric_data = None
        for attempt in range(2):
            rubric_data = self.llm.extract_json(self.llm.query(prompt, timeout=180))
            if rubric_data and rubric_data.get("rubric"):
                break
            if attempt == 0:
                print("    [trim5] Extraction rubric builder: empty response, retrying...")

        if not (rubric_data and isinstance(rubric_data.get("rubric"), list)
                and rubric_data["rubric"]):
            raise RuntimeError(
                "trim5: extraction rubric builder returned empty after 2 attempts; "
                "no static fallback by design (Plan 12-08 user directive)."
            )
        items = [str(r).strip() for r in rubric_data["rubric"] if str(r).strip()]
        if not items:
            raise RuntimeError(
                "trim5: extraction rubric builder returned an empty list; "
                "no static fallback by design."
            )

        self._trim5_rubric_calls += 1
        rubric = (
            "WHAT TO FIND (generated for this document):\n"
            + "\n".join(f"- {r}" for r in items)
        )
        print(f"[trim5 rubric, call {self._trim5_rubric_calls}]")
        print(rubric)
        return rubric

    def _learn_document_knowledge_enriched(self, sentences, components):
        """Extraction with runtime rubric; judge step unchanged."""
        comp_names = [c.name for c in components]
        doc_lines = [s.text for s in sentences]

        # NEW: build extraction rubric at runtime
        extraction_rubric = self._build_extraction_rubric(components, sentences)

        prompt1 = f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

{extraction_rubric}

{ALIAS_SCOPE_SCHEMA}

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": [{{"term": "short_form", "component": "FullComponent", "scope": "global"}}],
  "synonyms":      [{{"term": "specific_alternative_name", "component": "FullComponent", "scope": "local"}}]
}}
JSON only:"""

        data1 = None
        for attempt in range(2):
            data1 = self.llm.extract_json(self.llm.query(prompt1, timeout=300))
            if data1:
                break
            if attempt == 0:
                print("    Doc knowledge: empty response, retrying...")

        all_mappings: dict = {}
        all_scopes: dict = {}
        if data1:
            abbr_recs = data1.get("abbreviations", [])
            syn_recs = data1.get("synonyms", [])
            if isinstance(abbr_recs, dict):
                abbr_recs = [{"term": k, "component": v, "scope": "local"}
                             for k, v in abbr_recs.items()]
            if isinstance(syn_recs, dict):
                syn_recs = [{"term": k, "component": v, "scope": "local"}
                            for k, v in syn_recs.items()]
            for rec in abbr_recs:
                if not isinstance(rec, dict):
                    continue
                term = rec.get("term")
                full = rec.get("component")
                scope = rec.get("scope", "local")
                if term and full in comp_names:
                    all_mappings[term] = full
                    all_scopes[term] = scope
            for rec in syn_recs:
                if not isinstance(rec, dict):
                    continue
                term = rec.get("term")
                full = rec.get("component")
                scope = rec.get("scope", "local")
                if term and full in comp_names:
                    all_mappings[term] = full
                    all_scopes[term] = scope

        knowledge = DocumentKnowledge()
        if not all_mappings:
            return knowledge

        mapping_list = [
            f"'{k}' -> {v}" for k, v in list(all_mappings.items())[:25]
        ]

        prompt2 = f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

{DOC_KNOWLEDGE_JUDGE_EXAMPLES}

{DOC_KNOWLEDGE_JUDGE_RULES}

Return JSON:
{{
  "approved": ["term1", "term2"]
}}
JSON only:"""

        data2 = None
        for attempt in range(2):
            data2 = self.llm.extract_json(self.llm.query(prompt2, timeout=120))
            if data2 and data2.get("approved"):
                break
            if attempt == 0:
                print("    Doc knowledge judge: empty response, retrying...")
        approved = set(data2.get("approved", [])) if data2 else set(all_mappings.keys())

        for term, comp in all_mappings.items():
            if term in approved:
                scope = all_scopes.get(term, "local")
                if scope not in ("global", "local"):
                    scope = "local"
                knowledge.aliases[term] = AliasEntry(component=comp, scope=scope)
                print(f"    Alias: {term} -> {comp} [{scope}]")

        return knowledge
