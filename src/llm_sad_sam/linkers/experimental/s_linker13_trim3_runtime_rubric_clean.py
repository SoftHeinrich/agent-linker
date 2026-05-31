"""S-Linker13 Trim3 Runtime Rubric Clean — Phase 12 Step 3 (Plan 12-05).

REMOVED_FROM: prompts_v2.DOC_KNOWLEDGE_JUDGE_RULES (static rubric)
REPLACED_BY: inference-time rubric builder (AHE + Agentic Rubrics mechanism,
             per .planning/research/PROMPT-HARNESS-SURVEY-SUPPLEMENT-ERDOS.md
             Techniques 2 + 3 — sourced from arXiv 2604.25850 + 2601.04171)
PRESERVED: DOC_KNOWLEDGE_JUDGE_EXAMPLES (7 worked examples — V35a guard;
           static example removal regressed Claude in V35a)

Highest-risk trim in Phase 12: introduces a NEW LLM call to the layer1 budget
(the rubric builder) and depends on the rubric builder NOT introducing
benchmark-derived phrasing into the generated rubric body.

Mechanism (per supplement §3 cross-cutting theme 2 — "generate the rubric,
don't write it"):

  1. Step 1 — Extraction (unchanged from parent): the LLM proposes candidate
     alias mappings from the document.
  2. Step 2 — Rubric builder (NEW): a small LLM call receives a generic SE-textbook
     seed example, the project document, and the candidate mappings; it emits a
     4-6 item rubric tailored to the current document.
  3. Step 3 — Judge (modified): the generated rubric replaces the static
     DOC_KNOWLEDGE_JUDGE_RULES in the judge prompt; the 7 worked examples are
     preserved verbatim.

Fallback path: if the rubric builder returns empty after 2 attempts, the variant
degrades to the static parent DOC_KNOWLEDGE_JUDGE_RULES so downstream phases
never see empty doc knowledge. Fallback occurrences are counted in
``self._trim3_fallback_count`` for the SUMMARY's risk_notes section.

GATE-06: both RUBRIC_BUILDER_SEED_EXAMPLE and RUBRIC_BUILDER_PROMPT are
benchmark-clean (compiler-style seed example, abstract placeholder content in
the JSON template). The generated-rubric audit in Plan 12-05 Task 4 re-runs the
full BENCHMARK_TABOO sweep against every rubric body actually emitted across
the 10 ablation runs.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental import prompts_v2
from llm_sad_sam.linkers.experimental.s_linker13_clean import (
    SLinker13Clean,
    AliasEntry,
    ALIAS_SCOPE_SCHEMA,
)
from llm_sad_sam.core.data_types_v2 import DocumentKnowledge
from llm_sad_sam.linkers.experimental.prompts_v3 import (
    DOC_KNOWLEDGE_EXTRACTION_RULES,
)


# ---------------------------------------------------------------------------
# Module-level constants — exported for GATE-06 audit + test pinning
# ---------------------------------------------------------------------------

#: Generic SE-textbook seed example for the rubric builder. The compiler-style
#: domain (Lexer / Parser / CodeGenerator / SymbolTable / Optimizer) is
#: benchmark-clean per BENCHMARK_TABOO.md "Safe SE Textbook Examples".
RUBRIC_BUILDER_SEED_EXAMPLE = (
    "EXAMPLE (a generic compiler-style system, for reference shape only — NOT "
    "the project you will analyze):\n"
    "Components: Lexer, Parser, CodeGenerator, SymbolTable, Optimizer.\n"
    "Candidate mappings to judge:\n"
    "  \"AST\" -> AbstractSyntaxTree (abbrev)\n"
    "  \"Table\" -> SymbolTable (synonym)\n"
    "  \"the generator\" -> CodeGenerator (synonym)\n"
    "A good 5-item rubric for this example would be:\n"
    "  - Approve abbreviations whose letters appear in the component name\n"
    "  - Approve trailing words of multi-word component names when unambiguous\n"
    "  - Approve descriptive phrases that consistently refer to one component\n"
    "  - Reject ordinary words used in their dictionary sense, not as a name\n"
    "  - When uncertain, prefer approve — downstream filters catch false approvals\n"
    "The rubric above is illustrative; build YOUR rubric from the project document below."
)


#: Rubric-builder prompt template. Three placeholders ({seed_example},
#: {document_text}, {candidate_mappings}) are filled per-document at inference.
#: The JSON-template list uses abstract "item N" placeholders rather than real
#: rubric content (V35c lesson: concrete output examples bias model output).
RUBRIC_BUILDER_PROMPT = """You are building a 4-6 item rubric for judging whether candidate alias mappings are valid for the architecture components in this specific document.

{seed_example}

PROJECT DOCUMENT:
{document_text}

CANDIDATE MAPPINGS (to be judged later, not now):
{candidate_mappings}

Produce a 4-6 item rubric grounded in patterns the document actually uses. Cover both criteria for when an alias clearly refers to a component and criteria for when a term is too generic. Do NOT pre-decide any mapping — the rubric is the decision criteria, not the decisions themselves.

Return JSON:
{{"rubric": ["item 1", "item 2", "item 3", "item 4", "item 5"]}}
JSON only:"""


#: Worked examples kept byte-equal to prompts_v2 — V35a guard.
DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3 = prompts_v2.DOC_KNOWLEDGE_JUDGE_EXAMPLES


# ---------------------------------------------------------------------------
# Variant class — subclass that overrides only _learn_document_knowledge_enriched
# ---------------------------------------------------------------------------


class SLinker13Trim3RuntimeRubricClean(SLinker13Clean):
    """Phase 12 Step 3 trim: inference-time rubric replaces DOC_KNOWLEDGE_JUDGE_RULES.

    Inherits everything else from SLinker13Clean. The override forks
    ``_learn_document_knowledge_enriched`` to insert the rubric-builder call
    between extraction and judge.
    """

    _VARIANT_NAME = "s_linker13_trim3_runtime_rubric_clean"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track how often the rubric builder failed and we fell back to the
        # static parent rubric — surfaced in verdict.risk_notes.
        self._trim3_fallback_count = 0

    def _learn_document_knowledge_enriched(self, sentences, components):
        """Discover aliases via LLM extraction + RUNTIME-RUBRIC-BUILDER + judge.

        Three-step pipeline (Step 2 is new vs the parent):
          1. Extraction prompt → candidate alias mappings.
          2. Rubric-builder prompt → 4-6 item document-grounded rubric.
          3. Judge prompt → embeds the generated rubric (NOT the static
             DOC_KNOWLEDGE_JUDGE_RULES) + the 7 preserved worked examples.
        """
        comp_names = [c.name for c in components]
        doc_lines = [s.text for s in sentences]

        # ── Step 1 — Extraction (same as parent) ──
        prompt1 = f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

{DOC_KNOWLEDGE_EXTRACTION_RULES}

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

        all_mappings: dict[str, str] = {}
        all_scopes: dict[str, str] = {}
        if data1:
            abbr_recs = data1.get("abbreviations", [])
            syn_recs = data1.get("synonyms", [])
            if isinstance(abbr_recs, dict):
                abbr_recs = [
                    {"term": k, "component": v, "scope": "local"}
                    for k, v in abbr_recs.items()
                ]
            if isinstance(syn_recs, dict):
                syn_recs = [
                    {"term": k, "component": v, "scope": "local"}
                    for k, v in syn_recs.items()
                ]
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

        # ── Step 2 — NEW: rubric builder ──
        rubric_prompt = RUBRIC_BUILDER_PROMPT.format(
            seed_example=RUBRIC_BUILDER_SEED_EXAMPLE,
            document_text=chr(10).join(doc_lines),
            candidate_mappings=chr(10).join(mapping_list),
        )
        rubric_data = None
        for attempt in range(2):
            rubric_data = self.llm.extract_json(
                self.llm.query(rubric_prompt, timeout=180)
            )
            if rubric_data and rubric_data.get("rubric"):
                break
            if attempt == 0:
                print("    Rubric builder: empty response, retrying...")

        fallback_used = False
        if rubric_data and isinstance(rubric_data.get("rubric"), list) \
                and rubric_data["rubric"]:
            rubric_items = [
                str(r).strip() for r in rubric_data["rubric"] if str(r).strip()
            ]
            if rubric_items:
                generated_rubric = (
                    "DECISION RUBRIC (generated for this document):\n"
                    + "\n".join(f"- {r}" for r in rubric_items)
                )
            else:
                generated_rubric = prompts_v2.DOC_KNOWLEDGE_JUDGE_RULES
                fallback_used = True
        else:
            generated_rubric = prompts_v2.DOC_KNOWLEDGE_JUDGE_RULES
            fallback_used = True

        if fallback_used:
            self._trim3_fallback_count += 1
            print(
                "    Rubric builder fell back to static parent rubric "
                f"(fallback_count={self._trim3_fallback_count})"
            )
        else:
            # Audit log line — Task 4 grep-parses sweep.log for this prefix.
            print(generated_rubric)

        # ── Step 3 — Judge (modified: generated rubric replaces static rules) ──
        prompt2 = f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

{DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3}

{generated_rubric}

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
