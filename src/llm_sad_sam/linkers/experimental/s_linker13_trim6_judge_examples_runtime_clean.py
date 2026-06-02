"""S-Linker13 Trim6 Judge-Examples Runtime Clean — Phase 12 EXTENSION (Plan 12-09).

REMOVED_FROM: prompts_v2.DOC_KNOWLEDGE_JUDGE_EXAMPLES (the 7 hand-written
              worked examples used to calibrate the alias judge).
REPLACED_BY: inference-time examples generator (AHE + Agentic Rubrics, per
             .planning/research/PROMPT-HARNESS-SURVEY-SUPPLEMENT-ERDOS.md
             Techniques 2 + 3). A rubric-builder LLM call receives a generic
             SE-textbook seed example, the project document, the component
             list, and the candidate mappings; it emits 4-6 document-grounded
             worked examples (each with verdict + rationale) that replace the
             7 static examples in the judge prompt.

KEEP: trim1's distilled DOC_KNOWLEDGE_JUDGE_RUBRIC_V3 (Technique 3 + 8
      prose-form rules, ACCEPTED in Plan 12-03). This variant is orthogonal
      to trim1: trim1 distilled the RULES, trim6 generates the EXAMPLES at
      runtime. The two compose if both are accepted.

NO STATIC FALLBACK: if the examples builder returns empty after 2 attempts,
the variant RAISES.

GATE-06: EXAMPLE_BUILDER_SEED_EXAMPLE uses compiler-style domain consistent
with trim3/trim8/trim9. Cross-dataset isolation is the testable criterion.

V35a guard discussion: the V35a result said "EXAMPLE removal regresses Claude
by -2.5pp" — this variant does NOT remove examples; it REGENERATES them at
runtime. The hypothesis (per supplement §3 cross-cutting theme 2) is that
document-grounded examples preserve calibration density without leaking
training-distribution bias.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker13_clean import (
    SLinker13Clean,
    AliasEntry,
    ALIAS_SCOPE_SCHEMA,
)
from llm_sad_sam.core.data_types_v2 import DocumentKnowledge
from llm_sad_sam.linkers.experimental.prompts_v3 import (
    DOC_KNOWLEDGE_EXTRACTION_RULES,
)
# Reuse trim1's accepted distilled rubric.
from llm_sad_sam.linkers.experimental.s_linker13_trim1_judge_clean import (
    DOC_KNOWLEDGE_JUDGE_RUBRIC_V3 as _TRIM1_DISTILLED_RUBRIC,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

EXAMPLE_BUILDER_SEED_EXAMPLE = (
    "EXAMPLE (a generic compiler-style system, for reference shape only — NOT "
    "the project you will analyze):\n"
    "Components: Lexer, Parser, CodeGenerator, SymbolTable, Optimizer.\n"
    "Sample candidate mappings:\n"
    "  'AST' -> AbstractSyntaxTree (abbrev)\n"
    "  'Table' -> SymbolTable (synonym)\n"
    "  'handler' -> CodeGenerator (synonym)\n"
    "A good set of 4-6 worked examples for this system would be:\n"
    "  Example 1 — APPROVE: 'AST' -> AbstractSyntaxTree. Reason: the term is\n"
    "    the conventional initialism for the component name.\n"
    "  Example 2 — APPROVE: 'Table' -> SymbolTable. Reason: trailing word of\n"
    "    a multi-word component name when no other component shares it.\n"
    "  Example 3 — REJECT: 'handler' -> CodeGenerator. Reason: ordinary English\n"
    "    noun used generically across many contexts.\n"
    "  Example 4 — REJECT: 'compiler' -> Lexer. Reason: refers to the whole\n"
    "    system, not this specific component.\n"
    "The examples above are illustrative; build YOUR examples from the project document below."
)


EXAMPLE_BUILDER_PROMPT = """You are producing 4-6 worked examples to calibrate a downstream judge that decides whether a candidate alias mapping is valid for the architecture components in this specific document.

{seed_example}

PROJECT DOCUMENT:
{document_text}

PROJECT COMPONENTS:
{component_list}

CANDIDATE MAPPINGS (for context — NOT to be pre-decided):
{candidate_mappings}

Produce 4-6 worked examples grounded in patterns the document actually uses. Each example must include the candidate mapping, the verdict (APPROVE or REJECT), and a one-line reason. Include both APPROVE and REJECT cases. Do NOT just restate the candidate mappings as verdicts — choose examples that calibrate the BOUNDARY between valid and invalid mappings for this document.

Return JSON:
{{"examples": [
  {{"mapping": "'term' -> Component (kind)", "verdict": "APPROVE" or "REJECT", "reason": "one line"}},
  ...
]}}
JSON only:"""


# ---------------------------------------------------------------------------
# Variant class
# ---------------------------------------------------------------------------


class SLinker13Trim6JudgeExamplesRuntimeClean(SLinker13Clean):
    """Phase 12 EXTENSION (Plan 12-09): runtime-built judge examples.

    Override surface:
      - ``_learn_document_knowledge_enriched`` — adds an examples-builder call
        between extraction and judge; the generated examples REPLACE the 7
        static worked examples. The judge rubric uses trim1's distilled
        rubric (DOC_KNOWLEDGE_JUDGE_RUBRIC_V3).

    Fails loudly if the examples builder returns empty (no static fallback).
    """

    _VARIANT_NAME = "s_linker13_trim6_judge_examples_runtime_clean"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trim6_builder_calls = 0

    def _build_judge_examples(self, components, sentences, candidate_mappings):
        """Build worked examples for this document. Raise on empty."""
        comp_names = [c.name for c in components]
        component_list = "\n".join(f"- {n}" for n in comp_names)
        doc_lines = [s.text for s in sentences]

        candidate_list = "\n".join(
            f"  '{k}' -> {v}" for k, v in candidate_mappings.items()
        ) or "  (no candidates discovered)"

        prompt = EXAMPLE_BUILDER_PROMPT.format(
            seed_example=EXAMPLE_BUILDER_SEED_EXAMPLE,
            document_text=chr(10).join(doc_lines),
            component_list=component_list,
            candidate_mappings=candidate_list,
        )

        ex_data = None
        for attempt in range(2):
            ex_data = self.llm.extract_json(self.llm.query(prompt, timeout=180))
            if ex_data and ex_data.get("examples"):
                break
            if attempt == 0:
                print("    [trim6] Examples builder: empty response, retrying...")

        if not (ex_data and isinstance(ex_data.get("examples"), list)
                and ex_data["examples"]):
            raise RuntimeError(
                "trim6: examples builder returned empty after 2 attempts; "
                "no static fallback by design (Plan 12-09 user directive)."
            )

        examples_block_lines = ["EXAMPLES — study these to calibrate your judgment:"]
        n = 0
        for ex in ex_data["examples"]:
            if not isinstance(ex, dict):
                continue
            mapping = str(ex.get("mapping", "")).strip()
            verdict = str(ex.get("verdict", "")).strip().upper()
            reason = str(ex.get("reason", "")).strip()
            if not (mapping and verdict and reason):
                continue
            n += 1
            examples_block_lines.append("")
            examples_block_lines.append(
                f"Example {n} — {verdict}:")
            examples_block_lines.append(f"  {mapping}")
            examples_block_lines.append(f"  Verdict: {verdict}. {reason}")

        if n == 0:
            raise RuntimeError(
                "trim6: examples builder produced no well-formed examples; "
                "no static fallback by design."
            )

        block = "\n".join(examples_block_lines)
        self._trim6_builder_calls += 1
        print(f"[trim6 examples, call {self._trim6_builder_calls}]")
        print(block)
        return block

    def _learn_document_knowledge_enriched(self, sentences, components):
        """Extraction + runtime examples builder + judge with trim1's distilled rubric."""
        comp_names = [c.name for c in components]
        doc_lines = [s.text for s in sentences]

        # Step 1 — Extraction (same as parent)
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

        # Step 2 — NEW: build runtime examples (REPLACES the 7 static)
        examples_block = self._build_judge_examples(
            components, sentences, all_mappings)

        # Step 3 — Judge (uses trim1's distilled rubric + runtime examples)
        mapping_list = [
            f"'{k}' -> {v}" for k, v in list(all_mappings.items())[:25]
        ]

        prompt2 = f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

{examples_block}

{_TRIM1_DISTILLED_RUBRIC}

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
