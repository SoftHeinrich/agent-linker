"""S-Linker13 Trim4 Ambiguity Runtime Clean — Phase 12 EXTENSION (Plan 12-07).

REMOVED_FROM: prompts_v2.AMBIGUITY_FEW_SHOT (4 calibration examples) +
              prompts_v2.AMBIGUITY_RULES (architectural-vs-ambiguous rule set).
REPLACED_BY: inference-time rubric builder (AHE + Agentic Rubrics). A small
             rubric-builder LLM call receives a GENERIC SE-textbook seed
             example and the actual component name list; it emits 3-6
             calibration examples + criteria tailored to the actual component
             vocabulary about to be classified.
PRESERVED: post-classification structural guard (single-word filter on
           ambiguous names).

NO STATIC FALLBACK: RuntimeError on empty rubric.

GATE-06: AMBIGUITY_RUBRIC_BUILDER_SEED_EXAMPLE uses compiler-style + OS-style
domains (consistent with the original AMBIGUITY_FEW_SHOT, which already used
safe textbook domains). Cross-dataset isolation tests the actual rubric body.

Variant priority: FOURTH in EXTENSION batch — Tier 1 ambiguity classifier.
Higher risk than trim8/trim9/trim6 because the classification feeds the
downstream generic-word filter (Tier 2) and a regression here cascades.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker13_clean import (
    SLinker13Clean,
)
from llm_sad_sam.core.data_types_v2 import ModelKnowledge


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

AMBIGUITY_RUBRIC_BUILDER_SEED_EXAMPLE = (
    "EXAMPLE (a generic compiler-style system, for reference shape only — NOT "
    "the project you will analyze):\n"
    "Component names: Lexer, Parser, CodeGenerator, Optimizer, Core, Util, AST, SymbolTable, Base.\n"
    "Architectural (refer to a specific role): Lexer, Parser, CodeGenerator, Optimizer, AST, SymbolTable.\n"
    "Ambiguous (could be ordinary words in documentation): Core, Util, Base.\n"
    "Reasoning: Lexer / Parser / Optimizer name specific compilation roles. CamelCase compounds and common\n"
    "abbreviations (API, TCP, RPC) are always architectural. Single words like Core / Util / Base are\n"
    "organizational labels that tell you nothing about what the component does. Single words that name\n"
    "specific mechanisms (Scheduler, Dispatcher, Multiplexer) stay architectural; single words that name\n"
    "generic categories (Connector, Wrapper, Agent, Worker) often appear AMBIGUOUS.\n"
    "The rubric above is illustrative; build YOUR rubric from the project component names below."
)


AMBIGUITY_RUBRIC_BUILDER_PROMPT = """You are building a 4-6 item rubric for classifying software architecture component names as ARCHITECTURAL (specific roles or compounds) vs AMBIGUOUS (single words that writers regularly use generically in documentation).

{seed_example}

PROJECT COMPONENT NAMES:
{name_list}

Produce a 4-6 item rubric grounded in the actual names above. Cover both criteria for ARCHITECTURAL (multi-word, CamelCase compound, abbreviation, specific mechanism) and AMBIGUOUS (organizational labels, generic functional categories). Where the seed example provides illustrative criteria, refine them against the actual component vocabulary so the downstream classifier has document-grounded examples. Do NOT pre-classify any specific name — the rubric is the criteria, not the verdicts.

Return JSON:
{{"rubric": ["item 1", "item 2", "item 3", "item 4", "item 5"]}}
JSON only:"""


# ---------------------------------------------------------------------------
# Variant class
# ---------------------------------------------------------------------------


class SLinker13Trim4AmbiguityRuntimeClean(SLinker13Clean):
    """Phase 12 EXTENSION (Plan 12-07): runtime-built ambiguity rubric."""

    _VARIANT_NAME = "s_linker13_trim4_ambiguity_runtime_clean"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trim4_rubric_calls = 0

    def _build_ambiguity_rubric(self, names):
        """Build the ambiguity-classification rubric. Raise on empty."""
        name_list = ", ".join(names)
        prompt = AMBIGUITY_RUBRIC_BUILDER_PROMPT.format(
            seed_example=AMBIGUITY_RUBRIC_BUILDER_SEED_EXAMPLE,
            name_list=name_list,
        )

        rubric_data = None
        for attempt in range(2):
            rubric_data = self.llm.extract_json(self.llm.query(prompt, timeout=180))
            if rubric_data and rubric_data.get("rubric"):
                break
            if attempt == 0:
                print("    [trim4] Ambiguity rubric builder: empty response, retrying...")

        if not (rubric_data and isinstance(rubric_data.get("rubric"), list)
                and rubric_data["rubric"]):
            raise RuntimeError(
                "trim4: ambiguity rubric builder returned empty after 2 attempts; "
                "no static fallback by design (Plan 12-07 user directive)."
            )
        items = [str(r).strip() for r in rubric_data["rubric"] if str(r).strip()]
        if not items:
            raise RuntimeError(
                "trim4: ambiguity rubric builder returned an empty list; "
                "no static fallback by design."
            )

        self._trim4_rubric_calls += 1
        rubric = (
            "DECISION RUBRIC (generated for this component list):\n"
            + "\n".join(f"- {r}" for r in items)
        )
        print(f"[trim4 rubric, call {self._trim4_rubric_calls}]")
        print(rubric)
        return rubric

    def _classify_components(self, names, knowledge):
        """Classify components with a runtime-built rubric replacing the few-shot + rules."""
        rubric = self._build_ambiguity_rubric(names)

        prompt = f"""Classify these software architecture component names.

NAMES: {', '.join(names)}

{rubric}

NOW CLASSIFY THE NAMES ABOVE.

Return JSON:
{{
  "architectural": ["names that identify specific components"],
  "ambiguous": ["names that could easily be used as ordinary words in documentation"]
}}
JSON only:"""

        data = None
        for attempt in range(2):
            data = self.llm.extract_json(self.llm.query(prompt, timeout=100))
            if data:
                break
            if attempt == 0:
                print("    Ambiguity classification: empty response, retrying...")
        if data:
            valid = set(names)
            raw_ambiguous = set(data.get("ambiguous", [])) & valid
            # Preserve the parent's structural post-filter: only single-word
            # names can be ambiguous (compound/CamelCase names are always
            # architectural).
            knowledge.ambiguous_names = {
                n for n in raw_ambiguous if len(n.split()) == 1
            }
