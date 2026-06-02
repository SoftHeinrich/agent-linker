"""S-Linker13 Trim9 Seed-Disambiguation Runtime Clean — Phase 12 EXTENSION (Plan 12-12).

REMOVED_FROM: s_linker13_clean.SLinker13Clean.SEED_DISAMBIGUATION_RULES (the
              static class-attribute rubric used inside _run_seed_validation).
REPLACED_BY: inference-time rubric builder (AHE + Agentic Rubrics, per
             .planning/research/PROMPT-HARNESS-SURVEY-SUPPLEMENT-ERDOS.md
             Techniques 2 + 3). A small rubric-builder LLM call receives a
             GENERIC SE-textbook seed example plus the project document and
             component list; it emits a 4-6 item rubric that replaces the
             static SEED_DISAMBIGUATION_RULES in the per-component
             seed-disambiguation prompts.

PRESERVED: per-component seed-disambiguation structure (component dossier,
           anchor sentences, mention-context classification). The override
           only swaps the rubric body in the prompt.

NO STATIC FALLBACK: if the rubric builder returns empty after 2 attempts, the
variant RAISES. (User directive for Phase 12 EXTENSION: clean attribution.)

GATE-06: VALIDATION_RUBRIC_BUILDER_SEED_EXAMPLE uses a compiler-style domain,
consistent with trim3/trim8. Cross-dataset isolation is the testable criterion
(operationalized in Plan 12-05-REVISIT).

Variant priority: SECOND in EXTENSION batch — Tier 2 judge surface, single
prompt scope, lowest risk among the six runtime variants.

Rubric is built ONCE per document (not per component) and reused across all
component dossiers; this caps the runtime cost at +1 LLM call per dataset.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker13_clean import (
    SLinker13Clean,
)
from llm_sad_sam.linkers.experimental.helper_v3 import (
    has_standalone_mention,
    build_component_profile,
    get_comp_names,
)
from llm_sad_sam.core.data_types_v2 import SadSamLink


# ---------------------------------------------------------------------------
# Module-level constants — exported for GATE-06 audit + test pinning
# ---------------------------------------------------------------------------

SEED_RUBRIC_BUILDER_SEED_EXAMPLE = (
    "EXAMPLE (a generic compiler-style system, for reference shape only — NOT "
    "the project you will analyze):\n"
    "Components: Lexer, Parser, CodeGenerator, SymbolTable, Optimizer.\n"
    "Sample seed cases to disambiguate:\n"
    "  Case A: \"The Parser builds an AST from the lexer's tokens.\" -> Parser\n"
    "  Case B: \"This module uses parser-combinator algorithms.\" -> Parser\n"
    "A good 5-item disambiguation rubric for this example would be:\n"
    "  - COMPONENT when the name is the grammatical subject or object of an\n"
    "    architectural action (builds, consumes, dispatches, stores)\n"
    "  - COMPONENT when the name appears in a list of architectural components\n"
    "    or is named as a participant\n"
    "  - OTHER when the sentence describes an algorithm or technique that\n"
    "    shares the name but is not the component itself\n"
    "  - OTHER when the name appears only inside a dotted path or qualified\n"
    "    identifier (e.g. compiler.parser.ASTBuilder)\n"
    "  - When uncertain, choose COMPONENT — these candidates passed independent\n"
    "    seed extraction and carry prior evidence\n"
    "The rubric above is illustrative; build YOUR rubric from the project document below."
)


SEED_RUBRIC_BUILDER_PROMPT = """You are building a 4-6 item rubric for distinguishing true component references from look-alike usages (code paths, algorithms, generic English) in a software architecture document.

{seed_example}

PROJECT DOCUMENT:
{document_text}

PROJECT COMPONENTS:
{component_list}

Produce a 4-6 item rubric grounded in patterns the document actually uses. Cover both criteria for COMPONENT (when the name refers to the architectural component) and OTHER (when it does not). Do NOT pre-decide any case — the rubric is the decision criteria, not the decisions themselves.

Return JSON:
{{"rubric": ["item 1", "item 2", "item 3", "item 4", "item 5"]}}
JSON only:"""


# ---------------------------------------------------------------------------
# Variant class
# ---------------------------------------------------------------------------


class SLinker13Trim9SeedRuntimeClean(SLinker13Clean):
    """Phase 12 EXTENSION (Plan 12-12): runtime-built seed-disambiguation rubric.

    Override surface:
      - ``_run_seed_validation`` — inserts a rubric-builder LLM call before
        the per-component disambiguation loop. The generated rubric replaces
        the static ``SEED_DISAMBIGUATION_RULES`` constant in every dossier
        prompt.

    Fails loudly (RuntimeError) if the rubric builder returns empty after 2
    attempts. NO STATIC FALLBACK — clean attribution per user directive.
    """

    _VARIANT_NAME = "s_linker13_trim9_seed_runtime_clean"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trim9_rubric_calls = 0

    def _build_seed_rubric(self, components, sent_map):
        """Build a per-document seed-disambiguation rubric. Raise on empty."""
        comp_names = get_comp_names(components)
        component_list = "\n".join(f"- {n}" for n in comp_names)
        doc_lines = [s.text for s in sorted(sent_map.values(),
                                            key=lambda x: x.number)]

        prompt = SEED_RUBRIC_BUILDER_PROMPT.format(
            seed_example=SEED_RUBRIC_BUILDER_SEED_EXAMPLE,
            document_text=chr(10).join(doc_lines),
            component_list=component_list,
        )

        rubric_data = None
        for attempt in range(2):
            rubric_data = self.llm.extract_json(
                self.llm.query(prompt, timeout=180)
            )
            if rubric_data and rubric_data.get("rubric"):
                break
            if attempt == 0:
                print("    [trim9] Seed rubric builder: empty response, retrying...")

        if not (rubric_data and isinstance(rubric_data.get("rubric"), list)
                and rubric_data["rubric"]):
            raise RuntimeError(
                "trim9: seed rubric builder returned empty after 2 attempts; "
                "no static fallback by design (Plan 12-12 user directive)."
            )
        items = [str(r).strip() for r in rubric_data["rubric"] if str(r).strip()]
        if not items:
            raise RuntimeError(
                "trim9: seed rubric builder returned an empty list; "
                "no static fallback by design."
            )

        self._trim9_rubric_calls += 1
        rubric = (
            "DECISION RUBRIC (generated for this document):\n"
            + "\n".join(f"- {r}" for r in items)
        )
        print(f"[trim9 rubric, call {self._trim9_rubric_calls}]")
        print(rubric)
        return rubric

    def _run_seed_validation(self, raw_seed_links, components, sent_map):
        """Knowledge-aware seed reference disambiguation with runtime rubric."""
        if not raw_seed_links:
            return []

        # Build the per-document rubric ONCE (shared across all components).
        rubric = self._build_seed_rubric(components, sent_map)

        # Group seeds by component
        by_comp: dict = {}
        for sl in raw_seed_links:
            by_comp.setdefault(sl.component_name, []).append(sl)

        verified = []

        for comp_name, seeds in sorted(by_comp.items()):
            seed_snums = {sl.sentence_number for sl in seeds}

            profile = build_component_profile(
                comp_name, self.model_knowledge, self.doc_knowledge)

            anchor_lines = []
            for s in sorted(sent_map.values(), key=lambda x: x.number):
                if s.number in seed_snums:
                    continue
                if has_standalone_mention(comp_name, s.text):
                    anchor_lines.append(f'  S{s.number}: "{s.text}"')
                    if len(anchor_lines) >= 5:
                        break

            if anchor_lines:
                anchor_section = (
                    f'KNOWN REFERENCES (these definitely refer to "{comp_name}"):\n'
                    + "\n".join(anchor_lines) + "\n\n"
                )
            else:
                anchor_section = (
                    f'NOTE: No standalone proper-case references to "{comp_name}" found '
                    f"elsewhere in the document. This component may not be discussed "
                    f"architecturally — be extra careful to verify each case.\n\n"
                )

            case_lines = []
            valid_seeds = []
            for sl in seeds:
                sent = sent_map.get(sl.sentence_number)
                if not sent:
                    continue
                valid_seeds.append(sl)
                prev = sent_map.get(sl.sentence_number - 1)
                prev_text = f' [prev: "{prev.text[:80]}"]' if prev else ""
                match_ctx = self._classify_mention(comp_name, sent.text)
                case_lines.append(
                    f'  Case {len(valid_seeds)} (S{sl.sentence_number}): '
                    f'"{sent.text}"{prev_text}\n    Mention: {match_ctx}'
                )

            if not valid_seeds:
                continue

            prompt = f"""REFERENCE DISAMBIGUATION for component "{comp_name}"

COMPONENT PROFILE:
{profile}

{anchor_section}CASES TO VERIFY:
{chr(10).join(case_lines)}

{rubric}

Return JSON:
{{"disambiguations": [{{"case": 1, "meaning": "component", "reason": "brief"}}]}}
JSON only:"""

            for attempt in range(2):
                data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
                if data and data.get("disambiguations"):
                    break
                if attempt == 0:
                    print(f"    [{comp_name}] Empty response, retrying...")
            if not data:
                verified.extend(valid_seeds)
                continue

            results = {}
            for d in data.get("disambiguations", []):
                idx = d.get("case", 0) - 1
                results[idx] = d

            approved = 0
            for i, sl in enumerate(valid_seeds):
                r = results.get(i, {})
                meaning = (r.get("meaning", "component") or "component").lower().strip()
                if meaning == "other":
                    reason = r.get("reason", "")
                    print(f"    Seed disambig reject: S{sl.sentence_number} -> "
                          f"{comp_name} ({reason})")
                else:
                    verified.append(sl)
                    approved += 1

            print(f"    [{comp_name}] {approved}/{len(valid_seeds)} seeds kept")

        return [SadSamLink(s.sentence_number, s.component_id,
                           s.component_name, source="seed")
                for s in verified]
