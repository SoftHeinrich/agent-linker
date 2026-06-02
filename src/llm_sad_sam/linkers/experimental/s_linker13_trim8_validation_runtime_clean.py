"""S-Linker13 Trim8 Validation Runtime Clean — Phase 12 EXTENSION (Plan 12-11).

REMOVED_FROM: prompts_v2.VALIDATION_RULES (static APPROVE/REJECT criteria
              embedded in the per-pass validation prompt).
REPLACED_BY: inference-time rubric builder (AHE + Agentic Rubrics, per
             .planning/research/PROMPT-HARNESS-SURVEY-SUPPLEMENT-ERDOS.md
             Techniques 2 + 3). A small "rubric builder" call receives a
             GENERIC SE-textbook seed example, the project document and
             component list, and the candidate cases being validated; it emits
             a 4-6 item rubric tailored to the current document, which then
             replaces the static VALIDATION_RULES in BOTH validation passes
             (participation + specificity).
PRESERVED: 2-pass validation structure (participation + specificity) with
           intersection voting, evidence bundles, generic-word pre-pass.

NO STATIC FALLBACK: if the rubric builder returns empty after 2 attempts, the
variant RAISES. (User directive for the Phase 12 EXTENSION: clean attribution —
no silent degradation to the parent rubric.)

GATE-06: RUBRIC_BUILDER_SEED_EXAMPLE uses a compiler-style domain (Lexer /
Parser / CodeGenerator / SymbolTable / Optimizer), consistent with the seed
example used in s_linker13_trim3_runtime_rubric_clean and the BENCHMARK_TABOO
"Safe SE Textbook Examples" family. The RUBRIC_BUILDER_PROMPT template
contains no benchmark-derived vocabulary. Cross-dataset rubric isolation is
the testable criterion (operationalized in Plan 12-05-REVISIT) — every rubric
the builder emits at runtime is grounded in a single dataset's input
document; the builder cannot output cross-dataset-specific tokens unless they
exist as model priors from training, which is the standard runtime LLM
analysis pattern CLAUDE.md mandates.

Variant priority: HIGHEST in EXTENSION batch — Tier 2 judge surface is the
most productive trim target per Phase 11 survey §0. Differs from 12-04
(VAL-EXT MERGE) by NOT merging extraction + validation; the runtime mechanism
operates on validation alone.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker13_clean import (
    SLinker13Clean,
)
from llm_sad_sam.linkers.experimental.helper_v3 import (
    has_standalone_mention,
    get_comp_names,
)
import re


# ---------------------------------------------------------------------------
# Module-level constants — exported for GATE-06 audit + test pinning
# ---------------------------------------------------------------------------

#: Generic SE-textbook seed example for the validation rubric builder.
#: Compiler-style domain (Lexer / Parser / CodeGenerator / SymbolTable /
#: Optimizer) per BENCHMARK_TABOO.md "Safe SE Textbook Examples".
VALIDATION_RUBRIC_BUILDER_SEED_EXAMPLE = (
    "EXAMPLE (a generic compiler-style system, for reference shape only — NOT "
    "the project you will analyze):\n"
    "Components: Lexer, Parser, CodeGenerator, SymbolTable, Optimizer.\n"
    "Candidate cases to validate:\n"
    "  Case A (S12): \"The Parser consumes tokens from the Lexer.\" -> Parser\n"
    "  Case B (S15): \"This implements a parser-combinator strategy.\" -> Parser\n"
    "A good 5-item validation rubric for this example would be:\n"
    "  - APPROVE when the named component is the grammatical subject or object\n"
    "    of an architectural action (consume, emit, store, dispatch)\n"
    "  - APPROVE when a section heading names the component as its topic\n"
    "  - APPROVE when the sentence describes the component's responsibilities or\n"
    "    interactions with other named components\n"
    "  - REJECT when the name modifies a noun phrase that denotes a TECHNIQUE\n"
    "    rather than the component (e.g., \"parser-combinator\", \"observer pattern\")\n"
    "  - REJECT when the name appears inside a dotted code path or as a generic\n"
    "    English word with no architectural intent\n"
    "The rubric above is illustrative; build YOUR rubric from the project document below."
)


#: Validation rubric-builder prompt template. Three placeholders are filled
#: per-document at inference: {seed_example}, {document_text}, {component_list}.
#: The "case sample" placeholder shows the model what KIND of case it is
#: rubric-ing about, without committing to any verdict (V35c guard).
VALIDATION_RUBRIC_BUILDER_PROMPT = """You are building a 4-6 item rubric for judging whether a candidate component reference in a sentence is a true architectural reference to that specific component, or a homonymous / generic usage.

{seed_example}

PROJECT DOCUMENT:
{document_text}

PROJECT COMPONENTS:
{component_list}

CANDIDATE CASE SAMPLES (representative — to be judged later, not now):
{candidate_sample}

Produce a 4-6 item rubric grounded in patterns the document actually uses. Cover both criteria for APPROVE (when a sentence is a true architectural reference) and criteria for REJECT (when the name is used generically, as a technique, in a dotted path, or as part of a different proper name). Do NOT pre-decide any case — the rubric is the decision criteria, not the decisions themselves.

Return JSON:
{{"rubric": ["item 1", "item 2", "item 3", "item 4", "item 5"]}}
JSON only:"""


# ---------------------------------------------------------------------------
# Variant class
# ---------------------------------------------------------------------------


class SLinker13Trim8ValidationRuntimeClean(SLinker13Clean):
    """Phase 12 EXTENSION (Plan 12-11): runtime-built validation rubric.

    Override surface:
      - ``_validate_with_evidence`` — inserts a rubric-builder LLM call before
        the 2-pass validation. The generated rubric replaces the static
        ``VALIDATION_RULES`` constant in BOTH passes.

    Fails loudly (RuntimeError) if the rubric builder returns empty after 2
    attempts. NO STATIC FALLBACK — clean attribution per user directive.
    """

    _VARIANT_NAME = "s_linker13_trim8_validation_runtime_clean"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trim8_rubric_calls = 0

    # ------------------------------------------------------------------
    # Rubric builder
    # ------------------------------------------------------------------

    def _build_validation_rubric(self, components, sent_map, candidates):
        """Call the rubric builder; return a multi-line rubric string.

        Raises:
            RuntimeError: rubric builder returned empty after 2 attempts.
        """
        comp_names = get_comp_names(components)
        component_list = "\n".join(f"- {n}" for n in comp_names)

        # Document text: all sentences in order (deterministic).
        doc_lines = [s.text for s in sorted(sent_map.values(),
                                            key=lambda x: x.number)]

        # A small representative sample of candidate cases — not the full set
        # (rubric must NOT pre-decide; we cap at 8 cases for a representative
        # surface without flooding the prompt).
        sample_lines = []
        for i, c in enumerate(candidates[:8]):
            s = sent_map.get(c.sentence_number)
            if not s:
                continue
            sample_lines.append(
                f'  Case (S{c.sentence_number}): "{s.text}" -> {c.component_name}'
            )
        candidate_sample = "\n".join(sample_lines) if sample_lines else \
            "  (no candidates — use the document content alone)"

        prompt = VALIDATION_RUBRIC_BUILDER_PROMPT.format(
            seed_example=VALIDATION_RUBRIC_BUILDER_SEED_EXAMPLE,
            document_text=chr(10).join(doc_lines),
            component_list=component_list,
            candidate_sample=candidate_sample,
        )

        rubric_data = None
        for attempt in range(2):
            rubric_data = self.llm.extract_json(
                self.llm.query(prompt, timeout=180)
            )
            if rubric_data and rubric_data.get("rubric"):
                break
            if attempt == 0:
                print("    [trim8] Validation rubric builder: empty response, retrying...")

        if not (rubric_data and isinstance(rubric_data.get("rubric"), list)
                and rubric_data["rubric"]):
            raise RuntimeError(
                "trim8: validation rubric builder returned empty after 2 attempts; "
                "no static fallback by design (Plan 12-11 user directive)."
            )

        items = [str(r).strip() for r in rubric_data["rubric"] if str(r).strip()]
        if not items:
            raise RuntimeError(
                "trim8: validation rubric builder returned an empty list; "
                "no static fallback by design."
            )

        self._trim8_rubric_calls += 1
        rubric = (
            "DECISION RUBRIC (generated for this document):\n"
            + "\n".join(f"- {r}" for r in items)
        )
        # Audit log — Plan 12-EXTENSION grep-parses for this prefix.
        print(f"[trim8 rubric, call {self._trim8_rubric_calls}]")
        print(rubric)
        return rubric

    # ------------------------------------------------------------------
    # Overridden _validate_with_evidence
    # ------------------------------------------------------------------

    def _validate_with_evidence(self, candidates, bundles, components, sent_map):
        """3-step LLM validation — same as parent, but VALIDATION_RULES
        replaced by a runtime-generated rubric.
        """
        if not candidates:
            return [], {}

        comp_names = get_comp_names(components)
        decisions: dict = {}

        # ── Pre-pass: generic-word filter (identical to parent) ──
        generic_candidates: dict = {}
        non_generic: list = []
        for c in candidates:
            sent = sent_map.get(c.sentence_number)
            if not sent:
                non_generic.append(c)
                continue
            comp_lower = c.component_name.lower()
            has_exact_case = has_standalone_mention(c.component_name, sent.text)
            has_lowercase = (not has_exact_case and
                             re.search(rf'\b{re.escape(comp_lower)}\b', sent.text))
            if has_lowercase and self.model_knowledge \
                    and self.model_knowledge.ambiguous_names \
                    and c.component_name in self.model_knowledge.ambiguous_names:
                generic_candidates.setdefault(c.component_name, []).append(c)
            else:
                non_generic.append(c)

        remaining = list(non_generic)
        for comp_name, cands in generic_candidates.items():
            anchor_lines = []
            for s in sent_map.values():
                if has_standalone_mention(comp_name, s.text):
                    anchor_lines.append(f"  S{s.number}: {s.text}")
                    if len(anchor_lines) >= 5:
                        break

            case_lines = []
            for i, c in enumerate(cands):
                s = sent_map.get(c.sentence_number)
                prev = sent_map.get(c.sentence_number - 1)
                prev_text = f" [prev: {prev.text[:60]}]" if prev else ""
                case_lines.append(
                    f"  Case {i+1} (S{c.sentence_number}): {s.text}{prev_text}"
                )

            anchor_section = ""
            if anchor_lines:
                anchor_section = (
                    f'FULL-NAME REFERENCES (these definitely refer to the {comp_name} component):\n'
                    + '\n'.join(anchor_lines) + '\n\n'
                )

            prompt = f"""CONTEXTUAL WORD USAGE: Does the word refer to the architecture component "{comp_name}", or is it used as an ordinary English word?

{anchor_section}SENTENCES TO CHECK (the component name appears only in lowercase or as part of a compound phrase):
{chr(10).join(case_lines)}

For each case, determine:
- COMPONENT: The word refers to the specific "{comp_name}" component as a system entity
  (e.g., "the {comp_name.lower()} handles requests" = component reference)
- GENERIC: The word is used as ordinary English describing a general concept, activity, or modifier
  (e.g., "provides {comp_name.lower()} access" or "{comp_name.lower()} operations" = generic usage)

Key distinction: A component reference names a specific system entity as a participant.
A generic use describes a type of activity or quality that happens to share the word.

Return JSON:
{{"results": [{{"case": 1, "usage": "component" or "generic", "reason": "brief"}}]}}
JSON only:"""

            for attempt in range(2):
                data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
                if data and data.get("results"):
                    break
                if attempt == 0:
                    print(f"    Generic filter [{comp_name}]: empty response, retrying...")
            if not data:
                remaining.extend(cands)
                continue

            results_map = {}
            for r in data.get("results", []):
                idx = r.get("case", 0) - 1
                results_map[idx] = r

            for i, c in enumerate(cands):
                result = results_map.get(i, {})
                usage = (result.get("usage", "component") or "component").lower()
                key = (c.sentence_number, c.component_id)
                if usage == "generic":
                    reason = result.get("reason", "")
                    print(f"    LLM generic reject: S{c.sentence_number} -> {c.component_name} ({reason})")
                    decisions[key] = {"approved": False, "path": f"generic_filter: {reason}"}
                else:
                    remaining.append(c)

        if not remaining:
            return [], decisions

        # ── NEW: build the validation rubric at runtime ──
        rubric = self._build_validation_rubric(components, sent_map, remaining)

        # ── 2-pass validation (modified: rubric replaces static VALIDATION_RULES) ──
        print(f"    LLM 2-pass validation (+runtime rubric): {len(remaining)} candidates")
        twopass_approved = []
        for batch_start in range(0, len(remaining), 25):
            batch = remaining[batch_start:batch_start + 25]
            cases = []
            for i, c in enumerate(batch):
                prev = sent_map.get(c.sentence_number - 1)
                p = f"[prev: {prev.text[:60]}] " if prev else ""

                bundle = bundles.get((c.sentence_number, c.component_id))
                evidence_block = self._format_evidence(bundle) if bundle else ""
                case_text = (
                    f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n'
                    f'  {p}"{c.sentence_text}"\n'
                    f'{evidence_block}'
                )
                cases.append((case_text, c))

            case_strings = [ct for ct, _ in cases]

            r1 = self._run_validation_pass_with_rubric(
                comp_names, case_strings, rubric,
                "Check architectural participation: does the sentence name this component as an architectural participant — performing operations, providing services, or taking part in the described system behavior?")
            r2 = self._run_validation_pass_with_rubric(
                comp_names, case_strings, rubric,
                "Check referential specificity: is the component name used to identify this specific architectural element, or does it serve as a generic technical term in this sentence?")

            for i, (case_text, c) in enumerate(cases):
                p1 = r1.get(i, False)
                p2 = r2.get(i, False)
                approved = p1 and p2
                key = (c.sentence_number, c.component_id)
                decisions[key] = {
                    "approved": approved,
                    "p1": p1,
                    "p2": p2,
                    "path": "twopass" if approved else "twopass_reject",
                }
                if approved:
                    twopass_approved.append(c)

        return twopass_approved, decisions

    def _run_validation_pass_with_rubric(self, comp_names, cases, rubric, focus):
        """Single validation pass with runtime rubric replacing VALIDATION_RULES."""
        prompt = f"""Validate component references in a software architecture document. {focus}

COMPONENTS: {', '.join(comp_names)}

{rubric}

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true}}]}}
JSON only:"""

        for attempt in range(2):
            data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
            if data and data.get("validations"):
                break
            if attempt == 0:
                print(f"    Validation pass: empty response, retrying...")
        results = {}
        if data:
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    val = v.get("approve", False)
                    results[idx] = val is True or (isinstance(val, str) and val.lower() == "true")
        return results
