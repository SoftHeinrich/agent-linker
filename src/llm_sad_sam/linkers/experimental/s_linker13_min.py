"""S-Linker13 Min — Phase 13 Plan 13-01 composed promotion candidate.

Targets: PROMPT-03, GATE-03.

REMOVED_FROM:
  - s_linker13_clean.DOC_KNOWLEDGE_JUDGE_RULES (3-numbered-rule structure +
    IMPORTANT closer; trim1 distillation via Technique 3 + 8)
  - s_linker13_clean.SLinker13Clean.SEED_DISAMBIGUATION_RULES (static
    class-attribute rubric used inside _run_seed_validation; trim9 runtime
    builder)
  - Step 0 dead-code drop (carried passively by prompts_v3 — 7 unused
    constants from prompts_v2 not imported here)

REPLACED_BY:
  - DOC_KNOWLEDGE_JUDGE_RUBRIC_V3 (trim1, lossless prose-form rubric,
    773 → 888 bytes, "When in doubt APPROVE" emitted BEFORE the decision
    wording per Technique 8 reasoning-before-conclusion order)
  - Inference-time seed-disambiguation rubric builder (trim9, AHE + Agentic
    Rubrics per .planning/research/PROMPT-HARNESS-SURVEY-SUPPLEMENT-ERDOS.md
    Techniques 2 + 3). A small rubric-builder LLM call receives a generic
    compiler-style seed example plus the project document and component
    list; it emits a 4-6 item rubric that replaces SEED_DISAMBIGUATION_RULES
    in the per-component seed-disambiguation prompts.
  - prompts_v3 (Step 0 — Phase 12 Plan 12-01). 9 prompts byte-equal to
    prompts_v2; 7 dead constants dropped.

KEEP:
  - DOC_KNOWLEDGE_JUDGE_EXAMPLES preserved verbatim (V35a guard — example
    removal regresses Claude; trim1 carry-forward).
  - Per-component seed-disambiguation structure (component dossier, anchor
    sentences, mention-context classification) — trim9 only swaps the
    rubric body in the prompt.
  - 4 AUTO-APPROVE sub-categories (abbreviations / trailing-word /
    CamelCase / multi-word phrases) inside the trim1 distilled rubric.
  - Generic-word exclusion + whole-system rejection retained inside the
    distilled judge rubric.
  - All other Tier 1 / Tier 2 / Tier 3 pipeline phases inherit from
    SLinker13Clean unchanged.

CLEAN:
  - Standalone subclass of SLinker13Clean (per user preference for
    standalone-class-not-inheritance-chain — duplicate the trim1 +
    trim9 mechanism inline, do NOT import from trim1_judge_clean or
    trim9_seed_runtime_clean modules).
  - Both module-level prompt constants and the rubric builder method
    body are duplicated here verbatim from the source trim files.
  - Module-scope monkey-patch via try/finally keeps the trim1 override
    surgical and reviewer-defensible.
  - NO STATIC FALLBACK for trim9: variant raises RuntimeError on empty
    rubric (clean attribution per Phase 12 EXTENSION user directive).

NO STATIC FALLBACK: if the seed-rubric builder returns empty after 2
attempts, the variant RAISES. Documented in Phase 12 Plan 12-12
acceptance.

GATE-06 spot check: zero benchmark-component-name substrings in the
distilled rubric or the seed-rubric example (both audited PASS in
12-06-AUDIT-REPORT.md §4 + §5 with reviewer-defensibility narrative).

GATE-07: registered in run_ablation.py CANONICAL_VARIANTS + VARIANT_SPECS
with `canonical=True` (the promotion bit — flipped after this plan's
sweeps pass).

Phase 12 carry-forward composition rationale (12-FRONTIER-MAP-SUMMARY.md):
- trim1 hits Tier 1 alias judge (rule restructure dominates rule
  regeneration for Tier-1 judges).
- trim9 hits Tier 2 seed disambiguation (only runtime variant where
  Claude improves AND gpt-5.4 stays inside cross-model floor).
- The two trims target disjoint pipeline stages — judge vs seed — so
  interaction effects are expected small (validated empirically by
  this plan's sweep).
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker13_clean import (
    SLinker13Clean,
)
from llm_sad_sam.linkers.experimental.prompts_v2 import (
    DOC_KNOWLEDGE_JUDGE_EXAMPLES as _V2_JUDGE_EXAMPLES,
)
from llm_sad_sam.linkers.experimental.helper_v3 import (
    has_standalone_mention,
    build_component_profile,
    get_comp_names,
)
from llm_sad_sam.core.data_types_v2 import SadSamLink


# ===========================================================================
# trim1 — Distilled DOC_KNOWLEDGE_JUDGE_RULES (Technique 3 + 8)
# Constants copied verbatim from s_linker13_trim1_judge_clean.py.
# ===========================================================================

# Byte-equal alias — V35a guard: example removal regresses Claude. The 7
# worked examples are the calibration substrate the model relies on.
DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3 = _V2_JUDGE_EXAMPLES


# Technique 3 (lossless rubric distillation) + Technique 8 (reasoning-before-
# conclusion order). Single prose block. No numbered rules. The tie-breaker
# leads the decision discussion; the verdict-format directive is in the
# consumer prompt template, not here.
DOC_KNOWLEDGE_JUDGE_RUBRIC_V3 = """DECISION RUBRIC.

When in doubt, APPROVE — false approvals are filtered by later pipeline stages, while false rejections cause permanent recall loss, so the bar to reject sits above the bar to approve.

The following four shapes are always valid mappings and should be approved on sight: abbreviations formed from the component name's initials or words, trailing words of multi-word component names provided no other component shares that word, CamelCase identifiers, and multi-word phrases that contain the component name. Beyond these four shapes, approve any term that plausibly refers to exactly one component and is not a bare generic word such as "system", "process", "utility", "component", or "module". Reject only when the term is clearly generic and could refer to anything, or when it clearly refers to a different component or to the whole system rather than the proposed one."""


# ===========================================================================
# trim9 — Runtime seed-disambiguation rubric builder
# Constants copied verbatim from s_linker13_trim9_seed_runtime_clean.py.
# ===========================================================================

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


# ===========================================================================
# Variant class — composes trim1 (judge distillation) + trim9 (runtime seed)
# ===========================================================================


class SLinker13Min(SLinker13Clean):
    """Phase 13 Plan 13-01: composed promotion candidate.

    Composes:
      - Step 0 (prompts_v3 dead-code drop, carried passively at module level)
      - trim1 (DOC_KNOWLEDGE_JUDGE_RULES distilled via Technique 3 + 8;
        DOC_KNOWLEDGE_JUDGE_EXAMPLES byte-equal to v2 per V35a guard)
      - trim9 (runtime-built SEED_DISAMBIGUATION_RULES via AHE + Agentic
        Rubrics; NO STATIC FALLBACK)

    Override surface:
      - ``_learn_document_knowledge_enriched`` — try/finally monkey-patches
        ``DOC_KNOWLEDGE_JUDGE_RULES`` + ``DOC_KNOWLEDGE_JUDGE_EXAMPLES`` in
        the parent module scope for the duration of the call. Carries trim1.
      - ``_run_seed_validation`` — full body override that inserts a
        rubric-builder LLM call before the per-component disambiguation
        loop. The generated rubric replaces the static
        ``SEED_DISAMBIGUATION_RULES`` in every dossier prompt. Carries
        trim9.

    All other pipeline phases inherit from SLinker13Clean unchanged.

    Fails loudly (RuntimeError) if the seed-rubric builder returns empty
    after 2 attempts. NO STATIC FALLBACK — clean attribution per Phase 12
    EXTENSION user directive.

    Variant is NOT thread-safe vs the parent SLinker13Clean module scope —
    the trim1 override monkey-patches the parent module's name bindings via
    try/finally. The ablation harness runs variants sequentially per
    dataset, so no contention occurs in the intended use.
    """

    _VARIANT_NAME = "s_linker13_min"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._min_rubric_calls = 0

    # -------------------------------------------------------------------
    # trim1 — judge prompt override (monkey-patch parent module scope)
    # -------------------------------------------------------------------

    def _learn_document_knowledge_enriched(self, sentences, components):
        """Run parent method with judge prompt constants rebound to V3.

        The parent's ``_learn_document_knowledge_enriched`` assembles
        ``prompt2`` via an f-string that references
        ``DOC_KNOWLEDGE_JUDGE_EXAMPLES`` and ``DOC_KNOWLEDGE_JUDGE_RULES``
        at module scope. We rebind those names in the parent module for
        the duration of the call and restore them in a finally clause so
        no external state leaks across invocations.
        """
        import llm_sad_sam.linkers.experimental.s_linker13_clean as _parent_mod
        orig_rules = _parent_mod.DOC_KNOWLEDGE_JUDGE_RULES
        orig_examples = _parent_mod.DOC_KNOWLEDGE_JUDGE_EXAMPLES
        try:
            _parent_mod.DOC_KNOWLEDGE_JUDGE_RULES = DOC_KNOWLEDGE_JUDGE_RUBRIC_V3
            _parent_mod.DOC_KNOWLEDGE_JUDGE_EXAMPLES = DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3
            return super()._learn_document_knowledge_enriched(sentences, components)
        finally:
            _parent_mod.DOC_KNOWLEDGE_JUDGE_RULES = orig_rules
            _parent_mod.DOC_KNOWLEDGE_JUDGE_EXAMPLES = orig_examples

    # -------------------------------------------------------------------
    # trim9 — runtime seed rubric builder + full _run_seed_validation
    # -------------------------------------------------------------------

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
                print("    [min] Seed rubric builder: empty response, retrying...")

        if not (rubric_data and isinstance(rubric_data.get("rubric"), list)
                and rubric_data["rubric"]):
            raise RuntimeError(
                "s_linker13_min: seed rubric builder returned empty after 2 "
                "attempts; no static fallback by design (carries trim9 "
                "user directive)."
            )
        items = [str(r).strip() for r in rubric_data["rubric"] if str(r).strip()]
        if not items:
            raise RuntimeError(
                "s_linker13_min: seed rubric builder returned an empty list; "
                "no static fallback by design."
            )

        self._min_rubric_calls += 1
        rubric = (
            "DECISION RUBRIC (generated for this document):\n"
            + "\n".join(f"- {r}" for r in items)
        )
        print(f"[min rubric, call {self._min_rubric_calls}]")
        print(rubric)
        return rubric

    def _run_seed_validation(self, raw_seed_links, components, sent_map):
        """Knowledge-aware seed reference disambiguation with runtime rubric.

        Body duplicated verbatim from s_linker13_trim9_seed_runtime_clean.py
        per the user's "standalone files / duplicate code intentionally"
        preference.
        """
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
