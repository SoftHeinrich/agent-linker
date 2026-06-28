"""s_linker21 — v2.6.6 CANONICAL linker (layered no-reasoning validator).

Canonical promotion of `s_linker20_union_layered` (spike-004 "v4"). This is the new
paper "Full" variant; it supersedes `s_linker13_min` as the reported canonical for the
SAD-SAM trace-link recovery study. The body below is a verbatim copy of the proven
`s_linker20_union_layered` (only the class name, `_VARIANT_NAME`, and this docstring
differ), so S21 is behaviourally byte-identical to that experimental origin.

A thin subclass of :class:`SLinker20Union` that changes ONLY the validation-gate prompt.
Everything upstream (knowledge, extraction, coref discovery) and the Phase-6 merge are
inherited unchanged, so this is byte-compatible with `s_linker20_union` except at the two
quality gates.

Provenance — spike `004-nogap-validator-ab` (2026-06-27). With extended thinking disabled
(`CLAUDE_DISABLE_THINKING=1`, reasoning effort 0) `s_linker20_union` loses ~3 macro-F1 on
Sonnet, almost entirely as false positives. The spike showed that relocating the deleted
deliberation into the prompt/output recovers the false-positive half of that loss WITHOUT
spending the implicit (`name_in_text=False`) true links:

  * Mode 5 — a forced per-candidate justification field ("claim") emitted before the
    verdict, so the reasoning is billed as output tokens instead of thinking tokens.
  * Mode 1 — an ARCHITECTURAL-CLAIM rubric ("a claim attaches to this component", not
    "the name appears"), which protects implicit references.
  * The rubric is ASYMMETRIC (the s20 entity/coref gates already are): the entity gate is
    lenient (a named mention is a link unless it is a code path / negation / different
    entity / generic technique word); the coref gate is strict (require a genuine
    referring expression + an architectural claim).

This is the "v4" config from the spike. It improves BOTH backends at no implicit-recall cost
(validator-replay, N≥1):
  * Sonnet, no thinking : 89.7 -> 90.8 (+1.1); matches thinking-on's FP profile exactly
    (entity 25 / coref 7). Does NOT close the full gap to thinking-on (92.8) — the
    remainder is upstream candidate generation, not validation.
  * gpt-5.4, no reasoning: 89.4 -> 93.8 (+4.4); every dataset up, coref FP 13 -> 2. Larger
    because gpt reasoning is net-negative, so the prompt is gpt's only lever.
Opt-in only; run with reasoning disabled (Sonnet: `CLAUDE_DISABLE_THINKING=1`;
OpenAI: leave `OPENAI_REASONING_EFFORT` unset / `none`).

canonical=True, experimental=False. All rubric text is generic English structure — no
benchmark vocabulary (BENCHMARK_TABOO clean).
"""
from llm_sad_sam.linkers.experimental.s_linker20_union import (
    SLinker20Union, COREF_VALIDATION_FOCUS)

# Entity gate — lenient: a named mention is a link unless a hard reject signal fires.
LAYERED_ENTITY_RULES = (
    "Approve the link by default: the component is named here and the document treats "
    "it as part of the system. A bare mention, a heading, or a list that includes the "
    "component name all count as valid links — approve them. Reject ONLY when one of "
    "these clearly holds: (1) the component is referred to only through a code-level or "
    "package/member path of the form x.y or x.y.z, even if that path is described as "
    "doing something; (2) the mention is negated (it is NOT a ...); (3) the matching "
    "word actually names a DIFFERENT entity; (4) the matching word is used as a generic "
    "technique or technology term, not as this system's component. When none of these "
    "reject-conditions clearly applies, approve."
)

# Coref gate — strict: the component is NOT named in the sentence, so demand a genuine
# referring expression plus an architectural claim.
LAYERED_COREF_RULES = (
    "These are coreference links: a pronoun or noun phrase in the sentence is claimed to "
    "refer back to the component, which is NOT named in the sentence itself. Approve only "
    "when the sentence contains a genuine referring expression (a pronoun or definite "
    "noun phrase) that unambiguously points to THIS component AND the sentence makes an "
    "architectural claim about it (it performs an operation, provides/consumes a service, "
    "stores or routes data, connects to another element). Reject when: the sentence is a "
    "bare continuation fragment, gerund phrase, or list item with no referring expression; "
    "the antecedent could equally be a different component; or the reference is only to a "
    "code/package path (x.y.z). When uncertain, reject."
)


class SLinker21(SLinker20Union):
    """s_linker21 — canonical s_linker20_union + layered (Mode 5 + Mode 1) validation prompt."""

    _VARIANT_NAME = "s_linker21"

    @staticmethod
    def _prompt_validation(comp_names, cases, focus) -> str:
        # Asymmetric rubric: the coref gate is detected by its focus string.
        rules = (LAYERED_COREF_RULES
                 if focus.startswith("Check coref resolution")
                 else LAYERED_ENTITY_RULES)
        # Mode 5: require the justification ("claim") before the verdict.
        return f"""Validate components in a document. {focus}

COMPONENTS: {', '.join(comp_names)}

{rules}

For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim), then decide approve true/false based on that claim.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>", "approve": true}}]}}
JSON only:"""
