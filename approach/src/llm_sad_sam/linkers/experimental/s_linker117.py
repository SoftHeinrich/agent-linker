"""S-Linker117 — the lenient gate writes the verdict first, as the sortal gate does.

The other half of "can all three judges write one thing". `s_linker112` moved the
sortal gate's reply into the lenient gate's order — quote, then verdict — and level 2
refused it: terra +1.7 gold, luna −3.3, a sign flip on a gate whose whole population
is one project. That refusal leaves the question of *which* order the three should
share unanswered, because it was asked at the gate least able to answer it.

This asks it at the gate that can. The lenient gate carries 150 gold links a run
across five projects with a control spread of 2.0, so a real effect the size of
`s112`'s is visible here and noise the size of `s112`'s is not. The arm is the head's
lenient prompt with the sortal gate's field order: decide, then quote.

Two substitutions on the head's own output, each of which must fire exactly once. No
rule, clause, rubric or case changes; the reply carries the same two fields.

**The prediction, which is `s_linker48`'s and this branch's:** demanding a committed
quote is worth 35.2 TP and verifying it is worth nothing, and a JSON reply commits in
field order — so the verdict written before the quote is a verdict the quote did not
constrain. If that is right this arm loses gold at the lenient gate and the
unification must go the other way, which is what `s_linker112` tried. If it is
neutral, the three judges can share the sortal gate's order for free and `s112`'s
refusal was a statement about its gate's population and not about order.

GATE-01: `s_linker92` is untouched. GATE-06/07: nothing is worded here.
"""

from __future__ import annotations

from dataclasses import replace

from llm_sad_sam.linkers.experimental.s_linker114 import SLinker114


class SLinker117(SLinker114):
    """The head, with the lenient gate's verdict written before its quote."""

    _VARIANT_NAME = "s_linker117"

    #: The head's lenient instruction, and the same instruction with the two demands
    #: in the sortal gate's order. The quote's shape is stated identically in both.
    INSTRUCTION = ("""For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim), then decide approve true/false based on that claim.""")
    REORDERED = ("""For each case, first decide approve true/false under the rules above, then quote
the EXACT words from the sentence that state the architectural claim about the
component (or write "none" if the sentence makes no such claim).""")

    #: The reply, verdict first. The head's fields, in the sortal gate's order.
    SCHEMA = '{"case": 1, "claim": "<exact quote or none>", "approve": true}'
    SCHEMA_REORDERED = '{"case": 1, "approve": true, "claim": "<exact quote or none>"}'

    @classmethod
    def _prompt_entity_verdict_first(cls, linker, comp_names, cases) -> str:
        """The head's lenient prompt with the instruction and the schema reordered."""
        prompt = linker._prompt_validation(comp_names, cases, "", strict=False)
        for old, new in ((cls.INSTRUCTION, cls.REORDERED),
                         (cls.SCHEMA, cls.SCHEMA_REORDERED)):
            if prompt.count(old) != 1:
                raise RuntimeError(
                    f"s_linker117: {prompt.count(old)} sites for a substitution that "
                    f"must have exactly one -- the head's lenient prompt moved")
            prompt = prompt.replace(old, new, 1)
        return prompt

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker117 (lenient gate: the verdict is written before the quote)")


SLinker117.ENTITY = replace(
    SLinker114.ENTITY,
    prompt=lambda linker, names, cases, shared, ctx:
        SLinker117._prompt_entity_verdict_first(linker, names, cases),
)
