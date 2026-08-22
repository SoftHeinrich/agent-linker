"""S-Linker92e — the lenient gate names the surface before it judges the claim.

`s_linker92a` hands the full-name gate 44.7 more candidates a run than the LLM
extraction pass did, and the gate handles most of them: 21.0 pairs a run whose name
is written only inside a dotted identifier are rejected 21/21 on terra and 12/19 on
luna -- and the seven luna approvals contain no gold -- so `QUALIFIED_CLAUSE` is doing
its own work and no code gate is owed for it.
What leaks is the other clause. Of the pairs the scan adds outside a qualified path,
the gate approves 44% on terra and 74% on luna at roughly one gold in three, and every
one of the recurring approvals has the same shape: a lowercased common noun that also
happens to be a component's name, or a generic term the alias stage bound to a
component. (The surfaces are named in `../results/regex_round/README.md`; naming them
here would put benchmark vocabulary in the module, which GATE-06 forbids.)

That is `STRICTER_CLAUSE`'s population exactly — *"Some sentences use an ordinary
English word that happens to coincide with a component's name ... Capitalization is
evidence for a name and its absence is evidence against"* — and the clause is already
in the prompt. **Restating it would be the wrong repair**: s86 measured a restatement
at this gate as redundant, and the typed round measured that at the lenient gate a
restatement buys nothing while at the strict gate it reinforces.

So this variant adds no rule and no gate. It changes the **order the reply is
written in**, which is the mechanism `s_linker106` uses at the resolver and `s83`
used at the coreference judge: a JSON-only reply whose first field is the answer
lets the model commit before it has looked, so the case's own surface is moved in
front of the verdict. The reply now names the expression the sentence actually
writes, then the claim, then approves. Three fields, one of which is new, no
sentence of rule text added or changed.

`_validate_with_evidence` reads `claim` and `approve` and ignores anything else, so
the added field is free to the rest of the pipeline. The strict branch of the
builder is byte-identical to `s_linker92`'s — the coreference judge is untouched.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker92 import (
    LAYERED_COREF_RULES, LAYERED_ENTITY_RULES, QUALIFIED_CLAUSE, STRICTER_CLAUSE,
)
from llm_sad_sam.linkers.experimental.s_linker92a import SLinker92a


class SLinker92e(SLinker92a):
    """The lenient gate writes down the surface it is judging before it decides."""

    _VARIANT_NAME = "s_linker92e"

    @staticmethod
    def _prompt_validation(comp_names, cases, focus, strict: bool = False) -> str:
        """`s_linker92`'s builder, with one reordered instruction on the lenient side.

        Everything the strict side renders is the head's, character for character.
        """
        rules = LAYERED_COREF_RULES if strict else LAYERED_ENTITY_RULES
        decide = (
            " then decide approve true/false based on that claim."
            if not strict else
            " then state the strongest ground there is for rejecting this case under the\n"
            "rules above (or \"none\" if there is none), then decide: approve unless that "
            "ground is one\nthe rules above make decisive. An objection you could raise "
            "against most sentences is not\na ground for rejecting this one."
        )
        field = "" if not strict else ', "objection": "<strongest ground to reject, or none>"'
        tail = "" if strict else f"\n{QUALIFIED_CLAUSE}\n{STRICTER_CLAUSE}\n"

        # The whole variant. The lenient side's per-case instruction gains one step
        # in front of the two it had, and the schema gains the field that step writes
        # to. No clause of the rubric is restated: the model is asked what the words
        # are, not what to conclude from them.
        if strict:
            head = ("For each case, first quote the EXACT words from the sentence "
                    "that state the\narchitectural claim about the component (or write "
                    '"none" if the sentence makes no\nsuch claim),')
            schema = ('{"case": 1, "claim": "<exact quote or none>"' + field
                      + ', "approve": true}')
        else:
            head = (
                "For each case, first quote the expression in the sentence that the "
                "case is about,\nexactly as the sentence writes it. Then quote the EXACT "
                "words from the sentence that\nstate the architectural claim about the "
                'component (or write "none" if the sentence\nmakes no such claim),')
            schema = ('{"case": 1, "surface": "<exact words from the sentence>", '
                      '"claim": "<exact quote or none>", "approve": true}')

        return f"""Validate components in a document.{f" {focus}" if focus else ""}

COMPONENTS: {', '.join(comp_names)}

{rules}
{tail}
{head}{decide}

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{schema}]}}
JSON only:"""
