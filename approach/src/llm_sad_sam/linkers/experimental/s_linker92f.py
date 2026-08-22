"""S-Linker92f — the lenient gate weighs the readings of the surface before it commits.

`s_linker92e` moves the case's own surface in front of the verdict. This goes one
step further along the same mechanism and stops there: the reply lists **the readings
the quoted expression could have in this sentence**, names the one it has, and only
then states the claim and decides.

That is `s_linker106`'s template, transposed. At the resolver, a reply that names the
referring expression, lists every component the context makes a possible antecedent,
and only then commits, is chain-of-thought applied to a choice among candidates. The
use/mention call the lenient gate leaks on is the same shape of choice: a surface like
a lowercased common noun has two readings, one of which is the component, and a
JSON-only reply whose first field is the verdict commits before either is written
down.

Nothing enumerates the readings for the model. The prompt does not say "the name or
the ordinary word" — that is `STRICTER_CLAUSE`'s sentence, already in the prompt, and
restating it is the repair this round refuses. The model supplies its own candidate
readings, which is what makes this a thinking template rather than a rule.

Two fields more than the head, no rule text added, changed or removed, and the strict
branch of the builder renders `s_linker92`'s bytes exactly.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker92 import (
    LAYERED_COREF_RULES, LAYERED_ENTITY_RULES, QUALIFIED_CLAUSE, STRICTER_CLAUSE,
)
from llm_sad_sam.linkers.experimental.s_linker92a import SLinker92a


class SLinker92f(SLinker92a):
    """The lenient gate writes down what the expression could be doing, then decides."""

    _VARIANT_NAME = "s_linker92f"

    @staticmethod
    def _prompt_validation(comp_names, cases, focus, strict: bool = False) -> str:
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

        if strict:
            head = ("For each case, first quote the EXACT words from the sentence "
                    "that state the\narchitectural claim about the component (or write "
                    '"none" if the sentence makes no\nsuch claim),')
            schema = '{"case": 1, "claim": "<exact quote or none>"' + field + ', "approve": true}'
        else:
            head = (
                "Work each case out before you answer it. First quote the expression in "
                "the sentence\nthat the case is about, exactly as the sentence writes it. "
                "Then list the readings that\nexpression could have here, and name the one "
                "it has. Then quote the EXACT words from\nthe sentence that state the "
                'architectural claim about the component (or write "none"\nif the sentence '
                "makes no such claim),")
            schema = ('{"case": 1, "surface": "<exact words from the sentence>", '
                      '"readings": ["...", "..."], "reading": "<the one it has>", '
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
