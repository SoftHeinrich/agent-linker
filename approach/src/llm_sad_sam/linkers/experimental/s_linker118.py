"""S-Linker118 — the sortal gate is asked for the ground too.

`s_linker116` puts the strict gate's `objection` field at the lenient gate; this puts
it at the sortal one. Together they are the two arms a single reply schema for all
three judges needs, taken one gate at a time so a result is attributable to a gate.

The sortal gate is the one the branch has the least room to move: 0.31 / 0.19 gold
among its own candidates, and of the five projects only one contributes gold to it on
either model. `s_linker112` and `s_linker113` both changed sign between models here.
So this arm is measured for completeness and read with that ceiling in mind — what it
can establish is that the field is not free, not that it is worth having.

One line and one field, and the line is the strict gate's demand narrowed to the two
readings this gate names. The question, `QUALIFIED_CLAUSE`, the withheld target, the
sentence table, the case list and the batch size are the head's.

GATE-01: `s_linker92` is untouched. GATE-06/07: the added line names no surface, no
component and no term of any document; it states the strict gate's demand in the
vocabulary this gate's own question already uses.
"""

from __future__ import annotations

from dataclasses import replace

from llm_sad_sam.linkers.experimental.s_linker114 import SLinker114


class SLinker118(SLinker114):
    """The head, with the sortal gate's reply carrying a ground for the other reading."""

    _VARIANT_NAME = "s_linker118"

    #: The head's line about the quote, and the same line followed by the ground.
    QUOTE_LINE = "Claim must be a contiguous exact substring of the source sentence."
    WITH_GROUND = (
        "Claim must be a contiguous exact substring of the source sentence.\n"
        "State also the strongest ground there is for the other reading (or \"none\" "
        "if\nthere is none). A ground you could raise against most expressions is not "
        "a\nground here.")

    #: The head's reply, and the same reply carrying the ground.
    SCHEMA = '"claim":"exact source quote"}]}'
    SCHEMA_WITH_GROUND = ('"claim":"exact source quote",\n'
                          '"objection":"<strongest ground for the other reading, or '
                          'none>"}]}')

    @classmethod
    def _prompt_denotation_ground(cls, linker, comp_names, cases, shared, context):
        """The head's denotation prompt with one line and one field added."""
        prompt = SLinker114._denotation_prompt(linker, comp_names, cases, shared,
                                               context)
        for old, new in ((cls.QUOTE_LINE, cls.WITH_GROUND),
                         (cls.SCHEMA, cls.SCHEMA_WITH_GROUND)):
            if prompt.count(old) != 1:
                raise RuntimeError(
                    f"s_linker118: {prompt.count(old)} sites for a substitution that "
                    f"must have exactly one -- the head's sortal prompt moved")
            prompt = prompt.replace(old, new, 1)
        return prompt

    @staticmethod
    def _denotation_decision(item, verdict, keep, context):
        """The head's row, plus the ground the reply now carries."""
        row = SLinker114._denotation_decision(item, verdict, keep, context)
        return {**row, "objection": verdict["objection"]}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker118 (sortal gate: the ground for the other reading is asked for)")


SLinker118.DENOTATION = replace(
    SLinker114.DENOTATION,
    prompt=SLinker118._prompt_denotation_ground,
    decision=SLinker118._denotation_decision,
)
