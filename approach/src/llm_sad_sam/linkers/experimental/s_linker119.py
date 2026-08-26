"""S-Linker119 — one reply schema at all three judges, which means moving the sortal one.

The question the round asks is whether the three judges can write one thing. Two of
them already do: the lenient and strict gates both reply `{"validations":[{case,
claim[, objection], approve}]}`, built by one method (`s_linker92._prompt_validation`).
The sortal gate is the whole divergence — a different reply key, the verdict before the
quote, and an enum where the others carry a boolean.

This arm moves it, all three ways at once, because a single schema is what is being
priced and half of one is not a schema:

| | head, sortal gate | this arm |
| --- | --- | --- |
| reply key | `judgments` | `validations` |
| field order | verdict, then quote | quote, then ground, then verdict |
| verdict | `denotation`: participant / associated | `approve`: true / false |

What does **not** move: the question the gate asks, the withheld target, the withheld
catalog (`shows_catalog=False`), `QUALIFIED_CLAUSE`, the sentence table, the case list,
the batch size, and the polarity — a case is kept only when the gate positively
approves it, which is the enum's `participant` demand restated as the boolean the other
two use.

`s_linker118` prices the ground alone and `s_linker112` priced the order alone (refused,
sign flip); this is the composition plus the verdict's type, which nothing has priced.
Read together with `s_linker116` at the lenient gate, the two arms are the whole
uniform design: if both are neutral or better, the three judges share one schema and
one parser; if either loses, the schema stays split and `s_linker114`'s `verdict_field`
is where the split is declared instead of duplicated.

**The known risk, stated before the run.** This branch's law puts a gate's default
polarity at the base rate of the stream feeding it, and this gate's is 0.31 / 0.19 — the
dirtiest of the three. The enum makes the gate answer a classification question; a
boolean `approve` is the lenient gate's own vocabulary, and the lenient gate's default
is the opposite of this one's. If the arm loses gold by approving more, that is not the
schema failing, it is the vocabulary carrying a polarity, and it is the most useful
thing this arm can find out.

GATE-01: `s_linker92` is untouched. GATE-06/07: `QUALIFIED_CLAUSE` is imported; the
decide clause is `s_linker92`'s strict clause narrowed to this gate's two readings, and
names no surface, component or document term.
"""

from __future__ import annotations

import json
from dataclasses import replace

from llm_sad_sam.linkers.experimental.s_linker92 import QUALIFIED_CLAUSE
from llm_sad_sam.linkers.experimental.s_linker114 import SLinker114


class SLinker119(SLinker114):
    """The head, with all three judges replying in one schema."""

    _VARIANT_NAME = "s_linker119"

    @staticmethod
    def _prompt(linker, comp_names, cases, shared, context) -> str:
        """The head's sortal question, in the schema the other two judges use.

        The first paragraph is the head's own question with its two readings named as
        the verdict they now write; everything below the clause is the head's prompt.
        """
        return f"""Decide whether each expression itself denotes a software
participant in its local context: approve when the expression denotes a software
participant, reject when it denotes something merely associated with software.

{QUALIFIED_CLAUSE}

SENTENCES
{json.dumps(shared)}

CASES
{json.dumps(cases)}

For each case, first quote the exact words of the source sentence the expression is
used in -- a contiguous exact substring -- then state the strongest ground there is
for rejecting this case (or "none" if there is none), then decide: approve unless
that ground is decisive.

JSON only:
{{"validations":[{{"case":1,"claim":"exact source quote",
"objection":"<strongest ground to reject, or none>","approve":true}}]}}
"""

    @staticmethod
    def _decision(item, verdict, keep, context):
        """The head's row, written from a boolean verdict.

        `denotation` is kept in the row because the stage's feedback view and every
        audit tool on this branch read it; it now records which reading the boolean
        committed to rather than a word the model chose.
        """
        return {"approved": False, "requested_keep": False,
                "evidence_valid": verdict["valid"],
                "claim": verdict["claim"], "objection": verdict["objection"],
                "denotation": "participant" if keep else "associated",
                "alternative": "not reviewed", "path": "denotation",
                "stage": "partial_name"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker119 (one reply schema at all three judges)")


SLinker119.DENOTATION = replace(
    SLinker114.DENOTATION,
    reply_key="validations",
    prompt=SLinker119._prompt,
    verdict_field="approve", verdict_values=None,
    keep=lambda v: v["verdict"] == "approve",
    decision=SLinker119._decision,
)
