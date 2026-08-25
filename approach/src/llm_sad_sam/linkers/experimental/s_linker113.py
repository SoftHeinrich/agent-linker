"""S-Linker113 — the sortal gate enumerates the readings, then commits.

`s_linker112` restores the branch's own order at this gate: evidence before verdict.
This adds the step that order exists to make room for, and it is `s_linker92f`'s, at the
second of the two sites where the law it came from applies.

**The law, and why this site qualifies.** *The alternative set is a fact when the case
contains it and a weighing when it does not.* The partial-name gate's case is an
expression and its sentence, and the question is which of the readings that expression
could carry in that sentence it actually carries -- a lowercased word of a component's
name has one reading on which it is that component's word and others on which it is
ordinary vocabulary. No table holds those readings, so the model is the only thing that
can enumerate them, which is the condition `s_linker106` failed and `s_linker92f` met.

**Why not more information instead.** The identity question at this gate cannot be
repaired by telling the judge more: the target is withheld by measurement
(`s_linker25`, -5.5 gold when shown), the alias table is refused at the judges
(`s_linker108`, macro F2 -0.40 and precision -3.1), and the sibling set is a fact the
judge cannot be shown, which is why `s_linker109` settles it in code instead. A level-1
audit of the head's own runs (`pilot/chooser_audit.py`) closes the last of those doors:
after `s_linker109` the whole ceiling of a contrastive chooser here is **2.3 false
positives a run on terra and 0.7 on luna**, against 8.0 and 3.7 true positives it would
put at risk -- the consolidation round priced that arm at -8.3 FP on the base it has
since replaced, and the repair it adopted ate the arm's own headroom. What is left at
this gate is not information the case lacks. It is the weighing of what the case
already carries.

Two strings over `s_linker112`, no rule text and no new field of the parser: the reply
lists the readings and names the one it has, between the quote and the verdict.

**REFUSED at level 2: the sign flips between models, opposite to `s_linker112`'s**
(`results/judge_round/README.md`). terra gold 21.3 -> 14.3 at spurious 8.0 -> 8.0 -- a
loss with no compensating side; luna gold 21.3 -> 21.7 at spurious 9.7 -> 3.3, the best
arm of either model. Two arms at one gate changing sign in opposite directions is the
signature of a gate that was never separated from noise, not of two models disagreeing.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker112 import SLinker112


class SLinker113(SLinker112):
    """The head, with the sortal gate weighing readings between evidence and verdict."""

    _VARIANT_NAME = "s_linker113"

    QUOTE_LINE = ("For each case, first quote the exact words of the source sentence "
                  "that the expression\nis used in -- a contiguous exact substring. "
                  "Then list the readings that expression\ncould have here, and name "
                  "the one it has. Then classify what it denotes there.")

    SCHEMA = ('{"judgments":[{"case":1,"claim":"exact source quote",\n'
              '"readings":["...","..."],"reading":"<the one it has>",\n'
              '"denotation":"participant"}]}')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker113 (sortal gate: readings enumerated between quote and verdict)")
