"""S-Linker122 — `s_linker121` with the anchor block gone and one clause in its place.

The union judge's case carries four pieces of evidence computed from the match. Three
are one line each; the fourth, `anchors`, is up to `ANCHOR_LIMIT` whole sentences of the
document, and it is **27.9% of a judging call** (the figure the compaction round measured
and `s_linker88` compacted rather than removed). This variant removes it.

**What goes, and what replaces it.** The block is not computed and not printed, and the
rule's `anchors` line goes with it — a line about a field no case carries measures
something else. In its place, one sentence:

    That a surface can name this component is not evidence that it does here.

73 bytes against the block's ~350 per case. It is a **weighing**, which is what the
design law permits a prompt to carry, and it is not a restatement of `STRICTER_CLAUSE`:
that clause is about an ordinary English word coinciding with a name, and the row the
anchors were actually holding is the one the alias stage supplies, where the surface is
not an ordinary English word at all.

**Why a clause at all, and why this one.** The stage round's error analysis
(`pilot/anchor_diff.py`, `../results/s121_ablations/`) located what the anchors were
buying on terra: of the 33 sample-counts the anchor-free arm keeps and the head does not,
30 are on cases whose block was actually printed, **22 are the alias row, and 18 of those
are one component** — six sentences about a platform reading as sentences about the
component an alias bound to its name. The `writes` line asserts that this surface means
this component and, with the anchors gone, nothing else in the call can contradict it.
The clause says it may be doubted. A first, 456-byte version of the same weighing was
measured and is kept as an arm: it enumerated the readings a surface could have and
closed by restating `STRICTER_CLAUSE`, and this branch has priced both shapes before
(s71/s72 for the enumeration, s86/s87 for the restatement at the lenient gate).

**What this is not.** It is not a compaction of the anchors — `s_linker88` did that,
losslessly, and this throws the fact away. The stage arms say the trade is real on terra
(`noanchor` spurious +7.0 a run, net −11.0, p = 0.016) and near-neutral on luna, so this
variant exists to be measured end to end, where the coreference linker behind the name
stage gets to re-propose what the name stage now drops.

A subclass, not a standalone file: the branch's one-file-per-variant policy is for the
**reported** arm. If this is adopted it gets inlined the way `s_linker120` was.

Measurements: `../results/s121_ablations/README.md`.
"""
from __future__ import annotations

from llm_sad_sam.linkers.experimental import s_linker121 as base
from llm_sad_sam.linkers.experimental.s_linker121 import SLinker121, UNION_DEMAND

#: The rule's anchors line, computed off the head's own constant rather than retyped: it
#: is the last line of `_FIELD_LINES`, and removing it is what "no anchors" means in the
#: rule. A drift in the constant is a drift here, and `_prompt_union` asserts the slice
#: actually removed something rather than silently matching nothing.
ANCHOR_RULE_LINE = "\n  anchors -- " + base._FIELD_LINES.split("  anchors -- ", 1)[1]

#: The weighing that replaces the fact. One sentence, no enumeration, no restatement.
#: Ground: general — use versus mention, which holds for any text and names no surface
#: form, no component and no document shape (GATE-06/07).
SURFACE_NOT_EVIDENCE = (
    "That a surface can name this component is not evidence that it does here."
)


class SLinker122(SLinker121):
    """`s_linker121` minus the anchor evidence, plus `SURFACE_NOT_EVIDENCE`."""

    _VARIANT_NAME = "s_linker122"

    def _union_evidence(self, candidate, components, sent_map):
        """The head's evidence with `anchors` emptied.

        Emptied rather than removed: `_format_union_case` prints a field only when the
        match computed one, so an empty list is exactly "this case has no anchors" and
        the case shape stays the head's for every other field.
        """
        evidence = super()._union_evidence(candidate, components, sent_map)
        evidence["anchors"] = []
        return evidence

    def _prompt_union(self, comp_names, sentence_table, cases) -> str:
        """The head's call, minus the anchors line, plus the clause before the demand."""
        prompt = super()._prompt_union(comp_names, sentence_table, cases)
        without = prompt.replace(ANCHOR_RULE_LINE, "", 1)
        assert without != prompt, "the anchors line was not removed"
        placed = without.replace(
            UNION_DEMAND, f"{SURFACE_NOT_EVIDENCE}\n\n{UNION_DEMAND}", 1)
        assert placed != without, "the clause was not placed"
        return placed
