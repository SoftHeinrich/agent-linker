"""S-Linker96 — the merged reading asked at the resolution question's granularity.

The stage pilot on terra refuted rung A as a drop-in, and said precisely where.
Merging the two proposal questions is fine on three of five documents (mediastore,
teastore and jabref hold gold exactly and cut spurious to near zero). The whole
aggregate loss sits on the one document whose gold is coreference-heavy:

    in-block resolvable refer-back gold   merged gold vs control
    jabref            0                    0.0
    mediastore        2                    0.0
    teastore          6                   +0.7
    teammates         7                   -1.7
    bigbluebutton    23                   -8.3

The merge did not fail because one call cannot hold two questions — where there is
little to resolve it is strictly better. It failed because it silently changed a
second thing at the same time: **granularity**. The head gives the resolution
question 8-9 calls at ``COREFERENCE_BATCH`` sentences apiece; rung A gave it a
share of 2 calls at ``EXTRACTION_BATCH``. Per-sentence effort on the resolution
question fell about fivefold, and the document with 23 resolvable refer-backs paid
for it.

That is the branch's own measurement, from the NLI probe: **wide premises hurt
monotonically**, and SummaC reports the same premise-granularity effect
independently. The two stages differed in the question they asked *and* in the
window they asked it over. Only the first is redundant.

So this variant merges the question and keeps the window. It is rung A with one
constant changed — the reading runs at the head's own ``COREFERENCE_BATCH``, the
value the head already chose for the question that turns on locality. Everything
else, including the carried per-component note that spans block boundaries, the
routing, all three judges, the alias module and the deterministic scan, is
inherited untouched.

The cost win narrows from ~48% fewer LLM calls to ~18% (40 reading calls across
the five documents against the head's 49 proposal calls), and the architectural
win is the one that was actually wanted: **three proposal stages become two** — one
reading pass and one deterministic scan — and three batch constants become two.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker90 import SLinker90
from llm_sad_sam.linkers.experimental.s_linker94 import SLinker94


class SLinker96(SLinker94):
    """One reading pass, at the granularity the resolution question needs."""

    _VARIANT_NAME = "s_linker96"

    #: The head's own value for the question that turns on locality. Not a new
    #: number: the reading asks the resolution question, so it reads at the size
    #: the resolution question was already measured at.
    READING_BATCH = SLinker90.COREFERENCE_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"SLinker96 (one reading pass at batch {self.READING_BATCH})")
