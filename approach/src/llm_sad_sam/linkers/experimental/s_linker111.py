"""S-Linker111 — the lenient gate weighs the readings of the surface, at the head.

`s_linker92f` is the one arm of the regex round that the consolidation round left
owing and did not retire: **best terra macro F1 of every arm measured, with false
positives below the control**, refused only on luna's trade and only end to end. It was
built on `s_linker92a`; the head has moved twice since (`s_linker109`'s scan refusal,
`s_linker110`'s antecedent shortlist), and neither change touches the lenient gate. This
is that arm at the head, and nothing else.

**Why the law predicts this site and only this shape.** The consolidation round's
transferable result is that *the alternative set is a fact when the case contains it and
a weighing when it does not*. At the lenient gate the case line is already
`Case 1: "<surface>" -> <Name>`: how the sentence writes the name and how the catalog
writes it stand side by side in every case, so the fact is present and `s_linker92e`'s
refusal is what supplying it again is worth — nothing. What is absent is the weighing:
which readings that surface could carry in that sentence. `STRICTER_CLAUSE` states the
standard for choosing between them and no field of the reply makes the choice, so a
JSON-only reply whose first field is the verdict commits before either reading is
written down.

Measured off the head's own recorded runs (`pilot/lenient_audit.py`, six runs, no
calls), the gate's false positives concentrate exactly where that choice is hard: where
every writing of the name in the sentence is **lowercased** while the catalog
capitalises it, precision in the gate's kept links falls to 0.55 on terra and 0.43 on
luna against 0.90 and 0.80 overall, and that bucket holds **60% and 78% of all its false
positives**. The bucket is a fact about the case, computed from `spans` and nothing
else; it is not passed to the model, because `STRICTER_CLAUSE` already says how a
writing is to be weighed — *"how the word is written is evidence either way and never
settles it on its own"* — and naming the bucket in the prompt would supply a polarity
the rule denies it.

The template is `s_linker92f`'s, delegated rather than copied so the two cannot drift.
The strict branch renders `s_linker92`'s bytes exactly, so the coreference judge is
unchanged, and the partial-name gate does not use this builder at all.

**REFUSED at level 2 on both models, under F2** (`results/judge_round/README.md`). Over
the lenient gate's own fixed candidates, three samples a model: terra gold 150.0 -> 142.0
at spurious 13.0 -> 8.7; luna gold 153.3 -> 148.0 at spurious 32.0 -> 22.0. That is 0.54
and 1.9 false positives saved per gold link lost, against the three this budget demands,
so `3*gold - spurious` falls on both (-19.7, -6.0). The regex round's end-to-end reading
of the same template as terra's best macro F1 is not contradicted -- it is the same trade
priced at F1 instead of F2.

The arm also **destabilises the verdict**: on a candidate set that does not move, the
control's runs differ by 2.0 links on terra and 10.0 on luna, and this arm's by 10.7 and
26.0. No call failed and the longest reply used 1642 of 4096 completion tokens, so it is
not truncation. On both models the loss concentrates on the one project whose stream is
half spurious rather than 84-90% gold.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker92f import SLinker92f
from llm_sad_sam.linkers.experimental.s_linker110 import SLinker110


class SLinker111(SLinker110):
    """The head, with the lenient gate's reply enumerating readings before it decides."""

    _VARIANT_NAME = "s_linker111"

    #: `s_linker92f`'s builder unchanged. Both branches of it are inherited: the
    #: strict one is byte-identical to `s_linker92`'s, so only the lenient gate moves.
    _prompt_validation = staticmethod(SLinker92f._prompt_validation)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker111 (lenient gate: readings enumerated, then the verdict)")
