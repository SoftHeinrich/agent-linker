"""S-Linker92d — both fidelities, because the branch's own table says they do not nest.

The name-relation note in `approach/CLAUDE.md` states the rule this variant applies:
*"Two cells do not nest ... so compound splitting is a different normalization and the
linker takes the union."*  `s_linker92b` scans one cell and `s_linker92c` the other;
this scans both and unions the spans, which is the only one of the three that is the
whole-name row of the relation rather than a point of it.

It is the completion of the family, not a new idea, and it is priced as such
(`pilot/regex_extract_audit.py`, 30 recorded runs x 5 projects, no calls):

    proposer         pairs   gold   prec    +gold   -gold   newgold  atrisk
    LLM extraction   175.3  150.1  0.856      -       -        -       -
    92b (any_case)   196.9  158.3  0.804   +10.6    -2.4    +10.2    -2.4
    92c (spelling)   198.2  159.1  0.803   +10.8    -1.8    +10.2    -1.8
    92d (union)      198.6  159.5  0.803   +10.8    -1.4    +10.2    -1.4

    arm   policy      TP     FP   macro F1   macro F2
    ctrl     -     180.6   36.4      91.04      92.93
    92b   reject  178.5   32.3      91.19      92.58
    92b   approve 187.8   54.4      89.29      93.84
    92d   reject  179.5   32.7      91.32      92.84
    92d   approve 188.8   54.8      89.43      94.11

**+1.2 gold and +1.7 pairs a run over `92b`**, the best bracket of the four arms at
both ends, and it costs `s_linker92c`'s second relation point plus one union. The
whole fidelity axis above `ANY_CASE` is 1.2 gold wide, which is the honest way to
report it: the swap from an LLM extractor to a scan is worth ~8 net gold a run, and
*which* whole-name fidelity the scan uses is worth ~1.

Every gold pair this adds over `92b` is already in the recorded pipeline's final link
set by another route (`newgold` is 10.2 for all three), so what it buys is redundancy
at the proposer, not reach — and redundancy at a proposer is what the reading round
found the two proposal stages were buying with their call count.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker92b import SLinker92b
from llm_sad_sam.linkers.experimental.s_linker92c import SLinker92c


class SLinker92d(SLinker92c):
    """The whole-name row of the relation: both fidelities, unioned."""

    _VARIANT_NAME = "s_linker92d"

    def _named_spans(self, text, name):
        """The spans either fidelity reaches.

        Duplicates are harmless — `_writes_name` takes the first surviving span and
        `_extract_named_mentions` keys candidates by (sentence, component) — so the
        union is a concatenation and not a set operation over spans.
        """
        return (SLinker92b._named_spans(self, text, name)
                + SLinker92c._named_spans(self, text, name))
