"""S-Linker99 — the merged reading told a sentence may name several (rung H).

The ladder's plateau has one cause, and it is not the one the round set out to
test. Measured over three samples on bigbluebutton, whose gold puts 26 of its 62
links on 12 sentences that reference more than one component:

    arm       sentences given >1 component   gold on those sentences
    control              9.9                       19.7 / 26
    grain  (batch 10)    4.7                       14.0 / 26
    cases  (F)           4.3                       14.0 / 26
    window (G)           4.0                       14.0 / 26

Every merged arm finds exactly 14.0 regardless of batch size, per-case obligation
or context table, and the 5.7-link gap is the whole of the deficit. On teammates,
where only 7 sentences carry several components, every arm finds 19/19 and no arm
loses anything.

So what two proposal stages actually buy is not two questions. It is **two
independent looks at the same sentence**: the extractor reports the component
whose name is written, the resolver independently reports the one referred to, and
the union carries both. One look reports roughly one component per sentence and
moves on — which is also why the merged arms hold near-perfect precision while
shedding recall, and why sampling the same look three times (rung C) recovers
almost nothing: the looks are correlated, the two stages' looks are not.

This rung tests whether a single look can be *told* to do what the second look did.
The only addition is that a sentence may reference more than one component and each
is to be reported separately — a general property of language, that a clause can
have several participants, and not a fact about any document. Everything else is
rung F byte for byte.

If this recovers bigbluebutton, the round's finding is that the two stages are
mergeable and the multiplicity they supplied has to be asked for. If it does not,
the finding is that the second look is doing something a single look cannot be
instructed into, and the head's two proposal stages stand on a measured reason
rather than on habit -- which is worth recording either way, since no row of the
s26-s35 line ever tested it.
"""

from __future__ import annotations

import json

from llm_sad_sam.linkers.experimental.s_linker90 import (
    COREF_RULES,
    ENTITY_EXTRACTION_RULES,
    QUALIFIED_CLAUSE,
)
from llm_sad_sam.linkers.experimental.s_linker97 import SLinker97


class SLinker99(SLinker97):
    """Rung F, plus: one sentence may reference several components."""

    _VARIANT_NAME = "s_linker99"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"SLinker99 (merged reading, multiplicity asked for, "
              f"batch {self.READING_BATCH})")

    def _prompt_reading(self, comp_names, mappings, batch, established, table=None) -> str:
        carried = (
            "ESTABLISHED EARLIER (component: the last sentence before this block "
            "that named it)\n" + json.dumps(established)
            if established else ""
        )
        blocks = [
            f"--- Case {i} ---\nTARGET S{s.number}: {s.text}"
            for i, s in enumerate(batch, 1)
        ]
        return f"""Find every reference to a component, taking each sentence in turn.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings)}' if mappings else ''}
{carried}

For each TARGET sentence below, report two kinds of reference, and account for
every case:

1. The sentence writes one of the component's names itself.

{ENTITY_EXTRACTION_RULES}

{QUALIFIED_CLAUSE}

2. The sentence refers to a component without writing any of its names, through a
pronoun or a noun phrase that refers back. Give the earlier sentence that states
the name it refers back to. A reference of the first kind needs no antecedent;
leave it null.

{COREF_RULES}

A sentence can describe several components at once, and both kinds of reference can
occur in the same sentence. Report every component the sentence references, one
entry each, and do not stop at the first.

If a target sentence refers to no listed component in either way, return nothing
for it.

{chr(10).join(blocks)}

Return JSON:
{{"references": [{{"case": 1, "sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence", "antecedent_sentence": M_INTEGER_OR_NULL, "antecedent_text": "exact quote naming the component, or null"}}]}}
JSON only:"""
