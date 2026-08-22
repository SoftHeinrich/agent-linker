"""S-Linker97 — the merged reading asked case by case (ladder rung F).

Rungs A and E both lose the same thing on the same document, and the dump says
exactly what: of the 13 gold links rung E misses on bigbluebutton, **11 are
refer-backs** (9 sentences write no name at all, 2 write only some other
component's name), and the union over three samples equals the per-sample mean —
the model skips the *same* links every time. A deterministic blind spot, not
instability, so no amount of sampling (rung C) reaches it.

Rung E ruled out the window: asking the merged question at ``COREFERENCE_BATCH``
instead of ``EXTRACTION_BATCH`` moved bigbluebutton's gold from 40.3 to 41.7
against control's 50.3. Granularity was not the mechanism.

What the head's resolver does and the merged reading did not is **oblige a
decision per sentence**. ``_prompt_coref`` renders each target as its own case —
``--- Case N --- TARGET Sx: <text>`` — and asks "for each TARGET sentence below".
The merged reading rendered the block as a flat document listing, where no
sentence is a case and nothing requires the model to account for any particular
one.

That distinction is not a guess; this branch already priced it. ``_prompt_coref``'s
own docstring records s82, which replaced the per-case target text with a
table-only form: gold resolutions held (79.3 against 80.3) but spurious ones rose
to 60.0 from 45.7 and the strict judge downstream kept **30.7 gold instead of
40.3**. The recorded diagnosis was that "the target sentence had simply stopped
being salient next to the question about it."

Why it costs refer-backs specifically, and not named mentions: a written name is
its own anchor — free-form extraction finds it because the string is there. A
refer-back has no surface anchor, so it is found only if the sentence is examined
on purpose. Free-form "extract ALL references" is therefore biased toward exactly
the half of the question the extractor already answered, which is also why the
merged arms keep near-perfect precision while shedding recall.

So this rung merges the question and keeps the obligation: one call, one pass, but
the block is presented as cases and every sentence is one. The case structure,
the batch size and every rule constant are the head's own; the only authored
sentence is the one asking both reference forms of each case, which rung A already
carried.
"""

from __future__ import annotations

import json

from llm_sad_sam.linkers.experimental.s_linker90 import (
    COREF_RULES,
    ENTITY_EXTRACTION_RULES,
    QUALIFIED_CLAUSE,
)
from llm_sad_sam.linkers.experimental.s_linker96 import SLinker96


class SLinker97(SLinker96):
    """One reading pass; every sentence in the block is a case."""

    _VARIANT_NAME = "s_linker97"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"SLinker97 (one reading pass, case by case, batch {self.READING_BATCH})")

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

If a target sentence refers to no listed component in either way, return nothing
for it.

{chr(10).join(blocks)}

Return JSON:
{{"references": [{{"case": 1, "sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence", "antecedent_sentence": M_INTEGER_OR_NULL, "antecedent_text": "exact quote naming the component, or null"}}]}}
JSON only:"""
