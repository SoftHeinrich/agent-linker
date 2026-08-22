"""S-Linker98 — the merged reading with the resolver's whole presentation (rung G).

The ladder converged on this by elimination, and every step is recorded:

    arm                      bigbluebutton gold   what it changed
    control (head, 2 calls)          49.0         --
    A  flat block, batch 50          40.3         merged the question
    E  flat block, batch 10          41.7         + the resolver's batch size
    F  cases,      batch 10          44.3         + the resolver's per-case obligation
    G  cases + window table          this arm     + the resolver's context table

bigbluebutton is the document that decides, because it is the only one of the five
whose gold is coreference-heavy: 28 of its 62 links are refer-backs and 23 of those
have a naming sentence earlier to resolve against. Where there is little to
resolve, every merged arm already matches or beats control on gold and cuts
spurious sharply.

Rung F established the mechanism — a written name is its own anchor, a refer-back
is not, so a sentence must be examined on purpose for its refer-back to be found —
and recovered 4 of the 8.7 gold rung A lost. What it still lacked is the last piece
of the head's resolver presentation. ``_prompt_coref`` shows a SENTENCES table
built from the union of each target's ``_window``, so a case standing at the start
of its batch still has the text before it. A contiguous reading block gives its
first case nothing earlier at all, and an antecedent sits a median of 2 sentences
back — so at every block boundary the nearest anchor is exactly what is missing.

This rung adds that table and changes nothing else. The reading now carries the
resolution question's batch size, its per-case obligation and its context window,
which is the whole of what the dedicated stage supplied; only the *question* is
merged. If bigbluebutton's gold returns here, the round's finding is that the two
proposal questions are mergeable and their presentations are not — and the branch
keeps one proposal stage instead of two for the price of presenting the block the
way the resolver already presented it.
"""

from __future__ import annotations

import json

from llm_sad_sam.linkers.experimental.s_linker90 import (
    COREF_RULES,
    ENTITY_EXTRACTION_RULES,
    QUALIFIED_CLAUSE,
)
from llm_sad_sam.linkers.experimental.s_linker97 import SLinker97


class SLinker98(SLinker97):
    """One reading pass, presented the way the resolver presents its batch."""

    _VARIANT_NAME = "s_linker98"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"SLinker98 (one reading pass, cases + window table, "
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

SENTENCES (the document text the cases are drawn from)
{json.dumps(table or [])}

For each TARGET sentence below, report two kinds of reference, and account for
every case:

1. The sentence writes one of the component's names itself.

{ENTITY_EXTRACTION_RULES}

{QUALIFIED_CLAUSE}

2. The sentence refers to a component without writing any of its names, through a
pronoun or a noun phrase that refers back. Read the TARGET's context in SENTENCES
and give the earlier sentence that states the name it refers back to. A reference
of the first kind needs no antecedent; leave it null.

{COREF_RULES}

If a target sentence refers to no listed component in either way, return nothing
for it.

{chr(10).join(blocks)}

Return JSON:
{{"references": [{{"case": 1, "sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence", "antecedent_sentence": M_INTEGER_OR_NULL, "antecedent_text": "exact quote naming the component, or null"}}]}}
JSON only:"""
