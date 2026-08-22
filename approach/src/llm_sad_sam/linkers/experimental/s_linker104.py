"""S-Linker104 — the merged reading with the refer-back question's own standard.

Splitting every merged arm's output by reference kind shows the merge costs the
naming question nothing at all, and costs the refer-back question its nerve:

    arm       kind     proposed/run   correct/run   precision
    control   names        34.8          31.5         94.1%
    merged    names        34.9          31.7         94.3%
    control   refers       31.5          12.6         43.9%
    merged    refers       21.7          10.2         53.9%

The merged reader emits **31% fewer refer-back claims at 10 points higher
precision**. That is a raised evidence threshold, not lost ability: it is being
more careful about the question it should be least careful about.

The likely cause is co-location. ``ENTITY_EXTRACTION_RULES`` is strict about
surface form -- report only when the sentence writes the name, spelled as the list
spells it -- and when that rule sits in the same prompt as the refer-back
question, the strict standard appears to carry over to a question that has no
surface form to check.

The head keeps the two standards apart by keeping the two questions in separate
calls. It also gives its resolver an **active search instruction** that the merged
prompt never had: *"For each TARGET sentence below, identify any pronoun or noun
phrase in THAT sentence that refers back to a component listed above."* The merged
prompt only ever described what a refer-back is; it never told the model to go
looking for one.

This variant adds that instruction, verbatim from ``_prompt_coref``, and nothing
else. No new authored rationale (GATE-07: the sentence is the head's own text),
no structural change, no extra call. If recall on the refer-back stream returns,
the merge's failure was a threshold artefact and is fixable inside one prompt; if
it does not, the two standards genuinely need two calls to stay apart.
"""

from __future__ import annotations

import json

from llm_sad_sam.linkers.experimental.s_linker90 import (
    COREF_RULES,
    ENTITY_EXTRACTION_RULES,
    QUALIFIED_CLAUSE,
)
from llm_sad_sam.linkers.experimental.s_linker97 import SLinker97


class SLinker104(SLinker97):
    """Rung F, with the resolver's own search instruction restored."""

    _VARIANT_NAME = "s_linker104"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"SLinker104 (merged reading, refer-back standard restored, "
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
every case. The two kinds are judged on their own standards; the requirements of
the first do not apply to the second.

1. The sentence writes one of the component's names itself.

{ENTITY_EXTRACTION_RULES}

{QUALIFIED_CLAUSE}

2. The sentence refers to a component without writing any of its names. For each
TARGET sentence below, identify any pronoun or noun phrase in THAT sentence that
refers back to a component listed above. Give the earlier sentence that states the
name it refers back to. A reference of the first kind needs no antecedent; leave
it null.

{COREF_RULES}

If a target sentence refers to no listed component in either way, return nothing
for it.

{chr(10).join(blocks)}

Return JSON:
{{"references": [{{"case": 1, "sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence", "antecedent_sentence": M_INTEGER_OR_NULL, "antecedent_text": "exact quote naming the component, or null"}}]}}
JSON only:"""
