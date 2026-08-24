"""S-Linker106 — deliberation we implement ourselves, not reasoning we rent.

The API's ``reasoning_effort`` is a vendor knob: opaque, priced per hidden token,
unavailable on other backends, and applied to every call in the process whether it
needs it or not. This variant asks the same question a different way -- **make the
model deliberate in the answer it returns**, which is chain-of-thought (Wei et al.,
NeurIPS 2022) applied to the one question this branch has measured as
inference-bound.

Where it goes, and why only there. Splitting merged-arm proposals by reference
kind showed the pipeline's two proposal questions are not alike:

    question                     proposed/run   precision
    naming (surface-anchored)        34.8         94.1%
    refer-back (inference)           31.5         43.9%

The naming question reads a string off the page and is already at 94%; there is
nothing to deliberate about, and a scratchpad would only cost tokens. The
refer-back question has to decide what "it" points at among a dozen candidates,
sits at 44%, and is the question whose evidence bar the merge study showed rising
whenever the model is put under any pressure to be careful.

A JSON-only reply actively suppresses reasoning: the schema's first field is the
answer, so the model commits before it works anything out. This variant reverses
that order inside the same call -- the reply names the referring expression, lists
the candidate antecedents it weighed, and only then commits to one. **No extra LLM
call, no extra stage, no vendor feature.**

Deliberately not changed: the rule constants, the case structure, the SENTENCES
table, the batch size, every judge, and the parser -- the added fields are ignored
by ``_resolve_references``, which reads only the fields it always read, so the
deliberation is free to the rest of the pipeline.

**Base.** Built on ``s_linker101``, not the head: the head lost the paired F2
comparison on terra in all three runs (93.1/93.8/94.0 against 95.3/95.1/95.6).
"""

from __future__ import annotations

import json

from llm_sad_sam.linkers.experimental.s_linker90 import COREF_RULES
from llm_sad_sam.linkers.experimental.s_linker101 import SLinker101


class SLinker106(SLinker101):
    """The resolver weighs its candidates aloud before it commits to one."""

    _VARIANT_NAME = "s_linker106"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker106 (resolver deliberates in-reply before committing)")

    @staticmethod
    def _prompt_coref(comp_names, sentence_table, targets) -> str:
        blocks = [
            f"--- Case {t['case']} ---\n"
            f"TARGET S{t['target']}: {t['text']}"
            for t in targets
        ]
        return f"""Resolve references (pronouns and noun phrases that refer back) to components.

COMPONENTS: {', '.join(comp_names)}

SENTENCES (the document text the cases are drawn from)
{json.dumps(sentence_table)}

For each TARGET sentence below, identify any pronoun or noun phrase in THAT sentence
that refers back to a component listed above. Read the TARGET's context in SENTENCES.
If a target sentence has no such reference to a listed component, return no resolution
for it. Be conservative — only include resolutions you are CERTAIN about.

Work each case out before you answer it. In the reply, first quote the referring
expression, then list every component the surrounding sentences make a possible
antecedent for it, and only then name the one it refers to. Fill those fields in
that order: what the expression is, what it could point to, what it does point to.

{chr(10).join(blocks)}

{COREF_RULES}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "reference": "the server", "candidates": ["Name", "OtherName"], "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

JSON only:"""
