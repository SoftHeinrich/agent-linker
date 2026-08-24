"""S-Linker107 — deliberation shaped by this question, not by a generic template.

Chain-of-thought asks a model to "think step by step" without saying which steps.
We know which steps. Resolving a refer-back is two jobs stapled together:

  1. **bookkeeping** -- which components were named in the sentences just before
     this one, and in which sentence?
  2. **judgement** -- which of those does this pronoun or noun phrase point at?

Job 1 is a fact about the document. ``_states_a_name`` computes it exactly, and the
window is already assembled; making the model re-derive it from raw text is asking
it to do lookup with attention. Job 2 is the linguistic decision only the model can
make. This branch's design law says facts stay in code and weighings go in the
prompt, so this variant does job 1 deterministically and hands the model job 2.

Per case, the prompt now carries **NAMED BEFORE THIS CASE** -- every component the
window actually names ahead of the target, with the sentence number that names it,
latest first. The model quotes the referring expression, picks from that list or
answers none, and the antecedent it cites is one it was handed rather than one it
had to remember.

Why this should reach the failure the round measured. Merged and dedicated arms
alike lose refer-backs by raising their evidence bar (31% fewer claims at +10
precision), and the bar cannot be argued down -- ``s_linker99`` and ``s_linker104``
both tried and both went the wrong way. A shortlist attacks the cause instead of
the symptom: caution is cheap when the candidates are already enumerated and
sourced, and expensive when the model must first convince itself it has not missed
one.

It is also the opposite of ``s_linker102``, which failed. That variant demanded a
verdict on **every** component in the model -- a long, mostly-negative checkbox
that primed rejection and cost 6.5 gold a project-run. Here the list is only what
the window actually names, typically two or three entries, and a refer-back's
antecedent sits a median of 2 sentences back, so the true answer is almost always
on it.

Only ``_prompt_coref`` changes. The loop, the batch size, the SENTENCES table, the
rule constants, every judge and the parser are inherited; the added fields are
ignored downstream. **No extra call, no vendor reasoning feature.**

**Base.** ``s_linker101``, not the head, which lost the paired F2 comparison on
terra in all three runs.
"""

from __future__ import annotations

import json

from llm_sad_sam.linkers.experimental.s_linker90 import COREF_RULES
from llm_sad_sam.linkers.experimental.s_linker101 import SLinker101


class SLinker107(SLinker101):
    """The window's named components are computed in code and handed over."""

    _VARIANT_NAME = "s_linker107"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker107 (per-case antecedent shortlist computed deterministically)")

    def _named_before(self, comp_names, sentence_table, target):
        """Components the table names strictly before `target`, latest first.

        Exact, not heuristic: the same name relation the rest of the module reads
        names with, applied to the sentences the case was already shown.
        """
        latest: dict[str, int] = {}
        for row in sentence_table:
            n = row.get("sentence")
            if not isinstance(n, int) or n >= target:
                continue
            for name in comp_names:
                if self._states_a_name(row.get("text", ""), name):
                    latest[name] = max(latest.get(name, 0), n)
        return sorted(latest.items(), key=lambda kv: -kv[1])

    def _prompt_coref(self, comp_names, sentence_table, targets) -> str:
        blocks = []
        for t in targets:
            near = self._named_before(comp_names, sentence_table, t["target"])
            listed = (", ".join(f"{name} (S{n})" for name, n in near)
                      if near else "none")
            blocks.append(
                f"--- Case {t['case']} ---\n"
                f"TARGET S{t['target']}: {t['text']}\n"
                f"NAMED BEFORE THIS CASE: {listed}"
            )
        return f"""Resolve references (pronouns and noun phrases that refer back) to components.

COMPONENTS: {', '.join(comp_names)}

SENTENCES (the document text the cases are drawn from)
{json.dumps(sentence_table)}

For each TARGET sentence below, identify any pronoun or noun phrase in THAT sentence
that refers back to a component listed above. Read the TARGET's context in SENTENCES.
If a target sentence has no such reference to a listed component, return no resolution
for it. Be conservative — only include resolutions you are CERTAIN about.

Each case lists NAMED BEFORE THIS CASE: the components the sentences above it
actually name, with the sentence that names each, nearest first. That list has
already been checked against the document, so it is where the antecedent will be if
there is one. Quote the referring expression first, then say which entries of that
list could be what it points to, then name the one it does point to.

{chr(10).join(blocks)}

{COREF_RULES}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "reference": "the server", "candidates": ["Name", "OtherName"], "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

JSON only:"""
