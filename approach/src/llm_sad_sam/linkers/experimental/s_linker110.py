"""S-Linker110 — the resolver's candidate antecedents, enumerated in code.

`s_linker107` measured this at the proposal stage and it is the only arm of the
reading round that moved spurious without moving gold: over three samples on five
projects, **spurious 27.0 → 17.0 a project-run at gold 45.6 → 45.3**, precision
0.628 → 0.727. It was built on `s_linker101`, which the consolidation audit has since
retired — two thirds of what the third blind look added is gold the scan proposes for
free, and the remainder is inside the recorded null floor. So the shortlist is rebased
here onto the adopted head instead, where the arm is one prompt override and nothing
else.

**The rule the round arrived at, stated once.** Three arms asked the same structural
question — enumerate the alternatives, then commit — at two places, and the four
results only agree under one reading:

    where                        who enumerates the alternatives     result
    resolver (`s_linker106`)     the model                           spurious +6.6
    resolver (`s_linker107`)     code                                spurious -10.0
    lenient gate (`s_linker92e`) nobody: quote the surface, no more   refuted
    lenient gate (`s_linker92f`) the model                           best terra F1

**The alternative set is a fact when the case contains it and a weighing when it does
not.** Which components the sentences above name is a fact — `_states_a_name` computes
it exactly, and asking the model to re-derive it from the raw table is asking it to do
lookup with attention, which it does by inventing. Which readings a lowercased word
could have in its sentence is in no table, so only the model can enumerate them, and
`s_linker92e` fails precisely because it enumerates nothing and merely echoes. That is
the branch's design law — facts in code, weighings in the prompt — applied to the
*alternative set* rather than to the rule, and it is what makes `s_linker106` and
`s_linker92f` agree instead of contradict.

Per case the prompt carries **NAMED BEFORE THIS CASE**: every component the window
actually names ahead of the target, with the sentence that names it, nearest first.
The model quotes the referring expression, says which entries of that list it could
point to, and names the one it does — or none. A refer-back's antecedent sits a median
of 2 sentences back, so the true answer is almost always on it. Measured over the
resolver's own windows (`pilot/test_s110_shortlist.py`, 122 checks, no calls), the list
carries **1.8 to 4.5 of a catalog's 6 to 14 components a case** — mediastore 2.8/14,
teastore 1.8/11, teammates 2.2/8, bigbluebutton 2.2/12, jabref 4.5/6. It is a shortlist
in fact and not only in intent, which is what separates it from `s_linker102`: that
variant demanded a verdict on **every** component in the model, a long mostly-negative
checkbox that primed rejection and cost 6.5 gold a project-run.

Only `_prompt_coref` changes. The loop, the batch size, the `SENTENCES` table, the rule
constants, every judge and the parser are inherited, and the two added reply fields are
ignored downstream. **No extra call and no vendor reasoning feature.**

**Level 2, measured on both models** (`pilot/reading_pilots.py --arms control
shortlist2`, three samples x five projects, both arms in the same invocation per model;
`control` is the head's resolver, whose `_prompt_coref` is byte-identical between s90 and
s92, so the arms differ at the resolver and nowhere else):

    model  arm         proposals   gold   spurious   precision
    terra  control        53.7      36.7    16.9       0.684
    terra  shortlist2     48.8      36.5    12.3       0.749
    luna   control        74.8      36.4    38.4       0.487
    luna   shortlist2     59.1      35.9    23.1       0.608

**Spurious down on both models at a gold cost of 0.2 and 0.5** — luna -15.3, above the
recorded null floor of FP 10.7; terra -4.7, inside it. This is the reading round's
result reproduced on a second model and a second base: the shortlist is the only arm of
that round that moved spurious without moving gold, and it does so again here.

**Level 4 is owed and this variant does not claim it.** A stage arm screens candidates
and does not decide them, and the composition risk is the resolver's usual one: a
refer-back the shortlist withholds is a pair the strict judge never sees. The batch is
`pilot/run_consolidation_e2e.sh`.

**Base.** `s_linker109` — the adopted head plus the nesting refusal, which is decided
at level 1 and owes nothing.
"""

from __future__ import annotations

import json

from llm_sad_sam.linkers.experimental.s_linker92 import COREF_RULES
from llm_sad_sam.linkers.experimental.s_linker109 import SLinker109


class SLinker110(SLinker109):
    """The window's named components are computed in code and handed over."""

    _VARIANT_NAME = "s_linker110"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker110 (per-case antecedent shortlist computed deterministically)")

    def _named_before(self, comp_names, sentence_table, target):
        """Components the table names strictly before ``target``, latest first.

        Exact, not heuristic: the same name relation the rest of the module reads
        names with, applied to the sentences the case was already shown.
        """
        latest: dict[str, int] = {}
        for row in sentence_table:
            number = row.get("sentence")
            if not isinstance(number, int) or number >= target:
                continue
            for name in comp_names:
                if self._states_a_name(row.get("text", ""), name):
                    latest[name] = max(latest.get(name, 0), number)
        return sorted(latest.items(), key=lambda item: -item[1])

    def _prompt_coref(self, comp_names, sentence_table, targets) -> str:
        blocks = []
        for target in targets:
            near = self._named_before(comp_names, sentence_table, target["target"])
            listed = (", ".join(f"{name} (S{number})" for name, number in near)
                      if near else "none")
            blocks.append(
                f"--- Case {target['case']} ---\n"
                f"TARGET S{target['target']}: {target['text']}\n"
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
