"""S-Linker110-NoCodeRef — the coreference linker with nothing computed for it.

The floor arm for RQ4's coreference half. `s_linker110_noevidence` removed the
*context* code computes and kept the window, on the argument that a refer-back with no
earlier sentences is unanswerable in principle. That argument is wrong at the right
limit: it holds only if the model is given *less* than the document. Given the whole
document it holds nothing back, and the window is revealed as what it always was --
a code-computed shortcut, not a precondition of the task.

So this arm hands the resolver the two things a human would be handed, and nothing
else: **the document, and the component list.** Everything the head decides for it
goes:

    what the head computes                          this arm
    -----------------------------------------------------------------------
    which sentences are targets, batched 10 at a    no targets; the model reads the
    time (`_iter_batches`, COREFERENCE_BATCH)       document and finds them itself
    a +/-5 window union per batch, rendered as      the whole document, once
    the SENTENCES table (`_window`)
    NAMED BEFORE THIS CASE -- the components the    nothing; the model works out
    document names ahead of each target, checked    what was named where
    against the document (`_named_before`, s110)

**What is deliberately kept.** The reply schema, so the parser is the head's and the
strict coreference judge downstream receives the same `reference` / `antecedent_sentence`
/ `antecedent_text` fields it always has -- those are the *model's own* output carried
forward, not a fact code supplied, and the ablation is about the latter. The
post-hoc validity checks stay too (a returned sentence number must name a real
sentence, a returned component must be in the catalog): they reject a reply, they do
not inform one, so they are parsing rather than evidence. `COREF_RULES` is rendered
verbatim -- this arm removes computed input, not authored rules, which is the finetune
round's question and not this one.

**The confound this arm carries, stated rather than hidden.** `s_linker27` measured a
whole-document call and found accuracy tracks document length (jabref, 13 sentences,
100.0; teammates, 198, 84.1). Batching is itself a code-computed decision, so the floor
cannot keep it and stay a floor -- but a loss here is therefore *at most* an upper bound
on what the computed context is worth, and on the longest document some of it is s27's
effect rather than the shortlist's. Read the per-project numbers before the macro.

**The other two linkers' floors, for the record.** The full-name linker's
"nothing computed" form already exists and is already measured: it is `s_linker92`'s
extraction pass, which the regex round replaced with the scan. The partial-name
linker has no measured floor -- its candidates are entirely code-proposed.

**Measurement owed:** level 4, paired with `s_linker110` in every invocation.
Unmeasured at the time of writing.
"""

from __future__ import annotations

import json

from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.linkers.experimental.s_linker92 import COREF_RULES, get_comp_names, parse_snum
from llm_sad_sam.linkers.experimental.s_linker110_noevidence import SLinker110NoEvidence


class SLinker110NoCodeRef(SLinker110NoEvidence):
    """The resolver is given the document and the components, and nothing else."""

    _VARIANT_NAME = "s_linker110_nocoderef"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker110NoCodeRef (resolver: whole document, no targets, no window)")

    def _prompt_coref_document(self, comp_names, sentences) -> str:
        """The whole document and the component list. No targets, no window, no list.

        `COREF_RULES` and the reply schema are the head's; what is missing is every
        field the head computes -- which sentences to look at, which sentences to look
        at them against, and which components were named before each one.
        """
        document = [{"sentence": s.number, "text": s.text} for s in sentences]
        return f"""Resolve references (pronouns and noun phrases that refer back) to components.

COMPONENTS: {', '.join(comp_names)}

DOCUMENT
{json.dumps(document)}

Read the whole document. Find every sentence that refers to one of the components
above by a pronoun or a noun phrase instead of writing its name. For each one, quote
the referring expression, name the component it points to, and give the earlier
sentence that names that component together with the exact words that name it. If a
sentence carries no such reference to a listed component, return nothing for it. Be
conservative -- only include resolutions you are CERTAIN about.

{COREF_RULES}

Return JSON:
{{"resolutions": [{{"sentence": N_INTEGER, "reference": "the server", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

JSON only:"""

    def _resolve_references(self, sentences, components, name_to_id, sent_map):
        """One call over the document. The parsing and the checks are the head's.

        The body below is `s_linker92._resolve_references` with its batching loop,
        its target construction and its window table removed; every line that reads a
        reply is unchanged, so a resolution this arm keeps is one the head would have
        kept from the same reply.
        """
        comp_names = get_comp_names(components)
        all_coref = []
        coref_metadata: dict = {}
        self.llm.set_phase("phase_25_coreference")

        data = self._ask(
            self._prompt_coref_document(comp_names, sentences), timeout=600,
            label="Coref whole document", require_present="resolutions",
        )
        if not data:
            return all_coref, coref_metadata

        for res in data.get("resolutions", []):
            comp = res.get("component")
            snum = parse_snum(res.get("sentence"))
            if snum is None or snum not in sent_map:
                continue
            if not comp or comp not in name_to_id:
                continue
            ant_snum = parse_snum(res.get("antecedent_sentence"))
            if ant_snum is None:
                print(f"    Coref skip (no antecedent): S{snum} -> {comp}")
                continue
            ant_sent = sent_map.get(ant_snum)
            if not ant_sent:
                continue
            cid = name_to_id[comp]
            all_coref.append(SadSamLink(snum, cid, comp, source="coreference"))
            coref_metadata[(snum, cid)] = {
                "reference": res.get("reference", ""),
                "antecedent_sentence": ant_snum,
                "antecedent_text": res.get("antecedent_text", ""),
                "raw_resolution": res,
            }
        return all_coref, coref_metadata
