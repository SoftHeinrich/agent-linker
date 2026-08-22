"""S-Linker93 — both calls kept, the resolver narrowed to the sentences it is for.

The safe rung of the reading ladder. If merging the two proposal calls loses on
either model, this buys most of the same cost and most of the same narrative
without merging anything: the resolver keeps its own call, its own prompt and its
own batch size, and is simply not asked about sentences that write a name.

The population it is asked about today is every sentence in the document, and the
compaction round measured the consequence — **51.6% of the resolver's output is
for sentences that write the component's name**, which `LAYERED_COREF_RULES`
opens by saying is not a coreference link. That round recorded it as an open
question because fixing it inside the prompt meant *adding* a clause. It does not
need a clause. It is a fact about the case, and this branch's design law says
facts stay in code: the deterministic layer already knows which sentences write a
name, and `_states_a_name` is the predicate that says so.

Priced from the recorded runs before it was built:

* 52% of sentences write no name of any component, so the resolver's batches fall
  from 8.0 to 4.5 per project-run — **44% fewer resolver calls**, the resource
  bound `s_linker76` could only reduce by paying TP −7.0 for a wider batch.
* Of the 4.0 coreference links the pipeline keeps per project-run, 1.2 are on a
  sentence that does state a name, and **0.7 of those are gold**. That is the
  whole exposure, against a recorded null arm that moves TP by 4.8.

Nothing else changes: the prompt, the strict judge, `COREFERENCE_BATCH`, the
alias module, the scan and the two other linkers are inherited untouched.
"""

from __future__ import annotations

from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names, parse_snum
from llm_sad_sam.linkers.experimental.s_linker90 import SLinker90


class SLinker93(SLinker90):
    """The resolver is asked only about sentences that write no name."""

    _VARIANT_NAME = "s_linker93"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker93 (resolver narrowed to sentences that write no name)")

    def _nameless(self, sentences, components):
        """Sentences that write no name of any component, under the same relation
        the rest of the module reads names with."""
        return [
            s for s in sentences
            if not any(self._states_a_name(s.text, c.name) for c in components)
        ]

    def _resolve_references(self, sentences, components, name_to_id, sent_map):
        """Same resolver, same prompt, same batch — a smaller target population.

        The head's loop batches the sentences it is given *and* draws each case's
        context window from that same list, and ``_window`` selects by sentence
        number. Handing it the filtered list would therefore also filter the
        context — removing exactly the name-writing sentences a refer-back needs
        as its antecedent. So the loop is restated here with the two roles split:
        **batch over the targets, window over the whole document.** Every prompt,
        bound and check below is the head's; only the target set differs.
        """
        targets = self._nameless(sentences, components)
        print(f"    Coref targets: {len(targets)} of {len(sentences)} sentences "
              f"write no name")
        if not targets:
            return [], {}

        comp_names = get_comp_names(components)
        all_coref: list[SadSamLink] = []
        coref_metadata: dict = {}
        self.llm.set_phase("phase_25_coreference")

        for batch_num, batch in self._iter_batches(targets, self.COREFERENCE_BATCH):
            cases = []
            window_ids: set[int] = set()
            for i, sent in enumerate(batch, 1):
                window = [w.number for w in self._window(sent.number, sentences)]
                window_ids.update(window)
                cases.append({"case": i, "target": sent.number,
                              "text": sent.text, "context": window})
            sentence_table = [
                {"sentence": n, "text": sent_map[n].text}
                for n in sorted(window_ids) if n in sent_map
            ]
            data = self._ask(
                self._prompt_coref(comp_names, sentence_table, cases), timeout=600,
                label=f"Coref batch {batch_num}", require_present="resolutions",
            )
            if not data:
                continue
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
                if sent_map.get(ant_snum) is None:
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
