"""S-Linker110-OneCall — the whole workflow replaced by one linking call.

The **total floor** for RQ4 (D3): the model is given the document, the component list
and the project's alias table, and returns the final link set. Nothing else runs. No
scan, no window, no evidence bundle, no antecedent shortlist, no judge, no union.

This is the arm that prices the workflow itself. Every other ablation on this branch
removes one stage and keeps the rest; this removes the arrangement and keeps only the
model.

**It is not `s_linker27`, and the ledger has no equivalent.** s27 merged the two
*reading* stages into one call and then ran s25's three linkers, both scans, all three
judges and the subtraction rule unchanged -- its own docstring says so ("Everything
after the reading is s_linker25 unchanged"). Its macro F1 of 91.70 prices a merged
reading, not an absent workflow. **D3 has never been measured here.**

**Fair-baseline decisions, all three taken deliberately.**

*Same rules.* The prompt carries the head's own authored rubrics verbatim --
`LAYERED_ENTITY_RULES` (the lenient gate's standard), `LAYERED_COREF_RULES` (the strict
one's), `QUALIFIED_CLAUSE` and `STRICTER_CLAUSE`. The finetune and prompt rounds
measured this text as load-bearing on its own (`s_linker51`: removing nine constants
cost 2.4 macro F1), so a floor given a bare task line would remove structure and
guidance together and price neither. Here the only thing removed is the arrangement.

*Knowledge kept.* `link()` still runs `_learn_document_knowledge`, and the discovered
aliases go into the prompt. The knowledge module is RQ3's ablation, not this one; taking
it away here would move two things at once. "One call" therefore means one *linking*
call -- the knowledge stage is an LLM stage, not code, and it is named in the phase log.

*No quote demanded.* The head requires every judge to quote the sentence before ruling,
and that demand alone is worth 35.2 TP (`results/s25_design_pilots/`). This arm does not
ask for it. **That is a second thing removed alongside the arrangement**, and any loss
here is therefore partly attributable to it; the arm is an upper bound on what the
workflow is worth, not a point estimate. Stated here rather than discovered later.

**The length confound, also stated.** `s_linker27` measured a whole-document call and
found accuracy tracks document length -- jabref (13 sentences) 100.0, teammates (198)
84.1. This arm sends every document whole, so the per-project numbers must be read
before the macro, and a macro loss is partly a length effect on the two long documents.

**What the phase log will and will not contain.** `knowledge` and `final` are written as
the head writes them, so `score_runs.py` and the links CSVs work unchanged. There are no
`linker_*` phases, so the RQ3/RQ4 engines (`evaluation/mini-src/rq34.py`) cannot read it -- by
construction, since it has no stages to attribute. Score it end to end.

**Measurement owed:** level 4, paired with `s_linker110` in every invocation, three runs
a model on both backends. Unmeasured at the time of writing.
"""

from __future__ import annotations

import json
import time

from llm_sad_sam.core.data_types_v2 import DocumentKnowledge, SadSamLink
from llm_sad_sam.core.document_loader_v2 import load_sentences
from llm_sad_sam.linkers.experimental.s_linker92 import (
    LAYERED_COREF_RULES,
    LAYERED_ENTITY_RULES,
    QUALIFIED_CLAUSE,
    STRICTER_CLAUSE,
    build_sent_map,
    get_comp_names,
    parse_snum,
)
from llm_sad_sam.linkers.experimental.s_linker110 import SLinker110
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository


class SLinker110OneCall(SLinker110):
    """Document and components in, final links out. One call, no stages."""

    _VARIANT_NAME = "s_linker110_onecall"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker110OneCall (one linking call; no scan, no judge, no union)")

    def _prompt_one_call(self, comp_names, aliases, sentences) -> str:
        """The head's rubrics, the document, the components, the aliases.

        The four rubrics are rendered verbatim from `s_linker92` so the arm cannot
        drift from the head's guidance. They are stated here as one standard because
        this arm has no stages to distribute them across.
        """
        document = [{"sentence": s.number, "text": s.text} for s in sentences]
        alias_line = (", ".join(f"{term}={component}"
                                for term, component in aliases.items())
                      or "none")
        return f"""Link sentences of an architecture document to the components of its architecture model.

COMPONENTS: {', '.join(comp_names)}

KNOWN ALIASES: {alias_line}

DOCUMENT
{json.dumps(document)}

Report every sentence that refers to a component, whether it writes the component's
whole name, only one word of that name, or refers back to it with a pronoun or a noun
phrase. Judge each one against the standard below and report only the links that meet it.

{LAYERED_ENTITY_RULES}

{QUALIFIED_CLAUSE}

{STRICTER_CLAUSE}

{LAYERED_COREF_RULES}

Return JSON:
{{"links": [{{"sentence": N_INTEGER, "component": "Name"}}]}}

JSON only:"""

    def link(self, text_path, model_path, **_kwargs):
        """`s_linker92.link` with the three linkers replaced by one call.

        The knowledge stage, the phase log, the summary log and the `final` checkpoint
        are the head's, so the runner writes the same links CSV and `score_runs.py`
        reads this arm exactly as it reads any other.
        """
        self._phase_log = []
        self._llm_calls.clear()
        self._phase_metrics = {}
        started = time.time()

        components = parse_pcm_repository(model_path)
        sentences = load_sentences(text_path)
        name_to_id = {component.name: component.id for component in components}
        sent_map = build_sent_map(sentences)
        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        print("\n[Knowledge] Document aliases")
        self.doc_knowledge = (
            DocumentKnowledge() if self.no_knowledge
            else self._learn_document_knowledge(sentences, components)
        )
        self._save_phase(text_path, "knowledge",
                         {"doc_knowledge": self.doc_knowledge})

        print("\n[OneCall] whole document, no stages")
        self.llm.set_phase("phase_25_one_call")
        aliases = self.doc_knowledge.aliases if self.doc_knowledge else {}
        data = self._ask(
            self._prompt_one_call(get_comp_names(components), aliases, sentences),
            timeout=600, label="One call", require_present="links",
        )

        current: list[SadSamLink] = []
        seen = set()
        for entry in (data or {}).get("links", []):
            comp = entry.get("component")
            snum = parse_snum(entry.get("sentence"))
            # The head's own validity checks: a returned number must name a real
            # sentence and a returned name must be in the catalog. These reject a
            # reply, they do not inform one, so keeping them is parsing and not help.
            if snum is None or snum not in sent_map:
                continue
            if not comp or comp not in name_to_id:
                continue
            cid = name_to_id[comp]
            if (snum, cid) in seen:
                continue
            seen.add((snum, cid))
            current.append(SadSamLink(snum, cid, comp, source="one_call"))

        self.workflow = []
        self._phase_metrics = {}
        self._log(
            "s25_summary",
            {"components": len(components), "sentences": len(sentences)},
            {
                "workflow": [],
                "final": len(current),
                "elapsed_s": round(time.time() - started, 2),
                "llm_calls": len(self._llm_calls),
                "phase_metrics": {},
            },
            current,
        )
        self._save_phase(text_path, "final", {
            "final": current,
            "workflow": [],
            "elapsed_s": round(time.time() - started, 2),
        })
        self._save_log(text_path)
        print(f"\nFinal: {len(current)} links "
              f"({time.time() - started:.1f}s, {len(self._llm_calls)} LLM calls)")
        return current
