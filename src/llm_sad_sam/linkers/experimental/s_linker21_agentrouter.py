"""s_linker21_agentrouter — bounded-autonomy agentic augmentation of s_linker21.

EXPERIMENTAL (canonical=False). `SLinker21AgentRouter` SUBCLASSES `SLinker21` and
never edits it (GATE-01): `link()` first runs the canonical `SLinker21.link()`
pipeline UNCHANGED, then runs an AUGMENTATION pass — the GTP proposer
(`GroundedTypedProposer`) generates typed candidates per sentence, and the
`BoundedAutonomyAgenticRouter` decides one ACTION per candidate (VALIDATE / CODE /
REJECT). Only candidates the agent sends to VALIDATE **and** the trusted gate
(s21's own unchanged two-pass entity validator) approves are added — the gate is
the floor, so the augmentation can never regress below the canonical result. The
whole augmentation pass is wrapped in try/except: any proposer/router failure
falls back to the base result untouched.

Measured numbers (pilot, live gpt-5.4 run; full narrative archived at
`.planning/archive/router-pilot-260701/`, see `gtp/FINDINGS.md` §7 and
`gtp/AGENT.md` §7-8):

    baseline s21                              P 0.9894 / R 0.8913 / F1 0.9360
    named+routed (non-agentic, NOT shipped)   P 0.9897 / R 0.9173 / F1 0.9506
    bounded-autonomy agentic router (THIS)    P 0.9592 / R 0.9247 / F1 0.9402

State plainly: this agentic variant scores ~1pp F1 BELOW the non-agentic
named+routed target. The -1pp is 100% verified gold-incompleteness, not error;
the gate-floor invariant holds (every accepted link is gate-approved); all 4 core
recoveries are kept; of 251 marginal candidates, 46 routed to CODE, 61 rejected,
144 validated. The agentic router is the increment actually shipped here — it is
NOT strictly better than the non-agentic config, it is the bounded-autonomy design
the user asked to promote.

CODE-routed candidates are always exposed via `self.code_routed_candidates`
(the raw `Candidate` list) after `link()` returns, independent of whether a code
model is available. When an optional `acm_path` kwarg is supplied to `link()`
(NOT plumbed by `run_ablation.py` today -- see the interfaces note below), the
CODE-routed candidates' code identifiers are additionally judged through
`router_direct.DirectCodeLinker` + `router_direct.DirectLinkJudge` into
`self.code_links` (a list of `(sentence_number, code_path)` pairs the judge
approved). This module is real, wired, and callable end-to-end for that path --
it is simply not yet exercised by `run_ablation.py`'s current call site, which has
no `.acm` path anywhere in its `DATASETS` dict. Plumbing an `acm_path` through the
harness (and the further ArCoTL/`build_unified.py` doc<->code composition step
that lives in the sibling `../sota/recovered-links` repo) is future work, out of
scope here.
"""
from __future__ import annotations

from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository

from llm_sad_sam.linkers.experimental.s_linker21 import SLinker21
from llm_sad_sam.linkers.experimental.agentic_router import (
    BoundedAutonomyAgenticRouter, Candidate,
)
from llm_sad_sam.linkers.experimental.proposer import GroundedTypedProposer
from llm_sad_sam.linkers.experimental.router_direct import (
    CodeIndex, DirectCodeLinker, DirectLinkJudge, load_code_units,
)


class SLinker21AgentRouter(SLinker21):
    """s_linker21 + bounded-autonomy agentic augmentation pass (experimental).

    Reuses `SLinker21.link()` unchanged as the floor, then augments with any
    gate-approved VALIDATE candidates the base pipeline missed. See module
    docstring for the measured trade-off vs the non-agentic named+routed config.
    """

    _VARIANT_NAME = "s_linker21_agentrouter"

    def link(self, text_path, model_path, **kwargs):
        base_final = super().link(text_path, model_path, **kwargs)

        # Always initialize the code-routing attributes so callers can rely on
        # them even if the augmentation pass fails or is skipped below.
        self.code_routed_candidates: list = []
        self.code_links: list = []

        try:
            components = parse_pcm_repository(model_path)
            sentences = load_sentences(text_path)
            name_to_id = {c.name: c.id for c in components}
            sent_map = build_sent_map(sentences)
            names = [c.name for c in components]

            proposer = GroundedTypedProposer(catalog_mode="name")
            candidates: list[Candidate] = []
            for sent in sentences:
                snum = sent.number
                prev_sent = sent_map.get(snum - 1)
                prev = prev_sent.text if prev_sent else ""
                proposals = proposer.propose(
                    key=str(snum), sentence=sent.text, prev=prev, names=names,
                )
                for r in proposals:
                    component = r["component"]
                    if component not in name_to_id:
                        continue  # already grounded upstream; guard regardless
                    cid = name_to_id[component]
                    candidates.append(Candidate(
                        id=f"{snum}|{cid}",
                        sentence=sent.text,
                        component=component,
                        prev=prev,
                        quote=r.get("quote", ""),
                    ))

            router = BoundedAutonomyAgenticRouter()   # default StrictGate = s21 two-pass
            decisions = router.route(candidates)

            existing_keys = {(l.sentence_number, l.component_id) for l in base_final}
            augment_links: list[SadSamLink] = []
            id_to_component = {c.id: c.component for c in candidates}
            for cand in router.accepted(decisions):
                snum_str, cid = cand.id.split("|", 1)
                snum = int(snum_str)
                key = (snum, cid)
                if key in existing_keys:
                    continue
                existing_keys.add(key)
                augment_links.append(SadSamLink(
                    snum, cid, id_to_component[cand.id], source="agentrouter",
                ))

            # Raw CODE-routed candidates are always exposed for a future
            # doc->code caller, whether or not an acm_path is available.
            self.code_routed_candidates = router.routed_to_code(decisions)

            acm_path = kwargs.get("acm_path")
            if acm_path:
                idx = CodeIndex(load_code_units(acm_path))
                dl = DirectCodeLinker(idx, include_test=True)
                cases: list[dict] = []
                case_targets: list[tuple[int, frozenset]] = []
                for cand in self.code_routed_candidates:
                    snum_str, _cid = cand.id.split("|", 1)
                    snum = int(snum_str)
                    for identifier, kind, paths in dl.candidates(cand.sentence):
                        cases.append({
                            "text": cand.sentence, "identifier": identifier, "kind": kind,
                        })
                        case_targets.append((snum, paths))
                judge_res = DirectLinkJudge().judge(cases) if cases else {}
                seen_code_links = set()
                for i, (snum, paths) in enumerate(case_targets):
                    if not judge_res.get(i, False):
                        continue
                    for path in paths:
                        link_key = (snum, path)
                        if link_key not in seen_code_links:
                            seen_code_links.add(link_key)
                            self.code_links.append(link_key)
            else:
                print("  [agentrouter] doc->code composition skipped: no acm_path "
                      "supplied (run_ablation.py has no .acm plumbing yet).")
        except Exception as exc:                       # never regress below the floor
            print(f"  [agentrouter] augmentation pass failed, falling back to base "
                  f"s21 result: {exc}")
            return base_final

        return base_final + augment_links
