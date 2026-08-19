"""Can the hand-written gates be handed to the LLM without losing F2?

`gate_audit.py` located every code-driven decision and priced the three that are
linguistic heuristics rather than sanity or grounding checks:

    H1  the stated-name contract filter rejects 22 of 228 extractor proposals,
        9 of them gold;
    H2  the partial-name proposer is a prefix-and-uniqueness rule: 57 proposals,
        17 of them gold-reachable, and on one project 28 proposals contain a
        single gold pair;
    H3  the coreference antecedent test blocks 20 of 133 reported resolutions,
        7 of them gold -- the largest code-driven rejection in the pipeline.

Each arm hands that decision to the LLM and is scored on precision, recall, F1
and F2. F2 is the score to watch: these gates buy precision with recall, and a
recall-weighted measure is where that trade shows.

    --pilot contract    drop `_keep_stated_names`; the extractor proposes and the
                        two-pass judge decides.
    --pilot antecedent  drop `_antecedent_states_name`; the strict coreference
                        judge decides alone.
    --pilot proposer    replace `_name_word_candidates` with an LLM proposer:
                        one generic prompt asking which sentences refer to a
                        component by part of its name only. Same two-step judge
                        downstream, so only the proposal step changes.

Upstream stages come from a promoted run's checkpoints, as everywhere else here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from ab_stats import permutation_report
from design_pilots import (
    OUT, RUNS, SOURCE_RUN, collect, inputs_with_gold, new_linker, report,
    _coref_inputs, _prepare_extraction,
)
from design_audit import load_phase

from llm_sad_sam.core.data_types_v2 import CandidateLink
from llm_sad_sam.linkers.experimental.s_linker25 import SLinker25

# Generic: no benchmark vocabulary, no component names in the instruction text.
PARTIAL_PROPOSAL_RULES = (
    "A sentence refers to a component by PART of its name when it uses one word "
    "of a multi-word name on its own, or an inflected or shortened form of a "
    "name word, and does not state the whole name anywhere in that sentence. "
    "Report such a reference only when the word is used to talk about that "
    "component as part of the system, not as ordinary English and not inside a "
    "code-level or package path. Do not report a sentence that states the whole "
    "name: another step already handles those. Favor inclusion when the word "
    "plausibly stands for the component."
)


class NoContractFilter(SLinker25):
    """The extractor's proposals go straight to the judge."""

    def _keep_stated_names(self, candidates):
        return list(candidates)


class NoAntecedentGate(SLinker25):
    """Every resolution the model reports reaches the coreference judge."""

    def _antecedent_states_name(self, comp_name, ant_text):
        return True


class LLMPartialProposer(SLinker25):
    """An LLM proposes the partial-name references instead of a prefix rule."""

    PROPOSAL_BATCH = 50

    @staticmethod
    def _prompt_partial(comp_names, batch) -> str:
        return f"""Find sentences that refer to a component by part of its name only.

COMPONENTS: {', '.join(comp_names)}

{PARTIAL_PROPOSAL_RULES}

DOCUMENT:
{chr(10).join(f"S{s.number}: {s.text}" for s in batch)}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "the partial word as it appears in the sentence"}}]}}
JSON only:"""

    def _name_word_candidates(self, sentences, components):
        comp_names = [c.name for c in components]
        name_to_id = {c.name: c.id for c in components}
        sent_map = {s.number: s for s in sentences}
        candidates = {}
        self.llm.set_phase("phase_25_partial_propose")
        for _, batch in self._iter_batches(sentences, self.PROPOSAL_BATCH):
            data = self._ask(
                self._prompt_partial(comp_names, batch),
                timeout=240, label="Partial proposal", require_present="references")
            for ref in data.get("references", []):
                comp = ref.get("component")
                snum = ref.get("sentence")
                if isinstance(snum, str) and snum.lstrip("Ss").isdigit():
                    snum = int(snum.lstrip("Ss"))
                if not isinstance(snum, int) or comp not in name_to_id:
                    continue
                sent = sent_map.get(snum)
                if sent is None:
                    continue
                matched = str(ref.get("matched_text", ""))
                if not matched or matched.lower() not in sent.text.lower():
                    continue
                candidates[(snum, name_to_id[comp])] = CandidateLink(
                    snum, sent.text, comp, name_to_id[comp], matched,
                    source="partial_name_candidate")
        print(f"    LLM proposed {len(candidates)} partial-name candidates")
        return list(candidates.values())


# ── scoring: F2, not just TP and FP ──────────────────────────────────────────

def full_scorers(gold_by_project):
    total_gold = sum(len(g) for g in gold_by_project.values())

    def tp(pairs):
        return sum(1 for p, pair in pairs if tuple(pair) in gold_by_project[p])

    def fp(pairs):
        return len(pairs) - tp(pairs)

    def f_beta(pairs, beta):
        hits = tp(pairs)
        if not hits:
            return 0.0
        precision = hits / len(pairs)
        recall = hits / total_gold
        b2 = beta * beta
        return round(100 * (1 + b2) * precision * recall / (b2 * precision + recall), 1)

    return {"TP": tp, "FP": fp,
            "F1": lambda s: f_beta(s, 1.0),
            "F2": lambda s: f_beta(s, 2.0)}


# ── H1: the contract filter ──────────────────────────────────────────────────

def pilot_contract(inputs):
    print("\n### contract — drop the stated-name filter, let the judge decide")
    _prepare_extraction(inputs)
    gold = {n: inputs[n]["gold"] for n in inputs}

    def one(arm, run, name):
        item = inputs[name]
        cls = SLinker25 if arm == "A_filtered" else NoContractFilter
        linker = new_linker(cls, item["knowledge"]["doc_knowledge"],
                            item["knowledge"]["model_knowledge"])
        candidates = linker._keep_stated_names(item["extraction"])
        candidates = linker._add_spelling_variants(
            candidates, item["sentences"], item["components"])
        bundles = {
            (c.sentence_number, c.component_id):
                linker._build_evidence_bundle(c, item["sent_map"])
            for c in candidates
        }
        approved, _ = linker._validate_with_evidence(
            candidates, bundles, item["components"], item["sent_map"],
            p1_tag="gate_contract_p1", p2_tag="gate_contract_p2",
            stage_label="full_name")
        return {(name, (c.sentence_number, c.component_id)) for c in approved}

    arms = {}
    for arm in ("A_filtered", "B_unfiltered"):
        arms[arm] = collect(arm, lambda run, name, a=arm: one(a, run, name), inputs)
        print(f"  {arm}: {[len(s) for s in arms[arm]]}")
    report("gate_contract_filter", permutation_report(
        arms, full_scorers(gold), title="contract — lexical admission removed"))


# ── H3: the antecedent gate ──────────────────────────────────────────────────

def pilot_antecedent(inputs):
    print("\n### antecedent — drop the structural gate, judge alone decides")
    prepared = _coref_inputs(inputs)
    gold = {n: inputs[n]["gold"] for n in inputs}

    def one(arm, run, name):
        item = inputs[name]
        cls = SLinker25 if arm == "A_gated" else NoAntecedentGate
        linker = new_linker(cls, item["knowledge"]["doc_knowledge"],
                            item["knowledge"]["model_knowledge"])
        resolved, _ = linker._resolve_references(
            item["sentences"], item["components"], item["name_to_id"],
            item["sent_map"])
        fresh = linker._unlinked(resolved, prepared[name]["prior"])
        final = set(prepared[name]["prior"])
        if fresh:
            approved, _ = linker._validate_coref_links(
                fresh, item["sent_map"], item["components"])
            final |= {(l.sentence_number, l.component_id) for l in approved}
        return {(name, pair) for pair in final}

    arms = {}
    for arm in ("A_gated", "B_ungated"):
        arms[arm] = collect(arm, lambda run, name, a=arm: one(a, run, name), inputs)
        print(f"  {arm}: {[len(s) for s in arms[arm]]}")
    report("gate_antecedent", permutation_report(
        arms, full_scorers(gold), title="antecedent — structural gate removed"))


# ── H2: the partial-name proposer ────────────────────────────────────────────

def pilot_proposer(inputs):
    print("\n### proposer — LLM proposes partial-name references, not a prefix rule")
    # The partial-name linker runs second, so its prior is the full-name links
    # alone. Subtracting the promoted run's partial-name links as well would
    # remove exactly the pairs this stage is supposed to find.
    prior = {
        name: {(l.sentence_number, l.component_id)
               for l in load_phase(SOURCE_RUN, name, "linker_full_name")["links"]}
        for name in inputs
    }
    gold = {n: inputs[n]["gold"] for n in inputs}

    def one(arm, run, name):
        item = inputs[name]
        cls = SLinker25 if arm == "A_rule" else LLMPartialProposer
        linker = new_linker(cls, item["knowledge"]["doc_knowledge"],
                            item["knowledge"]["model_knowledge"])
        proposals = linker._unlinked(
            linker._name_word_candidates(item["sentences"], item["components"]),
            prior[name])
        if not proposals:
            return set()
        approved, _ = linker._judge_partial_names(proposals, item["sentences"])
        return {(name, (c.sentence_number, c.component_id)) for c in approved}

    arms = {}
    for arm in ("A_rule", "B_llm"):
        arms[arm] = collect(arm, lambda run, name, a=arm: one(a, run, name), inputs)
        print(f"  {arm}: {[len(s) for s in arms[arm]]}")
    # Scored on the partial-name stage alone: precision here is precision of the
    # links this linker contributes, not of the pipeline.
    report("gate_partial_proposer", permutation_report(
        arms, full_scorers(gold), title="proposer — rule vs LLM proposal"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", nargs="+", required=True,
                        choices=["contract", "antecedent", "proposer"])
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY unset (map OAI_KEY into it inline)")
    started = time.time()
    inputs = inputs_with_gold()
    if "contract" in args.pilot:
        pilot_contract(inputs)
    if "antecedent" in args.pilot:
        pilot_antecedent(inputs)
    if "proposer" in args.pilot:
        pilot_proposer(inputs)
    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
