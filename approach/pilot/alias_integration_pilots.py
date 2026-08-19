"""Can the alias module be folded into the extractor?

The audit (`alias_integration_audit.py`) established three things:

  * the alias table is load-bearing -- 29 full-name links across the five
    projects are admitted only via an alias, 23 of them gold -- so it cannot be
    dropped;
  * a naive projection of it from the extractor's `matched_text` field recovers
    only 41% of the discovered aliases and adds 28 spurious surfaces (pronouns,
    phrases containing the name, morphological variants), so the field is a span,
    not a name;
  * of the six consumers of the table, exactly **one** runs before extraction --
    the `KNOWN ALIASES` line of the extraction prompt. Every other consumer runs
    after it and could read a table the extractor itself produced.

That last point is what makes a merge structurally possible. Three arms:

    --pilot hint      does the extraction prompt need `KNOWN ALIASES` at all? If
                      not, the ordering constraint disappears and the merge is
                      free. Downstream consumers keep the discovered table in
                      both arms, so this isolates the prompt hint.
    --pilot judge     is the alias judge (a second call over the proposed
                      mappings) worth its place? Alias extraction is computed
                      once per project and shared, so only the judge varies.
    --pilot unified   the merge itself: one prompt per sentence batch returning
                      both references and any alias the batch establishes, with
                      the table accumulated across batches and fed forward. No
                      document-wide alias pass, no alias judge -- two calls and
                      two prompts fewer, and one stage fewer in the paper.

Scored on TP / FP / F1 / F2, five runs per side, permutation-tested.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from ab_stats import permutation_report
from design_pilots import (
    SOURCE_RUN, cached, collect, inputs_with_gold, new_linker, report, OUT,
)
from gate_pilots import full_scorers

from llm_sad_sam.core.data_types_v2 import CandidateLink, DocumentKnowledge
from llm_sad_sam.linkers.experimental.s_linker25 import (
    SLinker25, ALIAS_EXCLUSION_RULES, DOC_KNOWLEDGE_EXTRACTION_RULES,
    ENTITY_EXTRACTION_RULES,
)
from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names


class NoAliasHint(SLinker25):
    """Extraction prompt without the KNOWN ALIASES line."""

    def _extract_named_mentions(self, sentences, components, name_to_id, sent_map):
        comp_names = get_comp_names(components)
        candidates = self._run_extraction_pass(
            sentences, comp_names, [], name_to_id, sent_map,
            phase_tag="phase_25_full_name_extract")
        print(f"    Extracted (no alias hint): {len(candidates)}")
        return candidates


class UnifiedExtractor(SLinker25):
    """One prompt per batch: references, plus the aliases the batch establishes.

    The table accumulates across batches and is fed forward, so a definition in
    an early batch is available to later ones -- which is what the separate
    document-wide pass bought. Batches therefore run in order, not in parallel;
    the production extractor is already sequential over batches.
    """

    def _prompt_unified(self, comp_names, mappings, batch) -> str:
        return f"""Extract ALL references to components from this document, and report any alternative name the document uses for a component.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings)}' if mappings else ''}

{ENTITY_EXTRACTION_RULES}

{DOC_KNOWLEDGE_EXTRACTION_RULES}

{ALIAS_EXCLUSION_RULES}

DOCUMENT:
{chr(10).join(f"S{s.number}: {s.text}" for s in batch)}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence"}}],
 "aliases":    [{{"term": "alternative name", "component": "FullComponent"}}]}}
JSON only:"""

    def _learn_document_knowledge(self, sentences, components):
        """No separate pass: the table is built during extraction."""
        return DocumentKnowledge()

    def _extract_named_mentions(self, sentences, components, name_to_id, sent_map):
        comp_names = get_comp_names(components)
        table = dict(self.doc_knowledge.aliases) if self.doc_knowledge else {}
        candidates = {}
        self.llm.set_phase("phase_25_unified_extract")
        for batch_num, batch in self._iter_batches(sentences, self.EXTRACTION_BATCH):
            mappings = [f"{t}={c}" for t, c in table.items()]
            data = self._ask(
                self._prompt_unified(comp_names, mappings, batch),
                timeout=240, label=f"unified batch {batch_num}",
                require_present="references")
            for ref in data.get("references", []):
                cname = ref.get("component")
                snum = self._snum(ref.get("sentence"))
                if snum is None or cname not in name_to_id:
                    continue
                sent = sent_map.get(snum)
                if sent is None:
                    continue
                matched = str(ref.get("matched_text", ""))
                if matched and matched.lower() not in sent.text.lower():
                    continue
                candidates.setdefault((snum, name_to_id[cname]), CandidateLink(
                    snum, sent.text, cname, name_to_id[cname], matched,
                    source="full_name"))
            for item in data.get("aliases", []):
                if not isinstance(item, dict):
                    continue
                term, comp = item.get("term"), item.get("component")
                if term and comp in name_to_id and term not in table:
                    table[term] = comp
        self.doc_knowledge = DocumentKnowledge()
        self.doc_knowledge.aliases.update(table)
        print(f"    Unified: {len(candidates)} references, "
              f"{len(table)} aliases {sorted(table)}")
        return candidates

    @staticmethod
    def _snum(value):
        if isinstance(value, str) and value.lstrip("Ss").isdigit():
            return int(value.lstrip("Ss"))
        return value if isinstance(value, int) else None


def _full_name_stage(linker, item, name):
    candidates = linker._keep_stated_names(list(
        linker._extract_named_mentions(
            item["sentences"], item["components"], item["name_to_id"],
            item["sent_map"]).values()))
    candidates = linker._add_spelling_variants(
        candidates, item["sentences"], item["components"])
    bundles = {
        (c.sentence_number, c.component_id):
            linker._build_evidence_bundle(c, item["sent_map"])
        for c in candidates
    }
    approved, _ = linker._validate_with_evidence(
        candidates, bundles, item["components"], item["sent_map"],
        p1_tag="alias_p1", p2_tag="alias_p2", stage_label="full_name")
    return {(name, (c.sentence_number, c.component_id)) for c in approved}


# ── arms ─────────────────────────────────────────────────────────────────────

def pilot_hint(inputs):
    print("\n### hint — extraction with and without the KNOWN ALIASES line")
    gold = {n: inputs[n]["gold"] for n in inputs}

    def one(arm, run, name):
        item = inputs[name]
        cls = SLinker25 if arm == "A_with_hint" else NoAliasHint
        linker = new_linker(cls, item["knowledge"]["doc_knowledge"],
                            item["knowledge"]["model_knowledge"])
        return _full_name_stage(linker, item, name)

    arms = {}
    for arm in ("A_with_hint", "B_no_hint"):
        arms[arm] = collect(arm, lambda run, name, a=arm: one(a, run, name), inputs)
        print(f"  {arm}: {[len(s) for s in arms[arm]]}")
    report("alias_hint", permutation_report(
        arms, full_scorers(gold), title="hint — KNOWN ALIASES removed from extraction"))


def pilot_judge(inputs):
    print("\n### judge — the alias judge on and off")
    gold = {n: inputs[n]["gold"] for n in inputs}

    proposed = {}
    for name, item in inputs.items():
        linker = new_linker()

        def build(linker=linker, item=item):
            comp_names = [c.name for c in item["components"]]
            data = linker._ask(
                linker._prompt_doc_knowledge_extract(
                    comp_names, [s.text for s in item["sentences"]]),
                timeout=300, label="alias extract")
            table = {}
            for key in ("abbreviations", "synonyms"):
                for rec in data.get(key, []) or []:
                    if isinstance(rec, dict) and rec.get("component") in comp_names:
                        if rec.get("term"):
                            table[rec["term"]] = rec["component"]
            return table

        proposed[name] = cached(OUT / f"alias_proposed_{name}.pkl", build)
        print(f"  {name:14s} proposed {len(proposed[name])} aliases")

    def one(arm, run, name):
        item = inputs[name]
        linker = new_linker()
        table = dict(proposed[name])
        if arm == "A_judged" and table:
            comp_names = [c.name for c in item["components"]]
            data = linker._ask(
                linker._prompt_doc_knowledge_judge(
                    comp_names, [f"'{k}' -> {v}" for k, v in table.items()]),
                timeout=120, label="alias judge", require="approved")
            approved = set(data.get("approved", [])) if data else set(table)
            table = {k: v for k, v in table.items() if k in approved}
        knowledge = DocumentKnowledge()
        knowledge.aliases.update(table)
        linker.doc_knowledge = knowledge
        return _full_name_stage(linker, item, name)

    arms = {}
    for arm in ("A_judged", "B_unjudged"):
        arms[arm] = collect(arm, lambda run, name, a=arm: one(a, run, name), inputs)
        print(f"  {arm}: {[len(s) for s in arms[arm]]}")
    report("alias_judge", permutation_report(
        arms, full_scorers(gold), title="judge — alias judge removed"))


def pilot_unified(inputs):
    print("\n### unified — one extraction prompt that also reports aliases")
    print("     arm A is the production two-stage design, recomputed so both")
    print("     arms pay for their own alias discovery")
    gold = {n: inputs[n]["gold"] for n in inputs}

    def one(arm, run, name):
        item = inputs[name]
        if arm == "A_two_stage":
            linker = new_linker()
            linker.doc_knowledge = linker._learn_document_knowledge(
                item["sentences"], item["components"])
        else:
            linker = new_linker(UnifiedExtractor)
            linker.doc_knowledge = DocumentKnowledge()
        return _full_name_stage(linker, item, name)

    arms = {}
    for arm in ("A_two_stage", "B_unified"):
        arms[arm] = collect(arm, lambda run, name, a=arm: one(a, run, name), inputs)
        print(f"  {arm}: {[len(s) for s in arms[arm]]}")
    report("alias_unified", permutation_report(
        arms, full_scorers(gold), title="unified — alias discovery folded into extraction"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", nargs="+", required=True,
                        choices=["hint", "judge", "unified"])
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY unset (map OAI_KEY into it inline)")
    started = time.time()
    inputs = inputs_with_gold()
    if "hint" in args.pilot:
        pilot_hint(inputs)
    if "judge" in args.pilot:
        pilot_judge(inputs)
    if "unified" in args.pilot:
        pilot_unified(inputs)
    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
