"""Every remaining s25 decision that has never been ablated, in one place.

Off checkpoints, as everywhere here: upstream stages come from a promoted run
(`AB_SOURCE_RUN`), exactly one thing changes per arm, N runs per side, permutation
test, scored on TP / FP / F1 / F2.

What was already measured elsewhere and is NOT repeated here: the subtraction
rule, the second extraction sample, alias scope, the ambiguity map, the judge's
quote request and its verification, a second coreference pass, judge batch size,
the evidence window, the bundle's rationale line and anchor primitive,
`antecedent_via_alias`, the stated-name contract filter, the coreference
antecedent gate, and the LLM partial-name proposer. See
`results/s25_{design,complexity,gate,micro}_*`.

What this file adds:

  FULL-NAME STAGE (shared control, cached extraction)
    variants      drop `_add_spelling_variants` -- ~50 lines, a CamelCase-splitting
                  signature regex, a nested span scan, a separator test and a
                  unique-owner test, for 2 gold links on one project.
    p2            drop the uniqueness pass, keep relevance only. The two-pass
                  design was never priced from this side.
    prev          drop the preceding sentence from the bundle.
    anchors       drop the anchor sentences from the bundle.
    span          drop the matched span from the bundle.
    source        drop the source tag from the bundle.

  PARTIAL-NAME STAGE (prior = the promoted run's full-name links)
    denotation    skip the target-blind step; judge identity directly. The audit
                  says it rejects 38 of 57 proposals at a cost of 1 gold link,
                  but it has never been run as an arm.
    exactword     require a sentence word to equal a name word instead of
                  beginning with one (the "morphology approximation").
    multiowner    drop the unique-owner test; propose every owner.
    qualified     drop the qualified-identifier skip.
    wholename     drop the "sentence states no whole name" condition.

Usage:
    AB_RUNS=5 ... ../.venv/bin/python -u pilot/ablate_all.py --arms all
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from ab_stats import permutation_report
from design_audit import load_phase
from design_pilots import (
    P1Only, SOURCE_RUN, collect, inputs_with_gold, new_linker, report,
    _judge_stage, _prepare_extraction,
)
from gate_pilots import full_scorers

from llm_sad_sam.core.data_types_v2 import CandidateLink
from llm_sad_sam.linkers.experimental.s_linker25 import SLinker25

WORD = re.compile(r"[A-Za-z]+[A-Za-z0-9]*|\d+")


# ── full-name stage variants ─────────────────────────────────────────────────

class NoSpellingVariants(SLinker25):
    """No deterministic spelling-variant proposer."""

    def _add_spelling_variants(self, candidates, sentences, components):
        return list(candidates)


def _bundle_line_variant(drop):
    """A subclass whose evidence line omits one field."""

    class Variant(SLinker25):
        def _format_evidence(self, bundle) -> str:
            parts = []
            if drop != "source":
                parts.append(f"source={bundle.source}")
            if drop != "span":
                parts.append(f'span="{bundle.matched_span}"')
            parts.append(f"mention={bundle.mention_type}")
            lines = [f"  Evidence: {', '.join(parts)}"]
            if drop != "prev" and bundle.preceding_text:
                lines.append(f'  [prev: "{bundle.preceding_text}"]')
            if drop != "anchors" and bundle.anchor_sentences:
                lines.append("  Anchors (confirmed refs):")
                for anchor in bundle.anchor_sentences:
                    lines.append(f"    {anchor}")
            return "\n".join(lines)

    Variant.__name__ = f"Bundle_no_{drop}"
    return Variant


# ── partial-name stage variants ──────────────────────────────────────────────

class NoDenotation(SLinker25):
    """Skip the target-blind denotation step; go straight to grounded identity."""

    def _judge_partial_names(self, candidates, sentences):
        approved, decisions = self._review_identity(candidates, sentences)
        return approved, decisions


class _ProposerVariant(SLinker25):
    """Base for arms that change one condition of `_name_word_candidates`.

    Flags, all default to the production behaviour:
      EXACT_WORD       require equality instead of a prefix match
      UNIQUE_OWNER     require exactly one owning component
      SKIP_QUALIFIED   skip spans inside a dotted or joined identifier
      SKIP_WHOLE_NAME  skip sentences that state a whole name
    """

    EXACT_WORD = False
    UNIQUE_OWNER = True
    SKIP_QUALIFIED = True
    SKIP_WHOLE_NAME = True

    def _name_word_candidates(self, sentences, components):
        words_by_component = {
            component.id: [w.casefold() for w in WORD.findall(component.name)]
            for component in components
        }
        names_by_component = self._names_by_component()
        candidates = {}
        for sentence in sentences:
            for match in WORD.finditer(sentence.text):
                if self.SKIP_QUALIFIED and self._inside_qualified_identifier(
                        sentence.text, match.start(), match.end()):
                    continue
                surface = match.group(0).casefold()
                owners = [
                    component for component in components
                    if any(surface == word if self.EXACT_WORD
                           else surface.startswith(word)
                           for word in words_by_component[component.id])
                ]
                if self.UNIQUE_OWNER and len(owners) != 1:
                    continue
                for component in owners:
                    names = [component.name,
                             *names_by_component.get(component.name, [])]
                    if self.SKIP_WHOLE_NAME and any(
                            self._find_exact_form(sentence.text, n) for n in names):
                        continue
                    candidates[(sentence.number, component.id)] = CandidateLink(
                        sentence.number, sentence.text, component.name,
                        component.id, match.group(0),
                        source="partial_name_candidate")
        return list(candidates.values())


class ExactWord(_ProposerVariant):
    EXACT_WORD = True


class MultiOwner(_ProposerVariant):
    UNIQUE_OWNER = False


class NoQualifiedSkip(_ProposerVariant):
    SKIP_QUALIFIED = False


class NoWholeNameSkip(_ProposerVariant):
    SKIP_WHOLE_NAME = False


# ── stage runners ────────────────────────────────────────────────────────────

def _full_name_arms(inputs, wanted):
    print("\n### full-name stage — one control, one arm per dropped element")
    _prepare_extraction(inputs)
    gold = {n: inputs[n]["gold"] for n in inputs}

    def stage(cls, tag):
        return collect("x", lambda run, name: _judge_stage(
            cls, inputs[name], name,
            inputs[name]["knowledge"]["doc_knowledge"],
            inputs[name]["knowledge"]["model_knowledge"], tag), inputs)

    control = stage(SLinker25, "abl_control")
    print(f"  A_current: {[len(s) for s in control]}")

    plan = {
        "variants": (NoSpellingVariants, "no spelling-variant proposer"),
        "p2": (P1Only, "relevance pass only"),
        "prev": (_bundle_line_variant("prev"), "no preceding sentence"),
        "anchors": (_bundle_line_variant("anchors"), "no anchor sentences"),
        "span": (_bundle_line_variant("span"), "no matched span"),
        "source": (_bundle_line_variant("source"), "no source tag"),
    }
    for key, (cls, label) in plan.items():
        if key not in wanted:
            continue
        arm = stage(cls, f"abl_{key}")
        print(f"  B_{key}: {[len(s) for s in arm]}")
        report(f"ablate_full_name_{key}", permutation_report(
            {"A_current": control, f"B_{key}": arm}, full_scorers(gold),
            title=f"full-name: {label}"))


def _partial_arms(inputs, wanted):
    print("\n### partial-name stage — prior is the promoted run's full-name links")
    prior = {
        name: {(l.sentence_number, l.component_id)
               for l in load_phase(SOURCE_RUN, name, "linker_full_name")["links"]}
        for name in inputs
    }
    gold = {n: inputs[n]["gold"] for n in inputs}

    def one(cls, run, name):
        item = inputs[name]
        linker = new_linker(cls, item["knowledge"]["doc_knowledge"],
                            item["knowledge"]["model_knowledge"])
        proposals = linker._unlinked(
            linker._name_word_candidates(item["sentences"], item["components"]),
            prior[name])
        final = set(prior[name])
        if proposals:
            approved, _ = linker._judge_partial_names(proposals, item["sentences"])
            final |= {(c.sentence_number, c.component_id) for c in approved}
        return {(name, pair) for pair in final}

    control = collect("x", lambda run, name: one(SLinker25, run, name), inputs)
    print(f"  A_current: {[len(s) for s in control]}")

    plan = {
        "denotation": (NoDenotation, "no target-blind denotation step"),
        "exactword": (ExactWord, "exact word match, no prefix rule"),
        "multiowner": (MultiOwner, "no unique-owner requirement"),
        "qualified": (NoQualifiedSkip, "no qualified-identifier skip"),
        "wholename": (NoWholeNameSkip, "no whole-name exclusion"),
    }
    for key, (cls, label) in plan.items():
        if key not in wanted:
            continue
        arm = collect("y", lambda run, name, c=cls: one(c, run, name), inputs)
        print(f"  B_{key}: {[len(s) for s in arm]}")
        report(f"ablate_partial_{key}", permutation_report(
            {"A_current": control, f"B_{key}": arm}, full_scorers(gold),
            title=f"partial-name: {label}"))


FULL = ("variants", "p2", "prev", "anchors", "span", "source")
PARTIAL = ("denotation", "exactword", "multiowner", "qualified", "wholename")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["all"])
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY unset (map OAI_KEY into it inline)")
    wanted = set(FULL + PARTIAL) if "all" in args.arms else set(args.arms)
    started = time.time()
    inputs = inputs_with_gold()
    if wanted & set(FULL):
        _full_name_arms(inputs, wanted)
    if wanted & set(PARTIAL):
        _partial_arms(inputs, wanted)
    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
