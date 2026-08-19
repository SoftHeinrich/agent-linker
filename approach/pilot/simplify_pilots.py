"""A/B pilots for the simplifications `complexity_audit.py` could not settle.

The audit answered the deterministic questions off checkpoints. Three
simplifications change prompt text, so they need arms:

    --pilot bundle    Two changes to the evidence bundle, both of which the audit
                      showed carry no decision of their own:
                        * drop the `Rationale:` line -- one distinct value across
                          every candidate on all five projects, so it is a
                          constant the judge is told every time;
                        * build anchors with `_find_exact_form` instead of
                          `has_standalone_mention`, collapsing two primitives for
                          one question into one. The two disagree on 47 of 3697
                          (name, sentence) pairs, always in the same direction
                          (exact matches, standalone does not), and flip the
                          coreference antecedent gate on 0 of the promoted run's
                          resolutions -- so only anchor lists move, never a gate.
    --pilot mention   The bundle above, plus no `mention=` field: this is what
                      retires `MentionType` (5 values), `_classify_mention_typed`
                      and `_all_occurrences_in_qualified_path`. The audit found
                      the distribution non-degenerate (122 proper / 42 alias / 11
                      code-token / 10 lowercase / 3 indirect), so this one cannot
                      be argued away and has to be measured.
    --pilot corefref  Drop `antecedent_via_alias` from the coreference prompt:
                      the closing sentence of COREF_RULES that asks for it, the
                      whole 488-byte ANTECEDENT_ALIAS_RULES block that defines
                      it, and the response field. The audit confirmed the model
                      sets it true on 64 resolutions and no gate reads any of
                      them.

Same discipline as `design_pilots.py`: upstream from a promoted run's
checkpoints, one stage varied, N runs per side, permutation test.
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
    RUNS, OUT, SOURCE_RUN, collect, inputs_with_gold, new_linker, report,
    scorers, _coref_inputs, _judge_stage, _prepare_extraction,
)

from llm_sad_sam.linkers.experimental.helper_v3 import has_standalone_mention
from llm_sad_sam.linkers.experimental.s_linker25 import (
    SLinker25, EvidenceBundle, COREF_RULES,
)

VIA_ALIAS_SENTENCE = (
    " When the antecedent sentence uses a known alias rather than the full "
    "canonical name, set antecedent_via_alias=true."
)


class SlimBundle(SLinker25):
    """One name primitive, and no constant Rationale line."""

    def _build_evidence_bundle(self, candidate, sent_map,
                               rationale="Named-mention extraction"):
        comp_name = candidate.component_name
        snum = candidate.sentence_number
        mention_type = self._classify_mention_typed(
            comp_name, candidate.sentence_text
        ).value
        prev_sent = sent_map.get(snum - 1)
        anchors = []
        for s in sorted(sent_map.values(), key=lambda x: x.number):
            if s.number == snum:
                continue
            if self._find_exact_form(s.text, comp_name):
                anchors.append(f"S{s.number}: {s.text}")
                if len(anchors) >= self.ANCHOR_LIMIT:
                    break
        return EvidenceBundle(
            source=candidate.source,
            matched_span=candidate.matched_text or comp_name,
            mention_type=mention_type,
            preceding_text=prev_sent.text if prev_sent else "",
            anchor_sentences=anchors,
            extraction_rationale=rationale,
        )

    def _format_evidence(self, bundle) -> str:
        lines = [
            f"  Evidence: source={bundle.source}, span=\"{bundle.matched_span}\", "
            f"mention={bundle.mention_type}",
        ]
        if bundle.preceding_text:
            lines.append(f"  [prev: \"{bundle.preceding_text}\"]")
        if bundle.anchor_sentences:
            lines.append("  Anchors (confirmed refs):")
            for anchor in bundle.anchor_sentences:
                lines.append(f"    {anchor}")
        return "\n".join(lines)


class SlimNoMentionType(SlimBundle):
    """The slim bundle without the mention-type field at all."""

    def _format_evidence(self, bundle) -> str:
        lines = [
            f"  Evidence: source={bundle.source}, span=\"{bundle.matched_span}\"",
        ]
        if bundle.preceding_text:
            lines.append(f"  [prev: \"{bundle.preceding_text}\"]")
        if bundle.anchor_sentences:
            lines.append("  Anchors (confirmed refs):")
            for anchor in bundle.anchor_sentences:
                lines.append(f"    {anchor}")
        return "\n".join(lines)


class OneNamePrimitive(SLinker25):
    """Collapse the last name-matching asymmetry into `_find_exact_form`.

    `_states_name_alone` is case-sensitive for single-word names and
    case-insensitive for multi-word ones; `_find_exact_form` is
    case-insensitive throughout. They disagree on 47 of 3697 (name, sentence)
    pairs. The coreference antecedent gate is provably unaffected (0 flips
    measured), so what this arm varies is the mention-type field of the
    full-name judge's evidence bundle.
    """

    def _states_name_alone(self, comp_name, text) -> bool:
        return bool(self._find_exact_form(text, comp_name))


class OneTestCascade(SLinker25):
    """One name test, and the case distinction made explicit instead of implied.

    `_states_name_alone` exists only to separate a proper-case mention from a
    lowercase one. `_find_exact_form` already returns the matched surface, so the
    distinction is a string comparison, not a second predicate with its own case
    rules. All five mention labels stay reachable, which the plain
    `OneNamePrimitive` arm gives up (it makes CODE_TOKEN unreachable).
    """

    def _classify_mention_typed(self, comp_name: str, text: str):
        from llm_sad_sam.linkers.experimental.s_linker25 import MentionType
        matched = self._find_exact_form(text, comp_name)
        if matched:
            if self._all_occurrences_in_qualified_path(comp_name.lower(), text):
                return MentionType.CODE_TOKEN
            return (MentionType.PROPER_STANDALONE if matched == comp_name
                    else MentionType.LOWERCASE_PROSE)
        if self.doc_knowledge:
            for alias, component in self.doc_knowledge.aliases.items():
                if component == comp_name and self._find_exact_form(text, alias):
                    return MentionType.VIA_ALIAS
        return MentionType.INDIRECT


class CorefNoViaAlias(SLinker25):
    """Coreference prompt without the alias self-report it never reads."""

    @staticmethod
    def _prompt_coref(comp_names, cases) -> str:
        rules = COREF_RULES.replace(VIA_ALIAS_SENTENCE, "")
        assert rules != COREF_RULES, "the via-alias sentence moved"
        prompt = f"""Resolve references (pronouns and noun phrases that refer back) to components.

COMPONENTS: {', '.join(comp_names)}

For each TARGET sentence below, identify any pronoun or noun phrase that
refers back to a component listed above. If a target sentence has no such
reference to a listed component, return no resolution for it. Be conservative — only include resolutions you are CERTAIN about.

"""
        for i, case in enumerate(cases):
            prompt += f"--- Case {i+1}: S{case['sent'].number} ---\n"
            prompt += "CONTEXT:\n" + "\n".join(case["context"]) + "\n"
            prompt += f"TARGET: S{case['sent'].number} (marked with >>>)\n\n"
        prompt += f"""{rules}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "reference": "the server", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

JSON only:"""
        return prompt


# ── bundle arms, over the full-name judge ────────────────────────────────────

def pilot_bundle(inputs, which):
    print("\n### bundle arms — one control, two slimmer evidence bundles")
    _prepare_extraction(inputs)
    gold = {n: inputs[n]["gold"] for n in inputs}

    def stage(cls, tag):
        return collect("x", lambda run, name: _judge_stage(
            cls, inputs[name], name,
            inputs[name]["knowledge"]["doc_knowledge"],
            inputs[name]["knowledge"]["model_knowledge"], tag), inputs)

    control = stage(SLinker25, "simp_control")
    print(f"  A_current: {[len(s) for s in control]}")

    if "bundle" in which:
        arm = stage(SlimBundle, "simp_bundle")
        print(f"  B_slim_bundle: {[len(s) for s in arm]}")
        report("bundle_slim", permutation_report(
            {"A_current": control, "B_slim_bundle": arm}, scorers(gold),
            title="bundle — one primitive, no constant rationale line"))

    if "cascade" in which:
        arm = stage(OneTestCascade, "simp_cascade")
        print(f"  B_one_test_cascade: {[len(s) for s in arm]}")
        report("one_test_cascade", permutation_report(
            {"A_current": control, "B_one_test_cascade": arm}, scorers(gold),
            title="cascade — one name test, explicit case comparison"))

    if "primitive" in which:
        arm = stage(OneNamePrimitive, "simp_primitive")
        print(f"  B_one_primitive: {[len(s) for s in arm]}")
        report("one_name_primitive", permutation_report(
            {"A_current": control, "B_one_primitive": arm}, scorers(gold),
            title="primitive — case asymmetry removed, one name test"))

    if "mention" in which:
        arm = stage(SlimNoMentionType, "simp_mention")
        print(f"  B_no_mention_type: {[len(s) for s in arm]}")
        report("bundle_no_mention_type", permutation_report(
            {"A_current": control, "B_no_mention_type": arm}, scorers(gold),
            title="mention — slim bundle with the mention-type field removed"))


# ── coreference prompt arm ───────────────────────────────────────────────────

def pilot_corefref(inputs):
    """Vary the resolution prompt, then judge; score the final set.

    Prior links come from the promoted run, as in the `sequence` pilot, so both
    arms compose against the same upstream output.
    """
    print("\n### corefref — coreference prompt without antecedent_via_alias")
    prepared = _coref_inputs(inputs)
    for name, state in prepared.items():
        print(f"  {name:14s} prior {len(state['prior']):3d}")

    def one(arm, run, name):
        item = inputs[name]
        cls = SLinker25 if arm == "A_with_via_alias" else CorefNoViaAlias
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
    for arm in ("A_with_via_alias", "B_without"):
        arms[arm] = collect(arm, lambda run, name, a=arm: one(a, run, name), inputs)
        print(f"  {arm}: {[len(s) for s in arms[arm]]}")

    gold = {n: inputs[n]["gold"] for n in inputs}
    report("coref_no_via_alias", permutation_report(
        arms, scorers(gold), title="corefref — via-alias rules removed"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", nargs="+", required=True,
                        choices=["bundle", "mention", "primitive", "cascade",
                                 "corefref"])
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY unset (map OAI_KEY into it inline)")
    started = time.time()
    inputs = inputs_with_gold()
    bundle_arms = [p for p in args.pilot
                   if p in ("bundle", "mention", "primitive", "cascade")]
    if bundle_arms:
        pilot_bundle(inputs, bundle_arms)
    if "corefref" in args.pilot:
        pilot_corefref(inputs)
    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
