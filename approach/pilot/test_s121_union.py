"""Invariants of `s_linker121` — what the union does not change. No LLM calls.

The variant merges two judging passes into one, and judges every candidate through one
prompt shape. Everything that is not allowed to vary is pinned here, over all five
projects and with the recorded alias tables in place:

  T1  the stream       the merged candidate set is exactly `full ∪ partial` at the
                       head's own bytes, and every candidate keeps the stage label its
                       links and its phase log are read by.
  T2  one case shape   every case names its component and carries its span, its
                       sentence, its preceding sentence and its anchors; the evidence
                       fields come from one fixed vocabulary; every word-only case is
                       covered by the call's SENTENCES table, which is what the
                       denotation judge shows it today.
  T3  the evidence is a fact
                       `naming` agrees with `_states_a_name` and with the alias table;
                       `alternatives` never names the case's own component and never
                       names one whose words the sentence does not carry.
  T4  one call shape   every call carries the same catalog line, the same rule, the
                       same demand and the same reply contract — no call is routed to
                       a second structure by what its cases happen to contain.
  T5  the verdict path
                       with `_ask` stubbed, the decision record has the head's keys and
                       shapes, approvals become links with the head's two sources, and
                       a reply the model never sent is a rejection, not a keep.

Usage, from the approach/ directory:
    ../.venv/bin/python pilot/test_s121_union.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import PROJECTS                                    # noqa: E402
from simmerge_audit import (                                         # noqa: E402
    arm_full, arm_partial, arm_partial_all, head_instance, load_project,
)
from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker121 import (           # noqa: E402
    NameForm, SLinker121, TRACE_LINK_RULE, UNION_DEMAND, UNION_REPLY,
)


class _Recorder:
    """A stub `llm` attribute: records the phase, never calls anything."""

    def __init__(self):
        self.phases = []

    def set_phase(self, phase):
        self.phases.append(phase)


def _variant(data):
    linker = SLinker121.__new__(SLinker121)
    linker.doc_knowledge = data["linker"].doc_knowledge
    linker.llm = _Recorder()
    return linker


def _prompts_and_cases(linker, data, answer):
    """Run the union judging pass with `_ask` stubbed. Returns prompts and results.

    One contract, so the stub answers one field. That is the point of the variant:
    there is no second reply shape for the parser to branch on.
    """
    prompts = []

    def fake_ask(prompt, **_kwargs):
        prompts.append(prompt)
        numbers = [int(line.split()[1].rstrip(":"))
                   for line in prompt.splitlines() if line.startswith("Case ")]
        return {"validations": [
            {"case": n, "claim": "quoted words", "approve": answer(n)}
            for n in numbers]}

    linker._ask = fake_ask
    sent_map = {s.number: s for s in data["sentences"]}
    candidates = linker._name_candidates(
        data["sentences"], data["components"], data["id_of"], sent_map)
    approved, decisions = linker._judge_union(
        candidates, data["components"], data["sentences"], sent_map)
    return prompts, candidates, approved, decisions


def main():
    checks = failures = 0

    def check(condition, what):
        nonlocal checks, failures
        checks += 1
        if not condition:
            failures += 1
            print(f"  FAIL {what}")

    for project in PROJECTS:
        data = load_project(project, True)
        linker = _variant(data)
        sent_map = {s.number: s for s in data["sentences"]}
        # The one-word reference is the UNREFUSED scan: `s_linker121` dropped the
        # ancestor's nesting refusal (`../results/s121_ablations/`), so the pairs it
        # used to end are cases here. The dropped set is pinned, not waved through.
        head_full, head_partial = arm_full(data), arm_partial_all(data)
        head_refused = head_partial - arm_partial(data)

        # T1 — the stream
        candidates = linker._name_candidates(
            data["sentences"], data["components"], data["id_of"], sent_map)
        pairs = {(c.sentence_number, c.component_id) for c in candidates}
        check(pairs == head_full | head_partial,
              f"{project}: merged stream is full | partial")
        check(head_refused <= pairs,
              f"{project}: the ancestor's refused pairs are cases here "
              f"({len(head_refused)})")
        check(len(candidates) == len(pairs), f"{project}: no duplicate pairs")
        stages = {(c.sentence_number, c.component_id): linker._stage_of(c)
                  for c in candidates}
        check(all(stages[p] == "full_name" for p in head_full),
              f"{project}: full-name candidates keep their stage label")
        check(all(stages[p] == "partial_name" for p in head_partial - head_full),
              f"{project}: partial-name candidates keep their stage label")

        # T2/T3 — the cases and the evidence
        by_id = {c.id: c for c in data["components"]}
        names = {component.name for component in data["components"]}
        for candidate in candidates:
            evidence = linker._union_evidence(
                candidate, data["components"], sent_map)
            case = linker._format_union_case(1, candidate, evidence, sent_map)
            check(candidate.sentence_text in case, f"{project}: case shows its sentence")
            # One format for every candidate: the same header, the same evidence
            # line, the same field vocabulary. What differs between cases is the
            # VALUE of a field, never whether the case is a different kind of case.
            check(case.startswith(f'Case 1: "{evidence["span"]}" '
                                  f'-> {candidate.component_name}'),
                  f"{project}: every case names its component in the header")
            check("Evidence: writes=" in case,
                  f"{project}: every case carries the writes field")
            check(linker.WRITES[evidence["naming"]] in case,
                  f"{project}: the writes field says what the sentence writes")
            fields = {part.split("=")[0].strip() for line in case.splitlines()
                      if line.strip().startswith("Evidence:")
                      for part in line.split("Evidence:")[1].split(", ")
                      if "=" in part}
            check(fields <= {"writes", "alternatives", "mention"},
                  f"{project}: no case invents an evidence field ({fields})")
            check(evidence["span"] and evidence["span"] in case,
                  f"{project}: case shows its span")
            whole = bool(linker._writes_name(candidate.sentence_text,
                                             candidate.component_name))
            check((evidence["naming"] == "whole name") == whole,
                  f"{project}: naming=whole name iff the name is written")
            if evidence["naming"] == "alias":
                check(linker._states_a_name(candidate.sentence_text,
                                            candidate.component_name),
                      f"{project}: an alias row states a name of N(c)")
            if evidence["naming"] == "word only":
                check(not linker._states_a_name(candidate.sentence_text,
                                                candidate.component_name),
                      f"{project}: a word-only row states no name of N(c)")
            check(candidate.component_name not in evidence["alternatives"],
                  f"{project}: alternatives exclude the case's own component")
            for other in evidence["alternatives"]:
                check(bool(linker._name_spans(candidate.sentence_text, other,
                                              NameForm.ANY_WORD)),
                      f"{project}: an alternative's word is in the sentence")

        # T2 — the window every word-only case is shown today
        prompts, candidates, approved, decisions = _prompts_and_cases(
            linker, data, lambda n: n % 2 == 1)
        word_only = [c for c in candidates
                     if linker._union_evidence(c, data["components"],
                                               sent_map)["naming"] == "word only"]
        covered = set()
        for prompt in prompts:
            for sentence in data["sentences"]:
                if f'"sentence": {sentence.number},' in prompt:
                    covered.add(sentence.number)
        needed = {s.number for c in word_only
                  for s in linker._window(c.sentence_number, data["sentences"])}
        check(needed <= covered,
              f"{project}: every word-only case's window is in its call's table")

        # T4 — one call shape
        for prompt in prompts:
            check(f"COMPONENTS: {', '.join(sorted(names) and get_comp_names(data['components']))}"
                  in prompt, f"{project}: every call carries the catalog")
            check(TRACE_LINK_RULE in prompt, f"{project}: every call carries the rule")
            check(UNION_DEMAND in prompt, f"{project}: every call carries one demand")
            check(UNION_REPLY in prompt, f"{project}: every call carries one contract")
        # Everything above the cases, minus the sentence window, is one string. The
        # window is the only part of the envelope a call's own cases decide, and it
        # is content the head already showed those cases, not a second structure.
        envelopes = {re.sub(r"\nSENTENCES\n.*\n\n", "\n", prompt.split("CASES:")[0])
                     for prompt in prompts}
        check(len(envelopes) == 1,
              f"{project}: every call has the same envelope ({len(envelopes)} seen)")

        # T5 — the verdict path
        check(all(phase == "phase_25_name_union_judge"
                  for phase in linker.llm.phases),
              f"{project}: the judging phase is tagged once, under one name")
        check(len(decisions) == len(candidates),
              f"{project}: one decision record per candidate")
        check(all(set(record) == {"approved", "claim", "naming", "path", "stage"}
                  for record in decisions.values()),
              f"{project}: decision records keep the head's fields")
        check(all(record["stage"] == "name_union_judge"
                  for record in decisions.values()),
              f"{project}: decisions name the union judge")
        approved_pairs = {(c.sentence_number, c.component_id) for c in approved}
        check(all(decisions[p]["approved"] for p in approved_pairs),
              f"{project}: approvals and decisions agree")
        links, _feedback = linker._run_name_linker(
            data["sentences"], data["components"], data["id_of"], sent_map)
        check({link.source for link in links} <= {"full_name", "partial_name"},
              f"{project}: links carry the head's two sources")
        check({(link.sentence_number, link.component_id) for link in links}
              <= head_full | head_partial,
              f"{project}: no link outside the scans")

        # one verdict field decides every case: there is no case the boolean
        # cannot reach, because there is no second contract to reach it with.
        contract = _variant(data)
        contract._ask = lambda prompt, **_k: {"validations": [
            {"case": int(line.split()[1].rstrip(":")), "claim": "quoted words",
             "approve": True}
            for line in prompt.splitlines() if line.startswith("Case ")]}
        kept_contract, _decisions = contract._judge_union(
            candidates, data["components"], data["sentences"], sent_map)
        check(len(kept_contract) == len(candidates),
              f"{project}: one verdict field decides every case")

        # a reply the model never sent must not keep anything
        silent = _variant(data)
        silent._ask = lambda *_args, **_kwargs: {}
        _approved, silent_decisions = silent._judge_union(
            candidates, data["components"], data["sentences"], sent_map)
        check(not _approved, f"{project}: an empty reply keeps nothing")
        check(all(not record["approved"] for record in silent_decisions.values()),
              f"{project}: an empty reply rejects by record, not by omission")

    print(f"\n{checks - failures}/{checks} checks pass")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
