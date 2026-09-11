"""Invariants of `s_linker120` — what the union does not change. No LLM calls.

The variant merges two judging passes into one. Everything the merge is not allowed to
touch is pinned here, over all five projects and with the recorded alias tables in
place:

  T1  the stream       the merged candidate set is exactly `full ∪ partial` at the
                       head's own bytes, and every candidate keeps the stage label its
                       links and its phase log are read by.
  T2  no case shown less
                       every case carries its span, its target, its sentence, its
                       preceding sentence and its anchors; every word-only case is
                       covered by the call's SENTENCES table, which is what the
                       denotation judge shows it today.
  T3  the evidence is a fact
                       `naming` agrees with `_states_a_name` and with the alias table;
                       `alternatives` never names the case's own component and never
                       names one whose words the sentence does not carry.
  T4  the verdict path
                       with `_ask` stubbed, the decision record has the head's keys and
                       shapes, approvals become links with the head's two sources, and
                       a reply the model never sent is a rejection, not a keep.

Usage, from the approach/ directory:
    ../.venv/bin/python pilot/test_s120_union.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import PROJECTS                                    # noqa: E402
from simmerge_audit import (                                         # noqa: E402
    arm_full, arm_partial, head_instance, load_project,
)
from llm_sad_sam.linkers.experimental.s_linker120 import (           # noqa: E402
    NameForm, SLinker120,
)


class _Recorder:
    """A stub `llm` attribute: records the phase, never calls anything."""

    def __init__(self):
        self.phases = []

    def set_phase(self, phase):
        self.phases.append(phase)


def _variant(data):
    linker = SLinker120.__new__(SLinker120)
    linker.doc_knowledge = data["linker"].doc_knowledge
    linker.llm = _Recorder()
    return linker


def _prompts_and_cases(linker, data, answer):
    """Run the union judging pass with `_ask` stubbed. Returns prompts and results.

    The stub answers **in each row's own field** — the boolean for a naming row, the
    enum for a word-only one — which is what the variant's parser reads. A stub that
    answered one field for both would have exercised half the parse path and passed.
    """
    prompts = []

    def fake_ask(prompt, **_kwargs):
        prompts.append(prompt)
        count = prompt.count("\nCase ") + prompt.startswith("Case ")
        lines = prompt.splitlines()
        numbers = [int(line.split()[1].rstrip(":"))
                   for line in lines if line.startswith("Case ")]
        blind = set()
        current = None
        for line in lines:
            if line.startswith("Case "):
                current = int(line.split()[1].rstrip(":"))
            elif current and "naming=word only" in line:
                blind.add(current)
        replies = []
        for n in numbers:
            reply = {"case": n, "claim": "quoted words"}
            if n in blind:
                reply["denotation"] = "participant" if answer(n) else "associated"
            else:
                reply["approve"] = answer(n)
            replies.append(reply)
        return {"validations": replies} if count else {"validations": []}

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
        head_full, head_partial = arm_full(data), arm_partial(data)

        # T1 — the stream
        candidates = linker._name_candidates(
            data["sentences"], data["components"], data["id_of"], sent_map)
        pairs = {(c.sentence_number, c.component_id) for c in candidates}
        check(pairs == head_full | head_partial,
              f"{project}: merged stream is full | partial")
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
            if evidence["naming"] == "word only":
                # The slot the match could not fill stays empty: no component name
                # may appear outside the sentence the case quotes.
                printed = "\n".join(line for line in case.splitlines()
                                     if not line.strip().startswith('"')
                                     and "[prev:" not in line)
                check(not any(name in printed for name in names),
                      f"{project}: a case with no computed component names none")
            else:
                check(candidate.component_name in case,
                      f"{project}: every case with a computed component shows it")
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

        # T4 — the verdict path
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

        # the verdict contract follows the case: an iteration that answers `per_row`
        # decides a case carrying no component by the denotation enum, and a boolean
        # cannot approve one. An iteration that answers `boolean` decides every case
        # by the boolean. Either way the contract is a property of the iteration,
        # not of a rubric the prompt switches between.
        contract = _variant(data)
        contract._ask = lambda prompt, **_k: {"validations": [
            {"case": int(line.split()[1].rstrip(":")), "claim": "quoted words",
             "approve": True}
            for line in prompt.splitlines() if line.startswith("Case ")]}
        kept_contract, _decisions = contract._judge_union(
            candidates, data["components"], data["sentences"], sent_map)
        blind_pairs = {(c.sentence_number, c.component_id) for c in candidates
                       if (linker.iteration.blind_word_only
                           or linker.iteration.contract_follows_batch)
                       and linker._union_evidence(c, data["components"],
                                                  sent_map)["naming"] == "word only"}
        decided_by_boolean = {(c.sentence_number, c.component_id)
                              for c in kept_contract}
        if (linker.iteration.verdict == "per_row"
                or linker.iteration.contract_follows_batch):
            check(not (decided_by_boolean & blind_pairs),
                  f"{project}: a boolean cannot approve a case with no component")
        else:
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
