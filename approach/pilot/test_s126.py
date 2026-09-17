#!/usr/bin/env python3
"""Deterministic contract checks for the composed s126 mechanisms."""
from __future__ import annotations

import contextlib
import io
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.linkers.experimental.s_linker125 import SLinker125  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker126 import (            # noqa: E402
    NAMING_OF, SLinker126, TRACE_LINK_RULE,
)
from coref_exact_pilots import load                                       # noqa: E402
from reading_pilots import DATASETS                                    # noqa: E402

RUN = ROOT.parent / "results/shortlistmark_e2e_terra_r1_20260914"
PASSED = FAILED = 0


def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        print(f"FAIL {label}{(': ' + detail) if detail else ''}")


class Recorder:
    def __init__(self):
        self.phase = ""

    def set_phase(self, phase):
        self.phase = phase


def probe(cls, data):
    linker = cls.__new__(cls)
    linker.doc_knowledge = data["knowledge"]
    linker.llm = Recorder()
    linker._llm_calls = []
    return linker


def candidates(linker, data):
    with contextlib.redirect_stdout(io.StringIO()):
        return linker._name_candidates(
            data["sentences"], data["components"],
            data["name_to_id"], data["sent_map"])


def main():
    check("variant name", SLinker126._VARIANT_NAME == "s_linker126")
    check("written labels follow approach", SLinker126.WRITTEN == (
        "exact", "alias", "part", "qualified name"))
    check("judge prompt defines written labels", all(
        label in TRACE_LINK_RULE for label in (
            "exact", "alias", "part", "qualified name")))
    check("antecedent forms use written labels",
          SLinker126.ANTECEDENT_FORMS == ("exact", "alias"))
    check("naming projection remains stable", NAMING_OF == {
        "exact": "whole name",
        "qualified name": "whole name",
        "alias": "alias",
        "part": "word only",
    })

    label_probe = SLinker126.__new__(SLinker126)
    label_probe.doc_knowledge = SimpleNamespace(aliases={"DB": "Database"})
    check("written exact", label_probe._written_as(
        "The Database component stores records.", "Database") == "exact")
    check("written alias", label_probe._written_as(
        "The DB component stores records.", "Database") == "alias")
    check("written part", label_probe._written_as(
        "Each client connects here.", "HTML5 Client") == "part")
    check("written qualified name", label_probe._written_as(
        "pkg.db.read() is called.", "DB") == "qualified name")
    check("competitors rule removed", "competitors --" not in TRACE_LINK_RULE)

    for project in sorted(DATASETS):
        data = load(project, RUN)
        ctl = candidates(probe(SLinker125, data), data)
        arm_linker = probe(SLinker126, data)
        arm = candidates(arm_linker, data)
        ctl_keys = {(c.sentence_number, c.component_id) for c in ctl}
        arm_keys = {(c.sentence_number, c.component_id) for c in arm}
        check(f"{project}: greedy only removes", arm_keys <= ctl_keys)
        if project != "bigbluebutton":
            check(f"{project}: candidate set unchanged", arm_keys == ctl_keys)
        else:
            check("bigbluebutton: greedy removes twelve candidates",
                  len(ctl_keys - arm_keys) == 12, str(len(ctl_keys - arm_keys)))
            groups = {}
            for candidate in arm:
                groups.setdefault(arm_linker._group_key(candidate), []).append(candidate)
            ambiguous = [group for group in groups.values() if len(group) > 1]
            check("bigbluebutton: only S27/S31 remain ambiguous",
                  {group[0].sentence_number for group in ambiguous} == {27, 31})

        captured = []
        arm_linker._ask = lambda prompt, **_: captured.append(prompt) or {
            "validations": [], "resolutions": []}
        arm_linker._judge_union(
            arm, data["components"], data["sentences"], data["sent_map"])
        check(f"{project}: no competitors field reaches a prompt",
              all("competitors" not in prompt for prompt in captured))
        if project == "bigbluebutton":
            check("bigbluebutton: no ambiguity call",
                  arm_linker.llm.phase != "phase_25_name_ambiguous_judge")
            approved, decisions = arm_linker._judge_union(
                arm, data["components"], data["sentences"], data["sent_map"])
            ambiguous_keys = {
                (candidate.sentence_number, candidate.component_id)
                for group in ambiguous for candidate in group}
            approved_keys = {
                (candidate.sentence_number, candidate.component_id)
                for candidate in approved}
            check("bigbluebutton: ambiguous candidates discarded",
                  not ambiguous_keys & approved_keys)
            check("bigbluebutton: discards recorded",
                  all(decisions[key]["stage"] == "name_ambiguous_discard"
                      for key in ambiguous_keys))

        resolver = probe(SLinker126, data)
        prompts = []
        resolver._ask = lambda prompt, **_: prompts.append(prompt) or {"resolutions": []}
        resolver._resolve_references(
            data["sentences"], data["components"],
            data["name_to_id"], data["sent_map"])
        check(f"{project}: resolver shortlist remains absent",
              all("NAMED BEFORE THIS CASE" not in prompt for prompt in prompts))

    print(f"{PASSED}/{PASSED + FAILED} checks passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
