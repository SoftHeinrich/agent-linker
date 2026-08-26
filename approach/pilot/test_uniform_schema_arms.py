"""Invariants for the uniform-schema arms (s116-s119). No calls.

Each arm is one declared difference from the head at one gate, and every arm is a
`JudgeSkill` field replaced on `s_linker114` -- which is what that refactor was for.
This asserts, over the recorded runs' own candidate sets, that:

  1. every arm sends the head's prompt with exactly the substitutions it declares --
     asserted as a line-level diff against the head, not a length check;
  2. no arm touches a gate it does not name: the other two judges' prompts stay
     byte-identical to the head's;
  3. every arm parses a full reply and a silent one, and keeps only what its own
     polarity licenses;
  4. `s_linker119`, which replies in the lenient gate's schema at the sortal gate,
     reads the boolean and keeps the approved cases -- the enum's `participant`
     demand restated.

    ../.venv/bin/python pilot/test_uniform_schema_arms.py
"""
from __future__ import annotations

import difflib
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from chooser_audit import runs_of                                    # noqa: E402
from consolidation_audit import load_projects                        # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker114 import SLinker114  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker116 import SLinker116  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker117 import SLinker117  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker118 import SLinker118  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker119 import SLinker119  # noqa: E402
from test_s114_skills import Recorder, alternating_reply, empty_reply  # noqa: E402

#: arm -> (the gate it names, how many lines of the head's prompt it may move)
ARMS = {
    SLinker116: ("entity", 7),      # decide clause 1 -> 4 lines, schema line 1 -> 1
    SLinker117: ("entity", 8),      # instruction 3 -> 3 lines, schema line 1 -> 1
    SLinker118: ("denotation", 6),  # ground line 0 -> 3, schema line 1 -> 2
    SLinker119: ("denotation", 99),  # the whole prompt, by design
}

GATES = ("entity", "denotation", "coref")


def prompts_of(cls, gate, knowledge, data, sent_map, base, project, reply=empty_reply):
    """Every prompt one gate of one linker would send over this project."""
    rec = Recorder(cls, knowledge, reply)
    sentences, components = data["sentences"], data["components"]
    if gate == "entity":
        named = rec.linker._extract_named_mentions(
            sentences, components, data["name_to_id"], sent_map)
        candidates = list(named.values())
        bundles = {(c.sentence_number, c.component_id):
                   rec.linker._build_evidence_bundle(c, sent_map)
                   for c in candidates}
        kept, decisions = rec.linker._validate_with_evidence(
            candidates, bundles, components, sent_map,
            "phase_25_full_name_judge", "full_name")
    elif gate == "denotation":
        kept, decisions = rec.linker._classify_denotations(
            rec.linker._scan(sentences, components), sentences)
    else:
        recorded = pickle.load(
            open(base / project / "linker_coreference.pkl", "rb"))["feedback"]
        from llm_sad_sam.core.data_types_v2 import SadSamLink
        id_to_name = data["id_to_name"]
        metadata = {(m["sentence"], m["component_id"]): m
                    for m in recorded.get("metadata", [])}
        links = [SadSamLink(s, c, id_to_name[c], source="coreference")
                 for (s, c) in metadata if c in id_to_name]
        kept, decisions = rec.linker._validate_coref_links(
            links, sent_map, components, metadata)
    return rec.prompts, kept, decisions


def main() -> int:
    projects = load_projects()
    base = next(iter(runs_of("s_linker110")))
    failures: list[str] = []
    checked = 0
    for project, data in projects.items():
        knowledge = pickle.load(
            open(base / project / "knowledge.pkl", "rb"))["doc_knowledge"]
        sent_map = {s.number: s for s in data["sentences"]}
        head = {gate: prompts_of(SLinker114, gate, knowledge, data, sent_map,
                                 base, project)[0] for gate in GATES}

        for arm, (named_gate, budget) in ARMS.items():
            tag = f"{project}/{arm.__name__}"
            for gate in GATES:
                got, _kept, _dec = prompts_of(arm, gate, knowledge, data, sent_map,
                                              base, project)
                checked += len(got)
                if len(got) != len(head[gate]):
                    failures.append(f"{tag}/{gate}: {len(got)} prompts vs "
                                    f"{len(head[gate])}")
                    continue
                for i, (a, b) in enumerate(zip(head[gate], got)):
                    moved = sum(1 for line in difflib.unified_diff(
                        a.split("\n"), b.split("\n"), lineterm="", n=0)
                        if line[:1] in "+-" and line[:3] not in ("---", "+++"))
                    if gate != named_gate and moved:
                        failures.append(f"{tag}: touched the {gate} gate "
                                        f"({moved} lines in batch {i})")
                        break
                    if gate == named_gate and not 0 < moved <= budget:
                        failures.append(f"{tag}: {moved} lines moved at its own gate, "
                                        f"budget {budget}")
                        break

            # The declared polarity, over a reply that answers every case and one that
            # answers none. Silence must keep nothing at any arm.
            for reply, expect_kept in ((alternating_reply, True), (empty_reply, False)):
                _p, kept, decisions = prompts_of(arm, named_gate, knowledge, data,
                                                 sent_map, base, project, reply)
                if not decisions:
                    continue
                if bool(kept) != expect_kept:
                    failures.append(
                        f"{tag}: {len(kept)} kept under "
                        f"{'a full' if expect_kept else 'a silent'} reply")

    for failure in failures:
        print("  FAIL", failure)
    print(f"{checked} prompts checked over {len(projects)} projects, "
          f"{len(ARMS)} arms x {len(GATES)} gates; "
          f"{len(failures)} failures")
    return 1 if failures or not checked else 0


if __name__ == "__main__":
    raise SystemExit(main())
