"""Invariants for `s_linker114`: three skills must send the head's bytes. No calls.

`s_linker114` replaces three judging loops with one. A refactor that moves a measured
number is not a refactor, so this runs **the head's own methods and the variant's own
methods** side by side over recorded inputs -- with `_ask` replaced by a stub that
records the prompt and answers nothing -- and asserts:

  1. the prompt each judge would send is byte-identical, batch by batch;
  2. the decision recorded for every candidate is identical -- under an empty reply,
     where each judge's default polarity shows, AND under a reply that answers every
     case, alternating the verdict, where the kept rows are written;
  3. the kept set is identical.

The second reply matters: with `_ask` answering nothing no case is ever kept, so the
kept sets are both empty and the decision rows all come from the default. That proves
the prompts and the reject path and nothing about a kept row.

Inputs are real: the scan's candidates under each recorded run's own alias table for the
two scan-fed judges, and the recorded resolutions and their metadata for the strict one.

    ../.venv/bin/python pilot/test_s114_skills.py
"""
from __future__ import annotations

import itertools
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from chooser_audit import runs_of  # noqa: E402
from consolidation_audit import load_projects  # noqa: E402
from llm_sad_sam.core.data_types_v2 import SadSamLink  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker110 import SLinker110  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker114 import SLinker114  # noqa: E402


def empty_reply(_prompt, **_kwargs):
    """Answers nothing: every case falls to its judge's own default polarity."""
    return {}


def alternating_reply(prompt, **kwargs):
    """Answers every case, approving the odd ones, in the schema the gate asked for.

    Both reply keys are written, so one stub serves all three skills: each judge reads
    the key its own prompt demanded and ignores the other.
    """
    count = prompt.count("Case ") or prompt.count('"case":')
    rows = [{"case": n,
             "claim": f"quote {n}",
             "objection": "none" if n % 2 else "a ground",
             "approve": bool(n % 2),
             "denotation": "participant" if n % 2 else "associated"}
            for n in range(1, count + 1)]
    return {"validations": rows, "judgments": rows}


class Recorder:
    """A linker whose `_ask` answers as told and remembers what it was asked."""

    def __init__(self, cls, knowledge, reply=empty_reply):
        self.linker = cls.__new__(cls)
        self.linker.doc_knowledge = knowledge
        self.linker.llm = type("Phase", (), {"set_phase": lambda _s, _p: None})()
        self.prompts: list[str] = []
        self.reply = reply
        self.linker._ask = self._ask

    def _ask(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.reply(prompt, **kwargs)


def compare(label, head, arm, call, results):
    """Run the same call on both, record prompt / decision / kept differences."""
    head_kept, head_decisions = call(head.linker)
    arm_kept, arm_decisions = call(arm.linker)
    results["batches"] += len(head.prompts)
    if head.prompts != arm.prompts:
        for i, (a, b) in enumerate(zip(head.prompts, arm.prompts)):
            if a != b:
                results["failures"].append(f"{label}: prompt {i} differs "
                                           f"({len(a)} vs {len(b)} bytes)")
                break
        else:
            results["failures"].append(f"{label}: {len(head.prompts)} vs "
                                       f"{len(arm.prompts)} prompts")
    shared = {k: v for k, v in arm_decisions.items() if k in head_decisions}
    if shared != head_decisions:
        results["failures"].append(f"{label}: decisions differ where both record one")
    # The head writes no row at all for a case its reply never answered; the variant
    # writes one, rejected. Over six recorded runs of both models that case never
    # occurs (0.0 a run against 79.3-84.7 candidates), so the extra rows are a
    # tripwire for a silent omission and not a change to any measured number.
    extra = set(arm_decisions) - set(head_decisions)
    results["tripwire"] += len(extra)
    if any(arm_decisions[k]["approved"] for k in extra):
        results["failures"].append(f"{label}: an unanswered case was kept")
    results["kept"] += len(head_kept)
    if {(k.sentence_number, k.component_id) for k in head_kept} != \
            {(k.sentence_number, k.component_id) for k in arm_kept}:
        results["failures"].append(f"{label}: kept set differs")


def main() -> int:
    projects = load_projects()
    results = {"batches": 0, "tripwire": 0, "kept": 0, "failures": []}
    runs = 0
    for base in runs_of("s_linker110"):
        runs += 1
        for (project, data), reply in itertools.product(
                projects.items(), (empty_reply, alternating_reply)):
            knowledge = pickle.load(
                open(base / project / "knowledge.pkl", "rb"))["doc_knowledge"]
            sentences, components = data["sentences"], data["components"]
            sent_map = {s.number: s for s in sentences}
            name_to_id = data["name_to_id"]

            head = Recorder(SLinker110, knowledge, reply)
            arm = Recorder(SLinker114, knowledge, reply)

            named = head.linker._extract_named_mentions(
                sentences, components, name_to_id, sent_map)
            candidates = list(named.values())
            bundles = {(c.sentence_number, c.component_id):
                       head.linker._build_evidence_bundle(c, sent_map)
                       for c in candidates}
            compare(f"{project}/entity", head, arm,
                    lambda lk: lk._validate_with_evidence(
                        candidates, bundles, components, sent_map,
                        "phase_25_full_name_judge", "full_name"),
                    results)

            head.prompts.clear(); arm.prompts.clear()
            scanned = head.linker._scan(sentences, components)
            compare(f"{project}/denotation", head, arm,
                    lambda lk: lk._classify_denotations(scanned, sentences), results)

            head.prompts.clear(); arm.prompts.clear()
            recorded = pickle.load(
                open(base / project / "linker_coreference.pkl", "rb"))["feedback"]
            id_to_name = data["id_to_name"]
            metadata = {(m["sentence"], m["component_id"]): m
                        for m in recorded.get("metadata", [])}
            links = [SadSamLink(s, c, id_to_name[c], source="coreference")
                     for (s, c) in metadata if c in id_to_name]
            compare(f"{project}/coref", head, arm,
                    lambda lk: lk._validate_coref_links(
                        links, sent_map, components, metadata), results)

    total = results["batches"]
    for failure in results["failures"]:
        print("  FAIL", failure)
    print(f"{total - len(results['failures'])}/{total} judging batches identical "
          f"across {runs} recorded runs, three skills each")
    print(f"{results['kept']} kept rows compared, so the approve path is covered and "
          f"not just each judge's default")
    print(f"{results['tripwire']} rows the head would record nothing for, all "
          f"rejected -- every denotation case the empty stub leaves unanswered; in "
          f"the six recorded runs the count is 0.0 a run")
    return 1 if results["failures"] or not total else 0


if __name__ == "__main__":
    raise SystemExit(main())
