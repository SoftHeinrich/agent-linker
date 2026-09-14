#!/usr/bin/env python3
"""`s_linker123` is staged on `s_linker122` and changes only the evidence vocabulary.

The two features the arm is stacked on -- the `SURFACE_NOT_EVIDENCE` clause that
replaced the anchor block, and the code-computed coreference shortlist -- must arrive
here untouched. This asserts that rather than assuming it: every shared method is
byte-identical to the head's, the shortlist agrees sentence by sentence, and the clause
is present in every judging call the arm makes. No LLM calls.
"""
from __future__ import annotations

import glob
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import written_field_audit as A                                          # noqa: E402

from llm_sad_sam.linkers.experimental import s_linker122 as b122         # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker122 import SLinker122      # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker123 import SLinker123      # noqa: E402

#: Everything the arm must not have touched: both scans, the merged stream, the whole
#: judging loop, and the entire coreference linker including its shortlist.
SHARED = (
    "_named_before", "_prompt_coref", "_prompt_coref_validation",
    "_resolve_references", "_validate_coref_links", "_run_coreference_linker",
    "_run_validation_pass", "_window", "_states_a_name", "_name_spans", "_scan",
    "_extract_named_mentions", "_name_candidates", "_judge_union", "_stage_of",
    "_writes_name", "_find_exact_form", "_in_dotted_path", "_names_by_component",
)


def main() -> int:
    failures = 0

    def check(condition, label):
        nonlocal failures
        if condition:
            print(f"  ok    {label}")
        else:
            failures += 1
            print(f"  FAIL  {label}")

    print("MRO:", " -> ".join(c.__name__ for c in SLinker123.__mro__))
    check(SLinker123.__mro__[:2] == (SLinker123, SLinker122),
          "the arm is staged directly on the head")

    drift = [m for m in SHARED
             if inspect.getsource(getattr(SLinker123, m))
             != inspect.getsource(getattr(SLinker122, m))]
    check(not drift, f"{len(SHARED) - len(drift)}/{len(SHARED)} shared methods "
                     f"byte-identical{'' if not drift else ': ' + str(drift)}")

    overridden = sorted(k for k in vars(SLinker123)
                        if not k.startswith("__") and k != "_VARIANT_NAME")
    print(f"  overrides: {overridden}")
    check(set(overridden) == {"WRITTEN", "_only_in_identifier", "_written_as",
                              "_union_evidence", "_prompt_union",
                              "_format_union_case"},
          "the arm overrides exactly the evidence vocabulary and nothing else")

    runs = sorted(glob.glob(A.RUNS))
    if not runs:
        print(f"no recorded runs at {A.RUNS}", file=sys.stderr)
        return 2
    run = runs[0]

    total = agree = 0
    for project in A.PROJECTS:
        knowledge = A._knowledge(run, project, "s_linker122")
        if knowledge is None:
            continue
        sentences, components, _, _ = A._load(project)
        head = A._arm(SLinker122, knowledge)
        arm = A._arm(SLinker123, knowledge)
        names = [c.name for c in components]
        table = [{"sentence": s.number, "text": s.text} for s in sentences]
        for sentence in sentences:
            total += 1
            agree += (head._named_before(names, table, sentence.number)
                      == arm._named_before(names, table, sentence.number))
    check(agree == total, f"coreference shortlist identical on {agree}/{total} sentences")

    projects = clause_ok = 0
    for project in A.PROJECTS:
        knowledge = A._knowledge(run, project, "s_linker122")
        if knowledge is None:
            continue
        projects += 1
        _, prompts = A._prompts(A._arm(SLinker123, knowledge), *A._load(project))
        clause_ok += bool(prompts) and all(
            b122.SURFACE_NOT_EVIDENCE in p for p in prompts)
    check(clause_ok == projects,
          f"the clause is in every judging call on {clause_ok}/{projects} projects")

    print(f"\n{'PASS' if not failures else 'FAIL'} — {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
