"""`s_linker125` = `s_linker123` with no antecedent shortlist, and nothing else. No calls.

The s124 round's lesson was that an equivalence test must pin the variant against the arm
that was MEASURED, not only against its ancestor — a file that ships something no pilot
priced is a file with no number behind it. So:

  A. **The judge is untouched.** Every union-judging prompt is `s_linker123`'s, byte for
     byte, on all five projects.
  B. **The shortlist is gone, and only the shortlist.** Every resolver prompt equals
     `s_linker123`'s with the `NAMED BEFORE THIS CASE` lines removed and the paragraph
     about them replaced — checked by reconstructing it, not by eyeballing a diff.
  C. **It IS the measured arm.** The resolver prompts are byte-identical to
     `coref_shortlist_pilots.NoShortlist`, which read terra F1 95.52 / luna F1 90.53.
  D. **The scan is not run.** `_named_before` returns nothing for every case.
  E. **Nothing else was overridden.** Exactly two methods differ from `s_linker123`.

Run:

    /path/to/.venv/bin/python pilot/test_s125.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.linkers.experimental.s_linker123 import SLinker123  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker125 import (  # noqa: E402
    _CASE_LINE, _NO_LIST_PARA, _SHORTLIST_PARA, SLinker125)

from coref_shortlist_pilots import NoShortlist, load  # noqa: E402
from reading_pilots import DATASETS  # noqa: E402

RUN = ROOT.parent / "results/shortlistmark_e2e_terra_r1_20260914"

DECLARED = {"_named_before", "_prompt_coref"}

PASSED = FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  FAIL {name}{(': ' + detail) if detail else ''}")


class _Recorder:
    def set_phase(self, phase_name):
        pass


def probe(cls, data):
    linker = cls.__new__(cls)
    linker.doc_knowledge = data["knowledge"]
    linker.llm = _Recorder()
    return linker


def resolver_prompts(linker, data):
    sent = []
    linker._ask = lambda prompt, **_: sent.append(prompt) or {"resolutions": []}
    linker._resolve_references(
        data["sentences"], data["components"],
        data["name_to_id"], data["sent_map"])
    return sent


def judge_prompts(linker, data):
    sent = []
    linker._ask = lambda prompt, **_: sent.append(prompt) or {"validations": []}
    candidates = linker._name_candidates(
        data["sentences"], data["components"],
        data["name_to_id"], data["sent_map"])
    linker._judge_union(
        candidates, data["components"], data["sentences"], data["sent_map"])
    return sent


def expected_from(base_prompt):
    """What `s_linker123`'s prompt must become: lines dropped, paragraph swapped."""
    without = "\n".join(line for line in base_prompt.splitlines()
                        if not line.startswith(_CASE_LINE))
    return without.replace(_SHORTLIST_PARA, _NO_LIST_PARA, 1)


def main() -> int:
    print("E. exactly two methods differ from `s_linker123`")
    differing = set()
    for name in dir(SLinker123):
        if name.startswith("__") or not callable(getattr(SLinker123, name, None)):
            continue
        try:
            if inspect.getsource(getattr(SLinker123, name)) != \
                    inspect.getsource(getattr(SLinker125, name)):
                differing.add(name)
        except (OSError, TypeError):
            continue
    check("only the declared overrides differ", differing == DECLARED,
          f"differing={sorted(differing)}")
    check("the variant name moved", SLinker125._VARIANT_NAME == "s_linker125")

    for project in sorted(DATASETS):
        data = load(project, RUN)

        mine = resolver_prompts(probe(SLinker125, data), data)
        base = resolver_prompts(probe(SLinker123, data), data)
        arm = resolver_prompts(probe(NoShortlist, data), data)

        print(f"\n{project}")
        check(f"{project}: same resolver call count", len(mine) == len(base))
        check(f"{project}: the prompt IS s123's minus the shortlist",
              mine == [expected_from(p) for p in base])
        check(f"{project}: no shortlist line survives",
              not any(l.startswith(_CASE_LINE)
                      for p in mine for l in p.splitlines()))
        check(f"{project}: no paragraph about a list survives",
              all(_SHORTLIST_PARA not in p for p in mine))
        check(f"{project}: byte-identical to the MEASURED arm", mine == arm)

        jm = judge_prompts(probe(SLinker125, data), data)
        jb = judge_prompts(probe(SLinker123, data), data)
        check(f"{project}: union judging prompts are s123's byte for byte", jm == jb)

        listed = sum(1 for p in base for l in p.splitlines()
                     if l.startswith(_CASE_LINE))
        saved = sum(len(p) for p in base) - sum(len(p) for p in mine)
        print(f"   resolver {len(mine)} calls, judge {len(jm)} calls, "
              f"{listed} shortlists removed, {saved} B saved")

    print("\nD. the scan is not run")
    data = load("teammates", RUN)
    linker = probe(SLinker125, data)
    table = [{"sentence": s.number, "text": s.text} for s in data["sentences"]]
    names = [c.name for c in data["components"]]
    check("`_named_before` returns nothing",
          all(linker._named_before(names, table, t) == []
              for t in (5, 50, 150)))

    print(f"\n{PASSED}/{PASSED + FAILED} checks passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
