"""`s_linker124` = `s_linker123` + the shortlist mark, and nothing else. No calls.

s124 composes two mechanisms that landed on `s_linker122` at the same time and were
priced on separate bases. A composition is only as trustworthy as the claim that its
parts do not interact, so that claim is checked rather than asserted:

  A. **The judge is untouched.** Every union-judging prompt s124 sends is
     `s_linker123`'s, byte for byte, on all five projects. The evidence-vocabulary
     round's numbers therefore still describe this file's judge.
  B. **Only the shortlist line differs.** Every resolver prompt differs from
     `s_linker123`'s in the `NAMED BEFORE THIS CASE` lines and in nothing else — so the
     resolver's authored English is the ancestor's and no rule speaks about the mark.
  C. **The mark is the measured arm's.** `coref_annot_pilots.Annot` is what read terra
     macro F2 +0.31 / luna +0.04. s124's shortlist lines must be identical to that arm's
     on the same inputs, or the file ships something no round priced.
  D. **Nothing else was overridden.** Exactly two methods differ from `s_linker123`.
  E. **It degrades rather than breaks** when the name stage kept nothing.

Run:

    /path/to/.venv/bin/python pilot/test_s124.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.linkers.experimental.s_linker123 import SLinker123  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker124 import SLinker124  # noqa: E402

from coref_annot_pilots import ARMS, load  # noqa: E402
from reading_pilots import DATASETS  # noqa: E402

RUN = ROOT.parent / "results/noanchor_e2e_terra_r1_20260914"

DECLARED = {"_run_linker", "_named_before"}

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


def probe(cls, data, linked=None):
    linker = cls.__new__(cls)
    linker.doc_knowledge = data["knowledge"]
    linker.llm = _Recorder()
    if linked is not None:
        if cls is SLinker124:
            linker._linked_mentions = frozenset(linked)
        else:
            linker.name_links = linked
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


def shortlist_lines(prompts):
    return [line for p in prompts for line in p.splitlines()
            if line.startswith("NAMED BEFORE THIS CASE:")]


def strip_shortlist(prompts):
    return ["\n".join(line for line in p.splitlines()
                      if not line.startswith("NAMED BEFORE THIS CASE:"))
            for p in prompts]


def main() -> int:
    print("D. exactly two methods differ from `s_linker123`")
    differing = set()
    for name in dir(SLinker123):
        if name.startswith("__") or not callable(getattr(SLinker123, name, None)):
            continue
        try:
            if inspect.getsource(getattr(SLinker123, name)) != \
                    inspect.getsource(getattr(SLinker124, name)):
                differing.add(name)
        except (OSError, TypeError):
            continue
    check("only the declared overrides differ", differing == DECLARED,
          f"differing={sorted(differing)}")
    check("the variant name moved", SLinker124._VARIANT_NAME == "s_linker124")

    for project in sorted(DATASETS):
        data = load(project, RUN)
        linked = data["name_links_by_name"]

        mine = resolver_prompts(probe(SLinker124, data, linked), data)
        base = resolver_prompts(probe(SLinker123, data), data)
        arm = resolver_prompts(probe(ARMS["annot"], data, linked), data)

        print(f"\n{project}")
        check(f"{project}: same resolver call count", len(mine) == len(base))
        check(f"{project}: only the shortlist lines differ from s123",
              strip_shortlist(mine) == strip_shortlist(base))
        check(f"{project}: the shortlist lines DO differ", mine != base)
        check(f"{project}: shortlist identical to the measured arm",
              shortlist_lines(mine) == shortlist_lines(arm))

        jm = judge_prompts(probe(SLinker124, data, linked), data)
        jb = judge_prompts(probe(SLinker123, data), data)
        check(f"{project}: union judging prompts are s123's byte for byte", jm == jb)
        marked = sum(l.count(", linked)") + l.count(", named only)")
                     for l in shortlist_lines(mine))
        print(f"   resolver {len(mine)} calls, judge {len(jm)} calls, "
              f"{len(shortlist_lines(mine))} shortlists, {marked} marked entries")

    print("\nE. with nothing linked, every entry reads `named only`")
    data = load("mediastore", RUN)
    empty = resolver_prompts(probe(SLinker124, data, frozenset()), data)
    check("no entry claims `linked`",
          all(", linked)" not in l for l in shortlist_lines(empty)))
    check("entries are still listed",
          any(", named only)" in l for l in shortlist_lines(empty)))

    print(f"\n{PASSED}/{PASSED + FAILED} checks passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
