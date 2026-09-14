"""`s_linker123` against its ancestor, and against the arm that measured it. No calls.

`s_linker123` is a STANDALONE file by the branch's one-file-per-reported-variant policy,
so it is a copy of `s_linker122` with one delta. Two things have to hold and neither is
worth asserting in prose:

  A. **Everything that did not change, did not change.** Method by method against
     `s_linker122`, and constant by constant. The declared changes are enumerated here
     and anything else differing is a failure.
  B. **The shipped file IS the arm that was measured.** `pilot/coref_annot_pilots.py`'s
     `Annot` arm is what read terra macro F2 +0.31 / luna +0.04. If `s_linker123`'s
     resolver prompts are not byte-identical to that arm's over all five projects, the
     file ships something the round never priced. This is the check the compaction round
     said to write BEFORE adopting a change, not after.

Run:

    /path/to/.venv/bin/python pilot/test_s123_standalone.py
"""
from __future__ import annotations

import inspect
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences  # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker122 as ancestor  # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker123 as shipped  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402

from coref_annot_pilots import ARMS, load  # noqa: E402
from reading_pilots import BENCH, DATASETS  # noqa: E402

RUN = ROOT.parent / "results/noanchor_e2e_terra_r1_20260914"

#: The methods this variant declares changed, and why. Anything else that differs is a
#: regression, and anything listed here that does NOT differ means the delta was lost.
DECLARED = {
    "_run_linker": "takes `linked` and passes it to the coreference linker",
    "_run_coreference_linker": "takes `linked` and passes it down",
    "_resolve_references": "takes `linked` and passes it to the prompt builder",
    "_prompt_coref": "marks each shortlist entry via `_mark_shortlist`",
    "link": "passes what it has accumulated into `_run_linker`",
}

#: New on this variant; the ancestor has no counterpart.
ADDED = {"_mark_shortlist"}

PASSED = FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  FAIL {name}{(': ' + detail) if detail else ''}")


def source(cls, name):
    return inspect.getsource(getattr(cls, name))


def test_methods():
    print("A. every method against `s_linker122`")
    old, new = ancestor.SLinker122, shipped.SLinker123
    old_names = {n for n in dir(old) if callable(getattr(old, n, None))
                 and not n.startswith("__")}
    new_names = {n for n in dir(new) if callable(getattr(new, n, None))
                 and not n.startswith("__")}

    check("no method was dropped", old_names <= new_names,
          f"missing {sorted(old_names - new_names)}")
    check("only the declared additions are new", new_names - old_names == ADDED,
          f"unexpected {sorted(new_names - old_names - ADDED)}")

    identical = changed = 0
    for name in sorted(old_names & new_names):
        try:
            same = source(old, name) == source(new, name)
        except (OSError, TypeError):
            continue
        if same:
            identical += 1
            check(f"{name} is unchanged and not declared changed",
                  name not in DECLARED, "declared changed but is byte-identical")
        else:
            changed += 1
            check(f"{name} differs and is declared", name in DECLARED,
                  "differs from the ancestor but is not a declared change")
    print(f"   {identical} methods byte-identical, {changed} changed "
          f"({len(DECLARED)} declared)")


def test_constants():
    print("B. every rule constant and bound")
    for name in sorted(n for n in dir(ancestor) if n.isupper()):
        old = getattr(ancestor, name)
        if not isinstance(old, (str, int, float, tuple)):
            continue
        check(f"module constant {name} unchanged",
              hasattr(shipped, name) and getattr(shipped, name) == old)
    for name in ("CONTEXT_SENTENCES", "JUDGE_BATCH", "COREFERENCE_BATCH",
                 "ASK_ATTEMPTS", "LINKERS"):
        check(f"bound {name} unchanged",
              getattr(shipped.SLinker123, name) == getattr(ancestor.SLinker122, name))
    check("the marks are declared on the class",
          shipped.SLinker123.MARK_LINKED == "linked"
          and shipped.SLinker123.MARK_NAMED_ONLY == "named only")


def prompts(linker, data, linked=()):
    """Every resolver prompt this instance would send. No calls.

    `linked` is passed only to a resolver whose signature takes it — the pilot's arm
    and the ancestor both read the s122 signature, and the point of the comparison is
    that the BYTES agree, not that the call shape does.
    """
    sent = []

    class _Recorder:
        def set_phase(self, phase_name):
            pass

    linker.llm = _Recorder()
    linker._ask = lambda prompt, **_: sent.append(prompt) or {"resolutions": []}
    args = (data["sentences"], data["components"],
            data["name_to_id"], data["sent_map"])
    takes = len(inspect.signature(linker._resolve_references).parameters)
    linker._resolve_references(*args, *((linked,) if takes > 4 else ()))
    return sent


def test_is_the_measured_arm():
    print("C. the shipped file IS `coref_annot_pilots.Annot`, over five projects")
    for project in sorted(DATASETS):
        data = load(project, RUN)
        linked = data["name_links_by_name"]

        mine = shipped.SLinker123.__new__(shipped.SLinker123)
        mine.doc_knowledge = data["knowledge"]

        arm = ARMS["annot"].__new__(ARMS["annot"])
        arm.doc_knowledge = data["knowledge"]
        arm.name_links = linked

        base = ancestor.SLinker122.__new__(ancestor.SLinker122)
        base.doc_knowledge = data["knowledge"]

        got = prompts(mine, data, linked)
        want = prompts(arm, data, ())          # the arm reads `self.name_links`
        head = prompts(base, data, ())         # the ancestor takes no `linked`
        check(f"{project}: same number of calls as the arm", len(got) == len(want))
        check(f"{project}: byte-identical to the measured arm", got == want)
        check(f"{project}: differs from the ancestor", got != head)
        marked = sum(p.count(", linked)") + p.count(", named only)") for p in got)
        plain = sum(p.count("NAMED BEFORE THIS CASE:") for p in got)
        check(f"{project}: every case carries a marked shortlist", marked > 0)
        print(f"   {project:<14} {len(got)} calls, {plain} cases, "
              f"{marked} marked entries")


def test_empty_linked_is_safe():
    print("D. with nothing linked, every entry reads `named only` and nothing breaks")
    data = load("mediastore", RUN)
    empty = shipped.SLinker123.__new__(shipped.SLinker123)
    empty.doc_knowledge = data["knowledge"]
    got = prompts(empty, data, ())
    check("no entry claims `linked`", all(", linked)" not in p for p in got))
    check("entries are still listed", any(", named only)" in p for p in got))


def test_prompt_english_unchanged():
    print("E. no authored English changed: the rule text is the ancestor's")
    data = load("mediastore", RUN)
    mine = shipped.SLinker123.__new__(shipped.SLinker123)
    mine.doc_knowledge = data["knowledge"]
    base = ancestor.SLinker122.__new__(ancestor.SLinker122)
    base.doc_knowledge = data["knowledge"]
    got = prompts(mine, data, data["name_links_by_name"])[0]
    head = prompts(base, data, ())[0]
    # Strip the shortlist lines from both; what is left must be equal byte for byte.
    strip = lambda text: "\n".join(
        line for line in text.splitlines()
        if not line.startswith("NAMED BEFORE THIS CASE:"))
    check("every non-shortlist byte of the resolver prompt is unchanged",
          strip(got) == strip(head))
    check("the explanatory paragraph is untouched",
          "That list has" in got and "the weaker antecedent" not in got)


def main() -> int:
    test_methods()
    test_constants()
    test_is_the_measured_arm()
    test_empty_linked_is_safe()
    test_prompt_english_unchanged()
    print(f"\n{PASSED}/{PASSED + FAILED} checks passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
