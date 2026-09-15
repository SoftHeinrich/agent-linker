"""`s_linker126` = `s_linker125` + the antecedent contract, in code. No LLM calls.

s126 makes two claims that have to be checked rather than asserted, because they pull in
opposite directions: it REMOVES prompt text and ADDS a refusal, and a reader has to know
that the refusal is not quietly changing what the resolver is asked.

  A. **The prompt is `s_linker125`'s, byte for byte.** The refusal acts on the reply, not
     on the case, so every resolver prompt and every union-judging prompt is unchanged.
  B. **No shortlist survives**, inherited from `s_linker125`.
  C. **The predicate is the measured one.** It accepts exactly the two `written` values
     the audit found gold in (`whole name`, `short form`) and refuses the two it found
     none in -- checked against `_written_as` over real sentences, not against a literal.
  D. **It is the arm that was priced.** The refusal behaves identically to
     `coref_exact_pilots.RefuseAntecedent`, which read terra FP 13.0 -> 12.3 and luna
     40.3 -> 38.3 at zero TP cost.
  E. **It degrades rather than breaks** when a resolution cites no antecedent sentence --
     that case is `_resolve_references`' guard, not this one's.
  F. **Exactly one method and one predicate are added** over `s_linker125`.

Run:

    /path/to/.venv/bin/python pilot/test_s126.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.linkers.experimental.s_linker125 import SLinker125  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker126 import SLinker126  # noqa: E402

from coref_exact_pilots import NAMES_IT, RefuseAntecedent, load  # noqa: E402
from reading_pilots import DATASETS  # noqa: E402

RUN = ROOT.parent / "results/shortlistmark_e2e_terra_r1_20260914"

DECLARED = {"_validate_coref_links", "_antecedent_names_it"}

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


class _Link:
    """The two fields the predicate reads off a link."""

    def __init__(self, sentence_number, component_id, component_name):
        self.sentence_number = sentence_number
        self.component_id = component_id
        self.component_name = component_name


def main() -> int:
    print("F. exactly one method and one predicate are added over `s_linker125`")
    differing = set()
    for name in set(dir(SLinker126)) | set(dir(SLinker125)):
        if name.startswith("__"):
            continue
        a = getattr(SLinker125, name, None)
        b = getattr(SLinker126, name, None)
        if not callable(b):
            continue
        try:
            if a is None or inspect.getsource(a) != inspect.getsource(b):
                differing.add(name)
        except (OSError, TypeError):
            continue
    check("only the declared additions differ", differing == DECLARED,
          f"differing={sorted(differing)}")
    check("the variant name moved", SLinker126._VARIANT_NAME == "s_linker126")
    check("the accepted forms are the measured ones",
          tuple(SLinker126.ANTECEDENT_FORMS) == tuple(NAMES_IT),
          f"{SLinker126.ANTECEDENT_FORMS} vs {NAMES_IT}")

    for project in sorted(DATASETS):
        data = load(project, RUN)
        mine = resolver_prompts(probe(SLinker126, data), data)
        base = resolver_prompts(probe(SLinker125, data), data)
        print(f"\n{project}")
        check(f"{project}: resolver prompts are s125's byte for byte", mine == base)
        check(f"{project}: no shortlist survives",
              not any(l.startswith("NAMED BEFORE THIS CASE:")
                      for p in mine for l in p.splitlines()))
        jm = judge_prompts(probe(SLinker126, data), data)
        jb = judge_prompts(probe(SLinker125, data), data)
        check(f"{project}: union judging prompts are s125's byte for byte", jm == jb)

        # C / D: the predicate's verdict on every real (sentence, component) pair, against
        # `_written_as` directly and against the arm that was measured.
        linker = probe(SLinker126, data)
        priced = probe(RefuseAntecedent, data)
        agree = expected = 0
        accepted = collections_counter = {}
        for component in data["components"]:
            for sentence in data["sentences"]:
                link = _Link(sentence.number + 1, component.id, component.name)
                meta = {(link.sentence_number, component.id):
                        {"antecedent_sentence": sentence.number}}
                verdict = linker._antecedent_names_it(link, data["sent_map"], meta)
                written = linker._written_as(sentence.text, component.name)
                if verdict == (written in NAMES_IT):
                    expected += 1
                if verdict == priced._antecedent_names_it(
                        link, data["sent_map"], meta):
                    agree += 1
                accepted[written] = accepted.get(written, 0) + int(verdict)
        total = len(data["components"]) * len(data["sentences"])
        check(f"{project}: the predicate IS `_written_as` in NAMES_IT",
              expected == total, f"{expected}/{total}")
        check(f"{project}: identical to the priced arm", agree == total,
              f"{agree}/{total}")
        print(f"   {len(mine)} resolver calls, {len(jm)} judge calls, "
              f"{total} pairs checked; accepted by written: "
              + ", ".join(f"{k}={v}" for k, v in sorted(accepted.items()) if v))

    print("\nE. a resolution citing no antecedent sentence is left to the existing guard")
    data = load("mediastore", RUN)
    linker = probe(SLinker126, data)
    component = data["components"][0]
    link = _Link(3, component.id, component.name)
    check("no antecedent recorded -> not refused here",
          linker._antecedent_names_it(link, data["sent_map"], {}))
    check("antecedent sentence absent from the map -> not refused here",
          linker._antecedent_names_it(
              link, data["sent_map"],
              {(3, component.id): {"antecedent_sentence": 99999}}))

    print(f"\n{PASSED}/{PASSED + FAILED} checks passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
