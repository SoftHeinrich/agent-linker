"""`s_linker70` is the arm that was measured, and nothing else changed.

`pilot/fold_pilots.py --pilot foldstricter` measured a *dynamically built* arm: an
`SLinker69` subclass with `skip_stricter` switched off on the spelling row and
`STRICTER_CLAIM`-style text appended to the full-name judging prompt. `s_linker70` is
that arm written out as a module, plus one further deletion the audit proved inert.
Between the measurement and the shipped file there are two places to be wrong, and
both are checked here rather than asserted in a comment:

    T1  the judging prompt s70 renders is byte-identical to the arm's, on both
        rubrics, and the clause reaches the lenient rubric only
    T2  the candidate sets s70's three rows produce are byte-identical to the arm's on
        all five projects -- which is also the proof that dropping `unique_owner` from
        the spelling row is the identity `pilot/gate_inventory.py` priced at 0 pairs
    T3  nothing else moved: every other method body and class attribute is identical
        to s69's, and the clause introduces no benchmark vocabulary (GATE-06)
    T4  the two remaining gates are on one row, and the judge that row reports to is
        shown neither the target nor the catalog -- the fold law's terminus, checked
        against the prompt builder rather than believed

    ../.venv/bin/python pilot/test_s70_stricter.py
"""
from __future__ import annotations

import inspect
import re
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import PROJECTS, load_project                        # noqa: E402
import llm_sad_sam.linkers.experimental.s_linker69 as L69              # noqa: E402
import llm_sad_sam.linkers.experimental.s_linker70 as L70              # noqa: E402
from fold_pilots import STRICTER_CLAUSE, fullname_judge_arm            # noqa: E402

PASS, FAIL = [], []


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{('  — ' + detail) if detail else ''}")


class P70(L70.SLinker70):
    def __init__(self):                                                # noqa: D107
        self.doc_knowledge = type("K", (), {"aliases": {}})()


class P69(L69.SLinker69):
    def __init__(self):                                                # noqa: D107
        self.doc_knowledge = type("K", (), {"aliases": {}})()


def t1():
    print("\nT1  the judging prompt is the arm's, byte for byte")
    arm = fullname_judge_arm("t70", extra=STRICTER_CLAUSE)
    names = ["Alpha", "Beta Gamma"]
    cases = ["Case 1: S3 | Alpha | ...", "Case 2: S9 | Beta Gamma | ..."]
    for focus in ("Check architectural participation: ...", "Another focus"):
        for strict in (False, True):
            a = arm._prompt_validation(names, cases, focus, strict)
            b = L70.SLinker70._prompt_validation(names, cases, focus, strict)
            check(f"rendered prompt identical (strict={strict}, focus={focus[:18]!r})",
                  a == b, f"{len(a)} vs {len(b)} chars")
    lenient = L70.SLinker70._prompt_validation(names, cases, "f", False)
    strict = L70.SLinker70._prompt_validation(names, cases, "f", True)
    check("clause is in the lenient rubric", STRICTER_CLAUSE in lenient)
    check("clause is NOT in the coreference rubric", STRICTER_CLAUSE not in strict)
    s69_strict = L69.SLinker69._prompt_validation(names, cases, "f", True)
    check("coreference prompt unchanged from s69", strict == s69_strict)


def t2():
    print("\nT2  the candidate sets are the arm's, on all five projects")
    p70, p69 = P70(), P69()
    arm_spelling = replace(L69.SCANS["spelling"], skip_stricter=False)
    totals = {"spelling": 0, "stated_name": 0, "name_word": 0}
    diffs = {"spelling": 0, "stated_name": 0, "name_word": 0}
    gate_free = 0
    for name in PROJECTS:
        info = load_project(name)
        args = (info["sentences"], info["components"])

        def pairs(probe, scan):
            return {(c.sentence_number, c.component_id, c.matched_text, c.source)
                    for c in probe._scan(*args, scan)}

        # the arm: skip_stricter off, unique_owner still on
        a = pairs(p69, arm_spelling)
        b = pairs(p70, L70.SCANS["spelling"])
        totals["spelling"] += len(b)
        diffs["spelling"] += len(a ^ b)
        # the gate's own effect, for the record
        gate_free += len(b) - len(pairs(p69, L69.SCANS["spelling"]))
        for row in ("stated_name", "name_word"):
            x = pairs(p69, L69.SCANS[row])
            y = pairs(p70, L70.SCANS[row])
            totals[row] += len(y)
            diffs[row] += len(x ^ y)
    for row in totals:
        check(f"row {row!r}: {totals[row]} candidates, identical to the arm's",
              diffs[row] == 0, f"{diffs[row]} differing")
    check("dropping `unique_owner` from the spelling row is an identity",
          diffs["spelling"] == 0,
          "0 pairs freed, so the arm's option and its absence agree")
    print(f"        for the record: the folded gate admits {gate_free} more pairs "
          f"across the five projects than s69's spelling row")


def t3():
    print("\nT3  nothing else moved")
    skip = {"_prompt_validation", "_scan"}
    same = 0
    for attr, obj in vars(L69.SLinker69).items():
        if attr in skip or not callable(getattr(obj, "__func__", obj)):
            continue
        other = vars(L70.SLinker70).get(attr)
        if other is None:
            check(f"method {attr} present in s70", False)
            continue
        # the banner line prints the class name; normalize it, it is not behaviour
        a = inspect.getsource(getattr(obj, "__func__", obj)).replace(
            "SLinker69", "SLinker70")
        b = inspect.getsource(getattr(other, "__func__", other))
        if a != b:
            check(f"method {attr} identical", False)
        else:
            same += 1
    check(f"all {same} other method bodies identical to s69's", True)
    attrs = {k: v for k, v in vars(L69.SLinker69).items()
             if not callable(getattr(v, "__func__", v)) and not k.startswith("__")}
    bad = [k for k, v in attrs.items()
           if k != "_VARIANT_NAME" and vars(L70.SLinker70).get(k) != v]
    check(f"all {len(attrs) - 1} class attributes identical (bar _VARIANT_NAME)",
          not bad, ", ".join(bad))
    # GATE-06: the clause may name no component of any benchmark.
    catalog = {c.name.lower() for name in PROJECTS
               for c in load_project(name)["components"]}
    words = set(re.findall(r"[a-z]+", STRICTER_CLAUSE.lower()))
    hits = sorted(w for w in words if w in catalog)
    check("clause carries no benchmark component name (GATE-06)", not hits,
          ", ".join(hits))
    check("clause names no surface form and no capitalization rule as a boundary",
          "evidence" in STRICTER_CLAUSE and "neither settles it" in STRICTER_CLAUSE)


def t4():
    print("\nT4  the fold law's terminus, read off the prompt builders")
    left = {row: [g for g in ("unique_owner", "skip_when_named")
                  if getattr(scan, g)]
            for row, scan in L70.SCANS.items()}
    check("every remaining gate is on the partial-name row",
          not left["stated_name"] and not left["spelling"] and left["name_word"],
          f"{left}")
    check("`SurfaceScan` no longer carries `skip_stricter`",
          not hasattr(L70.SCANS["spelling"], "skip_stricter"))
    # the partial-name row reports to `_classify_denotations`; that prompt must show
    # neither the target nor the catalog, which is why its two gates cannot be asked.
    src = inspect.getsource(L70.SLinker70._classify_denotations)
    check("the denotation prompt is built without the component catalog",
          "comp_names" not in src, "catalog would make `unique_owner` askable")
    check("the denotation prompt is built without the target component",
          "component_name" not in src,
          "the target would make `skip_when_named` askable")
    fn = inspect.getsource(L70.SLinker70._prompt_validation)
    check("the full-name judge IS shown the catalog (why the two folds worked)",
          "COMPONENTS: {', '.join(comp_names)}" in fn)


def main():
    print("\ns_linker70 — the measured arm, written out\n" + "=" * 62)
    t1(); t2(); t3(); t4()
    print("\n" + "=" * 62)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"    FAILED: {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
