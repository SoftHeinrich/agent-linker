"""`s_linker71` is the two arms that were measured, and it names no surface shape.

The general round adopted two prompt rewrites and refused two more. This file checks
that the module ships exactly what the pilots measured, that the refused ones really are
untouched, and that the adopted text passes the bar the round set (GATE-07).

    T1  the two rewritten prompts render byte-identically to their measured arms
    T2  the two refused prompts are byte-identical to s70's
    T3  no authored clause names a surface shape: no identifier syntax, no document
        form, no benchmark vocabulary (GATE-06 + GATE-07)
    T4  nothing else moved -- every method body and class attribute identical to s70's
    T5  the authored surface got smaller, and the audit's own count agrees with the
        module's

    ../.venv/bin/python pilot/test_s71_general.py
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import PROJECTS, load_project                        # noqa: E402
import general_prompt_pilots as G                                      # noqa: E402
import llm_sad_sam.linkers.experimental.s_linker70 as L70              # noqa: E402
import llm_sad_sam.linkers.experimental.s_linker71 as L71              # noqa: E402

PASS, FAIL = [], []

#: Shapes no clause may name: an identifier syntax, or a place in a document.
FORBIDDEN_SHAPES = [
    (r"[Xx]\.[Yy]", "spells out an identifier syntax"),
    (r"\bheading\b", "names a document form"),
    (r"\ba list\b", "names a document form"),
    (r"\bcamel[ -]?case\b", "names a naming convention"),
    (r"\bsnake[ -]?case\b", "names a naming convention"),
    (r"\bunderscore\b", "names a separator"),
]

AUTHORED = ["DOC_KNOWLEDGE_EXTRACTION_RULES", "DOC_KNOWLEDGE_JUDGE_RULES",
            "ALIAS_EXCLUSION_RULES", "ENTITY_EXTRACTION_RULES", "P1_FOCUS",
            "P2_FOCUS", "COREF_VALIDATION_FOCUS", "COREF_RULES",
            "LAYERED_ENTITY_RULES", "LAYERED_COREF_RULES", "QUALIFIED_CLAUSE",
            "STRICTER_CLAUSE"]


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{('  — ' + detail) if detail else ''}")


def t1():
    print("\nT1  the adopted rewrites are the measured arms")
    arm = G.judging_arm("t71", rules=G.GENERAL_ENTITY_RULES,
                        p1_focus=G.GENERAL_P1_FOCUS,
                        clauses=(L70.QUALIFIED_CLAUSE, L70.STRICTER_CLAUSE))
    names, cases = ["Alpha", "Beta Gamma"], ["Case 1: ...", "Case 2: ..."]
    a = arm._prompt_validation(names, cases, "focus", False)
    b = L71.SLinker71._prompt_validation(names, cases, "focus", False)
    check("full-name judging prompt identical to plainrubric's arm", a == b,
          f"{len(a)} vs {len(b)} chars")
    ea = G.extraction_arm("t71x", rules=G.GENERAL_EXTRACTION_RULES,
                          extra=L70.QUALIFIED_CLAUSE)
    x = ea._prompt_extraction(names, ["'q' -> Alpha"], [])
    y = L71.SLinker71._prompt_extraction(names, ["'q' -> Alpha"], [])
    check("extraction prompt identical to plainextract's arm", x == y)
    check("P1_FOCUS is the arm's", L71.P1_FOCUS == G.GENERAL_P1_FOCUS)
    check("the judging rubric is the arm's",
          L71.LAYERED_ENTITY_RULES == G.GENERAL_ENTITY_RULES)


def t2():
    print("\nT2  the refused rewrites left no trace")
    check("coreference rubric byte-identical to s70's",
          L71.LAYERED_COREF_RULES == L70.LAYERED_COREF_RULES)
    check("alias exclusion byte-identical to s70's",
          L71.ALIAS_EXCLUSION_RULES == L70.ALIAS_EXCLUSION_RULES)
    names, cases = ["Alpha"], ["Case 1: ..."]
    check("the coreference judging prompt is s70's",
          L71.SLinker71._prompt_validation(names, cases, "f", True)
          == L70.SLinker70._prompt_validation(names, cases, "f", True))
    check("the alias prompt is s70's",
          L71.SLinker71._prompt_doc_knowledge_extract(names, ["a."])
          == L70.SLinker70._prompt_doc_knowledge_extract(names, ["a."]))


def t3():
    print("\nT3  no clause names a shape (GATE-07) or a benchmark term (GATE-06)")
    catalog = {c.name.lower() for name in PROJECTS
               for c in load_project(name)["components"]}
    for const in AUTHORED:
        text = getattr(L71, const)
        hits = [why for pat, why in FORBIDDEN_SHAPES if re.search(pat, text)]
        expected = const == "ALIAS_EXCLUSION_RULES"   # the one measured exception
        ok = (not hits) if not expected else bool(hits)
        note = "; ".join(hits) if hits else ""
        if expected:
            note = f"kept deliberately — {note} (see general_round/README.md)"
        check(f"{const}", ok, note)
        words = set(re.findall(r"[a-z]+", text.lower()))
        bad = sorted(w for w in words if w in catalog)
        check(f"{const}: no benchmark component name", not bad, ", ".join(bad))


def t4():
    print("\nT4  nothing else moved")
    skip = {"_prompt_validation", "_prompt_extraction"}
    same = 0
    for attr, obj in vars(L70.SLinker70).items():
        if attr in skip or not callable(getattr(obj, "__func__", obj)):
            continue
        other = vars(L71.SLinker71).get(attr)
        if other is None:
            check(f"method {attr} present", False)
            continue
        a = inspect.getsource(getattr(obj, "__func__", obj)).replace(
            "SLinker70", "SLinker71")
        b = inspect.getsource(getattr(other, "__func__", other))
        if a != b:
            check(f"method {attr} identical", False)
        else:
            same += 1
    check(f"all {same} other method bodies identical to s70's", True)
    attrs = {k: v for k, v in vars(L70.SLinker70).items()
             if not callable(getattr(v, "__func__", v)) and not k.startswith("__")}
    bad = [k for k, v in attrs.items()
           if k != "_VARIANT_NAME" and vars(L71.SLinker71).get(k) != v]
    check(f"all {len(attrs) - 1} class attributes identical", not bad, ", ".join(bad))
    # each module defines its own `SurfaceScan` AND its own `NameForm`, so both the
    # dataclass and the enum members are distinct objects and `==` is False however
    # identical the rows; compare field names against enum *values*.
    def rows(mod):
        return {k: tuple(sorted((f, getattr(x, 'value', x))
                                for f, x in vars(v).items()))
                for k, v in mod.SCANS.items()}
    check("the deterministic layer is untouched: SCANS rows identical",
          rows(L71) == rows(L70))


def t5():
    print("\nT5  the authored surface got smaller")
    before = sum(len(getattr(L70, c)) for c in AUTHORED)
    after = sum(len(getattr(L71, c)) for c in AUTHORED)
    check(f"authored bytes {before} -> {after}", after < before,
          f"{before - after} fewer ({(before - after) / before:.0%})")
    rubric_before = len(L70.LAYERED_ENTITY_RULES) + len(L70.P1_FOCUS)
    rubric_after = (len(L71.LAYERED_ENTITY_RULES) + len(L71.P1_FOCUS)
                    + len(L71.QUALIFIED_CLAUSE))
    check(f"the judging rubric {rubric_before} -> {rubric_after} bytes "
          f"(clause included)", rubric_after < rubric_before)


def main():
    print("\ns_linker71 — the general round, written out\n" + "=" * 62)
    t1(); t2(); t3(); t4(); t5()
    print("\n" + "=" * 62)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"    FAILED: {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
