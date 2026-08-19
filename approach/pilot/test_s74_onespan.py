"""`s_linker74` differs from `s_linker70` in one span, and no clause names a corpus shape.

The general round tried to generalize the authored prompts four times and kept losing
ground; s74 is what survived. This file pins the claim that makes the result reportable:
**exactly one span changed**, so the end-to-end parity (F1 95.60 against 95.74, both n=3,
inside either arm's run range) is attributable to that span and to nothing else.

    T1  one span, and it is the one claimed: the rubric differs from s70's only by the
        identifier-syntax clause, and every other authored constant is byte-identical
    T2  the syntax is gone from the judging path, and the two spans the round measured
        as load-bearing are still there
    T3  the one remaining spelled syntax in the module is the alias prompt's, which is
        kept deliberately and measured
    T4  no method body, class attribute or `SCANS` row moved

    ../.venv/bin/python pilot/test_s74_onespan.py
"""
from __future__ import annotations

import difflib
import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

import llm_sad_sam.linkers.experimental.s_linker70 as L70              # noqa: E402
import llm_sad_sam.linkers.experimental.s_linker74 as L74              # noqa: E402

PASS, FAIL = [], []

AUTHORED = ["DOC_KNOWLEDGE_EXTRACTION_RULES", "DOC_KNOWLEDGE_JUDGE_RULES",
            "ALIAS_EXCLUSION_RULES", "ENTITY_EXTRACTION_RULES", "P1_FOCUS",
            "P2_FOCUS", "COREF_VALIDATION_FOCUS", "COREF_RULES",
            "LAYERED_ENTITY_RULES", "LAYERED_COREF_RULES", "QUALIFIED_CLAUSE",
            "STRICTER_CLAUSE"]

SYNTAX = re.compile(r"[Xx]\.[Yy]")


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{('  — ' + detail) if detail else ''}")


def t1():
    print("\nT1  one span, and it is the one claimed")
    changed = [c for c in AUTHORED if getattr(L70, c) != getattr(L74, c)]
    check("exactly one authored constant differs from s70", changed == ["LAYERED_ENTITY_RULES"],
          ", ".join(changed) or "none")
    a = re.split(r"(?<=[;.]) ", L70.LAYERED_ENTITY_RULES)
    b = re.split(r"(?<=[;.]) ", L74.LAYERED_ENTITY_RULES)
    diff = [d for d in difflib.ndiff(a, b) if d[0] in "+-"]
    check("the difference is one replaced clause", len(diff) == 2,
          f"{len(diff)} changed fragments")
    for d in diff:
        print(f"        {d[0]} {d[2:][:96]}")


def t2():
    print("\nT2  the syntax is gone; the load-bearing spans are not")
    check("no identifier syntax in the judging rubric",
          not SYNTAX.search(L74.LAYERED_ENTITY_RULES))
    check("the four numbered reject-conditions survive (s71/s72: -0.8 F1 without them)",
          all(f"({i})" in L74.LAYERED_ENTITY_RULES for i in (1, 2, 3, 4)))
    check("'a heading, or a list' survives (s73: -2.7 TP in each of 3 runs without it)",
          "a heading, or a list" in L74.LAYERED_ENTITY_RULES)
    check("the approve-by-default standard of proof is unchanged",
          L74.LAYERED_ENTITY_RULES.startswith("Approve the link by default")
          and L74.LAYERED_ENTITY_RULES.endswith("clearly applies, approve."))
    check("condition (1) now states the compositional ground, not the shape",
          "longer joined or dotted identifier" in L74.LAYERED_ENTITY_RULES)


def t3():
    print("\nT3  the one spelled syntax left in the module")
    spelled = [c for c in AUTHORED if SYNTAX.search(getattr(L74, c))]
    check("exactly one authored constant still spells a syntax",
          spelled == ["ALIAS_EXCLUSION_RULES"], ", ".join(spelled) or "none")
    print("        kept by measurement, not by oversight: every general rewording admits")
    print("        the same 0 identifier fragments but grows the alias table from 24.0 to")
    print("        36.7-37.3 terms per run (../results/general_round/README.md).")


def t4():
    print("\nT4  nothing else moved")
    same = 0
    for attr, obj in vars(L70.SLinker70).items():
        if not callable(getattr(obj, "__func__", obj)):
            continue
        other = vars(L74.SLinker74).get(attr)
        if other is None:
            check(f"method {attr} present", False)
            continue
        a = inspect.getsource(getattr(obj, "__func__", obj)).replace(
            "SLinker70", "SLinker74")
        b = inspect.getsource(getattr(other, "__func__", other))
        if a != b:
            check(f"method {attr} identical", False)
        else:
            same += 1
    check(f"all {same} method bodies identical to s70's", True)
    attrs = {k: v for k, v in vars(L70.SLinker70).items()
             if not callable(getattr(v, "__func__", v)) and not k.startswith("__")}
    bad = [k for k, v in attrs.items()
           if k != "_VARIANT_NAME" and vars(L74.SLinker74).get(k) != v]
    check(f"all {len(attrs) - 1} class attributes identical", not bad, ", ".join(bad))

    def rows(mod):
        return {k: tuple(sorted((f, getattr(x, "value", x))
                                for f, x in vars(v).items()))
                for k, v in mod.SCANS.items()}
    check("SCANS rows identical — the deterministic layer is untouched",
          rows(L74) == rows(L70))


def main():
    print("\ns_linker74 — one span\n" + "=" * 62)
    t1(); t2(); t3(); t4()
    print("\n" + "=" * 62)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"    FAILED: {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
