"""`s_linker75` changes four authored spans and nothing else — asserted.

s74 removed the one span the general round's bar caught in the judging path. What it
left standing was four more places where the same distinction is restated in a bespoke
wording, and one clause that still spells `X.Y or X.Y.Z` outright. s75 removes all four.
This file pins the claim that makes the end-to-end number attributable: **only the
authored English changed**, the deterministic layer is byte-identical, and the two spans
three rounds measured as load-bearing are still there.

    T1  exactly the four claimed constants differ from s74's; every other authored
        constant is byte-identical
    T2  no authored constant in the module spells an identifier syntax any more, and
        `QUALIFIED_CLAUSE` is the single sentence that carries the distinction
    T3  the spans measured as load-bearing survive: the four numbered reject-conditions
        (s71/s72, -0.8 F1 without them), "a heading, or a list" (s73, -2.7 TP), the
        approve-by-default standard of proof, the coreference rubric's reject-when-
        uncertain tie-break, and the alias judge's opposite one
    T4  the distinction is stated exactly once per prompt that needs it — no prompt
        carries both `QUALIFIED_CLAUSE` and a restatement of it
    T5  no method body, class attribute or `SCANS` row moved: the deterministic layer
        is s74's

    ../.venv/bin/python pilot/test_s75_nofinetune.py
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

import llm_sad_sam.linkers.experimental.s_linker74 as L74              # noqa: E402
import llm_sad_sam.linkers.experimental.s_linker75 as L75              # noqa: E402

PASS, FAIL = [], []

AUTHORED = ["DOC_KNOWLEDGE_EXTRACTION_RULES", "DOC_KNOWLEDGE_JUDGE_RULES",
            "ALIAS_EXCLUSION_RULES", "ENTITY_EXTRACTION_RULES", "P1_FOCUS",
            "P2_FOCUS", "COREF_VALIDATION_FOCUS", "COREF_RULES",
            "LAYERED_ENTITY_RULES", "LAYERED_COREF_RULES", "QUALIFIED_CLAUSE",
            "STRICTER_CLAUSE"]

#: The four spans this variant changes, and the reason each is changed.
CHANGED = {
    "ALIAS_EXCLUSION_RULES": "spelled X.Y or X.Y.Z (the last one)",
    "ENTITY_EXTRACTION_RULES": "carried a bespoke 'code-level path' clause",
    "P1_FOCUS": "carried a bespoke 'code-level identifier' tail",
    "LAYERED_COREF_RULES": "carried a fifth restatement of the same distinction",
}

SYNTAX = re.compile(r"[Xx]\.[Yy]")
#: The bespoke wordings, in any of the forms the five copies used.
BESPOKE = re.compile(r"code-level (path|identifier)", re.I)


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{('  — ' + detail) if detail else ''}")


def t1():
    print("\nT1  four spans changed, and they are the four claimed")
    changed = sorted(c for c in AUTHORED if getattr(L74, c) != getattr(L75, c))
    check("exactly the four claimed constants differ from s74",
          changed == sorted(CHANGED), ", ".join(changed) or "none")
    for name in sorted(CHANGED):
        print(f"        {name}: {CHANGED[name]}")
    untouched = [c for c in AUTHORED if c not in CHANGED]
    check(f"the other {len(untouched)} authored constants are byte-identical",
          all(getattr(L74, c) == getattr(L75, c) for c in untouched))


def t2():
    print("\nT2  no authored constant names a syntax or a bespoke shape")
    spelled = [c for c in AUTHORED if SYNTAX.search(getattr(L75, c))]
    check("no constant spells an identifier syntax", not spelled,
          ", ".join(spelled) or "none")
    bespoke = [c for c in AUTHORED
               if c != "QUALIFIED_CLAUSE" and BESPOKE.search(getattr(L75, c))]
    check("no constant restates the distinction in its own wording", not bespoke,
          ", ".join(bespoke) or "none")
    check("`QUALIFIED_CLAUSE` is unchanged and is the one carrier",
          L75.QUALIFIED_CLAUSE == L74.QUALIFIED_CLAUSE
          and "longer joined or dotted identifier" in L75.QUALIFIED_CLAUSE)
    check("the alias prohibition survives without the shape",
          "not an alias" in L75.ALIAS_EXCLUSION_RULES
          and not SYNTAX.search(L75.ALIAS_EXCLUSION_RULES))


def t3():
    print("\nT3  every span measured as load-bearing survives")
    rules = L75.LAYERED_ENTITY_RULES
    check("the rubric is byte-identical to s74's", rules == L74.LAYERED_ENTITY_RULES)
    check("the four numbered reject-conditions survive (s71/s72: -0.8 F1 without them)",
          all(f"({i})" in rules for i in (1, 2, 3, 4)))
    check("'a heading, or a list' survives (s73: -2.7 TP in each of 3 runs)",
          "a heading, or a list" in rules)
    check("approve-by-default is unchanged",
          rules.startswith("Approve the link by default")
          and rules.endswith("clearly applies, approve."))
    check("the coreference rubric still rejects when uncertain",
          L75.LAYERED_COREF_RULES.endswith("When uncertain, reject."))
    check("the alias judge still approves when uncertain (a third of MediaStore)",
          L75.DOC_KNOWLEDGE_JUDGE_RULES.endswith("When uncertain, prefer APPROVE."))
    check("the referring-expression and ambiguity criteria survive",
          "genuine referring expression" in L75.LAYERED_COREF_RULES
          and "could equally be a different" in L75.LAYERED_COREF_RULES)


def _render(mod, cls):
    """Every prompt this module builds, on inert stand-in data."""
    class _Sent:
        number = 1
    names, cases, docs = ["A", "B"], ["Case 1: x"], ["S1: t"]
    coref_cases = [{"sent": _Sent(), "context": [">>> S1: t"]}]
    return {
        "extraction": cls._prompt_extraction(names, ["'a' -> B"], []),
        "full-name judging": cls._prompt_validation(names, cases, mod.P1_FOCUS),
        "coreference judging": cls._prompt_validation(
            names, cases, mod.COREF_VALIDATION_FOCUS, strict=True),
        "alias extraction": cls._prompt_doc_knowledge_extract(names, docs),
        "alias judging": cls._prompt_doc_knowledge_judge(names, ["'a' -> B"]),
        "coreference resolution": cls._prompt_coref(names, coref_cases),
    }


def t4():
    print("\nT4  one sentence per distinction, per prompt")
    rendered = _render(L75, L75.SLinker75)
    for label, text in rendered.items():
        check(f"{label}: no spelled syntax", not SYNTAX.search(text))
        check(f"{label}: no bespoke restatement",
              not BESPOKE.search(text))
        check(f"{label}: the general clause appears at most once",
              text.count(L75.QUALIFIED_CLAUSE) <= 1)
    carriers = [l for l, t in rendered.items() if L75.QUALIFIED_CLAUSE in t]
    print(f"        prompts carrying `QUALIFIED_CLAUSE`: {', '.join(carriers)}")
    print("        (the rubric states the same ground inside reject-condition (1), so")
    print("         the full-name prompt does not also carry the clause)")


def t5():
    print("\nT5  the deterministic layer is s74's")
    same = 0
    for attr, obj in vars(L74.SLinker74).items():
        if not callable(getattr(obj, "__func__", obj)):
            continue
        other = vars(L75.SLinker75).get(attr)
        if other is None:
            check(f"method {attr} present", False)
            continue
        a = inspect.getsource(getattr(obj, "__func__", obj)).replace(
            "SLinker74", "SLinker75")
        b = inspect.getsource(getattr(other, "__func__", other))
        if a != b:
            # `_prompt_extraction` is the one body that changes: it gains the clause.
            if attr == "_prompt_extraction":
                check("method _prompt_extraction differs by the added clause only",
                      b.replace("\n{QUALIFIED_CLAUSE}\n", "") == a)
                continue
            check(f"method {attr} identical", False)
        else:
            same += 1
    check(f"all {same} other method bodies identical to s74's", True)
    attrs = {k: v for k, v in vars(L74.SLinker74).items()
             if not callable(getattr(v, "__func__", v)) and not k.startswith("__")}
    bad = [k for k, v in attrs.items()
           if k != "_VARIANT_NAME" and vars(L75.SLinker75).get(k) != v]
    check(f"all {len(attrs) - 1} class attributes identical", not bad, ", ".join(bad))

    def rows(mod):
        return {k: tuple(sorted((f, getattr(x, "value", x))
                                for f, x in vars(v).items()))
                for k, v in mod.SCANS.items()}
    check("SCANS rows identical — no gate added, moved or removed",
          rows(L75) == rows(L74))
    check("the inflection list is unchanged and is still the only word list",
          L75.INFLECTIONS == L74.INFLECTIONS)


def main():
    print("\ns_linker75 — the authored surface, and nothing else\n" + "=" * 62)
    t1(); t2(); t3(); t4(); t5()
    print("\n" + "=" * 62)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"    FAILED: {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
