"""`s_linker78` is the elegance round's head — its structural claims, asserted.

The head is claimed to be the smallest artifact this branch has produced that is not worse
than what it was cut from. Three of those words are checkable here; the fourth (`not
worse`) is the E2E in `../results/finetune_round/README.md`.

    T1  the deterministic layer is ONE row of the relation, and `_add_scan` is gone
    T2  no prompt enumerates: no numbered conditions, no named document shapes, no syntax,
        no bespoke restatement of the qualified-name distinction
    T3  the two grounds the enumeration rested on are carried by clauses in the same
        prompt, each stated once
    T4  every judging asymmetry that was measured as load-bearing survives
    T5  the chain to `s_linker75` is exactly two cuts: `SCANS`/`_add_scan`/the extraction
        paragraph, and the rubric. Nothing else in any method body moved.

    ../.venv/bin/python pilot/test_s78_head.py
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

import llm_sad_sam.linkers.experimental.s_linker75 as L75              # noqa: E402
import llm_sad_sam.linkers.experimental.s_linker78 as L78              # noqa: E402

PASS, FAIL = [], []

AUTHORED = [c for c in dir(L78) if c.isupper() and isinstance(getattr(L78, c), str)]
SYNTAX = re.compile(r"[Xx]\.[Yy]")
BESPOKE = re.compile(r"code-level (path|identifier)", re.I)
NUMBERED = re.compile(r"\(\d\)")


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{('  — ' + detail) if detail else ''}")


def prompts(mod, cls):
    class _Sent:
        number = 1
    names, cases, docs = ["A", "B"], ["Case 1: x"], ["S1: t"]
    return {
        "extraction": cls._prompt_extraction(names, ["'a' -> B"], []),
        "full-name judging": cls._prompt_validation(names, cases, mod.P1_FOCUS),
        "coreference judging": cls._prompt_validation(
            names, cases, mod.COREF_VALIDATION_FOCUS, strict=True),
        "alias extraction": cls._prompt_doc_knowledge_extract(names, docs),
        "alias judging": cls._prompt_doc_knowledge_judge(names, ["'a' -> B"]),
        "denotation": cls._prompt_denotation(names, cases) if hasattr(
            cls, "_prompt_denotation") else "",
        "coreference resolution": cls._prompt_coref(
            names, [{"sent": _Sent(), "context": [">>> S1: t"]}]),
    }


def t1():
    print("\nT1  one row, no scan machinery")
    check("`SCANS` has exactly one row", list(L78.SCANS) == ["name_word"],
          ", ".join(L78.SCANS))
    check("`_add_scan` is gone", not hasattr(L78.SLinker78, "_add_scan"))
    check("`_name_spans` and all four `NameForm` values remain",
          hasattr(L78.SLinker78, "_name_spans") and len(list(L78.NameForm)) == 4)
    row = L78.SCANS["name_word"]
    check("the row keeps the two options the fold round refused to move",
          row.unique_owner and row.skip_when_named)
    check("the extraction prompt states the recall floor the tight rows drew",
          "however incidental the mention" in L78.ENTITY_EXTRACTION_RULES
          and "spacing, hyphenation or compound joining" in L78.ENTITY_EXTRACTION_RULES)


def t2():
    print("\nT2  nothing enumerates")
    check("the rubric has no numbered conditions",
          not NUMBERED.search(L78.LAYERED_ENTITY_RULES))
    check("the rubric names no document shape",
          not any(w in L78.LAYERED_ENTITY_RULES for w in ("heading", "list")))
    for label, text in prompts(L78, L78.SLinker78).items():
        if not text:
            continue
        check(f"{label}: no spelled syntax", not SYNTAX.search(text))
        check(f"{label}: no bespoke restatement", not BESPOKE.search(text))


def t3():
    print("\nT3  the enumeration's grounds are stated once each")
    judging = L78.SLinker78._prompt_validation(["A"], ["Case 1: x"], L78.P1_FOCUS)
    check("`QUALIFIED_CLAUSE` carries old condition (1)",
          judging.count(L78.QUALIFIED_CLAUSE) == 1)
    check("`STRICTER_CLAUSE` carries old conditions (3) and (4)",
          judging.count(L78.STRICTER_CLAUSE) == 1)
    check("negation is the principle's own last clause",
          "denies what it would otherwise say" in L78.LAYERED_ENTITY_RULES)
    check("the standard of proof is unchanged",
          L78.LAYERED_ENTITY_RULES.startswith("Approve the link by default")
          and "positive ground" in L78.LAYERED_ENTITY_RULES)
    strict = L78.SLinker78._prompt_validation(
        ["A"], ["Case 1: x"], L78.COREF_VALIDATION_FOCUS, strict=True)
    check("the coreference prompt carries neither clause — its cases contain no name",
          L78.QUALIFIED_CLAUSE not in strict and L78.STRICTER_CLAUSE not in strict)


def t4():
    print("\nT4  the measured asymmetries survive")
    check("full-name gate approves by default",
          L78.LAYERED_ENTITY_RULES.startswith("Approve the link by default"))
    check("coreference gate rejects when uncertain",
          L78.LAYERED_COREF_RULES.endswith("When uncertain, reject."))
    check("alias judge approves when uncertain (a third of MediaStore)",
          L78.DOC_KNOWLEDGE_JUDGE_RULES.endswith("When uncertain, prefer APPROVE."))
    check("claim-before-verdict survives (worth 35.2 TP)",
          "quote the EXACT words" in L78.SLinker78._prompt_validation(
              ["A"], ["Case 1: x"], L78.P1_FOCUS))
    check("two focused judging passes survive (FP 4.8 vs 8.3 merged)",
          L78.P1_FOCUS != L78.P2_FOCUS)
    check("the mention label is still computed in code (-10.7 TP without it)",
          hasattr(L78.SLinker78, "_classify_mention_typed"))


def t5():
    print("\nT5  exactly two cuts from s75")
    changed_consts = sorted(c for c in AUTHORED
                            if getattr(L75, c, None) != getattr(L78, c))
    check("two authored constants differ from s75's",
          changed_consts == ["ENTITY_EXTRACTION_RULES", "LAYERED_ENTITY_RULES"],
          ", ".join(changed_consts))
    moved, same = [], 0
    for attr, obj in vars(L75.SLinker75).items():
        if not callable(getattr(obj, "__func__", obj)):
            continue
        other = vars(L78.SLinker78).get(attr)
        if other is None:
            moved.append(f"{attr} (deleted)")
            continue
        a = inspect.getsource(getattr(obj, "__func__", obj)).replace("SLinker75", "X")
        b = inspect.getsource(getattr(other, "__func__", other)).replace(
            "SLinker78", "X")
        if a != b:
            moved.append(attr)
        else:
            same += 1
    check("only `_run_full_name_linker` changed and only `_add_scan` was deleted",
          sorted(moved) == ["_add_scan (deleted)", "_prompt_validation",
                            "_run_full_name_linker"], ", ".join(sorted(moved)))
    check(f"all {same} other method bodies identical to s75's", True)
    attrs = {k: v for k, v in vars(L75.SLinker75).items()
             if not callable(getattr(v, "__func__", v)) and not k.startswith("__")}
    bad = [k for k, v in attrs.items()
           if k != "_VARIANT_NAME" and vars(L78.SLinker78).get(k) != v]
    check(f"all {len(attrs) - 1} class attributes identical (bounds untouched)",
          not bad, ", ".join(bad))


def main():
    print("\ns_linker78 — the elegance round's head\n" + "=" * 62)
    t1(); t2(); t3(); t4(); t5()
    print("\n" + "=" * 62)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"    FAILED: {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
