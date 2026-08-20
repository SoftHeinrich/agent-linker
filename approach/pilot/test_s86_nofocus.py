"""`s_linker86` differs from `s_linker85` in one deleted restatement and nothing else.

    T1  the constant is gone, and no authored string of the module mentions it
    T2  every other rule constant is byte-identical to s85's
    T3  every method body is byte-identical to s85's apart from the two that carry the
        deletion (`_validate_with_evidence`, which stops passing a focus, and
        `_prompt_validation`, which stops printing a dangling space)
    T4  the two prompt builders render byte-identically to s85's once s85's focus is
        substituted back: the lenient judging prompt with the focus restored, and the
        strict one unchanged, because the coreference judge keeps its question
    T5  what the deletion removes is exactly the focus sentence and the space in front
        of it -- 244 B per judging call, nothing else
    T6  the resource bounds and the deterministic layer are untouched
    T7  GATE-06: no benchmark component name appears in any authored string

    ../.venv/bin/python pilot/test_s86_nofocus.py
"""
from __future__ import annotations

import csv
import glob
import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

import llm_sad_sam.linkers.experimental.s_linker85 as L85              # noqa: E402
import llm_sad_sam.linkers.experimental.s_linker86 as L86              # noqa: E402

BASE = "/mnt/hostshare/ardoco-home/alinker-replication-package"
PASS, FAIL = [], []

#: The two bodies the deletion is allowed to touch.
CHANGED = {"_validate_with_evidence", "_prompt_validation"}

#: The names that differ by construction and say nothing about behaviour.
RENAMED = {"_VARIANT_NAME"}


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{('  — ' + detail) if detail else ''}")


def constants(mod):
    return {n: getattr(mod, n) for n in dir(mod)
            if n.isupper() and isinstance(getattr(mod, n), str)}


def bodies(cls):
    out = {}
    for name, member in vars(cls).items():
        fn = member.__func__ if isinstance(member, staticmethod) else member
        if callable(fn):
            try:
                out[name] = inspect.getsource(fn)
            except (OSError, TypeError):
                pass
    return out


def main():
    old, new = L85.SLinker85, L86.SLinker86

    print("T1  the constant is gone")
    check("VALIDATION_FOCUS not defined in s86",
          not hasattr(L86, "VALIDATION_FOCUS"))
    check("no authored string carries the focus sentence",
          not any("architectural participation and referential specificity" in v
                  for v in constants(L86).values()))
    check("s85 had it", hasattr(L85, "VALIDATION_FOCUS"))

    print("T2  every other rule constant is byte-identical")
    old_c, new_c = constants(L85), constants(L86)
    for name, value in old_c.items():
        if name in RENAMED or name == "VALIDATION_FOCUS":
            continue
        check(f"{name} identical", new_c.get(name) == value)
    check("s86 introduces no new authored constant",
          not (set(new_c) - set(old_c) - RENAMED),
          ", ".join(sorted(set(new_c) - set(old_c) - RENAMED)))

    print("T3  every method body but the two identical")
    ob, nb = bodies(old), bodies(new)
    check("same method set", set(ob) == set(nb),
          f"{sorted(set(ob) ^ set(nb))}")
    for name in sorted(set(ob) & set(nb)):
        if name in CHANGED:
            check(f"{name} differs (expected)", ob[name] != nb[name])
        else:
            check(f"{name} identical", ob[name] == nb[name])

    print("T4  the builders render identically once the focus is substituted back")
    names, cases = ["A", "B"], ["Case 1: x"]
    lenient85 = old._prompt_validation(names, cases, L85.VALIDATION_FOCUS)
    lenient86 = new._prompt_validation(names, cases, L85.VALIDATION_FOCUS)
    check("lenient prompt identical with the focus supplied", lenient85 == lenient86)
    strict85 = old._prompt_validation(names, cases, L85.COREF_VALIDATION_FOCUS,
                                      strict=True)
    strict86 = new._prompt_validation(names, cases, L86.COREF_VALIDATION_FOCUS,
                                      strict=True)
    check("strict prompt identical", strict85 == strict86)
    check("the coreference judge keeps its question",
          L86.COREF_VALIDATION_FOCUS == L85.COREF_VALIDATION_FOCUS)
    for builder, args in (("_prompt_extraction", (names, ["a=B"], [])),
                          ("_prompt_doc_knowledge_judge",
                           (names, [{"term": "a", "component": "B"}]))):
        check(f"{builder} renders identically",
              getattr(old, builder)(*args) == getattr(new, builder)(*args))

    print("T5  the deletion is exactly the sentence and the space")
    empty86 = new._prompt_validation(names, cases, "")
    check("s86's lenient prompt is s85's minus the focus and one space",
          empty86 == lenient85.replace(" " + L85.VALIDATION_FOCUS, "", 1),
          f"{len(lenient85) - len(empty86)} B removed")
    check("that is 244 B", len(lenient85) - len(empty86) == 244,
          str(len(lenient85) - len(empty86)))

    print("T6  bounds and the deterministic layer untouched")
    for bound in ("CONTEXT_SENTENCES", "ANCHOR_LIMIT", "EXTRACTION_BATCH",
                  "JUDGE_BATCH", "COREFERENCE_BATCH", "LINKERS",
                  "RETAINED_MENTION_TYPES"):
        if hasattr(old, bound):
            # each module defines its own MentionType, so compare by value
            def values(x):
                return (sorted(m.value for m in x) if isinstance(x, frozenset) else x)
            check(f"{bound} identical",
                  values(getattr(new, bound)) == values(getattr(old, bound)))
    check("LEMMA_READINGS identical", L86.LEMMA_READINGS == L85.LEMMA_READINGS)
    check("WORD_PATTERN identical", L86.WORD_PATTERN == L85.WORD_PATTERN)

    print("T7  GATE-06")
    seen = set()
    for model in glob.glob(os.path.join(BASE, "benchmark", "*", "model_*",
                                        "pcm", "*.repository")):
        with open(model, errors="ignore") as fh:
            text = fh.read()
        for token in ("entityName=\"",):
            start = 0
            while (i := text.find(token, start)) != -1:
                j = text.index('"', i + len(token))
                seen.add(text[i + len(token):j])
                start = j
    authored = "\n".join(constants(L86).values())
    leaked = sorted(n for n in seen if len(n) > 3 and n in authored)
    check("no benchmark component name in any authored string", not leaked,
          ", ".join(leaked))

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
