"""`s_linker87` differs from `s_linker86` in one deleted restatement and nothing else.

    T1  `COREF_RULES` loses its opening sentence and keeps the rest, exactly
    T2  every other rule constant is byte-identical to s86's
    T3  every method body is byte-identical to s86's -- the cut is a constant, so no
        code moves at all
    T4  every prompt builder renders byte-identically to s86's once s86's rules are
        substituted back, and the resolver prompt's input-format contract (the preamble
        s56 priced at TP -16.2) is still in the rendering
    T5  the deletion is 163 B, carried in the 40 resolver calls a five-project run makes
    T6  the bounds, the deterministic layer and the judging rubrics are untouched
    T7  GATE-06: no benchmark component name appears in any authored string
    T8  the chain to s_linker85 is exactly two deletions and nothing else

    ../.venv/bin/python pilot/test_s87_dedup.py
"""
from __future__ import annotations

import glob
import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

import llm_sad_sam.linkers.experimental.s_linker85 as L85              # noqa: E402
import llm_sad_sam.linkers.experimental.s_linker86 as L86              # noqa: E402
import llm_sad_sam.linkers.experimental.s_linker87 as L87              # noqa: E402

BASE = "/mnt/hostshare/ardoco-home/alinker-replication-package"
PASS, FAIL = [], []

#: The sentence this variant deletes.
RESTATEMENT = ("For each case, decide whether a pronoun or noun phrase that refers "
               "back in the target sentence refers back to a component named or "
               "aliased earlier in the context. ")

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


class _Sentence:
    def __init__(self, number, text):
        self.number, self.text = number, text


def main():
    old, new = L86.SLinker86, L87.SLinker87

    print("T1  COREF_RULES loses its opening sentence and keeps the rest")
    check("s86 carries the restatement", RESTATEMENT in L86.COREF_RULES)
    check("s87 does not", RESTATEMENT not in L87.COREF_RULES)
    check("the remainder is byte-identical",
          L87.COREF_RULES == L86.COREF_RULES.replace(RESTATEMENT, ""))
    check("the standard survives whole",
          "Avoid resolving when two or more equally plausible antecedents exist"
          in L87.COREF_RULES)

    print("T2  every other rule constant is byte-identical")
    old_c, new_c = constants(L86), constants(L87)
    for name, value in old_c.items():
        if name in RENAMED or name == "COREF_RULES":
            continue
        check(f"{name} identical", new_c.get(name) == value)
    check("no new authored constant",
          not (set(new_c) - set(old_c) - RENAMED),
          ", ".join(sorted(set(new_c) - set(old_c) - RENAMED)))

    print("T3  no method body moves")
    ob, nb = bodies(old), bodies(new)
    check("same method set", set(ob) == set(nb), f"{sorted(set(ob) ^ set(nb))}")
    for name in sorted(set(ob) & set(nb)):
        check(f"{name} identical", ob[name] == nb[name])

    print("T4  the builders render identically, contract included")
    names, cases = ["A", "B"], ["Case 1: x"]
    table = [{"sentence": 1, "text": "t"}]
    targets = [{"case": 1, "target": 1, "text": "t", "context": [1]}]
    coref86 = old._prompt_coref(names, table, targets)
    coref87 = new._prompt_coref(names, table, targets)
    check("resolver prompt is s86's minus the restatement",
          coref87 == coref86.replace(RESTATEMENT, "", 1))
    for contract in (
        "identify any pronoun or noun phrase in THAT sentence",
        "return no resolution",
        "Read the TARGET's context in SENTENCES",
    ):
        check(f"the input-format contract survives: {contract[:40]}…",
              contract in coref87)
    for builder, args in (("_prompt_extraction", (names, ["a=B"], [])),
                          ("_prompt_validation", (names, cases, "")),
                          ("_prompt_doc_knowledge_judge",
                           (names, [{"term": "a", "component": "B"}]))):
        check(f"{builder} renders identically",
              getattr(old, builder)(*args) == getattr(new, builder)(*args))
    check("the strict judging prompt is untouched",
          old._prompt_validation(names, cases, L86.COREF_VALIDATION_FOCUS, strict=True)
          == new._prompt_validation(names, cases, L87.COREF_VALIDATION_FOCUS,
                                    strict=True))

    print("T5  the deletion is 163 B")
    check("163 B", len(coref86) - len(coref87) == 163,
          str(len(coref86) - len(coref87)))

    print("T6  bounds, deterministic layer and judging rubrics untouched")
    for bound in ("CONTEXT_SENTENCES", "ANCHOR_LIMIT", "EXTRACTION_BATCH",
                  "JUDGE_BATCH", "COREFERENCE_BATCH", "LINKERS"):
        if hasattr(old, bound):
            check(f"{bound} identical", getattr(new, bound) == getattr(old, bound))
    check("RETAINED_MENTION_TYPES identical by value",
          sorted(m.value for m in new.RETAINED_MENTION_TYPES)
          == sorted(m.value for m in old.RETAINED_MENTION_TYPES))
    check("LAYERED_ENTITY_RULES identical",
          L87.LAYERED_ENTITY_RULES == L86.LAYERED_ENTITY_RULES)
    check("LAYERED_COREF_RULES identical",
          L87.LAYERED_COREF_RULES == L86.LAYERED_COREF_RULES)
    check("WORD_PATTERN identical", L87.WORD_PATTERN == L86.WORD_PATTERN)

    print("T7  GATE-06")
    seen = set()
    for model in glob.glob(os.path.join(BASE, "benchmark", "*", "model_*",
                                        "pcm", "*.repository")):
        text = open(model, errors="ignore").read()
        start = 0
        while (i := text.find('entityName="', start)) != -1:
            j = text.index('"', i + 12)
            seen.add(text[i + 12:j])
            start = j
    authored = "\n".join(constants(L87).values())
    leaked = sorted(n for n in seen if len(n) > 3 and n in authored)
    check("no benchmark component name in any authored string", not leaked,
          ", ".join(leaked))

    print("T8  the chain to s_linker85 is exactly two deletions")
    a, c = constants(L85), constants(L87)
    differing = {n for n in set(a) | set(c)
                 if n not in RENAMED and a.get(n) != c.get(n)}
    check("two constants differ from s85", differing == {"VALIDATION_FOCUS",
                                                         "COREF_RULES"},
          ", ".join(sorted(differing)))
    total85 = sum(len(v) for v in a.values() if len(v) > 60)
    total87 = sum(len(v) for v in c.values() if len(v) > 60)
    check("authored rule text 3485 -> 3079 B",
          (total85, total87) == (3485, 3079), f"{total85} -> {total87}")

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
