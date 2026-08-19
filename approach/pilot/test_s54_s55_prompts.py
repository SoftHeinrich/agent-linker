"""Containment invariants of the four bisect arms of s_linker51. No LLM calls.

s51 rewrote nine rule constants at once. Four arms cut that rewrite three ways, and
the only thing that makes them interpretable is that each constant is *either*
s_linker49's or s_linker51's — never a third wording. This asserts exactly that, per
constant, per arm, plus the usual structural identity with s_linker49.

    family        constants                                     s51 s52 s53 s54 s55
    knowledge     DOC_KNOWLEDGE_{EXTRACTION,JUDGE}_RULES,       gen s49 gen gen s49
                  ALIAS_EXCLUSION_RULES                          (s53: one clause back)
    full-name     ENTITY_EXTRACTION_RULES, P1_FOCUS,            gen gen gen s49 s49
                  LAYERED_ENTITY_RULES
    coreference   COREF_RULES, COREF_VALIDATION_FOCUS,          gen gen gen gen gen
                  LAYERED_COREF_RULES

Run from the approach/ directory:
    ../.venv/bin/python pilot/test_s54_s55_prompts.py
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, "src")

assert Path("src").is_dir(), "run from the approach/ directory"

from llm_sad_sam.linkers.experimental import s_linker49 as L49
from llm_sad_sam.linkers.experimental import s_linker51 as L51
from llm_sad_sam.linkers.experimental import s_linker52 as L52
from llm_sad_sam.linkers.experimental import s_linker53 as L53
from llm_sad_sam.linkers.experimental import s_linker54 as L54
from llm_sad_sam.linkers.experimental import s_linker55 as L55
from llm_sad_sam.linkers.experimental import s_linker49_null as LNULL
from llm_sad_sam.linkers.experimental.s_linker49 import SLinker49
from llm_sad_sam.linkers.experimental.s_linker54 import SLinker54
from llm_sad_sam.linkers.experimental.s_linker55 import SLinker55
from llm_sad_sam.linkers.experimental.s_linker49_null import SLinker49Null

KNOWLEDGE = ("DOC_KNOWLEDGE_EXTRACTION_RULES", "ALIAS_EXCLUSION_RULES",
             "DOC_KNOWLEDGE_JUDGE_RULES")
FULL_NAME = ("ENTITY_EXTRACTION_RULES", "P1_FOCUS", "LAYERED_ENTITY_RULES")
COREF = ("COREF_VALIDATION_FOCUS", "LAYERED_COREF_RULES", "COREF_RULES")
RULES = KNOWLEDGE + FULL_NAME + COREF + ("P2_FOCUS",)

CALLS_PER_RUN = {
    "DOC_KNOWLEDGE_EXTRACTION_RULES": 5, "ALIAS_EXCLUSION_RULES": 5,
    "DOC_KNOWLEDGE_JUDGE_RULES": 5, "ENTITY_EXTRACTION_RULES": 9,
    "P1_FOCUS": 9, "P2_FOCUS": 9, "LAYERED_ENTITY_RULES": 18,
    "COREF_VALIDATION_FOCUS": 7, "LAYERED_COREF_RULES": 7, "COREF_RULES": 40,
}

# family -> module the arm takes that family's wording from
PLAN = {
    "s52": {"knowledge": L49, "full_name": L51, "coref": L51},
    "s54": {"knowledge": L51, "full_name": L49, "coref": L51},
    "s55": {"knowledge": L49, "full_name": L49, "coref": L51},
}
MODULES = {"s52": L52, "s54": L54, "s55": L55}

bad = 0


def check(label, ok, ill="*** FAILED ***"):
    global bad
    bad += not ok
    print(f"    {label:<66} {'OK' if ok else ill}")


def bodies(cls):
    out = {}
    for name, member in vars(cls).items():
        target = member.__func__ if isinstance(member, (staticmethod, classmethod)) \
            else member
        if callable(target):
            try:
                out[name] = inspect.getsource(target)
            except (OSError, TypeError):
                pass
    return out


def rename(text):
    for a in ("SLinker54", "SLinker55", "SLinker49Null"):
        text = text.replace(a, "SLinker49")
    for a in ("s_linker54", "s_linker55", "s_linker49_null"):
        text = text.replace(a, "s_linker49")
    return text


print("\n1. structural identity with s_linker49")
base = bodies(SLinker49)
for cls, label in ((SLinker54, "s54"), (SLinker55, "s55"),
                   (SLinker49Null, "s49_null")):
    other = bodies(cls)
    check(f"{label}: same set of methods ({len(base)})", set(other) == set(base))
    differing = [n for n in base if n in other and rename(other[n]) != base[n]]
    check(f"{label}: all {len(base)} method bodies byte-identical", not differing,
          ill=f"*** differ: {differing} ***")
    attrs = [a for a in vars(SLinker49)
             if a.isupper() and a != "_VARIANT_NAME"
             and not callable(vars(SLinker49)[a])]
    check(f"{label}: all {len(attrs)} class attributes identical",
          all(getattr(cls, a) == getattr(SLinker49, a) for a in attrs))

print("\n2. the null arm is byte-identical on every rule")
check("s49_null: all 10 rule constants == s49's",
      all(getattr(LNULL, r) == getattr(L49, r) for r in RULES))
check("s49_null: only the checkpoint namespace differs",
      SLinker49Null._VARIANT_NAME == "s_linker49_null"
      and SLinker49._VARIANT_NAME == "s_linker49")

print("\n3. every constant of every bisect arm is either s49's or s51's")
FAMILY = {**{r: "knowledge" for r in KNOWLEDGE},
          **{r: "full_name" for r in FULL_NAME},
          **{r: "coref" for r in COREF}}
for label, plan in PLAN.items():
    module = MODULES[label]
    for rule in RULES:
        if rule == "P2_FOCUS":
            check(f"{label}.P2_FOCUS carried verbatim by everything",
                  module.P2_FOCUS == L49.P2_FOCUS == L51.P2_FOCUS)
            continue
        want = plan[FAMILY[rule]]
        name = "s49" if want is L49 else "s51"
        check(f"{label}.{rule} == {name}'s", getattr(module, rule) == getattr(want, rule))

print("\n4. s53 is s51 plus one clause, and touches nothing else")
for rule in RULES:
    if rule == "DOC_KNOWLEDGE_JUDGE_RULES":
        continue
    check(f"s53.{rule} == s51's", getattr(L53, rule) == getattr(L51, rule))
clause = ", including a grouping that encompasses several elements"
check("s53's alias judge == s51's plus that one clause",
      L53.DOC_KNOWLEDGE_JUDGE_RULES.replace(clause, "")
      == L51.DOC_KNOWLEDGE_JUDGE_RULES)

print("\n5. byte budget of every arm")
for label, module in (("s49", L49), ("s51", L51), ("s52", L52), ("s53", L53),
                      ("s54", L54), ("s55", L55)):
    text = sum(len(getattr(module, r)) for r in RULES)
    per_run = sum(len(getattr(module, r)) * CALLS_PER_RUN[r] for r in RULES)
    base_text = sum(len(getattr(L49, r)) for r in RULES)
    base_run = sum(len(getattr(L49, r)) * CALLS_PER_RUN[r] for r in RULES)
    print(f"    {label:5s} rule text {text:5d} B ({100 * (1 - text / base_text):5.1f}% off)"
          f"   instruction/run {per_run:6d} B "
          f"({100 * (1 - per_run / base_run):5.1f}% off)")

print(f"\n{'ALL CHECKS PASSED' if not bad else f'{bad} CHECK(S) FAILED'}")
sys.exit(1 if bad else 0)
