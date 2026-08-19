"""Design invariants of s_linker52 and s_linker53 — the bisect of s_linker51.

s51 rewrote nine rule constants at once and lost F1. Both variants here are s51
with part of that rewrite reverted, so the pair localizes the loss:

  s52  s51 with all three knowledge-side rules back at s49's wording (the alias
       extraction rules, the qualified-name exclusion, the alias judge). Six of
       nine rewrites survive. If s52 holds, the loss is entirely on the side of
       the pipeline that builds the alias table.
  s53  s51 with ONE subordinate clause restored inside the alias judge: that a
       phrase naming a grouping of several elements is not an alias for one of
       them. The surrounding principle already entails it. If s53 holds where s51
       failed, the finding is that an enumerated case does work its own entailing
       principle does not.

The test asserts the containment relation exactly: every constant of s52 and s53
is either s51's or s49's, in the intended places, with s53 differing from s51 in
exactly one constant by exactly one added clause; and every method body and class
attribute is s49's.

Run from the approach/ directory:
    ../.venv/bin/python pilot/test_s52_s53_prompts.py
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, "src")

assert Path("src").is_dir(), "run from the approach/ directory"

from llm_sad_sam.core.document_loader_v2 import load_sentences
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental import s_linker49 as L49
from llm_sad_sam.linkers.experimental import s_linker51 as L51
from llm_sad_sam.linkers.experimental import s_linker52 as L52
from llm_sad_sam.linkers.experimental import s_linker53 as L53
from llm_sad_sam.linkers.experimental.s_linker49 import SLinker49
from llm_sad_sam.linkers.experimental.s_linker52 import SLinker52
from llm_sad_sam.linkers.experimental.s_linker53 import SLinker53

BENCH = Path("../../ardoco/core/tests-base/src/main/resources/benchmark")
if not BENCH.is_dir():
    BENCH = Path("../benchmark")
DATASETS = {
    "mediastore": ("mediastore/text_2016/mediastore.txt",
                   "mediastore/model_2016/pcm/ms.repository"),
    "teastore": ("teastore/text_2020/teastore.txt",
                 "teastore/model_2020/pcm/teastore.repository"),
    "teammates": ("teammates/text_2021/teammates.txt",
                  "teammates/model_2021/pcm/teammates.repository"),
    "bigbluebutton": ("bigbluebutton/text_2021/bigbluebutton.txt",
                      "bigbluebutton/model_2021/pcm/bbb.repository"),
    "jabref": ("jabref/text_2021/jabref.txt",
               "jabref/model_2021/pcm/jabref.repository"),
}

RULES = ("DOC_KNOWLEDGE_EXTRACTION_RULES", "ALIAS_EXCLUSION_RULES",
         "DOC_KNOWLEDGE_JUDGE_RULES", "ENTITY_EXTRACTION_RULES",
         "P1_FOCUS", "P2_FOCUS", "LAYERED_ENTITY_RULES",
         "COREF_VALIDATION_FOCUS", "LAYERED_COREF_RULES", "COREF_RULES")
KNOWLEDGE = {"DOC_KNOWLEDGE_EXTRACTION_RULES", "ALIAS_EXCLUSION_RULES",
             "DOC_KNOWLEDGE_JUDGE_RULES"}
ADDED_CLAUSE = ", including a grouping that encompasses several elements"

bad = 0


def check(label, ok, ill="*** FAILED ***"):
    global bad
    bad += not ok
    print(f"    {label:<66} {'OK' if ok else ill}")
    return ok


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
    for a in ("SLinker52", "SLinker53"):
        text = text.replace(a, "SLinker49")
    for a in ("s_linker52", "s_linker53"):
        text = text.replace(a, "s_linker49")
    return text


print("\n1. every method body and class attribute identical to s_linker49's")
base = bodies(SLinker49)
for cls, label in ((SLinker52, "s52"), (SLinker53, "s53")):
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


print("\n2. s52 — s51 with the knowledge side reverted, and nothing else")
for rule in RULES:
    expected = getattr(L49 if rule in KNOWLEDGE else L51, rule)
    where = "s49" if rule in KNOWLEDGE else "s51"
    check(f"s52.{rule} == {where}'s", getattr(L52, rule) == expected)
changed = {r for r in RULES if getattr(L52, r) != getattr(L49, r)}
check(f"s52 still generalizes 6 of the 9 rules ({len(changed)})", len(changed) == 6)


print("\n3. s53 — s51 with one clause restored inside the alias judge, and nothing else")
for rule in RULES:
    if rule == "DOC_KNOWLEDGE_JUDGE_RULES":
        continue
    check(f"s53.{rule} == s51's", getattr(L53, rule) == getattr(L51, rule))
check("s53's alias-judge rule differs from s51's",
      L53.DOC_KNOWLEDGE_JUDGE_RULES != L51.DOC_KNOWLEDGE_JUDGE_RULES)
check("s53 = s51 plus exactly that one clause, in place",
      L53.DOC_KNOWLEDGE_JUDGE_RULES.replace(ADDED_CLAUSE, "")
      == L51.DOC_KNOWLEDGE_JUDGE_RULES)
check(f"the clause is {len(ADDED_CLAUSE)} bytes and s53 stays shorter than s49's rule",
      len(L53.DOC_KNOWLEDGE_JUDGE_RULES) < len(L49.DOC_KNOWLEDGE_JUDGE_RULES))
check("s53 keeps the leniency sentence",
      "prefer approve" in L53.DOC_KNOWLEDGE_JUDGE_RULES.lower())
check("s53 does NOT restore s49's other two invalidity cases verbatim",
      "names the whole system" not in L53.DOC_KNOWLEDGE_JUDGE_RULES
      and "names a different entity" not in L53.DOC_KNOWLEDGE_JUDGE_RULES)


print("\n4. rendered prompts: only the knowledge prompts differ between s53 and s51")
for project in DATASETS:
    text, model = DATASETS[project]
    sentences = load_sentences(str(BENCH / text))
    components = parse_pcm_repository(str(BENCH / model))
    names = [c.name for c in components]
    mappings = [f"{n.lower()[:4]} -> {n}" for n in names[:3]]
    judge52 = SLinker52._prompt_doc_knowledge_judge(names, mappings)
    judge53 = SLinker53._prompt_doc_knowledge_judge(names, mappings)
    judge49 = SLinker49._prompt_doc_knowledge_judge(names, mappings)
    check(f"{project}: s52's alias-judge prompt == s49's", judge52 == judge49)
    from llm_sad_sam.linkers.experimental.s_linker51 import SLinker51
    judge51 = SLinker51._prompt_doc_knowledge_judge(names, mappings)
    check(f"{project}: s53's alias-judge prompt == s51's once the clause is removed",
          judge53.replace(ADDED_CLAUSE, "") == judge51)
    check(f"{project}: s53's alias-judge prompt is shorter than s49's",
          len(judge53) < len(judge49))
    for builder, label in ((SLinker52._prompt_extraction, "extraction"),):
        check(f"{project}: s52 {label} prompt == s51's shape",
              builder(names, mappings, sentences[:5])
              != SLinker49._prompt_extraction(names, mappings, sentences[:5]))

print(f"\n{'ALL CHECKS PASSED' if not bad else f'{bad} CHECK(S) FAILED'}")
sys.exit(1 if bad else 0)
