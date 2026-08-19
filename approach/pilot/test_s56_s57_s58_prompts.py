"""Containment invariants of the four arms built on s_linker55. No LLM calls.

s55 confirmed that the coreference family's enumerations restate as guidelines for
nothing, and exposed the mechanism: an instruction is removable when a later,
independent step already enforces it. These arms apply that idea elsewhere, each as
one change on top of s55. Three were then measured by single-stage ablation
(`pilot/prompt_stage_pilots.py`) rather than end to end, and two of the three were
refuted before any five-project run was paid for:

  s56  the *same* prompt states its task twice — `_prompt_coref`'s opening paragraph
       and the `COREF_RULES` block appended below it. The paragraph goes. Every rule
       constant is s55's byte for byte; the only difference in the whole file is that
       builder. REFUTED: TP -16.2 (p = 0.01) at the resolution stage.
  s57  the two alias *proposer* rules generalized, the alias judge carried verbatim.
       NOT SUPPORTED: FP +1.8 (p = 0.33), wrong direction in a costly family.
  s58  the full-name *proposer* rule generalized, both judging rubrics verbatim.
       REFUTED: FP +20.2 (p = 0.01) at the extraction stage.
  s59  what survives: the coreference family, `P1_FOCUS` and the alias judge rubric,
       each cleared at p = 1.00 or better, and composed at TP -1.0 (p = 0.23).

What has to hold for the round to be interpretable: every constant of every arm is
either s49's or s51's, in the intended place; s56 changes no constant at all; and no
method body other than `_prompt_coref` (s56 only) differs from s49's.

Run from the approach/ directory:
    ../.venv/bin/python pilot/test_s56_s57_s58_prompts.py
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
from llm_sad_sam.linkers.experimental import s_linker55 as L55
from llm_sad_sam.linkers.experimental import s_linker56 as L56
from llm_sad_sam.linkers.experimental import s_linker57 as L57
from llm_sad_sam.linkers.experimental import s_linker58 as L58
from llm_sad_sam.linkers.experimental import s_linker59 as L59
from llm_sad_sam.linkers.experimental.s_linker49 import SLinker49
from llm_sad_sam.linkers.experimental.s_linker55 import SLinker55
from llm_sad_sam.linkers.experimental.s_linker56 import SLinker56
from llm_sad_sam.linkers.experimental.s_linker57 import SLinker57
from llm_sad_sam.linkers.experimental.s_linker58 import SLinker58
from llm_sad_sam.linkers.experimental.s_linker59 import SLinker59

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

KNOWLEDGE_PROPOSER = ("DOC_KNOWLEDGE_EXTRACTION_RULES", "ALIAS_EXCLUSION_RULES")
COREF = ("COREF_VALIDATION_FOCUS", "LAYERED_COREF_RULES", "COREF_RULES")
RULES = KNOWLEDGE_PROPOSER + ("DOC_KNOWLEDGE_JUDGE_RULES", "ENTITY_EXTRACTION_RULES",
                              "P1_FOCUS", "P2_FOCUS", "LAYERED_ENTITY_RULES") + COREF

CALLS_PER_RUN = {
    "DOC_KNOWLEDGE_EXTRACTION_RULES": 5, "ALIAS_EXCLUSION_RULES": 5,
    "DOC_KNOWLEDGE_JUDGE_RULES": 5, "ENTITY_EXTRACTION_RULES": 9,
    "P1_FOCUS": 9, "P2_FOCUS": 9, "LAYERED_ENTITY_RULES": 18,
    "COREF_VALIDATION_FOCUS": 7, "LAYERED_COREF_RULES": 7, "COREF_RULES": 40,
}

# arm -> the module each constant's wording must come from
GENERALIZED = {
    "s55": set(COREF),
    "s56": set(COREF),
    "s57": set(COREF) | set(KNOWLEDGE_PROPOSER),
    "s58": set(COREF) | {"ENTITY_EXTRACTION_RULES"},
    # s59 takes the three clauses single-stage ablation cleared and no others.
    "s59": set(COREF) | {"P1_FOCUS", "DOC_KNOWLEDGE_JUDGE_RULES"},
}
MODULES = {"s55": L55, "s56": L56, "s57": L57, "s58": L58, "s59": L59}
CLASSES = {"s55": SLinker55, "s56": SLinker56, "s57": SLinker57,
           "s58": SLinker58, "s59": SLinker59}

bad = 0


def check(label, ok, ill="*** FAILED ***"):
    global bad
    bad += not ok
    print(f"    {label:<68} {'OK' if ok else ill}")


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
    for a in ("SLinker55", "SLinker56", "SLinker57", "SLinker58", "SLinker59"):
        text = text.replace(a, "SLinker49")
    for a in ("s_linker55", "s_linker56", "s_linker57", "s_linker58", "s_linker59"):
        text = text.replace(a, "s_linker49")
    return text


print("\n1. every constant is either s49's or s51's, in the intended place")
for label, generalized in GENERALIZED.items():
    module = MODULES[label]
    wrong = []
    for rule in RULES:
        want = L51 if rule in generalized else L49
        if getattr(module, rule) != getattr(want, rule):
            wrong.append(rule)
    check(f"{label}: {len(generalized)} generalized, {len(RULES) - len(generalized)} "
          f"at s49's wording", not wrong, ill=f"*** wrong: {wrong} ***")

print("\n2. s56 changes no constant — its one change is a builder")
check("s56: all 10 rule constants == s55's",
      all(getattr(L56, r) == getattr(L55, r) for r in RULES))
base, other = bodies(SLinker49), bodies(SLinker56)
differing = [n for n in base if n in other and rename(other[n]) != base[n]]
check("s56: `_prompt_coref` is the only method body differing from s49's",
      differing == ["_prompt_coref"], ill=f"*** differ: {differing} ***")

print("\n3. s57, s58 and s59 change no method body at all")
for label in ("s57", "s58", "s59"):
    other = bodies(CLASSES[label])
    differing = [n for n in base if n in other and rename(other[n]) != base[n]]
    check(f"{label}: all {len(base)} method bodies byte-identical to s49's",
          not differing, ill=f"*** differ: {differing} ***")

print("\n4. class attributes untouched everywhere")
attrs = [a for a in vars(SLinker49)
         if a.isupper() and a != "_VARIANT_NAME"
         and not callable(vars(SLinker49)[a])]
for label, cls in CLASSES.items():
    check(f"{label}: all {len(attrs)} class attributes identical",
          all(getattr(cls, a) == getattr(SLinker49, a) for a in attrs))

print("\n5. s57 keeps the alias judge, s58 keeps both link judges — the point of each arm")
check("s57: DOC_KNOWLEDGE_JUDGE_RULES == s49's",
      L57.DOC_KNOWLEDGE_JUDGE_RULES == L49.DOC_KNOWLEDGE_JUDGE_RULES)
check("s57: its two proposer rules are strictly shorter than s49's",
      all(len(getattr(L57, r)) < len(getattr(L49, r)) for r in KNOWLEDGE_PROPOSER))
check("s58: P1_FOCUS, P2_FOCUS and LAYERED_ENTITY_RULES == s49's",
      L58.P1_FOCUS == L49.P1_FOCUS and L58.P2_FOCUS == L49.P2_FOCUS
      and L58.LAYERED_ENTITY_RULES == L49.LAYERED_ENTITY_RULES)
check("s58: ENTITY_EXTRACTION_RULES is strictly shorter than s49's",
      len(L58.ENTITY_EXTRACTION_RULES) < len(L49.ENTITY_EXTRACTION_RULES))
check("every arm keeps the coreference gate's reject-when-uncertain",
      all("when uncertain, reject" in getattr(m, "LAYERED_COREF_RULES").lower()
          for m in MODULES.values()))
check("every arm keeps the full-name gate's approve-by-default",
      all(getattr(m, "LAYERED_ENTITY_RULES").lower().rstrip().endswith("approve.")
          for m in MODULES.values()))
check("every arm keeps the alias judge's leniency",
      all("prefer approve" in getattr(m, "DOC_KNOWLEDGE_JUDGE_RULES").lower()
          for m in MODULES.values()))
check("s59 keeps LAYERED_ENTITY_RULES and ENTITY_EXTRACTION_RULES at s49's wording "
      "(both refuted)",
      L59.LAYERED_ENTITY_RULES == L49.LAYERED_ENTITY_RULES
      and L59.ENTITY_EXTRACTION_RULES == L49.ENTITY_EXTRACTION_RULES)
check("s59 keeps the coreference prompt's preamble (refuted as a format contract)",
      "Be conservative" in SLinker59._prompt_coref(
          ["A"], [{"sent": type("S", (), {"number": 1, "text": "x"})(),
                   "context": ["   S1: x"]}]))

print("\n6. rendered prompts on real project data")
for project in DATASETS:
    text, model = DATASETS[project]
    sentences = load_sentences(str(BENCH / text))
    names = [c.name for c in parse_pcm_repository(str(BENCH / model))]
    cases = [{"sent": s, "context": [f"    S{s.number}: {s.text}"]}
             for s in sentences[:3]]
    p49 = SLinker49._prompt_coref(names, cases)
    p55 = SLinker55._prompt_coref(names, cases)
    p56 = SLinker56._prompt_coref(names, cases)
    check(f"{project}: coref prompt shrinks s49 {len(p49)} > s55 {len(p55)} > "
          f"s56 {len(p56)}", len(p49) > len(p55) > len(p56))
    check(f"{project}: s56's coref prompt still carries COREF_RULES and the cases",
          L56.COREF_RULES in p56 and "--- Case 1:" in p56 and "CONTEXT:" in p56)
    check(f"{project}: s56 dropped the duplicated instruction paragraph",
          "Be conservative" in p55 and "Be conservative" not in p56)
    mappings = [f"{n.lower()[:4]} -> {n}" for n in names[:3]]
    check(f"{project}: s57's alias-judge prompt == s49's",
          SLinker57._prompt_doc_knowledge_judge(names, mappings)
          == SLinker49._prompt_doc_knowledge_judge(names, mappings))
    check(f"{project}: s58's two validation prompts == s49's",
          SLinker58._prompt_validation(names, ["Case 1"], L49.P1_FOCUS)
          == SLinker49._prompt_validation(names, ["Case 1"], L49.P1_FOCUS)
          and SLinker58._prompt_validation(names, ["Case 1"], L49.P2_FOCUS)
          == SLinker49._prompt_validation(names, ["Case 1"], L49.P2_FOCUS))

print("\n7. instruction bytes removed per five-project run, against s_linker49")
# The coreference prompt is measured by rendering it, because s56's change is in the
# builder rather than in a constant; the data part of the rendering is identical across
# arms, so the *difference* between two renderings is instruction text exactly.
sentences = load_sentences(str(BENCH / DATASETS["mediastore"][0]))
names = [c.name for c in parse_pcm_repository(str(BENCH / DATASETS["mediastore"][1]))]
cases = [{"sent": s, "context": [f"    S{s.number}: {s.text}"]} for s in sentences[:3]]
NON_COREF = [r for r in RULES if r not in COREF]
base_other = sum(len(getattr(L49, r)) * CALLS_PER_RUN[r] for r in NON_COREF)
base_coref = len(SLinker49._prompt_coref(names, cases)) * 40
base_total = base_other + base_coref
print(f"    {'s49 baseline':6s} other rules {base_other:6d} B/run   coreference prompt "
      f"{base_coref:6d} B/run   (data included in the second)")
for label in ("s55", "s56", "s57", "s58", "s59"):
    other = sum(len(getattr(MODULES[label], r)) * CALLS_PER_RUN[r] for r in NON_COREF)
    coref = len(CLASSES[label]._prompt_coref(names, cases)) * 40
    removed = base_total - (other + coref)
    print(f"    {label:6s} other rules {other:6d}   coreference prompt {coref:6d}   "
          f"removed {removed:6d} B/run")
    check(f"{label}: removes at least as much as s55", removed >= 14640 - 1)

print(f"\n{'ALL CHECKS PASSED' if not bad else f'{bad} CHECK(S) FAILED'}")
sys.exit(1 if bad else 0)
