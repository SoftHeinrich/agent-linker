"""Design invariants of s_linker50 and s_linker51. No LLM calls.

Both variants change prompt *wording* and nothing else, which is the easiest kind of
claim to make and the easiest to break by accident: a rewritten constant is one
character away from a rewritten prompt shape, and this workflow has measured twice
that removing evidence *content* from a prompt is pipeline-negative even when the
stage arm reads neutral. So the test is structural and total:

  1. every method body of both variants is byte-identical to s_linker49's;
  2. every class attribute (batch sizes, context window, anchor limit) is identical;
  3. of the ten rule constants, exactly the intended ones differ — one for s50, nine
     for s51 — and P2_FOCUS is carried verbatim by both;
  4. every prompt *builder*, rendered on real project data, differs from s49's only
     by the substitution of those constants: splice s49's constants back into the
     rendered text and it compares byte-identical. This is what separates "the rule
     is worded differently" from "the prompt is shaped differently";
  5. the generalized text introduces no benchmark vocabulary (GATE-06) and is
     strictly shorter.

Run from the approach/ directory:
    ../.venv/bin/python pilot/test_s50_s51_prompts.py
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, "src")

assert Path("src").is_dir(), "run from the approach/ directory"

from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental import s_linker49 as L49
from llm_sad_sam.linkers.experimental import s_linker50 as L50
from llm_sad_sam.linkers.experimental import s_linker51 as L51
from llm_sad_sam.linkers.experimental.s_linker49 import SLinker49
from llm_sad_sam.linkers.experimental.s_linker50 import SLinker50
from llm_sad_sam.linkers.experimental.s_linker51 import SLinker51

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

S50_CHANGED = {"COREF_RULES"}
S51_CHANGED = set(RULES) - {"P2_FOCUS"}

# The calls each rule is carried into per five-project run, counted from the six
# recorded s49 runs. Used only to report the byte saving.
CALLS_PER_RUN = {
    "DOC_KNOWLEDGE_EXTRACTION_RULES": 5, "ALIAS_EXCLUSION_RULES": 5,
    "DOC_KNOWLEDGE_JUDGE_RULES": 5, "ENTITY_EXTRACTION_RULES": 9,
    "P1_FOCUS": 9, "P2_FOCUS": 9, "LAYERED_ENTITY_RULES": 18,
    "COREF_VALIDATION_FOCUS": 7, "LAYERED_COREF_RULES": 7, "COREF_RULES": 40,
}

bad = 0


def check(label, ok, good="OK", ill="*** FAILED ***"):
    global bad
    bad += not ok
    print(f"    {label:<66} {good if ok else ill}")
    return ok


def load(name):
    text, model = DATASETS[name]
    sentences = load_sentences(str(BENCH / text))
    components = parse_pcm_repository(str(BENCH / model))
    return sentences, components, build_sent_map(sentences)


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


def normalize(text, variant_from, variant_to):
    """Variant self-references are expected to differ; nothing else is."""
    return text.replace(variant_from, variant_to).replace(
        variant_from.replace("_", "").replace("slinker", "SLinker"), variant_to)


def rename(text, other):
    for a, b in (("SLinker50", "SLinker49"), ("SLinker51", "SLinker49"),
                 ("s_linker50", "s_linker49"), ("s_linker51", "s_linker49")):
        text = text.replace(a, b)
    return text


# ── 1/2. method bodies and class attributes ──────────────────────────────────

print("\n1. every method body and class attribute identical to s_linker49's")
base = bodies(SLinker49)
for cls, label in ((SLinker50, "s50"), (SLinker51, "s51")):
    other = bodies(cls)
    check(f"{label}: same set of methods ({len(base)})", set(other) == set(base))
    differing = [n for n in base
                 if n in other and rename(other[n], base[n]) != base[n]]
    check(f"{label}: all {len(base)} method bodies byte-identical", not differing,
          good="OK", ill=f"*** differ: {differing} ***")
    # `_VARIANT_NAME` is the checkpoint/log namespace and must differ per variant.
    attrs = [a for a in vars(SLinker49)
             if a.isupper() and a != "_VARIANT_NAME"
             and not callable(vars(SLinker49)[a])]
    same = [a for a in attrs if getattr(cls, a) == getattr(SLinker49, a)]
    check(f"{label}: all {len(attrs)} class attributes identical "
          f"({', '.join(attrs)})", len(same) == len(attrs))


# ── 3. exactly the intended constants differ ─────────────────────────────────

print("\n2. exactly the intended rule constants differ")
for module, label, expected in ((L50, "s50", S50_CHANGED), (L51, "s51", S51_CHANGED)):
    differing = {r for r in RULES if getattr(module, r) != getattr(L49, r)}
    check(f"{label}: changed constants == {sorted(expected)}", differing == expected,
          ill=f"*** got {sorted(differing)} ***")
    check(f"{label}: P2_FOCUS carried verbatim",
          module.P2_FOCUS == L49.P2_FOCUS)
    shorter = all(len(getattr(module, r)) < len(getattr(L49, r)) for r in differing)
    check(f"{label}: every changed constant is strictly shorter", shorter)

before = sum(len(getattr(L49, r)) * CALLS_PER_RUN[r] for r in RULES)
for module, label in ((L50, "s50"), (L51, "s51")):
    after = sum(len(getattr(module, r)) * CALLS_PER_RUN[r] for r in RULES)
    text_before = sum(len(getattr(L49, r)) for r in RULES)
    text_after = sum(len(getattr(module, r)) for r in RULES)
    print(f"    {label}: rule text {text_before} -> {text_after} B "
          f"({100 * (1 - text_after / text_before):.1f}%), instruction bytes per "
          f"five-project run {before} -> {after} "
          f"({100 * (1 - after / before):.1f}%)")


# ── 4. prompt builders differ only by the constant substitution ──────────────

print("\n3. every prompt builder, rendered on real project data, differs only by "
      "the\n   substitution of those constants")


class Sent:
    def __init__(self, number, text):
        self.number = number
        self.text = text


def render_all(module, cls, sentences, components):
    """Every static prompt builder, on real data. Returns {name: text}."""
    names = [c.name for c in components]
    batch = sentences[:8]
    mappings = [f"{n.lower()[:4]} -> {n}" for n in names[:3]]
    cases = [f"Case 1: S{sentences[0].number} — {names[0]}",
             f"Case 2: S{sentences[1].number} — {names[1]}"]
    coref_cases = [{"sent": s,
                    "context": [f"    S{s.number}: {s.text}"]}
                   for s in sentences[:3]]
    return {
        "doc_knowledge_extract": cls._prompt_doc_knowledge_extract(
            names, [s.text for s in batch]),
        "doc_knowledge_judge": cls._prompt_doc_knowledge_judge(names, mappings),
        "extraction": cls._prompt_extraction(names, mappings, batch),
        "validation_p1": cls._prompt_validation(names, cases, module.P1_FOCUS),
        "validation_p2": cls._prompt_validation(names, cases, module.P2_FOCUS),
        "validation_coref": cls._prompt_validation(
            names, cases, module.COREF_VALIDATION_FOCUS, strict=True),
        "coref": cls._prompt_coref(names, coref_cases),
    }


def substitute_back(text, module):
    """Put s49's wording back where the variant's wording is; then it must match."""
    for rule in sorted(RULES, key=lambda r: -len(getattr(module, r))):
        variant, original = getattr(module, rule), getattr(L49, rule)
        if variant != original:
            text = text.replace(variant, original)
    return text


for project in DATASETS:
    sentences, components, _ = load(project)
    reference = render_all(L49, SLinker49, sentences, components)
    for module, cls, label in ((L50, SLinker50, "s50"), (L51, SLinker51, "s51")):
        rendered = render_all(module, cls, sentences, components)
        mismatched = [name for name, text in rendered.items()
                      if substitute_back(text, module) != reference[name]]
        check(f"{label} / {project}: {len(reference)} prompts identical after "
              f"back-substitution", not mismatched,
              ill=f"*** differ: {mismatched} ***")
        # And the variant's own rendering must actually be shorter, i.e. the
        # constants really reach the prompts they are supposed to reach.
        touched = [name for name, text in rendered.items()
                   if len(text) < len(reference[name])]
        expected_touched = 7 if label == "s51" else 1
        check(f"{label} / {project}: {expected_touched} of {len(reference)} prompts "
              f"shrink", len(touched) == expected_touched,
              ill=f"*** shrank: {touched} ***")


# ── 5. GATE-06: no benchmark vocabulary, and the specific shapes are gone ────

print("\n4. GATE-06 and the specific surface forms the rewrite is supposed to drop")
component_names = set()
for project in DATASETS:
    _, components, _ = load(project)
    component_names |= {c.name.lower() for c in components}
# Names that are ordinary English words cannot be evidence of leakage.
ENGLISH = {"database", "facade", "registry", "logic", "core", "web", "client",
           "apps", "gui", "model", "cache", "search", "server", "api", "ui",
           "authentication", "recommender", "image", "persistence", "preview",
           "storage", "media", "management", "access", "packaging"}
suspicious = sorted(n for n in component_names if len(n) > 3 and n not in ENGLISH)

for module, label in ((L50, "s50"), (L51, "s51")):
    blob = " ".join(getattr(module, r) for r in RULES).lower()
    leaked = [n for n in suspicious if n in blob]
    check(f"{label}: no benchmark component name in any rule", not leaked,
          ill=f"*** leaked: {leaked} ***")

gone = {"x.y.z": "qualified-path shape",
        "the module": "listed role phrase",
        "gerund": "listed fragment shape",
        "terminal word": "listed alias shape"}
blob51 = " ".join(getattr(L51, r) for r in RULES).lower()
blob49 = " ".join(getattr(L49, r) for r in RULES).lower()
for needle, what in gone.items():
    check(f"s51: {what} ({needle!r}) present in s49, absent in s51",
          needle in blob49 and needle not in blob51)

# The two properties the rewrite must NOT touch.
check("s51: full-name gate still approves by default",
      "approve the link by default" in L51.LAYERED_ENTITY_RULES.lower()
      and L51.LAYERED_ENTITY_RULES.lower().rstrip().endswith("approve."))
check("s51: alias judge still lenient ('prefer APPROVE')",
      "prefer approve" in L51.DOC_KNOWLEDGE_JUDGE_RULES.lower())
check("s51: coreference gate still rejects when uncertain",
      "when uncertain, reject" in L51.LAYERED_COREF_RULES.lower())
check("s50: the three properties above unchanged from s49",
      L50.LAYERED_ENTITY_RULES == L49.LAYERED_ENTITY_RULES
      and L50.DOC_KNOWLEDGE_JUDGE_RULES == L49.DOC_KNOWLEDGE_JUDGE_RULES
      and L50.LAYERED_COREF_RULES == L49.LAYERED_COREF_RULES)

print(f"\n{'ALL CHECKS PASSED' if not bad else f'{bad} CHECK(S) FAILED'}")
sys.exit(1 if bad else 0)
