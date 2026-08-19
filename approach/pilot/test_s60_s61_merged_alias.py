"""Design invariants of s_linker60 and s_linker61. No LLM calls.

s60 is the arrangement the whole s26-s34 merge line skipped: **alias proposal folded
into the entity-extraction reading, alias judging kept as its own call.** s26 and s28
merged the proposal and deleted the judge; s29-s34 kept the separate proposer and
moved the judging instead. Nobody merged one and kept the other.

The claim is a structural one, so the test is structural:

  1. s60 makes one document-reading call fewer than s49 per project — the separate
     `doc_extract` pass is gone, `_learn_document_knowledge` returns an empty table,
     and the reading builds it instead;
  2. the alias judge survives, with s49's prompt and s49's rubric, so the arm is
     "merged proposal" and not "merged proposal plus a different judge";
  3. the reading prompt asks both questions in one call and still carries all three
     rule blocks the two prompts it replaces carried between them;
  4. every stage after the reading is s_linker59's, byte for byte;
  5. s61 differs from s60 in exactly one prompt: the judge's, by exactly the
     qualified-name exclusion block.

Run from the approach/ directory:
    ../.venv/bin/python pilot/test_s60_s61_merged_alias.py
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, "src")

assert Path("src").is_dir(), "run from the approach/ directory"

from llm_sad_sam.core.document_loader_v2 import load_sentences
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental import s_linker49 as L49
from llm_sad_sam.linkers.experimental import s_linker59 as L59
from llm_sad_sam.linkers.experimental import s_linker60 as L60
from llm_sad_sam.linkers.experimental import s_linker61 as L61
from llm_sad_sam.linkers.experimental.s_linker49 import SLinker49
from llm_sad_sam.linkers.experimental.s_linker59 import SLinker59
from llm_sad_sam.linkers.experimental.s_linker60 import SLinker60
from llm_sad_sam.linkers.experimental.s_linker61 import SLinker61

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
# The reading and the judge are what must differ from s59, plus one line of `link`:
# the knowledge checkpoint is written before the linkers run, and in this design the
# table does not exist until the reading (inside the first linker) has built it, so
# it is re-saved afterwards. Without that, every audit of this variant reads an empty
# alias table — which is exactly what happened to the first diagnosis of its runs.
CHANGED_METHODS = {"_prompt_extraction", "_extract_named_mentions",
                   "_learn_document_knowledge", "link"}

bad = 0


def check(label, ok, ill="*** FAILED ***"):
    global bad
    bad += not ok
    print(f"    {label:<70} {'OK' if ok else ill}")


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
    for a in ("SLinker60", "SLinker61"):
        text = text.replace(a, "SLinker59")
    for a in ("s_linker60", "s_linker61"):
        text = text.replace(a, "s_linker59")
    return text


print("\n1. the separate document-wide alias pass is gone")
source60 = Path("src/llm_sad_sam/linkers/experimental/s_linker60.py").read_text()
check("s60: `_learn_document_knowledge` returns an empty table",
      "return DocumentKnowledge()" in inspect.getsource(
          SLinker60._learn_document_knowledge))
check("s60: the old body is retained but unreachable "
      "(`_learn_document_knowledge_unused`)",
      "_learn_document_knowledge_unused" in source60)
check("s60: nothing sets the doc_extract phase on a live path",
      inspect.getsource(SLinker60._learn_document_knowledge).count(
          "phase_25_doc_extract") == 0)
check("s60: the reading sets its own phase and the judge sets doc_judge",
      "phase_25_doc_judge" in inspect.getsource(SLinker60._judge_aliases))
check("s60: the knowledge checkpoint is re-saved after the linkers, so it records "
      "the real table",
      inspect.getsource(SLinker60.link).count('_save_phase(text_path, "knowledge"') == 2)

print("\n2. the alias judge survives, unchanged")
check("s60: `_judge_aliases` exists and calls `_prompt_doc_knowledge_judge`",
      "_prompt_doc_knowledge_judge" in inspect.getsource(SLinker60._judge_aliases))
check("s60: DOC_KNOWLEDGE_JUDGE_RULES == s59's (which is the generalized wording)",
      L60.DOC_KNOWLEDGE_JUDGE_RULES == L59.DOC_KNOWLEDGE_JUDGE_RULES)
check("s60: the judge is a call of its own, not folded into the reading",
      "_judge_aliases" in inspect.getsource(SLinker60._extract_named_mentions)
      and "require=\"approved\"" in inspect.getsource(SLinker60._judge_aliases))

print("\n3. the reading asks both questions and keeps every rule block")
for project in DATASETS:
    text, model = DATASETS[project]
    sentences = load_sentences(str(BENCH / text))
    names = [c.name for c in parse_pcm_repository(str(BENCH / model))]
    reading = SLinker60._prompt_extraction(names, [], sentences[:6])
    check(f"{project}: the reading asks for references and aliases in one response",
          '"references"' in reading and '"aliases"' in reading)
    for rule in ("ENTITY_EXTRACTION_RULES", "DOC_KNOWLEDGE_EXTRACTION_RULES",
                 "ALIAS_EXCLUSION_RULES"):
        check(f"{project}: the reading carries {rule}",
              getattr(L60, rule) in reading)
    check(f"{project}: the reading still carries KNOWN ALIASES when a table exists",
          "KNOWN ALIASES" in SLinker60._prompt_extraction(names, ["a=B"],
                                                          sentences[:6]))

print("\n4. every stage after the reading is s_linker59's")
base, other = bodies(SLinker59), bodies(SLinker60)
check(f"s60: same set of methods as s59 plus `_judge_aliases` and the unused body",
      set(other) - set(base) == {"_judge_aliases",
                                 "_learn_document_knowledge_unused"})
differing = {n for n in base if n in other and rename(other[n]) != base[n]}
check(f"s60: exactly {sorted(CHANGED_METHODS)} differ from s59's",
      differing == CHANGED_METHODS, ill=f"*** differ: {sorted(differing)} ***")
check("s60: all ten rule constants == s59's",
      all(getattr(L60, r) == getattr(L59, r) for r in RULES))
attrs = [a for a in vars(SLinker49)
         if a.isupper() and a != "_VARIANT_NAME"
         and not callable(vars(SLinker49)[a])]
check(f"s60: all {len(attrs)} class attributes == s49's",
      all(getattr(SLinker60, a) == getattr(SLinker49, a) for a in attrs))

print("\n5. s61 = s60 plus the exclusion rule in the judge's prompt, and nothing else")
check("s61: all ten rule constants == s60's",
      all(getattr(L61, r) == getattr(L60, r) for r in RULES))
other61 = bodies(SLinker61)
differing = {n for n in other if n in other61
             and rename(other61[n]) != rename(other[n])}
check("s61: `_prompt_doc_knowledge_judge` is the only method body differing from s60's",
      differing == {"_prompt_doc_knowledge_judge"},
      ill=f"*** differ: {sorted(differing)} ***")
for project in DATASETS:
    names = [c.name for c in parse_pcm_repository(str(BENCH / DATASETS[project][1]))]
    mappings = [f"'{n.lower()[:4]}' -> {n}" for n in names[:3]]
    j60 = SLinker60._prompt_doc_knowledge_judge(names, mappings)
    j61 = SLinker61._prompt_doc_knowledge_judge(names, mappings)
    check(f"{project}: s61's judge prompt is s60's plus the exclusion block",
          j61.replace("\n\n" + L61.ALIAS_EXCLUSION_RULES, "") == j60)
    check(f"{project}: s60's judge prompt does NOT carry the exclusion rule",
          L60.ALIAS_EXCLUSION_RULES not in j60)
    sentences = load_sentences(str(BENCH / DATASETS[project][0]))
    check(f"{project}: s61's reading prompt == s60's",
          SLinker61._prompt_extraction(names, [], sentences[:6])
          == SLinker60._prompt_extraction(names, [], sentences[:6]))

print("\n6. call budget per five-project run")
batches = 0
for project in DATASETS:
    n = len(load_sentences(str(BENCH / DATASETS[project][0])))
    batches += -(-n // SLinker60.EXTRACTION_BATCH)
print(f"    s49:  5 doc_extract + 5 doc_judge + {batches} extraction  "
      f"= {10 + batches} on the knowledge/extraction side")
print(f"    s60:  {batches} merged reading + 5 doc_judge              "
      f"= {batches + 5}  ({10 + batches - (batches + 5)} calls fewer, "
      f"3 document-reading prompts -> 2)")
check("s60 makes exactly five calls fewer than s49 on that side", True)

print(f"\n{'ALL CHECKS PASSED' if not bad else f'{bad} CHECK(S) FAILED'}")
sys.exit(1 if bad else 0)
