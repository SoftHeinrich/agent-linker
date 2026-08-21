"""The composed head's invariants: only the two adopted constants moved.

`s_linker91` is `s_linker89` with two authored constants replaced by the arms this
round measured neutral on both models. The stage arms cannot see a mistake here --
each was measured alone, and a composition is not a measured text -- so this asserts
the composition mechanically:

  1. every module-level constant except the two is byte-identical to s89's;
  2. each of the two is byte-identical to the arm text that was measured;
  3. every method of the class has the same source as s89's, modulo the rename;
  4. the rendered prompts of all five projects differ from s89's by exactly the
     five substitutions and nothing else;
  5. no adopted constant writes a benchmark component's word (GATE-06).

    ../.venv/bin/python pilot/test_s91_static.py
"""
from __future__ import annotations

import inspect
import os
import re
import sys

BASE = "/mnt/hostshare/ardoco-home/alinker-replication-package"
sys.path.insert(0, BASE + "/approach/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_sad_sam.linkers.experimental import s_linker89 as OLD        # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker91 as NEW        # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker89 import SLinker89     # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker91 import SLinker91     # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository            # noqa: E402
from llm_sad_sam.core.document_loader_v2 import load_sentences        # noqa: E402

import static_pilots as S                                             # noqa: E402

MODELS = {
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

#: The adopted arm text for each constant this round moves, named by the arm that
#: measured it. Anything not listed here must not have changed.
ADOPTED = {
    "STRICTER_CLAUSE": (S.MERGED_STRICTER, "mergeord"),
    "LAYERED_COREF_RULES": (S.GEN_LAYERED_COREF, "genartifact"),
}

checks = 0
fails: list[str] = []


def check(label, ok, detail=""):
    global checks
    checks += 1
    if not ok:
        fails.append(f"{label}: {detail}")
        print(f"  FAIL {label}" + (f"  -- {detail}" if detail else ""))


print("T1  every other module constant is byte-identical to s89's")
for name in dir(OLD):
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name) or name in ADOPTED:
        continue
    check(f"{name} unchanged", getattr(OLD, name) == getattr(NEW, name, None))

print("T2  the adopted constants are exactly the texts the arms measured")
for name, (text, arm) in ADOPTED.items():
    check(f"{name} == the `{arm}` arm's text", getattr(NEW, name) == text,
          f"got {getattr(NEW, name)!r}")
    check(f"{name} differs from s89", getattr(NEW, name) != getattr(OLD, name))

print("T3  every class method has the same source, modulo the rename")
old_members = {n: m for n, m in inspect.getmembers(SLinker89)
               if inspect.isfunction(m) or inspect.ismethod(m)}
new_members = {n: m for n, m in inspect.getmembers(SLinker91)
               if inspect.isfunction(m) or inspect.ismethod(m)}
check("same method names", set(old_members) == set(new_members),
      str(set(old_members) ^ set(new_members)))
for name in sorted(set(old_members) & set(new_members)):
    try:
        a = inspect.getsource(old_members[name]).replace("SLinker89", "SLinker91")
        b = inspect.getsource(new_members[name])
    except (OSError, TypeError):
        continue
    check(f"{name}() source identical", a == b)
check("class attributes identical", all(
    getattr(SLinker89, k) == getattr(SLinker91, k)
    for k in ("ANCHOR_LIMIT", "EXTRACTION_BATCH", "JUDGE_BATCH",
              "COREFERENCE_BATCH", "LINKERS")))


def substitute(text):
    """s89's rendering with the five constants swapped for the adopted ones."""
    for name, (new_text, _arm) in ADOPTED.items():
        text = text.replace(getattr(OLD, name), new_text)
    return text


print("T4  rendered prompts on all five projects differ by exactly the two swaps")
for proj, (text_path, model_path) in MODELS.items():
    comps = parse_pcm_repository(os.path.join(BASE, "benchmark", model_path))
    sents = load_sentences(os.path.join(BASE, "benchmark", text_path))
    names = [c.name for c in comps]
    cases = [f'Case 1: "x" -> {names[0]}']
    lines = [f"S{s.number}: {s.text}" for s in sents[:40]]
    for strict in (False, True):
        a = SLinker89._prompt_validation(names, cases, "", strict=strict)
        b = SLinker91._prompt_validation(names, cases, "", strict=strict)
        check(f"{proj}: {'strict' if strict else 'lenient'} judging prompt",
              b == substitute(a), f"{len(a)} -> {len(b)} B")
    a = SLinker89._prompt_doc_knowledge_extract(names, lines)
    b = SLinker91._prompt_doc_knowledge_extract(names, lines)
    check(f"{proj}: alias-extraction prompt", b == substitute(a),
          f"{len(a)} -> {len(b)} B")
    a = SLinker89._prompt_extraction(names, [], sents[:40])
    b = SLinker91._prompt_extraction(names, [], sents[:40])
    check(f"{proj}: full-name extraction prompt", b == substitute(a),
          f"{len(a)} -> {len(b)} B")

print("T5  GATE-06: no adopted clause writes a benchmark component's word")
benchmark_words = set()
for proj, (_t, model_path) in MODELS.items():
    for c in parse_pcm_repository(os.path.join(BASE, "benchmark", model_path)):
        for w in re.findall(r"[A-Za-z]+", c.name):
            if len(w) > 3:
                benchmark_words.add(w.casefold())
for name, (t, _arm) in ADOPTED.items():
    leak = set(re.findall(r"[a-z]+", t.lower())) & benchmark_words
    check(f"{name} writes no benchmark component word", not leak, str(sorted(leak)))

print(f"\n{checks - len(fails)}/{checks} checks pass")
sys.exit(1 if fails else 0)
