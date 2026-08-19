"""Design invariants of s_linker62/63/64 and the s59 null arm. No LLM calls.

The claim of this round is narrow and structural, so the test is too:

  1. s_linker59_null is s_linker59 byte for byte except its variant name -- if that is
     not true, the in-set null measures a difference instead of the harness;
  2. s_linker62 differs from s_linker59 in exactly one method body plus the new
     predicate, carries identical prompts and identical rule constants, and its
     `INFLECTIONS` list holds no benchmark vocabulary (GATE-06);
  3. the inflection predicate does what the docstring says on real name words:
     inflected forms pass, unrelated continuations do not;
  4. s_linker63 differs from s_linker62 in exactly `_inside_qualified_identifier`,
     and the guarded predicate differs from the unguarded one on exactly the
     sentence-initial and text-final spans -- nowhere else;
  5. every proposal set is what `pilot/partial_screen.py` measured, so the variants and
     the screen cannot drift apart: s62's proposer offers the two `WebRTC-SFU` gold
     candidates s59's drops, and does not offer `webcams -> BBB web`.

Run from the approach/ directory:
    ../.venv/bin/python pilot/test_s62_s63_proposer.py
"""
import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

assert Path("src").is_dir(), "run from the approach/ directory"

from llm_sad_sam.core.document_loader_v2 import load_sentences
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental import s_linker59 as L59
from llm_sad_sam.linkers.experimental import s_linker62 as L62
from llm_sad_sam.linkers.experimental import s_linker63 as L63
from llm_sad_sam.linkers.experimental import s_linker64 as L64
from llm_sad_sam.linkers.experimental.s_linker59 import SLinker59
from llm_sad_sam.linkers.experimental.s_linker59_null import SLinker59Null
from llm_sad_sam.linkers.experimental.s_linker62 import SLinker62
from llm_sad_sam.linkers.experimental.s_linker63 import SLinker63
from llm_sad_sam.linkers.experimental.s_linker64 import SLinker64

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
WORD = r"[A-Za-z]+[A-Za-z0-9]*|\d+"

bad = 0


def check(label, ok, ill="*** FAILED ***"):
    global bad
    bad += not ok
    print(f"    {label:<74} {'OK' if ok else ill}")


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


def rename(text, *names):
    for name in names:
        text = text.replace(name, "SLinker59").replace(name.lower(), "s_linker59")
    return text


class Probe59(SLinker59):
    def __init__(self, aliases):                    # noqa: D107
        self.doc_knowledge = type("K", (), {"aliases": aliases})()


class Probe62(SLinker62):
    def __init__(self, aliases):                    # noqa: D107
        self.doc_knowledge = type("K", (), {"aliases": aliases})()


class Probe63(SLinker63):
    def __init__(self, aliases):                    # noqa: D107
        self.doc_knowledge = type("K", (), {"aliases": aliases})()


class Probe64(SLinker64):
    def __init__(self, aliases):                    # noqa: D107
        self.doc_knowledge = type("K", (), {"aliases": aliases})()


def load(project):
    text, model = DATASETS[project]
    return (load_sentences(str(BENCH / text)),
            parse_pcm_repository(str(BENCH / model)))


print("\n1. the null arm is s_linker59, byte for byte")
base, null = bodies(SLinker59), bodies(SLinker59Null)
check("same set of methods", set(base) == set(null))
differing = {n for n in base
             if rename(null[n], "SLinker59Null", "s_linker59_null") != base[n]}
check("no method body differs once the variant name is substituted back",
      not differing, ill=f"*** differ: {sorted(differing)} ***")
attrs = [a for a in vars(SLinker59)
         if a.isupper() and a != "_VARIANT_NAME" and not callable(vars(SLinker59)[a])]
check(f"all {len(attrs)} class attributes are s59's",
      all(getattr(SLinker59Null, a) == getattr(SLinker59, a) for a in attrs))
check("the variant name is the only declared difference",
      SLinker59Null._VARIANT_NAME == "s_linker59_null")

print("\n2. s62 changes the proposer and nothing else")
other = bodies(SLinker62)
check("same methods as s59 plus `_is_inflection_of`",
      set(other) - set(base) == {"_is_inflection_of"})
differing = {n for n in base if n in other
             and rename(other[n], "SLinker62", "s_linker62") != base[n]}
check("`_name_word_candidates` is the only shared method body that differs",
      differing == {"_name_word_candidates"},
      ill=f"*** differ: {sorted(differing)} ***")
check("all ten rule constants are s59's",
      all(getattr(L62, r) == getattr(L59, r) for r in RULES))
check(f"all {len(attrs)} class attributes are s59's",
      all(getattr(SLinker62, a) == getattr(SLinker59, a) for a in attrs))
for project in DATASETS:
    sentences, components = load(project)
    names = [c.name for c in components]
    check(f"{project}: every prompt builder renders exactly as s59's",
          SLinker62._prompt_extraction(names, [], sentences[:6])
          == SLinker59._prompt_extraction(names, [], sentences[:6])
          and SLinker62._prompt_doc_knowledge_judge(names, ["'x' -> " + names[0]])
          == SLinker59._prompt_doc_knowledge_judge(names, ["'x' -> " + names[0]]))

print("\n3. GATE-06: the inflection list is English morphology, not vocabulary")
check("no entry is longer than four characters",
      all(len(x) <= 4 for x in L62.INFLECTIONS))
check("no component name of any project contains an entry as a whole word",
      not [x for x in L62.INFLECTIONS if x and any(
          x in {w.casefold() for w in re.findall(WORD, c.name)}
          for project in DATASETS for c in load(project)[1])])
check("no entry appears in any document as a standalone word of its own",
      not [x for x in L62.INFLECTIONS
           if x in {"es", "ed", "ing", "ings", "ers"}
           and any(re.search(rf"(?<!\w){x}(?!\w)", s.text)
                   for project in DATASETS for s in load(project)[0])])

print("\n4. the predicate does what it says")
for surface, word, want in (("clients", "client", True), ("testing", "test", True),
                            ("tested", "test", True), ("recordings", "recording", True),
                            ("server", "server", True), ("servers", "server", True),
                            ("webrtc", "web", False), ("webcams", "web", False),
                            ("database", "db", False), ("common", "commons", False)):
    check(f"{surface!r} inflects {word!r}: {want}",
          SLinker62._is_inflection_of(surface, word) is want)

print("\n5. s63 changes exactly the span test")
other63 = bodies(SLinker63)
differing = {n for n in other if n in other63
             and rename(other63[n], "SLinker63", "s_linker63")
             != rename(other[n], "SLinker62", "s_linker62")}
check("`_inside_qualified_identifier` is the only method body differing from s62's",
      differing == {"_inside_qualified_identifier"},
      ill=f"*** differ: {sorted(differing)} ***")
check("all ten rule constants are s62's",
      all(getattr(L63, r) == getattr(L62, r) for r in RULES))
disagree, boundary = 0, 0
for project in DATASETS:
    sentences, _ = load(project)
    for sentence in sentences:
        for match in re.finditer(WORD, sentence.text):
            a = SLinker62._inside_qualified_identifier(
                sentence.text, match.start(), match.end())
            b = SLinker63._inside_qualified_identifier(
                sentence.text, match.start(), match.end())
            if a != b:
                disagree += 1
                boundary += (match.start() == 0
                             or match.end() == len(sentence.text))
check(f"the two spellings disagree on {disagree} spans, every one of them at a "
      f"sentence boundary", disagree and disagree == boundary)
check("s62 is the one that hides them (the guarded test is never the stricter one)",
      all(not (SLinker63._inside_qualified_identifier(s.text, m.start(), m.end())
               and not SLinker62._inside_qualified_identifier(
                   s.text, m.start(), m.end()))
          for project in DATASETS for s in load(project)[0]
          for m in re.finditer(WORD, s.text)))

print("\n6. the proposal sets match what the deterministic screen measured")
# The run-independent part of the alias table: terms every project's own names imply.
# The screen reads a recorded table; here the empty table is enough, because the two
# pairs at issue are in sentences that state no name of WebRTC-SFU under any table.
for project, adds, drops in (
        # s65 and s73 are gold; s5 is the one spurious candidate the change adds, and
        # the denotation judge approves it -- that is the measured FP +1.0.
        ("bigbluebutton", {(65, "WebRTC-SFU"), (73, "WebRTC-SFU"), (5, "WebRTC-SFU")},
         {(69, "BBB web")}),
        ("teammates", set(), set())):
    sentences, components = load(project)
    by_id = {c.id: c.name for c in components}
    old = {(k[0], by_id[k[1]]) for k in
           {(c.sentence_number, c.component_id)
            for c in Probe59({})._name_word_candidates(sentences, components)}}
    new = {(k[0], by_id[k[1]]) for k in
           {(c.sentence_number, c.component_id)
            for c in Probe62({})._name_word_candidates(sentences, components)}}
    check(f"{project}: s62 adds {sorted(adds)}", adds <= new - old)
    check(f"{project}: s62 drops {sorted(drops)}", drops <= old - new)
    check(f"{project}: it changes nothing else",
          (new - old) - adds == set() and (old - new) - drops == set(),
          ill=f"*** +{sorted((new - old) - adds)} -{sorted((old - new) - drops)} ***")

print("\n7. s64 adds the stated-name net at the full-name proposer and nothing else")
other64 = bodies(SLinker64)
check("same methods as s62 plus `_add_stated_name_net`",
      set(other64) - set(other) == {"_add_stated_name_net"})
differing = {n for n in other if n in other64
             and rename(other64[n], "SLinker64", "s_linker64")
             != rename(other[n], "SLinker62", "s_linker62")}
check("`_run_full_name_linker` is the only shared method body that differs",
      differing == {"_run_full_name_linker"},
      ill=f"*** differ: {sorted(differing)} ***")
check("all ten rule constants are s62's",
      all(getattr(L64, r) == getattr(L62, r) for r in RULES))
check("the net runs after the spelling variants and before the earlier-wins "
      "subtraction",
      inspect.getsource(SLinker64._run_full_name_linker).index(
          "_add_stated_name_net")
      > inspect.getsource(SLinker64._run_full_name_linker).index(
          "_add_spelling_variants")
      < inspect.getsource(SLinker64._run_full_name_linker).index("_unlinked"))
net_code = re.sub(r'""".*?"""', "", inspect.getsource(SLinker64._add_stated_name_net),
                  flags=re.S)
check("the net is case-sensitive: no IGNORECASE and no lenient primitive in its code",
      "IGNORECASE" not in net_code and "re.I" not in net_code
      and "_find_exact_form" not in net_code)
net_total, net_gold_names = 0, set()
for project in DATASETS:
    sentences, components = load(project)
    probe = Probe64({})
    before = probe._name_word_candidates(sentences, components)     # unrelated stage
    added = [c for c in probe._add_stated_name_net([], sentences, components)]
    caseless = [(s.number, c.id) for s in sentences for c in components
                if re.search(rf"(?<!\w){re.escape(c.name)}(?!\w)", s.text, re.I)]
    check(f"{project}: the net is a strict subset of its case-insensitive reading "
          f"({len(added)} of {len(caseless)})",
          {(c.sentence_number, c.component_id) for c in added} <= set(caseless)
          and len(added) <= len(caseless))
    check(f"{project}: every net candidate is labelled `stated_name_candidate`",
          all(c.source == "stated_name_candidate" for c in added))
    check(f"{project}: the net never overwrites an existing candidate",
          len(probe._add_stated_name_net(list(before), sentences, components))
          >= len(before))
    net_total += len(added)
print(f"    the net offers {net_total} (sentence, component) pairs across the five "
      f"documents before the extractor's own proposals are subtracted")

print(f"\n{'ALL CHECKS PASSED' if not bad else f'{bad} CHECK(S) FAILED'}")
sys.exit(1 if bad else 0)
