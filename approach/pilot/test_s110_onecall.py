"""`s_linker110_onecall`'s invariants: one call, the head's rules, nothing else.

The total floor is only a fair baseline if it removes the *arrangement* and keeps the
*guidance*. Five things have to hold:

  1  **the rules are the head's, verbatim.** All four rubrics rendered byte for byte
     from `s_linker92`, so a loss cannot be blamed on a weaker prompt.
  2  **the inputs are the raw ones.** Whole document, component list, and the alias
     table the knowledge stage discovered -- nothing computed about any pair.
  3  **nothing else runs.** No scan, no window, no evidence bundle, no shortlist, no
     judge, no union: exactly one LLM call after the knowledge stage.
  4  **the head's validity checks survive.** A returned sentence number must name a
     real sentence and a returned component must be in the catalog; duplicates collapse.
  5  **the run is scoreable.** `final` is written and links carry a source, so the
     runner's CSV and `score_runs.py` read this arm like any other.

No LLM calls.

    ../.venv/bin/python pilot/test_s110_onecall.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.core.data_types_v2 import DocumentKnowledge                  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92 import (                     # noqa: E402
    LAYERED_COREF_RULES, LAYERED_ENTITY_RULES, QUALIFIED_CLAUSE, STRICTER_CLAUSE,
)
from llm_sad_sam.linkers.experimental.s_linker110 import SLinker110           # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker110_onecall import (            # noqa: E402
    SLinker110OneCall,
)

CHECKS = []


def check(condition, label):
    CHECKS.append((bool(condition), label))
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")


class _Sentence:
    def __init__(self, number, text):
        self.number, self.text = number, text


DOC = [_Sentence(n, f"Sentence {n} body.") for n in range(1, 41)]
DOC[19] = _Sentence(20, "The Storage component persists submissions.")
DOC[38] = _Sentence(39, "It also prunes them nightly.")
COMP_NAMES = ["Storage", "Logic", "UI"]
ALIASES = {"the store": "Storage", "the frontend": "UI"}


def build(cls):
    linker = cls.__new__(cls)
    linker.doc_knowledge = DocumentKnowledge(aliases=dict(ALIASES))
    return linker


arm = build(SLinker110OneCall)
prompt = arm._prompt_one_call(COMP_NAMES, ALIASES, DOC)


# ── 1. the rules are the head's ──────────────────────────────────────────────

print("\nthe rules")
for name, text in (("LAYERED_ENTITY_RULES", LAYERED_ENTITY_RULES),
                   ("QUALIFIED_CLAUSE", QUALIFIED_CLAUSE),
                   ("STRICTER_CLAUSE", STRICTER_CLAUSE),
                   ("LAYERED_COREF_RULES", LAYERED_COREF_RULES)):
    check(text in prompt, f"{name} rendered verbatim")


# ── 2. the inputs are the raw ones ───────────────────────────────────────────

print("\nthe inputs")
missing = [s.number for s in DOC if f'"sentence": {s.number}' not in prompt]
check(not missing, f"every one of the {len(DOC)} sentences is present (missing: {missing})")
check(all(c in prompt for c in COMP_NAMES), "the component list is handed over")
check("the store=Storage" in prompt and "the frontend=UI" in prompt,
      "the discovered alias table is handed over (knowledge is RQ3's ablation, not this one)")
check("all three reference forms named"
      if ("whole name" in prompt and "one word" in prompt and "refers back" in prompt)
      else False,
      "the task names all three reference forms, so no form is silently out of scope")


# ── 3. nothing computed reaches the prompt ───────────────────────────────────

print("\nnothing computed")
for absent in ("Evidence: source=", "[prev:", "NAMED BEFORE THIS CASE",
               "SENTENCES (the document text", "--- Case", "TARGET S",
               "CASES", '"context"', '"candidates"'):
    check(absent not in prompt, f"no {absent!r}")
check('"claim"' not in prompt and "quote the EXACT words" not in prompt,
      "no quote is demanded (a deliberate second removal, worth 35.2 TP on its own)")


# ── 4+5. one call, the head's checks, a scoreable run ────────────────────────

print("\nthe call, the checks, the output")
calls, phases, saved = [], [], {}
reply = {"links": [
    {"sentence": 20, "component": "Storage"},
    {"sentence": 20, "component": "Storage"},     # duplicate -> collapses
    {"sentence": 39, "component": "Logic"},
    {"sentence": 999, "component": "Storage"},    # unreal sentence -> dropped
    {"sentence": 20, "component": "Nonexistent"}, # unknown component -> dropped
]}


class _LLM:
    def set_phase(self, phase):
        phases.append(phase)


arm.llm = _LLM()
arm._llm_calls = []
arm.no_knowledge = False
arm._ask = lambda p, **k: (calls.append(p), reply)[1]
arm._learn_document_knowledge = lambda s, c: DocumentKnowledge(aliases=dict(ALIASES))
arm._save_phase = lambda tp, name, state: saved.__setitem__(name, state)
arm._log = lambda *a, **k: None
arm._save_log = lambda tp: None

import llm_sad_sam.linkers.experimental.s_linker110_onecall as MOD            # noqa: E402


class _Comp:
    def __init__(self, name):
        self.name, self.id = name, f"id_{name}"


MOD.parse_pcm_repository = lambda mp: [_Comp(n) for n in COMP_NAMES]
MOD.load_sentences = lambda tp: DOC
MOD.build_sent_map = lambda s: {x.number: x for x in s}

links = arm.link("doc.txt", "model.repository")

check(len(calls) == 1, f"exactly one linking call (made {len(calls)})")
check(phases == ["phase_25_one_call"], f"one phase tag: {phases}")
check(sorted((l.sentence_number, l.component_name) for l in links)
      == [(20, "Storage"), (39, "Logic")],
      f"only valid, deduplicated links survive ({[(l.sentence_number, l.component_name) for l in links]})")
check(all(l.source == "one_call" for l in links), "links carry a source, so the CSV is well-formed")
check("knowledge" in saved and "final" in saved,
      f"knowledge and final checkpoints written ({sorted(saved)})")
check(not any(k.startswith("linker_") for k in saved),
      "no linker_* phases -- by construction, there are no stages to attribute")

declared = {n for n, v in vars(SLinker110OneCall).items()
            if callable(v) or isinstance(v, staticmethod)}
check(declared == {"__init__", "_prompt_one_call", "link"},
      f"declares only the prompt and the entry point: {sorted(declared)}")
check(SLinker110OneCall.__mro__[1] is SLinker110, "the base is s_linker110")
check(SLinker110OneCall._VARIANT_NAME != SLinker110._VARIANT_NAME,
      "_VARIANT_NAME is its own, so it pairs with the head in one invocation")


passed = sum(1 for ok, _ in CHECKS if ok)
print(f"\n{passed}/{len(CHECKS)} checks passed")
sys.exit(0 if passed == len(CHECKS) else 1)
