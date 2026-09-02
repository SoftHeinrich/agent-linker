"""`s_linker110_nocoderef`'s invariants: the resolver gets the document, and nothing else.

The floor arm claims the resolver is handed only what a reader would be handed. That
is worth a number if four things hold:

  1  **nothing computed reaches the prompt.** No targets, no window table, no
     NAMED BEFORE THIS CASE, no per-case context range.
  2  **the whole document does.** Every sentence, with its number -- including the ones
     the head's window would never have shown for a given target.
  3  **one call, not COREFERENCE_BATCH of them.** Batching is itself a code-computed
     decision, so the floor cannot keep it.
  4  **the reply contract is the head's.** Same fields, so the parser is unchanged and
     the strict coreference judge downstream sees what it always sees; and a resolution
     this arm keeps is one the head would keep from the same reply.

No LLM calls.

    ../.venv/bin/python pilot/test_s110_nocoderef.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.core.data_types_v2 import DocumentKnowledge                  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92 import COREF_RULES           # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker110 import SLinker110           # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker110_noevidence import (         # noqa: E402
    SLinker110NoEvidence,
)
from llm_sad_sam.linkers.experimental.s_linker110_nocoderef import (          # noqa: E402
    SLinker110NoCodeRef,
)

OWN_METHODS = {"__init__", "_prompt_coref_document", "_resolve_references"}

CHECKS = []


def check(condition, label):
    CHECKS.append((bool(condition), label))
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")


class _Sentence:
    def __init__(self, number, text):
        self.number, self.text = number, text


class _Component:
    def __init__(self, name):
        self.name = name


class _OneCall:
    """An LLM stub that records every prompt and answers once."""

    def __init__(self, reply):
        self.prompts, self.reply, self.phases = [], reply, []

    def set_phase(self, phase):
        self.phases.append(phase)


def build(cls):
    linker = cls.__new__(cls)
    linker.doc_knowledge = DocumentKnowledge(aliases={})
    return linker


DOC = [_Sentence(n, f"Sentence {n} body.") for n in range(1, 41)]
DOC[19] = _Sentence(20, "The Storage component persists submissions.")
DOC[38] = _Sentence(39, "It also prunes them nightly.")
SENT_MAP = {s.number: s for s in DOC}
COMP_NAMES = ["Storage", "Logic", "UI"]
COMPONENTS = [_Component(n) for n in COMP_NAMES]
NAME_TO_ID = {n: f"id_{n}" for n in COMP_NAMES}


# ── 4a. structure ────────────────────────────────────────────────────────────

print("\nstructure")
check(SLinker110NoCodeRef.__mro__[1] is SLinker110NoEvidence,
      "the base is s_linker110_noevidence (it inherits the other removals)")
declared = {n for n, v in vars(SLinker110NoCodeRef).items()
            if callable(v) or isinstance(v, staticmethod)}
check(declared == OWN_METHODS,
      f"declares only the resolver and its prompt: {sorted(declared)}")
check(SLinker110NoCodeRef._VARIANT_NAME
      not in (SLinker110._VARIANT_NAME, SLinker110NoEvidence._VARIANT_NAME),
      f"_VARIANT_NAME is its own ({SLinker110NoCodeRef._VARIANT_NAME!r})")


# ── 1+2. the prompt ──────────────────────────────────────────────────────────

print("\nthe resolver prompt")
arm = build(SLinker110NoCodeRef)
head = build(SLinker110)
prompt = arm._prompt_coref_document(COMP_NAMES, DOC)

check("NAMED BEFORE THIS CASE" not in prompt, "no NAMED BEFORE THIS CASE")
check("--- Case" not in prompt and "TARGET S" not in prompt,
      "no targets: the model is not told which sentences to look at")
check("SENTENCES (the document text the cases are drawn from)" not in prompt,
      "no window table")
check('"context"' not in prompt, "no per-case context range")
check("DOCUMENT" in prompt, "the document is handed over whole")

missing = [s.number for s in DOC if f'"sentence": {s.number}' not in prompt]
check(not missing, f"every one of the {len(DOC)} sentences is present (missing: {missing})")
check("The Storage component persists submissions." in prompt
      and "It also prunes them nightly." in prompt,
      "including an antecedent 19 sentences before its refer-back, "
      "which the head's +/-5 window could never have shown")
check(COREF_RULES in prompt, "COREF_RULES rendered verbatim (rules are not this ablation)")
check(all(c in prompt for c in COMP_NAMES), "the component list is handed over")


# ── 4b. the reply contract ───────────────────────────────────────────────────

print("\nthe reply contract")
for field in ('"sentence"', '"reference"', '"component"',
              '"antecedent_sentence"', '"antecedent_text"'):
    check(field in prompt, f"reply schema still asks for {field}")
check('"candidates"' not in prompt, "no candidates field (that is s110's shortlist)")


# ── 3+4c. one call, and the head's parsing ───────────────────────────────────

print("\nthe call and the parsing")
reply = {"resolutions": [
    {"sentence": 39, "reference": "It", "component": "Storage",
     "antecedent_sentence": 20, "antecedent_text": "The Storage component"},
    {"sentence": 39, "reference": "It", "component": "Nonexistent",       # unknown comp
     "antecedent_sentence": 20, "antecedent_text": "x"},
    {"sentence": 999, "reference": "It", "component": "Storage",          # unreal sentence
     "antecedent_sentence": 20, "antecedent_text": "x"},
    {"sentence": 39, "reference": "It", "component": "Logic",             # no antecedent
     "antecedent_text": "x"},
    {"sentence": 39, "reference": "It", "component": "UI",                # unreal antecedent
     "antecedent_sentence": 998, "antecedent_text": "x"},
]}

calls = []


def _ask(prompt, **kwargs):
    calls.append(prompt)
    return reply


arm.llm = _OneCall(reply)
arm._ask = _ask
links, meta = arm._resolve_references(DOC, COMPONENTS, NAME_TO_ID, SENT_MAP)

check(len(calls) == 1, f"exactly one call for the whole document (made {len(calls)})")
check(arm.llm.phases == ["phase_25_coreference"],
      f"the head's phase tag, so the log and the checkpoint agree: {arm.llm.phases}")
check(len(links) == 1 and links[0].sentence_number == 39
      and links[0].component_name == "Storage",
      f"only the valid resolution survives ({[(l.sentence_number, l.component_name) for l in links]})")
check(links[0].source == "coreference", "tagged as a coreference link, as the head tags it")
m = meta.get((39, "id_Storage"), {})
check(m.get("reference") == "It" and m.get("antecedent_sentence") == 20
      and m.get("antecedent_text") == "The Storage component",
      "the metadata the strict judge reads is populated exactly as the head populates it")
check("raw_resolution" in m, "the raw reply is carried, as the head carries it")


# ── verdict ──────────────────────────────────────────────────────────────────

passed = sum(1 for ok, _ in CHECKS if ok)
print(f"\n{passed}/{len(CHECKS)} checks passed")
sys.exit(0 if passed == len(CHECKS) else 1)
