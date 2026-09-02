"""`s_linker110_noevidence`'s invariants: the four removals, and nothing else moved.

The RQ4 arm claims to remove *every context code computed* from the three judges and
to leave everything else alone. That claim is only worth a number if four things hold:

  1  **the removals happen.** No evidence block, no `[prev:]` in any case, the
     denotation step's window narrowed to the candidate's own sentence, and the
     resolver's `NAMED BEFORE THIS CASE` gone.
  2  **the coreference window survives.** The arm's whole defence is that every case
     stays answerable; a resolver prompt without its `SENTENCES` table prices
     impossibility instead. Asserted directly.
  3  **nothing else changed.** The base is `s_linker110`, only the four methods are
     declared, and every other attribute is the head's own object.
  4  **the narrowing is scoped.** `CONTEXT_SENTENCES` is restored after the denotation
     step -- including when it raises -- so the coreference resolver, which reads the
     same `_window` predicate, keeps its window.

Plus the pairing gate: `_VARIANT_NAME` must differ from the head's, or the two arms
clobber each other's phase states when run in one invocation.

No LLM calls.

    ../.venv/bin/python pilot/test_s110_noevidence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.core.data_types_v2 import DocumentKnowledge              # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92 import (                     # noqa: E402
    EvidenceBundle, SLinker92,
)
from llm_sad_sam.linkers.experimental.s_linker110 import SLinker110           # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker110_noevidence import (         # noqa: E402
    SLinker110NoEvidence,
)

#: What the arm is allowed to declare. Everything else must be inherited.
OWN_METHODS = {
    "__init__", "_format_evidence", "_prev_prefix",
    "_classify_denotations", "_prompt_coref",
}

CHECKS = []


def check(condition, label):
    CHECKS.append((bool(condition), label))
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")


class _NoCalls:
    def __getattr__(self, name):
        def explode(*_args, **_kwargs):
            raise AssertionError(f"the arm called the LLM: .{name}()")
        return explode


class _Sentence:
    def __init__(self, number, text):
        self.number, self.text = number, text


def build(cls, aliases=()):
    linker = cls.__new__(cls)                    # no backend, no credential
    linker.doc_knowledge = DocumentKnowledge(aliases=dict(aliases))
    linker.llm = _NoCalls()
    return linker


# ── 3. nothing else changed ──────────────────────────────────────────────────

print("\nstructure")
check(SLinker110NoEvidence.__mro__[1] is SLinker110, "the base is s_linker110")

declared = {n for n, v in vars(SLinker110NoEvidence).items()
            if callable(v) or isinstance(v, staticmethod)}
check(declared == OWN_METHODS,
      f"declares exactly the four overrides (+__init__): {sorted(declared)}")

def _identity(cls, name):
    """The underlying object, unbound.

    `getattr` on a classmethod builds a fresh bound method per access, so `is` can
    never hold across two classes even when the function is literally the same one.
    """
    value = getattr(cls, name, None)
    return getattr(value, "__func__", value)


drift = [a for a in dir(SLinker110)
         if a not in OWN_METHODS
         and not a.startswith("__")
         and a != "_VARIANT_NAME"                      # asserted to differ, below
         and _identity(SLinker110NoEvidence, a) is not _identity(SLinker110, a)]
check(not drift, f"every other attribute is the head's own object (drift: {drift})")

check(SLinker110NoEvidence._VARIANT_NAME != SLinker110._VARIANT_NAME,
      f"_VARIANT_NAME differs from the head's, so the arms can pair in one "
      f"invocation ({SLinker110NoEvidence._VARIANT_NAME!r})")


# ── 1. the removals happen ───────────────────────────────────────────────────

print("\nremovals")
arm = build(SLinker110NoEvidence)
head = build(SLinker110)

bundle = EvidenceBundle(
    source="scan",
    matched_span="preferences",
    mention_type="via known alias",
    preceding_text="The gui reads user settings on startup.",
    anchor_sentences=["S4: The preferences component stores them.",
                      "S9: JabRef reloads preferences on start."],
)
check(arm._format_evidence(bundle) == "", "the full-name Evidence block is empty")
check(head._format_evidence(bundle) != "", "  (the head's is not -- the check is live)")

sent_map = {10: _Sentence(10, "The gui reads user settings on startup."),
            11: _Sentence(11, "JabRef stores them in the preferences component.")}
check(arm._prev_prefix(11, sent_map) == "", "no [prev:] ahead of any case sentence")
check(head._prev_prefix(11, sent_map) != "", "  (the head's is not -- the check is live)")


# ── 4. the narrowing is scoped ───────────────────────────────────────────────

print("\nthe denotation window")
seen = {}


def _spy(self, candidates, sentences):
    seen["during"] = self.CONTEXT_SENTENCES
    return [], {}


original = SLinker110._classify_denotations
SLinker110._classify_denotations = _spy
try:
    arm._classify_denotations([], [])
finally:
    SLinker110._classify_denotations = original

check(seen.get("during") == 0, f"window is 0 during the step (saw {seen.get('during')})")
check(arm.CONTEXT_SENTENCES == SLinker92.CONTEXT_SENTENCES,
      f"restored afterwards to {SLinker92.CONTEXT_SENTENCES}")

sentences = [_Sentence(n, f"S{n}") for n in range(6, 17)]
check([s.number for s in arm._window(11, sentences)] == [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
      "the resolver's window is untouched outside the step")

arm.CONTEXT_SENTENCES = 0
check([s.number for s in arm._window(11, sentences)] == [11],
      "at 0 the window is the candidate's own sentence alone")
del arm.CONTEXT_SENTENCES


def _boom(self, candidates, sentences):
    raise RuntimeError("step failed")


SLinker110._classify_denotations = _boom
try:
    arm._classify_denotations([], [])
except RuntimeError:
    pass
finally:
    SLinker110._classify_denotations = original
check(arm.CONTEXT_SENTENCES == SLinker92.CONTEXT_SENTENCES,
      "restored even when the step raises")


# ── 1+2. the resolver prompt ─────────────────────────────────────────────────

print("\nthe resolver prompt")
comp_names = ["gui", "preferences", "logic"]
table = [{"sentence": 10, "text": "The gui reads user settings on startup."},
         {"sentence": 11, "text": "JabRef stores them in the preferences component."}]
targets = [{"case": 1, "target": 12, "text": "It also writes them back on exit."}]

arm_prompt = arm._prompt_coref(comp_names, table, targets)
head_prompt = head._prompt_coref(comp_names, table, targets)
base_prompt = SLinker92._prompt_coref(comp_names, table, targets)

check(arm_prompt == base_prompt, "renders s_linker92's resolver prompt byte for byte")
check(arm_prompt != head_prompt, "  (and differs from the head's -- the check is live)")
check("NAMED BEFORE THIS CASE" not in arm_prompt, "no NAMED BEFORE THIS CASE")
check("NAMED BEFORE THIS CASE" in head_prompt, "  (the head has it -- the check is live)")
check('"candidates"' not in arm_prompt, "no candidates reply field")
check("SENTENCES (the document text the cases are drawn from)" in arm_prompt,
      "the SENTENCES window SURVIVES -- the case stays answerable")
check("The gui reads user settings on startup." in arm_prompt,
      "and the antecedent is still in front of the model")
check("TARGET S12: It also writes them back on exit." in arm_prompt,
      "the target still carries its own text")


# ── verdict ──────────────────────────────────────────────────────────────────

passed = sum(1 for ok, _ in CHECKS if ok)
print(f"\n{passed}/{len(CHECKS)} checks passed")
sys.exit(0 if passed == len(CHECKS) else 1)
