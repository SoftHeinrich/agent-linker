"""Invariants for the reading ladder's fallback arms, s_linker92 and s_linker93.

The compaction round's lesson: *write the equivalence test before adopting a
compaction, not after* — a stage arm reports gold and spurious and will not
notice that a call is being handed a different question than the one it was
measured on.

s93 restates the resolver's loop (it must batch over targets while windowing over
the whole document, which delegation cannot express). A restated loop is exactly
where a silent divergence hides, so the test below runs both loops with the LLM
stubbed out and asserts the **rendered prompts are byte-identical** when the
target set is the whole document.

Run: ../.venv/bin/python pilot/test_s9293_ladder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_sad_sam.linkers.experimental import s_linker90 as head_mod  # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker90 as mod92     # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker90 import SLinker90    # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker94 import SLinker94    # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker95 import SLinker95    # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker93 import SLinker93    # noqa: E402

CHECKS: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    CHECKS.append((name, bool(cond)))


class _Sent:
    def __init__(self, number, text):
        self.number, self.text = number, text


class _Comp:
    def __init__(self, cid, name):
        self.id, self.name = cid, name


def _stub(linker, sink):
    """Capture every prompt; answer nothing."""
    linker._ask = lambda prompt, **kw: (sink.append(prompt), {})[1]

    class _LLM:
        def set_phase(self, *_a, **_k):
            pass
    linker.llm = _LLM()
    linker.doc_knowledge = None
    return linker


# ── s95: one call, two ordered sections ──────────────────────────────────────

check("s95 overrides only the reading prompt",
      {n for n in SLinker95.__dict__ if not n.startswith("__")} ==
      {"_VARIANT_NAME", "_prompt_reading"})
check("s95 inherits s94's routing and reading pass", issubclass(SLinker95, SLinker94))

for const in ("ENTITY_EXTRACTION_RULES", "QUALIFIED_CLAUSE", "COREF_RULES"):
    check(f"s95 reuses the head's {const} object",
          getattr(mod92, const) is getattr(head_mod, const))

batch = [_Sent(1, "The Image Provider stores files."), _Sent(2, "It also caches them.")]
p92 = SLinker95.__new__(SLinker95)._prompt_reading(
    ["Image Provider", "DB"], ["IP=Image Provider"], batch, {"DB": 3})
for const in ("ENTITY_EXTRACTION_RULES", "QUALIFIED_CLAUSE", "COREF_RULES"):
    check(f"s95 prompt carries {const} verbatim", getattr(mod92, const) in p92)
check("s95 prompt orders the two steps",
      p92.index("STEP 1") < p92.index("STEP 2"))
check("s95 step 2 is grounded in step 1's own output",
      "you reported in STEP 1" in p92)
check("s95 prompt carries the established note", "ESTABLISHED EARLIER" in p92)
check("s95 asks for one schema, not two", p92.count('"references"') == 1)

# ── s93: same resolver, smaller target set ───────────────────────────────────

check("s93 overrides only the target set and the loop",
      {n for n in SLinker93.__dict__ if not n.startswith("__")} ==
      {"_VARIANT_NAME", "_nameless", "_resolve_references"})
check("s93 does not touch the resolver prompt", "_prompt_coref" not in SLinker93.__dict__)
check("s93 does not touch the strict judge",
      "_validate_coref_links" not in SLinker93.__dict__)
check("s93 keeps the head's coreference batch",
      SLinker93.COREFERENCE_BATCH == SLinker90.COREFERENCE_BATCH)
check("s93 keeps the head's context window",
      SLinker93.CONTEXT_SENTENCES == SLinker90.CONTEXT_SENTENCES)

# byte-identity: when every sentence is nameless, s93 must render the head's prompts
sents = [_Sent(i, f"Sentence number {i} says something.") for i in range(1, 26)]
comps = [_Comp("c1", "Zzzqqq"), _Comp("c2", "Wwwvvv")]   # names no sentence writes
sent_map = {s.number: s for s in sents}
name_to_id = {c.name: c.id for c in comps}

head_prompts: list[str] = []
_stub(SLinker90.__new__(SLinker90), head_prompts)._resolve_references(
    sents, comps, name_to_id, sent_map)
narrow_prompts: list[str] = []
narrow = _stub(SLinker93.__new__(SLinker93), narrow_prompts)
narrow._resolve_references(sents, comps, name_to_id, sent_map)

check("s93 selects every sentence when none writes a name",
      len(narrow._nameless(sents, comps)) == len(sents))
check("s93 issues the same number of resolver calls as the head",
      len(narrow_prompts) == len(head_prompts) and len(head_prompts) > 1)
check("s93 renders the head's resolver prompts BYTE-IDENTICALLY",
      narrow_prompts == head_prompts)

# and when names are present, the target set really shrinks
named = [_Sent(1, "The Zzzqqq stores files."), _Sent(2, "It also caches them."),
         _Sent(3, "Wwwvvv reads it.")]
kept = SLinker93.__new__(SLinker93)._nameless(named, comps)
check("s93 drops sentences that write a name",
      [s.number for s in kept] == [2])

# ── report ───────────────────────────────────────────────────────────────────

failed = [n for n, ok in CHECKS if not ok]
for name, ok in CHECKS:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
sys.exit(1 if failed else 0)
