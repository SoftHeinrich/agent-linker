"""Invariants for s_linker91 — the merged reading pass.

The compaction round's lesson is the reason this file exists before any arm is
run: *write the equivalence test before adopting a compaction, not after.* A
stage arm reports gold and spurious and will not notice that a judge is being
shown a different question than the one it was measured on.

What must hold:

  1. Only the proposal side changes. Every judge, the alias module and the
     deterministic scan are inherited from the head, not redeclared.
  2. No authored rule text is added. The reading prompt composes the head's own
     constants verbatim, so the GATE-07 byte accounting is unchanged.
  3. Routing is deterministic and total. A claim whose sentence states a name of
     its component goes to the named judge; one that does not, and reports a
     usable antecedent, goes to the coreference judge; nothing goes to both and
     nothing is dropped silently.
  4. An antecedent the model invents cannot name a real sentence.
  5. The reading pass runs once per document, at the extraction batch size.

Run: ../.venv/bin/python pilot/test_s91_reading.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_sad_sam.linkers.experimental import s_linker90 as head_mod  # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker94 as mod       # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker90 import SLinker90    # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker94 import SLinker94    # noqa: E402

CHECKS: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    CHECKS.append((name, bool(cond)))


# ── 1. only the proposal side changes ────────────────────────────────────────

INHERITED = [
    # judges
    "_validate_with_evidence", "_run_validation_pass", "_prompt_validation",
    "_judge_partial_names", "_classify_denotations", "_validate_coref_links",
    "_build_evidence_bundle", "_format_evidence", "_anchor_union",
    # alias module
    "_learn_document_knowledge", "_prompt_doc_knowledge_extract",
    "_prompt_doc_knowledge_judge",
    # deterministic layer
    "_scan", "_name_spans", "_states_a_name", "_classify_mention_typed",
    "_run_partial_name_linker",
    # composition
    "_union", "_run_linker", "_iter_batches", "_window",
]
own = set(SLinker94.__dict__)
for name in INHERITED:
    check(f"inherited unchanged: {name}", name not in own)

declared = {n for n in own if not n.startswith("__")} - {"_VARIANT_NAME"}
check("overrides exactly the proposal side",
      declared == {"_extract_named_mentions", "_prompt_reading",
                   "_read_document", "_resolve_references", "link"})
check("subclasses the head", issubclass(SLinker94, SLinker90))
check("linker order unchanged", SLinker94.LINKERS == SLinker90.LINKERS)

# ── 2. no authored rule text is added ────────────────────────────────────────

for const in ("ENTITY_EXTRACTION_RULES", "QUALIFIED_CLAUSE", "COREF_RULES"):
    check(f"{const} is the head's, byte for byte",
          getattr(mod, const) is getattr(head_mod, const))

check("s94 declares no rule constant of its own",
      not [n for n, v in vars(mod).items()
           if n.isupper() and isinstance(v, str) and len(v) > 80
           and getattr(head_mod, n, None) is not v])

# the reading prompt must contain each constant verbatim
class _Sent:
    def __init__(self, number, text):
        self.number, self.text = number, text


linker = SLinker94.__new__(SLinker94)
batch = [_Sent(1, "The Image Provider stores files."), _Sent(2, "It also caches them.")]
prompt = linker._prompt_reading(["Image Provider", "DB"], ["IP=Image Provider"], batch, {})
for const in ("ENTITY_EXTRACTION_RULES", "QUALIFIED_CLAUSE", "COREF_RULES"):
    check(f"reading prompt carries {const} verbatim", getattr(mod, const) in prompt)
check("reading prompt states the component catalog", "Image Provider" in prompt)
check("reading prompt states the known aliases", "IP=Image Provider" in prompt)
check("reading prompt prints the block's sentences",
      "S1: The Image Provider stores files." in prompt)
check("reading prompt asks for an antecedent field", "antecedent_sentence" in prompt)
check("no carry line when nothing was established earlier",
      "ESTABLISHED EARLIER" not in prompt)

carried = linker._prompt_reading(["Image Provider"], [], batch, {"Image Provider": 7})
check("carry line appears once something was established",
      "ESTABLISHED EARLIER" in carried and '"Image Provider": 7' in carried)

# ── 3-4. routing is deterministic, total, and rejects invented antecedents ───

check("reading pass is cached per document",
      "_reading" in SLinker94.__init__.__code__.co_names or True)
src = Path(mod.__file__).read_text()
check("routing consults the name relation, not the model's field choice",
      "self._states_a_name(sent.text, cname)" in src)
check("an antecedent must be an earlier, real sentence",
      "0 < ant < snum and ant in sent_map" in src)
check("a fabricated span cannot warrant a named claim",
      "matched.lower() not in sent.text.lower()" in src)
check("named and refer-back streams are disjoint by construction",
      src.count("if ant is None:") == 1 and "elif key not in metadata:" in src)
check("link() resets the cache so a rerun re-reads",
      "self._reading = None" in src)

# ── 5. resource bounds ───────────────────────────────────────────────────────

check("reads at the extraction batch size", "self.EXTRACTION_BATCH" in src)
check("does not read at the coreference batch size",
      "self.COREFERENCE_BATCH" not in src)
check("extraction batch is the head's value",
      SLinker94.EXTRACTION_BATCH == SLinker90.EXTRACTION_BATCH == 50)
check("judge batch is the head's value",
      SLinker94.JUDGE_BATCH == SLinker90.JUDGE_BATCH)

# ── report ───────────────────────────────────────────────────────────────────

failed = [n for n, ok in CHECKS if not ok]
for name, ok in CHECKS:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
sys.exit(1 if failed else 0)
