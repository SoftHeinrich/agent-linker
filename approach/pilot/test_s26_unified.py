"""Design invariants of s_linker26. No LLM calls.

s26 is s25 with the two document-reading questions merged into one. The test
therefore checks two things: that the merge really happened, and that nothing
else moved -- every stage after the reading must be byte-identical to s25's, or
an end-to-end comparison between the two measures more than the merge.

Run from the approach/ directory:
    ../.venv/bin/python pilot/test_s26_unified.py
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, "src")

assert Path("src").is_dir(), "run from the approach/ directory"

from llm_sad_sam.core.document_loader_v2 import load_sentences
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental import s_linker25 as L25
from llm_sad_sam.linkers.experimental import s_linker26 as L26
from llm_sad_sam.linkers.experimental.s_linker25 import SLinker25
from llm_sad_sam.linkers.experimental.s_linker26 import SLinker26

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

bad = 0


def check(label, ok, good="OK", ill="*** FAILED ***"):
    global bad
    bad += not ok
    print(f"    {label:<52} {good if ok else ill}")


class _NoAliases:
    aliases: dict = {}


print("superclasses of SLinker26:", SLinker26.__mro__[1:])

# ── (a) the merge happened ───────────────────────────────────────────────────
print("\n(a) the two questions are one call")
check("no separate alias pass", not hasattr(SLinker26, "_learn_document_knowledge"))
check("no alias-extraction prompt",
      not hasattr(SLinker26, "_prompt_doc_knowledge_extract"))
check("no alias judge", not hasattr(SLinker26, "_prompt_doc_knowledge_judge")
      and not hasattr(L26, "DOC_KNOWLEDGE_JUDGE_RULES"))

names = ["Alpha", "Beta Gamma"]
sent = type("X", (), {"number": 1, "text": "Alpha does things"})()
prompt = SLinker26._prompt_extraction(names, ["b=Alpha"], [sent])
check("one prompt asks for both answers",
      '"references"' in prompt and '"aliases"' in prompt)
check("  it carries the accumulated table forward", "KNOWN ALIASES: b=Alpha" in prompt)
check("  it keeps s25's extraction rubric",
      L25.ENTITY_EXTRACTION_RULES in prompt)
check("  it keeps s25's alias-discovery rubric",
      L25.DOC_KNOWLEDGE_EXTRACTION_RULES in prompt
      and L25.ALIAS_EXCLUSION_RULES in prompt)

reading = inspect.getsource(SLinker26._extract_named_mentions)
check("the reading is sequential over batches", "_iter_batches(" in reading)
check("it feeds the table forward", "mappings" in reading and "table" in reading)
check("it writes the table every later stage reads",
      "self.doc_knowledge = DocumentKnowledge()" in reading)
check("it honours no_knowledge", "self.no_knowledge" in reading)

link_src = inspect.getsource(SLinker26.link)
check("link() has no knowledge stage",
      "_learn_document_knowledge" not in link_src)
check("link() still checkpoints the table once it exists",
      'self._save_phase(text_path, "knowledge"' in link_src)

# ── (b) nothing after the reading moved ──────────────────────────────────────
print("\n(b) every stage after the reading is s25's, byte for byte")
SHARED_METHODS = [
    "_unlinked", "_union", "_keep_stated_names", "_add_spelling_variants",
    "_spelling_variant_candidates", "_name_signature", "_in_dotted_path",
    "_inside_qualified_identifier", "_all_occurrences_in_qualified_path",
    "_classify_mention_typed", "_build_evidence_bundle", "_format_evidence",
    "_validate_with_evidence", "_run_validation_pass", "_prompt_validation",
    "_run_partial_name_linker", "_name_word_candidates", "_judge_partial_names",
    "_classify_denotations", "_review_identity", "_review_identity_batch",
    "_run_coreference_linker", "_resolve_references", "_antecedent_states_name",
    "_validate_coref_links", "_prompt_coref", "_find_exact_form",
    "_names_by_component", "_prev_prefix", "_iter_batches", "_ask",
]
for method in SHARED_METHODS:
    a = inspect.getsource(getattr(SLinker25, method))
    b = inspect.getsource(getattr(SLinker26, method))
    check(method, a == b, "IDENTICAL", "*** DIFFERS ***")

SHARED_CONSTANTS = [
    "ENTITY_EXTRACTION_RULES", "DOC_KNOWLEDGE_EXTRACTION_RULES",
    "ALIAS_EXCLUSION_RULES", "P1_FOCUS", "P2_FOCUS", "COREF_VALIDATION_FOCUS",
    "COREF_RULES", "LAYERED_ENTITY_RULES", "LAYERED_COREF_RULES",
]
print("\n    shared rubrics:")
for const in SHARED_CONSTANTS:
    check(const, getattr(L25, const) == getattr(L26, const),
          "IDENTICAL", "*** DIFFERS ***")

print("\n    resource bounds:")
for bound in ("CONTEXT_SENTENCES", "ANCHOR_LIMIT", "EXTRACTION_BATCH",
              "JUDGE_BATCH", "COREFERENCE_BATCH", "ASK_ATTEMPTS", "LINKERS"):
    check(bound, getattr(SLinker25, bound) == getattr(SLinker26, bound),
          "IDENTICAL", "*** DIFFERS ***")

# ── (c) the deterministic generators still agree ─────────────────────────────
print("\n(c) deterministic generators agree with s25 on every benchmark")
for name, (text_path, model_path) in DATASETS.items():
    sents = load_sentences(str(BENCH / text_path))
    comps = parse_pcm_repository(str(BENCH / model_path))
    old = SLinker25.__new__(SLinker25)
    old.doc_knowledge = _NoAliases()
    new = SLinker26.__new__(SLinker26)
    new.doc_knowledge = _NoAliases()
    for label, call in (
        ("spelling-variant", lambda l: l._spelling_variant_candidates(sents, comps)),
        ("partial-name", lambda l: l._name_word_candidates(sents, comps)),
    ):
        a = {(c.sentence_number, c.component_id, c.matched_text) for c in call(old)}
        b = {(c.sentence_number, c.component_id, c.matched_text) for c in call(new)}
        check(f"{name}: {label} ({len(b)})", a == b, "MATCH", "*** DIFF ***")

print(f"\n{'ALL CHECKS PASS' if bad == 0 else f'{bad} FAILURES'}")
raise SystemExit(1 if bad else 0)
