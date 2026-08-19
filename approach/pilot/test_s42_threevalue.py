"""Design invariants of s_linker42. No LLM calls.

s42 carries exactly two changes away from s_linker25: the full-name judge asks its
two criteria in one call (inherited from s_linker36), and the mention label has three
values instead of five. The test therefore checks that both changes are present and
that *nothing else* moved -- otherwise an end-to-end comparison against s25 measures
more than the two changes.

Run from the approach/ directory:
    ../.venv/bin/python pilot/test_s42_threevalue.py
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, "src")

assert Path("src").is_dir(), "run from the approach/ directory"

from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental import s_linker25 as L25
from llm_sad_sam.linkers.experimental import s_linker36 as L36
from llm_sad_sam.linkers.experimental import s_linker42 as L42
from llm_sad_sam.linkers.experimental.s_linker25 import SLinker25
from llm_sad_sam.linkers.experimental.s_linker36 import SLinker36
from llm_sad_sam.linkers.experimental.s_linker42 import MentionType, SLinker42

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
    print(f"    {label:<56} {good if ok else ill}")


class _NoAliases:
    aliases: dict = {}


# ── (a) change 1: one judging call, both criteria ────────────────────────────
print("\n(a) the full-name judge asks both criteria in one call")
check("_prompt_two_criteria exists", hasattr(SLinker42, "_prompt_two_criteria"))
check("  it is s_linker36's, byte for byte",
      inspect.getsource(SLinker42._prompt_two_criteria)
      == inspect.getsource(SLinker36._prompt_two_criteria),
      "IDENTICAL", "*** DIFFERS ***")
check("both focus texts are s_linker25's",
      L42.P1_FOCUS == L25.P1_FOCUS and L42.P2_FOCUS == L25.P2_FOCUS)
judge = inspect.getsource(SLinker42._validate_with_evidence)
check("one sample, not two", judge.count("_run_two_criteria_pass(") == 1)
check("a link still needs both criteria", "p1 and p2" in judge)

# ── (b) change 2: three mention values ───────────────────────────────────────
print("\n(b) the mention label has three values and no case rule")
check("three values", [m.name for m in MentionType]
      == ["NAME", "ALIAS", "QUALIFIED_ONLY"])
classifier = inspect.getsource(SLinker42._classify_mention_typed)
check("no case comparison of the matched surface",
      "== comp_name" not in classifier)
check("returns None when the name is not present", "return None" in classifier)
check("the bundle omits the field instead of carrying a filler value",
      'mention_type = mention.value if mention else ""'
      in inspect.getsource(SLinker42._build_evidence_bundle))
check("_format_evidence drops the field when it is empty",
      "if bundle.mention_type" in inspect.getsource(SLinker42._format_evidence))

linker = SLinker42.__new__(SLinker42)
linker.doc_knowledge = _NoAliases()
for text, want in (
    ("The Registry stores metadata.", MentionType.NAME),
    ("the registry stores metadata.", MentionType.NAME),
    ("See core.registry.get for details.", MentionType.QUALIFIED_ONLY),
    ("Nothing relevant here.", None),
):
    got = linker._classify_mention_typed("Registry", text)
    check(f'"{text[:34]}" -> {got}', got is want)

linker.doc_knowledge = type("K", (), {"aliases": {"Reg": "Registry"}})()
check('an introduced short form is ALIAS',
      linker._classify_mention_typed("Registry", "Reg accepts requests.")
      is MentionType.ALIAS)

# ── (c) nothing else moved ───────────────────────────────────────────────────
print("\n(c) every other method is s_linker25's, byte for byte")
SHARED_METHODS = [
    "link", "_run_linker", "_unlinked", "_union", "_keep_stated_names",
    "_add_spelling_variants", "_spelling_variant_candidates", "_name_signature",
    "_in_dotted_path", "_inside_qualified_identifier",
    "_all_occurrences_in_qualified_path", "_learn_document_knowledge",
    "_extract_named_mentions", "_run_extraction_pass", "_run_full_name_linker",
    "_prompt_doc_knowledge_extract", "_prompt_doc_knowledge_judge",
    "_prompt_extraction", "_prompt_validation", "_run_validation_pass",
    "_run_partial_name_linker", "_name_word_candidates", "_judge_partial_names",
    "_classify_denotations", "_review_identity", "_review_identity_batch",
    "_run_coreference_linker", "_resolve_references", "_antecedent_states_name",
    "_validate_coref_links", "_prompt_coref", "_find_exact_form",
    "_names_by_component", "_prev_prefix", "_iter_batches", "_ask",
]
for method in SHARED_METHODS:
    a = inspect.getsource(getattr(SLinker25, method))
    b = inspect.getsource(getattr(SLinker42, method))
    check(method, a == b, "IDENTICAL", "*** DIFFERS ***")

print("\n    shared rubrics:")
for const in ("ENTITY_EXTRACTION_RULES", "DOC_KNOWLEDGE_EXTRACTION_RULES",
              "DOC_KNOWLEDGE_JUDGE_RULES", "ALIAS_EXCLUSION_RULES", "P1_FOCUS",
              "P2_FOCUS", "COREF_VALIDATION_FOCUS", "COREF_RULES",
              "LAYERED_ENTITY_RULES", "LAYERED_COREF_RULES"):
    check(const, getattr(L25, const) == getattr(L42, const),
          "IDENTICAL", "*** DIFFERS ***")

print("\n    resource bounds:")
for bound in ("CONTEXT_SENTENCES", "ANCHOR_LIMIT", "EXTRACTION_BATCH",
              "JUDGE_BATCH", "COREFERENCE_BATCH", "ASK_ATTEMPTS", "LINKERS"):
    check(bound, getattr(SLinker25, bound) == getattr(SLinker42, bound),
          "IDENTICAL", "*** DIFFERS ***")

# ── (d) deterministic generators and the label's real effect ─────────────────
print("\n(d) candidate generators agree with s25; the label differs only as designed")
for name, (text_path, model_path) in DATASETS.items():
    sents = load_sentences(str(BENCH / text_path))
    comps = parse_pcm_repository(str(BENCH / model_path))
    old = SLinker25.__new__(SLinker25)
    old.doc_knowledge = _NoAliases()
    new = SLinker42.__new__(SLinker42)
    new.doc_knowledge = _NoAliases()
    for label, call in (
        ("spelling-variant", lambda l: l._spelling_variant_candidates(sents, comps)),
        ("partial-name", lambda l: l._name_word_candidates(sents, comps)),
    ):
        a = {(c.sentence_number, c.component_id, c.matched_text) for c in call(old)}
        b = {(c.sentence_number, c.component_id, c.matched_text) for c in call(new)}
        check(f"{name}: {label} ({len(b)})", a == b, "MATCH", "*** DIFF ***")

    # every (name, sentence) pair: the two labels must agree up to the intended map
    intended = {
        "proper case, standalone": "the name itself",
        "lowercase mention": "the name itself",
        "lowercase, inside qualified name": "only inside a qualified identifier",
        "via known alias": "a name the document introduces for it",
        "indirect/unclear match": None,
    }
    mismatch = 0
    pairs = 0
    for sentence in sents:
        for component in comps:
            pairs += 1
            was = old._classify_mention_typed(component.name, sentence.text).value
            now = new._classify_mention_typed(component.name, sentence.text)
            if intended[was] != (now.value if now else None):
                mismatch += 1
    check(f"{name}: {pairs} pairs relabel exactly as intended", mismatch == 0,
          "MATCH", f"*** {mismatch} UNINTENDED ***")

print(f"\n{'ALL CHECKS PASS' if bad == 0 else f'{bad} FAILURES'}")
raise SystemExit(1 if bad else 0)
