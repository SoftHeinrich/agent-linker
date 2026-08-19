"""Design invariants of s_linker44. No LLM calls.

s44 changes one thing in s_linker25: the mention label's two stated-name values,
which differed only in the case of the match, become one. The field is still built
for every candidate and still rendered on every case, because s_linker43 -- which
collapsed the label further *and* dropped the field for its residual value -- lost
1.3 F1 and 1.3 F2 at the n=3 p-floor. This test pins the difference to the enum and
the classifier, and shows the relabelling is exactly the intended merge.

Run from the approach/ directory:
    ../.venv/bin/python pilot/test_s44_nocasegrade.py
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, "src")

assert Path("src").is_dir(), "run from the approach/ directory"

from llm_sad_sam.core.document_loader_v2 import load_sentences
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental import s_linker25 as L25
from llm_sad_sam.linkers.experimental import s_linker44 as L44
from llm_sad_sam.linkers.experimental.s_linker25 import SLinker25
from llm_sad_sam.linkers.experimental.s_linker44 import MentionType, SLinker44

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
    print(f"    {label:<58} {good if ok else ill}")


class _NoAliases:
    aliases: dict = {}


print("\n(a) four values, and no case rule left")
check("four values", [m.name for m in MentionType]
      == ["STATED", "CODE_TOKEN", "VIA_ALIAS", "INDIRECT"])
check("the three untouched values keep s25's exact strings",
      [MentionType.CODE_TOKEN.value, MentionType.VIA_ALIAS.value,
       MentionType.INDIRECT.value]
      == [L25.MentionType.CODE_TOKEN.value, L25.MentionType.VIA_ALIAS.value,
          L25.MentionType.INDIRECT.value])
classifier = inspect.getsource(SLinker44._classify_mention_typed)
check("no comparison of the matched surface to the name",
      "== comp_name" not in classifier)
check("the classifier still returns a value for every candidate",
      "return None" not in classifier)

print("\n(b) the field is still built and still rendered — s43's other half is not here")
for method in ("_build_evidence_bundle", "_format_evidence",
               "_validate_with_evidence", "_prompt_validation",
               "_run_validation_pass"):
    check(method, inspect.getsource(getattr(SLinker25, method))
          == inspect.getsource(getattr(SLinker44, method)),
          "IDENTICAL", "*** DIFFERS ***")

print("\n(c) every other method is s_linker25's, byte for byte")
SHARED_METHODS = [
    "link", "_run_linker", "_unlinked", "_union", "_keep_stated_names",
    "_add_spelling_variants", "_spelling_variant_candidates", "_name_signature",
    "_in_dotted_path", "_inside_qualified_identifier",
    "_all_occurrences_in_qualified_path", "_learn_document_knowledge",
    "_extract_named_mentions", "_run_extraction_pass", "_run_full_name_linker",
    "_prompt_doc_knowledge_extract", "_prompt_doc_knowledge_judge",
    "_prompt_extraction", "_run_partial_name_linker", "_name_word_candidates",
    "_judge_partial_names", "_classify_denotations", "_review_identity",
    "_review_identity_batch", "_run_coreference_linker", "_resolve_references",
    "_antecedent_states_name", "_validate_coref_links", "_prompt_coref",
    "_find_exact_form", "_names_by_component", "_prev_prefix", "_iter_batches",
    "_ask",
]
for method in SHARED_METHODS:
    check(method, inspect.getsource(getattr(SLinker25, method))
          == inspect.getsource(getattr(SLinker44, method)),
          "IDENTICAL", "*** DIFFERS ***")

print("\n    shared rubrics:")
for const in ("ENTITY_EXTRACTION_RULES", "DOC_KNOWLEDGE_EXTRACTION_RULES",
              "DOC_KNOWLEDGE_JUDGE_RULES", "ALIAS_EXCLUSION_RULES", "P1_FOCUS",
              "P2_FOCUS", "COREF_VALIDATION_FOCUS", "COREF_RULES",
              "LAYERED_ENTITY_RULES", "LAYERED_COREF_RULES"):
    check(const, getattr(L25, const) == getattr(L44, const),
          "IDENTICAL", "*** DIFFERS ***")

print("\n    resource bounds:")
for bound in ("CONTEXT_SENTENCES", "ANCHOR_LIMIT", "EXTRACTION_BATCH",
              "JUDGE_BATCH", "COREFERENCE_BATCH", "ASK_ATTEMPTS", "LINKERS"):
    check(bound, getattr(SLinker25, bound) == getattr(SLinker44, bound),
          "IDENTICAL", "*** DIFFERS ***")

print("\n(d) the relabelling is exactly the intended merge, on every benchmark")
INTENDED = {
    "proper case, standalone": "the name itself",
    "lowercase mention": "the name itself",
    "lowercase, inside qualified name": "lowercase, inside qualified name",
    "via known alias": "via known alias",
    "indirect/unclear match": "indirect/unclear match",
}
for name, (text_path, model_path) in DATASETS.items():
    sents = load_sentences(str(BENCH / text_path))
    comps = parse_pcm_repository(str(BENCH / model_path))
    old = SLinker25.__new__(SLinker25)
    old.doc_knowledge = _NoAliases()
    new = SLinker44.__new__(SLinker44)
    new.doc_knowledge = _NoAliases()
    for label, call in (
        ("spelling-variant", lambda l: l._spelling_variant_candidates(sents, comps)),
        ("partial-name", lambda l: l._name_word_candidates(sents, comps)),
    ):
        a = {(c.sentence_number, c.component_id, c.matched_text) for c in call(old)}
        b = {(c.sentence_number, c.component_id, c.matched_text) for c in call(new)}
        check(f"{name}: {label} ({len(b)})", a == b, "MATCH", "*** DIFF ***")
    mismatch = pairs = merged = 0
    for sentence in sents:
        for component in comps:
            pairs += 1
            was = old._classify_mention_typed(component.name, sentence.text).value
            now = new._classify_mention_typed(component.name, sentence.text).value
            mismatch += INTENDED[was] != now
            merged += was in ("proper case, standalone", "lowercase mention")
    check(f"{name}: {pairs} pairs, {merged} in the merged value", mismatch == 0,
          "MATCH", f"*** {mismatch} UNINTENDED ***")

print(f"\n{'ALL CHECKS PASS' if bad == 0 else f'{bad} FAILURES'}")
raise SystemExit(1 if bad else 0)
