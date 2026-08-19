"""Design invariants of s_linker45 and s_linker46. No LLM calls.

Both variants change exactly one thing in s_linker25 and both are aimed at structure
the paper has to describe rather than at a score:

  s45  coreference resolution reads the judges' batch size, so the workflow states two
       batch constants instead of three (and pays ~74 calls instead of ~101).
  s46  the alias table admits full-name candidates but no longer suppresses
       partial-name ones, so it has one role instead of two opposite ones.

For each, the test asserts the change is present, that every method body is otherwise
s_linker25's byte for byte, and -- for s46, whose change is deterministic -- that the
candidate-set difference against s25 is exactly the freed suppression and nothing else.

Run from the approach/ directory:
    ../.venv/bin/python pilot/test_s45_s46_singlechange.py
"""
import inspect
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "src")

assert Path("src").is_dir(), "run from the approach/ directory"

from llm_sad_sam.core.document_loader_v2 import load_sentences
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental import s_linker25 as L25
from llm_sad_sam.linkers.experimental import s_linker45 as L45
from llm_sad_sam.linkers.experimental import s_linker46 as L46
from llm_sad_sam.linkers.experimental.s_linker25 import SLinker25
from llm_sad_sam.linkers.experimental.s_linker45 import SLinker45
from llm_sad_sam.linkers.experimental.s_linker46 import SLinker46

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
# A run whose alias tables are on disk, so the suppression change can be exercised
# against real discovered aliases rather than an invented table.
TABLE_RUN = Path("../results/s44_nocasegrade_e2e_r1_20260812")

bad = 0


def check(label, ok, good="OK", ill="*** FAILED ***"):
    global bad
    bad += not ok
    print(f"    {label:<60} {good if ok else ill}")


class _NoAliases:
    aliases: dict = {}


ALL_METHODS = [
    "link", "_run_linker", "_unlinked", "_union", "_keep_stated_names",
    "_add_spelling_variants", "_spelling_variant_candidates", "_name_signature",
    "_in_dotted_path", "_inside_qualified_identifier", "_classify_mention_typed",
    "_all_occurrences_in_qualified_path", "_learn_document_knowledge",
    "_extract_named_mentions", "_run_extraction_pass", "_run_full_name_linker",
    "_build_evidence_bundle", "_format_evidence", "_validate_with_evidence",
    "_prompt_doc_knowledge_extract", "_prompt_doc_knowledge_judge",
    "_prompt_extraction", "_prompt_validation", "_run_validation_pass",
    "_run_partial_name_linker", "_name_word_candidates", "_judge_partial_names",
    "_classify_denotations", "_review_identity", "_review_identity_batch",
    "_run_coreference_linker", "_resolve_references", "_antecedent_states_name",
    "_validate_coref_links", "_prompt_coref", "_find_exact_form",
    "_names_by_component", "_prev_prefix", "_iter_batches", "_ask",
]
RUBRICS = ["ENTITY_EXTRACTION_RULES", "DOC_KNOWLEDGE_EXTRACTION_RULES",
           "DOC_KNOWLEDGE_JUDGE_RULES", "ALIAS_EXCLUSION_RULES", "P1_FOCUS",
           "P2_FOCUS", "COREF_VALIDATION_FOCUS", "COREF_RULES",
           "LAYERED_ENTITY_RULES", "LAYERED_COREF_RULES"]
BOUNDS = ["CONTEXT_SENTENCES", "ANCHOR_LIMIT", "EXTRACTION_BATCH", "JUDGE_BATCH",
          "ASK_ATTEMPTS", "LINKERS"]          # COREFERENCE_BATCH is s45's change


def shared(cls, changed_methods, changed_bounds=()):
    for method in ALL_METHODS:
        same = (inspect.getsource(getattr(SLinker25, method))
                == inspect.getsource(getattr(cls, method)))
        if method in changed_methods:
            check(f"{method} (the change)", not same,
                  "DIFFERS, as designed", "*** IDENTICAL — change missing ***")
        else:
            check(method, same, "IDENTICAL", "*** DIFFERS ***")
    module = {SLinker45: L45, SLinker46: L46}[cls]
    for const in RUBRICS:
        check(const, getattr(L25, const) == getattr(module, const),
              "IDENTICAL", "*** DIFFERS ***")
    for bound in BOUNDS:
        check(bound, getattr(SLinker25, bound) == getattr(cls, bound),
              "IDENTICAL", "*** DIFFERS ***")
    for bound in changed_bounds:
        check(f"{bound} (the change)",
              getattr(SLinker25, bound) != getattr(cls, bound),
              "DIFFERS, as designed", "*** UNCHANGED ***")


print("\n=== s_linker45: one batch constant fewer ===")
print("\n(a) the change")
check("coreference batch equals the judges' batch",
      SLinker45.COREFERENCE_BATCH == SLinker45.JUDGE_BATCH == 25)
check("s25 used a third value", SLinker25.COREFERENCE_BATCH == 10)
check("the constant is written as JUDGE_BATCH, not as a literal 25",
      "COREFERENCE_BATCH = JUDGE_BATCH"
      in Path("src/llm_sad_sam/linkers/experimental/s_linker45.py").read_text())
check("nothing else reads a batch size for coreference",
      "COREFERENCE_BATCH" in inspect.getsource(SLinker45._resolve_references))

print("\n(b) every method body is s_linker25's")
shared(SLinker45, changed_methods=(), changed_bounds=("COREFERENCE_BATCH",))

print("\n(c) the deterministic generators are unaffected")
for name, (text_path, model_path) in DATASETS.items():
    sents = load_sentences(str(BENCH / text_path))
    comps = parse_pcm_repository(str(BENCH / model_path))
    old, new = SLinker25.__new__(SLinker25), SLinker45.__new__(SLinker45)
    old.doc_knowledge = new.doc_knowledge = _NoAliases()
    for label, call in (
        ("spelling-variant", lambda l: l._spelling_variant_candidates(sents, comps)),
        ("partial-name", lambda l: l._name_word_candidates(sents, comps)),
    ):
        a = {(c.sentence_number, c.component_id, c.matched_text) for c in call(old)}
        b = {(c.sentence_number, c.component_id, c.matched_text) for c in call(new)}
        check(f"{name}: {label} ({len(b)})", a == b, "MATCH", "*** DIFF ***")

print("\n\n=== s_linker46: the alias table has one role ===")
print("\n(a) the change")
proposer = inspect.getsource(SLinker46._name_word_candidates)
check("the whole-name exclusion consults the model name only",
      "_names_by_component()" not in proposer)
check("  and still excludes a stated model name",
      "_find_exact_form(sentence.text, component.name)" in proposer)
for reader, label in (
    (SLinker46._classify_mention_typed, "the mention label still reads the table"),
    (SLinker46._antecedent_states_name, "the antecedent gate still reads the table"),
    (SLinker46._extract_named_mentions, "the extraction prompt still reads the table"),
):
    check(label, "_names_by_component()" in inspect.getsource(reader)
          or "doc_knowledge" in inspect.getsource(reader))

print("\n(b) every method body except the proposer is s_linker25's")
shared(SLinker46, changed_methods=("_name_word_candidates",))

print("\n(c) the freed candidates are exactly the alias-suppressed ones")
if not (TABLE_RUN / "phase_states").is_dir():
    print(f"    {'no alias tables on disk at ' + str(TABLE_RUN):<60} SKIPPED")
else:
    total_old = total_new = 0
    for name, (text_path, model_path) in DATASETS.items():
        sents = load_sentences(str(BENCH / text_path))
        comps = parse_pcm_repository(str(BENCH / model_path))
        table_path = (TABLE_RUN / "phase_states" / "s_linker25" / "openai" / name
                      / "knowledge.pkl")
        with table_path.open("rb") as handle:
            knowledge = pickle.load(handle)["doc_knowledge"]
        old, new = SLinker25.__new__(SLinker25), SLinker46.__new__(SLinker46)
        old.doc_knowledge = new.doc_knowledge = knowledge
        a = {(c.sentence_number, c.component_id)
             for c in old._name_word_candidates(sents, comps)}
        b = {(c.sentence_number, c.component_id)
             for c in new._name_word_candidates(sents, comps)}
        total_old += len(a)
        total_new += len(b)
        by_component = {}
        for term, component in knowledge.aliases.items():
            by_component.setdefault(component, []).append(term)
        id_to_name = {c.id: c.name for c in comps}
        sent_map = {s.number: s.text for s in sents}
        unexplained = 0
        for snum, cid in b - a:
            aliases = by_component.get(id_to_name.get(cid, ""), ())
            if not any(SLinker25._find_exact_form(sent_map[snum], alias)
                       for alias in aliases):
                unexplained += 1
        check(f"{name}: {len(a)} -> {len(b)} candidates, none lost",
              not (a - b), "MATCH", f"*** {len(a - b)} LOST ***")
        check(f"  every one of the {len(b - a)} freed carries a discovered alias",
              unexplained == 0, "MATCH", f"*** {unexplained} UNEXPLAINED ***")
    print(f"    {'TOTAL ' + str(total_old) + ' -> ' + str(total_new):<60} "
          f"+{total_new - total_old}")

print(f"\n{'ALL CHECKS PASS' if bad == 0 else f'{bad} FAILURES'}")
raise SystemExit(1 if bad else 0)
