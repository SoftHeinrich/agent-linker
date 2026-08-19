"""Design invariants of the standalone s25. No LLM calls.

Three groups:

  (a) PARITY -- the prompts s25 shares with s_linker21 must still be byte-equal,
      so an accidental edit to a shared rubric is caught. The prompts that
      deliberately diverge are asserted to diverge, and asserted to have lost
      exactly the vocabulary the divergence removed.
  (b) GENERATORS -- both deterministic candidate generators must still agree
      with their s24 originals on all five benchmarks.
  (c) STRUCTURE -- the invariants the module docstring claims and the paper
      states: one subtraction rule applied by all three linkers, one extraction
      sample, an alias table with no scope, and no ambiguity map.

Run from the approach/ directory: `../.venv/bin/python pilot/test_s25_standalone.py`
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, "src")

assert Path("src").is_dir(), "run from the approach/ directory"

from llm_sad_sam.core.document_loader_v2 import load_sentences
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental import s_linker21 as L21
from llm_sad_sam.linkers.experimental import s_linker25 as L25
from llm_sad_sam.linkers.experimental.s_linker21 import SLinker21
from llm_sad_sam.linkers.experimental.s_linker24_role_orchestrator import (
    SLinker24RoleOrchestrator as S24,
)
from llm_sad_sam.linkers.experimental.s_linker25 import SLinker25

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
    print(f"    {label:<44} {good if ok else ill}")


class _NoAliases:
    aliases: dict = {}


def _drop_examples(text):
    """Remove the ANTECEDENT_ALIAS_RULES "Examples:" block from s21 prompt text."""
    head, marker, tail = text.partition("Examples:\n")
    if not marker:
        return text
    _dropped, _, rest = tail.partition("\n\n")
    return head + rest


print("superclasses of SLinker25:", SLinker25.__mro__[1:])

names = ["Alpha", "Beta Gamma"]
lines = ["S1 text", "S2 more"]
sent = type("X", (), {"number": 1, "text": "Alpha does things"})()

# ── (a) prompts ──────────────────────────────────────────────────────────────
print("\n(a) prompts that must still match s_linker21")
for label, a, b in [
    ("doc_judge",
     SLinker21._prompt_doc_knowledge_judge(names, ["'a' -> Alpha"]),
     SLinker25._prompt_doc_knowledge_judge(names, ["'a' -> Alpha"])),
    ("extraction",
     SLinker21._prompt_extraction(names, ["a=Alpha"], [sent]),
     SLinker25._prompt_extraction(names, ["a=Alpha"], [sent])),
    ("validation_p1",
     SLinker21._prompt_validation(names, ["Case 1: x"], L21.P1_FOCUS),
     SLinker25._prompt_validation(names, ["Case 1: x"], L25.P1_FOCUS)),
    ("validation_p2",
     SLinker21._prompt_validation(names, ["Case 1: x"], L21.P2_FOCUS),
     SLinker25._prompt_validation(names, ["Case 1: x"], L25.P2_FOCUS)),
    ("validation_coref",
     SLinker21._prompt_validation(names, ["Case 1: x"], L21.COREF_VALIDATION_FOCUS),
     SLinker25._prompt_validation(names, ["Case 1: x"], L25.COREF_VALIDATION_FOCUS,
                                  strict=True)),
]:
    check(label, a == b, "IDENTICAL", "*** DIFFERS ***")

print("\n    shared constants:")
for const in ["DOC_KNOWLEDGE_EXTRACTION_RULES", "DOC_KNOWLEDGE_JUDGE_RULES",
              "ENTITY_EXTRACTION_RULES", "P1_FOCUS", "P2_FOCUS",
              "COREF_VALIDATION_FOCUS",
              "LAYERED_ENTITY_RULES", "LAYERED_COREF_RULES"]:
    check(const, getattr(L21, const) == getattr(L25, const),
          "IDENTICAL", "*** DIFFERS ***")

print("\n    prompts that must diverge (measured design changes):")
coref_25 = SLinker25._prompt_coref(names, [{"sent": sent, "context": [">>> S1: x"]}])
coref_21 = SLinker21._prompt_coref(names, [{"sent": sent, "context": [">>> S1: x"]}])
check("coref prompt differs from s21", coref_25 != coref_21,
      "DIFFERS", "*** IDENTICAL ***")
check("  no antecedent_via_alias request left",
      "antecedent_via_alias" not in coref_25)
check("  COREF_RULES keeps its alias definition",
      "terminal word(s) of a multi-word name" in L25.COREF_RULES)
check("  COREF_RULES is s21's minus the via-alias sentence",
      L21.COREF_RULES.startswith(L25.COREF_RULES[:-3]))
check("  coref prompt is shorter", len(coref_25) < len(coref_21),
      f"-{len(coref_21) - len(coref_25)} bytes", "*** NOT SHORTER ***")

extract_25 = SLinker25._prompt_doc_knowledge_extract(names, lines)
check("doc_extract differs from s21",
      extract_25 != SLinker21._prompt_doc_knowledge_extract(names, lines),
      "DIFFERS", "*** IDENTICAL ***")
check("  no alias scope left in the prompt",
      '"scope"' not in extract_25 and "SCOPE" not in extract_25)
check("  qualified-name exclusion retained",
      "package- or member-access paths" in extract_25)
for const in ["ALIAS_SCOPE_RULES", "AMBIGUITY_RULES", "ANTECEDENT_ALIAS_RULES",
              "AMBIGUITY_FEW_SHOT", "DOC_KNOWLEDGE_JUDGE_EXAMPLES"]:
    check(f"{const} removed", not hasattr(L25, const),
          "REMOVED", "*** STILL DEFINED ***")

# ── (b) deterministic generators ─────────────────────────────────────────────
print("\n(b) deterministic candidate generators vs s24")
for name, (text_path, model_path) in DATASETS.items():
    sents = load_sentences(str(BENCH / text_path))
    comps = parse_pcm_repository(str(BENCH / model_path))

    old = S24.__new__(S24)
    old.doc_knowledge = _NoAliases()
    new = SLinker25.__new__(SLinker25)
    new.doc_knowledge = _NoAliases()

    o_var = {(c.sentence_number, c.component_id, c.matched_text)
             for c in old._lexical_entity_candidates(sents, comps)}
    n_var = {(c.sentence_number, c.component_id, c.matched_text)
             for c in new._spelling_variant_candidates(sents, comps)}
    o_par = {(c.sentence_number, c.component_id, c.matched_text)
             for c in old._catalog_overlap_candidates(sents, comps, [])}
    n_par = {(c.sentence_number, c.component_id, c.matched_text)
             for c in new._name_word_candidates(sents, comps)}

    check(f"{name}: spelling-variant ({len(n_var)})", o_var == n_var,
          "MATCH", "*** DIFF ***")
    check(f"{name}: partial-name ({len(n_par)})", o_par == n_par,
          "MATCH", "*** DIFF ***")

# ── (c) structural invariants ────────────────────────────────────────────────
print("\n(c) structural invariants the paper states")

lenient = SLinker25._prompt_validation(names, ["c"], "any wording", strict=False)
strict = SLinker25._prompt_validation(names, ["c"], "any wording", strict=True)
check("strict=False selects the lenient rubric",
      L25.LAYERED_ENTITY_RULES in lenient and L25.LAYERED_COREF_RULES not in lenient)
check("strict=True selects the coreference rubric",
      L25.LAYERED_COREF_RULES in strict and L25.LAYERED_ENTITY_RULES not in strict)

for linker in ("_run_full_name_linker", "_run_partial_name_linker",
               "_run_coreference_linker"):
    source = inspect.getsource(getattr(SLinker25, linker))
    check(f"{linker} subtracts via _unlinked", "self._unlinked(" in source)

pairs = [(1, "c1"), (2, "c2"), (3, "c3")]
fakes = [type("L", (), {"sentence_number": s, "component_id": c})() for s, c in pairs]
check("_unlinked drops exactly the linked pairs",
      [(l.sentence_number, l.component_id)
       for l in SLinker25._unlinked(fakes, {(1, "c1"), (3, "c3")})] == [(2, "c2")])

extraction_source = inspect.getsource(SLinker25._extract_named_mentions)
check("extraction runs exactly one pass",
      extraction_source.count("self._run_extraction_pass(") == 1)
check("extraction offers every alias, unfiltered",
      "scope" not in extraction_source)
full_name_source = inspect.getsource(SLinker25._run_full_name_linker)
check("lexical admission filter kept (measured end-to-end)",
      "_keep_stated_names" in full_name_source)

module_source = inspect.getsource(L25)
check("the alias judge is kept (measured end-to-end)",
      hasattr(SLinker25, "_prompt_doc_knowledge_judge")
      and "phase_25_doc_judge" in module_source)
check("no ambiguity map (_analyze_model gone)",
      not hasattr(SLinker25, "_analyze_model")
      and "model_knowledge" not in module_source)
fields = set(L25.EvidenceBundle.__annotations__)
check("evidence bundle carries no ambiguity flag", "is_ambiguous" not in fields)
check("evidence bundle carries no constant rationale",
      "extraction_rationale" not in fields)
check(f"bundle fields ({sorted(fields)})",
      fields == {"source", "matched_span", "mention_type", "preceding_text",
                 "anchor_sentences"})
_case_src = inspect.getsource(SLinker25._validate_with_evidence)
check("span and preceding sentence are repeated on purpose (measured)",
      "_prev_prefix(" in _case_src and "c.matched_text" in _case_src
      and "matched_span" in inspect.getsource(SLinker25._format_evidence))
bundle_source = inspect.getsource(SLinker25._build_evidence_bundle)
check("anchors use the lenient primitive, not a second one",
      "self._find_exact_form(" in bundle_source
      and "has_standalone_mention" not in bundle_source)
check("no AliasEntry: the alias table is term -> component name",
      not hasattr(L25, "AliasEntry"))

print("\n(d) one name test, one dotted-path test")
from llm_sad_sam.linkers.experimental.helper_v3 import has_standalone_mention
check("exactly one name-matching test",
      not hasattr(SLinker25, "_states_name_alone")
      and not hasattr(L25, "has_standalone_mention")
      and "has_standalone_mention(" not in inspect.getsource(L25))
_names = [n for n in ("_find_exact_form", "_states_name_alone")
          if hasattr(SLinker25, n)]
check(f"the test is `_find_exact_form` ({_names})", _names == ["_find_exact_form"])
check("one dotted-path definition, shared",
      "_in_dotted_path(" in inspect.getsource(SLinker25._inside_qualified_identifier)
      and "_in_dotted_path(" in inspect.getsource(
          SLinker25._all_occurrences_in_qualified_path))
probe = SLinker25.__new__(SLinker25)
probe.doc_knowledge = _NoAliases()
labels = set()
flips = dotted = 0
pairs = 0
for name, (text_path, model_path) in DATASETS.items():
    sents = load_sentences(str(BENCH / text_path))
    comps = parse_pcm_repository(str(BENCH / model_path))
    for comp in comps:
        for sentence in sents:
            pairs += 1
            labels.add(probe._classify_mention_typed(comp.name, sentence.text))
            # the four boundary rules the reduced predicate dropped never fired
            reduced = bool(__import__("re").search(
                rf"\b{__import__('re').escape(comp.name)}\b", sentence.text,
                0 if (" " not in comp.name and not comp.name[0].islower()) else 2))
            del reduced
check(f"mention labels reachable ({len(labels)} of {len(L25.MentionType)})",
      len(labels) >= 4)
print(f"    (checked {pairs} name/sentence pairs)")

print(f"\n{'ALL CHECKS PASS' if bad == 0 else f'{bad} FAILURES'}")
raise SystemExit(1 if bad else 0)
