"""Design invariants of s_linker47 and s_linker48. No LLM calls.

Both variants attack mechanism rather than cost, and they attack it from opposite ends:

  s47  removes a mechanism -- the partial-name linker's grounded identity review -- so
       the linker judges once, like the coreference linker.
  s48  removes no mechanism at all. It merges five near-duplicate conditions into two
       named predicates and deletes three conjuncts that never fired in 122 recorded
       cases. Every prompt byte, every call and every stage is s_linker25's.

s48's claim is the stronger one to check, because "no behaviour changed" is falsifiable:
the test renders every prompt builder on real project data and compares byte for byte,
and runs both merged predicates against the expressions they replaced over every
(name, sentence) pair on all five benchmarks.

Run from the approach/ directory:
    ../.venv/bin/python pilot/test_s47_s48_mechanisms.py
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, "src")

assert Path("src").is_dir(), "run from the approach/ directory"

from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental import s_linker25 as L25
from llm_sad_sam.linkers.experimental import s_linker47 as L47
from llm_sad_sam.linkers.experimental import s_linker48 as L48
from llm_sad_sam.linkers.experimental.s_linker25 import SLinker25
from llm_sad_sam.linkers.experimental.s_linker47 import SLinker47
from llm_sad_sam.linkers.experimental.s_linker48 import SLinker48

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
    print(f"    {label:<62} {good if ok else ill}")


class _Aliases:
    """A table with entries, so the alias branches are actually exercised."""
    aliases = {"Reg": "Registry", "BBB": "BigBlueButton", "DB": "Database"}


RUBRICS = ["ENTITY_EXTRACTION_RULES", "DOC_KNOWLEDGE_EXTRACTION_RULES",
           "DOC_KNOWLEDGE_JUDGE_RULES", "ALIAS_EXCLUSION_RULES", "P1_FOCUS",
           "P2_FOCUS", "COREF_VALIDATION_FOCUS", "COREF_RULES",
           "LAYERED_ENTITY_RULES", "LAYERED_COREF_RULES"]
BOUNDS = ["CONTEXT_SENTENCES", "ANCHOR_LIMIT", "EXTRACTION_BATCH", "JUDGE_BATCH",
          "COREFERENCE_BATCH", "ASK_ATTEMPTS", "LINKERS"]


def rubrics_and_bounds(cls, module):
    for const in RUBRICS:
        check(const, getattr(L25, const) == getattr(module, const),
              "IDENTICAL", "*** DIFFERS ***")
    for bound in BOUNDS:
        check(bound, getattr(SLinker25, bound) == getattr(cls, bound),
              "IDENTICAL", "*** DIFFERS ***")


# ── s47 ──────────────────────────────────────────────────────────────────────
print("\n=== s_linker47: one judging step for the partial-name linker ===")
print("\n(a) the mechanism is gone")
for gone in ("_review_identity", "_review_identity_batch"):
    check(f"{gone} removed", not hasattr(SLinker47, gone))
judge = inspect.getsource(SLinker47._judge_partial_names)
check("the judge calls only the denotation step",
      "_classify_denotations" in judge and "_review_identity" not in judge)
check("what denotation passes is approved", '"approved": True' in judge)
check("the target stays withheld: no identity prompt anywhere",
      "phase_25_partial_identity"
      not in Path("src/llm_sad_sam/linkers/experimental/s_linker47.py").read_text())

print("\n(b) the target-blind step is untouched, and so is everything else")
for method in ("_classify_denotations", "_name_word_candidates",
               "_run_partial_name_linker", "_keep_stated_names",
               "_validate_with_evidence", "_run_coreference_linker",
               "_resolve_references", "_antecedent_states_name",
               "_validate_coref_links", "_build_evidence_bundle",
               "_format_evidence", "_classify_mention_typed", "link"):
    check(method, inspect.getsource(getattr(SLinker25, method))
          == inspect.getsource(getattr(SLinker47, method)),
          "IDENTICAL", "*** DIFFERS ***")
rubrics_and_bounds(SLinker47, L47)

# ── s48 ──────────────────────────────────────────────────────────────────────
print("\n\n=== s_linker48: five conditions become two, three dead ones go ===")
print("\n(a) the merged predicates exist and the copies are gone")
check("_states_a_name exists", hasattr(SLinker48, "_states_a_name"))
check("_claim_supported exists", hasattr(SLinker48, "_claim_supported"))
for method, label in (
    ("_keep_stated_names", "the admission filter uses it"),
    ("_name_word_candidates", "the whole-name exclusion uses it"),
    ("_antecedent_states_name", "the antecedent gate uses it"),
):
    source = inspect.getsource(getattr(SLinker48, method))
    check(f"{label}", "_states_a_name(" in source)
    check(f"  and no longer spells the expression out",
          "_names_by_component()" not in source
          and "names_by_component.get" not in source)
for method in ("_classify_denotations", "_review_identity_batch"):
    source = inspect.getsource(getattr(SLinker48, method))
    check(f"{method} uses _claim_supported", "_claim_supported(" in source)
    check(f"  and carries no copy of the substring test",
          "claim.casefold() in" not in source)
identity = inspect.getsource(SLinker48._review_identity_batch)
check("the anchor-listed conjunct is gone from the gate",
      "anchor in allowed_anchors" not in identity)
check("the non-empty-alternative conjunct is gone from the gate",
      "and bool(alternative)" not in identity)
check("but the model is still ASKED for a listed anchor",
      "Use only a listed case anchor." in identity)
check("and still asked to name its strongest alternative",
      '"alternative":"strongest alternative or none"' in identity)
check("the anchors are still built and still shown",
      "allowed_anchors" in identity and "anchors" in identity)
check("the classifier keeps its own two calls, by design",
      inspect.getsource(SLinker48._classify_mention_typed)
      == inspect.getsource(SLinker25._classify_mention_typed),
      "IDENTICAL", "*** DIFFERS ***")

print("\n(b) no mechanism moved: every other method is s_linker25's")
UNCHANGED = [
    "link", "_run_linker", "_unlinked", "_union", "_add_spelling_variants",
    "_spelling_variant_candidates", "_name_signature", "_in_dotted_path",
    "_inside_qualified_identifier", "_all_occurrences_in_qualified_path",
    "_learn_document_knowledge", "_extract_named_mentions", "_run_extraction_pass",
    "_run_full_name_linker", "_build_evidence_bundle", "_format_evidence",
    "_validate_with_evidence", "_prompt_doc_knowledge_extract",
    "_prompt_doc_knowledge_judge", "_prompt_extraction", "_prompt_validation",
    "_run_validation_pass", "_run_partial_name_linker", "_judge_partial_names",
    "_review_identity", "_run_coreference_linker",
    "_validate_coref_links", "_prompt_coref", "_find_exact_form",
    "_names_by_component", "_prev_prefix", "_iter_batches", "_ask",
]
for method in UNCHANGED:
    check(method, inspect.getsource(getattr(SLinker25, method))
          == inspect.getsource(getattr(SLinker48, method)),
          "IDENTICAL", "*** DIFFERS ***")
rubrics_and_bounds(SLinker48, L48)

print("\n(c) the merged predicates agree with the expressions they replaced")
for name, (text_path, model_path) in DATASETS.items():
    sents = load_sentences(str(BENCH / text_path))
    comps = parse_pcm_repository(str(BENCH / model_path))
    old, new = SLinker25.__new__(SLinker25), SLinker48.__new__(SLinker48)
    old.doc_knowledge = new.doc_knowledge = _Aliases()
    names_by_component = old._names_by_component()
    pairs = flips = 0
    for sentence in sents:
        for component in comps:
            pairs += 1
            spelled_out = any(
                SLinker25._find_exact_form(sentence.text, n)
                for n in (component.name,
                          *names_by_component.get(component.name, [])))
            merged = new._states_a_name(sentence.text, component.name)
            flips += spelled_out != merged
    check(f"{name}: _states_a_name over {pairs} pairs", flips == 0,
          "MATCH", f"*** {flips} FLIPS ***")
    for label, call in (
        ("_keep_stated_names", None),
        ("partial-name proposal", lambda l: l._name_word_candidates(sents, comps)),
        ("spelling-variant proposal",
         lambda l: l._spelling_variant_candidates(sents, comps)),
    ):
        if call is None:
            continue
        a = {(c.sentence_number, c.component_id, c.matched_text) for c in call(old)}
        b = {(c.sentence_number, c.component_id, c.matched_text) for c in call(new)}
        check(f"{name}: {label} ({len(b)})", a == b, "MATCH", "*** DIFF ***")
    gate_flips = sum(
        old._antecedent_states_name(c.name, s.text)
        != new._antecedent_states_name(c.name, s.text)
        for s in sents for c in comps)
    check(f"{name}: antecedent gate over {len(sents) * len(comps)} pairs",
          gate_flips == 0, "MATCH", f"*** {gate_flips} FLIPS ***")

print("\n(d) _window selects what all three old spellings selected")
for name, (text_path, model_path) in DATASETS.items():
    sents = load_sentences(str(BENCH / text_path))
    sent_map = build_sent_map(sents)
    new_linker = SLinker48.__new__(SLinker48)
    width = SLinker25.CONTEXT_SENTENCES
    flips = 0
    for target in sents:
        by_filter = {s.number for s in sents
                     if abs(s.number - target.number) <= width}
        by_walk = {i for i in range(max(1, target.number - width),
                                    target.number + width + 1)
                   if sent_map.get(i)}
        merged = {s.number for s in new_linker._window(target.number, sents)}
        flips += (by_filter != merged) or (by_walk != merged)
    check(f"{name}: {len(sents)} targets, filter == walk == _window", flips == 0,
          "MATCH", f"*** {flips} DIVERGENCES ***")
# the marked context strings the coreference prompt receives, both ways
for name, (text_path, model_path) in DATASETS.items():
    sents = load_sentences(str(BENCH / text_path))
    sent_map = build_sent_map(sents)
    new_linker = SLinker48.__new__(SLinker48)
    width = SLinker25.CONTEXT_SENTENCES
    diffs = 0
    for target in sents:
        old_context = []
        lo = max(1, target.number - width)
        for i in range(lo, target.number + width + 1):
            s = sent_map.get(i)
            if s:
                marker = ">>>" if s.number == target.number else "   "
                old_context.append(f"{marker} S{s.number}: {s.text}")
        new_context = [
            f'{">>>" if s.number == target.number else "   "} S{s.number}: {s.text}'
            for s in new_linker._window(target.number, sents)
        ]
        diffs += old_context != new_context
    check(f"{name}: marked coreference context, {len(sents)} targets", diffs == 0,
          "BYTE-IDENTICAL", f"*** {diffs} DIFFER ***")

check("the coreference resolver no longer walks a range",
      "range(max(1," not in inspect.getsource(SLinker48._resolve_references))
check("  and marks the target sentence exactly as before",
      '">>>"' in inspect.getsource(SLinker48._resolve_references))
for method in ("_classify_denotations", "_review_identity_batch",
               "_resolve_references"):
    check(f"{method} uses _window",
          "_window(" in inspect.getsource(getattr(SLinker48, method)))

print("\n(e) the claim check is the same function of the same arguments")
for claim, text, want in (
    ("stores data", "The Registry stores data.", True),
    ("Stores DATA", "The Registry stores data.", True),
    ("", "The Registry stores data.", False),
    ("invents this", "The Registry stores data.", False),
):
    check(f'_claim_supported({claim!r}) -> {want}',
          SLinker48._claim_supported(claim, text) is want)

print("\n(f) every prompt builder renders byte-identically on real data")
for name, (text_path, model_path) in DATASETS.items():
    sents = load_sentences(str(BENCH / text_path))
    comps = parse_pcm_repository(str(BENCH / model_path))
    sent_map = build_sent_map(sents)
    comp_names = [c.name for c in comps]
    mappings = [f"{t}={c}" for t, c in _Aliases.aliases.items()]
    cases = [f'Case 1: "{comp_names[0]}" -> {comp_names[0]}\n  "{sents[0].text}"']
    renders = {
        "extraction": lambda cls: cls._prompt_extraction(
            comp_names, mappings, sents[:3]),
        "alias extract": lambda cls: cls._prompt_doc_knowledge_extract(
            comp_names, [s.text for s in sents[:5]]),
        "alias judge": lambda cls: cls._prompt_doc_knowledge_judge(
            comp_names, mappings),
        "judge p1": lambda cls: cls._prompt_validation(
            comp_names, cases, L25.P1_FOCUS),
        "judge p2": lambda cls: cls._prompt_validation(
            comp_names, cases, L25.P2_FOCUS),
        "coref judge": lambda cls: cls._prompt_validation(
            comp_names, cases, L25.COREF_VALIDATION_FOCUS, strict=True),
        "coref resolve": lambda cls: cls._prompt_coref(
            comp_names,
            [{"sent": sents[1], "context": [s.text for s in sents[:3]]}]),
    }
    for label, render in renders.items():
        check(f"{name}: {label} prompt",
              render(SLinker25) == render(SLinker48),
              "IDENTICAL", "*** DIFFERS ***")

print(f"\n{'ALL CHECKS PASS' if bad == 0 else f'{bad} FAILURES'}")
raise SystemExit(1 if bad else 0)
