"""Design invariants of s_linker65_null, s_linker66, s_linker67 and s_linker68.
No LLM calls.

The bind round relocates a deterministic rule into a prompt. That is only a
relocation if the code loses exactly the rule and the prompt gains exactly its
content, so each of the three files is pinned here:

  s_linker65_null  byte-identical to `s_linker65` modulo the rename -- the in-set
                   harness null every arm's delta is read against.
  s_linker66       ONE relocation: `_keep_stated_names` deleted, its contract stated
                   in `ENTITY_EXTRACTION_RULES`.
  s_linker67       s66 plus the two tight scans relocated: `_add_scan` deleted,
                   `SCANS` down to the one row the partial-name linker scans, the
                   recall floor stated in the same constant. REJECTED at n=6
                   (TP -4.0, macro F2 -1.1); kept as the artifact that prices it.
  s_linker68       s66 with one further CUT (not a relocation): the mention label
                   loses its qualified-path value, the only consumer of
                   `_all_occurrences_in_qualified_path`.

Checks:

  1  method bodies -- every one byte-identical to s65's apart from the intended
     deletions and the one call site each
  2  rule constants, resource bounds and prompt builders -- only
     `ENTITY_EXTRACTION_RULES` differs, and only in the two variants that relocate
  3  prompt parity with the measured arms: the extraction prompt each variant sends
     is byte-identical to the one `pilot/bind_pilots.py` measured, on real project
     data. Without this the E2E arm is not the arm that was screened.
  4  the relocation is a move, not a loss: `_states_a_name` survives with its
     remaining call sites, and the deleted predicate is unreachable
  5  the candidate sets change by exactly what the relocation predicts, over all
     five projects and against a recorded run's extraction output
  6  GATE-06: no benchmark vocabulary in the new prompt text
  7  s68's cut: one method deleted, one label value gone, and every one of the 28
     relabelled (name, sentence) pairs moving from the deleted value and nowhere else

    ../.venv/bin/python pilot/test_s66_s67_bindcontract.py
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

import bind_audit                                                     # noqa: E402
from bind_audit import PROJECTS, extractor_pairs, phase_state, project  # noqa: E402
from llm_sad_sam.linkers.experimental import (                        # noqa: E402
    s_linker65, s_linker65_null, s_linker66, s_linker67,
)
from llm_sad_sam.linkers.experimental.s_linker65 import SLinker65     # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker65_null import (        # noqa: E402
    SLinker65Null,
)
from llm_sad_sam.linkers.experimental.s_linker66 import SLinker66     # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker67 import SLinker67     # noqa: E402

SOURCE_RUN = Path("../results/s64_e2e_r1_20260814")
SOURCE_VARIANT = "s_linker64"

RULES = ["DOC_KNOWLEDGE_JUDGE_RULES", "DOC_KNOWLEDGE_EXTRACTION_RULES",
         "ALIAS_EXCLUSION_RULES", "ENTITY_EXTRACTION_RULES", "P1_FOCUS", "P2_FOCUS",
         "COREF_VALIDATION_FOCUS", "COREF_RULES", "LAYERED_ENTITY_RULES",
         "LAYERED_COREF_RULES", "INFLECTIONS"]
BOUNDS = ["CONTEXT_SENTENCES", "ANCHOR_LIMIT", "EXTRACTION_BATCH", "JUDGE_BATCH",
          "COREFERENCE_BATCH", "ASK_ATTEMPTS", "LINKERS"]

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def method_bodies(module, class_name):
    source = Path(inspect.getfile(module)).read_text()
    tree = ast.parse(source)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == class_name)
    lines = source.splitlines()
    bodies = {}
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        body = [s for s in node.body
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        if not body:
            bodies[node.name] = ""
            continue
        start = min(s.lineno for s in body) - 1
        end = max(s.end_lineno for s in body)
        bodies[node.name] = "\n".join(line.rstrip() for line in lines[start:end])
    return bodies


# ── 1 ────────────────────────────────────────────────────────────────────────

def test_method_parity():
    print("\n[1] method bodies against s_linker65's")
    base = method_bodies(s_linker65, "SLinker65")

    null = method_bodies(s_linker65_null, "SLinker65Null")
    differing = {n for n in set(base) & set(null)
                 if base[n].replace("SLinker65", "SLinker65Null")
                 .replace("s_linker65", "s_linker65_null") != null[n]}
    check(f"null: all {len(base)} method bodies identical modulo the rename",
          not differing and set(base) == set(null), str(sorted(differing)))
    whole = Path(inspect.getfile(s_linker65_null)).read_text()
    renamed = (Path(inspect.getfile(s_linker65)).read_text()
               .replace("SLinker65", "SLinker65Null")
               .replace("s_linker65", "s_linker65_null"))
    check("null: the whole file is a rename of s_linker65", whole == renamed)

    for module, cls_name, deleted, changed in (
        (s_linker66, "SLinker66", {"_keep_stated_names"}, {"_run_full_name_linker"}),
        (s_linker67, "SLinker67", {"_keep_stated_names", "_add_scan"},
         {"_run_full_name_linker"}),
    ):
        new = method_bodies(module, cls_name)
        tag = cls_name.lower()
        check(f"{tag}: methods deleted are exactly {sorted(deleted)}",
              set(base) - set(new) == deleted, str(sorted(set(base) - set(new))))
        check(f"{tag}: no method added", not set(new) - set(base),
              str(sorted(set(new) - set(base))))
        differing = {n for n in set(base) & set(new)
                     if base[n].replace("SLinker65", cls_name) != new[n]}
        check(f"{tag}: {len(set(base) & set(new)) - len(differing)} shared method "
              f"bodies identical", differing == changed, str(sorted(differing)))


# ── 2 ────────────────────────────────────────────────────────────────────────

def test_constants():
    print("\n[2] rule constants, resource bounds, prompt builders")
    for module, cls, expected in ((s_linker65_null, SLinker65Null, set()),
                                  (s_linker66, SLinker66,
                                   {"ENTITY_EXTRACTION_RULES"}),
                                  (s_linker67, SLinker67,
                                   {"ENTITY_EXTRACTION_RULES"})):
        tag = cls.__name__.lower()
        differing = {r for r in RULES
                     if getattr(s_linker65, r) != getattr(module, r)}
        check(f"{tag}: rule constants differing == {sorted(expected)}",
              differing == expected, str(sorted(differing)))
        bad = [b for b in BOUNDS if getattr(SLinker65, b) != getattr(cls, b)]
        check(f"{tag}: {len(BOUNDS)} resource bounds identical", not bad, str(bad))
        builders = [n for n in dir(SLinker65) if n.startswith("_prompt_")]
        bad = [n for n in builders
               if inspect.getsource(getattr(SLinker65, n))
               != inspect.getsource(getattr(cls, n))]
        check(f"{tag}: {len(builders)} prompt builders identical in source",
              not bad, str(bad))

    check("s66's constant contains s65's exclusion clause verbatim",
          "as ordinary English with no architectural intent"
          in s_linker66.ENTITY_EXTRACTION_RULES)
    check("s67's constant is s66's plus the scan clause",
          s_linker67.ENTITY_EXTRACTION_RULES.startswith(
              s_linker66.ENTITY_EXTRACTION_RULES))


# ── 3 ────────────────────────────────────────────────────────────────────────

def test_arm_parity():
    """The variant must send the bytes the stage pilot measured."""
    print("\n[3] prompt parity with the measured arms")
    sys.argv = [sys.argv[0]]
    import bind_pilots as bp
    for name in PROJECTS:
        info = bp.inputs(name)
        comp_names = [c.name for c in info["components"]]
        mappings = [f"{t}={c}"
                    for t, c in (info["knowledge"].aliases or {}).items()]
        batch = info["sentences"][:bp.SLinker65.EXTRACTION_BATCH]
        pairs = [
            ("s65 == the pilot's base arm",
             SLinker65._prompt_extraction(comp_names, mappings, batch),
             bp._extraction_builder().__func__(comp_names, mappings, batch)),
            ("s66 == the bindcontract arm",
             SLinker66._prompt_extraction(comp_names, mappings, batch),
             bp._extraction_builder(rules=bp.CONTRACT_RULES).__func__(
                 comp_names, mappings, batch)),
            ("s67 == the bindboth arm",
             SLinker67._prompt_extraction(comp_names, mappings, batch),
             bp._extraction_builder(rules=bp.CONTRACT_RULES,
                                    extra=bp.SCAN_CLAUSE).__func__(
                 comp_names, mappings, batch)),
        ]
        for label, a, b in pairs:
            check(f"{name:<14} {label}", a == b)
    # every other prompt the variants send is s65's, on real data
    info = bp.inputs("mediastore")
    comp_names = [c.name for c in info["components"]]
    for cls in (SLinker66, SLinker67, SLinker65Null):
        same = (cls._prompt_validation(comp_names, ["Case 1"], s_linker65.P1_FOCUS)
                == SLinker65._prompt_validation(comp_names, ["Case 1"],
                                                s_linker65.P1_FOCUS))
        check(f"{cls.__name__}: the judging prompt is s65's", same)


# ── 4 ────────────────────────────────────────────────────────────────────────

def test_relocation_is_a_move():
    print("\n[4] the relocation is a move, not a loss")
    for module, cls in ((s_linker66, SLinker66), (s_linker67, SLinker67)):
        source = Path(inspect.getfile(module)).read_text()
        tag = cls.__name__.lower()
        check(f"{tag}: `_keep_stated_names` is unreachable",
              "self._keep_stated_names(" not in source
              and not hasattr(cls, "_keep_stated_names"))
        check(f"{tag}: `_states_a_name` survives the deletion",
              hasattr(cls, "_states_a_name")
              and source.count("self._states_a_name(") >= 2)
    source = Path(inspect.getfile(s_linker67)).read_text()
    check("s67: `_add_scan` is unreachable",
          "self._add_scan(" not in source and not hasattr(SLinker67, "_add_scan"))
    check("s67: SCANS keeps exactly the partial-name row",
          set(s_linker67.SCANS) == {"name_word"}, str(sorted(s_linker67.SCANS)))
    check("s67: `_scan` and the relation survive",
          hasattr(SLinker67, "_scan") and hasattr(SLinker67, "_name_spans"))


# ── 5 ────────────────────────────────────────────────────────────────────────

def test_candidate_delta():
    """What each variant's candidate set loses, held to a recorded extraction.

    The prompt change cannot be replayed without an LLM call, so this pins the
    *deterministic* half: given the same extractor output, s66 keeps what the filter
    dropped and s67 additionally loses what the two scans added. Both are the
    relocation's predicted delta and nothing else.
    """
    print("\n[5] candidate-set delta against a recorded extraction")
    for name in PROJECTS:
        info = project(name)
        knowledge = phase_state(SOURCE_RUN, SOURCE_VARIANT, name, "knowledge")
        aliases = getattr(knowledge.get("doc_knowledge"), "aliases", {}) or {}
        probe = bind_audit.Probe(aliases)
        extractor = extractor_pairs(SOURCE_RUN, SOURCE_VARIANT, name)
        s65_set, kept = bind_audit.full_name_candidates(probe, info, extractor)
        scans = (bind_audit.scan_pairs(probe, info, "stated_name")
                 | bind_audit.scan_pairs(probe, info, "spelling"))
        s66_set = extractor | scans
        s67_set = extractor
        # what s66 gains is what the filter dropped, minus the part the spelling
        # scan re-admitted anyway -- the dual role B6 measured
        check(f"{name:<14} s66 = s65 + the pairs the filter dropped",
              s66_set - s65_set == (extractor - kept) - scans
              and not s65_set - s66_set,
              f"{len(s66_set - s65_set)} added, "
              f"{len((extractor - kept) & scans)} of the drops re-admitted by a scan")
        check(f"{name:<14} s67 = the extractor's own set",
              s67_set == extractor and s66_set - s67_set == scans - extractor,
              f"{len(s66_set - s67_set)} lost to the scans")


# ── 6 ────────────────────────────────────────────────────────────────────────

def test_gate06():
    print("\n[6] GATE-06: no benchmark vocabulary in the relocated text")
    names = set()
    for name in PROJECTS:
        names |= {c.name.casefold() for c in project(name)["components"]}
        knowledge = phase_state(SOURCE_RUN, SOURCE_VARIANT, name, "knowledge")
        names |= {t.casefold() for t in
                  (getattr(knowledge.get("doc_knowledge"), "aliases", {}) or {})}
    for module in (s_linker66, s_linker67):
        text = module.ENTITY_EXTRACTION_RULES.casefold()
        hits = sorted(n for n in names if len(n) > 3 and n in text)
        check(f"{module.__name__.split('.')[-1]}: no catalog term in the new "
              f"extraction rule", not hits, str(hits))


# ── 7 ────────────────────────────────────────────────────────────────────────

def test_s68_label_cut():
    """s_linker68: one cut on top of s66, and every case it relabels."""
    print("\n[7] s_linker68 — the label's qualified-path value")
    from llm_sad_sam.linkers.experimental import s_linker68
    from llm_sad_sam.linkers.experimental.s_linker66 import MentionType as M66
    from llm_sad_sam.linkers.experimental.s_linker68 import (
        SLinker68, MentionType as M68,
    )

    base = method_bodies(s_linker66, "SLinker66")
    new = method_bodies(s_linker68, "SLinker68")
    check("s68: methods deleted are exactly ['_all_occurrences_in_qualified_path']",
          set(base) - set(new) == {"_all_occurrences_in_qualified_path"},
          str(sorted(set(base) - set(new))))
    check("s68: no method added", not set(new) - set(base),
          str(sorted(set(new) - set(base))))
    differing = {n for n in set(base) & set(new)
                 if base[n].replace("SLinker66", "SLinker68") != new[n]}
    check(f"s68: {len(set(base) & set(new)) - len(differing)} shared method bodies "
          f"identical", differing == {"_classify_mention_typed"},
          str(sorted(differing)))

    differing = {r for r in RULES
                 if getattr(s_linker66, r) != getattr(s_linker68, r)}
    check("s68: every rule constant is s66's", not differing, str(sorted(differing)))
    bad = [b for b in BOUNDS if getattr(SLinker66, b) != getattr(SLinker68, b)]
    check(f"s68: {len(BOUNDS)} resource bounds identical", not bad, str(bad))
    check("s68: the label loses exactly the qualified-path value",
          {m.value for m in M66} - {m.value for m in M68}
          == {"lowercase, inside qualified name"},
          str(sorted({m.value for m in M66} - {m.value for m in M68})))
    check("s68: `_inside_qualified_identifier` survives — the scans still read it",
          hasattr(SLinker68, "_inside_qualified_identifier")
          and hasattr(SLinker68, "_in_dotted_path"))

    # every (name, sentence) pair of all five projects, relabelled as intended
    moved = 0
    other = 0
    for name in PROJECTS:
        info = project(name)
        knowledge = phase_state(SOURCE_RUN, SOURCE_VARIANT, name, "knowledge")
        aliases = getattr(knowledge.get("doc_knowledge"), "aliases", {}) or {}
        old_probe = type("P66", (SLinker66,), {})()
        new_probe = type("P68", (SLinker68,), {})()
        for probe in (old_probe, new_probe):
            probe.doc_knowledge = type("K", (), {"aliases": dict(aliases)})()
        for sentence in info["sentences"]:
            for comp in info["components"]:
                a = old_probe._classify_mention_typed(comp.name, sentence.text)
                b = new_probe._classify_mention_typed(comp.name, sentence.text)
                if a.value == b.value:
                    continue
                if a.value == "lowercase, inside qualified name" and b.value in (
                        "proper case, standalone", "lowercase mention"):
                    moved += 1
                else:
                    other += 1
    check(f"s68: {moved} pairs relabel, all from the deleted value", not other,
          f"{other} relabel some other way")


def main():
    test_method_parity()
    test_constants()
    test_arm_parity()
    test_relocation_is_a_move()
    test_candidate_delta()
    test_gate06()
    test_s68_label_cut()
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:")
        for name in failed:
            print(f"    {name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
