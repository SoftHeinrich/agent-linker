"""`s_linker65` is `s_linker64` with the deterministic layer written once — asserted.

This branch's standing rule is that a trace-derived equivalence is a hypothesis and six
paired runs are the test.  That rule is for *behavioural* claims.  `s_linker65` makes no
behavioural claim: it asserts an **identity**, and an identity is checked here, over the
real benchmark data, before any run is paid for.

Six groups of checks, 49 in all:

  1  every other method body is byte-identical to `s_linker64`'s
  2  no prompt constant, rubric or resource bound changed
  3  the relation reproduces `s_linker64`'s four lexical predicates on every
     (name, sentence) pair of all five projects
  4  each of the three candidate generators returns the identical candidate set --
     same pairs, same `matched_text`, same `source`, same mention labels
  5  the full-name linker's composed candidate list is identical, in order
  6  GATE-06: the module introduces no benchmark vocabulary, and its only word list is
     the English inflectional endings

    ../.venv/bin/python pilot/test_s65_one_relation.py
"""
from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import BENCH, PROJECTS                             # noqa: E402
from llm_sad_sam.core.document_loader_v2 import load_sentences       # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository           # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker64, s_linker65  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker64 import SLinker64    # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker65 import (            # noqa: E402
    SLinker65, SCANS, NameForm,
)
from llm_sad_sam.core.data_types_v2 import CandidateLink              # noqa: E402

#: Methods `s_linker65` deliberately replaces or deletes.  Everything else must be
#: byte-identical to `s_linker64`'s.
CHANGED = {
    "_find_exact_form",             # now the relation at ANY_CASE
    "_run_full_name_linker",       # two `_add_scan` calls replace two adders
    "_run_partial_name_linker",    # one `_scan` call replaces `_name_word_candidates`
    "_resolve_references",         # the deleted wrapper's call site
}
#: `_name_signature` keeps its body byte for byte and gains only the paragraph stating
#: that compound splitting does not nest with case folding, so it is not listed above.
DELETED = {
    "_add_spelling_variants", "_add_stated_name_net", "_spelling_variant_candidates",
    "_name_word_candidates", "_is_inflection_of", "_antecedent_states_name",
}
ADDED = {"_add_scan", "_scan", "_name_spans", "_realizes", "_owners"}

WORD = r"[A-Za-z]+[A-Za-z0-9]*|\d+"

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


class Probe64(SLinker64):
    def __init__(self, aliases=None):               # noqa: D107
        self.doc_knowledge = type("K", (), {"aliases": aliases or {}})()


class Probe65(SLinker65):
    def __init__(self, aliases=None):               # noqa: D107
        self.doc_knowledge = type("K", (), {"aliases": aliases or {}})()


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


# ── 1: every other method body byte-identical ────────────────────────────────────

def test_method_parity():
    print("\n[1] method bodies")
    old = method_bodies(s_linker64, "SLinker64")
    new = method_bodies(s_linker65, "SLinker65")

    check("methods deleted are exactly the four proposers plus two wrappers",
          set(old) - set(new) == DELETED, str(sorted(set(old) - set(new))))
    check("methods added are exactly the relation and the scan",
          set(new) - set(old) == ADDED, str(sorted(set(new) - set(old))))

    differing = {
        name for name in set(old) & set(new)
        # `_VARIANT_NAME` and the banner carry the variant's own name
        if old[name].replace("SLinker64", "SLinker65") != new[name]
    }
    unexpected = differing - CHANGED
    check(f"{len(set(old) & set(new)) - len(differing)} shared method bodies identical",
          not unexpected, f"unexpected: {sorted(unexpected)}" if unexpected else "")
    unchanged_but_listed = CHANGED - differing
    check("every method listed as changed really differs",
          not unchanged_but_listed, str(sorted(unchanged_but_listed)))


# ── 2: prompts, rubrics and bounds untouched ─────────────────────────────────────

def test_prompt_parity():
    print("\n[2] prompt constants, rubrics and resource bounds")
    rubrics = [
        "DOC_KNOWLEDGE_JUDGE_RULES", "DOC_KNOWLEDGE_EXTRACTION_RULES",
        "ALIAS_EXCLUSION_RULES", "ENTITY_EXTRACTION_RULES", "P1_FOCUS", "P2_FOCUS",
        "COREF_VALIDATION_FOCUS", "COREF_RULES", "LAYERED_ENTITY_RULES",
        "LAYERED_COREF_RULES", "INFLECTIONS",
    ]
    bad = [r for r in rubrics
           if getattr(s_linker64, r) != getattr(s_linker65, r)]
    check(f"{len(rubrics)} rule constants identical", not bad, str(bad))

    bounds = ["CONTEXT_SENTENCES", "ANCHOR_LIMIT", "EXTRACTION_BATCH", "JUDGE_BATCH",
              "COREFERENCE_BATCH", "ASK_ATTEMPTS", "LINKERS"]
    bad = [b for b in bounds if getattr(SLinker64, b) != getattr(SLinker65, b)]
    check(f"{len(bounds)} resource bounds identical", not bad, str(bad))

    builders = [n for n in dir(SLinker64) if n.startswith("_prompt_")]
    bad = [n for n in builders
           if inspect.getsource(getattr(SLinker64, n))
           != inspect.getsource(getattr(SLinker65, n))]
    check(f"{len(builders)} prompt builders identical", not bad, str(bad))


# ── 3: the relation reproduces the four predicates ───────────────────────────────

def test_relation(projects):
    print("\n[3] the relation against s_linker64's four predicates")
    old, new = Probe64(), Probe65()
    counts = {"pairs": 0}
    bad = {k: 0 for k in ("any_case", "as_spelled", "any_word", "realizes")}
    for data in projects.values():
        for sentence in data["sentences"]:
            text = sentence.text
            for component in data["components"]:
                name = component.name
                counts["pairs"] += 1

                if old._find_exact_form(text, name) != new._find_exact_form(text, name):
                    bad["any_case"] += 1

                want = bool(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text))
                got = bool(new._name_spans(text, name, NameForm.AS_SPELLED))
                if want != got:
                    bad["as_spelled"] += 1

                words = [w.casefold() for w in re.findall(WORD, name)]
                want_spans = {
                    (m.start(), m.end()) for m in re.finditer(WORD, text)
                    if any(old._is_inflection_of(m.group(0).casefold(), w)
                           for w in words)
                }
                if want_spans != set(new._name_spans(text, name, NameForm.ANY_WORD)):
                    bad["any_word"] += 1

                # `_realizes` at ANY_CASE must equal casefold equality, which is the
                # test `_spelling_variant_candidates` used for "already the plain name"
                for surface in (name, name.lower(), name.upper(), f"x {name}"):
                    if new._realizes(surface, name, NameForm.ANY_CASE) != (
                        surface.casefold() == name.casefold()
                    ):
                        bad["realizes"] += 1

    check(f"{counts['pairs']} (name, sentence) pairs checked", True)
    for key, n in bad.items():
        check(f"  {key}: 0 divergences", n == 0, f"{n} divergences" if n else "")


# ── 4: identical candidate sets, generator by generator ──────────────────────────

def view(candidates):
    return sorted(
        (c.sentence_number, c.component_id, c.matched_text, c.source,
         getattr(getattr(c, "mention_type", None), "value", None))
        for c in candidates
    )


def test_generators(projects):
    print("\n[4] candidate sets, generator by generator")
    for pname, data in projects.items():
        sentences, components = data["sentences"], data["components"]
        old, new = Probe64(), Probe65()

        pairs = [
            ("spelling",
             lambda: old._spelling_variant_candidates(sentences, components),
             lambda: new._scan(sentences, components, SCANS["spelling"])),
            ("name_word",
             lambda: old._name_word_candidates(sentences, components),
             lambda: new._scan(sentences, components, SCANS["name_word"])),
            ("stated_name",
             lambda: old._add_stated_name_net([], sentences, components),
             lambda: new._scan(sentences, components, SCANS["stated_name"])),
        ]
        for label, before, after in pairs:
            a, b = view(before()), view(after())
            check(f"{pname:<15} {label:<12} {len(a):>4} candidates identical", a == b,
                  "" if a == b else f"{len(set(a) ^ set(b))} differ")


# ── 5: the composed full-name candidate list ─────────────────────────────────────

def extractor_stand_ins(sentences, components, aliases):
    """Candidate lists standing in for what the LLM extractor may return.

    The two adders are merge steps, so an identity proved only from an empty starting
    list would leave the case that actually occurs -- a non-empty extractor output that
    already holds some of the pairs a scan finds -- unchecked.  Four stand-ins per
    project cover the shapes that matter: nothing, everything a scan will also find
    (so every `setdefault` collides and the existing candidate must win), a disjoint
    set (so none collide), and a half-overlap.
    """
    probe = Probe65(aliases)
    found = probe._add_scan([], sentences, components, "spelling")
    found = probe._add_scan(found, sentences, components, "stated_name")
    # a candidate the scans cannot produce: a distinct `matched_text` and `source`,
    # so a wrong merge order shows up as a changed field rather than a changed pair
    def shadow(candidate):
        return CandidateLink(
            candidate.sentence_number, candidate.sentence_text,
            candidate.component_name, candidate.component_id,
            "<from extractor>", source="full_name",
        )
    disjoint = [
        shadow(c) for c in probe._scan(sentences, components, SCANS["name_word"])
    ]
    return {
        "empty": [],
        "full overlap": [shadow(c) for c in found],
        "disjoint": disjoint,
        "half overlap": [shadow(c) for c in found[::2]] + disjoint[:5],
    }


def test_composition(projects):
    print("\n[5] the full-name linker's composed candidate list")
    for pname, data in projects.items():
        sentences, components = data["sentences"], data["components"]
        aliases = data.get("aliases", {})
        old, new = Probe64(aliases), Probe65(aliases)
        for label, start in extractor_stand_ins(
            sentences, components, aliases
        ).items():
            # `_run_full_name_linker` composes the two adders in this order
            a = old._add_stated_name_net(
                old._add_spelling_variants(list(start), sentences, components),
                sentences, components)
            b = new._add_scan(
                new._add_scan(list(start), sentences, components, "spelling"),
                sentences, components, "stated_name")
            ok = view(a) == view(b)
            check(f"{pname:<15} {label:<13} {len(start):>3} in -> {len(a):>3} out",
                  ok, "" if ok else f"{len(set(view(a)) ^ set(view(b)))} differ")


# ── 6: GATE-06 ───────────────────────────────────────────────────────────────────

def test_gate06(projects):
    print("\n[6] GATE-06: no benchmark vocabulary")
    source = Path(inspect.getfile(s_linker65)).read_text()
    # every quoted string literal in the module, minus the docstrings, against every
    # component name of every benchmark
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            doc = ast.get_docstring(node)
            if doc:
                docstrings.add(doc)
    # Field names of the trace and checkpoint records are not vocabulary: they never
    # reach a prompt and never gate a decision.  They are excluded by construction --
    # a literal in key position of a dict display -- not by being listed, so the check
    # cannot be loosened one collision at a time.
    record_keys = {
        key.value
        for node in ast.walk(tree) if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    literals = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value not in docstrings
    }
    names = {c.name.casefold()
             for data in projects.values() for c in data["components"]}
    # a component name is "introduced" only if a whole literal is one; substring
    # coincidences like "s" in a suffix list are not vocabulary
    hits = sorted({lit for lit in literals - record_keys if lit.casefold() in names})
    collisions = sorted({lit for lit in literals & record_keys
                         if lit.casefold() in names})
    check("no component name appears as a code literal", not hits, str(hits))
    if collisions:
        print(f"        (record field names colliding with a catalog name, "
              f"not vocabulary: {collisions})")
    check("the only word list is INFLECTIONS",
          len([n for n, v in vars(s_linker65).items()
               if isinstance(v, tuple) and v and all(isinstance(x, str) for x in v)
               and n.isupper()]) == 1)


def main():
    projects = {
        name: {
            "sentences": load_sentences(str(BENCH / PROJECTS[name][0])),
            "components": parse_pcm_repository(str(BENCH / PROJECTS[name][1])),
        }
        for name in PROJECTS
    }
    test_method_parity()
    test_prompt_parity()
    test_relation(projects)
    test_generators(projects)
    test_composition(projects)
    test_gate06(projects)

    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:")
        for name in failed:
            print(f"  {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
