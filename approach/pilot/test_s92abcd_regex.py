"""The regex round's invariants: the scan is the audit, and nothing else moved.

`s_linker92a`-`s_linker92d` replace one method of `s_linker92` -- the LLM extraction
pass -- with a scan. Two things have to be true before any of them is worth a call:

  1  **the variants are the audit.** `pilot/regex_extract_audit.py` priced four scans
     off 30 recorded runs; if the modules do not reproduce those exact candidate sets
     the prices belong to nothing. Checked pair by pair on all five projects, over a
     recorded alias table and over the empty one.
  2  **the swap is the only change.** Every other method's source is `s_linker92`'s,
     every module-level constant is `s_linker92`'s, and the proposer makes no LLM call
     at all -- asserted by handing it a client that raises on `query`.

Plus the two standing gates: GATE-06 (no benchmark vocabulary in the new authored
text) and the branch's own relation table, which the no-alias scans must reproduce
(`ANY_CASE` 172 pairs / 133 gold, `ANY_SPELLING` 176 / 137).

No LLM calls.

    ../.venv/bin/python pilot/test_s92abcd_regex.py
"""
from __future__ import annotations

import inspect
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

import regex_extract_audit as AUDIT                                   # noqa: E402
from design_audit import BENCH, PROJECTS, load_gold                    # noqa: E402
from llm_sad_sam.core.data_types_v2 import DocumentKnowledge           # noqa: E402
from llm_sad_sam.core.document_loader_v2 import (                      # noqa: E402
    build_sent_map, load_sentences)
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository             # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker92 as HEAD        # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92 import SLinker92      # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92a import SLinker92a    # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92b import SLinker92b    # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92c import SLinker92c    # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92d import SLinker92d    # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92e import SLinker92e    # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92f import SLinker92f    # noqa: E402

#: Each variant with the audit arm whose price it carries.
FAMILY = [
    (SLinker92a, dict(form="any_case", use_aliases=True, skip_dotted=False)),
    (SLinker92b, dict(form="any_case", use_aliases=True, skip_dotted=True)),
    (SLinker92c, dict(form="any_spelling", use_aliases=True, skip_dotted=True)),
    (SLinker92d, dict(form="any_case|any_spelling",
                      use_aliases=True, skip_dotted=True)),
    # The judge templates propose exactly what `s_linker92a` proposes.
    (SLinker92e, dict(form="any_case", use_aliases=True, skip_dotted=False)),
    (SLinker92f, dict(form="any_case", use_aliases=True, skip_dotted=False)),
]

#: The two judge-template variants, which may differ from the head only in the
#: LENIENT branch of the judging prompt.
TEMPLATES = (SLinker92e, SLinker92f)

#: The branch's own name-relation table (`approach/CLAUDE.md`), which the catalog-only
#: scans must reproduce exactly, summed over the five projects.
RELATION_TABLE = {"any_case": (172, 133), "any_spelling": (176, 137)}

#: One recorded run, for a real alias table to scan under.
RECORDED = Path("../results/solo_e2e_terra_r1_20260821/phase_states/s_linker89/openai")

checks = 0
fails: list[str] = []


def check(label, ok, detail=""):
    global checks
    checks += 1
    if not ok:
        fails.append(f"{label}: {detail}")


class _NoCalls:
    """An LLM client that fails the test if the proposer reaches for it."""

    def __getattr__(self, name):
        def explode(*_args, **_kwargs):
            raise AssertionError(f"the scan called the LLM: .{name}()")
        return explode


def build(cls, aliases):
    linker = cls.__new__(cls)                    # no backend, no credential
    linker.doc_knowledge = DocumentKnowledge(aliases=dict(aliases))
    linker.llm = _NoCalls()
    return linker


def main():
    projects = {}
    for name in PROJECTS:
        text, model, _ = PROJECTS[name]
        components = parse_pcm_repository(str(BENCH / model))
        sentences = load_sentences(str(BENCH / text))
        projects[name] = {
            "sentences": sentences,
            "components": components,
            "sent_map": build_sent_map(sentences),
            "name_to_id": {c.name: c.id for c in components},
            "gold": load_gold(name),
        }

    # ── 1. the variants are the audit ────────────────────────────────────────
    for name, project in projects.items():
        with open(RECORDED / name / "knowledge.pkl", "rb") as handle:
            recorded = dict(pickle.load(handle)["doc_knowledge"].aliases)
        for aliases, label in ((recorded, "recorded aliases"), ({}, "no aliases")):
            for cls, options in FAMILY:
                linker = build(cls, aliases)
                produced = set(linker._extract_named_mentions(
                    project["sentences"], project["components"],
                    project["name_to_id"], project["sent_map"]))
                expected = AUDIT.regex_keys(
                    f"{name}/{label}", project["sentences"], project["components"],
                    aliases, **options)
                check(f"{cls.__name__} == audit on {name} ({label})",
                      produced == expected,
                      f"+{len(produced - expected)} -{len(expected - produced)}")

    # ── 2. the surface each candidate carries is a writing of a name ─────────
    for name, project in projects.items():
        for cls, _ in FAMILY:
            linker = build(cls, {})
            for (snum, _cid), candidate in linker._extract_named_mentions(
                    project["sentences"], project["components"],
                    project["name_to_id"], project["sent_map"]).items():
                text = project["sent_map"][snum].text
                check(f"{cls.__name__} matched_text in sentence {name} S{snum}",
                      candidate.matched_text
                      and candidate.matched_text in text,
                      repr(candidate.matched_text))
                check(f"{cls.__name__} source tag {name} S{snum}",
                      candidate.source == "full_name", candidate.source)

    # ── 3. the branch's relation table, reproduced ───────────────────────────
    for form, (pairs, gold) in RELATION_TABLE.items():
        total_pairs = total_gold = 0
        for name, project in projects.items():
            keys = AUDIT.regex_keys(f"table/{name}", project["sentences"],
                                    project["components"], {}, form=form,
                                    use_aliases=False, skip_dotted=False)
            total_pairs += len(keys)
            total_gold += len(keys & project["gold"])
        check(f"relation table {form} pairs", total_pairs == pairs,
              f"{total_pairs} != {pairs}")
        check(f"relation table {form} gold", total_gold == gold,
              f"{total_gold} != {gold}")

    # ── 4. the swap is the only change ───────────────────────────────────────
    # `_VARIANT_NAME` is the checkpoint namespace and must differ; everything else
    # a variant defines is the swap itself.
    moved = {"_extract_named_mentions", "_named_spans", "_writes_name",
             "_VARIANT_NAME", "__init__"}
    for cls in TEMPLATES:
        moved_here = moved | {"_prompt_validation"}
        for attribute in dir(SLinker92):
            if attribute.startswith("__") or attribute in moved_here:
                continue
            head = getattr(SLinker92, attribute, None)
            if inspect.isfunction(head) or inspect.ismethod(head):
                check(f"{cls.__name__}.{attribute} source is the head's",
                      inspect.getsource(head)
                      == inspect.getsource(getattr(cls, attribute)), attribute)

    # ── 4b. the templates change the lenient judge only ──────────────────────
    sample = (["Alpha", "Beta"], ["--- Case 1 ---\nS1: text"], "FOCUS")
    for cls in TEMPLATES:
        check(f"{cls.__name__} strict prompt is the head's, byte for byte",
              cls._prompt_validation(*sample, strict=True)
              == SLinker92._prompt_validation(*sample, strict=True))
        check(f"{cls.__name__} lenient prompt differs",
              cls._prompt_validation(*sample, strict=False)
              != SLinker92._prompt_validation(*sample, strict=False))
        # No rule constant may be restated: every authored clause appears once.
        rendered = cls._prompt_validation(*sample, strict=False)
        for constant in ("LAYERED_ENTITY_RULES", "QUALIFIED_CLAUSE",
                         "STRICTER_CLAUSE"):
            text = getattr(HEAD, constant)
            check(f"{cls.__name__} states {constant} exactly once",
                  rendered.count(text) == 1, str(rendered.count(text)))
    for cls, _ in FAMILY:
        if cls in TEMPLATES:
            continue                      # checked above, with its own moved set
        for attribute in dir(SLinker92):
            if attribute.startswith("__") or attribute in moved:
                continue
            head = getattr(SLinker92, attribute, None)
            new = getattr(cls, attribute, None)
            if not (inspect.isfunction(head) or inspect.ismethod(head)):
                check(f"{cls.__name__}.{attribute} is the head's", head is new
                      or head == new, f"{head!r} != {new!r}")
                continue
            check(f"{cls.__name__}.{attribute} source is the head's",
                  inspect.getsource(head) == inspect.getsource(new), attribute)

    head_constants = {n: v for n, v in vars(HEAD).items()
                      if n.isupper() and isinstance(v, str)}
    for cls, _ in FAMILY:
        module = sys.modules[cls.__module__]
        for constant, value in head_constants.items():
            if hasattr(module, constant):
                check(f"{cls.__name__} {constant} unchanged",
                      getattr(module, constant) == value, constant)

    # ── 5. GATE-06: no benchmark vocabulary in the new authored text ─────────
    catalog = {c.name for project in projects.values()
               for c in project["components"]}
    for cls, _ in FAMILY:
        source = Path(inspect.getfile(cls)).read_text()
        # The prose cites the relation, never a component. A catalog name that is
        # also an ordinary lowercase English word cannot be told from prose, so the
        # gate tests the proper-noun-like names, at word boundaries.
        leaked = sorted(n for n in catalog
                        if len(n) > 4 and any(c.isupper() for c in n)
                        and re.search(rf"(?<!\w){re.escape(n)}(?!\w)", source))
        check(f"{cls.__name__} GATE-06", not leaked, ", ".join(leaked))

    print(f"{checks - len(fails)}/{checks} checks passed")
    for failure in fails:
        print(f"  FAIL {failure}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
