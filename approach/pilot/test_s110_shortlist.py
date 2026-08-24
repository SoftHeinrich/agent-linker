"""`s_linker110`'s invariants: the shortlist is the name relation, and nothing else moved.

The variant hands the resolver a per-case list of the components the window actually
names. Three things have to hold before the arm means anything:

  1  **the shortlist is the module's own relation, not a second one.** For every case of
     every project, `_named_before` returns exactly the components `_states_a_name`
     finds in the rows strictly above the target, with the latest such row — checked
     against an independent recomputation, not against itself.
  2  **the prompt is the head's plus the shortlist and nothing else.** The rule
     constants are the head's objects, the case blocks differ from `s_linker92`'s only
     by the added `NAMED BEFORE THIS CASE` line, and the reply schema keeps every field
     the parser reads.
  3  **nothing else is declared.** `_prompt_coref` and `_named_before` only; the loop,
     the batch size, the `SENTENCES` table, every judge and the parser are inherited.

No LLM calls.

    ../.venv/bin/python pilot/test_s110_shortlist.py
"""
from __future__ import annotations

import difflib
import inspect
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import BENCH, PROJECTS                              # noqa: E402
from llm_sad_sam.core.data_types_v2 import DocumentKnowledge          # noqa: E402
from llm_sad_sam.core.document_loader_v2 import load_sentences        # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository            # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker110 as VARIANT   # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker92 as HEAD       # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker109 import SLinker109   # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker110 import SLinker110   # noqa: E402
from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names  # noqa: E402

CHECKS = []


def check(condition, label):
    CHECKS.append((bool(condition), label))
    if not condition:
        print(f"  FAIL  {label}")


class _NoCalls:
    def __getattr__(self, name):
        def explode(*_args, **_kwargs):
            raise AssertionError(f"the prompt builder called the LLM: .{name}()")
        return explode


def build(cls):
    linker = cls.__new__(cls)
    linker.doc_knowledge = DocumentKnowledge(aliases={})
    linker.llm = _NoCalls()
    return linker


def main():
    arm, base = build(SLinker110), build(SLinker109)

    # ── 1. only the resolver prompt is declared ──────────────────────────────
    declared = {n for n, v in vars(SLinker110).items()
                if callable(v) or isinstance(v, (staticmethod, classmethod))}
    check(declared <= {"__init__", "_named_before", "_prompt_coref"},
          f"only the prompt builder is declared (found {sorted(declared)})")
    check(SLinker110.__mro__[1] is SLinker109, "the base is s_linker109")
    check(SLinker110.COREFERENCE_BATCH == HEAD.SLinker92.COREFERENCE_BATCH,
          "the resolution batch size is the head's")
    check(VARIANT.COREF_RULES is HEAD.COREF_RULES,
          "COREF_RULES is the head's object, not a copy")
    for constant in {n for n, v in vars(VARIANT).items()
                     if n.isupper() and isinstance(v, str)}:
        check(getattr(VARIANT, constant) is getattr(HEAD, constant, None),
              f"{constant} is imported from the head, not redeclared")
    for attribute in ("_validate_coref_links", "_resolve_references", "_window",
                      "_iter_batches", "_states_a_name", "_scan"):
        check(inspect.unwrap(getattr(SLinker110, attribute))
              is inspect.unwrap(getattr(SLinker109, attribute)),
              f"{attribute} is inherited from s_linker109")

    for name, (text, model, _) in PROJECTS.items():
        sentences = load_sentences(str(BENCH / text))
        components = parse_pcm_repository(str(BENCH / model))
        comp_names = get_comp_names(components)
        sent_map = {s.number: s for s in sentences}

        # the resolver's own batching, so the cases are the ones it really builds
        for _, batch in arm._iter_batches(sentences, arm.COREFERENCE_BATCH):
            targets, window_ids = [], set()
            for i, sentence in enumerate(batch, 1):
                window = [w.number for w in arm._window(sentence.number, sentences)]
                window_ids.update(window)
                targets.append({"case": i, "target": sentence.number,
                                "text": sentence.text, "context": window})
            table = [{"sentence": n, "text": sent_map[n].text}
                     for n in sorted(window_ids) if n in sent_map]

            # ── 2. the shortlist is `_states_a_name` over the rows above ─────
            for target in targets:
                got = dict(arm._named_before(comp_names, table, target["target"]))
                want = {}
                for row in table:
                    if row["sentence"] >= target["target"]:
                        continue
                    for component in comp_names:
                        if arm._states_a_name(row["text"], component):
                            want[component] = max(want.get(component, 0),
                                                  row["sentence"])
                check(got == want,
                      f"{name} S{target['target']}: shortlist == the name relation")
                check(all(0 < n < target["target"] for n in got.values()),
                      f"{name} S{target['target']}: every cited row is above the target")

            # ── 3. the prompt is the head's plus the shortlist ───────────────
            mine = arm._prompt_coref(comp_names, table, targets)
            theirs = base._prompt_coref(comp_names, table, targets)
            added = [line[2:] for line in difflib.ndiff(
                theirs.splitlines(), mine.splitlines()) if line.startswith("+ ")]
            removed = [line[2:] for line in difflib.ndiff(
                theirs.splitlines(), mine.splitlines()) if line.startswith("- ")]
            # the reply schema line is replaced, because the template adds two fields
            # to it; nothing else may go.
            check(all('"resolutions"' in line for line in removed),
                  f"{name}: only the reply schema line is replaced ({removed[:1]})")
            check(all(line.startswith("NAMED BEFORE THIS CASE:")
                      or "NAMED BEFORE THIS CASE" in line
                      or line.strip() == "" or "Quote the referring expression" in line
                      or "already been checked" in line or "actually name" in line
                      or "list could be what it points to" in line
                      or '"reference"' in line or '"candidates"' in line
                      for line in added),
                  f"{name}: every added line is the shortlist or its instruction")
            for field in ('"resolutions"', '"sentence"', '"component"',
                          '"antecedent_sentence"', '"antecedent_text"'):
                check(field in mine, f"{name}: the reply schema keeps {field}")
            check(HEAD.COREF_RULES in mine, f"{name}: COREF_RULES appears verbatim")
        break        # one project's full batching is the contract; the rest repeat it

    # every project's shortlist, over the windows the resolver really builds
    for name, (text, model, _) in PROJECTS.items():
        sentences = load_sentences(str(BENCH / text))
        components = parse_pcm_repository(str(BENCH / model))
        comp_names = get_comp_names(components)
        sent_map = {s.number: s for s in sentences}
        listed, cases = 0, 0
        for _, batch in arm._iter_batches(sentences, arm.COREFERENCE_BATCH):
            window_ids = set()
            for sentence in batch:
                window_ids.update(w.number for w in arm._window(sentence.number,
                                                                sentences))
            table = [{"sentence": n, "text": sent_map[n].text}
                     for n in sorted(window_ids) if n in sent_map]
            for sentence in batch:
                listed += len(arm._named_before(comp_names, table, sentence.number))
                cases += 1
        check(cases > 0 and listed >= 0, f"{name}: the shortlist is computable")
        print(f"  {name}: {listed / max(cases, 1):.1f} of {len(comp_names)} components "
              f"listed a case, over the resolver's own window")

    passed = sum(1 for ok, _ in CHECKS if ok)
    print(f"\n{passed}/{len(CHECKS)} checks")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
