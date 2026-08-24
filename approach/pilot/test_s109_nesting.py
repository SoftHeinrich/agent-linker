"""`s_linker109`'s invariants: the refusal is the audit, and nothing else moved.

The variant removes partial-name candidates whose word occurs only inside another
component's whole name. Three things have to be true before it is worth anything:

  1  **the module is the measurement.** `pilot/consolidation_audit.py` priced the
     refusal off six recorded runs of `s_linker92a`; if `SLinker109._scan` does not
     produce exactly that candidate set the price belongs to nothing. Replayed over
     every recorded alias table on all five projects, and over the empty table.
  2  **the refusal costs no gold, on both models.** Asserted per run, not on the
     mean — a mean of zero can hide a loss and a gain.
  3  **the scan is the only change.** Every other method's source is `s_linker92a`'s
     (hence `s_linker92`'s), every prompt builder renders byte-identically, no module
     constant is redeclared, and the scan makes no LLM call — asserted by handing it
     a client that raises on any attribute.

Plus the composition gate the measurement policy asks for at level 3: none of the
refused pairs is proposed by the coreference linker in any recorded run, so nothing
downstream is starved and no E2E is owed.

No LLM calls.

    ../.venv/bin/python pilot/test_s109_nesting.py
"""
from __future__ import annotations

import inspect
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import BENCH, PROJECTS, load_gold                    # noqa: E402
from llm_sad_sam.core.data_types_v2 import DocumentKnowledge           # noqa: E402
from llm_sad_sam.core.document_loader_v2 import load_sentences         # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository             # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker92 as HEAD        # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker109 as VARIANT    # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92a import SLinker92a    # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker109 import SLinker109    # noqa: E402

RESULTS = Path("../results")

#: What `s_linker109` is allowed to declare. Everything else must be inherited.
OWN_METHODS = {"__init__", "_scan", "_covering_names", "_only_inside_another_name"}

CHECKS = []


def check(condition, label):
    CHECKS.append((bool(condition), label))
    if not condition:
        print(f"  FAIL  {label}")


class _NoCalls:
    def __getattr__(self, name):
        def explode(*_args, **_kwargs):
            raise AssertionError(f"the scan called the LLM: .{name}()")
        return explode


def build(cls, aliases):
    linker = cls.__new__(cls)                    # no backend, no credential
    linker.doc_knowledge = DocumentKnowledge(aliases=dict(aliases))
    linker.llm = _NoCalls()
    return linker


def recorded_runs():
    for base in sorted(RESULTS.glob("*/phase_states/s_linker92a/openai")):
        if all((base / p / "knowledge.pkl").exists() for p in PROJECTS):
            yield base


def load(base, project, stage):
    with open(base / project / f"{stage}.pkl", "rb") as handle:
        return pickle.load(handle)


def main():
    projects = {}
    for name, (text, model, _) in PROJECTS.items():
        components = parse_pcm_repository(str(BENCH / model))
        projects[name] = {
            "sentences": load_sentences(str(BENCH / text)),
            "components": components,
            "gold": load_gold(name),
            "id_of": {c.name: c.id for c in components},
        }

    # ── 1. the change is confined to the scan ────────────────────────────────
    declared = {n for n, v in vars(SLinker109).items()
                if callable(v) or isinstance(v, (staticmethod, classmethod))}
    check(declared <= OWN_METHODS,
          f"only {sorted(OWN_METHODS)} are declared (found {sorted(declared)})")
    check(SLinker109.__mro__[1] is SLinker92a, "the base is s_linker92a")

    for attribute in ("_prompt_coref", "_prompt_validation", "_validate_coref_links",
                      "_run_partial_name_linker", "_run_coreference_linker",
                      "_extract_named_mentions", "_name_spans", "_states_a_name"):
        mine = getattr(SLinker109, attribute, None)
        theirs = getattr(SLinker92a, attribute, None)
        check(mine is not None and inspect.unwrap(getattr(mine, "__func__", mine))
              is inspect.unwrap(getattr(theirs, "__func__", theirs)),
              f"{attribute} is inherited, not redeclared")

    module_constants = {n for n, v in vars(VARIANT).items()
                        if n.isupper() and isinstance(v, str)}
    check(not module_constants,
          f"no rule constant is redeclared (found {sorted(module_constants)})")
    for name in ("ENTITY_EXTRACTION_RULES", "COREF_RULES", "QUALIFIED_CLAUSE",
                 "STRICTER_CLAUSE", "LAYERED_ENTITY_RULES", "LAYERED_COREF_RULES"):
        check(getattr(VARIANT, name, None) is None
              or getattr(VARIANT, name) is getattr(HEAD, name),
              f"{name} is the head's object")

    # ── 2. the module reproduces the audit, per run and per model ────────────
    #
    # The claim is about *links*, not candidates: a candidate the denotation judge
    # rejects is not in anyone's link set, so refusing it changes nothing. The audit
    # priced the refusal by replaying it over the recorded verdicts, and that is what
    # is asserted here. What the refusal does to gold *candidates* the judge already
    # declined is reported rather than asserted — it is the arm's standing risk, not
    # a present cost.
    runs = list(recorded_runs())
    check(len(runs) >= 6, f"at least six recorded runs to replay ({len(runs)})")

    totals = {"terra": [], "luna": []}
    gold_candidates_refused, gold_candidates_approved = 0, 0
    for base in runs:
        model = "terra" if "terra" in str(base) else "luna"
        lost_links, lost_gold_links = 0, 0
        for project, data in projects.items():
            aliases = dict(load(base, project, "knowledge")["doc_knowledge"].aliases)
            head = build(SLinker92a, aliases)
            arm = build(SLinker109, aliases)
            before = head._scan(data["sentences"], data["components"])
            after = arm._scan(data["sentences"], data["components"])
            key = lambda c: (c.sentence_number, c.component_id)
            kept = {key(c) for c in after}
            removed = {key(c) for c in before} - kept
            check(kept <= {key(c) for c in before},
                  f"{project}: the arm only removes candidates")

            stage = load(base, project, "linker_partial_name")
            linked = {(l.sentence_number, l.component_id) for l in stage["links"]}
            lost = removed & linked
            lost_links += len(lost)
            lost_gold_links += len(lost & data["gold"])

            approved = {(d["sentence"], d["component_id"])
                        for d in stage["feedback"].get("judge_decisions", [])
                        if d.get("approved")}
            for pair in removed & data["gold"]:
                gold_candidates_refused += 1
                gold_candidates_approved += pair in approved

            # the refusal's stated reason holds for every pair it fires on
            for candidate in before:
                if key(candidate) not in removed:
                    continue
                spans = arm._name_spans(candidate.sentence_text,
                                        candidate.component_name,
                                        HEAD.NameForm.ANY_WORD)
                cover = arm._covering_names(candidate.sentence_text,
                                            candidate.component_name,
                                            data["components"])
                check(spans and cover
                      and all(any(s <= a and b <= e for s, e in cover)
                              for a, b in spans),
                      f"{project} S{candidate.sentence_number}: refusal reason holds")

        check(lost_gold_links == 0,
              f"{base.parts[2]}: no gold LINK is refused ({lost_gold_links})")
        totals[model].append(lost_links)

    for model, counts in totals.items():
        if counts:
            print(f"  {model}: -{sum(counts) / len(counts):.1f} links a run "
                  f"over {len(counts)} runs, 0 of them gold")
    check(gold_candidates_approved == 0,
          f"no gold candidate the judge APPROVED is refused "
          f"({gold_candidates_approved} of {gold_candidates_refused} gold candidates "
          f"refused across {len(runs)} runs)")
    print(f"  risk: {gold_candidates_refused / max(len(runs), 1):.1f} gold candidates "
          f"a run are refused that the judge had already declined")

    # ── 3. the composition gate: nothing downstream re-proposes them ─────────
    #
    # Only a refused pair that was *linked* matters: an unlinked pair was already free
    # for the coreference linker under `_unlinked`, so refusing it earlier starves
    # nothing.
    returned = 0
    for base in runs:
        for project, data in projects.items():
            aliases = dict(load(base, project, "knowledge")["doc_knowledge"].aliases)
            head, arm = build(SLinker92a, aliases), build(SLinker109, aliases)
            key = lambda c: (c.sentence_number, c.component_id)
            removed = ({key(c) for c in head._scan(data["sentences"], data["components"])}
                       - {key(c) for c in arm._scan(data["sentences"], data["components"])})
            stage = load(base, project, "linker_partial_name")
            linked = {(l.sentence_number, l.component_id) for l in stage["links"]}
            coref = load(base, project, "linker_coreference")["feedback"]
            proposed = {(c["sentence"], data["id_of"].get(c["component"]))
                        for c in coref["candidates"]}
            returned += len(removed & linked & proposed)
    check(returned == 0,
          f"no refused LINK is proposed by the coreference linker ({returned})")

    # ── 4. the scan spends nothing ───────────────────────────────────────────
    arm = build(SLinker109, {})
    data = projects["mediastore"]
    arm._scan(data["sentences"], data["components"])           # _NoCalls would raise
    check(True, "the scan makes no LLM call")

    passed = sum(1 for ok, _ in CHECKS if ok)
    print(f"\n{passed}/{len(CHECKS)} checks")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
