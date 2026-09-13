"""Does `s_linker120` do what the ancestor does, everywhere it is not meant to differ?

The branch's policy is one self-contained file per reported variant: the paper's
supplement is the file, so a reported variant carries its whole workflow and no linker
base class. `s_linker120` used to subclass `SLinker110`, which made the claim "only the
name judging changes" true by construction and unreadable in the file.

**The claim used to be checked by source identity and is now checked by behaviour.**
Holding every shared method byte-identical made the file a diff against its ancestor
rather than a file in its own right: it kept 1:1 wrappers, a five-method proposer chain
and a four-method label chain, none of which a reader of the *approach* needs, and two
of which were the same function twice (`_writes_name` was `_find_exact_form` behind an
always-false flag). Those are inlined and declared in `INLINED`; what replaces the byte
comparison is T6, which runs the ancestor's own scans beside this file's one and
compares every candidate — pair, component, surface and source label. That is a
stronger statement than byte identity, because byte identity never said what the bytes
did. No calls:

  T1  structure      the MRO is `(SLinker120, object)`, and no linker module is
                     imported at module scope (the trail file is imported inside the
                     one method an experiment reaches).
  T2  the copy       every method the two classes share is **byte-identical source**,
                     and the only ones that are not, plus the only ones the union
                     drops, are declared: replaced by the union, rewritten, or
                     inlined into the one caller that had them.
  T3  the constants  every rule constant is byte-identical to the ancestor's.
  T4  the prompts    the knowledge prompts, the resolver prompt and the coreference
                     judging prompt render byte-identically to the ancestor's over all
                     five projects — the last one against `_prompt_validation(...,
                     strict=True)`, the call this variant specialized away.
  T5  the stream     both scans propose exactly what the ancestor's scans propose, and
                     the union's merged stream is exactly their union, per project,
                     under an empty alias table and a recorded one.
  T6  the proposer   the collapsed `_name_candidates` against `s_linker110`'s own
                     `_extract_named_mentions` + `_scan`, candidate for candidate,
                     including the matched surface a rewritten span loop would move.

    ../.venv/bin/python pilot/test_s120_standalone.py
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import PROJECTS                                     # noqa: E402
from simmerge_audit import arm_full, arm_partial, load_project        # noqa: E402
from llm_sad_sam.core.data_types_v2 import DocumentKnowledge          # noqa: E402
from llm_sad_sam.core.document_loader_v2 import build_sent_map        # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker110 as HEAD      # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker120 as UNION     # noqa: E402

#: The methods the union replaces, and the machinery only they reached. Anything else
#: missing from the standalone file is a copy that lost a block.
REPLACED = {
    "_prompt_validation",          # the two-rubric prompt builder: one rubric now
    "_run_full_name_linker",       # both name stages are `_run_name_linker`
    "_run_partial_name_linker",
    "_judge_partial_names",
    "_classify_denotations",       # its contract survives in `DENOTATION_DEMAND`
    "_build_evidence_bundle",      # the bundle is `_union_evidence`
    "_format_evidence",
    "_anchor_union",
    "_validate_with_evidence",
}

#: Methods the ancestor declares that this file does not, because their body now sits
#: in the one caller that had them — a standalone file is read top to bottom, and a
#: method whose whole body is `return some_infra_function(...)`, or which exists only
#: to be called once, costs a reader a jump and buys nothing. Behaviour is unchanged
#: and T5 is what says so: the collapsed proposer is compared candidate for candidate
#: against the ancestor's own scans, which is a stronger claim than the byte identity
#: it replaces.
INLINED = {
    # 1:1 delegations to `linker_infra`, now called at the site
    "_iter_batches": "linker_infra.iter_batches",
    "_link_view": "linker_infra.link_view",
    "_decision_view": "linker_infra.decision_view",
    "_linker_feedback": "linker_infra.linker_feedback",
    "_compute_phase_metrics": "linker_infra.phase_metrics",
    "_backend_tag": "linker_infra.backend_tag",
    "_checkpoint_dir": "linker_infra.checkpoint_dir",
    # the relation, reached directly instead of through a one-line indirection
    "_named_spans": "_name_spans(..., NameForm.ANY_CASE)",
    # `_writes_name` was `_find_exact_form` with an always-false guard in front of it
    "_writes_name": "_find_exact_form (the same function once SKIP_QUALIFIED went)",
    # the proposer chain, now four labelled blocks of `_name_candidates`
    "_extract_named_mentions": "_name_candidates",
    "_scan_all": "_name_candidates",
    "_scan": "_name_candidates",
    "_covering_names": "_name_candidates",
    "_only_inside_another_name": "_name_candidates",
    # pure hops: one caller, and a body that belongs in it
    "_run_linker": "link() — two entries dispatched by name is a dict, not a method",
    "_named_before": "_prompt_coref",
    # the label chain, now one `_mention_label`
    "_classify_mention_typed": "_mention_label",
    "_all_occurrences_in_qualified_path": "_mention_label",
    "_in_dotted_path": "_mention_label",
    "_retained_mention_label": "_mention_label",
}

#: The methods that differ in source, and why each had to. (`_run_linker` is not here:
#: two entries dispatched by name is a dict in `link()`, not a method.)
CHANGED = {
    "_run_validation_pass": "no `strict` argument: one rubric reaches it",
    "_validate_coref_links": "the same call, without the `strict` argument",
    # the rest differ only because an `INLINED` wrapper was removed from under them:
    # the body is the ancestor's with `self._x(...)` written as the infra call it was.
    "link": "calls `linker_feedback` and `phase_metrics` directly",
    "_save_phase": "calls `checkpoint_dir` directly",
    "_save_log": "calls `backend_tag` directly",
    "_resolve_references": "calls `iter_batches` and the views directly",
    "_run_coreference_linker": "calls the views directly",
    # absorbed `_named_before`; T4 is what holds it — the prompt it RENDERS is still
    # the ancestor's byte for byte on all five projects.
    "_prompt_coref": "the antecedent list is built in the loop that prints it",
}

#: Every authored rule constant. A standalone file that drifts from the ancestor on
#: any of these is no longer 'the head, with one stage changed'.
CONSTANTS = (
    "DOC_KNOWLEDGE_JUDGE_RULES", "DOC_KNOWLEDGE_EXTRACTION_RULES",
    "ALIAS_EXCLUSION_RULES", "COREF_VALIDATION_FOCUS", "COREF_RULES",
    "LAYERED_ENTITY_RULES", "LAYERED_COREF_RULES", "QUALIFIED_CLAUSE",
    "STRICTER_CLAUSE", "WORD_PATTERN", "LEMMA_READINGS",
)

CHECKS = FAILS = 0


def check(condition, what):
    global CHECKS, FAILS
    CHECKS += 1
    if not condition:
        FAILS += 1
        print(f"  FAIL {what}")
    return condition


def methods(cls):
    return {name for name, value in vars(cls).items()
            if not name.startswith("__")
            and (callable(getattr(value, "__func__", value))
                 or isinstance(value, (staticmethod, classmethod)))}


def main() -> int:
    print("T1 — structure")
    check(UNION.SLinker120.__mro__ == (UNION.SLinker120, object),
          "the MRO is (SLinker120, object)")
    tree = ast.parse(Path(UNION.__file__).read_text())
    imported = [
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.col_offset == 0 and node.module
    ]
    shared = ("linker_infra", "helper_v3")   # plumbing, not a linker's approach
    siblings = [m for m in imported
                if ".linkers.experimental." in m
                and not m.endswith(shared)]
    check(not siblings, f"no sibling linker imported at module scope ({siblings})")
    print(f"    MRO {[c.__name__ for c in UNION.SLinker120.__mro__]}, "
          f"{len(imported)} module-scope imports, {len(siblings)} of them linkers")

    print("\nT2 — the copy, method by method")
    head_methods, union_methods = methods(HEAD.SLinker110), methods(UNION.SLinker120)
    identical = []
    for name in sorted(head_methods & union_methods):
        left = inspect.getsource(getattr(HEAD.SLinker110, name))
        right = inspect.getsource(getattr(UNION.SLinker120, name))
        if left == right:
            identical.append(name)
        else:
            check(name in CHANGED, f"{name} differs from the ancestor and is declared")
    missing = set(head_methods) - set(union_methods)
    check(missing == REPLACED | set(INLINED),
          f"every method the ancestor has and this file does not is declared, as "
          f"replaced by the union or inlined into its caller "
          f"({sorted(missing - REPLACED - set(INLINED))} unexpected, "
          f"{sorted(REPLACED | set(INLINED) - missing)} declared but present)")
    for name, where in INLINED.items():
        check(name not in union_methods, f"{name} is inlined into {where}")
    for name in CHANGED:
        check(name in union_methods, f"{name} is present and rewritten")
    print(f"    {len(identical)} methods byte-identical to `s_linker110`, "
          f"{len(CHANGED)} rewritten ({', '.join(sorted(CHANGED))}), "
          f"{len(REPLACED)} replaced by the union")

    print("\nT3 — the rule constants")
    for name in CONSTANTS:
        check(getattr(UNION, name) == getattr(HEAD, name),
              f"{name} is byte-identical to the ancestor's")
    print(f"    {len(CONSTANTS)} constants byte-identical")

    print("\nT4 — the prompts, over all five projects")
    for project in sorted(PROJECTS):
        data = load_project(project, use_alias=True)
        components, sentences = data["components"], data["sentences"]
        sent_map = build_sent_map(sentences)
        names = [c.name for c in components]
        lines = [s.text for s in sentences]
        check(UNION.SLinker120._prompt_doc_knowledge_extract(names, lines)
              == HEAD.SLinker110._prompt_doc_knowledge_extract(names, lines),
              f"{project}: the alias extraction prompt")
        proposals = [{"term": n.split()[0], "component": n} for n in names[:3]]
        check(UNION.SLinker120._prompt_doc_knowledge_judge(names, proposals)
              == HEAD.SLinker110._prompt_doc_knowledge_judge(names, proposals),
              f"{project}: the alias judging prompt")

        union = UNION.SLinker120.__new__(UNION.SLinker120)
        head = HEAD.SLinker110.__new__(HEAD.SLinker110)
        knowledge = DocumentKnowledge()
        knowledge.aliases = dict(data["aliases"])
        union.doc_knowledge = head.doc_knowledge = knowledge
        table = [{"sentence": s.number, "text": s.text} for s in sentences[:20]]
        targets = [{"case": i, "target": s.number, "text": s.text}
                   for i, s in enumerate(sentences[:6], 1)]
        check(union._prompt_coref(names, table, targets)
              == head._prompt_coref(names, table, targets),
              f"{project}: the resolver prompt, shortlist included")

        cases = [f'Case {i}: pronoun/role-ref -> {name}\n  "{sentences[0].text}"'
                 for i, name in enumerate(names[:4], 1)]
        for focus in (HEAD.COREF_VALIDATION_FOCUS, ""):
            check(UNION.SLinker120._prompt_coref_validation(names, cases, focus)
                  == HEAD.SLinker110._prompt_validation(names, cases, focus, strict=True),
                  f"{project}: the coreference judging prompt (focus={bool(focus)})")
    print("    4 prompt families identical on every project, both focus settings")

    print("\nT5 — the stream")
    for use_alias in (False, True):
        for project in sorted(PROJECTS):
            data = load_project(project, use_alias)
            full, partial = arm_full(data), arm_partial(data)
            union = UNION.SLinker120.__new__(UNION.SLinker120)
            union.doc_knowledge = data["linker"].doc_knowledge
            sent_map = build_sent_map(data["sentences"])
            merged = union._name_candidates(
                data["sentences"], data["components"], data["id_of"], sent_map)
            pairs = {(c.sentence_number, c.component_id) for c in merged}
            check(pairs == full | partial,
                  f"{project} (aliases={use_alias}): the merged stream is full ∪ partial")
            check(len(pairs) == len(merged), f"{project}: one case per pair")
            labelled = {(c.sentence_number, c.component_id) for c in merged
                        if union._stage_of(c) == "full_name"}
            check(labelled == full,
                  f"{project}: the full-name stage label is the ancestor's stream")
            check({(c.sentence_number, c.component_id) for c in merged
                   if union._stage_of(c) == "partial_name"} == partial - full,
                  f"{project}: the partial-name label is the rest of the stream")
    print(f"    both alias settings x {len(PROJECTS)} projects")

    print("\nT6 — the collapsed proposer against the ancestor's own scans")
    # The claim T2 used to make by byte identity, made by behaviour instead: run
    # `s_linker110`'s two scan methods and this file's one, and compare every
    # candidate — pair, component, source label AND the matched surface, which is
    # what the case prints and what a rewritten span loop is most likely to move.
    for use_alias in (False, True):
        for project in sorted(PROJECTS):
            data = load_project(project, use_alias)
            sent_map = build_sent_map(data["sentences"])
            head = HEAD.SLinker110.__new__(HEAD.SLinker110)
            head.doc_knowledge = data["linker"].doc_knowledge
            union = UNION.SLinker120.__new__(UNION.SLinker120)
            union.doc_knowledge = data["linker"].doc_knowledge

            ancestor = dict(head._extract_named_mentions(
                data["sentences"], data["components"], data["id_of"], sent_map))
            for candidate in head._scan(data["sentences"], data["components"]):
                ancestor.setdefault(
                    (candidate.sentence_number, candidate.component_id), candidate)
            mine = {(c.sentence_number, c.component_id): c for c in
                    union._name_candidates(data["sentences"], data["components"],
                                           data["id_of"], sent_map)}
            check(set(ancestor) == set(mine),
                  f"{project} (aliases={use_alias}): the same pairs as the ancestor")
            shape = {key: (c.component_name, c.matched_text, c.source)
                     for key, c in ancestor.items()}
            check(shape == {key: (c.component_name, c.matched_text, c.source)
                            for key, c in mine.items()},
                  f"{project} (aliases={use_alias}): same component, surface and source")
    print(f"    every candidate compared, both alias settings x {len(PROJECTS)} projects")

    print(f"\n{CHECKS - FAILS}/{CHECKS} checks pass")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
