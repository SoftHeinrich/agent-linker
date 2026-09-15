"""Is `s_linker121` a complete, self-contained workflow? No LLM calls.

The branch's policy is one self-contained file per reported variant: the paper's
supplement is the file, so a reported variant carries its whole workflow and no linker
base class.

The file is not a copy any more — it has its own method set, because a method that
existed only to forward a call to another module, or to name one step of a caller that
had exactly one, is indirection rather than structure. So this does NOT compare the two
classes method by method. It checks the two things that survive a refactor and are what
the claim was ever about: that the file is self-contained and has nothing dead in it,
and that what it SENDS and what it PROPOSES are unchanged.

  T1  structure      the MRO is `(SLinker121, object)`, no linker module is imported
                     at module scope, and every method is reachable — called inside
                     the file, an entry point, or exercised by the pilots. A method
                     that nothing calls is a copy that lost its caller.
  T2  no indirection every method either has more than one caller or does more than
                     hand its arguments to something else. Reported per method, so a
                     thin wrapper growing back is visible in the diff.
  T3  the constants  every rule constant is byte-identical to `s_linker110`'s, which
                     is what makes the rule a quotation rather than a retyping.
  T4  the prompts    the knowledge prompts, the resolver prompt and the coreference
                     judging prompt render byte-identically to `s_linker110`'s over
                     all five projects — the last against `_prompt_validation(...,
                     strict=True)`, the call this variant specialized away.
  T5  the stream     both scans propose exactly what `s_linker110`'s scans propose,
                     and the merged stream is exactly their union, per project, under
                     an empty alias table and a recorded one.

    ../.venv/bin/python pilot/test_s121_standalone.py
"""
from __future__ import annotations

import ast
import collections
import inspect
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import PROJECTS                                     # noqa: E402
from simmerge_audit import (                                          # noqa: E402
    arm_full, arm_partial, arm_partial_all, load_project,
)
from llm_sad_sam.core.data_types_v2 import DocumentKnowledge          # noqa: E402
from llm_sad_sam.core.document_loader_v2 import build_sent_map        # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker110 as HEAD      # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker121 as UNION     # noqa: E402

#: Methods with one caller that earn their name anyway: each is a stage of the
#: pipeline, and folding it into its caller would produce a method too long to read
#: rather than one fewer indirection. Everything NOT listed here must have two callers.
STAGES = {
    "link", "_run_linker", "_run_name_linker", "_run_coreference_linker",
    "_learn_document_knowledge", "_resolve_references", "_validate_coref_links",
    "_run_validation_pass", "_judge_union", "_union_evidence", "_format_union_case",
    "_name_candidates", "_extract_named_mentions", "_scan",
    "_named_before", "_mention_label",
    "_prompt_union", "_prompt_coref", "_prompt_coref_validation",
    "_prompt_doc_knowledge_extract", "_prompt_doc_knowledge_judge",
}

#: Entry points and test surface: called from outside the file, so an in-file caller
#: count of zero is correct for them.
ENTRY = {"link", "__init__", "_name_candidates", "_union_evidence",
         "_format_union_case", "_judge_union", "_run_name_linker", "_stage_of",
         "_window", "_name_spans", "_states_a_name", "_writes_name",
         "_find_exact_form", "_prev_prefix", "_scan", "_extract_named_mentions"}

#: Every authored rule constant. A drift on any of these turns the rule from a
#: quotation of the branch's own text into a retyping of it.
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


def code_of(function) -> str:
    """A method's code with every docstring and comment removed.

    Comments never reach the AST; docstrings are stripped node by node. What is left
    is the executable content, which is the part the claim "only the name judging
    changes" is actually about.
    """
    source = inspect.getsource(function)
    # Not `textwrap.dedent`: these methods hold f-strings whose lines start at
    # column 0, so the common prefix is empty and dedent is a no-op. Strip the
    # method's own indent instead, measured off its first line.
    indent = len(source) - len(source.lstrip(" "))
    source = "\n".join(line[indent:] if line[:indent].isspace() else line
                       for line in source.splitlines())
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                              ast.Module))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def methods(cls):
    return {name for name, value in vars(cls).items()
            if not name.startswith("__")
            and (callable(getattr(value, "__func__", value))
                 or isinstance(value, (staticmethod, classmethod)))}


def main() -> int:
    print("T1 — structure")
    check(UNION.SLinker121.__mro__ == (UNION.SLinker121, object),
          "the MRO is (SLinker121, object)")
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
    print(f"    MRO {[c.__name__ for c in UNION.SLinker121.__mro__]}, "
          f"{len(imported)} module-scope imports, {len(siblings)} of them linkers")

    class_node = next(n for n in ast.parse(Path(UNION.__file__).read_text()).body
                      if isinstance(n, ast.ClassDef) and n.name == "SLinker121")
    defined = {n.name: n for n in class_node.body
               if isinstance(n, ast.FunctionDef)}
    callers = collections.Counter()
    for node in ast.walk(class_node):
        if isinstance(node, ast.Attribute) and node.attr in defined:
            callers[node.attr] += 1
    for name in sorted(defined):
        check(callers[name] or name in ENTRY,
              f"{name} is called somewhere, or is declared an entry point")
    print(f"    {len(defined)} methods, none unreachable "
          f"({len(ENTRY & set(defined))} entry points)")

    print("\nT2 — no method is pure indirection")
    def forwards_only(node):
        """One statement, returning or making a single call. Nothing of its own."""
        body = [n for n in node.body
                if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        if len(body) != 1:
            return False
        stmt = body[0]
        inner = (stmt.value if isinstance(stmt, (ast.Return, ast.Expr)) else None)
        return isinstance(inner, ast.Call)

    thin = sorted(name for name, node in defined.items()
                  if forwards_only(node) and callers[name] < 2)
    check(not thin, f"no single-caller method only forwards its arguments ({thin})")
    reused = sorted(name for name in defined if callers[name] >= 2)
    for name in sorted(defined):
        if callers[name] < 2 and name not in ENTRY:
            check(name in STAGES,
                  f"{name} has one caller and is declared a pipeline stage")
    print(f"    {len(reused)} methods with two or more callers; "
          f"the rest are declared stages or entry points, and none is a forwarder")

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
        check(UNION.SLinker121._prompt_doc_knowledge_extract(names, lines)
              == HEAD.SLinker110._prompt_doc_knowledge_extract(names, lines),
              f"{project}: the alias extraction prompt")
        proposals = [{"term": n.split()[0], "component": n} for n in names[:3]]
        check(UNION.SLinker121._prompt_doc_knowledge_judge(names, proposals)
              == HEAD.SLinker110._prompt_doc_knowledge_judge(names, proposals),
              f"{project}: the alias judging prompt")

        union = UNION.SLinker121.__new__(UNION.SLinker121)
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
            check(UNION.SLinker121._prompt_coref_validation(names, cases, focus)
                  == HEAD.SLinker110._prompt_validation(names, cases, focus, strict=True),
                  f"{project}: the coreference judging prompt (focus={bool(focus)})")
    print("    4 prompt families identical on every project, both focus settings")

    print("\nT5 — the stream")
    for use_alias in (False, True):
        for project in sorted(PROJECTS):
            data = load_project(project, use_alias)
            # The ancestor's one-word scan ends a case when the word is written only
            # inside another component's name; `s_linker121` does not (the refusal is
            # removed, `../results/s121_ablations/`), so the reference here is the
            # UNREFUSED scan and the refused set is pinned below as the difference.
            full = arm_full(data)
            partial = arm_partial_all(data)
            refused_by_ancestor = partial - arm_partial(data)
            union = UNION.SLinker121.__new__(UNION.SLinker121)
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
            check(refused_by_ancestor <= pairs,
                  f"{project}: the pairs the ancestor's nesting refusal dropped are "
                  f"cases here ({len(refused_by_ancestor)})")
    print(f"    both alias settings x {len(PROJECTS)} projects")

    print(f"\n{CHECKS - FAILS}/{CHECKS} checks pass")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
