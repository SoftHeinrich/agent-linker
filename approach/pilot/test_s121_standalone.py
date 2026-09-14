"""Is `s_linker121` a complete workflow that changed only its name judging? No calls.

The branch's policy is one self-contained file per reported variant: the paper's
supplement is the file, so a reported variant carries its whole workflow and no linker
base class. A subclass would make the claim "only the name judging changes" true by
construction and unreadable; `s_linker121` is standalone, which makes the claim
readable and unchecked — so this checks it, against `s_linker110`, which holds the
workflow it did not change. The comparison lives HERE and not in the linker, which is
the point: the file describes itself, and this file is what ties it to the branch.

  T1  structure      the MRO is `(SLinker121, object)`, and no linker module is
                     imported at module scope.
  T2  the copy       every method the two classes share has **identical code** —
                     same AST, docstrings and comments stripped — except the ones
                     declared in CHANGED, and the only methods the union drops are
                     the name-judging path it replaces. Prose is deliberately NOT
                     compared: this file describes itself, so its comments name no
                     other variant, and pinning them to another file's wording is
                     what would make it not standalone.
  T3  the constants  every rule constant is byte-identical to the ancestor's.
  T4  the prompts    the knowledge prompts, the resolver prompt and the coreference
                     judging prompt render byte-identically to the ancestor's over all
                     five projects — the last one against `_prompt_validation(...,
                     strict=True)`, the call this variant specialized away.
  T5  the stream     both scans propose exactly what the ancestor's scans propose, and
                     the union's merged stream is exactly their union, per project,
                     under an empty alias table and a recorded one.

    ../.venv/bin/python pilot/test_s121_standalone.py
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
from llm_sad_sam.linkers.experimental import s_linker121 as UNION     # noqa: E402

#: The methods the union replaces, and the machinery only they reached. Anything else
#: missing from the standalone file is a copy that lost a block.
REPLACED = {
    "_prompt_validation",          # the two-rubric prompt builder: one rubric now
    "_run_full_name_linker",       # both name stages are `_run_name_linker`
    "_run_partial_name_linker",
    "_judge_partial_names",
    "_classify_denotations",       # its question survives in the `writes` line
    "_build_evidence_bundle",      # the bundle is `_union_evidence`
    "_format_evidence",
    "_anchor_union",
    "_validate_with_evidence",
}

#: The methods that differ in source, and why each had to.
CHANGED = {
    "_run_linker": "two linkers, not three",
    "_run_validation_pass": "no `strict` argument: one rubric reaches it",
    "_validate_coref_links": "the same call, without the `strict` argument",
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

    print("\nT2 — the copy, method by method")
    head_methods, union_methods = methods(HEAD.SLinker110), methods(UNION.SLinker121)
    identical, reworded = [], []
    for name in sorted(head_methods & union_methods):
        left = getattr(HEAD.SLinker110, name)
        right = getattr(UNION.SLinker121, name)
        if code_of(left) == code_of(right):
            identical.append(name)
            if inspect.getsource(left) != inspect.getsource(right):
                reworded.append(name)
        else:
            check(name in CHANGED, f"{name} differs in CODE and is declared")
    check(set(head_methods) - set(union_methods) == REPLACED,
          f"exactly the name-judging path is replaced "
          f"({sorted(set(head_methods) - set(union_methods) - REPLACED)} unexpected)")
    for name in CHANGED:
        check(name in union_methods, f"{name} is present and rewritten")
    print(f"    {len(identical)} methods code-identical to `s_linker110` "
          f"({len(reworded)} of them reworded so this file describes itself), "
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
            full, partial = arm_full(data), arm_partial(data)
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
    print(f"    both alias settings x {len(PROJECTS)} projects")

    print(f"\n{CHECKS - FAILS}/{CHECKS} checks pass")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
