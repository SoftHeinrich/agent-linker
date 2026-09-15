"""Is `s_linker122` `s_linker121`'s text minus the anchors, and nothing else? No calls.

The branch's policy is one self-contained file per reported variant: the paper's
supplement is the file, so a reported variant carries its whole workflow and no linker
base class. `s_linker122` was a subclass while it was being measured, which made the
claim "only the anchor evidence changes" true by construction and unreadable in the
file. Standing it up standalone makes the claim readable and unchecked — so this checks
it instead:

  T1  structure      the MRO is `(SLinker122, object)` and no linker module other than
                     the shared plumbing is imported at module scope.
  T2  the copy       the two classes have the same method set, and every method is
                     **byte-identical code** except the three the anchors reached.
  T3  the constants  every rule constant is byte-identical to the ancestor's except
                     `_FIELD_LINES`, which loses exactly its `anchors` line.
  T4  the call       over all five projects, every judging prompt this variant sends
                     is the ancestor's prompt with the anchors line removed, every
                     anchor block removed, and the clause placed before the demand —
                     computed from the ancestor's own bytes, not retyped. No case
                     prints an `Anchors` line.
  T5  the stream     the candidates proposed are exactly the ancestor's, per project,
                     under the pinned alias table. Removing evidence must not move
                     what is judged.

    ../.venv/bin/python pilot/test_s122_standalone.py
"""
from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.linkers.experimental import s_linker121 as ANCESTOR   # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker122 as VARIANT    # noqa: E402
from s121_ablations import DATASETS, DEFAULT_RUN, load                 # noqa: E402

CHECKS = FAILS = 0

#: The methods the anchors reached, and why each had to move. Anything else that
#: differs is a copy that drifted; anything missing is a copy that lost a block.
CHANGED = {
    "_union_evidence": "the anchors are not computed",
    "_format_union_case": "the anchors are not printed, so no `shown_in`",
    "_judge_union": "nothing tracks which case showed the block first",
    "_prompt_union": "the clause stands where the block's line stood",
}

#: The rule constants that move, and what each loses. `TRACE_LINK_RULE` is derived —
#: it interpolates `_FIELD_LINES` — so it loses the same line, and checking it against
#: the same computed slice is what makes "only the anchors line goes" a claim about the
#: rule the judge actually reads rather than about one constant.
CHANGED_CONSTANTS = {
    "_FIELD_LINES": "its `anchors` line",
    "TRACE_LINK_RULE": "the same line, through `_FIELD_LINES`",
}

#: The plumbing a standalone variant is allowed to import.
SHARED = {"linker_infra", "helper_v3"}


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
    is the executable content, which is what "only the anchors change" is about.
    """
    source = inspect.getsource(function)
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


def prompts_of(cls, data):
    """Every judging prompt this class would send on this project. No calls."""
    linker = cls.__new__(cls)
    linker.doc_knowledge = data["knowledge"]
    sent = []

    class _Recorder:
        def set_phase(self, phase):
            pass

    linker.llm = _Recorder()
    linker._ask = lambda prompt, **_: sent.append(prompt) or {"validations": []}
    candidates = linker._name_candidates(
        data["sentences"], data["components"], data["name_to_id"], data["sent_map"])
    linker._judge_union(
        candidates, data["components"], data["sentences"], data["sent_map"])
    return sent, candidates


#: The per-case block, which is the only place `_format_union_case` emits four-space
#: continuation lines, in both of its forms (the block itself and the back-reference).
ANCHOR_BLOCK = re.compile(
    r"\n  Anchors \(other sentences naming it\): as shown in Case \d+\."
    r"|\n  Anchors \(other sentences naming it\):(?:\n    [^\n]*)*")


def strip_anchors(prompt: str) -> str:
    """The ancestor's prompt with its anchors line and every anchor block removed.

    The rule's line is sliced off the ancestor's own `_FIELD_LINES` rather than
    retyped, so a drift in the constant fails here instead of passing quietly.
    """
    line = "\n  anchors -- " + ANCESTOR._FIELD_LINES.split("  anchors -- ", 1)[1]
    assert line in prompt, "the ancestor's prompt has no anchors line"
    return ANCHOR_BLOCK.sub("", prompt.replace(line, "", 1))


def main() -> int:
    print("T1 — structure")
    check(VARIANT.SLinker122.__mro__ == (VARIANT.SLinker122, object),
          "the MRO is (SLinker122, object)")
    tree = ast.parse(Path(VARIANT.__file__).read_text())
    imported = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            name = getattr(node, "module", "") or ""
            if "linkers.experimental" in name:
                imported.add(name.rsplit(".", 1)[-1])
    check(imported <= SHARED,
          f"only the shared plumbing is imported at module scope (got {imported})")
    check(VARIANT.SLinker122._VARIANT_NAME == "s_linker122",
          "the variant names itself")

    print("T2 — the copy")
    mine, theirs = methods(VARIANT.SLinker122), methods(ANCESTOR.SLinker121)
    check(mine == theirs,
          f"the method set is the ancestor's (only mine: {mine - theirs}, "
          f"only theirs: {theirs - mine})")
    identical = 0
    for name in sorted(mine & theirs):
        same = (code_of(getattr(VARIANT.SLinker122, name))
                == code_of(getattr(ANCESTOR.SLinker121, name)))
        if name in CHANGED:
            check(not same, f"{name} differs — {CHANGED[name]}")
        else:
            check(same, f"{name} is byte-identical to the ancestor's")
            identical += same
    print(f"     {identical} methods byte-identical, {len(CHANGED)} declared changed")

    print("T3 — the constants")
    anchor_line = ("\n  anchors -- "
                   + ANCESTOR._FIELD_LINES.split("  anchors -- ", 1)[1])
    for name in sorted(vars(ANCESTOR)):
        if name.upper() != name:
            continue
        value = getattr(ANCESTOR, name)
        if not isinstance(value, str):
            continue
        mine_value = getattr(VARIANT, name, None)
        if name in CHANGED_CONSTANTS:
            check(anchor_line in value, f"{name} carries the anchors line to lose")
            check(mine_value == value.replace(anchor_line, "", 1),
                  f"{name} loses exactly {CHANGED_CONSTANTS[name]}")
        else:
            check(mine_value == value, f"{name} is byte-identical")
    check(isinstance(getattr(VARIANT, "SURFACE_NOT_EVIDENCE", None), str),
          "the variant declares the clause that replaces the block")

    print("T4 — the call")
    projects = sorted(DATASETS)
    for project in projects:
        data = load(project, DEFAULT_RUN)
        theirs_prompts, _ = prompts_of(ANCESTOR.SLinker121, data)
        mine_prompts, _ = prompts_of(VARIANT.SLinker122, data)
        if not check(len(mine_prompts) == len(theirs_prompts),
                     f"{project}: the same number of calls"):
            continue
        for index, (a, b) in enumerate(zip(mine_prompts, theirs_prompts), 1):
            want = strip_anchors(b).replace(
                VARIANT.UNION_DEMAND,
                f"{VARIANT.SURFACE_NOT_EVIDENCE}\n\n{VARIANT.UNION_DEMAND}", 1)
            check(a == want,
                  f"{project} call {index}: the ancestor's bytes minus the anchors, "
                  f"plus the clause")
            check("Anchors" not in a, f"{project} call {index}: no anchors printed")

    print("T5 — the stream")
    for project in projects:
        data = load(project, DEFAULT_RUN)
        _, mine_candidates = prompts_of(VARIANT.SLinker122, data)
        _, theirs_candidates = prompts_of(ANCESTOR.SLinker121, data)

        def key(candidates):
            return sorted((c.sentence_number, c.component_id, c.source)
                          for c in candidates)

        check(key(mine_candidates) == key(theirs_candidates),
              f"{project}: the candidates are the ancestor's")

    print(f"\n{CHECKS} checks, {FAILS} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
