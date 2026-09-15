"""`s_linker110`'s invariants: a standalone file, and the shortlist is the name relation.

The variant is the reported arm and a **standalone** file -- the whole workflow inlined,
no linker base class -- so the invariants come in two halves.

*That the inlining is faithful* (the half a subclass used to get for free):

  1  **no linker base class, and the extraction surface is gone.** `SLinker110`'s MRO
     is itself and `object`; `ENTITY_EXTRACTION_RULES`, `_prompt_extraction`,
     `_run_extraction_pass` and `EXTRACTION_BATCH` appear nowhere in the code.
  2  **every block that is not a marked delta is `s_linker92`'s, byte for byte**, and
     every marked delta is its own source file's block -- checked by re-splitting the
     files, so it tests the file rather than whatever produced it. One block is exempt:
     `_classify_mention_typed`, whose two unreachable `MentionType` members are pruned
     here, is checked by label equality over every (component, sentence) pair instead.
  3  **the composed behaviour is the chain's.** Over all five projects, this file's
     proposer agrees with `s_linker92a`'s, its partial-name scan with `s_linker109`'s,
     and its evidence bundles with `s_linker92`'s, under an empty alias table and a
     populated one.

*That the arm is what it claims* (the original three, unchanged):

  4  **the shortlist is the module's own relation, not a second one.** For every case of
     every project, `_named_before` returns exactly the components `_states_a_name`
     finds in the rows strictly above the target, with the latest such row -- checked
     against an independent recomputation, not against itself.
  5  **the prompt is the head's plus the shortlist and nothing else.** The case blocks
     differ from `s_linker92`'s only by the added `NAMED BEFORE THIS CASE` line, and the
     reply schema keeps every field the parser reads.
  6  **it is a shortlist in fact.** The per-project count of components listed a case,
     against the catalog's size.

No LLM calls.

    ../.venv/bin/python pilot/test_s110_shortlist.py
"""
from __future__ import annotations

import ast
import dataclasses
import difflib
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import BENCH, PROJECTS                                # noqa: E402
from llm_sad_sam.core.data_types_v2 import DocumentKnowledge            # noqa: E402
from llm_sad_sam.core.document_loader_v2 import load_sentences          # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository              # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker110 as VARIANT     # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker92 as HEAD         # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92 import SLinker92       # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92a import SLinker92a     # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker109 import SLinker109     # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker110 import SLinker110     # noqa: E402
from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names   # noqa: E402

EXPERIMENTAL = Path("src/llm_sad_sam/linkers/experimental")
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


def build(cls, aliases=()):
    linker = cls.__new__(cls)
    linker.doc_knowledge = DocumentKnowledge(aliases=dict(aliases))
    linker.llm = _NoCalls()
    return linker


# ── the class-body splitter ──────────────────────────────────────────────────
# `ast` line ranges, plus the comment/blank lines directly above each statement, so
# a `#:` doc-comment or a `HEAD DELTA` banner travels with the block it introduces.
# The blocks of one class body tile it exactly.

def class_blocks(path, class_name):
    lines = (EXPERIMENTAL / path).read_text().splitlines(keepends=True)
    tree = ast.parse("".join(lines))
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == class_name)
    out, prev_end = {}, None
    for stmt in cls.body:
        first = min([stmt.lineno]
                    + [d.lineno for d in getattr(stmt, "decorator_list", [])])
        lo = first
        if prev_end is not None:
            while lo - 1 > prev_end and (not lines[lo - 2].strip()
                                         or lines[lo - 2].lstrip().startswith("#")):
                lo -= 1
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = stmt.name
        elif isinstance(stmt, ast.Assign):
            name = ast.unparse(stmt.targets[0])
        else:
            name = "<docstring>"
        out[name] = "".join(lines[lo - 1:stmt.end_lineno])
        prev_end = stmt.end_lineno
    return out


def nomark(block):
    """The block without the `HEAD DELTA` banner above it."""
    return "".join(line for line in block.splitlines(keepends=True)
                   if "HEAD DELTA" not in line).strip()


def code_ast(block, rename=None):
    """The block's AST with every docstring dropped -- code, not prose."""
    source = textwrap.dedent(nomark(block))
    if rename:
        source = source.replace(*rename)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:]
    return ast.dump(tree)


def structure():
    """Groups 1 and 2 -- the file is s_linker92 plus three marked deltas."""
    mine = class_blocks("s_linker110.py", "SLinker110")
    head = class_blocks("s_linker92.py", "SLinker92")
    a92 = class_blocks("s_linker92a.py", "SLinker92a")
    n109 = class_blocks("s_linker109.py", "SLinker109")

    check(SLinker110.__mro__ == (SLinker110, object),
          f"no linker base class (mro={[c.__name__ for c in SLinker110.__mro__]})")
    code = (EXPERIMENTAL / "s_linker110.py").read_text().split('"""')[2]
    for token in ("ENTITY_EXTRACTION_RULES", "_prompt_extraction",
                  "_run_extraction_pass", "EXTRACTION_BATCH",
                  "SLinker92", "SLinker109"):
        check(token not in code, f"{token} does not appear in the code")

    #: The infrastructure blocks, byte-identical in all 72 linkers of the branch, now
    #: living in `linker_infra`. They are exempt from the byte comparison and get a
    #: stronger check instead: the block that stayed here must be a delegation to the
    #: named helper, and `pilot/test_linker_infra.py` proves each helper behaves as
    #: `s_linker92`'s bytes did. No mixin: the MRO check above still has to hold.
    delegated = {
        "_ask": "ask_json", "_iter_batches": "iter_batches",
        "_link_view": "link_view", "_decision_view": "decision_view",
        "_linker_feedback": "linker_feedback", "_backend_tag": "backend_tag",
        "_checkpoint_dir": "checkpoint_dir", "_save_phase": "save_phase_state",
        "_log": "log_entry", "_save_log": "write_run_logs",
        "_compute_phase_metrics": "phase_metrics",
    }
    for name, helper in sorted(delegated.items()):
        calls = {n.func.id for n in ast.walk(ast.parse(textwrap.dedent(mine[name])))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        check(helper in calls,
              f"{name} delegates to linker_infra.{helper} (calls={sorted(calls)})")
        check(len(mine[name].splitlines()) <= len(head[name].splitlines()),
              f"{name}'s delegation is no longer than the block it replaces")

    #: What the promotion is allowed to have touched. Anything else must be the head's.
    #: `_classify_mention_typed` is here for a pruning, not a delta: `MentionType` loses
    #: `LOWERCASE_PROSE` (blanked by `RETAINED_MENTION_TYPES`, so it never reached a
    #: prompt) and `INDIRECT` (unproducible once bundles cover only the full-name
    #: proposer). Byte-equality is replaced by `composed`'s label check, which compares
    #: what the prompt actually carries over every (component, sentence) pair.
    touched = {"<docstring>", "_VARIANT_NAME", "ASK_ATTEMPTS", "__init__",
               "_classify_mention_typed",
               "_run_full_name_linker", "SKIP_QUALIFIED", "_named_spans",
               "_writes_name", "_extract_named_mentions", "_scan_all",
               "_covering_names", "_only_inside_another_name", "_scan",
               "_named_before", "_prompt_coref"} | set(delegated)
    check(set(head) - set(mine) == {"EXTRACTION_BATCH", "_prompt_extraction",
                                   "_run_extraction_pass"},
          f"exactly the extraction surface is gone ({sorted(set(head) - set(mine))})")
    check(set(mine) - set(head) == {"SKIP_QUALIFIED", "_named_spans", "_writes_name",
                                    "_scan_all", "_covering_names",
                                    "_only_inside_another_name", "_named_before"},
          f"exactly the deltas are added ({sorted(set(mine) - set(head))})")
    for name in sorted(set(mine) & set(head) - touched):
        check(mine[name] == head[name], f"{name} is s_linker92's block byte for byte")

    for name in ("SKIP_QUALIFIED", "_named_spans", "_writes_name",
                 "_extract_named_mentions"):
        check(nomark(mine[name]) == nomark(a92[name]),
              f"HEAD DELTA 1: {name} is s_linker92a's block")
    for name in ("_covering_names", "_only_inside_another_name"):
        check(nomark(mine[name]) == nomark(n109[name]),
              f"HEAD DELTA 2: {name} is s_linker109's block")
    # the head's generator, renamed, and s109's filter over it instead of over super()
    check(code_ast(mine["_scan_all"], ("def _scan_all(", "def _scan("))
          == code_ast(head["_scan"]),
          "HEAD DELTA 2: _scan_all is the head's _scan, renamed (AST-equal)")
    check(code_ast(mine["_scan"], ("self._scan_all(", "super()._scan("))
          == code_ast(n109["_scan"]),
          "HEAD DELTA 2: _scan is s_linker109's filter, base call redirected")

    check(SLinker110.COREFERENCE_BATCH == SLinker92.COREFERENCE_BATCH,
          "the resolution batch size is the head's")
    for constant in {n for n, v in vars(VARIANT).items()
                     if n.isupper() and isinstance(v, str)}:
        check(getattr(VARIANT, constant) == getattr(HEAD, constant, None),
              f"{constant} is byte-equal to the head's")


def composed(project, sentences, components, sent_map, name_to_id, aliases, tag):
    """Group 3 -- the composed behaviour is s92a's proposer and s109's scan."""
    arm = build(SLinker110, aliases)
    prop, nest, base = (build(SLinker92a, aliases), build(SLinker109, aliases),
                        build(SLinker92, aliases))
    label = f"{project}/{tag}"

    got = arm._extract_named_mentions(sentences, components, name_to_id, sent_map)
    want = prop._extract_named_mentions(sentences, components, name_to_id, sent_map)
    check(got == want, f"{label}: the proposer is s_linker92a's ({len(want)} pairs)")

    check(arm._scan(sentences, components) == nest._scan(sentences, components),
          f"{label}: the partial-name scan is s_linker109's")
    check(arm._scan_all(sentences, components) == base._scan(sentences, components),
          f"{label}: _scan_all is the head's unfiltered generator")

    fields = {k: dataclasses.astuple(v) for k, v in (
        ((c.sentence_number, c.component_id), arm._build_evidence_bundle(c, sent_map))
        for c in want.values())}
    want_fields = {k: dataclasses.astuple(v) for k, v in (
        ((c.sentence_number, c.component_id), base._build_evidence_bundle(c, sent_map))
        for c in want.values())}
    # `EvidenceBundle` is this file's own class, so the dataclasses are compared by
    # field rather than by identity of their type.
    check(fields == want_fields,
          f"{label}: the evidence bundles are the head's ({len(fields)})")

    # The pruned classifier, checked where the byte comparison no longer runs: over
    # every (component, sentence) pair, not only the pairs a candidate exists for.
    labels = {(c.name, s.number): arm._retained_mention_label(c.name, s.text)
              for c in components for s in sentences}
    want_labels = {(c.name, s.number): base._retained_mention_label(c.name, s.text)
                   for c in components for s in sentences}
    check(labels == want_labels,
          f"{label}: the retained mention label is the head's ({len(labels)} pairs)")


def shortlist_and_prompt(project, sentences, components, sent_map, comp_names):
    """Groups 4 and 5 -- the shortlist is the name relation, the prompt is head+it."""
    arm, base = build(SLinker110), build(SLinker92)

    for _, batch in arm._iter_batches(sentences, arm.COREFERENCE_BATCH):
        targets, window_ids = [], set()
        for i, sentence in enumerate(batch, 1):
            window = [w.number for w in arm._window(sentence.number, sentences)]
            window_ids.update(window)
            targets.append({"case": i, "target": sentence.number,
                            "text": sentence.text, "context": window})
        table = [{"sentence": n, "text": sent_map[n].text}
                 for n in sorted(window_ids) if n in sent_map]

        # ── 4. the shortlist is `_states_a_name` over the rows above ─────────
        for target in targets:
            got = dict(arm._named_before(comp_names, table, target["target"]))
            want = {}
            for row in table:
                if row["sentence"] >= target["target"]:
                    continue
                for component in comp_names:
                    if arm._states_a_name(row["text"], component):
                        want[component] = max(want.get(component, 0), row["sentence"])
            check(got == want,
                  f"{project} S{target['target']}: shortlist == the name relation")
            check(all(0 < n < target["target"] for n in got.values()),
                  f"{project} S{target['target']}: every cited row is above the target")

        # ── 5. the prompt is the head's plus the shortlist ───────────────────
        mine = arm._prompt_coref(comp_names, table, targets)
        theirs = base._prompt_coref(comp_names, table, targets)
        diff = list(difflib.ndiff(theirs.splitlines(), mine.splitlines()))
        added = [line[2:] for line in diff if line.startswith("+ ")]
        removed = [line[2:] for line in diff if line.startswith("- ")]
        # the reply schema line is replaced, because the template adds two fields
        # to it; nothing else may go.
        check(all('"resolutions"' in line for line in removed),
              f"{project}: only the reply schema line is replaced ({removed[:1]})")
        check(all(line.startswith("NAMED BEFORE THIS CASE:")
                  or "NAMED BEFORE THIS CASE" in line
                  or line.strip() == "" or "Quote the referring expression" in line
                  or "already been checked" in line or "actually name" in line
                  or "list could be what it points to" in line
                  or '"reference"' in line or '"candidates"' in line
                  for line in added),
              f"{project}: every added line is the shortlist or its instruction")
        for field in ('"resolutions"', '"sentence"', '"component"',
                      '"antecedent_sentence"', '"antecedent_text"'):
            check(field in mine, f"{project}: the reply schema keeps {field}")
        check(HEAD.COREF_RULES in mine, f"{project}: COREF_RULES appears verbatim")


def main():
    structure()

    first = True
    for name, (text, model, _) in PROJECTS.items():
        sentences = load_sentences(str(BENCH / text))
        components = parse_pcm_repository(str(BENCH / model))
        comp_names = get_comp_names(components)
        sent_map = {s.number: s for s in sentences}
        name_to_id = {c.name: c.id for c in components}

        # a populated alias table drawn from the catalog itself, so the knowledge
        # regime the scans read is exercised and stays deterministic
        populated = {}
        for component in components:
            words = component.name.split()
            if len(words) > 1:
                populated.setdefault(words[-1].lower(), component.name)
        for tag, aliases in (("noknow", {}), ("aliases", populated)):
            composed(name, sentences, components, sent_map, name_to_id, aliases, tag)

        if first:
            # one project's full batching is the prompt contract; the rest repeat it
            shortlist_and_prompt(name, sentences, components, sent_map, comp_names)
            first = False

        # ── 6. the shortlist is a shortlist, over the resolver's own windows ──
        arm = build(SLinker110)
        listed, cases = 0, 0
        for _, batch in arm._iter_batches(sentences, arm.COREFERENCE_BATCH):
            window_ids = set()
            for sentence in batch:
                window_ids.update(w.number
                                  for w in arm._window(sentence.number, sentences))
            table = [{"sentence": n, "text": sent_map[n].text}
                     for n in sorted(window_ids) if n in sent_map]
            for sentence in batch:
                listed += len(arm._named_before(comp_names, table, sentence.number))
                cases += 1
        check(cases > 0, f"{name}: the shortlist is computable")
        print(f"  {name}: {listed / max(cases, 1):.1f} of {len(comp_names)} components "
              f"listed a case, over the resolver's own window")

    passed = sum(1 for ok, _ in CHECKS if ok)
    print(f"\n{passed}/{len(CHECKS)} checks")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
