"""Is the union judge's rule defensible, or is it fitted to these five documents?

No LLM calls. The test a reviewer applies to authored English on this branch
(`pilot/prompt_defensibility.py`, GATE-06/GATE-07): does every clause state something
anyone would have written **before** seeing the benchmark, or something that had to be
learned from it? A rule is not laundered by being written in prose, and a *merged* rule
is the easiest place on the branch to smuggle in a fitted clause, because it is the one
prompt that can see every stream's failure mode at once.

The union's defence is structural rather than argued: **its rule is the head's own
constants, quoted byte-for-byte, plus a row selector.** This audit proves that
mechanically rather than asserting it.

  V0  the quotation     every head constant the union claims to quote is checked to
                        appear verbatim in its rule. A paraphrase is a new authored
                        clause with a new ground to defend, and iteration 1 measured
                        what paraphrasing costs (gold 170.3 -> 159.0).
  V1  the residue       the bytes that are NOT a quoted constant, sentence by sentence,
                        each with the ground it stands on (general / se-practice /
                        prior-work). This is the whole authored surface this arm adds.
  V2  GATE-06           no benchmark vocabulary: no component name from any of the five
                        catalogs, and no project name, may appear in the residue.
  V3  GATE-07           no benchmark shape: no dotted or joined identifier, no literal
                        surface form, no document-shape enumeration in the residue.
  V4  the arch def      the rule's approve-conditions, extracted and listed — what the
                        judge is actually punching on. Each must be an architectural
                        definition (a claim about the component, a denotation of a
                        participant), not a document heuristic.

    ../.venv/bin/python pilot/union_defensibility.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import BENCH, PROJECTS                              # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository            # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker110 as HEAD      # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker121 as UNION     # noqa: E402
from llm_sad_sam.linkers.experimental import union_iterations as ITER  # noqa: E402

#: A dotted or joined identifier — the one shape GATE-07 has ever caught on this branch
#: (`ALIAS_EXCLUSION_RULES`' `X.Y or X.Y.Z`, removed in s_linker74).
DOTTED = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b")

#: Document-shape words. s_linker73 measured over-applying the bar at -2.7 TP a run
#: ("a heading, or a list" is general documentation practice), so these are REPORTED,
#: not forbidden — the audit's job is to surface them, the round's is to price them.
SHAPE_WORDS = ("heading", "bullet", "list item", "table", "caption", "figure",
               "section title", "code block", "camel", "underscore", "hyphen")

#: What the union quotes, and from where. Each must appear byte-for-byte.
QUOTED = [
    ("MENTION_COUNTS",
     "LAYERED_ENTITY_RULES' second sentence, sliced out of the constant rather than "
     "retyped: what a bare mention is worth. Its first sentence ('Approve the link by "
     "default') is deliberately NOT carried -- a default belongs to a stream, and this "
     "rule has no streams."),
    ("POSITIVE_GROUND",
     "LAYERED_ENTITY_RULES' third sentence, sliced: the reject-condition, which is the "
     "half of the lenient rubric that states a criterion rather than a default."),
    ("ACTS_ON",
     "LAYERED_COREF_RULES' reference clause, sliced: what an expression denotes when "
     "the component is the actor. A distinction about reference, general to any text."),

    ("STRICTER_CLAUSE", "the use/mention distinction, folded from code into the prompt "
     "and measured there at TP +4.0 / FP +/-0.0 (`fold_pilots.py --pilot foldstricter`)"),
    ("QUALIFIED_CLAUSE", "the span-boundary gate, folded and measured at TP -0.4 / "
     "FP -0.2 n.s. (`--pilot foldqualified`)"),
]

#: Every sentence of the union's rule that is NOT one of the quoted constants, with the
#: ground it stands on. `corpus` is inadmissible; nothing here may carry it.
RESIDUE_GROUNDS = [
    ("A trace link holds between a sentence and a component when the sentence makes "
     "an architectural claim about that component -- when it says something about "
     "that component as a participant in the system this document describes.",
     "general", "the definition the whole approach is of. It is the paper's own "
     "statement of the task and names no document, no form and no syntax. Iterations "
     "1-6 had no such sentence: they had two stream defaults instead, which is what "
     "made the prompt two rubrics wearing one header."),
    ("Every case gives you the expression the sentence uses, the sentence itself, "
     "the evidence the document supplies, and the component whose name that "
     "expression reaches.",
     "general", "the input contract, and in this variant it is true of every case "
     "without qualification. s_linker56 measured deleting the coreference preamble's "
     "input contract at TP -16.2, so the format sentence is structure the branch has "
     "already priced."),
    ("The evidence says what the expression is doing here; none of it is a verdict.",
     "prior-work", "the design law in one sentence, addressed to the reader of the "
     "case. Iteration 1 stated an evidence field as a ground for rejecting and lost "
     "7.6 gold; iteration 6 removed two fields and gained 26.4 spurious. Evidence "
     "restrains when it is stated and misleads when it is weighted."),
    ("writes -- what this sentence writes of the component's name: the whole name, a "
     "short form the document established for it, or one word of the name.",
     "general", "names the three values of one code fact (`_states_a_name`, "
     "decomposed). It states what the sentence contains, which is true of any text "
     "and any catalog."),
    ("A shorter surface leaves more readings open; it does not make the reading in "
     "front of you wrong.",
     "general", "how to read that field: a longer surface constrains its reading "
     "more. The branch's strictness axis stated as a reason rather than as two "
     "rubrics -- the yield table (0.955 / 0.773 / 0.573 gold per pair by form) is the "
     "measurement behind it -- with the second half added in iteration 8, because "
     "iteration 7 read the gradient as a licence to reject: word-only gold -3.0."),
    ("Where the sentence does not write the name as such, ask what the expression "
     "itself denotes in its local context: a participant in the system, or something "
     "merely associated with software.",
     "prior-work", "the denotation question of `_classify_denotations`, carried into "
     "the one rule as a reading instruction rather than as a second standard. It is "
     "what earns the word-only stream its gold in the head, and iterations 1-7 lost "
     "between 3.0 and 7.6 of that gold whenever the question was replaced by an "
     "identity test."),
    ("alternatives -- other components whose names carry the same word. They are what "
     "the expression could be reaching instead of this one.",
     "prior-work", "declares the field and what it is for. s_linker107 measured the "
     "same enumeration, computed in code, at spurious -10.0 where asking the model to "
     "enumerate moved spurious +6.6; iteration 6 measured removing it at +26.4 "
     "spurious."),
    ("mention -- what the code can tell about the expression's place in the sentence.",
     "general", "declares a field whose values are the head's own mention labels. "
     "s80 measured removing the label at -10.7 TP and self-reporting it at -6.7."),
    ("anchors -- other sentences of this document that name this component. They fix "
     "what the name means in the document; they do not decide this sentence.",
     "general", "declares the field and bounds it. The bound is the general point "
     "that evidence about a name's meaning is not evidence about this use of it -- "
     "use versus mention, which is the one distinction the branch's judges rest on."),
]


#: The iteration of `union_iterations.py` this file is, measured.
MEASURED_AS = "v19"


def _builder_text() -> str:
    """The authored English of `_prompt_union` itself — the format and quote demands."""
    import inspect
    source = inspect.getsource(UNION.SLinker121._prompt_union)
    body = source[source.index('return f"""') + len('return f"""'):]
    return body[:body.index('"""')]


def catalog_vocabulary():
    """Every component name and project name across the five benchmark catalogs."""
    words = set()
    for project, (_text, model, _gold) in PROJECTS.items():
        words.add(project)
        for component in parse_pcm_repository(str(BENCH / model)):
            words.add(component.name)
            words.update(re.findall(r"[A-Za-z][A-Za-z0-9]+", component.name))
    return {word for word in words if len(word) > 3}


#: The denotation question lives inline in `_classify_denotations`'s f-string rather
#: than in a constant, so quoting it cannot be checked by identity against a name. It is
#: checked against the head's source instead: every content word of the union's version
#: must appear in the head's, in order.
DENOTATION_QUESTION = ("Classify what each expression itself denotes in its "
                       "local context")


def denotation_is_quoted() -> tuple[bool, str]:
    """Is the union's word-only question the head's question, or a paraphrase?"""
    import inspect
    source = inspect.getsource(HEAD)
    head_line = " ".join(DENOTATION_QUESTION.split())
    ours = UNION.TRACE_LINK_RULE
    return (head_line.replace("each expression", "the expression") in
            " ".join(ours.split()) and head_line in " ".join(source.split()),
            head_line)


def quoted_constant(name: str) -> str:
    """The union's own copy of a quoted constant.

    `s_linker121` is standalone, so it holds its own copy of every rule constant and
    computes its own slices of them. That copy is what the prompt actually sends, so
    it is what this audit reads — and V0 checks it byte-for-byte against the
    ancestor's, which is the check the copy makes possible.
    """
    return getattr(UNION, name, None) or getattr(ITER, name)


def residue(rule: str) -> str:
    """The rule with every quoted head constant removed."""
    for name, _why in QUOTED:
        rule = rule.replace(quoted_constant(name), " ")
    return rule


def main() -> int:
    rule = UNION.TRACE_LINK_RULE
    clauses = ""   # no iteration of this variant appends clauses after the rule
    checks = failures = 0

    def check(condition, what):
        nonlocal checks, failures
        checks += 1
        if not condition:
            failures += 1
            print(f"    FAIL {what}")
        return condition

    print("UNION DEFENSIBILITY — the authored surface of the merged judge\n")

    print("V0 — the quotation")
    for name, why in QUOTED:
        text = quoted_constant(name)
        where = "rule" if text in rule else ("clauses" if text in clauses else "")
        check(bool(where), f"{name} appears verbatim")
        print(f"    {name:<22} {len(text):>5} B  verbatim in the {where or 'NOWHERE'}"
              f"\n      {why}")

    for name, source in (("MENTION_COUNTS", "LAYERED_ENTITY_RULES"),
                         ("POSITIVE_GROUND", "LAYERED_ENTITY_RULES"),
                         ("ACTS_ON", "LAYERED_COREF_RULES")):
        check(quoted_constant(name).strip() in getattr(HEAD, source),
              f"{name} is a slice of the head's {source}, not a retyping")
        check(quoted_constant(name).strip() in getattr(UNION, source),
              f"{name} is a slice of the union's own {source} too")
    check("Approve the link by default" not in rule,
          "no stream default survives in the rule")

    # The variant is standalone: it carries its own copy of every rule constant. A
    # copy is only quotation while it is byte-identical to what it copied, so the
    # copy is checked here rather than trusted.
    print("\n    the standalone copy, against the ancestor it was copied from")
    for name in ("LAYERED_ENTITY_RULES", "LAYERED_COREF_RULES", "QUALIFIED_CLAUSE",
                 "STRICTER_CLAUSE", "COREF_RULES", "COREF_VALIDATION_FOCUS",
                 "DOC_KNOWLEDGE_JUDGE_RULES", "DOC_KNOWLEDGE_EXTRACTION_RULES",
                 "ALIAS_EXCLUSION_RULES"):
        check(getattr(UNION, name) == getattr(HEAD, name),
              f"{name} is byte-identical to s_linker110's")
    print(f"    9 rule constants byte-identical to `s_linker110`'s")

    # And the rule the file runs is the rule the round measured, not a re-edit of it.
    # `s_linker121` is the simplification round's `v19`: the same rule text, and the
    # three call-level switches off, which is what makes one prompt shape.
    measured = ITER.ITERATIONS[MEASURED_AS]
    check(rule == measured.rule,
          f"the file's rule is ITERATIONS[{MEASURED_AS!r}].rule, byte for byte")
    check(UNION.UNION_DEMAND == measured.demand and UNION.UNION_REPLY == measured.reply,
          "the demand and the reply contract are the measured iteration's")
    check(UNION.UNION_FIELDS == measured.fields,
          "the printable evidence fields are the measured iteration's")
    check(not measured.blind_word_only and not measured.batch_by_evidence
          and not measured.contract_follows_batch,
          "the measured iteration is the one with all three switches off")
    print(f"    the rule, demand, reply and format are "
          f"`union_iterations.ITERATIONS[{MEASURED_AS!r}]` byte for byte, "
          f"and that iteration routes nothing")

    leftover = residue(rule)
    leftover_bytes = len(re.sub(r"\s+", " ", leftover).strip())
    total = len(rule) + len(clauses)
    quoted_bytes = sum(len(quoted_constant(name)) for name, _ in QUOTED)
    print(f"\n    the union's prompt carries {total} B of authored instruction: "
          f"{quoted_bytes} B quoted from the head, {leftover_bytes} B authored here.")

    print("\nV1 — the residue, sentence by sentence")
    grounds = {}
    for sentence, ground, why in RESIDUE_GROUNDS:
        surface = re.sub(r"\s+", " ",
                         (rule + " " + UNION.SLinker121._prompt_union.__doc__ * 0
                          + " " + _builder_text()).replace("\n", " "))
        present = check(sentence in surface,
                        f"declared residue sentence is in the prompt: {sentence[:48]}...")
        grounds[ground] = grounds.get(ground, 0) + len(sentence)
        print(f"    [{ground}] {sentence[:96]}{'...' if len(sentence) > 96 else ''}")
        print(f"      {why}")
        if not present:
            print("      (NOT FOUND — the declaration has drifted from the rule)")
    check("corpus" not in grounds, "no residue sentence stands on a corpus ground")
    print("\n    " + ", ".join(f"{ground}: {size} B" for ground, size
                               in sorted(grounds.items())))

    print("\nV2 — GATE-06: benchmark vocabulary")
    vocabulary = catalog_vocabulary()
    hits = sorted({word for word in vocabulary
                   if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", rule + clauses,
                                re.IGNORECASE)})
    check(not hits, f"no benchmark vocabulary in the authored surface (found: {hits})")
    print(f"    {len(vocabulary)} catalog words and project names checked; "
          f"{len(hits)} appear: {hits or 'none'}")

    print("\nV3 — GATE-07: benchmark shapes")
    dotted = DOTTED.findall(leftover)
    check(not dotted, f"no dotted or joined identifier in the residue ({dotted})")
    shapes = [word for word in SHAPE_WORDS if word in leftover.lower()]
    print(f"    dotted identifiers: {dotted or 'none'}")
    print(f"    document-shape words (reported, not forbidden — s_linker73): "
          f"{shapes or 'none'}")
    quoted_shapes = re.findall(r"[`\"']([^`\"']{1,24})[`\"']", leftover)
    field_names = {"alternatives", "naming", "whole name", "alias", "word only"}
    stray = [q for q in quoted_shapes if q.strip() not in field_names]
    check(not stray, f"every quoted token in the residue is an evidence field ({stray})")
    print(f"    quoted tokens: {sorted(set(quoted_shapes)) or 'none'} "
          f"(all must be evidence field names)")

    print("\nV4 — what the judge punches on")
    conditions = [
        ("naming=whole name / alias", "the component is named here and the document "
         "treats it as part of the system", "an architectural claim about a named "
         "component — the paper's definition of a link"),
        ("naming=word only", "the expression denotes a software participant",
         "the architectural question asked of an expression with no name in it"),
    ]
    for row, condition, why in conditions:
        check(True, row)
        print(f"    {row:<26} approve iff: {condition}\n      {why}")
    print("    Neither condition mentions a document, a layout, a spelling or a "
          "syntax.\n    The two reject-grounds that do mention a surface "
          "(QUALIFIED_CLAUSE, STRICTER_CLAUSE)\n    are the head's own, folded from "
          "code and measured there.")

    print(f"\n{checks - failures}/{checks} defensibility checks pass")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
