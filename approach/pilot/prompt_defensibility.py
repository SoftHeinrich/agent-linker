"""Which authored sentences are defensible a priori, and which are corpus-shaped.

The deterministic layer of this workflow has been audited to exhaustion; the ~4 kB of
authored English carried into 91 calls per five-project run has not, since `s_linker49`.
Two folds have since *moved* rules into that English (`QUALIFIED_CLAUSE`,
`STRICTER_CLAUSE`), which makes the question sharper rather than softer: a rule is not
laundered by being written in prose. The test a reviewer will apply is whether each
sentence states something anyone would have written **before** seeing these five
documents, or something that had to be *learned* from them.

Two failure modes, and they are different:

    corpus-shaped  the sentence names a surface form or a syntax that is frequent in
                   this benchmark and need not be frequent anywhere else. The branch
                   already condemned one instance of exactly this in its own comment
                   (`ALIAS_EXCLUSION_RULES`: "naming the shape is a rule written for
                   one corpus") and then kept the shape in the string.
    redundant      the sentence restates something another sentence in the same prompt
                   already says. After two folds this is no longer hypothetical: the
                   folded clauses cover reject-conditions the enumerated list still
                   spells out.

No LLM calls. Everything is read off `s_linker70`'s own three recorded runs.

    D0  INVENTORY      every authored clause, with the specific reason it is or is not
                       defensible a priori
    D1  THE SYNTAX     the qualified-name stipulation, its copies, and for each copy how
                       much of the population it governs even contains such an identifier
    D2  THE LIST       the four numbered reject-conditions: what each one reaches in the
                       recorded rejections, and which are now covered by a folded clause
    D3  THE EXAMPLES   the three enumerated approve-shapes ("bare mention, heading, list")

A prohibition acts through absence, so reach bounds what a clause *can* be doing, not
what it does. Where reach is zero the clause is provably inert; where it is non-zero the
arm still has to be run. That asymmetry is stated per finding, not hidden.

    ../.venv/bin/python pilot/prompt_defensibility.py
    ../.venv/bin/python pilot/prompt_defensibility.py --runs '../results/s70_solo_r*_20260817'
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

import llm_sad_sam.linkers.experimental.s_linker70 as L               # noqa: E402

#: A dotted or joined identifier: the shape two prompts spell out literally.
DOTTED = re.compile(r"\b[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+\b")

#: Every authored clause, scored on the only test that matters for publication: what
#: does it stand on? Three admissible grounds --
#:
#:   general      a general rule: logic, or a distinction from linguistics that holds
#:                for any text (use vs mention, reference, negation, ambiguity)
#:   se-practice  a property of software as it is written anywhere: qualified names
#:                compose, identifiers are named after what they are
#:   prior-work   a decision this branch or the TLR literature already measured and
#:                defended, and which the paper cites rather than re-argues
#:
#: -- and one inadmissible one: `corpus`, meaning the clause names a surface form or a
#: syntax whose frequency is a fact about these five documents. A clause with no
#: admissible ground has to go, whatever it scores.
INVENTORY = [
    ("DOC_KNOWLEDGE_EXTRACTION_RULES", "general",
     "states what an alias IS -- an equivalence the document establishes -- plus the "
     "ordinary-English caution. Names no form."),
    ("DOC_KNOWLEDGE_JUDGE_RULES", "prior-work",
     "one criterion and a stated tie-break direction. The asymmetry against "
     "LAYERED_COREF_RULES's opposite tie-break is this branch's measured principle: "
     "the looser the proposer, the stricter the judge behind it."),
    ("ALIAS_EXCLUSION_RULES", "corpus",
     "spells out `X.Y or X.Y.Z`. The SE-practice fact behind it is real -- qualified "
     "names compose -- but the clause states the syntax, not the fact, and the "
     "module's own comment three lines above calls naming the shape 'a rule written "
     "for one corpus'."),
    ("ENTITY_EXTRACTION_RULES", "mixed",
     "the admission contract is the stage's definition (general). 'a name that "
     "appears only inside a code-level path' is the syntax again; QUALIFIED_CLAUSE is "
     "the same fact with an SE-practice ground and no syntax."),
    ("P1_FOCUS", "mixed",
     "'architectural participant' is the stage's question (general). 'rather than "
     "only as part of a code-level identifier' is a third copy of the syntax."),
    ("P2_FOCUS", "general",
     "generic-term vs specific-element is the use/mention distinction."),
    ("COREF_VALIDATION_FOCUS", "general", "states the stage's question."),
    ("COREF_RULES", "general",
     "antecedent clarity and abstention under ambiguity are general discourse "
     "criteria; the phrase lists were removed by prompt_audit.py's P3/P4."),
    ("LAYERED_ENTITY_RULES", "corpus",
     "a numbered four-condition reject list. (1) is the syntax, a fourth copy; (2) "
     "negation has a general ground but not a general form; (3) and (4) are the "
     "use/mention distinction, restated by STRICTER_CLAUSE in this same prompt. The "
     "approve side names three document shapes -- bare mention, heading, list -- which "
     "stand on nothing at all and are already licensed by the default."),
    ("LAYERED_COREF_RULES", "mixed",
     "referring-expression and ambiguity criteria are general; 'only to a code-level "
     "identifier' is a fifth copy of the syntax."),
    ("QUALIFIED_CLAUSE", "se-practice",
     "states that a piece of a compound identifier names a piece of it. True of every "
     "language with member access; names no separator, case convention or syntax. "
     "This is what the other five copies should be."),
    ("STRICTER_CLAUSE", "general",
     "the use/mention distinction, with capitalization offered as evidence and "
     "explicitly denied decisive force. Names no form and no component."),
]


#: A case starts at a `Case n:` header (judging, denotation) or an `Sn:` line
#: (extraction, alias). Everything up to the next such marker belongs to it -- the
#: sentence, the `[prev: ...]` line, the evidence line and the anchors. Splitting on
#: the header alone and reading only that line is what makes a prohibition look inert
#: when the identifier it prohibits sits two lines below.
CASE_START = re.compile(r"^(?:Case \d+:|S\d+:|\s*\{?\"?\d+\"?\s*[:.)]) ?", re.M)


def case_blocks(prompt: str):
    """The population a prompt governs, one string per case, with its full context.

    Four prompt shapes, and each hides its text somewhere different. The denotation
    prompt is the one that matters most here: its cases carry only an expression and a
    sentence *number*, so a copy of a prohibition about identifiers looks inert unless
    the numbered sentence is joined back on from the SENTENCES table.
    """
    if '"expression":' in prompt:
        table = {}
        for block in re.findall(r'\{"sentence":\s*(\d+),\s*"text":\s*"(.*?)"\}',
                                prompt):
            table[int(block[0])] = block[1]
        out = []
        for case in re.finditer(
                r'\{"case":\s*\d+,\s*"source":\s*(\d+),\s*"expression":\s*"(.*?)"\}',
                prompt):
            out.append(f'{case.group(2)} :: {table.get(int(case.group(1)), "")}')
        return out
    if "DOCUMENT:" in prompt:
        body = prompt.split("DOCUMENT:", 1)[-1]
    elif "CASES:" in prompt:
        body = prompt.split("CASES:", 1)[-1]
    else:
        body = prompt
    marks = [m.start() for m in CASE_START.finditer(body)]
    if not marks:
        # the alias prompt writes the document as bare lines, unnumbered
        return [line for line in body.splitlines() if line.strip()]
    bounds = marks + [len(body)]
    return [body[bounds[i]:bounds[i + 1]] for i in range(len(marks))]


def load_calls(runs):
    calls = []
    for run in runs:
        for path in sorted((run / "llm_logs").glob("*_calls.json")):
            with path.open() as handle:
                for call in json.load(handle):
                    call["_run"] = run.name
                    calls.append(call)
    return calls


def d0():
    print("=== D0  the authored surface, clause by clause ===\n")
    total = 0
    by = Counter()
    print(f"{'constant':<34}{'bytes':>7}  ground")
    for name, verdict, why in INVENTORY:
        text = getattr(L, name)
        total += len(text)
        by[verdict] += len(text)
        print(f"{name:<34}{len(text):>7}  {verdict}")
        for line in re.findall(r".{1,74}(?:\s|$)", why):
            print(f"{'':<41}{line.strip()}")
    admissible = by["general"] + by["se-practice"] + by["prior-work"]
    print(f"\n    {total} bytes of authored instruction:")
    for ground in ("general", "se-practice", "prior-work", "mixed", "corpus"):
        if by[ground]:
            print(f"      {ground:<12}{by[ground]:>6} bytes")
    print(f"\n    {admissible} bytes stand on an admissible ground; "
          f"{by['mixed'] + by['corpus']} do not, or not entirely.\n")


def d1(calls):
    print("=== D1  the qualified-name syntax: five copies, and what each governs ===\n")
    copies = [
        ("ALIAS_EXCLUSION_RULES", "alias extraction"),
        ("ENTITY_EXTRACTION_RULES", "reference extraction"),
        ("P1_FOCUS", "full-name judging, pass 1"),
        ("LAYERED_ENTITY_RULES", "full-name judging, both passes"),
        ("LAYERED_COREF_RULES", "coreference judging"),
        ("QUALIFIED_CLAUSE", "denotation judging"),
    ]
    n = len({c["_run"] for c in calls}) or 1
    print(f"{'copy':<34}{'calls':>7}{'cases':>8}{'with a dotted id':>19}")
    for name, where in copies:
        text = getattr(L, name)
        hits = [c for c in calls if text in c["prompt"]]
        cases = 0
        dotted = 0
        for call in hits:
            for block in case_blocks(call["prompt"]):
                cases += 1
                dotted += bool(DOTTED.search(block))
        print(f"{name:<34}{len(hits) / n:>7.1f}{cases / n:>8.1f}{dotted / n:>19.1f}"
              f"   ({where})")
    print("\n    Reach bounds what a prohibition can be doing. A copy whose population")
    print("    contains no dotted identifier cannot have changed a decision in these")
    print("    runs; a copy whose population does contain them still needs an arm.\n")


def d2(calls):
    print("=== D2  the four numbered reject-conditions, against what was rejected ===\n")
    # conditions as the rubric states them, and the folded clause that now covers each
    conds = [
        ("(1) code-level or package/member path",
         "COVERED by QUALIFIED_CLAUSE (denotation prompt) -- but that clause is not "
         "in this prompt"),
        ("(2) the mention is negated", "not covered by any folded clause"),
        ("(3) the word names a DIFFERENT entity",
         "COVERED by STRICTER_CLAUSE, which is in this prompt"),
        ("(4) generic technique or technology term",
         "COVERED by STRICTER_CLAUSE, which is in this prompt"),
    ]
    n = len({c["_run"] for c in calls}) or 1
    rules = L.LAYERED_ENTITY_RULES
    judged = rejected = 0
    negated = dotted_case = 0
    for call in calls:
        if rules not in call["prompt"]:
            continue
        try:
            data = json.loads(re.search(r"\{.*\}", call["response_text"], re.S).group())
        except Exception:                                          # noqa: BLE001
            continue
        for v in data.get("validations", []):
            judged += 1
            if not v.get("approve"):
                rejected += 1
                claim = str(v.get("claim", ""))
                negated += bool(re.search(r"\bnot\b|\bno\b|\bnever\b", claim, re.I))
                dotted_case += bool(DOTTED.search(claim))
    print(f"    {judged / n:.1f} cases judged per run under this rubric, "
          f"{rejected / n:.1f} rejected\n")
    print(f"    of the rejections, claims containing a negation: {negated / n:.1f}")
    print(f"    of the rejections, claims containing a dotted identifier: "
          f"{dotted_case / n:.1f}\n")
    for cond, note in conds:
        print(f"    {cond:<44}{note}")
    print("\n    Three of the four conditions are now restated elsewhere in the same")
    print("    prompt or by a clause written to be general. The enumeration is what")
    print("    is left of the accretion, and (2) is the only condition with no")
    print("    general statement standing behind it.\n")


def d3(calls):
    print("=== D3  the three enumerated approve-shapes ===\n")
    n = len({c["_run"] for c in calls}) or 1
    rules = L.LAYERED_ENTITY_RULES
    shapes = {"a heading": 0, "a list": 0, "a bare mention": 0}
    cases = 0
    for call in calls:
        if rules not in call["prompt"]:
            continue
        for line in call["prompt"].splitlines():
            if not line.startswith("Case "):
                continue
            cases += 1
    print(f"    'A bare mention, a heading, or a list ... all count as valid links'")
    print(f"    governs {cases / n:.1f} cases per run. The three shapes are document")
    print("    forms, not architectural criteria: they say *where* a name may sit, and")
    print("    the sentence that follows them ('approve unless a reject-condition")
    print("    fires') already licenses every one of them. The general statement is")
    print("    the default itself.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="../results/s70_solo_r*_20260817")
    args = ap.parse_args()
    runs = sorted(Path().glob(args.runs))
    if not runs:
        raise SystemExit(f"no runs matched {args.runs}")
    calls = load_calls(runs)
    print(f"\nprompt defensibility — s_linker70, {len(runs)} runs, "
          f"{len(calls)} recorded calls\n")
    d0()
    d1(calls)
    d2(calls)
    d3(calls)


if __name__ == "__main__":
    main()
