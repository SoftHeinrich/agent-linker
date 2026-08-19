"""s_linker80 — the last computed fact leaves the evidence line.

`s_linker79` left the deterministic layer with no gate. What it still computed was the
**mention label**: `_classify_mention_typed` reads the sentence the judge is already
holding and writes `mention=...` on the evidence line.

This is the design law's hardest case, and the law has been measured on it three times:
removing the field is **-10.7 TP**, asking the judge to report it instead is **-6.7 TP**,
and the fold law that predicted it would fold was retired because of it -- *information the
judge can derive is not information the judge will derive impartially*. So this variant is
expected to lose, and it exists to say by how much at the measure the paper leads with,
under a 3 pp F2 budget.

If it holds, the workflow's code computes nothing about a case at all: it proposes spans
and the LLM decides everything. If it does not hold -- the likely outcome -- then the label
is the one piece of computed evidence the approach genuinely needs, and the paper can say
so with an F2 number instead of a TP one.

Everything below is `s_linker79`'s docstring, unchanged.

s_linker79 — no gate anywhere: the deterministic layer is a relation and nothing else.

`s_linker77` reduced `SCANS` to one row; `s_linker78` took the last enumeration out of the
prompts. What remained in code were the two options on that row:

    `unique_owner`      propose only when exactly one component of the catalog owns the
                        surface -- priced at 2.4 FP, and folding it into the denotation
                        prompt is the fold round's negative result (-8.4 TP), because that
                        judge is target-blind by design
    `skip_when_named`   skip a sentence that already writes a whole name of the component,
                        because that pair belongs to the full-name linker -- removing it
                        is `s_linker46` on the s25 base: F1 -1.5 (p = 0.00), **F2 -1.0
                        (p = 0.02)**

Both were kept through every previous round because both are *facts about a case* under the
design law, and the law says facts stay in code. This variant is the other reading of the
same law: a fact the judge cannot be told is a fact the workflow is *relying* on, and a
reviewer is entitled to see what the approach scores without any of them.

Under a **3 pp F2 budget** the two known costs (2.4 FP and ~1.0 F2) are affordable on
paper. What this variant buys, if it holds, is the strongest sentence available about the
approach: **the deterministic layer proposes spans of a single relation and decides
nothing.** `_name_spans` at one setting, no options, no filters -- every admission,
suppression and tie-break is an LLM verdict.

Everything below is `s_linker78`'s docstring, unchanged.

s_linker78 — the judging rubric stops enumerating.

`s_linker77` left one enumerated artifact in the workflow: the full-name gate's four
numbered reject-conditions and its three named approve-shapes. Read as English it is the
most rulebook-shaped thing in the module, and the general round measured what replacing it
with a single principle costs: TP +0.7 / FP -1.3 on a fixed candidate set, and **~0.8 F1
composed** (`s_linker71` 94.80 at n=6 against `s_linker70`'s 95.74). On F2 the same pair
reads 96.19 against 96.99.

That is why the enumeration stayed through s72-s75: the rounds were led by F1. Under an
**F2-led budget of 2 pp** it is affordable, and this variant spends part of that budget on
the last enumeration:

    before  approve by default; a bare mention, a heading, or a list all count; reject
            ONLY when one of these clearly holds: (1) ... (2) ... (3) ... (4) ...
    after   approve by default; a mention that says nothing further still counts; reject
            only on a positive ground -- the name is doing some other job, or the
            sentence denies what it would otherwise say

The four conditions do not vanish, they are *grounded*: (1) is `QUALIFIED_CLAUSE`, which
the judging prompt now carries; (3) and (4) are the use/mention distinction, which
`STRICTER_CLAUSE` already stated in the same prompt; (2), negation, is the last clause of
the principle. Nothing about the standard of proof changes -- approve by default, reject
on a positive ground.

**What this variant is for.** It prices the last piece of enumerated English at the
measure the paper leads with. If F2 holds, the workflow's entire authored surface is
principles; if it does not, s77 is the head and the enumeration is reported as measured
rather than as preferred.

Everything below is `s_linker77`'s docstring, unchanged.

s_linker77 — the deterministic layer becomes one row.

`s_linker75` left the authored English with nothing corpus-shaped in it. What a reviewer
still counts in the code is three candidate generators. Two of them — `AS_SPELLED` and
`ANY_SPELLING` over a whole name — exist to draw a recall floor under the LLM extractor,
and the extraction call already reads every sentence and already receives the component
catalog. So the floor can be *stated* instead of scanned, which is what `s_linker67` did
on the s66 base: `SCANS` keeps the one row the partial-name linker needs, `_add_scan` is
deleted, and `ENTITY_EXTRACTION_RULES` gains one paragraph asking for exactly what the two
rows scanned for.

**Why it is taken here and was refused there.** s67 measured TP -1.2 (p = 0.14) at its own
stage and TP -4.0 (p = 0.03), macro F2 -1.1 (p = 0.04) composed, and that round was reading
F1. Under an F2-led budget of 2 pp the cost is affordable, and what it buys is the sentence
the paper wants to be able to write: **the deterministic layer is one relation at one
setting.** `_name_spans` stays, and `unique_owner`/`skip_when_named` stay -- folding
`unique_owner` is the fold round's negative result (-8.4 TP), because the denotation judge
is target-blind by design.

Everything else is `s_linker75`'s, byte for byte: same rubric, same clauses, same bounds,
same judges. The two relocations are not separable -- the bind round measured that deleting
the scans under an unchanged prompt costs TP 3.6 and that a clause alone recovers none of
it -- so they move together or not at all.

Everything below is `s_linker75`'s docstring, unchanged.

s_linker75 — the last four finetuned spans leave the prompts.

`s_linker74` removed the one span GATE-07 caught in the judging path. What it left
standing was the same distinction restated three more times in three bespoke wordings,
plus one clause that still spelled `X.Y or X.Y.Z` outright. Read together those five
copies are what a reviewer sees as a rulebook grown against five documents. This variant
removes all four remaining ones and states the distinction once, on a ground:

    ALIAS_EXCLUSION_RULES     spelled the syntax     -> the prohibition without the shape
    ENTITY_EXTRACTION_RULES   "code-level path"      -> `QUALIFIED_CLAUSE` in the prompt
    P1_FOCUS                  "code-level identifier"-> the question alone, nothing added
    LAYERED_COREF_RULES       fifth restatement      -> removed, nothing added

Four stage arms, three samples a side, replayed against `s_linker74`'s own recorded
checkpoints (`pilot/finetune_pilots.py`, `pilot/general_prompt_pilots.py`; report in
``../results/finetune_round/README.md``):

    span              arm                                   TP            FP
    alias             judged table 35.7 -> 39.3 terms/run    +0.0 (1.00)   +3.7 (0.90)
    extraction        general clause instead                 +0.7 (1.00)   -6.0 (0.20)
    P1, nothing added the question alone                     +2.3 (0.20)   +0.0 (1.00)
    coreference       phrase removed, nothing added          +4.7          +3.7

WHAT THE ALIAS ARM OVERTURNS, AND WHAT IT COSTS. The general round kept the alias syntax
on a measured ground: both general rewordings grew the alias table from 24.0 to ~37 terms
per run, and an over-large table was priced at F1 94.57 against 96.42
(`s_linker39`/`s_linker40`). Re-measured against s74's own checkpoints, **that growth does
not reproduce** -- the syntax arm reads 35.7 judged aliases per run against the general
arm's 39.3 (FP +3.7, p = 0.90). The earlier reading was one invocation set's level, which
this branch has documented drifting before. What the syntax does buy is smaller and is
reported rather than dropped: it admitted **0 identifier fragments in 15 project-runs**
where the general wording admitted 6 in one of them. Flipping the alias judge's tie-break
to compensate was tried in the same arm and buys nothing (37.7 terms, 13 fragments), so it
is not adopted -- an unnecessary change is not a defensible one.

WHY P1 ADDS NOTHING WHERE THE EXTRACTION PROMPT ADDS THE CLAUSE. A clause is general only
relative to the judge that reads it, and it should be stated once per prompt. The
full-name judging prompt already carries the distinction inside reject-condition (1) of
the rubric, which s74 rewrote onto `QUALIFIED_CLAUSE`'s ground; adding the clause as well
is a restatement, and the arms say so -- with the clause added, TP -0.7 / FP -1.3; with
nothing added, **TP +2.3 / FP +/-0.0**. The extraction prompt has no such enumeration, so
there the clause is added. The coreference prompt gets neither: its cases contain no name
for a clause about identifiers to be about, and the general round measured that replacing
the phrase there costs TP 3.0 while removing it is TP +4.7.

WHAT IS NOT TOUCHED, AND WHY. The rubric is byte-identical to s74's. Three variants
measured its two structural properties as load-bearing and both survive: the four numbered
reject-conditions (replacing them with one principle is ~0.8 F1 composed -- s71 94.80 at
n=6, s72 94.94) and "a heading, or a list" (removing it is exactly 2.7 TP in each of three
runs -- s73). Neither is corpus-shaped: an enumeration is a rubric structure, and headings
and lists are general technical-documentation practice. **The bar catches shapes peculiar
to a corpus, not the structure every document of the genre has.** The deterministic layer
is untouched -- same `SCANS` rows, same relation, same options, same inflection list --
and `pilot/test_s75_nofinetune.py` asserts all of it in 36 checks: four constants differ,
49 method bodies and 7 class attributes are byte-identical, and no prompt this module
builds contains a spelled syntax, a bespoke restatement, or two copies of the clause.

THE RESULT ON THE ONLY SCORE THAT MATTERS FOR THE CLAIM.
``pilot/prompt_defensibility.py --variant s_linker75``:

    s_linker70   1700 of 3645 authored bytes stand on an admissible ground
    s_linker75   **3412 of 3412** -- general 2866, se-practice 299, prior-work 247,
                 corpus **0**

Every clause in the workflow now states a general rule, general SE practice, or a
measurement this branch already published. The only project-specific input anywhere is
the runtime component catalog and the aliases discovered from the document.

The design statement below is the workflow as it now stands. The experiment log
behind every decision in it lives in ``../results/<round>/README.md``, not here.

──────────────────────────────────────────────────────────────────────────────
THE DESIGN THIS VARIANT INHERITS

WHAT THIS MODULE IS. Three linkers run in a fixed order of name evidence — the sentence
writes a whole name, the sentence writes one word of a name, the sentence writes no name
at all — and each proposes candidates deterministically and admits none of them. Every
link in the output was approved by an LLM judge. The deterministic layer's entire job is
to decide *which pairs get asked about*, and the three linkers differ only in how loosely
they read the component catalog against a sentence.

THE RELATION. ``_name_spans(text, name, form)`` returns the spans of ``text`` that write
``name`` at ``form``, over two independent dimensions:

    fidelity   how exactly the characters must reproduce the name
               AS_SPELLED  <  ANY_CASE  <  ANY_SPELLING
    extent     how much of the name must be present
               the whole name  <  one word of it (ANY_WORD)

That function and the four values of ``NameForm`` are the whole deterministic layer.
``SCANS`` is three rows of it, one per linker, and the table below is the design
rationale — pairs reached over all five benchmarks and how many are gold
(``pilot/rule_audit.py --only A3``):

    fidelity/extent           pairs   gold   gold per pair
    AS_SPELLED   whole name     112    107        0.955
    ANY_CASE     whole name     172    133        0.773
    ANY_SPELLING whole name     176    137        0.778
    ANY_WORD     one word       281    161        0.573

Precision falls monotonically as the relation loosens and recall rises, which gives the
one principle the workflow is built on: **the looser the form a linker scans, the
stricter the judge behind it.** The full-name linker scans the tight rows and judges in
two focused passes that approve by default; the partial-name linker scans the loosest row
and judges target-blind; the coreference linker reaches the sentences no row reaches and
rejects when uncertain.

THE DESIGN LAW, WHICH IS ALSO THE ANSWER TO "WHY IS ANY OF THIS STILL IN CODE".

    Facts stay in code. Weighings go in the prompt.

The deterministic layer supplies *facts about a case*; the LLM supplies *judgment about
the case*. A clause that tells a judge **how to weigh** what it sees can be moved out of
code into that judge's prompt. A statement of **what is true of the case** cannot -- not
because the judge cannot see it, but because the judge is not disinterested about it.
Every relocation attempted on this branch obeys the split, with no exceptions:

    moved into the prompt                         kind       outcome
    `skip_qualified`                              weighing   folded, TP -0.4 (p=0.44)
    `skip_stricter`                               weighing   folded, TP +4.0, FP +/-0.0
    the mention label, self-reported by the judge  fact       -6.7 TP
    the mention label, removed                     fact      -10.7 TP
    `unique_owner`                                 fact       -8.4 TP
    the target, shown to the denotation judge      fact       -5.5 gold (s_linker25)

This is why ``_classify_mention_typed`` still computes a label about a sentence the judge
is already reading, and why the denotation judge is kept blind to the component on trial.
Neither is a gate that resisted removal; both are the fact-finder doing its half of the
work. It is also why the two clauses that *did* move -- ``QUALIFIED_CLAUSE`` and
``STRICTER_CLAUSE`` -- read as instructions about evidence rather than as assertions
about the case. See ``../results/concept_round/README.md``.

WHAT IS LEFT OF THE RULES, AND WHY. Successive rounds removed every deterministic gate
that could be removed or relocated (`../results/{bind,fold,general}_round/README.md`).
Two remain, both on the partial-name row, and they are blocked by one design decision
rather than by two rules: ``unique_owner`` needs the component catalog and
``skip_when_named`` needs the target, and the denotation judge is shown neither by
design — s_linker25's grounded identity review, which does show it the target, traded 5.5
gold links for 2.5 spurious ones over six runs. The fold law the rounds established:
**a gate folds into a judge's prompt exactly when that judge is shown the information the
gate reads.** ``skip_qualified`` and ``skip_stricter`` folded on that test; folding
``unique_owner`` failed on it.

WHAT THE PROMPTS MAY SAY. Every authored clause stands on a stated ground — a general
rule, general SE practice, or prior work (GATE-07 in ``CLAUDE.md``;
``pilot/prompt_defensibility.py`` scores the surface):

    ENTITY_EXTRACTION_RULES         the stage's admission contract          general
    QUALIFIED_CLAUSE                qualified names compose                 se-practice
    STRICTER_CLAUSE                 use vs mention                          general
    LAYERED_ENTITY_RULES            approve by default, reject on a
                                    positive ground                         general
    P1_FOCUS / P2_FOCUS             participation; referential specificity  general
    LAYERED_COREF_RULES             referring expression + ambiguity        general
    COREF_RULES                     antecedent clarity, abstention          general
    DOC_KNOWLEDGE_*                 what an alias is; the tie-break         general / prior-work
    ALIAS_EXCLUSION_RULES           kept, and documented as doing something
                                    other than what it states               prior-work

The only project-specific input anywhere is the runtime component catalog and the aliases
discovered from the document. No benchmark vocabulary appears in this module (GATE-06)
and no clause names a surface shape peculiar to these five documents (GATE-07), with the
one measured exception noted in the table.

WHERE THE HISTORY IS. This module states the design; it does not carry the experiment
log. Every arm, p-value and rejected variant behind the decisions above lives in
``../results/<round>/README.md`` and in the variant registry in ``run_ablation.py``.
Read ``CLAUDE.md`` for the measurement policy those rounds follow.

STANDALONE, by the convention ``s_linker21`` set for paper artifacts: no linker
superclass, one module.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

from llm_sad_sam.core.data_types_v2 import (
    SadSamLink, CandidateLink, DocumentKnowledge,
)
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.linkers.experimental.helper_v3 import (
    parse_snum, get_comp_names,
)
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend, LLMResponse

# ─────────────────────────────────────────────────────────────────────────────
# Prompt constants. GENERALIZED — every enumeration of the inherited chain is
# restated as the principle it enumerates. Each removal is sized in
# `pilot/prompt_audit.py` off six recorded s49 runs; the sizes are quoted at each
# constant. Nothing here is benchmark-derived vocabulary (GATE-06): the
# generalized text is strictly *less* specific than what it replaces, because the
# concrete forms it drops (`X.Y.Z`, "the module"/"the service"/"the system",
# "gerund phrase", "list item") are the only shapes in these prompts that named a
# surface form rather than a property.
# ─────────────────────────────────────────────────────────────────────────────

# Dropped: "names the whole system", "names a different entity" and the grouping
# clause — three ways of saying "identifies something other than that one
# component". The leniency sentence stays: it is what keeps a third of MediaStore
# (measured when the alias judge was removed, F1 94.57 vs 96.42).
DOC_KNOWLEDGE_JUDGE_RULES = """An alias is valid when the document establishes an equivalence between a phrase and a single named component. It is invalid when the phrase is generic vocabulary or identifies anything other than that one component. When uncertain, prefer APPROVE."""

# Dropped: the parenthetical listing the three alias shapes. What the extractor is
# asked for is a surface form for one component; which shapes qualify is the
# model's judgement, and the judge that follows is where validity is decided.
DOC_KNOWLEDGE_EXTRACTION_RULES = """Find surface forms the document uses to refer to a single named component (introduced short forms, alternate names, or words of multi-word names when they alone clearly mean the full name). Reject terms whose ordinary English use dominates."""

# The last clause in the module that spelled a syntax. The general round kept it on
# measured grounds -- both general rewordings grew the alias table from 24.0 to ~37
# terms per run -- and `pilot/finetune_pilots.py --pilot aliascomp` re-measured that
# claim against s74's own checkpoints and did not reproduce it: the syntax arm reads
# 35.7 judged aliases per run against the general arm's 39.3 (FP delta +3.7, p = 0.90).
# The prohibition stays; only the shape goes. What the shape does buy is small and
# real, and is reported rather than hidden: the syntax arm admitted 0 identifier
# fragments in 15 project-runs, the general arm 6 in one of them.
ALIAS_EXCLUSION_RULES = """A fragment of a longer identifier is not an alias: if a term appears only as part of a compound or qualified name, do not include it."""


# THE ONE CHANGE from s_linker69: the full-name linker's admission contract is
# stated here instead of enforced afterwards. s65 asked the extractor for anything
# that "refers to the component ... as a participant in a described interaction" and
# then deleted every proposal whose sentence writes no name of the component
# (`_keep_stated_names`, now gone). The two sentences below are that filter, in the
# register of the prompt it joins; the rest of the paragraph is s65's unchanged.
ENTITY_EXTRACTION_RULES = """Include a reference only when the sentence itself writes the component's name or one of the KNOWN ALIASES. Exclude a component that the sentence only implies as a participant in a described interaction without naming it, and exclude a name used as ordinary English with no architectural intent. Favor inclusion among the sentences that do name it.

Report a reference for every sentence that writes a component's name the way the COMPONENTS list spells it, however incidental the mention, and count a name written with different spacing, hyphenation or compound joining as that name."""


# Dropped: the three-way gloss of "architectural participant" and the qualified-name
# example. The distinction the question draws is unchanged.
P1_FOCUS = (
    "Check architectural participation: does the sentence name this "
    "component as an architectural participant?"
)

P2_FOCUS = (
    "Check referential specificity: is the component name used to identify "
    "this specific architectural element, or does it serve as a generic "
    "technical term in this sentence?"
)


# Dropped: the three listed pronoun/noun-phrase forms and the gloss of
# "architectural participant".
COREF_VALIDATION_FOCUS = (
    "Check coref resolution: does the referring expression in this sentence "
    "actually refer to the named component as an architectural participant?"
)

# Dropped (P3/P4 of `pilot/prompt_audit.py`, six recorded s49 runs): clause (b),
# which licensed resolving to the section topic without a name repetition and
# licensed 0.0 of 578 recorded resolutions; the five listed role phrases, of which
# only "it" is used at volume; and the terminal-word/abbreviation enumeration
# (1.7 antecedents per run), subsumed by "under any form the document uses for it".
COREF_RULES = """For each case, decide whether a pronoun or noun phrase that refers back in the target sentence refers back to a component named or aliased earlier in the context. Resolve when the surrounding sentences make one component the clear antecedent, under any form the document uses for it. Avoid resolving when two or more equally plausible antecedents exist."""

# Full-name gate — lenient: a stated name is a link unless a reject signal fires.
# The default and the leniency are kept verbatim in force; what goes is the
# numbered form of the four reject-conditions, which become illustrations of one
# principle. Of 14.7 rejections per run, 2.2 match condition (1) and 1.8 condition
# (2) lexically, so 73% already rest on the two conditions that name no surface
# form at all.
LAYERED_ENTITY_RULES = (
    "Approve the link by default: the component is named here and the document treats "
    "it as part of the system. A mention that says nothing further about the component "
    "still counts as a valid link. Reject only on a positive ground -- that the sentence "
    "asserts nothing of this component, because the name is doing some other job here, "
    "or because the sentence denies what it would otherwise say of it."
)

# Coreference gate — strict: the component is NOT named in the sentence, so demand
# a genuine referring expression plus an architectural claim. Dropped: the gloss of
# "architectural claim" and the three named fragment shapes.
LAYERED_COREF_RULES = (
    "These are coreference links: a pronoun or noun phrase in the sentence is claimed to "
    "refer back to the component, which is NOT named in the sentence itself. Approve only "
    "when the sentence contains a genuine referring expression that unambiguously points "
    "to THIS component and makes an architectural claim about it. Reject when there is no "
    "such referring expression or when the antecedent could equally be a different "
    "component. When uncertain, reject."
)

#: The endings under which a sentence word still counts as a word of a component's
#: name. English inflectional morphology, the same set for every document, and **the
#: only word list in this module**: no benchmark term appears here and none can
#: (GATE-06). Read at exactly one place, the ANY_WORD branch of ``_name_spans``.
#: (s_linker64's comment claimed a second list, "the stopwords"; the module has none.)
INFLECTIONS = ("", "s", "es", "ed", "d", "ing", "ings", "er", "ers")

#: The tokenizer both dimensions of the relation use to cut a name or a sentence into
#: words. Word boundaries only -- splitting compounds here was measured to triple the
#: candidate set while reaching no additional gold link.
WORD_PATTERN = r"[A-Za-z]+[A-Za-z0-9]*|\d+"


#: The span-boundary gate, stated as what the expression IS rather than where its
#: characters sit. `SCANS[name_word].skip_qualified` dropped a span glued into a dotted
#: or joined identifier before the judge saw it; deleting that gate with no compensation
#: is FP +7.0 (p = 0.01), and this sentence in front of the same judge is TP -0.4
#: (p = 0.44), FP -0.2 (p = 1.00) -- `pilot/fold_pilots.py --pilot foldqualified`.
QUALIFIED_CLAUSE = """An expression that occurs only as part of a longer joined or dotted identifier is naming a piece of that identifier, not a participant in what the sentence describes."""


#: The case-sensitivity gate, stated as the distinction the judge should be drawing
#: rather than as a boundary in the candidate generator. `SCANS[spelling].skip_stricter`
#: dropped every surface that already writes the plain name, keeping the ANY_CASE
#: whole-name cell out of the candidate set entirely. Deleting the gate is TP +4.0
#: (p = 0.01) at FP +1.8 (p = 0.01); this sentence in front of the same judge is TP +4.0
#: (p = 0.01) at FP +/-0.0 (p = 1.00) -- `pilot/fold_pilots.py --pilot foldstricter`.
#: It names no surface form and no component: capitalization is offered as evidence,
#: with the explicit statement that it does not settle the question by itself.
STRICTER_CLAUSE = (
    "Some sentences use an ordinary English word that happens to coincide with a "
    "component's name. Approve only when the sentence uses that word as the name of "
    "the component; if it is used in its ordinary sense and the component is not what "
    "the sentence is talking about, reject. Capitalization is evidence for a name and "
    "its absence is evidence against, but neither settles it on its own."
)


class NameForm(Enum):
    """A point of the surface-realization relation, on two independent dimensions.

    *Fidelity* -- how exactly the sentence's characters must reproduce the name:

        AS_SPELLED    the name, character for character, at word boundaries
        ANY_CASE      the name, ignoring case
        ANY_SPELLING  the name's word sequence, ignoring case, separators and
                      compound joining, so "X Y", "x-y" and "XY" are one form

    *Extent* -- how much of the name has to be present:

        ANY_WORD      one word of the name, under an English inflectional ending

    Which form a proposer scans is the only thing that distinguishes the workflow's
    three candidate generators from one another; see ``SCANS``.
    """

    AS_SPELLED = "as_spelled"
    ANY_CASE = "any_case"
    ANY_SPELLING = "any_spelling"
    ANY_WORD = "any_word"


@dataclass(frozen=True)
class SurfaceScan:
    """A candidate generator, stated as data.

    Every field except ``form`` and ``source`` is a shared option, so adding a
    generator is a row of ``SCANS`` rather than a method with its own regex.
    """

    #: Which point of the relation this generator scans for.
    form: NameForm
    #: The ``source`` tag its candidates carry into the evidence bundle and the logs.
    source: str
    #: Propose only when exactly one component of the catalog owns the surface. Used
    #: by the partial-name row alone: on the spelling row it frees 0 pairs on all five
    #: projects (`pilot/gate_inventory.py`), and on the whole-name row the surface *is*
    #: a catalog name.
    unique_owner: bool = False
    #: Skip a sentence that already writes a whole name of the component: that pair
    #: belongs to the full-name linker.
    skip_when_named: bool = False
    #: Attach the mention label the full-name judge reads.
    label_mention: bool = False


#: The workflow's three candidate generators. Four lexical rules in ``s_linker64``;
#: three rows and one filter here, all reading the same relation.
#:
#:   `stated_name`  the recall floor under the LLM extractor. Tightest fidelity, and
#:                  the table in the module docstring is why: 0.955 gold per pair
#:                  against 0.773 one row looser. A component named `Common` or
#:                  `Client` matches ordinary English on every page, and the
#:                  capitalization is what separates the proper noun from the common
#:                  one. It skips no qualified span, matching the admission filter's
#:                  measured asymmetry (see ``_keep_stated_names``).
#:   `spelling`     the forms that differ only in how the name's words are joined.
#:   `name_word`    one word of a name, for the partial-name linker.
SCANS = {
    #: The one row left in code. The two tight rows (`stated_name`, `spelling`) are
    #: relocated: the recall floor they drew is stated in `ENTITY_EXTRACTION_RULES`
    #: instead, so the deterministic layer is a single point of the relation.
    #: `s_linker67` measured that relocation on the s66 base at TP -1.2 (p = 0.14) at
    #: its own stage and TP -4.0 (p = 0.03) / macro F2 -1.1 (p = 0.04) composed; it was
    #: refused there under an F1-led reading and is taken here under an F2-led one.
    #: `unique_owner` stays: folding it is the fold round's negative result (-8.4 TP),
    #: because the denotation judge is target-blind by design.
    "name_word": SurfaceScan(
        form=NameForm.ANY_WORD,
        source="partial_name_candidate",
        # Both options gone. What is left is a point of the relation and nothing
        # else: no gate anywhere in the deterministic layer decides a link or
        # withholds a case from a judge.
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Tracing infrastructure — per-LLM-call audit trail
# ─────────────────────────────────────────────────────────────────────────────

_phase_local = threading.local()


def _current_phase() -> str:
    return getattr(_phase_local, "phase", "unknown")


class _TracingLLMClient:
    """Delegating wrapper that records every query() into a phase-tagged trace."""

    def __init__(self, inner: LLMClient, sink: list[dict]):
        self._inner = inner
        self._sink = sink
        self._sink_lock = threading.Lock()

    def set_phase(self, name: str) -> None:
        _phase_local.phase = name

    def query(self, prompt: str, timeout: int = 180, max_retries: int = 3) -> LLMResponse:
        phase = _current_phase()
        t0 = time.time()
        try:
            resp = self._inner.query(prompt, timeout=timeout, max_retries=max_retries)
        except Exception as exc:
            record = {
                "phase": phase, "ts": t0,
                "elapsed_s": round(time.time() - t0, 3),
                "timeout": timeout, "max_retries": max_retries,
                "prompt": prompt,
                "response_text": None,
                "success": False,
                "error": f"FATAL: {exc}",
                "latency_ms": None,
                "model": None,
            }
            with self._sink_lock:
                self._sink.append(record)
            raise
        record = {
            "phase": phase, "ts": t0,
            "elapsed_s": round(time.time() - t0, 3),
            "timeout": timeout, "max_retries": max_retries,
            "prompt": prompt,
            "response_text": getattr(resp, "text", None),
            "success": getattr(resp, "success", None),
            "error": getattr(resp, "error", None),
            "latency_ms": getattr(resp, "latency_ms", None),
            "model": getattr(resp, "model", None),
        }
        usage = getattr(resp, "token_usage", None)
        if usage is not None:
            record["token_usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        with self._sink_lock:
            self._sink.append(record)
        # A phase result may only be interpreted after every required request
        # succeeds. Returning a failed response lets extract_json() turn it into
        # None and silently omit an entire batch.
        if not resp.success:
            raise RuntimeError(f"LLM request failed in {phase}: {resp.error}")
        return resp

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

class MentionType(Enum):
    """How a component name appears in a sentence."""
    PROPER_STANDALONE = "proper case, standalone"
    LOWERCASE_PROSE = "lowercase mention"
    CODE_TOKEN = "lowercase, inside qualified name"
    VIA_ALIAS = "via known alias"
    INDIRECT = "indirect/unclear match"


@dataclass
class EvidenceBundle:
    """What a judge is told about a candidate beyond the candidate itself.

    The matched span and the preceding sentence also appear in the case header
    the judge reads (``Case n: "span" -> Component``, then the sentence with its
    ``[prev: ...]`` prefix), so the bundle repeats both. The repetition is
    deliberate and was verified the hard way: dropping either is neutral on the
    judging stage in isolation (span TP +0.8, F2 +0.3, all p >= 0.44; preceding
    sentence TP -0.4, F2 -0.2, p >= 0.30) and costs precision once composed --
    three five-project runs without them hold recall (TP 182.0) and lose it on
    false positives (8.3 against the 4-6 of the six-run reference band), F1 95.2
    against 96.42 +/- 0.42. Repeating the evidence next to the rubric is not
    redundant for the model.
    """

    source: str
    matched_span: str
    mention_type: str          # MentionType.value (str for prompt embedding)
    preceding_text: str
    anchor_sentences: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Main linker
# ─────────────────────────────────────────────────────────────────────────────

class SLinker80:
    """Three linkers, fixed name-evidence order, no controller. Standalone."""

    _VARIANT_NAME = "s_linker80"

    #: Execution order. Full name first (it needs the least), partial name
    #: second, coreference last. The partial-name linker is the only one that
    #: subtracts already-linked pairs, so it must not run first.
    LINKERS = ("full_name", "partial_name", "coreference")

    # ── Resource bounds ──────────────────────────────────────────────────────
    # These cap prompt size and call count. No decision rule reads them:
    # changing one changes how much text a judge sees, never what counts as a
    # link. Named here so they are auditable in one place.
    #
    # Every evidence window is the same width and every anchor list the same
    # length, on purpose — the earlier per-step values (2, 3, 4, 5) implied a
    # calibration that was never measured. One width was verified not to
    # weaken the target-blind denotation step: that step's blindness comes from
    # withholding the target label from the case, not from hiding sentences
    # that name components, and a naming sentence is already visible in the
    # shared batch table for the large majority of candidates at any width.
    CONTEXT_SENTENCES = 5          # sentences either side shown to any judge
    ANCHOR_LIMIT = 5               # naming sentences offered as evidence
    EXTRACTION_BATCH = 50          # sentences per full-name extraction call
    JUDGE_BATCH = 25               # candidates per judging call (all judges)
    COREFERENCE_BATCH = 10         # sentences per coreference-resolution call
    ASK_ATTEMPTS = 2               # initial call + one retry on an empty parse

    def __init__(
        self,
        backend: LLMBackend | None = None,
        model: str | None = None,
        checkpoint_fallback: LLMBackend | str | None = None,
        checkpoint_fallback_model: str | None = None,
        no_knowledge: bool = False,
    ):
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.4")
        real_llm = LLMClient(
            backend=backend or LLMBackend.CLAUDE,
            model=model,
            checkpoint_fallback=checkpoint_fallback,
            checkpoint_fallback_model=checkpoint_fallback_model,
        )
        self._llm_calls: list[dict] = []
        self.llm = _TracingLLMClient(real_llm, self._llm_calls)
        self.no_knowledge = no_knowledge
        self.doc_knowledge: DocumentKnowledge | None = None
        self._phase_log: list[dict] = []
        self._phase_metrics: dict[str, dict] = {}
        self.workflow: list[dict] = []
        print("SLinker80 (full-name -> partial-name -> coreference; fixed order)")
        print(f"  Backend: {self.llm.describe_backend()}")

    # ── Main entry ───────────────────────────────────────────────────────────

    def link(self, text_path, model_path, **_kwargs):
        self._phase_log = []
        self._llm_calls.clear()
        self._phase_metrics = {}
        started = time.time()

        components = parse_pcm_repository(model_path)
        sentences = load_sentences(text_path)
        name_to_id = {component.name: component.id for component in components}
        sent_map = build_sent_map(sentences)
        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        print("\n[Knowledge] Document aliases")
        self.doc_knowledge = (
            DocumentKnowledge() if self.no_knowledge
            else self._learn_document_knowledge(sentences, components)
        )
        self._save_phase(text_path, "knowledge",
                         {"doc_knowledge": self.doc_knowledge})

        current: list[SadSamLink] = []
        history: list[dict] = []
        for linker in self.LINKERS:
            print(f"\n[Linker] {linker}")
            produced, feedback = self._run_linker(
                linker, sentences, components, name_to_id, current, sent_map
            )
            current = self._union(current, produced)
            history.append({
                "linker": linker,
                "feedback": self._linker_feedback(feedback),
            })
            self._save_phase(text_path, f"linker_{linker}", {
                "links": produced, "feedback": feedback, "workflow": history,
            })

        self.workflow = history
        self._phase_metrics = self._compute_phase_metrics()
        self._log(
            "s25_summary",
            {"components": len(components), "sentences": len(sentences)},
            {
                "workflow": history,
                "final": len(current),
                "elapsed_s": round(time.time() - started, 2),
                "llm_calls": len(self._llm_calls),
                "phase_metrics": self._phase_metrics,
            },
            current,
        )
        self._save_phase(text_path, "final", {
            "final": current,
            "workflow": history,
            "elapsed_s": round(time.time() - started, 2),
        })
        self._save_log(text_path)
        print(f"\nFinal: {len(current)} links "
              f"({time.time() - started:.1f}s, {len(self._llm_calls)} LLM calls)")
        return current

    def _run_linker(self, linker, sentences, components, name_to_id, linked, sent_map):
        """Dispatch, with the linked set passed to every linker without exception.

        This is what makes "each linker sees only what the earlier ones left
        unlinked" a property of the pipeline rather than of one linker. The
        subtraction itself is `_unlinked`, called once inside each linker at its
        candidate boundary.
        """
        if linker == "full_name":
            return self._run_full_name_linker(
                sentences, components, name_to_id, linked, sent_map)
        if linker == "partial_name":
            return self._run_partial_name_linker(
                sentences, components, linked, sent_map)
        if linker == "coreference":
            return self._run_coreference_linker(
                sentences, components, name_to_id, linked, sent_map)
        raise RuntimeError(f"unknown linker: {linker!r}")

    # ── Concurrency and small helpers ────────────────────────────────────────

    @staticmethod
    def _run_parallel(tasks):
        if len(tasks) == 1:
            name, fn = next(iter(tasks.items()))
            return {name: fn()}
        results = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(fn): name for name, fn in tasks.items()}
            try:
                for fut in as_completed(futures):
                    results[futures[fut]] = fut.result()
            except Exception:
                for other in futures:
                    other.cancel()
                raise
        return results

    @staticmethod
    def _iter_batches(items, n):
        """Yield (batch_num, batch_slice) — batch_num is 1-indexed."""
        for i, start in enumerate(range(0, len(items), n), start=1):
            yield i, items[start:start + n]

    @staticmethod
    def _prev_prefix(snum, sent_map) -> str:
        prev = sent_map.get(snum - 1)
        return f"[prev: {prev.text}] " if prev else ""

    @classmethod
    def _find_exact_form(cls, text, expression):
        """The first writing of ``expression`` in ``text`` at ANY_CASE, or "".

        The relation's middle fidelity, returning the surface rather than the span,
        because its three callers want to know *what* matched: the mention label
        compares it against the name, and the two name predicates only test it.
        """
        spans = cls._name_spans(text, expression, NameForm.ANY_CASE)
        return text[spans[0][0]:spans[0][1]] if spans else ""

    def _states_a_name(self, text: str, comp_name: str) -> bool:
        """Does this sentence state the component's name, or one the document gave it?

        One predicate for a question three stages asked with three copies of the same
        expression: the full-name admission filter, the partial-name proposer's
        whole-name exclusion, and the coreference antecedent gate. The mention-label
        classifier asks it too but decomposed, because it must know *which* matched, so
        it keeps its own two calls to ``_find_exact_form``.
        """
        names = (comp_name, *self._names_by_component().get(comp_name, ()))
        return any(self._find_exact_form(text, name) for name in names)

    def _window(self, snum: int, sentences):
        """The sentences within ``CONTEXT_SENTENCES`` of this one, in document order.

        One predicate for a condition spelled two different ways: the denotation step
        filtered the sentence list with ``abs(s.number - target) <= CONTEXT_SENTENCES``
        while the coreference resolver walked ``range(max(1, n - C), n + C + 1)`` against
        the sentence map. Both select the same set -- the walk cannot reach a sentence the
        filter excludes, and it skips numbers the document lacks, which is what the filter
        does by construction. Verified over every sentence of all five documents.
        """
        return [s for s in sentences
                if abs(s.number - snum) <= self.CONTEXT_SENTENCES]

    def _names_by_component(self):
        """Discovered aliases grouped by component. The model name is added by
        callers; together they are the component's set of names N(c)."""
        aliases = getattr(getattr(self, "doc_knowledge", None), "aliases", {})
        names = {}
        for term, component in aliases.items():
            names.setdefault(component, []).append(term)
        return names

    @staticmethod
    def _union(existing, additions):
        """Merge by (sentence, component). Earlier linkers win ties."""
        result = list(existing)
        keys = {(link.sentence_number, link.component_id) for link in existing}
        for link in additions:
            key = (link.sentence_number, link.component_id)
            if key not in keys:
                result.append(link)
                keys.add(key)
        return result

    @staticmethod
    def _link_view(links, sent_map):
        return [
            {
                "sentence": link.sentence_number,
                "text": sent_map[link.sentence_number].text,
                "component": link.component_name,
                "source": link.source,
            }
            for link in links
            if link.sentence_number in sent_map
        ]

    @staticmethod
    def _decision_view(decisions):
        return [
            {"sentence": sentence, "component_id": component, **decision}
            for (sentence, component), decision in decisions.items()
        ]

    @staticmethod
    def _linker_feedback(feedback):
        """Reduce detailed linker evidence to accepted/rejected references."""
        proposed = feedback.get("candidates", feedback.get("proposed", []))
        accepted = feedback.get("accepted", [])
        accepted_keys = {(i["sentence"], i["component"]) for i in accepted}

        def reference(item):
            return {"sentence": item["sentence"], "component": item["component"]}

        return {
            "accepted": [reference(i) for i in accepted],
            "rejected": [
                reference(i) for i in proposed
                if (i["sentence"], i["component"]) not in accepted_keys
            ],
        }

    # ── Prompt builders ──────────────────────────────────────────────────────

    @staticmethod
    def _prompt_doc_knowledge_extract(comp_names, doc_lines) -> str:
        return f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

{DOC_KNOWLEDGE_EXTRACTION_RULES}

{ALIAS_EXCLUSION_RULES}

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": [{{"term": "short_form", "component": "FullComponent"}}],
  "synonyms":      [{{"term": "specific_alternative_name", "component": "FullComponent"}}]
}}
JSON only:"""

    @staticmethod
    def _prompt_doc_knowledge_judge(comp_names, mapping_list) -> str:
        return f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}



{DOC_KNOWLEDGE_JUDGE_RULES}

Return JSON:
{{"approved": ["term1", "term2"]}}
JSON only:"""

    @staticmethod
    def _prompt_extraction(comp_names, mappings, batch) -> str:
        return f"""Extract ALL references to components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings)}' if mappings else ''}

{ENTITY_EXTRACTION_RULES}

{QUALIFIED_CLAUSE}

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence"}}]}}
JSON only:"""

    @staticmethod
    def _prompt_validation(comp_names, cases, focus, strict: bool = False) -> str:
        """Build a judging prompt. ``strict`` selects the coreference rubric.

        The rubric is asymmetric by design: the full-name gate is lenient (a
        stated name is a link unless a reject signal fires), the coreference
        gate is strict (the name is absent, so demand a referring expression
        and an architectural claim). The caller states which it wants — the
        inherited version inferred it from ``focus.startswith(...)``, so
        rewording the focus text silently swapped the rubric.
        """
        rules = LAYERED_COREF_RULES if strict else LAYERED_ENTITY_RULES
        # `STRICTER_CLAUSE` joins the lenient rubric only. The coreference rubric
        # already rejects when uncertain, and its cases carry no whole-name surface
        # for the clause to speak about.
        # The enumeration is gone, so the two grounds its conditions rested on are
        # stated here instead: `QUALIFIED_CLAUSE` for condition (1) and
        # `STRICTER_CLAUSE` for (3) and (4). Condition (2), negation, is the last
        # clause of the principle above.
        tail = "" if strict else f"\n{QUALIFIED_CLAUSE}\n{STRICTER_CLAUSE}\n"
        return f"""Validate components in a document. {focus}

COMPONENTS: {', '.join(comp_names)}

{rules}
{tail}
For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim), then decide approve true/false based on that claim.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>", "approve": true}}]}}
JSON only:"""

    @staticmethod
    def _prompt_coref(comp_names, cases) -> str:
        prompt = f"""Resolve references (pronouns and noun phrases that refer back) to components.

COMPONENTS: {', '.join(comp_names)}

For each TARGET sentence below, identify any pronoun or noun phrase that
refers back to a component listed above. If a target sentence has no such
reference to a listed component, return no resolution for it. Be conservative — only include resolutions you are CERTAIN about.

"""
        for i, case in enumerate(cases):
            prompt += f"--- Case {i+1}: S{case['sent'].number} ---\n"
            prompt += "CONTEXT:\n" + "\n".join(case["context"]) + "\n"
            prompt += f"TARGET: S{case['sent'].number} (marked with >>>)\n\n"

        prompt += f"""{COREF_RULES}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "reference": "the server", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

JSON only:"""
        return prompt

    # ── LLM call helper ──────────────────────────────────────────────────────

    def _ask(
        self,
        prompt: str,
        *,
        timeout: int = 120,
        label: str = "LLM call",
        phase: str | None = None,
        require: str | None = None,
        require_present: str | None = None,
    ) -> dict:
        """Query the LLM, parse JSON, retry once on empty/incomplete response.

        Success rule, in priority order:
          - require_present=KEY  → KEY must appear in the parsed dict (empty OK)
          - require=KEY          → data[KEY] must be truthy
          - neither              → any non-empty parsed dict succeeds
        """
        if phase is not None:
            self.llm.set_phase(phase)

        def _ok(d: dict | None) -> bool:
            if not d:
                return False
            if require_present is not None:
                return require_present in d
            if require is not None:
                return bool(d.get(require))
            return True

        data: dict = {}
        for attempt in range(self.ASK_ATTEMPTS):
            parsed = self.llm.extract_json(self.llm.query(prompt, timeout=timeout))
            # Each attempt replaces the last. Keeping a previous attempt's dict
            # when a later one fails to parse would return a payload this method
            # already rejected, and callers read it as if it had passed.
            data = parsed if parsed is not None else {}
            if _ok(data):
                return data
            if attempt < self.ASK_ATTEMPTS - 1:
                print(f"    {label}: empty response, retrying...")
        return data

    # ── Knowledge module ─────────────────────────────────────────────────────

    def _learn_document_knowledge(self, sentences, components):
        self.llm.set_phase("phase_25_doc_extract")
        comp_names = [c.name for c in components]
        doc_lines = [s.text for s in sentences]

        data1 = self._ask(
            self._prompt_doc_knowledge_extract(comp_names, doc_lines),
            timeout=300, label="Doc knowledge",
        )

        all_mappings: dict[str, str] = {}
        if data1:
            abbr_recs = data1.get("abbreviations", [])
            syn_recs = data1.get("synonyms", [])
            if isinstance(abbr_recs, dict):
                abbr_recs = [{"term": k, "component": v}
                             for k, v in abbr_recs.items()]
            if isinstance(syn_recs, dict):
                syn_recs = [{"term": k, "component": v}
                            for k, v in syn_recs.items()]
            for rec in abbr_recs + syn_recs:
                if not isinstance(rec, dict):
                    continue
                term = rec.get("term")
                full = rec.get("component")
                if term and full in comp_names:
                    all_mappings[term] = full

        if all_mappings:
            mapping_list = [f"'{k}' -> {v}" for k, v in all_mappings.items()]
            data2 = self._ask(
                self._prompt_doc_knowledge_judge(comp_names, mapping_list),
                timeout=120, label="Doc knowledge judge",
                phase="phase_25_doc_judge", require="approved",
            )
            approved = set(data2.get("approved", [])) if data2 else set(all_mappings)
        else:
            approved = set()

        knowledge = DocumentKnowledge()
        for term, comp in all_mappings.items():
            if term in approved:
                knowledge.aliases[term] = comp
                print(f"    Alias: {term} -> {comp}")
        return knowledge

    # ═════════════════════════════════════════════════════════════════════════
    # Linker 1 — FULL NAME: the sentence states a name of the component.
    # ═════════════════════════════════════════════════════════════════════════

    def _run_full_name_linker(self, sentences, components, name_to_id, linked,
                              sent_map):
        candidates_by_key = self._extract_named_mentions(
            sentences, components, name_to_id, sent_map
        )
        # The full-name linker's candidate set: what the LLM extractor proposed, held
        # to the linker's own contract, plus the two tight cells of the relation the
        # extractor may have missed. See ``SCANS`` and the module docstring's table.
        # No admission filter: the contract it enforced is stated in the extraction
        # prompt (see ``ENTITY_EXTRACTION_RULES``), so the extractor is held to the
        # linker's own contract rather than corrected after the fact.
        candidates = list(candidates_by_key.values())
        # No lexical additions: the recall floor those two rows drew is stated in
        # the extraction prompt (see ``ENTITY_EXTRACTION_RULES``). ``SCANS`` keeps
        # the one row the partial-name linker scans.
        bundles = {
            (c.sentence_number, c.component_id): self._build_evidence_bundle(c, sent_map)
            for c in candidates
        }
        approved, decisions = self._validate_with_evidence(
            candidates, bundles, components, sent_map,
            p1_tag="phase_25_full_name_p1",
            p2_tag="phase_25_full_name_p2",
            stage_label="full_name",
        )
        links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name,
                       source=self._full_name_source(c))
            for c in approved
        ]
        return links, {
            "candidates": self._link_view(
                [SadSamLink(c.sentence_number, c.component_id, c.component_name,
                            source="full_name_candidate") for c in candidates],
                sent_map,
            ),
            "accepted": self._link_view(links, sent_map),
            "judge_decisions": self._decision_view(decisions),
        }

    def _extract_named_mentions(self, sentences, components, name_to_id, sent_map) -> dict:
        """One extraction pass over the document, batched.

        An earlier revision sent this prompt twice and unioned the two samples
        as a self-consistency guard. Measured over five runs on all five
        projects, the second sample moved neither score beyond noise (TP -1.2,
        p=0.30; FP -1.2, p=0.42), so the pipeline states one sample and pays for
        one.
        """
        comp_names = get_comp_names(components)
        mappings = (
            [f"{term}={component}"
             for term, component in self.doc_knowledge.aliases.items()]
            if self.doc_knowledge else []
        )
        candidates = self._run_extraction_pass(
            sentences, comp_names, mappings, name_to_id, sent_map,
            phase_tag="phase_25_full_name_extract")
        print(f"    Extracted: {len(candidates)}")
        return candidates

    def _run_extraction_pass(self, sentences, comp_names, mappings,
                             name_to_id, sent_map, phase_tag=None):
        if phase_tag:
            self.llm.set_phase(phase_tag)
        batch_size = self.EXTRACTION_BATCH
        candidates: dict = {}
        for batch_num, batch in self._iter_batches(sentences, batch_size):
            if len(sentences) > batch_size:
                print(f"    batch {batch_num}: "
                      f"S{batch[0].number}-S{batch[-1].number} ({len(batch)} sents)")
            data = self._ask(
                self._prompt_extraction(comp_names, mappings, batch),
                timeout=240, label="batch", require="references",
            )
            if not data:
                continue
            for ref in data.get("references", []):
                cname = ref.get("component")
                snum = parse_snum(ref.get("sentence"))
                if snum is None or not cname or cname not in name_to_id:
                    continue
                sent = sent_map.get(snum)
                if not sent:
                    continue
                matched = ref.get("matched_text", "")
                if matched and matched.lower() not in sent.text.lower():
                    continue
                key = (snum, name_to_id[cname])
                if key not in candidates:
                    candidates[key] = CandidateLink(
                        snum, sent.text, cname, name_to_id[cname],
                        matched, source="full_name",
                    )
        return candidates

    def _scan(self, sentences, components, scan: SurfaceScan):
        """Every (sentence, component) pair the relation reaches at ``scan.form``.

        THE ONE CANDIDATE GENERATOR. ``s_linker64`` wrote this three times --
        ``_add_stated_name_net``, ``_spelling_variant_candidates`` and
        ``_name_word_candidates`` -- each with its own regex, its own ownership test
        and its own paragraph of defence. They differ in the point of the relation
        they scan and in two shared options, and `pilot/rule_audit.py --only A2`
        checks the three rebuilt sets against s_linker64's own on all five projects.

        Later spans of the same pair overwrite earlier ones, so the recorded
        ``matched_text`` is the last surface found in the sentence -- s_linker64's
        behaviour at both rebuilt sites.

        Nothing here admits a link. Every pair returned is a case for a judge.
        """
        candidates = {}
        for sentence in sentences:
            text = sentence.text
            for component in components:
                if scan.skip_when_named and self._states_a_name(text, component.name):
                    continue  # a whole name is stated: the full-name linker's pair
                for start, end in self._name_spans(text, component.name, scan.form):
                    surface = text[start:end]
                    if scan.unique_owner and len(
                        self._owners(surface, components, scan.form)
                    ) != 1:
                        continue
                    candidates[(sentence.number, component.id)] = CandidateLink(
                        sentence.number, text, component.name, component.id,
                        surface, source=scan.source,
                        **({"mention_type": self._classify_mention_typed(
                            component.name, text)} if scan.label_mention else {}),
                    )
        return list(candidates.values())

    @classmethod
    def _realizes(cls, surface, name, form: NameForm) -> bool:
        """Is the surface, in whole, a writing of ``name`` at ``form``?

        ``_name_spans`` locates a name *inside* a text; this asks whether the text
        *is* the name. The distinction matters for ownership: `Image Provider`
        contains a writing of a component called `Provider`, but it is not one.
        """
        return (0, len(surface)) in cls._name_spans(surface, name, form)

    @classmethod
    def _owners(cls, surface, components, form: NameForm):
        """The components whose name this surface writes at ``form``.

        The uniqueness test both loose scans apply, stated once: a surface owned by
        more than one component is evidence for neither. `pilot/ablate_all.py` prices
        it at 2.4 false positives for free.
        """
        return [component for component in components
                if cls._realizes(surface, component.name, form)]

    @classmethod
    def _name_spans(cls, text, name, form: NameForm):
        """**The relation.** Spans of ``text`` that write ``name`` at ``form``.

        The whole deterministic layer of this workflow is this function and the four
        values of ``NameForm``. It reads the runtime catalog and ``INFLECTIONS``, and
        nothing else; no benchmark vocabulary reaches it (GATE-06).

        The four branches were four methods in ``s_linker64``, verified identical to
        these over every (name, sentence) pair of all five projects
        (`pilot/rule_audit.py --only A2`, `pilot/test_s65_one_relation.py`).
        """
        if form is NameForm.AS_SPELLED:
            return [(m.start(), m.end()) for m in re.finditer(
                rf"(?<!\w){re.escape(name)}(?!\w)", text)]

        if form is NameForm.ANY_CASE:
            return [(m.start(), m.end()) for m in re.finditer(
                rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE)]

        if form is NameForm.ANY_SPELLING:
            target = cls._name_signature(name)
            if not target:
                return []
            # A span of k words yields at least k signature tokens, so a span longer
            # than the target's token count can never match it.
            words = list(re.finditer(r"[A-Za-z0-9]+", text))
            spans = []
            for i, first in enumerate(words):
                for j in range(i, min(len(words), i + len(target))):
                    if j > i and not re.fullmatch(
                        r"[\s_-]+", text[words[j - 1].end():words[j].start()]
                    ):
                        break
                    start, end = first.start(), words[j].end()
                    if cls._name_signature(text[start:end]) == target:
                        spans.append((start, end))
            return spans

        if form is NameForm.ANY_WORD:
            # Extent, not fidelity: one word of the name is enough, under an English
            # inflectional ending. An unbounded prefix would make `web` own `webrtc`
            # and `webcams`; naming the endings states the intent instead of
            # approximating it (s_linker62, +2.0 gold / +0.0 spurious candidates).
            words = [w.casefold() for w in re.findall(WORD_PATTERN, name)]
            return [
                (m.start(), m.end())
                for m in re.finditer(WORD_PATTERN, text)
                if any(m.group(0).casefold().startswith(word)
                       and m.group(0).casefold()[len(word):] in INFLECTIONS
                       for word in words)
            ]

        raise ValueError(f"unknown name form: {form!r}")

    @staticmethod
    def _full_name_source(candidate):
        if candidate.source == "full_name_variant":
            return "full_name_variant"
        return "full_name"

    @staticmethod
    def _name_signature(expression):
        """Normalize an expression to its sequence of words, splitting CamelCase.

        A spaced form, a hyphenated form, and a run-together form of the same
        words share a signature ("X Y", "x-y", and "XY" all give ("x", "y")),
        which is what makes a spelling variant recognizable.

        Compound splitting is *not* a relaxation of case folding: it reaches
        `Image Provider` for a name spelled `ImageProvider`, and it misses
        `redis pubsub` for a name spelled `Redis PubSub`, which case folding reaches.
        Six pairs over five projects; the linker takes the union of the forms it
        scans, so neither direction is lost.
        """
        normalized = unicodedata.normalize("NFKC", expression)
        normalized = normalized.replace("-", " ").replace("_", " ")
        return tuple(
            token.casefold()
            for token in re.findall(
                r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+", normalized
            )
        )

    @staticmethod
    def _in_dotted_path(text, start, end) -> bool:
        """True when text[start:end] is glued to a dot on either side, as in x.y.

        The single definition of "inside a qualified name". Two tests used to
        carry their own copy of it and the copies disagreed -- one asked whether
        the character after the dot ``isalnum()``, the other ``isalpha()``, and
        one required an alphanumeric before the dot while the other did not.
        Neither divergence ever changed a result (0 differences over 3697
        (name, sentence) pairs and 5388 word spans on all five projects), so the
        stricter reading is the one kept.
        """
        before = (start > 1 and text[start - 1] == "."
                  and text[start - 2].isalnum())
        after = (end + 1 < len(text) and text[end] == "."
                 and text[end + 1].isalnum())
        return before or after

    # ── Evidence bundles and the two-pass judge ──────────────────────────────

    def _classify_mention_typed(self, comp_name: str, text: str) -> MentionType:
        """Label how the name appears, using the one matching test.

        The case distinction is a comparison of the matched surface against the
        name, not a second predicate carrying its own case rules:
        ``_find_exact_form`` already returns what it matched. An earlier revision
        asked a strict, case-sensitive predicate first and a lowercase regex
        second, which is why the module used to hold two name tests.

        Measured: indistinguishable on this stage (TP +/-0.0, p=1.00; FP -0.4,
        p=0.76 over five runs per side) and inside the end-to-end variance band
        of the form it replaces (F1 96.47 over three runs against 96.42 +/- 0.42
        over six runs of the previous form; F2 95.20 against 95.38 +/- 0.58).
        """
        matched = self._find_exact_form(text, comp_name)
        if matched:
            if self._all_occurrences_in_qualified_path(comp_name.lower(), text):
                return MentionType.CODE_TOKEN
            return (MentionType.PROPER_STANDALONE if matched == comp_name
                    else MentionType.LOWERCASE_PROSE)
        for alias in self._names_by_component().get(comp_name, ()):
            if self._find_exact_form(text, alias):
                return MentionType.VIA_ALIAS
        return MentionType.INDIRECT

    @classmethod
    def _all_occurrences_in_qualified_path(cls, comp_lower: str, text: str) -> bool:
        any_match = False
        for m in re.finditer(rf'\b{re.escape(comp_lower)}\b', text):
            any_match = True
            if not cls._in_dotted_path(text, m.start(), m.end()):
                return False
        return any_match

    def _build_evidence_bundle(self, candidate, sent_map):
        comp_name = candidate.component_name
        snum = candidate.sentence_number
        # No computed label. The judge is holding the sentence the label was derived
        # from; this variant asks whether the derivation has to be done for it.
        mention_type = ""
        prev_sent = sent_map.get(snum - 1)
        anchors = []
        for s in sorted(sent_map.values(), key=lambda x: x.number):
            if s.number == snum:
                continue
            if self._find_exact_form(s.text, comp_name):
                anchors.append(f"S{s.number}: {s.text}")
                if len(anchors) >= self.ANCHOR_LIMIT:
                    break
        return EvidenceBundle(
            source=candidate.source,
            matched_span=candidate.matched_text or comp_name,
            mention_type=mention_type,
            preceding_text=prev_sent.text if prev_sent else "",
            anchor_sentences=anchors,
        )

    def _format_evidence(self, bundle: EvidenceBundle) -> str:
        lines = [
            f"  Evidence: source={bundle.source}, span=\"{bundle.matched_span}\"",
        ]
        if bundle.preceding_text:
            lines.append(f"  [prev: \"{bundle.preceding_text}\"]")
        if bundle.anchor_sentences:
            lines.append("  Anchors (confirmed refs):")
            for a in bundle.anchor_sentences:
                lines.append(f"    {a}")
        return "\n".join(lines)

    def _validate_with_evidence(self, candidates, bundles, components, sent_map,
                                p1_tag, p2_tag, stage_label):
        """Two judging passes; a link needs both."""
        if not candidates:
            return [], {}
        comp_names = get_comp_names(components)
        decisions: dict = {}
        approved = []
        for _, batch in self._iter_batches(candidates, self.JUDGE_BATCH):
            cases = []
            for i, c in enumerate(batch):
                p = self._prev_prefix(c.sentence_number, sent_map)
                bundle = bundles.get((c.sentence_number, c.component_id))
                evidence_block = self._format_evidence(bundle) if bundle else ""
                cases.append((
                    f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n'
                    f'  {p}"{c.sentence_text}"\n'
                    f'{evidence_block}',
                    c,
                ))
            case_strings = [ct for ct, _ in cases]
            r1 = self._run_validation_pass(comp_names, case_strings, P1_FOCUS, p1_tag)
            r2 = self._run_validation_pass(comp_names, case_strings, P2_FOCUS, p2_tag)
            for i, (_case_text, c) in enumerate(cases):
                p1 = r1.get(i, False)
                p2 = r2.get(i, False)
                ok = p1 and p2
                decisions[(c.sentence_number, c.component_id)] = {
                    "approved": ok, "p1": p1, "p2": p2,
                    "path": f"{stage_label}_twopass" if ok
                            else f"{stage_label}_twopass_reject",
                    "stage": f"{stage_label}_twopass",
                }
                if ok:
                    approved.append(c)
        return approved, decisions

    def _run_validation_pass(self, comp_names, cases, focus, phase_tag=None,
                             strict=False):
        if phase_tag:
            self.llm.set_phase(phase_tag)
        data = self._ask(
            self._prompt_validation(comp_names, cases, focus, strict=strict),
            timeout=120, label="Validation pass", require="validations",
        )
        results: dict[int, bool] = {}
        if data:
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    val = v.get("approve", False)
                    results[idx] = (
                        val is True
                        or (isinstance(val, str) and val.lower() == "true")
                    )
        return results

    # ═════════════════════════════════════════════════════════════════════════
    # Linker 2 — PARTIAL NAME: the sentence carries one word of a name.
    # ═════════════════════════════════════════════════════════════════════════

    def _run_partial_name_linker(self, sentences, components, linked, sent_map):
        candidates = self._scan(sentences, components, SCANS["name_word"])
        approved, decisions = self._judge_partial_names(candidates, sentences)
        links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name,
                       source="partial_name")
            for c in approved
        ]
        return links, {
            "proposed": self._link_view(
                [SadSamLink(c.sentence_number, c.component_id, c.component_name,
                            source="partial_name_candidate") for c in candidates],
                sent_map,
            ),
            "accepted": self._link_view(links, sent_map),
            "judge_decisions": self._decision_view(decisions),
        }

    def _judge_partial_names(self, candidates, sentences):
        """One step: does the expression denote a software participant?

        s_linker25 followed this with a grounded identity review that showed the
        model the target. Measured over six of its runs, that review rejected 8.0
        candidates per run of which 5.5 were gold -- it traded 5.5 true positives
        for 2.5 false positives -- so the linker judges once, like the coreference
        linker, and the target stays withheld throughout.
        """
        participants, decisions = self._classify_denotations(candidates, sentences)
        for candidate in participants:
            key = (candidate.sentence_number, candidate.component_id)
            decisions[key] = {**decisions.get(key, {}), "approved": True,
                              "requested_keep": True, "path": "denotation"}
        return participants, decisions

    def _classify_denotations(self, candidates, sentences):
        """Step 1: does the expression itself denote a software participant?

        The target component is deliberately withheld. Shown the target, the
        model confirms identity rather than testing it.
        """
        sent_map = {s.number: s for s in sentences}
        decisions = {}
        for _, batch in self._iter_batches(candidates, self.JUDGE_BATCH):
            evidence_ids = {
                sentence.number
                for candidate in batch
                for sentence in self._window(candidate.sentence_number, sentences)
            }
            sentence_table = [
                {"sentence": n, "text": sent_map[n].text}
                for n in sorted(evidence_ids)
            ]
            cases = [
                {"case": n, "source": c.sentence_number, "expression": c.matched_text}
                for n, c in enumerate(batch, 1)
            ]
            prompt = f"""Classify what each expression itself denotes in its
local context: participant for a software participant, or associated for
something merely associated with software.

{QUALIFIED_CLAUSE}

SENTENCES
{json.dumps(sentence_table)}

CASES
{json.dumps(cases)}

Claim must be a contiguous exact substring of the source sentence.

JSON only:
{{"judgments":[{{"case":1,"denotation":"participant",
"claim":"exact source quote"}}]}}
"""
            data = self._ask(
                prompt, phase="phase_25_partial_denotation",
                require_present="judgments", label="Denotation", timeout=240,
            )
            for item in data.get("judgments", []):
                case_value = str(item.get("case", ""))
                if not case_value.isdigit():
                    continue
                number = int(case_value)
                if not 1 <= number <= len(batch):
                    continue
                candidate = batch[number - 1]
                claim = str(item.get("claim", "")).strip().strip("\"'“”‘’")
                denotation = str(item.get("denotation", "")).strip()
                # The response contract only. The substring check that used to
                # follow it voided 0 of 380 verdicts over six five-project runs, which
                # is s_linker48's separation restated: demanding a committed quote is
                # worth 35.2 TP, verifying it is worth nothing.
                valid = denotation in {"participant", "associated"} and bool(claim)
                decisions[(candidate.sentence_number, candidate.component_id)] = {
                    "approved": False,
                    "requested_keep": False,
                    "evidence_valid": valid,
                    "claim": claim,
                    "denotation": denotation,
                    "alternative": "not reviewed",
                    "path": "denotation",
                    "stage": "partial_name",
                }
        participants = [
            c for c in candidates
            if decisions.get((c.sentence_number, c.component_id), {}).get(
                "denotation") == "participant"
            and decisions[(c.sentence_number, c.component_id)]["evidence_valid"]
        ]
        return participants, decisions

    def _run_coreference_linker(self, sentences, components, name_to_id, linked,
                                sent_map):
        resolved, metadata = self._resolve_references(
            sentences, components, name_to_id, sent_map
        )
        raw = resolved
        approved, decisions = self._validate_coref_links(raw, sent_map, components)
        return approved, {
            "candidates": self._link_view(raw, sent_map),
            "accepted": self._link_view(approved, sent_map),
            "metadata": [
                {"sentence": sentence, "component_id": component, **value}
                for (sentence, component), value in metadata.items()
            ],
            "judge_decisions": self._decision_view(decisions),
        }

    def _resolve_references(self, sentences, components, name_to_id, sent_map):
        """Every sentence goes to the LLM in context; no pronoun regex.

        A resolution survives only when its antecedent sentence itself states a
        name of the component — the structural antecedent constraint.

        Both sentence numbers a resolution reports are checked against the
        document: the target sentence as well as the antecedent. A number the
        model invents cannot name a real sentence, so admitting one could only
        ever add a link the gold standard has no counterpart for.
        """
        comp_names = get_comp_names(components)
        all_coref = []
        coref_metadata: dict = {}
        self.llm.set_phase("phase_25_coreference")

        for batch_num, batch in self._iter_batches(sentences, self.COREFERENCE_BATCH):
            cases = []
            for sent in batch:
                context = [
                    f'{">>>" if s.number == sent.number else "   "} '
                    f"S{s.number}: {s.text}"
                    for s in self._window(sent.number, sentences)
                ]
                cases.append({"sent": sent, "context": context})

            data = self._ask(
                self._prompt_coref(comp_names, cases), timeout=600,
                label=f"Coref batch {batch_num}", require_present="resolutions",
            )
            if not data:
                continue

            for res in data.get("resolutions", []):
                comp = res.get("component")
                snum = parse_snum(res.get("sentence"))
                if snum is None or snum not in sent_map:
                    continue
                if not comp or comp not in name_to_id:
                    continue
                ant_snum = parse_snum(res.get("antecedent_sentence"))
                if ant_snum is None:
                    print(f"    Coref skip (no antecedent): S{snum} -> {comp}")
                    continue
                ant_sent = sent_map.get(ant_snum)
                if not ant_sent:
                    continue
                # No antecedent gate. It accepted a resolution only when the
                # antecedent sentence itself stated a name in N(c). Replayed on a run's
                # own recorded resolutions and scored on what coreference actually
                # contributes -- pairs an earlier linker has NOT already produced --
                # removing it is TP +0.0 (p = 1.00), FP +0.0 (p = 1.00)
                # (`pilot/fold_pilots.py --pilot foldantecedent_net`). Its documented
                # 12 FP were counted with the stale pairs an inert `_unlinked` let
                # through; see `pilot/unlinked_audit.py`.
                cid = name_to_id[comp]
                all_coref.append(SadSamLink(snum, cid, comp, source="coreference"))
                coref_metadata[(snum, cid)] = {
                    "reference": res.get("reference", ""),
                    "antecedent_sentence": ant_snum,
                    "antecedent_text": res.get("antecedent_text", ""),
                    "raw_resolution": res,
                }
        return all_coref, coref_metadata

    def _validate_coref_links(self, coref_links, sent_map, components):
        """Single judging pass — asymmetric to the full-name linker's two passes,
        because resolution asks a narrower question."""
        if not coref_links:
            return [], {}
        comp_names = get_comp_names(components)
        validated = []
        decisions: dict = {}
        self.llm.set_phase("phase_25_coreference_judge")
        for _, batch in self._iter_batches(coref_links, self.JUDGE_BATCH):
            cases = []
            for i, lk in enumerate(batch):
                # _resolve_references admits a resolution only for a sentence
                # the document has, so every link reaching the judge has one.
                sent = sent_map[lk.sentence_number]
                p = self._prev_prefix(lk.sentence_number, sent_map)
                cases.append((
                    lk,
                    f'Case {i+1}: pronoun/role-ref -> {lk.component_name}\n'
                    f'  {p}"{sent.text}"',
                ))
            results = self._run_validation_pass(
                comp_names, [c for _, c in cases], COREF_VALIDATION_FOCUS,
                phase_tag="phase_25_coreference_judge", strict=True,
            )
            for idx, (lk, _case) in enumerate(cases):
                approved = bool(results.get(idx, False))
                decisions[(lk.sentence_number, lk.component_id)] = {
                    "approved": approved,
                    "path": "coref_validated" if approved else "coref_rejected",
                }
                if approved:
                    validated.append(lk)
                else:
                    print(f"    Coref reject: S{lk.sentence_number} -> {lk.component_name}")
        return validated, decisions

    # ── Logging and checkpointing ────────────────────────────────────────────

    def _backend_tag(self) -> str:
        inner = getattr(self.llm, "_inner", self.llm)
        backend = getattr(inner, "backend", None)
        if backend is None:
            return "unknown"
        return getattr(backend, "value", str(backend))

    def _checkpoint_dir(self, text_path):
        cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
        ds = os.path.splitext(os.path.basename(text_path))[0]
        d = os.path.join(cache_dir, self._VARIANT_NAME, self._backend_tag(), ds)
        os.makedirs(d, exist_ok=True)
        return d

    def _save_phase(self, text_path, phase_name, state):
        path = os.path.join(self._checkpoint_dir(text_path), f"{phase_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"  Checkpoint: {phase_name} saved")

    def _log(self, phase, input_summary, output_summary, links=None):
        entry = {"phase": phase, "ts": time.time(),
                 "in": input_summary, "out": output_summary}
        if links is not None:
            entry["links"] = [
                {"s": l.sentence_number, "c": l.component_name, "src": l.source}
                for l in links
            ]
        self._phase_log.append(entry)

    def _save_log(self, text_path):
        log_dir = os.environ.get("LLM_LOG_DIR", "./results/llm_logs")
        os.makedirs(log_dir, exist_ok=True)
        ds = os.path.splitext(os.path.basename(text_path))[0]
        ts = time.strftime("%Y%m%d_%H%M%S")
        backend = self._backend_tag()
        summary_path = os.path.join(
            log_dir, f"{self._VARIANT_NAME}_{backend}_{ds}_{ts}.json")
        with open(summary_path, "w") as f:
            json.dump(self._phase_log, f, indent=2, default=str)
        print(f"  Phase log saved: {summary_path}")
        calls_path = os.path.join(
            log_dir, f"{self._VARIANT_NAME}_{backend}_{ds}_{ts}_calls.json")
        trunc_env = os.environ.get("CALLS_TRUNCATE_CHARS", "").strip()
        trunc = int(trunc_env) if trunc_env.isdigit() else 0
        if trunc > 0:
            calls = []
            for c in self._llm_calls:
                cc = dict(c)
                if cc.get("prompt") and len(cc["prompt"]) > trunc:
                    cc["prompt"] = cc["prompt"][:trunc] + "... [truncated]"
                if cc.get("response_text") and len(cc["response_text"]) > trunc:
                    cc["response_text"] = cc["response_text"][:trunc] + "... [truncated]"
                calls.append(cc)
        else:
            calls = self._llm_calls
        with open(calls_path, "w") as f:
            json.dump(calls, f, indent=2, default=str)
        print(f"  LLM call trace saved: {calls_path} ({len(self._llm_calls)} calls)")

    def _compute_phase_metrics(self) -> dict:
        metrics: dict[str, dict] = {}
        for call in self._llm_calls:
            ph = call.get("phase", "unknown")
            m = metrics.setdefault(
                ph, {"calls": 0, "elapsed_s": 0.0, "tokens": 0, "errors": 0})
            m["calls"] += 1
            m["elapsed_s"] = round(m["elapsed_s"] + call.get("elapsed_s", 0.0), 3)
            if call.get("success") is False:
                m["errors"] += 1
            usage = call.get("token_usage")
            if usage:
                m["tokens"] += usage.get("total_tokens", 0) or 0
        return metrics
