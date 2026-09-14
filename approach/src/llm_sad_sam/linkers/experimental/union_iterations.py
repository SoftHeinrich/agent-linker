"""Every iteration of the union judge, in one file, side by side.

`s_linker120` merges the head's two name judges into one call. What that judge *says*
went through nine iterations, each one a single named change measured against a control
that ran beside it in the same invocation. This module holds all nine as data: the rule
text, the case format, the verdict contract, and the numbers each one read. The variant
imports the active one; the pilot can run any of them as an arm; the defensibility audit
scores whichever is active.

    from llm_sad_sam.linkers.experimental.union_iterations import ITERATIONS, ACTIVE
    ITERATIONS["v3"].rule           # what that iteration told the judge
    ITERATIONS["v3"].measured       # what it read, per model

**How to read the numbers.** Every row is *this arm minus the control that ran in the
same invocation*, three samples a side, five projects, per five-project run; `net` is
`3*gold - spurious`, the branch's F2 exchange rate at the head's operating point. Absolute
levels drift between invocations, so only the deltas are comparable across rows, and only
within a model.

**The shape of the trail.** Iterations 1-6 kept the head's two standards and routed
between them inside one prompt by a code fact -- a union of the *prompts*, not of the
*question*. Iterations 7-13 dropped the routing: **one rule that says what a trace link
is and how to read each piece of evidence**, with every candidate in one case format,
and with the evidence deciding what a case carries, which batch it joins, and what its
call can be asked. **v13 is adopted**: gold neutral on both models, spurious -12.6 a run
on terra (p = 0.000, n = 5) and -17.7 on luna (p = 0.008, n = 3), at the same 14 calls
and one prompt instead of two.

**What each design cost, as a mechanism rather than a number.** The routed designs kept
losing the word-only stream to whichever sentence asked for identity (v1 -11.3 gold);
the row-free designs kept losing it to the *company* those cases keep, not to the rule
at all -- three successive rewrites of the rule left luna's word-only row at ~10 gold
against a control's ~21, and grouping the cases by what the match computed recovered it
in one step. The lesson the round ends on: **for a judge, what a case is shown beside is
as much a part of its evidence as what the case says.**
"""
from __future__ import annotations

from dataclasses import dataclass, field

from llm_sad_sam.linkers.experimental.s_linker110 import (
    LAYERED_ENTITY_RULES, LAYERED_COREF_RULES, QUALIFIED_CLAUSE, STRICTER_CLAUSE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Slices of the head's own constants, so quotation is mechanical, not retyped.
# `pilot/union_defensibility.py` checks each one against the constant it came from.
# ─────────────────────────────────────────────────────────────────────────────

_ENTITY_SENTENCES = [s.strip() for s in LAYERED_ENTITY_RULES.split(". ") if s.strip()]

#: "A mention that says nothing further about the component still counts as a valid link."
MENTION_COUNTS = _ENTITY_SENTENCES[1] + ". "

#: "Reject only on a positive ground -- that the sentence asserts nothing of this
#: component, because the name is doing some other job here, ..."
POSITIVE_GROUND = _ENTITY_SENTENCES[2] if len(_ENTITY_SENTENCES) > 2 else ""

#: The reference clause of the strict gate: what an expression denotes when the
#: component is the actor.
ACTS_ON = ("An expression denoting what a component acts on or produces refers to "
           "that thing and not to the component, however clearly the component is "
           "the one acting on it.")
assert ACTS_ON in LAYERED_COREF_RULES

#: The head's denotation question, as `_classify_denotations` asks it.
DENOTATION_QUESTION = ("Classify what each expression itself denotes in its local "
                       "context: participant for a software participant, or "
                       "associated for something merely associated with software.")


@dataclass(frozen=True)
class Iteration:
    """One measured version of the union judge.

    `rule`        what the prompt says above the cases.
    `clauses`     extra clauses appended after the rule (empty once they moved inside).
    `demand`      the quote-before-verdict instruction, and the reply contract it names.
    `reply`       the JSON template shown to the model.
    `fields`      which evidence fields a case may print.
    `blind_word_only`
                  True when a case whose sentence writes only one word of a name
                  carries no component, no anchors and no alternatives.
    `verdict`     "boolean" (one `approve` for every case) or "per_row" (the boolean
                  for naming rows, the head's `denotation` enum for word-only ones).
    `rows`        True while the rule states a standard per `naming` row.
    """

    name: str
    summary: str
    rule: str
    demand: str
    reply: str
    fields: tuple[str, ...]
    blind_word_only: bool
    verdict: str
    rows: bool
    clauses: str = ""
    #: True when cases are grouped by what the match computed — every case carrying a
    #: component in one batch, every case carrying none in another. One rule and one
    #: prompt template either way; what the grouping changes is the company a case
    #: keeps, and the call count is unchanged (the control pays the same two batches).
    batch_by_evidence: bool = False
    #: Where a case whose sentence writes only one word of a name still prints the
    #: component that word came from: `hidden` (v13), `header`, or `evidence` (on the
    #: `writes` line). Only read when `blind_word_only` is True — an un-blinded case
    #: names its component in the header by construction.
    word_only_component: str = "hidden"
    #: Whether a word-only case prints its anchors — the other sentences that name
    #: this component. False through v17: an anchor names a component and a blind
    #: case carries none.
    word_only_anchors: bool = False
    #: True when a call whose cases carry no component is not given the catalog and
    #: answers the head's denotation contract. The prompt carries what the batch's
    #: evidence has: a call with no component in any case has no catalog to check
    #: against and nothing to approve against, so it classifies instead. The rule
    #: above the cases is the same rule.
    contract_follows_batch: bool = False
    changed: str = ""
    measured: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Iterations 1-6 — two standards, routed inside one prompt by the `naming` fact
# ─────────────────────────────────────────────────────────────────────────────

_V1_RULE = """A link says the sentence makes an architectural claim about the component named in the case. Each case carries an evidence line stating how the sentence reaches that component; the row you are given decides how far to extend the case before asking it for more.

naming=whole name — the component's name is written here and the document treats it as part of the system. Approve by default: a mention that says nothing further about the component still counts as a valid link. Reject only on a positive ground -- that the sentence asserts nothing of this component, because the name is doing some other job here, or because the sentence denies what it would otherwise say of it.

naming=alias — as above, reached through a short form the document itself established for that component. The same default holds, and the ground for rejecting is the same one.

naming=word only — the sentence writes one word of the name and never the whole name. Approve only when that word is being used to name this component here; if it is used in its ordinary sense, or if a component listed under alternatives is the one the sentence means, reject."""

_V2_RULE = f"""Each case's evidence line states how the sentence reaches the component. Apply the standard for that row, and only that one.

naming=whole name, naming=alias -- the component's name, or a short form the document established for it, is written in the sentence. {LAYERED_ENTITY_RULES}
{STRICTER_CLAUSE}

naming=word only -- the sentence writes one word of the component's name and never the whole name, so the case is not asking whether the name is written. Classify what the expression itself denotes in its local context: approve when it denotes a software participant, reject when it denotes something merely associated with software. {ACTS_ON}

An `alternatives` entry lists the other components whose names carry the same word. It is context for what the expression could denote, not a ground for rejecting this case."""

_V3_RULE = _V2_RULE.replace(
    "naming=word only -- the sentence writes one word of the component's name and "
    "never the whole name, so the case is not asking whether the name is written.",
    "naming=word only -- the case gives you an expression alone, with no component "
    "named: the sentence writes one word of some component's name and never the whole "
    "name.")

_V5_RULE = _V3_RULE.replace(
    "Classify what the expression itself denotes in its local context: approve when "
    "it denotes a software participant, reject when it denotes something merely "
    "associated with software.",
    "Classify what the expression itself denotes in its local context: participant "
    "for a software participant, or associated for something merely associated with "
    "software.")

_V6_RULE = _V5_RULE.split("\n\nAn `alternatives` entry")[0]

_DEMAND_HEAD = ('first quote the EXACT words from the sentence that state the '
                'architectural claim about the component (or write "none" if the '
                'sentence makes no such claim), then decide approve true/false based '
                'on that claim.')
_DEMAND_ROW_AWARE = ('first quote the EXACT words from the sentence the verdict rests '
                     'on -- for a naming row, the words that state the architectural '
                     'claim about the component (or write "none" if the sentence makes '
                     'no such claim); for a word-only row, the words that fix what the '
                     'expression denotes -- then decide approve true/false based on '
                     'that quote.')
_DEMAND_PER_ROW = ('For a naming row, first quote the EXACT words from the sentence '
                   'that state the architectural claim about the component (or write '
                   '"none" if the sentence makes no such claim), then decide approve '
                   'true/false based on that claim.\n\nFor a word-only row, quote as '
                   'the claim a contiguous exact substring of the source sentence, '
                   'then answer denotation with participant or associated.')

_REPLY_BOOLEAN = ('{"validations": [{"case": 1, "claim": "<exact quote or none>", '
                  '"approve": true}]}')
_REPLY_PER_ROW = ('{"validations": [{"case": 1, "claim": "<exact quote or none>", '
                  '"approve": true},\n                 {"case": 2, "claim": '
                  '"<exact source quote>", "denotation": "participant"}]}')

# ─────────────────────────────────────────────────────────────────────────────
# Iterations 7-9 — one rule: what a trace link is, and how to read the evidence
# ─────────────────────────────────────────────────────────────────────────────

_DEFINITION = ("A trace link holds between a sentence and a component when the "
               "sentence makes an architectural claim about that component -- when it "
               "says something about that component as a participant in the system "
               "this document describes. ")

_FORMAT_V7 = ("Every case gives you the same four things: the expression the sentence "
              "uses, the component that expression may reach, the sentence itself, and "
              "the evidence the document supplies for the reach.")

_FORMAT_V9 = ("Every case gives you the expression the sentence uses, the sentence "
              "itself, the evidence the document supplies, and -- where the sentence "
              "writes a name of it -- the component that name reaches. A case whose "
              "sentence writes no name of any component carries no component: decide "
              "what the expression denotes there, and nothing about identity.")

_WRITES_V7 = ("  writes -- what this sentence writes of the component's name: the "
              "whole name, a short form the document established for it, or one word "
              "of the name. The more of a name a sentence writes, the less room is "
              "left for the expression to be doing something else; one word alone "
              "leaves the most.")

_WRITES_V8 = ("  writes -- what this sentence writes of the component's name: the "
              "whole name, a short form the document established for it, or one word "
              "of the name. A shorter surface leaves more readings open; it does not "
              "make the reading in front of you wrong. Where the sentence does not "
              "write the name as such, ask what the expression itself denotes in its "
              "local context: a participant in the system, or something merely "
              "associated with software.")

_FIELD_LINES = (
    "  alternatives -- other components whose names carry the same word. They are "
    "what the expression could be reaching instead of this one.\n"
    "  mention -- what the code can tell about the expression's place in the "
    "sentence.\n"
    "  anchors -- other sentences of this document that name this component. They fix "
    "what the name means in the document; they do not decide this sentence."
)


# ─────────────────────────────────────────────────────────────────────────────
# Iterations 14-16 — the unblinding round: can an instruction replace the refusal?
#
# v13 keeps the word-only row healthy by removing every way a merged prompt can ask
# an identity question of it: the target in the header (`blind_word_only`), the
# catalog above it and the claim demand (`contract_follows_batch`), and the named
# cases beside it (`batch_by_evidence`). Three of those four are properties of the
# CALL and stay. These arms put the fourth back -- the case names its component --
# and ask whether the rule can say what that slot means well enough to keep the
# question a denotation question.
#
# The cell is untested: every earlier un-blinded arm (v7, v8) was un-blinded with
# all three of the other carriers still present, so v8's -10.3 gold on luna prices
# "un-blinded AND catalogued AND claim-demanded AND mixed", not the slot.
# ─────────────────────────────────────────────────────────────────────────────

#: The input contract with the component always present. `_FORMAT_V9`'s second
#: sentence is about a case that carries none, so it cannot be carried as it stands:
#: an un-blinded prompt that keeps it is describing a case format it does not use.
#: This is the naive un-blinding -- the contract restated truthfully and nothing more.
_FORMAT_V14 = ("Every case gives you the expression the sentence uses, the sentence "
               "itself, the evidence the document supplies, and the component whose "
               "name that expression reaches.")

#: The same contract, plus what the component slot IS where the sentence writes one
#: word of a name: the provenance of the word, not a claim under test. The closing
#: clause is `_FORMAT_V9`'s own -- the question the head's denotation judge asks --
#: moved from "the case carries no component" to "the case carries one, and here is
#: what it is for".
_FORMAT_V15 = _FORMAT_V14 + (
    " Where the sentence writes only one word of that name, the component tells you "
    "which name the word came from and nothing more: decide what the expression "
    "denotes there, and nothing about identity.")

#: `STRICTER_CLAUSE`'s scope, stated by what the SENTENCE writes rather than by what
#: the case carries. v11 scoped it "Where the case gives you a component"; once every
#: case gives one, that phrase excludes nothing and the use/mention clause becomes
#: live on the word-only row -- which is an identity question in the one place these
#: arms are trying not to ask one. The clause is about an ordinary word coinciding
#: with a component's *name*, so the condition it was always about is that the
#: sentence writes the name.
_STRICTER_SCOPE_BY_WRITING = "Where the sentence writes a name of the component: "


def _unblinded_rule(form: str) -> str:
    """`_rowless_rule` with the use/mention clause scoped by what the sentence writes."""
    return _rowless_rule(form, _WRITES_V8).replace(
        STRICTER_CLAUSE, _STRICTER_SCOPE_BY_WRITING + STRICTER_CLAUSE, 1)


def _rowless_rule(form: str, writes: str, scope_stricter: bool = False) -> str:
    """The row-free rule: what a trace link is, then how to read each piece of evidence.

    `scope_stricter` states where the use/mention clause applies. That clause is about
    an ordinary word coinciding with *a component's name*, so a case that carries no
    component has no subject for it; the general round measured the same mistake in
    the other direction -- `QUALIFIED_CLAUSE` moved into the coreference rubric cost
    TP 3.0 because those cases contain no identifier for it to be about. Scoping is
    not a second rubric: it is the clause saying what it is about.
    """
    stricter = (f"Where the case gives you a component: {STRICTER_CLAUSE}"
                if scope_stricter else STRICTER_CLAUSE)
    return f"""{_DEFINITION}{MENTION_COUNTS}

{form} The evidence says what the expression is doing here; none of it is a verdict.

{writes}
{_FIELD_LINES}

{POSITIVE_GROUND}

{stricter}

{QUALIFIED_CLAUSE} {ACTS_ON}"""


_DEMAND_ROWLESS = ('For each case, first quote the EXACT words from the sentence the '
                   'verdict rests on -- the words that state the architectural claim '
                   'about the component, or "none" if the sentence makes no such '
                   'claim -- then decide approve true/false based on that quote.')


ITERATIONS: dict[str, Iteration] = {
    "v1": Iteration(
        name="v1", rows=True, blind_word_only=False, verdict="boolean",
        summary="two standards, paraphrased; the alternative set is a ground to reject",
        rule=_V1_RULE, clauses=f"{QUALIFIED_CLAUSE}\n{STRICTER_CLAUSE}",
        demand=_DEMAND_HEAD, reply=_REPLY_BOOLEAN,
        fields=("source", "naming", "mention", "alternatives", "last_named"),
        changed="the first build",
        measured={"terra": dict(gold=-11.3, spurious=-0.7, net=-33.2,
                                note="word only 21.3 -> 13.7; whole name 128.0 -> "
                                     "124.3. An identity demand is not the denotation "
                                     "question, and an alternative set offered as a "
                                     "reject-ground fires on a 0.765-gold bucket.")},
    ),
    "v2": Iteration(
        name="v2", rows=True, blind_word_only=False, verdict="boolean",
        summary="both standards quoted verbatim; the alternative set made inert",
        rule=_V2_RULE, demand=_DEMAND_HEAD, reply=_REPLY_BOOLEAN,
        clauses=QUALIFIED_CLAUSE,
        fields=("source", "naming", "mention", "alternatives", "last_named"),
        changed="paraphrase -> quotation; alternatives context, not ground",
        measured={"terra": dict(gold=-4.0, spurious=-12.7, net=+0.7,
                                note="both naming rows beat the control (whole name "
                                     "+2.0 gold / -1.0 spurious, alias +0.3 / -4.3); "
                                     "all of the deficit is word only.")},
    ),
    "v3": Iteration(
        name="v3", rows=True, blind_word_only=True, verdict="boolean",
        summary="+ the word-only case carries no component, no anchors, no alternatives",
        rule=_V3_RULE, demand=_DEMAND_HEAD, reply=_REPLY_BOOLEAN,
        clauses=QUALIFIED_CLAUSE,
        fields=("source", "naming", "mention", "alternatives", "last_named"),
        changed="s25's refusal honoured in the bundle rather than by a second prompt",
        measured={"terra": dict(gold=-4.0, spurious=-13.4, net=+1.4,
                                note="blinding recovered 1.4 of the 6.4 word-only "
                                     "gold, so the shown target was not the whole "
                                     "mechanism. Best net of the routed designs.")},
    ),
    "v4": Iteration(
        name="v4", rows=True, blind_word_only=True, verdict="boolean",
        summary="+ the quote demand made row-aware",
        rule=_V3_RULE, demand=_DEMAND_ROW_AWARE, reply=_REPLY_BOOLEAN,
        clauses=QUALIFIED_CLAUSE,
        fields=("source", "naming", "mention", "alternatives", "last_named"),
        changed="'the architectural claim about the component' asked of a case with no "
                "component in it is s86's contradiction",
        measured={"terra": dict(gold=-0.7, spurious=+3.3, net=-5.4,
                                note="loosened the NAMING rows (+4.0 gold / +9.0 "
                                     "spurious) and left word only flat (+0.3). The "
                                     "contradiction was not what held that row down.")},
    ),
    "v5": Iteration(
        name="v5", rows=True, blind_word_only=True, verdict="per_row",
        summary="+ each row answers in the field the head gives it (enum for word-only)",
        rule=_V5_RULE, demand=_DEMAND_PER_ROW, reply=_REPLY_PER_ROW,
        clauses=QUALIFIED_CLAUSE,
        fields=("source", "naming", "mention", "alternatives", "last_named"),
        changed="s119's mechanism, run backwards: a schema carries a default",
        measured={"terra": dict(gold=-3.0, spurious=+3.7, net=-12.7,
                                note="the enum recovered the word-only row (-1.6 gold "
                                     "against -4.6 in v3) -- the single largest lever "
                                     "found for that row -- while the naming rows "
                                     "drifted.")},
    ),
    "v6": Iteration(
        name="v6", rows=True, blind_word_only=True, verdict="per_row",
        summary="- alternatives and recency removed from every case",
        rule=_V6_RULE, demand=_DEMAND_PER_ROW, reply=_REPLY_PER_ROW,
        clauses=QUALIFIED_CLAUSE,
        fields=("source", "naming", "mention"),
        changed="tests whether the union's extra evidence fields were loosening the "
                "naming rows",
        measured={"terra": dict(gold=+0.7, spurious=+26.4, net=-24.3,
                                note="refuted, hard, and the most informative result "
                                     "of the round: an evidence field that names what "
                                     "a case could reach INSTEAD is what restrains "
                                     "the judge. Whole name kept 158.0 of 172.")},
    ),
    "v7": Iteration(
        name="v7", rows=False, blind_word_only=False, verdict="boolean",
        summary="one rule: what a trace link is and how to read the evidence; one "
                "case format for every candidate",
        rule=_rowless_rule(_FORMAT_V7, _WRITES_V7),
        demand=_DEMAND_ROWLESS, reply=_REPLY_BOOLEAN,
        fields=("writes", "alternatives", "mention"),
        changed="no rows, no defaults, no per-stream standard",
        measured={"terra": dict(gold=-4.0, spurious=-7.4, net=-4.6,
                                note="precision 0.883 -> 0.916. The strictness "
                                     "gradient in the `writes` line read as a licence "
                                     "to reject: word only -3.0 gold.")},
    ),
    "v8": Iteration(
        name="v8", rows=False, blind_word_only=False, verdict="boolean",
        summary="+ the `writes` reading carries the denotation question instead of a "
                "strictness gradient",
        rule=_rowless_rule(_FORMAT_V7, _WRITES_V8),
        demand=_DEMAND_ROWLESS, reply=_REPLY_BOOLEAN,
        fields=("writes", "alternatives", "mention"),
        changed="'a shorter surface leaves more readings open; it does not make the "
                "reading in front of you wrong' + the head's denotation question",
        measured={"terra": dict(gold=-0.7, spurious=-1.0, net=-1.1,
                                note="PARITY on terra at one prompt instead of two, "
                                     "precision 0.888 -> 0.893."),
                  "luna": dict(gold=-10.3, spurious=-7.0, net=-23.9,
                               note="refused on the second model: word only 21.7 -> "
                                    "12.3 gold. With the component shown, luna judges "
                                    "that row by identity -- s25's refusal, "
                                    "model-dependent exactly as s86 warns.")},
    ),
    "v9": Iteration(
        name="v9", rows=False, blind_word_only=True, verdict="boolean",
        summary="+ the component slot is filled by the evidence, so a case whose match "
                "computed no component carries none",
        rule=_rowless_rule(_FORMAT_V9, _WRITES_V8),
        demand=_DEMAND_ROWLESS, reply=_REPLY_BOOLEAN,
        fields=("writes", "alternatives", "mention"),
        changed="one rule and one format kept; what differs is what the match had to "
                "put in the case",
        measured={},
    ),
}

ITERATIONS["v10"] = Iteration(
    name="v10", rows=False, blind_word_only=True, verdict="per_row",
    summary="+ a case that carries no component answers what the expression denotes, "
            "because there is no component in it to approve against",
    rule=_rowless_rule(_FORMAT_V9, _WRITES_V8),
    demand=_DEMAND_ROWLESS + (
        "\n\nA case that carries no component cannot be approved against one: for "
        "those, quote as the claim a contiguous exact substring of the source "
        "sentence, then answer denotation with participant or associated."),
    reply=_REPLY_PER_ROW,
    fields=("writes", "alternatives", "mention"),
    changed="the reply follows the case the same way the rule does: nothing to "
            "approve against, so the answer is the denotation",
    measured={"luna": dict(gold=-21.0, spurious=-22.0, net=-41.0,
                           note="refuted by non-compliance, not by judgement: luna "
                                "kept answering `approve` for the cases the demand "
                                "asked a `denotation` of, and the parser rejects what "
                                "the contract did not answer -- word only kept 1.0 of "
                                "81. A row-free prompt cannot carry a per-row reply "
                                "contract; v5 could, because it had rows to hang it "
                                "on.")},
)

ITERATIONS["v11"] = Iteration(
    name="v11", rows=False, blind_word_only=True, verdict="boolean",
    summary="+ the use/mention clause says where it applies, so a case with no "
            "component is not asked an identity question by it",
    rule=_rowless_rule(_FORMAT_V9, _WRITES_V8, scope_stricter=True),
    demand=_DEMAND_ROWLESS, reply=_REPLY_BOOLEAN,
    fields=("writes", "alternatives", "mention"),
    changed="STRICTER_CLAUSE scoped to cases that carry a component; v10's enum "
            "reverted, luna never answered in it (word only kept 1.0 of 81)",
    measured={"luna": dict(gold=-12.0, spurious=-22.3, net=-13.7,
                           note="the whole-name row is the union's by a wide margin "
                                "(-1.0 gold at -21.4 spurious, precision 0.798 -> "
                                "0.882); the word-only row is unmoved at 9.7 against "
                                "21.3, so the identity pressure was not in that "
                                "clause either.")},
)

ITERATIONS["v12"] = Iteration(
    name="v12", rows=False, blind_word_only=True, verdict="boolean",
    batch_by_evidence=True,
    summary="+ the evidence decides the batch as well as the case: cases carrying no "
            "component are judged together, under the same one rule",
    rule=_rowless_rule(_FORMAT_V9, _WRITES_V8, scope_stricter=True),
    demand=_DEMAND_ROWLESS, reply=_REPLY_BOOLEAN,
    fields=("writes", "alternatives", "mention"),
    changed="three iterations moved the rule and left luna's word-only row at ~10 "
            "gold against ~21; the remaining difference from the head is the company "
            "those cases keep in a batch",
    measured={"luna": dict(gold=-7.3, spurious=-6.3, net=-15.6,
                           note="grouping recovered the row's gold (9.7 -> 15.3) and "
                                "loosened it (4.7 -> 20.3 spurious): the company a "
                                "case keeps is a bigger lever on that row than any "
                                "sentence of the rule was.")},
)

ITERATIONS["v13"] = Iteration(
    name="v13", rows=False, blind_word_only=True, verdict="boolean",
    batch_by_evidence=True, contract_follows_batch=True,
    summary="+ the call carries what its batch's evidence has: no component in any "
            "case means no catalog and the denotation contract",
    rule=_rowless_rule(_FORMAT_V9, _WRITES_V8, scope_stricter=True),
    demand=_DEMAND_ROWLESS, reply=_REPLY_BOOLEAN,
    fields=("writes", "alternatives", "mention"),
    changed="v12 recovered luna's word-only gold by grouping (9.7 -> 15.3) and paid "
            "15.6 spurious for it; the two things the head's denotation call still "
            "has that the union's grouped call did not are the absent catalog and "
            "the enum",
    measured={
        "terra": dict(gold=+0.2, spurious=-12.6, net=+13.2, samples=5, p_gold=1.000,
                      p_spurious=0.000, p_net=0.025,
                      note="ADOPTED. 25 paired (sample, project) units: gold neutral, "
                           "spurious down at p = 0.000, net up at p = 0.025. The "
                           "whole-name row is significantly better on BOTH axes "
                           "(gold +0.36/unit p = 0.007, spurious -0.60 p = 0.002); "
                           "the word-only row loses 0.32 gold a unit (p = 0.37, n.s.) "
                           "and 1.32 spurious (p = 0.004). Precision 0.878 -> 0.939."),
        "luna": dict(gold=+0.3, spurious=-17.7, net=+18.7, samples=3, p_gold=1.000,
                     p_spurious=0.008, p_net=0.011,
                     note="15 units: gold neutral, spurious down at p = 0.008, net up "
                          "at p = 0.011. Precision 0.789 -> 0.859. The whole-name row "
                          "carries it: -1.3 gold at -21.0 spurious."),
    },
)

ITERATIONS["v14"] = Iteration(
    name="v14", rows=False, blind_word_only=True, verdict="boolean",
    batch_by_evidence=True, contract_follows_batch=True,
    word_only_component="header",
    summary="+ the word-only case names its component again, with the input contract "
            "restated and nothing else said about the slot",
    rule=_rowless_rule(_FORMAT_V14, _WRITES_V8, scope_stricter=True),
    demand=_DEMAND_ROWLESS, reply=_REPLY_BOOLEAN,
    fields=("writes", "alternatives", "mention"),
    changed="v13's three call-level removals kept; the header slot filled. What a "
            "reader would write if they simply stopped withholding the name",
    measured={
        "terra": dict(gold=-2.3, spurious=-2.3, net=-4.7, samples=3, p_gold=0.656,
                      p_spurious=0.438, p_net=0.750,
                      note="against v13 in the same invocation. Word-only row 20.7 -> "
                           "18.3 gold at 7.0 -> 6.0 spurious; precision 0.909 -> "
                           "0.919. Replicated in a second invocation: gold -3.3 "
                           "(p = 0.562) at spurious +/-0.0, word-only 22.0 -> 19.0. "
                           "Gold-negative in 2 of 2, each inside its own noise."),
        "luna": dict(gold=-0.7, spurious=-17.7, net=+15.7, samples=3, p_gold=0.906,
                     p_spurious=0.250, p_net=0.406,
                     note="against v13 in the same invocation. Word-only row 22.7 -> "
                          "22.3 gold (p = 1.000 on that row) at 18.0 -> 11.3 spurious; "
                          "precision 0.770 -> 0.835. Second invocation: gold -0.7 "
                          "(p = 0.875), spurious -4.7, precision 0.773 -> 0.789. "
                          "**v8 read -10.3 gold on this model for the same slot** -- "
                          "what it was pricing was the slot PLUS the catalog, the "
                          "claim demand and the mixed batch, not the slot."),
    },
)

ITERATIONS["v15"] = Iteration(
    name="v15", rows=False, blind_word_only=True, verdict="boolean",
    batch_by_evidence=True, contract_follows_batch=True,
    word_only_component="header",
    summary="+ the rule says what the component slot is for on a word-only case "
            "(provenance of the word, not a claim under test) and scopes the "
            "use/mention clause by what the sentence writes",
    rule=_unblinded_rule(_FORMAT_V15),
    demand=_DEMAND_ROWLESS, reply=_REPLY_BOOLEAN,
    fields=("writes", "alternatives", "mention"),
    changed="the instruction the round is testing: two sentences against v14",
    measured={"luna": dict(gold=-1.0, spurious=+25.3, net=-28.3, samples=3,
                           note="against v14 in the same invocation: word-only "
                                "spurious 11.3 -> 35.0 at 22.3 -> 21.3 gold. The "
                                "rescoping is the loosener, not the sentence -- v17 "
                                "moves the sentence alone and tightens instead. An "
                                "instruction that says which clause does NOT apply "
                                "removes a restraint and states no criterion in its "
                                "place.")},
)

ITERATIONS["v16"] = Iteration(
    name="v16", rows=False, blind_word_only=True, verdict="boolean",
    batch_by_evidence=True, contract_follows_batch=True,
    word_only_component="evidence",
    summary="+ the component printed on the `writes` line instead of the case header",
    rule=_unblinded_rule(_FORMAT_V15),
    demand=_DEMAND_ROWLESS, reply=_REPLY_BOOLEAN,
    fields=("writes", "alternatives", "mention"),
    changed="placement: the same bytes of information, off the slot that reads as the "
            "claim under test and onto the field it is evidence of",
    measured={"luna": dict(gold=-0.3, spurious=+19.4, net=-20.3, samples=3,
                           note="against v14 in the same invocation: word-only 22.0 "
                                "gold at 30.7 spurious against v14's 22.3 at 11.3. "
                                "Placement is not the lever -- v16 carries v15's "
                                "rescoping and reads v15's loosening, off the header "
                                "slot entirely.")},
)

ITERATIONS["v17"] = Iteration(
    name="v17", rows=False, blind_word_only=True, verdict="boolean",
    batch_by_evidence=True, contract_follows_batch=True,
    word_only_component="header",
    summary="+ v14 with the provenance sentence, and the use/mention clause left "
            "where v13 had it",
    rule=_rowless_rule(_FORMAT_V15, _WRITES_V8, scope_stricter=True),
    demand=_DEMAND_ROWLESS, reply=_REPLY_BOOLEAN,
    fields=("writes", "alternatives", "mention"),
    changed="the decomposition arm: v15 moved two things at once (the sentence and "
            "the clause's scope) and read 35.0 spurious on the word-only row against "
            "v14's 11.3. This holds the scope and moves only the sentence",
    measured={
        "luna": dict(gold=-4.7, spurious=-6.3, net=-7.7, samples=3, p_gold=0.094,
                     p_spurious=0.469, p_net=0.340,
                     note="against v14 in the same invocation. The sentence alone is "
                          "a TIGHTENER: word-only 23.3 -> 19.7 gold at 25.7 -> 15.7 "
                          "spurious. So v15's loosening was the rescoping, and the "
                          "instruction itself costs gold at p = 0.094."),
        "terra": dict(gold=0.0, spurious=+1.0, net=-1.0, samples=3,
                      note="against v14 in the same invocation: 170.0 gold either "
                           "way, word-only 18.3 -> 18.7. The sentence buys nothing "
                           "on the model that pays for the slot."),
    },
)

ITERATIONS["v18"] = Iteration(
    name="v18", rows=False, blind_word_only=True, verdict="boolean",
    batch_by_evidence=True, contract_follows_batch=True,
    word_only_component="header", word_only_anchors=True,
    summary="+ the word-only case carries its anchors too: the other sentences of "
            "this document that name the component whose word it writes",
    rule=_rowless_rule(_FORMAT_V14, _WRITES_V8, scope_stricter=True),
    demand=_DEMAND_ROWLESS, reply=_REPLY_BOOLEAN,
    fields=("writes", "alternatives", "mention"),
    changed="v14's remaining gold losses are all a generic word of a multi-word name "
            "whose whole name the document writes elsewhere; the anchors line is the "
            "evidence that says so, and v13 could not carry it because its case "
            "carried no component for the anchors to be about",
    measured={
        "terra": dict(gold=+1.3, spurious=+3.7, net=+0.3, samples=3, p_gold=0.250,
                      p_spurious=0.062, p_net=1.000,
                      note="against v14 in the same invocation: word-only 19.0 -> "
                           "20.0 gold at 6.3 -> 9.0 spurious. The anchors recover "
                           "about a third of what the slot costs terra and loosen the "
                           "row at p = 0.062 doing it -- against v13 the arm is still "
                           "gold -2.0 at spurious +3.7."),
        "luna": dict(gold=+0.3, spurious=+4.3, net=-3.3, samples=3, p_gold=1.000,
                     p_spurious=0.438,
                     note="against v14 in the same invocation: word-only 23.3 -> 24.0 "
                          "gold at 17.0 -> 26.3 spurious. Against v13 the arm reads "
                          "gold -0.3 / spurious -0.3, i.e. it gives back the precision "
                          "v14 won."),
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# Iterations 19-20 — the simplification round: one call shape for every case
#
# v13 buys its numbers with three call-level arrangements (`batch_by_evidence`,
# `contract_follows_batch`, `blind_word_only`) that make a word-only candidate travel
# a different prompt from a named one: a different batch, a different contract, a
# different case format. These arms switch all three OFF, so a partial-name case and
# a whole-name case differ **only in the content of the evidence fields they print**
# -- `writes=one word of the name` against `writes=the whole name`, and a populated
# `alternatives`. Same batch, same catalog, same demand, same reply contract, same
# case format, same call count (the batching is by size again, and 90 candidates take
# 4 calls either way).
#
# This is v8's arrangement. v8 read parity on terra and -10.3 gold on luna, the
# word-only row 21.7 -> 12.3: shown the component beside named cases under an approve
# contract, luna judged that row by identity. What v14-v18 added since is a rule text
# that no longer describes a case format it does not use, and a measurement of the one
# sentence that tells the judge what the component slot is FOR on such a case (v17:
# a tightener worth -10.0 word-only spurious on luna). These two arms are that text
# in this arrangement, with and without that sentence.
#
# No new authored prompt bytes: every sentence here was authored and grounded for
# v7-v17, so GATE-06/GATE-07 score exactly what they already scored.
# ─────────────────────────────────────────────────────────────────────────────

ITERATIONS["v19"] = Iteration(
    name="v19", rows=False, blind_word_only=False, verdict="boolean",
    batch_by_evidence=False, contract_follows_batch=False,
    summary="one call shape for every case: no blinding, no evidence batching, no "
            "contract routing -- a partial-name case differs from a whole-name one "
            "only in what its evidence fields say",
    rule=_rowless_rule(_FORMAT_V14, _WRITES_V8),
    demand=_DEMAND_ROWLESS, reply=_REPLY_BOOLEAN,
    fields=("writes", "alternatives", "mention"),
    changed="v8's arrangement with v14's truthful input contract: the three call-level "
            "removals v13 pays its complexity for, all reverted at once",
    measured={
        "terra": dict(gold=-4.0, spurious=+4.7, net=-16.7, samples=3, p_gold=0.344,
                      p_spurious=0.066, p_net=0.199, f2_projected=-1.20,
                      note="against v13 in the same invocation, replicated in a second "
                           "(gold -4.0 again, spurious +4.7 again, F2 -1.38). "
                           "Precision 0.919 -> 0.894; word-only row 22.0 -> 17.7 gold. "
                           "Projected macro F2 -1.2 / -1.4, and ALL of it is one "
                           "project: bigbluebutton -6.2 / -6.4, every other project "
                           "flat or better."),
        "luna": dict(gold=-7.3, spurious=-32.7, net=+10.7, samples=3, p_gold=0.316,
                     p_spurious=0.055, p_net=0.749, f2_projected=+0.36,
                     note="against v13 in the same invocation, replicated in a second "
                           "(gold -7.7, spurious -19.3 at p = 0.031, F2 -0.08). "
                           "Precision 0.752 -> 0.871; word-only row 22.3 -> 14.3 gold "
                           "at 23.7 -> 8.0 spurious. **v8 read -10.3 gold here for "
                           "this arrangement and could not say why**: the loss is not "
                           "the arrangement in general, it is 5-7 named gold pairs, "
                           "all in bigbluebutton, all one generic word of a multi-word "
                           "name ('server' -> HTML5 Server, 'WebRTC' -> WebRTC-SFU) in "
                           "a sentence whose claim is about a phrase the catalog does "
                           "not spell. Composition risk 7 distinct, ABOVE the recorded "
                           "TP floor of 4.8, so an E2E batch can see it and is owed."),
    },
)

ITERATIONS["v20"] = Iteration(
    name="v20", rows=False, blind_word_only=False, verdict="boolean",
    batch_by_evidence=False, contract_follows_batch=False,
    summary="+ the rule says what the component slot is for where the sentence writes "
            "one word of the name: provenance, not a claim under test",
    rule=_rowless_rule(_FORMAT_V15, _WRITES_V8),
    demand=_DEMAND_ROWLESS, reply=_REPLY_BOOLEAN,
    fields=("writes", "alternatives", "mention"),
    changed="the one instruction v17 measured in isolation, moved into the arrangement "
            "that needs it: v19 asks a mixed batch to approve links and shows a "
            "word-only case a component, which is the shape v8 lost 10.3 gold in",
    measured={
        "terra": dict(gold=-5.7, spurious=+4.3, net=-21.3, samples=3, p_gold=0.062,
                      p_spurious=0.156, p_net=0.004, f2_projected=-1.27,
                      note="REFUTED against v13 in the same invocation, and worse than "
                           "v19 beside it: net -21.3 at p = 0.004, the round's only "
                           "significant net loss. The sentence costs the WHOLE-NAME "
                           "row (130.3 -> 127.3) as well as the word-only one."),
        "luna": dict(gold=-10.3, spurious=-24.0, net=-7.0, samples=3, p_gold=0.217,
                     p_spurious=0.047, p_net=0.746, f2_projected=-0.54,
                     note="REFUTED: word-only 22.3 -> 13.3 gold, worse than v19's 14.3 "
                          "beside it. v17 measured this sentence as a TIGHTENER inside "
                          "v13's blind call; in a mixed approve-contract call it "
                          "tightens the row that is already losing and buys nothing. "
                          "An instruction's sign is a property of the call it is read "
                          "in, not of the sentence."),
    },
)

#: `_DEMAND_ROW_AWARE` with the prefix `_DEMAND_ROWLESS` already carries. One demand
#: for every case, applied uniformly; which limb of it a case falls under is decided by
#: the case's own `naming` field, which is content, not structure.
_DEMAND_ROW_AWARE_ALL = "For each case, " + _DEMAND_ROW_AWARE

ITERATIONS["v21"] = Iteration(
    name="v21", rows=False, blind_word_only=False, verdict="boolean",
    batch_by_evidence=False, contract_follows_batch=False,
    summary="+ the one demand names both readings, and the case prints the `naming` "
            "field the demand refers to",
    rule=_rowless_rule(_FORMAT_V14, _WRITES_V8),
    demand=_DEMAND_ROW_AWARE_ALL, reply=_REPLY_BOOLEAN,
    fields=("naming", "writes", "alternatives", "mention"),
    changed="v19's loss is the word-only row answering the claim question instead of "
            "the denotation one: every one of the 5-7 gold pairs it drops is a generic "
            "word of a multi-word name ('server' -> HTML5 Server, 'WebRTC' -> "
            "WebRTC-SFU) whose sentence makes its claim about a phrase the catalog does "
            "not spell. The rule already states the denotation reading (`_WRITES_V8`'s "
            "last sentence); the DEMAND directly under the cases does not, and asks "
            "every case for the architectural claim about the component. This is the "
            "last lever that is field content rather than call structure",
    measured={
        "terra": dict(gold=-7.7, spurious=+4.7, net=-27.7, samples=3, f2_projected=-1.81,
                      note="REFUTED. Against v19 in the same invocation: -3.7 gold at "
                           "+/-0.0 spurious. It buys 0.07 gold a unit on the row it "
                           "targets and costs 0.80 on the whole-name row (131.3 -> "
                           "127.3) -- naming the word-only reading in the demand tells "
                           "the NAMED cases there is a second reading available."),
        "luna": dict(gold=-7.7, spurious=-0.3, net=-22.7, samples=3, p_gold=0.293,
                     p_spurious=1.000, p_net=0.305, f2_projected=-0.33,
                     note="REFUTED, and more sharply: identical word-only gold to v19 "
                          "(13.3 either way, so the demand recovers NOTHING on the "
                          "target row) at whole-name spurious 8.3 -> 24.0. v19's whole "
                          "precision win, given back for nothing."),
    },
)

#: The iteration the variant runs. Override per process with `UNION_ITERATION`.
ACTIVE = "v13"


def active(name: str | None = None) -> Iteration:
    import os
    return ITERATIONS[name or os.environ.get("UNION_ITERATION", ACTIVE)]


def table() -> str:
    """The trail as one table — what changed, and what it read."""
    lines = [f"{'ver':<5}{'rows':<6}{'blind':<7}{'verdict':<9}"
             f"{'terra gold':>11}{'terra sp':>9}{'net':>7}"
             f"{'luna gold':>11}{'luna sp':>9}{'net':>7}  change"]
    for key, it in ITERATIONS.items():
        terra = it.measured.get("terra", {})
        luna = it.measured.get("luna", {})

        def cell(source, field, width):
            value = source.get(field)
            return f"{value:>{width}.1f}" if isinstance(value, (int, float)) \
                else f"{'-':>{width}}"

        lines.append(
            f"{key:<5}{str(it.rows):<6}{str(it.blind_word_only):<7}{it.verdict:<9}"
            f"{cell(terra, 'gold', 11)}{cell(terra, 'spurious', 9)}"
            f"{cell(terra, 'net', 7)}"
            f"{cell(luna, 'gold', 11)}{cell(luna, 'spurious', 9)}"
            f"{cell(luna, 'net', 7)}  {it.changed[:60]}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
    print("\nEvery number is this arm minus the control that ran beside it, three "
          "samples, five projects, per five-project run. net = 3*gold - spurious.")
