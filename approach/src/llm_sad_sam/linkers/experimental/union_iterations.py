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
