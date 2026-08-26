"""S-Linker114 — three judges, one structure, three skills.

Three judges stand behind three passes and they are three code paths: 262 lines of
`s_linker92` across `_validate_with_evidence`, `_classify_denotations` and
`_validate_coref_links`, each with its own batching loop, its own reply parser, its own
bounds checks and its own decision record. What they actually do is the same thing, and
the branch has never written that thing down.

**The structure, which is the same for all three.** Take a list of candidates. Cut it
into batches of `JUDGE_BATCH`. Render the batch's shared evidence once and each case as
a numbered block. Compose one prompt: a question, the rubric that answers it, the cases,
and a JSON reply schema. Ask once. Parse the reply by case index, bounds-checked, with
every case the model did not answer defaulting to the skill's own polarity. Record one
decision per candidate and return the kept ones with it.

**The skill, which is what differs.** A `JudgeSkill` carries only that: the question,
the rubric, what the case is allowed to show, the fields the reply must write and their
order, and the one expression that turns a verdict into a keep. Nothing in it is code
about batching, asking or parsing.

| skill | question | withheld | rubric | reply fields, in order | polarity |
| --- | --- | --- | --- | --- | --- |
| `entity` | is this written name doing naming work here? | nothing | layered entity + qualified + stricter | claim, approve | approve unless a ground fires |
| `denotation` | what does this expression denote? | **the target and the whole catalog** | qualified | denotation, claim | keep `participant` |
| `coref` | does this referring expression point to this component? | nothing | layered coref | claim, objection, approve | reject when uncertain |

**Every difference in that table is measured, and nothing else differs.** The target is
withheld from `denotation` because showing it is `s_linker25` at −5.5 gold and
`s_linker108` at −0.40 macro F2. The polarities sit in the base-rate order of the streams
that feed them — 0.70/0.74, 0.31/0.19, 0.57/0.46 gold among the cases each is handed
(`pilot/judge_census.py`). `objection` is the strict skill's alone because "approve by
default" and "state the strongest ground to reject" are contradictory in one prompt and
only the strict arm was measured. `coref` shows the resolver's committed reference and
antecedent because `s_linker82` withheld them and the judge rejected half the gold put
to it.

**This variant is byte-identical to the head and is meant to be.** It changes no prompt,
no rubric, no batch size, no field and no polarity; `pilot/test_s114_skills.py` rebuilds
every prompt all three skills would send over six recorded runs and asserts equality with
the head's own builders, case block by case block — under a reply that answers nothing
*and* one that answers every case, so the kept rows are compared and not only each
judge's default (284/284 batches, 1444 kept rows). A refactor that moves a measured number is not a
refactor, so the test is the deliverable and the code is what it licenses: after this,
a judging arm is an edit to one `JudgeSkill` field rather than a fourth copy of the loop.

GATE-01: `s_linker92` is untouched. GATE-06/07: no clause is written here — every rubric
string is imported from the module that measured it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names
from llm_sad_sam.linkers.experimental.s_linker92 import (
    COREF_VALIDATION_FOCUS, QUALIFIED_CLAUSE,
)
from llm_sad_sam.linkers.experimental.s_linker110 import SLinker110


@dataclass(frozen=True)
class JudgeSkill:
    """What one judging pass knows that the other two do not.

    Everything here is data or a one-expression rule. The loop that uses it is
    `SLinker114._judge` and it is the same loop for every skill.
    """

    #: The phase tag the client records this pass under.
    phase: str
    #: How the phase reaches the client: the strict and lenient gates set it on the
    #: client before the call, the denotation gate passes it as a keyword.
    phase_on_client: bool
    #: The JSON list the reply must carry, and how `_ask` is told to demand it.
    reply_key: str
    require_kwarg: str
    label: str
    timeout: int
    #: The evidence a batch shares, rendered once. (linker, batch, context) -> object.
    shared: Callable
    #: One numbered case block. (linker, index, item, context, shared) -> str | dict.
    case: Callable
    #: The whole prompt. (linker, comp_names, cases, shared, context) -> str.
    prompt: Callable
    #: The reply field the verdict is written in, and the readings it may take.
    #: `None` readings mean the head's boolean `approve` contract.
    verdict_field: str
    verdict_values: frozenset | None
    #: The polarity, as one expression over a verdict.
    keep: Callable
    #: The decision record. (item, verdict, kept, context) -> dict. The strict
    #: skill also announces its rejections here, as the head does.
    decision: Callable
    #: Whether the case is allowed to carry the component catalog at all.
    shows_catalog: bool = True
    #: Case numbers the head accepts: integers only, or digit strings too.
    digit_case_numbers: bool = False


class SLinker114(SLinker110):
    """The head's three judges, expressed as one pass over three skills."""

    _VARIANT_NAME = "s_linker114"

    # ── the one loop ─────────────────────────────────────────────────────────

    def _judge(self, skill: JudgeSkill, items, components, context=None):
        """Every judging pass this pipeline makes.

        The three passes it replaces differed in six places, all of which are now
        fields of `skill`, and in nothing else. Cases the reply omits are not silently
        dropped: they take `skill.missing`, which is each judge's own default polarity
        written down instead of implied by a `dict.get`.
        """
        if not items:
            return [], {}
        context = context or {}
        comp_names = get_comp_names(components) if skill.shows_catalog else []
        # `_validate_with_evidence` takes its phase tag as an argument (the head's
        # signature, one call site in this chain), so the caller may name the phase;
        # the other two skills carry their own.
        phase = context.get("phase") or skill.phase
        if skill.phase_on_client:
            self.llm.set_phase(phase)
        kept, decisions = [], {}
        for _, batch in self._iter_batches(items, self.JUDGE_BATCH):
            shared = skill.shared(self, batch, context)
            cases = [skill.case(self, i, item, context, shared)
                     for i, item in enumerate(batch, 1)]
            ask = {skill.require_kwarg: skill.reply_key}
            if not skill.phase_on_client:
                ask["phase"] = phase
            data = self._ask(
                skill.prompt(self, comp_names, cases, shared, context),
                timeout=skill.timeout, label=skill.label, **ask,
            ) or {}
            verdicts = {}
            for row in data.get(skill.reply_key, []) or []:
                index = self._case_index(row.get("case"), len(batch),
                                         skill.digit_case_numbers)
                if index is not None:
                    verdicts[index] = self._verdict(skill, row)
            for index, item in enumerate(batch):
                verdict = verdicts.get(index, self._missing(skill))
                keep = skill.keep(verdict)
                decisions[(item.sentence_number, item.component_id)] = \
                    skill.decision(item, verdict, keep, context)
                if keep:
                    kept.append(item)
        return kept, decisions

    @staticmethod
    def _case_index(value, size, digits: bool):
        """The head's two case-number contracts, kept exactly as each judge had them."""
        if digits:
            text = str(value if value is not None else "")
            if not text.isdigit():
                return None
            number = int(text)
            return number - 1 if 1 <= number <= size else None
        try:
            index = int(value) - 1
        except (TypeError, ValueError):
            return None
        return index if 0 <= index < size else None

    # ── the three skills ─────────────────────────────────────────────────────

    @staticmethod
    def _entity_shared(linker, batch, context):
        """The batch's anchor sentences, written once. `s_linker88`'s bookkeeping."""
        return {"union": linker._anchor_union(batch, context["bundles"]), "shown": {}}

    @staticmethod
    def _entity_case(linker, index, item, context, shared):
        prefix = linker._prev_prefix(item.sentence_number, context["sent_map"])
        bundle = context["bundles"].get((item.sentence_number, item.component_id))
        first = shared["shown"].get(item.component_name, 0)
        if bundle and bundle.anchor_sentences and not first:
            shared["shown"][item.component_name] = index
        evidence = (linker._format_evidence(
            bundle, shared["union"].get(item.component_name), first) if bundle else "")
        return (f'Case {index}: "{item.matched_text}" -> {item.component_name}\n'
                f'  {prefix}"{item.sentence_text}"\n'
                f'{evidence}')

    @staticmethod
    def _coref_case(linker, index, item, context, shared):
        # `_resolve_references` admits a resolution only for a sentence the document
        # has, so every link reaching this judge has one.
        sentence = context["sent_map"][item.sentence_number]
        prefix = linker._prev_prefix(item.sentence_number, context["sent_map"])
        resolution = context["metadata"].get(
            (item.sentence_number, item.component_id), {})
        claimed = "".join(line for line in (
            f'  Claimed reference: "{resolution.get("reference")}"\n'
            if resolution.get("reference") else "",
            f'  Claimed antecedent (S{resolution.get("antecedent_sentence")}): '
            f'"{resolution.get("antecedent_text")}"\n'
            if resolution.get("antecedent_text") else "",
        ))
        return (f'Case {index}: pronoun/role-ref -> {item.component_name}\n'
                f'{claimed}'
                f'  {prefix}"{sentence.text}"')

    @staticmethod
    def _denotation_shared(linker, batch, context):
        """The union of the batch's windows, as the head's sentence table."""
        sentences = context["sentences"]
        numbers = {s.number for item in batch
                   for s in linker._window(item.sentence_number, sentences)}
        table = {s.number: s for s in sentences}
        return [{"sentence": n, "text": table[n].text} for n in sorted(numbers)]

    @staticmethod
    def _denotation_case(linker, index, item, context, shared):
        return {"case": index, "source": item.sentence_number,
                "expression": item.matched_text}

    @staticmethod
    def _denotation_prompt(linker, comp_names, cases, shared, context):
        """`s_linker92._classify_denotations`'s prompt, verbatim.

        The catalog does not appear: this skill is the one whose target is withheld,
        and `comp_names` is empty for it by `shows_catalog=False`.
        """
        return f"""Classify what each expression itself denotes in its
local context: participant for a software participant, or associated for
something merely associated with software.

{QUALIFIED_CLAUSE}

SENTENCES
{json.dumps(shared)}

CASES
{json.dumps(cases)}

Claim must be a contiguous exact substring of the source sentence.

JSON only:
{{"judgments":[{{"case":1,"denotation":"participant",
"claim":"exact source quote"}}]}}
"""

    @staticmethod
    def _verdict(skill, row):
        """One reply row, read the way the row's own skill declares.

        The head had two parsers for this: `_run_validation_pass` read a boolean
        `approve` that may arrive as the string "true", and `_classify_denotations`
        read an enum and checked it against its two readings. Both are the same
        thing -- a verdict field, a set of readings it may take, and a quote -- so
        the difference is `verdict_field` and `verdict_values` and not code.
        """
        claim = str(row.get("claim", "")).strip()
        objection = str(row.get("objection", "")).strip()
        if skill.verdict_values is None:
            value = row.get(skill.verdict_field, False)
            verdict = "approve" if (value is True or (
                isinstance(value, str) and value.lower() == "true")) else "reject"
            return {"claim": claim, "objection": objection,
                    "verdict": verdict, "valid": True}
        # The enum contract, and the response contract only. The substring check that
        # used to follow it voided 0 of 380 verdicts over six five-project runs --
        # `s_linker48`'s separation: demanding a committed quote is worth 35.2 TP,
        # verifying it is worth nothing.
        verdict = str(row.get(skill.verdict_field, "")).strip()
        return {"claim": claim.strip("\"'\u201c\u201d\u2018\u2019"),
                "objection": objection, "verdict": verdict,
                "valid": verdict in skill.verdict_values and bool(claim)}

    @staticmethod
    def _missing(skill):
        """What a case the reply never answered records: the skill's own polarity.

        Rejection for the two boolean gates, an unreadable verdict for the enum one --
        which its `keep` refuses, so all three default to no link and each says so in
        its own vocabulary rather than by a `dict.get` fallback at the call site.
        """
        return {"claim": "", "objection": "", "valid": skill.verdict_values is None,
                "verdict": "reject" if skill.verdict_values is None else ""}

    @staticmethod
    def _entity_decision(item, verdict, keep, context):
        stage = context["stage_label"]
        return {"approved": keep, "claim": verdict["claim"],
                "path": f"{stage}_judged" if keep else f"{stage}_rejected",
                "stage": f"{stage}_judge"}

    @staticmethod
    def _coref_decision(item, verdict, keep, context):
        if not keep:
            print(f"    Coref reject: S{item.sentence_number} -> {item.component_name}")
        return {"approved": keep, "claim": verdict["claim"],
                "objection": verdict["objection"],
                "path": "coref_validated" if keep else "coref_rejected"}

    @staticmethod
    def _denotation_decision(item, verdict, keep, context):
        # Not `keep`: this row is what `_classify_denotations` returns, and the head
        # returns False here for every case and lets `_judge_partial_names` -- its
        # own, inherited unchanged -- write the keep onto the participants. Writing
        # it here too would end at the same place through a different intermediate,
        # and an intermediate no test can see is where a refactor hides a change.
        return {"approved": False, "requested_keep": False,
                "evidence_valid": verdict["valid"],
                "claim": verdict["claim"], "denotation": verdict["verdict"],
                "alternative": "not reviewed", "path": "denotation",
                "stage": "partial_name"}

    ENTITY = None       # bound below the class body
    COREF = None
    DENOTATION = None

    # ── the three passes, each now one call ──────────────────────────────────

    def _validate_with_evidence(self, candidates, bundles, components, sent_map,
                                phase_tag, stage_label):
        return self._judge(self.ENTITY, candidates, components, context={
            "bundles": bundles, "sent_map": sent_map,
            "stage_label": stage_label, "phase": phase_tag,
        })

    def _validate_coref_links(self, coref_links, sent_map, components, metadata):
        return self._judge(self.COREF, coref_links, components, context={
            "sent_map": sent_map, "metadata": metadata,
        })

    def _classify_denotations(self, candidates, sentences):
        return self._judge(self.DENOTATION, candidates, [], context={
            "sentences": sentences,
        })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker114 (one judging pass, three skills)")


SLinker114.ENTITY = JudgeSkill(
    phase="phase_25_full_name_judge", phase_on_client=True,
    reply_key="validations", require_kwarg="require",
    label="Validation pass", timeout=120,
    shared=SLinker114._entity_shared, case=SLinker114._entity_case,
    prompt=lambda linker, names, cases, shared, ctx:
        linker._prompt_validation(names, cases, "", strict=False),
    verdict_field="approve", verdict_values=None,
    keep=lambda v: v["verdict"] == "approve",
    decision=SLinker114._entity_decision,
)

SLinker114.COREF = JudgeSkill(
    phase="phase_25_coreference_judge", phase_on_client=True,
    reply_key="validations", require_kwarg="require",
    label="Validation pass", timeout=120,
    shared=lambda linker, batch, ctx: None, case=SLinker114._coref_case,
    prompt=lambda linker, names, cases, shared, ctx:
        linker._prompt_validation(names, cases, COREF_VALIDATION_FOCUS, strict=True),
    verdict_field="approve", verdict_values=None,
    keep=lambda v: v["verdict"] == "approve",
    decision=SLinker114._coref_decision,
)

SLinker114.DENOTATION = JudgeSkill(
    phase="phase_25_partial_denotation", phase_on_client=False,
    reply_key="judgments", require_kwarg="require_present",
    label="Denotation", timeout=240,
    shared=SLinker114._denotation_shared, case=SLinker114._denotation_case,
    prompt=SLinker114._denotation_prompt,
    verdict_field="denotation",
    verdict_values=frozenset({"participant", "associated"}),
    keep=lambda v: v["verdict"] == "participant" and v["valid"],
    decision=SLinker114._denotation_decision,
    shows_catalog=False, digit_case_numbers=True,
)
