"""S-Linker116 — the lenient gate is asked for the ground, like the strict one.

Three judges, and today two reply schemas. The strict gate writes `claim`,
`objection`, `approve`; the lenient gate writes `claim`, `approve`; the sortal gate
writes `denotation`, `claim`. The question this arm asks is the first half of
"can all three write one thing": what does the lenient gate lose or gain when its
reply carries the strict gate's `objection` field?

`s_linker92` states the reason it does not, and states it as an argument rather than a
measurement: *"Approve by default and state the strongest ground for rejecting are
contradictory standards to put in one prompt, and only the strict arm was measured."*
The argument is not obviously right. The strict gate's own decide clause does not say
"reject unless" — it says *approve unless that ground is one the rules above make
decisive*, which is the lenient polarity written out, with the ground named first. So
the arm here is the head's lenient rubric with the head's strict decide clause and the
head's strict reply field, and nothing else: same rules, same `QUALIFIED_CLAUSE`, same
`STRICTER_CLAUSE`, same cases, same batch size, same polarity, same catalog.

The two strings are copied from `s_linker92._prompt_validation` verbatim, where they
already run against every coreference batch this pipeline judges. Nothing is worded
here.

**What it costs, priced before it ran** (`pilot/objection_audit.py`, six recorded
five-project runs). The strict gate's `objection` averages 78 characters on terra and
85 on luna — 5 and 22 on the rows it approves, 112 and 104 on the rows it rejects. The
two gates that would gain the field judge 300.3 / 305.7 cases a run, so a uniform
schema buys its uniformity for **+5.9k / +6.5k completion tokens a run against the
28.6k judging already spends: +20% / +23%, not a saving.** A schema unification has to
pay for itself in verdicts, and this arm is how that is measured.

GATE-01: `s_linker92` is untouched. GATE-06/07: every clause and every rubric string is
imported from, or copied verbatim out of, the module that measured it.
"""

from __future__ import annotations

from dataclasses import replace

from llm_sad_sam.linkers.experimental.s_linker92 import (
    LAYERED_ENTITY_RULES, QUALIFIED_CLAUSE, STRICTER_CLAUSE,
)
from llm_sad_sam.linkers.experimental.s_linker114 import SLinker114


class SLinker116(SLinker114):
    """The head, with the lenient gate's reply carrying the strict gate's ground."""

    _VARIANT_NAME = "s_linker116"

    #: `s_linker92._prompt_validation`'s strict decide clause, verbatim. It names the
    #: ground before the verdict and keeps the approve-unless polarity.
    DECIDE = (" then state the strongest ground there is for rejecting this case under "
              "the\nrules above (or \"none\" if there is none), then decide: approve "
              "unless that ground is one\nthe rules above make decisive. An objection "
              "you could raise against most sentences is not\na ground for rejecting "
              "this one.")

    #: The same module's strict reply field, verbatim, in the same place.
    FIELD = ', "objection": "<strongest ground to reject, or none>"'

    #: The lenient decide clause it replaces, also verbatim, so the substitution
    #: either fires exactly once or the arm refuses to build a prompt at all.
    LENIENT_DECIDE = " then decide approve true/false based on that claim."

    #: Where the field goes: the head writes it in this position at the strict gate.
    SCHEMA_ANCHOR = '"claim": "<exact quote or none>", "approve": true'

    @classmethod
    def _prompt_entity_ground(cls, linker, comp_names, cases) -> str:
        """The head's own lenient prompt, with two substitutions and nothing else.

        Built by calling the head's builder and editing its output, not by copying its
        template: a copy drifts the moment a rubric changes upstream, and the point of
        the arm is that the rubric, the clauses, the cases and the polarity are the
        head's. Each substitution must fire exactly once or the prompt is refused.
        """
        prompt = linker._prompt_validation(comp_names, cases, "", strict=False)
        for old, new in ((cls.LENIENT_DECIDE, cls.DECIDE),
                         (cls.SCHEMA_ANCHOR,
                          cls.SCHEMA_ANCHOR.replace(', "approve"',
                                                    f'{cls.FIELD}, "approve"'))):
            if prompt.count(old) != 1:
                raise RuntimeError(
                    f"s_linker116: {prompt.count(old)} sites for a substitution that "
                    f"must have exactly one -- the head's lenient prompt moved")
            prompt = prompt.replace(old, new, 1)
        return prompt

    @staticmethod
    def _entity_decision(item, verdict, keep, context):
        """The head's row, plus the ground the reply now carries."""
        stage = context["stage_label"]
        return {"approved": keep, "claim": verdict["claim"],
                "objection": verdict["objection"],
                "path": f"{stage}_judged" if keep else f"{stage}_rejected",
                "stage": f"{stage}_judge"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker116 (lenient gate: the ground is asked for, as at the strict one)")


SLinker116.ENTITY = replace(
    SLinker114.ENTITY,
    prompt=lambda linker, names, cases, shared, ctx:
        SLinker116._prompt_entity_ground(linker, names, cases),
    decision=SLinker116._entity_decision,
)
