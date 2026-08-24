"""S-Linker105 — thinking spent only where the branch's headroom is: the judges.

Every recorded round on this branch pins ``OPENAI_REASONING_EFFORT=none`` -- the
typed, compaction, audit, static, s25-design and null-calibration rounds all set
it explicitly -- so reasoning is an untried cell, not a re-derivation.

Where it should pay is not uniform across the pipeline, and the branch has already
measured which half is starved:

* an **oracle discriminator** is worth +2.4 points of macro F1, so the judging tier
  leaves real headroom on the table;
* the naming proposer is surface-anchored (94% precision) and has nothing to
  deliberate about;
* the refer-back proposer is an inference at 44% precision, and the merge study
  showed its failure is a *raised evidence bar*, not lost ability.

Reasoning effort is a process-wide environment variable, so a run either pays for
thinking on all 17 calls or none of them. This variant makes it selective: the
effort is applied per phase, defaulting to the judges only, so the cost lands on
the decisions rather than on reading names off a page.

**Base.** This builds on ``s_linker101``, not on ``s_linker90``. The head lost the
paired F2 comparison on terra across three runs (93.1/93.8/94.0 against
95.3/95.1/95.6), so it is no longer the arm to extend; s101 keeps the head's two
blind proposers and adds the merged reading as a third, which is what bought that
recall. The luna result is the standing caveat: s101's recall holds there
(92.2 -> 96.9) but its precision falls (84.8 -> 72.7), so the extra candidates
arrive faster than the judges filter them -- which is precisely the tier this
variant spends reasoning on.

It is self-contained. ``_apply_openai_reasoning`` reads the environment at call
time and ``_current_phase`` already reports which phase a call belongs to, so the
variant sets the variable around its own calls and restores it afterwards. No
shared file changes, no prompt changes, and with ``THINK_PHASES`` empty the
behaviour is s101's exactly.

Configure with ``ALINKER_THINK_EFFORT`` (default ``medium``) and
``ALINKER_THINK_PHASES`` (default the three judging phases). Raise
``OPENAI_MAX_COMPLETION_TOKENS`` when using it: hidden reasoning tokens count
against the completion budget, and at the 4096 default they starve the JSON.
"""

from __future__ import annotations

import os

from llm_sad_sam.linkers.experimental.s_linker90 import _current_phase
from llm_sad_sam.linkers.experimental.s_linker101 import SLinker101


class SLinker105(SLinker101):
    """The current best arm, with reasoning spent on the judging phases only."""

    _VARIANT_NAME = "s_linker105"

    #: Substrings of phase names that should be answered with reasoning on.
    THINK_PHASES = tuple(
        p for p in os.environ.get(
            "ALINKER_THINK_PHASES",
            "full_name_judge,partial_denotation,coreference_judge",
        ).split(",") if p.strip()
    )
    THINK_EFFORT = os.environ.get("ALINKER_THINK_EFFORT", "medium")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"SLinker105 (reasoning '{self.THINK_EFFORT}' on phases: "
              f"{', '.join(self.THINK_PHASES) or 'none'})")

    def _ask(self, *args, **kwargs):
        phase = _current_phase()
        if not any(p.strip() in phase for p in self.THINK_PHASES):
            return super()._ask(*args, **kwargs)
        previous = os.environ.get("OPENAI_REASONING_EFFORT")
        os.environ["OPENAI_REASONING_EFFORT"] = self.THINK_EFFORT
        try:
            return super()._ask(*args, **kwargs)
        finally:
            if previous is None:
                os.environ.pop("OPENAI_REASONING_EFFORT", None)
            else:
                os.environ["OPENAI_REASONING_EFFORT"] = previous
