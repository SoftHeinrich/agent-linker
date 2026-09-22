# Paper review prompt: wording and coverage verification

Date: 2026-09-22 UTC. Scope: the new `docs/PAPER_REVIEW_PROMPT.md`; no manuscript,
prompt for the linker, or benchmark implementation was changed.

Configuration: static, read-only review of the reusable review prompt against
`AGENTS.md`, `approach/CLAUDE.md`, the current paper sources, and
`verification/2026-09-21-approach-end-systematic-review.md`. No model calls,
benchmark runs, or fixed-input semantic audit were needed: this prompt reviews
a paper; it does not alter a linker decision rule. The manuscript itself was
not re-reviewed by this check.

Coverage to check: evidence provenance and invocation pairing; novelty relative
to primary prior work; complete argument chain and cross-section consistency;
the authored-wording gate; readability; actionable, uncertainty-tagged output.

Commands and text results (run from the repository root):

```text
rg -n '^[1-5]\. \*\*|^End with:|^Do not edit' docs/PAPER_REVIEW_PROMPT.md
  26: Evidence and scope
  39: Novelty and positioning
  55: Logical continuity
  70: Method and wording
  83: Readability and presentation
  105: End with
  115: Do not edit the manuscript or run expensive experiments

rg -n 'measured result|invocation set|prior work|missing logical bridge|fixed-input|Readability|before/after|unverified distinction' docs/PAPER_REVIEW_PROMPT.md
  Matched at lines 26, 30, 40, 45, 53, 80, 83, and 110.
  The logical-bridge instruction is at lines 67-68; the literal search term
  spans a line break, so it was not part of that match list.

rg -n 'Artemis|JabRef|MediaStore|s_linker' docs/PAPER_REVIEW_PROMPT.md
  No matches: no benchmark-specific surface forms in the authored criteria.

git add -N docs/PAPER_REVIEW_PROMPT.md verification/2026-09-22-paper-review-prompt.md
git diff --check
  Exit 0; no whitespace errors. Existing modification to the paper submodule
  was left untouched.
```

The initial `git diff --no-index --check /dev/null <new-file>` invocation
returned exit 1 with no diagnostics because `--no-index` reports a difference
from `/dev/null`. The subsequent index-aware `git diff --check` succeeded.
This static verification checks prompt coverage and wording, not whether a
future review will find every problem in a manuscript.
