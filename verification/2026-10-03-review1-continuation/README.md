# Review1 continuation, 2026-10-03

Recovered scope from `tmux capture-pane -p -t ardoco-home:review1 -S -350`:
use differences of displayed table scores for percentage-point claims; investigate
RQ3 labels; fix the abstract typo, the project-size claim (#5), and the route-gain
ratio (#12); verify the FSE document class with a subagent.

The earlier pane had fixed the abstract typo, its F2/harmonic gaps, and the intro
F2 gaps before its usage limit. This continuation completes those edits. It also
uses two-decimal score values in the intro and conclusion, states the displayed
score convention, and removes the obsolete abstract numeric comment.

The project-size sentence was already corrected in paper commit b1278c8. It now
states 4–159 thousand lines of code; no further change was needed.

## Verification configuration and commands

Run from the agent-linker root. Arm s126, five ARDoCo projects, GPT-5.6-terra,
three runs for each stochastic system. RQ3 uses rescoring of full runs; RQ4's
no-knowledge results come from separate recorded runs. No model calls or metric
recomputations were needed for these presentation edits. The table inputs and
scores are unchanged.

```sh
python3 verification/2026-10-03-review1-continuation/verify.py > verification/2026-10-03-review1-continuation/check.txt 2>&1
python3 evaluation/mini-src/sync_paper.py --check paper > verification/2026-10-03-review1-continuation/sync.txt 2>&1
mkdir -p /tmp/review1-paper-build
/tmp/rq2-tectonic.grZsvV/tectonic -k --keep-logs -o /tmp/review1-paper-build paper/main.tex > verification/2026-10-03-review1-continuation/build.txt 2>&1
```

- Arithmetic/prose audit: 23 assertions passed. Uses the renderer's displayed
  values before subtraction; source CSV SHA-256 values are in `check.txt`.
- Main and appendix RQ3 tables regenerate exactly; all 4 main rows and 32
  appendix rows retain their numerical cells and bolding.
- Paper synchronization: all 29 generated files match; the two floor artifacts
  are absent for this arm as expected.
- PDF build and layout result: see the build outcome below.

RQ3 now labels rows Full, NoName, NoCitation, NoValidator, matching the experiment
section. A shared generator note explicitly distinguishes active-judge counts
from ablated-configuration scores in both tables. The note also discloses that
counts include deterministic exclusions. This preserves the limitation recorded
in `paper/verification/results-interpretation-2026-10-03/RESULT.md`; it does not
claim the counts isolate LLM judgments or repair the underlying attribution.

## Document class

A dedicated read-only subagent checked the official
[FSE 2027 Research Papers submission instructions](https://conf.researchr.org/track/fse-2027/fse-2027-papers)
on 2026-10-03. The prescribed declaration is:

```latex
\documentclass[acmsmall,screen,review,anonymous]{acmart}
```

`paper/main.tex` matches exactly, so no class change was made.

## Working-tree scope

Other panes have concurrent manuscript, figure, and cost-table edits. They were
preserved and excluded from this task's commits. The PDF and synchronization
checks use the shared working tree, including those edits. Local commits disable
post-commit hooks for this invocation because the configured hooks push to
GitHub and Overleaf; publishing was not part of this request.

## Build outcome

Tectonic completed with exit 0. The PDF has 33 pages; the revised main and appendix
RQ3 tables render on pages 15 and 26. Visual inspection found both labels and
notes readable and within the page. The final log has no undefined-reference or
undefined-citation warnings. Font/PDF warnings and box warnings remain in other
content, including the wide RQ4 appendix tables; see `build.txt` and
`pdf-check.txt`. This check does not certify conference page-limit compliance.

Reproduce the PDF text and diagnostic check with:

```sh
python3 verification/2026-10-03-review1-continuation/verify_pdf.py /tmp/review1-paper-build > verification/2026-10-03-review1-continuation/pdf-check.txt 2>&1
```

The text check uses the environment's existing PyMuPDF package. The arithmetic
and regeneration check uses only the standard library and the project renderer.

The captured build output has trailing whitespace removed; diagnostic text is otherwise unchanged.
