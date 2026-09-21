# RQ2 white-space investigation

## Cause

The inserted per-project layout allocates one outer row per project. Its doc-code cell contains three metric rows (link, worst component, and harmonic component), while its doc-model cell contains one combined link/CMR row. The doc-code cell therefore determines the project-row height and leaves the equivalent of two rows unused on the doc-model side.

## Tested redesign

`rq2-system-rows-test.tex` uses one row per approach within each project. The columns carry the task metrics:

- doc-model: link F1/F2 and CMR;
- doc-code: link, worst-component, and harmonic-component F1/F2.

Each project still occupies three lines, so the redesign uses the same basic vertical budget as the inserted layout. Every line contains values for both tasks. The approach order stays fixed within every project, and the deterministic SWATTR-to-TransArc pipeline occupies one row with a footnote identifying the stage used for each task.

## Configuration and commands

```text
Paper class: acmsmall,screen,review,anonymous
Table font: \small
Spacing: \tabcolsep=2pt, \arraystretch=0.98
Scores: two decimals; CMR: one decimal

$ (cd verification && tectonic -Z shell-escape-cwd=. rq2-system-rows-test-document.tex)
PASS: ACM acmsmall build completed with no overfull or underfull boxes
```

The rendered table fits at `\small` without scaling and remains on one page. Visual inspection confirms that every project block uses all three lines on both task sides; no task-specific blank region remains. The render is stored as `rq2-system-rows-test.png`, and the ACM PDF as `rq2-system-rows-test-document.pdf`.

## Alternatives considered

1. Split doc-model link F1, link F2, and CMR into three metric rows. This fills the space with the smallest structural change, but it breaks the F1/F2 pairing used everywhere else in the table.
2. Put approaches in doc-model rows and metrics in doc-code rows. This fills both halves but changes the row meaning at the task boundary.
3. Add another doc-model measurement. No additional measurement is justified by the requested comparison, so this would add scope to solve a layout problem.

The system-row design preserves each F1/F2 pair, gives both task halves the same row meaning, and introduces no additional measurement.
