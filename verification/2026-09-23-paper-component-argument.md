# Paper component-level argument check, 2026-09-23

## Scope

Only two sentences were changed for this request: the link-pair versus
project-average distinction in `paper/sections/intro.tex` and the opening of
`paper/sections/metric.tex`. The existing concentration figures, metric
definitions, results, and other prose were retained. The paper submodule
already had uncommitted edits to these two sections and to the gold table
before this request; those edits were preserved.

Inputs: five-project gold distribution in
`paper/table/gold_concentration.csv`; no system arm or new inference run.

## Commands and results

Run from the repository root:

```text
$ python3 evaluation/mini-inequality/inequality.py --check-only
SANITY CHECK PASSED (tol: Gini<=0.005, counts exact)

$ python3 evaluation/mini-src/sync_paper.py --check --only gold paper
IN SYNC: all 2 paper file(s) match the generated output.

$ git -C paper diff --check -- sections/intro.tex sections/metric.tex
(no output; exit 0)
```

The repository's broader numeric-claim audit remains blocked by its stale
line-range policy, a failure already recorded in
`verification/2026-09-22-paper-numeric-audit-failure.md`:

```text
$ python3 scripts/check-paper-numeric-claims.py --self-test
PASS self-test: TeX comments, numeric tokens, citation stripping, and change detection
FAIL paper numeric-claim audit: 50 error(s); 208 active statements
- numeric-statement inventory changed: expected f035029e85319fbaae8338758a65c175829e0054625394ad3d801c38a2307dd7, found 2beafe0753ca969918b2f3217a6bccc27819a59509dc1f42520c2cc3e754949c
exit=1
```

The two edited sentences contain no new numeric result. The full audit needs
an evidence review and updated line classifications for the current paper;
this narrow wording edit does not resolve that existing policy failure.
