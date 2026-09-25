# Resumed dev2 work — 2026-09-24

Scope: s126/terra quality results on MediaStore, TeaStore, Teammates,
BigBlueButton, and JabRef; three runs per project. MediaStore uses the three
September 24 replacement runs, other projects the September 16 runs.
Artemis quality remains the prior terra cohort. The token-only comparison
uses the new September 24 Artemis/luna cohort (three runs per project).
No new model calls were made during this continuation.

The RQ1 table has DM/DC rows, inline sample SD for P and R, and F1/F2.
SD is computed in rq12.py before reshaping. SD requires three observations;
missing data cannot silently become zero. Its Average row uses SD across
run-wise project means. The exported scores are rounded to four decimals;
the independent SD check allows 0.000062 error for that rounding (sqrt(3/2)
times 0.00005, plus SD serialization error).

The token engine counts reported usage once per response, includes logged
repair responses, and writes source paths and SHA-256 hashes. User correction:
the paper table contains input/output tokens only, no calls or timings.
The original dev2 cost estimate had apparently summed duplicated call records;
the retained per-call JSON gives 98.562k input and 22.691k output for s126,
versus 20.034333k input and 23.368k output for Artemis. These are per-benchmark
means across three runs, not monetary prices or a controlled model comparison.

Original MediaStore links, phase states, and JSON are retained alongside the
replacement slots. The user requested replacing the whole three-run set.
A network fault was suspected in the prior conversation but never established.
The paper now discloses selection after inspecting outcomes and the separate
no-knowledge invocation set; identical replacement scores do not establish
stability or invalidate the original measurements.

Authored headline values in the abstract, introduction, results, and conclusion
were checked against the regenerated CSVs. Results states that whole-pp gaps
are rounded from the underlying scores; the headline sections use one decimal
pp. RQ3 uses the replacement candidate sets, and RQ4's unique named-route count
is now 151. The repeated numerical Summary is replaced by token usage,
observed variance, and bounded interpretation. The original gold-link shares
for DB/Reencoding were reversed in prose; the corrected ordering follows the
existing causal-claims report.

Commands, configuration, text output, and exit statuses are in the adjacent
logs. `focused.txt` checks SD, replacement source equality, backups, token
means/totals, table shape, and sync. `goldens.txt` checks the metric core;
`reproduce.txt` regenerates all engine CSVs including SD and token usage;
`sync.txt` compares paper tables. `build.txt` and `numeric-ledger.txt` retain
failures as well as successes. The manuscript-wide numeric ledger predates
these edits and has stale line scopes/digest across multiple sections; it is
not refreshed automatically as if those other claims had been reviewed.

Local commits disable the automatic post-commit publishing hook for that
command only; this continuation does not authorize publishing to Overleaf.

Final checks: focused verification PASS; metric goldens PASS; all engine CSVs
reproduce byte for byte; all 28 available paper files are in sync. PDF build
is blocked by missing latexmk. The global numeric ledger fails with 44 errors
(201 active statements), including pre-existing unreviewed Introduction and
Discussion scopes. These failures are recorded, not treated as passed gates.
A font-metric estimate puts the revised RQ1 table below the available width,
but does not substitute for compiling and inspecting the PDF.

Raw retained logs/backups preserve their original whitespace. Full staged
`git diff --check` reports whitespace in those source artifacts; authored-code
and prose checks pass. New generated token CSVs use LF line endings.
