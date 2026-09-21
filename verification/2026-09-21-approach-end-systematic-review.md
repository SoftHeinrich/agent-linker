# Systematic review: Approach through the end of the paper

Review date: 2026-09-21. Scope: the live text included by `paper/main.tex` from
`sections/approach.tex` through the Data Availability Statement, its five body
tables, and the supplementary appendix files. The introduction and motivation
were consulted only where needed to check cross-section consistency. This is a
review, not an edit of the manuscript. Existing worktree changes were retained.

## Overall assessment

The central workflow and the headline Terra averages are recoverable from the
repository. The present manuscript is not yet submission-ready. The highest-risk
problems are an effectively empty Related Work section, an unavailable cited
replication package, appendix results advertised in the body but excluded from
the submission build, and an RQ3 table whose configuration labels misstate its
rows. Several metric definitions and causal interpretations also exceed the
recorded evidence. Copyediting is needed throughout Evaluation, Results, and
Threats to Validity.

Findings below distinguish a demonstrated contradiction from a claim that needs
evidence. Suggested wording is conceptual: any change to a load-bearing method
criterion should be audited on fixed inputs under the project wording gate.

## 1. Publication and interpretation blockers

| ID | Priority | Location | Finding and evidence | Required action |
| --- | --- | --- | --- | --- |
| B1 | Critical | [Related Work](../paper/sections/rw.tex#L1) | Every substantive paragraph is commented out; the only live content is a bare `\cite{11334548}` at line 54. Readers cannot assess the novelty or baseline choices, including the unqualified “first” claim in [Conclusion](../paper/sections/conclusion.tex#L7). | Write a live, cited synthesis that positions the workflow and metrics against the relevant traceability and LLM approaches; qualify novelty to what the comparison establishes. |
| B2 | Critical | [main.tex](../paper/main.tex#L101), [Results](../paper/sections/results.tex#L11) | `\showappendixfalse` excludes every table in `appendix/detailed-results.tex`, while Results says Luna, per-project, and per-run results “are in appendix.” Those breakdowns are absent from the active submission build. | Include the appendix in the submitted artifact or provide an accessible supplementary artifact and point to it accurately. |
| B3 | Critical | [Data Availability](../paper/main.tex#L180), [bibliography](../paper/agent-linker.bib#L7) | The paper says code, data, raw responses, and results are in the replication package, but its bibliographic record reads `Replication Package: TODOTODO` / `todo todo` and provides no locator. The claimed availability is presently unverifiable from the paper. | Supply a real, accessible artifact citation and reconcile the availability statement with its actual contents. |
| B4 | Critical | [RQ3 body table](../paper/table/rq3-confusion.tex#L13), [table builder](../evaluation/mini-src/rq_tables.py#L290) | The first and last configurations both render as “Full”; the middle rows render as `name judge` and `coreference judge`. In the builder, those middle rows pair a judge's own reject/keep counts with the **pipeline score when that judge is off** (`NoNameValid` / `NoCitation`). The caption and labels do not disclose this split. The same defect repeats in [appendix RQ3](../paper/appendix/rq3-runs.tex#L13). | Label configuration columns “both on / name off / coreference off / both off,” and separately label whose decisions the count columns describe. Regenerate both tables. |
| B5 | High | [numeric audit](../verification/2026-09-21-paper-numeric-claims-check.md), [current Results](../paper/sections/results.tex#L176) | The current numeric-claim checker passes its self-test but fails the manuscript: 200 active numeric statements, 20 audit errors, including the six numeric Results-summary lines and seven Discussion lines. Its digest is stale. This is an **audit coverage failure**, not proof that every number is wrong. | Review the `--report` inventory against the current tables, update the evidence policy only after review, and rerun the gate. |
| B6 | High | [main.tex](../paper/main.tex#L190), [appendix](../paper/appendix/detailed-results.tex#L9), [per-project table](../paper/appendix/big-table-perproject.tex#L15) | The main-file appendix comment still describes “Claude Sonnet.” The appendix says baselines are deterministic and claims the full suite for all four systems “on both backends,” but its table shows three Artemis runs on Terra, a separate GPT-5.4 Artemis row, and no Luna Artemis row. These descriptions misstate the actual comparison. | Rewrite the appendix setup from the current run manifest; identify exactly which system/backend combinations were evaluated. |

## 2. Method and metric validity

| ID | Priority | Location | Finding and evidence | Required action |
| --- | --- | --- | --- | --- |
| M1 | High | [Approach](../paper/sections/approach.tex#L65), [implementation](../approach/src/llm_sad_sam/linkers/experimental/s_linker126.py#L1073) | The paper says the candidate generator derives the `written` value. In the implementation `_union_evidence` calls `_written_as` during judging. This matters because the method's three-stage figure claims an evidence handoff between generation and judging. | Attribute evidence derivation to the judging pass, or describe the stage boundary without assigning this computation to the generator. |
| M2 | High | [Approach](../paper/sections/approach.tex#L65), [implementation](../approach/src/llm_sad_sam/linkers/experimental/s_linker126.py#L1138) | The current prose omits two deterministic candidate-ending rules: a whole written name owns its constituent words, and a residual surface proposed for multiple components is discarded before judging. These rules are present only in commented-out paper lines 66–68. Their omission hides recall-affecting behavior and conflicts with “every candidate” receiving a judge at line 33. | Restore a concise account of both rules; use “remaining/admitted candidates” when discussing judge coverage. |
| M3 | High | [Approach](../paper/sections/approach.tex#L100), [implementation](../approach/src/llm_sad_sam/linkers/experimental/s_linker126.py#L1358) | The text says the antecedent hook verifies an exact name “or via alias,” but the actual accepted forms are `exact` and `alias`; `part` and `qualified name` are excluded. The paper does not state that latter distinction, which is material to the coreference route. | State the accepted forms and excluded forms explicitly, with an evidence-grounded rationale. |
| M4 | High | [metric definition](../paper/sections/metric.tex#L50), [metric implementation](../evaluation/mini-src/metrics.py#L423), [dataset table](../paper/table/gold_concentration.tex#L6) | `\mathcal K` is defined as all architecture-model components and the prose says the suite weights “every component.” The scorer actually takes the **gold-reachable component universe** (`gold_by_c`); the table comment says the same. This changes the denominator and the scope of “every.” | Define `\mathcal K` as components with at least one gold link and report exclusions, or change the scorer and regenerate results. |
| M5 | High | [metric definition](../paper/sections/metric.tex#L29), [CMR formula](../paper/sections/metric.tex#L83) | The suite introduction says it weights every component equally. Worst and harmonic scores do; CMR weights each abandoned component by its number of gold sentence links. The [scorer](../evaluation/mini-src/metrics.py#L490) confirms sentence weighting. | Describe the suite as giving components visibility, then distinguish equal-component doc-code metrics from sentence-weighted CMR. |
| M6 | High | [metric section](../paper/sections/metric.tex#L12) | “A link-level macro average assumes” a uniform per-component distribution is mathematically false. Within a project, the link-level score pools links and large components dominate; the paper's macro average is across projects. The finite-sample Gini maximum is also below 1 for a fixed number of components, so “1 = all links in one component” is only a limiting intuition. | Distinguish within-project pooled link scores from across-project macro averaging; give the Gini endpoint as an approximation or specify normalization. |
| M7 | High | [metric section](../paper/sections/metric.tex#L43), [scorer](../evaluation/mini-src/metrics.py#L173) | The paper says `F_2` weights recall twice as heavily. In the displayed `F_\beta` definition, `\beta=2` gives recall fourfold weight relative to precision in the equivalent error-count weighting. The scorer states `beta^2 = 4`. | Say `F_2` uses `\beta=2` and gives recall fourfold weight; keep the interpretation consistent in Results. |
| M8 | Medium | [metric section](../paper/sections/metric.tex#L75) | “No folder directory expansion” does not imply doc-model component counts lack a long tail. The paper provides no doc-model distribution here, so the claim that tail metrics have “no long tail to expose” does not follow. | Show the doc-model distribution or explain the narrower measured reason for choosing CMR at that grain. |
| M9 | Medium | [metric section](../paper/sections/metric.tex#L66) | The harmonic-mean equation is undefined when any component score is zero; the zero convention comes later in prose. The corpus also permits overlapping file-to-component mappings, so a file link can enter multiple component slices; the formula alone does not make that clear. | Put the zero convention directly next to the formula and define component slices, including overlap. |
| M10 | Medium | [Approach](../paper/sections/approach.tex#L58), [Conclusion](../paper/sections/conclusion.tex#L8) | “Exact mention,” “written name,” “whole name,” “name word,” “alias,” and “qualified name” alternate as if they were one taxonomy. The conclusion lists three forms but leaves aliases and qualified names unaccounted for. | Define one hierarchy: two routes; within the name route, complete names, aliases, and name-word matches, with qualified occurrence as an evidence value. Use it consistently. |

## 3. Experimental design and strength of claims

| ID | Priority | Location | Finding and evidence | Required action |
| --- | --- | --- | --- | --- |
| E1 | High | [Evaluation](../paper/sections/eval.tex#L71), [Results](../paper/sections/results.tex#L11) | The main text says both LLM systems have three runs and the same backend, but does not identify invocation pairing, model settings, aggregation order, or variability in the body. “Same backend” alone does not isolate the workflow. The [appendix per-run table](../paper/appendix/big-table-perrun.tex#L15) contains spread but is excluded. | State the exact comparison configuration and run unit; present variability or per-run data in an available artifact; frame observed mean differences as measured on this setup. |
| E2 | High | [Evaluation](../paper/sections/eval.tex#L155), [run recipe](../evaluation/HOWTO-REGENERATE-RQ.md#L346) | The no-knowledge variant was run in a separate E2E sweep from the full variant. “Both routes remain unchanged, isolating the contribution” implies a paired causal comparison that the run structure does not provide. | Describe it as an unpaired three-run ablation comparison, or run full and no-knowledge in the same invocation before making an isolated-effect claim. |
| E3 | High | [Threats](../paper/sections/discussion.tex#L27) | Rerunning Artemis on the same model does not rule out pretraining exposure: prompt and workflow differences can interact with memorized knowledge. RQ3/4 ablations likewise do not rule out extra LLM calls as the source of the RQ1 gap because no call-matched control is reported for this arm; the [run recipe](../evaluation/HOWTO-REGENERATE-RQ.md#L365) says it has no one-call floor. | Present these as residual threats; use a call-matched control if the paper wants to dismiss the call-budget explanation. |
| E4 | High | [Results summary](../paper/sections/results.tex#L176), [RQ1 table](../paper/table/rq1-results.tex#L13) | “Improves ... at every granularity we measure” is too broad: MediaStore doc-code `F_1` is 0.88 versus Artemis 0.93, and precision is lower than TransArc in the aggregate. The paper itself notes the MediaStore reversal at lines 40–43. | Limit the summary to the stated five-project average and named metrics, retaining the project reversal. |
| E5 | High | [Results](../paper/sections/results.tex#L181) | “The judging layer is worth only half as much without the alias knowledge” is not supported by a live table or analysis; the relevant interaction block is commented out at lines 156–164. Its ablation also draws from separate sweeps. | Reintroduce a traceable interaction analysis with its invocation caveat, or remove this assertion from the summary. |
| E6 | Medium | [Evaluation](../paper/sections/eval.tex#L95) | Artemis is called significantly better than LiSSA and LiSSA's runtime “much more higher,” but no metric, dataset, model, repetitions, test, or measured call-cost comparison accompanies the assertion. The cited paper may support part of it, but the current text does not identify the scope. | Give a scoped, sourced comparison or state the practical baseline-selection criterion without unsupported significance/cost language. |
| E7 | Medium | [Evaluation](../paper/sections/eval.tex#L60), [dataset table](../paper/table/gold_concentration.tex#L23) | “Largest benchmark” and “two orders of magnitude in size” are ambiguous. For example, tabled lines of code range from 4k to 159k (about 40×, under two orders); expanded links range from 59 to 8,268 (about 140×). | Name the size measure and source; retain the statement only for the measure that supports it. |
| E8 | Medium | [Results](../paper/sections/results.tex#L26), [Threats](../paper/sections/discussion.tex#L30) | Several mechanistic sentences read as established causes: candidate generation and judging “make both possible,” transitive gains arise “because” doc-model links improve, and ablations “rule out” extra calls. The tables measure associations and counterfactual rescoring, with one separately rerun variant. | Separate observation, plausible mechanism, and experimental identification. Replace causal language where no isolating comparison exists. |
| E9 | Medium | [Discussion](../paper/sections/discussion.tex#L15) | The proposed dashboard and configurable operating modes are not described in Approach or Evaluation; repository search found “dashboard” only here. This is an unsupported product implication in a paper about the measured workflow. | Document and evaluate the interface, or remove the dashboard claim and keep the discussion to the evaluated link sets. |
| E10 | Medium | [Evaluation](../paper/sections/eval.tex#L71), [Threats](../paper/sections/discussion.tex#L26) | Evaluation says deterministic baselines are run once, while Threats says “every measured system” is the mean of three runs. This is a direct inconsistency even after its syntax is repaired. | State three-run means only for the LLM systems actually repeated and one run for deterministic baselines. |

## 4. Results and table presentation

| ID | Priority | Location | Finding and evidence | Required action |
| --- | --- | --- | --- | --- |
| R1 | High | [Evaluation](../paper/sections/eval.tex#L80) | The intended two-grain list contains `\item Doc-code` followed immediately by a second `\item Following prior evaluations`; the latter becomes an unintended third list item. | Merge the second item into the Doc-code paragraph. |
| R2 | High | [RQ2 prose](../paper/sections/results.tex#L62), [RQ2 table](../paper/table/rq2-results.tex#L32) | The prose names TransArc as a comparator “on both tasks.” The table footnote says the doc-model rows are SWATTR, its deterministic doc-model stage, while TransArc is doc-code only. | Name SWATTR for doc-model and TransArc for doc-code everywhere. |
| R3 | Medium | [RQ2 prose](../paper/sections/results.tex#L64) | `CMR = 0%` means every gold-reachable component has at least one correct link, not necessarily that a system “covers more components” in an absolute count; CMR weights the missed components by gold sentence links. | Explain CMR's meaning precisely and, if claiming component counts, report the counts. |
| R4 | Medium | [RQ3 prose](../paper/sections/results.tex#L99) | Overlap of the two judges' link sets explains why rejection totals do not add, but it does not by itself prove they “made different decisions for the same candidate link” (line 104). The table does not show a cross-judge decision matrix. | Either show overlap/discordance counts or limit the explanation to possible shared candidate links. |
| R5 | Medium | [RQ4 answer](../paper/sections/results.tex#L170), [RQ4 table](../paper/table/rq4-results.tex#L15) | “Removing a route or the knowledge module barely moves link-level F1” conflicts with the table: full-to-no-knowledge doc-code F1 falls from 0.88 to 0.77 (10.9 pp), and doc-model F1 falls about 6.7 pp. | Replace “barely” with the actual changes, separated by module and task. |
| R6 | Medium | [RQ4 prose](../paper/sections/results.tex#L142), [RQ4 source](../evaluation/reports/tex_src/rq4.csv#L2) | Unique true-link counts of 150 and 15 are presented without saying whether they are run means, project totals, or pooled unique links. The source has single integer fields, whereas other RQ4 scores are three-run means. | Define the counting unit and aggregation; avoid placing these integers beside mean performance as if they have identical run scope. |
| R7 | Medium | [RQ2 table](../paper/table/rq2-results.tex#L9) | The table uses two side-by-side project panels and repeats its header bands. The method is valid, but a 21-column `tabular*` inside `\linewidth` needs a visual PDF check; the current environment has no TeX engine. | Inspect the rendered page for legibility, clipping, and correspondence between the left/right project labels and values before submission. |
| R8 | Low | [Conclusion](../paper/sections/conclusion.tex#L14) | “The standard metric would call the task saturated” has no defined saturation criterion; doc-code mean F1 is 0.882. | State only the measured contrast between aggregate and weakest-component scores. |

## 5. Language, structure, and terminology pass

These are copyedits unless they alter a decision rule; retain the original
meaning and rerun the fixed-input audit for any load-bearing method rewrite.

| Section | Location | Text problem | Suggested repair |
| --- | --- | --- | --- |
| Approach | [lines 8–15](../paper/sections/approach.tex#L8) | Enumeration items start `the`, `references`, then `Third`; line 12 is a run-on list and the quote is malformed. | Use parallel item openings and punctuate the alternatives. |
| Approach | [line 33](../paper/sections/approach.tex#L33) | “forming a evidence-backed judging” is ungrammatical. | “forming an evidence-backed judging stage” or “providing evidence-backed judgment.” |
| Approach | [line 60](../paper/sections/approach.tex#L60) | “exact mention” is too narrow for aliases, inflections, and qualified identifiers described below. | “written expression in the target sentence.” |
| Approach | [lines 100–105](../paper/sections/approach.tex#L100) | “These candidate”; “writes either ... or via alias”; “pre-hook” is unexplained jargon. | Use “These candidates,” “writes the exact name or an approved alias,” and “deterministic check.” |
| Metric | [lines 10, 21–23](../paper/sections/metric.tex#L10) | Repeated “expansion that expands”; abrupt architect/benchmark aside; “regardless the benchmark score.” | Define expansion once and use “regardless of the benchmark score.” |
| Metric | [lines 29–37](../paper/sections/metric.tex#L29) | “its automatically,” “a system recover,” “component lost where a system never reaches”; three items are written as a dangling sentence after three claimed advantages. | Correct agreement and give the three metrics their own grammatically parallel list. |
| Metric | [lines 60–79](../paper/sections/metric.tex#L60) | “Same as Fβ, Worst-component ...”; capital `The` after semicolon; “Let Rk for ...”; unclear “linked sentences.” | Normalize case, punctuation, and mathematical definitions. |
| Evaluation | [lines 21–32](../paper/sections/eval.tex#L21) | “recover combining”; missing “the” in “both doc-model recovery task”; “We found that the long-tailed distribution ...”; subject disagreement in “misses which is difficult.” | Rewrite as short grammatical statements without altering the question scope. |
| Evaluation | [lines 53–74](../paper/sections/eval.tex#L53) | “two reference forms” blurs route/form distinction; “the the”; “trace-links”; “Because ... therefore”; mixed `ArDoCo` / `ardoco` presentation. | Use the route/form taxonomy, remove duplicate and double connective, and set one proper-name style. |
| Evaluation | [lines 81–100](../paper/sections/eval.tex#L81) | “a system recover”; “the same approach ArCoTL”; “has shown be”; “much more higher”; redundant `\item` (R1). | Edit for grammar and specify what ArCoTL provides. |
| Evaluation | [lines 103–114](../paper/sections/eval.tex#L103) | “suite add”; “sentences describing it” after plural components. | “suite adds”; “sentences describing each component.” |
| Results | [lines 22–43](../paper/sections/results.tex#L22) | “similiar”; “extracting mention”; “gain is through both precision and recall improvement”; “This is a case that ... loses at doc-code part”; inconsistent `+1.3 pp` versus rounded `+2 pp` for MediaStore. | Correct grammar; use one precision for the same project contrast. |
| Results | [lines 92–127](../paper/sections/results.tex#L92) | “Their link counts overlap and therefore does not sum”; “while cost”; table/prose mixes judge counts and off-config scores. | Correct agreement and explain the distinct row grains before interpreting numbers. |
| Results | [lines 140–154](../paper/sections/results.tex#L140) | “That is because its scope”; bare `6.7pp`; “linkded”; unmatched double quote around `Reencoding`; “its real weight” vague. | Use complete clauses, space `pp`, correct spelling and TeX quotation, name the measured metric. |
| Discussion | [lines 23–31](../paper/sections/discussion.tex#L23) | “Two factors” introduces nondeterminism, memorization, and extra calls; “approach{}” lacks backslash; “re-ran base line approach Artemis”; “Therefore, two systems then”; `Second` followed by `Secondly`. | Rebuild the paragraph as three separately scoped threats and mitigations. |
| Discussion | [lines 38–42](../paper/sections/discussion.tex#L38) | “weighted missed component rates” misdefines singular CMR; sentence fragment across line 39/40. | Define CMR as a sentence-weighted share of entirely missed gold components. |
| Related Work | [entire section](../paper/sections/rw.tex#L1) | No live prose; a naked citation is not a section. | Restore a focused synthesis, then copyedit the live text. |
| Conclusion | [lines 7–17](../paper/sections/conclusion.tex#L7) | Scope shifts from document-to-model in the first sentence to doc-code in the headline; line 13 is an overloaded sentence; “worst component still sits at F1 0.77” omits that this is a mean of project minima. | Name the primary recovery task and downstream composition; split the result sentence and name the aggregation. |

### Terminology decisions to apply consistently

1. Use **doc-model** for sentence-to-component recovery and **doc-code** for
   sentence-to-file links composed through component-to-file mappings. Do not
   call a doc-code metric “file F1” in one place and “link-level F1” elsewhere
   without first defining them as the same scored links.
2. Keep **route** for the two end-to-end paths, **generator** for proposing
   candidates, and **judge** for accepting them. `\linkerN` and `\linkerC` still
   render “name linker”/“coreference linker” in generated RQ4 tables; change the
   display mapping when the table generator is next revised.
3. Use **evidence-backed judge** consistently; Discussion says
   “evidence-based judges” while Approach and Evaluation use the former.
4. Reserve **average F1/F2** for the five-project arithmetic mean, **link-level
   F1/F2** for the within-project scored links, **worst-component** and
   **harmonic-mean per-component** for doc-code, and **CMR** for doc-model.
5. Distinguish **gold-reachable component** from all model components. This
   distinction affects the metric universe and the meaning of “missed.”

## Verification performed and limits

Commands were run from the repository root with the current worktree, without
changing manuscript or generated tables:

```text
python3 scripts/check-paper-numeric-claims.py --self-test
  self-test PASS; paper audit FAIL (20 errors; 200 active numeric statements;
  current inventory digest differs from policy)

PAPER_DIR="$PWD/paper" python3 evaluation/mini-src/sync_paper.py --check
  exit 0; all 26 generated paper files in sync (two absent floor files expected
  for this arm)

git -C paper diff --check && git diff --check
  exit 0; no whitespace errors in current diffs

command -v tectonic; command -v pdflatex; command -v latexmk
  none found; no fresh PDF build or visual table check was possible

./scripts/build-paper.sh
  exit 1: latexmk is required to build the paper

source-link check (all relative report links and line anchors)
  77 links checked; 0 missing files or out-of-range line anchors
```

The sync result proves generated TeX matches its current generator output; it
does not validate labels, prose, metric definitions, or numerical evidence. A
stale `paper/main.log` was not treated as a current build result. Bibliographic
claims and priority/novelty comparisons were checked only for internal support,
not independently against external publications. The original user worktree
modifications, including dirty figure sources in the `paper` submodule, were
left untouched.

## Recommended repair order

1. Make the active artifact complete: Related Work, appendix availability,
   replication citation, RQ3 labels, and a fresh PDF build.
2. Correct metric universe and definitions, then audit all cross-section terms
   and table captions against the scorer.
3. Reframe causal and generality claims to the measured arm, five projects,
   two backends, three runs, and the correct paired or unpaired comparison.
4. Copyedit the live prose and regenerate any changed tables through the
   established CSV-to-TeX pipeline.
5. Rerun the numeric-claim audit after reviewing every changed statement,
   then inspect the final rendered paper page by page.
