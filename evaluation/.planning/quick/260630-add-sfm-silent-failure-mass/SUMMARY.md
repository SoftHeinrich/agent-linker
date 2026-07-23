---
task: add-sfm-silent-failure-mass
type: quick
status: complete
created: 2026-06-30
completed: 2026-06-30
repos:
  - infra: /mnt/hostshare/ardoco-home/transarc-emp   # branch mini, commit 1c3c493
  - paper: /mnt/hostshare/ardoco-home/alinker-paper  # branch sfm-doc-model-metric, commit 79011b8
commits:
  infra: 1c3c493   # feat(metrics): add doc-model Silent-Failure Mass (SFM/SFC)
  paper: 79011b8   # paper: report Silent-Failure Mass (SFM) as the doc-model size-aware metric
---

# Add Silent-Failure Mass (SFM) as the doc-model size-aware metric

One-liner: introduced **Silent-Failure Mass (SFM)** + integer companion **SFC** as the
**doc-model-only** size-aware metric (distinct-sentence denominator), wired through the eval
infra/tables (transarc-emp) and reported across the paper (alinker-paper), keeping the doc-code
F1-tail suite ("three size-aware metrics": coverage, worst-component F1, harmonic-mean F1)
entirely unchanged.

## SFM definition (canonical, distinct-sentence denominator)

For each gold component `k`, a component is **ABANDONED** iff the system recovers zero correct
links for it (`recall_k = 0`). Let `S_k` be the set of DISTINCT documentation sentences gold-linked
to `k`. Then:

    SFC = #{ k : k abandoned }                                              (integer companion)
    SFM = |union of S_k over abandoned k| / |union of S_k over all k| x 100  (in [0,100], report %)

A sentence linked to several components is counted once in both numerator and denominator.
`SFM = 0` means every documented component recovers >=1 correct link. SFM is reported on the
**doc-model (sad-sam) task ONLY**; `compute_sad_code` was never touched.

### Why doc-model only
- **doc-code** already captures silent component failure via the worst-component F1 (goes to 0 when
  a component is abandoned), so SFM would be redundant there. The doc-code suite keeps its **three**
  size-aware metrics.
- **doc-model** does NOT report worst/harmonic: on doc-model they are redundant with link-F1
  (Spearman **+0.87** worst-comp / **+0.83** harmonic vs link-F1, 30 cells). SFM fills the gap; it
  stays independent of link-F1 on doc-model (Spearman **-0.50**).

## Doc-model SFM numbers (verified; used verbatim in the prose + tables)

| System              | macro doc-model SFM | abandoned components |
|---------------------|---------------------|----------------------|
| approach (GPT-5.4)  | **0.0%** (SFC 0)    | none on any project  |
| Artemis (GPT-5.4)   | 7.06% (table 7.1)   | GAE Datastore @ Teammates (11.11%), Presentation Conversion @ BBB (4.17%), preferences @ JabRef (20.00%) |
| TransArC / SWATTR   | 7.41% (table 7.4)   | DB, FileStorage, Reencoding @ MediaStore (37.04%, SFC 3) |

RQ1 doc-model table SFM% column: approach **0.0** (bold), Artemis **7.1**, SWATTR **7.4**.

## Files changed

### infra - transarc-emp (commit 1c3c493, branch `mini`) - Tasks 1-6, done previously
- `mini-src/metrics.py` - SFM/SFC in `compute_sad_sam` only; PANELS["sad-sam"] + HEADERS; docstring.
- `mini-src/check.py` - sad-sam GOLDEN tuples extended with SFM%/SFC; sad-code goldens untouched.
- `mini-src/rq12.py` - `doc_to_model_silent_failure_mass` (+ `_count`) in the SS column group only.
- `mini-src/rq_tables.py`, `mini-src/csv_to_tex.py` - `dm_sfm` column wired into the RQ1 table.
- regenerated: `reports/RQ12_BIGTABLE.csv`, `reports/RQ12_PERPROJECT.csv`,
  `reports/tex/rq1-results.tex`, `reports/tex_src/rq1.csv`.

### paper - alinker-paper (commit 79011b8, branch `sfm-doc-model-metric`) - Tasks 7-12, this half
- `sections/metric.tex` (Task 7) - SFM/SFC definition + `eq:sfm`; two notation-table rows; scope
  comment reworded/appended (F1-tail=doc-code / SFM=doc-model); "three size-aware metrics" intact.
- `sections/results.tex` (Task 8) - doc-model SFM result paragraph in RQ1 (approach 0.0/SFC 0;
  Artemis 7.1; SWATTR 7.4 with named abandonments); doc-code RQ2 prose untouched.
- `sections/eval.tex` (Task 8) - scope sentence now F1-tail=doc-code / SFM=doc-model; stale
  `%DONE [scope]` comment appended with 2026-06-30 note.
- `sections/intro.tex` (Task 9) - SFM added to suite description + contributions bullet; "three" kept.
- `sections/discussion.tex`, `sections/conclusion.tex`, `sections/approach.tex` (Task 9) - SFM added
  as the doc-model size-aware metric; doc-code worst/harmonic statements intact.
- `sections/motivation.tex` (Task 10) - doc-model SFM framing added; JabRef preferences/46x doc-code
  example kept; no 20/32 doc-code figure attributed to SFM.
- `table/rq1-results.tex` (Task 11) - replaced with the regenerated GENERATED file (doc-model SFM%
  column); `table/rq2-results.tex` untouched.

#### Note on pre-existing edits carried in the paper commit
`sections/intro.tex` and `sections/motivation.tex` had pre-existing uncommitted working-tree edits
that predate this task: an intro-redesign SPEC block ("NEW INTRO DESIGN ... PROSE NOT YET WRITTEN",
plan `glimmering-soaring-stroustrup.md`) and a motivation data-verification fix (preferences 0.44% /
46x). Because git stages whole files, those edits rode along in commit 79011b8. No prose authored by
this task depends on them; they can be split out later if desired. The unrelated `figures/*` and
`notes/*` working-tree changes were deliberately left UNSTAGED.

## Verification results (V1-V6)

| Gate | Scope | Result |
|------|-------|--------|
| **V1** infra regression `check.py` PASS (sad-sam SFM/SFC frozen; sad-code unchanged) | infra half | PASS (verified in infra half, commit 1c3c493) |
| **V2** doc-model SFM sanity: approach 0.0/SFC 0; baselines >0 with named abandonments | infra half (human-verified Task 3) | PASS - numbers carried verbatim into this half |
| **V3** doc-code "three size-aware" INTACT; no "four size-aware" | paper half | PASS - `grep -rniE "three size-aware" sections/` hits metric.tex L70, intro.tex L201/L211; `grep -rniE "four size-aware" sections/` empty |
| **V4** SFM never doc-code/both-task; worst/harmonic doc-code wording present | paper half | PASS - `! grep -rniE "SFM.*doc-?code\|SFM.*both task\|both-task.*SFM" sections/ table/` empty; worst/harmonic still in metric.tex |
| **V5** rq12 regenerates with doc-model SFM column only (no doc-code SFM) | infra half | PASS (RQ12_BIGTABLE.csv carries `doc_to_model_silent_failure_mass`, no `doc_to_code_*` SFM) |
| **V6** no new deps / no benchmark leakage | infra half | PASS (stdlib-only; abandoned set derived from gold/result sets) |

## Deviations from plan

- **[Rule 3 - Blocking]** Two comment lines written during Tasks 7 and 8 tripped the V4-family
  greps (`SFM.*doc-?code` and the broader Task-8 `SFM.*both`) because they placed "SFM" before the
  literal token "doc-code"/"both" on one line. Reworded both comments ("doc-to-code grain",
  "not used on the file grain") to preserve meaning while passing the gate. No prose meaning changed.
- Paper was on default branch `main`; created branch `sfm-doc-model-metric` before committing, per
  the plan's branch-first rule.

## Self-Check: PASSED
- Paper commit 79011b8 exists on branch `sfm-doc-model-metric`; 9 files, 0 deletions, no transarc-emp path.
- Infra commit 1c3c493 exists on branch `mini` with the 5 source files + regenerated reports.
- All paper files edited exist and pass their per-task greps; V3/V4 final gates PASS.

---

## Amendment 2026-06-30 (two corrections: relocate SFM + suite count three -> four)

User-requested follow-up. SFM computation (`compute_sad_sam`) and the SFM definition / `eq:sfm`
were NOT touched. Two corrections only:

1. **SFM moved RQ1 -> RQ2 + big tables.** SFM was originally placed in the RQ1 link-recovery
   table; it belongs with the size-aware reporting. Dropped the `dm_sfm` column from RQ1
   (table returns to dm P/R/F1 + dc P/R/F1); added SFM to the RQ2 size-aware table and to the
   RQ1+RQ2 big tables (avg + per-project) as the last column of the size-aware group.
2. **Suite count three -> four.** The size-aware suite is now described as FOUR metrics: sentence
   coverage, worst-component F1, harmonic-component F1, and SFM. Per-task scoping unchanged
   (coverage/worst/harmonic on doc-code; SFM on doc-model only). SFM is never called a doc-code
   or both-task metric.

### New commit hashes
- **infra (transarc-emp, branch `mini`): `2438374`** - `chore(tables): move SFM to RQ2 + big table; drop from RQ1`
- **paper (alinker-paper): `f9df03c`** - `paper: SFM joins the size-aware suite (four metrics); report in RQ2 + big table`

### Files changed (amendment)

**infra - transarc-emp (commit `2438374`)**
- `mini-src/rq_tables.py` - `build_rq1` drops the `dm_sfm` column; `build_rq2` adds
  `silent_failure_mass` (from `doc_to_model_silent_failure_mass`); `SUITE_COLS` gains
  `doc_to_model_silent_failure_mass`.
- `mini-src/csv_to_tex.py` - RQ1 SPEC: removed the SFM column, doc-model group span 4 -> 3.
  RQ2 SPEC: added SFM\% column (bold min) + new four-metric caption. `SUITE9`/`SUITE9_GROUPS`:
  appended `doc_to_model_silent_failure_mass` (header "SFM", bold min) as the last size-aware
  column, group relabel "size-aware (doc-code)" -> "size-aware", span 3 -> 4. big-table footnote
  reworded so it no longer claims the suite is "doc-to-code only" (now: Cov/Worst/Harm on
  doc-code plus doc-model SFM).
- regenerated: `reports/tex_src/rq1.csv`, `rq2.csv`, `bigtable_rq12_avg.csv`,
  `bigtable_rq12_perproject.csv`; `reports/tex/rq1-results.tex`, `rq2-results.tex`,
  `big-table.tex`, `big-table-perproject.tex`.
- Guard honored: regenerating re-rendered `reports/tex/rq4-bigtable-perproject.tex` (manual
  edits); RESTORED to committed state and excluded from the commit.

**paper - alinker-paper (commit `f9df03c`)**
- `table/rq1-results.tex` - regenerated; SFM column now GONE.
- `table/rq2-results.tex` - regenerated; SFM\% column + four-metric caption.
- `appendix/big-table-perproject.tex` - SFM column added to the size-aware group ONLY; the
  per-system Average rows + caption + wording (manual paper edits, not present in the generator
  output) were preserved by surgical edit rather than blind copy; all doc-model link / doc-code
  file numbers unchanged; footnote "size-aware ... doc-code only" reworded.
- `sections/metric.tex` - L70 SPEC comment + L75 prose: "three size-aware" -> "four size-aware",
  SFM folded into the enumeration; SFM definition block / `eq:sfm` / per-task scoping intact.
- `sections/intro.tex` - prose + contributions bullet: "three size-aware" -> "four size-aware",
  the three-plus-SFM split merged into one four-metric enumeration (worst/harm on doc-code, SFM
  on doc-model).

### Deviations (amendment)
- **[Rule 1 - Bug/consistency]** The big-table (avg + per-project) footnote still asserted the
  size-aware suite is "defined on doc-code only", which contradicts SFM (doc-model) now sitting in
  that group. Reworded both footnotes to "Cov/Worst/Harm on doc-code plus the doc-model SFM". No
  numbers changed.
- **[scope guard]** `appendix/big-table-perproject.tex` was NOT blind-copied from the regenerated
  render: the paper's committed version carries manual per-system Average rows + a custom
  caption/wording absent from the generator. A blind copy would have deleted those. Per the
  amendment's "only the SFM column added" instruction, the SFM column was added surgically and
  everything else preserved.
- **[scope isolation]** `sections/intro.tex` carried pre-existing uncommitted comment-block WIP
  (PARA 3/4/5 authoring notes, plan `glimmering-soaring-stroustrup.md`) unrelated to SFM. Only the
  two SFM content hunks were staged/committed; the comment WIP remains UNSTAGED in the working
  tree. `figures/*` and `notes/*` working-tree changes left untouched.

### Flipped-gate verification (amendment)
| Gate | Command | Result |
|------|---------|--------|
| four size-aware present | `grep -rniE "four size-aware" sections/` | PASS - metric.tex + intro.tex (L221, L231) |
| no three size-aware | `! grep -rniE "three size-aware" sections/` | PASS - empty (unrelated "Three factors" in discussion.tex untouched) |
| SFM in RQ2 | `grep -iE "SFM\|silent" table/rq2-results.tex` | PASS - caption + SFM\% column |
| SFM absent from RQ1 | `! grep -iE "SFM\|silent" table/rq1-results.tex` | PASS - empty |
| SFM in appendix big-table | `grep -iE "SFM" appendix/big-table-perproject.tex` | PASS - header column + footnote |
| SFM not doc-code/both-task | `! grep -rniE "SFM.*doc-?code\|SFM.*both task\|both-task.*SFM" sections/ table/ appendix/` | PASS - empty |
| infra computation unchanged | `python3 mini-src/check.py` | PASS - frozen golden panel reproduced |

---

## Amendment 2 (2026-06-30): RQ2 + big table CLUSTERED BY TASK

User-requested follow-up. SFM was a **DOC-MODEL** metric but sat as a lone column in the
otherwise-doc-code RQ2 table and inside the doc-code "size-aware" group of the big table. This
amendment restructures both tables into **task-clustered groups** so each task shows
[reference \fone\ + its size-aware metric(s)]. **No metric computation (`compute_sad_sam` /
`eq:sfm`) and no numeric value changed** — layout only; numbers come from the regenerated CSVs.

1. **tab:rq2 → two task groups.** doc-model (span 2): Link \fone\ (= `doc_to_model_link_f1`,
   reference) | SFM\% (= `doc_to_model_silent_failure_mass`, size-aware, bold MIN). doc-code
   (span 4): File \fone\ (reference) | Sent.\ cov. | Worst \fone | Harm.\ \fone. **Doc-model link
   \fone\ added** to RQ2 as the doc-model reference column.
2. **tab:detailed-perproject (big table) → SFM moved into the doc-model group.** SFM relocated
   out of the size-aware group to immediately after `doc_to_model_link_f1`. Group spans: doc-model
   3 → 4, size-aware 4 → 3; size-aware group relabeled "size-aware (doc-code)"; the
   "plus the doc-model SFM" clause removed from that group's footnote (SFM is now under doc-model).

### New commit hashes
- **infra (transarc-emp, branch `mini`): `0471fb6`** — `chore(tables): cluster RQ2 + big table by task; SFM under doc-model with link F1`
- **paper (alinker-paper, branch `sfm-doc-model-metric`): `ee2a295`** — `paper: cluster RQ2 + big table by task (doc-model: link F1 + SFM)`

### Files changed (Amendment 2)

**infra — transarc-emp (commit `0471fb6`)**
- `mini-src/rq_tables.py` — `build_rq2` adds `link_f1` (from `doc_to_model_link_f1`); write_csv field
  order → `["system","link_f1","silent_failure_mass","file_f1","sentence_coverage","worst_component_f1","harmonic_component_f1"]`.
- `mini-src/csv_to_tex.py` — RQ2 SPEC: `groups [("doc-model",2),("doc-code",4)]`, columns reordered
  to Link \fone / SFM\% / File \fone / Sent. cov. / Worst \fone / Harm. \fone, new task-clustered
  caption. `SUITE9`: `doc_to_model_silent_failure_mass` moved to immediately after
  `doc_to_model_link_f1`. `SUITE9_GROUPS` → `[("doc-to-model (link \fone)",4),("doc-to-code (file \fone)",3),("size-aware (doc-code)",3)]`.
  big-table.tex footnote reworded (size-aware (doc-code) = Cov/Worst/Harm; SFM sits with doc-to-model).
- regenerated: `reports/tex_src/rq2.csv`; `reports/tex/rq2-results.tex`, `big-table.tex`,
  `big-table-perproject.tex`.
- Guard honored: regenerating re-rendered `reports/tex/rq4-bigtable-perproject.tex` (manual edits);
  RESTORED to committed state and excluded.

**paper — alinker-paper (commit `ee2a295`)**
- `table/rq2-results.tex` — replaced with the regenerated GENERATED file (two groups: doc-model
  Link \fone + SFM\%; doc-code File \fone/cov/worst/harm).
- `appendix/big-table-perproject.tex` — SURGICAL edit (manual per-system Average rows + custom
  caption/footnote preserved): SFM column moved from the size-aware group to the doc-model group;
  `\multicolumn` spans (doc-model 3→4, size-aware 3-span) + `\cmidrule` ranges updated; every data
  row and every manual Average row repositioned (values unchanged — verified by per-row cell-multiset
  diff); footnote reworded (SFM now under doc-model). Pre-existing unrelated working-tree edits
  (`sections/intro.tex`, `figures/*`) left UNSTAGED.

### Deviations (Amendment 2)
- **[Rule 1 — gate consistency]** The amendment-specified RQ2 caption ("...SFM on doc-model; ...
  on doc-code") would false-trip the acceptance gate `! grep -rniE "SFM.*doc-?code"` because the
  greedy `.*` spans from "SFM" to "doc-code" on one line. Reordered the caption clause so the
  doc-code metrics are described **before** SFM ("...sentence coverage, worst- and harmonic-component
  \fone\ on doc-code, and SFM on doc-model."), preserving the doc-model attribution of SFM and all
  information while passing the gate. Applied in the generator (source of truth) and re-synced to the
  paper. No numbers changed.

### Verification (Amendment 2 gates)
| Gate | Command | Result |
|------|---------|--------|
| infra computation unchanged | `python3 mini-src/check.py` | PASS — frozen golden panel reproduced |
| RQ2 shows doc-model + doc-code groups | `grep -A3 tabular table/rq2-results.tex` | PASS — doc-model (Link \fone + SFM\%, span 2) + doc-code (File \fone/cov/worst/harm, span 4) |
| SFM not doc-code/both-task | `! grep -rniE "SFM.*doc-?code\|SFM.*both task" sections/ table/ appendix/` | PASS — empty |
| four size-aware present | `grep -rniE "four size-aware" sections/` | PASS — intro.tex |
| no three size-aware | `! grep -rniE "three size-aware" sections/` | PASS — empty |
| big-table SFM under doc-model | header inspection | PASS — SFM in `\multicolumn{4}{doc-model}` group, not size-aware; all P/R/F1/Cov/Worst/Harm numbers unchanged (per-row multiset diff identical) |
