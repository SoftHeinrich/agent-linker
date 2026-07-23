---
task: add-sfm-silent-failure-mass
type: quick
created: 2026-06-30
revised: 2026-06-30   # SCOPE REVERSAL: SFM is now DOC-MODEL ONLY (was both-task)
repos:
  - infra: /mnt/hostshare/ardoco-home/transarc-emp        # git root
  - paper: /mnt/hostshare/ardoco-home/alinker-paper       # SEPARATE git root
autonomous: false   # V2 doc-model sanity + V3/V4 paper greps are human-verifiable gates
constraints:
  - Python stdlib only; no new deps (workspace rule)
  - No benchmark leakage (no hardcoded component/project word lists)
  - Two separate git repos: atomic commit PER repo; never cross-commit
  - Do NOT change anything in compute_sad_code: worst/harmonic numbers and their
    doc-code-only scoping stay exactly as they are
  - Do NOT change the doc-code suite's "three size-aware metrics" wording
---

<objective>
Add a new size-aware evaluation metric, **Silent-Failure Mass (SFM)** (integer companion
**SFC**), to (a) the evaluation INFRA (`transarc-emp/mini-src`), (b) the result TABLES, and
(c) the PAPER (`alinker-paper`). SFM is reported on the **doc-model (sad-sam) task ONLY**.

## Why doc-model only (put this rationale in the paper)
Each task keeps the size-aware metric that is *informative* for it:
- **doc-code** already has a metric that captures silent component failure: the **worst-component
  F1** goes to 0 the moment a component is abandoned. SFM would be redundant on doc-code, so the
  doc-code suite is left UNCHANGED — it keeps its **three** size-aware metrics (sentence coverage,
  worst-component F1, harmonic-mean F1).
- **doc-model** does NOT report worst/harmonic: those go redundant with link-F1 on doc-model
  (Spearman **0.87 worst-comp / 0.83 harmonic** vs link-F1, 30 cells — verified by
  `explore-tail/sfm_vs_worst.py` + `abandmass_corr.py`), so doc-model is left with no size-aware
  metric. **SFM fills exactly that gap** — it stays independent of link-F1 on doc-model
  (Spearman **-0.50**, same run). So: doc-code uses the F1-tail (worst/harmonic); doc-model uses SFM.

## Canonical SFM definition (distinct-sentence denominator)
For each gold component k, let `gold_sents(k)` = the set of DISTINCT documentation sentences
gold-linked to k, and `correct(k)` = the correct links recovered for k.
`recall_k = |correct(k)| / |gold links for k|`. Component k is **ABANDONED** iff `recall_k = 0`
(zero correct links recovered for k). Then:

    SFC = #{ k : k is abandoned }                                  (integer companion)
    SFM = |{ distinct documented sentences belonging to >=1 abandoned component }|
          / |{ distinct documented sentences }|   x 100            (in [0,100], report %)

SFM = 0 means every documented component recovers >=1 correct link.

**CRITICAL — the prototype denominator is wrong on purpose.** `explore-tail/abandmass_table.py`
and `abandmass_corr.py` use a `(sentence,component)`-DECISION denominator
(`tot = sum(len(gb[c]) for c in gb)`), which double-counts a sentence linked to multiple
components. The canonical metric uses the **DISTINCT-SENTENCE** denominator and numerator (dedupe
sentences across components). Port the *structure* from the prototype, but recompute over distinct
sentences. The acceptance anchor (V2) is now doc-MODEL: **approach SFM == 0 on doc-model**, and
**the deterministic/LLM baselines abandon >=1 doc-model component (SFM > 0)** — exact percentages
are computed by the canonical impl, NOT hardcoded.

Output:
- INFRA: SFM + SFC in `compute_sad_sam` ONLY (compute_sad_code untouched); PANELS["sad-sam"] +
  HEADERS; frozen in `check.py` (sad-sam cells only); `doc_to_model_silent_failure_mass` (+SFC)
  in `rq12.py`; SFM wired into the doc-model RQ1 table.
- PAPER: SFM introduced as the doc-model size-aware metric (equation + notation row in
  `metric.tex`); doc-model SFM result in `results.tex`; scope fix in `eval.tex` (F1-tail = doc-code,
  SFM = doc-model); SFM as the doc-model size-aware contribution in `intro.tex`, `discussion.tex`,
  `conclusion.tex`, `approach.tex`; a doc-model SFM framing in `motivation.tex`; SFM column added to
  the doc-model RQ1 table. The doc-code RQ2 table and its "three" wording stay UNCHANGED.
</objective>

<context>
@/mnt/hostshare/ardoco-home/transarc-emp/mini-src/metrics.py
@/mnt/hostshare/ardoco-home/transarc-emp/mini-src/check.py
@/mnt/hostshare/ardoco-home/transarc-emp/mini-src/rq12.py
@/mnt/hostshare/ardoco-home/transarc-emp/mini-src/rq_tables.py
@/mnt/hostshare/ardoco-home/transarc-emp/mini-src/csv_to_tex.py
@/mnt/hostshare/ardoco-home/transarc-emp/explore-tail/abandmass_table.py
@/mnt/hostshare/ardoco-home/transarc-emp/explore-tail/abandmass_corr.py
@/mnt/hostshare/ardoco-home/alinker-paper/sections/metric.tex
@/mnt/hostshare/ardoco-home/alinker-paper/sections/results.tex
@/mnt/hostshare/ardoco-home/alinker-paper/sections/eval.tex
@/mnt/hostshare/ardoco-home/alinker-paper/sections/motivation.tex
@/mnt/hostshare/ardoco-home/alinker-paper/table/rq1-results.tex

<interfaces>
Reuse, do not re-derive, existing structure in metrics.py:
- compute_sad_sam (~L372-388): `gold` = set[(comp, sent)]; it already builds `gold_by_s`/`res_by_s`
  (sent -> {comp}). For SFM build `gold_by_c`/`res_by_c` (comp -> {sent}) from `gold`/`res`, then:
    abandoned = { c in gold_by_c : not res_by_c.get(c) }     # zero correct sentences for c
    abandoned_sents = union of gold_by_c[c] over abandoned c (empty if none)
    all_sents       = union of gold_by_c.values()            # = distinct documented sentences
    SFM = len(abandoned_sents)/len(all_sents)*100  (0.0 if empty); SFC = len(abandoned)
- prf() convention (L123): empty res scores (0,0,0) — an abandoned component has recall 0 by
  construction; reuse, do not special-case.
- compute_sad_code is UNCHANGED. Do not add SFM there.
- PANELS / HEADERS (L393-406): add the two new keys to PANELS["sad-sam"] ONLY (NOT sad-code).
- check.py GOLDEN (L40-47 sad-sam block) is a positional tuple in PANELS["sad-sam"] order — adding
  two columns to PANELS["sad-sam"] means appending two values to EACH sad-sam project tuple.
  The sad-code GOLDEN block is UNTOUCHED.
- rq12.py COLUMNS (L133-147): add only doc_to_model_silent_failure_mass + _count (SS task).
  RQ2_COLS (L123-128, doc-code) is UNCHANGED.
- Doc-model table: rq_tables.build_rq1 (L83-107) writes rq1.csv (dm_/dc_ columns);
  csv_to_tex rq1 SPEC (L185-200) renders the doc-model group. SFM is a doc-model column there.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1 (INFRA): Add SFM + SFC to compute_sad_sam ONLY</name>
  <files>/mnt/hostshare/ardoco-home/transarc-emp/mini-src/metrics.py</files>
  <behavior>
    - compute_sad_sam return dict gains `silent_failure_mass` (float, %, [0,100]) and
      `silent_failure_count` (int). compute_sad_code is NOT modified.
    - Abandoned component c: c in gold_by_c AND res_by_c.get(c) empty/zero-correct.
    - DISTINCT-SENTENCE denominator + numerator (a sentence linked to two comps counts once).
    - Empty-gold guard: denominator 0 -> SFM 0.0, SFC 0.
  </behavior>
  <action>
    In compute_sad_sam build `gold_by_c`/`res_by_c` (comp -> set[sentence]) from `gold`/`res`,
    then compute SFM/SFC per the <interfaces> formulas (a tiny inline helper is fine; reference
    decision: distinct-sentence denominator). Add `silent_failure_mass` + `silent_failure_count`
    to the return dict. Add both keys to PANELS["sad-sam"] (do NOT touch PANELS["sad-code"]).
    Add HEADERS entries (e.g. "SFM%", "SFC"). Update the module docstring (L14-21): the sad-sam
    panel now also reports SFM/SFC; clarify that only the MICRO per-component F1 collapses onto
    link-F1 with no enrolment, while SFM/SFC do NOT collapse — they are the doc-model size-aware
    metric; note worst/harmonic remain doc-code-only. Leave compute_sad_code and all its cells
    exactly as-is.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/transarc-emp && python3 mini-src/metrics.py --task sad-sam --project jabref | grep -qi SFM && ! ( python3 mini-src/metrics.py --task sad-code --project jabref | grep -qi SFM )</automated>
    <note>The sad-code guard is `! ( ... | grep -qi SFM )` so it FAILS if SFM leaks into the
    sad-code panel. (Do not use `grep -vqi SFM` — that exits 0 on any line lacking SFM and proves nothing.)</note>
  </verify>
  <done>sad-sam panel prints SFM%/SFC; sad-code panel shows NO SFM column (compute_sad_code
  untouched); PANELS["sad-sam"] lists the two new keys, PANELS["sad-code"] does not.</done>
</task>

<task type="auto">
  <name>Task 2 (INFRA): Extend the frozen golden panel (sad-sam cells only)</name>
  <files>/mnt/hostshare/ardoco-home/transarc-emp/mini-src/check.py</files>
  <action>
    GOLDEN["sad-sam"][project] tuples are positional in PANELS["sad-sam"] order. Append two values
    (SFM%, SFC) to EACH sad-sam project tuple. Obtain authoritative values by running
    `python3 mini-src/metrics.py --task sad-sam` over the bundled TransArc results and transcribing
    the printed SFM%/SFC — do NOT invent numbers. Leave GOLDEN["sad-code"] UNCHANGED. Update the
    sad-sam header comment row to include the two new labels. Keep TOL=1e-4. Update the docstring
    provenance note: SFM/SFC added 2026-06-30, doc-model only, distinct-sentence denominator.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/transarc-emp && python3 mini-src/check.py</automated>
  </verify>
  <done>check.py prints PASS with the new sad-sam SFM/SFC cells; sad-code goldens unchanged (V1).</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 3 (INFRA): Doc-model SFM sanity (V2)</name>
  <what-built>SFM/SFC computed for the doc-model task from the canonical distinct-sentence
  denominator, over the live SOTA model-doc dump used by rq12.py.</what-built>
  <how-to-verify>
    Score the doc-model (sad-sam) SOTA links through `metrics.compute_sad_sam` (canonical
    denominator, NOT the prototype grain):
      1. approach (GPT-5.4, mean of run1/2/3) doc-model SFM == **0.0** (and SFC == 0).
      2. The baselines abandon >=1 doc-model component: TransArC/SWATTR and Artemis doc-model
         SFM are **> 0** (report the per-project SFM and the named abandoned component(s) the
         executor finds — these become the prose numbers in Task 8). Do NOT expect the doc-code
         20%/32% figures here; those were doc-code and no longer apply to SFM.
    If approach is not 0 on doc-model, or no baseline shows SFM>0, the denominator/abandoned-set
    logic is wrong — fix metrics.py before proceeding.
  </how-to-verify>
  <resume-signal>Type "approved" once approach==0 on doc-model and baselines>0 are confirmed,
  with the named abandoned components recorded; or report the deviation.</resume-signal>
</task>

<task type="auto">
  <name>Task 4 (INFRA): Add the doc-model SFM column to rq12.py and regenerate</name>
  <files>/mnt/hostshare/ardoco-home/transarc-emp/mini-src/rq12.py</files>
  <action>
    Add to COLUMNS (L133-147), in the doc-model (SS) group only:
    `("doc_to_model_silent_failure_mass", SS, "silent_failure_mass")` and
    `("doc_to_model_silent_failure_count", SS, "silent_failure_count")`.
    Do NOT add any doc-code SFM column. Leave RQ2_COLS / build_rq2_panel / PAPER_RQ2_OUTDATED
    (all doc-code) UNCHANGED. Update the module docstring to note the doc-model big table now
    carries SFM. Every cell still flows from metrics.compute_sad_sam — no arithmetic here.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/transarc-emp && python3 mini-src/rq12.py >/dev/null && grep -q doc_to_model_silent_failure_mass reports/RQ12_BIGTABLE.csv && ! grep -q doc_to_code_silent_failure_mass reports/RQ12_BIGTABLE.csv</automated>
  </verify>
  <done>RQ12_BIGTABLE.csv regenerates with doc_to_model_silent_failure_mass (+ count) and NO
  doc-code SFM column (this is V5).</done>
</task>

<task type="auto">
  <name>Task 5 (INFRA): Wire doc-model SFM into the RQ1 table pipeline</name>
  <files>/mnt/hostshare/ardoco-home/transarc-emp/mini-src/rq_tables.py, /mnt/hostshare/ardoco-home/transarc-emp/mini-src/csv_to_tex.py</files>
  <action>
    Report SFM compactly in the doc-MODEL table (decision #4 spirit, now on doc-model). In
    rq_tables.build_rq1 (L83-107): add a `dm_sfm` column sourced from
    `doc_to_model_silent_failure_mass`, included for rows that have doc-model output (approach,
    Artemis, SWATTR) and "" for the doc-code-only TransArC row. In csv_to_tex.py rq1 SPEC
    (L185-200): add an SFM column inside the "doc-to-model" group (header e.g. "SFM\\%", kind
    "f1", bold "min" since lower is better) and bump the group span from 3 to 4. The doc-code RQ2
    spec/table is UNCHANGED. Numbers come from the regenerated rq1.csv; do not hand-key cells.
    (The prototype `abandmass_table.py` is a reference only; the canonical home is the RQ1
    doc-model table, not a standalone float.)
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/transarc-emp && python3 mini-src/rq_tables.py && python3 mini-src/csv_to_tex.py && grep -iE "SFM|silent" reports/tex/rq1-results.tex && grep -L -iE "SFM|silent" reports/tex/rq2-results.tex</automated>
  </verify>
  <done>reports/tex/rq1-results.tex gains an SFM column under doc-model; reports/tex/rq2-results.tex
  (doc-code) has NO SFM and is otherwise unchanged.</done>
</task>

<task type="auto">
  <name>Task 6 (INFRA): Atomic commit of the infra repo</name>
  <files>/mnt/hostshare/ardoco-home/transarc-emp (git root)</files>
  <action>
    Commit ONLY transarc-emp changes (metrics.py, check.py, rq12.py, rq_tables.py, csv_to_tex.py,
    regenerated reports/). Do NOT add any alinker-paper path. Branch first if on default branch.
    Message: `feat(metrics): add doc-model Silent-Failure Mass (SFM/SFC)`.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/transarc-emp && git log --oneline -1 && git status --porcelain</automated>
  </verify>
  <done>Single infra commit; clean status; touches no alinker-paper file.</done>
</task>

<task type="auto">
  <name>Task 7 (PAPER): metric.tex — introduce SFM as the doc-model metric; fix scope</name>
  <files>/mnt/hostshare/ardoco-home/alinker-paper/sections/metric.tex</files>
  <action>
    (a) KEEP the doc-code suite's "three size-aware metrics" wording (L64-68) INTACT — do NOT
    change three->four. The doc-code suite is unchanged.
    (b) Add a new SFM block (equation + definition paragraph) introducing SFM as the
    **doc-model** size-aware metric. Use the distinct-sentence denominator; tie the name to the
    existing term "silent component failure" (L29). Mention SFC = count of abandoned components.
    SFM in [0,1] (report %). State the principle: on doc-model the F1-tail (worst/harmonic) is
    uninformative (redundant with link-F1, Spearman 0.83 harmonic / 0.87 worst-comp), so SFM is the
    independent size-aware metric there (Spearman -0.50). Cite the single verified pair consistently
    — do NOT write a lone ".83" for worst-comp (worst-comp is .87; .83 is harmonic).
    (c) Add an SFM symbol row to the notation table (L92-102).
    (d) REWORD the 2026-06-29 doc-code-only scope comment (L19-26) and L67 to:
    "the F1-tail (worst/harmonic) is doc-code-only because it is redundant with link-F1 on
    doc-model; doc-model instead reports SFM." Append (do not delete) this clarification dated
    2026-06-30. Leave eq:worst / eq:harm and their doc-code-only claim intact.
    Do NOT claim SFM applies to doc-code or to "both tasks" anywhere.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/alinker-paper && grep -niE "Silent-Failure Mass|SFM" sections/metric.tex && grep -niE "three size-aware" sections/metric.tex && ! grep -niE "SFM.*doc-?code|SFM.*both task" sections/metric.tex</automated>
  </verify>
  <done>metric.tex keeps "three size-aware" for doc-code, defines SFM (equation + notation row) as
  the doc-model metric, reworded scope explains F1-tail=doc-code / SFM=doc-model, and nowhere
  claims SFM is doc-code or both-task.</done>
</task>

<task type="auto">
  <name>Task 8 (PAPER): results.tex + eval.tex — doc-model SFM result + scope</name>
  <files>/mnt/hostshare/ardoco-home/alinker-paper/sections/results.tex, /mnt/hostshare/ardoco-home/alinker-paper/sections/eval.tex</files>
  <action>
    results.tex: add the doc-MODEL SFM result — approach SFM = 0 on doc-model; the baselines
    abandon doc-model component(s) with SFM > 0 (use the exact macro + named-failure values
    recorded in Task 3 from the regenerated RQ12_BIGTABLE.csv; mark them GENERATED-sourced). Do
    NOT touch the doc-code RQ2 prose (worst/harmonic +41pp/+42pp, sentence coverage at L171 stay).
    eval.tex (sec:exp:rq2, ~L87-89): update the scope sentence so it reads: the F1-tail
    (worst/harmonic) is reported on doc-code; SFM is reported on doc-model. Do not claim SFM is
    both-task. Also APPEND (do not delete) a 2026-06-30 note to the stale `%DONE [scope]` comment
    at eval.tex ~L87 so the comment matches the reworded prose (F1-tail=doc-code / SFM=doc-model).
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/alinker-paper && grep -niE "SFM|silent.failure" sections/results.tex && grep -niE "SFM|silent.failure" sections/eval.tex && ! grep -niE "SFM.*doc-?code|SFM.*both" sections/results.tex sections/eval.tex</automated>
    <note>eval.tex grep requires SFM/silent-failure specifically — NOT "doc-model" (eval.tex already
    contains "doc-model" on L24/64/67/73/78, so a "doc-model" match would false-pass without the edit).</note>
  </verify>
  <done>results.tex reports doc-model SFM (approach 0, baselines>0 with named failures); eval.tex
  scope reads F1-tail=doc-code / SFM=doc-model; no SFM-on-doc-code claim.</done>
</task>

<task type="auto">
  <name>Task 9 (PAPER): intro / discussion / conclusion / approach — SFM as doc-model contribution</name>
  <files>/mnt/hostshare/ardoco-home/alinker-paper/sections/intro.tex, /mnt/hostshare/ardoco-home/alinker-paper/sections/discussion.tex, /mnt/hostshare/ardoco-home/alinker-paper/sections/conclusion.tex, /mnt/hostshare/ardoco-home/alinker-paper/sections/approach.tex</files>
  <action>
    Do NOT change the doc-code "three size-aware metrics" wording (intro L199/L208 etc. stay
    "three" for the doc-code suite). Instead ADD SFM as the doc-MODEL size-aware metric:
    intro.tex (L119 S3, L129 B2, L199, L208): augment the contribution/suite description to state
    that the suite also reports SFM on doc-model (the F1-tail is doc-code, SFM is doc-model).
    discussion.tex (L25): add that on doc-model the suite expresses silent component failure via
    SFM (worst/harmonic stay doc-code).
    conclusion.tex + approach.tex: where the suite is described, add the doc-model SFM mention;
    keep the doc-code "three" and worst/harmonic doc-code-only statements intact.
    Nowhere claim SFM applies to doc-code or both tasks.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/alinker-paper && for f in intro discussion conclusion approach; do grep -niqE "SFM|Silent-Failure" sections/$f.tex || { echo "MISSING SFM in $f.tex"; exit 1; }; done && grep -niE "three size-aware|three" sections/intro.tex && ! grep -rniE "SFM.*doc-?code|SFM.*both task" sections/intro.tex sections/discussion.tex sections/conclusion.tex sections/approach.tex</automated>
    <note>Positive SFM grep is asserted for EACH of the four files (loop), not just intro.tex —
    so editing intro alone can no longer false-pass while discussion/conclusion/approach stay un-updated.</note>
  </verify>
  <done>Doc-code "three" wording intact; SFM introduced as the doc-model size-aware metric in all
  four files; no SFM-on-doc-code/both-task claim.</done>
</task>

<task type="auto">
  <name>Task 10 (PAPER): motivation.tex — doc-model framing for SFM</name>
  <files>/mnt/hostshare/ardoco-home/alinker-paper/sections/motivation.tex</files>
  <action>
    Leave the JabRef preferences / 46x example as the DOC-CODE motivation (it motivates the
    worst-component F1, which already exists). Add a short doc-MODEL framing for SFM: the same
    silent-failure principle applied to doc-model, where the F1-tail is uninformative — a system
    can score well on link-level F1 yet abandon a documented component's sentences. Use the
    doc-model SFM numbers the executor computed (approach 0%; baselines abandon component(s) on
    doc-model). Do NOT reuse the 20%/32% doc-code figures for SFM.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/alinker-paper && grep -niE "SFM|silent.failure|doc-model" sections/motivation.tex && grep -niE "preferences|46" sections/motivation.tex</automated>
  </verify>
  <done>motivation.tex keeps the doc-code JabRef example AND adds a doc-model SFM framing with
  doc-model numbers; no 20/32 figure attributed to SFM.</done>
</task>

<task type="auto">
  <name>Task 11 (PAPER): doc-model RQ1 table — add SFM column</name>
  <files>/mnt/hostshare/ardoco-home/alinker-paper/table/rq1-results.tex</files>
  <action>
    rq1-results.tex is GENERATED. Copy the regenerated
    `transarc-emp/reports/tex/rq1-results.tex` (Task 5, now carrying the doc-model SFM column)
    over it. Keep the GENERATED banner. Confirm the doc-code (file F1) block and the existing
    doc-model P/R/F1 values are unchanged — only the SFM column is added under doc-model. Do NOT
    touch table/rq2-results.tex (doc-code; unchanged).
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/alinker-paper && grep -iE "SFM|silent" table/rq1-results.tex && head -2 table/rq1-results.tex | grep -i GENERATED && ! grep -iE "SFM|silent" table/rq2-results.tex</automated>
  </verify>
  <done>table/rq1-results.tex has the doc-model SFM column + GENERATED banner; table/rq2-results.tex
  unchanged (no SFM).</done>
</task>

<task type="auto">
  <name>Task 12 (PAPER): Atomic commit of the paper repo</name>
  <files>/mnt/hostshare/ardoco-home/alinker-paper (git root)</files>
  <action>
    Commit ONLY alinker-paper changes (sections/*.tex, table/rq1-results.tex). Do NOT add any
    transarc-emp path. Branch first if on default branch. Message:
    `paper: report Silent-Failure Mass (SFM) as the doc-model size-aware metric`.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/alinker-paper && git log --oneline -1 && git status --porcelain</automated>
  </verify>
  <done>Single paper commit; clean status; touches no transarc-emp file.</done>
</task>

</tasks>

<verification>
Explicit gates (the user requires these). All commands use absolute repo roots.

**V1 — infra regression PASS:**
`cd /mnt/hostshare/ardoco-home/transarc-emp && python3 mini-src/check.py` prints `PASS` (sad-sam
SFM/SFC cells frozen; sad-code goldens unchanged).

**V2 — doc-model SFM sanity (human-verified, Task 3):**
Via `metrics.compute_sad_sam` over the SOTA model-doc dump: approach (GPT-5.4, mean of 3 runs)
doc-model SFM == **0.0** (SFC 0); baselines (TransArC/SWATTR, Artemis) abandon >=1 doc-model
component, SFM **> 0**, with the named abandoned component(s) recorded. (No doc-code 20/32 check.)

**V3 — doc-code "three" wording INTACT, SFM not over-counted:**
`cd /mnt/hostshare/ardoco-home/alinker-paper && grep -rniE "three size-aware" sections/` STILL
returns the doc-code suite hits (intact); confirm no text claims "four size-aware" for the
doc-code suite (`! grep -rniE "four size-aware" sections/`).

**V4 — SFM is doc-model, never doc-code or both-task:**
`cd /mnt/hostshare/ardoco-home/alinker-paper && ! grep -rniE "SFM.*doc-?code|SFM.*both task|both-task.*SFM" sections/ table/`
returns nothing; and the F1-tail (worst/harmonic) doc-code-only wording is still present
(`grep -rniE "worst|harmonic" sections/metric.tex` scope lines intact).

**V5 — rq12 regenerates with the doc-model SFM column only:**
`cd /mnt/hostshare/ardoco-home/transarc-emp && python3 mini-src/rq12.py` exits 0;
`grep -q doc_to_model_silent_failure_mass reports/RQ12_BIGTABLE.csv` succeeds and
`! grep -q doc_to_code_silent_failure_mass reports/RQ12_BIGTABLE.csv`.

**V6 — no new deps / no benchmark leakage:**
`cd /mnt/hostshare/ardoco-home/transarc-emp && git diff --unified=0 -- mini-src | grep -E "^\+import |^\+from "`
shows only stdlib; the SFM helper derives abandoned components purely from gold/result sets, never
from a literal name list.
</verification>

<success_criteria>
- [ ] V1 PASS; V2 (approach doc-model SFM 0, baselines>0) confirmed; V3 doc-code "three" intact;
      V4 SFM never doc-code/both-task; V5 doc-model SFM column present (no doc-code SFM); V6 clean.
- [ ] SFM + SFC added to compute_sad_sam ONLY; compute_sad_code and PANELS["sad-code"] UNCHANGED.
- [ ] check.py sad-sam goldens extended and PASS; sad-code goldens untouched.
- [ ] rq12.py emits doc_to_model_silent_failure_mass (+ count); no doc-code SFM; RQ2 panel unchanged.
- [ ] Doc-model RQ1 table gains an SFM column; doc-code RQ2 table unchanged.
- [ ] Paper: metric.tex defines SFM as the doc-model metric, keeps doc-code "three", reworded scope
      (F1-tail=doc-code / SFM=doc-model); results/eval/intro/discussion/conclusion/approach/motivation
      updated for doc-model SFM; rq1-results.tex regenerated with SFM + GENERATED banner.
- [ ] Worst/harmonic doc-code numbers and doc-code-only scoping UNCHANGED.
- [ ] Exactly two commits: one transarc-emp, one alinker-paper. No cross-repo staging.
</success_criteria>

<rollback>
Two independent git roots → roll back per repo.
- INFRA (transarc-emp): `git -C /mnt/hostshare/ardoco-home/transarc-emp revert <infra-commit>` (or
  `git reset --hard <prev>`). Regenerated reports are reproducible via
  `python3 mini-src/rq12.py && python3 mini-src/rq_tables.py && python3 mini-src/csv_to_tex.py`;
  reverting the sources and rerunning restores prior tables. `check.py` PASS on the prior commit
  confirms a clean rollback. Because compute_sad_code was never touched, the doc-code suite cannot
  regress.
- PAPER (alinker-paper): `git -C /mnt/hostshare/ardoco-home/alinker-paper revert <paper-commit>`.
  rq1-results.tex is GENERATED; if only it is wrong, recopy the prior
  `transarc-emp/reports/tex/rq1-results.tex`. table/rq2-results.tex was never modified.
- Atomic per-repo commits mean either repo reverts independently. The prototype files in
  `explore-tail/` are read-only references and need no rollback. No deps added → no environment
  state to undo.
</rollback>

<output>
Create `.planning/quick/260630-add-sfm-silent-failure-mass/SUMMARY.md` when done.
</output>
