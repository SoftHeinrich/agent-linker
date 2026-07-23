# Phase 3: Motivation & Paper Hooks — MOTIV-01, OUT-02 - Context

**Gathered:** 2026-06-21
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — all 4 grey areas accepted by user

<domain>
## Phase Boundary

Close the loop inequality → metric suite: empirically show that trivial baselines
(Top-3, random) exploit the benchmark's distributional inequality to inflate
file-/link-level micro-F1 (MOTIV-01), explain why each of the four suite metrics
(per-component F1, sentence coverage, noise rate, file-level F1) is needed, fill
the trivial-baseline file-F1 placeholder, and emit paper-ready Gini/Lorenz
table + figure source (OUT-02).

IN scope: trivial baselines scored on the GOLD/benchmark (no system/TransArc
results), reusing the Phase-1 engine + copied metric/baseline defs; paper-ready
TeX/CSV table + pgfplots Lorenz snippet. OUT of scope: any real system's scores
(strongest-pipeline / \approach file-F1 placeholders stay deferred — need
published numbers); the TransArc error cascade.

**Isolation (HARD):** all work under `mini-inequality/`, project root =
`mini-inequality/`. Never touch repo-root `.planning/`. Commit only
`mini-inequality/**`.
</domain>

<decisions>
## Implementation Decisions

### Area 1 — Baseline definitions
- **Top-3 baseline** = predict, for every gold sentence, the 3 components/files
  with the most GOLD links overall (`baseline_top3_by_gold_links`, mirroring
  `src/bias/rq2_doc_to_model_prestudy.py:205`). Popularity-only, content-blind —
  exploits the skew.
- **Random baseline** = random (sentence, target) pairs at gold density
  (`target_size = len(gold)`), reproducible via `random.Random(0)` over SORTED
  sentence/target lists (`baseline_random`, prestudy:181). **Seed = 0**, fixed.
- **Both tasks**: sad-code (file-level micro-F1) and sad-sam (link-level micro-F1).
- **Deterministic**: single fixed seed; the seed is printed in the output.

### Area 2 — Metrics shown (the contrast)
- For each baseline report **micro-F1 (the inflated metric) + the 4-metric suite**:
  per-component macro F1, sentence coverage, noise rate, and (sad-code) file F1 —
  so the suite visibly discriminates the content-blind baseline.
- **Copy** `prf`/`sentence_coverage`/`noise_rate` from `mini-src/metrics.py` and
  `per_component_macro_f1` from `src/bias/rq2_doc_to_model_prestudy.py:227`
  (verbatim, no import); **reuse** `inequality.py` loaders (`load_gs_sad_sam`,
  `load_gs_sad_code_raw`, `enroll`, `load_code_model_files`, `load_sam_code`).
- **Framing**: Top-3 shows high micro-F1 but near-zero per-component macro /
  coverage → exposed as content-blind.
- **Baselines only** — no real system anchor (stays gold-only).

### Area 3 — OUT-02 paper-ready table/figure
- Emit a **booktabs `.tex` table + a CSV**, columns matching the paper's
  inequality tables (`tab:sent_gini` / `tab:samcode_skew` style: project, Gini,
  top-k, min/median/max).
- Emit a **pgfplots `\addplot` snippet** over the Phase-1
  `reports/lorenz_sad_code_sentence.csv`.
- Emitted under **`mini-inequality/reports/`** (isolation): `out02_concentration.tex`,
  `out02_concentration.csv`, `out02_lorenz.tex`.
- Table content = the **Gini / top-k concentration** table (per-sentence + samcode).

### Area 4 — MOTIV report & deferred placeholder
- **`reports/MOTIVATION.md`** — the baseline results table + an explicit
  driver→metric map.
- **Fill the trivial-baseline file-F1 placeholder NOW** (intro.tex:64) with the
  computed Top-3 (and/or random) sad-code file-level micro-F1; leave the
  strongest-pipeline and \approach file-F1 placeholders **deferred** (need
  published system scores). Note the resolved value (update CLAIM_CHECK context in
  the report / a small note).
- **Self-check**: assert Top-3 micro-F1 > random micro-F1 (the exploitation claim
  holds); fail loud (non-zero exit) otherwise.
- **Suite mapping**: enrollment inflation → file-level F1 caveat; component
  concentration → per-component F1; long-tail coverage gaps → sentence coverage;
  narrative/non-link sentences → noise rate.

### Claude's Discretion
- Exact CSV/TeX column order, MOTIVATION.md layout, k for Top-k (default 3),
  whether to also report per-sentence macro F1 (nice-to-have).
</decisions>

<code_context>
## Existing Code Insights (copy/reuse)
- `mini-inequality/inequality.py` — `import inequality` to reuse gold loaders +
  `enroll` + `load_sam_code` + helpers (self-contained reuse).
- COPY verbatim (no import): `prf`, `sentence_coverage`, `noise_rate` from
  `../mini-src/metrics.py`; `baseline_top3_by_gold_links` (L205), `baseline_random`
  (L181), `per_component_macro_f1` (L227), `measure` (L319) from
  `../src/bias/rq2_doc_to_model_prestudy.py`.
- Reference numbers (existing reports, for sanity): doc-to-model Top-3 micro-F1
  ≈ 0.38 vs random ≈ 0.12 (3.1×) — `reports/RQ2_DOC_TO_MODEL_PRESTUDY.md:101`.
  Top-3 should clearly beat random on both tasks.
- `random` is stdlib (allowed); seed via `random.Random(0)`. Path-relativity:
  `motivation.py` at the study root, run from `mini-inequality/`.

### sad-sam orientation gotcha
`inequality.load_gs_sad_sam` returns `(modelElementID, sentence)`. The baseline /
metric helpers expect `(sentence, component)` pairs — flip to `(s, c)` before use.
</code_context>

<canonical_refs>
## Canonical References
- `../mini-src/metrics.py` — `prf`, `sentence_coverage`, `noise_rate` (copy).
- `../src/bias/rq2_doc_to_model_prestudy.py` — baseline + macro-F1 + measure defs
  (copy; L181-328).
- `../src/bias/rq2_trivial_baselines.py` — doc-to-code trivial-baseline reference.
- `/mnt/hostshare/ardoco-home/alinker-paper/sections/eval.tex` (L79) + `intro.tex`
  (L64) — the Top-3/random + trivial-baseline-F1 claims this phase grounds.
- `mini-inequality/inequality.py` + `reports/*` — Phase-1 numbers (reuse for OUT-02).
- `mini-inequality/CLAIM_CHECK.md` — the deferred baseline-F1 placeholder filled here.
</canonical_refs>

<specifics>
## Specific Ideas
- The motivation: a few large components own most gold links, so a popularity-only
  Top-3 baseline scores high micro-F1 while failing per-component/coverage —
  proving micro-F1 alone is unsafe and the 4-metric suite is needed.
- Fill intro.tex:64 trivial-baseline file F1 with the computed Top-3 sad-code
  file-level micro-F1 (note it is a gold-only trivial baseline, not a real linker).
</specifics>

<deferred>
## Deferred Ideas
- Strongest-published-pipeline file F1 and \approach F1 / improvement-pp
  placeholders — need published system scores, out of this gold-only study.
</deferred>
