# Phase 2: Claim Verification — CLAIM-01, CLAIM-02, CLAIM-03 - Context

**Gathered:** 2026-06-21
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — grey-area tables accepted by user (all 4 areas)

<domain>
## Phase Boundary

Turn Phase-1's gold numbers into a paper-claim audit: extract every
distributional-inequality claim the paper makes (alinker `metric.tex`/`eval.tex`/
`intro.tex` + local `writing/eval.tex` Ch1), label each MATCH / MISMATCH / STALE
against the computed value, resolve the gold-derivable `XX` placeholders in
`intro.tex`, and record it all in `mini-inequality/CLAIM_CHECK.md`.

IN scope: auditing GOLD distributional-inequality claims using the Phase-1 engine
outputs. OUT of scope: the TransArc actual-error cascade/amplification claims
(`tab:amplification` 36→3,457, block-correlation) — system-specific, recorded but
not MATCH/MISMATCH; baseline/pipeline file-F1 placeholders — need system scores,
deferred to Phase 3. No new measurement of the dataset (Phase 1 did that).

**Isolation (HARD):** all work under `mini-inequality/`, project root =
`mini-inequality/`. Never touch repo-root `.planning/` (v1.2). Commit only
`mini-inequality/**`.
</domain>

<decisions>
## Implementation Decisions

### Area 1 — Claim scope & the excluded cascade
- **Audit ALL gold-distribution claims** across all four sources: alinker
  `metric.tex` (expansion 1.0×→217.6×; per-component skew/long-tail both tasks;
  link-F1 summarises large components), alinker `eval.tex` (skew both tasks L23/25;
  long-tail dominates the average L26; expansion factor + long-tail L79/136), alinker
  `intro.tex` (XX placeholders), local `writing/eval.tex` Ch1 (tab:sent_gini,
  tab:samcode_skew, tab:sadcode_conc, tab:enrollment, 70%-on-3-sentences).
- **The cascade / error-amplification claim (`tab:amplification` 36→3,457,
  block-correlation) = SYSTEM-SPECIFIC / out-of-scope**: recorded in the checklist
  as a TransArc empirical quantity (not a gold property), labelled accordingly —
  NOT MATCH/MISMATCH. Cross-reference `reports/TRANSARC_EMPIRICAL_STUDY.md` for
  provenance, but do not audit it here.
- **Long-tail "both tasks"**: verify with BOTH sad-code AND sad-sam per-component
  Gini (both emitted by the engine).

### Area 2 — MATCH / MISMATCH / STALE rubric
- **Tolerance = the engine gate's**: Gini ±0.005, integer counts exact, factors at
  1-dp, percentages ±0.5.
- **3 labels**: MATCH = paper value agrees with computed; MISMATCH = paper value
  contradicted by computed; STALE = an unfilled `XX` placeholder or a paper number
  superseded by the current computation.
- **Evidence per row**: paper location (file:line / table label) + computed value +
  its source (`reports/INEQUALITY.md` / a specific CSV).
- **Reuse Phase-1 outputs / engine**: the audit imports the study's OWN
  `inequality.py` (self-contained reuse — allowed; the isolation rule forbids only
  `src/`/`mini-src/` imports) and/or reads `reports/*.csv`. No new dataset
  measurement.

### Area 3 — XX placeholder resolution
- **Fill the GOLD-derivable placeholders now**: "XX projects" = **5**;
  "XX complementary metrics" = **4**; "an XX% concentration of the gold mass on
  three sentences of one project" = **70%** (JabRef, = engine JabRef per-sentence
  Top-3 % = 70.0).
- **Defer baseline/pipeline file-F1 placeholders to Phase 3** (strongest-pipeline
  file F1; trivial-baseline file F1 + "within XX points"; \approach F1 + improvement
  pp) — they require system scores. Mark "deferred → Phase 3" in CLAIM_CHECK.md.
- **"70% on three sentences"**: report JabRef per-sentence Top-3 % and name the
  project (JabRef).
- **Output**: a resolved-placeholders table with paste-ready values + `intro.tex`
  line references (17, 40, 54, 64, 79).

### Area 4 — CLAIM_CHECK.md output
- **Markdown table** (claim | source loc | paper value | computed value | label |
  evidence) + a separate resolved-placeholders section, at
  **`mini-inequality/CLAIM_CHECK.md`**.
- **Produced by a small stdlib script** (`claim_check.py`) that `import inequality`
  (own engine) and/or reads `reports/*.csv`, holds the hand-authored claim→source
  map, computes labels, and writes CLAIM_CHECK.md (reproducible).
- **Self-check**: assert the paper's stated numbers equal the engine's computed
  numbers (already gate-verified in Phase 1); flag any divergence loudly (non-zero
  exit on an unexpected MISMATCH among claims expected to MATCH).
</decisions>

<code_context>
## Existing Code Insights
- `mini-inequality/inequality.py` — the Phase-1 engine. `import inequality` to reuse
  `compute_sad_code_dist`, `compute_sad_sam_dist`, `compute_samcode_skew`,
  `compute_expansion`, and the `EXPECTED` frozen-literal dict. Self-contained reuse.
- `mini-inequality/reports/*.csv` + `reports/INEQUALITY.md` — Phase-1 computed
  numbers (per-sentence Gini/Top-3, samcode Gini/top3-conc, enrollment factors).
- Path-relativity: `claim_check.py` lives at the study root; run from
  `mini-inequality/`. Paper sources are at `/mnt/hostshare/ardoco-home/alinker-paper/sections/`
  and `../writing/eval.tex`.

### Claim inventory (CLAIM-01 seed — hand-authored map)
| Claim | Source loc | Paper value | Computed (engine) | Expect |
|-------|-----------|-------------|-------------------|--------|
| Enrollment expansion range | metric.tex:11; eval.tex tab:enrollment | 1.0×→217.6× | 1.0/…/217.6×; 35.5× avg; 525→18,660 | MATCH |
| One dir decision → hundreds of pairs (JabRef) | metric.tex:11 | "hundreds" | JabRef max fan-out 972; 38→8,268 | MATCH |
| Per-component skew / long-tail both tasks | metric.tex:14-16; eval.tex L23,25 | qualitative | samcode Gini 0.400→0.694; sad-sam comp Gini>0 | MATCH |
| Per-sentence Gini range | eval.tex(local) tab:sent_gini L237-253 | 0.331→0.645 | 0.331/0.448/0.645/0.472/0.527 | MATCH |
| 3 sentences = 70% of gold (JabRef) | eval.tex(local) L258 | 70% | JabRef per-sentence Top-3 % = 70.0 | MATCH |
| SAM-CODE files-per-AE skew + JabRef top-3 98.6% | eval.tex(local) tab:samcode_skew L199-210 | Gini 0.400→0.694; 98.6% | Gini exact; JabRef top3_conc 98.6% | MATCH |
| SAD-CODE top AE 44-48% of gold | eval.tex(local) tab:sadcode_conc L214-229 | 44-48% | top-1 component link share (via engine) | MATCH |
| Cascade 36→3,457 / block correlation | eval.tex(local) tab:amplification L156-179 | 36→3,457 (96×) | (system-specific) | SYSTEM-SPECIFIC / out-of-scope |
</code_context>

<canonical_refs>
## Canonical References (downstream agents MUST read)
- `/mnt/hostshare/ardoco-home/alinker-paper/sections/metric.tex` — expansion
  1.0×→217.6× (L11); skew/long-tail both tasks (L14-16). Claim source.
- `/mnt/hostshare/ardoco-home/alinker-paper/sections/eval.tex` — skew both tasks
  (L23,25); average dominated by long tail (L26); expansion + long-tail (L79,136).
- `/mnt/hostshare/ardoco-home/alinker-paper/sections/intro.tex` — `XX` placeholders
  (L17, 40, 54, 64, 79). CLAIM-03 targets.
- `writing/eval.tex` — local Ch1 mirror: `tab:sent_gini` (L241-256, 0.331→0.645,
  70% on 3 sentences L258), `tab:samcode_skew` (L191-210), `tab:sadcode_conc`
  (L217-232), `tab:enrollment` (L57-74), `tab:amplification` (L156-179,
  system-specific/out-of-scope).
- `mini-inequality/inequality.py` + `reports/*` — Phase-1 computed numbers (reuse).
- `reports/TRANSARC_EMPIRICAL_STUDY.md` (repo) — provenance of the excluded cascade.
</canonical_refs>

<specifics>
## Specific Ideas
- `XX` resolutions enabled by Phase 1: 5 projects; 4 metrics; 70% (JabRef, 3
  sentences). Baseline/pipeline F1 placeholders deferred to Phase 3.
- Every "expected MATCH" claim should provably MATCH because Phase 1's gate already
  verified the engine reproduces these exact literals — the audit re-confirms and
  fails loud if not.
</specifics>

<deferred>
## Deferred Ideas
- Baseline/pipeline file-F1 placeholder resolution (trivial-baseline F1, strongest
  pipeline F1, \approach improvement pp) → Phase 3 (MOTIV-01 / baselines).
- Auditing the TransArc cascade/amplification claim as MATCH/MISMATCH → belongs to
  the TransArc empirical pillar, not this study.
</deferred>
