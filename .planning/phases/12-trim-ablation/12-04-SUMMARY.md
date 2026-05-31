---
plan: 12-04
phase: 12
title: Step 2 Entity + Validation Merge (ent+val)
status: rejected
completed: 2026-05-31
requirements: [PROMPT-01, PROMPT-02]
verdict: REJECT
duration: ~25min (Round 1 + Round 2)
files_created:
  - src/llm_sad_sam/linkers/experimental/s_linker13_trim2_entval_clean.py
  - tests/test_s_linker13_trim2_entval_registration.py
  - results/ablation_results/12_04_trim2_entval/verdict.json
  - results/ablation_results/12_04_trim2_entval/claude/s_linker13_trim2_entval_clean/<5 datasets>/entity_*.json
  - .planning/phases/12-trim-ablation/12-04-SUMMARY.md
files_modified:
  - run_ablation.py (CANONICAL_VARIANTS + VARIANT_SPECS entries; canonical=False)
  - tests/fixtures/v2_0_baseline.json (added trim2 variant to 'missing' list)
files_unchanged_frozen:
  - prompts_v2.py, s_linker13.py, s_linker13_clean.py, data_types_v2.py,
    document_loader_v2.py, pcm_parser_v2.py
---

# Plan 12-04 — Step 2 Entity + Validation Merge — SUMMARY

## Verdict: REJECT

The merged ENTITY_EXTRACTION_RULES + VALIDATION_RULES rubric fails GATE-01
Claude on macro-F1 floor AND on the bigbluebutton per-dataset tolerance.
Both gate arms must pass for ACCEPT. The variant is NOT carried forward to
Plan 12-06 (defensibility audit) or Plan 13-01 (s_linker13_min promotion).

### Gate Failures (Claude Sonnet)

| Gate                       | Required        | Observed | Status |
|----------------------------|-----------------|----------|--------|
| macro F1                   | >= 0.93         | 0.9235   | FAIL   |
| bigbluebutton delta vs baseline | >= -0.06   | -0.0659  | FAIL   |
| mediastore delta vs baseline    | >= -0.02   | -0.0159  | PASS   |
| teastore delta vs baseline      | >= -0.02   | +0.0000  | PASS   |
| teammates delta vs baseline     | >= -0.02   | +0.0011  | PASS   |
| jabref delta vs baseline        | >= -0.02   | +0.0000  | PASS   |
| GATE-06 benchmark-leakage probe | empty match  | empty    | PASS   |

### Gate Status: gpt-5.4 (Cross-Model)

**SKIPPED.** Per strategic execution plan, Round 3 is skipped when the
Claude arm fails — both gate arms must pass for ACCEPT. No gpt-5.4 ablation
ran; no live API cost incurred.

---

## Strategic Execution Path Taken

### Round 1 — Claude probe on mediastore

Both `entity_candidates` and `entity_decisions` phases probed
(per CRITICAL HARNESS CONTRACT — Step 2 spans both sub-phases). Decision
gate: F1 must be within 3pp of baseline to proceed to Round 2.

| Phase             | F1     | delta_F1 | fp | fn |
|-------------------|--------|----------|----|----|
| entity_candidates | 0.9677 | -0.0159  | 1  | 1  |
| entity_decisions  | 0.9677 | -0.0159  | 1  | 1  |

Result: **PROCEED to Round 2** (delta well within 3pp tolerance).

### Round 2 — Claude × 5 datasets

Single-step ablation via 12-02 harness, phase=`entity_candidates` which
cascades into `entity_decisions` + `final` per DOWNSTREAM_DEPS. The
variant's monkey-patch overrides both `ENTITY_EXTRACTION_RULES` (extraction)
and `VALIDATION_RULES` (validation) during the cascade. Per the CRITICAL
HARNESS CONTRACT, `_run_seed_validation` and `_run_coreference` are blocked
during the surgical re-run — zero live LLM calls on the seed_val and coref
tracks.

| Dataset        | F1     | Precision | Recall | FP | FN | baseline F1 | delta_F1 |
|----------------|--------|-----------|--------|----|----|-------------|----------|
| mediastore     | 0.9677 | 0.9677    | 0.9677 |  1 |  1 | 0.9836      | -0.0159  |
| teastore       | 1.0000 | 1.0000    | 1.0000 |  0 |  0 | 1.0000      | +0.0000  |
| teammates      | 0.9391 | 0.9310    | 0.9474 |  4 |  3 | 0.9381      | +0.0011  |
| bigbluebutton  | 0.7377 | 0.7500    | 0.7258 | 15 | 17 | 0.8036      | -0.0659  |
| jabref         | 0.9730 | 0.9474    | 1.0000 |  1 |  0 | 0.9730      | +0.0000  |
| **macro F1**   | **0.9235** |       |        |    |    |             |          |

### Round 3 — gpt-5.4 cross-model

**SKIPPED.** Round 2 failed the Claude arm; the strategic execution plan
mandates skipping Round 3 in this case because both gate arms must pass
for ACCEPT.

---

## FP / FN Delta vs Baseline (Claude)

Baseline = `s_linker13_clean` cached `final.pkl` (the Phase 10 anchor used
by the harness's fallback when v2.0 fixture has the variant under
`missing`).

| Dataset        | FP delta | FN delta | Net change |
|----------------|----------|----------|------------|
| mediastore     | +0       | +1       | +1 error   |
| teastore       | +0       | +0       | 0          |
| teammates      | -1       | +2       | +1 error   |
| bigbluebutton  | +7       | +6       | +13 errors |
| jabref         | +1       | -1       | 0          |

BBB is the dominant failure: +13 errors (7 FP, 6 FN added). The merged
rubric loses both precision and recall on BBB simultaneously.

---

## Rule-Count Reduction

| Constant                  | v2 rules | v3 rules | delta |
|---------------------------|----------|----------|-------|
| ENTITY_EXTRACTION_RULES   | 6 incl + 2 excl = 8 | (merged into shared core) | — |
| VALIDATION_RULES          | 3 APPROVE + 3 REJECT = 6 | (merged into shared core) | — |
| ENTVAL_MERGED_RUBRIC_V3   | —        | 10 (6 incl + 4 excl) | — |
| Combined                  | 14       | 10       | **-4** |

The 4-rule reduction matches the Phase 11 survey §5 row 2 estimate
("estimated 4-rule reduction"). The reduction is rubric-shared /
decision-divergent: a single shared rubric core wrapped by two role-specific
headers (extraction proposer + validation judge).

---

## Failure Analysis

### Why BBB collapsed

BBB has the highest variance in component-mention surface forms:
PubSub/streaming components named after technologies (FreeSWITCH, kurento,
Redis), compound names (Recording Service, HTML5 Server), and component
references through interaction patterns. The original
ENTITY_EXTRACTION_RULES had 6 inclusion rules tuned by V31/V32 work for
exactly this surface, including the load-bearing "Favor inclusion" tie-breaker.

The merged rubric retains "Favor inclusion" on the extraction-side wrapper
but folds the EXTRACTION-side inclusion criteria together with the
VALIDATION-side REJECT criteria into a single body. This changes the
LLM's reading: the proposer now sees the rejection criteria in the same
list as the inclusion criteria, which biases against borderline cases that
V31's two-stage design intentionally allowed through (extract aggressively,
validate strictly).

### Failure mechanism (consistent with V35a lesson)

This matches the V35a finding documented in `MEMORY.md` and Phase 11
survey §6: even "lossless" rule rewrites that collapse information density
or merge prompt boundaries tend to regress Claude on the dataset with the
highest surface-form variance (BBB). The V31 prompt design exploits
information density Claude leverages; merging the two prompts erases the
extraction-vs-validation boundary the LLM appears to use as a signal.

### Why teammates / jabref / teastore / mediastore held

These datasets have lower surface variance and more canonical naming:
- teastore: all components are CamelCase, distinct, dominant by exact match
- jabref: small component set, high overlap with English vocabulary already
  handled by the merged rubric's rule 8 (ordinary English word exclusion)
- teammates: mid-variance, +1 error net but within tolerance
- mediastore: -1 FN, marginal regression within tolerance

---

## Cross-Reference to Plan 12-03 (Trim1 Judge)

Plan 12-03 (Step 1 — judge trim) ran in parallel. Its outcome is recorded
in its own SUMMARY (not authored by this plan). The trim2 entval merge is
independent of trim1: each modifies different prompts (trim1 modifies
DOC_KNOWLEDGE_JUDGE_*, trim2 modifies ENTITY_EXTRACTION_RULES + VALIDATION_RULES).
The REJECT verdict on trim2 does not affect trim1's status.

---

## Artifacts Created

### Source
- `src/llm_sad_sam/linkers/experimental/s_linker13_trim2_entval_clean.py`
  - SLinker13Trim2EntvalClean subclass of SLinker13Clean
  - ENTVAL_MERGED_RUBRIC_V3 (10-rule shared core)
  - ENTITY_EXTRACTION_RULES_V3 = extraction header + shared core
  - VALIDATION_RULES_V3 = validation header + shared core
  - Override surface: `_run_single_extraction_pass` + `_validate_with_evidence`

### Tests
- `tests/test_s_linker13_trim2_entval_registration.py` (12 tests, all passing)
  - Imports + class identity
  - Shared-core substring contract
  - Rule-count contraction (10 ≤ 10)
  - Coverage preservation (9 semantic markers present)
  - GATE-06 benchmark-leakage probe
  - "Favor inclusion" preservation
  - Role-specific framings detectable
  - Frozen-file safety (git diff --quiet)
  - Smoke instantiation

### Results
- `results/ablation_results/12_04_trim2_entval/verdict.json` — REJECT verdict
- `results/ablation_results/12_04_trim2_entval/claude/s_linker13_trim2_entval_clean/<5 datasets>/entity_candidates.json`
- `results/ablation_results/12_04_trim2_entval/claude/sweep.log`
- `results/ablation_results/12_04_trim2_entval/claude/round1_mediastore_*.log`

### Registration
- `run_ablation.py` — CANONICAL_VARIANTS + VARIANT_SPECS entry, canonical=False
- `tests/fixtures/v2_0_baseline.json` — variant added to 'missing' list

---

## Frozen-File Safety (T-12-04-01)

`git diff --quiet` against:
- `src/llm_sad_sam/linkers/experimental/prompts_v2.py`
- `src/llm_sad_sam/linkers/experimental/s_linker13.py`
- `src/llm_sad_sam/linkers/experimental/s_linker13_clean.py`
- `src/llm_sad_sam/core/data_types_v2.py`
- `src/llm_sad_sam/core/document_loader_v2.py`
- `src/llm_sad_sam/pcm_parser_v2.py`

All exit 0. Zero edits to v2.0 frozen files or to s_linker13_clean.

GATE-02 unaffected (`pytest tests/test_v20_baseline_regression.py -q`:
35 passed, 20 xfailed — the trim2 variant is in `missing` with all 5
datasets nulled, xfailing cleanly).

---

## Requirements Status

- **PROMPT-01** (v2 → v3 mapping): trim2 merged rubric documented but
  NOT promoted to `prompts_v3.py` (verdict REJECT). The two original
  constants stay in `prompts_v3.py` unchanged from the v2 form. The v2→v3
  mapping table should annotate: "ENTITY_EXTRACTION_RULES + VALIDATION_RULES
  — investigated for merge via Technique 3; rejected (BBB regression -6.6pp);
  kept as separate constants."

- **PROMPT-02** (per-prompt trim ablation): trim2 step completed with
  explicit REJECT verdict. The variant remains in CANONICAL_VARIANTS for
  future re-investigation (e.g., decision-divergent / rubric-shared with
  stricter extraction-side header preserving the "broad recall" stance).

---

## Carry-Forward Status

- **Plan 12-06 (defensibility audit):** trim2 variant NOT included as
  primary audit target. The merged rubric body is retained in the variant
  source file for archival reference but is not promoted to `prompts_v3.py`.
- **Plan 13-01 (s_linker13_min promotion):** trim2 NOT carried forward.

The variant file + tests + registration remain in the codebase as ablation
infrastructure (re-runnable for future investigation if a stricter
extraction header design is proposed).

---

## Deviations from Plan

None. Plan executed exactly as written, including the strategic 3-round
execution path (Round 1 probe → Round 2 full Claude → Round 3 gpt-5.4 cross-model).
Round 3 was correctly skipped per the decision gate (Round 2 failed → skip
cross-model arm; both arms must pass for ACCEPT).

One minor operational detail: the harness expects `<phase_cache>/<variant>/`
to exist for upstream checkpoints. Per the Plan 12-00 reuse-of-baseline
decision (s_linker13_clean cache is the upstream anchor), a symlink was
created:
  `results/phase_cache/s_linker13_trim2_entval_clean -> s_linker13_clean`

This is the same pattern Plan 12-03 will use (or already uses) for trim1.
Documented here for reproducibility but not a substantive deviation.

---

## Self-Check: PASSED

- variant file exists: `src/llm_sad_sam/linkers/experimental/s_linker13_trim2_entval_clean.py`
- tests exist + pass: `tests/test_s_linker13_trim2_entval_registration.py` (12/12)
- verdict.json exists: `results/ablation_results/12_04_trim2_entval/verdict.json`
- 5 Claude result JSONs exist: mediastore, teastore, teammates, bigbluebutton, jabref
- Frozen files unchanged: `git diff --quiet` against all 6 frozen modules — clean
- GATE-02 regression: 35 passed, 20 xfailed
- GATE-06 probe: clean (enforced by registration test)
- Commits:
  - 6cd66d0 (variant + tests + run_ablation registration)
  - 8b1868b (Claude ablation results + verdict)
