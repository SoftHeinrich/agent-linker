---
phase: 12-trim-ablation
plan: 12-05
title: Step 3 Inference-Time Rubric — REJECT (GATE-06 generated-rubric leakage)
status: completed
verdict: REJECT
completed: 2026-05-31
requirements: [PROMPT-01, PROMPT-02]
tags: [trim, prompt-engineering, runtime-rubric, AHE, agentic-rubrics, gate-06, leakage]
dependency-graph:
  requires: [12-01, 12-02, 12-CONTEXT]
  provides: [variant s_linker13_trim3_runtime_rubric_clean, verdict.json, generated_rubric_audit]
  affects: [13-01]
tech-stack:
  added: ["RUBRIC_BUILDER_PROMPT", "RUBRIC_BUILDER_SEED_EXAMPLE", "_trim3_fallback_count counter"]
  patterns: ["AHE (arXiv 2604.25850) + Agentic Rubrics (arXiv 2601.04171) inference-time rubric generation"]
key-files:
  created:
    - src/llm_sad_sam/linkers/experimental/s_linker13_trim3_runtime_rubric_clean.py
    - tests/test_s_linker13_trim3_runtime_rubric_registration.py
    - results/ablation_results/12_05_trim3_runtime_rubric/verdict.json
  modified:
    - run_ablation.py (CANONICAL_VARIANTS + VARIANT_SPECS; pre-listed by 12-04 commit, idempotent edit)
    - tests/fixtures/v2_0_baseline.json (variant added to missing list)
decisions:
  - "REJECT trim3_runtime_rubric: 3/3 generated rubrics leaked benchmark-component vocabulary (15 total taboo hits)."
  - "Round 2 halted at 3/5 datasets after the W4 risk path was confirmed — F1 was within tolerance, but the mechanism's adaptation is itself the leakage vector."
  - "Round 3 (gpt-5.4) skipped per strategic mode (cross-model arm runs only if Round 2 passes)."
  - "Static surface (RUBRIC_BUILDER_SEED_EXAMPLE + RUBRIC_BUILDER_PROMPT) verified GATE-06-clean: 0 taboo hits. The leakage is purely in the runtime-emitted rubric, which is the unique GATE-06 risk surface of this trim."
metrics:
  duration_min: ~26
  tasks_completed: 4
  llm_calls_round2: ~3 rubric-builder calls + extraction + judge + tier2 (mediastore + teastore full; teammates partial)
  files_modified: 4
  commits: 2 (RED + GREEN; Task 1)
---

# Phase 12 Plan 12-05: Step 3 Inference-Time Rubric — SUMMARY (REJECT)

**One-liner:** Inference-time rubric builder replacing static DOC_KNOWLEDGE_JUDGE_RULES (AHE + Agentic Rubrics — supplement Techniques 2+3) — REJECTED: every emitted rubric leaked benchmark-component vocabulary, confirming the plan-checker W4 risk path.

## Verdict

**REJECT.** GATE-06 generated-rubric audit fails: 3/3 audited Claude runs produced rubrics that incorporated project-specific component vocabulary (15 taboo hits total). The static prompt + seed example are clean — the unique GATE-06 risk surface of this trim is the rubric the builder emits at inference, and that surface fails.

| Gate | Evaluated | Result |
| --- | --- | --- |
| GATE-06 static surface (seed example + prompt template) | YES | PASS — 0 taboo hits |
| GATE-06 generated rubric (inference-time output) | YES | **FAIL — 15 taboo hits across 3 datasets** |
| GATE-01 Claude macro F1 ≥ 0.93 | NO (only 2/5 datasets completed) | not evaluable |
| GATE-01 cross-model (gpt-5.4) | NO (Round 3 skipped) | not evaluable |
| GATE-02 frozen-file invariant | YES | PASS — `git diff --quiet` clean |

## What was built (Task 1)

| Task | Commit | Files |
| --- | --- | --- |
| 1 RED: failing registration tests | c368279 | `tests/test_s_linker13_trim3_runtime_rubric_registration.py` |
| 1 GREEN: variant + rubric builder + registration | a2e0ea3 | `src/llm_sad_sam/linkers/experimental/s_linker13_trim3_runtime_rubric_clean.py`, `run_ablation.py`, `tests/fixtures/v2_0_baseline.json` |

The variant subclasses `SLinker13Clean` and overrides `_learn_document_knowledge_enriched` end-to-end. The override:

1. Runs the same extraction prompt as the parent (Step 1).
2. **NEW:** Runs a rubric-builder LLM call that receives a generic compiler-style seed example + the project document + the candidate mappings, and emits a 4-6 item rubric (Step 2).
3. Runs the judge prompt with the generated rubric replacing the static `DOC_KNOWLEDGE_JUDGE_RULES`, with the 7 worked examples preserved byte-equal (Step 3).

Fallback path: if the rubric builder returns empty after 2 attempts, degrade to the static parent rubric and increment `self._trim3_fallback_count`. Fallback was **NEVER triggered** in Round 2 — every call produced a non-empty rubric.

## Round 2 results (Claude Sonnet, partial)

### Per-dataset F1

| Dataset | F1 | P | R | FP | FN | s_linker13 baseline F1 | Delta | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mediastore | 0.9667 | 1.000 | 0.935 | 0 | 2 | 0.9841 | -1.74pp | COMPLETED |
| teastore | 1.0000 | 1.000 | 1.000 | 0 | 0 | (n/a in fixture) | (perfect) | COMPLETED |
| teammates | — | — | — | — | — | — | — | HALTED in tier2 entity extraction |
| bigbluebutton | — | — | — | — | — | — | — | NOT STARTED |
| jabref | — | — | — | — | — | — | — | NOT STARTED |

Macro F1 over the 2 completed datasets: **0.9833** (would be competitive with s_linker13_clean baseline IF the GATE-06 audit had passed — but it did not).

### F1 observation

On the 2 datasets where F1 was computable, the trim was competitive (mediastore -1.74pp, within Claude run-to-run variance) or perfect (teastore +0.0pp). This is consistent with the strategic-mode hypothesis: "F1 alone could mask the 'not actually adapting' failure mode" — except here, the OPPOSITE failure mode applied. The rubric IS strongly adapting (good F1), and the strong adaptation IS itself the leakage vector (bad GATE-06).

## Generated-rubric audit (the rejection basis)

3/3 rubrics emitted in Round 2 leaked benchmark-component vocabulary:

| Dataset | Rubric taboo hits | Unique leaked terms |
| --- | --- | --- |
| mediastore | 2 | storage, database |
| teastore | 5 | ImageProvider, Persistence, registry, UI |
| teammates | 8 | Client, Common, internal, logic, Persistence, storage, UI |
| **Total** | **15** | — |

### Specific leakage example (teammates rubric, verbatim)

> - Approve if the alias is formed by dropping 'Component' from one of the seven named architecture components **(UI, Logic, Storage, Common, Test Driver, E2E, Client)**, since the document itself consistently uses this shorthand in running prose.

The rubric explicitly enumerates 7 of the 8 Teammates benchmark components by name. This is a categorical GATE-06 failure: the runtime-generated prompt body contains the full component vocabulary of the project under evaluation.

### Why this is unfixable in the current pattern

The rubric builder is GIVEN the project document as input and asked to ground its rubric in document-specific patterns. "Document-specific" inherently means "benchmark-vocabulary-specific" when the document IS a benchmark. The prompt instruction "Produce a 4-6 item rubric grounded in patterns the document actually uses" cannot be satisfied without naming the components the document is about.

Two hypothetical mitigation paths, both rejected:

- **Path A — constrain the builder to avoid project nouns.** Would defeat the adaptation purpose (the supplement's core hypothesis is that adaptation breaks the V35 ceiling). A constrained builder reduces to a static rubric, i.e., back to the prompts_v3 baseline.
- **Path B — accept leakage, document inherently.** Reviewer-credibility cost is severe. The ICSE Universal Taboo invariant exists precisely to prevent variants that incorporate benchmark vocabulary into prompt bodies, regardless of whether the incorporation is static or runtime.

The verdict is REJECT, and the rejection generalizes: the inference-time rubric pattern (per AHE + Agentic Rubrics) is **NOT GATE-06-safe for evaluation on a closed benchmark set**. It may be GATE-06-safe in production deployments where the input documents are not themselves the evaluation gold standard, but that is a different setting from v2.1.

## Mechanism-adaptation check (the strategic-mode signal)

The strategic mode flagged TWO failure modes:

1. **F1 dropped > 3pp:** Mechanism regressing on output quality. **DID NOT OCCUR.**
2. **Rubric is identical generic across docs:** Mechanism not actually adapting. **DID NOT OCCUR — the opposite occurred.**

The rubrics were highly distinct across the 3 datasets — each cleanly mapped to the input document's surface vocabulary. The mechanism works exactly as designed. The W4 risk path identified by the plan-checker was specifically: "if all 5 generated rubrics look identical across datasets, flag as 'mechanism suspect even if F1 passes'". The observed failure is the OPPOSITE mode: the rubrics ARE distinct, and the distinctiveness IS what leaks taboo.

## Threat model resolution

| Threat ID | Disposition planned | Outcome |
| --- | --- | --- |
| T-12-05-01 (rubric builder emits benchmark vocabulary) | mitigate via Task 4 audit; REJECT if hits > 0 | **Confirmed; trim REJECTED.** |
| T-12-05-02 (accidental frozen-file edits) | mitigate via git diff --quiet check | PASS — frozen files unchanged. |
| T-12-05-03 (rubric builder consistently fails, silent no-op) | track fallback_count | Fallback NEVER triggered; mechanism was always live. |
| T-12-05-04 (which rubric used in which run not recorded) | mitigate via sweep.log | sweep.log captured for the 3 emitted rubrics. |
| T-12-05-05 (extra LLM call cost) | accept per CLAUDE.md | ~3 rubric-builder calls spent in Round 2; cost-acceptable. |

## Files

| Created | Modified |
| --- | --- |
| `src/llm_sad_sam/linkers/experimental/s_linker13_trim3_runtime_rubric_clean.py` | `run_ablation.py` (idempotent — entries pre-listed by 12-04 commit 6cd66d0) |
| `tests/test_s_linker13_trim3_runtime_rubric_registration.py` | `tests/fixtures/v2_0_baseline.json` (added to missing list) |
| `results/ablation_results/12_05_trim3_runtime_rubric/verdict.json` | — |
| `results/ablation_results/12_05_trim3_runtime_rubric/claude/{mediastore,teastore}/layer1.json` | — |
| `results/ablation_results/12_05_trim3_runtime_rubric/claude/sweep.log` (3 rubrics) | — |
| `.planning/phases/12-trim-ablation/12-05-SUMMARY.md` (this file) | — |

(Frozen files NOT modified: prompts_v2.py, s_linker13.py, s_linker13_clean.py, data_types_v2.py, document_loader_v2.py, pcm_parser_v2.py.)

## Deviations from plan

### Halted before Round 3 (gpt-5.4)
- **Found during:** Round 2 dataset 3 (teammates) generated rubric audit.
- **Issue:** GATE-06 audit failed on 3/3 Round 2 datasets with rubric leakage. Strategic mode authorized HALT.
- **Action:** Killed the bg sequential ablation process after teammates rubric was emitted (and confirmed taboo-laden) but before teammates F1 was computed; bigbluebutton + jabref not started; Round 3 (gpt-5.4) skipped per strategic mode "Round 3 only if Round 2 passes".
- **Tracked as:** [Strategic-mode deviation — early halt on confirmed W4 risk signal] — not a Rule 1-4 deviation; explicit strategic gate.

### No fix applied (Rule 1-3 inapplicable)
- The pattern's failure mode is INHERENT to the inference-time rubric design (rubric builder is given the document and asked to ground in it; leakage follows). Rule 1 (auto-fix bug) inapplicable: not a bug. Rule 2 (auto-add missing functionality) inapplicable: nothing missing — the spec was implemented exactly. Rule 3 (auto-fix blocking issue) inapplicable: nothing was blocking.
- Rule 4 (architectural change) was NOT escalated to the user because the strategic-mode gate explicitly authorizes the executor to record REJECT and document; this is the intended halt protocol for plan-checker W4 risks.

## Stub tracking

None — no stubs introduced. The variant is a fully wired subclass with a complete override.

## Threat flags (new surface discovered)

None — no new threat surface discovered beyond the threat-model already enumerated in the plan. T-12-05-01 was the predicted threat; it materialized exactly as predicted.

## Requirements progress

- **PROMPT-01:** "v2 → v3 mapping" entry for DOC_KNOWLEDGE_JUDGE_RULES under Step 3 updates to "Inference-time rubric — REJECTED (GATE-06 generated-rubric leakage)" under Plan 12-06's mapping-doc maintenance.
- **PROMPT-02:** Highest-risk trim ablated, verdict recorded with explicit generated-rubric leakage audit. The promise of "single-step ablation per trim" was honored within strategic-mode constraints (Round 2 partial halt).

## Downstream signals

- **Plan 12-06 (Gate-06 + defensibility audit):** This trim should be the PRIMARY AUDIT CASE STUDY for the milestone defensibility narrative. The audit story is: "Phase 12 took the strongest theoretical V35-escape mechanism, implemented it faithfully per supplement Techniques 2+3, ablated it under strict GATE-06, and recorded an honest negative result. The static surface was clean; the runtime-generated rubrics leaked benchmark vocabulary on 3/3 datasets. The trim is rejected. This is the kind of negative result a reviewer-defensible methodology MUST be able to report."
- **Plan 13-01 (Promotion):** The trim is NOT carried forward. `s_linker13_min` excludes the trim3 mechanism.

## Self-Check: PASSED

Verification checks:
- `src/llm_sad_sam/linkers/experimental/s_linker13_trim3_runtime_rubric_clean.py` — EXISTS.
- `tests/test_s_linker13_trim3_runtime_rubric_registration.py` — EXISTS, 11 tests pass.
- `results/ablation_results/12_05_trim3_runtime_rubric/verdict.json` — EXISTS.
- `results/ablation_results/12_05_trim3_runtime_rubric/claude/mediastore/layer1.json` — EXISTS (F1 0.9667).
- `results/ablation_results/12_05_trim3_runtime_rubric/claude/teastore/layer1.json` — EXISTS (F1 1.0).
- `results/ablation_results/12_05_trim3_runtime_rubric/claude/sweep.log` — EXISTS (153 lines, 3 emitted rubrics).
- Commit c368279 (RED) — present in `git log`.
- Commit a2e0ea3 (GREEN) — present in `git log`.
- GATE-02 (`pytest tests/test_v20_baseline_regression.py -q`) — PASSED (35 passed, 20 xfailed; trim3 in 'missing' list).
- Frozen files unchanged: `git diff --quiet` on the 6-file frozen set — PASSED.
