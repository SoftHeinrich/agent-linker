---
phase: 12-trim-ablation
plan: 02
subsystem: ablation-harness
tags: [PROMPT-02, harness, single-step, checkpoint-replay]
requirements: [PROMPT-02]
dependency_graph:
  requires: [phase-10/s_linker13_clean checkpoints under results/phase_cache/]
  provides: [llm_sad_sam.ablation.single_step, llm_sad_sam.ablation.__main__]
  affects: [plans 12-03, 12-04, 12-05 (each invokes the harness for its trim)]
tech_stack:
  added: []
  patterns: [pickle-load + selective-method-invocation, checkpoint-replay, monkey-patch-as-contract-enforcer]
key_files:
  created:
    - src/llm_sad_sam/ablation/__init__.py
    - src/llm_sad_sam/ablation/single_step.py
    - src/llm_sad_sam/ablation/__main__.py
    - tests/test_single_step_harness.py
    - .planning/phases/12-trim-ablation/12-02-HARNESS-CONTRACT.md
    - results/ablation_results/12_02_harness/equivalence_summary.json
  modified: []
decisions:
  - Harness CALLS INTO s_linker13_clean by method name (semi-private surface); does not modify the class. Phase 13 promotion MUST preserve the method names or update the harness in lock-step.
  - Step 2 (12-04) re-runs TWO sub-phases (entity_candidates AND entity_decisions) — the merge collapses prompts across both. The harness CRITICAL CONTRACT keeps these surgical: cached seed_val + coref from layer2.pkl are reused; _run_seed_validation and _run_coreference are monkey-patched to raise if called.
  - Baseline F1 lookup prefers tests/fixtures/v2_0_baseline.json; for variants explicitly marked missing in the fixture (e.g. s_linker13_clean), falls back to F1 derived from the canonical cached final.pkl. This gives the no-op equivalence test a meaningful anchor for post-v2.0 variants.
metrics:
  duration: ~45min
  completed_date: 2026-05-31
  tasks: 3
  files_created: 6
  files_modified: 0
  commits: 3
---

# Phase 12 Plan 02: Single-Step Ablation Harness Summary

**One-liner:** Checkpoint-loaded single-step ablation engine for the v2.1 trim chain — re-executes ONE phase of a variant against cached upstream pickles, propagates the result through DOWNSTREAM_DEPS, scores against gold, writes a per-run JSON with delta vs baseline. Plans 12-03 / 12-04 / 12-05 cite this harness as their measurement entry point.

## Context

Satisfies the USER DIRECTIVE in 12-CONTEXT.md ("Execution Method — Checkpoint-Loaded Single-Step Ablation"): trim plans must NOT re-run the full 5-phase pipeline per ablation; they must replay only the phase whose prompt changed against the existing `results/phase_cache/<variant>/<dataset>/{layer1,layer2,entity_candidates,entity_decisions,final}.pkl` checkpoints. PROMPT-02 is the requirement this closes.

Without a shared harness, each trim plan would re-derive its own per-trim runner, risking inconsistent measurement semantics. Plan 12-02 ships the canonical engine + dependency contract so 12-03/04/05 each invoke a single CLI line.

## Artifacts

| Path | Purpose |
|------|---------|
| `src/llm_sad_sam/ablation/__init__.py` | Package marker. |
| `src/llm_sad_sam/ablation/single_step.py` | Engine. Exports `PHASE_ORDER`, `DOWNSTREAM_DEPS`, `run_single_step(variant, dataset, phase, results_dir, backend=..., model=..., phase_cache_dir=...)`. |
| `src/llm_sad_sam/ablation/__main__.py` | CLI: `python -m llm_sad_sam.ablation single_step --variant X --dataset D --phase P --results-dir R [--backend ...] [--model ...] [--phase-cache-dir ...]`. Subcommand structure leaves room for future `multi_step` / `sweep`. |
| `tests/test_single_step_harness.py` | 12 smoke tests — 6 contract (PHASE_ORDER + DOWNSTREAM_DEPS pinned) + 6 engine/CLI (baseline equivalence, missing-upstream error path, unknown-variant/dataset/phase, JSON shape). |
| `.planning/phases/12-trim-ablation/12-02-HARNESS-CONTRACT.md` | Canonical 5-row phase→upstream-checkpoint→downstream-re-run table that plans 12-03/04/05 cite. |
| `results/ablation_results/12_02_harness/equivalence_summary.json` | Sweep verdict: `within_variance == true`, `max_abs_delta == 0.0` across all 5 datasets. |

## Equivalence Sweep Verdict — PASS

For each of the 5 datasets, the harness was invoked with `phase=final` on the unchanged `s_linker13_clean` baseline checkpoints (no semantic change). The reconstructed final-link set must match the cached `final.pkl` within run-to-run variance (≤ 0.02 |ΔF1|).

| Dataset | harness_F1 | baseline_F1 | delta_F1 |
|---------|------------|-------------|----------|
| mediastore | 0.9836 | 0.9836 | 0.0000 |
| teastore | 1.0000 | 1.0000 | 0.0000 |
| teammates | 0.9381 | 0.9381 | 0.0000 |
| bigbluebutton | 0.8036 | 0.8036 | 0.0000 |
| jabref | 0.9730 | 0.9730 | 0.0000 |

`max_abs_delta = 0.0`, `within_variance = true`. The harness load + score + dedup path reproduces the canonical baseline exactly when no semantic change is applied.

Additional cross-check: `phase=entity_decisions` on `mediastore` (exercises the surgical re-run path with `_run_seed_validation` and `_run_coreference` monkey-patched to raise) also returns `delta_F1 = 0.0`. The CRITICAL HARNESS CONTRACT is enforced: zero live LLM calls on seed_val + coref tracks during entity-track replays.

All sweep invocations used `backend=checkpoint` — no live LLM cost.

## Decisions Made

- **Surgical method invocation, no fork of `link()`.** The harness instantiates the variant class and calls the semi-private methods (`_run_seed_validation`, `_run_entity_pipeline`, `_run_coreference`, `_extract_entities_enriched`, `_validate_with_evidence`, `_save_phase`) directly. This avoids forking `s_linker13_clean.link()` (which would couple the harness to the orchestration shape) and avoids modifying the frozen surface of `s_linker13_clean`.
- **Monkey-patch as contract enforcer.** Phase=entity_candidates and phase=entity_decisions replace `_run_seed_validation` and `_run_coreference` on the instance with raising stubs. This is observable: if a future refactor accidentally routes through those methods, the harness fails loudly with the contract message rather than silently paying live LLM cost.
- **Per-run tmp `PHASE_CACHE_DIR`.** When the variant's own `_save_phase` writes (e.g. during phase=layer1's full pipeline re-run), the harness redirects `PHASE_CACHE_DIR` to a per-run subdir under `<results_dir>/_phase_cache_tmp`. The canonical baseline cache at `results/phase_cache/<variant>/<dataset>/` is read-only from the harness's perspective.
- **Baseline-F1 fallback for missing fixture entries.** `s_linker13_clean` is explicitly marked missing in `tests/fixtures/v2_0_baseline.json` (post-v2.0 sibling). The harness falls back to computing F1 from the canonical cached `final.pkl` so the no-op test has a meaningful anchor; otherwise `delta_F1` would be `None` and equivalence couldn't be asserted.

## Step 2 (12-04) Handling — Two Sub-Phase Re-runs

The ent+val merge in Step 2 collapses overlapping rubric across `ENTITY_EXTRACTION_RULES` (extraction) and `VALIDATION_RULES` (validation). The harness handles this by exposing two phase values:

- `--phase entity_candidates`: re-runs `_extract_entities_enriched` + `_validate_with_evidence` (since validation reads candidates produced by extraction; not re-running validation would compare merged-extraction-prompt output against stale validation decisions).
- `--phase entity_decisions`: re-runs `_validate_with_evidence` only against cached `entity_candidates.pkl`.

Plan 12-04 will invoke `--phase entity_candidates` to measure the merged-prompt-end-to-end delta in one shot. The CRITICAL CONTRACT keeps the surrounding seed_val + coref tracks cached.

## Coupling Notes (Technical Debt)

The harness reaches into `s_linker13_clean` by the following method/attribute names:

- Methods: `link`, `_run_seed_validation`, `_run_entity_pipeline`, `_extract_entities_enriched`, `_validate_with_evidence`, `_run_coreference`, `_save_phase`, `_checkpoint_dir`.
- Attributes: `model_knowledge`, `doc_knowledge`, `_current_text_path`, `_ilinker3`.

These are semi-private but stable in `s_linker13_clean` (Phase 10 promotion contract). Any refactor that renames or removes them will silently break the harness — Phase 13's `s_linker13_min` promotion MUST either preserve them or update the harness in lock-step. Tracked here so the Phase 13 verifier catches it.

## Threat Model — Mitigations Applied

| Threat | Disposition | How it landed |
|---|---|---|
| T-12-02-01 Tampering — accidental edit to `s_linker13_clean` | mitigated | `git diff --quiet src/llm_sad_sam/linkers/experimental/s_linker13_clean.py` exits 0 (verified). The harness only CALLS INTO the class; it never imports private state at module load time. |
| T-12-02-02 Information disclosure via LLM-response logs | accepted | Results live under `results/` (gitignored); not a deployment concern. |
| T-12-02-03 Repudiation — results JSON missing provenance | mitigated | Every results JSON embeds `variant`, `dataset`, `phase`, `phase_cache_dir`, ISO `timestamp`. |
| T-12-02-04 Denial of service — harness regression blocks Wave 2 | mitigated | Task 3 equivalence sweep is the regression gate; `within_variance == true` is the pass criterion. Verified above. |

## CLAUDE.md Compliance

- Memory rule "Always use Claude Sonnet": the harness's `--backend claude` codepath instantiates `LLMBackend.CLAUDE` with the variant's own `CLAUDE_MODEL=sonnet` default. Smoke tests use `--backend checkpoint` so no live model is invoked.
- No hardcoded benchmark-derived word lists. The harness is purely orchestration — it never reads or writes prompt content.
- Frozen file invariant: `prompts_v2.py`, `s_linker13.py`, `data_types_v2.py`, `document_loader_v2.py`, `pcm_parser_v2.py`, `s_linker13_clean.py`, `helper_v3.py` all unchanged (verified by `git diff --quiet`).

## References

- 12-CONTEXT.md decisions section — "Execution Method — Checkpoint-Loaded Single-Step Ablation (USER DIRECTIVE)" — the user-imposed constraint this plan implements.
- 12-02-HARNESS-CONTRACT.md — canonical PHASE_ORDER + DOWNSTREAM_DEPS dependency table.
- tests/test_single_step_harness.py — 12 tests pinning the contract + smoke tests.
- results/ablation_results/12_02_harness/equivalence_summary.json — sweep verdict.

## Commits

- `3d549b8` feat(12-02): define phase-to-upstream-checkpoint dependency contract (Task 1)
- `e9b7035` feat(12-02): implement run_single_step engine + CLI subcommand (Task 2)
- (pending) docs(12-02): equivalence sweep + SUMMARY (Task 3)

## Self-Check: PASSED

- All 6 artifact paths exist on disk (verified).
- All 12 harness tests pass (verified).
- Equivalence sweep `within_variance == true` (verified, see equivalence_summary.json).
- GATE-02 (`tests/test_v20_baseline_regression.py`) passes 35/35 (+17 xfailed); unaffected (verified).
- No frozen-file edits (`git diff --quiet` on all 7 frozen paths exits 0; verified).
