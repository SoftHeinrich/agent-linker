# Phase 9 CROSS-03 — Cross-Model Validation: s_linker13 on gpt-5.4

**Date:** 2026-05-31
**Requirements addressed:** CROSS-01, CROSS-02, CROSS-03
**Upstream summaries:**
- `09-01-SUMMARY.md` — GATE-06 harness audit (CLEAN)
- `09-02-SUMMARY.md` — BBB probe on gpt-5.4 (F1 0.8037, sanity floor cleared)
- `09-03-SUMMARY.md` — Full 5-dataset gpt-5.4 sweep (macro F1 0.9077, informational)
- `.planning/milestones/v1.0-phases/05-promote-and-ablation-artifact/ABLATION-TABLE.md` — Claude Sonnet baseline provenance

## 1. Scope

This report compares `s_linker13` on Claude Sonnet (the v1.0 PROMO-01 promoted artifact) against `s_linker13` on `gpt-5.4` across the full 5-dataset SAD-SAM benchmark (mediastore, teastore, teammates, bigbluebutton, jabref). The Claude evaluation was NOT re-run; the baseline values are reused from `results/ablation_results/ablation_20260529_215932.json` per the v1.0 final artifact. Per D-02, `s_linker14` is not evaluated as a separate arm — Phase 8 retro-designated `s_linker13` as the COMBINE artifact, and no `s_linker14.py` file exists. The single gpt-5.4 sweep therefore satisfies both CROSS-01 and CROSS-02.

## 2. Per-Dataset Comparison

| Dataset       | Claude Sonnet F1 | gpt-5.4 F1 | Δ (Claude − gpt-5.4) | gpt-5.4 source JSON                  |
|---------------|------------------|------------|----------------------|--------------------------------------|
| mediastore    | 0.984            | 0.9677     | +0.0163              | `ablation_20260531_063446.json`      |
| teastore      | 1.000            | 1.0000     | +0.0000              | `ablation_20260531_063446.json`      |
| teammates     | 0.947            | 0.7939     | +0.1531              | `ablation_20260531_063446.json`      |
| bigbluebutton | 0.821            | 0.8037     | +0.0173              | `ablation_20260531_055235.json` (Plan 09-02 probe; reused per D-03 Step 3) |
| jabref        | 1.000            | 0.9730     | +0.0270              | `ablation_20260531_063446.json`      |
| **Macro**     | **0.9506**       | **0.9077** | **+0.0429**          | (computed mean across the 5 rows above) |

The Claude Sonnet column reproduces the rounded per-dataset values published in `ABLATION-TABLE.md` (row `s_linker13`, source JSON `ablation_20260529_215932.json`, macro 0.9506).

## 3. GATE-01 Cross-Model Statement

**GATE-01 does NOT hold cross-model: macro F1 = 0.9077 < 0.93 on gpt-5.4 for s_linker13. Per the standing v2.0 policy (PROJECT.md Key Decisions; CONTEXT D-05), this is a model-provider-property finding, NOT a defect in s_linker13. The harness was audited CLEAN under GATE-06 (09-01-SUMMARY.md), no backend-specific tailoring was introduced, and the result represents the inherent capability difference of gpt-5.4 relative to Claude Sonnet on this task.**

Dominant driver: the teammates dataset accounts for the bulk of the macro gap (Δ = +0.1531, contributing ~3.06pp of the macro Δ on its own). The remaining four datasets are within ~2.7pp of Claude Sonnet — teastore is identical, mediastore / bigbluebutton / jabref each differ by < 0.03 in F1. The teammates regression pattern (over-extraction on a doc-heavy project with deep dotted-path code references) mirrors the v1.0 13d / VAR-04 negative result, where dotted-path / Java-package casing convention was shown to be regex territory that pure-LLM emission cannot reproduce reliably; that pattern is now observed again across model providers on the same dataset.

## 4. Variance Disclosure

All gpt-5.4 numbers in §2 are single-shot per dataset, per Plan 09-03's adoption of the D-09 default cost-discipline policy (no variance retests). Two pieces of jitter evidence bound the uncertainty around the reported macro F1:

- **GPT run-to-run jitter (cross-provider, prior evidence on a DIFFERENT artifact):** project memory (MEMORY.md §"GPT-5.2 Compatibility") documents GPT backends as having a ~5–12 link stdev on the V32 artifact, with the caveat "not fixable by temperature/seed." That envelope is the disclosed cross-provider variance band; a re-run on gpt-5.4 could shift individual dataset F1s by several percentage points.
- **BBB intra-provider jitter (same artifact, prior evidence):** the v1.0 Phase 6 06-SUMMARY.md observed ~4pp same-session jitter on Claude Sonnet for `s_linker13` on bigbluebutton; the gpt-5.x family's prior history (also memory) shows wider BBB variance.

Implication: a single-shot gpt-5.4 macro F1 of 0.9077, reported here, sits comfortably below the 0.93 GATE-01 threshold even when read with the jitter band in mind — the teammates Δ (15.3pp) dominates the macro gap and is too large to be jitter-explained.

## 5. Model-Provider-Property Framing (D-05 Core)

Any Claude-Sonnet-vs-gpt-5.4 gap observed here is a property of the model provider, not of `s_linker13`. The supporting evidence:

- **Harness was audited CLEAN under GATE-06.** The Plan 09-01 audit (verdict: CLEAN — see `09-01-SUMMARY.md` and `09-GATE-06-AUDIT.md`) confirmed that `llm_client.py`, `run_ablation.py`, and `s_linker13.py`'s module-level env defaults contain no benchmark-derived branching, no per-project special cases, and no per-model conditional logic.
- **No backend-specific prompt tailoring was introduced.** Plan 09-02 and Plan 09-03 each invoked gpt-5.4 via the env-override surface only (`LLM_BACKEND=openai`, `OPENAI_MODEL_NAME=gpt-5.4`, `PHASE_CACHE_DIR=./results/phase_cache_gpt54`); no edits to `run_ablation.py`, `llm_client.py`, `s_linker13.py`, `prompts.py`, or `prompts_v2.py` were made. This is reaffirmed by the GATE-06 verdict line in `09-01-SUMMARY.md` and by `09-03-SUMMARY.md`'s post-run `git status` verification.
- **The same prompts, the same code paths, and the same env-override invocation surface produce both the Claude and gpt-5.4 evidence.** The only difference between the two columns in §2 is the OpenAI-vs-Anthropic backend dispatch in `llm_client.py:_infer_backend_from_model`, which uses generic `startswith("gpt")` / `startswith("claude")` prefix dispatch with no per-version special casing.

Per the v2.0 ship rule (CONTEXT D-04, D-05; PROJECT.md Key Decisions), this report is deliberately terminal: it does not propose follow-on prompt tailoring, backend-specific judges, or any closure of the cross-provider gap. The v2.0 thesis publishes both verdicts ("holds cross-model" / "does not hold cross-model") as terminal states, and Phase 9 ships regardless of which obtains. The gap is documented here as empirical evidence about the model provider, not as a problem to chase.

## 6. Prior Cross-Model Evidence (Background Only)

For context, the project's prior cross-model evidence on a different artifact:

- V32 (a pre-v1.0 linker, NOT `s_linker13`) on gpt-5.2 scored macro 0.906 (vs Claude 0.945, Δ +0.039)
- V32 on gpt-5.4 scored macro 0.877 (-0.039 vs gpt-5.2; an OpenAI intra-family drop)

These V32 numbers are background only. They are NOT a ground truth for `s_linker13`: V32's pipeline, prompts, and rule-removal chain differ from `s_linker13`'s. The authoritative `s_linker13` / gpt-5.4 numbers are those in §2 of this report, derived directly from `ablation_20260531_063446.json` and `ablation_20260531_055235.json`. One useful contrast worth noting against the background: `s_linker13` on gpt-5.4 scores macro 0.9077 — a +3.1pp gain over V32's 0.877 on the same backend. That gain is informational, not a goal of this phase.

## 7. Requirement Closure

| Requirement | Status     | Evidence                                                                                                                                                       |
|-------------|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CROSS-01    | SATISFIED  | `s_linker13` cross-model JSON evidence exists for all 5 datasets: `results/ablation_results/ablation_20260531_063446.json` (MS/TS/TM/JAB) + `results/ablation_results/ablation_20260531_055235.json` (BBB) |
| CROSS-02    | SATISFIED  | Per D-02 collapse: `s_linker13` IS the COMBINE artifact per Phase 8 retro-designation (`08-SUMMARY.md`); no separate `s_linker14.py` exists. The single sweep above is the s_linker14 evidence. |
| CROSS-03    | SATISFIED  | This report (`09-CROSS-REPORT.md`) — per-dataset comparison table, explicit GATE-01 cross-model verdict, variance disclosure, model-provider-property framing.  |

## 8. GATE-06 Reference

The harness + adapter shim was audited CLEAN at the start of this phase (`09-01-SUMMARY.md` verdict line: *"Verdict: CLEAN — harness layer (llm_client.py + run_ablation.py + s_linker13 env defaults) carries no benchmark-derived branching, no per-project special cases, and no new GPT-only prompt files. Cross-model sweep on gpt-5.4 unblocked."*). No re-audit was performed in Plan 09-04; the Plan 09-01 verdict carries forward through Plan 09-02's BBB probe and Plan 09-03's full sweep, both of which used the env-override invocation surface only.

---
*Phase: 09-cross-gpt-5-2-cross-model-validation*
*Plan: 04 — CROSS-03 comparison report*
*Compiled: 2026-05-31*
