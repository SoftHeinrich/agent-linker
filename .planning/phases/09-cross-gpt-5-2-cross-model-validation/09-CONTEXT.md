# Phase 9: CROSS — Cross-Model Validation — Context

**Gathered:** 2026-05-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Run the COMBINE artifact (= `s_linker13`, per Phase 8 retro-designation) on a non-Claude LLM backend across all 5 datasets, produce JSON results, and write a comparison report that frames any Claude-vs-other-model gap as a **model-provider-property finding**, not a defect to fix (standing v2.0 policy).

In scope:
- Cross-model harness invocation on `s_linker13.py` against the chosen non-Claude model id
- BBB-first hybrid execution: short single-dataset probe on BBB → reasonableness check → full 5-dataset sweep on go-decision
- Per-dataset + macro F1 JSON outputs
- Comparison markdown report (CROSS-03)
- GATE-06 audit of the harness + model-adapter shim (no benchmark values, no per-project branching)

Out of scope:
- Re-running Claude Sonnet (Claude baseline numbers already established: 0.9506 macro from `ablation_20260529_215932.json`)
- `s_linker14` separate arm — CROSS-02 collapses into CROSS-01 per Phase 8 retro-designation (s_linker13 IS the COMBINE artifact, no s_linker14.py file exists)
- New rule removals, new linker variants, prompt redesigns (v2.0 finishing phase)
- Backend switch to Opus or production-target reframing (Sonnet remains development model per standing user preference)

</domain>

<decisions>
## Implementation Decisions

### Cross-Model Target

- **D-01:** **Cross-model evaluation target = `gpt-5.4`** (OpenAI). User-selected 2026-05-31 as "newest balanced model" for this evaluation. Note: project memory documents `gpt-5.4` scored worse than `gpt-5.2` on V32 (87.7% vs 89.1% macro F1) — but that was on V32, not `s_linker13`. Phase 9 re-tests `gpt-5.4` on the v2.0 COMBINE artifact (`s_linker13`).
- **D-02:** **Single arm only.** CROSS-01 (s_linker13 on gpt-5.4) AND CROSS-02 (s_linker14 on gpt-5.4) collapse — Phase 8 retro-designated s_linker13 as the COMBINE artifact, no s_linker14.py exists. Single sweep satisfies both requirements.

### Execution Strategy

- **D-03:** **Hybrid execution: BBB probe → reasonableness gate → full 5-dataset sweep.**
  - Step 1: Run `s_linker13` on `gpt-5.4` against **BBB only** (hardest dataset, biggest variance band).
  - Step 2: Reasonableness gate — if BBB F1 ≥ 0.6 (i.e., pipeline ran end-to-end without catastrophic failure that signals a harness bug), proceed to Step 3. Below 0.6 → halt, debug harness, escalate to user.
  - Step 3: Full 5-dataset sweep on `gpt-5.4`. Reuse BBB result from Step 1 (no re-run needed unless variance retest warranted).
  - Rationale: BBB is the hardest dataset, ~4pp jitter band on Claude Sonnet, likely largest jitter on GPT. If GPT can clear a minimum-sanity floor on BBB, the rest of the sweep is worth the API cost.

- **D-04:** **No backend-specific prompt tailoring.** CROSS-01 success criterion explicitly bans this. Only the existing model-adapter shim (in `src/llm_sad_sam/llm_client.py`) handles backend differences. If a prompt-level tailoring temptation arises, it MUST be rejected per the standing v2.0 policy ("any Claude-vs-GPT gap is a model-provider-property finding, not a failure to fix").

### Report Framing (CROSS-03)

- **D-05:** **Comparison report (markdown) is a model-provider-property report**, not a regression-investigation. Required content:
  - Per-dataset Claude Sonnet vs gpt-5.4 F1 (5 rows: MS, TS, TM, BBB, JAB)
  - Macro F1 delta with explicit statement on whether GATE-01 (macro ≥ 0.93) holds cross-model
  - Variance disclosure (LLM run-to-run jitter, especially on BBB)
  - Framing: any gap is a property of the model provider, not of `s_linker13`. No fix-it action items. Acceptable conclusions include "macro ≥ 0.93 holds cross-model" OR "macro < 0.93 — model-provider-property finding" — both satisfy CROSS-03.
- **D-06:** **Memory context** documents the project's prior cross-model evidence on different artifacts:
  - V32 on gpt-5.2 = 90.6% macro (-3.9pp vs Claude 94.5%)
  - V32 on gpt-5.4 = 87.7% (worse than 5.2)
  - GPT has massive run-to-run variance (5-12 link stdev, not fixable by temperature/seed)
  - These are prior findings on a DIFFERENT artifact (V32 ≠ s_linker13). Report should cite as background context, not as ground truth for s_linker13. Phase 9 produces fresh evidence specifically for s_linker13 on gpt-5.4.

### GATE-06 (Generality on Cross-Model Run)

- **D-07:** **Harness + adapter shim audit.** Per CROSS-01/02/03 success criterion 4 (standing GATE-06): the cross-model run is the strongest empirical evidence for the generality claim and must itself be clean. Audit scope:
  - `src/llm_sad_sam/llm_client.py` model-adapter shim — no benchmark-derived branching
  - Any new harness wrapper or runner script — no per-project special cases
  - Mechanical scan: grep harness code for BENCHMARK_TABOO terms; reviewer-defensibility check
- **D-08:** **No new prompts.** If the audit surfaces a temptation to add a "GPT-only" prompt variant, that's a generality violation per v2.0 thesis. Reject.

### Cost & Cancellation

- **D-09:** **API budget:** user has stated "no LLM budget limit" (per memory). Run full sweep without cost caps as long as harness is healthy. If sweep crashes mid-way (provider rate-limit, transient failures), checkpoint-resume per s_linker13's existing per-variant cache namespacing.
- **D-10:** **Cancellation rule:** if BBB probe (D-03 Step 1) returns macro < 0.6 OR persistent harness errors → halt sweep, surface checkpoint to user with options: debug harness / switch model / abandon cross-model / accept partial-evidence report.

### Claude's Discretion

- **Exact harness invocation pattern** — likely a shell wrapper around existing `run_ablation.py` with `OPENAI_MODEL_NAME=gpt-5.4` env override. Planner / executor decides.
- **JSON output paths** — follow project convention (`results/ablation_results/ablation_{timestamp}.json`).
- **Report file location** — `09-CROSS-REPORT.md` in the phase directory.
- **Variance retest policy** — whether to run BBB twice for the variance band. Default: single run per dataset (cost discipline); planner may override.

</decisions>

<canonical_refs>
## Canonical References

### v2.0 Milestone & Phase Definition
- `.planning/ROADMAP.md` — Phase 9 success criteria (CROSS-01/02/03), standing GATE-01/05/06/07
- `.planning/REQUIREMENTS.md` — CROSS-01 (s_linker13 on cross-model, 5 datasets), CROSS-02 (s_linker14 on cross-model — collapsed per Phase 8), CROSS-03 (comparison report)
- `.planning/PROJECT.md` — generality constraint, cross-model policy

### Phase 6 & 8 Outcomes (informs scope)
- `.planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-SUMMARY.md` — EXT-01 closed empty
- `.planning/phases/08-combine-s-linker14-stack-or-unify-combined-llm-primitives/08-SUMMARY.md` — Phase 8 closed no-op, CROSS-02 collapses to CROSS-01

### Generality Audit Input
- `BENCHMARK_TABOO.md` — taboo list + tailored-code anti-patterns (added Phase 6)

### Artifact Under Evaluation
- `src/llm_sad_sam/linkers/experimental/s_linker13.py` — the COMBINE artifact under cross-model evaluation
- `src/llm_sad_sam/llm_client.py` — model-adapter shim (GATE-06 audit target)
- `run_ablation.py` — invocation entry point; `OPENAI_MODEL_NAME` env override at line 436

### Claude Baseline Reference
- `results/ablation_results/ablation_20260529_215932.json` — s_linker13 on Claude Sonnet, 5-dataset reference (macro 0.9506)
- `.planning/milestones/v1.0-phases/05-promote-and-ablation-artifact/ABLATION-TABLE.md` — per-dataset Claude Sonnet baselines (MS 0.984, TS 1.000, TM 0.947, BBB 0.821, JAB 1.000)

### Prior Cross-Model Evidence (background context only)
- Project memory `MEMORY.md` §"GPT-5.2 Compatibility" — V32 on gpt-5.2 = 90.6%, V32 on gpt-5.4 = 87.7%, ~5-12 link stdev variance. NOT a ground truth for s_linker13.

</canonical_refs>

<code_context>
## Existing Code Insights

### Cross-Model Harness Surface
- `run_ablation.py:436` — `os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.2")` (default override site)
- `src/llm_sad_sam/llm_client.py:81-202` — backend dispatch, default model fallback chain
- `s_linker13.py` and all `s_linker13*.py` siblings — set `OPENAI_MODEL_NAME` default at module import time

### Established Patterns
- **Per-variant `_VARIANT_NAME` + `_checkpoint_dir` cache namespacing** — s_linker13's cache is namespaced. A gpt-5.4 run should namespace its cache by model OR by a fresh `_VARIANT_NAME` suffix to avoid cross-contamination with Claude Sonnet's cached state.
- **Approve-biased fallback** on LLM failure throughout pipeline — already accommodates flaky backends.
- **JSON output convention** — `results/ablation_results/ablation_{timestamp}.json` schema. Per-dataset F1 + macro F1 + per-link TP/FP/FN.
- **Checkpoint resume** — s_linker13 caches per phase; if sweep crashes, re-invocation resumes from last completed phase.

### Integration Points
- **`OPENAI_MODEL_NAME=gpt-5.4`** env override is the only switch needed at the harness level (per the no-prompt-tailoring rule). Set via shell, invocation script, or `run_ablation.py` flag.
- **Backend dispatch** in `llm_client.py:192` (`_infer_backend_from_model`) — OpenAI route already in place. No code changes anticipated.
- **Per-dataset invocation pattern** in `run_ablation.py` — likely supports `--datasets bigbluebutton` flag for the BBB probe step (D-03 Step 1).

</code_context>

<specifics>
## Specific Ideas

- BBB probe (D-03 Step 1) is **specifically the right hardest-first checkpoint** because (a) project memory shows GPT has massive BBB variance (5-12 link stdev), (b) BBB is the dataset where Claude-vs-GPT gaps are widest on prior artifacts, (c) if BBB F1 < 0.6 the harness is broken (not a model quality issue).
- D-04's "no backend-specific prompt tailoring" is the **hard differentiator** between this phase and a tuning phase. Any prompt modification = scope violation. The harness + shim is the only allowed layer.
- D-05's "model-provider-property framing" is **v2.0's published thesis** — the cross-model run validates or rejects the generality claim. Report MUST NOT include "fix-it" action items even if numbers regress; that would re-open the question rather than close it.

</specifics>

<deferred>
## Deferred Ideas

- **Anthropic Haiku 4.5 cross-tier validation** — would test cross-TIER (Sonnet → Haiku, same provider) rather than cross-PROVIDER. Out of v2.0 scope.
- **Variance-band tightening for GPT** — memory notes GPT has 5-12 link stdev; reducing this is EXT-04 territory (deferred to v2.1+).
- **Reduplication or retry-with-temperature strategies** — would constitute prompt-level cross-model tailoring; banned per D-04.
- **gpt-5.2 run on s_linker13** — not the user's choice. If gpt-5.4 results raise questions later, gpt-5.2 may be revisited in v2.1+.

</deferred>

---

*Phase: 09-cross-gpt-5-2-cross-model-validation* (directory slug retains "5-2" from milestone-kickoff; actual cross-model target = gpt-5.4 per D-01)
*Context gathered: 2026-05-31*
