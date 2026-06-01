# Phase 12: Trim Ablation — Context

**Gathered:** 2026-05-31
**Status:** Ready for planning
**Mode:** Smart discuss compressed — Phase 11 survey + user-supplied execution method (checkpoint loading) define the operating contract.

<domain>
## Phase Boundary

Per-prompt rule trimming, executed as single-step ablations against pre-saved checkpoints (NOT full-pipeline sweeps). Produce `prompts_v3.py` carrying every accepted trim, plus per-trim ablation variants in `s_linker13_clean`'s sibling family. Each accepted trim passes BOTH GATE-01 Claude (macro F1 ≥ 0.93, BBB tolerance 6pp, other ≤ 2pp) AND GATE-01 cross-model (gpt-5.4 macro F1 ≥ 0.8977 / within 1.0pp of 0.9077) AND GATE-06 (no benchmark leakage). Rejected trims documented in milestone summary with failing dataset(s).

</domain>

<decisions>
## Implementation Decisions

### Execution Method — Checkpoint-Loaded Single-Step Ablation (USER DIRECTIVE)
Do NOT run the full 5-phase pipeline for every trim variant. Use the existing checkpoint infrastructure (`results/phase_cache/<variant>/<dataset>/{layer1,layer2,entity_candidates,entity_decisions,final}.pkl`) to re-run only the phase whose prompt changed. Confirmed extant for `s_linker13_clean` on all 5 datasets (mediastore, teastore, teammates, bigbluebutton, jabref) after Phase 10 sweep.

Per memory, this matches V30d's existing `resume_from_phase` / `run_single_phase()` ablation tooling. Phase 12 must extend (or reuse-without-modification) the same pattern.

For each trim:
1. Modify the affected prompt in a new `prompts_v3.py` (or a per-trim helper).
2. Create the ablation variant as a standalone sibling under `s_linker13_<trim_id>_clean.py` per existing convention (or use the `--variants` toggle if `run_ablation.py` already supports prompt overrides via env / kwarg — investigate during planning).
3. Load checkpoints from the previous phase, re-run only the modified phase end-to-end (which feeds forward into subsequent phases without re-calling them).
4. Compare F1 delta vs the baseline checkpoint's `final.pkl`.

For phases downstream of the modified one, the executor MUST either (a) re-run them too if the modified phase's output changes their input, OR (b) use the existing checkpoint if the phase is independent. Plan-phase must enumerate this dependency per trim.

### Trim Steps (from Phase 11 survey §5 + supplement §4, ordered)
Take all 4 steps. User confirmed scope via "single step ablation" directive (not the "Step 0 only" minimal option).

- **Step 0 — Free win, no LLM cost.** Create `prompts_v3.py` containing only the 9 prompts actively imported by `s_linker13_clean`. Drop the 7 dead constants (`WORD_USAGE_PROMPT`, 6 `STANDALONE_MENTION_*` EXT-01 variants). Net deletion: ~150 LOC / ~36 rules. No ablation needed — purely a registration check.
- **Step 1 — Restructure `DOC_KNOWLEDGE_JUDGE_RULES` + `DOC_KNOWLEDGE_JUDGE_EXAMPLES`.** Apply Technique 3 (lossless rubric distillation) + Technique 8 (directive ordering, arXiv 2603.13351: reasoning-before-conclusion). Single-step re-run of Tier 1 alias-judge stage. Variant: `s_linker13_trim1_judge_clean`.
- **Step 2 — Merge `ENTITY_EXTRACTION_RULES` + `VALIDATION_RULES`.** Apply Technique 3 — collapse overlapping architectural-participant rubric across both prompts. Estimated 4-rule reduction. Re-run Tier 2 entity stages. Variant: `s_linker13_trim2_entval_clean`.
- **Step 3 — Inference-time rubric (supplement Techniques 2+3 from AHE / Agentic Rubrics).** Apply to alias-judge stage. Rubric regenerated at inference rather than statically pruned. Higher risk because the pattern is new to this codebase. Variant: `s_linker13_trim3_runtime_rubric_clean`.

### Lineage Clarification — s_linker13 prompts are unablated territory
The Phase 11 survey + earlier drafts of this CONTEXT cited "V35 ceiling" evidence as if it set a prior on s_linker13. **It does not.** V30/V35 are V-series / S-Linker family variants (V26a → V30/V31/V32 / S1-S11) — a different pipeline lineage that ablated V-series prompts (CONVENTION_GUIDE, P8c filter, judge advocate/prosecutor patterns), NOT s_linker13's prompts (`DOC_KNOWLEDGE_JUDGE_RULES`, `ENTITY_EXTRACTION_RULES`, `VALIDATION_RULES`, `SEED_DISAMBIGUATION_RULES`, etc.). What IS transferable:
- **Claude Sonnet model-level lessons** (concrete output examples bias sentence-number distribution; static deletion that drops info density tends to regress). These are model behaviors, not pipeline-specific.

What is NOT transferable:
- Any pre-judgment that trimming s_linker13's specific prompts will regress. **Trims must be measured, not pre-rejected.** The V35 conservative prior is dropped from Phase 12 design — design and ablate aggressively, let single-step measurements decide.

### Acceptance Per Trim (the gate)
Each trim variant must:
- Pass GATE-01 Claude Sonnet: macro F1 ≥ 0.93 AND BBB drop ≤ 6pp AND other-dataset drop ≤ 2pp.
- Pass GATE-01 cross-model gpt-5.4: macro F1 ≥ 0.8977 (Phase 10 codified floor) on all 5 datasets.
- Pass GATE-06: BENCHMARK_TABOO grep on the new prompt body returns empty.
- Pass reviewer-defensibility: trim rationale documented in the variant's docstring; rule that was removed is justified as "covered by another rule" / "model handles natively" / "dead — never fired in V32 audit" with evidence.

Trims that fail any gate are documented in the milestone summary's "rejected trims" table and NOT merged into the final `s_linker13_min`.

### Variant Naming + Registration
Each trim variant lands as a standalone `.py` file under `src/llm_sad_sam/linkers/experimental/`. Registered in `CANONICAL_VARIANTS` + `VARIANT_SPECS` with `canonical=False`. Plan 13-01 (Phase 13) will promote the union of accepted trims into `s_linker13_min.py` as the candidate canonical.

### Out of Scope
- Full 5-dataset, full-pipeline sweeps per variant (user-rejected, too expensive).
- Modifying `prompts_v2.py` or frozen variants (forbidden — Phase 10 invariant).
- Pursuing a trim that requires extended-thinking enablement without first establishing the baseline non-thinking number (deferred to a measurement sub-task, not a primary trim).

</decisions>

<code_context>
## Existing Code Insights

### Checkpoint Infrastructure (verified existent)
- `results/phase_cache/s_linker13_clean/{mediastore,teastore,teammates,bigbluebutton,jabref}/` each contains `layer1.pkl`, `layer2.pkl`, `entity_candidates.pkl`, `entity_decisions.pkl`, `final.pkl`.
- Same for `s_linker13/`.
- `_save_phase()` in `s_linker13.py:1172-1177` writes pickles per phase under `PHASE_CACHE_DIR` (default `./results/phase_cache`).
- Memory: V30c "Saves pickle checkpoints after every phase for single-step ablation experiments"; V30d "resume_from_phase + run_single_phase() for targeted ablation"; `test_heuristics.py` is the offline single-phase ablation pattern.

### Phase-to-Prompt Map (from Phase 11 survey §0)
| Phase | Prompts that fire (per `prompts_v2.py`) | Trim affects |
|-------|------------------------------------------|--------------|
| layer1 (Tier 1 knowledge) | `AMBIGUITY_FEW_SHOT`, `AMBIGUITY_RULES`, `DOC_KNOWLEDGE_EXTRACTION_RULES`, `DOC_KNOWLEDGE_JUDGE_EXAMPLES`, `DOC_KNOWLEDGE_JUDGE_RULES` | Step 1, Step 3 |
| layer2 (Tier 2 recovery) | `ENTITY_EXTRACTION_RULES`, `VALIDATION_RULES`, `COREF_RULES`, `SEED_DISAMBIGUATION_RULES` | Step 2 |
| Step 0 | (no prompts run) | — |

When a layer1 prompt changes, downstream layer2 / entity / final phases generally need re-running because the layer1 output (doc knowledge, aliases) feeds into layer2's seed extraction. When a layer2 prompt changes (Step 2), layer1 can be reused as-is.

### Existing Ablation Variants (registered)
~30 variants already in `VARIANT_SPECS`. Phase 12 trims add 3 more (`s_linker13_trim1_judge_clean`, `s_linker13_trim2_entval_clean`, `s_linker13_trim3_runtime_rubric_clean`). Each `canonical=False`.

### Cross-Model Sweep (gpt-5.4)
The cross-model checkpoint dir is `results/phase_cache_gpt54/`. Verify it has `s_linker13_clean` populated; if not, a one-time baseline cross-model sweep on `s_linker13_clean` is needed before single-step ablation can be applied. Investigate at plan-phase.

</code_context>

<specifics>
## Specific Ideas

- **Cheapest first**: Start with Step 0 (free win, no LLM) — verify `prompts_v3.py` registration + clean import in one plan. Use as smoke test for the rest of Phase 12.
- **Plan-phase must investigate** whether `run_ablation.py` supports a `--phase-only` / `--resume-from` flag, or if the test pattern from `test_heuristics.py` (memory) is the right entry point. If neither, build a minimal CLI: `python -m llm_sad_sam.ablation single_step --variant X --dataset D --phase P`.
- **gpt-5.4 baseline**: If `phase_cache_gpt54/s_linker13_clean/` doesn't exist, plan a ONE-TIME 5-dataset cross-model baseline sweep as Plan 12-00 — explicitly user-authorized at the time it's needed, not assumed.
- **GATE-06 strictness**: every trim's new prompt body must pass the BENCHMARK_TABOO grep before going to ablation. Catch leakage early.
- **Stochastic LLM tolerance**: Per Phase 10 SUMMARY's parity verdict, accept per-dataset F1 within Claude run-to-run variance band (the existing GATE-01 tolerances), not the (unrealistic) `< 1e-4` threshold.

</specifics>

<deferred>
## Deferred Ideas

- **Extended-thinking enablement** (main survey §6). Deferred unless Step 1's static-restructure trim alone leaves an obvious gap.
- **opencode / codex pattern adoption** (survey §3). Higher cost, more architectural — likely v2.2 territory.
- **Self-consistency K-sample voting** (Technique 4) on `s_linker13`. Memory notes existing intersect/union voting already covers this — empirical investigation deferred until Phase 12 results reveal whether judge variance is a remaining problem.

</deferred>
