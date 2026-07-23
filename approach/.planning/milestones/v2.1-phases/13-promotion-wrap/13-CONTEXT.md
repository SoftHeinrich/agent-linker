# Phase 13: Promotion & Wrap — Context

**Gathered:** 2026-06-01
**Status:** Ready for planning
**Mode:** Smart discuss compressed — Phase 12 outcomes define the entire scope.

<domain>
## Phase Boundary

Either ship `s_linker13_min.py` composing the 3 Phase 12 accepted simplifications (Step 0 dead-code drop + trim1 distilled judge + trim9 runtime seed disambig) after a full 5-dataset confirmatory sweep on BOTH Claude Sonnet AND gpt-5.4 passes the v2.1 gates, OR publish a documented negative result. Either way, regenerate `ABLATION-TABLE.md` with v2.1 rows + the `.tex` artifact.

</domain>

<decisions>
## Implementation Decisions

### What `s_linker13_min` composes
- **Step 0** (dead-code drop in `prompts_v3.py`): 7 unused constants gone (Phase 12 Plan 12-01).
- **trim1** (distilled `DOC_KNOWLEDGE_JUDGE_RULES` via Technique 3 + Technique 8): Phase 12 Plan 12-03 accepted, Claude 0.9553 / gpt-5.4 0.9173.
- **trim9** (runtime-generated `SEED_DISAMBIGUATION_RULES`): Phase 12 Plan 12-12 accepted, Claude 0.9474 / gpt-5.4 0.9007.

These hit disjoint pipeline phases (trim1 = Tier 1 alias judge; trim9 = Tier 2 seed disambiguation). Composition is expected safe.

### Confirmatory sweep — FULL pipeline this time, not single-step
- The single-step ablation harness only modifies ONE phase at a time. For promotion, we need a full-pipeline 5-dataset sweep on the composed variant (uses BOTH trim1 in layer1 AND trim9 in layer2).
- 5 datasets × 2 backends (Claude Sonnet, gpt-5.4) = **10 full-pipeline runs**.
- Cost estimate: ~$30-50 total (Claude burns more on heavy datasets; gpt-5.4 cheaper).
- Wallclock: ~2-4h.

### Gates that must hold for promotion
- **Claude Sonnet**: macro F1 ≥ 0.93 AND BBB drop ≤ 6pp vs baseline AND other-dataset drop ≤ 2pp (the ORIGINAL gate — not Scenario E. The min variant is for promotion, not exploration. It must pass the strict bar.)
- **gpt-5.4 cross-model**: macro F1 ≥ 0.8977 (T = 1.0pp off baseline 0.9077).
- **GATE-06**: already verified by Plan 12-06 audit (both trim1 + trim9 PASS).
- **GATE-07**: registered in `CANONICAL_VARIANTS` + `VARIANT_SPECS`; standalone file; structured docstring.
- **GATE-02**: tests/test_v20_baseline_regression.py still passes (frozen compat unchanged).

### What "negative result" looks like
If composed `s_linker13_min` fails either gate:
- Document which gate failed on which dataset.
- Compare to trim1-alone and trim9-alone performance — does composition regress vs the individual trims?
- Publish as "compositional fragility" finding — even with both components accepted individually, composition broke.
- v2.1 still ships dead-code drop (Step 0) + the individually-canonical trim1 and trim9 as separate variants, but no `_min` canonical promotion.

### File layout for `s_linker13_min`
Standalone `src/llm_sad_sam/linkers/experimental/s_linker13_min.py`:
- Class `SLinker13Min`, `__bases__ == (object,)` (or subclasses `SLinker13Clean` per trim3 template — verify what the trim1/trim9 variants did and match).
- Imports `prompts_v3` (Step 0 + trim1's distilled rubric replacement) + `helper_v3`.
- Implements trim9's runtime seed-disambiguation mechanism inline (the rubric-builder call).
- Structured GATE-07 docstring documenting v2.0 → v2.1 evolution.
- Registered in `CANONICAL_VARIANTS` + `VARIANT_SPECS` with `canonical=True` (the promotion bit) OR `canonical=False` if we're being cautious until ABLATION-TABLE evidence is on record.

### ABLATION-TABLE.md v2.1 rows (GATE-03)
Must add rows for:
- `s_linker13_clean` (Phase 10 baseline, Claude + gpt-5.4 5-dataset macros)
- `s_linker13_trim1_judge_clean` (Phase 12 Plan 12-03 accepted)
- `s_linker13_trim9_seed_runtime_clean` (Phase 12 Plan 12-12 accepted)
- `s_linker13_min` (this phase's composition, if promoted)
- Optional frontier-only rows: trim4/5/6/7/8 with Scenario-E annotation

`.tex` artifact regenerated from the markdown table.

</decisions>

<code_context>
## Existing Code Insights

- Phase 12 final mapping: `.planning/phases/12-trim-ablation/12-06-V2_TO_V3_MAPPING-FINAL.md` (16 rows, every v2 constant dispositioned).
- Phase 12 audit: `12-06-AUDIT-REPORT.md` (GATE-06 PASS on all retained code).
- Phase 12 frontier: `12-FRONTIER-MAP-SUMMARY.md` (9-variant scoreboard, Scenario E framing).
- Existing `ABLATION-TABLE.md` from v1.0 + v2.0 — DO NOT touch existing rows; v2.1 is additive.
- run_ablation.py registration shape: `CANONICAL_VARIANTS` list + `VARIANT_SPECS` dict.
- Single-step harness in `src/llm_sad_sam/ablation/single_step.py` — NOT what we need for the promotion sweep; we need the regular `run_ablation.py --variants s_linker13_min --datasets all` full-pipeline path.

</code_context>

<specifics>
## Specific Ideas

- Confirmation sweep: dispatch via existing `run_ablation.py` CLI, NOT single-step harness. This is the full pipeline because composition spans multiple phases.
- If Claude budget remains constrained: run gpt-5.4 sweep first (user's stated priority); if that passes, run Claude.
- If gpt-5.4 macro is borderline (~0.90), consider whether to relax to Scenario E for promotion — but the default is strict promotion gate.
- The Voyager pilot result (whenever it lands) gets a section in the milestone summary but does NOT block Phase 13 close.

</specifics>

<deferred>
## Deferred Ideas

- Voyager-TLR train-test methodology → v2.2 anchor (whether the gpt-5.4 pilot succeeds or fails, the methodology becomes the next milestone's first plan).
- Link provenance data structure (`12-PROVENANCE-DEFERRAL-NOTE.md`) → v2.2 candidate.
- Extended-thinking variants (survey Top 3 #1) → v2.2 candidate.
- Self-Refine layered on accepted variants (survey Top 3 #2) → v2.2 candidate.
- Per-model adaptive prompts (ADAPTER-01 re-opening) → v2.2 candidate.

</deferred>
