# llm-sad-sam-v45: Fully-LLM-Driven s_linker

## What This Is

Empirical evolution of `s_linker12c` into a fully LLM-driven SAD-SAM traceability linker. Each milestone swaps one (or a small group of) structural rule/heuristic for an LLM-based replacement using the cite-evidence pattern validated in Spike 001, producing a ranked ablation of which rules can be retired without regressing macro F1.

## Core Value

Every rule removed from `s_linker12c` and replaced by an LLM primitive must either hold the pipeline at macro F1 ≥ 93% or be rejected — the deliverable is a defensible claim that traceability linking can be done without hand-crafted structural rules.

## Requirements

### Validated

- ✓ Reproducible `s_linker12c` baseline (per-dataset F1 + macro F1 + FP/FN table) — v1.0 (INFRA-01)
- ✓ Spike 001 LLM trailing-word enrichment integrated; `_split_component_name` retired (`s_linker13a`) — v1.0 (VAR-01)
- ✓ `_is_structurally_unambiguous` post-filter removed; LLM ambiguity classification trusted end-to-end (`s_linker13b`) — v1.0 (VAR-02)
- ✓ `_is_ambiguous_name_component` wrapper inlined and removed (`s_linker13c`) — v1.0 (VAR-03)
- ✓ Alias-discovery prompt extended with `scope: global|local`; `_is_strong_alias` + `_get_strong_alias_mappings` retired (`s_linker13e`) — v1.0 (VAR-05)
- ✓ Strong-alias-mention signal folded into coref prompt; `_has_strong_alias_mention` retired (`s_linker13f`) — v1.0 (VAR-06)
- ✓ `_has_standalone_mention` KEEP decision logged (RISKY per Spike 002; replacement deferred to v2 EXT-01) — v1.0 (PROMO-02)
- ✓ Ablation table generated (markdown + LaTeX, 8 rows) — v1.0 (PROMO-03)
- ✓ Winning variant promoted as `s_linker13.py` with zero non-trivial rules (macro F1 0.9509, +1.04 pp vs 12c) — v1.0 (PROMO-01)
- ⚠ Spike 003 LLM mention classifier integration attempted — REJECTED (VAR-04 retired). LLM cannot reproduce dotted-path Java-package convention; 33 entity-source FPs on TeaMMates → −18.8 pp regression. Documented as publishable negative result in METHODOLOGY.md §4 — v1.0

### Active

v2.0 scope (Complete Rule Removal + Cross-Model — Generality First) — **shipped 2026-05-31, see [milestone archive](milestones/v2.0-ROADMAP.md):**
- ⚠ **EXT-01** — Replace `_has_standalone_mention` — CLOSED EMPTY (negative). 2 design generations + 3-direction feasibility probe converged on "BBB recall gap is upstream of the gate". Published as thesis-boundary finding.
- — **EXT-02** — Drop dotted-path guard — AUTO-SKIPPED per gating (EXT-01 did not pass dual floor).
- ✓ **COMBINE** — Retro-satisfied: research found the 3 in-scope rule-removal primitives (Spike-001 trailing-words + scope-field + alias-coref-fold) were already unified inside `_learn_document_knowledge_enriched` during the v1.0 chain. s_linker13 retro-designated as the COMBINE artifact. No s_linker14.py built.
- ✓ **CROSS** — gpt-5.4 5-dataset sweep: macro F1 0.9077 (Δ -4.3pp vs Claude 0.9506). GATE-01 cross-model does NOT hold; TM dominates the gap via dotted-path/generic-English/GAE-platform conflation. Framed as model-provider-property finding per v2.0 thesis.

(Deferred to later milestone: EXT-04 emit-biased boundary prompting — variance work, not rule removal.)

## Current Milestone: v2.1 Cleanup + Prompt Simplification

**Goal:** Slim `s_linker13` + its dependency surface and trim prompt-rule scaffolding to the minimum that holds GATE-01 on Claude Sonnet AND a cross-model gate on gpt-5.4 — without breaking any currently-runnable variant.

**Target features:**
- Standalone cleaned `s_linker13` with its own (possibly duplicated) helper copies; new dependencies on `prompts_v3` allowed
- `prompts_v3.py` side-by-side with `prompts_v2.py`; only prompts actually used by the new `s_linker13` carried forward
- Per-prompt rule trimming as ablation variants, each gated by GATE-01 (Claude) + cross-model floor (gpt-5.4 macro ≥ 0.9077 within tolerance)
- Dead-code sweep across `s_linker13`'s actual dependency tree (`data_types_v2`, `document_loader_v2`, `pcm_parser_v2`, used `ilinker*`) — unreferenced helpers/imports/constants removed
- Frozen-compat guarantee: every variant in `CANONICAL_VARIANTS` / `run_ablation.py` continues to produce identical F1 by importing `prompts_v2` (untouched)

**Standing constraints carried forward:** GATE-01, GATE-06 (generality / zero benchmark-derived values), GATE-07 (canonical registration), Claude Sonnet default, BENCHMARK_TABOO compliance.

## Past State

**Shipped:** v2.0 — Complete Rule Removal + Cross-Model (2026-05-31). All 8 active requirements traced to closing artifacts. All 4 standing gates held. Audit verdict: PASSED (mixed-result). Production artifact remains **`s_linker13.py`** (macro F1 0.9506 Claude Sonnet, 0.9077 gpt-5.4).

**Key v2.0 findings:**
1. The "rule replaced by LLM primitive" thesis has a clean boundary: rules with project-specific surface conventions (dotted-path, casing) cannot be replaced without project-specific calibration. Same failure class hit v1.0 13d and v2.0 EXT-01.
2. Knowledge injection (alias context) yields measurable but bounded lift (+0.7-2.1pp on BBB) — pattern worth preserving for future LLM judge layers.
3. Probe-first methodology validated: cheap feasibility study cut Phase 6 short before a 4th sub-variant cycle.

## Past Milestones

- **v1.0** (2026-05-29) — Rule-to-LLM Ablation (`s_linker12c` → `s_linker13`). 6 rules removed, 1 rejected (VAR-04 dotted-path). Final macro 0.9509. See `milestones/v1.0-ROADMAP.md`.
- **v2.0** (2026-05-31) — Complete Rule Removal + Cross-Model. EXT-01 closed empty, CROSS evidence published. See `milestones/v2.0-ROADMAP.md`.

## Next Milestone Candidates

Active milestone is v2.1 (above). Topics retained for later milestones (v2.2+):
- **EXT-04** — Emit-biased boundary prompting on alias-discovery (BBB variance band tightening 3pp → 1pp). Variance work, not rule removal.
- **Upstream-tier rule removal** — v2.0 EXT-01 evidence suggests the BBB recall gap lives in the extraction/coref tier. A future milestone could target a rule there instead.
- **Multi-model adapter exploration** — v2.0 CROSS evidence isolated dataset-shape-dependent model gaps. Investigate whether a project-agnostic backend-adaptive harness layer is reviewer-defensible (would need fresh GATE-06 thinking).

<details>
<summary>Past milestone scope (archived v2.0 active section)</summary>

## v2.0 Active (archived — milestone shipped 2026-05-31)

**Original goal:** Finish the no-hand-crafted-rules thesis by replacing the last structural rule with a project-agnostic LLM primitive, validate on GPT-5.2, and explore stacking/unifying LLM primitives into `s_linker14`.

**Actual outcome:** EXT-01 closed empty (negative); EXT-02 auto-skipped; COMBINE retro-satisfied via existing v1.0 unification; CROSS done on gpt-5.4 with mixed-result published as model-provider-property finding. See `milestones/v2.0-ROADMAP.md` and `milestones/v2.0-MILESTONE-AUDIT.md`.

**Hard generality constraint (GATE-06):** Held throughout v2.0. Zero benchmark-derived values shipped.

</details>

### Out of Scope (general — applies across milestones)

- New seed/linker approaches (ILinker3+, cross-model ensembles) — this project is rule-reduction on 12c, not exploration
- Non-SAD-SAM tasks (SAM-Code, SAD-Code) — out of dataset scope
- Cost optimization — user has set "no LLM budget limit"; rule-replaceability is the only constraint
- Changes to retained upstream components (`ilinker*`, `prompts_v2`, `data_types_v2`) unless required by a rule removal
- Bench leakage: no benchmark-derived words may enter prompt examples (enforced via BENCHMARK_TABOO.md)

## Context

- **Codebase**: retained `s_linker` family through `s_linker12e` plus `ilinker1-3`. Runner is `run_ablation.py`. Default model: Claude Sonnet.
- **Baseline memory**: `s_linker12c` (ICSE clean) reports ~94% macro F1; `S-Linker10` memory entry shows 95.9% macro F1 with prompts_v2 and LLM word-usage enrichment.
- **Validated spikes** (re-validate once integrated):
  - 001 `llm-trailing-words` — single LLM call with evidence guardrail replaces structural gate + LLM verify for trailing-word alias enrichment.
  - 002 `rules-audit` — classified all 12 `s_linker12c` helpers: 9 REPLACEABLE, 1 RISKY (`_has_standalone_mention`), 4 ESSENTIAL (parsers/formatters). Ranked removal order defined.
  - 003 `llm-mention-classifier` — LLM enum emission matches regex `_classify_mention` byte-identically, piggybacked on existing entity-extraction prompt.
- **GPT/Claude gap known**: Claude Sonnet is the target backend; GPT-5.2 compatibility is a side concern, not a gate (per prior memory: inherent model capability gap, not fixable).
- **Dataset strategy**: hard-tier-first (teammates, bigbluebutton — most rule-sensitive) during development, full 5-project macro F1 for every promoted variant.

## Constraints

- **Quality**: Every promoted variant must hold macro F1 ≥ 93% on the 5-project benchmark. Variants below the floor are reported but not promoted.
- **LLM budget**: No upper bound on calls; replaceability trumps cost.
- **Model**: Claude Sonnet only (per user preference; do not switch to Opus or GPT).
- **Data leakage**: Zero benchmark-derived words in prompts (cascade/throttle rule, per `BENCHMARK_TABOO.md`). Spike findings must be re-audited before integration.
- **Codebase hygiene**: Each rule-removal lands as its own standalone linker variant (e.g., `s_linker13a.py`) — user prefers duplicated standalone files over inheritance chains.
- **Naming**: Promoted successor is `s_linker13` (12e is the last current sibling).

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Base = `s_linker12c` (not 12e) | Spikes 001/002/003 target 12c; 12d/12e are enrichment side-experiments | ✓ Good — chain landed on 12c; 13f beat 12c by +1.04 pp macro F1 |
| Success = zero non-trivial rules | User explicitly picked "zero non-trivial rules" over F1-parity goal | ✓ Good — `s_linker13` retains only `_has_standalone_mention` + parsers/formatters |
| Re-validate spikes in-pipeline | Spike validation was isolated; pipeline integration can surface new failure modes | ✓ Good — Spike 003 in-pipeline integration surfaced TM regression that isolated validation missed (VAR-04 retired) |
| F1 floor = 93% macro | Baseline is ~94%; allow ≤1pp regression per milestone | ✓ Good — floor held across all 6 successful removals; final macro 0.9509 (+1.04 pp) |
| Ablation unit = linker variant, not individual rule | User wants "F1 contribution per linker", not per-rule | ✓ Good — 7 standalone variant files enabled clean per-step ΔF1 attribution |
| Dataset schedule = hard-tier-first, then all 5 | Teammates/BBB are most rule-sensitive; cheap signal before full sweep | ⚠ Revisit — VAR-06 (13f) was hard-tier marginal but full-sweep best-in-chain; standing policy retained "full sweep is decisive" |
| Keep `_has_standalone_mention` tentatively | Spike 002 classified it RISKY (O(N·M) anchor collection); decide after other removals land | ✓ Good — formalized as KEEP in Phase 5; EXT-01 spike deferred to v2 |
| KEEP `_has_standalone_mention` in `s_linker13` | Spike 002 classified it RISKY (O(N×M) anchor-collection; replacing it with an LLM call would require a full-component-list × full-sentence-list scan). Phase 5 confirms KEEP — replacement deferred to v2 (EXT-01 spike) under a relaxed budget. EXT-02 (drop dotted-path guard) is a narrower follow-up also deferred to v2. See `.planning/spikes/002-rules-audit/` for the full classification. | KEPT (Phase 5, 2026-05-29) |
| GATE-06 generality audit (v2.0) | User flagged at v2.0 kickoff: every new prompt + helper must read as sound/clean/general to any project; no tailored rules in prompt OR logic. Reviewer-defensibility, not just BENCHMARK_TABOO scan. | Standing policy from v2.0 onward (2026-05-30) |
| LLM-COMBINE stack-vs-unify decision deferred | EXT-01 cost/quality signal will choose between (1) stacked separate LLM primitives in `s_linker14` and (3) unified single-prompt variant. Premature lock would bias the comparison. | Decide after EXT-01 lands |
| GATE-01 cross-model tolerance T = 1.0pp (v2.1) | Pins the loose REQUIREMENTS GATE-01 phrasing "≤ 1pp regression" to a concrete numeric tolerance so Phase 12 trim acceptance and Phase 13 promotion sweeps can be evaluated deterministically. Baseline 0.9077 is the v2.0 CROSS evidence on gpt-5.4 (see v2.0-MILESTONE-AUDIT.md "09-CROSS-REPORT.md §GATE-01"). T = 1.0pp means a variant passes iff gpt-5.4 macro F1 ≥ 0.9077 − 0.01 = 0.8977 absolute on the full 5-dataset sweep. | Codified 2026-05-31 (Phase 10, Plan 10-04) |
| GATE-01 relaxation (v2.1 Phase 12) | The original GATE-01 Claude floor (macro ≥ 0.93 + BBB drop ≤ 6pp) is the s_linker13 v2.0 promotion bar. It is too tight to test aggressive "super-simple-prompt" trim mechanisms — any trim that loses 1-2pp on BBB is rejected, including trims whose simplification value justifies the modest regression. v2.1 Phase 12 explicitly relaxes the Claude floor to macro F1 ≥ 0.90 and BBB absolute F1 ≥ 0.79 (the swattr SwattrEvaluationProject SAD-SAM expected value from `tlr/tests-tlr/.../approach/SwattrEvaluationProject.java`). Other-dataset drop tolerance (-2pp) unchanged. Cross-model gpt-5.4 floor 0.8977 unchanged (the cross-model gate is the v2.1 thesis claim and cannot be relaxed). Rationale: align the v2.1 cleanup acceptance with the project's own externally-validated integration-test bar, not the v2.0 promotion peak. Trim rejections under the relaxed gate are stronger evidence of mechanism failure than rejections under the v2.0 peak. | Codified 2026-05-31 (Phase 12, user directive mid-Wave-2) |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-31 — v2.1 kickoff (Cleanup + Prompt Simplification)*
