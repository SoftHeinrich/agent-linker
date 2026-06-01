# llm-sad-sam-v45: Fully-LLM-Driven s_linker

## What This Is

Empirical evolution of `s_linker12c` into a fully LLM-driven SAD-SAM traceability linker. Each milestone swaps one (or a small group of) structural rule/heuristic for an LLM-based replacement using the cite-evidence pattern validated in Spike 001, producing a ranked ablation of which rules can be retired without regressing macro F1. v2.1 extended this to prompt-rule trimming (lossless distillation + runtime rubric mechanisms) under a cross-model gate.

## Core Value

Every rule removed from `s_linker12c` (and every prompt-rule trimmed from its successor's prompts) must either hold the pipeline at macro F1 ≥ 93% on Claude Sonnet AND macro within tolerance of the v2.0 cross-model baseline (0.9077) on gpt-5.4 — or be rejected. The deliverable is a defensible claim that traceability linking can be done without hand-crafted structural rules, validated across model providers, with prompt scaffolding minimized to what each backend can actually use.

## Requirements

### Validated

**v1.0 (Rule-to-LLM Ablation):**
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

**v2.0 (Complete Rule Removal + Cross-Model — Generality First):**
- ⚠ **EXT-01** — Replace `_has_standalone_mention` — CLOSED EMPTY (negative). 2 design generations + 3-direction feasibility probe converged on "BBB recall gap is upstream of the gate". Published as thesis-boundary finding.
- — **EXT-02** — Drop dotted-path guard — AUTO-SKIPPED per gating (EXT-01 did not pass dual floor).
- ✓ **COMBINE** — Retro-satisfied: research found the 3 in-scope rule-removal primitives (Spike-001 trailing-words + scope-field + alias-coref-fold) were already unified inside `_learn_document_knowledge_enriched` during the v1.0 chain. s_linker13 retro-designated as the COMBINE artifact. No s_linker14.py built.
- ✓ **CROSS** — gpt-5.4 5-dataset sweep: macro F1 0.9077 (Δ -4.3pp vs Claude 0.9506). GATE-01 cross-model does NOT hold; TM dominates the gap via dotted-path/generic-English/GAE-platform conflation. Framed as model-provider-property finding per v2.0 thesis.

**v2.1 (Cleanup + Prompt Simplification):**
- ✓ **CLEAN-01/02** — Standalone `s_linker13_clean.py` + `helper_v3.py` extracted; `s_linker13.py` and `prompts_v2.py` frozen byte-equal. Phase 10.
- ✓ **GATE-01/02** — Cross-model T=1.0pp codified (floor 0.8977 absolute, later Scenario-E loosened for runtime variants); frozen-compat regression test green (35 passed, 28 xfailed). Phase 10.
- ✓ **PROMPT-05** — Prompt-minimization survey shipped with 8 techniques scored; drove Phase 12 trim strategy. Phase 11.
- ✓ **PROMPT-01** — `prompts_v3.py` with 9 active constants + final v2→v3 mapping. Phase 12.
- ✓ **PROMPT-02** — 9 trim mechanisms ablated; 2 ACCEPT (trim1 Tier-1 judge distillation, trim9 Tier-2 seed runtime rubric); 7 REJECT documented as frontier map. Phase 12.
- ✓ **PROMPT-04** — Full GATE-06 + BENCHMARK_TABOO + reviewer-defensibility audit on all 12 retained files PASSES. Phase 12.
- ✓ **PROMPT-03** — `s_linker13_min` PROMOTED (composed canonical of trim1 + trim9). Claude macro 0.9506, gpt-5.4 macro 0.9069. Both gates clear with safety margin (+2.06pp / +0.92pp). Phase 13.
- ✓ **GATE-03** — ABLATION-TABLE.md + .tex v2.1 addendum (11 new rows); v1.0/v2.0 rows preserved byte-equal. Phase 13.

### Active

**v2.3 (Trained Multi-Role Prompt Replacement, β architecture):**
- **REQ-V23-01** — v2.3 v4 β training harness produces a trained per-slot JSON bank from gpt-5.4 outer-loop training
- **REQ-V23-02** — `s_linker14_voyager` consumes trained bank via Voyager-mode runtime injection (extends `s_linker13_skill_learned_clean` pattern)
- **REQ-V23-03** — Training loop = L (linker) + O (text-aware Oracle, failure-mode analysis) + D (text-blind Distillator with CoT-A inline) + P (mechanical probation gate)
- **REQ-V23-04** — Oracle output schema = failure-mode-centric (FMs with affected_slot + symptom + apparent_cause + suggested_direction + evidence_count + abstract_example_pair + newly_introduced_errors with bank-pattern attribution)
- **REQ-V23-05** — Promotion verdict computed against 3-tier bar (STRONG ≥0.9173 / WEAK [0.87, 0.9173) / FAIL <0.87) on gpt-5.4 macro F1, 5-dataset
- **REQ-V23-06** — Dual-artifact registration: `s_linker13_min` retains `canonical=True` (GATE-01 bound); `s_linker14_voyager` ships `experimental=True` (bound only to 0.87 floor)
- **REQ-V23-07** — Mainline train/test split = train MS+TS+TM, test BBB+JAB (Voyager v2 split 1). 3-split Confirmation sweep conditional on mainline ≥ 0.87.
- **REQ-V23-08** — Compact-B fallback (R345 single CoT role) auto-triggers as v2.3 mainline replacement on v4 FAIL (<0.87)
- **REQ-V23-09** — GATE-06 BENCHMARK_TABOO grep + reviewer critic LLM at bank-entry boundary (not O/D handoff)
- **REQ-V23-10** — Per-(text_stem, comp_hash, backend, model) cache infrastructure applied uniformly across L, O, D, reviewer-critic
- **REQ-V23-11** — D's pattern proposals constrained to discourse-syntactic-functional vocabulary (Probe A' R3 vocab anchor carry-forward); forbidden: role nouns + architectural-style names
- **REQ-V23-12** — Slot-uniform bank coverage: all 9 axiom slots (`AMBIGUITY_RULES`, `AMBIGUITY_FEW_SHOT`, `DOC_KNOWLEDGE_EXTRACTION_RULES`, `DOC_KNOWLEDGE_JUDGE_EXAMPLES`, `DOC_KNOWLEDGE_JUDGE_RULES`, `ENTITY_EXTRACTION_RULES`, `VALIDATION_RULES`, `COREF_RULES`, `SEED_DISAMBIGUATION_RULES`) receive trained patterns
- **REQ-V23-13** — Convergence = macro ≥ 0.90 on training projects, max 5 outer passes
- **REQ-V23-14** — Budget cap ~$100 gpt-5.4 total (Probe $5-10 → Range $15-25 → Confirmation $40-60)
- **REQ-V23-15** — Comparison reference: primary `s_linker14_voyager` vs `s_linker13_min` (hand-authored `prompts_v3.py`) gpt-5.4 macro F1; secondary vs `prompts_v3_axiom.py` (axiom-only floor)

## Current State

**Status:** v2.3 kickoff — Trained Multi-Role Prompt Replacement (β architecture) IN PROGRESS. v2.2 archived 2026-06-01. See `.planning/v2.3-prep/v2.3-ARCHITECTURE.md` for locked architecture spec; `.planning/v2.3-prep/v2.3-KICKOFF-SEED.md` for resolved sub-question decisions; `.planning/notes/v23-architectural-endpoint-reasoning.md` for endpoint-choice reasoning.

**Canonical artifact:** `src/llm_sad_sam/linkers/experimental/s_linker13_min.py` (v2.1 promoted, `canonical=True`). Composes trim1 (distilled Tier-1 judge rubric) + trim9 (runtime Tier-2 seed disambiguation rubric) over `prompts_v3` + `helper_v3` + `s_linker13_clean_v3`. Claude macro F1 0.9506; gpt-5.4 macro F1 0.9069.

**Opt-in carve-out (v2.2 shipped 2026-06-01):** `src/llm_sad_sam/linkers/experimental/s_linker14_probe_d_upstream_clean.py` — runtime coref rubric replacing static `COREF_RULES`, enable ONLY when `LLM_BACKEND == openai`. Mediastore gpt-5.4 +1.59pp; BBB gpt-5.4 mean +2.2pp over 2 obs. Claude not promoted (FAIL was confounded — per-backend cache fix unblocks re-test). NOT canonical. See `.planning/v2.2-prep/probe-D-upstream-SUMMARY.md`.

**Frozen artifacts (preserved byte-equal):** `s_linker13.py`, `prompts_v2.py`, `data_types_v2.py`, `document_loader_v2.py`, `pcm_parser_v2.py`, `ilinker*.py`.

**Frontier evidence:** 7 rejected trim variants (`s_linker13_trim2..8_*_runtime_clean.py`) shipped as published exploration evidence under Scenario E. Voyager-TLR pilot infrastructure (`s_linker13_skill_learned_clean.py` + `prompts_v3_axiom.py`) parked as v2.2 anchor.

**Standing constraints carried forward:** GATE-01 (Claude floor + cross-model floor), GATE-02 (frozen-compat regression), GATE-06 (generality / zero benchmark-derived values), GATE-07 (canonical registration), Claude Sonnet default, BENCHMARK_TABOO compliance.

## Past Milestones

- **v1.0** (2026-05-29) — Rule-to-LLM Ablation (`s_linker12c` → `s_linker13`). 6 rules removed, 1 rejected (VAR-04 dotted-path). Final macro 0.9509. See [`milestones/v1.0-ROADMAP.md`](milestones/v1.0-ROADMAP.md).
- **v2.0** (2026-05-31) — Complete Rule Removal + Cross-Model. EXT-01 closed empty (negative); EXT-02 auto-skipped; COMBINE retro-satisfied; CROSS evidence published on gpt-5.4 (mixed-result, model-provider-property framing). See [`milestones/v2.0-ROADMAP.md`](milestones/v2.0-ROADMAP.md) and [`milestones/v2.0-MILESTONE-AUDIT.md`](milestones/v2.0-MILESTONE-AUDIT.md).
- **v2.1** (2026-06-01) — Cleanup + Prompt Simplification. 3 trims shipped (Step 0 dead-code + trim1 distilled judge + trim9 runtime seed rubric); 7 frontier variants documented; Voyager-TLR methodology validated for v2.2. `s_linker13_min` promoted (Claude 0.9506, gpt-5.4 0.9069). All 10 requirements + 4 standing gates held. See [`milestones/v2.1-ROADMAP.md`](milestones/v2.1-ROADMAP.md), [`milestones/v2.1-REQUIREMENTS.md`](milestones/v2.1-REQUIREMENTS.md), and [`milestones/v2.1-MILESTONE-AUDIT.md`](milestones/v2.1-MILESTONE-AUDIT.md).
- **v2.2** (2026-06-01) — Probe-Wave Trimmed Close. 4 probes; 1 STRONG survivor (Probe D upstream coref rubric) ships as opt-in gpt-5.4-only carve-out; canonical unchanged (`s_linker13_min`); Voyager v4 multi-role + per-backend cache infra + Probe A' vocab fix deferred to v2.3 as proven prereqs. See [`milestones/v2.2-ROADMAP.md`](milestones/v2.2-ROADMAP.md), [`milestones/v2.2-REQUIREMENTS.md`](milestones/v2.2-REQUIREMENTS.md), and [`milestones/v2.2-MILESTONE-AUDIT.md`](milestones/v2.2-MILESTONE-AUDIT.md).
- **v2.3** (2026-06-01) — Trained Multi-Role Prompt Replacement (β architecture). Verdict: **WEAK** (cross-split macro 90.5% gpt-5.4, +1.6pp over axiom-only floor 88.9%, −0.19pp vs canonical 90.7%). Total cost ~$111. What shipped: `s_linker14_voyager` (`experimental=True`, `canonical=False`) with cross-split bank of 2 patterns in 2 slots (`results/voyager_v4_beta/confirmation/cross_split_final_bank.json`); `s_linker13_min` canonical unchanged. Key finding: **split-fragility** — BBB as training dataset causes all-rollback (0 of 14 patterns committed across 5 passes) due to 6 compounding bugs in the probation gate; without fix, any re-run on BBB-containing splits produces an empty bank. Secondary finding: sentence-local axiom vocabulary ceiling reached for BBB/TM (14 FNs from section-context naming, 7 FPs from responsibility-list gerunds). Phase 18 (Compact-B) not triggered. Next (v2.4): fix probation gate (traceability gate replacing F1-delta gate) + address axiom gaps. See [`milestones/v2.3-ROADMAP.md`](milestones/v2.3-ROADMAP.md) and [`milestones/v2.3-MILESTONE-AUDIT.md`](milestones/v2.3-MILESTONE-AUDIT.md).

**Key v2.0 findings:**
1. The "rule replaced by LLM primitive" thesis has a clean boundary: rules with project-specific surface conventions (dotted-path, casing) cannot be replaced without project-specific calibration. Same failure class hit v1.0 13d and v2.0 EXT-01.
2. Knowledge injection (alias context) yields measurable but bounded lift (+0.7-2.1pp on BBB) — pattern worth preserving for future LLM judge layers.
3. Probe-first methodology validated: cheap feasibility study cut Phase 6 short before a 4th sub-variant cycle.

**Key v2.1 findings:**
1. Lossless rule distillation + reasoning-before-conclusion ordering (trim1) improves macro F1 on BOTH backends (+0.5pp Claude, +0.96pp gpt-5.4). Cross-model Pareto-positive.
2. Runtime rubric generation works for narrow Tier-2 judges (trim9 seed disambiguation) — ships across both backends. Generalizes from alias-judge baseline.
3. Textually-lossless prompt merges can be semantically lossy — role-divergent routing in proposer/judge boundaries is load-bearing.
4. Cross-model penalty is consistent ~3–4pp for runtime-mechanism variants regardless of target prompt — model-capability finding, not mechanism failure.
5. 6/6 runtime variants ACCEPT under Scenario E but only 1/6 ACCEPT under strict gate: the prompt-reduction × accuracy frontier is mappable.
6. GATE-06 cross-dataset isolation methodology operationalized — testable empirical criterion for runtime LLM discovery, replacing strict-grep on outputs.

## Next Milestone Candidates (v2.3+)

Active milestone: none. v2.3 anchor: Voyager v4 multi-role with proven per-backend cache infrastructure + Probe A' vocab-aligned R3 (do NOT re-explore — see `.planning/v2.3-prep/v2.3-KICKOFF-SEED.md`). Additional topics retained from v2.1 + v2.2 deferred items:

- **Voyager v4 multi-role architecture (R1-R5)** — deferred from v2.2 with vocab-aligned R3 + per-backend cache infra as proven prereqs. Fallback: Compact-B (single-role R345 reconciliation per `v2.2-SCOPE-DECISION.md`). v2.3 anchor.
- **Claude Probe D re-test with per-backend cache fix** — methodologically ready; will produce a Claude-authored coref rubric and a clean cross-backend Probe D verdict. Cost ~$1.5. Carried from v2.2 (cache-fix wave 2026-06-01).
- **Per-backend cache infrastructure** — proven in `s_linker14_probe_d_upstream_clean.py` per v2.2 SANITY_PASS; v2.3 should adopt as the default pattern for any runtime LLM rubric. Carried from v2.2.
- **ADAPTER-01** — Multi-model backend-adaptive prompts (re-opened by v2.1 trim4/5/6/7 single-FP rejections). Requires fresh GATE-06 thinking.
- **Self-Refine on accepted variants** (contingent) — Probe C WEAK_PASS on mediastore; judge at ceiling. Re-test only if v2.3 mainline fails.
- **Extended Thinking on judge stages** — Tier-1 ambiguity classifier shows gpt-5.4 calibration errors extended-thinking could recover.
- **Link provenance data structure** — Phase 12 deferred; v2.3 candidate for evidence-trail audit infrastructure.
- **EXT-04** — Emit-biased boundary prompting on alias-discovery (BBB variance band tightening 3pp → 1pp). Variance work, not rule removal.

### Out of Scope (general — applies across milestones)

- New seed/linker approaches (ILinker3+, cross-model ensembles) — this project is rule-reduction on 12c, not exploration
- Non-SAD-SAM tasks (SAM-Code, SAD-Code) — out of dataset scope
- Cost optimization — user has set "no LLM budget limit"; rule-replaceability is the only constraint
- Changes to frozen artifacts (`s_linker13.py`, `prompts_v2.py`, `ilinker*`, `data_types_v2`, `document_loader_v2`, `pcm_parser_v2`) unless required by a rule removal
- Bench leakage: no benchmark-derived words may enter prompt examples (enforced via BENCHMARK_TABOO.md)

## Context

- **Codebase**: retained `s_linker` family through `s_linker12e` plus `ilinker1-3`, v2.0 production artifact `s_linker13`, v2.1 canonical `s_linker13_min` (composed of trim1 + trim9 over `prompts_v3` + `helper_v3`). Runner is `run_ablation.py`. Default model: Claude Sonnet.
- **Baseline memory**: `s_linker12c` (ICSE clean) ~94% macro F1; `s_linker13` (v2.0) macro 0.9509 Claude / 0.9077 gpt-5.4; `s_linker13_min` (v2.1) macro 0.9506 Claude / 0.9069 gpt-5.4.
- **Validated spikes** (re-validate once integrated):
  - 001 `llm-trailing-words` — single LLM call with evidence guardrail replaces structural gate + LLM verify for trailing-word alias enrichment.
  - 002 `rules-audit` — classified all 12 `s_linker12c` helpers: 9 REPLACEABLE, 1 RISKY (`_has_standalone_mention`), 4 ESSENTIAL (parsers/formatters). Ranked removal order defined.
  - 003 `llm-mention-classifier` — LLM enum emission matches regex `_classify_mention` byte-identically, piggybacked on existing entity-extraction prompt.
- **GPT/Claude gap known**: Claude Sonnet is the target backend; gpt-5.4 is the v2.0 CROSS arm and the v2.1 cross-model gate target. v2.1 frontier-map data: cross-model gap ranges 1.81–5.41pp across trim mechanisms; composition does not widen the gap.
- **Dataset strategy**: hard-tier-first (teammates, bigbluebutton — most rule-sensitive) during development, full 5-project macro F1 for every promoted variant.

## Constraints

- **Quality**: Every promoted variant must hold macro F1 ≥ 93% on the 5-project benchmark (relaxed gates and Scenario-E framing per milestone — see Key Decisions for current numeric tolerances).
- **LLM budget**: No upper bound on calls; replaceability trumps cost.
- **Model**: Claude Sonnet only (per user preference; do not switch to Opus or GPT). gpt-5.4 used for cross-model validation only.
- **Data leakage**: Zero benchmark-derived words in prompts OR logic (enforced via GATE-06 + BENCHMARK_TABOO.md). v2.1 GATE-06 cross-dataset isolation methodology operationalizes the test for runtime LLM discovery.
- **Codebase hygiene**: Each rule-removal/trim lands as its own standalone linker variant — user prefers duplicated standalone files over inheritance chains. Frozen artifacts preserved byte-equal across milestones.

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
| KEEP `_has_standalone_mention` in `s_linker13` | Spike 002 classified it RISKY; replacement deferred to v2 (EXT-01 spike) | KEPT (Phase 5, 2026-05-29) |
| GATE-06 generality audit (v2.0) | User flagged at v2.0 kickoff: every new prompt + helper must read as sound/clean/general to any project; no tailored rules in prompt OR logic | Standing policy from v2.0 onward (2026-05-30) |
| LLM-COMBINE stack-vs-unify decision deferred | EXT-01 cost/quality signal will choose between (1) stacked separate LLM primitives in `s_linker14` and (3) unified single-prompt variant. Premature lock would bias the comparison. | Decided post-EXT-01: COMBINE retro-satisfied via existing v1.0 unification; no s_linker14.py |
| GATE-01 cross-model tolerance T = 1.0pp (v2.1) | Pins the loose REQUIREMENTS GATE-01 phrasing "≤ 1pp regression" to a concrete numeric tolerance so Phase 12 trim acceptance and Phase 13 promotion sweeps can be evaluated deterministically. Baseline 0.9077 is the v2.0 CROSS evidence on gpt-5.4. T = 1.0pp means absolute floor 0.8977. | Codified 2026-05-31 (v2.1 Phase 10, Plan 10-04) |
| GATE-01 relaxation (v2.1 Phase 12) | The original GATE-01 Claude floor was too tight to test aggressive "super-simple-prompt" trim mechanisms. v2.1 Phase 12 relaxed the Claude floor to macro F1 ≥ 0.90 and BBB absolute F1 ≥ 0.79 (swattr SAD-SAM expected). Other-dataset drop tolerance (-2pp) unchanged. Cross-model gpt-5.4 floor 0.8977 unchanged. | Codified 2026-05-31 (v2.1 Phase 12, user directive mid-Wave-2) |
| GATE-01 Scenario E (v2.1 Phase 12 runtime-variant exploration) | After the 6-variant runtime extension landed (Plans 12-07 through 12-12), 5/6 variants missed prior relaxation by 0.5–1.5pp on a single dataset or 0.4pp on cross-model. Reframed Phase 12 as a frontier map of prompt-reduction × accuracy, not strict pass/fail. Scenario E: per-dataset drop tolerance -4pp; cross-model floor 0.89 macro F1 (~1.8pp off 0.9077 baseline). Static-distillation variants kept tighter 0.93/0.8977 gates. | Codified 2026-05-31 (v2.1 Phase 12 close, user directive) |
| GATE-06 cross-dataset isolation methodology (v2.1) | Strict-reading of GATE-06 ("project terms in LLM output = leakage") would invalidate every LLM call in the pipeline. CLAUDE.md actually MANDATES dynamic runtime LLM discovery of domain-specific knowledge. Operationalized as: term t in dataset A's runtime artifact is a leak iff (a) t is a PCM component of dataset B ≠ A AND (b) t is NOT in A's PCM AND (c) t is NOT in A's input doc. PASSES on both backends across all 10 trim9 rubrics. | Codified 2026-05-31 (v2.1 Phase 12 Plan 12-05 revisit) |
| Frontier-map vs strict-pass (v2.1) | Phase 12 exploration crossed 9 trim mechanisms; 2 ACCEPT + 7 REJECT under strict gate. Frontier map captures the full design space WITHOUT moving the promotion bar. Future milestones can revisit Scenario-E-feasible variants under a different gate regime. | Codified 2026-06-01 (v2.1 close) |
| v2.2 trimmed-scope close (ship s_linker13_min unchanged + Probe D opt-in gpt-5.4-only + defer v4 to v2.3) | Probe wave found no Pareto-positive cross-backend mechanism; Probe D is gpt-5.4-only; Voyager v4 architecture is dataset-conditional per Probe A' BBB WEAK_PASS. Hybrid path (Option 1 + Option 3 from `v2.2-RANGE-A-PRIME-BBB-AND-CACHE-FIX-SUMMARY.md`) preserves both gpt-5.4 lift and Claude floor while deferring v4 to a milestone with adequate prereqs. | Codified 2026-06-01 (v2.2 close) |
| v2.3 architectural endpoint = (A) Voyager-bank canonical, β architecture (L + O + D-with-CoT-A + P) | (C) Hybrid runtime-rubric + static-rules across slots ruled out via slot-asymmetry-is-ugly principle. (B) Full runtime-rubric demoted to contingency-only (all-slot uniformity required). β chosen over γ full-v4 because A folded as CoT inside D is strictly cheaper without losing the abstraction check; the separate R5 LLM role would over-engineer given Probe A' R5 0/8 BBB evidence. β chosen over α fully-merged because role-separation (text-blind D) closes the v2/v3 leak at lower architectural risk than CoT-discipline alone. | Codified 2026-06-01 (v2.3 kickoff `/gsd:new-milestone` discussion) |
| v2.3 dual-artifact policy (canonical=True s_linker13_min retained; experimental=True s_linker14_voyager ships separately under 0.87 floor) | 0.87 promotion floor conflicts with standing GATE-01 (≥ 0.8977 gpt-5.4 absolute). Dual-track preserves GATE-01 for canonical artifacts while unblocking architectural-exploration test of v4 thesis under a lenient research-grade floor. v2.3 publishes v4 finding (positive or negative) without violating cross-model floor. | Codified 2026-06-01 (v2.3 kickoff) |
| v2.3 Oracle = text-aware (mode i), Distillator = text-blind. Leak defense at bank-entry boundary. | Role-separation (text-blind D) is the minimal architectural barrier that distinguishes v4 from v2/v3 split-fragility. Making O ALSO text-blind would degrade O's error-analysis quality without strengthening the leak defense, because the leak-relevant boundary is the bank-entry filter (GATE-06 grep + reviewer critic on D's pattern proposals), not the O/D handoff. Testing the minimal sufficient defense surfaces whether further text-restriction is necessary. | Codified 2026-06-01 (v2.3 kickoff) |
| v2.3 backend policy = gpt-5.4 only; Claude only if super necessary | User-set hard rule (carried from v2.2 close 2026-06-01). All v2.3 probes, ranges, sweeps default to gpt-5.4. Cross-model verification deferred to v2.4 if reviewers demand it. | Codified 2026-06-01 (v2.2 close; reaffirmed at v2.3 kickoff) |

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
*Last updated: 2026-06-01 — v2.3 milestone kickoff (Trained Multi-Role Prompt Replacement, β architecture)*
