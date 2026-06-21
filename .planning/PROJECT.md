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

**v2.5 (Oracle Cache Fix + 15-Slot Expansion + Re-run):**

See `.planning/milestones/v2.5-REQUIREMENTS.md` for full REQ-V25-xx list.

Key requirements:
- **REQ-V25-01** — Oracle cache contamination fixed: `bank_content_hash` added to oracle key (`voyager_train_tlr_v4_beta.py` line 455)
- **REQ-V25-02** — Probation variance fix: multi-run averaging or threshold hardening for BBB-containing splits
- **REQ-V25-03** — D slot steering: underfilled-slot list in D prompt (ENTITY_EXTRACTION_RULES, AMBIGUITY_FEW_SHOT, DOC_KNOWLEDGE_JUDGE_EXAMPLES)
- **REQ-V25-04** — 6 new bank slots in `prompts_v3_axiom.py`: SEED_EXTRACTION_RULES, SEED_ACTOR_RULES, GENERIC_WORD_USAGE_RULES, ALIAS_SCOPE_RULES, ANTECEDENT_ALIAS_RULES, COREF_TERMINAL_SPECIFICITY_RULES
- **REQ-V25-05** — `ilinker3.py` `_prompt_extract()` + `_prompt_actor()` wired for bank injection (empty string = current behavior)
- **REQ-V25-06** — `s_linker14_voyager.py` inline static prompts replaced by slot injection (ALIAS_SCOPE_SCHEMA, ANTECEDENT_ALIAS_GUIDE, generic filter, `_classify_specific_terminals`)
- **REQ-V25-07** — Training harness Oracle + Distillator updated to recognize and propose for all 15 slots
- **REQ-V25-08** — Full 3-split Confirmation re-run with clean infrastructure; promotion verdict vs STRONG threshold ≥ 0.9173

## Current Milestone: v2.6.6 — Standalone RQ3/RQ4 Eval Infra (s_linker20_union)

**Goal:** Build a small, fully self-contained eval bundle under `../working/` that deterministically replays the frozen `s_linker20_union` per-run checkpoints (both backends, N≥3) to compute RQ3 (validator-contribution) and RQ4 (per-module + knowledge A/B) ablation results as full-detailed CSVs + SUMMARY.md, reproducible from that directory alone.

**Target features:**
- Source = frozen per-run `s_linker20_union` phase_caches (**NOT s19**): gpt `results/v2.6.5_s20union/gpt/run{1..N}/phase_cache` + sonnet `results/v2.6.5_s20union_sonnet/run{1..N}/phase_cache`.
- Extraction bridge (agent-linker side): layer1–4 + final decision state → neutral, stdlib-loadable JSON (entity candidates/validated/decisions incl. p1/p2 gates; coref raw/validated/decisions; knowledge layer model_knowledge+doc_knowledge; final links + provenance/source).
- RQ3: Full / NoEntityValid / NoCitation / NoValidator validator ablation by replay; per-validator TP-preserved vs FP-removed, ΔF1-if-removed, per-component distribution; N≥3 mean ± range.
- RQ4 (redesigned, symmetric): (a) per-linker-module entity-only / coref-only / union → F1, unique TPs, UpSet overlap (|only_E|/|both|/|only_C|), coverage, noise; (b) **Full vs No-Knowledge** A/B from a new knowledge-disabled run.
- No-Knowledge: add a knowledge-disable path to `s_linker20_union`; run 5 proj × {gpt, sonnet} × N≥1 (bounded live calls — the only non-replay scope).
- Output: full-detailed per-run × per-config × per-project CSVs + per-link audit CSV + macro summaries + SUMMARY.md; both backends.
- Bundle: `../working/` fully self-contained (vendored neutral extracts + sad→sam gold + ported stdlib metric core; one `run.py`; no sibling-repo path deps).

**Standing gates:**
- GATE-01: canonical/paper artifacts untouched — `s_linker13_min.py`, `s_linker19.py`, and `s_linker20_union.py` full-knowledge behavior byte-/snapshot-stable.
- GATE-06: no benchmark-derived vocabulary introduced in any new code.
- PARITY: standalone Full-config macro reproduces the frozen `s_linker20_union` run numbers within tolerance; `run.py` reruns bit-identical.

**Scope note:** Deterministic replay for RQ3 + RQ4-modules (zero LLM); bounded live runs only for the No-Knowledge axis. Outputs are CSVs + SUMMARY.md (paper TeX untouched).

---

## Paused Milestone: v2.6.4 — Per-Prompt Unit-Tested Minimization + Generality Pass on s_linker19

**Goal:** Audit every LLM-call site in `s_linker19` (6 prompt builders + their imported PROMPT CONSTANTS) with per-prompt golden-replay unit tests; ship `s_linker20.py` whose prompts are at the Pareto-best of size-cut × generality, without regressing the 17e-line macro F1 floor (gpt-5.4 92.3%, T=1.0pp → floor 91.3%).

**Target features:**
- Per-prompt unit-test harness using v2.6.3 `phase_cache/openai/<project>/{layer1..4,final}.pkl` artefacts as golden-replay fixtures (zero new LLM calls for harness build)
- Pytest + snapshot/syrupy on parsed JSON outputs for each of the 6 s19 prompt sites: ambiguity, doc-knowledge extract, doc-knowledge judge, extraction, validation, coref
- Audit + rewrite of imported PROMPT CONSTANTS (`AMBIGUITY_FEW_SHOT`, `AMBIGUITY_RULES`, `DOC_KNOWLEDGE_EXTRACTION_RULES`, `ALIAS_SCOPE_RULES`, `DOC_KNOWLEDGE_JUDGE_EXAMPLES`, `DOC_KNOWLEDGE_JUDGE_RULES`, `ENTITY_EXTRACTION_RULES`, `VALIDATION_RULES`, `COREF_RULES`) AND in-class f-string scaffolding
- Generality target: "look general but still SAD/SAM-tuned" — strip benchmark + jargon surface vocabulary; behaviour stays tuned to SAD→SAM
- Lexical scope: rewrite few-shot examples + RULES constants to neutral domain; drop few-shots entirely when removal does not move the per-prompt golden test
- Size-cut policy: Pareto-frontier per prompt — keep removals only when per-prompt golden tests are byte-equal on parsed structured outputs
- New variant `s_linker20.py` (`experimental=True`, `canonical=False`); `s_linker19.py` preserved byte-equal (paper RQ1–RQ4 replay determinism)
- Backend: gpt-5.4 only (per v2.3 standing policy)

**Standing gates:**
- GATE-01: `s_linker13_min.py` + `s_linker19.py` byte-equal (canonical/paper untouched)
- GATE-06: no benchmark-derived vocabulary in any prompt or constant (re-verified across both layers)
- Macro F1 floor: `s_linker20` GPT-5.4 5-dataset macro ≥ 91.3% (s17e 92.3% − T 1.0pp)

**Frozen / deferred carried forward:** v2.7 (BBB recall closure, Phases 38–42) FROZEN; v2.6 close (Phase 37 GATE-06 'Persistence' taboo fix) DEFERRED.

**v2.6.3 Outcome (prior):** PASSED — paper RQ1–RQ4 cells populated via s_linker19 checkpoint replay; zero new LLM calls. See `milestones/v2.6.3-MILESTONE-AUDIT.md`.

## Current State

**Status:** v2.6.6 (Standalone RQ3/RQ4 Eval Infra) — in progress (started 2026-06-21). **Phase 50 EXTRACT complete & verified 2026-06-21** (6/6 must-haves): `scripts/extract_s20union_caches.py` bridges all 30 frozen `s_linker20_union` phase_caches (gpt+sonnet × run1-3 × 5 projects) → deterministic neutral stdlib JSON under `results/v2.6.6_extracts/`, 30/30 faithfulness PASS, byte-identical re-run, GATE-01/06 clean (EXTRACT-01/02/03). Next: Phase 51 (NOKNOW). Prior milestone **v2.6.4 PAUSED** after Phase 48 (negative s20 minimization result; Phase 49 CLOSE not run). v2.6.4 detail below: Phases 44–47 complete. Phase 47 (SHIP) complete 2026-06-09: `s_linker20.py` shipped as a standalone variant (1086 lines) with all 13 Phase 46 minimized prompt constants inlined, no inheritance from s19, registered in `run_ablation.py` (`experimental=True`, `canonical=False`). GATE-01 byte-equal preserved (s19/s13_min/prompts_v5 unchanged), GATE-06 taboo-clean, 8 registration tests + 97 golden snapshots pass. REQ-V264-08 satisfied. Next: Phase 48 SWEEP (gpt-5.4 macro F1 floor ≥ 91.3%, ≤$20 — gated behind explicit user go-ahead).

**Canonical artifact:** `src/llm_sad_sam/linkers/experimental/s_linker13_min.py` (v2.1 promoted, `canonical=True`, unchanged). Claude macro F1 0.9506; gpt-5.4 macro F1 0.9069.

**Experimental artifact:** `src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py` (`experimental=True`, `canonical=False`). 2-pattern cross-split bank from v2.4. Will be retrained in v2.5.

**Frozen artifacts (preserved byte-equal):** `s_linker13.py`, `prompts_v2.py`, `data_types_v2.py`, `document_loader_v2.py`, `pcm_parser_v2.py`, `ilinker*.py`, `s_linker13_min.py`.

**Standing constraints carried forward:** GATE-01 (Claude floor + cross-model floor), GATE-06 (generality / zero benchmark-derived values), GATE-07 (canonical registration), gpt-5.4 default backend, BENCHMARK_TABOO compliance.

## Past Milestones

- **v1.0** (2026-05-29) — Rule-to-LLM Ablation (`s_linker12c` → `s_linker13`). 6 rules removed, 1 rejected (VAR-04 dotted-path). Final macro 0.9509. See [`milestones/v1.0-ROADMAP.md`](milestones/v1.0-ROADMAP.md).
- **v2.0** (2026-05-31) — Complete Rule Removal + Cross-Model. EXT-01 closed empty (negative); EXT-02 auto-skipped; COMBINE retro-satisfied; CROSS evidence published on gpt-5.4 (mixed-result, model-provider-property framing). See [`milestones/v2.0-ROADMAP.md`](milestones/v2.0-ROADMAP.md) and [`milestones/v2.0-MILESTONE-AUDIT.md`](milestones/v2.0-MILESTONE-AUDIT.md).
- **v2.1** (2026-06-01) — Cleanup + Prompt Simplification. 3 trims shipped (Step 0 dead-code + trim1 distilled judge + trim9 runtime seed rubric); 7 frontier variants documented; Voyager-TLR methodology validated for v2.2. `s_linker13_min` promoted (Claude 0.9506, gpt-5.4 0.9069). All 10 requirements + 4 standing gates held. See [`milestones/v2.1-ROADMAP.md`](milestones/v2.1-ROADMAP.md), [`milestones/v2.1-REQUIREMENTS.md`](milestones/v2.1-REQUIREMENTS.md), and [`milestones/v2.1-MILESTONE-AUDIT.md`](milestones/v2.1-MILESTONE-AUDIT.md).
- **v2.2** (2026-06-01) — Probe-Wave Trimmed Close. 4 probes; 1 STRONG survivor (Probe D upstream coref rubric) ships as opt-in gpt-5.4-only carve-out; canonical unchanged (`s_linker13_min`); Voyager v4 multi-role + per-backend cache infra + Probe A' vocab fix deferred to v2.3 as proven prereqs. See [`milestones/v2.2-ROADMAP.md`](milestones/v2.2-ROADMAP.md), [`milestones/v2.2-REQUIREMENTS.md`](milestones/v2.2-REQUIREMENTS.md), and [`milestones/v2.2-MILESTONE-AUDIT.md`](milestones/v2.2-MILESTONE-AUDIT.md).
- - **v2.5** (2026-06-02) — Oracle Cache Fix + 15-Slot Expansion + Re-run. Verdict: **WEAK** (cross-split macro F1 89.1%, gpt-5.4, 5-dataset, 12-pattern bank). Oracle cache fix (REQ-V25-01) validated: split-2 committed 12 patterns in Pass 1 vs 0/5 in v2.4. 15-slot expansion: 5/6 new slots received committed patterns (`SEED_EXTRACTION_RULES` + `SEED_ACTOR_RULES` not populated). Lift +1.5pp over axiom-only floor (87.6%). Remaining debts: BBB always in test-only position (C-1, structural), TM FM-1/FM-2 FPs (C-2). Total cost ~$62 (under $80 cap). GATE-01 PASS: s_linker13_min unchanged (gpt-5.4 0.9069, Claude 0.9506). See [`milestones/v2.5-ROADMAP.md`](milestones/v2.5-ROADMAP.md) and [`milestones/v2.5-MILESTONE-AUDIT.md`](milestones/v2.5-MILESTONE-AUDIT.md).
- **v2.4** (2026-06-01) — Probation Gate Fix + Axiom Improvements + v4 Re-run. Verdict: **WEAK** (cross-split 90.5% gpt-5.4, −0.2pp vs canonical 90.7%). Gate fix (Gate A+B) operational; split-2 still 0/5 commits — post-hoc investigation found D cache collision invalidated cross-split diversity (all splits got identical distillator proposals). Axiom improvements (D-2/D-3) zero net effect. Two new infrastructure bugs catalogued for v2.5: oracle cache contamination (line 455, not yet fixed) + slot steering gap (ENTITY_EXTRACTION_RULES never proposed). Additionally: full prompt audit found 6 static complex prompts outside axiom scope; v2.5 expands bank 9→15 slots. See [`milestones/v2.4-ROADMAP.md`](milestones/v2.4-ROADMAP.md) and [`milestones/v2.4-MILESTONE-AUDIT.md`](milestones/v2.4-MILESTONE-AUDIT.md).
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
*Last updated: 2026-06-21 — v2.6.6 milestone started (Standalone RQ3/RQ4 Eval Infra on s_linker20_union, under ../working); v2.6.4 paused.*
