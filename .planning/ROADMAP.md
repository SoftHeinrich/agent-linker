# Roadmap: llm-sad-sam-v45

## Milestones

- ✅ **v1.0 — Rule-to-LLM Ablation** (`s_linker12c` → `s_linker13`) — Phases 1–5 — shipped 2026-05-29. Final macro F1 0.9509. See [`milestones/v1.0-ROADMAP.md`](milestones/v1.0-ROADMAP.md).
- ✅ **v2.0 — Complete Rule Removal + Cross-Model — Generality First** — Phases 6–9 — shipped 2026-05-31. EXT-01 closed empty (negative), CROSS evidence published on gpt-5.4. See [`milestones/v2.0-ROADMAP.md`](milestones/v2.0-ROADMAP.md) and [`milestones/v2.0-MILESTONE-AUDIT.md`](milestones/v2.0-MILESTONE-AUDIT.md).
- ✅ **v2.1 — Cleanup + Prompt Simplification** — Phases 10–13 — shipped 2026-06-01. `s_linker13_min` PROMOTED (Claude macro 0.9506, gpt-5.4 macro 0.9069). 3 trims shipped (Step 0 dead-code + trim1 distillation + trim9 runtime seed rubric) + 7 frontier variants documented. Voyager-TLR methodology validated for v2.2. See [`milestones/v2.1-ROADMAP.md`](milestones/v2.1-ROADMAP.md) and [`milestones/v2.1-MILESTONE-AUDIT.md`](milestones/v2.1-MILESTONE-AUDIT.md).
- ✅ **v2.2 — Probe-Wave Trimmed Close** — `s_linker13_min` unchanged + Probe D opt-in (gpt-5.4 only) — shipped 2026-06-01. 4 probes ran; 1 strong survivor (Probe D upstream coref rubric) shipped as opt-in carve-out. Voyager v4 multi-role + per-backend cache infrastructure + Probe A' vocab fix carried to v2.3 as proven prereqs. See [`milestones/v2.2-ROADMAP.md`](milestones/v2.2-ROADMAP.md) and [`milestones/v2.2-MILESTONE-AUDIT.md`](milestones/v2.2-MILESTONE-AUDIT.md).
- ✅ **v2.3 — Trained Multi-Role Prompt Replacement (β architecture)** — Phases 14–19 — shipped 2026-06-01. WEAK verdict (cross-split macro 90.5%, gpt-5.4). `s_linker14_voyager` ships experimental=True. See [`milestones/v2.3-ROADMAP.md`](milestones/v2.3-ROADMAP.md) and [`milestones/v2.3-MILESTONE-AUDIT.md`](milestones/v2.3-MILESTONE-AUDIT.md).
- ✅ **v2.4 — Probation Gate Fix + Axiom Improvements + v4 Re-run** — Phases 20–24 — shipped 2026-06-01. WEAK verdict (cross-split macro 90.5%, gpt-5.4). Oracle cache contamination + slot steering gap identified; v2.5 scoped to fix both. See [`milestones/v2.4-ROADMAP.md`](milestones/v2.4-ROADMAP.md) and [`milestones/v2.4-MILESTONE-AUDIT.md`](milestones/v2.4-MILESTONE-AUDIT.md).
- ✅ **v2.5 — Oracle Cache Fix + 15-Slot Expansion + Re-run** — Phases 25–30 — shipped 2026-06-02. WEAK verdict (cross-split macro 89.1%, gpt-5.4). Oracle cache fix validated (split-2 committed Pass 1); 15-slot expansion: 5/6 new slots with committed patterns. See [`milestones/v2.5-ROADMAP.md`](milestones/v2.5-ROADMAP.md) and [`milestones/v2.5-MILESTONE-AUDIT.md`](milestones/v2.5-MILESTONE-AUDIT.md).
- ❄️ **v2.6 — ILinker4 + LLM-Driven Training + Axiom Re-run** — Phases 31–37 — close FROZEN behind v2.6.1 (2026-06-02). Phase 37 close tasks (GATE-06 'Persistence' fix + audit) deferred. See [`milestones/v2.6-ROADMAP.md`](milestones/v2.6-ROADMAP.md).
- ✅ **v2.6.1 — No-Training Axiom Linker (s_linker15) + Axiom FP Fixes (PATCH)** — shipped 2026-06-03. Dropped Voyager training; `s_linker15` = axiom-only standalone (inlined prompts + 3 FP fixes), no bank. macro 89.1% gpt-5.4 / 92.7% Claude. Finding: training adds nothing (s15 = trained s14 on gpt); FP fixes fire on Claude, inert on gpt. See [`milestones/v2.6.1-ROADMAP.md`](milestones/v2.6.1-ROADMAP.md) and [`milestones/v2.6.1-MILESTONE-AUDIT.md`](milestones/v2.6.1-MILESTONE-AUDIT.md).
- ✅ **v2.6.2 — Multi-Framing Extraction Design (s_linker17a/17b)** — shipped 2026-06-03. 17a (rename-only) ≈ s15 within GPT variance (validates ICSE Framing A/B/C naming). 17b (k=2 unified) regresses TM −4.2pp / BBB −7.4pp (k=2 too conservative). ICSE decision: use 17a naming for paper. See [`milestones/v2.6.2-ROADMAP.md`](milestones/v2.6.2-ROADMAP.md) and [`milestones/v2.6.2-MILESTONE-AUDIT.md`](milestones/v2.6.2-MILESTONE-AUDIT.md).
- ❄️ **v2.7 — BBB Recall Closure** — Phases 38–42 — FROZEN behind v2.6.2 (2026-06-03). ⚠ Phases 40–41 (recall-oracle training redesign + training re-runs) need re-evaluation vs the v2.6.1 no-training finding before execution. See [`milestones/v2.7-ROADMAP.md`](milestones/v2.7-ROADMAP.md).
- ✅ **v2.6.3 — Paper RQ1–RQ4 Eval via s_linker19 Checkpoint Replay** — Phase 43 — shipped 2026-06-05. Offline replay of `s_linker19` phase-cache pickles populated RQ1/RQ3/RQ4 tables, figures, and paper text. Zero new LLM calls; GATE-01 byte-equal. RQ4 ΔF1 switched to true linker-ablation per D-05. See [`milestones/v2.6.3-ROADMAP.md`](milestones/v2.6.3-ROADMAP.md) and [`milestones/v2.6.3-MILESTONE-AUDIT.md`](milestones/v2.6.3-MILESTONE-AUDIT.md).
- 🔄 **v2.6.4 — Per-Prompt Unit-Tested Minimization + Generality Pass on s_linker19** — Phases 44–49 — IN PROGRESS (started 2026-06-05). Audit every LLM-call site in s_linker19; ship s_linker20 at Pareto-best of size × generality; floor: gpt-5.4 macro ≥ 91.3%.

## Phases

<details>
<summary>✅ v1.0 — Rule-to-LLM Ablation (Phases 1–5) — SHIPPED 2026-05-29</summary>

Phases 1–5 complete. See `milestones/v1.0-ROADMAP.md` for full detail.

</details>

<details>
<summary>✅ v2.0 — Complete Rule Removal + Cross-Model (Phases 6–9) — SHIPPED 2026-05-31</summary>

Phases 6–9 complete. See `milestones/v2.0-ROADMAP.md` for full detail.

</details>

<details>
<summary>✅ v2.1 — Cleanup + Prompt Simplification (Phases 10–13) — SHIPPED 2026-06-01</summary>

Phases 10–13 complete. See `milestones/v2.1-ROADMAP.md` for full detail.

</details>

<details>
<summary>✅ v2.2 — Probe-Wave Trimmed Close — SHIPPED 2026-06-01</summary>

Probe wave (4 mechanisms) + trimmed close. No new canonical promoted; `s_linker13_min` carried forward unchanged. Probe D ships as opt-in gpt-5.4-only carve-out (`s_linker14_probe_d_upstream_clean`). See `milestones/v2.2-ROADMAP.md`.

</details>

<details>
<summary>✅ v2.3 — Trained Multi-Role Prompt Replacement (β architecture) — SHIPPED 2026-06-01 — WEAK verdict (90.5%)</summary>

### v2.3 Phase Summary

- [x] **Phase 14: β Training Harness Infrastructure** — All code (L/O/D/P modules, bank schema, cache adapter, `s_linker14_voyager` linker, GATE-06 helpers) implemented and unit-tested. Zero LLM budget consumed. (completed 2026-06-01)
- [x] **Phase 15: Probe Tier** — 1–2 outer passes on mainline split (gpt-5.4, $5–10). Verdict: CONTINUE (train macro 0.9152 after pass 1). (completed 2026-06-01)
- [x] **Phase 16: Range Tier** — Train to convergence (macro ≥ 0.90 or pass 5 cap), 5-dataset evaluation, 3-tier verdict. (completed 2026-06-01, verdict=WEAK, macro=89.8%)
- [x] **Phase 17: Confirmation Tier** — 3-split sweep, cross-split aggregation, final evaluation, dual-artifact registration. (completed 2026-06-01, verdict=WEAK, cross-split macro=90.5%)
- [x] **Phase 19: Milestone Close** — Milestone audit, requirements close-out, archive. (completed 2026-06-01)

Key findings: split-fragility (BBB in training → split2 empty bank), minimal cross-split consensus (2 patterns survive), probation gate broken (6 bugs). 3 debts carried to v2.4. See `.planning/milestones/v2.3-ROADMAP.md` and `.planning/milestones/v2.3-MILESTONE-AUDIT.md`.

</details>

<details>
<summary>✅ v2.4 — Probation Gate Fix + Axiom Improvements + v4 Re-run — SHIPPED 2026-06-01 — WEAK verdict (90.5%)</summary>

### v2.4 Phase Summary

- [x] **Phase 20: Infrastructure Prep** — Fixed traceability gate (Gate A + Gate B), improved axioms (COREF_RULES + SEED_DISAMBIGUATION_RULES), 5-dataset pre-training baseline eval. (completed 2026-06-01)
- [x] **Phase 21: Probe Tier** — β training with fixed gate + improved axioms on mainline split. Verdict: CONTINUE. (completed 2026-06-01)
- [x] **Phase 22: Range Tier** — Full convergence run; WEAK verdict (macro 90.5%). (completed 2026-06-01)
- [x] **Phase 23: Confirmation Tier** — 3-split sweep; WEAK verdict (cross-split 90.5%). Split-2 zero commits — oracle cache contamination identified as root cause. (completed 2026-06-01)
- [x] **Phase 24: Milestone Close** — Milestone audit, requirements close-out, archive. (completed 2026-06-01)

Key findings: oracle cache contamination (all splits got identical D proposals), slot steering gap (ENTITY_EXTRACTION_RULES never proposed), 2 infrastructure bugs catalogued for v2.5. See `.planning/milestones/v2.4-ROADMAP.md` and `.planning/milestones/v2.4-MILESTONE-AUDIT.md`.

</details>

<details>
<summary>✅ v2.5 — Oracle Cache Fix + 15-Slot Expansion + Re-run — SHIPPED 2026-06-02 — WEAK verdict (89.1%)</summary>

### v2.5 Phase Summary

- [x] **Phase 25: Infrastructure Fixes** — Oracle cache key fix (`bank_content_hash`), probation threshold raised to `delta >= 0.005`, D prompt underfilled-slot steering. (completed 2026-06-02)
- [x] **Phase 26: 15-Slot Expansion** — 6 new slot constants in `prompts_v3_axiom.py`, `ILinker3Injected` subclass, 4 inline prompts wired as slots, harness updated for 15-slot schema. (completed 2026-06-02)
- [x] **Phase 27: Probe Tier** — 2-pass mainline run; oracle fix validated; CONTINUE verdict. (completed 2026-06-02)
- [x] **Phase 28: Range Tier** — Convergence run; WEAK verdict (macro 89.3%); 5/6 new slots received patterns. (completed 2026-06-02)
- [x] **Phase 29: Confirmation Tier** — 3-split sweep; WEAK verdict (cross-split macro 89.1%); split-2 committed 12 patterns (oracle fix confirmed). (completed 2026-06-02)
- [x] **Phase 30: Milestone Close** — Milestone audit, requirements close-out, archive. (completed 2026-06-02)

Key findings: oracle cache fix validated (split-2 committed 12 patterns vs 0/5 in v2.4), 15-slot expansion: 5/6 slots populated (SEED_EXTRACTION_RULES + SEED_ACTOR_RULES empty — ILinker4 needed), lift +1.5pp over axiom-only (87.6%). Debts carried to v2.6: ILinker4, LLM Assessor gate redesign, 3 axiom gaps. See `.planning/milestones/v2.5-ROADMAP.md` and `.planning/milestones/v2.5-MILESTONE-AUDIT.md`.

</details>

<details>
<summary>❄️ v2.6 — ILinker4 + LLM-Driven Training + Axiom Re-run — FROZEN (close deferred)</summary>

### v2.6 Phase Summary

- [x] **Phase 31: ILinker4 + Prompt Hygiene** — `ilinker4.py` built (Voyager-native standalone, first-class SEED slots); 0 inline behavioral rules needing migration; wired into `s_linker14_voyager.py`. GATE-06 clean. (completed 2026-06-02)
- [x] **Phase 32: LLM-Driven Training Loop (v5)** — `voyager_train_tlr_v5.py` built: OD role (O+D merged), LLM Assessor (accept/reject/revise, max 1 revision), [TRAIN]/[TEST] log separation, `run_confirmation` with axiom-only baseline. dry-run verified. (completed 2026-06-02)
- [x] **Phase 33: Axiom Gap Fixes** — `prompts_v4_axiom.py` created; COREF_RULES (Gap 1 SCN + Gap 3 alias), SEED_DISAMBIGUATION_RULES (Gap 2 gerund), ALIAS_SCOPE_RULES + ANTECEDENT_ALIAS_RULES axiomized; s_linker14_voyager imports v4. (completed 2026-06-02)
- [x] **Phase 34: Probe Tier** — KILL verdict. [TEST] macro=84.86% < 87% threshold. Assessor active (9 decisions pass 1). v4 Gap 2 gerund rule over-aggressive (teammates train 81.97%). Phases 35+36 SKIPPED. (completed 2026-06-02)
- [ ] **Phase 35: Range Tier** — CONDITIONAL (Phase 34 CONTINUE). Convergence run, max 5 passes, 5-dataset eval, 3-tier verdict. Budget ≤ $25.
- [ ] **Phase 36: Confirmation Tier** — CONDITIONAL (Phase 35 ≥ 0.87). 3-split cross-validation, axiom-only baseline per held-out, final table vs v2.5 (89.1%) and canonical (90.69%). Budget ≤ $60.
- [ ] **Phase 37: Milestone Close** — Unconditional. Milestone audit, requirements close-out, archive.

See `.planning/milestones/v2.6-ROADMAP.md` for full phase details, plans, success criteria, and requirement coverage.

</details>

<details>
<summary>✅ v2.6.3 — Paper RQ1–RQ4 Eval via s_linker19 Checkpoint Replay — SHIPPED 2026-06-05</summary>

Archived → see [`milestones/v2.6.3-ROADMAP.md`](milestones/v2.6.3-ROADMAP.md), [`milestones/v2.6.3-REQUIREMENTS.md`](milestones/v2.6.3-REQUIREMENTS.md), [`milestones/v2.6.3-MILESTONE-AUDIT.md`](milestones/v2.6.3-MILESTONE-AUDIT.md).

**Highlights:**

- Phase 43 closed at 11/11 verification score; all 8 REQ-V263 satisfied.
- 5 plans across 3 waves: REQ + GATE-01 baseline → replay scripts + 60 CSVs → RQ1 + RQ3/RQ4 tables/figures → paper text + GATE-01 verify.
- Code review fixed 9 findings; WR-01 switched RQ4 ΔF1 metric to true linker-ablation per CONTEXT D-05 (Claude Macro \linkerB +0.584, \linkerC +0.060, overlap 21).
- GATE-01 byte-equal preserved throughout. Zero new LLM calls.
- Out-of-scope deferrals: 4 LiSSA cells + 13 RQ2 cells in results.tex remain `\todo{}` for follow-up work.

</details>

## Progress Table (v2.6 — FROZEN)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 31. ILinker4 + Prompt Hygiene | 1/1 | ✅ Complete | 2026-06-02 |
| 32. LLM-Driven Training Loop (v5) | 1/1 | ✅ Complete | 2026-06-02 |
| 33. Axiom Gap Fixes | 1/1 | ✅ Complete | 2026-06-02 |
| 34. Probe Tier | 1/1 | ✅ Complete — KILL verdict ([TEST] 84.86% < 87%) | 2026-06-02 |
| 35. Range Tier | — | ⏭ SKIPPED (Phase 34 KILL) | — |
| 36. Confirmation Tier | — | ⏭ SKIPPED (Phase 34 KILL) | — |
| 37. Milestone Close | 0/TBD | FROZEN | — |

## Progress Table (v2.6.4 — IN PROGRESS)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 44. HARNESS | 2/2 | Complete    | 2026-06-07 |
| 45. AUDIT | 8/8 | Complete    | 2026-06-08 |
| 46. MINIMIZE | 8/8 | Complete   | 2026-06-08 |
| 47. SHIP | 2/2 | Complete    | 2026-06-09 |
| 48. SWEEP | 0/TBD | Not started | — |
| 49. MILESTONE CLOSE | 0/TBD | Not started | — |

## Phase Details

### Phase 44: HARNESS

**Goal**: A pytest snapshot harness backed by existing phase-cache pickles gives zero-cost per-prompt golden-replay tests for all 6 s19 prompt sites, so any subsequent prompt change can be verified without triggering a single LLM call.
**Depends on**: Nothing (first phase of v2.6.4). Prerequisite artefacts already exist: `results/phase_cache/openai/<project>/{layer1..4,final}.pkl` from v2.6.3.
**Requirements**: REQ-V264-01, REQ-V264-02
**Success Criteria** (what must be TRUE):

  1. `tests/harness/` (or equivalent) loads all 5-project phase-cache pkls and exposes `(prompt_built, llm_response, parsed_output)` triples for each of the 6 s19 prompt sites — zero new LLM calls during load.
  2. Six pytest test modules exist (`test_s_linker20_prompt_{ambiguity,doc_extract,doc_judge,extraction,validation,coref}.py`), each rebuilding the prompt from the fixture and asserting snapshot equality on parsed structured output.
  3. All snapshot tests pass on the unmodified s19 baseline (initial snapshots captured from s19 byte-equal run).
  4. Running the full harness suite with `pytest tests/harness/` completes with exit code 0 and zero LLM API calls (verified by absence of network I/O or mock assertion).**Plans**: 2 plans

**Wave 1**

- [x] 44-01-fixture-infrastructure-PLAN.md — Build tests/harness/ package (manifest, loader, ReplayClient, D-03 adapter map) + add syrupy + pytest-socket dev deps

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 44-02-snapshot-modules-PLAN.md — Six pytest snapshot modules + initial snapshot capture + GATE-01/zero-LLM-call invariants

**UI hint**: no

### Phase 45: AUDIT

**Goal**: Every imported PROMPT CONSTANT and every in-class f-string scaffold used by s_linker19 has a documented generality verdict and a concrete list of candidate cuts, so Phase 46 has an unambiguous input list rather than open-ended exploration.
**Depends on**: Phase 44 (harness must exist so audit verdicts can be checked against it)
**Requirements**: REQ-V264-03, REQ-V264-04
**Success Criteria** (what must be TRUE):

  1. `s_linker20-PROMPT-AUDIT.md` exists and covers all 9 imported PROMPT CONSTANTS (`AMBIGUITY_FEW_SHOT`, `AMBIGUITY_RULES`, `DOC_KNOWLEDGE_EXTRACTION_RULES`, `ALIAS_SCOPE_RULES`, `DOC_KNOWLEDGE_JUDGE_EXAMPLES`, `DOC_KNOWLEDGE_JUDGE_RULES`, `ENTITY_EXTRACTION_RULES`, `VALIDATION_RULES`, `COREF_RULES`) with current LOC, generality verdict (`clean` / `domain-loaded` / `benchmark-leak`), and line-level cut candidates.
  2. The audit also covers all 6 in-class f-string scaffolds (`_prompt_ambiguity`, `_prompt_doc_knowledge_extract`, `_prompt_doc_knowledge_judge`, `_prompt_extraction`, `_prompt_validation`, `_prompt_coref`) with the same columns.
  3. Every `benchmark-leak` finding has a proposed neutral rewording included in the audit doc.
  4. Zero code changes to s19, s13_min, or any imported prompt module — audit is read-only.

**Plans**: 8 plans

Plans:

- [x] 45-01-PLAN.md — Wave 1 — Bootstrap audit doc skeleton (anchors, rubric recap, gating reference table, cut_id legend, placeholder Verdict Summary table)
- [x] 45-02-PLAN.md — Wave 2 — Audit AMB section (AMBIGUITY_FEW_SHOT + AMBIGUITY_RULES + _prompt_ambiguity; CUT-AMB-NN incl. drop-block per REQ-V264-06)
- [x] 45-03-PLAN.md — Wave 2 — Audit DKX section (DOC_KNOWLEDGE_EXTRACTION_RULES + ALIAS_SCOPE_RULES canonical row + _prompt_doc_knowledge_extract; CUT-DKX-NN)
- [x] 45-04-PLAN.md — Wave 2 — Audit DKJ section (DOC_KNOWLEDGE_JUDGE_EXAMPLES benchmark-leak + DOC_KNOWLEDGE_JUDGE_RULES + _prompt_doc_knowledge_judge; CUT-DKJ-NN incl. drop-block + Family A/B rewordings)
- [x] 45-05-PLAN.md — Wave 2 — Audit EXT section (ENTITY_EXTRACTION_RULES + _prompt_extraction; CUT-EXT-NN, both phase_2_framing_c tags)
- [x] 45-06-PLAN.md — Wave 2 — Audit VAL section (VALIDATION_RULES + _prompt_validation + folded P1_FOCUS/P2_FOCUS/COREF_VALIDATION_FOCUS per CD-6; CUT-VAL-NN, 3 phase tags incl. phase_5_coref_validation)
- [x] 45-07-PLAN.md — Wave 2 — Audit COR section (COREF_RULES + ANTECEDENT_ALIAS_RULES + _prompt_coref + ALIAS_SCOPE_RULES back-reference; CUT-COR-NN)
- [x] 45-08-PLAN.md — Wave 3 — Finalize summary table + REQ-V264-03/04 tick-off + GATE-01 byte-equal verification + commit

**UI hint**: no

### Phase 46: MINIMIZE

**Goal**: Each candidate cut from the audit is trialled against the Phase 44 golden tests and either committed (snapshot byte-equal, no benchmark vocab introduced) or reverted, producing a minimized prompt set whose Pareto-frontier position (size cut × generality) is fully logged and reproducible.
**Depends on**: Phase 44 (golden tests), Phase 45 (candidate-cut list)
**Requirements**: REQ-V264-05, REQ-V264-06, REQ-V264-07
**Success Criteria** (what must be TRUE):

  1. `s_linker20-MINIMIZE-LOG.md` exists with one row per candidate cut listing: which prompt/constant, the change attempted, verdict (kept / reverted / unsafe), and which golden snapshot(s) were checked.
  2. For every kept cut, the golden test suite passes byte-equal on parsed structured outputs after the cut is applied.
  3. Few-shot blocks (`AMBIGUITY_FEW_SHOT`, `DOC_KNOWLEDGE_JUDGE_EXAMPLES`) have been tested with full-block removal; where removal breaks byte-equality, the smallest passing replacement (synthetic-domain examples or empty) is documented in the log.
  4. All surviving vocabulary in the minimized constants is free of benchmark-derived terms (GATE-06 cross-dataset isolation check applied per constant).
  5. Zero new LLM calls during the minimize loop — all decisions are driven by the cached golden fixtures.

**Plans**: TBD
**UI hint**: no

### Phase 47: SHIP

**Goal**: `s_linker20.py` exists as a self-contained standalone variant with minimized inlined constants, is registered in the runner, and does not touch the byte-equal state of s_linker19 or s_linker13_min.
**Depends on**: Phase 46 (locked minimized prompt set)
**Requirements**: REQ-V264-08, GATE-01
**Success Criteria** (what must be TRUE):

  1. `src/llm_sad_sam/linkers/experimental/s_linker20.py` exists with `experimental=True`, `canonical=False`, no inheritance from `s_linker19`, and all minimized prompt constants inlined directly in the file.
  2. `run_ablation.py --variants s_linker20` executes without error (dry-run or cached mode sufficient; no LLM calls required).
  3. `git diff` on `s_linker19.py` and `s_linker13_min.py` (against their v2.6.3 close hashes) is empty — GATE-01 verified.
  4. The constants imported by `s_linker19` are unchanged on disk (byte-equal) — paper RQ1–RQ4 replay determinism preserved.

**Plans**: 2 plans

Plans:

- [x] 47-01-PLAN.md — Wave 1 — Create standalone s_linker20.py (copy s19 + remove prompts_v5 import + inline 13 minimized constants + 5 builder text edits + class/_VARIANT_NAME rename, no inheritance) + register in run_ablation.py (CANONICAL_VARIANTS + VARIANT_SPECS) — COMPLETE 2026-06-09 (de3b48e, a267a96)
- [x] 47-02-PLAN.md — Wave 2 — Verify: dry-run load (LLM_BACKEND=checkpoint + --list-variants, zero LLM calls) + GATE-01 byte-equal (git diff + sha256sum) + GATE-06 taboo re-grep + s_linker20 registration guard test + CLAUDE.md Active Surface update

**UI hint**: no

### Phase 48: SWEEP

**Goal**: `s_linker20` is validated at gpt-5.4 macro F1 ≥ 91.3% across all 5 datasets within the $20 budget cap, confirming that Pareto-minimized prompts do not regress the 17e-line breakthrough floor.
**Depends on**: Phase 47 (s_linker20 wired into runner)
**Requirements**: REQ-V264-09, GATE-06, GATE-08
**Success Criteria** (what must be TRUE):

  1. `logs/v2.6.4_s_linker20_gpt.log` exists and records a completed 5-dataset gpt-5.4 sweep on `s_linker20`.
  2. Macro F1 ≥ 91.3% (= s17e 92.3% − T 1.0pp).
  3. No individual dataset drops more than 2pp vs s17e per-dataset numbers (MediaStore 94.9%, TeaStore 96.3%, TeaMmates 89.8%, BigBlueButton 80.4%, JabRef 100.0%).
  4. GATE-06 re-verified on `s_linker20`: zero benchmark-derived vocabulary in any inlined constant or f-string scaffold (cross-dataset isolation methodology from v2.1).
  5. Total API cost for this sweep ≤ $20 (GATE-08); cost logged or estimated from token counts.

**Plans**: TBD
**UI hint**: no

### Phase 49: MILESTONE CLOSE

**Goal**: The v2.6.4 milestone is formally closed: all gates verified, MILESTONES.md updated with outcome, and the archive artifacts exist so future milestones have a clean handoff.
**Depends on**: Phase 48 (sweep result in hand)
**Requirements**: GATE-01 (final), GATE-06 (final), GATE-08 (final)
**Success Criteria** (what must be TRUE):

  1. GATE-01 final check passes: `s_linker13_min.py` and `s_linker19.py` SHA-256 byte-equal to their v2.6.3 close hashes.
  2. GATE-06 final check passes: `s_linker20` prompt audit confirms zero benchmark-derived vocabulary remaining.
  3. GATE-08 final check passes: total sweep cost ≤ $20 recorded.
  4. `MILESTONES.md` updated with v2.6.4 shipped entry (verdict, macro F1, key findings, phase count).
  5. `s_linker20-PROMPT-AUDIT.md` and `s_linker20-MINIMIZE-LOG.md` are committed to the repo (or referenced under `.planning/milestones/v2.6.4-*/`).

**Plans**: TBD
**UI hint**: no

## Next Milestone

**v2.6.3 shipped 2026-06-05** — Phase 43 paper-eval closed via checkpoint replay. See [`milestones/v2.6.3-MILESTONE-AUDIT.md`](milestones/v2.6.3-MILESTONE-AUDIT.md) for headline numbers.

**v2.6.4 active** — Phases 44–49. Per-prompt unit-tested minimization + generality pass on s_linker19; ship s_linker20.

**Frozen candidates (after v2.6.4):**

- **v2.6 close (Phase 37)** — GATE-06 'Persistence' taboo fix + v2.6 audit. Frozen since 2026-06-02.
- **v2.7 — BBB Recall Closure (Phases 38–42)** — frozen since 2026-06-03. ⚠ Phases 40–41 (recall-oracle training redesign + training re-runs) need re-evaluation vs the v2.6.1 no-training finding before execution.
- **Out-of-scope writing pass** — backfill LiSSA cells (results.tex lines 18, 24) and RQ2 cells (lines 45, 50, 55, 61, 76) once the LiSSA pipeline and RQ2 metrics work land.
