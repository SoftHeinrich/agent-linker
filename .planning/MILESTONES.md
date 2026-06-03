# Milestones — llm-sad-sam-v45

Historical record of shipped milestones. See `milestones/v[X.Y]-ROADMAP.md`, `milestones/v[X.Y]-REQUIREMENTS.md`, and `milestones/v[X.Y]-MILESTONE-AUDIT.md` for full per-milestone archives.

---

## v2.6.1 — No-Training Axiom Linker (s_linker15) + Axiom FP Fixes (PATCH)

**Shipped:** 2026-06-03
**Audit verdict:** PASSED (research patch). **Tag:** v2.6.1
**Production artifact:** `src/llm_sad_sam/linkers/experimental/s_linker15.py` (`experimental=True`, `canonical=False`)
**Canonical unchanged:** `s_linker13_min.py`

### Delivered

`s_linker15` — drop Voyager training, commit to the axiom-only floor. s_linker14_voyager with all
bank/training machinery removed; axiom prompts inlined (B-variant + three FP root-cause fixes: tier/
platform alias, code-path prefix, functional-alias-as-workflow); ILinker4 on empty seed rules.
Registered alongside s14. Dual-backend validated + zero-cost FP attribution.

### Key findings

- **Training adds nothing:** s15 no-training GPT-5.4 macro 89.1% == trained s14_voyager (89.11%).
- **FP fixes fire on Claude, inert on GPT-5.4** (attribution): TM FP Claude 17→6, GPT 17.
- **BBB Claude 83.5 ≈ canonical s13_min (~85)** — dropping training is free on the hard dataset.

### Stats

| Item | Value |
|------|-------|
| s15 macro F1 (gpt-5.4) | 89.1% |
| s15 macro F1 (Claude Sonnet) | 92.7% |
| Total FP (gpt / Claude) | 31 / 12 |
| Phases | 3 (FP fixes, s_linker15, cleanup+docs) |
| GATE-01 / GATE-06 | PASS / PASS |

### Per-dataset (s15, Claude Sonnet)

| Dataset | F1 |
|---------|----|
| MediaStore | 95.1% |
| TeaStore | 96.4% |
| TeaMMates | 91.4% |
| BigBlueButton | 83.5% |
| JabRef | 97.3% |

### Archive

- `milestones/v2.6.1-ROADMAP.md`
- `milestones/v2.6.1-MILESTONE-AUDIT.md`
- Phase directories retained under `.planning/phases/v2.6.1-*` (no cleanup at this time)

---

## v2.3 — Trained Multi-Role Prompt Replacement (β architecture)

**Shipped:** 2026-06-01
**Audit verdict:** WEAK — cross-split macro F1 = 90.5% (gpt-5.4, 5-dataset). Above the 0.87 floor; below the STRONG threshold (0.9173).
**Production artifact:** `src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py` (`experimental=True`; `canonical=False`)
**Canonical unchanged:** `src/llm_sad_sam/linkers/experimental/s_linker13_min.py` (v2.1 canonical; unchanged)

### Delivered

β multi-role training harness (L + O + D-with-CoT-A + P) producing per-slot JSON pattern banks. `s_linker14_voyager` ships with 2-pattern cross-split bank delivering +1.6pp lift over axiom-only floor. Three splits run (Confirmation tier); 0/3 converged due to broken probation gate. Split2 (MS+TS+BBB) produced 0 committed patterns across 5 passes. Gap to canonical: −0.19pp. Key published finding: split-fragility mechanism (BBB as training data → LLM variance ±3.8pp swamps ≤1pp signal), plus probation gate failure analysis (6 compounding bugs). Three design debts filed as v2.4 candidates.

### Stats

| Item | Value |
|------|-------|
| Cross-split macro F1 (gpt-5.4) | 90.5% |
| Gap to canonical (`s_linker13_min` 90.7%) | −0.19pp |
| Axiom-only floor (cross-split) | 88.9% |
| Trained lift over axiom-only | +1.6pp |
| Patterns in cross-split final bank | 2 (in 2 slots) |
| Splits converged | 0/3 |
| Split2 commits | 0/5 passes (gate broken) |
| Total cost | ~$111 vs $100 cap |
| Phases completed | 5/5 (14, 15, 16, 17, 19; Phase 18 not triggered) |

### Per-dataset (cross-split final bank, gpt-5.4)

| Dataset | F1 |
|---------|----|
| MediaStore | 96.7% |
| TeaStore | 93.7% |
| TeaMMates | 84.6% |
| BigBlueButton | 78.0% |
| JabRef | 100.0% |
| **Macro** | **90.5%** |

### Key Findings

1. **Split-fragility**: BBB as training data causes all-rollback. LLM variance (±3.4–3.8pp per dataset) completely swamps the pattern effect signal (≤1pp).
2. **Minimal cross-split consensus**: 10 raw patterns → 8 clusters → 2 survive the ≥2-split survival filter. The 2 surviving patterns deliver +1.6pp lift.
3. **Probation gate broken**: 6 compounding bugs. Root cause of split2 empty bank and the primary blocker for future re-runs. Fixed in v2.4.
4. **Axiom vocabulary ceiling**: 14 FNs from SCN (BBB+TM) + 7 FPs from responsibility-list gerunds (TM) are not addressable by the current 9 sentence-local slots. Fixed in v2.4.

### Debts Carried to v2.4

| ID | Debt | v2.4 Phase |
|----|------|-----------|
| D-1 | Probation gate redesign (6 bugs → Traceability Gate) | 20-P1 |
| D-2 | Axiom vocabulary gaps (SCN + gerund FPs) | 20-P2 |
| D-3 | Refined v3-style axiom diffs implementation | 20-P2 |

### Files

- Archive: [`milestones/v2.3-ROADMAP.md`](milestones/v2.3-ROADMAP.md), [`milestones/v2.3-REQUIREMENTS.md`](REQUIREMENTS.md), [`milestones/v2.3-MILESTONE-AUDIT.md`](milestones/v2.3-MILESTONE-AUDIT.md)
- Kickoff seed: [`v2.3-prep/v2.3-KICKOFF-SEED.md`](v2.3-prep/v2.3-KICKOFF-SEED.md)
- Architecture spec: [`v2.3-prep/v2.3-ARCHITECTURE.md`](v2.3-prep/v2.3-ARCHITECTURE.md)

---

## v2.2 — Probe-Wave Trimmed Close

**Shipped:** 2026-06-01
**Audit verdict:** `passed` (trimmed scope — see Outcome)
**Production artifact unchanged:** `src/llm_sad_sam/linkers/experimental/s_linker13_min.py` (v2.1 canonical carried forward; no new canonical promoted in v2.2)
**Opt-in carve-out shipped:** `src/llm_sad_sam/linkers/experimental/s_linker14_probe_d_upstream_clean.py` (gpt-5.4 only)

### Delivered

A probe-wave methodology applied to 4 mechanism pillars (Voyager v4 multi-role, problem-statement preamble + cached rubric, Self-Refine on alias judge, upstream-tier rule removal). One strong survivor (Probe D upstream coref rubric, +1.59pp mediastore gpt-5.4, matches Claude baseline) ships as an opt-in gpt-5.4-only carve-out — NOT promoted to canonical because Range BBB Claude FAILED (CONFOUNDED by cross-backend cache reuse, methodologically unblocked but not re-run this milestone). v2.2's canonical is `s_linker13_min` unchanged.

### Stats

| Item | Count |
|------|-------|
| Probes run | 4 (A Voyager v4, B preamble+rubric, C Self-Refine, D upstream rule removal) |
| Probe strong-pass | 1 (D) |
| Probe weak-pass | 1 (C — declined as primary) |
| Probe fail | 1 (B — declined) |
| Probe falsification | 1 (A — R5 100% reject; fixed as A' but BBB WEAK_PASS, deferred to v2.3) |
| New canonical promoted | 0 (s_linker13_min unchanged) |
| Opt-in carve-out shipped | 1 (Probe D, gpt-5.4 only) |
| Variant files retained (negative-finding) | 2 (Probe B, Probe C) |
| Cumulative cost | ~$3 of $200 envelope |

### Per-dataset (carve-out only — Probe D)

| Dataset | Backend | F1 | Anchor | Δ | Verdict |
|---|---|---|---|---|---|
| mediastore | gpt-5.4 | 0.9836 | 0.9677 | +0.0159 | STRONG_PASS (matches Claude baseline) |
| bigbluebutton (original) | gpt-5.4 | 0.7965 | 0.7636 | +0.0329 | STRONG_PASS |
| bigbluebutton (cache-fix re-run) | gpt-5.4 | 0.7748 | 0.7636 | +0.0112 | STRONG_PASS (mean +2.2pp over 2 obs) |
| bigbluebutton | Claude | 0.8073 | 0.8496 | -0.0423 | FAIL — CONFOUNDED by cross-backend cache reuse; per-backend cache fix unblocks re-test |

### Key v2.2 lessons

1. **Probe-wave methodology pays at the milestone-scoping tier.** Four cheap parallel probes (~$3) cut decisively across 4 mechanism pillars in one day, replacing what the original v2.2-MILESTONE-PROPOSAL planned as a 5-phase exploration.
2. **Per-backend cache-key methodology is a precondition for fair cross-model evaluation of runtime LLM rubrics.** The Range D Claude FAIL was indistinguishable from a true cross-model failure until the cache-key fix isolated cross-backend rubric reuse as a confound. Carried to v2.3 as a proven prerequisite.
3. **Vocabulary deadlock is a recurring failure mode in multi-role LLM training architectures.** Probe A's R3/R5 mutual inconsistency on textbook SE vocabulary surfaced via 100% R5 reject. Probe A' resolved it on mediastore by tightening R3 to discourse/syntactic terms — but BBB remained R5 0/8. v4 architecture is dataset-conditional, not universally promotable. Carried to v2.3.
4. **Runtime per-dataset rubrics generalize across pipeline tiers (Phase 12 trim9 seed → Probe D coref) on the same backend, but do NOT trivially transfer across backends.** The trim9 + Probe D pair forms an internal Pareto frontier (Claude likes static, gpt-5.4 likes runtime); shipping both as backend-conditional is a viable v2.3 architecture pattern.

### Files

- Archive: [`milestones/v2.2-ROADMAP.md`](milestones/v2.2-ROADMAP.md), [`milestones/v2.2-REQUIREMENTS.md`](milestones/v2.2-REQUIREMENTS.md), [`milestones/v2.2-MILESTONE-AUDIT.md`](milestones/v2.2-MILESTONE-AUDIT.md)
- Probe-wave SUMMARYs: [`v2.2-prep/v2.2-PROBE-WAVE-SUMMARY.md`](v2.2-prep/v2.2-PROBE-WAVE-SUMMARY.md), [`v2.2-prep/v2.2-RANGE-A-PRIME-BBB-AND-CACHE-FIX-SUMMARY.md`](v2.2-prep/v2.2-RANGE-A-PRIME-BBB-AND-CACHE-FIX-SUMMARY.md), [`v2.2-prep/v2.2-MILESTONE-CLOSE-SUMMARY.md`](v2.2-prep/v2.2-MILESTONE-CLOSE-SUMMARY.md)
- Per-probe: [`v2.2-prep/probe-D-upstream-SUMMARY.md`](v2.2-prep/probe-D-upstream-SUMMARY.md), [`v2.2-prep/range-D-bbb-SUMMARY.md`](v2.2-prep/range-D-bbb-SUMMARY.md), [`v2.2-prep/probe-D-cachekey-fix-SUMMARY.md`](v2.2-prep/probe-D-cachekey-fix-SUMMARY.md), [`v2.2-prep/probe-A-voyager-v4-SUMMARY.md`](v2.2-prep/probe-A-voyager-v4-SUMMARY.md), [`v2.2-prep/probe-A-prime-vocab-aligned-SUMMARY.md`](v2.2-prep/probe-A-prime-vocab-aligned-SUMMARY.md), [`v2.2-prep/probe-A-prime-range-bbb-SUMMARY.md`](v2.2-prep/probe-A-prime-range-bbb-SUMMARY.md), [`v2.2-prep/probe-B-preamble-rubric-SUMMARY.md`](v2.2-prep/probe-B-preamble-rubric-SUMMARY.md), [`v2.2-prep/probe-C-selfrefine-SUMMARY.md`](v2.2-prep/probe-C-selfrefine-SUMMARY.md)
- Scope decision: [`v2.2-prep/v2.2-SCOPE-DECISION.md`](v2.2-prep/v2.2-SCOPE-DECISION.md)

---

## v2.0 — Complete Rule Removal + Cross-Model — Generality First

**Shipped:** 2026-05-31
**Audit verdict:** `passed` (mixed-result — see Findings)
**Production artifact unchanged:** `src/llm_sad_sam/linkers/experimental/s_linker13.py` (v1.0 final; v2.0 ships no new canonical variant)

### Delivered

A published thesis-boundary finding: the "rule replaced by LLM primitive" approach has a clean structural limit. Rules whose correct answer depends on a project-specific surface convention (Java dotted-path, casing convention, abbreviation/coref bridging across upstream tiers) cannot be replaced without project-specific calibration — the same failure class hit v1.0 13d/VAR-04 and v2.0 EXT-01 close-empty AND v2.0 Phase 9 TM cross-model regression. Cross-model evidence for the v1.0 artifact on a non-Claude backend (gpt-5.4) published as a model-provider-property finding.

### Stats

| Item | Count |
|------|-------|
| Phases | 4 (6 closed-empty, 7 auto-skipped, 8 closed no-op, 9 done) |
| Plans | 14 (9 in Phase 6, 0 in Phase 7, 0 in Phase 8, 4 in Phase 9, 1 in Phase 8 documentation) |
| Variant files (rejected baselines, retained for ablation) | 6 (`s_linker13g_pre/sem` + 4 alias-aware) |
| Probe scripts | 3 (P1 document-level, P2 hybrid, P3 pure-removal) |
| Rule replacements attempted | 1 (`_has_standalone_mention` — EXT-01, NEGATIVE) |
| Rule replacements shipped | 0 |
| Cross-model evaluations | 1 (gpt-5.4, 5-dataset) |
| Timeline | 2026-05-30 (kickoff) → 2026-05-31 (shipped) |

### Per-dataset cross-model (gpt-5.4 vs Claude Sonnet)

| Dataset | Claude | gpt-5.4 | Δ |
|---|---:|---:|---:|
| MediaStore | 0.984 | 0.9677 | -1.6pp |
| TeaStore | 1.000 | 1.0000 | 0.0 |
| TeaMMates | 0.947 | 0.7939 | **-15.3pp** |
| BigBlueButton | 0.821 | 0.8037 | -1.7pp |
| JabRef | 1.000 | 0.9730 | -2.7pp |
| **Macro** | **0.9506** | **0.9077** | **-4.3pp** |

GATE-01 cross-model: **does NOT hold** (macro < 0.93). TeaMMates drives the gap via dotted-path + generic-English component naming + GAE-platform conflation (same failure class as v1.0 13d/VAR-04). Per v2.0 thesis: model-provider-property finding, not a defect.

### Key v2.0 lessons

1. **Probe-first methodology pays.** Three lightweight feasibility probes (BBB-only, ~250 LLM calls total) cleanly ruled out the entire EXT-01 design space — saving a fourth sub-variant cycle. Pattern: when two sub-variant generations fail the same gate, probe before iterating again.
2. **Knowledge injection has bounded value.** Alias context lifted BBB by +0.7-2.1pp on the LLM judge layer over pure-LLM, but couldn't close a structural-rule recall gap. Worth preserving as a design pattern but not as a load-bearing fix.
3. **The thesis boundary is clean.** Rules whose answer requires project-specific surface conventions resist LLM replacement. Both v1.0 (13d/VAR-04) and v2.0 (EXT-01) hit the same wall. Future rule-removal candidates should pass a "is this a surface-convention rule?" check upfront.
4. **Dataset shape matters more than model quality.** gpt-5.4 holds within tolerance on 4 of 5 datasets. The cross-model regression concentrates on the one dataset (TM) where component names overlap with generic English AND dotted-path identifiers AND the platform name.

### Files

- Archive: [`milestones/v2.0-ROADMAP.md`](milestones/v2.0-ROADMAP.md), [`milestones/v2.0-REQUIREMENTS.md`](milestones/v2.0-REQUIREMENTS.md), [`milestones/v2.0-MILESTONE-AUDIT.md`](milestones/v2.0-MILESTONE-AUDIT.md)
- Phase artifacts: [`milestones/v2.0-phases/`](milestones/v2.0-phases/)
- CROSS report: `milestones/v2.0-phases/09-cross-gpt-5-2-cross-model-validation/09-CROSS-REPORT.md`
- ABLATION-TABLE addendum: `milestones/v1.0-phases/05-promote-and-ablation-artifact/ABLATION-TABLE.md` (v2.0 rows + 2 explanatory paragraphs)

---

## v1.0 — Rule-to-LLM Ablation (`s_linker12c` → `s_linker13`)

**Shipped:** 2026-05-29 (re-audit `passed`: 2026-05-30)
**Audit verdict:** `passed` (upgraded from `tech_debt` on 2026-05-30 after BBB root-cause investigation)
**Final artifact:** `src/llm_sad_sam/linkers/experimental/s_linker13.py`

### Delivered

Defensible empirical claim that 6 of 7 targeted structural rules in `s_linker12c` can be replaced by LLM primitives without regressing macro F1 below 0.93. Final macro F1 = **0.9509** (+1.04 pp vs the 0.9405 `s_linker12c` baseline).

Per-dataset (13f sweep, used as `s_linker13` row): MediaStore 0.984 / TeaStore 1.000 / TeaMMates 0.947 / BigBlueButton 0.821 / JabRef 1.000.

### Stats

| Item | Count |
|------|-------|
| Phases | 5 (5 complete) |
| Plans | 13 (13 complete; Phase 3 closed empty with negative result) |
| Variant files | 7 (`s_linker13a`-`s_linker13f` + canonical `s_linker13`) |
| Helpers retired | 6 of 7 targeted (`_split_component_name`, `_is_structurally_unambiguous`, `_is_ambiguous_name_component`, `_is_strong_alias`, `_get_strong_alias_mappings`, `_has_strong_alias_mention`) |
| Helpers retired-as-rejection | 1 (`_classify_mention` — VAR-04 negative result) |
| Helpers KEPT on cost grounds | 1 (`_has_standalone_mention` — RISKY per Spike 002 O(N×M)) |
| Timeline | 2026-04-21 (project init) → 2026-05-29 (milestone close) → 2026-05-30 (re-audit `passed`) |

### Key Accomplishments

1. **Phase 1 (Baseline + Infrastructure):** 12c baseline JSON captured; per-variant `_checkpoint_dir` namespacing landed; `diskcache>=5.6.1` + `tabulate>=0.9.0` added; `s_linker13a` (Spike 001 LLM trailing-word) ships under user-loosened BBB 4pp tolerance — macro F1 0.9364.
2. **Phase 2 (Ambiguity Cleanup):** `s_linker13b` removes `_is_structurally_unambiguous` (+0.0114 macro vs 12c); `s_linker13c` inlines and removes `_is_ambiguous_name_component` (parity probe 5/5 byte-identical), macro 0.9314 under user-loosened BBB 6pp.
3. **Phase 3 (Mention Classifier Migration — closed empty):** `s_linker13d` collapses TeaMMates F1 from 0.938 to 0.750 on 33 entity-source FPs from dotted-path Java-package references; VAR-04 retired-as-rejection per user direction; `s_linker13d.py` left in tree as rejection artifact. **This is the milestone's primary publishable finding.**
4. **Phase 4 (Alias Scope + Coref Fold):** `s_linker13e` introduces `scope: global|local` LLM field, retires `_is_strong_alias` + `_get_strong_alias_mappings`; dual-hard-tier protocol clean (|Δ|=0.008). `s_linker13f` folds `_has_strong_alias_mention` into coref prompt — macro **0.9509, best in chain**.
5. **Phase 5 (Promote + Ablation Artifact):** `s_linker13.py` promoted (byte-equivalent to `s_linker13f.py` modulo class/banner per D-44a); `_has_standalone_mention` KEEP-decision logged in `PROJECT.md`; `ABLATION-TABLE.md` + `.tex` generated via `tabulate` (8 rows); `METHODOLOGY.md` shipped (7 sections covering thesis, chain, policy evolution, 13d negative result, dual-hard-tier protocol, deferred items).
6. **Post-milestone root-cause analysis (2026-05-30):** `BBB-ROOT-CAUSE.md` and `BBB-DEEP-SEMANTIC-ANALYSIS.md` produced; identified alias-count → recovery-handle-count correlation as mechanism for intermediate-variant BBB drift; deliverable confirmed not to consume tolerance (BBB band 0.821-0.842 overlaps 12c band 0.818-0.844); audit re-classified `tech_debt` → `passed`.

### Deferred to v2 (4 items)

- **EXT-01** — Spike on replacing `_has_standalone_mention` with LLM primitive (relaxed budget)
- **EXT-02** — Drop dotted-path guard in `_has_standalone_mention` (narrower follow-up to EXT-01)
- **EXT-03** — GPT-5.2 cross-model re-evaluation of `s_linker13`
- **EXT-04** — Emit-biased boundary prompting on alias-discovery to shrink BBB borderline-4 variance band from ~3pp to ~1pp (NEW; motivated by BBB-ROOT-CAUSE.md / BBB-DEEP-SEMANTIC-ANALYSIS.md)

### Standing-Policy Decisions

- BBB per-dataset tolerance loosened from 2 pp → 4 pp → 6 pp during the chain (used by intermediate variants 13a/13c/13e; NOT consumed by deliverable `s_linker13`)
- Macro F1 floor stayed at 0.93 throughout
- Other-dataset tolerance stayed at 2 pp throughout
- Dual-hard-tier protocol applied to widest-blast-radius variant (VAR-05 / `s_linker13e`)

### Known Limitations (Documented Empirical Findings)

- **VAR-04 retirement** — `_classify_mention` cannot be LLM-replaced for dotted-path Java-package conventions; documented as publishable negative result. The no-hand-crafted-rules thesis holds with this caveat: classification of project-specific language-construct references is regex territory.
- **15-sentence BBB structural dead zone** — HTML5 Client/Server and WebRTC-SFU partial mentions (S6, S9-13, S19, S39, S47, S65, S73). Identical across 12c and all variants — neither regex nor LLM globally aliases "the client"/"the server" (over-fire risk). Remediation requires per-sentence partial-injection (EXT-01).
- **BBB borderline-4 variance band** (~3 pp) — S38 BBB web, S73/S76/S79 HTML5 Client. Recovery correlates monotonically with alias-discovery emit count. EXT-04 addresses.

### Archive

- `milestones/v1.0-ROADMAP.md`
- `milestones/v1.0-REQUIREMENTS.md`
- `milestones/v1.0-MILESTONE-AUDIT.md`
- Phase directories retained under `.planning/phases/` (user explicitly requested no `gsd-cleanup` at this time)
