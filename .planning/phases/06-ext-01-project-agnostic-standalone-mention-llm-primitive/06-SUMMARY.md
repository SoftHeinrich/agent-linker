---
phase: 06
status: complete-negative
verdict: gate-05-fail-fundamental
requirements: [EXT-01]
date: 2026-05-31
tags: [ext01, negative-result, gate-05, gate-06, standalone-mention, probe-first, milestone-v2.0]
key-files:
  retained:
    - src/llm_sad_sam/linkers/experimental/s_linker13g_pre.py
    - src/llm_sad_sam/linkers/experimental/s_linker13g_sem.py
    - src/llm_sad_sam/linkers/experimental/s_linker13g_pre_alias.py
    - src/llm_sad_sam/linkers/experimental/s_linker13g_sem_alias.py
    - src/llm_sad_sam/linkers/experimental/s_linker13g_pre_full.py
    - src/llm_sad_sam/linkers/experimental/s_linker13g_sem_full.py
    - src/llm_sad_sam/linkers/experimental/prompts_v2.py
    - scripts/ext01_probe_p1.py
    - scripts/ext01_probe_p2.py
    - scripts/ext01_probe_p3.py
  not-created:
    - src/llm_sad_sam/linkers/experimental/s_linker13g.py  # canonical EXT-01 deliverable; NOT created (close-empty)
decisions:
  - "Phase 6 closes empty: EXT-01 as scoped not viable on BBB"
  - "All probes recovered 0/14 baseline FNs — gap is upstream of _has_standalone_mention"
  - "Phase 7 (EXT-02) auto-skipped per ROADMAP gating"
  - "Phase 8 (COMBINE) parent = s_linker13; no EXT-01 primitive to stack"
  - "Probe-first methodology validated as a cheap negative-result pattern"
---

# Phase 6 — EXT-01 Project-Agnostic Standalone-Mention LLM Primitive

## Phase Verdict

**FAIL (negative result).** EXT-01 as scoped — replacing
`_has_standalone_mention` with a project-agnostic LLM primitive — is
empirically unwinnable on BBB across THREE independent attempts:

1. Pure-LLM sub-variants (Plan 06-04): -8.8pp BBB vs parent → GATE-05 FAIL.
2. Alias-aware sub-variants (Plan 06-08): -5.7pp best BBB vs parent → GATE-05 FAIL.
3. Three-direction feasibility probes (Plan 06-09): 0/14 baseline FNs recovered, all below GATE-05 floor.

The phase ships no canonical `s_linker13g.py`. v2.0 milestone proceeds to
Phase 8 (COMBINE) with `s_linker13` as the parent and Phase 9 (CROSS),
with Phase 7 (EXT-02) auto-skipped per ROADMAP gating.

## Methodology

Phase 6 executed two full design generations (pure-LLM, alias-aware) plus
a three-direction feasibility probe study before closing empty:

- **Design generation 1 (Plans 06-01..06-04):** Two pure-LLM sub-variants
  (`s_linker13g_pre`: regex pre-filter + LLM judge; `s_linker13g_sem`:
  LLM-only with dotted-path in prompt). Offline anchor-diff stage,
  finalist selection, GATE-05 hard-tier dev loop. Result: both FAIL on
  BBB by -8.8pp.
- **Design generation 2 (Plans 06-05..06-08):** Four alias-aware
  sub-variants ({pre, sem} × {alias-only, full-knowledge}) feeding
  Tier-1-discovered project knowledge (alias map, link map) into the
  standalone-mention LLM call. Offline anchor-diff, finalist selection
  (all four), GATE-05 hard-tier dev loop. Result: best variant
  `s_linker13g_pre_alias` 0.8319 BBB = -5.7pp vs parent, GATE-05 FAIL.
- **Feasibility probe study (Plan 06-09):** Three cheap BBB-only probes
  (P1 document-level full-context, P2 hybrid regex + LLM rejection, P3
  pure removal) before committing to a third sub-variant build cycle.
  Result: 0/14 baseline FNs recovered by ALL three probes; all below
  GATE-05 floor.

**Probe-first methodology validated.** ~50 LLM calls total across three
probes diagnosed the upstream gap that two prior full sweeps had failed
to surface. This pattern — "build no new sub-variant linker file until a
probe proves at least one direction can clear the floor" — is now an
established negative-result protocol for the project.

## Iteration Table

| Iteration | Variant | TM F1 / Δ vs 0.9374 | BBB F1 / Δ vs 0.8890 | GATE-05 | Plan |
| --------- | ------- | ------------------- | -------------------- | :-----: | ---- |
| Parent baseline | s_linker13 (Plan 05) | 0.9474 | 0.8990 | — | — |
| Gen 1 (pure-LLM) | s_linker13g_pre | 0.9381 (PASS) | 0.8108 (-8.8pp) | **FAIL** | 06-04 |
| Gen 1 (pure-LLM) | s_linker13g_sem | 0.9217 (-2.6pp) | 0.8108 (-8.8pp) | **FAIL** | 06-04 |
| Gen 2 (alias-aware) | s_linker13g_pre_alias | 0.9310 (-0.6pp) | 0.8319 (-5.7pp) | **FAIL** | 06-08 |
| Gen 2 (alias-aware) | s_linker13g_sem_alias | 0.9643 (+2.7pp) | 0.8000 (-8.9pp) | **FAIL** | 06-08 |
| Gen 2 (alias-aware) | s_linker13g_pre_full | 0.9231 (-1.4pp) | 0.8182 (-7.1pp) | **FAIL** | 06-08 |
| Gen 2 (alias-aware) | s_linker13g_sem_full | 0.9204 (-1.7pp) | 0.8257 (-6.3pp) | **FAIL** | 06-08 |
| Probe (P1) | doc-level full-context judge | — | 0.8421 (-4.7pp) | **FAIL** | 06-09 |
| Probe (P2) | hybrid regex + LLM rejection | — | 0.8468 (-4.2pp) | **FAIL** | 06-09 |
| Probe (P3) | pure removal (always-True gate) | — | 0.8319 (-5.7pp) | **FAIL** | 06-09 |

Per-plan detail in the linked SUMMARYs (see "Pointer to all 9 plan SUMMARYs"
below). Probe-row F1 numbers come from same-session BBB-only runs and are
within the LLM-variance band of the same-session parent baseline (0.8571);
none rises to the GATE-05 reference floor (0.8890).

## Critical Finding

**All three probes recovered 0/14 baseline FNs.** Across THREE independent
design directions in the probe study — and across SIX sub-variant
implementations across the two prior design generations — the FN cluster
driving the BBB recall gap (HTML5 Client / HTML5 Server abbreviation
references + the "the client" / "the server" coref chain) is never
recovered.

**Diagnosis:** the recall gap lives in the extraction / coref tier, not
in the standalone-mention rule. "The client" / "the server" / bare
"HTML5" must be resolved to the named BBB component by upstream tiers
BEFORE the standalone-mention gate sees them as candidates. The gate
itself — regex or LLM — never receives these sentences as input. No
rewrite of `_has_standalone_mention` (LLM or otherwise) can recover them.

This is the empirical fact the close-empty adjudication rests on. It is
invariant to LLM jitter (FN-set membership is determined by gold-pair
presence, not by overall F1) and it converges across 9 independent
attempts at the EXT-01 surface. Future work targeting this recall gap
must address the upstream alias / coref / entity-extraction tier, not
the standalone-mention rule.

## Variance Context

Same-session parent gave BBB 0.8571 vs the Plan 05 reference 0.8990 — a
~4pp gap on the SAME variant on the SAME dataset with the SAME LLM model.
This is consistent with the documented LLM-variance band (MEMORY.md: "LLM
Variance — same model gives different behaviour across days, ±4pp on
BBB"). The variance band complicates F1-based comparison across runs but
does NOT affect the FN-recovery metric, which is the empirical lever the
close-empty decision rests on.

Implication for future probe-first methodology: F1 is a noisy signal at
the run-of-1 level; FN-recovery counts against a shared same-session
baseline are the robust signal. The probe pattern should always write a
same-session baseline (as P3 did) so the three directions share a
reconciled reference set.

## EXT-01 cost/quality signal (Phase 8 input)

**EXT-01 NOT SHIPPED.** No EXT-01 primitive exists in any current or
forthcoming linker. `s_linker13` is the Phase 8 parent.

**Empirical evidence justifying the no-ship decision:**

| Generation | Approach | Best BBB F1 | GATE-05 status | LLM cost vs s_linker13 |
| ---------- | -------- | ----------: | :------------: | ---------------------- |
| Gen 1a | pure-LLM pre-filter | 0.8108 | FAIL (-8.8pp) | ↑ (per-sentence judge call) |
| Gen 1b | pure-LLM LLM-only | 0.8108 | FAIL (-8.8pp) | ↑ (per-sentence judge call) |
| Gen 2a | alias-aware pre-filter | 0.8319 | FAIL (-5.7pp) | ↑ (per-sentence + alias context) |
| Gen 2b | alias-aware LLM-only | 0.8000 | FAIL (-8.9pp) | ↑ (per-sentence + alias context) |
| Gen 2c | full-knowledge pre-filter | 0.8182 | FAIL (-7.1pp) | ↑↑ (per-sentence + alias + linkmap) |
| Gen 2d | full-knowledge LLM-only | 0.8257 | FAIL (-6.3pp) | ↑↑ (per-sentence + alias + linkmap) |
| Probe P1 | doc-level full-context | 0.8421 | FAIL (-4.7pp) | ↓ (one call/component) |
| Probe P2 | hybrid rejection | 0.8468 | FAIL (-4.2pp) | ≈ (regex + rare LLM call) |
| Probe P3 | pure removal | 0.8319 | FAIL (-5.7pp) | ↓ (zero gate calls) |

Nine independent attempts, nine GATE-05 failures, zero FNs recovered.

**Phase 8 stack-vs-unify guidance:** The stack-vs-unify decision is
**unconstrained by EXT-01** — there is no EXT-01 primitive to stack.
Phase 8 should treat the available LLM primitives (Spike 001
trailing-words, scope-field, alias-coref fold) on their own merits and
pick stack-vs-unify from those four signals alone. The standalone-mention
rule remains as the only structural rule kept in s_linker13 (i.e.
s_linker13 keeps it; s_linker14 keeps it).

**Future direction (deferred to v2.1+):** address the upstream BBB
extraction / coref gap (HTML5 Client / Server abbreviation chain
resolution). That work is OUT of scope for v2.0 — v2.0 ships the
COMBINE artifact (s_linker14) and the cross-model CROSS report on the
existing s_linker13 baseline.

## GATE-06 audit pointer

`06-GATE-06-AUDIT.md` is **final** for the prompts that WERE built but
will NOT ship. Specifically: the two pure-LLM prompts (Plan 06-01) and
the four alias-aware prompts (Plan 06-05) all received PRE-CLEARANCE
audits with `NO HITS` against the word-bounded BENCHMARK_TABOO scan. No
canonical post-promotion audit was authored because no canonical
`s_linker13g.py` was created.

The three probe scripts (Plan 06-09) received a final GATE-06 audit in
`06-FEASIBILITY.md` ("Generality Audit (GATE-06)" section) with verdict
**CLEAN** — no benchmark surface forms in any probe prompt body; alias
map fed to P1 is LLM-discovered project data per D-11.

## Phase 7 disposition

**AUTO-SKIPPED** per ROADMAP.md Phase 7 gating language:

> "Phase 7 is only attempted if Phase 6 passes the dual floor AND its
> GATE-06 audit is clean. If Phase 6 fails either gate, Phase 7 is
> skipped/deferred and the milestone proceeds without EXT-02."

Phase 6 failed GATE-05 (which feeds GATE-01 / dual-floor) across all 9
attempts. The "skipped/deferred" branch applies. Phase 7 plans not
authored, no Phase-7 sub-directory created, no EXT-02 work performed.
EXT-02 remains an open requirement in REQUIREMENTS.md (status: deferred
to v2.1+ along with the upstream-gap work).

## Artifacts retained in tree

Per D-09 / D-12, all rejected sub-variant files are retained as ablation
baselines for the milestone documentation. Nothing is deleted.

- **Linker files (6 sub-variants):**
  - `src/llm_sad_sam/linkers/experimental/s_linker13g_pre.py`
  - `src/llm_sad_sam/linkers/experimental/s_linker13g_sem.py`
  - `src/llm_sad_sam/linkers/experimental/s_linker13g_pre_alias.py`
  - `src/llm_sad_sam/linkers/experimental/s_linker13g_sem_alias.py`
  - `src/llm_sad_sam/linkers/experimental/s_linker13g_pre_full.py`
  - `src/llm_sad_sam/linkers/experimental/s_linker13g_sem_full.py`
- **Prompts (6 standalone-mention prompt constants):**
  - `src/llm_sad_sam/linkers/experimental/prompts_v2.py` —
    `STANDALONE_MENTION_RULES_{PRE_FILTERED,LLM_ONLY}` (pure-LLM) and
    `STANDALONE_MENTION_RULES_{PRE_FILTERED,LLM_ONLY}_{ALIAS_AWARE,FULL_KNOWLEDGE}`
    (alias-aware quartet).
- **Probe scripts:**
  - `scripts/ext01_probe_p1.py`, `scripts/ext01_probe_p2.py`,
    `scripts/ext01_probe_p3.py`
- **Ablation harness entries:**
  - `run_ablation.py` — 6 sub-variants registered with `canonical=False`
    (kept as rejected baselines per D-09 / D-12).
- **Phase artifacts:**
  - `06-CONTEXT.md`, `06-RESEARCH.md`, `06-PATTERNS.md`,
    `06-DISCUSSION-LOG.md`, `06-DIFF-MATRIX.md`,
    `06-DIFF-MATRIX-ALIAS.md`, `06-GATE-06-AUDIT.md`,
    `06-FEASIBILITY.md`
- **NOT created:** `src/llm_sad_sam/linkers/experimental/s_linker13g.py`
  (canonical EXT-01 deliverable — close-empty, no promotion).

## Plan SUMMARYs (pointers)

- [06-01-SUMMARY.md](06-01-SUMMARY.md) — Prompt design + GATE-06 pre-clearance (pure-LLM)
- [06-02-SUMMARY.md](06-02-SUMMARY.md) — Pure-LLM sub-variant scaffolding (s_linker13g_pre / _sem)
- [06-03-SUMMARY.md](06-03-SUMMARY.md) — Pure-LLM offline anchor-diff + finalist set
- [06-04-SUMMARY.md](06-04-SUMMARY.md) — GATE-05 fail #1 → design pivot to alias-aware (D-07..D-11)
- [06-05-SUMMARY.md](06-05-SUMMARY.md) — Alias-aware prompt design + GATE-06 pre-clearance
- [06-06-SUMMARY.md](06-06-SUMMARY.md) — 4 alias-aware sub-variants (s_linker13g_{pre,sem}_{alias,full})
- [06-07-SUMMARY.md](06-07-SUMMARY.md) — Alias-aware offline anchor-diff + finalist set (all 4)
- [06-08-SUMMARY.md](06-08-SUMMARY.md) — GATE-05 fail #2 → design pivot to feasibility-first probes (D-12..D-15)
- [06-09-SUMMARY.md](06-09-SUMMARY.md) — 3-direction feasibility probe study + user adjudication (close-empty)

## Self-Check: PASSED

- `06-FEASIBILITY.md` contains "User adjudication" section — verified
- `06-09-SUMMARY.md` exists — verified (commit 3018210)
- Probe scripts exist in tree — verified (P1/P2/P3 all FOUND)
- Six sub-variant linker files retained — per 06-08-SUMMARY.md disposition
- No `s_linker13g.py` canonical file created — per close-empty decision
- All prior plan SUMMARYs exist on disk — verified (06-01..06-09 SUMMARYs present in phase dir)
