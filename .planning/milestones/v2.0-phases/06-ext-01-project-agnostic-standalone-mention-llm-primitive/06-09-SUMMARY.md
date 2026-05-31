---
phase: 06
plan: 06-09
status: complete-negative
date: 2026-05-31
requirements: [EXT-01]
verdict: feasibility-probes-negative → close-empty
tags: [ext01, feasibility, probe, bbb, standalone-mention, negative-result]
key-files:
  created:
    - scripts/ext01_probe_p1.py
    - scripts/ext01_probe_p2.py
    - scripts/ext01_probe_p3.py
    - .planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-FEASIBILITY.md
    - .planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-09-SUMMARY.md
decisions:
  - "User picked close-empty: EXT-01 not viable; gap is upstream of standalone-mention rule"
  - "Probe-first methodology validated (cost ~50 LLM calls vs multi-day full sweep)"
  - "Phase 7 auto-skipped per ROADMAP gating; milestone proceeds to Phase 8 + 9 on s_linker13 parent"
---

# Plan 06-09 — Feasibility-First Probe Study (Negative Result, Close-Empty)

## One-liner

Three cheap BBB-only probes (P1 document-level full-context judge, P2 hybrid
regex + LLM rejection, P3 pure removal) all recovered 0/14 baseline FNs and
all fell below the GATE-05 BBB floor (0.8890), confirming the EXT-01 recall
gap lives upstream of `_has_standalone_mention` — no standalone-mention
rewrite can recover it. User adjudicated `close-empty`; Phase 6 closes as a
publishable negative result, Phase 7 auto-skips.

## Probe Methodology

Per CONTEXT D-12 (probe-first policy after the second GATE-05 negative): three
cheap, BBB-only probes test the three remaining EXT-01 design directions
before any new sub-variant linker file is built. Each probe is a standalone
script under `scripts/`; no new code under
`src/llm_sad_sam/linkers/experimental/`.

- **P1 — Document-level full-context judge.** ONE LLM call per
  (component, doc) with the focal component name, the entire BBB document
  as numbered sentences, and the LLM-discovered alias map (per D-11,
  project-agnostic data, not benchmark leakage). Output: sentence-set
  verdict. Probe budget: ~12 LLM calls.
- **P2 — Hybrid regex + LLM rejection.** Keep
  `_has_standalone_mention` as default (regex). When regex returns True,
  layer an LLM keep/drop call. Approve-biased on malformed output. Probe
  budget cap: ~200 LLM calls.
- **P3 — Pure removal.** Monkeypatch
  `_has_standalone_mention → return True`. Zero LLM calls for the gate
  itself. Measures the F1 floor when downstream tiers carry standalone
  semantics on their own. Also wrote `baseline_fn_set.json` from a
  same-session unpatched s_linker13 BBB sweep, providing the canonical
  shared FN reference for all three probes.

Probe scripts: `scripts/ext01_probe_p{1,2,3}.py`. Outputs:
`results/ablation_results/ext01_probes/{p1_bbb,p2_bbb,p3_bbb,baseline_fn_set,baseline_fp_set}.json` (gitignored).

## Probe Results

| Probe | BBB F1 | Δ vs parent reference (0.8990) | Δ vs same-session baseline (0.8571) | Δ vs pure-LLM floor (0.8108) | LLM calls | FNs recovered (of 14) | New FPs vs baseline | GATE-05 plausible? |
| ----- | -----: | -----------------------------: | ----------------------------------: | ---------------------------: | --------: | --------------------: | ------------------: | :----------------: |
| P1 — document-level full-context judge | 0.8421 | -0.0569 | -0.0150 | +0.0313 | 12 | 0/14 | 3 | MARGINAL* |
| P2 — hybrid regex + LLM rejection | 0.8468 | -0.0522 | -0.0103 | +0.0360 | 37 | 0/14 (2 TPs lost) | 1 | NO |
| P3 — pure removal (always-True gate) | 0.8319 | -0.0671 | -0.0252 | +0.0211 | 0 | 0/14 | 3 | NO |

\* P1 is marginal because its delta vs the same-session parent (-0.015) sits
within the LLM-variance band; however the run-of-1 BBB result is below the
GATE-05 floor (0.8890) and recovers no baseline FNs.

For reference:

| Reference | BBB F1 | Notes |
| --------- | -----: | ----- |
| s_linker13 parent (Plan 05) | 0.8990 | regex baseline; the GATE-05 reference bar |
| s_linker13 parent (same-session) | 0.8571 | same-day baseline written by P3 (LLM variance) |
| Pure-LLM (Plan 06-04 best) | 0.8108 | rejected baseline floor (D-09) |
| Alias-aware (Plan 06-08 pre_alias) | 0.8319 | best alias-aware variant; still failed GATE-05 |

## Key Finding

**All three probes recovered 0/14 baseline FNs.** The 14 FNs in the
same-session baseline (canonical reference for this run) concentrate on
HTML5 Client / HTML5 Server abbreviation references — exactly the same FN
cluster that 06-04 (pure-LLM) and 06-08 (alias-aware) identified across
two prior negative GATE-05 results. Across THREE independent design
directions (full-context LLM judge, hybrid regex+LLM, pure removal),
NONE of them recovers any of the 14 FNs.

**Diagnosis:** the BBB recall gap is upstream of
`_has_standalone_mention`. The FNs are extraction / coref failures —
"the client" / "the server" / bare "HTML5" must be resolved to the named
BBB component by the alias/coref tier BEFORE the standalone-mention gate
ever runs. The gate (regex or LLM) never sees these as standalone-mention
candidates; they are simply not in its input set. Replacing the gate has
no effect on the FN cluster.

**Conclusion:** EXT-01 as scoped (replace `_has_standalone_mention` with
an LLM primitive) is empirically unwinnable on BBB without first
addressing the upstream extraction / coref gap. The probe-first
methodology proved this in ~50 LLM calls total — orders of magnitude
cheaper than another full 5-dataset sweep would have cost.

## Variance Context

The Plan 05 reference BBB F1 (0.8990) and the same-session baseline written
by P3 (0.8571) differ by ~4pp. This is consistent with the LLM-variance
observation recorded in MEMORY.md: same model, different days, different
sentence-level decisions (especially in Tier-1 ambiguity + Tier-2
validation passes). All three probes' deltas vs the same-session baseline
are within this variance band (P1: -0.015, P2: -0.010, P3: -0.025). A
single-run BBB number cannot distinguish "this design is worse" from "this
run got unlucky" — but the FN-recovery metric is invariant to F1 jitter
(it depends only on which gold pairs are missing from the prediction set,
not on the overall F1) and it is unambiguously 0/14 for all three probes.
This is the empirical fact the close-empty decision rests on.

## User Adjudication (per D-15)

**Resume signal:** `close-empty`

**User reasoning (verbatim):**

> After two GATE-05 negatives (pure-LLM + alias-aware) and a 3-direction
> feasibility probe showing all probes recover 0/14 baseline FNs, the recall
> gap is upstream of `_has_standalone_mention` (extraction/coref tier), not
> at the standalone-mention rule. EXT-01 as scoped is not viable; the
> probe-first methodology proved this cheaply. Phase 7 (EXT-02)
> auto-skipped per ROADMAP gating. Milestone proceeds to Phase 8 (COMBINE
> on s_linker13 parent) + Phase 9 (CROSS).

**Decision timestamp (UTC):** 2026-05-31T02:58:23Z

## Disposition

- **Plan 06-09:** COMPLETE (all 5 tasks executed: 3 probes built/run,
  feasibility report authored, user decision recorded).
- **Plan 06-10:** NOT AUTHORED. Per D-15 / user adjudication, Phase 6
  closes empty — no further sub-variant build cycles.
- **Phase 6:** CLOSE-EMPTY. Verdict = FAIL (publishable negative result,
  parallel to v1.0 VAR-04). See `06-SUMMARY.md` for the phase-level
  disposition.
- **Phase 7 (EXT-02):** AUTO-SKIPPED per ROADMAP gating ("Phase 7 is only
  attempted if Phase 6 passes the dual floor AND its GATE-06 audit is
  clean. If Phase 6 fails either gate, Phase 7 is skipped/deferred").
- **Phase 8 (COMBINE):** Parent variant is `s_linker13` (no
  `s_linker13g` exists to stack with). Stack-vs-unify decision is
  unconstrained by EXT-01 (no primitive to stack).
- **Phase 9 (CROSS):** Unchanged — `s_linker13` and (forthcoming)
  `s_linker14` on GPT-5.2.

## EXT-01 cost/quality signal (Phase 8 input)

**Probes alone are NOT the canonical D-06 signal.** Per the 06-09-PLAN
`<output>` block, the canonical signal would require a full 5-dataset
sweep on a built sub-variant. Because Phase 6 closed empty (no
sub-variant was promoted), there is no canonical EXT-01 cost/quality
signal to emit. The Phase 8 grep-target block lives in the phase-level
`06-SUMMARY.md` and documents the no-ship case ("EXT-01 NOT SHIPPED").
Phase 8 should read that block.

## Files

- `scripts/ext01_probe_p1.py` — P1 document-level full-context judge
- `scripts/ext01_probe_p2.py` — P2 hybrid regex + LLM-rejection
- `scripts/ext01_probe_p3.py` — P3 pure removal
- `results/ablation_results/ext01_probes/p1_bbb.json` (gitignored)
- `results/ablation_results/ext01_probes/p2_bbb.json` (gitignored)
- `results/ablation_results/ext01_probes/p3_bbb.json` (gitignored)
- `results/ablation_results/ext01_probes/baseline_fn_set.json` (gitignored)
- `results/ablation_results/ext01_probes/baseline_fp_set.json` (gitignored)
- `.planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-FEASIBILITY.md`
- `.planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-SUMMARY.md` (phase-level)

## Self-Check: PASSED

- `scripts/ext01_probe_p1.py` — FOUND
- `scripts/ext01_probe_p2.py` — FOUND
- `scripts/ext01_probe_p3.py` — FOUND
- `.planning/phases/06-ext-01-.../06-FEASIBILITY.md` — FOUND (with user-adjudication appended)
- Prior probe commits in history: `8783477` (FEASIBILITY), `b19b691` (P2), `6286eae` (P3), `8f1c5dc` (P1), `47d51e7` (adjudication)
