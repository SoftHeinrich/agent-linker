---
phase: 06
plan: 06-09
status: feasibility-study
date: 2026-05-31
requirements: [EXT-01]
---

# Phase 6 EXT-01 — Feasibility Probe Report

## Methodology

Per CONTEXT D-12 (probe-first policy after the second GATE-05 negative): three
cheap, BBB-only probes test the three remaining EXT-01 design directions before
any new sub-variant linker file is built. Each probe is a standalone script
under `scripts/`; no new code under `src/llm_sad_sam/linkers/experimental/`
(per D-12).

Each probe ran once on the BigBlueButton (BBB) text/model. P3 additionally ran
an *unpatched* s_linker13 BBB sweep as a same-session baseline so the three
probes share a reconciled FN set (canonical "17 FN" set per CONTEXT — see
"Run-vs-Reference Variance" note below).

**Reference baselines** (from prior plans / CONTEXT.md interfaces):

- **s_linker13 parent BBB F1 (CONTEXT reference):** 0.8990 (Plan 05 promotion)
- **GATE-05 BBB floor:** 0.8890 (parent − 1pp)
- **Pure-LLM rejected baseline (Plan 06-04):** 0.8108 — the floor the probes must beat
- **Alias-aware best (Plan 06-08 pre_alias):** 0.8319 — the last empirical lift
- **Reference FN set ("17 FNs"):** documented in 06-04 / 06-08 SUMMARYs

**Same-session reference (this run's unpatched s_linker13 baseline, written by P3):**

- **s_linker13 parent BBB F1 (same-session):** 0.8571 (TP=48, FP=2, FN=14)
- baseline_fn_set.json captures the 14 FNs as the canonical reference for the
  three probes' FN-recovery metric.

The same-session baseline (0.8571) sits ~4pp below the Plan 05 reference
(0.8990). This is consistent with prior LLM-variance observations
(MEMORY.md: "LLM Variance" — same model gives different behaviour across days).
We use the same-session baseline for delta-vs-baseline reconciliation so all
three probes are compared against a single shared reference run; we keep the
Plan 05 0.8990 number for GATE-05 floor calculations because that is the
documented gate.

## Feasibility Table

| Direction                                  |  BBB F1 | Δ vs parent reference (0.8990) | Δ vs same-session baseline (0.8571) | Δ vs pure-LLM floor (0.8108) | LLM cost (probe)         | Projected sweep cost (5 datasets, rough) | FNs recovered (of 14) | New FPs introduced (vs baseline) | Generality cost                                                                                                                       | Implementation cost                                  | GATE-05 plausible? |
| ------------------------------------------ | ------: | -----------------------------: | ----------------------------------: | ---------------------------: | ------------------------ | ---------------------------------------: | --------------------: | -------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | :----------------: |
| **P1** document-level full-context judge   |  0.8421 |                        -0.0569 |                             -0.0150 |                      +0.0313 | 12 calls (cap 24)        |                                       72 |                  0/14 |                                3 | No regex; LLM-discovered alias map fed at runtime (project-agnostic); safe SE textbook examples only. Cleanest EXT-01 story.          | ~1–2 days (precompute standalone_map + dict lookups) |     MARGINAL\*     |
| **P2** hybrid regex + LLM rejection        |  0.8468 |                        -0.0522 |                             -0.0103 |                      +0.0360 | 37 calls (cap ~200)      |                                      240 | 0/14 (2 TPs lost net) |                                1 | **KEEPS the regex** — demotes it to a cheap pre-filter. Weakens the EXT-01 thesis; requires explicit reframing.                       | ~0.5–1 day (wrap original + LLM-cached rejection)    |         NO         |
| **P3** pure removal (always-True gate)     |  0.8319 |                        -0.0671 |                             -0.0252 |                      +0.0211 | 0 calls (gate is `True`) |                                        0 |                  0/14 |                                3 | Removes the rule entirely; downstream tiers carry standalone semantics on their own. Reviewer-strongest story.                        | Trivial: one-line change (~0.1 day)                  |         NO         |

(For reference, the rejected baselines from prior plans:)

| Reference                          | BBB F1 | Notes                                                |
| ---------------------------------- | -----: | ---------------------------------------------------- |
| s_linker13 parent (Plan 05)        | 0.8990 | regex baseline; the bar to clear (GATE-05 reference) |
| s_linker13 parent (same-session)   | 0.8571 | same-day baseline run by P3 (LLM variance)           |
| Pure-LLM (Plan 06-04 best)         | 0.8108 | rejected baseline floor (D-09)                       |
| Alias-aware (Plan 06-08 pre_alias) | 0.8319 | best alias-aware variant; still failed GATE-05       |

\* **MARGINAL note for P1:** The probe BBB F1 (0.8421) is below the GATE-05
floor of 0.8890 but is within the same-session LLM-variance band of the
parent (-0.015 vs 0.8571). At the same time, the probe's run-of-1 results
cannot rule out luck. P1 is the only candidate that does not regress vs
the same-session parent by more than the variance band, and it is the only
direction that keeps the EXT-01 generality story intact (no regex kept).
The recommendation accounts for this nuance below.

## TP/FP Breakdown (probe-specific)

### P1 — Document-level

- LLM calls: 12, JSON parse failures: 0, fallback-regex hits: 0
- Result: BBB F1 0.8421, TP=48, FP=4, FN=14. Identical FN set to the
  same-session parent baseline (14 of 14 FNs shared) — P1 does not recover
  any baseline FNs but also does not introduce any new FNs of its own.
- Implication: the document-level judge picks essentially the same set of
  standalone-mention sentences as the regex (precision +/-0.03), but does
  not reach into the alias/coref tier to recover the HTML5 Client / Server
  abbreviation references that the regex misses.

### P2 — Hybrid rejection

- Regex would have approved (regex_true_count): 219 calls into
  `_has_standalone_mention` returned True. Of these, the LLM rejection
  prompt fired on 37 unique (component, sentence) pairs (the rest were cache
  hits).
- LLM dropped: 2 (Apps@S51, Apps@S54). Both were TPs in the gold standard.
- FPs killed by LLM drop: 0.
- TPs lost to LLM drop: 2.
- Net signal: NEGATIVE (TPs-lost > FPs-killed). The rejection prompt is
  conservative enough to leave most regex verdicts alone, and when it does
  fire it is more likely to kill a TP than catch an FP on BBB.

### P3 — Pure removal

- Baseline FNs (14) and patched FNs (15): the always-True gate adds 1 new
  FN net. Most baseline FNs (HTML5 Client / HTML5 Server abbreviation
  references) are not recovered because they were never an
  `_has_standalone_mention` False — they were an extraction/coref failure
  upstream of the gate.
- Raw new FPs introduced vs same-session baseline: 3. Anchor explosion is
  smaller than feared — downstream tiers (entity validation, generic
  filter, coref antecedent check) catch most of the extra anchor noise.

## Run-vs-Reference Variance

The Plan 05 reference BBB F1 (0.8990) and the same-session baseline written
by P3 (0.8571) differ by ~4pp. This is consistent with the LLM-variance
observation recorded in MEMORY.md: same model, different days, different
sentence-level decisions (especially in Tier 1 ambiguity + Tier 2
validation passes). All three probes' deltas vs the same-session baseline
are within this variance band (P1: -0.015, P2: -0.010, P3: -0.025).
Implication: a single-run BBB number cannot distinguish "this design is
worse" from "this run got unlucky". A full 5-dataset sweep with 2-3
repetitions would be required for a final verdict.

## Recommendation

**Picked: `close-empty` (Phase 6 closes with documented negative result).**

Empirical evidence supporting this verdict (mechanically derived from the
table):

1. **No probe clears the GATE-05 floor (0.8890).** P1 (0.8421), P2 (0.8468),
   P3 (0.8319) are all 4–6pp below the floor. The Plan 06-04 and 06-08
   negative results already established that the pure-LLM and alias-aware
   sub-variants also fail this floor on BBB. Three independent design
   directions, three independent failure modes, all converge below 0.8890.
2. **None of the probes recovered ANY of the 14 baseline FNs (0/14 for all
   three).** The HTML5 Client / HTML5 Server abbreviation-reference FNs that
   drive the BBB recall gap are upstream of `_has_standalone_mention` — they
   are extraction / coref failures, not gate failures. No standalone-mention
   replacement can fix them.
3. **P1 is the only candidate within the same-session variance band of the
   parent.** Even so, its run-of-1 evidence is insufficient — and even if a
   3-run sweep confirmed the lift, P1's lift would not be over the GATE-05
   reference floor (0.8890), only the same-session parent (0.8571).
4. **P2 has a negative net signal** (2 TPs lost, 0 FPs killed) AND it
   weakens the EXT-01 generality story by keeping the regex.

This matches the v1.0 VAR-04 retirement pattern (CONTEXT.md: negative result
from removing a structural rule without a sufficient LLM substitute on a
high-abbreviation dataset). Phase 6 closes as a publishable negative result.

**Carry-over for the milestone narrative:** the BBB recall gap is an
extraction / coref problem, not a standalone-mention problem. Future work
addressing the HTML5 Client / Server abbreviation issue should target the
alias map, the coref tier, or the entity extraction prompt — not the
`_has_standalone_mention` rule. EXT-01 as defined (replace the rule) is
empirically unwinnable without first addressing the upstream gap.

## Generality Audit (GATE-06)

Confirmation of safe-domain prompt examples and no-benchmark-leakage in the
three probe scripts:

```text
$ grep -REi '(html5|kurento|freeswitch|bigbluebutton|redis|teammates|jabref|teastore|mediastore)' \
       scripts/ext01_probe_p1.py scripts/ext01_probe_p2.py scripts/ext01_probe_p3.py \
       | grep -v -E 'BBB = |bigbluebutton/'
scripts/ext01_probe_p1.py:    "dataset": "bigbluebutton",
scripts/ext01_probe_p2.py:        "dataset": "bigbluebutton",
scripts/ext01_probe_p3.py:        "dataset": "bigbluebutton",
```

The only matches are JSON output-blob string literals identifying the dataset
in the result file. Prompt bodies (PROMPT_TEMPLATE constants in P1/P2) use
only safe SE textbook domains per BENCHMARK_TABOO.md §"Safe SE Textbook
Examples":
- P1 prompt examples: Parser, Scheduler (kernel module), ShoppingCart (with
  alias "cart"), InvoiceHandler.
- P2 prompt examples: Parser (with kernel module / compiler / AST builder),
  Scheduler (generic English use).

No BBB / TS / TM / JAB / MS component name appears in any prompt body. The
alias map fed to P1 at runtime is *LLM-discovered* project data
(doc_knowledge phase output), not hand-curated benchmark surface forms (per
D-11 / GATE-06).

GATE-06 audit verdict: **CLEAN** for all three probe scripts.

## Files

- `scripts/ext01_probe_p1.py` — P1 document-level full-context judge
- `scripts/ext01_probe_p2.py` — P2 hybrid regex+LLM-rejection
- `scripts/ext01_probe_p3.py` — P3 pure removal
- `results/ablation_results/ext01_probes/p1_bbb.json` (gitignored)
- `results/ablation_results/ext01_probes/p2_bbb.json` (gitignored)
- `results/ablation_results/ext01_probes/p3_bbb.json` (gitignored)
- `results/ablation_results/ext01_probes/baseline_fn_set.json` (gitignored;
  shared FN reference, written by P3)
- `results/ablation_results/ext01_probes/baseline_fp_set.json` (gitignored;
  shared FP reference, written by P3)
- `logs/ext01_probe_p1.log`, `logs/ext01_probe_p2.log`,
  `logs/ext01_probe_p3.log` (gitignored)

## User adjudication

- **Option chosen:** `close-empty`
- **Decision timestamp (UTC):** 2026-05-31T02:58:23Z
- **User reasoning (verbatim):**

> After two GATE-05 negatives (pure-LLM + alias-aware) and a 3-direction
> feasibility probe showing all probes recover 0/14 baseline FNs, the recall
> gap is upstream of `_has_standalone_mention` (extraction/coref tier), not
> at the standalone-mention rule. EXT-01 as scoped is not viable; the
> probe-first methodology proved this cheaply. Phase 7 (EXT-02)
> auto-skipped per ROADMAP gating. Milestone proceeds to Phase 8 (COMBINE
> on s_linker13 parent) + Phase 9 (CROSS).

- **Consequence:** Phase 6 closes as a publishable negative result. No
  canonical `s_linker13g.py` is created. Phase 7 (EXT-02) auto-skipped per
  ROADMAP gating ("Phase 7 is only attempted if Phase 6 passes the dual
  floor AND its GATE-06 audit is clean"). v2.0 milestone proceeds to
  Phase 8 (COMBINE) with `s_linker13` as the parent (no EXT-01 primitive
  to stack with) and Phase 9 (CROSS). See `06-09-SUMMARY.md` and
  `06-SUMMARY.md` for the full phase-level disposition.
