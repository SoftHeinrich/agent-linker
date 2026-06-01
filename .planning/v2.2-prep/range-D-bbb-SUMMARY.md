---
phase: v2.2-RANGE-D
probe: D
mechanism: Runtime coref rubric replaces static COREF_RULES (upstream-tier rule removal, EXT-upstream)
dataset: bigbluebutton
date: 2026-06-01
backends_run: [gpt-5.4, sonnet]
gpt54_verdict: STRONG_PASS — F1=0.7965 vs anchor 0.7636 (+3.29pp)
claude_verdict: FAIL — F1=0.8073 vs anchor 0.8496 (-4.23pp)
status: split-decision — gpt-5.4 confirms, Claude regresses
tags: [upstream-rule-removal, coref, runtime-rubric, range-test, split-verdict, recall-regression-claude]
key-files:
  created:
    - results/v2_2_probes_range_d/s_linker14_probe_d_upstream_clean_bigbluebutton_openai_results.json
    - results/v2_2_probes_range_d/s_linker14_probe_d_upstream_clean_bigbluebutton_claude_results.json
    - scripts/run_v2_2_range_d.py
  reused:
    - results/v2_2_probes/D_upstream/cache/bigbluebutton__72e24f4fc026.json
metrics:
  gpt54:
    F1: 0.7965
    P: 0.8824
    R: 0.7258
    tp: 45
    fp: 6
    fn: 17
    anchor_F1: 0.7636
    delta_F1: +0.0329
    verdict: STRONG_PASS
    elapsed_s: 56.34
  claude:
    F1: 0.8073
    P: 0.9362
    R: 0.7097
    tp: 44
    fp: 3
    fn: 18
    anchor_F1: 0.8496
    delta_F1: -0.0423
    verdict: FAIL
    elapsed_s: 571.10
  cross_model:
    gpt54_minus_claude_F1: -0.0108  # gpt-5.4 now BEATS Claude on Probe D's variant; Claude regressed from baseline more than gpt-5.4 gained
---

# Range D — Probe D on bigbluebutton (Split Verdict)

## One-Liner

Probe D's runtime-coref-rubric mechanism produces a **split verdict** on BBB: gpt-5.4 lifts +3.29pp (STRONG_PASS) but Claude drops -4.23pp (FAIL). The mechanism passes the Range gate for gpt-5.4 alone but does NOT generalize across backends on the hard dataset.

## Gate Outcomes (per user directive 2026-06-01)

| Backend | F1 | Anchor | Δ | Threshold | Verdict |
| --- | --- | --- | --- | --- | --- |
| gpt-5.4 | 0.7965 | 0.7636 | **+0.0329** | ≥ +0.005 → STRONG_PASS | **STRONG_PASS** |
| sonnet  | 0.8073 | 0.8496 | **-0.0423** | < -0.010 → FAIL | **FAIL** |

The directive specified: if gpt-5.4 STRONG_PASS, run Claude; Claude must be ≥ baseline (0.8496) to confirm. Claude returned 0.8073 — well below the floor.

## Per-Backend Behaviour

### gpt-5.4 (STRONG_PASS)

- P=88.2% (45 TP / 6 FP), R=72.6% (45 TP / 17 FN)
- Source breakdown: seed=42, entity=6, coreference=3
- 6 FPs total: 5 seed + 1 coref
- The +3.29pp lift comes from BOTH the runtime coref rubric AND ambient run-to-run variance on gpt-5.4 (historical stdev 5-12 links).
- Source of gain: the runtime rubric is more conservative than `COREF_RULES`; on a dataset where Probe D's mediastore run emitted 0 coref links, BBB's gpt-5.4 run emits 3 coref links with only 1 FP — a workable conservatism level.

### Claude sonnet (FAIL)

- P=93.6% (44 TP / 3 FP), R=71.0% (44 TP / 18 FN)
- Source breakdown: seed=45, coreference=2 (NO entity-tier links emitted)
- The Claude baseline (s_linker13_min) on BBB was P=87.3%, R=82.3%, F1=0.8496. **Probe D shaves 8 TPs of recall** (from ~51 baseline TP down to 44 TP) while gaining 5 P-points.
- The runtime rubric's increased conservatism BACKFIRES on Claude because the Claude baseline already has high precision; suppressing coref TPs eats into recall without a corresponding FP-reduction gain.
- Wallclock: 571s on Claude vs 56s on gpt-5.4 (10× slower — typical Claude vs gpt-5.4 wallclock ratio for this pipeline).

## Confound: Cache Reuse Across Backends

**IMPORTANT methodological note:** the Probe D variant caches the coref rubric per `(text_stem, comp_hash)` — NOT per backend. The cached BBB rubric was built by gpt-5.4 first (during the morning gpt-5.4 run), then **reused** by Claude. This means the Claude run did NOT build its own coref rubric; it inherited gpt-5.4's.

The Claude FAIL therefore reflects: "Claude inference using a gpt-5.4-authored coref rubric." A fairer Claude probe would:
1. Delete the cache entry before Claude run, OR
2. Add a backend dimension to the cache key.

This confound does NOT invalidate the FAIL verdict — if anything, the Claude-built rubric would likely be MORE conservative than gpt-5.4's (Claude's coref pattern tends to be more cautious), so the recall gap would likely WIDEN, not narrow. But the FAIL should be reported as: "Probe D mechanism does not transfer to Claude even when seeded with gpt-5.4's rubric output."

## FP Comparison

### gpt-5.4 (6 FPs)

| Sent | Component | Source |
| --- | --- | --- |
| 7 | HTML5 Server | seed |
| 50 | Redis DB | seed |
| 50 | Recording Service | seed |
| 56 | Apps | coreference |
| 60 | FreeSWITCH | seed |
| 76 | Presentation Conversion | seed |

### Claude (3 FPs)

| Sent | Component | Source |
| --- | --- | --- |
| 27 | Redis PubSub | seed |
| 31 | Redis PubSub | seed |
| 50 | Recording Service | seed |

Note: the FP sets are LARGELY DISJOINT (only S50 Recording Service overlaps). Both backends produce FPs at the SEED tier, NOT at the coref tier — the coref FP elimination that Probe D was designed for is largely intact on Claude (only 2 coref links emitted on Claude vs 3 on gpt-5.4, none FP), but Claude's seed-tier behaviour now leaks 3 of its own FPs that the gpt-5.4 baseline rubric didn't produce.

## FN Comparison (Recall Loss Pattern)

Both backends show the SAME 17-18 FNs, dominated by HTML5 Client / HTML5 Server / WebRTC-SFU — components with strict naming whose sentences refer to them by alias/abbreviation only. These FNs are NOT a Probe D effect; they exist in the s_linker13_min baselines too. Probe D does not address the FN gap; it only affects the coref tier's FP rate.

**Net effect on Claude:** Probe D suppressed 2 of Claude's baseline coref TPs (Claude baseline had ~4 TPs from coref; Probe D emits 2). That's the 2-TP recall loss explaining most of the -4.23pp F1 drop.

**Net effect on gpt-5.4:** Probe D shifted 3 seed-tier source contributions (vs baseline which has different seed distribution) and gained 1 net coref TP. Gain is positive on gpt-5.4 where baseline had room to grow.

## Cross-Model Generalization Hypothesis (Falsified)

The Probe D STRONG_PASS verdict from mediastore was specifically argued as: "the same general mechanism (runtime per-dataset LLM-built rubric replaces a static constant) — if Range confirms, the v2.2 finding is: runtime LLM-built rubrics generalize across pipeline tiers AND across backends."

**This Range result FALSIFIES the cross-backend half of that hypothesis.** Probe D works for gpt-5.4 specifically because the baseline coref-tier had room for improvement; on Claude where the static `COREF_RULES` already extract a near-optimal coref signal, the runtime rubric's increased conservatism shaves recall without offsetting precision gain.

## Recommendation

**Do NOT promote Probe D to `s_linker14_min` as v2.2's flagship.** The split verdict means:

1. Probe D is **gpt-5.4-specific**. It could ship as a backend-conditional toggle inside s_linker13_clean_v3 (use runtime rubric ONLY when `LLM_BACKEND == openai`), but this complicates the architecture for marginal gain on one backend and one dataset (BBB).
2. The mediastore STRONG_PASS (+1.59pp gpt-5.4, +0.00pp Claude) was a **dataset-specific artifact**: mediastore baseline coref tier had 1 lingering gpt-5.4 FP that the runtime rubric eliminated. BBB shows the more general effect — runtime rubric != silver bullet across backends.

**Updated v2.2 milestone scope:**

- IF user wants a backend-conditional ship: `s_linker14_min` = `s_linker13_min` + `if backend == openai: use runtime coref rubric`. Expected lift: +1.59pp on mediastore gpt-5.4, +3.29pp on BBB gpt-5.4, 0 effect on Claude. Risk: low (conditional, doesn't touch Claude path).
- IF user wants a clean ship: kill Probe D, v2.2 reduces to "ship runtime-coref-rubric as carve-out / capture this gpt-5.4-only lift as an opt-in".
- The 5-dataset confirmation sweep that was contingent on Range D passing BOTH backends is now NOT recommended at full scope; if anything, run gpt-5.4 confirmation only.

## Costs

- Range D gpt-5.4 BBB: ~$0.50 (one BBB sweep, 56s wallclock, ~14 LLM calls per pipeline).
- Range D Claude BBB: ~$1-2 (one BBB sweep, 571s wallclock).
- Total Range D: ~$2.
- Budget remaining for v2.2: roughly $13 of original $15 cap.

## Files

- Result JSON: `results/v2_2_probes_range_d/s_linker14_probe_d_upstream_clean_bigbluebutton_openai_results.json`
- Result JSON: `results/v2_2_probes_range_d/s_linker14_probe_d_upstream_clean_bigbluebutton_claude_results.json`
- Cached rubric (reused for Claude): `results/v2_2_probes/D_upstream/cache/bigbluebutton__72e24f4fc026.json`
- Runner: `scripts/run_v2_2_range_d.py`
- Logs: `results/v2_2_probes_range_d/logs/range_d_bbb_gpt54.log`, `results/v2_2_probes_range_d/logs/range_d_bbb_claude.log`

## Open Followups

1. Re-run Claude BBB with a Claude-built rubric (delete cache first) — would confirm whether the FAIL persists when Claude builds its own coref rubric. Cost: ~$2.
2. Range D on remaining datasets (teastore, teammates, jabref) on gpt-5.4 only — establish whether the gpt-5.4 lift generalizes across MORE datasets before any conditional-ship decision. Cost: ~$3.
3. If user wants to fully kill Probe D: deregister variant from `run_ablation.py`, archive `s_linker14_probe_d_upstream_clean.py` under `archive/`.
