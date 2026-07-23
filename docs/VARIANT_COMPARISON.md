# Experimental linker variants

This note distinguishes results that can support a paper claim from useful but
incomplete pilot evidence. It is derived from the vendored study notes and source
in `approach/pilot/`; it does not replace the frozen S21 paper panel in
`sota-links/`.

## S23 verification family

| Variant | What changes relative to S21 | Evidence available | Interpretation |
| --- | --- | --- | --- |
| `s_linker23_verify` | Adds router-proposed candidates behind a separate, unchanged S21 two-pass evidence gate. The regular S21 floor is preserved. | One all-five-dataset default-tier run: macro F1 92.7 vs S21 92.8; macro F2 91.6 vs 89.9; 9 vs 4 FP. The alias-wired run likewise reports F1 parity (92.8) and F2 +1.4pp, but has a separate stochastic S21 floor. | Best-supported S23 *reference design*: a conservative augmentation with a preserved floor. It buys recall/F2, not demonstrated F1 improvement. |
| `s_linker23_verify1p` | Same as `verify`, but the **augmentation gate only** uses S21 evidence pass P1 and drops P2. The ordinary S21 floor still uses P1 AND P2. | Offline, router-less five-dataset tier cache: augmentation gate F1 0.920 (183 TP, 20 FP) versus two-pass 0.896 (168 TP, 12 FP). | Most promising cost/recall hypothesis for a new live study. It retains the S21-floor non-regression design, but has no end-to-end, repeated result yet. Do not present 0.920 as system F1. |
| `s_linker23_verify1p_all` | Extends single-pass gating to both the augmentation and S21's own Phase-4 Framing-C floor. Phase-5 coreference is unchanged. | Checkpoint/replay ablation is described in code; the available pilot cache models the whole candidate union but is not a fresh live run. | A useful cost ablation, not the primary augmentation claim: it changes S21's protected floor, so it loses `verify`/`verify1p`'s floor-preservation argument. |

### What the one-pass cache actually says

`approach/pilot/simplify_verify_pilot.py` is runnable without credentials:

```bash
cd approach
../.venv/bin/python pilot/simplify_verify_pilot.py
```

Across its cached 269 candidates / 195 gold links, P1-only retains 23 additional
links: +15 TP and +8 FP relative to P1-and-P2. Its pooled F1 rises 0.024 and F2
0.056. The cache is explicitly router-less; it isolates a gate decision under
fixed candidates and cannot establish the end-to-end performance of an LLM
router, proposer, or S21 floor.

## Design comparison

`s_linker21_agentrouter` is an earlier design: it gives each sentence a typed
proposer and lets an LLM router choose `VALIDATE`, `CODE`, or `REJECT`. Its pilot
comparison used frozen final S21 outputs, so it is useful for mechanism analysis
but is not a live repeated S21-vs-router study. The live OpenAI attempt recorded
in `results/RUN_STATUS.md` did not finish and must not be scored.

S22 puts typed extraction directly into the live pipeline. It has a favorable
single-run headline in the pilot notes, but a subsequent clean comparison fell
to macro F1 90.9 and the raw headline artifact is absent. Treat it as a
high-variance pilot, not a replication-package default.

The S23 verification design keeps the S21 accepted links as a separate floor and
subjects only new candidates to risk. This matters: the pilot's raw union/shared
gate design was reported at macro F1 88.9 with 21 FP, versus S21 93.3 with 4 FP;
re-batching the floor with speculative candidates also breaks output-equivalence
as an experimental control.

## Recommendation for the package

Keep S21 as the released, reproducible reference. Include `s_linker23_verify`
as the S23 experimental reference because it has the cleanest end-to-end evidence
and a clear safety story. Make `s_linker23_verify1p` the next live candidate, not
the headline: it cuts the augmentation validation calls in half while leaving the
S21 floor intact. Include `s_linker23_verify1p_all` only as a labelled ablation
for the cost of removing P2 everywhere.

Before selecting a replacement, run S21, `verify`, and `verify1p` side-by-side
for at least three independent all-five-dataset runs with a fixed backend/model
and `OPENAI_SERVICE_TIER=default`; report per-run as well as aggregate F1, F2,
precision, recall, FP, API calls, latency, and failures. Add `verify1p_all` to
that matrix only to quantify its changed-floor tradeoff. The currently tracked
OpenAI router run used a responsive-call trace but did not complete, so it is not
evidence for or against any variant.
