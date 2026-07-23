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
| `s_linker23_verify1p_all` | Extends single-pass gating to both the augmentation and S21's own Phase-4 Framing-C floor. Phase-5 coreference is unchanged. | One fresh all-five-dataset run with GPT-5.6-terra/Flex/explicit no-reasoning: macro P 82.31%, R 97.07%, F1 88.88%, F2 93.56% (186 TP, 52 FP, 9 FN). | A useful cost ablation, not the primary augmentation claim: it changes S21's protected floor, loses `verify`/`verify1p`'s floor-preservation argument, and this N=1 run has a substantial precision regression. |

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
S21 agent-router attempt used a responsive-call trace but did not complete, so it
is not evidence for or against that earlier router variant.

## 2026-07-23 `verify1p_all` whole-dataset run

The complete N=1 result is in
`results/s23_verify1p_all_gpt56terra_flex_noreasoning_central_20260723/` and was
independently scored with `evaluation/mini-src/metrics.py`. Configuration was
`OPENAI_MODEL_NAME=gpt-5.6-terra`, `OPENAI_SERVICE_TIER=flex`,
`OPENAI_ENFORCE_FLEX=1`, and `OPENAI_REASONING_EFFORT=none`. The latter sends
`reasoning_effort=none` and omits `temperature`.

| Dataset | P | R | F1 | F2 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mediastore | 93.94 | 100.00 | 96.88 | 98.73 | 31 | 2 | 0 |
| teastore | 77.14 | 100.00 | 87.10 | 94.41 | 27 | 8 | 0 |
| teammates | 74.67 | 98.25 | 84.85 | 92.41 | 56 | 19 | 1 |
| bigbluebutton | 71.05 | 87.10 | 78.26 | 83.33 | 54 | 22 | 8 |
| jabref | 94.74 | 100.00 | 97.30 | 98.90 | 18 | 1 | 0 |
| **Macro** | **82.31** | **97.07** | **88.88** | **93.56** | **186** | **52** | **9** |

This result is not comparable as a replacement claim against the older GPT-5.4
pilot or paper panel: it changes the model, drops P2 from the S21 floor, and has
only one stochastic run. It does, however, falsify the idea that `verify1p_all`
is an obviously safe default under this requested configuration: its recall/F2
are high, while false positives are concentrated in teammates and BigBlueButton.

### Why teammates and BigBlueButton regress

The tracked result attributes every final link to `entity`, `coreference`, or
`llmrouter`. The two regressions are dominated by the agentic augmentation,
which is substantially broader than the S21 floor:

| Dataset | Router proposed → accepted | FP: router | FP: P1-only entity floor | FP: coreference |
| --- | ---: | ---: | ---: | ---: |
| teammates | 161 → 9 | 8 | 9 | 2 |
| bigbluebutton | 93 → 22 | 14 | 8 | 0 |

The router mistakes generic document vocabulary for component-specific evidence.
On teammates this includes UI/browser flow prose and generic data-layer duties;
on BigBlueButton it maps *frontends/backends* to `HTML5 Server` and Redis-event
prose to `Redis PubSub`. These account for 22 of the 41 FPs across the two sets.
The one-pass floor is the other material cause: it admits code/package-path or
weak alias cases such as `common.datatransfer`, `e2e.util`, `logic.api`, generic
`conversion process`, and diagram/meta prose.

A diagnostic replay of P2 on the original P1 validation batches (same GPT-5.6-
terra/Flex/no-reasoning configuration) rejected observed entity FPs including
TM `Common` at S160/S161 and `GAE Datastore` at S139, plus BBB `Presentation
Conversion` at S83/S84 and `Apps` at S87. It also approved some observed entity
FPs, so restoring P2 alone cannot repair the collapse. The replay is evidence
about the omitted gate, not a replacement end-to-end run: it is a new stochastic
request and cannot prove the exact counterfactual output.

## 2026-07-23 `verify` changed-phase replay

To isolate the cost of returning to the two-pass design, the `verify1p_all`
call traces above seeded a prompt-hash checkpoint cache. The replay reused the
unchanged GPT-5.6-terra S21 acquisition/extraction/coreference prompts, while
new P2 and router/gate prompts used GPT-5.6-terra with Flex and explicit
`reasoning_effort=none`. `scripts/seed-checkpoint-from-traces.py` makes this
procedure reproducible from the tracked full call traces.

| Variant / replay | Macro P | Macro R | Macro F1 | Macro F2 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `verify1p_all` full N=1 | 82.31 | 97.07 | 88.88 | 93.56 | 186 | 52 | 9 |
| `verify` changed-phase replay | 81.58 | 94.81 | 87.11 | 91.37 | 179 | 51 | 16 |

Per dataset, `verify` scores F1 96.88 (mediastore), 77.14 (teastore), 87.50
(teammates), 74.02 (BigBlueButton), and 100.00 (JabRef). `mini-src` reproduces
those CSV scores. The raw output is in
`results/s23_verify_gpt56terra_flex_noreasoning_changed_phase_replay_20260723/`.

This is informative but not a pure P2-only estimate: the S23 proposer/router
calls were not available as complete prompts in the original trace and therefore
were fresh, stochastic calls. The replay demonstrates that restoring both P2
passes did **not** rescue performance under this configuration; it does not
justify the stronger claim that P2 caused the 1.77pp macro-F1 difference.

## 2026-07-23 canonical S21 GPT-5.4 control

The fresh OpenAI control used `gpt-5.4` (provider response model
`gpt-5.4-2026-03-05`), `OPENAI_SERVICE_TIER=flex`,
`OPENAI_ENFORCE_FLEX=1`, and `OPENAI_REASONING_EFFORT=none`. The request-path
test asserts that this configuration sends `reasoning_effort=none` and omits
`temperature`; all five saved call traces identify the returned GPT-5.4 model.

| Dataset | P | R | F1 | F2 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mediastore | 100.00 | 90.32 | 94.92 | 92.11 | 28 | 0 | 3 |
| teastore | 100.00 | 100.00 | 100.00 | 100.00 | 27 | 0 | 0 |
| teammates | 94.23 | 85.96 | 89.91 | 87.50 | 49 | 3 | 8 |
| bigbluebutton | 100.00 | 77.42 | 87.27 | 81.08 | 48 | 0 | 14 |
| jabref | 100.00 | 100.00 | 100.00 | 100.00 | 18 | 0 | 0 |
| **Macro** | **98.85** | **90.74** | **94.42** | **92.14** | **170** | **3** | **25** |

`mini-src` reproduces all five CSV scores in
`results/s21_gpt54_openai_flex_noreasoning_20260723/`. This is N=1 and does not
prove that model choice alone explains the S23 outcomes, but it is a strong
control against the hypothesis that explicit no-reasoning was silently ignored:
canonical S21 produces a high-precision result under the same request policy.
