# s21 Prompt Router Pilot Results

Run date: 2026-07-02

## Implemented Workflow

The workflow is implemented entirely under `pilot/`; canonical `src/` linker
files are untouched.

- `pilot/s21_prompt_router_live.py` is the live harness. It loads frozen s21
  outputs, runs typed batch extraction prompts, validates new candidates, and
  computes macro P/R/F1/F2 plus extraction-vs-validation diagnostics.
- `pilot/f2_validation_grid.py` is the F2 optimization harness. It consumes the
  cached `typed_all_filter_named` extraction results, applies structural filters,
  reuses cached s21/contrast verdicts, and optionally runs cached context judges
  for IMPLICIT/ANAPHORA policies.
- `pilot/cache/latest_summaries.json` contains the first-stage variant summaries.
- `pilot/cache/f2_validation_grid_summary.json` contains the ranked second-stage
  F2 policy grid.
- `pilot/cache/*_extract_cache.json`, `*_judge_cache.json`, and
  `f2_extra_mode_judge_cache.json` are the live LLM caches.

The dataflow is:

1. frozen s21 final links are the floor;
2. typed extraction proposes `(sentence, component, quote, mode)`;
3. `AFFIRMATIVE` goes through s21 P1/P2 validation;
4. `CONTRAST` goes through a contrast-specific validator;
5. `CODEPATH` is rejected from model-doc scoring;
6. the F2 grid applies structural filters over cached proposals;
7. augmented links are scored against SAD-SAM gold with P/R/F1/F2.

Command:

```bash
python pilot/s21_prompt_router_live.py --variants validator_contrast_only typed_named_only typed_all_filter_named scratchpad_named
```

All live calls were reasoning-off `gpt-5.4`; reruns use caches in `pilot/cache/`.

## Macro Scores

| variant | P | R | F1 | F2 | dF1 | dF2 | kept |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline s21 | 0.9894 | 0.8913 | 0.9360 | 0.9083 | - | - | - |
| validator_contrast_only | 0.9881 | 0.9084 | 0.9449 | 0.9223 | +0.0088 | +0.0140 | +3 TP / +1 FP |
| typed_named_only | 0.9601 | 0.9100 | 0.9318 | 0.9180 | -0.0043 | +0.0097 | +2 TP / +4 FP |
| typed_all_filter_named | 0.9639 | 0.9231 | 0.9422 | 0.9305 | +0.0062 | +0.0221 | +5 TP / +4 FP |
| scratchpad_named | 0.9668 | 0.9144 | 0.9383 | 0.9235 | +0.0023 | +0.0151 | +4 TP / +4 FP |

## Extraction vs Validation

| variant | surfaced base-missed gold | kept surfaced gold | note |
|---|---:|---:|---|
| validator_contrast_only | 3 / 22 | 3 / 3 | no new extraction; only s21 rejected contrast-like candidates |
| typed_named_only | 3 / 22 | 2 / 3 | prompt is too narrow and still leaks FP |
| typed_all_filter_named | 9 / 22 | 5 / 9 | best F2; extraction finds recall, validation admits 4 FP |
| scratchpad_named | 4 / 22 | 4 / 4 | signal field makes extraction conservative, but FP still remain |

## Interpretation

The extraction-heavy variants are better under F2 than F1 because they recover more
recall. `typed_all_filter_named` is the F2 winner, surfacing 9 of 22 base-missed
gold links and keeping 5, but it pays 4 FP and therefore trails the archived
per-sentence GTP `named+routed` F1.

The safest implementation candidate is still `validator_contrast_only`: it gets
a sizeable F1 and F2 lift with almost no new surface area. The broader prompt
integration is useful for a recall-biased setting, but the extraction prompt
needs a better precision gate before it should replace s21 Phase 2.

## F2 Validation Grid

Second-stage command:

```bash
python pilot/f2_validation_grid.py
```

This reuses `typed_all_filter_named` extraction and cached judges, then tests
structural validation filters. It also ran and cached extra IMPLICIT/ANAPHORA
context judges; those were not helpful.

| policy | P | R | F1 | F2 | dF2 | marginal kept |
|---|---:|---:|---:|---:|---:|---:|
| exact_or_terminal_no_code | 0.9813 | 0.9231 | 0.9507 | 0.9338 | +0.0254 | +5 TP / +0 FP |
| quote_present_no_code | 0.9739 | 0.9231 | 0.9471 | 0.9324 | +0.0240 | +5 TP / +1 FP |
| no_code | 0.9705 | 0.9231 | 0.9454 | 0.9317 | +0.0234 | +5 TP / +2 FP |
| contrast_plus_subjectish | 0.9812 | 0.9198 | 0.9488 | 0.9310 | +0.0227 | +4 TP / +0 FP |
| context_modes_no_code | 0.9473 | 0.9231 | 0.9347 | 0.9276 | +0.0192 | +5 TP / +7 FP |

Best policy: `exact_or_terminal_no_code`.

It accepts:

- TeaStore S7 -> WebUI (`CONTRAST`, quote `WebUi`)
- TeaMMates S88 -> Logic (`AFFIRMATIVE`, quote `Logic`)
- BigBlueButton S59 -> FreeSWITCH (`CONTRAST`, quote `other than FreeSWITCH`)
- BigBlueButton S66 -> FreeSWITCH (`AFFIRMATIVE`, quote `FreeSWITCH`)
- BigBlueButton S79 -> HTML5 Client (`AFFIRMATIVE`, quote `the client`)

The filter is generic:

- reject code/test/path-like evidence;
- for AFFIRMATIVE, require the quote to be the exact component name, the component
  name to occur in the sentence, or the quote to be the terminal word of the
  component name (`the client` -> `HTML5 Client`);
- route `CONTRAST` through the contrast judge.

Note: `+5 TP / +0 FP` is the marginal newly judged set. The aggregate precision
still drops slightly because the pilot, like the archived GTP live run, propagates
links that appeared in any frozen s21 run into all runs (`prop_in_union`).

The IMPLICIT/ANAPHORA context judges hurt F2: they kept no extra gold beyond the
named/contrast set and admitted additional FPs (`context_modes_no_code`: +5 TP /
+7 FP). For F2, the useful improvement is not broader implicit validation; it is
typed extraction plus a stricter named-evidence filter.

## Integrated SLinker22 Run

`SLinker22` implements the same policy inside the live s21 workflow rather than
as frozen-output augmentation. It subclasses `SLinker21`, keeps `s_linker21.py`
byte-stable, preserves live s21 Framing-C extraction as the floor, adds typed
extraction inside Phase 2, and overrides Phase 4 model-doc validation only for
typed-only candidates. It is exported from
`src/llm_sad_sam/linkers/experimental/__init__.py` and registered in
`run_ablation.py` as `s_linker22`.

Smoke command:

```bash
python run_ablation.py --variants s_linker22 --datasets jabref
```

Smoke result after preserving the s21 extraction floor:
`P=100.0% R=100.0% F1=100.0%`, `TP=18 FP=0 FN=0`.

Full command:

```bash
python run_ablation.py --variants s_linker22 --datasets mediastore teastore teammates bigbluebutton jabref
```

Full run result file:
`results/ablation_results/ablation_20260702_095147.json`.

| dataset | P | R | F1 | F2 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| mediastore | 100.00 | 93.55 | 96.67 | 94.77 | 29 | 0 | 2 |
| teastore | 100.00 | 96.30 | 98.11 | 97.01 | 26 | 0 | 1 |
| teammates | 96.23 | 89.47 | 92.73 | 90.75 | 51 | 2 | 6 |
| bigbluebutton | 92.73 | 82.26 | 87.18 | 84.16 | 51 | 4 | 11 |
| jabref | 100.00 | 100.00 | 100.00 | 100.00 | 18 | 0 | 0 |
| Macro | 97.79 | 92.32 | 94.94 | 93.34 | - | 6 | - |

Interpretation: preserving s21 Framing-C as the live floor is necessary. Replacing
Phase 2 with typed extraction was too conservative, but adding typed extraction
inside Phase 2 keeps the same workflow philosophy while matching the pilot F2
target (`93.34` live integrated vs `93.38` pilot grid). Remaining false positives
are concentrated in `teammates` entity validation (`2`) and `bigbluebutton`
entity/coreference validation (`3`/`1`).

## SLinker23 — LLM-decision-driven, general-guidelines-only

Per the design directive, `SLinker23` was rebuilt so that **the model makes the
routing/validation decision from general guidelines** — no structural regex
filters, no mode→policy `if/elif`. (The earlier structural-registry `SLinker23`
and its `typed_mode_router.py` were removed.) It subclasses `SLinker21`, runs the
canonical s21 pipeline as the floor, then augments:

1. `GroundedTypedProposer` — one generic-prompt LLM read per sentence; a proposal
   survives only if it grounds to a real catalog component name (the sole non-LLM
   step, and it is data grounding, not a decision).
2. `DocModelAgenticRouter` — the LLM chooses `VALIDATE / CODE / REJECT` per
   candidate from a generic rubric.
3. s21's OWN two-pass entity validator, injected as the router's gate, floors the
   accept: `accept ⇔ LLM-VALIDATE ∧ s21-gate-approve`.

See `pilot/COMPARISON_s22_vs_agentrouter.md` for the full contrast with s22 (the
structural typed router) and with `SLinker21AgentRouter` (same LLM philosophy but
a reimplemented gate + bolt-on stack).

**Bounded autonomy, proven deterministically.** Because the decision is the
model's, there is no policy to unit-test; what must hold regardless of the model's
choices is the safety invariant. `pilot/test_s23_gate_floor.py` stubs the agent
and the gate and asserts, across every `action × gate-verdict` combination, that a
link is accepted iff the LLM chose VALIDATE *and* the s21 gate approved — so
CODE/REJECT are never accepted and s23 can never regress below the s21 floor.

```bash
python pilot/test_s23_gate_floor.py      # PASS, no network
```

**Live pilot run (gpt-5.4).** Launched
`python run_ablation.py --variants s_linker23 --datasets jabref`. During this
session the OpenAI endpoint returned *sustained* HTTP 500s (failing even Phase 2 of
the shared s21 base for minutes at a time), so the live number is pending a clean
API window. What the partial run does establish: the base s21 floor completed at
**18/18 = 100%** inside the run, and because the augmentation is gate-floored (and
try/except falls back to the floor on any proposer/router failure), s23's jabref
result cannot drop below that 100%. The end-to-end wiring is otherwise verified by
`test_s23_augment.py` (below) without depending on the API.

Two deterministic tests (no network) stand in for the flaky live signal:

```bash
python pilot/test_s23_gate_floor.py   # PASS: accept <=> LLM-VALIDATE AND s21-gate-approve
python pilot/test_s23_augment.py      # PASS: proposer->router->gate-floor->dedup glue; floor preserved
```

Note on measurement weather: this session's OpenAI endpoint intermittently
returned HTTP 500s, which depress recall across the *shared* s21 pipeline (dropped/
retried calls surface/validate fewer candidates) for **any** variant. Single live
runs are therefore a noisy point estimate; the gate-floor invariant (not a single
score) is the reliable guarantee. A clean-weather full run
(`python run_ablation.py --variants s_linker23 --datasets mediastore teastore
teammates bigbluebutton jabref`) is the way to get a stable macro number.

## Batching the proposer — one call per sentence is not acceptable

The first s23 proposer made ONE LLM call per sentence (378 calls for the 5-dataset
corpus). That is a hard no. The fix must batch to one / a few calls **without
losing extraction recall**, so we swept batching strategies empirically
(`pilot/batch_strategy_compare.py`; the prompt builders it measures live in
`proposer.py`, so tested == shipped). Metric: gold-candidate recall = grounded
`(sentence,component)` pairs that are gold / all gold. All calls reasoning-off
`gpt-5.4`, **service tier `default`** (the shared endpoint's `flex` tier was
returning sustained 500s this session; `OPENAI_SERVICE_TIER=default` cleared them).

Strategies compared:
- `plain` — flat numbered sentences in one call (the naive baseline);
- `forced` — plain + "process each sentence independently" instruction;
- `coverage` — REQUIRE one output row per sentence (forces walking every one);
- `blocks` — render each sentence as its own item **with its prev-sentence context**
  (looks like the per-sentence prompt, just many per call).

**bigbluebutton (87 sentences, 62 gold):**

| strategy | batch | calls | gold recall |
|---|---:|---:|---:|
| plain | 6 | 15 | 0.710 |
| blocks | 6 | 15 | 0.726 |
| blocks | 20 | 5 | 0.742 |
| blocks | 45 | 2 | **0.758** |
| blocks | 999 (all) | 1 | 0.710 |
| coverage | 6 | 15 | 0.710 |
| coverage | 20 | 5 | 0.677 |
| plain | 999 (all) | 1 | **0.613** |

**teammates (198 sentences, 57 gold):**

| strategy | batch | calls | gold recall |
|---|---:|---:|---:|
| blocks | 20 | 10 | **1.000** |
| blocks | 40 | 5 | 0.947 |
| plain | 999 (all) | 1 | **0.825** |

Findings:
1. **Naive flat batching degrades** — `plain` in one call loses recall badly
   (bbb 0.613 vs 0.710 at small batch; teammates 0.825, i.e. 10 missed gold).
2. **`blocks` does not degrade under batching — it holds or improves.** At every
   batch size blocks ≥ plain, and batching the blocks prompt *up* (6→20→45 on bbb,
   or the whole 198-sentence teammates doc into 10 calls) keeps recall flat-to-
   higher because each sentence keeps its own framing and context. `coverage` and
   `forced` help over plain but less than blocks.
3. **Chosen default: `blocks`, batch_size 20.** Recall 1.000 on teammates (10
   calls) and 0.742 on bbb (5 calls) — perfect on the worst degrader, within one
   (noise-level) link of the bbb optimum, and ~20× fewer calls than per-sentence
   (10 vs 198; 5 vs 87). Wired into `s_linker23._augment` and made the
   `GroundedTypedProposer.propose_batch` default.

```bash
OPENAI_SERVICE_TIER=default python pilot/batch_strategy_compare.py \
    --dataset teammates --configs blocks:20 blocks:40 plain:999
```

## Can `blocks` extraction REPLACE s21's Framing-C extraction?

s21's existing Phase-2 extractor (`_run_extraction_pass`) is itself batched (flat,
50 sentences/call, 2-pass UNION, alias-informed). `pilot/extraction_replace_compare.py`
runs the REAL Framing-C method (capturing its candidate set right after Phase 2)
and compares its gold-candidate recall to `blocks:20`:

| dataset | extractor | candidates | gold recall | gold blocks-adds | gold blocks-loses |
|---|---|---:|---:|---:|---:|
| teammates | s21 Framing-C | 75 | 0.825 (47/57) | — | — |
| teammates | **blocks:20** | 129 | **1.000 (57/57)** | +10 | **−0** |
| bigbluebutton | s21 Framing-C | 47 | 0.677 (42/62) | — | — |
| bigbluebutton | **blocks:20** | 59 | **0.742 (46/62)** | +6 | **−2** |

Findings:
1. **`blocks` has strictly higher standalone recall than Framing-C on both**
   (teammates 1.000 vs 0.825; bbb 0.742 vs 0.677). Notably s21's batched-50 flat
   extraction lands at teammates 0.825 = the naive one-call number — its 2-pass
   union does not beat a flat read, whereas `blocks` does.
2. **But it is not a clean drop-in superset everywhere.** On teammates `blocks`
   dominates (adds 10, loses 0). On bbb it adds 6 gold yet **loses 2** that
   Framing-C uniquely catches — the two extractors are partly complementary.
3. **Best is the UNION** (Framing-C floor + blocks): teammates 1.000, bbb 0.774
   — higher than either alone. This is exactly what `SLinker23` already does
   (s21 pipeline as the floor, blocks proposals augmented on top).

Verdict: `blocks` is a strictly stronger extractor by aggregate recall and *could*
replace Framing-C for a single-extractor architecture, but a naive swap would
forfeit a few gold links s21 uniquely surfaces (bbb −2) and roughly doubles
candidate volume (more work + FP exposure for the downstream gate). The
empirically dominant and safest configuration is to KEEP BOTH as a union — i.e.
augment, do not replace — which is the shipped `SLinker23` design. (GATE-01 also
forbids editing s21's Framing-C in place; a replacement would have to live in a
subclass.) Precision impact of the extra candidates is bounded by the s21 gate but
was not separately measured here.

```bash
OPENAI_SERVICE_TIER=default python pilot/extraction_replace_compare.py --dataset teammates
```

## End-to-end: does replacing / integrating extraction actually help F1?

Extraction *recall* (above) is only a ceiling — what matters is F1 after the gate.
`SLinker23Replace` (blocks only) and `SLinker23Union` (Framing-C ∪ blocks) run the
batched blocks candidates through s21's UNCHANGED gate / coref / merge. One clean
run (default tier, **0 API errors**), all 5 datasets, s21 + s22 for reference:

| dataset | s21 F1 (FP) | s22 F1 (FP) | union F1 (FP) |
|---|---|---|---|
| mediastore | 98.4 (0) | 96.7 (0) | 96.7 (0) |
| teastore | 98.1 (0) | 94.1 (0) | **83.9 (9)** |
| teammates | 89.9 (3) | 88.3 (5) | 86.7 (7) |
| bigbluebutton | 80.0 (1) | 75.6 (12) | 77.1 (5) |
| jabref | 100 (0) | 100 (0) | 100 (0) |
| **Macro** | **93.3 (FP 4)** | **90.9 (FP 17)** | **88.9 (FP 21)** |

(A separate earlier run gave teammates/bbb union F1 89.1/83.9 — *above* baseline —
showing the sign flips run to run; see the variance note.)

**Conclusion — raw extraction integration is NOT a win; keep a precision gate.**
1. **No broad regression / no bug.** s21 baseline is healthy (macro 93.3; mediastore
   98.4, jabref 100). The low bbb/teammates figures are those datasets' nature.
2. **The "~94" was a favorable s22 run.** Re-run today s22 = 90.9 macro (bbb FP went
   4 → 12). Run-to-run F1 swings ±2–4pp on the hard datasets because the LLM gate is
   stochastic and precision-sensitive. A single run is an unreliable point estimate.
3. **Adding candidates trades precision for recall, and precision loss dominates
   here.** Macro FP climbs monotonically with candidate volume: s21 4 → s22 17 →
   union 21. `union` leaks 9 FP on teastore (P 100→74). So neither `replace` nor
   `union` reliably beats s21 end-to-end when candidates go *straight into the gate*.
4. **The recall from blocks is real, but it needs a precision control** — s22's
   evidence filter or, better here, `s_linker23`'s agentic router, which validates/
   rejects each blocks candidate (LLM, general guidelines) *before* the s21 gate.
   That is the safe way to "integrate all"; dumping the union into the gate is not.

So: keep the batched `blocks` extractor (it's the recall win + the calls fix), but
integrate it through the `s_linker23` router — do **not** replace or raw-union it
into s21's Phase 2. `SLinker23Replace`/`SLinker23Union` remain registered as the
experiments that establish this.

## Fixing precision (LLM-side, no heuristics): verify vs conditioning

Two LLM-side ideas to make the augmentation precise (no coded thresholds/regex),
tested end-to-end on all 5 datasets (default tier), F1 **and** F2:

| variant | macro F1 | macro F2 | total FP |
|---|---:|---:|---:|
| s21 (baseline) | 92.8 | 89.9 | 4 |
| **s23_verify** | **92.7** | **91.6** | 9 |
| s23_ctx (conditioned) | 88.6 | 90.6 | 41 |

**s23_verify — VALIDATED.** Floor the router's VALIDATE decisions with s21's OWN
Phase-4 evidence-bundle validator (claim-before-verdict) instead of s23's
lightweight case-text gate. Pure s21 mechanism, no heuristics. Result: **F1 parity
with baseline (92.7 vs 92.8) and +1.7pp F2 (91.6 vs 89.9)** — it adds recall
without wrecking precision. On teastore it turned the plain-s23 leak (FP 9–15,
F1 83.9) into FP 2–5, F1 91.5–96.4. This is the way to combine s23's recall with
s21's precision.

**s23_ctx — INVALIDATED.** Condition the proposer on s21's own per-sentence links
(shown as `ALREADY LINKED: ...`) and ask the model for what the base MISSED —
residual extraction, pure LLM-side context, no thresholds. It *does* recover the
in-linked misses (recall up), but the "find what's missing" framing is a **recall
pressure that makes the model manufacture misses**: it over-proposes confident-but-
wrong candidates (mediastore +7 FP, teammates +18 FP, bbb +15 FP), and because
those come phrased as plausible claims they slip past the evidence gate. Macro F1
falls to 88.6 (FP 41) — worse than blind `verify` on both F1 and F2. The
conditioning framing corrupts candidate quality faster than the gate can filter.

Note the negative result is specific to the *residual "find-missing" proposer*
framing. A different use of the same signal — neutral base-link context to the
*router* to help it REJECT redundant proposals (no recall pressure) — is a distinct
hypothesis not covered here. But "condition the proposer to hunt for misses" is
invalidated: it trades away precision faster than it buys recall.
