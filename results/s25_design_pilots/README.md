# s25 design pilots — 2026-08-10

Ablations for the workflow points a reviewer would have to read about in the
paper. Every arm is five runs per side (three where noted) on all five projects,
scored against the gold standard, compared with the permutation test in
`approach/pilot/ab_stats.py`.

Nothing here is a fresh five-project end-to-end run. Upstream stages come from
the promoted run `results/s25_cleanup_verify_20260810` — its per-linker
`phase_states` and per-call `llm_logs` — and each pilot re-runs exactly one
stage. `approach/pilot/design_audit.py` sizes each question deterministically
off those checkpoints first, so an arm is only paid for where a decision can
actually move.

The linker the arms ran against is preserved verbatim as
`s_linker25_pre_pilot_baseline.py`; `approach/src/.../s_linker25.py` is that file
plus the outcomes marked ADOPTED below.

## Adopted

| Change | Report | TP | FP | Verdict |
|---|---|---|---|---|
| All three linkers subtract the already-linked set, not just the partial-name one | `sequence_subtraction.json` | +0.8 (p=0.05) | **−6.8 (p=0.01)** | adopted; also removes 57% of the coreference judge's cases |
| One extraction sample instead of two unioned | `extraction_union.json` | −1.2 (p=0.30) | −1.2 (p=0.42) | adopted; neutral, halves extraction cost |
| No alias scope: every discovered alias offered to extraction | `alias_scope.json` | **+3.0 (p=0.01)** | +1.0 (p=0.59) | adopted; deletes a rubric block, a schema field and a paper paragraph |
| No ambiguity map, no model-understanding call, no bundle flag | `ambiguity_map.json` | −0.2 (p=1.00) | +0.8 (p=0.40) | adopted; neutral, deletes a subsection and a prompt |

## Rejected, and why the current design stands

| Candidate change | Report | TP | FP | Outcome |
|---|---|---|---|---|
| Drop the quote request from the full-name and coreference judges | `no_claim_request.json` | **−35.2 (p=0.01)** | +1.4 (p=0.40) | rejected — the quote is load-bearing as a commit-to-text device even though it is never string-matched |
| Instruct contiguity and enforce the quote as a substring | `claim_check.json` | ±0.0 (p=1.00) | +1.6 (p=0.02) | rejected — the check voided **0** verdicts in 25 project-runs; only the added instruction moved anything |
| Second judging pass for the coreference judge | `coref_judge_passes.json` | −0.6 (p=0.40) | −0.8 (p=0.17) | rejected — the asymmetry with the full-name judge is confirmed, not assumed |
| One candidate per judging call instead of 25 (n=3) | `judge_batch.json` | +0.7 (p=0.60) | +0.3 (p=1.00) | rejected — batching does not decide links, so the cheap form stays |
| Halve `CONTEXT_SENTENCES`/`ANCHOR_LIMIT` to 2 (n=3) | `evidence_window.json` | −2.0 (p=0.20) | ±0.0 (p=1.00) | rejected — narrower is mildly worse and saves nothing |

The last two were run to answer objections rather than to change anything: that
a judge's verdict depends on its batch neighbours, and that the window constants
are arbitrary.

## Note on the two batch results

`sequence_subtraction` shows batch composition moving seven false positives,
while `judge_batch` shows 25→1 changing nothing. These are consistent: the
subtraction removes 57% of the coreference judge's cases, a far larger change to
what a batch contains than resizing the full-name judge's batches, and the two
judges are not the same judge.

## Reproduce

```bash
cd approach
../.venv/bin/python pilot/design_audit.py                     # no LLM calls
AB_RUNS=5 OPENAI_API_KEY="$OAI_KEY" LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra OPENAI_REASONING_EFFORT=none \
  ../.venv/bin/python -u pilot/design_pilots.py --pilot sequence
```

`--pilot` takes `sequence`, `alias`, `union`, `corefpass`, `claim`, `noclaim`,
`batch`, `window`, `ambiguity`. The arms that target surfaces since removed
(`claim`, `noclaim`, `ambiguity`) need `s_linker25_pre_pilot_baseline.py` on the
path to run again.

## Composed result

The four adopted changes were then run together, three five-project E2E runs at
`results/s25_postpilot_e2e_r{1,2,3}_20260810` (summary in r1's `SUMMARY.md`):
macro F1 94.7 +/- 0.8, pooled 93.6 +/- 1.2, TP 179.7, FP 9.3 -- against
94.2 / 91.6 / 179 / 17 for the single pre-change run. Recall flat, precision the
gain, which is the composition the individual arms predict.

## Still outstanding

The paper's no-knowledge ablation (`results.tex`, 5.8pp) was measured with the
ambiguity map in place and has not been re-measured.
