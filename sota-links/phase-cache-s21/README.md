# phase-cache-s21 — RQ3/RQ4 reproduction inputs

The minimal run artifacts needed to **recompute RQ3 and RQ4** of the eval study
(`transarc-emp/mini-rq34/`) from scratch. RQ1/RQ2 read the link CSVs elsewhere
in this repo; RQ3/RQ4 instead need the per-linker / per-validator decisions that
only survive in the linker's `phase_cache` pickles, which is what lives here.

These are regenerable agent-linker (`s_linker21`, v2.6.6) outputs — the same
role as the rest of this `sota` data tree. The bulky `llm_logs/` and
`llm_checkpoint/` from each run are intentionally **excluded**; only the files
`rq34.py` / `rq34_rq2.py` actually read are kept.

## Layout

```
phase-cache-s21/<slot>/run{1,2,3}/
    phase_cache/s_linker21/<backend>/<project>/{layer1..4,final}.pkl   # the data
    <project>/ablation_*.json                                         # validate=OK cross-check
```

Four slots (GPT-5.4 = paper body, Claude = appendix mirror; full + no-knowledge):

| Slot | Backend | Knowledge |
|------|---------|-----------|
| `v2.6.6_s21_gpt`           | GPT-5.4 | full     |
| `v2.6.6_s21_sonnet`        | Claude  | full     |
| `v2.6.6_s21_noknow_gpt`    | GPT-5.4 | no-knowledge |
| `v2.6.6_s21_noknow_sonnet` | Claude  | no-knowledge |

Per slot: 75 pickles (3 runs × 5 projects × 5 layers) + 15 ablation JSONs.

## How to use (recompute RQ3/RQ4 from a clone)

`rq34.py` vendors the pickle classes (`_alinker_types` + a custom unpickler) and
is stdlib-only, so unpickling needs **no agent-linker install** — just point the
slot env vars here and supply the gold tree:

```bash
cd transarc-emp                       # the mini branch
SOTA=/path/to/sota/recovered-links    # this repo's clone

# full slots (RQ3 + RQ4 body / appendix)
TRANSARC_BENCHMARK=/path/to/ardoco/.../benchmark \
RQ34_OPENAI_SLOT=$SOTA/phase-cache-s21/v2.6.6_s21_gpt \
RQ34_CLAUDE_SLOT=$SOTA/phase-cache-s21/v2.6.6_s21_sonnet \
  python3 mini-rq34/rq34.py && \
TRANSARC_BENCHMARK=/path/to/ardoco/.../benchmark RQ34_VARIANT=s_linker21 \
RQ34_OPENAI_SLOT=$SOTA/phase-cache-s21/v2.6.6_s21_gpt \
RQ34_CLAUDE_SLOT=$SOTA/phase-cache-s21/v2.6.6_s21_sonnet \
  python3 mini-rq34/rq34_rq2.py

# no-knowledge slots (the RQ4 "No knowledge" row) — see transarc-emp HOWTO §4
```

Sanity: canonical (median-macro) runs are `claude` run1 ≈ 0.9318,
`openai` run3 ≈ 0.9338; `rq34.py` prints `validate=OK (15 checked)`.
