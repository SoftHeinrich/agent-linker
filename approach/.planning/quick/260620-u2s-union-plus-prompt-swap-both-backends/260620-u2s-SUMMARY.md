---
quick_id: 260620-u2s
slug: union-plus-prompt-swap-both-backends
date: 2026-06-20
type: experiment
status: complete
verdict: NEGATIVE — drop the combination, keep s_linker20_union
---

# SUMMARY — union + prompt-swap, both backends

**Ask:** combine `s_linker20_union` (Framing-C 2-pass UNION consensus) with the
"prompt swap" (= aliasb: `ANTECEDENT_ALIAS_RULES` few-shot rewritten to the non-SE
hardware `PowerSupplyUnit` example, lio Candidate B) and run on gpt-5.4 + Sonnet.

**Verdict: NEGATIVE on both backends.** The two changes do **not** stack — union's
BBB-recall gain survives, but the alias prompt swap erodes coref everywhere else,
netting **below union** on gpt-5.4 and an unexceptional Sonnet macro. **Keep
`s_linker20_union`; do not ship the combination.**

## Result — s_linker20_union_aliasb (per-dataset F1)

| backend | N | MS | TS | TM | BBB | JAB | **macro** |
|---|---|---|---|---|---|---|---|
| gpt-5.4 | 3 | 0.935 | 0.981 | 0.829 | 0.806 | 0.907 | **0.8915** ±0.014 |
| sonnet  | 1\* | 0.931 | 0.915 | 0.912 | 0.720 | 1.000 | **0.8957** |

\*Sonnet reconstructed: run1 supplied MS/TS/TM; BBB/JAB came from a separate
completion run (run1's BBB/JAB and runs 2–3 failed — see "Environment" below).

## gpt-5.4 comparison (vs existing v2.6.5 baselines)

| variant | macro | vs union |
|---|---|---|
| s_linker20_union | 0.9058 | — (best) |
| s_linker20 | 0.9026 | −0.3 |
| **s_linker20_union_aliasb (this)** | **0.8915** | **−1.43pp** |

- **Union half works:** BBB recall preserved (TP 44.7 vs union 44.8; F1 0.806 vs 0.811).
- **Swap is pure downside:** TM 0.829 / FP↑13.7 (its known standalone behavior — the
  hardware example is *worse* than the original TaskScheduler one for SE-domain coref),
  plus consistent MS −3.2pp (0.935 every run) and JAB −3.1pp. The PowerSupplyUnit
  example shifts coref resolution globally, not just on the alias edge cases.

## Sonnet read

Macro 0.8957 (N=1) — no same-family Sonnet baseline exists (v2.6.5 only ran gpt-5.4),
so this is standalone, but it sits well below the canonical `s_linker13_min` Claude
0.9506. **BBB = 0.720 (FN 26, recall ~0.58)** is the weak spot; without a Sonnet
`union` baseline we can't isolate whether that's the variant or Sonnet's general BBB
difficulty — which re-confirms the open gap: **the s20 family has no Sonnet validation.**

## Environment finding (blocker for Sonnet at scale)

Sonnet runs go through the `claude` CLI, which **shares quota with the live Claude Code
session**. Under sustained load the CLI begins returning **empty responses**:
teammates alone took 45 min, then after ~70 min run1's BBB/JAB failed (0 links) and
runs 2–3 failed entirely (16 calls each, all empty → "Final: 0"). A fresh, lighter
BBB+JAB run (after cooldown) succeeded with 0 empty responses. **Takeaway: N≥3 full
Sonnet sweeps are not reliably runnable from inside a session** — split into small
dataset batches with cooldowns, or run Sonnet out-of-band.

## Disposition

- `s_linker20_union_aliasb` kept in tree as a registered experimental variant
  (NOT canonical), this negative result documented — so the combination isn't retried.
- Recommended next step (separate from this task): run `s_linker20` + `s_linker20_union`
  N≥3 on **Sonnet** (batched) to close the dual-model gap for the union ship candidate.

## Artifacts

- Variant: `src/llm_sad_sam/linkers/experimental/s_linker20_union_aliasb.py` (+ registration)
- Drivers: `run_union_aliasb_backend.sh`, `run_union_aliasb_both.sh`
- Results: `results/v2.6.5_union_aliasb/{gpt_run1..3, sonnet_run1, sonnet_bbbjab}/`
- Logs: `logs/v2.6.5_union_aliasb/`
