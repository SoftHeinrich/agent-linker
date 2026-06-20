---
quick_id: 260620-s2r
slug: s20-family-results-deep-dive
date: 2026-06-20
type: investigation
status: complete
---

# SUMMARY — s20-family results, deep dive

**Bottom line:** All v2.6.5 s20-family runs are **gpt-5.4** (not Sonnet). On
gpt-5.4 the family clusters at macro **0.895–0.906**, statistically tied within
its own ±1.4pp run-to-run noise. `s20_union` is the marginal best (0.906) and its
entire edge is a **BigBlueButton recall** effect; the Phase-48 "FAIL" (88.9%) is
confirmed a single-run variance artifact. Two real gaps remain: **no Sonnet
re-validation** of the family in v2.6.5, and headline **0.93 is not met on
gpt-5.4** by any variant.

## 1. Backend (decisive context)

Every one of the 44 v2.6.5 run dirs logs `Backend: openai (gpt-5.4)`. There are
**no Sonnet runs** for the s20 family in v2.6.5. So the remediation correctly
re-tested the model where the regression appeared (Phase-48 was a gpt run), but
the dual-model gate's **Sonnet side is unverified** for `s20`/`union`.

## 2. Macro-F1 (gpt-5.4, FULL-5 runs only)

| variant | N | macro | ±sd | min | max |
|---|---|---|---|---|---|
| **s_linker20_union** | 6 | **0.9058** | 0.0144 | 0.888 | 0.929 |
| s_linker20 | 3 | 0.9026 | 0.0051 | 0.898 | 0.910 |
| s_linker17e (ref) | 3 | 0.9014 | 0.0079 | 0.890 | 0.908 |
| s_linker19 (ref) | 3 | 0.8974 | 0.0058 | 0.890 | 0.904 |
| s_linker20_ablpleonasm | 3 | 0.8947 | 0.0025 | 0.891 | 0.897 |

Spread across variant means = 1.1pp; union's own sd alone = 1.4pp. **The ranking
is inside the noise.** Union buys its +0.3pp over s20 by **tripling the variance**
(0.0144 vs 0.0051).

## 3. Where the differences live (per-dataset mean F1, gpt-5.4)

| variant | MS | TS | TM | BBB | JAB |
|---|---|---|---|---|---|
| s19 | 0.955 | 0.963 | 0.857 | 0.789 | 0.923 |
| s17e | 0.920 | 0.963 | 0.845 | 0.779 | **1.000** |
| s20 | 0.967 | **0.987** | 0.833 | 0.785 | 0.941 |
| union | 0.967 | 0.978 | 0.835 | **0.811** | 0.938 |

- **Union = a pure BBB-recall play.** BBB TP 41.7→44.8 (+3.1), FN 20.3→17.2,
  recall 0.672→0.723, F1 0.785→**0.811** (+2.6pp). That single dataset drives the
  whole macro gain, lightly offset by small TS/JAB dips.
- ⚠️ The code comment in `s_linker20_union.py` ("killed ~5 BBB TPs for **0 FP**
  saved") is optimistic: measured is **+3.1 TP and +0.4 FP** (3.3→3.7), not zero.
- BBB is everyone's floor (~0.78–0.81) — the recall bottleneck. TM is everyone's
  precision floor (coref FPs).

## 4. Phase-46-cut bisection (each probe REVERTS one cut; gpt-5.4)

Baselines: s20 TM 0.835, s20 BBB 0.779 (6-run), s20 macro 0.903.

| probe | dataset | cut reverted | F1 | vs base | verdict |
|---|---|---|---|---|---|
| ablrules | TM | COR-01/02 (COREF_RULES body) | 0.820, FP 14.0 | −1.5pp | **cut was GOOD — keep** |
| ablopener | TM | COR-03/04 (coref opener/inline) | 0.848 | +1.3pp | cut mildly **over-trimmed** |
| ablgate | TM | VAL-03 (coref validation focus) | 0.834 | ≈0 | cut was **FREE** |
| abldrop | BBB | AMB-01+DKJ-01 (empty few-shots) | 0.791, FP 2.7 | +1.2pp* | within noise |
| ablpleonasm | macro | 5 generality/jargon cuts | 0.895 | −0.8pp | **cuts were GOOD — keep** |

\*abldrop vs s20's 6-run BBB 0.779. **Takeaway:** the COREF_RULES trim is the one
load-bearing coref cut (reverting it costs FP 11.6→14.0); the opener cut (COR-03/04)
is the only candidate that *might* have over-trimmed (revert helps TM +1.3pp, but
within s20's own 0.021 sd). The 5 generality cuts and the validation-focus cut are
confirmed safe.

## 5. Alias experiment (carries over the 260610-lio task; TM, gpt-5.4)

| variant | TM F1 | FP mean[range] | read |
|---|---|---|---|
| s20 (baseline) | 0.835±0.021 | 11.6 [6–14] | — |
| aliasa (few-shots **CUT**) | **0.860±0.034** | 10.3 **[5–16]** | best mean, **worst variance** — lucky low-FP draws |
| aliasb (**hardware** example) | 0.829±0.009 | 13.7 [12–15] | **stable**, slightly lower + more FP |

Confirms lio's verdict: the **cut spikes coref-FP variance** (range 5–16); the
**hardware rewrite is stable** but trades ~0.6pp TM for benchmark-distance +
predictability. Neither is a clear F1 win on gpt-5.4.

## 6. Reconciling the Phase-48 "FAIL"

Phase-48 `v2.6.4_s_linker20_gpt.log` = a **single** gpt run: macro **88.9%, FP 17**
(an unlucky high-FP TM draw). v2.6.5 N=3 s20 = **0.903** (runs 0.898/0.900/0.910).
The 88.9% sits ~2.6pp below the N=3 mean — fully inside the family's variance
(union alone produced a 0.888 run). **STATE's "FAIL was variance" call is
data-supported.** Single-run gpt conclusions on this family are unreliable.

## 7. Open gaps / recommendations

1. **Sonnet re-validation missing.** The gate is Sonnet AND gpt-5.4; v2.6.5 only
   re-ran gpt-5.4. Union changes recall behavior (intersection→union) — run
   `s_linker20` + `s_linker20_union` N≥3 on Sonnet before any ship.
2. **0.93 is not met on gpt-5.4** by any variant (max 0.906). The re-baselined
   floor (~0.891, N≥3 mean−1) passes, but CLAUDE.md's headline "≥0.93 on both
   models" should be reconciled — the 0.93 was single-run-inflated.
3. **Union ship is a noisy +0.3pp.** Real benefit is BBB recall; cost is 3× macro
   variance + a slightly-wrong code comment. If shipping union, fix the comment to
   "+3.1 TP / +0.4 FP" and prefer N≥3 reporting.
4. **Bisection payoff:** keep COREF_RULES + the 5 generality cuts; reconsider only
   the COR-03/04 opener cut (the one probe that helped on revert).

## Artifacts examined

- `results/v2.6.5/` — 44 gpt-5.4 run dirs (`ablation_*.json` + link CSVs)
- `logs/v2.6.5/*.log` (backend confirmation), `logs/v2.6.4_s_linker20_gpt.log`
- No files changed; no new LLM calls.
