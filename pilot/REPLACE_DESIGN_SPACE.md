# Replace-Framing-C design space — can the blocks proposer replace s21's entity extractor?

Run date: 2026-07-03/04. Question: replace (or union) s21's Phase-2 Framing-C
extractor with the blocks proposer, and if that doesn't beat s21, find out why and
test whether a STRONGER VALIDATOR rescues it. Two arms: end-to-end ablation
(`SLinker23Replace`/`SLinker23Union`, now alias-informed) and a candidate-level
validator grid (`pilot/replace_validator_grid.py`) that isolates the validator from
the extractor.

## Arm 1 — end-to-end (alias-informed replace / union vs s21)

Single clean run (one pipeline each — no floor-resample confound), default tier, all
5 datasets:

| dataset | s21 F1 (FP) | replace F1 (FP) | union F1 (FP) |
|---|---|---|---|
| mediastore | 96.7 (0) | 93.1 (0) | 94.9 (0) |
| teastore | 94.1 (0) | **96.2 (0)** | 92.6 (2) |
| teammates | 90.9 (3) | 87.9 (**8**) | 89.1 (4) |
| bigbluebutton | 83.6 (2) | 84.2 (4) | **87.5 (1)** |
| jabref | 100 (0) | 100 (0) | 100 (0) |
| **Macro** | **93.1 (FP 5)** | **92.3 (FP 12)** | **92.8 (FP 7)** |

**Big win over the BLIND replace/union** (RESULTS.md: union macro **88.9 / FP 21**,
teastore union **83.9 / FP 9**): alias injection + the generic-alias frequency filter
cut the FP leak ~3× (union → 92.8 / FP 7; teastore union → 92.6 / FP 2). The alias
work materially fixed the precision blowup that killed the blind replace.

**But neither replace nor union beats s21 (93.1).** The failure is now *localized*,
not a general explosion:
- **replace** (blocks only) under-recalls where blocks misses gold Framing-C uniquely
  catches (mediastore R87 vs 93) AND leaks FP on teammates (FP 8 = 7 entity + 1 coref)
  — an extractor problem union fixes, plus a gate-leak problem.
- **union** (keep both) ≈ s21 (92.8 vs 93.1) but still carries residual gate-FP
  (teammates 4, teastore 2).

## Arm 2 — validator grid (isolate the validator)

Same candidate sets through two validators: `g_s21` (s21's 2-pass evidence gate,
what replace/union use) vs `g_router` (DocModelAgenticRouter VALIDATE/CODE/REJECT →
the s21 gate; the s23_verify stack applied to ALL candidates). Entity-gate only (no
coref), so cleaner than e2e.

Pre-gate recall ceiling (alias-informed): **BLK ≥ FC on every dataset** — teammates
55 vs 47, teastore 27 vs 21, bbb 42 vs 42. Blocks is a recall superset now.

UNION through each validator:

| dataset | g_s21 (F1 / TP / FP) | g_router (F1 / TP / FP) | router effect |
|---|---|---|---|
| teammates | **0.909** / 50 / 3 | 0.871 / 44 / 0 | −6 TP to drop 3 FP → worse |
| teastore | 0.816 / 20 / 2 | **0.851** / 20 / 0 | −0 TP, −2 FP → better |
| bigbluebutton | **0.826** / 45 / 2 | 0.811 / 43 / 1 | −2 TP, −1 FP → worse |

**A stronger validator (the router) does NOT rescue replace.** It removes FP
everywhere but at an inconsistent TP cost: clean win only on teastore; on teammates
it rejects **6 true links** to kill 3 false ones. Mechanism: the router's CODE/REJECT
actions misfire on datasets with code-like component references, discarding real
architecture links. It is too blunt to be a general replacement gate — the plain s21
evidence gate is better on 2 of 3.

Also visible in the grid: BLK through the *plain* s21 gate is already competitive with
FC (teammates BLK 0.893 > FC 0.863; bbb BLK 0.765 ≈ FC 0.769), i.e. the alias-informed
blocks candidates don't need a stronger gate — s21's gate handles them.

## The overriding finding: it's noise-limited

The variants do not separate above the measurement noise at N=1:
- teammates blocks-entity FP was **7** in the e2e replace but **0** in the grid — the
  same extractor+gate, different stochastic sample. FP counts of 0–8 on 57–62 gold
  swing by several between runs.
- `g_s21` vs `g_router` also differ in gate BATCHING (25 vs 8 per batch) → extra
  variance on top of the router's triage.

So the ranking s21 ≈ union ≈ replace sits inside a ±1–2 F1 / ±several FP band. A
single run cannot order them, and tuning a validator against these numbers would be
fitting the noise.

## Verdict — design space exhausted

| axis | options tried | result |
|---|---|---|
| extractor | FC, BLK(blind), BLK(alias), UNION | alias-BLK is a recall superset of FC; blind-BLK is not |
| integration | replace, union | union ≥ replace (replace under-recalls); both ≈ s21, neither beats it |
| validator | s21 2-pass gate, router→gate | router is too blunt (net worse on 2/3); plain gate is best |

**Conclusion.** The alias-informed proposer CAN stand in for Framing-C on recall
(superset) and, unioned and run through s21's own gate, reaches **parity with s21
(92.8 vs 93.1, within noise)** — a large improvement over the blind replace, but NOT
a win. A stronger validator does not help: the router trades away more true links than
false ones except incidentally on teastore. The remaining gap to s21 is below the
run-to-run noise floor, so no replace/union/validator configuration reliably beats
s21, and chasing one at this noise level would be overfitting.

**Practical takeaway:** keep s21's Framing-C as the extractor (canonical, cheapest,
best-or-tied); the alias-informed blocks proposer's proven role is the s23_verify
AUGMENTATION (recall/F2 tilt), not a replacement. If a replacement is ever wanted,
`SLinker23Union` (alias-informed, s21 gate) is the closest — parity, not gain — and
would need repeated-run evaluation to distinguish from s21 at all.

### Untried cells (deliberately not pursued)
A 3-pass / consensus (k-of-n) gate could in principle remove FP more surgically than
the blunt router, but (a) the plain gate already handles alias-BLK candidates, and
(b) with the ranking inside the noise band, any gain would be indistinguishable from
sampling variance without a multi-run harness — i.e. it would be overfitting the
noise floor, not a real design improvement. The honest next step for ANY of these is
a 3–5×-repeat evaluation, not another N=1 validator variant.
