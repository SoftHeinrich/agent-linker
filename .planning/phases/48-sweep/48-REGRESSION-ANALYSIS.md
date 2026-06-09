# Phase 48 — s_linker20 Regression Attribution (free analysis, 2026-06-09)

**Question:** which of the 12 Phase-46 cuts caused the gpt-5.4 macro regression (88.9% vs floor 91.3%)?

**Method:** all-free. (1) Establish the true control = `s_linker19` (un-minimized parent; `s_linker20 = s19 + the 12 cuts`). (2) Per-dataset compare. (3) Tally false-positives/false-negatives by link `source` (entity vs coreference) against the ARDoCo gold standards, using the s20 sweep's own per-link CSVs (`results/ablation_results/s_linker20_<ds>_links.csv`).

## Control: s_linker19 is the right baseline, and the cuts ARE guilty

- `s_linker19` gpt-5.4 macro = **0.922** (v2.6.3 RQ1 replay) — and a v2.6.4-era full sweep `logs/s19_clean_20260604_065728.log` gives **0.907** macro. The s17e 92.3% the floor was set against ≈ s19's own score.
- `s_linker20` = s19 + 12 cuts → **0.889**. Since the ONLY delta s19→s20 is the cuts, the ~−2 to −3pp drop is cut-caused, not an s19≠s17e artifact.
- **Caveat:** gpt-5.4 has real run-to-run variance (s19 across runs: MS 0.877–0.949, TM 0.862–0.904, TM FP 6–9). Single-run per-dataset deltas inside that band are not trustworthy on their own.

## Per-dataset, vs s19_clean control

| Dataset | s19 (clean) | s20 | Δ F1 | Signature |
|---|---|---|---|---|
| MediaStore | TP25 FP1 .877 | TP29 FP0 .967 | +9.0 | improved (favorable noise) |
| TeaStore | TP26 FP0 .981 | TP26 FP0 .981 | 0 | unchanged |
| **TeaMmates** | TP50 **FP9** .862 | TP50 **FP13** .833 | **−2.9** | **+4 FP, recall flat → precision loss** |
| **BigBlueButton** | **TP44** FP2 .815 | **TP39** FP3 .750 | **−6.5** | **−5 TP → recall loss** |
| JabRef | TP18 FP0 1.000 | TP16 FP1 .914 | −8.6 | −2 TP +1 FP (tiny dataset) |

## DECISIVE finding — TeaMmates precision loss = coref cuts (100%)

Tallying s20's TeaMmates predictions by `source` against gold:

- **TP:** 45 entity + 5 coreference
- **FP:** **13 — ALL coreference** (0 entity FP). Every FP confidence 1.00.

→ The entire TM regression is attributable to the **coref-family cuts**, which broadened coreference acceptance so ~13 false coref links pass. The false links cluster on generic component names resolved by pronoun/role-reference ("UI", "Logic", "Storage", "Common", "GAE Datastore").

**Suspect cuts (coref site, 5 of the 12):**
- `CUT-COR-01` — COREF_RULES: "pronoun or **role-referential** noun phrase…refers back" → "pronoun or noun phrase that refers back"
- `CUT-COR-02` — COREF_RULES: "section-established topic" → "topic of the surrounding section"
- `CUT-COR-03` — `_prompt_coref` opener rewrite (batched w/ COR-04, commit f8f873f)
- `CUT-COR-04` — `_prompt_coref` inline restatement (same commit)
- `CUT-VAL-03` — **COREF_VALIDATION_FOCUS** ("role-referential phrase…actually" jargon swap). This is the *gate* that should reject bad coref links — prime suspect since 3 coref-validation calls ran yet 13 FP survived.

These 5 were trialed as a batch in Phase 46 (golden tests were byte-equal because they replay cached parsed outputs — blind to live behavior), so the single worst cut is not yet isolated.

## BigBlueButton recall loss (−5 TP) — weaker, partly variance

- s20 BBB: TP 39 (35 entity + 4 coref), FN 23. s19 found 44 → 5 more true links.
- s20 made only 6 coref predictions on BBB (vs 18 on TM) — the coref cuts did NOT balloon here.
- The 5 lost links are mostly entity-recall; BBB recall is the *known* hard problem (s19 BBB R already 0.710) and BBB is the highest-variance dataset (665s). **Plausible suspects:** the drop-by-empty cuts removing guidance — `CUT-DKJ-01` (DOC_KNOWLEDGE_JUDGE_EXAMPLES → "") and `CUT-AMB-01` (AMBIGUITY_FEW_SHOT → "") — OR pure variance. **Not attributable from one run.**

## JabRef (−2 TP, +1 coref FP) — almost certainly variance

18-link dataset; ±2 links = ±8pp. The single FP is coref (consistent with the coref-cut story). Treat as noise.

## Attribution verdict

| Regression | Cause | Confidence |
|---|---|---|
| TeaMmates −2.9pp (+4 FP) | **coref cuts (COR-01/02/03/04, VAL-03)** | **HIGH** (13/13 FP coref, gold-checked) |
| BigBlueButton −6.5pp (−5 TP) | drop-by-empty few-shots (DKJ-01/AMB-01) **or variance** | LOW |
| JabRef −8.6pp | variance | — |

## Recommended confirmation (paid, targeted, cheap)

Single-dataset re-sweeps on a scratch s20 with cuts selectively reverted (each ~$1–4, far below a full $7.71 sweep):

1. **TM, coref-family reverted** (restore COREF_RULES + `_prompt_coref` opener/inline + COREF_VALIDATION_FOCUS to s19 text) → expect TM FP 13→~6–9. Confirms the family. (~$3)
2. **TM bisect:** revert `CUT-VAL-03` alone (the coref gate) first; then COR-01/02 → pinpoint the single worst cut. (~$3 each)
3. **BBB, drop-by-empty reverted** (restore AMBIGUITY_FEW_SHOT + DOC_KNOWLEDGE_JUDGE_EXAMPLES) → re-sweep BBB **2×** to beat variance. (~$4)

GATE-06 must be re-checked on any restored text (the cuts were partly benchmark-generality trims; reverting may reintroduce flagged vocabulary — but COR/VAL cuts were "domain-loaded jargon swaps", block verdict `clean`, so reverting is GATE-06-safe per the minimize log).

**Process fix for v2.6.5:** Phase 46 trusted cached-replay byte-equality as "behavior-preserving". It is not. Any future cut to a behavior-bearing prompt (coref/validation/extraction) needs a live-call canary on ≥1 sensitive dataset (TeaMmates for coref) before acceptance.
