# Literature: views on per-group / tail / coverage metrics, mapped to our data

Two research sweeps (early traceability + data mining/ML), cross-referenced against
the empirical findings in [README.md](README.md) and the bar ablation
(`bar_ablation.py`). Goal: ground the choice of a size-aware metric in prior work
and locate where our metric sits.

## The shared critique (both fields, independently)

Pooled **micro**-F1 weights each *link/instance* equally → the link-richest group
dominates → a system can abandon an entire component yet post high micro-F1 if that
component is small. Every metric below restores a *per-group* view.

- **TLR (Hayes/Dekhtyar/Sundaram, TSE 2006; REJ 2010):** rejected P/R/F1 as
  "primary measures" (the returned *set* only). Two arguments we inherit:
  (1) **recall ≫ precision** — a missed link is an expensive omission, FPs are
  "reviewed away" (→ a coverage/reach metric is in-tradition); (2) **"identical P/R,
  very different list"** — *"Precision and recall do not suffice for evaluating the
  results from the analyst's perspective."* Their per-requirement measures: **Lag**
  (FPs ranked above each true link, per requirement → mean), **MAP** (per-query AP →
  mean = macro). *Gap:* they report the **mean/distribution** per requirement,
  **never a worst-case minimum or a count of fully-abandoned requirements.*
- **DM/ML:** macro-vs-micro (Sokolova & Lapalme 2009; Yang 1999; Manning IIR §13.6
  — *"to get a sense of effectiveness on small classes, compute macroaveraged
  results"*); worst-group (Sagawa GroupDRO 2020 — minimize max-over-groups loss,
  Rawlsian floor); long-tailed tail-accuracy (Liu OLTR 2019; Du & Wu 2023 "No One
  Left Behind" — report **lowest recall**); imbalanced **G-mean** (Kubat & Matwin
  1997 — multiplicative, collapses to 0 on any dead class); **catalog coverage**
  (Herlocker 2004 — *"accuracy in isolation rewards abandoning hard items"*; coverage
  = the whole-catalog denominator); **success@k / hits@k** (IIR — fraction of queries
  with ≥1 relevant hit); **CVaR_α** (Rockafellar–Uryasev 2000; Williamson–Menon 2019
  — tunable dial: α→1 = worst-group, α→0 = average).

## The two camps map onto OUR measured frontier

| camp | metrics | reaction to a dead component | our doc-model result |
|------|---------|------------------------------|----------------------|
| **collapse-to-zero** (Rawlsian / multiplicative) | min-recall, worst-group, G-mean, harmonic, worst-F1 | →0 instantly | **sharp** (spread .5–.6) but **rho .85–.88 vs link-F1 → redundant** |
| **linear** (equal-per-group mean) | macro-recall, macro-F1 | drops ~1/k | middling (rho ~.78–.94) |
| **coverage-count** (completeness gate) | success@k, catalog coverage, **count of abandoned comps** | drops 1/k as a fraction; sharp as a **count** | **independent** (rho .45–.53) but **flat as a fraction** (.92–1.0); **sharp only as a raw count** (0 vs 3) |

So the literature's "sharp" metrics (worst-group, G-mean — Sagawa, Du&Wu) are
exactly the ones our data shows are **redundant with link-F1 on doc-model** (no
enrollment → tail magnitude tracks the pooled score). The literature's
**coverage/completeness** metrics (Herlocker catalog coverage, IIR success@k) are the
**independent** axis — and our finding that the bar is **binary** (a component is
reached or abandoned; `≥1` and `R≥0.25` give identical numbers) is precisely the
success@k / one-error / catalog-coverage notion: *did each group get served
*something*.* Herlocker's complement-of-catalog-coverage is *"the one statistic that
distinguishes 'served every region thinly' from 'served the head and abandoned the
tail.'"* That is our **silent-component-failure count**.

## Where our metric sits (the contribution framing)

- **Silent-component-failure count** = complement of **catalog coverage** (Herlocker
  2004) at component grain = component-level **one-error / 1−success@1** (IIR,
  Schapire–Singer 2000). Independent of link-F1 (rho .53), sharp as a count (0 vs 3 of
  40). It imports a **DM-standard completeness view** into TLR, where it is a genuine
  gap (TLR reported per-requirement *means*, never an abandonment count).
- **worst-component F1 / harmonic** = the **worst-group (Sagawa) / Du&Wu lowest-recall
  / G-mean** camp. Defensible and sharp, but on doc-model redundant with link-F1
  (rho .85+). Keep for doc-code (rho .67/.70, where enrollment makes it bite).
- **bar ablation = an empirical CVaR_α sweep** (Williamson–Menon): tightening the
  recall bar interpolates average→worst-case and trades independence (rho .45→.88) for
  sharpness (spread .08→.25). No bar is both — so the **count at the loosest bar** is
  the principled pick (max independence; the binary failure mode makes the bar
  non-arbitrary).

## Recommended reporting (both fields say: report a spread, never one number)

Pair global micro link-F1 with: (a) **macro / equal-per-component** recall-F1
(Sokolova; the per-requirement-equal-weight tradition of MAP/Lag); (b) a **worst-case**
statistic — min-recall or harmonic (Sagawa / Du&Wu) — **on doc-code**; (c) **coverage
= count of abandoned components** (Herlocker / IIR success@k) — the independent,
legible headline, strongest **on doc-model**. Optionally CVaR_α as the single tunable
knob. The thesis "abandoning a component = failing" is enforced by the collapse
metrics and made legible by the abandonment count.

Citations verified by the two research agents. **HDS 2006 primary now in hand and
verified** (see below); Antoniol original (`Antoniol02a.pdf`) also downloaded.

### HDS 2006 — VERIFIED from primary (`Advancing Candidate Link Generation…pdf`)

Quality-band table (verbatim), confirming the previously "as commonly cited" numbers
**exactly**, and adding a Lag band:

| Measure | Acceptable | Good | Excellent |
|---------|-----------|------|-----------|
| Recall | 60%–69% | 70%–79% | 80%–100% |
| Precision | 20%–29% | 30%–49% | 50%–100% |
| **Lag** | 3–4 | 2–3 | 0–2 |

**Lag** (Def. 1, verbatim): *"Lag of the link (q,d)… is the number of false positive
links (q,d') that have higher relevance score than (q,d),"* averaged over true links.
**Critical for us: Lag is defined over RELEVANCE SCORES (a ranked list)** — our
set-only output has none, so Lag/MAP/success@k are unavailable without adding a score.

## Paywalled primaries to download (user has access)

1. **Antoniol, Canfora, Casazza, De Lucia, Merlo — "Recovering Traceability Links
   between Code and Documentation."** IEEE TSE 28(10):970–983, Oct 2002.
   **DOI 10.1109/TSE.2002.1041053.** Extract: exact P/R definitions in their setting;
   the precision/recall-vs-cutpoint (number-of-documents-retrieved) evaluation; any
   per-code-component recall reporting. Needed to verify the recall-centric / ranked
   framing we cite — and to confirm they did NOT report a worst-case or coverage count.
2. **Hayes, Dekhtyar, Sundaram — "Advancing Candidate Link Generation for Requirements
   Tracing: The Study of Methods."** IEEE TSE 32(1):4–19, Jan 2006.
   **DOI 10.1109/TSE.2006.3.** Extract: (a) exact recall/precision QUALITY BANDS
   (acceptable/good/excellent boundaries — currently "as commonly cited", UNVERIFIED);
   (b) the primary-vs-secondary-measure framing; (c) exact Lag definition; (d) the
   recall≫precision argument verbatim. This is the keystone citation for the
   "P/R insufficient" claim — verify the bands before quoting.

Open-access already in hand: Sundaram/Hayes/Dekhtyar/Holbrook REJ 2010 (selab.netlab
.uky.edu PDF) restates the HDS secondary measures; Borg thesis (arXiv:1602.07633)
summarizes Antoniol. Use these only as backups — the two DOIs above are the primaries.

## NEW set-only metrics (no ranked list) — see newmetrics.py

Constraint: we have predicted link SETS, no scores → all ranking metrics out (Lag,
MAP, success@k, one-error, CVaR). Component-grain SET retrieval (each component once,
volume-independent), recall + precision halves:
  G=gold comps, R+=gold comps reached by ≥1 CORRECT link, P=comps touched.
  SFC = |G|−|R+|  (silent-failure count, FN)   PHC = |P|−|R+|  (phantom-comp count, FP)
  comp_coverage=|R+|/|G|, comp_precision=|R+|/|P|, comp_set_F1=harmonic.

Measured (rho vs link-F1; lower |rho| = more independent):

| metric | rho doc-model | rho doc-code | spread | note |
|--------|--:|--:|--|------|
| **PHC** (phantom count) | **−0.32** | **−0.23** | 0.07→0.67 | **most independent + sharp; precision-side noise link-F1 hides** |
| **SFC** (silent-failure count) | −0.53 | −0.42 | 0→0.6 | independent + sharp; recall-side abandonment |
| comp_set_F1 | +0.36 | +0.20 | flat (.94–1.0) | independent but SATURATED as a rate |
| comp_precision / coverage | +.33/.51 | +.23/.41 | flat | bounded rates → report as COUNTS |
| gmean_recall / worst_recall | +.83/.87 | +.75/.75 | sharp | set-only Sagawa/Du&Wu camp; redundant on doc-model |
| macro_recall | +0.90 | +0.79 | mid | linear penalty (Sokolova) |

Headline: **PHC is genuinely new** — s20u Claude posts high link-F1 (.928) yet touches
0.67 wrong components/project (worst PHC), invisible to link-F1; S21 GPT 0.07. SFC+PHC
form a component-grain P/R pair, both **counts** (escape the .9x saturation), both the
most independent metrics tested. They are the set-only stand-in for the analyst-noise
concern that Lag/selectivity (ranked) addressed.
