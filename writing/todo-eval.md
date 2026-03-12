# Research Directions: Better Evaluation for Architectural Trace Link Recovery

Based on deep analysis of the ARDoCo benchmark distribution (5 projects, 195 SAD-SAM links, 40 model elements) and 6 existing empirical studies in `../transarc-emp/`.

---

## Priority Map

| # | Direction | Impact | Feasibility | Novelty | Scope |
|---|---|---|---|---|---|
| 1 | Task reduction to component classification | Very high | High | High | Current paper |
| 2 | Ceiling-aware evaluation | High | High | Medium | Current paper |
| 3 | Sensitivity-based confidence | High | High | High | Current paper |
| 4 | Bimodal recovery analysis | Medium-high | Medium | Medium | Current paper (partial) |
| 5 | Enrollment-invariant metrics | High | Medium | Medium | Current paper |
| 6 | Gold standard completeness audit | Very high | Low | High | Future work |
| 7 | Per-link impact weighting | Medium | High | Medium | Future work |

Directions 1-5 are actionable for the ICSE paper. Directions 6-7 are future work the paper should identify and motivate.

---

## Direction 1: Task Reduction — SAD-CODE Is Component Classification

### The Finding

Oracle-Subset (correct component per sentence, then link all files) achieves F1=0.987. This means transitive SAD-CODE evaluation **reduces** to sentence-level component classification. The file-level expansion is almost entirely mechanical.

### Evidence

- Oracle-Subset F1=0.987 across projects
- 93.7% of all file-level FPs originate from SAD-SAM errors, not SAM-CODE errors
- Keyword-Grep (trivial baseline) matches or beats TransArC on 2/5 projects at file level
- JabRef appears far better than MediaStore (F1=0.943 vs 0.588) despite making 994 wrong file predictions vs 1

### Proposed Formalization

Define SAD-CODE evaluation as two separable levels:

**Level 1 — Component Decision:** Did the system correctly assign sentence S to component C?
```
Evaluate at (sentence, component) granularity
CD_P = |correct (s,c) pairs| / |recovered (s,c) pairs|
CD_R = |correct (s,c) pairs| / |gold (s,c) pairs|
CD_F1 = 2 * CD_P * CD_R / (CD_P + CD_R)
```

**Level 2 — Enrollment Fidelity:** Given correct component assignment, does the code model correctly enumerate the component's files? This is a property of the code model (`.acm`), not the tracing approach.

### Why This Matters

Current evaluation conflates both levels. A system that gets the component right but has an incomplete code model gets penalized as if it made a tracing error. Separating them lets the community measure what actually matters: architectural understanding (Level 1) vs code model completeness (Level 2).

### TODO
- [ ] Compute CD_F1 for TransArC, our pipeline, and single-shot LLM on all 5 projects
- [ ] Compare rankings under CD_F1 vs standard file-level F1 — do they differ?
- [ ] Present as an evaluation contribution: "SAD-CODE evaluation should report component-decision accuracy"

---

## Direction 2: Ceiling-Aware Evaluation

### The Finding

Teammates has 69% structurally impossible FNs (544/790 file-level FNs). The gold standard contains 137 non-transitive links (sentence-to-specific-class, e.g., S87 → `Logic.java`). No transitive `SAD-SAM × SAM-CODE` approach can recover them. The transitive product covers only 91 of 228 gold entries (40%).

Other projects have near-zero structural limits (MediaStore, TeaStore, JabRef all ~100% match between transitive product and gold).

### Evidence

- Teammates ceiling-adjusted recall: 90.2% → 96.7% when impossible FNs excluded (+6.5pp)
- The 137 non-transitive Teammates links are annotation artifacts — direct sentence→file links that skip the architecture model
- This makes cross-project comparison unfair: Teammates looks harder for transitive methods than it actually is

### Proposed Metrics

For any transitive approach, compute:

```
Transitive_Product = {(s, f) : ∃ component c s.t. (s,c) ∈ SAD-SAM_gold AND (c,f) ∈ SAM-CODE_gold}
Ceiling_R = |gold ∩ Transitive_Product| / |gold|
Adjusted_R = |recovered ∩ gold| / |gold ∩ Transitive_Product|
Adjusted_F1 = 2 * P * Adjusted_R / (P + Adjusted_R)
```

Report both raw and ceiling-adjusted metrics. Report `Ceiling_R` per project so readers see which projects have structural limits.

### Broader Implication

Gold standards should annotate whether each link is transitively reachable. This enables fair comparison between transitive approaches (ours, TransArC) and direct approaches (LiSSA, future end-to-end models).

### TODO
- [ ] Compute Ceiling_R for all 5 projects
- [ ] Report ceiling-adjusted F1 for TransArC and our pipeline
- [ ] Add to the evaluation framework section of the paper
- [ ] Flag as a recommendation: "future benchmarks should annotate transitive reachability"

---

## Direction 3: Sensitivity-Based Confidence Reporting

### The Finding

JabRef's implied 95% CI is ±0.005 (based on 8,268 enrolled file points) but actual CI is ±0.074 (based on 38 independent decisions) — a **15x overstatement** of statistical precision. Removing a single raw entry swings F1 by up to 4pp.

### Evidence

- 18,660 total enrolled data points derive from only ~525 annotator decisions (36:1 inflation overall)
- 99.6% of all TPs and 91.1% of all FNs are artifacts of directory enrollment
- Block homogeneity is 96-100%: files within a directory block are all TP or all FN together
- Which system "wins" changes depending on metric granularity (file F1, decision F1, component F1 each give different rankings)

### Proposed Protocol

Any TLR paper using the ARDoCo benchmark (or any benchmark with directory enrollment) should report:

**1. Effective sample size:**
```
N_eff = number of independent annotation decisions (not enrolled file count)
```
For the ARDoCo benchmark: ~525 decisions, not ~18,660 files.

**2. Influence scores:** For each gold entry g, compute:
```
Influence(g) = |F1_full - F1_{without g}|
```
Report the top-5 most influential entries and their F1 impact.

**3. Top-k instability:**
```
Instability_k = max over all subsets of size k of |F1_full - F1_{without subset}|
```
For k=1: how much does removing the single most influential entry change F1?
For k=5: how much does removing the 5 most influential entries change F1?

**4. Standard format:**
> "Effective sample size: 38 decisions (8,268 enrolled files). Top-1 instability: ±4.0pp F1. 95% CI (decision-level): ±7.4pp."

### Why This Is Publishable Standalone

The 15x CI overstatement finding applies to ALL published TLR results on the ARDoCo benchmark. It means confidence intervals reported (or implied) in TransArC, LiSSA, ExArch, and SWATTR papers are dramatically tighter than warranted. This is a methodological contribution applicable beyond our specific approach.

### TODO
- [ ] Compute influence scores for all gold entries across all 5 projects
- [ ] Compute top-1 and top-5 instability for TransArC and our pipeline
- [ ] Report effective sample sizes per project
- [ ] Include as a subsection in the evaluation framework or threats-to-validity
- [ ] Consider whether this alone merits a short paper / tool demo

---

## Direction 4: Bimodal Recovery Analysis

### The Finding

Recovery is strongly bimodal across all projects. Sentences are either perfectly recovered (R=100%) or completely missed (R=0%), with almost nothing in between:

| Project | Perfect (R=100%) | Zero (R=0%) | Partial (0<R<100%) |
|---------|:-:|:-:|:-:|
| MediaStore | 60% | 36% | 4% |
| TeaStore | 70% | 30% | 0% |
| Teammates | 55% | 40% | 4% |
| BigBlueButton | 16% | — | — |
| JabRef | 100% | 0% | 0% |

### Research Questions

1. **What makes zero-recall sentences hard?** Classify each zero-recall sentence into the 4 failure modes (abstraction mismatch, generic terminology, discourse dependence, ambiguous reference). Report the distribution.

2. **Is hardness predictable?** Build a sentence difficulty model using features: sentence length, presence of component names, pronouns, section position, number of potential component matches. Can we predict which sentences will be missed?

3. **Does bimodality hold for our pipeline?** If our pipeline converts some zero-recall sentences to perfect-recall, which failure modes did it solve? This directly validates the failure taxonomy.

### Connection to the Paper

This analysis grounds the failure taxonomy (Section 3 of the paper) in empirical data. Instead of asserting "these are the 4 failure modes," we show: "37 zero-recall sentences in Teammates break down as: 12 discourse dependence, 10 generic terminology, 8 abstraction mismatch, 7 ambiguous reference." That makes the taxonomy data-driven.

### TODO
- [ ] Get zero-recall sentence lists for TransArC on all 5 projects
- [ ] Classify each into failure modes (manual or LLM-assisted)
- [ ] Compare: which zero-recall sentences does our pipeline recover?
- [ ] Map recovered sentences to the pipeline phase that recovered them
- [ ] Include as evidence for the failure taxonomy in Section 3

---

## Direction 5: Enrollment-Invariant Metrics

### The Finding

The same approach gets radically different scores depending on enrollment granularity. MediaStore uses individual files (1.04x inflation). JabRef uses directories (177.8x inflation). Teammates uses 100% directories. This isn't a property of the task — it's a property of how gold standards were annotated.

### Evidence

- TransArC standard F1 average: 0.803. Under PDR (popularity-debiased): 0.684. Under ACF1: also ~0.684.
- BBB: standard F1=0.831, ACF1=0.347 (−0.484 drop)
- JabRef random baseline F1=0.335 (inflated by enrollment density of 41.4%) — high standard F1 is less impressive when normalized

### Proposed Metrics (for the paper)

**CLF1 (Component-Level F1):** Macro-average F1 over model elements. Invariant to enrollment because it operates at component granularity.

**ACF1 (Amplification-Corrected F1):** For transitive evaluation, compute F1 per model element over the enrolled file set, then macro-average. Corrects for the distortion where 3 directory entries expand to 972 files.

**NDG (Normalized Decision Gain):** Normalizes between random and oracle baselines per project. Makes cross-project comparison fair by accounting for project difficulty.
```
NDG = (F1_system - F1_random) / (F1_oracle - F1_random)
```
JabRef random baseline is 0.335 due to dense enrollment; NDG adjusts for this.

### Recommendation for the Paper

Report at minimum:
1. Standard P/R/F1 (for comparability with prior work)
2. CLF1 (for doc→model, enrollment-invariant)
3. ACF1 (for transitive doc→code, enrollment-corrected)
4. Enrollment factors per project (so readers can assess metric reliability)

### TODO
- [ ] Compute CLF1 for TransArC and our pipeline on SAD-SAM (already have aggregate, need per-project)
- [ ] Compute ACF1 for TransArC and our pipeline on SAD-CODE
- [ ] Compute NDG for both approaches
- [ ] Report enrollment factors per project in evaluation setup table
- [ ] Frame as: "We recommend these metrics for future architectural TLR evaluation"

---

## Direction 6: Gold Standard Completeness Audit (Future Work)

### The Finding

47-53% of documentation sentences are "dark matter" — architecturally relevant but have no gold standard code link. A system correctly identifying TeaStore S15 as related to ImageProvider gets **punished** with 64 FPs because the gold standard doesn't include that link.

### Evidence

- 47-53% of sentences are unannotated across projects
- Teammates SAD-SAM gold covers only 22.7% of sentences (45/198), but SAD-CODE gold covers 47.0% (93/198) — a 48-sentence mismatch
- Interfaces never appear in SAD-SAM gold but do appear in SAM-CODE gold, creating "ghost elements" that inflate FN counts
- 808 dormant SAM-CODE errors in Teammates are masked by SAD-SAM failures — they would surface if SAD-SAM improved

### Proposed Research

1. **Dark matter audit:** For each project, identify all sentences that discuss architectural components but lack gold standard links. Classify as: genuinely unlinked (component not implemented), annotation gap (should have a link), or ambiguous.

2. **Dark-matter-adjusted precision:** Exclude FPs where the sentence genuinely discusses the linked component but the gold standard is incomplete.
```
DM_P = |TP| / (|TP| + |FP| - |FP_dark_matter|)
```

3. **Gold standard extension:** Where annotation gaps are found, extend the gold standard and re-evaluate all approaches. This would be a benchmark contribution.

### Why This Is Future Work (Not Current Paper)

Manual annotation is expensive and requires domain expertise. The current paper can **identify** this problem (with the dark matter statistics) and **motivate** the audit, but executing it is a separate effort.

### TODO (for future work section of paper)
- [ ] Compute dark matter statistics per project (sentences with component mentions but no gold link)
- [ ] List 5-10 concrete examples of "punished correct predictions"
- [ ] Frame as: "The benchmark may systematically penalize high-recall approaches"
- [ ] Recommend: "Gold standard revision and completeness audit as community effort"

---

## Direction 7: Per-Link Impact Weighting (Future Work)

### The Finding

A single SAD-SAM link can induce 0 to 972 downstream effects. Current evaluation treats all links as equally valuable, but their downstream cascade effect varies by orders of magnitude.

### Evidence

- JabRef logic S5: 972 induced file-level FPs (47% of ALL JabRef FPs)
- Teammates UI S23: 348 FPs
- Teammates Client has negative net value (−56): its FPs produce more damage (188 FPs) than its TPs contribute (120 TPs)
- BBB FreeSWITCH S59 is a correct SAD-SAM link that produces 0 TPs and 95 FPs — architecturally correct but no code-level gold counterpart

### Proposed Metrics

**Link efficiency:** Ratio of downstream TPs to total downstream links per SAD-SAM decision.
```
Efficiency(s, c) = downstream_TPs(s,c) / (downstream_TPs(s,c) + downstream_FPs(s,c))
```

**Weighted precision:** Each FP weighted by its downstream amplification factor.
```
WP = Σ TP / (Σ TP + Σ_{fp} amplification(fp))
```

**Toxic link detection:** Flag links with negative net value (more downstream FPs than TPs). A system that avoids toxic links is more valuable than one with higher raw recall.

### Why This Is Future Work

Per-link impact weighting requires computing the full transitive cascade for each individual link, which is computationally straightforward but conceptually complex — it changes what "precision" means from "fraction of correct predictions" to "fraction of useful downstream impact."

### TODO (for future work section of paper)
- [ ] Compute link efficiency for all SAD-SAM TPs and FPs across projects
- [ ] Identify all toxic links (negative net value)
- [ ] Frame as: "Evaluation should account for the asymmetric downstream impact of individual trace links"

---

## Integration Plan: What Goes Where in the ICSE Paper

### In the evaluation framework section (Section 5):
- Direction 1: Task reduction finding (motivates component-level evaluation)
- Direction 5: CLF1 and ACF1 metric definitions with enrollment factor reporting

### In the evaluation results (Section 6, RQ2):
- Direction 2: Ceiling-adjusted metrics for Teammates
- Direction 3: Sensitivity analysis (top-1 instability, effective sample size)
- Direction 4: Bimodal analysis results (which zero-recall sentences our pipeline recovers)

### In threats to validity (Section 8):
- Direction 3: Effective sample size disclosure
- Direction 6: Dark matter acknowledgment (limitation of the benchmark)

### In future work / discussion (Section 7):
- Direction 6: Gold standard completeness audit
- Direction 7: Per-link impact weighting
- Direction 4: Full sentence difficulty model (beyond what we can do in this paper)

---

## Quick Wins (< 1 Day Each)

- [ ] Compute CLF1 per project for TransArC and our pipeline on SAD-SAM
- [ ] Compute Ceiling_R per project
- [ ] Compute top-1 instability per project
- [ ] Compute effective sample sizes per project
- [ ] Get zero-recall sentence counts for TransArC per project

These 5 computations take ~1 hour each and provide concrete numbers for the paper.
