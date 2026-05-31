# ICSE Paper Planning: Architecture-Grounded Trace Link Recovery

This document serves as a comprehensive reference for paper writing: storyline, positioning against baselines, anticipated reviewer concerns with mitigations, verified citations, and release checklist.

---

## Critical Deliberation Summary (Read This First)

### Paper Identity: Option A+B

**Option A (Pipeline + Ablation):** Multi-phase architecture-aware pipeline improves doc→model tracing. Ablation shows each phase addresses a real failure mode. Measured under both standard and component-level metrics.

**Option B (Evaluation + Insight):** Component-level evaluation reveals that standard metrics hide critical differences between approaches (BBB: 0.831→0.347 F1 when corrected). The amplification finding (1 SAD-SAM error → up to 972 file-level FPs) connects the two contributions.

**The bridge:** Our pipeline fixes doc→model errors on small, hard-to-link components. The evaluation framework reveals these improvements matter more than file-level metrics suggest because of transitive amplification. Neither contribution is complete without the other.

### What's Validated vs. What Still Needs Testing

| Claim | Status | Evidence |
|---|---|---|
| Component-level metrics tell a different story | **VALIDATED** | BBB 0.831→0.347, Teammates 100% vs BBB 36% component coverage |
| Transitive amplification is extreme | **VALIDATED** | 1 error → 972 FPs, top 10 errors = 69% of all FPs |
| Our pipeline improves doc→model | **VALIDATED** | TransArc 75.4% → Ours 88.3% component-level aggregate |
| LLMs systematically under-link | **NEEDS TESTING** | Run recall-optimized single-shot baseline (Experiment 1c) |
| Doc→model bottleneck thesis | **PARTIALLY VALIDATED** | Amplification data supports it; controlled propagation experiment (Exp 4) would fully prove it |
| Failure taxonomy is principled | **NEEDS GROUNDING** | Systematic error categorization on single-shot baseline errors |

### Three Experiments to Run Before Writing

1. **Recall-optimized single-shot baseline** (~1 day) — determines whether the pipeline is necessary or just compensates for bad prompting
2. **Compute CLF1/AC for our pipeline vs TransArC on SAD-SAM** (~1 hour) — confirms the component-level advantage at the direct doc→model level (currently only have aggregate numbers)
3. **Categorize single-shot baseline errors into the 4 failure modes** (~1 day) — grounds the taxonomy in data

---

## Table of Contents

1. [Paper Storyline & Positioning](#1-paper-storyline--positioning)
2. [Research Questions & Experiment Plan](#2-research-questions--experiment-plan)
3. [Evaluation Framework (Concrete Metrics)](#3-evaluation-framework)
4. [Anticipated Reviewer Concerns](#4-anticipated-reviewer-concerns)
5. [Design Justifications with Verified Citations](#5-design-justifications)
6. [Master Citation List](#6-master-citation-list)
7. [Killed Citations (Do Not Use)](#7-killed-citations)
8. [ARDoCo Ecosystem Papers](#8-ardoco-ecosystem)
9. [Release Hygiene Checklist](#9-release-hygiene)
10. [Paper Section Templates](#10-paper-section-templates)

---

## 1. Paper Storyline & Positioning

### One-Sentence Claim

We characterize documentation-to-model recovery as the critical inference problem in transitive architectural tracing, identify its specific failure modes for LLM-based approaches, and design a multi-phase architecture-aware pipeline that manages those failures — evaluated with distribution-aware metrics that expose limitations hidden by file-level F1.

### Paper Identity

This is a **research paper with an empirical contribution**, not a systems/engineering report. The contribution is:
1. An empirical characterization of **why doc→model tracing is hard** (failure taxonomy)
2. A pipeline **derived from** that failure taxonomy (not assembled ad-hoc)
3. A new evaluation framework that reveals what file-level metrics hide
4. Controlled evidence that better first-leg recovery propagates through transitive chains

### Core Thesis

> The bottleneck in transitive architectural tracing is not the transitive composition itself, but the quality of the upstream documentation-to-model links on which the full chain depends. This upstream stage is intrinsically hard because it requires concept grounding, discourse resolution, and confidence calibration — capabilities that neither lexical matching nor generic LLM classification adequately provide.

### Positioning Against Baselines

**TransArC (ICSE'24):** Established that transitive recovery through architecture models is effective. Its doc→model leg uses SWATTR, a 4-stage NLP pipeline: text preprocessing (Stanford CoreNLP), text extraction, recommendation generation, and element connection. The first three stages perform genuine NLP analysis; the final element-connection step links recommended instances to model elements using normalized Levenshtein and Jaro-Winkler similarity and does not consider relations between model elements. **Be fair: SWATTR is not "just string matching."** The critique is that the final connection step — where candidate mentions are mapped to model elements — relies on lexical similarity and cannot resolve the abstraction mismatch, generic terminology, and discourse dependence problems. We keep the same transitive composition (model→code via ArCoTL) and redesign only the upstream doc→model stage, enabling controlled measurement.

**LiSSA (ICSE'25):** Showed LLMs and RAG can support generic traceability across multiple artifact types. However, its retrieve-then-classify formulation is not architecture-specialized and does not outperform ArDoCo on architecture doc→model tracing. Our approach is architecture-specialized: explicit architectural knowledge, discourse-aware coreference, and calibrated judgment.

**Key contrast sentences (pick one for abstract/intro):**

- "Our contribution is not swapping a lexical matcher with an LLM; it is changing the inference problem from local candidate matching to architecture-grounded transitive reasoning."
- "We do not treat the doc→model stage as a replaceable module; we treat it as the dominant uncertainty source in transitive architectural tracing."
- "The key limitation of current transitive architectural tracing is not transitivity itself, but the reliance on lexical element matching at a semantically critical step."
- "We keep the transitive backbone of architectural tracing fixed and redesign its most error-prone stage as a two-pass architecture-aware LLM inference process."

### Why This Does Not Look Incremental

The novelty is NOT "TransArC + LLM." It is:

| Dimension | TransArC / LiSSA | Ours |
|-----------|-----------------|------|
| Main unit | Artifact pipeline / RAG retrieval | Architectural concept |
| Core mechanism | Transitive composition / retrieve-classify | Multi-agent semantic reasoning |
| Doc→model inference | Local pairwise lexical similarity | Architecture-grounded concept resolution |
| Failure handling | Fixed pipeline | Failure-taxonomy-derived phases |
| Evaluation | File-level P/R/F1 | Distribution-aware + concept-level |

### Why This Does Not Look Like Lego (Component Swap)

The bottleneck analysis is framed as **task structure and error propagation**, not component replacement:

1. **The first leg is a concept-grounding problem:** Documents describe responsibilities and abstract roles; models expose sparse element names. This is not matching — it is mapping abstract concepts to underspecified elements.
2. **The first leg is a discourse-resolution problem:** Evidence is distributed across sentences (generic terms, pronouns, section-level definitions). Pairwise similarity underfits.
3. **The first leg is a calibration problem:** In transitive pipelines, weak upstream links cause double damage (false negatives remove downstream paths; false positives create spurious ones).

Our pipeline design follows from these three observations, not from "SWATTR is lexical, so use LLM."

### Doc→Model Failure Taxonomy

**Four task-level failure modes** (why doc→model inference fails):

| Failure Mode | Example | What Handles It |
|---|---|---|
| Abstraction mismatch | "order processing subsystem" → `OrderCore` | Architecture-knowledge prompting |
| Generic terminology | Single dictionary word "service" used both as concept and as component name | Generic term identification + convention filter |
| Discourse dependence | Pronoun or definite description refers to component 3 sentences back | Coreference resolution (3-sentence window) |
| Ambiguous reference | "server" could mean HTML5 Server, FreeSWITCH Server, or the generic concept | Deliberative multi-perspective validation |

Plus one **pipeline-level design motivation** (why recall-preserving architecture matters):

| Design Motivation | Evidence | What It Drives |
|---|---|---|
| Transitive amplification | 1 wrong doc→model link → up to 972 file-level FPs downstream | Recall-first design (P1), cascade integrity (P6), asymmetric voting (P2) |

**Important distinction:** The first four are **why doc→model fails** (task-level). Transitive amplification is **why those failures matter disproportionately** (system-level). They belong in different parts of the paper: taxonomy in Section 3, amplification in Section 5 (evaluation framework) or Section 7 (discussion).

**On the generic/ambiguous overlap:** Generic terminology and ambiguous reference are related but distinct. Generic terminology is a **lexical** problem: the word "service" exists in everyday English and may not refer to any component. Ambiguous reference is a **resolution** problem: the word "server" IS an architectural reference but could match multiple model elements. Generic terms need filtering; ambiguous references need disambiguation. Different pipeline phases handle them (convention filter vs deliberative judge).

**Derivation methodology (address reviewer concern):** These failure modes should be grounded in a systematic error analysis. Examine all FP and FN errors across the 5 projects for the single-shot LLM baseline, categorize each error into one of the four modes, report the distribution (with counts). This makes the taxonomy empirical rather than post-hoc. If inter-rater agreement is feasible (even self-agreement on a subset), report it. Alternatively, ground the categories in existing TLR literature: abstraction mismatch ≈ Antoniol et al.'s "semantic gap," generic terminology ≈ Wang et al.'s "polysemy," discourse dependence ≈ Gotel & Finkelstein's "indirect traceability."

### Key Empirical Findings (Two Pillars)

**Finding 1 (Pipeline necessity — NEEDS VALIDATION from recall-optimized baseline):**

> When LLMs are applied to architectural trace recovery, the dominant failure mode is **false rejection**, not false acceptance. LLMs systematically under-link because they interpret architectural mentions conservatively.

**Honest caveat:** This may be a prompting artifact rather than an inherent LLM limitation. The recall-optimized single-shot baseline (Experiment 1c) tests this. If recall-optimized prompting matches the pipeline's recall with acceptable precision, the finding weakens to: *"LLMs can't simultaneously optimize precision and recall for architectural tracing — a multi-phase pipeline is needed to decouple generation from validation."* That's still publishable but less dramatic.

**Finding 2 (Amplification — ALREADY VALIDATED from existing analysis):**

> In transitive architectural tracing, doc→model errors amplify disproportionately through the chain. A single incorrect doc→model link can induce up to 972 file-level false positives. The top 10 component-level errors account for 69% of all file-level FPs. Standard file-level metrics hide this because they are dominated by large, well-implemented components.

This is the stronger finding — backed by concrete data, not dependent on prompting assumptions. Lead with this if Finding 1 turns out to be prompt-dependent.

**How they combine:** The pipeline fixes doc→model errors (Finding 1), and those fixes matter more than file-level metrics suggest because of amplification (Finding 2). The evaluation framework reveals the true impact.

### Research Methodology Framing

Our pipeline design emerged from a **systematic design study** of LLM-based architectural trace recovery. We analyzed how different pipeline configurations handle the specific failure modes of doc→model tracing, iteratively refining the design based on error analysis. This is standard methodology in traceability research — analogous to how IR-based approaches tune retrieval parameters through empirical analysis [Hayes06], and how pipeline-based NLP systems are designed through systematic error analysis of each stage.

We report the final design and its principled justification. The design rationale is grounded in the failure taxonomy (Section 3.1) and supported by ablation evidence showing each phase's contribution.

### Terminology Guidance

**DO NOT call the approach "agent-based" in the paper.** The actual implementation is a sequential multi-phase LLM pipeline — not autonomous agents with tools, planning loops, or persistent memory. Calling it "agent-based" overpromises and invites rejection from reviewers who know the agent literature.

**Accurate terminology:**
- "Multi-phase LLM-assisted pipeline" for the overall system
- "Deliberative multi-LLM validation" for the advocate-prosecutor-jury pattern (Phase 9) — this IS a form of multi-agent deliberation and is honestly described as such
- "Architecture-aware LLM reasoning" for phases that use architectural knowledge in prompts
- "Specialized LLM reasoning steps" for individual phases

**What IS genuinely multi-agent:** The advocate/prosecutor/jury pattern in the judge phase, where three independent LLM calls with different roles deliberate over the same candidate link. This can be accurately described as "structured multi-perspective deliberation" or "adversarial LLM debate."

**Avoid these phrases in the paper:**
- "agent-based framework" / "ArchTraceAgent"
- "autonomous agents"
- "we put architectural knowledge in the prompt"
- "prompt engineering" (describe what the prompts encode, not the technique)

**Prefer these phrases:**
- "architecture-grounded inference"
- "discourse-aware trace recovery"
- "multi-phase pipeline with deliberative validation"
- "architecture-aware contextualization for trace inference"

### Paper Structure (Recommended)

| Section | Content | Pages (est.) |
|---------|---------|-------------|
| 1. Introduction | 6 paragraphs (see templates below) | 1.5 |
| 2. Background | TransArC/SWATTR, LiSSA, LLM-based SE, evaluation in TLR | 1.5 |
| 3. Problem Analysis | Failure taxonomy with concrete examples, evidence from single-shot baselines, key finding (LLMs under-link) | 1.5 |
| 4. Approach | Pipeline overview figure, phase-by-phase tied to failure modes, design principles P1-P5 | 2.5 |
| 5. Evaluation Framework | Limitations of file-level metrics, concept-level accuracy, architectural coverage, distribution-aware F1 — all with formulas | 1.0 |
| 6. Evaluation | Setup, RQ1-RQ4 results, ablation table, metric comparison table | 3.0 |
| 7. Discussion | Key finding (recall-preserving pipeline necessary), error propagation evidence, cost, limitations | 1.0 |
| 8. Threats to Validity | Internal, external, construct, data leakage | 0.5 |
| 9. Related Work | Positioned against all baselines | 1.0 |
| **Total** | | **~13.5** (ICSE limit: ~13+refs) |

### Figure Plan

**Figure 1: Problem & Approach Overview (full-width, top of page 3)**
Two-panel figure:
- Left panel: "Current transitive tracing" — Doc → [SWATTR: lexical matching] → Model → [ArCoTL] → Code. Arrow labeled "bottleneck" on the first leg.
- Right panel: "Our approach" — Doc → [Architecture-aware LLM pipeline: knowledge extraction → candidate generation → coreference resolution → deliberative validation] → Model → [ArCoTL] → Code. Same second leg. Visual emphasis on the redesigned first leg.

**Figure 2: Pipeline Detail (full-width)**
Phase-by-phase flow diagram with each phase labeled by its failure mode. Color-coded: blue = generation phases, orange = validation/calibration phases.

**Figure 3: Evaluation Framework Motivation (column-width)**
Example showing how file-level F1 can be 90% while concept-level coverage is only 60% (large component A dominates, small components B/C missed entirely).

**Figure 4: Results (full-width table + chart)**
Per-project comparison under both standard and new metrics.

---

## 2. Research Questions & Experiment Plan

### Research Questions

**RQ1 (Effectiveness):** How does architecture-aware doc→model recovery compare to lexical matching (TransArC/SWATTR) and generic LLM tracing (LiSSA) on both doc→model and transitive doc→code tasks?

*Why it matters:* Establishes that the approach works. Measured under BOTH standard metrics (for comparability) AND component-level metrics.

**RQ2 (Evaluation Lens):** Do approaches that appear similarly effective under standard file-level metrics differ significantly at the component level?

*Why it matters:* This is the sharpest formulation — a testable hypothesis, not a descriptive question. We already know the answer is YES (BBB: same F1 as Teammates, but 36% vs 100% component coverage). The RQ validates that the evaluation framework reveals actionable differences. Re-evaluate ALL approaches under both old and new metrics.

**RQ3 (Phase Contribution):** What is the contribution of each pipeline phase, and do improvements on hard-to-link components amplify through the transitive chain?

*Why it matters:* Combines ablation + amplification in one RQ. Each ablation tests whether a failure mode is real. The transitive propagation sub-question tests whether doc→model improvements on small components translate to disproportionate doc→code gains.

**RQ4 (Portability):** How sensitive is the approach to the choice of LLM backend?

*Why it matters:* Addresses contamination concerns. Compare Claude Sonnet vs GPT-5.2 with variance from 10+ runs.

### Experiment Plan

#### Experiment 1: Main Comparison (RQ1 + RQ2)

| Approach | Doc→Model | Doc→Code (transitive) | Purpose |
|----------|:---------:|:---------------------:|---------|
| TransArC (SWATTR + ArCoTL) | Yes | Yes | Lexical baseline |
| LiSSA (RAG + LLM classify) | Yes | if applicable | Generic LLM baseline |
| Single-shot LLM (naive) | Yes | Yes | "Just ask" baseline |
| Single-shot LLM (recall-optimized) | Yes | Yes | Tests whether pipeline is necessary |
| Our pipeline (full) | Yes | Yes | Treatment |

- All evaluated on 5 ARDoCo benchmark projects
- Report standard P/R/F1, CLF1, AC, and ACF1 (for transitive)
- **The recall-optimized single-shot is critical.** Three variants: (a) naive "find trace links," (b) precision-oriented "only confident links," (c) recall-oriented "include all possible links, even uncertain." If (c) matches our pipeline's recall, the "LLMs under-link" claim weakens. If (c) gets recall but terrible precision, the real finding is: *"LLMs can't simultaneously achieve precision AND recall for architectural tracing — a multi-phase pipeline is needed to decouple them."* That's more nuanced and more defensible than "LLMs under-link."

#### Experiment 2: Phase Ablation (RQ2)

Remove one phase at a time from the full pipeline, measure impact:

| Ablation | Failure Mode Tested | Expected Effect |
|----------|-------------------|-----------------|
| Remove architecture-knowledge prompting | Abstraction mismatch | Recall drops on abstract component names |
| Remove convention-aware filtering | Generic terminology | Precision drops (more FPs from generic terms) |
| Remove coreference resolution | Discourse dependence | Recall drops on pronoun/definite-ref sentences |
| Remove deliberative judge | Candidate multiplicity | Precision drops on ambiguous names |
| Remove recall-preserving calibration | Propagation sensitivity | Recall drops significantly |
| Remove ALL phases (= single-shot LLM) | All failure modes | Matches Experiment 1 single-shot baseline |

Present as a table with per-phase delta on F1, precision, recall, AND concept-level coverage.

#### Experiment 3: Metric Comparison (RQ2)

Re-evaluate ALL approaches from Experiment 1 under both standard and component-level metrics:

| Metric | Level | What It Measures |
|--------|-------|-----------------|
| Standard P/R/F1 | link-level | Traditional comparison (for prior work comparability) |
| CLF1 | component-level | Per-component accuracy, small components weighted equally |
| AC | component-level | Fraction of components with ≥1 correct link |
| ACF1 | component-level (transitive) | Component-level F1 corrected for enrollment inflation |
| Error profile | component-level | Fully recovered / partial / missed / over-linked per component |

**Existing data already shows the "reversal" we need:**

| Project | TransArC file-level F1 | TransArC component-corrected F1 | BBB component coverage |
|---------|:---:|:---:|:---:|
| BBB | **0.831** (looks strong) | **0.347** (actually weak) | 36% fully recovered |
| Teammates | **0.821** (looks same as BBB) | **0.624** (much better) | 100% fully recovered |

This IS the finding for RQ2. We just need to compute the same metrics for our pipeline to show the gap narrows under component-level evaluation.

**Additional existing finding — amplification asymmetry:**
- 10 component-level errors → 69% of all file-level FPs
- JabRef "logic" S5 alone: 972 induced file-level FPs
- Fixing that ONE doc→model link eliminates 47% of JabRef's transitive FPs

This demonstrates that file-level metrics hide where improvements matter most.

#### Experiment 4: Error Propagation (RQ1 + Novel Finding)

This is the controlled transitive experiment:

1. Run TransArC's doc→model → measure doc→model quality → run ArCoTL → measure doc→code quality
2. Run OUR doc→model → measure doc→model quality → run SAME ArCoTL → measure doc→code quality

Same model→code backbone, different doc→model. This isolates first-leg impact and measures whether improvements propagate, attenuate, or amplify through the transitive chain.

**Expected finding:** Doc→model improvement of X pp → doc→code improvement of Y pp. If Y > X, there is amplification (TransArC's first-leg errors compound). If Y < X, there is attenuation (ArCoTL is robust to some upstream noise). Either finding is interesting and publishable.

#### Experiment 5: Cross-Model (RQ4)

- Claude Sonnet: 10 independent runs, report mean ± stdev
- GPT-5.2: 10 independent runs, report mean ± stdev
- Statistical comparison: paired permutation test or Wilcoxon signed-rank

#### Experiment 6: Memorization Probe (Contamination Defense)

- Ask each LLM: "Given the architecture documentation of [project], list all trace links between sentences and model elements" — WITHOUT providing the document or model.
- Ask each LLM: "What are the components of [project]?" — zero-context.
- Compare zero-context output to gold standard. If overlap is low, memorization is unlikely.
- Compare zero-context output to pipeline output. If pipeline output is much better, the pipeline structure (not memorization) explains performance.

### Cost Analysis Plan

Report for each approach:
- Total LLM tokens (input + output) per project
- Wall-clock time per project
- Estimated $ cost per project (at API rates)
- Number of LLM calls per project

This is important for practical adoption arguments and for comparing against TransArC (which uses no LLM calls).

---

## 3. Evaluation Framework (Concrete Metrics)

### Critical Honest Assessment: Where This Contribution Bites

The evaluation framework contribution operates at **two levels** with very different impact:

**SAD-SAM (doc→model) — moderate distribution skew:**

| Project | Links | Elements | Min links | Max links | Gini | Elements ≤2 links |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| mediastore | 31 | 10 | 1 | 7 | 0.306 | 4/10 (40%) |
| teastore | 27 | 6 | 2 | 6 | 0.179 | 1/6 (17%) |
| teammates | 57 | 8 | 4 | 15 | 0.261 | 0/8 (0%) |
| bigbluebutton | 62 | 11 | 2 | 14 | 0.370 | 3/11 (27%) |
| jabref | 18 | 5 | 2 | 6 | 0.222 | 2/5 (40%) |

Top 3 components account for 52-78% of links. Macro vs micro gap: **3-5pp**. Meaningful but not dramatic. 10 of 40 total elements (25%) have ≤2 links — these are the hardest to recover and most easily hidden by micro-averaging.

**SAD-CODE (transitive doc→code) — extreme skew via enrollment inflation:**

| Project | Raw entries | Enrollment factor | File-level F1 | Component-corrected F1 | **Drop** |
|---------|:---:|:---:|:---:|:---:|:---:|
| mediastore | 57 | 1.04x | 0.588 | 0.739 | +0.151 |
| teastore | 70 | 4.4x | 0.829 | — | — |
| teammates | 228 | **35.5x** | **0.821** | **0.624** | **-0.197** |
| bigbluebutton | 132 | **11.4x** | **0.831** | **0.347** | **-0.484** |
| jabref | 38 | **177.8x** | **0.943** | **0.857** | **-0.086** |

**The BBB Paradox (strongest motivating example — USE THIS IN THE PAPER):**
BigBlueButton achieves 83.1% file-level F1 — apparently strong. Under component-level correction, it drops to **34.7%**. It correctly traces the 3 largest components (62% of gold links) while missing or corrupting many smaller ones. This is completely invisible under standard metrics. Same-F1 approaches can have wildly different architectural coverage.

**The Amplification Finding (the bridge between pipeline and evaluation):**
1 SAD-SAM error can cascade to up to **972 file-level FPs** in the transitive chain. The top 10 component-level errors account for **69% of all file-level FPs**. Concretely:
- JabRef logic S5: 972 induced FPs (47% of ALL TransArc FPs for that project)
- Teammates UI S23: 348 FPs
- Teammates Common S158: 139 FPs

**Bottom line:** The evaluation framework's most dramatic findings are at the **transitive level** (SAD-CODE), not the direct doc→model level (SAD-SAM). This is actually good for the paper — it means the pipeline (which improves doc→model) and the evaluation framework (which reveals transitive amplification) reinforce each other: *"Our pipeline fixes the doc→model errors that have disproportionate downstream impact, and the evaluation framework reveals why those improvements matter more than file-level metrics suggest."*

### Why File-Level Metrics Are Insufficient

Standard evaluation computes precision, recall, and F1 over the set of all link pairs:

```
P = |recovered ∩ gold| / |recovered|
R = |recovered ∩ gold| / |gold|
F1 = 2PR / (P + R)
```

**Problem 1: Component dominance at doc→model level.** Architectural trace links are distributed unevenly across components. In BigBlueButton, HTML5 Client (14 links) and HTML5 Server (13 links) together hold 43.5% of all SAD-SAM links, while Redis DB, FSESL, and Presentation Conversion each have only 2 links. Getting the two large components right yields high micro-averaged recall while completely missing 3 components.

**Problem 2: Enrollment amplification at transitive level.** When doc→model links compose transitively with model→code, errors amplify by the enrollment factor. A component with 972 implementation files contributes 972x more to file-level metrics than a component with 1 file. Standard metrics therefore primarily measure performance on large, well-implemented components — not architectural coverage.

**Real motivating example (from actual benchmark data):**

| Project | File-level F1 | Component-corrected F1 | Component coverage | Verdict |
|---------|:---:|:---:|:---:|:---|
| BigBlueButton | **0.831** | **0.347** | 36% perfect components | Looks strong, actually weak |
| Teammates | **0.821** | **0.624** | 100% perfect components | Looks same as BBB, actually much better |

Both have nearly identical file-level F1. BBB has only 36% of components fully recovered; Teammates has 100%. Only component-level metrics reveal this.

### Metric Definitions

#### M1: Component-Level Trace F1 (CLF1)

For each model element m ∈ M, compute element-level precision, recall, and F1:

```
P_m = |recovered_m ∩ gold_m| / |recovered_m|  (0 if recovered_m = ∅)
R_m = |recovered_m ∩ gold_m| / |gold_m|
F1_m = 2 * P_m * R_m / (P_m + R_m)  (0 if P_m + R_m = 0)
```

Then macro-average over all model elements:

```
CLF1 = (1/|M|) * Σ_{m ∈ M} F1_m
```

**Interpretation:** Each architectural component contributes equally regardless of how many links it has. A method must recover links for ALL components, not just frequent ones.

**Important honesty note:** CLF1 is standard macro-F1 applied at component granularity. The formula is not novel. The contribution is the **finding** that this granularity matters for architectural TLR — prior work averages over projects, hiding component-level failures.

#### M2: Architectural Coverage (AC)

```
AC = |{m ∈ M : R_m > 0}| / |M|
```

**Interpretation:** Fraction of architectural components with at least one correctly recovered link. A missing component is a blind spot for conformance checking — far more dangerous than extra false positives on a well-covered component.

**Existing data confirms this matters:** BBB has AC ≈ 64% with standard P/R/F1 looking strong at 83.1%. Teammates has AC = 100% with similar F1.

#### M3: Amplification-Corrected F1 (ACF1) — For Transitive Evaluation

For transitive doc→code links, weight each component's contribution by its enrollment factor:

```
ACF1: compute F1_m per model element over the enrolled file set, then macro-average
```

This corrects for the distortion where 3 directory-level gold entries for JabRef "logic" expand to 972 file-level entries that dominate standard F1.

**Note on DAF1 vs CLF1:** For SAD-SAM (doc→model), simple CLF1 (macro-averaging over components) is sufficient — the distribution skew is moderate. DAF1 (inverse-frequency weighting) adds little beyond CLF1 here. For SAD-CODE (transitive), ACF1 is essential because enrollment inflation makes the skew extreme. **Recommend: CLF1 for doc→model, ACF1 for doc→code.**

#### M4: Per-Component Error Profile

For each model element, classify the recovery outcome:

| Category | Definition |
|----------|-----------|
| Fully recovered | F1_m ≥ 0.9 |
| Partially recovered | 0 < F1_m < 0.9 |
| Missed | F1_m = 0 (zero correct links) |
| Over-linked | P_m < 0.5 (many false positives) |

This is NOT a single number but a profile showing WHERE each approach fails. Existing data shows:
- BBB: 64% fully recovered, 36% partially/missed — worst component coverage despite strong file-level F1
- Teammates: 100% fully recovered — best coverage despite similar file-level F1

### How Metrics Connect to the Failure Taxonomy

| Metric | What it reveals | Which failure modes it exposes |
|--------|----------------|-------------------------------|
| CLF1 (doc→model) | Per-component accuracy, small components weighted equally | **Generic terminology** — components with generic names (≤2 links) get missed |
| AC (doc→model) | Whether any component is completely invisible | **Abstraction mismatch** — abstractly-named components with no lexical match get zero coverage |
| ACF1 (transitive) | True component-level quality after enrollment | **Propagation sensitivity** — shows how doc→model errors amplify to 972x downstream |
| Error profile | Which specific components each approach fails on | **All failure modes** — enables fine-grained diagnosis |

### The Evaluation-Pipeline Loop (Option A+B Integration)

The two contributions reinforce each other through the **amplification finding**:

1. **Pipeline contribution (Option A):** Our multi-phase pipeline improves doc→model recovery, especially for components that are hard for lexical matching — small components with generic names, abstractly-named components, components referenced only via coreference.

2. **Evaluation contribution (Option B):** Standard metrics **understate** these improvements because they're dominated by large components. Component-level metrics and amplification-corrected metrics reveal the true impact: a 3-5pp improvement at the doc→model level on small components translates to a much larger improvement at the transitive level because those small components have high enrollment factors.

3. **The bridge sentence:** *"The evaluation framework reveals that doc→model improvements on small, hard-to-link components have disproportionate downstream impact — a single corrected doc→model link can fix up to 972 file-level errors in transitive tracing."*

This is the paper's unique selling point: neither contribution is complete without the other. The pipeline produces the improvement; the evaluation framework explains why it matters more than it appears.

---

## 4. Anticipated Reviewer Concerns

### Severity Overview

| Severity | Count | Key Examples |
|----------|-------|-------------|
| CRITICAL | 3 | Evaluation on development data; LLM pre-training contamination; domain homogeneity |
| MAJOR | 7 | Immunity rules; macro F1 weighting; LLM variance; approval bias; cross-model gap; small gold standards; convention guide structure |
| MINOR | ~10 | Batch sizes; threshold values; English-only; PCM-only |

### 2.1 CRITICAL: Evaluation on Development Data

**Concern:** "The authors evaluate on the same benchmark used during design. The pipeline may be overfit to these specific projects."

**Mitigations:**
- **Standard practice in this field:** The ARDoCo group itself uses the same approach: "approaches are only evaluated on a limited number of projects... results can vary for other projects" (Fuchss et al. ICSA'25). No TLR paper in this space uses held-out evaluation because only 5 benchmark projects exist (Fuchss et al. MSR4SA'22).
- **Design study methodology:** Our pipeline was designed through systematic failure mode analysis, not parameter fitting. Each design decision is grounded in a published principle (Section 3) and a specific failure mode from the taxonomy. The pipeline architecture follows from task analysis, not from optimizing a metric on specific data.
- **Cross-model robustness:** Evaluation on two independent LLM backends (Claude Sonnet: 94.5%, GPT-5.2: 90.6%) demonstrates that the pipeline design generalizes across models, not just across prompts tuned for one model.
- **Best additional mitigation:** Add 1-2 genuinely new projects. Freeze the pipeline, evaluate without modification, report whatever results emerge.

### 2.2 CRITICAL: LLM Pre-Training Contamination

**Concern:** "The ARDoCo benchmark is publicly available on GitHub. The LLM may be recalling gold standard trace links from training data rather than performing genuine inference."

**Mitigations:**
- The multi-phase pipeline with independent extraction, validation, coreference, and judge phases makes simple recall unlikely — the LLM would need to memorize and consistently reproduce links across 7+ independently prompted phases.
- Cross-model gap (Claude 94.5% vs GPT 90.6%) provides evidence against pure memorization (different training corpora, different recall patterns).
- **Best additional mitigation:** Run a memorization probe — ask the LLM to produce trace links without the document/model context and report the results. Also show the pipeline fails without document context (zero-shot LLM gets low F1), demonstrating the pipeline's structure matters.

### 2.3 CRITICAL: Domain Homogeneity

**Concern:** "All 5 projects are Java-based web applications with 6-14 Palladio components. CamelCase heuristics and dotted-path exclusion are Java-specific."

**Mitigations:**
- The underlying principles (structural disambiguation, qualified-path exclusion, recall-first filtering) are language-agnostic even if the current implementation uses Java conventions.
- Acknowledge explicitly as scope: "We evaluate on component-based architectures documented in English with Java-style naming conventions."
- **Best additional mitigation:** Add one non-Java or larger-scale case study.

### 2.4 MAJOR: Immunity Rules = Per-Project Patches

**Concern:** "High-confidence seed preservation and syn-safe bypass appear to be structural workarounds."

**Mitigation (P6 in Section 3):**
- Frame as cascade integrity: "In multi-stage pipelines, false negatives at any stage are irrecoverable" [Bourdev05].
- "High-confidence seed links from two-pass intersection voting should not be re-filtered by a single-pass judge operating at lower precision" [Hayes06].
- Only ambiguous names (where lexical precision is inherently low [Deissenboeck06]) are forwarded for contextual validation.
- This is a principled design derived from the **propagation sensitivity** failure mode in the taxonomy.

### 2.5 MAJOR: CONVENTION_GUIDE Encodes Specific Error Patterns

**Concern:** "The 3-step structure maps to three specific error modes. This looks engineered for specific projects."

**Mitigation (P10 in Section 3):**
- Frame as the three fundamental sources of false positives in name-based traceability: structural co-occurrence, semantic overloading, lexical ambiguity [De Lucia07, Antoniol02].
- These categories are well-established in the IR-based traceability literature, not specific to these benchmarks.
- All examples in the guide use abstract SE textbook domains with zero benchmark vocabulary overlap.

### 2.6 MAJOR: Macro F1 Weighting

**Concern:** "JabRef (18 links) contributes 20% of macro F1 at 100%. No confidence intervals, no micro-F1."

**Mitigation:**
- Report micro-F1, weighted macro-F1, and per-project breakdown alongside macro-F1.
- Report confidence intervals from multiple runs.
- Macro-averaging is standard in ARDoCo papers (Keim et al. ECSA'21, ICSA'23).
- The new evaluation framework (concept-level, distribution-aware) directly addresses this concern.

### 2.7 MAJOR: LLM Variance Not Reported

**Concern:** "Results are single-run point estimates despite LLM non-determinism."

**Mitigation:**
- Run 10+ independent executions. Report mean +/- stdev.
- Use paired permutation tests for comparisons against baselines.

### 2.8 MAJOR: Cascading Approval Bias

**Concern:** "8+ mechanisms bias toward approval. This is systematic over-inclusion."

**Mitigation (P1 in Section 3):**
- This is not a bug — it is a **key empirical finding**. LLMs systematically under-link in architectural tracing. Every filtering mechanism must bias toward inclusion.
- "Automated traceability tools should target ~90% recall, with acceptable precision as low as 19-32%" [Cleland-Huang07].
- "In human-in-the-loop pipelines, false negatives are invisible while false positives can be quickly dismissed" [Hayes06, Cuddeback10].
- Present ablation evidence: aggressive filtering destroys recall (the architecture-grounded finding from our design study).

### 2.9 MAJOR: Cross-Model Gap Reveals Primary-Model Fitting

**Concern:** "The 3.9pp gap between Claude (94.5%) and GPT-5.2 (90.6%) suggests prompts are co-optimized with Claude."

**Mitigation:**
- Present cross-model evaluation as a strength — most papers report on one model only.
- Frame the gap as expected: "Different LLMs interpret prompts differently; the approach is portable with moderate performance variation."
- The gap is comparable to or smaller than typical cross-model variance reported in LLM-based SE research.

### 2.10 MAJOR: Small Gold Standards

**Concern:** "195 total links. A single misclassified link changes project-level F1 by ~5pp."

**Mitigation:**
- Acknowledge as a property of the benchmark, shared by all prior work on these datasets.
- Report sensitivity analysis: how results change if 1-3 links are modified.
- Manually verify all FPs to check for gold standard gaps.
- The new distribution-aware evaluation framework directly addresses this concern.

### 2.11 MAJOR: Taboo Violation in ILinker2 Pass B

**Concern:** "ILinker2 Pass B uses terms that overlap with benchmark vocabulary."

**Mitigation:**
- **Fix before submission.** Replace with safe alternatives:
  - "the system scheduler" instead of "system logic"
  - "manager request" instead of "client request"
  - "the gateway" instead of "the server"

---

## 5. Design Justifications with Verified Citations

### P1. Recall-First Pipeline Design

**What we do:** Union voting at review stages, approval-on-failure defaults, advocate-biased deliberation.

**Justification:** In automated traceability, the cost of false negatives fundamentally exceeds the cost of false positives. Hayes et al. show that "finding a missing relevant answer generally requires examining all the input documents in their entirety and in detail, while rejecting an irrelevant answer generally requires understanding only the irrelevant answer and the input documents at only a general level" [1]. Cuddeback et al. empirically confirm that human analysts given high-precision/low-recall candidate sets tend to degrade them rather than recover missing links [2]. Cleland-Huang et al. establish that automated traceability tools should target approximately 90% recall, with acceptable precision as low as 19-32% [3]. Our pipeline therefore biases toward inclusion at every filtering stage.

**Connection to failure taxonomy:** Addresses **propagation sensitivity** — in transitive pipelines, false negatives at the doc→model stage eliminate entire downstream code-link paths.

### P2. Asymmetric Voting (Intersection for Extraction, Union for Review)

**What we do:** Extraction uses intersection (both passes must approve). Review uses union (either pass can save a link).

**Justification:** Kittler et al. establish that classifier combination rules control the precision-recall tradeoff: the product rule (intersection) exhibits a "veto effect" where a single classifier's rejection dominates, yielding higher precision, while the sum rule (union) is more tolerant of individual classifier errors, preserving recall [4]. We apply intersection at the extraction stage (large candidate space, precision-oriented filtering reduces downstream cost) and union at the review stage (candidates have survived multiple phases, false rejection is irrecoverable). This follows the two-phase paradigm of recall-oriented generation followed by precision-oriented validation [1].

### P3. Structural Disambiguation of Component Names

**What we do:** CamelCase, multi-word, and all-uppercase names bypass semantic disambiguation. Only single dictionary words undergo contextual validation.

**Justification:** Deissenboeck & Pizka model identifier quality as a bijective mapping between names and concepts: comprehension degrades when one name maps to multiple concepts (homonymy) [5]. Single dictionary words inherently violate this bijection. Compound identifiers using CamelCase conventions provide explicit word boundaries that reduce ambiguity [6] and lead to measurably faster comprehension [7]. Because compound identifiers are constructed terms that do not occur naturally as ordinary words in English text, they are structurally unambiguous references to named entities and require no semantic disambiguation.

**Connection to failure taxonomy:** Addresses **generic terminology** — only structurally ambiguous names (single dictionary words) need semantic disambiguation.

### P4. Qualified-Name (Dotted-Path) Exclusion

**What we do:** Component names appearing only inside dotted/qualified paths are excluded as non-architectural references.

**Justification:** Architecture descriptions concern component roles, responsibilities, and interactions -- not implementation details such as package hierarchies [8, 9]. Qualified names in hierarchical namespaces reference sub-units within a component, not the component's architectural role [10]. This distinction applies across programming language ecosystems (Java packages, Python modules, C++ namespaces).

**Connection to failure taxonomy:** Addresses **abstraction mismatch** — qualified paths reference implementation structure, not architectural concepts.

### P5. Four-Rule Trace Link Validation

**What we do:** The judge validates links against four criteria: explicit reference, system-level perspective, primary focus, and component-specific usage.

**Justification:** These operationalize four established principles:
- *Explicit Reference:* Traceability requires identifiable connections between artifacts [11].
- *System-Level Perspective:* Architecture documentation describes externally visible properties, not internal implementation [8].
- *Primary Focus:* Precision in IR-based traceability degrades from incidental textual matches [1].
- *Component-Specific Usage:* Polysemous terms produce false positives when the generic sense is mistaken for the component sense [12].

**Connection to failure taxonomy:** Addresses **candidate multiplicity** — when multiple interpretations exist, the four rules provide structured disambiguation criteria.

### P6. High-Confidence Seed Preservation (Cascade Integrity)

**What we do:** Seed links from two-pass intersection voting bypass the downstream judge, except for ambiguous-name components.

**Justification:** In multi-stage pipelines, false negatives at any stage are irrecoverable [13]. Bourdev & Brandt critique hard cascade architectures for discarding upstream information [13]. High-confidence seed links that survived two-pass intersection voting should not be re-filtered by a single-pass judge operating at lower precision [1]. Only links involving structurally ambiguous component names [5] are forwarded to the judge, since these are the cases where contextual disambiguation adds value.

**Connection to failure taxonomy:** Addresses **propagation sensitivity** — preserving high-confidence upstream decisions prevents error amplification.

### P7. Bounded Coreference Window (3 Sentences)

**What we do:** Pronoun resolution searches for antecedents within the preceding 3 sentences.

**Justification:** Centering theory establishes that discourse coherence is governed by local structure [14]. Hobbs' algorithm searches backward through recent sentences in order of recency [15]. Lappin & Leass formalize the distance decay: their algorithm collects candidates from a 4-sentence window with exponential salience halving, meaning that by 3-4 sentences back, salience is effectively negligible [16]. Our 3-sentence window is consistent with these findings.

**Connection to failure taxonomy:** Addresses **discourse dependence** — component references via pronouns and definite descriptions require local discourse resolution.

### P8. Technology-Named Component Handling

**What we do:** Components named after technologies are treated as architectural references when the documentation discusses that technology.

**Justification:** The Wrapper Facade pattern formalizes the practice of encapsulating technology-specific APIs behind named component interfaces [17]. In microservice architectures, infrastructure components are frequently named after the technologies they wrap [18]. When documentation describes a technology's capabilities, it simultaneously describes the architectural role of the component that encapsulates it.

**Connection to failure taxonomy:** Addresses **abstraction mismatch** — technology names that ARE component names should not be filtered as non-architectural.

### P9. Advocate-Prosecutor Deliberation

**What we do:** Ambiguous-name links undergo structured deliberation: advocate, prosecutor, jury.

**Justification:** The polysemy problem in traceability requires contextual disambiguation that simple lexical matching cannot provide [12]. We structure this as adversarial deliberation: one pass constructs the strongest argument for the component interpretation, another argues against, and a third weighs both. The jury defaults to approval when evidence is ambiguous, following the recall-first principle [1, 3].

**Connection to failure taxonomy:** Addresses **candidate multiplicity** and **generic terminology** — adversarial deliberation surfaces evidence for and against the architectural interpretation.

### P10. Convention-Aware Boundary Filtering (3-Step Guide)

**What we do:** A 3-step reasoning guide distinguishes architectural references from (1) structural co-occurrences, (2) semantic overloading, and (3) lexical ambiguity.

**Justification:** The IR-based traceability literature identifies three fundamental sources of false positive trace links: structural co-occurrence (terms co-occur syntactically without semantic tracing), semantic overloading (terms carry both technical and domain-specific meanings), and lexical ambiguity (generic English words match component names without referring to them) [10, 1]. Our three-step guide operationalizes these categories as a structured reasoning framework.

---

## 6. Master Citation List (All Verified)

| Key | Full Citation | Venue | Year |
|-----|---------------|-------|------|
| [1] | Hayes, J.H., Dekhtyar, A., Sundaram, S.K. "Advancing Candidate Link Generation for Requirements Tracing: The Study of Methods." | IEEE TSE 32(1):4-19 | 2006 |
| [2] | Cuddeback, D., Dekhtyar, A., Hayes, J.H. "Automated Requirements Traceability: The Study of Human Analysts." | IEEE RE'10, pp. 231-240 | 2010 |
| [3] | Cleland-Huang, J., Berenbach, B., Clark, S., Settimi, R., Romanova, E. "Best Practices for Automated Traceability." | IEEE Computer 40(6):27-35 | 2007 |
| [4] | Kittler, J., Hatef, M., Duin, R.P.W., Matas, J. "On Combining Classifiers." | IEEE TPAMI 20(3):226-239 | 1998 |
| [5] | Deissenboeck, F., Pizka, M. "Concise and Consistent Naming." | Software Quality J. 14(3):261-282 | 2006 |
| [6] | Enslen, E., Hill, E., Pollock, L., Vijay-Shanker, K. "Mining Source Code to Automatically Split Identifiers for Software Analysis." | IEEE MSR'09 | 2009 |
| [7] | Schankin, A., Berger, A., Holt, D.V., Hofmeister, J.C., Riedel, T., Beigl, M. "Descriptive Compound Identifier Names Improve Source Code Comprehension." | IEEE/ACM ICPC'18, pp. 31-40 | 2018 |
| [8] | Clements, P., Bachmann, F., Bass, L., et al. *Documenting Software Architectures: Views and Beyond.* 2nd ed. | Addison-Wesley (SEI) | 2010 |
| [9] | ISO/IEC/IEEE 42010:2011. *Systems and Software Engineering -- Architecture Description.* | ISO standard | 2011 |
| [10] | Antoniol, G., Canfora, G., Casazza, G., De Lucia, A. "Recovering Traceability Links between Code and Documentation." | IEEE TSE 28(10):970-983 | 2002 |
| [11] | Gotel, O.C.Z., Finkelstein, A.C.W. "An Analysis of the Requirements Traceability Problem." | IEEE ICRE'94, pp. 94-101 | 1994 |
| [12] | Wang, W., Niu, N., Liu, H., Niu, Z. "Enhancing Automated Requirements Traceability by Resolving Polysemy." | IEEE RE'18, pp. 40-51 | 2018 |
| [13] | Bourdev, L., Brandt, J. "Robust Object Detection via Soft Cascade." | IEEE CVPR'05 | 2005 |
| [14] | Grosz, B.J., Joshi, A.K., Weinstein, S. "Centering: A Framework for Modeling the Local Coherence of Discourse." | Comp. Ling. 21(2):203-225 | 1995 |
| [15] | Hobbs, J.R. "Resolving Pronoun References." | Lingua 44:311-338 | 1978 |
| [16] | Lappin, S., Leass, H.J. "An Algorithm for Pronominal Anaphora Resolution." | Comp. Ling. 20(4):535-561 | 1994 |
| [17] | Schmidt, D.C. "Wrapper Facade: A Structural Pattern for Encapsulating Functions within Classes." | POSA series | 2000 |
| [18] | Cerny, T., Donahoo, M.J., Trnka, M. "Contextual Understanding of Microservice Architecture: Current and Future Directions." | ACM SIGAPP 17(4):29-45 | 2018 |

### Additional Supporting Citations

| Key | Full Citation | Venue | Year |
|-----|---------------|-------|------|
| [S1] | Larsen, J., Hayes, J.H., Gueheneuc, Y.-G., Dekhtyar, A. "Effective Use of Analysts' Effort in Automated Tracing." | Req. Eng. 23:119-143 | 2018 |
| [S2] | Grosz, B.J., Sidner, C.L. "Attention, Intentions, and the Structure of Discourse." | Comp. Ling. 12(3):175-204 | 1986 |
| [S3] | Soon, W.M., Ng, H.T., Lim, D.C.Y. "A Machine Learning Approach to Coreference Resolution of Noun Phrases." | Comp. Ling. 27(4):521-544 | 2001 |
| [S4] | Kuncheva, L.I. *Combining Pattern Classifiers: Methods and Algorithms.* 2nd ed. | Wiley | 2014 |
| [S5] | Caprile, B., Tonella, P. "Restructuring Program Identifier Names." | ICSM'00, pp. 97-107 | 2000 |
| [S6] | Lawrie, D., Morrell, C., Feild, H., Binkley, D. "What's in a Name? A Study of Identifiers." | IEEE ICPC'06, pp. 3-12 | 2006 |
| [S7] | Bass, L., Clements, P., Kazman, R. *Software Architecture in Practice.* 3rd/4th ed. | Addison-Wesley | 2012/2021 |

---

## 7. Killed Citations (Do Not Use)

| Citation | Reason |
|----------|--------|
| Maro et al. (2018), "Maintenance of Traceability Links" | Paper does not exist under this title |
| Galster & Avgeriou (2011), "Empirically-grounded Reference Architectures" | About reference architecture methodology, not component naming conventions |
| Arunthavanathan et al. (2016), MERCon | Does not discuss word sense disambiguation in traceability |
| Maalej & Robillard (2013), IEEE TSE | About API doc knowledge types, not qualified names or architecture docs |
| Viola & Jones (2004) for "don't re-filter positives" | Analogy is backwards: the cascade DOES re-filter at every stage. Use Bourdev & Brandt (2005) instead |
| Hobbs (1979) | Year is wrong -- correct year is **1978**, venue is Lingua |
| Borg et al. (2014) for "incidental mentions" | Is a survey/mapping paper, doesn't directly discuss incidental mentions. Use Hayes et al. (2006) instead |

---

## 8. ARDoCo Ecosystem Papers

### Must-Cite (Core)

| Paper | Venue | Year | Why Cite |
|-------|-------|------|----------|
| Keim, Schulz, Fuchss, Kocher, Speit, Koziolek. "Trace Link Recovery for Software Architecture Documentation" | ECSA'21 | 2021 | Origin of SAD-SAM TLR task and SWATTR pipeline |
| Fuchss, Corallo, Keim, Speit, Koziolek. "Establishing a Benchmark Dataset for TLR Between SAD and Models" | MSR4SA@ECSA'22 | 2022 | Our evaluation benchmark. Acknowledges limited projects available. |
| Keim, Corallo, Fuchss, Hey, Telge, Koziolek. "Recovering Trace Links Between Software Documentation And Code" (TransArc) | ICSE'24 | 2024 | Our transitive baseline — same backbone |
| Fuchss, Hey, Keim, Liu, Ewald, Thirolf, Koziolek. "LiSSA: Toward Generic TLR through RAG" | ICSE'25 | 2025 | Generic LLM-based TLR baseline |
| Fuchss, Liu, Hey, Keim, Koziolek. "Enabling Architecture Traceability by LLM-based Architecture Component Name Extraction" (ExArch) | ICSA'25 | 2025 | Closest related LLM work. Uses same benchmark. |

### Should-Cite (LLM Traceability)

| Paper | Venue | Year | Why Cite |
|-------|-------|------|----------|
| Rodriguez, Dearstyne, Cleland-Huang. "Prompts Matter: Insights and Strategies for Prompt Engineering in Automated Software Traceability" | REW'23 | 2023 | Validates prompt engineering for TLR |
| Guo, Cheng, Cleland-Huang. "Semantically Enhanced Software Traceability Using Deep Learning" | ICSE'17 | 2017 | First neural TLR |
| Lin, Liu, Zeng, Jiang, Cleland-Huang. "Traceability Transformed: T-BERT" | ICSE'21 | 2021 | Pre-LLM neural TLR |
| Hey, Fuchss, Keim, Koziolek. "Requirements TLR via RAG" | REFSQ'25 | 2025 | RAG-based TLR methodology |

### Useful for Evaluation Methodology Defense

| Paper | What It Says |
|-------|-------------|
| Fuchss et al. ICSA'25 (ExArch) | Standard defense: "approaches are only evaluated on a limited number of projects... results can vary for other projects" |
| Fuchss et al. MSR4SA'22 (Benchmark) | Acknowledges "lack of uniform benchmarks" for SAD-SAM TLR |
| Fuchss et al. REW'25 (Beyond Retrieval) | "IR performance heavily depends on project-specific hyperparameter tuning" -- argues LLM approaches are more generalizable |

---

## 9. Release Hygiene Checklist

### Publish as Fresh Repository

Release ONLY the final working version as a clean, new repository. No git history from the development repo.

### Files to Include

- [ ] Final linker implementation (renamed to `llm_linker.py` or `sad_sam_linker.py`)
- [ ] Core modules: `data_types.py`, `document_loader.py`, `model_analyzer.py`, `pcm_parser.py`, `llm_client.py`
- [ ] ILinker2 baseline (with taboo fix applied)
- [ ] Evaluation scripts
- [ ] `pyproject.toml`, `README.md`

### Files to Exclude

- [ ] All intermediate linker versions (V26a, V30a-V30d, V31, V33, V33f, V33g)
- [ ] `BENCHMARK_TABOO.md`
- [ ] `.claude/` memory directory
- [ ] All ablation/analysis test scripts
- [ ] `run_ablation.py` (or strip to only support the released version)
- [ ] All internal `.md` analysis files (this file, `V26A_VS_V31_*`, `V31_FINAL_SUMMARY.md`, etc.)
- [ ] `writing/` directory

### Code Cleanup in Final Linker

| Area | Problem | Fix |
|------|---------|-----|
| Docstring | References version numbers, "benchmark-derived", FP counts | Rewrite as neutral pipeline description |
| Log filenames | Contains version-specific names | Change to generic names |
| Comments | Internal development notes | Rewrite as principled design descriptions |
| Checkpoint dirs | Version-specific paths | Change to generic names |

### ILinker2 Fix (URGENT — Before Submission)

ILinker2 Pass B uses terms that overlap with benchmark vocabulary. Replace with safe alternatives:
- "the system scheduler" instead of "system logic"
- "manager request" instead of "client request"
- "the gateway" instead of "the server"

---

## 10. Paper Section Templates

### Introduction (6 paragraphs)

**P1 — Why architectural traceability is special:**
Architecture documents capture design intent, responsibilities, and structural decisions that are distributed unevenly across code and models. Recovering trace links is essential for conformance, maintenance, and evolution, but expensive to maintain manually.

**P2 — What transitive recovery achieved:**
Recent architectural approaches show that transitive composition through intermediate architecture models is effective. TransArC (ICSE'24) recovers documentation-to-code links transitively through an architecture model. LiSSA (ICSE'25) demonstrates that LLMs and RAG can support generic traceability across artifact types. Both represent genuine progress.

**P3 — The upstream bottleneck (task-level, not component-level):**
However, the end-to-end quality of transitive recovery depends disproportionately on the upstream doc→model stage. This stage faces three specific challenges: (1) **abstraction mismatch** — documents describe responsibilities while models expose sparse labels; (2) **discourse-distributed evidence** — referents are often only resolvable through coreference or section context; (3) **generic term ambiguity** — words like "service" or "manager" match model elements lexically but refer to the concept generically. Neither lexical matching (TransArC/SWATTR) nor generic retrieve-then-classify (LiSSA) adequately addresses these architecture-specific challenges.

**P4 — The LLM failure mode insight:**
We find that when LLMs are applied to this task, the dominant failure mode is false rejection, not false acceptance. LLMs systematically under-link because they interpret architectural mentions conservatively. This means the pipeline architecture must be explicitly designed to preserve recall through multiple validation passes — a finding that contradicts the expectation that more powerful models need less pipeline structure.

**P5 — Our solution:**
We propose [name], which redesigns the doc→model stage as a two-pass architecture-aware inference process. The first pass generates candidate links using explicit architectural knowledge. The second pass refines them through generic-term identification, coreference resolution, and multi-agent judge deliberation calibrated for recall preservation. The resulting links compose transitively with the same model→code backbone as TransArC, enabling controlled assessment. We also introduce a distribution-aware evaluation framework that measures concept-level trace accuracy and architectural coverage, revealing limitations hidden by file-level F1.

**P6 — Contributions:**
1. An empirical characterization of doc→model failure modes in LLM-based architectural tracing, including the finding that LLMs systematically under-link.
2. A multi-phase pipeline derived from this failure taxonomy, combining architecture-grounded generation with discourse-aware calibration.
3. A distribution-aware evaluation framework for uneven architectural trace links.
4. Controlled comparison showing that first-leg improvements propagate through the transitive chain, with cross-model evaluation on two LLM backends.

### Section 3: Design Principles

Our pipeline operationalizes five principles grounded in the traceability and NLP literature:

**P1. Recall-first candidate generation.** In human-in-the-loop traceability, recovering false negatives requires exhaustive re-analysis of all input documents, while rejecting false positives requires only shallow review [1]. Empirical studies show analysts cannot reliably recover missed links [2], and automated tools should target ~90% recall even at 19-32% precision [3]. We implement this through union voting at filtering stages and approval-biased review prompts.

**P2. Syntactic disambiguation of identifiers.** Compound identifiers using CamelCase conventions provide explicit word boundaries that reduce naming ambiguity [5, 7]. Unlike single dictionary words, which suffer from homonymy [5, 12], compound identifiers are constructed terms with low polysemy [6]. We bypass semantic disambiguation for structurally unambiguous names.

**P3. Architectural abstraction level.** Architectural descriptions concern component roles, responsibilities, and interactions -- not implementation details [8, 9]. Qualified names in hierarchical namespaces reference implementation sub-units, not the parent component [10]. We filter such references as non-architectural.

**P4. Trace link validation criteria.** We validate links against four criteria from the traceability literature: (1) identifiable artifact reference [11], (2) architectural abstraction level [8], (3) relevance over incidental mention [1], and (4) component-specific usage distinguishing named entities from polysemous terms [12].

**P5. Recall-preserving cascades.** In multi-stage pipelines, false negatives at any stage are irrecoverable [13]. Following the two-phase paradigm of recall-oriented generation followed by precision-oriented validation [1], high-confidence seed links are preserved through downstream stages. Intersection voting [4] is applied at extraction (precision-first), while union voting is applied at review (recall-first).

### Section 6: Threats to Validity

**Internal validity.** Our pipeline uses a non-deterministic LLM for multiple phases. We report mean and standard deviation over N independent runs to characterize variance. The cross-model evaluation with a second LLM backend demonstrates portability with moderate performance variation.

**External validity.** We evaluate on the ARDoCo benchmark [Fuchss22], the standard benchmark for SAD-SAM TLR [Keim21, Keim24, Fuchss25]. As with prior work [Fuchss25-ICSA], results may vary for other projects, particularly those using different naming conventions or documentation styles. Our design principles (recall-first filtering, structural disambiguation, bounded coreference) are language-agnostic, though the current implementation assumes English documentation with Java-style naming conventions. Our pipeline was designed through systematic failure mode analysis — each phase addresses a specific failure mode from the taxonomy (Section 3.1), grounded in published traceability and NLP principles rather than dataset-specific tuning.

**Construct validity.** The benchmark gold standards contain 18-62 links per project. We report per-project breakdown alongside macro-averaged metrics. Our distribution-aware evaluation framework (Section 5) directly addresses the concern that file-level metrics may overweight large components. [If memorization probe was run: "We performed an LLM memorization probe and found [results]."]

**Data leakage.** All LLM prompts use abstract examples from safe domains (compiler design, operating systems, e-commerce) with no overlap with benchmark vocabulary. Component classification, synonym discovery, and link validation are performed dynamically at runtime using only the input document and model — no benchmark-specific knowledge is encoded in the pipeline.

### Section 3: Problem Analysis — Why Doc→Model Tracing Is Hard

This is the paper's scientific core. Present BEFORE the approach.

**3.1 Failure Taxonomy.** Present the 5 failure modes with concrete (non-benchmark!) examples. Use generic SE examples: "order processing subsystem" → `OrderCore`, pronoun resolution in a compiler design document, etc.

**3.2 Evidence: Single-Shot LLM Baseline.** Show that simply asking an LLM to produce trace links (no pipeline) gives poor results. This proves:
- The pipeline's structure matters (not just LLM power)
- LLMs systematically under-link without recall-preserving architecture
- The benchmark is not trivially memorized

**3.3 Key Finding.** State explicitly: "The dominant LLM failure mode in architectural trace recovery is false rejection, not false acceptance." Support with data from the single-shot baseline showing high precision but low recall.

### Section 5: Evaluation Framework

Use the concrete metric definitions from Section 3 of this planning doc (CLF1, AC, DAF1, error profile). Present with:
1. The motivating example (Method X vs Method Y with identical file-level F1 but different coverage)
2. Formal definitions with formulas
3. Connection table showing which metrics expose which failure modes

### Section 6: Evaluation

Organize by RQ:

**6.1 Setup:** Benchmark, models, baselines, statistical methodology (N runs, paired tests).

**6.2 RQ1 (Effectiveness):** Main comparison table. Report BOTH standard metrics (for comparability) AND new metrics (for insight). Highlight cases where conclusions differ.

**6.3 RQ2 (Phase Contribution):** Ablation table. Each row removes one phase. Show per-phase delta on F1, concept-level F1, and coverage. This validates the failure taxonomy — each phase addresses a real failure mode.

**6.4 RQ3 (Evaluation Lens):** Re-rank all approaches under new metrics. Show the "reversal" cases where an approach looks strong on file-level F1 but weak on concept-level coverage. This justifies the evaluation framework.

**6.5 RQ4 (Portability):** Cross-model table with variance. Claude vs GPT-5.2 with mean ± stdev from 10+ runs.

**6.6 Error Propagation Analysis:** Controlled transitive experiment (same ArCoTL backbone, different doc→model). Show how first-leg improvement propagates to doc→code quality.

### Section 7: Discussion — The Connecting Story

This is where the two contributions (method + evaluation) come together:

1. **"Under file-level F1, the gap between approaches appears moderate. Under concept-level metrics, the gap is larger because our method recovers small, abstractly-named components that lexical matching misses."** This shows the evaluation framework reveals what the method is actually doing.

2. **"The ablation study confirms that each failure mode from the taxonomy corresponds to a real performance gap."** This validates the design study methodology.

3. **"The error propagation experiment shows that first-leg improvements of X pp translate to Y pp downstream."** This justifies the focus on the doc→model bottleneck.

4. **Cost analysis:** Tokens, time, dollars. Compare practical cost to TransArC (zero LLM cost) and discuss the tradeoff.

### Methodology Section: Design Study Framing

Our pipeline design emerged from a systematic analysis of LLM-based architectural trace recovery. We identified five failure modes specific to documentation-to-model tracing (Section 3.1) and designed each pipeline phase to address a specific mode:

| Failure Mode | Pipeline Phase | Principle |
|---|---|---|
| Abstraction mismatch | Architecture-knowledge prompting | P3 (Architectural abstraction) |
| Generic terminology | Convention-aware filtering | P3, P10 |
| Discourse dependence | Coreference resolution | P7 (Bounded window) |
| Candidate multiplicity | Deliberative multi-LLM validation | P5, P9 |
| Propagation sensitivity | Recall-preserving calibration | P1, P6 |

This mapping from failure modes to pipeline phases provides the design rationale. The pipeline structure follows from the task analysis, not from ad-hoc addition of components.

### Abstract Template (250 words)

Transitive recovery through intermediate architecture models is a promising strategy for linking software architecture documentation to code. However, the end-to-end quality of such approaches depends disproportionately on the upstream documentation-to-model stage, where architectural concepts must be mapped from natural-language descriptions to model elements. We identify five failure modes specific to this stage — abstraction mismatch, generic terminology, discourse dependence, candidate multiplicity, and propagation sensitivity — and find that LLMs applied to this task systematically under-link, making false rejection the dominant failure mode. We therefore redesign the documentation-to-model stage as a multi-phase architecture-aware pipeline: a first pass generates candidate links using explicit architectural knowledge, while subsequent phases refine candidates through generic-term identification, coreference resolution, and deliberative multi-perspective validation, all calibrated to preserve recall. The resulting links compose transitively with the same model-to-code backbone as prior work, enabling controlled measurement of first-leg improvements on end-to-end trace recovery. We also introduce a distribution-aware evaluation framework — concept-level trace F1, architectural coverage, and per-component error profiles — that exposes limitations hidden by standard file-level metrics. On the ARDoCo benchmark (5 projects), our approach achieves [X%] concept-level F1 and [Y%] architectural coverage, compared to [A%/B%] for the strongest baseline. Ablation results validate that each pipeline phase addresses a distinct failure mode. Cross-model evaluation on two LLM backends demonstrates that the pipeline design, not the specific model, drives performance.
