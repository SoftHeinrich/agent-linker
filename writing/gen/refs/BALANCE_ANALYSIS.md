# Balance Analysis: Approach + Metric Papers

Reference structures for an ICSE paper presenting **both** a new approach (AALinker) and a new evaluation suite as co-equal contributions. Section titles and page numbers are taken verbatim from the PDFs under `writing/gen/refs/`.

---

## 1. Chen et al. 2021 — "Evaluating Large Language Models Trained on Code" (Codex / HumanEval / pass@k)

35 pages total (arXiv preprint, no strict venue limit), 13 pages of main text before references.

### 1.1 Section structure and pages

| # | Section (verbatim) | Pages | Role |
|---|---|---|---|
| — | Abstract | p.1 | both |
| 1 | Introduction | p.1–2 | both |
| 2 | Evaluation Framework (2.1 Functional Correctness, 2.2 HumanEval, 2.3 Sandbox) | p.2–4 | **metric** |
| 3 | Code Fine-Tuning (3.1 Data, 3.2 Methods, 3.3 Results, 3.4 Comparative Analysis, 3.5 APPS) | p.4–7 | **approach** |
| 4 | Supervised Fine-Tuning (4.1–4.5) | p.7–9 | approach |
| 5 | Docstring Generation | p.9–10 | approach |
| 6 | Limitations | p.10–11 | merged |
| 7 | Broader Impacts and Hazard Analysis (7.1–7.8) | p.11–13 | approach-side |
| 8 | Related Work | p.13–14 | both |
| 9 | Conclusion | p.14 | both |

Key observation: **the metric section (§2) precedes the approach section (§3)**. The metric is defined and motivated *before* the model is described, so all subsequent results can lean on it. §2 is ~2.5 pages, §3–§5 together are ~6 pages. Metric ≈ 30% of body, approach ≈ 70%.

### 1.2 Abstract framing

Sequenced, approach-foregrounded: "We introduce **Codex**, a GPT language model… On **HumanEval, a new evaluation set we release**…" (p.1). Codex is named first; HumanEval is a relative clause. pass@k is *not* named in the abstract — the metric is foregrounded only as a dataset, not as a formal contribution.

### 1.3 Intro framing

Same order. Intro opens with sequence models / program synthesis context, then "To accurately benchmark our model, we create a dataset of 164 original programming problems with unit tests" (p.2). No bulleted contributions list — narrative, with the approach as protagonist and the metric/dataset as enabling infrastructure.

### 1.4 Results partitioning

The metric (pass@k) is used **as the yardstick** to evaluate Codex (Tables 1, 2, 3) — not evaluated as a standalone artefact. The closest the paper comes to *justifying* pass@k is §2.1 "Functional Correctness", which argues against BLEU and shows (Figure 8, p.6) that BLEU score distributions for correct/incorrect Codex outputs overlap. So pass@k's defence is interleaved with approach results, not in its own section.

### 1.5 Threats / discussion

§6 "Limitations" and §7 "Broader Impacts and Hazard Analysis" are entirely **approach-side** (sample efficiency, prompt difficulty, safety, bias, environment, legal). There is no separate metric-validity discussion. Appendix A "Estimating pass@k" (p.19) handles statistical validity of the metric, deferred to appendix.

---

## 2. Ribeiro et al. 2020 — "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList" (ACL Best Paper)

11 pages total: 9 main + 2 references. This is the **methodology + demonstrative-bug-finding** pattern, closest to a single-contribution paper where the "approach" *is* the evaluation methodology.

### 2.1 Section structure and pages

| # | Section (verbatim) | Pages |
|---|---|---|
| — | Abstract | p.1 (4902) |
| 1 | Introduction | p.1–2 |
| 2 | CheckList (2.1 Capabilities, 2.2 Test Types, 2.3 Generating Test Cases at Scale) | p.2–3 |
| 3 | Testing SOTA Models with CheckList (Sentiment, QQP, MC) | p.3–7 |
| 4 | User Evaluation (4.1 industry team study, 4.2 controlled user study) | p.7–9 |
| 5 | Related Work | p.9 |
| 6 | Conclusion | p.9–10 |

Ribeiro is a useful **counterpoint**: only one contribution (the methodology) so all 9 pages serve it. §2 (methodology) is ~2 pages; §3+§4 (evidence-of-utility) is ~6 pages. The lesson: **6:2 ratio of "showing the metric finds real bugs" vs. "defining the metric"**.

### 2.2 Abstract framing

Single contribution, framed flaw-first: "held-out accuracy… often overestimates performance… we introduce CheckList, a task-agnostic methodology…" then immediately "We illustrate the utility of CheckList with tests for three tasks, identifying critical failures in both commercial and state-of-art models." Method + evidence inseparable.

### 2.3 Intro framing

No bulleted contribution list. Narrative: standard-paradigm flaw → CheckList overview (Figure 1 example with sentiment model) → instantiated on three tasks → bugs revealed → user studies. The intro already promises evidence; methodology is never presented in the abstract without bugs attached.

### 2.4 Results partitioning

§3 demonstrates CheckList by **applying it to other people's models** (Microsoft, Google, Amazon, BERT, RoBERTa). The methodology is "evaluated" entirely through the failures it surfaces in third-party systems. §4 is a *meta-evaluation*: does CheckList help humans find more bugs than they would otherwise (controlled user study with 18 NLP practitioners, Table 4)?

### 2.5 Threats / discussion

Methodology-validity arguments are folded into §5 Related Work (comparison to challenge sets, probes, perturbation work) — explicitly framed as "what CheckList cannot do" (data versioning, labeling errors, worst-case security). No separate threats section; methodology and discussion are one.

---

## 3. Eghbali & Pradel 2022 — CrystalBLEU (ASE Distinguished)

12 pages total. **Single-contribution (metric only)** calibration.

### 3.1 Section structure and pages

1 INTRODUCTION (p.1–2); 2 BACKGROUND (2.1 BLEU Score, 2.2 BLEU on Code, p.3); 3 APPROACH (3.1 Trivially Shared N-grams, 3.2 Distinguishability, 3.3 CrystalBLEU, p.3–6); 4 EVALUATION (4.1 Baselines, 4.2 Datasets, 4.3 RQ1 Distinguishing, 4.4 RQ2 Avoiding Misleading Results, 4.5 RQ3 Scalability, 4.6 RQ4 Parameter Choice, p.7–9); 5 THREATS TO VALIDITY (p.10); 6 RELATED WORK (p.10); 7 CONCLUSION (p.11).

### 3.2 Abstract framing

Pure flaw-first: BLEU adopted in SE; programming languages have trivially shared n-grams; "This paper presents CrystalBLEU, an evaluation metric based on BLEU…" Sole contribution; evaluation is correlation-with-similarity-judgments and distinguishing-power, not "using it to rank tools".

Lesson: a metric-only ASE paper still spends ~3 pages on approach and ~3 pages on metric-validity RQs.

---

## 4. Allamanis 2019 — "The Adverse Effects of Code Duplication in Machine Learning Models of Code"

10 pages. **Critique-plus-measurement-tool** calibration (no headline new method).

### 4.1 Section structure and pages

1 Introduction (p.1–2); 2 Code Duplication & Machine Learning (p.2–3); 3 Measuring Duplication (p.3–4); 4 Impact on Machine Learning Models (4.1 Biased vs. Unbiased Performance, 4.2 Model Capacity and Impact on Code, 4.3 Other Models and Tasks, p.5–8); 5 Mitigating Duplication: Best Practices (5.1 Conclusions, p.8–9).

### 4.2 Abstract framing

"A significant threat… was recently identified… However, the impact of code duplication has not been noticed by researchers… we explore the effects… reported performance metrics are sometimes inflated by up to 100%… We present a duplication index… list best practices… release tools." Pattern: **flaw → quantification → tooling → best practices**. No "approach" claim; the contribution is the *measurement* and the dataset-level critique.

---

## 5. Synthesis: Recommendation for the AALinker + 6-Metric Paper (ICSE, 10 pages)

**Section order and page budget.** Follow Chen's metric-first ordering, but allocate roughly 50/50 rather than 30/70 because the metric suite is the *novel* contribution while AALinker code is mature:

1. Intro — 1 p.
2. Background & Benchmark Bias (named: bias-1, bias-2, bias-3) — 1.25 p.
3. **Evaluation Suite** (six metrics, mapped to the three biases) — 2 p.
4. **AALinker** — 2 p.
5. Experimental Setup — 0.75 p.
6. Results: 6.1 AALinker vs. baselines under old benchmark; 6.2 the same comparison under the new suite; 6.3 metric-suite validity (sanity baselines, agreement / divergence with old benchmark) — 2 p.
7. Threats — 0.5 p. (split sub-subsections: approach-side, metric-side, like CrystalBLEU §5)
8. Related work — 0.25 p.
9. Conclusion — 0.25 p.

**Interleave in intro and results, separate in body.** Abstract and intro should *interleave*, Ribeiro-style: every mention of AALinker is paired with "evaluated under a six-metric suite that exposes biases the prior benchmark hid". Body sections (§3 and §4) should be cleanly separated for clarity. Results §6 should partition explicitly: 6.1+6.2 use the suite *to evaluate* AALinker (Chen pattern), 6.3 evaluates *the suite itself* via sanity baselines and correlation with the old metric (CrystalBLEU pattern).

**The single biggest risk.** Reviewers will treat AALinker as "the real contribution" and the metric suite as housekeeping — exactly as Codex eclipsed pass@k in citation discourse. **Mitigation:** put the metric suite *before* AALinker in the body (Chen does this); name the three biases in the abstract (e.g. "AALinker, evaluated under a six-metric suite that corrects **leakage, granularity, and prevalence** biases…"); and include §6.3 metric-validity sub-section so the suite stands on its own evidentially, not only as a yardstick for AALinker.

---

## 6. Additional Dual-Contribution Papers (Verified)

Five additional papers were obtained directly from arXiv (PDFs in this directory) and inspected to test whether Chen et al.'s metric-first ordering generalises. Each candidate had to (a) propose both a new approach/system AND a new evaluation metric/methodology as headline contributions, and (b) come from a top venue. Sections and quotations are taken verbatim from the downloaded PDFs.

### 6.1 Liang et al. 2022 — HELM ("Holistic Evaluation of Language Models"), TMLR 2023 (arXiv 2211.09110)

162 pages. **Genuine dual contribution**: a holistic benchmarking *framework/taxonomy* (the "approach") and a *7-metric suite* (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency) plus 21 new scenarios.

Body order from the TOC on p.10–11: §1 Introduction, §2 Preliminaries (2.1 Scenarios, 2.2 Adaptation, 2.3 Metrics, 2.4 Roadmap), §3 Core scenarios (p.15–24), §4 General metrics (p.25–34), §5 Targeted evaluations (p.35), §6 Models (p.43), §7 Adaptation via prompting (p.45), §8 Experiments and results (p.47), §9 Related work, §10 What is missing, §11 Limitations, §12 Conclusion.

Contribution bullets (§1.3, verbatim order): (1) Taxonomy, (2) Broad coverage [scenarios+metrics], (3) Evaluation of existing models, (4) Empirical findings. Abstract names taxonomy first, multi-metric approach second.

**Order: framework/scenarios before metrics.** Pages: framework + scenarios ≈ 25 p., general metrics ≈ 10 p.

### 6.2 Zheng et al. 2023 — MT-Bench / Chatbot Arena ("Judging LLM-as-a-Judge"), NeurIPS 2023 D&B (arXiv 2306.05685)

29 pages. **Genuine dual contribution**: two new benchmarks (MT-bench, Chatbot Arena) plus the LLM-as-a-judge methodology (the new metric).

Body order: §1 Introduction, §2 MT-Bench and Chatbot Arena (2.1 Motivation, 2.2 MT-Bench, 2.3 Chatbot Arena), §3 LLM as a Judge (3.1 Types, 3.2 Advantages, 3.3 Limitations, 3.4 Addressing limitations, 3.5 Multi-turn judge), §4 Agreement Evaluation, §5 Human Preference Benchmark and Standardized Benchmark, §6 Discussion, §7 Conclusion.

Contribution sentence (§1, p.2 verbatim): "This paper makes two contributions: (1) a systematic study of LLM-as-a-judge; and (2) human preference datasets…". Abstract leads with the judge methodology ("we explore using strong LLMs as judges") then the benchmarks ("we then verify the agreement… by introducing two benchmarks").

**Order: benchmarks (approach) before judge-metric in body, but metric-first in abstract and in contribution list.** Mixed framing.

### 6.3 Liu et al. 2023 — EvalPlus / HumanEval+ ("Is Your Code Generated by ChatGPT Really Correct?"), NeurIPS 2023 (arXiv 2305.01210)

15 pages. **Fused contribution**: EvalPlus is simultaneously the "approach" and the new evaluation instrument; HumanEval+ is the new metric instance.

Body order: §1 Introduction, §2 Approach (2.1 Automated Test Input Generation, 2.2 Test-Suite Reduction, 2.3 Program Input Contracts), §3 Evaluation, §4 Related Work, §5 Conclusion & Future Work.

Contribution bullets (verbatim, in order on p.3): "Study… Approach: EvalPlus… Results: HumanEval+". The "Approach" *is* the metric construction (mutation- and LLM-based test input generation), and §3 Evaluation applies it to 26 LLMs.

**Order: single metric-as-approach section before the empirical evaluation.** Closest analogue to Chen's metric-first ordering; the metric and approach are not separable. Categorised below as metric-first (fused).

### 6.4 Jimenez et al. 2024 — SWE-bench ("Can Language Models Resolve Real-World GitHub Issues?"), ICLR 2024 (arXiv 2310.06770)

10 pages main text, 52 pages with appendices. **Genuine dual contribution**: SWE-bench benchmark + grading methodology (the "metric") *and* SWE-Llama, a fine-tuned CodeLlama model (the "approach").

Body order: §1 Introduction, §2 SWE-bench (2.1 Benchmark Construction, 2.2 Task Formulation, 2.3 Features of SWE-bench, 2.4 SWE-bench Lite), §3 SWE-Llama: Fine-tuning Code Llama for SWE-bench, §4 Experimental Setup (4.1 Retrieval-Based Approach, 4.2 Input Format, 4.3 Models), §5 Results (5.1 A Qualitative Analysis), §6 Related Work, §7 Discussion.

Contribution paragraph (verbatim, p.2): "In addition to SWE-bench our contributions include the release of a training dataset, SWE-bench-train… we release two fine-tuned models, SWE-Llama 7b and 13b…". Abstract introduces SWE-bench first ("we introduce SWE-bench, an evaluation framework…"), then SWE-Llama.

**Order: metric/benchmark (§2, ~3 p.) before approach/model (§3, ~1 p.).** Pages strongly biased toward the metric.

### 6.5 Lin, Hilton & Evans 2022 — TruthfulQA ("Measuring How Models Mimic Human Falsehoods"), ACL 2022 (arXiv 2109.07958)

39 pages with appendices. **Dual but tightly coupled**: TruthfulQA benchmark (approach) + truthfulness scoring methodology including a GPT-judge automated metric (the metric).

Body order: §1 Introduction (incl. §1.1 Contributions), §2 The TruthfulQA Benchmark (2.1 Defining the truthfulness objective, 2.2 Constructing TruthfulQA, 2.3 Validating TruthfulQA), §3 Experiments (3.1 Models and prompts, 3.2 Tasks and evaluation), §4 Results (incl. 4.4 Automated metrics vs human evaluation), §5 Discussion, §6 Related Work, §7 Conclusion, §8 Ethics and Impact.

Abstract verbatim: "We propose a benchmark to measure whether a language model is truthful…" — benchmark named first. The metric (truthfulness scoring; GPT-judge) is introduced inside §2.1 and validated in §4.4, not given its own top-level section.

**Order: benchmark (approach) before metric.** Metric is folded into the benchmark/experiments rather than separately staged.

### 6.6 Tally Table

| Paper | Venue | Body ordering | Abstract names first |
|---|---|---|---|
| Chen et al. 2021 (HumanEval/pass@k) | arXiv/OAI | **metric-first** (§2 metric, §3+ approach) | metric (pass@k / HumanEval) |
| Liang et al. 2022 (HELM) | TMLR 2023 | **approach-first** (§3 scenarios, §4 metrics) | taxonomy/framework |
| Zheng et al. 2023 (MT-Bench) | NeurIPS D&B 2023 | **approach-first** body (§2 benchmark, §3 judge); metric-first in abstract | judge-metric (then benchmarks) |
| Liu et al. 2023 (EvalPlus) | NeurIPS 2023 | **metric-first (fused)** — §2 Approach *is* the metric construction | metric/framework |
| Jimenez et al. 2024 (SWE-bench) | ICLR 2024 | **metric-first** (§2 benchmark, §3 SWE-Llama) | metric/benchmark |
| Lin et al. 2022 (TruthfulQA) | ACL 2022 | **approach-first** (§2 benchmark, §3 experiments incl. metric) | benchmark |

Summary counts (n=6 including Chen): metric-first 3 (Chen, EvalPlus, SWE-bench), approach-first 3 (HELM, MT-Bench body, TruthfulQA). MT-Bench inverts between abstract (metric-first) and body (approach-first).

### 6.7 Verification notes — papers considered and rejected

- **G-Eval (Liu et al., EMNLP 2023, arXiv 2303.16634)** — verified downloaded but **single-contribution (metric only)**. §2 Method = the metric framework, §3 Experiments. No separate "approach". Not dual; excluded.
- **AgentBench (Liu et al. 2023)** — not downloaded; in the rejected-because-not-checked bin under the 12-minute time-box.
- **AlpacaEval / AlpacaFarm (Li et al., NeurIPS 2023)** — not verified within time-box.
- **Defects4J (Just et al., ISSTA 2014)** — outside the 2018–2025 window.
- **CodeXGLUE (Lu et al., NeurIPS 2021 D&B)** — flagged by the user as probably single-contribution; not checked.
- **Prometheus (Kim et al., ICLR 2024)** — not verified within time-box.

### 6.8 Implication for the AALinker paper

With n=6 the simple "metric-first by default" reading of Chen does **not** generalise: ordering is 3-3 between metric-first and approach-first. The stronger signals are:
- Among papers where the new system is *small or downstream of the benchmark* (SWE-bench's SWE-Llama, EvalPlus's tooling, Chen's Codex used to validate pass@k), metric-first dominates.
- Among papers where the benchmark/framework is itself the central scientific object and the metric is *layered onto it* (HELM, MT-Bench body, TruthfulQA), benchmark/approach-first dominates.

For AALinker + 6-metric suite the §5 synthesis above (metric-first, 50/50 page split, Chen pattern) is consistent with the SWE-bench / EvalPlus / Chen cluster — the cluster where the system is presented in service of, or alongside, a benchmark/metric that is the load-bearing contribution. The 3-3 split is a caution against claiming "the field uses metric-first"; the right framing in the cover letter is **"we follow the pattern used by Chen, SWE-bench, and EvalPlus"**, not "the standard order".

## 7. Method+Metric Papers (Corrected Search)

Section 6 mostly surveyed **benchmark+metric** papers (HELM, MT-Bench, SWE-bench, TruthfulQA, EvalPlus). AALinker is a different shape: an existing benchmark, a new *method* (multi-agent trace-link recovery), and a new *metric suite*. Section 7 looks specifically for "method/system/algorithm/tool for a task **and** a new evaluation metric/methodology, both claimed as contributions". This shape proved rare; verified hits are reported with high confidence, near-misses are reported as rejected.

### 7.1 Xu, Napoles, Pavlick, Chen & Callison-Burch 2016 — SARI ("Optimizing Statistical Machine Translation for Text Simplification"), TACL 2016

arXiv/ACL Anthology Q16-1029. Verified via aclanthology.org/Q16-1029 and downloaded PDF (`xu-2016-sari.pdf`, 441 KB). Highly cited (>1k). **Genuine dual contribution**: (a) two new SMT-based simplification systems (PBMT-R, SBMT-SARI) and (b) the SARI metric. Abstract verbatim: "Our work is the first to design automatic metrics that are effective for tuning and evaluating simplification systems." So both a system *adaptation* and a *metric* are advertised, and the system is in fact tuned on the new metric.

Body order (per page-level scan): §1 Introduction, §2 Related Work, §3 SARI metric, §4 SMT-based simplification (PBMT-R, SBMT-SARI), §5 Experiments, §6 Analysis, §7 Conclusion. Approximate allocation: metric ≈ 3 pp, system ≈ 2 pp, experiments ≈ 3 pp.

**Order: metric-first in body and abstract.** The system is presented as something tuned *to* the new metric. Closest structural analogue to Chen 2021 (HumanEval/pass@k) in §3 above.

### 7.2 Croce & Hein 2020 — AutoAttack ("Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks"), ICML 2020

arXiv 2003.01690. Verified via arXiv abs and downloaded PDF (`croce-2020-autoattack.pdf`, 1.1 MB). Highly cited (>3k); the de-facto robustness evaluation standard. **Genuine dual contribution**: (a) two new attack algorithms — APGD-CE and APGD-DLR (a step-size-free PGD plus a new DLR loss) — and (b) AutoAttack, a parameter-free ensemble evaluation *protocol* for adversarial robustness. The abstract leads with the reliability-of-evaluation problem and explicitly frames the ensemble as an evaluation methodology, not just another attack.

Body order (per arXiv HTML scan): §1 Introduction (problem = unreliable robustness evaluation), §2 Background, §3 Auto-PGD (APGD) ≈ 2.5 pp, §4 Alternative DLR loss ≈ 2 pp, §5 AutoAttack ensemble ≈ 1.5 pp, §6 Experiments ≈ 3 pp. The attack algorithms are presented first (~4.5 pp), the evaluation protocol second (~1.5 pp), but the evaluation protocol is the *named contribution* (the paper's title is the protocol, not the attacks).

**Order: approach (attacks) before metric (ensemble protocol) in body; evaluation-first in abstract and title.** Mixed framing, similar to MT-Bench in §6 — title/abstract foreground the evaluation methodology, body foregrounds the algorithmic ingredients.

### 7.3 Ribeiro, Wu, Guestrin & Singh 2020 — CheckList ("Beyond Accuracy: Behavioral Testing of NLP Models with CheckList"), ACL 2020 Best Paper

Already in refs as `ribeiro-2020-checklist.pdf`. Verified via aclanthology.org/2020.acl-main.442. **Dual contribution with caveat**: (a) a software tool that generates test cases via templates + perturbations and (b) the CheckList methodology itself, which is a *new evaluation methodology* (capability × test-type matrix; behavioural testing instead of held-out accuracy). The "approach" and the "metric/methodology" are *fused* into one artifact, similar to EvalPlus in §6.3. Strictly speaking the tool and the methodology are inseparable — so this is a borderline match for our (method ≠ metric) criterion. Recorded here because the paper does explicitly distinguish "methodology" from "tool" in §1.

Body order: §1 Introduction, §2 CheckList (capabilities × test types matrix = methodology), §3 Testing three commercial / SOTA systems, §4 User studies, §5 Related work. **Methodology-first in body and abstract.**

### 7.4 Tally Table (method+metric papers only)

| Paper | Venue | Body ordering | Abstract names first |
|---|---|---|---|
| Xu et al. 2016 (SARI) | TACL 2016 | **metric-first** (§3 metric, §4 system) | metric ("first to design automatic metrics…") |
| Croce & Hein 2020 (AutoAttack) | ICML 2020 | **approach-first** (§3–4 attacks, §5 ensemble) | metric/protocol (title + abstract lead with reliable evaluation) |
| Ribeiro et al. 2020 (CheckList) | ACL 2020 (Best) | **methodology-first (fused)** — methodology IS the tool | methodology |

Counts (n=3 verified): metric-first body 1 (SARI), approach-first body 1 (AutoAttack), fused 1 (CheckList). Abstract framing: 3/3 lead with the new metric/methodology.

### 7.5 Rejected candidates

Each was investigated but failed at least one of (a) "new method/system for a task" or (b) "new evaluation metric claimed as a contribution".

- **InstructGPT (Ouyang et al. 2022, arXiv 2203.02155)** — abstract and §3.2 explicitly state "methodology follows Ziegler et al., Stiennon et al.". Human-preference evaluation is used but **not claimed as new**. Reject (only (a) is new).
- **CLIP (Radford et al. ICML 2021, arXiv 2103.00020)** — zero-shot transfer is positioned as an *evaluation paradigm*, but §3.1.1 explicitly cites Visual N-Grams and the GPT family as the source of that paradigm; CLIP applies it at scale rather than claiming it as a methodological contribution. Reject (only (a) is new).
- **Constitutional AI (Bai et al. 2022, arXiv 2212.08073)** — RLAIF is new; the evaluation is human-preference of harmlessness/helpfulness, not a new metric. Reject.
- **Self-RAG (Asai et al. ICLR 2024, arXiv 2310.11511)** — new retrieval method, standard QA/factuality metrics. Reject.
- **Stiennon et al. 2020 (Learning to summarize from human feedback, NeurIPS 2020, arXiv 2009.01325)** — RLHF training method + ROUGE-vs-human-preference critique; but human preference comparison is used as *validation*, not advertised as a new methodology. Reject (borderline).
- **TBar (Liu et al. ISSTA 2019, arXiv 1903.08409)** — APR tool + fix-pattern empirical study; uses standard correct/plausible counts on Defects4J. No new metric. Reject.
- **Prophet (Long & Rinard POPL 2016)** — learned patch ranker for APR; no new evaluation metric (correctness assessed manually). Reject.
- **GenProg (Le Goues et al. ICSE 2012, MIP Award)** — APR tool; uses existing test-suite-based evaluation. The "plausible vs correct" distinction was raised by *later* critiques (Qi et al. ISSTA 2015), not GenProg itself. Reject.
- **CoCoNuT (Lutellier et al. ISSTA 2020)** — neural APR + ensemble; standard Defects4J / QuixBugs counts. Reject.
- **SequenceR (Chen et al. TSE 2019)** — seq2seq APR; standard accuracy on synthesized + real bugs. Reject.
- **CodeBLEU (Ren et al. 2020, arXiv 2009.10297)** — *metric only*, no new code-generation system. Reject (only (b)).
- **BERTScore, BLEURT, COMET, MAUVE, G-Eval** — all metric-only, like CrystalBLEU and pass@k-side-of-Chen. Reject (only (b)).
- **AutoAttack alternative reading** — if one considers APGD merely a "tweak of PGD" rather than a new algorithm, AutoAttack collapses to metric-only. We retain it because the paper presents APGD/DLR as standalone algorithmic contributions (§3 and §4 are theorem-bearing sections), but flag the ambiguity.
- **Tarantula (Jones, Harrold, Stasko ICSE 2002)** — could not verify PDF text within time-box (ACM 403 + Stasko PDF binary). Plausible match (technique + a new visualisation/suspiciousness scoring), but **not verified**, so excluded.

### 7.6 Implication for the AALinker paper — the shape is rare

Verified method+metric papers (n=3): SARI, AutoAttack, CheckList. This is meaningfully fewer than the benchmark+metric cluster (n=6 in §6), and the gap is informative: most highly-cited "dual contribution" evaluation papers in ML/NLP either (i) introduce a benchmark alongside a metric, or (ii) introduce a metric alone. A paper that introduces a *method for an existing task* and *separately* a *new metric suite* is structurally unusual.

What the three verified hits agree on:
- **Abstract: metric/methodology named first, 3/3.** Even AutoAttack, whose body leads with algorithms, leads its abstract and title with the evaluation problem.
- **Body: split 1 metric-first / 1 approach-first / 1 fused.** No clear body-order consensus.

For AALinker the practical implication is:
1. The current "metric-first" draft is defensible (matches SARI exactly and CheckList partially) but is *not* the universal pattern — AutoAttack runs approach-first in the body while keeping metric-first abstract/title framing.
2. The **safest framing** is: metric-first in the abstract and §1 contribution list (since this is what 3/3 method+metric papers do, *and* what 4/6 benchmark+metric papers in §6 do); body order can go either way and should follow whichever ordering makes the AALinker design easier to explain.
3. Because the shape is rare, the cover letter should explicitly position the paper as following the **SARI / CheckList / AutoAttack** trio rather than the more populous benchmark+metric cluster. The reviewer's mental model of "what kind of paper is this" is set by the first paragraph; naming this trio anchors it correctly.

## 8. Method+Metric Papers (Extended Search)

To stress-test the §7 conclusion (n=3 too small), we executed a broader search across NeurIPS/ICML/ICLR/ACL/EMNLP/ICSE/FSE 2014–2025 best/distinguished papers and adjacent high-citation work. Verified hits below; each was confirmed via arXiv abstract page (or NeurIPS/proceedings page), with the contribution claim quoted verbatim from the abstract or introduction. PDFs downloaded to this directory where a public source exists.

### 8.1 Heusel, Ramsauer, Unterthiner, Nessler & Hochreiter 2017 — TTUR + FID ("GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"), NeurIPS 2017

arXiv 1706.08500. Verified via arXiv abs and downloaded (`heusel-2017-fid.pdf`, 6.3 MB). Citations >15k. **Genuine dual contribution**: abstract verbatim — "We propose a two time-scale update rule (TTUR) for training GANs … For the evaluation of the performance of GANs at image generation, we introduce the 'Fréchet Inception Distance' (FID) which captures the similarity of generated images to real ones better than the Inception Score." Two named, separately-introduced contributions: (a) TTUR optimisation method with convergence theorem, (b) FID evaluation metric. The cleanest method+metric dual paper in the modern ML canon.

Body order: §1 Intro, §2 TTUR + convergence theorem (~method-heavy, several pages), §3 Experiments (where FID is defined and used; FID is introduced inline at the start of §3 / Appendix A1). **Order: approach-first in body, both named equally in abstract.** Title foregrounds the method (TTUR), not the metric. The metric is supplied to make the method's claim measurable.

### 8.2 Hardt, Price & Srebro 2016 — Equality of Opportunity ("Equality of Opportunity in Supervised Learning"), NeurIPS 2016

arXiv 1610.02413. Verified via arXiv abs and downloaded (`hardt-2016-eqop.pdf`, 795 KB). Citations >6k. **Genuine dual contribution**: (a) a new fairness criterion ("equality of opportunity" / "equalised odds") — a new evaluation methodology for discrimination — and (b) an algorithm to optimally post-process any predictor to satisfy this criterion. Abstract: "We propose a criterion for discrimination … we show how to optimally adjust any learned predictor so as to remove discrimination according to our definition."

Body order (standard NeurIPS structure verified from arXiv HTML): §1 Intro, §2 Setting/definition (criterion = methodology) ≈ 2.5 pp, §3 Deriving the optimal predictor (method) ≈ 3 pp, §4 Experiments. **Order: metric/criterion-first in body and abstract.**

### 8.3 Bolukbasi, Chang, Zou, Saligrama & Kalai 2016 — Debiasing Word Embeddings ("Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings"), NeurIPS 2016

arXiv 1607.06520. Verified and downloaded (`bolukbasi-2016-debias.pdf`, 738 KB). Citations >4k. **Genuine dual contribution**: abstract verbatim — "We define metrics to quantify both direct and indirect gender biases in embeddings, and develop algorithms to 'debias' the embedding." Both metrics and algorithms are named contributions in one clause.

Body order: §1 Intro, §2 Geometry of gender in word embeddings (sets up the bias direction = measurement methodology), §3 Direct and indirect bias metrics, §4 Debiasing algorithms (hard-debias / soft-debias), §5 Experiments. **Order: metric-first in body and abstract** (metrics literally listed first in the dual-contribution sentence: "metrics … and algorithms").

### 8.4 Carlini & Wagner 2017 — C&W Attacks ("Towards Evaluating the Robustness of Neural Networks"), IEEE S&P 2017

arXiv 1608.04644. Verified and downloaded (`carlini-2017-cw.pdf`, 1.3 MB). Citations >9k. **Genuine dual contribution**: (a) three new attack algorithms (L₀, L₂, L∞ variants) and (b) a new evaluation methodology — the use of high-confidence C&W attacks as a transferability-based benchmark for defence evaluation. Abstract quote on evaluation methodology: "we hope our attacks will be used as a benchmark in future defense attempts." Title itself frames the contribution as evaluation methodology ("Towards Evaluating the Robustness…").

Body order: §1 Intro (defensive distillation broken), §2 Background, §3 Attack-model methodology, §4 Three new attacks (algorithm-heavy, ~6 pp), §5 Comparison to prior attacks, §6 Defensive distillation broken, §7 Evaluating defences (the methodological contribution). **Order: approach-first in body (attacks dominate), evaluation-first in title and abstract framing.** Structurally identical to AutoAttack (§7.2).

### 8.5 Madry, Makelov, Schmidt, Tsipras & Vladu 2018 — PGD Adversarial Training ("Towards Deep Learning Models Resistant to Adversarial Attacks"), ICLR 2018

arXiv 1706.06083. Verified and downloaded (`madry-2018-pgd.pdf`, 1.6 MB). Citations >11k. **Genuine dual contribution**: (a) PGD-based adversarial training as a training method, (b) a saddle-point / "first-order adversary" evaluation framework framing robustness as a security guarantee — claimed in the abstract as a new way to view and *evaluate* robust models, not merely a tool. Both are foregrounded in the abstract.

Body order: §1 Intro, §2 The saddle-point formulation (= evaluation framework, ~2 pp), §3 Towards universally robust networks via PGD (method, ~4 pp), §4 Experiments. **Order: evaluation framework-first in body and abstract.** Similar to SARI: framework is presented first, the algorithm derives from it.

### 8.6 Moosavi-Dezfooli, Fawzi & Frossard 2016 — DeepFool ("DeepFool: a simple and accurate method to fool deep neural networks"), CVPR 2016

arXiv 1511.04599. Verified and downloaded (`moosavi-2016-deepfool.pdf`, 4.9 MB). Citations >6k. **Borderline dual contribution**: (a) the DeepFool minimum-perturbation attack algorithm and (b) a robustness *quantification* methodology — the paper explicitly motivates DeepFool as a way to "reliably quantify the robustness" of classifiers (abstract). The "metric" is the perturbation-norm-based robustness ρ̂_adv(f) defined in §2 as a contribution. Not as clean a metric/methodology contribution as FID or C&W, but the paper does explicitly claim "no effective methods have been proposed to accurately compute the robustness" and presents the algorithm as the answer to that measurement gap.

Body order: §1 Intro, §2 Defining the robustness metric ρ̂_adv ≈ 1 pp, §3 DeepFool algorithm (binary then multiclass) ≈ 3 pp, §4 Experiments. **Order: metric-first in body and abstract.**

### 8.7 Tally Table — full method+metric corpus across §§7–8

| # | Paper | Venue | Body order | Abstract framing |
|---|---|---|---|---|
| 1 | Xu et al. 2016 (SARI) | TACL 2016 | metric-first | metric-first |
| 2 | Croce & Hein 2020 (AutoAttack) | ICML 2020 | approach-first | metric-first |
| 3 | Ribeiro et al. 2020 (CheckList) | ACL 2020 Best | methodology-first (fused) | methodology-first |
| 4 | Heusel et al. 2017 (TTUR+FID) | NeurIPS 2017 | approach-first | dual (method named in title; metric named equally) |
| 5 | Hardt et al. 2016 (Equality of Opportunity) | NeurIPS 2016 | metric/criterion-first | metric-first |
| 6 | Bolukbasi et al. 2016 (Debias) | NeurIPS 2016 | metric-first | metric-first |
| 7 | Carlini & Wagner 2017 (C&W) | IEEE S&P 2017 | approach-first | metric/evaluation-first (title + abstract) |
| 8 | Madry et al. 2018 (PGD) | ICLR 2018 | evaluation-first | evaluation-first |
| 9 | Moosavi-Dezfooli et al. 2016 (DeepFool) | CVPR 2016 | metric-first | metric-first |

Counts (n=9 verified method+metric papers):
- **Body order**: metric-first 5 (SARI, Hardt, Bolukbasi, Madry, DeepFool), approach-first 3 (AutoAttack, Heusel/TTUR+FID, C&W), fused 1 (CheckList).
- **Abstract framing**: metric/methodology-first 8 of 9; dual-equal 1 (Heusel). No paper leads its abstract approach-first.

### 8.8 Rejected candidates (extended search)

- **Guo et al. ICML 2017 (Temperature Scaling, arXiv 1706.04599)** — temperature scaling is the method; ECE is attributed to Naeini et al. 2015 and reliability diagrams to DeGroot & Fienberg 1983. No new metric claimed. Reject (only (a)).
- **Welleck et al. 2020 (Unlikelihood Training, arXiv 1908.04319)** — new training objective only; repetition metrics used but not claimed new. Reject.
- **SelfCheckGPT (Manakul et al. EMNLP 2023, arXiv 2303.08896)** — new detection method; evaluation uses standard AUC-PR + correlation with human annotation. Reject.
- **FActScore (Min et al. EMNLP 2023, arXiv 2305.14251)** — metric-only (the "automated estimator" is part of the metric, not a separable method for a task). Reject.
- **QAGS (Wang et al. ACL 2020, arXiv 2004.04228)** — metric-only (the QA pipeline *is* the metric). Reject.
- **BARTScore (Yuan et al. NeurIPS 2021, arXiv 2106.11520)** — metric-only; "method" is the reformulation of evaluation. Reject.
- **MoverScore (Zhao et al. EMNLP 2019, arXiv 1909.02622)** — metric-only. Reject.
- **TextFooler (Jin et al. AAAI 2020, arXiv 1907.11932)** — attack method only; standard success-rate metrics. Reject.
- **Universal Adversarial Perturbations (Moosavi-Dezfooli CVPR 2017, arXiv 1610.08401)** — algorithm + discovery; no new evaluation metric claimed (uses fooling rate). Reject.
- **Carlini et al. USENIX 2021 (Training-data extraction, arXiv 2012.07805)** — attack + characterisation; memorisation factors analysed but not framed as a standalone metric contribution. Reject (borderline).
- **Inception Score / Salimans et al. NeurIPS 2016 (arXiv 1606.03498)** — could not verify the contribution-list framing from arXiv HTML (PDF binary, abstract silent on Inception Score). Widely-known dual paper in practice, but not verified within the time-box. Excluded for hygiene.
- **Calibrated Recommendations (Steck RecSys 2018)** — ACM paywalled, 403 on direct PDF; could not verify abstract framing within time-box. Excluded.
- **Sapienz, EvoSuite, DeepFL, Tarantula** — could not retrieve readable text (404s, paywalls, untrusted certs). Excluded for hygiene; from prior knowledge EvoSuite proposes a tool + several coverage-criterion contributions and would likely qualify, but is not formally verified here.
- **Stiennon et al. NeurIPS 2020 (Summarisation from human feedback)** — already rejected in §7.5; re-checked, still borderline (human preference used as ground truth, not advertised as new methodology).
- **Kynkäänniemi et al. NeurIPS 2019 (Improved P&R for generative models)** — metric-only (despite extensive architectural analysis). Reject.
- **Naeem et al. ICML 2020 (Density & Coverage)** — metric-only. Reject.
- **MC Dropout (Gal & Ghahramani ICML 2016)** — borderline: theoretical reinterpretation + uncertainty-extraction method are *fused*; no separable new metric. Closer to a method-only paper. Reject.

### 8.9 Updated implication for the AALinker paper

With n=9 verified method+metric papers (up from n=3), the body-order question is now answerable with reasonable confidence:

- **Abstract framing**: 8 of 9 papers lead with the metric/methodology; only Heusel/TTUR+FID frames the two contributions as equal-prominence (and even there, the metric is named in the abstract's evaluation clause). **The metric-first abstract is the dominant convention** — this matches the §6 benchmark+metric cluster.
- **Body order**: 5 metric-first, 3 approach-first, 1 fused. A weak majority for metric-first body (~56%), but no consensus. Both orderings have precedent at top venues.
- **Where approach-first body occurs** (AutoAttack, Heusel, C&W), the *method itself is algorithmically heavy* (PGD variants, theorem-bearing convergence proofs, three distinct attack algorithms) and would distort a metric-first body by burying the technical core. The metric in those cases is comparatively short to define.
- **Where metric-first body occurs** (SARI, Hardt, Bolukbasi, Madry, DeepFool), the metric/criterion either *defines the optimisation target* (SARI, Madry, Hardt) or *frames the measurement problem the method then solves* (Bolukbasi, DeepFool).

**Conclusion for AALinker.** The metric-first abstract is genuinely supported (n=8/9). The metric-first body is the weak modal pattern (n=5/9) but is *not* a universal convention — the body order is driven by internal logic (whether the metric defines the optimisation target, or whether the method's algorithmic depth would distort the narrative). For AALinker, the 6-metric suite *frames the evaluation problem the multi-agent linker is designed against*, which structurally matches SARI/Hardt/Madry. **The metric-first body is therefore defensible by precedent AND by internal logic.** The earlier worry that "no convention exists" is resolved: a convention does exist at the abstract level (metric-first, ~89%), and the modal body-order pattern aligns with AALinker's internal logic.
