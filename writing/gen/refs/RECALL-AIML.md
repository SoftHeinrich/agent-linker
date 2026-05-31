# RECALL-AIML.md — Pretraining-memory dump of dual-contribution (method + new metric) papers

Scope: papers I recall that propose BOTH (a) a method/model/algorithm for a task and (b) a new evaluation metric/methodology as a claimed contribution. Years 2014–2025, top venues. Drawn entirely from pretraining memory; no web access used.

---

## HIGH-confidence entries

### 1. Calibrated Recommendations
- **Title**: Calibrated Recommendations
- **Authors**: Harald Steck
- **Year + Venue**: RecSys 2018 (Best Paper)
- **Method contribution**: post-hoc reranking algorithm that calibrates a recommender's output distribution over genres/categories to match the user's historical interaction distribution, via a greedy submodular optimisation trading off relevance and calibration.
- **Metric contribution**: a calibration metric defined as the KL divergence between the user's historical genre distribution and the recommended-list genre distribution (C_KL).
- **Body order if recalled**: metric-first then method — Steck motivates miscalibration, formalises the divergence, then derives the reranker that minimises it.
- **Abstract framing**: problem (miscalibration) named first; metric defined; then the reranking method introduced as the fix.
- **Confidence**: HIGH

### 2. Unlikelihood Training for Neural Text Generation
- **Title**: Neural Text Generation With Unlikelihood Training
- **Authors**: Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, Jason Weston
- **Year + Venue**: ICLR 2020
- **Method contribution**: unlikelihood loss that explicitly penalises probability mass on undesired tokens (repeats) during training and at sequence level.
- **Metric contribution**: a suite of repetition/degeneration diagnostics — seq-rep-n, uniq, and the distinct-n-style framing — used as the primary evaluation axis alongside perplexity. (Note: distinct-n itself predates this; the paper's contribution is the consolidated seq-rep / uniq token/sequence-level repetition metrics.)
- **Body order**: approach-first (loss formulation) then evaluation suite.
- **Abstract framing**: method named first ("unlikelihood training"), metrics named as the lens that exposes the degeneration problem.
- **Confidence**: HIGH

### 3. Bolukbasi et al. — Word embedding debiasing
- **Title**: Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
- **Authors**: Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, Adam Kalai
- **Year + Venue**: NeurIPS 2016
- **Method contribution**: a geometric debiasing procedure ("hard-debias" / neutralise-and-equalise) that projects gender-neutral words off the identified gender subspace.
- **Metric contribution**: a direct-bias and indirect-bias measure on the gender subspace; analogy-based bias quantification ("she:homemaker :: he:?") used as a quantitative bias score.
- **Body order**: interleaved — define bias subspace + bias metric, then the debiasing operator.
- **Abstract framing**: problem (bias) framed first; method named "debiasing"; metric framed as the diagnostic that motivates and validates it.
- **Confidence**: HIGH

### 4. Equality of Opportunity in Supervised Learning
- **Title**: Equality of Opportunity in Supervised Learning
- **Authors**: Moritz Hardt, Eric Price, Nati Srebro
- **Year + Venue**: NeurIPS 2016
- **Method contribution**: post-processing algorithm that derives a group-wise thresholded predictor from any score function so as to satisfy equalised odds / equal opportunity.
- **Metric contribution**: the equal-opportunity and equalised-odds fairness criteria themselves, as new quantitative evaluation definitions (TPR/FPR-gap across protected groups).
- **Body order**: metric-first — the criterion is defined, characterised, then the post-processing construction follows.
- **Abstract framing**: criterion (metric) named first; method described as how to satisfy it.
- **Confidence**: HIGH

### 5. SelfCheckGPT
- **Title**: SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative LLMs
- **Authors**: Potsawee Manakul, Adian Liusie, Mark Gales
- **Year + Venue**: EMNLP 2023
- **Method contribution**: a sampling-based consistency procedure that detects hallucinated sentences by drawing multiple stochastic samples from the same LLM and measuring agreement with the original answer (via BERTScore, NLI, n-gram, or QA variants).
- **Metric contribution**: the SelfCheckGPT score itself, presented as a sentence-level factuality/hallucination metric and evaluated against human factuality annotations on WikiBio-GPT3.
- **Body order**: interleaved — the method *is* the metric; evaluated as a hallucination-detection score.
- **Abstract framing**: method named first; the score reported as both the detection mechanism and the new metric.
- **Confidence**: HIGH

### 6. FActScore
- **Title**: FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation
- **Authors**: Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, Hannaneh Hajishirzi
- **Year + Venue**: EMNLP 2023
- **Method contribution**: an atomic-fact decomposition + retrieval-grounded verification pipeline (decompose generation into atomic facts, look each up against a knowledge source, judge support).
- **Metric contribution**: FActScore — the fraction of supported atomic facts — as a new long-form factual-precision metric, with human and automated estimators.
- **Body order**: metric-first conceptually (definition of FActScore), then the estimator pipeline.
- **Abstract framing**: metric named first ("FActScore"); pipeline named as how to compute it.
- **Confidence**: HIGH

### 7. QAGS — Asking and Answering Questions to Evaluate Faithfulness
- **Title**: Asking and Answering Questions to Evaluate the Factual Consistency of Summaries
- **Authors**: Alex Wang, Kyunghyun Cho, Mike Lewis
- **Year + Venue**: ACL 2020
- **Method contribution**: a QA-based pipeline that generates questions from a summary, answers them against both the summary and the source, and compares answers — a generic faithfulness-evaluation procedure (also usable to rerank/select summaries).
- **Metric contribution**: QAGS score itself (answer-overlap across source vs. summary) as a new faithfulness metric, validated against human factuality judgments.
- **Body order**: approach-first then metric correlation studies.
- **Abstract framing**: method described first; metric introduced as its output.
- **Confidence**: HIGH

### 8. Carlini et al. — Extracting Training Data from Large Language Models
- **Title**: Extracting Training Data from Large Language Models
- **Authors**: Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Úlfar Erlingsson, Alina Oprea, Colin Raffel
- **Year + Venue**: USENIX Security 2021 (outside the listed venue set, flagging — `~venue` for this prompt)
- **Method contribution**: a training-data extraction attack that samples from the model and ranks candidates by a membership signal (zlib ratio, lowercase ratio, small-model perplexity ratio).
- **Metric contribution**: the *k*-eidetic memorisation definition (a string memorised if it appears in ≤k documents and can be extracted) — a new memorisation-evaluation framework.
- **Body order**: interleaved — attack steps, with the k-eidetic definition introduced to quantify success.
- **Abstract framing**: attack (method) named first; memorisation definition introduced to make the claim measurable.
- **Confidence**: HIGH (verify venue eligibility before citing for this prompt)

---

## MEDIUM-confidence entries

### 9. MMR — Maximal Marginal Relevance
- **Title**: The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries
- **Authors**: Jaime Carbonell, Jade Goldstein
- **Year + Venue**: SIGIR 1998 (outside 2014–2025 window — flagging)
- **Method contribution**: MMR reranking objective combining relevance and novelty.
- **Metric contribution**: the marginal-relevance score itself, framed as an evaluation/selection criterion for diversity.
- **Body order**: approach-first.
- **Confidence**: MEDIUM on framing-as-metric; HIGH on the method. Outside the requested year window — flag.

### 10. BLEURT (potential dual contribution)
- **Title**: BLEURT: Learning Robust Metrics for Text Generation
- **Authors**: Thibault Sellam, Dipanjan Das, Ankur Parikh
- **Year + Venue**: ACL 2020
- **Method contribution**: a pretraining-then-finetune *recipe* (synthetic perturbations → mid-training on millions of synthetic pairs → finetune on WMT) for learned metrics.
- **Metric contribution**: BLEURT itself.
- **Body order**: metric-first.
- **Note**: This is borderline — the "method" here is a training procedure for a metric, not for a task. If "method for a task" is read strictly, BLEURT does not qualify. Listed as MEDIUM for that reason.
- **Confidence**: MEDIUM (eligibility), HIGH on facts.

### 11. COMET
- **Title**: COMET: A Neural Framework for MT Evaluation
- **Authors**: Ricardo Rei, Craig Stewart, Ana Farinha, Alon Lavie
- **Year + Venue**: EMNLP 2020
- **Method contribution**: a regression / ranking architecture over cross-lingual encoders (XLM-R) with source + hypothesis + reference, plus a triplet-margin training variant.
- **Metric contribution**: the COMET score.
- **Note**: same caveat — the "method" is for building the metric, not for an MT system. MEDIUM for eligibility.
- **Confidence**: MEDIUM.

### 12. Fairness-aware reranking (FA*IR)
- **Title**: FA*IR: A Fair Top-k Ranking Algorithm
- **Authors**: Meike Zehlike, Francesco Bonchi, Carlos Castillo, Sara Hajian, Mohamed Megahed, Ricardo Baeza-Yates
- **Year + Venue**: CIKM 2017 (`~venue` — outside the listed set)
- **Method contribution**: a fair top-k reranking algorithm enforcing a minimum proportion of protected-group items at every prefix.
- **Metric contribution**: a statistical-test-based fairness criterion on prefix proportions.
- **Confidence**: MEDIUM. Verify venue/year before citing.

### 13. CheckList
- **Title**: Beyond Accuracy: Behavioral Testing of NLP Models with CheckList
- **Authors**: Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, Sameer Singh
- **Year + Venue**: ACL 2020 (Best Paper)
- **Method contribution**: the CheckList *methodology* — templates, MFT/INV/DIR test types, perturbation tooling.
- **Metric contribution**: per-capability failure-rate as the new evaluation methodology (not a scalar score).
- **Note**: This is essentially an evaluation-methodology paper; (a) is arguably a "tool for testing", not a method that solves a downstream task. Listed for completeness, but probably **rejects** under strict reading.
- **Confidence**: MEDIUM eligibility.

### 14. MAUVE
- **Title**: MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers
- **Authors**: Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, Zaid Harchaoui
- **Year + Venue**: NeurIPS 2021 (Outstanding Paper)
- **Method contribution**: divergence-frontier estimation procedure using quantised embedding clusters + KL frontier integration — an algorithm for computing the metric.
- **Metric contribution**: MAUVE itself.
- **Note**: same eligibility caveat as BLEURT/COMET.
- **Confidence**: MEDIUM eligibility.

---

## Rejections from memory (look like candidates, but the metric is borrowed or the method is absent)

- **SelfCheckGPT extensions / G-Eval / GPTScore** — propose a *metric* using an LLM-as-judge but no new task-solving method. Reject.
- **TruthfulQA (Lin et al. ACL 2022)** — benchmark + metric, no new task-solving model. Reject (also explicitly excluded by prompt).
- **HELM / MT-Bench / SWE-bench / BIG-bench / BEIR** — benchmark frameworks. Reject (explicitly excluded).
- **BERTScore (Zhang et al. ICLR 2020)** — metric only, no paired task method. Reject.
- **BLEURT / COMET / MAUVE** — see Medium block: arguably metric-only papers. Reject under strict reading.
- **Pointer-Generator Networks (See et al. ACL 2017)** — new summarisation method but evaluated with ROUGE; no new metric. Reject.
- **Transformer (Vaswani et al. NeurIPS 2017)** — new method, BLEU borrowed. Reject.
- **GPT-3 (Brown et al. NeurIPS 2020)** — new method, borrowed metrics. Reject.
- **AlphaCode / Codex** — Codex paper (Chen et al. 2021) *does* introduce pass@k as a metric paired with the code-generation method — this is actually a candidate, not a reject. Promote: see below.
- **DPO (Rafailov et al. NeurIPS 2023)** — new training method, borrowed win-rate eval. Reject.
- **RLHF / InstructGPT (Ouyang 2022)** — new training method, human pref eval borrowed. Reject.
- **AutoAttack (Croce & Hein ICML 2020)** — already verified upstream, not relisted.

### Promote from rejects

- **Codex / pass@k**: Chen et al. 2021, "Evaluating Large Language Models Trained on Code" — method: Codex (code-LM finetune); metric: pass@k unbiased estimator. Body order: method-first then metric. HIGH confidence on facts; venue is arXiv/tech report (not in the listed venue set) — flag.

---

## Pattern from memory

Across the HIGH-confidence list, body order leans **metric-first or interleaved** when the *metric is the headline contribution* (Calibrated Recommendations, Equality of Opportunity, FActScore) and **method-first** when the *method is the headline* and the metric is introduced as a diagnostic that justifies or quantifies the method (Unlikelihood Training, QAGS, Carlini extraction, Bolukbasi debiasing).

Concrete examples from this list:
- Metric-first: Calibrated Recommendations (KL calibration defined, then reranker); Hardt et al. (criterion defined, then post-processing); FActScore (score defined, then estimator).
- Method-first: Unlikelihood Training (loss first, repetition metrics as evidence); QAGS (pipeline first, score as output); Carlini (attack first, k-eidetic definition to quantify).
- Interleaved: Bolukbasi (subspace + bias metric + debias operator co-developed); SelfCheckGPT (the method is the metric).

So the honest answer: **genuinely mixed**, driven by which contribution the authors foreground. I do not have a stable single-direction memory of the pattern; the framing follows the headline.
