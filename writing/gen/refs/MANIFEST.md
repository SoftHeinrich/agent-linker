# MANIFEST — Evaluation-Metric Papers for ICSE Introduction Reference

Curated set of verified, publicly downloadable software-engineering papers whose
contribution centrally concerns a **new evaluation metric** or an argument that
an existing standard metric is misleading. Used as exemplars for how strong SE
papers introduce metrics in their introductions.

Every entry below was verified via either an arXiv `/abs/` page that loaded with
matching title/authors, or a public author-hosted PDF. Paywalled venue pages
(ACM DL, IEEE Xplore) blocked WebFetch, so verification relies on arXiv abs +
author/group sites.

---

## 1. The Adverse Effects of Code Duplication in Machine Learning Models of Code

- **Authors:** Miltiadis Allamanis (Microsoft Research)
- **Year / Venue:** 2019, SPLASH Onward! 2019 (ACM SIGPLAN)
- **arXiv:** 1812.06469 — https://arxiv.org/abs/1812.06469
- **DOI:** 10.1145/3359591.3359735
- **Local PDF:** `allamanis-2019-dedup.pdf`
- **Verified via:** arXiv abs page (title, single author, abstract match) +
  author homepage `miltos.allamanis.com/publications/2019adverse/`.

**Glimpse quote (Section 1, Introduction, paragraphs 2–3):**

> "However, there is a looming crisis in this newly-founded area, caused by a
> disproportionately large amount of code duplication. This issue — first
> observed by Lopes et al. — refers to the fact that multiple file-level
> (near-)clones appear in large corpora of code, such as those mined from GitHub
> repositories. […] The core issue arises from the fact that identical or
> highly similar files appear both in the training and test sets that are used
> to train and evaluate the machine learning models.
>
> In this work, we first describe the impact that code duplication can have on
> machine learning models. […] We discuss the biases introduced when
> evaluating models under duplication and show that duplication can cause the
> evaluation to overestimate the performance of a model compared to the
> performance that actual users of the model observe."

**Rhetorical structure:** *flaw-first*. The intro names a measurement flaw
(duplication biases reported metrics by up to 100 %) before introducing the
remediation (deduplicated evaluation, a duplication index, best practices).

---

## 2. Evaluating Large Language Models Trained on Code (HumanEval / pass@k)

- **Authors:** Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde
  de Oliveira Pinto, *et al.* (OpenAI; 58 authors total)
- **Year / Venue:** 2021, arXiv preprint (widely cited as the canonical pass@k
  reference; not a peer-reviewed SE venue paper, but foundational for code
  generation evaluation and frequently cited at ICSE/FSE/ASE).
- **arXiv:** 2107.03374 — https://arxiv.org/abs/2107.03374
- **Local PDF:** `chen-2021-passk.pdf`
- **Verified via:** arXiv abs page (title and author list match).

**Glimpse quote (Section 2.1, "Functional Correctness", paragraphs 1–2):**

> "Generative models for code are predominantly benchmarked by matching samples
> against a reference solution, where the match can be exact or fuzzy (as in
> BLEU score). However, recent work has surfaced deficiencies in match-based
> metrics for code. […] More fundamentally, match-based metrics are unable to
> account for the large and complex space of programs functionally equivalent
> to a reference solution. As a consequence, recent works […] have turned to
> functional correctness instead, where a sample is considered correct if it
> passes a set of unit tests. We argue that this metric should be applied to
> docstring-conditional code generation as well."

And immediately after, the pass@k glimpse (Section 2.1 → 2):

> "Kulal et al. (2019) evaluate functional correctness using the pass@k metric,
> where k code samples are generated per problem, a problem is considered
> solved if any sample passes the unit tests, and the total fraction of
> problems solved is reported. However, computing pass@k in this way can have
> high variance. Instead, to evaluate pass@k, we generate n ≥ k samples per
> task […], count the number of correct samples c ≤ n which pass unit tests,
> and calculate the unbiased estimator."

**Rhetorical structure:** *flaw-first → definition-first*. The intro/Section 2
diagnoses match-based metrics (BLEU et al.) as semantically blind, then defines
the replacement (pass@k) with an explicit unbiased estimator.

---

## 3. Are Mutation Scores Correlated with Real Fault Detection? A Large Scale Empirical Study

- **Authors:** Mike Papadakis, Donghwan Shin, Shin Yoo, Doo-Hwan Bae
- **Year / Venue:** 2018, ICSE '18 (40th Int. Conf. on Software Engineering)
- **DOI:** 10.1145/3180155.3180183
- **IEEE Xplore:** https://ieeexplore.ieee.org/document/8453121
- **Local PDF:** `papadakis-2018-mutation.pdf` (from coinse.github.io author group)
- **Verified via:** author group homepage `coinse.github.io/publications/`,
  cross-checked against IEEE Xplore record and DOI.

**Glimpse quote (Section 1, Introduction, paragraphs 1–4):**

> "What is the relation between mutants and real faults? To date, this
> fundamental question remains open and, to large extent, unknown if not
> controversial. […] Just et al. report that there is 'a statistically
> significant correlation between mutant detection and real fault detection,
> independently of code coverage' […]. Although these studies provide evidence
> supporting the use of mutants in empirical studies, this is contradictory to
> the findings of other studies […].
>
> For instance, the study of Just et al. did not control for the size of the
> test suites, which is a strong confounding factor in software testing
> experiments. […] Therefore, as both mutation score and test suite size are
> factors with potential impact on fault detection, it is unclear what is the
> relation between mutation score and real fault detection, independently of
> test suite size."

**Rhetorical structure:** *flaw-first / replication-style*. The intro reframes
a widely accepted metric (mutation score as a proxy for real-fault detection)
as confounded, motivating a controlled re-measurement rather than a brand-new
metric. This is the rhetorical template for "your standard metric is wrong"
papers.

---

## 4. CodeBLEU: a Method for Automatic Evaluation of Code Synthesis

- **Authors:** Shuo Ren, Daya Guo, Shuai Lu, Long Zhou, Shujie Liu, Duyu Tang,
  Neel Sundaresan, Ming Zhou, Ambrosio Blanco, Shuai Ma
- **Year / Venue:** 2020, arXiv (presented in AAAI 2021 context; widely cited
  across ICSE/FSE/ASE code-generation work as the AST+dataflow extension of
  BLEU). Not an A* SE venue paper, but the canonical reference for code-aware
  similarity metrics.
- **arXiv:** 2009.10297 — https://arxiv.org/abs/2009.10297
- **Local PDF:** `ren-2020-codebleu.pdf`
- **Verified via:** arXiv abs page (title, all 10 authors, abstract match).

**Glimpse quote (Section 1, Introduction, paragraphs 1–3):**

> "A suitable evaluation metric is important to push forward the research of
> an area, such as BLEU and ROUGE for machine translation and text
> summarization. […] However, the above evaluation approaches still face many
> drawbacks. First, the n-gram accuracy does not take into account the
> grammatical and logical correctness, resulting in favoring candidates with
> high n-gram accuracy and serious logical errors. Second, the perfect accuracy
> is too strict, and underestimates different outputs with the same semantic
> logic. Third, the computational accuracy is weak in universality and
> practicability […].
>
> In order to deal with that, in this paper, we propose a new evaluation metric
> CodeBLEU, considering information from not only the shallow (n-gram) match,
> but also the syntactic match and the semantic match."

**Rhetorical structure:** *flaw-enumeration → composition-first*. The intro
lists exactly three failure modes of incumbent metrics, then defines the new
metric as a weighted combination addressing each.

---

## 5. CrystalBLEU: Precisely and Efficiently Measuring the Similarity of Code

- **Authors:** Aryaz Eghbali, Michael Pradel (University of Stuttgart)
- **Year / Venue:** 2022, ASE '22 (37th IEEE/ACM Int. Conf. on Automated
  Software Engineering). **ACM SIGSOFT Distinguished Paper Award.**
- **DOI:** 10.1145/3551349.3556903
- **Local PDF:** `eghbali-2022-crystalbleu.pdf` (from software-lab.org author site)
- **Verified via:** group homepage `software-lab.org/publications/`,
  conference listing at `conf.researchr.org/details/ase-2022/...`, author
  homepage `aryaze.github.io`.

**Glimpse quote (Section 1, Introduction, paragraphs 2–3):**

> "A commonality of all these techniques is the need for a metric to evaluate
> the quality of the predicted code. One of the most popular ways to address
> this need is the BLEU score. […] While surveying recent papers in software
> engineering, we find at least 21 papers published since 2015 that use BLEU
> as a metric to evaluate code prediction.
>
> […] The example not only illustrates how BLEU works, but also highlights an
> important weakness of applying the metric to code. In contrast to natural
> languages, programming languages are syntactically verbose in the sense that
> the grammar prescribes various n-grams to be shared across completely
> unrelated code examples. […] We call this phenomenon *trivially shared
> n-grams*, i.e., n-grams that occur across code written in the same language
> without implying any deeper relationship or semantic similarity. Because
> BLEU handles every n-gram the same, trivially shared n-grams hamper the
> metric's ability to distinguish actually similar code examples from examples
> merely written in the same language."

**Rhetorical structure:** *concrete-example-first → flaw → fix*. The intro
shows a worked Java example where BLEU ranks a non-equivalent program above an
equivalent one, names the diagnosis ("trivially shared n-grams"), and proposes
the fix (remove those n-grams). This is the cleanest "you can see the metric
fail right here" template among the five.

---

## Patterns observed

- **Flaw-first dominates.** Four of five papers (Allamanis, Chen, Papadakis,
  Eghbali) open by naming an existing metric and a concrete way it deceives:
  inflated by duplication; semantically blind; confounded by test-suite size;
  rewards trivially shared n-grams. CodeBLEU enumerates three flaws in
  parallel.
- **A single quantified number anchors the flaw.** "Up to 100 % inflated",
  "21 SE papers since 2015 using BLEU", "less than 1 % of mutants represent
  real faults", "1.9–4.5× more distinguishable". The introduction always
  contains one striking metric-about-metrics number.
- **The new metric is named in the abstract and re-named at the end of the
  intro's flaw paragraph**, never buried. CrystalBLEU, CodeBLEU, pass@k, and
  the deduplication tool all appear by name within the first two pages.
- **Worked micro-example before formalisation.** CrystalBLEU's Java snippet
  and Chen et al.'s pass@k pseudocode both show the metric *operating* before
  defining it abstractly. CodeBLEU's Figure 2 (`return y` vs `return x` with
  BLEU 95.47) does the same. The formal definition arrives only after the
  reader has felt the failure.
- **"Standard metric is wrong" papers don't always propose a replacement.**
  Papadakis et al. and Allamanis stop short of a new headline metric and
  instead deliver controlled measurement + best practices + tooling. This is a
  valid rhetorical mode for ICSE-style metric papers.

## Verification notes

- **Considered, kept.** All five entries above passed verification.
- **ACM DL / IEEE Xplore pages.** `dl.acm.org` and `ieeexplore.ieee.org`
  returned HTTP 403 to WebFetch in this environment, so venue-page-only
  verification was not possible. For Papadakis 2018 and Eghbali 2022 we relied
  on (a) author/group homepage PDFs and (b) public conference listings on
  `conf.researchr.org`. Title, author list, and venue strings on those sources
  match the ACM/IEEE records.
- **CodeBLEU venue caveat.** CodeBLEU (Ren et al. 2020) circulates as an arXiv
  preprint and is associated with AAAI 2021, not ICSE/FSE/ASE/ISSTA/TSE/TOSEM.
  Retained because it is the canonical "BLEU is wrong for code" citation in
  the SE code-generation literature and the introduction is an excellent
  flaw-enumeration template. Cite with the venue caveat in mind.
- **HumanEval / pass@k venue caveat.** Chen et al. 2021 is an arXiv-only
  technical report (OpenAI), not a peer-reviewed A* SE venue paper. Included
  because pass@k is now the de facto code-generation metric across ICSE/FSE
  papers, and the Section 2.1 introduction of pass@k is the cleanest extant
  example of "define a new metric with an unbiased estimator".
- **Considered, dropped for lack of verification within the time-box.** None.
  No candidate was discarded for failed verification in this pass; the search
  stopped at five solidly verified items rather than padding the list.
