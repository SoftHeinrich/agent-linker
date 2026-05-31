# MANIFEST-AIML

Verified AI / ML / NLP papers from top venues that introduce a new evaluation
metric (or critique an existing standard and replace it). Each entry is verified
via arXiv abs and/or ACL Anthology landing pages that loaded and matched.

Companion file to `MANIFEST.md` (SE-domain agent). Do not cross-edit.

---

## 1. BERTScore: Evaluating Text Generation with BERT

- **Authors:** Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, Yoav Artzi
- **Year:** 2019 (camera-ready 2020)
- **Venue:** ICLR 2020
- **arXiv:** 1904.09675 (v3, 24 Feb 2020)
- **Award status:** Highly cited but no formal ICLR award I could verify; *award status unverified*.
- **Local PDF:** `zhang-2020-bertscore.pdf`
- **Verbatim glimpse (Section 1, Introduction, paragraphs 1-2):**
  > "Automatic evaluation of natural language generation, for example in machine translation and caption generation, requires comparing candidate sentences to annotated references. The goal is to evaluate semantic equivalence. However, commonly used methods rely on surface-form similarity only. For example, BLEU (Papineni et al., 2002), the most common machine translation metric, simply counts n-gram overlap between the candidate and the reference. While this provides a simple and general measure, it fails to account for meaning-preserving lexical and compositional diversity.
  >
  > In this paper, we introduce BERTSCORE, a language generation evaluation metric based on pre-trained BERT contextual embeddings (Devlin et al., 2019). BERTSCORE computes the similarity of two sentences as a sum of cosine similarities between their tokens' embeddings. BERTSCORE addresses two common pitfalls in n-gram-based metrics..."
- **Rhetorical structure:** flaw-first (BLEU surface-form failure) -> proposal.

## 2. MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers

- **Authors:** Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, Zaid Harchaoui
- **Year:** 2021
- **Venue:** NeurIPS 2021 (Oral)
- **arXiv:** 2102.01454 (v3, 23 Nov 2021)
- **Award status:** Oral / Outstanding Paper Award at NeurIPS 2021 (widely reported; arXiv page confirms "Oral Presentation"). Formal "Outstanding Paper" badge *partially verified* via NeurIPS page references.
- **Local PDF:** `pillutla-2021-mauve.pdf`
- **Verbatim glimpse (Section 1, Introduction, paragraphs 2-3):**
  > "To evaluate how close a generation model's distribution is to that of human-written text, we must consider two types of errors: (I) where the model assigns high probability to sequences which do not resemble human-written text, and, (II) where the model distribution does not cover the human distribution, i.e., it fails to yield diverse samples. However, quantifying these aspects in a principled yet computationally tractable manner is challenging, as the text distributions are high-dimensional and discrete, accessed only through samples or expensive model evaluations.
  >
  > We develop MAUVE, a comparison measure for open-ended text generation. The proposed measure is efficient, interpretable, and practical for evaluating modern text generation models. It captures both types of errors (Figure 1) by building upon information divergence frontiers..."
- **Rhetorical structure:** problem-decomposition-first (two error types) -> proposal.

## 3. Beyond Accuracy: Behavioral Testing of NLP Models with CheckList

- **Authors:** Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, Sameer Singh
- **Year:** 2020
- **Venue:** ACL 2020
- **ACL Anthology:** 2020.acl-main.442
- **Award status:** **Best Overall Paper, ACL 2020** (verified via ACL Anthology metadata).
- **Local PDF:** `ribeiro-2020-checklist.pdf`
- **Verbatim glimpse (Section 1, Introduction, paragraphs 1-3):**
  > "One of the primary goals of training NLP models is generalization. Since testing 'in the wild' is expensive and does not allow for fast iterations, the standard paradigm for evaluation is using train-validation-test splits to estimate the accuracy of the model... While performance on held-out data is a useful indicator, held-out datasets are often not comprehensive, and contain the same biases as the training data, such that real-world performance may be overestimated. Further, by summarizing the performance as a single aggregate statistic, it becomes difficult to figure out where the model is failing, and how to fix it.
  >
  > ...Software engineering research, on the other hand, has proposed a variety of paradigms and tools for testing complex software systems. In particular, 'behavioral testing' (also known as black-box testing) is concerned with testing different capabilities of a system by validating the input-output behavior, without any knowledge of the internal structure...
  >
  > In this work, we propose CheckList, a new evaluation methodology and accompanying tool for comprehensive behavioral testing of NLP models."
- **Rhetorical structure:** flaw-first (held-out accuracy hides bugs) -> analogy (SE behavioral testing) -> proposal.

## 4. COMET: A Neural Framework for MT Evaluation

- **Authors:** Ricardo Rei, Craig Stewart, Ana C Farinha, Alon Lavie
- **Year:** 2020
- **Venue:** EMNLP 2020
- **ACL Anthology:** 2020.emnlp-main.213
- **Award status:** No best-paper award verified; widely adopted as a WMT shared-task winner. *Award status unverified.*
- **Local PDF:** `rei-2020-comet.pdf`
- **Verbatim glimpse (Section 1, Introduction, paragraphs 2-4):**
  > "Modern neural approaches to MT result in much higher quality of translation that often deviates from monotonic lexical transfer between languages. For this reason, it has become increasingly evident that we can no longer rely on metrics such as BLEU to provide an accurate estimate of the quality of MT.
  >
  > While an increased research interest in neural methods for training MT models and systems has resulted in a recent, dramatic improvement in MT quality, MT evaluation has fallen behind... The findings of the above-mentioned task highlight two major challenges to MT evaluation which we seek to address herein. Namely, that current metrics struggle to accurately correlate with human judgement at segment level and fail to adequately differentiate the highest performing MT systems.
  >
  > In this paper, we present COMET, a PyTorch-based framework for training highly multilingual and adaptable MT evaluation models that can function as metrics."
- **Rhetorical structure:** flaw-first (BLEU obsolete) + number-first (153 vs 24 submissions) -> proposal.

## 5. BLEURT: Learning Robust Metrics for Text Generation

- **Authors:** Thibault Sellam, Dipanjan Das, Ankur P. Parikh
- **Year:** 2020
- **Venue:** ACL 2020
- **arXiv:** 2004.04696 (v5, 21 May 2020)
- **Award status:** No award verified.
- **Local PDF:** `sellam-2020-bleurt.pdf`
- **Verbatim glimpse (Section 1, Introduction, paragraphs 3-4):**
  > "The first generation of metrics relied on hand-crafted rules that measure the surface similarity between the sentences. To illustrate, BLEU (Papineni et al., 2002) and ROUGE (Lin, 2004), two popular metrics, rely on N-gram overlap. Because those metrics are only sensitive to lexical variation, they cannot appropriately reward semantic or syntactic variations of a given reference. Thus, they have been repeatedly shown to correlate poorly with human judgment, in particular when all the systems to compare have a similar level of accuracy.
  >
  > Increasingly, NLG researchers have addressed those problems by injecting learned components in their metrics... Our insight is that it is possible to combine expressivity and robustness by pre-training a fully learned metric on large amounts of synthetic data, before fine-tuning it on human ratings. To this end, we introduce BLEURT, a text generation metric based on BERT."
- **Rhetorical structure:** flaw-first (BLEU/ROUGE correlate poorly) -> gap (expressivity vs robustness) -> proposal.

## 6. Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data

- **Authors:** Emily M. Bender, Alexander Koller
- **Year:** 2020
- **Venue:** ACL 2020
- **ACL Anthology:** 2020.acl-main.463
- **Award status:** **Best Theme Paper, ACL 2020** (widely reported).
- **Local PDF:** `bender-2020-nlu.pdf`
- **Verbatim glimpse (Section 1, Introduction, paragraphs 1-3):**
  > "The current state of affairs in NLP is that the large neural language models (LMs), such as BERT or GPT-2, are making great progress on a wide range of tasks, including those that are ostensibly meaning-sensitive. This has led to claims, in both academic and popular publications, that such models 'understand' or 'comprehend' natural language or learn its 'meaning'. From our perspective, these are overclaims caused by a misunderstanding of the relationship between linguistic form and meaning.
  >
  > We argue that the language modeling task, because it only uses form as training data, cannot in principle lead to learning of meaning...
  >
  > ...genuine progress in our field — climbing the right hill, not just the hill on whose slope we currently sit — depends on maintaining clarity around big picture notions such as meaning and understanding in task design and reporting of experimental results."
- **Rhetorical structure:** flaw-first / critique (the field is measuring the wrong thing), no replacement metric proposed — included as a *meta-critique* exemplar.

---

## Patterns observed

- **Flaw-first dominates.** 5 of 6 verified papers open by naming a specific deficiency of an entrenched metric (BLEU, ROUGE, held-out accuracy, BERT-as-understanding) before naming their contribution. The flaw is named within the first 1-2 paragraphs of the introduction.
- **A concrete failure example often appears before the formal definition.** BERTScore uses the "people like foreign cars" paraphrase; CheckList shows a 76.4% negation failure rate in a figure on page 2; COMET cites 153 vs 24 submissions to contrast MT progress with metric stagnation. Numbers and micro-examples carry rhetorical weight.
- **Software-engineering analogies recur.** CheckList explicitly imports "behavioral / black-box testing" from SE. This is the most ICSE-relevant rhetorical move: borrowing a mature SE testing concept to legitimize a new NLP evaluation methodology.
- **Correlation-with-human-judgment is the de facto meta-metric.** BERTScore, MAUVE, BLEURT, COMET all justify themselves primarily by showing higher correlation with human judgment than the incumbent — i.e., the new metric is validated *against* a gold standard (humans), not against a competing automatic metric on its own terms.
- **The proposal sentence is structurally identical across papers:** "In this paper / In this work, we introduce <NAME>, a <category> for <task>" — a near-formula. Useful for ICSE intro: an explicit naming sentence anchors the contribution.

## Verification notes

- **HumanEval / pass@k (Chen et al. 2021, arXiv 2107.03374):** another agent has already downloaded `chen-2021-passk.pdf` to the same `refs/` directory; not duplicated here. Verified to exist on arXiv but skipped per coordination rule.
- **HELM (Liang et al. 2022/2023):** not downloaded. Existence is well-known but I did not WebFetch the canonical Stanford CRFM / arXiv page within the time-box; *omitted to comply with the no-hallucination rule.*
- **Dynabench (Kiela et al., NAACL 2021):** not downloaded; not verified within time-box.
- **GLUE / SuperGLUE (Wang et al.):** not downloaded; not verified within time-box.
- **BIG-bench:** not downloaded; award status / metric focus not verified.
- **BLEURT URL hazard:** `aclanthology.org/2020.acl-main.463` initially looked like a plausible BLEURT URL but actually maps to Bender & Koller's "Climbing towards NLU". BLEURT was instead retrieved from arXiv (2004.04696). Logged here as a verification near-miss.
