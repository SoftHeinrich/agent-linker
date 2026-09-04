# Literature survey: obtaining semantic, scalable gold (or usefulness signals) for documentation-to-architecture TLR at OSS scale

Context: ALinker links sentences of an architecture/developer document to components. Benchmarks so far: five small projects (13-198 sentences, 6-14 components) with human gold (ARDoCo benchmark). Target: rustc-dev-guide (1,762-6,522 sentences) x 79 compiler crates. Hand labelling is infeasible; syntactic anchors (markdown hyperlinks to a crate's rustdoc page, verbatim crate names) were rejected as "regex-based, not semantic". This file surveys how related work obtained free supervision, how it validated it, and what that implies for us.

Conventions. Each entry: citation (venue, year, DOI/arXiv) -- method -- free signal -- validation against humans -- reported numbers -- applicability. Items I could not confirm from a primary source are tagged [unverified]. Searches ran 2026-09-04 (about 60 queries, abstracts/HTML read where accessible; Springer and ACM pages were paywalled, so a few details come from mirrors/arXiv).

Reading of the supervisor's objection in the vocabulary of this literature: the rejected anchors are *distant supervision* (Mintz et al., ACL 2009; Wikipedia anchor text for entity linking). NLP accepts that as *training* signal, never as the *test set*, because it inherits selection bias: only the easy, explicitly-marked cases get labelled. SE has the same finding under other names -- "linkage bias" in bug-fix datasets (Bird et al. 2009), 44 % wrong mappings in a bug-localisation benchmark built from linked commits (Kim & Lee 2021), 50 %+ performance collapse when issue-commit benchmarks are given realistic candidate sets (Huang et al. 2025). The literature's consistent answer is: keep the cheap signal as *one* noisy source, add other independent sources, combine them with a label model, and validate the combined silver labels against a *stratified* human sample sized by a published recipe.

---

## 1. Self-/weakly-/distantly-supervised gold for TLR and related SE linking tasks

### 1.1 Issue-commit link recovery (explicit issue keys as gold)

**FRLink** -- Sun, Wang, Yang. *Information and Software Technology* 84, 2017, pp. 33-47. DOI 10.1016/j.infsof.2016.11.008 [DOI unverified].
Method: classifier over issue/commit metadata + textual and code similarity, filters irrelevant source and adds non-source documents as a feature. Free signal: Jira issue keys in commit messages define positives; unlinked pairs are negatives. Human validation: none; explicit links are treated as truth. Numbers: F-measure +40.75 % over RCLinker. Applicability: template for "explicit reference = positive" gold; shows the whole subfield accepted a syntactic anchor as gold, which is precisely what was rejected for us.

**DeepLink (Ruan et al.)** -- *Journal of Systems and Software* 158, 2019. DOI 10.1016/j.jss.2019.110406. RNN + word embeddings over issue title/description and commit message/diff. Same gold as FRLink. **DeepLink (Xie et al.)** -- SANER 2019 [venue unverified], code knowledge graph from ASTs + RNN/SVM. Same gold.

**T-BERT** -- Lin, Liu, Zeng, Jiang, Cleland-Huang. ICSE 2021. DOI 10.1109/ICSE43902.2021.00040; arXiv 2102.04411.
Method: three-stage transfer -- pretrain on CodeSearchNet docstring-code pairs, intermediate training, fine-tune on issue-commit links; Single-, Twin-, Siamese-BERT. Free signal: (a) docstring-code pairing as pretraining, (b) GitHub issue-PR/commit links created by "closes #N" keywords as gold. Human validation: none. Numbers: +60.31 % MAP over VSM on three OSS projects. Applicability: the docstring<->code pairing is the closest analogue to "crate-level rustdoc <-> code"; the paper also shows that a *large noisy* related task can bootstrap a *small clean* one, which is the direction we would go (use the 5-project human gold as the clean anchor).

**Rath, Rendall, Guo, Cleland-Huang, Mader -- "Traceability in the wild"** -- ICSE 2018. DOI 10.1145/3180155.3180207; arXiv 1804.02433.
Method: classifier on process (time, committer/assignee), text, and stakeholder features to recover *missing* issue tags. Free signal: issue IDs in commit messages; finding that only ~60 % of commits in six OSS projects are tagged. Human validation: none reported in abstract; evaluation is hold-out on tagged commits. Numbers: ~96 % recall / 33 % precision for missing-link detection; >89 % precision / 50 % recall when augmenting existing links. Applicability: explicitly quantifies how incomplete anchor-based gold is (40 % missing), so recall computed against anchors is meaningless; we would have the same 40 %-style hole.

**EALink** -- ASE 2023. DOI 10.1109/ASE56229.2023.00059. Knowledge-distilled CodeBERT for issue-commit linking; gold = Jira links in Apache projects; +15-409 % over baselines; no human validation. **MTLink** (JKSU-CIS 2024), **MPLinker** (arXiv 2501.19026), **LinkAnchor** (PACMSE 2026, DOI 10.1145/3808191, arXiv 2508.12232; LLM agent) -- same gold construction.
**PLink** -- I could not locate a paper under this name in issue-commit linking [unverified]; possibly a confusion with "Ruan et al. DeepLink".

**Huang et al. -- "Back to the Basics: Rethinking Issue-Commit Linking with LLM-Assisted Retrieval"** -- arXiv 2507.09199 (2025; 11 authors incl. D. Lo).
Method: audit of the evaluation protocol; proposes a Realistic Distribution Setting (RDS) with the full commit population as candidates; EasyLink = vector retrieval + LLM rerank. Finding: SOTA deep models lose >50 % under RDS; EasyLink P@1 ~75 % across 20 projects. **Morgan et al. -- "Think Harder and Don't Overlook Your Options"** -- arXiv 2605.00447 (2026): dense retrieval beats sparse for candidate generation; classic ML rerankers beat LLM rerankers. Applicability: warns that a gold set built from anchors also fixes an unrealistically small *candidate* set (only sentences with anchors); our evaluation must include the unanchored population, which is where our FNs live.

**Bird, Bachmann, Aune, Duffy, Bernstein, Filkov, Devanbu -- "Fair and balanced? Bias in bug-fix datasets"** -- ESEC/FSE 2009 (Test-of-Time award). DOI 10.1145/1595696.1595716.
Finding: commit-message-based bug links are a biased sample of all fixes (severity, developer experience) [specific bias dimensions from memory; unverified here]. Applicability: the canonical citation for why anchor-based gold is not just incomplete but *systematically* skewed; must be cited when we use hyperlinks as a labelling function.

### 1.2 Version-history mining as traceability evidence

**Kagdi, Maletic, Sharif -- "Mining software repositories for traceability links"** -- ICPC 2007, pp. 145-154. (Read from full PDF.)
Method: sequential-pattern mining over Subversion change-sets of KDE, grouped by three heuristics (time interval, committer, committer+day); mines ordered co-change patterns *across artifact types* (e.g. `kalzium.cpp` -> `kdesvn-build/index.docbook`). Free signal: co-commit of documentation and source files. Validation: no human labels; temporal hold-out -- 8 months training (13,037 files), 4 months evaluation; metrics coverage/recall/precision of predicted patterns. Numbers: committer+day heuristic gives highest precision (per-pattern precision tables; overall "high precision, low recall"). Applicability: the earliest doc<->code co-change TLR paper; directly transferable to `rustc-dev-guide/src/*.md` <-> `compiler/rustc_*` co-changes in rust-lang/rust PRs; also the template for a *temporal* validation that needs no human labels.

**Ali, Gueheneuc, Antoniol -- Trustrace** -- *IEEE TSE* 39(5):725-741, 2013. DOI 10.1109/TSE.2012.71.
Method: IR (VSM/JSM) requirement-to-code links re-weighted by a trust model whose "experts" are CVS/SVN logs and bug reports (Histrace). Free signal: commits and bug-report-linked changes as independent evidence for a link. Validation: against manual traceability oracles of the case studies. Numbers: up to +22.7 % precision, +7.66 % recall on average. Applicability: the canonical "combine IR with history evidence" TLR paper; our label model should treat co-change as exactly this kind of expert.

**Kagdi, Gethers, Poshyvanyk -- conceptual + evolutionary couplings for change impact analysis** -- WCRE 2010; extended *EMSE* 18(5) 2013, DOI 10.1007/s10664-012-9233-9.
Method: IR-based conceptual coupling blended with co-change (evolutionary) coupling. Gold: actual change sets in later commits of httpd, ArgoUML, iBatis, KOffice. Numbers: statistically significant accuracy gains over either alone across cut points. Applicability: shows how commit history can serve as *gold for a downstream task* (what else changes) rather than as gold for links.

**Hata, Treude, Kula, Ishio -- "9.6 million links in source code comments"** -- ICSE 2019, DOI 10.1109/ICSE.2019.00123, arXiv 1901.07440; and "18 million links in commit messages" -- *EMSE* 2023, DOI 10.1007/s10664-023-10325-8.
Method: large-scale mining of URLs in comments/commit messages; mixed-methods classification of link targets/purposes; decay analysis. Free signal: hyperlinks as explicit traceability. Validation: manual coding of samples by the authors. Applicability: hyperlinks *are* accepted as trace links in MSR, but the papers document decay and one-directionality -- exactly the weaknesses of our anchor gold; useful to cite when arguing anchors are a noisy *source*, not truth.

### 1.3 API-mention recognition and linking (entity linking against a code knowledge base)

**Dagenais, Robillard -- RecoDoc** -- ICSE 2012, ACM 2337223.2337230.
Method: links code-like terms in documentation, tutorials and mailing lists to API elements by resolving partial names using code context and heuristics. Free signal: the API's own type system as knowledge base; no training labels. Validation: manual assessment of link samples in four OSS systems (high precision, >90 % [exact figures unverified]). Applicability: the "semantic anchor" upgrade path -- resolve a *mention* (`TyCtxt`, "the type context", "the borrow checker") to an item, then item -> crate via the symbol index; the sentence need not name the crate.

**Ye, Xing et al. -- APIReal** -- *EMSE* 23(6), 2018, DOI 10.1007/s10664-018-9608-7.
Method: semi-supervised CRF using name synonyms and context for API *recognition* in Stack Overflow text; linking by mention-mention similarity, scope filtering, mention-entry similarity. Validation: 1,205 manually annotated API mentions (pandas/numpy/matplotlib). Applicability: shows how to handle aliases/abbreviations of code entities in prose without hand lists (synonym discovery from data) -- compatible with the no-hardcoded-lists rule.

**Huo et al. -- ARCLIN** -- ICSE 2022, DOI 10.1145/3510003.3510158.
Method: BiLSTM-CRF API recogniser trained *without human annotations* (distant labels from API knowledge base), plus context-aware mention-entry scoring for linking. Validation: human-labelled test set. Applicability: a fully unsupervised mention->entity pipeline, i.e. evidence that "semantic anchoring" (resolving informal mentions) is achievable at scale and is a recognised step above regex.

**Rigby, Robillard -- "Discovering essential code elements in informal documentation"** -- ICSE 2013. Related: which code elements in a document are salient vs incidental; relevant to filtering "mentions in passing".

### 1.4 Docstring/code pairing and code-search datasets (silver at scale + small expert gold)

**Husain, Wu, Gazit, Allamanis, Brockschmidt -- CodeSearchNet** -- arXiv 1909.09436 (2019).
Method: 2 M (docstring, function) pairs scraped mechanically across six languages; the *benchmark* is 99 natural-language queries with ~4k expert relevance judgements on a 0-3 scale. Free signal: docstring-code co-location. Validation: silver only for training; evaluation on the small expert-annotated set. Applicability: the standard pattern "huge silver for training, small gold for evaluation" -- for us, crate-level `//!` docs <-> crate is the silver pairing; a few hundred expert-judged (sentence, crate) pairs is the gold.

**Sun, Li, Liu, Du, Li -- "On the importance of building high-quality training datasets for neural code search"** -- ICSE 2022, DOI 10.1145/3510003.3510160, arXiv 2202.06649.
Finding: >1/3 of CodeSearchNet queries are noisy; a rule-based + model-based filter (NLQF) yields +19.2 % MRR. Applicability: even the accepted silver pairing needs a two-stage filter; expect the same for guide-sentence <-> crate-doc alignment.

**Bench4BL / IR bug-localisation datasets** -- Lee, Kim, Bissyande, Le Traon: Bench4BL, ISSTA 2018, DOI 10.1145/3213846.3213856; Kim, Lee: "Are datasets for IR-based bug localization techniques trustworthy?", *EMSE* 2021, DOI 10.1007/s10664-021-09946-8.
Gold: files changed in commits that reference a bug ID. Finding: up to 44 % incorrect bug-file mappings (tangled changes, misclassified reports). Applicability: quantitative precedent for how wrong commit-derived gold can be; a co-change labelling function for us must be down-weighted accordingly and its accuracy estimated, not assumed.

**Lin, Poudel, Yu, Zeng, Jiang, Cleland-Huang -- NLTrace, "transfer learning from open-world data"** -- arXiv 2207.01084 (2022).
Method: pretrain trace models on issue-commit links mined from GitHub 2016-21, then transfer; +20 % MAP. Applicability: open-world noisy links as pretraining only, evaluation still on curated sets.

**Zhou et al. -- DocPrompting** -- ICLR 2023, arXiv 2207.05987. Retrieves documentation for code generation; introduces tldr benchmark. Applicability: shows docs->code retrieval evaluated by *execution* (a downstream objective truth) rather than link gold; not directly transferable but a reminder that an objective downstream metric can replace link labels.

### 1.5 Architecture-documentation traceability (our own lineage)

**SWATTR** -- Keim, Schulz, Fuchss, Kocher, Speit, Koziolek, ECSA 2021, DOI 10.1007/978-3-030-86044-8_7. Heuristic SAD->SAM TLR; human gold.
**Benchmark** -- Fuchss, Corallo, Keim, Speit, Koziolek, "Establishing a benchmark dataset for TLR between SAD and SAM", MSR4SA @ ECSA 2022, DOI 10.1007/978-3-031-36889-9_30; repo github.com/ArDoCo/Benchmark. Five projects; per ArTEMiS (arXiv 2511.02434) table: MediaStore 37 sentences / 14 components / 31 SAD-SAM links; TeaStore 43 / 11 / 27; TEAMMATES 198 / 8 / 57; BigBlueButton 87 / 12 / 62; JabRef 13 / 6 / 18. Gold created manually by the authors; inter-annotator agreement not reported in the pages I could access [unverified].
**ArDoCo inconsistency detection** -- Keim, Corallo, Fuchss, Koziolek, ICSA 2023 (Zenodo 7649370). Uses TLR to find missing model elements; gold = manual.
**TransArC** -- Fuchss, Hey, Keim, Liu, Ewald, Thirolf, Koziolek, ICSE 2024, DOI 10.1145/3597503.3639130. Transitive SAD->SAM->code; SAM-code F1 0.98, SAD-code F1 0.82; gold manual (SAD-code links 59 / 707 / 8,097 / 1,529 / 8,268).
**LiSSA** -- Fuchss et al., ICSE 2025, DOI 10.1109/ICSE55347.2025.00186. RAG-based generic TLR; evaluated on the same human gold.
**ExArch** -- Fuchss et al., ICSA 2025 (fuchss.org/assets/pdf/2025/icsa-25.pdf). LLM extracts component names from SAD + code to build a SAM when none exists; evaluated by *manually matching* extracted components to the real SAM.
**ArTEMiS** -- Fuchss, Liu, Corallo, Hey, Keim, von Geisau, Koziolek, *ACM TAAS* 2025, DOI 10.1145/3807453, arXiv 2511.02434. NER-style architecture-entity extraction + matching; GPT-5 F1 0.81 vs SWATTR 0.80 on SAD-SAM (recall 0.85 vs 0.77).
Applicability: all evaluation in this lineage is on <=198-sentence human gold; none has scaled. ExArch's "LLM-derived SAM, manually matched" is the closest precedent for treating a *code-derived* decomposition (our 79 crates) as the model.

**Alor, Khatoonabadi, Shihab -- "Evaluating the use of LLMs for documentation to code traceability"** -- arXiv 2506.16440 (2025). (Read from HTML.)
Method: Claude 3.5 Sonnet / GPT-4o / o3-mini link Markdown documentation segments to classes/methods/attributes in Unity Catalog and Crawl4AI. Gold: two annotators, five-step protocol, Cohen kappa 0.94; Crawl4AI 112 segments / 29 code artifacts / 645 links; Unity Catalog 32 / 76 / 155. Numbers: best F1 79.4 % and 80.4 %; fully correct explanations 42.9-71.1 %. Error taxonomy of false positives: implicit-assumption errors 51.9-98.2 %, phantom links 0-30.7 %, architecture-pattern bias 0-32.6 %, implementation over-link 0-47.4 %. Applicability: the only recent doc->code LLM TLR paper with a *reported* annotation protocol and kappa; its FP taxonomy matches our own error analysis (sibling-component confusion = "architecture pattern bias"), and its two-annotator kappa 0.94 is a target for our validation sample.

**Moran et al. -- Comet** -- ICSE 2020, DOI 10.1145/3377811.3380418. Hierarchical Bayesian network combining several textual similarity measures, developer feedback and transitive relations into a link probability; practitioner survey at Cisco. Applicability: an in-SE precedent for a probabilistic *label-model-like* combination of heterogeneous link evidence.

**Nishikawa et al. -- Connecting Links Method** -- ICSME 2015 ERA. Transitive links via a third artifact. Applicability: precedent for using crate rustdoc as the intermediate artifact between guide sentence and crate.

**Hey, Chen, Weigelt, Tichy -- FTLR** -- ICSME 2021, DOI 10.1109/ICSME52107.2021.00008. Fine-grained requirement-to-code relations with word embeddings; evaluated on classic human-gold datasets (eTOUR, iTrust, SMOS, eANCI, LibEST) [dataset list from memory]. Applicability: none for gold construction; cited as the state of the art the ARDoCo line compares against.

**Distant supervision origins** -- Mintz, Bills, Snow, Jurafsky, ACL 2009 (relation extraction from Freebase); Wikipedia-anchor entity linking (Fan et al., arXiv 1505.03823; Botha et al., EMNLP 2020, 684 M anchor mentions in 104 languages). Applicability: the theoretical home of "hyperlink as label"; note that these works always evaluate on human-annotated test sets (e.g. AIDA-CoNLL), never on anchors.

---

## 2. LLM-as-annotator, silver labels with human validation, and weak-supervision frameworks

### 2.1 Agreement studies (NLP / social science)

**Gilardi, Alizadeh, Kubli -- "ChatGPT outperforms crowd workers for text-annotation tasks"** -- *PNAS* 120(30), 2023, DOI 10.1073/pnas.2305016120, arXiv 2303.15056.
n = 6,183 tweets/news; tasks relevance, stance, topic, frame. Gold: trained annotators. Numbers: zero-shot accuracy ~25 pp above MTurk; ChatGPT intercoder agreement exceeds both crowd and trained annotators; cost <$0.003/item. Caveat: agreement was measured against a *trained-annotator* gold, not claimed to replace it.

**Tornberg -- GPT-4 vs experts and crowd on political tweets** -- arXiv 2304.06588 (2023); journal version *Social Science Computer Review* 2025, DOI 10.1177/08944393241286471.
Trick worth copying: ground truth is *external and objective* (the poster's actual party), so LLM, experts and crowd are all scored against something no annotator produced. GPT-4: higher accuracy and reliability, equal or lower bias than humans.

**Ziems et al. -- "Can LLMs transform computational social science?"** -- *Computational Linguistics* 50(1):237-291, 2024, arXiv 2305.03514.
13 models x 25 benchmarks. Numbers: moderate-to-good agreement (kappa 0.40-0.65) on 8/17 classification tasks; stance F1 76.0, kappa 0.58. Recommendation: LLMs as annotation *assistants*, not replacements.

**Pangakis, Wolken, Fasching -- "Automated annotation with generative AI requires validation"** -- arXiv 2306.00176 (2023). (Read from PDF.)
27 tasks / 11 datasets replicated with GPT-4. Numbers: median accuracy 0.850, median F1 0.707; 9/27 tasks had precision or recall <0.5; within one dataset F1 ranged 0.259-0.811 across tasks; recall generally > precision. Recipe: codebook -> at least two subject-matter experts and the LLM label the *same* random sample of 250-1,250 texts -> compute accuracy/P/R/F1 per task -> iterate the codebook at most once -> choose a use case (full automation, LLM-first with human review of low-confidence, LLM data to fine-tune a classifier, or abandon). Consistency score: classify each item >= 3 times at temperature >0; proportion matching the mode; strongly correlated with correctness; 85.1 % of items were fully consistent. Applicability: the most concrete published sample-size and workflow recipe; directly usable for our validation set.

**Tan et al. -- "LLMs for data annotation and synthesis: a survey"** -- EMNLP 2024, arXiv 2402.13446. Taxonomy: LLM-based annotation, assessing LLM annotations, learning with LLM annotations. Use as the umbrella citation.

**Calderon, Reichart, Dror -- "The alternative annotator test (alt-test)"** -- arXiv 2501.10970 (2025) [venue unverified]. (Read from HTML.)
Procedure: for each human annotator j, leave j out, compute the LLM's alignment with the remaining annotators vs j's own alignment with them; advantage probability rho_j; cost-benefit epsilon (0.2 experts, 0.15 skilled, 0.1 crowd) penalises humans; paired t-test per annotator with Benjamini-Yekutieli FDR 0.05; the LLM "wins" an annotator if H0 rejected; winning rate omega >= 0.5 -> LLM may replace humans. Requirements: >= 3 human annotators, >= 30 instances (50-100 typical). Limits: assumes gold ~ annotator consensus; weak with high human disagreement; contamination unaddressed. Applicability: gives us a *statistical* justification statement ("LLM annotations pass the alt-test against three annotators on n = 100 stratified pairs") rather than a bare kappa.

### 2.2 Agreement studies in software engineering

**Ahmed, Devanbu, Treude, Pradel -- "Can LLMs replace manual annotation of SE artifacts?"** -- MSR 2025 (ACM SIGSOFT Distinguished Paper), arXiv 2408.05534. (Read from HTML.)
10 tasks / 5 datasets (code-summary quality ratings x4, name-value inconsistency, causality in requirements, semantic similarity x3, static-analysis-warning resolution); six LLMs (GPT-4, GPT-3.5, Claude-3.5-Sonnet, Gemini-1.5-Pro, Llama3-70B, Mixtral-8x22B). Metric: Krippendorff's alpha. Human-human vs human-model alpha: code accuracy 0.38 vs 0.48; name-value 0.52 vs 0.49; semantic similarity (goals) 0.83 vs 0.77; static-analysis warnings 0.80 vs 0.15; causality 0.44 vs 0.22. Model-model alpha 0.39-0.83 and correlates with human-model alpha (Spearman 0.65, p<0.05); proposed gate: model-model alpha > 0.5 => task suitable. Confidence gating (GPT-4 token probabilities): 50-100 % of items can be delegated without significantly changing agreement on suitable tasks; overall 9-33 % human-effort saving. Threats: 10 tasks; possible contamination; single-rater replacement only. Applicability: gives us two pre-registerable gates that need *no* human labels up front (model-model alpha across families; confidence threshold) and the human-labelled subset to confirm.

**Wang, Guo, Gao, Fan, Chong, Xia -- "Can LLMs replace human evaluators? LLM-as-a-judge in SE"** -- ISSTA 2025, PACMSE 2(ISSTA) art. 86, arXiv 2502.06193. Output-based judging: Pearson 81.32 (code translation) and 68.51 (generation) with human scores vs ChrF++ 34.23/64.92. Applicability: SE-specific evidence that direct-judgement prompts beat pairwise/score-decomposition variants.

**"LLM-as-a-Judge for SE: literature review, vision, road ahead"** -- *ACM TOSEM* 2025/26, DOI 10.1145/3797276, arXiv 2510.24367. Names *circularity* (same or similar LLM generating and judging, possibly incidentally) as a validity threat; also test-retest reliability and prompt-sensitive bias. **"Bias in the loop: auditing LLM-as-a-judge for SE"** -- arXiv 2604.16790 (2026): audits explicit/implicit prompt biases. Applicability: cite when arguing why the judge model families must be disjoint from the linker's.

**De Martino, Castano, Palomba, Franch, Martinez-Fernandez -- PRIMES 2.0, "A methodological framework for LLM-based MSR"** -- arXiv 2508.02233 (2025). Six stages / 23 substeps mapped to 9 threats and 25 mitigations. Applicability: checklist citation for reporting our LLM-labelling pipeline.

### 2.3 Judge biases (why single-LLM gold is suspect)

**Zheng et al. -- "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena"** -- NeurIPS 2023 D&B, arXiv 2306.05685. GPT-4 agrees with humans at roughly the human-human rate in aggregate, but shows position bias, verbosity bias and self-enhancement bias.
**Panickssery, Bowman, Feng -- "LLM evaluators recognize and favor their own generations"** -- NeurIPS 2024, arXiv 2404.13076. Self-recognition accuracy correlates linearly with self-preference strength. Follow-ups: "Self-preference bias in LLM-as-a-judge" (arXiv 2410.21819), "Do LLM evaluators prefer themselves for a reason?" (arXiv 2504.03846).
**Dubois et al. -- Length-controlled AlpacaEval** -- COLM 2024, arXiv 2404.04475. Regression-based debiasing of a known confound (length) raises Spearman with Chatbot Arena from 0.94 to 0.98. Applicability: analogous confounds for us -- sentence length, number of code spans, presence of a hyperlink -- can be regressed out of a judge's verdict rate.
**Verga et al. -- "Replacing judges with juries" (PoLL)** -- arXiv 2404.18796 (2024). A panel of smaller judges from disjoint model families beats a single large judge on 6 datasets, with less intra-model bias, at ~1/7 the cost. Applicability: the design for a multi-family LLM jury as one labelling source.

### 2.4 Weak supervision / data programming

**Ratner, Bach, Ehrenberg, Fries, Wu, Re -- Snorkel** -- *PVLDB* 11(3), 2017, DOI 10.14778/3157794.3157797, arXiv 1711.10160.
Labelling functions (patterns, distant supervision, heuristics, models) may abstain, conflict, and be correlated; a generative label model learns their accuracies and correlations from agreement structure *without* ground truth and emits probabilistic labels. Numbers: +45.5 % average predictive performance vs 7 h of hand labelling; SMEs 2.8x faster. Applicability: the exact machinery for combining hyperlink, symbol-index, co-change, alignment and LLM-vote signals; the learned per-source accuracies are themselves a reportable result ("hyperlink LF precision 0.93, co-change LF 0.41").
**WRENCH** -- Zhang, Yu, Li, Wang, Yang, Yang, Ratner, NeurIPS 2021 D&B, arXiv 2109.11377. 22 datasets; majority vote is often competitive with learned label models; small validation sets matter. Applicability: report majority vote as the baseline label model.
**Smith et al. -- "Language models in the loop: incorporating prompting into weak supervision"** -- arXiv 2205.02318 (2022), *ACM/IMS J. Data Science* 2024, DOI 10.1145/3617130. Prompted LLM answers to multiple sub-questions are mapped to votes/abstains and denoised by Snorkel; -19.5 % error vs zero-shot on WRENCH. Applicability: direct precedent for treating each LLM judge prompt as a labelling function rather than as truth.
**MSR with a collaborative heuristic repository** -- arXiv 2103.01722 (2021). Snorkel-style labelling functions over commit messages for MSR labelling. Applicability: SE precedent for data programming.

---

## 3. Co-change / version-history mining for doc-code consistency (drift as free supervision)

**Wen, Nagy, Bavota, Lanza -- "A large-scale empirical study on code-comment inconsistencies"** -- ICPC 2019, DOI 10.1109/ICPC.2019.00019.
1.3 B AST-level changes across 1,500 systems; taxonomy of comment-code co-evolution; inconsistent changes ~1.5x more likely to be in bug-introducing commits. Validation: manual analysis to build the taxonomy [sample size unverified]. Signal: whether a comment changed in the same commit as its code.

**Panthaplackel, Nie, Gligoric, Li, Mooney -- "Learning to update NL comments based on code changes"** -- ACL 2020, DOI 10.18653/v1/2020.acl-main.168, arXiv 2004.12169; and **"Deep just-in-time inconsistency detection between comments and source code"** -- AAAI 2021, arXiv 2010.01625.
Gold construction: from commit histories, method+comment changed together => positive "update" example; code changed with comment unchanged => "consistent" (heuristic negatives). Validation: manual inspection of samples, extrinsic evaluation by chaining detection with update. Applicability: the standard recipe for turning co-change into labels for *consistency*, including its known noise (unchanged comment may simply be neglected -- the same neglect Rath quantifies).

**Liu, Xia, Lo, Yan, Li -- CUP^2 (OCD + CUP)** -- *IEEE TSE* 49(1), 2023, DOI 10.1109/TSE.2021.3138909. Same commit-derived data; OCD +17.1 % P/R/F1 over baselines. **DocChecker** -- Dau, Guo, Bui, EACL 2024 demo, arXiv 2306.06347: UniXcoder-based, 72.3 % accuracy on JIT inconsistency detection. **Investigating the impact of code-comment inconsistency on bug introduction** -- arXiv 2409.10781 (2024).

**Ratol, Robillard -- "Detecting fragile comments" (Fraco)** -- ASE 2017, pp. 112-122. Comments fragile w.r.t. identifier renaming; Eclipse plugin; validated on rename refactorings with manual judgement [details unverified]. Applicability: identifier renames in `compiler/rustc_*` are a *mechanically detectable* event that must invalidate some guide sentences -- an objective test of whether our links point at the right crate.

**Zhong, Su -- "Detecting API documentation errors"** -- OOPSLA 2013, DOI 10.1145/2509136.2509523. NLP + program analysis finds >1,000 doc errors (outdated usage, wrong names) in real APIs; validation by developer confirmation. **Zhou et al. -- DRONE, "Analyzing APIs documentation and code to detect directive defects"** -- ASE 2017 [DOI unverified]. **Lee, Wu, Cheung, Kang -- FreshDoc** -- *IEEE TSE* 47(4):653-675, 2021, DOI 10.1109/TSE.2019.2901459: derives API renames from code revisions, detects outdated names in docs with 48 % higher accuracy than prior work; matched 82 % of developers' own doc updates; 75 % of 40 reported cases accepted. **Tan, Wagner, Treude -- DOCER, "Detecting outdated code element references in software repository documentation"** -- *EMSE* 2024 (ICSE 2024 journal-first), DOI 10.1007/s10664-023-10397-6, arXiv 2212.01479: regex-extracted code-element references in README/wiki checked for existence across snapshots; >3,000 GitHub projects; most had at least one outdated reference, surviving for years; filed fixes accepted. **"Wait, wasn't that code here before?"** -- arXiv 2307.04291 (2023, tool). **READU** -- arXiv 2607.15780 (2026): JIT README bug detection.
Applicability: these give an *objective, developer-confirmed* downstream signal: a link (sentence -> crate) is useful if it lets us flag the sentence when the crate's referenced items disappear or are renamed, and developers accept the report. Acceptance rate is a usefulness metric independent of link gold.

**Dagenais, Robillard -- AdDoc, "Using traceability links to recommend adaptive changes for documentation evolution"** -- *IEEE TSE* 40(11):1126-1146, 2014, DOI 10.1109/TSE.2014.2347969.
Mines "documentation patterns" (sets of code elements documented together) from RecoDoc links and reports violations as code evolves. Retrospective evaluation on four Java OSS: >= 50 % of documentation changes were related to existing patterns. Applicability: the closest published *downstream* evaluation of doc->code links: predict which documentation should change given a code change, scored against the project's own later doc edits.

**Hyperlink decay** (Hata 2019/2023, above) and **co-change reliability** ("Is code co-committal an indicator of evolutionary coupling?", *Software* 5(1), 2026, DOI 10.3390/software5010011 [journal unverified]; 14 K commits, five repos) -- co-commit is a noisy but persistent indicator.

---

## 4. Bitext / cross-document alignment as supervision

**Thompson, Koehn -- Vecalign** -- EMNLP 2019, DOI 10.18653/v1/D19-1136. Sentence alignment via multilingual embeddings + dynamic-programming approximation, linear time; +5 F1 over Bleualign on de-fr. **Artetxe, Schwenk -- margin-based bitext mining** -- ACL 2019 [DOI unverified]; **LaBSE** -- Feng et al., ACL 2022; **WikiMatrix** -- arXiv 1907.05791. Margin criterion (cosine relative to k-NN average in both directions) suppresses hubness and is the standard for mining *comparable* (not parallel) corpora. Applicability: rustc-dev-guide prose and each crate's `//!` crate-level rustdoc are two descriptions of the same artefact in the same language -- a comparable corpus; margin-based mining plus NLI can produce sentence -> crate alignments that are semantic by construction. Bitext mining is validated by downstream MT BLEU (a usefulness metric), rarely by direct human alignment judgements -- we would still need the human sample.

**Petrosyan, Robillard, De Mori -- "Discovering information explaining API types using text classification"** -- ICSE 2015. Supervised classification of tutorial sections as explaining a given API type; gold = manual annotation of five tutorials; precision 0.69-0.87 within-tutorial, 0.74-0.94 cross-tutorial. **Jiang et al. -- FRAPT, unsupervised tutorial-fragment-to-API relevance** -- ICSE 2017, arXiv 1703.01552 / 1703.01553. **Treude, Robillard -- SISE, "Augmenting API documentation with insights from Stack Overflow"** -- ICSE 2016, DOI 10.1145/2884781.2884800: links SO sentences to API types; human raters judge "insight"; developer task-completion study. Applicability: SE precedent for aligning free-text sections to code entities across documentation sources, all validated on small human-labelled sets (hundreds of sections).

**Fazelnia et al. -- "Lessons from the use of NLI in requirements engineering tasks"** -- RE 2024. NLI reformulation (entailment/contradiction/neutral) beats prompt/transfer baselines in few/zero-shot for classification, defect and conflict detection. Applicability: supports using an NLI verifier (premise = crate rustdoc, hypothesis = guide sentence) as one labelling function -- consistent with our own NLI probe (verifier AUC 0.84, poor as a matcher).

**Cross-source consistency checkers** (DOCER for README/wiki, comment checkers, API-doc checkers; see arXiv 2606.09090 for a 2026 survey framing). No SE paper I found aligns *multiple prose documentation sources* (e.g. guide vs crate docs) to each other as supervision -- this appears to be an open niche.

---

## 5. Downstream-task evaluation of trace links (usefulness without link gold)

**Mader, Egyed -- "Do developers benefit from requirements traceability when evolving and maintaining a software system?"** -- *EMSE* 20(2):413-441, 2015, DOI 10.1007/s10664-014-9314-z. Controlled experiment, 71 subjects, real maintenance tasks on two projects, half with / half without trace links; with traceability 24 % faster and more correct solutions (the paper reports a large correctness gain, ~50 % [unverified]). Metrics: task time, solution correctness. Applicability: the canonical "usefulness" study; a scaled-down version (n ~ 20 rustc contributors, tasks "which crate implements X?") is feasible.

**Dagenais, Robillard -- AdDoc** (Sec. 3): documentation-update recommendation, retrospective precision/recall against later real doc edits.

**Kagdi et al. 2007 / 2010 / 2013** (Sec. 1.2): change-impact prediction with temporal hold-out; metrics coverage, recall, precision of predicted co-changes.

**Feature location and bug localisation datasets** -- Dit, Revelle, Gethers, Poshyvanyk, "Feature location in source code: a taxonomy and survey", *JSEP* 25(1), 2013; Dit et al., "A dataset from change history to support evaluation of software maintenance tasks", MSR 2013 (ACM 2487085.2487114): gold = methods/files changed in issue-linked commits; Bench4BL (Sec. 1.4). Metrics: effectiveness (rank of first relevant), MAP, MRR, Top-k. Applicability: a link is useful if, given an issue that names a crate-level concept, the linked sentences/crates rank the eventually-changed crate high.

**Moran et al. -- Comet** (Sec. 1.5): practitioner survey with a Jenkins plugin at Cisco.

**Agrawal, Cleland-Huang -- "Leveraging traceability to integrate safety analysis artifacts into the software development process"** -- arXiv 2307.07437 (2023). Trace links connect artefacts to safety assurance cases and visualise change impact; illustrated on a UAV system; no quantitative downstream metric in the abstract. Dietrich/Cleland-Huang usefulness-for-safety work beyond this could not be pinned down [unverified].

**Borg, Runeson, Ardo -- "Recovering from a decade: a systematic mapping of IR approaches to software traceability"** -- *EMSE* 19(6):1809-1855, 2014, DOI 10.1007/s10664-013-9255-y; Borg, "Advancing trace recovery evaluation", arXiv 1602.07633. Finding: nearly all IR-TLR evaluation is in-vitro P/R on small datasets; calls for in-vivo evaluation. **"The impact of traceability on software maintenance and evolution: a mapping study"** -- *JSEP* 33(10), 2021, DOI 10.1002/smr.2374, arXiv 2108.02133: only ~30 % of traceability papers address *usage*; practitioners split on usefulness for change impact analysis. Applicability: justification that a downstream evaluation is itself a contribution.

**Hey et al. -- NoRBERT** (RE 2020) is requirements *classification*, not TLR; not applicable except as an example of transfer learning in RE.

---

## 6. Large-scale documentation-to-architecture datasets and ground-truth architectures

**No published sentence-level documentation-to-architecture dataset larger than the ARDoCo benchmark (198 sentences) was found.** Searches for Linux `Documentation/`, Kubernetes KEPs, Rust RFCs/rustc-dev-guide, Eclipse or Apache design docs as TLR datasets returned nothing; the only OSS doc->code LLM datasets are Alor et al. (112 + 32 segments) and the ARDoCo SAD-code links (up to 8,268 links but from <=198 sentences). Hata et al.'s 9.6 M comment links and 18 M commit-message links are the only large link corpora, and they are URL-level, not sentence-to-component.

**Ground-truth architectures (component decompositions usable as the "model")**
- Garcia, Krka, Mattmann, Medvidovic -- "Obtaining ground-truth software architectures", ICSE 2013, pp. 901-910 [DOI unverified]. Framework using domain/application/context information and limited engineer time; GT architectures for four OSS systems (Bash, OODT, Hadoop, ArchStudio) built *with* their engineers.
- Lutellier et al. -- "Comparing software architecture recovery techniques using accurate dependencies", ICSE-SEIP 2015; extended "Measuring the impact of code dependencies on software architecture recovery techniques", *IEEE TSE* 44(2):159-181, 2018. GT for Chromium (10 M SLOC; two years with developers), ITK, Bash, Hadoop, ArchStudio; submodule-based bootstrapping of GT.
- Schmitt Laser, Medvidovic, Le, Garcia -- ARCADE, ESEC/FSE 2020 tool demo; Garcia et al. -- SAIN, ICSA 2021: shared infrastructure and datasets for architecture recovery.
- Link et al. -- RELAX, ICSSP 2019, arXiv 1903.06895: concern-oriented recovery via text classification.
- LLM-based recovery: "Deductive software architecture recovery via chain-of-thought prompting", ICSE-NIER 2024, DOI 10.1145/3639476.3639776; "Software architecture meets LLMs" (SLR), arXiv 2505.16697; ArchAgent, arXiv 2601.13007 (claims a benchmark of eight production GitHub projects with architecture diagram + document [unverified]).
Applicability: the Garcia/Lutellier method ("recover a preliminary decomposition from build/module structure, then confirm with core developers in bounded time") is exactly how to promote the 79 crates (or a coarser grouping of them along the guide's own "Overview of the compiler" chapters) into a defensible architecture model; their reported cost (two years for Chromium) is the warning. ExArch (Sec. 1.5) is the in-lineage precedent for an LLM-derived SAM validated by manual matching.

---

## Synthesis for our case: ranked supervision designs

Ordering criterion: how much of the literature directly supports the design, how well it answers "semantic, not regex", and how cheaply it can be validated. Designs 1 and 2 are complementary (2 is the validation layer of 1); 3-5 are labelling functions that feed 1; 6-7 are alternatives to link gold.

**1. Programmatic weak supervision: a label model over five heterogeneous sources, validated on a stratified human sample.**
Sources (labelling functions, each may abstain): (a) markdown hyperlink to a crate's rustdoc page; (b) mention resolution through the compiler's symbol index (items, paths, aliases -> defining crate; RecoDoc/APIReal/ARCLIN style, Sec. 1.3); (c) co-change of the guide chapter file and `compiler/rustc_*` directories in rust-lang/rust PRs (Kagdi 2007, Trustrace); (d) margin-based embedding alignment + NLI entailment against crate-level rustdoc (Sec. 4); (e) votes from a jury of >= 3 LLMs from model families disjoint from the linker (PoLL). Combine with Snorkel's generative label model (report majority vote as baseline per WRENCH), emit probabilistic silver labels, and report the *learned accuracy of every source* -- that number is what makes the gold "semantic and auditable" rather than regex.
Backing: Ratner 2017; Zhang 2021; Smith 2022/24; Moran 2020 (SE precedent for probabilistic evidence fusion); Trustrace; Kagdi 2007.
Validation recipe: stratify (sentence, crate) candidates by evidence pattern (anchor-only, mention-only, alignment-only, LLM-only, multi-source, no-source) and by label-model confidence; draw 300-500 pairs (Pangakis: 250-1,250; alt-test: >= 30, 50-100 typical per test); three annotators (alt-test minimum); report Krippendorff alpha (Ahmed) and Cohen kappa against Alor's 0.94; run the alt-test with epsilon = 0.2 (experts); report precision/recall of the silver labels per stratum, not pooled.
Main threat: the sources are not conditionally independent -- (a), (b) and (d) all fire on sentences that name things, and (e) sees the same text. The label model will over-trust their agreement; the "no-source" stratum, which is where 95 % of our FNs live, is systematically under-labelled. Mitigate by modelling the (a)-(b) dependency explicitly and by over-sampling the no-source stratum for human labelling.

**2. Pre-registered LLM-annotation validity gates (model-model agreement, confidence, alt-test) -- the mandatory validation layer.**
Before any human labelling, compute Krippendorff alpha between >= 3 disjoint-family LLM annotators on 500 random candidate pairs; only proceed if alpha > 0.5 (Ahmed's gate, Spearman 0.65 with human-model agreement). Use output-probability or 3-run consistency (Pangakis: >= 3 runs, temperature > 0; consistency 1.0 items were markedly more accurate) to route low-confidence pairs to humans. Justify replacing humans on the rest with the alt-test winning rate >= 0.5. Control circularity: the judge families must differ from the linker's; report the delta when a same-family judge is swapped in (self-preference check, Panickssery).
Backing: Ahmed 2025; Pangakis 2023; Calderon 2025; Zheng 2023; Panickssery 2024; Verga 2024; TOSEM LLM-judge review (circularity).
Validation recipe: as above; additionally regress judge verdict rate on sentence length, number of code spans and hyperlink presence (Dubois-style length control) to show the jury is not just re-detecting anchors.
Main threat: SE tasks with human-human alpha around 0.4 (Ahmed's causality/code-accuracy rows) leave little room for any annotator; if our human alpha on unanchored sentences is that low, no gold is defensible and we must fall back to design 6.

**3. Semantic mention resolution as the successor of the rejected anchors (entity linking against the compiler's symbol index).**
Replace "verbatim crate name" with: detect code-like and natural-language mentions of compiler entities (types, functions, passes, queries, MIR/HIR concepts), resolve them to items via rustdoc's search index and the guide's own glossary/aliases (discovered from data, not hardcoded), and map items to defining crates. A sentence saying "the type context caches query results" links to `rustc_middle` without containing the string. This is the APIReal/ARCLIN/RecoDoc pipeline; ARCLIN shows it works without human labels.
Backing: Dagenais & Robillard 2012; Ye et al. 2018; Huo et al. 2022; Rigby & Robillard 2013 (salient vs incidental mentions); Botha et al. 2020 (anchor-based EL at scale).
Validation recipe: mention-level precision on a 200-mention sample (APIReal used 1,205); crate-level precision of the induced links on the design-1 human sample; report coverage (fraction of sentences with >= 1 resolvable mention -- expect well under 50 %).
Main threat: it is still mention-bound: sentences describing behaviour without naming an entity get nothing; and it *re-inherits* the supervisor's objection at the mention level if resolution is mostly exact string matching. It earns "semantic" only if a measured share of resolved mentions are non-verbatim (aliases, descriptions, partial paths). Measure and report that share.

**4. Comparable-corpus alignment between the guide and crate-level rustdoc (bitext mining + NLI).**
Each `rustc_*` crate carries `//!` documentation; treat the guide and the union of crate docs as a comparable corpus. Mine sentence -> crate alignments with margin-based scoring over sentence embeddings (Artetxe & Schwenk; LaBSE-style encoders), then filter with an NLI verifier (premise = crate doc, hypothesis = guide sentence; our probe shows NLI is a good verifier, AUC 0.84, and a poor matcher). Use as labelling function (d) and as a standalone "alignment gold" for the subset of crates with substantive docs.
Backing: Thompson & Koehn 2019; Artetxe & Schwenk 2019; Petrosyan 2015; Treude & Robillard 2016; Fazelnia 2024; Nishikawa 2015 (intermediate artefact).
Validation recipe: human judgement of 200 aligned pairs (Petrosyan-scale), plus the sanity check that alignment recovers >= 90 % of hyperlink anchors (it should, or the encoder is wrong).
Main threat: coverage bias -- crates with thin or absent `//!` docs (many leaf crates) cannot be aligned to, so the gold under-represents exactly the sibling crates the linker confuses; and text reuse (guide chapters that were copied into crate docs or vice versa) makes some alignments trivial. Report per-crate doc length and exclude near-duplicate pairs.

**5. Version-history evidence: doc-code co-change as a labelling function and as a temporal-hold-out test.**
Mine rust-lang/rust (which vendors the guide under `src/doc/rustc-dev-guide`) and the guide repo for PRs touching both a guide chapter and one or more `compiler/rustc_*` trees; compute chapter -> crate association strength (support/confidence, Kagdi's committer+day grouping; Trustrace-style reweighting). Use (i) as labelling function (c) at chapter granularity and (ii) as a *usefulness* test: links learned on history before date T should predict which chapters are edited when crate X changes after T (Kagdi 2007 protocol, AdDoc retrospective).
Backing: Kagdi 2007; Ali 2013; Kagdi 2010/13; Dagenais & Robillard 2014; Wen 2019; Panthaplackel 2020/21 (co-change labels and their noise).
Validation recipe: manual inspection of 100 co-change pairs for tangling; temporal hold-out MAP/Recall@k on "chapters edited after crate change"; report Bird-style linkage-bias checks (are co-changing chapters longer, newer, owned by fewer authors?).
Main threat: granularity (chapter, not sentence), tangled and bulk commits (submodule syncs, reformatting, link-fixing sweeps), and the well-documented lag of documentation behind code (DOCER: outdated references survive for years), so absence of co-change is weak negative evidence. Bench4BL's 44 % error rate is the realistic prior for this source.

**6. Downstream usefulness instead of link gold: documentation-drift detection and update recommendation.**
Define the task: given a merged PR touching crate X (or a rename/removal of item I in X), rank guide sentences that should be reviewed; gold = sentences actually edited in the guide within a window, or references that DOCER/FreshDoc-style checks prove stale. Compare ALinker's links against baselines (lexical anchors, co-change, embedding retrieval). Optional confirmatory step: file a bounded number of "this sentence is now stale" issues on the guide and report acceptance (FreshDoc: 75 % of 40; DOCER: fixes accepted).
Backing: Dagenais & Robillard 2014; Lee et al. 2021; Tan et al. 2024; Zhong & Su 2013; Ratol & Robillard 2017; Mader & Egyed 2015 (usefulness framing); Borg 2014 and the 2021 mapping study (why in-vivo evaluation is a contribution).
Validation recipe: retrospective MAP/MRR/Recall@k over >= 100 PR events with temporal split; developer acceptance rate on <= 30 filed reports; a small task-based study with contributors if feasible (Mader & Egyed protocol: time and correctness on "which crate implements this?" questions, with/without links).
Main threat: low base rate and long lag of doc edits make retrospective recall noisy; filed-issue acceptance measures the *report*, not the link; and this evaluates only the subset of links that become stale, biased toward frequently changing crates.

**7. Promote the crate decomposition to a validated architecture model, and evaluate at two granularities.**
Follow Garcia/Lutellier: derive a preliminary component view from the guide's own "Overview of the compiler" structure (parsing/expansion, name resolution, HIR/type checking, MIR/borrowck, codegen, driver, ...) grouping the 79 crates into ~10-15 components; confirm the grouping with two or three rustc maintainers in bounded time (or with the compiler team's published crate ownership/`triagebot` mappings as a proxy). Evaluate ALinker at both crate and component level; the component level neutralises sibling-crate confusion (our dominant FP class, Alor's "architecture pattern bias") and matches the granularity of the existing 6-14-component benchmarks.
Backing: Garcia 2013; Lutellier 2015/2018; Link 2019; Fuchss ICSA 2025 (ExArch: LLM-derived SAM validated by manual matching); ArTEMiS 2025.
Validation recipe: report inter-maintainer agreement on the grouping; report MoJoFM or a2a between the guide-derived grouping and a dependency-clustering baseline (ARCADE) to show the grouping is not arbitrary.
Main threat: the grouping is itself a judgement, and evaluating at a coarser level can hide the very errors the fine level reveals; report both and do not let the coarse score stand alone. Garcia's cost figures (months to years with developers) mean maintainer time must be tightly bounded.

**What not to do, per the literature.** Do not report recall against anchor-only gold (Rath: 40 % of true links are unanchored; Bird: the anchored ones are a biased subset). Do not use a single LLM of the same family as the linker as gold (Panickssery; TOSEM review). Do not pool precision over strata with very different evidence density (Huang 2025: realistic candidate distributions halve headline numbers). Do not present co-change or hyperlink evidence as truth; present the label model's learned accuracy for it.

