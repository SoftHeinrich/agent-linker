"""Authored findings for the paper snapshot identified in evidence/snapshot.json."""
ISSUES = []


def add(number, severity, criteria, source, line, page, anchor, title, problem,
        evidence, repair, recheck, status="Verified"):
    ISSUES.append(dict(id=f"FSE-{number:02d}", severity=severity, criteria=criteria,
        source=source, line=line, page=page, anchor=anchor, title=title,
        problem=problem, evidence=evidence, repair=repair, recheck=recheck,
        status=status))


add(1, "Blocker", "Soundness / open science", "paper/agent-linker.bib", 31, 19,
    "Replication Package: TODOTODO", "The replication package has no usable citation or access link",
    "The abstract, experiment design, and Data Availability statement promise accessible code, prompts, responses, and results, but reference [2] renders a placeholder title and note. A reader cannot obtain the promised evidence from the submission. This is an unresolved artifact reference, not an accusation of a fabricated literature paper.",
    "Rendered p.19 reference [2]; main.tex:160,193–194; eval.tex:75; current bibliography entry has neither URL nor DOI. FSE's open-science policy asks for access or an explanation of its absence.",
    "Supply a tested anonymous package URL and a precise inventory, release identifier, and availability statement. Distinguish replaying saved outputs from making new hosted-model calls.",
    "Open the package without author credentials; reproduce the tables from that release; inspect its anonymity and correspondence to the submitted PDF.")

add(2, "Blocker", "Presentation / submission format", "paper/main.tex", 110, 23,
    "Detailed Results", "The current full PDF exceeds the submission page limit",
    "The reviewed build has 31 pages: body and availability through p.18, references on pp.19–22, and appendices on pp.23–31. FSE 2027 permits 18 pages of text/figures plus four reference pages, with the availability statement exempted. The appended nine pages cannot simply be treated as outside that limit because they follow the references.",
    "Fresh Tectonic build; main.tex enables showappendixtrue. Official FSE 2027 How to Submit. The main body itself fits the nominal 18-page allowance.",
    "Prepare a submission build within the limit and place additional material in the accessible replication package. Replace appendix pointers with valid supplement references. Follow any separately published supplementary-material instructions if relying on an additional upload.",
    "Count pages in the actual submission PDF and check every reference after removing the appended material.")

add(3, "High", "Soundness / evaluation", "paper/sections/eval.tex", 124, 11,
    "accept every candidate retained", "RQ3 removes deterministic filters as well as LLM judges",
    "The stated ablations retain deterministic checks. In the evaluator, however, every rejected decision is restored, including name_ambiguous_discard and antecedent_form_rejected. On s126/terra, three runs on five projects, keeping those gates and disabling only the LLM judges gives macro doc-model F1=0.818798, versus the reported no-judge 0.786310. Against Full=0.936233, the observed gap is 11.74pp rather than 14.99pp. These are fixed-output reconstructions, not new model runs.",
    "audit.py / evidence/audit-results.json, judge_audit_means; rq34.py:394–445; s_linker126.py:1144–1229,1373–1399. Mean 27.67 additional false links per five-project run are restored only by removing gates.",
    "Either recompute RQ3 with gates retained, or rename the treatment as removing the combined filtering stage. Report gate and model-judge contributions separately before attributing the gain to LLM judging.",
    "Regenerate RQ3 counts, quality metrics, downstream composition, result boxes, discussion, abstract/conclusion claims where affected, and both backends.")

add(4, "High", "Evaluation / soundness", "paper/sections/results.tex", 42, 13,
    "Averaged cost per project", "The cost comparison mixes models and invocation sets",
    "Table 3 compares s126 terra usage with Artemis luna usage, while the body quality comparison uses terra for both. Neither the table nor its surrounding prose identifies the different Artemis backend. The text then prices both rows as GPT-5.6-terra and attributes the input-token ratio to the workflow. The arithmetic is reproducible, but this is not the cost of the matched-model quality experiment.",
    "INFERENCE_COST_PERRUN.csv explicitly records approach=gpt-5.6-terra and Artemis=gpt-5.6-luna. inference_cost.py:1–5,26–68 reads separate September 24 runs, including replacement MediaStore calls. The quality roster reads different run slots. Table totals reproduce.",
    "Measure usage from the exact quality invocations, or label this as a separate cross-model usage sample. Cite the price source and distinguish an estimated repricing from actual billed cost; state omitted embedding, retry, and cache costs.",
    "Join quality and usage by model, project, run, configuration, and response-log hash; update results.tex:42–48 and discussion.tex:15.")

add(5, "High", "Soundness / presentation", "paper/sections/motivation.tex", 6, 3,
    "rejected by the recorded judge", "The motivating rejection is not an outcome of the evaluated runs",
    "Figure 1 presents the database-access candidate as rejected. Its verification script uses an older s124 run. In all three evaluated s126 terra MediaStore runs, the name judge approves sentence 27→DB, and that false link survives. Section 3's worked rejection therefore cannot stand as evidence of the reported system's behavior without a variant/run qualification.",
    "paper/figures/verification/verify_mediastore_example.py:70 onward identifies s124. audit-results.json:mediastore_s27_decisions records s126 approval in run1, run2, and run3. The gold standard omits sentence 27→DB.",
    "Keep the example as a gold-standard challenge and explicitly show the current system's failure, or use a verified success from the evaluated invocation set. Label any design illustration as hypothetical or historical.",
    "Cross-check Figures 1, 4, and 5 and their prose against gold labels and the exact raw model response of the stated run.")

add(6, "High", "Presentation / soundness", "paper/sections/approach.tex", 84, 6,
    "Span=DB, Evidence: written=alias", "The worked figures contradict the evidence schema",
    "Figure 4 names DB literally but labels it alias; it labels database as exact although the catalog target is DB. The candidate target is drawn as database and one case quotes S1 instead of a matched expression. Its claimed quote is explanatory prose with spelling errors, not words from the source sentence; Figure 5 similarly uses a paraphrase in claim. Figure 4 also displays objection even though the name-judge schema has no such field. These are substantive mismatches with the method and appendix.",
    "Rendered pp.6–7, Figures 4–5; approach.tex:76–82; appendix/prompts.tex name-judge JSON and exact-quote instruction. The figure's accepted input also changes the benchmark's Database to DB.",
    "Regenerate the example records from an actual candidate/evidence/response triple. Preserve catalog names, source sentences, exact versus alias labels, and the emitted JSON schema. Clearly label shortened display text.",
    "Verify every displayed field against the implementation and raw response, then inspect the rendered figures.")

add(7, "High", "Soundness / method", "paper/sections/approach.tex", 59, 6,
    "An internal alias judge", "The alias judge cannot inspect the document it purportedly verifies",
    "Section 3.1 says the alias judge evaluates whether the document establishes each mapping. Its prompt contains component names and proposed mappings, with no document or supporting quotation. The extraction call sees the document; the subsequent judging call does not. A document-grounded validation cannot be inferred from this input, especially with an approve-when-uncertain default.",
    "Appendix B.1 alias-judge template; s_linker126.py:600–618. The prompt builder accepts only comp_names and proposals and supplies no extraction conversation.",
    "Describe the current stage as a plausibility filter over proposed mappings, or change it to receive document evidence and evaluate that changed method. Do not present adding evidence as copyediting.",
    "Inspect actual request payloads and repeat the affected alias and end-to-end evaluations if the inputs change.")

add(8, "High", "Soundness / method", "paper/sections/rw.tex", 60, 18,
    "structural earlier-sentence constraint", "The claimed earlier-sentence constraint is not enforced",
    "The resolver checks that target and antecedent sentence numbers exist, but does not require antecedent_sentence < target_sentence. The later gate checks written-name form only. The replay audit finds 296 metadata records with a same-or-later antecedent, including 56 records associated with accepted links across two backends, three runs each, and five projects. Records can repeat a link; these are not 56 distinct gold links.",
    "s_linker126.py:1327–1371; audit-results.json:non_earlier_antecedents. The appendix asks references to refer back, but prompting is not a structural ordering check.",
    "Narrow the description to the actual name-form gate, or enforce the claimed ordering and reevaluate. Explain whether same-sentence references are in scope.",
    "Check all accepted resolutions, route overlap, the structural-constraint novelty claim, and RQ3/RQ4 results after any semantic change.")

add(9, "High", "Evaluation / validity", "paper/sections/discussion.tex", 50, 17,
    "free of any benchmark-specific tuning", "Development on the evaluation benchmark is undisclosed",
    "No fine-tuning and no hard-coded benchmark vocabulary do not establish freedom from benchmark-driven design selection. The recorded s126 promotion compared predecessor variants on the same five projects and used those results in the decision. The paper gives no independent holdout or account of this adaptive development. This supports a selection-bias threat, not a claim that the observed scores are fabricated or that overfitting has been quantified.",
    "Tracked s126 design/provenance header and ARM_COMPARE_s126_vs_s123gctl.csv; git history aa45a1b4/c218f273; discussion.tex:50. The current benchmark is also the basis of the project's reference-form and failure analyses.",
    "Disclose which data informed rules, prompts, and arm selection. Limit the no-training statement to its actual meaning. Evaluate a frozen method on genuinely unused projects before claiming transfer, or frame this as an exploratory benchmark study.",
    "Audit the development/evaluation split, selection history, and every claim of tuning independence.")

add(10, "High", "Evaluation / internal validity", "paper/sections/discussion.tex", 41, 16,
    "The ablations", "Ablations do not rule out additional computation as the explanation",
    "Removing a module removes both its mechanism and its computation. The current ablations therefore cannot rule out the benefit of additional LLM calls or deliberation. There is no equal-budget repeated-baseline or alternative-workflow control. The existing experiment measures the recorded module removal, subject also to FSE-03.",
    "discussion.tex:41–42 explicitly says the ablations rule this out; eval.tex:117–158 describes removal and overlap experiments, with no matched compute control.",
    "State extra inference effort as an unresolved alternative explanation. If retaining the stronger claim, add a prespecified equal-model, comparable-budget control such as repeated baseline inference or a simpler refinement workflow.",
    "Compare quality and actual usage on the same invocation sets; revise causal language throughout the discussion and conclusion.")

add(11, "High", "Soundness / internal validity", "paper/sections/discussion.tex", 38, 16,
    "very likely encountered", "Using one backend does not control away memorization",
    "Public availability does not establish that these precise artifacts were in pretraining. Sharing a backend controls model identity, but different workflows may exploit any memorized vocabulary or links differently. Therefore the inference that only workflow separates the systems does not remove contamination as a threat to transfer or causal interpretation.",
    "discussion.tex:37–44. No training-corpus membership evidence or unseen-project experiment is supplied; the paper itself notes possible vocabulary memorization.",
    "Describe exposure as possible and acknowledge differential use of memorized information. Retain matched-backend comparisons as a useful control with a limited scope.",
    "Ensure abstract, introduction, and validity discussion do not treat shared model identity as proof of equivalent exposure effects.")

add(12, "High", "Appropriate comparison to related work", "paper/sections/intro.tex", 84, 1,
    "word overlap as the signal", "The baseline characterizations are materially inaccurate",
    "The introduction collapses the baselines into word-overlap recovery and says implicit references are seldom considered, without a scoped measurement. The cited Artemis method already extracts contextual occurrences, aliases, and coreferences with two prompts before matching. LiSSA uses retrieval and zero-shot pair classification; calling it a simple few-shot baseline is incorrect. A missing separate judge does not establish that a method lacks semantic checks or contextual reasoning.",
    "Artemis primary paper §4, pp.9–10; LiSSA primary paper §III.C, p.5. Also eval.tex:22,95 and rw.tex:18–23,63. See sources.md for direct primary-source links.",
    "Compare the actual decision mechanisms: extraction plus matching, retrieval plus classification, and the proposed evidence-bearing candidate judgment. Scope empirical weaknesses to measured cases.",
    "Reconcile every baseline description across abstract, introduction, evaluation, and related work with the cited version and evaluated implementation.")

add(13, "High", "Originality / appropriate comparison", "paper/sections/rw.tex", 58, 18,
    "the first architecture-to-code", "The novelty claim is broader than the established distinction",
    "The related-work claim combines a separate knowledge layer, implicit references, and varying judge strictness; the conclusion separately claims the first training-free multistage workflow. The first two capabilities overlap with the cited baseline, while the method says the written-name forms use one rubric. The metric section also claims that no existing TLR benchmark addresses the evaluation gap without a supporting comparison. The paper has not established these priority claims.",
    "rw.tex:58–63; conclusion.tex:4; intro.tex:118; metric.tex:31–33; Artemis §4. A verified chronology and a feature-by-feature comparison are absent.",
    "State the narrow contribution as the particular candidate/evidence/judgment design and its measured behavior. Identify which pieces are established adaptations. Remove first/none claims unless a systematic, task-scoped comparison supports them.",
    "Use one consistent novelty statement across introduction, method, related work, and conclusion; check it against the evaluated mechanism.")

add(14, "High", "Soundness", "paper/sections/metric.tex", 17, 8,
    "richest component", "The Gini interpretation is mathematically invalid",
    "Gini=0.59 does not determine the ratio between the largest component and the bottom 80%. The review script constructs two five-component distributions with exactly that Gini but largest-to-bottom-80% ratios of 3.762 and 2.028. Neither supports the proposed approximately-equal-income explanation. A Gini value summarizes pairwise dispersion, not a unique Lorenz-curve point.",
    "metric.tex:17; audit-results.json:gini_counterexamples, with explicit distributions and formula. This is a logical counterexample independent of the benchmark.",
    "Remove the income-ratio inference. Illustrate concentration using the actual counts or an explicitly computed Lorenz-curve share for a stated project.",
    "Recalculate the replacement illustration and verify that its denominator and project are stated.")

add(15, "High", "Soundness / metric interpretation", "paper/sections/intro.tex", 74, 1,
    "documented components entirely", "CMR is misreported as an ordinary component percentage",
    "The 7.1% headline is a macro average of sentence-assignment-weighted abandonment, not the percentage of components missed. Likewise CMR=0 means at least one correct link per gold-documented component, not that its documentation is fully recovered or that the trace matrix has no missing links. Ten components with 100 gold links each and one correct link each have CMR=0 and recall=1%.",
    "Equation 4 / metric.tex:81–91; metrics.component_miss_rate; intro.tex:74; results.tex:69–72. audit-results.json:cmr_counterexample. The implementation follows the weighted definition.",
    "Call this weighted component abandonment consistently; report unweighted counts separately if discussing a fraction of components. Replace the no-documentation-lost inference with the exact at-least-one-correct-link statement.",
    "Audit every CMR interpretation, including the intro, RQ2 answer, table headers, and practitioner advice.")

add(16, "High", "Soundness / metric specification", "paper/sections/metric.tex", 48, 9,
    "Let", "The component metrics change the evaluated domain as well as the weights",
    "The component evaluator uses gold SAM→code ownership, drops Interface elements, can count a shared file in several components, and excludes predictions outside the gold-reachable component universe. Section 4.2 does not fully specify these choices. Thus differences from link-level F1 cannot be attributed only to giving components different weights. In s126 terra BigBlueButton, 192/199/192 false file links in runs 1/2/3 fall outside the evaluated components; four do on JabRef in every run.",
    "metrics.py:299–315,362–461; audit-results.json:component_metric_exclusions. Table 1 mentions shared/unmapped files for concentration statistics but not the full scoring rule.",
    "Define the ownership source, gold-only universe, zero cases, interface treatment, and unmapped-prediction policy. Report excluded links and explain what errors the suite does not penalize.",
    "Reconcile the equations, implementation, Table 1, and claims that only weighting changes. Include a small worked example with a shared and an unmapped target.")

add(17, "High", "Soundness / construct validity", "paper/sections/results.tex", 40, 12,
    "recall is focused on small components", "The downstream-loss explanation omits incompatible gold standards",
    "MediaStore reaches perfect doc-model recall, yet composing the gold doc-model links through the evaluated mapping reaches only 52 of 59 doc-code gold links. Teammates reaches 6380 of 8097, leaving 1717 gold file links outside that composition. BigBlueButton's composition additionally creates 280 pairs absent from its file gold. These task differences can reward doc-model false positives downstream. A small-component explanation alone does not establish the cause of the observed losses.",
    "audit-results.json:composition_limits, freshly recomputed with rq34_rq2.compose_doc_code. The recorded Teammates transfer study also describes this issue. See discussion.tex:28–29 and RQ1 task definitions.",
    "Disclose the relationship between both gold standards and the oracle composition result. Explain downstream errors by reachable versus non-reachable gold and mapping errors before attributing them to component size.",
    "Recheck the MediaStore and Teammates explanations and all statements that doc-model improvements carry through to code.")

add(18, "Medium", "Soundness / presentation", "paper/sections/motivation.tex", 79, 4,
    "a sixth of the code depends on", "The motivation changes the denominator of its importance measures",
    "The 17.6% value is Ca(component)/sum(Ca), where Ca counts distinct external files depending on a component. It is not a fraction of all code or a count of dependency edges. For preferences the stored Ca_share is 18.8% of external files. The 20% sentence value is 2/10 gold-linked sentences, not 20% of the 13 document sentences. These distinctions matter to the 40×/46× comparisons and the claim of architectural importance.",
    "jabref_depshare.csv; jabref_motivation_data.csv; studies/mini-depimport/depimport.py and reports/DEPIMPORT.md; Table 1's 13 JabRef sentences.",
    "Name each denominator and dependency unit in the text or caption, cite the dependency-extraction method and source revision, and describe the measures as proxies for importance.",
    "Recompute the percentage ratios from unrounded values and ensure labels do not equate normalized coupling share with a fraction of code.")

add(19, "Medium", "Evaluation / reproducibility", "paper/sections/eval.tex", 69, 10,
    "released configurations", "The experiment configuration is insufficiently identified",
    "The manuscript gives backend labels and three runs but omits the exact dataset revision, baseline commits/configurations, provider/model snapshot mapping, decoding/reasoning settings, run dates, cache isolation, and failure/retry policy. Runtime aliases terra/luna alone do not let a reviewer reproduce the requests. The cost run substitution in FSE-04 makes invocation identifiers especially necessary.",
    "eval.tex:61–75; Appendix A opening; local run/provenance records contain settings not exposed in the manuscript. No accessible package presently resolves the omissions (FSE-01).",
    "Add a compact configuration/provenance table, with detailed machine-readable manifests in the package. Identify the revised benchmark and all model-specific defaults actually used.",
    "Reconstruct each reported arm from a clean checkout using the published configuration, and confirm independent runs do not reuse response caches.")

add(20, "High", "Soundness / construct validity", "paper/sections/discussion.tex", 53, 17,
    "Construct validity", "Gold-label uncertainty is missing from the validity discussion",
    "The new operational link definition and judge labels are treated as architectural truth, while evaluation labels come from an existing curated benchmark. The project has recorded suspected gold omissions and cross-task disagreements. A model output absent from this gold is a measured false positive, but is not thereby established as a hallucination or a semantically unsupported statement. The current construct-validity paragraph discusses only metric preferences.",
    "Section 3's link definition; discussion.tex:53–57; studies/causal-claims/CH2-mode1-analysis.md flags suspected gold gaps, without independent adjudication here; freshly verified composition differences in FSE-17.",
    "Describe annotation provenance, revision, scope, and known disagreements. Separate gold-relative error counts from semantic error classifications. Independently adjudicate a documented sample if claiming that the judges remove hallucinations.",
    "Track adjudicators, sampling, rubric, disagreement resolution, and sensitivity to disputed labels; keep original benchmark scores visible.")

add(21, "Medium", "Evaluation", "paper/sections/results.tex", 11, 12,
    "means of three runs", "Uncertainty is thin for the metrics carrying the main claims",
    "The body shows SD for precision and recall, but not F1/F2 or the worst/harmonic component scores that carry the headline gains. Three calls per project estimate limited run variability; the projects are five separate systems, not fifteen independent samples from a broad population. The appendix helps by showing runs, but the conclusion does not distinguish exploratory observed differences from general performance expectations.",
    "Table 2 footnote; Tables 4–6; Appendix A; RQ12 per-run reports. This review does not require a significance test merely to report descriptive differences.",
    "Report the existing per-run spread for the main metrics and project-level effects, with a clear unit of replication. Use uncertainty intervals only with assumptions justified for this small study; keep cross-project claims scoped.",
    "Verify aggregation order, SD definitions, and that separately executed variants are not analyzed as paired random realizations.")

add(22, "Medium", "Evaluation / completeness", "paper/sections/eval.tex", 74, 10,
    "two language-model backends", "The second backend changes a central comparison but is not discussed",
    "The appendix contains luna results, so they are not missing. However, the body uses two backends as a safeguard without explaining the observed sensitivity. The matched-backend worst/harmonic F1 gaps are 11.61/15.36pp on luna versus 29.60/27.58pp on terra. Artemis's component metrics exceed TransArc on luna, reversing the terra comparison used in the motivation.",
    "RQ12_BIGTABLE.csv and audit-results.json:headline_deltas_pp. Luna Artemis worst/harmonic=.5650/.7288; TransArc=.5149/.6756. Both stochastic systems have three runs across the same five projects.",
    "Add a brief main-text sensitivity result. Scope the Artemis-versus-TransArc reversal to terra and avoid implying that the component-level finding has the same magnitude or direction across backends.",
    "Check matched-backend comparisons, all ranking statements, and any generalized claim based only on the body tables.")

add(23, "Medium", "Reproducibility / method", "paper/appendix/prompts.tex", 4, 29,
    "uses five LLM prompts", "The appendix templates are edited versions of the executed prompts",
    "The appendix is presented as prompt templates, but it shortens decision-bearing text. For example, the executed coreference judge asks for a decisive rejection ground and an approve-unless rule after the strict rubric; the appendix omits that qualification. The resolver also omits the implemented allowance for any document-used form. These are potentially semantic changes, not merely placeholder substitution.",
    "Compare appendix/prompts.tex with s_linker126.py:127,623–653,666–705. The root working agreement treats reworded decision criteria as method changes absent a fixed-input equivalence audit.",
    "Export the actual evaluated templates and explain dynamic fields. If the appendix is only a summary, label it explicitly and provide exact templates plus representative complete requests in the package.",
    "Diff all five prompt builders and saved request payloads against the supplied templates; rerun if substantive instructions are changed.")

add(24, "High", "Soundness / method completeness", "paper/sections/approach.tex", 73, 6,
    "The two scans produce one candidate stream", "The named-route description omits result-changing deterministic decisions",
    "The displayed method moves from two scans directly to evidence and judgment. It omits whole-name span ownership and discarding a residual span that maps to multiple components. Those decisions determine which candidates ever reach a model judge and are central to the evaluated s126 variant. They also complicate RQ3's treatment definition. Source comments do not communicate them to readers.",
    "s_linker126.py:_only_inside_another_name, _name_candidates, _judge_union; approach.tex:68–82; FSE-03 quantifies the combined gate issue.",
    "Include the deterministic ownership, disambiguation/abstention, and deduplication steps in prose or pseudocode. State candidate order, batch/context construction, and failure defaults in the artifact configuration.",
    "Trace examples through the full candidate pipeline and ensure prose, algorithm, diagram, prompts, and ablation boundaries agree.")

add(25, "High", "Originality / evaluation", "paper/sections/intro.tex", 116, 2,
    "a plain judge scores an output alone", "The distinctive evidence-bundle contribution is not isolated",
    "The proposed distinction is that judges use structured evidence. RQ3 compares filtering with no filtering; it does not compare the same judge with and without the supplied span/name-form/antecedent fields, or against a simpler contextual judgment. The definition of a plain judge as output-only is unsupported. The present evidence establishes a combined filtering benefit, not the incremental value of this advertised distinction.",
    "intro.tex:116–118; eval.tex:117–144; Appendix B. No corresponding fixed-candidate evidence-input comparison appears in the presented RQs.",
    "Narrow the claim to the evaluated workflow, or add a fixed-candidate comparison that varies evidence inputs while preserving model, candidates, and judgment criterion. Do not conflate this with the equal-compute issue in FSE-10.",
    "Measure both retained/rejected gold-relative links and output quality under identical candidate sets; distinguish the evidence effect from generic additional judgment.")

add(26, "High", "Soundness / authored wording gate", "paper/appendix/prompts.tex", 64, 30,
    "An expression occurring only as part", "Some prompt rules turn conventions into unconditional semantic exclusions",
    "The identifier rule says that a name inside a qualified identifier cannot denote an architectural participant; the rejection rule also treats a negated claim as a ground against linking. Neither implication holds for arbitrary architecture text: qualified names can identify components, and statements that a component does not perform a responsibility still concern that component. These may be chosen operational restrictions, but the paper presents them as general semantic facts.",
    "Appendix B.2; s_linker126.py:186 and LAYERED_ENTITY_RULES; Section 3's architectural-claim definition. The local wording gate requires a general logical basis, software-engineering practice, or scoped measured/literature evidence.",
    "State and justify the restricted link semantics, or replace the unconditional exclusion with a supported criterion. Keep benchmark observations explicitly scoped. Treat a rule change as a method change.",
    "Use fixed inputs containing both valid and invalid qualified/negative references, then reevaluate the changed workflow without silently reusing old scores.")

add(27, "Medium", "Soundness / result interpretation", "paper/sections/results.tex", 108, 14,
    "144.3 distinct false positives", "The rejected-link count is not the number removed from final output",
    "The union of route rejections contains links that the other route retains. Full rejects 144.33 distinct false links at some route, but no-judge FP=166 and Full FP=25 imply only 141 net false links removed. The 3.33 difference survives through another route. Saying the counts do not sum addresses overlap between rejections, but does not explain this rejection/acceptance overlap. It also includes deterministic decisions (FSE-03).",
    "rq34.py:489–501 uses union(rejected) without subtracting final for rejected_fp. audit-results.json:judge_audit_means verifies the 3.33 surviving-FP overlap for terra over three five-project runs.",
    "Distinguish route-local rejections, distinct rejected candidates, and links actually removed from the final union. Use the net count when describing the final precision benefit.",
    "Reconcile both TP and FP set algebra with no-judge/full predictions for each run before averaging.")

add(28, "Medium", "Presentation / evaluation", "paper/table/rq3-confusion.tex", 13, 14,
    "Judges", "Table 5 combines incompatible row meanings and omits aggregation units",
    "A row labeled name judge lists that judge's rejected/kept counts, but its quality metrics describe the configuration with that judge disabled. A reader naturally interprets the row as enabling the named judge. Full and No judge instead name configurations. Decimal counts also need an explanation: they are five-project totals averaged over runs, whereas the quality columns are macro project means. The short caption does not disclose either distinction.",
    "Table 5; evaluation/reports/tex_src/rq3.csv; rq34.py and rq_tables.py. E.g., name-judge row F1=.84 is NoNameValid, not name-judge-only recovery.",
    "Split the local audit from the ablation scores, or use separate explicit columns for audited judge and disabled judge. State task, backend, run count, aggregation, and gate policy.",
    "Check the same labels in the appendix and verify that count denominators and metric denominators are unambiguous.")

add(29, "Medium", "Importance / metric evaluation", "paper/sections/metric.tex", 63, 9,
    "potentially noisy", "The metric suite's usefulness is motivated, but not validated by larger gaps",
    "Worst and harmonic scores deliberately emphasize low components; a larger method gap under them does not establish that they better represent developer utility. Both collapse to zero when any component scores zero, so adding the harmonic mean does not resolve that minimum's zero sensitivity. The paper acknowledges proxies, but does not compare these choices with ordinary component macro-F1, report sensitivity to component partitions, or establish the claimed architectural significance through users.",
    "Equations 2–4; metric.tex:63–75; RQ2; discussion.tex:53–57. No user study or metric-choice/partition sensitivity study is reported.",
    "Present the suite as complementary diagnostics with explicit trade-offs. Add simple arithmetic component means and per-component results if claiming the selected aggregation adds useful information; reserve utility claims for suitable evidence.",
    "Check zero cases, split/merged components, and whether conclusions depend on choosing harmonic/minimum rather than another defensible summary.")

add(30, "Medium", "Importance / practical implications", "paper/sections/discussion.tex", 17, 16,
    "directing human review", "The proposed deployment diagnostic requires unavailable ground truth",
    "The suite needs known gold links and ownership. In a deployment where the purpose is to recover missing trace links, practitioners generally cannot compute recall, F1, or gold-relative component abandonment without first establishing that truth. The three scalar summaries also do not identify which components need inspection. The current advice moves from benchmark diagnostics to practical targeting without stating this prerequisite.",
    "Equations 1–4; discussion.tex:17–18. The artifact computes per-component values but the manuscript mostly presents aggregates.",
    "Scope the advice to benchmark evaluation or audited subsets with gold labels, and say that per-component diagnostic outputs are needed to target review. Label deployment without gold as future work.",
    "Ensure practical claims name the required inputs and do not promise automatic detection of unknown missing links.")

add(31, "Medium", "Soundness / interpretation", "paper/sections/discussion.tex", 24, 16,
    "make uniform coverage difficult", "Residual-error explanations exceed the recorded analysis",
    "BigBlueButton's Gini, component count, and document length are used to explain difficulty, but no controlled or per-error analysis establishes that relationship. The following text calls 0.72 versus 0.77 well below and treats a 0.77 link score as near-saturation without a stated criterion. These are interpretations presented with more confidence than the evidence supports.",
    "discussion.tex:23–29; Table 4. FSE-17 shows a directly measured alternative explanation for some downstream discrepancies.",
    "Report the observed weak component scores plainly. Label proposed causes as hypotheses, and replace near-saturation or well-below rhetoric with the actual gap and its task/project scope.",
    "Check each because/therefore statement in the discussion against a recorded analysis that distinguishes competing explanations.")

add(32, "Medium", "Presentation / claim scope", "paper/main.tex", 145, 1,
    "the hardest to recover", "The abstract overstates the problem and underspecifies its largest gains",
    "No comparative study establishes architectural links as the hardest class to recover. The per-sentence-LLM explanation is not a description of all relevant baselines. The 30/28pp statement does not name worst-component and harmonic F1 or identify doc-code, and the abstract omits the particular backend and repetitions supporting its quantitative claims. These omissions invite a broader interpretation than the evaluation supports.",
    "Abstract, main.tex:145–159; the backend-specific deltas in audit-results.json. The headline arithmetic itself agrees with the recorded terra averages.",
    "Describe the abstraction/vocabulary difficulty without a superlative; identify the primary doc-model task, transitive doc-code task, comparison backend, three runs, and the two metrics associated with 30/28pp.",
    "Trace each abstract claim to the corrected method, same invocation set, and explicit table columns.")

add(33, "Medium", "Appropriate comparison / evidence attribution", "paper/sections/intro.tex", 55, 1,
    "the strongest reaching", "The opening 84% result is not identified in its cited study",
    "The introduction attributes 84% F1 to the cited Artemis study without task, model, or aggregation. That paper's best reported average doc-model result is 0.81 for GPT-5 across five runs. The local single GPT-5.4 row is 0.8355, which rounds to 84%, but it is a different invocation set from the cited paper and the matched terra comparison. The present attribution is untraceable as written.",
    "Artemis §5.4.2/Table 8; RQ12_BIGTABLE.csv: Artemis (GPT-5.4), single. The review does not equate published weighted/doc-code results with the local doc-model macro score.",
    "Either cite and accurately identify a published result, or call 0.8355 a local rerun and give its configuration/provenance. Prefer the matched comparison when motivating this experiment.",
    "Verify task, dataset revision, model, repetitions, and averaging for every imported prior-result number.")

add(34, "Low", "Presentation", "paper/sections/rw.tex", 11, 17,
    "manage traceability information models", "Visible fragments, terminology drift, and repeated setup weaken readability",
    "Related work contains a subjectless sentence beginning manage and two detached citations at its end. Other visible defects include lexical and LLM technique, pools link, because ... therefore, few shot basline, because its scope, and a missing period before This motivates us. The introduction introduces two written forms after naming two routes, leaving their referents unclear. Challenges are restated in the introduction, motivation, and method, while terms such as pre-hook and arm s126 appear without useful reader context.",
    "intro.tex:69,118,131; motivation.tex:106–117; eval.tex:72,95; results.tex:11,152; rw.tex:11,64–65; approach.tex:112. These are rendered text, not ignored source comments.",
    "Repair the fragments and grammar; use named-route versus coreference-route consistently, explain the two lexical scans locally, define or replace implementation terms, and consolidate repeated challenge summaries.",
    "Read the rebuilt PDF continuously, including figure text and captions; ensure copyediting has not strengthened scientific claims.")

add(35, "Low", "Presentation", "paper/appendix/rq4-run1.tex", 7, 27,
    "RQ4 module ablation, run 1", "Several tables exceed the text width and are difficult to scan",
    "The fresh build reports 33.18pt overfull boxes for the four RQ4 run tables; the rendered tables extend beyond the normal text block. Table 4 packs two nine-metric panels into a single table and reuses project abbreviations without expanding them locally. Tables 2, 5, and 6 pack four metrics into slash/semicolon cells, making individual comparisons and variant meanings difficult to follow.",
    "evidence/build.txt; visual inspection of pp.13–14,26–28. This is a readability issue; no unsupported minimum-font-size rule is being asserted.",
    "Keep tables within the intended text width, use self-contained abbreviations and captions, and separate the comparisons that readers actually need. Put detailed tables in the package when preparing the submission build.",
    "Inspect rendered tables at ordinary reading scale and check all overfull warnings after layout changes.")

add(36, "Medium", "Evaluation / reproducibility", "paper/sections/eval.tex", 69, 10,
    "deterministic baselines", "The deterministic baseline provenance needs a precise release explanation",
    "The paper says released configurations are used. The artifact documentation records that the canonical BigBlueButton baseline differs substantially from a self-contained ICSE24 rerun: SWATTR/TransArc F1=.79/.83 versus .29/.35, with the other four projects agreeing. Choosing the stronger canonical output is not evidence of unfairness, but without an exact source revision and explanation a reader cannot tell which published benchmark/configuration is reproduced.",
    "sota-links/README.md, caveat 4 and regeneration section; current Table 2 uses the canonical values. The current review reproduces saved-link scoring, not these Java baseline executions.",
    "Identify the exact baseline release, revised benchmark, dependencies, and command that produced the canonical links. Explain the known discrepancy and clearly label replay-only outputs if the generating environment cannot be reconstructed.",
    "Run the baseline from the documented release or preserve the failure evidence and configuration limitation; confirm all systems are scored on the same normalized gold inputs.")
