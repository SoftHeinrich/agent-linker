# Measurements stripped from sections/approach.tex

Removed 2026-09-03 by the RW-PASS refactor. The approach section is a design section:
it justifies conceptually and carries no measured numbers. Everything below is preserved
verbatim so `results.tex` can absorb what it wants. The `%DONE` audit notes still in
approach.tex are the provenance for these numbers.


## nutshell

- *forward-refs CMR before sec:metric defines it; states contribution 2's finding as the approach's aim*
  > By recovering the references other tools silently drop, \approach{} aims to leave no documented component abandoned---the abandonment that the doc-model \cmrname{} (\cmr) of \autoref{sec:metric} measures.


## sec:knowledge

- *retired module, post-mortem lead*
  > An earlier design paired it with a \emph{model-understanding module} that asked an \ac{LLM} which component names double as ordinary English---in JabRef (\autoref{fig:example}) the component \texttt{preferences} shares its name with the everyday word---and passed the answer to the judges as an advisory flag.

- *retired module, measured*
  > The flag reached $78$ of $188$ candidates, so it was not idle, but removing the module, its call and its flag changes neither score ($\mathrm{TP}$ $-0.2$, $p=1.00$; $\mathrm{FP}$ $+0.8$, $p=0.40$ over five runs on all five projects).

- *retired alias scope grade, measured*
  > An earlier design instead graded each alias by how distinctive it was and withheld the ordinary-English ones from \linkerB{}; grading them cost $3.0$ true positives ($p=0.01$) for no measurable gain in precision ($+1.0$ false positives, $p=0.59$), so the grade was dropped and the judges are left to decide in context.


## sec:entity-linker

- *retired LLM extractor, post-mortem lead*
  > An earlier design asked an \ac{LLM} to extract the named mentions and held its output to that same relation afterwards.

- *extractor replay, measured*
  > Replaying the extractor's own recorded runs showed the proposer, not the judge behind it, to be where the headroom was: the scan reaches $7.8$ more gold pairs per run than the extractor ever emitted, and a mention the extractor never proposes is one no judge can recover.

- *scan-vs-extractor, measured*
  > Substituting the scan raises macro F2 on both backends ($+2.0$, $p=0.10$; $+1.8$, $p=0.20$) at unchanged macro F1 ($+0.4$ and $-0.9$, neither significant), and retires one \ac{LLM} call per $50$ sentences---nine of them in a five-project run.


## sec:partial-linker

- *nesting refusal, measured*
  > Replayed pair by pair over twelve recorded runs, the refusal fires on exactly $12$ candidates in every run of both backends and removes $5.2$ false positives a run on GPT-5.6-terra and $10.8$ on GPT-5.6-luna, at no cost in true links.


## sec:coref-linker

- *shortlist density, measured*
  > The list is a shortlist in fact and not only in intent: over the resolver's own windows it names $1.8$ to $4.5$ of a catalog's $6$ to $14$ components per case, and a refer-back's antecedent sits a median of two sentences behind it.

- *shortlist effect, measured*
  > Handing it over costs no extra call and cuts spurious resolutions from $16.9$ to $12.3$ a project-run on GPT-5.6-terra and from $38.4$ to $23.1$ on GPT-5.6-luna, at a gold cost of $0.2$ and $0.5$.


## sec:validators

- *reference-form field, measured*
  > Withholding it costs $6.6$ true positives ($p=0.01$), so the judge is using it.

- *reference-form values, measured*
  > Its distinctions are finer than they look necessary: the judge approves proper-case standalone mentions and lowercase ones at $96.9\%$ and $100.0\%$, so the grade appears to carry nothing.

- *reference-form merge, measured*
  > Merging those two values anyway costs $0.9$ macro F1 ($p=0.05$ over six paired runs), concentrated on the shortest document, where a shift of one and a half false positives moves F1 by four points.

- *measurement-derived claim; belongs in results*
  > Equal approval rates per value are therefore not evidence that a distinction is inert.

- *announces measurements*
  > Both halves of that claim are measured.

- *quote demand, measured*
  > Dropping the quote from those two prompts costs $35.2$ true positives ($p=0.01$), so the demand is not decoration; adding the mechanical check to them voids nothing at all ($0$ verdicts in $25$ project-runs), so the check would be.

- *measured lead of the retired grounded review*
  > The \partValidator{} rules in one target-blind step, worth $12$ false positives, and the measurement is why it stops there.

- *retired second pass, post-mortem lead*
  > An earlier design followed it with a grounded review that showed the model the target together with the sentences naming it and asked whether the two denote the same participant.

- *grounded review, measured*
  > That review trades recall for precision at a bad rate: it rejects $8.0$ candidates per run of which $5.5$ are gold.

- *grounded review removal, measured*
  > Dropping it leaves F1 unchanged ($+0.2$, $p=0.53$) and improves F2 by $1.3$ ($p=0.01$) over six paired runs, and it helps only one of the five documents, the one that proposes the most partial names, while costing recall on the one where partial names carry the most links.

- *coref second pass, measured*
  > Giving \linkerC{} a second pass as well confirms this rather than improving on it: it moves neither score ($\mathrm{TP}$ $-0.6$, $p=0.40$; $\mathrm{FP}$ $-0.8$, $p=0.17$).

- *batch-size ablation, measured*
  > Resolution and judging read the same batch size, so the workflow states two batch sizes rather than three; giving resolution a smaller window of its own costs a quarter of the pipeline's calls and buys nothing ($\mathrm{F1}$ $-0.2$, $p=0.52$; $\mathrm{F2}$ $-0.0$, $p=0.91$ over six paired runs).

- *retired two-pass conjunction, measured*
  > An earlier design sent the prompt twice, once per bar, and approved only on agreement; replayed over its own recorded runs the two passes disagree on $4.0$ of roughly $196$ candidates per five-project run and neither direction of disagreement is stable, so the conjunction was buying noise at twice the calls.


## Measurement clauses excised from surviving sentences

- was:
  > Scanning at \emph{any case} rather than at \emph{as spelled} is the same trade one level down: \autoref{tab:forms} prices it at $26$ more true links against a precision of $0.77$ instead of $0.96$, and \approach buys the reach and leaves the precision to the judge, because a component named \texttt{Common} or \texttt{Client} needs the sentence read before the mention can be ruled on at all.
- now:
  > Scanning at \emph{any case} rather than at \emph{as spelled} buys reach at the cost of precision, and \approach leaves that precision to the judge, because a component named \texttt{Common} or \texttt{Client} needs the sentence read before the mention can be ruled on at all.

- was:
  > It does not require the antecedent to name the component: demanding that costs and buys nothing on the pairs \linkerC alone contributes ($\mathrm{TP}$ $\pm 0.0$, $\mathrm{FP}$ $\pm 0.0$ replayed over the runs' own resolutions), so \approach states the requirement to the resolver and leaves the verdict to the judge.
- now:
  > It does not require the antecedent to name the component, so \approach states the requirement to the resolver and leaves the verdict to the judge.

- was:
  > \approach therefore leaves ordinary-English ambiguity to the judges, which see the sentence itself, and keeps one table instead of two.
- now:
  > \approach leaves ordinary-English ambiguity to the judges, which see the sentence itself.

