# Plan: Rewrite metric.tex — add F2, cut verbosity, restructure

## Context

`sections/metric.tex` (172 lines) defines the architecture-driven evaluation suite. Two changes:
(1) cut dry/verbose prose (repeats motivation.tex, over-formalizes, hedges);
(2) add F2 everywhere — in TLR, missing a link matters more than a false alarm.

This plan covers **metric.tex only**. Follow-up pass cascades F2 into results/eval/intro/tables.

## Resolved decisions

| Decision | Choice |
|----------|--------|
| F2 scope | F2 everywhere (link-level + per-component) |
| F2 equation | General F_β, name F1 (β=1) and F2 (β=2) |
| Notation table | Cut entirely |
| Prestudy | Compress to 1 paragraph |
| SFM → renamed | **Component miss rate (CMR)** via macro `\cmrname` / `\cmr` |
| Doc-code / doc-model split | Structural: separate paragraphs with directory-expansion reasoning |

## SFM → Component miss rate (CMR)

**Why rename:** "silent-failure mass" has three problems: (1) "mass" is physics jargon with
no meaning in SE; (2) "silent-failure" is a compound that takes parsing; (3) the name
doesn't tell the reader what the metric measures.

**Why CMR:** The failure unit is the *component* — a component the system completely fails to
reach (R_k = 0). "Component miss rate" names that failure directly: did you miss a component
entirely? "Rate" is the standard suffix for a [0,1] fraction. The formula weights each missed
component by its documentation share (see below), but the thing being counted as failed is the
component — so the name tracks the failure unit, not the weighting.

**Formula unchanged (doc-weighted):** CMR keeps the SFM/DMR formula — share of documented
sentences belonging to zero-recall components — NOT a raw component count. The weighting keeps
CMR sensitive to *how much documentation* a missed component carries: missing a heavily
documented component costs more than missing a marginal one.

**Macros** (defined in `main.tex` or a `commands.tex` preamble):
```latex
\newcommand{\cmrname}{component miss rate}
\newcommand{\cmr}{\mathrm{CMR}}
```
One `\renewcommand` changes the name paper-wide. The label stays `eq:sfm` for now (rename
in the follow-up pass when all files are updated).

## Doc-code vs doc-model: the directory expansion split

The suite's structure is organized by TASK, not by metric. The reason:

- **Doc-code** has directory expansion: a directory-level annotation expands to every file in the
  directory, creating a long-tailed link distribution (1.0×–217.6× factor). This long tail
  is what makes link-level F_β blind to small components. → The tail metrics (worst-component
  F_β, harmonic F_β) expose what the long tail hides.

- **Doc-model** has NO directory expansion: each link is one sentence → one component, directly annotated.
  The per-component distribution is flatter, so worst-component and harmonic F_β track
  link-level F_β almost exactly — they add nothing. But a system can still completely fail to
  reach a component (R_k = 0). → The CMR catches this binary failure, weighted by how much
  documentation the missed component represents.

This directory-expansion reasoning is the ORGANIZING PRINCIPLE of §4.2. The section should
make the reader feel: "of course doc-code needs tail metrics, and of course doc-model needs
a different one."

## Labels

**Keep**: `sec:metric`, `sec:metric:prestudy`, `sec:metric:suite`, `eq:cov`, `eq:worst`, `eq:harm`, `eq:sfm`.
**Add**: `eq:fbeta` (new general F_β equation).
**Drop**: `eq:ref-f1`, `eq:comp-f1`, `tab:metric-notation` (all self-contained in metric.tex).

## Wang moves applied

| Move | Description | Where applied |
|------|-------------|---------------|
| **M8** "This is because" | Each metric followed by a one-sentence causal link to the inequality | Each metric ¶ |
| **M5** (i)/(ii)/(iii) enumeration | List suite members before defining them | §4.2 ¶1 |
| **M11** no-qualifier declarative | No "we believe" / "we argue" | Every sentence |
| **M12** forward pointer | Close with forward ref to RQ where metric is used | §4.2 ¶6 |
| **T5** single-sentence scope | Open section with one scope sentence | §4 opener S1 |
| **T6** walkthrough: DEF→EQ→INTERP | Definition → equation → interpretation per metric | Each metric ¶ |
| **B1** front-loaded subject | Subject = the metric or the system, active verb | Every definition |
| **B3** -ing consequence tail | Attach payoff as participial tail | Metric reads |
| **C2** colon-then-list | Colon before inline enumeration or equation | §4.2 ¶1, ¶3, ¶4 |
| **D1** strong verb | "exposes / catches / rewards / hides / abandons" | Every sentence |
| **A1** number+mechanism+consequence | Pack Gini + interpretation into one clause | §4.1 |

ICSE expectations (SECTION_PLAYBOOK §5.B):
1. Justify why each metric is needed (deficiency of the standard metric)
2. Define precisely with formula
3. State range and interpretation
4. Every metric must "do work" in §Results

---

## Sentence-level spec

### SECTION OPENER — §4 heading + 3 sentences

```
\section{Architecture-Driven Evaluation}
\label{sec:metric}
```

**S1.** [T5 scope] [BRIDGE from sec:motivation]
> The link distribution is long-tailed, so link-level metrics hide whether a system
> covers every documented component.
- Wang move: T5 single-sentence scope. PREMISE (long-tailed) → CONSEQUENCE (hides) in one sentence.
- Constraint: do NOT re-explain "long-tailed" — motivation.tex did that.

**S2.** [T5 continued] [SCOPE for this section]
> This section quantifies the concentration across all five projects and builds a metric
> suite that exposes silent component failures.
- Wang move: T5 scope, naming the two subsections ("quantifies" = §4.1, "builds" = §4.2).
- Arg logic: SCOPE.
- Constraint: "silent component failures" = the defined term from motivation.tex.

---

### §4.1 LINK CONCENTRATION — 1 paragraph + table

```
\subsection{Link Concentration in the Benchmark}
\label{sec:metric:prestudy}
```

[PURPOSE: quantify the skew across all five projects. Only NEW info vs motivation.tex =
 the five-project numbers + table. Do NOT re-explain directory expansion mechanics or the JabRef
 example — motivation.tex did both.]

**S1.** [A1 number+mechanism+consequence] [EVIDENCE]
> The directory expansion factor — how many file-level links one directory annotation produces —
> spans from 1.0× on MediaStore to 217.6× on JabRef.
- Wang move: A1 packs range + mechanism. C1 em-dash defines "directory expansion factor" inline.
- Data: inequality_expansion.csv (prose-only, not in tab:gold_concentration).
- Constraint: ONE sentence for directory expansion. No second sentence explaining what it is.

%NT: reader not know that, but we also no explain in formula, only intuition of gini here, make it understanable while no detail formula
**S2.** [A1] [EVIDENCE — table reference]
> Within each project the expanded links concentrate in a few components:
> \autoref{tab:gold_concentration} reports per-component Gini coefficients from 0.474
> (TeaStore) to 0.591 (JabRef), and on JabRef three of six components hold 99.3% of
> the links.
- Wang move: A1 (Gini + project + % in one clause). C2 colon introduces the table.
- Data: tab:gold_concentration, columns Gini and Top-3%.

**S3.** [M8 consequence] [CLAIM — punchline]
> Link-level metrics therefore mainly reflect these few large components, hiding whether
> the system recovers the rest.
- Wang move: M8. M11 no-qualifier declarative.
- Arg logic: CONSEQUENCE of S1+S2. "Hiding" = the deficiency, transitions to §4.2.

**S4.** [BRIDGE to §4.2]
> A suite that weights each component equally exposes what the link-level average buries.
- Wang move: M11 declarative. B1 subject = "a suite".
- Constraint: "exposes" echoes opener S2. No "we therefore propose."

**FLOAT**: `\input{table/gold_concentration}` — unchanged.

---

### §4.2 THE METRIC SUITE

```
\subsection{The Metric Suite}
\label{sec:metric:suite}
```

[PURPOSE: define the suite. ORGANIZED BY TASK — doc-code then doc-model — with
 directory-expansion reasoning as the structural principle.

 Structure:
   ¶1  F_β definition + inline metric preview (both tasks)
   ¶2  Sentence coverage (both tasks — no task-specific reasoning needed)
   ¶3  Doc-code tail: worst-component + harmonic F_β (directory expansion → long tail → need tail metrics)
   ¶4  Doc-model: CMR (no directory expansion → flat distribution → tail metrics redundant → need CMR)
   ¶5  Suite summary + forward pointer
]

#### ¶1 — F_β definition + inline metric preview (5 sentences + 1 display equation)

**S1.** [T5 scope] [M5 preview]
> The suite keeps link-level F_β as the reference metric and adds four size-aware
> metrics — sentence coverage, worst-component F_β, harmonic-mean F_β, and the
> \cmrname{} — each giving every gold component the same weight.
- Wang move: T5 + M5 (4 items in em-dash appositive). B3 -ing tail ("each giving...").
- Arg logic: SCOPE + DEFINITION.
- Constraint: use `\cmrname{}` macro for the fourth metric. Name all four before any formula.

**S2.** [DEFINITION — F_β family]
> We define the family over the F_β score,
- Lead-in for the display equation.

**DISPLAY eq:fbeta:**
```latex
\begin{equation}
F_\beta = \frac{(1+\beta^2)\,P\,R}{\beta^2\,P + R},
\label{eq:fbeta}
\end{equation}
```

**S3.** [A2 metric-as-adjective] [DEFINITION — name F1, F2]
> where F_1 (β = 1) weights precision and recall equally, and F_2 (β = 2) weights
> recall twice as heavily.
- Wang move: A2. Parallel "weights...equally" / "weights...twice" makes the asymmetry land.

%argument should be past study argue recall more important in TLR tasks, put cite and xxx place holder
**S4.** [M8 cause — F2 motivation, part 2: the use case] [CLAIM → CAUSE]
> The suite reports both: a developer who uses trace links for impact analysis cares
> more about finding every component (recall) than about filtering false alarms (precision).
- Wang move: M8 (colon carries the causal link). M11 no-qualifier.
- Arg logic: CLAIM → CAUSE.
- Constraint: this is the SECOND and LAST F2 motivation sentence. Part 1 was §4 S3
  (the principle). Together: S3 = missed > false alarm, S4 = because impact analysis.
  Do NOT add more F2 justification.

**S5.** [DEFINITION — per-component F_β] [inline, no display equation]
> Each gold component k owns a set of code files; F_β(k) denotes F_β computed over
> only the links whose target belongs to k.
- Wang move: B5 appositive compression.
- Constraint: NO display equation — F restricted to a subset is standard notation.

#### ¶2 — Sentence coverage (2 sentences + 1 display equation)

```
\paragraph{Sentence coverage.}
```

[NOTE: sentence coverage applies to BOTH tasks. No task-specific reasoning needed here.]

**S1.** [T6: DEFINITION → equation] [ICSE §5.B: formula]
> Sentence coverage is the fraction of gold sentences for which the system recovers
> at least one correct link:
- Wang move: B1 front-loaded subject. C2 colon. M11 declarative (IS, not "we define as").

**DISPLAY eq:cov:** [KEEP existing equation verbatim]
```latex
\begin{equation}
\mathrm{cov} = \frac{1}{|\mathcal{S}|}\sum_{s \in \mathcal{S}}
  \mathbf{1}\!\left[\text{the system recovers a correct link for } s\right].
\label{eq:cov}
\end{equation}
```

**S2.** [M8 mechanism] [ICSE §5.B: interpretation]
> It measures developer-facing reach — whether the system finds something for each
> documented sentence, regardless of how many links that sentence carries.
- Wang move: M8. C1 em-dash gloss.
- "Developer-facing reach" = the takeaway. "Regardless of how many links" = what link-level
  F_β cannot show (the M8 cause).

#### ¶3 — Doc-code tail metrics (6 sentences + 2 display equations)

```
\paragraph{Doc-code: worst-component and harmonic-mean \texorpdfstring{$F_\beta$}{F-beta}.}
```

[KEY STRUCTURAL MOVE: the paragraph heading names the TASK (doc-code) so the reader
 immediately knows these metrics are task-specific. The directory expansion reasoning follows
 in S1 as the elegant justification.]

**S1.** [CLAIM → CAUSE — the directory-expansion justification for doc-code tail metrics]
> On the doc-code task, directory expansion maps each directory annotation into file-level links,
> creating the long-tailed distribution that lets a system score high on link-level F_β
> while missing a whole component.
- Wang move: M8 (cause-consequence packed into one). B3 -ing tail ("creating...").
- Arg logic: PREMISE (directory expansion) → CAUSE (long tail) → CONSEQUENCE (high F_β + missed component).
- Constraint: this is THE sentence that justifies why doc-code needs tail metrics. It packs
  three ideas into one sentence via the -ing tail. The reader should think: "directory expansion →
  long tail → link-level F_β is blind → need tail metrics."
- Cite: [du2023noone] for worst-category practice, [sagawa2020distributionally] for
  worst-group robustness. Attach citations HERE, not to the equation sentences.

**S2.** [T6 DEFINITION] [ICSE §5.B: formula]
> The worst-component F_β is the minimum over the gold components:
- Wang move: B1 front-loaded subject. C2 colon before equation. M11 declarative.

**DISPLAY eq:worst:**
```latex
\begin{equation}
F_\beta^{\min} = \min_{k \in \mathcal{K}} F_\beta(k).
\label{eq:worst}
\end{equation}
```
[REVISED: F_β(k) replaces F_1(k).]

**S3.** [M8 mechanism] [ICSE §5.B: interpretation]
> A single missed component drives it to zero, even when link-level F_β stays high.
- Wang move: M8. D2 "even" as a-fortiori compressor.
- Constraint: SHORT punch line. "Drives to zero" = the sharp consequence.

**S4.** [T6 DEFINITION — harmonic mean] [ICSE §5.B: formula]
> Because a single minimum can be noisy, we also report the harmonic mean of the
> per-component F_β:
- Wang move: B6 "Because..." opener. C2 colon.
- Constraint: "also report" — not "we propose." Only "we" as subject in the whole section.

**DISPLAY eq:harm:**
```latex
\begin{equation}
F_\beta^{H} = \frac{|\mathcal{K}|}{\sum_{k \in \mathcal{K}} 1/F_\beta(k)}.
\label{eq:harm}
\end{equation}
```
[REVISED: F_β(k) replaces F_1(k).]

**S5.** [M8 mechanism + interpretation]
> It down-weights every low-scoring component, not just the worst, and also collapses
> to zero when any component is missed entirely.
- Arg logic: INTERPRETATION. "Collapses to zero" echoes S3 — both metrics punish total failure.

**S6.** [BRIDGE — the two metrics agree]
> The two agree on the system ranking.
- Constraint: SHORT. Justifies reporting both (they're not contradictory). One sentence.

#### ¶4 — Doc-model: component miss rate (4 sentences + 1 display equation)

```
\paragraph{Doc-model: \cmrname.}
```

[KEY STRUCTURAL MOVE: the paragraph heading names the TASK (doc-model) and the metric,
 mirroring ¶3's "Doc-code: ..." heading. The reader immediately sees: different task,
 different metric.]

[NOTE: Do NOT explain Spearman correlations, do NOT say "uninformative." Give the ELEGANT
 reason: doc-model has no directory expansion, so the distribution is not long-tailed, so the tail
 metrics add nothing. Instead, measure the binary failure: did you reach this component?]

**S1.** [CLAIM → CAUSE — the directory-expansion justification for CMR]
> Doc-model links are sentence-to-component with no directory expansion: the per-component
> distribution is not long-tailed, so the worst-component and harmonic F_β add little
> over link-level F_β.
- Wang move: M8 (cause → consequence). M11 declarative.
- Arg logic: PREMISE (no directory expansion) → CAUSE (not long-tailed) → CONSEQUENCE (tail metrics
  redundant).
- Constraint: THIS is the elegant reason the user asked for. One sentence. The logic chain:
  doc-model has no directory expansion → distribution is flatter → tail metrics track link-level F_β →
  they add nothing new. Do NOT cite Spearman numbers. The reason is STRUCTURAL (no directory expansion),
  not statistical (high correlation).

**S2.** [CONTRAST — what doc-model DOES need] [T6 DEFINITION]
> But a system can still miss a component entirely, leaving its documented sentences
> with no correct link.
- Wang move: CONTRAST ("But"). B3 -ing tail ("leaving...").
- Arg logic: CONTRAST → CONSEQUENCE.
- Constraint: sets up the CMR definition. "Miss a component entirely" = R_k = 0.
  "Leaving its documented sentences with no correct link" = what the developer experiences.

**S3.** [T6 DEFINITION — formal] [ICSE §5.B: formula]
> The \cmrname{} (\cmr) measures the share of documented sentences belonging to
> such components:
- Wang move: B1 front-loaded subject. B5 appositive for acronym. C2 colon before equation.
- Arg logic: DEFINITION.
- Constraint: use `\cmrname{}` and `\cmr` macros. "Such components" = the zero-recall
  components from S2. The colon promises the equation.

**DISPLAY eq:sfm:** [label kept for cross-ref compatibility; rename in follow-up pass]
```latex
\begin{equation}
\cmr = \frac{\bigl|\bigcup_{k:\,R_k=0}\, \mathcal{S}_k\bigr|}
             {\bigl|\bigcup_{k \in \mathcal{K}}\, \mathcal{S}_k\bigr|}
  \in [0,1].
\label{eq:sfm}
\end{equation}
```
[REVISED: `\cmr` replaces `\mathrm{SFM}` in the equation. Label `eq:sfm` kept for now.]

**S4.** [M8 interpretation] [ICSE §5.B: range + interpretation]
> \cmr{} = 0 means every documented component is reached — no component's documentation
> is silently lost.
- Wang move: M8. C1 em-dash gloss.
- Arg logic: INTERPRETATION.
- Constraint: "silently lost" echoes the section opener's "silent component failures" without
  using the old "silent-failure mass" name. Closes the loop.

#### ¶5 — Suite summary + forward pointer (3 sentences)

[PURPOSE: tell the reader exactly what results tables will contain. M12 forward pointer.]

**S1.** [M12 — doc-code suite]
> On doc-code the suite reports sentence coverage, worst-component F_β, and
> harmonic-mean F_β, each at both β = 1 and β = 2.
- Wang move: M12 (implicit forward pointer — names what's in the results table).
- Constraint: enumerate in the ORDER they appear in tab:rq2.
  "Each at both β = 1 and β = 2" = the F2 payoff.

**S2.** [M12 — doc-model suite]
> On doc-model it reports the \cmrname.
- Constraint: SHORT. One sentence.

**S3.** [M12 — explicit forward pointer]
> \autoref{sec:exp:rq2} applies this suite to all systems on the benchmark.
- Wang move: M12 explicit section reference.
- Constraint: do NOT preview results.

---

## Macros to add (in main.tex preamble or commands.tex)

```latex
% Component miss rate — the doc-model size-aware metric (formerly SFM).
% Change the name paper-wide by editing these two lines:
\newcommand{\cmrname}{component miss rate}
\newcommand{\cmr}{\mathrm{CMR}}
```

---

## Tone rules

- Average 15–20 words/sentence; hard cap ~30. Total: 22 sentences.
- Lead with consequence, then formula, then one-line interpretation (M8).
- No "we therefore" / "we propose" / "we argue" / "we believe" — declarative (M11).
- No elegant variation: "component" not "entity"; "links" not "pairs"; "suite" not "battery".
- Strong verbs (D1): "exposes / catches / rewards / hides / buries / abandons / drives / collapses".
- Active voice (B1). Subject = metric or system. Passive only for setup.
- One em-dash gloss per sentence max (C1 caveat).
- Two F2 motivation sentences total: §4 S3 (principle) + §4.2 ¶1 S4 (use case). No more.
- Doc-code / doc-model split: ¶ headings name the task. Enrollment reasoning in first sentence of each.

---

## Verification

1. `grep -rn 'eq:ref-f1\|eq:comp-f1\|tab:metric-notation' sections/ appendix/ main.tex` — no dangling refs
2. `grep -rn 'eq:fbeta' sections/` — new label exists
3. All kept labels present: `sec:metric`, `sec:metric:prestudy`, `sec:metric:suite`,
   `eq:cov`, `eq:worst`, `eq:harm`, `eq:sfm`
4. `\cmrname` and `\cmr` macros defined before `\begin{document}`
5. Read end-to-end: opener → prestudy (1¶) → F_β → coverage → doc-code tail → doc-model CMR → summary
6. Sentence count: 22 (current ~35)
7. Word count: ~400–500 (current ~700)
8. Equations: 5 display (eq:fbeta, eq:cov, eq:worst, eq:harm, eq:sfm)
9. Floats: 1 table (tab:gold_concentration)
