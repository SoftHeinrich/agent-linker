# ICSE-style review checklist — Introduction & Background/Motivation

Captured 2026-06-29 from a 5-reviewer ICSE-mimic review of `sections/intro.tex` +
`sections/motivation.tex`. **Reviewed the rendered PDF prose only** (LaTeX comments,
`%TODO`/`%SPEC`, and the carried-over JK block were stripped from the reviewers' input so
they could not bias the panel). **Abstract excluded** (unfinished). Reviewers were told the
body supplies the standard apparatus (N=3 runs mean±std, exact model versions/settings, full
Artemis/SWATTR/TransArc comparison + ablations, formal metric defs), so "missing
variance/comparison/definitions" was off the table.

**Panel outcome:** unanimous **Weak Accept (+1 ×5)**. (A first, biased run that leaked our own
comments produced 4× Weak Reject — those rejections were mostly manufactured by the leak and
do not appear here.)

Status tags: **[decide]** authorial call · **[mech]** mechanical once a target is picked · **[chore]** housekeeping.

---

## Confirmed OK — do NOT re-touch (panel cleared these)
- **The "the UI" illustrative alias** (`motivation.tex:44-46`). R3 praised the running example
  as "consistent and well threaded"; no clean reviewer flagged it. It reads fine as an
  illustration — leave it.
- **Novelty vs Artemis** (`intro.tex:37`). R2 (domain Expert) affirms the delta is "real and
  conceptually well-grounded… not a re-skin." The qualified "first… that separates… and checks…"
  framing survives. (One refinement remains — see C4.)

---

## Tier 1 — fixes that move the verdict (all prose-only, no new runs)

- [ ] **C1. Headline is reported in the metric the paper discredits.** [decide]
  `intro.tex:52` sells doc-code **+5.7pp**, but doc-code = link-level \fone — the exact metric
  `intro.tex:57` calls "hides the failures that matter." R1/R2/R4 (3/5) all flagged the
  self-undercut. **Fix:** at `:51-52` also report the doc-code gain under the size-aware suite
  (or state plainly why link-level is the fair head-to-head here). Turns a contradiction into support.

- [ ] **C2. doc-code \fone (0.849) > doc-model \fone (0.836) contradicts "doc-model is the bottleneck."** [decide]
  `intro.tex:14-15`. A composition of doc-model with sub-1.0 model-code scoring *above* its
  weaker factor reads as a contradiction (R1/R2/R4). **Fix:** add one clause noting this is the
  file-expansion inflation your Ch2 metric critique predicts — so it *reinforces* the metric
  story instead of looking like an error.

- [ ] **C3. The baseline is never named.** [mech once picked]
  `intro.tex:15` "the strongest published pipeline" (0.836/0.849) is never mapped onto Artemis
  (`motivation.tex:111-112`, 0.998) or TransArc (`:112`, 0.943). R1/R3/R4. **Fix:** name the
  system once at `:15`; state how SWATTR / Artemis / TransArc relate so the reader isn't juggling
  four labels with mismatched numbers.

- [ ] **C4. "Score each sentence in isolation" may be a strawman on the load-bearing claim.** [decide — needs fact-check]
  `intro.tex:27` (and the Artemis description `:23`). The whole novelty delta hinges on "isolation."
  R2: if Artemis's per-sentence call actually receives the document or component list, "isolation"
  overstates it. **Fix:** verify what context Artemis feeds the LLM; if it sees more than one
  sentence, replace "score each sentence in isolation" with the precise limitation ("recognizes
  named mentions but performs no antecedent resolution / no persisted disambiguation"). Novelty
  still wins; the overstatement is the only risk.

- [ ] **C5. De-totalize the LLM overclaims.** [mech]
  `intro.tex:45` "\linkerB … recovers **every** named reference" → "targets named references"
  (or a measured-recall claim). `intro.tex:47` "rejects **these** hallucinated links" — dangling
  "these" (hallucination not yet introduced) → "rejects any link the evidence does not support."
  R1/R2/R3/R4 (4/5). Matches the existing TODO at `intro.tex:36(a,b)`.

---

## Tier 2 — measurement contribution (R5, deepest single review)

- [ ] **M1. "Link-level \fone is the wrong metric" is a construct-invalidity overclaim.** [decide]
  `intro.tex:60`. Micro-\fone is *valid*; it answers a *different* question. **Fix:** reframe to a
  construct mismatch and **name the stakeholder/decision** the size-aware suite serves (e.g., an
  architect who pays a real cost when a whole component is dropped). Without that anchor, "wrong"
  is rhetoric. Same point in `motivation.tex:62-63` opening.

- [ ] **M2. The skew is partly a benchmark artifact, not a metric defect.** [decide]
  `motivation.tex:73-77` itself says the annotation "expands to the link level — one link for
  every file." So 99.3%/0.4% is *manufactured by the gold-expansion convention*. **Fix:** argue
  explicitly why a new *aggregation* over expanded file-links beats simply scoring at the
  sentence–component unit humans annotated. Right now expansion is described, then treated as
  immutable.

- [ ] **M3. Position the three metrics as the known constructs they are.** [decide]
  `intro.tex:63` / `metric.tex`. Sentence coverage = class-coverage/recall; worst-component \fone
  = worst-group/minimax (DRO fairness); harmonic-mean-of-per-component-\fone = low-value-penalizing
  macro-aggregate, kin to TREC GMAP. R2 + R5. **Fix:** cite macro-averaging / worst-group accuracy
  / GMAP and claim the **architecture-aware framing** as the contribution, not metric novelty.

- [ ] **M4. "46×" crosses units, and 20/0.4 = 50 ≠ 46.** [decide/mech]
  `motivation.tex:96-97` compares 20% **share-of-sentences** vs 0.4% **share-of-file-links** —
  different unit spaces (R1 + R5). The note at `:99-100` explains 46 = 20/0.4352 (unrounded), but a
  reader sees 20/0.4 = 50 and distrusts it. **Fix:** compare like-with-like, or drop the multiplier
  and show both raw shares with units made explicit.

- [ ] **M5. doc-model vs doc-code task conflation in the motivation.** [decide]
  `intro.tex:62` "a fifth of the document's sentences" is **doc-model** (no file expansion);
  `motivation.tex:93-94` "0.4%" is **doc-code** (expansion). R5. **Fix:** keep the two granularities
  separate; state the expansion argument applies only to doc-code. For doc-model the missed
  component is a component-level recall miss that **macro-\fone-over-components already captures** —
  so show what the suite adds *beyond* macro-\fone there.

---

## Tier 3 — clarity & presentation (R3)

- [ ] **P1. Two co-equal bold theses dilute the headline.** [decide]
  `intro.tex:31` (workflow) and `:57` (metric) carry equal weight; the title + contribution (1)
  foreground the workflow. **Fix:** keep `:31` bold; demote `:57` to plain and present it as the
  motivation for contribution (2).

- [ ] **P2. "architecture model" undefined on first use; collides with "LLM".** [mech]
  `intro.tex:6` "the matching architecture model" — add one clause ("the set of components and
  their relations"). Watch `:43` "from the document and model with \acp{LLM}" where *model* and
  *LLM* sit in one clause.

- [ ] **P3. The two "three-item" lists collide.** [decide]
  `intro.tex:26-29` (three reasons prior work fails) vs `:39-41` (three reasons knowledge is hard);
  `:39(ii,iii)` partly restate `:27-28`. Already flagged in the SPEC at `intro.tex:35`. **Fix:**
  signpost `:26-29` as the failure triple that maps 1:1 onto the design, and give `:39-41` a clearly
  different label ("challenges in the knowledge step").

- [ ] **P4. No overview/architecture figure.** [decide]
  The pipeline (knowledge module → DLinker/ILinker → per-linker judge) is carried entirely by dense
  prose in `intro.tex:37-48`; Fig 1 is an example, not an architecture. **Fix:** a small workflow
  diagram would offload the terminology load (≈12 new terms before any anchor).

- [ ] **P5. Number dump in the eval motivation.** [mech]
  `motivation.tex:91-113` packs ~7 figures (0.998, 0.943, 99.3%, 0.4%, 20%, 46×, 0.800). **Fix:**
  keep the two that carry the argument (zero-on-preferences vs weakest-0.800); move the rest to results.

- [ ] **P6. "As simple as supplying the document and the model" understates the system.** [mech]
  `intro.tex:50` — same paragraph lists an ambiguity map, alias table, two linkers, two judges.
  R4. **Fix:** rescope to the *interface* ("one invocation, no labels"), separate from the internal
  multi-module workflow.

### Line-by-line nits
- [ ] `intro.tex:16` "precision and recall are each well short of complete, so the output both
  misses true links and includes wrong ones" — near-vacuous restatement of \fone<1 (R1). Cut or quantify.
- [ ] `intro.tex:50` "no fine-tuning, no labeled trace links, and no benchmark-specific tuning" —
  three near-synonyms; collapse, and **scope** "no benchmark-specific tuning" (define what counts as
  tuning for a prompt-engineered pipeline developed on these 5 projects). R4-W1.
- [ ] `intro.tex:21-22` SWATTR "hand-written linguistic patterns" / "word for word" slightly undersells
  its linguistic/dependency machinery — soften to avoid an easy rebuttal (R2).
- [ ] `intro.tex:63` / `:72` "architecture-driven" and "size-aware" label the same suite — pick one.
- [ ] DLinker mnemonic: "D" is unmotivated against "named-mention linker" — gloss as "DLinker, for
  direct/named mentions" (R3).
- [ ] `intro.tex:12` still carries a `%TODO` on the model-code 0.98 number — verify/pin. [chore]

---

## Top 4 highest-leverage (do these first)
1. **C1 + C2** — reconcile the two contributions (report doc-code under the suite; connect the
   0.849>0.836 inversion to the metric story). Single biggest win.
2. **C3** — name the baseline.
3. **C4** — pin the Artemis "isolation" claim.
4. **C5** — de-totalize the overclaims.
