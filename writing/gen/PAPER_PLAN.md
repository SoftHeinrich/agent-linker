# AALinker Paper Plan (ICSE)

Two co-equal contributions:
1. **AALinker** — multi-agent trace-link recovery between architecture doc and code. Code: `approach/`.
2. **Metric suite** — corrects three structural biases in the established benchmark. Data: `evaluation/`.

Target venue: ICSE (10 + 2 page format, IEEE conference template).
Existing skeleton: `writing/writen-paper/sections/{intro, prestudy, approach, metric, eval, results, discussion, rw, conclusion}.tex` + `main.tex`.
Convention: AI drafts under `writing/gen/`, human edits under `writing/writen-paper/`.

---

## Section-by-section plan

### 1. Introduction — `gen/intro.tex` [drafted, ~1 page]

**Purpose:** Frame the problem, preview both contributions, plant the +12.9pp / +22.7pp forward-reference.

**Content (5 paragraphs):**
1. Problem framing — why doc-to-code trace links matter. [TODO]
2. Prior-work limitations — lexical pipelines (SWATTR) miss aliases/pronouns; single-pass LLMs (LiSSA) lack runtime knowledge. [TODO]
3. AALinker glimpse — knowledge layer + four agents + two validators; headline number (file F1 0.931 vs 0.803). [TODO]
4. **Metric glimpse — DONE.** Six-sentence CrystalBLEU-shaped paragraph; structural-inequality diagnosis; named suite; +12.9 / +22.7 forward-reference; co-equal stance.
5. Contribution list + roadmap. **DONE.**

**Page budget:** 1 page.

### 2. Pre-study — **DROPPED.**

Motivation merged into metric chapter (full version) and intro (glimpse version). The empirical bias evidence (5 inequality sources, JabRef 98.6\% concentration, cascade asymmetry) is the metric chapter's motivation and does double duty. Update `writen-paper/main.tex` to remove `\input{sections/prestudy}` and delete (or archive) `prestudy.tex`. Frees ~0.5 page.

### 3. AALinker Approach — `writen-paper/sections/approach.tex` [human-authored, partial]

**Status:** human draft exists; key content present (knowledge discovery, four agents, validation patterns, strategies table). Reads as ~70% done.

**Gaps to fill (human):**
- `\todo{define runtime}`, `\todo{cite source}` markers throughout.
- §Architectural Model Understanding — partial-word discussion needs example.
- §Architectural Document Understanding — examples for synonym/abbreviation/partial categories.
- Pipeline figure (referenced as Figure 1; check `pictures/`).

**Page budget:** 2.5–3 pages.

**Data:** all in `approach/src/`. Key prompts in `approach/prompts/`. Phase contributions documented in `approach/PHASE_CONTRIBUTION_ANALYSIS.md`.

### 4. Metric Suite — `gen/metric.tex` [motivation drafted, definitions stub]

**Purpose:** Formal home of the metric contribution. Reader-buys-it section.

**Structure (4 subsections):**
1. **Background and Motivation — DONE.** Background paragraph + three structural inequality sources + what this means for F1 + stance with forward-reference.
2. **Metric Suite — STUB.** Six metrics in plain English:
   - Per-component F1 (down-weights file links by component size)
   - Per-sentence F1 (averages F1 over gold sentences)
   - Sentence coverage / noise rate (developer experience)
   - Coverage-and-purity (harmonic mean of two)
   - Skill score (rescaled between random and oracle)
   - Decision-level F1 (matched to human-decision granularity)

   Each: name → one-sentence motivation → formula → one-sentence intuition. No more than 3 lines per metric.
3. **Worked example.** One sentence on JabRef showing how each metric scores TransArc differently. (CrystalBLEU move.)
4. **Metric assessment.** Short table or paragraph: which metrics agree with F1, which diverge, which are redundant. Source: `evaluation/reports/CREATIVE_METRICS.md`, `HOLISTIC_METRICS.md`.

**Page budget:** 1.5–2 pages.

**Data source:** `evaluation/writing/eval.tex` has the long-form definitions and per-project tables. Port and shorten to ICSE length.

### 5. Experiment Design — `gen/eval.tex` [drafted]

**Status:** four RQs + per-RQ design + dataset + replication. ~85% done.

**Gaps:**
- Verify TransArc / SWATTR / LiSSA baseline numbers exist in `evaluation/` or are cited from authors. LiSSA may need new runs — **CHECK.**
- `\input{table/dataset-stat}` placeholder — needs the table file (build from `evaluation/lib` loaders).
- `\input{table/enrollment}` referenced in `metric.tex` — same.

**Page budget:** 1.5 pages.

### 6. Results — `gen/results.tex` [empty]

**Purpose:** Answer the four RQs. This is the body of the paper.

**Structure (one subsection per RQ):**

**RQ1 — SOTA comparison.** Two tables:
- Doc-to-model F1/P/R for AALinker vs TransArc(SWATTR) vs LiSSA across 5 projects.
- Doc-to-code F1/P/R for the same.
Source: `evaluation/reports/SADSAM_S11_S13F_VS_TRANSARC.csv` (doc-to-model) and `SADCODE_S11_S13F_VS_TRANSARC.csv` (doc-to-code). **LiSSA numbers TODO.**

Headline numbers (verified): file F1 0.803 → 0.931 (+12.9pp), decision F1 0.596 → 0.823 (+22.7pp).

**RQ2 — Architecture-driven metrics.** One grid figure (4 systems × 5 projects × full metric panel) and one summary table with mean |Δ| from standard F1. The mechanism story is the gap-shape: largest under decision/weighted F1, smallest under component F1 (due to Teammates regression).

Source: `evaluation/reports/HOLISTIC_METRICS.md`, all metrics columns in the CSV.

**RQ3 — Validator contribution.** Table: Full vs NoConsensus vs NoCitation vs NoValidator on 5 projects × {F1, per-component F1, coverage, noise} + token-cost column.

Source: `approach/results/ablation_results/*.json`. **CHECK if NoConsensus / NoCitation variants exist; if not, run them.**

**RQ4 — Per-agent ablation.** Table: $-$Explicit / $-$Contextual / $-$Anaphoric / $-$Abbreviated vs Full + Explicit-only floor.

Source: `approach/results/ablation_results/*.json`, `approach/PHASE_CONTRIBUTION_ANALYSIS.md`.

**Page budget:** 2.5 pages.

### 7. Discussion / Threats — `gen/discussion.tex` [empty]

**Purpose:** Honest accounting of limitations + non-obvious findings.

**Content (4 subsections):**

1. **Why AALinker wins more on the new metrics.** The mechanism: AALinker's gains concentrate on per-decision granularity (where the benchmark's structural inequality hides them under file-level averaging).
2. **The Teammates regression at component level.** Honest discussion of where AALinker underperforms TransArc on component F1, and what it implies. (Likely cause: AALinker's stricter validators kill some TransArc TPs on a project with many sentences linked to common-English-name components.)
3. **Threats to validity.**
   - *Construct:* LLM non-determinism; mitigated by fixed seed / two-pass consensus.
   - *Internal:* prompt engineering on benchmark = leakage risk; mitigated per `approach/BENCHMARK_TABOO.md`.
   - *External:* five Java projects, one benchmark; the metric suite is task-agnostic but AALinker validation is on one corpus.
   - *Conclusion:* statistical reliability of file-level F1 reported via raw-decision CI (see `evaluation/reports/EVALUATION_CRITIQUE.md`).
4. **Implications for the field.** Recommended evaluation protocol: report standard F1 plus at least per-component F1, sentence coverage, and noise rate.

**Page budget:** 1 page.

### 8. Related Work — `gen/rw.tex` [empty]

**Three threads:**

1. **Architecture-to-code traceability.** SWATTR (Keim et al.), TransArc (Fuchss et al.), LiSSA (Hey et al.). Existing surveys.
2. **LLM-based trace-link recovery.** Recent LLM TLR work (2023–2025). Compare/contrast with AALinker's multi-agent + knowledge-layer approach.
3. **Evaluation critique and metric proposals.** CrystalBLEU (ASE 2022, ours is closest analogue), pass@k (Chen et al.), CodeBLEU, Allamanis dedup, Papadakis mutation, CheckList (ACL 2020). Position our metric suite among these.

Sources for citations: `evaluation/writing/eval.tex` has TransArc/replication bibkeys; `smelly-discussion.bib` exists in writen-paper. AI/ML/NLP refs from `writing/gen/refs/` knowledge base.

**Page budget:** 1 page.

### 9. Conclusion — `gen/conclusion.tex` [empty]

**Content:** Restate the two contributions, the headline numbers (file F1 +12.9pp, decision F1 +22.7pp), and one sentence on the recommended evaluation protocol. 5–6 sentences.

**Page budget:** 0.25 page.

---

## Tables and figures

Mandatory:
- **Table 1 (dataset):** per-project sentences / components / files / raw gold / enrolled gold / enrollment factor. — `writen-paper/table/dataset-stat.tex` to be built.
- **Table 2 (enrollment):** raw → enrolled, 1.0×–217.6× spread. — `gen/metric.tex` placeholder; build under `writen-paper/table/enrollment.tex`.
- **Table 3 (strategies):** already in `approach.tex` (`tab:strategies`).
- **Figure 1 (pipeline):** referenced in `approach.tex`; check `pictures/`.
- **Table 4 (RQ1 doc-to-model):** AALinker vs TransArc/SWATTR/LiSSA × 5 projects × {P, R, F1}.
- **Table 5 (RQ1 doc-to-code):** same structure.
- **Figure 2 (RQ2 grid):** 4 systems × 5 projects × full metric panel (small-multiples).
- **Table 6 (RQ2 summary):** mean |Δ| from standard F1 per metric.
- **Table 7 (RQ3 validators):** 4 variants × 5 projects × 4 metrics + token cost.
- **Table 8 (RQ4 agents):** 5 ablations × {ΔF1, Δcoverage, Δnoise, unique TPs lost}.

Optional:
- **Table 9 (metric assessment):** like Table~III in `evaluation/writing/eval.tex` (high/medium/low value), trimmed.

---

## Data sources and gaps

**Have:**
- `evaluation/reports/SADCODE_S11_S13F_VS_TRANSARC.csv` (doc-to-code, all metrics, TransArc + s11 + s13f).
- `evaluation/reports/SADSAM_S11_S13F_VS_TRANSARC.csv` (doc-to-model).
- `evaluation/reports/metrics_sad-code.csv`, `metrics_sad-sam.csv`.
- `evaluation/writing/eval.tex` — full metric definitions and per-project tables.
- `approach/results/ablation_results/*.json` — agent-level ablation data (s11 family).
- `approach/V31_FINAL_SUMMARY.md` — phase summary.
- `approach/PHASE_CONTRIBUTION_ANALYSIS.md` — per-phase contribution data.

**Gaps (need to obtain or run):**
- **LiSSA numbers** on the five benchmark projects — not in `evaluation/reports/`. **Run or cite from LiSSA paper.**
- **NoConsensus / NoCitation validator-ablation variants** — confirm they exist in `approach/results/`; if not, run.
- **Dataset-stat table** content — derive from `evaluation/lib` loaders.
- **Enrollment table** content — already in `evaluation/writing/eval.tex` as a literal table; port to `table/enrollment.tex`.
- **Pipeline figure** — confirm `writen-paper/pictures/` has one; if not, draw.

---

## Writing order (recommendation)

1. **Now:** fill `gen/intro.tex` TODO paragraphs (1, 2, 3) — needs nothing external, just polish.
2. **Next:** write `gen/metric.tex` §Metric Suite (port from `evaluation/writing/eval.tex` in plain English, short form).
3. **Then:** results — RQ1 (numbers verified), RQ2 (grid figure), RQ4 (data available). RQ3 last because of the validator-variant data gap.
4. **Parallel:** human fills approach.tex todos and provides pipeline figure.
5. **Last:** discussion, related work, conclusion.

---

## Page budget summary (target: 10 pages)

| Section | Pages |
|---|---|
| Intro | 1.0 |
| Approach | 2.5 |
| Metric suite | 1.75 |
| Experiment design | 1.5 |
| Results | 2.5 |
| Discussion / threats | 0.75 |
| Related work | 0.75 |
| Conclusion | 0.25 |
| **Total** | **11.0** |

Over budget by ~1 page (prestudy already removed but the rest is still tight). Remaining levers, in priority order:
- Trim metric definitions; collapse the skill-score and decision-F1 motivations into footnotes.
- Drop the metric-assessment table (keep prose).
- Move per-agent ablation full table to replication package; keep only aggregate row in main text.
- Drop one of the RQ1 tables — *no*: doc-to-model and doc-to-code are both needed for the generalisation claim. Keep both.
- Shrink RQ2 grid figure to 2-column width with abbreviated metric labels.

---

## Open questions — resolved

1. **LiSSA inclusion — keep as placeholder.** We can obtain numbers; LiSSA is weaker than TransArc and serves only to represent the single-pass LLM-only family. RQ1 tables include a LiSSA column with `\todo{numbers}` for now.
2. **Doc-to-model vs doc-to-code emphasis — both.** Generalisation is defensible: AALinker improves doc-to-model link F1 by $+15.2$pp and doc-to-code file F1 by $+12.9$pp / decision F1 by $+22.7$pp (verified from CSVs). Reframe headline as *architectural trace recovery* covering both tasks. Intro mentions both numbers; doc-to-model is the cleanest single headline (the task AALinker directly produces).
3. **Skill score baselines — available.** Random and oracle F1 per project are in `evaluation/reports/EXTREME_BASELINES.md` and `METRIC_LIMITATIONS_ANALYSIS.md`. Random avg $\approx0.155$; oracle-subset avg $\approx0.987$. Skill score is computable; AALinker $\approx0.93$, TransArc $\approx0.78$.
4. **Pipeline figure — placeholder.** Insert `\todo{Figure 1: AALinker DAG (3 layers, 4 agents)}` in approach.tex; human or `pictures/` to provide.
5. **Prestudy — drop; merge into metric motivation.** The empirical bias evidence (5 inequality sources, JabRef 98.6\% concentration, cascade asymmetry) belongs in the metric chapter's motivation, where it does double duty. Removes a section from main.tex, frees ~0.5 page.

## What to send to the replication package (out of main text)

ICSE has no formal appendix in the 10-page limit; "appendix" means the replication package. Outsource:
- Full per-project metric dashboard (~30 metrics × 5 projects) — main text shows summary; package has the full panel from `evaluation/writing/eval.tex`.
- Per-agent ablation full table (RQ4) — main text shows aggregate Δ; package has per-project rows.
- Validator-ablation token-cost figures (RQ3 cost column) — main text mentions "roughly doubles call count"; package has exact numbers.
- Sensitivity analysis: max $|\Delta\fone|$ when a single raw gold entry is excluded (already in `evaluation/writing/eval.tex` Table~II) — package.
- Block-homogeneity table (96\%–100\%) — main text gives the range in one sentence; package has the full table.
- All prompts, agent code, raw \ac{LLM} responses, hyperparameters — package.
- Per-project random- and oracle-F1 baselines (5 projects × 2 baselines) — package; main text reports the average and uses it to compute skill score.

---

## Files in `writing/gen/`

| File | Status |
|---|---|
| `intro.tex` | drafted; metric glimpse + contributions filled; paras 1–3 + roadmap are TODO |
| `metric.tex` | motivation drafted with verified numbers; §Metric Suite is a stub |
| `eval.tex` | drafted; needs LiSSA decision + verified table inputs |
| `prestudy.tex`, `results.tex`, `discussion.tex`, `rw.tex`, `conclusion.tex` | empty |
| `refs/` | 11 verified PDFs + `MANIFEST.md`, `MANIFEST-AIML.md` knowledge base |
| `PAPER_PLAN.md` | this file |
