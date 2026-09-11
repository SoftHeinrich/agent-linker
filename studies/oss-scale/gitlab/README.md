# GitLab — a large real-world system whose architecture document is in the benchmark's style

Built 2026-09-04. The rustc / Linux / PostgreSQL probes (`../README.md` §6–§9) scaled the
*size* but not the *genre*: developer guides, subsystem references, in-tree design notes,
snake_case implementation names. The five benchmark texts are something else — a
project-authored **architecture overview** (a wiki "Services" page, an "Architecture" docs
page, a "Design" page) naming a few dozen components as proper nouns and describing how
they interact. This directory takes the same genre from a large system.

## 1. Source and dataset (`build.py`, pinned)

`gitlab-org/gitlab`, `doc/development/architecture.md` at commit `7eb01fc436a2`
(2026-08-10): a simplified and a complete **component diagram with connectors** (Mermaid),
a **component list** (name + one-line description + layer), one short section per
component, and narrative sections (web/SSH request cycles, system layout, troubleshooting).
Benchmark adaptation as in `benchmark/*/text_*/README.md`: headings and captions dropped,
links replaced by their text, code spans unwrapped, tables/diagrams/link-only bullets
removed, one sentence per line.

| | |
|---|---|
| sentences | **213** (`data/sentences.txt`; `sentence_meta.json` keeps the heading path) |
| components | **37**, the document's own `#### <name>` sections under "Component details" (`data/components.json`, with the table description, layer and process name) |
| runner spec | `data/datasets.json` → `ALINKER_EXTRA_DATASETS`; PCM stub `data/gitlab_arch.repository` |
| structural prior | 73 (sentence, component) pairs = sentences inside a component's own section (`data/gold_structural.csv`; a vote, not the gold) |

The linker sees the flat id/name list only, as on the benchmark.

## 2. Gold: the semantic label model, no retrieval needed (`annotate.py`, `label_model.py`)

Same recipe as `../rustc/semgold` (§8 there), simplified because 37 names fit in every
prompt. Two model families label every sentence ABOUT / REFERS per component with ±2
sentences of context and the heading path; the annotator sees the component descriptions,
layer, process name and own-section text — evidence the linker never gets. Votes:

| vote | what |
|---|---|
| terra | ABOUT in ≥ 2 of 3 gpt-5.6-terra sentence-view runs (cache salts `""`, `r2`, `r3`) |
| claude | ABOUT in the Claude Sonnet sentence-view run (local CLI) |
| compview | terra component-view run: one prompt per component over the whole document |
| structural | the sentence sits in the component's own section |

`gold` = terra ∧ claude; `gold_plus` = gold ∪ (one family ∧ (compview ∨ structural)).
Report (`out/label_model_report.json`):

| | |
|---|---|
| gold / gold_plus pairs | 118 / **134** on 110 sentences (**0.52** of the document), 31 components |
| κ terra–claude (sentence × component grid) | **0.80**; pair Jaccard 0.66 |
| κ component-view vs gold | 0.86 |
| terra pairs reproduced in all 3 runs | 0.88 of gold_plus |
| structural prior confirmed by gold_plus | 0.92 (67/73) |
| gold_plus pairs naming the component verbatim | **0.58** |
| tiers | gold 118, gold_plus_only 16, silver 42, refers 34 |

Cost: 43 prompts per sentence-view run (~128k prompt tokens, ~2 min terra; ~10 min via
the Claude CLI), 37 component-view prompts (~270k tokens).

## 3. Is it the benchmark's style? (`../tools/style_table.py`)

| dataset | sents | comps | gold | sent w/ gold | verbatim | caps | snake | shared-word names |
|---|---|---|---|---|---|---|---|---|
| mediastore | 37 | 14 | 31 | 0.73 | 0.55 | 1.00 | 0.00 | 0.50 |
| teastore | 43 | 11 | 27 | 0.53 | 0.59 | 1.00 | 0.00 | 0.55 |
| teammates | 198 | 8 | 57 | 0.23 | 0.86 | 1.00 | 0.00 | 0.00 |
| bigbluebutton | 87 | 12 | 62 | 0.55 | 0.55 | 0.92 | 0.08 | 0.50 |
| jabref | 13 | 6 | 18 | 0.77 | 1.00 | 0.00 | 0.00 | 0.00 |
| **gitlab** | **213** | **37** | **134** | **0.52** | **0.58** | **1.00** | **0.00** | **0.46** |
| rustc core | 1,762 | 79 | 1,327 | 0.68 | 0.11 | 0.00 | 0.99 | 1.00 |

Every column of the GitLab row lies inside the benchmark's range (gold density, share of
explicit links, capitalised proper-noun names, no snake_case, about half the names sharing
a word with another); the rustc row lies outside on four of five. What GitLab adds is
width: 37 components against 6–14, and a document as long as the largest benchmark text.

## 4. Results (`score.py`; three s110 runs, three one-call runs, one SWATTR run; gold_plus)

`run_s110.sh` / `run_onecall.sh` (paper backend: gpt-5.6-terra, flex, no reasoning;
`results/oss_scale_gitlab_*_20260904`), `../tools/run_swattr.sh` (ArDoCo CLI, no LLM).

| arm | links | TP | P | R | F1 | P lenient¹ | R explicit | R implicit |
|---|---|---|---|---|---|---|---|---|
| s110 as shipped | 514 | 93.0 | 0.181 | 0.694 | **0.287** | 0.231 | 0.987 | 0.286 |
| — full-name stage | 117 | 83.0 | 0.709 | 0.619 | 0.661 | 0.932 | 0.987 | 0.107 |
| — partial-name stage | 389 | 1.3 | 0.003 | 0.010 | 0.005 | 0.003 | 0.000 | 0.024 |
| — coreference stage | 8.7 | 8.7 | 1.000 | 0.065 | 0.121 | 1.000 | 0.000 | 0.155 |
| s110 minus partial-name | 126 | 91.7 | 0.729 | 0.684 | **0.706** | 0.936 | 0.987 | 0.262 |
| `s_linker110_onecall` (one call per 100 sentences, 3 calls) | 124 | 98.7 | 0.794 | 0.736 | **0.763** | 0.960 | 0.996 | 0.375 |
| SWATTR (ArDoCo SAD-SAM, deterministic) | 101 | 59.0 | 0.584 | 0.440 | **0.502** | 0.861 | 0.756 | 0.000 |

¹ REFERS pairs not counted as false positives. F1 spread over the three runs: s110
0.282–0.289, minus partial 0.698–0.713, one-call 0.745–0.796. Cost: s110 60 calls /
~155 s a run; one-call 3 calls / ~10 s.

Robustness to the gold (F1, minus-partial / one-call / SWATTR): strict gold 0.703 / 0.761 /
0.502; Claude family alone 0.715 / 0.771 / 0.489; three-way 0.700 / 0.753 / 0.493. Same
ordering under every gold.

### 4.1 What the numbers say

1. **The partial-name stage collapses on a compositional catalog even when every name is a
   proper noun.** 359–372 of its 383–394 links a run are a `GitLab *` component (8 of the
   37 names start with the system's name) proposed for a sentence that says "GitLab" (89 of
   213 do). s109's refusal — a word written only inside *another* component's name — cannot
   fire when the word is inside eight names *and* is the system. On rustc the trigger was
   domain vocabulary (`hir`, `mir`, `query`); here it is the vendor prefix. Both are
   predictable from the catalog before any call: a name word carried by ≥ 3 names, or equal
   to the system's name, has no discriminating power. The benchmark never poses this case:
   its system names prefix at most one component name (`BBB web`) and occur in 3–28% of
   the sentences (TeaStore 6/43, BigBlueButton 24/87); its most-shared name word
   (`Recommender`, 4 TeaStore components) is not a word the prose uses on its own.
2. **Without that stage, s110 transfers.** P 0.73 / R 0.68 / F1 0.71 on a 37-component
   catalog, lenient precision 0.94: 26 of its 34 full-name false positives a run are pairs
   both annotators marked REFERS (the sentence names the component while being about
   something else — "postgres_exporter … delivers data about PostgreSQL to Prometheus"),
   6 are silver (one family said ABOUT). The explicit stratum is essentially solved
   (R 0.99). The coreference stage is precise (8.7/8.7) and, as on rustc, barely fires.
3. **The one-call floor beats the workflow here.** RQ4's floor arm — the whole document
   and the catalog in one prompt, no stages — scores F1 0.76 at 3 calls, above the
   workflow's best view (0.71) and 47 pp above the workflow as shipped. On the benchmark
   the workflow is worth +8.5 pp over this arm (`rq4-onecall-floor`); on this document the
   sign flips, and the gap comes from the implicit stratum (R 0.375 vs 0.262) with higher
   precision. The floor's variance is also larger (F1 0.745–0.796 across runs).
4. **SWATTR does not collapse, it just stays behind.** Proper-noun names give ArDoCo's
   name-similarity something to match (F1 0.50, R explicit 0.76) where rustc's snake_case
   ids gave it nothing (F1 0.008); it finds no implicit link at all. Every LLM arm beats it
   by ≥ 20 pp F1, and the gap is entirely the implicit half of the gold.
5. **The implicit tail is a topic tail.** 40 of the 56 implicit gold pairs are never linked
   by any s110 run (minus partial). They are continuations inside a component's own section
   ("It also keeps default branch and hook information with the bare repository." →
   gitaly; "You can read more in the project's README." → the section's component) and
   sentences that say "GitLab" or "the GitLab application" meaning the Rails application
   (→ puma). Neither is a referring expression; §9.5's topic propagation is the mechanism
   that reaches them.

## 5. Files

```
build.py            fetch + adapt the document, component list, structural prior
annotate.py         two-family sentence-view and component-view annotation (OSS_DIR-aware)
label_model.py      votes → tiers → out/gold_semantic*.csv, out/label_model_report.json
score.py            links.csv (one or several runs) vs a gold: per stage, explicit/implicit, per component
run_s110.sh         3 × s_linker110 through approach/run_ablation.py (ALINKER_EXTRA_DATASETS)
run_onecall.sh      3 × s_linker110_onecall
run_annotate.sh     the four terra annotation runs
data/               sentences.txt, sentence_meta.json, components.json, gold_structural.csv,
                    datasets.json, gitlab_arch.repository, annotator_notes.txt, architecture.md
out/                annotations_*.json, semantic_labels.csv, gold_semantic*.csv, swattr/
```
