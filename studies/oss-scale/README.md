# oss-scale — can the s110 linker be shown useful on a large open-source system?

Status: **investigation, 2026-09-03**. No paper table is written from here. Candidate
probes live in `candidates/`; the rubric they follow is `RUBRIC.md`.

## 1. What the linker actually needs (measured, not assumed)

The reported arm (`approach/.../s_linker110.py`) consumes exactly two artifacts:

| Input | Form | Source in the benchmark |
|---|---|---|
| architecture document | plain text, one sentence per line (`DocumentLoader.load_sentences`) | `benchmark/<p>/text_*/<p>.txt` |
| component model | flat list of `(id, name)` (`pcm_parser.ArchitectureComponent`; only `id`/`name` are read by any linker) | PCM `.repository` |

It never reads code. Output is a set of `(sentence, component)` links; gold is the
same shape (`goldstandard_sad_*-sam_*.csv`). So a "large-scale case" needs a prose
document and a *named component list* — a code→component map is needed only to
build gold, not to run the linker.

Benchmark scale it was built on: 12–197 sentences, 6–14 components a project.

## 2. Cost scaling (one s110 run, terra, `results/consolidation_e2e_terra_r1_20260825`)

| project | sentences | components | calls | prompt tok | completion tok | wall s |
|---|---|---|---|---|---|---|
| jabref | 12 | 6 | 6 | 6.5k | 1.0k | 10 |
| mediastore | 36 | 14 | 9 | 12.0k | 1.5k | 14 |
| teastore | 42 | 11 | 9 | 12.5k | 2.2k | 16 |
| bigbluebutton | 86 | 12 | 18 | 30.5k | 7.6k | 44 |
| teammates | 197 | 8 | 31 | 54.1k | 8.2k | 78 |

≈ 275 prompt tokens and 0.4 s a sentence, linear (batches: extraction 50 sentences,
judges 25 candidates, coreference 10 targets over a ±5 window; the catalog is pasted
per call, so 60–80 component names add a constant, not a factor). A 3,000-sentence
document is therefore ~0.8M prompt tokens and ~20 min a run; N=3 is affordable.
The one thing that does not scale for free is the coreference stage's `NAMED BEFORE
THIS CASE` shortlist — with 60 components it will be longer, and s110's own
measurement (1.8–4.5 of 6–14 a case) has to be re-taken.

## 3. Why this is worth doing (the gap)

Every SAD→SAM system we compare against — TransArC, SWATTR, ArDoCode, LiSSA
(ICSE'25), ArTEMiS/ExArch (ICSA'25, arXiv:2511.02434) — is measured on the same
five ARDoCo projects; none reports a document larger than ~200 sentences or a model
larger than 14 components. Our own threat statement (`paper/sections/discussion.tex`,
"External validity") concedes this. A case where the document is 10–50× longer and the
model 5× wider is the missing evidence, and the only prior "in the wild" traceability
datasets at that scale (issue–commit, release-note–PR: arXiv:2511.18187) are not
document→architecture.

## 4. Three ways to show usefulness without hand-labelling

The goal statement asks for two things in order: (a) a self-supervised gold so the
links can be *scored*, else (b) a downstream task the links *support*. Both exist,
and they are not exclusive.

### P1 — Masked-anchor self-supervision (scores the links)

Many sentences in developer-facing architecture prose carry an **explicit anchor**:
a path (`kernel/sched/core.c`), a crate/module name (`rustc_borrowck`), a hyperlink to a
component's own reference page, or a `:c:func:` reference. A project-authored
code→component map (MAINTAINERS `F:`, moz.build `BUG_COMPONENT`, `compiler/<crate>`,
OWNERS labels) resolves each anchor to a component. That gives a `(sentence,
component)` gold pair for free.

Two protocols on the same gold:

* **P1-open**: run the linker on the document as written. Measures whether the
  linker's *name relation* copes with anchors that are not the catalog's spelling
  (`drivers/gpu/drm/i915/` vs `INTEL DRM DRIVERS`). This is the realistic input.
* **P1-masked**: replace every anchor with a neutral placeholder and run again. What
  survives is the linker's *implicit* linking — the coreference / partial-name tail
  that RQ2 argues the standard metric hides. Gold precision is exact (the anchor
  was there); recall is bounded by how much the prose still says.

Noise to declare: an anchor names a component the sentence is *about* only most of
the time (a sentence can cite a file as a counter-example). The probe reports must
spot-check 10 and give the rate.

### P2 — Co-change as gold (scores the links, from history)

In a monorepo where documentation and code change in the same commit (PostgreSQL
by convention; Linux, CPython, Firefox often), a commit that edits sentence *s* of the
architecture document and code under component *C* — and nothing else — is a
project-authored assertion that *s* and *C* are related. Mining `git log --name-only`
plus a diff of the document gives `(s, C)` pairs at scale; restricting to commits
touching exactly one component keeps it clean. This gold is *independent of the
sentence's wording*, so unlike P1 it also covers sentences with no anchor at all.

### P3 — Downstream tasks the links support (usefulness without gold links)

* **T-a Doc-update recommendation.** Given a code change in component *C*, rank the
  document's sentences to review. Links give the ranking; historical co-change
  commits (P2's raw material, used as a *task* gold rather than a link gold) score
  it as MRR / recall@k against what the maintainers actually edited. Baselines:
  BM25 over the component's name, embedding retrieval, one-call LLM. This is the
  "docs are always stale" pain every large project has; the metric is what a
  maintainer would feel.
* **T-b Drift detection (MME/UME at scale).** Sentences whose extracted mention
  resolves to no catalog entry are Missing-Model-Element candidates; catalog entries
  no sentence links to are Undocumented-Model-Element candidates. Git decides: a
  mention whose component *was* in the tree and was removed is a confirmed stale
  sentence; a component with zero links whose directory is < 1 year old is a
  confirmed doc gap. Reported as counts with a manual check of the flagged set (tens,
  not thousands — checkable).
* **T-c Routing** (issue/KEP/PR → component) is available on Kubernetes and Firefox
  with label gold, but it is text classification with strong existing baselines
  (Mozilla's BugBug) and it is not the paper's task; listed as a fallback only.

## 5. Candidate comparison (six probes, 2026-09-03/04; details in `candidates/`)

All six came back READY-WITH-WORK; none is ready as-is and none is hopeless. The
column that separates them is **S1**, the sentence-level gold that costs no labelling.

| candidate | D1 arch prose (sentences) | D2 components (project-authored) | S1 sentence gold, free | P2 co-change | T-b staleness at HEAD | licence |
|---|---|---|---|---|---|---|
| **rustc** (rustc-dev-guide) | 6,522 in 156 chapters; core subset 1,762 | 79 crates under `compiler/` (names = dir names) | **~600 (9.2%) reference-style links to `nightly-rustc/<crate>` — crate invisible in the text**; +265 verbatim; ~15% "defining crate" noise | 69 commits since 2025 (guide is a josh subtree) | 0 dead anchors (linkcheck CI); staleness only from history (`rustc_typeck` rename: 25–67 d lag; `rustc_mir` split: 69 d) | MIT/Apache |
| Linux (Documentation/) | 169k total; mm/ 2,541, scheduler 1,351, vfs.rst 495 | MAINTAINERS 3,331 entries; usable "core" level 42, MM family 20 | ~1% paths (10/17 resolve uniquely); 4.6% `func()` refs need a symbol index | 28% of doc commits co-change code (n=29) | **~200 dangling code paths tree-wide**, 3 verified | GPL-2.0 |
| Firefox (firefox-source-docs) | 1,029 overview; 7,938 per-module | mots.yaml 116 modules (Core ~60–70); BUG_COMPONENT 318 pairs, ~100% file coverage | 8.1% identifier sentences, 30% off-topic; cross-module sentences 2% | sparse, needs full history | 2/49 dead paths | MPL-2.0 |
| CPython (InternalDocs + devguide) | 1,551 | none matches doc granularity: 8 top dirs (41% of paths fall in `Python/`), CODEOWNERS people-keyed 20–38% | **19% code-ref sentences (295)**, 3/10 are "also update X" not "is about" | 35 commits / 27 months, 19 sentence-pinnable | none at HEAD | PSF |
| PostgreSQL (manual internals + backend READMEs) | 641 core (1,462 manual) + 4,058 README | 26 backend dirs; doc names ≠ dir names (planner↔optimizer, WAL↔access/transam, `tcop` never named) | 1.6% manual / 6.3% README, ~30% off-topic | 1,665 sgml+backend commits but **9** clean prose+single-dir | 1/5 spot-checked real | PostgreSQL |
| Kubernetes (website + KEPs) | 537 | 13 components whose names the prose uses (SIGs 24, never named) | 6.5% (~35) | none: docs in another repo | none (dockershim/PSP sentences are all migration text) | CC-BY/Apache |

**Decision: rustc first.** It is the only candidate where a *sentence-level* gold of
hundreds of pairs exists without a human, and it is the cleanest form of P1: 97% of
the anchors are reference-style markdown links, so the crate is genuinely absent from
the sentence the linker reads — no masking step, no argument about what masking
destroys. It also has the widest project-authored model (79 named crates vs 6–14 in the
benchmark) and a permissive licence. Its two known weaknesses are declared up front:
a quarter of link-gold points at `rustc_middle` (the crate that *defines* MIR/ty types,
which the prose calls "MIR" and "ty", never by crate), and ~15% of anchors name the
defining crate rather than the described component — so precision is a lower bound
and the per-crate table matters more than the macro number.

Second arm if a *drift* story is wanted: **Linux `Documentation/mm/`** against the
20 MEMORY MANAGEMENT entries, where the ~200 dangling paths are a ready-made T-b gold.
CPython is the best-*anchored* prose (19%) but has no defensible component list.

## 6. Pilot: s110 on the rustc core chapters

Dataset: `rustc/build_dataset.py` (guide 40155f22, crates from rust a69a6326) →
`rustc/data/core/`: 32 core-architecture chapters, **1,762 sentences, 79 crates,
393 gold pairs on 341 anchored sentences** (220 link-only = invisible anchors, 173
verbatim), 38 crates carry gold. Run through `approach/run_ablation.py` unchanged via
`ALINKER_EXTRA_DATASETS` (an additive hook added 2026-09-04) on gpt-5.6-terra/flex,
the paper's backend. Scored with `rustc/score.py`, which reports what a partial gold
licenses: recall by anchor kind, precision on anchored sentences, and a sample of the
unanchored predictions for a manual check.

### 6.1 Result (one run, `results/oss_scale_rustc_core_r1_20260904_0011`, 451 calls, 20 min)

| view | links | TP | strict P | R (all gold) | R verbatim | R link-only | P on anchored sentences |
|---|---|---|---|---|---|---|---|
| s110 as shipped | 3,176 | 219 | 0.069 | 0.557 | 0.948 | 0.250 | 0.283 |
| minus the partial-name stage (offline filter) | 640 | 199 | 0.311 | 0.506 | 0.948 | 0.159 | 0.748 |
| full-name stage alone | 618 | 198 | 0.320 | 0.504 | 0.948 | 0.155 | 0.756 |

Per stage: full-name 618 links / 198 TP; coreference 22 / 1; **partial-name 2,536 / 20**.
Scores in `rustc/reports/`. Three findings, each with its number:

1. **The partial-name relation does not survive a 79-name catalog built from the
   domain's own vocabulary.** The words that triggered the 2,536 partial links are
   `hir` 215, `type` 205, `mir` 192, `macros` 168, `ast` 146, `lint` 146, `crate` 113,
   `trait` 98, `macro` 93, `query` 72 — 23 of the words in the crate names are shared by
   two or more crates, and the rest (`crate`, `query`, `macro`) are the document's
   commonest nouns. s109's refusal ("a word written only inside another component's
   name") fired 14,236 times and was not enough; a simulated refusal of every shared
   word still leaves 395 links with 7 TP (`rustc_crate_store` 115 from "crate",
   `rustc_proc_macro` 99, `rustc_query_impl` 72). This is the paper's own external-validity
   caveat made concrete: the relation's benchmark precision rests on component names
   being rare capitalized tokens, and snake_case crate names are neither. On this
   catalog the stage must be off, or gated by a document-frequency rule that the
   benchmark never needed.
2. **The full-name stage plus the document-discovered alias table is the working
   linker at this scale.** The knowledge stage found `THIR -> rustc_mir_build`,
   `AST -> rustc_ast`, `proc-macros -> rustc_proc_macro`, `MIR -> rustc_middle` and
   others from the document alone; with them, 483 sentences link to 34 crates at
   precision 0.756 on anchored sentences. A manual read of 30 unanchored full-name links
   (`reports/`, seed 7): ~20 clearly right, ~6 defensible, 4 wrong — 3 of the 4 are the
   catalog entry `rustc` (the binary crate) absorbing every mention of "the compiler".
   Chapter level: the linker's top crate is the chapter's majority-gold crate in 6/17
   chapters, top-3 in 14/17; the misses are `rustc` and `rustc_middle` outranking the
   specific crate.
3. **The free link gold is mostly a code fact, not a documentation fact.** Of the 220
   link-only pairs, 189 are links on a *cited item* (`TyCtxt`, `Diag`, `DefId`) and only
   31 on a prose expression; recall on the item kind is 0.17 and that number measures
   whether the model knows which crate defines `TyCtxt`, which a doc-and-model linker is
   not asked to know. The verbatim gold (173 pairs, recall 0.95) is clean but explicit.
   So this dataset scores *explicit* linking and *sibling discrimination* well and the
   implicit tail poorly; a hand-labelled slice (~300 sentences) is still needed for the
   RQ2-style tail claim.

## 7. Verdict and what to do next

**Data readiness is solved for one system**, and the recipe transfers: a component
list is a directory listing, a document is a developer guide, and gold comes from the
project's own hyperlinks (rustc), ownership files (Linux, Firefox) or history. The
runner takes any such pair via `ALINKER_EXTRA_DATASETS`.

**Usefulness is shown only after one design change.** s110 as shipped scores 6.9%
precision here and would embarrass the paper; without the partial-name stage it is a
0.75-precision, 0.50-recall linker over 1,762 sentences and 79 crates, at 20 min and
~0.4M tokens a run, with the alias table doing the non-trivial work. The honest
paper sentence is: the name relation's whole-name and coreference forms transfer,
its one-word form does not, and the failure is predictable from the catalog alone
(shared words / document-frequency of name words) before any call is made.

Next steps, in order:
1. A real N=3 run of an s110 variant with the partial stage off (or gated on the
   catalog's shared-word set), so the row above is a measurement and not a filter.
2. Drop the `rustc` entry or treat "the compiler" as the system, not a component —
   decide by rule (a component named as the system is the system), not by hand.
3. Hand-label ~300 sentences from 6 chapters (two annotators) for the implicit tail.
4. Downstream, doc-update recommendation: 69 co-change commits (guide prose + one
   crate) already mined in `/tmp/oss-case/rustc/work/cochange.txt`; protocol in §4 T-a,
   baseline BM25 on crate name + aliases.
5. Second system for the drift story: Linux `Documentation/mm/` vs the 20 MEMORY
   MANAGEMENT entries, ~200 dangling paths as T-b gold (`candidates/linux.md`).
