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

## 8. Semantic gold: from anchors to a label model (2026-09-04)

The §6 gold was rejected for the right reason: a hyperlink or a verbatim crate name is
a *syntactic* event, it labels 19% of the sentences, and it is exactly the event the
full-name stage detects, so scoring against it is circular for the explicit part and
blind for the implicit part. The literature (`LITERATURE.md`, designs 1–3) says the
same thing in older words: hyperlinks are distant supervision for *training*, never a
test set (Mintz 2009; Rath ICSE'18: ~40% of true links carry no anchor; Bird FSE'09:
anchored links are a biased sample). What replaces it is a **label model over semantic
sources**, each of which may abstain, with the anchors demoted to one vote.

### 8.1 Sources (`rustc/semgold/`, all reproducible from the cache)

* **Component self-descriptions, project-authored.** Each crate's `//!` crate doc,
  README, module-level `//!` docs, public module names, most-referenced public items and
  source-file names (`profiles.py`): 41/79 crates carry a crate doc, 54 carry module
  docs, 18 small utility crates are described only by their names, items and files.
  This is what the *annotator* sees and the *linker* never sees (the linker gets the
  flat id/name list; §1).
* **Symbol grounding.** 14,591 public identifiers → defining crate(s) (`symbols.py`);
  700 sentences name an identifier, 371 resolve to ≤3 crates. A deterministic code fact,
  used as annotator evidence and as a vote ("`Diag` is defined in rustc_errors").
* **Grounded LLM annotation, sentence view** (`annotate.py`). Two model families —
  the linker's own (gpt-5.6-terra, flex, no reasoning) and another (Claude Sonnet via
  the local CLI) — label every one of the 1,762 sentences, five at a time with ±2
  sentences of context, the chapter, the resolved symbols, BM25-retrieved candidate
  profiles (top-8 contains the anchor crate for 269/341 anchored sentences; with
  symbols 308/341) and the full 79-name list. Labels per (sentence, crate): **ABOUT**
  (maintainers of the crate would have to fix the sentence if the crate changed) or
  **REFERS** (an item of the crate is named, the sentence is about something else).
  Cost: 364 calls, 1.65M prompt tokens, ~10 min per terra run; ~35 min for the CLI run.
  Three terra runs (fresh samples under a cache salt) give a per-pair consistency score.
* **Grounded LLM annotation, crate view** (`annotate_crateview.py`). The opposite
  direction, component first: given one crate's profile and one whole chapter, list the
  sentences ABOUT it (299 (chapter, crate) prompts). Different failure mode, third vote.
* **Co-change** (`cochange.py`): commits that edit a core chapter and 1–3 crates in the
  rust monorepo. The guide's history is mirrored back to 2018 (1,144 commits touch the
  32 chapters) but code co-changes exist only since the 2025 subtree merge: 15 focused
  commits, 11 sentence-level pairs. Kept as an independent check, not as a source of
  scale on this system.
* **Anchors** (§6 gold, 393 pairs): now one vote.

### 8.2 Label model and agreement (`label_model.py`, `out/label_model_report.json`)

| tier | definition | pairs | sentences |
|---|---|---|---|
| gold | ABOUT by both families | 1,236 | 1,146 |
| gold_plus | gold ∪ (ABOUT by one family ∧ symbol/anchor/co-change vote) | **1,327** | **1,200** (68% of the document), 55 crates |
| silver | ABOUT by one family, unsupported | 681 | — |
| refers | REFERS by either, not ABOUT | 427 | — |

Agreement between the two families: Cohen's κ **0.76** on the full sentence × crate
grid (139k cells), pair-level Jaccard 0.62, exact ABOUT-set agreement per sentence 0.68,
"has any ABOUT" agreement 0.87. Terra's three runs reproduce 1,509 of its 2,022 union
pairs in all three (75%); 1,214 of the 1,327 gold_plus pairs (91%) have full
consistency. Both exceed Ahmed et al.'s MSR'25 suitability gate (model–model α > 0.5)
and land between Ahmed's SE tasks and Alor et al.'s κ 0.94 on a hand-made doc→code set.

Is the jury just re-detecting anchors? ABOUT rate by sentence surface (gold_plus):
anchored 0.965, unanchored with a code span 0.665, unanchored plain prose 0.578. The
plain-prose rate is the number the anchors could never give: more than half of the
sentences that name nothing are still ABOUT a crate, and two families agree on which.

Anchors re-judged semantically (`validate.py`): of the 312 item-link pairs (the
"defining-crate code facts" §6 distrusted) the jury calls 218 gold, 35 gold_plus, 30
REFERS, 29 nothing — so the item links were mostly right, and the 10% that are REFERS
are exactly the `Span`/`Symbol`/`DefId` mentions. Of 11 co-change pairs, 6 are in
gold_plus, 1 REFERS, 4 unlabelled.

**Crate view as third vote** (`annotate_crateview.py`, terra, 299 prompts, 0.68M
tokens): 1,671 pairs on 1,246 sentences; it recovers 78% of gold_plus at precision 0.62
against gold_plus and 0.78 against "either family said ABOUT" (κ 0.69 vs gold_plus on
the grid). 331 of its pairs have no sentence-view vote at all — the component-first
reading finds sentences the sentence-first reading skipped, and those go to the human
sheet, not into gold. **Three-way gold** (both families ∧ crate view): 980 pairs
(`gold_semantic_3way.csv`).

### 8.3 What the semantic gold says about s110 (`rescore.py`, same run as §6.1)

| view | links | TP | P | R | F1 | P lenient¹ | R explicit² | R implicit² |
|---|---|---|---|---|---|---|---|---|
| all stages | 3,176 | 454 | 0.143 | 0.342 | 0.202 | 0.149 | 0.973 | 0.264 |
| full-name stage | 618 | 263 | 0.426 | 0.198 | 0.270 | 0.506 | 0.973 | 0.102 |
| partial-name stage | 2,536 | 184 | 0.073 | 0.139 | 0.095 | 0.073 | 0.000 | 0.156 |
| minus partial-name | 640 | 270 | 0.422 | 0.203 | 0.275 | 0.498 | 0.973 | 0.108 |

¹ REFERS pairs not counted as false positives. ² gold_plus split by whether the sentence
contains the crate name verbatim: **147 explicit vs 1,180 implicit** pairs.

Robustness to the choice of gold (full-name stage; `reports/semgold_s110_r1_*.txt`):

| gold | pairs | P | R | F1 | R implicit |
|---|---|---|---|---|---|
| anchors (§6) | 393 | 0.320 | 0.504 | 0.392 | 0.156 (link-only) |
| gold_plus (two families + votes) | 1,327 | 0.426 | 0.198 | 0.270 | 0.102 |
| gold (both families) | 1,236 | 0.383 | 0.192 | 0.256 | 0.103 |
| three-way (+ crate view) | 980 | 0.359 | 0.227 | 0.278 | 0.123 |
| Claude family alone | 1,479 | 0.416 | 0.174 | 0.245 | 0.096 |

Every semantic gold tells the same story; only the anchor gold told a different one.

Three things change relative to §6.1. (i) The recall ceiling moves: the anchor gold
made s110 look like a 0.50-recall linker; against what the document is actually about
it is a 0.20-recall linker, because 89% of the gold is implicit and the full-name
stage finds 10% of that. This is the paper's RQ2 claim ("the standard metric hides
the tail") measured on 1,180 pairs instead of argued. (ii) Full-name precision drops
from 0.75 (on anchored sentences) to 0.43: the 209 full-name links no annotator
labelled are, on a 25-sample read, half the `rustc` entry absorbing "rustc" the system
(the §7 rule is now a measured need) and half a verbatim word standing for a sibling
("HIR" → rustc_hir when the sentence is about rustc_hir_typeck; "procedural macros" →
rustc_proc_macro when the sentence is about the lexer). (iii) The partial-name stage
is refuted semantically as well as syntactically: 184/2,536 correct, 107 more silver.

### 8.4 Validation status and how this scales

* **Human check.** `out/human_check_sheet.csv`: 255 pairs stratified by tier ×
  evidence pattern (llm-only, anchor, symbol, multi-source; 40 bm25-top-1 negatives),
  the no-source stratum over-sampled as design 1 demands. Verdict column is blank; this
  is the sheet a human fills. Until a human fills it, a **declared stand-in** read all 255 pairs with the source
  tree, symbol index and chapter open (a Claude-family agent in this session — same family
  as one annotator, so it is a calibration check, not the paper's number;
  `out/model_check_sheet.csv`, `reports/semgold_model_check_agreement.txt`):

  | tier | n | judged ABOUT | ABOUT or REFERS |
  |---|---|---|---|
  | gold (both families) | 110 | 0.86 | 0.94 |
  | gold_plus_only (one family + vote) | 35 | 0.60 | 0.94 |
  | silver (one family, no vote) | 40 | 0.42 | 0.65 |
  | refers | 30 | 0.13 (0.50 REFERS) | — |
  | bm25-top-1 negatives | 40 | 0.03 | — |

  The tiers are ordered as designed, the negatives are clean, and the symbol-vote
  promotion is the weak rule (6/15 ABOUT): a symbol hit plus one family is not enough.
  Consequence: **cite the strict `gold` tier (1,236 pairs) for precision claims and
  `gold_plus` only for recall**; the gold-tier disagreements are almost all the
  defining-crate vs implementing-crate boundary (SVH hashing attributed to
  rustc_incremental, style guidelines naming an API), i.e. the REFERS line, not noise.
* **Circularity.** The linker is terra; the gold requires the Claude family to agree,
  the annotator sees crate self-descriptions and symbols the linker does not, and
  `gold_semantic_a2only.csv` (Claude alone, 1,479 pairs) exists for a same-family-free
  re-score. Rule "never hardcode benchmark words" is untouched: profiles are read from
  the project tree at run time, and the annotator is dataset construction, not the linker.
* **Scale.** Everything is linear in sentences: ~940 prompt tokens and ~1.7 s of wall
  time per sentence per annotator run at 6 workers. The full guide (6,522 sentences) is
  ~40 min per family; Linux `Documentation/mm` (2,541) or Firefox overview (1,029) need
  only a profile extractor (MAINTAINERS + Kconfig help; mots.yaml descriptions) and a
  symbol index (ctags). The runner hook, scorer and label model are system-agnostic.
* **Still open.** Human verdicts on the 255 sheet (κ human–jury is the number the
  paper needs); the crate grouping into ~12 maintainer-confirmed components (design 7),
  so the sibling confusions can be scored at both granularities; co-change is not a
  scalable source on rustc and should be tried on PostgreSQL/Linux instead.
  *Partly closed on 2026-09-04*: §9.4 and §9.6 audit the recipe against two
  developer-written assignments (Linux `MAINTAINERS`, PostgreSQL in-tree READMEs) instead
  of paid annotators — 0.784 and 0.869 of sentences put the human owner in ABOUT, 20 of 22
  documents vote for it.

## 9. Where the recall goes: surface strata, coreference, and a human audit (2026-09-04)

§8 split the gold into *explicit* (crate named verbatim) and *implicit*. That split was too
coarse: it counts "the borrow checker" as implicit even though a matcher working on the
component name alone can fire on it. `semgold/surface.py` re-splits by what surface of the
name the sentence carries, using one generic morphological rule — a crate-id token (minus
the vendor prefix every component shares) and a sentence token that agree on their first
four characters. No word lists.

### 9.1 Three strata, and where s110 lives (`reports/semgold_surface_strata.txt`)

| stratum | gold_plus pairs | share |
|---|---|---|
| verbatim (`rustc_borrowck`) | 147 | 0.111 |
| name-echo ("the borrow checker", "MIR", "name resolution") | 504 | 0.380 |
| no-surface (only meaning connects them) | 676 | 0.509 |

The rule was audited rather than assumed: the tokens that trigger an echo are `lint` (51
pairs), `mir` (48), `trait` (33), `expand` (33), `hir` (29), `infer`, `query`, `borrowck`,
`resolve` — domain words, not stopwords — and only 73 of the 504 echo pairs fire on a token
that appears in more than 5% of sentences.

| view | links | P | R | R verbatim | R name-echo | R no-surface |
|---|---|---|---|---|---|---|
| all stages | 3,176 | 0.143 | 0.342 | 0.973 | 0.530 | **0.065** |
| full_name | 618 | 0.426 | 0.198 | 0.973 | 0.167 | 0.053 |
| partial_name | 2,536 | 0.073 | 0.139 | 0.000 | 0.363 | 0.001 |
| coreference | 22 | 0.318 | 0.005 | 0.000 | 0.000 | 0.010 |

Three things this changes:

* The "full-name" stage is not a literal matcher — **392 of its 618 links (63%) are not a
  verbatim occurrence** of the id. It is already doing name-word paraphrase, which is why it
  reaches part of the echo stratum.
* The partial-name stage, refuted three times on precision, is **the only stage covering the
  name-echo stratum** (R 0.363 there against full-name's 0.167). Dropping it still wins on F1,
  but the honest statement is that it buys echo recall at P 0.073, and nothing replaces it.
* Half the gold has no name surface at all and the whole pipeline recovers 6.5% of it. That
  is the tail, stated in one number.

### 9.2 Coreference: the ceiling is the design, not the prompt (`semgold/coref_headroom.py`)

s110's coreference stage resolves a referring expression to a component that an **earlier
sentence names**; `_prompt_coref` is handed the "NAMED BEFORE THIS CASE" list, so no
antecedent means no link. Against the semantic gold:

| antecedent must be … | share of no-surface gold reachable, window 3 | anywhere earlier in chapter |
|---|---|---|
| a literal crate id | 0.109 | 0.451 |
| any name echo | **0.479** | **0.846** |

With literal antecedents, **60.5% of implicit gold has no preceding mention in the chapter
at all** (514 pairs whose crate is never named in the chapter; 211 gold pairs involve crates
never named anywhere in the document, and only 33 of 79 crates are ever named). The stage
produced 22 links and 7 true ones — 5.6% of even the narrow literal-antecedent ceiling.

The cheap fix is not a better coref prompt, it is a wider antecedent: name echoes are already
what the linker's own stages fire on, and they raise the reachable share of the hard stratum
from 0.11 to 0.48. A sticky-topic baseline (propagate every naming sentence forward K
sentences, no LLM at all) is a useful yardstick — K=0: P 0.628 R 0.111; K=3: P 0.370 R 0.206;
K=10: P 0.261 R 0.310; whole chapter: P 0.104 R 0.613. The pipeline's all-stages point
(P 0.143, R 0.342) sits *below* the K=10 line.

### 9.3 Developer-annotated links in the wild (`DATA-SOURCES.md`)

Where do developers already record which component a piece of prose belongs to, as part of
normal work? Four patterns generalise: an **ownership registry** (components with their code
*and doc* paths), **per-directory metadata**, a **doc filed inside the unit it describes**, and
**explicit citation directives**. The first three do not say what a sentence names, they say
what a document belongs to — so they escape the anchor bias §8 measured. Measured, not quoted:

| source | measured | what it gives |
|---|---|---|
| Linux `MAINTAINERS` | 3,395 subsystems, **1,715 with `Documentation/` paths**, 2,432 doc patterns, **607 docs owned by exactly one** | doc file → component + its code paths |
| Chromium `DIR_METADATA` | ≥525 files (listing truncated) | directory → component, with `docs/*.md` in the same directories |
| Mozilla `mots.yaml` | 163 modules, 149 with authored descriptions | component model with prose, for free |
| `CODEOWNERS` (grafana) | 1,276 path rules, 51 teams | path → team, in any repo that keeps one |
| rust `triagebot.toml` | 8 compiler autolabels, 52 path descriptions | label → crate, for this very system |
| PostgreSQL `src/**/README` | 91 in-tree design documents | prose already filed under its module |
| `.. kernel-doc::` / Sphinx `automodule` / Doxygen `@ingroup` | few hundred / thousands / uneven | explicit citations — votes, never gold |
| Bugzilla / JIRA / `area/*` labels | 10^5 reports | text → component, but bug text |

### 9.4 The audit: our recipe against Linux `MAINTAINERS` (`linux/`)

12 documentation files that exactly one subsystem claims, 383 sentences, candidates by BM25
over 3,282 subsystem profiles, the §8 ABOUT/REFERS prompt, one family (terra). The annotator
is never shown the owner.

* **owner among the ABOUT labels: 298/380 sentences = 0.784**; 0.879 of the sentences that got
  any ABOUT; 0.93 ABOUT labels per sentence, so not a yes-to-everything labeller.
* **10 of 12 documents** have the human owner as their most-voted subsystem.
* Both misses are ownership-vs-content disagreements (`dlmfs.rst` is maintained by OCFS2 but is
  about the DLM; `afbc.rst` is maintained by a DRM driver but describes a buffer format) — the
  human source is *who maintains the file*, which is not exactly aboutness.

This is the external check §8.4 was missing, on a system with no relation to rustc, against a
mapping written by maintainers for their own use.

It also produced the most actionable number of the round: BM25 finds the human owner in the
top 12 of 3,282 for **0.346** of 5-sentence batches but **0.750** of whole documents. Same
index, same components, only the query changes. Candidate retrieval belongs at document level;
judging stays at sentence level.

### 9.5 Coreference, fixed: topic propagation with a judge (`semgold/topic_probe.py`)

§9.2 says the coreference stage's ceiling is its antecedent gate, not its prompt. This tests
the replacement without touching the linker: take every link the full-name stage made, propose
the same component for the next K sentences of the chapter, and put each proposal in front of
one judging call ("is the target still making a claim about this component, given the sentence
that established it?"). No new retrieval, no new knowledge — only a wider antecedent.

Three runs of the judge (cache salts `""`, `r2`, `r3`), k=3, gold_plus:

| view | links | TP | P | R | F1 |
|---|---|---|---|---|---|
| s110 coreference stage | 22 | 7 | 0.318 | 0.005 | 0.010 |
| topic propagation, judged | 214 / 225 / 219 | 99 / 97 / 102 | 0.463 / 0.431 / 0.466 | 0.075 | 0.128 |
| full-name stage alone | 618 | 263 | 0.426 | 0.198 | 0.270 |
| **full-name + topic propagation** | **832 / 843 / 837** | **362 / 360 / 365** | **0.435** | **0.273** | **0.335** |
| all s110 stages | 3,176 | 454 | 0.143 | 0.342 | 0.202 |
| all s110 stages + topic propagation | 3,390 | 553 | 0.163 | 0.417 | 0.234 |

* 14x the true links of the stage it replaces, at *higher* precision than the full-name stage
  that seeded it (0.46 vs 0.43), and the three runs agree to +/-3 links and +/-3 true links --
  well outside the +/-55-link coreference noise this project measured on the small benchmark.
* +6.5 pp F1 on the precision-side arm (full-name only: 0.270 -> 0.335) for 107 calls.
* k=5 proposes 1,568 instead of 1,066 and the judge approves 275: recall +0.007, precision
  -0.04, F1 unchanged (0.335). k=3 is the knee.
* All of it lands in the no-surface stratum, which is where §9.1 says the tail is.

The verdict on coreference is therefore not "it does not work at scale" but **"the antecedent
gate, not the resolver, is what fails"** -- and the gate is cheap to widen.

### 9.6 Second audit, second pattern: PostgreSQL in-tree READMEs (`postgres/`)

The Linux audit tests an *ownership registry*. PostgreSQL tests the other common in-the-wild
pattern — a design document filed **inside** the directory it describes (42 such READMEs in C
source directories). Components are the 141 source directories with at least two C files; same
prompt, same scorer, 10 documents, 278 sentences.

| | Linux `MAINTAINERS` | PostgreSQL READMEs |
|---|---|---|
| components in the index | 3,282 | 141 |
| owner among ABOUT | 0.784 | **0.869** |
| documents voting for the owner | 10/12 | **10/10** |
| BM25 owner in top-12, per sentence batch | 0.346 | 0.831 |
| BM25 owner in top-12, per document | 0.750 | 1.000 |

The recipe holds on a C codebase and on design notes rather than a guide, and the gap between
the two audits is index size, not labelling — the same failure mode as the rustc partial-name
stage. Residual misses in both are content-vs-placement disagreements, which is the known limit
of any human source that records *where code lives* rather than *what prose is about*.

### 9.7 The deterministic baseline: SWATTR on the same dataset (`rustc/reports/swattr_rustc_core.txt`)

ArDoCo's own SAD-SAM stage (SWATTR) run through `ardoco-cli` on exactly this dataset —
`sentences.txt` + the generated PCM repository, no API key, no LLM.

```
java -jar ardoco-cli-*-jar-with-dependencies.jar -t sad-sam -n rustc_core \
  -d studies/oss-scale/rustc/data/core/sentences.txt \
  -m studies/oss-scale/rustc/data/core/rustc_core.repository -o <out>
```

Wall clock 36 min: preprocessing 17:03, text extraction 14:15, recommendation 13 s,
connection 4:08. (The generated repository first had to gain a repository `id`; ArDoCo's PCM
parser rejects the file without one. Fixed in `tools/make_dataset.py`.) ArDoCo re-splits the
text with CoreNLP into 1,915 sentences, so its sentence ids were mapped back to dataset lines
through the same splitter before scoring (`semgold/from_ardoco.py --corenlp-json`, which
reproduces the committed `reports/swattr_rustc_core_links.csv` exactly).

| gold | links | TP | P | R |
|---|---|---|---|---|
| gold_plus | 266 | 2 | 0.008 | 0.002 |
| gold (strict) | 266 | 2 | 0.008 | 0.002 |
| three-way | 266 | 2 | 0.008 | 0.002 |
| anchor (the old syntactic gold) | 266 | 0 | 0.000 | 0.000 |

**All 266 links point at one component — `rustc`, the umbrella entry.** Not a single link to
any of the other 78 crates. The failure is not sentence selection: **198 of its 266 sentences
(74%) are gold for some crate** — it picks documentation sentences that really are about the
architecture and then attributes every one of them to the only component name that surfaces as
an ordinary noun phrase. The two true positives are the two sentences the annotators did mark
as being about `rustc` itself.

This is the baseline number the study needed, and it is worth stating plainly: the deterministic
SAD-SAM stage does not degrade at OSS scale, it collapses. Snake-case implementation names
(`rustc_mir_transform`) do not appear as noun phrases in prose, so name-similarity has nothing
to match, while s110 — however weak in absolute terms (§9.1) — still reaches 0.973 of the
verbatim stratum and 0.53 of the name-echo stratum on the same input.
