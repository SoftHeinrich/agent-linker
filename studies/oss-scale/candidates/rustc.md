# Candidate: rustc (rust-lang/rust + rustc-dev-guide)

Measured 2026-09-03/04. Raw artifacts under `/tmp/oss-case/rustc/` (not in repo):
`rustc-dev-guide/` (full clone, 8,631 commits, HEAD 40155f22 2026-09-03),
`rust/` (blobless clone, full 333,578-commit history, no blobs except the ones named below, 428 MB),
`team/` (rust-lang/team, shallow), `triagebot.toml` (from rust HEAD a69a6326 2026-09-03),
`work/` (scripts `split.py`, `signals.py`, `analyse.py`, `sentences.json`, `cochange.txt`).

## D1 — Architecture prose

- Source: https://github.com/rust-lang/rustc-dev-guide, `src/` (246 .md files). Also vendored
  in rust-lang/rust as a josh subtree at `src/doc/rustc-dev-guide/` (a tree, not a submodule,
  since PR #135127 "rustc-dev-guide subtree update", 2025-01-05; 64 such sync PRs to date).
- License: MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE` in repo root). Git-tracked, yes.
- Corpus definition: the five architecture parts of `src/SUMMARY.md` — "High-level compiler
  architecture", "Source code representation", "Supporting infrastructure", "Analysis",
  "MIR to binaries" — excluding `appendix/`. Building/testing/contributing parts excluded.
- Size (measured by `work/split.py`: fenced code, tables, headings, HTML comments and
  link-reference definitions stripped; regex sentence split; sentences < 4 words dropped):
  **156 chapters, 6,522 sentences, 133,952 words.**
  Per part: Analysis 2,872 · MIR-to-binaries 1,208 · High-level architecture 1,134 ·
  Source code representation 795 · Supporting infrastructure 513.
  Largest chapters: diagnostics.md 240, lldb-visualizers.md 159, macro-expansion.md 156,
  overview.md 149, incremental-compilation-in-detail.md 144.
- A "core architecture" subset (32 chapters: overview, compiler-src, query, memory,
  serialization, parallel-rustc, the-parser, macro-expansion, name-resolution, attributes,
  hir, hir/lowering, thir, mir/{index,construction,passes,dataflow,optimizations},
  rustc-driver/intro, diagnostics, ty, type-inference, traits/resolution, solve/trait-solving,
  hir-typeck/summary, coherence, borrow-check, const-eval, backend/{monomorph,lowering-mir,
  codegen,libs-and-metadata}) = **1,762 sentences**.
- How components are named (sentence counts, `work/analyse.py`):
  - verbatim crate names in backticks: 265 sentences (4.1%) name one of the 79 real crates;
    49 distinct crates. Top: rustc_middle 40, rustc_codegen_ssa 24, rustc_interface 18,
    rustc_driver 15, rustc_hir 15, rustc_type_ir 14, rustc_codegen_llvm 13.
  - concept names dominate: `MIR` 228, `LLVM` 268, `HIR` 154, `diagnostic*` 135,
    `type check*` 80, `codegen` 78, `trait solv*` 64, `parser` 59, `borrow check*` 41,
    `name resolution` 28, `THIR` 25, `macro expansion` 24, `query system` 22,
    `HIR/AST lowering` 20, `monomorphiz*` 18, `the driver` 0.
  - So the doc's component vocabulary is concept nouns (IR names, pass names), with the
    crate name given once per chapter as an anchor. The linker will have to bridge
    "the borrow checker" -> `rustc_borrowck` (easy) and "MIR"/"ty" -> `rustc_middle` (hard).
- 10 example sentences naming a crate (one per chapter, from `analyse.py`):
  1. [overview.md] Command line argument parsing occurs in the `rustc_driver`.
  2. [compiler-src.md] ... roughly it is something like this: `rustc` (the binary) calls `rustc_driver::main`.
  3. [query.md] Almost all extern providers wind up going through the `rustc_metadata` crate, which loads the information from the crate metadata.
  4. [queries/incremental-compilation.md] The query DAG code is stored in `compiler/rustc_middle/src/dep_graph`.
  5. [queries/incremental-compilation-in-detail.md] `rustc_middle::query::modifiers`: The query system allows for applying modifiers to queries.
  6. [memory.md] The `rustc_middle::ty::tls` module is used to access these thread-locals, although you should rarely need to touch it.
  7. [serialization.md] To work around this, the `RefDecodable` trait is defined in `rustc_middle`.
  8. [parallel-rustc.md] The underlying thread-safe data-structures used in the parallel compiler can be found in the `rustc_data_structures::sync` module.
  9. [rustdoc-internals.md] This is also the step where cross-crate inlining is performed, which requires converting `rustc_middle` data structures into the cleaned `AST`.
  10. [the-parser.md] The parser is defined in `rustc_parse`, along with a high-level interface to the lexer and some validation routines that run after macro expansion.

## D2 — Component model

(a) **Crates under `compiler/`** — `git ls-tree HEAD compiler/` in rust-lang/rust: **79 dirs**
    (78 `rustc_*` + `rustc`). Cargo `name` equals the dir name (checked 6 Cargo.toml blobs:
    rustc_middle, rustc_borrowck, rustc_next_trait_solver, rustc_codegen_cranelift,
    rustc_public_bridge; the exception is `compiler/rustc` -> `rustc-main`). Project-authored,
    stable strings, the same strings the doc uses when it does name crates. Recommended level.
(b) **triagebot.toml** (`/tmp/oss-case/rustc/triagebot.toml`, 1,862 lines, parsed with tomllib):
    - `[autolabel]`: 55 entries (49 with `trigger_files`); label prefixes A-19, O-17, T-11,
      F-2, S-2, I-1, plus needs-triage, PG-exploit-mitigations, WG-trait-system-refactor.
      Only **8** autolabel entries have trigger_files under `compiler/` (A-LLVM, A-attributes,
      A-query-system, A-translation, F-autodiff, O-apple, PG-exploit-mitigations,
      WG-trait-system-refactor), covering 19 distinct `compiler/` paths. Weak as a model.
    - `[mentions]`: 118 entries, **52 under `compiler/`**, mostly sub-crate paths
      (e.g. `compiler/rustc_mir_build/src/thir/pattern`, `compiler/rustc_trait_selection/src/solve/`).
    - `[assign.owners]`: 103 keys, **30 under `/compiler`** (crate level for 20, sub-crate for
      `rustc_middle/src/{mir,ty,query,traits,dep_graph,ich.rs}`, `rustc_parse/src/lexer`, ...).
    - `[assign.adhoc_groups]`: 20 reviewer groups (arena, ast_lowering, borrowck, codegen,
      debuginfo, diagnostics, incremental, lexer, mir, mir-opt, parser, query-system, ...).
      These names are concept-level and match the doc vocabulary better than crate names,
      but there are only ~12 compiler-relevant ones.
(c) **rust-lang/team** (`team/teams/*.toml`): 166 teams; **20 with `subteam-of = "compiler"`**
    (compiler itself, 75 members; types; wg-mir-opt, wg-parallel-rustc, wg-diagnostics,
    wg-const-eval, wg-llvm, wg-macros, wg-linker, wg-compiler-performance, miri,
    rust-analyzer, rustc-dev-guide, ...). Too coarse and only partly architectural.
- GitHub labels: 231 `A-*` labels among the first 500 labels fetched (more pages exist).
  Compiler-area labels exist for every major doc concept (A-borrow-checker, A-resolve,
  A-trait-system, A-parser, A-macros, A-mir-opt, A-codegen, A-incr-comp, A-query-system,
  A-inference, A-HIR, A-lints, ...), but they are not a code-path map.
- Verdict on level: **crates (79)**, optionally reduced to the **34 crates with >= 5 anchored
  sentences** (see S1); 29 crates (rustc_arena, rustc_hashes, rustc_fs_util, rustc_graphviz,
  rustc_codegen_gcc/cranelift, rustc_public*, rustc_traits, ...) are never anchored in the
  corpus and would only act as distractors.

## D3 — Code -> component map

- `compiler/<crate>/` is the map: 2,905 files (2,173 `.rs`) under `compiler/` map 100% to
  the 79 crates by path prefix. Granularity: crate. Project-authored by construction.
- Finer: triagebot `[assign.owners]` 30 compiler paths + `[mentions]` 52 compiler paths give
  sub-crate ownership for the biggest crates (rustc_middle split into mir/ty/query/traits/
  dep_graph). Coverage of those finer maps is partial (rustc_middle, rustc_parse,
  rustc_trait_selection, rustc_const_eval, rustc_codegen_llvm mainly).

## S1 — Self-supervised sentence-level gold

Measured with `work/signals.py` / `work/analyse.py` over the 156-chapter corpus.

| signal | sentences | notes |
|---|---|---|
| verbatim real crate name (`rustc_x`) in the sentence | 265 (4.1%) | 49 crates, 61 chapters |
| `compiler/rustc_x/src/...` path in the sentence | 24 | subset of the above |
| resolvable link to `doc.rust-lang.org/nightly/nightly-rustc/<crate>/...` | 592-602 (~9.2%) | 706 link occurrences in the raw md; 81/156 chapters (52%); 43 distinct crates; only 17 links are inline, the rest are `[text][ref]` with reference definitions at chapter end (so the anchor is invisible in the sentence text — exactly what a masked-anchor task wants) |
| union (crate name or link) | **706 of 6,280 (11.2%)** | 50/79 crates anchored, 34 with >= 5 sentences |
| co-change (same commit edits guide .md + compiler/) | 69 commits since 2025-01-05 | see T1; not sentence-level without further diffing |

- Multi-crate sentences: 68 of ~600 linked sentences point to > 1 crate.
- Link target distribution is skewed: **rustc_middle = 175/706 (24.8%)**, then rustc_hir 43,
  rustc_borrowck 39, rustc_span 38, rustc_expand 32, rustc_mir_dataflow 27,
  rustc_codegen_ssa 24, rustc_hir_typeck 23, rustc_parse 22, rustc_errors 21.
- Noise spot-check (20 random linked sentences, `analyse.py` seed 3): 17/20 the linked crate is
  the crate the sentence is about; **3/20 the link points to the crate that *defines the
  type*, not the component being described** (coercion sentence in hir-typeck/coercions.md ->
  `TypeckResults` in rustc_middle; "Session and ParseSess have buffer_lint methods" in
  diagnostics.md -> rustc_session; opaque-type inference sentence -> `OpaqueTy` in rustc_hir).
  Estimated noise ~15%, systematically biased toward rustc_middle/rustc_hir/rustc_span
  (the "data-definition" crates). A gold built from links measures "which crate owns the
  item this sentence cites", which is a defensible but not identical notion of "component
  this sentence describes". Mitigation: exclude rustc_span/rustc_data_structures/rustc_index
  style utility crates, or evaluate at chapter-majority level (S2).
- Stale anchors: **0** links to non-existent crates (the guide runs `ci/linkcheck.sh`), and the
  only `rustc_*` tokens in prose that are not crates are macro/attribute names
  (`rustc_queries!` 4, `#[rustc_private]` 3, `rustc_diagnostic_item` 3, ...). Removed crate
  names survive only outside prose: `rustc_typeck` in a table in parallel-rustc.md (2 rows)
  and a stack trace in compiler-debugging.md; `rustc_mir` in a version-pinned URL. So the
  live doc is *not* a source of staleness examples; history is (T1).

## S2 — Doc-level gold

- 82 chapters carry >= 1 nightly-rustc link; **49 chapters have >= 3 linked sentences and a
  single crate with >= 50% of the links** (majority crate = chapter label). Examples:
  name-resolution.md -> rustc_resolve 13/14; borrow-check/region-inference.md ->
  rustc_borrowck 16/17; hir/lowering.md -> rustc_ast_lowering 6/7; mir/optimizations.md ->
  rustc_mir_transform 6/7; hir-typeck/coercions.md -> rustc_hir_typeck 18/23;
  diagnostics.md -> rustc_errors 12/21; const-eval/interpret.md -> rustc_const_eval 6/12.
- Chapter titles imply crates for roughly 25 more chapters without links (the-parser ->
  rustc_parse, macro-expansion -> rustc_expand, thir/mir/construction -> rustc_mir_build,
  backend/codegen -> rustc_codegen_ssa/llvm, ...), but that mapping is ours, not the project's.
- triagebot `[assign.owners]` has `/src/doc/rustc-dev-guide` as one entry (the whole guide ->
  rustc-dev-guide team) — no per-chapter ownership.

## T1 — Downstream task

(i) **Masked-anchor recovery.** Hide the `[text][ref]` link target (the sentence text is
    unchanged because 97% of links are reference-style), ask the linker for the crate, score
    against the link's crate. Gold size: ~600 sentences × 43 crates, 81 chapters. Zero human
    labelling. Caveat from S1: ~15% of gold is "defining crate" rather than "described
    component"; report both strict and chapter-majority-relaxed scores.

(ii) **Doc staleness.**
   - Project-declared anchors: `<!-- date-check -->` markers — **47 in 31 files** (48 raw
     occurrences incl. the one in contributing-to-guide.md that documents the mechanism).
     By year: 2021: 2, 2022: 9, 2023: 3, 2024: 10, 2025: 7, 2026: 16. A monthly cron
     (`.github/workflows/date-check.yml`, `cron: '00 12 01 * *'`, runner `ci/date-check/`)
     opens an issue listing markers older than a threshold. Honest reading: the markers date a
     *claim*, not a component; they say "re-check this sentence", not "against which crate".
     Links would supply the crate. 47 is a small gold, and 9 of them are already 2022-era.
   - Historical staleness from crate renames (guide history, `git log -S`):
     `rustc_typeck` -> `rustc_hir_analysis` merged 2022-09-27 (rust PR #102306); guide links
     rewritten 2022-10-22 (bba24898) and 2022-12-03 (48d78e34): **25-67 day stale window**.
     `rustc_mir` split merged 2021-09-08 (rust PR #80522); last guide link removed 2021-11-16
     (34 commits over the years touched `nightly-rustc/rustc_mir/` links): **69 days**.
     156 guide commits mention link fixes. Protocol: for each rename PR, take the guide at
     the rename date, run the linker, check that sentences it links to the old crate are the
     ones the later fix commits touched. Gold is a handful of events (2-4 major renames), so
     this is a case study, not a benchmark.
   - Co-change (rust-lang/rust, non-merge commits since 2024-01-01, `work/cochange.txt`):
     18,065 commits touch `compiler/`; 1,569 touch guide .md (most are josh-imported guide
     history); **69 single commits touch both** (39 in 2026, 30 in 2025, 0 before the josh
     switch) = 0.38% of compiler commits, ~3.5/month. Examples: "rename OutlivesPredicate to
     OutlivesClause" (2 guide files, 60 compiler files), "Many predicate-to-clause renamings"
     (5 guide files, 101 compiler files), "Rename *CombinedModuleLateLintPass" (lintstore.md +
     rustc_lint/src/lib.rs). Usable as a doc-update-recommendation gold (code change in crate
     C -> which guide chapter/sentence changed), but thin: 69 events, mostly renames.

(iii) **PR / issue routing.** GitHub search API counts (repo:rust-lang/rust):
     T-compiler PRs 32,324 (single label, useless as target); area labels on **PRs** are
     sparse: A-LLVM 1,943, A-query-system 1,361, A-diagnostics 353, A-mir-opt 161,
     A-lints 144, A-codegen 130, A-macros 74, A-parser 69, A-resolve 50, A-MIR 43,
     A-trait-system 42, A-incr-comp 32, A-type-system 30, A-HIR 16, A-borrow-checker 11,
     A-inference 5. On **issues** they are dense: A-lints 1,783, A-type-system 1,481,
     A-trait-system 1,388, A-codegen 1,300, A-macros 1,104, A-incr-comp 864, A-parser 687,
     A-resolve 672, A-borrow-checker 510, A-MIR 492, A-inference 457 (T-compiler 22,894).
     So routing gold exists for *issues -> area*, not PRs, and area labels are concepts, not
     crates; mapping them onto crates is ours. triagebot autolabel already routes by path for
     the 8 compiler areas it covers, so "the linker enables routing" is a weak pitch here.

## K — Killer-case pitch

The rustc-dev-guide is the onboarding document for the most-contributed-to compiler in the
world (32k T-compiler PRs), 6.5k sentences long, maintained by a dedicated team with its own
staleness CI — and it already encodes ~600 sentence-to-crate anchors as reference-style
links, so a linker can be scored on it at scale without a human labelling a single sentence.

## C — Cost / feasibility

- Pair count: 6,522 sentences × 79 crates = 515k pairs (full); 6,522 × 34 anchored crates =
  222k; core subset 1,762 × 79 = 139k. At benchmark scale (30-200 sentences × 6-14
  components) this is 30-100× the largest benchmark project in sentences and 6-13× in
  components; the linker's per-sentence LLM budget will dominate, so plan on the core subset
  or a per-part run.
- Licence: MIT OR Apache-2.0 — vendoring the markdown (or the extracted one-sentence-per-line
  file) into the replication package is fine with attribution. Component list is the
  `compiler/` directory listing (MIT/Apache-2.0 as well). triagebot.toml likewise.
- Reproducibility: pin guide commit 40155f22 and rust commit a69a6326; the blobless rust
  clone is 428 MB, the guide 18 MB; only 6 Cargo.toml blobs and triagebot.toml were fetched.
- Risks: (1) the rustc_middle skew — a quarter of gold links point to one crate and the doc
  never calls it by a concept name, so precision on rustc_middle will look bad for reasons
  unrelated to the linker; (2) the corpus mixes architecture prose with how-to chapters
  (updating-llvm.md, debugging.md, lldb-visualizers.md) that name no component — use the
  core subset or filter chapters with zero anchors; (3) the gold is "cited-item crate", not
  "described component" (~15% disagreement on the spot check).

## Verdict

READY-WITH-WORK: prose (6.5k sentences, MIT/Apache), component list (79 crates) and code map (100% by path) all exist and are project-authored; ~600 masked-link sentence anchors across 43 crates are the strongest self-supervision seen, but skewed to rustc_middle and ~15% off-target.
Work needed: choose the crate subset (34 anchored crates), decide strict vs chapter-majority scoring, and build the rename-based staleness cases from history (rustc_typeck 2022, rustc_mir 2021) since the live doc has zero dead anchors.
Downstream story is masked-anchor recovery plus doc-update recommendation from 69 co-change commits; PR routing is not a credible pitch here (area labels are sparse on PRs and triagebot already routes by path).
