# Candidate: Firefox / mozilla-central

Measured 2026-09-03/04 against the GitHub mirror https://github.com/mozilla-firefox/firefox,
HEAD `edfb027f321fd2584c8fd2df528e665b7f5b8dac` (2026-09-03 16:28 UTC), blobless sparse
clone in `/tmp/oss-case/firefox/repo` (sparse: `docs`, `tools/moztreedocs`, `dom/docs`,
`netwerk/docs`, `gfx/docs`, `layout/docs`, `js/src/doc`, `ipc/docs`, `xpcom/docs`,
`toolkit/docs`, `browser/docs`). Tree: 475,445 files (`git ls-tree -r HEAD`), 55 top-level
directories (47 excluding 8 dot-dirs), 2,843 `moz.build` files (all 2,843 fetched via
raw.githubusercontent into `/tmp/oss-case/firefox/mozbuild/`). Scripts and intermediate
JSON are in `/tmp/oss-case/firefox/` (`docstats.py`, `mots_cov.py`, `parse_mots.py`,
`sentences.json`, `mozbuild_bug_components.json`, `bug_component_rules.json`,
`mots_modules.json`, `docfiles_bug_component.json`, `docfiles_mots.json`).

## D1 — Architecture prose

Source: firefox-source-docs (Sphinx/MyST, built from in-tree `docs/` and per-module
`docs/` dirs; `docs/config.yml` registers 37 `source_doc` trees). All git-tracked, MPL-2.0
(no per-file licence header in any of the 10 docs; `LICENSE` defers to
`toolkit/content/license.html`, which names the MPL 4 times).

Ten overview-style docs, markup and fenced code stripped, sentences split (`docstats.py`,
minimum 4 tokens per sentence):

| doc (path in tree; all render 200 at https://firefox-source-docs.mozilla.org/<x>.html) | sentences |
|---|---|
| docs/overview/gecko.md (overview/gecko) | 90 |
| gfx/docs/GraphicsOverview.md | 42 |
| gfx/docs/RenderingOverview.md | 131 |
| layout/docs/LayoutOverview.md | 179 |
| layout/docs/StyleSystemOverview.md | 46 |
| ipc/docs/processes.md | 313 |
| dom/docs/ipc/process_model.md | 89 |
| netwerk/docs/necko_lingo.md (networking/necko_lingo) | 51 |
| netwerk/docs/http/lifecycle.md | 33 |
| js/src/doc/index.md (js/index) | 55 |
| **total** | **1,029** |

Whole per-module corpus in the sparse checkout (same splitter): dom/docs 23 md / 1,653
sentences; netwerk/docs 25 / 1,015; gfx/docs 8 / 714; layout/docs 8 / 502; ipc/docs 5 / 940;
xpcom/docs 17 / 907; js/src/doc 29 / 2,117; docs/overview 90 -> 7,938 sentences in 115 files.
Tree-wide there are 2,923 `.md`/`.rst` files under a `docs/` or `doc/` dir, but 500 are
web-platform-tests docs and 481 are NSS docs (see S2), so the architecture-relevant corpus
is the ~8k sentences above plus toolkit/browser/devtools user docs. Most of it is
how-to/reference (IPDL grammar, Debugger API, cache2 internals); genuinely
architecture-overview prose is the ~600 sentences of gecko.md + the *Overview.md files.

Naming form (occurrence counts over the 1,029 sentences): proper names dominate for the
engine-level parts: `Gecko` 27, `XPCOM` 17, `IPDL` 23, `IPC` 18, `WebRender` 11, `Necko` 7,
`SpiderMonkey` 3, `APZ` 3, `Servo|Stylo` 2; `DOM` 22 (`the DOM` 4). Layout is named as a
lowercase noun (`layout` 28 vs `Layout` 5). Networking is never called `netwerk` (0) —
the directory name and the prose name differ. Process names are frequent
(`content process` 29, `GPU process` 14, `parent process` 11) and are *not* components in
any of the D2 lists. Class identifiers: `ns[A-Z]\w+` 74, `Foo::Bar` 67.

Ten example sentences naming modules (verbatim from `sentences.json`):
1. [gecko.md] "It is made up of HTML parsing and rendering, networking, JavaScript, IPC, DOM, OS widget abstractions and much much more."
2. [gecko.md] "Toolkit consists of components that can be shared across multiple applications built on top of Gecko."
3. [StyleSystemOverview.md] "In order to display the content, Gecko needs to compute the styles relevant to each DOM node."
4. [necko_lingo.md] "The code is jointly maintained by DOM team and Necko team"
5. [necko_lingo.md] "Necko is responsible for maintaining part of this code along with DOM."
6. [necko_lingo.md] "Glue code between Gecko and NSS." (about PSM)
7. [process_model.md] "The fork server must run before having initialized XPCOM or the IPC layer, and therefore uses a custom low-level IPC system called `MiniTransceiver` rather than IPDL to communicate."
8. [process_model.md] "For more details on how process types are added and managed by IPC, see the process creation documentation `Gecko Processes`."
9. [RenderingOverview.md] "...the GPU process talks back to the content process; in particular, when APZ scrolls out of bounds, it asks Content to enlarge..."
10. [LayoutOverview.md] "Talk: An Overview of Gecko Layout (Cameron McCormack :heycam, 2018-06-13)" — note this is a bibliography line the splitter kept; the docs contain many such non-architectural lines.

Skeptical note: sentences 4–6 show that the strongest "module" sentences are *ownership*
statements (who maintains what), not structural ones; and only 38/90 gecko.md sentences,
5/179 LayoutOverview sentences, 3/33 lifecycle sentences name any module at all
(`anymod` in `docstats.json`). Deep docs are about one module and rarely name it.

## D2 — Component model

Three project-authored options, all counted:

(a) **Bugzilla `BUG_COMPONENT` in moz.build** (`mozbuild_bug_components.json`): 647/2,843
moz.build files carry an assignment (646 literal tuples + js/src using variables
`component_engine` etc.). Distinct `Product::Component` pairs: 311 literal + 7 js variable
= 318, across 26 products (Core 132+7, Firefox 51, Toolkit 37, DevTools 22, Testing 19, …,
plus test fixtures `FooProduct::…`). Live Bugzilla (REST `product?names=Core`,
`bz_core.json`) lists Core 159, Firefox 60, Toolkit 43, DevTools 25, GeckoView 6 components;
so the tree references ~87% of live Core components. Names are Bugzilla labels
(`Core::DOM: Core & HTML`, `Core::Networking: HTTP`, `Core::Graphics: WebRender`).

(b) **Top-level directories**: 55 (47 non-dot). 44 have a moz.build; 40 of those set a
BUG_COMPONENT in their own moz.build. Names are path tokens (`netwerk`, `gfx`, `caps`,
`xpfe`) that the prose mostly does not use (see D1: `netwerk` 0 mentions).

(c) **mots module ownership** (`mots.yaml`, 120,837 bytes, `updated_at` 2026-08-30;
rendered at https://firefox-source-docs.mozilla.org/mots/index.html, HTTP 200; the export
file is generated, not tracked — `docs/mots/index.md` is absent from the tree):
116 top-level modules + 47 submodules = 163 (no deeper level). 111 top-level and 42
submodules carry `includes:` (343 patterns, 341 path globs, 2 URLs). 84 top-level modules
are `Core: *` engine modules (80 with includes); `Desktop Firefox` has 27 submodules,
`Toolkit` 5, `Remote Protocol` 4. 131 modules also list Bugzilla components in
`meta.components` (195 distinct; 126 of them coincide with a BUG_COMPONENT pair in
moz.build).

Name overlap with the prose (measured on the 1,029 sentences, case-insensitive, `Core: `
prefix and parenthetical stripped): 40/163 mots names appear verbatim (e.g. Necko, IPC,
XPCOM, Layout Engine, Graphics, Style System, Widget, Editor, GeckoView, Sandboxing (macOS));
only 13/284 Bugzilla component names appear verbatim (Graphics, Layout, Media, XPCOM,
Security, String, MathML, …), because Bugzilla names are compound (`DOM: Core & HTML`).

**Recommendation:** use mots top-level `Core:` modules as the component list, pruned to the
engine (drop tooling/testing/policy modules such as "Code Review Policy", "Tree Sheriffs",
"mots config"): roughly 60–70 entries, or the ~40 whose names the prose actually uses.
Bugzilla level (318) is too fine and its names are not the prose's names; top-level dirs
(47) have path names the prose avoids. mots is also the only option that is a *list
authored as an architecture* (with a one-paragraph description per module usable as
context).

## D3 — Code -> component map

Two project-authored maps, both measured over all 475,445 tree files:

- **moz.build `Files()` + BUG_COMPONENT** (Files()-aware resolver, last-matching-block-wins
  per moz.build semantics, nearest ancestor moz.build): 403,541 files (84.9%) resolve
  through a literal tuple; the remaining 71,904 unresolved are *all* under `js/`, whose
  `js/moz.build` has `with Files("**"): BUG_COMPONENT = component_engine` (variable, not
  matched by my regex). Effective coverage is therefore ~100%; 99.7% of files sit under a
  non-root moz.build's component (the root moz.build assigns only per-file blocks such as
  `LICENSE`, `mots.yaml`, `.cargo/**`). 306 distinct components are actually assigned to at
  least one file. Granularity is uneven: `dom/` splits into 85 components (largest is
  `Core::Graphics: CanvasWebGL` with 9,324 files — WebGL conformance tests), `toolkit/` 105,
  `browser/` 84, `layout/` 27, `netwerk/` 9, `gfx/` 6, `xpcom/` 6, `ipc/` 4. 75 moz.build
  files carry more than one component via multiple `Files()` blocks.
- **mots `includes:` globs** (`mots_cov.py`): 418,965/475,445 files (88.1%) match at least
  one module; 206,314/212,428 (97.1%) after excluding vendored trees (`third_party/`,
  `gfx/wr`, `gfx/skia`, `security/nss`, `nsprpub`, `testing/web-platform`, …). 44/55
  top-level dirs have covered files; uncovered are dot-dirs, `chrome`, `gradle`,
  `other-licenses`. Granularity = module (coarser than BUG_COMPONENT; `Core: Necko` has 18
  include globs, `Core: XPCOM` 19, `Core: Layout Engine` 8).

Caveat: neither map is 1:1 with the other (126 shared Bugzilla labels out of 318 / 195),
and both are *triage/ownership* maps, not architecture; e.g. `docshell/base/nsAboutRedirector.cpp`
resolves to `Core::General` via BUG_COMPONENT but to `Core: docshell` via mots.

## S1 — Self-supervised sentence-level gold

Signals per sentence in the 10 docs (counted on unstripped sentences, `docstats.json`):

| signal | sentences (of 1,029) | note |
|---|---|---|
| tree-rooted path (`dom/…`, `layout/…`) | 49 path tokens in raw text; only 11 distinct paths survive in stripped sentences (≈10 sentences, ~1%) | 8 of the 11 are one list in LayoutOverview.md ("layout/base/ contains…") |
| strict class/namespace identifier (`nsFoo`, `Foo::Bar`, `mozilla::`) | 83 (8.1%) | 33 in processes.md, 25 in LayoutOverview.md; 0 in both gfx overviews |
| backticked token | 289 (28%) | mostly IPDL keywords, types, prefs — not components |
| `{ref}`/`:ref:`/relative doc link | 34 (3.3%) | 18 are in gecko.md and point at module doc trees (`{ref}\`networking\``, `spidermonkey`, `xpcom`, `toolkit`, `xpidl`) — the one place where links are component-resolvable |

Path resolution: 20 doc-mentioned paths resolved (list in the transcript; all 20 resolve
via BUG_COMPONENT and all 20 via mots). Path->component is almost never wrong, but
sentence->component is: of the 20, the resolved component matches the sentence's topic in
14; in 6 it does not (five IDL paths in gecko.md are *examples* in a sentence about XPIDL,
so they resolve to `Core::Networking`/`Core::Security: CAPS` while the sentence is about
XPCOM; `toolkit/locales/.../processTypes.ftl` resolves to `Firefox Build System::General`).
Noise estimate for the path signal: ~30%. Class-name resolution: 10 identifiers from the
docs queried at https://searchfox.org/mozilla-central/search?q=<id> (JSON): 10/10 return a
definition file; 8/10 land in one directory (`nsHttpChannel` -> netwerk/protocol/http,
`ServoStyleSet` -> layout/style, `WebRenderBridgeParent` -> gfx/layers/wr, …); `PresShell`
and `MozPromise` are ambiguous (definitions listed under accessible/, dom/, xpcom/).
Co-change: not available from this clone (depth 1) — see T1.

Bottom line: usable sentence-level gold is ~80–100 sentences per 1,000 (identifier-bearing
sentences resolved through searchfox + BUG_COMPONENT, minus ~30% noise) — an order of
magnitude thinner than "every sentence has a code reference" would suggest, and biased
toward the two docs (processes.md, LayoutOverview.md) that are closest to reference material.

## S2 — Doc-level gold

- BUG_COMPONENT (Files()-aware): 2,894/2,923 in-tree doc files (`.md`/`.rst` under
  `docs/`|`doc/`) get a component; 98 distinct components. Largest owners:
  `Testing::web-platform-tests` 500, `Core::Security: PSM` 481 (vendored NSS docs),
  `Firefox Build System::General` 186, `DevTools::General` 179, `GeckoView::General` 156.
- mots: 2,726/2,923 covered; 460 files match more than one module (nested includes such as
  `Desktop Firefox` + `Toolkit` + `Core: Crash reporting`).
- Trivial in the sense the rubric warns about: a module's docs live in its own directory,
  so the label is "this doc is about itself". Cross-module content is thin: across all
  dom/docs (23 files, 1,653 sentences) only 31 sentences (1.9%) name a non-DOM module
  (IPC 16, XPCOM 10, SpiderMonkey/JS 4, Toolkit 2, docshell 1; Necko, Layout, Graphics,
  Widget, Style: 0 each). The doc-level label therefore yields ~1 correct link per
  doc-sentence for the owning module and almost nothing else.

## T1 — Downstream task

1. **Bug triage -> component.** Bugzilla history (REST API, no auth for public bugs) gives
   unlimited gold: (bug text, final component). But Mozilla already ships a trained
   classifier for exactly this — BugBug's `component` model
   (https://github.com/mozilla/bugbug, HTTP 200; also linked from `mots.yaml`'s
   "AI for Development" module) — so a doc-link-based triager would be measured against a
   strong supervised baseline that uses bug text, not architecture prose. Plausible only
   as a cold-start/explanation study.
2. **Doc staleness.** Measured signals are weak: in the 10 docs, 2/49 tree-rooted paths are
   dead (4.1%: `security/sandbox/mac/SandboxPolicy` — actually a template placeholder —
   and `devtools/shared/webconsole/network-helper.js`); 15 searchfox links are pinned to old
   revisions vs 27 unpinned; XBL (removed from the tree — 0 `xbl/` files) is mentioned in
   2 docs / 5 places among the checked-out docs. `layout/xul` still exists. The recent
   wiki->source-docs port (Bug 1889202, 2024-04) and the RST->MD conversion (Bug 2038819,
   2026-06) mean the overview docs were re-touched within two years, so staleness at the
   module level is low; the case would need per-sentence checking of class names against
   searchfox (feasible, ~83 sentences per 1,000).
3. **Doc-update recommendation via co-change.** Sampled via the GitHub commits API: the
   last 8 commits to LayoutOverview.md (2024-04..2026-05) and the single commits to
   RenderingOverview.md and processes.md are all doc-only (0 non-doc files in each of the
   4 commits inspected; Bug 1896210 Part 3 changed *code comments to point at the doc*,
   the opposite direction). Building co-change gold needs a full-history clone
   (`git clone https://github.com/mozilla/gecko-dev`, tens of GB) or `hg log --template`
   against https://hg.mozilla.org/mozilla-central, then joining commits that touch both a
   `docs/` file and a code dir; the sample suggests such commits are rare for overview docs.
4. **Newcomer navigation** ("which module does this paragraph talk about") is the natural
   fit; gold = mots module of the doc's directory + the 34 `{ref}` links; small.

## K — Killer-case pitch

Firefox is the one large OSS system whose architecture prose, module list (mots) and
code->module map are all project-authored and in the same tree, so (sentence, module)
links would let a 30M-line codebase's 8k sentences of design docs be navigated by module
without anyone hand-labelling — but the honest measurement says the prose is
single-module per file (1.9% cross-module in dom/docs) and only ~8% of sentences carry a
resolvable identifier, so the case reads more as "large document, few interesting links".

## C — Cost / feasibility

- Scale: 1,029 sentences x ~40 mots engine modules = ~41k candidate pairs for the 10-doc
  set; 7,938 x 40 = ~320k for the eight-dir corpus. Both far above the benchmark's
  200 x 14 = 2.8k, and the linker's per-sentence LLM calls scale with sentences, so the
  full corpus is a multi-hour run per repetition (N>=3 needed per the variance memory).
- Licence: MPL-2.0 for docs and code (no separate doc licence; no CC-BY notice found in
  `docs/`); vendoring the 10 markdown files (~330 KB raw) into the replication package with
  the MPL notice and commit hash is fine.
- Artifacts: `/tmp/oss-case/firefox/` (207 MB incl. sparse repo; mots.yaml, 2,843
  moz.build, sentence files `sent_*.txt`, Bugzilla JSON).

## Verdict

READY-WITH-WORK. Inputs exist and are measured (1,029–7,938 sentences; mots gives a 60–70
module list whose names the prose uses; code->module map ~100%). What is missing is gold:
identifier-bearing sentences are 8% and cross-module sentences 2%, so a study needs either a
hand-labelled slice (~300 sentences) or a searchfox-resolved silver set with ~30% noise
accepted; downstream tasks are weak (BugBug already owns triage, doc co-change is rare).
