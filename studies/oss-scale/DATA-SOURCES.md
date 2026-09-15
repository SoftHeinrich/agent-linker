# Developer-annotated documentation→component links in the wild

Mined 2026-09-04 for the OSS-scale case. The question is **not** which research datasets
exist (they are small, and the biggest one is the five-project ArDoCo benchmark this project
already uses); it is **where developers themselves already record which component a piece of
prose belongs to**, as part of their normal work, at a scale no annotation budget reaches.

Every count below was measured here, with the command shown. Nothing is quoted from a paper.

## The four patterns worth mining

| pattern | what it is | who writes it | link it yields |
|---|---|---|---|
| **ownership registry** | a file listing components with their code paths, and often their doc paths | maintainers, kept current or the bot complains | doc file → component (+ its code) |
| **per-directory metadata** | a metadata file per directory naming the component/team that owns it | maintainers | any doc in that directory → component |
| **doc inside the unit** | a `README`/`docs/*.md` living in the component's own directory | the author of the code | that prose → that component |
| **explicit citation directive** | doc markup that pulls in or points at a source file (`.. kernel-doc::`, `automodule`, `@ingroup`) | doc author | doc section → file/module |

The first three are *not* citations, so they do not carry the bias that sank the anchor gold
in `README.md` §8: they say what a thing belongs to, not what a sentence happens to name. The
fourth does carry it, and stays a vote, never a gold.

## Measured inventory

### Linux `MAINTAINERS` — the best one
```bash
curl -s https://raw.githubusercontent.com/torvalds/linux/master/MAINTAINERS
```
* 3,395 subsystem entries; 3,282 have code paths.
* **1,715 entries (51%) list `Documentation/` paths beside their code paths** — 2,432 doc
  patterns in total.
* **607 documentation files are claimed by exactly one subsystem**, so the assignment is
  unambiguous without any tie-breaking.
* Example: `9P FILE SYSTEM` ← `Documentation/filesystems/9p.rst`, `fs/9p/`, `net/9p/`.

This is a ~2.4k-link developer-made doc→component mapping, on a system nobody tuned a linker
against. **Used**: `linux/` audits our label model against it — 0.784 of sentences put the
human owner in ABOUT, 10 of 12 documents vote for the human owner. See `README.md` §9.4.

### Chromium `DIR_METADATA`
```bash
curl -s "https://api.github.com/repos/chromium/chromium/git/trees/main?recursive=1"
```
* ≥525 `DIR_METADATA` files in the first 57k tree entries (listing truncated, so a floor);
  each assigns a directory to a maintained component, with `docs/*.md` sitting in those same
  directories. Per-directory metadata + doc-inside-the-unit at once.

### Mozilla `mots.yaml`
* 163 modules/submodules, **149 with a developer-written description**, 153 with path globs.
* Only 5 include documentation paths, so it is a weak doc→component source but the best
  *component model with authored prose* we found — the artefact our `profiles.py` has to
  synthesise from crate docs, for free.

### `CODEOWNERS` (any GitHub repo; measured on grafana/grafana)
* 1,276 path rules, 51 owning teams, 42 rules whose path is documentation.
* Generic: every repo that uses it publishes a path→team map, and team names are usually the
  project's own component vocabulary. Coverage varies (kubernetes/kubernetes has none;
  grafana has 88 KB of it), so it is a per-project check, not a guarantee.

### rust-lang `triagebot.toml` — for the very system in this study
* 8 autolabels whose `trigger_files` point into `compiler/` (`A-query-system` →
  `rustc_query_impl`, `A-LLVM` → `rustc_llvm` + `rustc_codegen_llvm`, …): a maintainer-written
  label→crate map.
* 52 `[mentions."compiler/…"]` entries pair a path with a one-line human description
  ("Some changes occurred to the CTFE machinery" → `rustc_const_eval/src/interpret`). Small,
  but it is developer prose against an exact path, and it is about *our* components.

### PostgreSQL in-tree `README`s — doc inside the unit
* **91 `README` files under `src/`**, each a long design document sitting in the directory it
  describes (`src/backend/access/nbtree/README` is 64 KB), plus 432 `doc/*.sgml` files.
* Same pattern in most large C projects: the prose is already filed under the component.

### Explicit citation directives (kept as votes, not gold)
* Linux `.. kernel-doc::`: 4,769 text files under `Documentation/`; a 30-file random sample
  had directives in 2 files, 3 directives — order of a few hundred doc-section→file links.
* Sphinx `automodule`/`autosummary`/`currentmodule`: 8 in a single numpy reference page; the
  Python ecosystem publishes these by the thousand.
* Doxygen `@defgroup`/`@ingroup`: real but uneven — 1 of 15 sampled OpenCV module headers
  carried a marker, while a single core header carries 28.

### Bug trackers and triage labels
Bugzilla `product::component`, JIRA `Component`, GitHub `A-*` / `area/*` labels: developers
assign a component to natural-language text at 10^5 scale, continuously. The text is a bug
report rather than architecture documentation, and the label is a triage queue rather than an
implementation unit — usable as auxiliary supervision (component vocabulary, alias discovery),
not as our gold.

### Kubernetes in-tree `OWNERS` and LLVM `Maintainers.md`
* kubernetes/kubernetes: **599 `OWNERS` files**; 5 of 12 sampled carry `sig/*` or `area/*`
  labels, so roughly 250 directories name their owning SIG inside the tree.
* llvm/llvm-project: `llvm/Maintainers.md` is 93 headings deep — a component list
  ("Transforms and analyses", "Generic backend and code generation", "Backends / Targets", …)
  with owners, written as prose by the project.

### Kubernetes `sigs.yaml`
24 SIGs, 234 subprojects with owner files. Component model only; the docs live in another repo
without SIG tags.

## What this is worth

* Two of these already changed the study: Linux `MAINTAINERS` gave the external audit §8.4 was
  missing, and it exposed the retrieval bottleneck (owner in BM25 top-12 of 3,282: 0.35 per
  sentence batch, 0.75 per document).
* For a second system, the pieces are all in the wild: Mozilla and Grafana hand you the
  component model, Chromium and PostgreSQL hand you documents already filed under a component,
  Linux hands you both.
* The bias to watch is not annotation noise but *what ownership means*: `MAINTAINERS` says who
  maintains a file, which is not the same as what the file is about. Our two disagreements in
  the audit were exactly that (`dlmfs.rst` maintained by OCFS2, about the DLM). Treat these
  sources as a strong prior with a known failure direction, not as truth.

Research datasets were checked once and set aside: nothing published covers architecture
documentation → component at this scale, and the in-the-wild sources above are larger, current,
and free.
