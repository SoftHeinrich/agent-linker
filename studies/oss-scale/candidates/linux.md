# Candidate: Linux kernel

Snapshot: torvalds/linux `a500db7819c5` (committed 2026-09-03), sparse clone of
`Documentation/` + `MAINTAINERS` at `/tmp/oss-case/linux/repo` (`git clone --depth 1
--filter=blob:none --sparse`; tree objects are present, so `git ls-tree -r HEAD`
gives the full 96,026-file list without code — `/tmp/oss-case/linux/tree_files.txt`).
Scripts: `/tmp/oss-case/linux/{rstsent,parse_maint,fmatch,level,alldocs,cochange}.py`.
All counts below come from those scripts unless marked otherwise.

## D1 — Architecture prose

**Where it lives.** `Documentation/` has 11,439 files, 4,042 `.rst`. Excluding
`translations/` and `devicetree/`, 3,459 rst files split into 169,253 sentences
(splitter: strip RST directives/literal blocks/tables/underlines, join paragraphs,
split on `[.!?]` + capital, keep >=4-word sentences with a lowercase letter).
Subtrees that are architecture prose vs. not:

| subtree | rst | raw lines | sentences | verdict |
|---|---|---|---|---|
| `core-api/` | 76 | 17,817 | 4,553 | mixed: workqueue, cpu_hotplug, genericirq are prose; kernel-api, mm-api are kernel-doc stubs |
| `mm/` | 47 | 8,635 | 2,541 | yes: physical_memory, page_migration, page_reclaim, multigen_lru, ... |
| `scheduler/` | 19 | 4,893 | 1,351 | yes: sched-design-CFS, sched-eevdf, sched-domains, sched-deadline |
| `block/` | 17 | 3,228 | 1,023 | yes but thin: blk-mq, biovecs, data-integrity |
| `locking/` | 18 | 4,554 | 1,192 | yes: locktypes, lockdep-design, mutex-design |
| `filesystems/` | 148 | 51,768 | 14,437 | vfs.rst, locking.rst, mount_api.rst are prose; the rest is per-FS |
| `networking/` | 263 | 70,673 | 17,167 | scaling.rst, kapi, msg_zerocopy are prose; ~85% per-driver/protocol |
| `driver-api/` | 360 | 67,423 | 15,680 | overview pages prose; most is kernel-doc |
| `admin-guide/` | 399 | 87,923 | 23,067 | no: user-facing, excluded |

**Candidate docs measured** (sentences / raw lines): `filesystems/vfs.rst` 495/1,509;
`mm/physical_memory.rst` 231/632; `networking/scaling.rst` 222/598;
`core-api/workqueue.rst` 214/795; `locking/locktypes.rst` 134/555;
`mm/page_migration.rst` 92/207; `scheduler/sched-design-CFS.rst` 69/255;
`block/blk-mq.rst` 50/153. Total 1,507 sentences, dumps in
`/tmp/oss-case/linux/sent_*.txt`. Per-file URLs:
`https://github.com/torvalds/linux/blob/master/Documentation/<path>`.

**How components are named.** Lowercase nouns and acronyms, not proper names.
Counts over the 1,499 candidate-doc sentences (`grep -ci`): "VFS" 77 / "the VFS" 59,
"workqueue" 53, "spinlock" 35, "the kernel" 29, "the filesystem" 28, "CFS" 24, "RSS" 21,
"RCU" 15, "XPS" 13, "mutex" 11, "page cache" 10, "the scheduler" 9, "the block layer" 9,
"the page allocator" 7, "buddy allocator" 6, "cgroup" 6, "blk-mq" 6, "slab" 2,
"the memory management" 1. Sentences are long and technical (vfs.rst: "To look up an
inode requires that the VFS calls the lookup() method of the parent directory inode.").
Git-tracked: yes. Licence: `COPYING` = GPL-2.0 WITH Linux-syscall-note; 2,439/4,042 rst
files carry an SPDX `GPL-2.0` header, including vfs.rst and sched-design-CFS.rst.

## D2 — Component model

**Artifact:** `MAINTAINERS` (30,194 lines,
`https://raw.githubusercontent.com/torvalds/linux/master/MAINTAINERS`), parsed into
`/tmp/oss-case/linux/maint.json`: **3,331 entries** (3,311 with `F:`), status
Maintained 2,231 / Supported 807 / Orphan 142 / Odd Fixes 104 / Obsolete 2.

**Hierarchy** is by title convention, not structure: 86 titles use `PARENT - CHILD`
(e.g. `MEMORY MANAGEMENT - PAGE ALLOCATOR`), 67 use `PARENT [CHILD]` (e.g.
`NETWORKING [TCP]`). Families with >=3 children: MEMORY MANAGEMENT 20 (1 parent + 19),
NETWORKING 18, BPF 16, PIN CONTROLLER 13, SOUND 9, RUST 7, CPUIDLE 7, FILESYSTEMS 5,
CONTROL GROUP 4.

**Candidate levels** (from `level.py`, by `F:` directory depth):
- entries owning a top-level dir: **12** (BLOCK LAYER, CRYPTO API, IO_URING, MEMORY
  MANAGEMENT, NETWORKING [GENERAL], RUST, SECURITY SUBSYSTEM, SOUND, ... plus THE REST `*/`)
  — too coarse.
- entries owning a depth<=1 dir: 323; excluding drivers/ arch/ tools/ samples/ scripts/:
  **150** (mostly individual filesystems `fs/ext4/` and protocols `net/sctp/`).
- "core kernel" = depth<=1 dir under kernel/ mm/ block/ io_uring/ lib/ init/ ipc/ plus
  VFS, PAGE CACHE, NETWORKING [GENERAL/IPv4-IPv6/SOCKETS/TCP]: **42** entries (SCHEDULER,
  MEMORY MANAGEMENT, BLOCK LAYER, LOCKING PRIMITIVES, RCU, IRQ SUBSYSTEM, PRINTK,
  CGROUP, MODULE SUPPORT, TRACING, IO_URING, ...). Sits in the 20–80 band. Note
  WORKQUEUE and TIMEKEEPING are *not* in it (they own files, not dirs) — any level
  rule needs a hand-curated exception list.
- the MEMORY MANAGEMENT family alone: **20** project-authored sub-components for the
  2,541-sentence `Documentation/mm/` corpus. Cleanest single arm.

**Do docs use the MAINTAINERS strings?** No. Ten pairs (doc phrase with count above →
entry title): "the VFS"→`FILESYSTEMS (VFS and infrastructure)`; "the scheduler"/"CFS"→
`SCHEDULER`; "workqueue"→`WORKQUEUE`; "the block layer"/"blk-mq"→`BLOCK LAYER`;
"the page allocator"/"buddy allocator"→`MEMORY MANAGEMENT - PAGE ALLOCATOR`;
"RCU"→`READ-COPY UPDATE (RCU)`; "cgroup"→`CONTROL GROUP (CGROUP)`; "spinlock"/"mutex"/
"lockdep"→`LOCKING PRIMITIVES`; "page cache"→`PAGE CACHE`; "RSS"/"XPS" (scaling.rst)→
`NETWORKING [GENERAL]`. Head nouns match after lower-casing and stripping
parentheticals; acronyms (CFS, RSS, XPS, blk-mq) and mechanism names (spinlock)
need an alias step. This is the same shape as the benchmark's "generic term" problem
but at higher density: "the kernel" (29) and "the filesystem" (28) are not components.

## D3 — Code→component map

`F:` lines: **9,868** total (1,501 with wildcards; 28 entries use `X:` exclusions;
105 `N:` regex lines not evaluated). 2,432 `F:` lines point into `Documentation/`;
1,715 entries list at least one doc file.

Coverage of the 96,026-file tree (matcher: dir-prefix for `x/`, exact-or-prefix for
files, `fnmatch` on each path prefix for wildcards; excludes THE REST `*/`, which would
make it 100% by construction): **92,222 (96.0%) by non-wildcard patterns, 93,758
(97.6%) with wildcards; 2,268 uncovered**, concentrated in `include/` 1,260, `tools/`
513, `drivers/` 191, `kernel/` 86. Granularity is per-entry and mixed: dirs
(`kernel/sched/`) next to single files (`kernel/fork.c` under SCHEDULER,
`mm/vmscan.c` split between RECLAIM and MGLRU). Overlap is the norm: `fs/*` under VFS
matches every filesystem; `mm/` under MEMORY MANAGEMENT matches all 19 children.
"Most-specific pattern wins" (get_maintainer.pl semantics) resolves most of it.

## S1 — Self-supervised sentence-level gold

Signal density per candidate doc (sentences carrying it):

| doc | sent | path | `func()` | `:doc:`/`:ref:` |
|---|---|---|---|---|
| vfs.rst | 495 | 4 | 62 | 0 |
| physical_memory.rst | 231 | 8 | 29 | 6 |
| scaling.rst | 222 | 2 | 7 | 0 |
| workqueue.rst | 214 | 2 | 7 | 0 |
| locktypes.rst | 134 | 0 | 19 | 0 |
| page_migration.rst | 92 | 1 | 12 | 1 |
| sched-design-CFS.rst | 69 | 2 | 0 | 0 |
| blk-mq.rst | 50 | 0 | 0 | 4 |

Across all 169,253 sentences: **2,402 (1.4%) contain a path, 7,829 (4.6%) a
`func()`, 2,721 (1.6%) an `:doc:`/`:ref:`/hyperlink**.

**(a) Paths → MAINTAINERS.** In the 8 chosen docs there are only **17 distinct path
mentions** (not 20 — the docs do not carry more). Resolution through `F:` (umbrella
DOCUMENTATION/THE REST excluded): **7/17 unique at any level, 10/17 unique at
most-specific pattern, 6/17 no owner** (4 are `Documentation/...` targets nobody claims,
2 are `tools/workqueue/*.py`, which exist but have no `F:`). Second sample, the 20
most-mentioned existing code paths tree-wide: **10/20 unique any-level, 15/20 unique at
most-specific**; failures are `init/main.c` and `include/linux/compat.h` (no owner) and
`include/linux/pm.h` (3 owners). Tree-wide, clean path mentions (file ext or trailing
`/`, no wildcards) are 1,402 distinct in 2,143 sentence-occurrences, 949 of them code
paths; 237 (16.9%) do not exist in the tree (see T1; ~10 are lwn.net URL fragments
captured as `net/Articles/...`, ~15 are placeholders like `sound/xxx.h`).
**Yield: ~1% of sentences, ~60–75% of those resolve uniquely, so roughly 0.6–1.0% of
sentences get a masked-anchor label.** For the 8 docs that is ~10 labelled sentences.
Too thin to be the gold; usable as a sanity probe only.

**(b) `func()` references.** Much denser (vfs.rst 12.5% of sentences) and spot-check of
10 random vfs.rst hits is 10/10 real function names (`d_instantiate_new()`,
`get_tree()`, `d_iput()`, `release_folio()`, `d_splice_alias()`, ...). But resolving a
symbol to an `F:` pattern requires a symbol index over the full tree (ctags/kernel-doc),
which this probe did not build; and the resolution lands on sibling ambiguity —
`d_*` → `fs/dcache.c` → VFS, `release_folio` → `include/linux/pagemap.h` → PAGE CACHE
— which is exactly the linker's known hard case (sibling-component confusion).

**(c) Cross-links.** 1.6% tree-wide; in the 8 docs mostly physical_memory.rst pointing
at other mm docs. Resolving `:doc:` through S2 gives a doc-level owner, not a component
distinct from the current doc's own owner, so the signal is nearly circular.

**Noise spot-check (10 path sentences).** 10/10 are genuine "see X" or "declared in X"
references; the path names the *document* or *header*, not the component the sentence
is *about* ("Filesystem locking is described in Documentation/filesystems/locking.rst"
resolves to PAGE CACHE, which owns locking.rst — see S2 quirk).

## S2 — Doc-level gold

Of 11,439 `Documentation/` files (matcher as in D3, umbrellas excluded): **3,377 (29.5%)
owned by exactly one entry, 1,144 (10.0%) by none, 6,918 (60.5%) by >=2**; with
most-specific-pattern tie-break **10,061 (88.0%) resolve to one entry**. Restricting
to `.rst`: 4,042 files, **2,685 (66.4%) exactly one owner, 577 (14.3%) none**.
Quirk that matters: `Documentation/filesystems/vfs.rst` and `locking.rst` are owned by
`PAGE CACHE`, not `FILESYSTEMS (VFS and infrastructure)` (which claims `fs/*` but no
docs). Doc-level ownership therefore labels the *maintainer who volunteered*, not the
component the prose is about.

## T1 — Downstream task

**Co-change (GitHub commits API, `/tmp/oss-case/linux/cochange*.json`).** 5 docs,
33 commits fetched with file lists; 4 were merge commits (the `?path=` API lists
merges, which inflate "code touched" — vfs.rst's 168-file hit is `mm-stable` merge).
Non-merge commits that also touch code: sched-design-CFS 0/6, page_migration 2/6,
workqueue 2/7, blk-mq 1/7 (a tree-wide `:c:type` sweep), vfs.rst 3/3.
**8/29 = 28%** overall; excluding vfs.rst, 5/26 = 19%. Doc edits are mostly doc-only
("fix bracket", "correct places -> place", translation sync). Co-change gold would be
small and skewed to vfs.rst-style API docs.

**Doc staleness (verified against `tree_files.txt`).** Three concrete, live cases:
1. `Documentation/networking/can.rst` L232, L409: "defined in `include/linux/can.h`" —
   file is `include/uapi/linux/can.h` (moved to uapi in 2012).
2. `Documentation/driver-api/usb/writing_musb_glue_layer.rst` L93, L431:
   `arch/mips/jz4740/platform.c` — `arch/mips/jz4740/` has 0 files in the tree.
3. `Documentation/admin-guide/device-mapper/dm-log.rst`: `include/linux/dm-log-userspace.h`
   — now `include/uapi/linux/dm-log-userspace.h`.
Plus proof maintainers fix these by hand: commit `73b5d07990a0` (2026-08-12) "Docs/mm:
fix outdated 'radix tree' in page_migration" — and 8 rst files still say "radix tree".
Protocol: for every sentence linked to component C, check that C's `F:` files still
contain the referenced symbols/paths; gold = the 236 missing code paths tree-wide
(after removing URL fragments/placeholders, ~200), plus git history of doc fixes with
"outdated"/"stale"/"rename" in the subject. Rough size: ~200 file-level, unknown at
symbol level (needs a ctags index over the full tree).

**Doc-update recommendation** from co-change: at 19–28% code co-change and mostly
API-level docs, gold would be a few dozen (doc-sentence, code-dir) pairs per
subsystem — small.

## K — Killer-case pitch

The kernel is the one system where "which maintainer entry does this paragraph of
`vfs.rst` belong to" is a real question people answer by hand (get_maintainer.pl only
does files, not prose), and where 200+ doc sentences point at files that no longer
exist; sentence-level links would turn `MAINTAINERS` into a doc-routing table.

## C — Cost / feasibility

- Sizes: 8-doc core arm = 1,507 sentences x 42 components = 63k pairs (benchmark
  max ~200 x 14 = 2.8k, i.e. 22x); mm arm = 2,541 x 20 = 51k; full
  core-api+mm+scheduler+block+locking = 10,660 sentences x 42 = 448k (160x).
  The per-doc size (50–495 sentences) is inside the benchmark's range; the scale
  comes from doc count, so the linker can run per doc unchanged.
- Component list must be hand-curated from MAINTAINERS (42 or 20); names need an
  alias/normalisation pass (D2).
- Licence: GPL-2.0 (WITH Linux-syscall-note). Vendoring the rst files into the
  replication package is fine with the licence text and SPDX headers preserved; keep
  them in a clearly GPL-2.0 subdirectory if the package licence differs.
- No sentence-level gold exists without human labelling: S1 path yield ~1%, `func()`
  yield ~5–12% but needs a symbol index and resolves to sibling ambiguity, doc-level
  ownership is maintainer-volunteered (vfs.rst → PAGE CACHE).

## Verdict

READY-WITH-WORK — prose (D1), component list (D2, 42 or mm-20) and code map (D3, 97.6%) are all project-authored and measured.
Blocked on gold: S1 gives ~1% path-anchored sentences (10/17 and 15/20 resolve uniquely), so a human-labelled subset (e.g. vfs.rst 495 + physical_memory.rst 231) is unavoidable.
Best arm: `Documentation/mm/` (2,541 sentences) against the 20 MEMORY MANAGEMENT entries, with the ~200 stale-path references as the downstream staleness task.
