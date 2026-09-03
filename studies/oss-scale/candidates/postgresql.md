# PostgreSQL as an OSS-scale case for the s110 linker

Measured 2026-09-03 on postgres/postgres master `f7a563ccd1` (blobless clone, sparse
checkout of `doc/src/sgml src/backend src/include src/common`).
Raw artifacts and scripts: `/tmp/oss-case/postgresql/` (repo/, sentences/, split.py,
cochange.py, coderefs.py, log_since2018.txt, funcdefs.txt, fileindex.txt, readme_dates.txt).
Every number below comes from those scripts unless stated otherwise.

## D1 Architecture prose

Two prose families exist; both are git-tracked, both under the PostgreSQL License
(`COPYRIGHT`, covers "this software and its documentation").

**(a) Manual chapters** (`doc/src/sgml/*.sgml`, rendered at
https://www.postgresql.org/docs/current/). Sentence counts after stripping SGML tags,
`<programlisting>/<screen>/<synopsis>`, tables and titles, then splitting on `[.!?]` +
capital, keeping sentences of >= 4 words (`split.py`):

| file | lines | sentences | nature |
|---|---|---|---|
| arch-dev.sgml ("Overview of PostgreSQL Internals") | 575 | 136 | true architecture prose |
| storage.sgml ("Database Physical Storage") | 1,157 | 231 | architecture (on-disk) |
| wal.sgml ("Reliability and the Write-Ahead Log") | 1,104 | 274 | architecture + tuning |
| indexam.sgml ("Index Access Method Interface") | 1,588 | 356 | API description, prose-heavy |
| mvcc.sgml ("Concurrency Control") | 2,003 | 334 | user-facing semantics |
| bgworker.sgml ("Background Worker Processes") | 339 | 77 | API |
| xact.sgml ("Transaction Processing") | 205 | 54 | user-facing |
| **total** | 6,971 | **1,462** | |

The clearly architectural core (arch-dev + storage + wal) is 641 sentences. mvcc/xact
are about isolation levels, not structure: only 9/334 and 3/54 of their sentences name
any subsystem (density measurement in S1).

**(b) In-tree design READMEs** (`src/backend/**/README*`): 39 files, 13,199 lines,
**4,058 sentences** after dropping indented code, ascii tables, and rules
(two files, `utils/mb/README` and `README.euc_jp`, are file lists and yield 0).
Largest: optimizer/README 482, nbtree/README 434, transam/README 296, lmgr/README 245,
gin/README 202. Full per-file table in `sentences/`. These are the richest
architecture prose in the repo but they are per-subsystem, so the "which component"
question is partly answered by file location (see S2).

**Naming form.** Lowercase common nouns with a definite article, occasionally an
acronym; almost never a directory name. Ten sentences (arch-dev/storage/wal):

1. "The task of the planner/optimizer is to create an optimal execution plan."
2. "The parser has to check the query string (which arrives as plain text) for valid syntax."
3. "So the executor recursively calls itself to process the subplans (it starts with the subplan attached to lefttree)."
4. "The first one worked using row level processing and was implemented deep in the executor."
5. "This delay allows other server processes to add their commit records to the WAL buffers so that all of them will be flushed by the leader's eventual sync operation."
6. "By doing this, during crash recovery PostgreSQL can restore partially-written pages from WAL."
7. "Tables also have a visibility map, stored in a fork with the suffix _vm, to track which pages are known to have no dead tuples."
8. "Visibility map bits are only set by vacuum, but are cleared by any data-modifying operations on a page."
9. "The WAL record headers are described in access/xlogrecord.h; the record content is dependent on the type of event that is being logged."
10. "There are two internal functions to write WAL data to disk: XLogWrite and issue_xlog_fsync."

Whole-word frequency across the 1,462 SGML sentences / 4,058 README sentences:
WAL 86/131, access method 70/4, heap 28/84, background worker 26/2, planner 20/26,
executor 15/39, postmaster 12/15, parser 12/14, optimizer 10/9, planner/optimizer 6/0,
visibility map 8/2, free space map 5/4, rule system 3/0, rewriter 2/0,
buffer manager 1/4, storage manager 1/3, lock manager 1/11, traffic cop 0/0, tcop 0/0,
bufmgr 0/6, smgr 0/7, lmgr 0/4, xlog 0/6. So the manual uses prose names; the READMEs
mix prose names with code names (bufmgr, smgr, xlog).

## D2 Component model

Project-authored lists, all directory-based:
- `src/backend/Makefile` `SUBDIRS` and `src/backend/meson.build` lines 10-37: **26**
  build subsystems (access archive backup bootstrap catalog commands executor foreign
  jit lib libpq main nodes optimizer parser partitioning port postmaster regex
  replication rewrite statistics storage tcop tsearch utils) + po.
- Directory tree: **28** top-level dirs (the 26 above + po + snowball), **53** second-level
  dirs (`backend_l1.txt`, `backend_l2.txt`). Second level is where the doc's nouns live:
  access/{brin,common,gin,gist,hash,heap,index,nbtree,rmgrdesc,sequence,spgist,table,
  tablesample,transam}, storage/{aio,buffer,file,freespace,ipc,large_object,lmgr,page,
  smgr,sync}, optimizer/{geqo,path,plan,prep,util}, replication/{libpqwalreceiver,
  logical,pgoutput,pgrepack}, utils/{activity,adt,cache,error,fmgr,hash,init,mb,misc,
  mmgr,resowner,sort,time}, jit/llvm, port/*, snowball/*, tsearch/dicts.
- arch-dev.sgml's own enumeration (sect1 titles, lines 26-112): connection
  establishment, parser stage, rewrite system, planner/optimizer, executor, plus "the
  storage system". That is **5-6 named stages**, i.e. the same size as the benchmark
  projects; it is not a 20-80 list.
- No other list: no MAINTAINERS/OWNERS/CODEOWNERS (`git ls-tree -r | grep -i` = none),
  `src/tools/` has no subsystem registry.

**Doc name vs dir name.** Mismatches are the rule, not the exception:

| doc says | dir | note |
|---|---|---|
| planner / planner/optimizer | optimizer | 20 "planner" vs 10 "optimizer" in SGML |
| rewrite system / rewriter / rule system | rewrite | "rewrite" alone 5 hits, all as verb/noun |
| WAL / write-ahead log | access/transam (xlog.c) | 86 SGML mentions, dir name never appears |
| buffer manager / shared buffers | storage/buffer | level 2 |
| storage manager | storage/smgr | level 2 |
| lock manager | storage/lmgr | level 2 |
| (never named: "traffic cop") | tcop | 0 mentions in 5,520 sentences |
| background worker | postmaster/bgworker.c | file level |
| visibility map | access/heap/visibilitymap.c | file level |
| free space map | storage/freespace | level 2 |
| executor, parser, postmaster, catalog | same | the only clean matches |

And the reverse hazard: bare dir names that are common English words hit sentences
that are not about that dir. SGML sentence hits: access 132 (mostly "access method"),
storage 33, commands 28 (SQL commands), main 25, statistics 6, backup 5, nodes 4 (61 in
READMEs, mostly "plan nodes"). A component list built from raw dir names would invite
exactly the sibling-confusion FPs the linker already suffers.

## D3 Code -> component map

The directory tree is the map; coverage is 100% by construction at both levels.
`src/backend` holds 919 `.c/.h/.y/.l` files (1,316 files total); `src/include` mirrors
the same dir names for headers (1,753 name->path entries in `fileindex.txt`).
Files per top-level dir (C-family / all): utils 238/403, access 157/198, executor 65/68,
storage 59/91, commands 56/58, snowball 56/78, optimizer 52/66, catalog 34/43,
replication 31/43, parser 24/29, libpq 17/23, postmaster 16/18, nodes 16/21,
tsearch 15/26, backup 14/16, regex 13/17, port 10/17, lib 9/12, rewrite 8/10,
statistics 8/13, tcop 7/9, jit 5/15, bootstrap 3/6, partitioning 3/5, archive 1/3,
foreign 1/3, main 1/3, po 0/17. utils is a catch-all (26% of code) that no doc
sentence names as such ("utils": 2 SGML hits).

## S1 Self-supervised sentence-level gold

**Code references in sentences** (`coderefs.py`; regexes: `src/(backend|include)/...`,
`name()`, `name.[ch]`; resolved via a 17,735-name function-definition index built from
`^name(` lines in `src/backend/**/*.c` and the header/file index):

| corpus | sentences | any ref | path | func() | .c/.h | resolve to exactly 1 top dir |
|---|---|---|---|---|---|---|
| 7 SGML chapters | 1,462 | **24 (1.6%)** | 8 | 12 | 10 | 18 |
| 39 READMEs | 4,060 | **254 (6.3%)** | 16 | 173 | 80 | 195 |

Twenty resolved examples are in the coderefs.py output; judging the 20 by hand,
14 are on-topic (sentence is about the dir it resolves to, e.g. "See
src/backend/storage/freespace/README for more details on how the FSM is structured"
-> storage; "The record decoding functions and macros in xlogreader.c/h" -> access),
6 are off-topic (the code name is incidental: "Functions returning such types will need
to palloc() their result space" -> utils/mmgr; "can be obtained using the
pg_relation_filenode() function" -> utils/adt while the sentence is about on-disk
layout; "Async_Notify()" -> commands in a bgworker sentence). Noise estimate ~30%.
Net: **~18 usable gold sentences in the manual chapters**, ~140 in the READMEs. That is
not a gold standard; it is a sanity probe.

**Co-change** (`git log --since=2018-01-01 --name-only -- doc/src/sgml src/backend`,
14,547 commits, `cochange.py`):
- commits touching both an `.sgml` and `src/backend/`: **1,665**; with exactly one
  top-level backend dir: **685**.
- per top-level dir (commits with that dir + some sgml / commits where it is the only
  backend dir): utils 898/232, commands 551/119, catalog 443/44, access 370/58,
  replication 273/60, parser 240/21, storage 216/19, postmaster 189/16, executor 177/10,
  optimizer 153/13, nodes 135/1, tcop 132/9, libpq 100/35, statistics 55/14,
  rewrite 46/4, bootstrap 43/1, jit 34/6, partitioning 33/0, port 31/6, tsearch 24/0,
  backup 21/5, main 18/0, foreign 18/1, lib 16/0, snowball 16/5, regex 16/3.
- but the sgml files hit are reference material: config.sgml 322, func.sgml 243,
  monitoring.sgml 191, catalogs.sgml 158, alter_table.sgml 70, protocol.sgml 69.
  Commits touching one of the 7 architecture chapters: **81** (wal 21, indexam 27,
  storage 15, mvcc 12, bgworker 7, xact 1, arch-dev 0).
- 5 random both-commits: a14e75eb0b (create_collation/initdb docs -> commands);
  9219093cab (config.sgml -> libpq,postmaster,tcop,utils); c7db01e325 (initdb.sgml ->
  commands); 70a7732007 (extend/contrib docs -> commands,parser); 8e72d914c5
  (dml/glossary docs -> executor,nodes,optimizer,parser,rewrite,utils).
- 5 random arch-doc both-commits: 92fe23d93a (indexam+btree -> access,commands,utils);
  ebcc7bf949 (indexam -> access,utils); 6c349d83b6 (wal+config -> access,storage,utils);
  1d257577e0 (wal -> access,catalog,postmaster,storage,utils); a228cc13ae (wal ->
  access,catalog,postmaster,replication,storage,utils). Multi-dir is typical.

**Sentence-level feasibility.** `git show <h> -- <arch sgml>` does yield added text,
but: (i) lines are wrapped mid-sentence so sentences must be re-joined from the
post-image; (ii) in indexam.sgml the added lines are mostly struct fields inside
`<programlisting>` (commits ce62f2f2a0, af4002b381, c09e5a6a01); (iii) many are typo
sweeps (410aa248e5 touched 10 sgml files for grammar). Over all 81 arch-doc commits:
60 add >= 1 non-tag prose line of >= 6 words, distribution 0:21 1:13 2:5 3:7 4:4 5:9
6+:22 (`archboth_prose.txt`); **prose + single backend dir: 9 commits**. So co-change
gives on the order of 9 clean / ~60 noisy sentence-to-dir instances for the
architecture chapters since 2018. `git log -L` was not needed; plain diff suffices.

**README co-change** is cleaner but circular: 227 commits touch a backend README;
129 also touch code in the same dir, 30 only other dirs, 68 README-only. Since the
README's location already labels it (S2), co-change adds nothing there.

## S2 Doc-level gold

- Manual chapters: **none**. No ownership file of any kind in the repo.
- READMEs: location is the label. 39 files -> 33 distinct dirs (level 1, 2 or 3), covering
  **4,058 sentences** with a doc-level component. Caveat measured in D1: README
  sentences routinely name other subsystems (nbtree/README mentions WAL, buffer
  locks, VACUUM), so "location = component of every sentence" is a prior, not gold.

## T1 Downstream task

1. **Doc-update recommendation** ("commit touched dir X; which architecture sentences
   should be re-read?"). Gold from co-change above: 9 clean, ~60 noisy instances for
   the architecture chapters; 685 clean single-dir instances if the whole manual is
   admitted, but those point to config/func/monitoring reference pages, which the
   linker is not built for (no architecture prose). Protocol would be: for each of
   the 60, take the sentences whose lines the commit modified, ask whether the linker
   links them to the modified dir; report recall@k over the dir's linked sentences.
   Honest size: small, and the 60 skew to indexam (API text) and wal.
2. **Doc staleness.** Spot-check of 5 doc-named functions with no `.c` definition:
   `union_planner()` (optimizer/plan/README, last modified 2010-11-23) — **stale**, the
   function was replaced by `grouping_planner()` (planner.c:153), the README still
   describes it as "old planner() function"; `PickSplitFn()` — placeholder for a
   user-supplied opclass function, not stale; `errsave()`, `GinDataPageGetRightBound()`,
   `pg_enable_data_checksums()` — exist (macros / SQL function). 1 of 5 stale. README
   freshness: 4 of 39 untouched since 2018 (2007, 2010, 2014, 2017); 16 modified in
   2025-2026 (`readme_dates.txt`). The manual chapters are actively maintained
   (wal.sgml: 21 co-change commits). Staleness gold is therefore thin: a handful of
   README paragraphs, not a stream.
3. **Newcomer navigation** is the realistic use: arch-dev's "path of a query" ->
   optimizer/, executor/, parser/, rewrite/, tcop/ is exactly the mapping a newcomer
   lacks, and the doc never says "tcop" or "optimizer" for two of the five stages.
   No history-derived gold for this; it would be a hand-labelled set.

## K Killer-case pitch

PostgreSQL's most-read internals chapter describes five stages by prose names
("planner/optimizer", "rewrite system") that map onto 28 directories whose names
never appear in the text ("tcop", "optimizer", "transam"); links from those 641
sentences to the tree are the missing index between the manual and `src/backend`.

## C Cost / feasibility

- Scale: 1,462 SGML sentences x 28 top-level components (or 641 core sentences x
  ~20 doc-native names), 7x the largest benchmark doc and 2-4x its component count;
  optionally 4,058 README sentences x 53 level-2 components.
- Licence: PostgreSQL License (BSD-like, permissive), explicitly covers documentation;
  vendoring the 7 sgml files + 39 READMEs into the replication package with the
  `COPYRIGHT` notice is fine.
- Work needed before running: (1) decide the component list — raw dir names are
  hazardous (D2 table), so a project-derived list with doc-native aliases
  (optimizer="planner", rewrite="rewrite system", access/transam="WAL") must be
  justified without hand-tuning to the doc; (2) hand-label a gold subset, since S1
  yields ~18 sentences and 9 clean co-change commits for the chapters — plan roughly
  the 641 core sentences x ~20 components; (3) drop mvcc/xact/bgworker from the
  architecture input (density 3-12%).

## Verdict

READY-WITH-WORK: prose, component list, and code map all exist, are project-authored,
git-tracked, and permissively licensed, at 5-7x benchmark scale; but doc names and
directory names disagree for the key subsystems, and the only self-supervised gold
is ~18 code-reference sentences plus 9 clean co-change commits, so a hand-labelled
gold set is unavoidable for a quantitative claim.
