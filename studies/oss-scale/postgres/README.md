# Second audit: a document filed inside the component (PostgreSQL `src/**/README`)

The Linux audit (`../linux/`) tests the *ownership registry* pattern. This one tests the other
common in-the-wild pattern: **a design document that lives in the directory it describes**.
PostgreSQL keeps 42 such `README` files in source directories that hold C code — written by the
module's own authors, with no registry, no labels and no links.

```bash
curl -s "https://api.github.com/repos/postgres/postgres/git/trees/master?recursive=1" -o /tmp/pg.json
python3 build.py --docs 10
OPENAI_API_KEY="$OAI_KEY" OPENAI_SERVICE_TIER=flex OPENAI_REASONING_EFFORT=none \
  python3 ../linux/annotate.py --dir . --backend openai --model gpt-5.6-terra --tag terra
python3 ../linux/score.py --dir . --tag terra
```

* Components: the **141 source directories with ≥2 C files**, profiled by path and file names.
* Documents: 10 READMEs, 278 sentences (optimizer/plan, access/brin, storage/page, storage/smgr,
  access/rmgrdesc, replication, statistics, nodes, timezone, test_shm_mq).
* Same prompt and scorer as the Linux audit; the annotator is not told which directory the
  README came from.

| | Linux `MAINTAINERS` | PostgreSQL in-tree READMEs |
|---|---|---|
| components | 3,282 subsystems | 141 directories |
| sentences | 380 | 274 |
| **owner among ABOUT** | 0.784 | **0.869** |
| of sentences with any ABOUT | 0.879 | 0.926 |
| **documents voting for the owner** | 10/12 | **10/10** |
| BM25 owner in top-12, per batch | 0.346 | 0.831 |
| BM25 owner in top-12, per document | 0.750 | 1.000 |

Two things follow:

* The recipe holds on a C codebase, a different documentation style (design notes, not a guide)
  and a different kind of human assignment — nothing about it is rustc-shaped.
* The gap between the two audits is **retrieval, not labelling**: 141 candidates is an easy
  index, 3,282 is not, and pooling the query over the whole document recovers most of the loss
  in both. That is the same failure the rustc partial-name stage shows at 79 components.

The residual misses are the same kind as in the Linux audit: sentences that describe a
neighbouring module (`optimizer/plan` prose about `optimizer/path`, `nodes` prose about the
executor) — content-vs-placement disagreements, which is exactly what a placement-based human
source cannot resolve.
