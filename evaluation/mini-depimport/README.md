# mini-depimport — dependency-based component importance

**Question.** Does a component's *code footprint* (how many files / model-code links
it owns) predict its *importance* (how much of the codebase depends on it)? For the
ARDoCo JabRef benchmark the answer is **no**, and that is the point: file/link-level
metrics under-weight small-footprint components that the rest of the system leans on.

This is the code-side companion to [`mini-inequality/`](../mini-inequality/README.md)
(which shows the doc-link side: small components own almost none of the links but are
documented as much as any other).

## Result (regenerated into `reports/`)

| component | files | Ca | Ca_share | Instab. | betweenness rank |
|-----------|------:|---:|---------:|--------:|-----------------:|
| model | 198 | 716 | 57.1% | 0.04 | #1 / 1451 |
| logic | 575 | 428 | 48.9% | 0.27 | #9 |
| **preferences** | **18** | **270** | **18.8%** | 0.22 | #7 |
| **globals** | **1** | **97** | **6.7%** | 0.18 | **#6** |
| gui | 641 | 25 | 3.1% | 0.93 | #2 |
| cli | 5 | 1 | 0.1% | 0.98 | #89 |

**`preferences` (18 files) and `globals` (1 file) are depended on by far more code
than the 641-file `gui`.** Footprint and importance invert. `cli` is the exception —
it is a consumer/entry point (Ca=1, Instability 0.98), not something depended upon.

## Metrics and citations

Argue with the **size-independent, textbook** metrics (top group). The centralities
(bottom group) are corroboration only — cite both the origin paper and an SE-application.

| metric | definition | citation | ρ vs file count |
|--------|------------|----------|----------------:|
| **Ca** (afferent coupling) | # distinct external files depending on the component | R.C. Martin 1994 | +0.26 |
| **Ce** (efferent coupling) | # distinct external files the component depends on | R.C. Martin 1994 | — |
| **Instability** I = Ce/(Ca+Ce) | 0 = stable/depended-on, 1 = unstable/consumer | R.C. Martin 1994 | +0.20 |
| CBO | distinct external files coupled either way | Chidamber & Kemerer 1994 | +0.71 *(avoid)* |
| Ca_share | Ca ÷ (all other files) — afferent reach fraction | derived ratio | +0.26 |
| PageRank | random-walk importance on the dependency graph | Page & Brin 1999 | corroboration |
| betweenness | fraction of shortest paths through the node | Freeman 1977 / Brandes 2001 | corroboration |

**Do not** build the argument on CBO, PageRank-total, or transitive "blast radius" —
they are size-confounded (ρ ≈ +0.7). `Ca` is Martin's metric at its native type/class
granularity (not the coarse 0–5 component-neighbour count).

Boundary alignment (dependency graph vs the model-code partition): 53.4% of reference
weight stays within a component; directed modularity Q = 0.246.

## Run

```bash
python3 depimport.py            # writes reports/{COMPONENT_VALUES,IMPORTANCE}.csv + DEPIMPORT.md
python3 depimport.py --check    # also asserts the frozen regression panel
```

- **Stdlib only** — PageRank (power iteration) and betweenness (Brandes) are implemented
  in-repo; no `pip install`. Runs in ~1.3 s.
- **Inputs:** the bundled dependency graph `data/jabref-maindeps-file.json.gz` and the
  model-code gold standard read from `$TRANSARC_BENCHMARK` (default sibling layout).
- **Regenerating the graph** from JabRef source (only needed if the benchmark changes):
  see [`data/PROVENANCE.md`](data/PROVENANCE.md).
