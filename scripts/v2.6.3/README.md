# v2.6.3 Replay Pipeline (Phase 43)

Replay-stage scripts that read `results/phase_cache/s_linker19/{backend}/{project}/{layer1..4,final}.pkl`
and emit flat CSVs under `results/v2.6.3/{backend}/{project}/`. These CSVs are
the **contract** between the replay stage (this directory, in `agent-linker`)
and the format stage (Plans 03 and 04, in `transarc-emp/src/paper/`).

All four CSV schemas below are pinned. Plans 03 and 04 read them as-is.

## Invariants

1. **Zero LLM calls.** Every script calls `replay_common.assert_no_llm_env()`
   at the top of `main()` and hard-fails (RuntimeError) if `OPENAI_API_KEY`,
   `ANTHROPIC_API_KEY`, or `LLM_BACKEND` (anything other than unset/`checkpoint`)
   is set in the environment.
2. **GATE-01 byte-equality** of `src/llm_sad_sam/linkers/experimental/s_linker19.py`
   and `s_linker13_min.py` is preserved — no edits under `src/llm_sad_sam/` in
   this phase.
3. **Stdlib-only** in `evaluation/` (the downstream format stage). This replay
   stage uses stdlib + the existing `transarc-emp/src/lib/transarc_error_analysis`
   helpers (no new pip deps).

## RQ3 derivation table (CONTEXT D-08)

The four RQ3 variants derive deterministically from `layer3.pkl` and `layer4.pkl`:

| Variant         | Definition                                              | Derivation                                                  |
| --------------- | ------------------------------------------------------- | ----------------------------------------------------------- |
| `Full`          | Entity-validator ON + Coref/Citation-validator ON       | `layer3.validated  ∪ layer4.coref_validated` (= `final.pkl`)|
| `NoEntityValid` | Skip layer3 entity validator                            | `layer3.candidates ∪ layer4.coref_validated`                |
| `NoCitation`    | Skip layer4 coref/citation validator                    | `layer3.validated  ∪ layer4.coref_raw`                      |
| `NoValidator`   | Skip both LLM-call validators                           | `layer3.candidates ∪ layer4.coref_raw`                      |

All set operations key links on `(sentence_number: int, component_id: str)`.

In plain prose (matching the executor's grep contract):

    Full = layer3.validated ∪ layer4.coref_validated
    NoEntityValid = layer3.candidates ∪ layer4.coref_validated
    NoCitation = layer3.validated ∪ layer4.coref_raw
    NoValidator = layer3.candidates ∪ layer4.coref_raw

## Usage

```bash
# 5 projects × 2 backends × 2 CSVs each = 20 RQ1 CSVs
python3 scripts/v2.6.3/replay_s19_to_csv.py --all

# 5 × 2 = 10 RQ3 CSVs (plus 10 rq3_audit.csv)
python3 scripts/v2.6.3/replay_s19_rq3.py --all

# 5 × 2 = 10 RQ4 CSVs (plus 10 rq4_upset.csv)
python3 scripts/v2.6.3/replay_s19_rq4.py --all

# One-liner for the whole pipeline:
python3 scripts/v2.6.3/replay_s19_to_csv.py --all \
  && python3 scripts/v2.6.3/replay_s19_rq3.py --all \
  && python3 scripts/v2.6.3/replay_s19_rq4.py --all
```

Per-project / per-backend slices are available via
`--backend {claude,openai}` and `--project {mediastore,teastore,teammates,bigbluebutton,jabref}`.
Default `--out-root` is `<repo>/results/v2.6.3`.

## CSV schemas (the contract)

All CSVs are RFC 4180 with a single header row. Rows are sorted by their
natural key (the first column or the explicit ordering documented below).
Numeric columns are unrounded (six-decimal text in P/R/F1 fields; integer counts
verbatim). Plans 03/04 may re-round on render.

## sad-sam.csv

One file per `(backend, project)` at
`results/v2.6.3/<backend>/<project>/sad-sam.csv`.

| column            | datatype | semantics                                                                              |
| ----------------- | -------- | -------------------------------------------------------------------------------------- |
| `modelElementID`  | str      | UUID-style component ID from the SAM (matches gold-standard `modelElementID` column)   |
| `sentence`        | str      | Sentence number, stringified (matches transarc-emp's `load_result_sad_sam_standalone`) |
| `source`          | str      | `entity` or `coreference` — diagnostic (extra column; ignored by `csv.DictReader`)     |

Rows: one per link in `final.pkl.final`, deduplicated on `(sentence_number, component_id)`.

**Consumed by:** `transarc-emp/src/lib/metrics_api.py --task sad-sam` (via the
sandboxed `RESULTS` dir set up by Plan 03).

## sad-code.csv

One file per `(backend, project)` at
`results/v2.6.3/<backend>/<project>/sad-code.csv`.

| column      | datatype | semantics                                                                |
| ----------- | -------- | ------------------------------------------------------------------------ |
| `sentence`  | str      | Sentence number, stringified                                             |
| `codeID`    | str      | Normalized code file path (no `Implementation/` prefix)                  |

Rows: cartesian product of `sad-sam` links with the gold
`(modelElementID -> code_files)` map from
`transarc_error_analysis.load_gs_sam_code_maps`.

**Consumed by:** `transarc-emp/src/lib/metrics_api.py --task sad-code` (Plan 03).

## rq3.csv

One file per `(backend, project)` at
`results/v2.6.3/<backend>/<project>/rq3.csv`.

| column      | datatype | semantics                                              |
| ----------- | -------- | ------------------------------------------------------ |
| `variant`   | str      | `Full` \| `NoEntityValid` \| `NoCitation` \| `NoValidator` |
| `tp`        | int      | `|predicted ∩ gold|`                                   |
| `fp`        | int      | `|predicted - gold|`                                   |
| `fn`        | int      | `|gold - predicted|`                                   |
| `precision` | float    | `tp / (tp + fp)`, 0 when denominator is 0              |
| `recall`    | float    | `tp / (tp + fn)`, 0 when denominator is 0              |
| `f1`        | float    | harmonic mean, 0 when both are 0                       |

Rows: exactly four, in the order `Full, NoEntityValid, NoCitation, NoValidator`
(matches the CONTEXT D-08 table top-to-bottom).

**Consumed by:** `transarc-emp/src/paper/rq3_table.py` (Plan 04).

## rq3_audit.csv

Auxiliary per-validator audit. One file per `(backend, project)` at
`results/v2.6.3/<backend>/<project>/rq3_audit.csv`.

| column            | datatype | semantics                                                                  |
| ----------------- | -------- | -------------------------------------------------------------------------- |
| `validator`       | str      | `entity` \| `coref`                                                        |
| `killed_gold`     | int      | candidates the validator REJECTED that were IN gold (validator over-kills) |
| `killed_spurious` | int      | candidates the validator REJECTED that were NOT in gold (validator wins)   |
| `kept_gold`       | int      | candidates the validator APPROVED that were IN gold (validator preserves)  |
| `kept_spurious`   | int      | candidates the validator APPROVED that were NOT in gold (validator misses) |

Derivation: entity row reads `layer3.candidates` against `layer3.decisions[k].approved`;
coref row reads `layer4.coref_raw` against `layer4.coref_decisions[k].approved`.

Rows: exactly two, in the order `entity, coref`.

**Consumed by:** `transarc-emp/src/paper/rq3_table.py` (Plan 04, footer row).

## rq4.csv

One file per `(backend, project)` at
`results/v2.6.3/<backend>/<project>/rq4.csv`.

| column                | datatype | semantics                                                                              |
| --------------------- | -------- | -------------------------------------------------------------------------------------- |
| `linker`              | str      | `Entity` \| `Coref`                                                                    |
| `tps_caught`          | int      | `|linker ∩ gold|` (TPs this linker contributed)                                        |
| `unique_tps`          | int      | `|(linker - other_linker) ∩ gold|`                                                     |
| `fps`                 | int      | `|linker - gold|`                                                                      |
| `delta_f1_if_removed` | float    | `f1(E ∪ C) - f1(other_linker alone)` — true linker-ablation: removing this linker leaves the other linker's full predictions, including shared TPs. Typically ≥ 0; can be **negative** when the removed linker contributed no unique TPs but did contribute FPs (the other linker reproduces all the TPs and the union accrues only the removed linker's FPs). |

Rows: exactly two, in the order `Entity, Coref`. Both linker sets are
**post-validator** (`layer3.validated` for Entity, `layer4.coref_validated`
for Coref).

**Consumed by:** `transarc-emp/src/paper/rq4_table.py` (Plan 04).

## rq4_upset.csv

Auxiliary 3-cell UpSet decomposition. One file per `(backend, project)` at
`results/v2.6.3/<backend>/<project>/rq4_upset.csv`.

| column   | datatype | semantics                            |
| -------- | -------- | ------------------------------------ |
| `cell`   | str      | `only_E` \| `both` \| `only_C`       |
| `count`  | int      | `|cell ∩ gold|`                      |

Derivation:
- `only_E = (E - C) ∩ G`
- `both   = (E ∩ C) ∩ G`
- `only_C = (C - E) ∩ G`

Rows: exactly three, in the order `only_E, both, only_C`.

**Consumed by:** `transarc-emp/src/paper/rq4_upset.py` (Plan 04, UpSet figure).

## File counts produced by `--all`

| Script              | Files per (backend, project) | Total over 5 projects × 2 backends |
| ------------------- | ---------------------------- | ---------------------------------- |
| `replay_s19_to_csv.py` | `sad-sam.csv` + `sad-code.csv` | 20 |
| `replay_s19_rq3.py`    | `rq3.csv` + `rq3_audit.csv`    | 20 (10 + 10) |
| `replay_s19_rq4.py`    | `rq4.csv` + `rq4_upset.csv`    | 20 (10 + 10) |

Grand total: 60 CSV files under `results/v2.6.3/{claude,openai}/{mediastore,teastore,teammates,bigbluebutton,jabref}/`.
