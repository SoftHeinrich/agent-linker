# Recovered Trace Links — SOTA baselines (ArDoCo dataset)

Persisted, normalized recovered trace links for the doc-to-code and doc-to-model
traceability tasks, collected from the SOTA replication packages so they survive
`mvn clean` and are directly comparable.

> **Filling the paper?** See the cross-system results index:
> `transarc-emp/reports/RESULTS_INDEX.md` — per-system locations + RQ1/RQ2 numbers.

## Tasks / layout

| Folder | ArDoCo task | Meaning | target_id is |
|--------|-------------|---------|--------------|
| `doc-code/`  | SAD-Code | architecture-documentation sentence → source-code file | code file path |
| `model-doc/` | SAD-SAM  | architecture-documentation sentence → architecture-model element | PCM model element id (`_...`) |

Full ArDoCo dataset = 5 projects: `bigbluebutton`, `jabref`, `mediastore`, `teammates`, `teastore`.

## File format (normalized)

CSV with header `sentence_id,target_id`; one recovered link per row. Row count =
predicted-positive count (TP + FP) for that run.

Naming: `<approach>-<project>[-<model>].csv`.

## Sources & provenance

| Approach | Task(s) | Model | Origin | Note |
|----------|---------|-------|--------|------|
| **LiSSA**    | doc-code, model-doc | gpt-4o-mini-2024-07-18 **and** gpt-5-mini (flex), seed 133742243 | `lissa-replication/results/tracelinks/{d2c,d2m}/` (committed) | two model variants each |
| **Artemis**  | doc-code, model-doc | gpt-5.4 (flex) | TAAS25 `…/tlr/tests-tlr/target/raw-tracelinks/*.tsv` (build output) | ArTEMiS = SAD-SAM via NER; doc-code is ArTEMiS+TransArC |
| **TransArC** | doc-code | none (deterministic) | **canonical** `transarc-emp/results/*/sad-code/sadCodeTlr_*.csv` | classic SAD→SAM→Code; ICSE24 self-contained run reproduces 4/5 — see caveat 4 |
| **SWATTR**   | model-doc | none (deterministic) | **canonical** `transarc-emp/results/*/sad-sam/sadSamTlr_*.csv` | classic heuristic SAD-SAM; ICSE24 self-contained run reproduces 4/5 — see caveat 4 |

`manifest.csv` lists every persisted file with its row count.

## Coverage status (36 files)

- **model-doc (SAD-SAM):** LiSSA 5/5 ✅, Artemis 5/5 ✅, SWATTR 5/5 ✅.
- **doc-code (SAD-Code):** Artemis 5/5 ✅, TransArC 5/5 ✅, LiSSA **3/5** (⚠️ missing `jabref`, `teammates` — gold standards exist; LiSSA needs an OpenAI key to generate).

## Normalization

All files are canonicalized by `normalize.py` (idempotent): header `sentence_id,target_id`,
integer sentence id, rows deduplicated and sorted by `(sentence_id, target_id)`.

**Sentence indexing is uniform — verified, not assumed.** Every file's sentence ids are
**1-based and aligned to the ArDoCo gold standard**. This was confirmed empirically per
file by testing shifts {−1, 0, +1} against the gold standard and taking the one that
maximizes gold overlap (TP): shift **0** wins decisively for every Artemis, LiSSA,
SWATTR, and TransArC file, both tasks. (Internally `SentenceEntity#getSentenceNumber()`
is 0-based and the ArDoCo dumps add +1; the NER occurrence used by Artemis model-doc is
already 1-based — so all approaches land on the same 1-based convention. No shifting is
applied.)

## Caveats before computing metrics

1. **Model mismatch.** LiSSA has two model runs (gpt-4o-mini, gpt-5-mini); Artemis is a
   single gpt-5.4 run; SWATTR/TransArC are deterministic. Pair comparably
   (e.g. LiSSA gpt-5-mini ↔ Artemis gpt-5.4).
2. **IDs align across approaches:** model-doc uses the same PCM element ids; doc-code uses
   the same code file paths.
3. **doc-code gold granularity.** The `sad-code` gold standard lists *directory /
   compilation-unit prefixes* (e.g. `…/registry/`), while recovered links are full file
   paths. Match by prefix (and enroll the gold to compilation units) when scoring — a
   raw row-equality intersection will undercount.
4. **SWATTR/TransArC bigbluebutton provenance (decided: canonical).** Two runs exist;
   they are **identical for 4/5 projects** and differ only on bbb. The folder holds the
   **canonical** `transarc-emp/results` run (bbb SWATTR link F1 `.79`, TransArC file F1
   `.83`; macro ~`.80`; matches the paper headline & `reports/*_rq1_*.csv`). The
   **self-contained ICSE24 run** (`run-transarc-icse24.sh`, reproducible from Central with
   no unpublished deps) scores bbb lower (SWATTR `.29`, TransArC `.35`; macro ~`.70`) — it
   recovers different bbb links, *not* an id/model-version mismatch. ICSE24 bbb variants
   are kept in `baseline-repos/transarc-icse24/.../target/raw-links/` if needed.

## Evaluation (mini-src)

Scored with the paper's own `transarc-emp/mini-src/metrics.py` (stdlib, auto-detects the
`recovered-links` dialect). Panels saved in `eval/`. Macro averages over the 5 projects:

| Task | Approach | F1 | coverage | noise |
|------|----------|----|----------|-------|
| model-doc (link F1) | artemis (gpt-5.4) | **.836** | .79 | .07 |
| model-doc | swattr | **.799** | .79 | .14 |
| model-doc | lissa (gpt-5-mini) | .425 | .84 | .60 |
| model-doc | lissa (gpt-4o-mini) | .280 | .89 | .73 |
| doc-code (file F1) | artemis (gpt-5.4) | **.849** | .71 | .06 |
| doc-code | transarc | **.803** | .75 | .13 |
| doc-code | lissa (gpt-5-mini, 3 proj) | ~.20 | — | — |

**Validation (numbers are correct):**
- SWATTR/TransArC reproduce the paper headline (~.80) **and** the ArDoCo built-in eval
  (`TraceLinkEvaluationIT#evaluateSadSamTlrIT`/`#evaluateSadSamCodeTlrIT`) passes **10/10**
  against the packages' published `ExpectedResults`.
- LiSSA reproduces its replication's reported per-project F1 **exactly** (sad-sam: MS .307,
  TS .258, TM .106, BBB .173, JR .554; avg .280 — cf. `lissa-replication/results/COMPARISON.md`).

## Regeneration

Artemis raw dumps (needs `OPENAI_API_KEY`):
```
cd Replication-Package-TAAS25_.../Replication-Package-TAAS25/tlr
mvn -pl tests-tlr -am test-compile failsafe:integration-test -Dit.test=RawTraceLinksIT
```

SWATTR + TransArC baselines (deterministic, no API key) — from the **self-contained
ICSE24 package** (`baseline-repos/transarc-icse24`, no NER, no unpublished snapshot deps).
The TAAS25 tooling can't build these here: it needs unpublished dev snapshots
(`io.github.ardoco:named-architecture-entity-recognition:1.0.0-SNAPSHOT`, only `2.0.0`
on Central). The ICSE24 package bundles its own ArDoCo core, so it builds from Central.
```
bash recovered-links/run-transarc-icse24.sh
# builds ardoco+arcotl reactor (rev 0.18.0-ICSE24-SNAPSHOT), runs RawLinkDumpIT
# outputs: …/ardoco+arcotl/tests/tests-tlr/target/raw-links/{SWATTR-*-sad-sam.csv, TRANSARC-*-sad-code.csv}
```
Then fold into the folder: SWATTR — swap columns (`modelElementId,sentence` →
`sentence_id,target_id`) into `model-doc/`; TransArC — `sentence,code` as-is into
`doc-code/`; finally run `python3 normalize.py`. Dump test:
`baseline-repos/transarc-icse24/ardoco+arcotl/tests/tests-tlr/src/test/java/.../integration/RawLinkDumpIT.java`.

---

# Unified archive — all three tasks + our approach

The baseline files above cover two tasks (`doc-code`, `model-doc`) for SOTA systems
only. `build_unified.py` generalizes this into one place that **also** holds the
third link type (`model-code` / ArCoTL) and **our approach's** runs, organized so
you can see exactly which running config produced each link set. Additive — it
never rewrites the flat baseline CSVs above.

## Tasks (well separated)

| Folder | Link type | source → target | normalized header |
|--------|-----------|-----------------|-------------------|
| `model-doc/`  | **doc→model** (SAD-SAM)   | sentence → architecture-model element id | `sentence_id,target_id` |
| `doc-code/`   | **doc→code** (SAD-Code)   | sentence → code file path | `sentence_id,target_id` |
| `model-code/` | **model→code** (ArCoTL)   | model-element id → code file path | `source_id,target_id` |

## Our approach (`aalinker`), organized by running config

Native output is **doc→model**; **doc→code** is composed (our doc-model ∘ ArCoTL
model-code), matching how `s_linker20` produces SAD-Code.

```
model-doc/aalinker/<backend>_<knowledge>/<run>/<project>.csv      # normalized
                                          <project>.raw.csv        # verbatim extract (5 cols)
doc-code/aalinker-composed/<backend>_<knowledge>/<run>/<project>.csv      # normalized
                                                  <project>.raw.csv       # bridge: sentence_id,via_component,target_id
```

Config is encoded in the path — `backend` ∈ {`gpt-5.4`, `sonnet`}, `knowledge` =
`full` (full-knowledge only; no-knowledge pending the paused Phase-51 sweep),
`run` ∈ {run1,run2,run3}. Two source variants are tracked side by side:

| config slot | source variant | role (D-04 REVISED) | builder |
|-------------|----------------|---------------------|---------|
| `gpt-5.4_full`, `sonnet_full` | `s_linker20_union` (v2.6.6 extracts) | baseline | `build_unified.py` |
| `gpt-5.4_s21`, `sonnet_s21`   | **`s_linker21`** canonical Full (v2.6.6_extracts_s21[_sonnet]) | **paper Full** (gpt = body, sonnet = appendix mirror) | `build_s21_dump.py` |

The `_manifest.csv` in each `aalinker*` dir records the `*_full` baseline config +
provenance + row count + sha + (for model-doc) P/R/F1 vs gold; the S21 add-ons land in
`_manifest_s21.csv` (gpt-5.4) and `_manifest_s21_sonnet.csv` (Claude).
`UNIFIED_MANIFEST.csv` aggregates **every** per-task manifest (`_manifest.csv` +
`_manifest_*.csv`), so it holds all four aalinker configs + arcotl (125 rows).

## raw + normalized coexist

Every generated link set ships **twice**, side by side: `<name>.csv` (normalized —
deduped, sorted, `Implementation/` prefix stripped to match the existing baselines)
and `<name>.raw.csv` (verbatim source, for integrity checks). The composed
doc-code `.raw.csv` keeps the bridging component so each doc→code link is traceable
back through the model layer.

## Gold standards

Vendored per task under `<task>/gold/<project>.csv` (+ `.raw.csv`) from the ARDoCo
benchmark, so the folder is self-sufficient for scoring. Ground truth, not
recovered — kept clearly separate.

## model-code (ArCoTL)

`model-code/arcotl/<project>.csv` — deterministic recovered model→code links from
`transarc-emp/results/*/sam-code/`. Single set, shared by all our runs (the
composition reuses it). No per-run/backend variation (heuristic, no LLM).

## Doc→code scoring: enrollment modes (standard vs the others)

The `doc-code` gold standard lists many targets at **directory/package**
granularity (a row whose path ends in `/`, e.g. `…/registry/`), not individual
files. How you reconcile package-level gold with file-level predictions is a
*scoring* choice — it does not change the recovered links — and it moves the
numbers a lot. Three modes exist; pick deliberately.

| Mode | What it does | Implementation | Use |
|------|--------------|----------------|-----|
| **Enrolled** *(the standard — what the benchmark & TransArc/ARDoCo papers report)* | Expands each package gold row into one `(sentence, file)` pair per concrete code-model file under that prefix, then set P/R/F1. Recovering files under a package matches the enrolled pairs. | `metrics.enroll(gold, code_files)` → `metrics.prf`; headline `file_f1` in `metrics.compute_sad_code` | Headline numbers, comparability with prior work |
| **No-enroll (atomic-package)** *(the un-inflated diagnostic)* | Each gold package counts as **one** atomic target. A predicted file collapses to the most-specific gold package it falls under (naming a package once satisfies it once — no multi-credit); predictions under no gold target stay as FPs. | `noenroll.py:noenroll_prf` | Measuring how much headline F1 is enrollment artifact |
| **Strict / file-exact** *(naive; undercounts)* | Direct `(sentence, exact_file)` set equality, no expansion. A concrete predicted file never equals a package gold row, so package targets are unmatchable. | plain `prf` on raw gold | Not for headlines — illustrates why some reconciliation is needed |

**Standard = enrolled.** The "others" are *no-enroll* (the honest middle ground
the paper uses to expose inflation) and *strict* (the naive lower bound). The
inflation gap `Δ = enrolled_F1 − noenroll_F1` is the quantity Ch2 critiques.

### No-enroll vs enrolled, all systems (macro)

| System | no-enroll F1 | enrolled F1 | Δ inflation |
|--------|-------------:|------------:|------------:|
| artemis (gpt-5.4)  | 0.632 | 0.849 | +0.217 |
| transarc           | 0.391 | 0.803 | **+0.412** |
| lissa (gpt-5-mini) | 0.138 | 0.198 | +0.060 |
| **aalinker / s20U (ours)** | **0.682** | **0.906** | +0.224 |

Full per-project breakdown + interpretation:
`transarc-emp/reports/NOENROLL_DOC_CODE.md` (regenerate with
`python3 transarc-emp/mini-src/noenroll.py`). Note enrollment does **not** preserve
ranking — TransArc and artemis look close enrolled (.80/.85) but separate sharply
no-enroll (.39/.63).

## Integrity check (built in)

Both builders recompute our model-doc P/R/F1 vs gold while building. `build_unified.py`
reproduces the `s_linker20_union` band (gpt-5.4 ≈0.894, sonnet ≈0.928); `build_s21_dump.py`
reproduces the canonical S21 Full band (gpt-5.4 **0.9360**, sonnet/Claude **0.9265**),
confirming the vendored links match the source runs.

## Regenerate

```
# baseline + s20_union full slots + arcotl bridge + gold (reads transarc-emp/results + benchmark)
python3 build_unified.py

# canonical S21 Full slots (gpt body + Claude appendix mirror); each pass also rebuilds
# UNIFIED_MANIFEST.csv by aggregating every per-task manifest, so run order does not matter.
# (git-tracked companion: transarc-emp/mini-src/build_s21_dump.py)
cd ../../transarc-emp
python3 mini-src/build_s21_dump.py                                          # gpt-5.4_s21
EXTRACTS_S21=../agent-linker/results/v2.6.6_extracts_s21_sonnet \
  S21_BE_DIR=sonnet S21_BE_TAG=claude S21_CONFIG=sonnet_s21 \
  S21_MANIFEST_TAG=s21_sonnet python3 mini-src/build_s21_dump.py            # sonnet_s21
```

`build_s21_dump.py` reads gold + the ArCoTL bridge from the already-built dump here
(not from `transarc-emp/results`), so it regenerates the S21 slots and the unified
manifest even when the upstream `results/` tree is absent. End the regenerate sequence
with `build_s21_dump.py` so `UNIFIED_MANIFEST.csv` includes the S21 rows.

## Caveats specific to the unified additions

1. **Composed doc-code uses *recovered* ArCoTL** (not gold model-code), mirroring
   the real `s_linker20` SAD-Code pipeline. So its quality is bounded by the ArCoTL
   model→code layer, which is the known SAM-Code bottleneck.
2. **Path prefix.** Normalized code paths strip a leading `Implementation/` (aligns
   our additions with the existing `transarc` doc-code files); the gold and Artemis
   baselines keep it — match by suffix/prefix when scoring (see caveat 3 above).
3. **No-knowledge runs absent.** Only full-knowledge `s_linker20_union` extracts are
   clean today; add a `<backend>_noknow/` config tier once the Phase-51 sweep runs.
