# No-enroll doc→code (SAD-Code) evaluation — all systems

Doc→code scored under the **no-enroll** regime (each gold package = one atomic
target) next to the **enrolled** regime that the benchmark/TransArc papers report.
The gap between them is the enrollment inflation this study critiques.

Reproduce: `mini-src/noenroll.py` — **retired 2026-08-26** with the s21 arm; this
file is the frozen last output (machine-readable: `reports/NOENROLL_DOC_CODE.csv`).
Recover the script from git history to re-run it.

## Macro over covered projects

| System | n | no-enroll P | no-enroll R | **no-enroll F1** | enrolled F1 | inflation Δ |
|--------|---|-------------|-------------|------------------|-------------|-------------|
| artemis (gpt-5.4)   | 5 | 0.642 | 0.691 | **0.632** | 0.849 | +0.217 |
| transarc            | 5 | 0.463 | 0.691 | **0.391** | 0.803 | **+0.412** |
| lissa (gpt-5-mini)  | 3 | 0.079 | 0.572 | **0.138** | 0.198 | +0.060 |
| lissa (gpt-4o-mini) | 3 | 0.070 | 0.585 | **0.124** | 0.196 | +0.072 |
| **s20U (ours, mean3)** | 5 | 0.646 | 0.795 | **0.682** | 0.906 | +0.224 |

## Per-project no-enroll F1 (MS / TS / TM / BBB / JR)

| System | MS | TS | TM | BBB | JR |
|--------|----|----|----|-----|----|
| artemis (gpt-5.4)  | 0.871 | 0.790 | 0.129 | 0.445 | 0.923 |
| transarc           | 0.568 | 0.824 | 0.094 | 0.398 | 0.071 |
| lissa (gpt-5-mini) | 0.130 | 0.218 | —     | 0.067 | —     |
| **s20U (ours)**    | 0.889 | 0.968 | 0.339 | 0.505 | 0.710 |

## Reading the result

- **Enrollment moves every system, but not equally.** TransArc inflates most
  (+0.412): its links are package/component-coarse, so enrolling the gold to
  concrete files credits a single coarse prediction across many enrolled file
  pairs. LLM systems that already predict at file granularity inflate less
  (artemis +0.217, ours +0.224).
- **Ranking is not preserved.** Under the headline enrolled F1, TransArc (0.803)
  and artemis (0.849) look close; under no-enroll, TransArc collapses to 0.391
  while artemis holds 0.632 — i.e. enrollment compresses real quality differences.
- **Ours leads in both regimes** (0.682 no-enroll / 0.906 enrolled), and its
  inflation gap (+0.224) is in the file-granular band, not the coarse-prediction
  band.
- **TransArc jabref (0.071 no-enroll)** is the extreme case: near-zero real
  file-level agreement that enrollment lifts into a respectable headline.

Provenance: SOTA links from `sota/recovered-links/doc-code/`; s20U from
`agent-linker/results/v2.6.5_s20union_sonnet` (runs 1–3, per-run mean). lissa
covers 3/5 projects (mediastore, teastore, bigbluebutton). See
`mini-src/noenroll.py` (no-enroll scoring) and `mini-src/metrics.py`
(`enroll`, `prf`, `compute_sad_code`) for the implementations.
