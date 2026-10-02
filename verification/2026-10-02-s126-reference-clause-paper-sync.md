# 2026-10-02: s126 reference clause — paper result prose re-derived from the new sweep

Sweep: `results/greedymerge{,_noknow}_e2e_{terra,luna}_r{1,2,3}_20261002`
(commit e812f02e). Tables were synced with
`PAPER_DIR=$PWD/paper python3 evaluation/mini-src/sync_paper.py`; `--check` reported
`IN SYNC: all 28 paper file(s)`. Every prose number below was recomputed from the
synced CSVs. Gaps are differences of the unrounded scores, rounded afterwards.

## Body (terra) prose updated

`paper/table/{rq1-results,rq2-results,rq3-confusion,rq4-results,inference-cost}.csv`
feed `sections/results.tex` (25 replacements), the intro (2), the conclusion (4), the
discussion (4) and the abstract (`main.tex`, 2). Claims whose *meaning* changed:

- Smallest doc-model gain over Artemis: now JabRef (+6pp F1, +5pp F2). MediaStore is +7pp.
- Doc-code direction over Artemis: now 5/5 projects (was 4/5). The MediaStore
  "+2pp doc-model becomes −6pp doc-code" observation no longer holds (doc-code 0.937
  vs 0.927). It is replaced by BigBlueButton over TransArC (+9pp doc-model, −5pp
  doc-code; doc-code precision 0.68 vs 0.82). The causal clause "because the
  additional recall … comes from small components" was **removed**, because no
  measurement supports it for the new case.
- Ranking reversal example: moved from Teammates (no longer reversed: file F1 0.84 vs
  TransArC 0.82) to JabRef (TransArC file F1 0.94 vs 0.93; worst-component 0.80 vs 0.89).
- RQ4 zero-recovery components without knowledge: unchanged (MediaStore DB 47.5%,
  Reencoding 1.7%, TeaStore ImageProvider 45.3% of gold doc-code links; zero in 3/3
  no-knowledge terra runs; `results/regen_s126_noknow_zero_components_20261002.txt`).

## Luna paragraphs added (`sections/results.tex`, RQ1 and RQ2)

Source: `evaluation/reports/RQ12_BIGTABLE.csv` (average rows) and
`RQ12_PERPROJECT.csv`. These are the same data as `tab:detailed-approach` and
`tab:detailed-artemis`. Both systems are compared on the same backend.

| quantity | approach luna | Artemis luna | gap |
|---|---|---|---|
| doc-model F1 | 0.9000 | 0.7792 | +12.1pp |
| doc-code F1 | 0.8843 | 0.8017 | +8.3pp |
| doc-code worst-component F1 | 0.7514 | 0.5650 | +18.6pp |
| doc-code harmonic F1 | 0.9049 | 0.7288 | +17.6pp |
| doc-model CMR | 0.0% | 1.4% | |

- Run ranges. Approach luna doc-model F1 runs are 0.913 / 0.895 / 0.892, against
  Artemis luna 0.770 / 0.780 / 0.788. Doc-code is 0.903 / 0.880 / 0.871 against
  0.796 / 0.796 / 0.813. The lowest approach run exceeds the highest Artemis run on
  both tasks.
- Exception: on MediaStore, Artemis luna is ahead (doc-model 0.973 vs 0.926; doc-code
  0.942 vs 0.834).
- Backend difference. Approach doc-model F1 is 0.944 on terra and 0.900 on luna
  (−4.4pp), with precision 0.930 → 0.877 and recall 0.960 → 0.932. Artemis is 0.809
  on terra and 0.779 on luna (−3.0pp).

Not verified here: the paper was not compiled, because no LaTeX toolchain is
installed on this host.
