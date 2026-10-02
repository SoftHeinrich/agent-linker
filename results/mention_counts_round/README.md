# The `MENTION_COUNTS` deletion round: one sentence out of the s126 name judge

Question: what happens to the RQ1–RQ4 numbers if we delete only the sentence
`MENTION_COUNTS` from the s126 union-judge rule (`TRACE_LINK_RULE`)? The sentence is
"A mention that says nothing further about the component still counts as a valid
link." Every other byte of the prompt stays the same.

This is different from the earlier `v14mention` arm (`results/labelrule_round`). That
arm reworded the sentence on `s_linker120`. This round deletes it, on `s_linker126`.

## Design: a single-stage replay on the recorded runs

In `s_linker126` no linker sees an earlier linker's links (`SLinker126._run_linker`).
The final set is the by-pair merge of the name links and the coreference links
(`SLinker126.link`). Only the name judge reads `TRACE_LINK_RULE`. So for each of the
12 recorded runs (`greedymerge{,_noknow}_e2e_{terra,luna}_r{1,2,3}_20260916v2`) the
pilot does four things:

1. It loads that run's `knowledge.pkl` (the alias table) and rebuilds the name
   candidates. The scan is deterministic, and the rebuilt candidates matched the
   recorded `linker_name.pkl` candidates on all 60 project-runs (the script exits if
   they ever differ).
2. It re-asks only the name union judge, once per arm, in one invocation:
   `control` uses the head rule byte for byte, and `nomention` uses the rule with
   `MENTION_COUNTS` removed.
3. It merges the new name links with that run's recorded coreference links.
4. It writes a `run_ablation.py`-layout run directory per (arm, run), which the
   unchanged RQ scorers then score.

Deltas are read against `control` from the same invocation, not against the 09-16
recording. The in-set control already differs from the recording by 11–64 name links
per backend (`vs_recorded_*` in the summary CSV). That gap is API drift between
2026-09-16 and 2026-10-02, and it is as large as the arm effect.

Scope limit: the coreference links and the alias table are held fixed at their recorded
values, so this measures the name judge's sensitivity alone. That is exact for this
architecture, but it holds one recorded sample of each other stage and does not
re-sample them.

## Commands

```bash
OPENAI_REASONING_EFFORT=none OPENAI_SERVICE_TIER=default LLM_BACKEND=openai \
  python3 approach/pilot/mention_counts_replay.py --stamp 20261002 --workers 10
# 120 replays (2 arms x 2 models x 3 runs x 5 projects x {full, noknow}), 330 calls

for arm in control nomention; do
  python3 evaluation/mini-src/build_alinker_extracts.py --variant s_linker126 \
    --out $PWD/results/mcreplay_${arm}_extracts \
    --model terra results/mcreplay_${arm}_e2e_terra_r{1,2,3}_20261002 \
    --model luna  results/mcreplay_${arm}_e2e_luna_r{1,2,3}_20261002
  EXTRACTS_DIR=$PWD/results/mcreplay_${arm}_extracts DUMP_CONFIG=terra_mc$arm \
    DUMP_MANIFEST_TAG=mc${arm}_terra python3 evaluation/mini-src/build_dump.py
  EXTRACTS_DIR=$PWD/results/mcreplay_${arm}_extracts DUMP_BE_DIR=luna \
    DUMP_BE_TAG=gpt-5.6-luna DUMP_CONFIG=luna_mc$arm DUMP_MANIFEST_TAG=mc${arm}_luna \
    python3 evaluation/mini-src/build_dump.py
  python3 evaluation/mini-src/rq12.py --arm mc$arm
  T="mcreplay_${arm}_e2e_{model}_r{i}_20261002"
  RQ34_S92_DIR_TMPL="$T" python3 evaluation/mini-src/rq34.py --runs-from "$T" \
    --csv-root evaluation/reports/rq34/mcreplay_20261002/$arm
  RQ34_S92_DIR_TMPL="$T" python3 evaluation/mini-src/rq34_rq2.py --runs-from "$T" \
    --csv-root evaluation/reports/rq34/mcreplay_20261002/$arm
  T="mcreplay_${arm}_noknow_e2e_{model}_r{i}_20261002"
  for be in terra luna; do
    RQ34_S92_DIR_TMPL="$T" RQ34_ABLATION_KEY=s_linker126_noknow \
      python3 evaluation/mini-src/rq34.py --runs-from "$T" \
      --ablation-key s_linker126_noknow --backends $be \
      --csv-root evaluation/reports/rq34/mcreplay_20261002/${arm}_noknow_$be
  done
done
python3 studies/compare_arms.py mcnomention --base mccontrol \
  --csv evaluation/reports/ARM_COMPARE_mcnomention_vs_mccontrol.csv
```

Every `rq34.py` run reported `validate=OK`: the Full tp/fp/fn reproduced the replay's
own `ablation_*_replay.json`.

## Results (N = 3 runs per backend, paired in one invocation)

### RQ1/RQ2 (`ARM_COMPARE_mcnomention_vs_mccontrol.csv`, points, nomention − control)

| metric | terra mean | terra per-run | verdict | luna mean | luna per-run | verdict |
|---|---|---|---|---|---|---|
| doc-model F1 | −1.04 | −0.90 −0.66 −1.57 | WORSE 3/3 | +0.13 | −0.70 +2.35 −1.26 | inside noise |
| doc-model F2 | −1.26 | −1.16 −1.02 −1.60 | WORSE 3/3 | −0.63 | −1.07 +0.86 −1.69 | inside noise |
| doc-code F1 | −1.90 | −1.87 −1.46 −2.36 | WORSE 3/3 | −0.07 | −1.09 +1.98 −1.09 | inside noise |
| doc-code F2 | −2.10 | −1.62 −2.36 −2.32 | WORSE 3/3 | −0.97 | −1.82 +0.24 −1.33 | inside noise |
| doc-model CMR% | −0.22 (worse) | 0 / 0.65 / 0 | inside noise | 0.00 | 0 0 0 | no change |
| doc-code worst F1 | −2.98 | +0.89 −8.02 −1.80 | inside noise | +1.65 | −1.89 +10.00 −3.17 | inside noise |
| doc-code harmonic F1 | −6.54 | −1.09 −16.04 −2.49 | WEAK 3/3 | +0.19 | −0.99 +2.46 −0.89 | inside noise |

Averages (doc-model P / R / F1). terra: control 0.912 / 0.934 / 0.923, nomention
0.906 / 0.920 / 0.912. luna: control 0.863 / 0.946 / 0.896, nomention
0.876 / 0.934 / 0.898. Doc-model recall falls on both backends, and in 2 of 3 runs on
luna. Luna's precision gain cancels its recall loss in F1.

The terra run-2 CMR and harmonic-F1 swing is one component. Without the sentence,
bigbluebutton's `Presentation Conversion` loses both of its gold links (S80
"Presentation conversion flow.", S81). Both are bare whole-name mentions.

### Final link counts (summed over 3 runs × 5 projects, `mcreplay_summary_20261002.csv`)

| knowledge | backend | control TP / FP | nomention TP / FP | Δ TP | Δ FP |
|---|---|---|---|---|---|
| full | terra | 527 / 69 | 514 / 72 | −13 | +3 |
| full | luna | 539 / 143 | 531 / 129 | −8 | −14 |
| none | terra | 485 / 43 | 473 / 40 | −12 | −3 |
| none | luna | 454 / 72 | 449 / 47 | −5 | −25 |

### RQ3 (`rq34/mcreplay_20261002/*/rq3_validators.csv`, average of 3 runs)

Name judge: rejected TP terra 19.7 → 24.0, luna 15.3 → 18.7. Kept FP terra
21.0 → 22.0, luna 45.0 → 40.3. Coreference rows are identical by construction. The
macro-F1 of the Full arm drops terra 0.922 → 0.912 and luna 0.896 → 0.897. The
NoNameValid and NoValidator rows are unchanged because they never use the judge.

### RQ4 (`rq4_linkers.csv`, the knowledge row from the `*_noknow_*` dirs)

Name linker, TP caught: terra 160.3 → 156.0, luna 164.7 → 161.3. Unique TP: terra
141.3 → 137.0, luna 129.7 → 127.0. ΔF1-if-removed: terra +0.647 → +0.637, luna
+0.524 → +0.526. Coref row: unchanged up to overlap shifts.

No-knowledge Full macro-F1: terra 0.867 → 0.864, luna 0.835 → 0.841. The knowledge
gain (Full − no-knowledge) therefore goes terra +5.6 → +4.8 points and luna
+6.1 → +5.7 points.

### Where the verdicts flip (`mcreplay_flip_breakdown_20261002.txt`, name-judge level)

These are name-judge verdict flips summed over 30 project-runs per knowledge setting.
With knowledge, on the whole-name row, the arm loses 22 gold and 7 non-gold
approvals and gains 5 gold and 11 non-gold. On the word-only row it loses 9 gold and
19 non-gold. On the alias row it loses 3 gold and 14 non-gold. The no-knowledge runs
show the same pattern: whole-name loses 20 gold and 9 non-gold. Some name-stage
losses are rescued at the final level by the coreference linker, which is why the
final ΔTP is smaller than the name-stage loss.

## Reading

- Measured: deleting `MENTION_COUNTS` lowers recall on both backends and in both
  knowledge settings. On terra, every RQ1 headline metric is worse in 3 of 3 paired
  runs (about −1 point doc-model F1, about −2 points doc-code F1). On luna, the recall
  loss comes with a precision gain, and every RQ1/RQ2 delta is inside noise.
- Interpretation: on the whole-name row, the sentence keeps gold bare mentions that the
  judge otherwise rejects for making no further claim. That matches its stated purpose
  and the earlier paraphrase result (`labelrule_round`, `v14mention`: gold −5.3 a run).
  Its precision cost is concentrated on luna and on the alias and word-only rows.
- Limits: N = 3 per backend. One stage was re-sampled while the other stages were held
  at one recorded sample. The deltas are against an in-set control and are not
  comparable to the paper's committed s126 numbers, which come from a different
  invocation set.
