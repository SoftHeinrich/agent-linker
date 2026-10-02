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

---

# Round 2: what the diff set shows, and three reference-criterion replacements

## The diff set (`diff_set_20261002.tsv`, made by `approach/pilot/mention_counts_diff.py`)

Over 60 project-runs, the control and nomention judges disagree on 225 verdicts,
covering 94 distinct (sentence, component) pairs. The most stable flips, those lost in
6–8 of the 12 cells, fall into two groups:

1. **Gold links lost: bare references.** These are sentences that only name the
   component: a title, a list entry, a sentence whose subject is the component but
   whose predicate is about something else. bigbluebutton S61 "FreeSWITCH.", S67
   "Kurento and WebRTC-SFU." and S80 "Presentation conversion flow." are examples, as
   is teammates S168 "This component automates the testing of TEAMMATES." Under
   control, the judge's quoted claim is the reference itself. Under nomention it is
   `none`. The cause is that `_DEFINITION` asks for "an architectural claim" and
   `UNION_DEMAND` asks for "the words that state the architectural claim". A bare
   reference predicates nothing, so without `MENTION_COUNTS` the judge has nothing
   it may quote.
2. **Non-gold links lost: one-word coincidences.** Examples are bigbluebutton
   S82/S83/S84 "…SVG conversion flow", "the conversion fallback" →
   `Presentation Conversion`, and teammates S13/S14/S17 "…testing…" → `Test Driver`.
   Under control the judge approves these because "a mention … still counts": it
   reads an occurrence of one word of the name as a mention of the component.

Interpretation: `MENTION_COUNTS` is an exception attached to a predication-based
definition, and its word "mention" does not separate *referring to the component*
from *a surface of its name occurring*. That is the use–mention distinction
`SURFACE_NOT_EVIDENCE` already draws for the shorter rows. The deletion removes the
exception's benefit (group 1) together with its cost (group 2), and that is why
recall falls while luna's precision rises.

## Arms (`approach/pilot/mention_counts_replay.py`, `RULES`)

| arm | change to the union prompt |
|---|---|
| `corollary` | `MENTION_COUNTS` → "Referring to the component as a participant is itself such a claim, even when the sentence says nothing further about it." |
| `refdef` | `_DEFINITION + MENTION_COUNTS` → "A trace link holds … when an expression in the sentence refers to that component as a participant in the system this document describes, whether or not the sentence says anything further about it." |
| `refdemand` | `refdef`, plus `UNION_DEMAND` asks for "the expression that refers to the component, together with what the sentence says of it if it says anything, or "none" if no expression in the sentence refers to the component" |

Each arm states one general distinction, reference versus predication, and names no
surface form, document shape or component.

```bash
OPENAI_REASONING_EFFORT=none OPENAI_SERVICE_TIER=default LLM_BACKEND=openai \
  python3 approach/pilot/mention_counts_replay.py --stamp 20261002b \
  --arms corollary refdef refdemand --workers 12          # 180 replays
# then the same extracts -> build_dump -> rq12 --arm mc<arm> -> rq34/rq34_rq2 loop as
# round 1 with stamp 20261002b and --csv-root evaluation/reports/rq34/mcreplay_20261002b/
python3 studies/compare_arms.py mc<arm> --base mccontrol \
  --csv evaluation/reports/ARM_COMPARE_mc<arm>_vs_mccontrol.csv
python3 approach/pilot/mention_counts_diff.py --stamp 20261002 --arms control corollary \
  --cand-stamp 20261002b > results/mention_counts_round/diff_set_corollary_20261002b.tsv
```

**Control reuse.** The control is round 1's `mccontrol`, run a few hours earlier on
2026-10-02 over the same recorded inputs (the project's 3-day control-reuse rule).
These arms therefore ran in a *different invocation* from their control. Same-day API
drift is not excluded.

## Results (N = 3 runs per backend; points vs `mccontrol`)

| metric | corollary terra | corollary luna | refdef terra | refdef luna | refdemand terra | refdemand luna |
|---|---|---|---|---|---|---|
| dm F1 | **+1.34 BETTER 3/3** | +0.41 noise | −1.18 WORSE | +0.62 noise | −1.20 WORSE | +1.72 BETTER |
| dm F2 | **+1.86 BETTER 3/3** | −0.37 noise | −0.71 WORSE | −0.10 noise | −1.14 WORSE | +0.70 noise |
| dc F1 | −0.39 noise | +0.25 noise | −3.08 WORSE | −0.36 noise | −2.83 WORSE | +0.66 noise |
| dc F2 | +0.39 noise | −0.40 noise | −1.76 WORSE | −0.81 WORSE | −1.77 WORSE | −0.23 noise |
| dc worst F1 | **+3.56 BETTER** | +1.58 WEAK | −1.49 WORSE | −1.59 noise | −2.12 WORSE | +4.22 BETTER |
| dc harm F1 | **+1.93 BETTER** | +0.55 BETTER | −0.53 noise | +0.08 noise | −0.89 WORSE | +1.34 BETTER |

Final links (3 runs × 5 projects, with knowledge), Δ TP / Δ FP vs control:
corollary terra +19 / −1, luna −5 / −18; refdef terra −3 / +9, luna −4 / −32;
refdemand terra −9 / −2, luna −2 / −45.

RQ4 no-knowledge Full macro-F1 (control → corollary): terra 0.867 → 0.868, luna
0.835 → 0.827 (per run 0.821/0.848/0.837 → 0.817/0.826/0.838). Corollary's knowledge
gain is therefore terra +5.6 → +6.8 points and luna +6.1 → +7.3 points.

Corollary diff set (`diff_set_corollary_20261002b.tsv`). With knowledge, on the
whole-name row it gains 31 gold approvals and loses 11. That includes the teammates S1
enumeration ("Architecture contains UI Component, Logic Component, …"), gained in 3
cells, which `results/s121_ablations` recorded as the sentence the old exception and
an unscoped weighing fought over. It still sheds some one-word coincidences
(bigbluebutton S82–S84). Without knowledge it loses 21 gold word-only approvals
against 4 gained, and that is the luna no-knowledge regression.

## Reading

- Measured: `corollary` is the only arm that is not worse than control on any RQ1/RQ2
  headline metric on either backend with knowledge. On terra it is better 3/3 on
  doc-model F1/F2 and on both size-aware doc-code metrics. On luna it is inside noise,
  except harmonic F1, which is better 3/3. `refdef` and `refdemand` replace the
  definition itself, and both are worse 3/3 on terra on every headline metric.
- Interpretation: keeping the predication definition and stating the reference as a
  *consequence* of it ("referring to a participant is itself such a claim") holds the
  link on bare references without "mention" licensing a coincident word.
  Rewriting the definition into a pure reference criterion lowers the bar on terra
  more than the judge can safely use. This is the same direction as `v14mention`:
  the precise register of this one clause moves the result.
- Open: corollary costs one-word recall on luna without knowledge (RQ4 no-knowledge
  F1 −0.8, 2 of 3 runs lower). The evidence is a stage replay with the other stages
  held at one recorded sample, read against a control from an earlier same-day
  invocation, N = 3. Promoting it to `s_linker126` would need a paired end-to-end run
  with an in-set control before any paper number moves.

---

# Round 3: "as a participant" vs "as an architectural participant"

Question: can `corollary` say "architectural participant" (the term
`COREF_VALIDATION_FOCUS` already uses) without losing performance?

| arm | sentence replacing `MENTION_COUNTS` |
|---|---|
| `corollary` (repeat) | "Referring to the component as a participant is itself such a claim, even when the sentence says nothing further about it." |
| `corollaryarch` | "Referring to the component as an architectural participant is itself such a claim, even when the sentence says nothing further about it." |

Both arms ran in one invocation (`--stamp 20261002c`, 120 replays), so the
arch-vs-plain read is paired. Each is also read against round 1's `mccontrol`
(not re-run, under the 3-day reuse rule). Slots are `mccorollaryc` and
`mccorollaryarch`. RQ3/RQ4 output is in `evaluation/reports/rq34/mcreplay_20261002c/`,
and every `rq34.py` call reported `validate=OK`.

```bash
OPENAI_REASONING_EFFORT=none OPENAI_SERVICE_TIER=default LLM_BACKEND=openai \
  python3 approach/pilot/mention_counts_replay.py --stamp 20261002c \
  --arms corollary corollaryarch --workers 12
python3 studies/compare_arms.py mccorollaryarch --base mccorollaryc \
  --csv evaluation/reports/ARM_COMPARE_mccorollaryarch_vs_mccorollaryc.csv
```

## Results (N = 3 per backend, points)

| metric | corollary repeat vs control: terra | luna | arch vs control: terra | luna | **arch vs corollary (paired): terra** | **luna** |
|---|---|---|---|---|---|---|
| dm F1 | +0.80 BETTER | +1.54 BETTER | +0.22 BETTER | +1.26 noise | **−0.58 WORSE 3/3** | −0.28 noise |
| dm F2 | +1.62 BETTER | +0.23 noise | +1.65 BETTER | −0.45 noise | +0.03 noise | −0.68 noise |
| dc F1 | −0.28 noise | +0.58 noise | −1.18 WORSE | −0.01 noise | **−0.90 WORSE 3/3** | −0.59 noise |
| dc F2 | +0.68 BETTER | −0.69 noise | +0.71 BETTER | −1.52 WORSE | +0.02 noise | −0.83 WORSE 3/3 |
| dc worst F1 | +3.70 BETTER | +3.02 noise | −0.10 noise | +5.49 BETTER | **−3.80 WORSE 3/3** | +2.47 noise |
| dc harm F1 | +1.73 BETTER | +0.79 noise | +1.18 BETTER | +1.43 BETTER | **−0.55 WORSE 3/3** | +0.64 BETTER |

No-knowledge Full macro-F1, RQ4 (control 0.867 / 0.835):
`corollary` repeat 0.873 / 0.833, `corollaryarch` 0.867 / 0.832 (terra / luna).

## Reading

- Measured: the `corollary` result replicates. Its second sample is again better 3/3 on
  terra's doc-model F1/F2 and size-aware doc-code metrics, and luna doc-model F1 is
  better 3/3 this time. Its luna no-knowledge cost shrinks from −0.8 to −0.2 points.
  The two samples bracket the effect size; neither is the estimate.
- Measured: adding "architectural" is worse than the plain wording on terra in 3/3
  paired runs (dm F1, dc F1, worst, harmonic). On luna it is mixed (dc F2 worse 3/3,
  harmonic better 3/3). It stays above control on most terra metrics, so it is not
  harmful compared with the head, only weaker than the plain wording.
- Interpretation: this is the `v14mention` mechanism (`labelrule_round`) at smaller
  size. The adjective makes the judge test the reference for architectural
  significance, which is the bar the sentence exists to lower.
- Defensibility does not need the adjective in the prompt. "Participant" in the clause
  refers back to the definition directly before it ("an architectural claim about that
  component -- … as a participant in the system this document describes"), so the
  clause already means a participant in the described architecture. Paper prose can say
  so by quoting the definition. It should not render the clause as "architectural
  participant", because that is a different prompt and was measured here as weaker.

---

## Promotion and where the scripts went

`REFERENCE_CLAIM` (the `corollary` wording) replaced `MENTION_COUNTS` in
`s_linker126.py` on 2026-10-02. That was the author's decision after round 3. The
rule's bytes differ from the pre-promotion head by that one sentence only.

Both pilot scripts were moved out of the active pilot surface into `scripts/` here.
They replay arms against the **pre-promotion** head: `nomention` and `corollary`
substitute into `MENTION_COUNTS`, which no longer exists. To re-run them, check out
commit `20eecbb5` and run them from `approach/pilot/`, where they were written.
The paper numbers come from fresh end-to-end runs of the promoted code, not from these
replays.
