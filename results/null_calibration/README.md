# What this harness reports when nothing changed — 2026-08-14

Every conclusion in this branch — twenty-odd variants, seven "held" and thirteen
"failed" — rests on the same procedure: run two arms in the same invocation, six
times, permutation-test the pooled link sets. The procedure has never been run
against a **null**: two arms that are the same code.

`s_linker49_null` is `s_linker49` with one difference, the checkpoint and log
namespace (`_VARIANT_NAME`). All ten rule constants, all 52 method bodies and all
seven resource bounds are byte-identical — asserted by
`approach/pilot/test_s54_s55_prompts.py`. Six paired runs, both arms in the same
invocation, `s_linker49` first: `results/null_e2e_r{1..6}_20260813`.

## The null is not zero

| | s_linker49 | s_linker49_null | delta | p |
|---|---|---|---|---|
| TP | 189.5 | 184.7 | **-4.8** | **0.00** |
| FP | 11.7 | 10.0 | -1.7 | 0.42 |
| macro F1 | 97.02 | 96.36 | **-0.7** | **0.03** |
| macro F2 | 97.49 | 96.25 | **-1.2** | **0.00** |
| composition | | | +3.1 | **0.01** |

`score_runs.py` calls this **QUALITY-CHANGING**. The two arms are the same program.

The sign is the same in **6 of 6 runs** (TP 189/190/190/189/190/189 against
184/186/184/184/185/185), which looks systematic — the reversed-order set below shows
it is not arm position, and there is no code path by which it could be the variant.

Per source, the split is informative:

| source | TP delta | p |
|---|---|---|
| full_name | -2.7 | **0.01** |
| partial_name | -2.0 | 0.18 |
| coreference | -0.2 | 1.00 |

and per project it is almost all teammates, the 198-sentence document whose extraction
runs in four batches (TP 54.7 → 51.0 over the first three runs).

## What follows for every number in this branch

1. **A |macro F1| difference of 0.7 and a TP difference of 5 are inside what this
   harness produces from nothing.** Any arm judged on a delta of that size — in
   either direction — was judged on the harness.
2. **p values from the pooled permutation test are anti-conservative here.** The test
   assumes arms are exchangeable; two arms of the same code are not exchangeable in
   this harness, so its null distribution is too narrow.
3. **The composition statistic is affected too** (+3.1, p = 0.01 between identical
   programs), which is the statistic this branch used to certify `s_linker48` as
   behaviour-preserving.

Corrected readings for this session's arms, subtracting the null offset:

| arm | raw macro F1 vs s49 | minus the null's -0.7 | reading |
|---|---|---|---|
| s50 (coreference rule general) | -0.2 | **+0.5** | at or above parity |
| s51 (all nine general) | -2.4 | -1.7 | real loss |
| s52 (knowledge reverted) | -2.1 | -1.4 | real loss |
| s53 (grouping clause back) | -2.5 | -1.8 | real loss |

The subtraction is a first-order correction, not a fix: it assumes the effect is
additive and depends only on being second. `s_linker51` was in position 3 in round 1
and position 2 in round 2 and read -2.14 and -2.35, so position beyond the first does
not seem to matter much — but that is two points.

## The arms send the same bytes

`stage_diff.py --prompt-identity` compares the first call of each phase on each
project across all six runs:

    doc_extract          30 of 30 byte-identical
    coreference          30 of 30 byte-identical
    doc_judge            14 of 30
    full_name_extract    14 of 30
    full_name_p1         12 of 30

The two phases whose input is fixed by the document and the component list alone are
identical every time; everything downstream diverges because it reads an earlier
phase's *response*. So the divergence begins at the first LLM call of the run, with
byte-identical prompts, and the file-level diff confirms there is nothing else to
find: `sed 's/SLinker49Null/SLinker49/' s_linker49_null.py | diff - s_linker49.py` is
empty.

## It is not arm order

`results/nullrev_e2e_r{1..6}_20260813` runs the same pair with the order **reversed**,
`s_linker49_null` first:

| | s_linker49_null (first) | s_linker49 (second) | delta | p |
|---|---|---|---|---|
| TP | 186.3 | 188.2 | +1.8 | 0.09 |
| FP | 20.2 | 16.8 | -3.3 | 0.40 |
| macro F1 | 94.91 | 95.67 | +0.8 | 0.23 |
| macro F2 | 96.00 | 96.60 | +0.6 | 0.12 |

`s_linker49` leads in **both** orders — +0.7 F1 running first, +0.8 F1 running second
— so the effect does not follow position. And it cannot follow the code: the only
difference is `_VARIANT_NAME`, which reaches nothing but the checkpoint directory and
two log filenames (`grep _VARIANT_NAME` gives three call sites, all `os.path.join`),
and the document-determined prompts are byte-identical.

The remaining reading is the honest one: **this is the pipeline's own
nondeterminism**, and six paired runs are not enough to average it out. Sampling is
not pinned — `OPENAI_REASONING_EFFORT` is set, so `llm_client` sends
`reasoning_effort` and omits temperature, and `seed=42` is best-effort — so two runs
of one program are two draws, and a six-run mean of draws this wide lands 0.7 F1
apart often enough that a permutation test over 924 splits calls it significant.

The second ordering is also the better evidence for the size of the problem: absolute
levels drift between invocation sets (s49's FP mean reads 11.7, 12.5, 10.7, 14.5 and
16.8 across the five sets run in one day, and a single run hit 33). Only
within-invocation pairing means anything at all here, and even that has this floor.

## Standing rules this adds

1. **Every A/B in this workflow needs its null.** Six paired runs of identical code
   cost one invocation set and are the only thing that says what a p value means here.
   Report an arm's delta *against the null's delta*, not against zero.
2. **Treat |macro F1| ≤ 0.7 and |TP| ≤ 5 as indistinguishable from the harness**, in
   either direction, at six paired runs.
3. **For a change confined to one stage, test that stage.** `pilot/source_stats.py`
   restricts the same permutation test to the links one linker produced. It is what
   showed `s_linker50`'s headline "TP -3.0, p = 0.01" to be assembled out of the
   full-name and partial-name stages, which its change cannot reach; on coreference,
   the only stage it touches, it reads TP +0.2 (p = 1.00).
4. **Never compare across invocation sets.** Absolute levels drift by more than any
   effect this series has measured.
