# The audit round — what the elegance cuts cost, put back, then what the prompt audit removes for free

`s_linker78` → `s_linker81` → `s_linker82`. Two questions, two paired invocation sets:

1. **The elegance round cut three things at once and lost 2.6 pp of macro F1 doing it.**
   Which of the three cost it? `s_linker81` puts back the two that are statements about
   *what a judge can see* and leaves the third out.
2. **What does the prompt/dead-code audit cost?** `s_linker82` applies every fix the audit
   found and is priced against `s81` in the same invocations.

Answer to both: the restoration recovers the loss in full, and the audit is free — one
judging pass instead of two, 10% fewer LLM calls, a quarter of the module gone, and no
measurable quality change.

## The reference the round is read against

`results/elegance_e2e_r{1,2,3}_20260819`, arms `s_linker78` and `s_linker80` in the same
invocation:

| arm | TP | FP | macro F1 | macro F2 |
|---|---|---|---|---|
| `s_linker78` | 184.3 | 22.0 | 93.14 | 94.43 |
| `s_linker80` | 180.7 | 32.7 | 90.59 | 92.30 |
| delta | **−3.7** (p = 0.10) | **+10.7** (p = 0.10) | **−2.6** (p = 0.10) | **−2.1** (p = 0.10) |

`score_runs.py` calls this **QUALITY-CHANGING**, and every p sits on the n=3 floor. The
three cuts, separated by replaying the arms' own recorded phase states:

| cut | what it did to the candidate set |
|---|---|
| `skip_when_named` dropped | **+140 of the 161 added candidates.** The partial-name row re-proposes the full-name linker's own pairs; the denotation judge that hears them is target-blind and single-pass. 43 of 50 partial-name FP are pairs the full-name linker never saw, 7 are pairs it **rejected**. |
| `unique_owner` dropped | +21 candidates, all on one project, but pathological: a surface two components both own, assigned to whichever the scan paired it with. Both pairs proposed, both approved, 3 of 3 runs, because the judge is never shown the target. |
| mention label dropped | **−26 of 480 full-name approvals** (18 gold), concentrated: 100% of `lowercase, inside qualified name` and 73% of `via known alias`, against 3% of `proper case, standalone`. |

## `s_linker81` — the two that name a fact the judge cannot see

    RESTORED   skip_when_named. It is not a gate on a case; it says which linker owns
               one, and no judge is shown the pipeline.
    RESTORED   the mention label at VIA_ALIAS and CODE_TOKEN only. Those two values name
               a fact absent from the sentence -- one needs the alias table, the other
               needs every occurrence tested against a dotted path.
    STILL OUT  unique_owner, and the other three mention labels, as facts the judge is
               already holding.

`results/elegance_e2e_s81_r{1,2,3}_20260819`, arms `s_linker78` and `s_linker81` in the
same invocation:

| arm | n | TP | FP | macro F1 | macro F2 | calls | F1 range |
|---|---|---|---|---|---|---|---|
| `s_linker78` | 3 | 182.7 | 24.0 | 92.64 | 93.94 | 89 | 1.06 |
| `s_linker81` | 3 | 181.7 | 28.0 | 92.69 | 93.69 | 90 | 1.12 |
| delta | | −1.0 (p = 0.80) | +4.0 (p = 0.50) | **+0.0** (p = 0.80) | −0.2 (p = 1.00) | +1 | |

**QUALITY-NEUTRAL.** Restoring two of the three cuts returns the arm to its own base; the
2.6 pp was not the price of removing structure in general, it was the price of removing
these two facts in particular. `unique_owner` stays out and nothing goes looking for it.
Composition +6.9, p = 0.10 at the floor.

## `s_linker82` — the prompt audit, priced

Every fix the audit found, applied together, because each one is either a restatement or
a dead branch and none of them is a decision rule:

| fix | evidence it rests on |
|---|---|
| **one full-name judging pass**, not two AND-ed | s81 sent the same prompt twice (P1 architectural participation, P2 referential specificity). Replaying s81's own three runs, the passes disagree on **4.0 of ~196 candidates** per five-project run, and both questions are already in the shared rubric. |
| **deduplicated coreference batch** | one sentence table per batch instead of a pasted ±5 window per case. |
| **alias judge cannot fail open** | its no-answer path was three behaviours — a parse failure approved everything, a keyless reply approved nothing. Now one documented default. |
| **extraction prompt states one admission rule** | it carried two that contradicted each other. |
| **judged claims recorded in the trace** | they were decided and discarded. |
| **dead deterministic layer removed** | nothing read it. |

`results/audit_e2e_s82_r{1,2,3}_20260820`, arms `s_linker81` and `s_linker82` in the same
invocation:

| arm | n | TP | FP | macro F1 | macro F2 | calls | F1 range | module |
|---|---|---|---|---|---|---|---|---|
| `s_linker81` | 3 | 180.3 | 33.0 | 91.46 | 92.75 | 89 | 0.86 | 96,229 B |
| `s_linker82` | 3 | 179.0 | 35.3 | 90.07 | 92.09 | 80 | 2.43 | 71,835 B |
| delta | | −1.3 (p = 0.70) | +2.3 (p = 0.70) | −1.4 (p = 0.30) | −0.7 (p = 0.50) | **−9** | | **−25%** |

**QUALITY-NEUTRAL** on all four statistics. Composition +6.4, p = 0.10, on the floor —
the arms do produce somewhat different link sets, which is expected when a judging pass
is removed, but nothing separates them on quality. The −1.4 F1 is the largest delta in
the round and is not significant at any n this design can reach; read it next to the
recorded ±55-link run-to-run swing, not as a cost.

**The one thing to watch:** `s82`'s F1 range is 2.43 against `s81`'s 0.86 in the same
invocations, and its two low runs are r1 and r2. Dropping the second judging pass removes
an AND, so a single judge's variance is no longer damped by a second one. If a later
round needs a tighter arm, that is where the spread now lives.

## Reproducing

Both invocation sets, five projects, `gpt-5.6-terra` at `OPENAI_REASONING_EFFORT=none`:

```bash
cd approach
OPENAI_API_KEY="$OAI_KEY" \
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_REASONING_EFFORT=none \
PHASE_CACHE_DIR=../results/<run>/phase_states \
LLM_LOG_DIR=../results/<run>/llm_logs \
  ../.venv/bin/python run_ablation.py \
  --variants s_linker82 s_linker81 \
  --datasets mediastore teammates teastore bigbluebutton jabref \
  --results-dir ../results/<run>
```

Scoring and the paired permutation test:

```bash
../.venv/bin/python pilot/score_runs.py \
    --arm s_linker81 ../results/audit_e2e_s82_r*_20260820 \
    --arm s_linker82 ../results/audit_e2e_s82_r*_20260820

../.venv/bin/python pilot/finetune_e2e_table.py \
    ../results/audit_e2e_s82_r*_20260820 --control s_linker81
```

## Invariants

- **Arms are compared only inside the invocation that produced them.** The s78/s80 table,
  the s78/s81 table and the s81/s82 table come from three different invocation sets and
  their absolute numbers are not comparable across the three; only the deltas are.
- **No null arm.** The harness floor is measured (six rounds; see
  `results/finetune_round/README.md` for the one loud reading) and the measurement policy
  says not to spend an arm re-measuring a constant.
- **The p floor at n=3 is 0.10.** With 3 runs a side there are C(6,3)/2 = 10 distinct
  labellings, so an effect can be the most extreme of all of them and still read 0.10.
  Every p in this file is reported against that floor.
