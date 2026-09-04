# The claim-contiguity round — one unscored line, priced on both models

`s_linker110`'s partial-name denotation prompt ends with a line no named rule constant
holds, which is why `pilot/prompt_defensibility.py` has never scored it:

    Claim must be a contiguous exact substring of the source sentence.

66 B, rendered once per denotation call, 4 calls a five-project run — 264 B a run, the
cheapest removable span anyone has costed on this branch (`s_linker89`'s adopted cut was
12 961 B a run, 49x larger). It has been in that prompt since `s_linker25` and no arm has
ever priced it **there**.

**Verdict: REFUSED. The head keeps the line.** Terra alone would have adopted it.

Arm: `approach/pilot/claim_contiguity_pilot.py`. 96 calls bought the round.

## Why it was not already answered

The branch's one measurement of this clause is `pilot/design_pilots.py`'s `ClaimChecked`
(`--pilot claim`), which *added* it — with an enforcing substring check — to
`_prompt_validation`, the **full-name/coreference** gate: TP +/-0.0 (p = 1.00),
**FP +1.6 (p = 0.02)**, and the check itself voided 0 verdicts in 25 project-runs. That
result is at a gate this line does not sit in, and the `-35.2 TP` quoted in
`s_linker110.py:1355` is from the same batch's `no_claim_request` arm — also
`_prompt_validation`, also not this gate. The denotation judge builds its own prompt in
`_classify_denotations` and was in neither arm.

## Level 1 — what the checkpoints settle, and where they stop

Off three recorded terra runs x five projects (`consolidation_e2e_terra_r*_20260825`),
238 partial-name decisions, no calls:

| | |
|---|---|
| decisions | 238 (80 `participant`, 158 `associated`) |
| empty claim | **0** |
| `participant` + empty claim (link lost to `evidence_valid`) | **0** |
| claims that are contiguous exact substrings | **236 / 238 (99.2%)** |
| claims that are not | 2 — bigbluebutton S49, `'recorded events'` welded across `"recorded, all events"`, both `associated` rejections |

So the parse-side gate is inert (`valid = ... and bool(claim)` never fires) and nothing
outside the linker reads the field — not `linker_infra`, not `run_ablation.py`, not
`evaluation/`, not `pilot/score_runs.py`. Deleting the line also does not move the
GATE-07 score, because that audit walks module-level rule constants and this is an
inline f-string.

Level 1 cannot go further, for the reason `pilot/prompt_audit.py` states: **a prohibition
has its effect through absence.** 236/238 compliance is equally consistent with "the rule
works" and "the rule is unnecessary". The counterfactual is a different prompt.

## The arm

`NoContiguityLine` is the control with a **post-processor on `_ask`**, not a re-declared
`_classify_denotations`. The measurement policy requires asserting that a re-declared
builder renders byte-identically to the variant's own; deleting the line from the
control's rendered prompt makes that hold by construction, and the arm raises if the line
is not present exactly once. Verified: byte delta **67** (the line plus its newline), and
`control.replace(LINE + "\n", "", 1) == arm`.

Candidates come from `s_linker109`'s deterministic scan with a recorded run's alias table
pinned, so the candidate set is **81.0 in both arms in every sample**. Only teammates and
bigbluebutton propose partial-name candidates at all; mediastore, teastore and jabref
propose 0, so the two-project run is the same 4 calls a five-project run makes.

## Level 2 — six samples a side, both arms in one invocation per model

**terra** (`gpt-5.6-terra`, flex, reasoning none):

| arm | candidates | kept | TP | FP | precision |
|---|---|---|---|---|---|
| `control` | 81.0 | 29.8 | 21.0 | 8.8 | 0.704 |
| `nocontig` | 81.0 | 25.3 | **21.0** | **4.3** | **0.829** |

```
gold      control [21,21,21,21,21,21] -> nocontig [21,21,21,21,21,21]   delta +0.0  p = 1.00
spurious  control [ 9, 9, 9, 9, 8, 9] -> nocontig [ 4, 4, 4, 4, 6, 4]   delta -4.5  p = 0.03
kept      control [30,30,30,30,29,30] -> nocontig [25,25,25,25,27,25]   delta -4.5  p = 0.03
```

Recall is not merely neutral, it is **identical** — 21 gold in all twelve samples of both
arms. An n=3 first look read TP +0.7; six samples pin it at zero, which is the branch's
own "six paired runs is the bar" rule earning its keep again.

**luna** (`gpt-5.6-luna`, same form, own invocation):

| arm | candidates | kept | TP | FP | precision |
|---|---|---|---|---|---|
| `control` | 81.0 | 32.3 | 21.8 | 10.5 | 0.675 |
| `nocontig` | 81.0 | 35.7 | 22.5 | **13.2** | **0.631** |

```
gold      control [22,22,22,21,22,22] -> nocontig [19,22,24,24,21,25]   delta +0.7  p = 0.62
spurious  control [13,10, 9,12,12, 7] -> nocontig [10,13,16, 9,16,15]   delta +2.7  p = 0.25
kept      control [35,32,31,33,34,29] -> nocontig [29,35,40,33,37,40]   delta +3.3  p = 0.31
```

## What decides it is the variance, not the means

| model / arm | gold mean | gold sd | gold range | kept sd | kept range |
|---|---|---|---|---|---|
| terra / control | 21.0 | 0.00 | 0 | 0.41 | 1 |
| terra / `nocontig` | 21.0 | 0.00 | 0 | 0.82 | 2 |
| luna / control | 21.8 | 0.41 | 1 | 2.16 | 6 |
| luna / **`nocontig`** | 22.5 | **2.26** | **6** | **4.27** | **11** |

Luna's control is nearly deterministic; take the line away and gold swings 19-25 and kept
29-40. Luna's p = 0.25 on FP is not a weak effect, it is an effect drowned in variance the
arm itself introduced. **The clause is a format constraint, and on the laxer model a
format constraint is what holds the reply stable** — 5.5x the gold spread, 2x the kept
spread, from deleting one sentence about quoting.

## Fourth instance of one asymmetry

Every cut on this branch that reads clean on terra and costs luna precision has been a
weakening of a judge's framing: the typed coreference rubric (FP +34.0), the same rubric
with its default restated (+34.0), `COREF_VALIDATION_FOCUS` deleted (+6.3), and now this
(+2.7). **Terra holds the discipline without being told; luna needs it written down.**

## The clause still fails its own stated purpose

| | claims | non-contiguous |
|---|---|---|
| terra control | 486 | **6 (1.2%)** — bigbluebutton S49 x6, `'recorded events'` |
| terra `nocontig` | 486 | **0** |
| luna control | 486 | 0 |
| luna `nocontig` | 486 | 1 — bigbluebutton S20, `'BigBlueButton 2.3'` |

Seven violations in 1944 claims, and on terra **all of them are in the arm that carries
the instruction**. Its measurable effect is on which verdicts get written, not on whether
quotes are contiguous — the general round's `X.Y or X.Y.Z` lesson at a second clause: *a
clause's measurable effect is not the effect it states.*

## Level 3 — terra's gain is real, and no E2E was owed

Dropped by the arm in all six terra samples: 4 pairs, all bigbluebutton, all FP.

```
FP  S27 'Redis PubSub'   coref-proposes=False   in-final=True
FP  S27 'Redis DB'       coref-proposes=False   in-final=True
FP  S31 'Redis PubSub'   coref-proposes=False   in-final=True
FP  S31 'Redis DB'       coref-proposes=False   in-final=True
=> re-proposable downstream: 0 of 4
```

Zero are proposed by the recorded coreference stage, so `_unlinked` frees nothing
re-proposable, and all four sit in the recorded pipeline's **final** link set today. By
the measurement policy's level 3 the stage arm **is** the pipeline answer on terra, so the
-4.5 FP would have reached the output and an E2E would only have measured model drift.
teammates drops nothing at all; its 3 gold links are untouched in every sample.

Both S27 and S31 are the one word `redis` claimed simultaneously by `Redis PubSub` **and**
`Redis DB` — the sibling-confusion class the error analysis puts at 68% of FPs. The
control's judge calls that word a `participant` twice per sentence; the arm's does not.

## Why it is refused anyway

A cut that costs the second model precision and quintuples its recall variance is not a
defensible removal of a 66 B line. The finetune round's rule applies as written: **an
unnecessary change is not a defensible one.** The head keeps the line, now with a
measurement behind it instead of inheritance from `s_linker25`.

## Reproduce

```bash
cd approach
OPENAI_API_KEY="$OAI_KEY" LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_SERVICE_TIER=flex OPENAI_ENFORCE_FLEX=1 OPENAI_REASONING_EFFORT=none \
  ../.venv/bin/python pilot/claim_contiguity_pilot.py --samples 6
# ... and again with OPENAI_MODEL_NAME=gpt-5.6-luna
```

Per-sample series above are the tool's own output. `--dump` writes the kept pairs and
every claim with its contiguity flag, which is what the claim-shape table is counted from.
