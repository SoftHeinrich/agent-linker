# What is left to simplify after s38 — 2026-08-12

s38 reached statistical parity with s25 on every measure (F1 −0.47, p = 0.071;
F2 +0.20; TP +1.17; FP +2.33; six runs each side, exact permutation over 924
splits) by replacing s25's two judging passes with one prompt sampled twice and
ANDed. This audit asks what the remaining structure is worth, entirely off s38's
own six runs. **No LLM call.** Script: `approach/pilot/s38_audit.py`.

Headline, and it runs against the audit's own suggestions: **the audit's readings
were all correct and none of them licensed a removal.** s38's stated mechanism is not
what holds its parity (A1), the mention label looks over-specified (A5) — and every
variant built on those readings loses end-to-end. Sections A1–A7 are the readings;
"What this produced" is what happened when they were acted on.

## A1 — the self-agreement gate is nearly inert

| | per run |
|---|---|
| candidates judged | 174.7 |
| both samples approve | 162.3 |
| both samples reject | 11.3 |
| **samples split** | **1.0 (0.6%)** |

Of the 1.0 split cases, 0.3 are gold and 0.7 are not. So ANDing rather than ORing
the two samples is worth **0.7 false positives against 0.3 true positives per
run** — inside noise on a pipeline whose F1 band is ±0.43. Every split case over
six runs:

```
jabref     S7   preferences     not gold   in 2/6 runs
teammates  S122 GAE Datastore   GOLD       in 1/6 runs
teammates  S131 Storage         not gold   in 1/6 runs
teammates  S141 GAE Datastore   GOLD       in 1/6 runs
teammates  S160 Common          not gold   in 1/6 runs
```

s38's docstring claimed the self-agreement gate "is where the precision comes
from". It is not. Both docstring and inline comment are corrected in place.

### …and the mechanism the audit uncovered instead

Taking `s_linker36` (the same merged prompt, asked **once**) to six runs against
s25's six settles the direction: **macro F1 −0.7 (p = 0.01), FP +3.5 (p = 0.01)**,
TP +0.8 (p = 0.44), macro F2 +0.0 (p = 1.00), at 79 calls against 89. The three
variants then line up on precision exactly as one mechanism predicts:

| | judging arrangement | FP/run | verdicts that disagree |
|---|---|---|---|
| s25 | two prompts, one focus each | **4.8** | **4.7 of 172.3 (2.7%)** — 1.0 gold, 3.7 not |
| s38 | one merged prompt, two samples | 7.2 | 1.0 of 174.7 (0.6%) — 0.3 gold |
| s36 | one merged prompt, one sample | 8.3 | — |

Both arrangements reject the same **11.3** candidates unanimously. The unanimous
rejections survive the merge; the *disagreements* do not — and the 3.7 false
positives that s25's disagreements remove are the 3.5 by which it leads s36.

**Independence has to come from asking a different question, not from resampling
the same one.** Two focuses are 4.5× more independent than two samples of one
prompt. That retires the whole s32–s38 line: the judging arrangement stays as s25
has it.

## A2 / A6 — the two criteria are one prompt but not one question

Relevance and uniqueness disagree on **3.2 of 400.0 answers per run (0.8%)**,
always in the same direction (relevant but not unique), and almost entirely on
Teammates. Joined to the gold standard: those 3.2 are **2.7 false positives and
0.5 gold**. Uniqueness therefore stays — it costs 169 bytes of prompt and buys
2.7 FP. Note that this replaces the figure the two-pass form produced (10 FP);
in the one-call form the criterion is worth less, but still positive.

Also visible: the judge writes `claim: none` on 12.3 cases per run on
BigBlueButton and approves them anyway, so claim-before-verdict is a
commit-to-text device, not a filter — which is what the s25 ablation already
implied (removing the request costs 35.2 TP; enforcing it voids 0 verdicts).

## A3 — three judging protocols, and what one protocol would cost

| stage | calls/run | protocol |
|---|---|---|
| coreference: resolve | 46.3 | single pass |
| full-name: propose | 10.3 | single pass |
| full-name: judge, sample 1 | 10.3 | two samples, ANDed |
| full-name: judge, sample 2 | 10.3 | two samples, ANDed |
| coreference: judge | 7.0 | single pass |
| knowledge: propose aliases | 5.7 | single pass |
| knowledge: judge aliases | 5.7 | single pass |
| partial-name: denote (target-blind) | 4.7 | two questions in sequence |
| partial-name: grounded identity | 1.2 | two questions in sequence |
| **total** | **101.5** | |

Making every judge sample twice costs +12.8 calls/run. Given A1 that is 12.8
calls for nothing, so uniformity would have to go the other way — one call per
judging step, which is s36's arrangement — and that loses 0.7 F1 at p = 0.01.
There is no free uniformity here in either direction.

The remaining asymmetry — the full-name gate approves by default while the
coreference gate rejects when uncertain — is *principled*, not a wart: the
full-name gate rules on a sentence that states the name, the coreference gate on
a sentence that does not. Opposite defaults follow from opposite evidence, and
the ablation prices the coreference gate's strictness at 12 FP.

## A4 — what the judges actually reject, and where

| project | full-name candidates | judge rejects (gold) | partial-name proposed → accepted | coreference reported → accepted |
|---|---|---|---|---|
| mediastore | 27.7 | **0.0** (0.0) | 0.0 → 0.0 | 10.0 → 7.7 |
| teastore | 21.7 | 0.7 (0.0) | 0.0 → 0.0 | 20.8 → 7.2 |
| teammates | 61.8 | **9.7 (1.5)** | 28.0 → **0.0** | 44.3 → 10.7 |
| bigbluebutton | 43.3 | 0.7 (0.0) | 30.7 → **11.0** | 23.3 → 4.5 |
| jabref | 20.2 | 1.3 (0.0) | 0.0 → 0.0 | 6.0 → **0.0** |

Three facts a reviewer will find on their own, so the paper should state them:

* the full-name judge rejects **12.4 of 174.7 candidates**, and **78% of those
  rejections are on one project**. On MediaStore it rejects nothing. This is why
  every judging-arrangement variant from s32 to s38 lands within noise — there is
  almost nothing to reject on four of five projects, so **the arrangement of
  judging is not what makes the approach work; admission is.** It also prices
  dropping the judge outright: TP +1.5, FP +10.9;
* the **partial-name linker yields links on one project** (BigBlueButton, 11.0).
  On Teammates it proposes 28.0 and its two-step judge accepts 0;
* **coreference yields nothing on JabRef** (6.0 resolutions, all rejected).

Remaining prompt surface: **10 rubric constants, 4022 bytes** (the largest are
COREF_RULES 760, LAYERED_COREF_RULES 723, LAYERED_ENTITY_RULES 692). One
name-matching primitive, verified by inspection of the code (`_find_exact_form`,
8 call sites; `has_standalone_mention` no longer referenced) — the s25 docstring
still claimed two and is corrected.

Deterministic proposer conditions, recomputed over all five documents:

| project | suppressed inside a qualified identifier | dropped for >1 owner | dropped because the whole name is stated | proposed (prefix-only) |
|---|---|---|---|---|
| mediastore | 63 | 0 | 17 | 0 (0) |
| teastore | 72 | 0 | 17 | 1 (1) |
| teammates | 396 | 0 | 69 | 59 (30) |
| bigbluebutton | 175 | 15 | 29 | 37 (5) |
| jabref | 15 | 0 | 21 | 0 (0) |

## A5 — the five-value mention label *looks* collapsible to three

| value | cases/run | approval | gold rate |
|---|---|---|---|
| proper case, standalone | 107.0 | 96.9% | 97.0% |
| via known alias | 33.0 | **82.8%** | 74.7% |
| lowercase mention | 25.2 | 100.0% | 100.0% |
| lowercase, inside qualified name | 7.8 | **57.4%** | 25.5% |
| indirect/unclear match | 1.7 | 100.0% | 100.0% |

Three of the five values are approved at 96.9 / 100.0 / 100.0%, which reads as a
label whose case grading changes no verdict. It does — see "What this produced".
The two values that separate are *how* the name is present: via a discovered alias
(82.8%) and only inside a qualified identifier (57.4%, gold rate 25.5%).

The change this reading suggested was a label of three values — `the name itself`,
`a name the document introduces for it`, `only inside a qualified identifier` — with
the residual value (neither matched, 1.7 cases) becoming *no field*, since "how the
name is present" has no answer when it is not present. That would also delete the
workflow's last case-sensitivity rule (`matched == comp_name`). Both the full change
and the half of it the traces support were built and measured, and both lose.

## A7 — the largest call consumer cannot be narrowed

Coreference resolution reads every sentence of every document in batches of ten:
46.3 of 101.5 calls, and 64% of what it reports is a pair an earlier linker
already produced. Asking only about sentences that carry no link yet would make
the paper's "each linker sees only what the earlier ones left unlinked" literally
true of the *input* and cut the biggest stage — and it would **lose 14.5 of the
30.0 coreference links per run, 13.2 of them gold**. A sentence can state one
component's name and refer back to another, so restricting by sentence is not a
subtraction of duplicates. The subtraction belongs at the pair level, where it
already is.

## What this produced

**`s_linker42`** = `s_linker36`'s single judging call + the three-value mention
label. Three runs a side against s36
(`results/s42_threevalue_e2e_r{1,2,3}_20260812`, which also bring s36 to n=6):
the label collapse is **free** — TP ±0.0 (p = 1.00), FP +1.7 (p = 0.30),
F1 −0.1 (p = 0.50), F2 −0.1 (p = 0.70). But it sits on a base that loses F1
significantly, so:

**`s_linker43`** = **s25 with the three-value label and nothing else** — and it is
**rejected.** Three runs a side, paired with s25 in the same invocations
(`results/s43_threevalue_e2e_r{1,2,3}_20260812`):

| | s25 | s43 | difference |
|---|---|---|---|
| TP | 180.7 | 179.0 | −1.7, p = 0.40 |
| FP | 5.7 | 8.3 | +2.7, p = 0.30 |
| macro F1 | 96.37 | 95.06 | **−1.3, p = 0.10 (the n=3 floor)** |
| macro F2 | 95.49 | 94.18 | **−1.3, p = 0.10 (the n=3 floor)** |

Both scores are the most extreme of all ten labellings, which is as strong as
three runs can show. So a label change that is *free* on the merged-judging base
(s42 vs s36: TP ±0.0, F1 −0.1) **costs 1.3 F1 on s25.** That is the sixth time in
this workflow an arm measured neutral in one composition comes out negative in
another, and the first where both compositions were end-to-end. Equal approval
rates per label value are a *screen*, not a proof: they are an aggregate over
cases, and rewriting the field changes the prompt for 132 cases per run.

`approach/pilot/test_s43_threevalue.py` asserts the single change (37 shared
methods including `_validate_with_evidence` and both judging prompts, 10 rubrics,
7 resource bounds, both deterministic generators, and every one of 3697
(name, sentence) pairs relabelling exactly as intended), so the loss is the label
and nothing else.

**`s_linker44`** splits the failed pair — and **it fails too.** s43 changed two
things at once: it merged the two stated-name values *and* omitted the field for the
residual value. Only the first is supported by the traces (96.9% vs 100.0%
approval); the second removes evidence content, which this workflow has twice
measured as pipeline-negative. s44 merges only the case grading — five values become
four, the field is always present, and `_validate_with_evidence`,
`_build_evidence_bundle`, `_format_evidence` and `_prompt_validation` are s25's byte
for byte (`approach/pilot/test_s44_nocasegrade.py` pins the difference to the enum
and the classifier over 3697 (name, sentence) pairs).

**Six runs a side, paired inside the same invocations**
(`results/s44_nocasegrade_e2e_r{1..6}_20260812`):

| | s25 | s44 | difference |
|---|---|---|---|
| TP | 181.3 | 181.7 | +0.3, p = 0.87 |
| FP | 8.8 | 10.0 | +1.2, p = 0.55 |
| macro F1 | 95.92 | 95.06 | **−0.9, p = 0.05** |
| macro F2 | 95.41 | 94.95 | −0.5, p = 0.21 |

### The n=3 result said the opposite, and that is the finding

At three runs the same comparison read **TP +2.0, FP −0.7, macro F1 −0.0 (p = 1.00),
macro F2 +0.3** — a clean neutral, with s44 holding the *tighter* within-arm spread.
Runs 4–6 put s44 at 94.5–94.9 against s25's 96.4–96.5 and the verdict inverted. This
is the same lesson as the earlier over-tight ±0.1 band, now demonstrated in the
other direction: **three runs of this pipeline can manufacture a neutral as easily
as a regression.** Six runs a side, paired, is the bar.

Where the loss lands:

| project | s25 F1 | s44 F1 | ΔF1 | mechanism |
|---|---|---|---|---|
| jabref | 99.55 | 95.61 | **−3.94** | FP 0.2 → 1.7 on a 13-sentence, 18-link document |
| teastore | 100.00 | 98.43 | −1.57 | TP 27.0 → 26.2 |
| bigbluebutton | 91.22 | 90.77 | −0.45 | |
| mediastore | 98.08 | 97.83 | −0.25 | |
| teammates | 90.78 | **92.67** | **+1.89** | FP 6.7 → 4.0 |

JabRef carries the highest share of merged-value pairs of any project (20 of 78) and
is the smallest, so a 1.5-false-positive shift moves its F1 by four points and macro
averaging weights it like everything else. Teammates, the largest, improves.

## Verdict on the judging and label axes: every suggestion fails end-to-end

The audit's readings were all correct as *readings* — the second sample really does
split on 0.6% of cases, the label's three values really are approved at 96.9 / 100.0
/ 100.0%. Neither licensed a removal. **Equal aggregate behaviour per label value
does not mean the distinction is inert**, because rewriting the field changes the
prompt for every case that carries it (132 per run for the case grading), and this
model's verdicts move with prompt text that carries no new information.

That is the seventh instance in this workflow of an arm passing a screen and failing
composition, and the first where the screen was *trace-based* rather than
stage-based. The practical rule is now: a trace-derived equivalence is a hypothesis,
and the only test is six paired runs.

What that says about s25: across twenty variants — twelve on the knowledge side,
three on judging arrangement, three on the mention label, two on the axes below —
**no element of the workflow has been removed without a measured cost**, and the one
change that holds removes no element at all. It retires a tuned constant. For the
paper that is a stronger claim than any single ablation table.

## The one that holds: `s_linker45`

Two axes were left after the judging and label attempts failed, and A3/A7 pointed at
both. The first is the **only resource bound the workflow never ablated**:
coreference resolution reads 10 sentences per call while the judges read 25 and
extraction reads 50, and that third value is what makes coreference resolution 46% of
all calls. `s_linker45` sets it to `JUDGE_BATCH` — a value chosen by *unification*, not
search, and tested once.

**Six paired runs** (`results/s4546_e2e_r{1..6}_20260812`, carrying s25, s45 and s46
in the same invocations):

| | s25 | **s45** | difference |
|---|---|---|---|
| batch constants | 3 | **2** | |
| calls / run | 88.8 | **65.3** | **−26%** |
| coreference resolve calls | 40.0 | **17.0** | |
| TP | 181.3 | **182.2** | +0.8, p = 0.56 |
| FP | 7.0 | 9.2 | +2.2, p = 0.34 |
| macro F1 | 96.11 | 95.91 | −0.2, **p = 0.52** |
| macro F2 | 95.47 | 95.44 | −0.0, **p = 0.91** |

Nothing is within reach of significance and recall is the higher of the two. No
project collapses — mediastore **+0.26** F1, jabref **+0.43**, teammates −0.26,
bigbluebutton −0.46, teastore −0.96.

`s_linker27`'s passage-length effect (F1 98.4 at 37 sentences, 79.7 at 87) does not
reach from 10 to 25 for *this* question: resolving a back-reference needs the
sentences either side of the target, not a short window. **This is the simplification
that holds** — one fewer tuned constant to justify and a quarter of the cost gone,
with the architecture, every prompt and every rubric untouched.

## The other axis: `s_linker46`, and the dual role is real

The alias table has two opposite roles — it *admits* full-name candidates (29
alias-only links, 23 gold) and *suppresses* partial-name ones, because the proposer
treats every discovered alias as a whole name. The s26 diagnosis flagged this as an
architectural liability no single-stage arm can see. s46 gives it one role. Freed
candidates, sized off six real tables first: 59 → 75 over the five projects, 3.8 gold
per run.

**Rejected:** TP −2.0 (p = 0.39), **FP +6.5 (p = 0.01)**, **macro F1 −1.5 (p = 0.00)**,
macro F2 −1.0 (p = 0.02) — and it loses at n=3 as well, so this is not a variance
artefact.

Note the direction: **freeing 16 candidates cost 2.0 true positives.** Adding
candidates cannot remove a link directly, so the loss is batch composition in the
two-step partial-name judge — the same mechanism the `_unlinked` arm measured in the
other direction (−6.8 FP purely from changing which cases share a batch). The dual
role is load-bearing *both* ways, and the paper should state it as a property of the
design: the table opens one gate and closes another, and removing the second costs
1.5 F1.

## Proposals not built, and why

* **drop `source=` from the evidence line.** It renders one value 396.0 of 400.0
  times per run — the same argument that removed the constant `Rationale:` line.
  Not attempted: two earlier removals of evidence content (`matched_span`,
  `preceding_text`) were stage-neutral and pipeline-negative (FP 8.3 against the
  4–6 band). Repeating evidence next to the rubric is not redundant for this
  model;
* **merge `LAYERED_COREF_RULES` into `LAYERED_ENTITY_RULES`** (723 + 692 bytes of
  the 4022-byte rubric surface). Not attempted: their defaults are deliberately
  opposite — approve-unless for a stated name, reject-when-uncertain for an absent
  one — and the coreference gate's strictness is priced at 12 FP;
* **drop the uniqueness criterion.** A6 prices it at 2.7 FP for 0.5 gold per run.
  Kept;
* **restrict coreference to unlinked sentences.** A7 prices it at 13.2 gold links
  per run. Kept;
* **sample every judge twice.** A1 and A3: +12.8 calls/run for what A1 shows is
  nothing.
