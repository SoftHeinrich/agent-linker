# The bind round — can the remaining rules be stated in a prompt instead of coded?

`pilot/rule_audit.py` answered *how many* rules the workflow has: one surface-realization
relation at four settings, a mention label, and four structural predicates. This round
asks the question a reviewer asks next, and the one the paper needs an answer to:

> **the extraction call already reads every sentence, and the judging call is already
> shown the sentence a label was computed from — so could the prompts do this work?**

Every arm here is a **relocation**, not a removal: the rule leaves the code and its
content enters a prompt, in the register of the prompt it joins. Each pilot therefore
carries three arms wherever the rule is worth something — the linker as it stands, the
rule deleted with no compensation, and the rule deleted with its content stated in the
prompt — so "the binding works" can be told apart from "the rule was never worth
anything".

Reproduce:

    cd approach
    ../.venv/bin/python pilot/bind_audit.py                       # deterministic, no calls
    AB_RUNS=5 ../.venv/bin/python pilot/bind_pilots.py --pilot all
    ../.venv/bin/python pilot/test_s66_s67_bindcontract.py        # 56/56
    OAI_KEY=... bash pilot/run_s6667_e2e.sh

Files: `audit_s64.txt` (2 runs, the arm carrying every scan), `audit_s62_6runs.txt`
(6 runs, the stable read), `pilots.log`, `pilots_bindboth.log`, `bind*.json`.

## The answer in one table

| rule | where it could bind | deleted, no compensation | **relocated into the prompt** |
|---|---|---|---|
| `_keep_stated_names` (admission contract) | extractor | TP +4.8, **FP +10.6** (p = 0.01 both) | **TP -1.4 (p = 0.21), FP -1.8 (p = 0.47)** |
| `SCANS[stated_name]` + `SCANS[spelling]` | extractor | **TP -3.6 (p = 0.01)**, FP +1.0 | **TP -3.8 (p = 0.01)**, FP +3.2 (p = 0.11) |
| both of the above together | extractor | — | **TP -1.2 (p = 0.14), FP -1.2 (p = 0.37)** |
| `_classify_mention_typed` (mention label) | judge | **TP -8.4 (p = 0.01)**, FP -1.0 | asked of the judge: TP -3.6 (p = 0.02), FP +2.4 (p = 0.03) · asked + alias table: **TP +4.0, FP +5.0** (p = 0.01 both) |
| `SCANS[name_word]` (partial-name proposer) | extractor | — | **TP -12.4 (p = 0.01)**, FP +1.0 (p = 0.83) |
| `_iter_batches`, `_window`, `_unlinked`, `_union` | — | — | **no prompt form** (see B5) |

Five samples a side, five projects, exact permutation test on the stage's own output
against fixed recorded inputs.

**One rule group reaches parity: the extractor-side one.** The admission contract binds
on its own, and the contract *plus* the two tight scans binds together — three
deterministic rules become two sentences of English. Nothing else does: the loosest scan
loses two thirds of its true positives to an extractor asked for the same pairs, and the
mention label can only be recovered by also handing the judge the alias table, which
buys +4.0 TP for +5.0 FP.

## The deterministic audit, before any call was paid for

`pilot/bind_audit.py`. **B0** is the self-check that licenses the rest: each run's
full-name candidate set is rebuilt from the extraction call's own response plus the
scans and compared with the checkpoint — **0 extra, 0 missing over 30 project-runs**.

### B1 — where each predicate could bind

18 predicates: 5 extractor-bindable, 4 judge-bindable, 2 both, **4 with no prompt form**.
A rule that only *proposes* can be a question to the extractor; a rule that only *labels*
can be a question to the judge, because the judge is already shown the sentence the label
was computed from; structure is not a statement about text.

### B2 — the binding gap (6 runs)

What a prompt-bound extractor would have to newly produce:

| scan | pairs/run | gold/pair | the extractor did not propose | gold |
|---|---|---|---|---|
| `stated_name` | 112.0 | 0.96 | **1.0** | 0.5 |
| `spelling` | 6.0 | 1.00 | **0.0** | 0.0 |
| `name_word` | 61.0 | 0.32 | **53.8** | 15.8 |

The gap is concentrated exactly where the partial-name linker runs (teammates 27.8,
bigbluebutton 26.0; zero on the other three).

### B3 — the admission filter is a router, not a gate

Per five-project run: 205.0 extractor proposals, the filter keeps 180.2 (88%) and drops
24.8, of which 8.0 are gold — and **7.7 of those 8.0 are linked anyway** by the
partial-name or coreference linker. Net recall cost **0.3 gold pairs per run**. What the
filter buys is precision through `_unlinked`: a pair admitted early is locked into the
union *and* taken away from the later, stricter judges.

### B4 — the label is a precomputation, except for one value

Judged full-name cases per run, by label:

| label | cases | approved | gold | approval |
|---|---|---|---|---|
| proper case, standalone | 108.0 | 103.0 | 104.0 | 95.4% |
| lowercase mention | 26.5 | 26.2 | 26.0 | 98.7% |
| lowercase, inside qualified name | 7.7 | 5.3 | 2.0 | 69.6% |
| via known alias | 39.0 | 28.3 | 23.2 | 72.6% |
| indirect/unclear match | 1.3 | 1.3 | 1.3 | 100% |

**143.5 of 182.5 labels are computable from the sentence the judge is already shown.**
The other 39.0 are `via known alias`, and the judging prompt is **the only prompt in the
workflow that never receives the alias table** — so this binding is not judge-side, it is
judge-side *and* extractor-side. That is what the third arm of `bindlabel` tests, and it
is why the first two lose recall.

The label also carries the workflow's two priced defects: the `"" in "-_"` reading
suppresses 41 candidate spans in the two qualified-skipping scans, and the lowercase
search in `_all_occurrences_in_qualified_path` moves 3 of 28 `CODE_TOKEN` labels. A judge
asked the question directly carries neither.

### B5 — what has no prompt form

`_iter_batches` (priced by `s_linker27`: one call for the whole document is macro F1
91.70, and accuracy tracks document length), `_window`, `_unlinked` (the mechanism behind
this branch's stage-vs-pipeline reversals), `_union`. **The floor on "how many
hand-written rules does this workflow have" is above zero, and the floor is structural.**

### B6 — composition risk, and a mechanism the round did not go looking for

| relocation, uncompensated | candidates moved | in the final link set | also proposed later | risk |
|---|---|---|---|---|
| `bindscans` removes | 2.0 | 2.0 | 0.0 | 2.0 |
| `bindcontract` adds | 21.5 | 8.0 | 11.5 | 19.5 |
| `bindpartial` removes | 60.0 | 27.5 | 5.5 | 33.0 |

All three are non-zero, so `composition_check.py`'s precondition says the stage arms
screen and an end-to-end run decides. Splitting the 2.0 by row is the finding:

    stated_name   1.0 pair/run, 0.5 gold — the extractor never proposed it
    spelling      1.0 pair/run, 1.0 gold — the extractor DID propose it, and the
                  admission filter would have dropped it

**The spelling row is not a proposer; it is a widening of the admission filter.** Its
surfaces (`X Y` for a component spelled `XY`) do not write the name at `ANY_CASE`, so
`_states_a_name` rejects the extractor's own proposal and only the scan puts it back.
That is why the two relocations are not separable, and why `bindboth` reads better than
`bindscans`: once the contract is in the prompt, the row it was widening is gone.

## What the arms measured

* **`bindcontract` — binds.** Deleting the filter with no compensation is TP +4.8 /
  **FP +10.6**, both p = 0.01, which prices the rule. Stating the contract in
  `ENTITY_EXTRACTION_RULES` instead is **TP -1.4 (p = 0.21), FP -1.8 (p = 0.47)** — the
  only arm in the round that clears the harness's own neutrality threshold on both
  measures, and it is on the *cleaner* side of precision.
* **`bindscans` — does not bind alone.** Deleting the two tight scans under s65's
  unchanged prompt costs TP 3.6 (p = 0.01); adding a clause that asks for exactly what
  they scan recovers **none** of it (TP -3.8, FP +3.2). Telling the model to report every
  sentence that writes a name as spelled does not make it do so. This is the third
  independent measurement of that, after `s_linker64`'s 3.0-pairs-per-run hole and
  `pilot/statednet_screen.py`'s 0.86-vs-0.06 gold rate.
* **`bindboth` — binds.** Contract *and* both tight scans relocated: **TP -1.2
  (p = 0.14), FP -1.2 (p = 0.37)**. Three deterministic rules and a `SCANS` row leave the
  code for two sentences. The apparent contradiction with `bindscans` is B6's mechanism:
  under the bound contract there is nothing left for the spelling row to re-admit.
* **`bindlabel` — the content binds, the performance does not.** Dropping the label costs
  **TP 8.4 (p = 0.01)**, which is the sharpest number yet on a field two earlier rounds
  argued about (`s_linker43`/`s_linker44` measured the same field's *values*). Asking the
  judge for it instead is worse on both measures (TP -3.6, FP +2.4). Asking the judge for
  it *and* giving the judging prompt the alias table recovers recall past the control and
  spends precision: **TP +4.0, FP +5.0**, both p = 0.01 — the merge law this branch has
  now seen six times (consolidating two decisions into one call raises recall and lowers
  precision), arriving from a new direction.
* **`bindpartial` — refuted.** An extractor asked, in the same call that already reads
  every sentence, to report single-word references reaches **TP 5.6 against the scan's
  18.0** at the same false-positive count. Third refutation of this hand-off, after
  `pilot/gate_pilots.py` (an LLM asked directly recovers 4.0 of 11.0 gold links) and the
  53.8-pair binding gap in B2. **The loosest cell of the relation is the one thing the
  extractor will not do, and it is the cell the partial-name linker exists to scan.**

## The variants

| variant | change | invariants |
|---|---|---|
| `s_linker65_null` | rename only — the in-set harness null | whole file is a rename of s65 |
| `s_linker66` | `_keep_stated_names` deleted, contract stated in `ENTITY_EXTRACTION_RULES` | one method deleted, one call site changed |
| `s_linker67` | s66 plus `_add_scan` deleted and `SCANS` down to `name_word` | two methods deleted, one call site changed |

`pilot/test_s66_s67_bindcontract.py` — **56/56**: every other method body byte-identical
to s65's, all rule constants / resource bounds / prompt builders identical apart from the
one constant, **the extraction prompt each variant sends is byte-identical to the arm the
stage pilot measured on all five projects** (without which the E2E arm is not the arm that
was screened), the deleted predicates unreachable while `_states_a_name` survives with its
remaining call sites, the candidate-set delta exactly as predicted on all five projects,
and GATE-06 over every catalog name and discovered alias.

## What is left to cut, priced before paying (B7)

Once the extractor side is bound, the only scan left is the partial-name row, carrying
three options and one span-boundary predicate. Freed candidates per five-project run,
and their gold:

| the partial-name scan | pairs | gold | gold/pair |
|---|---|---|---|
| as it stands | 60.0 | 19.0 | 0.32 |
| + no span-boundary test (`skip_qualified`) | 9.0 | 1.0 | 0.11 |
| + no unique-owner test (`unique_owner`) | 12.0 | **0.0** | 0.00 |
| + no whole-name exclusion (`skip_when_named`) | 151.0 | 127.0 | 0.84 |

* **`skip_qualified` is worth an arm** (`--pilot cutqualified`): 9.0 freed pairs carry
  1.0 gold, so a target-blind denotation judge that rejects 8 of 9 makes the predicate
  redundant — and after `s_linker67` that predicate has exactly **one** consumer left,
  so the arm decides whether `_inside_qualified_identifier`, `_in_dotted_path` and the
  documented `"" in "-_"` defect leave the workflow entirely.
* **`unique_owner` is not worth one.** It frees 12.0 pairs and **0.0 gold**, so removing
  it cannot raise recall and can only cost precision; `pilot/ablate_all.py` already
  priced it at 2.4 FP. Priced, not paid for.
* **`skip_when_named` is not a cut at all.** The 151 pairs it frees are the full-name
  linker's, and this is the alias table's suppression role — `s_linker46` measured its
  removal at macro F1 -1.5 (p = 0.00).

The label's qualified-path value is the other open cut (`--pilot cutcodetoken`): it is
the only consumer of `_all_occurrences_in_qualified_path`, it fires on 7.7 cases per run,
and the judge approves those at 69.6% against 95-99% for every other value.

**A consequence of s67 worth stating**: with the two tight scans gone, `SCANS` has one
row, and that row sets `skip_stricter=False` and `label_mention=False`, so both branches
in `_scan` are dead, `_full_name_source` can only return one value, and the relation's
four settings collapse to **two reachable ones** (`ANY_CASE` through `_find_exact_form`,
`ANY_WORD` through the scan). The monotone table's two tight rows stop being code and
become what they always were: a measurement.

## End to end

Six paired runs carrying s65, the in-set null, s66 and s67 in the same invocations:
`pilot/run_s6667_e2e.sh`, results in `../results/s6667_e2e_r{1..6}_20260817`, scored by
`pilot/score_runs.py` into `score_e2e.txt`. Read against the null arm's own delta rather
than against zero — this harness has produced 0.7 macro F1 from nothing in one set and
+0.4 in another; **in this set the null is quiet** (TP -1.2, F1 -0.1, F2 -0.2, all
p >= 0.39), so the arms' deltas are their own.

| arm | TP | FP | macro F1 | macro F2 | calls |
|---|---|---|---|---|---|
| `s_linker65` (control) | 189.8 | 12.3 | 96.1 | 97.1 | 88 |
| `s_linker65_null` | -1.2 (p = 0.39) | -0.7 (0.98) | -0.1 (0.80) | -0.2 (0.66) | 88 |
| **`s_linker66`** | **-2.5 (p = 0.15)** | -0.2 (1.00) | **-0.2 (p = 0.76)** | -0.5 (0.44) | 88 |
| `s_linker67` | **-4.0 (p = 0.03)** | +0.2 (1.00) | -0.6 (0.35) | **-1.1 (p = 0.04)** | 88 |

* **`s_linker66` holds — the admission contract can be a sentence of English.** Both
  scores are far from significance and false positives are identical; the recall delta
  is 2.5 TP against a null of -1.2 in the same set and does not reach significance
  (p = 0.15). It is stated here rather than buried: this is the least quiet parity the
  round claims, and the reason to accept it is that the arm removes a rule at F1 and F2
  parity, not that it improves anything.
* **`s_linker67` is rejected, and it is the round's methodological result.** Relocating
  the two tight scans *as well* reads TP -1.2 (p = 0.14) at its own stage
  (`--pilot bindboth`) and **TP -4.0 (p = 0.03), macro F2 -1.1 (p = 0.04)** composed.
  That is the ninth instance on this branch of a stage arm pointing the wrong way, and
  the mechanism is the one `_unlinked` always supplies: the pairs the scans add are
  pairs the later, stricter linkers never get a second chance at.
* **The relocation buys rule count, not calls** — 88 against 88. Nothing here is a
  performance claim; the claim is that the workflow states one rule fewer.

## The cutting arms (`pilots_cuts.log`)

Two arms on what B7 nominated, five samples a side against the same recorded inputs:

| cut | TP | FP | verdict |
|---|---|---|---|
| **the label's qualified-path value** (`cutcodetoken`) | **+0.0 (p = 1.00)** | -0.2 (p = 1.00) | **free at the stage** |
| the partial-name scan's span-boundary test (`cutqualified`) | +2.0 (p = 0.01) | **+5.8 (p = 0.01)** | **rejected** |

* **`cutcodetoken` — the label's qualified-path value goes.** It fires on 7.7 of 182.5
  cases per run and is the only consumer of `_all_occurrences_in_qualified_path`, a
  second, case-blind reading of the boundary question the scans already ask through
  `_inside_qualified_identifier`. Replayed on the recorded candidates the judge behaves
  identically — TP +0.0, FP -0.2, **composition p = 1.00**, i.e. the two arms' approved
  sets differ no more between arms than within one. **The judge sees the dotted path in
  the sentence it is shown and does not need to be told about it.** Carried into
  `s_linker68`.
* **`cutqualified` — rejected, and the defect stays.** Dropping the span-boundary test
  frees 9.0 candidates per run of which 1.0 is gold, and the target-blind denotation
  judge does *not* reject the other 8: TP +2.0 but **FP +5.8**, both p = 0.01. So
  `_inside_qualified_identifier` and `_in_dotted_path` earn their code, and the
  `"" in "-_"` reading stays where `s_linker63` left it — a documented validity threat
  that this benchmark rewards (its repair cost FP +1.2). This is the second time the
  same predicate has been priced from opposite directions and kept.
* **Not paid for, priced instead**: `unique_owner` frees 12.0 pairs and 0.0 gold, so it
  cannot raise recall; `skip_when_named` is the alias table's suppression role
  (`s_linker46`, F1 -1.5); the spelling row's own boundary test frees **0.0** pairs and
  is provably inert.

`s_linker68` = `s_linker66` + this cut: **two deterministic predicates and one label
value fewer than `s_linker65`.** Invariants in `pilot/test_s66_s67_bindcontract.py`
(64/64), including all 28 relabelled (name, sentence) pairs moving from the deleted value
and nowhere else. Confirmation batch: `pilot/run_s68_e2e.sh`, stopped at three paired runs, carrying
s65, the null, s66 (to nine runs) and s68 — owed because the label is the one field this
branch has twice been wrong about in this exact direction (`s_linker43`: trace-screen
neutral, E2E macro F1 -1.3; `s_linker44`: n=3 neutral, n=6 macro F1 -0.9). The outcome is
below, and it is neither of those.

## Batch 2 — `s_linker68` rejected, and `s_linker66` replicated

`pilot/run_s68_e2e.sh`, three paired runs (`../results/s68_e2e_r{1,2,3}_20260817`,
scored into `score_s68_n3.txt`; the batch was stopped at three by decision, so read
every p against the n=3 floor of 0.10). The null arm is quiet again: TP ±0.0, macro
F1 -0.2, F2 -0.1.

| comparison | TP | FP | macro F1 | macro F2 |
|---|---|---|---|---|
| `s_linker65` vs `s_linker66` | +1.7 (p = 0.40) | +2.7 (0.10) | **-0.0 (p = 1.00)** | +0.3 (0.40) |
| `s_linker65_null` vs `s_linker66` | ±0.0 (1.00) | +1.0 (0.40) | -0.2 (0.40) | -0.1 (0.80) |
| `s_linker68` vs `s_linker66` (macro — see below, not the measurement) | -5.0 (p = 0.10) | +1.0 (1.00) | -0.9 (0.20) | -1.6 (p = 0.10) |

* **`s_linker66` replicates in a second, independent invocation set** — macro F1 -0.0
  (p = 1.00) against s65 here, after -0.2 (p = 0.76) at n=6 in batch 1. Nine paired runs
  in two sets, two quiet nulls, and the relocation has never moved either score. This is
  the round's adopted arm.
* **`s_linker68` is undecided, and the way it looked rejected is the lesson.** The macro
  reads TP -5.0 with every run agreeing, which is what a rejection looks like. It is not
  one. s68 and s66 send **byte-identical** knowledge, extraction and later-linker prompts
  — the cut can only reach full-name judging verdicts on the 28 pairs that carried the
  deleted value — so the macro was decomposed by where each lost gold link dies:

  | the 5.7 gold links per run s66 has and s68 lacks | s68 | control (s65 vs its null) |
  |---|---|---|
  | s68's extraction call never proposed it | **4.0** | 0.3 |
  | both proposed it, s68's judge rejected it | 1.0 | 0.7 |
  | neither proposed it (a later linker's) | 0.7 | 0.7 |

  **Four fifths of the gap is at a stage whose prompt the change does not touch.** It is
  entirely bigbluebutton (5.3/run, zero on three of five projects), and in r2 the two arms
  built identical alias tables and the gap persisted. None of the lost links carried the
  deleted label value: 4.0 are `lowercase mention`, 1.0 `proper case, standalone`, 0.7
  `indirect`.

  Restricted to what the change can reach — approvals on the candidates **both** arms
  proposed — s68 is neutral and the identical-code control moves as much:

  | full-name approvals on shared candidates | TP | FP | composition |
  |---|---|---|---|
  | s68 vs s66 | -1.0 (p = 0.30) | -0.3 (1.00) | -0.6 (0.50) |
  | control: s65 vs its null | -0.7 (0.70) | -1.0 (0.10) | +0.2 (0.50) |

  So the cut is **not adopted and not refuted**; deciding it needs more runs or a
  per-source read with more samples. `s_linker66` is the confirmed endpoint either way.
* **The methodological point is about the reader, not the arm.** This branch added the
  rule "read an arm on the `source` its change can reach, and use the stages it cannot
  reach as the control" in the partial-name round, after s62 read p = 0.00 at its source
  and p = 0.81 in the macro. Here the same rule reverses a *negative* macro instead of
  rescuing a positive one, on an arm whose reachable surface is 28 pairs. **A macro F1 on
  a four-arm invocation mixes one stage's effect with two stages of sampling; when the
  change's reachable surface is small, the macro is not the measurement.**

## Standing result

**Of five rule groups, one relocates.** The admission contract becomes two sentences in
the extraction prompt at parity; the two tight scans, the partial-name scan and the
mention label do not relocate at any wording tried, and the four structural predicates
have no prompt form at all. The floor on "how many hand-written rules does this workflow
have" is therefore not zero, and every element of it is now priced rather than asserted.

**`s_linker66` is the endpoint of the cutting, and the deterministic layer is
exhausted.** Every remaining element has now been measured, and none can be removed or
relocated at parity:

| what is left | why it stays | measured |
|---|---|---|
| the relation (`_name_spans`, `_find_exact_form`, `_realizes`, `_owners`, `_name_signature`) | the three scans read it | identity over 3697 pairs (`rule_audit` A2) |
| `SCANS[stated_name]`, `SCANS[spelling]` | relocation rejected | `s_linker67`: TP -4.0, F2 -1.1 (n=6) |
| `SCANS[name_word]` | relocation refuted three ways | stage TP 18.0 → 5.6; gap 53.8 pairs; `gate_pilots` 4.0 of 11.0 |
| `unique_owner` | frees 0.0 gold | B7; `ablate_all` 2.4 FP |
| `skip_when_named` | the alias table's suppression role | `s_linker46`: F1 -1.5 |
| `skip_qualified` + `_inside_qualified_identifier` + `_in_dotted_path` | the judge does not reject what it frees | `cutqualified`: FP +5.8 (p = 0.01) |
| `_classify_mention_typed` (five values) | dropping it is the largest single loss measured on the field | -8.4 TP (stage); `s43` -1.3 F1; `s44` -0.9 F1 |
| `_all_occurrences_in_qualified_path` | `s68` cuts it; neutral at its own source, undecided in the macro | shared-candidate TP -1.0 (p = 0.30) |
| `_states_a_name` (2 sites) | whole-name exclusion, antecedent gate | `s46` F1 -1.5; the gate is worth 12 FP |
| `_iter_batches`, `_window`, `_unlinked`, `_union` | no prompt form | `s_linker27` F1 91.70; coref scope A3 |

So the answer to "how few rules can this workflow have" is: **one relation at four
settings, one mention label, one name predicate, and four structural predicates — with
the admission contract now written in English instead of code.**
