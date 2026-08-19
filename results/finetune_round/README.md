# The finetune round — every remaining corpus-shaped span, removed or re-grounded

`s_linker74` → `s_linker75`. The general round scored the authored surface and found
**1700 of 3645 bytes** standing on an admissible ground. s74 fixed the one span in the
judging path. This round asks what is left, removes it, and — where a clause turns out to
be defensible after all — says so with the measurement rather than quietly keeping it.

The budget for the round was set in advance: **up to 2 pp of macro F1 may be spent to
remove finetuning.** Nothing below cost anywhere near that, which is itself the result.

## What was still finetuned in s74

| span | what it said | kind |
|---|---|---|
| `ALIAS_EXCLUSION_RULES` | `X.Y or X.Y.Z`, spelled | **corpus** — the last spelled syntax in the module |
| `ENTITY_EXTRACTION_RULES` | "a name that appears only inside a code-level path" | bespoke restatement |
| `P1_FOCUS` | "rather than only as part of a code-level identifier" | bespoke restatement |
| `LAYERED_COREF_RULES` | "or when the reference is only to a code-level identifier" | bespoke restatement |

One distinction, five wordings, one of which names a syntax. A sixth already existed that
names none: `QUALIFIED_CLAUSE`, se-practice ground, written in the fold round.

## The arms (step 2 of the measurement policy — one stage, recorded inputs, 3 a side)

`pilot/finetune_pilots.py` (new: `rubricsyntax`, `rubricsyntax_min`, `aliascomp`) and
`pilot/general_prompt_pilots.py` (`plainextract`, `plaincoref_min`, already run in the
general round against prompts s74 carries byte-identically). Every base arm asserts first
that its re-declared prompt builders render byte-identically to `s_linker74`'s.

| arm | what varies | TP | FP |
|---|---|---|---|
| `plainextract` | extraction: general clause instead of the code-path one | +0.7 (p = 1.00) | −6.0 (p = 0.20) |
| `rubricsyntax` | P1's tail dropped, `QUALIFIED_CLAUSE` **added** | −0.7 (p = 0.90) | −1.3 (p = 0.40) |
| **`rubricsyntax_min`** | **P1's tail dropped, nothing added** | **+2.3 (p = 0.20)** | **±0.0 (p = 1.00)** |
| `plaincoref_min` | coreference: phrase removed, nothing added | +4.7 | +3.7 |

**A clause should be stated once per prompt, and the arms can tell.** The full-name
judging prompt already carries the distinction inside reject-condition (1), which s74
re-grounded; adding `QUALIFIED_CLAUSE` as well is a restatement and reads worse on recall
than removing the tail alone. The extraction prompt has no such enumeration, so there the
clause is added. The coreference prompt gets neither — its cases contain no name for a
clause about identifiers to be about (the general round measured replacing it at TP −3.0).

## The alias arm — the general round's reason for keeping the syntax does not reproduce

`aliascomp`, read on the **judged** table (what the extraction prompt actually receives),
three arms in one invocation so the loss and the attempted recovery are visible together:

| arm | judged aliases per run | identifier fragments admitted (15 project-runs) |
|---|---|---|
| s74 — spells out `X.Y` | 36 / 37 / 34 → **35.7** | **0** |
| general wording, judge unchanged | 33 / 49 / 36 → 39.3 (FP +3.7, p = 0.90) | 6, in 1 of 15 |
| general wording, judge tie-break flipped | 41 / 35 / 37 → 37.7 (FP +2.0, p = 0.50) | 13, in 3 of 15 |

The general round measured 24.0 terms with the syntax against 36.7 without and kept the
clause on that basis, documented as "doing something other than what it states". Against
s74's own checkpoints the syntax arm itself reads **35.7** — the gap is gone, and the
earlier number was one invocation set's level, which this branch has documented drifting
before (s49's FP mean read 10.7 → 16.8 across five sets in one day).

Two honest consequences, both stated rather than buried:

1. **The clause's cost is now smaller than its defence assumed, so it goes.**
2. **It was not doing nothing.** On its own stated prohibition it buys 6 admitted
   identifier fragments in one project-run out of fifteen. That is the price of the
   removal and it is reported, not rounded away.

**Compensating at the judge does not work.** Flipping the alias judge's tie-break from
APPROVE to REJECT — the branch's own law, *the looser the proposer, the stricter the judge
behind it* — neither shrinks the table nor keeps fragments out. Not adopted: an
unnecessary change is not a defensible one.

## What was left alone, and why that is not an exemption

The rubric `LAYERED_ENTITY_RULES` is byte-identical to s74's, and it was on the general
round's corpus list. Three variants measured its two structural properties:

- the four numbered reject-conditions — replacing them with one principle is TP +0.7 /
  FP −1.3 on a fixed candidate set and **~0.8 F1 composed** (`s71` 94.80 n=6, `s72` 94.94);
- "a heading, or a list" — removing it is **exactly 2.7 TP in each of three runs** (`s73`).

Neither is a corpus shape. An enumeration is a rubric structure, and headings and lists
are general technical-documentation practice. **GATE-07 catches shapes peculiar to a
corpus, not the structure every document of the genre has.** Since s74 re-grounded
condition (1), every clause in the rubric now stands on logic (negation), the use/mention
distinction (3, 4, and `STRICTER_CLAUSE`), or the compositionality of qualified names (1).

## The score the round exists for

`pilot/prompt_defensibility.py --variant s_linker75` (no LLM calls):

| | s_linker70 | **s_linker75** |
|---|---|---|
| authored bytes | 3645 | 3412 |
| general | 1287 | **2866** |
| se-practice | 166 | 299 |
| prior-work | 247 | 247 |
| mixed | 1131 | 0 |
| **corpus** | **814** | **0** |
| admissible | 1700 (47%) | **3412 (100%)** |

## What is a strength and not a finetune — the parts nothing in this round touched

Removing fitted text is only half the claim; the other half is that what remains is
mechanism, each priced:

| element | what it is worth | measured in |
|---|---|---|
| claim-before-verdict (the judge quotes the sentence first) | **35.2 TP** | s25 design pilots |
| the evidence bundle shown with each case | FP 8.3 vs the 4–6 band when trimmed | s25 complexity round |
| target-blind denotation before any identity review | **12 FP** | s25 ablate-all |
| two focused judging passes instead of one merged prompt | FP 4.8 vs 8.3; the passes disagree 4.5× more than two samples of one prompt | s36/s38 audit |
| lenient full-name gate vs strict coreference gate | opposite defaults follow from opposite evidence; the strict one is worth 12 FP | s38 audit |
| the mention label computed in code, not asked of the judge | **−6.7 TP** if asked, **−10.7 TP** if removed | concept round |
| `unique_owner` kept in code | **−8.4 TP** if folded into the prompt | fold round |

The design law that separates the two halves: **facts stay in code, weighings go in the
prompt.** Everything this round removed was a weighing stated in a corpus-shaped way;
nothing it removed was a fact about a case.

## The non-prompt surface, audited on the same terms

Removing fitted English is only half the question a reviewer asks; the other half is
whether the numbers and word lists in the code were fitted. All of them, deterministically
(no LLM calls):

| element | value | ground |
|---|---|---|
| `INFLECTIONS` | 9 English endings | **5 of the 9 never fire on any of the 3697 (name, sentence) pairs** — only `""` (338), `ing` (21), `s` (14) and `ed` (1) do. A list fitted to this benchmark would contain those four; this one is general English morphology and is *larger* than the benchmark needs. It is also the module's only word list (GATE-06). |
| `WORD_PATTERN` | word boundaries only | splitting compounds here was measured to triple the candidate set and reach no additional gold link |
| `CONTEXT_SENTENCES` / `ANCHOR_LIMIT` | 5 / 5 | one value, not two; halving both to 2 costs 2.0 TP (p = 0.20) for no precision gain (s25 design pilots) |
| `EXTRACTION_BATCH` | 50 | reference reading degrades with passage length — `s_linker27` reads F1 98.4 at 37 sentences, 79.7 at 87, 84.1 at 198 |
| `JUDGE_BATCH` | 25 | judging one candidate per call instead of 25 is neutral (TP +0.7 p = 0.60, FP +0.3 p = 1.00), so batching does not decide links |
| `COREFERENCE_BATCH` | 10 → **`JUDGE_BATCH`** | the only bound with no counterpart, and 40.0 of 91.7 calls per run. Unified in `s_linker76`; `s_linker45` measured the same unification at parity over six paired runs on the s25 base (F1 -0.2 p = 0.52, F2 -0.0 p = 0.91, 65.3 calls against 88.8) |

**A value chosen by unification is defensible; a value chosen by search is not.** That is
the line this round draws, and the reason `s_linker76` sets the coreference batch equal to
a number the module already states rather than sweeping for the best one.

## End to end

Three arms in one invocation — `s_linker75`, its in-set null, and `s_linker74` as the
control — three runs, flex tier (`pilot/run_s75_e2e.sh`,
`../results/s75_e2e_r{1,2,3}_20260819`). Composition risk is non-zero (two of the four
spans are in the alias and extraction prompts, which feed every later stage), which is
what the batch is paid for; the stage arms could not decide it.

| arm | n | TP | FP | macro F1 | macro F2 | calls | F1 range |
|---|---|---|---|---|---|---|---|
| `s_linker74` (control) | 3 | 182.7 | 15.0 | 94.42 | 94.46 | 90 | 2.51 |
| `s_linker75_null` (harness null) | 3 | 184.0 | 25.7 | 92.84 | 94.27 | 90 | 1.54 |
| **`s_linker75`** | 3 | 182.7 | 22.7 | **93.59** | **94.49** | 90 | **0.84** |

**Read against the null, not against zero — and this set is why the rule exists.** The
null differs from the control by **F1 -1.58 and FP +10.7** on code whose only difference
is a checkpoint namespace. Against that reference (`pilot/score_runs.py`, exact paired
permutation):

| comparison | TP | FP | macro F1 | macro F2 | verdict |
|---|---|---|---|---|---|
| **s75 vs its null** | -1.3 (p = 0.60) | **-3.0 (p = 0.30)** | **+0.7 (p = 0.30)** | +0.2 (p = 0.70) | **QUALITY-NEUTRAL**, composition +0.0 (p = 0.50) |
| s75 vs s74 | **±0.0 (p = 1.00)** | +7.7 (p = 0.10, the n=3 floor) | -0.8 (p = 0.40) | **±0.0 (p = 1.00)** | flagged on FP — but the null moved FP by +10.7 against the same control, *more* than s75 did |

**The round's answer: removing every finetuned span costs at most 0.8 macro F1 (p = 0.40),
nothing at all on recall (TP ±0.0) or on F2 (±0.0), and against the harness's own null it
costs nothing on any measure.** The budget was 2 pp; the price is a quarter of it at the
worst reading and zero at the honest one. s75 also has the tightest run spread of the
three arms (F1 range 0.84 against the control's 2.51), which is what a prompt with fewer
bespoke clauses in front of a judge should look like.

Per-run macro F1: s74 [95.69, 94.39, 93.18], null [93.48, 91.94, 93.10],
s75 [93.26, 94.10, 93.41].

**Caveat stated rather than buried:** arm order inside each invocation is s75, null, s74,
and s74 scores highest in all three runs. The prompt round documented the same position
effect and ruled it out only by reversing the order in a second batch
(`../results/nullrev_e2e_*`). This batch did not pay for the reversal, so the s75-vs-s74
row is confounded with position and the s75-vs-null row is the one to quote.



## `s_linker76` — the last tuned number, priced and **not** adopted

Three paired runs, two arms (`pilot/run_s76_e2e.sh`,
`../results/s76_e2e_r{1,2,3}_20260819`):

| arm | TP | FP | macro F1 | macro F2 | calls |
|---|---|---|---|---|---|
| `s_linker75` | 184.7 | 22.7 | 93.79 | 94.75 | 89 |
| `s_linker76` (`COREFERENCE_BATCH = JUDGE_BATCH`) | 177.7 | 18.0 | 93.07 | 92.93 | **65** |

`s_linker76 minus s_linker75`: **TP -7.0**, FP -4.7, macro F1 -0.7, **macro F2 -1.8**
(every p at the n=3 floor of 0.10), at **-27% LLM calls**.

**Base-dependence, again.** `s_linker45` measured this identical unification on the s25
base over six paired runs at parity — TP +0.8 (p = 0.56), macro F2 -0.0 (p = 0.91). On the
s70-s75 base it costs seven true positives. That is the branch's standing finding arriving
once more: **an arm measured neutral in one composition is not neutral in another**, and
the difference here is that s75's coreference stage sits behind three linkers that subtract
from it, so a wider resolution batch changes which cases share a prompt.

**Verdict: kept as the cheap variant, not adopted as the head.** The cost is inside the
2 pp F2 budget, but it spends that budget on *call count* rather than on defensibility, and
it spends it in recall — the direction an F2-led paper least wants. The tuned number is
therefore reported with its price rather than removed: `COREFERENCE_BATCH = 10` stays in
the head, and `s_linker76` exists so the alternative is priced rather than assumed.


# The elegance round — how much structure comes out for 3 pp of macro F2

The finetune round removed the fitted *English*. This round asks the same question of the
*structure*, at a budget the paper's lead measure sets: **3 pp of macro F2, F1 not a
constraint.** Four arms, each the previous plus one cut, all in **one invocation set** with
an in-set null, so every comparison below is paired and none is read across sets
(`pilot/run_elegance_e2e.sh`, `../results/elegance_e2e_r{1,2,3}_20260819`).

| arm | the cut it adds | TP | FP | macro F1 | macro F2 | calls |
|---|---|---|---|---|---|---|
| `s_linker75` | control — the finetune round's head | 181.0 | 22.0 | 92.99 | 93.68 | 89 |
| `s_linker75_null` | in-set null | 181.3 | 25.0 | 92.10 | 93.31 | 89 |
| `s_linker77` | two tight `SCANS` rows → the extraction prompt (**3 rows → 1**) | 177.3 | 25.0 | 91.25 | 91.92 | 87 |
| **`s_linker78`** | **+ the rubric's four numbered conditions → one principle** | **184.3** | **22.0** | **93.15** | **94.41** | 89 |
| `s_linker79` | + the row's last two options (**no gate anywhere**) | 182.0 | 39.0 | 89.66 | 92.26 | 98 |
| `s_linker80` | + the computed mention label (**nothing computed**) | 180.7 | 32.7 | 90.59 | 92.30 | 98 |

Paired permutation against the control (`pilot/score_runs.py`; the null reads
macro F2 −0.37):

| arm | TP | FP | macro F1 | macro F2 |
|---|---|---|---|---|
| **`s_linker78`** | **+3.3 (p = 0.10)** | **±0.0 (p = 1.00)** | +0.2 (p = 0.90) | **+0.7 (p = 0.20)** |
| `s_linker77` | −3.7 (p = 0.40) | +3.0 (p = 0.70) | −1.7 (p = 0.40) | −1.8 (p = 0.30) |
| `s_linker79` | +1.0 (p = 0.70) | **+17.0 (p = 0.10)** | −3.3 (p = 0.10) | −1.4 (p = 0.20) |
| `s_linker80` | −0.3 (p = 1.00) | +10.7 (p = 0.10) | −2.4 (p = 0.10) | −1.4 (p = 0.10) |

## The result, and the mechanism behind it

**`s_linker78` is the head: it removes more structure than any variant in this branch and
is not worse than the thing it removes it from — TP +3.3, FP ±0.0, F2 +0.7.**

The interesting part is that `s78` *contains* `s77`'s cut, and `s77` alone reads −1.8 F2.
**The two cuts are complements, and the mechanism is legible.** Relocating the two tight
scans makes the extraction call responsible for the incidental mentions they used to
guarantee, and it duly proposes more of them; the *enumerated* rubric then rejects them
(condition (4), "a generic technique or technology term", and condition (1)) while the
one-principle rubric approves them. So the enumeration was carrying precision exactly
against candidates the scans were not producing — which is why every earlier round, all of
which kept the scans, measured its removal as a loss (`s71` 94.80, `s72` 94.94 against
`s70`'s 95.74).

**Standing lesson, and the tenth instance of it on this branch:** a clause is not
independently priceable. Two changes that each lose ground can gain it together when one
changes what the other's population contains, and no single-arm round could have seen it —
`s71` measured the rubric against the *old* candidate set, and that is the number that
kept the enumeration for four variants.

## The frontier, priced rather than adopted

Both remaining cuts are **inside the 3 pp F2 budget** and neither is adopted, for a stated
reason rather than a score:

- **`s_linker79` (no gate anywhere)** — F2 −1.4 for **+17.0 false positives**. `unique_owner`
  and `skip_when_named` are worth about 17 spurious links between them, which is a bigger
  number than either had been given before (2.4 FP and F1 −1.5 respectively, both on the
  s25 base). The workflow can be stated with no deterministic gate at all, and the price of
  saying so is now measured.
- **`s_linker80` (nothing computed)** — F2 −1.4, FP +10.7 against `s79`'s +17.0, i.e.
  removing the mention label on top of the gates *recovers* precision relative to `s79`
  while costing nothing further on F2. The concept round priced the label at −10.7 TP with
  the gates in place; without them it is worth almost nothing. **A fact's value depends on
  what else the code is doing** — the label was compensating for the same candidates
  `unique_owner` was.

Under the stated budget a reviewer could take `s_linker80` and claim a workflow whose code
computes nothing about a case. This branch does not, because `s_linker78` is *better* on
every measure at nearly the same simplicity, and taking a worse variant for a rhetorical
sentence is the kind of choice this whole audit exists to prevent.

## Invariants

`pilot/test_s78_head.py` — 34 checks on the head: `SCANS` is one row and `_add_scan` is
gone; no prompt enumerates, names a syntax, or restates the distinction bespoke-ly; the
enumeration's two grounds are each stated once by a clause in the same prompt; every
measured asymmetry survives (approve-by-default, reject-when-uncertain, the alias judge's
opposite tie-break, claim-before-verdict, two focused passes, the computed mention label);
and the chain to s75 is exactly two cuts — two authored constants, two method bodies, one
deletion, 47 other method bodies and all 7 bounds byte-identical.

`pilot/test_s76_twobatches.py` — 7 checks on the priced batch-constant variant.

`pilot/test_s75_nofinetune.py` — 36 checks: exactly four authored constants differ from
s74's; no constant spells a syntax or restates the distinction bespoke-ly; every
load-bearing span survives; no prompt carries the clause twice; 49 method bodies, 7 class
attributes, all `SCANS` rows and the inflection list are byte-identical to s74's.
