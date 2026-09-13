# The label-and-rule round — what the union judge is shown, and in whose words

Four questions asked of `s_linker120`'s one judge, all at level 2 of the measurement
policy (stage arms on fixed recorded candidates, three samples a side, every arm in the
same invocation, alias table pinned from `consolidation_e2e_terra_r1_20260825`), plus a
level-1 census and a byte-level refactor guard that cost no calls at all.

    ../.venv/bin/python pilot/union_render_snapshot.py --check before.json   # level 0
    OPENAI_MODEL_NAME=gpt-5.6-terra ... pilot/union_pilots.py \
      --arms union aliasmute alllabels nomention v14n --samples 3 --dump <dump>
    ../.venv/bin/python pilot/union_stats.py <dump> --arms union <arm>

Artifacts here: `dump_terra_labels.json` (every arm's kept pairs per sample per
project), `run_terra.log` (the invocation).

## The arms

The candidate set is byte-identical across every arm and sample — 296 cases, 180 gold,
14 judging calls — so the only thing that varies is what the cases say and in whose
words.

| arm | change | cases it changes |
|---|---|---|
| `union` | the head, `v13` | control |
| `alllabels` | `RETAINED_MENTION_TYPES` = every `MentionType` | 144 of 296 gain a `mention=` line |
| `aliasmute` | drop `VIA_ALIAS`, which restates the case's own `writes` line | 43 lose one |
| `nomention` | print no computed label at all | 71 lose one |
| `v14n` | every carried criterion clause paraphrased into general English | all 296 (the rule above them) |

## Results — three samples, five projects, per five-project run

| arm | kept | gold | p | spurious | p | net | p | precision |
|---|---|---|---|---|---|---|---|---|
| `union` | 189.0 | 174.3 | — | 14.7 | — | — | — | 0.922 |
| `alllabels` | 190.3 | ±0.0 | 1.000 | +1.3 | 0.562 | −1.3 | 0.750 | 0.916 |
| `aliasmute` | 190.0 | ±0.0 | 1.000 | +1.0 | 0.672 | −1.0 | 0.875 | 0.918 |
| `v14n` | 195.3 | +0.3 | 1.000 | +6.0 | 0.109 | −5.0 | 0.336 | 0.894 |
| `nomention` | 197.3 | −1.7 | 0.375 | +10.0 | 0.125 | **−15.0** | **0.031** | 0.875 |

Deltas are against the control that ran in the same invocation, over 15 paired
(sample, project) units, `net = 3*gold − spurious`, two-sided sign-flip permutation p.

## What each one settles

**Showing every label is possible, free to implement, and buys exactly nothing.** It is
one frozenset — no method changes, because `_retained_mention_label` already reads the
set. It puts a `mention=` line on 144 cases that are silent today, 131 of them gold, and
it moves gold by **0.00 a unit at p = 1.000**. The level-1 census says why: the label is
a function of the case's own `naming` row plus its capitalization, so
`proper case, standalone` (108 cases, 0.963 gold), `lowercase mention` (36, 0.750) and
`indirect/unclear match` (81, 0.321) are all things a judge holding the sentence reads
off the sentence. `STRICTER_CLAUSE` already tells it what capitalization is worth. **The
retained set was chosen by an argument about re-derivability, and the argument is right.**

**But the field as a whole is load-bearing, and that is the round's one significant
result.** `nomention` is net **−15.0 a run at p = 0.031**, all of it at the whole-name
row (spurious +2.00 a unit) — the 28 cases labelled `lowercase, inside qualified name`,
a bucket that is 0.071 gold. `QUALIFIED_CLAUSE` is in the rule for every one of those
cases and does not, on its own, do what the label does. **A clause that states the
criterion is not a substitute for a fact that says this case is an instance of it** —
the design law, from the side that is usually taken for granted.

**The redundant half is free to within the noise, and is kept anyway.** `VIA_ALIAS` is
printed on 43 cases and every one of them already prints `writes=a short form the
document established for it` — the same fact, twice, in one `Evidence:` line. Removing
it is gold-neutral at net −1.0 (p = 0.875). The point estimate is unfavourable and
nothing is bought, so the finetune round's rule applies: **an unnecessary change is not
a defensible one.** Recorded as free-if-wanted, not adopted.

**Naturalizing the rule loses, and it was never free.** `v14n` holds the definition, the
field lines, the format contract, the demand, the reply, the fields and every flag, and
paraphrases only the four carried criterion clauses into plain general English —
"a mention that says nothing further about the component still counts as a valid link"
becoming "an architectural mention of the component is enough to justify a link, even
where the sentence says nothing further about it", and so on. It reads **gold +0.3
(p = 1.000) at spurious +6.0, net −5.0**, with the control ahead or level on 8 of 15
units. Nothing is significant, which is the point: the paraphrase buys no gold, its
whole point estimate is spurious, and it is concentrated on the word-only row (+1.07
spurious a unit against +0.33 on whole-name).

**And a tie would still have been a loss.** Quotation is not a stylistic choice here: it
is what lets `pilot/union_defensibility.py` check each criterion clause against the
ancestor constant it was sliced from. A paraphrase has to be scored as authored text
against GATE-07 instead, so the arm had to *win* to be worth adopting, and it did not.
**This is the branch's first clean measurement of quotation against paraphrase** —
iterations v1 → v2 moved the same way and cannot be read for it, because they moved the
alternative set from a reject-ground to context in the same step.

## The level-0 result: no dead code, and why the file cannot get much shorter

A mechanical pass over `s_linker120.py` (defs, module constants, class attributes,
imports, all counted against the source with comments and docstrings stripped) found
**no unreferenced symbol, no commented-out block, no TODO and one genuinely dead line**
— a function-level `from ... import get_comp_names` shadowing the module-level import of
the same name, now removed.

What *looks* dead at the head is not:

* the `source`, `naming` and `last_named` label branches in `_evidence_facts`, and the
  `clauses` slot, the `per_row` verdict and the un-grouped batching in `_prompt_union` /
  `_judge_union`, are all **reachable by iterations v1–v6 of the trail**, which
  `pilot/union_pilots.py --arms control v3` runs as arms. They are the price of the
  iterations being data rather than prose.
* 41 of the file's methods are **`s_linker110`'s text byte for byte**, pinned by
  `pilot/test_s120_standalone.py` T2. That copy is the evidence for the claim the file
  exists to make — that only the name judging changed — so refactoring it would delete
  the claim, not clean it. `_named_spans` is a one-line indirection whose documented
  purpose (an override point for `s_linker92c`/`92d`) nothing in this file's lineage
  uses, and it stays for exactly that reason.

So the cleanup is confined to the 9 methods that are this round's own, and it is
decomposition rather than deletion: `_judge_union` 93 lines → 40 plus five named steps,
`_union_evidence` 46 → 21 plus three named facts, and the case renderer's
dict-of-lambdas → one `_evidence_facts`. **The evidence is now computed once per
candidate** and read by the grouping, the window, the case and the decision record
alike, where it used to be computed twice and could in principle have bucketed a
candidate on one reading of its match and printed it on another.

`pilot/union_render_snapshot.py` is the guard the compaction round's lesson asks for —
written before the refactor, not after. It hashes every case, every prompt and every
judged decision (under a stub that answers both contracts and alternates the verdict)
over 5 projects × 2 alias tables × 14 iterations: **280 renderings, all identical across
the refactor.** `test_s120_standalone.py` 85/85, `test_s120_union.py` 2593/2593,
`union_defensibility.py` 40/40.

**A note on speed, since it was checked before it was claimed.** The deterministic layer
first reads 3.4 s on bigbluebutton, which is WordNet's one-time corpus load and not the
relation: warm, the whole scan-and-evidence pass over all five projects is **513 ms**.
Computing the evidence once instead of twice is worth ~250 ms a run and is done for
readability, not for speed.

## Cost

210 judging calls in one invocation. No E2E was bought: every arm is either refused or
gold-neutral-and-not-adopted, so the head does not move and there is nothing to compose.
