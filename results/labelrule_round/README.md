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

## The naturalization ablation — which clause pays

`v14n` moves four paragraphs at once, so a second batch asks each of them alone: five
arms, each `v13` with exactly one paragraph paraphrased and everything else the head's
bytes, plus `v14n` re-run **in the same invocation** so the composite and its parts are
comparable. Seven arms, three samples, 294 calls. Dump: `dump_terra_naturalize.json`.

| arm | paragraph | gold | p | spurious | p | net | p |
|---|---|---|---|---|---|---|---|
| `v14def` | the trace-link definition | −1.0 | 0.500 | −2.3 | 0.125 | −0.7 | 0.938 |
| **`v14mention`** | **`MENTION_COUNTS`** | **−5.3** | **0.062** | −2.3 | 0.250 | **−13.7** | **0.062** |
| `v14ground` | `POSITIVE_GROUND` | −2.7 | 0.250 | −2.0 | 0.125 | −6.0 | 0.250 |
| `v14stricter` | `STRICTER_CLAUSE` | −0.3 | 1.000 | −1.7 | 0.438 | **+0.7** | 0.906 |
| `v14ref` | `QUALIFIED_CLAUSE` + `ACTS_ON` | −0.7 | 0.625 | +1.3 | 0.562 | −3.3 | 0.188 |
| `v14n` | all four | ±0.0 | 1.000 | +0.3 | 1.000 | −0.3 | 1.000 |

**The clause that pays is the one a reader is likeliest to want reworded.**
`MENTION_COUNTS` is "a mention that says nothing further about the component still
counts as a valid link". The obvious naturalization — "an architectural mention of the
component is enough to justify a link, even where the sentence says nothing further
about it" — reads as a synonym and is not one. **The original lowers a bar; the
paraphrase restates it.** Saying the mention must be *architectural* reimports exactly
the criterion the sentence exists to relax, and the judge duly applies it: gold −5.3 a
run at p = 0.062, the lowest p in either naturalization batch.

**It lands where the mechanism predicts.** The cost is concentrated on the word-only
row — the cases with the least surface to go on — which goes from 29.0 kept / 22.0 gold
to 22.7 / 18.3. The whole-name row, where the sentence writes the name outright and the
licence is not what carries the verdict, barely moves. `v14ground` is the same effect at
half the size, from spelling out "a positive ground": word-only kept 29.0 → 24.7.

**`STRICTER_CLAUSE` is the one paragraph that is safe to reword** (net +0.7, the only
non-negative estimate in the family), and the asymmetry is readable: it is a *test*, not
a licence, so restating it does not move what it licenses. `v14ref` is the only arm that
costs precision rather than recall, and both of its clauses are folded code gates — the
one part of the rule with a measured deletion cost behind it (FP +7.0, p = 0.01).

**The parts do not sum to the whole, and this is the round's transferable result.**
`v14mention` alone is net −13.7 and `v14ground` alone −6.0, yet all four paraphrased
together is **−0.3, every p = 1.000**. **A clause is not independently priceable** —
s77/s78 measured that with two losers composing to a winner; this is the third
instance and the third direction, two losers composing to a wash. A rule read as a
whole has a register, and moving every paragraph into the same register is not the sum
of moving each one into it alone.

**And the same arm read differently in two invocation sets.** `v14n` is net −5.0 in the
label batch and −0.3 here. Both are real; neither is a trend. This is why the branch's
rule is that arms are comparable only inside one invocation, and it is why the
composite was re-run here instead of being compared across.

**Verdict unchanged, reason sharpened.** Naturalization is refused — but not as "prose
is risky". It is refused because one specific sentence in it is not a paraphrase at all,
and because none of the five arms buys anything to pay for the quotation it spends.

## The level-0 result: the file stops being a diff against its ancestor

A mechanical pass over `s_linker120.py` (defs, module constants, class attributes,
imports, all counted against the source with comments and docstrings stripped) found
**no unreferenced symbol, no commented-out block, no TODO and one genuinely dead line**
— a function-level `from ... import get_comp_names` shadowing the module-level import of
the same name, now removed.

What *looks* dead is not: the `source` / `naming` / `last_named` label branches, the
`clauses` slot, the `per_row` verdict and the un-grouped batching are all **reachable by
trail iterations v1–v6**, which `union_pilots.py --arms control v3` runs as arms.

**The real finding is structural, and it was hiding behind the byte-identity rule.**
Holding every shared method identical to `s_linker110` is what kept the file a diff
against its ancestor rather than a file in its own right, and what it kept was:

* `HEAD DELTA 1/2/3` banners naming **`s_linker110`'s** derivation from `s_linker92` —
  archaeology from two variants back, in the file that is the current paper supplement,
  with no key in its own docstring;
* seven 1:1 wrappers around `linker_infra` (`_iter_batches`, `_link_view`,
  `_decision_view`, `_linker_feedback`, `_compute_phase_metrics`, `_backend_tag`,
  `_checkpoint_dir`) plus `_named_spans`, a one-line indirection whose documented
  purpose was an override point for two variants that do not fork this file;
* a five-method proposer chain (`_extract_named_mentions`, `_scan_all`, `_scan`,
  `_covering_names`, `_only_inside_another_name`) and a four-method label chain
  (`_classify_mention_typed`, `_all_occurrences_in_qualified_path`, `_in_dotted_path`,
  `_retained_mention_label`);
* **`_writes_name`, which is `_find_exact_form`** — the same function, behind a
  `SKIP_QUALIFIED` flag the head declares `False`. Two names for one predicate, kept
  apart only because the ancestor had them apart.

All 18 are inlined and declared in `INLINED`; `SKIP_QUALIFIED` is deleted. The proposer
is one `_name_candidates` in four labelled blocks, the label is one `_mention_label`,
the judge is `_judge_union`'s four blocks. **51 methods → 33**, and the evidence is
computed once per candidate instead of twice.

**The claim the byte comparison was making is now made by behaviour, and more
strictly.** T6 runs `s_linker110`'s own `_extract_named_mentions` and `_scan` beside this
file's `_name_candidates` and compares every candidate — pair, component, source label
and the matched surface, which is what a rewritten span loop would move first — over
five projects under both alias settings. Byte identity never said what the bytes did;
this does. The suite goes **85 → 133 checks**, not 85 → fewer.

`pilot/union_render_snapshot.py` is the guard the compaction round's lesson asks for —
written before the refactor, not after. It hashes every case, every prompt and every
judged decision (under a stub that answers both contracts and alternates the verdict)
over 5 projects × 2 alias tables × 14 iterations: **280 renderings, all identical across
the refactor** (380 once the ablation arms are in the trail). `test_s120_standalone.py`
133/133, `test_s120_union.py` 2593/2593, `union_defensibility.py` 40/40,
`test_s110_shortlist.py` 235/235.

**A note on speed, since it was checked before it was claimed.** The deterministic layer
first reads 3.4 s on bigbluebutton, which is WordNet's one-time corpus load and not the
relation: warm, the whole scan-and-evidence pass over all five projects is **513 ms**.
Computing the evidence once instead of twice is worth ~250 ms a run and is done for
readability, not for speed.

## Cost

210 judging calls in one invocation. No E2E was bought: every arm is either refused or
gold-neutral-and-not-adopted, so the head does not move and there is nothing to compose.
