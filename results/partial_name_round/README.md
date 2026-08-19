# A better partial-name linker — 2026-08-14

The merged-alias round ended with the partial-name linker named as the frontier: the two
projects whose partial-name linker fires carry essentially all of the workflow's false
positives (teammates FP 6.7, bigbluebutton FP 5.2, against 0.0 / 0.3 / 0.0 elsewhere),
and every attempt to change the alias table failed *through* that linker's suppression
role. So this round audits it directly, before proposing anything.

## The stage's error budget, measured first

`pilot/partial_audit.py`, six runs of `../results/s5960_e2e_r*_20260813`, arm s49:

| project | open gold | proposed | of which gold | approved | TP | FP | judge recall | judge precision |
|---|---|---|---|---|---|---|---|---|
| teammates | 10.5 | 30.0 | 3.0 | 2.7 | 2.2 | 0.5 | 72% | 81% |
| bigbluebutton | 21.0 | 30.3 | 15.7 | 18.5 | 15.5 | 3.0 | 99% | 84% |
| mediastore / teastore / jabref | 4.0 / 6.0 / 0.0 | 0 | 0 | 0 | 0 | 0 | — | — |
| **total** | **41.5** | **60.3** | **18.7** | **21.2** | **17.7** | **3.5** | **95%** | **83%** |

**A perfect judge over the same candidates would be +1.0 TP and −3.5 FP.** That is the
whole of the judging headroom, and it says the denotation judge is not the bottleneck —
the proposer is. 41.5 gold pairs are still open when this linker runs and it offers 18.7.

## Where the other 22.8 go

`pilot/partial_gap.py` attributes every open gold pair the proposer declines, by the
deterministic reason `_name_word_candidates` declined it, and cross-references the final
link set:

| reason | per run | recovered later | lost |
|---|---|---|---|
| `no_hook` — no sentence word relates to the name at all | 15.0 | **15.0** | 0.0 |
| `states_a_name` — the sentence states a whole name | 5.8 | 0.2 | **5.7** |
| `ambiguous` — a hook exists, but two components own it | 2.0 | 0.0 | **2.0** |
| total | 22.8 | 15.2 | 7.7 |

The 15.0 are not a loss: the coreference linker recovers **every one** of them. That is
the division of labour working, and it is worth stating because a naive reading of the
proposer's recall would have called it a 22.8-pair hole.

The remaining 7.7 are lost outright — no stage of the workflow sees them again. Against
a total residual recall loss of 8.0 gold pairs per run, **this one stage's two declines
are the pipeline's entire remaining recall loss.**

## The bigger bucket, and why its obvious repair fails

`states_a_name` is a hand-off: the sentence states a whole name, so the pair belongs to
the full-name linker. `pilot/partial_hole.py` splits it by what the full-name stage
actually did:

    3.0/run   the extraction call never proposed the pair
    2.7/run   it did, and a full-name judge rejected it
    0.2/run   the coreference linker recovered it anyway

so the hand-off is unconditional while the recipient is not. The obvious repair is to
defer only where the full-name stage actually *ruled* on the pair. Measured
deterministically over all six runs, that is **+0.7 gold and +10.0 spurious**:

    6x  teammates s196 Client 'client'     6x  teammates s117 Logic 'logic'
    6x  teammates s49  Common 'common'     6x  teammates s79  Logic 'logic'
    6x  teammates s173 Test Driver         5x  teammates s139 GAE Datastore 'GAE'
    3x  GOLD teammates s7 Logic 'logic'    1x  GOLD teammates s87 Logic 'Logic'

Refuted, and the reason is the one this branch keeps re-finding: the whole-name test is
not only a hand-off, it is **the alias table doing suppression work**. Almost every
sentence containing `logic` states a discovered alias of `Logic`, so conditioning the
test on consideration turns the alias table off.

## The smaller bucket is a defect in the ownership test

`_name_word_candidates` decides that a sentence word is a word of a component's name
with `surface.startswith(word)`, in both directions. `WebRTC` is therefore owned by
`WebRTC-SFU` (exactly) *and* by `BBB web` (as a continuation of `web`), the proposer
requires a unique owner, and the pair is dropped. Two gold links go with it every run.

`pilot/partial_screen.py`, over all five documents (base 60.3 candidates, 18.7 gold):

| repair | gold | spurious |
|---|---|---|
| exact word match outranks a prefix-only one | +2.0 | +1.0 |
| **prefix must be an English inflection** | **+2.0** | **+0.0** |

The inflection bound dominates: the same two gold candidates, and it also drops
`webcams -> BBB web`, because neither `rtc` nor `cams` is an inflection of `web`. It is
also what the old docstring already claimed — *"a sentence word that begins with a name
word is accepted, so inflected forms pass without a suffix list"* — so naming the nine
endings states the intent instead of approximating it. English morphology only; GATE-06
is asserted in the test suite against every component name and every document.

With the real denotation judge behind it, five samples a side
(`pilot/partial_pilots.py --pilot proposer`): **TP +2.0 (p = 0.01), FP +1.0 (p = 0.01)**.
bigbluebutton reaches 61.7 of its 62 gold pairs. This is `s_linker62`.

## A real bug, whose repair costs precision

Auditing the same predicate turned up a defect in `_inside_qualified_identifier`:

```python
before = text[start - 1] if start else ""
joined = (before in "-_" or ...)          # "" in "-_" is True
```

`"" in "-_"` is `True` in Python, so every span starting at a sentence's first character
— and every span ending at the document's last — has been reported as sitting inside a
qualified identifier and dropped, by every variant in this branch. **344 spans per run
across the five documents.** A sentence-initial component name is invisible to this
proposer.

Repairing it is **TP ±0.0 (p = 1.00), FP +1.2 (p = 0.01)** at the stage
(`--pilot guard`): the two candidates it un-hides are both spurious and the judge
approves them. On this benchmark the defect is load-bearing. `s_linker63` carries the
repair so the defect is priced rather than quietly kept, and the end-to-end number
decides which spelling the paper artifact carries.

## Back to the bigger bucket, at the stage that owns it

The partial-name linker's hand-off is *correct*; the fix belongs where it points. For
the 3.0/run the extraction call never proposed, `pilot/statednet_screen.py` prices a
deterministic scan for a sentence that states the name, in three readings:

| reading | new pairs/run | gold | gold per pair |
|---|---|---|---|
| every discovered alias, case-insensitive | 41.2 | 3.0 | 0.07 |
| the model name, case-insensitive | 31.3 | 1.8 | 0.06 |
| **the model name, as spelled** | **1.2** | **1.0** | **0.86** |

The extraction call's own proposals run at 0.87 gold per pair, so only the third reading
is not a precision sacrifice. **Case is the entire design.** A component named `Common`
or `Client` matches ordinary English on every page; the capitalization is what separates
the proper noun from the common one. It is also the one site in this workflow where the
single lenient name primitive `_find_exact_form` is the wrong tool — everywhere else,
leniency was measured to be right.

Behind the unchanged two-pass full-name judge, five samples a side
(`--pilot statednet`): **TP +1.2 (p = 0.01), FP +0.4 (p = 0.44)**. This is `s_linker64`.

## The precision side has no lexical handle, checked and closed

The judge's 3.5 approved false positives per run are stable and look mechanical —
bigbluebutton s68 (`HTML5 Server` and `WebRTC-SFU`) and s18 (`HTML5 Server`) in 6 of 6
runs, from the words *server* and *SFU*. They are not separable by the matched word:

    bigbluebutton gold candidates:      server 7, client 8, clients 1
    bigbluebutton non-gold candidates:  conversion 5, server 2, web 2, presentation 2, sfu 1, ...

`server` and `client` are the *most* productive gold words and also appear among the
false positives, so no restriction on which words may be offered can separate them. A
longest-match rule was screened too — reject a candidate whose matched word sits inside
a longer phrase naming a *different* component, which would catch `Server` inside
`Kurento Media Server` — and its measured reach is **zero**: the sentence also contains
the free-standing phrase *"a media server"*, so the word occurs outside any other
component's name and the rule cannot fire. The precision side of this stage is a semantic
judgment, and the audit already bounds what perfect judging there would buy: −3.5 FP.

## The four arms

| variant | change | stage result |
|---|---|---|
| `s_linker59_null` | byte-identical to s59 | the in-set harness null |
| `s_linker62` | inflection-bounded prefix in the partial-name proposer | TP +2.0 (0.01), FP +1.0 (0.01) |
| `s_linker63` | s62 + the `"" in "-_"` boundary repair | TP ±0.0 (1.00), FP +1.2 (0.01) |
| `s_linker64` | s62 + the case-sensitive stated-name net | TP +1.2 (0.01), FP +0.4 (0.44) |

`pilot/test_s62_s63_proposer.py` pins all of them: the null differs from s59 in nothing
but its variant name, s62 differs in exactly one method body plus the new predicate and
renders every prompt byte-identically, s63 differs in exactly the span test and the two
spellings disagree on exactly the 344 sentence-boundary spans, s64 differs in exactly
the proposer chain, and each proposal set is asserted to be what the deterministic
screen measured — so a variant and its screen cannot drift apart.

## End to end, and the instrument that made it readable

Six paired runs, arms s59 / s59_null / s62 / s63 in one invocation
(`pilot/run_s6263_e2e.sh`, `../results/s6263_e2e_r*_20260814`):

| | TP | FP | macro F1 | macro F2 |
|---|---|---|---|---|
| s59 | 185.8 | 17.0 | 95.2 | 95.9 |
| **null arm** | +1.3 (p = 0.08) | −1.8 (0.44) | **+0.4 (0.35)** | +0.3 (0.26) |
| s62 | +1.8 (0.06) | +2.7 (0.34) | −0.1 (0.81) | +0.2 (0.57) |
| s63 | +1.7 (0.32) | +4.7 (0.10) | −0.6 (0.45) | −0.1 (0.98) |

**This invocation set is far noisier than the s5960 one** — s59's own macro F1 range
across the six runs is 4.44 and its FP is 17.0 against 12.2 for the same code on
2026-08-13, driven almost entirely by teammates (FP 9.7 here, 3.8 there). The null arm
moves +1.3 TP and +0.4 F1, so **no whole-pipeline delta of this size is readable in this
set**, in either direction.

The fix is the instrument this branch already built for exactly this: restrict the
permutation test to the links whose *source* the change can reach.

| | partial-name links only | | full-name links only | |
|---|---|---|---|---|
| | TP | FP | TP | FP |
| s59 | 16.2 | 4.7 | 154.3 | 9.5 |
| null arm | +1.2 (p = 0.13) | −1.0 (0.48) | −0.3 (0.82) | −0.7 (0.79) |
| **s62** | **+2.3 (p = 0.00)** | **+0.7 (0.67)** | −0.7 (0.51) | +1.0 (0.66) |
| s63 | +2.0 (0.13) | **+3.8 (0.03)** | −0.5 (0.72) | +0.8 (0.74) |

Produced by `pilot/source_stats.py --reachable-from partial_name`, which already splits
every comparison by link source and labels which stages the change can reach; full
output in `source_stats_s6263.txt`. The coreference source is neutral for all three arms
(TP +0.2 … +0.5, all p ≥ 0.57).

Two things follow, and the second validates the first:

- **s62 does exactly what the stage measured.** At the source it can reach, TP +2.3 at
  p = 0.00 with precision held (FP +0.7, p = 0.67) — the predicted +2.0/+1.0. Per
  project the gain lands where predicted: bigbluebutton TP 57.8 → 59.7. Its whole-
  pipeline macro F1 is *indistinguishable*, and that is a statement about this set's
  noise, not about the change.
- **The full-name column is neutral for every arm** (TP ±0.2, all p = 1.00), which is
  the control: neither change can reach that stage, and `stage_diff.py`'s ±4-link
  gained/lost swings there are drift. An instrument that reported those as real would be
  the wrong instrument.
- **s63 is refuted end to end.** The boundary repair costs FP +3.8 (p = 0.03) at the
  partial-name source, three times what the stage pilot predicted and the largest
  significant precision loss in the set. The `"" in "-_"` defect stays, documented and
  priced, rather than repaired at that cost.

**Adoption:** `s_linker62`'s inflection bound is taken — it is significant at its own
source, neutral everywhere else, +2 gold links per run, and it replaces an unbounded
approximation with the predicate the code already claimed. Its macro F1 claim is
"indistinguishable from s59 in a set whose null reads +0.4", not "better".

## And the end-to-end run was not needed, which is checkable

The reason this branch required end-to-end confirmation is one episode: dropping
`_keep_stated_names` was F2-positive on its own stage and quadrupled false positives
composed. That mechanism is `_unlinked` — every linker subtracts what earlier ones
produced, so a link admitted early is both locked into the union and **stolen from the
later, stricter linkers**. It is deterministic, and so is its precondition: a stage
change can only compose badly if the pairs it adds or removes are pairs a *later* stage
would otherwise have proposed.

`pilot/composition_check.py` reads that off the recorded checkpoints, with no calls:

| change | stage | added | removed | also proposed later | already in the final links | removed-but-linked |
|---|---|---|---|---|---|---|
| `infl` (s62) | partial_name | 3.0 | 1.0 | **0.0** | **0.0** | **0.0** |
| `statednet` (s64) | full_name | 1.2 | 0.0 | **0.0** | **0.0** | — |

**Composition risk 0.0 pairs per run for both.** Nothing either change adds is claimed by
any later stage, and nothing either removes was in the final link set — so the stage arm
*is* the pipeline answer, and a five-project end-to-end run measures the model's
run-to-run drift instead of the change. Batch 1 confirms it after the fact: s62's
per-source E2E reading (TP +2.3) reproduced its stage pilot (+2.0) while the macro F1 it
was mixed into read −0.1 against a null of +0.4.

So `s_linker64` is adopted on its stage evidence — **TP +1.2 (p = 0.01), FP +0.4
(p = 0.44)** behind the unchanged two-pass judge, composition risk zero — and the second
end-to-end batch was stopped after two runs. Those two runs are kept at
`../results/s64_e2e_r{1,2}_20260814` as corroboration, not as the test: on the stage each
arm changes, s62 reads partial-name TP 19.5 against s59's 15.5 at flat FP, and s64 reads
full-name TP 155.5 against 151.5 — but the null arm reads 156.0 on that same stage, which
is exactly why two runs decide nothing and the stage pilot does.

**Standing method for this workflow:** ablate at the stage that changed; run
`composition_check.py` to show the change cannot reach past it; only pay for end-to-end
runs when that check is non-zero.
