# The consolidation round (s109, s110) — two rounds composed, four ideas refused for free

The reading round and the regex round ran on different bases and neither was measured
against the other. This round asks what survives when they are put in one pipeline, and
answers every question at **level 1** — six recorded runs of `s_linker92a`, no LLM calls
spent. Four candidate changes are refused, one is adopted outright, and one is built and
left owing a stage pilot.

Tooling: `approach/pilot/consolidation_audit.py` (the five questions),
`approach/pilot/test_s109_nesting.py` (129 invariant checks, replays both scans pair by
pair). Data: `consolidation_audit.json`.

## What was refused, and what it cost to find out

Nothing. Every refusal below is a replay of recorded checkpoints.

### 1. The third blind proposer is redundant in front of a scan

`s_linker101` won the paired terra F2 comparison 3/3, so it was the natural base. It is
not, once the extractor is a scan. Its gold over `s_linker90`, and how much of that gold
the scan proposes by itself:

| set | model | gold s101 adds | scan reaches, canonical | scan reaches, with a recorded alias table |
| --- | --- | ---: | ---: | ---: |
| `f2round_e2e_luna_r1` | luna | 16 | 7 | 9 |
| `f2round_e2e_terra_r2` | terra | 5 | 3 | 4 |
| `f2round_e2e_terra_r3` | terra | 9 | 4 | 5 |
| `thirdlook_e2e_terra_r1` | terra | 11 | 6 | 10 |

Mean added gold 10.3, of which the scan reaches **7.0 (68%)** for no call at all. The
remainder is **3.3 pairs a run against a recorded null floor of TP 4.8** — it does not
clear noise. The third look costs ~4 extra calls a project and, on luna, took FP from 43
to 106. **The scan is the cheaper way to buy the same recall**, so `s_linker101` is
retired as a base and the two arms built on it (`s_linker107`, `s_linker108`) are
rebased or dropped.

### 2. Narrowing the resolver to no-name sentences, again refused — now on the scan base

`s_linker93`'s filter (ask the resolver only about sentences that write no name) saves
44% of the resolver's cases: 378 sentences a five-project run to 212. The reading round
refused it for a defect; this measures that defect on the scan base, where the
population is different:

| run | coref links | dropped | rescued by the scan | lost | gold lost | of which on a sentence naming only *another* component |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| luna r1 | 68 | 47 | 36 | 11 | 8 | 7 |
| luna r2 | 69 | 45 | 41 | 4 | 2 | 2 |
| luna r3 | 46 | 26 | 21 | 5 | 3 | 2 |
| terra r1 | 28 | 10 | 6 | 4 | 1 | 1 |
| terra r2 | 31 | 13 | 8 | 5 | 1 | 1 |
| terra r3 | 39 | 23 | 18 | 5 | 4 | 2 |

**3.2 gold a run, 2.5 of it the exact defect the reading round named.** The scan rescues
most dropped cases — it proposes the same pair through the named route — but not the
ones where the sentence names X and refers back to Y. Refused a second time, on a
second base. *An early version of this audit measured the filter per (sentence,
component) pair and read 0.7 gold; the filter is per sentence, and measuring the wrong
predicate flattered it by a factor of four.*

### 3. The sibling confusion is not one expression judged twice

The regex round's residual false negatives concentrate on sibling components at the
partial-name denotation judge, and that judge is target-blind: its case carries the
expression and the sentence, never the component. The obvious hypothesis is that one
expression reaching two components is shown to it as the same case twice, so it must
answer both the same way. **Refuted:** 3.2 shared `(sentence, quoted claim)` groups a
run, holding **0.0 TP and 0.8 FP**. A chooser over identical cases would have almost
nothing to choose.

### 4. A contrastive discriminator as a new stage — priced, not built

Splitting the partial-name gate's own decisions by whether a **code-enumerable sibling**
exists (catalog names sharing a signature word — a fact, not a list):

| | per run | with a sibling | in a group another member owns in that sentence |
| --- | ---: | ---: | ---: |
| false positives | 23.3 | 14.0 | **8.3** |
| false negatives at this gate | 6.2 | 4.8 | **2.3** |

A chooser would be asked 20.2 questions a run over 10 groups — one batched call a
project. Its **ceiling is −8.3 FP / +2.3 TP**, which is below the recorded FP floor of
10.7: an E2E could not resolve it, and only a stage pilot on fixed candidates could.
Recorded and not built, because question 5 found the cheaper repair.

## What was adopted: `s_linker109`

Reading the contested cases rather than counting them gives the mechanism, and it is not
a judging failure at all:

    bigbluebutton S46   candidate: Redis DB     judge quoted: "Redis PubSub"   participant
    bigbluebutton S48   candidate: Redis PubSub judge quoted: "Redis DB"       participant

The judge is right every time. The expression *is* a participant — it is simply a
different one. The partial-name scan proposed `Redis DB` because the sentence carries
the word `Redis`, and the only place it carries that word is inside the whole name of a
sibling. The judge cannot see this: it is target-blind by design, and telling it the
target is the refusal the design law records at −5.5 gold (s25).

**On terra the judge quotes the longer name it saw; on luna it quotes the bare shared
word (`Redis`, `HTML5`).** Same wrong pair, two different surfaces — which is exactly
why the repair cannot be a clause in either prompt.

`_scan` already refuses a pair for one reason of this kind — *"unless the sentence writes
a whole name of it: that pair is the full-name linker's."* `s_linker109` adds the sibling
case beside it: **if every writing of this component's word sits inside a span where the
sentence writes another component's whole name, the pair is that component's.**

| model | candidates | refused | partial links | links lost | **gold lost** | final FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| terra | 101.0 | 12.0 | 34.7 | 5.0 | **0.0** | **−5.0** |
| luna | 94.3 | 12.0 | 41.3 | 10.3 | **0.0** | **−10.3** |

Six runs of six, both models, zero gold. The refusal fires on **exactly 12 candidates in
every run of both models**: it reads the catalog and the document and nothing that is
sampled, so it is the one arm on this branch with no run-to-run band at all.

**No E2E is owed** (measurement policy, level 3): **0.0** of the removed links are
proposed by the coreference linker in any of the six runs, so freeing them from
`_unlinked` gives a later stage nothing to re-propose. This is the `s_linker85`
precedent — a deterministic relation settled by replaying the relation.

### The one place the alias table is not allowed in

The first version consulted N(c) — catalog names *and* discovered aliases — as the module's
scans do. It **cost 3 gold links in one luna run**, all three where that run's table bound
a document term to the *sibling* of the component the gold names. The fix is a principle,
not a threshold: the module's scans may use the alias table because a scan only ever
*admits* a case for a judge, and nothing in the deterministic layer admits a link. This
predicate is the one thing in the layer that **ends** a case, so it may rest only on what
is given. A catalog name is an input; the alias table is the output of an LLM stage that
varies by ~2.8 terms a run. **A discovered fact may open a case and may not close one.**

GATE-07: the ground is `QUALIFIED_CLAUSE`'s own — *"an expression that appears only as a
fragment of a longer identifier is naming a piece of that identifier"* — at the second
way names nest. Not a new rule; the same rule at an extent the module already scans.

## What is built and owing: `s_linker110`

`s_linker107`'s antecedent shortlist, rebased off the retired `s_linker101` onto the head.
Terra proposal stage, three samples × five projects: **spurious 27.0 → 17.0 at gold 45.6
→ 45.3**, precision 0.628 → 0.727. It is the only reading-round arm that moved spurious
without moving gold. **Level 2 owed on luna** (`pilot/reading_pilots.py --arm shortlist`);
composition risk is the resolver's usual one and is not zero.

## The transferable result: who enumerates the alternatives

Four arms asked one structural question — enumerate the alternatives, then commit — at two
places, and they only agree under one reading:

| where | who enumerates | result |
| --- | --- | --- |
| resolver (`s_linker106`) | the model | spurious **+6.6** |
| resolver (`s_linker107`) | code | spurious **−10.0** |
| lenient gate (`s_linker92e`) | nobody — quote the surface and stop | **refuted** |
| lenient gate (`s_linker92f`) | the model | **best terra macro F1**, FP below control |

**The alternative set is a fact when the case contains it and a weighing when it does
not.** Which components the sentences above name is a fact — `_states_a_name` computes it
exactly, and making the model re-derive it is asking it to do lookup with attention, which
it does by inventing. Which readings a lowercased word could have in its sentence is in no
table, so only the model can enumerate them; `s_linker92e` fails because it enumerates
nothing and merely echoes.

This is the branch's design law — facts in code, weighings in the prompt — applied to the
**alternative set** rather than to the rule, and it is what makes `s_linker106` and
`s_linker92f` agree instead of contradict. `s_linker109` is the same law at its limit: the
alternative set is a fact, *and the judge that would weigh it cannot be shown it*, so code
is not the better place for the distinction but the only one.

## The pipeline this leaves

| stage | proposer | judge | proposal calls |
| --- | --- | --- | ---: |
| full name | **scan** (`s_linker92a`) | lenient, per component | **0** |
| partial name | scan, **minus what another name covers** (`s_linker109`) | target-blind denotation | **0** |
| coreference | LLM resolver *(+ shortlist, owing)* | strict, rejects when uncertain | ~8 a project |

Two of the three proposers make no LLM call at all. Measured off the recorded runs
(`ablation_*.json`, per-project `llm_calls`), the pipeline is **75.3 calls a
five-project run against the pre-scan 83.2**, ~15 a project. Neither change this round
adopts or defers moves that number: `s_linker109` adds no call and removes none, and
`s_linker110` changes one prompt's text and not the number of times it is sent.

## What is owed

| | level | what would decide it |
| --- | --- | --- |
| `s_linker109` | **done** | nothing — level 1 decided it, composition gate clean |
| `s_linker110` | 2 | `pilot/reading_pilots.py --arm shortlist` on **luna**; terra is measured |
| the sibling chooser | 2 | a stage pilot on fixed candidates; an E2E cannot see −8.3 FP against a 10.7 floor |
| `s_linker92f` at the lenient gate | 4 | terra is clean (best macro F1, FP below control); luna trades 7.7 FP for **6.0 TP**, which an F2 budget does not buy |
| `s_linker92d`'s second fidelity | 4 | re-opened by the regex round at +2.0 gold / +0.3 non-gold on the tables actually run |
