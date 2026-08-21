# The static round — does the authored text state a concept or a recipe?

The compaction round answered "can the prompts be smaller" and found the answer was
not in the prose: authored rules are **5.3%** of a full-name judging call and **4.3%**
of a resolver call, and what is big is repetition. It removed the repetition and
deliberately left every authored word alone.

This round asks the question that is left, and it is not a size question. **Does each
static clause state a concept a paper can defend as general, or a recipe fitted to the
surfaces these five documents happen to show?** A recipe is a leakage-shaped liability
even when GATE-06 passes, because it names no benchmark word and still encodes what
the benchmark looked like. Every arm here is therefore a **paraphrase**: the clause
keeps its job and loses its recipe. A neutral verdict adopts the arm. A negative one is
the more interesting outcome — it says the recipe was doing work the concept does not,
which is a finding about the method, not about the wording.

Total authored text: **3079 B**, unchanged since `s_linker89`.

## Which clauses read as recipes (`pilot/static_audit.py`, no LLM calls, no checkpoints)

Two deterministic screens over the ten static constants: clauses that legislate an
**orthography or syntax** (spacing, hyphenation, capitalization, dotted paths) rather
than say what a thing is, and clauses that **enumerate instances** where a concept
would name the class.

| constant | B | family, calls/run | flagged clause |
|---|---|---|---|
| `ENTITY_EXTRACTION_RULES` | 518 | full-name extraction, 9.0 | SURFACE: "spelled as… spells it", "different **spacing, hyphenation or compound joining**" |
| `STRICTER_CLAUSE` | 384 | lenient judging, 8.7 | SURFACE: "**Capitalization** is evidence for a name and its absence is evidence against" |
| `QUALIFIED_CLAUSE` | 166 | judging + extraction + denotation, 22.7 | SURFACE: "longer **joined or dotted** identifier" |
| `ALIAS_EXCLUSION_RULES` | 133 | alias proposal, 5.0 | SURFACE: "**compound** or qualified name" |
| `LAYERED_COREF_RULES` | 666 | strict judging, 3.6 | ENUM(4): "— the data, the artifact, the request, the result —" |
| `DOC_KNOWLEDGE_EXTRACTION_RULES` | 240 | alias proposal, 5.0 | ENUM(3): "(introduced short forms, alternate names, or words of multi-word names…)" |

And the same principle authored twice, ranked by content-word overlap of the clause
pair (Jaccard over non-stopword lemmas):

| J | pair | shared content |
|---|---|---|
| 0.50 | `DOC_KNOWLEDGE_EXTRACTION_RULES`#2 ↔ `STRICTER_CLAUSE`#1 | english, ordinary, use |
| 0.19 | `ALIAS_EXCLUSION_RULES`#1 ↔ `QUALIFIED_CLAUSE`#1 | identifier, longer, part |
| 0.14 | `COREF_VALIDATION_FOCUS`#1 ↔ `LAYERED_COREF_RULES`#2 | architectural, referring |

(The J = 1.00 pair the audit reports — "When uncertain, prefer APPROVE" against "When
uncertain, reject" — is the two gates' *opposite* defaults sharing one content word.
It is the layering, not a duplication.)

## What population each recipe speaks to (`pilot/static_screen.py`, no LLM calls)

Read off the head's own end-to-end batch, `results/compact_e2e_terra_r{1,2,3}_20260821`,
three runs, variant `s_linker89`. A clause with an empty population can be generalized
on the text alone; a clause with gold in its population needs a paired arm on both
models before a word of it moves.

**S1 — how the sentence writes the name** (`STRICTER_CLAUSE`#3, `ENTITY_EXTRACTION_RULES`#1)

| the writing | /run | gold | judge approved |
|---|---|---|---|
| exact, as the COMPONENTS list gives it | 106.3 | 103.0 | 104.3 |
| via a known alias only | 38.7 | 24.7 | 30.3 |
| **case only** | **28.7** | **24.7** | 24.7 |
| **separators (spacing/hyphen/compound)** | **1.0** | **1.0** | 1.0 |

So the capitalization sentence speaks to a real population — 28.7 candidates a run
whose case the model does not use, 24.7 of them gold — and the three-operation
spelling recipe licenses **one candidate a run**, which is gold. Neither can be
deleted; both can be *stated as concepts*, which is what the arms do.

**S2 — the span inside a longer identifier** (`QUALIFIED_CLAUSE`, `ALIAS_EXCLUSION_RULES`)

Counting only candidates where **every** writing of the name in the sentence is
embedded, since the clause says "occurs *only* as part of":

| | /run | gold |
|---|---|---|
| only inside a **joined** identifier | 13.3 | **13.0** |
| only inside a **dotted** path | 5.7 | 2.0 |
| embedded somewhere, but also written free | 2.0 | 2.0 |
| only inside a longer word | 1.0 | 1.0 |

**This is the round's first real finding.** The clause tells the judge to discount an
expression inside a "joined or dotted" identifier — and the *joined* population is
98% gold, because a component name written as one word **is** the whole name, not a
piece of a longer one. The clause survives only because a reader disambiguates
"joined" correctly against the rest of the sentence ("naming a piece of that
identifier"). That is exactly the defensibility problem this round is about: the
wording over-reaches and the model's charity covers for it. The paraphrase says what
the clause means — a **fragment** of a longer identifier.

**S3 — what the strict judge's objections lean on** (`LAYERED_COREF_RULES`#4)

| the objection's ground | /run | gold |
|---|---|---|
| some other objection | 122.0 | 70.7 |
| cited via a **listed word** only | 10.7 | **3.0** |
| cited via the **principle** only | 0.7 | 0.0 |
| listed word and principle | 0.7 | 0.0 |

The list is what the judge reaches for: 10.7 objections a run use one of the four
nouns and 0.7 state the ground without one. Those rejections cost 3.0 gold a run, and
last round deleting the whole ground cost luna 6.7 gold resolutions a run — so the
clause is load-bearing and the *list* is how it is being applied. Dropping only the
instances is a genuine risk and is paired on both models.

**S4 — what the alias extractor returns** (`DOC_KNOWLEDGE_EXTRACTION_RULES`#1)

| shape | /run |
|---|---|
| a short form of the name | 17.3 |
| an alternate name sharing no word with it | 11.7 |
| an initialism | 1.0 |
| **a word of a multi-word name** | **0.0** |

The parenthetical lists three shapes; the third is returned zero times a run. The
paraphrase names the condition all of them satisfy instead of listing them.

## The arms (`pilot/static_pilots.py`)

Each swaps module constants on the head and re-runs the one stage that reads them,
over the head's own checkpoints, every other stage held at what that run recorded.

| group | arm | constant | B | what changes |
|---|---|---|---|---|
| `qual1` | `genqual` | `QUALIFIED_CLAUSE` | 166 → 156 | "joined or dotted identifier" → "fragment of a longer identifier" |
| `qual1` | `genform` | `STRICTER_CLAUSE` | 384 → 354 | "Capitalization is evidence…" → "How the word is written is evidence either way and never settles it" |
| `qual1` | `mergeord` | `STRICTER_CLAUSE` | 384 → 363 | opens with the shared use/mention sentence the alias stage would also carry |
| `strict1` | `genartifact` | `LAYERED_COREF_RULES` | 666 → 612 | the four-noun list deleted, the ground kept verbatim |
| `extract1` | `genextract` | `ENTITY_EXTRACTION_RULES` | 518 → 508 | the three-operation spelling recipe → "differs only in how the name's own words are separated" |
| `extract1` | `genqual` | `QUALIFIED_CLAUSE` | 166 → 156 | the same paraphrase, priced at its second consumer |

`check_parity()` asserts the control renders the head's exact constants and that every
arm differs on the key it names — a misspelled key would otherwise measure nothing.

**Byte change is not the point and is reported only for completeness: −136 B of 3079
if every arm is adopted.** What changes is that six clauses stop describing surfaces
and start describing concepts.

## The stage arms, three paired runs a side on both models

Every arm judges (or extracts from) the same recorded checkpoints, with the two
stages it does not touch held at what that run recorded, so the paraphrase is the
only difference. Statistics are the branch's paired sign-flip permutation test,
p floor 0.10 at n = 3 (`pilot/static_round_stats.py`).

### The lenient judging prompt (`qual1`), 8.7 calls a run on terra, 9.3 on luna

| model | arm | stage gold | stage spurious | composed F1 | composed F2 | composed TP | composed FP |
|---|---|---|---|---|---|---|---|
| terra | `ctl` | 151.3 | 8.3 | 94.32 | 95.82 | 186.7 | 23.7 |
| terra | `genqual` | 152.0 | 10.0 | 94.21 | 95.90 | 187.3 | 25.3 |
| terra | `genform` | 151.7 | 10.3 | 93.84 | 95.69 | 187.0 | 25.7 |
| terra | `mergeord` | 152.0 | 8.7 | 94.08 | 95.72 | 186.7 | 23.7 |
| luna | `ctl` | 152.3 | 19.7 | 89.15 | 91.82 | 178.7 | 47.0 |
| luna | `genqual` | 153.0 | 22.0 | 89.05 | 91.91 | 179.3 | 48.7 |
| luna | `genform` | 153.0 | 21.7 | 89.07 | 91.91 | 179.3 | 48.7 |
| luna | `mergeord` | 153.0 | 18.7 | **89.50** | **92.09** | 179.3 | **45.7** |

**All three arms are QUALITY-NEUTRAL on both models, at stage and composed** — the
lowest p over the twelve readings is 0.60, six times the floor. Every arm reads
+0.3 to +0.7 gold on both models; `genqual` and `genform` cost 2 spurious a run and
`mergeord` costs none, and on luna `mergeord` is the only arm of the three that
*reduces* false positives (−1.3), which is why it is the one composed first.

### The strict judging prompt (`strict1`), 9.0 calls a run on terra, 12.7 on luna

| model | arm | stage gold | stage spurious | composed F1 | composed F2 | composed TP | composed FP |
|---|---|---|---|---|---|---|---|
| terra | `ctl` | 39.0 | 3.7 | 93.75 | 95.37 | 185.7 | 25.3 |
| terra | **`genartifact`** | **41.0** | 4.0 | **94.07** | **95.63** | 186.3 | 25.7 |
| luna | `ctl` | 55.0 | 5.3 | 89.45 | 91.62 | 177.0 | 44.0 |
| luna | **`genartifact`** | **56.3** | 5.7 | **89.52** | **91.77** | 177.7 | 44.3 |

QUALITY-NEUTRAL on both models (lowest p 0.80), and **this is the round's
substantive finding**. Last round's `noartifact` deleted this clause's whole ground
and luna lost **6.7 gold resolutions a run**. This arm keeps the ground word for word
and deletes only the four-noun list, and luna reads **+1.3 gold**.

**A clause that carries a concept and a clause that supplies vocabulary to quote are
not the same thing, and the earlier round could not tell them apart.** The screen
showed the judge citing a listed noun in 10.7 objections a run against 0.7 stating
the ground without one — which reads like the list is doing the work, and is exactly
the inference this arm refutes. Asked to state the ground in its own words, the judge
finds at least as many correct rejections on both models. Only the concept is
load-bearing; only the concept is defensible in a paper.

### The full-name extraction prompt (`extract1`), 17.7 calls a run

| model | arm | stage gold | stage spurious | composed F1 | composed TP | composed FP |
|---|---|---|---|---|---|---|
| terra | `ctl` | 149.7 | 7.3 | 94.01 | 184.3 | 22.7 |
| terra | `genextract` | 149.0 | 7.7 | 94.12 | 185.0 | 23.0 |
| terra | `genqual` | 149.7 | 8.0 | 94.07 | 184.0 | 23.3 |
| luna | `ctl` | 151.3 | 14.0 | 89.55 | 178.0 | 41.7 |
| luna | **`genextract`** | **147.7** | 13.7 | 88.97 | 176.0 | 41.0 |
| luna | `genqual` | 150.0 | 15.0 | 89.24 | 178.3 | 42.7 |

**`genextract` is refused.** Terra reads it neutral on every statistic; luna reads
**stage gold −3.7 at p = 0.10**, exactly the n = 3 floor, which is QUALITY-CHANGING,
and this branch reads the stage before the composition. `genqual` is neutral on both
models at this, its second consumer.

**The refusal is the round's second finding, and it is about the screen, not the arm.**
S1 priced this clause by the population it *admits* — 1.0 candidate a run written with
the name's words joined differently, and it is gold — and concluded the recipe was
nearly free. But the paraphrase also loosened "spelled as the COMPONENTS list **spells**
it" to "as the list **gives** it", and on the laxer model that costs 3.7 gold candidates
a run. **A checkpoint screen can price what a clause admits; it cannot price the
strictness the clause's wording imposes on everything else.** `genqual` changing the
same prompt neutrally shows this is the specific wording, not paraphrasing as such.

### The alias family (`alias1`), 27–28 calls a run

An alias arm cannot be read at its own stage — its output is a vocabulary, not links —
so each arm learns its own document knowledge and is read at that knowledge's consumer,
the full-name extraction and judging pair, composed with the recorded other two stages.

| model | arm | stage gold | stage spurious | composed F1 | composed TP | composed FP |
|---|---|---|---|---|---|---|
| terra | `ctl` | 145.7 | 11.3 | 92.48 | 178.7 | 26.3 |
| terra | `genalias` | 147.7 | 9.3 | 93.36 | 182.3 | 24.7 |
| terra | `mergefrag` | 153.0 | 11.7 | 94.28 | 190.0 | 27.0 |
| terra | `mergeord` | 145.7 | 6.0 | 93.13 | 181.0 | **21.3** |
| luna | `ctl` | 152.0 | 19.3 | 88.40 | 178.0 | 46.7 |
| luna | `genalias` | 153.3 | 21.3 | 88.84 | 179.0 | 48.0 |
| luna | `mergefrag` | 153.0 | 18.3 | 88.60 | 178.7 | 45.3 |
| luna | `mergeord` | 151.7 | 13.0 | **89.36** | 177.3 | **40.3** |

**All three are QUALITY-NEUTRAL on luna.** On terra all three read *above* control and
two are flagged QUALITY-CHANGING in the positive direction (`mergefrag` stage gold
+7.3, p = 0.10; composed TP +11.3, p = 0.10).

**Those positive flags are not claimed as improvements, for a reason this group makes
visible.** Every arm here re-learns document knowledge, so the knowledge stage's own
stochasticity sits inside the comparison — and it is large. This group's control reads
**145.7** stage gold on terra where `qual1`'s control, same model and same judge, reads
**151.3**; the only difference is that `qual1` reuses the knowledge the run recorded.
A fresh knowledge draw is worth about **−5.6 gold** on its own, which is larger than two
of the three arm effects here. The defensible statement is that no alias paraphrase
costs anything on either model, not that `mergefrag` gains eleven links.

`mergeord` is the one arm that reduces false positives in both groups it appears in and
on both models (`qual1` luna FP −1.3, `alias1` luna FP −6.4, `alias1` terra FP −5.0),
which is why it is the one composed.

### The merged-and-generalized alias rule (`alias2`) — refused

`genalias` and `mergeord` both rewrite `DOC_KNOWLEDGE_EXTRACTION_RULES` and cannot
compose, so the head needs one constant doing both jobs. `mergealias` is that text:
the three-shape enumeration replaced by the condition all three satisfy, and the
use/mention principle stated once and shared with the judging prompt.

| model | arm | stage gold | stage spurious | composed F1 | composed F2 | composed TP | composed FP |
|---|---|---|---|---|---|---|---|
| terra | `ctl` | 147.7 | 11.7 | 92.63 | 93.89 | 181.0 | 27.0 |
| terra | `mergealias` | 143.7 | 8.0 | 91.81 | **92.51** | 179.3 | 23.3 |
| luna | `ctl` | 150.7 | 24.0 | 87.71 | 90.70 | 176.7 | 51.0 |
| luna | `mergealias` | 152.3 | 22.0 | 88.91 | 91.92 | 179.7 | 49.3 |

Neutral at the stage on both models and neutral throughout on luna, but terra's
composed reading is flagged with **macro F2 p = 0.20 in the negative direction** — the
only downward flag any alias arm produced. `genalias` achieves the generalization
without it, so `mergealias` is refused and the use/mention sentence is shared only into
`STRICTER_CLAUSE`.

**A note on the verdict rule, because it is what flags these.** `pilot/ab_stats.py`
calls an arm QUALITY-NEUTRAL when **every** statistic has p > 0.20 — it is not the
0.10 permutation floor that decides. At n = 3 the floor is 0.10, so a p of 0.10 or 0.20
is the strongest evidence the design can produce, and both count as changing.

**And a caution this round earned.** The `alias1` and `alias2` controls are the same
specification measured twice, and they read **145.7** and **147.7** stage gold on terra
— a 2.0 spread between identical arms. Every arm in these two groups re-learns document
knowledge, so the knowledge stage's own stochasticity sits inside every comparison. All
six alias readings are inside that spread.

## What is adopted

| constant | adopted form | measured by | terra | luna |
|---|---|---|---|---|
| `QUALIFIED_CLAUSE` | fragment, not "joined or dotted" | `genqual`, at **both** consumers | neutral | neutral |
| `LAYERED_COREF_RULES` | ground kept, four-noun list dropped | `genartifact` | neutral, +2.0 gold | neutral, +1.3 gold |
| `STRICTER_CLAUSE` | "how the word is written", shared use/mention opening | `mergeord` | neutral | neutral, FP −1.3 |
| `DOC_KNOWLEDGE_EXTRACTION_RULES` | the condition, not the three shapes | `genalias` | above ctl | neutral |
| `ALIAS_EXCLUSION_RULES` | the judging prompt's own fragment clause | `mergefrag` | above ctl | neutral |
| `ENTITY_EXTRACTION_RULES` | **unchanged** | `genextract` **refused** | neutral | **−3.7 gold, p = 0.10** |

Composed as **`s_linker90`**. `pilot/test_s90_static.py` asserts the composition in
**90 checks**: every other module constant byte-identical to s89's, each adopted
constant byte-identical to the arm text that was measured, every class method's source
identical modulo the rename, the lenient judging / strict judging / alias-extraction /
full-name-extraction prompts of all five projects equal to s89's with exactly those five
substitutions, and no adopted clause writing a benchmark component's word (GATE-06).

Authored text **2107 → 2053 B**. The byte figure is incidental — two of the five
constants *grew*, because merging states a principle once and then refers to it.

## Reproducing

```bash
# the deterministic work, no LLM calls
../.venv/bin/python pilot/static_audit.py
../.venv/bin/python pilot/static_screen.py --variant s_linker89 \
    "compact_e2e_terra_r*_20260821"

# one stage group, one model (arms are paired inside the invocation)
OPENAI_API_KEY="$OAI_KEY" LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_SERVICE_TIER=flex OPENAI_REASONING_EFFORT=none AB_OUT=../results/static_round \
  ../.venv/bin/python pilot/static_pilots.py --group qual1 --model terra --runs 3
../.venv/bin/python pilot/static_round_stats.py --group qual1 --model terra

# the composed head's invariants, and its end-to-end batch
../.venv/bin/python pilot/test_s90_static.py
TYPED_E2E_CONTROL=s_linker89 TYPED_E2E_TAG=static bash pilot/run_typed_e2e.sh \
  s_linker90 terra luna
../.venv/bin/python pilot/score_runs.py \
  --arm s_linker89 ../results/static_e2e_terra_r{1,2,3}_* \
  --arm s_linker90 ../results/static_e2e_terra_r{1,2,3}_*   # and luna
```

## `s_linker90` end to end — refused on both models

`TYPED_E2E_CONTROL=s_linker89 TYPED_E2E_TAG=static bash pilot/run_typed_e2e.sh
s_linker90 terra luna`, three paired runs a side, both arms in every invocation, all
six clean. `results/static_e2e_{terra,luna}_r{1,2,3}_20260821`.

| model | arm | TP | FP | macro F1 | macro F2 |
|---|---|---|---|---|---|
| terra | `s_linker89` | 179.0 | **21.0** | **93.23** | 93.57 |
| terra | `s_linker90` | 181.0 | 38.3 | 90.72 | 93.05 |
| terra | delta | +2.0 (p = 0.80) | **+17.3 (0.10)** | **−2.5 (0.10)** | −0.5 (0.70) |
| luna | `s_linker89` | **180.0** | 51.3 | **89.0** | **91.8** |
| luna | `s_linker90` | 171.3 | 45.3 | 87.7 | 89.6 |
| luna | delta | **−8.7 (0.10)** | −6.0 (0.40) | −1.3 (0.30) | **−2.2 (0.10)** |

**QUALITY-CHANGING on both models, downward on both. The composed head is refused.**

**Five paraphrases that are each neutral alone are not neutral together.** That is the
round's headline and it is a result about method, not about wording: this branch has a
law that a clause is not independently priceable, and the same holds one level up — an
*arm* is not independently adoptable. Nothing here was adopted on weak evidence; all
five held on both models at the stage they were measured.

The two models fail differently, which is itself informative:

| | terra | luna |
|---|---|---|
| shape | precision collapse | recall loss |
| where | teammates alone: FP +17.0, F1 −12.3 | bigbluebutton −6.0 TP, teammates −3.0 TP |
| source | **58 of 66 teammates FPs are `partial_name`**, where s89 produces **0** | FP also falls (−6.0): the head is simply more conservative |

### The missing consumer, and what it turned out not to explain

`QUALIFIED_CLAUSE` enters **three** prompt families — lenient judging, full-name
extraction, and denotation — which the round's own byte inventory states (22.7 calls a
run). `genqual` had been measured at the first two and never at the third, and the third
is the `partial_name` stage the terra failure comes from. `denot3` is that missing group:
`genqual` and the alias pair (`genalias` + `mergefrag`), priced at the denotation stage
on both models.

| model | arm | stage gold | stage spurious | composed F1 | composed TP | composed FP |
|---|---|---|---|---|---|---|
| terra | `ctl` | 20.7 | 13.3 | 94.17 | 186.3 | 24.3 |
| terra | `genqual` | 19.7 | **11.0** | **94.46** | 186.3 | **22.0** |
| terra | `aliaspair` | 20.7 | 11.7 | 94.39 | 186.3 | 22.7 |
| luna | `ctl` | 12.3 | 23.0 | 89.82 | 180.7 | 43.3 |
| luna | `genqual` | 11.7 | **20.3** | **89.95** | 180.0 | **41.0** |
| luna | `aliaspair` | 9.3 | 21.3 | 89.32 | 176.7 | 41.7 |

**Both arms are QUALITY-NEUTRAL at the denotation stage on both models, and `genqual` is
the better arm there on both** — fewer spurious denotations and fewer composed false
positives, the exact opposite of the E2E failure's direction.

**So the missing group does not explain the failure, and the first reading of it was
wrong.** With `denot3` the round has priced every one of the four consumers these five
constants have, on both models, and **all eleven stage readings are neutral**. The
composed head still loses on both. The failure is therefore not any clause at any
consumer: it is a cross-stage interaction, where `partial_name` sees a different upstream
state because `full_name` has already answered differently. No single-stage arm can see
that by construction — each holds the other stages at what the recorded run produced.

**This is the stronger version of the finding, and it is the one to write down.** It is
not "an arm was mispriced". It is: *every constituent was measured neutral at every
consumer it has, on both models, and the composition still moved.*

## `s_linker91` — the minimal head the step-3 gate permits

If arms cannot be composed on stage evidence alone, the composition needs its own
evidence. The branch's step-3 gate asks what the arms' changed link sets have in common:

| model | `mergeord` changes | `genartifact` changes | pairs **both** touch |
|---|---|---|---|
| terra | 2.3 /run (+1.7 / −0.7) | 21.0 /run (+11.7 / −9.3) | **0.0** |
| luna | 3.0 /run (+1.3 / −1.7) | 23.7 /run (+12.7 / −11.0) | **0.0** |

These two are the only adopted constants with **exactly one consumer each**
(`STRICTER_CLAUSE` → lenient judging, `LAYERED_COREF_RULES` → strict judging), neither
feeds a vocabulary to a later stage, and they touch disjoint link sets on both models.
`QUALIFIED_CLAUSE` has three consumers and the two alias constants feed every stage
downstream of them, which is what `s_linker90` composed and lost.

`s_linker91` is `s_linker89` plus those two paraphrases and nothing else.
`pilot/test_s91_static.py` asserts it in **84 checks**. Its end-to-end batch, three
paired runs a side against `s_linker89`, is tag `minimal`.

### `s_linker91` end to end — also refused

Three paired runs a side against `s_linker89`, tag `minimal`.

| model | arm | TP | FP | macro F1 | macro F2 |
|---|---|---|---|---|---|
| terra | `s_linker89` | 180.3 | 31.3 | 91.6 | 93.1 |
| terra | `s_linker91` | 179.7 | **26.7** | **92.7** | **93.5** |
| terra | delta | −0.7 (p = 0.80) | −4.7 (0.30) | **+1.1 (0.20)** | +0.4 (0.60) |
| luna | `s_linker89` | **180.7** | **43.7** | **90.0** | **92.5** |
| luna | `s_linker91` | 177.7 | 50.7 | 88.3 | 91.2 |
| luna | delta | −3.0 (0.70) | **+7.0 (0.10)** | −1.8 (0.20) | −1.2 (0.40) |

**Terra's only flag points up (macro F1 +1.1) and luna's points down (FP +7.0,
macro F1 −1.8). Refused: it does not hold on both.**

Two constants, one consumer each, disjoint changed link sets, both neutral at their
stage on both models — and the pair still moves in opposite directions on the two
models end to end. The step-3 gate was satisfied and did not predict the outcome.

## `s_linker92` — one constant, the round's last candidate

If two constants cannot be composed, the question left is whether *one* can be adopted
at all. `s_linker92` is `s_linker89` with **only** `LAYERED_COREF_RULES` paraphrased:
the four-noun list deleted, the ground kept word for word. It is the round's most
valuable single generalization — an enumeration is the clearest recipe in the module —
and its stage evidence is the round's strongest (terra gold 39.0 → 41.0, luna 55.0 →
56.3, neutral on both). One consumer: the strict judge. `pilot/test_s92_static.py`
asserts it in 82 checks. Batch tag `solo`.
