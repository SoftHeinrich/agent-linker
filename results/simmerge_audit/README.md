# The similarity-merge audit — one SWATTR-style proposer in place of the two scans

**Question.** `s_linker110` proposes twice in code — `_extract_named_mentions` (the
sentence writes a whole name of the component, `ANY_CASE`) and `_scan` (the sentence
writes one word of a name at any WordNet reading, `ANY_WORD`, minus the pairs another
component's whole name covers) — and judges each stream with its own judge. SWATTR
(Keim et al., the ArDoCo lexical line) proposes **once**, with a fuzzy string relation.
Can one similarity relation produce the same candidates, or a superset of them, so that
the two stages collapse into one pipeline whose single judge is allowed to refuse?

**Answer, in one line.** The superset exists and is not worth buying: it **doubles the
judge's caseload (296 → 604 pairs a five-project run, 14 → 26 calls) to add three gold
pairs that the coreference linker already recovers in 6 recorded runs of 6, on both
models** — a marginal yield of zero. Merging the two *scans* into one stage, by
contrast, is free (same 296 cases, same 14 calls) and is a live option; what the
evidence refuses is merging the two *judges* behind them.

Tooling: `approach/pilot/simmerge_audit.py` (M1–M7, **no LLM calls**). Data: `audit.txt`
and the six CSVs beside it. Reproduce from `approach/`:

    ../.venv/bin/python pilot/simmerge_audit.py

## M1 — what SWATTR's relation actually is

Reimplemented from the ArDoCo tree (`core/framework/common`), at its shipped
`CommonTextToolsConfig.properties`, and checked by 16 assertions:

    areWordsSimilar(a, b)  =  same space-token count           (splitLengthTest)
                          AND ( a equalsIgnoreCase b           (EqualityMeasure)
                              | levenshtein(a, b) <= min(1, 0.9 * min|a|,|b|)
                                  — with containment required below length 2
                              | jaroWinkler(a, b) >= 0.90 )    (AT_LEAST_ONE)

    nameParts(name)  =  splitCamelCase(splitSnakeAndKebabCase(name)) + the identifier
                        e.g. UserDBAdapter -> [User, DB, Adapter, UserDBAdapter]

Two properties matter downstream. It is **looser than an edit-distance limit suggests**
— `Cache ~ Caches2` (two edits) passes on the Jaro-Winkler prefix boost, and so does
`Store ~ Storage`. And it is **case-sensitive where it is loose**: Jaro-Winkler gets the
raw terms, so a prefix that differs in case earns no boost.

Arms, all as sets of `(sentence, component id)` pairs: `sim_part` is ArDoCo's
`isWordSimilarToEntity` (a sentence word similar to a name part); `sim_whole` is its
phrase-level test (`areWordsOfListsSimilar` over every sentence window of the name's
width). Both read N(c) — the catalog name and the run's recorded aliases — as the head's
scans do.

## M2 — the yield table, five projects, 3697 possible pairs, 195 gold

| arm | pairs | gold | gold/pair | recall |
| --- | ---: | ---: | ---: | ---: |
| `full` (ANY_CASE whole name) | 215 | 154 | **0.716** | 0.790 |
| `partial` (ANY_WORD, refusal on) | 81 | 26 | 0.321 | 0.133 |
| **HEAD UNION** | **296** | **180** | **0.608** | **0.923** |
| `sim_whole` (fuzzy whole name) | 151 | 92 | 0.609 | 0.472 |
| `sim_part` (fuzzy name part) | 603 | 183 | 0.303 | 0.938 |
| **SIM UNION** | **603** | **183** | **0.303** | **0.938** |
| all pairs (sentence × component) | 3697 | 195 | 0.053 | 1.000 |
| SWATTR's published links | 188 | 148 | 0.787 | 0.759 |

This is `CLAUDE.md`'s name-relation table with the fuzzy rows added, and it lands in the
same order: **the looser the relation, the lower the gold per pair.** The similarity
union reaches 1.5 pp more of the gold than the head's union and is **2.04× the size**.

## M3 — containment: a superset, minus one pair

| | pairs | gold |
| --- | ---: | ---: |
| head union | 296 | 180 |
| sim union | 603 | 183 |
| sim adds | 308 | **3** |
| sim loses | **1** | 0 |
| gold neither scan reaches (coreference territory) | — | 12 |

**The one pair the similarity relation loses is the one the WordNet swap was adopted
for.** bigbluebutton S49, `Recording Service` against the sentence's *recorded*: a
lemmatizer applied to both sides makes `Recording` and `recorded` the same word
(`s_linker85`), while Jaro-Winkler sees no shared prefix at 0.90 and the edit distance
is 3. So **a fuzzy relation is not a superset of the head's by itself** — a merged
proposer would have to be `similarity ∪ lemma-word`, keeping the WordNet dependency it
was supposed to replace. (That pair is not gold; the loss is a property of the relation,
not a recall cost.)

## M4 — the form label: where the 308 added pairs sit

The merged candidate set, split by the **code-computed** form of the match — the fact a
single judge would have to be graded by:

| form | pairs | gold | gold/pair |
| --- | ---: | ---: | ---: |
| exact whole name | 215 | 154 | **0.716** |
| fuzzy whole name (not exact) | 9 | 0 | **0.000** |
| lemma word of name | 81 | 26 | 0.321 |
| fuzzy name part only | 299 | 3 | **0.010** |

Both fuzzy rows are empty of gold to three significant figures. The relation's extra
mass is **71× less gold-dense than the exact row it is meant to generalize**, and 32× less
than the lemma row. Per project the picture is the same everywhere: mediastore
104 fuzzy-only pairs / 1 gold, teastore 61 / 2, teammates 79 / 0, bigbluebutton 54 / 0,
jabref 1 / 0.

## M5 — cost, and SWATTR's own output as a floor

Judge cases and calls at `JUDGE_BATCH = 25`, five projects:

| arrangement | cases | calls |
| --- | ---: | ---: |
| today: two stages, two judges | 215 + 81 = 296 | 14 |
| **one stage over the head's union** | **296** | **14** |
| one stage over the similarity union | 604 | 26 |

**Merging the two scans is free in calls** — batching is per project and per stage, and
one stream of 296 cases fills the same 14 calls that two streams of 215 and 81 do
(2/1/6/4/1 a project either way). Merging in the similarity relation is **+86% judging
calls**.

SWATTR's published links (`sota-links/model-doc/swattr-*.csv`) put a floor under what
its real pipeline proposes, and the floor is entirely inside what the head already scans:

| | links | gold | in head union | in sim union | in neither |
| --- | ---: | ---: | ---: | ---: | ---: |
| all five projects | 188 | 148 | **187** | 188 | 0 |

The single link SWATTR emits that the head's scans do not propose is mediastore S37,
`Reencoding` against *re-encoding* — a hyphenation variant, the `ANY_SPELLING` row
`s_linker82` deleted — **and it is not gold**. In the other direction the head's scans
propose **32 gold pairs SWATTR never outputs**. The comparison is candidate-set against
published output, so it is one-sided by construction; it is reported because the
one-sidedness runs the way the head's design predicts.

## M6 — the two judges, and why the split is not decoration

Six recorded runs (`../consolidation_e2e_{terra,luna}_r{1,2,3}_20260825`), per
five-project run. **Both scans were recomputed with each run's own alias table and
matched the scan that run wrote, 60 of 60** — so every number above is checked against
the head's own bytes, not against a paraphrase of them.

| model | stage | cases | gold in | kept | TP | FP | keep rate | gold kept | precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| terra | full_name | 221.0 | 155.7 | 167.0 | 149.7 | 17.3 | 0.756 | 0.961 | 0.896 |
| terra | partial_name | 79.3 | 24.3 | 26.7 | 19.7 | 7.0 | 0.336 | 0.808 | 0.738 |
| luna | full_name | 221.0 | 163.7 | 203.7 | 162.3 | 41.3 | 0.922 | 0.992 | 0.797 |
| luna | partial_name | 84.7 | 16.3 | 25.7 | 11.0 | 14.7 | 0.303 | 0.673 | 0.429 |

The two streams are judged with **opposite defaults**: the lenient full-name judge keeps
76–92% of its cases, the target-blind denotation judge keeps 30–34% of its. That is not a
stylistic difference — it tracks the gold density of the streams (0.716 against 0.321),
and the uniform-schema round already paid to learn what happens when one default is
imported onto the other stream: `s_linker119`, the sortal gate replying in the lenient
gate's boolean, is **net −9.0 terra / −16.0 luna**, the worst arm of that round on both
models.

## M7 — the marginal gold is already on the board

Every gold pair the similarity relation adds, checked against the recorded final link
sets of six runs:

| pair | runs found | by |
| --- | ---: | --- |
| mediastore S28 `MediaAccess` | 6/6 | coreference |
| teastore S11 `ImageProvider` | 6/6 | coreference |
| teastore S24 `Persistence` | 6/6 | coreference |

**All three, in every run, on both models.** The similarity relation's entire recall
advantage is territory the coreference linker already covers — which is what the design
predicts, since that linker exists precisely to reach what no name relation reaches. Of
the 12 gold pairs *no* scan reaches, 11 are likewise found 6/6 by coreference and one
(teastore S26 `Persistence`) 2/6. **The proposer frontier is not where the remaining
recall is.**

## Verdict

1. **A single similarity proposer that supersets both scans exists** — `sim_part ∪
   sim_whole ∪ lemma-word`, since the fuzzy rows alone drop the `Recording/recorded`
   pair. **Refused**: +308 cases (+104%), +12 judge calls (+86%), +3 gold, and all 3 are
   already produced by the coreference linker in 6/6 runs. The exchange rate is
   1% gold per added case against a stream the head judges at 60%.
2. **A single proposer *at the head's own fidelity* is free and is the live option.**
   The two scans are already deterministic, already share `_name_spans`, and already
   cost zero calls; unioning them changes neither the candidate set nor the call count
   (296 cases, 14 calls either way). The merge that would actually simplify the module
   is *one scan emitting a `form` field*, not a looser relation.
3. **What must not be merged is the judging.** The form is a fact the code computes, and
   the two forms are judged at opposite defaults (keep rate 0.76–0.92 against 0.30–0.34)
   because their gold densities differ 2.2×. `s_linker119` measured the cost of
   collapsing those defaults into one schema: net −9.0 / −16.0. A single judge over the
   merged stream is only defensible if the `form` field selects the rubric *inside* the
   call — which is the design law (facts in code, weighings in the prompt) applied to
   the rubric rather than to the candidate, and is the arm this audit leaves registered
   and unbuilt.
4. **SWATTR's relation is not a recall opportunity here.** Its published output is
   187/188 inside the head's candidate set, its one outside pair is a non-gold
   hyphenation variant, and the head proposes 32 gold pairs it never emits.

Level 1 throughout: every number is a replay of documents, catalogs, recorded
checkpoints and published baseline files. **No E2E is owed for the refusal** — an arm
whose measured marginal gold is zero against the existing pipeline cannot move a paired
run except through its extra false positives.
