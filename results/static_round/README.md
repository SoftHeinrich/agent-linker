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
