# Merging alias extraction into the reading, keeping the judge — 2026-08-14

The s26–s34 line asked twelve ways whether the knowledge module can be merged away and
concluded it cannot. **One arrangement in that space was never built.**

| | alias proposal | alias judging |
|---|---|---|
| s25 / s49 | separate document-wide pass | separate call |
| s26, s28 | folded into the reading | **deleted** |
| s29, s30, s31, s32, s33, s34 | separate pass kept | moved into a grounding check, into the proposing call, into the extraction calls |
| **s60, s61** | **folded into the reading** | **separate call, unchanged** |

s26 changed both at once and lost 2.2 macro F1. Its own diagnosis named two causes and
only one of them is about judging:

> (a) a batch cannot see a definition stated elsewhere, so the reading loses short forms
> defined once and used far away (`ui`, `webui`, `e2e`, `gae`);
> (b) nothing judges what the reading collects, so the table gains descriptive phrases
> and generic words — and even the dotted forms the alias rubric forbids, because
> "the rule is followed by a dedicated prompt and violated when appended to an
> extraction prompt".

`s_linker60` fixes (b) and leaves (a) standing. That makes it the arm that says how
much of s26's loss was judging and how much was granularity.

## Why it should be the same

The architecture exploration defended the two-stage split with a **granularity**
argument: references degrade with passage length (s27: F1 98.4 at 37 sentences, 79.7 at
87) while alias definitions are stated once and used far away, so names need the whole
document. One pass cannot serve both.

That argument is about the *proposer*. The exploration's other law — a judging step must
be **separate, semantic, lenient, independent** and have **undivided attention** — is
about the *judge*, and it is the law every failed merge in that line actually broke:
s29/s30 replaced the judge, s31–s34 gave the judging job to a call whose main job was
something else. **s60 breaks neither law.** Its judge is a dedicated call, with its own
prompt, the same rubric, the same leniency, and no other work to do. Only the proposal
becomes batch-local.

So the whole theoretical gap is (a), and (a) is measurable in advance.

## What (a) actually costs

Every term the two-stage pass finds and the merged reading does not, scored by the
(sentence, component) pairs it alone could admit — the sentence carries the term and not
the component's own name:

| project | term | → component | gold | spurious |
|---|---|---|---|---|
| teastore | `UI` | WebUI | 1 | 5 |
| teammates | `datastore` | GAE Datastore | **3** | 0 |
| teammates | `GAE` | GAE Datastore | 0 | 12 |
| teammates | `E2E`, `Test Driver`, `end-to-end component` | | 0 | 0 |
| bigbluebutton | `HTML5 client`, `HTML5 server`, `Redis pubsub`, `apps` | | 0 | 0 |
| **total** | | | **4** | **17** |

Batch-locality costs **4 gold pairs, three of them from one lowercase short form**, and
saves 17 spurious ones. The zero-reach entries are case variants: every sentence carrying
`HTML5 client` also carries `HTML5 Client`, so the alias never admits anything the name
did not already admit.

## Stage screen

`approach/pilot/prompt_stage_pilots.py --pilot mergedalias`, five samples a side, the
alias table scored by the same reach measure:

| | gold | spurious |
|---|---|---|
| two-stage (s49) | 25.0 | 25.4 |
| merged reading + dedicated judge | 24.4 | **8.8** |
| | −0.6 (p = 0.80) | **−16.6 (p = 0.01)** |

Recall on the alias side is held; the table's spurious reach collapses. Both of the
diagnosis's predictions appear in the term lists: only the two-stage pass finds `E2E`,
`GAE`, `UI`, `SFU`, `BBB web`, `apps` (cause (a)), and the judge removes most of what the
reading adds — **except** `logic.api` and `logic.core`, which is (b) resurfacing in the
one place the merged design does not state the rule.

## The two variants

**`s_linker60`** — the merged reading builds the table, `_learn_document_knowledge`
returns nothing, `_judge_aliases` runs the unchanged judge over what the reading
collected. **Three document-reading prompts become two**; the knowledge/extraction side
falls from 19 calls to 14 per five-project run, and the whole run from 88 to 83.
Everything after the reading is `s_linker59`'s, byte for byte.

**`s_linker61`** — s60 with `ALIAS_EXCLUSION_RULES` also stated in the *judge's* prompt.
The two-stage design only ever needed it in the proposer, because that proposer was
dedicated; the merged design needs it where a dedicated call enforces it. 122 bytes, no
new call. **Measured reach on this benchmark: zero** — both leaked terms have 0 gold and
0 spurious pairs, since every sentence containing `logic.api` also contains `logic`. It
is adopted as design integrity, not claimed as an improvement.

`approach/pilot/test_s60_s61_merged_alias.py` pins both: the separate pass is gone, the
judge survives with s49's prompt and rubric, the reading carries all three rule blocks
the two prompts it replaces carried between them, every stage after the reading is
s59's byte for byte, and s61 differs from s60 in exactly one prompt by exactly one block.

## The residual asymmetry, stated rather than hidden

s49's extraction prompt receives a **judged, complete** alias table before it reads a
single sentence — and removing that `KNOWN ALIASES` line costs 5.2 true positives, so it
matters. s60's reading receives an **unjudged, partial** table fed forward batch by
batch: batch 1 sees nothing, batch 4 sees what batches 1–3 established, and none of it
has been judged yet.

Two things make that the right kind of wrong. The table's job *inside the extraction
prompt* is a recall hint, where an unjudged suggestion costs little and the two full-name
judges screen it afterwards. And the table every **downstream** stage reads — the
contract filter, the mention classifier, the partial-name suppressor, the coreference
antecedent gate — is the judged one, because judging happens before the reading returns.

## And the stage screen was right about the table and wrong about the pipeline

Six paired runs, carrying s49, an **in-set null arm**, s59 and s60 in the same
invocations (`results/s5960_e2e_r{1..6}_20260813`):

| | TP | FP | macro F1 | macro F2 | calls |
|---|---|---|---|---|---|
| s49 | 187.0 | 12.2 | 96.47 | 96.78 | 87 |
| **null arm** | +0.7 (p = 0.54) | −1.5 (0.50) | **+0.1 (0.71)** | +0.0 (0.84) | 87 |
| s59 | **+1.5 (p = 0.05)** | −2.2 (0.40) | +0.6 (0.18) | **+0.5 (0.03)** | 87 |
| **s60** | **−5.0 (p = 0.00)** | **+11.2 (0.01)** | **−2.7 (0.00)** | **−2.2 (0.00)** | 83 |

The null arm is neutral on every measure in this set, so there is no harness offset to
subtract — the s60 result is the arm's own. It is the ninth time in this branch that a
stage arm has pointed the opposite way to the composed pipeline, and the first where the
stage arm was not merely optimistic but measuring a different thing entirely.

`stage_diff.py` puts the loss in one place:

| source | gained gold | gained spurious | lost gold |
|---|---|---|---|
| full_name | 3.0 | 2.5 | 2.5 |
| coreference | 0.0 | 1.5 | 0.5 |
| **partial_name** | **0.0** | **13.5** | 2.5 |

The alias table's admission role improved, exactly as screened. Its **suppression** role
got worse: `_name_word_candidates` excludes a sentence from partial-name proposals when
it states any name in N(c), so a *tighter* alias table frees partial-name candidates.
This is the third measurement of the dual role the s26 diagnosis named and s46 priced
from the other direction, and it is the sharpest, because here the table got better by
every alias-side measure and the pipeline still lost 2.7 F1.

Deterministically, on teammates, and with no LLM call at all:

| alias table | partial-name candidates | of which gold |
|---|---|---|
| s49's (6 terms) | 31 | 4 |
| s60's (10 terms) | 40 | 4 |
| s60's **plus the single term `GAE`** | **30** | **4** |

**One missing alias explains the whole regression** — `GAE` for `GAE Datastore`, a short
form the document introduces once ("Google App Engine (GAE)") and uses far from its
definition, which is precisely what a batch-local reading cannot see and a document-wide
pass can.

## The obvious fix, refuted in minutes

If the loss is one name word, propose name words deterministically and let the unchanged
judge decide. `--pilot namewordalias` offers every word of a multi-word component name
that the document uses standalone, then runs s49's alias judge over them:

| project | proposed | judge kept | partial-name candidates | of which gold | (s49's) |
|---|---|---|---|---|---|
| teammates | 4 | 4.0 | 13.0 | **1.0** | 31 / 4 |
| bigbluebutton | 11 | 10.0 | 3.0 | **0.0** | 31 / 16 |

The judge approves name words wholesale — they are not "generic vocabulary", they are
literally parts of the component's name, so the rubric has no ground to reject them — and
the partial-name linker is suppressed out of existence, destroying all 16 of
bigbluebutton's gold partial-name candidates. Refuted at a cost of fifteen LLM calls.

## What this establishes

The document-wide alias pass is not removable, and the reason is new. It is **not** that
a merged reading finds worse aliases — it finds better ones. It is that the alias table
doubles as the partial-name linker's exclusion list, and that role needs a table with a
particular *shape*: broad enough to hold document-introduced short forms like `GAE`,
narrow enough to exclude ordinary name words like `Server` and `Client`. Only a pass that
reads the whole document and is asked "reject terms whose ordinary English use dominates"
produces that shape. A batch-local proposer misses the first; a deterministic proposer
plus a lenient judge cannot enforce the second.

**So the theoretical argument for parity was right about everything it covered and the
design still fails, because the alias table is not only an alias table.** That is the
sentence the paper should carry.

## Per project: the failure lands exactly where the partial-name linker runs

Six runs, macro F1 and false positives per project:

| project | s49 | s60 | partial-name candidates/run |
|---|---|---|---|
| jabref | 100.00 (FP 0.0) | **100.00** (FP 0.0) | 0 |
| mediastore | 97.83 (FP 0.3) | 97.02 (FP 0.7) | 0 |
| teastore | 100.00 (FP 0.0) | 99.09 (FP 0.5) | 0 |
| **teammates** | 91.40 (FP 6.7) | **83.56** (FP 12.8) | ~30 |
| **bigbluebutton** | 93.10 (FP 5.2) | **89.02** (FP 9.3) | ~30 |

The three projects whose partial-name linker never fires are untouched (0.0, −0.8, −0.9).
The two where it does fire lose 7.8 and 4.1 F1. That is the suppression mechanism
confirmed a third way, at the project level, without any further measurement: the merge
is harmless to the alias table's *admission* role everywhere, and costly wherever its
*exclusion* role has work to do.

## The last cheap avenue, also closed

If the reading cannot *propose* `GAE` as an alias, perhaps it already *records* it: the
reading returns a `matched_text` span for every reference, and a span that is not the
component's name is evidence of a short form. Reading s60's own responses on teammates
across the runs:

    18x Logic         <- 'Logic component'      9x E2E    <- 'E2E tests'
    15x Storage       <- 'Storage component'    9x UI     <- 'UI component'
    12x Common        <- 'Common component'     6x Client <- 'Client component'
    GAE Datastore: no spans at all

The reading never reports a reference to `GAE Datastore` in the first place, so there is
nothing to project from. The definition is in batch 1 and the uses are elsewhere, and
nothing connects `GAE` to that component without a document-wide view. (This also matches
the earlier measurement that the table "cannot be projected from the extractor's
`matched_text` — 41% recall, 28 spurious surfaces; that field is a span, not a name".)

## Final accounting

Restoring a document-wide alias pass to fix the suppression shape restores s49 exactly:
s49 is `doc_extract` + `doc_judge` + N extraction calls = N+2 per project, and s60 is
N reading calls + 1 judge = N+1. **The entire benefit of the merge is one call per
project — five per five-project run, 88 → 83 — and it costs 2.4 macro F1.** There is no
intermediate design: any fix that gives the table its suppressive shape back is a
document-wide pass, which is the call the merge was removing.

## Fixed along the way

s60/s61 write the knowledge checkpoint before the linkers run, and in this design the
table does not exist until the reading — inside the *first* linker — has built it. The
checkpoint therefore recorded an empty table, and the first pass at diagnosing these runs
read it and drew a wrong conclusion. Both variants now re-save the checkpoint after the
linker loop, and `pilot/test_s60_s61_merged_alias.py` asserts it.
