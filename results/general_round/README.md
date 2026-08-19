# The general round — every authored sentence scored on what it stands on

`s_linker70` → `s_linker71`. The deterministic layer was audited to exhaustion in the
bind and fold rounds; what had never been audited on the same terms was the ~3.6 kB of
authored English carried into 91 LLM calls per five-project run. Two folds had just
*moved* rules into that English, which makes the question sharper rather than softer:
**a rule is not laundered by being written in prose.**

## The bar

A prompt clause or a code gate is admissible only if it stands on one of three grounds:

| ground | meaning |
|---|---|
| **general** | a general rule: logic, or a distinction that holds for any text — use vs mention, reference, negation, ambiguity |
| **se-practice** | a property of software as written anywhere: qualified names compose; identifiers are named after what they are |
| **prior-work** | a decision this branch or the TLR literature already measured and defended, cited rather than re-argued |

Inadmissible: **corpus** — the clause names a surface form or a syntax whose frequency is
a fact about these five documents.

## What the audit found (`pilot/prompt_defensibility.py`, no LLM calls)

Of **3645 authored bytes in `s_linker70`**:

| ground | bytes |
|---|---|
| general | 1287 |
| se-practice | 166 |
| prior-work | 247 |
| mixed | 1131 |
| corpus | 814 |

**1700 bytes stood on an admissible ground; 1945 did not, or not entirely.**

The same distinction — "a name inside a longer identifier is not the component" — was
written **five times in five prompts**, in five wordings. A sixth copy already existed
that names no syntax at all: `QUALIFIED_CLAUSE`, written in the fold round for the
denotation prompt. Correcting an overstatement made mid-audit: only **two** of the five
copies actually spell out the syntax (`LAYERED_ENTITY_RULES` condition (1) and
`ALIAS_EXCLUSION_RULES`, both "of the form X.Y or X.Y.Z"); the other three say
"code-level path/identifier" and name no shape.

Reach of each copy, from `s_linker70`'s own three runs — the population it governs, and
how much of that population even contains a dotted identifier:

| copy | calls/run | cases/run | with a dotted id |
|---|---|---|---|
| `ALIAS_EXCLUSION_RULES` | 5.0 | 408.0 | 73.0 |
| `ENTITY_EXTRACTION_RULES` | 9.0 | 378.0 | 73.0 |
| `P1_FOCUS` | 10.0 | 206.0 | 91.0 |
| `LAYERED_ENTITY_RULES` | 20.0 | 412.0 | 182.0 |
| `LAYERED_COREF_RULES` | 8.0 | 141.3 | 31.3 |
| `QUALIFIED_CLAUSE` | 4.0 | 71.0 | 22.0 |

None is provably inert, so every one needed an arm. Of 90.3 rejections per run under the
enumerated rubric, **0.0** cite a negation in their claim and **0.3** cite a dotted
identifier — the two conditions that name the most are the two that are cited least.

## The four arms (`pilot/general_prompt_pilots.py`, 3 samples per arm, checkpoint replay)

### Adopted

**`plainrubric` — the full-name judging rubric loses its enumeration.**
Four numbered reject-conditions and three named approve-shapes → one principle plus
`QUALIFIED_CLAUSE` and `STRICTER_CLAUSE`; `P1_FOCUS` drops its code-level-identifier tail.

| arm | TP | FP |
|---|---|---|
| s70 (enumerated) | 150.7 | 4.7 |
| general rubric | **+0.7 (p = 0.80)** | **−1.3 (p = 0.20)** |

850 authored bytes → 579. Conditions (3) and (4) were the use/mention distinction, which
`STRICTER_CLAUSE` already states in the same prompt; the three approve-shapes say *where*
a name may sit in a document and are already licensed by the approve-by-default sentence
in front of them.

**`plainextract` — the extraction prompt says the same thing generally.**

| arm | TP | FP (non-gold candidates) |
|---|---|---|
| s70 (names the shape) | 156.0 | 52.0 |
| general clause instead | **+0.7 (p = 1.00)** | **−6.0 (p = 0.20)** |

### Refused — and the more informative half

**The coreference rubric keeps its clause.**

| arm | TP | FP |
|---|---|---|
| remove the phrase, add nothing | +4.7 | +3.7 |
| remove it, add `QUALIFIED_CLAUSE` | −3.0 | +4.0 |

(both p = 0.10, the floor at n = 3; the coreference stage is the branch's noisiest —
see the standing note on run-to-run variance). A clause about identifiers *misleads* a
judge whose cases contain no name at all: the coreference stage's premise is that the
component is not written in the sentence. The phrase also names no syntax, so it already
meets the bar as written.

**The alias prompt keeps its syntax — the round's most useful finding.**
`ALIAS_EXCLUSION_RULES` is the one clause the module's own comment condemned ("naming the
shape is a rule written for one corpus"). Two replacements tried:

| arm | alias table size/run | identifier fragments admitted |
|---|---|---|
| s70, spells out `X.Y or X.Y.Z` | 24.0 | **0** |
| `QUALIFIED_CLAUSE` instead | 36.7 | **0** |
| imperative general rewording | 37.3 | **0** |

On its *stated* prohibition the syntax buys exactly nothing — neither wording admits a
single identifier fragment. But the table grows by ~13 terms per run under either general
form. **The clause's measurable effect is not the effect it states**: a flatly prohibitive
sentence naming a concrete shape makes the extractor conservative about everything, not
only about what it prohibits. Since an over-large alias table was already measured to cost
F1 (`s_linker39`/`s_linker40`: macro 94.57 against 96.42), the clause stays — now
documented as doing something other than what it says. That is a *worse* defence than
before the audit in one sense and a much better one in another: it is now a measured
mechanism rather than a plausible-sounding rule.

## End to end

Single-arm batches, three five-project runs each, flex tier. Absolute levels only —
different invocation sets, so no p-value is legitimate across rows.

| variant | what changed vs `s70` | n | TP | FP | macro F1 | macro F2 | calls | F1 range |
|---|---|---|---|---|---|---|---|---|
| `s_linker69` | (predecessor) | 3 | 187.3 | 14.0 | 95.51 | 96.45 | 88 | 2.12 |
| **`s_linker70`** | **the starting point** | 3 | 189.7 | 16.0 | **95.74** | **96.99** | 91 | 1.76 |
| `s_linker71` | rubric restructured + extraction generalized | 6 | 187.7 | 19.3 | 94.80 | 96.19 | 92 | 1.73 |
| `s_linker72` | extraction reverted; rubric still restructured | 3 | 187.0 | 17.7 | 94.94 | 96.10 | 91 | 1.13 |
| `s_linker73` | rubric structure kept; syntax **and** "heading, or a list" removed | 3 | 187.0 | 13.7 | 95.25 | 95.94 | 92 | 0.34 |
| **`s_linker74`** | **the syntax alone, unspelled** | 3 | 189.0 | 14.7 | **95.60** | **96.66** | 92 | 1.36 |

**`s_linker74` is the adopted head.** F1 −0.14 and F2 −0.33 against `s70`, both far
inside either arm's own run range — parity, with the last corpus-shaped syntax gone from
the judging path.

### The three losses that located the one admissible span

The checkpoint pilots said both rewrites were fine on their own stage. Composed, they
were not, and taking them apart is the round's methodological result:

1. **`s71` → `s72`: the extraction half was not the cause.** Reverting it recovered
   0.14 F1 of the 0.94 lost. The suspicion was reasonable — the extractor feeds every
   later stage, so a prompt change there is not stage-local however it is measured — but
   it was wrong.
2. **The enumeration carries precision.** Replacing the four numbered reject-conditions
   with one principle reads TP +0.7 / FP −1.3 against a fixed candidate set and costs
   ~0.8 F1 composed. *Restructuring* a rubric is not the same edit as *degeneralizing*
   one, and only the second is what the bar asks for.
3. **`s73`: "a heading, or a list" carries recall — and was never inadmissible.**
   Removing it costs **exactly 2.7 TP in each of three runs**. It should not have been on
   the corpus-shaped list at all: headings and lists are general technical-documentation
   practice, not a property of these five documents. **The bar catches shapes peculiar to
   a corpus, not the structure every document of the genre has** — and my first pass
   applied it too widely. Dotted identifiers *are* peculiar to this corpus (62 of 198
   sentences on one benchmark, 0–6 on the other four), which is why that span, and only
   that span, is the one to change.

## What the round changes about how the artifact reads

Before: five bespoke restatements of one distinction, a numbered reject list, and three
named document shapes — a rulebook a reviewer can fairly read as fitted to this
benchmark. After: **one sentence per distinction, each with a stated ground**, and two
places where the general form was tried, failed, and the failure is reported with its
mechanism instead of the clause being quietly kept.
