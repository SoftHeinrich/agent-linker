# Rule audit — how many hand-written rules does the linker actually have?

Deterministic, no LLM calls. Reproduce with:

    cd approach
    ../.venv/bin/python pilot/rule_audit.py              # full report -> report.txt
    ../.venv/bin/python pilot/test_s65_one_relation.py   # the identity, 49 checks

## The question

Every earlier round on this branch ablated a rule and priced its removal. This round asks
the prior question: **how many distinct rules are there?** By `s_linker64` the workflow
carried four lexical rules, each with its own regex, its own ownership test and its own
paragraph of defence:

| rule | what it did |
|---|---|
| `_keep_stated_names` | the full-name linker's admission filter |
| `_spelling_variant_candidates` | orthographic variants the extractor missed |
| `_add_stated_name_net` | the s64 addition: the model name **as spelled** |
| `_name_word_candidates` | one word of a name, under English inflection |

Read as four rules they read as accretion, and a reviewer is entitled to ask which
benchmark document taught each one.

## A1 — nothing in the deterministic layer admits a link

18 deterministic predicates, 148 code lines, 128 docstring lines defending them.
Classified by what each decides: **0 of 18 put a link in the output without an LLM
verdict.** Every one either produces a *candidate* for a judge, computes a *label* a judge
reads, or is structural (batching, windowing, set subtraction).

## A2 — the four rules are one relation, and it is an identity

A single `_name_spans(text, name, form)` reproduces all four predicates over every
**3697** (name, sentence) pair of the five projects:

| predicate | result |
|---|---|
| `_find_exact_form` | IDENTICAL |
| `_add_stated_name_net`'s scan | IDENTICAL |
| `_is_inflection_of` over the name's words | IDENTICAL |
| `_spelling_variant_candidates` (whole candidate sets) | IDENTICAL |

Two dimensions, four settings:

* **fidelity** — how exactly the characters must reproduce the name:
  `AS_SPELLED` < `ANY_CASE` < `ANY_SPELLING`
* **extent** — how much of the name must be present: the whole name, or `ANY_WORD`

Which cell a proposer scans is the only thing that distinguishes it from the others.

## A3 — one monotone table replaces four arguments

Pairs reached over all five projects, and how many are gold:

| fidelity / extent | pairs | gold | gold per pair |
|---|---|---|---|
| `AS_SPELLED` whole name | 112 | 107 | **0.955** |
| `ANY_CASE` whole name | 172 | 133 | 0.773 |
| `ANY_SPELLING` whole name | 176 | 137 | 0.778 |
| `ANY_WORD` one word | 281 | 161 | 0.573 |

Precision falls monotonically as the relation loosens and reach rises. **That table is the
design rationale: the looser the form a linker scans, the stricter the judge behind it.**
The full-name linker scans the two tight rows and judges in two focused calls that approve
by default; the partial-name linker scans the loosest row and judges target-blind; the
coreference linker reaches what no row reaches and rejects when uncertain.

s64's case sensitivity stops being a bespoke rule and becomes the top row — 0.955 against
0.773 one row down, matching the 0.86-vs-0.06 reading `pilot/statednet_screen.py` took on
*new* pairs.

**Two cells do not nest**, and the tidier claim would have been the false one:

* teastore's `ImageProvider` is written `Image Provider` — `ANY_SPELLING` reaches it,
  `ANY_WORD` does not, because the *name* is split on word boundaries only and
  `imageprovider` is one word;
* bigbluebutton's `Redis PubSub` is written `redis pubsub` — `ANY_CASE` reaches it,
  `ANY_SPELLING` does not, because the signature splits `PubSub` and the document does not.

Compound splitting is a *different* normalization, not a strictly looser one, so a linker
takes the **union** of the cells it scans. Six pairs over five projects.

## A4 — two defects, priced not hidden

* `_inside_qualified_identifier` tests `before in "-_"` with `before == ""` for a
  sentence-initial span, and `"" in "-_"` is `True` in Python. **378 spans over the five
  documents** — exactly one per sentence. `s_linker63` repaired it and measured
  **FP +1.2 (p = 0.01) at TP ±0.0**, so on this benchmark the defect is load-bearing.
* `_all_occurrences_in_qualified_path` lowercases the name and searches the raw sentence,
  so it only ever sees lowercase spellings. It can produce 28 `CODE_TOKEN` labels as
  written and 25 with case handled the way the rest of the module handles it — **3 labels
  would flip**.

Both are carried forward unchanged in `s_linker65`, which changes no behaviour, and both
are named in its docstring and in the paper's threats section. A retained defect that a
benchmark rewards is a validity threat to state, not a design choice.

## A5 — `s_linker65`

`s_linker64` with the deterministic layer written once. No mechanism change, no prompt
change, no behaviour change.

| | s_linker64 | s_linker65 |
|---|---|---|
| lexical-layer methods | 9 | 8 |
| lexical-layer code lines | 106 | 90 |
| distinct lexical rules to defend | 4 | 1 relation, 4 settings, 3 scanned |
| **methods to read to know what a name match is** | **5** | **2** |

The line count barely moves and the matching still has to happen; what moves is how many
places a reviewer must read. `_antecedent_states_name`, a one-line wrapper with one call
site, is also deleted.

`pilot/test_s65_one_relation.py` asserts the identity in **49/49 checks**: 44 shared method
bodies byte-identical, 11 rule constants / 7 resource bounds / 5 prompt builders identical,
the relation against all four predicates on 3697 pairs, every candidate set of all three
generators on all five projects (same pairs, same `matched_text`, same `source`, same
mention labels), the composed full-name candidate list under four extractor stand-ins per
project (empty, full overlap, disjoint, half overlap), and GATE-06.

**No end-to-end run is owed.** This is an identity over the candidate sets, not a
behavioural arm; `pilot/composition_check.py`'s precondition is vacuous when the candidate
sets are equal. The scan is 3-6x slower in wall-clock (0.02s → 0.07s on the largest
project) against runs of ~65 LLM calls, so the cost is not measurable end to end.
