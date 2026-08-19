# s25 complexity audit and simplification pilots — 2026-08-10

Second pass over the linker, this time asking of every hand-coded path: does it
change a decision, or does it only change text? `approach/pilot/complexity_audit.py`
answers that deterministically off the promoted run's checkpoints (no LLM call).
`approach/pilot/simplify_pilots.py` runs arms for the three answers that needed
LLM evidence — five runs per side, all five projects, permutation-tested.

## Deterministic findings (`audit.json`)

| Path | Finding |
|---|---|
| Two name primitives | `has_standalone_mention` and `_find_exact_form` disagree on **47 of 3697** (name, sentence) pairs, always with the lenient one matching. They flip the coreference antecedent gate on **0** of the promoted run's resolutions — so only anchor lists ever moved. |
| Mention-type classifier | Distribution is **not** degenerate: 122 proper-standalone, 42 via-alias, 11 code-token, 10 lowercase, 3 indirect. Cannot be argued away; needed an arm. |
| Spelling variants | 6 candidates generated across five projects; **2 accepted, both gold**, both on BigBlueButton (S30, S78 `BBB web`) — pairs extraction never proposed. |
| `_inside_qualified_identifier` | All four disjuncts do work: of 721 suppressed spans, **175** are suppressed by the dotted tests alone. A span adjacent to `.` is not adjacent to an alphanumeric, so the joined tests cannot see it. The test does not reduce to an adjacency test. |
| `extraction_rationale` | Exactly **1 distinct value** across every candidate on every project — a constant the judge is told each time. |
| `antecedent_via_alias` | Model sets it true on **64** resolutions; **no gate reads any of them**; the rules block defining it is **488 bytes** of every coreference prompt. |

## Adopted

| Change | Report | TP | FP |
|---|---|---|---|
| Slim bundle: drop the constant `Rationale:` line, build anchors with the lenient primitive | `bundle_slim.json` | ±0.0 (p=1.00) | **−2.2 (p=0.01)** |
| Drop `antecedent_via_alias`: the request sentence, the 488-byte rules block, the response field | `coref_no_via_alias.json` | +0.6 (p=0.17) | −0.8 (p=0.05) |

## Rejected — kept with a number

| Candidate change | Report | TP | FP |
|---|---|---|---|
| Also drop the mention-type field (would retire `MentionType`, `_classify_mention_typed`, `_all_occurrences_in_qualified_path`) | `bundle_no_mention_type.json` | **−6.6 (p=0.01)** | −0.2 (p=1.00) |

Two name primitives therefore remain, each with a measured role rather than by
accident: the strict one decides the mention type (a field worth 6.6 links) and
the coreference antecedent gate (where the lenient one would buy nothing, 0
flips); the lenient one decides admission, the partial-name suppressor, and now
anchors.

## Side result: what the ambiguity map was actually for

The jabref false positives the map existed to prevent are S5 `logic` ("only a
little bit of logic attached") and S7 `preferences` ("the gui knows the user and
his preferences") — the paper's own motivating example. Paired full-pipeline runs
on jabref, five runs each:

| Linker | jabref FP |
|---|---|
| pre-pilot, with the ambiguity map | 1 / 0 / 2 / 0 / 0 (mean 0.6) |
| post-pilot, no map | 2 / 2 / 2 / 2 / 2 |
| post-pilot **+ slim bundle** | **0 / 0 / 0 / 0 / 0** |

So the map never reliably suppressed them, and showing the judge more anchor
sentences does. A name-level prior was standing in for sentence-level evidence.

## Composed result

Three five-project E2E runs after both adopted changes
(`results/s25_simplified_e2e_r{1,2,3}_20260810`, summary in r1's `SUMMARY.md`):
macro F1 **96.8**, pooled **95.5**, TP 182.3, FP **4.3**, FN 12.7 -- against
94.7 / 93.6 / 179.7 / 9.3 / 15.3 before this round and 94.2 / 91.6 / 179 / 17 / 16
before the design pilots. No project regressed.

Read the +/- 0.1 those three runs suggest as luck, not stability: three later runs
of verified-identical code gave 96.2 / 95.8 / 96.2. Pooled over six runs the band
is macro F1 96.42 +/- 0.42, macro F2 95.38 +/- 0.58 -- see
`results/s25_micro_audit/README.md`.
