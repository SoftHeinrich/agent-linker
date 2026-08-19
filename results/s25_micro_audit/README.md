# s25 micro-conditions: every `if A and not B`, priced — 2026-08-10

The big gates were priced in `results/s25_gate_audit/`. This round is the fine
print inside them: the boolean clauses, the case rules, the conjunctions, and the
places where two nearly identical tests used different predicates.

`approach/pilot/micro_audit.py` counts, for each condition, how often it fires
and how often firing changes the result — deterministically, exhaustively over
all 3697 (component name, sentence) pairs and all 5388 word spans of the five
projects. No LLM call. Arms for the two that change prompt text are in
`approach/pilot/simplify_pilots.py`.

## Dead conditions — cut, provably no behaviour change

| Condition | Fires | Changes a result | Action |
|---|---|---|---|
`has_standalone_mention`: reject a match preceded by `.` | — | **0 / 3697** | cut |
`has_standalone_mention`: reject a match followed by `.X` | — | **0 / 3697** | cut |
`has_standalone_mention`: reject a match preceded by `-` | — | **0 / 3697** | cut |
`has_standalone_mention`: reject a match followed by `-` | — | **0 / 3697** | cut |
`_all_occurrences_in_qualified_path` asks `isalpha()` where `_inside_qualified_identifier` asks `isalnum()` | 4 spans differ corpus-wide | **0 / 3697** name pairs | unified |
`_all_occurrences_in_qualified_path` does not require an alphanumeric before the dot; the other test does | — | **0 / 3697** | unified |

Both qualified-path tests now call one `_in_dotted_path`. The four boundary
rules are gone with the predicate that carried them. Verified: `0` flips against
the pre-change code on every name pair and every word span, asserted in
`pilot/test_s25_standalone.py`.

## The last asymmetry — cut after measurement

`has_standalone_mention` was case-sensitive for single-word names and
case-insensitive for multi-word ones, while `_find_exact_form` — the primitive
the rest of the linker uses — is case-insensitive throughout. They disagree on
**47 of 3697** pairs. Two arms:

| Arm | TP | FP | Verdict |
|---|---|---|---|
| Collapse everything to `_find_exact_form` | ±0.0 (p=1.00) | ±0.0 (p=1.00) | identical link sets in all 10 runs — but it makes `CODE_TOKEN` unreachable |
| One test + explicit case comparison of the matched surface | ±0.0 (p=1.00) | −0.4 (p=0.76) | **adopted**: all labels stay reachable |

End-to-end, three runs each: the restructured form reads macro F1 96.47 / F2
95.20 against 96.42 ± 0.42 / 95.38 ± 0.58 for six runs of the form it replaces —
inside the band, so adopted on the grounds of having fewer conditions.

This one was adopted, reverted, and re-adopted, and the detour is the useful
part. The first three runs of the previous form happened to land at 96.8 ± 0.1,
which made the restructured form's 96.5 look like a regression. It was not: code
verified identical to that configuration later produced 96.2 / 95.8 / 96.2. The
identity was established two ways — 0 flips of every deterministic predicate over
3697 name/sentence pairs, and 0 mention-label mismatches over the 170 judged
cases recorded in the 96.8 run's own trace. **Three runs do not estimate this
pipeline's spread; comparing a new three-run mean against an old three-run mean
will manufacture regressions that are not there.**

`_find_exact_form` already returns what it matched, so "proper case" is a string
comparison, not a second predicate with its own case rules. The coreference
antecedent gate was measured at **0** flips either way, so it collapsed to one
line as well.

**The workflow now has exactly one name-matching test and one dotted-path test.**

## Live conditions — kept, each with a count

| Condition | Measured |
|---|---|
`_classify_mention_typed` five labels | all five occur: 122 / 42 / 11 / 10 / 3 |
`_name_signature` four regex alternatives | all four match: 181 / 5485 / 20 / 52 |
`_name_word_candidates` prefix rule (`startswith`, not `==`) | admits **34** candidates an exact word match would not |
`_name_word_candidates` unique-owner test | fires 15 times (BigBlueButton) |
`_spelling_variant_candidates` separator test | 690 breaks |
`_spelling_variant_candidates` unique-owner test | fires **0** times — kept as a correctness guard, not a heuristic: without it the code would silently pick `targets[0]` |
`_inside_qualified_identifier` dotted vs joined disjuncts | 175 of 721 suppressions come from the dotted side alone |

## Inert but kept, and why

| Condition | Measured | Why it stays |
|---|---|---|
denotation `evidence_valid` (label ∈ set, claim non-empty, claim substring) | **0** failures in 38 decisions | it is the fabrication guarantee the paper cites for \linkerD; the earlier claim-check episode showed a check can be inert precisely *because* the prompt demands the quote |
identity `evidence_valid` (anchor membership, claim substring, non-empty alternative) | **0** failures in 19 decisions | same |
parser tolerances: `approve` as a string, alias payload as a dict, sentence number as a string, case number as a string | **0 / 492**, **0 / 10**, **0 / 535**, **0 / 76**; 0 unparsable responses | defensive parsing against model variability, not domain heuristics — a different backend will return a string boolean |

The distinction that matters: a *heuristic* encodes an assumption about language
and must justify itself; a *guarantee* and a *tolerance* do not decide anything
and cost one line each.

## Net effect on the code

- one name-matching test where there were two, with the case rule explicit;
- one dotted-path definition where there were two that disagreed;
- four dead boundary conditions gone;
- no dependency on `helper_v3.has_standalone_mention` (a shared module still on
  `s_linker21`'s import path, so this also removes a cross-variant coupling).
