# approach.tex restructure — grilling session state

**Opened** 2026-09-03. **Closed 2026-09-04. Status: frontier EMPTY — Q17–Q29 answered.**
Written so this session can be resumed on another machine. Nothing here has been
executed on `approach.tex` beyond steps 1–3 (see "Already executed").

**The spec is not a file.** It lives inline in `paper/sections/approach.tex` as `%SPEC`
annotations at the passage each one governs, plus a slim global header (N1-N11 and the
execution order). `grep -n '^%SPEC' paper/sections/approach.tex` is the queue; delete each
tag as you execute it. `motivation.tex` carries the two out-of-section `%SPEC` items.
This file keeps only the interview record: what was asked, what was answered, and the
facts the answers rest on.

## How to resume

The interview is over. Read the header of `paper/sections/approach.tex` for the global
rules and the execution order, then work the `%SPEC` tags in file order. Re-open the grill
only for the four questions deferred with the frozen table (Q23, Q25, Q26, Q29 — the
`%SPEC` block at `\begin{table}` carries them), and only once the table's replacement is
being designed.

## The task

Rewrite/refactor `paper/sections/approach.tex` (ArchLinker — LLM-based traceability
link recovery between architecture documentation and an architecture model) to match
(a) the new story and (b) the s110 design as actually implemented, using the
`writing-kb` corpus as the style authority.

Standing user mandate, given 2026-09-03 and overriding earlier framing locks:

> the root cause - insight - challenge - 3 decisions are dynamic, we can adjust any
> of them for maximum simplicity + readability + selling power

Governing skill rules in force: ask the whole *frontier* per round, numbered, each
with a recommendation, then wait. **Finding facts is the agent's job, never the
user's** — dispatch sub-agents for environment facts. **Decisions are the user's.**

## Where the story lives

`paper/sections/intro.tex` lines 4–215 — the new story is a commented sentence-by-sentence
spec, not a standalone document. Locked there:

- PRIOR-WORK AXIS: the gap is **how a link is decided**, not "single call" or
  "no reusable knowledge". `DELETE "isolation" AND "single call" AND "Artemis has
  no reusable knowledge"`.
- Root cause: *each link is decided in a single unchecked step, by matching surface
  names rather than weighing the evidence the sentence gives.*
- `[ONE problem + its implication. NOT three separate problems.]`
- Artemis is a STRONG baseline: it DOES extract aliases, DOES resolve bounded
  coreference, DOES build a reusable entity/alias artifact. Do not strawman it.

Target venue: **FSE 2027** (deadline 2026-10-02). `\documentclass[acmsmall,screen,review,anonymous]{acmart}`,
single-column, 18 pages + 4 for references. `main.tex` is already correct — do not
change the class. (FSE *2026* is two-column `sigconf`; that is a different venue year.)

## Already executed (steps 1–3, verified)

| | before | after |
|---|---|---|
| total lines | 668 | 325 |
| comment lines | 406 | 90 (one header) |
| inline annotations in body | 54 RW + 15 DONE/NOTE + 12 dividers | 0 |
| prose words | 3602 | 2881 |
| measured numbers / p-values | 48 | 0 |
| `---` in prose | 16 | 0 |

Provenance, nothing lost:
- `paper/notes/approach-stripped-measurements.md` — the 23 deleted measurement
  sentences + 3 partial excisions, grouped by subsection, for `results.tex` to absorb.
- `paper/notes/approach-audit-trail.md` — the legacy G1–G4 spec header, all 15
  `%DONE`/`%NOTE` code-audit blocks, and all 227 cleared body comment lines.

Environments balance: `figure*` 1/1, `table` 1/1, `promptL` 4/4.
**Never compiled** — no `pdflatex`/`latexmk` in this environment. Verification is
`git diff` filtered for non-comment changes, begin/end balance, residual-pattern greps.

## Section as it stands today

```
lead: "We propose ArchLinker…" + formalism (D, S, C, L)
lead: phenomenon paragraph (the rev-5 opener; does NOT follow N1)
fig:approach-overview
"ArchLinker in a nutshell." — answers the three challenges in three steps
6.1 Design of ArchLinker   — three decisions: knowledge / reference form / reliability
                             + the writing relation, tab:forms, nesting-of-forms
6.2 Project Knowledge Discovery
  6.2.1 Alias Discovery
6.3 \linkerB   6.4 \linkerD   6.5 \linkerC
6.6 Evidence-Backed Judges
```

## Decisions settled in this session

| Q | Question | Answer |
|---|---|---|
| Q1 | Where the task formalism sits | **Keep as is** — stays at the top (reopened as Q20 below) |
| Q2 | How many overview layers | Merge the nutshell paragraph **into 6.1**; that merged paragraph owns the figure reference; kill the duplication. Keep the three-challenge→decision structure, readjusted to the new story |
| Q3 | `tab:forms` | **Keep the table**, rewrite what it conveys: columns = *name / example / how to scan / how to judge* |
| Q4 | The four prompt boxes | **Comment all four out** now |
| Q5 | The "nesting of forms" paragraph | **Drop entirely** (not even to threats) — too detailed |
| Q6 | The mechanical quote check | **Outdated — update the paper to what is implemented.** No code change |
| Q7 | Three judges or four | The three are **link judges**; the alias step is never called a judge |
| Q8 | Alias-judge leniency vs the N6 delta | **Too detailed, don't care** — write nothing about leniency |
| Q9 | Subsection list | **(b) paired** — each linker subsection owns its judge; no pooled judges subsection. Flatten 6.2 + 6.2.1 into one `\subsection{Alias Discovery}`. **Keep judge strictness opaque** — too detailed |
| Q10 | What the three decisions are | **(a) unchanged axes** — "one root cause then 3 challenges again, nothing changed" (now relaxed by the standing mandate above) |
| Q11 | Rows in `tab:forms` | **(a) three rows**, one per reference form |
| Q12 | Where the merge is stated | **Do not state it at all** — trivial |
| Q13 | How deep the writing relation goes | **Remove it entirely.** No `$s \models_f n$`, no fidelity/extent. Plain English form names, maximum readability |
| Q14 | Where the shared judge design goes | **(a)** — 6.1 states what a judge *is* once (bundle, context-augmented, the term); each linker subsection says only what its own judge is handed. 6.1 never says what a particular judge sees |
| Q15 | Paired subsection titles | **(b)** — "Full-Name Links", "Partial-Name Links", "Coreference Links". Macros stay for in-prose mentions |
| Q16 | Who states the root cause | **Combine the phenomenon paragraph with 6.1** |

## Round 2 decisions (Q17–Q22), settled 2026-09-04

| Q | Question | Answer |
|---|---|---|
| Q17 | The chain of 6.1 | **(a) five moves, four paragraphs + table** — para 1 = phenomenon → root cause → insight → one design sentence; then one paragraph per challenge closing on its decision. Draft of para 1 is in the spec |
| Q18 | The three form names | **(a) full name / partial name / coreference** — matches all six macros, no renames. "Exact name" is rejected as false: the scan runs at `NameForm.ANY_CASE` |
| Q19 | The strictness sentence | **Drop it.** User: *"strictness does not help understanding the core, drop"*. N4's second required sentence is retired |
| Q20 | The `D/S/C/L` formalism | **(a) keep as settled in Q1** — the block stays at the top |
| Q21 | What justifies one judge per linker, with strictness gone | **evidence + question**, both halves. Wording confirmed at Q28 |
| Q22 | Does the notation get reused | **(a) light reuse** — `(s,c)` survives in the three linker subsections; `N(c)`, `\models_f`, fidelity and extent go |

## Round 3 decisions (Q27–Q28), settled 2026-09-04

| Q | Question | Answer |
|---|---|---|
| Q27 | The softened design law | 6.1 says *"Everything before a judge is scanning and pre-filtering: it generates candidates and never decides a link."* The two drops are named in the subsections that own them, not counted in 6.1. User: *"change the law to be soften and deterministically, say thats for scan / pre filter / candidate gen"* |
| Q28 | The pairing sentence | *"Each form leaves the judge a different thing to read and a different question to answer, so each linker has its own judge."* |

## Deferred with the frozen table — re-ask, do not guess

User ruling 2026-09-04: **"we will replace the table, current fronzen it."** `tab:forms`
is untouched in this pass, so Q3 (new columns) and Q11 (three rows) are **not** applied
and these four stay open:

- **Q23 / Q25 — what column 4 holds.** The user's reading is on record: *"its mainly the
  structure of prompt, the evidence context structure, not concete test"* — the shape of
  the evidence context each judge is handed, not the question and not the rubric.
- **Q26 — Q14 vs the table.** Q14 says 6.1 never says what a particular judge sees, but
  `tab:forms` sits in 6.1, so any per-judge column 4 breaks it. Unresolved; moot while
  frozen.
- **Q29 — the "different question" half of Q28's sentence** has no table support if
  column 4 is context-only. Currently carried by the subsections.

Facts gathered for that round, so it need not be re-derived — the three judge prompts as
the code builds them:

| judge | site | what the prompt contains |
|---|---|---|
| full-name | `_prompt_validation(strict=False)` `:607` | component list, source sentence, matched span, preceding sentence, ≤5 other sentences naming the component → quote the claim, then approve |
| partial-name | `_classify_denotations` `:1276` | ±`CONTEXT_SENTENCES` window as a sentence table, the expression, **no component name anywhere** → does this expression denote a software participant or something merely associated? |
| coreference | `_prompt_validation(strict=True)` `:607` | component list, the resolver's committed referring expression **and** antecedent sentence, the sentence plus its predecessor → quote the claim, state the strongest ground to reject, approve unless decisive |

`_prompt_validation`'s own docstring (`:609`): *"The rubric is asymmetric by design: the
full-name gate is lenient …, the coreference gate is strict."* That asymmetry is exactly
what Q9 and Q19 keep out of the paper, so a fully faithful column 4 and those two answers
cannot all three hold. That is the collision to settle when the table is replaced.

## Facts established (do not re-derive)

### s110 pipeline, as the code runs it today

`approach/src/llm_sad_sam/linkers/experimental/s_linker110.py` (1507 lines).
Driven by `link()` (`:407-465`): a literal `for` loop over
`LINKERS = ("full_name", "partial_name", "coreference")`, `current = self._union(current, produced)`
after each. No controller, no router — `_run_linker` (`:467-483`) is a three-way `if`.

| stage | LLM calls | note |
|---|---|---|
| parse model / load sentences | 0 | |
| `_learn_document_knowledge` (`:731`) | 2 (extract + judge) | 0 if `no_knowledge=True` |
| full-name scan `_extract_named_mentions` (`:863`) | **0** | `NameForm.ANY_CASE`, scans catalog name **+ aliases** |
| full-name judge `_run_validation_pass(strict=False)` | ⌈n/25⌉ | |
| partial-name scan `_scan`/`_scan_all` (`:886`) | **0** | `NameForm.ANY_WORD`, catalog name only |
| denotation judge `_classify_denotations` (`:1276`) | ⌈n/25⌉ | target withheld |
| coreference resolver `_resolve_references` (`:1370`) | ⌈len(sentences)/10⌉ | **every** sentence is a target |
| coreference judge `_run_validation_pass(strict=True)` | ⌈n/25⌉ | |
| `_union` (`:539`) | 0 | |

Six LLM decision points at five `_ask` sites (`:1220` serves two judges via `strict`).
**Four LLM judging prompts, not three** — the alias judge counts if you count every
call that rules on a proposal; only the two link judges share `_prompt_validation`.

Design law, stated correctly: **nothing in the deterministic layer admits a link,
though it may end a case.** It ends one in exactly two places — the alias table
(a term two components both claim names neither and is dropped) and the partial-name
nesting refusal.

Refusal predicates and which table each reads — the load-bearing asymmetry:
- `_only_inside_another_name` → `_covering_names` (`:918-943`) reads the **catalog only**,
  never `_names_by_component()`. It **ends** a case, so it may rest only on what is given.
  8 executable lines — do not call it a module.
- `_states_a_name` (the whole-name skip in `_scan_all`, `:907`) **does** read the alias
  table. Safe because the set it removes is exactly what the full-name scan admits
  (same N(c), same form) — it **routes**, it does not end. These are two different acts;
  the section must distinguish them or someone will "fix" the skip.
- `_named_before` (`:652-666`), the coreference shortlist, also reads the alias table.
  It supplies a fact to a prompt; it ends nothing. ~8 lines. Keeps only the **latest**
  naming sentence per component, nearest first.
- The full-name scan has **no** refusal: its `SKIP_QUALIFIED` predicate is dead (`= False`, `:376`).

`EvidenceBundle` (`:314-329`), 5 fields, and **only the full-name judge ever sees one**:
`source`, `matched_span`, `mention_type`, `preceding_text` (S−1, printed twice by design),
`anchor_sentences` (≤ `ANCHOR_LIMIT = 5`, catalog name only, aliases not used here).
`mention_type` is filtered by `RETAINED_MENTION_TYPES = {VIA_ALIAS, CODE_TOKEN}` — **two**
labels, not the four the paper describes; everything else renders empty.

`_union` (`:539-549`): keys on `(sentence_number, component_id)`, first writer wins,
`full_name > partial_name > coreference`. Every link has `confidence = 1.0`, so linker
order decides **nothing but which linker's `source` label survives**. No linker is
shown earlier links.

`linker_infra.py` (313 lines, commit `64e22373`, 2026-09-03) is a **pure
extraction-to-infra refactor** — eleven method bodies became one-line delegations;
no prompt string, rule constant, `NameForm` branch, scan, judge, threshold or `_union`
line touched. **No conceptual stage was introduced, removed or renamed.** Working tree
is clean for both files; only `paper/sections/approach.tex` is modified.

### Paper-side facts

- `paper/abbrev.tex`: `\linkerB` = full-name linker, `\linkerC` = coreference linker,
  `\linkerD` = partial-name linker, `\entValidator` = full-name judge,
  `\partValidator` = partial-name judge, `\corefValidator` = coreference judge,
  `\evidenceBacked` = evidence-backed (line 28 — the one line the rename touches).
  **The judge macros already read correctly**; Q7 changes only the collective noun
  and the subsection title.
- `figures/drawio/approach-overview.drawio` labels: ① Knowledge Discovery / ② Linkers /
  ③ Judges, with `whole name · scan`, `one word · scan`, `no name · LLM`. **No stale
  "evidence-backed" or "validator" text — the N5 rename forces no regeneration.**
  Two boxes/captions do need edits: the **Document Understanding** box (follows the
  retitle) and the caption + `\Description`, which both end on "then merged" — a term
  Q12 removes from the prose.
- `benchmark/jabref/text_2021/jabref.txt` has 13 real sentences. Real S7:
  *"Only the gui knows the user and his preferences and can interact with him to help
  him solve tasks."* All JabRef components are single-word (`cli, globals, gui, JabRef,
  logic, model, preferences`), and the gold standard has no link for S13, the only
  name-free sentence. So the running example **cannot** be extended to cover
  coreference honestly.
- `figures/jabref_trace_example.py` draws a **fabricated** S6 ("…the UI renders the main
  application window.") and a fabricated pronoun reading of S7, attributed to a real
  public benchmark. User instruction: declare the figure adjusted, renumber S1…S5,
  say it is adapted for space and information density.
- `paper/sections/motivation.tex` claims S7 shows "A reference uses no component name" —
  **false** for the real S7. Its `fig:example` caption is entirely commented out
  (an artefact of the ICSE→FSE template switch, per the user).

## Prose corrections to apply — facts, no decision needed

- `\linkerC` says "for each sentence with a pronoun like 'it'". The resolver runs over
  **every** sentence; there is no pronoun filter.
- The evidence bundle's reference-form field is described four ways; the code retains
  **two** labels. Prose goes to two.
- **There is no mechanical quote check, for any judge** (Q6). The substring check was
  removed — recorded inline as voiding 0 of 380 verdicts over six five-project runs.
  What remains is `valid = denotation in {"participant","associated"} and bool(claim)`,
  a non-emptiness check. The prompt still asserts "Claim must be a contiguous exact
  substring of the source sentence" — a stated, unenforced contract. **Never write a
  sentence implying a fabricated quote cannot approve a link.**
- The section's current claim that the deterministic layer ends a case "in two places"
  is **correct** and survives.

## Spec-header edits pending (N1–N10 live in approach.tex:2-94)

- **N4** reads "one root cause, not three challenges". Must be amended to *license* the
  three-challenge derivation (one root cause → three challenges → three decisions),
  per Q10 and the standing mandate. Also: the writing relation is retired (Q13), and
  the second bridge sentence is subject to Q19.
- **LOAD-BEARING** note "the mechanical quote check holds for the partial-name judge
  ONLY" is obsolete twice over. Replace with the prohibition above.
- Add a **don't**: judge strictness stays opaque (Q9) — no rubric, no approve-by-default,
  no reject-when-uncertain, anywhere.
- **N9** is superseded by Q4: all four prompt boxes commented out. Keep the one-clause
  prose description of each prompt's shape so a box can be restored without rewriting.

## Execution queue (SUPERSEDED by the %SPEC tags in approach.tex, kept for provenance)

1. Rebuild 6.1 on the Q17 chain: phenomenon + root cause + insight + one design
   sentence, then three challenge→decision paragraphs, then the three-row table.
   Absorbs the nutshell paragraph (Q2) and the lead phenomenon paragraph (Q16).
2. Rewrite `tab:forms` to *name / example / how to scan / how to judge*, three rows.
3. Flatten 6.2 + 6.2.1 to `\subsection{Alias Discovery}`; retitle the drawio box.
4. Retitle the three linker subsections to the link objects (Q15); fold each judge in;
   state the shared judge design once in 6.1 (Q14); delete the pooled judges subsection.
5. Delete the nesting-of-forms paragraph (Q5) and the writing relation (Q13); restate
   the nesting refusal, the whole-name skip and the coreference shortlist in plain English.
6. Comment out all four `promptL` boxes (Q4).
7. Apply the prose corrections above; drop the merge sentence and the caption's
   "then merged" (Q12).
8. Term rename `evidence-backed` / `context-augmented` → **evidence-augmented**:
   `abbrev.tex:28` (one line), `approach.tex`, `intro.tex` (three live sites), the
   subsection title, the prompt-box title.
9. Regenerate `figures/jabref_trace_example.py` with S1…S5, declared adjusted; restore
   the `fig:example` caption; fix `motivation.tex`'s false "no component name" claim.
10. Cut the contribution-claim sentence from the judges passage (N7).

## Retired designs — do not reintroduce

Sequential linkers ("sees only what the earlier ones left unlinked") — the reported arm
passes no link set to any linker. The structural antecedent constraint / coreference
alias gate. The LLM named-mention extractor and its `{entity}` prompt box — the
full-name proposer is a code scan and issues no prompt. The two-pass p1/p2 full-name
judge — all three link judges rule once. The model-understanding module and its
ambiguity map — one table only. The alias scope grade ("global"/"local"). The grounded
identity review after the target-blind denotation step. Ambiguous-name handling of any
kind ("drop all ambiguous name at all, we dont use them").

## Carried forward, still blocked

Position prior work by **linking mechanism** (lexical / string-similarity /
neural-embedding), not by generic glossary and IR citations. Overlaps `rw.tex`, which
is commented out in its entirety, so decide there first. The five citations were
fact-checked 2026-06-28 and all support their claims (see the audit trail).

## Flagged, not authorised

`abstract`'s retired axis and s92a-era numbers; `results.tex:58`'s "no reusable
architecture knowledge"; the empty `rw.tex`; `approach/CLAUDE.md:1104`'s wrong
`mini-rq34/rq4_floor.py` path and `:1017`'s stale 129-check count.
