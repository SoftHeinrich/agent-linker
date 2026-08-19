"""s_linker68 — s_linker64's four lexical rules restated as one relation.

NO MECHANISM CHANGES, NO PROMPT CHANGES, NO NEW BEHAVIOUR. This variant is
``s_linker64`` with its deterministic layer written once instead of four times, and
`pilot/test_s65_one_relation.py` asserts the identity: every one of 3697
(name, sentence) pairs, every candidate set of all three proposers on all five
projects, and every other method body byte-identical to s_linker64's.

WHY, AND WHAT A REVIEWER SEES. By ``s_linker64`` the workflow carried four separate
lexical rules, each with its own regex, its own uniqueness test and its own paragraph
of defence:

    ``_keep_stated_names``            the full-name linker's admission filter
    ``_spelling_variant_candidates``  orthographic variants the extractor missed
    ``_add_stated_name_net``          the s64 addition: the model name **as spelled**
    ``_name_word_candidates``         one word of a name, under English inflection

Read as four rules they read as accretion -- four things that had to be *discovered*,
and a reviewer is entitled to ask which benchmark document taught each one.
`pilot/rule_audit.py` shows they are not four rules. They are **one relation at four
settings**, and the check is an identity rather than a claim: a single
``_name_spans(text, name, form)`` reproduces all four exactly on every (name, sentence)
pair of all five projects, with **0 divergences** in each of the four comparisons.

THE RELATION. ``realizes(s, n, form)`` -- the spans of sentence ``s`` that write name
``n`` at ``form`` -- over two independent dimensions:

    fidelity  how exactly the characters must reproduce the name
              AS_SPELLED   < ANY_CASE   < ANY_SPELLING
    extent    how much of the name must be present
              the whole name           < one word of it (ANY_WORD)

Four of the six cells are used, and which cell a proposer scans is the *only* thing
that distinguishes it. The rest of each proposer is two shared options -- whether a
surface owned by more than one component may be proposed, and whether a sentence that
already writes a whole name belongs to an earlier linker -- so ``SCANS`` below is three
rows of a table, not three rules.

WHY EACH LINKER SITS AT ITS CELL, IN ONE TABLE INSTEAD OF FOUR ARGUMENTS. Pairs
reached over all five projects and how many are gold (``pilot/rule_audit.py --only A3``):

    fidelity/extent           pairs   gold   gold per pair
    AS_SPELLED   whole name     112    107        0.955
    ANY_CASE     whole name     172    133        0.773
    ANY_SPELLING whole name     176    137        0.778
    ANY_WORD     one word       281    161        0.573

Precision falls monotonically as the relation loosens and recall rises, and that single
table is the whole design rationale: **the looser the form a linker scans, the stricter
the judge behind it.** The full-name linker scans the two tight rows and judges in two
focused calls that approve by default; the partial-name linker scans the loosest row
and judges target-blind; the coreference linker reaches the sentences no row reaches at
all and rejects when uncertain. The case-sensitivity of the s64 net stops being a
bespoke rule and becomes the top row of this table -- 0.955 gold per pair against 0.773
one row down, which is why the recall floor under the LLM extractor is drawn there.

TWO CELLS THAT DO NOT NEST, AND THE REASON IS WORTH STATING. The fidelity dimension is
a chain except where compound splitting disagrees with case folding
(``pilot/rule_audit.py --only A3`` reports both): teastore's ``ImageProvider`` is
written ``Image Provider``, which ANY_SPELLING reaches and ANY_WORD does not, because
the *name* is split on word boundaries only and ``imageprovider`` is one word;
bigbluebutton's ``Redis PubSub`` is written ``redis pubsub``, which ANY_CASE reaches and
ANY_SPELLING does not, because the signature splits ``PubSub`` and the document does
not. So ANY_SPELLING is a different normalization, not a strictly looser one, and the
linker takes the **union** of the cells it scans. Six pairs over five projects; stated
because a chain would have been the tidier claim and it is not the true one.

WHAT THE RELATION DOES NOT DO. Nothing here admits a link. Every scan produces a
*candidate* for an LLM judge, and `pilot/rule_audit.py --only A1` asserts the column is
empty: 0 predicates put a link in the output without a verdict. The relation reads only
the runtime catalog -- component names and the discovered alias table -- plus one word
list, the English inflectional endings of ``INFLECTIONS``. No benchmark vocabulary
appears in it and none can (GATE-06).

ALSO REMOVED. ``_antecedent_states_name`` was a one-line wrapper over
``_states_a_name`` with a single call site; the call site now asks the predicate
directly.

TWO DEFECTS CARRIED FORWARD DELIBERATELY, both priced rather than quietly fixed:

  * ``_inside_qualified_identifier`` tests ``before in "-_"`` with ``before == ""`` for
    a sentence-initial span, and ``"" in "-_"`` is ``True`` in Python, so one span per
    sentence -- **378 over the five documents** -- is treated as sitting inside a
    qualified identifier. ``s_linker63`` repaired it and measured **FP +1.2
    (p = 0.01) at TP +/-0.0**, so on this benchmark the defect is load-bearing.
  * ``_all_occurrences_in_qualified_path`` lowercases the name and searches the raw
    sentence, so it can only see lowercase spellings. Handling case the way the rest of
    the module does would move **3 mention labels over all five projects**
    (``pilot/rule_audit.py --only A4``). Left alone here because this variant changes no
    behaviour; it is a separate arm, not a free repair.

The rest of this docstring is s_linker64's.

s_linker64 — s_linker62 plus the stated-name net the partial-name linker defers to.

ONE ADDITION over ``s_linker62``: ``_add_stated_name_net`` at the full-name proposer.
Its full account is on the method. In one line: the partial-name proposer declines a
pair whenever the sentence states a whole name, because that pair belongs to the
full-name linker -- and for 3.0 gold pairs per run the full-name linker's extraction
call never proposed it, so nothing in the workflow ever looks again. A deterministic
scan for the model name **as spelled** offers exactly those pairs to the unchanged
two-pass judge, and case is what makes it affordable: the same scan run
case-insensitively is 31.3 new pairs per run at 0.06 gold each, this one is 1.2 at
0.86. At the stage, five samples a side: **TP +1.2 (p = 0.01), FP +0.4 (p = 0.44)**.

The rest of this docstring is s_linker62's account of the partial-name proposer.

s_linker62 — s_linker59 with an inflection-bounded partial-name proposer.

ONE CHANGE from s_linker59: ``_name_word_candidates`` decided that a sentence word is
a word of a component's name with ``surface.startswith(word)``. That accepts *any*
continuation, not only an inflected one, and the two failures it causes were found by
auditing where the workflow's remaining errors live rather than by inspection.

WHY THE PARTIAL-NAME LINKER AT ALL. Over the six paired runs in
``../results/s5960_e2e_r*_20260813``, this workflow's false positives are almost
entirely in the two projects whose partial-name linker fires (teammates FP 6.7,
bigbluebutton FP 5.2, against 0.0/0.3/0.0 elsewhere), and its remaining recall loss is
8.0 gold pairs per run. ``pilot/partial_audit.py`` splits the stage's error budget:

    proposed 60.3 candidates/run, 18.7 of them gold -> 21.2 approved, 17.7 TP, 3.5 FP
    the denotation judge: 95% recall over the gold candidates, 83% precision
    a *perfect* judge over the same candidates: +1.0 TP, -3.5 FP

So the judge is not the bottleneck; the proposer is. ``pilot/partial_gap.py`` accounts
for the 22.8 gold pairs still open at this stage that it never offers:

    15.0/run  no word of the sentence relates to the name at all -- and the
              coreference linker recovers **every one** of them. Division of labour,
              not loss.
     5.8/run  the sentence states a whole name, so the proposer defers to the
              full-name linker, which did not produce the pair. 5.7 lost outright.
     2.0/run  a hook exists but the proposer requires a unique owner and found two.

THE REPAIR THAT WAS REFUTED FIRST. The 5.8 looks like the bigger prize, and the
obvious repair is to defer only where the full-name stage actually ruled on the pair
rather than wherever a name is stated. Measured deterministically over all six runs
(``pilot/partial_hole.py``): **+0.7 gold, +10.0 spurious**. The whole-name test is not
only a hand-off, it is the alias table doing suppression work -- the same dual role the
merged-alias round priced from the other side -- so nearly every sentence containing
`logic` states an alias of `Logic` and would become a candidate. Not adopted.

THE REPAIR ADOPTED. The 2.0 is a defect in the ownership test itself.
``WebRTC`` is an exact name word of ``WebRTC-SFU`` *and* a prefix continuation of
``web`` in ``BBB web``, so the proposer sees two owners and drops the pair; two gold
links go with it every run. Bounding the prefix to English inflections fixes it from
the other direction -- ``rtc`` is not an inflection of ``web`` -- and states the
morphology the old docstring already claimed. Deterministic screen over all five
documents (``pilot/partial_screen.py``):

    base (bare prefix)                60.3 candidates, 18.7 gold
    exact match outranks prefix       +2.0 gold, +1.0 spurious
    **inflection-bounded prefix**     **+2.0 gold, +0.0 spurious**

The inflection bound dominates: the same two gold candidates, and it also drops
``webcams -> BBB web``, which the exact-match ranking keeps. With the real denotation
judge behind it, five samples a side (``pilot/partial_pilots.py --pilot proposer``):
**TP +2.0 (p = 0.01), FP +1.0 (p = 0.01)** -- bigbluebutton reaches 61.7 of its 62
gold pairs.

NOT TAKEN, AND WHY (see ``s_linker63``). ``_inside_qualified_identifier`` computes
``before = text[start - 1] if start else ""`` and then tests ``before in "-_"``.
``"" in "-_"`` is ``True`` in Python, so every span starting at a sentence's first
character is reported as sitting inside a qualified identifier and dropped -- 344 spans
per run across the five documents. It is a defect, and repairing it on this benchmark
costs **FP +1.2 (p = 0.01) at TP +/-0.0**: the spans it wrongly hides are two spurious
candidates the judge would approve. ``s_linker63`` is this variant with the guard, so
the defect is priced rather than quietly kept.

Everything below describes s_linker59 and is unchanged.

The E2E rounds priced prompt *families*; `pilot/prompt_stage_pilots.py` prices the
clauses inside them, by replaying one stage with the two wordings against the same
recorded inputs. Eleven arms, five samples each, no end-to-end run. Three clearings
and four refusals:

  FREE (this variant takes them)
    the coreference family              TP -1.5 / FP -1.5 end-to-end (s55), and the
                                        judging rubric alone reads FP **-1.2**
                                        (p = 0.05) at its own stage
    `P1_FOCUS`                          TP +/-0.0, FP +0.2, both p = 1.00
    `DOC_KNOWLEDGE_JUDGE_RULES`         TP +/-0.0, FP -0.4, both p = 1.00

  NOT FREE (this variant keeps s49's wording)
    `LAYERED_ENTITY_RULES`              FP **+2.4** (p = 0.01) on the same candidates
    `ENTITY_EXTRACTION_RULES`           FP **+20.2** (p = 0.01), TP +6.2
    the alias proposer rules            FP +1.8 (p = 0.33) -- not significant, but the
                                        family it belongs to costs 1.1 F1 end-to-end
    the coreference prompt's preamble   TP **-16.2** (p = 0.01) when deleted; it is an
                                        input-format contract, not duplicated text

So the full-name judge splits: its 289-byte focus line generalizes for nothing while
its 692-byte rubric does not, which no family-level arm could see. Rule text
4022 -> 2960 B (-26%); instruction bytes per five-project run 60 892 -> 40 081 (-34%).

The rest of this docstring is s_linker55's.

s_linker55 — only the coreference side generalized; the safe maximum.

Fourth arm of the bisect of `s_linker51`. `s_linker50` already showed that
generalizing `COREF_RULES` alone is neutral on the one stage it can reach, and the
round-2 arms point the loss at the knowledge and full-name families. This arm is
what survives if both of those must keep their enumerations: the three coreference
rules general, everything else exactly s_linker49's.

It is the deliverable version of the question "how much of this prompt suite is
accreted?" — not the maximum trim measurable, the maximum trim that holds.

Rule text 4022 B -> 3249 B (-19%); instruction bytes per five-project run
60 892 -> 42 050 (-31%), because `COREF_RULES` alone is half the instruction budget.
`pilot/test_s54_s55_prompts.py` asserts the containment relation.

The rest of this docstring is s_linker51's.

s_linker55 — every prompt rule stated as a guideline instead of an enumeration.

The claim this variant tests is the one a reviewer will ask for directly: that the
workflow's prompts are *general guidelines*, not a rulebook accreted against five
benchmark documents. s_linker50 tests it on the single largest clause; this tests
it on all ten at once, so the two together say whether the answer is local or
general.

Nine of the ten rule constants are rewritten (P2_FOCUS is already a general
question and is carried verbatim). No prompt, call, stage, batch constant or
method body changes — only the wording of the rules — and every removal is an
enumeration replaced by the principle it enumerates:

  DOC_KNOWLEDGE_EXTRACTION_RULES  the three alias shapes -> "surface forms"
  ALIAS_EXCLUSION_RULES           `X.Y` / `X.Y.Z` -> "code-level identifiers"
  DOC_KNOWLEDGE_JUDGE_RULES       three ways of saying "not that one component"
  ENTITY_EXTRACTION_RULES         the "even if semantically related" aside
  P1_FOCUS                        the gloss of "architectural participant"
  LAYERED_ENTITY_RULES            four numbered reject-conditions -> one principle
  COREF_VALIDATION_FOCUS          three listed pronoun forms
  LAYERED_COREF_RULES             the claim gloss and three fragment shapes
  COREF_RULES                     clause (b), the role-phrase list, alias shapes

4022 B of rule text become 2461 (-39%), and 60.9 kB of instruction sent per
five-project run become 34.1 kB (-44%).

Two properties are deliberately preserved verbatim in force, because both are
measured elsewhere in this series and neither is an enumeration: the full-name
gate's approve-by-default and the alias judge's "when uncertain, prefer APPROVE"
(removing that leniency collapsed MediaStore's recall to 61.3% in three separate
variants), and the coreference gate's reject-when-uncertain. The asymmetry between
them is the design, not the wording.

Sizes for each removal are in `pilot/prompt_audit.py`, read off six recorded s49
runs; `pilot/test_s50_s51_prompts.py` asserts that these constants are the only
difference from s_linker49.

The rest of this docstring is s_linker49's.

s_linker49 — one mechanism fewer and eight condition copies fewer, composed.

s_linker47 and s_linker48 each held against s_linker25 over six paired runs, and they
touch different things, so this composes them. Both results first:

  s47  the grounded identity review removed. TP +6.2 (p = 0.00), FP +6.8 (p = 0.00),
       **macro F1 +0.2 (p = 0.53), macro F2 +1.3 (p = 0.01)** -- an LLM stage, a prompt,
       an anchor-bookkeeping block and a four-conjunct gate gone, F1 unchanged and F2
       significantly better.
  s48  eight copies of three conditions in five shapes merged into three named
       predicates, plus three never-firing conjuncts deleted, with every prompt byte
       unchanged. TP +0.7 (p = 0.65), FP -1.3 (p = 0.50), F1 +0.3 (p = 0.50), F2 +0.2
       (p = 0.57), and a **composition statistic of -0.2 (p = 0.59)**: the two arms' link
       sets differ less between arms than within, which is what a behaviour-preserving
       merge should look like.

Composing them is not additive by default -- this workflow has seven instances of an arm
that held alone and failed in another composition -- so the six paired runs decide it.

Note what composition removes from s48's side: with the identity review gone, its
duplicated claim-substring check has only one call site left, so the merge that mattered
there becomes moot and `_claim_supported` is not carried. What remains of s48 is the two
merges whose duplication survives:

  `_states_a_name`  the identical expression at three sites -- the full-name admission
                    filter, the partial-name whole-name exclusion, and the coreference
                    antecedent gate.
  `_window`         "the sentences within CONTEXT_SENTENCES of this one", spelled two
                    ways in s_linker25 (an `abs(...) <= C` filter in the denotation step
                    and a `range(max(1, n-C), n+C+1)` walk against the sentence map in
                    the coreference resolver) and once more in the review this variant
                    deletes.

So the design this variant describes is: the partial-name linker judging in **one step**
like the coreference linker -- four judging steps in the workflow rather than five -- and
two named conditions where there were eight copies. The full-name judge keeps its two
focused calls on purpose: merging those is the one change this series measured as
significantly worse (`s_linker36`, F1 -0.7 and FP +3.5, both p = 0.01).

The rest of this docstring is s_linker47's account of the mechanism removal.

The partial-name linker judges once, not twice.

ONE MECHANISM REMOVED from s_linker25: the grounded identity review. Everything else
below describes s_linker25 and is unchanged.

s_linker25 judges a partial-name candidate in two LLM steps. Step 1 asks, with the
target component deliberately withheld, whether the expression itself denotes a
software participant -- that target-blindness is worth 12 false positives and stays.
Step 2 then shows the target together with the sentences that state one of its names
and asks whether the two denote the same participant. Step 2 has never been priced on
its own, and the six runs of s_linker25 recorded in `../results/s4546_e2e_r*_20260812`
price it now:

  20.3 candidates per run reach the identity review; it keeps 12.3 (12.2 gold) and
  rejects 8.0, **of which 5.5 are gold.**

So the step trades **5.5 true positives for 2.5 false positives per run** -- a bad
trade for F1 and a worse one for F2. Dropping it makes the partial-name linker
structurally the same shape as the coreference linker: one proposer, one judging call.
It also deletes the review's prompt, its anchor bookkeeping, its four-conjunct evidence
gate and the `alternative` response field.

Both estimates are deterministic reads off recorded decisions, and the pipeline can
still move them: partial-name links feed `_unlinked`, so approving 5.5 more of them
removes 5.5 pairs from the coreference linker's input. Only the end-to-end runs decide.

s_linker25's own description follows.

Three linkers in fixed name-evidence order.

A sentence gives a component's whole name, part of it, or only a reference to
it. \\approach tries them in that order:

1. FULL-NAME linker   — the sentence states a name of the component (its model
   name, a discovered alias, or a spelling variant of either). One extraction
   pass, a lexical stated-name contract filter, then the two-pass evidence
   judge.
2. PARTIAL-NAME linker — the sentence carries one word of a name and states no
   whole name; the whole name appears elsewhere in the document. Deterministic
   proposal, then a two-step judge: denotation *without* the target, then
   grounded identity against a sentence that states the name.
3. COREFERENCE linker — the sentence states no name and is resolved through the
   earlier sentence it refers back to. Per-sentence LLM discovery, structural
   antecedent gate, single-pass judge.

Each linker runs exactly once and sees only what the earlier ones left unlinked.
That is one rule, ``_unlinked``, applied by all three at their candidate
boundary; an earlier revision applied it in the partial-name linker alone, which
left the sentence above true of the pipeline's output but not of its work.

MEASURED DESIGN DECISIONS. Every arm below is five runs per side on all five
projects, scored against the gold standard, with the permutation test in
``pilot/ab_stats.py``; the scripts are ``pilot/design_audit.py`` (deterministic
sizing from the promoted run's checkpoints) and ``pilot/design_pilots.py``.

  * SUBTRACTION IN ALL THREE LINKERS. 57% of the coreference judge's cases used
    to restate a pair the union already held. Removing them cannot change the
    final set, but it changes the batches the remaining cases are judged in:
    -6.8 false positives (p=0.01) at +0.8 true positives (p=0.05). Adopted.
  * ONE EXTRACTION SAMPLE. The second sample of the same prompt moved neither
    score beyond noise (TP -1.2, p=0.30; FP -1.2, p=0.42). Dropped, halving the
    extraction cost.
  * NO ALIAS SCOPE. Aliases were discovered with a "global"/"local" scope that
    only ever filtered the extraction prompt; every other consumer read the
    table without it. Offering all of them instead is worth +3.0 true positives
    (p=0.01) at +1.0 false positives (p=0.59). Scope dropped.
  * NO AMBIGUITY MAP. The model-understanding call classified component names
    as ambiguous, and its only consumer was one boolean of the evidence bundle.
    Removing the call, the field and the prompt is quality-neutral (TP -0.2,
    p=1.00; FP +0.8, p=0.40). Dropped.
  * CLAIM BEFORE VERDICT, KEPT. Each judge must quote the sentence words it
    rules on before deciding. The full-name and coreference judges do not
    string-match that quote, so it looks like unread output -- but removing the
    request costs 35.2 true positives (p=0.01), and enforcing it changes
    nothing: with the contiguity instruction the partial-name prompts carry, the
    check voided zero verdicts in 25 project-runs while the added instruction
    alone cost +1.6 false positives (p=0.02). The quote is a commit-to-text
    device where the name is already matched, and evidence to be verified where
    it is not (the partial-name judge, which does check it).
  * ONE COREFERENCE JUDGING PASS. A second pass, mirroring the full-name
    judge's, changes neither score (TP -0.6, p=0.40; FP -0.8, p=0.17). The
    asymmetry with the full-name judge stands as measured, not assumed.
  * SLIMMER EVIDENCE BUNDLE. The bundle carried a ``Rationale:`` line whose
    value was the same string for every candidate on every project, and it
    selected anchors with a second name-matching primitive
    (``has_standalone_mention``) where the rest of the linker uses
    ``_find_exact_form``. Dropping the line and using one primitive for anchors
    leaves recall untouched and removes 2.2 false positives (TP +0.0, p=1.00;
    FP -2.2, p=0.01). Adopted.
  * NO ``antecedent_via_alias``. The coreference prompt asked the model to
    self-report whether the antecedent used an alias, spent a 488-byte rules
    block defining it, and no gate ever read the answer. Removing the request,
    the block and the response field is neutral-to-better (TP +0.6, p=0.17;
    FP -0.8, p=0.05). Adopted.

MEASURED AND KEPT. Six hand-coded paths look arbitrary and are not; the numbers
come from ``pilot/complexity_audit.py``, ``pilot/simplify_pilots.py`` and
``pilot/gate_audit.py`` / ``pilot/gate_pilots.py``.

    READ THE FOLLOWING WITH THIS VARIANT'S NAMES. Every measurement below stands --
    they are behavioural results and this variant changes no behaviour -- but three
    of the methods they name no longer exist as separate code. ``_name_word_candidates``
    is ``SCANS["name_word"]``, ``_spelling_variant_candidates`` is
    ``SCANS["spelling"]``, and ``_antecedent_states_name`` is ``_states_a_name`` at
    its coreference call site. The claim below that the partial-name proposer is
    "the one hand-written proposer in the workflow" was already wrong when it was
    written -- there were four lexical rules by ``s_linker64`` -- and is now
    superseded outright: there is one relation, and the partial-name linker scans
    its loosest cell.

  * the LEXICAL ADMISSION FILTER (``_keep_stated_names``) is the one case where a
    single-stage arm pointed the wrong way. It rejects 22 of 228 extractor
    proposals, 9 of them gold, and on its own stage removing it is F2-positive
    (F2 +0.9, p=0.01). Composed, three five-project runs without it hold recall
    (TP 182.0 vs 182.3) and quadruple the false positives (17.3 vs 4.3), macro F2
    94.9 vs 95.9. Kept, and the episode is why every arm here is now confirmed
    end-to-end before adoption;
  * the COREFERENCE ANTECEDENT GATE (``_antecedent_states_name``) is the largest
    code-driven rejection left: it blocks 20 of 133 reported resolutions, 7 of
    them gold. Removing it and letting the strict judge decide alone is worse on
    every score (TP -1.2, p=0.03; FP +12.0, p=0.01; F1 -3.2; F2 -1.7, p=0.01);
  * the PARTIAL-NAME PROPOSER (``_name_word_candidates``) is the one hand-written
    proposer in the workflow. Replacing it with an LLM asked the same question
    directly -- which sentences refer to a component by part of its name only --
    recovers 4.0 of the 11.0 gold links the rule reaches (F2 -4.4, p=0.01). One
    generic prompt was tried, not a prompt search, so this bounds the swap rather
    than refuting it in general;

  * the mention-type classifier (``MentionType``, ``_classify_mention_typed``,
    ``_all_occurrences_in_qualified_path``) produces one substring of one judge
    prompt, and all five of its values occur in practice (122 proper-standalone,
    42 via-alias, 11 code-token, 10 lowercase, 3 indirect across the five
    projects). Removing the field costs 6.6 true positives (p=0.01);
  * ``_spelling_variant_candidates`` generates 6 candidates across all five
    projects and wins 2 gold links, both on BigBlueButton, that extraction never
    proposed. Small, but it is recall the extractor cannot reach;
  * ``_inside_qualified_identifier`` is a four-way disjunction and all four
    disjuncts do work: of 721 suppressed spans, 175 are suppressed by the dotted
    tests alone (a span adjacent to ``.`` is not adjacent to an alphanumeric, so
    the joined tests cannot see it). The test does not reduce to an adjacency
    test.

ONE NAME-MATCHING PRIMITIVE. The module used to hold two: a strict
``has_standalone_mention`` (case rules for single-word names, dot and hyphen
boundaries) deciding the mention type and the coreference antecedent gate, and a
lenient ``_find_exact_form`` deciding admission, the partial-name suppressor and
the anchors. They disagree on 47 of 3697 (name, sentence) pairs, always with the
lenient one matching, and flip the antecedent gate on none of the promoted run's
resolutions; replacing the strict one in the classifier is indistinguishable on
that stage (TP +/-0.0, p=1.00; FP -0.4, p=0.76) and inside the end-to-end band
(F1 96.47 over three runs against 96.42 +/- 0.43 over six). So the strict
primitive is gone and ``_find_exact_form`` decides every "is this name stated
here" question in the workflow.

STANDALONE. Earlier revisions made this a subclass chain
``SLinker68 -> SLinker24RoleOrchestrator -> _SLinker24OrchestratorBase ->
SLinker21``. That chain is now inlined here and the module has no linker
superclass, matching the convention `s_linker21` set for paper artifacts.
Inlining dropped four dead surfaces the chain carried:

  * the LLM controller (``_choose_tool``, ``_tool_catalog``,
    ``_controller_link_view``) — it selected the identical order on all five
    projects in both promoted runs, so the order is stated instead;
  * the ``coverage_audit`` linker and its prompt — never in this variant's
    linker set;
  * ``MentionType.ANAPHORIC`` — defined but never produced;
  * the empty ``_project_profile`` document dump the controller consumed.

Two dead prompt surfaces were also dropped. ``AMBIGUITY_FEW_SHOT`` and
``DOC_KNOWLEDGE_JUDGE_EXAMPLES`` were empty strings still interpolated into two
prompts; deleting the constants while keeping the blank line leaves both
rendered prompts byte-identical, so no decision can change. The
``antecedent_via_alias`` machinery went the same way but by measurement rather
than by inspection: its ``Examples:`` block held the only concrete invented name
in the prompt set and was cut first, and the whole surface -- request sentence,
rules block and response field -- was removed once the arm above priced it.

It also restores per-phase LLM metrics, which the S24 base set to ``{}`` and
never computed.

A later cleanup pass removed three surfaces that no decision reads. Each is
prompt-invariant: every ``_prompt_*`` builder, both deterministic generators,
and the denotation and identity prompts rendered from real project data compare
byte-identical before and after on all five benchmarks.

  * the partial-name linker re-read the document from a ``_current_text_path``
    attribute to build a sentence list identical to the one ``link()`` had
    already loaded. It now takes that list, and the attribute is gone;
  * the coreference judge carried a keep-and-warn branch for a link whose
    sentence is missing from ``sent_map``. The hole it guarded was upstream:
    ``_resolve_references`` checked the antecedent sentence number the model
    reported but not the target one, so an invented target number became an
    approved link that ``_link_view`` then hid from the trace. The number is
    checked where the resolution is read; a number that names no sentence
    cannot match a gold link, so the check can only drop false positives. The
    judge indexes ``sent_map`` directly;
  * the denotation step batched with a hand-rolled ``range`` slice instead of
    ``_iter_batches``, the helper every other batching site uses.

Link ``source`` tags are renamed to the paper's vocabulary. Prior variants
emitted ``entity`` / ``s24_entity_orthographic`` / ``s24_relation_role`` /
``coreference``; this variant emits ``full_name`` / ``full_name_variant`` /
``partial_name`` / ``coreference``.

Three differences from the inherited chain are behavioural and must be
measured, not assumed:

  * the shared judge prompt, the extraction prompt and the alias judge prompt
    are byte-identical to s_linker21's, as are nine rubric constants, and both
    deterministic candidate generators produce identical candidate sets — all
    asserted in ``pilot/test_s25_standalone.py``;
  * ``_prompt_coref`` and ``COREF_RULES`` diverge by the removed
    ``antecedent_via_alias`` surface, and ``_prompt_doc_knowledge_extract`` by
    the removed alias scope. Coreference resolutions and alias sets are
    therefore not guaranteed identical; both divergences are priced above;
  * the denotation and identity prompts now carry a wider evidence window
    (``CONTEXT_SENTENCES``) and a longer anchor list (``ANCHOR_LIMIT``), so
    those two judges see more text than they did. Link decisions from those two
    steps are therefore not guaranteed identical.

The knowledge module is the alias table alone: one extraction call over the
document and one judging call over the proposed mappings. Aliases are stored as
the declared ``DocumentKnowledge`` type, a term-to-component-name map.

  * THE ALIAS JUDGE STAYS. On the full-name stage in isolation, removing it
    looks like a gain (TP +4.6, p=0.04; F2 +1.9, p=0.04) -- an unjudged table is
    larger, so more candidates are admitted. Composed, the noisy aliases it lets
    through cost precision and recall alike: TP 179.0 against the 180.8 of the
    reference band, FP 8.7 against 4.8, macro F1 94.57 against 96.42 +/- 0.42.
    Reverted.
  * THE ALIAS PASS AND THE EXTRACTOR ARE THE SAME QUESTION, ASKED TWICE, and
    keeping them apart is worth 2.2 F1. ``s_linker26`` merges them -- one prompt
    per sentence batch returning both the references and any name the passage
    establishes, table accumulated and fed forward. On this stage the merge is
    exactly neutral (TP +0.2, p=1.00; F2 +/-0.0, p=0.98); over three five-project
    runs it gives macro F1 94.27 and F2 93.47 against this variant's 96.42 +/-
    0.42 and 95.38 +/- 0.58, at TP 175.7 vs 180.8 and FP 11.0 vs 4.8. A batch
    cannot see a definition stated elsewhere, and nothing judges what it collects.
    The extraction prompt does need a table -- removing the ``KNOWN ALIASES`` line
    costs 5.2 true positives (p=0.02) and 2.0 F2 (p=0.02) -- so the up-front pass
    earns its place rather than merely occupying it. See
    ``pilot/alias_integration_*.py`` and ``../results/s26_unified_e2e_r1_*/``.

Every rubric is generic English structure — no benchmark vocabulary. The only
project-specific input is the runtime component set and the aliases discovered
from the document.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

from llm_sad_sam.core.data_types_v2 import (
    SadSamLink, CandidateLink, DocumentKnowledge,
)
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.linkers.experimental.helper_v3 import (
    parse_snum, get_comp_names,
)
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend, LLMResponse

# ─────────────────────────────────────────────────────────────────────────────
# Prompt constants. GENERALIZED — every enumeration of the inherited chain is
# restated as the principle it enumerates. Each removal is sized in
# `pilot/prompt_audit.py` off six recorded s49 runs; the sizes are quoted at each
# constant. Nothing here is benchmark-derived vocabulary (GATE-06): the
# generalized text is strictly *less* specific than what it replaces, because the
# concrete forms it drops (`X.Y.Z`, "the module"/"the service"/"the system",
# "gerund phrase", "list item") are the only shapes in these prompts that named a
# surface form rather than a property.
# ─────────────────────────────────────────────────────────────────────────────

# Dropped: "names the whole system", "names a different entity" and the grouping
# clause — three ways of saying "identifies something other than that one
# component". The leniency sentence stays: it is what keeps a third of MediaStore
# (measured when the alias judge was removed, F1 94.57 vs 96.42).
DOC_KNOWLEDGE_JUDGE_RULES = """An alias is valid when the document establishes an equivalence between a phrase and a single named component. It is invalid when the phrase is generic vocabulary or identifies anything other than that one component. When uncertain, prefer APPROVE."""

# Dropped: the parenthetical listing the three alias shapes. What the extractor is
# asked for is a surface form for one component; which shapes qualify is the
# model's judgement, and the judge that follows is where validity is decided.
DOC_KNOWLEDGE_EXTRACTION_RULES = """Find surface forms the document uses to refer to a single named component (introduced short forms, alternate names, or words of multi-word names when they alone clearly mean the full name). Reject terms whose ordinary English use dominates."""

# Dropped: the spelled-out `X.Y` / `X.Y.Z` shapes. Documents with any dotted
# identifier at all are one of five benchmarks (62/198 sentences on that one, 0-6
# on the rest), so naming the shape is a rule written for one corpus.
ALIAS_EXCLUSION_RULES = """Qualified-name fragments (package- or member-access paths of the form X.Y or X.Y.Z) are NOT aliases — do not include them."""


# THE ONE CHANGE from s_linker68: the full-name linker's admission contract is
# stated here instead of enforced afterwards. s65 asked the extractor for anything
# that "refers to the component ... as a participant in a described interaction" and
# then deleted every proposal whose sentence writes no name of the component
# (`_keep_stated_names`, now gone). The two sentences below are that filter, in the
# register of the prompt it joins; the rest of the paragraph is s65's unchanged.
ENTITY_EXTRACTION_RULES = """Include a reference only when the sentence itself writes the component's name or one of the KNOWN ALIASES. Exclude a component that the sentence only implies as a participant in a described interaction without naming it, and exclude a name that appears only inside a code-level path -- even if the compound identifier is semantically related to the component -- or as ordinary English with no architectural intent. Favor inclusion among the sentences that do name it."""


# Dropped: the three-way gloss of "architectural participant" and the qualified-name
# example. The distinction the question draws is unchanged.
P1_FOCUS = (
    "Check architectural participation: does the sentence name this "
    "component as an architectural participant, rather than only as part "
    "of a code-level identifier?"
)

P2_FOCUS = (
    "Check referential specificity: is the component name used to identify "
    "this specific architectural element, or does it serve as a generic "
    "technical term in this sentence?"
)


# Dropped: the three listed pronoun/noun-phrase forms and the gloss of
# "architectural participant".
COREF_VALIDATION_FOCUS = (
    "Check coref resolution: does the referring expression in this sentence "
    "actually refer to the named component as an architectural participant?"
)

# Dropped (P3/P4 of `pilot/prompt_audit.py`, six recorded s49 runs): clause (b),
# which licensed resolving to the section topic without a name repetition and
# licensed 0.0 of 578 recorded resolutions; the five listed role phrases, of which
# only "it" is used at volume; and the terminal-word/abbreviation enumeration
# (1.7 antecedents per run), subsumed by "under any form the document uses for it".
COREF_RULES = """For each case, decide whether a pronoun or noun phrase that refers back in the target sentence refers back to a component named or aliased earlier in the context. Resolve when the surrounding sentences make one component the clear antecedent, under any form the document uses for it. Avoid resolving when two or more equally plausible antecedents exist."""

# Full-name gate — lenient: a stated name is a link unless a reject signal fires.
# The default and the leniency are kept verbatim in force; what goes is the
# numbered form of the four reject-conditions, which become illustrations of one
# principle. Of 14.7 rejections per run, 2.2 match condition (1) and 1.8 condition
# (2) lexically, so 73% already rest on the two conditions that name no surface
# form at all.
LAYERED_ENTITY_RULES = (
    "Approve the link by default: the component is named here and the document treats "
    "it as part of the system. A bare mention, a heading, or a list that includes the "
    "component name all count as valid links — approve them. Reject ONLY when one of "
    "these clearly holds: (1) the component is referred to only through a code-level or "
    "package/member path of the form x.y or x.y.z, even if that path is described as "
    "doing something; (2) the mention is negated (it is NOT a ...); (3) the matching "
    "word actually names a DIFFERENT entity; (4) the matching word is used as a generic "
    "technique or technology term, not as this system's component. When none of these "
    "reject-conditions clearly applies, approve."
)

# Coreference gate — strict: the component is NOT named in the sentence, so demand
# a genuine referring expression plus an architectural claim. Dropped: the gloss of
# "architectural claim" and the three named fragment shapes.
LAYERED_COREF_RULES = (
    "These are coreference links: a pronoun or noun phrase in the sentence is claimed to "
    "refer back to the component, which is NOT named in the sentence itself. Approve only "
    "when the sentence contains a genuine referring expression that unambiguously points "
    "to THIS component and makes an architectural claim about it. Reject when there is no "
    "such referring expression, when the antecedent could equally be a different "
    "component, or when the reference is only to a code-level identifier. When uncertain, "
    "reject."
)

#: The endings under which a sentence word still counts as a word of a component's
#: name. English inflectional morphology, the same set for every document, and **the
#: only word list in this module**: no benchmark term appears here and none can
#: (GATE-06). Read at exactly one place, the ANY_WORD branch of ``_name_spans``.
#: (s_linker64's comment claimed a second list, "the stopwords"; the module has none.)
INFLECTIONS = ("", "s", "es", "ed", "d", "ing", "ings", "er", "ers")

#: The tokenizer both dimensions of the relation use to cut a name or a sentence into
#: words. Word boundaries only -- splitting compounds here was measured to triple the
#: candidate set while reaching no additional gold link.
WORD_PATTERN = r"[A-Za-z]+[A-Za-z0-9]*|\d+"


class NameForm(Enum):
    """A point of the surface-realization relation, on two independent dimensions.

    *Fidelity* -- how exactly the sentence's characters must reproduce the name:

        AS_SPELLED    the name, character for character, at word boundaries
        ANY_CASE      the name, ignoring case
        ANY_SPELLING  the name's word sequence, ignoring case, separators and
                      compound joining, so "X Y", "x-y" and "XY" are one form

    *Extent* -- how much of the name has to be present:

        ANY_WORD      one word of the name, under an English inflectional ending

    Which form a proposer scans is the only thing that distinguishes the workflow's
    three candidate generators from one another; see ``SCANS``.
    """

    AS_SPELLED = "as_spelled"
    ANY_CASE = "any_case"
    ANY_SPELLING = "any_spelling"
    ANY_WORD = "any_word"


@dataclass(frozen=True)
class SurfaceScan:
    """A candidate generator, stated as data.

    Every field except ``form`` and ``source`` is a shared option, so adding a
    generator is a row of ``SCANS`` rather than a method with its own regex.
    """

    #: Which point of the relation this generator scans for.
    form: NameForm
    #: The ``source`` tag its candidates carry into the evidence bundle and the logs.
    source: str
    #: Propose only when exactly one component of the catalog owns the surface. Off
    #: for the whole-name scan, where the surface *is* a catalog name.
    unique_owner: bool = False
    #: Skip a sentence that already writes a whole name of the component: that pair
    #: belongs to the full-name linker.
    skip_when_named: bool = False
    #: Skip a surface that already realizes the name at a stricter fidelity -- it is
    #: the plain name, not a variant of it.
    skip_stricter: bool = False
    #: Skip a span glued into a dotted path or a larger word.
    skip_qualified: bool = True
    #: Attach the mention label the full-name judge reads.
    label_mention: bool = False


#: The workflow's three candidate generators. Four lexical rules in ``s_linker64``;
#: three rows and one filter here, all reading the same relation.
#:
#:   `stated_name`  the recall floor under the LLM extractor. Tightest fidelity, and
#:                  the table in the module docstring is why: 0.955 gold per pair
#:                  against 0.773 one row looser. A component named `Common` or
#:                  `Client` matches ordinary English on every page, and the
#:                  capitalization is what separates the proper noun from the common
#:                  one. It skips no qualified span, matching the admission filter's
#:                  measured asymmetry (see ``_keep_stated_names``).
#:   `spelling`     the forms that differ only in how the name's words are joined.
#:   `name_word`    one word of a name, for the partial-name linker.
SCANS = {
    "stated_name": SurfaceScan(
        form=NameForm.AS_SPELLED,
        source="stated_name_candidate",
        skip_qualified=False,
        label_mention=True,
    ),
    "spelling": SurfaceScan(
        form=NameForm.ANY_SPELLING,
        source="full_name_variant",
        unique_owner=True,
        skip_stricter=True,
    ),
    "name_word": SurfaceScan(
        form=NameForm.ANY_WORD,
        source="partial_name_candidate",
        unique_owner=True,
        skip_when_named=True,
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Tracing infrastructure — per-LLM-call audit trail
# ─────────────────────────────────────────────────────────────────────────────

_phase_local = threading.local()


def _current_phase() -> str:
    return getattr(_phase_local, "phase", "unknown")


class _TracingLLMClient:
    """Delegating wrapper that records every query() into a phase-tagged trace."""

    def __init__(self, inner: LLMClient, sink: list[dict]):
        self._inner = inner
        self._sink = sink
        self._sink_lock = threading.Lock()

    def set_phase(self, name: str) -> None:
        _phase_local.phase = name

    def query(self, prompt: str, timeout: int = 180, max_retries: int = 3) -> LLMResponse:
        phase = _current_phase()
        t0 = time.time()
        try:
            resp = self._inner.query(prompt, timeout=timeout, max_retries=max_retries)
        except Exception as exc:
            record = {
                "phase": phase, "ts": t0,
                "elapsed_s": round(time.time() - t0, 3),
                "timeout": timeout, "max_retries": max_retries,
                "prompt": prompt,
                "response_text": None,
                "success": False,
                "error": f"FATAL: {exc}",
                "latency_ms": None,
                "model": None,
            }
            with self._sink_lock:
                self._sink.append(record)
            raise
        record = {
            "phase": phase, "ts": t0,
            "elapsed_s": round(time.time() - t0, 3),
            "timeout": timeout, "max_retries": max_retries,
            "prompt": prompt,
            "response_text": getattr(resp, "text", None),
            "success": getattr(resp, "success", None),
            "error": getattr(resp, "error", None),
            "latency_ms": getattr(resp, "latency_ms", None),
            "model": getattr(resp, "model", None),
        }
        usage = getattr(resp, "token_usage", None)
        if usage is not None:
            record["token_usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        with self._sink_lock:
            self._sink.append(record)
        # A phase result may only be interpreted after every required request
        # succeeds. Returning a failed response lets extract_json() turn it into
        # None and silently omit an entire batch.
        if not resp.success:
            raise RuntimeError(f"LLM request failed in {phase}: {resp.error}")
        return resp

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

class MentionType(Enum):
    """How a component name appears in a sentence."""
    PROPER_STANDALONE = "proper case, standalone"
    LOWERCASE_PROSE = "lowercase mention"
    VIA_ALIAS = "via known alias"
    INDIRECT = "indirect/unclear match"


@dataclass
class EvidenceBundle:
    """What a judge is told about a candidate beyond the candidate itself.

    The matched span and the preceding sentence also appear in the case header
    the judge reads (``Case n: "span" -> Component``, then the sentence with its
    ``[prev: ...]`` prefix), so the bundle repeats both. The repetition is
    deliberate and was verified the hard way: dropping either is neutral on the
    judging stage in isolation (span TP +0.8, F2 +0.3, all p >= 0.44; preceding
    sentence TP -0.4, F2 -0.2, p >= 0.30) and costs precision once composed --
    three five-project runs without them hold recall (TP 182.0) and lose it on
    false positives (8.3 against the 4-6 of the six-run reference band), F1 95.2
    against 96.42 +/- 0.42. Repeating the evidence next to the rubric is not
    redundant for the model.
    """

    source: str
    matched_span: str
    mention_type: str          # MentionType.value (str for prompt embedding)
    preceding_text: str
    anchor_sentences: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Main linker
# ─────────────────────────────────────────────────────────────────────────────

class SLinker68:
    """Three linkers, fixed name-evidence order, no controller. Standalone."""

    _VARIANT_NAME = "s_linker68"

    #: Execution order. Full name first (it needs the least), partial name
    #: second, coreference last. The partial-name linker is the only one that
    #: subtracts already-linked pairs, so it must not run first.
    LINKERS = ("full_name", "partial_name", "coreference")

    # ── Resource bounds ──────────────────────────────────────────────────────
    # These cap prompt size and call count. No decision rule reads them:
    # changing one changes how much text a judge sees, never what counts as a
    # link. Named here so they are auditable in one place.
    #
    # Every evidence window is the same width and every anchor list the same
    # length, on purpose — the earlier per-step values (2, 3, 4, 5) implied a
    # calibration that was never measured. One width was verified not to
    # weaken the target-blind denotation step: that step's blindness comes from
    # withholding the target label from the case, not from hiding sentences
    # that name components, and a naming sentence is already visible in the
    # shared batch table for the large majority of candidates at any width.
    CONTEXT_SENTENCES = 5          # sentences either side shown to any judge
    ANCHOR_LIMIT = 5               # naming sentences offered as evidence
    EXTRACTION_BATCH = 50          # sentences per full-name extraction call
    JUDGE_BATCH = 25               # candidates per judging call (all judges)
    COREFERENCE_BATCH = 10         # sentences per coreference-resolution call
    ASK_ATTEMPTS = 2               # initial call + one retry on an empty parse

    def __init__(
        self,
        backend: LLMBackend | None = None,
        model: str | None = None,
        checkpoint_fallback: LLMBackend | str | None = None,
        checkpoint_fallback_model: str | None = None,
        no_knowledge: bool = False,
    ):
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.4")
        real_llm = LLMClient(
            backend=backend or LLMBackend.CLAUDE,
            model=model,
            checkpoint_fallback=checkpoint_fallback,
            checkpoint_fallback_model=checkpoint_fallback_model,
        )
        self._llm_calls: list[dict] = []
        self.llm = _TracingLLMClient(real_llm, self._llm_calls)
        self.no_knowledge = no_knowledge
        self.doc_knowledge: DocumentKnowledge | None = None
        self._phase_log: list[dict] = []
        self._phase_metrics: dict[str, dict] = {}
        self.workflow: list[dict] = []
        print("SLinker68 (full-name -> partial-name -> coreference; fixed order)")
        print(f"  Backend: {self.llm.describe_backend()}")

    # ── Main entry ───────────────────────────────────────────────────────────

    def link(self, text_path, model_path, **_kwargs):
        self._phase_log = []
        self._llm_calls.clear()
        self._phase_metrics = {}
        started = time.time()

        components = parse_pcm_repository(model_path)
        sentences = load_sentences(text_path)
        name_to_id = {component.name: component.id for component in components}
        sent_map = build_sent_map(sentences)
        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        print("\n[Knowledge] Document aliases")
        self.doc_knowledge = (
            DocumentKnowledge() if self.no_knowledge
            else self._learn_document_knowledge(sentences, components)
        )
        self._save_phase(text_path, "knowledge",
                         {"doc_knowledge": self.doc_knowledge})

        current: list[SadSamLink] = []
        history: list[dict] = []
        for linker in self.LINKERS:
            print(f"\n[Linker] {linker}")
            produced, feedback = self._run_linker(
                linker, sentences, components, name_to_id, current, sent_map
            )
            current = self._union(current, produced)
            history.append({
                "linker": linker,
                "feedback": self._linker_feedback(feedback),
            })
            self._save_phase(text_path, f"linker_{linker}", {
                "links": produced, "feedback": feedback, "workflow": history,
            })

        self.workflow = history
        self._phase_metrics = self._compute_phase_metrics()
        self._log(
            "s25_summary",
            {"components": len(components), "sentences": len(sentences)},
            {
                "workflow": history,
                "final": len(current),
                "elapsed_s": round(time.time() - started, 2),
                "llm_calls": len(self._llm_calls),
                "phase_metrics": self._phase_metrics,
            },
            current,
        )
        self._save_phase(text_path, "final", {
            "final": current,
            "workflow": history,
            "elapsed_s": round(time.time() - started, 2),
        })
        self._save_log(text_path)
        print(f"\nFinal: {len(current)} links "
              f"({time.time() - started:.1f}s, {len(self._llm_calls)} LLM calls)")
        return current

    def _run_linker(self, linker, sentences, components, name_to_id, linked, sent_map):
        """Dispatch, with the linked set passed to every linker without exception.

        This is what makes "each linker sees only what the earlier ones left
        unlinked" a property of the pipeline rather than of one linker. The
        subtraction itself is `_unlinked`, called once inside each linker at its
        candidate boundary.
        """
        if linker == "full_name":
            return self._run_full_name_linker(
                sentences, components, name_to_id, linked, sent_map)
        if linker == "partial_name":
            return self._run_partial_name_linker(
                sentences, components, linked, sent_map)
        if linker == "coreference":
            return self._run_coreference_linker(
                sentences, components, name_to_id, linked, sent_map)
        raise RuntimeError(f"unknown linker: {linker!r}")

    @staticmethod
    def _unlinked(candidates, linked):
        """Drop every proposal for a pair an earlier linker already produced.

        Removing them cannot change the final link set -- the union already
        holds each one -- but it does keep them out of the judging batches they
        would otherwise share with the pairs still in question. Measured over
        five runs on all five projects, it removes 57% of the coreference
        judge's cases and, with them, 6.8 false positives (p=0.01) at +0.8 true
        positives (p=0.05).
        """
        return [c for c in candidates
                if (c.sentence_number, c.component_id) not in linked]

    # ── Concurrency and small helpers ────────────────────────────────────────

    @staticmethod
    def _run_parallel(tasks):
        if len(tasks) == 1:
            name, fn = next(iter(tasks.items()))
            return {name: fn()}
        results = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(fn): name for name, fn in tasks.items()}
            try:
                for fut in as_completed(futures):
                    results[futures[fut]] = fut.result()
            except Exception:
                for other in futures:
                    other.cancel()
                raise
        return results

    @staticmethod
    def _iter_batches(items, n):
        """Yield (batch_num, batch_slice) — batch_num is 1-indexed."""
        for i, start in enumerate(range(0, len(items), n), start=1):
            yield i, items[start:start + n]

    @staticmethod
    def _prev_prefix(snum, sent_map) -> str:
        prev = sent_map.get(snum - 1)
        return f"[prev: {prev.text}] " if prev else ""

    @classmethod
    def _find_exact_form(cls, text, expression):
        """The first writing of ``expression`` in ``text`` at ANY_CASE, or "".

        The relation's middle fidelity, returning the surface rather than the span,
        because its three callers want to know *what* matched: the mention label
        compares it against the name, and the two name predicates only test it.
        """
        spans = cls._name_spans(text, expression, NameForm.ANY_CASE)
        return text[spans[0][0]:spans[0][1]] if spans else ""

    def _states_a_name(self, text: str, comp_name: str) -> bool:
        """Does this sentence state the component's name, or one the document gave it?

        One predicate for a question three stages asked with three copies of the same
        expression: the full-name admission filter, the partial-name proposer's
        whole-name exclusion, and the coreference antecedent gate. The mention-label
        classifier asks it too but decomposed, because it must know *which* matched, so
        it keeps its own two calls to ``_find_exact_form``.
        """
        names = (comp_name, *self._names_by_component().get(comp_name, ()))
        return any(self._find_exact_form(text, name) for name in names)

    def _window(self, snum: int, sentences):
        """The sentences within ``CONTEXT_SENTENCES`` of this one, in document order.

        One predicate for a condition spelled two different ways: the denotation step
        filtered the sentence list with ``abs(s.number - target) <= CONTEXT_SENTENCES``
        while the coreference resolver walked ``range(max(1, n - C), n + C + 1)`` against
        the sentence map. Both select the same set -- the walk cannot reach a sentence the
        filter excludes, and it skips numbers the document lacks, which is what the filter
        does by construction. Verified over every sentence of all five documents.
        """
        return [s for s in sentences
                if abs(s.number - snum) <= self.CONTEXT_SENTENCES]

    def _names_by_component(self):
        """Discovered aliases grouped by component. The model name is added by
        callers; together they are the component's set of names N(c)."""
        aliases = getattr(getattr(self, "doc_knowledge", None), "aliases", {})
        names = {}
        for term, component in aliases.items():
            names.setdefault(component, []).append(term)
        return names

    @staticmethod
    def _union(existing, additions):
        """Merge by (sentence, component). Earlier linkers win ties."""
        result = list(existing)
        keys = {(link.sentence_number, link.component_id) for link in existing}
        for link in additions:
            key = (link.sentence_number, link.component_id)
            if key not in keys:
                result.append(link)
                keys.add(key)
        return result

    @staticmethod
    def _link_view(links, sent_map):
        return [
            {
                "sentence": link.sentence_number,
                "text": sent_map[link.sentence_number].text,
                "component": link.component_name,
                "source": link.source,
            }
            for link in links
            if link.sentence_number in sent_map
        ]

    @staticmethod
    def _decision_view(decisions):
        return [
            {"sentence": sentence, "component_id": component, **decision}
            for (sentence, component), decision in decisions.items()
        ]

    @staticmethod
    def _linker_feedback(feedback):
        """Reduce detailed linker evidence to accepted/rejected references."""
        proposed = feedback.get("candidates", feedback.get("proposed", []))
        accepted = feedback.get("accepted", [])
        accepted_keys = {(i["sentence"], i["component"]) for i in accepted}

        def reference(item):
            return {"sentence": item["sentence"], "component": item["component"]}

        return {
            "accepted": [reference(i) for i in accepted],
            "rejected": [
                reference(i) for i in proposed
                if (i["sentence"], i["component"]) not in accepted_keys
            ],
        }

    # ── Prompt builders ──────────────────────────────────────────────────────

    @staticmethod
    def _prompt_doc_knowledge_extract(comp_names, doc_lines) -> str:
        return f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

{DOC_KNOWLEDGE_EXTRACTION_RULES}

{ALIAS_EXCLUSION_RULES}

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": [{{"term": "short_form", "component": "FullComponent"}}],
  "synonyms":      [{{"term": "specific_alternative_name", "component": "FullComponent"}}]
}}
JSON only:"""

    @staticmethod
    def _prompt_doc_knowledge_judge(comp_names, mapping_list) -> str:
        return f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}



{DOC_KNOWLEDGE_JUDGE_RULES}

Return JSON:
{{"approved": ["term1", "term2"]}}
JSON only:"""

    @staticmethod
    def _prompt_extraction(comp_names, mappings, batch) -> str:
        return f"""Extract ALL references to components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings)}' if mappings else ''}

{ENTITY_EXTRACTION_RULES}

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence"}}]}}
JSON only:"""

    @staticmethod
    def _prompt_validation(comp_names, cases, focus, strict: bool = False) -> str:
        """Build a judging prompt. ``strict`` selects the coreference rubric.

        The rubric is asymmetric by design: the full-name gate is lenient (a
        stated name is a link unless a reject signal fires), the coreference
        gate is strict (the name is absent, so demand a referring expression
        and an architectural claim). The caller states which it wants — the
        inherited version inferred it from ``focus.startswith(...)``, so
        rewording the focus text silently swapped the rubric.
        """
        rules = LAYERED_COREF_RULES if strict else LAYERED_ENTITY_RULES
        return f"""Validate components in a document. {focus}

COMPONENTS: {', '.join(comp_names)}

{rules}

For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim), then decide approve true/false based on that claim.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>", "approve": true}}]}}
JSON only:"""

    @staticmethod
    def _prompt_coref(comp_names, cases) -> str:
        prompt = f"""Resolve references (pronouns and noun phrases that refer back) to components.

COMPONENTS: {', '.join(comp_names)}

For each TARGET sentence below, identify any pronoun or noun phrase that
refers back to a component listed above. If a target sentence has no such
reference to a listed component, return no resolution for it. Be conservative — only include resolutions you are CERTAIN about.

"""
        for i, case in enumerate(cases):
            prompt += f"--- Case {i+1}: S{case['sent'].number} ---\n"
            prompt += "CONTEXT:\n" + "\n".join(case["context"]) + "\n"
            prompt += f"TARGET: S{case['sent'].number} (marked with >>>)\n\n"

        prompt += f"""{COREF_RULES}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "reference": "the server", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

JSON only:"""
        return prompt

    # ── LLM call helper ──────────────────────────────────────────────────────

    def _ask(
        self,
        prompt: str,
        *,
        timeout: int = 120,
        label: str = "LLM call",
        phase: str | None = None,
        require: str | None = None,
        require_present: str | None = None,
    ) -> dict:
        """Query the LLM, parse JSON, retry once on empty/incomplete response.

        Success rule, in priority order:
          - require_present=KEY  → KEY must appear in the parsed dict (empty OK)
          - require=KEY          → data[KEY] must be truthy
          - neither              → any non-empty parsed dict succeeds
        """
        if phase is not None:
            self.llm.set_phase(phase)

        def _ok(d: dict | None) -> bool:
            if not d:
                return False
            if require_present is not None:
                return require_present in d
            if require is not None:
                return bool(d.get(require))
            return True

        data: dict = {}
        for attempt in range(self.ASK_ATTEMPTS):
            parsed = self.llm.extract_json(self.llm.query(prompt, timeout=timeout))
            # Each attempt replaces the last. Keeping a previous attempt's dict
            # when a later one fails to parse would return a payload this method
            # already rejected, and callers read it as if it had passed.
            data = parsed if parsed is not None else {}
            if _ok(data):
                return data
            if attempt < self.ASK_ATTEMPTS - 1:
                print(f"    {label}: empty response, retrying...")
        return data

    # ── Knowledge module ─────────────────────────────────────────────────────

    def _learn_document_knowledge(self, sentences, components):
        self.llm.set_phase("phase_25_doc_extract")
        comp_names = [c.name for c in components]
        doc_lines = [s.text for s in sentences]

        data1 = self._ask(
            self._prompt_doc_knowledge_extract(comp_names, doc_lines),
            timeout=300, label="Doc knowledge",
        )

        all_mappings: dict[str, str] = {}
        if data1:
            abbr_recs = data1.get("abbreviations", [])
            syn_recs = data1.get("synonyms", [])
            if isinstance(abbr_recs, dict):
                abbr_recs = [{"term": k, "component": v}
                             for k, v in abbr_recs.items()]
            if isinstance(syn_recs, dict):
                syn_recs = [{"term": k, "component": v}
                            for k, v in syn_recs.items()]
            for rec in abbr_recs + syn_recs:
                if not isinstance(rec, dict):
                    continue
                term = rec.get("term")
                full = rec.get("component")
                if term and full in comp_names:
                    all_mappings[term] = full

        if all_mappings:
            mapping_list = [f"'{k}' -> {v}" for k, v in all_mappings.items()]
            data2 = self._ask(
                self._prompt_doc_knowledge_judge(comp_names, mapping_list),
                timeout=120, label="Doc knowledge judge",
                phase="phase_25_doc_judge", require="approved",
            )
            approved = set(data2.get("approved", [])) if data2 else set(all_mappings)
        else:
            approved = set()

        knowledge = DocumentKnowledge()
        for term, comp in all_mappings.items():
            if term in approved:
                knowledge.aliases[term] = comp
                print(f"    Alias: {term} -> {comp}")
        return knowledge

    # ═════════════════════════════════════════════════════════════════════════
    # Linker 1 — FULL NAME: the sentence states a name of the component.
    # ═════════════════════════════════════════════════════════════════════════

    def _run_full_name_linker(self, sentences, components, name_to_id, linked,
                              sent_map):
        candidates_by_key = self._extract_named_mentions(
            sentences, components, name_to_id, sent_map
        )
        # The full-name linker's candidate set: what the LLM extractor proposed, held
        # to the linker's own contract, plus the two tight cells of the relation the
        # extractor may have missed. See ``SCANS`` and the module docstring's table.
        # No admission filter: the contract it enforced is stated in the extraction
        # prompt (see ``ENTITY_EXTRACTION_RULES``), so the extractor is held to the
        # linker's own contract rather than corrected after the fact.
        candidates = list(candidates_by_key.values())
        candidates = self._add_scan(candidates, sentences, components, "spelling")
        candidates = self._add_scan(candidates, sentences, components, "stated_name")
        candidates = self._unlinked(candidates, linked)
        bundles = {
            (c.sentence_number, c.component_id): self._build_evidence_bundle(c, sent_map)
            for c in candidates
        }
        approved, decisions = self._validate_with_evidence(
            candidates, bundles, components, sent_map,
            p1_tag="phase_25_full_name_p1",
            p2_tag="phase_25_full_name_p2",
            stage_label="full_name",
        )
        links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name,
                       source=self._full_name_source(c))
            for c in approved
        ]
        return links, {
            "candidates": self._link_view(
                [SadSamLink(c.sentence_number, c.component_id, c.component_name,
                            source="full_name_candidate") for c in candidates],
                sent_map,
            ),
            "accepted": self._link_view(links, sent_map),
            "judge_decisions": self._decision_view(decisions),
        }

    def _extract_named_mentions(self, sentences, components, name_to_id, sent_map) -> dict:
        """One extraction pass over the document, batched.

        An earlier revision sent this prompt twice and unioned the two samples
        as a self-consistency guard. Measured over five runs on all five
        projects, the second sample moved neither score beyond noise (TP -1.2,
        p=0.30; FP -1.2, p=0.42), so the pipeline states one sample and pays for
        one.
        """
        comp_names = get_comp_names(components)
        mappings = (
            [f"{term}={component}"
             for term, component in self.doc_knowledge.aliases.items()]
            if self.doc_knowledge else []
        )
        candidates = self._run_extraction_pass(
            sentences, comp_names, mappings, name_to_id, sent_map,
            phase_tag="phase_25_full_name_extract")
        print(f"    Extracted: {len(candidates)}")
        return candidates

    def _run_extraction_pass(self, sentences, comp_names, mappings,
                             name_to_id, sent_map, phase_tag=None):
        if phase_tag:
            self.llm.set_phase(phase_tag)
        batch_size = self.EXTRACTION_BATCH
        candidates: dict = {}
        for batch_num, batch in self._iter_batches(sentences, batch_size):
            if len(sentences) > batch_size:
                print(f"    batch {batch_num}: "
                      f"S{batch[0].number}-S{batch[-1].number} ({len(batch)} sents)")
            data = self._ask(
                self._prompt_extraction(comp_names, mappings, batch),
                timeout=240, label="batch", require="references",
            )
            if not data:
                continue
            for ref in data.get("references", []):
                cname = ref.get("component")
                snum = parse_snum(ref.get("sentence"))
                if snum is None or not cname or cname not in name_to_id:
                    continue
                sent = sent_map.get(snum)
                if not sent:
                    continue
                matched = ref.get("matched_text", "")
                if matched and matched.lower() not in sent.text.lower():
                    continue
                key = (snum, name_to_id[cname])
                if key not in candidates:
                    candidates[key] = CandidateLink(
                        snum, sent.text, cname, name_to_id[cname],
                        matched, source="full_name",
                    )
        return candidates

    def _add_scan(self, candidates, sentences, components, scan_name):
        """Add what ``SCANS[scan_name]`` finds and the caller does not already hold.

        Both of the full-name linker's lexical additions are this one call at two
        different points of the relation. Existing candidates win: the extractor's
        own ``matched_text`` is what the evidence bundle should carry when it found
        the pair first.
        """
        merged = {(c.sentence_number, c.component_id): c for c in candidates}
        for candidate in self._scan(sentences, components, SCANS[scan_name]):
            merged.setdefault(
                (candidate.sentence_number, candidate.component_id), candidate
            )
        return list(merged.values())

    def _scan(self, sentences, components, scan: SurfaceScan):
        """Every (sentence, component) pair the relation reaches at ``scan.form``.

        THE ONE CANDIDATE GENERATOR. ``s_linker64`` wrote this three times --
        ``_add_stated_name_net``, ``_spelling_variant_candidates`` and
        ``_name_word_candidates`` -- each with its own regex, its own ownership test
        and its own paragraph of defence. They differ in the point of the relation
        they scan and in two shared options, and `pilot/rule_audit.py --only A2`
        checks the three rebuilt sets against s_linker64's own on all five projects.

        Later spans of the same pair overwrite earlier ones, so the recorded
        ``matched_text`` is the last surface found in the sentence -- s_linker64's
        behaviour at both rebuilt sites.

        Nothing here admits a link. Every pair returned is a case for a judge.
        """
        candidates = {}
        for sentence in sentences:
            text = sentence.text
            for component in components:
                if scan.skip_when_named and self._states_a_name(text, component.name):
                    continue  # a whole name is stated: the full-name linker's pair
                for start, end in self._name_spans(text, component.name, scan.form):
                    if scan.skip_qualified and self._inside_qualified_identifier(
                        text, start, end
                    ):
                        continue
                    surface = text[start:end]
                    if scan.unique_owner and len(
                        self._owners(surface, components, scan.form)
                    ) != 1:
                        continue
                    if scan.skip_stricter and self._realizes(
                        surface, component.name, NameForm.ANY_CASE
                    ):
                        continue  # already the plain name; not a variant
                    candidates[(sentence.number, component.id)] = CandidateLink(
                        sentence.number, text, component.name, component.id,
                        surface, source=scan.source,
                        **({"mention_type": self._classify_mention_typed(
                            component.name, text)} if scan.label_mention else {}),
                    )
        return list(candidates.values())

    @classmethod
    def _realizes(cls, surface, name, form: NameForm) -> bool:
        """Is the surface, in whole, a writing of ``name`` at ``form``?

        ``_name_spans`` locates a name *inside* a text; this asks whether the text
        *is* the name. The distinction matters for ownership: `Image Provider`
        contains a writing of a component called `Provider`, but it is not one.
        """
        return (0, len(surface)) in cls._name_spans(surface, name, form)

    @classmethod
    def _owners(cls, surface, components, form: NameForm):
        """The components whose name this surface writes at ``form``.

        The uniqueness test both loose scans apply, stated once: a surface owned by
        more than one component is evidence for neither. `pilot/ablate_all.py` prices
        it at 2.4 false positives for free.
        """
        return [component for component in components
                if cls._realizes(surface, component.name, form)]

    @classmethod
    def _name_spans(cls, text, name, form: NameForm):
        """**The relation.** Spans of ``text`` that write ``name`` at ``form``.

        The whole deterministic layer of this workflow is this function and the four
        values of ``NameForm``. It reads the runtime catalog and ``INFLECTIONS``, and
        nothing else; no benchmark vocabulary reaches it (GATE-06).

        The four branches were four methods in ``s_linker64``, verified identical to
        these over every (name, sentence) pair of all five projects
        (`pilot/rule_audit.py --only A2`, `pilot/test_s65_one_relation.py`).
        """
        if form is NameForm.AS_SPELLED:
            return [(m.start(), m.end()) for m in re.finditer(
                rf"(?<!\w){re.escape(name)}(?!\w)", text)]

        if form is NameForm.ANY_CASE:
            return [(m.start(), m.end()) for m in re.finditer(
                rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE)]

        if form is NameForm.ANY_SPELLING:
            target = cls._name_signature(name)
            if not target:
                return []
            # A span of k words yields at least k signature tokens, so a span longer
            # than the target's token count can never match it.
            words = list(re.finditer(r"[A-Za-z0-9]+", text))
            spans = []
            for i, first in enumerate(words):
                for j in range(i, min(len(words), i + len(target))):
                    if j > i and not re.fullmatch(
                        r"[\s_-]+", text[words[j - 1].end():words[j].start()]
                    ):
                        break
                    start, end = first.start(), words[j].end()
                    if cls._name_signature(text[start:end]) == target:
                        spans.append((start, end))
            return spans

        if form is NameForm.ANY_WORD:
            # Extent, not fidelity: one word of the name is enough, under an English
            # inflectional ending. An unbounded prefix would make `web` own `webrtc`
            # and `webcams`; naming the endings states the intent instead of
            # approximating it (s_linker62, +2.0 gold / +0.0 spurious candidates).
            words = [w.casefold() for w in re.findall(WORD_PATTERN, name)]
            return [
                (m.start(), m.end())
                for m in re.finditer(WORD_PATTERN, text)
                if any(m.group(0).casefold().startswith(word)
                       and m.group(0).casefold()[len(word):] in INFLECTIONS
                       for word in words)
            ]

        raise ValueError(f"unknown name form: {form!r}")

    @staticmethod
    def _full_name_source(candidate):
        if candidate.source == "full_name_variant":
            return "full_name_variant"
        return "full_name"

    @staticmethod
    def _name_signature(expression):
        """Normalize an expression to its sequence of words, splitting CamelCase.

        A spaced form, a hyphenated form, and a run-together form of the same
        words share a signature ("X Y", "x-y", and "XY" all give ("x", "y")),
        which is what makes a spelling variant recognizable.

        Compound splitting is *not* a relaxation of case folding: it reaches
        `Image Provider` for a name spelled `ImageProvider`, and it misses
        `redis pubsub` for a name spelled `Redis PubSub`, which case folding reaches.
        Six pairs over five projects; the linker takes the union of the forms it
        scans, so neither direction is lost.
        """
        normalized = unicodedata.normalize("NFKC", expression)
        normalized = normalized.replace("-", " ").replace("_", " ")
        return tuple(
            token.casefold()
            for token in re.findall(
                r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+", normalized
            )
        )

    @staticmethod
    def _in_dotted_path(text, start, end) -> bool:
        """True when text[start:end] is glued to a dot on either side, as in x.y.

        The single definition of "inside a qualified name". Two tests used to
        carry their own copy of it and the copies disagreed -- one asked whether
        the character after the dot ``isalnum()``, the other ``isalpha()``, and
        one required an alphanumeric before the dot while the other did not.
        Neither divergence ever changed a result (0 differences over 3697
        (name, sentence) pairs and 5388 word spans on all five projects), so the
        stricter reading is the one kept.
        """
        before = (start > 1 and text[start - 1] == "."
                  and text[start - 2].isalnum())
        after = (end + 1 < len(text) and text[end] == "."
                 and text[end + 1].isalnum())
        return before or after

    @classmethod
    def _inside_qualified_identifier(cls, text, start, end):
        """True when the span sits inside a dotted path or a larger word."""
        before = text[start - 1] if start else ""
        after = text[end] if end < len(text) else ""
        joined = (before in "-_" or (before and before.isalnum())
                  or after in "-_" or (after and after.isalnum()))
        return cls._in_dotted_path(text, start, end) or joined

    # ── Evidence bundles and the two-pass judge ──────────────────────────────

    def _classify_mention_typed(self, comp_name: str, text: str) -> MentionType:
        """Label how the name appears, using the one matching test.

        The case distinction is a comparison of the matched surface against the
        name, not a second predicate carrying its own case rules:
        ``_find_exact_form`` already returns what it matched. An earlier revision
        asked a strict, case-sensitive predicate first and a lowercase regex
        second, which is why the module used to hold two name tests.

        Measured: indistinguishable on this stage (TP +/-0.0, p=1.00; FP -0.4,
        p=0.76 over five runs per side) and inside the end-to-end variance band
        of the form it replaces (F1 96.47 over three runs against 96.42 +/- 0.42
        over six runs of the previous form; F2 95.20 against 95.38 +/- 0.58).
        """
        matched = self._find_exact_form(text, comp_name)
        if matched:
            # No qualified-path value. It was the only consumer of
            # ``_all_occurrences_in_qualified_path`` -- a second, case-blind reading of
            # the same boundary question the scans ask through
            # ``_inside_qualified_identifier`` -- and it fired on 7.7 of 182.5 cases a
            # run. Replaying the judge on the recorded candidates with and without it,
            # five samples a side, is TP +/-0.0 (p = 1.00), FP -0.2 (p = 1.00),
            # composition p = 1.00 (`pilot/bind_pilots.py --pilot cutcodetoken`): the
            # judge sees the dotted path in the sentence it is shown and does not need
            # to be told. The two stated-name values stay separate -- merging *those*
            # is what `s_linker44` measured at macro F1 -0.9 (p = 0.05) over six runs.
            return (MentionType.PROPER_STANDALONE if matched == comp_name
                    else MentionType.LOWERCASE_PROSE)
        for alias in self._names_by_component().get(comp_name, ()):
            if self._find_exact_form(text, alias):
                return MentionType.VIA_ALIAS
        return MentionType.INDIRECT

    def _build_evidence_bundle(self, candidate, sent_map):
        comp_name = candidate.component_name
        snum = candidate.sentence_number
        mention_type = self._classify_mention_typed(
            comp_name, candidate.sentence_text
        ).value
        prev_sent = sent_map.get(snum - 1)
        anchors = []
        for s in sorted(sent_map.values(), key=lambda x: x.number):
            if s.number == snum:
                continue
            if self._find_exact_form(s.text, comp_name):
                anchors.append(f"S{s.number}: {s.text}")
                if len(anchors) >= self.ANCHOR_LIMIT:
                    break
        return EvidenceBundle(
            source=candidate.source,
            matched_span=candidate.matched_text or comp_name,
            mention_type=mention_type,
            preceding_text=prev_sent.text if prev_sent else "",
            anchor_sentences=anchors,
        )

    def _format_evidence(self, bundle: EvidenceBundle) -> str:
        lines = [
            f"  Evidence: source={bundle.source}, span=\"{bundle.matched_span}\", "
            f"mention={bundle.mention_type}",
        ]
        if bundle.preceding_text:
            lines.append(f"  [prev: \"{bundle.preceding_text}\"]")
        if bundle.anchor_sentences:
            lines.append("  Anchors (confirmed refs):")
            for a in bundle.anchor_sentences:
                lines.append(f"    {a}")
        return "\n".join(lines)

    def _validate_with_evidence(self, candidates, bundles, components, sent_map,
                                p1_tag, p2_tag, stage_label):
        """Two judging passes; a link needs both."""
        if not candidates:
            return [], {}
        comp_names = get_comp_names(components)
        decisions: dict = {}
        approved = []
        for _, batch in self._iter_batches(candidates, self.JUDGE_BATCH):
            cases = []
            for i, c in enumerate(batch):
                p = self._prev_prefix(c.sentence_number, sent_map)
                bundle = bundles.get((c.sentence_number, c.component_id))
                evidence_block = self._format_evidence(bundle) if bundle else ""
                cases.append((
                    f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n'
                    f'  {p}"{c.sentence_text}"\n'
                    f'{evidence_block}',
                    c,
                ))
            case_strings = [ct for ct, _ in cases]
            r1 = self._run_validation_pass(comp_names, case_strings, P1_FOCUS, p1_tag)
            r2 = self._run_validation_pass(comp_names, case_strings, P2_FOCUS, p2_tag)
            for i, (_case_text, c) in enumerate(cases):
                p1 = r1.get(i, False)
                p2 = r2.get(i, False)
                ok = p1 and p2
                decisions[(c.sentence_number, c.component_id)] = {
                    "approved": ok, "p1": p1, "p2": p2,
                    "path": f"{stage_label}_twopass" if ok
                            else f"{stage_label}_twopass_reject",
                    "stage": f"{stage_label}_twopass",
                }
                if ok:
                    approved.append(c)
        return approved, decisions

    def _run_validation_pass(self, comp_names, cases, focus, phase_tag=None,
                             strict=False):
        if phase_tag:
            self.llm.set_phase(phase_tag)
        data = self._ask(
            self._prompt_validation(comp_names, cases, focus, strict=strict),
            timeout=120, label="Validation pass", require="validations",
        )
        results: dict[int, bool] = {}
        if data:
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    val = v.get("approve", False)
                    results[idx] = (
                        val is True
                        or (isinstance(val, str) and val.lower() == "true")
                    )
        return results

    # ═════════════════════════════════════════════════════════════════════════
    # Linker 2 — PARTIAL NAME: the sentence carries one word of a name.
    # ═════════════════════════════════════════════════════════════════════════

    def _run_partial_name_linker(self, sentences, components, linked, sent_map):
        candidates = self._unlinked(
            self._scan(sentences, components, SCANS["name_word"]), linked)
        approved, decisions = self._judge_partial_names(candidates, sentences)
        links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name,
                       source="partial_name")
            for c in approved
        ]
        return links, {
            "proposed": self._link_view(
                [SadSamLink(c.sentence_number, c.component_id, c.component_name,
                            source="partial_name_candidate") for c in candidates],
                sent_map,
            ),
            "accepted": self._link_view(links, sent_map),
            "judge_decisions": self._decision_view(decisions),
        }

    def _judge_partial_names(self, candidates, sentences):
        """One step: does the expression denote a software participant?

        s_linker25 followed this with a grounded identity review that showed the
        model the target. Measured over six of its runs, that review rejected 8.0
        candidates per run of which 5.5 were gold -- it traded 5.5 true positives
        for 2.5 false positives -- so the linker judges once, like the coreference
        linker, and the target stays withheld throughout.
        """
        participants, decisions = self._classify_denotations(candidates, sentences)
        for candidate in participants:
            key = (candidate.sentence_number, candidate.component_id)
            decisions[key] = {**decisions.get(key, {}), "approved": True,
                              "requested_keep": True, "path": "denotation"}
        return participants, decisions

    def _classify_denotations(self, candidates, sentences):
        """Step 1: does the expression itself denote a software participant?

        The target component is deliberately withheld. Shown the target, the
        model confirms identity rather than testing it.
        """
        sent_map = {s.number: s for s in sentences}
        decisions = {}
        for _, batch in self._iter_batches(candidates, self.JUDGE_BATCH):
            evidence_ids = {
                sentence.number
                for candidate in batch
                for sentence in self._window(candidate.sentence_number, sentences)
            }
            sentence_table = [
                {"sentence": n, "text": sent_map[n].text}
                for n in sorted(evidence_ids)
            ]
            cases = [
                {"case": n, "source": c.sentence_number, "expression": c.matched_text}
                for n, c in enumerate(batch, 1)
            ]
            prompt = f"""Classify what each expression itself denotes in its
local context: participant for a software participant, or associated for
something merely associated with software.

SENTENCES
{json.dumps(sentence_table)}

CASES
{json.dumps(cases)}

Claim must be a contiguous exact substring of the source sentence.

JSON only:
{{"judgments":[{{"case":1,"denotation":"participant",
"claim":"exact source quote"}}]}}
"""
            data = self._ask(
                prompt, phase="phase_25_partial_denotation",
                require_present="judgments", label="Denotation", timeout=240,
            )
            for item in data.get("judgments", []):
                case_value = str(item.get("case", ""))
                if not case_value.isdigit():
                    continue
                number = int(case_value)
                if not 1 <= number <= len(batch):
                    continue
                candidate = batch[number - 1]
                claim = str(item.get("claim", "")).strip().strip("\"'“”‘’")
                denotation = str(item.get("denotation", "")).strip()
                valid = (
                    denotation in {"participant", "associated"}
                    and bool(claim)
                    and claim.casefold() in candidate.sentence_text.casefold()
                )
                decisions[(candidate.sentence_number, candidate.component_id)] = {
                    "approved": False,
                    "requested_keep": False,
                    "evidence_valid": valid,
                    "claim": claim,
                    "denotation": denotation,
                    "alternative": "not reviewed",
                    "path": "denotation",
                    "stage": "partial_name",
                }
        participants = [
            c for c in candidates
            if decisions.get((c.sentence_number, c.component_id), {}).get(
                "denotation") == "participant"
            and decisions[(c.sentence_number, c.component_id)]["evidence_valid"]
        ]
        return participants, decisions

    def _run_coreference_linker(self, sentences, components, name_to_id, linked,
                                sent_map):
        resolved, metadata = self._resolve_references(
            sentences, components, name_to_id, sent_map
        )
        raw = self._unlinked(resolved, linked)
        approved, decisions = self._validate_coref_links(raw, sent_map, components)
        return approved, {
            "candidates": self._link_view(raw, sent_map),
            "accepted": self._link_view(approved, sent_map),
            "metadata": [
                {"sentence": sentence, "component_id": component, **value}
                for (sentence, component), value in metadata.items()
            ],
            "judge_decisions": self._decision_view(decisions),
        }

    def _resolve_references(self, sentences, components, name_to_id, sent_map):
        """Every sentence goes to the LLM in context; no pronoun regex.

        A resolution survives only when its antecedent sentence itself states a
        name of the component — the structural antecedent constraint.

        Both sentence numbers a resolution reports are checked against the
        document: the target sentence as well as the antecedent. A number the
        model invents cannot name a real sentence, so admitting one could only
        ever add a link the gold standard has no counterpart for.
        """
        comp_names = get_comp_names(components)
        all_coref = []
        coref_metadata: dict = {}
        self.llm.set_phase("phase_25_coreference")

        for batch_num, batch in self._iter_batches(sentences, self.COREFERENCE_BATCH):
            cases = []
            for sent in batch:
                context = [
                    f'{">>>" if s.number == sent.number else "   "} '
                    f"S{s.number}: {s.text}"
                    for s in self._window(sent.number, sentences)
                ]
                cases.append({"sent": sent, "context": context})

            data = self._ask(
                self._prompt_coref(comp_names, cases), timeout=600,
                label=f"Coref batch {batch_num}", require_present="resolutions",
            )
            if not data:
                continue

            for res in data.get("resolutions", []):
                comp = res.get("component")
                snum = parse_snum(res.get("sentence"))
                if snum is None or snum not in sent_map:
                    continue
                if not comp or comp not in name_to_id:
                    continue
                ant_snum = parse_snum(res.get("antecedent_sentence"))
                if ant_snum is None:
                    print(f"    Coref skip (no antecedent): S{snum} -> {comp}")
                    continue
                ant_sent = sent_map.get(ant_snum)
                if not ant_sent:
                    continue
                # The structural antecedent gate: a resolution is accepted only when
                # the antecedent sentence itself states a name in N(c). The same
                # predicate the full-name linker admits on and the partial-name scan
                # defers on -- one condition, three stages. Priced at 12 FP
                # (`pilot/gate_pilots.py`).
                if not self._states_a_name(ant_sent.text, comp):
                    continue
                cid = name_to_id[comp]
                all_coref.append(SadSamLink(snum, cid, comp, source="coreference"))
                coref_metadata[(snum, cid)] = {
                    "reference": res.get("reference", ""),
                    "antecedent_sentence": ant_snum,
                    "antecedent_text": res.get("antecedent_text", ""),
                    "raw_resolution": res,
                }
        return all_coref, coref_metadata

    def _validate_coref_links(self, coref_links, sent_map, components):
        """Single judging pass — asymmetric to the full-name linker's two passes,
        because resolution asks a narrower question."""
        if not coref_links:
            return [], {}
        comp_names = get_comp_names(components)
        validated = []
        decisions: dict = {}
        self.llm.set_phase("phase_25_coreference_judge")
        for _, batch in self._iter_batches(coref_links, self.JUDGE_BATCH):
            cases = []
            for i, lk in enumerate(batch):
                # _resolve_references admits a resolution only for a sentence
                # the document has, so every link reaching the judge has one.
                sent = sent_map[lk.sentence_number]
                p = self._prev_prefix(lk.sentence_number, sent_map)
                cases.append((
                    lk,
                    f'Case {i+1}: pronoun/role-ref -> {lk.component_name}\n'
                    f'  {p}"{sent.text}"',
                ))
            results = self._run_validation_pass(
                comp_names, [c for _, c in cases], COREF_VALIDATION_FOCUS,
                phase_tag="phase_25_coreference_judge", strict=True,
            )
            for idx, (lk, _case) in enumerate(cases):
                approved = bool(results.get(idx, False))
                decisions[(lk.sentence_number, lk.component_id)] = {
                    "approved": approved,
                    "path": "coref_validated" if approved else "coref_rejected",
                }
                if approved:
                    validated.append(lk)
                else:
                    print(f"    Coref reject: S{lk.sentence_number} -> {lk.component_name}")
        return validated, decisions

    # ── Logging and checkpointing ────────────────────────────────────────────

    def _backend_tag(self) -> str:
        inner = getattr(self.llm, "_inner", self.llm)
        backend = getattr(inner, "backend", None)
        if backend is None:
            return "unknown"
        return getattr(backend, "value", str(backend))

    def _checkpoint_dir(self, text_path):
        cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
        ds = os.path.splitext(os.path.basename(text_path))[0]
        d = os.path.join(cache_dir, self._VARIANT_NAME, self._backend_tag(), ds)
        os.makedirs(d, exist_ok=True)
        return d

    def _save_phase(self, text_path, phase_name, state):
        path = os.path.join(self._checkpoint_dir(text_path), f"{phase_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"  Checkpoint: {phase_name} saved")

    def _log(self, phase, input_summary, output_summary, links=None):
        entry = {"phase": phase, "ts": time.time(),
                 "in": input_summary, "out": output_summary}
        if links is not None:
            entry["links"] = [
                {"s": l.sentence_number, "c": l.component_name, "src": l.source}
                for l in links
            ]
        self._phase_log.append(entry)

    def _save_log(self, text_path):
        log_dir = os.environ.get("LLM_LOG_DIR", "./results/llm_logs")
        os.makedirs(log_dir, exist_ok=True)
        ds = os.path.splitext(os.path.basename(text_path))[0]
        ts = time.strftime("%Y%m%d_%H%M%S")
        backend = self._backend_tag()
        summary_path = os.path.join(
            log_dir, f"{self._VARIANT_NAME}_{backend}_{ds}_{ts}.json")
        with open(summary_path, "w") as f:
            json.dump(self._phase_log, f, indent=2, default=str)
        print(f"  Phase log saved: {summary_path}")
        calls_path = os.path.join(
            log_dir, f"{self._VARIANT_NAME}_{backend}_{ds}_{ts}_calls.json")
        trunc_env = os.environ.get("CALLS_TRUNCATE_CHARS", "").strip()
        trunc = int(trunc_env) if trunc_env.isdigit() else 0
        if trunc > 0:
            calls = []
            for c in self._llm_calls:
                cc = dict(c)
                if cc.get("prompt") and len(cc["prompt"]) > trunc:
                    cc["prompt"] = cc["prompt"][:trunc] + "... [truncated]"
                if cc.get("response_text") and len(cc["response_text"]) > trunc:
                    cc["response_text"] = cc["response_text"][:trunc] + "... [truncated]"
                calls.append(cc)
        else:
            calls = self._llm_calls
        with open(calls_path, "w") as f:
            json.dump(calls, f, indent=2, default=str)
        print(f"  LLM call trace saved: {calls_path} ({len(self._llm_calls)} calls)")

    def _compute_phase_metrics(self) -> dict:
        metrics: dict[str, dict] = {}
        for call in self._llm_calls:
            ph = call.get("phase", "unknown")
            m = metrics.setdefault(
                ph, {"calls": 0, "elapsed_s": 0.0, "tokens": 0, "errors": 0})
            m["calls"] += 1
            m["elapsed_s"] = round(m["elapsed_s"] + call.get("elapsed_s", 0.0), 3)
            if call.get("success") is False:
                m["errors"] += 1
            usage = call.get("token_usage")
            if usage:
                m["tokens"] += usage.get("total_tokens", 0) or 0
        return metrics
