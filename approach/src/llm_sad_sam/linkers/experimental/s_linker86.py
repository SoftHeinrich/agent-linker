"""s_linker86 -- s_linker85 with the full-name judge's focus line removed.

One change, and it is a deletion of a restatement rather than of a rule.
`VALIDATION_FOCUS` asked the lenient judge to "check architectural participation and
referential specificity"; the rubric printed under it already asks both -- 
`LAYERED_ENTITY_RULES` makes architectural participation the approve-condition and
`STRICTER_CLAUSE` is about nothing else than a name identifying this element rather
than serving as ordinary vocabulary. 243 B leave every one of the ~18.5 judging calls
a five-project run makes.

Measured as a stage arm on both models before it was written, three runs a side,
every arm judging the SAME extraction pass so the candidate set is constant
(`pilot/typed_prompt_pilots.py --group fullname3`, statistics in
`pilot/typed_round_stats.py`, report in `../results/typed_round/README.md`):

    model   stage gold        composed TP      macro F1        macro F2
    terra   150.7 -> 151.7    182.0 -> 183.0   94.2 -> 93.6 (p = 0.40)   -0.0 (p = 0.80)
    luna    148.0 -> 149.7    174.7 -> 175.7   88.8 -> 88.9 (p = 0.90)   +0.3 (p = 0.60)

Recall moves +1.0 on both models and nothing reaches the n=3 p floor of 0.10.

WHAT THE SAME ROUND REFUSED, and why this variant carries none of it. The round asked
whether each judging rubric could be a closed set of verdicts instead of prose -- the
module already has one typed judge, the denotation step, which answers `participant`
or `associated`. Every typed arm was measured on both models and every one failed:

    typed full-name rubric        terra gold 151.3 -> 134.7 (p = 0.10)
      + approving NO_CLAIM        terra gold -5.0, macro F2 -1.7
      + the default restated      terra gold -8.3 (p = 0.10)
    typed coreference rubric      terra macro F1 -1.2 (p = 0.10)
      + the default restated      terra neutral, luna FP +34.0, macro F1 -3.8
    typed alias rubric            terra table 27.0 -> 31.3 terms, macro F1 -1.4

The mechanism is one sentence: **typing a rubric deletes its default, and the default
is what each judge's asymmetry was carrying.** The lenient gate lost recall (naming
three reject types with no "approve by default" invites the judge to reach for one);
the strict gate lost strictness (naming three reject types instead of "when uncertain,
reject" makes a merely-plausible resolution reachable). Restating the default inside
the typed rubric repairs the strict judge on terra and not on luna, and does not
repair the lenient judge on either. A typed rubric is also not smaller: 66 chars
larger per call at the coreference judge, 304 at the alias judge.

Two further single points were priced and left alone. The extraction rule's
morphology clause ("count a name written with different spacing, hyphenation or
compound joining as that name") is the only instruction admitting a candidate whose
sentence writes no name at all; removing it is terra gold -3.3, luna gold -5.0 and
macro F2 -1.9, so it stays -- and the 9.7 spurious candidates per run a deterministic
audit had attributed to it on luna do not go away when it does. The inert sentence of
`LAYERED_ENTITY_RULES` ("A mention that says nothing further about the component still
counts as a valid link") is genuinely inert -- claim="none" was rejected in 105 of 105
recorded cases -- and deleting it is neutral alone on both models but negative
combined with the focus deletion, so it stays too: the round removes one clause, not
two.


Two independent changes meet here and neither touches the other:

    the judge         the coreference judge is shown the resolution it is judging, its
                      rubric carries the actor/artifact distinction, and it states the
                      ground for rejecting a case before deciding it. This is the round's
                      answer to a question s82 could not answer: how to make the laxer of
                      two models better without making the stricter one worse.
    the morphology   `INFLECTIONS`, the module's last authored word list -- nine English
                      endings stripped off the sentence token -- is replaced by WordNet's
                      lemmatizer over both sides of the comparison. Priced span by span
                      before it was written: 3697 (name, sentence) pairs, the spans differ
                      on 2, partial-name candidates 109 -> 110 with gold 28 -> 28
                      (`pilot/lemma_swap_pilot.py --only E1`). The one gain is a name
                      whose own word is inflected -- `Recording Service` reaching
                      `recorded` -- which stripping endings off the token cannot do.
                      Two weaker rules were measured and refused: lemmatizing the
                      sentence token alone loses exactly that pair, and a POS-
                      disambiguated lemmatizer loses 7 candidates including 1 gold
                      (`--only E2`). The cost is one dependency, `nltk` plus the
                      `wordnet` corpus.

The two cannot interact. The morphology is read at one place, `_name_spans` at
`ANY_WORD`, which only the partial-name scan uses; the judge changes are in the
coreference stage, which scans nothing. So this module carries **no authored word list at
all** and its judging surface is the one measured below.

WHAT THE JUDGE CHANGES SCORE. Three runs a side on recorded inputs, each arm's kept
coreference pairs unioned with the *same* run's recorded full-name and partial-name
links, so upstream sampling is held fixed and the only variance is the judge itself. This
is an exact pipeline score, not a projection: replaying s82's own kept pairs through it
reproduces its recorded end-to-end numbers to the decimal.

    arm                              terra F1 / F2      luna F1 / F2
    s_linker82 (control)             92.25 / 94.08      85.38 / 91.16
    + judge shown the resolution
      + actor/artifact clause        92.44 / 94.31      86.36 / 91.59
    + ground-for-rejecting first     93.69 / 94.51      89.20 / 92.43
      (TP/FP: terra 182.0 / 22.3, luna 181.7 / 51.7, against 183.0 / 30.0 and
       184.0 / 77.0)

**Luna gains F1 +3.82 / F2 +1.27 and terra F1 +1.44 / F2 +0.43** -- the laxer model gains
most and the stricter one does not regress, which is what the round was opened to find.

CONFIRMED END TO END, three paired runs per model, s85 against s82 in the same
invocations (`../results/s85_e2e_{terra,luna}_r{1,2,3}_20260820`):

    model   arm    macro F1          macro F2          TP      FP     calls
    terra   s85    93.68 (sd 0.57)   94.24 (sd 0.52)   181.0   21.0      82
            s82    91.13 (sd 0.32)   93.24 (sd 0.76)   181.7   34.3      82
    luna    s85    89.48 (sd 0.63)   91.28 (sd 0.91)   174.7   41.3      84
            s82    83.83 (sd 0.24)   89.91 (sd 1.02)   182.0   80.0      84

Macro F1 +2.55 on terra (3 of 3 runs) and +5.65 on luna (3 of 3); macro F2 +0.99
(2 of 3) and +1.37 (3 of 3). Both p = 0.25, the exact-permutation floor at n = 3,
with unanimous sign. **Luna's false positives are halved** -- the model that was
admitting twice terra's junk now admits 41.3 against terra's 21.0 -- and the
laxer model gains more than twice what the stricter one does, which is what the
round was opened to find. The recall cost is real and stated: TP -7.3 on luna and
-0.7 on terra, which F2 pays for and F1 rewards.

THE ACCOUNTING THAT MADE IT VISIBLE, AND THE ERROR IT CORRECTS. Seven judge settings were
built and refused before this one, all of them scored on the coreference stage's own gold
count -- and that count is the wrong measure. **57.6% of terra's approved coreference gold
links and 73.9% of luna's are pairs an earlier linker already produced**, which `_union`
merges away, while 97% and 89% of the *spurious* ones are new. So the stage's recall is
mostly free-riding and its precision is not: luna's coreference contributes 19.3 new gold
against 33.7 new spurious per run, 44% of that model's entire false-positive mass. Every
strict arm had been charged for losing duplicates that cost the pipeline nothing. The
ground-for-rejecting contract, refused on stage counts as a recall collapse, is on this
accounting the round's largest gain.

WHAT REMAINS REFUSED, re-derived on the same exact scoring or measured free off recorded
runs: naming which catalog component a partial-name phrase refers to (terra 7.0 gold per
run against 18.3); requiring that judge to quote the whole phrase; the judges'
self-reported certainty (luna's spurious approvals are confidently wrong -- 17.7 gold ->
11.3 to save 5.3 spurious); verifying the judge's own claim quote against the sentence
(0.0 ungrounded spurious approvals on either model, so s48's "verifying it is worth
nothing" holds for the laxer model too); and enforcing the coreference stage's own
contract in code, dropping a resolution whose referring expression writes a name of the
component (terra -17.7 gold against -0.0 spurious, luna -40.3 against -5.0).

Everything else is s_linker83 and, under it, s_linker82. Their docstrings follow.

s_linker83 -- the coreference judge is shown the resolution it is judging, and six
other judge settings are priced and refused.

WHY THE ROUND EXISTS. s_linker82 beat s_linker81 on both gpt-5.6-terra and gpt-5.6-luna,
by very different amounts (macro F2 +1.76 against +3.80). The stage tables say why: the
two models propose the same links and *judge* them differently. Gold proposals are within
a link of each other at every proposer and the final recall is the same (TP 183.0 terra,
184.0 luna); the whole gap is precision, FP 30.0 against 77.0, spread over all three
judges. Luna keeps 47% of the spurious full-name candidates it sees against terra's
37.5%, 38.6% of the partial-name ones against 17.8%, 28.8% of the coreference ones
against 19.9%.

THE CHANGE. The coreference judge saw a sentence and a component name, so it could not
check an antecedent it was never shown, and it rejected half the gold resolutions put to
it. The resolver had already committed to the referring expression and to the sentence it
read as the antecedent; both now appear in the case. Nothing else changes: no rule, no
gate, no call, no stage.

    stage pilot, three runs a side on recorded inputs, kept links per five-project run

        terra   45.3 gold / 11.3 spurious   against s82's 41.7 / 11.7
        luna    76.0 gold / 50.7 spurious   against s82's 74.0 / 37.7

    Projected onto the recorded pipeline totals: terra F2 93.66 against 92.15 and F1
    90.76 against 89.71; luna F2 88.07 against 88.38, inside that model's run-to-run
    spread (sd 0.67). **No end-to-end run is owed**: coreference is the last linker, so
    nothing downstream can be starved of candidates and the composition check is
    structurally vacuous (`CLAUDE.md`, measurement policy, step 3).

WHAT THIS ROUND DID NOT ACHIEVE, AND THE FINDING THAT REPLACES IT. The round was opened
to make the *lax* model better. It does not. Six further judge settings were built and
measured, three runs a side on both models, and every one of them is refused; the
projected pipeline F2 delta against s82 is in brackets, terra first.

    partial name, name which catalog component the phrase refers to      [-4.66, -0.90]
    partial name, quote the whole phrase and answer about it             [+0.07, -0.25]
    partial name, state the ground for answering associated first        [-4.67, -3.40]
    partial name, that ground with the default restored                  [-3.47, -1.16]
    coreference, judge quotes reference and antecedent itself            [-3.09, -15.08]
    coreference, state the ground for rejecting first                    [-7.62, -20.89]
    coreference, that ground with the default restored                   [-6.40, -15.63]
    coreference, drop links whose antecedent does not name the component [-0.81, -1.59]
        (deterministic, replayed free off both models' recorded runs: terra -3.3 gold /
         -6.0 spurious per run, luna -7.3 / -15.7. F1-positive on both models -- +0.43
         and +0.92 -- and F2-negative on both, so it is refused under an F2-led budget
         and named here as what an F1-led one would take.)

**The two models do not differ in what they understand by the question; they differ in
which way they lean when the case is close.** Every wording above moves both models along
one precision/recall dial, in whichever direction it leans, and none of them changes the
order in which cases are approved. Asking for the ground against a link before the verdict
-- the mirror of this module's oldest measured rule, claim-before-verdict, worth 35.2 TP
-- is the sharpest instance: it takes coreference precision to 1.0 spurious links per run
on both models and takes recall with it (luna 74.0 gold to 18.0). So a judge's calibration
is not addressable by sharpening the judge's question, and s82's wordings stand at the F2
optimum of the eight settings measured. What the surviving change buys is not calibration
but evidence: the one judge in the module that was asked to check a claim it could not see.

Everything else is s_linker82, whose docstring follows.

s_linker82 -- s_linker81 with the audit fixes: one judging pass, a deduplicated
coreference prompt, an alias judge that cannot fail open, and no dead code.

WHAT IT DOES. Three linkers run in a fixed order over one document and one component
catalog. Each proposes (sentence, component) pairs and an LLM judge decides them:

    full_name     an LLM extractor reads the document in batches of 50 and reports the
                  sentences that write a component's name or a discovered alias; a judge
                  sees each proposal with its evidence line and decides.
    partial_name  a scan proposes every sentence carrying one word of a name (under an
                  English inflectional ending) that does not already write a whole name;
                  a target-blind judge asks only whether the expression denotes a
                  software participant.
    coreference   the LLM resolves referring expressions to components inside a +/-5
                  sentence window; a strict judge re-checks each resolution.

Aliases are learned first (extract, then judge) and feed the full-name extractor. Links
are unioned; earlier linkers win ties.

THE DESIGN LAW. Facts stay in code, weighings go in the prompt. The deterministic layer
proposes spans and computes only the facts a judge cannot see for itself -- the alias
table, and whether every occurrence of a name sits inside a dotted identifier. Every
admission, suppression and tie-break is an LLM verdict.

WHAT CHANGED FROM s_linker81

    one judging pass    s81 sent the same full-name prompt twice -- P1 asked about
                        architectural participation, P2 about referential specificity --
                        and AND-ed the verdicts. Replaying the three recorded s81 runs
                        (`results/elegance_e2e_s81_r*_20260819`), the passes disagree on
                        4.0 of ~196 candidates per five-project run: dropping P2 is +0.7
                        gold / +1.7 spurious, dropping P1 is +1.3 gold / +0.7 spurious,
                        both inside the run-to-run band (spurious approvals alone move
                        4 -> 11 across identical runs). `pilot/simplify_verify_pilot.py`
                        reads the same way on its own population (S1 "p1 only" F1 +0.024,
                        F2 +0.056 pooled). The rubric already carries both questions --
                        `LAYERED_ENTITY_RULES` states participation, `STRICTER_CLAUSE`
                        states specificity -- so the module asks once, with both in the
                        focus line, at half the judging calls. The replay bounds the
                        change but does not price it: only an e2e run does.
    coreference prompt  s81 pasted a +/-5 window per case, so a batch of 10 targets
                        rendered ~110 sentence copies. The batch now carries one
                        deduplicated sentence table, and each case carries its own
                        target sentence with its context named by number. The table
                        alone -- the denotation step's shape, cases that are bare
                        numbers -- was measured and rejected: it held gold resolutions
                        (79.3 per five-project run against s81's 80.3) but raised
                        spurious ones to 60.0 from 45.7 and cost the stage 9.6 approved
                        gold links. Restoring the target text is 96.7 gold and 51.0
                        spurious per run, the best of the three arms on both axes.
    alias judge         s81 tested the verdict for truthiness, which gave three
                        behaviours for one event. A reply that did not parse approved
                        *every* proposal; a reply that parsed without an `approved` key
                        approved none; an empty approval list was honoured but burned a
                        retry call first. The judge now returns term-with-component
                        pairs, an empty list is an empty result with no retry, and both
                        shapes of "the judge did not answer" take the same documented
                        lenient default, logged when it fires. A term two components both
                        hold approval for is dropped -- s81 kept whichever the extractor
                        recorded last.
    extraction prompt   s81's `ENTITY_EXTRACTION_RULES` both excluded "a name used as
                        ordinary English" and demanded "every sentence ... however
                        incidental". The use/mention call belongs to the judge that holds
                        `STRICTER_CLAUSE`, so the extractor is asked for named mentions
                        only.
    claims recorded     the judging prompt asks for the exact words stating the claim
                        before the verdict; s81 parsed only the verdict, so the trace
                        could not audit it. Both are recorded now.
    dead code gone      `SurfaceScan`/`SCANS` (one row, two options never set),
                        `NameForm.AS_SPELLED`/`ANY_SPELLING` and their branches,
                        `_owners`, `_realizes`, `_name_signature`, `_run_parallel`,
                        `_full_name_source`, and the `linked` argument threaded through
                        three linkers that never read it. The comments that described
                        them -- an antecedent gate, an admission filter, an `_unlinked`
                        subtraction -- described code s79-s81 had already deleted.

WHAT IT SCORES. Paired against s_linker81, three five-project runs, gpt-5.6-terra/flex
(`results/audit_e2e_s82hy_r{1,2,3}_20260820`):

    macro F1   92.25 (sd 0.81)  against  91.91 (sd 0.21)   +0.34, p = 0.75
    macro F2   94.08 (sd 0.64)  against  92.32 (sd 0.38)   +1.76, positive in 3 of 3
                                                           runs (exact permutation
                                                           p = 0.25, the floor at n=3)
    TP 183.0 / FP 30.0 against 176.7 / 27.0; 81 LLM calls per run against 88.

By stage, per run, counting each (sentence, component) pair once: full name TP 148.0 /
FP 6.0 against 142.3 / 3.7, coreference 41.7 / 11.7 against 33.3 / 12.0, partial name
18.3 / 12.7 against 17.3 / 11.7.

Replicated on a second model, gpt-5.6-luna/flex, same three-run paired form
(`results/audit_e2e_s82luna_r{1,2,3}_20260820`). Luna is the looser judge -- both
variants carry three times the false positives they do on terra -- and the fixes are
worth more there:

    macro F1   85.38 (sd 0.95)  against  81.80 (sd 1.51)   +3.58, 3 of 3 runs
    macro F2   91.16 (sd 0.67)  against  87.36 (sd 0.89)   +3.80, 3 of 3 runs
    TP 184.0 / FP 77.0 against 176.0 / 88.0; 84 LLM calls per run against 93.

Luna moves the stages differently. Its full-name stage is the same on both variants
(TP 148.3, FP 13.0 against 14.0) and its partial-name stage is where s81 bleeds
(FP 33.0 against 30.3 at TP 13.7 against 17.7); its coreference stage keeps the same
gold on both (74.0 against 74.7) and differs on spurious links (37.7 against 47.3).
Terra's gain is concentrated in coreference recall (41.7 against 33.3), luna's in
precision everywhere.

The direction is the same on both models and every measure; only the size moves. Neither
result is significant at n = 3 (0.25 is the exact permutation floor), so what the pair
establishes is a consistent sign across two models, not a p-value.

MEASUREMENT POLICY. No benchmark vocabulary appears in this module, and after the
morphology swap no authored word list of any kind (GATE-06). The experiment log lives in
`../results/<round>/README.md` and the variant registry in `run_ablation.py`; read
`CLAUDE.md` for the measurement policy.

STANDALONE, by the convention `s_linker21` set for paper artifacts: no linker
superclass, one module.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from nltk.stem import WordNetLemmatizer

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
# Prompt constants. Every clause states a principle, not an enumeration of shapes
# (`pilot/prompt_audit.py` sized each generalization off six recorded s49 runs).
# No benchmark vocabulary appears here (GATE-06) and no clause names a surface
# form peculiar to these documents (GATE-07).
# ─────────────────────────────────────────────────────────────────────────────

#: What makes an alias valid. The leniency is load-bearing: removing this judge
#: entirely reads F1 94.57 against 96.42.
DOC_KNOWLEDGE_JUDGE_RULES = """An alias is valid when the document establishes an equivalence between a phrase and a single named component. It is invalid when the phrase is generic vocabulary or identifies anything other than that one component. When uncertain, prefer APPROVE."""

#: What the alias extractor is asked for. Which shapes qualify is the model's
#: judgement; the judge above decides validity.
DOC_KNOWLEDGE_EXTRACTION_RULES = """Find surface forms the document uses to refer to a single named component (introduced short forms, alternate names, or words of multi-word names when they alone clearly mean the full name). Reject terms whose ordinary English use dominates."""

#: The one prohibition the alias extractor carries. Stating it as a syntax instead
#: of a principle is worth little and is reported rather than hidden: the syntax
#: arm admitted 0 identifier fragments in 15 project-runs against the general
#: arm's 6 in one (`pilot/finetune_pilots.py --pilot aliascomp`).
ALIAS_EXCLUSION_RULES = """A fragment of a longer identifier is not an alias: if a term appears only as part of a compound or qualified name, do not include it."""

#: The full-name linker's admission contract, stated to the extractor instead of
#: enforced by a filter afterwards. s81 also told the extractor to drop names used
#: as ordinary English while demanding every mention "however incidental" -- two
#: orders at once. The use/mention call is the judge's (`STRICTER_CLAUSE`), so the
#: extractor is asked for named mentions and nothing else.
ENTITY_EXTRACTION_RULES = """Report a reference only when the sentence itself writes the component's name, spelled as the COMPONENTS list spells it or as one of the KNOWN ALIASES; count a name written with different spacing, hyphenation or compound joining as that name.

Do not report a component that the sentence only implies as a participant in a described interaction without naming it. Among the sentences that do name it, report every one, however incidental the mention: whether the mention carries an architectural claim is decided later."""

#: The full-name judge has no focus line. s81 asked its two halves as separate
#: passes and AND-ed them; s82 merged them into one `VALIDATION_FOCUS` sentence,
#: and this variant removes that sentence, because the rubric under it already
#: states both halves: `LAYERED_ENTITY_RULES` makes architectural participation the
#: approve-condition, and `STRICTER_CLAUSE` is entirely about a name identifying
#: this element rather than serving as ordinary vocabulary. Measured as a stage arm
#: over three runs a side on both models, over the same extraction pass
#: (`pilot/typed_prompt_pilots.py --group fullname3`): terra TP +1.0 (p = 0.50),
#: macro F2 -0.0 (p = 0.80); luna TP +1.0 (p = 0.80), macro F1 +0.1 (p = 0.90),
#: macro F2 +0.3 (p = 0.60). The strict side keeps its question, which no clause of
#: its rubric repeats.
#: The coreference judging question.
COREF_VALIDATION_FOCUS = (
    "Check coref resolution: does the referring expression in this sentence "
    "actually refer to the named component as an architectural participant?"
)

#: The coreference resolver's standard. Dropped in the general round: the
#: section-topic licence (0.0 of 578 recorded resolutions), the five listed role
#: phrases, and the terminal-word/abbreviation enumeration (1.7 antecedents per
#: run), all subsumed by "under any form the document uses for it".
COREF_RULES = """For each case, decide whether a pronoun or noun phrase that refers back in the target sentence refers back to a component named or aliased earlier in the context. Resolve when the surrounding sentences make one component the clear antecedent, under any form the document uses for it. Avoid resolving when two or more equally plausible antecedents exist."""

#: Full-name gate -- lenient: a stated name is a link unless a reject signal fires.
#: The four numbered reject-conditions this replaces are grounded elsewhere in the
#: same prompt: (1) in `QUALIFIED_CLAUSE`, (3) and (4) in `STRICTER_CLAUSE`, (2),
#: negation, in the last clause here.
LAYERED_ENTITY_RULES = (
    "Approve the link by default: the component is named here and the document treats "
    "it as part of the system. A mention that says nothing further about the component "
    "still counts as a valid link. Reject only on a positive ground -- that the sentence "
    "asserts nothing of this component, because the name is doing some other job here, "
    "or because the sentence denies what it would otherwise say of it."
)

#: Coreference gate -- strict: the component is NOT named in the sentence, so a
#: genuine referring expression plus an architectural claim is demanded.
LAYERED_COREF_RULES = (
    "These are coreference links: a pronoun or noun phrase in the sentence is claimed to "
    "refer back to the component, which is NOT named in the sentence itself. Approve only "
    "when the sentence contains a genuine referring expression that unambiguously points "
    "to THIS component and makes an architectural claim about it. Reject when there is no "
    "such referring expression or when the antecedent could equally be a different "
    "component. An expression denoting what a component acts on or produces -- the data, "
    "the artifact, the request, the result -- refers to that thing and not to the "
    "component, however clearly the component is the one acting on it. When uncertain, "
    "reject."
)

#: The morphology under which a sentence word still counts as a word of a component's
#: name. WordNet's lemmatizer over its noun and verb readings, applied to both sides:
#: the sentence token and the name's word are the same word when any reading of one
#: equals any reading of the other. **No word list is written here and none is written
#: anywhere else in this module** (GATE-06) -- the morphology is a general English
#: resource, not a set chosen against these documents. Read at one place: `_name_spans`.
#: The swap off `INFLECTIONS` was measured span by span over every (name, sentence)
#: pair of all five projects: 3697 pairs, the spans differ on 2, partial-name candidates
#: 109 -> 110 with gold 28 -> 28 (`pilot/lemma_swap_pilot.py --only E1`).
_LEMMATIZER = WordNetLemmatizer()

#: The readings a token is normalized under. Nouns and verbs are the two open classes a
#: component's name draws its words from.
LEMMA_READINGS = ("n", "v")


@lru_cache(maxsize=None)
def lemmas(word: str) -> frozenset:
    """Every ``LEMMA_READINGS`` reading of ``word``, casefolded.

    WordNet is a lexicon with an identity fallback: a word it does not know comes back
    unchanged, so a domain token no dictionary carries is compared by its own surface
    and nothing is invented for it.
    """
    folded = word.casefold()
    try:
        return frozenset(_LEMMATIZER.lemmatize(folded, reading)
                         for reading in LEMMA_READINGS)
    except LookupError as missing:  # the corpus is data, not a pip dependency
        raise RuntimeError(
            "s_linker85 needs WordNet: python -m nltk.downloader wordnet"
        ) from missing

#: The tokenizer that cuts a name or a sentence into words. Word boundaries only --
#: splitting compounds here tripled the candidate set and reached no extra gold link.
WORD_PATTERN = r"[A-Za-z]+[A-Za-z0-9]*|\d+"

#: The span-boundary gate, stated as what the expression IS rather than where its
#: characters sit. Deleting the code gate with no compensation is FP +7.0 (p = 0.01);
#: this sentence in front of the same judge is TP -0.4 / FP -0.2, both n.s.
#: (`pilot/fold_pilots.py --pilot foldqualified`).
QUALIFIED_CLAUSE = """An expression that occurs only as part of a longer joined or dotted identifier is naming a piece of that identifier, not a participant in what the sentence describes."""

#: The case-sensitivity gate, stated as the distinction the judge should draw.
#: Deleting the code gate is TP +4.0 at FP +1.8 (both p = 0.01); this sentence in
#: front of the same judge is TP +4.0 at FP +/-0.0 (`--pilot foldstricter`). It
#: names no surface form and no component.
STRICTER_CLAUSE = (
    "Some sentences use an ordinary English word that happens to coincide with a "
    "component's name. Approve only when the sentence uses that word as the name of "
    "the component; if it is used in its ordinary sense and the component is not what "
    "the sentence is talking about, reject. Capitalization is evidence for a name and "
    "its absence is evidence against, but neither settles it on its own."
)


class NameForm(Enum):
    """A point of the surface-realization relation, on two independent dimensions.

    *Fidelity* -- how exactly the sentence's characters must reproduce the name:

        ANY_CASE   the name, ignoring case

    *Extent* -- how much of the name has to be present:

        ANY_WORD   one word of the name, under an English inflectional ending

    s_linker64 scanned four points with four hand-written methods; s79-s81 retired
    all but these two, and s82 deletes the two enum members and `_name_spans`
    branches nothing reached (`AS_SPELLED`, `ANY_SPELLING`). ANY_CASE is the name
    test every stage shares; ANY_WORD is what the partial-name linker scans.
    """

    ANY_CASE = "any_case"
    ANY_WORD = "any_word"


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
    CODE_TOKEN = "lowercase, inside qualified name"
    VIA_ALIAS = "via known alias"
    INDIRECT = "indirect/unclear match"


@dataclass
class EvidenceBundle:
    """What a judge is told about a candidate beyond the candidate itself.

    The span and the preceding sentence also appear in the case header, and the
    repetition is deliberate: dropping either is neutral at the judging stage in
    isolation but costs precision composed (FP 8.3 against a 4-6 reference band,
    F1 95.2 against 96.42 +/- 0.42 over three five-project runs).
    """

    source: str
    matched_span: str
    mention_type: str          # MentionType.value (str for prompt embedding)
    preceding_text: str
    anchor_sentences: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Main linker
# ─────────────────────────────────────────────────────────────────────────────

class SLinker86:
    """Three linkers, fixed name-evidence order, no controller. Standalone."""

    _VARIANT_NAME = "s_linker86"

    #: Execution order. Full name first (it needs the least), partial name
    #: second, coreference last. No linker is shown what the earlier ones linked:
    #: the partial-name scan skips a sentence that states a whole name, which is
    #: a property of the sentence, not of the link set. Order therefore decides
    #: only which linker's `source` label survives `_union` on a duplicate pair.
    LINKERS = ("full_name", "partial_name", "coreference")

    # ── Resource bounds ──────────────────────────────────────────────────────
    # These cap prompt size and call count. No decision rule reads them: changing
    # one changes how much text a judge sees, never what counts as a link. Every
    # window is the same width on purpose -- the earlier per-step values (2, 3, 4,
    # 5) implied a calibration that was never measured.
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
        print("SLinker85 (full-name -> partial-name -> coreference; fixed order)")
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
                linker, sentences, components, name_to_id, sent_map
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

    def _run_linker(self, linker, sentences, components, name_to_id, sent_map):
        """Dispatch. No linker receives the links the earlier ones produced.

        s25 passed that set to every linker and subtracted it at each candidate
        boundary (`_unlinked`); s79 removed the subtraction and s81 kept passing an
        argument no linker read, under a docstring describing the deleted design.
        Whatever a later linker re-proposes, `_union` merges by pair.
        """
        if linker == "full_name":
            return self._run_full_name_linker(
                sentences, components, name_to_id, sent_map)
        if linker == "partial_name":
            return self._run_partial_name_linker(sentences, components, sent_map)
        if linker == "coreference":
            return self._run_coreference_linker(
                sentences, components, name_to_id, sent_map)
        raise RuntimeError(f"unknown linker: {linker!r}")

    # ── Concurrency and small helpers ────────────────────────────────────────

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

        One predicate for a question three stages once asked with three copies of the
        same expression. Two of those callers are gone (the full-name admission filter,
        s79; the coreference antecedent gate, s80), so the live caller is the
        partial-name scan's whole-name exclusion. The mention-label classifier asks the
        same question decomposed, because it must know *which* name matched.
        """
        names = (comp_name, *self._names_by_component().get(comp_name, ()))
        return any(self._find_exact_form(text, name) for name in names)

    def _window(self, snum: int, sentences):
        """The sentences within ``CONTEXT_SENTENCES`` of this one, in document order.

        One predicate for a condition the denotation step and the coreference resolver
        used to spell two ways; verified to select the same set over every sentence of
        all five documents.
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
    def _prompt_doc_knowledge_judge(comp_names, proposals) -> str:
        """Judge the proposed aliases, term with component.

        s81 asked for bare terms, so a term two components both claimed came back
        undecidable and the caller kept whichever the extractor recorded last.
        """
        return f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{json.dumps(proposals)}

{DOC_KNOWLEDGE_JUDGE_RULES}

Return JSON, echoing each approved mapping in full:
{{"approved": [{{"term": "term1", "component": "FullComponent"}}]}}
JSON only:"""

    @staticmethod
    def _prompt_extraction(comp_names, mappings, batch) -> str:
        return f"""Extract ALL references to components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings)}' if mappings else ''}

{ENTITY_EXTRACTION_RULES}

{QUALIFIED_CLAUSE}

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
        # The ground against the link is asked of the strict gate only. "Approve by
        # default" and "state the strongest ground for rejecting" are contradictory
        # standards to put in one prompt, and only the strict arm was measured.
        decide = (
            " then decide approve true/false based on that claim."
            if not strict else
            " then state the strongest ground there is for rejecting this case under the\n"
            "rules above (or \"none\" if there is none), then decide: approve unless that "
            "ground is one\nthe rules above make decisive. An objection you could raise "
            "against most sentences is not\na ground for rejecting this one."
        )
        field = "" if not strict else ', "objection": "<strongest ground to reject, or none>"'

        # `QUALIFIED_CLAUSE` and `STRICTER_CLAUSE` join the lenient rubric only: they
        # ground the reject-conditions the enumeration used to list, and the
        # coreference cases carry no whole-name surface for either to speak about.
        tail = "" if strict else f"\n{QUALIFIED_CLAUSE}\n{STRICTER_CLAUSE}\n"
        return f"""Validate components in a document.{f" {focus}" if focus else ""}

COMPONENTS: {', '.join(comp_names)}

{rules}
{tail}
For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim),{decide}

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>"{field}, "approve": true}}]}}
JSON only:"""

    @staticmethod
    def _prompt_coref(comp_names, sentence_table, targets) -> str:
        """One batch: the window sentences once, each case carrying its target's text.

        s81 pasted every target's +/-5 window inline, so a batch of 10 targets rendered
        ~110 copies of ~20 distinct sentences. s82 first replaced that with the
        denotation step's shape -- one table, cases that are numbers -- and paid for it:
        gold resolutions held (79.3 per five-project run against 80.3) but spurious ones
        rose to 60.0 from 45.7, and the strict judge downstream, handed noisier batches,
        kept 30.7 gold instead of 40.3. The resolutions still landed on a declared
        target 100% of the time, so nothing was misread; the target sentence had simply
        stopped being salient next to the question about it.

        The form below is the pilot's third arm, measured on the same stage over three
        five-project runs: the table stays, and each case shows its target's text with
        the context named by number. Gold resolutions 96.7 per run against 80.3 (s81)
        and 79.3 (the table alone), spurious 51.0 against 45.7 and 60.0 -- the best
        precision of the three at the highest recall.
        """
        blocks = [
            f"--- Case {t['case']} ---\n"
            f"TARGET S{t['target']}: {t['text']}\n"
            f"CONTEXT: sentences S{min(t['context'])}-S{max(t['context'])} above."
            for t in targets
        ]
        return f"""Resolve references (pronouns and noun phrases that refer back) to components.

COMPONENTS: {', '.join(comp_names)}

SENTENCES (the document text the cases are drawn from)
{json.dumps(sentence_table)}

For each TARGET sentence below, identify any pronoun or noun phrase in THAT sentence
that refers back to a component listed above. Read the TARGET's context in SENTENCES.
If a target sentence has no such reference to a listed component, return no resolution
for it. Be conservative — only include resolutions you are CERTAIN about.

{chr(10).join(blocks)}

{COREF_RULES}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "reference": "the server", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

JSON only:"""

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
        """Propose aliases over the whole document, then judge them.

        s81 tested the judge's reply for truthiness, and one event -- "the judge did
        not answer" -- came out three ways: an unparseable reply approved *every*
        proposal, a parsed reply with no ``approved`` key approved none, and a genuine
        empty approval list was honoured only after a wasted retry. Here an empty list
        is an empty result, both no-answer shapes fall back to the lenient default this
        stage is documented to have, and the fallback says so. Proposals are carried as
        (term, component) pairs rather than a term-keyed dict, so two components
        claiming one term reach the judge as two proposals instead of collapsing to
        whichever the extractor reported last.
        """
        self.llm.set_phase("phase_25_doc_extract")
        comp_names = [c.name for c in components]
        doc_lines = [s.text for s in sentences]

        data1 = self._ask(
            self._prompt_doc_knowledge_extract(comp_names, doc_lines),
            timeout=300, label="Doc knowledge",
        )

        proposals: list[dict] = []
        seen: set[tuple[str, str]] = set()
        if data1:
            recs = []
            for key in ("abbreviations", "synonyms"):
                value = data1.get(key, [])
                if isinstance(value, dict):
                    value = [{"term": k, "component": v} for k, v in value.items()]
                if isinstance(value, list):
                    recs += value
            for rec in recs:
                if not isinstance(rec, dict):
                    continue
                term, full = rec.get("term"), rec.get("component")
                if term and full in comp_names and (term, full) not in seen:
                    seen.add((term, full))
                    proposals.append({"term": term, "component": full})

        approved: set[tuple[str, str]] = set()
        if proposals:
            data2 = self._ask(
                self._prompt_doc_knowledge_judge(comp_names, proposals),
                timeout=120, label="Doc knowledge judge",
                phase="phase_25_doc_judge", require_present="approved",
            )
            verdicts = data2.get("approved") if isinstance(data2, dict) else None
            if isinstance(verdicts, list):
                for v in verdicts:
                    if isinstance(v, dict):
                        pair = (v.get("term"), v.get("component"))
                        if pair in seen:
                            approved.add(pair)
            else:
                # No `approved` key: the judge did not answer. Prior work's default.
                print("    Doc knowledge judge: no verdict, approving all proposals")
                approved = set(seen)

        knowledge = DocumentKnowledge()
        # A term two components both hold approval for names no single component, so
        # no alias is derivable from it. It is dropped and reported.
        claimants: dict[str, list[str]] = {}
        for term, comp in approved:
            claimants.setdefault(term, []).append(comp)
        for term, comps in claimants.items():
            if len(comps) > 1:
                print(f"    Alias dropped (ambiguous): {term} -> {', '.join(comps)}")
                continue
            knowledge.aliases[term] = comps[0]
            print(f"    Alias: {term} -> {comps[0]}")
        return knowledge

    # ═════════════════════════════════════════════════════════════════════════
    # Linker 1 — FULL NAME: the sentence states a name of the component.
    # ═════════════════════════════════════════════════════════════════════════

    def _run_full_name_linker(self, sentences, components, name_to_id, sent_map):
        candidates_by_key = self._extract_named_mentions(
            sentences, components, name_to_id, sent_map
        )
        # The candidate set is what the LLM extractor proposed and nothing else.
        # No admission filter and no lexical top-up: the contract the filter enforced
        # and the recall floor the two tight scans drew are both stated in
        # ``ENTITY_EXTRACTION_RULES``, so the extractor is held to the linker's own
        # contract rather than corrected after the fact.
        candidates = list(candidates_by_key.values())
        bundles = {
            (c.sentence_number, c.component_id): self._build_evidence_bundle(c, sent_map)
            for c in candidates
        }
        approved, decisions = self._validate_with_evidence(
            candidates, bundles, components, sent_map,
            phase_tag="phase_25_full_name_judge",
            stage_label="full_name",
        )
        links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name,
                       source="full_name")
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

        An earlier revision sampled this prompt twice and unioned the results; the
        second sample moved neither score beyond noise, so the pipeline pays for one.
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

    def _scan(self, sentences, components):
        """THE CANDIDATE GENERATOR: every (sentence, component) pair whose sentence
        carries one word of the component's name, unless the sentence writes a whole
        name of it -- that pair is the full-name linker's, and the target-blind,
        single-pass denotation judge that hears partial names is not the judge for
        it. Of the 161 pairs the ungated scan adds, 140 come from dropping that skip
        (s80 post-mortem, replayed over the five catalogs).

        s_linker64 wrote three generators with three regexes; s79-s81 reduced them to
        one row of a `SCANS` table with two options that were never set, and s82
        deletes the table. Nothing here admits a link: every pair is a case for a
        judge. Later spans of the same pair overwrite earlier ones, so the recorded
        `matched_text` is the last surface found in the sentence -- s_linker64's
        behaviour at both rebuilt sites.
        """
        candidates = {}
        for sentence in sentences:
            text = sentence.text
            for component in components:
                if self._states_a_name(text, component.name):
                    continue  # a whole name is stated: the full-name linker's pair
                for start, end in self._name_spans(text, component.name,
                                                   NameForm.ANY_WORD):
                    candidates[(sentence.number, component.id)] = CandidateLink(
                        sentence.number, text, component.name, component.id,
                        text[start:end], source="partial_name_candidate",
                    )
        return list(candidates.values())

    @classmethod
    def _name_spans(cls, text, name, form: NameForm):
        """**The relation.** Spans of ``text`` that write ``name`` at ``form``.

        The whole deterministic layer of this workflow is this function and the two
        values of ``NameForm``. It reads the runtime catalog and WordNet's morphology,
        and nothing else; no benchmark vocabulary reaches it, and since the swap off
        `INFLECTIONS` no word list either (GATE-06).

        The branches were separate methods in ``s_linker64``, verified identical to
        these over every (name, sentence) pair of all five projects
        (`pilot/rule_audit.py --only A2`, `pilot/test_s65_one_relation.py`).
        """
        if form is NameForm.ANY_CASE:
            return [(m.start(), m.end()) for m in re.finditer(
                rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE)]

        if form is NameForm.ANY_WORD:
            # Extent, not fidelity: one word of the name is enough, at any inflection of
            # it. Lemmatizing both sides makes the test symmetric -- an inflected word
            # inside a name reaches the base form in the sentence, which stripping
            # endings off the sentence token alone cannot do -- and it is a prefix of
            # nothing: `web` does not own `webrtc` or `webcams`.
            forms = [lemmas(w) for w in re.findall(WORD_PATTERN, name)]
            return [
                (m.start(), m.end())
                for m in re.finditer(WORD_PATTERN, text)
                if any(lemmas(m.group(0)) & word for word in forms)
            ]

        raise ValueError(f"unknown name form: {form!r}")

    @staticmethod
    def _in_dotted_path(text, start, end) -> bool:
        """True when text[start:end] is glued to a dot on either side, as in x.y.

        The single definition of "inside a qualified name". Two divergent copies used
        to exist; the divergence never changed a result over 3697 (name, sentence)
        pairs, so the stricter reading is the one kept.
        """
        before = (start > 1 and text[start - 1] == "."
                  and text[start - 2].isalnum())
        after = (end + 1 < len(text) and text[end] == "."
                 and text[end + 1].isalnum())
        return before or after

    # ── Evidence bundles and the judge ───────────────────────────────────────

    def _classify_mention_typed(self, comp_name: str, text: str) -> MentionType:
        """Label how the name appears, using the one matching test.

        The case distinction compares the matched surface against the name rather
        than running a second case-sensitive predicate: `_find_exact_form` already
        returns what it matched. Measured indistinguishable from the two-predicate
        form it replaces, at the stage and end to end.
        """
        matched = self._find_exact_form(text, comp_name)
        if matched:
            if self._all_occurrences_in_qualified_path(comp_name.lower(), text):
                return MentionType.CODE_TOKEN
            return (MentionType.PROPER_STANDALONE if matched == comp_name
                    else MentionType.LOWERCASE_PROSE)
        for alias in self._names_by_component().get(comp_name, ()):
            if self._find_exact_form(text, alias):
                return MentionType.VIA_ALIAS
        return MentionType.INDIRECT

    @classmethod
    def _all_occurrences_in_qualified_path(cls, comp_lower: str, text: str) -> bool:
        any_match = False
        for m in re.finditer(rf'\b{re.escape(comp_lower)}\b', text):
            any_match = True
            if not cls._in_dotted_path(text, m.start(), m.end()):
                return False
        return any_match

    def _build_evidence_bundle(self, candidate, sent_map):
        comp_name = candidate.component_name
        snum = candidate.sentence_number
        # The label is computed and all but two of its values are dropped. Replaying
        # s80's 610 full-name decisions against s79's, the loss from removing the field
        # sits in the two values whose fact is not in the sentence the judge holds:
        # `via known alias` (73% of those approvals lost) needs the alias table and
        # `lowercase, inside qualified name` (100%) needs every occurrence tested
        # against a dotted path. The other three restate the case header and cost 3-21%.
        mention_type = self._retained_mention_label(
            comp_name, candidate.sentence_text
        )
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

    #: The mention labels the judge cannot re-derive from the sentence it is shown.
    #: Everything else `_classify_mention_typed` can say is a restatement of the case
    #: header, and s80 measured the cost of dropping it at 3-21% of those approvals.
    RETAINED_MENTION_TYPES = frozenset({
        MentionType.VIA_ALIAS,
        MentionType.CODE_TOKEN,
    })

    def _retained_mention_label(self, comp_name: str, text: str) -> str:
        """The computed label, or "" where the judge is holding the fact already."""
        mention = self._classify_mention_typed(comp_name, text)
        return mention.value if mention in self.RETAINED_MENTION_TYPES else ""

    def _format_evidence(self, bundle: EvidenceBundle) -> str:
        mention = (f", mention={bundle.mention_type}"
                   if bundle.mention_type else "")
        lines = [
            f"  Evidence: source={bundle.source}, span=\"{bundle.matched_span}\""
            f"{mention}",
        ]
        if bundle.preceding_text:
            lines.append(f"  [prev: \"{bundle.preceding_text}\"]")
        if bundle.anchor_sentences:
            lines.append("  Anchors (confirmed refs):")
            for a in bundle.anchor_sentences:
                lines.append(f"    {a}")
        return "\n".join(lines)

    def _validate_with_evidence(self, candidates, bundles, components, sent_map,
                                phase_tag, stage_label):
        """One judging pass over the candidates, with their evidence lines.

        s81 sent this prompt twice with two focus lines and AND-ed the verdicts. Over
        its three recorded runs the passes disagree on 4.0 of ~196 candidates per
        five-project run and neither disagreement direction is stable, so the two
        questions are asked once, at half the calls -- and s86 asks them only in the
        rubric: the focus sentence restated the two halves the rubric already states,
        so this pass sends no focus line at all.
        """
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
            verdicts = self._run_validation_pass(
                comp_names, case_strings, "", phase_tag)
            for i, (_case_text, c) in enumerate(cases):
                ok, claim, _ = verdicts.get(i, (False, "", ""))
                decisions[(c.sentence_number, c.component_id)] = {
                    "approved": ok,
                    # The quote the prompt demands before the verdict. s81 asked for
                    # it and parsed only the verdict, so no trace could check that
                    # the claim it rests on is in the sentence.
                    "claim": claim,
                    "path": f"{stage_label}_judged" if ok
                            else f"{stage_label}_rejected",
                    "stage": f"{stage_label}_judge",
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
        results: dict[int, tuple[bool, str, str]] = {}
        if data:
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    val = v.get("approve", False)
                    approve = (
                        val is True
                        or (isinstance(val, str) and val.lower() == "true")
                    )
                    results[idx] = (approve, str(v.get("claim", "")).strip(),
                                    str(v.get("objection", "")).strip())
        return results

    # ═════════════════════════════════════════════════════════════════════════
    # Linker 2 — PARTIAL NAME: the sentence carries one word of a name.
    # ═════════════════════════════════════════════════════════════════════════

    def _run_partial_name_linker(self, sentences, components, sent_map):
        candidates = self._scan(sentences, components)
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

{QUALIFIED_CLAUSE}

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
                # The response contract only. The substring check that used to
                # follow it voided 0 of 380 verdicts over six five-project runs, which
                # is s_linker48's separation restated: demanding a committed quote is
                # worth 35.2 TP, verifying it is worth nothing.
                valid = denotation in {"participant", "associated"} and bool(claim)
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

    def _run_coreference_linker(self, sentences, components, name_to_id, sent_map):
        resolved, metadata = self._resolve_references(
            sentences, components, name_to_id, sent_map
        )
        raw = resolved
        approved, decisions = self._validate_coref_links(
            raw, sent_map, components, metadata)
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

        There is no antecedent gate: requiring the antecedent sentence to state a name
        of the component is TP +/-0.0 / FP +/-0.0 on what coreference actually
        contributes -- pairs no earlier linker produced -- when replayed on the runs'
        own recorded resolutions (`pilot/fold_pilots.py --pilot foldantecedent_net`).
        The resolution must still *report* an antecedent, and both sentence numbers it
        reports are checked against the document: a number the model invents cannot
        name a real sentence.
        """
        comp_names = get_comp_names(components)
        all_coref = []
        coref_metadata: dict = {}
        self.llm.set_phase("phase_25_coreference")

        for batch_num, batch in self._iter_batches(sentences, self.COREFERENCE_BATCH):
            targets = []
            window_ids = set()
            for i, sent in enumerate(batch, 1):
                window = [w.number for w in self._window(sent.number, sentences)]
                window_ids.update(window)
                targets.append({"case": i, "target": sent.number,
                                "text": sent.text, "context": window})
            sentence_table = [
                {"sentence": n, "text": sent_map[n].text}
                for n in sorted(window_ids) if n in sent_map
            ]

            data = self._ask(
                self._prompt_coref(comp_names, sentence_table, targets), timeout=600,
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
                cid = name_to_id[comp]
                all_coref.append(SadSamLink(snum, cid, comp, source="coreference"))
                coref_metadata[(snum, cid)] = {
                    "reference": res.get("reference", ""),
                    "antecedent_sentence": ant_snum,
                    "antecedent_text": res.get("antecedent_text", ""),
                    "raw_resolution": res,
                }
        return all_coref, coref_metadata

    def _validate_coref_links(self, coref_links, sent_map, components, metadata):
        """Single judging pass, shown the resolution it is judging.

        s82 gave this judge a sentence and a component name, so it had to guess which
        expression was claimed to refer and to what, and it rejected half the gold
        resolutions put to it (terra kept 49.8%). The resolver had already committed to
        both -- the referring expression and the quote it read as the antecedent --
        and neither is recoverable from the case.
        """
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
                res = metadata.get((lk.sentence_number, lk.component_id), {})
                claimed = "".join(
                    line for line in (
                        f'  Claimed reference: "{res.get("reference")}"\n'
                        if res.get("reference") else "",
                        f'  Claimed antecedent (S{res.get("antecedent_sentence")}): '
                        f'"{res.get("antecedent_text")}"\n'
                        if res.get("antecedent_text") else "",
                    )
                )
                cases.append((
                    lk,
                    f'Case {i+1}: pronoun/role-ref -> {lk.component_name}\n'
                    f'{claimed}'
                    f'  {p}"{sent.text}"',
                ))
            results = self._run_validation_pass(
                comp_names, [c for _, c in cases], COREF_VALIDATION_FOCUS,
                phase_tag="phase_25_coreference_judge", strict=True,
            )
            for idx, (lk, _case) in enumerate(cases):
                approved, claim, objection = results.get(idx, (False, "", ""))
                decisions[(lk.sentence_number, lk.component_id)] = {
                    "approved": approved,
                    "claim": claim,
                    "objection": objection,
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
