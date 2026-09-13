"""S-Linker120 — one judge, one rule: what a trace link is and how to read the evidence.

`s_linker110` judges its two name streams with two prompts whose rubrics state opposite
defaults, and routes a case to one or the other by which scan proposed it. This variant
asks **one question of every candidate**: a trace link holds when the sentence makes an
architectural claim about the component. There is no lenient row and no strict row. What
differs between candidates is the **evidence computed from the match** — what the
sentence writes of the name, which components the same word could reach, what the code
can tell about the expression's place in the sentence, and which other sentences name
the component — and the rule says how to read each of those, not which rubric to apply.

**Two stages become one.** `LINKERS` is `("name", "coreference")`: both scans are merged
by pair, judged in one pass, and relabelled `full_name` / `partial_name` at the link, so
every downstream view — the links CSV, the phase log, the RQ3/RQ4 attribution — reads the
two stages it always read. The coreference linker is untouched and outside this union:
its cases are not matches — there is no span the code computed — so evidence computed
from the match has nothing to say about them.

**Stage measurement** (`pilot/union_pilots.py`, `../results/union_round/`): fixed
recorded candidates, both arms in the same invocation, five projects, alias table pinned.

    model  samples  gold      p        spurious   p        net      p       precision
    terra  5        +0.2      1.000    -12.6      0.000    +13.2    0.025   0.878 -> 0.939
    luna   3        +0.3      1.000    -17.7      0.008    +18.7    0.011   0.789 -> 0.859

**Gold-neutral on both models, spurious down on both, at the same 14 judging calls and
one prompt instead of two.** Composition is level-3 clean on terra (0 gold pairs the
union removes that nothing downstream re-proposes) and carries 2 distinct pairs on luna,
below the recorded TP floor of 4.8 (`pilot/union_composition.py`).

**Thirteen iterations, each one change, each measured against the control beside it.**
They live in `union_iterations.py` as data — rule text, case format, verdict contract and
numbers — so the trail can be read and re-run (`pilot/union_pilots.py --arms control v3
v13` puts two of them in one invocation; this file runs `v13`, written out below, and
`pilot/union_defensibility.py` checks the two are the same bytes). Three results from the
round are worth more than the arm:

  * **An evidence field restrains when it is stated and misleads when it is weighted.**
    Iteration 1 stated the alternative set as a ground for rejecting and lost 7.6 gold on
    a bucket that is 0.765 gold; iteration 6 removed the same field and gained **26.4
    spurious**. Between those two numbers is the whole design law, measured twice in
    opposite directions.
  * **The company a case keeps is part of its evidence.** Three successive rewrites of
    the rule left luna's word-only row at ~10 gold against a control's ~21. Grouping
    those cases by what the match computed — one code fact, no prompt change — recovered
    the row, and letting the call carry what its batch's evidence has (no catalog, the
    head's denotation contract) finished it.
  * **`s_linker25`'s refusal is about the arrangement, not the target.** Showing the
    component to a case whose sentence writes only one word of a name cost gold on luna
    (-9.4) and nothing much on terra; blinding it recovered 1.4 of 6.4 on terra and
    nothing on luna. What that stream actually loses to is being asked an identity
    question, in any of the several ways a merged prompt can ask one.

**Defensibility is enforced, not asserted.** `pilot/union_defensibility.py` (25 checks)
holds the rule to GATE-06/GATE-07: every clause that states a criterion is a **verbatim
slice of a rule constant this branch already had** — `MENTION_COUNTS` and
`POSITIVE_GROUND` are computed slices of `LAYERED_ENTITY_RULES`, `ACTS_ON` is asserted
inside `LAYERED_COREF_RULES`, `QUALIFIED_CLAUSE` and `STRICTER_CLAUSE` are carried whole
— and the residue is the definition of a trace link plus one line per evidence field,
each with a declared ground. Zero benchmark words of 63 catalog names, zero dotted
identifiers, zero document-shape enumerations, zero corpus-grounded sentences. What the
judge punches on is exactly two things: whether the sentence makes an architectural
claim about the component, and what the expression denotes where no name is written.
`LAYERED_ENTITY_RULES`' first sentence — "Approve the link by default" — is deliberately
**not** carried, and its absence is asserted: a default belongs to a stream, and this
rule has none.

**Invariants.** `pilot/test_s120_union.py` (five projects, no calls): the merged stream
is exactly `full ∪ partial` at the head's own bytes, every candidate keeps the stage
label its links and phase log are read by, every case carries its sentence and span, a
case whose match computed no component names none, no case invents an evidence field,
the verdict contract follows the batch, and an empty reply keeps nothing.

**Standalone.** This file is the whole workflow, not a subclass: the scan, the name
relation, the knowledge module, the union judge, the coreference linker, every prompt and
every rule constant are here, and `SLinker120`'s MRO is `(SLinker120, object)`. What is
inherited from `s_linker110` is its *text*, copied — which is what makes the diff between
the two files exactly the change this round measures and nothing else
(`pilot/test_s120_standalone.py` checks that copy block by block against the ancestor,
including the coreference prompt byte for byte). The eleven blocks that are byte-identical
across the whole family — the tracing wrapper, the JSON call path, the checkpoint and log
writers, the per-phase metrics, the batching and the log's views — live in `linker_infra`
and are called from the methods that used to hold them, exactly as in the ancestor.

**Lineage.** `s_linker110`, unchanged except at the judging of the two name streams.
"""
from __future__ import annotations

import json
import os
import re
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
from llm_sad_sam.linkers.experimental.linker_infra import (
    TracingLLMClient, ask_json, backend_tag, checkpoint_dir, save_phase_state,
    log_entry, write_run_logs, phase_metrics, iter_batches, link_view,
    decision_view, linker_feedback,
)
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend

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

#: The coreference judging question.
COREF_VALIDATION_FOCUS = (
    "Check coref resolution: does the referring expression in this sentence "
    "actually refer to the named component as an architectural participant?"
)

#: The coreference resolver's standard, and only the standard. Dropped in the general
#: round: the section-topic licence (0.0 of 578 recorded resolutions), the five listed
#: role phrases, and the terminal-word/abbreviation enumeration (1.7 antecedents per
#: run), all subsumed by "under any form the document uses for it". Dropped here: the
#: opening sentence, which restated the question the prompt's own preamble already asks
#: ("identify any pronoun or noun phrase in THAT sentence that refers back to a
#: component listed above"). s56 measured deleting that preamble at TP -16.2, because
#: it is also the input-format contract -- which block is the TARGET, that a target
#: with no referring expression yields nothing -- and this cut is the untried other
#: half: the contract stays, the restatement goes. 163 B leave each of the 40 resolver
#: calls a five-project run makes, the largest instruction item in the module
#: (`pilot/typed_prompt_pilots.py --group resolve`): terra composed TP +1.7, macro F1
#: -0.2 (p = 0.80), F2 +0.2; luna TP +/-0.0 (p = 1.00), macro F1 +0.2, F2 +0.3.
COREF_RULES = """Resolve when the surrounding sentences make one component the clear antecedent, under any form the document uses for it. Avoid resolving when two or more equally plausible antecedents exist."""

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
LAYERED_COREF_RULES = """These are coreference links: a pronoun or noun phrase in the sentence is claimed to refer back to the component, which is NOT named in the sentence itself. Approve only when the sentence contains a genuine referring expression that unambiguously points to THIS component and makes an architectural claim about it. Reject when there is no such referring expression or when the antecedent could equally be a different component. An expression denoting what a component acts on or produces refers to that thing and not to the component, however clearly the component is the one acting on it. When uncertain, reject."""

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

# ─────────────────────────────────────────────────────────────────────────────
# The union rule. One rule for every candidate: what a trace link is, and how to
# read each piece of evidence the match computed. No row, no default, no second
# rubric — what differs between candidates is the value of the evidence fields.
#
# Every clause that states a criterion is a **slice of a rule constant above**,
# computed rather than retyped, so quotation is mechanical and a drift in the
# ancestor's text is a drift here (`pilot/union_defensibility.py` checks each one
# against the constant it came from, and against `s_linker110`'s copy of it).
# ─────────────────────────────────────────────────────────────────────────────

_ENTITY_SENTENCES = [s.strip() for s in LAYERED_ENTITY_RULES.split(". ") if s.strip()]

#: "A mention that says nothing further about the component still counts as a valid
#: link." Carried because it is the one thing the lenient rubric says that is about
#: links rather than about a stream's default.
MENTION_COUNTS = _ENTITY_SENTENCES[1] + ". "

#: "Reject only on a positive ground -- that the sentence asserts nothing of this
#: component, because the name is doing some other job here, or because the sentence
#: denies what it would otherwise say of it."
POSITIVE_GROUND = _ENTITY_SENTENCES[2] if len(_ENTITY_SENTENCES) > 2 else ""

#: The reference clause of the strict gate: what an expression denotes when the
#: component is the actor. It is about reference, not about coreference, so it is
#: about every case here.
ACTS_ON = ("An expression denoting what a component acts on or produces refers to "
           "that thing and not to the component, however clearly the component is "
           "the one acting on it.")
assert ACTS_ON in LAYERED_COREF_RULES

#: `LAYERED_ENTITY_RULES`' first sentence is deliberately absent: "Approve the link by
#: default" is a *stream's* default, and this rule judges one stream.
assert _ENTITY_SENTENCES[0] not in MENTION_COUNTS + POSITIVE_GROUND

#: What a trace link is. The only authored sentence of the rule that states a
#: criterion, and it states the architectural definition and nothing else.
_DEFINITION = ("A trace link holds between a sentence and a component when the "
               "sentence makes an architectural claim about that component -- when it "
               "says something about that component as a participant in the system "
               "this document describes. ")

#: The input contract: what a case contains, and what follows from a case that
#: contains no component. Not a criterion — the shape of the input.
_FORMAT = ("Every case gives you the expression the sentence uses, the sentence "
           "itself, the evidence the document supplies, and -- where the sentence "
           "writes a name of it -- the component that name reaches. A case whose "
           "sentence writes no name of any component carries no component: decide "
           "what the expression denotes there, and nothing about identity.")

#: One line per evidence field, each saying what the field is evidence *of*. The
#: `writes` line ends in the head's own denotation question, which is what a case
#: with no component is being asked (`_classify_denotations`, s_linker25).
_WRITES_LINE = ("  writes -- what this sentence writes of the component's name: the "
                "whole name, a short form the document established for it, or one word "
                "of the name. A shorter surface leaves more readings open; it does not "
                "make the reading in front of you wrong. Where the sentence does not "
                "write the name as such, ask what the expression itself denotes in its "
                "local context: a participant in the system, or something merely "
                "associated with software.")

_FIELD_LINES = (
    "  alternatives -- other components whose names carry the same word. They are "
    "what the expression could be reaching instead of this one.\n"
    "  mention -- what the code can tell about the expression's place in the "
    "sentence.\n"
    "  anchors -- other sentences of this document that name this component. They fix "
    "what the name means in the document; they do not decide this sentence."
)

#: `STRICTER_CLAUSE` says where it applies. The clause is about an ordinary word
#: coinciding with *a component's name*, so a case that carries no component has no
#: subject for it; the general round measured the same mistake in the other direction
#: -- `QUALIFIED_CLAUSE` moved into the coreference rubric cost TP 3.0, because those
#: cases contain no identifier for it to be about. Scoping is not a second rubric: it
#: is the clause saying what it is about.
TRACE_LINK_RULE = f"""{_DEFINITION}{MENTION_COUNTS}

{_FORMAT} The evidence says what the expression is doing here; none of it is a verdict.

{_WRITES_LINE}
{_FIELD_LINES}

{POSITIVE_GROUND}

Where the case gives you a component: {STRICTER_CLAUSE}

{QUALIFIED_CLAUSE} {ACTS_ON}"""

#: The quote-before-verdict demand, and the reply contract it names. Demanding a
#: committed quote is worth 35.2 TP (s_linker48); verifying it against the sentence
#: voided 0 of 380 verdicts over six runs, so it is demanded and not re-checked.
UNION_DEMAND = ('For each case, first quote the EXACT words from the sentence the '
                'verdict rests on -- the words that state the architectural claim '
                'about the component, or "none" if the sentence makes no such claim '
                '-- then decide approve true/false based on that quote.')

UNION_REPLY = ('{"validations": [{"case": 1, "claim": "<exact quote or none>", '
               '"approve": true}]}')


@dataclass(frozen=True)
class RuleSpec:
    """What the judge says, what a case prints, and what the reply must answer.

    Held as data rather than spelled into `_prompt_union`, so that the thirteen
    measured versions of this judge (`union_iterations.py`) and the one this file
    runs are the same kind of object: `pilot/union_pilots.py --arms control v3 v13`
    puts any two of them in one invocation, and `pilot/union_defensibility.py`
    checks that `UNION_V13` below is byte-for-byte `ITERATIONS["v13"]`.
    """

    rule: str
    demand: str
    reply: str
    #: Which evidence fields a case may print. A field the match did not compute is
    #: not printed; a field not named here is not printable at all.
    fields: tuple[str, ...]
    #: A case whose sentence writes only one word of a name carries no component,
    #: no anchors and no alternatives — every one of those names a component.
    blind_word_only: bool = True
    #: Where a blind case may still print the component its word came from. `hidden`
    #: is v13: the slot is filled by the evidence, so a match that computed no
    #: component prints none. `header` puts the component back in the case header;
    #: `evidence` puts it on the `writes` line, which is the fact it came from. The
    #: two are the arms of the unblinding round — the question is whether an
    #: instruction can say what that slot means well enough that withholding it is
    #: not needed (`union_iterations` v14-v16).
    word_only_component: str = "hidden"
    #: Whether a blind case prints its anchors — the other sentences of the document
    #: that name this component. False is v13 (an anchor names a component, and a
    #: blind case carries none). A case that already names its component can carry
    #: them, and they are what fixes a generic word as this document's name for it.
    word_only_anchors: bool = False
    #: Cases are grouped by what the match computed: every case carrying a component
    #: in one batch, every case carrying none in another. Same rule, same template,
    #: same call count — what it changes is the company a case keeps.
    batch_by_evidence: bool = True
    #: A call with no component in any of its cases is given no catalog and answers
    #: the head's denotation contract: there is nothing to approve against.
    contract_follows_batch: bool = True
    verdict: str = "boolean"
    clauses: str = ""


#: The adopted version. `union_iterations.ITERATIONS["v13"]` is the same text with the
#: trail's metadata attached; this is the copy the workflow runs.
UNION_V13 = RuleSpec(
    rule=TRACE_LINK_RULE, demand=UNION_DEMAND, reply=UNION_REPLY,
    fields=("writes", "alternatives", "mention"),
)

#: The iteration name this file's rule is. Any other name is fetched from the trail.
ACTIVE_ITERATION = "v13"


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
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

class MentionType(Enum):
    """How a component name appears in a sentence."""
    PROPER_STANDALONE = "proper case, standalone"
    LOWERCASE_PROSE = "lowercase mention"
    CODE_TOKEN = "lowercase, inside qualified name"
    VIA_ALIAS = "via known alias"
    INDIRECT = "indirect/unclear match"


# ─────────────────────────────────────────────────────────────────────────────
# Main linker
# ─────────────────────────────────────────────────────────────────────────────
class SLinker120:
    """Two linkers, one rule over both name streams, no controller. Standalone.

    The two name scans of `s_linker110` are merged by pair and judged in one pass
    against one rule — what a trace link is, and how to read the evidence the match
    computed — and the coreference linker is the head's, untouched. No linker base
    class: see the module docstring for what is inlined from where.
    """

    _VARIANT_NAME = "s_linker120"

    #: Execution order. The two name scans are one stage here: they propose into one
    #: stream, one judge reads it, and the links carry the stage labels the head's
    #: two linkers gave them (`_stage_of`). Coreference runs last, as before. No
    #: linker is shown what the earlier one linked.
    LINKERS = ("name", "coreference")

    # ── Resource bounds ──────────────────────────────────────────────────────
    # These cap prompt size and call count. No decision rule reads them: changing
    # one changes how much text a judge sees, never what counts as a link. Every
    # window is the same width on purpose -- the earlier per-step values (2, 3, 4,
    # 5) implied a calibration that was never measured.
    CONTEXT_SENTENCES = 5          # sentences either side shown to any judge
    ANCHOR_LIMIT = 5               # naming sentences offered as evidence
    JUDGE_BATCH = 25               # candidates per judging call (all judges)
    COREFERENCE_BATCH = 10         # sentences per coreference-resolution call
    ASK_ATTEMPTS = 2               # initial call + one retry on an empty parse

    # ══ HEAD DELTA 1 (s_linker92a) ══════════════════════════════════════════
    #: Whether a name written inside a longer dotted identifier is skipped here.
    #: False in this variant: `QUALIFIED_CLAUSE` states the same thing to the judge
    #: that reads every one of these cases, and `s_linker92b` is the arm that asks
    #: whether saying it twice is worth the pairs.
    SKIP_QUALIFIED = False

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
        self.llm = TracingLLMClient(real_llm, self._llm_calls)
        self.no_knowledge = no_knowledge
        self.doc_knowledge: DocumentKnowledge | None = None
        self._phase_log: list[dict] = []
        self._phase_metrics: dict[str, dict] = {}
        self.workflow: list[dict] = []
        print("SLinker120 (name scan -> one evidence-graded judge -> coreference;"
              " one rule, per-case antecedent shortlist)")
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
        """Dispatch. No linker receives the links the earlier one produced.

        Two entries, not three: the full-name and partial-name scans propose into
        one stream judged by one call, and what used to be the order between them is
        now a fact in the case (`naming`). Whatever coreference re-proposes, `_union`
        merges by pair, and the merge is decided by `_stage_of`, not by order.
        """
        if linker == "name":
            return self._run_name_linker(
                sentences, components, name_to_id, sent_map)
        if linker == "coreference":
            return self._run_coreference_linker(
                sentences, components, name_to_id, sent_map)
        raise RuntimeError(f"unknown linker: {linker!r}")

    # ── Concurrency and small helpers ────────────────────────────────────────

    @staticmethod
    def _iter_batches(items, n):
        """Yield (batch_num, batch_slice) — batch_num is 1-indexed."""
        return iter_batches(items, n)

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
        return link_view(links, sent_map)

    @staticmethod
    def _decision_view(decisions):
        return decision_view(decisions)

    @staticmethod
    def _linker_feedback(feedback):
        """Reduce detailed linker evidence to accepted/rejected references."""
        return linker_feedback(feedback)

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
    def _prompt_coref_validation(comp_names, cases, focus) -> str:
        """The coreference judging prompt. The head's strict rubric, byte for byte.

        The ancestor built this and the full-name judge's prompt from one function
        with a ``strict`` flag, because it had two judging rubrics to select between.
        This variant has one name rule and one coreference rule, and they are not
        variants of each other: the name cases carry a surface the code matched and
        the coreference cases carry a resolution the model committed to. So the flag
        is gone and this is what it built when it was set — `pilot/test_s120_
        standalone.py` checks the two byte for byte over recorded cases.

        The ground against the link is asked here and nowhere else. "Approve by
        default" and "state the strongest ground for rejecting" are contradictory
        standards to put in one prompt, and only the strict arm was measured.
        """
        return f"""Validate components in a document.{f" {focus}" if focus else ""}

COMPONENTS: {', '.join(comp_names)}

{LAYERED_COREF_RULES}

For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim), then state the strongest ground there is for rejecting this case under the
rules above (or "none" if there is none), then decide: approve unless that ground is one
the rules above make decisive. An objection you could raise against most sentences is not
a ground for rejecting this one.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>", "objection": "<strongest ground to reject, or none>", "approve": true}}]}}
JSON only:"""

    # ══ HEAD DELTA 3 (s_linker110) ══════════════════════════════════════════
    def _named_before(self, comp_names, sentence_table, target):
        """Components the table names strictly before ``target``, latest first.

        Exact, not heuristic: the same name relation the rest of the module reads
        names with, applied to the sentences the case was already shown.
        """
        latest: dict[str, int] = {}
        for row in sentence_table:
            number = row.get("sentence")
            if not isinstance(number, int) or number >= target:
                continue
            for name in comp_names:
                if self._states_a_name(row.get("text", ""), name):
                    latest[name] = max(latest.get(name, 0), number)
        return sorted(latest.items(), key=lambda item: -item[1])

    def _prompt_coref(self, comp_names, sentence_table, targets) -> str:
        blocks = []
        for target in targets:
            near = self._named_before(comp_names, sentence_table, target["target"])
            listed = (", ".join(f"{name} (S{number})" for name, number in near)
                      if near else "none")
            blocks.append(
                f"--- Case {target['case']} ---\n"
                f"TARGET S{target['target']}: {target['text']}\n"
                f"NAMED BEFORE THIS CASE: {listed}"
            )
        return f"""Resolve references (pronouns and noun phrases that refer back) to components.

COMPONENTS: {', '.join(comp_names)}

SENTENCES (the document text the cases are drawn from)
{json.dumps(sentence_table)}

For each TARGET sentence below, identify any pronoun or noun phrase in THAT sentence
that refers back to a component listed above. Read the TARGET's context in SENTENCES.
If a target sentence has no such reference to a listed component, return no resolution
for it. Be conservative — only include resolutions you are CERTAIN about.

Each case lists NAMED BEFORE THIS CASE: the components the sentences above it
actually name, with the sentence that names each, nearest first. That list has
already been checked against the document, so it is where the antecedent will be if
there is one. Quote the referring expression first, then say which entries of that
list could be what it points to, then name the one it does point to.

{chr(10).join(blocks)}

{COREF_RULES}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "reference": "the server", "candidates": ["Name", "OtherName"], "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

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

        `ASK_ATTEMPTS` is this variant's declared bound and stays above; the retry
        rule and the three success predicates are `linker_infra.ask_json`.
        """
        return ask_json(
            self.llm, prompt,
            attempts=self.ASK_ATTEMPTS, timeout=timeout, label=label,
            phase=phase, require=require, require_present=require_present,
        )


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


    # ══ HEAD DELTA 1 (s_linker92a) ══════════════════════════════════════════
    # ── the proposer ─────────────────────────────────────────────────────────

    def _named_spans(self, text, name):
        """Spans of ``text`` that write ``name`` whole, at this variant's fidelity.

        The one place a subclass changes the relation point. `s_linker92c` and
        `s_linker92d` override it; nothing else in the family does.
        """
        return self._name_spans(text, name, NameForm.ANY_CASE)

    def _writes_name(self, text, name):
        """The surface ``text`` uses to write ``name``, or "" if it does not.

        The first span wins, which is `_find_exact_form`'s rule and the one the
        recorded `matched_text` of every earlier full-name candidate follows.
        """
        for start, end in self._named_spans(text, name):
            if self.SKIP_QUALIFIED and self._in_dotted_path(text, start, end):
                continue
            return text[start:end]
        return ""

    def _extract_named_mentions(self, sentences, components, name_to_id, sent_map):
        """Every pair whose sentence writes a name of the component. No call.

        Signature and return type are the head's — a dict keyed
        (sentence, component_id) of `CandidateLink` — so `_run_full_name_linker`,
        the evidence bundles and the judge are reached unchanged.
        """
        by_component = self._names_by_component()
        candidates: dict = {}
        for sentence in sentences:
            for component in components:
                for name in (component.name, *by_component.get(component.name, ())):
                    surface = self._writes_name(sentence.text, name)
                    if not surface:
                        continue
                    candidates[(sentence.number, component.id)] = CandidateLink(
                        sentence.number, sentence.text, component.name,
                        component.id, surface, source="full_name",
                    )
                    break
        print(f"    Extracted: {len(candidates)} (scan, 0 calls)")
        return candidates

    def _scan_all(self, sentences, components):
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

        Called only by `_scan`, which is this loop plus HEAD DELTA 2's refusal.
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

    # ══ HEAD DELTA 2 (s_linker109) ══════════════════════════════════════════
    def _covering_names(self, text, exclude, components):
        """Spans where ``text`` writes some *other* component's **catalog** name.

        `_name_spans` at `ANY_CASE` is the whole-name row of the relation — the row
        `_states_a_name` reads and the row `s_linker92a` scans — so this introduces no
        fidelity the module does not already implement.

        **Discovered aliases are deliberately not consulted here, and that asymmetry
        is the design.** The module's scans use N(c) = the catalog name *and* the
        run's aliases, because a scan only ever *admits* a case for a judge, and the
        branch's law is that nothing in the deterministic layer admits a link. This
        predicate is the one thing in the layer that *ends* a case, so it may rest
        only on what is given: a catalog name is an input, while the alias table is
        the output of an LLM stage that varies by ~2.8 terms a run. Letting it veto
        makes one stage's sampling a silent refusal in another's — and it does, if
        allowed: the alias form of this predicate costs **3 gold links in one recorded
        luna run**, all three where the run's table bound a document term to the
        sibling of the component the gold names.
        """
        covering = []
        for component in components:
            if component.name == exclude:
                continue
            covering.extend(
                self._name_spans(text, component.name, NameForm.ANY_CASE))
        return covering

    def _only_inside_another_name(self, text, name, components) -> bool:
        """True when every writing of ``name``'s word lies inside another whole name.

        *Every* one, not any: a sentence that also writes the word on its own has said
        something about this component somewhere else in it, and that pair is a case
        for the judge as before. The predicate is the reason for the refusal stated
        exactly — it does not fire on a component that merely has a sibling.
        """
        mine = self._name_spans(text, name, NameForm.ANY_WORD)
        if not mine:
            return False
        covering = self._covering_names(text, name, components)
        if not covering:
            return False
        return all(any(start <= a and b <= end for start, end in covering)
                   for a, b in mine)

    def _scan(self, sentences, components):
        """`_scan_all`, minus the pairs another component's name covers.

        The loop is the head's, called and filtered rather than restated: a restated
        candidate loop is where this line's one real bug hid.
        """
        kept, refused = [], 0
        for candidate in self._scan_all(sentences, components):
            if self._only_inside_another_name(
                    candidate.sentence_text, candidate.component_name, components):
                refused += 1
                continue
            kept.append(candidate)
        if refused:
            print(f"    Partial-name scan refused {refused} "
                  f"(word written only inside another component's name)")
        return kept

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


    # ── Mention labels ────────────────────────────────────────────────────────

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


    # ═════════════════════════════════════════════════════════════════════════
    # Linker 1 — NAME: both scans, one stream, one rule, one judging pass.
    # ═════════════════════════════════════════════════════════════════════════

    # ── The union judge's declarations ───────────────────────────────────────

    #: The iteration this instance runs. `None` is this file's own rule; any other
    #: name is fetched from `union_iterations`, which is how `pilot/union_pilots.py`
    #: holds two versions of the judge in one invocation. Set on the instance, so
    #: one process can hold two arms.
    iteration_name: str | None = None

    @property
    def iteration(self) -> RuleSpec:
        """The rule, demand, reply contract and case format this run judges with."""
        name = self.iteration_name or os.environ.get("UNION_ITERATION")
        if not name or name == ACTIVE_ITERATION:
            return UNION_V13
        # Only an experiment reaches here: the trail file imports the ancestor.
        from llm_sad_sam.linkers.experimental.union_iterations import ITERATIONS
        return ITERATIONS[name]

    #: The contract a call whose cases carry no component answers in. Both lines are
    #: the head's own denotation prompt: there is nothing to approve against, so the
    #: call classifies, and `s_linker119` measured what happens when that stream is
    #: made to answer the other contract instead (net -9.0 terra / -16.0 luna).
    DENOTATION_DEMAND = (
        "For each case, quote as the claim a contiguous exact substring of the source "
        "sentence, then answer denotation with participant or associated."
    )
    DENOTATION_REPLY = ('{"validations": [{"case": 1, "claim": "exact source quote", '
                        '"denotation": "participant"}]}')

    #: How the evidence line says what the sentence writes of the name. These are the
    #: three values of one code fact (`_states_a_name` decomposed), phrased as the
    #: evidence they are rather than as the name of a rule to apply.
    WRITES = {
        "whole name": "the whole name",
        "alias": "a short form the document established for it",
        "word only": "one word of the name",
    }

    # ── the merged stream ────────────────────────────────────────────────────

    def _name_candidates(self, sentences, components, name_to_id, sent_map):
        """Both scans, unchanged, merged by pair. The whole-name scan wins a tie.

        There is no tie by construction — `_scan_all` skips a pair whose sentence
        states a whole name — and the rule is stated anyway so the merge does not
        depend on that property holding in a fork.
        """
        merged = dict(self._extract_named_mentions(
            sentences, components, name_to_id, sent_map))
        for candidate in self._scan(sentences, components):
            merged.setdefault(
                (candidate.sentence_number, candidate.component_id), candidate)
        return [merged[key] for key in sorted(merged)]

    @staticmethod
    def _stage_of(candidate):
        """The stage label the head would have recorded for this candidate.

        `_scan` marks its candidates `partial_name_candidate`; the head relabels at
        the link, and so does this variant, so every downstream view — the links CSV,
        the phase log, the RQ3/RQ4 attribution — reads the two stages it always read.
        """
        return "full_name" if candidate.source == "full_name" else "partial_name"

    def _union_evidence(self, candidate, components, sent_map):
        """Every fact of the match this case carries. No weighing lives here.

        `naming` is `_states_a_name` decomposed into which of N(c) matched;
        `alternatives` is the same relation asked of every other component, which is
        `s_linker107`'s enumeration moved from the resolver to the name streams;
        `last_named` is the recency the anchors were being read for.
        """
        text = candidate.sentence_text
        name = candidate.component_name
        whole = self._writes_name(text, name)
        alias = ""
        if not whole:
            for term, owner in getattr(
                    getattr(self, "doc_knowledge", None), "aliases", {}).items():
                if owner == name and self._find_exact_form(text, term):
                    alias = term
                    break
        naming = ("whole name" if whole else "alias" if alias else "word only")
        mine = {text[start:end].casefold() for start, end
                in self._name_spans(text, name, NameForm.ANY_WORD)}
        alternatives = []
        for other in components:
            if other.name == name:
                continue
            spans = self._name_spans(text, other.name, NameForm.ANY_WORD)
            if spans and {text[s:e].casefold() for s, e in spans} & mine:
                alternatives.append(other.name)
        anchors, last_named = [], -1
        for sentence in sorted(sent_map.values(), key=lambda s: s.number):
            if sentence.number == candidate.sentence_number:
                continue
            if self._find_exact_form(sentence.text, name):
                if sentence.number < candidate.sentence_number:
                    last_named = candidate.sentence_number - sentence.number
                if len(anchors) < self.ANCHOR_LIMIT:
                    anchors.append(f"S{sentence.number}: {sentence.text}")
        return {
            "source": self._stage_of(candidate),
            "span": candidate.matched_text or name,
            "naming": naming,
            "mention": self._retained_mention_label(name, text),
            "alternatives": alternatives,
            "last_named": last_named,
            "anchors": anchors,
        }

    # ── the one judging call ─────────────────────────────────────────────────

    def _prompt_union(self, comp_names, sentence_table, cases, named=True) -> str:
        """The active iteration's rule, demand and reply contract, around the cases.

        Every version this round measured is a row of `union_iterations.ITERATIONS`;
        nothing about the prompt is written here, so the file that holds the trail is
        the file a reader compares versions in.
        """
        spec = self.iteration
        table = (f"\nSENTENCES\n{json.dumps(sentence_table)}\n"
                 if sentence_table else "")
        clauses = f"\n{spec.clauses}\n" if spec.clauses else ""
        blind_call = spec.contract_follows_batch and not named
        catalog = "" if blind_call else f"\nCOMPONENTS: {', '.join(comp_names)}\n"
        demand = (self.DENOTATION_DEMAND if blind_call else spec.demand)
        reply = (self.DENOTATION_REPLY if blind_call else spec.reply)
        return f"""Validate components in a document.
{catalog}
{spec.rule}
{clauses}{table}
{demand}

CASES:
{chr(10).join(cases)}

Return JSON:
{reply}
JSON only:"""

    def _format_union_case(self, index, candidate, evidence, sent_map, shown_in=0):
        """One case, in the active iteration's format.

        The component slot is filled by the evidence, not by the case's existence: an
        iteration with `blind_word_only` leaves it empty where the match wrote only one
        word of a name, because that match computed no component for this sentence.
        Measured: with the slot filled for every case, luna's word-only gold reads 12.3
        of 26 against a control's 21.7 (`union_iterations.ITERATIONS['v8'].measured`) —
        `s_linker25`'s refusal, reappearing on the model the branch reads second.
        """
        spec = self.iteration
        blind = spec.blind_word_only and evidence["naming"] == "word only"
        # Where this case names its component. A case that is not blind always names it
        # in the header; a blind one names it where its iteration says, or nowhere.
        carries = (getattr(spec, "word_only_component", "hidden") if blind
                   else "header")
        of_name = (f' of "{candidate.component_name}"' if carries == "evidence"
                   else "")
        previous = self._prev_prefix(candidate.sentence_number, sent_map)
        labels = {
            "source": lambda: f"source={evidence['source']}",
            "naming": lambda: f"naming={evidence['naming']}",
            "writes": lambda: f"writes={self.WRITES[evidence['naming']]}{of_name}",
            "mention": (lambda: f"mention={evidence['mention']}"
                        if evidence["mention"] else None),
            "alternatives": (lambda: "alternatives="
                             + ", ".join(evidence["alternatives"])
                             if evidence["alternatives"] else None),
            "last_named": (lambda: f"named {evidence['last_named']} sentences earlier"
                           if evidence["last_named"] >= 0 else None),
        }
        facts = []
        for name in spec.fields:
            if blind and name in ("alternatives", "mention", "last_named"):
                continue          # every one of these names a component
            rendered = labels[name]()
            if rendered:
                facts.append(rendered)
        lines = [
            (f'Case {index}: "{evidence["span"]}" -> {candidate.component_name}'
             if carries == "header" else f'Case {index}: "{evidence["span"]}"'),
            f'  {previous}"{candidate.sentence_text}"',
            f"  Evidence: {', '.join(facts)}",
        ]
        if evidence["anchors"] and (
                not blind or getattr(spec, "word_only_anchors", False)):
            if shown_in:
                lines.append(f"  Anchors (other sentences naming it): "
                             f"as shown in Case {shown_in}.")
            else:
                lines.append("  Anchors (other sentences naming it):")
                lines.extend(f"    {anchor}" for anchor in evidence["anchors"])
        return "\n".join(lines)

    def _judge_union(self, candidates, components, sentences, sent_map):
        """One pass over the merged stream. The head's batching and parser.

        The word-only rows are shown the window the denotation judge shows them today,
        as one table per call rather than one per stream, so no case is shown less.
        """
        if not candidates:
            return [], {}
        from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names
        comp_names = get_comp_names(components)
        approved, decisions = [], {}
        # The evidence groups the cases as well as filling them: a case whose match
        # computed no component is judged among its own kind. One rule, one prompt
        # template, and the same call count the head pays for its two stages.
        if self.iteration.batch_by_evidence:
            named, blind = [], []
            for candidate in candidates:
                bucket = (blind if self._union_evidence(
                    candidate, components, sent_map)["naming"] == "word only"
                    else named)
                bucket.append(candidate)
            groups = [group for group in (named, blind) if group]
        else:
            groups = [candidates]
        batches = [batch for group in groups
                   for _, batch in self._iter_batches(group, self.JUDGE_BATCH)]
        for batch in batches:
            evidences = {
                (c.sentence_number, c.component_id):
                    self._union_evidence(c, components, sent_map)
                for c in batch
            }
            window = set()
            for candidate in batch:
                if evidences[(candidate.sentence_number,
                              candidate.component_id)]["naming"] == "word only":
                    window.update(s.number for s in
                                  self._window(candidate.sentence_number, sentences))
            table = [{"sentence": n, "text": sent_map[n].text}
                     for n in sorted(window) if n in sent_map]
            cases, shown = [], {}
            for index, candidate in enumerate(batch, 1):
                evidence = evidences[(candidate.sentence_number,
                                      candidate.component_id)]
                first = shown.get(candidate.component_name, 0)
                if evidence["anchors"] and not first:
                    shown[candidate.component_name] = index
                cases.append(self._format_union_case(
                    index, candidate, evidence, sent_map, first))
            named_batch = any(
                evidences[(c.sentence_number, c.component_id)]["naming"] != "word only"
                for c in batch)
            self.llm.set_phase("phase_25_name_union_judge")
            data = self._ask(
                self._prompt_union(comp_names, table, cases, named=named_batch),
                timeout=120, label="Union validation", require="validations",
            )
            verdicts = {}
            for item in (data or {}).get("validations", []):
                position = item.get("case", 0) - 1
                if not 0 <= position < len(batch):
                    continue
                candidate = batch[position]
                row = evidences[(candidate.sentence_number,
                                 candidate.component_id)]["naming"]
                claim = str(item.get("claim", "")).strip().strip("\"'\u201c\u201d\u2018\u2019")
                blind_call = (self.iteration.contract_follows_batch
                              and not named_batch)
                if blind_call or (self.iteration.verdict == "per_row"
                                  and row == "word only"):
                    # The head's denotation contract, unchanged: the enum keeps only a
                    # positive classification and the quote must be committed to.
                    keep = (str(item.get("denotation", "")).strip() == "participant"
                            and bool(claim))
                else:
                    value = item.get("approve", False)
                    keep = (value is True
                            or (isinstance(value, str) and value.lower() == "true"))
                verdicts[position] = (keep, claim)
            for position, candidate in enumerate(batch):
                ok, claim = verdicts.get(position, (False, ""))
                stage = self._stage_of(candidate)
                decisions[(candidate.sentence_number, candidate.component_id)] = {
                    "approved": ok,
                    "claim": claim,
                    "naming": evidences[(candidate.sentence_number,
                                         candidate.component_id)]["naming"],
                    "path": f"{stage}_judged" if ok else f"{stage}_rejected",
                    "stage": "name_union_judge",
                }
                if ok:
                    approved.append(candidate)
        return approved, decisions

    def _run_name_linker(self, sentences, components, name_to_id, sent_map):
        candidates = self._name_candidates(
            sentences, components, name_to_id, sent_map)
        approved, decisions = self._judge_union(
            candidates, components, sentences, sent_map)
        links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name,
                       source=self._stage_of(c))
            for c in approved
        ]
        return links, {
            "candidates": self._link_view(
                [SadSamLink(c.sentence_number, c.component_id, c.component_name,
                            source=f"{self._stage_of(c)}_candidate")
                 for c in candidates],
                sent_map,
            ),
            "accepted": self._link_view(links, sent_map),
            "judge_decisions": self._decision_view(decisions),
        }

    # ═════════════════════════════════════════════════════════════════════════
    # Linker 2 — COREFERENCE: the head's resolver and its strict gate.
    # ═════════════════════════════════════════════════════════════════════════

    def _run_validation_pass(self, comp_names, cases, focus, phase_tag=None):
        """One coreference judging call, and the verdicts it answered.

        The ancestor's parser, minus the ``strict`` argument it no longer has to
        pass: the name streams parse their own reply in `_judge_union`, where the
        contract depends on what the batch's evidence carries.
        """
        if phase_tag:
            self.llm.set_phase(phase_tag)
        data = self._ask(
            self._prompt_coref_validation(comp_names, cases, focus),
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
                phase_tag="phase_25_coreference_judge",
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
        return backend_tag(self.llm)

    def _checkpoint_dir(self, text_path):
        return checkpoint_dir(text_path, self._VARIANT_NAME, self._backend_tag())

    def _save_phase(self, text_path, phase_name, state):
        save_phase_state(self._checkpoint_dir(text_path), phase_name, state)

    def _log(self, phase, input_summary, output_summary, links=None):
        self._phase_log.append(
            log_entry(phase, input_summary, output_summary, links))

    def _save_log(self, text_path):
        write_run_logs(text_path, self._VARIANT_NAME, self._backend_tag(),
                       self._phase_log, self._llm_calls)

    def _compute_phase_metrics(self) -> dict:
        return phase_metrics(self._llm_calls)
