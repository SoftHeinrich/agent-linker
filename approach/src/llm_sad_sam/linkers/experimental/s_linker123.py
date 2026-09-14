"""S-Linker123 — the antecedent shortlist says what the system already decided about it.

`s_linker122` gives the coreference resolver a per-case shortlist of the components the
sentences above it name:

    NAMED BEFORE THIS CASE: kurento (S68), WebRTC-SFU (S68), FreeSWITCH (S66)

`_named_before` computes that list from `_states_a_name` — a purely **lexical** fact:
which sentences write a name. By the time the resolver runs, the union judge has already
ruled on every one of those mentions, and the module was throwing the verdicts away and
offering all of them as equals. This variant carries them:

    NAMED BEFORE THIS CASE: kurento (S68, linked), WebRTC-SFU (S68, named only),
                            FreeSWITCH (S66, linked)

`linked` is a mention the judge kept; `named only` is one it saw and did not keep. No
rule in the prompt speaks about the mark, and that is deliberate — see below.

**The design argument, which is the reason this file exists.** The shortlist is the one
place in the module where a stage is shown a fact the pipeline has already refined and
is shown the *unrefined* version of it. Every other piece of evidence any judge reads is
the best the system knows at that point. This is not a new signal, a new call or a new
rule: it is the same fact at the resolution the system already has it at. One fact
source, stated once, read everywhere.

**What it is measured at, stated without rounding up.** Level 2, three samples a side,
both arms in one invocation per model, the alias table and the name-link set pinned from
a recorded run so only the resolver is resampled
(`pilot/coref_annot_pilots.py`, `../results/coref_annot_{terra,luna}_20260914`):

    terra   macro F2 +0.31 (p 0.50)   macro F1 +0.04 (p 1.00)   TP +0.67   FP +0.67
    luna    macro F2 +0.04 (p 1.00)   macro F1 -0.06 (p 1.00)   TP +0.67   FP +0.67

**QUALITY-NEUTRAL on both models with the F2 point estimate favourable on both**, at the
same call count — the standard `s_linker86`, `s_linker89` and `s_linker110`-on-luna were
adopted under. It is **not** a precision result and the file does not claim one: the
exchange rate is **+0.67 gold and +0.67 spurious a run on both models**, and F2's 4:1
recall weighting is what turns a one-for-one trade into a positive number. Nothing here
reaches the n = 3 sign-flip floor of p = 0.25; settling the terra F2 estimate would take
~13 paired samples and the luna one ~690, and the round did not buy them.

**Why the mark alone, and no clause about it.** Two arms were built. `annot_clause` adds
one sentence — "a `named only` entry is the weaker antecedent" — and it is the *worse*
of the two on both models (terra F2 -0.19, luna -0.16), because the two arms move the
resolver 32 pairs apart in opposite directions and only the unweighted one lands on the
favourable side. The design law says where a weighing belongs when you want one; it does
not say you want one. **Here the fact is enough and the weighing costs.**

**What was refused, and it is the same refusal `s_linker109` records.** A third arm,
`annot_only`, drops the refused entries from the list instead of marking them. It is the
only arm on either model that moves anything, and it moves the wrong way: luna TP -1.0,
FP +4.0, **macro F2 -0.92**, worse in three samples of three. It proposes 36 fewer pairs
a run and its net spurious *doubles* — **+17 false positives added against 5 removed** —
because taking an entry off the list does not make the resolver abstain, it makes the
resolver attach the same referring expression to the next component down. A name verdict
is a *discovered* fact, resampled every run; it may open a case and it may not close one.
Marking is opening; withholding is closing. **This file marks and does not withhold.**

**The blindness this does and does not break.** `s_linker100` conditioned the second
proposer on the first's *output list* and added zero pairs in two of three samples. This
does not do that: the resolver still reads every sentence, still proposes independently,
and is never told which pairs to produce. What it receives is a property of each
candidate *antecedent*, which is evidence about a case and not a proposal to copy. The
two proposal stages stay blind to each other's link sets.

**Ground.** The mark is a fact the system computed, carried verbatim; it names no surface
form, no component and no document shape (GATE-06, GATE-07). Authored rule text is
**unchanged from `s_linker122`** — the mark is rendered in `_named_before`, not written
into any constant — so the defensibility accounting does not move.

Everything else is `s_linker122`'s text, method for method: the scan, the name relation,
the merged stream, the union rule, the reply contract, the coreference rubric and its
judge. `pilot/test_s123_standalone.py` checks that rather than asserting it.

Measurements, and the round that arrived at this file, are in
`../results/coref_annot_round/README.md`. None of them live here.
"""
from __future__ import annotations

import json
import os
import re
import time
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
# (`pilot/prompt_audit.py` sized each generalization off six recorded runs).
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
#: component listed above"). Deleting that preamble whole costs TP -16.2, because
#: it is also the input-format contract -- which block is the TARGET, that a target
#: with no referring expression yields nothing -- and this cut is the untried other
#: half: the contract stays, the restatement goes. 163 B leave each of the 40 resolver
#: calls a five-project run makes, and cutting it is composed-neutral on both models.
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
            "this linker needs WordNet: python -m nltk.downloader wordnet"
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
# computed rather than retyped, so quotation is mechanical and a drift in a rule
# constant is a drift in the rule (`pilot/s121_defensibility.py` checks each slice
# against the constant it was computed from).
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

#: The input contract: what a case contains. Not a criterion — the shape of the
#: input, and every case has the same shape.
_FORMAT = ("Every case gives you the expression the sentence uses, the sentence "
           "itself, the evidence the document supplies, and the component whose "
           "name that expression reaches.")

#: One line per evidence field, each saying what the field is evidence *of*. The
#: `writes` line ends in the denotation question, which is the reading a case is left
#: with when the sentence writes one word of a name rather than the name.
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
    "sentence."
)

#: The rule, in one piece: what a trace link is, then what each evidence field is
#: evidence of, then the two grounds for rejecting. `STRICTER_CLAUSE` carries no scope
#: guard here — it is about an ordinary word coinciding with a component's name, and
#: every case names a component, so it has a subject in all of them.
TRACE_LINK_RULE = f"""{_DEFINITION}{MENTION_COUNTS}

{_FORMAT} The evidence says what the expression is doing here; none of it is a verdict.

{_WRITES_LINE}
{_FIELD_LINES}

{POSITIVE_GROUND}

{STRICTER_CLAUSE}

{QUALIFIED_CLAUSE} {ACTS_ON}"""

#: The quote-before-verdict demand, and the reply contract it names. Demanding a
#: committed quote is measurably worth its bytes; verifying the quote back against the
#: sentence voided nothing over six runs, so it is demanded and not re-checked.
#: The one weighing that replaces the fact, and the only line this variant adds to
#: the ancestor's rule. It is SCOPED, and the scope is the whole of it: an unscoped
#: version of this sentence ("That a surface can name this component is not evidence
#: that it does here") was measured and reached the whole-name row, where it contradicts
#: `MENTION_COUNTS` -- on luna it cost 2.07 gold a unit there, and end to end it cost
#: seven gold links on one bare enumeration of component names. Scoped to the rows where
#: the sentence does not write the name in full, the two cannot meet: a whole-name case
#: is out of its reach, and the alias row the anchors were actually holding is not.
#:
#: Ground: general -- use versus mention, which holds for any text and names no surface
#: form, no component and no document shape (GATE-06/07).
SURFACE_NOT_EVIDENCE = (
    "Where the sentence does not write the name in full, that a surface can name this "
    "component is not evidence that it does here."
)

UNION_DEMAND = ('For each case, first quote the EXACT words from the sentence the '
                'verdict rests on -- the words that state the architectural claim '
                'about the component, or "none" if the sentence makes no such claim '
                '-- then decide approve true/false based on that quote.')

UNION_REPLY = ('{"validations": [{"case": 1, "claim": "<exact quote or none>", '
               '"approve": true}]}')


#: Which evidence fields a case may print, in the order it prints them. A field the
#: match did not compute is not printed; a field not named here is not printable.
UNION_FIELDS = ("writes", "alternatives", "mention")


class NameForm(Enum):
    """A point of the surface-realization relation, on two independent dimensions.

    *Fidelity* -- how exactly the sentence's characters must reproduce the name:

        ANY_CASE   the name, ignoring case

    *Extent* -- how much of the name has to be present:

        ANY_WORD   one word of the name, under an English inflectional ending

    These two points are the whole relation: ANY_CASE is the name test every stage
    shares, and ANY_WORD is what the partial-name scan looks for. Fidelity below
    case-folding and extent between one word and the whole name were both measured
    and neither earned a member.
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
class SLinker123:
    """Two linkers, one rule over both name streams, no controller. Standalone.

    The two name scans are merged by pair and judged in one pass against one rule —
    what a trace link is, and how to read the evidence the match computed. The
    coreference linker runs behind them against its own rule. There is no linker base
    class and no controller: the module docstring says what the whole file holds.
    """

    _VARIANT_NAME = "s_linker123"

    #: Execution order. The two name scans are one stage: they propose into one
    #: stream, one judge reads it, and each link carries the label of the scan that
    #: proposed it (`_stage_of`). Coreference runs last, and it is shown what the name
    #: stage kept -- not as a list of pairs to produce, but as the verdict already
    #: reached about each candidate antecedent on its shortlist (`_mark_shortlist`).
    #: No stage is told which links to make; the proposal stages remain blind to each
    #: other's link sets, which is the property `s_linker100` measured.
    LINKERS = ("name", "coreference")

    # ── Resource bounds ──────────────────────────────────────────────────────
    # These cap prompt size and call count. No decision rule reads them: changing
    # one changes how much text a judge sees, never what counts as a link. Every
    # window is the same width on purpose -- the earlier per-step values (2, 3, 4,
    # 5) implied a calibration that was never measured.
    CONTEXT_SENTENCES = 5          # sentences either side shown to any judge
    JUDGE_BATCH = 25               # candidates per judging call (all judges)
    COREFERENCE_BATCH = 10         # sentences per coreference-resolution call
    ASK_ATTEMPTS = 2               # initial call + one retry on an empty parse

    #: Whether a name written inside a longer dotted identifier is skipped by the
    #: scan. False: `QUALIFIED_CLAUSE` states the same thing to the judge that reads
    #: every one of these cases, and a deterministic layer that only ever admits a
    #: case for a judge should not also refuse one. Since the nesting refusal came out
    #: of `_scan` that second clause is **literally true of this module**: no predicate
    #: anywhere in the deterministic layer ends a case.
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
        print("SLinker123 (name scan -> one evidence-graded judge -> coreference;"
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
                linker, sentences, components, name_to_id, sent_map,
                # What is known so far, by (sentence, component name). The coreference
                # linker marks its antecedent shortlist with it; the name linker runs
                # first and is passed an empty set by construction.
                {(link.sentence_number, link.component_name) for link in current},
            )
            # Merge by pair; an earlier linker wins a tie. `_stage_of` decides the
            # label a kept pair carries, so the merge does not depend on order.
            seen = {(link.sentence_number, link.component_id) for link in current}
            for link in produced:
                key = (link.sentence_number, link.component_id)
                if key not in seen:
                    current.append(link)
                    seen.add(key)
            history.append({
                "linker": linker,
                "feedback": linker_feedback(feedback),
            })
            self._save_phase(text_path, f"linker_{linker}", {
                "links": produced, "feedback": feedback, "workflow": history,
            })

        self.workflow = history
        self._phase_metrics = phase_metrics(self._llm_calls)
        self._phase_log.append(log_entry(
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
        ))
        self._save_phase(text_path, "final", {
            "final": current,
            "workflow": history,
            "elapsed_s": round(time.time() - started, 2),
        })
        write_run_logs(text_path, self._VARIANT_NAME, backend_tag(self.llm),
                       self._phase_log, self._llm_calls)
        print(f"\nFinal: {len(current)} links "
              f"({time.time() - started:.1f}s, {len(self._llm_calls)} LLM calls)")
        return current

    def _run_linker(self, linker, sentences, components, name_to_id, sent_map,
                    linked=()):
        """Dispatch. No linker is told which pairs to produce; one is told what is known.

        Two entries, not three: the full-name and partial-name scans propose into
        one stream judged by one call, and what used to be the order between them is
        now a fact in the case (`naming`). Whatever coreference re-proposes, `link`
        merges by pair, and the label is decided by `_stage_of`, not by order.

        `linked` is what the linkers before this one kept, and only the coreference
        linker reads it — as a property of each candidate ANTECEDENT on its shortlist,
        never as a list of pairs to produce. The proposal stages stay blind to each
        other's link sets, which is the property `s_linker100` measured and this does
        not spend (`../results/coref_annot_round/README.md`).
        """
        if linker == "name":
            return self._run_name_linker(
                sentences, components, name_to_id, sent_map)
        if linker == "coreference":
            return self._run_coreference_linker(
                sentences, components, name_to_id, sent_map, linked)
        raise RuntimeError(f"unknown linker: {linker!r}")

    # ── Concurrency and small helpers ────────────────────────────────────────

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

        One predicate for a question that was once asked with three copies of the
        same expression. The live caller is the partial-name scan's whole-name
        exclusion. The mention-label classifier asks the same question decomposed,
        because it must know *which* name matched.
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

        The reply pairs a term with the component it was claimed for, because asking
        for bare terms leaves a term two components both claimed undecidable and the
        caller keeping whichever the extractor recorded last.
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
        """The coreference judging prompt, and the only rubric it has.

        The name cases and the coreference cases are not variants of one question:
        a name case carries a surface the code matched, and a coreference case
        carries a resolution the model committed to. So there is no flag selecting
        between two rubrics here — this prompt is built one way, and the name
        stream's prompt is built in `_prompt_union`.

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

    #: What an entry's mark says: the verdict the union judge reached about that very
    #: mention. Named after the verdict and not after a quality ("strong"/"weak") so the
    #: entry stays a statement of fact — whether a `named only` antecedent is worth less
    #: is a weighing, and the round measured that adding it costs (HEAD DELTA, below).
    MARK_LINKED = "linked"
    MARK_NAMED_ONLY = "named only"

    def _mark_shortlist(self, near, linked):
        """HEAD DELTA — the shortlist entries, each carrying its own mention's verdict.

        `_named_before` returns (name, sentence) for a mention the document writes;
        `linked` holds the (sentence, name) pairs the linkers before this one kept. An
        entry is `linked` when the judge kept that mention and `named only` when it saw
        it and did not. The sentence number is what carries the mark, so the name stays
        the catalog's own string and the reply contract is untouched.

        With `linked` empty every entry reads `named only`, which is why the caller
        passes what `link` has accumulated rather than a default: a resolver run with no
        name stage in front of it would otherwise be told the document names nothing.
        """
        return [(name, f"{number}, "
                 f"{self.MARK_LINKED if (number, name) in linked else self.MARK_NAMED_ONLY}")
                for name, number in near]

    def _prompt_coref(self, comp_names, sentence_table, targets, linked=()) -> str:
        blocks = []
        for target in targets:
            near = self._mark_shortlist(
                self._named_before(comp_names, sentence_table, target["target"]),
                linked)
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

        Testing the judge's reply for truthiness makes one event -- "the judge did
        not answer" -- come out three ways: an unparseable reply approves *every*
        proposal, a parsed reply with no ``approved`` key approves none, and a genuine
        empty approval list is honoured only after a wasted retry. Here an empty list
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


    # ── the proposer ─────────────────────────────────────────────────────────

    def _writes_name(self, text, name):
        """The surface ``text`` uses to write ``name``, or "" if it does not.

        The first span wins, which is `_find_exact_form`'s rule and the one the
        recorded `matched_text` of every earlier full-name candidate follows.
        """
        for start, end in self._name_spans(text, name, NameForm.ANY_CASE):
            if self.SKIP_QUALIFIED and self._in_dotted_path(text, start, end):
                continue
            return text[start:end]
        return ""

    def _extract_named_mentions(self, sentences, components, name_to_id, sent_map):
        """Every pair whose sentence writes a name of the component. No call.

        Returns a dict keyed (sentence, component_id) of `CandidateLink`, which is
        what `_name_candidates` merges the partial-name scan into.
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

    def _scan(self, sentences, components):
        """Every (sentence, component) pair whose sentence writes one word of a name.

        Nothing here admits a link and **nothing here ends one either**: every pair the
        relation finds is a case for the judge. Later spans of the same pair overwrite
        earlier ones, so the recorded `matched_text` is the last surface found in the
        sentence.

        This is the whole deterministic layer of the one-word stream — the relation and
        the whole-name exclusion, and no third rule. `s_linker109`'s nesting refusal
        (`_only_inside_another_name`: a word written only inside some *other*
        component's name is that name's) stood here and is **removed as of
        `../results/s121_ablations/`**. It was never able to gain a link — over five
        projects it fires on one, drops 12 pairs and 0 of them are gold — and the judge
        rejects those pairs on its own: **140 of 144 case-samples across two models**,
        under `QUALIFIED_CLAUSE` and `STRICTER_CLAUSE`, which are about exactly this.
        What it cost was a judging call on the project it touched. Its remaining value
        was the 4 of 144 the laxer model approved, all of them non-gold, which is 1.3
        spurious links a run on that model and 0.0 on the other.
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

        The two branches were once two methods, verified identical to these over
        every (name, sentence) pair of all five projects.
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

    def _mention_label(self, comp_name: str, text: str) -> str:
        """What the code can say about the expression's place in the sentence.

        Returns "" wherever the judge is holding the fact already: everything outside
        `RETAINED_MENTION_TYPES` restates the case header, which the judge can read.

        The case distinction compares the matched surface against the name rather
        than running a second case-sensitive predicate: `_find_exact_form` already
        returns what it matched. Measured indistinguishable from the two-predicate
        form it replaces, at the stage and end to end.
        """
        def label(mention):
            return mention.value if mention in self.RETAINED_MENTION_TYPES else ""

        matched = self._find_exact_form(text, comp_name)
        if matched:
            # CODE_TOKEN iff EVERY writing of the name is inside a dotted path: a
            # name written once plainly and once qualified is still a plain mention.
            occurrences = list(re.finditer(
                rf"\b{re.escape(comp_name.lower())}\b", text))
            if occurrences and all(self._in_dotted_path(text, m.start(), m.end())
                                   for m in occurrences):
                return label(MentionType.CODE_TOKEN)
            return label(MentionType.PROPER_STANDALONE if matched == comp_name
                         else MentionType.LOWERCASE_PROSE)
        for alias in self._names_by_component().get(comp_name, ()):
            if self._find_exact_form(text, alias):
                return label(MentionType.VIA_ALIAS)
        return label(MentionType.INDIRECT)

    #: The mention labels the judge cannot re-derive from the sentence it is shown.
    #: Everything else `_classify_mention_typed` can say is a restatement of the case
    #: header, and dropping it was measured to cost 3-21% of those approvals.
    RETAINED_MENTION_TYPES = frozenset({
        MentionType.VIA_ALIAS,
        MentionType.CODE_TOKEN,
    })


    # ═════════════════════════════════════════════════════════════════════════
    # Linker 1 — NAME: both scans, one stream, one rule, one judging pass.
    # ═════════════════════════════════════════════════════════════════════════

    # ── The union judge's declarations ───────────────────────────────────────

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
        """Which scan proposed this candidate.

        `_scan` marks its candidates `partial_name_candidate`; the label is resolved
        at the link, so every downstream view — the links CSV, the phase log, the
        RQ3/RQ4 attribution — reads two name stages even though one judge read both.
        """
        return "full_name" if candidate.source == "full_name" else "partial_name"

    def _union_evidence(self, candidate, components, sent_map):
        """Every fact of the match this case carries. No weighing lives here.

        `naming` is `_states_a_name` decomposed into which of N(c) matched;
        `alternatives` is the same relation asked of every other component, which is
        the components a case could be reaching instead of the one it names.
        Every key here is printed by `_format_union_case` — a fact the case does not
        carry is a fact this method does not compute.
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
        return {
            "span": candidate.matched_text or name,
            "naming": naming,
            "mention": self._mention_label(name, text),
            "alternatives": alternatives,
        }

    # ── the one judging call ─────────────────────────────────────────────────

    def _prompt_union(self, comp_names, sentence_table, cases) -> str:
        """One call shape for every case: the rule, the demand, the cases, the reply.

        The sentence window is printed whenever some case in this call writes only one
        word of a name — what such an expression denotes is a question about its
        local context, so the context is printed.
        """
        table = (f"\nSENTENCES\n{json.dumps(sentence_table)}\n"
                 if sentence_table else "")
        return f"""Validate components in a document.

COMPONENTS: {', '.join(comp_names)}

{TRACE_LINK_RULE}
{table}
{SURFACE_NOT_EVIDENCE}

{UNION_DEMAND}

CASES:
{chr(10).join(cases)}

Return JSON:
{UNION_REPLY}
JSON only:"""

    def _format_union_case(self, index, candidate, evidence, sent_map):
        """One case: the span and its component, the sentence, and the evidence.

        Every case is this shape. What a whole-name case and a one-word case differ in
        is the *content* of the evidence fields — `writes=the whole name` against
        `writes=one word of the name`, and an `alternatives` list where the word is
        one several components' names carry. A field the match did not compute is not
        printed, which is the only reason two cases ever print different field names.
        """
        previous = self._prev_prefix(candidate.sentence_number, sent_map)
        facts = [f"writes={self.WRITES[evidence['naming']]}"]
        if evidence["alternatives"]:
            facts.append("alternatives=" + ", ".join(evidence["alternatives"]))
        if evidence["mention"]:
            facts.append(f"mention={evidence['mention']}")
        lines = [
            f'Case {index}: "{evidence["span"]}" -> {candidate.component_name}',
            f'  {previous}"{candidate.sentence_text}"',
            f"  Evidence: {', '.join(facts)}",
        ]
        return "\n".join(lines)

    def _judge_union(self, candidates, components, sentences, sent_map):
        """One pass over the merged stream. The head's batching and parser.

        Batches are by size, in candidate order, so every call holds whatever mix of
        naming the document gave it and nothing routes a case to a second call shape.
        """
        if not candidates:
            return [], {}
        from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names
        comp_names = get_comp_names(components)
        approved, decisions = [], {}
        for _, batch in iter_batches(candidates, self.JUDGE_BATCH):
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
            cases = []
            for index, candidate in enumerate(batch, 1):
                evidence = evidences[(candidate.sentence_number,
                                      candidate.component_id)]
                cases.append(self._format_union_case(
                    index, candidate, evidence, sent_map))
            self.llm.set_phase("phase_25_name_union_judge")
            data = self._ask(
                self._prompt_union(comp_names, table, cases),
                timeout=120, label="Union validation", require="validations",
            )
            verdicts = {}
            for item in (data or {}).get("validations", []):
                position = item.get("case", 0) - 1
                if not 0 <= position < len(batch):
                    continue
                claim = str(item.get("claim", "")).strip().strip("\"'\u201c\u201d\u2018\u2019")
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
            "candidates": link_view(
                [SadSamLink(c.sentence_number, c.component_id, c.component_name,
                            source=f"{self._stage_of(c)}_candidate")
                 for c in candidates],
                sent_map,
            ),
            "accepted": link_view(links, sent_map),
            "judge_decisions": decision_view(decisions),
        }

    # ═════════════════════════════════════════════════════════════════════════
    # Linker 2 — COREFERENCE: the resolver and its strict gate.
    # ═════════════════════════════════════════════════════════════════════════

    def _run_validation_pass(self, comp_names, cases, focus, phase_tag=None):
        """One coreference judging call, and the verdicts it answered.

        There is no ``strict`` argument to pass: this is the coreference rubric and
        the only one, and the name stream parses its own reply in `_judge_union`.
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

    def _run_coreference_linker(self, sentences, components, name_to_id, sent_map,
                                linked=()):
        resolved, metadata = self._resolve_references(
            sentences, components, name_to_id, sent_map, linked
        )
        raw = resolved
        approved, decisions = self._validate_coref_links(
            raw, sent_map, components, metadata)
        return approved, {
            "candidates": link_view(raw, sent_map),
            "accepted": link_view(approved, sent_map),
            "metadata": [
                {"sentence": sentence, "component_id": component, **value}
                for (sentence, component), value in metadata.items()
            ],
            "judge_decisions": decision_view(decisions),
        }

    def _resolve_references(self, sentences, components, name_to_id, sent_map,
                            linked=()):
        """Every sentence goes to the LLM in context; no pronoun regex.

        `linked` is carried through to `_prompt_coref`, which marks each shortlist
        entry with the verdict the union judge reached about that mention. It changes
        no control flow here: the same sentences are batched, the same windows are
        shown and the same resolutions are admitted.

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

        for batch_num, batch in iter_batches(sentences, self.COREFERENCE_BATCH):
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
                self._prompt_coref(comp_names, sentence_table, targets, linked),
                timeout=600,
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

        Shown only a sentence and a component name, this judge has to guess which
        expression was claimed to refer and to what, and it rejects about half the
        gold resolutions put to it. The resolver had already committed to both -- the
        referring expression and the quote it read as the antecedent -- and neither is
        recoverable from the case, so both are printed.
        """
        if not coref_links:
            return [], {}
        comp_names = get_comp_names(components)
        validated = []
        decisions: dict = {}
        self.llm.set_phase("phase_25_coreference_judge")
        for _, batch in iter_batches(coref_links, self.JUDGE_BATCH):
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

    def _save_phase(self, text_path, phase_name, state):
        """Write one phase's state under this run's checkpoint directory."""
        save_phase_state(
            checkpoint_dir(text_path, self._VARIANT_NAME, backend_tag(self.llm)),
            phase_name, state)

