"""S-Linker126 — greedy whole-name ownership, ambiguous residue discarded, no shortlist.

**Two changes over the union-judge design (`s_linker120`/`s_linker122`).**

*Name ownership.* A written whole catalog name owns every word it contains: the
partial-name scan does not separately propose a different component for a word that
sits entirely inside another component's whole name in the same sentence
(`_only_inside_another_name`, the catalog-only form `s_109`/`s_121` measured). Cases are
then grouped by `(sentence, surface)`. A group of one is judged as before; a group of
more than one — one surface reaching several components, none of them owning it whole —
is discarded outright and logged as `name_ambiguous_discard`, because the written
evidence does not identify a single component and there is no second judge to arbitrate
between them. The evidence's `competitors` field is therefore always empty and the rule
carries no clause about it.

*No antecedent shortlist; an antecedent-form gate instead.* The coreference resolver is
shown no per-case list of what has been named before it — no scan, no paragraph
endorsing one — and decides from the sentences alone. A resolution reaches its judge only
when the antecedent sentence it cited actually WRITES the component's name: the whole
name, or a document-established short form. A resolution whose citation only writes one
word of a name, or nothing the catalog recognizes at all, is rejected before judging and
logged as `antecedent_form_rejected` — the same accounting a judge's own rejection gets,
so an ablation that reconstructs "no validator" from `judge_decisions` alone still sees
these candidates. Both discards are FACTS about a case computed from given input (the
catalog, the document, the alias table), never a judge's verdict — a fact may close a
case; a discovered verdict may only open one.

**Why the shortlist went and the gate replaced it, in one line.** The list narrows what
is PROPOSED; the gate filters what was CITED — they are not interchangeable, and the
document reads: the list bought nothing on gold-recall for either model, while the
gate holds where the list-plus-mark composition (`s_linker124`) did not.

**Measured, not claimed.** Six paired flex-tier E2E runs (three each on two backends,
five projects, order alternating, in-invocation control `s_linker123`,
`results/greedymerge_e2e_{terra,luna}_r{1,2,3}_20260916v2` — the rerun that also
carries the `antecedent_form_rejected` logging fix above, so RQ3's NoCitation
ablation reads correctly) put doc-model link F1/F2 at terra `0.9347`/`0.9491` and
luna `0.8953`/`0.9192`, deltas over the in-set `s123gctl` control of terra
`+1.06`/`+1.85` and luna `+0.72`/`-0.07`. The component-weighted doc-code gate that
decides the REPORTED arm reads a mixed signal, not a clearance: terra doc-code file
F2 `+0.90` (3/3 BETTER) alongside doc-code worst-component F1 `-2.43` (3/3 WORSE);
luna doc-code file F1 `-0.43` (3/3 WORSE), every other tracked metric on both
backends INSIDE NOISE. Full provenance: `../results/s127_greedy_merge/README.md`
(the pre-implementation design audit), `../evaluation/reports/ARM_COMPARE_s126_vs_s123gctl.csv`
(the E2E gate read above).

**Promoted to the REPORTED arm on 2026-09-16 by explicit author decision on
simplicity, overriding the gate above rather than waiting for it to clear.** It is
the smallest change that still enforces the shortlist's own stated contract as a
code predicate rather than a paragraph the model may or may not honour, and the
mixed doc-code signal is recorded here rather than hidden: a real per-backend cost
sits beside the per-backend gain. `s_linker120` is the arm it replaced; see
`../approach/CLAUDE.md`'s status header for the full promotion entry.

**Standalone.** This file is the whole workflow — no linker base class, nothing imported
from a sibling `s_linkerNNN` module. It flattens the `s_linker122 -> s_linker123 ->
s_linker125 -> s_linker126` subclass chain the design history above passed through: the
union judge and its evidence-merged field (`written`/`competitors` in place of
`writes`/`alternatives`/`mention`), the shortlist's removal, and this file's own two
deltas, all inlined at the point Python's MRO already resolves each of them to.
`pilot/test_s126.py` (27 checks, no LLM calls) is the standing behavioural check.
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

#: One field's line in the rule: how much of the component's name this sentence
#: writes. Carries the sentence defining "qualified" -- the whole name appears, but
#: every writing of it sits inside a longer joined or dotted identifier -- so the
#: field block states four values from one line instead of three fields that have to
#: agree (`_written_as`). There is no `competitors` line: this variant's evidence
#: never carries that field (`_union_evidence` always clears it), so the rule states
#: nothing about it.
_WRITTEN_LINE = (
    "  written -- how the sentence writes the component's name. Use exact when the "
    "component's full catalog name is written as a name; alias when the document "
    "established an alternate form for that component and this sentence writes it; "
    "part when only one word of the component's multi-word name is written; and "
    "qualified name when the full catalog name occurs only inside a longer joined or "
    "dotted identifier. A shorter surface leaves more readings open; "
    "it does not make the reading in front of you wrong. Where the sentence does not "
    "write the name as such, ask what the expression itself denotes in its local "
    "context: a participant in the system, or something merely associated with "
    "software.")

#: The rule, in one piece: what a trace link is, then the one evidence field's line,
#: then the two grounds for rejecting. `STRICTER_CLAUSE` carries no scope guard here —
#: it is about an ordinary word coinciding with a component's name, and every case
#: names a component, so it has a subject in all of them.
TRACE_LINK_RULE = f"""{_DEFINITION}{MENTION_COUNTS}

{_FORMAT} The evidence says what the expression is doing here; none of it is a verdict.

{_WRITTEN_LINE}

{POSITIVE_GROUND}

{STRICTER_CLAUSE}

{QUALIFIED_CLAUSE} {ACTS_ON}"""

#: The one weighing this rule's whole-name/short-form/word-only distinction licenses
#: and the rule text itself does not state: a shorter surface is not, by itself,
#: evidence the case is right. SCOPED to the rows where the sentence does not write the
#: name in full -- an unscoped form was measured and reached the whole-name row, where
#: it contradicts `MENTION_COUNTS`, costing seven gold links on one bare enumeration of
#: component names on luna. Scoped, the two clauses cannot meet: a whole-name case is
#: out of this one's reach. Ground: general -- use versus mention, which holds for any
#: text and names no surface form, no component and no document shape (GATE-06/07).
SURFACE_NOT_EVIDENCE = (
    "Where the sentence does not write the name in full, that a surface can name this "
    "component is not evidence that it does here."
)

#: The quote-before-verdict demand, and the reply contract it names. Demanding a
#: committed quote is measurably worth its bytes; verifying the quote back against the
#: sentence voided nothing over six runs, so it is demanded and not re-checked.
UNION_DEMAND = ('For each case, first quote the EXACT words from the sentence the '
                'verdict rests on -- the words that state the architectural claim '
                'about the component, or "none" if the sentence makes no such claim '
                '-- then decide approve true/false based on that quote.')

UNION_REPLY = ('{"validations": [{"case": 1, "claim": "<exact quote or none>", '
               '"approve": true}]}')

#: Which evidence fields a case may print, in the order it prints them. `competitors`
#: is never populated in this variant -- an ambiguous surface is discarded before it
#: reaches a case at all -- so no case ever prints it, but the name stays on the wire
#: contract for anything reading a case's shape.
UNION_FIELDS = ("written", "competitors")

    #: The recorded decision-record projection. `written`'s four paper-facing values
    #: collapse to the three the phase log and every downstream RQ script already
    #: expect -- `written` and `naming` cannot disagree because `naming` is derived
    #: from it, never computed beside it.
NAMING_OF = {
    "exact": "whole name",
    "qualified name": "whole name",
    "alias": "alias",
    "part": "word only",
}


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


#: The per-case line the antecedent shortlist used to render on, kept only as the
#: marker a stray copy of the ancestor's prompt would still contain -- this variant
#: never writes the line in the first place, so nothing here strips it at runtime.
_CASE_LINE = "NAMED BEFORE THIS CASE:"


# ─────────────────────────────────────────────────────────────────────────────
# Main linker
# ─────────────────────────────────────────────────────────────────────────────
class SLinker126:
    """Greedy whole-name ownership over the union stream; an antecedent-form gate
    over the coreference resolver's citations. Standalone.

    Two name scans still merge by pair into one stream judged by one rule, and the
    coreference linker still runs behind them against its own rule -- the union-judge
    design `s_linker120`/`s_linker122` established. What is new here: a written whole
    catalog name owns every word it contains, so the partial-name scan does not
    propose a second component out of a word another component's name has already
    claimed; a surface that still reaches more than one component after that is
    discarded rather than judged, because the written evidence does not identify one
    component and there is no second judge to arbitrate; and the coreference resolver,
    already shown no antecedent shortlist, has its citations checked against what the
    cited sentence actually writes before a resolution ever reaches its judge. There is
    no linker base class and no controller: the module docstring says what the whole
    file holds.
    """

    _VARIANT_NAME = "s_linker126"

    #: The antecedent-form values a coreference citation must write to reach the
    #: judge: the exact name, or a document-established alias. A part of a name, or
    #: nothing the catalog recognizes, is rejected before judging.
    ANTECEDENT_FORMS = ("exact", "alias")

    #: Execution order. The two name scans are one stage: they propose into one
    #: stream, one judge reads it, and each link carries the label of the scan that
    #: proposed it (`_stage_of`). Coreference runs last. No linker is shown what the
    #: earlier one linked.
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
    #: case for a judge should not also refuse one.
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
        print("SLinker122 (name scan -> one evidence-graded judge -> coreference;"
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

    def _run_linker(self, linker, sentences, components, name_to_id, sent_map):
        """Dispatch. No linker receives the links the earlier one produced.

        Two entries, not three: the full-name and partial-name scans propose into
        one stream judged by one call, and what used to be the order between them is
        now a fact in the case (`naming`). Whatever coreference re-proposes, `link`
        merges by pair, and the label is decided by `_stage_of`, not by order.
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
        """No antecedent shortlist is computed. Always empty.

        The resolver's case prints no per-case list of what has been named before it
        (`_prompt_coref` below): the earlier round's list was an attachment prior with
        no measured recall benefit on either model, so it is not scanned for here at
        all rather than scanned for and withheld.
        """
        return []

    def _prompt_coref(self, comp_names, sentence_table, targets) -> str:
        """The resolver's call: every sentence in context, no antecedent shortlist.

        `_named_before` above always answers "no antecedent found", so a case block
        that rendered its line would always read `NAMED BEFORE THIS CASE: none` — the
        line and the paragraph endorsing it are omitted instead of rendered and then
        discarded, so the prompt never carries dead text (`_CASE_LINE` is what that
        omitted line would have started with, kept only so a stray reintroduction of
        the ancestor's rendering is easy to grep for). The procedural half of the
        paragraph survives: quoting the referring expression before naming the
        component is what makes the reply auditable and was never about the list.
        """
        blocks = []
        for target in targets:
            self._named_before(comp_names, sentence_table, target["target"])
            blocks.append(
                f"--- Case {target['case']} ---\n"
                f"TARGET S{target['target']}: {target['text']}"
            )
        return f"""Resolve references (pronouns and noun phrases that refer back) to components.

COMPONENTS: {', '.join(comp_names)}

SENTENCES (the document text the cases are drawn from)
{json.dumps(sentence_table)}

For each TARGET sentence below, identify any pronoun or noun phrase in THAT sentence
that refers back to a component listed above. Read the TARGET's context in SENTENCES.
If a target sentence has no such reference to a listed component, return no resolution
for it. Be conservative — only include resolutions you are CERTAIN about.

Quote the referring expression first, then name the component it points to.

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

    def _covering_names(self, text, name, components):
        """Spans of every OTHER component's whole name written in ``text``.

        What a candidate's matched word could instead be owned by: the catalog names
        this sentence writes in full, excluding the candidate's own component.
        """
        spans = []
        for component in components:
            if component.name != name:
                spans.extend(self._name_spans(
                    text, component.name, NameForm.ANY_CASE))
        return spans

    def _only_inside_another_name(self, text, name, components) -> bool:
        """Whether every matched word is owned by another written whole name.

        The catalog-only ownership predicate `s_109`/`s_121` measured: a one-word
        partial-name match is another component's if every span it matched sits
        strictly inside a whole name that component writes, and that containing name
        is the longer string at every span. Reads only given input -- the catalog and
        the document -- so it may END a case rather than merely open one for a judge,
        the same ground `s_109`'s nesting refusal stood on.
        """
        mine = self._name_spans(text, name, NameForm.ANY_WORD)
        covering = self._covering_names(text, name, components)
        return bool(mine and covering) and all(
            any(start <= a and b <= end and end - start > b - a
                for start, end in covering)
            for a, b in mine)

    def _scan(self, sentences, components):
        """Every (sentence, component) pair whose sentence writes one word of a name,
        excluding a pair a written whole OTHER name already owns.

        Nothing here admits a link and **nothing here ends one either** except the
        ownership exclusion: every other pair the relation finds is a case for the
        judge. Later spans of the same pair overwrite earlier ones, so the recorded
        `matched_text` is the last surface found in the sentence.

        This is the whole deterministic layer of the one-word stream — the relation,
        the whole-name exclusion, and the ownership exclusion, and no fourth rule.
        `s_linker109`'s nesting refusal stood here once, was removed as neutral in
        `s_linker121`'s round, and is reinstated here on its own terms: a word written
        only inside some *other* component's whole name is that name's.
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
        return [candidate for candidate in candidates.values()
                if not self._only_inside_another_name(
                    candidate.sentence_text, candidate.component_name, components)]

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
    #: Everything else `_mention_label` can say is a restatement of the case
    #: header, and dropping it was measured to cost 3-21% of those approvals.
    RETAINED_MENTION_TYPES = frozenset({
        MentionType.VIA_ALIAS,
        MentionType.CODE_TOKEN,
    })


    # ═════════════════════════════════════════════════════════════════════════
    # Linker 1 — NAME: both scans, one stream, one rule, one judging pass, greedy
    # whole-name ownership over what the partial-name scan can still propose.
    # ═════════════════════════════════════════════════════════════════════════

    # ── The union judge's declarations ───────────────────────────────────────

    #: How the old three-field evidence line said what the sentence writes of the
    #: name, kept only as a resolved class attribute -- nothing in this file reads it,
    #: since `_format_union_case` prints `written` directly rather than mapping
    #: through it, exactly as the ancestor chain this file flattens already did.
    WRITES = {
        "exact": "the exact component name",
        "alias": "a short form the document established for it",
        "part": "one word of the name",
        "qualified name": "the exact component name inside a longer identifier",
    }

    #: The four values of ``written``, as printed in the entity-judge evidence bundle.
    WRITTEN = ("exact", "alias", "part", "qualified name")

    # ── the merged stream ────────────────────────────────────────────────────

    def _name_candidates(self, sentences, components, name_to_id, sent_map):
        """Both scans, unchanged, merged by pair. The whole-name scan wins a tie.

        There is no tie by construction — `_scan` skips a pair whose sentence
        states a whole name, and now also drops a pair a written whole OTHER name
        already owns — and the rule is stated anyway so the merge does not depend on
        that property holding in a fork.
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

    def _only_in_identifier(self, text: str, name: str) -> bool:
        """Does every writing of the name in this sentence sit inside a longer path?

        The one fact the old `CODE_TOKEN` mention value contributed, as a predicate of
        its own rather than a value of an enum whose other four values are unreachable
        or redundant.
        """
        occurrences = list(re.finditer(rf"\b{re.escape(name.lower())}\b", text))
        return bool(occurrences) and all(
            self._in_dotted_path(text, m.start(), m.end()) for m in occurrences)

    def _written_as(self, text: str, name: str) -> str:
        """How much of the component's name this sentence writes: one of `WRITTEN`.

        One function for what an earlier round split across `naming` and
        `_mention_label`, which each re-derived the same `ANY_CASE` match and agreed
        only because they happened to call the same helper.
        """
        assert not self.SKIP_QUALIFIED, (
            "`qualified name` is a value of `written` because it entails the exact "
            "name; "
            "with SKIP_QUALIFIED set the two stop being the same relation")
        if self._writes_name(text, name):
            return ("qualified name" if self._only_in_identifier(text, name)
                    else "exact")
        for term, owner in getattr(
                getattr(self, "doc_knowledge", None), "aliases", {}).items():
            if owner == name and self._find_exact_form(text, term):
                return "alias"
        return "part"

    def _union_evidence(self, candidate, components, sent_map):
        """Every fact of the match this case carries. No weighing lives here.

        `written` is what the head split across `naming` and `mention`. `competitors`
        is always cleared: a surface still reaching more than one component after
        ownership is resolved is discarded in `_judge_union` before any case is built,
        so no case this method returns is ever ambiguous, and the field the rule no
        longer states a clause about carries nothing to contradict that.
        """
        text = candidate.sentence_text
        name = candidate.component_name
        written = self._written_as(text, name)
        return {
            "span": candidate.matched_text or name,
            "written": written,
            "competitors": [],
            "naming": NAMING_OF[written],
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
        """One case: the span and its component, the sentence, and two fields.

        Every case is this shape. What a whole-name case and a one-word case differ in
        is the *value* of `written`, and whether the match found competitors at all --
        which, in this variant, it never does.
        """
        previous = self._prev_prefix(candidate.sentence_number, sent_map)
        facts = [f"written={evidence['written']}"]
        if evidence["competitors"]:
            facts.append("competitors=" + ", ".join(evidence["competitors"]))
        return "\n".join([
            f'Case {index}: "{evidence["span"]}" -> {candidate.component_name}',
            f'  {previous}"{candidate.sentence_text}"',
            f"  Evidence: {', '.join(facts)}",
        ])

    @staticmethod
    def _group_key(candidate):
        """The `(sentence, surface)` a candidate's ownership is grouped by.

        Two candidates share a key exactly when the same surface in the same sentence
        was proposed for more than one component — the case a written whole name did
        not resolve.
        """
        return (candidate.sentence_number,
                (candidate.matched_text or candidate.component_name).casefold())

    def _judge_union(self, candidates, components, sentences, sent_map):
        """Group by surface ownership, judge the unambiguous groups in one pass,
        discard the rest without a call.

        A group of one candidate is judged exactly as the union judge always judged
        every candidate: batched by size, in candidate order, so every call holds
        whatever mix of naming the document gave it. A group of more than one — one
        surface still reaching several components after `_scan`'s ownership exclusion
        — is discarded outright: the written evidence does not identify a single
        component, and there is no second judge to arbitrate between them. The
        discard is logged into `decisions` exactly like a judge's own rejection, so an
        ablation reading `judge_decisions` alone still sees it.
        """
        grouped = {}
        for candidate in candidates:
            grouped.setdefault(self._group_key(candidate), []).append(candidate)
        singletons = [group[0] for group in grouped.values() if len(group) == 1]
        ambiguous = [group for group in grouped.values() if len(group) > 1]

        if not singletons:
            approved, decisions = [], {}
        else:
            comp_names = get_comp_names(components)
            approved, decisions = [], {}
            for _, batch in iter_batches(singletons, self.JUDGE_BATCH):
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
                    claim = str(item.get("claim", "")).strip().strip("\"'“”‘’")
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

        for group in ambiguous:
            for candidate in group:
                stage = self._stage_of(candidate)
                decisions[(candidate.sentence_number, candidate.component_id)] = {
                    "approved": False,
                    "claim": "",
                    "naming": "word only",
                    "path": f"{stage}_rejected",
                    "stage": "name_ambiguous_discard",
                }
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
    # Linker 2 — COREFERENCE: the resolver, no antecedent shortlist, and a
    # citation gate on top of its strict judge.
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

    def _run_coreference_linker(self, sentences, components, name_to_id, sent_map):
        resolved, metadata = self._resolve_references(
            sentences, components, name_to_id, sent_map
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

    def _resolve_references(self, sentences, components, name_to_id, sent_map):
        """Every sentence goes to the LLM in context; no pronoun regex.

        There is no antecedent gate at the proposal step: requiring the antecedent
        sentence to state a name of the component here would be a scan, and this
        variant checks the citation after the resolver commits to it instead
        (`_antecedent_names_it`), not before. The resolution must still *report* an
        antecedent, and both sentence numbers it reports are checked against the
        document: a number the model invents cannot name a real sentence.
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

    def _antecedent_names_it(self, link, sent_map, metadata) -> bool:
        """Does the resolution's cited antecedent sentence write the component's name?

        Given input only: the catalog, the document, and the alias table, never a
        judge's verdict, so this predicate may END a case (`_only_inside_another_name`
        stands on the same ground). A resolution with no recorded antecedent sentence
        is passed through unfiltered -- this predicate has nothing to test.
        """
        record = metadata.get((link.sentence_number, link.component_id), {})
        number = record.get("antecedent_sentence")
        sentence = sent_map.get(number) if number is not None else None
        if sentence is None:
            return True
        return self._written_as(
            sentence.text, link.component_name) in self.ANTECEDENT_FORMS

    def _validate_coref_links(self, coref_links, sent_map, components, metadata):
        """The antecedent-form gate, then the single judging pass shown the
        resolution it is judging.

        The gate runs first and rejects a link whose cited antecedent does not write
        the component's name in a form `ANTECEDENT_FORMS` accepts; that rejection is
        logged into `decisions` exactly like a judge's own ``approved=False``, not
        silently dropped from the input the judge sees, so an ablation reading
        `judge_decisions` alone still sees these candidates. Shown only a sentence and
        a component name, the judge behind the gate has to guess which expression was
        claimed to refer and to what, and it rejects about half the gold resolutions
        put to it. The resolver had already committed to both -- the referring
        expression and the quote it read as the antecedent -- and neither is
        recoverable from the case, so both are printed.
        """
        admitted, predicate_rejected = [], {}
        for link in coref_links:
            if self._antecedent_names_it(link, sent_map, metadata):
                admitted.append(link)
            else:
                predicate_rejected[(link.sentence_number, link.component_id)] = {
                    "approved": False,
                    "claim": "",
                    "objection": "antecedent does not write the component's name",
                    "path": "antecedent_form_rejected",
                }

        if not admitted:
            validated, decisions = [], {}
        else:
            comp_names = get_comp_names(components)
            validated = []
            decisions: dict = {}
            self.llm.set_phase("phase_25_coreference_judge")
            for _, batch in iter_batches(admitted, self.JUDGE_BATCH):
                cases = []
                for i, lk in enumerate(batch):
                    # Every link reaching the judge has an admitted resolution: either
                    # it has no recorded antecedent sentence, or the antecedent gate
                    # above already checked what that sentence writes.
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

        decisions.update(predicate_rejected)
        return validated, decisions


    # ── Logging and checkpointing ────────────────────────────────────────────

    def _save_phase(self, text_path, phase_name, state):
        """Write one phase's state under this run's checkpoint directory."""
        save_phase_state(
            checkpoint_dir(text_path, self._VARIANT_NAME, backend_tag(self.llm)),
            phase_name, state)
