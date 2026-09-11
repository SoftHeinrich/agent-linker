"""S-Linker120 — one judge, one rule: what a trace link is and how to read the evidence.

The head judges the two name streams with two prompts whose rubrics state opposite
defaults, and routes a case to one or the other by which scan proposed it. This variant
asks **one question of every candidate**: a trace link holds when the sentence makes an
architectural claim about the component. There is no lenient row and no strict row. What
differs between candidates is the **evidence computed from the match** — what the
sentence writes of the name, which components the same word could reach, what the code
can tell about the expression's place in the sentence, and which other sentences name
the component — and the rule says how to read each of those, not which rubric to apply.

**Measured, and adopted on that basis.** Stage pilot on fixed recorded candidates, both
arms in the same invocation, five projects, the alias table pinned
(`pilot/union_pilots.py`, `../results/union_round/`):

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
v13` puts two of them in one invocation). Three results from it are worth more than the
arm:

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
slice of one of the head's own constants** — checked against the constant it came from,
not retyped — and the residue is the definition of a trace link plus one line per
evidence field, each with a declared ground. Zero benchmark words of 63 catalog names,
zero dotted identifiers, zero document-shape enumerations, zero corpus-grounded
sentences. What the judge punches on is exactly two things: whether the sentence makes
an architectural claim about the component, and what the expression denotes where no
name is written.

**Invariants.** `pilot/test_s120_union.py` (2593 checks, five projects, no calls): the
merged stream is exactly `full ∪ partial` at the head's own bytes, every candidate keeps
the stage label its links and phase log are read by, every case carries its sentence and
span, a case whose match computed no component names none, no case invents an evidence
field, the verdict contract follows the batch, and an empty reply keeps nothing.

**Lineage.** `s_linker110`, unchanged except at the judging of the two name streams. The
coreference linker is untouched and outside this union: its cases are not matches — there
is no span the code computed — so evidence computed from the match has nothing to say
about them. Folding the strict gate in is the next rung, not this one.
"""
from __future__ import annotations

import json

from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.linkers.experimental.s_linker110 import (
    SLinker110, NameForm, QUALIFIED_CLAUSE, STRICTER_CLAUSE,
)
from llm_sad_sam.linkers.experimental.union_iterations import (
    ACTIVE, ITERATIONS, active,
)

#: **Iteration 7 — one rule, no rows.** Iterations 1-6 kept two standards inside one
#: prompt, selected by the `naming` row: the head's two defaults, addressed by a code
#: fact instead of by a stream. That is a union of the *prompts* and not of the
#: *question*, and it leaves the judge holding two definitions of a link. This rule
#: holds one: **what a trace link is, and how to read each piece of evidence.** Every
#: candidate is then one case in one format, and what differs between candidates is the
#: value of the evidence fields, not the standard applied to them.
#:
#: The measured trail that got here, all on gpt-5.6-terra, three samples, both arms in
#: one invocation, each iteration read against the control that ran beside it
#: (`../results/union_round/`):
#:
#:     iteration                                    gold      spurious   3*gold-sp
#:     1  paraphrased rows, alternatives reject    -11.3        -0.7       -33.2
#:     2  rows quoted verbatim, alternatives inert  -4.0       -12.7        +0.7
#:     3  + the word-only case blinded              -4.0       -13.4        +1.4
#:     4  + the quote demand made row-aware         -0.7        +3.3        -5.4
#:     5  + each row answering its own field        -3.0        +3.7       -12.7
#:     6  - alternatives and recency from the case  +0.7       +26.4       -24.3
#:
#: Two of those are facts this rule is built on rather than opinions about it.
#: **Iteration 6 is the sharpest**: dropping `alternatives` and the recency line from
#: the case cost **+26.4 spurious at +0.7 gold** -- an evidence field that names what a
#: case could reach *instead* is what restrains the judge, so evidence is worth stating
#: even when no clause tells the judge to weigh it. **Iteration 1 is its mirror**: the
#: same field, stated as a ground for rejecting, cost 7.6 gold on a bucket that is 0.765
#: gold. Evidence belongs in the case; how to read it belongs in the rule; a verdict
#: keyed to it belongs in neither.
#: The rule the variant runs, and every version of it that was measured, live in
#: `union_iterations.py` — one file to read the trail off. Override per process with
#: `UNION_ITERATION=v3`, which is how `pilot/union_pilots.py --iteration` runs an
#: older version as its own arm.
TRACE_LINK_RULE = ITERATIONS[ACTIVE].rule
UNION_LINK_RULES = TRACE_LINK_RULE
UNION_CLAUSES = ITERATIONS[ACTIVE].clauses


class SLinker120(SLinker110):
    """`s_linker110` with the two name judges unioned behind one rule.

    Both scans, the union rule, every other prompt, the parse path and the log's views
    are the head's. What changes is one stage: the two judging passes become one, over
    one stream, with an evidence line per case.
    """

    _VARIANT_NAME = "s_linker120"

    #: Full name and partial name are proposed by two scans and judged by one call.
    LINKERS = ("name", "coreference")

    #: The iteration this instance runs. Set on the instance so one process can hold
    #: two arms (`pilot/union_pilots.py` builds `control`, `union` and any named
    #: iteration in the same invocation).
    iteration_name: str | None = None

    @property
    def iteration(self):
        return active(self.iteration_name)

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

    def _run_linker(self, linker, sentences, components, name_to_id, sent_map):
        if linker == "name":
            return self._run_name_linker(
                sentences, components, name_to_id, sent_map)
        return super()._run_linker(linker, sentences, components, name_to_id, sent_map)

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
        previous = self._prev_prefix(candidate.sentence_number, sent_map)
        labels = {
            "source": lambda: f"source={evidence['source']}",
            "naming": lambda: f"naming={evidence['naming']}",
            "writes": lambda: f"writes={self.WRITES[evidence['naming']]}",
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
            (f'Case {index}: "{evidence["span"]}"' if blind else
             f'Case {index}: "{evidence["span"]}" -> {candidate.component_name}'),
            f'  {previous}"{candidate.sentence_text}"',
            f"  Evidence: {', '.join(facts)}",
        ]
        if not blind and evidence["anchors"]:
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
