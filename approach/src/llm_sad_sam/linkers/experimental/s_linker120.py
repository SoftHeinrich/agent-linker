"""S-Linker120 — one judge for both name streams; the rule is static, the evidence is not.

The head judges the two name streams with two prompts whose rubrics state opposite
defaults, and it routes a case to one or the other by which scan proposed it. This
variant asks the same question of both with **one rule**, and lets the difference be
carried by an **evidence line computed from the match**: `naming=whole name | alias |
word only`, the competing components the match could equally point to, and how long ago
the document last named this one. Nothing in the rule is new — every clause is one of
the head's own, re-scoped to speak about the evidence line instead of about the stream.

**Why this is the shape the branch's own results point at.** `pilot/gate_inventory.py`
states the fold law: *a gate folds into a judge's prompt exactly when that judge is
shown the information the gate reads.* Run forwards, the law says what a union needs —
not a better rubric but a bigger evidence line, because what a judge may be told
decides what its rule can say. `pilot/unijudge_audit.py` runs the census: of the twelve
axes on which the three judging calls differ, **five are facts of the match** (the
rubric's premise, the target, the catalog, the evidence bundle, the context window),
four are weighings, and three are the reply contract. This variant moves exactly the
five, keeps the four in one paragraph, and leaves the three alone.

**What the audit found the routing is actually doing** (six recorded head runs, five
projects, `../results/unijudge_audit/`):

    naming      case         gate          cases  base   kept/run  TP/run  FP/run
    whole name  capitalized  lenient         101  0.980     100.7    99.0     1.7
    whole name  lowercase    lenient          71  0.479      44.5    30.7    13.8
    alias       capitalized  lenient          39  0.487      24.5    19.0     5.5
    word only   lowercase    target-blind     72  0.306      12.8    11.0     1.8

The lenient gate's stream is **not one population**: two rows of it sit at base rate
0.98 and 0.48, and the low row carries **13.8 false positives a run, the largest single
bucket in the pipeline**, under a rubric that says "approve by default". The strictest
treatment in the workflow is spent on the row that costs 1.8. Routing by *stream* is a
coarser instrument than routing by *evidence*, and the evidence is a code fact either
way.

**The two results this variant must not walk into, and how it avoids each.**

*`s25`, −5.5 gold*: showing the denotation judge its target made the model confirm
identity instead of testing it. That was measured as a **second, rejecting pass** behind
the blind one — a target-shown reviewer that could only subtract. Here there is no
second pass: the target is shown once, in the only call the case gets, and the case
carries the competing components with it, which is `s_linker107`'s result (the
alternative set is a fact when the case contains it; enumerating it in code moved
spurious −10.0 where asking the model to enumerate moved it +6.6).

*`s119`, net −9.0 / −16.0*: making the sortal gate reply in the lenient gate's boolean
imported the lenient default onto a stream whose base rate is a third of it. Here the
defaults are **not** unified: the rule states one default per `naming` row, and the row
is selected by a fact the code computes, not by the schema. One reply shape, two
defaults, stated once.

**Level 1 only. This variant has never been run.** It exists so the arm is a file before
a pilot buys it, as the branch's policy requires; `pilot/test_s120_union.py` pins what
is structurally unchanged (both scans, the decision records, the link sources, the parse
path) and `pilot/unijudge_audit.py` prices what changes. What it does not answer —
whether one prompt asking the rule of both rows keeps what two prompts keep — is a stage
pilot on fixed recorded candidates, three samples a side, both models, **read per row
and not only in the total**: a union that trades the rows against each other reads
neutral in a sum and is not neutral.

**Cost, measured off the recorded runs.** The same 14 judging calls a five-project run
(296 cases in one stream fill the batches two streams of 215 and 81 fill), and 148.2 kB
of prompt against the 167.2 kB the two stages send today — the denotation window and the
duplicated anchors collapse into one table. No case is shown less than its current stage
shows it; word-only cases are shown *more* (the target, its anchors, the alternatives),
which is the whole point of the arm.

**Lineage.** `s_linker110`, unchanged except at the judging of the two name streams.
The coreference linker is untouched and deliberately outside this union: its cases are
not matches — there is no span the code computed — so "evidence computed from the match"
has nothing to say about them. Folding the strict gate in is the next rung, not this one.
"""
from __future__ import annotations

import json

from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.linkers.experimental.s_linker110 import (
    SLinker110, NameForm, QUALIFIED_CLAUSE, STRICTER_CLAUSE,
)

#: The one rule. Four clauses, each the head's own, plus one sentence per `naming` row
#: stating that row's default. `LAYERED_ENTITY_RULES` supplies the whole-name row
#: verbatim in substance ("approve by default ... reject only on a positive ground");
#: the word-only row states the standard the denotation judge applies today, now
#: sayable because the case carries the target and its alternatives.
UNION_LINK_RULES = """A link says the sentence makes an architectural claim about the component named in the case. Each case carries an evidence line stating how the sentence reaches that component; the row you are given decides how far to extend the case before asking it for more.

naming=whole name — the component's name is written here and the document treats it as part of the system. Approve by default: a mention that says nothing further about the component still counts as a valid link. Reject only on a positive ground -- that the sentence asserts nothing of this component, because the name is doing some other job here, or because the sentence denies what it would otherwise say of it.

naming=alias — as above, reached through a short form the document itself established for that component. The same default holds, and the ground for rejecting is the same one.

naming=word only — the sentence writes one word of the name and never the whole name. Approve only when that word is being used to name this component here; if it is used in its ordinary sense, or if a component listed under alternatives is the one the sentence means, reject."""

#: The two reject-grounds the lenient gate carries today, unchanged. They speak about a
#: surface in the sentence, which every case of this stream has, so both rows can read
#: them; the coreference gate still cannot, and still does not get them.
UNION_CLAUSES = f"{QUALIFIED_CLAUSE}\n{STRICTER_CLAUSE}"


class SLinker120(SLinker110):
    """`s_linker110` with the two name judges unioned behind one rule.

    Both scans, the union rule, every other prompt, the parse path and the log's views
    are the head's. What changes is one stage: the two judging passes become one, over
    one stream, with an evidence line per case.
    """

    _VARIANT_NAME = "s_linker120"

    #: Full name and partial name are proposed by two scans and judged by one call.
    LINKERS = ("name", "coreference")

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

    def _prompt_union(self, comp_names, sentence_table, cases) -> str:
        """One rule, one reply shape, the evidence carrying the rest."""
        table = (f"\nSENTENCES\n{json.dumps(sentence_table)}\n"
                 if sentence_table else "")
        return f"""Validate components in a document.

COMPONENTS: {', '.join(comp_names)}

{UNION_LINK_RULES}

{UNION_CLAUSES}
{table}
For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim), then decide approve true/false based on that claim.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>", "approve": true}}]}}
JSON only:"""

    def _format_union_case(self, index, candidate, evidence, sent_map, shown_in=0):
        previous = self._prev_prefix(candidate.sentence_number, sent_map)
        facts = [f"source={evidence['source']}", f"naming={evidence['naming']}"]
        if evidence["mention"]:
            facts.append(f"mention={evidence['mention']}")
        if evidence["alternatives"]:
            facts.append(f"alternatives={', '.join(evidence['alternatives'])}")
        if evidence["last_named"] >= 0:
            facts.append(f"named {evidence['last_named']} sentences earlier")
        lines = [
            f'Case {index}: "{evidence["span"]}" -> {candidate.component_name}',
            f'  {previous}"{candidate.sentence_text}"',
            f"  Evidence: {', '.join(facts)}",
        ]
        if evidence["anchors"]:
            if shown_in:
                lines.append(f"  Anchors (confirmed refs): as shown in Case {shown_in}.")
            else:
                lines.append("  Anchors (confirmed refs):")
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
        for _, batch in self._iter_batches(candidates, self.JUDGE_BATCH):
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
            self.llm.set_phase("phase_25_name_union_judge")
            data = self._ask(
                self._prompt_union(comp_names, table, cases),
                timeout=120, label="Union validation", require="validations",
            )
            verdicts = {}
            for item in (data or {}).get("validations", []):
                position = item.get("case", 0) - 1
                if 0 <= position < len(batch):
                    value = item.get("approve", False)
                    verdicts[position] = (
                        value is True
                        or (isinstance(value, str) and value.lower() == "true"),
                        str(item.get("claim", "")).strip(),
                    )
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
