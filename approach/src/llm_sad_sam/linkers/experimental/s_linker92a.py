"""S-Linker92a — the extraction call replaced by the scan its own prompt describes.

`ENTITY_EXTRACTION_RULES` asks the extractor for one thing and states it as a
surface test, not as a judgement: *"Report a reference only when the sentence itself
writes the component's name, spelled as the COMPONENTS list spells it or as one of
the KNOWN ALIASES"*, and *"Among the sentences that do name it, report every one,
however incidental."*  Whether a mention carries an architectural claim is explicitly
deferred — `LAYERED_ENTITY_RULES` and `STRICTER_CLAUSE` decide that, one stage later.

A contract with no judgement left in it is a scan. This variant runs it as one:
every (sentence, component) pair whose sentence writes a name of the component,
where the component's names are its catalog name and the aliases the knowledge
stage discovered for it — `_names_by_component()`, the set N(c) the rest of the
module already reads. Nothing else changes. The knowledge stage, the full-name
judge, the partial-name linker and the coreference linker are inherited untouched.

**Priced from the recorded runs before it was built** (`pilot/regex_extract_audit.py`,
30 recorded runs of the s89–s92 extractor × 5 projects, no calls spent). Per run,
against 195 gold pairs:

    proposer            pairs   gold   prec   +gold  -gold   newgold  atrisk
    LLM extraction      175.3  150.1  0.856     -      -        -       -
    this scan           221.9  158.3  0.713   +10.6   -2.4    +10.2    -2.4

`newgold` is gold the scan proposes that the whole recorded pipeline — all three
linkers — never linked; `atrisk` is gold the pipeline holds only because the
extractor proposed it. **The scan's ceiling is 7.8 gold pairs a run above the
extractor's**, which is the branch's own error shape read back: the proposer, not
the judge, is where the headroom is.

What it costs is 53.3 more pairs a run in front of a lenient gate. Replaying that
gate over the recorded verdicts brackets the arm — every pair the gate was never
shown counted as rejected, then as approved:

    arm         policy      TP     FP   macro F1   macro F2
    control        -     180.6   36.4      91.04      92.93
    this scan   reject   178.5   33.6      91.00      92.50
    this scan   approve  187.8   75.7      87.16      92.70

The bracket is the open question, and `s_linker92b` closes most of it: 20.8 of the
53.3 added pairs are a name inside a qualified identifier, which `QUALIFIED_CLAUSE`
already tells this judge to reject.

**Measured at level 2** (`pilot/regex_proposer_pilots.py`, four arms in one invocation
per model, three runs a side, composed with the same run's untouched partial-name and
coreference stages; permutation test in `pilot/regex_round_stats.py`, p floor 0.10):

    model  arm   TP      FP     macro F1        macro F2
    terra  ctl   180.3   27.3   91.98           93.09
    terra  scan  186.7   32.7   92.43 (+0.4)    95.12 (+2.0, p = 0.10)
    luna   ctl   180.0   39.7   90.46           92.57
    luna   scan  190.7   59.0   89.61 (-0.9)    94.35 (+1.8, p = 0.20)

**macro F2 up on both models, macro F1 neutral on both** (+0.4 / -0.9, neither
significant), TP +6.3 / +10.7. The bracket resolves near its approve end on recall and
nowhere near it on precision, because the gate rejects the qualified-path pairs itself
-- 21/21 on terra, 12/19 on luna, no gold among the approvals. `s_linker92b` was built
to not propose those and is refused for that reason. What the gate does leak is
`STRICTER_CLAUSE`'s population; `s_linker92f` is the priced answer to it and this
variant is the head because it is better on F2 on both models and is the smaller
change. Round report: `../results/regex_round/README.md`.

What goes: `_prompt_extraction`, `ENTITY_EXTRACTION_RULES`, `_run_extraction_pass`
and one LLM call per 50 sentences — 9.0 of the ~16.8 calls a five-project run makes.
What arrives is no new deterministic machinery at all: `_name_spans` at `ANY_CASE`
is the predicate `_states_a_name` already calls, and no word list, catalog constant
or benchmark vocabulary enters (GATE-06).
"""

from __future__ import annotations

from llm_sad_sam.core.data_types_v2 import CandidateLink
from llm_sad_sam.linkers.experimental.s_linker92 import NameForm, SLinker92


class SLinker92a(SLinker92):
    """The full-name proposer is a scan of the catalog and the discovered aliases."""

    _VARIANT_NAME = "s_linker92a"

    #: Whether a name written inside a longer dotted identifier is skipped here.
    #: False in this variant: `QUALIFIED_CLAUSE` states the same thing to the judge
    #: that reads every one of these cases, and `s_linker92b` is the arm that asks
    #: whether saying it twice is worth the pairs.
    SKIP_QUALIFIED = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"{type(self).__name__} (full-name proposer = scan, 0 calls)")

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
