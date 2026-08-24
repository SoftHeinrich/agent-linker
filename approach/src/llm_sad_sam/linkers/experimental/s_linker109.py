"""S-Linker109 — a name's word, written only inside another component's name.

`QUALIFIED_CLAUSE` has said the same thing since s25: *"an expression that appears
only as a fragment of a longer identifier is naming a piece of that identifier, not a
participant in what the sentence describes."*  It is stated about dotted identifiers,
because that is the nesting `_in_dotted_path` computes. Names nest a second way — one
component's name can be a *word of* another's — and nothing in the module speaks about
that. The regex round found it and named it as its third unexplained regression
mechanism: *"a catalog name matched inside a longer name of a different component,
which no clause speaks about."*

`_scan` already refuses a pair for one reason of exactly this kind — *"unless the
sentence writes a whole name of it: that pair is the full-name linker's"*. This adds
the sibling case beside it, in the same words: if every writing of this component's
word sits inside a span where the sentence writes **another** component's whole name,
the sentence is naming that other component here, and the pair is that component's.

**Why the judge cannot be asked instead.** The denotation judge is target-blind by
design — its case carries the expression and the sentence, never the component — so it
answers `participant` for a real participant and is right; it simply has no way to know
that the participant it validated is a different one. Telling it the target is the
refusal the design law's own table records at **−5.5 gold** (s25). The distinction is a
fact about the case, the case's judge cannot be shown it, and so code is not merely the
better place for it but the only one.

Recorded shape, both models (`pilot/consolidation_audit.py`, six recorded runs of
`s_linker92a`): terra's denotation judge quotes the *longer* name it saw and luna
quotes the bare shared word, so the surface differs per model while the pair being
wrong does not — which is why the repair is a fact in code and not a clause in either
judge's prompt.

**Priced at level 1, no LLM calls, six recorded five-project runs**
(`pilot/test_s109_nesting.py`, 129 checks, which replays both `_scan`s pair by pair):

    model   candidates   refused   partial links   links lost   gold lost   final FP
    terra      101.0      12.0         34.7           5.0          0.0        -5.0
    luna        94.3      12.0         41.3          10.3          0.0       -10.3

**−5.0 false positives a run on terra and −10.3 on luna, at zero gold, in six runs of
six.** The refusal fires on **exactly 12 candidates in every run of both models** —
it reads the catalog and the document and nothing that is sampled, so unlike every
other arm on this branch it has no run-to-run band at all.

**No E2E is owed** (measurement policy, level 3): of the pairs a run this removes,
**0.0 are proposed by the coreference linker** in any of the six runs, so freeing them
from `_unlinked` gives a later stage nothing to re-propose and a paired run would
measure model drift instead of the change. This is the `s_linker85` precedent — a
deterministic relation settled by replaying the relation.

**What it does not claim.** The same predicate at the *full-name* stage was already
priced by the regex round at 1.0 pair and 0.0 gold a run and refused as below anything
this branch adopts a rule for; it is not re-derived here. This is the partial-name
stage only, where the scan reaches one word and the judge behind it is target-blind.

GATE-06: the runtime catalog only, no authored vocabulary and no alias table.
GATE-07: the ground is `QUALIFIED_CLAUSE`'s, which the general round scored admissible
as general SE practice — qualified names compose — and this is that same composition at
the extent the module already scans.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker92 import NameForm
from llm_sad_sam.linkers.experimental.s_linker92a import SLinker92a


class SLinker109(SLinker92a):
    """The scan refuses a word that only ever occurs inside another name."""

    _VARIANT_NAME = "s_linker109"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker109 (partial-name scan refuses words covered by another name)")

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
        """The head's generator, minus the pairs another component's name covers.

        The loop is the head's, called and filtered rather than restated: a restated
        candidate loop is where this line's one real bug hid.
        """
        kept, refused = [], 0
        for candidate in super()._scan(sentences, components):
            if self._only_inside_another_name(
                    candidate.sentence_text, candidate.component_name, components):
                refused += 1
                continue
            kept.append(candidate)
        if refused:
            print(f"    Partial-name scan refused {refused} "
                  f"(word written only inside another component's name)")
        return kept
