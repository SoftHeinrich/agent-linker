"""S-Linker92c — the deleted prompt's morphology clause, made literal.

`s_linker92a` and `s_linker92b` implement all of `ENTITY_EXTRACTION_RULES` except its
second half: *"count a name written with different spacing, hyphenation or compound
joining as that name."*  `ANY_CASE` does not count it — `X Y` is not a
case-insensitive writing of `XY`. This variant scans the fidelity that is:
two expressions write the same name when their **signatures** agree, where a
signature is the word sequence left after CamelCase splitting and dropping separators.

That fidelity is not new to this branch. It is `s_linker76.NameForm.ANY_SPELLING`,
retired in s82 as a `_name_spans` branch nothing reached once the two tight scans had
been relocated into the extraction prompt. Deleting the prompt puts the branch back in
reach, so it is restored where it is used rather than re-added to the enum: the
relation point is this proposer's, and no other stage asks for it.

The typed round already priced the clause it transcribes and the answer was **keep**:
the population that writes a name at this fidelity and not at `ANY_CASE` is 3.3
pairs/run on terra and 12.0 on luna, and removing the clause removed none of it while
costing 5.0 gold on luna and 3.3 on terra.

What it buys as a scan is smaller than what it bought as a clause
(`pilot/regex_extract_audit.py`, 30 recorded runs x 5 projects):

    proposer         pairs   gold   prec    +gold   -gold   newgold  atrisk
    LLM extraction   175.3  150.1  0.856      -       -        -       -
    92b (any_case)   196.9  158.3  0.804   +10.6    -2.4    +10.2    -2.4
    92c (spelling)   198.2  159.1  0.803   +10.8    -1.8    +10.2    -1.8

**+0.8 gold and +1.3 pairs a run for a second fidelity in the code.** The gold it
adds is gold the recorded pipeline already links by another route (`newgold` is 10.2
either way); what it actually buys is 0.6 of the 2.4 pairs `92b` puts at risk. In the
replayed-gate bracket that is macro F1 +0.06 at the reject end and +0.08 at the
approve end — inside every noise band this branch has measured.

It is built because the clause it transcribes is measured load-bearing and dropping
it silently would be a design change disguised as a simplification. It is expected to
be **refused on elegance**: a 25-line second relation point for +0.8 candidate gold is
not a trade this branch has ever taken.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from llm_sad_sam.linkers.experimental.s_linker92b import SLinker92b

#: The tokenizer that cuts an expression into the words a signature is made of. It
#: splits on case boundaries as well as separators, which is the whole difference
#: between this fidelity and `ANY_CASE`. General English/identifier orthography; no
#: benchmark vocabulary and no word list (GATE-06).
SIGNATURE_PATTERN = r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+"

#: The separators a compound may be joined with, between two of its words.
JOINERS = r"[\s_-]+"


@lru_cache(maxsize=None)
def signature(expression: str) -> tuple:
    """An expression's word sequence, case-folded, separators and case boundaries gone.

    "X Y", "x-y" and "XY" share a signature, which is what makes a spelling variant
    recognizable. Compound splitting is **not** a relaxation of case folding: it
    reaches a spaced writing of a run-together name, and it misses a run-together
    writing whose case boundaries fall elsewhere, which case folding reaches. The two
    fidelities do not nest, and `s_linker92d` is the arm that takes their union.
    """
    normalized = unicodedata.normalize("NFKC", expression)
    normalized = normalized.replace("-", " ").replace("_", " ")
    return tuple(token.casefold()
                 for token in re.findall(SIGNATURE_PATTERN, normalized))


class SLinker92c(SLinker92b):
    """The scan counts spacing, hyphenation and compound joining as the same name."""

    _VARIANT_NAME = "s_linker92c"

    def _named_spans(self, text, name):
        """Spans of ``text`` whose signature is ``name``'s.

        A span of k words yields at least k signature tokens, so a span longer than
        the target's token count can never match it — which is what bounds the inner
        loop. The words of a span must be adjacent under `JOINERS`; anything else
        between them means they are not one compound.
        """
        target = signature(name)
        if not target:
            return []
        words = list(re.finditer(r"[A-Za-z0-9]+", text))
        found = []
        for index, first in enumerate(words):
            for last in range(index, min(len(words), index + len(target))):
                if last > index and not re.fullmatch(
                        JOINERS, text[words[last - 1].end():words[last].start()]):
                    break
                start, end = first.start(), words[last].end()
                if signature(text[start:end]) == target:
                    found.append((start, end))
        return found
