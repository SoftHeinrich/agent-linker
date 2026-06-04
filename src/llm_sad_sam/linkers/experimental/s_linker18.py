"""s_linker18 — Final clean unified variant.

Builds on s_linker18d (alias-aware antecedent check). Adds Cleanup A:
enum-based mention classification — replaces ad-hoc string returns from
`_classify_mention` with a typed `MentionType` enum + structured info.

This is a pure code-readability refactor; no behavior change vs 18d.

The chain summarised:
  17f (baseline)
    +18a  drop generic-filter (≈80 LOC removed, dead code at gpt-5.4)
    +18b  unified coref validation via entity twopass (+1 TP, −4 FP)
    +18c  drop Phase 4b code-path filter LLM phase (+1 FP on teammates)
    +18d  alias-aware structural antecedent check (replaces LLM bypass flag)
    +18   enum-based mention classification (this file)

Final design (the clean unified variant):
  Phase 1: knowledge acquisition (aliases, ambiguous names) — unchanged
  Phase 2: multi-framing extraction (A, B, C-of-2-passes ∩) — unchanged
  Phase 3: framing union — unchanged
  Phase 4: unified twopass validation (p1 = participation, p2 = specificity)
           — same validator for entity and coref candidates
  Phase 5: coref discovery + alias-aware antecedent gate
  Phase 6: dedup-by-key merge

experimental=True, canonical=False.
"""
from __future__ import annotations

import re
from enum import Enum

from llm_sad_sam.linkers.experimental.s_linker18d import SLinker18d
from llm_sad_sam.linkers.experimental.helper_v3 import has_standalone_mention


class MentionType(Enum):
    """Classification of how a component name appears in a sentence."""
    PROPER_STANDALONE = "proper case, standalone"          # CamelCase / capitalized
    LOWERCASE_PROSE = "lowercase mention"                   # word-boundary lowercase
    CODE_TOKEN = "lowercase, inside dotted path"            # foo.bar.name
    VIA_ALIAS = "via known alias"                           # alias surface form
    ANAPHORIC = "anaphoric reference"                       # pronoun / role-ref
    INDIRECT = "indirect/unclear match"                     # fallback


class SLinker18(SLinker18d):
    """Final unified variant — clean design with typed mention classification.

    Inherits all cleanups from 18a..18d. Refactors `_classify_mention` to use
    the `MentionType` enum; existing consumers receive the same string
    representation (`mention.value`) so behavior is preserved.
    """

    _VARIANT_NAME = "s_linker18"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"SLinker18 (clean unified variant — cleanups A+B+C+E+F applied)")

    def _classify_mention(self, comp_name: str, text: str) -> str:
        """Return a MentionType enum value describing the mention.

        The return type stays `str` for back-compat (consumers — primarily
        `_format_evidence` and twopass prompts — read the string). The
        enum gives the categories first-class identity in code.
        """
        mention = self._classify_mention_typed(comp_name, text)
        return mention.value

    def _classify_mention_typed(self, comp_name: str, text: str) -> MentionType:
        """Typed variant — useful when callers want the enum directly."""
        if has_standalone_mention(comp_name, text):
            return MentionType.PROPER_STANDALONE

        comp_lower = comp_name.lower()
        lowercase_match = re.search(rf'\b{re.escape(comp_lower)}\b', text)
        if lowercase_match:
            if self._all_occurrences_in_dotted_path(comp_lower, text):
                return MentionType.CODE_TOKEN
            return MentionType.LOWERCASE_PROSE

        if self.doc_knowledge:
            for alias, entry in self.doc_knowledge.aliases.items():
                if entry.component == comp_name and re.search(
                    rf'\b{re.escape(alias)}\b', text, re.IGNORECASE
                ):
                    return MentionType.VIA_ALIAS

        return MentionType.INDIRECT

    @staticmethod
    def _all_occurrences_in_dotted_path(comp_lower: str, text: str) -> bool:
        """True iff every word-boundary occurrence of comp_lower in text
        is immediately adjacent to a dot (i.e. a segment of a dotted
        identifier like `name.x`, `x.name`, or `a.name.b`)."""
        any_match = False
        for m in re.finditer(rf'\b{re.escape(comp_lower)}\b', text):
            any_match = True
            s, e = m.start(), m.end()
            in_dotted = (
                (s > 0 and text[s - 1] == ".") or
                (e < len(text) and text[e] == "." and e + 1 < len(text)
                 and text[e + 1].isalpha())
            )
            if not in_dotted:
                return False
        return any_match
