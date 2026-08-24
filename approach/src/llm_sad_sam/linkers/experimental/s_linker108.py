"""S-Linker108 — the alias table reaches the judges, not just the first linker.

The alias module discovers, judges and records the names a document gives each
component ("the front end" = UI, "Database" = DB). Exactly one stage is then told
about them: ``_prompt_extraction``. The coreference resolver and **all three
judges** never see the table.

That the resolver is blind costs little -- 86.8% of refer-back antecedents are
written canonically, and only 2 gold links across five projects have an antecedent
written solely as an alias. The judges are the exposure, and the head's own
residual false negatives sit there:

    S32: "All salted hashes of passwords are also stored in the Database component."
    S33: "... to decouple the DataStorage from the database ..."      <- missed, gold DB

``Database`` is an approved alias for ``DB`` that the alias module already found and
S32 already uses in that sense. The denotation judge is target-blind by design and
alias-blind by omission, so it sees an ordinary English word and rejects.

This variant passes the approved table to the judging prompts and changes nothing
else: same rubrics, same asymmetric defaults, same evidence bundles, same batch
sizes, no extra calls. The table is already on ``self.doc_knowledge`` and already
written at the checkpoint, so nothing new is computed.

**The risk is real and runs the other way.** The denotation judge exists to separate
a component reference from ordinary English, and s25 measured that showing it more
about the target traded 5.5 true positives for 2.5 false positives. Telling it
`database = DB` may make it approve generic uses of the word too. This is exactly
the trade an F2 budget is willing to consider -- a point of recall is worth +0.76
F2 against +0.24 for a point of precision -- but it has to be measured on both
models, not assumed.

**Base.** ``s_linker101``.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker101 import SLinker101


class SLinker108(SLinker101):
    """The judges are told the names the document gives each component."""

    _VARIANT_NAME = "s_linker108"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker108 (approved aliases passed to the judging prompts)")

    def _alias_line(self) -> str:
        """The approved table, rendered once, or "" when the module found nothing."""
        aliases = getattr(getattr(self, "doc_knowledge", None), "aliases", {}) or {}
        if not aliases:
            return ""
        pairs = ", ".join(f"{term}={component}"
                          for term, component in sorted(aliases.items()))
        return ("\nKNOWN ALIASES (names this document gives these components): "
                f"{pairs}\n")

    def _prompt_validation(self, comp_names, cases, focus, strict: bool = False) -> str:
        """The head's prompt with the alias line inserted after COMPONENTS.

        The insertion point is the components list, because that is what the line
        qualifies: it says how this document writes those names. Everything below
        it -- the rubric, the asymmetric default, the claim-quoting instruction and
        the return schema -- is the head's, untouched.
        """
        base = SLinker101._prompt_validation(comp_names, cases, focus, strict=strict)
        line = self._alias_line()
        if not line:
            return base
        marker = f"COMPONENTS: {', '.join(comp_names)}\n"
        return base.replace(marker, marker + line, 1)
