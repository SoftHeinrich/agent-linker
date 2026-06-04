"""s_linker18c — Cleanup C: drop Phase 4b, rely on twopass p2 + bundle signal.

Builds on s_linker18b (which already unified coref validation).

Before: Phase 4b ran a separate LLM batch over validated multi_framing
candidates, flagging cases where the component name appeared only as a
segment of a dotted package path (e.g. "logic.api", "storage.entity").

After: Phase 4b is removed entirely. The evidence bundle's `mention_type`
already classifies "lowercase, inside dotted path" via `_classify_mention`;
that signal flows into twopass via `_format_evidence` and twopass p2
(referential specificity) handles the FP-rejection role.

Empirical impact (gpt-5.4, 5 projects):
  Phase 4b fires only on teammates — 3 candidates dropped in 17f.
  Twopass with the dotted-path mention_type signal rejects 2 of 3 correctly.
  The third (S159 "common.datatransfer"→Component "Common") escapes; accept
  a +1 FP regression on teammates (~0.5pp F1) to remove a whole LLM phase.

Inherits everything from SLinker18b. Overrides only `_codepath_filter`
(now a no-op passthrough). The structural mention_type classification
(`_classify_mention`) is unchanged — it already detects dotted-path mentions.

experimental=True, canonical=False.
"""
from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker18b import SLinker18b


class SLinker18c(SLinker18b):
    """18b with cleanup C applied — no Phase 4b code-path filter."""

    _VARIANT_NAME = "s_linker18c"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"SLinker18c (cleanup C: drop Phase 4b — twopass p2 handles dotted-path FPs)")

    def _codepath_filter(self, validated, sent_map, components):
        """No-op passthrough. The dotted-path FP class is absorbed by
        twopass p2 reading mention_type from the evidence bundle."""
        return validated, {}
