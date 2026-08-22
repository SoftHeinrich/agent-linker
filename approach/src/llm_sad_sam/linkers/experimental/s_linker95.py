"""S-Linker95 — the merged reading with the cascade inside the call (rung B).

Rung A (``s_linker94``) asks both reference questions in one flat prompt. If that
loses precision, the ledger already says how: an undifferentiated question lets
the model report a refer-back for a sentence whose name it never established, and
those claims land on the lenient named judge.

This rung answers that mechanism without giving the call back. The head's own
ordering — names first, refer-backs resolved against what the names established —
is put *inside* the single call as two committed steps: STEP 1 reports every
sentence that writes a name, and STEP 2 resolves refer-backs **against the list
STEP 1 just produced** rather than against the whole component catalog.

This is the structure-first pattern that is the main positive result of the 2026
document-agent literature (DocSage: schema discovery, then extraction, then
reasoning; +27% over long-context and RAG baselines) and IRCoT's interleaving of
retrieval with the reasoning that consumes it.

Only the prompt changes. The routing, the carried per-component note, the
antecedent validity check, every judge, the alias module and the deterministic
scan are inherited from ``SLinker94`` and ``SLinker90`` untouched, and the three
authored rule constants are still composed verbatim, so GATE-07's byte accounting
is unchanged.
"""

from __future__ import annotations

import json

from llm_sad_sam.linkers.experimental.s_linker90 import (
    COREF_RULES,
    ENTITY_EXTRACTION_RULES,
    QUALIFIED_CLAUSE,
)
from llm_sad_sam.linkers.experimental.s_linker94 import SLinker94


class SLinker95(SLinker94):
    """One call, two ordered sections: names committed before refer-backs."""

    _VARIANT_NAME = "s_linker95"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker95 (one reading call, named section before refer-backs)")

    def _prompt_reading(self, comp_names, mappings, batch, established, table=None) -> str:
        carried = (
            "ESTABLISHED EARLIER (component: the last sentence before this block "
            "that named it)\n" + json.dumps(established)
            if established else ""
        )
        return f"""Extract ALL references to components from this document, in two steps.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings)}' if mappings else ''}
{carried}

STEP 1 — the sentences that write a name.

{ENTITY_EXTRACTION_RULES}

{QUALIFIED_CLAUSE}

STEP 2 — the sentences that refer back to one.

A sentence may also refer to a component without writing any of its names, through
a pronoun or a noun phrase that refers back. Work through STEP 2 only after STEP 1
is settled, and resolve each refer-back against the sentences you reported in STEP 1
together with any component listed as established earlier — not against the whole
COMPONENTS list. If no such sentence supplies the name, report nothing for it.

{COREF_RULES}

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Report both steps in one list. A sentence found in STEP 1 writes the name itself and
needs no antecedent; leave it null. A sentence found in STEP 2 must give the earlier
sentence that states the name it refers back to.

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence", "antecedent_sentence": M_INTEGER_OR_NULL, "antecedent_text": "exact quote naming the component, or null"}}]}}
JSON only:"""
