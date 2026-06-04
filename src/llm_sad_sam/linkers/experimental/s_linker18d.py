"""s_linker18d — Cleanup B-refactor: alias-aware antecedent check.

Builds on s_linker18c (which dropped Phase 4b).

Before: a coref resolution was kept iff
    has_standalone_mention(comp, antecedent_text)  OR  antecedent_via_alias
where `antecedent_via_alias` was a Boolean the LLM had to emit and the
post-filter had to trust. Empirically necessary — without the bypass,
bigbluebutton lost 6 coref TPs (alias-only antecedents).

After: a structural check that examines the antecedent sentence for either
the canonical name OR any known alias of the component. The LLM-emitted
`antecedent_via_alias` flag is retained only for trace metadata; it no
longer gates extraction. The gate is purely structural — same recall,
no coupling between LLM output and post-filter regex.

Inherits everything from SLinker18c and overrides only
`_coref_cases_in_context` plus a new helper.

experimental=True, canonical=False.
"""
from __future__ import annotations

import re

from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.linkers.experimental.s_linker18c import SLinker18c
from llm_sad_sam.linkers.experimental.helper_v3 import (
    get_comp_names, has_standalone_mention, parse_snum,
)


class SLinker18d(SLinker18c):
    """18c with cleanup B-refactor — antecedent check inspects aliases directly."""

    _VARIANT_NAME = "s_linker18d"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"SLinker18d (cleanup B-refactor: alias-aware antecedent check)")

    def _antecedent_supports_resolution(self, comp_name: str, ant_text: str) -> bool:
        """True iff the antecedent sentence contains a canonical OR alias mention.

        Replaces the LLM-emitted `antecedent_via_alias` bypass: instead of
        trusting the model to flag alias-based resolutions, we check the
        document_aliases (already discovered in Phase 1) directly.
        """
        if has_standalone_mention(comp_name, ant_text):
            return True
        if not self.doc_knowledge:
            return False
        for alias, entry in self.doc_knowledge.aliases.items():
            if entry.component != comp_name:
                continue
            # Standalone proper-case match
            if has_standalone_mention(alias, ant_text):
                return True
            # Lowercase / case-insensitive word boundary match — handles
            # local-scope aliases (single common-word lowercase forms)
            if re.search(rf'\b{re.escape(alias)}\b', ant_text, re.IGNORECASE):
                return True
        return False

    def _coref_cases_in_context(self, sentences, components, name_to_id, sent_map):
        """Same as 17f's coref-cases-in-context, except the antecedent gate uses
        `_antecedent_supports_resolution` (structural alias check) instead of the
        LLM-emitted antecedent_via_alias bypass."""
        comp_names = get_comp_names(components)
        all_coref = []
        coref_metadata: dict = {}

        comp_terminals = self._classify_specific_terminals(components)
        role_ref_pat = re.compile(
            r'\bthe (' + '|'.join(re.escape(w) for w in sorted(comp_terminals)) + r')\b',
            re.IGNORECASE
        ) if comp_terminals else None

        anaphoric_sents = [
            s for s in sentences
            if self.PRONOUN_PATTERN.search(s.text)
            or (role_ref_pat and role_ref_pat.search(s.text))
        ]
        anaphoric_snums = [s.number for s in anaphoric_sents]
        self.llm.set_phase("phase_5_coref")

        for batch_start in range(0, len(anaphoric_sents), 10):
            batch = anaphoric_sents[batch_start:batch_start + 10]
            cases = []
            for sent in batch:
                context = []
                for i in range(max(1, sent.number - 5), sent.number + 6):
                    s = sent_map.get(i)
                    if s:
                        marker = ">>>" if s.number == sent.number else "   "
                        context.append(f"{marker} S{s.number}: {s.text}")
                cases.append({"sent": sent, "context": context})

            prompt = f"""Resolve anaphoric references (pronouns and role-referential noun phrases) to architecture components.

COMPONENTS: {', '.join(comp_names)}

"""
            for i, case in enumerate(cases):
                prompt += f"--- Case {i+1}: S{case['sent'].number} ---\n"
                prompt += "CONTEXT:\n" + "\n".join(case["context"]) + "\n"
                prompt += f"TARGET: S{case['sent'].number} (marked with >>>)\n\n"

            prompt += f"""{self._COREF_RULES}

{self._ANTECEDENT_ALIAS_RULES}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "reference": "the server", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name", "antecedent_via_alias": false}}]}}

Only include resolutions you are CERTAIN about. JSON only:"""

            data = None
            for attempt in range(2):
                data = self.llm.extract_json(self.llm.query(prompt, timeout=300))
                if data and data.get("resolutions"):
                    break
                if attempt == 0:
                    print(f"    Coref batch: empty response, retrying...")
            if not data:
                continue

            for res in data.get("resolutions", []):
                comp = res.get("component")
                snum = parse_snum(res.get("sentence"))
                if snum is None or not comp or comp not in name_to_id:
                    continue
                ant_snum = parse_snum(res.get("antecedent_sentence"))
                if ant_snum is None:
                    print(f"    Coref skip (no antecedent): S{snum} -> {comp}")
                    continue
                ant_sent = sent_map.get(ant_snum)
                if not ant_sent:
                    continue
                # Cleanup B-refactor: structural alias-aware check (replaces
                # antecedent_via_alias LLM-flag bypass).
                if not self._antecedent_supports_resolution(comp, ant_sent.text):
                    continue
                cid = name_to_id[comp]
                all_coref.append(SadSamLink(snum, cid, comp, source="coreference"))
                coref_metadata[(snum, cid)] = {
                    "reference": res.get("reference", ""),
                    "antecedent_sentence": ant_snum,
                    "antecedent_text": res.get("antecedent_text", ""),
                    # Retain LLM's flag for trace/audit only — no longer a gate.
                    "antecedent_via_alias": bool(res.get("antecedent_via_alias", False)),
                    "raw_resolution": res,
                }

        return all_coref, coref_metadata, anaphoric_snums, set(comp_terminals)
