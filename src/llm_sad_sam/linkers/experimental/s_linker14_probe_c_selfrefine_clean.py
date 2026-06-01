"""S-Linker14 Probe C — Self-Refine / Reflexion-style loop on alias judge.

v2.2 PROBE WAVE (Phase 16 mechanism) — forked from `s_linker13_clean_v3`.
Tests Pillar A (inference-time refinement) Mechanism #2:

**Self-Refine 2-iter loop on the alias judge.**

ITER 0 (verifier): standard judge call returns per-mapping
``{verdict: "APPROVE"|"REJECT", weakness_class: "ambiguous"|"weak_evidence"|"none"}``.

ITER 1 (refine): for any mapping with ``weakness_class != "none"``, we
re-judge JUST those contested mappings with the verifier's
``weakness_class`` injected as additional context. The refine call gets
ALL mappings' iter-0 verdicts visible, but is only asked to revise the
contested ones.

Cap: 2 iters total (one verify + at most one refine). If iter 0 returns no
contested mappings, iter 1 is skipped entirely (the cheap path).

DESIGN NOTES
------------
- Only the alias-judge tier is wrapped; other tiers (seed-val, entity, coref)
  are byte-identical to parent.
- GATE-06: the verifier + refine prompts use abstract SE vocabulary only.
- We track per-call iteration counts to ``results/v2_2_probes/C_selfrefine/
  iter_counts/<text_stem>.json`` so the rollup can attribute cost.

USAGE
-----
Same constructor signature as ``SLinker13CleanV3``.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_sad_sam.linkers.experimental.s_linker13_clean_v3 import (
    SLinker13CleanV3,
    ALIAS_SCOPE_SCHEMA,
    AliasEntry,
)
from llm_sad_sam.linkers.experimental.prompts_v3 import (
    DOC_KNOWLEDGE_EXTRACTION_RULES,
    DOC_KNOWLEDGE_JUDGE_EXAMPLES,
    DOC_KNOWLEDGE_JUDGE_RULES,
)
from llm_sad_sam.core.data_types_v2 import DocumentKnowledge


# ─────────────────────────────────────────────────────────────────────────────
# Verifier prompt (iter 0)
# ─────────────────────────────────────────────────────────────────────────────
VERIFIER_PROMPT = """JUDGE: Review these component name mappings for correctness.

COMPONENTS: {component_list}

PROPOSED MAPPINGS:
{mapping_list}

{judge_examples}

{judge_rules}

For each mapping, emit a STRUCTURED verdict. The ``weakness_class`` field
is mandatory and must be one of:
- "ambiguous": the mapping is borderline; the term could plausibly refer
  to multiple components or to the system as a whole, and you are unsure
  which interpretation is correct.
- "weak_evidence": the mapping is plausible but you have low confidence
  because the term is too generic, or the linguistic evidence is thin.
- "none": you are confident in your verdict — APPROVE or REJECT — and no
  refinement is needed.

Return JSON:
{{
  "verdicts": [
    {{"term": "term1", "verdict": "APPROVE", "weakness_class": "none"}},
    {{"term": "term2", "verdict": "REJECT",  "weakness_class": "ambiguous"}}
  ]
}}
JSON only:"""


# ─────────────────────────────────────────────────────────────────────────────
# Refine prompt (iter 1) — only sent if iter 0 flags any contested mapping
# ─────────────────────────────────────────────────────────────────────────────
REFINE_PROMPT = """REFINE: A first-pass judge has flagged some mappings as uncertain.
You are now asked to reconsider ONLY the flagged mappings. The first-pass
verdict and its self-reported weakness are shown.

COMPONENTS: {component_list}

ALL FIRST-PASS VERDICTS (for context only; do not change un-flagged entries):
{all_verdicts}

CONTESTED MAPPINGS (revise these):
{contested_list}

{judge_examples}

{judge_rules}

For each contested mapping, weigh the first-pass weakness and emit a FINAL
verdict. If you cannot resolve the weakness, default to APPROVE (false
positives are filtered downstream; false negatives are unrecoverable).

Return JSON:
{{
  "verdicts": [
    {{"term": "term1", "verdict": "APPROVE"}},
    {{"term": "term2", "verdict": "REJECT"}}
  ]
}}
JSON only:"""


ITER_COUNTS_ROOT = Path("results/v2_2_probes/C_selfrefine/iter_counts")


class SLinker14ProbeCSelfRefineClean(SLinker13CleanV3):
    """Probe C: 2-iter Self-Refine loop on the alias judge.

    Iter 0 returns ``{verdict, weakness_class}`` for every mapping. If any
    mapping has ``weakness_class != "none"``, iter 1 re-judges just those
    mappings. Cap: 2 iters total.
    """

    _VARIANT_NAME = "s_linker14_probe_c_selfrefine_clean"

    def _record_iter_counts(
        self,
        total_mappings: int,
        iter0_contested: int,
        iter1_called: bool,
    ) -> None:
        text_path = self._current_text_path or "unknown"
        text_stem = Path(text_path).stem if text_path else "unknown"
        ITER_COUNTS_ROOT.mkdir(parents=True, exist_ok=True)
        p = ITER_COUNTS_ROOT / f"{text_stem}.json"
        payload = {
            "variant": self._VARIANT_NAME,
            "text_stem": text_stem,
            "total_mappings": total_mappings,
            "iter0_contested": iter0_contested,
            "iter1_called": iter1_called,
            "total_judge_calls": 1 + (1 if iter1_called else 0),
        }
        p.write_text(json.dumps(payload, indent=2))

    def _learn_document_knowledge_enriched(self, sentences, components):
        """Same extraction flow as parent; judge is the 2-iter refine loop."""
        comp_names = [c.name for c in components]
        doc_lines = [s.text for s in sentences]

        prompt1 = f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

{DOC_KNOWLEDGE_EXTRACTION_RULES}

{ALIAS_SCOPE_SCHEMA}

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": [{{"term": "short_form", "component": "FullComponent", "scope": "global"}}],
  "synonyms":      [{{"term": "specific_alternative_name", "component": "FullComponent", "scope": "local"}}]
}}
JSON only:"""

        for attempt in range(2):
            data1 = self.llm.extract_json(self.llm.query(prompt1, timeout=300))
            if data1:
                break
            if attempt == 0:
                print("    Doc knowledge: empty response, retrying...")

        all_mappings: dict[str, str] = {}
        all_scopes: dict[str, str] = {}
        if data1:
            abbr_recs = data1.get("abbreviations", [])
            syn_recs = data1.get("synonyms", [])
            if isinstance(abbr_recs, dict):
                abbr_recs = [{"term": k, "component": v, "scope": "local"} for k, v in abbr_recs.items()]
            if isinstance(syn_recs, dict):
                syn_recs = [{"term": k, "component": v, "scope": "local"} for k, v in syn_recs.items()]
            for rec in abbr_recs:
                if not isinstance(rec, dict):
                    continue
                term = rec.get("term")
                full = rec.get("component")
                scope = rec.get("scope", "local")
                if term and full in comp_names:
                    all_mappings[term] = full
                    all_scopes[term] = scope
            for rec in syn_recs:
                if not isinstance(rec, dict):
                    continue
                term = rec.get("term")
                full = rec.get("component")
                scope = rec.get("scope", "local")
                if term and full in comp_names:
                    all_mappings[term] = full
                    all_scopes[term] = scope

        approved: set[str] = set()
        if all_mappings:
            mapping_list = [f"'{k}' -> {v}" for k, v in list(all_mappings.items())[:25]]

            # ─────── ITER 0: verifier with weakness_class ───────
            prompt_iter0 = VERIFIER_PROMPT.format(
                component_list=", ".join(comp_names),
                mapping_list=chr(10).join(mapping_list),
                judge_examples=DOC_KNOWLEDGE_JUDGE_EXAMPLES,
                judge_rules=DOC_KNOWLEDGE_JUDGE_RULES,
            )
            iter0_data = None
            for attempt in range(2):
                iter0_data = self.llm.extract_json(self.llm.query(prompt_iter0, timeout=180))
                if iter0_data and iter0_data.get("verdicts"):
                    break

            iter0_verdicts: dict[str, dict[str, str]] = {}
            if iter0_data:
                for rec in iter0_data.get("verdicts", []):
                    if not isinstance(rec, dict):
                        continue
                    term = rec.get("term")
                    verdict = (rec.get("verdict") or "").upper()
                    weakness = (rec.get("weakness_class") or "none").lower()
                    if not term:
                        continue
                    if verdict not in ("APPROVE", "REJECT"):
                        verdict = "APPROVE"  # approve-biased default
                    if weakness not in ("ambiguous", "weak_evidence", "none"):
                        weakness = "none"
                    iter0_verdicts[term] = {"verdict": verdict, "weakness_class": weakness}

            # Anything iter 0 missed → default APPROVE, weakness "none".
            for term in all_mappings:
                if term not in iter0_verdicts:
                    iter0_verdicts[term] = {"verdict": "APPROVE", "weakness_class": "none"}

            contested = [
                t for t, v in iter0_verdicts.items()
                if v["weakness_class"] != "none"
            ]
            print(f"    [Probe C] iter0 verdicts: {len(iter0_verdicts)}, contested: {len(contested)}")

            iter1_called = False
            iter1_verdicts: dict[str, str] = {}
            if contested:
                # ─────── ITER 1: refine on contested only ───────
                iter1_called = True
                all_verdicts_lines = [
                    f"  '{t}' -> {all_mappings[t]} | iter0={iter0_verdicts[t]['verdict']} "
                    f"(weakness={iter0_verdicts[t]['weakness_class']})"
                    for t in iter0_verdicts
                ]
                contested_lines = [
                    f"  '{t}' -> {all_mappings[t]} | first-pass={iter0_verdicts[t]['verdict']}, "
                    f"weakness={iter0_verdicts[t]['weakness_class']}"
                    for t in contested
                ]
                prompt_iter1 = REFINE_PROMPT.format(
                    component_list=", ".join(comp_names),
                    all_verdicts=chr(10).join(all_verdicts_lines),
                    contested_list=chr(10).join(contested_lines),
                    judge_examples=DOC_KNOWLEDGE_JUDGE_EXAMPLES,
                    judge_rules=DOC_KNOWLEDGE_JUDGE_RULES,
                )
                iter1_data = None
                for attempt in range(2):
                    iter1_data = self.llm.extract_json(self.llm.query(prompt_iter1, timeout=180))
                    if iter1_data and iter1_data.get("verdicts"):
                        break
                if iter1_data:
                    for rec in iter1_data.get("verdicts", []):
                        if not isinstance(rec, dict):
                            continue
                        term = rec.get("term")
                        verdict = (rec.get("verdict") or "").upper()
                        if term in contested and verdict in ("APPROVE", "REJECT"):
                            iter1_verdicts[term] = verdict

            # Final approved set = iter 0 APPROVE entries overridden by iter 1.
            for term, v in iter0_verdicts.items():
                final_verdict = iter1_verdicts.get(term, v["verdict"])
                if final_verdict == "APPROVE":
                    approved.add(term)

            self._record_iter_counts(len(all_mappings), len(contested), iter1_called)

        knowledge = DocumentKnowledge()
        for term, comp in all_mappings.items():
            if term in approved:
                scope = all_scopes.get(term, "local")
                if scope not in ("global", "local"):
                    scope = "local"
                knowledge.aliases[term] = AliasEntry(component=comp, scope=scope)
                print(f"    Alias: {term} -> {comp} [{scope}]")

        return knowledge
