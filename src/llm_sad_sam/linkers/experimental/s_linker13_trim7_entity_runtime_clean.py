"""S-Linker13 Trim7 Entity-Extraction Runtime Clean — Phase 12 EXTENSION (Plan 12-10).

REMOVED_FROM: prompts_v2.ENTITY_EXTRACTION_RULES (the static include/exclude
              rules used inside the entity extractor's per-batch prompt).
REPLACED_BY: inference-time rubric builder. A small rubric-builder LLM call
             receives a GENERIC SE-textbook seed example, the project
             document, the component list, and known aliases; it emits 4-6
             extraction criteria tailored to the document.
PRESERVED: dual-pass extraction consensus (Pass A + B intersection), global-
           scope alias filtering, all downstream validation.

NO STATIC FALLBACK: RuntimeError on empty rubric.

GATE-06: compiler-style seed example, consistent with sibling variants.

Variant priority: SIXTH (last) — Tier 2 proposer side. 12-04 showed
proposer-side risks (extraction widening dominated by VAL-EXT merge).
Rubric is built ONCE per document (not per batch) and reused across all
batches, capping the runtime cost at +1 LLM call per dataset.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker13_clean import (
    SLinker13Clean,
)
from llm_sad_sam.linkers.experimental.helper_v3 import (
    get_comp_names, parse_snum,
)
from llm_sad_sam.core.data_types_v2 import CandidateLink


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

ENTITY_RUBRIC_BUILDER_SEED_EXAMPLE = (
    "EXAMPLE (a generic compiler-style system, for reference shape only — NOT "
    "the project you will analyze):\n"
    "Components: Lexer, Parser, CodeGenerator, SymbolTable, Optimizer.\n"
    "A good 5-item extraction rubric for this kind of document would be:\n"
    "  - Include when the component name (or a known alias) appears as a\n"
    "    standalone token referring to the system entity\n"
    "  - Include when a space-separated form matches a compound name\n"
    "    (\"Symbol Table\" -> SymbolTable)\n"
    "  - Include when the component is the actor in an interaction\n"
    "    (\"X sends tokens to Y\" — both X and Y are references)\n"
    "  - Include in passive or prepositional phrases (\"handled by X\",\n"
    "    \"data flows through X\")\n"
    "  - Exclude when the name appears only inside a dotted path or as an\n"
    "    ordinary English word\n"
    "The rubric above is illustrative; build YOUR rubric from the project document below."
)


ENTITY_RUBRIC_BUILDER_PROMPT = """You are building a 4-6 item rubric for extracting references to architecture components from sentences in this specific document.

{seed_example}

PROJECT DOCUMENT:
{document_text}

PROJECT COMPONENTS:
{component_list}

KNOWN ALIASES (extracted earlier; may be empty):
{alias_list}

Produce a 4-6 item rubric grounded in patterns the document actually uses. Cover both INCLUDE criteria (standalone tokens, compound forms, actor / target roles, prepositional and passive forms) and EXCLUDE criteria (dotted paths, ordinary English use). Bias toward INCLUSION — downstream verification filters borderline cases. Do NOT pre-extract any reference — the rubric is the criteria, not the verdicts.

Return JSON:
{{"rubric": ["item 1", "item 2", "item 3", "item 4", "item 5"]}}
JSON only:"""


# ---------------------------------------------------------------------------
# Variant class
# ---------------------------------------------------------------------------


class SLinker13Trim7EntityRuntimeClean(SLinker13Clean):
    """Phase 12 EXTENSION (Plan 12-10): runtime-built entity-extraction rubric."""

    _VARIANT_NAME = "s_linker13_trim7_entity_runtime_clean"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trim7_rubric_calls = 0
        self._trim7_cached_rubric = None  # built once per link() call

    def _build_entity_rubric(self, components, sentences, mappings):
        """Build the entity-extraction rubric. Raise on empty."""
        comp_names = [c.name for c in components]
        component_list = "\n".join(f"- {n}" for n in comp_names)
        doc_lines = [s.text for s in sentences]
        alias_list = ", ".join(mappings[:20]) if mappings else "(none)"

        prompt = ENTITY_RUBRIC_BUILDER_PROMPT.format(
            seed_example=ENTITY_RUBRIC_BUILDER_SEED_EXAMPLE,
            document_text=chr(10).join(doc_lines),
            component_list=component_list,
            alias_list=alias_list,
        )

        rubric_data = None
        for attempt in range(2):
            rubric_data = self.llm.extract_json(self.llm.query(prompt, timeout=180))
            if rubric_data and rubric_data.get("rubric"):
                break
            if attempt == 0:
                print("    [trim7] Entity rubric builder: empty response, retrying...")

        if not (rubric_data and isinstance(rubric_data.get("rubric"), list)
                and rubric_data["rubric"]):
            raise RuntimeError(
                "trim7: entity rubric builder returned empty after 2 attempts; "
                "no static fallback by design (Plan 12-10 user directive)."
            )
        items = [str(r).strip() for r in rubric_data["rubric"] if str(r).strip()]
        if not items:
            raise RuntimeError(
                "trim7: entity rubric builder returned an empty list; "
                "no static fallback by design."
            )

        self._trim7_rubric_calls += 1
        rubric = (
            "EXTRACTION RUBRIC (generated for this document):\n"
            + "\n".join(f"- {r}" for r in items)
        )
        print(f"[trim7 rubric, call {self._trim7_rubric_calls}]")
        print(rubric)
        return rubric

    def _extract_entities_enriched(self, sentences, components, name_to_id, sent_map):
        """Dual-pass extraction with runtime rubric (built ONCE, reused across batches + passes)."""
        comp_names = get_comp_names(components)
        mappings = (
            [f"{term}={entry.component}" for term, entry in self.doc_knowledge.aliases.items()
             if entry.scope == "global"]
            if self.doc_knowledge else []
        )

        # Build the rubric ONCE per link() call.
        self._trim7_cached_rubric = self._build_entity_rubric(
            components, sentences, mappings)

        print("    Extraction pass A + B (parallel):")
        results = self._run_parallel({
            "pass1": lambda: self._run_single_extraction_pass(
                sentences, comp_names, mappings, name_to_id, sent_map, pass_label="[P1] "),
            "pass2": lambda: self._run_single_extraction_pass(
                sentences, comp_names, mappings, name_to_id, sent_map, pass_label="[P2] "),
        })
        pass1 = results["pass1"]
        pass2 = results["pass2"]

        intersected = {key: pass1[key] for key in pass1 if key in pass2}
        print(f"    Extraction consensus: Pass1={len(pass1)}, Pass2={len(pass2)}, "
              f"Intersect={len(intersected)} (dropped {len(pass1) + len(pass2) - 2*len(intersected)} unique-to-one-pass)")
        return list(intersected.values())

    def _run_single_extraction_pass(self, sentences, comp_names, mappings,
                                     name_to_id, sent_map, pass_label=""):
        """Single batch loop using the runtime rubric (not ENTITY_EXTRACTION_RULES)."""
        rubric = self._trim7_cached_rubric
        if rubric is None:
            # Should not happen — the wrapper above builds it before invoking
            # this method. But guard for the case where a future change calls
            # this method without going through the wrapper.
            raise RuntimeError(
                "trim7: _run_single_extraction_pass called before rubric was built. "
                "Ensure _extract_entities_enriched is the entry point."
            )

        batch_size = 50
        candidates = {}

        for batch_start in range(0, len(sentences), batch_size):
            batch = sentences[batch_start:batch_start + batch_size]
            if len(sentences) > batch_size:
                print(f"    {pass_label}Entity batch {batch_start//batch_size + 1}: "
                      f"S{batch[0].number}-S{batch[-1].number} ({len(batch)} sents)")

            prompt = f"""Extract ALL references to software architecture components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings[:20])}' if mappings else ''}

{rubric}

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence"}}]}}
JSON only:"""

            for attempt in range(2):
                data = self.llm.extract_json(self.llm.query(prompt, timeout=240))
                if data and data.get("references"):
                    break
                if attempt == 0:
                    print(f"    {pass_label}Empty response, retrying batch...")

            if not data:
                continue

            for ref in data.get("references", []):
                cname = ref.get("component")
                snum = parse_snum(ref.get("sentence"))
                if snum is None or not cname or cname not in name_to_id:
                    continue
                sent = sent_map.get(snum)
                if not sent:
                    continue
                matched = ref.get("matched_text", "")
                if matched and matched.lower() not in sent.text.lower():
                    continue
                key = (snum, name_to_id[cname])
                if key not in candidates:
                    candidates[key] = CandidateLink(
                        snum, sent.text, cname, name_to_id[cname],
                        matched, source="entity")

        return candidates
