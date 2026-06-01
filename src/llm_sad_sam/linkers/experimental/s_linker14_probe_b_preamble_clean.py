"""S-Linker14 Probe B — Problem-statement preamble + cached per-dataset rubric.

v2.2 PROBE WAVE (Phase 15 mechanism) — forked from `s_linker13_clean_v3`.
Tests Pillar A (inference-time refinement) Mechanism #1:

1. **Problem-statement preamble**: a canonical TLR problem statement is
   prepended to the alias-judge prompt. The preamble gives the LLM the global
   task framing once, so the per-call rubric does not have to repeat it.

2. **Per-dataset cached rubric**: a one-shot rubric-generation call is made
   per dataset (keyed on text_path). The output is cached to disk under
   ``results/v2_2_probes/B_preamble_rubric/cache/<text_stem>.json`` and reused
   for every subsequent alias-judge call in that dataset. The cache is keyed
   on (a) text_stem and (b) a SHA1 hash of the component list, so cache
   reuse across runs is automatic and cache invalidation is automatic when
   the model_path or text_path differ.

DESIGN NOTES
------------
- The preamble + rubric are wired ONLY at the alias-judge tier
  (`_learn_document_knowledge_enriched`). Other tiers (seed-val, entity
  extraction, validation, coref) are unchanged from clean_v3. This is the
  cheapest place to test the mechanism per the v2.2 probe-wave directive.
- GATE-06: the preamble + the rubric-builder prompt use ONLY abstract SE
  vocabulary (no benchmark component names). The rubric output is scanned at
  runtime for benchmark term leakage before write.
- The rubric REPLACES the static ``DOC_KNOWLEDGE_JUDGE_RULES`` constant for
  the alias-judge call; ``DOC_KNOWLEDGE_JUDGE_EXAMPLES`` is preserved verbatim
  (the few-shot calibration signal stays intact).

USAGE
-----
Same constructor signature as ``SLinker13CleanV3``. Cache directory is fixed
to ``results/v2_2_probes/B_preamble_rubric/cache/`` per the v2.2-prep
directive.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from llm_sad_sam.linkers.experimental.s_linker13_clean_v3 import (
    SLinker13CleanV3,
    ALIAS_SCOPE_SCHEMA,
)
from llm_sad_sam.linkers.experimental.prompts_v3 import (
    DOC_KNOWLEDGE_EXTRACTION_RULES,
    DOC_KNOWLEDGE_JUDGE_EXAMPLES,
    DOC_KNOWLEDGE_JUDGE_RULES,  # fallback only — kept for resilience
)
from llm_sad_sam.core.data_types_v2 import DocumentKnowledge
from llm_sad_sam.linkers.experimental.s_linker13_clean_v3 import AliasEntry


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL TLR PROBLEM STATEMENT (preamble)
# ─────────────────────────────────────────────────────────────────────────────
# Abstract description of the TLR alias-judging task. Uses only generic SE
# vocabulary; no benchmark component names. Prepended verbatim to alias-judge
# prompts.
TLR_PROBLEM_PREAMBLE = """You are participating in Traceability Link Recovery (TLR)
for software architecture documentation. The full task: given a software
architecture document (SAD) and a software architecture model (SAM) containing
named components, identify all sentence-to-component trace links.

You are participating in ONE specific sub-task of TLR: the ALIAS JUDGE.
Earlier stages have proposed alternative names ("aliases") that might refer
to specific components in the document. Your job is to APPROVE the aliases
that genuinely refer to exactly one component, and REJECT the ones that are
too generic or refer to the system as a whole.

Why this matters: an approved alias becomes a permanent entry in the document
knowledge base and drives subsequent link-recovery stages. False approvals
introduce false positives that downstream stages may not catch. False
rejections cause permanent recall loss — the alias is gone, and any sentence
that uses that alias instead of the component's full name will not be linked.

Bias toward APPROVAL when in doubt: downstream filters can catch false
positives, but false negatives are unrecoverable.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Rubric-builder prompt (abstract — no benchmark vocabulary)
# ─────────────────────────────────────────────────────────────────────────────
RUBRIC_BUILDER_PROMPT = """You are building a per-document RUBRIC for the alias judge.
A "rubric" is a short, focused set of decision criteria that an alias judge
should apply when evaluating proposed alias -> component mappings on THIS
specific document.

Inputs to consider when constructing the rubric:
- The component list (architectural elements declared in the SAM).
- A bird's-eye view of the document's overall framing (a short summary
  inferred from the COMPONENTS list; do NOT request the full document).

Goals of the rubric:
1. Tell the judge which TYPES of aliases are typically valid on this document
   (e.g., abbreviations from initials; trailing words of multi-word names;
   CamelCase identifiers; multi-word descriptive phrases).
2. Tell the judge which TYPES of aliases are typically generic noise on this
   document (e.g., common English nouns/verbs not tied to a specific
   component; descriptive phrases that match the whole system rather than
   one component).
3. Keep the rubric short: at most 6 enumerated rules. Use abstract SE
   vocabulary ONLY — NEVER use any of the component names verbatim in the
   rubric body, and NEVER invent project-specific examples. If you need an
   example, use the abstract textbook placeholders: Lexer, Parser, Scheduler,
   Broker, Dispatcher, Controller — NEVER any of the components listed below.

COMPONENTS: {component_list}

Return JSON:
{{
  "rubric": "<the rubric text, 6 enumerated rules, abstract vocabulary only>"
}}
JSON only:"""


# ─────────────────────────────────────────────────────────────────────────────
# GATE-06 audit regex (mirrors voyager_train_tlr_v2 + BENCHMARK_TABOO.md)
# Used to scan the LLM-generated rubric for benchmark-term leakage before
# committing it to cache. Fail-loud: leakage raises ValueError at runtime.
# ─────────────────────────────────────────────────────────────────────────────
_TABOO_PATTERN = re.compile(
    r"(?i)\b("
    r"Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|"
    r"HTML5 Server|Nginx Proxy|Kafka Broker|Zookeeper|UserDBAdapter|"
    r"AudioWatermarking|MediaManagement|WebUI|Recommender|Persistence|"
    r"SlopeOneRecommender|ImageProvider|Datastore|JabRef|bibdatabase|bibentry|"
    r"mediastore|teastore|teammates|bigbluebutton|jabref|"
    r"PaymentSystem|UserDB|FrontEnd|Backend"
    r")\b"
)


CACHE_ROOT = Path("results/v2_2_probes/B_preamble_rubric/cache")


class SLinker14ProbeBPreambleClean(SLinker13CleanV3):
    """Probe B: TLR problem-statement preamble + cached per-dataset alias-judge rubric.

    Overrides ONLY ``_learn_document_knowledge_enriched``:
    - Prompt 1 (extraction) is byte-identical to parent.
    - Prompt 2 (judge) now (a) prepends ``TLR_PROBLEM_PREAMBLE`` and (b)
      substitutes a per-dataset cached rubric for ``DOC_KNOWLEDGE_JUDGE_RULES``.
    - Rubric build: 1 LLM call per dataset, cached under
      ``results/v2_2_probes/B_preamble_rubric/cache/<text_stem>.json``.
    """

    _VARIANT_NAME = "s_linker14_probe_b_preamble_clean"

    # ---------------------------------------------------------------
    # Cache machinery
    # ---------------------------------------------------------------
    def _cache_key(self, components) -> tuple[str, str]:
        """Return (text_stem, component_hash) used to key the rubric cache."""
        text_path = self._current_text_path or "unknown"
        text_stem = Path(text_path).stem if text_path else "unknown"
        comp_names = sorted(c.name for c in components)
        comp_hash = hashlib.sha1(
            "\n".join(comp_names).encode("utf-8")
        ).hexdigest()[:12]
        return text_stem, comp_hash

    def _cache_path(self, text_stem: str, comp_hash: str) -> Path:
        return CACHE_ROOT / f"{text_stem}__{comp_hash}.json"

    def _load_cached_rubric(self, components) -> str | None:
        text_stem, comp_hash = self._cache_key(components)
        p = self._cache_path(text_stem, comp_hash)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, ValueError):
            return None
        rubric = data.get("rubric")
        if not isinstance(rubric, str) or not rubric.strip():
            return None
        # Defensive: re-audit on read in case the file was tampered with.
        hits = _TABOO_PATTERN.findall(rubric)
        if hits:
            raise ValueError(
                f"Probe B cached rubric at {p} contains taboo tokens {hits!r}; "
                "delete cache entry and rebuild"
            )
        return rubric

    def _save_cached_rubric(self, components, rubric: str) -> None:
        text_stem, comp_hash = self._cache_key(components)
        p = self._cache_path(text_stem, comp_hash)
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        payload = {
            "variant": self._VARIANT_NAME,
            "text_stem": text_stem,
            "component_hash": comp_hash,
            "rubric": rubric,
        }
        p.write_text(json.dumps(payload, indent=2))

    def _build_rubric(self, components) -> str:
        """Generate a per-dataset alias-judge rubric via 1 LLM call.

        Output is GATE-06-audited before return; falls back to the static
        DOC_KNOWLEDGE_JUDGE_RULES if the audit fails (per probe instruction:
        fail loud but keep going — cache the static fallback so subsequent
        runs do not re-pay the cost).
        """
        comp_names = [c.name for c in components]
        prompt = RUBRIC_BUILDER_PROMPT.format(
            component_list=", ".join(comp_names),
        )
        data = None
        for _ in range(2):
            data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
            if data and isinstance(data, dict) and data.get("rubric"):
                break
        if not data or not isinstance(data, dict):
            print("    [Probe B] rubric build returned no JSON; falling back to static rules")
            return DOC_KNOWLEDGE_JUDGE_RULES
        rubric = data.get("rubric", "")
        if not isinstance(rubric, str) or not rubric.strip():
            print("    [Probe B] rubric build returned empty rubric; falling back to static rules")
            return DOC_KNOWLEDGE_JUDGE_RULES
        hits = _TABOO_PATTERN.findall(rubric)
        if hits:
            # GATE-06 fail-loud per probe directive.
            raise ValueError(
                f"Probe B rubric contains taboo tokens {hits!r}; refusing to cache. "
                "Investigate prompt design before retry."
            )
        return rubric

    def _get_rubric(self, components) -> str:
        cached = self._load_cached_rubric(components)
        if cached is not None:
            print(f"    [Probe B] using cached rubric ({len(cached)} chars)")
            return cached
        rubric = self._build_rubric(components)
        # Only cache if not the static fallback (so a retry can rebuild).
        if rubric is not DOC_KNOWLEDGE_JUDGE_RULES:
            self._save_cached_rubric(components, rubric)
            print(f"    [Probe B] built + cached rubric ({len(rubric)} chars)")
        return rubric

    # ---------------------------------------------------------------
    # Override the alias-judge tier
    # ---------------------------------------------------------------
    def _learn_document_knowledge_enriched(self, sentences, components):
        """Same flow as parent, but the judge prompt uses preamble + cached rubric.

        We reimplement the body inline to keep the divergence visible. The
        extraction prompt (prompt1) is byte-identical to parent. The judge
        prompt (prompt2) gains the preamble + cached rubric.
        """
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

        if all_mappings:
            mapping_list = [f"'{k}' -> {v}" for k, v in list(all_mappings.items())[:25]]

            # Build/load per-dataset rubric (1 LLM call per dataset, cached).
            rubric = self._get_rubric(components)

            prompt2 = f"""{TLR_PROBLEM_PREAMBLE}

JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

{DOC_KNOWLEDGE_JUDGE_EXAMPLES}

{rubric}

Return JSON:
{{
  "approved": ["term1", "term2"]
}}
JSON only:"""

            for attempt in range(2):
                data2 = self.llm.extract_json(self.llm.query(prompt2, timeout=120))
                if data2 and data2.get("approved"):
                    break
                if attempt == 0:
                    print("    Doc knowledge judge: empty response, retrying...")
            approved = set(data2.get("approved", [])) if data2 else set(all_mappings.keys())
        else:
            approved = set()

        knowledge = DocumentKnowledge()
        for term, comp in all_mappings.items():
            if term in approved:
                scope = all_scopes.get(term, "local")
                if scope not in ("global", "local"):
                    scope = "local"
                knowledge.aliases[term] = AliasEntry(component=comp, scope=scope)
                print(f"    Alias: {term} -> {comp} [{scope}]")

        return knowledge
