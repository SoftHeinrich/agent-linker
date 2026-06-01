"""S-Linker14 Probe D — Upstream-tier rule removal (COREF_RULES).

v2.2 PROBE WAVE (Phase 17 mechanism) — forked from `s_linker13_clean_v3`.
Tests Pillar C (upstream-tier rule removal):

**Replace static ``COREF_RULES`` with a runtime-built coref rubric.**

Rationale for picking COREF_RULES (not ENTITY_EXTRACTION_RULES):
- v2.0 EXT-01 analysis (06-09-SUMMARY.md) found the BBB FN gap lives
  UPSTREAM of the standalone-mention rule, NOT in the extraction-tier
  rules. Replacing ``ENTITY_EXTRACTION_RULES`` with an LLM primitive risks
  the same negative result as EXT-01.
- Coref is a different surface. The static ``COREF_RULES`` enforces 5
  numbered rules that may be over-restrictive on documents whose pronoun
  patterns differ from textbook (e.g., heavy use of "this" with implicit
  antecedent in BBB).
- Trim9 (seed-disambiguation rubric) shipped at +0.77pp Claude on the same
  runtime-rubric mechanism class. Trim9 is the closest sister to the
  Probe D design here, so the cost class is known.

DESIGN
------
1. ONE LLM call per dataset builds the coref rubric. Inputs: component list,
   abbreviated profile of the document (first ~30 sentences). Output: short
   coref rubric (max 6 enumerated rules).
2. The rubric REPLACES ``COREF_RULES`` in the per-batch coref prompt; everything
   else (component list, ANTECEDENT_ALIAS_GUIDE, JSON template) is unchanged.
3. GATE-06: rubric is taboo-scanned at build time. Failure → fail-loud.
4. The rubric is cached to disk under
   ``results/v2_2_probes/D_upstream/cache/<text_stem>.json`` and reused
   on subsequent runs of the same dataset.

USAGE
-----
Same constructor signature as ``SLinker13CleanV3``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from llm_sad_sam.linkers.experimental.s_linker13_clean_v3 import (
    SLinker13CleanV3,
    ANTECEDENT_ALIAS_GUIDE,
)
from llm_sad_sam.linkers.experimental.prompts_v3 import COREF_RULES  # fallback only
from llm_sad_sam.linkers.experimental.helper_v3 import (
    get_comp_names, has_standalone_mention, parse_snum,
)
from llm_sad_sam.core.data_types_v2 import SadSamLink


# ─────────────────────────────────────────────────────────────────────────────
# Coref rubric-builder prompt (abstract, no benchmark vocabulary)
# ─────────────────────────────────────────────────────────────────────────────
COREF_RUBRIC_BUILDER_PROMPT = """You are building a per-document RUBRIC for the
pronoun-coreference resolver in a software architecture documentation
analyzer. The resolver's job: given a sentence containing a pronoun ("it",
"they", "this", "these", "that", "those", "its", "their"), and surrounding
sentences (+- 5 sentence window), decide whether the pronoun grammatically
refers to a named architectural component.

Inputs you should consider when constructing the rubric:
- The component list (so you know which kinds of names exist in this SAM).
- An EXCERPT of the document (first ~30 sentences) to gauge the writing
  style: dense vs sparse references, formal vs informal, abbreviations
  commonly used, etc.

What the rubric should specify:
1. WHEN a pronoun in this style of document RELIABLY refers to a previously-
   mentioned component (e.g., when the previous sentence's subject is a
   component name).
2. WHEN the pronoun is too ambiguous to resolve (e.g., multiple components
   in the antecedent window; reference to a method or process rather than
   the component itself).
3. The MINIMUM evidence the resolver should require before emitting a
   resolution (e.g., the component name must appear verbatim in the
   antecedent sentence; the pronoun must be in subject position).
4. The DEFAULT when uncertain — bias toward NOT resolving (false coref
   creates false-positive links).

Constraints on the rubric:
- 4 to 6 enumerated rules total.
- Abstract SE vocabulary ONLY — NEVER quote any of the component names below
  in the rubric body, and NEVER invent project-specific examples. If you
  need an example use the abstract placeholders: Lexer, Parser, Scheduler,
  Broker, Dispatcher, Controller, Renderer — NEVER any of the components.
- The rubric must end with: "If you are unsure, do NOT resolve the pronoun."

COMPONENTS: {component_list}

DOCUMENT EXCERPT (first ~30 sentences):
{doc_excerpt}

Return JSON:
{{
  "rubric": "<rubric text, 4-6 enumerated rules, abstract vocabulary only>"
}}
JSON only:"""


# ─────────────────────────────────────────────────────────────────────────────
# GATE-06 audit regex (mirrors voyager_train_tlr_v2 + BENCHMARK_TABOO.md)
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


# NOTE (v2.2-RANGE-D-CACHEFIX, 2026-06-01): cache key extended to include
# (backend, model) so cross-backend probes do not share a rubric.
# The original Range D Claude test was confounded because Claude reused the
# gpt-5.4-built rubric (cache key was (text_stem, comp_hash) only). The new
# CACHE_ROOT is separate from the original one so existing gpt-5.4 rubrics
# under `results/v2_2_probes/D_upstream/cache/` remain untouched and the
# Probe D wave's STRONG_PASS provenance is preserved.
CACHE_ROOT = Path(
    os.environ.get(
        "PROBE_D_CACHE_ROOT",
        "results/v2_2_probes_range_d_cachefix/cache",
    )
)


class SLinker14ProbeDUpstreamClean(SLinker13CleanV3):
    """Probe D: runtime coref rubric replaces static ``COREF_RULES``.

    Builds the rubric once per (text_stem, comp_hash, backend, model) and
    caches it. Reuses the rubric across all coref batches in that
    (dataset, backend, model) triple.

    v2.2-RANGE-D-CACHEFIX: prior to this change the cache key was
    (text_stem, comp_hash) only, which caused cross-backend rubric reuse.
    The Range D Claude test reused gpt-5.4-authored rubrics, confounding
    the FAIL verdict. The cache key now includes backend+model.
    """

    _VARIANT_NAME = "s_linker14_probe_d_upstream_clean"

    # ---------------------------------------------------------------
    # Cache machinery
    # ---------------------------------------------------------------
    def _cache_key(self, components) -> tuple[str, str, str, str]:
        text_path = self._current_text_path or "unknown"
        text_stem = Path(text_path).stem if text_path else "unknown"
        comp_names = sorted(c.name for c in components)
        comp_hash = hashlib.sha1(
            "\n".join(comp_names).encode("utf-8")
        ).hexdigest()[:12]
        try:
            backend = self.llm.backend.value if hasattr(self.llm.backend, "value") else str(self.llm.backend)
        except Exception:
            backend = "unknown_backend"
        model = self.llm.get_active_model() or "unknown_model"
        # Sanitize model string for filename use
        model_safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(model))
        backend_safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(backend))
        return text_stem, comp_hash, backend_safe, model_safe

    def _cache_path(self, text_stem: str, comp_hash: str,
                    backend: str = "", model: str = "") -> Path:
        # Backward-compat: if called with 2 args, use empty backend/model
        # (only used by callers that already unpacked the new tuple).
        if backend or model:
            return CACHE_ROOT / f"{text_stem}__{comp_hash}__{backend}__{model}.json"
        return CACHE_ROOT / f"{text_stem}__{comp_hash}.json"

    def _load_cached_rubric(self, components) -> str | None:
        text_stem, comp_hash, backend, model = self._cache_key(components)
        p = self._cache_path(text_stem, comp_hash, backend, model)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, ValueError):
            return None
        rubric = data.get("rubric")
        if not isinstance(rubric, str) or not rubric.strip():
            return None
        hits = _TABOO_PATTERN.findall(rubric)
        if hits:
            raise ValueError(
                f"Probe D cached coref rubric at {p} contains taboo tokens {hits!r}; "
                "delete cache entry and rebuild"
            )
        return rubric

    def _save_cached_rubric(self, components, rubric: str) -> None:
        text_stem, comp_hash, backend, model = self._cache_key(components)
        p = self._cache_path(text_stem, comp_hash, backend, model)
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        payload = {
            "variant": self._VARIANT_NAME,
            "text_stem": text_stem,
            "component_hash": comp_hash,
            "backend": backend,
            "model": model,
            "rubric": rubric,
        }
        p.write_text(json.dumps(payload, indent=2))

    def _build_coref_rubric(self, sentences, components) -> str:
        comp_names = [c.name for c in components]
        excerpt = "\n".join(f"S{s.number}: {s.text}" for s in sentences[:30])
        prompt = COREF_RUBRIC_BUILDER_PROMPT.format(
            component_list=", ".join(comp_names),
            doc_excerpt=excerpt,
        )
        data = None
        for _ in range(2):
            data = self.llm.extract_json(self.llm.query(prompt, timeout=180))
            if data and isinstance(data, dict) and data.get("rubric"):
                break
        if not data or not isinstance(data, dict):
            print("    [Probe D] coref rubric build returned no JSON; falling back to static COREF_RULES")
            return COREF_RULES
        rubric = data.get("rubric", "")
        if not isinstance(rubric, str) or not rubric.strip():
            print("    [Probe D] coref rubric build returned empty; falling back to static COREF_RULES")
            return COREF_RULES
        hits = _TABOO_PATTERN.findall(rubric)
        if hits:
            raise ValueError(
                f"Probe D coref rubric contains taboo tokens {hits!r}; refusing to cache. "
                "Investigate prompt design before retry."
            )
        return rubric

    def _get_coref_rubric(self, sentences, components) -> str:
        cached = self._load_cached_rubric(components)
        if cached is not None:
            print(f"    [Probe D] using cached coref rubric ({len(cached)} chars)")
            return cached
        rubric = self._build_coref_rubric(sentences, components)
        if rubric is not COREF_RULES:
            self._save_cached_rubric(components, rubric)
            print(f"    [Probe D] built + cached coref rubric ({len(rubric)} chars)")
        return rubric

    # ---------------------------------------------------------------
    # Override the coref tier
    # ---------------------------------------------------------------
    def _coref_cases_in_context(self, sentences, components, name_to_id, sent_map):
        """Same per-batch flow as parent; COREF_RULES → runtime rubric."""
        coref_rubric = self._get_coref_rubric(sentences, components)
        comp_names = get_comp_names(components)
        all_coref = []
        pronoun_sents = [s for s in sentences if self.PRONOUN_PATTERN.search(s.text)]

        for batch_start in range(0, len(pronoun_sents), 10):
            batch = pronoun_sents[batch_start:batch_start + 10]
            cases = []
            for sent in batch:
                context = []
                for i in range(max(1, sent.number - 5), sent.number + 6):
                    s = sent_map.get(i)
                    if s:
                        marker = ">>>" if s.number == sent.number else "   "
                        context.append(f"{marker} S{s.number}: {s.text}")
                cases.append({"sent": sent, "context": context})

            prompt = f"""Resolve pronoun references to architecture components.

COMPONENTS: {', '.join(comp_names)}

"""
            for i, case in enumerate(cases):
                prompt += f"--- Case {i+1}: S{case['sent'].number} ---\n"
                prompt += "CONTEXT:\n" + "\n".join(case["context"]) + "\n"
                prompt += f"TARGET: S{case['sent'].number} (marked with >>>)\n\n"

            prompt += f"""{coref_rubric}

{ANTECEDENT_ALIAS_GUIDE}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "pronoun": "it", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name", "antecedent_via_alias": false}}]}}

Only include resolutions you are CERTAIN about. JSON only:"""

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
                if not (has_standalone_mention(comp, ant_sent.text) or
                        res.get("antecedent_via_alias", False)):
                    continue
                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, source="coreference"))

        return all_coref
