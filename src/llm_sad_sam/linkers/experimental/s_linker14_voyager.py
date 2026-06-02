"""S-Linker14 Voyager — v2.3 β architecture consumer linker.

Standalone file (per user preference). Does NOT inherit from s_linker13,
s_linker13_clean, or s_linker13_clean_v3. Does NOT import prompts_v2.

Pipeline logic copied verbatim from s_linker13_clean_v3 (Phase 12 Step 0).
Prompt source: prompts_v3_axiom (axiom skeletons only). Bank patterns are
injected at __init__ time via _wrap() — no runtime monkey-patching.

BANK FORMAT (slot-uniform, 15 slots)
-------------------------------------
{
  "version": "v4b",
  "slot_patterns": {
    "AMBIGUITY_RULES": [
      {
        "pattern_id": "p_001",
        "rule_text": "<2-4 sentence abstract rule>",
        "example_block": "TP: <synthesized example>\\nFP: <synthesized counter-example>"
      }
    ],
    ... 15 slots total (9 original + 6 new in v2.5) ...
  }
}

Empty bank or missing file: runs with pure axiom prompts (axiom-only floor mode).

REGISTRATION (GATE-07)
-----------------------
Registered in run_ablation.py CANONICAL_VARIANTS + VARIANT_SPECS with
experimental=True, canonical=False. Structured docstring documents β
architecture and trained-bank dependency.

GATE-06
-------
Bank patterns are GATE-06 scanned by the training harness before insertion.
This linker trusts the bank; no re-scan at inference time.

FROZEN ARTIFACT CONTRACT
-------------------------
Does not modify any frozen artifact:
  s_linker13.py, s_linker13_min.py, prompts_v2.py, ilinker*.py,
  data_types_v2.py, document_loader_v2.py, pcm_parser_v2.py.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_sad_sam.core.data_types_v2 import (
    SadSamLink, CandidateLink,
    ModelKnowledge, DocumentKnowledge,
)
from llm_sad_sam.core.document_loader_v2 import (
    Sentence, load_sentences, build_sent_map,
)
from llm_sad_sam.linkers.experimental.ilinker3 import ILinker3
from llm_sad_sam.linkers.experimental import prompts_v3_axiom as _axiom
from llm_sad_sam.linkers.experimental.helper_v3 import (
    coerce_mention_type,
    format_mention_string,
    has_standalone_mention,
    build_component_profile,
    parse_snum,
    get_comp_names,
)
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend


# ─────────────────────────────────────────────────────────────────────────────
# Bank loading + prompt-wrapping
# ─────────────────────────────────────────────────────────────────────────────

SLOT_NAMES = (
    "AMBIGUITY_FEW_SHOT",
    "AMBIGUITY_RULES",
    "DOC_KNOWLEDGE_EXTRACTION_RULES",
    "DOC_KNOWLEDGE_JUDGE_EXAMPLES",
    "DOC_KNOWLEDGE_JUDGE_RULES",
    "ENTITY_EXTRACTION_RULES",
    "VALIDATION_RULES",
    "COREF_RULES",
    "SEED_DISAMBIGUATION_RULES",
    # 15-slot expansion (REQ-V25-04): 6 new slots, empty default = v2.4 behavior preserved
    "SEED_EXTRACTION_RULES",
    "SEED_ACTOR_RULES",
    "GENERIC_WORD_USAGE_RULES",
    "ALIAS_SCOPE_RULES",
    "ANTECEDENT_ALIAS_RULES",
    "COREF_TERMINAL_SPECIFICITY_RULES",
)

_LEARNED_HEADER = (
    "\n\nLEARNED PATTERNS (apply when relevant; do not contradict the axioms above):"
)

DEFAULT_BANK_PATH = "results/voyager_v4b_v25/confirmation/cross_split_final_bank.json"


def _load_bank(path: str | os.PathLike[str] | None) -> dict[str, list[dict]]:
    """Load slot-uniform bank from path. Returns empty dict on missing/invalid."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, ValueError):
        return {}
    return data.get("slot_patterns", {}) if isinstance(data, dict) else {}


def _wrap(axiom: str, slot_name: str, slot_patterns: dict[str, list[dict]]) -> str:
    """Wrap an axiom prompt string with learned patterns for the given slot.

    Each pattern contributes rule_text and (if present) example_block.
    Returns the axiom unchanged if no patterns exist for this slot.
    """
    patterns = slot_patterns.get(slot_name, [])
    if not patterns:
        return axiom
    lines = []
    for p in patterns:
        rule = p.get("rule_text", "").strip()
        if not rule:
            continue
        lines.append(f"- {rule}")
        ex = p.get("example_block", "").strip()
        if ex:
            for ex_line in ex.splitlines():
                lines.append(f"  {ex_line}")
    if not lines:
        return axiom
    body = "\n".join(lines)
    return f"{axiom}{_LEARNED_HEADER}\n{body}"


# ─────────────────────────────────────────────────────────────────────────────
# ILinker3 injection subclass (REQ-V25-05)
# ilinker3.py stays frozen; injection lives here.
# ─────────────────────────────────────────────────────────────────────────────

class ILinker3Injected(ILinker3):
    """ILinker3 subclass that prepends bank-slot rules to seed extraction prompts.

    Empty slot strings → prompts identical to base ILinker3 (backward compatible).
    """

    def __init__(self, llm, seed_extraction_rules: str = "", seed_actor_rules: str = ""):
        super().__init__(llm=llm)
        self._seed_extraction_rules = seed_extraction_rules
        self._seed_actor_rules = seed_actor_rules

    def _prompt_extract(self, doc_block: str, comp_block: str) -> str:
        base = super()._prompt_extract(doc_block, comp_block)
        if not self._seed_extraction_rules:
            return base
        return f"{self._seed_extraction_rules}\n\n{base}"

    def _prompt_actor(self, doc_block: str, comp_block: str) -> str:
        base = super()._prompt_actor(doc_block, comp_block)
        if not self._seed_actor_rules:
            return base
        return f"{self._seed_actor_rules}\n\n{base}"


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses copied verbatim from s_linker13_clean_v3
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvidenceBundle:
    source: str
    matched_span: str
    mention_type: str
    preceding_text: str
    anchor_sentences: list[str]
    is_ambiguous: bool
    extraction_rationale: str


@dataclass(frozen=True)
class AliasEntry:
    component: str
    scope: str   # "global" | "local"


ALIAS_SCOPE_SCHEMA = """For each alias, also classify its SCOPE:
- "global": the alias is distinctive enough to refer unambiguously to the
  component anywhere it appears in the document.
  Typical shapes: multi-word forms ("Task Scheduler"), hyphenated forms
  ("task-scheduler"), CamelCase forms ("TaskScheduler"), all-caps
  abbreviations of length >= 2 ("RPC", "API"), or names whose first
  character is an uppercase letter ("Scheduler", "Broker").
- "local": the alias is a single all-lowercase word that overlaps with
  ordinary English vocabulary ("parser", "scheduler", "broker", "dispatcher").
  This alias is only safe to use where the surrounding sentence already
  establishes which component is being discussed.

Dotted-path fragments (tokens of the form X.Y or X.Y.Z that look like
package or module paths) are NOT aliases — do not include them.

Rule of thumb: if you would feel comfortable seeing the alias on its own
in any sentence of the document and immediately knowing which component it
names, it is "global"; otherwise it is "local".
"""


ANTECEDENT_ALIAS_GUIDE = """For each resolution, also set `antecedent_via_alias`:
- true:  the antecedent quote refers to the component by an ALIAS — an alternative
         name (an abbreviation, a hyphenated form, or a documented alternate name)
         rather than by the component's canonical name as listed in COMPONENTS above.
- false: the antecedent quote refers to the component by its CANONICAL NAME exactly
         as listed in COMPONENTS above.

Examples (abstract, generic patterns):
- COMPONENTS contains "TaskScheduler", antecedent quote is "The scheduler queues jobs"
  -> antecedent_via_alias = true  (uses an alternate form, not the canonical name).
- COMPONENTS contains "TaskScheduler", antecedent quote is "TaskScheduler queues jobs"
  -> antecedent_via_alias = false (uses the canonical name verbatim).

When you are unsure, default to false (the conservative side — only set true when
the antecedent clearly does not use the canonical name).
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main linker class
# ─────────────────────────────────────────────────────────────────────────────

class SLinker14Voyager:
    """v2.3 β architecture consumer linker — axiom prompts + trained bank patterns.

    Architecture: β (L + O + D-with-CoT-A + P training loop).
    Inference: axiom skeletons (prompts_v3_axiom) wrapped with slot-uniform
    bank patterns at __init__ time.

    experimental=True — research-grade; not canonical.
    canonical=False   — s_linker13_min remains canonical=True.

    Trained Bank (default, post-v2.5 Confirmation):
      results/voyager_v4b_v25/confirmation/cross_split_final_bank.json
    (override via bank_path constructor kwarg or VOYAGER4B_BANK_PATH env var).

    Empty bank / missing file: runs as axiom-only floor (valid for Phase 14
    infrastructure testing and for the iter-0 baseline measurement in Phase 15).

    v2.5 Confirmation Tier (Phase 29, 2026-06-02):
      - Infrastructure: oracle cache fix (REQ-V25-01) + 15-slot expansion (REQ-V25-04/05/06)
      - 3-split sweep (split1_replication, split2_bbb_in_train, split3_rotated_holdout)
      - Cross-split bank: results/voyager_v4b_v25/confirmation/cross_split_final_bank.json
      - Cross-split aggregation: Jaccard >= 0.6 dedup + >= 2-split survival filter
      - Bank: 12 patterns in 8 slots
      - Publishable macro F1 (gpt-5.4, 5-dataset): 89.1%
        MS 95.1%, TS 98.2%, TM 81.3%, BBB 73.7%, JAB 97.3%
      - Verdict: WEAK (5-dataset macro 89.1% in [0.87, 0.9173))
      - Lift over axiom-only floor: +1.5pp (87.6% -> 89.1%)
      - Default bank path updated to v2.5 cross_split_final_bank.json (2026-06-02)
    """

    _VARIANT_NAME = "s_linker14_voyager"

    PRONOUN_PATTERN = re.compile(
        r'\b(it|they|this|these|that|those|its|their)\b',
        re.IGNORECASE
    )

    def __init__(
        self,
        backend: LLMBackend | None = None,
        model: str | None = None,
        checkpoint_fallback: LLMBackend | str | None = None,
        checkpoint_fallback_model: str | None = None,
        bank_path: str | None = None,
    ):
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.4")
        self.llm = LLMClient(
            backend=backend or LLMBackend.CLAUDE,
            model=model,
            checkpoint_fallback=checkpoint_fallback,
            checkpoint_fallback_model=checkpoint_fallback_model,
        )
        self.model_knowledge: ModelKnowledge | None = None
        self.doc_knowledge: DocumentKnowledge | None = None
        self._phase_log: list[dict] = []
        self._current_text_path: str | None = None

        # Bank loading must happen before ilinker3 construction so injection slots are available
        resolved_bank = bank_path or os.environ.get("VOYAGER4B_BANK_PATH", DEFAULT_BANK_PATH)
        self._slot_patterns = _load_bank(resolved_bank)
        self._bank_path = str(resolved_bank)

        # Wrap all 9 axiom slots with bank patterns (empty patterns = axiom unchanged)
        self._AMBIGUITY_FEW_SHOT = _wrap(_axiom.AMBIGUITY_FEW_SHOT, "AMBIGUITY_FEW_SHOT", self._slot_patterns)
        self._AMBIGUITY_RULES = _wrap(_axiom.AMBIGUITY_RULES, "AMBIGUITY_RULES", self._slot_patterns)
        self._DOC_KNOWLEDGE_EXTRACTION_RULES = _wrap(_axiom.DOC_KNOWLEDGE_EXTRACTION_RULES, "DOC_KNOWLEDGE_EXTRACTION_RULES", self._slot_patterns)
        self._DOC_KNOWLEDGE_JUDGE_EXAMPLES = _wrap(_axiom.DOC_KNOWLEDGE_JUDGE_EXAMPLES, "DOC_KNOWLEDGE_JUDGE_EXAMPLES", self._slot_patterns)
        self._DOC_KNOWLEDGE_JUDGE_RULES = _wrap(_axiom.DOC_KNOWLEDGE_JUDGE_RULES, "DOC_KNOWLEDGE_JUDGE_RULES", self._slot_patterns)
        self._ENTITY_EXTRACTION_RULES = _wrap(_axiom.ENTITY_EXTRACTION_RULES, "ENTITY_EXTRACTION_RULES", self._slot_patterns)
        self._VALIDATION_RULES = _wrap(_axiom.VALIDATION_RULES, "VALIDATION_RULES", self._slot_patterns)
        self._COREF_RULES = _wrap(_axiom.COREF_RULES, "COREF_RULES", self._slot_patterns)
        self._SEED_DISAMBIGUATION_RULES = _wrap(_axiom.SEED_DISAMBIGUATION_RULES, "SEED_DISAMBIGUATION_RULES", self._slot_patterns)
        # 15-slot expansion (REQ-V25-06): wrap inline static prompts with bank slots
        self._ALIAS_SCOPE_RULES = _wrap(ALIAS_SCOPE_SCHEMA, "ALIAS_SCOPE_RULES", self._slot_patterns)
        self._ANTECEDENT_ALIAS_RULES = _wrap(ANTECEDENT_ALIAS_GUIDE, "ANTECEDENT_ALIAS_RULES", self._slot_patterns)
        # GENERIC_WORD_USAGE_RULES and COREF_TERMINAL_SPECIFICITY_RULES injected inline via _slot_text()

        # ILinker3Injected: wires SEED_EXTRACTION_RULES + SEED_ACTOR_RULES bank slots (REQ-V25-05)
        self._ilinker3 = ILinker3Injected(
            llm=self.llm,
            seed_extraction_rules=self._slot_text("SEED_EXTRACTION_RULES"),
            seed_actor_rules=self._slot_text("SEED_ACTOR_RULES"),
        )

        pattern_counts = {s: len(self._slot_patterns.get(s, [])) for s in SLOT_NAMES}
        total_patterns = sum(pattern_counts.values())
        print(f"SLinker14Voyager (v2.5 β consumer, axiom+bank, experimental=True)")
        print(f"  Backend: {self.llm.describe_backend()}")
        print(f"  Bank: {self._bank_path} ({total_patterns} patterns across {sum(1 for v in pattern_counts.values() if v > 0)}/15 slots)")

    def reload_bank(self, bank_path: str | None = None) -> int:
        """Reload bank from disk and recompute wrapped prompt strings.

        Returns total pattern count. Called by training harness between outer passes.
        """
        if bank_path is not None:
            self._bank_path = str(bank_path)
        self._slot_patterns = _load_bank(self._bank_path)
        self._AMBIGUITY_FEW_SHOT = _wrap(_axiom.AMBIGUITY_FEW_SHOT, "AMBIGUITY_FEW_SHOT", self._slot_patterns)
        self._AMBIGUITY_RULES = _wrap(_axiom.AMBIGUITY_RULES, "AMBIGUITY_RULES", self._slot_patterns)
        self._DOC_KNOWLEDGE_EXTRACTION_RULES = _wrap(_axiom.DOC_KNOWLEDGE_EXTRACTION_RULES, "DOC_KNOWLEDGE_EXTRACTION_RULES", self._slot_patterns)
        self._DOC_KNOWLEDGE_JUDGE_EXAMPLES = _wrap(_axiom.DOC_KNOWLEDGE_JUDGE_EXAMPLES, "DOC_KNOWLEDGE_JUDGE_EXAMPLES", self._slot_patterns)
        self._DOC_KNOWLEDGE_JUDGE_RULES = _wrap(_axiom.DOC_KNOWLEDGE_JUDGE_RULES, "DOC_KNOWLEDGE_JUDGE_RULES", self._slot_patterns)
        self._ENTITY_EXTRACTION_RULES = _wrap(_axiom.ENTITY_EXTRACTION_RULES, "ENTITY_EXTRACTION_RULES", self._slot_patterns)
        self._VALIDATION_RULES = _wrap(_axiom.VALIDATION_RULES, "VALIDATION_RULES", self._slot_patterns)
        self._COREF_RULES = _wrap(_axiom.COREF_RULES, "COREF_RULES", self._slot_patterns)
        self._SEED_DISAMBIGUATION_RULES = _wrap(_axiom.SEED_DISAMBIGUATION_RULES, "SEED_DISAMBIGUATION_RULES", self._slot_patterns)
        # 15-slot expansion: reload inline-prompt slots
        self._ALIAS_SCOPE_RULES = _wrap(ALIAS_SCOPE_SCHEMA, "ALIAS_SCOPE_RULES", self._slot_patterns)
        self._ANTECEDENT_ALIAS_RULES = _wrap(ANTECEDENT_ALIAS_GUIDE, "ANTECEDENT_ALIAS_RULES", self._slot_patterns)
        # Rebuild ILinker3Injected with updated seed rules
        self._ilinker3 = ILinker3Injected(
            llm=self.llm,
            seed_extraction_rules=self._slot_text("SEED_EXTRACTION_RULES"),
            seed_actor_rules=self._slot_text("SEED_ACTOR_RULES"),
        )
        return sum(len(v) for v in self._slot_patterns.values())

    def _slot_text(self, slot_name: str) -> str:
        """Return formatted bank patterns for slot_name, or empty string if slot is empty.

        Used for inline prompt injection (GENERIC_WORD_USAGE_RULES, COREF_TERMINAL_SPECIFICITY_RULES,
        SEED_EXTRACTION_RULES, SEED_ACTOR_RULES). Empty string = no change to existing prompt text.
        """
        patterns = self._slot_patterns.get(slot_name, [])
        if not patterns:
            return ""
        lines = [f"- {p.get('rule_text', '').strip()}" for p in patterns if p.get('rule_text', '').strip()]
        if not lines:
            return ""
        return "\n\nLEARNED PATTERNS:\n" + "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════
    # DAG Infrastructure (verbatim from s_linker13_clean_v3)
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _run_parallel(tasks):
        if len(tasks) == 1:
            name, fn = next(iter(tasks.items()))
            return {name: fn()}
        results = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(fn): name for name, fn in tasks.items()}
            try:
                for fut in as_completed(futures):
                    name = futures[fut]
                    results[name] = fut.result()
            except Exception:
                for other in futures:
                    other.cancel()
                raise
        return results

    # ═══════════════════════════════════════════════════════════════════════
    # Main Entry Point
    # ═══════════════════════════════════════════════════════════════════════

    def link(self, text_path, model_path, **_kwargs):
        self._phase_log = []
        self._current_text_path = text_path
        t0 = time.time()

        components = parse_pcm_repository(model_path)
        sentences = load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        sent_map = build_sent_map(sentences)

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        print("\n[Tier 1] Knowledge Acquisition (parallel)")
        acq = self._run_parallel({
            "model": lambda: self._analyze_model(components),
            "doc_knowledge": lambda: self._learn_document_knowledge_enriched(sentences, components),
            "seed": lambda: self._run_seed(sentences, components),
        })

        self.model_knowledge = acq["model"]
        self.doc_knowledge = acq["doc_knowledge"]
        raw_seed_links = acq["seed"]

        ambig = self.model_knowledge.ambiguous_names
        print(f"  Model: {len(ambig)} ambiguous (of {len(components)} components)")
        print(f"  Doc knowledge: {len(self.doc_knowledge.aliases)} aliases")
        print(f"  Seed: {len(raw_seed_links)} raw links")

        self._log("layer1", {"sents": len(sentences), "comps": len(components)},
                  {"ambig": len(ambig), "seed": len(raw_seed_links),
                   "aliases": len(self.doc_knowledge.aliases)})
        self._save_phase(text_path, "layer1", {
            "model_knowledge": self.model_knowledge,
            "doc_knowledge": self.doc_knowledge,
            "raw_seed_links": raw_seed_links,
        })

        print("\n[Tier 2] Link Recovery (parallel)")
        rec = self._run_parallel({
            "seed_val": lambda: self._run_seed_validation(raw_seed_links, components, sent_map),
            "entity": lambda: self._run_entity_pipeline(sentences, components, name_to_id, sent_map),
            "coref": lambda: self._run_coreference(sentences, components, name_to_id, sent_map),
        })

        seed_links = rec["seed_val"]
        validated = rec["entity"]
        coref_links = rec["coref"]
        print(f"  Seed validated: {len(seed_links)} / {len(raw_seed_links)}")
        print(f"  Entity pipeline: {len(validated)} validated")
        print(f"  Coreference: {len(coref_links)} links")

        self._save_phase(text_path, "layer2", {
            "seed_links": seed_links,
            "validated": validated,
            "coref_links": coref_links,
        })

        print("\n[Tier 3] Link Consolidation")
        entity_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, source=c.source)
            for c in validated
        ]
        all_links = seed_links + entity_links + coref_links
        seen: set[tuple] = set()
        final = []
        for lk in all_links:
            key = (lk.sentence_number, lk.component_id)
            if key not in seen:
                seen.add(key)
                final.append(lk)
        print(f"  After dedup: {len(final)} (from {len(all_links)} raw)")

        self._log("summary", {"total_time_s": round(time.time() - t0, 1)},
                  {"final": len(final)}, final)
        self._save_log(text_path)
        self._save_phase(text_path, "final", {"final": final})

        print(f"\nFinal: {len(final)} links ({time.time() - t0:.0f}s)")
        return final

    # ═══════════════════════════════════════════════════════════════════════
    # Tier 1 — Knowledge Acquisition (uses self._SLOT prompts)
    # ═══════════════════════════════════════════════════════════════════════

    def _analyze_model(self, components):
        names = [c.name for c in components]
        knowledge = ModelKnowledge()
        self._classify_components(names, knowledge)
        return knowledge

    def _classify_components(self, names, knowledge):
        prompt = f"""Classify these software architecture component names.

NAMES: {', '.join(names)}

{self._AMBIGUITY_FEW_SHOT}

NOW CLASSIFY THE NAMES ABOVE.

Return JSON:
{{
  "architectural": ["names that identify specific components"],
  "ambiguous": ["names that could easily be used as ordinary words in documentation"]
}}

{self._AMBIGUITY_RULES}

JSON only:"""

        data = None
        for attempt in range(2):
            data = self.llm.extract_json(self.llm.query(prompt, timeout=100))
            if data:
                break
            if attempt == 0:
                print("    Ambiguity classification: empty response, retrying...")
        if data:
            valid = set(names)
            raw_ambiguous = set(data.get("ambiguous", [])) & valid
            knowledge.ambiguous_names = {n for n in raw_ambiguous if len(n.split()) == 1}

    def _learn_document_knowledge_enriched(self, sentences, components):
        comp_names = [c.name for c in components]
        doc_lines = [s.text for s in sentences]

        prompt1 = f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

{self._DOC_KNOWLEDGE_EXTRACTION_RULES}

{self._ALIAS_SCOPE_RULES}

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": [{{"term": "short_form", "component": "FullComponent", "scope": "global"}}],
  "synonyms":      [{{"term": "specific_alternative_name", "component": "FullComponent", "scope": "local"}}]
}}
JSON only:"""

        data1 = None
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
            for rec in abbr_recs + syn_recs:
                if not isinstance(rec, dict):
                    continue
                term = rec.get("term")
                full = rec.get("component")
                scope = rec.get("scope", "local")
                if term and full in comp_names:
                    all_mappings[term] = full
                    all_scopes[term] = scope

        data2 = None
        if all_mappings:
            mapping_list = [f"'{k}' -> {v}" for k, v in list(all_mappings.items())[:25]]
            prompt2 = f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

{self._DOC_KNOWLEDGE_JUDGE_EXAMPLES}

{self._DOC_KNOWLEDGE_JUDGE_RULES}

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

    def _run_seed(self, sentences, components):
        return self._ilinker3.extract(sentences, components)

    # ═══════════════════════════════════════════════════════════════════════
    # Tier 2 — Link Recovery
    # ═══════════════════════════════════════════════════════════════════════

    def _run_seed_validation(self, raw_seed_links, components, sent_map):
        if not raw_seed_links:
            return []

        by_comp: dict[str, list[SadSamLink]] = {}
        for sl in raw_seed_links:
            by_comp.setdefault(sl.component_name, []).append(sl)

        comp_names = get_comp_names(components)
        verified = []

        for comp_name, seeds in sorted(by_comp.items()):
            seed_snums = {sl.sentence_number for sl in seeds}
            profile = build_component_profile(comp_name, self.model_knowledge, self.doc_knowledge)

            anchor_lines = []
            for s in sorted(sent_map.values(), key=lambda x: x.number):
                if s.number in seed_snums:
                    continue
                if has_standalone_mention(comp_name, s.text):
                    anchor_lines.append(f'  S{s.number}: "{s.text}"')
                    if len(anchor_lines) >= 5:
                        break

            if anchor_lines:
                anchor_section = (
                    f'KNOWN REFERENCES (these definitely refer to "{comp_name}"):\n'
                    + "\n".join(anchor_lines) + "\n\n"
                )
            else:
                anchor_section = (
                    f'NOTE: No standalone proper-case references to "{comp_name}" found '
                    f"elsewhere in the document. This component may not be discussed "
                    f"architecturally — be extra careful to verify each case.\n\n"
                )

            case_lines = []
            valid_seeds = []
            for sl in seeds:
                sent = sent_map.get(sl.sentence_number)
                if not sent:
                    continue
                valid_seeds.append(sl)
                prev = sent_map.get(sl.sentence_number - 1)
                prev_text = f' [prev: "{prev.text[:80]}"]' if prev else ""
                match_ctx = self._classify_mention(comp_name, sent.text)
                case_lines.append(
                    f'  Case {len(valid_seeds)} (S{sl.sentence_number}): '
                    f'"{sent.text}"{prev_text}\n    Mention: {match_ctx}'
                )

            if not valid_seeds:
                continue

            prompt = f"""REFERENCE DISAMBIGUATION for component "{comp_name}"

COMPONENT PROFILE:
{profile}

{anchor_section}CASES TO VERIFY:
{chr(10).join(case_lines)}

{self._SEED_DISAMBIGUATION_RULES}

Return JSON:
{{"disambiguations": [{{"case": 1, "meaning": "component", "reason": "brief"}}]}}
JSON only:"""

            data = None
            for attempt in range(2):
                data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
                if data and data.get("disambiguations"):
                    break
                if attempt == 0:
                    print(f"    [{comp_name}] Empty response, retrying...")
            if not data:
                verified.extend(valid_seeds)
                continue

            results: dict[int, dict] = {}
            for d in data.get("disambiguations", []):
                idx = d.get("case", 0) - 1
                results[idx] = d

            approved = 0
            for i, sl in enumerate(valid_seeds):
                r = results.get(i, {})
                meaning = (r.get("meaning", "component") or "component").lower().strip()
                if meaning == "other":
                    reason = r.get("reason", "")
                    print(f"    Seed disambig reject: S{sl.sentence_number} -> {comp_name} ({reason})")
                else:
                    verified.append(sl)
                    approved += 1

            print(f"    [{comp_name}] {approved}/{len(valid_seeds)} seeds kept")

        return [SadSamLink(s.sentence_number, s.component_id, s.component_name, source="seed")
                for s in verified]

    def _classify_mention(self, comp_name: str, text: str) -> str:
        if has_standalone_mention(comp_name, text):
            return "proper case, standalone"
        comp_lower = comp_name.lower()
        if re.search(rf'\b{re.escape(comp_lower)}\b', text):
            for m in re.finditer(rf'\b{re.escape(comp_lower)}\b', text):
                s, e = m.start(), m.end()
                in_dotted = (
                    (s > 0 and text[s - 1] == ".") or
                    (e < len(text) and text[e] == "." and e + 1 < len(text)
                     and text[e + 1].isalpha())
                )
                if in_dotted:
                    return "lowercase, inside dotted path"
            return "lowercase mention"
        if self.doc_knowledge:
            for alias, entry in self.doc_knowledge.aliases.items():
                if entry.component == comp_name and re.search(
                    rf'\b{re.escape(alias)}\b', text, re.IGNORECASE
                ):
                    return f'via known alias "{alias}"'
        return "indirect/unclear match"

    def _build_evidence_bundle(self, candidate, sent_map, rationale="dual-pass extraction consensus"):
        comp_name = candidate.component_name
        snum = candidate.sentence_number
        mention_type = self._classify_mention(comp_name, candidate.sentence_text)
        prev_sent = sent_map.get(snum - 1)
        preceding_text = prev_sent.text if prev_sent else ""
        anchors = []
        for s in sorted(sent_map.values(), key=lambda x: x.number):
            if s.number == snum:
                continue
            if has_standalone_mention(comp_name, s.text):
                anchors.append(f"S{s.number}: {s.text}")
                if len(anchors) >= 5:
                    break
        is_ambig = bool(
            self.model_knowledge
            and self.model_knowledge.ambiguous_names
            and comp_name in self.model_knowledge.ambiguous_names
        )
        return EvidenceBundle(
            source=candidate.source,
            matched_span=candidate.matched_text or comp_name,
            mention_type=mention_type,
            preceding_text=preceding_text,
            anchor_sentences=anchors,
            is_ambiguous=is_ambig,
            extraction_rationale=rationale,
        )

    def _format_evidence(self, bundle: EvidenceBundle) -> str:
        lines = [
            f"  Evidence: source={bundle.source}, span=\"{bundle.matched_span}\", "
            f"mention={bundle.mention_type}, ambiguous={bundle.is_ambiguous}",
            f"  Rationale: {bundle.extraction_rationale}",
        ]
        if bundle.preceding_text:
            lines.append(f"  [prev: \"{bundle.preceding_text[:80]}\"]")
        if bundle.anchor_sentences:
            lines.append("  Anchors (confirmed refs):")
            for a in bundle.anchor_sentences[:3]:
                lines.append(f"    {a[:100]}")
        return "\n".join(lines)

    def _run_entity_pipeline(self, sentences, components, name_to_id, sent_map):
        candidates = self._extract_entities_enriched(sentences, components, name_to_id, sent_map)
        print(f"    Entity extraction: {len(candidates)} candidates")
        bundles = {
            (c.sentence_number, c.component_id): self._build_evidence_bundle(c, sent_map)
            for c in candidates
        }
        if self._current_text_path:
            self._save_phase(self._current_text_path, "entity_candidates", {
                "entity_candidates": candidates, "bundles": bundles,
            })
        validated, decisions = self._validate_with_evidence(candidates, bundles, components, sent_map)
        print(f"    Validation: {len(validated)} / {len(candidates)}")
        if self._current_text_path:
            self._save_phase(self._current_text_path, "entity_decisions", {"decisions": decisions})
        return validated

    def _run_coreference(self, sentences, components, name_to_id, sent_map):
        anaphoric_count = sum(1 for s in sentences if self.PRONOUN_PATTERN.search(s.text))
        print(f"    Coreference: cases-in-context ({anaphoric_count} anaphoric sents / {len(sentences)} total)")
        return self._coref_cases_in_context(sentences, components, name_to_id, sent_map)

    def _run_single_extraction_pass(self, sentences, comp_names, mappings,
                                     name_to_id, sent_map, pass_label=""):
        batch_size = 50
        candidates: dict = {}
        for batch_start in range(0, len(sentences), batch_size):
            batch = sentences[batch_start:batch_start + batch_size]
            if len(sentences) > batch_size:
                print(f"    {pass_label}Entity batch {batch_start//batch_size + 1}: "
                      f"S{batch[0].number}-S{batch[-1].number} ({len(batch)} sents)")
            prompt = f"""Extract ALL references to software architecture components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings[:20])}' if mappings else ''}

{self._ENTITY_EXTRACTION_RULES}

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence"}}]}}
JSON only:"""

            data = None
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
                    candidates[key] = CandidateLink(snum, sent.text, cname, name_to_id[cname],
                                               matched, source="entity")
        return candidates

    def _extract_entities_enriched(self, sentences, components, name_to_id, sent_map):
        comp_names = get_comp_names(components)
        mappings = (
            [f"{term}={entry.component}" for term, entry in self.doc_knowledge.aliases.items()
             if entry.scope == "global"]
            if self.doc_knowledge else []
        )
        print("    Extraction pass A + B (parallel):")
        results = self._run_parallel({
            "pass1": lambda: self._run_single_extraction_pass(
                sentences, comp_names, mappings, name_to_id, sent_map, pass_label="[P1] "),
            "pass2": lambda: self._run_single_extraction_pass(
                sentences, comp_names, mappings, name_to_id, sent_map, pass_label="[P2] "),
        })
        pass1, pass2 = results["pass1"], results["pass2"]
        intersected = {key: pass1[key] for key in pass1 if key in pass2}
        print(f"    Extraction consensus: Pass1={len(pass1)}, Pass2={len(pass2)}, "
              f"Intersect={len(intersected)} (dropped {len(pass1) + len(pass2) - 2*len(intersected)} unique-to-one-pass)")
        return list(intersected.values())

    def _validate_with_evidence(self, candidates, bundles, components, sent_map):
        if not candidates:
            return [], {}

        comp_names = get_comp_names(components)
        decisions: dict = {}

        generic_candidates: dict[str, list] = {}
        non_generic = []
        for c in candidates:
            sent = sent_map.get(c.sentence_number)
            if not sent:
                non_generic.append(c)
                continue
            comp_lower = c.component_name.lower()
            has_exact_case = has_standalone_mention(c.component_name, sent.text)
            has_lowercase = (not has_exact_case and
                             re.search(rf'\b{re.escape(comp_lower)}\b', sent.text))
            if has_lowercase and self.model_knowledge \
                    and self.model_knowledge.ambiguous_names \
                    and c.component_name in self.model_knowledge.ambiguous_names:
                generic_candidates.setdefault(c.component_name, []).append(c)
            else:
                non_generic.append(c)

        remaining = list(non_generic)
        for comp_name, cands in generic_candidates.items():
            anchor_lines = []
            for s in sent_map.values():
                if has_standalone_mention(comp_name, s.text):
                    anchor_lines.append(f"  S{s.number}: {s.text}")
                    if len(anchor_lines) >= 5:
                        break
            case_lines = []
            for i, c in enumerate(cands):
                s = sent_map.get(c.sentence_number)
                prev = sent_map.get(c.sentence_number - 1)
                prev_text = f" [prev: {prev.text[:60]}]" if prev else ""
                case_lines.append(f"  Case {i+1} (S{c.sentence_number}): {s.text}{prev_text}")
            anchor_section = (
                f'FULL-NAME REFERENCES (these definitely refer to the {comp_name} component):\n'
                + '\n'.join(anchor_lines) + '\n\n'
                if anchor_lines else ""
            )
            prompt = f"""CONTEXTUAL WORD USAGE: Does the word refer to the architecture component "{comp_name}", or is it used as an ordinary English word?

{anchor_section}SENTENCES TO CHECK (the component name appears only in lowercase or as part of a compound phrase):
{chr(10).join(case_lines)}

For each case, determine:
- COMPONENT: The word refers to the specific "{comp_name}" component as a system entity
  (e.g., "the {comp_name.lower()} handles requests" = component reference)
- GENERIC: The word is used as ordinary English describing a general concept, activity, or modifier
  (e.g., "provides {comp_name.lower()} access" or "{comp_name.lower()} operations" = generic usage)

Key distinction: A component reference names a specific system entity as a participant.
A generic use describes a type of activity or quality that happens to share the word.
{self._slot_text("GENERIC_WORD_USAGE_RULES")}
Return JSON:
{{"results": [{{"case": 1, "usage": "component" or "generic", "reason": "brief"}}]}}
JSON only:"""

            data = None
            for attempt in range(2):
                data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
                if data and data.get("results"):
                    break
                if attempt == 0:
                    print(f"    Generic filter [{comp_name}]: empty response, retrying...")
            if not data:
                remaining.extend(cands)
                continue
            results_map: dict[int, dict] = {}
            for r in data.get("results", []):
                idx = r.get("case", 0) - 1
                results_map[idx] = r
            for i, c in enumerate(cands):
                result = results_map.get(i, {})
                usage = (result.get("usage", "component") or "component").lower()
                key = (c.sentence_number, c.component_id)
                if usage == "generic":
                    reason = result.get("reason", "")
                    print(f"    LLM generic reject: S{c.sentence_number} -> {c.component_name} ({reason})")
                    decisions[key] = {"approved": False, "path": f"generic_filter: {reason}"}
                else:
                    remaining.append(c)

        print(f"    LLM 2-pass validation (+evidence bundles): {len(remaining)} candidates")
        twopass_approved = []
        for batch_start in range(0, len(remaining), 25):
            batch = remaining[batch_start:batch_start + 25]
            cases = []
            for i, c in enumerate(batch):
                prev = sent_map.get(c.sentence_number - 1)
                p = f"[prev: {prev.text[:60]}] " if prev else ""
                bundle = bundles.get((c.sentence_number, c.component_id))
                evidence_block = self._format_evidence(bundle) if bundle else ""
                case_text = (
                    f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n'
                    f'  {p}"{c.sentence_text}"\n'
                    f'{evidence_block}'
                )
                cases.append((case_text, c))

            case_strings = [ct for ct, _ in cases]
            r1 = self._run_validation_pass(comp_names, case_strings,
                "Check architectural participation: does the sentence name this component as an architectural participant — performing operations, providing services, or taking part in the described system behavior?")
            r2 = self._run_validation_pass(comp_names, case_strings,
                "Check referential specificity: is the component name used to identify this specific architectural element, or does it serve as a generic technical term in this sentence?")

            for i, (case_text, c) in enumerate(cases):
                p1 = r1.get(i, False)
                p2 = r2.get(i, False)
                approved = p1 and p2
                key = (c.sentence_number, c.component_id)
                decisions[key] = {
                    "approved": approved, "p1": p1, "p2": p2,
                    "path": "twopass" if approved else "twopass_reject",
                }
                if approved:
                    twopass_approved.append(c)

        return twopass_approved, decisions

    def _run_validation_pass(self, comp_names, cases, focus):
        prompt = f"""Validate component references in a software architecture document. {focus}

COMPONENTS: {', '.join(comp_names)}

{self._VALIDATION_RULES}

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true}}]}}
JSON only:"""

        data = None
        for attempt in range(2):
            data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
            if data and data.get("validations"):
                break
            if attempt == 0:
                print(f"    Validation pass: empty response, retrying...")
        results: dict[int, bool] = {}
        if data:
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    val = v.get("approve", False)
                    results[idx] = val is True or (isinstance(val, str) and val.lower() == "true")
        return results

    def _classify_specific_terminals(self, components) -> set[str]:
        """LLM-driven: which multi-word component terminal words are specific enough for role-ref coref.

        Avoids generic architectural nouns (service, manager, handler...) that match
        every sentence. Cached per component-list within a linker instance (called once per run).
        """
        multi_word = [c for c in components if len(c.name.split()) > 1]
        if not multi_word:
            return set()
        terminal_words = sorted({c.name.split()[-1].lower() for c in multi_word})
        component_names = [c.name for c in multi_word]

        cache_key = tuple(terminal_words)
        if not hasattr(self, "_terminal_cache"):
            self._terminal_cache: dict = {}
        if cache_key in self._terminal_cache:
            return self._terminal_cache[cache_key]

        prompt = f"""Architecture components have multi-word names. Identify which terminal words (last word of each name) are SPECIFIC enough to serve as unambiguous role references in technical documentation.

Component names: {component_names}
Terminal words to evaluate: {terminal_words}

A terminal word is GENERIC if it could refer to any component in any system on its own — such as words that broadly describe a role or architectural tier.
A terminal word is SPECIFIC if it is distinctive or unusual enough that "the <word>" in a sentence about this system most likely refers to one specific component.
Also mark a terminal as GENERIC if multiple components in this list share the same terminal word (ambiguous).
{self._slot_text("COREF_TERMINAL_SPECIFICITY_RULES")}
Return JSON:
{{"specific": ["word1", "word2"], "generic": ["word3", "word4"]}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=60))
        if not data:
            print("  [coref] WARNING: terminal classification failed, role refs disabled")
            self._terminal_cache[cache_key] = set()
            return set()
        valid = set(terminal_words)
        result = {w.lower() for w in data.get("specific", []) if w.lower() in valid}
        self._terminal_cache[cache_key] = result
        return result

    def _coref_cases_in_context(self, sentences, components, name_to_id, sent_map):
        comp_names = get_comp_names(components)
        all_coref = []

        # Build role_ref_pat: matches "the <terminal_word>" for multi-word component names.
        # Uses LLM-driven specificity check to exclude generic architectural nouns
        # (e.g. "service", "manager") that would match unrelated sentences.
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
                if not (has_standalone_mention(comp, ant_sent.text) or res.get("antecedent_via_alias", False)):
                    continue
                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, source="coreference"))

        return all_coref

    # ═══════════════════════════════════════════════════════════════════════
    # Checkpoint & Logging (verbatim from s_linker13_clean_v3)
    # ═══════════════════════════════════════════════════════════════════════

    def _checkpoint_dir(self, text_path):
        cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
        ds = os.path.splitext(os.path.basename(text_path))[0]
        d = os.path.join(cache_dir, self._VARIANT_NAME, ds)
        assert self._VARIANT_NAME in d, (
            f"_checkpoint_dir must contain _VARIANT_NAME "
            f"('{self._VARIANT_NAME}' not in '{d}')"
        )
        os.makedirs(d, exist_ok=True)
        return d

    def _save_phase(self, text_path, phase_name, state):
        d = self._checkpoint_dir(text_path)
        path = os.path.join(d, f"{phase_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"  Checkpoint: {phase_name} saved")

    def _log(self, phase, input_summary, output_summary, links=None):
        entry = {"phase": phase, "ts": time.time(), "in": input_summary, "out": output_summary}
        if links is not None:
            entry["links"] = [
                {"s": l.sentence_number, "c": l.component_name, "src": l.source}
                for l in links
            ]
        self._phase_log.append(entry)

    def _save_log(self, text_path):
        log_dir = os.environ.get("LLM_LOG_DIR", "./results/llm_logs")
        os.makedirs(log_dir, exist_ok=True)
        ds = os.path.splitext(os.path.basename(text_path))[0]
        path = os.path.join(
            log_dir,
            f"{self._VARIANT_NAME}_{ds}_{time.strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(path, "w") as f:
            json.dump(self._phase_log, f, indent=2, default=str)
        print(f"  Phase log saved: {path}")
