#!/usr/bin/env python3
"""Lightweight ablation runner for the retained ILinker and S-Linker families."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))


def load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


load_dotenv()

from llm_sad_sam.core import DocumentLoader, SadSamLink
from llm_sad_sam.llm_client import LLMBackend, LLMClient
from llm_sad_sam.pcm_parser import parse_pcm_repository


CANONICAL_VARIANTS = [
    "i1",
    "i2",
    "s_linker",
    "s_linker2",
    "s_linker3",
    "s_linker4",
    "s_linker5",
    "s_linker6",
    "s_linker7",
    "s_linker7a",
    "s_linker7b",
    "s_linker8",
    "s_linker9",
    "s_linker9a",
    "s_linker9b",
    "s_linker9c",
    "s_linker9d",
    "s_linker9e",
    "s_linker10",
    "s_linker10a",
    "s_linker11",
    "s_linker11a",
    "s_linker11b",
    "s_linker11c",
    "s_linker11d",
    "s_linker11e",
    "s_linker12a",
    "s_linker12b",
    "s_linker12c",
    "s_linker12d",
    "s_linker12e",
    "s_linker13a",
    "s_linker13b",
    "s_linker13c",
    "s_linker13d",
    "s_linker13e",
    "s_linker13f",
    "s_linker13",   # canonical promotion of 13f (Phase 5)
    "s_linker13_clean",   # v2.1 scaffolding sibling of s_linker13 (Phase 10, CLEAN-01)
    "s_linker13_clean_v3",   # Phase 12 Step 0: prompts_v3 sibling of s_linker13_clean (PROMPT-01)
    "s_linker13_trim1_judge_clean",   # Phase 12 Step 1: alias-judge trim (Technique 3 + 8) on s_linker13_clean (PROMPT-01, PROMPT-02)
    "s_linker13_trim2_entval_clean",   # Phase 12 Step 2: ENTITY_EXTRACTION_RULES + VALIDATION_RULES merged via Technique 3 (PROMPT-02)
    "s_linker13_trim3_runtime_rubric_clean",   # Phase 12 Step 3: inference-time rubric replaces DOC_KNOWLEDGE_JUDGE_RULES (PROMPT-01, PROMPT-02)
    "s_linker13_trim4_ambiguity_runtime_clean",   # Phase 12 EXTENSION (Plan 12-07): runtime rubric replaces AMBIGUITY_FEW_SHOT + AMBIGUITY_RULES
    "s_linker13_trim5_extraction_runtime_clean",   # Phase 12 EXTENSION (Plan 12-08): runtime rubric replaces DOC_KNOWLEDGE_EXTRACTION_RULES
    "s_linker13_trim6_judge_examples_runtime_clean",   # Phase 12 EXTENSION (Plan 12-09): runtime examples replace DOC_KNOWLEDGE_JUDGE_EXAMPLES (uses trim1 distilled rules)
    "s_linker13_trim7_entity_runtime_clean",   # Phase 12 EXTENSION (Plan 12-10): runtime rubric replaces ENTITY_EXTRACTION_RULES
    "s_linker13_trim8_validation_runtime_clean",   # Phase 12 EXTENSION (Plan 12-11): runtime rubric replaces VALIDATION_RULES
    "s_linker13_trim9_seed_runtime_clean",   # Phase 12 EXTENSION (Plan 12-12): runtime rubric replaces SEED_DISAMBIGUATION_RULES
    "s_linker13_min",   # Phase 13 Plan 13-01: composed promotion candidate (trim1 distilled judge + trim9 runtime seed; PROMPT-03, GATE-03)
    "s_linker13_skill_learned_clean",   # Phase 12 EXTENSION (Voyager Pilot): axiom prompts + learned skill bank
    "s_linker13g_pre",   # EXT-01 sub-variant (a): regex pre-filter + LLM judge
    "s_linker13g_sem",   # EXT-01 sub-variant (b): LLM-only, dotted-path encoded in prompt
    "s_linker13g_pre_alias",   # EXT-01 alias-aware (a): regex pre-filter + LLM judge + alias map (Plan 06-06 / D-07)
    "s_linker13g_sem_alias",   # EXT-01 alias-aware (b): LLM-only + alias map
    "s_linker13g_pre_full",    # EXT-01 full-knowledge (a): regex pre-filter + LLM judge + alias + linkmap
    "s_linker13g_sem_full",    # EXT-01 full-knowledge (b): LLM-only + alias + linkmap
    "s_linker14_probe_b_preamble_clean",   # v2.2 PROBE WAVE — Probe B (Phase 15): preamble + cached rubric (INFER-01)
    "s_linker14_probe_c_selfrefine_clean",   # v2.2 PROBE WAVE — Probe C (Phase 16): 2-iter Self-Refine on alias judge (INFER-02)
    "s_linker14_probe_d_upstream_clean",   # v2.2 OPT-IN CARVE-OUT (gpt-5.4 only, shipped 2026-06-01): runtime coref rubric replaces COREF_RULES; NOT canonical
    "s_linker14_voyager",   # v2.3 β architecture consumer (experimental=True): axiom prompts + trained slot-uniform bank patterns; NOT canonical
    "s_linker15",   # v2.6.1 no-training axiom-only linker (experimental=True): inlined axioms + 3 FP fixes, no bank/training; NOT canonical
    "s_linker15b",  # v2.6.1 alias-recovery variant of s_linker15: entity pipeline replaced by targeted alias scan + seed_validation reuse; NOT canonical
    "s_linker15c",  # v2.6.1 ILinker4-entity hybrid: entity pipeline extraction replaced by ILinker4 Pass A+B with alias injection; intersection + validation unchanged; NOT canonical
    "s_linker17a",  # v2.6.1 Multi-Framing renaming variant of s_linker15: same logic, ICSE-friendly method names (Framing A+B+C taxonomy); NOT canonical
    "s_linker17b",  # v2.6.1 unified multi-framing k=2: sequential alias discovery then parallel Framings A/B/C, k≥2 voting merge + unified evidence-bundle validation; NOT canonical
    "s_linker17c",  # v2.6.2 unified multi-framing union: same as 17b but union merge (k=1) — validation is sole quality gate; NOT canonical
    "s_linker17d",  # v2.6.2 17c + validated-antecedent coref gate: coref only fires for components with ≥1 validated link; targets JAB coref over-firing; NOT canonical
    "s_linker17e",  # v2.6.2 17c + validated coref: Phase 5 coref output passes single-pass validation before Phase 6; all links share same quality gate; NOT canonical
    "s_linker17f",  # v2.6.2 17e + Phase 4b code-path filter: drops dotted package-path-only multi_framing links (logic.api, x.e2e); clean full-LLM, no regex; NOT canonical
    "s_linker17g",  # v2.6.2 17f + Framing C union (replaces L3 intersection): Phase 4 is sole gate; removes BBB recall loss from intersection; NOT canonical
    "s_linker18a", # v2.6.3 17f + cleanup F (drop generic-filter): subclass of 17f, single twopass validator path, no CONTEXTUAL WORD USAGE prompt; NOT canonical
    "s_linker18b", # v2.6.3 18a + cleanup E (unify coref validation via entity twopass); subclass of 18a; NOT canonical
    "s_linker18c", # v2.6.3 18b + cleanup C (drop Phase 4b): no separate code-path filter LLM call; twopass p2 catches the FP class via mention_type signal; NOT canonical
    "s_linker18d", # v2.6.3 18c + cleanup B-refactor (alias-aware antecedent check, replaces antecedent_via_alias LLM-flag bypass); NOT canonical
    "s_linker18",  # v2.6.3 18d + cleanup A (enum-based mention classification): clean unified variant; NOT canonical
    "s_linker19",  # v2.6.5 re-baseline: s20's un-minimized parent (paper variant), registered for live N=3 floor re-derivation; NOT canonical
    "s_linker19U",  # v2.6.5 s19 + Framing C 2-pass UNION (full un-minimized prompts incl. few-shots); the un-minimized counterpart of s_linker20_union, registered for the s19U-vs-s20_union few-shot head-to-head; NOT canonical
    "s_linker20",  # v2.6.4 minimized-prompt standalone variant (experimental=True): all constants inlined, no inheritance from s19; NOT canonical
    "s_linker21",  # v2.6.6 CANONICAL Full variant: standalone (s20U union pipeline inlined) + layered no-reasoning validator; supersedes s_linker13_min in reported RQ results; run no-reasoning
    "s_linker21_noknow",  # v2.6.6 RQ4 knowledge A/B: s_linker21 with no_knowledge=True; NOT canonical
    "s_linker21_agentrouter",  # quick-260701-ld4: s_linker21 + bounded-autonomy agentic augmentation pass (experimental=True); NOT canonical
    "s_linker22",  # typed extraction + exact/terminal/no-code validation policy inline in s21 workflow; F2-oriented experimental variant
    "s_linker23",  # LLM-decision-driven augmentation of s21: generic-prompt proposer + agentic router (VALIDATE/CODE/REJECT) floored by s21's real two-pass gate; no structural if/else policy
    "s_linker23_replace",  # s21 pipeline with Phase-2 extraction REPLACED by the batched blocks proposer (through s21's real gate); extraction-integration experiment
    "s_linker23_union",  # s21 pipeline with Phase-2 = Framing-C UNION blocks proposer (integrate all extractors, one gate); extraction-integration experiment
    "s_linker23_verify",  # s23 proposer+router with the VALIDATE floor = s21's REAL evidence-bundle validator (combine s23 recall with s21 precision)
    "s_linker23_verify1p",  # S1: s23_verify with a SINGLE evidence pass (drop P2) — offline +0.024 F1, half the gate API
    "s_linker23_verify1p_all",  # S1 extended: drop P2 also in s21's Framing-C Phase-4 floor (single-pass everywhere)
    "s_linker23_ctx",  # s23_verify + proposer conditioned on s21's per-sentence links as LLM context (residual extraction, no coded heuristics)
    "s_linker23_tier_f1",  # tiered ranking (not binary gate): emit FIRM+PROBABLE tiers; F1 operating point
    "s_linker23_tier_f2",  # tiered ranking: emit FIRM+PROBABLE+WEAK; recall/F2 operating point
    "s_linker24",  # anchored sibling/prefix recovery over unchanged S21 floor

    "s_linker20_aliasa",  # v2.6.5 quick-260610-lio: s20 + ANTECEDENT_ALIAS_RULES few-shot CUT; NOT canonical
    "s_linker20_aliasb",  # v2.6.5 quick-260610-lio: s20 + ANTECEDENT_ALIAS_RULES hardware-domain example (non-SE); NOT canonical
    "s_linker20_ablcorefall",  # v2.6.5 ablation probe (experimental=True): s20 + full coref-family revert (COR-01/02/03/04 + VAL-03); NOT canonical
    "s_linker20_ablgate",  # v2.6.5 ablation probe (experimental=True): s20 + COREF_VALIDATION_FOCUS revert only (VAL-03); NOT canonical
    "s_linker20_ablrules",  # v2.6.5 ablation probe (experimental=True): s20 + COREF_RULES revert only (COR-01/02); NOT canonical
    "s_linker20_ablopener",  # v2.6.5 ablation probe (experimental=True): s20 + _prompt_coref opener/inline revert only (COR-03/04); NOT canonical
    "s_linker20_abldrop",  # v2.6.5 ablation probe (experimental=True): s20 + drop-by-empty few-shots restored (AMB-01 + DKJ-01); NOT canonical
    "s_linker20_ablpleonasm",  # v2.6.5 ablation probe (experimental=True): s20 + 5 generality/jargon cuts reverted (AMB-02/EXT-01/VAL-01/VAL-02/DKJ-07); NOT canonical
]

VARIANT_SPECS = {
    "i1": dict(
        aliases=("ilinker1",),
        module="llm_sad_sam.linkers.experimental.ilinker1",
        class_name="ILinker1",
        description="ILinker1 three-pass precision cascade",
    ),
    "i2": dict(
        aliases=("ilinker2",),
        module="llm_sad_sam.linkers.experimental.ilinker2",
        class_name="ILinker2",
        description="ILinker2 two-pass explicit extractor",
    ),
    "s_linker": dict(
        aliases=("s_linker1",),
        module="llm_sad_sam.linkers.experimental.s_linker",
        class_name="SLinker",
        description="S-Linker base DAG pipeline",
    ),
    "s_linker2": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker2",
        class_name="SLinker2",
        description="S-Linker2",
    ),
    "s_linker3": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker3",
        class_name="SLinker3",
        description="S-Linker3",
    ),
    "s_linker4": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker4",
        class_name="SLinker4",
        description="S-Linker4",
    ),
    "s_linker5": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker5",
        class_name="SLinker5",
        description="S-Linker5",
    ),
    "s_linker6": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker6",
        class_name="SLinker6",
        description="S-Linker6",
    ),
    "s_linker7": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker7",
        class_name="SLinker7",
        description="S-Linker7",
    ),
    "s_linker7a": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker7a",
        class_name="SLinker7a",
        description="S-Linker7a",
    ),
    "s_linker7b": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker7b",
        class_name="SLinker7b",
        description="S-Linker7b",
    ),
    "s_linker8": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker8",
        class_name="SLinker8",
        description="S-Linker8",
    ),
    "s_linker9": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker9",
        class_name="SLinker9",
        description="S-Linker9",
    ),
    "s_linker9a": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker9a",
        class_name="SLinker9a",
        description="S-Linker9a",
    ),
    "s_linker9b": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker9b",
        class_name="SLinker9b",
        description="S-Linker9b",
    ),
    "s_linker9c": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker9c",
        class_name="SLinker9c",
        description="S-Linker9c",
    ),
    "s_linker9d": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker9d",
        class_name="SLinker9d",
        description="S-Linker9d",
    ),
    "s_linker9e": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker9e",
        class_name="SLinker9e",
        description="S-Linker9e",
    ),
    "s_linker10": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker10",
        class_name="SLinker10",
        description="S-Linker10",
    ),
    "s_linker10a": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker10a",
        class_name="SLinker10a",
        description="S-Linker10a",
    ),
    "s_linker11": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker11",
        class_name="SLinker11",
        description="S-Linker11",
    ),
    "s_linker11a": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker11a",
        class_name="SLinker11a",
        description="S-Linker11a",
    ),
    "s_linker11b": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker11b",
        class_name="SLinker11b",
        description="S-Linker11b: alias stratification (strong global / weak local)",
    ),
    "s_linker11c": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker11c",
        class_name="SLinker11c",
        description="S-Linker11c: evidence bundles + structured debate on rejects",
    ),
    "s_linker11d": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker11d",
        class_name="SLinker11d",
        description="S-Linker11d: no partial injection (ablation)",
    ),
    "s_linker11e": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker11e",
        class_name="SLinker11e",
        description="S-Linker11e: evidence bundles in validation, no debate",
    ),
    "s_linker12a": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker12a",
        class_name="SLinker12a",
        description="S-Linker12a: alias stratification + no partial injection",
    ),
    "s_linker12b": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker12b",
        class_name="SLinker12b",
        description="S-Linker12b: alias stratification + evidence bundles (ICSE)",
    ),
    "s_linker12c": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker12c",
        class_name="SLinker12c",
        description="S-Linker12c: 12b - dead Tier 2, intersection voting",
    ),
    "s_linker12d": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker12d",
        class_name="SLinker12d",
        description="S-Linker12d: 12c + trailing-word enrichment (separate step)",
    ),
    "s_linker12e": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker12e",
        class_name="SLinker12e",
        description="S-Linker12e: 12c + merged trailing-word enrichment",
    ),
    "s_linker13a": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13a",
        class_name="SLinker13a",
        description="S-Linker13a: 12c - _split_component_name (Spike 001 LLM trailing-word)",
    ),
    "s_linker13b": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13b",
        class_name="SLinker13b",
        description="S-Linker13b: 12c - _is_structurally_unambiguous (trust LLM ambiguity classification)",
    ),
    "s_linker13c": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13c",
        class_name="SLinker13c",
        description="S-Linker13c: 13b - _is_ambiguous_name_component (inlined dict-set lookup)",
    ),
    "s_linker13d": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13d",
        class_name="SLinker13d",
        description="S-Linker13d: 13b + Spike 003 LLM mention-type enum (no new LLM call)",
    ),
    "s_linker13e": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13e",
        class_name="SLinker13e",
        description="S-Linker13e: 13c - _is_strong_alias - _get_strong_alias_mappings (LLM emits alias scope per record)",
    ),
    "s_linker13f": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13f",
        class_name="SLinker13f",
        description="S-Linker13f: 13e - _has_strong_alias_mention (coref antecedent_via_alias in prompt schema)",
    ),
    "s_linker13": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13",
        class_name="SLinker13",
        description="S-Linker13: canonical promotion of s_linker13f (Phase 5) — 6 rules removed cumulatively from 12c",
        canonical=True,
    ),
    "s_linker13_clean": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13_clean",
        class_name="SLinker13Clean",
        description="S-Linker13 Clean: v2.1 scaffolding sibling — helpers in helper_v3, prompts_v2 unchanged, zero rules removed (CLEAN-01).",
        canonical=False,
    ),
    "s_linker13_clean_v3": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13_clean_v3",
        class_name="SLinker13CleanV3",
        description="S-Linker13 Clean V3: prompts_v3 sibling — Phase 12 Step 0 acceptance variant (byte-equal kept prompts; 7 dead constants dropped).",
        canonical=False,
    ),
    "s_linker13_trim1_judge_clean": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13_trim1_judge_clean",
        class_name="SLinker13Trim1JudgeClean",
        description="S-Linker13 Trim1 — Phase 12 Step 1: DOC_KNOWLEDGE_JUDGE_RULES distilled (Technique 3 + 8); 7 worked examples preserved verbatim",
        canonical=False,
    ),
    "s_linker13_trim2_entval_clean": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13_trim2_entval_clean",
        class_name="SLinker13Trim2EntvalClean",
        description="S-Linker13 Trim2 — Phase 12 Step 2: ENTITY_EXTRACTION_RULES + VALIDATION_RULES merged via Technique 3 (lossless rubric distillation). 10-rule shared core + role-specific extraction/validation headers; 4-rule reduction from 14 → 10.",
        canonical=False,
    ),
    "s_linker13_trim3_runtime_rubric_clean": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13_trim3_runtime_rubric_clean",
        class_name="SLinker13Trim3RuntimeRubricClean",
        description="S-Linker13 Trim3 — Phase 12 Step 3: DOC_KNOWLEDGE_JUDGE_RULES replaced by inference-time rubric builder (AHE + Agentic Rubrics mechanism — supplement Techniques 2+3); 7 worked examples preserved verbatim",
        canonical=False,
    ),
    "s_linker13_trim4_ambiguity_runtime_clean": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13_trim4_ambiguity_runtime_clean",
        class_name="SLinker13Trim4AmbiguityRuntimeClean",
        description="S-Linker13 Trim4 — Phase 12 EXTENSION (Plan 12-07): runtime rubric replaces AMBIGUITY_FEW_SHOT + AMBIGUITY_RULES; no static fallback (RuntimeError on empty rubric)",
        canonical=False,
    ),
    "s_linker13_trim5_extraction_runtime_clean": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13_trim5_extraction_runtime_clean",
        class_name="SLinker13Trim5ExtractionRuntimeClean",
        description="S-Linker13 Trim5 — Phase 12 EXTENSION (Plan 12-08): runtime rubric replaces DOC_KNOWLEDGE_EXTRACTION_RULES; no static fallback",
        canonical=False,
    ),
    "s_linker13_trim6_judge_examples_runtime_clean": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13_trim6_judge_examples_runtime_clean",
        class_name="SLinker13Trim6JudgeExamplesRuntimeClean",
        description="S-Linker13 Trim6 — Phase 12 EXTENSION (Plan 12-09): runtime worked-examples replace DOC_KNOWLEDGE_JUDGE_EXAMPLES; uses trim1's distilled rubric (orthogonal composition); no static fallback",
        canonical=False,
    ),
    "s_linker13_trim7_entity_runtime_clean": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13_trim7_entity_runtime_clean",
        class_name="SLinker13Trim7EntityRuntimeClean",
        description="S-Linker13 Trim7 — Phase 12 EXTENSION (Plan 12-10): runtime rubric replaces ENTITY_EXTRACTION_RULES; built once per document, reused across batches + passes",
        canonical=False,
    ),
    "s_linker13_trim8_validation_runtime_clean": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13_trim8_validation_runtime_clean",
        class_name="SLinker13Trim8ValidationRuntimeClean",
        description="S-Linker13 Trim8 — Phase 12 EXTENSION (Plan 12-11): runtime rubric replaces VALIDATION_RULES; no static fallback (RuntimeError if rubric builder empty after 2 attempts)",
        canonical=False,
    ),
    "s_linker13_trim9_seed_runtime_clean": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13_trim9_seed_runtime_clean",
        class_name="SLinker13Trim9SeedRuntimeClean",
        description="S-Linker13 Trim9 — Phase 12 EXTENSION (Plan 12-12): runtime rubric replaces SEED_DISAMBIGUATION_RULES; built once per document, reused across components; no static fallback",
        canonical=False,
    ),
    "s_linker13_min": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13_min",
        class_name="SLinker13Min",
        description="S-Linker13 Min — Phase 13 Plan 13-01 PROMOTED: composed canonical of trim1 (distilled DOC_KNOWLEDGE_JUDGE_RULES) + trim9 (runtime SEED_DISAMBIGUATION_RULES rubric builder). Claude macro 0.9506 (+1.09pp vs s_linker13_clean baseline); gpt-5.4 macro 0.9069 (≥0.8977 floor). No static fallback for trim9.",
        canonical=True,
    ),
    "s_linker13_skill_learned_clean": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13_skill_learned_clean",
        class_name="SLinker13SkillLearned",
        description="S-Linker13 Skill-Learned — Phase 12 EXTENSION (Voyager Pilot): axiom prompts (prompts_v3_axiom) wrapped at inference time with a JSON skill bank of learned abstract patterns. Skill path is constructor kwarg `skill_path` (defaults to results/voyager_pilot/skill_bank.json).",
        canonical=False,
    ),
    "s_linker13g_pre": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13g_pre",
        class_name="SLinker13gPre",
        description="S-Linker13g-pre: 13 - _has_standalone_mention via LLM with regex pre-filter for dotted-path (EXT-01 sub-variant a)",
    ),
    "s_linker13g_sem": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13g_sem",
        class_name="SLinker13gSem",
        description="S-Linker13g-sem: 13 - _has_standalone_mention via LLM-only (dotted-path encoded in prompt) (EXT-01 sub-variant b)",
    ),
    "s_linker13g_pre_alias": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13g_pre_alias",
        class_name="SLinker13gPreAlias",
        description="S-Linker13g-pre-alias: EXT-01 alias-aware (a): regex pre-filter + LLM judge + alias map injection",
    ),
    "s_linker13g_sem_alias": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13g_sem_alias",
        class_name="SLinker13gSemAlias",
        description="S-Linker13g-sem-alias: EXT-01 alias-aware (b): LLM-only + alias map injection",
    ),
    "s_linker13g_pre_full": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13g_pre_full",
        class_name="SLinker13gPreFull",
        description="S-Linker13g-pre-full: EXT-01 full-knowledge (a): regex pre-filter + LLM judge + alias + linkmap",
    ),
    "s_linker13g_sem_full": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13g_sem_full",
        class_name="SLinker13gSemFull",
        description="S-Linker13g-sem-full: EXT-01 full-knowledge (b): LLM-only + alias + linkmap",
    ),
    "s_linker14_probe_b_preamble_clean": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker14_probe_b_preamble_clean",
        class_name="SLinker14ProbeBPreambleClean",
        description="S-Linker14 Probe B — v2.2 PROBE WAVE (Phase 15, INFER-01): TLR problem-statement preamble + per-dataset cached alias-judge rubric. Forks from s_linker13_clean_v3. 1 rubric LLM call per dataset; cache in results/v2_2_probes/B_preamble_rubric/cache/. GATE-06 fail-loud on rubric leakage.",
        canonical=False,
    ),
    "s_linker14_probe_c_selfrefine_clean": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker14_probe_c_selfrefine_clean",
        class_name="SLinker14ProbeCSelfRefineClean",
        description="S-Linker14 Probe C — v2.2 PROBE WAVE (Phase 16, INFER-02): 2-iter Self-Refine loop on alias judge. Iter 0 verifier emits {verdict, weakness_class}; iter 1 refines only mappings with weakness_class != 'none'. Forks from s_linker13_clean_v3. Iter counts recorded under results/v2_2_probes/C_selfrefine/iter_counts/.",
        canonical=False,
    ),
    "s_linker14_probe_d_upstream_clean": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker14_probe_d_upstream_clean",
        class_name="SLinker14ProbeDUpstreamClean",
        description="S-Linker14 Probe D — v2.2 OPT-IN CARVE-OUT (gpt-5.4 only, shipped 2026-06-01): runtime coref rubric REPLACES static COREF_RULES. NOT canonical — v2.2 canonical remains s_linker13_min unchanged. mediastore gpt-5.4 +1.59pp; BBB gpt-5.4 mean +2.2pp over 2 obs; BBB Claude FAIL was confounded by cross-backend cache reuse (per-backend cache key fix landed). Enable only when LLM_BACKEND==openai. 1 builder LLM call per dataset; cache key per-(text_stem, comp_hash, backend, model); default cache root results/v2_2_probes_range_d_cachefix/cache/ (PROBE_D_CACHE_ROOT env override). Forks from s_linker13_clean_v3. See .planning/v2.2-prep/probe-D-upstream-SUMMARY.md + .planning/v2.2-prep/range-D-bbb-SUMMARY.md + .planning/v2.2-prep/probe-D-cachekey-fix-SUMMARY.md.",
        canonical=False,
    ),
    "s_linker14_voyager": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker14_voyager",
        class_name="SLinker14Voyager",
        description=(
            "S-Linker14 Voyager — v2.3 β architecture consumer (experimental=True, NOT canonical). "
            "Standalone pipeline (no inheritance from s_linker13_clean/v3). "
            "Prompt source: prompts_v3_axiom (9 axiom skeletons). "
            "Slot-uniform trained bank (results/voyager_v4_beta/confirmation/cross_split_final_bank.json — Phase 17 Confirmation cross-split bank) injected at init. "
            "Empty bank = axiom-only floor mode (valid for Phase 14 infra testing + Phase 15 iter-0). "
            "Training harness: scripts/voyager_train_tlr_v4_beta.py (L+O+D-with-CoT-A+P). "
            "Backend: gpt-5.4 (per backend policy). "
            "Promotion bar: STRONG>=0.9173 / WEAK [0.87,0.9173) / FAIL<0.87 on gpt-5.4 macro F1. "
            "s_linker13_min retains canonical=True regardless of outcome. "
            "Bank path override: VOYAGER4B_BANK_PATH env var. "
            "See .planning/v2.3-prep/v2.3-ARCHITECTURE.md + .planning/milestones/v2.3-ROADMAP.md."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker15": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker15",
        class_name="SLinker15",
        description=(
            "S-Linker15 — v2.6.1 no-training axiom-only linker (experimental=True, NOT canonical). "
            "Standalone pipeline (copied from s_linker14_voyager; no inheritance). "
            "ALL Voyager trained-bank machinery removed: no bank, no _wrap, no reload, no training coupling. "
            "Axiom prompts INLINED (prompts_v4_axiom B-variant + three v2.6.1 FP fixes: tier/platform alias, "
            "code-path prefix, functional-alias-as-workflow). Seed extractor: ILinker4 with empty seed rules. "
            "Registered ALONGSIDE s_linker14_voyager (both retained). "
            "Backend: gpt-5.4 (per backend policy). "
            "s_linker13_min retains canonical=True. "
            "See .planning/milestones/v2.6.1-ROADMAP.md."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker15b": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker15b",
        class_name="SLinker15b",
        description=(
            "S-Linker15b — v2.6.1 alias-recovery variant of s_linker15 (experimental=True, NOT canonical). "
            "Replaces full entity pipeline (2-pass LLM scan + 2-pass validation ~8-10 LLM calls) with a "
            "targeted alias-recovery pass: regex-scan sentences for global-scope alias mentions, then reuse "
            "_run_seed_validation for per-component LLM disambiguation (~1-2 LLM calls). No-op for datasets "
            "with no global aliases (MS, JAB). Eliminates entity pipeline FP surface (5 FP on TM) while "
            "preserving alias-based TPs (7 BBB). All other pipeline logic identical to s_linker15. "
            "s_linker13_min retains canonical=True."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker15c": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker15c",
        class_name="SLinker15c",
        description=(
            "S-Linker15c — v2.6.1 ILinker4-entity hybrid (experimental=True, NOT canonical). "
            "Entity pipeline extraction replaced by ILinker4 Pass A (explicit-mention) + Pass B (actor-framed) "
            "with global-scope aliases injected as learned patterns. Intersection merge and 2-pass "
            "evidence-bundle validation unchanged from s_linker15. Hypothesis: ILinker4 framing is more "
            "discriminating than the generic entity extraction prompt, reducing FP generation. "
            "s_linker13_min retains canonical=True."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker17a": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker17a",
        class_name="SLinker17a",
        description=(
            "S-Linker17a — v2.6.1 Multi-Framing renaming variant (experimental=True, NOT canonical). "
            "Conceptually identical to s_linker15; renamed for ICSE paper clarity. "
            "Three linguistic framings: Framing A (explicit-mention, ILinker4 Pass A), "
            "Framing B (actor-role, ILinker4 Pass B), Framing C (alias-aware entity pipeline). "
            "Framings A+B run in Tier 1 (parallel with alias discovery; no alias knowledge). "
            "Framing C runs in Tier 2 (after alias discovery; uses global aliases). "
            "ZERO logic changes vs s_linker15 — renaming only. "
            "s_linker13_min retains canonical=True."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker17b": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker17b",
        class_name="SLinker17b",
        description=(
            "S-Linker17b — v2.6.1 unified multi-framing k=2 (experimental=True, NOT canonical). "
            "Sequential alias discovery (Phase 1) followed by parallel Framings A, B, C (Phase 2), "
            "all with alias knowledge. k-voting merge (k=2, Phase 3): keep link if found by ≥2 framings. "
            "Unified evidence-bundle validation (Phase 4) on all k≥2 candidates. "
            "Eliminates the execution-order artifact of s_linker15/17a where Framings A+B lack alias knowledge. "
            "Framings A+B use ILinker4 Pass A/B directly (no ILinker4 internal merge). "
            "s_linker13_min retains canonical=True."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker17c": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker17c",
        class_name="SLinker17c",
        description=(
            "S-Linker17c — v2.6.2 unified multi-framing union (experimental=True, NOT canonical). "
            "Same as s_linker17b but uses union merge (Phase 3) instead of k=2 voting. "
            "All framing candidates enter unified evidence-bundle validation (Phase 4) — "
            "validation is the sole quality gate, not agreement count. "
            "Recovers recall lost by 17b on TM/BBB (k=2 dropped single-framing TPs). "
            "s_linker13_min retains canonical=True."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker17d": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker17d",
        class_name="SLinker17d",
        description=(
            "S-Linker17d — v2.6.2 17c + validated-antecedent coref gate (experimental=True, NOT canonical). "
            "Identical to s_linker17c except Phase 5 coreference is gated: resolutions are only "
            "accepted for components that already appear in Phase 4 validated output (≥1 validated link). "
            "Targets JAB coref over-firing (17c: 4 FP all from coreference on 13-sentence doc). "
            "Other datasets unaffected if coref TPs come from already-validated components. "
            "s_linker13_min retains canonical=True."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker17e": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker17e",
        class_name="SLinker17e",
        description=(
            "S-Linker17e — v2.6.2 17c + validated coref (experimental=True, NOT canonical). "
            "Identical to s_linker17c except Phase 5 coreference output is passed through "
            "a single-pass validation before Phase 6. All links — framing and coref — share "
            "the same quality gate. Targets coref FPs on ambiguous-name datasets (JAB: 4 FP "
            "coref; TM: 16 FP coref in 17c). "
            "s_linker13_min retains canonical=True."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker17f": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker17f",
        class_name="SLinker17f",
        description=(
            "S-Linker17f — v2.6.2 17e + code-path filter (experimental=True, NOT canonical). "
            "Identical to s_linker17e plus Phase 4b: one narrow full-LLM question drops "
            "multi_framing links whose component name appears only as a segment of a dotted "
            "package path (logic.api, storage.entity, x.e2e). No regex, no per-dataset rules. "
            "Targets the dominant residual FP source (TM: 8 dotted-path FP). Checkpoint: "
            "TM 87.4→92.0, 0 TP killed, macro 93.4→94.3 (Claude). "
            "s_linker13_min retains canonical=True."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker17g": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker17g",
        class_name="SLinker17g",
        description=(
            "S-Linker17g — v2.6.2 17f + Framing C union (experimental=True, NOT canonical). "
            "Identical to s_linker17f except Framing C 2-pass extraction uses UNION instead of "
            "intersection (L3 consensus gate removed). Empirical analysis: L3 intersection "
            "hurt BBB (5 TPs killed, 0 FPs saved) and was redundant with Phase 4 on TeaStore/JabRef. "
            "Phase 4 unified validation is the sole quality gate. "
            "Expected: +5 recall on BBB, minimal precision cost. "
            "s_linker13_min retains canonical=True."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker18a": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker18a",
        class_name="SLinker18a",
        description=(
            "S-Linker18a — v2.6.3 17f + cleanup F (experimental=True, NOT canonical). "
            "Subclass of s_linker17f. Removes the generic-filter pre-pass in Phase 4: all "
            "candidates go directly to twopass validation. The is_ambiguous bundle field "
            "carries ambiguity info into twopass p2. Empirically zero behavior change on "
            "gpt-5.4 (only 1 candidate across 5 projects ever hit generic-filter). "
            "Removes ~80 LOC and one LLM prompt variant."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker18b": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker18b",
        class_name="SLinker18b",
        description=(
            "S-Linker18b — v2.6.3 18a + cleanup E (experimental=True, NOT canonical). "
            "Subclass of s_linker18a. Unifies coref validation with entity twopass — coref "
            "candidates go through the same p1+p2 validator as multi_framing candidates. "
            "Feasibility study: +1 TP, −4 FP across 5 projects vs single-pass."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker18c": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker18c",
        class_name="SLinker18c",
        description=(
            "S-Linker18c — v2.6.3 18b + cleanup C (experimental=True, NOT canonical). "
            "Subclass of s_linker18b. Drops Phase 4b code-path filter entirely. The "
            "evidence bundle's mention_type='lowercase, inside dotted path' signal is "
            "surfaced to twopass p2, which catches 2 of 3 teammates dotted-path FPs. The "
            "third (common.datatransfer→Common) escapes — accept +1 FP for ~50 LOC removed."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker18d": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker18d",
        class_name="SLinker18d",
        description=(
            "S-Linker18d — v2.6.3 18c + cleanup B-refactor (experimental=True, NOT canonical). "
            "Subclass of s_linker18c. Replaces the antecedent_via_alias LLM-flag bypass with "
            "a structural alias-match check on the antecedent sentence. Equivalent recall, "
            "removes a coupling between LLM-emitted metadata and a post-filter regex."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker18": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker18",
        class_name="SLinker18",
        description=(
            "S-Linker18 — v2.6.3 clean unified variant (experimental=True, NOT canonical). "
            "Subclass of s_linker18d. Refactors _classify_mention to return a typed "
            "MentionType enum + structured info. Pure code readability — no behavior change "
            "vs 18d. Final clean variant; the unified design supporting the paper narrative."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker19": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker19",
        class_name="SLinker19",
        description=(
            "S-Linker19 — paper reference variant (experimental=True, NOT canonical). "
            "s20's un-minimized parent; registered in v2.6.5 for live N=3 floor re-derivation. "
            "s_linker19.py is byte-equal frozen (GATE-01) — registration does not modify it."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker19U": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker19U",
        class_name="SLinker19U",
        description=(
            "S-Linker19U — v2.6.5 (experimental=True, NOT canonical). "
            "s_linker19 (FULL un-minimized prompts, incl. the AMBIGUITY/DOC-KNOWLEDGE few-shots) "
            "with the Framing C 2-pass consensus changed from INTERSECTION to UNION (the single "
            "_run_framing_c change). It is the un-minimized counterpart of s_linker20_union: the two "
            "differ ONLY by the 12 Phase-46 prompt cuts (of which the few-shot drops AMB-01/DKJ-01 are "
            "two). Registered for the direct s19U-vs-s20_union head-to-head — does keeping the full "
            "prompts/few-shots beat the minimized prompts under identical union logic?"
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker20_aliasa": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker20_aliasa",
        class_name="SLinker20AliasA",
        description=(
            "S-Linker20AliasA — v2.6.5 quick-260610-lio (experimental=True, NOT canonical). "
            "s20 with the ANTECEDENT_ALIAS_RULES few-shot ('Examples:' block) CUT entirely. "
            "Tests whether the coref antecedent_via_alias few-shot is load-bearing (live behavior)."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker20_aliasb": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker20_aliasb",
        class_name="SLinker20AliasB",
        description=(
            "S-Linker20AliasB — v2.6.5 quick-260610-lio (experimental=True, NOT canonical). "
            "s20 with the ANTECEDENT_ALIAS_RULES few-shot rewritten to a non-SE hardware example "
            "(PowerSupplyUnit/'the unit'/voltage). Tests generality-neutral rewrite vs live behavior."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker21": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker21",
        class_name="SLinker21",
        description=(
            "S-Linker21 — v2.6.6 CANONICAL Full variant (canonical=True), STANDALONE. "
            "The former s20U ('s_linker20_union') union pipeline inlined verbatim (s20U now "
            "removed) plus the spike-004 'v4' layered no-reasoning validation prompt (Mode 5 "
            "forced per-candidate justification + Mode 1 architectural-claim rubric, asymmetric "
            "entity-lenient / coref-strict). Behaviour identical to the prior subclass form. The "
            "paper Full variant, superseding s_linker13_min in reported RQ results. "
            "Run no-reasoning (Sonnet: CLAUDE_DISABLE_THINKING=1; OpenAI: OPENAI_REASONING_EFFORT "
            "unset / none). gpt-5.4 macro 93.2 (+3.8 vs the no-reasoning union baseline), every "
            "dataset up, zero implicit-recall cost. Source: spike 004-nogap-validator-ab."
        ),
        canonical=True,
        experimental=False,
    ),
    "s_linker21_noknow": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker21",
        class_name="SLinker21",
        description=(
            "S-Linker21 NO-KNOWLEDGE — v2.6.6 RQ4 knowledge A/B (experimental=True, NOT "
            "canonical). s_linker21 with no_knowledge=True: skips the alias table + ambiguity "
            "map (layer1 LLM calls), sets empty ModelKnowledge / DocumentKnowledge directly; "
            "canonical-name-only matching. All other phases run unchanged. Source: RQ4-02 "
            "knowledge A/B ablation axis."
        ),
        canonical=False,
        experimental=True,
        kwargs=dict(no_knowledge=True),
    ),
    "s_linker21_agentrouter": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker21_agentrouter",
        class_name="SLinker21AgentRouter",
        description=(
            "S-Linker21 AGENT-ROUTER — quick-260701-ld4 bounded-autonomy agentic "
            "augmentation (experimental=True, NOT canonical). Subclasses s_linker21; "
            "runs the canonical link() unchanged as the floor, then an LLM agent "
            "(DocModelAgenticRouter + GroundedTypedProposer) proposes typed "
            "candidates per sentence and routes each to VALIDATE/CODE/REJECT -- only "
            "VALIDATE candidates the trusted s21 two-pass gate approves are added, so "
            "the result can never regress below s21. Measured (pilot, gpt-5.4): "
            "P 0.9592 / R 0.9247 / F1 0.9402, vs baseline s21 P 0.9894/R 0.8913/F1 0.9360 "
            "and the non-agentic named+routed target P 0.9897/R 0.9173/F1 0.9506 (NOT "
            "shipped). The -1pp F1 vs the named+routed target is 100% verified "
            "gold-incompleteness, not error; gate-floor holds; all 4 core recoveries "
            "kept. CODE-routed candidates are always exposed via "
            "self.code_routed_candidates; judged doc->code links land in "
            "self.code_links only when an acm_path kwarg is supplied (not plumbed by "
            "this harness today). Full narrative archived at "
            ".planning/archive/router-pilot-260701/."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker22": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker22",
        class_name="SLinker22",
        description=(
            "S-Linker22 — experimental F2-oriented successor to s21. Runs the same live "
            "six-phase workflow as s21 (no frozen-output postprocessing). Phase 2 preserves "
            "s21 Framing-C extraction as the live floor, then adds typed references for "
            "floor-missed candidates. Phase 4 applies the pilot's exact/terminal/no-code "
            "policy only to typed-only candidates: AFFIRMATIVE uses s21 P1/P2 after a "
            "generic evidence filter; CONTRAST uses a contrast-specific validator; "
            "IMPLICIT/ANAPHORA/CODEPATH are not accepted as model-doc links. Measured "
            "gpt-5.4 full run: P 0.9779 / R 0.9232 / F1 0.9494 / F2 0.9334."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker23": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker23",
        class_name="SLinker23",
        description=(
            "S-Linker23 — LLM-decision-driven augmentation of s21. Subclasses SLinker21 "
            "(never edits it), runs the canonical s21 pipeline as the floor, then augments: "
            "a generic-prompt GroundedTypedProposer surfaces floor-missed candidates and the "
            "agentic DocModelAgenticRouter lets the LLM choose VALIDATE/CODE/REJECT per "
            "candidate from general guidelines. No structural regex filters, no mode->policy "
            "if/elif: the only non-LLM step is grounding a proposal to a real catalog name. "
            "Bounded autonomy — an accepted link must also pass s21's OWN two-pass entity "
            "gate (injected as the router gate), so it can never regress below the s21 floor."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker23_replace": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker23_extract",
        class_name="SLinker23Replace",
        description=(
            "S-Linker23Replace — s21 pipeline with Phase-2 extraction REPLACED by the "
            "batched `blocks` proposer (per-sentence-shaped context, batch 20), then run "
            "through s21's UNCHANGED two-pass gate / coref / merge. Measures whether the "
            "blocks extractor can stand in for Framing-C end-to-end. Subclass of SLinker21 "
            "(GATE-01 safe)."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker23_union": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker23_extract",
        class_name="SLinker23Union",
        description=(
            "S-Linker23Union — s21 pipeline with Phase-2 = s21 Framing-C UNION the batched "
            "`blocks` proposer, all validated by s21's UNCHANGED gate (one gate, integrate "
            "all extractors). Measures the 'keep both' extraction integration end-to-end. "
            "Subclass of SLinker21 (GATE-01 safe)."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker23_verify": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker23_verify",
        class_name="SLinker23Verify",
        description=(
            "S-Linker23Verify — s23's blocks proposer + agentic router, but the router's "
            "VALIDATE decisions are floored by s21's REAL Phase-4 evidence-bundle two-pass "
            "validator (claim-before-verdict) instead of s23's lightweight case-text gate. "
            "Combines s23's recall augmentation with s21's precision mechanism; tests "
            "whether routing the proposed candidates through full s21 verification fixes the "
            "false-positive leak. Subclass of SLinker23 (GATE-01 safe)."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker23_verify1p": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker23_verify1p",
        class_name="SLinker23Verify1P",
        description=(
            "S-Linker23Verify1P (S1) — s23_verify with a SINGLE evidence pass. The "
            "augmentation gate keeps s21's evidence bundles and claim-before-verdict "
            "prompt but runs only P1, not P1 AND P2. Offline pilot: +0.024 F1 pooled "
            "(the 2nd pass removed ~2x more gold than non-gold on the aug population) "
            "at half the gate API cost. Subclass of SLinker23Verify (GATE-01 safe)."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker23_verify1p_all": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker23_verify1p_all",
        class_name="SLinker23Verify1PAll",
        description=(
            "S-Linker23Verify1PAll — the single-pass simplification extended UNIVERSALLY: "
            "drop s21's Phase-4 second validation pass on the Framing-C floor too, not "
            "only on the augmentation gate. Tests whether the 2nd pass earns its keep on "
            "the base FC candidates. Unlike s_linker23_verify1p this moves the s21 floor "
            "off two-pass, so it is not a pure augmentation. Subclass of SLinker23Verify1P "
            "(GATE-01 safe: overrides the hook, s21 file byte-stable)."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker23_tier_f1": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker23_tiered",
        class_name="SLinker23Tiered",
        description=(
            "S-Linker23Tiered — trace-linking as TIERED RANKING, not binary keep/reject. "
            "Union extraction (Framing-C + alias/sibling blocks proposer) is assigned an "
            "evidence tier from (name-match x gate-votes x source); emits FIRM+PROBABLE "
            "(precision/F1 operating point). Coref + merge inherited. Subclass of "
            "SLinker23Union (GATE-01 safe)."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker23_tier_f2": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker23_tiered",
        class_name="SLinker23TieredF2",
        description=(
            "S-Linker23TieredF2 — same tiered linker at the recall/F2 operating point "
            "(FIRM+PROBABLE+WEAK). Subclass of SLinker23Tiered (GATE-01 safe)."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker24": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker24",
        class_name="SLinker24",
        description=(
            "S-Linker24 — unchanged S21 floor plus a narrow, LLM-resolved recovery "
            "for locally anchored Client/Server siblings and unique technical-prefix "
            "shorthand. Every addition must pass S21's existing strict coreference "
            "validator; no broad proposer or router is used."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker23_ctx": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker23_ctx",
        class_name="SLinker23Ctx",
        description=(
            "S-Linker23Ctx — SLinker23Verify but the proposer is CONDITIONED on s21's own "
            "output: each sentence is shown with the components s21 already linked "
            "(ALREADY LINKED: ...) and the model is asked for what the base MISSED (residual "
            "extraction). Pure LLM-side context conditioning — no coded thresholds/heuristics. "
            "Tests whether feeding the base decisions as context improves the augmentation. "
            "Subclass of SLinker23Verify (GATE-01 safe)."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker20": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker20",
        class_name="SLinker20",
        description=(
            "S-Linker20 — v2.6.4 minimized-prompt standalone variant "
            "(experimental=True, NOT canonical). "
            "Same logic as the preceding paper variant; all prompt constants inlined directly "
            "(no import from prompts_v5). 12 Phase 46 cuts applied: "
            "AMBIGUITY_FEW_SHOT drop, DOC_KNOWLEDGE_JUDGE_EXAMPLES drop, "
            "5 cross-section pleonasm trims (AMB/EXT/VAL openers), "
            "4 lexical-jargon neutralizations (DKJ/VAL/COR). "
            "Target: gpt-5.4 macro F1 >= 91.3% (Phase 48 sweep)."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker20_ablcorefall": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker20_ablcorefall",
        class_name="SLinker20AblCorefAll",
        description=(
            "S-Linker20 ablation probe (v2.6.5 regression bisection; experimental=True, "
            "NOT canonical). Copy of s_linker20 with the full coref-family revert "
            "(COREF_RULES COR-01/02 + _prompt_coref opener/inline COR-03/04 + "
            "COREF_VALIDATION_FOCUS VAL-03 restored to s19 text)."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker20_ablgate": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker20_ablgate",
        class_name="SLinker20AblGate",
        description=(
            "S-Linker20 ablation probe (v2.6.5 regression bisection; experimental=True, "
            "NOT canonical). Copy of s_linker20 with only COREF_VALIDATION_FOCUS "
            "(VAL-03) reverted to s19 text."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker20_ablrules": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker20_ablrules",
        class_name="SLinker20AblRules",
        description=(
            "S-Linker20 ablation probe (v2.6.5 regression bisection; experimental=True, "
            "NOT canonical). Copy of s_linker20 with only COREF_RULES (COR-01/02) "
            "reverted to s19 text."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker20_ablopener": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker20_ablopener",
        class_name="SLinker20AblOpener",
        description=(
            "S-Linker20 ablation probe (v2.6.5 regression bisection; experimental=True, "
            "NOT canonical). Copy of s_linker20 with only the _prompt_coref opener + "
            "inline restatement (COR-03/04) reverted to s19 text."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker20_abldrop": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker20_abldrop",
        class_name="SLinker20AblDrop",
        description=(
            "S-Linker20 ablation probe (v2.6.5 regression bisection; experimental=True, "
            "NOT canonical). Copy of s_linker20 with the drop-by-empty few-shots "
            "restored (AMBIGUITY_FEW_SHOT AMB-01 + DOC_KNOWLEDGE_JUDGE_EXAMPLES DKJ-01)."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker20_ablpleonasm": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker20_ablpleonasm",
        class_name="SLinker20AblPleonasm",
        description=(
            "S-Linker20 ablation probe (v2.6.5 regression bisection; experimental=True, "
            "NOT canonical). Copy of s_linker20 with the 5 remaining generality/jargon "
            "cuts reverted to s19 text (AMB-02 + EXT-01 + VAL-01 + VAL-02 + DKJ-07); "
            "all coref + drop cuts left in s20 form."
        ),
        canonical=False,
        experimental=True,
    ),
}

VARIANTS = {
    canonical: {"canonical": canonical, "description": VARIANT_SPECS[canonical]["description"]}
    for canonical in CANONICAL_VARIANTS
}
for canonical, spec in VARIANT_SPECS.items():
    for alias in spec["aliases"]:
        VARIANTS[alias] = {"canonical": canonical, "description": f"Alias for {canonical}"}

# The replication package vendors the benchmark at its top level.  The
# environment variables preserve support for an external ARDoCo checkout and
# externally generated CLI baseline results.
BENCHMARK_BASE = Path(os.environ.get("ALINKER_BENCHMARK", ROOT.parent / "benchmark"))
CLI_RESULTS = Path(os.environ.get("ALINKER_CLI_RESULTS", ROOT.parent / "cli-results"))

DATASETS = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
        "gold_sam": BENCHMARK_BASE / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
        "transarc_sam": CLI_RESULTS / "mediastore-sad-sam/sadSamTlr_mediastore.csv",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
        "model": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
        "gold_sam": BENCHMARK_BASE / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
        "transarc_sam": CLI_RESULTS / "teastore-sad-sam/sadSamTlr_teastore.csv",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
        "gold_sam": BENCHMARK_BASE / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc_sam": CLI_RESULTS / "teammates-sad-sam/sadSamTlr_teammates.csv",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
        "gold_sam": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc_sam": CLI_RESULTS / "bigbluebutton-sad-sam/sadSamTlr_bigbluebutton.csv",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
        "gold_sam": BENCHMARK_BASE / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc_sam": CLI_RESULTS / "jabref-sad-sam/sadSamTlr_jabref.csv",
    },
}


def get_backend() -> LLMBackend:
    backend_name = os.environ.get("LLM_BACKEND", "openai").strip().lower()
    if backend_name == "openai":
        return LLMBackend.OPENAI
    if backend_name == "checkpoint":
        return LLMBackend.CHECKPOINT
    if backend_name == "codex":
        return LLMBackend.CODEX
    return LLMBackend.CLAUDE


os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.4")
os.environ.setdefault("CLAUDE_MODEL", "sonnet")


def describe_backend_target(backend: LLMBackend | None = None) -> str:
    backend = backend or get_backend()
    if backend == LLMBackend.CLAUDE:
        return f"claude ({os.environ.get('CLAUDE_MODEL', 'sonnet')})"
    if backend == LLMBackend.OPENAI:
        return f"openai ({os.environ.get('OPENAI_MODEL_NAME', 'gpt-5.4')})"
    if backend == LLMBackend.CHECKPOINT:
        fallback_model = os.environ.get("CHECKPOINT_FALLBACK_MODEL", "").strip().lower()
        if fallback_model in {"gpt", "openai"} or fallback_model.startswith("gpt"):
            model = os.environ.get("OPENAI_MODEL_NAME", "gpt-5.4")
            if fallback_model.startswith("gpt"):
                model = fallback_model
            return f"checkpoint -> openai ({model})"
        if fallback_model in {"claude", "sonnet"} or fallback_model.startswith("claude"):
            model = os.environ.get("CLAUDE_MODEL", "sonnet")
            if fallback_model not in {"claude", "sonnet"}:
                model = fallback_model
            return f"checkpoint -> claude ({model})"
        fallback_backend = os.environ.get("CHECKPOINT_FALLBACK", "claude").strip().lower() or "claude"
        if fallback_backend == "openai":
            return f"checkpoint -> openai ({os.environ.get('OPENAI_MODEL_NAME', 'gpt-5.4')})"
        if fallback_backend == "codex":
            return "checkpoint -> codex"
        return f"checkpoint -> claude ({os.environ.get('CLAUDE_MODEL', 'sonnet')})"
    return backend.value


def available_variants() -> list[str]:
    return list(CANONICAL_VARIANTS)


def canonical_variant(name: str) -> str:
    if name not in VARIANTS:
        raise KeyError(name)
    return VARIANTS[name]["canonical"]


def normalize_variants(names: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for name in names:
        canonical = canonical_variant(name)
        if canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized


def build_linker(variant_name: str, backend: LLMBackend | None = None):
    canonical = canonical_variant(variant_name)
    spec = VARIANT_SPECS[canonical]
    module = importlib.import_module(spec["module"])
    cls = getattr(module, spec["class_name"])
    extra = spec.get("kwargs", {})
    return cls(backend=backend or get_backend(), **extra)


def load_gold_sam(gold_path: str) -> set[tuple[int, str]]:
    links: set[tuple[int, str]] = set()
    with open(gold_path) as handle:
        for row in csv.DictReader(handle):
            component_id = row.get("modelElementID", "").strip()
            sentence_number = row.get("sentence", "").strip()
            if component_id and sentence_number:
                links.add((int(sentence_number), component_id))
    return links


def load_transarc_pairs(transarc_path: str) -> set[tuple[int, str]]:
    pairs: set[tuple[int, str]] = set()
    with open(transarc_path) as handle:
        for row in csv.DictReader(handle):
            component_id = row.get("modelElementID", "").strip()
            sentence_number = row.get("sentence", "").strip()
            if component_id and sentence_number:
                pairs.add((int(sentence_number), component_id))
    return pairs


def export_links_csv(links: list[SadSamLink], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sentence", "component_id", "component_name", "confidence", "source"])
        for link in sorted(links, key=lambda item: (item.sentence_number, item.component_id)):
            writer.writerow(
                [
                    link.sentence_number,
                    link.component_id,
                    link.component_name,
                    f"{link.confidence:.2f}",
                    link.source,
                ]
            )


def eval_metrics(predicted: set[tuple[int, str]], gold: set[tuple[int, str]]) -> dict[str, float]:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "P": precision, "R": recall, "F1": f1}


def require_existing(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def run_variant(
    variant_name: str,
    dataset_name: str,
    paths: dict[str, Path],
    gold_pairs: set[tuple[int, str]],
    transarc_pairs: set[tuple[int, str]],
    id_to_name: dict[str, str],
    sent_map: dict[int, object],
    results_dir: Path,
) -> dict[str, object]:
    print(f"\n  --- Variant: {variant_name} ---")
    linker = build_linker(variant_name)

    t0 = time.time()
    predictions = linker.link(
        text_path=str(paths["text"]),
        model_path=str(paths["model"]),
        transarc_csv=str(paths["transarc_sam"]),
    )
    elapsed = time.time() - t0

    predicted_pairs = {(link.sentence_number, link.component_id) for link in predictions}
    prediction_by_key = {(link.sentence_number, link.component_id): link for link in predictions}
    metrics = eval_metrics(predicted_pairs, gold_pairs)

    source_counts: defaultdict[str, int] = defaultdict(int)
    for link in predictions:
        source_counts[link.source] += 1

    fp_pairs = predicted_pairs - gold_pairs
    fp_by_source: defaultdict[str, int] = defaultdict(int)
    fp_details = []
    for sentence_number, component_id in sorted(fp_pairs):
        link = prediction_by_key[(sentence_number, component_id)]
        fp_by_source[link.source] += 1
        sentence = sent_map.get(sentence_number)
        fp_details.append(
            {
                "sentence": sentence_number,
                "component": id_to_name.get(component_id, component_id),
                "source": link.source,
                "confidence": link.confidence,
                "text": sentence.text[:120] if sentence else "",
            }
        )

    fn_pairs = gold_pairs - predicted_pairs
    fn_details = []
    for sentence_number, component_id in sorted(fn_pairs):
        sentence = sent_map.get(sentence_number)
        component_name = id_to_name.get(component_id, component_id)
        fn_details.append(
            {
                "sentence": sentence_number,
                "component": component_name,
                "name_in_text": component_name.lower() in sentence.text.lower() if sentence else False,
                "transarc_had": (sentence_number, component_id) in transarc_pairs,
            }
        )

    export_links_csv(predictions, results_dir / f"{variant_name}_{dataset_name}_links.csv")

    print(
        f"  {variant_name}: P={metrics['P']:.1%} R={metrics['R']:.1%} F1={metrics['F1']:.1%} "
        f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} ({elapsed:.0f}s)"
    )
    print(f"    Sources: {dict(source_counts)}")
    print(f"    FP by source: {dict(fp_by_source)}")

    return {
        "variant": variant_name,
        "P": metrics["P"],
        "R": metrics["R"],
        "F1": metrics["F1"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "n_links": len(predictions),
        "time": elapsed,
        "sources": dict(source_counts),
        "fp_by_source": dict(fp_by_source),
        "fp_details": fp_details,
        "fn_details": fn_details,
    }


def print_summary(all_results: dict[str, dict[str, dict[str, object]]], selected_variants: list[str]) -> None:
    print(f"\n{'=' * 120}")
    print("SUMMARY")
    print(f"{'=' * 120}")
    header = f"{'Dataset':<16}"
    for variant in selected_variants:
        header += f" | {variant:^18}"
    print(header)
    print(f"{'-' * 16}" + ("-+-" + "-" * 18) * len(selected_variants))

    for dataset_name, dataset_results in all_results.items():
        row = f"{dataset_name:<16}"
        for variant in selected_variants:
            result = dataset_results.get(variant)
            if result is None:
                row += " | " + f"{'--':^18}"
            else:
                row += " | " + f"F1 {result['F1']:.1%} FP {result['fp']:>3}"
        print(row)

    print(f"{'-' * 16}" + ("-+-" + "-" * 18) * len(selected_variants))
    row = f"{'Macro avg':<16}"
    for variant in selected_variants:
        values = [all_results[dataset][variant] for dataset in all_results if variant in all_results[dataset]]
        avg_f1 = sum(value["F1"] for value in values) / len(values)
        total_fp = sum(value["fp"] for value in values)
        row += " | " + f"F1 {avg_f1:.1%} FP {total_fp:>3}"
    print(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["s_linker11a"],
        help="Retained variants to evaluate",
    )
    parser.add_argument(
        "--results-dir",
        default="results/ablation_results",
        help="Directory for CSV and JSON output",
    )
    parser.add_argument("--list-datasets", action="store_true", help="Print supported datasets and exit")
    parser.add_argument("--list-variants", action="store_true", help="Print supported variants and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_datasets:
        print("\n".join(DATASETS.keys()))
        return 0
    if args.list_variants:
        print("\n".join(available_variants()))
        return 0

    unknown_datasets = [name for name in args.datasets if name not in DATASETS]
    if unknown_datasets:
        raise SystemExit(f"Unknown datasets: {', '.join(unknown_datasets)}")

    try:
        selected_variants = normalize_variants(args.variants)
    except KeyError as exc:
        raise SystemExit(f"Unknown variant: {exc.args[0]}") from exc

    datasets = {name: DATASETS[name] for name in args.datasets}
    backend = get_backend()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 120}")
    print("ABLATION STUDY: Retained ILinker and S-Linker Variants")
    print(f"Backend: {describe_backend_target(backend)}")
    print(f"Datasets: {', '.join(datasets.keys())}")
    print(f"Variants: {', '.join(selected_variants)}")
    print(f"{'=' * 120}")

    all_results: dict[str, dict[str, dict[str, object]]] = {}

    for dataset_name, paths in datasets.items():
        require_existing(paths["text"], f"{dataset_name} text")
        require_existing(paths["model"], f"{dataset_name} model")
        require_existing(paths["gold_sam"], f"{dataset_name} gold standard")

        print(f"\n{'=' * 120}")
        print(f"DATASET: {dataset_name}")
        print(f"{'=' * 120}")

        components = parse_pcm_repository(str(paths["model"]))
        id_to_name = {component.id: component.name for component in components}
        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        sent_map = {sentence.number: sentence for sentence in sentences}
        gold_pairs = load_gold_sam(str(paths["gold_sam"]))
        transarc_pairs = (
            load_transarc_pairs(str(paths["transarc_sam"]))
            if paths["transarc_sam"].exists()
            else set()
        )

        print(f"  Components: {len(components)}, Sentences: {len(sentences)}")
        print(f"  Gold links: {len(gold_pairs)}, TransArc baseline: {len(transarc_pairs)}")
        if transarc_pairs:
            metrics = eval_metrics(transarc_pairs, gold_pairs)
            print(f"  TransArc baseline: P={metrics['P']:.1%} R={metrics['R']:.1%} F1={metrics['F1']:.1%}")
        else:
            print("  TransArc baseline: (CSV not available)")

        all_results[dataset_name] = {}
        for variant_name in selected_variants:
            result = run_variant(
                variant_name=variant_name,
                dataset_name=dataset_name,
                paths=paths,
                gold_pairs=gold_pairs,
                transarc_pairs=transarc_pairs,
                id_to_name=id_to_name,
                sent_map=sent_map,
                results_dir=results_dir,
            )
            all_results[dataset_name][variant_name] = result

    print_summary(all_results, selected_variants)

    json_path = results_dir / f"ablation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with json_path.open("w") as handle:
        json.dump(all_results, handle, indent=2, default=str)
    print(f"\nResults saved to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
