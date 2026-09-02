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
    "s_linker24_role_orchestrator",  # latest S24: evidence-grounded replacement orchestrator
    "s_linker25",  # fixed name-evidence order (full-name, partial-name, coreference); no controller
    "s_linker26",  # s25 with alias discovery folded into the reading pass
    "s_linker27",  # s26 without batching: one call reads the whole document
    "s_linker28",  # s26 with the alias table no longer suppressing partial names
    "s_linker29",  # s25 with the alias judge replaced by a grounding check
    "s_linker30",  # s25 with alias judging folded into the extraction pass
    "s_linker31",  # s25 with the alias pass proposing and judging in one call
    "s_linker32",  # s25 with the alias judging carried by the extraction pass
    "s_linker33",  # s32 with the carried judging decided by majority vote
    "s_linker34",  # s32 with the carried judging requiring unanimity
    "s_linker35",  # s34 with the carried judging asked before the document
    "s_linker36",  # s25 with the full-name judge asking both criteria in one call
    "s_linker37",  # s36 with a quote committed per criterion inside the one call
    "s_linker38",  # one judging prompt sampled twice, verdicts ANDed
    "s_linker39",  # two alias judges: dedicated validity + carried usage, unioned
    "s_linker40",  # s38's single link-judging prompt + the two alias judges
    "s_linker42",  # s36 plus a three-value mention label (audit-supported)
    "s_linker43",  # s25 plus the three-value mention label, nothing else
    "s_linker44",  # s25 with only the label's case grading merged away
    "s_linker45",  # s25 with the coreference batch set to the judges' batch
    "s_linker46",  # s25 with the alias table no longer suppressing partial names
    "s_linker47",  # s25 with the grounded identity review removed
    "s_linker48",  # s25 with five near-duplicate conditions merged into two
    "s_linker49",  # s47 and s48 composed: one judging step each, two predicates
    "s_linker50",  # s49 with the coreference rule stated as one guideline
    "s_linker51",  # s49 with all nine enumerated rules stated as guidelines
    "s_linker52",  # s51 with the three knowledge-side rules reverted
    "s_linker53",  # s51 with one clause of the alias judge restored
    "s_linker49_null",  # byte-identical to s49: the harness null arm
    "s_linker54",  # s51 with the full-name family reverted
    "s_linker55",  # s51 with only the coreference side generalized
    "s_linker56",  # s55 with the coreference prompt de-duplicated
    "s_linker57",  # s55 + the alias proposer generalized, judge kept
    "s_linker58",  # s55 + the full-name proposer generalized, judges kept
    "s_linker59",  # every clause single-stage ablation clears, and no others
    "s_linker60",  # alias proposal folded into the reading, judge kept separate
    "s_linker61",  # s60 + the qualified-name exclusion stated to the judge
    "s_linker59_null",  # byte-identical to s59: the harness null arm
    "s_linker62",  # s59 + inflection-bounded partial-name proposer
    "s_linker63",  # s62 + the boundary defect in the span test repaired
    "s_linker64",  # s62 + the stated-name net at the full-name proposer
    "s_linker65",  # s64 with its four lexical rules restated as one relation
    "s_linker65_null",  # byte-identical to s65: the harness null arm
    "s_linker66",  # s65 with the admission contract stated in the extraction prompt
    "s_linker67",  # s66 with the two tight scans stated there too
    "s_linker68",  # s66 without the mention label's qualified-path value
    "s_linker66_null",  # byte-identical to s66: the harness null arm
    "s_linker69",  # s66 with the gates that do not decide folded or deleted
    "s_linker70",  # s69 with the spelling row's two options folded and deleted
    "s_linker71",  # s70 with the authored prompts restated as general principles
    "s_linker72",  # s71 with the extraction half reverted: the judging rubric only
    "s_linker73",  # s70 with only the two GATE-07-inadmissible spans reworded
    "s_linker74",  # s70 with only the identifier syntax unspelled
    "s_linker75",  # s74 with every remaining corpus-shaped prompt span gone
    "s_linker75_null",  # in-set harness null for the finetune round
    "s_linker76",  # s75 with two batch constants instead of three
    "s_linker77",  # s75 with the deterministic layer reduced to one SCANS row
    "s_linker78",  # s77 with the judging rubric stated as one principle
    "s_linker79",  # s78 with no deterministic gate at all
    "s_linker80",  # s79 with no computed evidence either
    "s_linker81",  # s80 with the two costed deletions restored
    "s_linker82",  # s81 with the prompt/dead-code audit fixes and one judging pass
    "s_linker83",  # s82 with the coreference judge shown the resolution it judges
    "s_linker85",  # s83's coreference judge composed with WordNet in place of INFLECTIONS
    "s_linker86",  # s85 with the full-name judge's focus line removed as a restatement
    "s_linker87",  # s86 with the coreference resolver's restated question removed
    "s_linker88",  # s87 with the judging prompt's repeated anchor sentences written once
    "s_linker89",  # s88 plus the resolver's per-case context range line removed
    "s_linker90",  # s89 with five authored clauses paraphrased from recipes to concepts
    "s_linker91",  # the two of those five whose consumer is a single stage
    "s_linker92",  # only the strict judge's enumeration deletion
    "s_linker92a",  # s92 with the LLM extraction pass replaced by the name scan
    "s_linker92a_noknow",  # RQ4 knowledge A/B for the reported arm: s92a, alias table off
    "s_linker92b",  # s92a, not proposing a name found only inside a dotted identifier
    "s_linker92c",  # s92b at the spelling-variant fidelity instead of case-folding
    "s_linker92d",  # s92b at both whole-name fidelities, unioned
    "s_linker92e",  # s92a with the lenient gate naming the surface before the verdict
    "s_linker92f",  # s92a with the lenient gate weighing the surface's readings first
    "s_linker94",  # s90 with the extractor and the resolver merged into one reading pass
    "s_linker95",  # s94 with the merged reading ordered into two sections inside the call
    "s_linker93",  # s90 with the resolver narrowed to sentences that write no name
    "s_linker96",  # the merged reading at the resolution question's batch size
    "s_linker97",  # the merged reading asked case by case
    "s_linker101",  # the head's two proposers plus the reading, all blind
    "s_linker103",  # candidates routed by evidence, not by proposing stage
    "s_linker106",  # resolver deliberates in-reply (self-implemented CoT)
    "s_linker107",  # antecedent shortlist computed in code, judgement in prompt
    "s_linker108",  # approved aliases reach the judges, not only the extractor
    "s_linker109",  # the partial-name scan refuses a word another name covers
    "s_linker110",  # s109 + the resolver's antecedent shortlist, computed
    "s_linker110_noknow",  # RQ4 knowledge A/B for the reported arm: s110, alias table off
    "s_linker110_noevidence",  # RQ4 evidence A/B for the reported arm: s110, computed context off
    "s_linker110_nocoderef",  # RQ4 floor: resolver gets the whole document, nothing computed
    "s_linker110_onecall",  # RQ4 total floor (D3): one linking call, no scan, no judge
    "s_linker111",  # s110 + the lenient gate enumerates the surface's readings
    "s_linker112",  # s110 + the sortal gate quotes before it commits
    "s_linker113",  # s112 + the sortal gate enumerates the readings too
    "s_linker114",  # the three judges as one pass over three skills
    "s_linker116",  # s114 + the lenient gate's reply carries the strict gate's ground
    "s_linker117",  # s114 + the lenient gate writes the verdict before the quote
    "s_linker118",  # s114 + the sortal gate's reply carries a ground too
    "s_linker119",  # s114 + one reply schema at all three judges

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
    "s_linker24_role_orchestrator": dict(
        aliases=(),
        module=(
            "llm_sad_sam.linkers.experimental."
            "s_linker24_role_orchestrator"
        ),
        class_name="SLinker24RoleOrchestrator",
        description=(
            "S-Linker24 Role Orchestrator — latest S24 replacement workflow with "
            "non-overlapping entity, coreference, catalog-handle, and exact "
            "catalog-identifier capabilities. Controller actions require exact "
            "document evidence."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker25": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker25",
        class_name="SLinker25",
        description=(
            "S-Linker25 — paper variant. Three linkers in fixed name-evidence "
            "order: full-name, then partial-name, then coreference. Standalone "
            "(no linker superclass); prompts byte-identical to s_linker21, "
            "controller and coverage-audit paths removed."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker26": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker26",
        class_name="SLinker26",
        description=(
            "S-Linker26 — s_linker25 with the knowledge module folded into the "
            "reading pass: one prompt per sentence batch returns the references "
            "and the names the passage establishes, and the table is fed forward "
            "to the next batch. No separate alias pass, no alias judge."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker27": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker27",
        class_name="SLinker27",
        description=(
            "S-Linker27 — the smallest variant: one call sends the whole document "
            "and returns both the references in it and the names it uses, so there "
            "is no alias stage, no alias judge and no extraction batching."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker28": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker28",
        class_name="SLinker28",
        description=(
            "S-Linker28 — s_linker26's merged reading, with the alias table no "
            "longer used to suppress partial-name candidates, so table size stops "
            "trading recall between two linkers."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker29": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker29",
        class_name="SLinker29",
        description=(
            "S-Linker29 — s_linker25 with the alias judge removed: the alias pass "
            "reports the sentence that establishes each alias and the sentence is "
            "checked lexically, so the workflow keeps both reading granularities "
            "with one prompt and one LLM call fewer."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker30": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker30",
        class_name="SLinker30",
        description=(
            "S-Linker30 — s_linker25 with the alias judge folded into the "
            "extraction pass: extraction reports which offered aliases it saw used "
            "as a name, and that confirmed subset is the table. One prompt and one "
            "call fewer, both reading granularities intact."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker31": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker31",
        class_name="SLinker31",
        description=(
            "S-Linker31 — s_linker25 with the alias pass proposing and reviewing "
            "its own list in one call: same rubric, same whole-document view, one "
            "prompt and one LLM call fewer."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker32": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker32",
        class_name="SLinker32",
        description=(
            "S-Linker32 — s_linker25 with the alias judge's question and rubric "
            "carried by the extraction pass: a second independent look at no extra "
            "call. One prompt and one LLM call fewer."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker33": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker33",
        class_name="SLinker33",
        description=(
            "S-Linker33 — s_linker32 with the carried alias judging decided by a "
            "majority of the reading batches instead of any single one, so the "
            "threshold no longer depends on document length."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker34": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker34",
        class_name="SLinker34",
        description=(
            "S-Linker34 — s_linker32 with the carried alias judging requiring every "
            "reading batch to approve, so a skimmed review rejects by default."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker35": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker35",
        class_name="SLinker35",
        description=(
            "S-Linker35 — s_linker34 with the carried alias judging asked as task 1, "
            "before the document is shown, so the review is not answered by a model "
            "that has already spent itself extracting."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker36": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker36",
        class_name="SLinker36",
        description=(
            "S-Linker36 — s_linker25 with the full-name judge asking its relevance "
            "and uniqueness criteria in one call instead of two, halving the calls "
            "of the workflow's largest stage."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker37": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker37",
        class_name="SLinker37",
        description=(
            "S-Linker37 — s_linker36 with each of the two judging criteria preceded "
            "by its own committed quote, so one call produces two answers that are "
            "each as expensive as a dedicated pass."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker38": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker38",
        class_name="SLinker38",
        description=(
            "S-Linker38 — one full-name judging prompt carrying both criteria, "
            "sampled twice with the verdicts ANDed: the relevance/uniqueness pass "
            "distinction leaves the architecture, independence comes from the two "
            "samples."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker39": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker39",
        class_name="SLinker39",
        description=(
            "S-Linker39 — two alias judges with distinct purposes: a dedicated "
            "context-free validity judge and an in-context usage review carried by "
            "the reading pass, combined as a union."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker40": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker40",
        class_name="SLinker40",
        description=(
            "S-Linker40 — one link-judging prompt sampled twice (from s_linker38) "
            "plus two alias judges with distinct purposes: a context-free validity "
            "judge and an in-context usage review carried by the reading, unioned."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker42": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker42",
        class_name="SLinker42",
        description=(
            "S-Linker42 — s_linker36's single full-name judging call plus a "
            "mention label of three values instead of five, both supported by the "
            "audit of s_linker38's six runs: the second sample splits on 0.6% of "
            "cases and three of the five label values are approved at the same "
            "rate."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker43": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker43",
        class_name="SLinker43",
        description=(
            "S-Linker43 — s_linker25 with one change: the mention label the judge "
            "is given has three values instead of five, dropping the case grading "
            "that changed no verdict over six audited runs and the residual value "
            "in favour of omitting the field."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker44": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker44",
        class_name="SLinker44",
        description=(
            "S-Linker44 — s_linker25 with one change: the mention label's two "
            "stated-name values, which differed only in the case of the match and "
            "were approved at the same rate, become one. The field is always "
            "present and the other three values are untouched."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker45": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker45",
        class_name="SLinker45",
        description=(
            "S-Linker45 — s_linker25 with one change: coreference resolution reads "
            "the batch size the judges use (25 sentences, not 10), so the workflow "
            "states two batch constants instead of three and pays about 74 calls "
            "per five-project run instead of 101."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker46": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker46",
        class_name="SLinker46",
        description=(
            "S-Linker46 — s_linker25 with one change: the alias table admits "
            "full-name candidates but no longer suppresses partial-name ones, so it "
            "has one role instead of two opposite ones. Frees 16 candidates over the "
            "five projects, 4 of them gold."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker47": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker47",
        class_name="SLinker47",
        description=(
            "S-Linker47 — s_linker25 with one mechanism removed: the partial-name "
            "linker's grounded identity review. Recorded over six s25 runs, that "
            "review rejected 8.0 candidates per run of which 5.5 were gold, so it "
            "traded 5.5 true positives for 2.5 false positives."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker48": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker48",
        class_name="SLinker48",
        description=(
            "S-Linker48 — s_linker25 with no mechanism removed and five "
            "near-duplicate conditions merged into two named predicates, plus the "
            "three conjuncts of the identity gate that never fired in 122 recorded "
            "cases. Every prompt byte, call and stage unchanged."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker49": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker49",
        class_name="SLinker49",
        description=(
            "S-Linker49 — s_linker47 and s_linker48 composed: the grounded identity "
            "review removed, so every linker judges in one call, and eight copies of "
            "two conditions merged into two named predicates. Both held separately "
            "over six paired runs."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker50": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker50",
        class_name="SLinker50",
        description=(
            "S-Linker50 — s_linker49 with one constant reworded: the coreference "
            "resolution rule states one guideline instead of two lettered clauses, a "
            "five-phrase list and an alias-shape enumeration. The dropped clause "
            "licensed 0 of 578 recorded resolutions. -27% instruction bytes per run; "
            "no stage, call or method body changed."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker51": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker51",
        class_name="SLinker51",
        description=(
            "S-Linker51 — s_linker49 with nine of ten rule constants rewritten from "
            "enumerations into the principles they enumerate. Rule text -39%, "
            "instruction bytes per five-project run -44%. Tests whether the prompts "
            "are general guidelines or a rulebook grown against this benchmark."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker52": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker52",
        class_name="SLinker52",
        description=(
            "S-Linker52 — s_linker51 with the three knowledge-side rules back at "
            "s_linker49's wording; the other six generalizations stand. Bisects "
            "s51's loss onto the side of the pipeline that builds the alias table."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker53": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker53",
        class_name="SLinker53",
        description=(
            "S-Linker53 — s_linker51 with one subordinate clause restored inside the "
            "alias judge: a phrase naming a grouping of several elements is not an "
            "alias for one of them. The surrounding principle already entails it, so "
            "this asks whether an enumerated case does work its entailing principle "
            "does not."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker49_null": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker49_null",
        class_name="SLinker49Null",
        description=(
            "S-Linker49Null — byte-identical to s_linker49 apart from the checkpoint "
            "namespace. A null arm: paired against s_linker49 it measures what this "
            "harness reports as a difference when there is none, which is the "
            "calibration every p value in this branch has been quoted without."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker54": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker54",
        class_name="SLinker54",
        description=(
            "S-Linker54 — s_linker51 with the three full-name rules reverted to "
            "s_linker49's, the knowledge and coreference generalizations standing. "
            "With s_linker52 it partitions s51's rewrite."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker55": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker55",
        class_name="SLinker55",
        description=(
            "S-Linker55 — only the three coreference rules generalized; knowledge and "
            "full-name exactly s_linker49's. Rule text -19%, instruction bytes per "
            "run -31%, because COREF_RULES alone is half the budget."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker56": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker56",
        class_name="SLinker56",
        description=(
            "S-Linker56 — s_linker55 with the coreference prompt's opening instruction "
            "paragraph deleted: it stated the same three things as the COREF_RULES "
            "block appended below it. 253 B x 40 calls per run, the workflow's only "
            "intra-prompt duplication."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker57": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker57",
        class_name="SLinker57",
        description=(
            "S-Linker57 — s_linker55 plus the two alias *proposer* rules generalized, "
            "the alias judge carried verbatim. Tests whether a proposer's enumeration "
            "is free when a dedicated judge follows, on a family whose judge is "
            "lenient by design."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker58": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker58",
        class_name="SLinker58",
        description=(
            "S-Linker58 — s_linker55 plus the full-name *proposer* rule generalized, "
            "both judging rubrics carried verbatim. The mirror of s_linker57 on the "
            "family with the largest measured effect."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker59": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker59",
        class_name="SLinker59",
        description=(
            "S-Linker59 — the coreference family plus P1_FOCUS plus the alias judge "
            "rubric generalized: every clause that single-stage ablation cleared, and "
            "no others. Rule text -26%, instruction bytes per five-project run -34%. "
            "LAYERED_ENTITY_RULES, ENTITY_EXTRACTION_RULES and the coreference "
            "prompt's preamble are kept at s49's wording because the same method "
            "refuted them."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker60": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker60",
        class_name="SLinker60",
        description=(
            "S-Linker60 — s_linker59 with the alias proposer folded into the "
            "entity-extraction reading and the alias judge kept as its own call: the "
            "one arrangement the s26-s34 merge line never built. Three "
            "document-reading prompts become two, 83 calls per run against 88. "
            "Stage-screened at TP -0.6 (p = 0.80) / FP -16.6 (p = 0.01) on the alias "
            "table."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker61": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker61",
        class_name="SLinker61",
        description=(
            "S-Linker61 — s_linker60 with ALIAS_EXCLUSION_RULES also stated in the "
            "alias judge's prompt, where a dedicated call enforces it. The merged "
            "reading leaks qualified-name fragments the two-stage proposer never "
            "did; measured reach on this benchmark is zero, so this is design "
            "integrity rather than a performance claim."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker59_null": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker59_null",
        class_name="SLinker59Null",
        description=(
            "S-Linker59Null — byte-identical to s_linker59 except its variant name. "
            "Carried in the same invocation as every arm measured against s59, so the "
            "harness's own paired-run offset is measured in-set rather than assumed "
            "to be zero."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker62": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker62",
        class_name="SLinker62",
        description=(
            "S-Linker62 — s_linker59 with the partial-name proposer's prefix test "
            "bounded to English inflections. The bare prefix made `WebRTC` an owner "
            "of both `WebRTC-SFU` and `BBB web`, so the pair was dropped as "
            "ambiguous and two gold links were lost every run. Deterministic screen "
            "+2.0 gold / +0.0 spurious candidates; with the real denotation judge "
            "TP +2.0 (p = 0.01), FP +1.0 (p = 0.01)."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker63": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker63",
        class_name="SLinker63",
        description=(
            "S-Linker63 — s_linker62 with the boundary defect in "
            "`_inside_qualified_identifier` repaired: `before in \"-_\"` was "
            "evaluated with `before == \"\"` for a sentence-initial span, and "
            "`\"\" in \"-_\"` is True, hiding 344 spans per run. At the stage the "
            "repair is TP +/-0.0 (p = 1.00), FP +1.2 (p = 0.01), so this arm prices "
            "a defect rather than proposing an improvement."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker65_null": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker65_null",
        class_name="SLinker65Null",
        description=(
            "S-Linker65-null — byte-identical to s_linker65 apart from the "
            "checkpoint namespace. The harness null arm: it measures what this "
            "pipeline produces from nothing in one invocation set, so every other "
            "arm's delta can be read against it rather than against zero."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker66": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker66",
        class_name="SLinker66",
        description=(
            "S-Linker66 — s_linker65 with ONE relocation: the full-name linker's "
            "admission contract leaves the code and is stated in the extraction "
            "prompt. `_keep_stated_names` is deleted; `ENTITY_EXTRACTION_RULES` "
            "asks for a reference only when the sentence itself writes the "
            "component's name or a known alias. Stage evidence "
            "(pilot/bind_pilots.py --pilot bindcontract, five samples a side): "
            "deleting the filter with no compensation is TP +4.8 / FP +10.6 (both "
            "p = 0.01); stating it in the prompt is TP -1.4 (p = 0.21), FP -1.8 "
            "(p = 0.47)."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker67": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker67",
        class_name="SLinker67",
        description=(
            "S-Linker67 — s_linker66 plus the two tight scans relocated: "
            "`SCANS[stated_name]` and `SCANS[spelling]` leave the code and their "
            "recall floor is stated in the same extraction prompt, so `SCANS` "
            "keeps one row and `_add_scan` is deleted. Stage evidence "
            "(pilot/bind_pilots.py --pilot bindboth): TP -1.2 (p = 0.14), FP -1.2 "
            "(p = 0.37). Note the same scans deleted under s65's *unchanged* "
            "prompt cost TP 3.6 (p = 0.01) and the clause alone recovered none of "
            "it (--pilot bindscans), so the two relocations are not separable."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker66_null": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker66_null",
        class_name="SLinker66Null",
        description=(
            "S-Linker66-null — byte-identical to s_linker66 apart from the checkpoint "
            "namespace. The in-set harness null for the fold round."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker69": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker69",
        class_name="SLinker69",
        description=(
            "S-Linker69 — s_linker66 with the gates that turned out not to decide "
            "either folded into a prompt or deleted: (1) the partial-name scan's "
            "span-boundary gate becomes one general sentence in the denotation prompt "
            "(deleting it outright is FP +7.0 p=0.01; folded it is TP -0.4 p=0.44, "
            "FP -0.2 p=1.00), which also retires `SurfaceScan.skip_qualified` and "
            "`_inside_qualified_identifier`; (2) the coreference antecedent gate is "
            "deleted (TP +0.0, FP +0.0, both p=1.00 when scored on what coreference "
            "actually contributes); (3) the denotation claim substring check is "
            "deleted (0 of 380 verdicts voided over six runs); (4) `_unlinked` is "
            "deleted — it compared a tuple against a list of SadSamLink and has never "
            "removed anything (pilot/unlinked_audit.py), and repairing it instead is "
            "TP -0.6 (p=0.17), FP +0.4 (p=0.71). `unique_owner` stays: folding it is "
            "the round's negative result."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker70": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker70",
        class_name="SLinker70",
        description=(
            "S-Linker70 — s_linker69 with the spelling row's two options gone. "
            "(1) `skip_stricter` becomes `STRICTER_CLAUSE` in the full-name judging "
            "prompt: the gate was keeping the ANY_CASE whole-name cell out of the "
            "candidate set and cost 4.0 true positives per run. Deleting it is TP "
            "+4.0 (p=0.01) at FP +1.8 (p=0.01); deleting it and stating the "
            "distinction for the judge is TP +4.0 (p=0.01) at FP +/-0.0 (p=1.00) "
            "(pilot/fold_pilots.py --pilot foldstricter, 5 samples per arm). "
            "(2) `unique_owner` is dropped from that row as an identity — it frees 0 "
            "pairs on all five projects in either configuration "
            "(pilot/gate_inventory.py); it stays on the partial-name row, where it is "
            "priced at 2.4 FP. The two gates that remain are both on the partial-name "
            "row and are blocked by one design fact rather than two rules: the "
            "denotation judge is target-blind, so neither question can be put to it."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker71": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker71",
        class_name="SLinker71",
        description=(
            "S-Linker71 - s_linker70 with the authored prompts scored on provenance "
            "and rewritten to general principles wherever they named a shape "
            "(pilot/prompt_defensibility.py: 1700 of 3645 bytes stood on an admissible "
            "ground). Adopted: (1) the full-name rubric's four numbered "
            "reject-conditions and three named approve-shapes become one principle "
            "plus QUALIFIED_CLAUSE and STRICTER_CLAUSE, and P1_FOCUS drops its "
            "code-level-identifier tail - 850 authored bytes become 579 at TP +0.7 "
            "(p=0.80), FP -1.3 (p=0.20); (2) the extraction prompt's code-path clause "
            "becomes QUALIFIED_CLAUSE - TP +0.7 (p=1.00), FP -6.0 (p=0.20). Refused "
            "and documented: the coreference rubric keeps its clause (removing it is "
            "TP +4.7/FP +3.7, replacing it TP -3.0/FP +4.0 - a clause about "
            "identifiers misleads a judge whose cases contain no name), and the alias "
            "prompt keeps its syntax because both wordings admit 0 identifier "
            "fragments while the general ones grow the table from 24.0 to 36.7-37.3 "
            "terms per run: the clause's effect is not the effect it states."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker72": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker72",
        class_name="SLinker72",
        description=(
            "S-Linker72 - s_linker71 split. KEPT: the full-name judging rubric's four "
            "numbered reject-conditions and three named approve-shapes become one "
            "principle plus QUALIFIED_CLAUSE and STRICTER_CLAUSE (850 authored bytes "
            "-> 579). REVERTED to s_linker70: ENTITY_EXTRACTION_RULES and the "
            "extraction prompt. s_linker71 generalized both at once and lost ground "
            "over six runs (macro F1 94.80 / F2 96.19 against s70's 95.74 / 96.99), "
            "although each half had read at parity on its own stage. The extraction "
            "prompt feeds every later stage, so a change there is not stage-local; and "
            "the alias arm in the same round showed the mechanism directly - a "
            "prohibitive clause naming a concrete shape makes the model conservative "
            "about everything, not only about what it prohibits (alias table 24.0 -> "
            "36.7 terms per run with 0 prohibited fragments admitted either way)."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker73": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker73",
        class_name="SLinker73",
        description=(
            "S-Linker73 - s_linker70 with only the two spans GATE-07 forbids reworded, "
            "after s_linker71 (F1 94.80, n=6) and s_linker72 (94.94, n=3) both lost "
            "~0.8 F1 against s70's 95.74 by restructuring the judging rubric. The "
            "enumeration carries precision a single principle does not, so it stays: "
            "reject-condition (1) loses the literal `x.y or x.y.z` and says 'only as "
            "part of a longer joined or dotted identifier' instead, and the approve "
            "side loses 'a heading, or a list' for 'a mention that says nothing "
            "further about the component'. Corpus-shaped bytes in the judging path go "
            "to zero with the rubric's structure, leniency and other three conditions "
            "byte-equivalent to s70's."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker76": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker76",
        class_name="SLinker76",
        description=(
            "S-Linker76 - s_linker75 with one changed number: COREFERENCE_BATCH is set "
            "to JUDGE_BATCH, so the module states two batch sizes instead of three. The "
            "third was the only bound with no counterpart and the module's largest cost "
            "- 40.0 of 91.7 calls per five-project run. s_linker45 measured the same "
            "unification on the s25 base over six paired runs at parity (F1 -0.2 "
            "p = 0.52, F2 -0.0 p = 0.91, 65.3 calls against 88.8); this variant carries "
            "that result into the s70-s75 line. Chosen by unification, not by search."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker77": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker77",
        class_name="SLinker77",
        description=(
            "S-Linker77 - s_linker75 with the deterministic layer reduced to one row: "
            "the two tight SCANS rows are relocated into the extraction prompt and "
            "`_add_scan` is deleted, so `SCANS` keeps only the partial-name row. "
            "s_linker67 measured the same relocation on the s66 base at TP -4.0 "
            "(p = 0.03) / macro F2 -1.1 (p = 0.04) composed and refused it under an "
            "F1-led reading; taken here under an F2-led budget of 2 pp. Everything else "
            "is s75's byte for byte."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker78": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker78",
        class_name="SLinker78",
        description=(
            "S-Linker78 - s_linker77 with the last enumeration gone: the full-name "
            "gate's four numbered reject-conditions and three named approve-shapes "
            "become one principle, with QUALIFIED_CLAUSE added to the judging prompt "
            "for condition (1) and STRICTER_CLAUSE (already there) carrying (3) and "
            "(4). s_linker71 measured this restructuring at ~0.8 F1 composed (94.80 "
            "n=6 against s70's 95.74) and F2 96.19 against 96.99. Priced here on F2."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker79": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker79",
        class_name="SLinker79",
        description=(
            "S-Linker79 - s_linker78 with the last two deterministic options gone: the "
            "one SCANS row drops `unique_owner` (priced at 2.4 FP; folding it into the "
            "target-blind denotation prompt is the fold round's -8.4 TP) and "
            "`skip_when_named` (s_linker46 on the s25 base: F1 -1.5, F2 -1.0). What is "
            "left proposes spans of one relation and decides nothing - no gate anywhere "
            "in the deterministic layer. Priced under a 3 pp F2 budget."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker80": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker80",
        class_name="SLinker80",
        description=(
            "S-Linker80 - s_linker79 with the mention label removed from the evidence "
            "line, so the code computes nothing about a case at all. Expected to lose: "
            "removing the field is -10.7 TP and asking the judge for it instead is "
            "-6.7 TP (concept round). Run to price the design law's hardest case on F2 "
            "rather than on TP."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker81": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker81",
        class_name="SLinker81",
        description=(
            "S-Linker81 - s_linker80 with the two deletions the post-mortem costed put "
            "back and the third left out. `skip_when_named` returns (it adds 140 of the "
            "161 ungated candidates, and it states which linker owns a case rather than "
            "gating one); the mention label returns at VIA_ALIAS and CODE_TOKEN only "
            "(100% and 73% of those approvals were lost against 3% of proper-case ones). "
            "`unique_owner` and the other three labels stay out, as facts the judge is "
            "already holding. Priced against s78 on F2."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker82": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker82",
        class_name="SLinker82",
        description=(
            "S-Linker82 - s_linker81 with the prompt audit's fixes. One full-name "
            "judging pass instead of two AND-ed ones (the passes disagree on 4.0 of "
            "~196 candidates per five-project run in s81's own recorded runs, and both "
            "questions are already in the shared rubric), a deduplicated coreference "
            "batch (one sentence table instead of a pasted window per case), an alias "
            "judge whose no-answer path is one documented default instead of three "
            "behaviours (a parse failure approved everything, a keyless reply nothing), "
            "an extraction "
            "prompt that no longer states two contradictory admission rules, judged "
            "claims recorded in the trace, and the dead deterministic layer removed. "
            "Priced against s81 over three paired five-project runs: macro F2 94.08 "
            "against 92.32 (positive in 3 of 3), macro F1 92.25 against 91.91, 81 LLM "
            "calls per run against 88. Same direction on gpt-5.6-luna: macro F2 91.16 against "
            "87.36, macro F1 85.38 against 81.80, both 3 of 3 runs."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker83": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker83",
        class_name="SLinker83",
        description=(
            "S-Linker83 - s_linker82 with three changes at the coreference judge: it is "
            "shown the resolution it is judging, its rubric distinguishes a component "
            "from what the component acts on or produces, and it states the strongest "
            "ground for rejecting a case before deciding. Exact pipeline scoring, three "
            "runs a side with upstream held fixed: terra F1 93.69 / F2 94.51 against "
            "92.25 / 94.08, luna 89.20 / 92.43 against 85.38 / 91.16 - the laxer model "
            "gains most and the stricter does not regress. Seven other judge settings "
            "were refused; see results/judge_calibration_round/README.md."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker85": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker85",
        class_name="SLinker85",
        description=(
            "S-Linker85 - s_linker83's coreference judge composed with a WordNet "
            "lemmatizer in place of the INFLECTIONS ending list, so the module "
            "carries no authored word list at all. The swap was priced span by span "
            "before it was written: over all 3697 (name, sentence) pairs of the five "
            "projects the two scans differ on 2, partial-name candidates go 109 -> "
            "110 at gold 28 -> 28, nothing is lost (pilot/lemma_swap_pilot.py). "
            "Needs nltk and the wordnet corpus. "
            "The judge is shown the resolution it is judging, its rubric carries the "
            "actor/artifact distinction, and it states the ground for rejecting a case "
            "before deciding. Exact pipeline scoring over three runs a side with "
            "upstream held fixed: terra F1 93.69 / F2 94.51 against s82's 92.25 / 94.08, "
            "luna 89.20 / 92.43 against 85.38 / 91.16. Confirmed end to end, three "
            "paired runs per model: terra macro F1 93.68 against 91.13 and luna 89.48 "
            "against 83.83, with luna's false positives halved (41.3 against 80.0) - "
            "the laxer model gains most and the stricter one does not regress."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker86": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker86",
        class_name="SLinker86",
        description=(
            "S-Linker86 - s_linker85 with one deletion: the full-name judge's focus "
            "line. VALIDATION_FOCUS asked for architectural participation and "
            "referential specificity, which the rubric under it already asks - "
            "LAYERED_ENTITY_RULES makes participation the approve-condition and "
            "STRICTER_CLAUSE is about nothing else than a name identifying this "
            "element rather than ordinary vocabulary. 243 B leave each of the ~18.5 "
            "judging calls per five-project run. Stage arm on both models, three runs "
            "a side over the same extraction pass: terra TP 182.0 -> 183.0, macro F2 "
            "-0.0 (p = 0.80); luna TP 174.7 -> 175.7, macro F1 +0.1 (p = 0.90), macro "
            "F2 +0.3 (p = 0.60). The same round measured typed verdict rubrics at all "
            "three judges and refused every one on the second model; see "
            "results/typed_round/README.md."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker87": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker87",
        class_name="SLinker87",
        description=(
            "S-Linker87 - the typed round's head: s_linker86 plus one more deleted "
            "restatement. COREF_RULES opened by asking the resolver the question its "
            "own prompt preamble already asks; s56 priced deleting that preamble at "
            "TP -16.2 because it is also the input-format contract, and this is the "
            "untried other half - the contract stays, the restatement goes. 163 B off "
            "each of the 40 resolver calls a five-project run makes, the module's "
            "largest instruction item. Stage arm over the resolver AND the strict "
            "judge behind it, three runs a side: terra composed TP +1.7, macro F1 "
            "-0.2 (p = 0.80), F2 +0.2; luna TP +/-0.0 (p = 1.00), F1 +0.2, F2 +0.3. "
            "With s86's cut, authored rule text 3485 -> 3079 B (-11.7%). See "
            "results/typed_round/README.md."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker94": dict(
        aliases=("read", "reading"),
        module="llm_sad_sam.linkers.experimental.s_linker94",
        class_name="SLinker94",
        description=(
            "S-Linker94 - one reading pass proposes for all three linkers. The "
            "named-reference extractor and the coreference resolver are merged "
            "into a single question over the same 50-sentence block, carrying a "
            "per-component note of the last sentence that named it; every judge, "
            "the alias module and the deterministic scan are inherited unchanged."
        ),
    ),
    "s_linker95": dict(
        aliases=("readordered",),
        module="llm_sad_sam.linkers.experimental.s_linker95",
        class_name="SLinker95",
        description=(
            "S-Linker95 - the merged reading with the cascade inside the call: "
            "a named section first, then refer-backs resolved against the list "
            "that same call just produced."
        ),
    ),
    "s_linker96": dict(
        aliases=("readgrain",),
        module="llm_sad_sam.linkers.experimental.s_linker96",
        class_name="SLinker96",
        description=(
            "S-Linker96 - the merged reading asked at the resolution question's "
            "batch size. Best macro F2 of the merged arms on terra (+0.5)."
        ),
    ),
    "s_linker97": dict(
        aliases=("readcases",),
        module="llm_sad_sam.linkers.experimental.s_linker97",
        class_name="SLinker97",
        description=(
            "S-Linker97 - the merged reading asked case by case, so every sentence "
            "is accounted for. Macro F1 +2.3 terra / +10.3 luna at the proposal "
            "stage, F2 -0.1 / +3.4."
        ),
    ),
    "s_linker92a": dict(
        aliases=("scan", "regexextract"),
        module="llm_sad_sam.linkers.experimental.s_linker92a",
        class_name="SLinker92a",
        description=(
            "S-Linker92a - the LLM extraction pass deleted and its own contract run "
            "as a scan: every pair whose sentence writes a name of the component at "
            "ANY_CASE, over the catalog and the discovered aliases. 9 of ~16.8 calls "
            "a run go with it. Proposer ceiling +7.8 net gold a run over the "
            "extractor, off 30 recorded runs (pilot/regex_extract_audit.py)."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker92a_noknow": dict(
        aliases=("scannoknow",),
        module="llm_sad_sam.linkers.experimental.s_linker92a",
        class_name="SLinker92a",
        description=(
            "S-Linker92a NO-KNOWLEDGE - RQ4 knowledge A/B for the arm the paper reports "
            "(experimental=True, NOT canonical). s_linker92a with no_knowledge=True: the "
            "document-alias stage is skipped and an empty DocumentKnowledge is set directly, "
            "so both the scan and the partial-name linker see canonical component names only. "
            "All other phases run unchanged. Mirrors s_linker21_noknow one lineage over. "
            "LANDMINE: _VARIANT_NAME stays 's_linker92a', so its phase states nest under "
            "phase_states/s_linker92a/ -- give every run its own PHASE_CACHE_DIR or it "
            "clobbers the Full arm's states."
        ),
        canonical=False,
        experimental=True,
        kwargs=dict(no_knowledge=True),
    ),
    "s_linker92b": dict(
        aliases=("scannoqual",),
        module="llm_sad_sam.linkers.experimental.s_linker92b",
        class_name="SLinker92b",
        description=(
            "S-Linker92b - s92a without the 20.8 pairs a run whose name is written "
            "only inside a longer dotted identifier. Same gold, 25.0 fewer pairs; "
            "worth building only if the judge does not apply QUALIFIED_CLAUSE itself."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker92c": dict(
        aliases=("scanspelling",),
        module="llm_sad_sam.linkers.experimental.s_linker92c",
        class_name="SLinker92c",
        description=(
            "S-Linker92c - s92b at the deleted prompt's other fidelity: spacing, "
            "hyphenation and compound joining count as the same name. +0.8 gold and "
            "+1.3 pairs a run for a second relation point in the code."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker92d": dict(
        aliases=("scanboth",),
        module="llm_sad_sam.linkers.experimental.s_linker92d",
        class_name="SLinker92d",
        description=(
            "S-Linker92d - the whole-name row of the relation: both fidelities "
            "unioned, as the name-relation table prescribes. Best proposer of the "
            "family at +1.2 gold over s92b, and the most code."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker92e": dict(
        aliases=("scansurface",),
        module="llm_sad_sam.linkers.experimental.s_linker92e",
        class_name="SLinker92e",
        description=(
            "S-Linker92e - s92a whose lenient gate quotes the case's own surface "
            "before the claim and the verdict. No rule added. REFUTED: stage gold "
            "152.0 -> 147.7 terra, FP 59.0 -> 70.7 luna."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker92f": dict(
        aliases=("scanreadings",),
        module="llm_sad_sam.linkers.experimental.s_linker92f",
        class_name="SLinker92f",
        description=(
            "S-Linker92f - s92a whose lenient gate lists the readings the surface "
            "could have here and names the one it has, then decides. No rule added. "
            "Best macro F1 of the round on terra (93.07, FP 26.3, below control); "
            "on luna it cuts the scan's added FP at 6.0 TP."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker101": dict(
        aliases=("thirdlook",),
        module="llm_sad_sam.linkers.experimental.s_linker101",
        class_name="SLinker101",
        description=(
            "S-Linker101 - the head's two proposers plus the merged reading as a "
            "third mutually blind proposer. Proposal recall +1.2 to +1.5 terra, "
            "+4.0 luna; recall-led, for an F2 budget."
        ),
    ),
    "s_linker103": dict(
        aliases=("evroute",),
        module="llm_sad_sam.linkers.experimental.s_linker103",
        class_name="SLinker103",
        description=(
            "S-Linker103 - candidates are routed to a judge by the evidence their "
            "sentence gives, not by the stage that proposed them. Recovers gold the "
            "coreference judge refuses for being named directly. No new calls."
        ),
    ),
    "s_linker106": dict(
        aliases=("deliberate",),
        module="llm_sad_sam.linkers.experimental.s_linker106",
        class_name="SLinker106",
        description=(
            "S-Linker106 - s101 with the resolver deliberating in-reply: quote the "
            "referring expression, list the candidates weighed, then commit. "
            "Chain-of-thought we implement, not vendor reasoning. No extra call."
        ),
    ),
    "s_linker107": dict(
        aliases=("shortlist",),
        module="llm_sad_sam.linkers.experimental.s_linker107",
        class_name="SLinker107",
        description=(
            "S-Linker107 - s101 with the antecedent shortlist computed in code and "
            "handed to the resolver per case (NAMED BEFORE THIS CASE). Deliberation "
            "tailored to the question: bookkeeping in code, judgement in the prompt."
        ),
    ),
    "s_linker109": dict(
        aliases=("nesting",),
        module="llm_sad_sam.linkers.experimental.s_linker109",
        class_name="SLinker109",
        description=(
            "S-Linker109 - s92a plus one refusal in the partial-name scan: a word "
            "written only inside another component's whole name is that component's "
            "pair, not this one. Level 1 decided, both models, 0 gold at -5.0/-10.3 "
            "false positives a run; no call added or removed."
        ),
    ),
    "s_linker110": dict(
        aliases=("shortlist2",),
        module="llm_sad_sam.linkers.experimental.s_linker110",
        class_name="SLinker110",
        description=(
            "S-Linker110 - s109 with the resolver's candidate antecedents enumerated "
            "in code per case (NAMED BEFORE THIS CASE). s107's arm rebased off the "
            "retired s101 onto the adopted head. Level 2 owed on luna."
        ),
    ),
    "s_linker110_noknow": dict(
        aliases=("shortlist2noknow",),
        module="llm_sad_sam.linkers.experimental.s_linker110",
        class_name="SLinker110",
        description=(
            "S-Linker110 NO-KNOWLEDGE - RQ4 knowledge A/B for the arm the paper reports "
            "(experimental=True, NOT canonical). s_linker110 with no_knowledge=True: the "
            "document-alias stage is skipped and an empty DocumentKnowledge is set directly, "
            "so both the scan and the partial-name linker see canonical component names only. "
            "All other phases run unchanged. Mirrors s_linker92a_noknow one lineage over, and "
            "reaches no_knowledge through s109 -> s92a -> s92, where the kwarg is defined. "
            "LANDMINE: _VARIANT_NAME stays 's_linker110', so its phase states nest under "
            "phase_states/s_linker110/ -- give every run its own PHASE_CACHE_DIR or it "
            "clobbers the Full arm's states."
        ),
        canonical=False,
        experimental=True,
        kwargs=dict(no_knowledge=True),
    ),
    "s_linker110_noevidence": dict(
        aliases=("noevidence",),
        module="llm_sad_sam.linkers.experimental.s_linker110_noevidence",
        class_name="SLinker110NoEvidence",
        description=(
            "S-Linker110 NO-EVIDENCE - RQ4 evidence A/B for the arm the paper reports "
            "(experimental=True, NOT canonical). s_linker110 with every code-computed "
            "context withheld from the judges: the full-name Evidence block, the [prev] "
            "prefix in every case, the partial-name step's +/-5 window (narrowed to the "
            "candidate's own sentence), and the resolver's NAMED BEFORE THIS CASE "
            "shortlist. The coreference SENTENCES window is KEPT -- without it a "
            "refer-back is unresolvable in principle, so removing it would price the "
            "task's impossibility rather than the shortlist's worth. Linkers, rubrics, "
            "batch sizes and reply shapes are unchanged, so the candidate set entering "
            "each judge is the head's and the parser is the head's. _VARIANT_NAME is its "
            "own, so it pairs with s_linker110 in ONE invocation without clobbering its "
            "phase states."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker110_nocoderef": dict(
        aliases=("nocoderef",),
        module="llm_sad_sam.linkers.experimental.s_linker110_nocoderef",
        class_name="SLinker110NoCodeRef",
        description=(
            "S-Linker110 NO-CODE COREFERENCE - the floor arm for RQ4's coreference half "
            "(experimental=True, NOT canonical). Extends s_linker110_noevidence: the "
            "resolver is handed the WHOLE DOCUMENT and the component list and nothing "
            "else -- no targets, no COREFERENCE_BATCH batching, no +/-5 window table, no "
            "NAMED BEFORE THIS CASE shortlist. The reply schema, the parser, the post-hoc "
            "validity checks and COREF_RULES are the head's, so the strict coreference "
            "judge downstream sees the same fields. CONFOUND, stated: s_linker27 measured "
            "a whole-document call and found accuracy tracks document length (jabref 13 "
            "sentences 100.0, teammates 198 84.1), and batching is itself code-computed, "
            "so a loss here is an UPPER BOUND on what the computed context is worth -- "
            "read per-project before macro."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker110_onecall": dict(
        aliases=("onecall",),
        module="llm_sad_sam.linkers.experimental.s_linker110_onecall",
        class_name="SLinker110OneCall",
        description=(
            "S-Linker110 ONE CALL - RQ4's total floor, D3 (experimental=True, NOT "
            "canonical). The document, the component list and the discovered alias table "
            "go into ONE linking call and the reply is the final link set: no scan, no "
            "window, no evidence bundle, no antecedent shortlist, no judge, no union. "
            "The head's four rubrics are rendered verbatim so the arm removes the "
            "arrangement and not the guidance. The knowledge stage is KEPT (it is RQ3's "
            "ablation), so 'one call' means one LINKING call. NOT s_linker27, which "
            "merged only the reading and kept every scan and judge. TWO stated confounds: "
            "no quote is demanded (worth 35.2 TP on its own), and whole-document calls "
            "carry s27's length effect (jabref 13 sentences vs teammates 198), so a loss "
            "is an UPPER BOUND on what the workflow is worth. Has no linker_* phases, so "
            "the mini-rq34 engines cannot read it; score end to end with score_runs.py."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker111": dict(
        aliases=("readings",),
        module="llm_sad_sam.linkers.experimental.s_linker111",
        class_name="SLinker111",
        description=(
            "S-Linker111 - s110 with s92f's lenient gate: the reply enumerates the "
            "readings the quoted surface could have here and names the one it has "
            "before it decides. The regex round's best terra macro F1, rebased onto "
            "the adopted head; the strict branch renders s92's bytes exactly."
        ),
    ),
    "s_linker112": dict(
        aliases=("quotefirst",),
        module="llm_sad_sam.linkers.experimental.s_linker112",
        class_name="SLinker112",
        description=(
            "S-Linker112 - s110 with the partial-name denotation gate demanding its "
            "quote BEFORE its verdict, the order the other two judges already use. "
            "Two strings, no rule text; s48's separation applied to the one judge "
            "whose reply committed in the wrong field order."
        ),
    ),
    "s_linker113": dict(
        aliases=("sortalreadings",),
        module="llm_sad_sam.linkers.experimental.s_linker113",
        class_name="SLinker113",
        description=(
            "S-Linker113 - s112 plus s92f's enumeration at the sortal gate: the reply "
            "lists the readings the expression could have here and names the one it "
            "has, between the quote and the verdict. The enumerate-then-commit law at "
            "its second judging site."
        ),
    ),
    "s_linker116": dict(
        aliases=("lenientground",),
        module="llm_sad_sam.linkers.experimental.s_linker116",
        class_name="SLinker116",
        description=(
            "S-Linker116 - s114 with the lenient gate's reply carrying the strict gate's `objection` field, using that gate's own decide clause verbatim. First half of the uniform-schema question: the head refuses this field here by argument (approve-by-default and name-the-ground contradict), never by measurement."
        ),
    ),
    "s_linker117": dict(
        aliases=("verdictfirst",),
        module="llm_sad_sam.linkers.experimental.s_linker117",
        class_name="SLinker117",
        description=(
            "S-Linker117 - s114 with the lenient gate writing the verdict before the quote, which is the sortal gate's field order. s112 asked the same question at the sortal gate, where one project carries the whole population; this asks it where 150 gold links a run do."
        ),
    ),
    "s_linker118": dict(
        aliases=("sortalground",),
        module="llm_sad_sam.linkers.experimental.s_linker118",
        class_name="SLinker118",
        description=(
            "S-Linker118 - s114 with the sortal gate's reply carrying a ground for the other reading. The second half of the `objection` question, at the gate with the least room to move."
        ),
    ),
    "s_linker119": dict(
        aliases=("uniformschema",),
        module="llm_sad_sam.linkers.experimental.s_linker119",
        class_name="SLinker119",
        description=(
            "S-Linker119 - s114 with all three judges replying in ONE structure: the sortal gate adopts the other two's reply key, field order and boolean verdict. Composed with s116 at the lenient gate this is the whole uniform design; the polarity, the withheld target and the withheld catalog do not move."
        ),
    ),
    "s_linker114": dict(
        aliases=("skills",),
        module="llm_sad_sam.linkers.experimental.s_linker114",
        class_name="SLinker114",
        description=(
            "S-Linker114 - the head's three judges expressed as one judging pass over "
            "three JudgeSkill declarations. Byte-identical to s110 by construction: "
            "142/142 batches over six recorded runs send the same prompt, record the "
            "same decision and keep the same set. A refactor, not an arm."
        ),
    ),
    "s_linker108": dict(
        aliases=("aliasjudge",),
        module="llm_sad_sam.linkers.experimental.s_linker108",
        class_name="SLinker108",
        description=(
            "S-Linker108 - s101 with the approved alias table passed to the judging "
            "prompts, not just the extractor. Targets residual FNs whose evidence "
            "sentence writes an alias the judge was never told about. No new calls."
        ),
    ),
    "s_linker93": dict(
        aliases=("narrowcoref",),
        module="llm_sad_sam.linkers.experimental.s_linker93",
        class_name="SLinker93",
        description=(
            "S-Linker93 - both proposal calls kept; the resolver is asked only "
            "about sentences that write no name. 44%% fewer resolver calls, "
            "0.7 gold a project-run at risk."
        ),
    ),
    "core": dict(
        aliases=("alinker_core",),
        module="llm_sad_sam.linkers.experimental.alinker_core",
        class_name="ALinkerCore",
        description=(
            "ALinker/Core - read, propose, resolve. One Claim contract, one "
            "decomposition axis; replaces the three orthographic linkers and "
            "their three judges."
        ),
    ),
    "s_linker88": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker88",
        class_name="SLinker88",
        description=(
            "S-Linker88 - the compaction round's head: s_linker87 with no English "
            "changed at all. The byte inventory says the authored rules are 5.3% of a "
            "full-name judging call and 27.9% of it is anchor sentences the same call "
            "already printed, because a batch of 25 cases repeats a component's "
            "naming sentences once per case. s88 writes each component's anchors - "
            "the union of what every case for it in the batch would show, so no case "
            "is shown less - into the first such case and points the rest at it: "
            "-32.8% of the judging prompt, verified lossless case by case in "
            "pilot/test_s88_anchors.py. See results/compaction_round/README.md."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker89": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker89",
        class_name="SLinker89",
        description=(
            "S-Linker89 - the compaction round's composed head: s_linker88 plus the "
            "resolver's per-case 'CONTEXT: sentences Sx-Sy above.' line removed. That "
            "line named the +/-5 window of one target while the SENTENCES table is the "
            "union of ten, and 16.3 antecedents a run on terra (42.7 on luna) already "
            "cite a sentence outside the range their own case declares. Stage arm on "
            "both models, resolver AND the strict judge behind it: terra stage gold "
            "32.3 -> 36.3, macro F1 +0.1 (p = 0.70); luna 47.7 -> 52.3, F1 +0.1 "
            "(p = 0.90). 324 B off each of the 40 resolver calls a run makes. Neither "
            "change deletes a rule. E2E owed. See results/compaction_round/README.md."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker90": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker90",
        class_name="SLinker90",
        description=(
            "S-Linker90 - the static round's composed head: s_linker89 with five "
            "authored clauses paraphrased from recipes into concepts. QUALIFIED_CLAUSE "
            "drops 'joined or dotted' for 'a fragment of a longer identifier' (the "
            "screen finds the joined population 13.0 gold of 13.3 a run, so the wording "
            "over-reached); LAYERED_COREF_RULES drops its four-noun list and keeps the "
            "ground verbatim; STRICTER_CLAUSE trades 'Capitalization is evidence' for "
            "'How the word is written is evidence either way' and opens with the shared "
            "use/mention sentence; DOC_KNOWLEDGE_EXTRACTION_RULES drops the three-shape "
            "enumeration whose third shape is returned 0.0 times a run; "
            "ALIAS_EXCLUSION_RULES becomes the same identifier-fragment clause the "
            "judging prompt carries. Every one measured as a stage arm on both models "
            "and QUALITY-NEUTRAL. ENTITY_EXTRACTION_RULES is NOT paraphrased: that arm "
            "cost luna 3.7 gold candidates a run (p = 0.10) and was refused. Authored "
            "text 2107 -> 2053 B, which is not the point -- the point is that five "
            "clauses stop describing surfaces. Invariants: pilot/test_s90_static.py "
            "(90 checks). See results/static_round/README.md."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker91": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker91",
        class_name="SLinker91",
        description=(
            "S-Linker91 - the static round's minimal head after s_linker90 was refused. "
            "Only the two paraphrased constants that have exactly ONE consumer each and "
            "whose changed link sets are disjoint: STRICTER_CLAUSE (lenient judging; "
            "'Capitalization is evidence' -> 'How the word is written is evidence either "
            "way', opening with the shared use/mention sentence) and LAYERED_COREF_RULES "
            "(strict judging; the four-noun list dropped, the ground kept verbatim). "
            "Step-3 gate: mergeord changes 2.3 pairs a run and genartifact 21.0, and the "
            "pairs BOTH touch are 0.0 on either model. QUALIFIED_CLAUSE and the two alias "
            "constants are left at s89's, because s_linker90 carried them and lost on "
            "both models even though every one was neutral at every consumer. "
            "Invariants: pilot/test_s91_static.py (84 checks). "
            "See results/static_round/README.md."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker92": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker92",
        class_name="SLinker92",
        description=(
            "S-Linker92 - the static round's last candidate: s_linker89 with ONE "
            "constant paraphrased. LAYERED_COREF_RULES loses its four-noun list ('-- the "
            "data, the artifact, the request, the result --') and keeps its ground word "
            "for word. Stage arm on both models: terra gold 39.0 -> 41.0, luna 55.0 -> "
            "56.3, QUALITY-NEUTRAL on both. One consumer, the strict judge. Tried alone "
            "because s_linker90 (five constants) and s_linker91 (two) were both refused "
            "end to end although every constituent was neutral at every consumer. "
            "Invariants: pilot/test_s92_static.py. See results/static_round/README.md."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker75_null": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker75_null",
        class_name="SLinker75Null",
        description=(
            "S-Linker75Null - byte-identical to s_linker75 apart from the checkpoint "
            "namespace. The in-set harness null for the finetune round: it measures "
            "what a paired invocation reports as a difference when there is none, so "
            "the round's deltas are read against it rather than against zero."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker75": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker75",
        class_name="SLinker75",
        description=(
            "S-Linker75 - s_linker74 with the last four finetuned prompt spans gone. "
            "(1) `ALIAS_EXCLUSION_RULES` stops spelling `X.Y or X.Y.Z` and states the "
            "prohibition generally; `pilot/finetune_pilots.py --pilot aliascomp` "
            "re-measured the general round's reason for keeping it and did not "
            "reproduce it - the judged table reads 35.7 terms/run with the syntax "
            "against 39.3 without (FP +3.7, p = 0.90), not 24.0 against 36.7. "
            "(2) `ENTITY_EXTRACTION_RULES` loses its code-path clause and the "
            "extraction prompt carries `QUALIFIED_CLAUSE` instead (stage: TP +0.7 "
            "p = 1.00, FP -6.0 p = 0.20). (3) `P1_FOCUS` loses its code-level tail with "
            "the reject-enumeration left byte-identical - the edit s71 bundled with "
            "restructuring and never ran alone. (4) `LAYERED_COREF_RULES` loses the "
            "fifth restatement and adds nothing (stage: TP +4.7, FP +3.7). One "
            "sentence per distinction; 0 corpus-shaped bytes in the authored surface."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker74": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker74",
        class_name="SLinker74",
        description=(
            "S-Linker74 - s_linker70 with exactly one span changed: reject-condition "
            "(1) of the full-name judging rubric no longer spells `x.y or x.y.z` and "
            "reads 'referred to only as part of a longer joined or dotted identifier' "
            "instead. Located by three losses: restructuring the rubric costs ~0.8 F1 "
            "(s71 94.80 n=6, s72 94.94 n=3, against s70's 95.74), and removing 'a "
            "heading, or a list' costs exactly 2.7 TP in each of three runs (s73 "
            "95.25) - headings and lists are general documentation practice and were "
            "wrongly listed as corpus-shaped. Dotted identifiers really are "
            "corpus-shaped (62/198 sentences on one benchmark, 0-6 on the other four), "
            "so the syntax is the one span the bar catches."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker68": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker68",
        class_name="SLinker68",
        description=(
            "S-Linker68 — s_linker66 with one further cut: the mention label loses "
            "its qualified-path value, which was the only consumer of "
            "`_all_occurrences_in_qualified_path`. Two deterministic predicates and "
            "one label value fewer than s_linker65, with the two stated-name values "
            "kept separate (merging those is s_linker44, macro F1 -0.9 at n=6). "
            "Stage evidence (pilot/bind_pilots.py --pilot cutcodetoken, five samples "
            "a side on the recorded candidate set): TP +/-0.0 (p = 1.00), FP -0.2 "
            "(p = 1.00), composition p = 1.00. Three paired runs read macro TP -5.0, "
            "but four fifths of that gap is gold the extraction call -- whose prompt "
            "is byte-identical to s66's -- never proposed, on one project; restricted "
            "to the candidates both arms proposed it is TP -1.0 (p = 0.30) against a "
            "control of -0.7. NOT ADOPTED and NOT REFUTED."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker65": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker65",
        class_name="SLinker65",
        description=(
            "S-Linker65 — s_linker64 with no mechanism, prompt or behaviour change: "
            "its four lexical rules (the admission filter, the spelling-variant "
            "proposer, the stated-name net and the partial-name proposer) are "
            "restated as one relation, `_name_spans(text, name, form)`, over two "
            "dimensions -- how exactly the surface must reproduce the name, and "
            "whether the whole name or one word of it must be present. Which cell a "
            "proposer scans is the only thing that distinguishes it, so the three "
            "generators become three rows of `SCANS`. The identity is asserted, not "
            "claimed: pilot/test_s65_one_relation.py checks all 3697 (name, "
            "sentence) pairs, every candidate set of all three generators on all "
            "five projects, and 44 other method bodies byte-identical (49/49). "
            "Also removes the `_antecedent_states_name` wrapper."
        ),
        canonical=False,
        experimental=True,
    ),
    "s_linker64": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker64",
        class_name="SLinker64",
        description=(
            "S-Linker64 — s_linker62 plus a deterministic net at the full-name "
            "proposer for sentences that write a component's model name as spelled "
            "and that the extraction call never proposed. That bucket is where the "
            "partial-name proposer defers, and it was 3.0 gold pairs per run that no "
            "stage looked at again. Case-sensitive by design: the same scan run "
            "case-insensitively is 31.3 new pairs per run at 0.06 gold each against "
            "1.2 at 0.86. Stage-screened behind the unchanged two-pass judge at "
            "TP +1.2 (p = 0.01), FP +0.4 (p = 0.44)."
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


os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.6-terra")
os.environ.setdefault("CLAUDE_MODEL", "sonnet")


def describe_backend_target(backend: LLMBackend | None = None) -> str:
    backend = backend or get_backend()
    if backend == LLMBackend.CLAUDE:
        return f"claude ({os.environ.get('CLAUDE_MODEL', 'sonnet')})"
    if backend == LLMBackend.OPENAI:
        return f"openai ({os.environ.get('OPENAI_MODEL_NAME', 'gpt-5.6-terra')})"
    if backend == LLMBackend.CHECKPOINT:
        fallback_model = os.environ.get("CHECKPOINT_FALLBACK_MODEL", "").strip().lower()
        if fallback_model in {"gpt", "openai"} or fallback_model.startswith("gpt"):
            model = os.environ.get("OPENAI_MODEL_NAME", "gpt-5.6-terra")
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
            return f"checkpoint -> openai ({os.environ.get('OPENAI_MODEL_NAME', 'gpt-5.6-terra')})"
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
    f2 = 5 * tp / (5 * tp + 4 * fn + fp) if (tp + fp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "P": precision,
        "R": recall,
        "F1": f1,
        "F2": f2,
    }


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
        f"  {variant_name}: P={metrics['P']:.1%} R={metrics['R']:.1%} "
        f"F1={metrics['F1']:.1%} F2={metrics['F2']:.1%} "
        f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} ({elapsed:.0f}s)"
    )
    print(f"    Sources: {dict(source_counts)}")
    print(f"    FP by source: {dict(fp_by_source)}")

    result = {
        "variant": variant_name,
        "P": metrics["P"],
        "R": metrics["R"],
        "F1": metrics["F1"],
        "F2": metrics["F2"],
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
    if hasattr(linker, "orchestrator_workflow"):
        result["workflow"] = linker.orchestrator_workflow
    if hasattr(linker, "_llm_calls"):
        result["llm_calls"] = len(linker._llm_calls)
    return result


def print_summary(all_results: dict[str, dict[str, dict[str, object]]], selected_variants: list[str]) -> None:
    print(f"\n{'=' * 120}")
    print("SUMMARY")
    print(f"{'=' * 120}")
    header = f"{'Dataset':<16}"
    for variant in selected_variants:
        header += f" | {variant:^27}"
    print(header)
    print(f"{'-' * 16}" + ("-+-" + "-" * 27) * len(selected_variants))

    for dataset_name, dataset_results in all_results.items():
        row = f"{dataset_name:<16}"
        for variant in selected_variants:
            result = dataset_results.get(variant)
            if result is None:
                row += " | " + f"{'--':^27}"
            else:
                row += " | " + (
                    f"F1 {result['F1']:.1%} F2 {result['F2']:.1%} "
                    f"FP {result['fp']:>3}"
                )
        print(row)

    print(f"{'-' * 16}" + ("-+-" + "-" * 27) * len(selected_variants))
    row = f"{'Macro avg':<16}"
    for variant in selected_variants:
        values = [all_results[dataset][variant] for dataset in all_results if variant in all_results[dataset]]
        avg_f1 = sum(value["F1"] for value in values) / len(values)
        avg_f2 = sum(value["F2"] for value in values) / len(values)
        total_fp = sum(value["fp"] for value in values)
        row += " | " + f"F1 {avg_f1:.1%} F2 {avg_f2:.1%} FP {total_fp:>3}"
    print(row)

    row = f"{'Pooled':<16}"
    for variant in selected_variants:
        values = [
            all_results[dataset][variant]
            for dataset in all_results
            if variant in all_results[dataset]
        ]
        tp = sum(value["tp"] for value in values)
        fp = sum(value["fp"] for value in values)
        fn = sum(value["fn"] for value in values)
        pooled_f1 = 2 * tp / (2 * tp + fp + fn)
        pooled_f2 = 5 * tp / (5 * tp + 4 * fn + fp)
        row += " | " + f"F1 {pooled_f1:.1%} F2 {pooled_f2:.1%} FP {fp:>3}"
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
