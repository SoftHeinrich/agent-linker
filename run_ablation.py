#!/usr/bin/env python3
"""Ablation Study: Quantify AgentLinker improvements independently.

Runs 6 ablation variants on 5 benchmark datasets to isolate the impact of:
- Debate-validated coreference (V44-style propose+judge)
- Implicit reference detection (Phase 8)
- Sliding-batch entity extraction (overlapping 100-sentence windows)

Usage:
    python run_ablation.py
    python run_ablation.py --datasets jabref --variants baseline debate_coref
    python run_ablation.py --datasets teammates --variants baseline debate_no_implicit all_fixes
"""

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load .env file for API keys
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.core import DocumentLoader, SadSamLink

VARIANTS = {
    "baseline":           dict(use_debate_coref=False, enable_implicit=True,  use_sliding_batch=False),
    "debate_coref":       dict(use_debate_coref=True,  enable_implicit=True,  use_sliding_batch=False),
    "no_implicit":        dict(use_debate_coref=False, enable_implicit=False, use_sliding_batch=False),
    "debate_no_implicit": dict(use_debate_coref=True,  enable_implicit=False, use_sliding_batch=False),
    "sliding_batch":      dict(use_debate_coref=False, enable_implicit=True,  use_sliding_batch=True),
    "all_fixes":          dict(use_debate_coref=True,  enable_implicit=False, use_sliding_batch=True),
    # --- Hybrid variants ---
    "discourse_judge":    dict(coref_mode="discourse_judge", implicit_mode="on"),
    "adaptive":           dict(coref_mode="adaptive", implicit_mode="adaptive"),
    "dj_no_implicit":     dict(coref_mode="discourse_judge", implicit_mode="off"),
    # --- CoT variants ---
    "cot_implicit":       dict(coref_mode="adaptive", implicit_mode="cot"),
    "cot_judge":          dict(coref_mode="adaptive", implicit_mode="adaptive", judge_mode="cot"),
    "cot_both":           dict(coref_mode="adaptive", implicit_mode="adaptive_cot", judge_mode="cot"),
    "cot_transarc":       dict(coref_mode="adaptive", implicit_mode="adaptive", judge_mode="cot_transarc"),
    # --- V2: qualitative (no numeric thresholds) ---
    "v2":                 dict(linker_class="v2"),
    "v2_adaptive":        dict(linker_class="v2", coref_mode="adaptive", implicit_mode="adaptive"),
    # --- V2 recovery ablation ---
    "v2_skip_ambig":      dict(linker_class="v2", coref_mode="adaptive", implicit_mode="adaptive", recovery_mode="skip_ambiguous"),
    "v2_no_recovery":     dict(linker_class="v2", coref_mode="adaptive", implicit_mode="adaptive", recovery_mode="off"),
    # --- V2 semantic filter ablation ---
    "v2_f_embed":         dict(linker_class="v2", coref_mode="adaptive", implicit_mode="adaptive", post_filter="embedding"),
    "v2_f_tfidf":         dict(linker_class="v2", coref_mode="adaptive", implicit_mode="adaptive", post_filter="tfidf"),
    "v2_f_lexical":       dict(linker_class="v2", coref_mode="adaptive", implicit_mode="adaptive", post_filter="lexical"),
    # --- V3: self-contained qualitative + semantic filters ---
    "v3":                 dict(linker_class="v3"),
    "v3_embed":           dict(linker_class="v3", post_filter="embedding"),
    "v3_tfidf":           dict(linker_class="v3", post_filter="tfidf"),
    "v3_lexical":         dict(linker_class="v3", post_filter="lexical"),
    "v3_selective":       dict(linker_class="v3", post_filter="selective"),
    "v3_selective_all":   dict(linker_class="v3", post_filter="selective_all"),
    # --- V4: no data leakage, all thresholds derived from input ---
    "v4":                 dict(linker_class="v4"),
    "v4_multi_vote":      dict(linker_class="v4", judge_mode="multi_vote"),
    "v4_source_lenient":  dict(linker_class="v4", judge_mode="source_lenient"),
    "v4_mv_selective":    dict(linker_class="v4", judge_mode="multi_vote", post_filter="selective_all"),
    "v4_sl_selective":    dict(linker_class="v4", judge_mode="source_lenient", post_filter="selective_all"),
    # V4 complexity fixes
    "v4_structural":      dict(linker_class="v4", complexity_mode="structural"),
    "v4_llm_v2":          dict(linker_class="v4", complexity_mode="llm_v2"),
    # V4 direction experiments
    "v4_str_high":        dict(linker_class="v4", complexity_mode="structural_high"),
    "v4_str_norec":       dict(linker_class="v4", complexity_mode="structural", recovery_mode="off_complex"),
    "v4_str_jrec":        dict(linker_class="v4", complexity_mode="structural", recovery_mode="judge"),
    # --- V5: consolidated best approach ---
    "v5":                 dict(linker_class="v5"),
    # --- V6: dot-filter fix + generic judge examples ---
    "v6":                 dict(linker_class="v6"),
    "v6b":                dict(linker_class="v6b"),  # V6 + abbreviation guard
    "v6c":                dict(linker_class="v6c"),  # V6 + deterministic boundary filters
    # --- V6 voting strategies ---
    "v6_vote":            dict(linker_class="v6_vote", n_runs=3),
    "v6_phase_vote":      dict(linker_class="v6_phase_vote", n_runs=3),
    # --- V7: learned confusion patterns ---
    "v7":                 dict(linker_class="v7"),
    # --- V8: refined approaches ---
    "v8a":                dict(linker_class="v8a", n_runs=3),
    "v8b":                dict(linker_class="v8b"),
    # --- V9: consolidated majority voting ---
    "v9":                 dict(linker_class="v9", n_runs=3),
    # V4 isolation: individual fixes toggled off (old behavior) to find regression source
    # Base = v4_str_jrec with all fixes (embed=name_only, unjudged=rejudge)
    "v4_iso_old_embed":   dict(linker_class="v4", complexity_mode="structural", recovery_mode="judge", embed_mode="context", unjudged_mode="rejudge"),
    "v4_iso_old_judge":   dict(linker_class="v4", complexity_mode="structural", recovery_mode="judge", embed_mode="name_only", unjudged_mode="approve"),
    "v4_iso_old_both":    dict(linker_class="v4", complexity_mode="structural", recovery_mode="judge", embed_mode="context", unjudged_mode="approve"),
    # --- V9T: error-driven features on V5 ---
    "v9t":      dict(linker_class="v9t"),  # all features on
    "v9t_A":    dict(linker_class="v9t", only_feature="A"),
    "v9t_B":    dict(linker_class="v9t", only_feature="B"),
    "v9t_C":    dict(linker_class="v9t", only_feature="C"),
    "v9t_D":    dict(linker_class="v9t", only_feature="D"),
    "v9t_E":    dict(linker_class="v9t", only_feature="E"),
    "v9t_F":    dict(linker_class="v9t", only_feature="F"),
    "v9t_noE":  dict(linker_class="v9t", ambiguous_recovery=False),  # all except riskiest
    "v9t_ABD":  dict(linker_class="v9t", fix_dot_filter=True, abbreviation_guard=True,
                     coref_distance_filter=False, generic_word_filter=True,
                     ambiguous_recovery=False, section_heading_safe=False),  # best safe combo
    "v9t_AB":   dict(linker_class="v9t", fix_dot_filter=True, abbreviation_guard=True,
                     coref_distance_filter=False, generic_word_filter=False,
                     ambiguous_recovery=False, section_heading_safe=False),  # A+B only
    # --- V11: generic-name-aware pipeline ---
    "v11":                dict(linker_class="v11"),
    # --- V12: V11 + contextual stoplist + test-infra filter ---
    "v12":                dict(linker_class="v12"),
    # --- V13: targeted generic filtering + TransArc protection ---
    "v13":                dict(linker_class="v13"),
    # --- V14: deterministic Phase 1 + structured judge + enriched prompts ---
    "v14":                dict(linker_class="v14"),
    # --- V15: V14 + TransArc immunity + split validation + targeted recovery ---
    "v15":                dict(linker_class="v15"),
    # --- V16: V15 + trimmed ambiguous list + no data leakage ---
    "v16":                dict(linker_class="v16"),
    # --- W16: V16 + 3 targeted fixes (parent overlap, link-aware antecedent, generic overrides) ---
    "w16":                dict(linker_class="w16"),
    # --- V17: V16 + togglable V6 features ---
    "v17_A":              dict(linker_class="v17", intersect_validation=True),
    "v17_BC":             dict(linker_class="v17", enable_implicit=True, enable_fn_recovery=True),
    "v17_D":              dict(linker_class="v17", nuanced_transarc=True),
    "v17_ABCD":           dict(linker_class="v17", intersect_validation=True, enable_implicit=True, enable_fn_recovery=True, nuanced_transarc=True),
    # --- V17 Judge prompt fixes (all include A:intersect as base) ---
    "v17_A_J1":           dict(linker_class="v17", intersect_validation=True, judge_doc_knowledge=True),
    "v17_A_J2":           dict(linker_class="v17", intersect_validation=True, judge_full_context=True),
    "v17_A_J3":           dict(linker_class="v17", intersect_validation=True, judge_validated_bias=True),
    # --- V17 Judge prompt v2 fixes ---
    "v17_A_J4":           dict(linker_class="v17", intersect_validation=True, judge_strict_def=True),
    "v17_A_J5":           dict(linker_class="v17", intersect_validation=True, judge_adaptive_ctx=True),
    "v17_A_J6":           dict(linker_class="v17", intersect_validation=True, judge_show_match=True),
    "v17_A_J456":         dict(linker_class="v17", intersect_validation=True, judge_strict_def=True, judge_adaptive_ctx=True, judge_show_match=True),
    "v17_A_J56":          dict(linker_class="v17", intersect_validation=True, judge_adaptive_ctx=True, judge_show_match=True),
    "v18":                dict(linker_class="v18"),
    "v19":                dict(linker_class="v19"),
    "v20a":               dict(linker_class="v20a"),
    "v20b":               dict(linker_class="v20b"),
    "v20c":               dict(linker_class="v20c"),
    "v20":                dict(linker_class="v20"),
    # --- V21-V23: prompt optimization series ---
    "v21":                dict(linker_class="v21"),
    "v22":                dict(linker_class="v22"),
    "v23":                dict(linker_class="v23"),
    "v23a":               dict(linker_class="v23a"),
    "v23b":               dict(linker_class="v23b"),
    "v23c":               dict(linker_class="v23c"),
    "v23d":               dict(linker_class="v23d"),
    "v23e":               dict(linker_class="v23e"),
    # --- V24: hardcoded lists removed ---
    "v24":                dict(linker_class="v24"),
    # --- V25: benchmark-clean SE textbook prompts ---
    "v25":                dict(linker_class="v25"),
    # --- V25a/b: V25 + deliberation TransArc judge ---
    "v25a":               dict(linker_class="v25a"),
    "v25b":               dict(linker_class="v25b"),
    "v26":                dict(linker_class="v26"),
    "v26a":               dict(linker_class="v26a"),
    "v26b":               dict(linker_class="v26b"),
    "v26c":               dict(linker_class="v26c"),
    "v26d":               dict(linker_class="v26d"),
    # --- V27 family: GoT / Deliberation experiments ---
    "v27g":               dict(linker_class="v27g"),  # GoT sub-judges + source-aware weighting
    "v27b":               dict(linker_class="v27b"),  # Deliberation Phase 3B generic judge
    "v27f":               dict(linker_class="v27f"),  # Parallel branch pipeline + GoT aggregation
    "v28":                dict(linker_class="v28"),    # V26d + package-path TransArc filter
    "v29":                dict(linker_class="v29"),    # V26a + decomposed contrastive TransArc judge
    # --- W24: V24 + deliberation TransArc judge ---
    "w24":                dict(linker_class="w24"),
    "w24-scratchpad":     dict(linker_class="w24", transarc_judge_strategy="scratchpad"),
    "w24-batna":          dict(linker_class="w24", transarc_judge_strategy="batna"),
    # --- ILinker: pure LLM (no TransArc) ---
    "i1":                 dict(linker_class="i1"),
    "i2":                 dict(linker_class="i2"),       # precision-focused, no contextual
    "v26a_i2":            dict(linker_class="v26a_i2"),  # V26a with ILinker2 replacing TransArc
    "v26a_i2_dk":         dict(linker_class="v26a_i2_dk"),  # V26a+I2 + DA warm-start
    "i2_pure":            dict(linker_class="i2_pure"),      # Zero-heuristic: I2 + LLM synonym/coref/validation
    "v26a_i2_ndf":        dict(linker_class="v26a_i2_ndf"),  # V26a+I2 without dot filter
    # --- Phase 3 stabilization variants ---
    "s1":                 dict(linker_class="s1"),   # Fix B restricted to abbrevs only
    "s2":                 dict(linker_class="s2"),   # Phase 6 min alias len >= 3
    "s3":                 dict(linker_class="s3"),   # s1 + s2 combined
    "s4":                 dict(linker_class="s4"),   # 3x extraction voting
    "s5":                 dict(linker_class="s5"),   # All-reject guard
    "s6":                 dict(linker_class="s6"),   # Full stack (s1+s2+s4+s5)
    # --- S7-S10: General-rule stabilization (no hardcoded length thresholds) ---
    "s7":                 dict(linker_class="s7"),   # Design A: prompt-hardened
    "s8":                 dict(linker_class="s8"),   # Design B: trust-the-judge
    "s9":                 dict(linker_class="s9"),   # Design C: doc-frequency gating
    "s10":                dict(linker_class="s10"),  # Design A+B combined
    "s11":                dict(linker_class="s11"),  # s7 cleaned: Fix B/V24 removed
    # --- V30/V30a: ILinker2 + V26a with hardened Phase 3 ---
    "v30":                dict(linker_class="v30"),   # ILinker2 + V26a + hardened prompts + Fix A/C
    "v30a":               dict(linker_class="v30a"),  # ILinker2 + V26a + prompt-only Phase 3
    "v30b":               dict(linker_class="v30b"),  # ILinker2 + V26a + few-shot calibrated judge
    "v30c":               dict(linker_class="v30c"),  # V30b + NDF: few-shot judge + no dot filter
    "v30d":               dict(linker_class="v30d"),  # V30c + resume + CamelCase Phase 3 override
    "v30d_r3":            dict(linker_class="v30d", resume_from=3),  # Resume from phase 3
    "v30d_r9":            dict(linker_class="v30d", resume_from=9),  # Resume from pre-judge
    "v30d_p3":            dict(linker_class="v30d", run_only=3),  # Phase 3: CamelCase only
    "v30d_p3_v24":        dict(linker_class="v30d", run_only=3, v30d_v24=True),  # Phase 3: CC + V24 overrides
    "v30d_p3_all":        dict(linker_class="v30d", run_only=3, v30d_v24=True, v30d_uc=True),  # Phase 3: CC + V24 + uppercase
    "v30d_p3_pt5":        dict(linker_class="v30d", run_only=3, v30d_pt=5),  # Phase 3: partial threshold 5
    "v30d_p3_pt7":        dict(linker_class="v30d", run_only=3, v30d_pt=7),  # Phase 3: partial threshold 7
    "v30d_p6":            dict(linker_class="v30d", run_only=6),  # Single: Phase 6 only
    "v30d_p7":            dict(linker_class="v30d", run_only=7),  # Single: Phase 7 only
    "v30d_p9":            dict(linker_class="v30d", run_only=9),  # Single: Phase 9 only
    # --- V30d heuristic ablation (resume from affected phase) ---
    "v30d_no_genm":       dict(linker_class="v30d", resume_from=6, no_genm=True),      # Phase 6: no _is_generic_mention
    "v30d_no_gencf":      dict(linker_class="v30d", resume_from=7, no_gencf=True),     # Phase 7: no _filter_generic_coref
    "v30d_no_pron":       dict(linker_class="v30d", resume_from=7, no_pron=True),      # Phase 7: no _deterministic_pronoun_coref
    "v30d_no_bf":         dict(linker_class="v30d", resume_from=8, no_bf=True),        # Phase 8c: no boundary filters
    "v30d_no_pi":         dict(linker_class="v30d", resume_from=8, no_pi=True),        # Phase 8b: no partial injection
    "v30d_no_synsafe":    dict(linker_class="v30d", resume_from=9, no_synsafe=True),   # Phase 9: no synonym-safe bypass
    "v30d_no_po":         dict(linker_class="v30d", resume_from=8, no_po=True),        # Phase 8: no parent-overlap guard
    "v30d_no_ag":         dict(linker_class="v30d", resume_from=5, no_ag=True),        # Phase 5: no abbreviation guard
    # Single-phase tests for LLM-dependent heuristics
    "v30d_p9_no_synsafe": dict(linker_class="v30d", run_only=9, no_synsafe=True),     # P9 only: judge sees all links
    "v30d_p9_no_pi":      dict(linker_class="v30d", run_only=9, no_pi=True),          # P9 only: no partial injection (uses pre_judge from v30c)
    # --- V31: V30c + CamelCase rescue only ---
    "v31":                dict(linker_class="v31"),
    # --- V32: V31 + convention filter covers partial_inject + zero prompt leakage ---
    "v32":                dict(linker_class="v32"),
    # --- ALinker: Adaptive Agent Linker with orchestrator, monitor, review agents ---
    "alinker":            dict(linker_class="alinker"),
    # --- V33: V32 + GPT-5.2 prompt fixes (generic mention, judge, validation, coref) ---
    "v33":                dict(linker_class="v33"),
    # --- V34: V33 + simplified prompts (26% fewer tokens) ---
    "v34":                dict(linker_class="v34"),
    # --- V35: V32 + 6 prompt proposals (example guide, compact P3, GPT self-consistency) ---
    "v35":                dict(linker_class="v35"),
    "v35a":               dict(linker_class="v35a"),  # P2 only: example-driven CONVENTION_GUIDE
    "v35b":               dict(linker_class="v35b"),  # P6 only: compact Phase 3 judge
    "v35c":               dict(linker_class="v35c"),  # P4 only: concrete JSON output examples

    "v36a":               dict(linker_class="v36a"),  # V32 + ILinker2 few-shot examples (Phase 4)
    "v36b":               dict(linker_class="v36b"),  # V32 + Phase 9 judge few-shot examples

    "v37":                dict(linker_class="v37"),    # V32 + structural syn-safe restriction
    "v38":                dict(linker_class="v38"),    # V32 + context-aware judge replacing syn-safe bypass
    "v39":                dict(linker_class="v39"),    # V38 + LLM partial usage classification for targeted syn-safe
    "v39a":               dict(linker_class="v39a"),   # V39 + Phase 5 two-pass intersection for variance reduction
    "v40a":               dict(linker_class="v40a"),   # V39a + exempt coref from Phase 9 judge
    "v40b":               dict(linker_class="v40b"),   # V39a + Phase 7 two-pass union coref + coref exempt
    "v40c":               dict(linker_class="v40c"),   # V39a + LLM generic mention detection + coref exempt
    "s_linker":           dict(linker_class="s_linker"), # DAG-based standalone (V39 + dead code removal + DAG tiers)
    "s_linker2":          dict(linker_class="s_linker2"), # S-Linker + V40c (LLM generic detection + coref exempt + bug fixes)
    "s_linker3":          dict(linker_class="s_linker3"), # S-Linker2 + unified coref (Variant E) + keep_coref (no judge)
    "s_linker4":          dict(linker_class="s_linker4"), # S-Linker3 + seed links go through convention filter (no immunity)
    # --- CNR: Component Name Recovery (no-model) ---
    "cnr":                dict(linker_class="cnr"),        # Discovery + simple extraction
    "cnr_i2":             dict(linker_class="cnr_i2"),     # Discovery + I2 two-pass
    "cnr_v26a":           dict(linker_class="cnr_v26a"),   # Discovery + full V26a pipeline
    # --- CNR-DK: Document Knowledge-Informed CNR ---
    "cnr_dk":             dict(linker_class="cnr_dk"),     # CNR + Document Analysis
    "cnr_dk_v26a":        dict(linker_class="cnr_dk_v26a"),  # CNR-DK + full V26a + Phase 3 warm-start
}

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
CLI_RESULTS = Path("/mnt/hostshare/ardoco-home/cli-results")

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

_backend_env = os.environ.get("LLM_BACKEND", "claude")
BACKEND = LLMBackend.OPENAI if _backend_env == "openai" else LLMBackend.CLAUDE
os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.2")
os.environ.setdefault("CLAUDE_MODEL", "sonnet")


def load_gold_sam(gold_path: str) -> set[tuple[int, str]]:
    links = set()
    with open(gold_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def load_transarc_pairs(transarc_path: str) -> set[tuple[int, str]]:
    pairs = set()
    with open(transarc_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                pairs.add((int(snum), cid))
    return pairs


def eval_metrics(predicted, gold):
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"tp": tp, "fp": fp, "fn": fn, "P": p, "R": r, "F1": f1}


def run_variant(variant_name: str, flags: dict, ds_name: str, paths: dict,
                gold_pairs: set, transarc_pairs: set, id_to_name: dict, sent_map: dict,
                resume_from_phase=None):
    """Run a single ablation variant on a single dataset."""
    from llm_sad_sam.linkers.experimental.agent_linker import export_links_csv

    print(f"\n  --- Variant: {variant_name} ---")
    print(f"  Flags: {flags}")

    t0 = time.time()
    flags = dict(flags)  # copy to avoid mutating VARIANTS
    linker_class = flags.pop("linker_class", None)
    if linker_class == "v9":
        from llm_sad_sam.linkers.experimental.agent_linker_v9 import AgentLinkerV9
        pf = flags.pop("post_filter", "none")
        n_runs = flags.pop("n_runs", 3)
        linker = AgentLinkerV9(backend=BACKEND, post_filter=pf, n_runs=n_runs)
    elif linker_class == "v8a":
        from llm_sad_sam.linkers.experimental.agent_linker_v8a import AgentLinkerV8a
        pf = flags.pop("post_filter", "none")
        n_runs = flags.pop("n_runs", 3)
        linker = AgentLinkerV8a(backend=BACKEND, post_filter=pf, n_runs=n_runs)
    elif linker_class == "v8b":
        from llm_sad_sam.linkers.experimental.agent_linker_v8b import AgentLinkerV8b
        pf = flags.pop("post_filter", "none")
        linker = AgentLinkerV8b(backend=BACKEND, post_filter=pf)
    elif linker_class == "v7":
        from llm_sad_sam.linkers.experimental.agent_linker_v7 import AgentLinkerV7
        pf = flags.pop("post_filter", "none")
        linker = AgentLinkerV7(backend=BACKEND, post_filter=pf)
    elif linker_class == "v6_vote":
        from llm_sad_sam.linkers.experimental.agent_linker_v6_vote import AgentLinkerV6Vote
        pf = flags.pop("post_filter", "none")
        n_runs = flags.pop("n_runs", 3)
        linker = AgentLinkerV6Vote(backend=BACKEND, post_filter=pf, n_runs=n_runs)
    elif linker_class == "v6_phase_vote":
        from llm_sad_sam.linkers.experimental.agent_linker_v6_phase_vote import AgentLinkerV6PhaseVote
        pf = flags.pop("post_filter", "none")
        n_runs = flags.pop("n_runs", 3)
        linker = AgentLinkerV6PhaseVote(backend=BACKEND, post_filter=pf, n_runs=n_runs)
    elif linker_class == "v23":
        from llm_sad_sam.linkers.experimental.agent_linker_v23 import AgentLinkerV23
        linker = AgentLinkerV23(backend=BACKEND)
    elif linker_class == "v23a":
        from llm_sad_sam.linkers.experimental.agent_linker_v23a import AgentLinkerV23a
        linker = AgentLinkerV23a(backend=BACKEND)
    elif linker_class == "v23b":
        from llm_sad_sam.linkers.experimental.agent_linker_v23b import AgentLinkerV23b
        linker = AgentLinkerV23b(backend=BACKEND)
    elif linker_class == "v23c":
        from llm_sad_sam.linkers.experimental.agent_linker_v23c import AgentLinkerV23c
        linker = AgentLinkerV23c(backend=BACKEND)
    elif linker_class == "v23d":
        from llm_sad_sam.linkers.experimental.agent_linker_v23d import AgentLinkerV23d
        linker = AgentLinkerV23d(backend=BACKEND)
    elif linker_class == "v23e":
        from llm_sad_sam.linkers.experimental.agent_linker_v23e import AgentLinkerV23e
        linker = AgentLinkerV23e(backend=BACKEND)
    elif linker_class == "v24":
        from llm_sad_sam.linkers.experimental.agent_linker_v24 import AgentLinkerV24
        linker = AgentLinkerV24(backend=BACKEND)
    elif linker_class == "w24":
        from llm_sad_sam.linkers.experimental.agent_linker_w24 import AgentLinkerW24
        strategy = cfg.get("transarc_judge_strategy", "advocate")
        linker = AgentLinkerW24(backend=BACKEND, transarc_judge_strategy=strategy)
    elif linker_class == "v25":
        from llm_sad_sam.linkers.experimental.agent_linker_v25 import AgentLinkerV25
        linker = AgentLinkerV25(backend=BACKEND)
    elif linker_class == "v25a":
        from llm_sad_sam.linkers.experimental.agent_linker_v25a import AgentLinkerV25a
        linker = AgentLinkerV25a(backend=BACKEND)
    elif linker_class == "v25b":
        from llm_sad_sam.linkers.experimental.agent_linker_v25b import AgentLinkerV25b
        linker = AgentLinkerV25b(backend=BACKEND)
    elif linker_class == "v26":
        from llm_sad_sam.linkers.experimental.agent_linker_v26 import AgentLinkerV26
        linker = AgentLinkerV26(backend=BACKEND)
    elif linker_class == "v26a":
        from llm_sad_sam.linkers.experimental.agent_linker_v26a import AgentLinkerV26a
        linker = AgentLinkerV26a(backend=BACKEND)
    elif linker_class == "v26b":
        from llm_sad_sam.linkers.experimental.agent_linker_v26b import AgentLinkerV26b
        linker = AgentLinkerV26b(backend=BACKEND)
    elif linker_class == "v26c":
        from llm_sad_sam.linkers.experimental.agent_linker_v26c import AgentLinkerV26c
        linker = AgentLinkerV26c(backend=BACKEND)
    elif linker_class == "v26d":
        from llm_sad_sam.linkers.experimental.agent_linker_v26d import AgentLinkerV26d
        linker = AgentLinkerV26d(backend=BACKEND)
    elif linker_class == "v29":
        from llm_sad_sam.linkers.experimental.agent_linker_v29 import AgentLinkerV29
        linker = AgentLinkerV29(backend=BACKEND)
    elif linker_class == "v28":
        from llm_sad_sam.linkers.experimental.agent_linker_v28 import AgentLinkerV28
        linker = AgentLinkerV28(backend=BACKEND)
    elif linker_class == "v27g":
        from llm_sad_sam.linkers.experimental.agent_linker_v27g import AgentLinkerV27g
        linker = AgentLinkerV27g(backend=BACKEND)
    elif linker_class == "v27b":
        from llm_sad_sam.linkers.experimental.agent_linker_v27b import AgentLinkerV27b
        linker = AgentLinkerV27b(backend=BACKEND)
    elif linker_class == "v27f":
        from llm_sad_sam.linkers.experimental.agent_linker_v27f import AgentLinkerV27f
        linker = AgentLinkerV27f(backend=BACKEND)
    elif linker_class == "v22":
        from llm_sad_sam.linkers.experimental.agent_linker_v22 import AgentLinkerV22
        linker = AgentLinkerV22(backend=BACKEND)
    elif linker_class == "v21":
        from llm_sad_sam.linkers.experimental.agent_linker_v21 import AgentLinkerV21
        linker = AgentLinkerV21(backend=BACKEND)
    elif linker_class == "v18":
        from llm_sad_sam.linkers.experimental.agent_linker_v18 import AgentLinkerV18
        linker = AgentLinkerV18(backend=BACKEND)
    elif linker_class == "v19":
        from llm_sad_sam.linkers.experimental.agent_linker_v19 import AgentLinkerV19
        linker = AgentLinkerV19(backend=BACKEND)
    elif linker_class == "v20a":
        from llm_sad_sam.linkers.experimental.agent_linker_v20a import AgentLinkerV20a
        linker = AgentLinkerV20a(backend=BACKEND)
    elif linker_class == "v20b":
        from llm_sad_sam.linkers.experimental.agent_linker_v20b import AgentLinkerV20b
        linker = AgentLinkerV20b(backend=BACKEND)
    elif linker_class == "v20c":
        from llm_sad_sam.linkers.experimental.agent_linker_v20c import AgentLinkerV20c
        linker = AgentLinkerV20c(backend=BACKEND)
    elif linker_class == "v20":
        from llm_sad_sam.linkers.experimental.agent_linker_v20 import AgentLinkerV20
        linker = AgentLinkerV20(backend=BACKEND)
    elif linker_class == "v17":
        from llm_sad_sam.linkers.experimental.agent_linker_v17 import AgentLinkerV17
        linker = AgentLinkerV17(
            backend=BACKEND,
            intersect_validation=flags.pop("intersect_validation", False),
            enable_implicit=flags.pop("enable_implicit", False),
            enable_fn_recovery=flags.pop("enable_fn_recovery", False),
            nuanced_transarc=flags.pop("nuanced_transarc", False),
            judge_doc_knowledge=flags.pop("judge_doc_knowledge", False),
            judge_full_context=flags.pop("judge_full_context", False),
            judge_validated_bias=flags.pop("judge_validated_bias", False),
            judge_strict_def=flags.pop("judge_strict_def", False),
            judge_adaptive_ctx=flags.pop("judge_adaptive_ctx", False),
            judge_show_match=flags.pop("judge_show_match", False),
        )
    elif linker_class == "w16":
        from llm_sad_sam.linkers.experimental.agent_linker_w16 import AgentLinkerW16
        linker = AgentLinkerW16(backend=BACKEND)
    elif linker_class == "v16":
        from llm_sad_sam.linkers.experimental.agent_linker_v16 import AgentLinkerV16
        linker = AgentLinkerV16(backend=BACKEND)
    elif linker_class == "v15":
        from llm_sad_sam.linkers.experimental.agent_linker_v15 import AgentLinkerV15
        linker = AgentLinkerV15(backend=BACKEND)
    elif linker_class == "v14":
        from llm_sad_sam.linkers.experimental.agent_linker_v14 import AgentLinkerV14
        pf = flags.pop("post_filter", "none")
        linker = AgentLinkerV14(backend=BACKEND, post_filter=pf)
    elif linker_class == "v13":
        from llm_sad_sam.linkers.experimental.agent_linker_v13 import AgentLinkerV13
        pf = flags.pop("post_filter", "none")
        linker = AgentLinkerV13(backend=BACKEND, post_filter=pf)
    elif linker_class == "v12":
        from llm_sad_sam.linkers.experimental.agent_linker_v12 import AgentLinkerV12
        pf = flags.pop("post_filter", "none")
        linker = AgentLinkerV12(backend=BACKEND, post_filter=pf)
    elif linker_class == "v11":
        from llm_sad_sam.linkers.experimental.agent_linker_v11 import AgentLinkerV11
        pf = flags.pop("post_filter", "none")
        linker = AgentLinkerV11(backend=BACKEND, post_filter=pf)
    elif linker_class == "v6":
        from llm_sad_sam.linkers.experimental.agent_linker_v6 import AgentLinkerV6
        pf = flags.pop("post_filter", "none")
        linker = AgentLinkerV6(backend=BACKEND, post_filter=pf)
    elif linker_class == "v6b":
        from llm_sad_sam.linkers.experimental.agent_linker_v6b import AgentLinkerV6B
        pf = flags.pop("post_filter", "none")
        linker = AgentLinkerV6B(backend=BACKEND, post_filter=pf)
    elif linker_class == "v6c":
        from llm_sad_sam.linkers.experimental.agent_linker_v6c import AgentLinkerV6C
        pf = flags.pop("post_filter", "none")
        linker = AgentLinkerV6C(backend=BACKEND, post_filter=pf)
    elif linker_class == "v9t":
        from llm_sad_sam.linkers.experimental.agent_linker_v9t import AgentLinkerV9T
        pf = flags.pop("post_filter", "none")
        only_feature = flags.pop("only_feature", None)
        # Feature flags: default all True unless only_feature is set
        feature_map = {"A": "fix_dot_filter", "B": "abbreviation_guard",
                       "C": "coref_distance_filter", "D": "generic_word_filter",
                       "E": "ambiguous_recovery", "F": "section_heading_safe"}
        if only_feature:
            feature_kwargs = {v: (k == only_feature) for k, v in feature_map.items()}
        else:
            feature_kwargs = {v: flags.pop(v, True) for v in feature_map.values()}
        linker = AgentLinkerV9T(backend=BACKEND, post_filter=pf, **feature_kwargs)
    elif linker_class == "v5":
        from llm_sad_sam.linkers.experimental.agent_linker_v5 import AgentLinkerV5
        pf = flags.pop("post_filter", "none")
        linker = AgentLinkerV5(backend=BACKEND, post_filter=pf)
    elif linker_class == "v4":
        from llm_sad_sam.linkers.experimental.agent_linker_v4 import AgentLinkerV4
        pf = flags.pop("post_filter", "none")
        jm = flags.pop("judge_mode", "default")
        cm = flags.pop("complexity_mode", "llm")
        rm = flags.pop("recovery_mode", "default")
        em = flags.pop("embed_mode", "name_only")
        um = flags.pop("unjudged_mode", "rejudge")
        linker = AgentLinkerV4(backend=BACKEND, post_filter=pf, judge_mode=jm, complexity_mode=cm, recovery_mode=rm, embed_mode=em, unjudged_mode=um)
    elif linker_class == "v3":
        from llm_sad_sam.linkers.experimental.agent_linker_v3 import AgentLinkerV3
        pf = flags.pop("post_filter", "none")
        linker = AgentLinkerV3(backend=BACKEND, post_filter=pf)
    elif linker_class == "v2":
        from llm_sad_sam.linkers.experimental.agent_linker_v2 import AgentLinkerV2
        if any(k in flags for k in ("coref_mode", "implicit_mode", "judge_mode", "recovery_mode", "post_filter")):
            from llm_sad_sam.linkers.experimental.agent_linker_v2_ablation import AgentLinkerV2Ablation
            linker = AgentLinkerV2Ablation(backend=BACKEND, **flags)
        else:
            linker = AgentLinkerV2(backend=BACKEND)
    elif linker_class == "i1":
        from llm_sad_sam.linkers.experimental.ilinker1 import ILinker1
        linker = ILinker1(backend=BACKEND)
    elif linker_class == "i2":
        from llm_sad_sam.linkers.experimental.ilinker2 import ILinker2
        linker = ILinker2(backend=BACKEND)
    elif linker_class == "v26a_i2":
        from llm_sad_sam.linkers.experimental.ilinker2_v26a import ILinker2V26a
        linker = ILinker2V26a(backend=BACKEND)
    elif linker_class == "v26a_i2_dk":
        from llm_sad_sam.linkers.experimental.ilinker2_v26a import ILinker2V26a
        linker = ILinker2V26a(backend=BACKEND, enable_da=True)
    elif linker_class == "i2_pure":
        from llm_sad_sam.linkers.experimental.ilinker2_pure import ILinker2Pure
        linker = ILinker2Pure(backend=BACKEND)
    elif linker_class == "v26a_i2_ndf":
        from llm_sad_sam.linkers.experimental.ilinker2_v26a_ndf import ILinker2V26aNDF
        linker = ILinker2V26aNDF(backend=BACKEND)
    elif linker_class == "s1":
        from llm_sad_sam.linkers.experimental.ilinker2_v26a_s1 import ILinker2V26aS1
        linker = ILinker2V26aS1(backend=BACKEND)
    elif linker_class == "s2":
        from llm_sad_sam.linkers.experimental.ilinker2_v26a_s2 import ILinker2V26aS2
        linker = ILinker2V26aS2(backend=BACKEND)
    elif linker_class == "s3":
        from llm_sad_sam.linkers.experimental.ilinker2_v26a_s3 import ILinker2V26aS3
        linker = ILinker2V26aS3(backend=BACKEND)
    elif linker_class == "s4":
        from llm_sad_sam.linkers.experimental.ilinker2_v26a_s4 import ILinker2V26aS4
        linker = ILinker2V26aS4(backend=BACKEND)
    elif linker_class == "s5":
        from llm_sad_sam.linkers.experimental.ilinker2_v26a_s5 import ILinker2V26aS5
        linker = ILinker2V26aS5(backend=BACKEND)
    elif linker_class == "s6":
        from llm_sad_sam.linkers.experimental.ilinker2_v26a_s6 import ILinker2V26aS6
        linker = ILinker2V26aS6(backend=BACKEND)
    elif linker_class == "s7":
        from llm_sad_sam.linkers.experimental.ilinker2_v26a_s7 import ILinker2V26aS7
        linker = ILinker2V26aS7(backend=BACKEND)
    elif linker_class == "s8":
        from llm_sad_sam.linkers.experimental.ilinker2_v26a_s8 import ILinker2V26aS8
        linker = ILinker2V26aS8(backend=BACKEND)
    elif linker_class == "s9":
        from llm_sad_sam.linkers.experimental.ilinker2_v26a_s9 import ILinker2V26aS9
        linker = ILinker2V26aS9(backend=BACKEND)
    elif linker_class == "s10":
        from llm_sad_sam.linkers.experimental.ilinker2_v26a_s10 import ILinker2V26aS10
        linker = ILinker2V26aS10(backend=BACKEND)
    elif linker_class == "s11":
        from llm_sad_sam.linkers.experimental.ilinker2_v26a_s11 import ILinker2V26aS11
        linker = ILinker2V26aS11(backend=BACKEND)
    elif linker_class == "v30":
        from llm_sad_sam.linkers.experimental.ilinker2_v30 import ILinker2V30
        linker = ILinker2V30(backend=BACKEND)
    elif linker_class == "v30a":
        from llm_sad_sam.linkers.experimental.ilinker2_v30a import ILinker2V30a
        linker = ILinker2V30a(backend=BACKEND)
    elif linker_class == "v30b":
        from llm_sad_sam.linkers.experimental.ilinker2_v30b import ILinker2V30b
        linker = ILinker2V30b(backend=BACKEND)
    elif linker_class == "v30c":
        from llm_sad_sam.linkers.experimental.ilinker2_v30c import ILinker2V30c
        linker = ILinker2V30c(backend=BACKEND)
    elif linker_class == "v30d":
        from llm_sad_sam.linkers.experimental.ilinker2_v30d import ILinker2V30d
        linker = ILinker2V30d(
            backend=BACKEND,
            enable_v24=flags.get("v30d_v24", False),
            enable_uppercase=flags.get("v30d_uc", False),
            partial_min_count=flags.get("v30d_pt", 3),
            disable_generic_mention=flags.get("no_genm", False),
            disable_generic_coref=flags.get("no_gencf", False),
            disable_pronoun_coref=flags.get("no_pron", False),
            disable_boundary_filters=flags.get("no_bf", False),
            disable_partial_injection=flags.get("no_pi", False),
            disable_syn_safe=flags.get("no_synsafe", False),
            disable_parent_overlap=flags.get("no_po", False),
            disable_abbrev_guard=flags.get("no_ag", False),
        )
    elif linker_class == "v31":
        from llm_sad_sam.linkers.experimental.ilinker2_v31 import ILinker2V31
        linker = ILinker2V31(backend=BACKEND)
    elif linker_class == "v32":
        from llm_sad_sam.linkers.experimental.ilinker2_v32 import ILinker2V32
        linker = ILinker2V32(backend=BACKEND)
    elif linker_class == "alinker":
        from llm_sad_sam.linkers.experimental.alinker import ALinker
        linker = ALinker(backend=BACKEND)
    elif linker_class == "v35":
        from llm_sad_sam.linkers.experimental.ilinker2_v35 import ILinker2V35
        linker = ILinker2V35(backend=BACKEND)
    elif linker_class == "v35a":
        from llm_sad_sam.linkers.experimental.ilinker2_v35a import ILinker2V35a
        linker = ILinker2V35a(backend=BACKEND)
    elif linker_class == "v35b":
        from llm_sad_sam.linkers.experimental.ilinker2_v35b import ILinker2V35b
        linker = ILinker2V35b(backend=BACKEND)
    elif linker_class == "v35c":
        from llm_sad_sam.linkers.experimental.ilinker2_v35c import ILinker2V35c
        linker = ILinker2V35c(backend=BACKEND)
    elif linker_class == "v36a":
        from llm_sad_sam.linkers.experimental.ilinker2_v36a import ILinker2V36a
        linker = ILinker2V36a(backend=BACKEND)
    elif linker_class == "v36b":
        from llm_sad_sam.linkers.experimental.ilinker2_v36b import ILinker2V36b
        linker = ILinker2V36b(backend=BACKEND)
    elif linker_class == "v37":
        from llm_sad_sam.linkers.experimental.ilinker2_v37 import ILinker2V37
        linker = ILinker2V37(backend=BACKEND)
    elif linker_class == "v38":
        from llm_sad_sam.linkers.experimental.ilinker2_v38 import ILinker2V38
        linker = ILinker2V38(backend=BACKEND)
    elif linker_class == "v39":
        from llm_sad_sam.linkers.experimental.ilinker2_v39 import ILinker2V39
        linker = ILinker2V39(backend=BACKEND)
    elif linker_class == "v39a":
        from llm_sad_sam.linkers.experimental.ilinker2_v39a import ILinker2V39a
        linker = ILinker2V39a(backend=BACKEND)
    elif linker_class == "v40a":
        from llm_sad_sam.linkers.experimental.ilinker2_v40a import ILinker2V40a
        linker = ILinker2V40a(backend=BACKEND)
    elif linker_class == "v40b":
        from llm_sad_sam.linkers.experimental.ilinker2_v40b import ILinker2V40b
        linker = ILinker2V40b(backend=BACKEND)
    elif linker_class == "v40c":
        from llm_sad_sam.linkers.experimental.ilinker2_v40c import ILinker2V40c
        linker = ILinker2V40c(backend=BACKEND)
    elif linker_class == "s_linker":
        from llm_sad_sam.linkers.experimental.s_linker import SLinker
        linker = SLinker(backend=BACKEND)
    elif linker_class == "s_linker2":
        from llm_sad_sam.linkers.experimental.s_linker2 import SLinker2
        linker = SLinker2(backend=BACKEND)
    elif linker_class == "s_linker3":
        from llm_sad_sam.linkers.experimental.s_linker3 import SLinker3
        linker = SLinker3(backend=BACKEND)
    elif linker_class == "s_linker4":
        from llm_sad_sam.linkers.experimental.s_linker4 import SLinker4
        linker = SLinker4(backend=BACKEND)
    elif linker_class == "v33":
        from llm_sad_sam.linkers.experimental.ilinker2_v33 import ILinker2V33
        linker = ILinker2V33(backend=BACKEND)
    elif linker_class == "v34":
        from llm_sad_sam.linkers.experimental.ilinker2_v34 import ILinker2V34
        linker = ILinker2V34(backend=BACKEND)
    elif linker_class == "v33f":
        from llm_sad_sam.linkers.experimental.ilinker2_v33f import ILinker2V33f
        linker = ILinker2V33f(backend=BACKEND)
    elif linker_class == "v33g":
        from llm_sad_sam.linkers.experimental.ilinker2_v33g import ILinker2V33g
        linker = ILinker2V33g(backend=BACKEND)
    elif linker_class == "cnr":
        from llm_sad_sam.linkers.experimental.cnr_linker import CNRLinker
        linker = CNRLinker(backend=BACKEND)
    elif linker_class == "cnr_i2":
        from llm_sad_sam.linkers.experimental.cnr_i2_linker import CNRI2Linker
        linker = CNRI2Linker(backend=BACKEND)
    elif linker_class == "cnr_v26a":
        from llm_sad_sam.linkers.experimental.cnr_v26a_linker import CNRV26aLinker
        linker = CNRV26aLinker(backend=BACKEND)
    elif linker_class == "cnr_dk":
        from llm_sad_sam.linkers.experimental.cnr_linker import CNRLinker
        linker = CNRLinker(backend=BACKEND, enable_da=True)
    elif linker_class == "cnr_dk_v26a":
        from llm_sad_sam.linkers.experimental.cnr_v26a_linker import CNRV26aLinker
        linker = CNRV26aLinker(backend=BACKEND, enable_da=True)
    else:
        from llm_sad_sam.linkers.experimental.agent_linker_ablation import AgentLinkerAblation
        linker = AgentLinkerAblation(backend=BACKEND, **flags)
    link_kwargs = dict(
        text_path=str(paths["text"]),
        model_path=str(paths["model"]),
        transarc_csv=str(paths["transarc_sam"]),
    )
    # Support run_only (single-phase) mode
    run_only = flags.get("run_only")
    if run_only is not None and hasattr(linker, 'run_single_phase'):
        result = linker.run_single_phase(str(paths["text"]), str(paths["model"]), int(run_only))
        elapsed = time.time() - t0
        print(f"\n  Single-phase {run_only} completed in {elapsed:.0f}s")
        print(f"  Output keys: {list(result.keys())}")
        # For single-phase, return empty results (no F1 scoring)
        return {"P": 0, "R": 0, "F1": 0, "tp": 0, "fp": 0, "fn": 0,
                "time": elapsed, "sources": {}, "fp_by_source": {},
                "phase_output": result}

    # Support resume_from_phase from CLI arg or variant dict
    rfp = flags.get("resume_from") or resume_from_phase
    if rfp is not None and "resume_from_phase" in linker.link.__code__.co_varnames:
        link_kwargs["resume_from_phase"] = int(rfp)
    preds = linker.link(**link_kwargs)
    elapsed = time.time() - t0

    pred_pairs = {(l.sentence_number, l.component_id) for l in preds}
    pred_by_key = {(l.sentence_number, l.component_id): l for l in preds}
    m = eval_metrics(pred_pairs, gold_pairs)

    # Source breakdown
    source_counts = defaultdict(int)
    for l in preds:
        source_counts[l.source] += 1

    # FP analysis by source
    fp_pairs = pred_pairs - gold_pairs
    fp_by_source = defaultdict(int)
    fp_details = []
    for sn, cid in sorted(fp_pairs):
        link = pred_by_key.get((sn, cid))
        source = link.source if link else "???"
        fp_by_source[source] += 1
        cname = id_to_name.get(cid, cid[:20])
        sent = sent_map.get(sn)
        fp_details.append({
            "sentence": sn,
            "component": cname,
            "source": source,
            "confidence": link.confidence if link else 0,
            "text": sent.text[:120] if sent else "???",
        })

    # FN analysis
    fn_pairs = gold_pairs - pred_pairs
    fn_details = []
    for sn, cid in sorted(fn_pairs):
        cname = id_to_name.get(cid, cid[:20])
        sent = sent_map.get(sn)
        name_in_text = cname.lower() in sent.text.lower() if sent else False
        transarc_had = (sn, cid) in transarc_pairs
        fn_details.append({
            "sentence": sn,
            "component": cname,
            "name_in_text": name_in_text,
            "transarc_had": transarc_had,
        })

    # Save CSV
    out_dir = Path("results/ablation_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    export_links_csv(preds, str(out_dir / f"{variant_name}_{ds_name}_links.csv"))

    print(f"  {variant_name}: P={m['P']:.1%} R={m['R']:.1%} F1={m['F1']:.1%} "
          f"TP={m['tp']} FP={m['fp']} FN={m['fn']} ({elapsed:.0f}s)")
    print(f"    Sources: {dict(source_counts)}")
    print(f"    FP by source: {dict(fp_by_source)}")

    return {
        "variant": variant_name,
        "P": m["P"], "R": m["R"], "F1": m["F1"],
        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
        "n_links": len(preds),
        "time": elapsed,
        "sources": dict(source_counts),
        "fp_by_source": dict(fp_by_source),
        "fp_details": fp_details,
        "fn_details": fn_details,
    }


def print_comparison_table(all_results: dict, selected_variants: list[str]):
    """Print side-by-side comparison table."""
    print(f"\n{'='*160}")
    print("ABLATION STUDY: SIDE-BY-SIDE COMPARISON")
    print(f"{'='*160}")

    # Sub-header with variant names
    sub_header = f"  {'Dataset':<16}"
    for v in selected_variants:
        sub_header += f" | {v:^30}"
    print(sub_header)

    # Column labels
    header = f"  {'':<16}"
    for v in selected_variants:
        header += f" | {'P':>6} {'R':>6} {'F1':>6} {'FP':>4} {'FN':>4}"
    print(header)
    print(f"  {'-'*16}" + (" | " + "-"*30) * len(selected_variants))

    ds_names = list(all_results.keys())
    for ds_name in ds_names:
        row = f"  {ds_name:<16}"
        for v in selected_variants:
            res = all_results[ds_name].get(v)
            if res:
                row += f" | {res['P']:>5.1%} {res['R']:>5.1%} {res['F1']:>5.1%} {res['fp']:>4} {res['fn']:>4}"
            else:
                row += f" | {'--':>6} {'--':>6} {'--':>6} {'--':>4} {'--':>4}"
        print(row)

    # Macro averages
    print(f"  {'-'*16}" + (" | " + "-"*30) * len(selected_variants))
    row = f"  {'MACRO AVG':<16}"
    for v in selected_variants:
        vals = [all_results[ds].get(v) for ds in ds_names if v in all_results[ds]]
        if vals:
            avg_p = sum(x["P"] for x in vals) / len(vals)
            avg_r = sum(x["R"] for x in vals) / len(vals)
            avg_f1 = sum(x["F1"] for x in vals) / len(vals)
            total_fp = sum(x["fp"] for x in vals)
            total_fn = sum(x["fn"] for x in vals)
            row += f" | {avg_p:>5.1%} {avg_r:>5.1%} {avg_f1:>5.1%} {total_fp:>4} {total_fn:>4}"
        else:
            row += f" | {'--':>6} {'--':>6} {'--':>6} {'--':>4} {'--':>4}"
    print(row)


def print_delta_table(all_results: dict, selected_variants: list[str]):
    """Print F1 delta from baseline for each variant."""
    if "baseline" not in selected_variants:
        return

    print(f"\n{'='*120}")
    print("DELTA FROM BASELINE (F1 percentage points)")
    print(f"{'='*120}")

    non_baseline = [v for v in selected_variants if v != "baseline"]
    header = f"  {'Dataset':<16} {'baseline':>10}"
    for v in non_baseline:
        header += f" {v:>18}"
    print(header)
    print(f"  {'-'*16} {'-'*10}" + f" {'-'*18}" * len(non_baseline))

    ds_names = list(all_results.keys())
    for ds_name in ds_names:
        base_res = all_results[ds_name].get("baseline")
        if not base_res:
            continue
        row = f"  {ds_name:<16} {base_res['F1']:>9.1%}"
        for v in non_baseline:
            res = all_results[ds_name].get(v)
            if res:
                delta = (res["F1"] - base_res["F1"]) * 100
                sign = "+" if delta >= 0 else ""
                row += f" {res['F1']:>7.1%} ({sign}{delta:>+5.1f}pp)"
            else:
                row += f" {'--':>18}"
        print(row)

    # Macro average deltas
    print(f"  {'-'*16} {'-'*10}" + f" {'-'*18}" * len(non_baseline))
    base_vals = [all_results[ds].get("baseline") for ds in ds_names if "baseline" in all_results[ds]]
    if base_vals:
        base_avg = sum(x["F1"] for x in base_vals) / len(base_vals)
        row = f"  {'MACRO AVG':<16} {base_avg:>9.1%}"
        for v in non_baseline:
            vals = [all_results[ds].get(v) for ds in ds_names if v in all_results[ds]]
            if vals:
                avg_f1 = sum(x["F1"] for x in vals) / len(vals)
                delta = (avg_f1 - base_avg) * 100
                sign = "+" if delta >= 0 else ""
                row += f" {avg_f1:>7.1%} ({sign}{delta:>+5.1f}pp)"
            else:
                row += f" {'--':>18}"
        print(row)


def print_fp_source_comparison(all_results: dict, selected_variants: list[str]):
    """Print FP breakdown by source across variants."""
    print(f"\n{'='*140}")
    print("FP BY SOURCE COMPARISON")
    print(f"{'='*140}")

    all_sources = set()
    for ds_results in all_results.values():
        for v_result in ds_results.values():
            all_sources.update(v_result.get("fp_by_source", {}).keys())
    all_sources = sorted(all_sources)

    for ds_name, ds_results in all_results.items():
        print(f"\n  {ds_name}:")
        header = f"    {'Source':<16}"
        for v in selected_variants:
            header += f" {v:>16}"
        print(header)
        print(f"    {'-'*16}" + f" {'-'*16}" * len(selected_variants))

        for src in all_sources:
            row = f"    {src:<16}"
            for v in selected_variants:
                res = ds_results.get(v, {})
                count = res.get("fp_by_source", {}).get(src, 0)
                row += f" {count:>16}"
            print(row)

        row = f"    {'TOTAL':<16}"
        for v in selected_variants:
            res = ds_results.get(v, {})
            total = res.get("fp", 0)
            row += f" {total:>16}"
        print(row)


def main():
    selected_datasets = list(DATASETS.keys())
    selected_variants = list(VARIANTS.keys())
    resume_from_phase = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--datasets":
            selected_datasets = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                selected_datasets.append(args[i])
                i += 1
        elif args[i] == "--variants":
            selected_variants = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                selected_variants.append(args[i])
                i += 1
        elif args[i] == "--resume-from-phase":
            i += 1
            resume_from_phase = int(args[i])
            i += 1
        else:
            i += 1

    datasets = {k: v for k, v in DATASETS.items() if k in selected_datasets}

    print(f"{'='*160}")
    print("ABLATION STUDY: AgentLinker Improvements")
    print(f"Backend: {BACKEND.value}, Model: {os.environ.get('CLAUDE_MODEL', 'default')}")
    print(f"Datasets: {', '.join(datasets.keys())}")
    print(f"Variants: {', '.join(selected_variants)}")
    print()
    for v in selected_variants:
        flags = VARIANTS[v]
        print(f"  {v:>20}: {flags}")
    print(f"{'='*160}")

    all_results = {}

    for ds_name, paths in datasets.items():
        print(f"\n{'='*160}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*160}")

        components = parse_pcm_repository(str(paths["model"]))
        id_to_name = {c.id: c.name for c in components}
        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        sent_map = {s.number: s for s in sentences}
        gold_pairs = load_gold_sam(str(paths["gold_sam"]))
        transarc_pairs = load_transarc_pairs(str(paths["transarc_sam"]))

        print(f"  Components: {len(components)}, Sentences: {len(sentences)}")
        print(f"  Gold links: {len(gold_pairs)}, TransArc baseline: {len(transarc_pairs)}")

        ta_m = eval_metrics(transarc_pairs, gold_pairs)
        print(f"  TransArc baseline: P={ta_m['P']:.1%} R={ta_m['R']:.1%} F1={ta_m['F1']:.1%}")

        all_results[ds_name] = {}

        for variant_name in selected_variants:
            if variant_name not in VARIANTS:
                print(f"  WARNING: Unknown variant '{variant_name}', skipping")
                continue

            result = run_variant(
                variant_name, VARIANTS[variant_name],
                ds_name, paths, gold_pairs,
                transarc_pairs, id_to_name, sent_map,
                resume_from_phase=resume_from_phase,
            )
            all_results[ds_name][variant_name] = result

    # ======= SUMMARY TABLES =======
    print_comparison_table(all_results, selected_variants)
    print_delta_table(all_results, selected_variants)
    print_fp_source_comparison(all_results, selected_variants)

    # ======= TIMING =======
    print(f"\n{'='*120}")
    print("TIMING COMPARISON")
    print(f"{'='*120}")
    header = f"  {'Dataset':<16}"
    for v in selected_variants:
        header += f" {v:>16}"
    print(header)
    print(f"  {'-'*16}" + f" {'-'*16}" * len(selected_variants))
    for ds_name in all_results:
        row = f"  {ds_name:<16}"
        for v in selected_variants:
            res = all_results[ds_name].get(v, {})
            t = res.get("time", 0)
            row += f" {t:>15.0f}s"
        print(row)

    # Save JSON
    results_dir = Path("results/ablation_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"ablation_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
