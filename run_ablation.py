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

BACKEND = LLMBackend.CLAUDE
os.environ["CLAUDE_MODEL"] = "sonnet"


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
                gold_pairs: set, transarc_pairs: set, id_to_name: dict, sent_map: dict):
    """Run a single ablation variant on a single dataset."""
    from llm_sad_sam.linkers.experimental.agent_linker_ablation import AgentLinkerAblation
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
    elif linker_class == "v6":
        from llm_sad_sam.linkers.experimental.agent_linker_v6 import AgentLinkerV6
        pf = flags.pop("post_filter", "none")
        linker = AgentLinkerV6(backend=BACKEND, post_filter=pf)
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
    else:
        linker = AgentLinkerAblation(backend=BACKEND, **flags)
    preds = linker.link(
        text_path=str(paths["text"]),
        model_path=str(paths["model"]),
        transarc_csv=str(paths["transarc_sam"]),
    )
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
