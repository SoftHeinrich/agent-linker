#!/usr/bin/env python3
"""Linker Comparison: Run V44, V45, V46, AgentLinker on all datasets.

Runs all 4 linkers on all 5 benchmark datasets, evaluates against gold standards,
and prints side-by-side comparison with FP/FN analysis per approach.

Usage:
    python run_comparison.py
    python run_comparison.py --datasets mediastore teastore
    python run_comparison.py --linkers v44 agent
    python run_comparison.py --datasets teammates --linkers v44 agent
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

# Lazy imports for linkers (only import when needed)
LINKER_REGISTRY = {
    "v44": {
        "module": "llm_sad_sam.linkers.experimental.transarc_refined_linker_v44",
        "class": "TransArcRefinedLinkerV44",
        "export": "export_links_csv",
        "label": "V44 (multi-comp + word-as-concept)",
    },
    "v45": {
        "module": "llm_sad_sam.linkers.experimental.transarc_refined_linker_v45",
        "class": "TransArcRefinedLinkerV45",
        "export": "export_links_csv",
        "label": "V45 (discourse-aware coref)",
    },
    "v46": {
        "module": "llm_sad_sam.linkers.experimental.transarc_refined_linker_v46",
        "class": "TransArcRefinedLinkerV46",
        "export": "export_links_csv",
        "label": "V46 (learned linguistic patterns)",
    },
    "agent": {
        "module": "llm_sad_sam.linkers.experimental.agent_linker",
        "class": "AgentLinker",
        "export": "export_links_csv",
        "label": "AgentLinker (conservative implicit + source-aware judge)",
    },
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
    """Load SAD-SAM gold standard as (sentence_number, component_id) pairs."""
    links = set()
    with open(gold_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def load_transarc_pairs(transarc_path: str) -> set[tuple[int, str]]:
    """Load TransArc baseline as (sentence_number, component_id) pairs."""
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


def get_linker_instance(linker_key: str):
    """Dynamically import and instantiate a linker."""
    import importlib
    info = LINKER_REGISTRY[linker_key]
    mod = importlib.import_module(info["module"])
    cls = getattr(mod, info["class"])
    export_fn = getattr(mod, info["export"])
    return cls(backend=BACKEND), export_fn


def run_linker(linker_key: str, ds_name: str, paths: dict, gold_pairs: set,
               transarc_pairs: set, id_to_name: dict, sent_map: dict):
    """Run a single linker on a single dataset and return results."""
    info = LINKER_REGISTRY[linker_key]
    print(f"\n  --- Running {info['label']} ---")

    t0 = time.time()
    linker, export_fn = get_linker_instance(linker_key)
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
    out_dir = Path("results/comparison_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    export_fn(preds, str(out_dir / f"{linker_key}_{ds_name}_links.csv"))

    print(f"  {info['label']}: P={m['P']:.1%} R={m['R']:.1%} F1={m['F1']:.1%} "
          f"TP={m['tp']} FP={m['fp']} FN={m['fn']} ({elapsed:.0f}s)")
    print(f"    Sources: {dict(source_counts)}")
    print(f"    FP by source: {dict(fp_by_source)}")

    return {
        "linker": linker_key,
        "P": m["P"], "R": m["R"], "F1": m["F1"],
        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
        "n_links": len(preds),
        "time": elapsed,
        "sources": dict(source_counts),
        "fp_by_source": dict(fp_by_source),
        "fp_details": fp_details,
        "fn_details": fn_details,
    }


def print_comparison_table(all_results: dict, selected_linkers: list[str]):
    """Print side-by-side comparison table."""
    print(f"\n{'='*140}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'='*140}")

    # Header
    header = f"  {'Dataset':<16}"
    for lk in selected_linkers:
        label = lk.upper()
        header += f" | {'P':>6} {'R':>6} {'F1':>6} {'FP':>4} {'FN':>4}"
    print(header)

    sub_header = f"  {'':<16}"
    for lk in selected_linkers:
        sub_header += f" | {lk:^30}"
    print(sub_header)
    print(f"  {'-'*16}" + (" | " + "-"*30) * len(selected_linkers))

    # Per-dataset rows
    ds_names = list(all_results.keys())
    for ds_name in ds_names:
        row = f"  {ds_name:<16}"
        for lk in selected_linkers:
            res = all_results[ds_name].get(lk)
            if res:
                row += f" | {res['P']:>5.1%} {res['R']:>5.1%} {res['F1']:>5.1%} {res['fp']:>4} {res['fn']:>4}"
            else:
                row += f" | {'--':>6} {'--':>6} {'--':>6} {'--':>4} {'--':>4}"
        print(row)

    # Macro averages
    print(f"  {'-'*16}" + (" | " + "-"*30) * len(selected_linkers))
    row = f"  {'MACRO AVG':<16}"
    for lk in selected_linkers:
        vals = [all_results[ds].get(lk) for ds in ds_names if lk in all_results[ds]]
        if vals:
            avg_p = sum(v["P"] for v in vals) / len(vals)
            avg_r = sum(v["R"] for v in vals) / len(vals)
            avg_f1 = sum(v["F1"] for v in vals) / len(vals)
            total_fp = sum(v["fp"] for v in vals)
            total_fn = sum(v["fn"] for v in vals)
            row += f" | {avg_p:>5.1%} {avg_r:>5.1%} {avg_f1:>5.1%} {total_fp:>4} {total_fn:>4}"
        else:
            row += f" | {'--':>6} {'--':>6} {'--':>6} {'--':>4} {'--':>4}"
    print(row)


def print_fp_source_comparison(all_results: dict, selected_linkers: list[str]):
    """Print FP breakdown by source across linkers."""
    print(f"\n{'='*140}")
    print("FP BY SOURCE COMPARISON")
    print(f"{'='*140}")

    # Collect all sources
    all_sources = set()
    for ds_results in all_results.values():
        for lk_result in ds_results.values():
            all_sources.update(lk_result.get("fp_by_source", {}).keys())
    all_sources = sorted(all_sources)

    for ds_name, ds_results in all_results.items():
        print(f"\n  {ds_name}:")
        header = f"    {'Source':<16}"
        for lk in selected_linkers:
            header += f" {lk:>10}"
        print(header)
        print(f"    {'-'*16}" + f" {'-'*10}" * len(selected_linkers))

        for src in all_sources:
            row = f"    {src:<16}"
            for lk in selected_linkers:
                res = ds_results.get(lk, {})
                count = res.get("fp_by_source", {}).get(src, 0)
                row += f" {count:>10}"
            print(row)

        # Total row
        row = f"    {'TOTAL':<16}"
        for lk in selected_linkers:
            res = ds_results.get(lk, {})
            total = res.get("fp", 0)
            row += f" {total:>10}"
        print(row)


def print_detailed_fp_diff(all_results: dict, selected_linkers: list[str], id_to_name_map: dict, sent_maps: dict):
    """Print FPs unique to each linker per dataset."""
    print(f"\n{'='*140}")
    print("UNIQUE FPs PER LINKER (FPs that only this linker produces)")
    print(f"{'='*140}")

    for ds_name, ds_results in all_results.items():
        # Build FP sets per linker
        fp_sets = {}
        for lk in selected_linkers:
            res = ds_results.get(lk, {})
            fp_sets[lk] = {(fp["sentence"], fp["component"]) for fp in res.get("fp_details", [])}

        print(f"\n  {ds_name}:")
        for lk in selected_linkers:
            # FPs unique to this linker
            other_fps = set()
            for other_lk in selected_linkers:
                if other_lk != lk:
                    other_fps |= fp_sets.get(other_lk, set())
            unique = fp_sets.get(lk, set()) - other_fps
            if unique:
                print(f"    {lk} unique FPs ({len(unique)}):")
                # Find details
                for fp in ds_results.get(lk, {}).get("fp_details", []):
                    if (fp["sentence"], fp["component"]) in unique:
                        print(f"      S{fp['sentence']} -> {fp['component']} (src={fp['source']}, conf={fp['confidence']:.2f})")
                        print(f"        \"{fp['text']}\"")

        # FPs shared by ALL linkers
        if len(selected_linkers) >= 2:
            shared = fp_sets.get(selected_linkers[0], set())
            for lk in selected_linkers[1:]:
                shared &= fp_sets.get(lk, set())
            if shared:
                print(f"    SHARED by all ({len(shared)}):")
                for sn, cname in sorted(shared):
                    # Get text from any linker's details
                    for lk in selected_linkers:
                        for fp in ds_results.get(lk, {}).get("fp_details", []):
                            if fp["sentence"] == sn and fp["component"] == cname:
                                print(f"      S{sn} -> {cname}: \"{fp['text'][:80]}\"")
                                break
                        else:
                            continue
                        break


def main():
    # Parse args
    selected_datasets = list(DATASETS.keys())
    selected_linkers = list(LINKER_REGISTRY.keys())

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--datasets":
            selected_datasets = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                selected_datasets.append(args[i])
                i += 1
        elif args[i] == "--linkers":
            selected_linkers = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                selected_linkers.append(args[i])
                i += 1
        else:
            i += 1

    datasets = {k: v for k, v in DATASETS.items() if k in selected_datasets}

    print(f"{'='*140}")
    print("LINKER COMPARISON: V44 vs V45 vs V46 vs AgentLinker")
    print(f"Backend: {BACKEND.value}, Model: {os.environ.get('CLAUDE_MODEL', 'default')}")
    print(f"Datasets: {', '.join(datasets.keys())}")
    print(f"Linkers: {', '.join(selected_linkers)}")
    print(f"{'='*140}")

    # all_results[dataset][linker] = metrics dict
    all_results = {}
    id_to_name_map = {}
    sent_maps = {}

    for ds_name, paths in datasets.items():
        print(f"\n{'='*140}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*140}")

        # Load shared data
        components = parse_pcm_repository(str(paths["model"]))
        id_to_name = {c.id: c.name for c in components}
        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        sent_map = {s.number: s for s in sentences}
        gold_pairs = load_gold_sam(str(paths["gold_sam"]))
        transarc_pairs = load_transarc_pairs(str(paths["transarc_sam"]))

        id_to_name_map[ds_name] = id_to_name
        sent_maps[ds_name] = sent_map

        print(f"  Components: {len(components)}, Sentences: {len(sentences)}")
        print(f"  Gold links: {len(gold_pairs)}, TransArc baseline: {len(transarc_pairs)}")

        ta_m = eval_metrics(transarc_pairs, gold_pairs)
        print(f"  TransArc baseline: P={ta_m['P']:.1%} R={ta_m['R']:.1%} F1={ta_m['F1']:.1%}")

        all_results[ds_name] = {}

        for linker_key in selected_linkers:
            if linker_key not in LINKER_REGISTRY:
                print(f"  WARNING: Unknown linker '{linker_key}', skipping")
                continue

            result = run_linker(
                linker_key, ds_name, paths, gold_pairs,
                transarc_pairs, id_to_name, sent_map
            )
            all_results[ds_name][linker_key] = result

    # ======= SUMMARY TABLES =======
    print_comparison_table(all_results, selected_linkers)
    print_fp_source_comparison(all_results, selected_linkers)
    print_detailed_fp_diff(all_results, selected_linkers, id_to_name_map, sent_maps)

    # ======= TIME COMPARISON =======
    print(f"\n{'='*140}")
    print("TIMING COMPARISON")
    print(f"{'='*140}")
    header = f"  {'Dataset':<16}"
    for lk in selected_linkers:
        header += f" {lk:>10}"
    print(header)
    print(f"  {'-'*16}" + f" {'-'*10}" * len(selected_linkers))
    for ds_name in all_results:
        row = f"  {ds_name:<16}"
        for lk in selected_linkers:
            res = all_results[ds_name].get(lk, {})
            t = res.get("time", 0)
            row += f" {t:>9.0f}s"
        print(row)

    # Save JSON
    results_dir = Path("results/comparison_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"comparison_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
