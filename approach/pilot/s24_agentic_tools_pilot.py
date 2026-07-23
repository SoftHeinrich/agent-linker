#!/usr/bin/env python3
"""Fixed-floor pilot for bounded, project-adaptive S24 tool selection.

The pilot deliberately does not rerun S21.  It loads S24's saved S21 floor and
Phase-1 knowledge, asks a controller which bounded recovery tools are worth
calling, executes only those tools, and scores additions independently.

No gold data is included in controller or tool prompts.  Gold is loaded only
after execution for evaluation.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from run_ablation import DATASETS, eval_metrics, export_links_csv, load_gold_sam

from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.linkers.experimental.s_linker24_agentic import SLinker24Agentic


PACKAGE_ROOT = ROOT.parent
DEFAULT_CACHE = PACKAGE_ROOT / "results/phase_cache/s_linker24/openai"
DEFAULT_RESULTS = PACKAGE_ROOT / "results/s24_agentic_tools_pilot_20260724"


def _load_checkpoint(dataset: str, name: str, cache_root: Path) -> dict:
    path = cache_root / dataset / f"{name}.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)  # trusted local checkpoint produced by S21


def run_dataset(dataset: str, cache_root: Path, results_dir: Path) -> dict:
    paths = DATASETS[dataset]
    layer1 = _load_checkpoint(dataset, "layer1", cache_root)
    final_checkpoint = _load_checkpoint(dataset, "final", cache_root)
    floor = list(final_checkpoint["final"])

    linker = SLinker24Agentic(backend=LLMBackend.CODEX)
    linker.model_knowledge = layer1["model_knowledge"]
    linker.doc_knowledge = layer1["doc_knowledge"]
    linker._phase_log = []
    linker._current_text_path = str(paths["text"])
    linker.agentic_tool_calls = []
    linker.agentic_plan_reason = ""
    linker._s24_stats = {"eligible": 0, "resolver_approved": 0, "kept": 0}
    final = linker._augment_floor(
        str(paths["text"]), str(paths["model"]), floor
    )
    floor_keys = {(link.sentence_number, link.component_id) for link in floor}
    unique_additions: dict[tuple[int, str], SadSamLink] = {}
    for link in final:
        key = (link.sentence_number, link.component_id)
        if key not in floor_keys:
            unique_additions.setdefault(key, link)

    gold = load_gold_sam(str(paths["gold_sam"]))
    floor_pairs = {(link.sentence_number, link.component_id) for link in floor}
    addition_pairs = set(unique_additions)
    final_pairs = floor_pairs | addition_pairs
    floor_metrics = eval_metrics(floor_pairs, gold)
    final_metrics = eval_metrics(final_pairs, gold)
    marginal_tp = len(addition_pairs & gold)
    marginal_fp = len(addition_pairs - gold)
    export_links_csv(final, results_dir / f"s24_agentic_pilot_{dataset}_links.csv")
    return {
        "dataset": dataset,
        "inventory": linker.agentic_inventory,
        "selected_tools": linker.agentic_tool_calls,
        "floor": floor_metrics,
        "final": final_metrics,
        "marginal": {
            "tp": marginal_tp,
            "fp": marginal_fp,
            "precision": marginal_tp / (marginal_tp + marginal_fp)
            if marginal_tp + marginal_fp
            else None,
            "links": [
                {
                    "sentence": sentence,
                    "component_id": component,
                    "gold": (sentence, component) in gold,
                    "source": unique_additions[(sentence, component)].source,
                }
                for sentence, component in sorted(addition_pairs)
            ],
        },
        "llm_calls": len(linker._llm_calls),
        "trace": linker._llm_calls,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    results = [
        run_dataset(dataset, args.cache_root, args.results_dir)
        for dataset in args.datasets
    ]
    marginal_tp = sum(item["marginal"]["tp"] for item in results)
    marginal_fp = sum(item["marginal"]["fp"] for item in results)
    macro_floor = sum(item["floor"]["F1"] for item in results) / len(results)
    macro_final = sum(item["final"]["F1"] for item in results) / len(results)
    distinct_plans = {tuple(item["selected_tools"]) for item in results}
    marginal_precision = (
        marginal_tp / (marginal_tp + marginal_fp)
        if marginal_tp + marginal_fp
        else 0.0
    )
    passed = (
        marginal_tp >= 1
        and marginal_fp <= 1
        and marginal_precision >= 0.95
        and macro_final > macro_floor
        and len(distinct_plans) >= 2
    )
    summary = {
        "protocol": "fixed S21 floor; Codex controller and recovery tools; no gold in prompts",
        "datasets": results,
        "aggregate": {
            "marginal_tp": marginal_tp,
            "marginal_fp": marginal_fp,
            "marginal_precision": marginal_precision,
            "macro_floor_f1": macro_floor,
            "macro_final_f1": macro_final,
            "macro_delta": macro_final - macro_floor,
            "distinct_plans": len(distinct_plans),
        },
        "pass_gate": {
            "at_least_one_tp": marginal_tp >= 1,
            "at_most_one_fp": marginal_fp <= 1,
            "marginal_precision_at_least_0_95": marginal_precision >= 0.95,
            "macro_f1_improves": macro_final > macro_floor,
            "project_adaptive_routes": len(distinct_plans) >= 2,
            "passed": passed,
        },
    }
    output = args.results_dir / "pilot_results.json"
    output.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"PASS={passed}")
    print(f"Results: {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
