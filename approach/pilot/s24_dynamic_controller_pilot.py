#!/usr/bin/env python3
"""Fixed-floor pilot for S24's sequential project-profile controller.

Gold is loaded only after the controller and bounded phases have completed.
The run-all comparison is the number of every non-empty phase inventory; it
tests whether controller quality improves over S21 while avoiding phase calls.
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
from llm_sad_sam.linkers.experimental.s_linker24_dynamic import SLinker24Dynamic


PACKAGE_ROOT = ROOT.parent
DEFAULT_CACHE = PACKAGE_ROOT / "results/phase_cache/s_linker24/openai"
DEFAULT_RESULTS = PACKAGE_ROOT / "results/s24_dynamic_controller_pilot_20260724"


def _checkpoint(dataset: str, name: str, root: Path) -> dict:
    with (root / dataset / f"{name}.pkl").open("rb") as handle:
        return pickle.load(handle)  # trusted local S21 checkpoint


def run_dataset(dataset: str, cache_root: Path, results_dir: Path) -> dict:
    paths = DATASETS[dataset]
    layer1 = _checkpoint(dataset, "layer1", cache_root)
    floor = list(_checkpoint(dataset, "final", cache_root)["final"])
    linker = SLinker24Dynamic(backend=LLMBackend.CODEX)
    linker.model_knowledge = layer1["model_knowledge"]
    linker.doc_knowledge = layer1["doc_knowledge"]
    linker._phase_log = []
    linker._current_text_path = str(paths["text"])
    linker._s24_stats = {"eligible": 0, "resolver_approved": 0, "kept": 0}
    final = linker._augment_floor(
        str(paths["text"]), str(paths["model"]), floor
    )

    floor_pairs = {(link.sentence_number, link.component_id) for link in floor}
    unique: dict[tuple[int, str], SadSamLink] = {}
    for link in final:
        key = (link.sentence_number, link.component_id)
        if key not in floor_pairs:
            unique.setdefault(key, link)
    additions = set(unique)

    # Evaluation begins here. Nothing above has access to the gold set.
    gold = load_gold_sam(str(paths["gold_sam"]))
    floor_metrics = eval_metrics(floor_pairs, gold)
    final_metrics = eval_metrics(floor_pairs | additions, gold)
    tp = len(additions & gold)
    fp = len(additions - gold)
    export_links_csv(final, results_dir / f"s24_dynamic_{dataset}_links.csv")
    return {
        "dataset": dataset,
        "profile": linker.dynamic_profile,
        "inventory": linker.agentic_inventory,
        "workflow": linker.dynamic_steps,
        "selected_tools": linker.agentic_tool_calls,
        "run_all_tool_count": sum(
            count > 0 for count in linker.agentic_inventory.values()
        ),
        "floor": floor_metrics,
        "final": final_metrics,
        "marginal": {
            "tp": tp,
            "fp": fp,
            "precision": tp / (tp + fp) if tp + fp else None,
            "links": [
                {
                    "sentence": sentence,
                    "component_id": component,
                    "gold": (sentence, component) in gold,
                    "source": unique[(sentence, component)].source,
                }
                for sentence, component in sorted(additions)
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
    datasets = [
        run_dataset(dataset, args.cache_root, args.results_dir)
        for dataset in args.datasets
    ]
    tp = sum(item["marginal"]["tp"] for item in datasets)
    fp = sum(item["marginal"]["fp"] for item in datasets)
    precision = tp / (tp + fp) if tp + fp else 0.0
    floor_f1 = sum(item["floor"]["F1"] for item in datasets) / len(datasets)
    final_f1 = sum(item["final"]["F1"] for item in datasets) / len(datasets)
    dynamic_tools = sum(len(item["selected_tools"]) for item in datasets)
    run_all_tools = sum(item["run_all_tool_count"] for item in datasets)
    workflows = {tuple(item["selected_tools"]) for item in datasets}
    profile_reasons = all(
        step.get("reason", "").strip()
        for item in datasets
        for step in item["workflow"]
        if step["action"] != "stop"
    )
    passed = (
        tp >= 1
        and fp <= 1
        and precision >= 0.95
        and final_f1 > floor_f1
        and len(workflows) >= 3
        and dynamic_tools < run_all_tools
        and profile_reasons
    )
    summary = {
        "protocol": (
            "fixed saved S21 floors; sequential Codex controller sees only "
            "runtime profiles and tool observations; gold loaded after execution"
        ),
        "datasets": datasets,
        "aggregate": {
            "marginal_tp": tp,
            "marginal_fp": fp,
            "marginal_precision": precision,
            "macro_floor_f1": floor_f1,
            "macro_final_f1": final_f1,
            "macro_delta": final_f1 - floor_f1,
            "distinct_workflows": len(workflows),
            "dynamic_tool_calls": dynamic_tools,
            "run_all_tool_calls": run_all_tools,
            "tool_calls_saved": run_all_tools - dynamic_tools,
        },
        "pass_gate": {
            "macro_f1_improves": final_f1 > floor_f1,
            "marginal_precision_at_least_0_95": precision >= 0.95,
            "at_most_one_fp": fp <= 1,
            "at_least_three_workflows": len(workflows) >= 3,
            "fewer_tools_than_run_all": dynamic_tools < run_all_tools,
            "decisions_cite_runtime_profile": profile_reasons,
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
