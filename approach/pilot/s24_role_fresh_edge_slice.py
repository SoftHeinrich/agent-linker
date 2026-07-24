#!/usr/bin/env python3
"""Fresh GPT-5.6-terra checkpoint on prior FN/FP sentence slices."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from run_ablation import DATASETS, eval_metrics, load_gold_sam

from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker24_role_orchestrator import (
    SLinker24RoleOrchestrator,
)


START = (
    PACKAGE_ROOT
    / "results/s24_replacement_orchestrator_pilot_all_iter5_participation_20260724"
    / "pilot_results.json"
)


def aggregate(rows, key):
    values = [row[key] for row in rows]
    tp = sum(value["tp"] for value in values)
    fp = sum(value["fp"] for value in values)
    fn = sum(value["fn"] for value in values)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "macro_f1": sum(value["F1"] for value in values) / len(values),
        "macro_f2": sum(value["F2"] for value in values) / len(values),
        "pooled_f1": 2 * tp / (2 * tp + fp + fn),
        "pooled_f2": 5 * tp / (5 * tp + 4 * fn + fp),
    }


def run_project(name, prior):
    paths = DATASETS[name]
    components = parse_pcm_repository(paths["model"])
    name_to_id = {component.name: component.id for component in components}
    full_sentences = load_sentences(paths["text"])
    prior_pairs = {
        (item["sentence"], name_to_id[item["component"]])
        for item in prior["links"]
    }
    gold = load_gold_sam(str(paths["gold_sam"]))

    positive_pairs = gold - prior_pairs
    negative_pairs = prior_pairs - gold
    focus_numbers = sorted({
        sentence
        for sentence, _ in positive_pairs | negative_pairs
    })
    focus_sentences = [
        sentence
        for sentence in full_sentences
        if sentence.number in focus_numbers
    ]
    focus_map = build_sent_map(focus_sentences)
    focus_gold = {
        pair for pair in gold if pair[0] in focus_numbers
    }

    linker = SLinker24RoleOrchestrator(backend=LLMBackend.OPENAI)
    linker._current_text_path = str(paths["text"])
    knowledge = linker._run_parallel(
        {
            "model": lambda: linker._analyze_model(components),
            "doc": lambda: linker._learn_document_knowledge(
                full_sentences, components
            ),
        }
    )
    linker.model_knowledge = knowledge["model"]
    linker.doc_knowledge = knowledge["doc"]

    entity, entity_feedback = linker._run_entity_tool(
        focus_sentences, components, name_to_id, focus_map
    )
    profile = linker._project_profile(focus_sentences, components)
    history = [
        {
            "action": "entity_pipeline",
            "reason": "fresh edge identity pass",
            "feedback": linker._controller_feedback(entity_feedback),
        }
    ]
    remaining = [
        tool
        for tool in linker._available_tools(profile)
        if tool == "relation_role_resolution"
    ]
    action, decision = linker._choose_tool(
        profile, remaining, history, entity, focus_map
    )
    role_feedback = {}
    if action == "relation_role_resolution":
        additions, role_feedback = linker._run_relation_role_tool(
            focus_sentences, components, entity, focus_map
        )
        final = linker._union(entity, additions)
    else:
        final = entity

    entity_pairs = {
        (link.sentence_number, link.component_id) for link in entity
    }
    final_pairs = {
        (link.sentence_number, link.component_id) for link in final
    }
    return {
        "dataset": name,
        "focus_sentences": focus_numbers,
        "prior_positive_edges": len(positive_pairs),
        "prior_negative_edges": len(negative_pairs),
        "action": action,
        "controller": decision,
        "entity": eval_metrics(entity_pairs, focus_gold),
        "final": eval_metrics(final_pairs, focus_gold),
        "marginal": {
            "tp": len((final_pairs - entity_pairs) & focus_gold),
            "fp": len((final_pairs - entity_pairs) - focus_gold),
        },
        "entity_feedback": entity_feedback,
        "role_feedback": role_feedback,
        "llm_calls": len(linker._llm_calls),
        "trace": linker._llm_calls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=(
            PACKAGE_ROOT
            / "results/s24_role_fresh_edge_slice_gpt56terra_20260724"
        ),
    )
    args = parser.parse_args()
    start = json.loads(START.read_text())
    by_name = {row["dataset"]: row for row in start["datasets"]}
    rows = [run_project(name, by_name[name]) for name in args.datasets]
    entity_metrics = aggregate(rows, "entity")
    final_metrics = aggregate(rows, "final")
    actions = {row["action"] for row in rows}
    passed = (
        "relation_role_resolution" in actions
        and "finalize" in actions
        and final_metrics["macro_f2"] > entity_metrics["macro_f2"]
        and final_metrics["pooled_f2"] > entity_metrics["pooled_f2"]
    )
    result = {
        "protocol": (
            "gold selects prior FN/FP sentences offline; all model/profile/"
            "entity/controller/role/validator responses fresh OpenAI calls"
        ),
        "model": "gpt-5.6-terra",
        "reasoning_effort": "none",
        "datasets": rows,
        "aggregate": {
            "entity": entity_metrics,
            "final": final_metrics,
            "actions": sorted(actions),
        },
        "pass_gate": passed,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output = args.results_dir / "pilot_results.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["aggregate"], indent=2))
    print(f"PASS={passed}")
    print(f"Results: {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
