#!/usr/bin/env python3
"""Fresh same-candidate edge pilot for S24 entity ownership."""
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

from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker24_role_orchestrator import (
    SLinker24RoleOrchestrator,
)


START = (
    PACKAGE_ROOT
    / "results/s24_role_e2e_v3_noaudit_gpt56terra_20260724"
    / "ablation_20260724_123747.json"
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
    gold = load_gold_sam(str(paths["gold_sam"]))

    prior_s24 = prior["s_linker24_role_orchestrator"]
    focus_numbers = sorted({
        item["sentence"]
        for item in prior_s24["fp_details"] + prior_s24["fn_details"]
    })
    focus_sentences = [
        sentence
        for sentence in full_sentences
        if sentence.number in focus_numbers
    ]
    full_map = build_sent_map(full_sentences)
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

    raw_by_key = linker._run_framing_c(
        full_sentences, components, name_to_id, full_map
    )
    raw_candidates = list(raw_by_key.values())
    bundles = {
        (candidate.sentence_number, candidate.component_id):
            linker._build_evidence_bundle(candidate, full_map)
        for candidate in raw_candidates
    }
    raw_approved, raw_decisions = linker._validate_with_evidence(
        raw_candidates,
        bundles,
        components,
        full_map,
        p1_tag="phase_11_entity_p1",
        p2_tag="phase_11_entity_p2",
        stage_label="entity_unfiltered",
    )
    owned = set(map(id, linker._select_entity_candidates(
        raw_approved, full_map
    )))
    unfiltered = [
        SadSamLink(
            candidate.sentence_number,
            candidate.component_id,
            candidate.component_name,
            source="entity",
        )
        for candidate in raw_approved
    ]
    filtered = [
        SadSamLink(
            candidate.sentence_number,
            candidate.component_id,
            candidate.component_name,
            source="entity",
        )
        for candidate in raw_approved
        if id(candidate) in owned
    ]

    profile = linker._project_profile(full_sentences, components)
    history = [{
        "action": "entity_pipeline",
        "reason": "fresh same-candidate ownership pass",
        "feedback": {
            "accepted": linker._controller_link_view(filtered),
            "rejected": [],
        },
    }]
    remaining = [
        tool
        for tool in linker._available_tools(profile)
        if tool == "relation_role_resolution"
    ]
    action, decision = linker._choose_tool(
        profile, remaining, history, filtered, full_map
    )
    role_feedback = {}
    if action == "relation_role_resolution":
        additions, role_feedback = linker._run_relation_role_tool(
            full_sentences, components, filtered, full_map
        )
        final = linker._union(filtered, additions)
    else:
        final = filtered

    def score(links):
        pairs = {
            (link.sentence_number, link.component_id)
            for link in links
            if link.sentence_number in focus_numbers
        }
        return eval_metrics(pairs, focus_gold)

    return {
        "dataset": name,
        "focus_sentences": focus_numbers,
        "action": action,
        "controller": decision,
        "unfiltered": score(unfiltered),
        "owned_entity": score(filtered),
        "final": score(final),
        "removed_by_ownership": len(unfiltered) - len(filtered),
        "raw_validator_decisions": linker._decision_view(raw_decisions),
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
            / "results/s24_f1_ownership_edge_gpt56terra_20260724"
        ),
    )
    args = parser.parse_args()
    start = json.loads(START.read_text())
    rows = [run_project(name, start[name]) for name in args.datasets]
    unfiltered = aggregate(rows, "unfiltered")
    owned_entity = aggregate(rows, "owned_entity")
    final = aggregate(rows, "final")
    actions = {row["action"] for row in rows}
    passed = (
        "relation_role_resolution" in actions
        and "finalize" in actions
        and final["macro_f1"] >= unfiltered["macro_f1"]
        and final["pooled_f1"] >= unfiltered["pooled_f1"]
        and final["macro_f2"] > unfiltered["macro_f2"]
        and final["pooled_f2"] > unfiltered["pooled_f2"]
    )
    result = {
        "protocol": (
            "gold predeclares prior S24 FP/FN sentences for scoring only; "
            "the intact document drives all fresh OpenAI profile, candidate, "
            "validation, and controller calls; unfiltered and owned states "
            "share identical accepted candidates"
        ),
        "model": "gpt-5.6-terra",
        "reasoning_effort": "none",
        "datasets": rows,
        "aggregate": {
            "unfiltered": unfiltered,
            "owned_entity": owned_entity,
            "final": final,
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
