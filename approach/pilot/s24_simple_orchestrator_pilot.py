#!/usr/bin/env python3
"""Replay and benchmark the active minimal S24 orchestration surface."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from run_ablation import DATASETS, eval_metrics, export_links_csv, load_gold_sam

from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.core.document_loader_v2 import load_sentences
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker24_role_orchestrator import (
    SLinker24RoleOrchestrator,
)

SLinker24SimplePilot = SLinker24RoleOrchestrator


def load_links(path, source_filter=None):
    links = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if source_filter and row["source"] != source_filter:
                continue
            links.append(
                SadSamLink(
                    int(row["sentence"]),
                    row["component_id"],
                    row["component_name"],
                    source=row["source"],
                )
            )
    return links


def score(links, gold):
    return eval_metrics(
        {
            (link.sentence_number, link.component_id)
            for link in links
        },
        gold,
    )


def prompt_metrics(linker):
    selected = [
        item
        for item in linker._llm_calls
        if "simple_controller" in item.get("phase", "")
        or "simple_participant" in item.get("phase", "")
    ]
    return {
        "calls": len(selected),
        "prompt_characters": sum(
            len(item.get("prompt", "")) for item in selected
        ),
        "prompt_tokens": sum(
            (item.get("token_usage") or {}).get(
                "prompt_tokens", 0
            )
            for item in selected
        ),
    }


def replay_project(name, baseline_dir, results_dir):
    paths = DATASETS[name]
    components = parse_pcm_repository(paths["model"])
    sentences = load_sentences(paths["text"])
    gold = load_gold_sam(str(paths["gold_sam"]))
    baseline_csv = (
        baseline_dir
        / f"s_linker24_role_orchestrator_{name}_links.csv"
    )
    baseline = load_links(baseline_csv)
    floor = load_links(
        baseline_csv, source_filter="s24_relation_role"
    )
    profile_path = (
        baseline_dir
        / "phase_states/s_linker24_role_orchestrator/openai"
        / name
        / "profile.pkl"
    )
    with profile_path.open("rb") as handle:
        profile = pickle.load(handle)
    linker = SLinker24SimplePilot(backend=LLMBackend.OPENAI)
    linker.model_knowledge = profile["model_knowledge"]
    linker.doc_knowledge = profile["doc_knowledge"]
    candidates = linker._apply_role_handles(
        linker._catalog_role_handles(components),
        sentences,
        components,
        floor,
    )
    approved, decisions = linker._review_role_candidates(
        candidates, sentences
    )
    additions = [
        SadSamLink(
            candidate.sentence_number,
            candidate.component_id,
            candidate.component_name,
            source="s24_relation_role",
        )
        for candidate in approved
    ]
    replacement = linker._union(floor, additions)
    export_links_csv(
        replacement,
        results_dir / f"s24_simple_{name}_links.csv",
    )
    role_pairs = {
        (link.sentence_number, link.component_id)
        for link in additions
    }
    return {
        "dataset": name,
        "baseline": score(baseline, gold),
        "replacement": score(replacement, gold),
        "role_tp": len(role_pairs & gold),
        "role_fp": len(role_pairs - gold),
        "accepted": [
            {
                "sentence": candidate.sentence_number,
                "component": candidate.component_name,
            }
            for candidate in approved
        ],
        "decisions": [
            {
                "sentence": candidate.sentence_number,
                "component": candidate.component_name,
                **decisions[
                    (
                        candidate.sentence_number,
                        candidate.component_id,
                    )
                ],
            }
            for candidate in candidates
        ],
        "prompt_metrics": prompt_metrics(linker),
        "trace": linker._llm_calls,
    }


def full_project(name, results_dir):
    paths = DATASETS[name]
    linker = SLinker24SimplePilot(backend=LLMBackend.OPENAI)
    links = linker.link(paths["text"], paths["model"])
    gold = load_gold_sam(str(paths["gold_sam"]))
    export_links_csv(
        links, results_dir / f"s24_simple_{name}_links.csv"
    )
    role = {
        (link.sentence_number, link.component_id)
        for link in links
        if link.source == "s24_relation_role"
    }
    return {
        "dataset": name,
        "score": score(links, gold),
        "role_tp": len(role & gold),
        "role_fp": len(role - gold),
        "workflow": linker.orchestrator_workflow,
        "prompt_metrics": prompt_metrics(linker),
        "llm_calls": len(linker._llm_calls),
        "trace": linker._llm_calls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("replay", "full"), required=True
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "replay" and args.baseline_dir is None:
        parser.error("--baseline-dir is required for replay")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        (
            replay_project(
                dataset, args.baseline_dir, args.results_dir
            )
            if args.mode == "replay"
            else full_project(dataset, args.results_dir)
        )
        for dataset in args.datasets
    ]
    role_tp = sum(row["role_tp"] for row in rows)
    role_fp = sum(row["role_fp"] for row in rows)
    passed = role_tp >= 10 and role_fp == 0
    result = {
        "mode": args.mode,
        "model": "gpt-5.6-terra",
        "reasoning_effort": "none",
        "datasets": rows,
        "role_tp": role_tp,
        "role_fp": role_fp,
        "pass_role_gate": passed,
    }
    output = args.results_dir / "pilot_results.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "mode": args.mode,
        "datasets": [
            {
                key: row[key]
                for key in (
                    "dataset",
                    "role_tp",
                    "role_fp",
                    "prompt_metrics",
                )
            }
            for row in rows
        ],
        "role_tp": role_tp,
        "role_fp": role_fp,
        "pass_role_gate": passed,
        "results": str(output),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
