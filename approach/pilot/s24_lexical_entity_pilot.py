#!/usr/bin/env python3
"""Pilot exact catalog-signature normalization inside S24's entity pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from run_ablation import DATASETS, eval_metrics, export_links_csv, load_gold_sam

from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.linkers.experimental.s_linker24_role_orchestrator import (
    SLinker24RoleOrchestrator,
)


class SLinker24LexicalEntityPilot(SLinker24RoleOrchestrator):
    """Trace-isolated name for the promoted lexical-entity design."""

    _VARIANT_NAME = "s_linker24_lexical_entity_pilot"


def run_project(name, results_dir, backend):
    paths = DATASETS[name]
    linker = SLinker24LexicalEntityPilot(backend=backend)
    links = linker.link(str(paths["text"]), str(paths["model"]))
    gold = load_gold_sam(str(paths["gold_sam"]))
    pairs = {
        (link.sentence_number, link.component_id) for link in links
    }
    metrics = eval_metrics(pairs, gold)
    lexical = [
        link for link in links
        if link.source == "s24_entity_orthographic"
    ]
    lexical_pairs = {
        (link.sentence_number, link.component_id) for link in lexical
    }
    lexical_tp = len(lexical_pairs & gold)
    lexical_fp = len(lexical_pairs - gold)
    export_links_csv(
        links,
        results_dir / f"s24_lexical_entity_{name}_links.csv",
    )
    return {
        "dataset": name,
        "metrics": metrics,
        "lexical_tp": lexical_tp,
        "lexical_fp": lexical_fp,
        "lexical_links": [
            {
                "sentence": link.sentence_number,
                "component": link.component_name,
            }
            for link in lexical
        ],
        "workflow": [
            item["action"] for item in linker.orchestrator_workflow
        ],
        "llm_calls": len(linker._llm_calls),
        "trace": linker._llm_calls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=["bigbluebutton"]
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("codex", "openai"),
        default="codex",
    )
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    backend = LLMBackend(args.backend)
    rows = [
        run_project(dataset, args.results_dir, backend)
        for dataset in args.datasets
    ]
    lexical_tp = sum(row["lexical_tp"] for row in rows)
    lexical_fp = sum(row["lexical_fp"] for row in rows)
    passed = lexical_tp >= 2 and lexical_fp == 0
    result = {
        "protocol": (
            "fresh intact-document S24 lexical-entity pilot; exact catalog "
            "signatures augment entity candidates before the unchanged "
            "entity validator; gold loaded only after inference"
        ),
        "backend": args.backend,
        "model": (
            "Codex CLI configured model"
            if args.backend == "codex"
            else "gpt-5.6-terra"
        ),
        "datasets": rows,
        "lexical_tp": lexical_tp,
        "lexical_fp": lexical_fp,
        "pass_gate": passed,
    }
    output = args.results_dir / "pilot_results.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "lexical_tp": lexical_tp,
        "lexical_fp": lexical_fp,
        "pass_gate": passed,
        "results": str(output),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
