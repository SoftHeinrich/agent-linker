#!/usr/bin/env python3
"""Fresh intact-document pilot for non-overlapping FN-recovery tools."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from run_ablation import DATASETS, eval_metrics, load_gold_sam

from llm_sad_sam.core.data_types_v2 import CandidateLink, SadSamLink
from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker24_role_orchestrator import (
    SLinker24RoleOrchestrator,
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


def exact_standalone(text, expression):
    match = re.search(
        rf"(?<![\w.-]){re.escape(expression)}(?![\w.-])",
        text,
        re.IGNORECASE,
    )
    return match.group(0) if match else ""


def entity_seen_keys(linker, name_to_id):
    seen = set()
    for step in linker.orchestrator_workflow:
        if step.get("action") != "entity_pipeline":
            continue
        feedback = step.get("feedback", {})
        for outcome in ("accepted", "rejected"):
            for item in feedback.get(outcome, []):
                component_id = name_to_id.get(item["component"])
                if component_id:
                    seen.add((item["sentence"], component_id))
    return seen


def workflow_seen_keys(linker, name_to_id):
    seen = set()
    for step in linker.orchestrator_workflow:
        feedback = step.get("feedback", {})
        for outcome in ("accepted", "rejected"):
            for item in feedback.get(outcome, []):
                component_id = name_to_id.get(item["component"])
                if component_id:
                    seen.add((item["sentence"], component_id))
    return seen


def alias_candidates(
    linker, sentences, components, current_links
):
    name_to_id = {component.name: component.id for component in components}
    current = {
        (link.sentence_number, link.component_id) for link in current_links
    }
    seen = entity_seen_keys(linker, name_to_id)
    aliases = getattr(linker.doc_knowledge, "aliases", {})
    candidates = {}
    for term, entry in aliases.items():
        target = getattr(entry, "component", entry)
        if term.casefold() == target.casefold() or target not in name_to_id:
            continue
        component_id = name_to_id[target]
        for sentence in sentences:
            matched = exact_standalone(sentence.text, term)
            key = (sentence.number, component_id)
            if not matched or key in current or key in seen:
                continue
            candidates[key] = CandidateLink(
                sentence.number,
                sentence.text,
                target,
                component_id,
                matched,
                source="approved_alias_coverage",
            )
    return list(candidates.values())


def identifier_tokens(expression):
    return [
        token.casefold()
        for token in re.findall(
            r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|\d+",
            expression.replace("-", " "),
        )
    ]


def identifier_candidates(
    linker, sentences, components, current_links
):
    name_to_id = {component.name: component.id for component in components}
    current = {
        (link.sentence_number, link.component_id) for link in current_links
    }
    seen = workflow_seen_keys(linker, name_to_id)
    role_handles = {
        item["expression"].casefold()
        for item in linker._catalog_role_handles(components)
    }
    component_tokens = {
        component.name: identifier_tokens(component.name)
        for component in components
    }
    pattern = re.compile(
        r"(?<![\w-])(?<!\w\.)(?:"
        r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+"
        r"|[A-Za-z]+[A-Z][A-Za-z0-9]*"
        r")(?![\w-])(?!\.\w)"
    )
    candidates = {}
    for sentence in sentences:
        for match in pattern.finditer(sentence.text):
            expression = match.group(0)
            if expression.casefold() in role_handles:
                continue
            tokens = identifier_tokens(expression)
            targets = [
                component
                for component in components
                if tokens == component_tokens[component.name]
            ]
            for component in targets:
                key = (sentence.number, component.id)
                if key in current or key in seen:
                    continue
                candidates[key] = CandidateLink(
                    sentence.number,
                    sentence.text,
                    component.name,
                    component.id,
                    expression,
                    source="identifier_identity",
                )
    return list(candidates.values())


def review_identifier_candidates(linker, candidates, sentences):
    if not candidates:
        return [], []
    targets = sorted({candidate.component_name for candidate in candidates})
    profiles = [
        {
            "target": target,
            "identity_anchors": [
                {
                    "sentence": sentence.number,
                    "text": sentence.text,
                }
                for sentence in sentences
                if exact_standalone(sentence.text, target)
            ],
        }
        for target in targets
    ]
    cases = [
        {
            "case": number,
            "sentence": candidate.sentence_number,
            "identifier": candidate.matched_text,
            "target": candidate.component_name,
            "text": candidate.sentence_text,
        }
        for number, candidate in enumerate(candidates, 1)
    ]
    prompt = f"""Resolve catalog-equivalent standalone component identifiers.
For each case, keep the mapping only when the highlighted alternate spelling
names the listed target component. Identity anchors show explicit usage.

TARGET PROFILES
{json.dumps(profiles)}

CASES
{json.dumps(cases)}

JSON only:
{{"judgments":[{{"case":1,"keep":true,"referent":"brief referent"}}]}}
"""
    data = linker._ask(
        prompt,
        phase="phase_24_identifier_identity_review",
        require_present="judgments",
        label="S24 identifier-identity review",
        timeout=240,
    )
    by_case = {
        int(item["case"]): {
            "keep": item.get("keep") is True,
            "referent": str(item.get("referent", "")).strip(),
        }
        for item in data.get("judgments", [])
        if str(item.get("case", "")).isdigit()
    }
    approved = [
        candidate
        for number, candidate in enumerate(candidates, 1)
        if by_case.get(number, {}).get("keep") is True
    ]
    decisions = [
        {
            **cases[number - 1],
            **by_case.get(number, {
                "keep": False,
                "referent": "missing judgment",
            }),
        }
        for number in range(1, len(candidates) + 1)
    ]
    return approved, decisions


def review_alias_candidates(linker, candidates, sentences):
    if not candidates:
        return [], []
    targets = sorted({candidate.component_name for candidate in candidates})
    profiles = [
        {
            "target": target,
            "identity_anchors": [
                {
                    "sentence": sentence.number,
                    "text": sentence.text,
                }
                for sentence in sentences
                if exact_standalone(sentence.text, target)
            ],
        }
        for target in targets
    ]
    cases = [
        {
            "case": number,
            "sentence": candidate.sentence_number,
            "alias": candidate.matched_text,
            "target": candidate.component_name,
            "text": candidate.sentence_text,
        }
        for number, candidate in enumerate(candidates, 1)
    ]
    prompt = f"""Resolve uncovered uses of document-approved component aliases.
For each case, keep the mapping only when the highlighted alias in that
sentence denotes the listed target component. Identity anchors show explicit
project usage.

TARGET PROFILES
{json.dumps(profiles)}

CASES
{json.dumps(cases)}

JSON only:
{{"judgments":[{{"case":1,"keep":true,"referent":"brief referent"}}]}}
"""
    data = linker._ask(
        prompt,
        phase="phase_24_alias_coverage_review",
        require_present="judgments",
        label="S24 alias-coverage review",
        timeout=240,
    )
    by_case = {
        int(item["case"]): {
            "keep": item.get("keep") is True,
            "referent": str(item.get("referent", "")).strip(),
        }
        for item in data.get("judgments", [])
        if str(item.get("case", "")).isdigit()
    }
    approved = [
        candidate
        for number, candidate in enumerate(candidates, 1)
        if by_case.get(number, {}).get("keep") is True
    ]
    decisions = [
        {
            **cases[number - 1],
            **by_case.get(number, {
                "keep": False,
                "referent": "missing judgment",
            }),
        }
        for number in range(1, len(candidates) + 1)
    ]
    return approved, decisions


def run_project(name, include_alias, include_identifier):
    paths = DATASETS[name]
    components = parse_pcm_repository(paths["model"])
    sentences = load_sentences(paths["text"])
    sent_map = build_sent_map(sentences)
    gold = load_gold_sam(str(paths["gold_sam"]))

    linker = SLinker24RoleOrchestrator(backend=LLMBackend.OPENAI)
    baseline_links = linker.link(str(paths["text"]), str(paths["model"]))
    candidates = (
        alias_candidates(linker, sentences, components, baseline_links)
        if include_alias
        else []
    )
    approved, decisions = (
        review_alias_candidates(linker, candidates, sentences)
        if include_alias
        else ([], [])
    )
    additions = [
        SadSamLink(
            candidate.sentence_number,
            candidate.component_id,
            candidate.component_name,
            source="s24_approved_alias_coverage",
        )
        for candidate in approved
    ]
    alias_links = linker._union(baseline_links, additions)
    identifier_proposals = (
        identifier_candidates(
            linker, sentences, components, alias_links
        )
        if include_identifier
        else []
    )
    identifier_approved, identifier_decisions = (
        review_identifier_candidates(
            linker, identifier_proposals, sentences
        )
        if include_identifier
        else ([], [])
    )
    identifier_links = [
        SadSamLink(
            candidate.sentence_number,
            candidate.component_id,
            candidate.component_name,
            source="s24_identifier_identity",
        )
        for candidate in identifier_approved
    ]
    final_links = linker._union(alias_links, identifier_links)

    def score(links):
        pairs = {
            (link.sentence_number, link.component_id) for link in links
        }
        return eval_metrics(pairs, gold)

    def view(items):
        return [
            {
                "sentence": item.sentence_number,
                "text": sent_map[item.sentence_number].text,
                "component": item.component_name,
                "matched_text": getattr(item, "matched_text", ""),
            }
            for item in items
        ]

    return {
        "dataset": name,
        "baseline_workflow": [
            step["action"] for step in linker.orchestrator_workflow
        ],
        "tool_available": bool(candidates),
        "baseline": score(baseline_links),
        "alias_final": score(alias_links),
        "final": score(final_links),
        "candidates": view(candidates),
        "accepted": view(approved),
        "decisions": decisions,
        "identifier_tool_available": bool(identifier_proposals),
        "identifier_candidates": view(identifier_proposals),
        "identifier_accepted": view(identifier_approved),
        "identifier_decisions": identifier_decisions,
        "llm_calls": len(linker._llm_calls),
        "trace": linker._llm_calls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--identifier", action="store_true")
    parser.add_argument("--no-alias", action="store_true")
    args = parser.parse_args()

    rows = [
        run_project(name, not args.no_alias, args.identifier)
        for name in args.datasets
    ]
    baseline = aggregate(rows, "baseline")
    alias_final = aggregate(rows, "alias_final")
    final = aggregate(rows, "final")
    available = [
        row["dataset"] for row in rows if row["tool_available"]
    ]
    identifier_available = [
        row["dataset"]
        for row in rows
        if row["identifier_tool_available"]
    ]
    passed = (
        bool(available or identifier_available)
        and len(set(available + identifier_available)) < len(rows)
        and final["macro_f1"] >= baseline["macro_f1"]
        and final["pooled_f1"] >= baseline["pooled_f1"]
        and final["macro_f2"] > baseline["macro_f2"]
        and final["pooled_f2"] > baseline["pooled_f2"]
    )
    result = {
        "protocol": (
            "fresh intact-document S24 baseline and fresh OpenAI tool review; "
            "gold is used only after inference for full-document scoring"
        ),
        "model": "gpt-5.6-terra",
        "reasoning_effort": "none",
        "datasets": rows,
        "aggregate": {
            "baseline": baseline,
            "alias_final": alias_final,
            "final": final,
            "tool_available_projects": available,
            "identifier_available_projects": identifier_available,
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
