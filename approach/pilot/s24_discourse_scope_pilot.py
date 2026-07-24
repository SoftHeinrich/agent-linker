#!/usr/bin/env python3
"""Same-floor pilot for discourse-scoped participant resolution."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from run_ablation import DATASETS, eval_metrics, export_links_csv, load_gold_sam

from llm_sad_sam.core.data_types_v2 import CandidateLink, SadSamLink
from llm_sad_sam.core.document_loader_v2 import load_sentences
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker24_role_orchestrator import (
    SLinker24RoleOrchestrator,
)


def terminal_role_handles(components):
    tokens_by_component = {
        component.name: re.findall(
            r"[A-Za-z]+[A-Za-z0-9]*|\d+", component.name
        )
        for component in components
    }
    owners = {}
    for component_name, tokens in tokens_by_component.items():
        if len(tokens) < 2:
            continue
        token = tokens[-1].casefold()
        owners.setdefault(token, set()).add(component_name)
    handles = []
    for component_name, tokens in tokens_by_component.items():
        if len(tokens) < 2:
            continue
        terminal = tokens[-1]
        if len(owners[terminal.casefold()]) != 1:
            continue
        expressions = [terminal]
        if terminal.isalpha() and not terminal.casefold().endswith("s"):
            expressions.append(f"{terminal}s")
        handles.append({
            "component": component_name,
            "expressions": expressions,
        })
    return handles


def discourse_candidates(
    linker, sentences, components, floor_links
):
    name_to_id = {component.name: component.id for component in components}
    current = {
        (link.sentence_number, link.component_id)
        for link in floor_links
    }
    forms_by_component = linker._identity_forms_by_component()
    candidates = {}
    for handle in terminal_role_handles(components):
        component = handle["component"]
        component_id = name_to_id[component]
        identity_forms = [
            component,
            *forms_by_component.get(component, []),
        ]
        for expression in handle["expressions"]:
            for sentence in sentences:
                key = (sentence.number, component_id)
                matched = linker._find_handle(
                    sentence.text, expression
                )
                if (
                    key in current
                    or not matched
                    or any(
                        linker._find_exact_form(sentence.text, form)
                        for form in identity_forms
                    )
                ):
                    continue
                candidates[key] = CandidateLink(
                    sentence.number,
                    sentence.text,
                    component,
                    component_id,
                    matched,
                    source="discourse_participant",
                )
    return list(candidates.values())


def identity_profiles(linker, targets, sentences):
    forms_by_component = linker._identity_forms_by_component()
    profiles = []
    for target in targets:
        forms = [target, *forms_by_component.get(target, [])]
        anchors = []
        for sentence in sentences:
            matches = [
                linker._find_exact_form(sentence.text, form)
                for form in forms
            ]
            matches = [match for match in matches if match]
            if matches:
                anchors.append({
                    "sentence": sentence.number,
                    "text": sentence.text,
                    "identity_forms": matches,
                })
        profiles.append({"target": target, "anchors": anchors})
    return profiles


def review_discourse_candidates(linker, candidates, sentences):
    if not candidates:
        return [], {}
    sent_map = {sentence.number: sentence for sentence in sentences}
    targets = sorted({
        candidate.component_name for candidate in candidates
    })
    profiles = identity_profiles(linker, targets, sentences)
    anchors_by_target = {
        item["target"]: {
            anchor["sentence"]: anchor
            for anchor in item["anchors"]
        }
        for item in profiles
    }
    cases = [
        {
            "case": number,
            "sentence": candidate.sentence_number,
            "participant": candidate.matched_text,
            "target": candidate.component_name,
            "text": candidate.sentence_text,
        }
        for number, candidate in enumerate(candidates, 1)
    ]
    document = [
        {"sentence": sentence.number, "text": sentence.text}
        for sentence in sentences
    ]
    prompt = f"""Resolve generic or inflected architectural participants using
the document's discourse structure. For each case, keep the mapping only when
the highlighted participant denotes the listed target component in its section
or discourse chain.

An approval must identify an explicit target identity anchor, the participant's
architectural role in the source sentence, and the strongest competing
referent. A common role noun may validly denote a deployed component instance,
its service endpoint, its owned state, or its user-facing instances; do not
reject it merely for being generic or plural when the active discourse scope
and architectural claim select the target. A competing referent must be an
explicit locally plausible entity, not an invented unnamed deployment. When
an apparent alternate label is resolved to the target by a local discourse
chain, cite the exact bridge sentence that makes that resolution possible.

Reject physical host capacity, users or browsers, protocol/technology use,
code paths, and cases where multiple explicit referents remain plausible.
The highlighted noun must itself be a semantic participant in a finite
architectural claim. Reject fragments and uses where it merely modifies the
name of a workflow, process, stage, diagram, artifact, or technology. Full
names, aliases, pronouns, and orthographic variants are owned by other tools.

DOCUMENT
{json.dumps(document)}

TARGET IDENTITY ANCHORS
{json.dumps(profiles)}

CASES
{json.dumps(cases)}

Return JSON only:
{{"judgments":[{{"case":1,"keep":true,
"section_anchor":"exact document quote",
"identity_anchor_sentence":1,
"identity_anchor":"exact quote from that anchor sentence",
"scope_bridge_sentence":1,
"scope_bridge":"exact quote establishing the local discourse chain",
"claim":"exact quote from the source sentence",
"participant_role":"brief role",
"competing_referent":"strongest alternative or none"}}]}}
"""
    data = linker._ask(
        prompt,
        phase="phase_24_discourse_scope_review",
        require_present="judgments",
        label="S24 discourse-scope review",
        timeout=240,
    )
    by_case = {}
    document_text = "\n".join(
        sentence.text for sentence in sentences
    ).casefold()
    for item in data.get("judgments", []):
        if not str(item.get("case", "")).isdigit():
            continue
        number = int(item["case"])
        if not 1 <= number <= len(candidates):
            continue
        candidate = candidates[number - 1]
        anchor_sentence = int(
            item.get("identity_anchor_sentence", 0) or 0
        )
        bridge_sentence = int(
            item.get("scope_bridge_sentence", 0) or 0
        )
        anchor = str(item.get("identity_anchor", "")).strip()
        bridge = str(item.get("scope_bridge", "")).strip()
        section = str(item.get("section_anchor", "")).strip()
        claim = str(item.get("claim", "")).strip()
        role = str(item.get("participant_role", "")).strip()
        competing = str(item.get("competing_referent", "")).strip()
        allowed_anchor = anchors_by_target.get(
            candidate.component_name, {}
        ).get(anchor_sentence)
        allowed_bridge = sent_map.get(bridge_sentence)
        evidence_valid = (
            bool(section)
            and section.casefold() in document_text
            and allowed_anchor is not None
            and bool(anchor)
            and anchor.casefold()
            in allowed_anchor["text"].casefold()
            and allowed_bridge is not None
            and bool(bridge)
            and bridge.casefold()
            in allowed_bridge.text.casefold()
            and bool(claim)
            and claim.casefold()
            in candidate.sentence_text.casefold()
            and bool(role)
            and bool(competing)
        )
        by_case[number] = {
            "approved": item.get("keep") is True and evidence_valid,
            "requested_keep": item.get("keep") is True,
            "evidence_valid": evidence_valid,
            "section_anchor": section,
            "identity_anchor_sentence": anchor_sentence,
            "identity_anchor": anchor,
            "scope_bridge_sentence": bridge_sentence,
            "scope_bridge": bridge,
            "claim": claim,
            "participant_role": role,
            "competing_referent": competing,
        }
    approved = [
        candidate
        for number, candidate in enumerate(candidates, 1)
        if by_case.get(number, {}).get("approved") is True
    ]
    decisions = {
        (candidate.sentence_number, candidate.component_id): {
            **by_case.get(number, {
                "approved": False,
                "requested_keep": False,
                "evidence_valid": False,
                "competing_referent": "missing judgment",
            }),
            "path": "discourse_scope_review",
            "stage": "relation_role_resolution",
        }
        for number, candidate in enumerate(candidates, 1)
    }
    return approved, decisions


def load_links(path, source_filter=None):
    links = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if source_filter and row["source"] == source_filter:
                continue
            links.append(SadSamLink(
                int(row["sentence"]),
                row["component_id"],
                row["component_name"],
                source=row["source"],
            ))
    return links


def run_project(name, baseline_dir, results_dir):
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
    linker = SLinker24RoleOrchestrator(
        backend=LLMBackend.OPENAI
    )
    linker.model_knowledge = profile["model_knowledge"]
    linker.doc_knowledge = profile["doc_knowledge"]
    candidates = discourse_candidates(
        linker, sentences, components, floor
    )
    approved, decisions = review_discourse_candidates(
        linker, candidates, sentences
    )
    additions = [
        SadSamLink(
            candidate.sentence_number,
            candidate.component_id,
            candidate.component_name,
            source="s24_discourse_participant",
        )
        for candidate in approved
    ]
    replacement = linker._union(floor, additions)

    def score(links):
        pairs = {
            (link.sentence_number, link.component_id)
            for link in links
        }
        return eval_metrics(pairs, gold)

    old_role = [
        link for link in baseline
        if link.source == "s24_relation_role"
    ]
    old_role_pairs = {
        (link.sentence_number, link.component_id)
        for link in old_role
    }
    new_role_pairs = {
        (link.sentence_number, link.component_id)
        for link in additions
    }
    old_role_tp = len(old_role_pairs & gold)
    old_role_fp = len(old_role_pairs - gold)
    new_role_tp = len(new_role_pairs & gold)
    new_role_fp = len(new_role_pairs - gold)
    precision = (
        new_role_tp / (new_role_tp + new_role_fp)
        if new_role_tp + new_role_fp
        else 0.0
    )
    export_links_csv(
        replacement,
        results_dir / f"s24_discourse_{name}_links.csv",
    )
    return {
        "dataset": name,
        "baseline": score(baseline),
        "floor_without_old_role": score(floor),
        "replacement": score(replacement),
        "old_role_tp": old_role_tp,
        "old_role_fp": old_role_fp,
        "new_role_tp": new_role_tp,
        "new_role_fp": new_role_fp,
        "new_role_precision": precision,
        "candidates": [
            {
                "sentence": candidate.sentence_number,
                "component": candidate.component_name,
                "participant": candidate.matched_text,
                "text": candidate.sentence_text,
            }
            for candidate in candidates
        ],
        "accepted": [
            {
                "sentence": candidate.sentence_number,
                "component": candidate.component_name,
                "participant": candidate.matched_text,
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
        "llm_calls": len(linker._llm_calls),
        "trace": linker._llm_calls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=["bigbluebutton"]
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        run_project(
            dataset, args.baseline_dir, args.results_dir
        )
        for dataset in args.datasets
    ]
    passed = all(
        row["new_role_tp"] >= row["old_role_tp"] + 3
        and row["new_role_precision"] >= 0.95
        and row["new_role_fp"] <= row["old_role_fp"]
        and row["replacement"]["F1"] > row["baseline"]["F1"]
        and row["replacement"]["F2"] > row["baseline"]["F2"]
        for row in rows
    )
    result = {
        "protocol": (
            "fresh discourse review replacing old role links on the exact "
            "saved spike-013 S24 non-role floor; gold loaded after inference"
        ),
        "model": "gpt-5.6-terra",
        "reasoning_effort": "none",
        "datasets": rows,
        "pass_gate": passed,
    }
    output = args.results_dir / "pilot_results.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "datasets": [
            {
                key: row[key]
                for key in (
                    "dataset",
                    "baseline",
                    "replacement",
                    "old_role_tp",
                    "old_role_fp",
                    "new_role_tp",
                    "new_role_fp",
                    "new_role_precision",
                )
            }
            for row in rows
        ],
        "pass_gate": passed,
        "results": str(output),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
