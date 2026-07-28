#!/usr/bin/env python3
"""Checkpoint replay for general S24 tool-owned discovery."""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from run_ablation import DATASETS, eval_metrics, load_gold_sam

from llm_sad_sam.core.data_types_v2 import CandidateLink, SadSamLink
from llm_sad_sam.core.document_loader_v2 import load_sentences
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker24_role_orchestrator import (
    SLinker24RoleOrchestrator,
)


class SLinker24GeneralDiscoveryPilot(SLinker24RoleOrchestrator):
    """S24 pilot with tool-owned discovery and a minimal semantic contract."""

    def discover_participants(
        self, sentences, components, current_links
    ):
        catalog = [component.name for component in components]
        name_to_component = {
            component.name.casefold(): component for component in components
        }
        form_owners = {
            component.name.casefold(): component.name
            for component in components
        }
        form_owners.update({
            term.casefold(): getattr(entry, "component", entry)
            for term, entry in getattr(
                self.doc_knowledge, "aliases", {}
            ).items()
        })
        current = {
            (link.sentence_number, link.component_id)
            for link in current_links
        }
        forms_by_component = self._identity_forms_by_component()
        candidates = {}
        for start in range(0, len(sentences), 25):
            batch = sentences[start:start + 25]
            prompt = f"""Propose every plausible shortened or contextual
expression for a catalog component. Another tool decides whether the link is
correct, so be exhaustive rather than certain. Include inflected forms.

CATALOG
{json.dumps(catalog)}

SENTENCES
{json.dumps([
    {"sentence": sentence.number, "text": sentence.text}
    for sentence in batch
])}

Return a proposal only when the sentence contains a quoted expression. Do not
return the component's full catalog name or a code/path fragment. The quote
must be contiguous exact source text.

JSON only:
{{"mentions":[{{"sentence":1,"target":"exact catalog name",
"quote":"exact shortened expression"}}]}}
"""
            data = self._ask(
                prompt,
                phase=f"phase_24_general_discovery_{start // 25 + 1}",
                require_present="mentions",
                label="S24 general participant discovery",
                timeout=240,
            )
            sent_map = {sentence.number: sentence for sentence in batch}
            for item in data.get("mentions", []):
                sentence_value = str(item.get("sentence", ""))
                if not sentence_value.isdigit():
                    continue
                sentence_number = int(sentence_value)
                sentence = sent_map.get(sentence_number)
                target = name_to_component.get(
                    str(item.get("target", "")).strip().casefold()
                )
                quote = str(item.get("quote", "")).strip()
                quote = quote.strip("\"'“”‘’")
                if sentence is None or target is None or not quote:
                    continue
                owner = form_owners.get(quote.casefold())
                if (
                    owner is not None
                    and owner.casefold() != target.name.casefold()
                ):
                    continue
                match_start = sentence.text.casefold().find(quote.casefold())
                if match_start < 0:
                    continue
                quote = sentence.text[
                    match_start:match_start + len(quote)
                ]
                key = (sentence_number, target.id)
                identity_forms = [
                    target.name,
                    *forms_by_component.get(target.name, []),
                ]
                if (
                    key in current
                    or any(
                        self._find_exact_form(sentence.text, form)
                        for form in identity_forms
                    )
                ):
                    continue
                candidates[key] = CandidateLink(
                    sentence_number,
                    sentence.text,
                    target.name,
                    target.id,
                    quote,
                    source="general_participant_discovery",
                )
        return list(candidates.values())

    def discover_catalog_overlap(
        self, sentences, components, current_links
    ):
        """Propose unique catalog-token continuations found in prose."""
        tokens_by_component = {
            component.id: [
                token.casefold()
                for token in re.findall(
                    r"[A-Za-z]+[A-Za-z0-9]*|\d+",
                    component.name,
                )
            ]
            for component in components
        }
        current = {
            (link.sentence_number, link.component_id)
            for link in current_links
        }
        forms_by_component = self._identity_forms_by_component()
        candidates = {}
        for sentence in sentences:
            words = re.finditer(
                r"[A-Za-z]+[A-Za-z0-9]*|\d+", sentence.text
            )
            for match in words:
                if self._qualified_identifier_boundary(
                    sentence.text, match.start(), match.end()
                ):
                    continue
                word = match.group(0)
                surface = word.casefold()
                owners = [
                    component
                    for component in components
                    if any(
                        surface.startswith(token)
                        for token in tokens_by_component[component.id]
                    )
                ]
                if len(owners) != 1:
                    continue
                component = owners[0]
                key = (sentence.number, component.id)
                identity_forms = [
                    component.name,
                    *forms_by_component.get(component.name, []),
                ]
                if (
                    key in current
                    or any(
                        self._find_exact_form(sentence.text, form)
                        for form in identity_forms
                    )
                ):
                    continue
                candidates[key] = CandidateLink(
                    sentence.number,
                    sentence.text,
                    component.name,
                    component.id,
                    word,
                    source="catalog_overlap_discovery",
                )
        return list(candidates.values())

    def review_minimal(self, candidates, full_sentences):
        participant_candidates, decisions = self._classify_denotations(
            candidates, full_sentences
        )
        approved, reviewed = self._review_minimal_batch(
            participant_candidates, full_sentences
        )
        for key, decision in reviewed.items():
            decisions[key] = {
                **decisions.get(key, {}),
                **decision,
            }
        return approved, decisions

    def _classify_denotations(self, candidates, full_sentences):
        sent_map = {
            sentence.number: sentence for sentence in full_sentences
        }
        by_key = {}
        for start in range(0, len(candidates), 25):
            batch = candidates[start:start + 25]
            evidence_ids = {
                sentence.number
                for candidate in batch
                for sentence in full_sentences
                if abs(sentence.number - candidate.sentence_number) <= 2
            }
            sentence_table = [
                {
                    "sentence": number,
                    "text": sent_map[number].text,
                }
                for number in sorted(evidence_ids)
            ]
            cases = [
                {
                    "case": number,
                    "source": candidate.sentence_number,
                    "expression": candidate.matched_text,
                }
                for number, candidate in enumerate(batch, 1)
            ]
            prompt = f"""Classify what each expression itself denotes in its
local context: participant for a software participant, or associated for
something merely associated with software.

SENTENCES
{json.dumps(sentence_table)}

CASES
{json.dumps(cases)}

Claim must be a contiguous exact substring of the source sentence.

JSON only:
{{"judgments":[{{"case":1,"denotation":"participant",
"claim":"exact source quote"}}]}}
"""
            data = self._ask(
                prompt,
                phase="phase_24_general_denotation",
                require_present="judgments",
                label="S24 general denotation",
                timeout=240,
            )
            for item in data.get("judgments", []):
                case_value = str(item.get("case", ""))
                if not case_value.isdigit():
                    continue
                number = int(case_value)
                if not 1 <= number <= len(batch):
                    continue
                candidate = batch[number - 1]
                claim = str(item.get("claim", "")).strip()
                claim = claim.strip("\"'“”‘’")
                denotation = str(item.get("denotation", "")).strip()
                valid = (
                    denotation in {"participant", "associated"}
                    and bool(claim)
                    and claim.casefold()
                    in candidate.sentence_text.casefold()
                )
                key = (
                    candidate.sentence_number,
                    candidate.component_id,
                )
                by_key[key] = {
                    "approved": False,
                    "requested_keep": False,
                    "evidence_valid": valid,
                    "claim": claim,
                    "denotation": denotation,
                    "alternative": "not reviewed",
                }
        participant_candidates = [
            candidate
            for candidate in candidates
            if by_key.get(
                (candidate.sentence_number, candidate.component_id),
                {},
            ).get("denotation") == "participant"
            and by_key[
                (candidate.sentence_number, candidate.component_id)
            ]["evidence_valid"]
        ]
        return participant_candidates, by_key

    def _review_minimal_batch(self, candidates, full_sentences):
        if not candidates:
            return [], {}
        forms_by_component = self._identity_forms_by_component()
        anchors_by_target = {}
        for target in {
            candidate.component_name for candidate in candidates
        }:
            forms = [target, *forms_by_component.get(target, [])]
            anchors_by_target[target] = [
                {"sentence": sentence.number, "text": sentence.text}
                for sentence in full_sentences
                if any(
                    self._find_exact_form(sentence.text, form)
                    for form in forms
                )
            ]
        sent_map = {
            sentence.number: sentence for sentence in full_sentences
        }
        cases = []
        allowed_anchors = {}
        evidence_sentences = set()
        for number, candidate in enumerate(candidates, 1):
            anchors = sorted(
                anchors_by_target.get(candidate.component_name, []),
                key=lambda item: (
                    abs(item["sentence"] - candidate.sentence_number),
                    item["sentence"],
                ),
            )[:3]
            anchor_ids = [item["sentence"] for item in anchors]
            context = [
                sentence.number
                for sentence in full_sentences
                if abs(sentence.number - candidate.sentence_number) <= 4
            ]
            allowed_anchors[number] = set(anchor_ids)
            evidence_sentences.update(context)
            evidence_sentences.update(anchor_ids)
            cases.append({
                "case": number,
                "source": candidate.sentence_number,
                "expression": candidate.matched_text,
                "target": candidate.component_name,
                "context": context,
                "anchors": anchor_ids,
            })
        sentence_table = [
            {"sentence": number, "text": sent_map[number].text}
            for number in sorted(evidence_sentences)
        ]
        prompt = f"""For each case, do the expression and target denote the
same participant? A longer or shorter label may denote the same participant.
Reject when a distinct referent is better supported. Keep only architectural
claims.

SENTENCES
{json.dumps(sentence_table)}

CASES
{json.dumps(cases)}

Use only a listed anchor. Claim must be one contiguous exact substring of the
source sentence.

JSON only:
{{"judgments":[{{"case":1,"keep":true,"anchor_sentence":1,
"claim":"exact source quote","alternative":"strongest alternative or none"}}]}}
"""
        data = self._ask(
            prompt,
            phase="phase_24_general_participant_review",
            require_present="judgments",
            label="S24 general participant review",
            timeout=240,
        )
        by_case = {}
        for item in data.get("judgments", []):
            case_value = str(item.get("case", ""))
            anchor_value = str(item.get("anchor_sentence", ""))
            if not case_value.isdigit():
                continue
            number = int(case_value)
            if not 1 <= number <= len(candidates):
                continue
            candidate = candidates[number - 1]
            anchor = int(anchor_value) if anchor_value.isdigit() else 0
            claim = str(item.get("claim", "")).strip()
            claim = claim.strip("\"'“”‘’")
            alternative = str(item.get("alternative", "")).strip()
            evidence_valid = (
                anchor in allowed_anchors[number]
                and bool(claim)
                and claim.casefold() in candidate.sentence_text.casefold()
                and bool(alternative)
            )
            by_case[number] = {
                "approved": (
                    item.get("keep") is True
                    and evidence_valid
                ),
                "requested_keep": item.get("keep") is True,
                "evidence_valid": evidence_valid,
                "anchor_sentence": anchor,
                "claim": claim,
                "alternative": alternative,
            }
        approved = [
            candidate
            for number, candidate in enumerate(candidates, 1)
            if by_case.get(number, {}).get("approved") is True
        ]
        decisions = {
            (candidate.sentence_number, candidate.component_id):
                by_case.get(number, {
                    "approved": False,
                    "requested_keep": False,
                    "evidence_valid": False,
                    "alternative": "missing judgment",
                })
            for number, candidate in enumerate(candidates, 1)
        }
        return approved, decisions


def pairs(links):
    return {
        (link.sentence_number, link.component_id) for link in links
    }


def load_phase(root, dataset, phase):
    path = (
        root
        / "s_linker24_role_orchestrator/openai"
        / dataset
        / f"{phase}.pkl"
    )
    with path.open("rb") as handle:
        return pickle.load(handle)


def run_project(dataset, checkpoint_root, discovery, judge):
    paths = DATASETS[dataset]
    components = parse_pcm_repository(paths["model"])
    sentences = load_sentences(paths["text"])
    gold = load_gold_sam(str(paths["gold_sam"]))
    profile = load_phase(checkpoint_root, dataset, "profile")
    entity = load_phase(
        checkpoint_root, dataset, "tool_entity_pipeline"
    )["links"]
    coreference = load_phase(
        checkpoint_root, dataset, "tool_coreference_pipeline"
    )["links"]
    original = load_phase(
        checkpoint_root, dataset, "tool_relation_role_resolution"
    )
    floor = SLinker24RoleOrchestrator._union(entity, coreference)
    linker = SLinker24GeneralDiscoveryPilot(backend=LLMBackend.OPENAI)
    linker.model_knowledge = profile["model_knowledge"]
    linker.doc_knowledge = profile["doc_knowledge"]
    linker._current_text_path = paths["text"]
    if discovery == "checkpoint":
        proposed = original["feedback"]["proposed"]
        by_key = {
            (item["sentence"], item["component"]): item
            for item in proposed
        }
        handles = linker._catalog_role_handles(components)
        candidates = linker._apply_role_handles(
            handles, sentences, components, floor
        )
        candidates = [
            candidate
            for candidate in candidates
            if (
                candidate.sentence_number,
                candidate.component_name,
            ) in by_key
        ]
    elif discovery == "general":
        candidates = linker.discover_participants(
            sentences, components, floor
        )
    else:
        candidates = linker.discover_catalog_overlap(
            sentences, components, floor
        )
    if judge == "current":
        approved, decisions = linker._review_role_candidates(
            candidates, sentences
        )
    else:
        approved, decisions = linker.review_minimal(
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
    original_role = original["links"]
    candidate_keys = pairs(candidates)
    original_gold = pairs(original_role) & gold
    score = eval_metrics(
        pairs(SLinker24RoleOrchestrator._union(floor, additions)),
        gold,
    )
    return {
        "dataset": dataset,
        "discovery": discovery,
        "judge": judge,
        "candidate_count": len(candidates),
        "candidate_tp_reach": len(candidate_keys & gold),
        "checkpoint_role_tp_recalled": len(
            candidate_keys & original_gold
        ),
        "checkpoint_role_tp_total": len(original_gold),
        "approved": [
            {
                "sentence": candidate.sentence_number,
                "component": candidate.component_name,
                "quote": candidate.matched_text,
            }
            for candidate in approved
        ],
        "role_tp": len(pairs(additions) & gold),
        "role_fp": len(pairs(additions) - gold),
        "score": score,
        "decisions": [
            {
                "sentence": candidate.sentence_number,
                "component": candidate.component_name,
                "quote": candidate.matched_text,
                **decisions[
                    (candidate.sentence_number, candidate.component_id)
                ],
            }
            for candidate in candidates
        ],
        "prompt_tokens": sum(
            (item.get("token_usage") or {}).get("prompt_tokens", 0)
            for item in linker._llm_calls
        ),
        "trace": linker._llm_calls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root", type=Path, required=True
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["bigbluebutton", "teammates"]
    )
    parser.add_argument(
        "--discovery",
        choices=("checkpoint", "general", "overlap"),
        required=True,
    )
    parser.add_argument(
        "--judge", choices=("current", "minimal"), required=True
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        run_project(
            dataset,
            args.checkpoint_root,
            args.discovery,
            args.judge,
        )
        for dataset in args.datasets
    ]
    result = {
        "model": "gpt-5.6-terra",
        "reasoning_effort": "none",
        "discovery": args.discovery,
        "judge": args.judge,
        "datasets": rows,
        "candidate_count": sum(
            row["candidate_count"] for row in rows
        ),
        "checkpoint_role_tp_recalled": sum(
            row["checkpoint_role_tp_recalled"] for row in rows
        ),
        "checkpoint_role_tp_total": sum(
            row["checkpoint_role_tp_total"] for row in rows
        ),
        "role_tp": sum(row["role_tp"] for row in rows),
        "role_fp": sum(row["role_fp"] for row in rows),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
    }
    result["pass"] = (
        result["checkpoint_role_tp_recalled"]
        == result["checkpoint_role_tp_total"]
        and result["role_tp"] >= 10
        and result["role_fp"] == 0
    )
    output = args.results_dir / "pilot_results.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        key: result[key]
        for key in (
            "discovery",
            "judge",
            "candidate_count",
            "checkpoint_role_tp_recalled",
            "checkpoint_role_tp_total",
            "role_tp",
            "role_fp",
            "prompt_tokens",
            "pass",
        )
    }, indent=2))
    print(f"Results: {output}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
