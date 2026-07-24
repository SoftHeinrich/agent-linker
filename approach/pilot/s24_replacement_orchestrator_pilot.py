#!/usr/bin/env python3
"""Checkpoint-backed pilot for an S24 replacement workflow controller.

S21 phase checkpoints are treated as recordings of reusable phase-tool outputs.
The S21 final checkpoint is used only after execution as a comparison baseline;
it is never exposed to the controller or used as a runtime floor.
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

from run_ablation import DATASETS, eval_metrics, load_gold_sam

from llm_sad_sam.core.data_types_v2 import CandidateLink, SadSamLink
from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker24_dynamic import SLinker24Dynamic


PACKAGE_ROOT = ROOT.parent
DEFAULT_CACHE = PACKAGE_ROOT / "results/phase_cache/s_linker24/openai"
DEFAULT_RESULTS = (
    PACKAGE_ROOT / "results/s24_replacement_orchestrator_pilot_20260724"
)
CURRENT_DYNAMIC = (
    PACKAGE_ROOT
    / "results/s24_dynamic_controller_pilot_iter2_oracle_frozen_20260724"
    / "pilot_results.json"
)
PHASE_TOOLS = ("entity_pipeline", "coreference_pipeline", "coverage_audit")


def _load(dataset: str, phase: str, root: Path):
    with (root / dataset / f"{phase}.pkl").open("rb") as handle:
        return pickle.load(handle)  # trusted local phase-tool recording


def _link_view(links, sent_map):
    return [
        {
            "sentence": link.sentence_number,
            "text": sent_map[link.sentence_number].text,
            "component": link.component_name,
            "source": link.source,
        }
        for link in links
        if link.sentence_number in sent_map
    ]


def _controller_feedback(feedback):
    proposed = feedback.get("candidates", feedback.get("proposed", []))
    accepted = feedback.get("accepted", [])
    accepted_keys = {
        (item["sentence"], item["component"]) for item in accepted
    }

    def reference(item):
        return {
            "sentence": item["sentence"],
            "component": item["component"],
        }

    return {
        "accepted": [reference(item) for item in accepted],
        "rejected": [
            reference(item)
            for item in proposed
            if (item["sentence"], item["component"]) not in accepted_keys
        ],
    }


def _controller_link_view(links):
    return [
        {
            "sentence": link.sentence_number,
            "component": link.component_name,
            "source": link.source,
        }
        for link in links
    ]


def _controller_action(linker, profile, remaining, history, current, sent_map):
    prompt = f"""You orchestrate trace-linking phase tools for one software project.
There is no base linker and no protected floor. Choose one available tool or
finalize the current result. You cannot propose, validate, add, or remove links.

TOOLS
- entity_pipeline: existing knowledge-aware entity extraction followed by the
  existing two-pass evidence validator.
- coreference_pipeline: existing reference-resolution extraction followed by
  its reference validator.
- coverage_audit: inspect the document, catalog, aliases, and CURRENT LINKS for
  semantically grounded omissions across identity and contextual-reference
  evidence, then use the existing two-pass evidence validator.
- finalize: return the union of links produced by completed tools.

WORKFLOW METHOD
- Use the document's naming/reference style, component ambiguity and aliases,
  current coverage, and concrete tool feedback.
- Entity, coreference, and audit tools cover different evidence modes; choose
  and order only those warranted by this project.
- Treat candidate/link counts as observations, never thresholds.
- Do not infer gold, use project identity, or create link decisions yourself.
- Finalize when the current result has covered the project's supported reference
  modes and no unused tool has complementary evidence.

PROJECT PROFILE
{json.dumps(profile, indent=2)}

COMPLETED TOOL FEEDBACK
{json.dumps(history, indent=2)}

CURRENT LINKS
{json.dumps(_controller_link_view(current), indent=2)}

AVAILABLE ACTIONS
{json.dumps(remaining + ["finalize"])}

Return JSON only:
{{
  "action": "one available action",
  "document_reference_style": "brief runtime assessment",
  "component_profile_effect": "brief runtime assessment",
  "coverage_gap": "complementary evidence still missing or none",
  "reason": "brief workflow reason"
}}
"""
    data = linker._ask(
        prompt,
        phase=f"phase_24_orchestrator_{len(history) + 1}",
        require_present="action",
        label="S24 replacement controller",
    )
    action = str(data.get("action", "")).strip()
    if action not in remaining + ["finalize"]:
        raise RuntimeError(f"invalid replacement action: {action!r}")
    decision = {
        key: str(data.get(key, "")).strip()
        for key in (
            "document_reference_style",
            "component_profile_effect",
            "coverage_gap",
            "reason",
        )
    }
    return action, decision


def _audit_candidates(
    linker, sentences, components, aliases, current_links, sent_map
):
    current_by_sentence = {}
    for link in current_links:
        current_by_sentence.setdefault(link.sentence_number, []).append(
            link.component_name
        )
    items = "\n\n".join(
        f"ITEM S{sentence.number}\n"
        f"  SENTENCE: {sentence.text}\n"
        f"  CURRENT LINKS: "
        f"{json.dumps(current_by_sentence.get(sentence.number, []))}"
        for sentence in sentences
    )
    prompt = f"""Audit trace-link coverage for one software design document.
Find supported component references missing from CURRENT LINKS. This is a
candidate-generation tool; a separate validator makes final decisions.

COMPONENT CATALOG
{json.dumps([component.name for component in components])}

APPROVED DOCUMENT ALIASES
{json.dumps(aliases)}

For each proposed omission, establish:
- REFERENT IDENTITY: the exact quoted phrase denotes or unambiguously invokes
  the catalog component itself, not merely an associated user, browser,
  algorithm, implementation, package, connection, data item, action, or result;
- ARCHITECTURAL PARTICIPATION: the sentence states an architectural fact in
  which that component participates as subject, object, endpoint, service,
  boundary, or contextual referent;
- COMPETING REFERENT: the strongest alternative interpretation, or "none".

Enumerate every distinct participating catalog component; do not stop after the
most salient one. Negated and contrastive relations still specify architectural
boundaries. A descriptive heading or caption is evidence when it identifies a
component-owned architectural flow. Do not repeat CURRENT LINKS. Reject
code/package-only mentions and unsupported group references. Use only exact
catalog names and exact source quotes.

DOCUMENT
{items}

Return JSON only:
{{"omissions":[{{"sentence":1,"component":"exact catalog name",
"quote":"exact source words","referent":"thing denoted",
"architectural_role":"how component participates",
"competing_referent":"alternative or none"}}]}}
"""
    data = linker._ask(
        prompt,
        phase="phase_24_coverage_audit",
        require_present="omissions",
        label="S24 coverage audit",
        timeout=240,
    )
    names = {component.name: component.id for component in components}
    current_keys = {
        (link.sentence_number, link.component_id) for link in current_links
    }
    candidates = {}
    for item in data.get("omissions", []):
        try:
            sentence_number = int(item["sentence"])
            component_name = str(item["component"]).strip()
            quote = str(item["quote"]).strip()
        except Exception:
            continue
        sentence = sent_map.get(sentence_number)
        if (
            component_name not in names
            or not sentence
            or not quote
            or quote.casefold() not in sentence.text.casefold()
        ):
            continue
        key = (sentence_number, names[component_name])
        if key in current_keys:
            continue
        candidates.setdefault(
            key,
            CandidateLink(
                sentence_number,
                sentence.text,
                component_name,
                names[component_name],
                quote,
                source="coverage_audit",
            ),
        )
    return list(candidates.values())


def _run_coverage_audit(
    linker, sentences, components, aliases, current_links, sent_map
):
    candidates = _audit_candidates(
        linker, sentences, components, aliases, current_links, sent_map
    )
    bundles = {
        (candidate.sentence_number, candidate.component_id):
            linker._build_evidence_bundle(
                candidate,
                sent_map,
                rationale=(
                    "semantic coverage audit: identity and architectural "
                    "participation"
                ),
            )
        for candidate in candidates
    }
    approved, decisions = linker._validate_with_evidence(
        candidates,
        bundles,
        components,
        sent_map,
        p1_tag="phase_24_audit_p1",
        p2_tag="phase_24_audit_p2",
        stage_label="coverage_audit",
    )
    links = [
        SadSamLink(
            candidate.sentence_number,
            candidate.component_id,
            candidate.component_name,
            source="s24_coverage_audit",
        )
        for candidate in approved
    ]
    return links, {
        "proposed": _link_view(
            [
                SadSamLink(
                    candidate.sentence_number,
                    candidate.component_id,
                    candidate.component_name,
                    source="coverage_audit_candidate",
                )
                for candidate in candidates
            ],
            sent_map,
        ),
        "accepted": _link_view(links, sent_map),
        "validator_decisions": [
            {
                "sentence": sentence,
                "component_id": component,
                **decision,
            }
            for (sentence, component), decision in decisions.items()
        ],
    }


def _decision_view(decisions):
    return [
        {
            "sentence": sentence,
            "component_id": component,
            **decision,
        }
        for (sentence, component), decision in decisions.items()
    ]


def _recorded_phase_tools(dataset, cache_root, sent_map):
    layer3 = _load(dataset, "layer3", cache_root)
    layer4 = _load(dataset, "layer4", cache_root)
    entity = [
        SadSamLink(
            candidate.sentence_number,
            candidate.component_id,
            candidate.component_name,
            source="entity",
        )
        for candidate in layer3["validated"]
    ]
    coreference = list(layer4["coref_validated"])
    entity_candidates = [
        SadSamLink(
            candidate.sentence_number,
            candidate.component_id,
            candidate.component_name,
            source="entity_candidate",
        )
        for candidate in layer3["candidates"]
    ]
    return (
        entity,
        {
            "candidates": _link_view(entity_candidates, sent_map),
            "accepted": _link_view(entity, sent_map),
            "validator_decisions": _decision_view(layer3["decisions"]),
        },
        coreference,
        {
            "candidates": _link_view(layer4["coref_raw"], sent_map),
            "accepted": _link_view(coreference, sent_map),
            "metadata": [
                {
                    "sentence": sentence,
                    "component_id": component,
                    **value,
                }
                for (sentence, component), value
                in layer4["coref_metadata"].items()
            ],
            "validator_decisions": _decision_view(
                layer4["coref_decisions"]
            ),
        },
    )


def run_dataset(dataset: str, cache_root: Path) -> dict:
    paths = DATASETS[dataset]
    layer1 = _load(dataset, "layer1", cache_root)
    components = parse_pcm_repository(paths["model"])
    sentences = load_sentences(paths["text"])
    sent_map = build_sent_map(sentences)
    (
        entity_links,
        entity_feedback,
        coref_links,
        coref_feedback,
    ) = _recorded_phase_tools(dataset, cache_root, sent_map)
    aliases = [
        {
            "alias": term,
            "target": getattr(entry, "component", entry),
            "scope": getattr(entry, "scope", "global"),
        }
        for term, entry in layer1["doc_knowledge"].aliases.items()
    ]
    profile = {
        "document": [
            {"sentence": sentence.number, "text": sentence.text}
            for sentence in sentences
        ],
        "component_catalog": [component.name for component in components],
        "ambiguous_components": sorted(
            layer1["model_knowledge"].ambiguous_names
        ),
        "approved_aliases": aliases,
    }

    linker = SLinker24Dynamic(backend=LLMBackend.CODEX)
    linker.model_knowledge = layer1["model_knowledge"]
    linker.doc_knowledge = layer1["doc_knowledge"]
    linker._current_text_path = str(paths["text"])
    remaining = list(PHASE_TOOLS)
    current = []
    history = []
    tool_outputs = {}
    while remaining:
        action, decision = _controller_action(
            linker, profile, remaining, history, current, sent_map
        )
        if action == "finalize":
            history.append({"action": "finalize", **decision})
            break
        if action == "entity_pipeline":
            produced = entity_links
            feedback = entity_feedback
        elif action == "coreference_pipeline":
            produced = coref_links
            feedback = coref_feedback
        else:
            produced, feedback = _run_coverage_audit(
                linker,
                sentences,
                components,
                aliases,
                current,
                sent_map,
            )
        existing = {
            (link.sentence_number, link.component_id) for link in current
        }
        current.extend(
            link
            for link in produced
            if (link.sentence_number, link.component_id) not in existing
        )
        tool_outputs[action] = feedback
        history.append(
            {
                "action": action,
                **decision,
                "feedback": _controller_feedback(feedback),
            }
        )
        remaining.remove(action)
    else:
        history.append(
            {
                "action": "finalize",
                "reason": "all phase capabilities consumed",
            }
        )

    final_pairs = {
        (link.sentence_number, link.component_id) for link in current
    }

    # Evaluation boundary. S21 final and gold are unavailable to runtime above.
    s21_links = list(_load(dataset, "final", cache_root)["final"])
    s21_pairs = {
        (link.sentence_number, link.component_id) for link in s21_links
    }
    gold = load_gold_sam(str(paths["gold_sam"]))
    return {
        "dataset": dataset,
        "workflow": history,
        "selected_tools": [
            step["action"] for step in history if step["action"] != "finalize"
        ],
        "s21": eval_metrics(s21_pairs, gold),
        "final": eval_metrics(final_pairs, gold),
        "difference_from_s21": {
            "added_tp": len((final_pairs - s21_pairs) & gold),
            "added_fp": len((final_pairs - s21_pairs) - gold),
            "removed_tp": len((s21_pairs - final_pairs) & gold),
            "removed_fp": len((s21_pairs - final_pairs) - gold),
        },
        "links": _link_view(current, sent_map),
        "tool_outputs": tool_outputs,
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
    datasets = [run_dataset(name, args.cache_root) for name in args.datasets]
    dynamic = json.loads(CURRENT_DYNAMIC.read_text())
    dynamic_rows = dynamic["datasets"]
    workflows = {tuple(item["selected_tools"]) for item in datasets}
    def aggregate(rows, key):
        tp = sum(item[key]["tp"] for item in rows)
        fp = sum(item[key]["fp"] for item in rows)
        fn = sum(item[key]["fn"] for item in rows)
        macro_f1 = sum(item[key]["F1"] for item in rows) / len(rows)
        macro_f2 = sum(
            5 * item[key]["tp"]
            / (
                5 * item[key]["tp"]
                + 4 * item[key]["fn"]
                + item[key]["fp"]
            )
            for item in rows
        ) / len(rows)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "macro_f1": macro_f1,
            "macro_f2": macro_f2,
            "pooled_f1": 2 * tp / (2 * tp + fp + fn),
            "pooled_f2": 5 * tp / (5 * tp + 4 * fn + fp),
        }

    s21_metrics = aggregate(datasets, "s21")
    final_metrics = aggregate(datasets, "final")
    dynamic_metrics = aggregate(dynamic_rows, "final")
    passed = (
        final_metrics["macro_f2"] > s21_metrics["macro_f2"]
        and final_metrics["pooled_f2"] > s21_metrics["pooled_f2"]
        and final_metrics["macro_f2"] > dynamic_metrics["macro_f2"]
        and final_metrics["pooled_f2"] > dynamic_metrics["pooled_f2"]
    )
    summary = {
        "protocol": (
            "replacement controller over recorded S21 phase tools and fresh "
            "semantic coverage audit; S21 final loaded only for "
            "post-run comparison"
        ),
        "datasets": datasets,
        "aggregate": {
            "s21": s21_metrics,
            "replacement": final_metrics,
            "current_dynamic": dynamic_metrics,
            "macro_f2_delta_vs_s21": (
                final_metrics["macro_f2"] - s21_metrics["macro_f2"]
            ),
            "pooled_f2_delta_vs_s21": (
                final_metrics["pooled_f2"] - s21_metrics["pooled_f2"]
            ),
            "macro_f2_delta_vs_dynamic": (
                final_metrics["macro_f2"] - dynamic_metrics["macro_f2"]
            ),
            "pooled_f2_delta_vs_dynamic": (
                final_metrics["pooled_f2"] - dynamic_metrics["pooled_f2"]
            ),
            "distinct_workflows": len(workflows),
        },
        "pass_gate": {
            "macro_f2_beats_s21": (
                final_metrics["macro_f2"] > s21_metrics["macro_f2"]
            ),
            "pooled_f2_beats_s21": (
                final_metrics["pooled_f2"] > s21_metrics["pooled_f2"]
            ),
            "macro_f2_beats_current_dynamic": (
                final_metrics["macro_f2"] > dynamic_metrics["macro_f2"]
            ),
            "pooled_f2_beats_current_dynamic": (
                final_metrics["pooled_f2"] > dynamic_metrics["pooled_f2"]
            ),
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
