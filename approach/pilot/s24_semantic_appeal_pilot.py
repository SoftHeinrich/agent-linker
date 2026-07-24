#!/usr/bin/env python3
"""Fixed-floor semantic appeal pilot for rejected S21 candidates.

All residual rejections are eligible. Gold is loaded only after appeal decisions.
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

from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker24_dynamic import SLinker24Dynamic


PACKAGE_ROOT = ROOT.parent
DEFAULT_CACHE = PACKAGE_ROOT / "results/phase_cache/s_linker24/openai"
DEFAULT_RESULTS = PACKAGE_ROOT / "results/s24_semantic_appeal_pilot_20260724"
CURRENT_DYNAMIC = (
    PACKAGE_ROOT
    / "results/s24_dynamic_controller_pilot_iter2_oracle_frozen_20260724"
    / "pilot_results.json"
)


def _load(dataset: str, phase: str, root: Path) -> dict:
    with (root / dataset / f"{phase}.pkl").open("rb") as handle:
        return pickle.load(handle)  # trusted local S21 checkpoint


def _entity_cases(layer3, floor_keys):
    return [
        {
            "key": (candidate.sentence_number, candidate.component_id),
            "sentence": candidate.sentence_text,
            "target": candidate.component_name,
            "reference": candidate.matched_text,
            "evidence": layer3["evidence_bundles"].get(
                (candidate.sentence_number, candidate.component_id), {}
            ),
            "original_decision": layer3["decisions"].get(
                (candidate.sentence_number, candidate.component_id), {}
            ),
        }
        for candidate in layer3["candidates"]
        if (candidate.sentence_number, candidate.component_id) not in floor_keys
    ]


def _coref_cases(layer4, floor_keys, sent_map, id_to_name):
    cases = []
    for link in layer4["coref_raw"]:
        key = (link.sentence_number, link.component_id)
        if key in floor_keys:
            continue
        metadata = layer4["coref_metadata"].get(key, {})
        sentence = sent_map.get(link.sentence_number)
        cases.append(
            {
                "key": key,
                "sentence": sentence.text if sentence else "",
                "target": id_to_name[link.component_id],
                "reference": metadata.get("reference", ""),
                "antecedent_sentence": metadata.get("antecedent_sentence"),
                "antecedent_text": metadata.get("antecedent_text", ""),
                "original_decision": layer4["coref_decisions"].get(key, {}),
            }
        )
    return cases


def _appeal(
    linker,
    kind: str,
    cases: list[dict],
    component_names: list[str],
    alias_profile: list[dict],
):
    if not cases:
        return set(), []
    rendered = []
    for number, case in enumerate(cases, 1):
        details = (
            f"  Original evidence: {json.dumps(case.get('evidence', {}))}"
            if kind == "entity"
            else (
                f"  Referring phrase: {case['reference']}\n"
                f"  Antecedent S{case['antecedent_sentence']}: "
                f"{case['antecedent_text']}"
            )
        )
        rendered.append(
            f"Case {number}: candidate {case['target']}\n"
            f"  Source sentence: {case['sentence']}\n"
            f"  Candidate phrase: {case['reference']}\n"
            f"{details}\n"
            f"  Prior conservative outcome: "
            f"{json.dumps(case['original_decision'])}"
        )
    contract = (
        "For an entity candidate, approve only when the candidate phrase denotes "
        "the target architecture component in this sentence and the sentence "
        "states an architectural fact about it."
        if kind == "entity"
        else
        "For a coreference candidate, approve only when the referring phrase "
        "resolves through the supplied antecedent to the target component and "
        "the source sentence states an architectural fact about that component."
    )
    prompt = f"""Act as an appellate reviewer for grounded trace-link candidates
rejected by a conservative first review. Rejection is context, not evidence of
correctness. Review every case from its supplied evidence.

COMPONENT CATALOG
{json.dumps(component_names)}

APPROVED DOCUMENT ALIASES
{json.dumps(alias_profile)}

CONTRACT
{contract}

Apply two independent semantic questions before the verdict:

1. REFERENT IDENTITY — What exact thing does the candidate phrase denote here?
   It must denote the target component itself, not merely a related user,
   browser, server, algorithm, implementation, package, connection, data item,
   action, or result. Runtime aliases may establish identity.
2. CLAIM OWNERSHIP — What exact thing owns the architectural claim? The target
   itself must be the subject whose behavior, structure, responsibility, or
   relation is asserted. Mere proximity to a component mention is insufficient.

For every case, state the phrase's referent, quote the architectural claim (or
"none"), identify the claim owner, and name the strongest competing referent
(or "none") before deciding. Reject code/package-only mentions, captions without
a component claim, and unsupported group references. The controller cannot
influence your verdict.

CASES
{chr(10).join(rendered)}

Return JSON only:
{{"appeals":[{{"case":1,"referent":"thing denoted by candidate phrase",
"claim":"exact quote or none","claim_owner":"thing the claim is about",
"competing_referent":"strongest alternative or none","approve":true}}]}}
"""
    data = linker._ask(
        prompt,
        phase=f"phase_24_{kind}_appeal",
        require_present="appeals",
        label=f"S24 {kind} appeal",
        timeout=180,
    )
    approved = set()
    for item in data.get("appeals", []):
        number = int(item.get("case", 0))
        if item.get("approve") is True and 1 <= number <= len(cases):
            approved.add(cases[number - 1]["key"])
    return approved, data.get("appeals", [])


def run_dataset(dataset: str, cache_root: Path) -> dict:
    paths = DATASETS[dataset]
    floor = list(_load(dataset, "final", cache_root)["final"])
    layer3 = _load(dataset, "layer3", cache_root)
    layer4 = _load(dataset, "layer4", cache_root)
    layer1 = _load(dataset, "layer1", cache_root)
    components = parse_pcm_repository(paths["model"])
    sentences = load_sentences(paths["text"])
    sent_map = build_sent_map(sentences)
    floor_keys = {(link.sentence_number, link.component_id) for link in floor}
    id_to_name = {component.id: component.name for component in components}
    entity = _entity_cases(layer3, floor_keys)
    coref = _coref_cases(layer4, floor_keys, sent_map, id_to_name)

    linker = SLinker24Dynamic(backend=LLMBackend.CODEX)
    linker._current_text_path = str(paths["text"])
    aliases = [
        {
            "alias": term,
            "target": getattr(entry, "component", entry),
            "scope": getattr(entry, "scope", "global"),
        }
        for term, entry in layer1["doc_knowledge"].aliases.items()
    ]
    entity_approved, _ = _appeal(
        linker,
        "entity",
        entity,
        [component.name for component in components],
        aliases,
    )
    coref_approved, _ = _appeal(
        linker,
        "coreference",
        coref,
        [component.name for component in components],
        aliases,
    )
    additions = entity_approved | coref_approved

    # Evaluation boundary: gold is unavailable above this line.
    gold = load_gold_sam(str(paths["gold_sam"]))
    floor_metrics = eval_metrics(floor_keys, gold)
    final_metrics = eval_metrics(floor_keys | additions, gold)
    return {
        "dataset": dataset,
        "inventory": {"entity_appeal": len(entity), "coreference_appeal": len(coref)},
        "approved": {
            "entity_appeal": len(entity_approved),
            "coreference_appeal": len(coref_approved),
        },
        "floor": floor_metrics,
        "final": final_metrics,
        "marginal": {
            "tp": len(additions & gold),
            "fp": len(additions - gold),
            "precision": (
                len(additions & gold) / len(additions) if additions else None
            ),
            "links": [
                {
                    "sentence": sentence,
                    "component_id": component,
                    "component": id_to_name[component],
                    "gold": (sentence, component) in gold,
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
    datasets = [run_dataset(name, args.cache_root) for name in args.datasets]
    tp = sum(item["marginal"]["tp"] for item in datasets)
    fp = sum(item["marginal"]["fp"] for item in datasets)
    precision = tp / (tp + fp) if tp + fp else 0.0
    floor_f1 = sum(item["floor"]["F1"] for item in datasets) / len(datasets)
    final_f1 = sum(item["final"]["F1"] for item in datasets) / len(datasets)
    dynamic_f1 = json.loads(CURRENT_DYNAMIC.read_text())["aggregate"][
        "macro_final_f1"
    ]
    passed = precision >= 0.95 and final_f1 > floor_f1 and final_f1 > dynamic_f1
    summary = {
        "protocol": (
            "all residual S21 entity/coreference rejections; one semantic appeal "
            "contract; no runtime eligibility thresholds; gold loaded after decisions"
        ),
        "datasets": datasets,
        "aggregate": {
            "marginal_tp": tp,
            "marginal_fp": fp,
            "marginal_precision": precision,
            "macro_floor_f1": floor_f1,
            "macro_appeal_f1": final_f1,
            "macro_delta": final_f1 - floor_f1,
            "current_dynamic_f1": dynamic_f1,
            "delta_vs_dynamic": final_f1 - dynamic_f1,
        },
        "pass_gate": {
            "improves_s21": final_f1 > floor_f1,
            "marginal_precision_at_least_0_95": precision >= 0.95,
            "beats_current_dynamic": final_f1 > dynamic_f1,
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
