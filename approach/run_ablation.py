#!/usr/bin/env python3
"""Ablation runner for the s126 paper arm (and its RQ4 no-knowledge sibling).

This is the s126-only consolidation of what used to be a 437-entry variant
registry spanning every explored linker family (i1/i2, s_linker through
s_linker126, the router, the S23 verification family, and more). Only the
paper arm and its RQ4 ablation remain registered here; the full registry and
every archived module are preserved on
origin/archive/master-pre-s126-consolidation.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))


def load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


load_dotenv()

from llm_sad_sam.core import DocumentLoader, SadSamLink
from llm_sad_sam.llm_client import LLMBackend, LLMClient
from llm_sad_sam.pcm_parser import parse_pcm_repository


CANONICAL_VARIANTS = [
    "s_linker126",  # PAPER ARM: s125 + greedy unambiguous names + exact-antecedent contract
    "s_linker126_noknow",  # RQ4 knowledge A/B for the greedy-merge arm: s126, alias table off
]

VARIANT_SPECS = {
    "s_linker126": dict(
        aliases=("antecedentrule", "greedymerge"),
        module="llm_sad_sam.linkers.experimental.s_linker126",
        class_name="SLinker126",
        description=(
            "S-Linker126 - s_linker125 (no antecedent shortlist) plus greedy whole-name "
            "span ownership, discard of each residual multi-component surface, and the "
            "exact-antecedent contract. The greedy scan and discard replace the "
            "competitors evidence field. A six-run fixed-input audit "
            "removed 72 candidates, 0 gold and one cached false positive; only BBB "
            "S27/S31 remained ambiguous and were discarded, both non-gold. The contract "
            "is the one the "
            "shortlist's paragraph used to ASSERT: a coreference resolution whose cited "
            "antecedent sentence does not write the component's name as a name is not "
            "put to the judge. The prompt used to claim the list `is where the "
            "antecedent will be if there is one` and could not enforce it; over six "
            "recorded runs 17 surviving false positives cite an antecedent that writes "
            "the name only inside a dotted identifier (10) or not at all (7), against "
            "ZERO gold, on both models. The predicate reads `_written_as`, already "
            "computed for every union-judging case, so no new computation and no new "
            "concept is introduced; `_named_before` is dead once the list goes and "
            "`_states_a_name` keeps its other caller. It may close a case where "
            "s_linker124's mark may not, because `written` is a GIVEN fact (catalog plus "
            "document, identical every run) and a judge's verdict is a DISCOVERED one - "
            "s_linker109's rule, and its nesting predicate is the precedent. The refusal "
            "measured with the list still present costs ZERO true positives on both "
            "models (terra FP 13.0 -> 12.3, luna 40.3 -> 38.3 with -2 in 3 of 3 samples). "
            "The name-stage change can starve coreference, so promotion requires paired "
            "end-to-end evaluation (pilot/test_s126.py; the deterministic audit that "
            "sized the greedy-merge removal is archived, see approach/CLAUDE.md). "
            "Promoted to the paper arm on 2026-09-16 "
            "by explicit author decision on simplicity, overriding the component-weighted "
            "doc-code gate's refusal; see approach/CLAUDE.md for the full ledger."),
        canonical=True,
        experimental=False,
    ),
    "s_linker126_noknow": dict(
        aliases=("greedymergenoknow",),
        module="llm_sad_sam.linkers.experimental.s_linker126",
        class_name="SLinker126",
        description=(
            "S-Linker126 NO-KNOWLEDGE - RQ4's knowledge A/B on the greedy-merge arm "
            "(experimental=True, NOT canonical). s_linker126 with no_knowledge=True: the "
            "document-alias stage is skipped and an empty DocumentKnowledge is set "
            "directly, so the name scan sees canonical component names only and the "
            "union's `writes` evidence can never read `alias`. All other phases run "
            "unchanged, including the greedy ownership scan, the ambiguous-surface "
            "discard, and the antecedent-form predicate. Mirrors s_linker120_noknow one "
            "arm over. "
            "LANDMINE: _VARIANT_NAME stays 's_linker126', so its phase states nest under "
            "phase_states/s_linker126/ -- give every run its own PHASE_CACHE_DIR or it "
            "clobbers the Full arm's states."
        ),
        canonical=False,
        experimental=True,
        kwargs=dict(no_knowledge=True),
    ),
}

VARIANTS = {
    canonical: {"canonical": canonical, "description": VARIANT_SPECS[canonical]["description"]}
    for canonical in CANONICAL_VARIANTS
}
for canonical, spec in VARIANT_SPECS.items():
    for alias in spec["aliases"]:
        VARIANTS[alias] = {"canonical": canonical, "description": f"Alias for {canonical}"}

# The replication package vendors the benchmark at its top level.  The
# environment variables preserve support for an external ARDoCo checkout and
# externally generated CLI baseline results.
BENCHMARK_BASE = Path(os.environ.get("ALINKER_BENCHMARK", ROOT.parent / "benchmark"))
CLI_RESULTS = Path(os.environ.get("ALINKER_CLI_RESULTS", ROOT.parent / "cli-results"))

DATASETS = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
        "gold_sam": BENCHMARK_BASE / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
        "transarc_sam": CLI_RESULTS / "mediastore-sad-sam/sadSamTlr_mediastore.csv",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
        "model": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
        "gold_sam": BENCHMARK_BASE / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
        "transarc_sam": CLI_RESULTS / "teastore-sad-sam/sadSamTlr_teastore.csv",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
        "gold_sam": BENCHMARK_BASE / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc_sam": CLI_RESULTS / "teammates-sad-sam/sadSamTlr_teammates.csv",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
        "gold_sam": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc_sam": CLI_RESULTS / "bigbluebutton-sad-sam/sadSamTlr_bigbluebutton.csv",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
        "gold_sam": BENCHMARK_BASE / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc_sam": CLI_RESULTS / "jabref-sad-sam/sadSamTlr_jabref.csv",
    },
}

# Extra datasets for out-of-benchmark cases (studies/oss-scale): a JSON file
# mapping name -> {text, model, gold_sam[, transarc_sam]} with paths relative to
# the JSON file.  Purely additive; the five benchmark entries above are untouched.
_EXTRA = os.environ.get("ALINKER_EXTRA_DATASETS")
if _EXTRA:
    _extra_path = Path(_EXTRA).resolve()
    with open(_extra_path) as _handle:
        for _name, _spec in json.load(_handle).items():
            DATASETS[_name] = {
                key: _extra_path.parent / value for key, value in _spec.items()
            }
            DATASETS[_name].setdefault("transarc_sam", _extra_path.parent / "no-transarc.csv")


def get_backend() -> LLMBackend:
    backend_name = os.environ.get("LLM_BACKEND", "openai").strip().lower()
    if backend_name == "openai":
        return LLMBackend.OPENAI
    if backend_name == "checkpoint":
        return LLMBackend.CHECKPOINT
    if backend_name == "codex":
        return LLMBackend.CODEX
    return LLMBackend.CLAUDE


os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.6-terra")
os.environ.setdefault("CLAUDE_MODEL", "sonnet")


def describe_backend_target(backend: LLMBackend | None = None) -> str:
    backend = backend or get_backend()
    if backend == LLMBackend.CLAUDE:
        return f"claude ({os.environ.get('CLAUDE_MODEL', 'sonnet')})"
    if backend == LLMBackend.OPENAI:
        return f"openai ({os.environ.get('OPENAI_MODEL_NAME', 'gpt-5.6-terra')})"
    if backend == LLMBackend.CHECKPOINT:
        fallback_model = os.environ.get("CHECKPOINT_FALLBACK_MODEL", "").strip().lower()
        if fallback_model in {"gpt", "openai"} or fallback_model.startswith("gpt"):
            model = os.environ.get("OPENAI_MODEL_NAME", "gpt-5.6-terra")
            if fallback_model.startswith("gpt"):
                model = fallback_model
            return f"checkpoint -> openai ({model})"
        if fallback_model in {"claude", "sonnet"} or fallback_model.startswith("claude"):
            model = os.environ.get("CLAUDE_MODEL", "sonnet")
            if fallback_model not in {"claude", "sonnet"}:
                model = fallback_model
            return f"checkpoint -> claude ({model})"
        fallback_backend = os.environ.get("CHECKPOINT_FALLBACK", "claude").strip().lower() or "claude"
        if fallback_backend == "openai":
            return f"checkpoint -> openai ({os.environ.get('OPENAI_MODEL_NAME', 'gpt-5.6-terra')})"
        if fallback_backend == "codex":
            return "checkpoint -> codex"
        return f"checkpoint -> claude ({os.environ.get('CLAUDE_MODEL', 'sonnet')})"
    return backend.value


def available_variants() -> list[str]:
    return list(CANONICAL_VARIANTS)


def canonical_variant(name: str) -> str:
    if name not in VARIANTS:
        raise KeyError(name)
    return VARIANTS[name]["canonical"]


def normalize_variants(names: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for name in names:
        canonical = canonical_variant(name)
        if canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized


def build_linker(variant_name: str, backend: LLMBackend | None = None):
    canonical = canonical_variant(variant_name)
    spec = VARIANT_SPECS[canonical]
    module = importlib.import_module(spec["module"])
    cls = getattr(module, spec["class_name"])
    extra = spec.get("kwargs", {})
    return cls(backend=backend or get_backend(), **extra)


def load_gold_sam(gold_path: str) -> set[tuple[int, str]]:
    links: set[tuple[int, str]] = set()
    with open(gold_path) as handle:
        for row in csv.DictReader(handle):
            component_id = row.get("modelElementID", "").strip()
            sentence_number = row.get("sentence", "").strip()
            if component_id and sentence_number:
                links.add((int(sentence_number), component_id))
    return links


def load_transarc_pairs(transarc_path: str) -> set[tuple[int, str]]:
    pairs: set[tuple[int, str]] = set()
    with open(transarc_path) as handle:
        for row in csv.DictReader(handle):
            component_id = row.get("modelElementID", "").strip()
            sentence_number = row.get("sentence", "").strip()
            if component_id and sentence_number:
                pairs.add((int(sentence_number), component_id))
    return pairs


def export_links_csv(links: list[SadSamLink], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sentence", "component_id", "component_name", "confidence", "source"])
        for link in sorted(links, key=lambda item: (item.sentence_number, item.component_id)):
            writer.writerow(
                [
                    link.sentence_number,
                    link.component_id,
                    link.component_name,
                    f"{link.confidence:.2f}",
                    link.source,
                ]
            )


def eval_metrics(predicted: set[tuple[int, str]], gold: set[tuple[int, str]]) -> dict[str, float]:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    f2 = 5 * tp / (5 * tp + 4 * fn + fp) if (tp + fp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "P": precision,
        "R": recall,
        "F1": f1,
        "F2": f2,
    }


def require_existing(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def run_variant(
    variant_name: str,
    dataset_name: str,
    paths: dict[str, Path],
    gold_pairs: set[tuple[int, str]],
    transarc_pairs: set[tuple[int, str]],
    id_to_name: dict[str, str],
    sent_map: dict[int, object],
    results_dir: Path,
) -> dict[str, object]:
    print(f"\n  --- Variant: {variant_name} ---")
    linker = build_linker(variant_name)

    t0 = time.time()
    predictions = linker.link(
        text_path=str(paths["text"]),
        model_path=str(paths["model"]),
        transarc_csv=str(paths["transarc_sam"]),
    )
    elapsed = time.time() - t0

    predicted_pairs = {(link.sentence_number, link.component_id) for link in predictions}
    prediction_by_key = {(link.sentence_number, link.component_id): link for link in predictions}
    metrics = eval_metrics(predicted_pairs, gold_pairs)

    source_counts: defaultdict[str, int] = defaultdict(int)
    for link in predictions:
        source_counts[link.source] += 1

    fp_pairs = predicted_pairs - gold_pairs
    fp_by_source: defaultdict[str, int] = defaultdict(int)
    fp_details = []
    for sentence_number, component_id in sorted(fp_pairs):
        link = prediction_by_key[(sentence_number, component_id)]
        fp_by_source[link.source] += 1
        sentence = sent_map.get(sentence_number)
        fp_details.append(
            {
                "sentence": sentence_number,
                "component": id_to_name.get(component_id, component_id),
                "source": link.source,
                "confidence": link.confidence,
                "text": sentence.text[:120] if sentence else "",
            }
        )

    fn_pairs = gold_pairs - predicted_pairs
    fn_details = []
    for sentence_number, component_id in sorted(fn_pairs):
        sentence = sent_map.get(sentence_number)
        component_name = id_to_name.get(component_id, component_id)
        fn_details.append(
            {
                "sentence": sentence_number,
                "component": component_name,
                "name_in_text": component_name.lower() in sentence.text.lower() if sentence else False,
                "transarc_had": (sentence_number, component_id) in transarc_pairs,
            }
        )

    export_links_csv(predictions, results_dir / f"{variant_name}_{dataset_name}_links.csv")

    print(
        f"  {variant_name}: P={metrics['P']:.1%} R={metrics['R']:.1%} "
        f"F1={metrics['F1']:.1%} F2={metrics['F2']:.1%} "
        f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} ({elapsed:.0f}s)"
    )
    print(f"    Sources: {dict(source_counts)}")
    print(f"    FP by source: {dict(fp_by_source)}")

    result = {
        "variant": variant_name,
        "P": metrics["P"],
        "R": metrics["R"],
        "F1": metrics["F1"],
        "F2": metrics["F2"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "n_links": len(predictions),
        "time": elapsed,
        "sources": dict(source_counts),
        "fp_by_source": dict(fp_by_source),
        "fp_details": fp_details,
        "fn_details": fn_details,
    }
    if hasattr(linker, "orchestrator_workflow"):
        result["workflow"] = linker.orchestrator_workflow
    if hasattr(linker, "_llm_calls"):
        result["llm_calls"] = len(linker._llm_calls)
    return result


def print_summary(all_results: dict[str, dict[str, dict[str, object]]], selected_variants: list[str]) -> None:
    print(f"\n{'=' * 120}")
    print("SUMMARY")
    print(f"{'=' * 120}")
    header = f"{'Dataset':<16}"
    for variant in selected_variants:
        header += f" | {variant:^27}"
    print(header)
    print(f"{'-' * 16}" + ("-+-" + "-" * 27) * len(selected_variants))

    for dataset_name, dataset_results in all_results.items():
        row = f"{dataset_name:<16}"
        for variant in selected_variants:
            result = dataset_results.get(variant)
            if result is None:
                row += " | " + f"{'--':^27}"
            else:
                row += " | " + (
                    f"F1 {result['F1']:.1%} F2 {result['F2']:.1%} "
                    f"FP {result['fp']:>3}"
                )
        print(row)

    print(f"{'-' * 16}" + ("-+-" + "-" * 27) * len(selected_variants))
    row = f"{'Macro avg':<16}"
    for variant in selected_variants:
        values = [all_results[dataset][variant] for dataset in all_results if variant in all_results[dataset]]
        avg_f1 = sum(value["F1"] for value in values) / len(values)
        avg_f2 = sum(value["F2"] for value in values) / len(values)
        total_fp = sum(value["fp"] for value in values)
        row += " | " + f"F1 {avg_f1:.1%} F2 {avg_f2:.1%} FP {total_fp:>3}"
    print(row)

    row = f"{'Pooled':<16}"
    for variant in selected_variants:
        values = [
            all_results[dataset][variant]
            for dataset in all_results
            if variant in all_results[dataset]
        ]
        tp = sum(value["tp"] for value in values)
        fp = sum(value["fp"] for value in values)
        fn = sum(value["fn"] for value in values)
        pooled_f1 = 2 * tp / (2 * tp + fp + fn)
        pooled_f2 = 5 * tp / (5 * tp + 4 * fn + fp)
        row += " | " + f"F1 {pooled_f1:.1%} F2 {pooled_f2:.1%} FP {fp:>3}"
    print(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["s_linker126"],
        help="Retained variants to evaluate",
    )
    parser.add_argument(
        "--results-dir",
        default="results/ablation_results",
        help="Directory for CSV and JSON output",
    )
    parser.add_argument("--list-datasets", action="store_true", help="Print supported datasets and exit")
    parser.add_argument("--list-variants", action="store_true", help="Print supported variants and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_datasets:
        print("\n".join(DATASETS.keys()))
        return 0
    if args.list_variants:
        print("\n".join(available_variants()))
        return 0

    unknown_datasets = [name for name in args.datasets if name not in DATASETS]
    if unknown_datasets:
        raise SystemExit(f"Unknown datasets: {', '.join(unknown_datasets)}")

    try:
        selected_variants = normalize_variants(args.variants)
    except KeyError as exc:
        raise SystemExit(f"Unknown variant: {exc.args[0]}") from exc

    datasets = {name: DATASETS[name] for name in args.datasets}
    backend = get_backend()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 120}")
    print("ABLATION STUDY: s126 (paper arm)")
    print(f"Backend: {describe_backend_target(backend)}")
    print(f"Datasets: {', '.join(datasets.keys())}")
    print(f"Variants: {', '.join(selected_variants)}")
    print(f"{'=' * 120}")

    all_results: dict[str, dict[str, dict[str, object]]] = {}

    for dataset_name, paths in datasets.items():
        require_existing(paths["text"], f"{dataset_name} text")
        require_existing(paths["model"], f"{dataset_name} model")
        require_existing(paths["gold_sam"], f"{dataset_name} gold standard")

        print(f"\n{'=' * 120}")
        print(f"DATASET: {dataset_name}")
        print(f"{'=' * 120}")

        components = parse_pcm_repository(str(paths["model"]))
        id_to_name = {component.id: component.name for component in components}
        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        sent_map = {sentence.number: sentence for sentence in sentences}
        gold_pairs = load_gold_sam(str(paths["gold_sam"]))
        transarc_pairs = (
            load_transarc_pairs(str(paths["transarc_sam"]))
            if paths["transarc_sam"].exists()
            else set()
        )

        print(f"  Components: {len(components)}, Sentences: {len(sentences)}")
        print(f"  Gold links: {len(gold_pairs)}, TransArc baseline: {len(transarc_pairs)}")
        if transarc_pairs:
            metrics = eval_metrics(transarc_pairs, gold_pairs)
            print(f"  TransArc baseline: P={metrics['P']:.1%} R={metrics['R']:.1%} F1={metrics['F1']:.1%}")
        else:
            print("  TransArc baseline: (CSV not available)")

        all_results[dataset_name] = {}
        for variant_name in selected_variants:
            result = run_variant(
                variant_name=variant_name,
                dataset_name=dataset_name,
                paths=paths,
                gold_pairs=gold_pairs,
                transarc_pairs=transarc_pairs,
                id_to_name=id_to_name,
                sent_map=sent_map,
                results_dir=results_dir,
            )
            all_results[dataset_name][variant_name] = result

    print_summary(all_results, selected_variants)

    json_path = results_dir / f"ablation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with json_path.open("w") as handle:
        json.dump(all_results, handle, indent=2, default=str)
    print(f"\nResults saved to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
