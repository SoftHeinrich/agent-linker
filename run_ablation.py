#!/usr/bin/env python3
"""Lightweight ablation runner for the retained ILinker and S-Linker families."""

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
    "i1",
    "i2",
    "i3",
    "s_linker",
    "s_linker2",
    "s_linker3",
    "s_linker4",
    "s_linker5",
    "s_linker6",
    "s_linker7",
    "s_linker7a",
    "s_linker7b",
    "s_linker8",
    "s_linker9",
    "s_linker9a",
    "s_linker9b",
    "s_linker9c",
    "s_linker9d",
    "s_linker9e",
    "s_linker10",
    "s_linker10a",
    "s_linker11",
    "s_linker11a",
    "s_linker11b",
    "s_linker11c",
    "s_linker11d",
    "s_linker11e",
    "s_linker12a",
    "s_linker12b",
    "s_linker12c",
    "s_linker12d",
    "s_linker12e",
    "s_linker13a",
]

VARIANT_SPECS = {
    "i1": dict(
        aliases=("ilinker1",),
        module="llm_sad_sam.linkers.experimental.ilinker1",
        class_name="ILinker1",
        description="ILinker1 three-pass precision cascade",
    ),
    "i2": dict(
        aliases=("ilinker2",),
        module="llm_sad_sam.linkers.experimental.ilinker2",
        class_name="ILinker2",
        description="ILinker2 two-pass explicit extractor",
    ),
    "i3": dict(
        aliases=("ilinker3",),
        adapter="ilinker3",
        description="ILinker3 v2-stack extractor adapter",
    ),
    "s_linker": dict(
        aliases=("s_linker1",),
        module="llm_sad_sam.linkers.experimental.s_linker",
        class_name="SLinker",
        description="S-Linker base DAG pipeline",
    ),
    "s_linker2": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker2",
        class_name="SLinker2",
        description="S-Linker2",
    ),
    "s_linker3": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker3",
        class_name="SLinker3",
        description="S-Linker3",
    ),
    "s_linker4": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker4",
        class_name="SLinker4",
        description="S-Linker4",
    ),
    "s_linker5": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker5",
        class_name="SLinker5",
        description="S-Linker5",
    ),
    "s_linker6": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker6",
        class_name="SLinker6",
        description="S-Linker6",
    ),
    "s_linker7": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker7",
        class_name="SLinker7",
        description="S-Linker7",
    ),
    "s_linker7a": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker7a",
        class_name="SLinker7a",
        description="S-Linker7a",
    ),
    "s_linker7b": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker7b",
        class_name="SLinker7b",
        description="S-Linker7b",
    ),
    "s_linker8": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker8",
        class_name="SLinker8",
        description="S-Linker8",
    ),
    "s_linker9": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker9",
        class_name="SLinker9",
        description="S-Linker9",
    ),
    "s_linker9a": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker9a",
        class_name="SLinker9a",
        description="S-Linker9a",
    ),
    "s_linker9b": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker9b",
        class_name="SLinker9b",
        description="S-Linker9b",
    ),
    "s_linker9c": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker9c",
        class_name="SLinker9c",
        description="S-Linker9c",
    ),
    "s_linker9d": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker9d",
        class_name="SLinker9d",
        description="S-Linker9d",
    ),
    "s_linker9e": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker9e",
        class_name="SLinker9e",
        description="S-Linker9e",
    ),
    "s_linker10": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker10",
        class_name="SLinker10",
        description="S-Linker10",
    ),
    "s_linker10a": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker10a",
        class_name="SLinker10a",
        description="S-Linker10a",
    ),
    "s_linker11": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker11",
        class_name="SLinker11",
        description="S-Linker11",
    ),
    "s_linker11a": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker11a",
        class_name="SLinker11a",
        description="S-Linker11a",
    ),
    "s_linker11b": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker11b",
        class_name="SLinker11b",
        description="S-Linker11b: alias stratification (strong global / weak local)",
    ),
    "s_linker11c": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker11c",
        class_name="SLinker11c",
        description="S-Linker11c: evidence bundles + structured debate on rejects",
    ),
    "s_linker11d": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker11d",
        class_name="SLinker11d",
        description="S-Linker11d: no partial injection (ablation)",
    ),
    "s_linker11e": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker11e",
        class_name="SLinker11e",
        description="S-Linker11e: evidence bundles in validation, no debate",
    ),
    "s_linker12a": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker12a",
        class_name="SLinker12a",
        description="S-Linker12a: alias stratification + no partial injection",
    ),
    "s_linker12b": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker12b",
        class_name="SLinker12b",
        description="S-Linker12b: alias stratification + evidence bundles (ICSE)",
    ),
    "s_linker12c": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker12c",
        class_name="SLinker12c",
        description="S-Linker12c: 12b - dead Tier 2, intersection voting",
    ),
    "s_linker12d": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker12d",
        class_name="SLinker12d",
        description="S-Linker12d: 12c + trailing-word enrichment (separate step)",
    ),
    "s_linker12e": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker12e",
        class_name="SLinker12e",
        description="S-Linker12e: 12c + merged trailing-word enrichment",
    ),
    "s_linker13a": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker13a",
        class_name="SLinker13a",
        description="S-Linker13a: 12c - _split_component_name (Spike 001 LLM trailing-word)",
    ),
}

VARIANTS = {
    canonical: {"canonical": canonical, "description": VARIANT_SPECS[canonical]["description"]}
    for canonical in CANONICAL_VARIANTS
}
for canonical, spec in VARIANT_SPECS.items():
    for alias in spec["aliases"]:
        VARIANTS[alias] = {"canonical": canonical, "description": f"Alias for {canonical}"}

BENCHMARK_BASE = ROOT / "../ardoco/core/tests-base/src/main/resources/benchmark"
CLI_RESULTS = Path("/mnt/hostshare/ardoco-home/cli-results")

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


class ILinker3Adapter:
    """Expose ILinker3's extract API via the runner's link interface."""

    def __init__(self, backend: LLMBackend | None = None):
        from llm_sad_sam.linkers.experimental.ilinker3 import ILinker3

        self.llm = LLMClient(backend=backend or get_backend())
        self._extractor = ILinker3(llm=self.llm)

    def link(self, text_path: str, model_path: str, transarc_csv: str | None = None):
        del transarc_csv
        from llm_sad_sam.core.document_loader_v2 import load_sentences
        from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository as parse_pcm_repository_v2

        sentences = load_sentences(text_path)
        components = parse_pcm_repository_v2(model_path)
        return self._extractor.extract(sentences, components)


def get_backend() -> LLMBackend:
    backend_name = os.environ.get("LLM_BACKEND", "claude").strip().lower()
    if backend_name == "openai":
        return LLMBackend.OPENAI
    if backend_name == "checkpoint":
        return LLMBackend.CHECKPOINT
    if backend_name == "codex":
        return LLMBackend.CODEX
    return LLMBackend.CLAUDE


os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.2")
os.environ.setdefault("CLAUDE_MODEL", "sonnet")


def describe_backend_target(backend: LLMBackend | None = None) -> str:
    backend = backend or get_backend()
    if backend == LLMBackend.CLAUDE:
        return f"claude ({os.environ.get('CLAUDE_MODEL', 'sonnet')})"
    if backend == LLMBackend.OPENAI:
        return f"openai ({os.environ.get('OPENAI_MODEL_NAME', 'gpt-5.2')})"
    if backend == LLMBackend.CHECKPOINT:
        fallback_model = os.environ.get("CHECKPOINT_FALLBACK_MODEL", "").strip().lower()
        if fallback_model in {"gpt", "openai"} or fallback_model.startswith("gpt"):
            model = os.environ.get("OPENAI_MODEL_NAME", "gpt-5.2")
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
            return f"checkpoint -> openai ({os.environ.get('OPENAI_MODEL_NAME', 'gpt-5.2')})"
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
    if canonical == "i3":
        return ILinker3Adapter(backend=backend or get_backend())

    spec = VARIANT_SPECS[canonical]
    module = importlib.import_module(spec["module"])
    cls = getattr(module, spec["class_name"])
    return cls(backend=backend or get_backend())


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
    return {"tp": tp, "fp": fp, "fn": fn, "P": precision, "R": recall, "F1": f1}


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
        f"  {variant_name}: P={metrics['P']:.1%} R={metrics['R']:.1%} F1={metrics['F1']:.1%} "
        f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} ({elapsed:.0f}s)"
    )
    print(f"    Sources: {dict(source_counts)}")
    print(f"    FP by source: {dict(fp_by_source)}")

    return {
        "variant": variant_name,
        "P": metrics["P"],
        "R": metrics["R"],
        "F1": metrics["F1"],
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


def print_summary(all_results: dict[str, dict[str, dict[str, object]]], selected_variants: list[str]) -> None:
    print(f"\n{'=' * 120}")
    print("SUMMARY")
    print(f"{'=' * 120}")
    header = f"{'Dataset':<16}"
    for variant in selected_variants:
        header += f" | {variant:^18}"
    print(header)
    print(f"{'-' * 16}" + ("-+-" + "-" * 18) * len(selected_variants))

    for dataset_name, dataset_results in all_results.items():
        row = f"{dataset_name:<16}"
        for variant in selected_variants:
            result = dataset_results.get(variant)
            if result is None:
                row += " | " + f"{'--':^18}"
            else:
                row += " | " + f"F1 {result['F1']:.1%} FP {result['fp']:>3}"
        print(row)

    print(f"{'-' * 16}" + ("-+-" + "-" * 18) * len(selected_variants))
    row = f"{'Macro avg':<16}"
    for variant in selected_variants:
        values = [all_results[dataset][variant] for dataset in all_results if variant in all_results[dataset]]
        avg_f1 = sum(value["F1"] for value in values) / len(values)
        total_fp = sum(value["fp"] for value in values)
        row += " | " + f"F1 {avg_f1:.1%} FP {total_fp:>3}"
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
        default=["s_linker11a"],
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
    print("ABLATION STUDY: Retained ILinker and S-Linker Variants")
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
