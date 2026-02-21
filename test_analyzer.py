#!/usr/bin/env python3
"""Test Phase 1 classification strategies in isolation.

Runs ONLY the ambiguity classification step (no full pipeline) on all 5 datasets,
comparing Strategy A (tight_prompt) and Strategy B (two_pass).

Usage:
    python test_analyzer.py                    # all datasets, both strategies
    python test_analyzer.py --strategy A       # only tight_prompt
    python test_analyzer.py --strategy B       # only two_pass
    python test_analyzer.py --datasets ms,ts   # subset of datasets
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
os.environ.setdefault("CLAUDE_MODEL", "sonnet")

from llm_sad_sam.core.model_analyzer import ModelAnalyzer
from llm_sad_sam.llm_client import LLMClient, LLMBackend

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)

DATASETS = {
    "ms": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
    "ts": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
    "tm": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
    "bbb": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
    "jab": BENCHMARK_BASE / "jabref/model_2023/pcm/jabref.repository",
}


def run_classification(model_path: Path, strategy: str, llm: LLMClient) -> dict:
    """Run Phase 1 classification on a single dataset."""
    analyzer = ModelAnalyzer(llm_client=llm, classify_strategy=strategy)
    components = analyzer.load_components(str(model_path))
    knowledge = analyzer.analyze(components)
    return {
        "names": sorted(c.name for c in components),
        "architectural": sorted(knowledge.architectural_names),
        "ambiguous": sorted(knowledge.ambiguous_names),
    }


def main():
    parser = argparse.ArgumentParser(description="Test Phase 1 classification strategies")
    parser.add_argument("--strategy", choices=["A", "B", "both"], default="both",
                        help="A=tight_prompt, B=two_pass, both=run both")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset keys (ms,ts,tm,bbb,jab)")
    args = parser.parse_args()

    strategies = []
    if args.strategy in ("A", "both"):
        strategies.append(("A (tight_prompt)", "tight_prompt"))
    if args.strategy in ("B", "both"):
        strategies.append(("B (two_pass)", "two_pass"))

    ds_keys = args.datasets.split(",") if args.datasets else list(DATASETS.keys())

    llm = LLMClient(backend=LLMBackend.CLAUDE)

    results = {}
    for ds_key in ds_keys:
        model_path = DATASETS.get(ds_key)
        if not model_path or not model_path.exists():
            print(f"  SKIP {ds_key}: model not found at {model_path}")
            continue

        for label, strategy in strategies:
            run_key = f"{ds_key}/{label}"
            print(f"\n{'='*60}")
            print(f"  {run_key}")
            print(f"{'='*60}")

            result = run_classification(model_path, strategy, llm)
            results[run_key] = result

            print(f"  Components: {result['names']}")
            print(f"  Architectural ({len(result['architectural'])}): {result['architectural']}")
            print(f"  Ambiguous ({len(result['ambiguous'])}): {result['ambiguous']}")

    # ── Summary table ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<8} {'Strategy':<20} {'#Arch':>6} {'#Ambig':>7}  Ambiguous names")
    print("-" * 80)
    for run_key, result in results.items():
        ds, strat = run_key.split("/", 1)
        print(f"{ds:<8} {strat:<20} {len(result['architectural']):>6} {len(result['ambiguous']):>7}  {result['ambiguous']}")

    # ── Cross-strategy comparison ──
    if len(strategies) == 2:
        print(f"\n{'='*60}")
        print("A vs B COMPARISON")
        print(f"{'='*60}")
        for ds_key in ds_keys:
            key_a = f"{ds_key}/A (tight_prompt)"
            key_b = f"{ds_key}/B (two_pass)"
            if key_a in results and key_b in results:
                a_set = set(results[key_a]["ambiguous"])
                b_set = set(results[key_b]["ambiguous"])
                common = a_set & b_set
                only_a = a_set - b_set
                only_b = b_set - a_set
                print(f"  {ds_key}: agree={sorted(common)}, only_A={sorted(only_a)}, only_B={sorted(only_b)}")


if __name__ == "__main__":
    main()
