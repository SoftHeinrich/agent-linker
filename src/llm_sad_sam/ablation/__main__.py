"""CLI entry point for the single-step ablation harness.

Usage:
    python -m llm_sad_sam.ablation single_step \
        --variant s_linker13_clean \
        --dataset mediastore \
        --phase layer1 \
        --results-dir results/ablation_results/12_02_harness \
        [--backend claude|openai|checkpoint] [--model M] [--phase-cache-dir DIR]

The subcommand structure leaves room for future commands such as
``multi_step`` or ``sweep`` without churning this module.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_sad_sam.ablation.single_step import run_single_step


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m llm_sad_sam.ablation")
    sub = parser.add_subparsers(dest="command", required=True)

    ss = sub.add_parser("single_step",
                        help="Re-run ONE phase of a variant against cached upstream "
                             "checkpoints; score against gold and write a results JSON.")
    ss.add_argument("--variant", required=True,
                    help="Variant key in VARIANT_SPECS (e.g. s_linker13_clean).")
    ss.add_argument("--dataset", required=True,
                    help="Dataset key in DATASETS (e.g. mediastore).")
    ss.add_argument("--phase", required=True,
                    choices=["layer1", "layer2", "entity_candidates",
                             "entity_decisions", "final"],
                    help="Target phase to re-execute.")
    ss.add_argument("--results-dir", required=True,
                    help="Where to write the per-run results JSON.")
    ss.add_argument("--backend", default="claude",
                    choices=["claude", "openai", "checkpoint", "codex"],
                    help="LLM backend (default: claude). Use 'checkpoint' for "
                         "no-cost replay against the cached LLM responses.")
    ss.add_argument("--model", default=None,
                    help="Optional model override (e.g. 'sonnet', 'gpt-5.4').")
    ss.add_argument("--phase-cache-dir", default=None,
                    help="Override PHASE_CACHE_DIR for upstream checkpoint root "
                         "(default: ./results/phase_cache).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "single_step":  # pragma: no cover (argparse enforces)
        parser.error(f"Unknown command: {args.command}")

    try:
        result = run_single_step(
            variant=args.variant,
            dataset=args.dataset,
            phase=args.phase,
            results_dir=Path(args.results_dir),
            backend=args.backend,
            model=args.model,
            phase_cache_dir=args.phase_cache_dir,
        )
    except KeyError as exc:
        # KeyError carries a single positional arg — surface it cleanly.
        print(f"ERROR: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4

    # Echo a compact summary line for human eyeballs.
    print(json.dumps({
        "variant": result["variant"],
        "dataset": result["dataset"],
        "phase": result["phase"],
        "F1": result["F1"],
        "delta_F1": result["delta_F1"],
        "fp": result["fp"],
        "fn": result["fn"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
