"""Range D — Probe D on bigbluebutton (and optional Claude cross-model).

v2.2 Range tier per user directive 2026-06-01. Wraps run_ablation.run_variant
with overridden output directory + per-dataset baselines.

USAGE
-----
    # gpt-5.4 BBB
    python scripts/run_v2_2_range_d.py --dataset bigbluebutton --backend openai --model gpt-5.4
    # Claude BBB (only if STRONG_PASS gpt-5.4)
    python scripts/run_v2_2_range_d.py --dataset bigbluebutton --backend claude --model claude-sonnet-4-5

Writes results to results/v2_2_probes_range_d/<variant>_<dataset>_<backend>_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


# Baseline anchors (per user directive)
# s_linker13_min BBB gpt-5.4: 0.7636 (from Phase 13)
# s_linker13_min Claude BBB: 0.8496
BBB_GPT54_ANCHOR_F1 = 0.7636
BBB_CLAUDE_ANCHOR_F1 = 0.8496

VARIANT = "s_linker14_probe_d_upstream_clean"


def _load_dotenv() -> None:
    env_file = _ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="bigbluebutton")
    parser.add_argument("--backend", default="openai", choices=["openai", "claude"])
    parser.add_argument("--model", default="gpt-5.4")
    args = parser.parse_args(argv)

    _load_dotenv()
    os.environ["LLM_BACKEND"] = args.backend
    os.environ["OPENAI_MODEL_NAME"] = args.model
    if args.backend == "claude":
        os.environ["CLAUDE_MODEL"] = args.model

    out_dir = Path(
        os.environ.get("RANGE_D_OUT_DIR", "results/v2_2_probes_range_d")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import — env must be set first.
    import run_ablation as _ra
    from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
    from llm_sad_sam.core import DocumentLoader

    if VARIANT not in _ra.VARIANT_SPECS:
        raise SystemExit(f"Unknown variant: {VARIANT}")
    if args.dataset not in _ra.DATASETS:
        raise SystemExit(f"Unknown dataset: {args.dataset}")

    paths = _ra.DATASETS[args.dataset]
    _ra.require_existing(paths["text"], f"{args.dataset} text")
    _ra.require_existing(paths["model"], f"{args.dataset} model")
    _ra.require_existing(paths["gold_sam"], f"{args.dataset} gold standard")

    components = parse_pcm_repository(str(paths["model"]))
    id_to_name = {c.id: c.name for c in components}
    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    sent_map = {s.number: s for s in sentences}
    gold_pairs = _ra.load_gold_sam(str(paths["gold_sam"]))
    transarc_pairs = (
        _ra.load_transarc_pairs(str(paths["transarc_sam"]))
        if paths["transarc_sam"].exists()
        else set()
    )

    print(f"\n{'=' * 80}")
    print(f"v2.2 RANGE D — {VARIANT}")
    print(f"Backend: {_ra.describe_backend_target()}")
    print(f"Dataset: {args.dataset}")
    print(f"{'=' * 80}")
    print(f"  Components: {len(components)}, Sentences: {len(sentences)}, Gold: {len(gold_pairs)}")

    t0 = time.time()
    try:
        result = _ra.run_variant(
            variant_name=VARIANT,
            dataset_name=args.dataset,
            paths=paths,
            gold_pairs=gold_pairs,
            transarc_pairs=transarc_pairs,
            id_to_name=id_to_name,
            sent_map=sent_map,
            results_dir=out_dir,
        )
    except Exception as exc:
        elapsed = time.time() - t0
        err_payload = {
            "variant": VARIANT,
            "dataset": args.dataset,
            "backend": args.backend,
            "model": args.model,
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": elapsed,
        }
        out_path = out_dir / f"{VARIANT}_{args.dataset}_{args.backend}_results.json"
        out_path.write_text(json.dumps(err_payload, indent=2))
        print(f"\n  [FAILED] {type(exc).__name__}: {exc}")
        raise

    elapsed = time.time() - t0

    # Per-dataset, per-backend anchor
    if args.dataset == "bigbluebutton" and args.backend == "openai":
        anchor = BBB_GPT54_ANCHOR_F1
    elif args.dataset == "bigbluebutton" and args.backend == "claude":
        anchor = BBB_CLAUDE_ANCHOR_F1
    else:
        anchor = None

    delta = (result["F1"] - anchor) if anchor is not None else None

    # Range D gate per user directive:
    #   F1 >= baseline + 0.005 -> STRONG_PASS
    #   F1 >= baseline - 0.010 -> WEAK_PASS
    #   F1 <  baseline - 0.010 -> FAIL
    if delta is None:
        verdict = "N/A"
    elif delta >= 0.005:
        verdict = "STRONG_PASS"
    elif delta >= -0.010:
        verdict = "WEAK_PASS"
    else:
        verdict = "FAIL"

    payload = {
        "phase": "v2.2-RANGE-D",
        "variant": VARIANT,
        "dataset": args.dataset,
        "backend": args.backend,
        "model": args.model,
        "F1": result["F1"],
        "P": result["P"],
        "R": result["R"],
        "tp": result["tp"],
        "fp": result["fp"],
        "fn": result["fn"],
        "anchor_F1": anchor,
        "delta_F1_vs_anchor": delta,
        "verdict": verdict,
        "elapsed_s": elapsed,
        "sources": result.get("sources", {}),
        "fp_by_source": result.get("fp_by_source", {}),
        "fp_count": result["fp"],
        "fn_count": result["fn"],
        "fp_details": result.get("fp_details", [])[:30],
        "fn_details": result.get("fn_details", [])[:30],
    }
    out_path = out_dir / f"{VARIANT}_{args.dataset}_{args.backend}_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  -> Wrote {out_path}")
    if anchor is not None:
        print(f"  F1={result['F1']:.4f}  anchor={anchor:.4f}  delta={delta:+.4f}  verdict={verdict}")
    else:
        print(f"  F1={result['F1']:.4f}  (no anchor configured for this dataset/backend)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
