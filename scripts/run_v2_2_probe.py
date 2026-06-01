"""Run ONE v2.2 PROBE WAVE variant on mediastore with gpt-5.4 + write per-probe results.

Wraps run_ablation.run_variant() with the right env (LLM_BACKEND=openai,
OPENAI_MODEL_NAME=gpt-5.4) and writes:
  results/v2_2_probes/<probe_label>/<variant>_<dataset>_results.json
  results/v2_2_probes/<probe_label>/<variant>_<dataset>_summary.md (brief)

USAGE
-----
    python scripts/run_v2_2_probe.py B s_linker14_probe_b_preamble_clean
    python scripts/run_v2_2_probe.py C s_linker14_probe_c_selfrefine_clean
    python scripts/run_v2_2_probe.py D s_linker14_probe_d_upstream_clean
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


PROBE_LABELS = {
    "B": "B_preamble_rubric",
    "C": "C_selfrefine",
    "D": "D_upstream",
}

# Baseline anchors (from .planning/milestones/v2.1-phases/13-promotion-wrap/13-01-SUMMARY.md)
GPT54_MEDIASTORE_ANCHOR_F1 = 0.9677   # s_linker13_min on mediastore gpt-5.4 (closest sister)
CLAUDE_MEDIASTORE_ANCHOR_F1 = 0.9836  # s_linker13_clean on mediastore Claude (parent baseline)


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
    parser.add_argument("probe", choices=list(PROBE_LABELS.keys()),
                        help="Probe letter (B, C, or D)")
    parser.add_argument("variant", help="Variant name registered in run_ablation.VARIANT_SPECS")
    parser.add_argument("--dataset", default="mediastore")
    parser.add_argument("--backend", default="openai", choices=["openai", "claude"])
    parser.add_argument("--model", default="gpt-5.4")
    args = parser.parse_args(argv)

    _load_dotenv()
    os.environ["LLM_BACKEND"] = args.backend
    os.environ["OPENAI_MODEL_NAME"] = args.model
    if args.backend == "claude":
        os.environ["CLAUDE_MODEL"] = args.model

    probe_label = PROBE_LABELS[args.probe]
    out_dir = Path("results/v2_2_probes") / probe_label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import — env must be set first.
    import run_ablation as _ra
    from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
    from llm_sad_sam.core import DocumentLoader

    if args.variant not in _ra.VARIANT_SPECS:
        raise SystemExit(f"Unknown variant: {args.variant}")
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
    print(f"v2.2 PROBE {args.probe} — {args.variant}")
    print(f"Backend: {_ra.describe_backend_target()}")
    print(f"Dataset: {args.dataset}")
    print(f"{'=' * 80}")
    print(f"  Components: {len(components)}, Sentences: {len(sentences)}, Gold: {len(gold_pairs)}")

    t0 = time.time()
    try:
        result = _ra.run_variant(
            variant_name=args.variant,
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
            "probe": args.probe,
            "variant": args.variant,
            "dataset": args.dataset,
            "backend": args.backend,
            "model": args.model,
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": elapsed,
        }
        out_path = out_dir / f"{args.variant}_{args.dataset}_results.json"
        out_path.write_text(json.dumps(err_payload, indent=2))
        print(f"\n  [FAILED] {type(exc).__name__}: {exc}")
        raise

    elapsed = time.time() - t0
    anchor = GPT54_MEDIASTORE_ANCHOR_F1 if args.backend == "openai" else CLAUDE_MEDIASTORE_ANCHOR_F1
    delta = result["F1"] - anchor

    # Probe verdict per directive
    if delta >= 0.005:
        verdict = "STRONG_PASS"
    elif delta >= -0.010:
        verdict = "WEAK_PASS"
    else:
        verdict = "FAIL"

    payload = {
        "probe": args.probe,
        "variant": args.variant,
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
        "fp_details": result.get("fp_details", [])[:20],
        "fn_details": result.get("fn_details", [])[:20],
    }
    out_path = out_dir / f"{args.variant}_{args.dataset}_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n  -> Wrote {out_path}")
    print(f"  F1={result['F1']:.4f}  anchor={anchor:.4f}  delta={delta:+.4f}  verdict={verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
