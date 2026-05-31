#!/usr/bin/env python3
"""EXT-01 Plan 06-09 Probe P3 — Pure removal (BBB only).

Per CONTEXT.md D-13:
  Monkeypatch `SLinker13._has_standalone_mention` to a lambda returning
  True for every call. P3 measures the BBB F1 floor when no
  standalone-mention gate exists; downstream tiers handle everything
  alone.

Also writes `baseline_fn_set.json` (and `baseline_fp_set.json`) for the
unpatched s_linker13 BBB baseline so Tasks 1, 2 can compute
"FN-recovered" / "new-FPs-introduced" metrics against a shared
reference. If a prior run already wrote those, the unpatched sweep is
skipped.

LLM cost for the gate itself: zero. (Regular pipeline LLM calls still
happen — Tier 1, Tier 2.)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_dotenv() -> None:
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


load_dotenv()
os.environ.setdefault("CLAUDE_MODEL", "sonnet")

from llm_sad_sam.core.document_loader_v2 import load_sentences  # noqa: E402
from llm_sad_sam.llm_client import LLMBackend  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker13 import SLinker13  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402


BENCHMARK_BASE = (REPO_ROOT / "../ardoco/core/tests-base/src/main/resources/benchmark").resolve()
BBB = {
    "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
    "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
    "gold_sam": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
}

PROBE_NAME = "ext01_probe_p3"
RESULTS_DIR = REPO_ROOT / "results/ablation_results/ext01_probes"
PHASE_CACHE_DIR_PATCHED = REPO_ROOT / "results/phase_cache" / PROBE_NAME
PHASE_CACHE_DIR_BASELINE = REPO_ROOT / "results/phase_cache" / "ext01_probe_baseline"


def load_gold_pairs(gold_path: str):
    import csv
    pairs = set()
    with open(gold_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                pairs.add((int(snum), cid))
    return pairs


def eval_metrics(predicted, gold):
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "P": precision, "R": recall, "F1": f1}


def run_unpatched_baseline(components, sentences, gold_pairs):
    """Run unpatched SLinker13 on BBB; write baseline_fn_set.json + baseline_fp_set.json."""
    print(f"[P3] Running UNPATCHED SLinker13 baseline on BBB ...")
    os.environ["PHASE_CACHE_DIR"] = str(PHASE_CACHE_DIR_BASELINE)
    PHASE_CACHE_DIR_BASELINE.mkdir(parents=True, exist_ok=True)
    linker = SLinker13(backend=LLMBackend.CLAUDE)
    t0 = time.time()
    predictions = linker.link(
        text_path=str(BBB["text"]),
        model_path=str(BBB["model"]),
    )
    elapsed = time.time() - t0
    pred_pairs = {(p.sentence_number, p.component_id) for p in predictions}
    metrics = eval_metrics(pred_pairs, gold_pairs)
    print(f"[P3] Baseline: P={metrics['P']:.4f} R={metrics['R']:.4f} "
          f"F1={metrics['F1']:.4f} TP={metrics['tp']} FP={metrics['fp']} "
          f"FN={metrics['fn']} time={elapsed:.0f}s")
    fn_set = sorted([list(it) for it in (gold_pairs - pred_pairs)])
    fp_set = sorted([list(it) for it in (pred_pairs - gold_pairs)])
    fn_blob = {
        "dataset": "bigbluebutton",
        "fn_set": fn_set,
        "fn_count": len(fn_set),
        "baseline_f1": metrics["F1"],
        "baseline_p": metrics["P"],
        "baseline_r": metrics["R"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_s": round(elapsed, 1),
    }
    fp_blob = {
        "dataset": "bigbluebutton",
        "fp_set": fp_set,
        "fp_count": len(fp_set),
        "baseline_f1": metrics["F1"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (RESULTS_DIR / "baseline_fn_set.json").write_text(json.dumps(fn_blob, indent=2))
    (RESULTS_DIR / "baseline_fp_set.json").write_text(json.dumps(fp_blob, indent=2))
    print(f"[P3] Wrote baseline_fn_set.json ({len(fn_set)} FNs) and "
          f"baseline_fp_set.json ({len(fp_set)} FPs).")
    return metrics


def main() -> int:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE_CACHE_DIR_PATCHED.mkdir(parents=True, exist_ok=True)

    for k, p in BBB.items():
        if not Path(p).exists():
            raise FileNotFoundError(f"BBB {k} missing: {p}")

    components = parse_pcm_repository(str(BBB["model"]))
    sentences = load_sentences(str(BBB["text"]))
    gold_pairs = load_gold_pairs(str(BBB["gold_sam"]))
    print(f"[P3] components={len(components)} sentences={len(sentences)} "
          f"gold_pairs={len(gold_pairs)}")

    # Step 1: unpatched baseline (only if not already on disk).
    baseline_fn_path = RESULTS_DIR / "baseline_fn_set.json"
    baseline_fp_path = RESULTS_DIR / "baseline_fp_set.json"
    if baseline_fn_path.exists() and baseline_fp_path.exists():
        print(f"[P3] Reusing existing baseline FN/FP sets at {baseline_fn_path}")
        baseline_blob = json.loads(baseline_fn_path.read_text())
        baseline_metrics = {
            "F1": baseline_blob.get("baseline_f1"),
            "P": baseline_blob.get("baseline_p"),
            "R": baseline_blob.get("baseline_r"),
        }
    else:
        baseline_metrics = run_unpatched_baseline(components, sentences, gold_pairs)

    # Step 2: patched (lambda True) sweep.
    os.environ["PHASE_CACHE_DIR"] = str(PHASE_CACHE_DIR_PATCHED)
    original_has_standalone = SLinker13._has_standalone_mention

    def always_true(comp_name, text):
        return True

    SLinker13._has_standalone_mention = staticmethod(always_true)
    try:
        print(f"[P3] Running PATCHED (lambda True) SLinker13 on BBB ...")
        linker = SLinker13(backend=LLMBackend.CLAUDE)
        t_sweep = time.time()
        predictions = linker.link(
            text_path=str(BBB["text"]),
            model_path=str(BBB["model"]),
        )
        sweep_time = time.time() - t_sweep
    finally:
        SLinker13._has_standalone_mention = staticmethod(original_has_standalone)

    predicted_pairs = {(p.sentence_number, p.component_id) for p in predictions}
    metrics = eval_metrics(predicted_pairs, gold_pairs)
    print(f"[P3] BBB sweep: P={metrics['P']:.4f} R={metrics['R']:.4f} "
          f"F1={metrics['F1']:.4f} TP={metrics['tp']} FP={metrics['fp']} "
          f"FN={metrics['fn']} time={sweep_time:.0f}s")

    # Load baseline for FN-recovery + new-FPs.
    baseline_fn_blob = json.loads((RESULTS_DIR / "baseline_fn_set.json").read_text())
    baseline_fns = {(int(s), c) for s, c in baseline_fn_blob.get("fn_set", [])}
    baseline_fn_count = baseline_fn_blob.get("fn_count", len(baseline_fns))
    fn_pairs = gold_pairs - predicted_pairs
    fns_recovered = len(baseline_fns - fn_pairs)

    baseline_fp_blob = json.loads((RESULTS_DIR / "baseline_fp_set.json").read_text())
    baseline_fps = {(int(s), c) for s, c in baseline_fp_blob.get("fp_set", [])}
    fp_pairs = predicted_pairs - gold_pairs
    new_fps = len(fp_pairs - baseline_fps)

    PARENT_F1 = 0.8990
    PURE_LLM_F1 = 0.8108

    result_blob = {
        "probe": "P3",
        "dataset": "bigbluebutton",
        "bbb_f1": metrics["F1"],
        "bbb_precision": metrics["P"],
        "bbb_recall": metrics["R"],
        "bbb_tp": metrics["tp"],
        "bbb_fp": metrics["fp"],
        "bbb_fn": metrics["fn"],
        "delta_vs_parent_s_linker13": round(metrics["F1"] - PARENT_F1, 4),
        "delta_vs_pure_llm_floor": round(metrics["F1"] - PURE_LLM_F1, 4),
        "llm_calls": 0,
        "total_latency_s": round(sweep_time, 1),
        "sweep_latency_s": round(sweep_time, 1),
        "projected_full_sweep_calls": 0,
        "fns_recovered_of_17": fns_recovered,
        "baseline_fn_count": baseline_fn_count,
        "new_fps_introduced": new_fps,
        "baseline_metrics": baseline_metrics,
        "generality_cost": (
            "Removes the rule entirely; reviewer-defensible IF F1 "
            "holds. No structural rule and no LLM replacement — "
            "downstream tiers carry the standalone-mention semantics "
            "on their own."
        ),
        "implementation_cost_estimate": (
            "Trivial: one-line change (delete the regex, replace its "
            "callers with True). ~0.1 day."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_s": round(time.time() - t0, 1),
    }

    out_path = RESULTS_DIR / "p3_bbb.json"
    out_path.write_text(json.dumps(result_blob, indent=2))
    print(f"[P3] Wrote {out_path}")
    print(f"[P3] DONE. F1={metrics['F1']:.4f} fns_recovered={fns_recovered}/"
          f"{baseline_fn_count} new_fps={new_fps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
