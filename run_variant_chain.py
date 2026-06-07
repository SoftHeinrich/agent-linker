"""Run s_linker18a..18 on all 5 datasets, compare F1 to 17f baseline.

Each variant writes to its own backend-namespaced phase cache, so checkpoint
hits from 17f's prompts cascade through (re-runs of unchanged prompts replay).

Usage:
    cd approach/
    LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.4 OPENAI_SERVICE_TIER=flex \\
        python run_variant_chain.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

ROOT = Path(__file__).parent
BENCH = ROOT / "../ardoco/core/tests-base/src/main/resources/benchmark"

VARIANTS = {
    "s_linker17f": ("llm_sad_sam.linkers.experimental.s_linker17f", "SLinker17f"),
    "s_linker18a": ("llm_sad_sam.linkers.experimental.s_linker18a", "SLinker18a"),
    "s_linker18b": ("llm_sad_sam.linkers.experimental.s_linker18b", "SLinker18b"),
    "s_linker18c": ("llm_sad_sam.linkers.experimental.s_linker18c", "SLinker18c"),
    "s_linker18d": ("llm_sad_sam.linkers.experimental.s_linker18d", "SLinker18d"),
    "s_linker18":  ("llm_sad_sam.linkers.experimental.s_linker18",  "SLinker18"),
    "s_linker19":  ("llm_sad_sam.linkers.experimental.s_linker19",  "SLinker19"),
    "s_linker19U": ("llm_sad_sam.linkers.experimental.s_linker19U", "SLinker19U"),
}

DATASETS = {
    "mediastore":    (BENCH/"mediastore/text_2016/mediastore.txt",
                      BENCH/"mediastore/model_2016/pcm/ms.repository",
                      BENCH/"mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"),
    "teastore":      (BENCH/"teastore/text_2020/teastore.txt",
                      BENCH/"teastore/model_2020/pcm/teastore.repository",
                      BENCH/"teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"),
    "teammates":     (BENCH/"teammates/text_2021/teammates.txt",
                      BENCH/"teammates/model_2021/pcm/teammates.repository",
                      BENCH/"teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "bigbluebutton": (BENCH/"bigbluebutton/text_2021/bigbluebutton.txt",
                      BENCH/"bigbluebutton/model_2021/pcm/bbb.repository",
                      BENCH/"bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "jabref":        (BENCH/"jabref/text_2021/jabref.txt",
                      BENCH/"jabref/model_2021/pcm/jabref.repository",
                      BENCH/"jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
}


def load_gold(p):
    g = set()
    for r in csv.DictReader(open(p)):
        cid = r.get("modelElementID", "").strip()
        sn = r.get("sentence", "").strip()
        if cid and sn:
            g.add((int(sn), cid))
    return g


def run_one(variant_name, module, class_name, datasets, backend):
    print(f"\n{'#'*80}\n# {variant_name}\n{'#'*80}")
    mod = __import__(module, fromlist=[class_name])
    cls = getattr(mod, class_name)
    linker = cls(backend=backend)
    results = {}
    for ds, (text_p, model_p, gold_p) in datasets.items():
        gold = load_gold(gold_p)
        t0 = time.time()
        try:
            final = linker.link(str(text_p), str(model_p))
        except Exception as e:
            print(f"  {ds}: ERROR {type(e).__name__}: {e}")
            results[ds] = {"error": str(e)}
            continue
        keys = {(lk.sentence_number, lk.component_id) for lk in final}
        tp = len(keys & gold); fp = len(keys - gold)
        p = tp / max(1, tp + fp); r = tp / max(1, len(gold))
        f1 = 2 * p * r / max(1e-9, p + r)
        elapsed = round(time.time() - t0, 1)
        results[ds] = {"tp": tp, "fp": fp, "p": round(p, 4), "r": round(r, 4),
                       "f1": round(f1, 4), "elapsed_s": elapsed,
                       "gold": len(gold)}
        print(f"  {ds:<14} TP={tp:>3} FP={fp:<3} P={p:.3f} R={r:.3f} F1={f1:.3f} ({elapsed}s)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS.keys()))
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    ap.add_argument("--backend", default=os.environ.get("LLM_BACKEND", "openai"))
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    from llm_sad_sam.llm_client import LLMBackend
    backend_map = {
        "claude": LLMBackend.CLAUDE, "openai": LLMBackend.OPENAI,
        "checkpoint": LLMBackend.CHECKPOINT,
    }
    backend = backend_map.get(args.backend, LLMBackend.OPENAI)
    os.environ["LLM_BACKEND"] = args.backend

    datasets = {k: DATASETS[k] for k in args.datasets if k in DATASETS}
    all_results = {}
    for v in args.variants:
        if v not in VARIANTS:
            print(f"  unknown variant '{v}', skipping")
            continue
        module, cls = VARIANTS[v]
        all_results[v] = run_one(v, module, cls, datasets, backend)

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print(f"SUMMARY: macro F1 per variant")
    print(f"{'='*100}")
    print(f"{'Variant':<14}" + "".join(f"{ds:>14}" for ds in datasets) + "  macro F1   micro TP/FP")
    print("-" * 100)
    for v, vr in all_results.items():
        f1s = []
        macro_tp = 0; macro_fp = 0
        cells = []
        for ds in datasets:
            r = vr.get(ds, {})
            if "error" in r or not r:
                cells.append(f"{'ERR':>14}")
                continue
            cells.append(f"{r['tp']}/{r['fp']} {r['f1']:.3f}".rjust(14))
            f1s.append(r["f1"])
            macro_tp += r["tp"]; macro_fp += r["fp"]
        macro = sum(f1s) / max(1, len(f1s))
        print(f"{v:<14}" + "".join(cells) + f"  {macro:>8.4f}   {macro_tp}/{macro_fp}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults: {args.out_json}")


if __name__ == "__main__":
    main()
