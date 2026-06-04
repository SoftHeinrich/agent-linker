"""Per-framing contribution analysis for s_linker17f.

Runs s_linker17f with checkpoint backend (all LLM calls replayed from cache,
zero new API calls) and reports per-framing TP/FP/unique breakdown.

Usage:
    cd approach/
    LLM_BACKEND=checkpoint python analyze_framings.py
"""
from __future__ import annotations

import csv
import os
import pickle
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
BENCHMARK_BASE = ROOT / "../ardoco/core/tests-base/src/main/resources/benchmark"
DATASETS = {
    "mediastore": {
        "text":     BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
        "model":    BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
        "gold_sam": BENCHMARK_BASE / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    },
    "teastore": {
        "text":     BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
        "model":    BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
        "gold_sam": BENCHMARK_BASE / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    },
    "teammates": {
        "text":     BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model":    BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
        "gold_sam": BENCHMARK_BASE / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "bigbluebutton": {
        "text":     BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model":    BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
        "gold_sam": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "jabref": {
        "text":     BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model":    BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
        "gold_sam": BENCHMARK_BASE / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
}


def load_gold(gold_path: Path) -> set[tuple[int, str]]:
    links: set[tuple[int, str]] = set()
    with open(gold_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def run_and_analyze(dataset_name: str, paths: dict, linker) -> dict:
    """Run linker, load phase cache, compute per-framing stats against gold."""
    text_path = str(paths["text"])
    model_path = str(paths["model"])
    gold = load_gold(paths["gold_sam"])

    print(f"\n{'='*60}")
    print(f"  {dataset_name}  (gold: {len(gold)} links)")
    print(f"{'='*60}")

    # Run the linker (checkpoint backend → replays cached LLM calls)
    final_links = linker.link(text_path, model_path)

    # Load phase cache written during this run
    cache_dir = Path(os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache"))
    ds_stem = Path(text_path).stem
    phase_dir = cache_dir / linker._VARIANT_NAME / ds_stem

    layer2_path = phase_dir / "layer2.pkl"
    layer3_path = phase_dir / "layer3.pkl"
    final_path  = phase_dir / "final.pkl"

    if not layer2_path.exists():
        print(f"  WARNING: no layer2.pkl at {layer2_path}, skipping framing breakdown")
        return {}

    with open(layer2_path, "rb") as f:
        l2 = pickle.load(f)
    with open(layer3_path, "rb") as f:
        l3 = pickle.load(f)
    with open(final_path, "rb") as f:
        lf = pickle.load(f)

    fa: dict = l2["framing_a"]   # (snum, cid) -> SadSamLink
    fb: dict = l2["framing_b"]   # (snum, cid) -> SadSamLink
    fc: dict = l2["framing_c"]   # (snum, cid) -> CandidateLink (pre-validation)
    decisions: dict = l3.get("decisions", {})
    validated_candidates = l3.get("validated", [])

    # Validated keys from Phase 4 (multi-framing)
    validated_mf: set = {(c.sentence_number, c.component_id) for c in validated_candidates}

    # Coref keys from final links
    all_final = lf["final"]
    coref_keys: set = {
        (lk.sentence_number, lk.component_id)
        for lk in all_final
        if lk.source == "coreference"
    }

    # Per-framing candidate sets (pre-validation)
    keys_a = set(fa.keys())
    keys_b = set(fb.keys())
    keys_c = set(fc.keys())

    # Per-framing validated sets: intersection of framing candidates with validated pool
    val_a = keys_a & validated_mf
    val_b = keys_b & validated_mf
    val_c = keys_c & validated_mf

    # Unique to each framing (not in any other framing's candidate set, but validated)
    uniq_a = val_a - keys_b - keys_c
    uniq_b = val_b - keys_a - keys_c
    uniq_c = val_c - keys_a - keys_b
    shared_ab = val_a & val_b - keys_c
    shared_ac = val_a & val_c - keys_b
    shared_bc = val_b & val_c - keys_a
    shared_abc = val_a & val_b & val_c

    # TP/FP per framing (unique only)
    def tp_fp(keys: set) -> tuple[int, int]:
        tp = len(keys & gold)
        fp = len(keys - gold)
        return tp, fp

    uniq_a_tp, uniq_a_fp = tp_fp(uniq_a)
    uniq_b_tp, uniq_b_fp = tp_fp(uniq_b)
    uniq_c_tp, uniq_c_fp = tp_fp(uniq_c)
    coref_tp, coref_fp = tp_fp(coref_keys)

    shared_ab_tp = len(shared_ab & gold)
    shared_abc_tp = len(shared_abc & gold)

    total_mf_val_keys = validated_mf
    total_tp, total_fp = tp_fp(total_mf_val_keys | coref_keys)

    print(f"\n  Raw candidates: A={len(keys_a)}  B={len(keys_b)}  C={len(keys_c)}")
    print(f"  Validated (Phase 4): A={len(val_a)}  B={len(val_b)}  C={len(val_c)}")
    print(f"\n  Overlap (validated):")
    print(f"    A∩B∩C : {len(shared_abc)}  (TPs: {shared_abc_tp})")
    print(f"    A∩B only: {len(shared_ab)}  (TPs: {shared_ab_tp})")
    print(f"    A∩C only: {len(shared_ac)}  (TPs: {len(shared_ac & gold)})")
    print(f"    B∩C only: {len(shared_bc)}  (TPs: {len(shared_bc & gold)})")
    print(f"\n  Unique contributions (validated, not proposed by other framings):")
    print(f"    A-only:    {len(uniq_a):3d} candidates → {uniq_a_tp} TP, {uniq_a_fp} FP")
    print(f"    B-only:    {len(uniq_b):3d} candidates → {uniq_b_tp} TP, {uniq_b_fp} FP")
    print(f"    C-only:    {len(uniq_c):3d} candidates → {uniq_c_tp} TP, {uniq_c_fp} FP")
    print(f"    Coref:     {len(coref_keys):3d} validated  → {coref_tp} TP, {coref_fp} FP")

    # Show C-unique TP sentences (the alias-backed evidence)
    if uniq_c:
        print(f"\n  C-unique validated candidates:")
        for k in sorted(uniq_c):
            cl = fc.get(k)
            if cl:
                is_tp = k in gold
                label = "TP" if is_tp else "FP"
                matched = getattr(cl, "matched_text", "?")
                text = getattr(cl, "sentence_text", "?")
                comp = getattr(cl, "component_name", "?")
                print(f"    [{label}] S{k[0]} -> {comp} (matched: \"{matched}\")")
                if text:
                    print(f"         {text[:100]}")

    # Show B-unique TP sentences (the actor-role distinction evidence)
    if uniq_b:
        print(f"\n  B-only validated candidates (actor-role unique):")
        for k in sorted(uniq_b)[:8]:
            lk = fb.get(k)
            is_tp = k in gold
            label = "TP" if is_tp else "FP"
            comp = getattr(lk, "component_name", "?") if lk else "?"
            matched = getattr(lk, "matched_text", "?") if lk else "?"
            print(f"    [{label}] S{k[0]} -> {comp} (matched: \"{matched}\")")

    if uniq_a:
        print(f"\n  A-only validated candidates (explicit-mention unique):")
        for k in sorted(uniq_a)[:8]:
            lk = fa.get(k)
            is_tp = k in gold
            label = "TP" if is_tp else "FP"
            comp = getattr(lk, "component_name", "?") if lk else "?"
            matched = getattr(lk, "matched_text", "?") if lk else "?"
            print(f"    [{label}] S{k[0]} -> {comp} (matched: \"{matched}\")")

    return {
        "dataset": dataset_name,
        "gold": len(gold),
        "uniq_a": {"cands": len(uniq_a), "tp": uniq_a_tp, "fp": uniq_a_fp},
        "uniq_b": {"cands": len(uniq_b), "tp": uniq_b_tp, "fp": uniq_b_fp},
        "uniq_c": {"cands": len(uniq_c), "tp": uniq_c_tp, "fp": uniq_c_fp},
        "coref":  {"cands": len(coref_keys), "tp": coref_tp, "fp": coref_fp},
        "shared_abc_tp": shared_abc_tp,
        "shared_ab_tp": shared_ab_tp,
    }


def main():
    os.environ.setdefault("LLM_BACKEND", "checkpoint")
    os.environ.setdefault("PHASE_CACHE_DIR", "./results/phase_cache")

    from llm_sad_sam.llm_client import LLMBackend
    from llm_sad_sam.linkers.experimental.s_linker17f import SLinker17f

    backend = LLMBackend.CHECKPOINT
    linker = SLinker17f(backend=backend)

    results = []
    for name, paths in DATASETS.items():
        r = run_and_analyze(name, paths, linker)
        if r:
            results.append(r)

    # Summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY: Per-framing unique TP/FP across all projects")
    print(f"{'='*70}")
    print(f"{'Dataset':<16} {'A-uniq TP/FP':>14} {'B-uniq TP/FP':>14} {'C-uniq TP/FP':>14} {'Coref TP/FP':>12} {'Shared-ABC TP':>14}")
    print("-"*70)
    totals = defaultdict(int)
    for r in results:
        a, b, c, co = r["uniq_a"], r["uniq_b"], r["uniq_c"], r["coref"]
        print(f"{r['dataset']:<16} "
              f"{a['tp']:>4}/{a['fp']:<8} "
              f"{b['tp']:>4}/{b['fp']:<8} "
              f"{c['tp']:>4}/{c['fp']:<8} "
              f"{co['tp']:>4}/{co['fp']:<6} "
              f"{r['shared_abc_tp']:>6}")
        for key in ["uniq_a", "uniq_b", "uniq_c", "coref"]:
            totals[f"{key}_tp"] += r[key]["tp"]
            totals[f"{key}_fp"] += r[key]["fp"]
        totals["shared_abc_tp"] += r["shared_abc_tp"]
    print("-"*70)
    print(f"{'TOTAL':<16} "
          f"{totals['uniq_a_tp']:>4}/{totals['uniq_a_fp']:<8} "
          f"{totals['uniq_b_tp']:>4}/{totals['uniq_b_fp']:<8} "
          f"{totals['uniq_c_tp']:>4}/{totals['uniq_c_fp']:<8} "
          f"{totals['coref_tp']:>4}/{totals['coref_fp']:<6} "
          f"{totals['shared_abc_tp']:>6}")


if __name__ == "__main__":
    main()
