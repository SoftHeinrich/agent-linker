"""Per-framing + per-phase contribution analysis for s_linker17f.

Runs s_linker17f (any backend) and reads the enriched phase cache to produce a
full pipeline breakdown:

  * Phase 2 raw framing counts (A / B / C / C-pass1 / C-pass2)
  * Phase 3 union size
  * Phase 4 generic-filter decisions (component vs generic)
  * Phase 4 twopass decisions (p1, p2, approve)
  * Phase 4b code-path filter drops with LLM reasons
  * Phase 5 coref raw vs validated vs rejected
  * Per-framing unique TP/FP (the original analysis)
  * Per-final-link provenance (which framings, which gates)
  * Per-phase LLM call counts and elapsed time

Usage:
    cd approach/
    # Replay from cache (no LLM calls):
    LLM_BACKEND=checkpoint python analyze_framings.py
    # Live gpt-5.4 run, namespaced to phase_cache/s_linker17f/openai/:
    LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.4 python analyze_framings.py
    # Subset:
    python analyze_framings.py --datasets mediastore jabref
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

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


def _resolve_phase_dir(linker, text_path: str) -> Path:
    """Use the linker's own checkpoint resolver — same dir it writes to."""
    return Path(linker._checkpoint_dir(text_path))


def _load_pkl(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def run_and_analyze(dataset_name: str, paths: dict, linker) -> dict:
    """Run linker, load enriched phase cache, compute per-phase + per-framing stats."""
    text_path = str(paths["text"])
    model_path = str(paths["model"])
    gold = load_gold(paths["gold_sam"])

    print(f"\n{'='*70}")
    print(f"  {dataset_name}  (gold: {len(gold)} links)")
    print(f"{'='*70}")

    # Run the linker. Writes to phase_cache/{variant}/{backend}/{ds}/.
    final_links = linker.link(text_path, model_path)

    phase_dir = _resolve_phase_dir(linker, text_path)
    l2 = _load_pkl(phase_dir / "layer2.pkl")
    l3 = _load_pkl(phase_dir / "layer3.pkl")
    l4 = _load_pkl(phase_dir / "layer4.pkl")
    lf = _load_pkl(phase_dir / "final.pkl")

    if not l2:
        print(f"  WARNING: no layer2.pkl at {phase_dir}, skipping")
        return {}

    fa: dict = l2["framing_a"]
    fb: dict = l2["framing_b"]
    fc: dict = l2["framing_c"]
    fc_pass1 = l2.get("framing_c_pass1") or {}
    fc_pass2 = l2.get("framing_c_pass2") or {}

    candidates = l3.get("candidates", []) if l3 else []
    validated_pre_4b = l3.get("validated_pre_4b", []) if l3 else []
    validated = l3.get("validated", []) if l3 else []
    p4_generic = l3.get("phase_4_generic_decisions", {}) if l3 else {}
    p4_twopass = l3.get("phase_4_twopass_decisions", {}) if l3 else {}
    p4b_decisions = l3.get("phase_4b_decisions", {}) if l3 else {}

    coref_raw = l4.get("coref_raw", []) if l4 else []
    coref_validated = l4.get("coref_validated", []) if l4 else []
    coref_meta = l4.get("coref_metadata", {}) if l4 else {}
    coref_decisions = l4.get("coref_decisions", {}) if l4 else {}
    coref_anaphoric = l4.get("coref_anaphoric_snums", []) if l4 else []
    coref_terminals = l4.get("coref_terminals", set()) if l4 else set()

    final = lf.get("final", []) if lf else []
    final_provenance = lf.get("final_provenance", {}) if lf else {}
    phase_metrics = lf.get("phase_metrics", {}) if lf else {}

    # ── Phase 2 / Framing counts ────────────────────────────────────────────
    keys_a, keys_b, keys_c = set(fa), set(fb), set(fc)
    print(f"\n  [Phase 2] Raw framing candidates:")
    print(f"    Framing A: {len(keys_a)}")
    print(f"    Framing B: {len(keys_b)}")
    print(f"    Framing C: {len(keys_c)}  (pass1={len(fc_pass1)} pass2={len(fc_pass2)})")

    # ── Phase 3 / Union ─────────────────────────────────────────────────────
    union_keys = keys_a | keys_b | keys_c
    print(f"\n  [Phase 3] Union: {len(union_keys)} candidates "
          f"({len(candidates)} after sent_map filter)")

    # ── Phase 4 / Generic + twopass ─────────────────────────────────────────
    gf_rejects = [k for k, v in p4_generic.items() if not v.get("approved")]
    gf_passes = [k for k, v in p4_generic.items() if v.get("approved")]
    tp_approves = [k for k, v in p4_twopass.items() if v.get("approved")]
    tp_rejects = [k for k, v in p4_twopass.items() if not v.get("approved")]
    print(f"\n  [Phase 4] Validation:")
    print(f"    Generic filter:  {len(p4_generic)} candidates evaluated, "
          f"{len(gf_rejects)} rejected ({len(gf_passes)} passed)")
    print(f"    Twopass (p1∧p2): {len(p4_twopass)} candidates, "
          f"{len(tp_approves)} approved ({len(tp_rejects)} rejected)")
    print(f"    Validated (pre-4b): {len(validated_pre_4b)}")

    # ── Phase 4b / Code-path filter ─────────────────────────────────────────
    p4b_drops = [k for k, v in p4b_decisions.items() if v.get("dropped")]
    print(f"\n  [Phase 4b] Code-path filter:")
    print(f"    Evaluated: {len(p4b_decisions)},  Dropped: {len(p4b_drops)}")
    if p4b_drops:
        gold_killed = sum(1 for k in p4b_drops if k in gold)
        print(f"    of which {gold_killed} were gold (TPs killed) "
              f"and {len(p4b_drops) - gold_killed} were FP-killing wins")
        for k in p4b_drops[:5]:
            v = p4b_decisions[k]
            label = "TP-killed" if k in gold else "FP-killed"
            print(f"      [{label}] S{k[0]} -> cid={k[1]}: {v.get('reason', '')[:80]}")

    # ── Phase 5 / Coref ─────────────────────────────────────────────────────
    raw_keys = {(lk.sentence_number, lk.component_id) for lk in coref_raw}
    val_keys = {(lk.sentence_number, lk.component_id) for lk in coref_validated}
    via_alias = sum(1 for m in coref_meta.values() if m.get("antecedent_via_alias"))
    coref_keep = sum(1 for v in coref_decisions.values() if v.get("approved"))
    coref_reject = sum(1 for v in coref_decisions.values() if not v.get("approved"))
    print(f"\n  [Phase 5] Coreference:")
    print(f"    Anaphoric sents considered: {len(coref_anaphoric)}")
    print(f"    Specific terminals (LLM-classified): {len(coref_terminals)}")
    print(f"    Raw resolutions: {len(raw_keys)} "
          f"({via_alias} via_alias, {len(raw_keys) - via_alias} via_canonical)")
    print(f"    Validation: {coref_keep} approved, {coref_reject} rejected")
    print(f"    Validated final: {len(val_keys)}")

    # ── Per-framing unique TP/FP (validated, original analysis) ─────────────
    validated_keys = {(c.sentence_number, c.component_id) for c in validated}
    val_a = keys_a & validated_keys
    val_b = keys_b & validated_keys
    val_c = keys_c & validated_keys
    uniq_a = val_a - keys_b - keys_c
    uniq_b = val_b - keys_a - keys_c
    uniq_c = val_c - keys_a - keys_b
    shared_ab = (val_a & val_b) - keys_c
    shared_ac = (val_a & val_c) - keys_b
    shared_bc = (val_b & val_c) - keys_a
    shared_abc = val_a & val_b & val_c

    def tp_fp(keys):
        return len(keys & gold), len(keys - gold)

    ua_tp, ua_fp = tp_fp(uniq_a)
    ub_tp, ub_fp = tp_fp(uniq_b)
    uc_tp, uc_fp = tp_fp(uniq_c)
    co_tp, co_fp = tp_fp(val_keys)

    print(f"\n  Per-framing UNIQUE contributions (validated post-4b):")
    print(f"    A-only:    {len(uniq_a):3d} -> {ua_tp} TP, {ua_fp} FP")
    print(f"    B-only:    {len(uniq_b):3d} -> {ub_tp} TP, {ub_fp} FP")
    print(f"    C-only:    {len(uniq_c):3d} -> {uc_tp} TP, {uc_fp} FP")
    print(f"    Coref:     {len(val_keys):3d} -> {co_tp} TP, {co_fp} FP")
    print(f"  Overlap:")
    print(f"    A∩B∩C: {len(shared_abc)}  (TP {len(shared_abc & gold)})")
    print(f"    A∩B only: {len(shared_ab)}  (TP {len(shared_ab & gold)})")
    print(f"    A∩C only: {len(shared_ac)}  (TP {len(shared_ac & gold)})")
    print(f"    B∩C only: {len(shared_bc)}  (TP {len(shared_bc & gold)})")

    # ── Final + provenance ─────────────────────────────────────────────────
    final_keys = {(lk.sentence_number, lk.component_id) for lk in final}
    f_tp, f_fp = tp_fp(final_keys)
    print(f"\n  [Phase 6] FINAL: {len(final)} links -> {f_tp} TP, {f_fp} FP")
    if gold:
        p = f_tp / max(1, f_tp + f_fp)
        r = f_tp / max(1, len(gold))
        f1 = 2 * p * r / max(1e-9, p + r)
        print(f"    P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

    # ── Per-phase metrics from trace ────────────────────────────────────────
    if phase_metrics:
        print(f"\n  [Trace] Per-phase LLM metrics:")
        for ph, m in sorted(phase_metrics.items()):
            if ph == "_total":
                continue
            print(f"    {ph:32s}  calls={m['calls']:3d}  "
                  f"elapsed={m['elapsed_s']:7.2f}s  tokens={m['tokens']}")
        tot = phase_metrics.get("_total", {})
        if tot:
            print(f"    {'_total':32s}  calls={tot.get('llm_calls', 0):3d}  "
                  f"elapsed={tot.get('elapsed_s', 0):7.2f}s")

    return {
        "dataset": dataset_name,
        "gold": len(gold),
        "raw_a": len(keys_a), "raw_b": len(keys_b), "raw_c": len(keys_c),
        "union": len(union_keys),
        "validated_pre_4b": len(validated_pre_4b),
        "validated": len(validated),
        "p4b_drops": len(p4b_drops),
        "coref_raw": len(raw_keys),
        "coref_validated": len(val_keys),
        "uniq_a": {"cands": len(uniq_a), "tp": ua_tp, "fp": ua_fp},
        "uniq_b": {"cands": len(uniq_b), "tp": ub_tp, "fp": ub_fp},
        "uniq_c": {"cands": len(uniq_c), "tp": uc_tp, "fp": uc_fp},
        "coref": {"cands": len(val_keys), "tp": co_tp, "fp": co_fp},
        "shared_abc_tp": len(shared_abc & gold),
        "shared_ab_tp": len(shared_ab & gold),
        "final_tp": f_tp, "final_fp": f_fp,
    }


def parse_backend(name: str):
    from llm_sad_sam.llm_client import LLMBackend
    name = (name or "").lower()
    return {
        "claude": LLMBackend.CLAUDE,
        "openai": LLMBackend.OPENAI,
        "codex": LLMBackend.CODEX,
        "checkpoint": LLMBackend.CHECKPOINT,
    }.get(name, LLMBackend.CHECKPOINT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backend", default=os.environ.get("LLM_BACKEND", "checkpoint"),
        help="LLM backend: claude|openai|codex|checkpoint (default: $LLM_BACKEND or checkpoint)"
    )
    ap.add_argument(
        "--datasets", nargs="*", default=list(DATASETS.keys()),
        help=f"Subset of {list(DATASETS.keys())}"
    )
    ap.add_argument(
        "--out-json", default=None,
        help="Optional JSON output path for the per-dataset summary"
    )
    args = ap.parse_args()

    os.environ.setdefault("PHASE_CACHE_DIR", "./results/phase_cache")
    os.environ["LLM_BACKEND"] = args.backend

    from llm_sad_sam.linkers.experimental.s_linker17f import SLinker17f

    backend = parse_backend(args.backend)
    print(f"\nRunning analyze_framings with backend={backend.value}")
    linker = SLinker17f(backend=backend)

    results = []
    for name in args.datasets:
        if name not in DATASETS:
            print(f"  Unknown dataset '{name}', skipping")
            continue
        r = run_and_analyze(name, DATASETS[name], linker)
        if r:
            results.append(r)

    # Summary table
    print(f"\n\n{'='*90}")
    print("SUMMARY: per-dataset pipeline funnel + per-framing unique TP/FP")
    print(f"{'='*90}")
    print(f"{'Dataset':<14} {'A':>4} {'B':>4} {'C':>4} {'Un':>4} {'V4':>4} "
          f"{'V4b':>4} {'CrR':>4} {'CrV':>4} "
          f"{'A-TP/FP':>9} {'B-TP/FP':>9} {'C-TP/FP':>9} {'CoTP/FP':>9} "
          f"{'F-TP/FP':>9}")
    print("-" * 110)
    totals = defaultdict(int)
    for r in results:
        a, b, c, co = r["uniq_a"], r["uniq_b"], r["uniq_c"], r["coref"]
        print(
            f"{r['dataset']:<14} "
            f"{r['raw_a']:>4} {r['raw_b']:>4} {r['raw_c']:>4} {r['union']:>4} "
            f"{r['validated_pre_4b']:>4} {r['validated']:>4} "
            f"{r['coref_raw']:>4} {r['coref_validated']:>4} "
            f"{a['tp']:>3}/{a['fp']:<5} "
            f"{b['tp']:>3}/{b['fp']:<5} "
            f"{c['tp']:>3}/{c['fp']:<5} "
            f"{co['tp']:>3}/{co['fp']:<5} "
            f"{r['final_tp']:>3}/{r['final_fp']:<5}"
        )
        for key in ["uniq_a", "uniq_b", "uniq_c", "coref"]:
            totals[f"{key}_tp"] += r[key]["tp"]
            totals[f"{key}_fp"] += r[key]["fp"]
        totals["final_tp"] += r["final_tp"]
        totals["final_fp"] += r["final_fp"]
        totals["shared_abc_tp"] += r["shared_abc_tp"]
    print("-" * 110)
    print(
        f"{'TOTAL':<14} {'':>4} {'':>4} {'':>4} {'':>4} {'':>4} {'':>4} "
        f"{'':>4} {'':>4} "
        f"{totals['uniq_a_tp']:>3}/{totals['uniq_a_fp']:<5} "
        f"{totals['uniq_b_tp']:>3}/{totals['uniq_b_fp']:<5} "
        f"{totals['uniq_c_tp']:>3}/{totals['uniq_c_fp']:<5} "
        f"{totals['coref_tp']:>3}/{totals['coref_fp']:<5} "
        f"{totals['final_tp']:>3}/{totals['final_fp']:<5}"
    )
    print("\nColumn key: A/B/C raw cands, Un=union, V4=Phase-4 validated, "
          "V4b=post-4b, CrR=coref raw, CrV=coref validated")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSummary written to {args.out_json}")


if __name__ == "__main__":
    main()
