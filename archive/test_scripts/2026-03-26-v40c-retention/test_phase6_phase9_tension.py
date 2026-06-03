#!/usr/bin/env python3
"""
Precision-Recall Tension Analysis: Phase 6 (generic-mention-reject) vs Phase 9 (4-rule judge) in V39a.

Measures how many TPs and FPs each filter catches/kills by comparing checkpoint states
against gold standards. Purely offline -- no LLM calls.

Usage:
    python test_phase6_phase9_tension.py
"""

import csv
import os
import pickle
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "results", "phase_cache", "v39a")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "ablation_results")
GOLD_DIR = os.path.join(
    BASE_DIR, "..", "ardoco", "core", "tests-base", "src", "main",
    "resources", "benchmark",
)

DATASETS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]

# Gold standard file patterns per dataset
GOLD_FILES = {
    "mediastore": "goldstandard_sad_2016-sam_2016.csv",
    "teastore": "goldstandard_sad_2020-sam_2020.csv",
    "teammates": "goldstandard_sad_2021-sam_2021.csv",
    "bigbluebutton": "goldstandard_sad_2021-sam_2021.csv",
    "jabref": "goldstandard_sad_2021-sam_2021.csv",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gold(dataset: str) -> set[tuple[int, str]]:
    """Load gold standard as set of (sentence_number, modelElementID)."""
    path = os.path.join(GOLD_DIR, dataset, "goldstandards", GOLD_FILES[dataset])
    gold = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gold.add((int(row["sentence"]), row["modelElementID"]))
    return gold


def load_pickle(dataset: str, phase: str) -> dict:
    path = os.path.join(CHECKPOINT_DIR, dataset, f"{phase}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def links_to_set(links, key_sent="sentence_number", key_id="component_id") -> set[tuple[int, str]]:
    """Convert a list of link objects to a set of (sentence, component_id)."""
    return {(getattr(l, key_sent), getattr(l, key_id)) for l in links}


def compute_prf(predicted: set, gold: set) -> dict:
    """Compute precision, recall, F1."""
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "P": p, "R": r, "F1": f1}


def format_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def format_link(sent: int, comp_id: str, objects, attr_name="component_name") -> str:
    """Find the component name for a (sent, comp_id) pair from a list of objects."""
    for obj in objects:
        s = getattr(obj, "sentence_number", None)
        c = getattr(obj, "component_id", None)
        if s == sent and c == comp_id:
            name = getattr(obj, attr_name, "?")
            source = getattr(obj, "source", "?")
            match = getattr(obj, "matched_text", "")
            extra = f" match='{match}'" if match else ""
            return f"S{sent} {name} (source={source}{extra})"
    return f"S{sent} {comp_id}"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_dataset(dataset: str) -> dict:
    """Analyze Phase 6 and Phase 9 impact for one dataset."""
    gold = load_gold(dataset)

    # Load checkpoints
    p5_data = load_pickle(dataset, "phase5")
    p6_data = load_pickle(dataset, "phase6")
    pj_data = load_pickle(dataset, "pre_judge")
    fn_data = load_pickle(dataset, "final")

    p5_candidates = p5_data["candidates"]
    p6_validated = p6_data["validated"]
    pj_preliminary = pj_data["preliminary"]
    fn_final = fn_data["final"]
    fn_rejected = fn_data.get("rejected", [])

    # Sets of (sentence, component_id)
    p5_set = links_to_set(p5_candidates)
    p6_set = links_to_set(p6_validated)
    pj_set = links_to_set(pj_preliminary)
    fn_set = links_to_set(fn_final)

    # --- Phase 6 analysis ---
    # Raw: candidates removed by Phase 6 validation
    p6_raw_rejected = p5_set - p6_set
    # Net: pairs that are truly absent from pre_judge (not recovered via other sources)
    p6_net_lost = p6_raw_rejected - pj_set
    # Also check: pairs that were rejected raw but re-entered later
    p6_re_entered = p6_raw_rejected & pj_set

    p6_tp_killed = p6_net_lost & gold
    p6_fp_caught = p6_net_lost - gold

    # --- Phase 9 analysis ---
    judge_rejected = pj_set - fn_set
    j_tp_killed = judge_rejected & gold
    j_fp_caught = judge_rejected - gold

    # --- Compute F1 with and without each filter ---
    # "Without Phase 6" = add back p6_net_lost links to final set
    fn_without_p6 = fn_set | p6_net_lost
    # "Without Phase 9" = add back judge_rejected links to final set
    fn_without_p9 = fn_set | judge_rejected
    # "Without both" = add both back
    fn_without_both = fn_set | p6_net_lost | judge_rejected

    prf_actual = compute_prf(fn_set, gold)
    prf_no_p6 = compute_prf(fn_without_p6, gold)
    prf_no_p9 = compute_prf(fn_without_p9, gold)
    prf_no_both = compute_prf(fn_without_both, gold)

    result = {
        "dataset": dataset,
        "gold_size": len(gold),
        # Phase 6
        "p6_raw_rejected": len(p6_raw_rejected),
        "p6_re_entered": len(p6_re_entered),
        "p6_net_lost": len(p6_net_lost),
        "p6_tp_killed": len(p6_tp_killed),
        "p6_fp_caught": len(p6_fp_caught),
        "p6_tp_killed_details": [],
        "p6_fp_caught_details": [],
        # Phase 9
        "j_rejected": len(judge_rejected),
        "j_tp_killed": len(j_tp_killed),
        "j_fp_caught": len(j_fp_caught),
        "j_tp_killed_details": [],
        "j_fp_caught_details": [],
        # F1 scenarios
        "prf_actual": prf_actual,
        "prf_no_p6": prf_no_p6,
        "prf_no_p9": prf_no_p9,
        "prf_no_both": prf_no_both,
    }

    # Collect details for killed TPs and caught FPs
    for sent, cid in sorted(p6_tp_killed):
        result["p6_tp_killed_details"].append(
            format_link(sent, cid, p5_candidates)
        )
    for sent, cid in sorted(p6_fp_caught):
        result["p6_fp_caught_details"].append(
            format_link(sent, cid, p5_candidates)
        )
    for sent, cid in sorted(j_tp_killed):
        result["j_tp_killed_details"].append(
            format_link(sent, cid, pj_preliminary)
        )
    for sent, cid in sorted(j_fp_caught):
        result["j_fp_caught_details"].append(
            format_link(sent, cid, pj_preliminary)
        )

    return result


def print_separator(char="=", width=100):
    print(char * width)


def print_results(results: list[dict]):
    print()
    print_separator()
    print("  V39a PRECISION-RECALL TENSION ANALYSIS")
    print("  Phase 6 (generic-mention-reject) vs Phase 9 (4-rule judge)")
    print_separator()

    # -----------------------------------------------------------------------
    # Per-dataset detailed results
    # -----------------------------------------------------------------------
    for r in results:
        ds = r["dataset"]
        print(f"\n{'─' * 100}")
        print(f"  {ds.upper()} (gold: {r['gold_size']} links)")
        print(f"{'─' * 100}")

        # Phase 6
        print(f"\n  Phase 6 (generic-mention-reject):")
        print(f"    Raw rejected:     {r['p6_raw_rejected']}")
        print(f"    Re-entered later: {r['p6_re_entered']}  (recovered via transarc/coref/etc)")
        print(f"    Net lost:         {r['p6_net_lost']}")
        print(f"    TPs killed:       {r['p6_tp_killed']}  {'** BAD **' if r['p6_tp_killed'] > 0 else '(clean)'}")
        print(f"    FPs caught:       {r['p6_fp_caught']}  {'(good)' if r['p6_fp_caught'] > 0 else '(none)'}")
        if r["p6_tp_killed_details"]:
            for d in r["p6_tp_killed_details"]:
                print(f"      TP KILLED: {d}")
        if r["p6_fp_caught_details"]:
            for d in r["p6_fp_caught_details"]:
                print(f"      FP caught: {d}")

        # Phase 9
        print(f"\n  Phase 9 (4-rule judge):")
        print(f"    Rejected:         {r['j_rejected']}")
        print(f"    TPs killed:       {r['j_tp_killed']}  {'** BAD **' if r['j_tp_killed'] > 0 else '(clean)'}")
        print(f"    FPs caught:       {r['j_fp_caught']}  {'(good)' if r['j_fp_caught'] > 0 else '(none)'}")
        if r["j_tp_killed_details"]:
            for d in r["j_tp_killed_details"]:
                print(f"      TP KILLED: {d}")
        if r["j_fp_caught_details"]:
            for d in r["j_fp_caught_details"]:
                print(f"      FP caught: {d}")

        # F1 comparison
        prf = r["prf_actual"]
        print(f"\n  F1 impact scenarios:")
        print(f"    {'Scenario':<25s} {'P':>7s} {'R':>7s} {'F1':>7s}  {'TP':>4s} {'FP':>4s} {'FN':>4s}")
        print(f"    {'─' * 65}")
        for label, key in [
            ("Actual (with both)", "prf_actual"),
            ("Without Phase 6", "prf_no_p6"),
            ("Without Phase 9", "prf_no_p9"),
            ("Without both", "prf_no_both"),
        ]:
            m = r[key]
            delta = m["F1"] - r["prf_actual"]["F1"]
            delta_str = f" ({'+' if delta >= 0 else ''}{delta * 100:.1f}pp)" if abs(delta) > 0.0005 else ""
            print(
                f"    {label:<25s} {format_pct(m['P']):>7s} {format_pct(m['R']):>7s} "
                f"{format_pct(m['F1']):>7s}  {m['TP']:>4d} {m['FP']:>4d} {m['FN']:>4d}{delta_str}"
            )

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print("  AGGREGATE SUMMARY")
    print(f"{'=' * 100}")

    # Phase 6 summary
    print(f"\n  Phase 6 (generic-mention-reject):")
    print(f"    {'Dataset':<16s} {'Raw Rej':>8s} {'Re-enter':>9s} {'Net Lost':>9s} {'TP Kill':>8s} {'FP Catch':>9s} {'Net':>6s}")
    print(f"    {'─' * 65}")
    totals_p6 = {"raw": 0, "re": 0, "net": 0, "tp": 0, "fp": 0}
    for r in results:
        net = r["p6_fp_caught"] - r["p6_tp_killed"]
        print(
            f"    {r['dataset']:<16s} {r['p6_raw_rejected']:>8d} {r['p6_re_entered']:>9d} "
            f"{r['p6_net_lost']:>9d} {r['p6_tp_killed']:>8d} {r['p6_fp_caught']:>9d} "
            f"{'+' if net >= 0 else ''}{net:>5d}"
        )
        totals_p6["raw"] += r["p6_raw_rejected"]
        totals_p6["re"] += r["p6_re_entered"]
        totals_p6["net"] += r["p6_net_lost"]
        totals_p6["tp"] += r["p6_tp_killed"]
        totals_p6["fp"] += r["p6_fp_caught"]
    net_total_p6 = totals_p6["fp"] - totals_p6["tp"]
    print(f"    {'─' * 65}")
    print(
        f"    {'TOTAL':<16s} {totals_p6['raw']:>8d} {totals_p6['re']:>9d} "
        f"{totals_p6['net']:>9d} {totals_p6['tp']:>8d} {totals_p6['fp']:>9d} "
        f"{'+' if net_total_p6 >= 0 else ''}{net_total_p6:>5d}"
    )

    # Phase 9 summary
    print(f"\n  Phase 9 (4-rule judge):")
    print(f"    {'Dataset':<16s} {'Rejected':>9s} {'TP Kill':>8s} {'FP Catch':>9s} {'Net':>6s}")
    print(f"    {'─' * 50}")
    totals_j = {"rej": 0, "tp": 0, "fp": 0}
    for r in results:
        net = r["j_fp_caught"] - r["j_tp_killed"]
        print(
            f"    {r['dataset']:<16s} {r['j_rejected']:>9d} {r['j_tp_killed']:>8d} "
            f"{r['j_fp_caught']:>9d} {'+' if net >= 0 else ''}{net:>5d}"
        )
        totals_j["rej"] += r["j_rejected"]
        totals_j["tp"] += r["j_tp_killed"]
        totals_j["fp"] += r["j_fp_caught"]
    net_total_j = totals_j["fp"] - totals_j["tp"]
    print(f"    {'─' * 50}")
    print(
        f"    {'TOTAL':<16s} {totals_j['rej']:>9d} {totals_j['tp']:>8d} "
        f"{totals_j['fp']:>9d} {'+' if net_total_j >= 0 else ''}{net_total_j:>5d}"
    )

    # -----------------------------------------------------------------------
    # Macro-average F1 impact
    # -----------------------------------------------------------------------
    print(f"\n  Macro-Average F1 Impact:")
    scenarios = [
        ("Actual (with both)", "prf_actual"),
        ("Without Phase 6", "prf_no_p6"),
        ("Without Phase 9", "prf_no_p9"),
        ("Without both", "prf_no_both"),
    ]
    print(f"    {'Scenario':<25s} {'Macro P':>8s} {'Macro R':>8s} {'Macro F1':>9s} {'Delta':>8s}")
    print(f"    {'─' * 60}")
    actual_macro_f1 = sum(r["prf_actual"]["F1"] for r in results) / len(results)
    for label, key in scenarios:
        macro_p = sum(r[key]["P"] for r in results) / len(results)
        macro_r = sum(r[key]["R"] for r in results) / len(results)
        macro_f1 = sum(r[key]["F1"] for r in results) / len(results)
        delta = macro_f1 - actual_macro_f1
        delta_str = f"{'+' if delta >= 0 else ''}{delta * 100:.2f}pp" if abs(delta) > 0.0005 else "---"
        print(f"    {label:<25s} {format_pct(macro_p):>8s} {format_pct(macro_r):>8s} {format_pct(macro_f1):>9s} {delta_str:>8s}")

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print("  VERDICT")
    print(f"{'=' * 100}")

    tp_killed_total = totals_p6["tp"] + totals_j["tp"]
    fp_caught_total = totals_p6["fp"] + totals_j["fp"]

    print(f"\n  Combined filters: {tp_killed_total} TPs killed, {fp_caught_total} FPs caught")
    if fp_caught_total > tp_killed_total:
        print(f"  Net: +{fp_caught_total - tp_killed_total} correct rejections (filters are net positive)")
    elif fp_caught_total < tp_killed_total:
        print(f"  Net: {fp_caught_total - tp_killed_total} (filters kill more TPs than FPs they catch)")
    else:
        print(f"  Net: 0 (filters are break-even)")

    # Per-filter verdict
    print(f"\n  Phase 6: {totals_p6['tp']} TPs killed, {totals_p6['fp']} FPs caught -> ", end="")
    if totals_p6["fp"] > totals_p6["tp"]:
        print(f"net positive (+{totals_p6['fp'] - totals_p6['tp']})")
    elif totals_p6["fp"] < totals_p6["tp"]:
        print(f"NET NEGATIVE ({totals_p6['fp'] - totals_p6['tp']})")
    else:
        print("break-even")

    print(f"  Phase 9: {totals_j['tp']} TPs killed, {totals_j['fp']} caught -> ", end="")
    if totals_j["fp"] > totals_j["tp"]:
        print(f"net positive (+{totals_j['fp'] - totals_j['tp']})")
    elif totals_j["fp"] < totals_j["tp"]:
        print(f"NET NEGATIVE ({totals_j['fp'] - totals_j['tp']})")
    else:
        print("break-even")

    print()


def main():
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))

    results = []
    for ds in DATASETS:
        checkpoint_dir = os.path.join(CHECKPOINT_DIR, ds)
        if not os.path.isdir(checkpoint_dir):
            print(f"SKIP {ds}: no checkpoint directory")
            continue
        # Check required files exist
        required = ["phase5.pkl", "phase6.pkl", "pre_judge.pkl", "final.pkl"]
        missing = [f for f in required if not os.path.isfile(os.path.join(checkpoint_dir, f))]
        if missing:
            print(f"SKIP {ds}: missing {missing}")
            continue

        results.append(analyze_dataset(ds))

    if not results:
        print("ERROR: No datasets had all required checkpoints.")
        sys.exit(1)

    print_results(results)


if __name__ == "__main__":
    main()
