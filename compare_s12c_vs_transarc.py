#!/usr/bin/env python3
"""Generate CSV comparing S-Linker12c, S-Linker12e vs TransArc across all 5 datasets."""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent
BENCHMARK = ROOT / "../ardoco/core/tests-base/src/main/resources/benchmark"
CLI_RESULTS = Path("/mnt/hostshare/ardoco-home/cli-results")
ABLATION = ROOT / "results/ablation_results"

DATASETS = {
    "mediastore": {
        "gold": BENCHMARK / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
        "transarc": CLI_RESULTS / "mediastore-sad-sam/sadSamTlr_mediastore.csv",
        "s12c": ABLATION / "s_linker12c_mediastore_links.csv",
        "s12e": ABLATION / "s_linker12e_mediastore_links.csv",
    },
    "teastore": {
        "gold": BENCHMARK / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
        "transarc": CLI_RESULTS / "teastore-sad-sam/sadSamTlr_teastore.csv",
        "s12c": ABLATION / "s_linker12c_teastore_links.csv",
        "s12e": ABLATION / "s_linker12e_teastore_links.csv",
    },
    "teammates": {
        "gold": BENCHMARK / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc": CLI_RESULTS / "teammates-sad-sam/sadSamTlr_teammates.csv",
        "s12c": ABLATION / "s_linker12c_teammates_links.csv",
        "s12e": ABLATION / "s_linker12e_teammates_links.csv",
    },
    "bigbluebutton": {
        "gold": BENCHMARK / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc": CLI_RESULTS / "bigbluebutton-sad-sam/sadSamTlr_bigbluebutton.csv",
        "s12c": ABLATION / "s_linker12c_bigbluebutton_links.csv",
        "s12e": ABLATION / "s_linker12e_bigbluebutton_links.csv",
    },
    "jabref": {
        "gold": BENCHMARK / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc": CLI_RESULTS / "jabref-sad-sam/sadSamTlr_jabref.csv",
        "s12c": ABLATION / "s_linker12c_jabref_links.csv",
        "s12e": ABLATION / "s_linker12e_jabref_links.csv",
    },
}


def load_gold(path: Path) -> set[tuple[int, str]]:
    links = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            mid = row["modelElementID"].strip()
            sent = row["sentence"].strip()
            if mid and sent:
                links.add((int(sent), mid))
    return links


def load_transarc(path: Path) -> set[tuple[int, str]]:
    links = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            mid = row["modelElementID"].strip()
            sent = row["sentence"].strip()
            if mid and sent:
                links.add((int(sent), mid))
    return links


def load_s12c(path: Path) -> set[tuple[int, str]]:
    """S-Linker12c CSV: sentence,component_id,component_name,confidence,source"""
    links = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            mid = row["component_id"].strip()
            sent = row["sentence"].strip()
            if mid and sent:
                links.add((int(sent), mid))
    return links


def metrics(predicted: set, gold: set) -> dict:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "Precision": p, "Recall": r, "F1": f1}


def main():
    rows = []
    transarc_f1s = []
    s12c_f1s = []
    s12e_f1s = []

    for dataset, paths in DATASETS.items():
        gold = load_gold(paths["gold"])
        ta = load_transarc(paths["transarc"])
        s12c = load_s12c(paths["s12c"])
        s12e = load_s12c(paths["s12e"])

        ta_m = metrics(ta, gold)
        s12c_m = metrics(s12c, gold)
        s12e_m = metrics(s12e, gold)

        transarc_f1s.append(ta_m["F1"])
        s12c_f1s.append(s12c_m["F1"])
        s12e_f1s.append(s12e_m["F1"])

        rows.append({
            "dataset": dataset,
            "gold_links": len(gold),
            "transarc_links": len(ta),
            "transarc_TP": ta_m["TP"],
            "transarc_FP": ta_m["FP"],
            "transarc_FN": ta_m["FN"],
            "transarc_P": round(ta_m["Precision"] * 100, 1),
            "transarc_R": round(ta_m["Recall"] * 100, 1),
            "transarc_F1": round(ta_m["F1"] * 100, 1),
            "s12c_links": len(s12c),
            "s12c_TP": s12c_m["TP"],
            "s12c_FP": s12c_m["FP"],
            "s12c_FN": s12c_m["FN"],
            "s12c_P": round(s12c_m["Precision"] * 100, 1),
            "s12c_R": round(s12c_m["Recall"] * 100, 1),
            "s12c_F1": round(s12c_m["F1"] * 100, 1),
            "s12c_delta_F1": round((s12c_m["F1"] - ta_m["F1"]) * 100, 1),
            "s12e_links": len(s12e),
            "s12e_TP": s12e_m["TP"],
            "s12e_FP": s12e_m["FP"],
            "s12e_FN": s12e_m["FN"],
            "s12e_P": round(s12e_m["Precision"] * 100, 1),
            "s12e_R": round(s12e_m["Recall"] * 100, 1),
            "s12e_F1": round(s12e_m["F1"] * 100, 1),
            "s12e_delta_F1": round((s12e_m["F1"] - ta_m["F1"]) * 100, 1),
        })

    # Macro average row
    rows.append({
        "dataset": "MACRO_AVG",
        "gold_links": "",
        "transarc_links": "",
        "transarc_TP": "",
        "transarc_FP": "",
        "transarc_FN": "",
        "transarc_P": "",
        "transarc_R": "",
        "transarc_F1": round(sum(transarc_f1s) / len(transarc_f1s) * 100, 1),
        "s12c_links": "",
        "s12c_TP": "",
        "s12c_FP": "",
        "s12c_FN": "",
        "s12c_P": "",
        "s12c_R": "",
        "s12c_F1": round(sum(s12c_f1s) / len(s12c_f1s) * 100, 1),
        "s12c_delta_F1": round((sum(s12c_f1s) - sum(transarc_f1s)) / len(s12c_f1s) * 100, 1),
        "s12e_links": "",
        "s12e_TP": "",
        "s12e_FP": "",
        "s12e_FN": "",
        "s12e_P": "",
        "s12e_R": "",
        "s12e_F1": round(sum(s12e_f1s) / len(s12e_f1s) * 100, 1),
        "s12e_delta_F1": round((sum(s12e_f1s) - sum(transarc_f1s)) / len(s12e_f1s) * 100, 1),
    })

    out = ROOT / "results/s12c_vs_transarc.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Written: {out}")

    # Print summary table
    print(f"\n{'Dataset':<16} {'TransArc':>22} {'S12c':>32} {'S12e':>32}")
    print(f"{'':16} {'P':>7} {'R':>7} {'F1':>7}  {'P':>7} {'R':>7} {'F1':>7} {'ΔF1':>6}  {'P':>7} {'R':>7} {'F1':>7} {'ΔF1':>6}")
    print("-" * 114)
    for row in rows[:-1]:
        print(
            f"{row['dataset']:<16}"
            f" {row['transarc_P']:>6}% {row['transarc_R']:>6}% {row['transarc_F1']:>6}%"
            f"  {row['s12c_P']:>6}% {row['s12c_R']:>6}% {row['s12c_F1']:>6}% {row['s12c_delta_F1']:>+5}pp"
            f"  {row['s12e_P']:>6}% {row['s12e_R']:>6}% {row['s12e_F1']:>6}% {row['s12e_delta_F1']:>+5}pp"
        )
    macro = rows[-1]
    print("-" * 114)
    print(
        f"{'Macro avg':<16}"
        f" {'':>7} {'':>7} {macro['transarc_F1']:>6}%"
        f"  {'':>7} {'':>7} {macro['s12c_F1']:>6}% {macro['s12c_delta_F1']:>+5}pp"
        f"  {'':>7} {'':>7} {macro['s12e_F1']:>6}% {macro['s12e_delta_F1']:>+5}pp"
    )


if __name__ == "__main__":
    main()
