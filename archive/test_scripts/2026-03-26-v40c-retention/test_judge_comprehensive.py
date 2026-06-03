#!/usr/bin/env python3
"""Comprehensive Phase 9 judge analysis for V39a.

Tests 4 scenarios across all 5 datasets:
  a) Current  — actual V39a output (post-judge)
  b) No judge — pre_judge preliminary links (skip Phase 9 entirely)
  c) Exempt coref — keep coref links regardless of judge decision
  d) Exempt coref+validated — keep coref AND validated links

Also analyzes each rejected link in detail and checks whether
judge-caught FPs could have been caught by other mechanisms.

No LLM calls. Uses only pickled checkpoints and gold standards.
"""

import csv
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── Setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from llm_sad_sam.core.data_types import SadSamLink

PROJECT_ROOT = Path(__file__).parent
PHASE_CACHE = PROJECT_ROOT / "results" / "phase_cache" / "v39a"
ABLATION_DIR = PROJECT_ROOT / "results" / "ablation_results"
BENCHMARK_DIR = Path("/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark")

DATASETS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]

# Mapping: dataset -> (gold standard CSV, text file)
GOLD_PATHS = {
    "mediastore": (
        BENCHMARK_DIR / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
        BENCHMARK_DIR / "mediastore/text_2016/mediastore.txt",
    ),
    "teastore": (
        BENCHMARK_DIR / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
        BENCHMARK_DIR / "teastore/text_2020/teastore.txt",
    ),
    "teammates": (
        BENCHMARK_DIR / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        BENCHMARK_DIR / "teammates/text_2021/teammates.txt",
    ),
    "bigbluebutton": (
        BENCHMARK_DIR / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        BENCHMARK_DIR / "bigbluebutton/text_2021/bigbluebutton.txt",
    ),
    "jabref": (
        BENCHMARK_DIR / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        BENCHMARK_DIR / "jabref/text_2021/jabref.txt",
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def load_gold_standard(csv_path):
    """Load gold standard as set of (sentence_number, component_id) tuples."""
    gold = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sent = int(row["sentence"])
            comp_id = row["modelElementID"]
            gold.add((sent, comp_id))
    return gold


def load_sentences(text_path):
    """Load sentences from text file (one per line, 1-indexed)."""
    sentences = {}
    with open(text_path) as f:
        for i, line in enumerate(f, 1):
            sentences[i] = line.strip()
    return sentences


def load_pre_judge(dataset):
    """Load pre_judge checkpoint: returns (preliminary_links, transarc_set)."""
    pkl_path = PHASE_CACHE / dataset / "pre_judge.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["preliminary"], data.get("transarc_set", set())


def load_final_pkl(dataset):
    """Load final.pkl: returns (final_links, reviewed_links, rejected_links)."""
    pkl_path = PHASE_CACHE / dataset / "final.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data.get("final", []), data.get("reviewed", []), data.get("rejected", [])


def load_output_csv(dataset):
    """Load final output CSV as set of (sentence, component_id) tuples."""
    csv_path = ABLATION_DIR / f"v39a_{dataset}_links.csv"
    links = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sent = int(row["sentence"])
            comp_id = row["component_id"]
            links.add((sent, comp_id))
    return links


def links_to_set(link_list):
    """Convert list of SadSamLink to set of (sentence_number, component_id)."""
    return {(l.sentence_number, l.component_id) for l in link_list}


def compute_metrics(predicted_set, gold_set):
    """Compute P, R, F1 given predicted and gold sets of (sent, comp_id)."""
    tp = len(predicted_set & gold_set)
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"P": p, "R": r, "F1": f1, "TP": tp, "FP": fp, "FN": fn}


# ═══════════════════════════════════════════════════════════════════════
# Convention filter / generic mention checks (offline replicas)
# ═══════════════════════════════════════════════════════════════════════

def has_standalone_mention(comp_name, sentence_text):
    """Check if the component name appears as a standalone word in the sentence."""
    return bool(re.search(rf'\b{re.escape(comp_name)}\b', sentence_text))


def is_generic_mention(comp_name, sentence_text):
    """Replica of V39a._is_generic_mention() for offline analysis."""
    if ' ' in comp_name or '-' in comp_name:
        return False
    if re.search(r'[a-z][A-Z]', comp_name):
        return False
    if comp_name.isupper():
        return False
    if comp_name[0].islower():
        return False
    if has_standalone_mention(comp_name, sentence_text):
        return False
    word_lower = comp_name.lower()
    if re.search(rf'\b{re.escape(word_lower)}\b', sentence_text):
        return True
    return False


def is_convention_filterable(comp_name, sentence_text):
    """Check if a link would be caught by convention filter heuristics.

    Convention filter catches links where the component name appears only
    in a dotted-path/qualified context or as a generic word, not as an
    architectural reference.
    """
    # Dotted path: "X.something" or "X/something"
    if re.search(rf'\b{re.escape(comp_name)}\.', sentence_text):
        if not has_standalone_mention(comp_name, sentence_text):
            return True, "dotted_path"

    # Generic word usage (lowercase only)
    if is_generic_mention(comp_name, sentence_text):
        return True, "generic_mention"

    return False, None


# ═══════════════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════════════

def build_scenarios(dataset):
    """Build 4 scenarios for a dataset. Returns dict of scenario -> predicted set."""
    prelim_links, transarc_set = load_pre_judge(dataset)
    final_links, reviewed_links, rejected_links = load_final_pkl(dataset)

    # Scenario a: Current (actual output)
    current_set = load_output_csv(dataset)

    # Scenario b: No judge at all (use preliminary links as-is)
    no_judge_set = links_to_set(prelim_links)

    # Scenario c: Exempt coref from judge
    # Start with current output + re-add any rejected coref links
    rejected_set = links_to_set(rejected_links)
    exempt_coref_set = set(current_set)  # copy
    for l in rejected_links:
        if l.source == "coreference":
            exempt_coref_set.add((l.sentence_number, l.component_id))

    # Scenario d: Exempt coref AND validated from judge
    exempt_coref_val_set = set(current_set)
    for l in rejected_links:
        if l.source in ("coreference", "validated"):
            exempt_coref_val_set.add((l.sentence_number, l.component_id))

    return {
        "a_current": current_set,
        "b_no_judge": no_judge_set,
        "c_exempt_coref": exempt_coref_set,
        "d_exempt_coref_val": exempt_coref_val_set,
    }


def analyze_rejected_links(dataset, sentences):
    """Analyze each rejected link in detail."""
    prelim_links, transarc_set = load_pre_judge(dataset)
    final_links, reviewed_links, rejected_links = load_final_pkl(dataset)
    gold = load_gold_standard(GOLD_PATHS[dataset][0])

    results = []
    for l in rejected_links:
        key = (l.sentence_number, l.component_id)
        is_tp = key in gold
        sent_text = sentences.get(l.sentence_number, "<not found>")

        # Check if convention filter would catch it
        conv_catchable, conv_reason = is_convention_filterable(l.component_name, sent_text)
        # Check generic mention
        generic = is_generic_mention(l.component_name, sent_text)
        # Check if it's in transarc_set
        is_ta = key in transarc_set

        results.append({
            "sentence": l.sentence_number,
            "component": l.component_name,
            "component_id": l.component_id,
            "source": l.source,
            "confidence": l.confidence,
            "is_TP": is_tp,
            "sent_text": sent_text,
            "convention_catchable": conv_catchable,
            "convention_reason": conv_reason,
            "generic_mention": generic,
            "is_transarc": is_ta,
        })
    return results


def main():
    print("=" * 100)
    print("COMPREHENSIVE PHASE 9 (4-RULE JUDGE) ANALYSIS — V39a")
    print("=" * 100)

    # ── Collect metrics for all scenarios across all datasets ──
    all_metrics = {}  # dataset -> scenario -> metrics
    all_rejected = {}  # dataset -> list of rejected link analyses

    for ds in DATASETS:
        gold = load_gold_standard(GOLD_PATHS[ds][0])
        sentences = load_sentences(GOLD_PATHS[ds][1])
        scenarios = build_scenarios(ds)

        all_metrics[ds] = {}
        for scenario_name, predicted_set in scenarios.items():
            m = compute_metrics(predicted_set, gold)
            all_metrics[ds][scenario_name] = m

        all_rejected[ds] = analyze_rejected_links(ds, sentences)

    # ════════════════════════════════════════════════════════════════════
    # SECTION 1: Comparison table
    # ════════════════════════════════════════════════════════════════════
    print("\n")
    print("=" * 100)
    print("SECTION 1: SCENARIO COMPARISON TABLE")
    print("=" * 100)

    scenario_names = ["a_current", "b_no_judge", "c_exempt_coref", "d_exempt_coref_val"]
    scenario_labels = {
        "a_current": "Current (baseline)",
        "b_no_judge": "No judge at all",
        "c_exempt_coref": "Exempt coref",
        "d_exempt_coref_val": "Exempt coref+validated",
    }

    for scenario in scenario_names:
        label = scenario_labels[scenario]
        print(f"\n--- {label} ---")
        print(f"{'Dataset':<16} {'P':>7} {'R':>7} {'F1':>7}   {'TP':>4} {'FP':>4} {'FN':>4}")
        print("-" * 60)

        sum_p, sum_r, sum_f1 = 0.0, 0.0, 0.0
        total_tp, total_fp, total_fn = 0, 0, 0

        for ds in DATASETS:
            m = all_metrics[ds][scenario]
            print(f"{ds:<16} {m['P']:>6.1%} {m['R']:>6.1%} {m['F1']:>6.1%}   {m['TP']:>4} {m['FP']:>4} {m['FN']:>4}")
            sum_p += m["P"]
            sum_r += m["R"]
            sum_f1 += m["F1"]
            total_tp += m["TP"]
            total_fp += m["FP"]
            total_fn += m["FN"]

        n = len(DATASETS)
        print("-" * 60)
        print(f"{'MACRO AVG':<16} {sum_p/n:>6.1%} {sum_r/n:>6.1%} {sum_f1/n:>6.1%}   {total_tp:>4} {total_fp:>4} {total_fn:>4}")

    # ════════════════════════════════════════════════════════════════════
    # SECTION 2: Side-by-side delta table
    # ════════════════════════════════════════════════════════════════════
    print("\n\n")
    print("=" * 100)
    print("SECTION 2: DELTA FROM CURRENT (positive = improvement)")
    print("=" * 100)

    for scenario in ["b_no_judge", "c_exempt_coref", "d_exempt_coref_val"]:
        label = scenario_labels[scenario]
        print(f"\n--- {label} vs Current ---")
        print(f"{'Dataset':<16} {'dP':>8} {'dR':>8} {'dF1':>8}   {'dTP':>5} {'dFP':>5} {'dFN':>5}")
        print("-" * 65)

        sum_dp, sum_dr, sum_df1 = 0.0, 0.0, 0.0
        for ds in DATASETS:
            cur = all_metrics[ds]["a_current"]
            alt = all_metrics[ds][scenario]
            dp = alt["P"] - cur["P"]
            dr = alt["R"] - cur["R"]
            df1 = alt["F1"] - cur["F1"]
            dtp = alt["TP"] - cur["TP"]
            dfp = alt["FP"] - cur["FP"]
            dfn = alt["FN"] - cur["FN"]
            print(f"{ds:<16} {dp:>+7.1%} {dr:>+7.1%} {df1:>+7.1%}   {dtp:>+5d} {dfp:>+5d} {dfn:>+5d}")
            sum_dp += dp
            sum_dr += dr
            sum_df1 += df1

        n = len(DATASETS)
        print("-" * 65)
        print(f"{'MACRO AVG':<16} {sum_dp/n:>+7.1%} {sum_dr/n:>+7.1%} {sum_df1/n:>+7.1%}")

    # ════════════════════════════════════════════════════════════════════
    # SECTION 3: Detailed rejected link analysis
    # ════════════════════════════════════════════════════════════════════
    print("\n\n")
    print("=" * 100)
    print("SECTION 3: DETAILED REJECTED LINK ANALYSIS")
    print("=" * 100)

    total_rejected = 0
    total_tp_killed = 0
    total_fp_caught = 0

    for ds in DATASETS:
        rejected = all_rejected[ds]
        if not rejected:
            print(f"\n[{ds}] No rejected links.")
            continue

        tp_killed = sum(1 for r in rejected if r["is_TP"])
        fp_caught = sum(1 for r in rejected if not r["is_TP"])
        total_rejected += len(rejected)
        total_tp_killed += tp_killed
        total_fp_caught += fp_caught

        print(f"\n[{ds}] {len(rejected)} rejected links: {tp_killed} TPs killed (BAD), {fp_caught} FPs caught (GOOD)")
        print("-" * 90)

        for r in rejected:
            status = "TP KILLED" if r["is_TP"] else "FP CAUGHT"
            print(f"  S{r['sentence']:<4} -> {r['component']:<25} src={r['source']:<15} [{status}]")
            # Truncate long sentences for readability
            sent_display = r["sent_text"][:120] + ("..." if len(r["sent_text"]) > 120 else "")
            print(f"         Sentence: \"{sent_display}\"")
            print(f"         TransArc: {r['is_transarc']}, GenericMention: {r['generic_mention']}, "
                  f"ConventionCatchable: {r['convention_catchable']} ({r['convention_reason']})")
            print()

    print(f"\nTOTAL across all datasets: {total_rejected} rejected, "
          f"{total_tp_killed} TPs killed, {total_fp_caught} FPs caught")
    print(f"Net judge value: {total_fp_caught} FPs caught - {total_tp_killed} TPs killed = "
          f"{total_fp_caught - total_tp_killed:+d} net links improved")

    # ════════════════════════════════════════════════════════════════════
    # SECTION 4: Could FPs be caught by other mechanisms?
    # ════════════════════════════════════════════════════════════════════
    print("\n\n")
    print("=" * 100)
    print("SECTION 4: ALTERNATIVE MECHANISMS FOR JUDGE-CAUGHT FPs")
    print("=" * 100)

    all_fp_caught = []
    for ds in DATASETS:
        for r in all_rejected[ds]:
            if not r["is_TP"]:
                r["dataset"] = ds
                all_fp_caught.append(r)

    if not all_fp_caught:
        print("\nNo FPs caught by judge — nothing to analyze.")
    else:
        print(f"\n{len(all_fp_caught)} FPs caught by judge. Could other mechanisms catch them?\n")
        print(f"{'Dataset':<16} {'S#':<5} {'Component':<25} {'Source':<15} {'Convention?':<14} {'Generic?':<10} {'TransArc?':<10}")
        print("-" * 100)

        conv_count = 0
        generic_count = 0
        ta_count = 0

        for r in all_fp_caught:
            conv_str = f"Yes ({r['convention_reason']})" if r["convention_catchable"] else "No"
            print(f"{r['dataset']:<16} S{r['sentence']:<4} {r['component']:<25} {r['source']:<15} "
                  f"{conv_str:<14} {'Yes' if r['generic_mention'] else 'No':<10} {'Yes' if r['is_transarc'] else 'No':<10}")
            if r["convention_catchable"]:
                conv_count += 1
            if r["generic_mention"]:
                generic_count += 1
            if r["is_transarc"]:
                ta_count += 1

        print(f"\nSummary:")
        print(f"  Convention filter would catch: {conv_count}/{len(all_fp_caught)}")
        print(f"  _is_generic_mention() would catch: {generic_count}/{len(all_fp_caught)}")
        print(f"  Are TransArc source (removing TA immunity would help): {ta_count}/{len(all_fp_caught)}")

        if conv_count == len(all_fp_caught):
            print("\n  --> ALL judge-caught FPs are also convention-catchable. Judge is REDUNDANT for FP catching.")
        elif conv_count > 0:
            print(f"\n  --> {conv_count}/{len(all_fp_caught)} overlap. Judge catches {len(all_fp_caught) - conv_count} FPs that convention filter misses.")
        else:
            print("\n  --> Convention filter catches NONE of these. Judge is the ONLY mechanism for these FPs.")

    # ════════════════════════════════════════════════════════════════════
    # SECTION 5: Per-source breakdown of judge decisions
    # ════════════════════════════════════════════════════════════════════
    print("\n\n")
    print("=" * 100)
    print("SECTION 5: JUDGE DECISIONS BY LINK SOURCE")
    print("=" * 100)

    # Aggregate across all datasets
    source_stats = defaultdict(lambda: {"submitted": 0, "approved": 0, "rejected_tp": 0, "rejected_fp": 0})

    for ds in DATASETS:
        prelim_links, transarc_set = load_pre_judge(ds)
        final_links, reviewed_links, rejected_links = load_final_pkl(ds)
        gold = load_gold_standard(GOLD_PATHS[ds][0])

        # Count what went through judge (non-safe links)
        for l in prelim_links:
            source_stats[l.source]["submitted"] += 1

        approved_set = links_to_set(reviewed_links)
        for l in reviewed_links:
            source_stats[l.source]["approved"] += 1

        for l in rejected_links:
            key = (l.sentence_number, l.component_id)
            if key in gold:
                source_stats[l.source]["rejected_tp"] += 1
            else:
                source_stats[l.source]["rejected_fp"] += 1

    print(f"\n{'Source':<16} {'Submitted':>10} {'Approved':>10} {'Rej(TP)':>10} {'Rej(FP)':>10} {'Rej Rate':>10}")
    print("-" * 70)
    for src in sorted(source_stats.keys()):
        s = source_stats[src]
        total_rej = s["rejected_tp"] + s["rejected_fp"]
        rej_rate = total_rej / s["submitted"] if s["submitted"] > 0 else 0
        print(f"{src:<16} {s['submitted']:>10} {s['approved']:>10} {s['rejected_tp']:>10} {s['rejected_fp']:>10} {rej_rate:>9.1%}")

    # ════════════════════════════════════════════════════════════════════
    # SECTION 6: Verdict and recommendation
    # ════════════════════════════════════════════════════════════════════
    print("\n\n")
    print("=" * 100)
    print("SECTION 6: VERDICT AND RECOMMENDATION")
    print("=" * 100)

    # Compute macro F1 for each scenario
    macro_f1 = {}
    for scenario in scenario_names:
        f1_sum = sum(all_metrics[ds][scenario]["F1"] for ds in DATASETS)
        macro_f1[scenario] = f1_sum / len(DATASETS)

    print(f"\nMacro F1 comparison:")
    for scenario in scenario_names:
        label = scenario_labels[scenario]
        delta = macro_f1[scenario] - macro_f1["a_current"]
        delta_str = f" ({delta:+.2%})" if scenario != "a_current" else ""
        print(f"  {label:<30} {macro_f1[scenario]:.2%}{delta_str}")

    best_scenario = max(macro_f1, key=macro_f1.get)
    print(f"\nBest scenario: {scenario_labels[best_scenario]} ({macro_f1[best_scenario]:.2%})")

    # Check if exempting coref is a Pareto improvement
    cur_m = {"TP": 0, "FP": 0, "FN": 0}
    exc_m = {"TP": 0, "FP": 0, "FN": 0}
    for ds in DATASETS:
        for k in ("TP", "FP", "FN"):
            cur_m[k] += all_metrics[ds]["a_current"][k]
            exc_m[k] += all_metrics[ds]["c_exempt_coref"][k]

    coref_rejected_tp = sum(1 for ds in DATASETS for r in all_rejected[ds] if r["is_TP"] and r["source"] == "coreference")
    coref_rejected_fp = sum(1 for ds in DATASETS for r in all_rejected[ds] if not r["is_TP"] and r["source"] == "coreference")

    print(f"\nCoref exemption analysis:")
    print(f"  Coref links rejected by judge: {coref_rejected_tp + coref_rejected_fp}")
    print(f"    - TPs killed: {coref_rejected_tp}")
    print(f"    - FPs caught: {coref_rejected_fp}")

    if coref_rejected_fp == 0 and coref_rejected_tp > 0:
        print(f"\n  VERDICT: Exempting coref from judge is a PURE WIN.")
        print(f"  It saves {coref_rejected_tp} TPs and loses 0 FPs.")
        print(f"  Macro F1: {macro_f1['a_current']:.2%} -> {macro_f1['c_exempt_coref']:.2%} "
              f"({macro_f1['c_exempt_coref'] - macro_f1['a_current']:+.2%})")
    elif coref_rejected_fp > 0:
        print(f"\n  VERDICT: Exempting coref has a trade-off.")
        print(f"  Saves {coref_rejected_tp} TPs but also re-admits {coref_rejected_fp} FPs.")
    else:
        print(f"\n  VERDICT: No coref links were rejected. Exemption has no effect.")

    # Check validated exemption additionally
    val_rejected_tp = sum(1 for ds in DATASETS for r in all_rejected[ds] if r["is_TP"] and r["source"] == "validated")
    val_rejected_fp = sum(1 for ds in DATASETS for r in all_rejected[ds] if not r["is_TP"] and r["source"] == "validated")

    print(f"\nValidated exemption analysis:")
    print(f"  Validated links rejected by judge: {val_rejected_tp + val_rejected_fp}")
    print(f"    - TPs killed: {val_rejected_tp}")
    print(f"    - FPs caught: {val_rejected_fp}")

    if val_rejected_tp == 0 and val_rejected_fp > 0:
        print(f"\n  Adding validated exemption would RE-ADMIT {val_rejected_fp} FPs with 0 TP gain.")
        print(f"  NOT recommended.")
    elif val_rejected_tp > 0 and val_rejected_fp == 0:
        print(f"\n  Adding validated exemption saves {val_rejected_tp} TPs with 0 FP cost.")
        print(f"  Recommended.")
    else:
        print(f"\n  Trade-off: {val_rejected_tp} TPs saved vs {val_rejected_fp} FPs re-admitted.")

    print(f"\n{'=' * 100}")
    print("END OF ANALYSIS")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
