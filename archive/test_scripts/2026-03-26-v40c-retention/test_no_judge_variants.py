#!/usr/bin/env python3
"""Test what happens if we remove the judge (Phase 9) but instead provide
better context/filtering to the phases that produce the links.

Loads V39 pre_judge checkpoints (links before Phase 9), applies various
no-judge strategies offline, and compares against gold standard.

Variants:
  baseline  — V39 final (with judge) for reference
  no_judge  — Drop the judge entirely (pre_judge links as final)
  conv_only — Drop judge, rely solely on convention filter (already applied in pre_judge)
  strict_antecedent — Drop judge, but tighten antecedent verification for coref links
  source_confidence — Drop judge, but apply source-aware confidence thresholds

Usage:
    python test_no_judge_variants.py
    python test_no_judge_variants.py --datasets mediastore bigbluebutton
"""

import csv
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.core.data_types import SadSamLink
from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.pcm_parser import parse_pcm_repository

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
DATASETS = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
        "gold": BENCHMARK_BASE / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
        "model": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
        "gold": BENCHMARK_BASE / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
        "gold": BENCHMARK_BASE / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
        "gold": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
        "gold": BENCHMARK_BASE / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
}
CACHE_DIR = Path("./results/phase_cache/v39")


def load_checkpoint(dataset, phase_name):
    path = CACHE_DIR / dataset / f"{phase_name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_gold(gold_path):
    links = set()
    with open(gold_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def eval_links(links, gold):
    predicted = {(l.sentence_number, l.component_id) for l in links}
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"tp": tp, "fp": fp, "fn": fn, "P": p, "R": r, "F1": f1}


def has_standalone_mention(comp_name, text):
    is_single = ' ' not in comp_name
    if is_single:
        cap_name = comp_name[0].upper() + comp_name[1:]
        pattern = rf'\b{re.escape(cap_name)}\b'
        flags = 0
    else:
        pattern = rf'\b{re.escape(comp_name)}\b'
        flags = re.IGNORECASE
    for m in re.finditer(pattern, text, flags):
        s, e = m.start(), m.end()
        if s > 0 and text[s-1] == '.':
            continue
        if e < len(text) and text[e] == '.' and e + 1 < len(text) and text[e+1].isalpha():
            continue
        if s > 0 and text[s-1] == '-':
            continue
        if e < len(text) and text[e] == '-' and '-' not in comp_name:
            continue
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════
# Variant: no_judge — just drop Phase 9 entirely
# ══════════════════════════════════════════════════════════════════════════

def variant_no_judge(pre_judge_links, **kwargs):
    """No judge at all — pre_judge links are final."""
    return list(pre_judge_links)


# ══════════════════════════════════════════════════════════════════════════
# Variant: source_filter — remove links from unreliable sources that
# don't have standalone mention AND aren't transarc
# ══════════════════════════════════════════════════════════════════════════

def variant_source_filter(pre_judge_links, sent_map=None, doc_knowledge=None, **kwargs):
    """Drop validated/coreference links where component not standalone-mentioned
    and source is low-priority (coreference, entity)."""
    result = []
    for l in pre_judge_links:
        # TransArc and partial_inject are always kept
        if l.source in ("transarc", "partial_inject"):
            result.append(l)
            continue
        # Validated with explicit mention — keep
        sent = sent_map.get(l.sentence_number)
        if sent and has_standalone_mention(l.component_name, sent.text):
            result.append(l)
            continue
        # Validated with alias — keep if alias exists in doc_knowledge
        if sent and doc_knowledge:
            text_lower = sent.text.lower()
            has_alias = False
            for syn, target in doc_knowledge.synonyms.items():
                if target == l.component_name:
                    if re.search(rf'\b{re.escape(syn.lower())}\b', text_lower):
                        has_alias = True
                        break
            if not has_alias:
                for partial, target in doc_knowledge.partial_references.items():
                    if target == l.component_name:
                        if re.search(rf'\b{re.escape(partial.lower())}\b', text_lower):
                            has_alias = True
                            break
            if has_alias:
                result.append(l)
                continue
        # No standalone mention and no alias — this is what the judge would review
        # Drop it (this is the conservative "no judge" approach)
        print(f"    source_filter drop: S{l.sentence_number} -> {l.component_name} (src={l.source})")
        # Actually, let's keep it but flag it — we want to measure what pure no-judge does
        # vs dropping only no-match links
    return result


def variant_drop_nomatch(pre_judge_links, sent_map=None, doc_knowledge=None, **kwargs):
    """Drop links where component name is not found in the sentence at all
    (neither standalone, nor as alias). These are the 'nomatch' links that the
    judge would review. Instead of judging, just drop them."""
    result = []
    dropped = []
    for l in pre_judge_links:
        sent = sent_map.get(l.sentence_number)
        if not sent:
            dropped.append(l)
            continue

        # Check standalone mention
        if has_standalone_mention(l.component_name, sent.text):
            result.append(l)
            continue

        # Check alias mention
        has_alias = False
        if doc_knowledge:
            text_lower = sent.text.lower()
            for syn, target in doc_knowledge.synonyms.items():
                if target == l.component_name:
                    if re.search(rf'\b{re.escape(syn.lower())}\b', text_lower):
                        has_alias = True
                        break
            if not has_alias:
                for partial, target in doc_knowledge.partial_references.items():
                    if target == l.component_name:
                        if re.search(rf'\b{re.escape(partial.lower())}\b', text_lower):
                            has_alias = True
                            break

        if has_alias:
            result.append(l)
            continue

        # Check TransArc — always keep (TransArc immunity)
        transarc_set = kwargs.get("transarc_set", set())
        if (l.sentence_number, l.component_id) in transarc_set:
            result.append(l)
            continue

        # No match found — drop instead of judging
        dropped.append(l)

    if dropped:
        print(f"    drop_nomatch: dropped {len(dropped)} links")
        for l in dropped:
            sent = sent_map.get(l.sentence_number)
            text = sent.text[:60] if sent else "?"
            print(f"      S{l.sentence_number} -> {l.component_name} (src={l.source}): {text}")
    return result


def variant_drop_nomatch_keep_coref(pre_judge_links, sent_map=None, doc_knowledge=None, **kwargs):
    """Like drop_nomatch, but KEEP coreference links (they were validated by
    antecedent verification already — no need for judge double-check).
    This tests: 'coref antecedent verification IS the context protection'."""
    result = []
    dropped = []
    for l in pre_judge_links:
        # Always keep coref links — antecedent verification is their context
        if l.source == "coreference":
            result.append(l)
            continue

        sent = sent_map.get(l.sentence_number)
        if not sent:
            dropped.append(l)
            continue

        if has_standalone_mention(l.component_name, sent.text):
            result.append(l)
            continue

        has_alias = False
        if doc_knowledge:
            text_lower = sent.text.lower()
            for syn, target in doc_knowledge.synonyms.items():
                if target == l.component_name:
                    if re.search(rf'\b{re.escape(syn.lower())}\b', text_lower):
                        has_alias = True
                        break
            if not has_alias:
                for partial, target in doc_knowledge.partial_references.items():
                    if target == l.component_name:
                        if re.search(rf'\b{re.escape(partial.lower())}\b', text_lower):
                            has_alias = True
                            break

        if has_alias:
            result.append(l)
            continue

        transarc_set = kwargs.get("transarc_set", set())
        if (l.sentence_number, l.component_id) in transarc_set:
            result.append(l)
            continue

        dropped.append(l)

    if dropped:
        print(f"    drop_nomatch_keep_coref: dropped {len(dropped)} links")
    return result


def variant_activity_partial_drop(pre_judge_links, sent_map=None, doc_knowledge=None,
                                   activity_partials=None, **kwargs):
    """Drop only activity-type partial links (the ones the calibrated judge targets).
    These are single-word generic partials classified as ORDINARY by Phase 3c.
    All other links pass through — no judge needed."""
    if not activity_partials:
        return list(pre_judge_links)

    result = []
    dropped = []
    for l in pre_judge_links:
        sent = sent_map.get(l.sentence_number)
        if not sent or not doc_knowledge:
            result.append(l)
            continue

        # Check if match is via an activity-type partial
        is_activity = False
        text_lower = sent.text.lower()
        for partial, target in doc_knowledge.partial_references.items():
            if target == l.component_name and partial in activity_partials:
                if re.search(rf'\b{re.escape(partial.lower())}\b', text_lower):
                    is_activity = True
                    break

        if is_activity:
            dropped.append(l)
        else:
            result.append(l)

    if dropped:
        print(f"    activity_partial_drop: dropped {len(dropped)} links")
        for l in dropped:
            print(f"      S{l.sentence_number} -> {l.component_name} (src={l.source})")
    return result


# ══════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════

ALL_VARIANTS = {
    "with_judge":      ("V39 final (with judge)", None),  # loaded from checkpoint
    "no_judge":        ("Drop judge entirely", variant_no_judge),
    "drop_nomatch":    ("Drop no-match links (no judge)", variant_drop_nomatch),
    "keep_coref":      ("Drop no-match except coref (antecedent=context)", variant_drop_nomatch_keep_coref),
    "activity_drop":   ("Drop activity-partial links only", variant_activity_partial_drop),
}


def run_test(dataset_names, variant_names):
    results = {}

    for ds_name in dataset_names:
        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*70}")

        # Load data
        paths = DATASETS[ds_name]
        components = parse_pcm_repository(str(paths["model"]))
        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        sent_map = DocumentLoader.build_sent_map(sentences)
        name_to_id = {c.name: c.id for c in components}
        gold = load_gold(paths["gold"])

        # Load checkpoints
        pre_judge_data = load_checkpoint(ds_name, "pre_judge")
        final_data = load_checkpoint(ds_name, "final")
        p3_data = load_checkpoint(ds_name, "phase3")

        if not pre_judge_data or not final_data:
            print(f"  Missing checkpoints for {ds_name}")
            continue

        pre_judge_links = pre_judge_data["preliminary"]
        final_links = final_data["final"]
        transarc_set = pre_judge_data.get("transarc_set", set())
        doc_knowledge = p3_data["doc_knowledge"] if p3_data else None
        activity_partials = p3_data.get("activity_partials", set()) if p3_data else set()

        # Analyze judge kills
        pre_set = {(l.sentence_number, l.component_id): l for l in pre_judge_links}
        final_set = {(l.sentence_number, l.component_id) for l in final_links}
        killed = [l for key, l in pre_set.items() if key not in final_set]

        killed_tp = [l for l in killed if (l.sentence_number, l.component_id) in gold]
        killed_fp = [l for l in killed if (l.sentence_number, l.component_id) not in gold]

        print(f"  Pre-judge: {len(pre_judge_links)} links")
        print(f"  Post-judge: {len(final_links)} links")
        print(f"  Judge killed: {len(killed)} ({len(killed_fp)} FP, {len(killed_tp)} TP)")

        if killed_fp:
            for l in killed_fp:
                sent = sent_map.get(l.sentence_number)
                text = sent.text[:70] if sent else "?"
                print(f"    [FP KILLED] S{l.sentence_number} -> {l.component_name} ({l.source}): {text}")
        if killed_tp:
            for l in killed_tp:
                sent = sent_map.get(l.sentence_number)
                text = sent.text[:70] if sent else "?"
                print(f"    [TP KILLED!] S{l.sentence_number} -> {l.component_name} ({l.source}): {text}")

        # Surviving FPs
        surviving_fps = [l for l in final_links
                        if (l.sentence_number, l.component_id) not in gold]
        if surviving_fps:
            print(f"  Surviving FPs: {len(surviving_fps)}")
            for l in surviving_fps:
                sent = sent_map.get(l.sentence_number)
                text = sent.text[:70] if sent else "?"
                is_ta = (l.sentence_number, l.component_id) in transarc_set
                ta_label = " [TransArc]" if is_ta else ""
                print(f"    [FP SURVIVES] S{l.sentence_number} -> {l.component_name} ({l.source}{ta_label}): {text}")

        # Run variants
        for var_name in variant_names:
            if var_name not in ALL_VARIANTS:
                continue

            desc, func = ALL_VARIANTS[var_name]
            print(f"\n  --- {var_name}: {desc} ---")

            if var_name == "with_judge":
                var_links = final_links
            else:
                var_links = func(
                    pre_judge_links,
                    sent_map=sent_map,
                    doc_knowledge=doc_knowledge,
                    transarc_set=transarc_set,
                    activity_partials=activity_partials,
                )

            m = eval_links(var_links, gold)
            print(f"  Links: {len(var_links)} | TP={m['tp']} FP={m['fp']} FN={m['fn']} | "
                  f"P={m['P']:.1%} R={m['R']:.1%} F1={m['F1']:.1%}")

            results[(ds_name, var_name)] = m

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print("SUMMARY — F1 per dataset")
    print(f"{'='*100}")

    header = f"{'Variant':<18}"
    for ds in dataset_names:
        header += f" | {ds[:10]:>10}"
    header += " | {'Macro F1':>10} | {'TP':>4} {'FP':>4} {'FN':>4}"
    print(header)
    print("-" * len(header))

    for var_name in variant_names:
        if var_name not in ALL_VARIANTS:
            continue
        row = f"{var_name:<18}"
        f1_vals = []
        total_tp, total_fp, total_fn = 0, 0, 0
        for ds in dataset_names:
            key = (ds, var_name)
            if key in results:
                m = results[key]
                row += f" | {m['F1']:>9.1%}"
                f1_vals.append(m["F1"])
                total_tp += m["tp"]
                total_fp += m["fp"]
                total_fn += m["fn"]
            else:
                row += f" | {'N/A':>10}"
        if f1_vals:
            macro = sum(f1_vals) / len(f1_vals)
            row += f" | {macro:>9.1%}  | {total_tp:>4} {total_fp:>4} {total_fn:>4}"
        print(row)

    # Delta vs with_judge
    print(f"\n{'Variant':<18} | {'Delta vs judge':>14} | {'TP delta':>8} | {'FP delta':>8}")
    print("-" * 60)
    judge_tp = sum(results.get((ds, "with_judge"), {}).get("tp", 0) for ds in dataset_names)
    judge_fp = sum(results.get((ds, "with_judge"), {}).get("fp", 0) for ds in dataset_names)
    judge_f1s = [results.get((ds, "with_judge"), {}).get("F1", 0) for ds in dataset_names]
    judge_macro = sum(judge_f1s) / len(judge_f1s) if judge_f1s else 0

    for var_name in variant_names:
        if var_name == "with_judge" or var_name not in ALL_VARIANTS:
            continue
        var_tp = sum(results.get((ds, var_name), {}).get("tp", 0) for ds in dataset_names)
        var_fp = sum(results.get((ds, var_name), {}).get("fp", 0) for ds in dataset_names)
        var_f1s = [results.get((ds, var_name), {}).get("F1", 0) for ds in dataset_names]
        var_macro = sum(var_f1s) / len(var_f1s) if var_f1s else 0

        delta_macro = var_macro - judge_macro
        delta_tp = var_tp - judge_tp
        delta_fp = var_fp - judge_fp
        print(f"{var_name:<18} | {delta_macro:>+13.1%} | {delta_tp:>+8} | {delta_fp:>+8}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"])
    parser.add_argument("--variants", nargs="+",
                        default=list(ALL_VARIANTS.keys()))
    args = parser.parse_args()
    run_test(args.datasets, args.variants)
