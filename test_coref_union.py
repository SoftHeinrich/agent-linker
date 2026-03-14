#!/usr/bin/env python3
"""Test: Does 2x union coref improve S-Linker3 results?

Loads S-Linker3 checkpoints, re-runs coref twice, takes the union,
replays full Tier 3 pipeline, compares against gold standard.

Zero LLM calls for Tier 3 replay — only coref passes use LLM.

Usage:
    python test_coref_union.py
    python test_coref_union.py --datasets mediastore teastore
"""

import argparse
import csv
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load .env
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

os.environ.setdefault("CLAUDE_MODEL", "sonnet")

from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.core import DocumentLoader, SadSamLink
from llm_sad_sam.linkers.experimental.s_linker3 import SLinker3

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
CHECKPOINT_BASE = Path("results/phase_cache/s_linker3")

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


def load_gold(gold_path):
    """Load gold standard as set of (sentence_number, component_id)."""
    gold = set()
    with open(gold_path) as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            snum = int(row["sentence"].strip())
            cid = row["modelElementID"].strip()
            gold.add((snum, cid))
    return gold


def evaluate(links, gold):
    """Compute P/R/F1 vs gold standard."""
    predicted = {(l.sentence_number, l.component_id) for l in links}
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1, tp, fp, fn


def replay_tier3(linker, seed_links, validated, coref_links, partial_links,
                 seed_set, sentences, components, sent_map):
    """Replay Tier 3 deterministic pipeline: merge + filter + keep_coref."""
    entity_links = [
        SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
        for c in validated
    ]
    all_links = seed_links + entity_links + coref_links + partial_links
    link_map = {}
    for lk in all_links:
        key = (lk.sentence_number, lk.component_id)
        if key not in link_map:
            link_map[key] = lk
        else:
            old_p = linker.SOURCE_PRIORITY.get(link_map[key].source, 0)
            new_p = linker.SOURCE_PRIORITY.get(lk.source, 0)
            if new_p > old_p:
                link_map[key] = lk
    preliminary = list(link_map.values())

    # Parent-overlap guard
    if linker.model_knowledge and linker.model_knowledge.impl_to_abstract:
        child_to_parent = linker.model_knowledge.impl_to_abstract
        sent_comps = defaultdict(set)
        for lk in preliminary:
            sent_comps[lk.sentence_number].add(lk.component_name)
        filtered_po = []
        for lk in preliminary:
            parent = child_to_parent.get(lk.component_name)
            if parent and parent in sent_comps[lk.sentence_number]:
                pass  # dropped
            else:
                filtered_po.append(lk)
        preliminary = filtered_po

    # Boundary filter
    preliminary, _ = linker._apply_boundary_filters(preliminary, sent_map, seed_set)

    # Keep-coref deterministic filter
    kept = []
    for l in preliminary:
        if l.source == "coreference":
            kept.append(l)
            continue
        is_seed = (l.sentence_number, l.component_id) in seed_set
        if is_seed:
            kept.append(l)
            continue
        sent = sent_map.get(l.sentence_number)
        if not sent:
            continue
        if linker._has_standalone_mention(l.component_name, sent.text):
            kept.append(l)
        elif linker._has_alias_mention(l.component_name, sent.text):
            kept.append(l)

    return kept


def run_test(dataset_name, cfg):
    """Run single-pass vs 2x-union coref comparison for one dataset."""
    print(f"\n{'='*70}")
    print(f"  {dataset_name}")
    print(f"{'='*70}")

    ckpt_dir = CHECKPOINT_BASE / dataset_name
    if not ckpt_dir.exists():
        print(f"  SKIP: no checkpoints at {ckpt_dir}")
        return None

    # Load raw data
    components = parse_pcm_repository(str(cfg["model"]))
    sentences = DocumentLoader.load_sentences(str(cfg["text"]))
    name_to_id = {c.name: c.id for c in components}
    sent_map = DocumentLoader.build_sent_map(sentences)
    gold = load_gold(cfg["gold"])

    # Load checkpoints
    with open(ckpt_dir / "tier1.pkl", "rb") as f:
        t1 = pickle.load(f)
    with open(ckpt_dir / "tier1_5.pkl", "rb") as f:
        t1_5 = pickle.load(f)
    with open(ckpt_dir / "tier2.pkl", "rb") as f:
        t2 = pickle.load(f)

    # Restore linker state
    linker = SLinker3.__new__(SLinker3)
    linker.__init__()
    linker.model_knowledge = t1["model_knowledge"]
    linker._is_complex = t1["is_complex"]
    linker.doc_knowledge = t1["doc_knowledge"]
    linker.GENERIC_COMPONENT_WORDS = t1["generic_component_words"]
    linker.GENERIC_PARTIALS = t1["generic_partials"]
    linker.learned_patterns = t1_5["learned_patterns"]
    linker._activity_partials = t1_5["activity_partials"]
    linker._components = components

    seed_links = t1["seed_links"]
    seed_set = t1["seed_set"]
    validated = t2["validated"]
    partial_links = t2["partial_links"]
    orig_coref = t2["coref_links"]

    # --- Baseline: replay with original single-pass coref ---
    baseline_final = replay_tier3(
        linker, seed_links, validated, orig_coref, partial_links,
        seed_set, sentences, components, sent_map)
    bp, br, bf1, btp, bfp, bfn = evaluate(baseline_final, gold)

    coref_keys_baseline = {(l.sentence_number, l.component_id) for l in orig_coref}
    print(f"  Baseline (1-pass): F1={bf1:.1%}  TP={btp} FP={bfp} FN={bfn}  coref={len(orig_coref)}")

    # --- 2x union: run coref twice, take union ---
    print(f"  Running coref pass 1...")
    t_start = time.time()
    coref_pass1 = linker._coref_cases_in_context(sentences, components, name_to_id, sent_map)
    t1_time = time.time() - t_start

    print(f"  Running coref pass 2...")
    t_start = time.time()
    coref_pass2 = linker._coref_cases_in_context(sentences, components, name_to_id, sent_map)
    t2_time = time.time() - t_start

    # Union: keep all unique (snum, cid) from either pass
    union_map = {}
    for l in coref_pass1 + coref_pass2:
        key = (l.sentence_number, l.component_id)
        if key not in union_map:
            union_map[key] = l
    coref_union = list(union_map.values())

    # Intersection for comparison
    keys1 = {(l.sentence_number, l.component_id) for l in coref_pass1}
    keys2 = {(l.sentence_number, l.component_id) for l in coref_pass2}
    intersect_keys = keys1 & keys2
    coref_intersect = [l for l in coref_pass1 if (l.sentence_number, l.component_id) in intersect_keys]

    print(f"  Pass 1: {len(coref_pass1)} links ({t1_time:.0f}s)")
    print(f"  Pass 2: {len(coref_pass2)} links ({t2_time:.0f}s)")
    print(f"  Union: {len(coref_union)}, Intersect: {len(coref_intersect)}")

    # Show differences
    only_p1 = keys1 - keys2
    only_p2 = keys2 - keys1
    if only_p1:
        for snum, cid in sorted(only_p1):
            name = next((l.component_name for l in coref_pass1 if l.sentence_number == snum and l.component_id == cid), "?")
            in_gold = "TP" if (snum, cid) in gold else "FP"
            print(f"    Only pass1: S{snum} -> {name} [{in_gold}]")
    if only_p2:
        for snum, cid in sorted(only_p2):
            name = next((l.component_name for l in coref_pass2 if l.sentence_number == snum and l.component_id == cid), "?")
            in_gold = "TP" if (snum, cid) in gold else "FP"
            print(f"    Only pass2: S{snum} -> {name} [{in_gold}]")

    # Replay with union coref
    union_final = replay_tier3(
        linker, seed_links, validated, coref_union, partial_links,
        seed_set, sentences, components, sent_map)
    up, ur, uf1, utp, ufp, ufn = evaluate(union_final, gold)
    print(f"  Union (2-pass):   F1={uf1:.1%}  TP={utp} FP={ufp} FN={ufn}  coref={len(coref_union)}")

    # Replay with intersect coref
    intersect_final = replay_tier3(
        linker, seed_links, validated, coref_intersect, partial_links,
        seed_set, sentences, components, sent_map)
    ip, ir, if1, itp, ifp, ifn = evaluate(intersect_final, gold)
    print(f"  Intersect:        F1={if1:.1%}  TP={itp} FP={ifp} FN={ifn}  coref={len(coref_intersect)}")

    # Delta
    delta_union = uf1 - bf1
    delta_intersect = if1 - bf1
    print(f"  Delta union:    {delta_union:+.1%}pp  (TP {utp-btp:+d}, FP {ufp-bfp:+d})")
    print(f"  Delta intersect:{delta_intersect:+.1%}pp  (TP {itp-btp:+d}, FP {ifp-bfp:+d})")

    return {
        "dataset": dataset_name,
        "baseline": {"f1": bf1, "tp": btp, "fp": bfp, "fn": bfn, "coref": len(orig_coref)},
        "union": {"f1": uf1, "tp": utp, "fp": ufp, "fn": ufn, "coref": len(coref_union)},
        "intersect": {"f1": if1, "tp": itp, "fp": ifp, "fn": ifn, "coref": len(coref_intersect)},
        "pass1": len(coref_pass1), "pass2": len(coref_pass2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    args = parser.parse_args()

    print("=" * 70)
    print("  S-LINKER3: 2x UNION COREF TEST")
    print("=" * 70)

    results = []
    for ds in args.datasets:
        if ds not in DATASETS:
            print(f"Unknown dataset: {ds}")
            continue
        r = run_test(ds, DATASETS[ds])
        if r:
            results.append(r)

    if results:
        print(f"\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Dataset':<16} | {'Baseline F1':>11} | {'Union F1':>11} | {'Delta':>7} | {'Intersect F1':>12} | {'Delta':>7}")
        print(f"  {'-'*16}-+-{'-'*11}-+-{'-'*11}-+-{'-'*7}-+-{'-'*12}-+-{'-'*7}")
        sum_bf1 = sum_uf1 = sum_if1 = 0
        for r in results:
            bf1, uf1, if1 = r["baseline"]["f1"], r["union"]["f1"], r["intersect"]["f1"]
            sum_bf1 += bf1; sum_uf1 += uf1; sum_if1 += if1
            print(f"  {r['dataset']:<16} | {bf1:>10.1%} | {uf1:>10.1%} | {uf1-bf1:>+6.1%} | {if1:>11.1%} | {if1-bf1:>+6.1%}")
        n = len(results)
        print(f"  {'-'*16}-+-{'-'*11}-+-{'-'*11}-+-{'-'*7}-+-{'-'*12}-+-{'-'*7}")
        print(f"  {'MACRO AVG':<16} | {sum_bf1/n:>10.1%} | {sum_uf1/n:>10.1%} | {(sum_uf1-sum_bf1)/n:>+6.1%} | {sum_if1/n:>11.1%} | {(sum_if1-sum_bf1)/n:>+6.1%}")


if __name__ == "__main__":
    main()
