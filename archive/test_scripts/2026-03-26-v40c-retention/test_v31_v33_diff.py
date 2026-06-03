#!/usr/bin/env python3
"""Compare V31 vs V33 checkpoints link-by-link to find Pareto gaps."""

import csv
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.pcm_parser import parse_pcm_repository

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
DATASETS = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
        "gold_sam": BENCHMARK_BASE / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
        "model": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
        "gold_sam": BENCHMARK_BASE / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
        "gold_sam": BENCHMARK_BASE / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
        "gold_sam": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
        "gold_sam": BENCHMARK_BASE / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
}

def load_gold(gold_path):
    links = set()
    with open(gold_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links

def load_final(version, ds_name):
    path = Path(f"./results/phase_cache/{version}/{ds_name}/final.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["final"]

def load_phase(version, ds_name, phase):
    path = Path(f"./results/phase_cache/{version}/{ds_name}/{phase}.pkl")
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def eval_metrics(predicted, gold):
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"tp": tp, "fp": fp, "fn": fn, "P": p, "R": r, "F1": f1}

for ds_name, paths in DATASETS.items():
    gold = load_gold(str(paths["gold_sam"]))
    components = parse_pcm_repository(str(paths["model"]))
    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    sent_map = {s.number: s for s in sentences}
    id_to_name = {c.id: c.name for c in components}

    v31_links = load_final("v31", ds_name)
    v33_links = load_final("v33", ds_name)

    v31_set = {(l.sentence_number, l.component_id) for l in v31_links}
    v33_set = {(l.sentence_number, l.component_id) for l in v33_links}
    v31_by_key = {(l.sentence_number, l.component_id): l for l in v31_links}
    v33_by_key = {(l.sentence_number, l.component_id): l for l in v33_links}

    v31_m = eval_metrics(v31_set, gold)
    v33_m = eval_metrics(v33_set, gold)

    print(f"\n{'='*100}")
    print(f"DATASET: {ds_name}")
    print(f"  V31: P={v31_m['P']:.1%} R={v31_m['R']:.1%} F1={v31_m['F1']:.1%} FP={v31_m['fp']} FN={v31_m['fn']}")
    print(f"  V33: P={v33_m['P']:.1%} R={v33_m['R']:.1%} F1={v33_m['F1']:.1%} FP={v33_m['fp']} FN={v33_m['fn']}")
    delta = v33_m['F1'] - v31_m['F1']
    print(f"  Delta F1: {delta:+.1%}")

    # Links V33 has but V31 doesn't
    v33_only = v33_set - v31_set
    v31_only = v31_set - v33_set

    if v33_only:
        print(f"\n  V33 GAINED ({len(v33_only)}):")
        for s, c in sorted(v33_only):
            is_tp = (s, c) in gold
            lk = v33_by_key[(s, c)]
            sent = sent_map.get(s)
            text = sent.text[:80] if sent else "?"
            print(f"    {'TP' if is_tp else 'FP'}: S{s} -> {id_to_name.get(c, c)} [{lk.source}] — {text}")

    if v31_only:
        print(f"\n  V33 LOST ({len(v31_only)}):")
        for s, c in sorted(v31_only):
            is_tp = (s, c) in gold
            lk = v31_by_key[(s, c)]
            sent = sent_map.get(s)
            text = sent.text[:80] if sent else "?"
            print(f"    {'TP' if is_tp else 'FP'}: S{s} -> {id_to_name.get(c, c)} [{lk.source}] — {text}")

    # Per-phase diff: compare ILinker2 seeds (phase4)
    v31_p4 = load_phase("v31", ds_name, "phase4")
    v33_p4 = load_phase("v33", ds_name, "phase4")
    if v31_p4 and v33_p4:
        v31_seed = {(l.sentence_number, l.component_id) for l in v31_p4["transarc_links"]}
        v33_seed = {(l.sentence_number, l.component_id) for l in v33_p4["transarc_links"]}
        seed_gained = v33_seed - v31_seed
        seed_lost = v31_seed - v33_seed
        if seed_gained or seed_lost:
            print(f"\n  SEED DIFF (Phase 4 ILinker2):")
            print(f"    V31 seed: {len(v31_seed)}, V33 seed: {len(v33_seed)}")
            for s, c in sorted(seed_gained):
                is_tp = (s, c) in gold
                print(f"    SEED+  {'TP' if is_tp else 'FP'}: S{s} -> {id_to_name.get(c, c)}")
            for s, c in sorted(seed_lost):
                is_tp = (s, c) in gold
                print(f"    SEED-  {'TP' if is_tp else 'FP'}: S{s} -> {id_to_name.get(c, c)}")

    # Phase 1 diff
    v31_p1 = load_phase("v31", ds_name, "phase1")
    v33_p1 = load_phase("v33", ds_name, "phase1")
    if v31_p1 and v33_p1:
        v31_ambig = set(v31_p1["model_knowledge"].ambiguous_names)
        v33_ambig = set(v33_p1["model_knowledge"].ambiguous_names)
        if v31_ambig != v33_ambig:
            print(f"\n  PHASE 1 AMBIGUOUS DIFF:")
            print(f"    V31: {sorted(v31_ambig)}")
            print(f"    V33: {sorted(v33_ambig)}")
