#!/usr/bin/env python3
"""Test s_linker9d with union-for-alias validation on all projects.

Uses checkpoints: tier2 (entity+coref from original 9d run) +
re-runs ONLY partial validation with the new union logic.
Isolates the impact of the union change on partial references.

LLM calls: only for partial validation (small batches).
"""
import csv, os, sys, pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

os.environ.setdefault("CLAUDE_MODEL", "sonnet")

from llm_sad_sam.core.data_types import SadSamLink
from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.linkers.experimental.s_linker9d import SLinker9d

BENCH = Path("/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark")
CACHE = Path("results/phase_cache/s_linker9d")

DATASETS = {
    "mediastore": {
        "text": "text_2016/mediastore.txt",
        "model": "model_2016/pcm/ms.repository",
        "gold": "goldstandards/goldstandard_sad_2016-sam_2016.csv",
    },
    "teastore": {
        "text": "text_2020/teastore.txt",
        "model": "model_2020/pcm/teastore.repository",
        "gold": "goldstandards/goldstandard_sad_2020-sam_2020.csv",
    },
    "teammates": {
        "text": "text_2021/teammates.txt",
        "model": "model_2021/pcm/teammates.repository",
        "gold": "goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "bigbluebutton": {
        "text": "text_2021/bigbluebutton.txt",
        "model": "model_2021/pcm/bbb.repository",
        "gold": "goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "jabref": {
        "text": "text_2021/jabref.txt",
        "model": "model_2021/pcm/jabref.repository",
        "gold": "goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
}

loader = DocumentLoader()
all_results = {}

for ds, paths in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"DATASET: {ds}")
    print(f"{'='*60}")

    # Load gold
    gold_path = BENCH / ds / paths["gold"]
    gold = set()
    with open(gold_path) as f:
        for row in csv.DictReader(f):
            gold.add((int(row["sentence"]), row["modelElementID"]))

    # Load sentences + components
    text_path = str(BENCH / ds / paths["text"])
    model_path = str(BENCH / ds / paths["model"])
    sentences = loader.load_sentences(text_path)
    components = parse_pcm_repository(model_path)
    name_to_id = {c.name: c.id for c in components}
    sent_map = {s.number: s for s in sentences}

    # Load checkpoints
    with open(CACHE / ds / "tier1.pkl", "rb") as f:
        tier1 = pickle.load(f)
    with open(CACHE / ds / "tier1_5.pkl", "rb") as f:
        tier1_5 = pickle.load(f)
    with open(CACHE / ds / "tier2.pkl", "rb") as f:
        tier2 = pickle.load(f)
    with open(CACHE / ds / "final.pkl", "rb") as f:
        final_orig = pickle.load(f)

    seed_links = tier1["seed_links"]
    seed_set = tier1["seed_set"]
    dk = tier1_5.get("doc_knowledge") or tier1.get("doc_knowledge")

    # Original results from checkpoint
    validated_orig = tier2["validated"]
    coref_orig = tier2["coref_links"]
    partial_orig = tier2["partial_validated"]
    final_links_orig = final_orig["final"]

    # Create linker with state
    linker = SLinker9d(backend=LLMBackend.CLAUDE)
    linker.doc_knowledge = dk
    linker.model_knowledge = tier1.get("model_knowledge")

    # Re-run ONLY partial validation with union logic
    validated_set = {(c.sentence_number, c.component_id) for c in validated_orig}
    coref_set = {(l.sentence_number, l.component_id) for l in coref_orig}

    partial_candidates = linker._inject_partial_candidates(
        sentences, components, name_to_id, sent_map,
        seed_set=seed_set, validated_set=validated_set, coref_set=coref_set
    )

    print(f"  Seed: {len(seed_links)}, Entity validated: {len(validated_orig)}, "
          f"Coref: {len(coref_orig)}, Partial candidates: {len(partial_candidates)}")

    if partial_candidates:
        partial_new = linker._validate_intersect(partial_candidates, components, sent_map)
        print(f"  Partial validated (union): {len(partial_new)} / {len(partial_candidates)}")
    else:
        partial_new = []
        print(f"  No partial candidates")

    # Merge: seed + entity validated + coref + new partials (same dedup as pipeline)
    all_links = []
    seen = set()

    for link in seed_links:
        key = (link.sentence_number, link.component_id)
        if key not in seen:
            seen.add(key)
            all_links.append(link)

    for c in validated_orig:
        key = (c.sentence_number, c.component_id)
        if key not in seen:
            seen.add(key)
            all_links.append(SadSamLink(c.sentence_number, c.component_id,
                                        c.component_name, 1.0, c.source))

    for link in coref_orig:
        key = (link.sentence_number, link.component_id)
        if key not in seen:
            seen.add(key)
            all_links.append(link)

    for c in partial_new:
        key = (c.sentence_number, c.component_id)
        if key not in seen:
            seen.add(key)
            all_links.append(SadSamLink(c.sentence_number, c.component_id,
                                        c.component_name, 1.0, "partial_inject"))

    # Score
    def score_links(links, gold, label):
        result_set = {(l.sentence_number, l.component_id) for l in links}
        tp = len(result_set & gold)
        fp = len(result_set - gold)
        fn = len(gold - result_set)
        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        print(f"  {label}: P={p:.1%} R={r:.1%} F1={f1:.1%} TP={tp} FP={fp} FN={fn}")
        return f1

    orig_set = {(l.sentence_number, l.component_id) for l in final_links_orig}
    orig_tp = len(orig_set & gold)
    orig_fp = len(orig_set - gold)
    orig_fn = len(gold - orig_set)
    orig_p = orig_tp / (orig_tp + orig_fp) if orig_tp + orig_fp > 0 else 0
    orig_r = orig_tp / (orig_tp + orig_fn) if orig_tp + orig_fn > 0 else 0
    orig_f1 = 2 * orig_p * orig_r / (orig_p + orig_r) if orig_p + orig_r > 0 else 0
    print(f"  ORIGINAL:  P={orig_p:.1%} R={orig_r:.1%} F1={orig_f1:.1%} TP={orig_tp} FP={orig_fp} FN={orig_fn}")

    new_f1 = score_links(all_links, gold, "UNION   ")

    delta = new_f1 - orig_f1
    print(f"  DELTA: {delta:+.1%}")

    all_results[ds] = {"orig": orig_f1, "union": new_f1, "delta": delta,
                       "partial_cands": len(partial_candidates),
                       "partial_approved": len(partial_new)}

# Summary
print(f"\n{'='*60}")
print("SUMMARY: 9d original vs 9d+union")
print(f"{'='*60}")
print(f"  {'Dataset':<16} {'Original':>8} {'Union':>8} {'Delta':>8} {'Partials':>10}")
print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

macro_orig = macro_union = 0
for ds in DATASETS:
    r = all_results[ds]
    macro_orig += r["orig"]
    macro_union += r["union"]
    print(f"  {ds:<16} {r['orig']:>7.1%} {r['union']:>7.1%} {r['delta']:>+7.1%} "
          f"{r['partial_approved']}/{r['partial_cands']}")

n = len(DATASETS)
macro_orig /= n
macro_union /= n
print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'MACRO AVG':<16} {macro_orig:>7.1%} {macro_union:>7.1%} {macro_union - macro_orig:>+7.1%}")
