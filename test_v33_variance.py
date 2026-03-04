#!/usr/bin/env python3
"""Measure V33 LLM variance by running Phases 1, 3, 4 three times.

Saves each run's output, then compares to find stable vs variant links.
Tests ensemble strategies (union, majority, intersection).
"""

import csv
import os
import pickle
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

os.environ["CLAUDE_MODEL"] = "sonnet"

from llm_sad_sam.core.data_types import SadSamLink
from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMBackend

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

N_RUNS = 3


def load_gold(gold_path):
    links = set()
    with open(gold_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def run_phases_1_3_4(text_path, model_path, run_id):
    """Run Phase 1, 3, 4 of V33 and return results."""
    from llm_sad_sam.linkers.experimental.ilinker2_v33 import ILinker2V33

    linker = ILinker2V33(backend=LLMBackend.CLAUDE)
    components = parse_pcm_repository(model_path)
    sentences = DocumentLoader.load_sentences(text_path)
    sent_map = {s.number: s for s in sentences}
    name_to_id = {c.name: c.id for c in components}

    # Phase 1
    linker.doc_profile = linker._learn_document_profile(sentences, components)
    linker._is_complex = linker._structural_complexity(sentences, components)
    model_knowledge = linker._analyze_model(components)

    ambiguous = sorted(model_knowledge.ambiguous_names)

    # Phase 3
    linker.model_knowledge = model_knowledge
    linker.GENERIC_COMPONENT_WORDS = set()
    for name in model_knowledge.ambiguous_names:
        if ' ' not in name and not name.isupper():
            linker.GENERIC_COMPONENT_WORDS.add(name.lower())
    linker.GENERIC_PARTIALS = set()
    for comp in components:
        parts = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', comp.name)
        for part in parts:
            p_lower = part.lower()
            if part.isupper():
                continue
            if len(p_lower) >= 3 and p_lower in model_knowledge.ambiguous_names or any(
                p_lower == a.lower() for a in model_knowledge.ambiguous_names
            ):
                linker.GENERIC_PARTIALS.add(p_lower)
    for name in model_knowledge.ambiguous_names:
        if ' ' not in name and not name.isupper():
            linker.GENERIC_PARTIALS.add(name.lower())

    doc_knowledge = linker._learn_document_knowledge_enriched(sentences, components)
    linker.doc_knowledge = doc_knowledge
    linker._enrich_multiword_partials(sentences, components)

    dk_items = set()
    for k, v in doc_knowledge.abbreviations.items():
        dk_items.add(("abbrev", k, v))
    for k, v in doc_knowledge.synonyms.items():
        dk_items.add(("syn", k, v))
    for k, v in doc_knowledge.partial_references.items():
        dk_items.add(("partial", k, v))

    # Phase 4 (ILinker2 seed)
    linker._cached_text_path = text_path
    linker._cached_model_path = model_path
    linker._cached_sent_map = sent_map
    linker._cached_components = components
    transarc_links = linker._process_transarc(None, {c.id: c.name for c in components}, sent_map, name_to_id)
    seed_set = {(l.sentence_number, l.component_id) for l in transarc_links}

    return {
        "ambiguous": ambiguous,
        "dk_items": dk_items,
        "seed_set": seed_set,
        "seed_count": len(transarc_links),
    }


def main():
    selected = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())

    from collections import Counter

    for ds_name in selected:
        paths = DATASETS[ds_name]
        gold = load_gold(str(paths["gold"]))
        components = parse_pcm_repository(str(paths["model"]))
        id_to_name = {c.id: c.name for c in components}

        print(f"\n{'='*100}")
        print(f"DATASET: {ds_name} (gold={len(gold)} links)")
        print(f"{'='*100}")

        runs = []
        for i in range(N_RUNS):
            print(f"\n  --- Run {i+1}/{N_RUNS} ---")
            t0 = time.time()
            result = run_phases_1_3_4(str(paths["text"]), str(paths["model"]), i)
            elapsed = time.time() - t0
            runs.append(result)
            print(f"  Ambiguous: {result['ambiguous']}")
            print(f"  Doc knowledge: {len(result['dk_items'])} items")
            print(f"  Seed: {result['seed_count']} links")
            tp = len(result['seed_set'] & gold)
            fp = result['seed_count'] - tp
            print(f"  Seed quality: {tp} TP, {fp} FP ({elapsed:.0f}s)")

        # ── Phase 1 analysis ──
        print(f"\n  PHASE 1 VARIANCE:")
        ambig_sets = [set(r["ambiguous"]) for r in runs]
        ambig_union = set()
        ambig_inter = ambig_sets[0].copy()
        for s in ambig_sets:
            ambig_union |= s
            ambig_inter &= s
        if ambig_union == ambig_inter:
            print(f"    STABLE: {sorted(ambig_union)}")
        else:
            print(f"    VARIES: union={sorted(ambig_union)}, intersection={sorted(ambig_inter)}")
            for i, s in enumerate(ambig_sets):
                print(f"    Run {i+1}: {sorted(s)}")

        # ── Phase 3 analysis ──
        print(f"\n  PHASE 3 VARIANCE:")
        dk_sets = [r["dk_items"] for r in runs]
        dk_union = set()
        dk_inter = dk_sets[0].copy()
        for s in dk_sets:
            dk_union |= s
            dk_inter &= s
        print(f"    Shared: {len(dk_inter)}, Union: {len(dk_union)}, Variant: {len(dk_union - dk_inter)}")
        for item in sorted(dk_union - dk_inter):
            present = [i+1 for i, s in enumerate(dk_sets) if item in s]
            print(f"    {item[0]}: \"{item[1]}\" -> {item[2]} | runs={present}")

        # ── Phase 4 analysis ──
        print(f"\n  PHASE 4 SEED VARIANCE:")
        seed_sets = [r["seed_set"] for r in runs]
        seed_union = set()
        seed_inter = seed_sets[0].copy()
        for s in seed_sets:
            seed_union |= s
            seed_inter &= s

        # Count occurrences
        link_counts = Counter()
        for s in seed_sets:
            for key in s:
                link_counts[key] += 1

        majority = {k for k, c in link_counts.items() if c >= 2}

        print(f"    Intersection (3/3): {len(seed_inter)} links")
        print(f"    Majority (2/3):     {len(majority)} links")
        print(f"    Union (1/3):        {len(seed_union)} links")
        print(f"    Variant links:      {len(seed_union - seed_inter)}")

        # Evaluate each strategy
        for label, s in [
            ("run1", seed_sets[0]),
            ("run2", seed_sets[1]),
            ("run3", seed_sets[2]),
            ("intersection", seed_inter),
            ("majority", majority),
            ("union", seed_union),
        ]:
            tp = len(s & gold)
            fp = len(s - gold)
            fn = len(gold - s)
            p = tp / (tp + fp) if (tp + fp) else 0
            r = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * p * r / (p + r) if (p + r) else 0
            print(f"    {label:15s}: {len(s):3d} links, {tp:2d} TP, {fp:2d} FP, seed P={p:.1%} R={r:.1%} F1={f1:.1%}")

        # Show variant links
        variant = seed_union - seed_inter
        if variant:
            print(f"\n    Variant links detail:")
            for s, c in sorted(variant):
                is_tp = (s, c) in gold
                count = link_counts[(s, c)]
                print(f"      {'TP' if is_tp else 'FP'}: S{s} -> {id_to_name.get(c, c)} ({count}/{N_RUNS} runs)")


if __name__ == "__main__":
    main()
