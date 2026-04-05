"""Variance analysis for 12e trailing-word enrichment and its entity pipeline impact.

Runs the trailing-word enrichment step N times on BBB and all datasets,
then runs entity extraction N times with/without enriched aliases to
measure downstream variance.

Usage:
    python tests/test_12e_variance.py                     # full analysis
    python tests/test_12e_variance.py --step tw           # trailing-word only
    python tests/test_12e_variance.py --step entity       # entity extraction only
    python tests/test_12e_variance.py --step entity-val   # entity + validation
    python tests/test_12e_variance.py --dataset bbb       # BBB only
    python tests/test_12e_variance.py --runs 5            # 5 runs per step
"""

import argparse
import os
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_sad_sam.linkers.experimental.s_linker12e import SLinker12e
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.core.data_types_v2 import DocumentKnowledge, ModelKnowledge

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = ROOT / "../ardoco/core/tests-base/src/main/resources/benchmark"

DATASETS = {
    "mediastore": {
        "text": BENCHMARK / "mediastore/text_2016/mediastore.txt",
        "model": BENCHMARK / "mediastore/model_2016/pcm/ms.repository",
        "gold_sam": BENCHMARK / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    },
    "teastore": {
        "text": BENCHMARK / "teastore/text_2020/teastore.txt",
        "model": BENCHMARK / "teastore/model_2020/pcm/teastore.repository",
        "gold_sam": BENCHMARK / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    },
    "teammates": {
        "text": BENCHMARK / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK / "teammates/model_2021/pcm/teammates.repository",
        "gold_sam": BENCHMARK / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "bbb": {
        "text": BENCHMARK / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK / "bigbluebutton/model_2021/pcm/bbb.repository",
        "gold_sam": BENCHMARK / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "jabref": {
        "text": BENCHMARK / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK / "jabref/model_2021/pcm/jabref.repository",
        "gold_sam": BENCHMARK / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
}

# Map short names to checkpoint dir names
DS_CACHE_NAMES = {
    "mediastore": "mediastore",
    "teastore": "teastore",
    "teammates": "teammates",
    "bbb": "bigbluebutton",
    "jabref": "jabref",
}

def load_gold(ds_key):
    gold_path = DATASETS[ds_key]["gold_sam"]
    links = set()
    with open(gold_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("modelElementID") or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                links.add((parts[0].strip(), int(parts[1].strip())))
    return links

def load_dataset(ds_key):
    ds = DATASETS[ds_key]
    text_path = str(ds["text"])
    model_path = str(ds["model"])
    sentences = load_sentences(text_path)
    components = parse_pcm_repository(model_path)
    return text_path, sentences, components

def load_checkpoint(variant, ds_name, phase):
    cache_dir = Path("results/phase_cache") / variant / ds_name
    path = cache_dir / f"{phase}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def strip_trailing_word_aliases(dk, components):
    """Remove trailing-word aliases from DocumentKnowledge, return the stripped ones."""
    stripped = {}
    new_aliases = {}
    for alias, comp in dk.aliases.items():
        # A trailing-word alias: single word, appears as trailing word of a multi-word component
        parts = re.split(r'[\s-]+', comp) if (' ' in comp or '-' in comp) else \
                re.findall(r'[A-Z][a-z]*|[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)', comp)
        if len(parts) >= 2 and alias == parts[-1] and len(alias) >= 4:
            stripped[alias] = comp
        else:
            new_aliases[alias] = comp
    clean_dk = DocumentKnowledge()
    clean_dk.aliases = new_aliases
    return clean_dk, stripped


def run_trailing_word_step(linker, sentences, components, n_runs):
    """Run trailing-word enrichment N times, return list of alias dicts."""
    results = []
    for i in range(n_runs):
        # Run the full doc knowledge extraction (includes trailing-word step)
        dk = linker._learn_document_knowledge_enriched(sentences, components)
        # Separate out trailing-word aliases
        _, tw_aliases = strip_trailing_word_aliases(dk, components)
        results.append({
            "all_aliases": dict(dk.aliases),
            "tw_aliases": tw_aliases,
            "total": len(dk.aliases),
            "tw_count": len(tw_aliases),
        })
        print(f"  Run {i+1}: {len(dk.aliases)} aliases ({len(tw_aliases)} trailing-word)")
        for a, c in sorted(tw_aliases.items()):
            print(f"    TW: {a} -> {c}")
    return results


def run_entity_step(linker, sentences, components, sent_map, dk_with_tw, dk_without_tw, mk, n_runs):
    """Run entity extraction N times with and without trailing-word aliases."""
    name_to_id = {c.name: c.id for c in components}
    results_with = []
    results_without = []

    for i in range(n_runs):
        # WITH trailing-word aliases
        linker.doc_knowledge = dk_with_tw
        linker.model_knowledge = mk
        candidates_w = linker._extract_entities_enriched(sentences, components, name_to_id, sent_map)
        links_w = {(c.sentence_number, c.component_id) for c in candidates_w}
        results_with.append(links_w)
        print(f"  Run {i+1} WITH tw: {len(candidates_w)} candidates")

        # WITHOUT trailing-word aliases
        linker.doc_knowledge = dk_without_tw
        linker.model_knowledge = mk
        candidates_wo = linker._extract_entities_enriched(sentences, components, name_to_id, sent_map)
        links_wo = {(c.sentence_number, c.component_id) for c in candidates_wo}
        results_without.append(links_wo)
        print(f"  Run {i+1} W/O tw: {len(candidates_wo)} candidates")

        # Diff
        gained = links_w - links_wo
        lost = links_wo - links_w
        if gained:
            for snum, cid in sorted(gained):
                cname = next((c.name for c in components if c.id == cid), cid)
                print(f"    +TW gained: S{snum} -> {cname}")
        if lost:
            for snum, cid in sorted(lost):
                cname = next((c.name for c in components if c.id == cid), cid)
                print(f"    -TW lost: S{snum} -> {cname}")

    return results_with, results_without


def run_entity_validation_step(linker, sentences, components, sent_map, dk_with_tw, dk_without_tw, mk, n_runs):
    """Run full entity pipeline (extraction + validation) N times."""
    name_to_id = {c.name: c.id for c in components}
    results_with = []
    results_without = []

    for i in range(n_runs):
        # WITH trailing-word aliases
        linker.doc_knowledge = dk_with_tw
        linker.model_knowledge = mk
        validated_w = linker._run_entity_pipeline(sentences, components, name_to_id, sent_map)
        links_w = {(v.sentence_number, v.component_id) for v in validated_w}
        results_with.append(links_w)
        print(f"  Run {i+1} WITH tw (validated): {len(validated_w)}")

        # WITHOUT trailing-word aliases
        linker.doc_knowledge = dk_without_tw
        linker.model_knowledge = mk
        validated_wo = linker._run_entity_pipeline(sentences, components, name_to_id, sent_map)
        links_wo = {(v.sentence_number, v.component_id) for v in validated_wo}
        results_without.append(links_wo)
        print(f"  Run {i+1} W/O tw (validated): {len(validated_wo)}")

    return results_with, results_without


def analyze_variance(results, label, gold=None, components=None):
    """Analyze variance across N runs of a link set."""
    if not results:
        return
    print(f"\n  === {label} ({len(results)} runs) ===")

    # Size stats
    sizes = [len(r) for r in results]
    print(f"  Size: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")

    # Common core (intersection of all runs)
    core = set.intersection(*results) if results else set()
    print(f"  Core (all runs agree): {len(core)}")

    # Union
    union = set.union(*results) if results else set()
    print(f"  Union (any run): {len(union)}")

    # Per-link stability
    link_counts = Counter()
    for r in results:
        for link in r:
            link_counts[link] += 1

    n = len(results)
    stable = sum(1 for c in link_counts.values() if c == n)
    unstable = [(link, cnt) for link, cnt in link_counts.items() if cnt < n]
    print(f"  Stable (100%): {stable}, Unstable: {len(unstable)}")

    if unstable and components:
        id_to_name = {c.id: c.name for c in components}
        print(f"  Unstable links:")
        for (snum, cid), cnt in sorted(unstable, key=lambda x: -x[1]):
            cname = id_to_name.get(cid, cid)
            in_gold = ""
            if gold:
                in_gold = " [TP]" if (cid, snum) in gold else " [FP]"
            print(f"    S{snum} -> {cname}: {cnt}/{n} runs{in_gold}")

    if gold and components:
        # Per-run metrics
        id_to_name = {c.id: c.name for c in components}
        name_to_id = {c.name: c.id for c in components}
        print(f"\n  Per-run metrics (gold={len(gold)} links):")
        for i, r in enumerate(results):
            # Convert to gold format (id, snum)
            r_gold_fmt = {(cid, snum) for snum, cid in r}
            tp = len(r_gold_fmt & gold)
            fp = len(r_gold_fmt - gold)
            fn = len(gold - r_gold_fmt)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r_val / (p + r_val) if (p + r_val) > 0 else 0
            print(f"    Run {i+1}: P={p:.1%} R={r_val:.1%} F1={f1:.1%} TP={tp} FP={fp} FN={fn}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["tw", "entity", "entity-val", "all"], default="all")
    parser.add_argument("--dataset", default="all",
                        help="Dataset: mediastore, teastore, teammates, bbb, jabref, or all")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)

    ds_keys = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    linker = SLinker12e()

    for ds_key in ds_keys:
        cache_name = DS_CACHE_NAMES[ds_key]
        print(f"\n{'='*80}")
        print(f"DATASET: {ds_key}")
        print(f"{'='*80}")

        text_path, sentences, components = load_dataset(ds_key)
        sent_map = build_sent_map(sentences)
        gold = load_gold(ds_key)

        # Load existing layer1 checkpoint for model_knowledge
        ckpt = load_checkpoint("s_linker12c", cache_name, "layer1")
        if ckpt:
            mk = ckpt["model_knowledge"]
        else:
            print("  WARNING: No 12c layer1 checkpoint, running model analysis...")
            mk = linker._analyze_model(components)

        # === Step 1: Trailing-word enrichment variance ===
        if args.step in ("tw", "all"):
            print(f"\n--- Trailing-word enrichment variance ({args.runs} runs) ---")
            tw_results = run_trailing_word_step(linker, sentences, components, args.runs)

            # Analyze what varies
            all_tw = set()
            for r in tw_results:
                for a, c in r["tw_aliases"].items():
                    all_tw.add((a, c))

            tw_counts = Counter()
            for r in tw_results:
                for a, c in r["tw_aliases"].items():
                    tw_counts[(a, c)] += 1

            print(f"\n  TW alias stability ({len(all_tw)} unique across runs):")
            for (a, c), cnt in sorted(tw_counts.items(), key=lambda x: -x[1]):
                pct = cnt / args.runs * 100
                print(f"    {a} -> {c}: {cnt}/{args.runs} ({pct:.0f}%)")

            # Also check non-TW alias variance
            all_base = set()
            for r in tw_results:
                for a, c in r["all_aliases"].items():
                    if (a, c) not in all_tw:
                        all_base.add((a, c))

            base_counts = Counter()
            for r in tw_results:
                for a, c in r["all_aliases"].items():
                    if (a, c) not in all_tw:
                        base_counts[(a, c)] += 1

            unstable_base = [(k, v) for k, v in base_counts.items() if v < args.runs]
            if unstable_base:
                print(f"\n  Base alias variance ({len(unstable_base)} unstable):")
                for (a, c), cnt in sorted(unstable_base, key=lambda x: -x[1]):
                    print(f"    {a} -> {c}: {cnt}/{args.runs}")

        # === Step 2: Entity extraction variance ===
        if args.step in ("entity", "all"):
            print(f"\n--- Entity extraction variance ({args.runs} runs) ---")

            # Build DK with and without trailing-word aliases from checkpoint
            if ckpt:
                dk_full = ckpt["doc_knowledge"]
                # Now strip TW aliases
                dk_stripped, tw_found = strip_trailing_word_aliases(dk_full, components)
                print(f"  Checkpoint aliases: {len(dk_full.aliases)} (TW: {len(tw_found)})")
                if tw_found:
                    for a, c in sorted(tw_found.items()):
                        print(f"    TW from checkpoint: {a} -> {c}")
            else:
                # Run doc knowledge once to get aliases
                dk_full = linker._learn_document_knowledge_enriched(sentences, components)
                dk_stripped, tw_found = strip_trailing_word_aliases(dk_full, components)

            linker.model_knowledge = mk
            results_with, results_without = run_entity_step(
                linker, sentences, components, sent_map,
                dk_full, dk_stripped, mk, args.runs
            )

            analyze_variance(results_with, "Entity WITH tw", gold, components)
            analyze_variance(results_without, "Entity W/O tw", gold, components)

        # === Step 3: Entity + validation variance ===
        if args.step in ("entity-val", "all"):
            print(f"\n--- Entity + validation variance ({args.runs} runs) ---")

            if ckpt:
                dk_full = ckpt["doc_knowledge"]
                dk_stripped, tw_found = strip_trailing_word_aliases(dk_full, components)
            else:
                dk_full = linker._learn_document_knowledge_enriched(sentences, components)
                dk_stripped, tw_found = strip_trailing_word_aliases(dk_full, components)

            linker.model_knowledge = mk
            results_with, results_without = run_entity_validation_step(
                linker, sentences, components, sent_map,
                dk_full, dk_stripped, mk, args.runs
            )

            analyze_variance(results_with, "Validated WITH tw", gold, components)
            analyze_variance(results_without, "Validated W/O tw", gold, components)


if __name__ == "__main__":
    main()
