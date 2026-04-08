#!/usr/bin/env python3
"""S-Linker11 Checkpoint Viewer — 逐层查看管道中间输出。

用法:
    python doc/s_linker11_checkpoint_viewer.py [dataset]

dataset 默认为 mediastore，可选: mediastore, teastore, teammates, bigbluebutton, jabref
"""

import pickle
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATASET = sys.argv[1] if len(sys.argv) > 1 else "mediastore"
BASE = Path(__file__).parent.parent / "results" / "phase_cache" / "s_linker11" / DATASET

if not BASE.exists():
    print(f"Checkpoint directory not found: {BASE}")
    print(f"Available datasets: {', '.join(p.name for p in BASE.parent.iterdir() if p.is_dir())}")
    sys.exit(1)


def load_phase(name):
    path = BASE / f"{name}.pkl"
    if not path.exists():
        print(f"  [SKIP] {name}.pkl not found")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─── Layer 1: Knowledge Acquisition ───────────────────────

print_separator(f"Layer 1: Knowledge Acquisition ({DATASET})")

layer1 = load_phase("layer1")
if layer1:
    mk = layer1["model_knowledge"]
    dk = layer1["doc_knowledge"]
    raw_seeds = layer1["raw_seed_links"]

    print(f"\n[Model Knowledge]")
    print(f"  Ambiguous names ({len(mk.ambiguous_names)}): {mk.ambiguous_names}")

    print(f"\n[Document Knowledge]")
    print(f"  Abbreviations ({len(dk.abbreviations)}):")
    for short, full in dk.abbreviations.items():
        print(f"    {short} → {full}")
    print(f"  Synonyms ({len(dk.synonyms)}):")
    for syn, full in dk.synonyms.items():
        print(f"    {syn} → {full}")
    print(f"  Partial references ({len(dk.partial_references)}):")
    for partial, full in dk.partial_references.items():
        print(f"    {partial} → {full}")

    print(f"\n[Raw Seed Links] ({len(raw_seeds)} total)")
    for lk in sorted(raw_seeds, key=lambda l: l.sentence_number):
        print(f"  S{lk.sentence_number:3d} → {lk.component_name}")


# ─── Layer 2: Partial-Reference Refinement ─────────────────

print_separator("Layer 2: Partial-Reference Refinement")

layer2 = load_phase("layer2")
if layer2:
    dk2 = layer2["doc_knowledge"]
    print(f"\n[Updated Partial References] ({len(dk2.partial_references)})")
    for partial, full in dk2.partial_references.items():
        print(f"    {partial} → {full}")

    # Compare with layer1
    if layer1:
        dk1 = layer1["doc_knowledge"]
        new_partials = set(dk2.partial_references.keys()) - set(dk1.partial_references.keys())
        if new_partials:
            print(f"\n  [NEW in Tier 2]: {new_partials}")
        else:
            print(f"\n  [No new partials added in Tier 2]")


# ─── Layer 3: Link Recovery ────────────────────────────────

print_separator("Layer 3: Link Recovery (parallel)")

layer3 = load_phase("layer3")
if layer3:
    seed_links = layer3["seed_links"]
    validated = layer3["validated"]
    coref_links = layer3["coref_links"]

    print(f"\n[Seed Validated] ({len(seed_links)} links)")
    for lk in sorted(seed_links, key=lambda l: l.sentence_number):
        print(f"  S{lk.sentence_number:3d} → {lk.component_name}")

    print(f"\n[Entity Validated] ({len(validated)} links)")
    for c in sorted(validated, key=lambda l: l.sentence_number):
        print(f"  S{c.sentence_number:3d} → {c.component_name} (matched: '{c.matched_text}')")

    print(f"\n[Coreference] ({len(coref_links)} links)")
    for lk in sorted(coref_links, key=lambda l: l.sentence_number):
        print(f"  S{lk.sentence_number:3d} → {lk.component_name}")

    # Compare seed validated vs raw
    if layer1:
        raw_set = {(l.sentence_number, l.component_name) for l in layer1["raw_seed_links"]}
        val_set = {(l.sentence_number, l.component_name) for l in seed_links}
        killed = raw_set - val_set
        if killed:
            print(f"\n  [Seed validation killed {len(killed)} links]:")
            for s, c in sorted(killed):
                print(f"    S{s} → {c}")
        else:
            print(f"\n  [All {len(raw_set)} seed links survived validation]")


# ─── Layer 4: Partial Recovery ─────────────────────────────

print_separator("Layer 4: Partial Recovery")

layer4 = load_phase("layer4")
if layer4:
    partial_val = layer4["partial_validated"]
    print(f"\n[Partial Validated] ({len(partial_val)} links)")
    for c in partial_val:
        print(f"  S{c.sentence_number:3d} → {c.component_name} (matched: '{c.matched_text}')")


# ─── Final: Consolidated Links ─────────────────────────────

print_separator("Final: Consolidated Links")

final = load_phase("final")
if final:
    final_links = final["final"]
    print(f"\n[Final Links] ({len(final_links)} total)")

    # Group by source
    by_source = {}
    for lk in final_links:
        by_source.setdefault(lk.source, []).append(lk)

    for src in ["seed", "entity", "coreference", "partial"]:
        links = by_source.get(src, [])
        print(f"\n  [{src}] ({len(links)} links)")
        for lk in sorted(links, key=lambda l: l.sentence_number):
            print(f"    S{lk.sentence_number:3d} → {lk.component_name}")

    # Summary table
    print(f"\n[Summary]")
    print(f"  {'Source':<15} {'Count':>5}")
    print(f"  {'-'*20}")
    for src in ["seed", "entity", "coreference", "partial"]:
        count = len(by_source.get(src, []))
        print(f"  {src:<15} {count:>5}")
    print(f"  {'-'*20}")
    print(f"  {'TOTAL':<15} {len(final_links):>5}")
