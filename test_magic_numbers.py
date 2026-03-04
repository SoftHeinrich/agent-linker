#!/usr/bin/env python3
"""Investigate two hardcoded magic numbers in V32 for potential overfitting.

1. Phase 3 window: sentences[:150] — only first 150 sentences sent to LLM
2. Phase 5b sampling: sentences[:30] + sentences[-30:] — head+tail for long docs

Uses saved V32 pickle checkpoints (zero LLM calls).
"""

import csv
import pickle
import re
import sys
from collections import defaultdict
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

PHASE_CACHE = Path("./results/phase_cache/v32")


def load_gold(gold_path):
    links = set()
    with open(gold_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def load_phase(ds_name, phase):
    path = PHASE_CACHE / ds_name / f"{phase}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def search_term_in_sentences(term, sentences, case_insensitive=True):
    """Find all sentence numbers where a term appears."""
    matches = []
    for s in sentences:
        text = s.text
        if case_insensitive:
            if term.lower() in text.lower():
                matches.append(s.number)
        else:
            if term in text:
                matches.append(s.number)
    return matches


# ============================================================================
print("=" * 100)
print("INVESTIGATION 1: Phase 3 window (sentences[:150])")
print("=" * 100)

for ds_name, paths in DATASETS.items():
    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    total = len(sentences)
    covered = min(total, 150)
    pct = covered / total * 100

    p3 = load_phase(ds_name, "phase3")
    dk = p3["doc_knowledge"] if p3 else None

    print(f"\n{'─'*80}")
    print(f"DATASET: {ds_name}")
    print(f"  Total sentences: {total}")
    print(f"  Covered by [:150]: {covered} ({pct:.1f}%)")
    print(f"  Sentences AFTER 150: {max(0, total - 150)}")

    if not dk:
        print("  [No Phase 3 checkpoint]")
        continue

    # Check where discovered terms appear in the FULL document
    all_terms = {}
    for label, mapping in [
        ("abbreviation", dk.abbreviations),
        ("synonym", dk.synonyms),
        ("partial", dk.partial_references),
    ]:
        for term, component in mapping.items():
            all_terms[term] = (label, component)

    if not all_terms:
        print("  Phase 3 discovered: 0 terms")
        continue

    print(f"  Phase 3 discovered: {len(all_terms)} terms")

    terms_after_150 = []
    terms_only_after_150 = []

    for term, (label, component) in sorted(all_terms.items()):
        occurrences = search_term_in_sentences(term, sentences)
        after_150 = [n for n in occurrences if n > 150]
        before_150 = [n for n in occurrences if n <= 150]

        if after_150:
            terms_after_150.append((term, label, component, before_150, after_150))
        if after_150 and not before_150:
            terms_only_after_150.append((term, label, component, after_150))

    if total <= 150:
        print("  --> Document fits entirely within 150. No truncation concern.")
    else:
        print(f"\n  Terms that ALSO appear after sentence 150 ({len(terms_after_150)}):")
        for term, label, comp, before, after in terms_after_150:
            print(f"    [{label}] '{term}' -> {comp}")
            print(f"      Before 150: {before[:5]}{'...' if len(before) > 5 else ''} ({len(before)} occurrences)")
            print(f"      After  150: {after[:5]}{'...' if len(after) > 5 else ''} ({len(after)} occurrences)")

        if terms_only_after_150:
            print(f"\n  TERMS ONLY appearing AFTER sentence 150 ({len(terms_only_after_150)}):")
            for term, label, comp, after in terms_only_after_150:
                print(f"    [{label}] '{term}' -> {comp} at sentences {after}")
        else:
            print(f"\n  Terms ONLY appearing after 150: NONE (all are also in first 150)")

    # Check: are there component names that ONLY appear after sentence 150?
    components = parse_pcm_repository(str(paths["model"]))
    comp_names = {c.name for c in components}
    gold = load_gold(str(paths["gold_sam"]))

    if total > 150:
        gold_after_150 = {(snum, cid) for snum, cid in gold if snum > 150}
        print(f"\n  Gold links after sentence 150: {len(gold_after_150)} of {len(gold)} total ({len(gold_after_150)/len(gold)*100:.1f}%)")
        if gold_after_150:
            id_to_name = {c.id: c.name for c in components}
            for snum, cid in sorted(gold_after_150):
                sent = next((s for s in sentences if s.number == snum), None)
                txt = sent.text[:80] if sent else "?"
                print(f"    S{snum} -> {id_to_name.get(cid, cid)}: {txt}")

        # Could the LLM discover MORE terms from the truncated part?
        print(f"\n  Component names appearing ONLY after sentence 150:")
        for comp in sorted(comp_names):
            before = [s for s in sentences if s.number <= 150 and comp.lower() in s.text.lower()]
            after = [s for s in sentences if s.number > 150 and comp.lower() in s.text.lower()]
            if after and not before:
                print(f"    '{comp}' - only in sentences: {[s.number for s in after]}")


# ============================================================================
print("\n\n" + "=" * 100)
print("INVESTIGATION 2: Phase 5b sampling (sentences[:30] + sentences[-30:])")
print("=" * 100)

for ds_name, paths in DATASETS.items():
    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    total = len(sentences)
    components = parse_pcm_repository(str(paths["model"]))
    gold = load_gold(str(paths["gold_sam"]))
    id_to_name = {c.id: c.name for c in components}
    name_to_id = {c.name: c.id for c in components}

    print(f"\n{'─'*80}")
    print(f"DATASET: {ds_name}")
    print(f"  Total sentences: {total}")

    # Determine if Phase 5b triggers (need to reconstruct unlinked)
    p5 = load_phase(ds_name, "phase5")
    p4 = load_phase(ds_name, "phase4")

    if not p5 or not p4:
        print("  [Missing Phase 4 or 5 checkpoint]")
        continue

    # Reconstruct which components were "unlinked" before targeted
    # Phase 5b checks: entity_comps | transarc_comps, finds unlinked
    # But the checkpoint saves candidates AFTER targeted was merged.
    # We need to separate: non-targeted candidates + transarc seeds = covered
    all_candidates = p5["candidates"]
    non_targeted = [c for c in all_candidates if c.match_type != "targeted"]
    targeted_cands = [c for c in all_candidates if c.match_type == "targeted"]

    entity_comps = {c.component_name for c in non_targeted}
    transarc_comps = {l.component_name for l in p4["transarc_links"]}
    covered_comps = entity_comps | transarc_comps
    unlinked = [c for c in components if c.name not in covered_comps]

    print(f"  Components: {len(components)}")
    print(f"  Covered by Phase 4+5a: {len(covered_comps)} ({', '.join(sorted(covered_comps))})")
    print(f"  Unlinked (trigger 5b): {len(unlinked)} ({', '.join(c.name for c in unlinked)})")
    print(f"  Targeted candidates found: {len(targeted_cands)}")

    if not unlinked:
        print("  --> Phase 5b NOT triggered. No concern.")
        continue

    # Does head+tail sampling miss important sentences?
    triggers_sampling = total > 60
    if not triggers_sampling:
        print(f"  --> Document has {total} sentences (<= 60). Full document used, no sampling.")
        continue

    print(f"  --> Document has {total} sentences (> 60). HEAD+TAIL sampling active.")
    head_range = set(range(1, 31))  # sentences 1-30
    tail_range = set(range(total - 29, total + 1))  # last 30
    middle_range = set(range(31, total - 29))
    sampled = head_range | tail_range

    print(f"  Head (S1-S30):    {len(head_range)} sentences")
    print(f"  Middle (S31-S{total-30}): {len(middle_range)} sentences (SKIPPED)")
    print(f"  Tail (S{total-29}-S{total}): {len(tail_range)} sentences")

    # For each unlinked component, check if gold links point to the middle
    for comp in unlinked:
        cid = name_to_id.get(comp.name)
        if not cid:
            continue
        comp_gold = {snum for snum, c in gold if c == cid}
        if not comp_gold:
            print(f"\n  Component '{comp.name}' (unlinked): 0 gold links (truly unlinked in gold)")
            continue

        in_head = comp_gold & head_range
        in_middle = comp_gold & middle_range
        in_tail = comp_gold & tail_range

        print(f"\n  Component '{comp.name}' (unlinked): {len(comp_gold)} gold links")
        print(f"    In head (sampled):   {sorted(in_head) if in_head else 'none'}")
        print(f"    In MIDDLE (MISSED):  {sorted(in_middle) if in_middle else 'none'}")
        print(f"    In tail (sampled):   {sorted(in_tail) if in_tail else 'none'}")

        if in_middle:
            print(f"    *** WARNING: {len(in_middle)} gold links in the MIDDLE would be invisible to Phase 5b! ***")
            for snum in sorted(in_middle):
                sent = next((s for s in sentences if s.number == snum), None)
                txt = sent.text[:100] if sent else "?"
                print(f"    S{snum}: {txt}")

    # Overall gold link distribution across document
    print(f"\n  GOLD LINK DISTRIBUTION across document positions:")
    all_gold_snums = sorted(snum for snum, _ in gold)
    in_head = [s for s in all_gold_snums if s in head_range]
    in_middle = [s for s in all_gold_snums if s in middle_range]
    in_tail = [s for s in all_gold_snums if s in tail_range]
    print(f"    Head (S1-S30):      {len(in_head)} gold links ({len(in_head)/len(gold)*100:.1f}%)")
    print(f"    Middle (S31-S{total-30}):  {len(in_middle)} gold links ({len(in_middle)/len(gold)*100:.1f}%)")
    print(f"    Tail (S{total-29}-S{total}):   {len(in_tail)} gold links ({len(in_tail)/len(gold)*100:.1f}%)")


# ============================================================================
print("\n\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

print("""
Phase 3 (sentences[:150]):
  - Concern: If docs have >150 sentences, Phase 3 cannot discover synonyms/abbreviations
    defined only in the latter part of the document.
  - Impact: Affects downstream phases that rely on doc_knowledge (Phase 5 matching,
    Phase 8c convention filter, Phase 9 judge).

Phase 5b (sentences[:30] + sentences[-30:]):
  - Concern: For docs >60 sentences, unlinked components are only searched in head+tail.
    Gold links in the middle portion are invisible to the targeted extraction.
  - Impact: Only matters when Phase 5b triggers (unlinked components exist) AND
    gold links for those components are in the skipped middle section.
""")
