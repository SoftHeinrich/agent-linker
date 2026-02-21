#!/usr/bin/env python3
"""Deep analysis of H4 vs H2 heuristics for TransArc link routing.

Answers 6 specific questions about H4 (package_path) and H2 (lowercase_only):
1. H4 vs H2 overlap (Venn diagram)
2. H4+H2 union stats
3. H4 auto-reject: which 2 TPs are killed
4. Can killed TPs be rescued by other pipeline steps?
5. H4 with judge instead of auto-reject
6. H2 standalone analysis
"""

import csv
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, "/mnt/hostshare/ardoco-home/llm-sad-sam-v45/src")
from llm_sad_sam.pcm_parser import parse_pcm_repository

# ─── Configuration ───────────────────────────────────────────────────────────

BENCHMARK = Path("/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark")
TRANSARC_RESULTS = Path("/mnt/hostshare/ardoco-home/transarc-emp/results")

DATASETS = {
    "mediastore": {
        "text": BENCHMARK / "mediastore/text_2016/mediastore.txt",
        "gold": BENCHMARK / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
        "transarc": TRANSARC_RESULTS / "mediastore/sad-sam/sadSamTlr_mediastore.csv",
        "model": BENCHMARK / "mediastore/model_2016/pcm/ms.repository",
    },
    "teastore": {
        "text": BENCHMARK / "teastore/text_2020/teastore.txt",
        "gold": BENCHMARK / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
        "transarc": TRANSARC_RESULTS / "teastore/sad-sam/sadSamTlr_teastore.csv",
        "model": BENCHMARK / "teastore/model_2020/pcm/teastore.repository",
    },
    "teammates": {
        "text": BENCHMARK / "teammates/text_2021/teammates.txt",
        "gold": BENCHMARK / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc": TRANSARC_RESULTS / "teammates/sad-sam/sadSamTlr_teammates.csv",
        "model": BENCHMARK / "teammates/model_2021/pcm/teammates.repository",
    },
    "bigbluebutton": {
        "text": BENCHMARK / "bigbluebutton/text_2021/bigbluebutton.txt",
        "gold": BENCHMARK / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc": TRANSARC_RESULTS / "bigbluebutton/sad-sam/sadSamTlr_bigbluebutton.csv",
        "model": BENCHMARK / "bigbluebutton/model_2021/pcm/bbb.repository",
    },
    "jabref": {
        "text": BENCHMARK / "jabref/text_2021/jabref.txt",
        "gold": BENCHMARK / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc": TRANSARC_RESULTS / "jabref/sad-sam/sadSamTlr_jabref.csv",
        "model": BENCHMARK / "jabref/model_2021/pcm/jabref.repository",
    },
}

# ─── Data types ──────────────────────────────────────────────────────────────

@dataclass
class Component:
    id: str
    name: str
    # Aliases: partials from splitting CamelCase, abbreviations, etc.
    partials: list[str] = field(default_factory=list)

@dataclass
class TransArcLink:
    dataset: str
    model_element_id: str
    sentence_num: int  # 1-indexed
    component_name: str
    sentence_text: str
    is_tp: bool

# ─── Loading functions ───────────────────────────────────────────────────────

def load_csv_pairs(csv_path: Path) -> set[tuple[str, int]]:
    pairs = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["modelElementID"]
            sent = int(row["sentence"])
            pairs.add((mid, sent))
    return pairs

def load_sentences(text_path: Path) -> dict[int, str]:
    sentences = {}
    with open(text_path) as f:
        for i, line in enumerate(f, 1):
            sentences[i] = line.strip()
    return sentences

def split_camel_case(name: str) -> list[str]:
    """Split CamelCase into words. E.g. 'WebStorage' -> ['Web', 'Storage']"""
    parts = re.sub(r'([A-Z])', r' \1', name).split()
    return [p for p in parts if len(p) > 2]

def get_component_partials(name: str) -> list[str]:
    """Get partial words from component name."""
    partials = []
    # CamelCase splitting
    cc_parts = split_camel_case(name)
    if len(cc_parts) > 1:
        partials.extend(cc_parts)
    # Space-separated parts
    if " " in name:
        partials.extend([p for p in name.split() if len(p) > 2])
    return list(set(p for p in partials if p.lower() != name.lower()))

# ─── Heuristic functions ────────────────────────────────────────────────────

def h2_lowercase_in_text(link):
    """H2: Component name appears in sentence only in lowercase."""
    name = link.component_name
    text = link.sentence_text
    if name.lower() not in text.lower():
        return False
    if re.search(r'\b' + re.escape(name) + r'\b', text):
        return False
    return True

def h4_package_path(link):
    """H4: Sentence contains a dotted path that includes the component name."""
    name = link.component_name
    text = link.sentence_text
    pattern = re.compile(r'\b[\w]+\.[\w.]*' + re.escape(name.lower()) + r'[\w.]*', re.IGNORECASE)
    if pattern.search(text):
        return True
    pattern2 = re.compile(r'\b[\w.]*' + re.escape(name.lower()) + r'[\w.]*\.[\w.]+', re.IGNORECASE)
    if pattern2.search(text):
        return True
    return False

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # Step 1: Load all data
    all_links: list[TransArcLink] = []
    all_sentences: dict[str, dict[int, str]] = {}
    all_components: dict[str, list[Component]] = {}

    for ds_name, paths in DATASETS.items():
        components_raw = parse_pcm_repository(paths["model"])
        id_to_name = {c.id: c.name for c in components_raw}
        components = [Component(id=c.id, name=c.name, partials=get_component_partials(c.name)) for c in components_raw]
        all_components[ds_name] = components

        sentences = load_sentences(paths["text"])
        all_sentences[ds_name] = sentences

        gold_pairs = load_csv_pairs(paths["gold"])
        transarc_pairs = load_csv_pairs(paths["transarc"])

        for mid, sent_num in transarc_pairs:
            comp_name = id_to_name.get(mid, f"UNKNOWN({mid})")
            sent_text = sentences.get(sent_num, "")
            is_tp = (mid, sent_num) in gold_pairs
            all_links.append(TransArcLink(
                dataset=ds_name,
                model_element_id=mid,
                sentence_num=sent_num,
                component_name=comp_name,
                sentence_text=sent_text,
                is_tp=is_tp,
            ))

    total_tp = sum(1 for l in all_links if l.is_tp)
    total_fp = sum(1 for l in all_links if not l.is_tp)
    total = len(all_links)

    # Classify every link
    h2_set = set()
    h4_set = set()
    for i, link in enumerate(all_links):
        if h2_lowercase_in_text(link):
            h2_set.add(i)
        if h4_package_path(link):
            h4_set.add(i)

    # ─── Q1: H4 vs H2 overlap (Venn diagram) ────────────────────────────────
    print("=" * 100)
    print("Q1: H4 vs H2 OVERLAP (Venn Diagram)")
    print("=" * 100)

    h4_only = h4_set - h2_set
    h2_only = h2_set - h4_set
    both = h4_set & h2_set
    neither = set(range(len(all_links))) - h4_set - h2_set

    def classify_set(indices):
        tp = sum(1 for i in indices if all_links[i].is_tp)
        fp = sum(1 for i in indices if not all_links[i].is_tp)
        return tp, fp

    h4_only_tp, h4_only_fp = classify_set(h4_only)
    h2_only_tp, h2_only_fp = classify_set(h2_only)
    both_tp, both_fp = classify_set(both)
    neither_tp, neither_fp = classify_set(neither)

    print(f"\n  FP breakdown (total FP = {total_fp}):")
    print(f"    H4 only:  {h4_only_fp}")
    print(f"    H2 only:  {h2_only_fp}")
    print(f"    Both:     {both_fp}")
    print(f"    Neither:  {neither_fp}")

    print(f"\n  TP breakdown (total TP = {total_tp}):")
    print(f"    H4 only:  {h4_only_tp}")
    print(f"    H2 only:  {h2_only_tp}")
    print(f"    Both:     {both_tp}")
    print(f"    Neither:  {neither_tp}")

    print(f"\n  ASCII Venn Diagram (FP / TP):")
    print(f"""
              H4                    H2
         ┌──────────┐         ┌──────────┐
         │          │         │          │
         │  H4 only │  Both   │  H2 only │
         │  {h4_only_fp} FP    │  {both_fp} FP   │  {h2_only_fp} FP    │
         │  {h4_only_tp} TP    │  {both_tp} TP   │  {h2_only_tp} TP    │
         │          │         │          │
         └──────────┘         └──────────┘

         Neither: {neither_fp} FP, {neither_tp} TP
    """)

    # Detail: links in each region
    print("  Links in H4-only:")
    for i in sorted(h4_only, key=lambda i: (all_links[i].dataset, all_links[i].sentence_num)):
        l = all_links[i]
        label = "TP" if l.is_tp else "FP"
        print(f"    [{label}] {l.dataset:<14} {l.component_name:<22} S{l.sentence_num:<4} {l.sentence_text[:70]}")

    print("\n  Links in H2-only:")
    for i in sorted(h2_only, key=lambda i: (all_links[i].dataset, all_links[i].sentence_num)):
        l = all_links[i]
        label = "TP" if l.is_tp else "FP"
        print(f"    [{label}] {l.dataset:<14} {l.component_name:<22} S{l.sentence_num:<4} {l.sentence_text[:70]}")

    print("\n  Links in Both (H4 AND H2):")
    for i in sorted(both, key=lambda i: (all_links[i].dataset, all_links[i].sentence_num)):
        l = all_links[i]
        label = "TP" if l.is_tp else "FP"
        print(f"    [{label}] {l.dataset:<14} {l.component_name:<22} S{l.sentence_num:<4} {l.sentence_text[:70]}")

    # ─── Q2: H4+H2 union ────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("Q2: H4+H2 UNION")
    print("=" * 100)

    union = h4_set | h2_set
    union_tp, union_fp = classify_set(union)
    union_total = len(union)
    baseline_fp_rate = total_fp / total

    print(f"\n  Union catches: {union_fp} FP out of {total_fp} (sensitivity: {union_fp/total_fp*100:.1f}%)")
    print(f"  Union routes:  {union_tp} TP out of {total_tp} (exposure: {union_tp/total_tp*100:.1f}%)")
    print(f"  Union total:   {union_total} links")
    print(f"  FP rate in union:   {union_fp/union_total*100:.1f}%" if union_total else "  FP rate in union: N/A")
    print(f"  FP rate in residual: {(total_fp - union_fp)/(total - union_total)*100:.1f}%" if (total - union_total) else "")
    enrichment = (union_fp / union_total) / baseline_fp_rate if union_total and baseline_fp_rate else 0
    print(f"  Enrichment:    {enrichment:.2f}x")

    # ─── Q3: H4 auto-reject: which TPs are killed ───────────────────────────
    print("\n" + "=" * 100)
    print("Q3: H4 AUTO-REJECT — WHICH TPs ARE KILLED?")
    print("=" * 100)

    h4_tp_indices = [i for i in h4_set if all_links[i].is_tp]
    print(f"\n  H4 flags {len(h4_tp_indices)} TP(s):\n")

    for i in h4_tp_indices:
        l = all_links[i]
        print(f"  TP #{i}:")
        print(f"    Dataset:        {l.dataset}")
        print(f"    Component:      {l.component_name} (ID: {l.model_element_id})")
        print(f"    Sentence #:     {l.sentence_num}")
        print(f"    Full sentence:  {l.sentence_text}")
        print()
        # Find the dotted path that triggered H4
        name = l.component_name
        text = l.sentence_text
        pattern = re.compile(r'\b[\w]+\.[\w.]*' + re.escape(name.lower()) + r'[\w.]*', re.IGNORECASE)
        m = pattern.search(text)
        if m:
            print(f"    H4 trigger:     '{m.group()}' (dotted path containing '{name}')")
        pattern2 = re.compile(r'\b[\w.]*' + re.escape(name.lower()) + r'[\w.]*\.[\w.]+', re.IGNORECASE)
        m2 = pattern2.search(text)
        if m2 and (not m or m2.group() != m.group()):
            print(f"    H4 trigger (2): '{m2.group()}'")
        print()

    # Assessment
    print("  ASSESSMENT:")
    for i in h4_tp_indices:
        l = all_links[i]
        # Check if component name also appears standalone (not just in dotted path)
        name = l.component_name
        text = l.sentence_text
        standalone = bool(re.search(r'(?<!\w\.)(?<!\.)' + r'\b' + re.escape(name) + r'\b' + r'(?!\.\w)', text))
        has_standalone_lower = bool(re.search(r'(?<!\w\.)(?<!\.)' + r'\b' + re.escape(name.lower()) + r'\b' + r'(?!\.\w)', text, re.IGNORECASE))
        print(f"    {l.dataset} S{l.sentence_num} {l.component_name}:")
        print(f"      Standalone capitalized mention: {standalone}")
        print(f"      Standalone mention (any case):  {has_standalone_lower}")
        print(f"      Edge case or important? {'EDGE CASE - name only in dotted path' if not has_standalone_lower else 'IMPORTANT - name appears standalone too'}")
        print()

    # ─── Q4: Can killed TPs be rescued by other pipeline steps? ──────────────
    print("\n" + "=" * 100)
    print("Q4: CAN KILLED TPs BE RESCUED BY OTHER PIPELINE STEPS?")
    print("=" * 100)

    for i in h4_tp_indices:
        l = all_links[i]
        name = l.component_name
        text = l.sentence_text
        ds = l.dataset
        sentences = all_sentences[ds]
        sent_num = l.sentence_num

        # Get component info
        comp = None
        for c in all_components[ds]:
            if c.id == l.model_element_id:
                comp = c
                break
        partials = comp.partials if comp else []

        print(f"\n  TP: {ds} S{sent_num} -> {name}")
        print(f"    Sentence: {text}")
        print(f"    Component partials: {partials}")

        # Phase 5: Entity extraction - would the component name or alias appear?
        name_in_text = bool(re.search(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE))
        print(f"\n    Phase 5 (Entity Extraction):")
        print(f"      Name '{name}' in sentence (case-insensitive): {name_in_text}")
        # Check if it appears only in dotted path
        # Remove dotted paths and check again
        text_no_dots = re.sub(r'\b[\w]+(?:\.[\w]+)+\b', ' ', text)
        name_outside_dots = bool(re.search(r'\b' + re.escape(name) + r'\b', text_no_dots, re.IGNORECASE))
        print(f"      Name '{name}' outside dotted paths: {name_outside_dots}")
        print(f"      Verdict: {'YES - extraction would likely find it' if name_outside_dots else 'UNLIKELY - name only in dotted path, extraction may miss it'}")

        # Phase 7: Coreference - is there a pronoun?
        pronouns = re.findall(r'\b(it|its|they|their|them|this|these|those)\b', text, re.IGNORECASE)
        print(f"\n    Phase 7 (Coreference):")
        print(f"      Pronouns in sentence: {pronouns}")
        # Check nearby sentences for the component name
        nearby_mentions = []
        for offset in [-2, -1, 1, 2]:
            nearby_num = sent_num + offset
            if nearby_num in sentences:
                nearby_text = sentences[nearby_num]
                if re.search(r'\b' + re.escape(name) + r'\b', nearby_text, re.IGNORECASE):
                    nearby_mentions.append((nearby_num, nearby_text[:80]))
        print(f"      '{name}' in nearby sentences (+-2): {len(nearby_mentions)}")
        for nm_num, nm_text in nearby_mentions:
            print(f"        S{nm_num}: {nm_text}")
        coref_possible = len(pronouns) > 0 and len(nearby_mentions) > 0
        print(f"      Verdict: {'POSSIBLE - pronoun + nearby mention' if coref_possible else 'UNLIKELY - ' + ('no pronouns' if not pronouns else 'no nearby mention')}")

        # Phase 8b: Partial injection - does a partial word appear?
        print(f"\n    Phase 8b (Partial Injection):")
        partial_found = []
        for p in partials:
            if re.search(r'\b' + re.escape(p) + r'\b', text, re.IGNORECASE):
                partial_found.append(p)
        print(f"      Partials found in sentence: {partial_found}")
        # Check outside dotted paths
        partial_outside = []
        for p in partials:
            if re.search(r'\b' + re.escape(p) + r'\b', text_no_dots, re.IGNORECASE):
                partial_outside.append(p)
        print(f"      Partials outside dotted paths: {partial_outside}")
        print(f"      Verdict: {'YES - partial would inject' if partial_outside else 'UNLIKELY - no partial outside dotted path'}")

        # Phase 5b: Targeted extraction - would targeted search find this sentence?
        print(f"\n    Phase 5b (Targeted Extraction):")
        # Check if component name or any alias/partial appears in sentence or nearby context
        # Targeted extraction searches ALL sentences for mentions of a specific component
        appears_anywhere = name_in_text  # even in dotted path
        print(f"      Name appears in sentence (even in dotted path): {appears_anywhere}")
        print(f"      Verdict: {'MAYBE - LLM might recognize it in dotted path context' if appears_anywhere and not name_outside_dots else 'YES - name clearly appears' if name_outside_dots else 'NO - name not in sentence'}")

        # Overall rescue assessment
        print(f"\n    OVERALL RESCUE ASSESSMENT:")
        rescuable = name_outside_dots or coref_possible or bool(partial_outside)
        if rescuable:
            mechanisms = []
            if name_outside_dots:
                mechanisms.append("Phase 5 extraction")
            if coref_possible:
                mechanisms.append("Phase 7 coreference")
            if partial_outside:
                mechanisms.append("Phase 8b partial injection")
            print(f"      RESCUABLE via: {', '.join(mechanisms)}")
        else:
            print(f"      NOT EASILY RESCUABLE — name only appears in dotted path")
            print(f"      Would need Phase 5b targeted extraction with LLM understanding dotted paths")

    # ─── Q5: H4 with judge instead of auto-reject ───────────────────────────
    print("\n" + "=" * 100)
    print("Q5: H4 WITH JUDGE INSTEAD OF AUTO-REJECT")
    print("=" * 100)

    for i in h4_tp_indices:
        l = all_links[i]
        ds = l.dataset
        sentences = all_sentences[ds]
        sent_num = l.sentence_num

        print(f"\n  TP: {ds} S{sent_num} -> {l.component_name}")
        print(f"  Full sentence:")
        print(f"    S{sent_num}: {l.sentence_text}")
        print(f"\n  Context (surrounding sentences):")
        for offset in range(-3, 4):
            ctx_num = sent_num + offset
            if ctx_num in sentences:
                marker = " >>>" if offset == 0 else "    "
                print(f"  {marker} S{ctx_num}: {sentences[ctx_num]}")

        # Find the dotted path
        name = l.component_name
        text = l.sentence_text
        # Remove dotted path to see what remains
        text_no_dots = re.sub(r'\b[\w]+(?:\.[\w]+)+\b', '[DOTTED_PATH]', text)
        print(f"\n  Sentence with dotted paths removed:")
        print(f"    {text_no_dots}")

        # Judge assessment
        print(f"\n  JUDGE ASSESSMENT:")
        # Check if the sentence is actually about the component
        name_standalone = bool(re.search(r'(?<!\.)' + r'\b' + re.escape(name) + r'\b' + r'(?!\.)', text))
        name_in_dots_only = not name_standalone and bool(re.search(re.escape(name), text, re.IGNORECASE))
        if name_standalone:
            print(f"    Name appears standalone — judge would likely APPROVE")
        elif name_in_dots_only:
            print(f"    Name only in dotted path — judge verdict depends on:")
            print(f"      - Is the sentence ABOUT the component (discussing its role/behavior)?")
            print(f"      - Or just mentioning a package/class path incidentally?")
            # Try to determine from context
            comp_mentioned_nearby = False
            for offset in [-2, -1, 1, 2]:
                ctx_num = sent_num + offset
                if ctx_num in sentences:
                    if re.search(r'\b' + re.escape(name) + r'\b', sentences[ctx_num]):
                        comp_mentioned_nearby = True
                        break
            if comp_mentioned_nearby:
                print(f"    Component mentioned standalone in nearby sentences — LIKELY APPROVE")
            else:
                print(f"    Component NOT mentioned standalone nearby — LIKELY REJECT")

    # Also show H4-flagged FPs for comparison
    h4_fp_indices = [i for i in h4_set if not all_links[i].is_tp]
    print(f"\n  For comparison, H4-flagged FPs ({len(h4_fp_indices)}):")
    for i in h4_fp_indices:
        l = all_links[i]
        print(f"    [FP] {l.dataset:<14} {l.component_name:<22} S{l.sentence_num:<4} {l.sentence_text[:80]}")

    # ─── Q6: H2 standalone analysis ─────────────────────────────────────────
    print("\n" + "=" * 100)
    print("Q6: H2 STANDALONE ANALYSIS")
    print("=" * 100)

    # FPs caught by H2 but NOT H4
    h2_not_h4_fp = [i for i in h2_only if not all_links[i].is_tp]
    print(f"\n  FPs caught by H2 but NOT H4 ({len(h2_not_h4_fp)}):")
    print(f"  {'='*90}")
    for i in h2_not_h4_fp:
        l = all_links[i]
        print(f"\n    Dataset:    {l.dataset}")
        print(f"    Component:  {l.component_name} (ID: {l.model_element_id})")
        print(f"    Sentence #: {l.sentence_num}")
        print(f"    Sentence:   {l.sentence_text}")
        # Show what H2 detected
        name = l.component_name
        text = l.sentence_text
        # Find the lowercase occurrence
        pattern = re.compile(r'\b' + re.escape(name.lower()) + r'\b', re.IGNORECASE)
        matches = [(m.start(), m.group()) for m in pattern.finditer(text)]
        print(f"    Occurrences of '{name}' in text: {matches}")

    # TPs routed by H2 (all of them)
    h2_tp_indices = [i for i in h2_set if all_links[i].is_tp]
    print(f"\n\n  ALL TPs routed by H2 ({len(h2_tp_indices)}):")
    print(f"  {'='*90}")

    nearby_standalone_count = 0
    for i in sorted(h2_tp_indices, key=lambda i: (all_links[i].dataset, all_links[i].sentence_num)):
        l = all_links[i]
        ds = l.dataset
        sentences = all_sentences[ds]
        name = l.component_name

        # Check nearby sentences for standalone capitalized mention
        has_nearby_standalone = False
        nearby_info = []
        for offset in range(-3, 4):
            if offset == 0:
                continue
            ctx_num = l.sentence_num + offset
            if ctx_num in sentences:
                ctx_text = sentences[ctx_num]
                if re.search(r'\b' + re.escape(name) + r'\b', ctx_text):
                    has_nearby_standalone = True
                    nearby_info.append(f"S{ctx_num}")

        if has_nearby_standalone:
            nearby_standalone_count += 1

        nearby_str = f" (standalone in: {', '.join(nearby_info)})" if nearby_info else " (NO nearby standalone)"
        print(f"    [{ds:<14}] S{l.sentence_num:<4} {l.component_name:<22} {l.sentence_text[:60]}")
        print(f"      -> nearby capitalized: {nearby_str}")

    print(f"\n  SUMMARY:")
    print(f"    Total H2-routed TPs: {len(h2_tp_indices)}")
    print(f"    With standalone capitalized name in nearby sentences (+-3): {nearby_standalone_count}")
    print(f"    Without nearby standalone: {len(h2_tp_indices) - nearby_standalone_count}")
    print(f"    Ratio with nearby evidence: {nearby_standalone_count/len(h2_tp_indices)*100:.1f}%" if h2_tp_indices else "")

    # ─── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    h4_fp_total = sum(1 for i in h4_set if not all_links[i].is_tp)
    h4_tp_total = sum(1 for i in h4_set if all_links[i].is_tp)
    h2_fp_total = sum(1 for i in h2_set if not all_links[i].is_tp)
    h2_tp_total = sum(1 for i in h2_set if all_links[i].is_tp)

    print(f"""
  H4 (package_path):
    FP caught: {h4_fp_total}/{total_fp} ({h4_fp_total/total_fp*100:.1f}%)
    TP exposed: {h4_tp_total}/{total_tp} ({h4_tp_total/total_tp*100:.1f}%)

  H2 (lowercase_only):
    FP caught: {h2_fp_total}/{total_fp} ({h2_fp_total/total_fp*100:.1f}%)
    TP exposed: {h2_tp_total}/{total_tp} ({h2_tp_total/total_tp*100:.1f}%)

  H4+H2 union:
    FP caught: {union_fp}/{total_fp} ({union_fp/total_fp*100:.1f}%)
    TP exposed: {union_tp}/{total_tp} ({union_tp/total_tp*100:.1f}%)

  H4 auto-reject kills {h4_tp_total} TP(s)
  H2 routes {h2_tp_total} TPs to judge (risk if judge is imperfect)
  Combined union routes {union_tp} TPs total

  Recommendation: Use H4 as auto-reject ONLY IF killed TPs are rescuable.
  Use H2 to route remaining suspicious links to judge.
    """)


if __name__ == "__main__":
    main()
