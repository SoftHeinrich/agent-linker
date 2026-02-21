#!/usr/bin/env python3
"""Test multiple heuristics for routing TransArc links to a judge.

For each heuristic, measures:
- How many TransArc FPs it catches (sensitivity)
- How many TransArc TPs it would send to judge (exposure/risk)
- FP enrichment ratio in the routed subset vs the full set
"""

import csv
import re
import sys
from pathlib import Path
from dataclasses import dataclass

# Add project src to path for pcm_parser
sys.path.insert(0, str(Path(__file__).parent / "src"))
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
class TransArcLink:
    dataset: str
    model_element_id: str
    sentence_num: int  # 1-indexed
    component_name: str
    sentence_text: str
    is_tp: bool


# ─── Loading functions ───────────────────────────────────────────────────────

def load_csv_pairs(csv_path: Path) -> set[tuple[str, int]]:
    """Load (modelElementID, sentence) pairs from CSV."""
    pairs = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["modelElementID"]
            sent = int(row["sentence"])
            pairs.add((mid, sent))
    return pairs


def load_sentences(text_path: Path) -> dict[int, str]:
    """Load sentences from text file. Returns {1-indexed line num: text}."""
    sentences = {}
    with open(text_path) as f:
        for i, line in enumerate(f, 1):
            sentences[i] = line.strip()
    return sentences


def is_ambiguous_name(name: str) -> bool:
    """Phase 1 deterministic filter: single word, not CamelCase, not all-uppercase, <=6 chars."""
    # Must be single word (no spaces)
    if " " in name:
        return False
    # Must be <= 6 chars
    if len(name) > 6:
        return False
    # Must NOT be CamelCase (has internal uppercase)
    if any(c.isupper() for c in name[1:]) and name[0].isupper():
        return False
    # Must NOT be all uppercase
    if name.isupper():
        return False
    return True


# ─── Heuristic functions ────────────────────────────────────────────────────
# Each takes (link: TransArcLink, all_links: list[TransArcLink], comp_names: dict)
# Returns True if the link should be routed to judge

def h1_ambiguous(link, all_links, comp_names):
    """H1: Component name is ambiguous (single word, not CamelCase, <=6 chars)."""
    return is_ambiguous_name(link.component_name)


def h2_lowercase_in_text(link, all_links, comp_names):
    """H2: Component name appears in sentence only in lowercase."""
    name = link.component_name
    text = link.sentence_text
    # Check if lowercase version appears
    if name.lower() not in text.lower():
        return False  # name not in text at all - don't route
    # Check if it EVER appears capitalized (as-is)
    if re.search(r'\b' + re.escape(name) + r'\b', text):
        return False  # appears capitalized - don't route
    # Appears only lowercase
    return True


def h3_modifier_usage(link, all_links, comp_names):
    """H3: Component name appears as part of a compound (preceded by modifier or followed by hyphen)."""
    name = link.component_name
    text = link.sentence_text
    name_lower = name.lower()
    text_lower = text.lower()

    # Check for hyphenated usage: "client-side", "client-facing"
    if re.search(re.escape(name_lower) + r'-\w+', text_lower):
        return True
    if re.search(r'\w+-' + re.escape(name_lower), text_lower):
        return True

    # Check for modifier before name (not articles/determiners)
    articles = {"the", "a", "an", "this", "that", "its", "their", "each", "every", "of"}
    # Find all occurrences of name (case-insensitive)
    pattern = re.compile(r'(\w+)\s+' + re.escape(name_lower) + r'\b', re.IGNORECASE)
    for m in pattern.finditer(text):
        preceding = m.group(1).lower()
        if preceding not in articles and preceding.isalpha():
            return True

    return False


def h4_package_path(link, all_links, comp_names):
    """H4: Sentence contains a dotted path that includes the component name."""
    name = link.component_name
    text = link.sentence_text
    # Look for dotted paths containing the name
    pattern = re.compile(r'\b[\w]+\.[\w.]*' + re.escape(name.lower()) + r'[\w.]*', re.IGNORECASE)
    if pattern.search(text):
        return True
    # Also check reverse
    pattern2 = re.compile(r'\b[\w.]*' + re.escape(name.lower()) + r'[\w.]*\.[\w.]+', re.IGNORECASE)
    if pattern2.search(text):
        return True
    return False


def h5_impl_keywords(link, all_links, comp_names):
    """H5: Sentence contains implementation-level keywords."""
    text = link.sentence_text
    pattern = re.compile(
        r'\b(package|class|interface|method|CRUD|test cases?|implements|framework|protocol|'
        r'servlet|servlet[s]?|API endpoint|endpoint|middleware|plug-?in|driver|handler|'
        r'library|module|runtime|daemon|socket|port \d+)\b',
        re.IGNORECASE
    )
    return bool(pattern.search(text))


def h6_multi_linked(link, all_links, comp_names):
    """H6: Sentence has 3+ TransArc links (multiple components linked to same sentence)."""
    same_sent = [l for l in all_links
                 if l.dataset == link.dataset and l.sentence_num == link.sentence_num]
    return len(same_sent) >= 3


def h7_no_standalone_mention(link, all_links, comp_names):
    """H7: Component name does NOT appear as standalone capitalized word."""
    name = link.component_name
    text = link.sentence_text
    # Check for standalone capitalized occurrence
    if re.search(r'\b' + re.escape(name) + r'\b', text):
        return False  # standalone capitalized found - don't route
    return True  # no standalone capitalized mention - route to judge


def h8_ambiguous_and_signal(link, all_links, comp_names):
    """H8: Ambiguous AND (lowercase OR modifier OR package_path)."""
    if not h1_ambiguous(link, all_links, comp_names):
        return False
    return (h2_lowercase_in_text(link, all_links, comp_names) or
            h3_modifier_usage(link, all_links, comp_names) or
            h4_package_path(link, all_links, comp_names))


def h9_ambiguous_or_no_standalone(link, all_links, comp_names):
    """H9: Ambiguous OR no standalone mention."""
    return (h1_ambiguous(link, all_links, comp_names) or
            h7_no_standalone_mention(link, all_links, comp_names))


def h10_all(link, all_links, comp_names):
    """H10: Route ALL TransArc links to judge (baseline)."""
    return True


HEURISTICS = {
    "H1:  ambiguous_name": h1_ambiguous,
    "H2:  lowercase_only": h2_lowercase_in_text,
    "H3:  modifier_usage": h3_modifier_usage,
    "H4:  package_path": h4_package_path,
    "H5:  impl_keywords": h5_impl_keywords,
    "H6:  multi_linked≥3": h6_multi_linked,
    "H7:  no_standalone": h7_no_standalone_mention,
    "H8:  ambig∧(lc|mod|pkg)": h8_ambiguous_and_signal,
    "H9:  ambig∨no_standalone": h9_ambiguous_or_no_standalone,
    "H10: all_links": h10_all,
}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # Step 1: Load all data
    all_links: list[TransArcLink] = []

    for ds_name, paths in DATASETS.items():
        # Load components
        components = parse_pcm_repository(paths["model"])
        id_to_name = {c.id: c.name for c in components}

        # Load sentences
        sentences = load_sentences(paths["text"])

        # Load gold standard
        gold_pairs = load_csv_pairs(paths["gold"])

        # Load TransArc links
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

    # Step 2: Summary
    total_tp = sum(1 for l in all_links if l.is_tp)
    total_fp = sum(1 for l in all_links if not l.is_tp)
    total = len(all_links)

    print("=" * 100)
    print("TRANSARC LINK ROUTING HEURISTIC ANALYSIS")
    print("=" * 100)
    print(f"\nTotal TransArc links: {total}")
    print(f"  True Positives:  {total_tp}")
    print(f"  False Positives: {total_fp}")
    print(f"  Baseline FP rate: {total_fp/total*100:.1f}%")

    # Per-dataset breakdown
    print(f"\n{'Dataset':<16} {'Total':>6} {'TP':>5} {'FP':>5} {'FP%':>6}")
    print("-" * 40)
    for ds in DATASETS:
        ds_links = [l for l in all_links if l.dataset == ds]
        ds_tp = sum(1 for l in ds_links if l.is_tp)
        ds_fp = sum(1 for l in ds_links if not l.is_tp)
        ds_total = len(ds_links)
        print(f"{ds:<16} {ds_total:>6} {ds_tp:>5} {ds_fp:>5} {ds_fp/ds_total*100 if ds_total else 0:>5.1f}%")

    # Show ambiguous components
    print("\n\nAMBIGUOUS COMPONENTS (Phase 1 filter: single word, <=6 chars, not CamelCase):")
    print("-" * 60)
    for ds_name, paths in DATASETS.items():
        components = parse_pcm_repository(paths["model"])
        ambig = [c.name for c in components if is_ambiguous_name(c.name)]
        all_names = [c.name for c in components]
        print(f"  {ds_name}: {ambig} (of {all_names})")

    # Show FP details
    print("\n\nALL FALSE POSITIVES:")
    print("-" * 100)
    print(f"{'Dataset':<14} {'Comp Name':<22} {'Sent#':>5} {'Ambig':>5}  Sentence (first 80 chars)")
    print("-" * 100)
    for l in sorted(all_links, key=lambda x: (x.dataset, x.sentence_num)):
        if not l.is_tp:
            ambig = "Y" if is_ambiguous_name(l.component_name) else ""
            print(f"{l.dataset:<14} {l.component_name:<22} {l.sentence_num:>5} {ambig:>5}  {l.sentence_text[:80]}")

    # Step 3: Evaluate heuristics
    print("\n\n" + "=" * 100)
    print("HEURISTIC EVALUATION")
    print("=" * 100)

    # Build per-dataset link lists for multi_linked heuristic
    ds_links_map = {}
    for ds in DATASETS:
        ds_links_map[ds] = [l for l in all_links if l.dataset == ds]

    comp_names_map = {}
    for ds_name, paths in DATASETS.items():
        components = parse_pcm_repository(paths["model"])
        comp_names_map[ds_name] = {c.id: c.name for c in components}

    results = {}

    for h_name, h_func in HEURISTICS.items():
        routed_fp = 0
        routed_tp = 0
        not_routed_fp = 0
        not_routed_tp = 0
        per_ds = {}

        for ds in DATASETS:
            ds_links = ds_links_map[ds]
            ds_routed_fp = 0
            ds_routed_tp = 0
            ds_not_routed_fp = 0
            ds_not_routed_tp = 0

            for link in ds_links:
                route = h_func(link, ds_links, comp_names_map[ds])
                if route:
                    if link.is_tp:
                        routed_tp += 1
                        ds_routed_tp += 1
                    else:
                        routed_fp += 1
                        ds_routed_fp += 1
                else:
                    if link.is_tp:
                        not_routed_tp += 1
                        ds_not_routed_tp += 1
                    else:
                        not_routed_fp += 1
                        ds_not_routed_fp += 1

            per_ds[ds] = {
                "routed_fp": ds_routed_fp,
                "routed_tp": ds_routed_tp,
                "not_routed_fp": ds_not_routed_fp,
                "not_routed_tp": ds_not_routed_tp,
                "total": len(ds_links),
            }

        routed_total = routed_fp + routed_tp
        not_routed_total = not_routed_fp + not_routed_tp

        results[h_name] = {
            "routed_fp": routed_fp,
            "routed_tp": routed_tp,
            "not_routed_fp": not_routed_fp,
            "not_routed_tp": not_routed_tp,
            "routed_total": routed_total,
            "not_routed_total": not_routed_total,
            "per_ds": per_ds,
        }

    # Print summary table
    print(f"\n{'Heuristic':<25} {'FP caught':>10} {'FP sens%':>9} {'TP routed':>10} {'TP exp%':>8} {'FP% routed':>11} {'FP% resid':>10} {'Enrich':>7}")
    print("-" * 95)

    baseline_fp_rate = total_fp / total if total else 0

    for h_name, r in results.items():
        fp_caught = r["routed_fp"]
        fp_sens = fp_caught / total_fp * 100 if total_fp else 0
        tp_routed = r["routed_tp"]
        tp_exp = tp_routed / total_tp * 100 if total_tp else 0
        routed_total = r["routed_total"]
        fp_rate_routed = fp_caught / routed_total * 100 if routed_total else 0
        not_routed_total = r["not_routed_total"]
        fp_resid = r["not_routed_fp"] / not_routed_total * 100 if not_routed_total else 0
        enrichment = (fp_rate_routed / 100) / baseline_fp_rate if baseline_fp_rate and routed_total else 0

        print(f"{h_name:<25} {fp_caught:>4}/{total_fp:<5} {fp_sens:>7.1f}% {tp_routed:>4}/{total_tp:<5} {tp_exp:>6.1f}% {fp_rate_routed:>9.1f}% {fp_resid:>8.1f}% {enrichment:>6.2f}x")

    # Per-dataset detail for each heuristic
    print("\n\n" + "=" * 100)
    print("PER-DATASET BREAKDOWN")
    print("=" * 100)

    for h_name, r in results.items():
        print(f"\n{h_name}")
        print(f"  {'Dataset':<16} {'FP caught':>10} {'TP routed':>10} {'FP% routed':>11} {'FP% resid':>10}")
        print(f"  {'-'*60}")
        for ds in DATASETS:
            d = r["per_ds"][ds]
            ds_total_fp = d["routed_fp"] + d["not_routed_fp"]
            ds_total_tp = d["routed_tp"] + d["not_routed_tp"]
            routed_total = d["routed_fp"] + d["routed_tp"]
            not_routed_total = d["not_routed_fp"] + d["not_routed_tp"]
            fp_rate_r = d["routed_fp"] / routed_total * 100 if routed_total else 0
            fp_rate_nr = d["not_routed_fp"] / not_routed_total * 100 if not_routed_total else 0
            print(f"  {ds:<16} {d['routed_fp']:>4}/{ds_total_fp:<5} {d['routed_tp']:>4}/{ds_total_tp:<5} {fp_rate_r:>9.1f}% {fp_rate_nr:>8.1f}%")

    # Ranking
    print("\n\n" + "=" * 100)
    print("RANKING: Best trade-off (high FP sensitivity, low TP exposure)")
    print("Score = FP_sensitivity - 0.5 * TP_exposure  (higher is better)")
    print("=" * 100)

    scored = []
    for h_name, r in results.items():
        fp_sens = r["routed_fp"] / total_fp * 100 if total_fp else 0
        tp_exp = r["routed_tp"] / total_tp * 100 if total_tp else 0
        score = fp_sens - 0.5 * tp_exp
        scored.append((h_name, fp_sens, tp_exp, score))

    scored.sort(key=lambda x: -x[3])

    print(f"\n{'Rank':>4} {'Heuristic':<25} {'FP sens%':>9} {'TP exp%':>8} {'Score':>7}")
    print("-" * 60)
    for i, (h_name, fp_sens, tp_exp, score) in enumerate(scored, 1):
        print(f"{i:>4} {h_name:<25} {fp_sens:>7.1f}% {tp_exp:>6.1f}% {score:>7.1f}")

    # Detail: Which FPs are caught by which heuristics
    print("\n\n" + "=" * 100)
    print("FP COVERAGE MATRIX: Which heuristics catch which FPs")
    print("=" * 100)

    fp_links = [l for l in all_links if not l.is_tp]
    fp_links.sort(key=lambda x: (x.dataset, x.sentence_num))

    h_names_short = [n.split(":")[0].strip() for n in HEURISTICS.keys()]
    header = f"{'Dataset':<12} {'Comp':<18} {'S#':>3}  " + " ".join(f"{h:>3}" for h in h_names_short)
    print(f"\n{header}")
    print("-" * len(header))

    for link in fp_links:
        ds_links = ds_links_map[link.dataset]
        flags = []
        for h_name, h_func in HEURISTICS.items():
            flags.append("  X" if h_func(link, ds_links, comp_names_map[link.dataset]) else "  .")
        flags_str = " ".join(flags)  # already 3 chars each
        # Truncate differently
        flags_list = []
        for h_name, h_func in HEURISTICS.items():
            flags_list.append("X" if h_func(link, ds_links, comp_names_map[link.dataset]) else ".")
        flags_str = "   ".join(f"{f}" for f in flags_list)
        print(f"{link.dataset:<12} {link.component_name:<18} {link.sentence_num:>3}  {flags_str}")

    # TP exposure detail for top heuristics
    print("\n\n" + "=" * 100)
    print("TP EXPOSURE DETAIL: Which TPs get routed by H1 (ambiguous_name)")
    print("=" * 100)
    for link in sorted(all_links, key=lambda x: (x.dataset, x.sentence_num)):
        if link.is_tp and h1_ambiguous(link, ds_links_map[link.dataset], comp_names_map[link.dataset]):
            print(f"  {link.dataset:<14} {link.component_name:<20} S{link.sentence_num:<4} {link.sentence_text[:70]}")

    print("\n\n" + "=" * 100)
    print("TP EXPOSURE DETAIL: Which TPs get routed by H7 (no_standalone_mention)")
    print("=" * 100)
    for link in sorted(all_links, key=lambda x: (x.dataset, x.sentence_num)):
        if link.is_tp and h7_no_standalone_mention(link, ds_links_map[link.dataset], comp_names_map[link.dataset]):
            print(f"  {link.dataset:<14} {link.component_name:<20} S{link.sentence_num:<4} {link.sentence_text[:70]}")


if __name__ == "__main__":
    main()
