#!/usr/bin/env python3
"""
Verify that 3 proposed deterministic seed filters do NOT kill any TPs.

Filters:
1. DOTTED-PATH: Reject if component name appears ONLY inside dotted paths
2. LOWERCASE AMBIGUOUS: Reject if ambiguous + appears only in lowercase
3. PARTIAL MISMATCH: Reject if partial match followed by different word than component name
"""

import csv
import pickle
import re
import sys
from pathlib import Path

PHASE_CACHE = Path("results/phase_cache/s_linker11")
BENCHMARK = Path("/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark")

DATASETS = {
    "mediastore": {
        "gs": "goldstandards/goldstandard_sad_2016-sam_2016.csv",
        "text": "text_2016/mediastore.txt",
    },
    "teastore": {
        "gs": "goldstandards/goldstandard_sad_2020-sam_2020.csv",
        "text": "text_2020/teastore.txt",
    },
    "teammates": {
        "gs": "goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "text": "text_2021/teammates.txt",
    },
    "bigbluebutton": {
        "gs": "goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "text": "text_2021/bigbluebutton.txt",
    },
    "jabref": {
        "gs": "goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "text": "text_2021/jabref.txt",
    },
}


def load_gold_standard(ds_name):
    """Load gold standard as set of (component_id, sentence_number) tuples."""
    path = BENCHMARK / ds_name / DATASETS[ds_name]["gs"]
    gold = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            comp_id = row["modelElementID"]
            sent_no = int(row["sentence"])
            gold.add((comp_id, sent_no))
    return gold


def load_sentences(ds_name):
    """Load sentences as dict: 1-indexed sentence number -> text."""
    path = BENCHMARK / ds_name / DATASETS[ds_name]["text"]
    sentences = {}
    with open(path) as f:
        for i, line in enumerate(f, 1):
            sentences[i] = line.strip()
    return sentences


def load_layer1(ds_name):
    """Load layer1 pickle: model_knowledge, raw_seed_links."""
    path = PHASE_CACHE / ds_name / "layer1.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def check_dotted_path(comp_name, sentence_text):
    """
    Filter 1: DOTTED-PATH
    Returns True if the component name appears ONLY inside dotted paths.
    Returns False (safe) if there's at least one clean standalone mention.
    """
    pattern = re.compile(re.escape(comp_name), re.IGNORECASE)
    matches = list(pattern.finditer(sentence_text))
    if not matches:
        # Component not found at all — shouldn't happen for seed links,
        # but don't reject
        return False

    for m in matches:
        start, end = m.start(), m.end()
        preceded_by_dot = start > 0 and sentence_text[start - 1] == "."
        # Check if followed by '.' + alpha
        followed_by_dot = (
            end < len(sentence_text)
            and sentence_text[end] == "."
            and end + 1 < len(sentence_text)
            and sentence_text[end + 1].isalpha()
        )
        if not preceded_by_dot and not followed_by_dot:
            # At least one clean mention — safe
            return False

    # ALL mentions are inside dotted paths
    return True


def check_lowercase_ambiguous(comp_name, sentence_text, ambiguous_names):
    """
    Filter 2: LOWERCASE AMBIGUOUS
    Returns True if component is ambiguous AND appears only in lowercase.
    """
    # Check if component is ambiguous (case-insensitive comparison)
    is_ambiguous = any(
        a.lower() == comp_name.lower() for a in ambiguous_names
    )
    if not is_ambiguous:
        return False

    # Find all mentions and check if ANY is capitalized
    pattern = re.compile(re.escape(comp_name), re.IGNORECASE)
    matches = list(pattern.finditer(sentence_text))
    if not matches:
        return False

    for m in matches:
        mention = m.group()
        # Check if first letter is uppercase
        if mention[0].isupper():
            return False

    # All mentions are lowercase and component is ambiguous
    return True


def check_partial_mismatch(comp_name, sentence_text):
    """
    Filter 3: PARTIAL MISMATCH
    For multi-word component names, check if the first word appears in the
    sentence followed by a DIFFERENT second word than in the component name.

    Returns True if partial mismatch detected (would reject).
    Returns False (safe) if no mismatch or single-word name.
    """
    words = comp_name.split()
    if len(words) < 2:
        return False

    first_word = words[0]
    expected_second = words[1]

    # Find all occurrences of the first word in the sentence
    pattern = re.compile(
        r"\b" + re.escape(first_word) + r"\b\s+(\w+)", re.IGNORECASE
    )
    matches = list(pattern.finditer(sentence_text))
    if not matches:
        return False

    has_correct_match = False
    has_wrong_match = False

    for m in matches:
        actual_second = m.group(1)
        if actual_second.lower() == expected_second.lower():
            has_correct_match = True
        else:
            has_wrong_match = True

    # Only reject if there's NO correct match and there IS a wrong match
    # (i.e., the component's full name never appears, only a partial with
    # a different second word)
    if has_wrong_match and not has_correct_match:
        return True

    return False


def main():
    total_tps_all = 0
    killed_all = {"dotted_path": [], "lowercase_ambiguous": [], "partial_mismatch": []}

    for ds_name in ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]:
        print(f"\n{'='*70}")
        print(f"  DATASET: {ds_name}")
        print(f"{'='*70}")

        gold = load_gold_standard(ds_name)
        sentences = load_sentences(ds_name)
        data = load_layer1(ds_name)

        raw_seeds = data["raw_seed_links"]
        mk = data["model_knowledge"]
        ambiguous_names = mk.ambiguous_names

        print(f"  Total seed links: {len(raw_seeds)}")
        print(f"  Gold standard pairs: {len(gold)}")
        print(f"  Ambiguous names: {ambiguous_names}")

        # Identify TP seeds
        tp_seeds = []
        fp_seeds = []
        for link in raw_seeds:
            pair = (link.component_id, link.sentence_number)
            if pair in gold:
                tp_seeds.append(link)
            else:
                fp_seeds.append(link)

        print(f"  TP seed links: {len(tp_seeds)}")
        print(f"  FP seed links: {len(fp_seeds)}")
        total_tps_all += len(tp_seeds)

        # Check each filter on TP seeds
        ds_killed = {"dotted_path": [], "lowercase_ambiguous": [], "partial_mismatch": []}

        for link in tp_seeds:
            sent_text = sentences.get(link.sentence_number, "")
            if not sent_text:
                print(f"  WARNING: No text for sentence {link.sentence_number}")
                continue

            # Filter 1: DOTTED-PATH
            if check_dotted_path(link.component_name, sent_text):
                ds_killed["dotted_path"].append(link)
                killed_all["dotted_path"].append((ds_name, link))

            # Filter 2: LOWERCASE AMBIGUOUS
            if check_lowercase_ambiguous(link.component_name, sent_text, ambiguous_names):
                ds_killed["lowercase_ambiguous"].append(link)
                killed_all["lowercase_ambiguous"].append((ds_name, link))

            # Filter 3: PARTIAL MISMATCH
            if check_partial_mismatch(link.component_name, sent_text):
                ds_killed["partial_mismatch"].append(link)
                killed_all["partial_mismatch"].append((ds_name, link))

        # Report per-dataset results
        for filter_name, killed_list in ds_killed.items():
            if killed_list:
                print(f"\n  !! {filter_name.upper()} would kill {len(killed_list)} TP(s):")
                for link in killed_list:
                    sent_text = sentences.get(link.sentence_number, "???")
                    print(f"     - [{link.component_name}] sent {link.sentence_number}: \"{sent_text[:120]}...\"" if len(sent_text) > 120 else f"     - [{link.component_name}] sent {link.sentence_number}: \"{sent_text}\"")
            else:
                print(f"  {filter_name.upper()}: 0 TPs killed (SAFE)")

        # Also show which FPs each filter WOULD catch (bonus info)
        print(f"\n  --- FP analysis (bonus) ---")
        for link in fp_seeds:
            sent_text = sentences.get(link.sentence_number, "")
            catches = []
            if check_dotted_path(link.component_name, sent_text):
                catches.append("DOTTED-PATH")
            if check_lowercase_ambiguous(link.component_name, sent_text, ambiguous_names):
                catches.append("LOWERCASE-AMBIG")
            if check_partial_mismatch(link.component_name, sent_text):
                catches.append("PARTIAL-MISMATCH")
            if catches:
                print(f"  FP caught by {', '.join(catches)}: [{link.component_name}] sent {link.sentence_number}: \"{sent_text[:100]}\"")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Total TPs checked across all datasets: {total_tps_all}")
    print()

    any_killed = False
    for filter_name in ["dotted_path", "lowercase_ambiguous", "partial_mismatch"]:
        killed = killed_all[filter_name]
        if killed:
            any_killed = True
            print(f"  {filter_name.upper()}: {len(killed)} TP(s) KILLED!")
            for ds, link in killed:
                print(f"    - {ds}: [{link.component_name}] sent {link.sentence_number} (id={link.component_id})")
        else:
            print(f"  {filter_name.upper()}: 0 TPs killed -- SAFE")

    print()
    if any_killed:
        print("  *** WARNING: Some filters kill TPs! Review above details. ***")
        return 1
    else:
        print("  ALL 3 FILTERS ARE SAFE: Zero TPs killed across all 5 datasets.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
