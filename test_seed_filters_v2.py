#!/usr/bin/env python3
"""
Verify 2 REFINED seed filters against all 151 true-positive seed links in S-Linker11.

Refined Filter 2 — Lowercase ambiguous (v2):
  - Only apply when component name starts with UPPERCASE letter
  - Check if component is in ambiguous_names
  - Check if component appears ONLY in lowercase in sentence text
  - If all three: reject

Refined Filter 3 — Partial mismatch (v2):
  - Only for multi-word component names
  - Extract first word of component name
  - Find first_word as STANDALONE word (not hyphenated, not dotted-path)
  - If found standalone AND next standalone word != second word of component: reject
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
    path = BENCHMARK / ds_name / DATASETS[ds_name]["gs"]
    gold = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gold.add((row["modelElementID"], int(row["sentence"])))
    return gold


def load_sentences(ds_name):
    path = BENCHMARK / ds_name / DATASETS[ds_name]["text"]
    sentences = {}
    with open(path) as f:
        for i, line in enumerate(f, 1):
            sentences[i] = line.strip()
    return sentences


def load_layer1(ds_name):
    path = PHASE_CACHE / ds_name / "layer1.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ===========================================================================
# REFINED FILTER 2 — Lowercase ambiguous (v2)
# ===========================================================================
def check_lowercase_ambiguous_v2(comp_name, sentence_text, ambiguous_names):
    """
    Returns (reject: bool, reason: str).

    Only apply when component name starts with UPPERCASE letter.
    Skip inherently lowercase names like "logic", "preferences".
    """
    # Condition 1: Component name must start with uppercase
    if not comp_name[0].isupper():
        return False, f"SKIP: comp '{comp_name}' starts lowercase"

    # Condition 2: Component must be in ambiguous_names
    is_ambiguous = any(a.lower() == comp_name.lower() for a in ambiguous_names)
    if not is_ambiguous:
        return False, f"NOT ambiguous"

    # Condition 3: Component appears ONLY in lowercase in the sentence
    pattern = re.compile(re.escape(comp_name), re.IGNORECASE)
    matches = list(pattern.finditer(sentence_text))
    if not matches:
        return False, f"NOT found in sentence"

    for m in matches:
        mention = m.group()
        if mention[0].isupper():
            return False, f"Found capitalized mention: '{mention}'"

    # All conditions met: reject
    return True, f"Ambiguous + uppercase comp + only lowercase mentions"


# ===========================================================================
# REFINED FILTER 3 — Partial mismatch (v2)
# ===========================================================================
def check_partial_mismatch_v2(comp_name, sentence_text):
    """
    Returns (reject: bool, reason: str, details: list).

    For multi-word component names:
    - Find first_word as STANDALONE (not inside hyphenated or dotted contexts)
    - If found standalone, check if next standalone word matches second word of component
    """
    words = comp_name.split()
    if len(words) < 2:
        return False, "SKIP: single-word name", []

    first_word = words[0]
    expected_second = words[1].lower()

    # Find all word-boundary matches of the first word
    # Then filter out matches that are inside hyphenated compounds or dotted paths
    pattern = re.compile(r"\b" + re.escape(first_word) + r"\b", re.IGNORECASE)
    matches = list(pattern.finditer(sentence_text))

    if not matches:
        return False, f"First word '{first_word}' not found in sentence", []

    details = []
    standalone_matches = []

    for m in matches:
        start, end = m.start(), m.end()
        # Check context: is it inside a hyphenated compound?
        preceded_by_hyphen = start > 0 and sentence_text[start - 1] == "-"
        followed_by_hyphen = end < len(sentence_text) and sentence_text[end] == "-"
        # Is it inside a dotted path?
        preceded_by_dot = start > 0 and sentence_text[start - 1] == "."
        followed_by_dot = (
            end < len(sentence_text)
            and sentence_text[end] == "."
            and end + 1 < len(sentence_text)
            and sentence_text[end + 1].isalpha()
        )

        is_standalone = not (preceded_by_hyphen or followed_by_hyphen or preceded_by_dot or followed_by_dot)

        # Show surrounding context (±30 chars)
        ctx_start = max(0, start - 30)
        ctx_end = min(len(sentence_text), end + 30)
        ctx = sentence_text[ctx_start:ctx_end]
        # Mark the match position
        marker_start = start - ctx_start
        marker_end = end - ctx_start

        # Find the next word after the match
        rest = sentence_text[end:]
        next_word_match = re.match(r"\s+(\w+)", rest)
        next_word = next_word_match.group(1) if next_word_match else "<END>"

        detail = {
            "match_text": m.group(),
            "position": start,
            "context": ctx,
            "preceded_by_hyphen": preceded_by_hyphen,
            "followed_by_hyphen": followed_by_hyphen,
            "preceded_by_dot": preceded_by_dot,
            "followed_by_dot": followed_by_dot,
            "is_standalone": is_standalone,
            "next_word": next_word,
            "next_matches_expected": next_word.lower() == expected_second if next_word != "<END>" else False,
        }
        details.append(detail)

        if is_standalone:
            standalone_matches.append(detail)

    if not standalone_matches:
        return False, f"No standalone matches of '{first_word}' (all inside compounds/paths)", details

    # Check if any standalone match is followed by the correct second word
    has_correct = any(d["next_matches_expected"] for d in standalone_matches)
    has_wrong = any(not d["next_matches_expected"] for d in standalone_matches)

    if has_wrong and not has_correct:
        return True, f"Standalone '{first_word}' never followed by '{expected_second}'", details

    return False, f"Has correct match(es)", details


def main():
    total_tps = 0
    total_fps = 0
    killed_tps_f2 = []
    killed_tps_f3 = []
    caught_fps_f2 = []
    caught_fps_f3 = []

    for ds_name in ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]:
        print(f"\n{'='*80}")
        print(f"  DATASET: {ds_name}")
        print(f"{'='*80}")

        gold = load_gold_standard(ds_name)
        sentences = load_sentences(ds_name)
        data = load_layer1(ds_name)

        raw_seeds = data["raw_seed_links"]
        mk = data["model_knowledge"]
        ambiguous_names = mk.ambiguous_names

        # Split into TP and FP
        tp_seeds = []
        fp_seeds = []
        for link in raw_seeds:
            if (link.component_id, link.sentence_number) in gold:
                tp_seeds.append(link)
            else:
                fp_seeds.append(link)

        print(f"  Seeds: {len(raw_seeds)} total, {len(tp_seeds)} TP, {len(fp_seeds)} FP")
        print(f"  Ambiguous names: {ambiguous_names}")
        total_tps += len(tp_seeds)
        total_fps += len(fp_seeds)

        # =====================================================================
        # FILTER 2 ANALYSIS: Lowercase ambiguous v2
        # =====================================================================
        print(f"\n  --- FILTER 2: Lowercase Ambiguous v2 ---")
        print(f"  (Only when comp starts uppercase + ambiguous + only lowercase in text)")

        # Show all ambiguous names and their case
        if ambiguous_names:
            print(f"\n  Ambiguous component names:")
            for name in sorted(ambiguous_names):
                starts_upper = name[0].isupper()
                print(f"    '{name}' — starts {'UPPER' if starts_upper else 'lower'} → {'ELIGIBLE' if starts_upper else 'EXCLUDED by v2 gate'}")

        # Check TPs
        print(f"\n  Checking {len(tp_seeds)} TPs...")
        for link in tp_seeds:
            sent_text = sentences.get(link.sentence_number, "")
            reject, reason = check_lowercase_ambiguous_v2(link.component_name, sent_text, ambiguous_names)
            if reject:
                killed_tps_f2.append((ds_name, link, reason, sent_text))
                print(f"    !! TP KILLED: [{link.component_name}] sent {link.sentence_number}")
                print(f"       Reason: {reason}")
                print(f"       Text: \"{sent_text[:150]}\"")
            # Show details for all ambiguous TPs (even if not killed)
            elif any(a.lower() == link.component_name.lower() for a in ambiguous_names):
                print(f"    SAFE TP (ambiguous): [{link.component_name}] sent {link.sentence_number} — {reason}")
                if "Found capitalized" in reason:
                    print(f"       Text: \"{sent_text[:150]}\"")

        # Check FPs
        print(f"\n  Checking {len(fp_seeds)} FPs...")
        for link in fp_seeds:
            sent_text = sentences.get(link.sentence_number, "")
            reject, reason = check_lowercase_ambiguous_v2(link.component_name, sent_text, ambiguous_names)
            if reject:
                caught_fps_f2.append((ds_name, link, reason, sent_text))
                print(f"    FP CAUGHT: [{link.component_name}] sent {link.sentence_number}")
                print(f"       Reason: {reason}")
                print(f"       Text: \"{sent_text[:150]}\"")
            elif any(a.lower() == link.component_name.lower() for a in ambiguous_names):
                print(f"    FP MISSED (ambiguous but safe): [{link.component_name}] sent {link.sentence_number} — {reason}")
                print(f"       Text: \"{sent_text[:150]}\"")

        # =====================================================================
        # FILTER 3 ANALYSIS: Partial mismatch v2
        # =====================================================================
        print(f"\n  --- FILTER 3: Partial Mismatch v2 ---")
        print(f"  (Multi-word only, standalone match, next word check)")

        # Show all multi-word component names in seeds
        multi_word_comps = set()
        for link in raw_seeds:
            if len(link.component_name.split()) > 1:
                multi_word_comps.add(link.component_name)
        if multi_word_comps:
            print(f"\n  Multi-word components in seeds: {sorted(multi_word_comps)}")
        else:
            print(f"\n  No multi-word components in seeds — filter N/A")

        # Check TPs (show all multi-word details)
        print(f"\n  Checking {len(tp_seeds)} TPs...")
        for link in tp_seeds:
            sent_text = sentences.get(link.sentence_number, "")
            reject, reason, details = check_partial_mismatch_v2(link.component_name, sent_text)

            if link.component_name.split().__len__() < 2:
                continue  # Skip single-word in output

            if reject:
                killed_tps_f3.append((ds_name, link, reason, sent_text, details))
                print(f"\n    !! TP KILLED: [{link.component_name}] sent {link.sentence_number}")
                print(f"       Reason: {reason}")
                print(f"       Text: \"{sent_text[:200]}\"")
                for d in details:
                    standalone_mark = "STANDALONE" if d["is_standalone"] else "COMPOUND"
                    print(f"       Match: '{d['match_text']}' @{d['position']} [{standalone_mark}] next='{d['next_word']}' expected='{link.component_name.split()[1]}' correct={d['next_matches_expected']}")
                    print(f"         Context: ...{d['context']}...")
            else:
                print(f"\n    SAFE TP: [{link.component_name}] sent {link.sentence_number} — {reason}")
                if details:
                    print(f"       Text: \"{sent_text[:200]}\"")
                    for d in details:
                        standalone_mark = "STANDALONE" if d["is_standalone"] else "COMPOUND"
                        next_ok = "YES" if d["next_matches_expected"] else "NO"
                        print(f"       Match: '{d['match_text']}' @{d['position']} [{standalone_mark}] next='{d['next_word']}' match={next_ok}")
                        print(f"         Context: ...{d['context']}...")

        # Check FPs
        print(f"\n  Checking {len(fp_seeds)} FPs...")
        for link in fp_seeds:
            sent_text = sentences.get(link.sentence_number, "")
            reject, reason, details = check_partial_mismatch_v2(link.component_name, sent_text)

            if link.component_name.split().__len__() < 2:
                continue

            if reject:
                caught_fps_f3.append((ds_name, link, reason, sent_text, details))
                print(f"\n    FP CAUGHT: [{link.component_name}] sent {link.sentence_number}")
                print(f"       Reason: {reason}")
                print(f"       Text: \"{sent_text[:200]}\"")
                for d in details:
                    standalone_mark = "STANDALONE" if d["is_standalone"] else "COMPOUND"
                    print(f"       Match: '{d['match_text']}' @{d['position']} [{standalone_mark}] next='{d['next_word']}' expected='{link.component_name.split()[1]}' correct={d['next_matches_expected']}")
                    print(f"         Context: ...{d['context']}...")
            else:
                print(f"\n    FP MISSED: [{link.component_name}] sent {link.sentence_number} — {reason}")
                if details:
                    print(f"       Text: \"{sent_text[:200]}\"")
                    for d in details:
                        standalone_mark = "STANDALONE" if d["is_standalone"] else "COMPOUND"
                        next_ok = "YES" if d["next_matches_expected"] else "NO"
                        print(f"       Match: '{d['match_text']}' @{d['position']} [{standalone_mark}] next='{d['next_word']}' match={next_ok}")
                        print(f"         Context: ...{d['context']}...")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"  Total seeds: {total_tps} TP + {total_fps} FP = {total_tps + total_fps}")
    print()

    # Filter 2 summary
    print(f"  FILTER 2 (Lowercase Ambiguous v2):")
    print(f"    TPs killed: {len(killed_tps_f2)}")
    if killed_tps_f2:
        for ds, link, reason, text in killed_tps_f2:
            print(f"      - {ds}: [{link.component_name}] sent {link.sentence_number}")
    print(f"    FPs caught: {len(caught_fps_f2)}")
    if caught_fps_f2:
        for ds, link, reason, text in caught_fps_f2:
            print(f"      - {ds}: [{link.component_name}] sent {link.sentence_number}: \"{text[:100]}\"")
    print()

    # Filter 3 summary
    print(f"  FILTER 3 (Partial Mismatch v2):")
    print(f"    TPs killed: {len(killed_tps_f3)}")
    if killed_tps_f3:
        for ds, link, reason, text, details in killed_tps_f3:
            print(f"      - {ds}: [{link.component_name}] sent {link.sentence_number}")
    print(f"    FPs caught: {len(caught_fps_f3)}")
    if caught_fps_f3:
        for ds, link, reason, text, details in caught_fps_f3:
            print(f"      - {ds}: [{link.component_name}] sent {link.sentence_number}: \"{text[:100]}\"")
    print()

    # Verdict
    total_killed = len(killed_tps_f2) + len(killed_tps_f3)
    total_caught = len(caught_fps_f2) + len(caught_fps_f3)
    if total_killed == 0:
        print(f"  VERDICT: BOTH FILTERS SAFE — 0 TPs killed, {total_caught} FPs caught")
        return 0
    else:
        print(f"  *** WARNING: {total_killed} TP(s) killed! ***")
        return 1


if __name__ == "__main__":
    sys.exit(main())
