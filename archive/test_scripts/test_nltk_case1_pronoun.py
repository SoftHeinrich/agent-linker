#!/usr/bin/env python3
"""NLTK Case 1: POS-based pronoun detection for deterministic coreference.

Compares regex approach vs NLTK POS tagger for finding pronoun-starting sentences.
Tests on TeaStore and BBB benchmark data. Shows which gold-standard coref links
each approach can recover without LLM.
"""
import csv
import re
import sys
sys.path.insert(0, "src")

import nltk
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk import word_tokenize, pos_tag

from llm_sad_sam.core import DocumentLoader
from llm_sad_sam.pcm_parser import parse_pcm_repository

BASE = "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"

DATASETS = {
    "teastore": {
        "text": f"{BASE}/teastore/text_2020/teastore.txt",
        "model": f"{BASE}/teastore/model_2020/pcm/teastore.repository",
        "gold": f"{BASE}/teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    },
    "bigbluebutton": {
        "text": f"{BASE}/bigbluebutton/text_2021/bigbluebutton.txt",
        "model": f"{BASE}/bigbluebutton/model_2021/pcm/bbb.repository",
        "gold": f"{BASE}/bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
}

# Pronouns that indicate coreference continuation
COREF_PRONOUNS = {'it', 'its', 'they', 'their', 'this', 'these'}


def has_standalone_mention(comp_name, text):
    """Check if component name appears cleanly in text (case-insensitive for single words)."""
    pattern = rf'\b{re.escape(comp_name)}\b'
    for m in re.finditer(pattern, text, re.IGNORECASE):
        s, e = m.start(), m.end()
        if s > 0 and text[s-1] == '.':
            continue
        if e < len(text) and text[e] == '.' and e+1 < len(text) and text[e+1].isalpha():
            continue
        if s > 0 and text[s-1] == '-':
            continue
        return True
    return False


def detect_pronoun_start_regex(text):
    """Current V18 regex approach."""
    return bool(re.match(r'^(It|As such, it)\b', text))


def detect_pronoun_start_nltk(text):
    """NLTK POS approach: detect if sentence starts with a pronoun subject."""
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    # Check first 4 tokens for pronoun patterns
    for i, (word, tag) in enumerate(tags[:5]):
        if word.lower() in COREF_PRONOUNS:
            if tag in ('PRP', 'PRP$', 'DT'):
                return True, word
    return False, None


def resolve_to_component(sent_num, sentences, comp_names, sent_map, lookback=3):
    """Look back up to N sentences for unambiguous component mention."""
    for offset in range(1, lookback + 1):
        prev_num = sent_num - offset
        prev = sent_map.get(prev_num)
        if not prev:
            continue
        mentioned = set()
        for cn in comp_names:
            if has_standalone_mention(cn, prev.text):
                mentioned.add(cn)
        if len(mentioned) == 1:
            return mentioned.pop(), prev_num
        elif len(mentioned) > 1:
            return None, None  # Ambiguous
    return None, None


def main():
    for ds_name, paths in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"  {ds_name.upper()}")
        print(f"{'='*70}")

        sentences = DocumentLoader.load_sentences(paths["text"])
        components = parse_pcm_repository(paths["model"])
        comp_names = [c.name for c in components]
        name_to_id = {c.name: c.id for c in components}
        id_to_name = {c.id: c.name for c in components}
        sent_map = DocumentLoader.build_sent_map(sentences)

        # Load gold standard
        gold = set()
        with open(paths["gold"]) as f:
            reader = csv.DictReader(f)
            for row in reader:
                gold.add((int(row["sentence"]), row["modelElementID"]))

        # Find sentences where component is NOT explicitly named (coref needed)
        coref_gold = set()
        for snum, cid in gold:
            cname = id_to_name.get(cid, "?")
            sent = sent_map.get(snum)
            if sent and not has_standalone_mention(cname, sent.text):
                coref_gold.add((snum, cid, cname))

        print(f"\nComponents: {', '.join(comp_names)}")
        print(f"Gold links: {len(gold)} total, {len(coref_gold)} need coreference")
        if coref_gold:
            print(f"  Coref gold: {sorted([(s, c) for s, _, c in coref_gold])}")

        # Test both approaches
        print(f"\n{'Sent':>4} | {'Regex':>5} | {'NLTK':>12} | {'Resolved':>20} | {'Gold?':>5} | Text")
        print("-" * 100)

        regex_hits = []
        nltk_hits = []

        for sent in sentences:
            r_match = detect_pronoun_start_regex(sent.text)
            n_match, n_word = detect_pronoun_start_nltk(sent.text)

            if not r_match and not n_match:
                continue

            resolved, from_sent = resolve_to_component(
                sent.number, sentences, comp_names, sent_map, lookback=3)

            # Check if this would be a gold hit
            gold_hit = False
            if resolved:
                cid = name_to_id.get(resolved)
                if cid and (sent.number, cid) in gold:
                    gold_hit = True

            if r_match and resolved:
                regex_hits.append((sent.number, resolved, gold_hit))
            if n_match and resolved:
                nltk_hits.append((sent.number, resolved, gold_hit))

            print(f"S{sent.number:>3} | {str(r_match):>5} | {f'PRP:{n_word}' if n_match else 'no':>12} | "
                  f"{f'{resolved} (S{from_sent})' if resolved else 'ambig/none':>20} | "
                  f"{'TP' if gold_hit else ('FP' if resolved else '-'):>5} | {sent.text[:60]}")

        # Summary
        print(f"\n--- Summary for {ds_name} ---")
        r_tp = sum(1 for _, _, g in regex_hits if g)
        r_fp = sum(1 for _, _, g in regex_hits if not g)
        n_tp = sum(1 for _, _, g in nltk_hits if g)
        n_fp = sum(1 for _, _, g in nltk_hits if not g)
        print(f"Regex:  {len(regex_hits)} links ({r_tp} TP, {r_fp} FP)")
        print(f"NLTK:   {len(nltk_hits)} links ({n_tp} TP, {n_fp} FP)")
        print(f"Coref gold not recovered (regex): {len(coref_gold) - r_tp}")
        print(f"Coref gold not recovered (NLTK):  {len(coref_gold) - n_tp}")


if __name__ == "__main__":
    main()
