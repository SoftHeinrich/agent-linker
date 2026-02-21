#!/usr/bin/env python3
"""NLTK Case 2: NP chunking for better component-mention detection.

Uses NLTK noun phrase extraction to find component references that
simple string matching misses. Tests against gold standard to find
links that NP-aware matching recovers vs regex-only matching.

Key question: Can NP chunking help with:
- CamelCase split detection ("File Storage" → FileStorage)
- Generic vs specific usage ("the persistence layer" vs "Persistence")
- Compound mentions ("the HTML5 client" → "HTML5 Client")
"""
import csv
import re
import sys
sys.path.insert(0, "src")

import nltk
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)
from nltk import word_tokenize, pos_tag
from nltk.chunk import RegexpParser

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
    "mediastore": {
        "text": f"{BASE}/mediastore/text_2016/mediastore.txt",
        "model": f"{BASE}/mediastore/model_2016/pcm/ms.repository",
        "gold": f"{BASE}/mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    },
}

# NP chunking grammar
NP_GRAMMAR = r"""
    NP: {<DT|PRP\$>?<JJ|NN.*>*<NN.*>}    # Standard NP
    NP: {<NNP>+}                            # Proper noun sequence
"""
NP_PARSER = RegexpParser(NP_GRAMMAR)


def extract_noun_phrases(text):
    """Extract all noun phrases from text using NLTK chunking."""
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    tree = NP_PARSER.parse(tagged)
    nps = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        np_text = ' '.join(word for word, tag in subtree.leaves())
        nps.append(np_text)
    return nps


def camelcase_split(name):
    """Split CamelCase: 'FileStorage' → 'File Storage'."""
    parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    parts = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', parts)
    return parts


def has_standalone_mention(comp_name, text):
    """Simple regex component matching."""
    pattern = rf'\b{re.escape(comp_name)}\b'
    return bool(re.search(pattern, text, re.IGNORECASE))


def np_matches_component(nps, comp_name):
    """Check if any extracted NP matches a component name."""
    comp_lower = comp_name.lower()
    cc_split = camelcase_split(comp_name).lower()

    matches = []
    for np in nps:
        np_lower = np.lower()
        # Direct match
        if comp_lower in np_lower:
            matches.append(("direct", np))
            continue
        # CamelCase split match: "File Storage" in NP for "FileStorage"
        if cc_split != comp_lower and cc_split in np_lower:
            matches.append(("camel_split", np))
            continue
        # Partial tail match: "persistence provider" ends with component partial
        # Only for multi-word components
        if ' ' in comp_name:
            last_word = comp_name.split()[-1].lower()
            if np_lower.endswith(last_word) and len(np_lower) > len(last_word):
                matches.append(("partial_tail", np))
                continue
        # NP contains component as head noun
        np_words = np_lower.split()
        if np_words and np_words[-1] == comp_lower:
            matches.append(("head_noun", np))
    return matches


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

        # Load gold
        gold = set()
        with open(paths["gold"]) as f:
            reader = csv.DictReader(f)
            for row in reader:
                gold.add((int(row["sentence"]), row["modelElementID"]))

        print(f"Components: {', '.join(comp_names)}")
        print(f"CamelCase splits: {[(c, camelcase_split(c)) for c in comp_names if camelcase_split(c) != c]}")
        print(f"Gold links: {len(gold)}")

        # For each sentence, compare regex vs NP-based matching
        regex_found = set()  # (snum, cid)
        np_found = set()
        np_only = []  # Found by NP but not regex

        for sent in sentences:
            nps = extract_noun_phrases(sent.text)

            for comp in components:
                key = (sent.number, comp.id)

                # Regex approach
                regex_hit = has_standalone_mention(comp.name, sent.text)
                if regex_hit:
                    regex_found.add(key)

                # NP approach
                np_matches = np_matches_component(nps, comp.name)
                if np_matches:
                    np_found.add(key)
                    if not regex_hit:
                        is_gold = key in gold
                        np_only.append((sent.number, comp.name, np_matches, is_gold, sent.text[:70]))

        # Gold coverage
        regex_tp = regex_found & gold
        np_tp = np_found & gold
        np_only_tp = (np_found - regex_found) & gold

        print(f"\n--- Matching Results ---")
        print(f"Regex found:  {len(regex_found)} matches ({len(regex_tp)} are gold TPs)")
        print(f"NP found:     {len(np_found)} matches ({len(np_tp)} are gold TPs)")
        print(f"NP-only:      {len(np_found - regex_found)} additional matches ({len(np_only_tp)} are gold TPs)")

        if np_only:
            print(f"\n--- NP-only matches (not found by regex) ---")
            # Show gold hits first
            gold_first = sorted(np_only, key=lambda x: (not x[3], x[0]))
            for snum, cname, matches, is_gold, text in gold_first[:20]:
                tag = "GOLD" if is_gold else "    "
                match_desc = "; ".join(f"{t}:{np}" for t, np in matches)
                print(f"  [{tag}] S{snum} -> {cname}: {match_desc}")
                print(f"         {text}")

        # Gold links missed by both
        missed = gold - regex_found - np_found
        if missed:
            print(f"\n--- Gold links missed by BOTH approaches ---")
            for snum, cid in sorted(missed):
                cname = id_to_name.get(cid, "?")
                sent = sent_map.get(snum)
                print(f"  S{snum} -> {cname}: {sent.text[:70] if sent else '?'}")


if __name__ == "__main__":
    main()
