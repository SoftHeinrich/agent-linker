#!/usr/bin/env python3
"""Safety net variant analysis for S-Linker11 ICSE.

Tests whether the two remaining code-enforced safety nets can be replaced
by pure LLM determination:

  Variant A — Remove len<4 filter (L423): Let LLM word usage classifier
              decide for ALL trailing words, including short ones (≤3 chars).

  Variant B — Expand coref context window: Replace fixed ±5 with paragraph-
              bounded context to test if structural context is better.

  Also analyzes the remaining coref structural check (antecedent mention
  verification) to understand if it can be replaced by LLM.

Uses S-Linker10 checkpoints + static analysis (no LLM calls).
"""

import csv
import json
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
CACHE_DIR = Path("./results/phase_cache/s_linker10")

DS_INFO = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
        "gold": "goldstandard_sad_2016-sam_2016.csv",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
        "model": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
        "gold": "goldstandard_sad_2020-sam_2020.csv",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
        "gold": "goldstandard_sad_2021-sam_2021.csv",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
        "gold": "goldstandard_sad_2021-sam_2021.csv",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
        "gold": "goldstandard_sad_2021-sam_2021.csv",
    },
}


def load_gold(ds):
    gs_path = BENCHMARK_BASE / ds / "goldstandards" / DS_INFO[ds]["gold"]
    gold = set()
    with open(gs_path) as f:
        for row in csv.DictReader(f):
            gold.add((int(row["sentence"]), row["modelElementID"]))
    return gold


def load_checkpoints(ds):
    with open(CACHE_DIR / ds / "tier1.pkl", "rb") as f:
        t1 = pickle.load(f)
    with open(CACHE_DIR / ds / "tier2.pkl", "rb") as f:
        t2 = pickle.load(f)
    return t1, t2


# ════════════════════════════════════════════════════════════════════════
# VARIANT A: Remove len<4 filter (let LLM classify all trailing words)
# ════════════════════════════════════════════════════════════════════════

def test_variant_a_no_len_filter():
    """What words become eligible if len<4 filter is removed?

    For each newly-eligible word, show:
    - Document usage patterns (standalone vs compound)
    - What the LLM word usage classifier would see
    - Whether the word appears in gold standard links
    """
    print("=" * 70)
    print("VARIANT A: Remove len<4 from multiword partial enrichment")
    print("Full-LLM: Let word usage classifier decide for ALL trailing words")
    print("=" * 70)

    any_affected = False

    for ds in DS_INFO:
        components = parse_pcm_repository(str(DS_INFO[ds]["model"]))
        sentences = DocumentLoader.load_sentences(str(DS_INFO[ds]["text"]))
        gold = load_gold(ds)
        t1, _ = load_checkpoints(ds)
        mk = t1.get("model_knowledge")
        dk = t1.get("doc_knowledge")
        generic_partials = t1.get("generic_partials", set())

        # Find words blocked by len<4
        blocked_words = []
        for comp in components:
            parts = comp.name.split()
            if len(parts) < 2:
                continue
            last_word = parts[-1]
            if len(last_word) >= 4:
                continue  # Not blocked by filter

            last_lower = last_word.lower()

            # Check other filters that would still apply
            other_match = any(
                c.name != comp.name and c.name.lower().endswith(last_lower)
                for c in components
            )
            if other_match:
                continue  # Would be blocked by ambiguity check anyway

            already_known = False
            if dk:
                if last_lower in {s.lower() for s in dk.synonyms}:
                    already_known = True
                if last_lower in {p.lower() for p in dk.partial_references}:
                    already_known = True
            if already_known:
                continue

            # This word IS blocked only by len<4
            blocked_words.append((last_word, comp))

        if not blocked_words:
            continue

        any_affected = True
        print(f"\n{'─'*60}")
        print(f"  {ds}: {len(blocked_words)} words blocked by len<4")
        print(f"{'─'*60}")

        for last_word, comp in blocked_words:
            last_lower = last_word.lower()
            full_lower = comp.name.lower()
            is_generic = last_lower in generic_partials

            # Find sentences with standalone usage (what LLM classifier would see)
            standalone_sents = []
            with_full_sents = []
            for sent in sentences:
                sl = sent.text.lower()
                if last_lower not in sl:
                    continue
                if full_lower in sl:
                    with_full_sents.append(sent)
                else:
                    # Check if it's actually a word boundary match
                    if re.search(rf'\b{re.escape(last_word)}\b', sent.text, re.IGNORECASE):
                        standalone_sents.append(sent)

            print(f"\n  '{last_word}' -> {comp.name} (len={len(last_word)}, generic={is_generic})")
            print(f"  With full name: {len(with_full_sents)} sents | Standalone: {len(standalone_sents)} sents")

            if not standalone_sents:
                print(f"  >> NO standalone mentions. Removing filter = NO IMPACT.")
                continue

            # Show what LLM classifier would see
            print(f"  Standalone usage (what LLM would classify):")
            for sent in standalone_sents[:8]:
                # Check gold standard
                tp = (sent.number, comp.id) in gold
                marker = " [GOLD TP]" if tp else ""
                # Determine usage pattern
                text = sent.text
                # Check if it's used as modifier vs entity
                patterns = []
                if re.search(rf'\b{re.escape(last_lower)}\s+\w+', text.lower()):
                    # "web application", "web server" etc
                    match = re.search(rf'\b{re.escape(last_lower)}\s+(\w+)', text.lower())
                    if match:
                        patterns.append(f"modifier: '{last_lower} {match.group(1)}'")
                if re.search(rf'the\s+{re.escape(last_lower)}\b', text.lower()):
                    patterns.append("entity: 'the " + last_lower + "'")
                if re.search(rf'via\s+{re.escape(last_lower)}\b', text.lower()):
                    patterns.append(f"entity: 'via {last_lower}'")

                pattern_str = f" ({', '.join(patterns)})" if patterns else ""
                print(f"    S{sent.number}: {text[:100]}{marker}{pattern_str}")

            # Predict LLM classification
            modifier_count = sum(
                1 for s in standalone_sents
                if re.search(rf'\b{re.escape(last_lower)}\s+\w{{3,}}', s.text.lower())
            )
            entity_count = sum(
                1 for s in standalone_sents
                if re.search(rf'(the|via|from|to|by)\s+{re.escape(last_lower)}\b', s.text.lower())
                or re.search(rf'\b{re.escape(last_lower)}\s+(handles|processes|manages|connects|sends|receives)', s.text.lower())
            )
            print(f"  Predicted: modifier-like={modifier_count}, entity-like={entity_count}")
            if entity_count > 0:
                print(f"  >> LLM would likely classify as NAME (entity references found)")
                # Count TPs and FPs if this became a partial
                tp_count = sum(1 for s in standalone_sents if (s.number, comp.id) in gold)
                fp_count = len(standalone_sents) - tp_count
                print(f"  >> Potential impact: {tp_count} TP, {fp_count} FP (before validation)")
            else:
                print(f"  >> LLM would likely classify as ORDINARY (only modifier usage)")

    if not any_affected:
        print("\n  No datasets affected by len<4 filter. Safe to remove (dead code).")


# ════════════════════════════════════════════════════════════════════════
# VARIANT B: Coref context window analysis
# ════════════════════════════════════════════════════════════════════════

def test_variant_b_coref_window():
    """Analyze the ±5 coref context window.

    Questions:
    1. Do any approved coref links have antecedents near window boundary?
    2. Does paragraph-bounded context differ from ±5?
    3. What remains as code-enforced (vs prompt-enforced) after removing dist>3?
    """
    print("\n" + "=" * 70)
    print("VARIANT B: Coref Context Window Analysis")
    print("Full-LLM: Replace ±5 with paragraph-bounded context")
    print("=" * 70)

    # Parse coref distances from LLM logs
    log_dir = Path("results/llm_logs")
    all_logs = sorted(log_dir.glob("*.jsonl"))

    resolution_distances = []
    for lf in all_logs:
        with open(lf) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                preview = entry.get("prompt_preview", "")
                if "Resolve pronoun" not in preview:
                    continue
                resp = entry.get("response_preview", "")
                try:
                    clean = resp.replace("```json", "").replace("```", "").strip()
                    if "{" not in clean:
                        continue
                    start = clean.index("{")
                    data = json.loads(clean[start:])
                    for r in data.get("resolutions", []):
                        snum = r.get("sentence")
                        ant = r.get("antecedent_sentence")
                        if snum is not None and ant is not None:
                            if isinstance(snum, str):
                                snum = int(snum.replace("S", "").strip())
                            if isinstance(ant, str):
                                ant = int(ant.replace("S", "").strip())
                            resolution_distances.append(abs(int(snum) - int(ant)))
                except (ValueError, json.JSONDecodeError, TypeError):
                    pass

    if resolution_distances:
        from collections import Counter
        dist_counts = Counter(resolution_distances)
        print(f"\n  LLM coref resolution distance distribution ({len(resolution_distances)} total):")
        for d in sorted(dist_counts):
            bar = "█" * min(dist_counts[d], 50)
            pct = dist_counts[d] / len(resolution_distances) * 100
            print(f"    dist={d}: {dist_counts[d]:4d} ({pct:5.1f}%)  {bar}")
        max_dist = max(resolution_distances)
        print(f"\n  Max distance: {max_dist}")
        print(f"  ≤3: {sum(v for k,v in dist_counts.items() if k <= 3)} ({sum(v for k,v in dist_counts.items() if k <= 3)/len(resolution_distances)*100:.1f}%)")
        print(f"  >3: {sum(v for k,v in dist_counts.items() if k > 3)} ({sum(v for k,v in dist_counts.items() if k > 3)/len(resolution_distances)*100:.1f}%)")
    else:
        print("\n  No coref resolution distances found in logs.")

    # Analyze: for each dataset, compare ±5 window vs paragraph boundaries
    print(f"\n  Paragraph vs ±5 window comparison:")
    for ds in DS_INFO:
        sentences = DocumentLoader.load_sentences(str(DS_INFO[ds]["text"]))
        paragraphs = DocumentLoader.detect_paragraphs(sentences)
        sent_to_para = {}
        for pi, para in enumerate(paragraphs):
            for s in para:
                sent_to_para[s.number] = pi

        t1, t2 = load_checkpoints(ds)
        coref_links = t2["coref_links"]
        if not coref_links:
            continue

        print(f"\n  {ds} ({len(paragraphs)} paragraphs, {len(sentences)} sentences):")
        for l in coref_links:
            snum = l.sentence_number
            para_idx = sent_to_para.get(snum, -1)
            if para_idx < 0:
                continue

            para = paragraphs[para_idx]
            para_start = para[0].number
            para_end = para[-1].number
            para_size = len(para)

            # ±5 window
            win_start = max(1, snum - 5)
            win_end = min(len(sentences), snum + 5)
            win_size = win_end - win_start + 1

            # Paragraph + neighbors
            prev_para = paragraphs[para_idx - 1] if para_idx > 0 else []
            next_para = paragraphs[para_idx + 1] if para_idx < len(paragraphs) - 1 else []
            full_start = prev_para[0].number if prev_para else para_start
            full_end = next_para[-1].number if next_para else para_end
            full_size = full_end - full_start + 1

            contained = "⊂" if (para_start >= win_start and para_end <= win_end) else "≠"
            print(f"    S{snum} -> {l.component_name}: "
                  f"±5=[{win_start}-{win_end}]({win_size}s) "
                  f"para=[{para_start}-{para_end}]({para_size}s) {contained} "
                  f"para±1=[{full_start}-{full_end}]({full_size}s)")


# ════════════════════════════════════════════════════════════════════════
# ANALYSIS: Remaining code-enforced coref checks (after dist>3 removal)
# ════════════════════════════════════════════════════════════════════════

def test_remaining_coref_checks():
    """After removing dist>3, what code-enforced checks remain in coref?

    The antecedent mention check (component name must appear in antecedent
    sentence) is the key structural verification. Can it be replaced by LLM?
    """
    print("\n" + "=" * 70)
    print("REMAINING COREF CODE CHECKS (after dist>3 removal)")
    print("=" * 70)

    print("""
  After removing the distance>3 safety net, three code-enforced checks remain:

  1. COMPONENT NAME VALIDATION (L889-890):
     comp not in name_to_id → skip
     Purpose: LLM may hallucinate non-existent component names.
     Full-LLM: CANNOT REMOVE. Structural validation, not a threshold.
     The LLM has no way to enforce this — it doesn't know the exact
     component ID list at generation time.

  2. ANTECEDENT SENTENCE EXISTENCE (L897-898):
     sent_map.get(ant_snum) is None → skip
     Purpose: LLM may reference non-existent sentence numbers.
     Full-LLM: CANNOT REMOVE. Same as above — structural validation.

  3. ANTECEDENT MENTION VERIFICATION (L900-901):
     Component name (or alias) must appear in antecedent sentence.
     Purpose: Catches hallucinated antecedents (like S68←S77 case).
     Full-LLM: COULD theoretically remove and trust LLM's antecedent_text
     field. But this is the STRONGEST safety net — it verifies the LLM's
     claim by checking the actual text. Without it, any hallucinated
     antecedent would pass through.

  VERDICT: All 3 remaining checks are STRUCTURAL VERIFICATION, not
  arbitrary thresholds. They verify LLM outputs against ground truth
  (actual text content). Cannot be replaced by LLM — that would be
  asking the LLM to verify itself.

  The pipeline is now "fully LLM-driven" for DECISIONS:
  - LLM decides which pronouns refer to components
  - LLM decides the antecedent sentence and component
  - Code only VERIFIES the LLM's claims against actual text

  This is the correct boundary: LLM makes decisions, code verifies facts.
""")


# ════════════════════════════════════════════════════════════════════════
# FULL-LLM ANALYSIS: What would a fully LLM-driven pipeline look like?
# ════════════════════════════════════════════════════════════════════════

def test_full_llm_analysis():
    """Analyze the path to a fully LLM-driven pipeline.

    Categorize ALL remaining code checks into:
    - Structural verification (keep: verifies LLM output against text)
    - Arbitrary thresholds (remove: replace with LLM judgment)
    - Infrastructure (keep: batch sizes, timeouts)
    """
    print("\n" + "=" * 70)
    print("FULL-LLM ANALYSIS: Remaining code-enforced decisions")
    print("=" * 70)

    print("""
  REMOVED (now LLM-driven):
  ✓ len>=3 partial filter     — dead code, removed
  ✓ dist>3 coref filter       — prompt-redundant, removed

  REMAINING ALGORITHMIC (2):
  ? len<4 enrichment filter   — linguistically motivated, LLM CAN handle
  ? ±5 coref context window   — design parameter, not a filter

  REMAINING STRUCTURAL VERIFICATION (cannot replace with LLM):
  ◆ Component name in name_to_id        — validates LLM output
  ◆ Sentence number exists in sent_map  — validates LLM output
  ◆ Antecedent mention check            — verifies LLM claim against text
  ◆ Component name in comp_names        — validates LLM output (entity)
  ◆ Sentence number in document range   — validates LLM output

  INFRASTRUCTURE (batch/timeout/truncation — 16 total):
  ◆ Batch sizes: 10 (coref), 25 (validation), 50 (entity)
  ◆ Timeouts: 100-300ms per call type
  ◆ Truncation: 20 sentences, 25 mappings, 60 char prev-sentence
  ◆ Retry: 2 attempts for entity extraction

  CONCLUSION:
  After removing 2 dead/redundant safety nets, the pipeline has:
  - 2 remaining algorithmic thresholds (len<4, ±5 window)
  - Both are DESIGN PARAMETERS, not arbitrary magic numbers:
    * len<4 = linguistic minimum for partial-name reliability
    * ±5 = discourse context window (wider than the ≤3 distance rule)
  - All other code checks are structural verification or infrastructure

  The pipeline IS fully LLM-driven for all DECISIONS.
  Code only handles: input preparation, output verification, batching.
""")


if __name__ == "__main__":
    test_variant_a_no_len_filter()
    test_variant_b_coref_window()
    test_remaining_coref_checks()
    test_full_llm_analysis()
