#!/usr/bin/env python3
"""Test S-Linker10 fixes from checkpoint.

Test 1: PRONOUN_PATTERN fix (removed "the component|the service")
  - Replays coref from tier1 checkpoint
  - Compares pronoun sentence count: old pattern vs new pattern
  - Runs coref with new pattern, compares results

Test 2: Enrichment variant (LLM word usage replacing count>=3)
  - Replays tier1.5 from tier1 checkpoint
  - Uses WORD_USAGE_PROMPT instead of count threshold
  - Compares discovered partials

Both use s_linker9d checkpoints (same pipeline state, before s_linker10 changes).
"""
import csv, os, sys, pickle, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

os.environ.setdefault("CLAUDE_MODEL", "sonnet")

from llm_sad_sam.core.data_types import SadSamLink
from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend
from llm_sad_sam.linkers.experimental.prompts import WORD_USAGE_PROMPT

BENCH = Path("/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark")
CACHE = Path("results/phase_cache/s_linker9d")

DATASETS = {
    "mediastore":    {"text": "text_2016/mediastore.txt",    "model": "model_2016/pcm/ms.repository",       "gold": "goldstandards/goldstandard_sad_2016-sam_2016.csv"},
    "teastore":      {"text": "text_2020/teastore.txt",      "model": "model_2020/pcm/teastore.repository",  "gold": "goldstandards/goldstandard_sad_2020-sam_2020.csv"},
    "teammates":     {"text": "text_2021/teammates.txt",     "model": "model_2021/pcm/teammates.repository", "gold": "goldstandards/goldstandard_sad_2021-sam_2021.csv"},
    "bigbluebutton": {"text": "text_2021/bigbluebutton.txt", "model": "model_2021/pcm/bbb.repository",       "gold": "goldstandards/goldstandard_sad_2021-sam_2021.csv"},
    "jabref":        {"text": "text_2021/jabref.txt",        "model": "model_2021/pcm/jabref.repository",    "gold": "goldstandards/goldstandard_sad_2021-sam_2021.csv"},
}

OLD_PATTERN = re.compile(
    r'\b(it|they|this|these|that|those|its|their|the component|the service)\b',
    re.IGNORECASE
)
NEW_PATTERN = re.compile(
    r'\b(it|they|this|these|that|those|its|their)\b',
    re.IGNORECASE
)

loader = DocumentLoader()

# ══════════════════════════════════════════════════════════════════════════
# TEST 1: Pronoun pattern impact on coref sentence selection
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: PRONOUN_PATTERN fix — impact analysis")
print("=" * 70)

for ds, paths in DATASETS.items():
    sentences = loader.load_sentences(str(BENCH / ds / paths["text"]))

    old_sents = [s for s in sentences if OLD_PATTERN.search(s.text)]
    new_sents = [s for s in sentences if NEW_PATTERN.search(s.text)]

    lost = set(s.number for s in old_sents) - set(s.number for s in new_sents)
    lost_texts = []
    if lost:
        for s in sentences:
            if s.number in lost:
                # Check which removed pattern matched
                matches = re.findall(r'\b(the component|the service)\b', s.text, re.IGNORECASE)
                lost_texts.append((s.number, matches, s.text[:80]))

    print(f"\n  {ds}: old={len(old_sents)} new={len(new_sents)} lost={len(lost)}")
    for sn, matches, text in lost_texts:
        print(f"    S{sn}: matched {matches} — \"{text}...\"")

    # Load checkpoint coref results to check if any lost sentences produced TPs
    with open(CACHE / ds / "tier2.pkl", "rb") as f:
        tier2 = pickle.load(f)
    coref_links = tier2["coref_links"]
    coref_snums = {l.sentence_number for l in coref_links}
    lost_coref = lost & coref_snums
    if lost_coref:
        print(f"    *** COREF IMPACT: {len(lost_coref)} coref links from lost sentences: {lost_coref}")
    else:
        print(f"    No coref impact (no lost sentences had coref links)")

# ══════════════════════════════════════════════════════════════════════════
# TEST 2: Enrichment variant — LLM word usage vs count>=3
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: Enrichment — LLM word usage vs count>=3 threshold")
print("=" * 70)

llm = LLMClient(backend=LLMBackend.CLAUDE)

for ds, paths in DATASETS.items():
    print(f"\n  {ds}:")

    sentences = loader.load_sentences(str(BENCH / ds / paths["text"]))
    components = parse_pcm_repository(str(BENCH / ds / paths["model"]))
    sent_map = {s.number: s for s in sentences}

    with open(CACHE / ds / "tier1.pkl", "rb") as f:
        tier1 = pickle.load(f)
    with open(CACHE / ds / "tier1_5.pkl", "rb") as f:
        tier1_5 = pickle.load(f)

    dk = tier1_5.get("doc_knowledge") or tier1.get("doc_knowledge")
    mk = tier1.get("model_knowledge")
    generic_partials = tier1.get("generic_partials", set())

    # Current count-based partials (from tier1_5 checkpoint)
    count_partials = dk.partial_references.copy()
    # Remove partials that came from Phase 3 (doc knowledge), keep only enrichment-added ones
    dk_orig = tier1.get("doc_knowledge")
    phase3_partials = dk_orig.partial_references if dk_orig else {}
    enrichment_partials = {k: v for k, v in count_partials.items() if k not in phase3_partials}

    print(f"    Phase 3 partials: {phase3_partials}")
    print(f"    Enrichment (count>=3): {enrichment_partials}")

    # Find multi-word components with potential trailing-word partials
    multiword = [c for c in components if ' ' in c.name]
    if not multiword:
        print(f"    No multi-word components — skipping LLM test")
        continue

    # For each multi-word component, check trailing word via LLM
    llm_partials = {}
    for comp in multiword:
        words = comp.name.split()
        last_word = words[-1]
        if len(last_word) < 4:
            continue
        # Check if another component shares this trailing word
        shared = any(
            c.name != comp.name and c.name.split()[-1].lower() == last_word.lower()
            for c in components if ' ' in c.name
        )
        if shared:
            continue
        # Already known from Phase 3?
        if last_word in phase3_partials:
            continue

        is_generic = last_word.lower() in {g.lower() for g in generic_partials}

        # Find sentences where trailing word appears without full name
        relevant_sents = []
        for sent in sentences:
            has_full = bool(re.search(rf'\b{re.escape(comp.name)}\b', sent.text, re.IGNORECASE))
            if is_generic:
                cap = last_word[0].upper() + last_word[1:]
                has_partial = bool(re.search(rf'\b{re.escape(cap)}\b', sent.text))
            else:
                has_partial = bool(re.search(rf'\b{re.escape(last_word)}\b', sent.text, re.IGNORECASE))
            if has_partial and not has_full:
                relevant_sents.append(sent)

        if not relevant_sents:
            continue

        # LLM word usage classification
        calibration = ""
        if is_generic:
            calibration = (f'NOTE: "{last_word}" is also an ordinary English word. '
                          f'Be careful to distinguish entity references from generic usage.\n\n')

        sent_block = "\n".join(f"  S{s.number}: {s.text}" for s in relevant_sents[:20])

        prompt = WORD_USAGE_PROMPT.format(
            partial=last_word,
            partial_lower=last_word.lower(),
            comp_name=comp.name,
            calibration=calibration,
            sent_block=sent_block,
        )

        data = llm.extract_json(llm.query(prompt, timeout=120))
        classification = data.get("classification", "ordinary") if data else "ordinary"
        reason = data.get("reason", "") if data else ""

        count = len(relevant_sents)
        count_decision = "YES" if count >= 3 else "NO"
        llm_decision = "YES" if classification == "name" else "NO"

        tag = ""
        if count_decision != llm_decision:
            tag = " *** DISAGREEMENT"

        print(f"    {last_word} -> {comp.name}: count={count} ({count_decision}) | "
              f"LLM={classification} ({llm_decision}){tag}")
        if reason:
            print(f"      Reason: {reason}")

        if classification == "name":
            llm_partials[last_word] = comp.name

    print(f"    LLM partials: {llm_partials}")
    if enrichment_partials != llm_partials:
        added = {k: v for k, v in llm_partials.items() if k not in enrichment_partials}
        removed = {k: v for k, v in enrichment_partials.items() if k not in llm_partials}
        if added:
            print(f"    LLM added: {added}")
        if removed:
            print(f"    LLM removed: {removed}")
    else:
        print(f"    Identical results")
