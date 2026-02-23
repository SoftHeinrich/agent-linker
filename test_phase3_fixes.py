#!/usr/bin/env python3
"""Phase 3 unit test: Do s7's prompts make the code fixes unnecessary?

Runs ONLY Phase 3 extraction + judge (s7 prompts), then reports:
  1. What the LLM approved/rejected BEFORE any code fix
  2. Which fixes would fire
  3. Whether critical mappings survive without fixes

If the LLM consistently approves DataStorage/Database/Datastore/etc.,
the code fixes are dead code and can be removed.
"""

import os
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))
os.environ["CLAUDE_MODEL"] = "sonnet"

from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.core import DocumentLoader
from llm_sad_sam.core.data_types import DocumentKnowledge
from llm_sad_sam.llm_client import LLMClient, LLMBackend

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)

DATASETS = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
        "critical": {"DataStorage": "FileStorage", "AudioAccess": "MediaAccess",
                     "Database": "DB", "ReEncoder": "Reencoding"},
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
        "model": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
        "critical": {"PersistenceProvider": "Persistence", "Image Provider": "ImageProvider"},
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
        "critical": {"Datastore": "GAE Datastore"},
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
        "critical": {"bbb-html5": "HTML5 Server", "bbb-web": "BBB web",
                     "akka-apps": "Apps", "BigBlueButton Apps": "Apps"},
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
        "critical": {},
    },
}

# s7's extraction prompt (with fixed examples)
EXTRACTION_PROMPT = """Find all alternative names used for these components in the document.

COMPONENTS: {comp_list}

WHAT TO FIND:
1. ABBREVIATIONS: Short forms explicitly introduced in the document.
   Rule: The abbreviation must be FORMALLY DEFINED in the text with a parenthetical pattern,
   e.g., "Full Name (FN)" introduces FN, or "FN (Full Name)" introduces FN.
   Only propose abbreviations you can point to a specific definitional sentence for.
   Do NOT propose abbreviations that merely "seem likely" — require explicit textual evidence.

2. SYNONYMS: Alternative names that SPECIFICALLY refer to one component.
   Rule: The alternative name must unambiguously identify exactly ONE component.
   APPROVE: A proper name, role title, or technical alias used interchangeably with the component
   REJECT: A generic description that could apply to anything (like "the system" or "the process")

3. PARTIAL REFERENCES: A shorter form of a multi-word component name used alone.
   Rule: A trailing word from a multi-word name that, in this document, consistently means the full name.
   APPROVE: Only if the short form is DISTINCTIVE — it would not be confused with any other concept
   REJECT: Common abbreviations or words that have well-known meanings beyond this specific component
   REJECT: Any partial that is also a widely-used acronym or abbreviation in computing
   (e.g., if a component is "MemoryManager", do not propose "MM" as a partial — "MM" is too
   generic to be a distinctive reference to that specific component)

DOCUMENT:
{doc_text}

Return JSON:
{{
  "abbreviations": {{"short_form": "FullComponent"}},
  "synonyms": {{"specific_alternative_name": "FullComponent"}},
  "partial_references": {{"partial_name": "FullComponent"}}
}}
JSON only:"""

JUDGE_PROMPT = """JUDGE: Review these component name mappings for correctness.

COMPONENTS: {comp_list}

PROPOSED MAPPINGS:
{mapping_list}

Apply these rules:

REJECT if ANY of these are true:
- The term is used in its ordinary English sense, NOT as a name for the component
  (e.g., "the scheduler runs every minute" uses "scheduler" as a generic concept, not as a named component)
- The term refers to a different component or to the system as a whole
- The mapping cannot be verified from the actual document text
- The term is a widely-used computing abbreviation or acronym (like API, OS, IO, VM, CPU)
  that is NOT formally defined in this document as referring to the specific component.
  A formal definition requires explicit text like "ComponentName (Abbrev)" or equivalent.

APPROVE if ALL of these are true:
- The term is used AS A NAME for the component in context (even if the word exists in a dictionary)
  (e.g., "The Dispatcher routes incoming requests" uses "Dispatcher" as a proper name for a specific component)
- The term appears in the document in a context that makes the reference clear
- The term unambiguously identifies exactly one component

Return JSON:
{{
  "approved": ["term1", "term2"],
  "generic_rejected": ["generic_term1"]
}}
JSON only:"""


def run_phase3_raw(dataset_name):
    """Run Phase 3 prompts and return raw LLM decisions before any fix."""
    paths = DATASETS[dataset_name]
    components = parse_pcm_repository(str(paths["model"]))
    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    comp_names = [c.name for c in components]
    doc_lines = [s.text for s in sentences[:150]]

    llm = LLMClient(backend=LLMBackend.CLAUDE)

    # Step 1: Extraction
    prompt1 = EXTRACTION_PROMPT.format(
        comp_list=', '.join(comp_names),
        doc_text=chr(10).join(doc_lines),
    )
    data1 = llm.extract_json(llm.query(prompt1, timeout=150))

    all_mappings = {}
    if data1:
        for short, full in data1.get("abbreviations", {}).items():
            if full in comp_names:
                all_mappings[short] = ("abbrev", full)
        for syn, full in data1.get("synonyms", {}).items():
            if full in comp_names:
                all_mappings[syn] = ("synonym", full)
        for partial, full in data1.get("partial_references", {}).items():
            if full in comp_names:
                all_mappings[partial] = ("partial", full)

    print(f"  Extraction proposed {len(all_mappings)} mappings:")
    for term, (typ, comp) in sorted(all_mappings.items()):
        print(f"    {term} -> {comp} ({typ})")

    if not all_mappings:
        return all_mappings, set(), set(), sentences, comp_names

    # Step 2: Judge
    mapping_list = [f"'{k}' -> {v[1]} ({v[0]})" for k, v in list(all_mappings.items())[:25]]
    prompt2 = JUDGE_PROMPT.format(
        comp_list=', '.join(comp_names),
        mapping_list=chr(10).join(mapping_list),
    )
    data2 = llm.extract_json(llm.query(prompt2, timeout=120))
    approved = set(data2.get("approved", [])) if data2 else set(all_mappings.keys())
    rejected = set(data2.get("generic_rejected", [])) if data2 else set()

    print(f"\n  Judge approved ({len(approved)}): {sorted(approved)}")
    print(f"  Judge rejected ({len(rejected)}): {sorted(rejected)}")

    return all_mappings, approved, rejected, sentences, comp_names


def check_fixes(all_mappings, approved, rejected, sentences, comp_names):
    """Check which fixes would fire and what they'd rescue."""
    results = {}

    # Fix A: CamelCase
    fix_a = [t for t in rejected if re.search(r'[a-z][A-Z]', t)]
    results["Fix A (CamelCase)"] = fix_a

    # Fix B: uppercase <=4
    fix_b = [t for t in rejected if t.isupper() and len(t) <= 4 and t in all_mappings]
    results["Fix B (uppercase)"] = fix_b

    # V24: exact-match or multi-word
    v24 = []
    for t in rejected:
        if t not in all_mappings:
            continue
        if any(t.lower() == cn.lower() for cn in comp_names):
            v24.append(f"{t} (exact)")
        elif t[0].isupper() and ' ' in t:
            v24.append(f"{t} (multi-word)")
    results["V24 (structural)"] = v24

    # Fix C: capitalized mid-sentence
    fix_c = []
    for term in rejected:
        if term not in all_mappings:
            continue
        _, target_comp = all_mappings[term]
        if ' ' in term or not term[0].isupper() or target_comp not in comp_names:
            continue
        for s in sentences[:100]:
            for m in re.finditer(rf'\b{re.escape(term)}\b', s.text):
                if m.start() > 0:
                    fix_c.append(term)
                    break
            if term in fix_c:
                break
    results["Fix C (capitalized)"] = fix_c

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                       default=["mediastore", "teastore", "teammates"],
                       choices=list(DATASETS.keys()))
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 3 FIX NECESSITY TEST")
    print("Which code fixes still fire with s7's enhanced prompts?")
    print("=" * 70)

    for ds in args.datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {ds}")
        print(f"{'='*70}")

        t0 = time.time()
        all_mappings, approved, rejected, sentences, comp_names = run_phase3_raw(ds)
        elapsed = time.time() - t0

        if not all_mappings:
            print(f"  No mappings proposed ({elapsed:.0f}s)")
            continue

        # Check fixes
        print(f"\n  --- Fix analysis ---")
        fixes = check_fixes(all_mappings, approved, rejected, sentences, comp_names)
        any_fires = False
        for name, rescued in fixes.items():
            if rescued:
                print(f"  {name}: WOULD FIRE -> {rescued}")
                any_fires = True
            else:
                print(f"  {name}: not needed")

        if not any_fires:
            print(f"  >>> ALL FIXES ARE DEAD CODE for this run")

        # Check critical mappings
        critical = DATASETS[ds].get("critical", {})
        if critical:
            print(f"\n  --- Critical mapping check (prompt-only, no fixes) ---")
            for term, target in critical.items():
                if term in approved:
                    print(f"  OK: {term} -> {target} (LLM approved directly)")
                elif term in rejected:
                    print(f"  FAIL: {term} (LLM rejected, needs fix)")
                elif term in all_mappings:
                    print(f"  WARN: {term} (proposed but not in approved/rejected lists)")
                else:
                    print(f"  MISS: {term} (not even proposed by extraction)")

        print(f"\n  ({elapsed:.0f}s)")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print("If all critical terms are 'OK' across multiple runs,")
    print("the code fixes can be safely removed.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
