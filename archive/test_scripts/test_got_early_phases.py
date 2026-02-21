#!/usr/bin/env python3
"""Test GoT union for Phase 1 and Phase 3A to stabilize LLM variance.

Phase 1: 3 different classification prompts, union ambiguous sets
Phase 3A: 3 different discovery prompts, union synonym/partial sets

Tests on mediastore (where Phase 1 just failed with 0 results).
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
os.environ.setdefault("CLAUDE_MODEL", "sonnet")

from llm_sad_sam.llm_client import LLMClient, LLMBackend
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.core import DocumentLoader

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)

DATASETS = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
    },
}


def test_phase1_single(llm, names):
    """Current Phase 1: single call."""
    prompt = f"""Classify these component names into two categories.

NAMES: {', '.join(names)}

RULES:
- "architectural": Names that are INVENTED proper names — they would not appear in a dictionary.
  Multi-word compounds, CamelCase names, and domain-specific terms are architectural.
- "ambiguous": Names that are ALSO common English words you would find in a dictionary.
  A single word that has everyday meaning beyond this system belongs here.
  Example: In a compiler, "Optimizer" could be the component name OR the general concept of optimization.
  In an OS kernel, "Scheduler" is architectural, but "Monitor" could be the component or the generic concept.

Return JSON:
{{
  "architectural": ["names that are clearly invented/technical"],
  "ambiguous": ["names that double as ordinary English words"]
}}
JSON only:"""

    data = llm.extract_json(llm.query(prompt, timeout=100))
    if data:
        return (set(data.get("architectural", [])) & set(names),
                set(data.get("ambiguous", [])) & set(names))
    return set(), set()


def test_phase1_got_union(llm, names):
    """GoT Phase 1: 3 different prompts, union ambiguous sets."""

    # Branch 1: Original prompt (dictionary test)
    prompt1 = f"""Classify these component names into two categories.

NAMES: {', '.join(names)}

RULES:
- "architectural": Names that are INVENTED proper names — they would not appear in a dictionary.
  Multi-word compounds, CamelCase names, and domain-specific terms are architectural.
- "ambiguous": Names that are ALSO common English words you would find in a dictionary.
  A single word that has everyday meaning beyond this system belongs here.

Return JSON:
{{
  "architectural": ["names that are clearly invented/technical"],
  "ambiguous": ["names that double as ordinary English words"]
}}
JSON only:"""

    # Branch 2: Sentence completion test
    prompt2 = f"""For each component name, determine if it could be misunderstood as a common English word.

NAMES: {', '.join(names)}

TEST: For each name, imagine reading it in a sentence WITHOUT knowing about this software system.
Would a reader think it refers to:
(a) A specific technical component (→ "safe") — names like "UserDBAdapter" or "MediaAccess" are unambiguous
(b) A general English concept (→ "risky") — names like "Cache", "Facade", "Logic" could mean the
    component OR the general concept

Return JSON:
{{
  "safe": ["clearly technical names"],
  "risky": ["names that double as English words"]
}}
JSON only:"""

    # Branch 3: Substitution test
    prompt3 = f"""Determine which component names are single common English words.

NAMES: {', '.join(names)}

For each name, check:
1. Is it a SINGLE word (no spaces, no CamelCase internal capitals)?
2. Would you find this exact word in an English dictionary as a common noun, verb, or adjective?
3. Could someone use this word in everyday speech without referring to software?

"compound_or_technical": Names that are multi-word, CamelCase, abbreviations, or domain-specific
"common_english": Names that are single common English words with everyday meanings

Return JSON:
{{
  "compound_or_technical": ["multi-word or clearly technical names"],
  "common_english": ["single common English words"]
}}
JSON only:"""

    print("    Branch 1 (dictionary test)...")
    data1 = llm.extract_json(llm.query(prompt1, timeout=100))
    print("    Branch 2 (sentence completion)...")
    data2 = llm.extract_json(llm.query(prompt2, timeout=100))
    print("    Branch 3 (substitution test)...")
    data3 = llm.extract_json(llm.query(prompt3, timeout=100))

    # Union ambiguous sets from all 3 branches
    ambig1 = set(data1.get("ambiguous", [])) & set(names) if data1 else set()
    ambig2 = set(data2.get("risky", [])) & set(names) if data2 else set()
    ambig3 = set(data3.get("common_english", [])) & set(names) if data3 else set()

    arch1 = set(data1.get("architectural", [])) & set(names) if data1 else set()
    arch2 = set(data2.get("safe", [])) & set(names) if data2 else set()
    arch3 = set(data3.get("compound_or_technical", [])) & set(names) if data3 else set()

    # Union of ambiguous (any branch flags it → ambiguous)
    ambig_union = ambig1 | ambig2 | ambig3
    # Architectural = all names not flagged as ambiguous
    arch_union = set(names) - ambig_union

    print(f"    Branch 1 ambiguous: {sorted(ambig1)}")
    print(f"    Branch 2 risky:     {sorted(ambig2)}")
    print(f"    Branch 3 common:    {sorted(ambig3)}")
    print(f"    UNION ambiguous:    {sorted(ambig_union)}")

    return arch_union, ambig_union


def test_phase3a_single(llm, comp_names, doc_lines):
    """Current Phase 3A: single discovery call."""
    prompt = f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

WHAT TO FIND:
1. ABBREVIATIONS: Short forms explicitly introduced in the document.
   Rule: The abbreviation must be defined in the text, e.g., "Full Name (FN)" introduces FN.

2. SYNONYMS: Alternative names that SPECIFICALLY refer to one component.
   Rule: The alternative name must unambiguously identify exactly ONE component.
   APPROVE: A proper name, role title, or technical alias used interchangeably with the component
   REJECT: A generic description that could apply to anything

3. PARTIAL REFERENCES: A shorter form of a multi-word component name used alone.
   Rule: A trailing word from a multi-word name that, in this document, consistently means the full name.
   APPROVE: Only if the short form is unambiguous
   REJECT: Common words that have ordinary English meanings beyond the component

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": {{"short_form": "FullComponent"}},
  "synonyms": {{"specific_alternative_name": "FullComponent"}},
  "partial_references": {{"partial_name": "FullComponent"}}
}}
JSON only:"""

    data = llm.extract_json(llm.query(prompt, timeout=150))
    return data


def test_phase3a_got_union(llm, comp_names, doc_lines):
    """GoT Phase 3A: 3 different discovery prompts, union findings."""

    doc_block = chr(10).join(doc_lines)

    # Branch 1: Original prompt (find all alternative names)
    prompt1 = f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

WHAT TO FIND:
1. ABBREVIATIONS: Short forms explicitly introduced (e.g., "Full Name (FN)" introduces FN)
2. SYNONYMS: Alternative names that specifically refer to one component
3. PARTIAL REFERENCES: Shorter form of a multi-word name used alone

DOCUMENT:
{doc_block}

Return JSON:
{{
  "abbreviations": {{"short_form": "FullComponent"}},
  "synonyms": {{"specific_alternative_name": "FullComponent"}},
  "partial_references": {{"partial_name": "FullComponent"}}
}}
JSON only:"""

    # Branch 2: Role-based discovery (find names used in ROLE descriptions)
    prompt2 = f"""Find how these components are REFERRED TO in the document — what names, aliases,
or short forms does the author use when discussing each component?

COMPONENTS: {', '.join(comp_names)}

For each component, scan the document for:
- Any name used INTERCHANGEABLY with the component name (synonyms)
- Any SHORT FORM of the component name (abbreviations or partials)
- Any ALTERNATIVE SPELLING or format (e.g., space-separated vs CamelCase)

DOCUMENT:
{doc_block}

Return JSON:
{{
  "abbreviations": {{"short_form": "FullComponent"}},
  "synonyms": {{"alternative_name": "FullComponent"}},
  "partial_references": {{"short_form": "FullComponent"}}
}}
JSON only:"""

    # Branch 3: Sentence-scanning discovery (scan each sentence for component mentions)
    prompt3 = f"""Scan each sentence and identify which component is being discussed,
even when the exact component name is not used.

COMPONENTS: {', '.join(comp_names)}

For each sentence, if a component is referenced by a different name than its official name,
record that alternative name. Look for:
- Abbreviations or acronyms
- Synonymous terms or role descriptions used as names
- Shortened forms of multi-word component names

DOCUMENT:
{doc_block}

Return JSON:
{{
  "abbreviations": {{"short_form": "FullComponent"}},
  "synonyms": {{"alternative_name": "FullComponent"}},
  "partial_references": {{"short_form": "FullComponent"}}
}}
JSON only:"""

    print("    Branch 1 (standard discovery)...")
    data1 = llm.extract_json(llm.query(prompt1, timeout=150))
    print("    Branch 2 (role-based discovery)...")
    data2 = llm.extract_json(llm.query(prompt2, timeout=150))
    print("    Branch 3 (sentence-scanning)...")
    data3 = llm.extract_json(llm.query(prompt3, timeout=150))

    # Union all findings
    all_abbrev = {}
    all_syn = {}
    all_partial = {}

    for data, label in [(data1, "B1"), (data2, "B2"), (data3, "B3")]:
        if not data:
            print(f"    {label}: empty response")
            continue
        abbr = {k: v for k, v in data.get("abbreviations", {}).items() if v in comp_names}
        syn = {k: v for k, v in data.get("synonyms", {}).items() if v in comp_names}
        part = {k: v for k, v in data.get("partial_references", {}).items() if v in comp_names}
        print(f"    {label}: {len(abbr)} abbrev, {len(syn)} syn, {len(part)} partial")
        for k, v in abbr.items():
            if k not in all_abbrev:
                print(f"      NEW abbrev: {k} -> {v}")
            all_abbrev.setdefault(k, v)
        for k, v in syn.items():
            if k not in all_syn:
                print(f"      NEW syn: {k} -> {v}")
            all_syn.setdefault(k, v)
        for k, v in part.items():
            if k not in all_partial:
                print(f"      NEW partial: {k} -> {v}")
            all_partial.setdefault(k, v)

    print(f"    UNION: {len(all_abbrev)} abbrev, {len(all_syn)} syn, {len(all_partial)} partial")
    return {"abbreviations": all_abbrev, "synonyms": all_syn, "partial_references": all_partial}


def main():
    ds_name = sys.argv[1] if len(sys.argv) > 1 else "mediastore"
    if ds_name not in DATASETS:
        print(f"Unknown dataset: {ds_name}. Choose from: {list(DATASETS.keys())}")
        return

    paths = DATASETS[ds_name]
    components = parse_pcm_repository(str(paths["model"]))
    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    names = [c.name for c in components]
    comp_names = sorted(set(c.name for c in components))

    llm = LLMClient(backend=LLMBackend.CLAUDE)
    print(f"Dataset: {ds_name} ({len(components)} components, {len(sentences)} sentences)")
    print(f"Components: {', '.join(names)}")

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Ambiguity Classification
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 1: Ambiguity Classification")
    print("=" * 70)

    print("\n  --- Single call (current) ---")
    arch_s, ambig_s = test_phase1_single(llm, names)
    print(f"  Architectural: {sorted(arch_s)}")
    print(f"  Ambiguous:     {sorted(ambig_s)}")

    print("\n  --- GoT Union (3 calls) ---")
    arch_g, ambig_g = test_phase1_got_union(llm, names)
    print(f"  Architectural: {sorted(arch_g)}")
    print(f"  Ambiguous:     {sorted(ambig_g)}")

    print("\n  --- Comparison ---")
    print(f"  Single:    {len(ambig_s)} ambiguous: {sorted(ambig_s)}")
    print(f"  GoT union: {len(ambig_g)} ambiguous: {sorted(ambig_g)}")
    print(f"  Only in GoT: {sorted(ambig_g - ambig_s)}")
    print(f"  Only in single: {sorted(ambig_s - ambig_g)}")

    # ═══════════════════════════════════════════════════════════════
    # Phase 3A: Synonym/Alias Discovery
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 3A: Synonym/Alias Discovery")
    print("=" * 70)

    doc_lines = [f"S{s.number}: {s.text}" for s in sentences[:100]]

    print("\n  --- Single call (current) ---")
    data_s = test_phase3a_single(llm, comp_names, doc_lines)
    if data_s:
        abbr_s = {k: v for k, v in data_s.get("abbreviations", {}).items() if v in comp_names}
        syn_s = {k: v for k, v in data_s.get("synonyms", {}).items() if v in comp_names}
        part_s = {k: v for k, v in data_s.get("partial_references", {}).items() if v in comp_names}
    else:
        abbr_s, syn_s, part_s = {}, {}, {}
    print(f"  Abbreviations: {abbr_s}")
    print(f"  Synonyms:      {syn_s}")
    print(f"  Partials:       {part_s}")

    print("\n  --- GoT Union (3 calls) ---")
    data_g = test_phase3a_got_union(llm, comp_names, doc_lines)
    abbr_g = data_g["abbreviations"]
    syn_g = data_g["synonyms"]
    part_g = data_g["partial_references"]
    print(f"  Abbreviations: {abbr_g}")
    print(f"  Synonyms:      {syn_g}")
    print(f"  Partials:       {part_g}")

    print("\n  --- Comparison ---")
    all_single = set(abbr_s.keys()) | set(syn_s.keys()) | set(part_s.keys())
    all_got = set(abbr_g.keys()) | set(syn_g.keys()) | set(part_g.keys())
    print(f"  Single total: {len(all_single)} mappings")
    print(f"  GoT total:    {len(all_got)} mappings")
    print(f"  Only in GoT:  {sorted(all_got - all_single)}")
    print(f"  Only in single: {sorted(all_single - all_got)}")

    # ═══════════════════════════════════════════════════════════════
    # Stability test: run single call 3 more times
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STABILITY: Run Phase 1 single call 3 more times")
    print("=" * 70)
    for run in range(3):
        a, b = test_phase1_single(llm, names)
        print(f"  Run {run+1}: arch={len(a)}, ambig={len(b)}: {sorted(b)}")


if __name__ == "__main__":
    main()
