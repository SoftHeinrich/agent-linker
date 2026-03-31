#!/usr/bin/env python3
"""Deep audit of prompts_v2.py: overlap with benchmark data, few-shot coverage,
and whether providing ambiguity results to extraction would reduce FPs.

Checks:
  1. Term overlap: prompt example terms vs actual benchmark component names
  2. Pattern overlap: prompt example patterns vs actual doc sentence patterns
  3. Abbreviation coverage: are benchmark abbreviation patterns taught by examples?
  4. Ambiguity info value: would passing ambiguity to extraction catch more FPs?
  5. Few-shot calibration: do judge examples match real approval/rejection patterns?
  6. Alias coverage: do extraction rules cover actual alias types in checkpoints?
"""

import csv
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.linkers.experimental.prompts_v2 import (
    AMBIGUITY_FEW_SHOT, AMBIGUITY_RULES,
    DOC_KNOWLEDGE_EXTRACTION_RULES, DOC_KNOWLEDGE_JUDGE_EXAMPLES,
    DOC_KNOWLEDGE_JUDGE_RULES,
    ENTITY_EXTRACTION_RULES, VALIDATION_RULES, COREF_RULES,
    WORD_USAGE_PROMPT,
)

BENCHMARK = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
CACHE = Path("results/phase_cache/s_linker11")

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


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_component_names():
    """Load all component names across all 5 benchmark datasets."""
    all_names = {}  # dataset -> set of names
    for ds in DATASETS:
        model_dirs = list((BENCHMARK / ds).glob("model_*/pcm/*.repository"))
        if not model_dirs:
            continue
        from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
        comps = parse_pcm_repository(str(model_dirs[0]))
        all_names[ds] = {c.name for c in comps}
    return all_names


def load_all_sentences():
    """Load all sentences across all 5 benchmark datasets."""
    all_sents = {}  # dataset -> {snum: text}
    for ds, info in DATASETS.items():
        path = BENCHMARK / ds / info["text"]
        sents = {}
        with open(path) as f:
            for i, line in enumerate(f, 1):
                sents[i] = line.strip()
        all_sents[ds] = sents
    return all_sents


def load_gold_standards():
    """Load gold standards as {dataset: set((snum, comp_id))}."""
    golds = {}
    for ds, info in DATASETS.items():
        path = BENCHMARK / ds / info["gs"]
        links = set()
        with open(path) as f:
            for row in csv.DictReader(f):
                sent = int(row.get("sentence", row.get("sentenceNo", 0)))
                links.add((sent, row["modelElementID"]))
        golds[ds] = links
    return golds


def load_checkpoints():
    """Load S-Linker11 layer1+layer2 checkpoints for all datasets."""
    ckpts = {}
    for ds in DATASETS:
        l1_path = CACHE / ds / "layer1.pkl"
        l2_path = CACHE / ds / "layer2.pkl"
        if not l1_path.exists() or not l2_path.exists():
            continue
        l1 = pickle.load(open(l1_path, "rb"))
        l2 = pickle.load(open(l2_path, "rb"))
        ckpts[ds] = {"layer1": l1, "layer2": l2}
    return ckpts


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 1: Term overlap — prompt example terms vs benchmark component names
# ═══════════════════════════════════════════════════════════════════════════════

def extract_example_terms(text):
    """Extract all noun/name terms from few-shot example text."""
    # Find quoted strings
    quoted = set(re.findall(r'"([^"]+)"', text))
    # Find CamelCase words
    camel = set(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text))
    # Find all-caps words (3+ chars)
    allcaps = set(re.findall(r'\b[A-Z]{3,}\b', text))
    # Find capitalized single words in example lists
    in_lists = set(re.findall(r'"(\w+)"', text))
    return quoted | camel | allcaps | in_lists


def check_term_overlap(all_comp_names):
    """Check if any terms in prompts_v2.py examples overlap with benchmark names."""
    print("=" * 70)
    print("CHECK 1: TERM OVERLAP — prompt examples vs benchmark component names")
    print("=" * 70)

    # Flatten all benchmark component names
    bench_names = set()
    bench_names_lower = {}  # lower -> (dataset, original)
    for ds, names in all_comp_names.items():
        for n in names:
            bench_names.add(n)
            bench_names_lower[n.lower()] = (ds, n)

    print(f"\nBenchmark components: {len(bench_names)} unique names across 5 datasets")

    # Extract terms from each prompt constant
    prompts = {
        "AMBIGUITY_FEW_SHOT": AMBIGUITY_FEW_SHOT,
        "AMBIGUITY_RULES": AMBIGUITY_RULES,
        "DOC_KNOWLEDGE_EXTRACTION_RULES": DOC_KNOWLEDGE_EXTRACTION_RULES,
        "DOC_KNOWLEDGE_JUDGE_EXAMPLES": DOC_KNOWLEDGE_JUDGE_EXAMPLES,
        "DOC_KNOWLEDGE_JUDGE_RULES": DOC_KNOWLEDGE_JUDGE_RULES,
        "ENTITY_EXTRACTION_RULES": ENTITY_EXTRACTION_RULES,
        "VALIDATION_RULES": VALIDATION_RULES,
        "COREF_RULES": COREF_RULES,
        "WORD_USAGE_PROMPT": WORD_USAGE_PROMPT,
    }

    total_overlaps = 0
    for pname, ptext in prompts.items():
        terms = extract_example_terms(ptext)
        overlaps = []
        for t in terms:
            if t.lower() in bench_names_lower:
                ds, orig = bench_names_lower[t.lower()]
                overlaps.append((t, ds, orig))
            # Also check substring containment
            for bn_lower, (ds, orig) in bench_names_lower.items():
                if len(t) >= 4 and t.lower() != bn_lower:
                    if t.lower() in bn_lower or bn_lower in t.lower():
                        overlaps.append((f"{t} ⊂ {orig}", ds, orig))

        if overlaps:
            print(f"\n  {pname}: {len(overlaps)} overlap(s)!")
            for term, ds, orig in overlaps:
                print(f"    ⚠ '{term}' matches [{ds}] '{orig}'")
            total_overlaps += len(overlaps)
        else:
            print(f"\n  {pname}: CLEAN — 0 overlaps")

    print(f"\n  TOTAL OVERLAPS: {total_overlaps}")
    return total_overlaps


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 2: Abbreviation pattern coverage
# ═══════════════════════════════════════════════════════════════════════════════

def check_abbreviation_coverage(all_sents, all_comp_names, ckpts):
    """Check if prompt examples teach the actual abbreviation patterns found in benchmarks."""
    print("\n" + "=" * 70)
    print("CHECK 2: ABBREVIATION PATTERN COVERAGE")
    print("=" * 70)

    # What abbreviation patterns exist in the real data?
    print("\n  Actual abbreviation patterns found in documents:")

    for ds, sents in all_sents.items():
        comp_names = all_comp_names.get(ds, set())
        # Find parenthetical abbreviation introductions: "Full Name (ABBR)"
        paren_abbrs = []
        # Find multi-word followed by abbreviation in parens
        for snum, text in sents.items():
            matches = re.findall(r'([A-Z][\w\s]+?)\s*\(([A-Z]{2,})\)', text)
            for full, abbr in matches:
                paren_abbrs.append((snum, full.strip(), abbr))

        # Also find component names that are abbreviations themselves
        comp_abbrs = [n for n in comp_names if n.isupper() and len(n) >= 2]

        # Check checkpoint-discovered abbreviations
        dk = ckpts.get(ds, {}).get("layer2", {}).get("doc_knowledge")
        discovered_abbrs = dict(dk.abbreviations) if dk else {}

        print(f"\n  [{ds}]")
        print(f"    Component abbreviation-names: {comp_abbrs or 'none'}")
        print(f"    Parenthetical introductions in doc: {len(paren_abbrs)}")
        for snum, full, abbr in paren_abbrs[:5]:
            print(f"      S{snum}: '{full}' ({abbr})")
        print(f"    Discovered by Phase 3: {discovered_abbrs or 'none'}")

    # What does the prompt teach?
    print("\n  Prompt teaches:")
    print(f"    EXTRACTION: '{DOC_KNOWLEDGE_EXTRACTION_RULES.splitlines()[1].strip()}'")
    print(f"    JUDGE Ex1: 'AST' -> AbstractSyntaxTree (parenthetical pattern)")
    print(f"    RULES: auto-approve abbreviations from initials")

    # Gap analysis
    print("\n  Gap analysis:")
    print("    ✓ Parenthetical '(ABBR)' pattern: taught by extraction rule + judge Ex1")
    print("    ? Non-parenthetical abbreviations (e.g., 'KMS' for 'Kurento Media Server'):")
    print("      → These get classified as SYNONYMS instead. Check if that causes issues...")

    # Check: do non-parenthetical abbreviations survive as synonyms?
    for ds in ckpts:
        dk = ckpts[ds]["layer2"]["doc_knowledge"]
        for syn, target in dk.synonyms.items():
            if syn.isupper() and len(syn) >= 2:
                print(f"    [{ds}] '{syn}' → {target} classified as SYNONYM (would be abbrev if parenthetical)")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 3: Ambiguity info value for entity extraction
# ═══════════════════════════════════════════════════════════════════════════════

def check_ambiguity_value(all_comp_names, all_sents, golds, ckpts):
    """Empirically check: would providing ambiguity classification to entity
    extraction reduce FPs without killing TPs?

    Method: For each entity-source link in S-Linker11 results, check:
    - Is the component ambiguous?
    - Is the mention lowercase-only?
    - Is it a TP or FP?
    If ambiguous + lowercase FPs >> ambiguous + lowercase TPs, then yes.
    """
    print("\n" + "=" * 70)
    print("CHECK 3: WOULD AMBIGUITY INFO HELP ENTITY EXTRACTION?")
    print("=" * 70)

    # Load layer3 (entity validation results) and layer4
    for ds in ckpts:
        l1 = ckpts[ds]["layer1"]
        l2 = ckpts[ds]["layer2"]
        mk = l1["model_knowledge"]
        ambig = mk.ambiguous_names
        gold = golds[ds]
        sents = all_sents[ds]
        raw_seeds = l1["raw_seed_links"]
        comp_names = all_comp_names.get(ds, set())

        # Build name→id map
        model_dirs = list((BENCHMARK / ds).glob("model_*/pcm/*.repository"))
        from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
        comps = parse_pcm_repository(str(model_dirs[0]))
        name_to_id = {c.name: c.id for c in comps}
        id_to_name = {c.id: c.name for c in comps}

        # Categorize ALL seed links by ambiguity and case
        seed_ambig_lower_tp = []
        seed_ambig_lower_fp = []
        seed_ambig_upper_tp = []
        seed_ambig_upper_fp = []
        seed_noambig_tp = []
        seed_noambig_fp = []

        for sl in raw_seeds:
            is_tp = (sl.sentence_number, sl.component_id) in gold
            is_ambig = sl.component_name in ambig
            sent_text = sents.get(sl.sentence_number, "")

            # Check if mention is lowercase-only
            comp_lower = sl.component_name.lower()
            has_uppercase = bool(re.search(
                rf'\b{re.escape(sl.component_name)}\b', sent_text
            )) if sl.component_name[0].isupper() else False
            has_lowercase = bool(re.search(
                rf'\b{re.escape(comp_lower)}\b', sent_text
            ))
            lowercase_only = has_lowercase and not has_uppercase

            if is_ambig:
                if lowercase_only:
                    (seed_ambig_lower_tp if is_tp else seed_ambig_lower_fp).append(sl)
                else:
                    (seed_ambig_upper_tp if is_tp else seed_ambig_upper_fp).append(sl)
            else:
                (seed_noambig_tp if is_tp else seed_noambig_fp).append(sl)

        print(f"\n  [{ds}]  ambiguous={sorted(ambig)}")
        print(f"    Seeds: ambig+lower TP={len(seed_ambig_lower_tp)} FP={len(seed_ambig_lower_fp)}")
        print(f"    Seeds: ambig+upper TP={len(seed_ambig_upper_tp)} FP={len(seed_ambig_upper_fp)}")
        print(f"    Seeds: non-ambig   TP={len(seed_noambig_tp)} FP={len(seed_noambig_fp)}")

        # Show the ambig+lower cases — these are what ambiguity info would catch
        for sl in seed_ambig_lower_tp:
            sent_text = sents.get(sl.sentence_number, "")[:80]
            print(f"      ambig+lower TP: S{sl.sentence_number} {sl.component_name}: \"{sent_text}\"")
        for sl in seed_ambig_lower_fp:
            sent_text = sents.get(sl.sentence_number, "")[:80]
            print(f"      ambig+lower FP: S{sl.sentence_number} {sl.component_name}: \"{sent_text}\"")

    # Also check: sentences in gold standard that mention ambiguous components
    print("\n  Cross-dataset summary: ambig+lowercase links")
    total_tp = total_fp = 0
    for ds in ckpts:
        l1 = ckpts[ds]["layer1"]
        ambig = l1["model_knowledge"].ambiguous_names
        gold = golds[ds]
        sents = all_sents[ds]
        raw_seeds = l1["raw_seed_links"]

        for sl in raw_seeds:
            if sl.component_name not in ambig:
                continue
            sent_text = sents.get(sl.sentence_number, "")
            comp_lower = sl.component_name.lower()
            has_upper = bool(re.search(rf'\b{re.escape(sl.component_name)}\b', sent_text)) if sl.component_name[0].isupper() else False
            if not has_upper and re.search(rf'\b{re.escape(comp_lower)}\b', sent_text):
                is_tp = (sl.sentence_number, sl.component_id) in gold
                if is_tp:
                    total_tp += 1
                else:
                    total_fp += 1

    print(f"\n  VERDICT: Ambig+lowercase filter would catch {total_fp} FP and kill {total_tp} TP")
    if total_tp > 0:
        print(f"    Kill ratio: {total_tp}:{total_fp} TP:FP — {'UNSAFE' if total_tp > total_fp else 'marginal'}")
    else:
        print(f"    Kill ratio: 0:{total_fp} — SAFE" if total_fp > 0 else "    No impact — neither TP nor FP affected")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 4: Few-shot judge calibration against actual approvals/rejections
# ═══════════════════════════════════════════════════════════════════════════════

def check_judge_calibration(ckpts, all_comp_names):
    """Check if the doc-knowledge judge examples match the actual patterns
    of approved vs rejected mappings in checkpoint data."""
    print("\n" + "=" * 70)
    print("CHECK 4: JUDGE EXAMPLE CALIBRATION vs ACTUAL DISCOVERIES")
    print("=" * 70)

    # Categorize all discovered mappings
    categories = defaultdict(list)  # category -> [(ds, term, comp, approved)]

    for ds in ckpts:
        l2 = ckpts[ds]["layer2"]
        dk = l2["doc_knowledge"]

        for term, comp in dk.abbreviations.items():
            categories["abbreviation"].append((ds, term, comp))
        for term, comp in dk.synonyms.items():
            # Sub-categorize synonyms
            if re.match(r'^[A-Z][a-z]+(?:[A-Z][a-z]+)+$', term):
                categories["synonym_camelcase"].append((ds, term, comp))
            elif term.isupper():
                categories["synonym_allcaps"].append((ds, term, comp))
            elif ' ' in term:
                categories["synonym_multiword"].append((ds, term, comp))
            else:
                categories["synonym_singleword"].append((ds, term, comp))
        for term, comp in dk.partial_references.items():
            if len(term) <= 3:
                categories["partial_short"].append((ds, term, comp))
            else:
                categories["partial_word"].append((ds, term, comp))

    print("\n  Actual approved mapping categories:")
    for cat, items in sorted(categories.items()):
        print(f"\n    {cat} ({len(items)}):")
        for ds, term, comp in items:
            print(f"      [{ds}] '{term}' → {comp}")

    # Check which categories are covered by judge examples
    print("\n  Judge example coverage:")
    example_categories = {
        "abbreviation": "Ex1: AST → AbstractSyntaxTree",
        "partial_trailing_word": "Ex2: Dispatcher → EventDispatcher, Ex4: Table → SymbolTable",
        "synonym_camelcase": "Ex3: RenderEngine → GameRenderEngine",
        "reject_ordinary": "Ex5: process → OrderProcessor",
        "reject_system": "Ex6: system → PaymentSystem",
    }
    for cat, desc in example_categories.items():
        print(f"    ✓ {cat}: {desc}")

    # Identify gaps
    uncovered = set(categories.keys()) - {
        "abbreviation", "partial_word", "partial_short",
        "synonym_camelcase", "synonym_allcaps",
        "synonym_singleword", "synonym_multiword",
    }
    if uncovered:
        print(f"\n    UNCOVERED CATEGORIES: {uncovered}")

    # Check multi-word synonyms specifically
    mw = categories.get("synonym_multiword", [])
    if mw:
        print(f"\n  Multi-word synonyms ({len(mw)}) — no judge example covers this type:")
        for ds, term, comp in mw:
            print(f"    [{ds}] '{term}' → {comp}")
        print("    → These are descriptive phrases. Judge Ex1-4 only show single-word/CamelCase.")
        print("    → Risk: LLM might reject these without calibration example.")

    # Check single-word synonyms that look like ordinary English
    sw = categories.get("synonym_singleword", [])
    borderline = [(ds, t, c) for ds, t, c in sw if t[0].islower()]
    if borderline:
        print(f"\n  Lowercase single-word synonyms ({len(borderline)}) — potential confusion with Ex5 'process' reject:")
        for ds, term, comp in borderline:
            print(f"    [{ds}] '{term}' → {comp}")
        print("    → These look like Ex5 'process' (ordinary English) but are APPROVED.")
        print("    → The judge might REJECT these by analogy to Ex5. Check if rules override.")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 5: Entity extraction — alias coverage
# ═══════════════════════════════════════════════════════════════════════════════

def check_extraction_alias_coverage(ckpts, all_sents, golds):
    """Check: how many gold-standard links depend on alias-type matching
    that the entity extraction rules need to cover?"""
    print("\n" + "=" * 70)
    print("CHECK 5: GOLD STANDARD LINK TYPES — what matching is required?")
    print("=" * 70)

    for ds in ckpts:
        l2 = ckpts[ds]["layer2"]
        dk = l2["doc_knowledge"]
        gold = golds[ds]
        sents = all_sents[ds]
        comp_names = set()
        comp_lower = set()

        model_dirs = list((BENCHMARK / ds).glob("model_*/pcm/*.repository"))
        from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
        comps = parse_pcm_repository(str(model_dirs[0]))
        name_to_id = {c.name: c.id for c in comps}
        id_to_name = {c.id: c.name for c in comps}

        # Categorize each gold link by how the component name appears
        match_types = Counter()
        examples_by_type = defaultdict(list)

        for snum, cid in gold:
            cname = id_to_name.get(cid, "?")
            text = sents.get(snum, "")
            if not text:
                match_types["missing_sentence"] += 1
                continue

            # Check exact proper-case match
            if re.search(rf'\b{re.escape(cname)}\b', text):
                match_types["exact_propercase"] += 1
                continue

            # Check case-insensitive exact match
            if re.search(rf'\b{re.escape(cname)}\b', text, re.IGNORECASE):
                match_types["exact_lowercase"] += 1
                examples_by_type["exact_lowercase"].append(
                    (ds, snum, cname, text[:80]))
                continue

            # Check space-separated CamelCase
            parts = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', cname)
            spaced = ' '.join(parts)
            if len(parts) > 1 and re.search(rf'\b{re.escape(spaced)}\b', text, re.IGNORECASE):
                match_types["camelcase_split"] += 1
                examples_by_type["camelcase_split"].append(
                    (ds, snum, cname, text[:80]))
                continue

            # Check abbreviation match
            found_abbr = False
            for abbr, target in dk.abbreviations.items():
                if target == cname and re.search(rf'\b{re.escape(abbr)}\b', text):
                    match_types["abbreviation"] += 1
                    examples_by_type["abbreviation"].append(
                        (ds, snum, cname, f"via '{abbr}': {text[:60]}"))
                    found_abbr = True
                    break
            if found_abbr:
                continue

            # Check synonym match
            found_syn = False
            for syn, target in dk.synonyms.items():
                if target == cname and re.search(rf'\b{re.escape(syn)}\b', text, re.IGNORECASE):
                    match_types["synonym"] += 1
                    examples_by_type["synonym"].append(
                        (ds, snum, cname, f"via '{syn}': {text[:60]}"))
                    found_syn = True
                    break
            if found_syn:
                continue

            # Check partial match
            found_partial = False
            for partial, target in dk.partial_references.items():
                if target == cname and re.search(rf'\b{re.escape(partial)}\b', text, re.IGNORECASE):
                    match_types["partial"] += 1
                    examples_by_type["partial"].append(
                        (ds, snum, cname, f"via '{partial}': {text[:60]}"))
                    found_partial = True
                    break
            if found_partial:
                continue

            # Check if component name appears as substring of any word
            if cname.lower() in text.lower():
                match_types["substring_embed"] += 1
                examples_by_type["substring_embed"].append(
                    (ds, snum, cname, text[:80]))
                continue

            # No match found — requires coreference or implicit reference
            match_types["no_name_match"] += 1
            examples_by_type["no_name_match"].append(
                (ds, snum, cname, text[:80]))

        print(f"\n  [{ds}] Gold links: {len(gold)}")
        for mtype, count in match_types.most_common():
            pct = 100 * count / len(gold)
            print(f"    {mtype:25s}: {count:3d} ({pct:5.1f}%)")

    # Show examples of hard categories
    for mtype in ["no_name_match", "synonym", "abbreviation", "partial",
                   "exact_lowercase", "camelcase_split", "substring_embed"]:
        examples = examples_by_type.get(mtype, [])
        if examples:
            print(f"\n  Examples of '{mtype}' ({len(examples)} total, showing ≤5):")
            for ds, snum, cname, text in examples[:5]:
                print(f"    [{ds}] S{snum} {cname}: {text}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 6: Extraction rule completeness — do rules cover all gold match types?
# ═══════════════════════════════════════════════════════════════════════════════

def check_extraction_rule_completeness():
    """Map each ENTITY_EXTRACTION_RULES rule to the gold-standard match types."""
    print("\n" + "=" * 70)
    print("CHECK 6: EXTRACTION RULE → GOLD MATCH TYPE MAPPING")
    print("=" * 70)

    mapping = {
        "Rule 1 (name/alias appears)": ["exact_propercase", "exact_lowercase", "abbreviation", "synonym"],
        "Rule 2 (space-separated)": ["camelcase_split"],
        "Rule 3 (describes what component does)": ["exact_propercase", "synonym"],
        "Rule 4 (known synonym/partial)": ["synonym", "partial"],
        "Rule 5 (interaction)": ["exact_propercase", "synonym"],
        "Rule 6 (passive/prepositional)": ["exact_propercase"],
        "Exclude 1 (dotted path)": ["filters exact matches in paths"],
        "Exclude 2 (ordinary English)": ["filters ambig lowercase matches"],
    }
    print("\n  Extraction rules → match types they enable:")
    for rule, types in mapping.items():
        print(f"    {rule}")
        print(f"      → {', '.join(types)}")

    print("\n  UNCOVERED gold match types:")
    print("    'no_name_match' → requires COREFERENCE (Rule not in extraction, handled by coref phase)")
    print("    'substring_embed' → component name embedded in longer word.")
    print("      → NOT covered by any extraction rule. These rely on LLM inference.")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 7: Seed disambiguation — FP rejection patterns
# ═══════════════════════════════════════════════════════════════════════════════

def check_seed_fp_patterns(ckpts, all_sents, golds):
    """Analyze seed FP patterns to check if SEED_DISAMBIGUATION_RULES covers them."""
    print("\n" + "=" * 70)
    print("CHECK 7: SEED FP PATTERNS vs DISAMBIGUATION RULES")
    print("=" * 70)

    from llm_sad_sam.linkers.experimental.s_linker11a import SLinker11a

    # Read the disambiguation rules from the class
    rules_text = SLinker11a.SEED_DISAMBIGUATION_RULES

    # Extract rejection categories
    reject_cats = re.findall(r'- ([A-Z][\w\s]+?):', rules_text)
    print(f"\n  Disambiguation rejection categories: {reject_cats}")

    print(f"\n  Actual seed FPs and their likely rejection category:")
    for ds in ckpts:
        l1 = ckpts[ds]["layer1"]
        gold = golds[ds]
        sents = all_sents[ds]
        raw_seeds = l1["raw_seed_links"]
        mk = l1["model_knowledge"]

        fps = [sl for sl in raw_seeds
               if (sl.sentence_number, sl.component_id) not in gold]

        for sl in fps:
            text = sents.get(sl.sentence_number, "")
            comp_lower = sl.component_name.lower()

            # Classify the FP
            category = "unknown"
            evidence = ""

            # Check dotted path
            if re.search(rf'\.{re.escape(comp_lower)}\.', text) or \
               re.search(rf'\.{re.escape(comp_lower)}\b', text):
                category = "Code-level notation"
                evidence = "name inside dotted path"

            # Check if ambiguous + lowercase
            elif sl.component_name in mk.ambiguous_names and \
                 not re.search(rf'\b{re.escape(sl.component_name)}\b', text) and \
                 re.search(rf'\b{re.escape(comp_lower)}\b', text):
                category = "Generic English"
                evidence = f"ambiguous '{sl.component_name}' only in lowercase"

            # Check if component name is part of a longer phrase
            elif re.search(rf'\b{re.escape(sl.component_name)}\b', text):
                # Name appears proper-case — check if it's embedded
                for m in re.finditer(rf'\b{re.escape(sl.component_name)}\b', text):
                    rest = text[m.end():m.end()+30].strip()
                    if rest and rest[0].isupper():
                        category = "Embedded sub-entity"
                        evidence = f"followed by '{rest.split()[0]}'"
                    else:
                        category = "Different entity / context unclear"
                        evidence = f"proper case but FP"

            # Check E2E-style adjective usage
            elif comp_lower in text.lower() and sl.component_name in {"E2E"}:
                category = "Generic English"
                evidence = "used as adjective (end-to-end)"

            else:
                category = "Technique or methodology / context unclear"
                evidence = "no clear match pattern"

            print(f"    [{ds}] S{sl.sentence_number} → {sl.component_name}")
            print(f"      Category: {category}")
            print(f"      Evidence: {evidence}")
            print(f"      Text: \"{text[:100]}\"")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading benchmark data...")
    all_comp_names = load_all_component_names()
    all_sents = load_all_sentences()
    golds = load_gold_standards()
    ckpts = load_checkpoints()
    print(f"Loaded: {len(all_comp_names)} datasets, "
          f"{sum(len(v) for v in all_sents.values())} total sentences, "
          f"{sum(len(v) for v in golds.values())} gold links, "
          f"{len(ckpts)} checkpoint sets\n")

    # Run all checks
    check_term_overlap(all_comp_names)
    check_abbreviation_coverage(all_sents, all_comp_names, ckpts)
    check_ambiguity_value(all_comp_names, all_sents, golds, ckpts)
    check_judge_calibration(ckpts, all_comp_names)
    check_extraction_alias_coverage(ckpts, all_sents, golds)
    check_extraction_rule_completeness()
    check_seed_fp_patterns(ckpts, all_sents, golds)

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
