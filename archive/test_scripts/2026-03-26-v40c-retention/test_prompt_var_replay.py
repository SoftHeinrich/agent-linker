#!/usr/bin/env python3
"""Single-step checkpoint replay for each prompt_var.py Pareto variant.

Loads S-Linker10 checkpoints, replays the affected phase with the variant
prompt, merges with other (unchanged) checkpoint data, evals vs gold.

Usage:
    python test_prompt_var_replay.py P1 [dataset]     # Word Usage threshold
    python test_prompt_var_replay.py P2 [dataset]     # Judge Rules scoped doubt
    python test_prompt_var_replay.py P3 [dataset]     # Alias Rule evidential
    python test_prompt_var_replay.py P4 [dataset]     # Extraction prefix rule
    python test_prompt_var_replay.py P5 [dataset]     # Generic Detection anchored
    python test_prompt_var_replay.py ALL [dataset]    # Run all variants
"""
import csv, json, os, pickle, re, sys, time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

os.environ.setdefault("CLAUDE_MODEL", "sonnet")

from llm_sad_sam.core.data_types import SadSamLink, CandidateLink, ModelKnowledge, DocumentKnowledge
from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend

BENCH = Path("/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark")
CACHE = Path("results/phase_cache/s_linker10")

DATASETS = {
    "mediastore":    {"text": BENCH/"mediastore/text_2016/mediastore.txt",       "model": BENCH/"mediastore/model_2016/pcm/ms.repository",       "gold": BENCH/"mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"},
    "teastore":      {"text": BENCH/"teastore/text_2020/teastore.txt",            "model": BENCH/"teastore/model_2020/pcm/teastore.repository",    "gold": BENCH/"teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"},
    "teammates":     {"text": BENCH/"teammates/text_2021/teammates.txt",          "model": BENCH/"teammates/model_2021/pcm/teammates.repository",  "gold": BENCH/"teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"},
    "bigbluebutton": {"text": BENCH/"bigbluebutton/text_2021/bigbluebutton.txt",  "model": BENCH/"bigbluebutton/model_2021/pcm/bbb.repository",    "gold": BENCH/"bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"},
    "jabref":        {"text": BENCH/"jabref/text_2021/jabref.txt",                "model": BENCH/"jabref/model_2021/pcm/jabref.repository",        "gold": BENCH/"jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"},
}


def load_gold(path):
    links = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def load_ckp(ds, phase):
    with open(CACHE / ds / f"{phase}.pkl", "rb") as f:
        return pickle.load(f)


def eval_m(predicted, gold):
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return {"tp": tp, "fp": fp, "fn": fn, "P": p, "R": r, "F1": f1}


def build_final(seed_links, validated, coref_links, partial_validated):
    """Merge all link sources with dedup (seed > entity > coref > partial)."""
    entity_links = [SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
                    for c in validated]
    partial_links = [SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, "partial_inject")
                     for c in partial_validated]
    all_links = seed_links + entity_links + coref_links + partial_links
    seen = set()
    final = []
    for lk in all_links:
        key = (lk.sentence_number, lk.component_id)
        if key not in seen:
            seen.add(key)
            final.append(lk)
    return final


def print_result(label, m, m_orig=None):
    print(f"  {label}: P={m['P']:.1%} R={m['R']:.1%} F1={m['F1']:.1%} (TP={m['tp']} FP={m['fp']} FN={m['fn']})")
    if m_orig:
        d = m["F1"] - m_orig["F1"]
        print(f"    Delta: F1 {d:+.1%} ({m['fp']-m_orig['fp']:+d} FP, {m['fn']-m_orig['fn']:+d} FN)")


# ═══════════════════════════════════════════════════════════════════════════════
# P1: WORD_USAGE_PROMPT_V — "recurring pattern" threshold
# Replays Tier 1.5 enrichment with variant prompt.
# ═══════════════════════════════════════════════════════════════════════════════

def test_P1(target_datasets=None):
    from llm_sad_sam.linkers.experimental.s_linker10 import SLinker10
    from llm_sad_sam.linkers.experimental.prompt_var import WORD_USAGE_PROMPT_V
    from llm_sad_sam.linkers.experimental.prompts_v2 import WORD_USAGE_PROMPT

    print("=" * 70)
    print("P1: WORD_USAGE_PROMPT_V — 'recurring pattern' vs 'even ONE'")
    print("=" * 70)

    llm = LLMClient(backend=LLMBackend.CLAUDE)
    results_all = {}

    for ds, paths in DATASETS.items():
        if target_datasets and ds not in target_datasets:
            continue

        gold = load_gold(paths["gold"])
        tier1 = load_ckp(ds, "tier1")
        tier1_5 = load_ckp(ds, "tier1_5")
        tier2 = load_ckp(ds, "tier2")

        components = parse_pcm_repository(str(paths["model"]))
        sentences = DocumentLoader.load_sentences(str(paths["text"]))

        # Original enrichment result
        dk_orig = tier1_5["doc_knowledge"]
        orig_partials = dk_orig.partial_references.copy()
        p3_partials = tier1["doc_knowledge"].partial_references.copy()
        enriched_orig = {k: v for k, v in orig_partials.items() if k not in p3_partials}

        # --- Replay enrichment with VARIANT prompt ---
        linker = SLinker10.__new__(SLinker10)
        linker.llm = llm
        linker.model_knowledge = tier1["model_knowledge"]
        linker.doc_knowledge = DocumentKnowledge()
        # Copy Phase 3 knowledge
        linker.doc_knowledge.abbreviations = tier1["doc_knowledge"].abbreviations.copy()
        linker.doc_knowledge.synonyms = tier1["doc_knowledge"].synonyms.copy()
        linker.doc_knowledge.partial_references = p3_partials.copy()
        linker._generic_partials = tier1["generic_partials"]
        linker._components = components

        # Monkey-patch the prompt used in enrichment
        import llm_sad_sam.linkers.experimental.prompts_v2 as pmod
        saved_prompt = pmod.WORD_USAGE_PROMPT
        pmod.WORD_USAGE_PROMPT = WORD_USAGE_PROMPT_V
        try:
            linker._enrich_multiword_partials(sentences, components)
        finally:
            pmod.WORD_USAGE_PROMPT = saved_prompt

        variant_partials = linker.doc_knowledge.partial_references.copy()
        enriched_variant = {k: v for k, v in variant_partials.items() if k not in p3_partials}

        print(f"\n  {ds}:")
        print(f"    Original enrichment: {enriched_orig}")
        print(f"    Variant enrichment:  {enriched_variant}")

        if enriched_orig == enriched_variant:
            print(f"    → IDENTICAL — no F1 impact")
            # Get original final for reference
            final_orig = load_ckp(ds, "final")["final"]
            orig_pairs = {(l.sentence_number, l.component_id) for l in final_orig}
            m = eval_m(orig_pairs, gold)
            results_all[ds] = {"orig": m, "variant": m, "changed": False}
        else:
            added = {k: v for k, v in enriched_variant.items() if k not in enriched_orig}
            removed = {k: v for k, v in enriched_orig.items() if k not in enriched_variant}
            if added: print(f"    Added by variant: {added}")
            if removed: print(f"    Removed by variant: {removed}")

            # Simulate downstream: re-inject partials with changed doc_knowledge
            # and re-validate, then build final
            linker.doc_knowledge.partial_references = variant_partials
            sent_map = DocumentLoader.build_sent_map(sentences)
            name_to_id = {c.name: c.id for c in components}
            seed_set = tier1["seed_set"]
            validated_set = {(c.sentence_number, c.component_id) for c in tier2["validated"]}
            coref_set = {(l.sentence_number, l.component_id) for l in tier2["coref_links"]}

            partial_cands = linker._inject_partial_candidates(
                sentences, components, name_to_id, sent_map, seed_set, validated_set, coref_set)
            if partial_cands:
                partial_validated = linker._validate_intersect(partial_cands, components, sent_map)
            else:
                partial_validated = []

            final_variant = build_final(tier1["seed_links"], tier2["validated"],
                                        tier2["coref_links"], partial_validated)
            variant_pairs = {(l.sentence_number, l.component_id) for l in final_variant}

            final_orig = load_ckp(ds, "final")["final"]
            orig_pairs = {(l.sentence_number, l.component_id) for l in final_orig}
            m_orig = eval_m(orig_pairs, gold)
            m_var = eval_m(variant_pairs, gold)
            print_result("Original", m_orig)
            print_result("Variant ", m_var, m_orig)
            results_all[ds] = {"orig": m_orig, "variant": m_var, "changed": True}

    _print_macro(results_all, "P1")


# ═══════════════════════════════════════════════════════════════════════════════
# P2: DOC_KNOWLEDGE_JUDGE_RULES_V — scoped doubt clause
# Re-runs Phase 3 (extract + judge) with variant judge rules.
# ═══════════════════════════════════════════════════════════════════════════════

def test_P2(target_datasets=None):
    from llm_sad_sam.linkers.experimental.prompts_v2 import (
        DOC_KNOWLEDGE_EXTRACTION_RULES, DOC_KNOWLEDGE_JUDGE_EXAMPLES,
        DOC_KNOWLEDGE_JUDGE_RULES,
    )
    from llm_sad_sam.linkers.experimental.prompt_var import DOC_KNOWLEDGE_JUDGE_RULES_V

    print("=" * 70)
    print("P2: DOC_KNOWLEDGE_JUDGE_RULES_V — scoped doubt (no blanket APPROVE)")
    print("=" * 70)

    llm = LLMClient(backend=LLMBackend.CLAUDE)
    results_all = {}

    for ds, paths in DATASETS.items():
        if target_datasets and ds not in target_datasets:
            continue

        gold = load_gold(paths["gold"])
        tier1 = load_ckp(ds, "tier1")
        components = parse_pcm_repository(str(paths["model"]))
        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        comp_names = [c.name for c in components]
        doc_lines = [s.text for s in sentences]

        print(f"\n  {ds}: re-running Phase 3 (extract + judge)...")

        # Step 1: Extract (same prompt as original)
        prompt1 = f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

{DOC_KNOWLEDGE_EXTRACTION_RULES}

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": {{"short_form": "FullComponent"}},
  "synonyms": {{"specific_alternative_name": "FullComponent"}},
  "partial_references": {{"partial_name": "FullComponent"}}
}}
JSON only:"""

        data1 = llm.extract_json(llm.query(prompt1, timeout=300))

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

        print(f"    Extracted {len(all_mappings)} raw mappings")

        if not all_mappings:
            print(f"    → No mappings to judge")
            final_orig = load_ckp(ds, "final")["final"]
            orig_pairs = {(l.sentence_number, l.component_id) for l in final_orig}
            m = eval_m(orig_pairs, gold)
            results_all[ds] = {"orig": m, "variant": m}
            continue

        mapping_list = [f"'{k}' -> {v[1]} ({v[0]})" for k, v in list(all_mappings.items())[:25]]

        # Step 2a: Judge with ORIGINAL rules
        def run_judge(rules_text, label):
            prompt2 = f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

{DOC_KNOWLEDGE_JUDGE_EXAMPLES}

{rules_text}

Return JSON:
{{
  "approved": ["term1", "term2"]
}}
JSON only:"""
            data2 = llm.extract_json(llm.query(prompt2, timeout=120))
            approved = set(data2.get("approved", [])) if data2 else set(all_mappings.keys())
            return approved

        orig_approved = run_judge(DOC_KNOWLEDGE_JUDGE_RULES, "original")
        var_approved = run_judge(DOC_KNOWLEDGE_JUDGE_RULES_V, "variant")

        # Compare
        added = var_approved - orig_approved
        removed = orig_approved - var_approved
        print(f"    Original approved: {sorted(orig_approved)}")
        print(f"    Variant approved:  {sorted(var_approved)}")
        if added: print(f"    +Added by variant: {sorted(added)}")
        if removed: print(f"    -Removed by variant: {sorted(removed)}")
        if not added and not removed: print(f"    → IDENTICAL judge decisions")

        # For F1 impact: we'd need to cascade through enrichment + entity pipeline
        # Just report judge-level diff here
        final_orig = load_ckp(ds, "final")["final"]
        orig_pairs = {(l.sentence_number, l.component_id) for l in final_orig}
        m = eval_m(orig_pairs, gold)
        results_all[ds] = {"orig": m, "variant": m, "judge_added": added, "judge_removed": removed}

    _print_macro(results_all, "P2 (judge-level, no cascade)")


# ═══════════════════════════════════════════════════════════════════════════════
# P3: ALIAS_RULE_V — evidential "CAN refer" framing
# Replays 2-pass validation with variant alias rule.
# ═══════════════════════════════════════════════════════════════════════════════

def test_P3(target_datasets=None):
    from llm_sad_sam.linkers.experimental.s_linker10 import SLinker10
    from llm_sad_sam.linkers.experimental.prompts_v2 import VALIDATION_RULES
    from llm_sad_sam.linkers.experimental.prompt_var import ALIAS_RULE_V

    print("=" * 70)
    print("P3: ALIAS_RULE_V — 'CAN refer' vs 'IS a reference'")
    print("=" * 70)

    llm = LLMClient(backend=LLMBackend.CLAUDE)
    results_all = {}

    for ds, paths in DATASETS.items():
        if target_datasets and ds not in target_datasets:
            continue

        gold = load_gold(paths["gold"])
        tier1 = load_ckp(ds, "tier1")
        tier2 = load_ckp(ds, "tier2")
        components = parse_pcm_repository(str(paths["model"]))
        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        sent_map = DocumentLoader.build_sent_map(sentences)
        comp_names = [c.name for c in components]

        # Get needs_validation candidates from tier2
        needs = [c for c in tier2["validated"] if c.needs_validation]
        bypass = [c for c in tier2["validated"] if not c.needs_validation]

        if not needs:
            print(f"\n  {ds}: no needs_validation candidates, skipping")
            final_orig = load_ckp(ds, "final")["final"]
            orig_pairs = {(l.sentence_number, l.component_id) for l in final_orig}
            m = eval_m(orig_pairs, gold)
            results_all[ds] = {"orig": m, "variant": m}
            continue

        print(f"\n  {ds}: re-validating {len(needs)} candidates with variant alias rule...")

        # Build alias map
        alias_map = {}
        dk = tier1["doc_knowledge"]
        for c in components:
            aliases = {}
            for a, cn in dk.abbreviations.items():
                if cn == c.name: aliases[a] = "abbreviation"
            for s, cn in dk.synonyms.items():
                if cn == c.name: aliases[s] = "synonym"
            for p, cn in dk.partial_references.items():
                if cn == c.name: aliases[p] = "partial reference"
            # Also check tier1_5 enrichment partials
            dk15 = load_ckp(ds, "tier1_5")["doc_knowledge"]
            for p, cn in dk15.partial_references.items():
                if cn == c.name and p not in aliases:
                    aliases[p] = "partial reference"
            alias_map[c.name] = aliases

        # Build cases with alias hints
        cases = []
        has_alias_flags = []
        for i, c in enumerate(needs):
            prev = sent_map.get(c.sentence_number - 1)
            p = f"[prev: {prev.text[:35]}...] " if prev else ""
            alias_hint = ""
            matched_lower = c.matched_text.lower() if c.matched_text else ""
            if matched_lower and matched_lower != c.component_name.lower():
                aliases = alias_map.get(c.component_name, {})
                for alias, atype in aliases.items():
                    if alias.lower() in matched_lower or matched_lower in alias.lower():
                        alias_hint = f'\n  [KNOWN ALIAS: "{alias}" is a known {atype} for "{c.component_name}"]'
                        break
            has_alias_flags.append(bool(alias_hint))
            cases.append(f'Case {i+1}: "{c.matched_text}" -> {c.component_name}{alias_hint}\n  {p}"{c.sentence_text}"')

        def run_validation_pass(focus, alias_rule_text):
            has_alias = any("[KNOWN ALIAS:" in c for c in cases)
            alias_rule = alias_rule_text if has_alias else ""

            prompt = f"""Validate component references in a software architecture document. {focus}

COMPONENTS: {', '.join(comp_names)}

{VALIDATION_RULES}{alias_rule}

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

            data = llm.extract_json(llm.query(prompt, timeout=120))
            results = {}
            if data:
                for v in data.get("validations", []):
                    idx = v.get("case", 0) - 1
                    if 0 <= idx < len(cases):
                        results[idx] = v.get("approve", False)
            return results

        # Original alias rule
        ORIG_ALIAS = ("\n- When a KNOWN ALIAS is indicated, the word IS a reference to that component "
                      "unless the sentence clearly uses it in an unrelated sense")

        # Run with original
        r1_orig = run_validation_pass(
            "Focus on ACTOR role: is the component performing an action or being described?",
            ORIG_ALIAS)
        r2_orig = run_validation_pass(
            "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component?",
            ORIG_ALIAS)

        # Run with variant
        r1_var = run_validation_pass(
            "Focus on ACTOR role: is the component performing an action or being described?",
            ALIAS_RULE_V)
        r2_var = run_validation_pass(
            "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component?",
            ALIAS_RULE_V)

        # Compare approval decisions
        orig_approved = set()
        var_approved = set()
        for i, c in enumerate(needs):
            p1o, p2o = r1_orig.get(i, False), r2_orig.get(i, False)
            p1v, p2v = r1_var.get(i, False), r2_var.get(i, False)
            # Evidence-stratified: union for alias, intersection for exact
            a_orig = (p1o or p2o) if has_alias_flags[i] else (p1o and p2o)
            a_var = (p1v or p2v) if has_alias_flags[i] else (p1v and p2v)
            key = (c.sentence_number, c.component_id)
            if a_orig: orig_approved.add(key)
            if a_var: var_approved.add(key)

        killed = orig_approved - var_approved
        gained = var_approved - orig_approved
        killed_tp = killed & gold
        killed_fp = killed - gold
        gained_tp = gained & gold
        gained_fp = gained - gold

        print(f"    Original approved: {len(orig_approved)} / {len(needs)}")
        print(f"    Variant approved:  {len(var_approved)} / {len(needs)}")
        if killed:
            id_to_name = {c.id: c.name for c in components}
            print(f"    KILLED ({len(killed)}): {len(killed_tp)} TP, {len(killed_fp)} FP")
            for snum, cid in sorted(killed):
                label = "TP" if (snum, cid) in gold else "FP"
                print(f"      {label}: S{snum} -> {id_to_name.get(cid, cid)}")
        if gained:
            id_to_name = {c.id: c.name for c in components}
            print(f"    GAINED ({len(gained)}): {len(gained_tp)} TP, {len(gained_fp)} FP")
            for snum, cid in sorted(gained):
                label = "TP" if (snum, cid) in gold else "FP"
                print(f"      {label}: S{snum} -> {id_to_name.get(cid, cid)}")
        if not killed and not gained:
            print(f"    → IDENTICAL validation decisions")

        # Build final with variant-approved validation
        var_validated_cands = bypass + [c for c in needs if (c.sentence_number, c.component_id) in var_approved]
        final_variant = build_final(tier1["seed_links"], var_validated_cands,
                                    tier2["coref_links"], tier2.get("partial_validated", []))
        var_pairs = {(l.sentence_number, l.component_id) for l in final_variant}

        final_orig = load_ckp(ds, "final")["final"]
        orig_pairs = {(l.sentence_number, l.component_id) for l in final_orig}
        m_orig = eval_m(orig_pairs, gold)
        m_var = eval_m(var_pairs, gold)
        print_result("Original", m_orig)
        print_result("Variant ", m_var, m_orig)
        results_all[ds] = {"orig": m_orig, "variant": m_var}

    _print_macro(results_all, "P3")


# ═══════════════════════════════════════════════════════════════════════════════
# P4: ENTITY_EXTRACTION_RULES_V — prefix disambiguation
# Re-runs entity extraction with variant rules, then validates + merges.
# ═══════════════════════════════════════════════════════════════════════════════

def test_P4(target_datasets=None):
    from llm_sad_sam.linkers.experimental.s_linker10 import SLinker10

    print("=" * 70)
    print("P4: ENTITY_EXTRACTION_RULES_V — prefix disambiguation rule")
    print("=" * 70)

    llm = LLMClient(backend=LLMBackend.CLAUDE)
    results_all = {}

    for ds, paths in DATASETS.items():
        if target_datasets and ds not in target_datasets:
            continue

        gold = load_gold(paths["gold"])
        tier1 = load_ckp(ds, "tier1")
        tier1_5 = load_ckp(ds, "tier1_5")
        tier2 = load_ckp(ds, "tier2")
        components = parse_pcm_repository(str(paths["model"]))
        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        sent_map = DocumentLoader.build_sent_map(sentences)
        name_to_id = {c.name: c.id for c in components}

        print(f"\n  {ds}: re-running entity extraction with variant rules...")

        # Build patched linker
        linker = SLinker10.__new__(SLinker10)
        linker.llm = llm
        linker.model_knowledge = tier1["model_knowledge"]
        linker.doc_knowledge = tier1_5["doc_knowledge"]
        linker._generic_partials = tier1["generic_partials"]
        linker._components = components

        # Monkey-patch extraction rules
        import llm_sad_sam.linkers.experimental.prompts_v2 as pmod
        from llm_sad_sam.linkers.experimental.prompt_var import ENTITY_EXTRACTION_RULES_V
        saved_rules = pmod.ENTITY_EXTRACTION_RULES
        pmod.ENTITY_EXTRACTION_RULES = ENTITY_EXTRACTION_RULES_V
        try:
            candidates = linker._extract_entities_enriched(sentences, components, name_to_id, sent_map)
        finally:
            pmod.ENTITY_EXTRACTION_RULES = saved_rules

        print(f"    Variant extraction: {len(candidates)} candidates")

        # Validate with ORIGINAL validation (only testing extraction change)
        validated = linker._validate_intersect(candidates, components, sent_map)
        print(f"    Validated: {len(validated)} / {len(candidates)}")

        # Build final
        seed_set = tier1["seed_set"]
        validated_set = {(c.sentence_number, c.component_id) for c in validated}
        coref_set = {(l.sentence_number, l.component_id) for l in tier2["coref_links"]}
        partial_cands = linker._inject_partial_candidates(
            sentences, components, name_to_id, sent_map, seed_set, validated_set, coref_set)
        partial_validated = linker._validate_intersect(partial_cands, components, sent_map) if partial_cands else []

        final_variant = build_final(tier1["seed_links"], validated,
                                    tier2["coref_links"], partial_validated)
        var_pairs = {(l.sentence_number, l.component_id) for l in final_variant}

        final_orig = load_ckp(ds, "final")["final"]
        orig_pairs = {(l.sentence_number, l.component_id) for l in final_orig}
        m_orig = eval_m(orig_pairs, gold)
        m_var = eval_m(var_pairs, gold)
        print_result("Original", m_orig)
        print_result("Variant ", m_var, m_orig)

        # Show diffs
        new_links = var_pairs - orig_pairs
        lost_links = orig_pairs - var_pairs
        id_to_name = {c.id: c.name for c in components}
        if new_links:
            print(f"    NEW links ({len(new_links)}):")
            for snum, cid in sorted(new_links):
                label = "TP" if (snum, cid) in gold else "FP"
                print(f"      {label}: S{snum} -> {id_to_name.get(cid, cid)}")
        if lost_links:
            print(f"    LOST links ({len(lost_links)}):")
            for snum, cid in sorted(lost_links):
                label = "TP" if (snum, cid) in gold else "FP"
                print(f"      {label}: S{snum} -> {id_to_name.get(cid, cid)}")

        results_all[ds] = {"orig": m_orig, "variant": m_var}

    _print_macro(results_all, "P4")


# ═══════════════════════════════════════════════════════════════════════════════
# P5: GENERIC_DETECTION_DISTINCTION_V — anchor contrast instruction
# Replays generic detection within validation using variant distinction.
# ═══════════════════════════════════════════════════════════════════════════════

def test_P5(target_datasets=None):
    from llm_sad_sam.linkers.experimental.s_linker10 import SLinker10
    from llm_sad_sam.linkers.experimental.prompt_var import GENERIC_DETECTION_DISTINCTION_V

    print("=" * 70)
    print("P5: GENERIC_DETECTION_DISTINCTION_V — anchor contrast instruction")
    print("=" * 70)

    llm = LLMClient(backend=LLMBackend.CLAUDE)
    results_all = {}

    for ds, paths in DATASETS.items():
        if target_datasets and ds not in target_datasets:
            continue

        gold = load_gold(paths["gold"])
        tier1 = load_ckp(ds, "tier1")
        tier1_5 = load_ckp(ds, "tier1_5")
        tier2 = load_ckp(ds, "tier2")
        components = parse_pcm_repository(str(paths["model"]))
        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        sent_map = DocumentLoader.build_sent_map(sentences)

        # Get needs_validation candidates to find generic-mention ones
        needs = [c for c in tier2["validated"] if c.needs_validation]
        bypass = [c for c in tier2["validated"] if not c.needs_validation]

        if not needs:
            print(f"\n  {ds}: no needs_validation candidates, skipping")
            final_orig = load_ckp(ds, "final")["final"]
            orig_pairs = {(l.sentence_number, l.component_id) for l in final_orig}
            m = eval_m(orig_pairs, gold)
            results_all[ds] = {"orig": m, "variant": m}
            continue

        # Build linker for helper methods
        linker = SLinker10.__new__(SLinker10)
        linker.llm = llm
        linker.model_knowledge = tier1["model_knowledge"]
        linker.doc_knowledge = tier1_5["doc_knowledge"]
        linker._generic_partials = tier1["generic_partials"]
        linker._components = components

        # Identify generic-mention candidates (same logic as _validate_intersect)
        generic_candidates = {}
        non_generic = []
        for c in needs:
            sent = sent_map.get(c.sentence_number)
            if not sent:
                non_generic.append(c)
                continue
            comp_lower = c.component_name.lower()
            has_exact_case = linker._has_standalone_mention(c.component_name, sent.text)
            has_lowercase = (not has_exact_case and
                             re.search(rf'\b{re.escape(comp_lower)}\b', sent.text))
            if not has_lowercase and linker.doc_knowledge:
                for partial, target in linker.doc_knowledge.partial_references.items():
                    if target == c.component_name:
                        partial_lower = partial.lower()
                        if (re.search(rf'\b{re.escape(partial_lower)}\b', sent.text.lower())
                                and not re.search(rf'\b{re.escape(partial)}\b', sent.text)):
                            has_lowercase = True
                            break
            if has_lowercase and linker._is_ambiguous_name_component(c.component_name):
                generic_candidates.setdefault(c.component_name, []).append(c)
            else:
                non_generic.append(c)

        if not generic_candidates:
            print(f"\n  {ds}: no generic-mention candidates, skipping P5")
            final_orig = load_ckp(ds, "final")["final"]
            orig_pairs = {(l.sentence_number, l.component_id) for l in final_orig}
            m = eval_m(orig_pairs, gold)
            results_all[ds] = {"orig": m, "variant": m}
            continue

        print(f"\n  {ds}: {sum(len(v) for v in generic_candidates.values())} generic-mention candidates "
              f"across {list(generic_candidates.keys())}")

        # Run generic detection with ORIGINAL distinction
        ORIG_DISTINCTION = ("Key distinction: A component reference names a specific system entity as a participant.\n"
                           "A generic use describes a type of activity or quality that happens to share the word.")

        def run_generic_detection(distinction_text, label):
            remaining = list(non_generic)
            for comp_name, cands in generic_candidates.items():
                anchor_lines = []
                for s in sent_map.values():
                    if linker._has_standalone_mention(comp_name, s.text):
                        anchor_lines.append(f"  S{s.number}: {s.text}")
                        if len(anchor_lines) >= 5:
                            break

                case_lines = []
                for i, c in enumerate(cands):
                    s = sent_map.get(c.sentence_number)
                    prev = sent_map.get(c.sentence_number - 1)
                    prev_text = f" [prev: {prev.text[:60]}]" if prev else ""
                    case_lines.append(f"  Case {i+1} (S{c.sentence_number}): {s.text}{prev_text}")

                anchor_section = ""
                if anchor_lines:
                    anchor_section = (
                        f'FULL-NAME REFERENCES (these definitely refer to the {comp_name} component):\n'
                        + '\n'.join(anchor_lines) + '\n\n'
                    )

                prompt = f"""CONTEXTUAL WORD USAGE: Does the word refer to the architecture component "{comp_name}", or is it used as an ordinary English word?

{anchor_section}SENTENCES TO CHECK (the component name appears only in lowercase or as part of a compound phrase):
{chr(10).join(case_lines)}

For each case, determine:
- COMPONENT: The word refers to the specific "{comp_name}" component as a system entity
- GENERIC: The word is used as ordinary English describing a general concept

{distinction_text}

Return JSON:
{{"results": [{{"case": 1, "usage": "component" or "generic", "reason": "brief"}}]}}
JSON only:"""

                data = llm.extract_json(llm.query(prompt, timeout=120))
                if not data:
                    remaining.extend(cands)
                    continue

                results_map = {}
                for r in data.get("results", []):
                    idx = r.get("case", 0) - 1
                    results_map[idx] = r.get("usage", "component")

                for i, c in enumerate(cands):
                    usage = results_map.get(i, "component")
                    if usage != "generic":
                        remaining.append(c)

            return remaining

        orig_remaining = run_generic_detection(ORIG_DISTINCTION, "original")
        var_remaining = run_generic_detection(GENERIC_DETECTION_DISTINCTION_V, "variant")

        orig_kept = {(c.sentence_number, c.component_id) for c in orig_remaining}
        var_kept = {(c.sentence_number, c.component_id) for c in var_remaining}

        killed = orig_kept - var_kept
        gained = var_kept - orig_kept

        id_to_name = {c.id: c.name for c in components}
        print(f"    Original kept: {len(orig_kept)} (of {len(needs)})")
        print(f"    Variant kept:  {len(var_kept)} (of {len(needs)})")
        if killed:
            killed_tp = killed & gold
            killed_fp = killed - gold
            print(f"    NEW REJECTS by variant ({len(killed)}): {len(killed_tp)} TP, {len(killed_fp)} FP")
            for snum, cid in sorted(killed):
                label = "TP" if (snum, cid) in gold else "FP"
                print(f"      {label}: S{snum} -> {id_to_name.get(cid, cid)}")
        if gained:
            gained_tp = gained & gold
            gained_fp = gained - gold
            print(f"    NEW KEEPS by variant ({len(gained)}): {len(gained_tp)} TP, {len(gained_fp)} FP")
            for snum, cid in sorted(gained):
                label = "TP" if (snum, cid) in gold else "FP"
                print(f"      {label}: S{snum} -> {id_to_name.get(cid, cid)}")
        if not killed and not gained:
            print(f"    → IDENTICAL generic detection decisions")

        final_orig = load_ckp(ds, "final")["final"]
        orig_pairs = {(l.sentence_number, l.component_id) for l in final_orig}
        m_orig = eval_m(orig_pairs, gold)
        results_all[ds] = {"orig": m_orig, "variant": m_orig, "killed": killed, "gained": gained}

    _print_macro(results_all, "P5 (generic detection only)")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary helper
# ═══════════════════════════════════════════════════════════════════════════════

def _print_macro(results, label):
    if not results:
        return
    print(f"\n{'=' * 70}")
    print(f"  {label} — MACRO SUMMARY")
    print(f"{'=' * 70}")
    for ds, r in results.items():
        mo, mv = r["orig"], r["variant"]
        d = mv["F1"] - mo["F1"]
        tag = f" ({d:+.1%})" if abs(d) > 0.001 else " (=)"
        print(f"  {ds:15s}: orig={mo['F1']:.1%}  variant={mv['F1']:.1%}{tag}")
    if len(results) > 1:
        orig_macro = sum(r["orig"]["F1"] for r in results.values()) / len(results)
        var_macro = sum(r["variant"]["F1"] for r in results.values()) / len(results)
        d = var_macro - orig_macro
        print(f"  {'MACRO':15s}: orig={orig_macro:.1%}  variant={var_macro:.1%} ({d:+.1%})")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    variant = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
    target = sys.argv[2:] if len(sys.argv) > 2 else None

    tests = {"P1": test_P1, "P2": test_P2, "P3": test_P3, "P4": test_P4, "P5": test_P5}

    if variant == "ALL":
        for name, fn in tests.items():
            fn(target)
    elif variant in tests:
        tests[variant](target)
    else:
        print(f"Unknown variant: {variant}")
        print("Usage: python test_prompt_var_replay.py [P1|P2|P3|P4|P5|ALL] [datasets...]")
