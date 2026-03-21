#!/usr/bin/env python3
"""Round 2: Targeted prompt variant tests from S-Linker10 checkpoints.

Tests new variants from prompt_var.py:
  P3a  — Two-tier alias rule (ALIAS_RULE_V2)
  P3b  — Role-verification alias (ALIAS_RULE_V3)
  P3c  — Combined CAN-refer + role-verify (ALIAS_RULE_V4)
  P4a  — Prefix rule + abbreviation exception (ENTITY_EXTRACTION_RULES_V2)
  V_RV — Validation with role-verification rules (VALIDATION_RULES_V)
  V_FO — Validation with alternative focus prompts
  CMB  — Combined: P3a + V_RV (best alias + best validation)

Usage:
    python test_prompt_var_v2.py P3a [dataset]
    python test_prompt_var_v2.py P3b [dataset]
    python test_prompt_var_v2.py P3c [dataset]
    python test_prompt_var_v2.py V_RV [dataset]
    python test_prompt_var_v2.py V_FO [dataset]
    python test_prompt_var_v2.py CMB [dataset]
    python test_prompt_var_v2.py ALL [dataset]
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
from llm_sad_sam.linkers.experimental.prompts_v2 import VALIDATION_RULES

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


def build_alias_map(components, tier1, ds):
    alias_map = {}
    dk = tier1["doc_knowledge"]
    dk15 = load_ckp(ds, "tier1_5")["doc_knowledge"]
    for c in components:
        aliases = {}
        for a, cn in dk.abbreviations.items():
            if cn == c.name: aliases[a] = "abbreviation"
        for s, cn in dk.synonyms.items():
            if cn == c.name: aliases[s] = "synonym"
        for p, cn in dk.partial_references.items():
            if cn == c.name: aliases[p] = "partial reference"
        for p, cn in dk15.partial_references.items():
            if cn == c.name and p not in aliases:
                aliases[p] = "partial reference"
        alias_map[c.name] = aliases
    return alias_map


def build_cases(needs, sent_map, alias_map):
    """Build validation case strings and alias flags."""
    cases = []
    has_alias = []
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
        has_alias.append(bool(alias_hint))
        cases.append(f'Case {i+1}: "{c.matched_text}" -> {c.component_name}{alias_hint}\n  {p}"{c.sentence_text}"')
    return cases, has_alias


def run_validation(llm, comp_names, cases, focus, rules_text, alias_rule_text):
    """Run one validation pass."""
    has_alias = any("[KNOWN ALIAS:" in c for c in cases)
    alias_rule = alias_rule_text if has_alias else ""

    prompt = f"""Validate component references in a software architecture document. {focus}

COMPONENTS: {', '.join(comp_names)}

{rules_text}{alias_rule}

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


def test_validation_variant(variant_name, alias_rule_text, rules_text, focus1, focus2, target_datasets=None):
    """Generic test: replay validation with given alias rule + rules + focus prompts."""
    print("=" * 70)
    print(f"  {variant_name}")
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

        needs = [c for c in tier2["validated"] if c.needs_validation]
        bypass = [c for c in tier2["validated"] if not c.needs_validation]

        if not needs:
            print(f"\n  {ds}: no needs_validation candidates, skipping")
            final_orig = load_ckp(ds, "final")["final"]
            orig_pairs = {(l.sentence_number, l.component_id) for l in final_orig}
            m = eval_m(orig_pairs, gold)
            results_all[ds] = {"orig": m, "variant": m}
            continue

        print(f"\n  {ds}: re-validating {len(needs)} candidates...")

        alias_map = build_alias_map(components, tier1, ds)
        cases, has_alias_flags = build_cases(needs, sent_map, alias_map)

        r1 = run_validation(llm, comp_names, cases, focus1, rules_text, alias_rule_text)
        r2 = run_validation(llm, comp_names, cases, focus2, rules_text, alias_rule_text)

        var_approved = set()
        for i, c in enumerate(needs):
            p1, p2 = r1.get(i, False), r2.get(i, False)
            approved = (p1 or p2) if has_alias_flags[i] else (p1 and p2)
            if approved:
                var_approved.add((c.sentence_number, c.component_id))

        # Build final with variant validation
        var_validated = bypass + [c for c in needs if (c.sentence_number, c.component_id) in var_approved]
        final_variant = build_final(tier1["seed_links"], var_validated,
                                    tier2["coref_links"], tier2.get("partial_validated", []))
        var_pairs = {(l.sentence_number, l.component_id) for l in final_variant}

        final_orig = load_ckp(ds, "final")["final"]
        orig_pairs = {(l.sentence_number, l.component_id) for l in final_orig}
        m_orig = eval_m(orig_pairs, gold)
        m_var = eval_m(var_pairs, gold)
        print_result("Original", m_orig)
        print_result("Variant ", m_var, m_orig)

        # Show specific changes
        new_links = var_pairs - orig_pairs
        lost_links = orig_pairs - var_pairs
        id_to_name = {c.id: c.name for c in components}
        if lost_links:
            print(f"    LOST ({len(lost_links)}):", end="")
            for snum, cid in sorted(lost_links):
                label = "TP" if (snum, cid) in gold else "FP"
                print(f" {label}:S{snum}->{id_to_name.get(cid, cid)[:15]}", end="")
            print()
        if new_links:
            print(f"    NEW  ({len(new_links)}):", end="")
            for snum, cid in sorted(new_links):
                label = "TP" if (snum, cid) in gold else "FP"
                print(f" {label}:S{snum}->{id_to_name.get(cid, cid)[:15]}", end="")
            print()

        results_all[ds] = {"orig": m_orig, "variant": m_var}

    _print_macro(results_all, variant_name)
    return results_all


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
# Test Functions
# ═══════════════════════════════════════════════════════════════════════════════

# Original prompts for reference
ORIG_ALIAS = ("\n- When a KNOWN ALIAS is indicated, the word IS a reference to that component "
              "unless the sentence clearly uses it in an unrelated sense")
ORIG_FOCUS1 = "Focus on ACTOR role: is the component performing an action or being described?"
ORIG_FOCUS2 = "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component, not a generic concept?"


def test_P3a(target):
    from llm_sad_sam.linkers.experimental.prompt_var import ALIAS_RULE_V2
    return test_validation_variant("P3a: Two-tier alias (strict generic, lenient clear)",
        ALIAS_RULE_V2, VALIDATION_RULES, ORIG_FOCUS1, ORIG_FOCUS2, target)


def test_P3b(target):
    from llm_sad_sam.linkers.experimental.prompt_var import ALIAS_RULE_V3
    return test_validation_variant("P3b: Role-verification alias (IS + verify role)",
        ALIAS_RULE_V3, VALIDATION_RULES, ORIG_FOCUS1, ORIG_FOCUS2, target)


def test_P3c(target):
    from llm_sad_sam.linkers.experimental.prompt_var import ALIAS_RULE_V4
    return test_validation_variant("P3c: CAN-refer + role-verification",
        ALIAS_RULE_V4, VALIDATION_RULES, ORIG_FOCUS1, ORIG_FOCUS2, target)


def test_V_RV(target):
    from llm_sad_sam.linkers.experimental.prompt_var import VALIDATION_RULES_V
    return test_validation_variant("V_RV: Validation rules with role-verification reject",
        ORIG_ALIAS, VALIDATION_RULES_V, ORIG_FOCUS1, ORIG_FOCUS2, target)


def test_V_FO(target):
    from llm_sad_sam.linkers.experimental.prompt_var import VALIDATION_FOCUS_1_V, VALIDATION_FOCUS_2_V
    return test_validation_variant("V_FO: Alternative focus prompts (SPECIFICITY + FUNCTION)",
        ORIG_ALIAS, VALIDATION_RULES, VALIDATION_FOCUS_1_V, VALIDATION_FOCUS_2_V, target)


def test_CMB(target):
    """Combined: P3a alias + V_RV validation rules."""
    from llm_sad_sam.linkers.experimental.prompt_var import ALIAS_RULE_V2, VALIDATION_RULES_V
    return test_validation_variant("CMB: P3a alias + V_RV validation rules",
        ALIAS_RULE_V2, VALIDATION_RULES_V, ORIG_FOCUS1, ORIG_FOCUS2, target)


def test_CMB2(target):
    """Combined: P3c alias + V_RV validation + alternative focus."""
    from llm_sad_sam.linkers.experimental.prompt_var import (
        ALIAS_RULE_V4, VALIDATION_RULES_V, VALIDATION_FOCUS_1_V, VALIDATION_FOCUS_2_V,
    )
    return test_validation_variant("CMB2: P3c + V_RV + alt focus (full stack)",
        ALIAS_RULE_V4, VALIDATION_RULES_V, VALIDATION_FOCUS_1_V, VALIDATION_FOCUS_2_V, target)


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    variant = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
    target = sys.argv[2:] if len(sys.argv) > 2 else None

    tests = {
        "P3A": test_P3a, "P3B": test_P3b, "P3C": test_P3c,
        "V_RV": test_V_RV, "V_FO": test_V_FO,
        "CMB": test_CMB, "CMB2": test_CMB2,
    }

    if variant == "ALL":
        for name, fn in tests.items():
            fn(target)
    elif variant in tests:
        tests[variant](target)
    else:
        print(f"Unknown variant: {variant}")
        print(f"Available: {', '.join(tests.keys())}")
