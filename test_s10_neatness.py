#!/usr/bin/env python3
"""S-Linker10 neatness tests: seed validation, needs_validation bypass, generic-mention.

Offline tests use checkpoints (zero LLM calls).
LLM tests require CLAUDE_MODEL=sonnet or OPENAI backend.

Usage:
    # Offline analysis (no LLM):
    python test_s10_neatness.py offline

    # LLM: validate seed links through evidence-stratified voting:
    python test_s10_neatness.py seed_val [dataset]

    # LLM: validate bypass candidates (needs_validation=False):
    python test_s10_neatness.py bypass_val [dataset]

    # LLM: merged generic-mention into validation prompt (no separate pre-check):
    python test_s10_neatness.py merged_generic [dataset]
"""

import csv
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load .env
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

BENCHMARK_BASE = Path("/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark")
CHECKPOINT_BASE = Path("results/phase_cache/s_linker10")

DATASETS = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
        "gold": BENCHMARK_BASE / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
        "model": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
        "gold": BENCHMARK_BASE / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
        "gold": BENCHMARK_BASE / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
        "gold": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
        "gold": BENCHMARK_BASE / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
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


def load_checkpoint(ds_name, phase):
    path = CHECKPOINT_BASE / ds_name / f"{phase}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def eval_metrics(predicted, gold):
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"tp": tp, "fp": fp, "fn": fn, "P": p, "R": r, "F1": f1}


# ═══════════════════════════════════════════════════════════════════════════════
# OFFLINE ANALYSIS (zero LLM calls)
# ═══════════════════════════════════════════════════════════════════════════════

def offline_analysis():
    """Analyze checkpoints to understand seed FPs, bypass candidates, and validation paths."""
    print("=" * 80)
    print("OFFLINE NEATNESS ANALYSIS — S-Linker10 Checkpoints")
    print("=" * 80)

    all_seed_fps = []
    all_bypass = []

    for ds_name, paths in DATASETS.items():
        gold = load_gold(paths["gold"])
        tier1 = load_checkpoint(ds_name, "tier1")
        tier2 = load_checkpoint(ds_name, "tier2")
        final_data = load_checkpoint(ds_name, "final")

        seed_links = tier1["seed_links"]
        seed_set = tier1["seed_set"]
        validated = tier2["validated"]
        coref_links = tier2["coref_links"]
        partial_validated = tier2.get("partial_validated", [])
        final_links = final_data["final"]

        # Load sentence text for context
        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        sent_map = {s.number: s for s in sentences}
        components = parse_pcm_repository(str(paths["model"]))
        id_to_name = {c.id: c.name for c in components}

        print(f"\n{'=' * 60}")
        print(f"  {ds_name.upper()}")
        print(f"{'=' * 60}")

        # --- Seed FP analysis ---
        seed_pairs = {(l.sentence_number, l.component_id) for l in seed_links}
        seed_fps = seed_pairs - gold
        seed_tps = seed_pairs & gold

        print(f"\n  Seed: {len(seed_links)} links ({len(seed_tps)} TP, {len(seed_fps)} FP)")

        # Check: would entity pipeline have caught seed FPs?
        entity_pairs = {(c.sentence_number, c.component_id) for c in validated}
        for snum, cid in sorted(seed_fps):
            cname = id_to_name.get(cid, "?")
            sent = sent_map.get(snum)
            text = sent.text[:80] if sent else "?"
            in_entity = (snum, cid) in entity_pairs
            source_in_entity = None
            if in_entity:
                for c in validated:
                    if c.sentence_number == snum and c.component_id == cid:
                        source_in_entity = f"validated(needs_val={c.needs_validation})"
                        break
            print(f"    SEED FP: S{snum} -> {cname} | entity_also={source_in_entity or 'NO'}")
            print(f"             \"{text}\"")
            all_seed_fps.append({
                "ds": ds_name, "snum": snum, "cid": cid, "cname": cname,
                "also_in_entity": in_entity,
            })

        # --- needs_validation bypass analysis ---
        bypass = [c for c in validated if not c.needs_validation]
        needs_val = [c for c in validated if c.needs_validation]
        bypass_pairs = {(c.sentence_number, c.component_id) for c in bypass}
        bypass_fps = bypass_pairs - gold

        print(f"\n  Entity validated: {len(validated)} ({len(bypass)} bypass, {len(needs_val)} LLM-validated)")
        print(f"  Bypass FPs: {len(bypass_fps)}")
        for snum, cid in sorted(bypass_fps):
            cname = id_to_name.get(cid, "?")
            sent = sent_map.get(snum)
            text = sent.text[:80] if sent else "?"
            c = next((c for c in bypass if c.sentence_number == snum and c.component_id == cid), None)
            matched = c.matched_text if c else "?"
            print(f"    BYPASS FP: S{snum} -> {cname} matched=\"{matched}\" | \"{text}\"")
            all_bypass.append({"ds": ds_name, "snum": snum, "cname": cname, "matched": matched})

        # --- Final metrics ---
        final_pairs = {(l.sentence_number, l.component_id) for l in final_links}
        m = eval_metrics(final_pairs, gold)
        print(f"\n  Final: P={m['P']:.1%} R={m['R']:.1%} F1={m['F1']:.1%} (TP={m['tp']} FP={m['fp']} FN={m['fn']})")

        # --- Simulation: what if seed went through validation? ---
        # Worst case: all seed FPs removed, all seed TPs survive
        # Best case: all seed FPs removed
        sim_no_seed_fp = final_pairs - seed_fps
        m_sim = eval_metrics(sim_no_seed_fp, gold)
        print(f"  Sim (seed FPs removed): P={m_sim['P']:.1%} R={m_sim['R']:.1%} F1={m_sim['F1']:.1%}")

        # Check: are any seed TPs NOT also found by entity/coref/partial?
        other_pairs = entity_pairs | {(l.sentence_number, l.component_id) for l in coref_links} | \
                      {(c.sentence_number, c.component_id) for c in partial_validated}
        seed_only_tps = seed_tps - other_pairs
        print(f"  Seed-only TPs (not found by other strategies): {len(seed_only_tps)}")
        for snum, cid in sorted(seed_only_tps):
            cname = id_to_name.get(cid, "?")
            print(f"    S{snum} -> {cname}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total seed FPs across all datasets: {len(all_seed_fps)}")
    also = sum(1 for f in all_seed_fps if f["also_in_entity"])
    print(f"    Also found by entity pipeline: {also}")
    print(f"    Seed-only FPs (validation could catch): {len(all_seed_fps) - also}")
    print(f"  Total bypass FPs: {len(all_bypass)}")


# ═══════════════════════════════════════════════════════════════════════════════
# LLM TEST: Route seed links through validation
# ═══════════════════════════════════════════════════════════════════════════════

def test_seed_validation(target_datasets=None):
    """Route seed links through _validate_intersect, measure FP reduction."""
    from llm_sad_sam.linkers.experimental.s_linker10 import SLinker10

    linker = SLinker10()
    results = {}

    for ds_name, paths in DATASETS.items():
        if target_datasets and ds_name not in target_datasets:
            continue

        gold = load_gold(paths["gold"])
        tier1 = load_checkpoint(ds_name, "tier1")
        tier2 = load_checkpoint(ds_name, "tier2")

        linker.model_knowledge = tier1["model_knowledge"]
        linker.doc_knowledge = tier1["doc_knowledge"]
        linker._generic_partials = tier1["generic_partials"]
        linker._components = parse_pcm_repository(str(paths["model"]))

        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        sent_map = DocumentLoader.build_sent_map(sentences)

        seed_links = tier1["seed_links"]
        seed_set = tier1["seed_set"]

        # Convert seed links to CandidateLink for validation
        name_to_id = {c.name: c.id for c in linker._components}
        seed_candidates = []
        for sl in seed_links:
            sent = sent_map.get(sl.sentence_number)
            if not sent:
                continue
            # Find matched text (the component name or closest match in sentence)
            matched = sl.component_name
            seed_candidates.append(CandidateLink(
                sentence_number=sl.sentence_number,
                sentence_text=sent.text,
                component_name=sl.component_name,
                component_id=sl.component_id,
                matched_text=matched,
                confidence=0.0,
                source="seed",
                match_type="exact",
                needs_validation=True,  # Force ALL through validation
            ))

        print(f"\n{'=' * 60}")
        print(f"  SEED VALIDATION: {ds_name} ({len(seed_candidates)} seed links)")
        print(f"{'=' * 60}")

        # Run validation
        t0 = time.time()
        validated = linker._validate_intersect(seed_candidates, linker._components, sent_map)
        elapsed = time.time() - t0

        # Compare
        original_pairs = {(sl.sentence_number, sl.component_id) for sl in seed_links}
        validated_pairs = {(c.sentence_number, c.component_id) for c in validated}
        killed = original_pairs - validated_pairs

        killed_tp = killed & gold
        killed_fp = killed - gold

        print(f"  Validated: {len(validated)} / {len(seed_candidates)} ({elapsed:.0f}s)")
        print(f"  Killed: {len(killed)} (FP: {len(killed_fp)}, TP: {len(killed_tp)})")

        # Full pipeline simulation with validated seed
        coref_links = tier2["coref_links"]
        entity_validated = tier2["validated"]
        partial_validated = tier2.get("partial_validated", [])

        # Rebuild final with validated seed instead of raw seed
        validated_seed = [SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, "seed")
                         for c in validated]
        entity_links = [SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
                       for c in entity_validated]
        partial_links = [SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, "partial_inject")
                        for c in partial_validated]

        all_links = validated_seed + entity_links + coref_links + partial_links
        seen = set()
        final = []
        for lk in all_links:
            key = (lk.sentence_number, lk.component_id)
            if key not in seen:
                seen.add(key)
                final.append(lk)

        final_pairs = {(l.sentence_number, l.component_id) for l in final}
        m = eval_metrics(final_pairs, gold)
        print(f"  Simulated final: P={m['P']:.1%} R={m['R']:.1%} F1={m['F1']:.1%} (TP={m['tp']} FP={m['fp']} FN={m['fn']})")

        # Compare to current
        orig_final = load_checkpoint(ds_name, "final")["final"]
        orig_pairs = {(l.sentence_number, l.component_id) for l in orig_final}
        m_orig = eval_metrics(orig_pairs, gold)
        delta = m["F1"] - m_orig["F1"]
        print(f"  Current final:   P={m_orig['P']:.1%} R={m_orig['R']:.1%} F1={m_orig['F1']:.1%}")
        print(f"  Delta F1: {delta:+.1%} ({m['fp']-m_orig['fp']:+d} FP, {m['fn']-m_orig['fn']:+d} FN)")

        results[ds_name] = {"before": m_orig, "after": m, "killed_fp": len(killed_fp), "killed_tp": len(killed_tp)}

    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print(f"  MACRO SUMMARY")
        print(f"{'=' * 60}")
        before_f1 = sum(r["before"]["F1"] for r in results.values()) / len(results)
        after_f1 = sum(r["after"]["F1"] for r in results.values()) / len(results)
        total_killed_fp = sum(r["killed_fp"] for r in results.values())
        total_killed_tp = sum(r["killed_tp"] for r in results.values())
        print(f"  Before: {before_f1:.1%} macro F1")
        print(f"  After:  {after_f1:.1%} macro F1 ({after_f1-before_f1:+.1%})")
        print(f"  Total killed: {total_killed_fp} FP, {total_killed_tp} TP")


# ═══════════════════════════════════════════════════════════════════════════════
# LLM TEST: Validate bypass candidates
# ═══════════════════════════════════════════════════════════════════════════════

def test_bypass_validation(target_datasets=None):
    """Route needs_validation=False candidates through validation, check for hidden FPs."""
    from llm_sad_sam.linkers.experimental.s_linker10 import SLinker10

    linker = SLinker10()

    for ds_name, paths in DATASETS.items():
        if target_datasets and ds_name not in target_datasets:
            continue

        gold = load_gold(paths["gold"])
        tier1 = load_checkpoint(ds_name, "tier1")
        tier2 = load_checkpoint(ds_name, "tier2")

        linker.model_knowledge = tier1["model_knowledge"]
        linker.doc_knowledge = tier1["doc_knowledge"]
        linker._generic_partials = tier1["generic_partials"]
        linker._components = parse_pcm_repository(str(paths["model"]))

        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        sent_map = DocumentLoader.build_sent_map(sentences)

        validated = tier2["validated"]
        bypass = [c for c in validated if not c.needs_validation]

        if not bypass:
            print(f"\n  {ds_name}: no bypass candidates, skipping")
            continue

        # Force all bypass candidates through validation
        for c in bypass:
            c.needs_validation = True

        print(f"\n{'=' * 60}")
        print(f"  BYPASS VALIDATION: {ds_name} ({len(bypass)} bypass candidates)")
        print(f"{'=' * 60}")

        t0 = time.time()
        revalidated = linker._validate_intersect(bypass, linker._components, sent_map)
        elapsed = time.time() - t0

        original_pairs = {(c.sentence_number, c.component_id) for c in bypass}
        revalidated_pairs = {(c.sentence_number, c.component_id) for c in revalidated}
        killed = original_pairs - revalidated_pairs

        killed_tp = killed & gold
        killed_fp = killed - gold

        print(f"  Revalidated: {len(revalidated)} / {len(bypass)} ({elapsed:.0f}s)")
        print(f"  Killed: {len(killed)} (FP: {len(killed_fp)}, TP: {len(killed_tp)})")

        id_to_name = {c.id: c.name for c in linker._components}
        for snum, cid in sorted(killed):
            is_tp = (snum, cid) in gold
            cname = id_to_name.get(cid, "?")
            sent = sent_map.get(snum)
            label = "TP KILLED" if is_tp else "FP CAUGHT"
            print(f"    {label}: S{snum} -> {cname} | \"{sent.text[:70]}\"")


# ═══════════════════════════════════════════════════════════════════════════════
# LLM TEST: Merged generic-mention detection (no separate pre-check)
# ═══════════════════════════════════════════════════════════════════════════════

def test_merged_generic(target_datasets=None):
    """Test variant: fold generic-mention detection INTO validation prompt.

    Instead of 3 steps (generic LLM check → pass 1 → pass 2),
    this variant does 2 steps: pass 1 → pass 2, where the prompt includes
    instructions to reject generic usage.
    """
    from llm_sad_sam.linkers.experimental.s_linker10 import SLinker10
    from llm_sad_sam.linkers.experimental.prompts_v2 import VALIDATION_RULES

    linker = SLinker10()

    for ds_name, paths in DATASETS.items():
        if target_datasets and ds_name not in target_datasets:
            continue

        gold = load_gold(paths["gold"])
        tier1 = load_checkpoint(ds_name, "tier1")
        tier2 = load_checkpoint(ds_name, "tier2")

        linker.model_knowledge = tier1["model_knowledge"]
        linker.doc_knowledge = tier1["doc_knowledge"]
        linker._generic_partials = tier1["generic_partials"]
        linker._components = parse_pcm_repository(str(paths["model"]))

        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        sent_map = DocumentLoader.build_sent_map(sentences)
        comp_names = [c.name for c in linker._components]

        # Get all candidates that need validation (both generic and non-generic)
        validated_all = tier2["validated"]
        needs = [c for c in validated_all if c.needs_validation]

        if not needs:
            print(f"\n  {ds_name}: no needs_validation candidates, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"  MERGED GENERIC: {ds_name} ({len(needs)} candidates)")
        print(f"{'=' * 60}")

        # Build alias lookup
        alias_map = {}
        for c in linker._components:
            aliases = {}
            if linker.doc_knowledge:
                for a, cn in linker.doc_knowledge.abbreviations.items():
                    if cn == c.name:
                        aliases[a] = "abbreviation"
                for s, cn in linker.doc_knowledge.synonyms.items():
                    if cn == c.name:
                        aliases[s] = "synonym"
                for p, cn in linker.doc_knowledge.partial_references.items():
                    if cn == c.name:
                        aliases[p] = "partial reference"
            alias_map[c.name] = aliases

        # Run 2-pass with merged generic detection in prompt
        t0 = time.time()
        approved = []
        for batch_start in range(0, len(needs), 25):
            batch = needs[batch_start:batch_start + 25]
            cases = []
            has_alias = []
            for i, c in enumerate(batch):
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
                # Add generic warning for ambiguous names appearing in lowercase
                generic_hint = ""
                if linker._is_ambiguous_name_component(c.component_name):
                    sent = sent_map.get(c.sentence_number)
                    if sent and not linker._has_standalone_mention(c.component_name, sent.text):
                        generic_hint = f'\n  [CAUTION: "{c.component_name}" is an ambiguous name — check if lowercase "{c.matched_text}" refers to the component or is generic English]'
                has_alias.append(bool(alias_hint))
                cases.append(f'Case {i+1}: "{c.matched_text}" -> {c.component_name}{alias_hint}{generic_hint}\n  {p}"{c.sentence_text}"')

            # Enhanced validation rules with generic detection folded in
            merged_rules = VALIDATION_RULES + """

GENERIC USAGE DETECTION:
- If the component name is flagged as [CAUTION: ambiguous], pay special attention
- A lowercase word that matches a component name is GENERIC if it describes a type of
  activity, quality, or concept rather than naming the specific system entity
  (e.g., "provides storage access" = generic vs "the Storage handles persistence" = component)
- When in doubt about ambiguous names, REJECT — false positives are costlier here"""

            r1 = _merged_validation_pass(linker, comp_names, cases, alias_map,
                "Focus on ACTOR role: is the component performing an action or being described?",
                merged_rules)
            r2 = _merged_validation_pass(linker, comp_names, cases, alias_map,
                "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component?",
                merged_rules)

            for i, c in enumerate(batch):
                p1 = r1.get(i, False)
                p2 = r2.get(i, False)
                if (p1 or p2) if has_alias[i] else (p1 and p2):
                    approved.append(c)

        elapsed = time.time() - t0

        # Compare with current (which has separate generic pre-check)
        current_pairs = {(c.sentence_number, c.component_id) for c in validated_all}
        merged_bypass = [c for c in validated_all if not c.needs_validation]
        merged_pairs = {(c.sentence_number, c.component_id) for c in merged_bypass + approved}

        current_m = eval_metrics(current_pairs, gold)
        merged_m = eval_metrics(merged_pairs, gold)

        print(f"  Current (3-step): P={current_m['P']:.1%} R={current_m['R']:.1%} F1={current_m['F1']:.1%} ({elapsed:.0f}s)")
        print(f"  Merged (2-step):  P={merged_m['P']:.1%} R={merged_m['R']:.1%} F1={merged_m['F1']:.1%}")
        delta = merged_m["F1"] - current_m["F1"]
        print(f"  Delta: {delta:+.1%} F1 ({merged_m['fp']-current_m['fp']:+d} FP, {merged_m['fn']-current_m['fn']:+d} FN)")


def _merged_validation_pass(linker, comp_names, cases, alias_map, focus, rules):
    """Validation pass with merged generic detection."""
    has_alias = any("[KNOWN ALIAS:" in c for c in cases)
    alias_rule = ""
    if has_alias:
        alias_rule = ("\n- When a KNOWN ALIAS is indicated, the word IS a reference to that component "
                      "unless the sentence clearly uses it in an unrelated sense")

    prompt = f"""Validate component references in a software architecture document. {focus}

COMPONENTS: {', '.join(comp_names)}

{rules}{alias_rule}

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

    data = linker.llm.extract_json(linker.llm.query(prompt, timeout=120))
    results = {}
    if data:
        for v in data.get("validations", []):
            idx = v.get("case", 0) - 1
            if 0 <= idx < len(cases):
                results[idx] = v.get("approve", False)
    return results


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "offline"
    target = sys.argv[2:] if len(sys.argv) > 2 else None

    if mode == "offline":
        offline_analysis()
    elif mode == "seed_val":
        test_seed_validation(target)
    elif mode == "bypass_val":
        test_bypass_validation(target)
    elif mode == "merged_generic":
        test_merged_generic(target)
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python test_s10_neatness.py [offline|seed_val|bypass_val|merged_generic] [datasets...]")
