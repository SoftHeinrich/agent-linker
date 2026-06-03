#!/usr/bin/env python3
"""Checkpoint replay test for S-Linker9 variants (9a, 9b, 9c).

Tests:
- 9a (remove auto-approval): replays validation, counts affected candidates
- 9b (remove enrichment): replays from tier1 without enrichment, measures partial impact
- 9c: needs E2E (prompt change), but we can measure what enrichment contributes

Uses s_linker9 checkpoints — zero LLM calls.
"""

import csv
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.core.data_types import SadSamLink, CandidateLink
from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.pcm_parser import parse_pcm_repository

CACHE = "results/phase_cache/s_linker9"
BENCHMARK = "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
DATASETS = ['mediastore', 'teastore', 'teammates', 'bigbluebutton', 'jabref']

TEXT_PATHS = {
    'mediastore': os.path.join(BENCHMARK, 'mediastore/text_2016/mediastore.txt'),
    'teastore': os.path.join(BENCHMARK, 'teastore/text_2020/teastore.txt'),
    'teammates': os.path.join(BENCHMARK, 'teammates/text_2021/teammates.txt'),
    'bigbluebutton': os.path.join(BENCHMARK, 'bigbluebutton/text_2021/bigbluebutton.txt'),
    'jabref': os.path.join(BENCHMARK, 'jabref/text_2021/jabref.txt'),
}
MODEL_PATHS = {
    'mediastore': os.path.join(BENCHMARK, 'mediastore/model_2016/pcm/ms.repository'),
    'teastore': os.path.join(BENCHMARK, 'teastore/model_2020/pcm/teastore.repository'),
    'teammates': os.path.join(BENCHMARK, 'teammates/model_2021/pcm/teammates.repository'),
    'bigbluebutton': os.path.join(BENCHMARK, 'bigbluebutton/model_2021/pcm/bbb.repository'),
    'jabref': os.path.join(BENCHMARK, 'jabref/model_2021/pcm/jabref.repository'),
}
GOLD_PATHS = {
    'mediastore': os.path.join(BENCHMARK, 'mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv'),
    'teastore': os.path.join(BENCHMARK, 'teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv'),
    'teammates': os.path.join(BENCHMARK, 'teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv'),
    'bigbluebutton': os.path.join(BENCHMARK, 'bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv'),
    'jabref': os.path.join(BENCHMARK, 'jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv'),
}


def load_checkpoint(ds, phase):
    with open(os.path.join(CACHE, ds, f"{phase}.pkl"), "rb") as f:
        return pickle.load(f)


def load_gold(ds):
    """Load gold standard as set of (sentence_number, model_element_id)."""
    gold = set()
    with open(GOLD_PATHS[ds]) as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            snum = int(row['sentence'])
            eid = row['modelElementID']
            gold.add((snum, eid))
    return gold


def final_to_set(links):
    return {(l.sentence_number, l.component_id) for l in links}


def compute_f1(predicted, gold):
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1, tp, fp, fn


def has_standalone_mention(comp_name, text):
    """Replica of SLinker9._has_standalone_mention."""
    if not comp_name:
        return False
    is_single = ' ' not in comp_name
    if is_single:
        cap_name = comp_name[0].upper() + comp_name[1:]
        pattern = rf'\b{re.escape(cap_name)}\b'
        flags = 0
    else:
        pattern = rf'\b{re.escape(comp_name)}\b'
        flags = re.IGNORECASE
    for m in re.finditer(pattern, text, flags):
        s, e = m.start(), m.end()
        if s > 0 and text[s-1] == '.':
            continue
        if e < len(text) and text[e] == '.' and e + 1 < len(text) and text[e+1].isalpha():
            continue
        if s > 0 and text[s-1] == '-':
            continue
        if e < len(text) and text[e] == '-' and '-' not in comp_name:
            continue
        return True
    return False


def has_clean_mention(term, text):
    """Replica of SLinker9._has_clean_mention."""
    pattern = rf'\b{re.escape(term)}\b'
    for m in re.finditer(pattern, text, re.IGNORECASE):
        s, e = m.start(), m.end()
        if s > 0 and text[s-1] == '.':
            continue
        if e < len(text) and text[e] == '.' and e + 1 < len(text) and text[e+1].isalpha():
            continue
        if (s > 0 and text[s-1] == '-') or (e < len(text) and text[e] == '-'):
            continue
        return True
    return False


def word_boundary_match(name, text):
    return bool(re.search(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE))


def test_9a_auto_approval_impact():
    """9a: Identify which candidates are currently auto-approved vs would need LLM 2-pass."""
    print("=" * 70)
    print("VARIANT 9a: Remove auto-approval (U1+U2)")
    print("=" * 70)
    print()

    for ds in DATASETS:
        t1 = load_checkpoint(ds, "tier1")
        t15 = load_checkpoint(ds, "tier1_5")
        t2 = load_checkpoint(ds, "tier2")
        fn = load_checkpoint(ds, "final")
        gold = load_gold(ds)

        doc_knowledge = t15["doc_knowledge"]
        components = parse_pcm_repository(MODEL_PATHS[ds])
        sentences = DocumentLoader.load_sentences(TEXT_PATHS[ds])
        sent_map = DocumentLoader.build_sent_map(sentences)

        # Build alias_map (same as SLinker9 does)
        alias_map = {}
        for c in components:
            aliases = {c.name}
            for a, cn in doc_knowledge.abbreviations.items():
                if cn == c.name:
                    aliases.add(a)
            for s, cn in doc_knowledge.synonyms.items():
                if cn == c.name:
                    aliases.add(s)
            for p, cn in doc_knowledge.partial_references.items():
                if cn == c.name:
                    aliases.add(p)
            alias_map[c.name] = aliases

        # Replay: identify auto-approved among validated candidates
        auto_approved = []
        llm_would_decide = []
        for c in t2["validated"]:
            if c.source != "validated":
                continue  # entity source = direct, no auto-approval
            # Replay the word-boundary check
            sent = sent_map.get(c.sentence_number)
            if not sent:
                continue
            matched = False
            for a in alias_map.get(c.component_name, set()):
                if len(a) >= 3:
                    if a.lower() in sent.text.lower():
                        matched = True
                        break
                elif len(a) >= 2:
                    if word_boundary_match(a, sent.text):
                        matched = True
                        break
            if matched:
                auto_approved.append(c)
            else:
                llm_would_decide.append(c)

        # Classify auto-approved as TP/FP
        auto_tp = [(c.sentence_number, c.component_id) for c in auto_approved
                    if (c.sentence_number, c.component_id) in gold]
        auto_fp = [(c.sentence_number, c.component_id) for c in auto_approved
                    if (c.sentence_number, c.component_id) not in gold]

        print(f"--- {ds} ---")
        print(f"  Auto-approved: {len(auto_approved)} ({len(auto_tp)} TP, {len(auto_fp)} FP)")
        print(f"  LLM 2-pass would need: {len(llm_would_decide)}")

        if auto_fp:
            print(f"  Auto-approved FPs:")
            for snum, cid in auto_fp:
                cname = next((c.component_name for c in auto_approved
                             if c.sentence_number == snum and c.component_id == cid), "?")
                sent = sent_map.get(snum)
                print(f"    S{snum} -> {cname}: {sent.text[:80]}...")

        if auto_tp:
            print(f"  Auto-approved TPs ({len(auto_tp)}):")
            for snum, cid in auto_tp[:5]:  # show first 5
                cname = next((c.component_name for c in auto_approved
                             if c.sentence_number == snum and c.component_id == cid), "?")
                sent = sent_map.get(snum)
                print(f"    S{snum} -> {cname}: {sent.text[:80]}...")
            if len(auto_tp) > 5:
                print(f"    ... and {len(auto_tp) - 5} more TPs")

        # Simulate 9a: remove auto-approved from final, see F1 impact
        # In 9a, these go to LLM 2-pass instead. We can simulate worst-case (LLM rejects all)
        # and best-case (LLM approves all TPs, rejects all FPs)
        final_set = final_to_set(fn["final"])

        # Worst case: LLM rejects ALL auto-approved
        worst = final_set - {(c.sentence_number, c.component_id) for c in auto_approved}
        _, _, f1_worst, _, fp_w, fn_w = compute_f1(worst, gold)

        # Best case: LLM approves TPs, rejects FPs
        best = final_set - set(auto_fp)
        _, _, f1_best, _, fp_b, fn_b = compute_f1(best, gold)

        _, _, f1_current, _, fp_c, fn_c = compute_f1(final_set, gold)

        print(f"  F1 current: {f1_current:.1%} ({fp_c} FP, {fn_c} FN)")
        print(f"  F1 best-case (LLM approves TPs, rejects FPs): {f1_best:.1%} ({fp_b} FP, {fn_b} FN)")
        print(f"  F1 worst-case (LLM rejects all auto-approved): {f1_worst:.1%} ({fp_w} FP, {fn_w} FN)")
        print()


def test_9b_enrichment_impact():
    """9b: Measure impact of removing multiword partial enrichment."""
    print("=" * 70)
    print("VARIANT 9b: Remove enrichment (U4+U5 Option C)")
    print("=" * 70)
    print()

    for ds in DATASETS:
        t1 = load_checkpoint(ds, "tier1")
        t15 = load_checkpoint(ds, "tier1_5")
        t2 = load_checkpoint(ds, "tier2")
        fn = load_checkpoint(ds, "final")
        gold = load_gold(ds)

        # What enrichment added
        p_before = dict(t1["doc_knowledge"].partial_references)
        p_after = dict(t15["doc_knowledge"].partial_references)
        enriched = {k: v for k, v in p_after.items() if k not in p_before}

        print(f"--- {ds} ---")
        print(f"  LLM-discovered partials: {p_before}")
        print(f"  Enrichment-added partials: {enriched}")

        if not enriched:
            print(f"  -> No enrichment effect. 9b = 9 for this dataset.")
            _, _, f1_current, _, fp_c, fn_c = compute_f1(final_to_set(fn["final"]), gold)
            print(f"  F1: {f1_current:.1%} ({fp_c} FP, {fn_c} FN)")
            print()
            continue

        # Which partial-validated links came from enriched partials?
        enriched_partial_links = []
        non_enriched_partial_links = []
        for c in t2["partial_validated"]:
            if c.matched_text in enriched:
                enriched_partial_links.append(c)
            else:
                non_enriched_partial_links.append(c)

        # Also check: do enriched partials appear in entity extraction's alias list?
        # Enriched partials are fed to entity extraction via KNOWN ALIASES prompt
        # Without them, entity extraction might miss some candidates

        # Count final links that depend on enriched partials
        final_set = final_to_set(fn["final"])
        enriched_keys = {(c.sentence_number, c.component_id) for c in enriched_partial_links}
        enriched_tp = enriched_keys & gold
        enriched_fp = enriched_keys - gold

        print(f"  Partial-validated from enrichment: {len(enriched_partial_links)} "
              f"({len(enriched_tp)} TP, {len(enriched_fp)} FP)")

        for c in enriched_partial_links:
            is_tp = "TP" if (c.sentence_number, c.component_id) in gold else "FP"
            print(f"    [{is_tp}] S{c.sentence_number} -> {c.component_name} "
                  f"(matched: '{c.matched_text}')")

        # Also check: validated links that used enriched aliases in auto-approval
        sentences = DocumentLoader.load_sentences(TEXT_PATHS[ds])
        sent_map = DocumentLoader.build_sent_map(sentences)

        # Which validated links ONLY match because of enriched partials?
        components = parse_pcm_repository(MODEL_PATHS[ds])
        alias_map_without = {}
        for c in components:
            aliases = {c.name}
            for a, cn in t1["doc_knowledge"].abbreviations.items():
                if cn == c.name:
                    aliases.add(a)
            for s, cn in t1["doc_knowledge"].synonyms.items():
                if cn == c.name:
                    aliases.add(s)
            for p, cn in t1["doc_knowledge"].partial_references.items():
                if cn == c.name:
                    aliases.add(p)
            alias_map_without[c.name] = aliases

        alias_map_with = {}
        for c in components:
            aliases = {c.name}
            for a, cn in t15["doc_knowledge"].abbreviations.items():
                if cn == c.name:
                    aliases.add(a)
            for s, cn in t15["doc_knowledge"].synonyms.items():
                if cn == c.name:
                    aliases.add(s)
            for p, cn in t15["doc_knowledge"].partial_references.items():
                if cn == c.name:
                    aliases.add(p)
            alias_map_with[c.name] = aliases

        # Validated links that auto-approved ONLY because of enriched alias
        enrichment_dependent_validated = []
        for c in t2["validated"]:
            if c.source != "validated":
                continue
            sent = sent_map.get(c.sentence_number)
            if not sent:
                continue

            # Would match WITHOUT enrichment?
            match_without = False
            for a in alias_map_without.get(c.component_name, set()):
                if len(a) >= 3:
                    if a.lower() in sent.text.lower():
                        match_without = True
                        break
                elif len(a) >= 2:
                    if word_boundary_match(a, sent.text):
                        match_without = True
                        break

            # Would match WITH enrichment?
            match_with = False
            for a in alias_map_with.get(c.component_name, set()):
                if len(a) >= 3:
                    if a.lower() in sent.text.lower():
                        match_with = True
                        break
                elif len(a) >= 2:
                    if word_boundary_match(a, sent.text):
                        match_with = True
                        break

            if match_with and not match_without:
                enrichment_dependent_validated.append(c)

        dep_tp = [(c.sentence_number, c.component_id) for c in enrichment_dependent_validated
                   if (c.sentence_number, c.component_id) in gold]
        dep_fp = [(c.sentence_number, c.component_id) for c in enrichment_dependent_validated
                   if (c.sentence_number, c.component_id) not in gold]

        if enrichment_dependent_validated:
            print(f"  Validated links dependent on enrichment: {len(enrichment_dependent_validated)} "
                  f"({len(dep_tp)} TP, {len(dep_fp)} FP)")
            for c in enrichment_dependent_validated:
                is_tp = "TP" if (c.sentence_number, c.component_id) in gold else "FP"
                sent = sent_map.get(c.sentence_number)
                print(f"    [{is_tp}] S{c.sentence_number} -> {c.component_name}: "
                      f"{sent.text[:80]}...")

        # Simulate removing all enrichment-dependent links
        removed = enriched_keys | {(c.sentence_number, c.component_id)
                                    for c in enrichment_dependent_validated}
        simulated = final_set - removed
        _, _, f1_sim, _, fp_s, fn_s = compute_f1(simulated, gold)
        _, _, f1_current, _, fp_c, fn_c = compute_f1(final_set, gold)

        total_tp_lost = len([k for k in removed if k in gold])
        total_fp_lost = len([k for k in removed if k not in gold])

        print(f"  F1 current: {f1_current:.1%} ({fp_c} FP, {fn_c} FN)")
        print(f"  F1 without enrichment (simulated): {f1_sim:.1%} ({fp_s} FP, {fn_s} FN)")
        print(f"  Net: {total_tp_lost} TP lost, {total_fp_lost} FP removed")
        print()


def test_9b_partial_injection_without_enrichment():
    """9b extra: What would partial injection produce with only LLM-discovered partials?"""
    print("=" * 70)
    print("VARIANT 9b DETAIL: Partial injection with LLM-only partials")
    print("=" * 70)
    print()

    for ds in DATASETS:
        t1 = load_checkpoint(ds, "tier1")
        t2 = load_checkpoint(ds, "tier2")
        fn = load_checkpoint(ds, "final")
        gold = load_gold(ds)

        llm_partials = dict(t1["doc_knowledge"].partial_references)
        if not llm_partials:
            print(f"--- {ds}: No LLM-discovered partials. Partial injection produces nothing. ---")
            continue

        components = parse_pcm_repository(MODEL_PATHS[ds])
        name_to_id = {c.name: c.id for c in components}
        sentences = DocumentLoader.load_sentences(TEXT_PATHS[ds])

        # Simulate partial injection with LLM-only partials
        seed_set = t1["seed_set"]
        validated_set = {(c.sentence_number, c.component_id) for c in t2["validated"]}
        coref_set = {(l.sentence_number, l.component_id) for l in t2["coref_links"]}
        existing = seed_set | validated_set | coref_set

        candidates = []
        for partial, comp_name in llm_partials.items():
            if comp_name not in name_to_id:
                continue
            comp_id = name_to_id[comp_name]
            for sent in sentences:
                key = (sent.number, comp_id)
                if key in existing:
                    continue
                if has_clean_mention(partial, sent.text):
                    is_tp = "TP" if key in gold else "FP"
                    candidates.append((sent.number, comp_id, comp_name, partial, is_tp))
                    existing.add(key)

        tp_count = sum(1 for c in candidates if c[4] == "TP")
        fp_count = sum(1 for c in candidates if c[4] == "FP")

        print(f"--- {ds} ---")
        print(f"  LLM partials: {llm_partials}")
        print(f"  Partial injection candidates: {len(candidates)} ({tp_count} TP, {fp_count} FP)")
        print(f"  These would then go through 2-pass validation (not simulated).")
        for c in candidates[:10]:
            print(f"    [{c[4]}] S{c[0]} -> {c[2]} (partial: '{c[3]}')")
        if len(candidates) > 10:
            print(f"    ... and {len(candidates) - 10} more")
        print()


def print_summary():
    """Print current S-Linker9 baseline for reference."""
    print("=" * 70)
    print("BASELINE: S-Linker9 current results")
    print("=" * 70)

    f1s = []
    for ds in DATASETS:
        fn = load_checkpoint(ds, "final")
        gold = load_gold(ds)
        final_set = final_to_set(fn["final"])
        p, r, f1, tp, fp, fn_count = compute_f1(final_set, gold)
        f1s.append(f1)
        print(f"  {ds:15s}: P={p:.1%} R={r:.1%} F1={f1:.1%} (TP={tp} FP={fp} FN={fn_count})")

    print(f"  {'MACRO F1':15s}: {sum(f1s)/len(f1s):.1%}")
    print()


if __name__ == "__main__":
    print_summary()
    test_9a_auto_approval_impact()
    test_9b_enrichment_impact()
    test_9b_partial_injection_without_enrichment()
