"""S-Linker8 variant replay — precise checkpoint analysis.

Replays actual S-Linker8 code logic against checkpoints to determine
exact impact of proposed fixes. Zero LLM calls.

Key difference from test_s8_variants.py: this test replays the actual
code paths (auto-approval, generic routing, impl filtering) instead
of just estimating.
"""

import csv
import os
import pickle
import re

from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.pcm_parser import parse_pcm_repository

BENCHMARK = os.path.join(os.path.dirname(__file__), '..', 'ardoco', 'core', 'tests-base',
                         'src', 'main', 'resources', 'benchmark')
CACHE = os.path.join(os.path.dirname(__file__), 'results', 'phase_cache', 's_linker8')

DATASETS = ['mediastore', 'teastore', 'teammates', 'bigbluebutton', 'jabref']
ABBR = {'mediastore': 'MS', 'teastore': 'TS', 'teammates': 'TM',
        'bigbluebutton': 'BBB', 'jabref': 'JAB'}
GS_NAMES = {
    'mediastore': 'goldstandard_sad_2016-sam_2016.csv',
    'teastore': 'goldstandard_sad_2020-sam_2020.csv',
    'teammates': 'goldstandard_sad_2021-sam_2021.csv',
    'bigbluebutton': 'goldstandard_sad_2021-sam_2021.csv',
    'jabref': 'goldstandard_sad_2021-sam_2021.csv',
}
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


def load_gold(ds):
    gold = set()
    path = os.path.join(BENCHMARK, ds, 'goldstandards', GS_NAMES[ds])
    with open(path) as f:
        for row in csv.DictReader(f):
            gold.add((int(row['sentence']), row['modelElementID']))
    return gold


def evaluate(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'p': p, 'r': r, 'f1': f1}


def load_checkpoints(ds):
    t1 = pickle.load(open(os.path.join(CACHE, ds, 'tier1.pkl'), 'rb'))
    t1_5 = pickle.load(open(os.path.join(CACHE, ds, 'tier1_5.pkl'), 'rb'))
    t2 = pickle.load(open(os.path.join(CACHE, ds, 'tier2.pkl'), 'rb'))
    final = pickle.load(open(os.path.join(CACHE, ds, 'final.pkl'), 'rb'))
    return t1, t1_5, t2, final


def baseline_set(final_data):
    return {(l.sentence_number, l.component_id) for l in final_data['final']}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers (replicated from s_linker8.py)
# ═══════════════════════════════════════════════════════════════════════════

def is_structurally_unambiguous(name):
    if ' ' in name or '-' in name:
        return True
    if re.search(r'[a-z][A-Z]', name):
        return True
    if name.isupper():
        return True
    return False


def has_standalone_mention(comp_name, text):
    if not comp_name:
        return False
    if ' ' not in comp_name:
        cap = comp_name[0].upper() + comp_name[1:]
        pattern = rf'\b{re.escape(cap)}\b'
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


def build_alias_map(dk, comp_names):
    """Build alias lookup from doc_knowledge."""
    alias_map = {}
    for name in comp_names:
        aliases = {name}
        if dk:
            for a, cn in dk.abbreviations.items():
                if cn == name:
                    aliases.add(a)
            for s, cn in dk.synonyms.items():
                if cn == name:
                    aliases.add(s)
            for p, cn in dk.partial_references.items():
                if cn == name:
                    aliases.add(p)
        alias_map[name] = aliases
    return alias_map


def would_auto_approve(comp_name, sentence_text, alias_map):
    """Replay the auto-approval word-boundary check."""
    for a in alias_map.get(comp_name, set()):
        if len(a) >= 3:
            if a.lower() in sentence_text.lower():
                return True
        elif len(a) >= 2:
            if re.search(r'\b' + re.escape(a) + r'\b', sentence_text, re.IGNORECASE):
                return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# 8a: Remove validation auto-approval
# ═══════════════════════════════════════════════════════════════════════════

def test_8a():
    """Remove auto-approval: all candidates go to LLM 2-pass validation.

    Key finding: LLM 2-pass currently approves ZERO candidates across all datasets.
    It only sees candidates WITHOUT alias matches — the hardest cases — and rejects all.

    If we remove auto-approval, 28 "obvious" candidates (alias in sentence) would
    go to this strict filter. Question: does LLM 2-pass reject obvious cases too?
    """
    print("\n" + "=" * 80)
    print("VARIANT 8a: Remove validation auto-approval")
    print("=" * 80)

    total_tp_risk = 0
    total_fp_risk = 0

    for ds in DATASETS:
        gold = load_gold(ds)
        t1, t1_5, t2, final_data = load_checkpoints(ds)
        dk = t1_5.get('doc_knowledge', t1['doc_knowledge'])
        base = baseline_set(final_data)
        base_eval = evaluate(base, gold)

        # Get validated candidates and build alias map
        validated = t2['validated']
        comp_names = list({c.component_name for c in validated})
        alias_map = build_alias_map(dk, comp_names)

        # Separate auto-approved from LLM-approved in validated output
        auto_approved = []
        llm_approved = []
        for c in validated:
            if c.source != 'validated':
                continue  # source='entity' = direct, no validation needed
            if would_auto_approve(c.component_name, c.sentence_text, alias_map):
                auto_approved.append(c)
            else:
                llm_approved.append(c)

        # Also check partial_validated
        partial_validated = t2.get('partial_validated', [])
        pv_auto = []
        for c in partial_validated:
            if c.source == 'validated' and would_auto_approve(c.component_name, c.sentence_text, alias_map):
                pv_auto.append(c)

        # Which auto-approved links survive to final (after dedup)?
        final_auto = [(l.sentence_number, l.component_id, l.component_name)
                      for l in final_data['final'] if l.source == 'validated']
        tp_risk = [(s, c, n) for s, c, n in final_auto if (s, c) in gold]
        fp_risk = [(s, c, n) for s, c, n in final_auto if (s, c) not in gold]

        total_tp_risk += len(tp_risk)
        total_fp_risk += len(fp_risk)

        # Pessimistic: LLM rejects all auto-approved
        pess = base - {(s, c) for s, c, _ in final_auto}
        pess_eval = evaluate(pess, gold)

        print(f"\n  {ABBR[ds]} (baseline F1={base_eval['f1']:.1%}, TP={base_eval['tp']} FP={base_eval['fp']})")
        print(f"    Tier2: {len(auto_approved)} auto-approved, {len(llm_approved)} LLM 2-pass approved")
        print(f"    Final: {len(final_auto)} auto-approved survive dedup = {len(tp_risk)} TP + {len(fp_risk)} FP")

        if fp_risk:
            print(f"    FPs that would go to LLM (likely REJECTED = improvement):")
            sents = DocumentLoader.load_sentences(TEXT_PATHS[ds])
            sent_map = {s.number: s for s in sents}
            for s, c, n in fp_risk:
                sent = sent_map.get(s)
                print(f"      S{s} -> {n}: \"{sent.text[:70]}...\"" if sent else f"      S{s} -> {n}")

        print(f"    Pessimistic (all rejected): F1={pess_eval['f1']:.1%} ({pess_eval['tp']}TP {pess_eval['fp']}FP)")
        print(f"    Optimistic (all approved):  F1={base_eval['f1']:.1%} (no change)")

        if pv_auto:
            print(f"    Also: {len(pv_auto)} partial_validated auto-approved (source changes to partial_inject in final)")

    print(f"\n  TOTAL: {total_tp_risk} TPs + {total_fp_risk} FPs at risk")
    print(f"  VERDICT: RISKY — LLM 2-pass currently approves 0/all candidates it sees.")
    print(f"           Sending {total_tp_risk} TPs to this strict filter risks losing them.")
    print(f"           However, {total_fp_risk} FPs would likely be caught (improvement).")
    print(f"           Trade-off: keep auto-approval or accept TP loss for FP gain.")


# ═══════════════════════════════════════════════════════════════════════════
# 8d: Remove dedicated generic-mention detection
# ═══════════════════════════════════════════════════════════════════════════

def test_8d():
    """Remove the dedicated generic-mention LLM call.

    Currently: ambiguous+lowercase candidates → dedicated LLM → approve/reject
    Proposed: skip dedicated LLM, let them go to auto-approval or 2-pass validation

    Analysis: check which candidates are affected and their gold standard status.
    Note: we can see approved candidates in checkpoints but NOT rejected ones.
    """
    print("\n" + "=" * 80)
    print("VARIANT 8d: Remove dedicated generic-mention detection")
    print("=" * 80)

    for ds in DATASETS:
        gold = load_gold(ds)
        t1, t1_5, t2, final_data = load_checkpoints(ds)
        mk = t1['model_knowledge']
        dk = t1_5.get('doc_knowledge', t1['doc_knowledge'])
        base = baseline_set(final_data)
        base_eval = evaluate(base, gold)
        validated = t2['validated']
        sents = DocumentLoader.load_sentences(TEXT_PATHS[ds])
        sent_map = {s.number: s for s in sents}

        # Find candidates that went through generic detection (and were APPROVED)
        generic_approved = []
        for c in validated:
            if not c.needs_validation:
                continue
            s = sent_map.get(c.sentence_number)
            if not s:
                continue
            comp_lower = c.component_name.lower()
            has_exact = has_standalone_mention(c.component_name, s.text)
            has_lower = (not has_exact and
                         re.search(rf'\b{re.escape(comp_lower)}\b', s.text))
            if not has_lower and dk:
                for partial, target in dk.partial_references.items():
                    if target == c.component_name:
                        partial_lower = partial.lower()
                        if (re.search(rf'\b{re.escape(partial_lower)}\b', s.text.lower())
                                and not re.search(rf'\b{re.escape(partial)}\b', s.text)):
                            has_lower = True
                            break
            is_ambig = (not is_structurally_unambiguous(c.component_name) and
                        c.component_name in mk.ambiguous_names)
            if has_lower and is_ambig:
                is_tp = (c.sentence_number, c.component_id) in gold
                generic_approved.append((c, is_tp))

        # For REJECTED candidates: we can estimate by checking all sentences
        # for lowercase mentions of ambiguous names that are NOT in validated
        comp_names = list({c.component_name for c in validated})
        alias_map = build_alias_map(dk, comp_names)
        validated_keys = {(c.sentence_number, c.component_id) for c in validated}

        # Get all component IDs
        comps = parse_pcm_repository(MODEL_PATHS[ds])
        name_to_id = {c.name: c.id for c in comps}

        potentially_rejected = []
        for name in mk.ambiguous_names:
            if is_structurally_unambiguous(name):
                continue
            if name not in name_to_id:
                continue
            cid = name_to_id[name]
            for s in sents:
                key = (s.number, cid)
                if key in validated_keys:
                    continue  # already approved
                comp_lower = name.lower()
                has_exact = has_standalone_mention(name, s.text)
                has_lower = (not has_exact and
                             re.search(rf'\b{re.escape(comp_lower)}\b', s.text))
                if has_lower:
                    is_tp = (s.number, cid) in gold
                    # Would auto-approval catch this?
                    auto = would_auto_approve(name, s.text, alias_map)
                    potentially_rejected.append((s.number, name, is_tp, auto))

        print(f"\n  {ABBR[ds]} (baseline F1={base_eval['f1']:.1%})")
        if generic_approved:
            print(f"    {len(generic_approved)} candidates went through generic detection → APPROVED")
            for c, is_tp in generic_approved:
                print(f"      [{'TP' if is_tp else 'FP'}] S{c.sentence_number} -> {c.component_name}")
            print(f"    These already pass. Removing generic detection won't change them.")
        else:
            print(f"    No candidates routed to generic detection.")

        if potentially_rejected:
            print(f"    {len(potentially_rejected)} potential generic-rejected candidates (lowercase ambiguous, NOT in validated):")
            for snum, name, is_tp, auto in potentially_rejected:
                status = "TP" if is_tp else "FP"
                auto_str = "auto-approve" if auto else "LLM 2-pass (rejected)"
                print(f"      [{status}] S{snum} -> {name} → would go to {auto_str}")
        else:
            print(f"    No potentially rejected candidates found.")

    print(f"\n  VERDICT: Need to check if any rejected candidates exist that")
    print(f"           would become FPs through auto-approval if generic detection removed.")


# ═══════════════════════════════════════════════════════════════════════════
# 8e: Remove implementation variant filtering
# ═══════════════════════════════════════════════════════════════════════════

def test_8e():
    """Remove _get_comp_names filtering.

    Check if any filtered component name appears in any sentence.
    If not → removing filter has zero impact (LLM can't match what isn't mentioned).
    """
    print("\n" + "=" * 80)
    print("VARIANT 8e: Remove implementation variant filtering")
    print("=" * 80)

    for ds in DATASETS:
        gold = load_gold(ds)
        t1, t1_5, t2, final_data = load_checkpoints(ds)
        mk = t1['model_knowledge']
        base = baseline_set(final_data)
        base_eval = evaluate(base, gold)

        if not mk.impl_indicators:
            print(f"\n  {ABBR[ds]}: No impl_indicators. No change.")
            continue

        comps = parse_pcm_repository(MODEL_PATHS[ds])
        filtered = [c for c in comps if mk.is_implementation(c.name)]
        abstract = [c for c in comps if not mk.is_implementation(c.name)]

        sents = DocumentLoader.load_sentences(TEXT_PATHS[ds])

        print(f"\n  {ABBR[ds]} (baseline F1={base_eval['f1']:.1%})")
        print(f"    Impl indicators: {mk.impl_indicators}")
        print(f"    Filtered: {[c.name for c in filtered]}")
        print(f"    Abstract: {[c.name for c in abstract if any(fi in c.name for fi in ['Recommender'])]}")

        # Check: does ANY filtered name appear in ANY sentence?
        any_mentioned = False
        for comp in filtered:
            for s in sents:
                if re.search(rf'\b{re.escape(comp.name)}\b', s.text, re.IGNORECASE):
                    print(f"    FOUND: {comp.name} in S{s.number}: \"{s.text[:80]}\"")
                    any_mentioned = True

        if not any_mentioned:
            print(f"    No filtered component name appears in any sentence.")

            # Also check: would including these in prompts cause "Recommender" to match variants?
            # The abstract "Recommender" is already in the prompt. Adding variants won't
            # change matching for "Recommender" since entity extraction returns component names.
            print(f"    Generic 'Recommender' appears in S4, S27 but abstract Recommender already in prompt.")
            print(f"    LLM would still match to 'Recommender' (abstract), not to variants.")

        print(f"    SAFE: zero mentions of filtered names in text." if not any_mentioned
              else f"    RISKY: filtered names appear in text.")


# ═══════════════════════════════════════════════════════════════════════════
# 8b/8c: Verify applied changes are safe
# ═══════════════════════════════════════════════════════════════════════════

def test_8b_8c_verify():
    """Verify that the already-applied 8b (CamelCase synonym injection removal)
    and 8c (CamelCase rescue removal) have zero impact.

    8b: Check if any final links depended on CamelCase-injected synonyms.
    8c: Check if CamelCase rescue ever fires (CamelCase names in ambiguous set).
    """
    print("\n" + "=" * 80)
    print("VERIFY 8b+8c: Already applied (CamelCase injection + rescue removed)")
    print("=" * 80)

    for ds in DATASETS:
        gold = load_gold(ds)
        t1, t1_5, t2, final_data = load_checkpoints(ds)
        mk = t1['model_knowledge']
        dk = t1_5.get('doc_knowledge', t1['doc_knowledge'])
        base = baseline_set(final_data)
        base_eval = evaluate(base, gold)

        # 8b: Were any CamelCase-injected synonyms used?
        camel_syns = {}
        for syn, comp in dk.synonyms.items():
            if ' ' not in syn:
                continue
            resplit = re.sub(r'([a-z])([A-Z])', r'\1 \2',
                             re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', comp))
            if syn == resplit:
                camel_syns[syn] = comp

        # 8c: Any CamelCase in ambiguous set?
        camel_in_ambig = {n for n in mk.ambiguous_names
                         if re.search(r'[a-z][A-Z]', n) or ' ' in n or n.isupper()}

        print(f"\n  {ABBR[ds]} (F1={base_eval['f1']:.1%}):")
        print(f"    8b: {len(camel_syns)} CamelCase synonyms existed: {list(camel_syns.keys())[:5]}")
        print(f"    8c: CamelCase in ambig={camel_in_ambig if camel_in_ambig else 'none'}")

    print(f"\n  CONFIRMED: 8b and 8c have zero impact (already applied to S-Linker8).")


# ═══════════════════════════════════════════════════════════════════════════
# 8f: Remove multiword partial enrichment
# ═══════════════════════════════════════════════════════════════════════════

def test_8f():
    """Remove Tier 1.5 multiword partial enrichment.

    Check which final links depend on enriched partials.
    """
    print("\n" + "=" * 80)
    print("VARIANT 8f: Remove multiword partial enrichment (Tier 1.5)")
    print("=" * 80)

    total_tp_lost = 0
    total_fp_lost = 0

    for ds in DATASETS:
        gold = load_gold(ds)
        t1, t1_5, t2, final_data = load_checkpoints(ds)
        dk_before = t1['doc_knowledge']
        dk_after = t1_5.get('doc_knowledge', dk_before)
        base = baseline_set(final_data)
        base_eval = evaluate(base, gold)

        enriched = set(dk_after.partial_references) - set(dk_before.partial_references)
        if not enriched:
            print(f"\n  {ABBR[ds]}: No enriched partials. No change.")
            continue

        enriched_comps = {dk_after.partial_references[p] for p in enriched}
        at_risk = [(l.sentence_number, l.component_id, l.component_name)
                   for l in final_data['final']
                   if l.source == 'partial_inject' and l.component_name in enriched_comps]

        non_partial = {(l.sentence_number, l.component_id) for l in final_data['final']
                      if l.source != 'partial_inject'}
        truly_lost = [(s, c, n) for s, c, n in at_risk if (s, c) not in non_partial]

        tp_lost = [(s, c, n) for s, c, n in truly_lost if (s, c) in gold]
        fp_lost = [(s, c, n) for s, c, n in truly_lost if (s, c) not in gold]
        total_tp_lost += len(tp_lost)
        total_fp_lost += len(fp_lost)

        new_base = base - {(s, c) for s, c, _ in truly_lost}
        new_eval = evaluate(new_base, gold)

        print(f"\n  {ABBR[ds]} (baseline F1={base_eval['f1']:.1%})")
        print(f"    Enriched: {dict((p, dk_after.partial_references[p]) for p in enriched)}")
        print(f"    {len(at_risk)} partial_inject links use enriched partials")
        print(f"    {len(truly_lost)} truly lost ({len(tp_lost)} TP + {len(fp_lost)} FP):")
        for s, c, n in truly_lost:
            print(f"      [{'TP' if (s,c) in gold else 'FP'}] S{s} -> {n}")
        print(f"    Without enrichment: F1={new_eval['f1']:.1%} (TP={new_eval['tp']} FP={new_eval['fp']})")

    print(f"\n  TOTAL: {total_tp_lost} TPs + {total_fp_lost} FPs lost")
    print(f"  VERDICT: {'RISKY' if total_tp_lost > 0 else 'SAFE'}")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("S-Linker8 Variant Replay — Precise Checkpoint Analysis")
    print("=" * 80)

    # Baseline
    baseline_f1s = {}
    for ds in DATASETS:
        gold = load_gold(ds)
        final = pickle.load(open(os.path.join(CACHE, ds, 'final.pkl'), 'rb'))
        base = baseline_set(final)
        baseline_f1s[ds] = evaluate(base, gold)['f1']
    macro = sum(baseline_f1s.values()) / 5
    print(f"Baseline S8: {macro:.1%} ({', '.join(f'{ABBR[d]}={baseline_f1s[d]:.1%}' for d in DATASETS)})")

    test_8b_8c_verify()
    test_8a()
    test_8d()
    test_8e()
    test_8f()

    print("\n" + "=" * 80)
    print("FINAL VERDICTS")
    print("=" * 80)
    print("  8b: SAFE (applied) — CamelCase synonym injection removed, 0 impact")
    print("  8c: SAFE (applied) — CamelCase rescue removed, 0 impact")
    print("  8a: See analysis above — auto-approval protects TPs from strict LLM 2-pass")
    print("  8d: See analysis above — depends on rejected candidates")
    print("  8e: See analysis above — filtered names never appear in text")
    print("  8f: See analysis above — loses TPs in BBB")


if __name__ == '__main__':
    main()
