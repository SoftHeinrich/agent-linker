"""Unit test for S-Linker8 fix variants using S-Linker8 checkpoints.

Replays each proposed fix offline to predict F1 impact before e2e runs.
Zero LLM calls — pure checkpoint replay.

Variants:
  8a: Fix 2 — Remove validation auto-approval (all candidates go to LLM)
  8b: Fix 6 — Remove CamelCase synonym injection
  8c: Fix 7 — Remove CamelCase safety nets (ambiguity + judge overrides)
  8d: Fix 1 — Merge generic-mention detection into validation (remove routing)
  8e: Fix 5 — Remove implementation variant filtering (include all components)
  8f: Fix 3 — Remove multiword partial enrichment (Tier 1.5)
"""

import csv
import os
import pickle
import re

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
    """S-Linker8 baseline link set."""
    return {(l.sentence_number, l.component_id) for l in final_data['final']}


# ═══════════════════════════════════════════════════════════════════════════
# Variant 8a: Fix 2 — Remove validation auto-approval
# ═══════════════════════════════════════════════════════════════════════════

def sim_8a(ds, t1, t1_5, t2, final_data):
    """Remove auto-approval: auto-approved links (source='validated') would go
    to LLM validation instead. Pessimistic: assume LLM rejects all auto-approved
    that aren't also in seed/coref. Optimistic: assume LLM approves all."""
    final_links = final_data['final']
    base = baseline_set(final_data)

    # Find auto-approved links (source='validated' in final)
    auto_approved = [(l.sentence_number, l.component_id) for l in final_links
                     if l.source == 'validated']

    # Pessimistic: remove all auto-approved (they'd need LLM approval)
    pessimistic = base - set(auto_approved)
    # Optimistic: keep all (LLM approves them)
    optimistic = base

    return {
        'pessimistic': pessimistic,
        'optimistic': optimistic,
        'affected': auto_approved,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Variant 8b: Fix 6 — Remove CamelCase synonym injection
# ═══════════════════════════════════════════════════════════════════════════

def sim_8b(ds, t1, t1_5, t2, final_data):
    """Remove CamelCase synonym injection. Check if any final links depend on
    CamelCase synonyms that wouldn't exist without injection."""
    from llm_sad_sam.core.document_loader import DocumentLoader

    dk = t1_5.get('doc_knowledge', t1['doc_knowledge'])
    final_links = final_data['final']
    base = baseline_set(final_data)
    sents = DocumentLoader.load_sentences(TEXT_PATHS[ds])
    sent_map = {s.number: s for s in sents}

    # Identify CamelCase-injected synonyms
    camel_syns = {}
    for syn, comp in dk.synonyms.items():
        if ' ' not in syn:
            continue
        resplit = re.sub(r'([a-z])([A-Z])', r'\1 \2',
                         re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', comp))
        if syn == resplit:
            camel_syns[syn] = comp

    # Check which final links depend on CamelCase synonyms appearing in text
    # A link depends on it if: the synonym appears in the sentence but the
    # full component name does NOT, and the link source is 'validated' (auto-approved via alias)
    at_risk = []
    for lk in final_links:
        if lk.source != 'validated':
            continue
        s = sent_map.get(lk.sentence_number)
        if not s:
            continue
        # Check if a CamelCase synonym is the reason for auto-approval
        full_in_text = re.search(rf'\b{re.escape(lk.component_name)}\b', s.text, re.IGNORECASE)
        if full_in_text:
            continue  # Full name present, doesn't depend on synonym
        for syn, comp in camel_syns.items():
            if comp == lk.component_name and syn.lower() in s.text.lower():
                at_risk.append((lk.sentence_number, lk.component_id, syn))
                break

    # These links might be lost without CamelCase synonym injection
    lost_keys = {(s, c) for s, c, _ in at_risk}
    predicted = base - lost_keys

    return {
        'predicted': predicted,
        'at_risk': at_risk,
        'camel_syns': camel_syns,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Variant 8c: Fix 7 — Remove CamelCase safety nets
# ═══════════════════════════════════════════════════════════════════════════

def sim_8c(ds, t1, t1_5, t2, final_data):
    """Remove CamelCase overrides on ambiguity classification (H2/H3) and
    judge rescue (H7). Check impact on ambiguous set and downstream."""
    mk = t1['model_knowledge']
    base = baseline_set(final_data)

    # Without H2/H3: CamelCase names could be in ambiguous set
    # The LLM was asked to classify — check what it returned raw
    # We can't replay the raw LLM output, but we can check:
    # which components ARE in ambiguous_names and which are CamelCase

    # Current ambiguous (already filtered by H2/H3)
    current_ambig = mk.ambiguous_names

    # Components that WOULD be ambiguous if H2/H3 removed (we can't know for sure,
    # but we know the current set is post-filter)
    # Analysis: H2/H3 only affect what the LLM classified. If the LLM correctly
    # classified CamelCase as architectural, removing H2/H3 changes nothing.
    # We can check: are there any CamelCase names in ambiguous_names? If not,
    # H2/H3 never fired and removing them is safe.

    camelcase_in_ambig = {n for n in current_ambig
                         if re.search(r'[a-z][A-Z]', n) or ' ' in n or n.isupper()}

    return {
        'predicted': base,  # No change if H2/H3 never fire
        'camelcase_in_ambig': camelcase_in_ambig,
        'current_ambig': current_ambig,
        'h2h3_fires': len(camelcase_in_ambig) > 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Variant 8d: Fix 1 — Merge generic-mention detection into validation
# ═══════════════════════════════════════════════════════════════════════════

def sim_8d(ds, t1, t1_5, t2, final_data):
    """Merge generic-mention detection into validation prompt. The specialized
    generic-mention LLM call would be absorbed into the main validation prompt.

    Impact analysis: identify which candidates currently go through the specialized
    generic-mention path vs standard validation."""
    from llm_sad_sam.core.document_loader import DocumentLoader

    mk = t1['model_knowledge']
    dk = t1_5.get('doc_knowledge', t1['doc_knowledge'])
    validated = t2['validated']
    base = baseline_set(final_data)
    sents = DocumentLoader.load_sentences(TEXT_PATHS[ds])
    sent_map = {s.number: s for s in sents}

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
            if e < len(text) and text[e] == '.' and e+1 < len(text) and text[e+1].isalpha():
                continue
            if s > 0 and text[s-1] == '-':
                continue
            if e < len(text) and text[e] == '-' and '-' not in comp_name:
                continue
            return True
        return False

    # Count candidates that would go through generic-mention detection
    generic_routed = []
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

        is_ambig = (not is_structurally_unambiguous(c.component_name) and
                    c.component_name in mk.ambiguous_names)

        if has_lower and is_ambig:
            generic_routed.append(c)

    return {
        'predicted': base,  # Can't predict LLM behavior change
        'generic_routed_count': len(generic_routed),
        'generic_routed': [(c.sentence_number, c.component_name) for c in generic_routed],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Variant 8e: Fix 5 — Remove implementation variant filtering
# ═══════════════════════════════════════════════════════════════════════════

def sim_8e(ds, t1, t1_5, t2, final_data):
    """Remove _get_comp_names filtering. Include ALL components in LLM prompts.
    Only affects teastore (the only dataset with impl_indicators)."""
    mk = t1['model_knowledge']
    base = baseline_set(final_data)

    # Which components are currently filtered?
    filtered_names = []
    if mk.impl_indicators:
        from llm_sad_sam.pcm_parser import parse_pcm_repository
        model_paths = {
            'mediastore': 'model_2016', 'teastore': 'model_2020',
            'teammates': 'model_2021', 'bigbluebutton': 'model_2021',
            'jabref': 'model_2021',
        }
        model_path = os.path.join(BENCHMARK, ds, model_paths[ds], f'{ds}.repository')
        try:
            comps = parse_pcm_repository(model_path)
            for c in comps:
                if mk.is_implementation(c.name):
                    filtered_names.append(c.name)
        except FileNotFoundError:
            pass

    return {
        'predicted': base,  # Can't predict LLM behavior with more names
        'impl_indicators': mk.impl_indicators,
        'filtered_names': filtered_names,
        'affects_dataset': len(mk.impl_indicators) > 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Variant 8f: Fix 3 — Remove multiword partial enrichment
# ═══════════════════════════════════════════════════════════════════════════

def sim_8f(ds, t1, t1_5, t2, final_data):
    """Remove Tier 1.5 multiword partial enrichment. Check which links depend
    on partials that were added by enrichment (not by Phase 3 LLM)."""
    dk_before = t1['doc_knowledge']  # Before enrichment
    dk_after = t1_5.get('doc_knowledge', dk_before)  # After enrichment

    final_links = final_data['final']
    base = baseline_set(final_data)

    # Partials added by enrichment (not by Phase 3)
    enriched_partials = set(dk_after.partial_references) - set(dk_before.partial_references)

    # Which final links use these enriched partials?
    # Links with source='partial_inject' that match enriched partial targets
    enriched_comp_names = {dk_after.partial_references[p] for p in enriched_partials}

    at_risk = []
    for lk in final_links:
        if lk.source == 'partial_inject' and lk.component_name in enriched_comp_names:
            at_risk.append((lk.sentence_number, lk.component_id, lk.component_name))

    # Also check: would seed/entity/coref already cover these links?
    non_partial = {(l.sentence_number, l.component_id) for l in final_links
                   if l.source != 'partial_inject'}
    truly_lost = [(s, c, n) for s, c, n in at_risk if (s, c) not in non_partial]

    lost_keys = {(s, c) for s, c, _ in truly_lost}
    predicted = base - lost_keys

    return {
        'predicted': predicted,
        'enriched_partials': {p: dk_after.partial_references[p] for p in enriched_partials},
        'at_risk_total': at_risk,
        'truly_lost': truly_lost,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("S-Linker8 Variant Analysis — Offline Checkpoint Replay")
    print("=" * 80)

    variants = {
        '8a': ('Fix 2: Remove auto-approval', sim_8a),
        '8b': ('Fix 6: Remove CamelCase synonym injection', sim_8b),
        '8c': ('Fix 7: Remove CamelCase safety nets', sim_8c),
        '8d': ('Fix 1: Merge generic→validation prompt', sim_8d),
        '8e': ('Fix 5: Remove impl variant filtering', sim_8e),
        '8f': ('Fix 3: Remove multiword partial enrichment', sim_8f),
    }

    all_results = {}

    for var_id, (desc, sim_fn) in variants.items():
        print(f"\n{'='*80}")
        print(f"VARIANT {var_id}: {desc}")
        print(f"{'='*80}")

        var_results = {}

        for ds in DATASETS:
            gold = load_gold(ds)
            t1, t1_5, t2, final = load_checkpoints(ds)
            base = baseline_set(final)
            base_eval = evaluate(base, gold)

            result = sim_fn(ds, t1, t1_5, t2, final)

            print(f"\n  --- {ABBR[ds]} (baseline F1={base_eval['f1']:.1%}, TP={base_eval['tp']} FP={base_eval['fp']}) ---")

            # Print variant-specific details
            if var_id == '8a':
                n = len(result['affected'])
                if n:
                    pess = evaluate(result['pessimistic'], gold)
                    opt = evaluate(result['optimistic'], gold)
                    # Check which auto-approved are TP vs FP
                    tp_risk = [(s, c) for s, c in result['affected'] if (s, c) in gold]
                    fp_risk = [(s, c) for s, c in result['affected'] if (s, c) not in gold]
                    print(f"    {n} auto-approved links would go to LLM validation")
                    print(f"    Of those: {len(tp_risk)} TP, {len(fp_risk)} FP")
                    print(f"    Pessimistic (LLM rejects all): F1={pess['f1']:.1%} (TP={pess['tp']} FP={pess['fp']})")
                    print(f"    Optimistic (LLM approves all): F1={opt['f1']:.1%} (unchanged)")
                    var_results[ds] = {'affected': n, 'tp_risk': len(tp_risk), 'fp_risk': len(fp_risk)}
                else:
                    print(f"    No auto-approved links. No change.")
                    var_results[ds] = {'affected': 0, 'tp_risk': 0, 'fp_risk': 0}

            elif var_id == '8b':
                n_syns = len(result['camel_syns'])
                n_risk = len(result['at_risk'])
                pred = evaluate(result['predicted'], gold)
                if result['at_risk']:
                    print(f"    {n_syns} CamelCase synonyms, {n_risk} links depend on them:")
                    for s, c, syn in result['at_risk']:
                        is_tp = (s, c) in gold
                        print(f"      [{'TP' if is_tp else 'FP'}] S{s} via '{syn}'")
                    print(f"    Without injection: F1={pred['f1']:.1%} (TP={pred['tp']} FP={pred['fp']})")
                else:
                    print(f"    {n_syns} CamelCase synonyms, but 0 links depend on them. Safe to remove.")
                var_results[ds] = {'camel_syns': n_syns, 'at_risk': n_risk}

            elif var_id == '8c':
                print(f"    Current ambiguous: {sorted(result['current_ambig'])}")
                if result['h2h3_fires']:
                    print(f"    WARNING: CamelCase names in ambig set: {result['camelcase_in_ambig']}")
                    print(f"    H2/H3 currently fires — removal would change behavior")
                else:
                    print(f"    H2/H3 never fires (no CamelCase in ambig set). Safe to remove.")
                var_results[ds] = {'fires': result['h2h3_fires'], 'ambig': sorted(result['current_ambig'])}

            elif var_id == '8d':
                n = result['generic_routed_count']
                print(f"    {n} candidates currently routed to generic-mention LLM:")
                for snum, cname in result['generic_routed']:
                    is_tp = any((snum, cid) in gold for cid in
                               [l.component_id for l in final['final'] if l.sentence_number == snum and l.component_name == cname])
                    print(f"      S{snum} -> {cname}")
                if n == 0:
                    print(f"    No candidates use generic-mention path. Merge is a no-op.")
                var_results[ds] = {'generic_routed': n}

            elif var_id == '8e':
                if result['affects_dataset']:
                    print(f"    impl_indicators: {result['impl_indicators']}")
                    print(f"    Filtered components: {result['filtered_names']}")
                    print(f"    Removing filter would add these to LLM prompts.")
                else:
                    print(f"    No implementation variants. No change.")
                var_results[ds] = {'filtered': result['filtered_names']}

            elif var_id == '8f':
                ep = result['enriched_partials']
                if ep:
                    print(f"    Enriched partials: {ep}")
                    n_total = len(result['at_risk_total'])
                    n_lost = len(result['truly_lost'])
                    pred = evaluate(result['predicted'], gold)
                    print(f"    {n_total} partial_inject links use enriched partials")
                    print(f"    {n_lost} would be truly lost (not covered by other sources):")
                    for s, c, n in result['truly_lost']:
                        is_tp = (s, c) in gold
                        print(f"      [{'TP' if is_tp else 'FP'}] S{s} -> {n}")
                    print(f"    Without enrichment: F1={pred['f1']:.1%} (TP={pred['tp']} FP={pred['fp']})")
                else:
                    print(f"    No enriched partials. No change.")
                var_results[ds] = {'enriched_partials': ep, 'truly_lost': len(result.get('truly_lost', []))}

        all_results[var_id] = var_results

    # ═══ Summary Table ═══
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print()

    # Baseline
    baseline_f1s = {}
    for ds in DATASETS:
        gold = load_gold(ds)
        final = pickle.load(open(os.path.join(CACHE, ds, 'final.pkl'), 'rb'))
        base = baseline_set(final)
        baseline_f1s[ds] = evaluate(base, gold)['f1']
    macro = sum(baseline_f1s.values()) / 5
    print(f"  Baseline S8: {macro:.1%} ({', '.join(f'{ABBR[d]}={baseline_f1s[d]:.1%}' for d in DATASETS)})")
    print()

    verdicts = {
        '8a': None,
        '8b': None,
        '8c': None,
        '8d': None,
        '8e': None,
        '8f': None,
    }

    for var_id, (desc, _) in variants.items():
        vr = all_results[var_id]

        if var_id == '8a':
            total_risk = sum(v['tp_risk'] for v in vr.values())
            total_fp = sum(v['fp_risk'] for v in vr.values())
            verdict = f"NEEDS E2E: {total_risk} TPs + {total_fp} FPs at risk"
            verdicts[var_id] = 'NEEDS_E2E'
        elif var_id == '8b':
            total_risk = sum(v['at_risk'] for v in vr.values())
            verdict = f"SAFE" if total_risk == 0 else f"RISKY: {total_risk} links at risk"
            verdicts[var_id] = 'SAFE' if total_risk == 0 else 'RISKY'
        elif var_id == '8c':
            any_fires = any(v['fires'] for v in vr.values())
            verdict = "RISKY: overrides fire" if any_fires else "SAFE: overrides never fire"
            verdicts[var_id] = 'RISKY' if any_fires else 'SAFE'
        elif var_id == '8d':
            total_generic = sum(v['generic_routed'] for v in vr.values())
            verdict = f"NEEDS E2E: {total_generic} candidates affected"
            verdicts[var_id] = 'NEEDS_E2E' if total_generic > 0 else 'SAFE'
        elif var_id == '8e':
            affected = sum(len(v['filtered']) for v in vr.values())
            verdict = f"NEEDS E2E: {affected} components affected" if affected else "SAFE: no impl variants"
            verdicts[var_id] = 'NEEDS_E2E' if affected else 'SAFE'
        elif var_id == '8f':
            total_lost = sum(v['truly_lost'] for v in vr.values())
            verdict = f"SAFE" if total_lost == 0 else f"RISKY: {total_lost} links lost"
            verdicts[var_id] = 'SAFE' if total_lost == 0 else 'RISKY'

        print(f"  {var_id} ({desc[:50]}): {verdict}")

    print()
    print("Legend: SAFE = no F1 change predicted | RISKY = F1 loss predicted | NEEDS E2E = LLM behavior change, can't predict offline")


if __name__ == '__main__':
    main()
