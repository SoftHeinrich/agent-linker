"""Unit test for S-Linker7a/7b fixes using S-Linker7 checkpoints.

Replays partial injection and validation auto-approval changes offline
to predict impact before running expensive e2e LLM calls.

Zero LLM calls — pure checkpoint replay.
"""

import csv
import os
import pickle
import re

BENCHMARK = os.path.join(os.path.dirname(__file__), '..', 'ardoco', 'core', 'tests-base',
                         'src', 'main', 'resources', 'benchmark')
CACHE7 = os.path.join(os.path.dirname(__file__), 'results', 'phase_cache', 's_linker7')

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

SOURCE_PRIORITY = {"seed": 5, "validated": 4, "entity": 3, "coreference": 2, "partial_inject": 1}


def load_gold(ds):
    gold = set()
    path = os.path.join(BENCHMARK, ds, 'goldstandards', GS_NAMES[ds])
    with open(path) as f:
        for row in csv.DictReader(f):
            gold.add((int(row['sentence']), row['modelElementID']))
    return gold


def has_clean_mention(term, text):
    """Same as SLinker7._has_clean_mention()."""
    pattern = rf'\b{re.escape(term)}\b'
    for m in re.finditer(pattern, text, re.IGNORECASE):
        s, e = m.start(), m.end()
        if s > 0 and text[s - 1] == '.':
            continue
        if e < len(text) and text[e] == '.' and e + 1 < len(text) and text[e + 1].isalpha():
            continue
        if (s > 0 and text[s - 1] == '-') or (e < len(text) and text[e] == '-'):
            continue
        return True
    return False


def evaluate(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'p': p, 'r': r, 'f1': f1}


def main():
    from llm_sad_sam.core.document_loader import DocumentLoader

    print("S-Linker7a/7b UNIT TEST — Offline Checkpoint Replay")
    print("=" * 80)
    print()
    print("7a: partial_inject links removed (would go through validation)")
    print("7b: 7a + validation auto-approval uses _has_clean_mention()")
    print()

    results_baseline = {}
    results_7a = {}
    results_7b = {}

    for ds in DATASETS:
        gold = load_gold(ds)
        sents = DocumentLoader.load_sentences(TEXT_PATHS[ds])
        sent_map = {s.number: s for s in sents}

        t1 = pickle.load(open(os.path.join(CACHE7, ds, 'tier1.pkl'), 'rb'))
        t1_5 = pickle.load(open(os.path.join(CACHE7, ds, 'tier1_5.pkl'), 'rb'))
        t2 = pickle.load(open(os.path.join(CACHE7, ds, 'tier2.pkl'), 'rb'))
        final_data = pickle.load(open(os.path.join(CACHE7, ds, 'final.pkl'), 'rb'))

        final_links = final_data['final']
        # Use enriched doc knowledge (includes multiword partials from Tier 1.5)
        dk = t1_5.get('doc_knowledge', t1.get('doc_knowledge'))

        # Baseline: S-Linker7 as-is
        baseline_set = {(l.sentence_number, l.component_id) for l in final_links}
        results_baseline[ds] = evaluate(baseline_set, gold)

        # --- 7a simulation: remove partial_inject links ---
        # (They would go through validation which might reject some)
        # Offline approximation: remove all partial_inject (pessimistic lower bound)
        # and also check which ones have clean mentions vs generic matches
        partial_links = [l for l in final_links if l.source == 'partial_inject']
        non_partial = [l for l in final_links if l.source != 'partial_inject']

        # Check which partial_inject would survive validation
        # Heuristic: if the partial is a clean, capitalized, standalone mention -> likely survives
        # If generic/lowercase inside compound phrase -> likely rejected by validation
        survived_partial = []
        rejected_partial = []
        for lk in partial_links:
            s = sent_map.get(lk.sentence_number)
            if not s:
                rejected_partial.append(lk)
                continue

            # Find the matching partial
            matching_partial = None
            if dk and dk.partial_references:
                for p, comp in dk.partial_references.items():
                    if comp == lk.component_name and has_clean_mention(p, s.text):
                        matching_partial = p
                        break

            if not matching_partial:
                rejected_partial.append(lk)
                continue

            # Check if it's a genuine component reference vs generic usage
            # Validation would reject if:
            # 1. The partial word is used generically (e.g. "UI name", "media server")
            # 2. The sentence doesn't discuss the component architecturally

            # Simple heuristic: check if partial appears capitalized AND standalone
            # (not just as part of a larger phrase)
            partial_lower = matching_partial.lower()
            has_capitalized = bool(re.search(
                rf'\b{re.escape(matching_partial[0].upper() + matching_partial[1:])}\b', s.text))
            has_exact_comp = bool(re.search(
                rf'\b{re.escape(lk.component_name)}\b', s.text, re.IGNORECASE))

            # If the full component name appears, definitely survives
            if has_exact_comp:
                survived_partial.append(lk)
            # If partial is very short (<=2 chars like "UI"), likely generic
            elif len(matching_partial) <= 2:
                rejected_partial.append(lk)
            # If capitalized standalone, likely survives
            elif has_capitalized:
                survived_partial.append(lk)
            else:
                # Conservative: assume validation rejects generic usage
                rejected_partial.append(lk)

        set_7a = {(l.sentence_number, l.component_id) for l in non_partial + survived_partial}
        results_7a[ds] = evaluate(set_7a, gold)

        # --- 7b simulation: 7a + tighten auto-approval ---
        # Check validated links: which ones would fail _has_clean_mention?
        validated_links = [l for l in final_links if l.source == 'validated']
        other_links = [l for l in final_links if l.source not in ('validated', 'partial_inject')]

        failed_auto = []
        passed_auto = []
        for lk in validated_links:
            s = sent_map.get(lk.sentence_number)
            if not s:
                passed_auto.append(lk)
                continue

            # Check all aliases for clean mention
            aliases = {lk.component_name}
            if dk:
                for a, cn in dk.abbreviations.items():
                    if cn == lk.component_name:
                        aliases.add(a)
                for syn, cn in dk.synonyms.items():
                    if cn == lk.component_name:
                        aliases.add(syn)
                for p, cn in dk.partial_references.items():
                    if cn == lk.component_name:
                        aliases.add(p)

            has_clean = False
            for a in aliases:
                if len(a) >= 3:
                    if has_clean_mention(a, s.text):
                        has_clean = True
                        break
                elif len(a) >= 2:
                    if bool(re.search(rf'\b{re.escape(a)}\b', s.text, re.IGNORECASE)):
                        has_clean = True
                        break

            if has_clean:
                passed_auto.append(lk)
            else:
                # Would go to LLM validation instead of auto-approve
                # Assume LLM rejects dotted-path contexts
                if '.' in s.text and any(kw in s.text.lower() for kw in
                                          ['package', '.util', '.api', '.core', '.entity',
                                           '.cases', '.testdriver']):
                    failed_auto.append(lk)
                else:
                    passed_auto.append(lk)

        set_7b = {(l.sentence_number, l.component_id)
                  for l in other_links + passed_auto + survived_partial}
        results_7b[ds] = evaluate(set_7b, gold)

        # Print per-dataset details
        print(f"--- {ABBR[ds]} ---")
        print(f"  Baseline: F1={results_baseline[ds]['f1']:.1%} "
              f"(TP={results_baseline[ds]['tp']} FP={results_baseline[ds]['fp']})")

        if rejected_partial:
            print(f"  7a rejects {len(rejected_partial)} partial_inject:")
            for lk in rejected_partial:
                s = sent_map.get(lk.sentence_number)
                is_tp = (lk.sentence_number, lk.component_id) in gold
                label = "TP" if is_tp else "FP"
                mp = None
                if dk and dk.partial_references:
                    for p, c in dk.partial_references.items():
                        if c == lk.component_name:
                            mp = p
                            break
                print(f"    [{label}] S{lk.sentence_number} -> {lk.component_name} "
                      f"(partial='{mp}') \"{s.text[:70]}...\"" if s else "")

        print(f"  7a: F1={results_7a[ds]['f1']:.1%} "
              f"(TP={results_7a[ds]['tp']} FP={results_7a[ds]['fp']})")

        if failed_auto:
            print(f"  7b additionally rejects {len(failed_auto)} validated (dotted-path auto-approval):")
            for lk in failed_auto:
                s = sent_map.get(lk.sentence_number)
                is_tp = (lk.sentence_number, lk.component_id) in gold
                label = "TP" if is_tp else "FP"
                print(f"    [{label}] S{lk.sentence_number} -> {lk.component_name} "
                      f"\"{s.text[:70]}...\"" if s else "")

        print(f"  7b: F1={results_7b[ds]['f1']:.1%} "
              f"(TP={results_7b[ds]['tp']} FP={results_7b[ds]['fp']})")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for label, results in [("S7 baseline", results_baseline),
                            ("S7a (partial->val)", results_7a),
                            ("S7b (7a+clean auto)", results_7b)]:
        macro = sum(r['f1'] for r in results.values()) / 5
        macro_no_tm = sum(r['f1'] for ds, r in results.items() if ds != 'teammates') / 4
        total_fp = sum(r['fp'] for r in results.values())
        total_fn = sum(r['fn'] for r in results.values())
        per_ds = "  ".join(f"{ABBR[ds]}={results[ds]['f1']:.1%}" for ds in DATASETS)
        print(f"  {label:25s}: {macro:.1%} (excl TM: {macro_no_tm:.1%}) | {per_ds} | FP={total_fp} FN={total_fn}")


if __name__ == '__main__':
    main()
