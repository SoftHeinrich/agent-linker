#!/usr/bin/env python3
"""Unit test: Does the multi-word synonym judge example (Ex5) change approval rates?

Replays the Phase 3 doc-knowledge judge on actual S-Linker11 checkpoint data.
Runs twice: once with OLD examples (no multi-word), once with NEW (Ex5 added).
Compares approval sets to measure impact.

Key question: Are the 12 multi-word synonyms that were previously approved
still approved? Does the new example cause any regressions?
"""

import csv
import os
import pickle
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
os.environ["CLAUDE_MODEL"] = "sonnet"

from llm_sad_sam.llm_client import LLMClient

CACHE = Path("results/phase_cache/s_linker11")
BENCHMARK = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
DATASETS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]

# ─── OLD examples (before this change) ───────────────────────────────────────

OLD_JUDGE_EXAMPLES = """EXAMPLES — study these to calibrate your judgment:

Example 1 — APPROVE (abbreviation from component name):
  'AST' -> AbstractSyntaxTree (abbrev)
  Verdict: APPROVE. "AST" is the initials of "AbstractSyntaxTree". Abbreviations
  formed from the component name's words are always valid.

Example 2 — APPROVE (trailing word of multi-word name):
  'Dispatcher' -> TaskDispatcher (partial)
  Verdict: APPROVE. "Dispatcher" is the last word of "TaskDispatcher".
  If no other component ends in "Dispatcher", this partial is unambiguous.

Example 3 — APPROVE (CamelCase identifier):
  'RenderEngine' -> GameRenderEngine (synonym)
  Verdict: APPROVE. CamelCase is a constructed identifier — always a proper name.

Example 4 — APPROVE (trailing word of multi-word name):
  'Table' -> SymbolTable (partial)
  Verdict: APPROVE. "Table" is the trailing word of "SymbolTable" and
  likely refers to this specific component when no other component uses "Table".

Example 5 — REJECT (ordinary English verb/noun):
  'handle' -> InvoiceHandler (partial)
  Verdict: REJECT. "handle" is an ordinary English verb used generically
  in many contexts ("handle requests", "the handler").

Example 6 — REJECT (refers to whole system):
  'system' -> PaymentSystem (partial)
  Verdict: REJECT. "system" is too generic — it could refer to the overall system."""

# ─── NEW examples (with multi-word synonym Ex5) ──────────────────────────────

from llm_sad_sam.linkers.experimental.prompts_v2 import (
    DOC_KNOWLEDGE_JUDGE_EXAMPLES as NEW_JUDGE_EXAMPLES,
    DOC_KNOWLEDGE_JUDGE_RULES,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_proposed_mappings(dataset):
    """Reconstruct the proposed mappings that Phase 3 judge would see.

    Loads layer1 (raw extraction before judge) by re-running the extraction
    step against the document. Since we can't replay Phase 3 extraction
    cheaply, we instead use the APPROVED results from layer2 as a proxy
    and add back known-rejected mappings to test the full set.

    For this test we use the approved set from layer2 directly — if the
    judge approves the same set with new examples, the new example is safe.
    """
    l1 = pickle.load(open(CACHE / dataset / "layer1.pkl", "rb"))
    l2 = pickle.load(open(CACHE / dataset / "layer2.pkl", "rb"))
    dk = l2["doc_knowledge"]

    # Reconstruct the mapping list as the judge would see it
    all_mappings = {}
    for short, full in dk.abbreviations.items():
        all_mappings[short] = ("abbrev", full)
    for syn, full in dk.synonyms.items():
        all_mappings[syn] = ("synonym", full)
    for partial, full in dk.partial_references.items():
        all_mappings[partial] = ("partial", full)

    # Get component names
    model_dirs = list((BENCHMARK / dataset).glob("model_*/pcm/*.repository"))
    from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
    comps = parse_pcm_repository(str(model_dirs[0]))
    comp_names = [c.name for c in comps]

    return all_mappings, comp_names


def run_judge(llm, comp_names, all_mappings, judge_examples, label=""):
    """Run the Phase 3 judge with given examples. Returns set of approved terms."""
    if not all_mappings:
        return set()

    mapping_list = [
        f"'{k}' -> {v[1]} ({v[0]})"
        for k, v in list(all_mappings.items())[:25]
    ]

    prompt = f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

{judge_examples}

{DOC_KNOWLEDGE_JUDGE_RULES}

Return JSON:
{{
  "approved": ["term1", "term2"]
}}
JSON only:"""

    data = llm.extract_json(llm.query(prompt, timeout=120))
    approved = set(data.get("approved", [])) if data else set()
    return approved


def classify_mapping(term):
    """Classify a mapping term for reporting."""
    if ' ' in term:
        return "multi-word"
    if re.match(r'^[A-Z][a-z]+(?:[A-Z][a-z]+)+$', term):
        return "CamelCase"
    if term.isupper() and len(term) >= 2:
        return "all-caps"
    if '-' in term:
        return "hyphenated"
    return "single-word"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    llm = LLMClient()

    print("=" * 70)
    print("UNIT TEST: Multi-word synonym judge example (Ex5) effectiveness")
    print("=" * 70)
    print(f"\nOLD examples: 4 APPROVE (abbrev, partial×2, CamelCase) + 2 REJECT")
    print(f"NEW examples: 5 APPROVE (+multi-word phrase) + 2 REJECT")

    results = []

    for dataset in DATASETS:
        print(f"\n{'─' * 60}")
        print(f"  {dataset}")
        print(f"{'─' * 60}")

        all_mappings, comp_names = load_proposed_mappings(dataset)
        if not all_mappings:
            print("  No mappings to judge — skip")
            continue

        # Classify mappings
        by_type = {}
        for term in all_mappings:
            t = classify_mapping(term)
            by_type.setdefault(t, []).append(term)

        print(f"  Mappings to judge: {len(all_mappings)}")
        for t, terms in sorted(by_type.items()):
            print(f"    {t}: {terms}")

        # Run OLD judge
        t0 = time.time()
        old_approved = run_judge(llm, comp_names, all_mappings, OLD_JUDGE_EXAMPLES, "OLD")
        old_time = time.time() - t0
        print(f"\n  OLD approved ({len(old_approved)}): {sorted(old_approved)}")

        # Run NEW judge
        t0 = time.time()
        new_approved = run_judge(llm, comp_names, all_mappings, NEW_JUDGE_EXAMPLES, "NEW")
        new_time = time.time() - t0
        print(f"  NEW approved ({len(new_approved)}): {sorted(new_approved)}")

        # Diff
        gained = new_approved - old_approved
        lost = old_approved - new_approved
        unchanged = old_approved & new_approved

        print(f"\n  DIFF:")
        print(f"    Unchanged: {len(unchanged)}")
        if gained:
            for t in sorted(gained):
                typ = classify_mapping(t)
                comp = all_mappings[t][1]
                print(f"    + GAINED: '{t}' -> {comp} [{typ}]")
        if lost:
            for t in sorted(lost):
                typ = classify_mapping(t)
                comp = all_mappings[t][1]
                print(f"    - LOST:   '{t}' -> {comp} [{typ}]")
        if not gained and not lost:
            print(f"    (identical)")

        # Check multi-word specifically
        mw_terms = by_type.get("multi-word", [])
        if mw_terms:
            old_mw = len([t for t in mw_terms if t in old_approved])
            new_mw = len([t for t in mw_terms if t in new_approved])
            print(f"\n  Multi-word synonyms: OLD {old_mw}/{len(mw_terms)}, NEW {new_mw}/{len(mw_terms)}")
            for t in mw_terms:
                old_ok = "✓" if t in old_approved else "✗"
                new_ok = "✓" if t in new_approved else "✗"
                changed = " ← CHANGED" if (t in old_approved) != (t in new_approved) else ""
                print(f"    {old_ok}→{new_ok} '{t}' -> {all_mappings[t][1]}{changed}")

        results.append({
            "ds": dataset,
            "total": len(all_mappings),
            "old": len(old_approved),
            "new": len(new_approved),
            "gained": len(gained),
            "lost": len(lost),
            "mw_total": len(mw_terms),
            "mw_old": len([t for t in mw_terms if t in old_approved]),
            "mw_new": len([t for t in mw_terms if t in new_approved]),
        })

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    hdr = f"{'Dataset':<15} {'Map':>4} {'OLD':>4} {'NEW':>4} │ {'Gain':>4} {'Lost':>4} │ {'MW':>3} {'MWo':>3} {'MWn':>3}"
    print(hdr)
    print("─" * len(hdr))
    for r in results:
        print(f"{r['ds']:<15} {r['total']:>4} {r['old']:>4} {r['new']:>4} │ "
              f"{r['gained']:>4} {r['lost']:>4} │ "
              f"{r['mw_total']:>3} {r['mw_old']:>3} {r['mw_new']:>3}")

    total_gained = sum(r["gained"] for r in results)
    total_lost = sum(r["lost"] for r in results)
    total_mw = sum(r["mw_total"] for r in results)
    total_mw_old = sum(r["mw_old"] for r in results)
    total_mw_new = sum(r["mw_new"] for r in results)
    print("─" * len(hdr))
    print(f"{'TOTAL':<15} {sum(r['total'] for r in results):>4} "
          f"{sum(r['old'] for r in results):>4} {sum(r['new'] for r in results):>4} │ "
          f"{total_gained:>4} {total_lost:>4} │ "
          f"{total_mw:>3} {total_mw_old:>3} {total_mw_new:>3}")

    print(f"\n  Multi-word approval: OLD {total_mw_old}/{total_mw} → NEW {total_mw_new}/{total_mw}")

    # Assertions
    if total_lost > 0:
        print(f"\n  ⚠ WARNING: {total_lost} mappings LOST — new example may be too restrictive")
    else:
        print(f"\n  ✓ Zero regressions")

    if total_mw_new >= total_mw_old:
        print(f"  ✓ Multi-word approval maintained or improved ({total_mw_old} → {total_mw_new})")
    else:
        print(f"  ⚠ Multi-word approval decreased ({total_mw_old} → {total_mw_new})")

    # Hard assertion: no regressions on multi-word
    assert total_mw_new >= total_mw_old, (
        f"FAIL: Multi-word approval regressed: {total_mw_old} → {total_mw_new}"
    )
    print(f"\n  PASS")


if __name__ == "__main__":
    main()
