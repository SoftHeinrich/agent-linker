#!/usr/bin/env python3
"""Deep phase-by-phase fault analysis: Claude V32 vs GPT V32.

Loads checkpoints from:
- Claude: results/phase_cache/v32/ (Mar 9 run)
- GPT: results/phase_cache/v36b/ (Mar 10 run — same pipeline as V32 through Phase 8)

Compares every phase output to identify exactly where and why GPT diverges.
"""

import pickle
import os
import sys

DATASETS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
CLAUDE_DIR = "results/phase_cache/v32"
GPT_DIR = "results/phase_cache/v36b"  # Same as V32 through Phase 8

# Gold standard link counts for reference
GOLD = {
    "mediastore": 31, "teastore": 27, "teammates": 57,
    "bigbluebutton": 62, "jabref": 18
}


def load_pkl(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def link_set(links):
    """Convert list of link objects to set of (sentence_id, component) tuples."""
    result = set()
    for l in links:
        sid = getattr(l, 'sentence_number', None) or getattr(l, 'sentence_id', None)
        comp = getattr(l, 'component_name', None) or getattr(l, 'component', None)
        if sid and comp:
            result.add((sid, comp))
    return result


def compare_links(claude_links, gpt_links, label=""):
    """Compare two link sets, return (common, claude_only, gpt_only)."""
    c_set = link_set(claude_links) if claude_links else set()
    g_set = link_set(gpt_links) if gpt_links else set()
    common = c_set & g_set
    c_only = c_set - g_set
    g_only = g_set - c_set
    return common, c_only, g_only


def analyze_phase0(claude, gpt, ds):
    """Document profile — should be identical (deterministic)."""
    print(f"\n  [Phase 0] Document Profile")
    if claude and gpt:
        c_complex = claude.get('is_complex', 'N/A')
        g_complex = gpt.get('is_complex', 'N/A')
        match = "MATCH" if c_complex == g_complex else "DIFFER"
        print(f"    Complex: Claude={c_complex}, GPT={g_complex} [{match}]")
        c_spc = claude.get('spc', 'N/A')
        g_spc = gpt.get('spc', 'N/A')
        print(f"    SPC: Claude={c_spc}, GPT={g_spc}")


def analyze_phase1(claude, gpt, ds):
    """Model structure — LLM classifies ambiguous components."""
    print(f"\n  [Phase 1] Model Structure (Ambiguity Classification)")
    if not claude or not gpt:
        print("    Missing checkpoint(s)")
        return

    c_amb = set(claude.get('generic_component_words', []))
    g_amb = set(gpt.get('generic_component_words', []))
    c_part = set(claude.get('generic_partials', []))
    g_part = set(gpt.get('generic_partials', []))

    print(f"    Ambiguous words: Claude={sorted(c_amb)}, GPT={sorted(g_amb)}")
    if c_amb != g_amb:
        print(f"      Claude-only: {sorted(c_amb - g_amb)}")
        print(f"      GPT-only: {sorted(g_amb - c_amb)}")
    else:
        print(f"      MATCH")

    if c_part != g_part:
        print(f"    Partials differ: Claude-only={sorted(c_part - g_part)}, GPT-only={sorted(g_part - c_part)}")


def analyze_phase3(claude, gpt, ds):
    """Document knowledge — synonyms, abbreviations, generics."""
    print(f"\n  [Phase 3] Document Knowledge")
    if not claude or not gpt:
        print("    Missing checkpoint(s)")
        return

    # Extract synonym mappings
    c_syns = claude.get('doc_knowledge', {})
    g_syns = gpt.get('doc_knowledge', {})

    if hasattr(c_syns, 'synonyms'):
        c_syn_set = set((k, v) for k, v in c_syns.synonyms.items()) if hasattr(c_syns.synonyms, 'items') else set()
        g_syn_set = set((k, v) for k, v in g_syns.synonyms.items()) if hasattr(g_syns.synonyms, 'items') else set()
    else:
        c_syn_set = set()
        g_syn_set = set()
        # Try dict-based access
        if isinstance(c_syns, dict):
            c_s = c_syns.get('synonyms', {})
            g_s = g_syns.get('synonyms', {})
            if isinstance(c_s, dict):
                c_syn_set = set(c_s.items())
                g_syn_set = set(g_s.items())

    print(f"    Synonyms: Claude={len(c_syn_set)}, GPT={len(g_syn_set)}")
    if c_syn_set != g_syn_set:
        c_only = c_syn_set - g_syn_set
        g_only = g_syn_set - c_syn_set
        if c_only:
            print(f"      Claude-only synonyms: {sorted(c_only)}")
        if g_only:
            print(f"      GPT-only synonyms: {sorted(g_only)}")


def analyze_phase4(claude, gpt, ds):
    """ILinker2 seed — the foundation of everything."""
    print(f"\n  [Phase 4] ILinker2 Seed")
    if not claude or not gpt:
        print("    Missing checkpoint(s)")
        return

    c_links = claude.get('transarc_links', [])
    g_links = gpt.get('transarc_links', [])

    print(f"    Links: Claude={len(c_links)}, GPT={len(g_links)}")

    common, c_only, g_only = compare_links(c_links, g_links)
    print(f"    Common: {len(common)}, Claude-only: {len(c_only)}, GPT-only: {len(g_only)}")

    if c_only:
        print(f"    Claude-only links (GPT missed):")
        for sid, comp in sorted(c_only):
            print(f"      S{sid} -> {comp}")
    if g_only:
        print(f"    GPT-only links (Claude missed):")
        for sid, comp in sorted(g_only):
            print(f"      S{sid} -> {comp}")


def analyze_phase5(claude, gpt, ds):
    """Entity extraction — additional candidates found."""
    print(f"\n  [Phase 5] Entity Extraction")
    if not claude or not gpt:
        print("    Missing checkpoint(s)")
        return

    c_cands = claude.get('entity_candidates', claude.get('candidates', []))
    g_cands = gpt.get('entity_candidates', gpt.get('candidates', []))

    print(f"    Candidates: Claude={len(c_cands)}, GPT={len(g_cands)}")

    common, c_only, g_only = compare_links(c_cands, g_cands)
    print(f"    Common: {len(common)}, Claude-only: {len(c_only)}, GPT-only: {len(g_only)}")

    if c_only:
        print(f"    Claude-only candidates:")
        for sid, comp in sorted(c_only):
            print(f"      S{sid} -> {comp}")
    if g_only:
        print(f"    GPT-only candidates:")
        for sid, comp in sorted(g_only):
            print(f"      S{sid} -> {comp}")


def analyze_phase6(claude, gpt, ds):
    """Validation — which candidates survive."""
    print(f"\n  [Phase 6] Validation")
    if not claude or not gpt:
        print("    Missing checkpoint(s)")
        return

    c_valid = claude.get('validated_links', claude.get('validated', []))
    g_valid = gpt.get('validated_links', gpt.get('validated', []))

    print(f"    Validated: Claude={len(c_valid)}, GPT={len(g_valid)}")

    common, c_only, g_only = compare_links(c_valid, g_valid)
    print(f"    Common: {len(common)}, Claude-only: {len(c_only)}, GPT-only: {len(g_only)}")


def analyze_phase7(claude, gpt, ds):
    """Coreference resolution."""
    print(f"\n  [Phase 7] Coreference")
    if not claude or not gpt:
        print("    Missing checkpoint(s)")
        return

    c_coref = claude.get('coref_links', [])
    g_coref = gpt.get('coref_links', [])

    print(f"    Coref links: Claude={len(c_coref)}, GPT={len(g_coref)}")

    if c_coref or g_coref:
        common, c_only, g_only = compare_links(c_coref, g_coref)
        print(f"    Common: {len(common)}, Claude-only: {len(c_only)}, GPT-only: {len(g_only)}")


def analyze_prejuge(claude, gpt, ds):
    """Pre-judge state — the links going INTO Phase 9."""
    print(f"\n  [Pre-Judge] Links entering Phase 9")
    if not claude or not gpt:
        print("    Missing checkpoint(s)")
        return

    c_links = claude.get('preliminary', claude.get('preliminary_links', []))
    g_links = gpt.get('preliminary', gpt.get('preliminary_links', []))

    print(f"    Pre-judge links: Claude={len(c_links)}, GPT={len(g_links)}")

    common, c_only, g_only = compare_links(c_links, g_links)
    print(f"    Common: {len(common)}, Claude-only: {len(c_only)}, GPT-only: {len(g_only)}")

    if c_only:
        print(f"    Claude-only (GPT lost before judge):")
        for sid, comp in sorted(c_only):
            # Find source
            src = "unknown"
            for l in c_links:
                s = getattr(l, 'sentence_id', None) or getattr(l, 'sentence_number', None)
                c = getattr(l, 'component_name', None) or getattr(l, 'component', None)
                if s == sid and c == comp:
                    src = getattr(l, 'source', 'unknown')
                    break
            print(f"      S{sid} -> {comp} [{src}]")
    if g_only:
        print(f"    GPT-only (Claude lost before judge):")
        for sid, comp in sorted(g_only):
            src = "unknown"
            for l in g_links:
                s = getattr(l, 'sentence_id', None) or getattr(l, 'sentence_number', None)
                c = getattr(l, 'component_name', None) or getattr(l, 'component', None)
                if s == sid and c == comp:
                    src = getattr(l, 'source', 'unknown')
                    break
            print(f"      S{sid} -> {comp} [{src}]")


def analyze_final(claude, gpt, ds):
    """Final links — what came out of the pipeline."""
    print(f"\n  [Final] Output Links")
    if not claude or not gpt:
        print("    Missing checkpoint(s)")
        return

    c_links = claude.get('final', claude.get('final_links', []))
    g_links = gpt.get('final', gpt.get('final_links', []))

    print(f"    Final links: Claude={len(c_links)}, GPT={len(g_links)}")

    common, c_only, g_only = compare_links(c_links, g_links)
    print(f"    Common: {len(common)}, Claude-only: {len(c_only)}, GPT-only: {len(g_only)}")

    if c_only:
        print(f"    Claude-only final:")
        for sid, comp in sorted(c_only):
            src = "unknown"
            for l in c_links:
                s = getattr(l, 'sentence_id', None) or getattr(l, 'sentence_number', None)
                c = getattr(l, 'component_name', None) or getattr(l, 'component', None)
                if s == sid and c == comp:
                    src = getattr(l, 'source', 'unknown')
                    break
            print(f"      S{sid} -> {comp} [{src}]")
    if g_only:
        print(f"    GPT-only final:")
        for sid, comp in sorted(g_only):
            src = "unknown"
            for l in g_links:
                s = getattr(l, 'sentence_id', None) or getattr(l, 'sentence_number', None)
                c = getattr(l, 'component_name', None) or getattr(l, 'component', None)
                if s == sid and c == comp:
                    src = getattr(l, 'source', 'unknown')
                    break
            print(f"      S{sid} -> {comp} [{src}]")


def main():
    # First, peek at checkpoint structure
    print("=" * 100)
    print("CHECKPOINT STRUCTURE INSPECTION")
    print("=" * 100)

    sample = load_pkl(f"{CLAUDE_DIR}/mediastore/phase4.pkl")
    if sample:
        print(f"\nPhase 4 keys: {sorted(sample.keys()) if isinstance(sample, dict) else 'NOT A DICT: ' + str(type(sample))}")
        if isinstance(sample, dict):
            for k, v in sample.items():
                if isinstance(v, list) and len(v) > 0:
                    print(f"  {k}: list of {len(v)} x {type(v[0]).__name__}")
                    # Show first element's attributes
                    elem = v[0]
                    if hasattr(elem, '__dict__'):
                        print(f"    attrs: {list(elem.__dict__.keys())}")
                elif isinstance(v, dict):
                    print(f"  {k}: dict with {len(v)} keys")
                elif isinstance(v, set):
                    print(f"  {k}: set of {len(v)}")
                else:
                    print(f"  {k}: {type(v).__name__} = {v}")

    # Check all phase structures
    for phase_name in ["phase0", "phase1", "phase3", "phase5", "phase6", "phase7", "pre_judge", "final"]:
        p = load_pkl(f"{CLAUDE_DIR}/mediastore/{phase_name}.pkl")
        if p and isinstance(p, dict):
            print(f"\n{phase_name} keys: {sorted(p.keys())}")
        elif p:
            print(f"\n{phase_name}: {type(p).__name__}")

    print("\n" + "=" * 100)
    print("PHASE-BY-PHASE FAULT ANALYSIS: Claude V32 vs GPT V32")
    print("=" * 100)

    for ds in DATASETS:
        print(f"\n{'=' * 80}")
        print(f"DATASET: {ds} (gold={GOLD[ds]})")
        print(f"{'=' * 80}")

        # Load all phases
        phases = {}
        for phase_name in ["phase0", "phase1", "phase2", "phase3", "phase4", "phase5", "phase6", "phase7", "pre_judge", "final"]:
            c = load_pkl(f"{CLAUDE_DIR}/{ds}/{phase_name}.pkl")
            g = load_pkl(f"{GPT_DIR}/{ds}/{phase_name}.pkl")
            phases[phase_name] = (c, g)

        analyze_phase0(*phases["phase0"], ds)
        analyze_phase1(*phases["phase1"], ds)
        analyze_phase3(*phases["phase3"], ds)
        analyze_phase4(*phases["phase4"], ds)
        analyze_phase5(*phases["phase5"], ds)
        analyze_phase6(*phases["phase6"], ds)
        analyze_phase7(*phases["phase7"], ds)
        analyze_prejuge(*phases["pre_judge"], ds)
        analyze_final(*phases["final"], ds)

    # Summary statistics
    print("\n" + "=" * 100)
    print("AGGREGATE FAULT SUMMARY")
    print("=" * 100)

    total_phase4_diff = 0
    total_final_diff = 0

    for ds in DATASETS:
        c4 = load_pkl(f"{CLAUDE_DIR}/{ds}/phase4.pkl")
        g4 = load_pkl(f"{GPT_DIR}/{ds}/phase4.pkl")
        cf = load_pkl(f"{CLAUDE_DIR}/{ds}/final.pkl")
        gf = load_pkl(f"{GPT_DIR}/{ds}/final.pkl")

        if c4 and g4:
            c_links = c4.get('transarc_links', c4.get('ilinker2_links', []))
            g_links = g4.get('transarc_links', g4.get('ilinker2_links', []))
            _, c_only, g_only = compare_links(c_links, g_links)
            diff4 = len(c_only) + len(g_only)
            total_phase4_diff += diff4

        if cf and gf:
            c_links = cf.get('final_links', cf.get('links', []))
            g_links = gf.get('final_links', gf.get('links', []))
            _, c_only, g_only = compare_links(c_links, g_links)
            diff_f = len(c_only) + len(g_only)
            total_final_diff += diff_f
            print(f"  {ds:20s}: Phase4 symmetric diff={diff4:3d}, Final symmetric diff={diff_f:3d}")

    print(f"  {'TOTAL':20s}: Phase4={total_phase4_diff}, Final={total_final_diff}")


if __name__ == "__main__":
    main()
