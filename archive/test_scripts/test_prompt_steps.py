#!/usr/bin/env python3
"""Step-level test: Run each LLM-calling phase N times to measure variance.

Tests the phases that had prompt changes (Phase 1, 3, 5, 6, 7, 9) independently
to verify prompt improvements don't regress and measure output stability.

Usage:
    python test_prompt_steps.py                           # all phases, mediastore, 3 runs
    python test_prompt_steps.py --phases 1 3 --runs 5     # specific phases, 5 runs
    python test_prompt_steps.py --dataset jabref           # different dataset
"""

import csv
import json
import os
import re
import sys
import time
import argparse
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.core import DocumentLoader, SadSamLink, CandidateLink, ModelKnowledge

os.environ["CLAUDE_MODEL"] = "sonnet"

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
CLI_RESULTS = Path("/mnt/hostshare/ardoco-home/cli-results")

DATASETS = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
        "gold_sam": BENCHMARK_BASE / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
        "transarc_sam": CLI_RESULTS / "mediastore-sad-sam/sadSamTlr_mediastore.csv",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
        "model": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
        "gold_sam": BENCHMARK_BASE / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
        "transarc_sam": CLI_RESULTS / "teastore-sad-sam/sadSamTlr_teastore.csv",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
        "gold_sam": BENCHMARK_BASE / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc_sam": CLI_RESULTS / "jabref-sad-sam/sadSamTlr_jabref.csv",
    },
}


def load_gold_sam(gold_path):
    links = set()
    with open(gold_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def load_data(dataset_name):
    """Load all data needed for testing."""
    paths = DATASETS[dataset_name]
    components = parse_pcm_repository(str(paths["model"]))
    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    name_to_id = {c.name: c.id for c in components}
    id_to_name = {c.id: c.name for c in components}
    sent_map = DocumentLoader.build_sent_map(sentences)
    gold = load_gold_sam(str(paths["gold_sam"]))
    return components, sentences, name_to_id, id_to_name, sent_map, gold, paths


def create_linker():
    """Create a fresh linker instance (V25 by default, V24 with --v24 flag)."""
    import sys
    if "--v24" in sys.argv:
        from llm_sad_sam.linkers.experimental.agent_linker_v24 import AgentLinkerV24
        return AgentLinkerV24(backend=LLMBackend.CLAUDE)
    from llm_sad_sam.linkers.experimental.agent_linker_v25 import AgentLinkerV25
    return AgentLinkerV25(backend=LLMBackend.CLAUDE)


def setup_linker_prereqs(linker, components, sentences, sent_map, paths):
    """Run Phase 0 (deterministic) to set up prerequisites."""
    linker._cached_sent_map = sent_map
    linker.doc_profile = linker._learn_document_profile(sentences, components)
    linker._is_complex = linker._structural_complexity(sentences, components)
    return linker


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Model Structure Analysis
# ═══════════════════════════════════════════════════════════════════

def test_phase1(linker, components, **_):
    """Phase 1: _analyze_model — classifies components as architectural vs ambiguous."""
    mk = linker._analyze_model(components)
    linker.model_knowledge = mk

    # Populate dynamic sets (same as in link())
    ambig = mk.ambiguous_names
    linker.GENERIC_COMPONENT_WORDS = set()
    for name in ambig:
        if ' ' not in name and not name.isupper():
            linker.GENERIC_COMPONENT_WORDS.add(name.lower())

    linker.GENERIC_PARTIALS = set()
    for comp in components:
        parts = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', comp.name)
        for part in parts:
            p_lower = part.lower()
            if part.isupper():
                continue
            if len(p_lower) >= 3 and p_lower in ambig or any(
                p_lower == a.lower() for a in ambig
            ):
                linker.GENERIC_PARTIALS.add(p_lower)
    for name in ambig:
        if ' ' not in name and not name.isupper():
            linker.GENERIC_PARTIALS.add(name.lower())

    return {
        "architectural": sorted(mk.architectural_names),
        "ambiguous": sorted(mk.ambiguous_names),
        "generic_words": sorted(linker.GENERIC_COMPONENT_WORDS),
        "generic_partials": sorted(linker.GENERIC_PARTIALS),
    }


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Document Knowledge
# ═══════════════════════════════════════════════════════════════════

def test_phase3(linker, components, sentences, **_):
    """Phase 3: _learn_document_knowledge_enriched — finds abbreviations, synonyms, partials."""
    dk = linker._learn_document_knowledge_enriched(sentences, components)
    linker.doc_knowledge = dk
    linker._enrich_multiword_partials(sentences, components)

    return {
        "abbreviations": dict(dk.abbreviations),
        "synonyms": dict(dk.synonyms),
        "partials": dict(dk.partial_references),
        "generic_rejected": sorted(dk.generic_terms),
    }


# ═══════════════════════════════════════════════════════════════════
# Phase 5: Entity Extraction
# ═══════════════════════════════════════════════════════════════════

def test_phase5(linker, components, sentences, name_to_id, sent_map, **_):
    """Phase 5: _extract_entities_enriched — finds component references in text."""
    candidates = linker._extract_entities_enriched(sentences, components, name_to_id, sent_map)
    return {
        "count": len(candidates),
        "links": sorted([(c.sentence_number, c.component_name) for c in candidates]),
    }


# ═══════════════════════════════════════════════════════════════════
# Phase 6: Validation
# ═══════════════════════════════════════════════════════════════════

def test_phase6(linker, components, sentences, name_to_id, sent_map, **_):
    """Phase 6: _validate_intersect — validates candidates via 2-pass + evidence."""
    # Need Phase 5 output first
    candidates = linker._extract_entities_enriched(sentences, components, name_to_id, sent_map)
    candidates = linker._apply_abbreviation_guard_to_candidates(candidates, sent_map)
    validated = linker._validate_intersect(candidates, components, sent_map)
    return {
        "candidates": len(candidates),
        "validated": len(validated),
        "links": sorted([(c.sentence_number, c.component_name) for c in validated]),
    }


# ═══════════════════════════════════════════════════════════════════
# Phase 9: Judge Review (tests the 4-rule prompt)
# ═══════════════════════════════════════════════════════════════════

def test_phase9(linker, components, sentences, name_to_id, id_to_name, sent_map, paths, **_):
    """Phase 9: _judge_review — reviews all links with 4-rule judge."""
    # Build full link set through phases 4-8 (deterministic where possible)
    transarc_links = linker._process_transarc(
        str(paths["transarc_sam"]), id_to_name, sent_map, name_to_id
    )
    transarc_set = {(l.sentence_number, l.component_id) for l in transarc_links}

    candidates = linker._extract_entities_enriched(sentences, components, name_to_id, sent_map)
    candidates = linker._apply_abbreviation_guard_to_candidates(candidates, sent_map)
    validated = linker._validate_intersect(candidates, components, sent_map)

    entity_links = [
        SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
        for c in validated
    ]

    # Minimal coref + partial for realistic judge input
    if linker._is_complex:
        coref_links = linker._coref_debate(sentences, components, name_to_id, sent_map)
    else:
        discourse_model = linker._build_discourse_model(sentences, components, name_to_id)
        coref_links = linker._coref_discourse(sentences, components, name_to_id, sent_map, discourse_model)
    coref_links = linker._filter_generic_coref(coref_links, sent_map)

    partial_links = linker._inject_partial_references(
        sentences, components, name_to_id, transarc_set,
        {(c.sentence_number, c.component_id) for c in validated},
        {(l.sentence_number, l.component_id) for l in coref_links},
        set(),
    )

    all_links = transarc_links + entity_links + coref_links + partial_links
    link_map = {}
    for lk in all_links:
        key = (lk.sentence_number, lk.component_id)
        if key not in link_map:
            link_map[key] = lk
        else:
            old_p = linker.SOURCE_PRIORITY.get(link_map[key].source, 0)
            new_p = linker.SOURCE_PRIORITY.get(lk.source, 0)
            if new_p > old_p:
                link_map[key] = lk
    preliminary = list(link_map.values())

    preliminary, _ = linker._apply_boundary_filters(preliminary, sent_map, transarc_set)
    reviewed = linker._judge_review(preliminary, sentences, components, sent_map, transarc_set)

    return {
        "input": len(preliminary),
        "approved": len(reviewed),
        "rejected": len(preliminary) - len(reviewed),
        "links": sorted([(l.sentence_number, l.component_name) for l in reviewed]),
    }


# ═══════════════════════════════════════════════════════════════════
# Main test runner
# ═══════════════════════════════════════════════════════════════════

PHASE_TESTS = {
    1: ("Model Structure (ambiguity classification)", test_phase1),
    3: ("Document Knowledge (abbreviations, synonyms)", test_phase3),
    5: ("Entity Extraction (component references)", test_phase5),
    6: ("Validation (2-pass + evidence)", test_phase6),
    9: ("Judge Review (4-rule prompt)", test_phase9),
}


def run_phase_test(phase_num, dataset_name, n_runs):
    """Run a single phase test N times and report variance."""
    phase_label, test_fn = PHASE_TESTS[phase_num]
    print(f"\n{'='*70}")
    print(f"PHASE {phase_num}: {phase_label}")
    print(f"Dataset: {dataset_name}, Runs: {n_runs}")
    print(f"{'='*70}")

    components, sentences, name_to_id, id_to_name, sent_map, gold, paths = load_data(dataset_name)

    results = []
    for run in range(n_runs):
        print(f"\n--- Run {run+1}/{n_runs} ---")
        t0 = time.time()

        linker = create_linker()
        setup_linker_prereqs(linker, components, sentences, sent_map, paths)

        # For phases > 1, need model knowledge from Phase 1
        if phase_num > 1:
            linker.model_knowledge = linker._analyze_model(components)
            ambig = linker.model_knowledge.ambiguous_names
            linker.GENERIC_COMPONENT_WORDS = set()
            for name in ambig:
                if ' ' not in name and not name.isupper():
                    linker.GENERIC_COMPONENT_WORDS.add(name.lower())
            linker.GENERIC_PARTIALS = set()
            for comp in components:
                parts = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', comp.name)
                for part in parts:
                    p_lower = part.lower()
                    if part.isupper():
                        continue
                    if len(p_lower) >= 3 and p_lower in ambig or any(
                        p_lower == a.lower() for a in ambig
                    ):
                        linker.GENERIC_PARTIALS.add(p_lower)
            for name in ambig:
                if ' ' not in name and not name.isupper():
                    linker.GENERIC_PARTIALS.add(name.lower())

        # For phases > 2, need patterns
        if phase_num > 2:
            linker.learned_patterns = linker._learn_patterns_with_debate(sentences, components)

        # For phases > 3, need doc knowledge
        if phase_num > 3:
            linker.doc_knowledge = linker._learn_document_knowledge_enriched(sentences, components)
            linker._enrich_multiword_partials(sentences, components)

        result = test_fn(
            linker=linker, components=components, sentences=sentences,
            name_to_id=name_to_id, id_to_name=id_to_name, sent_map=sent_map,
            gold=gold, paths=paths,
        )
        elapsed = time.time() - t0

        # Compute metrics if phase produces links
        if "links" in result:
            link_tuples = result["links"]
            # Convert (snum, comp_name) to (snum, comp_id) for eval
            pred_pairs = set()
            for snum, cname in link_tuples:
                cid = name_to_id.get(cname)
                if cid:
                    pred_pairs.add((snum, cid))

            tp = len(pred_pairs & gold)
            fp = len(pred_pairs - gold)
            fn = len(gold - pred_pairs)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            result["metrics"] = {"P": p, "R": r, "F1": f1, "TP": tp, "FP": fp, "FN": fn}
            print(f"  P={p:.1%} R={r:.1%} F1={f1:.1%} TP={tp} FP={fp} FN={fn} ({elapsed:.0f}s)")
        else:
            print(f"  Result: {json.dumps(result, indent=2)[:200]} ({elapsed:.0f}s)")

        result["elapsed"] = elapsed
        results.append(result)

    # ── Variance analysis ──
    print(f"\n{'─'*70}")
    print(f"VARIANCE ANALYSIS — Phase {phase_num} ({n_runs} runs)")
    print(f"{'─'*70}")

    if all("metrics" in r for r in results):
        f1s = [r["metrics"]["F1"] for r in results]
        fps = [r["metrics"]["FP"] for r in results]
        fns = [r["metrics"]["FN"] for r in results]
        tps = [r["metrics"]["TP"] for r in results]

        import statistics
        mean_f1 = statistics.mean(f1s)
        std_f1 = statistics.stdev(f1s) if len(f1s) > 1 else 0
        min_f1 = min(f1s)
        max_f1 = max(f1s)

        print(f"  F1: mean={mean_f1:.3f} std={std_f1:.3f} range=[{min_f1:.3f}, {max_f1:.3f}]")
        print(f"  TP: {tps}")
        print(f"  FP: {fps}")
        print(f"  FN: {fns}")

        # Check stability
        if std_f1 < 0.02:
            print(f"  STABLE (std < 0.02)")
        elif std_f1 < 0.05:
            print(f"  MODERATE VARIANCE (std < 0.05)")
        else:
            print(f"  HIGH VARIANCE (std >= 0.05) — INVESTIGATE")
    else:
        # For Phase 1: check if ambiguous sets are consistent
        all_ambig = [set(r.get("ambiguous", [])) for r in results]
        all_generic = [set(r.get("generic_words", [])) for r in results]

        if all(a == all_ambig[0] for a in all_ambig):
            print(f"  Ambiguous names: STABLE — {sorted(all_ambig[0])}")
        else:
            print(f"  Ambiguous names: VARIES")
            for i, a in enumerate(all_ambig):
                print(f"    Run {i+1}: {sorted(a)}")

        if all(g == all_generic[0] for g in all_generic):
            print(f"  Generic words: STABLE — {sorted(all_generic[0])}")
        else:
            print(f"  Generic words: VARIES")
            for i, g in enumerate(all_generic):
                print(f"    Run {i+1}: {sorted(g)}")

    # Link-set Jaccard similarity across runs
    if all("links" in r for r in results):
        link_sets = [set(tuple(l) for l in r["links"]) for r in results]
        if len(link_sets) > 1:
            # Pairwise Jaccard
            jaccards = []
            for i in range(len(link_sets)):
                for j in range(i+1, len(link_sets)):
                    inter = len(link_sets[i] & link_sets[j])
                    union = len(link_sets[i] | link_sets[j])
                    jaccards.append(inter / union if union > 0 else 1.0)
            mean_j = statistics.mean(jaccards)
            print(f"  Link-set Jaccard: mean={mean_j:.3f} (1.0 = identical across runs)")

            # Show unstable links (not in all runs)
            all_links = set()
            for ls in link_sets:
                all_links |= ls
            stable = set.intersection(*link_sets) if link_sets else set()
            unstable = all_links - stable
            if unstable:
                print(f"  Unstable links ({len(unstable)}):")
                for snum, cname in sorted(unstable):
                    presence = sum(1 for ls in link_sets if (snum, cname) in ls)
                    print(f"    S{snum} -> {cname} ({presence}/{n_runs} runs)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Step-level prompt variance test")
    parser.add_argument("--phases", nargs="+", type=int, default=list(PHASE_TESTS.keys()),
                        help="Phases to test (default: all)")
    parser.add_argument("--dataset", default="mediastore", choices=list(DATASETS.keys()))
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per phase")
    args = parser.parse_args()

    print("=" * 70)
    print(f"STEP-LEVEL PROMPT VARIANCE TEST")
    print(f"Dataset: {args.dataset}, Phases: {args.phases}, Runs: {args.runs}")
    print("=" * 70)

    all_results = {}
    for phase in args.phases:
        if phase not in PHASE_TESTS:
            print(f"\nSkipping unknown phase {phase}")
            continue
        all_results[phase] = run_phase_test(phase, args.dataset, args.runs)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for phase, results in all_results.items():
        label = PHASE_TESTS[phase][0]
        if all("metrics" in r for r in results):
            f1s = [r["metrics"]["F1"] for r in results]
            import statistics
            mean_f1 = statistics.mean(f1s)
            std_f1 = statistics.stdev(f1s) if len(f1s) > 1 else 0
            status = "STABLE" if std_f1 < 0.02 else ("MODERATE" if std_f1 < 0.05 else "HIGH VAR")
            print(f"  Phase {phase} ({label}): F1={mean_f1:.3f} +/- {std_f1:.3f} [{status}]")
        else:
            print(f"  Phase {phase} ({label}): see details above")


if __name__ == "__main__":
    main()
