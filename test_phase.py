#!/usr/bin/env python3
"""Test individual pipeline phases with real LLM calls.

Runs the linker phase-by-phase, evaluating after each phase to identify
which phase produces bad output.

Usage:
    python test_phase.py --dataset teastore --linker v16
    python test_phase.py --dataset teastore --linker v16 --stop-after 6
    python test_phase.py --dataset jabref --linker v15
"""

import csv
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.core import DocumentLoader, SadSamLink, CandidateLink

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
    "teammates": {
        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
        "gold_sam": BENCHMARK_BASE / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc_sam": CLI_RESULTS / "teammates-sad-sam/sadSamTlr_teammates.csv",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
        "gold_sam": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc_sam": CLI_RESULTS / "bigbluebutton-sad-sam/sadSamTlr_bigbluebutton.csv",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
        "gold_sam": BENCHMARK_BASE / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc_sam": CLI_RESULTS / "jabref-sad-sam/sadSamTlr_jabref.csv",
    },
}

os.environ["CLAUDE_MODEL"] = "sonnet"


def load_gold_sam(gold_path):
    links = set()
    with open(gold_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def eval_metrics(predicted, gold):
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"tp": tp, "fp": fp, "fn": fn, "P": p, "R": r, "F1": f1}


def print_eval(phase_name, link_list, gold, id_to_name, sent_map, prev_pairs=None):
    """Print evaluation for current cumulative link set."""
    pairs = {(l.sentence_number, l.component_id) for l in link_list}
    m = eval_metrics(pairs, gold)
    print(f"\n  >> EVAL after {phase_name}: P={m['P']:.1%} R={m['R']:.1%} F1={m['F1']:.1%} "
          f"(TP={m['tp']} FP={m['fp']} FN={m['fn']}) [{len(link_list)} links]")

    # New FPs added by this phase
    if prev_pairs is not None:
        new_fps = (pairs - gold) - (prev_pairs - gold)
        if new_fps:
            print(f"     New FPs from {phase_name}:")
            for sn, cid in sorted(new_fps):
                cname = id_to_name.get(cid, cid[:20])
                sent = sent_map.get(sn)
                txt = sent.text[:80] if sent else "???"
                # find the source
                src = "?"
                for l in link_list:
                    if l.sentence_number == sn and l.component_id == cid:
                        src = l.source
                        break
                print(f"       S{sn} -> {cname} [{src}] \"{txt}\"")

    # Remaining FNs (show first 10)
    fns = gold - pairs
    if fns:
        shown = sorted(fns)[:10]
        print(f"     FNs remaining ({len(fns)}):")
        for sn, cid in shown:
            cname = id_to_name.get(cid, cid[:20])
            sent = sent_map.get(sn)
            txt = sent.text[:80] if sent else "???"
            print(f"       S{sn} -> {cname} \"{txt}\"")
        if len(fns) > 10:
            print(f"       ... and {len(fns) - 10} more")

    return pairs


def create_linker(linker_name):
    """Create linker instance by name."""
    if linker_name == "w16":
        from llm_sad_sam.linkers.experimental.agent_linker_w16 import AgentLinkerW16
        return AgentLinkerW16(backend=LLMBackend.CLAUDE)
    elif linker_name == "v16":
        from llm_sad_sam.linkers.experimental.agent_linker_v16 import AgentLinkerV16
        return AgentLinkerV16(backend=LLMBackend.CLAUDE)
    elif linker_name == "v15":
        from llm_sad_sam.linkers.experimental.agent_linker_v15 import AgentLinkerV15
        return AgentLinkerV15(backend=LLMBackend.CLAUDE)
    elif linker_name == "v18":
        from llm_sad_sam.linkers.experimental.agent_linker_v18 import AgentLinkerV18
        return AgentLinkerV18(backend=LLMBackend.CLAUDE)
    elif linker_name == "v19":
        from llm_sad_sam.linkers.experimental.agent_linker_v19 import AgentLinkerV19
        return AgentLinkerV19(backend=LLMBackend.CLAUDE)
    elif linker_name == "v20":
        from llm_sad_sam.linkers.experimental.agent_linker_v20 import AgentLinkerV20
        return AgentLinkerV20(backend=LLMBackend.CLAUDE)
    elif linker_name == "v23":
        from llm_sad_sam.linkers.experimental.agent_linker_v23 import AgentLinkerV23
        return AgentLinkerV23(backend=LLMBackend.CLAUDE)
    elif linker_name == "v23a":
        from llm_sad_sam.linkers.experimental.agent_linker_v23a import AgentLinkerV23a
        return AgentLinkerV23a(backend=LLMBackend.CLAUDE)
    elif linker_name == "v23b":
        from llm_sad_sam.linkers.experimental.agent_linker_v23b import AgentLinkerV23b
        return AgentLinkerV23b(backend=LLMBackend.CLAUDE)
    elif linker_name == "v23c":
        from llm_sad_sam.linkers.experimental.agent_linker_v23c import AgentLinkerV23c
        return AgentLinkerV23c(backend=LLMBackend.CLAUDE)
    elif linker_name == "v23d":
        from llm_sad_sam.linkers.experimental.agent_linker_v23d import AgentLinkerV23d
        return AgentLinkerV23d(backend=LLMBackend.CLAUDE)
    elif linker_name == "v23e":
        from llm_sad_sam.linkers.experimental.agent_linker_v23e import AgentLinkerV23e
        return AgentLinkerV23e(backend=LLMBackend.CLAUDE)
    elif linker_name == "v22":
        from llm_sad_sam.linkers.experimental.agent_linker_v22 import AgentLinkerV22
        return AgentLinkerV22(backend=LLMBackend.CLAUDE)
    elif linker_name == "v21":
        from llm_sad_sam.linkers.experimental.agent_linker_v21 import AgentLinkerV21
        return AgentLinkerV21(backend=LLMBackend.CLAUDE)
    elif linker_name == "v20d":
        from llm_sad_sam.linkers.experimental.agent_linker_v20d import AgentLinkerV20d
        return AgentLinkerV20d(backend=LLMBackend.CLAUDE)
    elif linker_name == "v20e":
        from llm_sad_sam.linkers.experimental.agent_linker_v20e import AgentLinkerV20e
        return AgentLinkerV20e(backend=LLMBackend.CLAUDE)
    elif linker_name == "v6":
        from llm_sad_sam.linkers.experimental.agent_linker_v6 import AgentLinkerV6
        return AgentLinkerV6(backend=LLMBackend.CLAUDE)
    else:
        raise ValueError(f"Unknown linker: {linker_name}. Supported: v20, w16, v19, v18, v16, v15, v6")


def run_phases(dataset_name, linker_name, stop_after=None):
    paths = DATASETS[dataset_name]
    gold = load_gold_sam(str(paths["gold_sam"]))

    # Load data
    components = parse_pcm_repository(str(paths["model"]))
    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    name_to_id = {c.name: c.id for c in components}
    id_to_name = {c.id: c.name for c in components}
    sent_map = DocumentLoader.build_sent_map(sentences)

    print(f"{'='*80}")
    print(f"PER-PHASE TEST: {dataset_name} with {linker_name}")
    print(f"Components: {len(components)}, Sentences: {len(sentences)}, Gold: {len(gold)}")
    print(f"{'='*80}")

    # Create linker
    linker = create_linker(linker_name)
    linker._cached_sent_map = sent_map

    t0 = time.time()
    prev_pairs = set()

    # ── Phase 0: Document profile ──
    print("\n[Phase 0] Document Profile (no LLM)")
    linker.doc_profile = linker._learn_document_profile(sentences, components)
    linker._is_complex = linker._structural_complexity(sentences, components)
    from llm_sad_sam.core import LearnedThresholds
    linker.thresholds = LearnedThresholds(0, 0, 0, 0, "qualitative", 0)
    print(f"  Complex: {linker._is_complex}")
    if stop_after == 0:
        return

    # ── Phase 1: Model Structure (LLM) ──
    print("\n[Phase 1] Model Structure (LLM)")
    linker.model_knowledge = linker._analyze_model(components)
    arch = linker.model_knowledge.architectural_names
    ambig = linker.model_knowledge.ambiguous_names
    print(f"  Architectural ({len(arch)}): {sorted(arch)}")
    print(f"  Ambiguous ({len(ambig)}): {sorted(ambig)}")
    if stop_after == 1:
        return

    # ── Phase 2: Pattern Learning (LLM) ──
    print("\n[Phase 2] Pattern Learning (LLM)")
    linker.learned_patterns = linker._learn_patterns_with_debate(sentences, components)
    print(f"  Subprocess terms: {sorted(linker.learned_patterns.subprocess_terms)}")
    print(f"  Action indicators: {linker.learned_patterns.action_indicators[:5]}")
    if stop_after == 2:
        return

    # ── Phase 3: Document Knowledge (LLM) ──
    print("\n[Phase 3] Document Knowledge (LLM)")
    if hasattr(linker, '_learn_document_knowledge_enriched'):
        linker.doc_knowledge = linker._learn_document_knowledge_enriched(sentences, components)
    else:
        linker.doc_knowledge = linker._learn_document_knowledge(sentences, components)
    dk = linker.doc_knowledge
    print(f"  Abbreviations: {dk.abbreviations}")
    print(f"  Synonyms: {dk.synonyms}")
    print(f"  Partials: {dk.partial_references}")
    if hasattr(dk, 'generic_terms') and dk.generic_terms:
        print(f"  Generic (rejected): {dk.generic_terms}")
    if stop_after == 3:
        return

    # ── Phase 3b: Multi-word partial enrichment (no LLM) ──
    print("\n[Phase 3b] Multi-word Partial Enrichment (no LLM)")
    before = dict(dk.partial_references)
    linker._enrich_multiword_partials(sentences, components)
    added = {k: v for k, v in dk.partial_references.items() if k not in before}
    if added:
        print(f"  Added partials: {added}")
    else:
        print(f"  No new partials")

    # ── Phase 4: TransArc baseline (no LLM) ──
    print("\n[Phase 4] TransArc Baseline (no LLM)")
    transarc_links = linker._process_transarc(
        str(paths["transarc_sam"]), id_to_name, sent_map, name_to_id
    )
    transarc_set = {(l.sentence_number, l.component_id) for l in transarc_links}
    print(f"  TransArc links: {len(transarc_links)}")
    prev_pairs = print_eval("Phase 4 (TransArc)", transarc_links, gold, id_to_name, sent_map)
    if stop_after == 4:
        return

    # ── Phase 5: Entity Extraction (LLM) ──
    print("\n[Phase 5] Entity Extraction (LLM)")
    if hasattr(linker, '_extract_entities_enriched'):
        candidates = linker._extract_entities_enriched(sentences, components, name_to_id, sent_map)
    else:
        candidates = linker._extract_entities(sentences, components, name_to_id, sent_map)
    print(f"  Raw candidates: {len(candidates)}")

    # Abbreviation guard
    before_guard = len(candidates)
    candidates = linker._apply_abbreviation_guard_to_candidates(candidates, sent_map)
    if len(candidates) < before_guard:
        print(f"  After abbrev guard: {len(candidates)} (-{before_guard - len(candidates)})")

    # Eval: transarc + entity candidates (before validation)
    entity_as_links = [
        SadSamLink(c.sentence_number, c.component_id, c.component_name, 0.85, "entity")
        for c in candidates
    ]
    prev_pairs = print_eval("Phase 5 (TransArc + raw entity)",
                           transarc_links + entity_as_links, gold, id_to_name, sent_map, prev_pairs)
    if stop_after == 5:
        return

    # ── Phase 5b: Targeted recovery (LLM) ──
    entity_comps = {c.component_name for c in candidates}
    transarc_comps = {l.component_name for l in transarc_links}
    covered_comps = entity_comps | transarc_comps
    unlinked = [c for c in components if c.name not in covered_comps]

    extra = []
    if unlinked:
        print(f"\n[Phase 5b] Targeted Recovery (LLM) — {len(unlinked)} unlinked: {[c.name for c in unlinked]}")
        try:
            extra = linker._targeted_extraction(unlinked, sentences, name_to_id, sent_map,
                                                components=components, transarc_links=transarc_links,
                                                entity_candidates=candidates) or []
        except TypeError:
            extra = linker._targeted_extraction(unlinked, sentences, name_to_id, sent_map) or []
        candidates.extend(extra)
        print(f"  Found: {len(extra)} additional candidates")

        extra_as_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, 0.85, "entity")
            for c in extra
        ]
        prev_pairs = print_eval("Phase 5b (+ targeted)",
                               transarc_links + entity_as_links + extra_as_links,
                               gold, id_to_name, sent_map, prev_pairs)
    else:
        print("\n[Phase 5b] Targeted Recovery — no unlinked components")

    if stop_after is not None and stop_after <= 5.5:
        return

    # ── Phase 6: Validation (LLM) ──
    print("\n[Phase 6] Validation (LLM)")
    if hasattr(linker, '_validate_intersect'):
        validated = linker._validate_intersect(candidates, components, sent_map)
    elif hasattr(linker, '_validate_split'):
        validated = linker._validate_split(candidates, components, sent_map)
    else:
        validated = linker._validate_candidates(candidates, components, sent_map)
    print(f"  Validated: {len(validated)} of {len(candidates)}")

    entity_links = [
        SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
        for c in validated
    ]
    prev_pairs = print_eval("Phase 6 (TransArc + validated)",
                           transarc_links + entity_links, gold, id_to_name, sent_map, prev_pairs)
    if stop_after == 6:
        return

    # ── Phase 7: Coreference (LLM) ──
    print("\n[Phase 7] Coreference (LLM)")
    if linker._is_complex:
        print(f"  Mode: debate")
        coref_links = linker._coref_debate(sentences, components, name_to_id, sent_map)
    else:
        discourse_model = linker._build_discourse_model(sentences, components, name_to_id)
        print(f"  Mode: discourse")
        coref_links = linker._coref_discourse(sentences, components, name_to_id, sent_map, discourse_model)

    before_filter = len(coref_links)
    try:
        existing_for_antecedent = transarc_links + [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
            for c in validated
        ]
        coref_links = linker._filter_generic_coref(coref_links, sent_map, existing_for_antecedent)
    except TypeError:
        coref_links = linker._filter_generic_coref(coref_links, sent_map)
    if len(coref_links) < before_filter:
        print(f"  After generic filter: {len(coref_links)} (-{before_filter - len(coref_links)})")
    print(f"  Coref links: {len(coref_links)}")

    prev_pairs = print_eval("Phase 7 (+ coref)",
                           transarc_links + entity_links + coref_links, gold, id_to_name, sent_map, prev_pairs)
    if stop_after == 7:
        return

    # ── Phase 8: Implicit — SKIPPED ──
    print("\n[Phase 8] Implicit References — SKIPPED")
    implicit_links = []

    # ── Phase 8b: Partial injection (no LLM) ──
    print("\n[Phase 8b] Partial Injection (no LLM)")
    partial_links = linker._inject_partial_references(
        sentences, components, name_to_id, transarc_set,
        {(c.sentence_number, c.component_id) for c in validated},
        {(l.sentence_number, l.component_id) for l in coref_links},
        set(),
    )
    print(f"  Injected: {len(partial_links)}")

    all_pre_dedup = transarc_links + entity_links + coref_links + implicit_links + partial_links
    prev_pairs = print_eval("Phase 8b (+ partials)",
                           all_pre_dedup, gold, id_to_name, sent_map, prev_pairs)

    # ── Dedup ──
    print("\n[Dedup]")
    link_map = {}
    for lk in all_pre_dedup:
        key = (lk.sentence_number, lk.component_id)
        if key not in link_map:
            link_map[key] = lk
        else:
            old_p = linker.SOURCE_PRIORITY.get(link_map[key].source, 0)
            new_p = linker.SOURCE_PRIORITY.get(lk.source, 0)
            if new_p > old_p:
                link_map[key] = lk
    preliminary = list(link_map.values())
    print(f"  Raw: {len(all_pre_dedup)} -> Deduped: {len(preliminary)}")
    prev_pairs = print_eval("Dedup", preliminary, gold, id_to_name, sent_map, prev_pairs)

    # ── Phase 8c: Boundary filters (no LLM) ──
    print("\n[Phase 8c] Boundary Filters (no LLM)")
    preliminary, boundary_rejected = linker._apply_boundary_filters(
        preliminary, sent_map, transarc_set
    )
    if boundary_rejected:
        print(f"  Rejected: {len(boundary_rejected)}")
        for lk, reason in boundary_rejected:
            print(f"    [{reason}] S{lk.sentence_number} -> {lk.component_name} ({lk.source})")
    prev_pairs = print_eval("Phase 8c (boundary filtered)", preliminary, gold, id_to_name, sent_map, prev_pairs)
    if stop_after is not None and stop_after <= 8:
        return

    # ── Phase 9: Judge Review (LLM) ──
    print("\n[Phase 9] Judge Review (LLM)")
    reviewed = linker._judge_review(preliminary, sentences, components, sent_map, transarc_set)
    rejected = [l for l in preliminary if (l.sentence_number, l.component_id)
                not in {(r.sentence_number, r.component_id) for r in reviewed}]
    print(f"  Approved: {len(reviewed)}, Rejected: {len(rejected)}")
    if rejected:
        for l in rejected:
            sent = sent_map.get(l.sentence_number)
            txt = sent.text[:60] if sent else "???"
            print(f"    Rejected: S{l.sentence_number} -> {l.component_name} [{l.source}] \"{txt}\"")

    prev_pairs = print_eval("Phase 9 (final)", reviewed, gold, id_to_name, sent_map, prev_pairs)

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"DONE in {elapsed:.0f}s — Final: {len(reviewed)} links")
    print(f"{'='*80}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Per-phase pipeline tester")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--linker", default="v16", help="Linker version (v16, v15, v6)")
    parser.add_argument("--stop-after", type=int, default=None,
                       help="Stop after this phase number (0-9)")
    args = parser.parse_args()

    run_phases(args.dataset, args.linker, args.stop_after)


if __name__ == "__main__":
    main()
