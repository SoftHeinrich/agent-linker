#!/usr/bin/env python3
"""Per-phase LLM variance test for V19.

Runs the pipeline 3 times per dataset, recording key metrics at each LLM phase
to identify which phases are stable and which introduce variance.
"""

import csv
import os
import re
import sys
import time
import json
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


def run_single(dataset_name, run_id):
    """Run full V19 pipeline once, return per-phase metrics dict."""
    from llm_sad_sam.linkers.experimental.agent_linker_v19 import AgentLinkerV19
    from llm_sad_sam.core import LearnedThresholds

    paths = DATASETS[dataset_name]
    gold = load_gold_sam(str(paths["gold_sam"]))
    components = parse_pcm_repository(str(paths["model"]))
    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    name_to_id = {c.name: c.id for c in components}
    id_to_name = {c.id: c.name for c in components}
    sent_map = DocumentLoader.build_sent_map(sentences)

    linker = AgentLinkerV19(backend=LLMBackend.CLAUDE)
    linker._cached_sent_map = sent_map

    metrics = {}
    t0 = time.time()

    # Phase 0 (no LLM)
    linker.doc_profile = linker._learn_document_profile(sentences, components)
    linker._is_complex = linker._structural_complexity(sentences, components)
    linker.thresholds = LearnedThresholds(0, 0, 0, 0, "qualitative", 0)

    # Phase 1 (deterministic)
    linker.model_knowledge = linker._analyze_model(components)
    metrics["P1_arch"] = len(linker.model_knowledge.architectural_names)
    metrics["P1_ambig"] = sorted(linker.model_knowledge.ambiguous_names)

    # Phase 2 (LLM)
    linker.learned_patterns = linker._learn_patterns_with_debate(sentences, components)
    metrics["P2_subprocess"] = len(linker.learned_patterns.subprocess_terms)
    metrics["P2_terms"] = sorted(linker.learned_patterns.subprocess_terms)

    # Phase 3 (LLM)
    linker.doc_knowledge = linker._learn_document_knowledge_enriched(sentences, components)
    dk = linker.doc_knowledge
    metrics["P3_abbrev"] = len(dk.abbreviations)
    metrics["P3_syn"] = len(dk.synonyms)
    metrics["P3_partial"] = len(dk.partial_references)
    metrics["P3_generic"] = len(dk.generic_terms) if hasattr(dk, 'generic_terms') else 0
    metrics["P3_syn_list"] = sorted(f"{k}->{v}" for k, v in dk.synonyms.items())
    metrics["P3_generic_list"] = sorted(dk.generic_terms) if hasattr(dk, 'generic_terms') and dk.generic_terms else []

    # Phase 3b (no LLM)
    linker._enrich_multiword_partials(sentences, components)
    metrics["P3b_partial"] = len(dk.partial_references)

    # Phase 4 (no LLM)
    transarc_links = linker._process_transarc(
        str(paths["transarc_sam"]), id_to_name, sent_map, name_to_id
    )
    transarc_set = {(l.sentence_number, l.component_id) for l in transarc_links}
    metrics["P4_links"] = len(transarc_links)

    # Phase 5 (LLM)
    candidates = linker._extract_entities_enriched(sentences, components, name_to_id, sent_map)
    candidates = linker._apply_abbreviation_guard_to_candidates(candidates, sent_map)
    metrics["P5_candidates"] = len(candidates)
    # Eval Phase 5
    p5_pairs = transarc_set | {(c.sentence_number, c.component_id) for c in candidates}
    m5 = eval_metrics(p5_pairs, gold)
    metrics["P5_F1"] = round(m5["F1"] * 100, 1)
    metrics["P5_FP"] = m5["fp"]

    # Phase 5b (LLM)
    entity_comps = {c.component_name for c in candidates}
    transarc_comps = {l.component_name for l in transarc_links}
    covered = entity_comps | transarc_comps
    unlinked = [c for c in components if c.name not in covered]
    extra = []
    if unlinked:
        extra = linker._targeted_extraction(unlinked, sentences, name_to_id, sent_map) or []
        candidates.extend(extra)
    metrics["P5b_targeted"] = len(extra)

    # Phase 6 (LLM)
    if hasattr(linker, '_validate_split'):
        validated = linker._validate_split(candidates, components, sent_map)
    elif hasattr(linker, '_validate_intersect'):
        validated = linker._validate_intersect(candidates, components, sent_map)
    else:
        validated = linker._validate_candidates(candidates, components, sent_map)
    metrics["P6_validated"] = len(validated)
    metrics["P6_input"] = len(candidates)
    # Eval Phase 6
    entity_links = [
        SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
        for c in validated
    ]
    p6_pairs = transarc_set | {(c.sentence_number, c.component_id) for c in validated}
    m6 = eval_metrics(p6_pairs, gold)
    metrics["P6_F1"] = round(m6["F1"] * 100, 1)
    metrics["P6_FP"] = m6["fp"]

    # Phase 7 (LLM)
    if linker._is_complex:
        coref_links = linker._coref_debate(sentences, components, name_to_id, sent_map)
    else:
        discourse_model = linker._build_discourse_model(sentences, components, name_to_id)
        coref_links = linker._coref_discourse(sentences, components, name_to_id, sent_map, discourse_model)
    before_filter = len(coref_links)
    coref_links = linker._filter_generic_coref(coref_links, sent_map)
    metrics["P7_raw"] = before_filter
    metrics["P7_filtered"] = len(coref_links)
    # Eval Phase 7
    p7_pairs = p6_pairs | {(l.sentence_number, l.component_id) for l in coref_links}
    m7 = eval_metrics(p7_pairs, gold)
    metrics["P7_F1"] = round(m7["F1"] * 100, 1)
    metrics["P7_FP"] = m7["fp"]

    # Phase 8b (no LLM)
    partial_links = linker._inject_partial_references(
        sentences, components, name_to_id, transarc_set,
        {(c.sentence_number, c.component_id) for c in validated},
        {(l.sentence_number, l.component_id) for l in coref_links},
        set(),
    )
    metrics["P8b_injected"] = len(partial_links)

    # Dedup + boundary
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
    metrics["P8c_pre_judge"] = len(preliminary)

    # Phase 9 (LLM)
    reviewed = linker._judge_review(preliminary, sentences, components, sent_map, transarc_set)
    metrics["P9_approved"] = len(reviewed)
    metrics["P9_rejected"] = len(preliminary) - len(reviewed)

    # Final eval
    final_pairs = {(l.sentence_number, l.component_id) for l in reviewed}
    mf = eval_metrics(final_pairs, gold)
    metrics["final_F1"] = round(mf["F1"] * 100, 1)
    metrics["final_P"] = round(mf["P"] * 100, 1)
    metrics["final_R"] = round(mf["R"] * 100, 1)
    metrics["final_TP"] = mf["tp"]
    metrics["final_FP"] = mf["fp"]
    metrics["final_FN"] = mf["fn"]
    metrics["time"] = round(time.time() - t0)

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    ds = args.dataset
    n = args.runs

    print(f"{'='*80}")
    print(f"VARIANCE TEST: V19 on {ds} ({n} runs)")
    print(f"{'='*80}")

    all_runs = []
    for i in range(n):
        print(f"\n{'─'*60}")
        print(f"RUN {i+1}/{n}")
        print(f"{'─'*60}")
        m = run_single(ds, i)
        all_runs.append(m)
        # Print summary for this run
        print(f"  P5={m['P5_candidates']} cands, P6={m['P6_validated']}/{m['P6_input']}, "
              f"P7={m['P7_filtered']} coref, P9={m['P9_approved']}/{m['P8c_pre_judge']}")
        print(f"  Final: P={m['final_P']}% R={m['final_R']}% F1={m['final_F1']}% "
              f"(TP={m['final_TP']} FP={m['final_FP']} FN={m['final_FN']}) [{m['time']}s]")

    # Summary table
    print(f"\n{'='*80}")
    print(f"VARIANCE SUMMARY: {ds}")
    print(f"{'='*80}")

    key_metrics = [
        ("P2_subprocess", "Phase 2: subprocess terms"),
        ("P3_syn", "Phase 3: synonyms"),
        ("P3_partial", "Phase 3: partials"),
        ("P3_generic", "Phase 3: generic rejected"),
        ("P5_candidates", "Phase 5: candidates"),
        ("P5_F1", "Phase 5: cum F1%"),
        ("P5b_targeted", "Phase 5b: targeted"),
        ("P6_validated", "Phase 6: validated"),
        ("P6_F1", "Phase 6: cum F1%"),
        ("P7_raw", "Phase 7: raw coref"),
        ("P7_filtered", "Phase 7: filtered coref"),
        ("P7_F1", "Phase 7: cum F1%"),
        ("P8b_injected", "Phase 8b: partial inject"),
        ("P9_approved", "Phase 9: approved"),
        ("P9_rejected", "Phase 9: rejected"),
        ("final_F1", "Final F1%"),
        ("final_FP", "Final FP"),
        ("final_FN", "Final FN"),
    ]

    header = f"  {'Metric':<30s}" + "".join(f"  Run{i+1:>2d}" for i in range(n)) + "  Spread"
    print(header)
    print("  " + "─" * (30 + 8 * n + 8))

    for key, label in key_metrics:
        vals = [m[key] for m in all_runs]
        if isinstance(vals[0], (int, float)):
            spread = max(vals) - min(vals)
            row = f"  {label:<30s}" + "".join(f"  {v:>5}" for v in vals) + f"  {spread:>5}"
            # Mark unstable phases
            if spread > 0:
                if key.endswith("_F1"):
                    row += f"  {'!!' if spread > 3 else '!'}"
                elif spread > 3:
                    row += "  !!"
                elif spread > 0:
                    row += "  !"
        else:
            row = f"  {label:<30s}" + "  (list, see detail below)"
        print(row)

    # Print variable lists
    print(f"\n  Phase 3 synonym lists:")
    for i, m in enumerate(all_runs):
        print(f"    Run {i+1}: {m.get('P3_syn_list', [])}")

    print(f"\n  Phase 3 generic lists:")
    for i, m in enumerate(all_runs):
        print(f"    Run {i+1}: {m.get('P3_generic_list', [])}")

    print(f"\n  Phase 2 subprocess terms:")
    for i, m in enumerate(all_runs):
        print(f"    Run {i+1}: {m.get('P2_terms', [])}")


if __name__ == "__main__":
    main()
