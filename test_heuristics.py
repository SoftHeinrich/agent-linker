#!/usr/bin/env python3
"""Single-phase heuristic ablation: load V30c checkpoints, test each heuristic.

For each code-only heuristic, loads the phase input from V30c checkpoints,
runs WITH and WITHOUT the heuristic, and reports the delta. No LLM calls.
"""

import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.pcm_parser import parse_pcm_repository

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
DATASETS = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore" / "text_2016" / "mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore" / "model_2016" / "pcm" / "ms.repository",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore" / "text_2020" / "teastore.txt",
        "model": BENCHMARK_BASE / "teastore" / "model_2020" / "pcm" / "teastore.repository",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates" / "text_2021" / "teammates.txt",
        "model": BENCHMARK_BASE / "teammates" / "model_2021" / "pcm" / "teammates.repository",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton" / "text_2021" / "bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton" / "model_2021" / "pcm" / "bbb.repository",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref" / "text_2021" / "jabref.txt",
        "model": BENCHMARK_BASE / "jabref" / "model_2021" / "pcm" / "jabref.repository",
    },
}
CACHE_DIR = Path("./results/phase_cache/v30c")


def load_checkpoint(dataset, phase_name):
    path = CACHE_DIR / dataset / f"{phase_name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dataset(dataset):
    paths = DATASETS[dataset]
    components = parse_pcm_repository(str(paths["model"]))
    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    sent_map = DocumentLoader.build_sent_map(sentences)
    name_to_id = {c.name: c.id for c in components}
    return components, sentences, sent_map, name_to_id


# ── Import V26a heuristic methods ──────────────────────────────────────
# We instantiate a minimal linker just to get the methods
from llm_sad_sam.linkers.experimental.agent_linker_v26a import AgentLinkerV26a


class HeuristicTester(AgentLinkerV26a):
    """Thin wrapper to access heuristic methods without running the full pipeline."""
    def __init__(self):
        # Skip normal init, just set up enough state
        self.doc_knowledge = None
        self.model_knowledge = None
        self.GENERIC_COMPONENT_WORDS = set()
        self.GENERIC_PARTIALS = set()
        self.doc_profile = None
        self._is_complex = False

    def _in_dotted_path(self, text, comp_name):
        return False  # NDF


# ── Test functions ─────────────────────────────────────────────────────

def test_filter_generic_coref(tester, dataset):
    """Phase 7: _filter_generic_coref — removes coref links without antecedents."""
    data7 = load_checkpoint(dataset, "phase7")
    if not data7:
        return None
    coref_links = data7["coref_links"]

    # Split out pronoun links (added by _deterministic_pronoun_coref)
    # vs LLM coref links (which _filter_generic_coref was applied to)
    # We can't split them from the checkpoint, so we test on the full set

    components, sentences, sent_map, name_to_id = load_dataset(dataset)

    # Apply filter
    filtered = tester._filter_generic_coref(coref_links, sent_map)
    removed = len(coref_links) - len(filtered)

    if removed > 0:
        # Find which ones were removed
        filtered_set = {(l.sentence_number, l.component_id) for l in filtered}
        removed_links = [l for l in coref_links if (l.sentence_number, l.component_id) not in filtered_set]
        details = [(f"S{l.sentence_number}->{l.component_name}", l.source) for l in removed_links]
        return {"removed": removed, "total": len(coref_links), "details": details}
    return {"removed": 0, "total": len(coref_links), "details": []}


def test_deterministic_pronoun_coref(tester, dataset):
    """Phase 7: _deterministic_pronoun_coref — adds pronoun-continuation links."""
    data4 = load_checkpoint(dataset, "phase4")
    data6 = load_checkpoint(dataset, "phase6")
    data7 = load_checkpoint(dataset, "phase7")
    if not all([data4, data6, data7]):
        return None

    components, sentences, sent_map, name_to_id = load_dataset(dataset)
    transarc_set = data4["transarc_set"]
    validated = data6["validated"]
    coref_links = data7["coref_links"]

    # Reconstruct existing set (same as in V30c's Phase 7)
    coref_set = {(l.sentence_number, l.component_id) for l in coref_links}
    existing = transarc_set | {(c.sentence_number, c.component_id) for c in validated} | coref_set

    pronoun_links = tester._deterministic_pronoun_coref(
        sentences, components, name_to_id, sent_map, existing)

    if pronoun_links:
        details = [(f"S{l.sentence_number}->{l.component_name}") for l in pronoun_links]
        return {"added": len(pronoun_links), "details": details}
    return {"added": 0, "details": []}


def test_boundary_filters(tester, dataset):
    """Phase 8c: _apply_boundary_filters — removes weak/spurious links."""
    data_pj = load_checkpoint(dataset, "pre_judge")
    if not data_pj:
        return None

    # pre_judge checkpoint is AFTER boundary filters were applied.
    # We need to reconstruct pre-filter links.
    # Let's load from phase7 + phase4 + phase6 and rebuild preliminary.
    data4 = load_checkpoint(dataset, "phase4")
    data6 = load_checkpoint(dataset, "phase6")
    data7 = load_checkpoint(dataset, "phase7")
    data3 = load_checkpoint(dataset, "phase3")
    if not all([data4, data6, data7, data3]):
        return None

    components, sentences, sent_map, name_to_id = load_dataset(dataset)
    tester.doc_knowledge = data3["doc_knowledge"]

    transarc_links = data4["transarc_links"]
    transarc_set = data4["transarc_set"]
    validated = data6["validated"]
    coref_links = data7["coref_links"]

    # Rebuild partial injection
    from llm_sad_sam.core.data_types import SadSamLink
    partial_links = tester._inject_partial_references(
        sentences, components, name_to_id, transarc_set,
        {(c.sentence_number, c.component_id) for c in validated},
        {(l.sentence_number, l.component_id) for l in coref_links},
        set(),
    )

    entity_links = [
        SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
        for c in validated
    ]
    all_links = transarc_links + entity_links + coref_links + partial_links

    # Deduplicate
    SOURCE_PRIORITY = {"transarc": 1, "entity": 2, "validated": 3, "coreference": 4, "partial_inject": 0}
    link_map = {}
    for lk in all_links:
        key = (lk.sentence_number, lk.component_id)
        if key not in link_map:
            link_map[key] = lk
        else:
            old_p = SOURCE_PRIORITY.get(link_map[key].source, 0)
            new_p = SOURCE_PRIORITY.get(lk.source, 0)
            if new_p > old_p:
                link_map[key] = lk
    preliminary = list(link_map.values())

    # Apply parent-overlap guard (same as V30c)
    data1 = load_checkpoint(dataset, "phase1")
    if data1:
        tester.model_knowledge = data1["model_knowledge"]
    if tester.model_knowledge and tester.model_knowledge.impl_to_abstract:
        child_to_parent = tester.model_knowledge.impl_to_abstract
        sent_comps = defaultdict(set)
        for lk in preliminary:
            sent_comps[lk.sentence_number].add(lk.component_name)
        filtered_po = []
        for lk in preliminary:
            parent = child_to_parent.get(lk.component_name)
            if parent and parent in sent_comps[lk.sentence_number]:
                pass  # dropped
            else:
                filtered_po.append(lk)
        preliminary = filtered_po

    # Now apply boundary filters
    kept, rejected = tester._apply_boundary_filters(preliminary, sent_map, transarc_set)

    if rejected:
        details = [(f"S{lk.sentence_number}->{lk.component_name}", reason) for lk, reason in rejected]
        return {"removed": len(rejected), "total": len(preliminary), "details": details}
    return {"removed": 0, "total": len(preliminary), "details": []}


def test_partial_injection(tester, dataset):
    """Phase 8b: _inject_partial_references — adds partial-match links."""
    data3 = load_checkpoint(dataset, "phase3")
    data4 = load_checkpoint(dataset, "phase4")
    data6 = load_checkpoint(dataset, "phase6")
    data7 = load_checkpoint(dataset, "phase7")
    if not all([data3, data4, data6, data7]):
        return None

    components, sentences, sent_map, name_to_id = load_dataset(dataset)
    tester.doc_knowledge = data3["doc_knowledge"]
    transarc_set = data4["transarc_set"]
    validated = data6["validated"]
    coref_links = data7["coref_links"]

    partial_links = tester._inject_partial_references(
        sentences, components, name_to_id, transarc_set,
        {(c.sentence_number, c.component_id) for c in validated},
        {(l.sentence_number, l.component_id) for l in coref_links},
        set(),
    )

    if partial_links:
        details = [f"S{l.sentence_number}->{l.component_name}" for l in partial_links]
        return {"added": len(partial_links), "details": details}
    return {"added": 0, "details": []}


def test_parent_overlap(tester, dataset):
    """Between 8b-8c: parent-overlap guard — removes child links when parent linked."""
    data1 = load_checkpoint(dataset, "phase1")
    data4 = load_checkpoint(dataset, "phase4")
    data6 = load_checkpoint(dataset, "phase6")
    data7 = load_checkpoint(dataset, "phase7")
    data3 = load_checkpoint(dataset, "phase3")
    if not all([data1, data4, data6, data7, data3]):
        return None

    tester.model_knowledge = data1["model_knowledge"]
    if not tester.model_knowledge or not tester.model_knowledge.impl_to_abstract:
        return {"removed": 0, "total": 0, "details": [], "note": "no impl_to_abstract map"}

    components, sentences, sent_map, name_to_id = load_dataset(dataset)
    tester.doc_knowledge = data3["doc_knowledge"]

    transarc_links = data4["transarc_links"]
    transarc_set = data4["transarc_set"]
    validated = data6["validated"]
    coref_links = data7["coref_links"]

    from llm_sad_sam.core.data_types import SadSamLink
    partial_links = tester._inject_partial_references(
        sentences, components, name_to_id, transarc_set,
        {(c.sentence_number, c.component_id) for c in validated},
        {(l.sentence_number, l.component_id) for l in coref_links},
        set(),
    )

    entity_links = [
        SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
        for c in validated
    ]
    all_links = transarc_links + entity_links + coref_links + partial_links
    SOURCE_PRIORITY = {"transarc": 1, "entity": 2, "validated": 3, "coreference": 4, "partial_inject": 0}
    link_map = {}
    for lk in all_links:
        key = (lk.sentence_number, lk.component_id)
        if key not in link_map:
            link_map[key] = lk
        else:
            old_p = SOURCE_PRIORITY.get(link_map[key].source, 0)
            new_p = SOURCE_PRIORITY.get(lk.source, 0)
            if new_p > old_p:
                link_map[key] = lk
    preliminary = list(link_map.values())

    child_to_parent = tester.model_knowledge.impl_to_abstract
    sent_comps = defaultdict(set)
    for lk in preliminary:
        sent_comps[lk.sentence_number].add(lk.component_name)

    dropped = []
    for lk in preliminary:
        parent = child_to_parent.get(lk.component_name)
        if parent and parent in sent_comps[lk.sentence_number]:
            dropped.append(f"S{lk.sentence_number}->{lk.component_name} (parent: {parent})")

    return {"removed": len(dropped), "total": len(preliminary), "details": dropped}


def test_abbreviation_guard(tester, dataset):
    """Phase 5: _apply_abbreviation_guard_to_candidates — removes bad abbreviation matches."""
    data3 = load_checkpoint(dataset, "phase3")
    data5 = load_checkpoint(dataset, "phase5")
    if not all([data3, data5]):
        return None

    # The phase5 checkpoint is AFTER the guard was applied.
    # We can't undo it. But we can check if the guard method would remove anything
    # from the current candidates (it shouldn't, since they already passed).
    # Instead, report what doc_knowledge abbreviations exist — if none, guard is no-op.

    tester.doc_knowledge = data3["doc_knowledge"]
    abbrevs = tester.doc_knowledge.abbreviations

    return {"abbreviations": dict(abbrevs), "count": len(abbrevs),
            "note": "Guard only fires if abbreviations exist AND entity extraction returns them"}


def test_syn_safe_bypass(tester, dataset):
    """Phase 9: synonym-safe judge bypass — counts how many links skip judge."""
    data_pj = load_checkpoint(dataset, "pre_judge")
    data3 = load_checkpoint(dataset, "phase3")
    if not all([data_pj, data3]):
        return None

    components, sentences, sent_map, name_to_id = load_dataset(dataset)
    tester.doc_knowledge = data3["doc_knowledge"]

    preliminary = data_pj["preliminary"]
    transarc_set = data_pj["transarc_set"]

    syn_safe = []
    for l in preliminary:
        sent = sent_map.get(l.sentence_number)
        if sent and tester._has_alias_mention(l.component_name, sent.text):
            syn_safe.append(f"S{l.sentence_number}->{l.component_name}")

    return {"syn_safe_count": len(syn_safe), "total": len(preliminary), "details": syn_safe}


def test_generic_mention(tester, dataset):
    """Phase 6: _is_generic_mention — counts which candidates get flagged as generic."""
    data1 = load_checkpoint(dataset, "phase1")
    data5 = load_checkpoint(dataset, "phase5")
    if not all([data1, data5]):
        return None

    components, sentences, sent_map, name_to_id = load_dataset(dataset)
    tester.model_knowledge = data1["model_knowledge"]
    tester.GENERIC_COMPONENT_WORDS = data1.get("generic_component_words", set())
    tester.GENERIC_PARTIALS = data1.get("generic_partials", set())

    candidates = data5["candidates"]

    flagged = []
    for c in candidates:
        sent = sent_map.get(c.sentence_number)
        if sent and tester._is_generic_mention(c.component_name, sent.text):
            flagged.append(f"S{c.sentence_number}->{c.component_name}")

    return {"flagged": len(flagged), "total": len(candidates), "details": flagged}


# ── Main ───────────────────────────────────────────────────────────────

TESTS = {
    "_filter_generic_coref (P7)":     test_filter_generic_coref,
    "_deterministic_pronoun_coref (P7)": test_deterministic_pronoun_coref,
    "_apply_boundary_filters (P8c)":  test_boundary_filters,
    "_inject_partial_references (P8b)": test_partial_injection,
    "parent-overlap guard (P8)":      test_parent_overlap,
    "abbreviation guard (P5)":        test_abbreviation_guard,
    "syn-safe judge bypass (P9)":     test_syn_safe_bypass,
    "_is_generic_mention (P6)":       test_generic_mention,
}

if __name__ == "__main__":
    tester = HeuristicTester()
    datasets = list(DATASETS.keys())

    print("=" * 100)
    print("HEURISTIC ABLATION: Single-phase delta analysis (no LLM calls)")
    print("=" * 100)

    for test_name, test_fn in TESTS.items():
        print(f"\n{'─' * 80}")
        print(f"  {test_name}")
        print(f"{'─' * 80}")

        fires_anywhere = False
        for ds in datasets:
            # Reset tester state
            tester.doc_knowledge = None
            tester.model_knowledge = None
            tester.GENERIC_COMPONENT_WORDS = set()
            tester.GENERIC_PARTIALS = set()

            result = test_fn(tester, ds)
            if result is None:
                print(f"  {ds:20s} — SKIP (missing checkpoint)")
                continue

            # Determine if it fires
            delta = result.get("removed", 0) + result.get("added", 0) + result.get("flagged", 0) + result.get("syn_safe_count", 0)
            if "count" in result:
                delta = result["count"]

            if delta > 0:
                fires_anywhere = True
                details_str = ""
                if result.get("details"):
                    details = result["details"]
                    if isinstance(details[0], tuple):
                        details_str = "; ".join([f"{d[0]}({d[1]})" for d in details[:5]])
                    else:
                        details_str = "; ".join(details[:5])
                    if len(details) > 5:
                        details_str += f" (+{len(details)-5} more)"

                # Format based on type
                if "removed" in result:
                    print(f"  {ds:20s} — REMOVES {result['removed']}/{result['total']}: {details_str}")
                elif "added" in result:
                    print(f"  {ds:20s} — ADDS {result['added']}: {details_str}")
                elif "flagged" in result:
                    print(f"  {ds:20s} — FLAGS {result['flagged']}/{result['total']}: {details_str}")
                elif "syn_safe_count" in result:
                    print(f"  {ds:20s} — BYPASSES {result['syn_safe_count']}/{result['total']}: {details_str}")
                elif "count" in result:
                    print(f"  {ds:20s} — {result['count']} abbreviations: {result.get('abbreviations', {})}")
                else:
                    print(f"  {ds:20s} — ACTIVE: {result}")
            else:
                note = result.get("note", "")
                print(f"  {ds:20s} — no effect{' (' + note + ')' if note else ''}")

        verdict = "FIRES" if fires_anywhere else "ZERO EFFECT — safe to cut"
        print(f"  {'VERDICT':20s} — {verdict}")

    print(f"\n{'=' * 100}")
    print("DONE")
    print(f"{'=' * 100}")
