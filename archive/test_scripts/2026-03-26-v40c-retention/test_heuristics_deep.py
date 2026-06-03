#!/usr/bin/env python3
"""Deep heuristic analysis: check if each heuristic is already covered by LLM phases.

For each firing heuristic, loads checkpoint data and checks whether the LLM
phases before/after already handle the same cases. No LLM calls.
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
from llm_sad_sam.core.data_types import SadSamLink

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

# Load gold standards for checking TP/FP
GOLD_DIR = BENCHMARK_BASE


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
    id_to_name = {c.id: c.name for c in components}
    return components, sentences, sent_map, name_to_id, id_to_name


def load_gold(dataset):
    """Load gold standard links as set of (sentence_number, component_id)."""
    gold_path = GOLD_DIR / dataset
    import csv
    import glob
    pattern = str(gold_path / "**" / "goldstandard_sad_*-sam_*.csv")
    files = [f for f in glob.glob(pattern, recursive=True) if "UME" not in f and "code" not in f]
    if not files:
        return set()
    gold = set()
    for f in files:
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                cid = row.get("modelElementID", "")
                sid = row.get("sentence", "")
                if sid and cid:
                    gold.add((int(sid), cid))
    return gold


# ══════════════════════════════════════════════════════════════════════
# TEST 1: P8b partial injection — already covered by Phase 3+5?
# ══════════════════════════════════════════════════════════════════════

def analyze_p8b():
    print("\n" + "=" * 90)
    print("  P8b: _inject_partial_references — are these links already found by Phase 3+5?")
    print("=" * 90)

    for ds in DATASETS:
        data3 = load_checkpoint(ds, "phase3")
        data4 = load_checkpoint(ds, "phase4")
        data5 = load_checkpoint(ds, "phase5")
        data6 = load_checkpoint(ds, "phase6")
        data7 = load_checkpoint(ds, "phase7")
        if not all([data3, data4, data5, data6, data7]):
            continue

        components, sentences, sent_map, name_to_id, id_to_name = load_dataset(ds)
        doc_knowledge = data3["doc_knowledge"]
        gold = load_gold(ds)

        # What partials does Phase 3 discover?
        partials = doc_knowledge.partial_references
        if not partials:
            print(f"\n  {ds}: No partials in Phase 3 — P8b is no-op")
            continue

        print(f"\n  {ds}:")
        print(f"    Phase 3 partials: {dict(partials)}")

        # What links does P8b inject?
        from llm_sad_sam.linkers.experimental.agent_linker_v26a import AgentLinkerV26a

        class Tester(AgentLinkerV26a):
            def __init__(self):
                self.doc_knowledge = doc_knowledge
                self.model_knowledge = None
                self.GENERIC_COMPONENT_WORDS = set()
                self.GENERIC_PARTIALS = set()
            def _in_dotted_path(self, text, comp_name):
                return False

        tester = Tester()
        transarc_set = data4["transarc_set"]
        validated = data6["validated"]
        coref_links = data7["coref_links"]

        partial_links = tester._inject_partial_references(
            sentences, components, name_to_id, transarc_set,
            {(c.sentence_number, c.component_id) for c in validated},
            {(l.sentence_number, l.component_id) for l in coref_links},
            set(),
        )

        if not partial_links:
            print(f"    P8b injects: 0 links")
            continue

        # Check: are these already in transarc, entity, or coref?
        existing = (transarc_set
                    | {(c.sentence_number, c.component_id) for c in validated}
                    | {(l.sentence_number, l.component_id) for l in coref_links})

        # Also check Phase 5 raw candidates (before validation)
        candidates = data5["candidates"]
        cand_set = {(c.sentence_number, c.component_id) for c in candidates}

        print(f"    P8b injects: {len(partial_links)} links")
        for pl in partial_links:
            key = (pl.sentence_number, pl.component_id)
            in_gold = key in gold
            in_existing = key in existing
            in_candidates = key in cand_set
            sent = sent_map.get(pl.sentence_number)
            sent_text = sent.text[:80] if sent else "?"

            status = []
            if in_gold:
                status.append("TP")
            else:
                status.append("FP")
            if in_existing:
                status.append("ALREADY_LINKED")
            if in_candidates:
                status.append("IN_PHASE5")

            print(f"      S{pl.sentence_number}->{pl.component_name} [{', '.join(status)}]")
            print(f"        Sentence: \"{sent_text}...\"")

        # Count new TPs added
        new_tps = sum(1 for pl in partial_links
                      if (pl.sentence_number, pl.component_id) in gold
                      and (pl.sentence_number, pl.component_id) not in existing)
        new_fps = sum(1 for pl in partial_links
                      if (pl.sentence_number, pl.component_id) not in gold
                      and (pl.sentence_number, pl.component_id) not in existing)
        print(f"    NEW TPs: {new_tps}, NEW FPs: {new_fps}")
        print(f"    → If Phase 5 entity extraction found these, P8b would be redundant")


# ══════════════════════════════════════════════════════════════════════
# TEST 2: P8c boundary filters — what does NDF prompt miss?
# ══════════════════════════════════════════════════════════════════════

def analyze_p8c():
    print("\n" + "=" * 90)
    print("  P8c: _apply_boundary_filters — what does the NDF prompt fail to catch?")
    print("=" * 90)

    for ds in DATASETS:
        data3 = load_checkpoint(ds, "phase3")
        data4 = load_checkpoint(ds, "phase4")
        data6 = load_checkpoint(ds, "phase6")
        data7 = load_checkpoint(ds, "phase7")
        data1 = load_checkpoint(ds, "phase1")
        if not all([data3, data4, data6, data7, data1]):
            continue

        components, sentences, sent_map, name_to_id, id_to_name = load_dataset(ds)
        gold = load_gold(ds)

        from llm_sad_sam.linkers.experimental.agent_linker_v26a import AgentLinkerV26a

        class Tester(AgentLinkerV26a):
            def __init__(self):
                self.doc_knowledge = data3["doc_knowledge"]
                self.model_knowledge = data1["model_knowledge"]
                self.GENERIC_COMPONENT_WORDS = data1.get("generic_component_words", set())
                self.GENERIC_PARTIALS = data1.get("generic_partials", set())
            def _in_dotted_path(self, text, comp_name):
                return False

        tester = Tester()

        transarc_links = data4["transarc_links"]
        transarc_set = data4["transarc_set"]
        validated = data6["validated"]
        coref_links = data7["coref_links"]

        # Rebuild preliminary (same as V30c link())
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

        # Parent-overlap
        if tester.model_knowledge and tester.model_knowledge.impl_to_abstract:
            child_to_parent = tester.model_knowledge.impl_to_abstract
            sent_comps = defaultdict(set)
            for lk in preliminary:
                sent_comps[lk.sentence_number].add(lk.component_name)
            filtered = []
            for lk in preliminary:
                parent = child_to_parent.get(lk.component_name)
                if parent and parent in sent_comps[lk.sentence_number]:
                    pass
                else:
                    filtered.append(lk)
            preliminary = filtered

        # Apply boundary filters
        kept, rejected = tester._apply_boundary_filters(preliminary, sent_map, transarc_set)

        if not rejected:
            continue

        print(f"\n  {ds}: {len(rejected)} links rejected by boundary filters")
        for lk, reason in rejected:
            key = (lk.sentence_number, lk.component_id)
            is_tp = key in gold
            sent = sent_map.get(lk.sentence_number)
            sent_text = sent.text[:100] if sent else "?"

            print(f"    S{lk.sentence_number}->{lk.component_name} [{reason}] src={lk.source} {'TP!' if is_tp else 'FP'}")
            print(f"      \"{sent_text}\"")

            # For package_path: show the actual dotted path context
            if reason == "package_path" and sent:
                dots = re.findall(r'[\w]+(?:\.[\w]+){2,}', sent.text)
                if dots:
                    print(f"      Dotted paths: {dots}")

            # For generic_word: show what modifier triggered it
            if reason == "generic_word" and sent:
                name_lower = lk.component_name.lower()
                pattern = rf'\b(\w+)\s+{re.escape(name_lower)}\b'
                matches = re.findall(pattern, sent.text.lower())
                if matches:
                    print(f"      Modifier: '{matches[0]} {name_lower}'")

        # Summary
        tp_killed = sum(1 for lk, _ in rejected if (lk.sentence_number, lk.component_id) in gold)
        fp_killed = len(rejected) - tp_killed
        print(f"    SUMMARY: {fp_killed} correct rejections (FP killed), {tp_killed} wrong rejections (TP killed)")
        print(f"    → These are cases the NDF prompt FAILED to prevent at extraction time")


# ══════════════════════════════════════════════════════════════════════
# TEST 3: P9 syn-safe bypass — what would judge do with these links?
# ══════════════════════════════════════════════════════════════════════

def analyze_p9_synsafe():
    print("\n" + "=" * 90)
    print("  P9: syn-safe bypass — what are these links and are they TPs?")
    print("=" * 90)

    for ds in DATASETS:
        data3 = load_checkpoint(ds, "phase3")
        data_pj = load_checkpoint(ds, "pre_judge")
        data_final = load_checkpoint(ds, "final")
        if not all([data3, data_pj, data_final]):
            continue

        components, sentences, sent_map, name_to_id, id_to_name = load_dataset(ds)
        gold = load_gold(ds)

        from llm_sad_sam.linkers.experimental.agent_linker_v26a import AgentLinkerV26a

        class Tester(AgentLinkerV26a):
            def __init__(self):
                self.doc_knowledge = data3["doc_knowledge"]
                self.model_knowledge = None
                self.GENERIC_COMPONENT_WORDS = set()
                self.GENERIC_PARTIALS = set()

        tester = Tester()

        preliminary = data_pj["preliminary"]
        transarc_set = data_pj["transarc_set"]

        syn_safe = []
        for l in preliminary:
            sent = sent_map.get(l.sentence_number)
            if sent and tester._has_alias_mention(l.component_name, sent.text):
                key = (l.sentence_number, l.component_id)
                is_tp = key in gold
                syn_safe.append((l, is_tp))

        if not syn_safe:
            continue

        tp_count = sum(1 for _, is_tp in syn_safe if is_tp)
        fp_count = len(syn_safe) - tp_count

        print(f"\n  {ds}: {len(syn_safe)} syn-safe links ({tp_count} TP, {fp_count} FP)")

        # Show the FPs — these are what the judge COULD potentially catch
        fps = [(l, is_tp) for l, is_tp in syn_safe if not is_tp]
        if fps:
            print(f"    FPs that bypass judge:")
            for l, _ in fps:
                sent = sent_map.get(l.sentence_number)
                sent_text = sent.text[:80] if sent else "?"
                # What alias triggers the bypass?
                aliases = []
                for syn, target in tester.doc_knowledge.synonyms.items():
                    if target == l.component_name:
                        if re.search(rf'\b{re.escape(syn.lower())}\b', sent.text.lower()):
                            aliases.append(syn)
                for p, target in tester.doc_knowledge.partial_references.items():
                    if target == l.component_name:
                        if re.search(rf'\b{re.escape(p.lower())}\b', sent.text.lower()):
                            aliases.append(p)
                print(f"      S{l.sentence_number}->{l.component_name} src={l.source} alias={aliases}")
                print(f"        \"{sent_text}\"")
        else:
            print(f"    All syn-safe links are TPs — bypass is 100% correct")

        print(f"    → Without bypass: {tp_count} TPs go to judge (risk of killing)")
        print(f"    → Without bypass: {fp_count} FPs go to judge (chance of catching)")
        print(f"    → Risk/reward: lose {tp_count} TP risk vs catch {fp_count} FP opportunity")


# ══════════════════════════════════════════════════════════════════════
# TEST 4: P6 _is_generic_mention — TP/FP classification of flagged candidates
# ══════════════════════════════════════════════════════════════════════

def analyze_p6_generic():
    print("\n" + "=" * 90)
    print("  P6: _is_generic_mention — are flagged candidates TPs or FPs?")
    print("=" * 90)

    for ds in DATASETS:
        data1 = load_checkpoint(ds, "phase1")
        data5 = load_checkpoint(ds, "phase5")
        if not all([data1, data5]):
            continue

        components, sentences, sent_map, name_to_id, id_to_name = load_dataset(ds)
        gold = load_gold(ds)

        from llm_sad_sam.linkers.experimental.agent_linker_v26a import AgentLinkerV26a

        class Tester(AgentLinkerV26a):
            def __init__(self):
                self.doc_knowledge = None
                self.model_knowledge = data1["model_knowledge"]
                self.GENERIC_COMPONENT_WORDS = data1.get("generic_component_words", set())
                self.GENERIC_PARTIALS = data1.get("generic_partials", set())
            def _in_dotted_path(self, text, comp_name):
                return False

        tester = Tester()
        candidates = data5["candidates"]

        flagged = []
        for c in candidates:
            sent = sent_map.get(c.sentence_number)
            if sent and tester._is_generic_mention(c.component_name, sent.text):
                key = (c.sentence_number, c.component_id)
                is_tp = key in gold
                flagged.append((c, is_tp))

        if not flagged:
            continue

        tp_count = sum(1 for _, is_tp in flagged if is_tp)
        fp_count = len(flagged) - tp_count

        print(f"\n  {ds}: {len(flagged)} candidates flagged as generic ({tp_count} TP, {fp_count} FP)")
        for c, is_tp in flagged:
            sent = sent_map.get(c.sentence_number)
            sent_text = sent.text[:80] if sent else "?"
            print(f"    S{c.sentence_number}->{c.component_name} {'TP' if is_tp else 'FP'} src={c.source}")
            print(f"      \"{sent_text}\"")

        print(f"    → Flagging sends these to double-validation (LLM checks harder)")
        print(f"    → {tp_count} TPs must survive validation, {fp_count} FPs should be caught")
        print(f"    → Without flag: all go through normal validation (less scrutiny)")


# ══════════════════════════════════════════════════════════════════════
# TEST 5: P7 pronoun coref — TP/FP of added link
# ══════════════════════════════════════════════════════════════════════

def analyze_p7_pronoun():
    print("\n" + "=" * 90)
    print("  P7: _deterministic_pronoun_coref — TP/FP of added links")
    print("=" * 90)

    for ds in DATASETS:
        data4 = load_checkpoint(ds, "phase4")
        data6 = load_checkpoint(ds, "phase6")
        data7 = load_checkpoint(ds, "phase7")
        if not all([data4, data6, data7]):
            continue

        components, sentences, sent_map, name_to_id, id_to_name = load_dataset(ds)
        gold = load_gold(ds)

        from llm_sad_sam.linkers.experimental.agent_linker_v26a import AgentLinkerV26a

        class Tester(AgentLinkerV26a):
            def __init__(self):
                self.doc_knowledge = None
                self.model_knowledge = None
                self.GENERIC_COMPONENT_WORDS = set()
                self.GENERIC_PARTIALS = set()
            def _in_dotted_path(self, text, comp_name):
                return False

        tester = Tester()
        transarc_set = data4["transarc_set"]
        validated = data6["validated"]
        coref_links = data7["coref_links"]
        coref_set = {(l.sentence_number, l.component_id) for l in coref_links}
        existing = transarc_set | {(c.sentence_number, c.component_id) for c in validated} | coref_set

        pronoun_links = tester._deterministic_pronoun_coref(
            sentences, components, name_to_id, sent_map, existing)

        if not pronoun_links:
            continue

        print(f"\n  {ds}: {len(pronoun_links)} pronoun-coref links")
        for pl in pronoun_links:
            key = (pl.sentence_number, pl.component_id)
            is_tp = key in gold
            sent = sent_map.get(pl.sentence_number)
            sent_text = sent.text[:80] if sent else "?"
            print(f"    S{pl.sentence_number}->{pl.component_name} {'TP' if is_tp else 'FP'}")
            print(f"      \"{sent_text}\"")


# ══════════════════════════════════════════════════════════════════════
# TEST 6: parent-overlap — TP/FP of removed link
# ══════════════════════════════════════════════════════════════════════

def analyze_parent_overlap():
    print("\n" + "=" * 90)
    print("  Parent-overlap guard — TP/FP of removed links")
    print("=" * 90)

    for ds in DATASETS:
        data1 = load_checkpoint(ds, "phase1")
        data3 = load_checkpoint(ds, "phase3")
        data4 = load_checkpoint(ds, "phase4")
        data6 = load_checkpoint(ds, "phase6")
        data7 = load_checkpoint(ds, "phase7")
        if not all([data1, data3, data4, data6, data7]):
            continue

        components, sentences, sent_map, name_to_id, id_to_name = load_dataset(ds)
        gold = load_gold(ds)

        mk = data1["model_knowledge"]
        if not mk or not mk.impl_to_abstract:
            continue

        from llm_sad_sam.linkers.experimental.agent_linker_v26a import AgentLinkerV26a

        class Tester(AgentLinkerV26a):
            def __init__(self):
                self.doc_knowledge = data3["doc_knowledge"]
                self.model_knowledge = mk
                self.GENERIC_COMPONENT_WORDS = set()
                self.GENERIC_PARTIALS = set()
            def _in_dotted_path(self, text, comp_name):
                return False

        tester = Tester()
        transarc_links = data4["transarc_links"]
        transarc_set = data4["transarc_set"]
        validated = data6["validated"]
        coref_links = data7["coref_links"]

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

        child_to_parent = mk.impl_to_abstract
        sent_comps = defaultdict(set)
        for lk in preliminary:
            sent_comps[lk.sentence_number].add(lk.component_name)

        dropped = []
        for lk in preliminary:
            parent = child_to_parent.get(lk.component_name)
            if parent and parent in sent_comps[lk.sentence_number]:
                key = (lk.sentence_number, lk.component_id)
                is_tp = key in gold
                dropped.append((lk, is_tp, parent))

        if not dropped:
            continue

        print(f"\n  {ds}: {len(dropped)} links dropped by parent-overlap")
        for lk, is_tp, parent in dropped:
            sent = sent_map.get(lk.sentence_number)
            sent_text = sent.text[:80] if sent else "?"
            print(f"    S{lk.sentence_number}->{lk.component_name} (parent: {parent}) {'TP!' if is_tp else 'FP'}")
            print(f"      \"{sent_text}\"")


if __name__ == "__main__":
    print("=" * 90)
    print("  DEEP HEURISTIC ANALYSIS: TP/FP impact + LLM replaceability")
    print("=" * 90)

    analyze_p8b()
    analyze_p8c()
    analyze_p9_synsafe()
    analyze_p6_generic()
    analyze_p7_pronoun()
    analyze_parent_overlap()

    print("\n" + "=" * 90)
    print("  DONE")
    print("=" * 90)
