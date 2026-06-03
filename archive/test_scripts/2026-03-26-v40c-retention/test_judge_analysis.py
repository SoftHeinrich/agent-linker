#!/usr/bin/env python3
"""Analyze what FPs survive the judge — are they judge-immune or judge-missed?"""

import csv
import glob
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


def load_gold(dataset):
    """Load gold standard. Format: modelElementID,sentence (comma-delimited, SAD-SAM only)."""
    gold_path = BENCHMARK_BASE / dataset
    # Only load SAD-SAM gold standards (not code or UME)
    pattern = str(gold_path / "**" / "goldstandard_sad_*-sam_*.csv")
    files = [f for f in glob.glob(pattern, recursive=True) if "UME" not in f and "code" not in f]
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


def load_dataset(dataset):
    paths = DATASETS[dataset]
    components = parse_pcm_repository(str(paths["model"]))
    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    sent_map = DocumentLoader.build_sent_map(sentences)
    name_to_id = {c.name: c.id for c in components}
    id_to_name = {c.id: c.name for c in components}
    return components, sentences, sent_map, name_to_id, id_to_name


print("=" * 100)
print("  JUDGE EFFECTIVENESS ANALYSIS: What FPs survive and why?")
print("=" * 100)

total_fps = 0
total_immune = 0
total_reachable = 0

for ds in DATASETS:
    data_pj = load_checkpoint(ds, "pre_judge")
    data_final = load_checkpoint(ds, "final")
    data3 = load_checkpoint(ds, "phase3")
    data1 = load_checkpoint(ds, "phase1")
    data4 = load_checkpoint(ds, "phase4")
    if not all([data_pj, data_final, data3, data1, data4]):
        continue

    components, sentences, sent_map, name_to_id, id_to_name = load_dataset(ds)
    gold = load_gold(ds)

    preliminary = data_pj["preliminary"]
    transarc_set = data_pj["transarc_set"]
    final = data_final["final"]
    doc_knowledge = data3["doc_knowledge"]
    model_knowledge = data1["model_knowledge"]
    ambiguous_names = model_knowledge.ambiguous_names if model_knowledge else set()

    final_set = {(l.sentence_number, l.component_id) for l in final}

    # Find FPs in final output
    fps = [l for l in final if (l.sentence_number, l.component_id) not in gold]
    fn_keys = gold - final_set

    print(f"\n{'─' * 80}")
    print(f"  {ds}: {len(fps)} FPs, {len(fn_keys)} FNs, {len(final)} total links")
    print(f"{'─' * 80}")

    if not fps:
        print(f"  No FPs — nothing for judge to catch")
        continue

    from llm_sad_sam.linkers.experimental.agent_linker_v26a import AgentLinkerV26a

    class Tester(AgentLinkerV26a):
        def __init__(self):
            self.doc_knowledge = doc_knowledge
            self.model_knowledge = model_knowledge
            self.GENERIC_COMPONENT_WORDS = data1.get("generic_component_words", set())
            self.GENERIC_PARTIALS = data1.get("generic_partials", set())

    tester = Tester()

    immune_count = 0
    reachable_count = 0
    by_reason = defaultdict(list)

    for fp in fps:
        key = (fp.sentence_number, fp.component_id)
        is_ta = key in transarc_set
        sent = sent_map.get(fp.sentence_number)
        sent_text = sent.text[:100] if sent else "?"
        is_ambig = fp.component_name in ambiguous_names if ambiguous_names else False

        # Classify why this FP survives the judge
        immunity_reason = None

        # Check each triage path in order (same as _judge_review)
        # 1. Syn-safe bypass
        if sent and tester._has_alias_mention(fp.component_name, sent.text):
            aliases = []
            for syn, target in doc_knowledge.synonyms.items():
                if target == fp.component_name and re.search(rf'\b{re.escape(syn.lower())}\b', sent.text.lower()):
                    aliases.append(f"syn:{syn}")
            for p, target in doc_knowledge.partial_references.items():
                if target == fp.component_name and re.search(rf'\b{re.escape(p.lower())}\b', sent.text.lower()):
                    aliases.append(f"partial:{p}")
            immunity_reason = f"SYN-SAFE ({', '.join(aliases)})"

        # 2. TransArc non-ambiguous (immune)
        elif is_ta and not is_ambig:
            immunity_reason = "TRANSARC_IMMUNE (non-ambiguous name)"

        # 3. TransArc ambiguous → deliberation (could be caught)
        elif is_ta and is_ambig:
            immunity_reason = None  # goes to deliberation, judge COULD catch it

        # 4. Standalone mention → safe
        elif sent and tester._has_standalone_mention(fp.component_name, sent.text):
            immunity_reason = "STANDALONE_SAFE (name found in text)"

        # Otherwise: reaches judge as no-match
        # (judge could catch it but didn't)

        if immunity_reason:
            immune_count += 1
            by_reason[immunity_reason.split(" ")[0]].append(fp)
            print(f"  [IMMUNE] S{fp.sentence_number}->{fp.component_name} | {immunity_reason} | src={fp.source}")
            print(f"           \"{sent_text}\"")
        else:
            reachable_count += 1
            if is_ta and is_ambig:
                by_reason["DELIBERATION_MISS"].append(fp)
                print(f"  [DELIB_MISS] S{fp.sentence_number}->{fp.component_name} | TransArc ambiguous, judge deliberated but approved | src={fp.source}")
            else:
                by_reason["JUDGE_MISS"].append(fp)
                print(f"  [JUDGE_MISS] S{fp.sentence_number}->{fp.component_name} | Reached judge, judge approved wrongly | src={fp.source}")
            print(f"           \"{sent_text}\"")

    print(f"\n  BREAKDOWN: {len(fps)} FPs")
    for reason, links in sorted(by_reason.items()):
        print(f"    {reason}: {len(links)}")
    print(f"  Judge-immune: {immune_count}, Reaches judge: {reachable_count}")

    total_fps += len(fps)
    total_immune += immune_count
    total_reachable += reachable_count

print(f"\n{'=' * 100}")
print(f"  TOTAL ACROSS ALL DATASETS: {total_fps} FPs")
print(f"    Judge-immune: {total_immune} ({100*total_immune/max(1,total_fps):.0f}%)")
print(f"    Reaches judge (missed): {total_reachable} ({100*total_reachable/max(1,total_fps):.0f}%)")
print(f"{'=' * 100}")
