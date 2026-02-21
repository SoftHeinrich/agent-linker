#!/usr/bin/env python3
"""Unit test: Fix 2 — Subprocess-mention filter.

For each dataset, loads V21 Phase 2 subprocess terms + final FP links.
Checks which FPs have a subprocess term mentioned in the sentence text.
Deterministic — 1 run per dataset.
"""

import csv
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.core import DocumentLoader, SadSamLink

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


def load_gold(path):
    links = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def run_v21_phases(dataset_name):
    """Run V21 through Phase 4 (transarc) + Phase 2 (subprocesses) + Phase 5-9 to get final links."""
    from llm_sad_sam.linkers.experimental.agent_linker_v21 import AgentLinkerV21

    paths = DATASETS[dataset_name]
    linker = AgentLinkerV21(backend=LLMBackend.CLAUDE)

    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    components = parse_pcm_repository(str(paths["model"]))
    sent_map = {s.number: s for s in sentences}
    name_to_id = {c.name: c.id for c in components}
    id_to_name = {c.id: c.name for c in components}

    # Run full pipeline to get final links + Phase 2 data
    final_links = linker.link(str(paths["text"]), str(paths["model"]),
                               transarc_csv=str(paths["transarc_sam"]))

    return final_links, linker, sentences, components, sent_map, name_to_id, id_to_name


def subprocess_filter(sentence_text, subprocess_terms):
    """Check if sentence mentions any subprocess term (case-insensitive)."""
    text_lower = sentence_text.lower()
    for term in subprocess_terms:
        # Match as whole word or with common suffixes
        pattern = re.compile(r'\b' + re.escape(term.lower()) + r's?\b', re.IGNORECASE)
        if pattern.search(sentence_text):
            return term
    return None


def test_dataset(dataset_name):
    """Run the subprocess filter test on one dataset."""
    paths = DATASETS[dataset_name]
    gold = load_gold(paths["gold_sam"])

    # We need subprocess terms from Phase 2. Run V21 to get them.
    # But running full V21 is slow. Instead, parse from existing log.
    ds_short = {'mediastore':'ms','teastore':'ts','teammates':'tm','bigbluebutton':'bbb','jabref':'jab'}[dataset_name]
    log_path = f"/tmp/v21_run1_{ds_short}.log"

    # Parse subprocess terms from log
    subprocess_terms = []
    with open(log_path) as f:
        for line in f:
            if "Subprocess terms:" in line:
                # Extract list from string
                match = re.search(r"\[(.+)\]", line)
                if match:
                    subprocess_terms = [t.strip().strip("'\"") for t in match.group(1).split(",")]
                break

    # Parse final links from log
    # We need the actual link pairs. Parse from the Phase 9 eval section.
    # Actually, let's load the gold standard and compute FPs from the log's TP/FP counts.
    # Better: load sentences and parse the FP lines from the log.

    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    components = parse_pcm_repository(str(paths["model"]))
    sent_map = {s.number: s for s in sentences}
    name_to_id = {c.name: c.id for c in components}
    id_to_name = {c.id: c.name for c in components}

    # Parse ALL links that made it to Phase 9 final from the log
    # We need to identify which are FPs. Parse from "New FPs" lines.
    # Actually, let's collect all entity/coref links that survived to final.
    # Simpler: parse the log for all S## -> Component [source] lines after Phase 9

    # Parse FPs: all links that are in predicted but not in gold
    # We need the full predicted set. Parse from log.
    # The log shows final link count and TP/FP/FN. But not the actual pairs.

    # Best approach: collect all links mentioned in log that survived to final.
    # Look for FP links specifically — they appear in "New FPs" sections and
    # were NOT rejected in later phases.

    # Actually, let me just take the FP entity/coref links from the full log
    # and filter out ones that were rejected.

    # Simplest: parse the gold standard, and for each FP link in V21's output,
    # check if the sentence mentions a subprocess.

    # We need V21's final output links. Let me parse them differently.
    # The log shows phase-by-phase. After Phase 9, it shows FNs but not FPs explicitly.
    # Let me reconstruct from the log.

    # Actually, let me just collect all "S## -> Component [source]" lines that are FPs.
    # I'll gather all entity/coref links introduced, subtract those rejected.

    all_fps = []  # (snum, component_name, source, text)
    rejected = set()  # (snum, component_name) pairs rejected by judge/filters

    with open(log_path) as f:
        content = f.read()

    # Parse rejected links
    for m in re.finditer(r'Rejected: S(\d+) -> (.+?) \[', content):
        rejected.add((int(m.group(1)), m.group(2)))
    for m in re.finditer(r'Boundary filter.+?: S(\d+) -> (.+?) \(', content):
        rejected.add((int(m.group(1)), m.group(2)))

    # Parse all introduced entity/coref links (FPs only — not in gold)
    for m in re.finditer(r'S(\d+) -> (.+?) \[(entity|coreference|validated)\] "(.+?)"', content):
        snum = int(m.group(1))
        comp_name = m.group(2)
        source = m.group(3)
        text_preview = m.group(4)

        # Check if it's an FP (not in gold)
        cid = name_to_id.get(comp_name)
        if cid and (snum, cid) not in gold:
            # Check if it was rejected
            if (snum, comp_name) not in rejected:
                full_text = sent_map[snum].text if snum in sent_map else text_preview
                all_fps.append((snum, comp_name, source, full_text))

    # Deduplicate
    seen = set()
    unique_fps = []
    for snum, comp, src, txt in all_fps:
        key = (snum, comp)
        if key not in seen:
            seen.add(key)
            unique_fps.append((snum, comp, src, txt))

    print(f"\n{'='*70}")
    print(f"SUBPROCESS FILTER TEST: {dataset_name}")
    print(f"Subprocess terms: {subprocess_terms}")
    print(f"Total final FPs: {len(unique_fps)}")
    print(f"{'='*70}")

    caught = []
    missed = []

    for snum, comp, src, txt in sorted(unique_fps):
        match = subprocess_filter(txt, subprocess_terms)
        if match:
            caught.append((snum, comp, match, txt[:80]))
            print(f"  CAUGHT: S{snum} -> {comp} (subprocess: '{match}')")
            print(f"          \"{txt[:80]}\"")
        else:
            missed.append((snum, comp, txt[:80]))

    print(f"\n  Summary: {len(caught)}/{len(unique_fps)} FPs caught by subprocess filter")
    if missed:
        print(f"  Remaining FPs ({len(missed)}):")
        for snum, comp, txt in sorted(missed):
            print(f"    S{snum} -> {comp}: \"{txt}\"")

    # Check for TP kills (false negatives from filter)
    tp_kills = []
    for snum, comp, src, txt in sorted(unique_fps):
        cid = name_to_id.get(comp)
        if cid and (snum, cid) in gold:
            match = subprocess_filter(txt, subprocess_terms)
            if match:
                tp_kills.append((snum, comp, match))

    # Also check: would the filter kill any TPs in the FULL link set?
    # Parse TP links too
    print(f"\n  Checking TP safety...")
    tp_killed = 0
    for s in sentences:
        for comp in components:
            cid = comp.id
            if (s.number, cid) in gold:
                match = subprocess_filter(s.text, subprocess_terms)
                if match:
                    # This TP sentence mentions a subprocess — would be killed
                    print(f"  WARNING TP KILL: S{s.number} -> {comp.name} (subprocess: '{match}')")
                    tp_killed += 1

    if tp_killed == 0:
        print(f"  TP safety: OK (0 TPs would be killed)")
    else:
        print(f"  TP safety: DANGER ({tp_killed} TPs would be killed!)")

    return len(caught), len(unique_fps), tp_killed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    total_caught = 0
    total_fps = 0
    total_tp_kills = 0

    for ds in datasets:
        caught, fps, tp_kills = test_dataset(ds)
        total_caught += caught
        total_fps += fps
        total_tp_kills += tp_kills

    print(f"\n{'='*70}")
    print(f"OVERALL: {total_caught}/{total_fps} FPs caught, {total_tp_kills} TP kills")
    print(f"{'='*70}")
