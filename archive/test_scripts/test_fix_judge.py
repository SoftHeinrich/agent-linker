#!/usr/bin/env python3
"""Unit test: Enhanced Judge — 4-rule judge prompt.

Takes V21's FP links, sends them to an enhanced Phase 9 judge with explicit
rules about abstraction level, incidental mention, and name collision.
Runs 3 times per dataset for variance analysis.
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.llm_client import LLMClient, LLMBackend
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.core import DocumentLoader

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

ENHANCED_JUDGE_PROMPT = """You are reviewing trace links between documentation sentences and software architecture components.

A sentence S should be linked to component C ONLY if ALL of the following are true:
1. S **refers to** C (not just string-matches its name)
2. The reference is to C **as an architectural component** (not as a generic English word — e.g., "logic" meaning reasoning, "common" meaning shared)
3. C is the **topic or subject** of S (C is what S is about, not mentioned in passing or as context)
4. S describes C at the **architectural level** (role, behavior, interactions with other components) — NOT at the implementation level (package structure, class internals, API exception details)

REJECT if:
- S describes package/directory structure (e.g., "x.logic contains test cases for Logic")
- S describes internal sub-components or classes of C (e.g., "GateKeeper checks access rights" — this is about GateKeeper, a sub-component, not about Logic as an architectural unit)
- S mentions C incidentally while discussing something else (e.g., "react upon changes within the gui" — S is about architecture patterns, gui is incidental)
- C's name appears as a generic word, not a component reference (e.g., "cascade logic" is not about the Logic component)
- S describes API details, exception handling, or data format specs of C's internals

For each link below, respond with APPROVE or REJECT and a brief reason.

Components in this system: {components}

Links to review:
{links}

Respond in JSON format:
[
  {{"sentence": <number>, "component": "<name>", "verdict": "APPROVE|REJECT", "reason": "<brief>"}},
  ...
]"""


def load_gold(path):
    links = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def parse_fps_from_log(log_path, name_to_id, gold, sent_map):
    """Parse final FP links from V21 log."""
    rejected = set()
    with open(log_path) as f:
        content = f.read()

    for m in re.finditer(r'Rejected: S(\d+) -> (.+?) \[', content):
        rejected.add((int(m.group(1)), m.group(2)))
    for m in re.finditer(r'Boundary filter.+?: S(\d+) -> (.+?) \(', content):
        rejected.add((int(m.group(1)), m.group(2)))

    fps = []
    seen = set()
    for m in re.finditer(r'S(\d+) -> (.+?) \[(entity|coreference|validated)\] "(.+?)"', content):
        snum = int(m.group(1))
        comp_name = m.group(2)
        source = m.group(3)

        cid = name_to_id.get(comp_name)
        if cid and (snum, cid) not in gold and (snum, comp_name) not in rejected:
            key = (snum, comp_name)
            if key not in seen:
                seen.add(key)
                fps.append((snum, comp_name, source))
    return fps


def parse_tps_from_log(log_path, name_to_id, gold, sent_map):
    """Parse final TP links from V21 log (entity/coref only, not transarc)."""
    rejected = set()
    with open(log_path) as f:
        content = f.read()

    for m in re.finditer(r'Rejected: S(\d+) -> (.+?) \[', content):
        rejected.add((int(m.group(1)), m.group(2)))
    for m in re.finditer(r'Boundary filter.+?: S(\d+) -> (.+?) \(', content):
        rejected.add((int(m.group(1)), m.group(2)))

    tps = []
    seen = set()
    for m in re.finditer(r'S(\d+) -> (.+?) \[(entity|coreference|validated)\] "(.+?)"', content):
        snum = int(m.group(1))
        comp_name = m.group(2)
        source = m.group(3)

        cid = name_to_id.get(comp_name)
        if cid and (snum, cid) in gold and (snum, comp_name) not in rejected:
            key = (snum, comp_name)
            if key not in seen:
                seen.add(key)
                tps.append((snum, comp_name, source))
    return tps


def test_judge(dataset_name, run_id):
    """Test enhanced judge on one dataset, one run."""
    paths = DATASETS[dataset_name]
    gold = load_gold(paths["gold_sam"])

    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    components = parse_pcm_repository(str(paths["model"]))
    sent_map = {s.number: s for s in sentences}
    name_to_id = {c.name: c.id for c in components}

    ds_short = {'mediastore':'ms','teastore':'ts','teammates':'tm','bigbluebutton':'bbb','jabref':'jab'}[dataset_name]
    log_path = f"/tmp/v21_run1_{ds_short}.log"

    fps = parse_fps_from_log(log_path, name_to_id, gold, sent_map)
    tps = parse_tps_from_log(log_path, name_to_id, gold, sent_map)

    # Combine FPs + TPs for the judge (so we can measure both rejection and safety)
    all_links = [(s, c, src, "FP") for s, c, src in fps] + [(s, c, src, "TP") for s, c, src in tps]

    if not all_links:
        print(f"  {dataset_name} run{run_id}: No entity/coref links to judge")
        return 0, 0, 0, 0

    # Build prompt
    comp_names = sorted(set(c.name for c in components))
    link_texts = []
    for snum, comp, src, label in all_links:
        sent = sent_map.get(snum)
        txt = sent.text if sent else "???"
        link_texts.append(f"  S{snum} -> {comp}: \"{txt}\"")

    prompt = ENHANCED_JUDGE_PROMPT.format(
        components=", ".join(comp_names),
        links="\n".join(link_texts)
    )

    llm = LLMClient(backend=LLMBackend.CLAUDE)
    response = llm.query(prompt, timeout=240)

    # Parse JSON array from response (extract_json only handles dicts)
    results = None
    if response.success and response.text:
        text = response.text.strip()
        # Try to find JSON array
        for i, ch in enumerate(text):
            if ch == '[':
                depth = 0
                for j in range(i, len(text)):
                    if text[j] == '[': depth += 1
                    elif text[j] == ']': depth -= 1
                    if depth == 0:
                        try:
                            results = json.loads(text[i:j+1])
                        except json.JSONDecodeError:
                            pass
                        break
                if results is not None:
                    break
        # Fallback: try extract_json for dict responses
        if results is None:
            results = llm.extract_json(response)
            if isinstance(results, dict):
                if "sentence" in results:
                    results = [results]
                else:
                    for v in results.values():
                        if isinstance(v, list):
                            results = v
                            break

    if not results or not isinstance(results, list):
        print(f"  {dataset_name} run{run_id}: Judge returned unparseable response")
        return 0, 0, 0, 0

    # Map results back
    verdict_map = {}
    for r in results:
        if isinstance(r, str):
            continue
        key = (r.get("sentence"), r.get("component"))
        verdict_map[key] = r.get("verdict", "APPROVE").upper()

    fp_rejected = 0
    fp_total = len(fps)
    tp_killed = 0
    tp_total = len(tps)

    print(f"\n  {dataset_name} run{run_id}:")
    for snum, comp, src, label in all_links:
        verdict = verdict_map.get((snum, comp), "APPROVE")
        if label == "FP" and verdict == "REJECT":
            fp_rejected += 1
            print(f"    FP CAUGHT: S{snum} -> {comp} ({verdict})")
        elif label == "TP" and verdict == "REJECT":
            tp_killed += 1
            print(f"    TP KILLED: S{snum} -> {comp} ({verdict})")

    print(f"  FPs rejected: {fp_rejected}/{fp_total}")
    print(f"  TPs killed: {tp_killed}/{tp_total}")

    return fp_rejected, fp_total, tp_killed, tp_total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        print(f"\n{'='*70}")
        print(f"ENHANCED JUDGE TEST: {ds}")
        print(f"{'='*70}")
        for run in range(1, args.runs + 1):
            test_judge(ds, run)
