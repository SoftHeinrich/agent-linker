#!/usr/bin/env python3
"""Unit test: Fix 3 — Subject extraction filter.

For each FP link S→C, ask LLM to identify the grammatical subject of S.
If the subject is NOT C (or a synonym), reject the link.
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

SUBJECT_PROMPT = """For each sentence below, identify the **grammatical subject** — the entity or concept that the sentence is primarily ABOUT.

The subject should be a specific named entity, concept, or pronoun. If the sentence is about a sub-component or internal class (e.g., "GateKeeper", "EmailSender"), that is the subject — not the parent component.

For each sentence, also state whether the given component is the subject, or just mentioned incidentally.

Sentences and their linked components:
{links}

Respond in JSON:
[
  {{"sentence": <number>, "component": "<linked component>", "subject": "<actual subject of sentence>", "component_is_subject": true|false, "reason": "<brief>"}}
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


def parse_links_from_log(log_path, name_to_id, gold, sent_map, link_type="FP"):
    """Parse FP or TP entity/coref links from V21 log."""
    rejected = set()
    with open(log_path) as f:
        content = f.read()

    for m in re.finditer(r'Rejected: S(\d+) -> (.+?) \[', content):
        rejected.add((int(m.group(1)), m.group(2)))
    for m in re.finditer(r'Boundary filter.+?: S(\d+) -> (.+?) \(', content):
        rejected.add((int(m.group(1)), m.group(2)))

    results = []
    seen = set()
    for m in re.finditer(r'S(\d+) -> (.+?) \[(entity|coreference|validated)\] "(.+?)"', content):
        snum = int(m.group(1))
        comp_name = m.group(2)

        cid = name_to_id.get(comp_name)
        if not cid:
            continue
        is_fp = (snum, cid) not in gold
        is_rejected = (snum, comp_name) in rejected

        if is_rejected:
            continue

        if (link_type == "FP" and is_fp) or (link_type == "TP" and not is_fp):
            key = (snum, comp_name)
            if key not in seen:
                seen.add(key)
                results.append((snum, comp_name))
    return results


def test_subject(dataset_name, run_id):
    """Test subject extraction on one dataset, one run."""
    paths = DATASETS[dataset_name]
    gold = load_gold(paths["gold_sam"])

    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    components = parse_pcm_repository(str(paths["model"]))
    sent_map = {s.number: s for s in sentences}
    name_to_id = {c.name: c.id for c in components}

    ds_short = {'mediastore':'ms','teastore':'ts','teammates':'tm','bigbluebutton':'bbb','jabref':'jab'}[dataset_name]
    log_path = f"/tmp/v21_run1_{ds_short}.log"

    fps = parse_links_from_log(log_path, name_to_id, gold, sent_map, "FP")
    tps = parse_links_from_log(log_path, name_to_id, gold, sent_map, "TP")

    all_links = [(s, c, "FP") for s, c in fps] + [(s, c, "TP") for s, c in tps]

    if not all_links:
        print(f"  {dataset_name} run{run_id}: No links to test")
        return 0, 0, 0, 0

    # Build prompt
    link_texts = []
    for snum, comp, label in all_links:
        sent = sent_map.get(snum)
        txt = sent.text if sent else "???"
        link_texts.append(f"  S{snum} -> {comp}: \"{txt}\"")

    prompt = SUBJECT_PROMPT.format(links="\n".join(link_texts))

    llm = LLMClient(backend=LLMBackend.CLAUDE)
    response = llm.query(prompt, timeout=240)

    # Parse JSON array from response (extract_json only handles dicts)
    results = None
    if response.success and response.text:
        text = response.text.strip()
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
        print(f"  {dataset_name} run{run_id}: LLM returned unparseable response")
        return 0, 0, 0, 0

    # Map results
    subject_map = {}
    for r in results:
        if isinstance(r, str):
            continue
        key = (r.get("sentence"), r.get("component"))
        subject_map[key] = {
            "subject": r.get("subject", ""),
            "is_subject": r.get("component_is_subject", True),
            "reason": r.get("reason", ""),
        }

    fp_rejected = 0
    fp_total = len(fps)
    tp_killed = 0
    tp_total = len(tps)

    print(f"\n  {dataset_name} run{run_id}:")
    for snum, comp, label in all_links:
        info = subject_map.get((snum, comp), {"is_subject": True})
        if not info["is_subject"]:
            if label == "FP":
                fp_rejected += 1
                print(f"    FP CAUGHT: S{snum} -> {comp} (subject: {info.get('subject','?')})")
            elif label == "TP":
                tp_killed += 1
                print(f"    TP KILLED: S{snum} -> {comp} (subject: {info.get('subject','?')})")

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
        print(f"SUBJECT EXTRACTION TEST: {ds}")
        print(f"{'='*70}")
        for run in range(1, args.runs + 1):
            test_subject(ds, run)
