#!/usr/bin/env python3
"""Unit test: Enhanced Judge on V21 FINAL FPs only.

Reconstructs V21's final link set, identifies which are FPs,
then tests the enhanced judge on ONLY those FPs (+ TPs for safety).
3 runs for variance.
"""

import csv
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

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
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
        "model": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
        "gold_sam": BENCHMARK_BASE / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
        "gold_sam": BENCHMARK_BASE / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
        "gold_sam": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
        "gold_sam": BENCHMARK_BASE / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
}

DS_SHORT = {'mediastore':'ms','teastore':'ts','teammates':'tm','bigbluebutton':'bbb','jabref':'jab'}

ENHANCED_JUDGE_PROMPT = """JUDGE: Review trace links between documentation sentences and software architecture components.

A sentence S should be linked to component C ONLY if ALL FOUR of the following are true:

RULE 1 — REFERENCE: S actually refers to C (not just string-matches its name).
  REJECT if the component name appears as part of a generic English phrase rather than as a
  component reference (e.g., "cascade logic" does not refer to a component named "BusinessLogic";
  "on the client side" does not refer to a component named "WebClient").

RULE 2 — ARCHITECTURAL LEVEL: S describes C at the architectural level (role, behavior,
  interactions with other components), NOT at the implementation level.
  REJECT if S describes:
  - Package or directory structure (e.g., "x.auth contains test cases for the AuthModule")
  - Internal classes or sub-components (e.g., "TokenValidator checks tokens" — this is about
    TokenValidator, an internal class, not the parent AuthModule component)
  - API exception handling details (e.g., "not found throws NotFoundException")
  - Data format or schema specifications

RULE 3 — TOPIC: C is the topic or subject of S (what S is primarily about).
  REJECT if C is mentioned incidentally while S discusses something else (e.g., "react upon
  changes within the interface" — S is about architecture patterns, the interface component is
  incidental; "RequestHandler stores it via the DataLayer" — S is about RequestHandler, not DataLayer).

RULE 4 — NOT GENERIC: The reference is to C as a specific architectural component, not as a
  generic English word (e.g., "managing transactions" does not refer to a component named
  "CoreServices"; "common data format" does not refer to a component named "SharedUtils").

COMPONENTS: {components}

For each link, respond APPROVE or REJECT with brief reason.

LINKS:
{links}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""


def load_gold(path):
    links = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def get_final_links(log_path, name_to_id, gold):
    """Reconstruct V21's final link set from log, classify as FP/TP."""
    with open(log_path) as f:
        content = f.read()

    # Find all links that survived to final output
    # Strategy: collect all introduced links, subtract rejected ones

    # 1. TransArc links (always survive unless boundary-filtered)
    transarc_links = set()
    # Parse from Phase 4 eval — all links at that point are TransArc
    m = re.search(r'EVAL after Phase 4.*?TP=(\d+).*?FP=(\d+)', content)

    # Better: parse "TransArc links: N" and then all the link details
    # Actually, let's trace through the phases more carefully.

    # Simplest approach: collect ALL mentioned S## -> Component links,
    # track which got rejected, and reconstruct the final set.

    all_introduced = set()  # (snum, comp_name, source)
    rejected = set()  # (snum, comp_name)

    # TransArc links from Phase 4
    in_transarc = False
    for line in content.split('\n'):
        if '[Phase 4]' in line:
            in_transarc = True
        if in_transarc and 'EVAL after Phase 4' in line:
            in_transarc = False

        # Entity/coref/validated links
        m = re.search(r'S(\d+) -> (.+?) \[(entity|coreference|validated|partial_inject|transarc)\]', line)
        if m:
            all_introduced.add((int(m.group(1)), m.group(2), m.group(3)))

        # Rejections
        m = re.search(r'Rejected: S(\d+) -> (.+?) \[', line)
        if m:
            rejected.add((int(m.group(1)), m.group(2)))
        m = re.search(r'Boundary filter.+?: S(\d+) -> (.+?) \(', line)
        if m:
            rejected.add((int(m.group(1)), m.group(2)))
        # Phase 6 generic mention rejects
        m = re.search(r'Generic mention reject: S(\d+) -> (.+)', line)
        if m:
            rejected.add((int(m.group(1)), m.group(2).strip()))
        # Coref verify-fail
        m = re.search(r"Coref verify-fail.*?: S(\d+) -> (.+)", line)
        if m:
            rejected.add((int(m.group(1)), m.group(2).strip()))

    # Parse TransArc links from the link lines in Phase 4 section
    # Actually, TransArc links don't show as "S## -> Comp [transarc]" in the log
    # They show in the eval. Let me parse differently.

    # Get final TP/FP counts to verify
    final_m = re.search(r'EVAL after Phase 9 \(final\).*?TP=(\d+).*?FP=(\d+).*?FN=(\d+).*?\[(\d+) links\]', content)
    if final_m:
        expected_total = int(final_m.group(4))
        expected_tp = int(final_m.group(1))
        expected_fp = int(final_m.group(2))
    else:
        return [], [], 0, 0

    # Surviving = introduced - rejected (deduplicated by snum,comp)
    surviving = {}
    for snum, comp, src in all_introduced:
        if (snum, comp) not in rejected:
            key = (snum, comp)
            if key not in surviving:
                surviving[key] = src

    # Classify as FP/TP
    fps = []
    tps = []
    for (snum, comp), src in sorted(surviving.items()):
        cid = name_to_id.get(comp)
        if not cid:
            continue
        if (snum, cid) in gold:
            tps.append((snum, comp, src))
        else:
            fps.append((snum, comp, src))

    return fps, tps, expected_fp, expected_tp


def test_dataset(dataset_name, sent_map, comp_names, fps, tps, run_id):
    """Test enhanced judge on final FPs + non-TransArc TPs."""

    # Only test non-transarc links (transarc are immune in pipeline)
    test_fps = [(s, c, src) for s, c, src in fps if src != "transarc"]
    test_tps = [(s, c, src) for s, c, src in tps if src != "transarc"]

    all_links = [(s, c, src, "FP") for s, c, src in test_fps] + \
                [(s, c, src, "TP") for s, c, src in test_tps]

    if not all_links:
        print(f"  {dataset_name} run{run_id}: No non-transarc links to judge")
        return 0, len(test_fps), 0, len(test_tps)

    # Build cases (same format as pipeline judge)
    cases = []
    for i, (snum, comp, src, label) in enumerate(all_links):
        sent = sent_map.get(snum)
        txt = sent.text if sent else "???"
        prev = sent_map.get(snum - 1)
        ctx = f"    PREV: {prev.text[:60]}...\n" if prev else ""
        cases.append(f"Case {i+1}: S{snum} -> {comp} (src:{src})\n{ctx}    >>> S{snum}: {txt}")

    prompt = ENHANCED_JUDGE_PROMPT.format(
        components=", ".join(comp_names),
        links="\n".join(cases)
    )

    llm = LLMClient(backend=LLMBackend.CLAUDE)
    response = llm.query(prompt, timeout=240)

    # Parse JSON array
    results = None
    if response.success and response.text:
        text = response.text.strip()
        # Try to find JSON object with judgments
        for i, ch in enumerate(text):
            if ch == '{':
                depth = 0
                for j in range(i, len(text)):
                    if text[j] == '{': depth += 1
                    elif text[j] == '}': depth -= 1
                    if depth == 0:
                        try:
                            results = json.loads(text[i:j+1])
                        except json.JSONDecodeError:
                            pass
                        break
                if results is not None:
                    break

    if not results or not isinstance(results, dict):
        print(f"  {dataset_name} run{run_id}: Judge returned unparseable response")
        return 0, len(test_fps), 0, len(test_tps)

    judgments = results.get("judgments", [])
    verdict_map = {}
    for j in judgments:
        idx = j.get("case", 0) - 1
        if 0 <= idx < len(all_links):
            verdict_map[idx] = j.get("approve", True)

    fp_rejected = 0
    tp_killed = 0

    print(f"\n  {dataset_name} run{run_id} ({len(test_fps)} FPs, {len(test_tps)} TPs to judge):")
    for i, (snum, comp, src, label) in enumerate(all_links):
        approved = verdict_map.get(i, True)
        if not approved:
            if label == "FP":
                fp_rejected += 1
                sent = sent_map.get(snum)
                txt = sent.text[:60] if sent else "?"
                print(f"    FP CAUGHT: S{snum} -> {comp} \"{txt}\"")
            elif label == "TP":
                tp_killed += 1
                sent = sent_map.get(snum)
                txt = sent.text[:60] if sent else "?"
                print(f"    TP KILLED: S{snum} -> {comp} \"{txt}\"")

    print(f"  FPs rejected: {fp_rejected}/{len(test_fps)}, TPs killed: {tp_killed}/{len(test_tps)}")
    return fp_rejected, len(test_fps), tp_killed, len(test_tps)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for ds_name in datasets:
        paths = DATASETS[ds_name]
        gold = load_gold(paths["gold_sam"])
        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        components = parse_pcm_repository(str(paths["model"]))
        sent_map = {s.number: s for s in sentences}
        name_to_id = {c.name: c.id for c in components}
        comp_names = sorted(set(c.name for c in components))

        ds_short = DS_SHORT[ds_name]
        log_path = f"/tmp/v21_run1_{ds_short}.log"

        fps, tps, exp_fp, exp_tp = get_final_links(log_path, name_to_id, gold)

        print(f"\n{'='*70}")
        print(f"ENHANCED JUDGE (FINAL FPs): {ds_name}")
        print(f"  Final FPs: {len(fps)} (expected {exp_fp}), Final TPs: {len(tps)} (expected {exp_tp})")

        # Show what the FPs are
        non_ta_fps = [(s,c,src) for s,c,src in fps if src != "transarc"]
        ta_fps = [(s,c,src) for s,c,src in fps if src == "transarc"]
        print(f"  TransArc FPs (immune): {len(ta_fps)}")
        print(f"  Non-TransArc FPs (judgeable): {len(non_ta_fps)}")
        for s, c, src in sorted(non_ta_fps):
            sent = sent_map.get(s)
            txt = sent.text[:70] if sent else "?"
            print(f"    S{s} -> {c} [{src}] \"{txt}\"")
        print(f"{'='*70}")

        for run in range(1, args.runs + 1):
            test_dataset(ds_name, sent_map, comp_names, fps, tps, run)
