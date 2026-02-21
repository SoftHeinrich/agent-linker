#!/usr/bin/env python3
"""AgentLinker Evaluation: Run conservative linker on all datasets.

Runs AgentLinker on all 5 benchmark datasets,
evaluates against full gold standards, and prints detailed error analysis.
"""

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.core import DocumentLoader, SadSamLink
from llm_sad_sam.linkers.experimental.agent_linker import (
    AgentLinker,
    export_links_csv,
)

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

BACKEND = LLMBackend.CLAUDE
os.environ["CLAUDE_MODEL"] = "sonnet"


def load_gold_sam(gold_path: str) -> set[tuple[int, str]]:
    """Load SAD-SAM gold standard as (sentence_number, component_id) pairs."""
    links = set()
    with open(gold_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def load_transarc_pairs(transarc_path: str) -> set[tuple[int, str]]:
    """Load TransArc baseline as (sentence_number, component_id) pairs."""
    pairs = set()
    with open(transarc_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                pairs.add((int(snum), cid))
    return pairs


def eval_metrics(predicted, gold):
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"tp": tp, "fp": fp, "fn": fn, "P": p, "R": r, "F1": f1}


def main():
    print(f"{'='*120}")
    print("AGENTLINKER EVALUATION: Conservative Implicit Refs + Source-Aware Judge")
    print(f"Backend: {BACKEND.value}, Model: {os.environ.get('CLAUDE_MODEL', 'default')}")
    print(f"{'='*120}")

    all_results = {}

    for ds_name, paths in DATASETS.items():
        print(f"\n{'='*120}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*120}")

        # Load data
        components = parse_pcm_repository(str(paths["model"]))
        id_to_name = {c.id: c.name for c in components}
        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        sent_map = {s.number: s for s in sentences}

        gold_pairs = load_gold_sam(str(paths["gold_sam"]))
        transarc_pairs = load_transarc_pairs(str(paths["transarc_sam"]))

        print(f"  Components: {len(components)}, Sentences: {len(sentences)}")
        print(f"  Gold links: {len(gold_pairs)}, TransArc baseline: {len(transarc_pairs)}")

        # TransArc baseline metrics
        ta_m = eval_metrics(transarc_pairs, gold_pairs)
        print(f"  TransArc baseline: P={ta_m['P']:.1%} R={ta_m['R']:.1%} F1={ta_m['F1']:.1%}")

        # Run AgentLinker
        print(f"\n  Running AgentLinker...")
        t0 = time.time()
        v45 = AgentLinker(backend=BACKEND)
        v45_preds = v45.link(
            text_path=str(paths["text"]),
            model_path=str(paths["model"]),
            transarc_csv=str(paths["transarc_sam"]),
        )
        elapsed = time.time() - t0

        # Build prediction sets
        pred_pairs = {(l.sentence_number, l.component_id) for l in v45_preds}
        pred_by_key = {(l.sentence_number, l.component_id): l for l in v45_preds}

        # Metrics against full gold
        m = eval_metrics(pred_pairs, gold_pairs)
        print(f"\n  AgentLinker Results: P={m['P']:.1%} R={m['R']:.1%} F1={m['F1']:.1%} ({elapsed:.0f}s)")
        print(f"  Links: {len(v45_preds)}, TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}")

        # Source breakdown
        source_counts = defaultdict(int)
        for l in v45_preds:
            source_counts[l.source] += 1
        print(f"  Sources: {dict(source_counts)}")

        # Save CSV output
        out_dir = Path("results/evaluation_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        export_links_csv(v45_preds, str(out_dir / f"agent_{ds_name}_links.csv"))

        # ======= DETAILED FP ANALYSIS =======
        fp_pairs = pred_pairs - gold_pairs
        print(f"\n  {'='*100}")
        print(f"  FALSE POSITIVES ({len(fp_pairs)})")
        print(f"  {'='*100}")

        # Gold links indexed by sentence
        gold_by_sent = defaultdict(list)
        for sn, cid in gold_pairs:
            gold_by_sent[sn].append(id_to_name.get(cid, cid[:20]))

        fp_details = []
        for sn, cid in sorted(fp_pairs):
            cname = id_to_name.get(cid, cid[:20])
            link = pred_by_key.get((sn, cid))
            sent = sent_map.get(sn)
            sent_text = sent.text if sent else "???"
            source = link.source if link else "???"
            conf = link.confidence if link else 0.0

            print(f"\n    FP: S{sn} -> {cname} (source={source}, conf={conf:.2f})")
            print(f"        \"{sent_text[:120]}\"")

            gold_for_sent = gold_by_sent.get(sn, [])
            if gold_for_sent:
                print(f"        Gold for S{sn}: {gold_for_sent}")
            else:
                print(f"        Gold for S{sn}: NONE (no gold links)")

            fp_details.append({
                "sentence": sn, "component": cname, "source": source,
                "confidence": conf, "text": sent_text[:150],
                "gold_for_sentence": gold_for_sent,
            })

        # ======= DETAILED FN ANALYSIS =======
        fn_pairs = gold_pairs - pred_pairs
        print(f"\n  {'='*100}")
        print(f"  FALSE NEGATIVES ({len(fn_pairs)})")
        print(f"  {'='*100}")

        fn_details = []
        for sn, cid in sorted(fn_pairs):
            cname = id_to_name.get(cid, cid[:20])
            sent = sent_map.get(sn)
            sent_text = sent.text if sent else "???"

            # Check if component name appears in sentence
            name_in_text = cname.lower() in sent_text.lower() if sent else False

            # Check if TransArc had this link
            transarc_had = (sn, cid) in transarc_pairs

            # Check what AgentLinker predicted for this sentence
            preds_for_sent = [(l.component_name, l.source, f"{l.confidence:.2f}")
                              for l in v45_preds if l.sentence_number == sn]

            print(f"\n    FN: S{sn} -> {cname} (name_in_text={name_in_text}, transarc={transarc_had})")
            print(f"        \"{sent_text[:120]}\"")
            if preds_for_sent:
                print(f"        AgentLinker predicted for S{sn}: {preds_for_sent}")
            else:
                print(f"        AgentLinker predicted for S{sn}: NOTHING")

            fn_details.append({
                "sentence": sn, "component": cname,
                "name_in_text": name_in_text, "transarc_had": transarc_had,
                "text": sent_text[:150],
                "v45_predicted": preds_for_sent,
            })

        all_results[ds_name] = {
            "P": m["P"], "R": m["R"], "F1": m["F1"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            "n_links": len(v45_preds),
            "time": elapsed,
            "sources": dict(source_counts),
            "transarc": {"P": ta_m["P"], "R": ta_m["R"], "F1": ta_m["F1"]},
            "fps": fp_details,
            "fns": fn_details,
        }

    # ======= CROSS-DATASET SUMMARY =======
    print(f"\n{'='*120}")
    print("AGENTLINKER CROSS-DATASET SUMMARY")
    print(f"{'='*120}")

    # Source breakdown across all datasets
    fp_by_source = defaultdict(int)
    fn_by_type = defaultdict(int)
    for ds_name, res in all_results.items():
        for fp in res["fps"]:
            fp_by_source[fp["source"]] += 1
        for fn in res["fns"]:
            if fn["name_in_text"]:
                fn_by_type["name_in_text_missed"] += 1
            elif fn["transarc_had"]:
                fn_by_type["transarc_had_lost"] += 1
            else:
                fn_by_type["no_textual_evidence"] += 1

    print(f"\n  FP by source: {dict(fp_by_source)}")
    print(f"  FN by type: {dict(fn_by_type)}")

    # Summary table
    print(f"\n  {'Dataset':<16} {'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6} "
          f"{'Links':>6} {'TA F1':>8} {'Time':>6}")
    print(f"  {'-'*86}")
    for ds_name, res in all_results.items():
        print(f"  {ds_name:<16} {res['P']:>7.1%} {res['R']:>7.1%} {res['F1']:>7.1%} "
              f"{res['tp']:>6} {res['fp']:>6} {res['fn']:>6} "
              f"{res['n_links']:>6} {res['transarc']['F1']:>7.1%} {res['time']:>5.0f}s")

    if all_results:
        n = len(all_results)
        avg_p = sum(r["P"] for r in all_results.values()) / n
        avg_r = sum(r["R"] for r in all_results.values()) / n
        avg_f1 = sum(r["F1"] for r in all_results.values()) / n
        avg_ta = sum(r["transarc"]["F1"] for r in all_results.values()) / n
        avg_time = sum(r["time"] for r in all_results.values()) / n
        print(f"  {'-'*86}")
        print(f"  {'MACRO AVG':<16} {avg_p:>7.1%} {avg_r:>7.1%} {avg_f1:>7.1%} "
              f"{'':>6} {'':>6} {'':>6} {'':>6} {avg_ta:>7.1%} {avg_time:>5.0f}s")

    # Source distribution per dataset
    print(f"\n  Source distribution per dataset:")
    all_sources = set()
    for res in all_results.values():
        all_sources.update(res["sources"].keys())
    all_sources = sorted(all_sources)

    header = f"  {'Dataset':<16}" + "".join(f" {s:>12}" for s in all_sources)
    print(header)
    print(f"  {'-'*len(header)}")
    for ds_name, res in all_results.items():
        row = f"  {ds_name:<16}"
        for s in all_sources:
            row += f" {res['sources'].get(s, 0):>12}"
        print(row)

    # Save detailed JSON results
    results_dir = Path("results/evaluation_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"agent_evaluation_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    if "--datasets" in sys.argv:
        idx = sys.argv.index("--datasets")
        selected = []
        for v in sys.argv[idx + 1:]:
            if v.startswith("--"):
                break
            selected.append(v)
        DATASETS = {k: v for k, v in DATASETS.items() if k in selected}
    main()
