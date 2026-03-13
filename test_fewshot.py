#!/usr/bin/env python3
"""Unit test: Compare ILinker2 few-shot vs baseline on Phase 4 seed quality.

Runs ILinker2 standalone on each dataset and compares link precision/recall.
This tests ONLY the seed — no downstream phases (judge, coref, etc).
"""

import csv
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

os.environ["CLAUDE_MODEL"] = "sonnet"

from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.linkers.experimental.ilinker2 import ILinker2
from llm_sad_sam.linkers.experimental.ilinker2_fewshot import ILinker2FewShot

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)

DATASETS = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
        "gold": BENCHMARK_BASE / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
        "model": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
        "gold": BENCHMARK_BASE / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
        "gold": BENCHMARK_BASE / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
        "gold": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
        "gold": BENCHMARK_BASE / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
}


def load_gold(gold_path):
    links = set()
    with open(gold_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def eval_links(link_set, gold):
    tp = len(link_set & gold)
    fp = len(link_set - gold)
    fn = len(gold - link_set)
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return tp, fp, fn, p, r, f1


# Load .env file
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"])
    args = parser.parse_args()

    _backend_env = os.environ.get("LLM_BACKEND", "claude")
    backend = LLMBackend.OPENAI if _backend_env == "openai" else LLMBackend.CLAUDE
    print(f"Backend: {backend.value}")

    baseline = ILinker2(backend=backend)
    fewshot = ILinker2FewShot(backend=backend)

    summary = []

    for ds_name in args.datasets:
        paths = DATASETS[ds_name]
        gold = load_gold(str(paths["gold"]))
        text_path = str(paths["text"])
        model_path = str(paths["model"])

        print(f"\n{'=' * 90}")
        print(f"DATASET: {ds_name} (gold={len(gold)} links)")
        print(f"{'=' * 90}")

        for label, linker in [("baseline", baseline), ("fewshot", fewshot)]:
            print(f"\n  ── {label} ──")
            t0 = time.time()
            links = linker.link(text_path, model_path)
            elapsed = time.time() - t0

            link_set = {(l.sentence_number, l.component_id) for l in links}
            tp, fp, fn, p, r, f1 = eval_links(link_set, gold)

            print(f"  {label}: {len(link_set)} links, TP={tp} FP={fp} FN={fn} "
                  f"P={p:.1%} R={r:.1%} F1={f1:.1%} ({elapsed:.0f}s)")

            # Show FP details
            fp_links = link_set - gold
            if fp_links:
                from llm_sad_sam.pcm_parser import parse_pcm_repository
                comps = parse_pcm_repository(model_path)
                id_to_name = {c.id: c.name for c in comps}
                print(f"    FP: {', '.join(f'S{s}->{id_to_name.get(c,c)}' for s,c in sorted(fp_links))}")

            # Show FN details (only what seed misses)
            fn_links = gold - link_set
            if fn_links:
                from llm_sad_sam.pcm_parser import parse_pcm_repository
                comps = parse_pcm_repository(model_path)
                id_to_name = {c.id: c.name for c in comps}
                print(f"    FN: {', '.join(f'S{s}->{id_to_name.get(c,c)}' for s,c in sorted(fn_links))}")

            summary.append((ds_name, label, len(link_set), tp, fp, fn, p, r, f1))

    # Summary table
    print(f"\n{'=' * 90}")
    print("SUMMARY: ILinker2 Seed Quality (baseline vs fewshot)")
    print(f"{'=' * 90}")
    print(f"  {'Dataset':<15} {'Variant':<10} {'Links':>5} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'P':>7} {'R':>7} {'F1':>7}")
    print(f"  {'-' * 75}")

    f1_by_variant = {"baseline": [], "fewshot": []}
    for ds, var, links, tp, fp, fn, p, r, f1 in summary:
        print(f"  {ds:<15} {var:<10} {links:>5} {tp:>4} {fp:>4} {fn:>4} "
              f"{p:>6.1%} {r:>6.1%} {f1:>6.1%}")
        f1_by_variant[var].append(f1)

    print(f"  {'-' * 75}")
    for var in ["baseline", "fewshot"]:
        if f1_by_variant[var]:
            avg = sum(f1_by_variant[var]) / len(f1_by_variant[var])
            print(f"  {'MACRO AVG':<15} {var:<10} {'':>5} {'':>4} {'':>4} {'':>4} "
                  f"{'':>7} {'':>7} {avg:>6.1%}")

    # Delta
    print(f"\n  DELTA (fewshot - baseline):")
    for ds_name in args.datasets:
        bl = [x for x in summary if x[0] == ds_name and x[1] == "baseline"][0]
        fs = [x for x in summary if x[0] == ds_name and x[1] == "fewshot"][0]
        df1 = fs[8] - bl[8]
        dfp = int(fs[5] - bl[5])
        dfn = int(fs[6] - bl[6])
        print(f"    {ds_name:<15} ΔF1={df1:+.1%}  ΔFP={dfp:+d}  ΔFN={dfn:+d}")


if __name__ == "__main__":
    main()
