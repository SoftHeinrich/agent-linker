#!/usr/bin/env python3
"""Compare V33f (ILinker2 seed) vs V33g (Java TransArc seed) on all datasets.

V33g uses deterministic Java TransArc CSV → zero Phase 4 variance.
Single run each since V33g Phase 4 is deterministic.
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

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
CLI_RESULTS = Path("/mnt/hostshare/ardoco-home/cli-results")

DATASETS = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
        "gold": BENCHMARK_BASE / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
        "transarc": CLI_RESULTS / "mediastore-sad-sam/sadSamTlr_mediastore.csv",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
        "model": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
        "gold": BENCHMARK_BASE / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
        "transarc": CLI_RESULTS / "teastore-sad-sam/sadSamTlr_teastore.csv",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
        "gold": BENCHMARK_BASE / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc": CLI_RESULTS / "teammates-sad-sam/sadSamTlr_teammates.csv",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
        "gold": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc": CLI_RESULTS / "bigbluebutton-sad-sam/sadSamTlr_bigbluebutton.csv",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
        "gold": BENCHMARK_BASE / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
        "transarc": CLI_RESULTS / "jabref-sad-sam/sadSamTlr_jabref.csv",
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


def run_variant(variant, text_path, model_path, transarc_csv=None):
    """Run a variant and return final links."""
    if variant == "v33f":
        os.environ["PHASE_CACHE_DIR"] = "./results/phase_cache_seed_compare/v33f"
        from llm_sad_sam.linkers.experimental.ilinker2_v33f import ILinker2V33f
        linker = ILinker2V33f(backend=LLMBackend.CLAUDE)
        return linker.link(text_path, model_path)
    elif variant == "v33g":
        os.environ["PHASE_CACHE_DIR"] = "./results/phase_cache_seed_compare/v33g"
        from llm_sad_sam.linkers.experimental.ilinker2_v33g import ILinker2V33g
        linker = ILinker2V33g(backend=LLMBackend.CLAUDE)
        return linker.link(text_path, model_path, transarc_csv=transarc_csv)


def main():
    selected = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())

    summary = []

    for ds_name in selected:
        paths = DATASETS[ds_name]
        gold = load_gold(str(paths["gold"]))

        print(f"\n{'='*100}")
        print(f"DATASET: {ds_name} (gold={len(gold)} links)")
        print(f"{'='*100}")

        for variant in ["v33g", "v33f"]:
            print(f"\n  ── {variant.upper()} ──")
            t0 = time.time()
            final_links = run_variant(
                variant,
                str(paths["text"]),
                str(paths["model"]),
                transarc_csv=str(paths["transarc"]),
            )
            elapsed = time.time() - t0

            link_set = {(l.sentence_number, l.component_id) for l in final_links}
            tp, fp, fn, p, r, f1 = eval_links(link_set, gold)

            from collections import Counter
            src_counts = Counter(l.source for l in final_links)

            print(f"  {variant}: {len(link_set)} links, {tp} TP, {fp} FP, {fn} FN, "
                  f"P={p:.1%} R={r:.1%} F1={f1:.1%} ({elapsed:.0f}s)")
            print(f"    Sources: {dict(src_counts)}")

            summary.append((ds_name, variant, len(link_set), tp, fp, fn, p, r, f1))

    # ── Summary table ──
    print(f"\n{'='*100}")
    print(f"SUMMARY: V33f (ILinker2 seed) vs V33g (Java TransArc seed)")
    print(f"{'='*100}")
    print(f"  {'Dataset':<15} {'Variant':<8} {'Links':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>7} {'R':>7} {'F1':>7}")
    print(f"  {'-'*70}")

    f1_by_variant = {"v33f": [], "v33g": []}
    for ds, var, links, tp, fp, fn, p, r, f1 in summary:
        print(f"  {ds:<15} {var:<8} {links:>5} {tp:>4} {fp:>4} {fn:>4} {p:>6.1%} {r:>6.1%} {f1:>6.1%}")
        f1_by_variant[var].append(f1)

    print(f"  {'-'*70}")
    for var in ["v33f", "v33g"]:
        avg = sum(f1_by_variant[var]) / len(f1_by_variant[var]) if f1_by_variant[var] else 0
        print(f"  {'MACRO AVG':<15} {var:<8} {'':>5} {'':>4} {'':>4} {'':>4} {'':>7} {'':>7} {avg:>6.1%}")


if __name__ == "__main__":
    main()
