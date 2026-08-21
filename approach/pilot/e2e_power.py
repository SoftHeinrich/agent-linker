"""What effect an end-to-end batch of n runs can actually resolve. No LLM calls.

The static round measured nine paraphrase arms neutral at their stage and then lost
three composed heads end to end. Before reading that as "the paraphrases hurt", the
design deserves its own measurement: **how big must an effect be for a paired
sign-flip test over n end-to-end runs to see it?**

`s_linker89` was the control in four separate batches this session
(`compact`, `static`, `minimal`, `solo`), three runs each per model, all with
identical code and settings. Those runs differ only by sampling, so their spread IS
the pipeline's run-to-run noise, measured rather than assumed.

The script reports that spread and the smallest |delta| the permutation test would
call non-flat (p <= 0.20) at each n, estimated by resampling the observed control
runs against themselves with a shift applied.

    ../.venv/bin/python pilot/e2e_power.py --variant s_linker89 --model luna
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import os
import statistics as st
import sys

BASE = "/mnt/hostshare/ardoco-home/alinker-replication-package"

PROJECTS = ("mediastore", "teastore", "teammates", "bigbluebutton", "jabref")
GOLD = {
    "mediastore": "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    "teastore": "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    "teammates": "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "bigbluebutton": "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "jabref": "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
}


def gold(proj):
    with open(os.path.join(BASE, "benchmark", GOLD[proj])) as fh:
        return {(int(r["sentence"]), r["modelElementID"]) for r in csv.DictReader(fh)}


def links_of(run_dir, variant, proj):
    fn = os.path.join(run_dir, f"{variant}_{proj}_links.csv")
    if not os.path.exists(fn):
        return None
    out = set()
    with open(fn) as fh:
        for r in csv.DictReader(fh):
            sent = r.get("sentence") or r.get("sentenceID")
            comp = r.get("modelElementID") or r.get("component_id") or r.get("componentID")
            if sent and comp:
                out.add((int(sent), comp))
    return out


def run_totals(run_dir, variant):
    tp = fp = 0
    for proj in PROJECTS:
        links = links_of(run_dir, variant, proj)
        if links is None:
            return None
        g = gold(proj)
        t = len(links & g)
        tp += t
        fp += len(links) - t
    return tp, fp


def min_detectable(values, n, alpha=0.20, trials=400):
    """Smallest constant shift the paired test calls non-flat at p <= alpha.

    Two arms of n runs are drawn from the SAME pool -- so any difference the test
    sees is noise -- and the second arm is shifted by d. The reported d is the
    smallest where the test flags at least half the draws.
    """
    if len(values) < 2 * n:
        return None
    splits = [s for s in itertools.combinations(range(2 * n), n) if 0 in s]
    truth = tuple(range(n))

    def flags(vals):
        def delta(group):
            left = [vals[i] for i in group]
            right = [vals[i] for i in range(2 * n) if i not in group]
            return sum(right) / len(right) - sum(left) / len(left)
        obs = delta(truth)
        p = sum(1 for s in splits if abs(delta(s)) >= abs(obs)) / len(splits)
        return p <= alpha

    import random
    rng = random.Random(0)
    for d in range(0, 60):
        hits = 0
        for _ in range(trials):
            draw = [rng.choice(values) for _ in range(2 * n)]
            shifted = draw[:n] + [v + d for v in draw[n:]]
            hits += flags(shifted)
        if hits >= trials / 2:
            return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="s_linker89")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    runs = []
    for tag in ("compact", "static", "minimal", "solo", "solo2"):
        for d in sorted(glob.glob(os.path.join(
                BASE, "results", f"{tag}_e2e_{args.model}_r*_20260821"))):
            if not os.path.isdir(d):
                continue
            tot = run_totals(d, args.variant)
            if tot:
                runs.append((tag, os.path.basename(d), *tot))

    if len(runs) < 4:
        sys.exit(f"only {len(runs)} runs of {args.variant} on {args.model}")

    tps = [r[2] for r in runs]
    fps = [r[3] for r in runs]
    print(f"{args.variant} on {args.model}: {len(runs)} runs, identical code, "
          f"across {len({r[0] for r in runs})} batches\n")
    print(f"  {'batch':<10}{'run':<34}{'TP':>6}{'FP':>6}")
    for tag, name, tp, fp in runs:
        print(f"  {tag:<10}{name:<34}{tp:>6}{fp:>6}")
    print(f"\n  TP  mean {st.mean(tps):6.1f}  sd {st.pstdev(tps):5.1f}  "
          f"range {min(tps)}-{max(tps)}  spread {max(tps) - min(tps)}")
    print(f"  FP  mean {st.mean(fps):6.1f}  sd {st.pstdev(fps):5.1f}  "
          f"range {min(fps)}-{max(fps)}  spread {max(fps) - min(fps)}")

    print("\n  smallest effect a paired sign-flip test resolves (p <= 0.20, "
          "half the draws):")
    for n in (3, 4, 5, 6):
        d_tp = min_detectable(tps, n)
        d_fp = min_detectable(fps, n)
        if d_tp is None and d_fp is None:
            print(f"    n = {n}: needs {2 * n} control runs, have {len(runs)}")
            continue
        print(f"    n = {n}: TP {d_tp if d_tp is not None else '-'} links, "
              f"FP {d_fp if d_fp is not None else '-'} links")


if __name__ == "__main__":
    main()
