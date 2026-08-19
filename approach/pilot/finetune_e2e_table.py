"""The finetune round's E2E table, assembled from the run directories.

Reads every `ablation_*.json` in the given run directories, folds each run into per-arm
macro F1 / macro F2 / TP / FP, counts the LLM calls each arm actually sent from the call
logs, and prints the markdown table `../results/finetune_round/README.md` carries. Deltas
are printed against the arm named by `--control` **and** against the null arm, because
this harness's null is not zero (`../results/prompt_round`: TP -4.8, macro F1 -0.7 from an
empty file diff).

No permutation test here on purpose: `pilot/score_runs.py` is where paired testing lives,
and it is run separately on the same directories.

    ../.venv/bin/python pilot/finetune_e2e_table.py ../results/s75_e2e_r*_20260819
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


def runs_metrics(dirs):
    """arm -> list of (macroF1, macroF2, TP, FP), one entry per run directory."""
    out = defaultdict(list)
    for d in dirs:
        files = sorted(Path(d).glob("ablation_*.json"))
        if not files:
            continue
        data = json.load(files[-1].open())
        per = defaultdict(dict)
        for project, arms in data.items():
            if not isinstance(arms, dict):
                continue
            for arm, m in arms.items():
                if isinstance(m, dict) and "F1" in m:
                    per[arm][project] = m
        for arm, pm in per.items():
            if len(pm) != 5:
                continue
            n = len(pm)
            out[arm].append((
                100 * sum(x["F1"] for x in pm.values()) / n,
                100 * sum(x.get("F2", 0.0) for x in pm.values()) / n,
                sum(x["tp"] for x in pm.values()),
                sum(x["fp"] for x in pm.values()),
            ))
    return out


def calls(dirs, arm):
    """LLM calls per five-project run for one arm, from its own call logs."""
    totals = []
    for d in dirs:
        n = 0
        # `s_linker75_*` also matches `s_linker75_null_*`; the backend tag pins it.
        for path in glob.glob(f"{d}/llm_logs/{arm}_openai_*_calls.json"):
            try:
                n += len(json.load(open(path)))
            except Exception:
                pass
        if n:
            totals.append(n)
    return sum(totals) / len(totals) if totals else float("nan")


def mean(rows, i):
    return sum(r[i] for r in rows) / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--control", default="s_linker74")
    ap.add_argument("--null", default="s_linker75_null")
    args = ap.parse_args()

    metrics = runs_metrics(args.dirs)
    if not metrics:
        raise SystemExit("no complete five-project runs found")
    order = ([args.control] if args.control in metrics else []) + \
            ([args.null] if args.null in metrics else []) + \
            [a for a in sorted(metrics) if a not in (args.control, args.null)]

    base = metrics.get(args.control)
    print(f"\n| arm | n | TP | FP | macro F1 | macro F2 | calls | F1 range |")
    print("|---|---|---|---|---|---|---|---|")
    for arm in order:
        rows = metrics[arm]
        f1s = [r[0] for r in rows]
        print(f"| `{arm}` | {len(rows)} | {mean(rows, 2):.1f} | {mean(rows, 3):.1f} "
              f"| {mean(rows, 0):.2f} | {mean(rows, 1):.2f} | {calls(args.dirs, arm):.0f} "
              f"| {max(f1s) - min(f1s):.2f} |")

    if base:
        print(f"\ndeltas against `{args.control}` (same invocation set):")
        for arm in order:
            if arm == args.control:
                continue
            rows = metrics[arm]
            print(f"  {arm:20} F1 {mean(rows,0)-mean(base,0):+.2f}   "
                  f"F2 {mean(rows,1)-mean(base,1):+.2f}   "
                  f"TP {mean(rows,2)-mean(base,2):+.1f}   "
                  f"FP {mean(rows,3)-mean(base,3):+.1f}")
    null = metrics.get(args.null)
    if null and base:
        d = mean(null, 0) - mean(base, 0)
        print(f"\nthe null arm's own delta is F1 {d:+.2f} / TP "
              f"{mean(null,2)-mean(base,2):+.1f} — read every row above against it, "
              f"not against zero.")

    print("\nper run:")
    for arm in order:
        print(f"  {arm:20} F1 {[round(r[0], 2) for r in metrics[arm]]}")


if __name__ == "__main__":
    main()
