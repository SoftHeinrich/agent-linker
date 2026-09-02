#!/usr/bin/env python3
"""Score one \\approach arm against another and say whether the move is real.

The arm swap question ("is s110 better than s92a?") cannot be answered from the two
Average rows: both arms are stochastic, and on this benchmark a single run moves the
headline metrics by more than a typical arm delta. So this reports, per metric:

    mean delta, the per-run deltas behind it, and how many of the N runs agree on the sign

A delta whose runs disagree on the sign is reported as INSIDE NOISE no matter how large
the mean is -- that is the whole point of the tool, and the reason arm decisions are not
made off the Average row alone.

Run order (both arms must already be scored by rq12.py):

    python3 evaluation/mini-src/rq12.py               # incumbent -> RQ12_BIGTABLE.csv
    python3 evaluation/mini-src/rq12.py --arm s110    # candidate  -> RQ12_BIGTABLE_s110.csv
    python3 studies/compare_arms.py s110              # -> the verdict table
    python3 studies/compare_arms.py s110 --csv evaluation/reports/ARM_COMPARE_s110.csv

Exit 0 always: this reports, it does not gate. The call is the author's.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "evaluation" / "mini-src"))
import metrics as m   # noqa: E402  (the tree's one P/R/F1; never re-derived here)

REPORTS = m.REPO / "evaluation" / "reports"

# The \approach rows; the baselines are arm-independent, so comparing them is meaningless.
ARMS_UNDER_TEST = ["approach (GPT-5.6-terra)", "approach (GPT-5.6-luna)"]

# The metrics an arm decision turns on: the two headline link/file scores, and the
# size-aware suite the paper argues the decision should actually be made on.
HEADLINE = [
    ("doc_to_model_link_f1", "dm F1"),
    ("doc_to_model_link_f2", "dm F2"),
    ("doc_to_code_file_f1", "dc F1"),
    ("doc_to_code_file_f2", "dc F2"),
]
TAIL = [
    ("doc_to_model_component_miss_rate", "dm CMR%"),
    ("doc_to_code_worst_component_f1", "dc worst F1"),
    ("doc_to_code_harmonic_component_f1", "dc harm F1"),
]
# CMR is a miss rate: lower is better, so its delta sign is flipped when read as "gain".
LOWER_IS_BETTER = {"doc_to_model_component_miss_rate"}


def load(path):
    """{(system, run): {metric: float}} for one arm's RQ12_BIGTABLE."""
    if not path.is_file():
        raise SystemExit(f"[compare] missing {path}\n[compare] score that arm first: "
                         f"python3 mini-src/rq12.py --arm <arm>")
    out = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        vals = {}
        for k, v in row.items():
            if k in ("system", "backend", "run") or v in (None, ""):
                continue
            try:
                vals[k] = float(v)
            except ValueError:
                pass
        out[(row["system"], row["run"])] = vals
    return out


def runs_for(table, system):
    """The per-run keys for `system`, in file order, excluding the average row."""
    return [r for (s, r) in table if s == system and r not in ("average", "single")]


def compare(system, base, cand, metric):
    """(mean delta, per-run deltas, agreeing runs, n) in points, candidate minus base."""
    runs = [r for r in runs_for(base, system) if (system, r) in cand]
    deltas = []
    for r in runs:
        b, c = base.get((system, r), {}), cand.get((system, r), {})
        if metric in b and metric in c:
            d = (c[metric] - b[metric])
            # CMR is already a percentage in the CSV; the F-scores are 0-1 fractions.
            d = d if metric in LOWER_IS_BETTER else d * 100
            deltas.append(-d if metric in LOWER_IS_BETTER else d)
    if not deltas:
        return None
    agree = max(sum(1 for d in deltas if d > 0), sum(1 for d in deltas if d < 0))
    return st.mean(deltas), deltas, agree, len(deltas)


def verdict(mean, deltas, agree, n):
    """Sign agreement first, magnitude second -- a split sign is noise at any size."""
    if not any(deltas):
        return "NO CHANGE"
    if n < 2:
        return "1 RUN ONLY"
    if agree < n:
        return "INSIDE NOISE"
    sd = st.stdev(deltas)
    if sd and abs(mean) < sd:
        return "WEAK"
    return "BETTER" if mean > 0 else "WORSE"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate", help="candidate arm, e.g. s110")
    ap.add_argument("--base", default="s92a",
                    help="the arm to compare against (default: s92a, the arm s110 replaced)")
    ap.add_argument("--csv", default=None, help="also write the comparison to this CSV")
    args = ap.parse_args()

    def table_path(arm):
        # The incumbent is written unsuffixed, so the suffix rule has to follow whatever
        # rq12.py currently calls the incumbent -- hardcoding "s92a" here would send this
        # tool looking for RQ12_BIGTABLE_s110.csv after s110 was promoted. Read from the
        # source text, not by import: the arm names are the same length, so an edit and a
        # run in the same second leave a stale .pyc (same reason as check.py).
        found = re.search(r'^DEFAULT_ARM\s*=\s*["\'](?P<arm>[^"\']+)["\']',
                          (HERE / "rq12.py").read_text(encoding="utf-8"), re.M)
        incumbent = found.group("arm") if found else "s110"
        suffix = "" if arm == incumbent else f"_{arm}"
        return REPORTS / f"RQ12_BIGTABLE{suffix}.csv"

    base, cand = load(table_path(args.base)), load(table_path(args.candidate))
    print(f"arm comparison: {args.candidate} vs {args.base}  "
          f"(delta in points, candidate minus incumbent; CMR flipped so + is always better)\n")

    out_rows = []
    for system in ARMS_UNDER_TEST:
        if not any(s == system for s, _ in cand):
            print(f"  {system}: absent from the candidate arm -- skipped\n")
            continue
        print(f"  {system}")
        print(f"    {'metric':<14}{'mean':>8}{'per-run':>26}{'sign':>7}  verdict")
        for group in (HEADLINE, TAIL):
            for metric, label in group:
                got = compare(system, base, cand, metric)
                if not got:
                    continue
                mean, deltas, agree, n = got
                v = verdict(mean, deltas, agree, n)
                runs = " ".join(f"{d:+.2f}" for d in deltas)
                print(f"    {label:<14}{mean:>+8.2f}{runs:>26}{f'{agree}/{n}':>7}  {v}")
                out_rows.append({"system": system, "metric": metric, "label": label,
                                 "mean_delta": f"{mean:.4f}", "n_runs": n,
                                 "sign_agree": agree, "verdict": v,
                                 "per_run": " ".join(f"{d:.4f}" for d in deltas)})
            print()

    print("Reading: BETTER/WORSE = every run agrees on the sign and |mean| >= sd.")
    print("         WEAK = signs agree but the mean is inside one sd.")
    print("         INSIDE NOISE = the runs disagree on the sign; the mean is not evidence.")

    if args.csv:
        p = Path(args.csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\n[compare] wrote {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
