#!/usr/bin/env python3
"""Exploration: a broad family of TAIL and COVERAGE metrics over the per-component
F1 distribution, and each metric's Spearman correlation to the reference link/file
F1 across all (system, project) cells.

Goal: find metrics that add signal link-F1 does NOT (low |rho|) and are not
saturated, for BOTH tasks (sad-sam doc-model, sad-code doc-code). The paper's
current pair (worst, harmonic) sits at rho .67/.70 on doc-code and .79/.83 on
doc-model -- this sweep asks whether any other tail/coverage summary is more
independent while still ranking the approach first.

Stdlib only. Reuses metrics.py loaders + prf + the doc-code enrollment chain; the
per-component F1 list is rebuilt here for both tasks (copy-not-import for the
distribution, loaders imported)."""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from _roots import MINI_SRC, SOTA    # shared roots (see _roots.py)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(MINI_SRC))
import metrics as m            # loaders, prf, enrollment chain
from metrics import spearman   # tie-aware stdlib Spearman

PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]

# 6 systems x 5 projects = 30 cells (aalinker = mean of 3 runs; baselines single).
SYSTEMS = [
    ("S21 GPT",   "gpt-5.4_s21",  ["run1","run2","run3"]),
    ("S21 Claude","sonnet_s21",   ["run1","run2","run3"]),
    ("s20u GPT",  "gpt-5.4_full", ["run1","run2","run3"]),
    ("s20u Claude","sonnet_full", ["run1","run2","run3"]),
    ("Artemis",   "__artemis__",  [None]),
    ("TransArC",  "__transarc__", [None]),
]

def path_for(slot, task, run, project):
    if slot == "__artemis__":
        sub = "model-doc/artemis-{p}-gpt-5.4.csv" if task=="sad-sam" else "doc-code/artemis-{p}-gpt-5.4.csv"
        return SOTA / sub.format(p=project)
    if slot == "__transarc__":
        sub = "model-doc/swattr-{p}.csv" if task=="sad-sam" else "doc-code/transarc-{p}.csv"
        return SOTA / sub.format(p=project)
    if task == "sad-sam":
        return SOTA / f"model-doc/aalinker/{slot}/{run}/{project}.csv"
    return SOTA / f"doc-code/aalinker-composed/{slot}/{run}/{project}.csv"


def per_component(project, res, task):
    """Return (link_f1, per_component_F1_list over GOLD comps, sentence_coverage,
    n_components). For sad-sam component=model-element; for sad-code component=
    SAM-CODE component owning the file (D-12 interface drop, gold enrolled)."""
    if task == "sad-sam":
        gold = m.load_gs_sad_sam(project)            # (comp, sentence)
        link = m.prf(gold, res)[2]
        gb, rb = defaultdict(set), defaultdict(set)
        for c, s in gold: gb[c].add(s)
        for c, s in res:  rb[c].add(s)
        gold_by_s = defaultdict(set); res_by_s = defaultdict(set)
        for c, s in gold: gold_by_s[s].add(c)
        for c, s in res:  res_by_s[s].add(c)
    else:
        code_files = m.load_code_model_files(project)
        gold = m.enroll(m.load_gs_sad_code_raw(project), code_files)  # (sentence, file)
        f2c = m.load_file_to_comps(project, code_files)
        link = m.prf(gold, res)[2]
        def to_comp(pairs):
            out = set()
            for s, c in pairs:
                for comp in f2c.get(c, ()): out.add((s, comp))
            return out
        gold_c, res_c = to_comp(gold), to_comp(res)
        gb, rb = defaultdict(set), defaultdict(set)
        for s, c in gold_c: gb[c].add(s)
        for s, c in res_c:  rb[c].add(s)
        gold_by_s = defaultdict(set); res_by_s = defaultdict(set)
        for s, c in gold: gold_by_s[s].add(c)
        for s, c in res:  res_by_s[s].add(c)
    def cf1(c):
        g = {(x, c) for x in gb.get(c, set())}
        r = {(x, c) for x in rb.get(c, set())}
        return m.prf(g, r)[2]
    per = [cf1(c) for c in gb]
    sent_cov = m.sentence_coverage(gold_by_s, res_by_s)
    return link, per, sent_cov, len(per)


# ── metric family over the per-component F1 list P (+ sent_cov passed in) ────────
def gini(xs):
    if not xs or all(x == 0 for x in xs): return 0.0
    s = sorted(xs); n = len(s); cum = sum((i+1)*v for i, v in enumerate(s))
    return (2*cum)/(n*sum(s)) - (n+1)/n

def metrics_over(P, sent_cov):
    n = len(P); s = sorted(P)
    mean = sum(P)/n
    worst = s[0]
    second = s[1] if n > 1 else s[0]
    bottom2 = sum(s[:2])/min(2, n)
    k25 = max(1, math.ceil(0.25*n))
    cvar25 = sum(s[:k25])/k25                      # mean of worst quartile (CVaR)
    median = s[n//2] if n % 2 else (s[n//2-1]+s[n//2])/2
    harm = (n/sum(1/x for x in P)) if all(x > 0 for x in P) else 0.0
    geo = math.exp(sum(math.log(x) for x in P)/n) if all(x > 0 for x in P) else 0.0
    std = (sum((x-mean)**2 for x in P)/n)**0.5
    cv = std/mean if mean else 0.0
    comp_cov = sum(1 for x in P if x > 0)/n        # frac comps with >=1 correct
    strict_cov = sum(1 for x in P if x >= 0.999)/n # frac fully recovered
    frac_ge5 = sum(1 for x in P if x >= 0.5)/n
    n_missed = sum(1 for x in P if x == 0.0)
    return {
        "macro_mean_F1": mean, "median_F1": median,
        "worst_F1(min)": worst, "second_worst_F1": second, "bottom2_mean_F1": bottom2,
        "CVaR25_F1": cvar25, "harmonic_F1": harm, "geomean_F1": geo,
        "gini_F1(inv)": 1-gini(P),       # invert so higher=better, like others
        "neg_cv_F1": -cv,                 # higher=more even
        "comp_coverage": comp_cov, "strict_comp_cov": strict_cov,
        "frac_comp_ge.5": frac_ge5, "sent_coverage": sent_cov,
        "neg_n_missed": -float(n_missed),
    }

METRIC_KEYS = ["macro_mean_F1","median_F1","worst_F1(min)","second_worst_F1",
               "bottom2_mean_F1","CVaR25_F1","harmonic_F1","geomean_F1",
               "gini_F1(inv)","neg_cv_F1","comp_coverage","strict_comp_cov",
               "frac_comp_ge.5","sent_coverage","neg_n_missed"]


def build(task):
    cells = []        # each: {"sys","proj","link", **metrics}
    macro = defaultdict(lambda: defaultdict(list))
    for sysname, slot, runs in SYSTEMS:
        for p in PROJECTS:
            accs = defaultdict(list); links = []
            for run in runs:
                res = m.load_result(path_for(slot, task, run, p), task)
                if not res:
                    continue
                link, P, sc, _ = per_component(p, res, task)
                links.append(link)
                mv = metrics_over(P, sc)
                for k, v in mv.items(): accs[k].append(v)
            if not links:
                continue
            cell = {"sys": sysname, "proj": p, "link": sum(links)/len(links)}
            for k in METRIC_KEYS: cell[k] = sum(accs[k])/len(accs[k])
            cells.append(cell)
            macro[sysname]["link"].append(cell["link"])
            for k in METRIC_KEYS: macro[sysname][k].append(cell[k])
    return cells, macro


def report(task):
    cells, macro = build(task)
    link = [c["link"] for c in cells]
    rows = []
    for k in METRIC_KEYS:
        vals = [c[k] for c in cells]
        rho = spearman(link, vals)
        # does S21 GPT rank #1 on the macro of this metric?
        macavg = {s: sum(macro[s][k])/len(macro[s][k]) for s in macro}
        top = max(macavg, key=lambda s: macavg[s])
        rows.append((k, rho, top, macavg.get("S21 GPT")))
    rows.sort(key=lambda r: abs(r[1]))   # most independent first
    print(f"\n================ {task}  ({len(cells)} cells) ================")
    print(f"{'metric':<18}{'rho_vs_linkF1':>14}{'|rho|':>8}{'macro#1':>12}{'S21GPT_val':>12}")
    print("-"*64)
    for k, rho, top, s21 in rows:
        flag = "  <- approach #1" if top == "S21 GPT" else f"  ({top} #1)"
        print(f"{k:<18}{rho:>+14.3f}{abs(rho):>8.3f}{top:>12}{s21:>12.3f}{flag}")
    return cells, rows


def write_csv(task, cells, rows):
    out = HERE / "reports"
    with open(out/f"cells_{task}.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["sys","proj","link"]+METRIC_KEYS)
        for c in cells: w.writerow([c["sys"],c["proj"],f"{c['link']:.4f}"]+[f"{c[k]:.4f}" for k in METRIC_KEYS])
    with open(out/f"corr_{task}.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["metric","spearman_rho_vs_linkF1","abs_rho","macro_rank1","s21gpt_value"])
        for k, rho, top, s21 in rows: w.writerow([k,f"{rho:.4f}",f"{abs(rho):.4f}",top,f"{s21:.4f}"])


if __name__ == "__main__":
    for task in ("sad-sam", "sad-code"):
        cells, rows = report(task)
        write_csv(task, cells, rows)
    print(f"\n[explore] wrote reports/cells_*.csv, corr_*.csv under {HERE/'reports'}")
