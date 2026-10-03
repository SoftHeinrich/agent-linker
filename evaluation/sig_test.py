"""Significance tests for RQ1: ArchLinker vs each baseline, per project and task.

Produces:
  - Console summary
  - evaluation/sig_test_table.tex  (appendix table)

ArchLinker and Artemis are stochastic (3 runs each) -> Mann-Whitney U test.
SWATTR / TransArc are deterministic (1 run) -> one-sample Wilcoxon signed-rank
test of the 3 ArchLinker runs against the fixed baseline value.
"""
import csv
import os
from collections import defaultdict
from pathlib import Path
from scipy import stats
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "paper" / "appendix"

def load_runs(csv_path, system_prefix):
    """Return {(project, task): [f1_run1, f1_run2, f1_run3]}."""
    out = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if not row["system"].startswith(system_prefix):
                continue
            proj = row["project"]
            dm_f1 = float(row["doc_to_model_link_f1"])
            dc_f1 = float(row["doc_to_code_file_f1"])
            out[(proj, "DM")].append(dm_f1)
            out[(proj, "DC")].append(dc_f1)
    return dict(out)

approach_runs = load_runs(ROOT / "big-table-approach.csv", "approach (GPT-5.6-terra)")
artemis_runs  = load_runs(ROOT / "big-table-artemis.csv",  "Artemis (GPT-5.6-terra)")

# Deterministic baselines from rq1-results.csv
rq1_csv = ROOT.parent / "table" / "rq1-results.csv"
swattr_f1 = {}
transarc_f1 = {}
with open(rq1_csv) as f:
    for row in csv.DictReader(f):
        proj = row["project"]
        if proj == "Average":
            continue
        task = row["task"]
        pf1 = float(row["pipeline_f1"])
        if task == "DM":
            swattr_f1[(proj, "DM")] = pf1
        else:
            transarc_f1[(proj, "DC")] = pf1

PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
PROJ_ABBR = {"mediastore": "MS", "teastore": "TS", "teammates": "TM",
             "bigbluebutton": "BBB", "jabref": "JR"}

results = []

def test_3v3(a_runs, b_runs):
    a = np.array(a_runs)
    b = np.array(b_runs)
    delta = np.mean(a) - np.mean(b)
    if np.allclose(a, b):
        return delta, float("nan"), "tie"
    try:
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        p = float("nan")
    sig = "yes" if p < 0.05 else "no"
    return delta, p, sig

def test_3v1(a_runs, b_val):
    a = np.array(a_runs)
    delta = np.mean(a) - b_val
    diffs = a - b_val
    if np.allclose(diffs, 0):
        return delta, float("nan"), "tie"
    try:
        _, p = stats.wilcoxon(diffs, alternative="two-sided")
    except ValueError:
        p = float("nan")
    sig = "yes" if p < 0.05 else "no"
    return delta, p, sig

print(f"{'Comparison':<45} {'Delta':>7} {'p':>8} {'Sig?':>5}")
print("-" * 70)

for proj in PROJECTS:
    a_dm = approach_runs[(proj, "DM")]
    a_dc = approach_runs[(proj, "DC")]
    b_dm = artemis_runs[(proj, "DM")]
    b_dc = artemis_runs[(proj, "DC")]

    delta, p, sig = test_3v3(a_dm, b_dm)
    results.append((proj, "doc-model", "Artemis", delta, p, sig))
    print(f"{PROJ_ABBR[proj]+' DM: AL vs Artemis':<45} {delta:+.4f} {p:8.4f} {sig:>5}")

    delta, p, sig = test_3v1(a_dm, swattr_f1[(proj, "DM")])
    results.append((proj, "doc-model", "SWATTR", delta, p, sig))
    print(f"{PROJ_ABBR[proj]+' DM: AL vs SWATTR':<45} {delta:+.4f} {p:8.4f} {sig:>5}")

    delta, p, sig = test_3v3(a_dc, b_dc)
    results.append((proj, "doc-code", "Artemis", delta, p, sig))
    print(f"{PROJ_ABBR[proj]+' DC: AL vs Artemis':<45} {delta:+.4f} {p:8.4f} {sig:>5}")

    delta, p, sig = test_3v1(a_dc, transarc_f1[(proj, "DC")])
    results.append((proj, "doc-code", "TransArc", delta, p, sig))
    print(f"{PROJ_ABBR[proj]+' DC: AL vs TransArc':<45} {delta:+.4f} {p:8.4f} {sig:>5}")

# Macro-level: paired tests over 5 project-level means
print("\n" + "=" * 70)
print("Macro-level (paired over 5 project means)")
print("-" * 70)

for task, bl_label, bl_src in [
    ("DM", "Artemis", artemis_runs),
    ("DM", "SWATTR", swattr_f1),
    ("DC", "Artemis", artemis_runs),
    ("DC", "TransArc", transarc_f1),
]:
    a_means = [np.mean(approach_runs[(p, task)]) for p in PROJECTS]
    if bl_label in ("Artemis",):
        b_means = [np.mean(bl_src[(p, task)]) for p in PROJECTS]
    else:
        b_means = [bl_src[(p, task)] for p in PROJECTS]

    a_arr = np.array(a_means)
    b_arr = np.array(b_means)
    n_pos = int(np.sum(a_arr > b_arr))
    sign_p = stats.binomtest(n_pos, len(a_arr), 0.5, alternative="greater").pvalue
    try:
        _, wilc_p = stats.wilcoxon(a_arr, b_arr, alternative="greater")
    except ValueError:
        wilc_p = float("nan")
    task_name = "doc-model" if task == "DM" else "doc-code"
    results.append(("Avg", task_name, bl_label, np.mean(a_arr - b_arr), wilc_p,
                     "yes" if wilc_p < 0.05 else "no"))
    print(f"{task_name} AL vs {bl_label:<12}  "
          f"delta={np.mean(a_arr-b_arr):+.4f}  "
          f"pos={n_pos}/5  "
          f"sign p={sign_p:.4f}  "
          f"Wilcoxon p={wilc_p:.4f}")

# Generate LaTeX table
tex_path = Path(__file__).resolve().parent / "sig_test_table.tex"
with open(tex_path, "w") as f:
    f.write("% GENERATED by evaluation/sig_test.py. Do not edit by hand.\n")
    f.write("\\begin{table}[t]\n")
    f.write("\\caption{Significance tests on \\fone. "
            "\\approach{} vs \\Artemis{}: two-sided Mann--Whitney $U$ ($n=3$ vs $n=3$). "
            "\\approach{} vs SWATTR/\\TransArc{}: two-sided Wilcoxon signed-rank "
            "of 3~runs against the deterministic value. "
            "Avg row: one-sided Wilcoxon signed-rank over 5 project means.}\n")
    f.write("\\label{tab:sigtest}\n")
    f.write("\\centering\\footnotesize\n")
    f.write("\\begin{tabular}{@{}llllrl@{}}\n")
    f.write("\\toprule\n")
    f.write("Project & Task & Baseline & $\\Delta$ \\fone & $p$ & Sig.\\\\\n")
    f.write("\\midrule\n")

    prev_proj = None
    for proj, task, bl, delta, p, sig in results:
        proj_label = PROJ_ABBR.get(proj, proj)
        if prev_proj and proj != prev_proj:
            f.write("\\addlinespace[1.5pt]\n")
        prev_proj = proj
        p_str = f"{p:.3f}" if not np.isnan(p) else "---"
        if sig == "yes":
            sig_str = "$\\checkmark$"
        elif sig == "tie":
            sig_str = "---"
        else:
            sig_str = ""
        f.write(f"{proj_label} & {task} & {bl} "
                f"& ${delta:+.2f}$ & ${p_str}$ & {sig_str} \\\\\n")

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"\nLaTeX table written to {tex_path}")
