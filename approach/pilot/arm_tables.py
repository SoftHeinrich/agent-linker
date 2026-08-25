"""The full offline table set for ONE arm's recorded five-project runs.

`pilot/score_runs.py` pairs two arms and permutation-tests them; `compare_f2.py` and
`finetune_e2e_table.py` read the per-run `ablation_*.json`. This reads the predicted-link
CSVs -- the same key as `score_runs.py`, `(project, sentence, component_id)` -- and emits
every table one arm can support without a control: per run, per project, and per source.

No LLM calls. Use it when an arm has runs and its control does not (the regex round's
`s_linker92` control set is absent from this checkout), so the arm's own levels are
reported and no cross-set delta is implied.

    ../.venv/bin/python pilot/arm_tables.py --arm s_linker92a \
        --model terra ../results/regex_e2e_terra_r*_20260822 \
        --model luna  ../results/regex_e2e_luna_r*_20260822
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import PROJECTS, load_gold                          # noqa: E402

GOLD = {p: {(p, s, c) for s, c in load_gold(p)} for p in PROJECTS}
SOURCES = ("full_name", "partial_name", "coreference")


def load_run(run: Path, variant: str):
    """(links, links_by_source) for one run, or None when a project CSV is missing."""
    links, by_source = set(), {s: set() for s in SOURCES}
    for project in PROJECTS:
        path = run / f"{variant}_{project}_links.csv"
        if not path.exists():
            return None
        with path.open() as handle:
            for row in csv.DictReader(handle):
                key = (project, int(row["sentence"]), row["component_id"])
                links.add(key)
                by_source.setdefault(row["source"], set()).add(key)
    return links, by_source


def prf(hit, got, gold):
    precision = hit / got if got else 0.0
    recall = hit / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = (5 * precision * recall / (4 * precision + recall)
          if 4 * precision + recall else 0.0)
    return 100 * precision, 100 * recall, 100 * f1, 100 * f2


def project_scores(links, project):
    got = {k for k in links if k[0] == project}
    hit = len(got & GOLD[project])
    return prf(hit, len(got), len(GOLD[project]))


def run_scores(links):
    """TP, FP, FN and the macro (mean over projects) quality statistics."""
    gold_all = set().union(*GOLD.values())
    per_project = [project_scores(links, p) for p in PROJECTS]
    return {"TP": len(links & gold_all), "FP": len(links - gold_all),
            "FN": len(gold_all - links),
            "P": st.mean(s[0] for s in per_project),
            "R": st.mean(s[1] for s in per_project),
            "F1": st.mean(s[2] for s in per_project),
            "F2": st.mean(s[3] for s in per_project)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", nargs="+", action="append", required=True,
                        metavar=("NAME", "RUN"))
    args = parser.parse_args()

    models = {}
    for spec in args.model:
        name, runs = spec[0], []
        for path in spec[1:]:
            loaded = load_run(Path(path), args.arm)
            if loaded is None:
                print(f"<!-- {Path(path).name}: incomplete, skipped -->")
                continue
            runs.append((Path(path).name, *loaded))
        if runs:
            models[name] = runs

    gold_total = sum(len(g) for g in GOLD.values())
    print(f"# `{args.arm}` — recorded end-to-end runs\n")
    print(f"{gold_total} gold links over {len(PROJECTS)} projects. "
          f"Scored from the per-variant link CSVs, no LLM calls. "
          f"Macro = mean over the five projects.\n")

    print("## Every run\n")
    print("| model | run | TP | FP | FN | macro P | macro R | macro F1 | macro F2 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for model, runs in models.items():
        scored = [run_scores(links) for _n, links, _s in runs]
        for (name, _l, _s), s in zip(runs, scored):
            tag = name.split("_r")[-1].split("_")[0]
            print(f"| {model} | r{tag} | {s['TP']} | {s['FP']} | {s['FN']} | "
                  f"{s['P']:.2f} | {s['R']:.2f} | {s['F1']:.2f} | {s['F2']:.2f} |")
        mean = {k: st.mean(s[k] for s in scored) for k in scored[0]}
        f1s = [s["F1"] for s in scored]
        f2s = [s["F2"] for s in scored]
        print(f"| {model} | **mean (n={len(scored)})** | {mean['TP']:.1f} | "
              f"{mean['FP']:.1f} | {mean['FN']:.1f} | {mean['P']:.2f} | "
              f"{mean['R']:.2f} | **{mean['F1']:.2f}** | **{mean['F2']:.2f}** |")
        print(f"| {model} | run range | | | | | | {max(f1s) - min(f1s):.2f} | "
              f"{max(f2s) - min(f2s):.2f} |")

    print("\n## Per project, mean of the runs\n")
    print("| project | gold | " + " | ".join(
        f"{m} P | {m} R | {m} F1 | {m} F2" for m in models) + " |")
    print("|---" * (2 + 4 * len(models)) + "|")
    for project in PROJECTS:
        cells = []
        for runs in models.values():
            stats = [project_scores(links, project) for _n, links, _s in runs]
            cells += [f"{st.mean(s[i] for s in stats):.2f}" for i in range(4)]
        print(f"| {project} | {len(GOLD[project])} | " + " | ".join(cells) + " |")

    print("\n## Per source, mean of the runs\n")
    gold_all = set().union(*GOLD.values())
    print("| source | " + " | ".join(f"{m} TP | {m} FP" for m in models) + " |")
    print("|---" * (1 + 2 * len(models)) + "|")
    for source in SOURCES:
        cells = []
        for runs in models.values():
            tp = [len(by[source] & gold_all) for _n, _l, by in runs]
            fp = [len(by[source] - gold_all) for _n, _l, by in runs]
            cells += [f"{st.mean(tp):.1f}", f"{st.mean(fp):.1f}"]
        print(f"| `{source}` | " + " | ".join(cells) + " |")
    cells = []
    for runs in models.values():
        tp = [len(links & gold_all) for _n, links, _s in runs]
        fp = [len(links - gold_all) for _n, links, _s in runs]
        cells += [f"**{st.mean(tp):.1f}**", f"**{st.mean(fp):.1f}**"]
    print("| **total** | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
