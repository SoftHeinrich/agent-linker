#!/usr/bin/env python3
"""v2.6.5 RQ1 table generator for s_linker20_union (s20U), N=3 mean.

Regenerates the paper's two RQ1 tables — ``metrics_sad-sam.tex`` and
``metrics_sad-code.tex`` — from the s20U N=3 baseline runs, superseding the
old s_linker19 (v2.6.3) numbers.

  * sad-sam : scored directly from the s20U link CSVs (sentence, component_id).
  * sad-code: each s20U SAD-SAM link is composed with ArCoTL's **recovered**
    SAM->code links (transarc-emp/results/<project>/sam-code/samCodeTlr_*.csv),
    NOT the gold SAM->code — so doc-code is on the same footing as the
    TransArC / Artemis SOTA baselines (recovered SAD-SAM o recovered ArCoTL
    SAM->code). Verified elsewhere: SWATTR SAD-SAM ⨝ this map reproduces the
    canonical TransArC doc-code exactly (Jaccard 1.0, all 5 projects).
  * each cell = unweighted mean over the 3 independent s20U runs (run1/2/3).

Reuses metrics_api (the full 11-metric suite) and rq1_table's column sets /
label helpers verbatim — no metric math is reimplemented here (D-02). The
renderer is a local copy of rq1_table._render_wide_table with an s20U-correct
caption and source note.

Run from cwd = transarc-emp (metrics_api resolves the benchmark via a
cwd-relative ``../`` path):

    cd /mnt/hostshare/ardoco-home/transarc-emp
    python3 /mnt/hostshare/ardoco-home/agent-linker/scripts/v2.6.5/gen_s20u_rq1_tables.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# approach -> evaluation imports are allowed (D-02); the reverse is not.
_TE = Path("/mnt/hostshare/ardoco-home/transarc-emp")
sys.path.insert(0, str(_TE / "src" / "lib"))
sys.path.insert(0, str(_TE / "src" / "paper"))
import metrics_api                       # noqa: E402
import transarc_error_analysis as tea    # noqa: E402
import rq1_table                         # noqa: E402  (BACKENDS, BACKEND_LABEL, *_COLS, _metric_label, _project_display)

_AL = Path("/mnt/hostshare/ardoco-home/agent-linker")
PROJECTS = metrics_api.PROJECTS
RUNS = (1, 2, 3)
VARIANT = "s_linker20_union"

# rq1_table.BACKENDS == ("claude", "openai"); map each to its s20U N=3 run root.
S20U_ROOT = {
    "claude": _AL / "results" / "v2.6.5_s20union_sonnet",   # run{r}/<proj>/...
    "openai": _AL / "results" / "v2.6.5_s20union" / "gpt",
}


def _links_path(backend: str, run: int, proj: str) -> Path:
    return S20U_ROOT[backend] / f"run{run}" / proj / f"{VARIANT}_{proj}_links.csv"


def _arcotl_map(proj: str):
    """model_element_id -> {normalized_code_path} from recovered ArCoTL SAM->code."""
    m = defaultdict(set)
    for ae_id, code_path in tea.load_result_sam_code_standalone(proj):
        m[ae_id].add(code_path)
    return m


def _load_links(path: Path):
    """Return (sadsam_set[(component_id, sentence)], pairs[(sentence, component_id)])."""
    sadsam, pairs = set(), []
    with open(path) as f:
        for r in csv.DictReader(f):
            s = (r.get("sentence") or "").strip()
            c = (r.get("component_id") or "").strip()
            if s and c:
                sadsam.add((c, s))
                pairs.append((s, c))
    return sadsam, pairs


def _mean_metric(run_rows, col):
    """Mean of a metric across runs; NA if the metric is inapplicable (all NA)."""
    vals = [r[col] for r in run_rows]
    if all(isinstance(v, (int, float)) for v in vals):
        return sum(vals) / len(vals)
    return metrics_api.NA


def _backend_project_means(backend: str, task: str):
    """Return {project -> mean metric row over the N=3 runs}."""
    amap = {p: _arcotl_map(p) for p in PROJECTS}
    compute = (metrics_api.compute_sad_sam_metrics if task == "sad-sam"
               else metrics_api.compute_sad_code_metrics)
    out = {}
    for p in PROJECTS:
        run_rows = []
        for run in RUNS:
            fp = _links_path(backend, run, p)
            if not fp.exists():
                continue
            sadsam, pairs = _load_links(fp)
            if task == "sad-sam":
                res = sadsam
            else:
                res = {(s, code) for (s, c) in pairs for code in amap[p].get(c, ())}
            run_rows.append(compute(p, res))
        if not run_rows:
            print(f"WARNING: no s20U runs for {backend}/{p}", file=sys.stderr)
            continue
        row = {"project": p}
        for col in metrics_api.NUMERIC_COLS:
            row[col] = _mean_metric(run_rows, col)
        row["_n_runs"] = len(run_rows)
        out[p] = row
    return out


def _build_wide_rows(task: str):
    cols = rq1_table.SAD_SAM_COLS if task == "sad-sam" else rq1_table.SAD_CODE_COLS
    per_backend = {}
    n_runs_seen = set()
    for backend in rq1_table.BACKENDS:
        pm = _backend_project_means(backend, task)
        rows = [pm[p] for p in PROJECTS if p in pm]
        for r in rows:
            n_runs_seen.add(r["_n_runs"])
        macro = metrics_api.build_avg_row(rows)
        macro["project"] = "Macro"
        per_backend[backend] = {r["project"]: r for r in rows}
        per_backend[backend]["Macro"] = macro

    wide = []
    for proj in list(PROJECTS) + ["Macro"]:
        w = {"project": proj}
        for backend in rq1_table.BACKENDS:
            src = per_backend[backend].get(proj)
            for c in cols:
                key = f"{rq1_table.BACKEND_LABEL[backend]}.{c}"
                w[key] = src.get(c, metrics_api.NA) if src else metrics_api.NA
        wide.append(w)
    return wide, sorted(n_runs_seen)


# ── Renderer (local copy of rq1_table._render_wide_table; s20U caption + note) ──

_CAPTION = {
    "sad-sam": (
        "Doc-to-model link-level metrics for \\approach{} (\\texttt{s\\_linker20\\_union}) "
        "across two backends; each cell is the mean of three independent runs. "
        "Macro row is the unweighted mean across the five projects."
    ),
    "sad-code": (
        "Doc-to-code metrics for \\approach{} (\\texttt{s\\_linker20\\_union}): each "
        "SAD--SAM link is composed with ArCoTL's \\emph{recovered} SAM--code links "
        "(the same footing as the TransArC and Artemis baselines), across two "
        "backends; each cell is the mean of three independent runs. Macro row is the "
        "unweighted mean across the five projects."
    ),
}
_NOTE = {
    "sad-sam": (
        "Source: \\texttt{approach/results/v2.6.5\\_s20union*/run\\{1,2,3\\}/"
        "<project>/s\\_linker20\\_union\\_<project>\\_links.csv} (N=3 mean)."
    ),
    "sad-code": (
        "Source: s20U SAD--SAM links (N=3 mean) composed with recovered ArCoTL "
        "\\texttt{transarc-emp/results/<project>/sam-code/samCodeTlr\\_<project>.csv}."
    ),
}


def _render(task: str, wide_rows) -> str:
    cols = rq1_table.SAD_SAM_COLS if task == "sad-sam" else rq1_table.SAD_CODE_COLS
    K = len(cols)
    total = 1 + 2 * K
    align = "l" + "r" * (total - 1)
    label = "tab:metrics-sad-sam" if task == "sad-sam" else "tab:metrics-sad-code"
    top = (["Project"]
           + ["\\multicolumn{%d}{c}{%s}" % (K, rq1_table.BACKEND_LABEL[b])
              for b in rq1_table.BACKENDS])
    bottom = ["{}"] + [rq1_table._metric_label(c) for c in cols] * len(rq1_table.BACKENDS)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        "  \\caption{%s}" % _CAPTION[task],
        "  \\label{%s}" % label,
        "  \\begin{tabular}{%s}" % align,
        r"    \toprule",
        "    " + " & ".join(top) + r" \\",
        "    \\cmidrule(lr){2-%d} \\cmidrule(lr){%d-%d}" % (1 + K, 2 + K, 1 + 2 * K),
        "    " + " & ".join(bottom) + r" \\",
        r"    \midrule",
    ]
    for row in wide_rows:
        cells = [rq1_table._project_display(row["project"])]
        for backend in rq1_table.BACKENDS:
            for c in cols:
                v = row[f"{rq1_table.BACKEND_LABEL[backend]}.{c}"]
                cells.append(metrics_api.NA_TEX if (v == metrics_api.NA or v is None)
                             else metrics_api._fmt(v))
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        "  \\par\\smallskip\\footnotesize %s" % _NOTE[task],
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["sad-sam", "sad-code", "both"], default="both")
    ap.add_argument("--tex-out-dir",
                    default="/mnt/hostshare/ardoco-home/alinker-paper/tables")
    args = ap.parse_args(argv)

    out_dir = Path(args.tex_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = ["sad-sam", "sad-code"] if args.task == "both" else [args.task]
    for task in tasks:
        wide, n_runs = _build_wide_rows(task)
        out = out_dir / f"metrics_{task}.tex"
        out.write_text(_render(task, wide), encoding="utf-8")
        print(f"[s20u-rq1] task={task} backends={','.join(rq1_table.BACKENDS)} "
              f"runs/proj={n_runs} wrote={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
