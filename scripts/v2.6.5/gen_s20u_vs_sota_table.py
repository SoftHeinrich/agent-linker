#!/usr/bin/env python3
"""Cross-system comparison table: s20U vs SOTA baselines, both tasks.

Emits one paper table (``cross_system.tex``) comparing \\approach{}
(s_linker20_union, both backends, mean of N=3 runs) against the SOTA baselines
on the non-redundant metric panel, for both tasks:

  * SAD-SAM  (doc->model): link P/R/F1, sentence coverage, noise.
  * SAD-Code (doc->code) : file P/R/F1, component F1, sentence coverage, noise.

Baselines (from sota/recovered-links/, the same recovered-link dumps the paper
already uses):
  * Artemis (gpt-5.4)        — both tasks.
  * TransArC                 — SAD-Code; its SAD-SAM stage is SWATTR, shown in
                               the SAD-SAM block (marked with a dagger).

Fair footing: every system's SAD-Code is recovered SAD-SAM composed with the
*recovered* ArCoTL SAM->code. For \\approach{} that composition is done here
(s20U links x ArCoTL); for Artemis/TransArC it is already baked into the
recovered dumps. (Verified: SWATTR SAD-SAM x ArCoTL == canonical TransArC
doc-code, Jaccard 1.0 on all 5 projects.)

Stdlib-only; scores via transarc-emp/mini-src/metrics.py (the non-redundant
panel), which is __file__-relative so this runs from any cwd.

    python3 agent-linker/scripts/v2.6.5/gen_s20u_vs_sota_table.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

_TE = Path("/mnt/hostshare/ardoco-home/transarc-emp")
sys.path.insert(0, str(_TE / "mini-src"))
import metrics as mini   # noqa: E402

_AL = Path("/mnt/hostshare/ardoco-home/agent-linker")
_SOTA = Path("/mnt/hostshare/ardoco-home/sota/recovered-links")
PROJECTS = mini.PROJECTS
RUNS = (1, 2, 3)
VARIANT = "s_linker20_union"

_S20U_ROOT = {
    "sonnet": _AL / "results" / "v2.6.5_s20union_sonnet",
    "gpt": _AL / "results" / "v2.6.5_s20union" / "gpt",
}


def _arcotl_map(proj):
    m = defaultdict(set)
    with open(_TE / "results" / proj / "sam-code" / f"samCodeTlr_{proj}.csv") as f:
        for r in csv.DictReader(f):
            m[r["sentenceID"].strip()].add(mini.normalize_path(r["codeID"].strip()))
    return m


_AMAP = {p: _arcotl_map(p) for p in PROJECTS}


def _load_s20u_links(path):
    sadsam, pairs = set(), []
    with open(path) as f:
        for r in csv.DictReader(f):
            s, c = (r.get("sentence") or "").strip(), (r.get("component_id") or "").strip()
            if s and c:
                sadsam.add((c, s))
                pairs.append((s, c))
    return sadsam, pairs


def _macro(per_project_rows, cols):
    return {c: sum(r[c] for r in per_project_rows) / len(per_project_rows) for c in cols}


def _s20u_panel(backend, task):
    """Mean over N=3 runs, then macro over projects."""
    cols = mini.PANELS[task]
    proj_means = []
    for p in PROJECTS:
        runs = []
        for run in RUNS:
            fp = _S20U_ROOT[backend] / f"run{run}" / p / f"{VARIANT}_{p}_links.csv"
            if not fp.exists():
                continue
            sadsam, pairs = _load_s20u_links(fp)
            if task == "sad-sam":
                runs.append(mini.compute_sad_sam(p, sadsam))
            else:
                res = {(s, code) for (s, c) in pairs for code in _AMAP[p].get(c, ())}
                runs.append(mini.compute_sad_code(p, res))
        if runs:
            proj_means.append({c: sum(r[c] for r in runs) / len(runs) for c in cols})
    return _macro(proj_means, cols)


def _sota_panel(task, subdir, pattern):
    cols = mini.PANELS[task]
    compute = mini.compute_sad_sam if task == "sad-sam" else mini.compute_sad_code
    rows = []
    for p in PROJECTS:
        res = mini.load_result(mini.result_path(p, str(_SOTA / subdir), pattern), task)
        if res:
            rows.append(compute(p, res))
    return _macro(rows, cols)


# system -> {task: panel}. TransArC's SAD-SAM stage is SWATTR.
def _all_panels():
    return {
        "\\approach{} (Claude)": {
            "sad-sam": _s20u_panel("sonnet", "sad-sam"),
            "sad-code": _s20u_panel("sonnet", "sad-code"),
        },
        "\\approach{} (GPT-5.4)": {
            "sad-sam": _s20u_panel("gpt", "sad-sam"),
            "sad-code": _s20u_panel("gpt", "sad-code"),
        },
        "Artemis (GPT-5.4)": {
            "sad-sam": _sota_panel("sad-sam", "model-doc", "artemis-{project}-gpt-5.4.csv"),
            "sad-code": _sota_panel("sad-code", "doc-code", "artemis-{project}-gpt-5.4.csv"),
        },
        "TransArC$^{\\dagger}$": {
            "sad-sam": _sota_panel("sad-sam", "model-doc", "swattr-{project}.csv"),
            "sad-code": _sota_panel("sad-code", "doc-code", "transarc-{project}.csv"),
        },
    }


# Column order within each task block (the non-redundant panel).
_SS = ["link_p", "link_r", "link_f1", "sentence_coverage", "noise_rate"]
_SC = ["file_p", "file_r", "file_f1", "component_f1", "sentence_coverage", "noise_rate"]
_HEAD_SS = ["P", "R", "Link \\fone", "Cov.", "Noise\\,$\\downarrow$"]
_HEAD_SC = ["P", "R", "File \\fone", "Comp.\\ \\fone", "Cov.", "Noise\\,$\\downarrow$"]


def _fmt(v):
    return f"{v:.3f}"


def _render(panels):
    order = list(panels.keys())
    best_link = max(panels[s]["sad-sam"]["link_f1"] for s in order)
    best_file = max(panels[s]["sad-code"]["file_f1"] for s in order)

    def cell(val, is_best):
        s = _fmt(val)
        return "\\textbf{%s}" % s if is_best else s

    align = "l" + "r" * (len(_SS) + len(_SC))
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        "  \\caption{Cross-system comparison of \\approach{} (\\texttt{s\\_linker20\\_union}, "
        "mean of three runs) against the SOTA baselines on the non-redundant metric "
        "panel, for doc-to-model (SAD--SAM) and doc-to-code (SAD--Code). Every "
        "system's SAD--Code is recovered SAD--SAM composed with recovered ArCoTL "
        "SAM--code, so the comparison is on equal footing. Best \\fone{} per task in bold.}",
        r"  \label{tab:cross-system}",
        "  \\begin{tabular}{%s}" % align,
        r"    \toprule",
        "    & \\multicolumn{%d}{c}{SAD--SAM (doc$\\to$model)} & "
        "\\multicolumn{%d}{c}{SAD--Code (doc$\\to$code)} \\\\" % (len(_SS), len(_SC)),
        "    \\cmidrule(lr){2-%d} \\cmidrule(lr){%d-%d}"
        % (1 + len(_SS), 2 + len(_SS), 1 + len(_SS) + len(_SC)),
        "    System & " + " & ".join(_HEAD_SS + _HEAD_SC) + r" \\",
        r"    \midrule",
    ]
    for s in order:
        ss, sc = panels[s]["sad-sam"], panels[s]["sad-code"]
        cells = [s]
        for c in _SS:
            cells.append(cell(ss[c], c == "link_f1" and ss[c] == best_link))
        for c in _SC:
            cells.append(cell(sc[c], c == "file_f1" and sc[c] == best_file))
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        "  \\par\\smallskip\\footnotesize $^{\\dagger}$The SAD--SAM column for "
        "TransArC is SWATTR (its deterministic SAD--SAM stage); TransArC has no "
        "standalone doc-to-model system. \\approach{} SAD--Code composes the s20U "
        "links with recovered ArCoTL SAM--code (\\texttt{samCodeTlr}); for "
        "Artemis/TransArC that composition is already in the recovered dumps. "
        "Noise: lower is better.",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex-out-dir",
                    default="/mnt/hostshare/ardoco-home/alinker-paper/tables")
    ap.add_argument("--stdout", action="store_true", help="also print the table")
    args = ap.parse_args(argv)

    panels = _all_panels()
    tex = _render(panels)
    out = Path(args.tex_out_dir) / "cross_system.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(f"[cross-system] wrote {out}")
    if args.stdout:
        print(tex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
