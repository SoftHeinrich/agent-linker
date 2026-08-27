#!/usr/bin/env python3
"""Motivation & paper hooks (Phase 3 / MOTIV-01, OUT-02).

Shows that trivial baselines exploit the benchmark's distributional inequality:
a content-blind Top-3 (most-gold-linked) baseline scores a surprisingly high
file-/link-level micro-F1, while the size-aware suite (per-component macro F1
and the components it reaches) exposes it as content-blind. This motivates the
suite (MOTIV-01). Sentence coverage and noise rate were dropped from the paper's
suite on 2026-08-27 and are no longer computed here either -- this study must
describe the suite that is actually reported. It also emits the paper-ready component link-
concentration table (with Gini) + Lorenz figure source (OUT-02).

GOLD ONLY — no system/result files. Reuses the study's own engine
(`import inequality`) and copies the metric/baseline definitions verbatim from
`mini-src/metrics.py` and `src/bias/rq2_doc_to_model_prestudy.py` (isolation rule:
no imports from `src/`/`mini-src/`). Randomness is seeded (deterministic).

    python3 motivation.py     # write MOTIVATION.md, baselines.csv, OUT-02 source
"""

import random
import sys
from collections import defaultdict

import inequality as ineq

SEED = 0
REPORTS = ineq.REPORTS
P = list(ineq.PROJECTS)
TASKS = ["sad-code", "sad-sam"]


# ── Metric helpers (copied: mini-src/metrics.py prf; prestudy macro/component coverage) ──
def prf(gold, res):
    """Micro (precision, recall, f1) over link sets (mini-src/metrics.py)."""
    if not res:
        return 0.0, 0.0, 0.0
    tp = len(gold & res)
    precision = tp / len(res)
    recall = tp / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)
    return precision, recall, f1


def per_component_macro_f1(gold, result):
    """Macro F1 over targets (binary per target); rq2_doc_to_model_prestudy.py:227."""
    gold_by_c, res_by_c = defaultdict(set), defaultdict(set)
    for s, c in gold:
        gold_by_c[c].add(s)
    for s, c in result:
        res_by_c[c].add(s)
    comps = set(gold_by_c) | set(res_by_c)
    if not comps:
        return 0.0
    f1s = []
    for c in comps:
        g, r = gold_by_c.get(c, set()), res_by_c.get(c, set())
        tp, fp, fn = len(g & r), len(r - g), len(g - r)
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * rc / (p + rc) if (p + rc) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0



def component_coverage(gold, result):
    """Fraction of gold components with >=1 correct link -- the complement of the
    component miss rate (CMR) reported in the paper. The simple "components reached" proxy used in
    the paper's motivation; per-component macro F1 is its precision-aware refinement."""
    gold_by_c, res_by_c = defaultdict(set), defaultdict(set)
    for s, c in gold:
        gold_by_c[c].add(s)
    for s, c in result:
        res_by_c[c].add(s)
    comps = list(gold_by_c)
    if not comps:
        return 0.0
    return sum(1 for c in comps if gold_by_c[c] & res_by_c.get(c, set())) / len(comps)



# ── Baselines (copied: rq2_doc_to_model_prestudy.py:181,205) ──────────────────
def baseline_random(gold_sents, all_targets, target_size, rng):
    sent_list, tgt_list = sorted(gold_sents), sorted(all_targets)
    if not sent_list or not tgt_list:
        return set()
    result, attempts = set(), 0
    cap = max(target_size * 10, 10)
    while len(result) < target_size and attempts < cap:
        result.add((rng.choice(sent_list), rng.choice(tgt_list)))
        attempts += 1
    return result


def baseline_top3_by_gold_links(gold_sents, gold, k=3):
    link_count = defaultdict(int)
    for _s, c in gold:
        link_count[c] += 1
    top = [c for c, _ in sorted(link_count.items(), key=lambda x: x[1],
                                reverse=True)[:k]]
    return {(s, c) for s in gold_sents for c in top}


# ── Per-task gold + targets (reuse the engine) ────────────────────────────────
def gold_and_targets(project, task):
    """Returns (gold, targets, file_to_comps, comp_to_files).

    file_to_comps/comp_to_files are None for sad-sam."""
    if task == "sad-sam":
        raw = ineq.load_gs_sad_sam(project)            # (modelElementID, sentence)
        gold = {(s, c) for (c, s) in raw}              # flip to (sentence, comp)
        targets = sorted({c for (c, _s) in raw})
        return gold, targets, None, None
    code = ineq.load_code_model_files(project)
    gold = ineq.enroll(ineq.load_gs_sad_code_raw(project), code)  # (sentence, file)
    names, sam = ineq.load_sam_code(project, code)
    file_to_comps, comp_to_files = defaultdict(set), defaultdict(set)
    for ae, fp in sam:
        c = names.get(ae, ae)
        file_to_comps[fp].add(c)
        comp_to_files[c].add(fp)
    return gold, sorted(code), file_to_comps, comp_to_files


def top3_baseline(task, gold, gold_sents, file_to_comps, comp_to_files):
    """Inequality-exploiting Top-3 baseline.

    sad-sam: predict the 3 most-gold-linked components for every sentence.
    sad-code: predict ALL files under the 3 most-gold-linked components (the
    doc-to-code 'vote by enrolled file count' analogue — the big components own
    most of the gold mass, so this content-blind baseline scores a high file F1).
    """
    if task == "sad-sam":
        return baseline_top3_by_gold_links(gold_sents, gold, 3)
    comp_links = defaultdict(int)
    for s, f in gold:
        for c in file_to_comps.get(f, ()):
            comp_links[c] += 1
    top = [c for c, _ in sorted(comp_links.items(), key=lambda x: x[1],
                                reverse=True)[:3]]
    return {(s, f) for s in gold_sents for c in top
            for f in comp_to_files.get(c, ())}


def _collapse(pairs, file_to_comps):
    """(sentence, file) -> (sentence, component), mapped-only (drop unmapped)."""
    out = set()
    for s, f in pairs:
        for c in file_to_comps.get(f, ()):
            out.add((s, c))
    return out


def measure(name, result, gold, task, file_to_comps):
    # micro_f1 IS the file-level F1 (sad-code) / link-level F1 (sad-sam) — the
    # standard ruler. The suite adds per-component macro F1 and component coverage.
    micro = prf(gold, result)[2]
    if task == "sad-code":
        g_c = _collapse(gold, file_to_comps)
        r_c = _collapse(result, file_to_comps)
        comp_f1 = per_component_macro_f1(g_c, r_c)
        comp_cov = component_coverage(g_c, r_c)
    else:
        comp_f1 = per_component_macro_f1(gold, result)
        comp_cov = component_coverage(gold, result)
    return {
        "baseline": name, "micro_f1": micro, "comp_f1": comp_f1,
        "comp_cov": comp_cov,
    }


BASE_COLS = ["task", "project", "baseline", "micro_f1", "comp_f1",
             "comp_cov"]


def _fmt(v):
    return "NA" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def run_baselines():
    rows = []
    for task in TASKS:
        for project in P:
            gold, targets, fc, cf = gold_and_targets(project, task)
            gold_sents = {s for (s, _t) in gold}
            rng = random.Random(SEED)
            results = {
                "top3": top3_baseline(task, gold, gold_sents, fc, cf),
                "random": baseline_random(gold_sents, targets, len(gold), rng),
                "gold": set(gold),
            }
            for name, res in results.items():
                m = measure(name, res, gold, task, fc)
                m.update(task=task, project=project)
                rows.append(m)
    return rows


def _avg(rows, task, baseline, col):
    vals = [r[col] for r in rows
            if r["task"] == task and r["baseline"] == baseline
            and isinstance(r[col], float)]
    return sum(vals) / len(vals) if vals else None


def write_baselines_csv(rows):
    REPORTS.mkdir(parents=True, exist_ok=True)
    import csv
    with open(REPORTS / "baselines.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(BASE_COLS)
        for r in rows:
            w.writerow([_fmt(r[c]) for c in BASE_COLS])
        for task in TASKS:
            for bl in ("top3", "random", "gold"):
                w.writerow([task, "AVG", bl]
                           + [_fmt(_avg(rows, task, bl, c))
                              for c in ("micro_f1", "comp_f1", "comp_cov")])


DRIVER_MAP = [
    ("Enrollment inflation (1.0×→217.6×)",
     "file-level F1", "a few directory decisions dominate the score — report it but caveat it"),
    ("Component concentration (files-per-component Gini 0.400→0.694)",
     "per-component macro F1", "size-blind: small components count as much as the giants"),
    ("Long-tail per-sentence distribution (Gini 0.331→0.645)",
     "worst-component F1", "reports the single worst component, which any average hides"),
    ("Tail components carrying few documented sentences",
     "component miss rate (CMR)", "prices a component abandoned outright, which costs a link-level average almost nothing"),
]


def write_motivation(rows):
    L = ["# Motivation — Trivial Baselines Exploit the Inequality\n"]
    L.append("> Gold-only. A content-blind Top-3 (most-gold-linked) baseline scores "
             "a high file-/link-level micro-F1 *because* a few large components own "
             "most of the gold mass; the four-metric suite exposes it. No system "
             f"results are used; randomness is seeded ({SEED}).\n")
    for task in TASKS:
        t3 = _avg(rows, task, "top3", "micro_f1")
        rd = _avg(rows, task, "random", "micro_f1")
        ratio = (t3 / rd) if rd else float("inf")
        ruler = "file F1" if task == "sad-code" else "link F1"
        L.append(f"## {task} — Top-3 micro-F1 {t3:.3f} vs random {rd:.3f} "
                 f"({ratio:.1f}× random)\n")
        L.append(f"micro-F1 is the standard {ruler} ruler; the suite adds the next "
                 "two columns.\n")
        L.append("| Baseline | micro-F1 ("
                 + ruler + ") | per-comp macro F1 | comp-cov |")
        L.append("|----------|----------------|-------------------|----------|")
        for bl in ("top3", "random", "gold"):
            cells = [f"{_avg(rows, task, bl, c):.3f}" for c in
                     ("micro_f1", "comp_f1", "comp_cov")]
            L.append(f"| {bl} | " + " | ".join(cells) + " |")
        L.append("")
    L.append("**Reading:** Top-3 posts a respectable micro-F1 (≈2× random) but a "
             "far lower **per-component macro F1** (~0.19 vs a micro of ~0.35-0.38) "
             "— it nails the few popular components and scores ~0 on the long tail "
             "of small ones. The large micro−macro gap, not micro-F1 itself, is the "
             "tell: micro-F1 alone cannot separate this content-blind baseline from "
             "a real-but-weak linker; per-component F1 and component coverage can.\n")
    L.append("## What each metric carries (which are load-bearing here)\n")
    L.append("- **components covered** is the simple discriminator used in the "
             "paper's motivation: on sad-code it *flips the ranking* — random "
             "reaches more components than Top-3 (0.758 vs 0.398) — exposing the "
             "popularity baseline that micro-F1 rewards. **Per-component macro F1** "
             "flips too (0.243 vs 0.186); it is the precision-aware refinement "
             "reported in the metric suite.\n")
    L.append("- **micro-F1 = file/link F1** is the standard ruler being corrected "
             "— kept once (no separate redundant file-F1 column).\n")
    L.append("## Why each suite metric is needed (driver → metric)\n")
    L.append("| Inequality driver | Metric it motivates | What it catches |")
    L.append("|-------------------|---------------------|-----------------|")
    for drv, metric, why in DRIVER_MAP:
        L.append(f"| {drv} | **{metric}** | {why} |")
    L.append("")
    sc_t3_file = _avg(rows, "sad-code", "top3", "micro_f1")
    L.append("## Resolved placeholder (intro.tex:64)\n")
    L.append(f"- **Trivial-baseline file-level F1** = **{sc_t3_file:.3f}** "
             "(gold-only Top-3 popularity baseline, sad-code, avg over 5 projects). "
             "This is the trivial baseline that the standard \\fone lets look "
             "competitive.\n")
    L.append("- *Deferred → Phase 3+ (need published system scores):* "
             "strongest-published-pipeline file F1; \\approach file F1 + improvement pp.\n")
    (REPORTS / "MOTIVATION.md").write_text("\n".join(L) + "\n")


# ── OUT-02 paper-ready table + Lorenz figure ──────────────────────────────────
# Component grain: the prestudy unit is the architectural component the suite
# weights equally. The table reports enrolled DOC-TO-CODE links grouped by gold
# component (via the SAM-CODE model->code mapping) -- the distribution link-level
# F1 is actually dominated by. comp_n is the suite's component universe (D-12: the
# Component-typed model elements, interfaces dropped), so the table matches RQ2.
#
# The .tex output is PAPER-READY (project aliases + thousands separators baked in)
# so reports/out02_concentration.tex is copied VERBATIM into the paper's
# table/gold_concentration.tex; check_paper_table.py guards that they stay equal.
# Both paper-side artifacts are copied into alinker-paper/table/ and guarded byte-
# for-byte by check_paper_table.py: the .tex (aliases) and the machine-readable
# .csv companion (full project names).
OUT02_CSV_COLS = ["project", "sentences", "components_model", "components_gold",
                  "links", "median", "max", "gini", "top3_pct"]

# Compact aliases for the .tex; full names for the .csv companion.
DISPLAY_NAMES = {"mediastore": "MS", "teastore": "TS", "teammates": "TM",
                 "bigbluebutton": "BBB", "jabref": "JR"}
FULL_NAMES = {"mediastore": "MediaStore", "teastore": "TeaStore",
              "teammates": "Teammates", "bigbluebutton": "BigBlueButton",
              "jabref": "JabRef"}

# PCM repository (SAM) per project, matching the SAD-SAM gold's model year. The
# total component count here reproduces Table 3 of the benchmark paper
# (fuchs_establishing_2023: MS 14 / TS 11 / TM 8 / BBB 12 / JR 6).
SAM_REPOSITORY = {
    "mediastore":    "mediastore/model_2016/pcm/ms.repository",
    "teastore":      "teastore/model_2020/pcm/teastore.repository",
    "teammates":     "teammates/model_2021/pcm/teammates.repository",
    "bigbluebutton": "bigbluebutton/model_2021/pcm/bbb.repository",
    "jabref":        "jabref/model_2021/pcm/jabref.repository",
}


def _sentence_count(project):
    """# sentences in the architecture documentation (ARDoCo = one sentence/line)."""
    txt = sorted((ineq.BENCHMARK / project).glob(f"text_*/{project}.txt"))[0]
    return sum(1 for line in txt.read_text().splitlines() if line.strip())


def _model_component_count(project):
    """# components in the SAM (PCM repository): Basic + Composite components,
    interfaces excluded. Reproduces Table 3 of the benchmark paper."""
    text = (ineq.BENCHMARK / SAM_REPOSITORY[project]).read_text()
    return (text.count('xsi:type="repository:BasicComponent"')
            + text.count('xsi:type="repository:CompositeComponent"'))


def _out02_rows():
    rows = []
    for p in P:
        lc = ineq.compute_sadcode_link_conc(p)
        rows.append({
            "project": p, "sentences": _sentence_count(p),
            "comp_model": _model_component_count(p),
            "comp_n": lc["comp_n"],
            "links_total": lc["links_total"],
            "link_median": lc["link_median"], "link_max": lc["link_max"],
            "link_gini": lc["link_gini"], "link_top3_pct": lc["link_top3_pct"],
        })
    return rows


def write_out02_concentration():
    import csv
    rows = _out02_rows()

    def csv_num(v):
        # whole-number floats print as ints; a genuine .5 median keeps one decimal.
        if isinstance(v, float):
            return str(int(v)) if v.is_integer() else f"{v:.1f}"
        return v
    with open(REPORTS / "out02_concentration.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(OUT02_CSV_COLS)
        for r in rows:
            w.writerow([FULL_NAMES.get(r["project"], r["project"]),
                        r["sentences"], r["comp_model"], r["comp_n"],
                        r["links_total"], csv_num(r["link_median"]), r["link_max"],
                        f"{r['link_gini']:.3f}", f"{r['link_top3_pct']:.1f}"])

    def tex_sep(v):
        # integer-or-half value -> LaTeX with thousands separators, e.g.
        # 8097 -> "8{,}097", 152.5 -> "152.5", 3622 -> "3{,}622".
        if isinstance(v, float):
            ipart, frac = (int(v), "") if v.is_integer() else \
                (int(v), "." + str(v).split(".", 1)[1])
        else:
            ipart, frac = int(v), ""
        s = str(abs(ipart))
        grouped = ""
        while len(s) > 3:
            grouped = "{,}" + s[-3:] + grouped
            s = s[:-3]
        sign = "-" if ipart < 0 else ""
        return sign + s + grouped + frac

    L = [
        "% Dataset overview + gold-standard link concentration for sec:metric:prestudy",
        "% (also the dataset table referenced from eval.tex sec:dataset).",
        "% AUTO-GENERATED by evaluation/mini-inequality/motivation.py (OUT-02) from the",
        "% benchmark SAD text, the PCM (SAM) repository, and",
        "% inequality.compute_sadcode_link_conc. Columns: Sent.\\ = sentences in the SAD",
        "% (one per line); Comp.\\ = components in the architecture model (PCM repository,",
        "% reproduces benchmark-paper Table 3: MS 14/TS 11/TM 8/BBB 12/JR 6); K =",
        "% gold-reachable components the size-aware suite scores;",
        "% Links/Med/Max/Gini/Top-3\\% = enrolled doc-code link concentration over the",
        "% K components. (# code files per project is computed but not reported here.)",
        "% PAPER-READY (project aliases + thousands separators baked in): copy verbatim",
        "% into working/table/gold_concentration.tex. DO NOT hand-edit -- regenerate; the",
        "% two files are kept identical by mini-inequality/check_paper_table.py.",
        "% Project aliases (MS/TS/TM/BBB/JR) are defined in the running text.",
        "% Companion data (machine-readable): table/gold_concentration.csv",
        "\\begin{table}[t]", "\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}",
        "\\caption{The ardoco-benchmark statistics.}",
        "\\label{tab:gold_concentration}",
        "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
        "\\textbf{Project} & \\textbf{Sent.} & \\textbf{Comp.} & \\textbf{$K$} & "
        "\\textbf{Links} & \\textbf{Med} & \\textbf{Max} & "
        "\\textbf{Gini} & \\textbf{Top-3\\%} \\\\", "\\midrule",
    ]
    for r in rows:
        L.append(" & ".join([
            DISPLAY_NAMES.get(r["project"], r["project"]),
            tex_sep(r["sentences"]), tex_sep(r["comp_model"]), tex_sep(r["comp_n"]),
            tex_sep(r["links_total"]), tex_sep(r["link_median"]),
            tex_sep(r["link_max"]), f"{r['link_gini']:.3f}",
            f"{r['link_top3_pct']:.1f}",
        ]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (REPORTS / "out02_concentration.tex").write_text("\n".join(L) + "\n")


def write_out02_lorenz():
    L = [
        "% Auto-generated by motivation.py (OUT-02). Requires pgfplots.",
        "% Data: reports/lorenz_sad_code_sentence.csv "
        "(columns: project,cum_pop_pct,cum_mass_pct).",
        "\\begin{tikzpicture}",
        "\\begin{axis}[width=.7\\linewidth, xlabel={Cumulative share of sentences}, "
        "ylabel={Cumulative share of gold links}, xmin=0, xmax=1, ymin=0, ymax=1, "
        "legend pos=north west, legend cell align=left]",
        "\\addplot[dashed,gray,domain=0:1] {x};  % line of equality",
        "\\addlegendentry{equality}",
    ]
    for p in P:
        L.append(f"\\addplot table [x=cum_pop_pct, y=cum_mass_pct, col sep=comma, "
                 f"discard if not={{project}}{{{p}}}] "
                 f"{{reports/lorenz_sad_code_sentence.csv}};")
        L.append(f"\\addlegendentry{{{p}}}")
    L += ["\\end{axis}", "\\end{tikzpicture}"]
    (REPORTS / "out02_lorenz.tex").write_text("\n".join(L) + "\n"
        + "% Note: the per-project filter uses the pgfplotstable 'discard if not'\n"
        "% style; alternatively split the CSV per project. The data is emitted by\n"
        "% inequality.py (Phase 1).\n")


def main():
    # Only the OUT-02 table feeds the alinker-paper PDF, so that is all we emit.
    # The baselines (MOTIVATION.md, baselines.csv) and the Lorenz figure
    # (out02_lorenz.tex) are non-PDF; their output is silenced. The functions are
    # retained above and can be re-enabled here if those analyses are needed again.
    write_out02_concentration()
    print(f"[motivation] seed={SEED} reports={REPORTS} (OUT-02 table only)")


if __name__ == "__main__":
    main()
