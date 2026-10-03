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
(`import inequality`) and the tree's shared P/R/F1 (`metrics.prf`); the baseline
definitions are the only thing still copied here, from the retired
`src/bias/rq2_doc_to_model_prestudy.py`. Randomness is seeded (deterministic).

    python3 motivation.py     # write MOTIVATION.md, baselines.csv, OUT-02 source
"""

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mini-src"))
import inequality as ineq     # noqa: E402  (this study's gold engine)
from metrics import prf       # noqa: E402  (the tree's one P/R/F1 over link sets)

SEED = 0
REPORTS = ineq.REPORTS
P = list(ineq.PROJECTS)
TASKS = ["sad-code", "sad-sam"]
TABLE_SIZE = "\\footnotesize"


# ── Metric helpers (prf imported above; prestudy macro/component coverage) ────
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
    # One F1 per target, through the shared `prf`; a target with neither a gold
    # nor a predicted sentence is not scored at all (it is in neither universe).
    f1s = [prf(gold_by_c.get(c, set()), res_by_c.get(c, set()))[2]
           for c in comps
           if gold_by_c.get(c) or res_by_c.get(c)]
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
             "most of the gold mass; the size-aware suite exposes it. No system "
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
# The displayed component count is the full PCM repository count, as in the
# benchmark overview used by ArTEMiS. Each task's link distribution still uses
# its own gold-reachable component universe: doc-model uses scorer IDs; doc-code
# uses the SAM-CODE mapping after the interface exclusion.
#
# The .tex output is PAPER-READY (project names, abbreviations, and separators).
# sync_paper.py copies both generated artifacts into the paper and checks them
# byte-for-byte with --check.
OUT02_METRIC_NAMES = ("links", "median", "max", "gini", "top3_pct")
OUT02_TASKS = ("doc_model", "doc_code")
OUT02_CSV_COLS = ["project", "sentences", "lines_of_code_thousands", "components"] + [
    f"{task}_{metric}" for task in OUT02_TASKS for metric in OUT02_METRIC_NAMES
]

FULL_NAMES = {"mediastore": "MediaStore", "teastore": "TeaStore",
              "teammates": "Teammates", "bigbluebutton": "BigBlueButton",
              "jabref": "JabRef"}
PROJECT_ABBR = {"mediastore": "MS", "teastore": "TS", "teammates": "TM",
                "bigbluebutton": "BBB", "jabref": "JR"}

# Canonical PCM repositories used for the model input to this evaluation.
# Teammates also ships a separate details model; this table uses the base model.
PCM_REPOSITORIES = {
    "mediastore": "mediastore/model_2016/pcm/ms.repository",
    "teastore": "teastore/model_2020/pcm/teastore.repository",
    "teammates": "teammates/model_2021/pcm/teammates.repository",
    "bigbluebutton": "bigbluebutton/model_2021/pcm/bbb.repository",
    "jabref": "jabref/model_2021/pcm/jabref.repository",
}


def _component_count(project):
    """Count all component elements in the benchmark's base PCM repository."""
    path = ineq.BENCHMARK / PCM_REPOSITORIES[project]
    root = ElementTree.parse(path).getroot()
    ids = [element.get("id") for element in root.iter()
           if element.tag.rsplit("}", 1)[-1] == "components__Repository"]
    if not ids or None in ids or len(ids) != len(set(ids)):
        raise ValueError(f"invalid component IDs in {path}")
    return len(ids)


# Language selection used by the existing table, matching the primary-language
# columns reported in Table 1 of the benchmark paper. The code counts themselves
# are read from each checked-in benchmark README's cloc table.
PRIMARY_LANGUAGES = {
    "mediastore": ("Java",),
    "teastore": ("Java",),
    "teammates": ("Java", "TypeScript"),
    "bigbluebutton": ("Java", "JavaScript", "JSX", "Scala"),
    "jabref": ("Java",),
}


def _lines_of_code_thousands(project):
    """Sum individually rounded cloc code counts for the selected languages."""
    selected = set(PRIMARY_LANGUAGES[project])
    counts = {}
    path = ineq.BENCHMARK / project / "README.md"
    for line in path.read_text().splitlines():
        parts = line.rsplit(maxsplit=4)
        if len(parts) != 5 or parts[0] not in selected:
            continue
        if parts[0] in counts:
            raise ValueError(f"duplicate cloc language {parts[0]} in {path}")
        counts[parts[0]] = int(parts[-1])
    if counts.keys() != selected:
        raise ValueError(f"missing cloc languages {selected - counts.keys()} in {path}")
    return sum(round(counts[language] / 1000) for language in selected)


def _sentence_count(project):
    """# sentences in the architecture documentation (ARDoCo = one sentence/line)."""
    txt = sorted((ineq.BENCHMARK / project).glob(f"text_*/{project}.txt"))[0]
    return sum(1 for line in txt.read_text().splitlines() if line.strip())


def _out02_rows():
    rows = []
    for p in P:
        rows.append({
            "project": p,
            "sentences": _sentence_count(p),
            "loc_thousands": _lines_of_code_thousands(p),
            "components": _component_count(p),
            "doc_model": ineq.compute_sadsam_link_conc(p),
            "doc_code": ineq.compute_sadcode_link_conc(p),
        })
    return rows


def write_out02_concentration():
    rows = _out02_rows()

    def csv_num(v):
        # whole-number floats print as ints; a genuine .5 median keeps one decimal.
        if isinstance(v, float):
            return str(int(v)) if v.is_integer() else f"{v:.1f}"
        return v
    with open(REPORTS / "out02_concentration.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(OUT02_CSV_COLS)
        for r in rows:
            cells = [FULL_NAMES.get(r["project"], r["project"]),
                     r["sentences"], r["loc_thousands"], r["components"]]
            for task in OUT02_TASKS:
                lc = r[task]
                cells.extend([lc["links_total"], csv_num(lc["link_median"]),
                              lc["link_max"],
                              f"{lc['link_gini']:.3f}", f"{lc['link_top3_pct']:.1f}"])
            w.writerow(cells)

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
        "% GENERATED by evaluation/mini-inequality/motivation.py (OUT-02).",
        "% Do not edit by hand: regenerate with motivation.py and sync_paper.py.",
        "% Dataset overview + gold-standard link concentration for sec:metric:prestudy",
        "% (also the dataset table referenced from eval.tex sec:dataset).",
        "% Generated from the benchmark SAD text, the PCM (SAM) repository, and",
        "% inequality.compute_sadsam_link_conc and compute_sadcode_link_conc.",
        "% Components counts all base-PCM components. Link distributions use",
        "% task-specific gold-reachable component subsets. Doc-code links are",
        "% enrolled file links; shared files count under each mapped",
        "% component for median, maximum, Gini, and Top-3 share. Lines of code",
        "% come from the checked-in benchmark cloc reports, summing individually",
        "% rounded primary-language values as in Table 1 of the original benchmark",
        "% paper (Fuchß et al., ECSA 2022).",
        "% PAPER-READY (project names and abbreviations + thousands separators baked in):",
        "% mini-src/sync_paper.py copies this generated file into the paper and",
        "% --check verifies that the two files are byte-identical.",
        "% Companion data (machine-readable): table/gold_concentration.csv",
        "\\begin{table*}[t]", f"\\centering{TABLE_SIZE}\\setlength{{\\tabcolsep}}{{2pt}}",
        "\\caption{Gold link concentration by task. Sent. counts document sentences; kLOC sums the selected primary-language cloc counts in thousands. Comp. counts all components. Each Links column counts distinct gold pairs; Med., Max., Gini, and Top-3 share (\\%) use that task's gold-reachable components. Shared doc-code files contribute to each mapped component; unmapped files contribute only to Links.}",
        "\\label{tab:gold_concentration}",
        "\\adjustbox{max width=\\textwidth}{%",
        "\\begin{tabular}{lrrr@{\\hspace{4pt}}rrrrr@{\\hspace{8pt}}rrrrr}",
        "\\toprule",
        "\\textbf{Project} & \\textbf{Sent.} & \\textbf{kLOC} & "
        "\\textbf{Comp.} & \\multicolumn{5}{c}{\\textbf{Doc-model}} & "
        "\\multicolumn{5}{c}{\\textbf{Doc-code}} \\\\",
        "\\cmidrule(lr){5-9}\\cmidrule(lr){10-14}",
        "& & & & \\textbf{Links} & \\textbf{Med.} & "
        "\\textbf{Max.} & \\textbf{Gini} & \\textbf{Top-3} & "
        "\\textbf{Links} & \\textbf{Med.} & "
        "\\textbf{Max.} & \\textbf{Gini} & \\textbf{Top-3} \\\\",
        "\\midrule",
    ]
    for r in rows:
        cells = [
            f'{FULL_NAMES[r["project"]]} ({PROJECT_ABBR[r["project"]]})',
            tex_sep(r["sentences"]),
            tex_sep(r["loc_thousands"]),
            tex_sep(r["components"]),
        ]
        for task in OUT02_TASKS:
            lc = r[task]
            cells.extend([tex_sep(lc["links_total"]), tex_sep(lc["link_median"]),
                          tex_sep(lc["link_max"]),
                          f"{lc['link_gini']:.2f}".removeprefix("0"),
                          f"{lc['link_top3_pct']:.1f}"])
        L.append(" & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"]
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
