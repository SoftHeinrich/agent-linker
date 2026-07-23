#!/usr/bin/env python3
"""Self-contained, stdlib-only GOLD trace-link concentration-inequality engine.

Measures ONLY the intrinsic inequality of the ARDoCo benchmark gold standard —
how a few large components / files own most of the gold link mass — via Gini,
Lorenz curves, top-k concentration share, min/median/max, the enrollment
expansion factor, and the structural component->file amplification (files per
component). It reproduces the Chapter-1 inequality tables of writing/eval.tex and
self-checks every reproduced number against the frozen literals, failing loud on
any mismatch.

GOLD ONLY: this engine reads benchmark gold standards + .acm code models. It reads
NO results/ files and contains NO TransArc-/system-specific logic. INEQ-03's
"component->file cascade" is re-pivoted to the GOLD structural amplification driver
(SAM-CODE files-per-component fan-out); the TransArc actual-error attribution
eval.tex tab:amplification (36->3,457) is a system-specific quantity excluded from
this dataset study (user directive 2026-06-21).

Definitions are COPIED verbatim from mini-src/metrics.py (enroll, normalize_path,
gold loaders, path maps) and src/bias/component_suite.py (_gini) — never imported
— and sanity-checked for agreement (OUT-01 isolation rule).

Usage:
    python3 inequality.py                 # all gold inequality + sanity CHECK
    python3 inequality.py --task sad-code
    python3 inequality.py --project jabref
    python3 inequality.py --check-only    # run only the sanity gate
    python3 inequality.py --no-check      # skip the gate

Roots derive from this file's location (study-root/inequality.py -> ardoco-home);
override the benchmark root with $TRANSARC_BENCHMARK.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# -- Benchmark layout (mirrors mini-src/metrics.py) ----------------------------
# study-root/inequality.py -> parents[2] is <ardoco-home>.
_ARDOCO_HOME = Path(__file__).resolve().parents[2]
BENCHMARK = Path(os.environ.get(
    "TRANSARC_BENCHMARK",
    _ARDOCO_HOME / "ardoco/core/tests-base/src/main/resources/benchmark",
))
REPORTS = Path(__file__).resolve().parent / "reports"
# NOTE: no RESULTS constant — this engine reads no system results.

PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]

GS_SAD_SAM = {
    "mediastore":    "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    "teastore":      "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    "teammates":     "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "bigbluebutton": "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "jabref":        "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
}
GS_SAM_CODE = {
    "mediastore":    "mediastore/goldstandards/goldstandard_sam_2016-code_2016.csv",
    "teastore":      "teastore/goldstandards/goldstandard_sam_2020-code_2022.csv",
    "teammates":     "teammates/goldstandards/goldstandard_sam_2021-code_2023.csv",
    "bigbluebutton": "bigbluebutton/goldstandards/goldstandard_sam_2021-code_2023.csv",
    "jabref":        "jabref/goldstandards/goldstandard_sam_2021-code_2023.csv",
}
GS_SAD_CODE = {
    "mediastore":    "mediastore/goldstandards/goldstandard_sad_2016-code_2016.csv",
    "teastore":      "teastore/goldstandards/goldstandard_sad_2020-code_2022.csv",
    "teammates":     "teammates/goldstandards/goldstandard_sad_2021-code_2023.csv",
    "bigbluebutton": "bigbluebutton/goldstandards/goldstandard_sad_2021-code_2023.csv",
    "jabref":        "jabref/goldstandards/goldstandard_sad_2021-code_2023.csv",
}
ACM_FILES = {
    "mediastore":    "mediastore/model_2016/code/codeModel.acm",
    "teastore":      "teastore/model_2022/code/codeModel.acm",
    "teammates":     "teammates/model_2023/code/codeModel.acm",
    "bigbluebutton": "bigbluebutton/model_2023/code/codeModel.acm",
    "jabref":        "jabref/model_2023/code/codeModel.acm",
}

# ── Copied primitives (verbatim — do NOT import) ──────────────────────────────


def normalize_path(path):
    """Drop the leading 'Implementation/' segment used in the gold standard."""
    prefix = "Implementation/"
    return path[len(prefix):] if path.startswith(prefix) else path


def enroll(gold, code_files):
    """Expand directory-level gold entries (trailing '/') to individual files."""
    enrolled = set()
    for gid, gpath in gold:
        if gpath.endswith("/"):
            for fp in code_files:
                if fp.startswith(gpath):
                    enrolled.add((gid, fp))
        else:
            enrolled.add((gid, gpath))
    return enrolled


def _cell(row, keys):
    """First non-empty value among `keys` in a DictReader row, else None."""
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def _gini(values):
    """Gini coefficient (copied verbatim from src/bias/component_suite.py)."""
    xs = sorted(values)
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * sum(xs)) - (n + 1) / n


# ── Gold loaders (benchmark only) ─────────────────────────────────────────────


def load_code_model_files(project):
    """All compilation-unit paths from the .acm code model (normalized)."""
    files = set()
    with open(BENCHMARK / ACM_FILES[project]) as f:
        data = json.load(f)
    repo = data.get("codeItemRepository", {}).get("repository", {})
    for item in repo.values():
        if item.get("type") != "CodeCompilationUnit":
            continue
        parts, name, ext = (item.get("pathElements", []),
                            item.get("name", ""), item.get("extension", ""))
        if parts and name:
            full = "/".join(parts) + "/" + name + (f".{ext}" if ext else "")
            files.add(normalize_path(full))
    return files


def load_gs_sad_sam(project):
    """set[(modelElementID, sentence)]."""
    with open(BENCHMARK / GS_SAD_SAM[project]) as f:
        return {(r["modelElementID"], r["sentence"]) for r in csv.DictReader(f)}


def load_gs_sad_code_raw(project):
    """set[(sentenceID, normalized_path)] — pre-enrolment."""
    with open(BENCHMARK / GS_SAD_CODE[project]) as f:
        return {(r["sentenceID"], normalize_path(r["codeID"]))
                for r in csv.DictReader(f)}


def load_sam_code(project, code_files):
    """(names: ae_id->ae_name, sam_enrolled: set[(ae_id, file)]).

    Mirrors mini-src/metrics.py load_file_to_comps: read the SAM-CODE gold, then
    enroll directory entries against the code model.
    """
    names, raw = {}, set()
    with open(BENCHMARK / GS_SAM_CODE[project]) as f:
        for r in csv.DictReader(f):
            names[r["ae_id"]] = r["ae_name"]
            raw.add((r["ae_id"], normalize_path(r.get("ce_ids") or r.get("ce_id"))))
    sam_enrolled = enroll(raw, code_files)
    return names, sam_enrolled


# ── Generic inequality helpers ────────────────────────────────────────────────


def summary_stats(values):
    """dict(n, total, min, median, max) over a list of numbers."""
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return {"n": 0, "total": 0, "min": 0, "median": 0, "max": 0}
    mid = n // 2
    median = vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2
    return {"n": n, "total": sum(vals), "min": vals[0],
            "median": median, "max": vals[-1]}


def top_k_share(values, k):
    """Share of total mass held by the k largest values (0.0 if total==0)."""
    total = sum(values)
    if total == 0:
        return 0.0
    return sum(sorted(values, reverse=True)[:k]) / total


def lorenz_points(values):
    """Lorenz curve as [(cum_pop_pct, cum_mass_pct)] over ascending values.

    Starts at (0,0); ends at (1,1) for a non-empty, non-zero distribution.
    """
    vals = sorted(values)
    n = len(vals)
    total = sum(vals)
    pts = [(0.0, 0.0)]
    if n == 0 or total == 0:
        return pts
    cum = 0
    for i, v in enumerate(vals, 1):
        cum += v
        pts.append((i / n, cum / total))
    return pts


# ── Per-project GOLD distributions ────────────────────────────────────────────


def _sad_code_enrolled(project):
    code_files = load_code_model_files(project)
    enrolled = enroll(load_gs_sad_code_raw(project), code_files)
    return code_files, enrolled


def compute_sad_code_dist(project):
    """INEQ-01/02: per-sentence (the tab:sent_gini headline), per-file, and
    per-component gold concentration for sad-code."""
    code_files, enrolled = _sad_code_enrolled(project)

    per_sent = list(Counter(s for (s, _f) in enrolled).values())
    per_file = list(Counter(f for (_s, f) in enrolled).values())
    ss, sf = summary_stats(per_sent), summary_stats(per_file)

    # Per-component, mapped-only universe (mirror component_suite._code_inputs:
    # drop files with no SAM-CODE component so this agrees with gold_gini).
    names, sam_enrolled = load_sam_code(project, code_files)
    file_to_comps = defaultdict(set)
    for ae, fp in sam_enrolled:
        file_to_comps[fp].add(names.get(ae, ae))
    comp_sents = defaultdict(set)
    for s, f in enrolled:
        comps = file_to_comps.get(f)
        if not comps:
            continue
        for c in comps:
            comp_sents[c].add(s)
    spc = [len(v) for v in comp_sents.values()]
    sc = summary_stats(spc)

    return {
        "project": project,
        "sent_n": ss["n"], "sent_min": ss["min"],
        "sent_median": ss["median"], "sent_max": ss["max"],
        "sent_gini": _gini(per_sent),
        "sent_top3_pct": 100 * top_k_share(per_sent, 3),
        "file_n": sf["n"], "file_min": sf["min"],
        "file_median": sf["median"], "file_max": sf["max"],
        "file_gini": _gini(per_file),
        "file_top3_pct": 100 * top_k_share(per_file, 3),
        "comp_n": sc["n"], "comp_sent_min": sc["min"],
        "comp_sent_median": sc["median"], "comp_sent_max": sc["max"],
        "comp_sent_gini": _gini(spc),
        "comp_sent_top1_pct": 100 * top_k_share(spc, 1),
        "comp_sent_top3_pct": 100 * top_k_share(spc, 3),
    }


def compute_sadcode_link_conc(project):
    """Prestudy table (tab:gold_concentration): enrolled DOC-TO-CODE links grouped
    by architecture component via the SAM-CODE (model->code) mapping. Measures how
    the link-level F1 weight concentrates across exactly the component universe the
    size-aware suite scores.

    The universe is the suite's: D-12 drops ``Interface:`` model elements (code-
    twins of a Component with no doc signal) and components are keyed by ``ae_id``
    (same-named elements stay distinct). This mirrors mini-src/metrics.py
    load_file_to_comps so comp_n here equals the suite's component count (the
    'real' architectural units), NOT inequality.compute_sad_code_dist's comp_n,
    which keys by name and keeps interfaces. (Isolation rule: copy the rule, do
    not import mini-src.)

    A link whose target file realizes several components is counted under each, to
    mirror eq:comp-f1 (a shared target belongs to each such component). The per-
    component counts can therefore sum above the raw enrolled link total, which is
    reported separately as links_total."""
    code_files, enrolled = _sad_code_enrolled(project)
    names, sam_enrolled = load_sam_code(project, code_files)
    file_to_comps = defaultdict(set)
    for ae, fp in sam_enrolled:
        if names.get(ae, ae).startswith("Interface:"):   # D-12
            continue
        file_to_comps[fp].add(ae)
    per_comp = defaultdict(int)
    for _s, f in enrolled:
        for c in file_to_comps.get(f, ()):
            per_comp[c] += 1
    counts = list(per_comp.values())
    ss = summary_stats(counts)
    return {
        "project": project,
        "links_total": len(enrolled),
        "comp_n": ss["n"],
        "link_median": ss["median"], "link_max": ss["max"],
        "link_gini": _gini(counts),
        "link_top3_pct": 100 * top_k_share(counts, 3),
    }


def compute_sad_sam_dist(project):
    """INEQ-01: per-component (#distinct sentences per model element) gold
    concentration for sad-sam (by name where SAM-CODE provides one)."""
    gold = load_gs_sad_sam(project)
    code_files = load_code_model_files(project)
    names, _ = load_sam_code(project, code_files)
    comp_sents = defaultdict(set)
    for c, s in gold:
        comp_sents[names.get(c, c)].add(s)
    spc = [len(v) for v in comp_sents.values()]
    ss = summary_stats(spc)
    return {
        "project": project,
        "n_components": ss["n"],
        "sent_min": ss["min"], "sent_median": ss["median"], "sent_max": ss["max"],
        "comp_sent_gini": _gini(spc),
        "top1_pct": 100 * top_k_share(spc, 1),
        "top3_pct": 100 * top_k_share(spc, 3),
    }


def compute_samcode_skew(project):
    """INEQ-01 + INEQ-03 driver: SAM-CODE files-per-architectural-element fan-out
    (reproduces tab:samcode_skew). This is the structural amplification driver."""
    code_files = load_code_model_files(project)
    _names, sam_enrolled = load_sam_code(project, code_files)
    m2f = defaultdict(set)
    for ae, fp in sam_enrolled:
        m2f[ae].add(fp)
    fv = [len(v) for v in m2f.values()]
    ss = summary_stats(fv)
    return {
        "project": project,
        "aes": ss["n"],
        "enrolled": ss["total"],
        "min": ss["min"], "median": ss["median"], "max": ss["max"],
        "gini": _gini(fv),
        "top3_conc_pct": 100 * top_k_share(fv, 3),
        "mean_fanout": (ss["total"] / ss["n"]) if ss["n"] else 0.0,
    }


def compute_expansion(project):
    """INEQ-03: gold enrollment expansion (reproduces tab:enrollment, sad-code)."""
    raw = load_gs_sad_code_raw(project)
    raw_n = len(raw)
    dir_entries = sum(1 for (_s, p) in raw if p.endswith("/"))
    code_files = load_code_model_files(project)
    enrolled_n = len(enroll(raw, code_files))
    return {
        "project": project,
        "raw": raw_n,
        "dir_entries": dir_entries,
        "dir_pct": (100 * dir_entries / raw_n) if raw_n else 0.0,
        "enrolled": enrolled_n,
        "factor": (enrolled_n / raw_n) if raw_n else 0.0,
    }


def amplification_summary(project):
    """INEQ-03 (re-pivoted): the structural component->file amplification is a GOLD
    property — one component-level decision expands to up to max|files(m)| file
    pairs; aggregate amplification = the enrollment factor. NO system results,
    NO TransArc error attribution (user directive 2026-06-21)."""
    sk = compute_samcode_skew(project)
    ex = compute_expansion(project)
    return {
        "project": project,
        "n_components": sk["aes"],
        "mean_fanout": sk["mean_fanout"],
        "max_fanout": sk["max"],
        "enrollment_factor": ex["factor"],
    }


# ── Output ────────────────────────────────────────────────────────────────────


def _cellfmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v == float("inf"):
            return "inf"
        return f"{v:.3f}"
    return "" if v is None else str(v)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def write_csv(path, header, rows, agg_label=None):
    """Write header + rows; optionally append an aggregate row (mean of numeric
    columns, inf-skipped). For sum/total aggregates, build the row explicitly and
    pass it in `rows` with agg_label=None."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([_cellfmt(r.get(c)) for c in header])
        if agg_label and rows:
            agg = {header[0]: agg_label}
            for c in header[1:]:
                vals = [r[c] for r in rows
                        if _is_num(r.get(c)) and r[c] != float("inf")]
                agg[c] = (sum(vals) / len(vals)) if vals else ""
            w.writerow([_cellfmt(agg.get(c)) for c in header])


SAD_CODE_HEADER = [
    "project", "sent_n", "sent_min", "sent_median", "sent_max", "sent_gini",
    "sent_top3_pct", "file_n", "file_min", "file_median",
    "file_max", "file_gini", "file_top3_pct", "comp_n", "comp_sent_min",
    "comp_sent_median", "comp_sent_max", "comp_sent_gini", "comp_sent_top1_pct",
    "comp_sent_top3_pct",
]
SAD_SAM_HEADER = [
    "project", "n_components", "sent_min", "sent_median", "sent_max",
    "comp_sent_gini", "top1_pct", "top3_pct",
]
SAMCODE_HEADER = [
    "project", "aes", "enrolled", "min", "median", "max", "gini",
    "top3_conc_pct", "mean_fanout",
]
EXPANSION_HEADER = ["project", "raw", "dir_entries", "dir_pct", "enrolled", "factor"]


def write_all_csvs(projects, task="both"):
    # Only inequality_expansion.csv feeds the alinker-paper PDF (the enrollment
    # factor 1.0x-217.6x cited in sec:metric:prestudy). The per-sentence,
    # per-component, samcode-skew and Lorenz CSVs are non-PDF; their output is
    # silenced. (`task` is kept for CLI compatibility; it no longer selects CSVs.)
    exp_rows = [compute_expansion(p) for p in projects]
    total = {
        "project": "Total",
        "raw": sum(r["raw"] for r in exp_rows),
        "dir_entries": sum(r["dir_entries"] for r in exp_rows),
        "dir_pct": "",
        "enrolled": sum(r["enrolled"] for r in exp_rows),
    }
    total["factor"] = (total["enrolled"] / total["raw"]) if total["raw"] else 0.0
    write_csv(REPORTS / "inequality_expansion.csv", EXPANSION_HEADER,
              exp_rows + [total])


# ── Sanity gate (recompute vs frozen eval.tex GOLD literals) ──────────────────

# All values are GOLD/benchmark properties taken from writing/eval.tex Chapter 1.
EXPECTED = {
    # tab:sent_gini — per-sentence enrolled sad-code link distribution.
    "sent_gini": {"mediastore": 0.331, "teastore": 0.448, "teammates": 0.645,
                  "bigbluebutton": 0.472, "jabref": 0.527},
    # tab:samcode_skew — SAM-CODE files per architectural element.
    "samcode_gini": {"mediastore": 0.400, "teastore": 0.694, "teammates": 0.452,
                     "bigbluebutton": 0.513, "jabref": 0.612},
    "samcode_aes": {"mediastore": 19, "teastore": 19, "teammates": 14,
                    "bigbluebutton": 22, "jabref": 6},
    "samcode_enrolled": {"mediastore": 60, "teastore": 164, "teammates": 1616,
                         "bigbluebutton": 730, "jabref": 1956},
    "samcode_max": {"mediastore": 16, "teastore": 64, "teammates": 348,
                    "bigbluebutton": 94, "jabref": 972},
    # tab:enrollment — gold enrollment expansion.
    "enroll_enrolled": {"mediastore": 59, "teastore": 707, "teammates": 8097,
                        "bigbluebutton": 1529, "jabref": 8268},
    "enroll_factor": {"mediastore": 1.0, "teastore": 10.1, "teammates": 35.5,
                      "bigbluebutton": 11.6, "jabref": 217.6},
    "enroll_total_enrolled": 18660,
    "enroll_total_raw": 525,
}
GINI_TOL = 0.005


def run_check(projects):
    """Recompute each GOLD value and compare to the frozen eval.tex literals.
    Gini within +/-0.005; integer counts exact. Returns (ok, rows)."""
    sc = {p: compute_sad_code_dist(p) for p in projects}
    sk = {p: compute_samcode_skew(p) for p in projects}
    ex = {p: compute_expansion(p) for p in projects}
    rows = []  # (metric, scope, expected, computed, delta, ok)

    def add_gini(metric, exp_map, getter):
        for p in projects:
            if p not in exp_map:
                continue
            e, c = exp_map[p], getter(p)
            rows.append((metric, p, e, round(c, 3), round(c - e, 4),
                         abs(c - e) <= GINI_TOL))

    def add_int(metric, exp_map, getter):
        for p in projects:
            if p not in exp_map:
                continue
            e, c = exp_map[p], int(getter(p))
            rows.append((metric, p, e, c, c - e, c == e))

    def add_round1(metric, exp_map, getter):
        # The paper reports the enrollment factor to 1 decimal place; compare at
        # that precision (the load-bearing enrolled counts are checked exactly).
        for p in projects:
            if p not in exp_map:
                continue
            e, c = exp_map[p], round(getter(p), 1)
            rows.append((metric, p, e, c, round(c - e, 4), c == e))

    add_gini("sent_gini", EXPECTED["sent_gini"], lambda p: sc[p]["sent_gini"])
    add_gini("samcode_gini", EXPECTED["samcode_gini"], lambda p: sk[p]["gini"])
    add_int("samcode_aes", EXPECTED["samcode_aes"], lambda p: sk[p]["aes"])
    add_int("samcode_enrolled", EXPECTED["samcode_enrolled"], lambda p: sk[p]["enrolled"])
    add_int("samcode_max", EXPECTED["samcode_max"], lambda p: sk[p]["max"])
    add_int("enroll_enrolled", EXPECTED["enroll_enrolled"], lambda p: ex[p]["enrolled"])
    add_round1("enroll_factor", EXPECTED["enroll_factor"], lambda p: ex[p]["factor"])

    if set(projects) == set(PROJECTS):
        te = sum(ex[p]["enrolled"] for p in projects)
        tr = sum(ex[p]["raw"] for p in projects)
        rows.append(("enroll_total_enrolled", "ALL", EXPECTED["enroll_total_enrolled"],
                     te, te - EXPECTED["enroll_total_enrolled"],
                     te == EXPECTED["enroll_total_enrolled"]))
        rows.append(("enroll_total_raw", "ALL", EXPECTED["enroll_total_raw"],
                     tr, tr - EXPECTED["enroll_total_raw"],
                     tr == EXPECTED["enroll_total_raw"]))

    return all(r[5] for r in rows), rows


def print_check(rows):
    print(f"\n=== SANITY CHECK (vs writing/eval.tex; tol: Gini<={GINI_TOL}, counts exact) ===")
    print(f"{'metric':22}{'scope':14}{'expected':>10}{'computed':>10}{'delta':>10}  ok")
    for metric, scope, e, c, d, ok in rows:
        print(f"{metric:22}{scope:14}{str(e):>10}{str(c):>10}{str(d):>10}  "
              f"{'PASS' if ok else 'FAIL'}")


# ── Markdown report ───────────────────────────────────────────────────────────


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_cellfmt(r.get(c)) for c in header) + " |")
    return "\n".join(out)


def write_report(projects, check_ok, check_rows):
    sc = [compute_sad_code_dist(p) for p in projects]
    sm = [compute_sad_sam_dist(p) for p in projects]
    sk = [compute_samcode_skew(p) for p in projects]
    ex = [compute_expansion(p) for p in projects]
    amp = [amplification_summary(p) for p in projects]
    exp_total_enr = sum(r["enrolled"] for r in ex)
    exp_total_raw = sum(r["raw"] for r in ex)
    max_fanout = max((a["max_fanout"] for a in amp), default=0)

    L = []
    L.append("# Trace-Link Data-Inequality — Gold Benchmark Distribution\n")
    L.append("> Self-contained, stdlib-only, **gold/benchmark only** (no system "
             "results). Reproduces the inequality tables of `writing/eval.tex` "
             "Chapter 1 and self-checks against their frozen literals.\n")

    L.append("## Per-sentence gold concentration — Gini "
             f"{sc[0]['sent_gini']:.3f} (MediaStore) → "
             f"{max(r['sent_gini'] for r in sc):.3f} (peak)\n")
    L.append("The per-sentence enrolled sad-code link distribution is heavily "
             "right-skewed (eval.tex `tab:sent_gini`).\n")
    L.append(_md_table(
        ["project", "sent_n", "sent_min", "sent_median", "sent_max",
         "sent_gini", "sent_top3_pct"], sc) + "\n")

    L.append("## Enrollment expansion — "
             f"{exp_total_raw} raw decisions → {exp_total_enr:,} enrolled file "
             f"links ({exp_total_enr / exp_total_raw:.1f}× avg, up to "
             f"{max(r['factor'] for r in ex):.1f}× on JabRef)\n")
    L.append("Directory-level gold entries are enrolled to every file beneath "
             "them (eval.tex `tab:enrollment`), inflating a few hundred decisions "
             "into tens of thousands of file-level points.\n")
    L.append(_md_table(EXPANSION_HEADER, ex) + "\n")

    L.append("## Structural component→file amplification — files-per-component "
             f"Gini {sk[0]['gini']:.3f} → {max(r['gini'] for r in sk):.3f}; one "
             f"component decision expands to up to {int(max_fanout)} file pairs\n")
    L.append("The dataset-intrinsic amplification driver is the SAM-CODE "
             "files-per-architectural-element fan-out (eval.tex `tab:samcode_skew`): "
             "a single component-level decision structurally maps to up to "
             f"{int(max_fanout)} code files (JabRef `logic` = 972, Teammates `ui` "
             "= 348). Aggregate amplification equals the enrollment factor. This is "
             "a GOLD property — NO system results are used. (eval.tex "
             "`tab:amplification`, the TransArc actual-error cascade 36→3,457, is a "
             "system-specific quantity and is intentionally excluded from this "
             "gold-only study.)\n")
    L.append(_md_table(SAMCODE_HEADER, sk) + "\n")
    L.append("**Structural amplification potential (gold):**\n")
    L.append(_md_table(["project", "n_components", "mean_fanout", "max_fanout",
                        "enrollment_factor"], amp) + "\n")

    L.append("## Per-component & per-sentence concentration (supplementary skew)\n")
    L.append("Per-component #sentences inequality, sad-sam side (eval.tex "
             "`gold_gini` analogue):\n")
    L.append(_md_table(SAD_SAM_HEADER, sm) + "\n")

    status = "PASS ✓" if check_ok else "FAIL ✗"
    L.append("## Sanity check\n")
    L.append(f"Status: **{status}** — every reproduced number agrees with the "
             f"frozen `writing/eval.tex` literals (tolerance: Gini ≤ {GINI_TOL}, "
             "integer counts exact). No system results were read; the engine is "
             "stdlib-only and self-contained.\n")
    fails = [r for r in check_rows if not r[5]]
    if fails:
        L.append("Failing rows: " + "; ".join(
            f"{m}/{s} exp {e} got {c}" for m, s, e, c, _d, _ok in fails) + "\n")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "INEQUALITY.md").write_text("\n".join(L) + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["sad-code", "sad-sam", "both"],
                    default="both", help="which CSVs to write (the gate always "
                    "checks the full gold set)")
    ap.add_argument("--project", default=None, help="single project (default: all)")
    ap.add_argument("--check-only", action="store_true",
                    help="run only the sanity gate")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the sanity gate")
    args = ap.parse_args()

    if args.project and args.project not in PROJECTS:
        sys.exit(f"unknown project {args.project!r}; expected one of {PROJECTS}")
    projects = [args.project] if args.project else list(PROJECTS)

    if args.check_only:
        ok, rows = run_check(projects)
        print_check(rows)
        print("SANITY CHECK PASSED (tol: Gini<=0.005, counts exact)" if ok
              else "SANITY CHECK FAILED")
        sys.exit(0 if ok else 1)

    write_all_csvs(projects, args.task)

    ok, rows = (True, [])
    if not args.no_check:
        ok, rows = run_check(projects)

    # INEQUALITY.md (the full study report) is non-PDF; its output is silenced.
    # write_report() is retained above and can be re-enabled if needed.

    print(f"[inequality] projects={len(projects)} reports={REPORTS} "
          f"(inequality_expansion.csv only)")
    if not args.no_check:
        print_check(rows)
        if ok:
            print("SANITY CHECK PASSED (tol: Gini<=0.005, counts exact)")
        else:
            print("SANITY CHECK FAILED")
            sys.exit(1)


if __name__ == "__main__":
    main()
