#!/usr/bin/env python3
"""The metrics for ARDoCo doc-to-code / doc-to-model trace-link recovery.

A single, self-contained, stdlib-only module — the project's SOLE metrics
implementation AND the shared core of the whole ``evaluation/`` tree: the
benchmark layout (roots, project list, gold-standard paths), the confusion
matrix / F-measures, the gold loaders and the CSV writer live here once and
every other engine imports them (``rq12``, ``rq34``, ``rq34_rq2``, ``rq4_floor``,
``build_dump``, ``mini-inequality/inequality``, ``../studies/``). Import
it; never re-copy a definition out of it. The former canonical stack it was reduced from
(``src/lib/metrics_api.py``, ``src/bias/component_suite.py``, the RQ2/bias
side-analyses, ``generate_tables.py``) has been retired to ``archive/``; only
the base loaders (``src/lib/transarc_error_analysis.py``) are kept. The
redundancy analysis that justified the reduction showed the dropped columns
carry no independent ranking signal (Spearman rho >= 0.85 with a kept metric,
~0 system-pair reversals): no MCC/MAP/ACF1/NDG/HUS/decision-F1/weighted-F1/
sentence-F1/per-component-macro.

Panel
-----
    sad-code (doc-to-code) : file P/R/F1, per-component F1 (micro),
                             worst-component F1, harmonic-mean component F1
    sad-sam  (doc-to-model): link P/R/F1, Component Miss Rate (CMR%)
                             (the MICRO per-component F1 collapses onto link F1 with
                             no enrolment, so it is dropped; CMR does NOT collapse
                             — it is the doc-model size-aware metric, added
                             2026-06-30, component--sentence denominator)

The companion Component Miss Count (CMC, the integer number of abandoned
components) was dropped 2026-09-01: no .tex table and no downstream engine ever
read the column, and CMR already prices the same abandonment. The paper's prose
counts ("abandons one documented component") were taken from it; nothing
regenerates them now.

The worst-component + harmonic pair is the paper's ``metric.tex`` size-aware
headline for DOC-CODE (weight each architecture component equally, not each link
pair); they stay doc-code-only (redundant with link-F1 on doc-model). The
doc-model size-aware metric is instead CMR (a missed component: the share of
documented component--sentence links whose component recovers no correct link),
added to ``compute_sad_sam`` only — ``compute_sad_code`` is untouched.
``mini-src/check.py`` pins every cell to a frozen golden table (validated at
retirement against the then-canonical ``metrics_api`` and the interface-dropped
``component_suite``). The whole computation lives here, in ~450 lines.

Definitions: see ``compute_sad_code`` (per-component grouping D-01, interface
drop D-12, worst/harmonic) and ``load_file_to_comps``. Briefly:
  * per-component F1 (micro) = one P/R/F1 over all (sentence, component) pairs.

Usage
-----
    python3 mini-src/metrics.py --task sad-code
    python3 mini-src/metrics.py --task sad-sam --project jabref
    python3 mini-src/metrics.py --task sad-code \
        --results-dir /path/to/run \
        --result-pattern 's_linker20_union_{project}_links.csv' \
        --csv /tmp/panel.csv

Result CSV columns are auto-detected, so both the TransArc dialect
(modelElementID, codeId / sentence) and the agent-linker dialect
(sentence, component_id, ...) are accepted.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# ── Benchmark layout (mirrors src/lib/transarc_error_analysis.py) ─────────────
# Defaults are derived from this file's location, so pressing Run on any script
# here works with no environment set:
#   <repo>/evaluation/mini-src/metrics.py  →  parents[2] is the repo root.
# Env vars still override, for an out-of-tree benchmark or result root. (Until
# 2026-09-01 these defaults pointed at the pre-nesting `<ardoco-home>/transarc-emp`
# layout, i.e. at nothing, so every bare run died on a missing path.)
REPO = Path(__file__).resolve().parents[2]
BENCHMARK = Path(os.environ.get("TRANSARC_BENCHMARK", REPO / "benchmark"))
DEFAULT_RESULTS = Path(os.environ.get("TRANSARC_RESULTS_DIR", REPO / "evaluation/mini-data"))

# Where the engines write. RQ1/RQ2 land straight in `reports/`; RQ3/RQ4 are arm-scoped,
# one directory per arm under `reports/rq34/`, because three engines (`rq34`, `rq34_rq2`,
# `rq4_floor`) write into the same arm directory and `rq_tables.py` reads all three from
# it. Kept here so the four modules cannot disagree about the layout.
REPORTS = REPO / "evaluation" / "reports"
RQ34_REPORTS = REPORTS / "rq34"

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

# Column-name candidates so result CSVs in every dialect are accepted:
#   - TransArc / legacy : modelElementID, codeId / sentence
#   - agent-linker      : sentence, component_id, codeID, ...
#   - recovered-links   : sentence_id, target_id (sota/recovered-links/*; the
#                         normalized SOTA-baseline dump — target_id is the PCM
#                         element id for sad-sam and the code file path for
#                         sad-code). Probed first-non-empty, so adding these is
#                         additive: files lacking the column are unaffected.
_SADSAM_COMPONENT_KEYS = ("modelElementID", "component_id", "componentId", "target_id")
_SADSAM_SENTENCE_KEYS = ("sentence", "sentence_id")
# `modelElementID` is overloaded: in a sad-code dump it holds the SENTENCE
# NUMBER, but in a sad-sam dump it holds a model-element GUID. Probe the
# dedicated sentence columns FIRST so a CSV that carries both a real `sentence`
# column and a GUID `modelElementID` column is read correctly; fall back to
# `modelElementID` only for the TransArc sad-code dialect that has nothing else.
_SADCODE_SENTENCE_KEYS = ("sentence", "sentence_id", "modelElementID")
_SADCODE_CODE_KEYS = ("codeId", "codeID", "code_path", "target_id")


# ── Core metric primitives ────────────────────────────────────────────────────

def prf_counts(gold, res):
    """(tp, fp, fn, precision, recall, f1) for two link sets.

    The tree's ONE confusion-matrix + F1 computation. Every engine reads it --
    ``prf`` below (RQ1/RQ2), ``rq34.prf``/``rq34.prf3`` (RQ3/RQ4, which also
    report the raw counts) and ``build_dump``'s manifest integrity figure -- so
    no caller re-derives an F-measure.

    Convention: an empty prediction always scores 0 -- including the degenerate
    empty-gold/empty-res case. This is deliberate and load-bearing: the
    worst/harmonic suite relies on an abandoned component (empty ``res``)
    yielding F1 = 0, so "predicted nothing" is never treated as vacuously
    perfect.
    """
    tp = len(gold & res)
    fp = len(res - gold)
    fn = len(gold - res)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)
    return tp, fp, fn, precision, recall, f1


def prf(gold, res):
    """(precision, recall, f1) treating gold/res as sets of links.

    Thin projection of ``prf_counts``; see it for the empty-prediction rule.
    """
    return prf_counts(gold, res)[3:]


def fbeta(precision, recall, beta=2.0):
    """F-beta from a precision/recall pair; ``beta=2`` is the paper's \\ftwo.

    Recall-weighted: a missed gold link costs beta^2 = 4x what a spurious one
    does. Reported next to every link-level and file-level F1 because a
    developer reading recovered links can discard a wrong one but cannot see a
    link that was never proposed. Same empty-prediction convention as ``prf``:
    precision = recall = 0 -> 0.0, never vacuously perfect.
    """
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def _avg_ranks(xs):
    """1-based ranks with ties resolved to the average of the tied positions."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # mean of 0-based positions i..j, made 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman's rho (Pearson on average ranks; tie-aware). Stdlib only.

    Not part of any reported panel: it is the shared rank-correlation primitive
    the side analyses (``studies/explore-tail/``) use to check whether a candidate
    metric carries ranking signal the reference F1 does not. Lives here so those
    studies have one implementation to import, alongside the loaders.
    """
    n = len(xs)
    rx, ry = _avg_ranks(xs), _avg_ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx * vy) ** 0.5


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


# ── Loaders ───────────────────────────────────────────────────────────────────

def _cell(row, keys):
    """First non-empty value among `keys` in a DictReader row, else None."""
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


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
    """(names: ae_id -> ae_name, enrolled: set[(ae_id, file)]) from the SAM-CODE gold.

    The raw model->code mapping with directory entries enrolled against the code
    model and nothing dropped. ``load_file_to_comps`` layers the suite's D-12
    interface drop on top; ``mini-inequality`` reads the unfiltered pair (its
    files-per-element skew counts every architecture element).
    """
    names, raw = {}, set()
    with open(BENCHMARK / GS_SAM_CODE[project]) as f:
        for r in csv.DictReader(f):
            names[r["ae_id"]] = r["ae_name"]
            raw.add((r["ae_id"], normalize_path(r.get("ce_ids") or r.get("ce_id"))))
    return names, enroll(raw, code_files)


def load_file_to_comps(project, code_files):
    """file_path -> {component ae_id}, from the enrolled SAM-CODE gold.

    Components are keyed by ``ae_id`` (guaranteed unique), not ``ae_name``: two
    architecture elements that happened to share a display name would otherwise
    silently merge into one component bucket and distort the worst/harmonic
    suite. (``ae_id`` <-> ``ae_name`` is currently 1:1 for every non-interface
    component, so this keying does not change any panel value.)
    """
    names, sam_enrolled = load_sam_code(project, code_files)
    file_to_comps = defaultdict(set)
    for ae, fp in sam_enrolled:
        if names.get(ae, ae).startswith("Interface:"):  # D-12: see compute_sad_code
            continue
        file_to_comps[fp].add(ae)
    return file_to_comps


def load_result(path, task):
    """Read a result CSV into a link set, auto-detecting the column dialect.

    sad-code -> set[(sentence, normalized_path)]
    sad-sam  -> set[(component_id, sentence)]
    Empty set if the file is absent.
    """
    links = set()
    if not path.exists():
        return links
    guid_sentences = False
    with open(path) as f:
        for row in csv.DictReader(f):
            if task == "sad-code":
                s = _cell(row, _SADCODE_SENTENCE_KEYS)
                c = _cell(row, _SADCODE_CODE_KEYS)
                if s and c:
                    # sad-code sentences are sentence NUMBERS; a GUID-shaped
                    # value means we are probably scoring a sad-sam dump as
                    # sad-code (see _SADCODE_SENTENCE_KEYS). Flag, don't drop.
                    if s.startswith("_"):
                        guid_sentences = True
                    links.add((s, normalize_path(c)))
            else:
                c = _cell(row, _SADSAM_COMPONENT_KEYS)
                s = _cell(row, _SADSAM_SENTENCE_KEYS)
                if c and s:
                    links.add((c, s))
    if guid_sentences:
        print(f"WARNING: {path} has GUID-shaped sad-code sentence ids "
              f"(e.g. '_...') — is this actually a sad-sam result? "
              f"Scored as sad-code anyway.", file=sys.stderr)
    return links


def result_path(project, results_dir, result_pattern, task):
    """Default TransArc layout, or `result_pattern.format(project=...)`."""
    root = Path(results_dir) if results_dir else DEFAULT_RESULTS
    if result_pattern:
        return root / result_pattern.format(project=project)
    sub, prefix = (("sad-code", "sadCodeTlr") if task == "sad-code"
                   else ("sad-sam", "sadSamTlr"))
    return root / project / sub / f"{prefix}_{project}.csv"


# ── Per-project metric rows ───────────────────────────────────────────────────

def compute_sad_code(project, res):
    """Primary panel + size-aware suite for one doc-to-code result set.

    Per-component grouping (D-01): each (sentence, file) link maps to one
    (sentence, component) pair per SAM-CODE component that owns the file; files
    with NO component are DROPPED (same rule for gold and result).

    D-12 (interface drop): ``Interface:`` model elements are excluded from the
    file->component map (see ``load_file_to_comps``). In the SAM-CODE gold every
    interface shares its code extent with a ``Component:`` twin (0 interface-only
    files) and the doc-to-model gold never links a sentence to an interface, so
    interfaces add no unique code and no documentation signal -- they only
    duplicate components or, where partially distinct (mediastore/teastore),
    inflate the per-component failure count. Keeping only ``Component:`` elements
    makes the component count the distinct architectural units (7/10/6/9/6); the
    size-aware tail metrics are invariant to duplicating a component, so worst
    and harmonic are unchanged by the drop.

    Size-aware suite (the paper's ``metric.tex`` headline, weighting each
    architecture component equally rather than each link pair):
      * ``component_f1``           -- micro F1 over all (sentence, component) pairs.
                                      NOTE this one IS the projected grain (a sentence
                                      counts once per component it reaches); it is a
                                      CSV-only diagnostic and feeds no paper float.
      * ``worst_component_f1``     -- min per-component F1 over GOLD components, each
                                      scored on the (sentence, file) links whose target
                                      belongs to it (eq:worst); one abandoned component
                                      drives it to 0
      * ``harmonic_component_f1``  -- harmonic mean of the same per-component F1 over
                                      GOLD components; also 0 if any component is missed

    Each of the three component F1s ships with its recall-weighted twin
    (``component_f2``, ``worst_component_f2``, ``harmonic_component_f2``), computed
    the same way but closing with ``fbeta`` instead of F1, so every F1 this module
    reports has an \ftwo beside it. The aggregation is over the F2 scores, not the
    F2 of the worst-F1 component: worst-F2 is the min of the per-component F2s and
    harmonic-F2 their harmonic mean. The two usually name the same component, and
    both flavours zero out on the same abandoned one.
    """
    code_files = load_code_model_files(project)
    gold = enroll(load_gs_sad_code_raw(project), code_files)
    file_to_comps = load_file_to_comps(project, code_files)

    # NOTE: only the GOLD is enrolled (directory entries -> concrete files).
    # `res` is intentionally left un-enrolled: result producers emit concrete
    # file paths, and enrolling a predicted directory would let a system claim
    # credit for every file under it -- exactly the enrollment inflation this
    # work studies. Do NOT add symmetric enrollment of `res` here.
    fp_, fr_, ff1 = prf(gold, res)

    def to_comp(pairs):
        out = set()
        for s, c in pairs:
            for comp in file_to_comps.get(c, ()):
                out.add((s, comp))
        return out
    gold_c, res_c = to_comp(gold), to_comp(res)
    comp_p, comp_r, comp_f1 = prf(gold_c, res_c)

    # Per-component scores over the GOLD-only component universe -> worst + harmonic.
    # GRAIN (metric.tex eq:worst / eq:harm, literally): "each component k owns a set
    # of code files; F_beta(k) is F_beta computed over only the LINKS whose target
    # belongs to k". So each component is scored on its own slice of the (sentence,
    # file) link set -- NOT on a (sentence, component) projection. The distinction is
    # load-bearing: under a projection, recovering ONE file of k scores the same as
    # recovering all of them, which makes the tail metrics blind to how much of a
    # component was actually reached (and, empirically, invariant across runs). A file
    # realizing two components contributes its link to both slices, as eq:worst implies.
    gold_by_c, res_by_c = defaultdict(set), defaultdict(set)
    for s, f in gold:
        for comp in file_to_comps.get(f, ()):
            gold_by_c[comp].add((s, f))
    for s, f in res:
        for comp in file_to_comps.get(f, ()):
            res_by_c[comp].add((s, f))

    def comp_score(c):
        """(F1, F2) for one gold component, over the links whose target belongs to c."""
        p, rec, f1 = prf(gold_by_c.get(c, set()), res_by_c.get(c, set()))
        return f1, fbeta(p, rec)

    def tail(scores):
        """(worst, harmonic mean) of the per-component scores.

        Both collapse to 0 as soon as one gold component is abandoned: ``prf`` and
        ``fbeta`` score an empty prediction 0, and a single 0 zeroes a harmonic mean.
        That is what the pair buys over link-level F -- which absorbs a missed small
        component -- and it holds for the F1 and the F2 flavour alike.
        """
        if not scores:
            return 0.0, 0.0
        harmonic = (len(scores) / sum(1.0 / x for x in scores)
                    if all(x > 0 for x in scores) else 0.0)
        return min(scores), harmonic

    per_gold = [comp_score(c) for c in gold_by_c]
    worst_f1, harmonic_f1 = tail([f1 for f1, _ in per_gold])
    worst_f2, harmonic_f2 = tail([f2 for _, f2 in per_gold])

    return {
        "project": project,
        "file_p": fp_, "file_r": fr_, "file_f1": ff1, "file_f2": fbeta(fp_, fr_),
        "component_f1": comp_f1, "component_f2": fbeta(comp_p, comp_r),
        "worst_component_f1": worst_f1, "worst_component_f2": worst_f2,
        "harmonic_component_f1": harmonic_f1, "harmonic_component_f2": harmonic_f2,
    }


def compute_sad_sam(project, res):
    """Primary panel + Component Miss Rate for one doc-to-model result set.

    The Component Miss Rate (CMR) is the doc-model size-aware metric (the doc-code
    worst/harmonic tail is redundant with link-F1 here, so it is NOT reported on
    doc-model; see the module docstring). Definitions, over GOLD components and
    gold documentation assignments:

      * component c is ABANDONED iff ``recall_c == 0`` -- it recovers no correct
        link (zero correct sentences for c), reusing ``prf``'s convention that an
        empty/all-wrong prediction scores recall 0.
      * CMR = sum(|gold sentences for c| for abandoned c)
              / sum(|gold sentences for c| for gold c) * 100            (%, [0,100])

    NOTE the unit: ``metric.tex`` defines CMR as a share in [0,1]; this returns the
    same quantity in PERCENT, which is what every table and every prose figure in
    the paper prints ("1.9%"). Do not divide again downstream.

    CMR is sentence-weighted: every gold (sentence, component) assignment
    contributes one unit of mass. Thus, a sentence documented for two components
    contributes twice, once for each component. Empty gold -> CMR 0.0.

    The integer count of abandoned components (CMC) was returned here until
    2026-09-01; it was dropped unread (see the module docstring).
    """
    gold = load_gs_sad_sam(project)
    lp, lr, lf1 = prf(gold, res)

    # Group gold assignments by component, then retain only exact gold hits. A
    # component is abandoned when its set of correct sentences remains empty.
    gold_sentences_by_component = defaultdict(set)
    correct_sentences_by_component = defaultdict(set)
    for c, s in gold:
        gold_sentences_by_component[c].add(s)
    for c, s in (gold & res):
        correct_sentences_by_component[c].add(s)
    abandoned_components = {
        c for c in gold_sentences_by_component
        if not correct_sentences_by_component.get(c)
    }

    # Each (sentence, component) assignment contributes one unit of mass. Do
    # not union the sentence sets: shared sentences deliberately retain one unit
    # for every documented component.
    abandoned_assignment_count = sum(
        len(gold_sentences_by_component[c]) for c in abandoned_components
    )
    total_assignment_count = sum(
        len(sentences) for sentences in gold_sentences_by_component.values()
    )
    cmr = (abandoned_assignment_count / total_assignment_count * 100
           if total_assignment_count else 0.0)

    return {
        "project": project,
        "link_p": lp, "link_r": lr, "link_f1": lf1, "link_f2": fbeta(lp, lr),
        "component_miss_rate": cmr,
    }


# ── CLI / output ──────────────────────────────────────────────────────────────

PANELS = {
    "sad-code": ["file_p", "file_r", "file_f1", "file_f2",
                 "component_f1", "component_f2",
                 "worst_component_f1", "worst_component_f2",
                 "harmonic_component_f1", "harmonic_component_f2"],
    "sad-sam":  ["link_p", "link_r", "link_f1", "link_f2",
                 "component_miss_rate"],
}
HEADERS = {
    "file_p": "file_P", "file_r": "file_R", "file_f1": "file_F1", "file_f2": "file_F2",
    "link_p": "link_P", "link_r": "link_R", "link_f1": "link_F1", "link_f2": "link_F2",
    "component_f1": "comp_F1", "component_f2": "comp_F2",
    "worst_component_f1": "worst_C_F1", "worst_component_f2": "worst_C_F2",
    "harmonic_component_f1": "harm_C_F1", "harmonic_component_f2": "harm_C_F2",
    "component_miss_rate": "CMR%",
}

def average_row(rows, cols):
    # Label carries the contributor count so a partial average (some projects
    # skipped for missing results) is never silently presented as a full one.
    avg = {"project": f"Average (n={len(rows)})"}
    for c in cols:
        avg[c] = sum(r[c] for r in rows) / len(rows) if rows else 0.0
    return avg


def print_table(task, rows):
    cols = PANELS[task]
    w = max(13, max((len(r["project"]) for r in rows), default=7) + 1)
    # Derived, not fixed: the F1/F2 header pairs are long enough that a hardcoded
    # width silently runs them together.
    cw = max(10, max(len(HEADERS[c]) for c in cols) + 1)
    head = "project".ljust(w) + "".join(HEADERS[c].rjust(cw) for c in cols)
    print(head)
    print("-" * len(head))
    for r in rows:
        line = r["project"].ljust(w) + "".join(f"{r[c]:{cw}.4f}" for c in cols)
        print(line)
    if len(rows) > 1:   # an average over a single project is just that project
        avg = average_row(rows, cols)
        print("-" * len(head))
        print(avg["project"].ljust(w) + "".join(f"{avg[c]:{cw}.4f}" for c in cols))


def write_csv(task, rows, path):
    cols = PANELS[task]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["project"] + cols)
        for r in list(rows) + ([average_row(rows, cols)] if len(rows) > 1 else []):
            w.writerow([r["project"]] + [f"{r[c]:.4f}" for c in cols])


def write_dict_csv(path, fieldnames, rows):
    """Write dict rows to `path`: parents created, LF line endings, UTF-8.

    The tree's one CSV writer for dict rows -- ``rq34``, ``rq34_rq2`` and
    ``rq_tables`` all emit through it, so the dialect cannot drift between the
    engines that feed the same paper tables.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["sad-code", "sad-sam"])
    ap.add_argument("--project", default=None,
                    help="single project (default: all five)")
    ap.add_argument("--results-dir", default=None,
                    help="root holding result CSVs (default: bundled mini-data/)")
    ap.add_argument("--result-pattern", default=None,
                    help="filename pattern with {project}, relative to --results-dir")
    ap.add_argument("--csv", default=None, help="also write the panel to this CSV")
    args = ap.parse_args()

    if args.project and args.project not in PROJECTS:
        ap.error(f"unknown project {args.project!r}; expected one of {PROJECTS}")
    projects = [args.project] if args.project else PROJECTS
    compute = compute_sad_code if args.task == "sad-code" else compute_sad_sam

    rows = []
    for proj in projects:
        path = result_path(proj, args.results_dir, args.result_pattern, args.task)
        res = load_result(path, args.task)
        if not res:
            print(f"WARNING: no {args.task} results for {proj} "
                  f"(looked in {path}), skipping", file=sys.stderr)
            continue
        rows.append(compute(proj, res))

    print_table(args.task, rows)
    if args.csv:
        write_csv(args.task, rows, args.csv)
        print(f"\n[mini-metrics] wrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
