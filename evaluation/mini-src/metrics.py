#!/usr/bin/env python3
"""The metrics for ARDoCo doc-to-code / doc-to-model trace-link recovery.

A single, self-contained, stdlib-only module — the project's SOLE metrics
implementation. The former canonical stack it was reduced from
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
                             worst-component F1, harmonic-mean component F1,
                             sentence coverage, noise rate
    sad-sam  (doc-to-model): link P/R/F1, sentence coverage, noise rate,
                             Silent-Failure Mass (SFM%) + Silent-Failure Count (SFC)
                             (the MICRO per-component F1 collapses onto link F1 with
                             no enrolment, so it is dropped; SFM/SFC do NOT collapse
                             — they are the doc-model size-aware metric, added
                             2026-06-30, distinct-sentence denominator)

The worst-component + harmonic pair is the paper's ``metric.tex`` size-aware
headline for DOC-CODE (weight each architecture component equally, not each link
pair); they stay doc-code-only (redundant with link-F1 on doc-model). The
doc-model size-aware metric is instead SFM/SFC (silent component failure: the
share of documented sentences whose component recovers no correct link), added
to ``compute_sad_sam`` only — ``compute_sad_code`` is untouched.
``mini-src/check.py`` pins every cell to a frozen golden table (validated at
retirement against the then-canonical ``metrics_api`` and the interface-dropped
``component_suite``). The whole computation lives here, in ~450 lines.

Definitions: see ``compute_sad_code`` (per-component grouping D-01, interface
drop D-12, worst/harmonic) and ``load_file_to_comps``. Briefly:
  * per-component F1 (micro) = one P/R/F1 over all (sentence, component) pairs.
  * sentence coverage = fraction of gold sentences with >=1 *correct* hit.
  * noise rate = mean over *predicted* sentences of FP/(TP+FP); lower is better.

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
# Defaults are derived from this file's location rather than hardcoded:
#   <ardoco-home>/transarc-emp/mini-src/metrics.py  →  parents[2] is ardoco-home.
# Env vars still override for out-of-tree benchmark / result roots.
_ARDOCO_HOME = Path(__file__).resolve().parents[2]
BENCHMARK = Path(os.environ.get(
    "TRANSARC_BENCHMARK",
    _ARDOCO_HOME / "ardoco/core/tests-base/src/main/resources/benchmark",
))
DEFAULT_RESULTS = Path(os.environ.get(
    "TRANSARC_RESULTS_DIR",
    _ARDOCO_HOME / "transarc-emp/mini-data",
))

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

def prf(gold, res):
    """(precision, recall, f1) treating gold/res as sets of links.

    Convention: an empty prediction always scores (0, 0, 0) — including the
    degenerate empty-gold/empty-res case. This is deliberate and load-bearing:
    the worst/harmonic suite relies on an abandoned component (empty ``res``)
    yielding F1 = 0, so "predicted nothing" is never treated as vacuously
    perfect.
    """
    if not res:
        return 0.0, 0.0, 0.0
    tp = len(gold & res)
    precision = tp / len(res)
    recall = tp / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)
    return precision, recall, f1


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


def sentence_coverage(gold_by_s, res_by_s):
    """Fraction of gold sentences with >=1 correct prediction."""
    if not gold_by_s:
        return 0.0
    covered = sum(1 for s in gold_by_s if gold_by_s[s] & res_by_s.get(s, set()))
    return covered / len(gold_by_s)


def noise_rate(gold_by_s, res_by_s):
    """Mean FP/(TP+FP) across predicted sentences; lower is better."""
    vals = []
    for s, r in res_by_s.items():
        g = gold_by_s.get(s, set())
        tp = len(g & r)
        fp = len(r - g)
        if tp + fp > 0:
            vals.append(fp / (tp + fp))
    return sum(vals) / len(vals) if vals else 0.0


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


def load_file_to_comps(project, code_files):
    """file_path -> {component ae_id}, from the enrolled SAM-CODE gold.

    Components are keyed by ``ae_id`` (guaranteed unique), not ``ae_name``: two
    architecture elements that happened to share a display name would otherwise
    silently merge into one component bucket and distort the worst/harmonic
    suite. (``ae_id`` <-> ``ae_name`` is currently 1:1 for every non-interface
    component, so this keying does not change any panel value.)
    """
    names, raw = {}, set()
    with open(BENCHMARK / GS_SAM_CODE[project]) as f:
        for r in csv.DictReader(f):
            names[r["ae_id"]] = r["ae_name"]
            raw.add((r["ae_id"], normalize_path(r.get("ce_ids") or r.get("ce_id"))))
    file_to_comps = defaultdict(set)
    for ae, fp in enroll(raw, code_files):
        name = names.get(ae, ae)
        if name.startswith("Interface:"):   # D-12: see compute_sad_code docstring
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
      * ``component_f1``           -- micro F1 over all (sentence, component) pairs
      * ``worst_component_f1``     -- min per-component F1 over GOLD components;
                                      one abandoned component drives it to 0
      * ``harmonic_component_f1``  -- harmonic mean of per-component F1 over GOLD
                                      components; also 0 if any component is missed
      * ``sentence_coverage``      -- fraction of gold sentences with >=1 hit
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
    comp_f1 = prf(gold_c, res_c)[2]

    # Per-component F1 over the GOLD-only component universe -> worst + harmonic.
    gold_by_c, res_by_c = defaultdict(set), defaultdict(set)
    for s, c in gold_c:
        gold_by_c[c].add(s)
    for s, c in res_c:
        res_by_c[c].add(s)

    def comp_score(c):
        g = {(s, c) for s in gold_by_c.get(c, set())}
        r = {(s, c) for s in res_by_c.get(c, set())}
        return prf(g, r)[2]
    per_gold = [comp_score(c) for c in gold_by_c]
    worst = min(per_gold) if per_gold else 0.0
    harmonic = (len(per_gold) / sum(1.0 / x for x in per_gold)
                if per_gold and all(x > 0 for x in per_gold) else 0.0)

    gold_by_s, res_by_s = defaultdict(set), defaultdict(set)
    for s, c in gold:
        gold_by_s[s].add(c)
    for s, c in res:
        res_by_s[s].add(c)

    return {
        "project": project,
        "file_p": fp_, "file_r": fr_, "file_f1": ff1, "file_f2": fbeta(fp_, fr_),
        "component_f1": comp_f1,
        "worst_component_f1": worst,
        "harmonic_component_f1": harmonic,
        "sentence_coverage": sentence_coverage(gold_by_s, res_by_s),
        "noise_rate": noise_rate(gold_by_s, res_by_s),
    }


def compute_sad_sam(project, res):
    """Primary panel + Silent-Failure Mass/Count for one doc-to-model result set.

    Silent-Failure Mass (SFM) / Count (SFC) is the doc-model size-aware metric
    (the doc-code worst/harmonic tail is redundant with link-F1 here, so it is
    NOT reported on doc-model; see the module docstring). Definitions, over GOLD
    components and DISTINCT documentation sentences:

      * component c is ABANDONED iff ``recall_c == 0`` -- it recovers no correct
        link (zero correct sentences for c), reusing ``prf``'s convention that an
        empty/all-wrong prediction scores recall 0.
      * SFC = #{abandoned gold components}                              (integer)
      * SFM = |distinct documented sentences belonging to >=1 abandoned component|
              / |distinct documented sentences| * 100                  (%, [0,100])

    The denominator and numerator are DISTINCT sentences (a sentence gold-linked
    to two components is counted once) -- NOT the prototype's (sentence,component)
    decision grain. Empty gold -> SFM 0.0, SFC 0.
    """
    gold = load_gs_sad_sam(project)
    lp, lr, lf1 = prf(gold, res)

    gold_by_s, res_by_s = defaultdict(set), defaultdict(set)
    for c, s in gold:
        gold_by_s[s].add(c)
    for c, s in res:
        res_by_s[s].add(c)

    # SFM/SFC: comp -> distinct gold sentences; abandoned = comp with no CORRECT
    # link (gold & res), per the recall_c == 0 definition.
    gold_by_c, correct_by_c = defaultdict(set), defaultdict(set)
    for c, s in gold:
        gold_by_c[c].add(s)
    for c, s in (gold & res):
        correct_by_c[c].add(s)
    abandoned = {c for c in gold_by_c if not correct_by_c.get(c)}
    abandoned_sents = set().union(*(gold_by_c[c] for c in abandoned)) if abandoned else set()
    all_sents = set().union(*gold_by_c.values()) if gold_by_c else set()
    sfm = len(abandoned_sents) / len(all_sents) * 100 if all_sents else 0.0

    return {
        "project": project,
        "link_p": lp, "link_r": lr, "link_f1": lf1, "link_f2": fbeta(lp, lr),
        "sentence_coverage": sentence_coverage(gold_by_s, res_by_s),
        "noise_rate": noise_rate(gold_by_s, res_by_s),
        "silent_failure_mass": sfm,
        "silent_failure_count": len(abandoned),
    }


# ── CLI / output ──────────────────────────────────────────────────────────────

PANELS = {
    "sad-code": ["file_p", "file_r", "file_f1", "file_f2", "component_f1",
                 "worst_component_f1", "harmonic_component_f1",
                 "sentence_coverage", "noise_rate"],
    "sad-sam":  ["link_p", "link_r", "link_f1", "link_f2",
                 "sentence_coverage", "noise_rate",
                 "silent_failure_mass", "silent_failure_count"],
}
HEADERS = {
    "file_p": "file_P", "file_r": "file_R", "file_f1": "file_F1", "file_f2": "file_F2",
    "link_p": "link_P", "link_r": "link_R", "link_f1": "link_F1", "link_f2": "link_F2",
    "component_f1": "comp_F1", "worst_component_f1": "worst_C",
    "harmonic_component_f1": "harm_C", "sentence_coverage": "sent_cov",
    "noise_rate": "noise",
    "silent_failure_mass": "SFM%", "silent_failure_count": "SFC",
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
    head = "project".ljust(w) + "".join(HEADERS[c].rjust(10) for c in cols)
    print(head)
    print("-" * len(head))
    for r in rows:
        line = r["project"].ljust(w) + "".join(f"{r[c]:10.4f}" for c in cols)
        print(line)
    if len(rows) > 1:   # an average over a single project is just that project
        avg = average_row(rows, cols)
        print("-" * len(head))
        print(avg["project"].ljust(w) + "".join(f"{avg[c]:10.4f}" for c in cols))


def write_csv(task, rows, path):
    cols = PANELS[task]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["project"] + cols)
        for r in list(rows) + ([average_row(rows, cols)] if len(rows) > 1 else []):
            w.writerow([r["project"]] + [f"{r[c]:.4f}" for c in cols])


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
