#!/usr/bin/env python3
"""No-enroll doc-to-code (sad-code) comparison: SOTA baselines vs s20U.

Companion diagnostic to ``mini-src/metrics.py``. It scores sentence->file trace
links **without enrolling** the gold standard's directory ("package") entries,
to expose how much of the benchmark's enrolled file-F1 is an artifact of
package-enrollment (ARDoCo's ``enrollGoldStandard`` / ``metrics.enroll``).

No-enroll scoring (symmetric, atomic-package)
---------------------------------------------
Each gold row is ONE atomic target: a concrete file, or a package ``.../``.
A predicted file is credited to the most-specific gold package it falls under
(predictions collapse to that package, so naming a package once == naming all
its files); predictions under no gold target keep their own identity and count
as false positives. Then a plain set P/R/F1. This is the un-inflated counterpart
to ``metrics.compute_sad_code``'s enrolled ``file_f1`` -- the gap between the two
is the enrollment inflation.

Systems
-------
SOTA doc-code raw links (``sota/recovered-links/doc-code/``, ``sentence_id,target_id``):
  * artemis  (gpt-5.4)        5/5
  * transarc (deterministic)  5/5
  * lissa    (gpt-5-mini)     3/5  -- mediastore, teastore, bigbluebutton
s20U (``s_linker20_union``, claude/sonnet canonical N=3 sweep): emits doc->model
  links only, so its doc->code links are composed TransArc-style by chaining
  through ARDoCo's RECOVERED sam-code (``mini-data/<p>/sam-code/samCodeTlr_*.csv``),
  per run, then aggregated over the 3 runs (default: mean of per-run metrics).

Both no-enroll and enrolled file-F1 are printed so the inflation gap is visible.

Usage
-----
    python3 mini-src/noenroll.py
    python3 mini-src/noenroll.py --lissa-model gpt-4o-mini --s20-agg union
    python3 mini-src/noenroll.py --csv /tmp/noenroll.csv

Reuses the loaders/primitives in ``mini-src/metrics.py`` (the sole metric impl);
this file adds only the no-enroll scoring and the SOTA/s20U result adapters.
Roots override via ``$SOTA_LINKS`` and ``$S20U_SLOT``.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics as m   # noqa: E402  (mini-src/metrics.py — shared loaders/primitives)

# ── Result roots (env-overridable) ────────────────────────────────────────────
SOTA_DOCCODE = Path(os.environ.get(
    "SOTA_LINKS", m._ARDOCO_HOME / "sota/recovered-links")) / "doc-code"
S20U_SLOT = Path(os.environ.get(
    "S20U_SLOT", m._ARDOCO_HOME / "agent-linker/results/v2.6.5_s20union_sonnet"))
S20U_RUNS = ["run1", "run2", "run3"]


# ── No-enroll scoring ─────────────────────────────────────────────────────────

def noenroll_prf(gold_raw, res):
    """No-enroll (precision, recall, f1) for one doc-code result set.

    gold_raw : set[(sentence, path)] BEFORE enrollment — a path ending in '/'
               is a package (one atomic target), else a concrete file.
    res      : set[(sentence, file)] concrete predictions.

    Each predicted file collapses to the most-specific gold package it falls
    under (else keeps its own path); then a set P/R/F1 over atomic targets.
    """
    dirs_by_s = defaultdict(list)
    gold = set()
    for s, p in gold_raw:
        gold.add((s, p))
        if p.endswith("/"):
            dirs_by_s[s].append(p)
    collapsed = set()
    for s, f in res:
        cands = [d for d in dirs_by_s.get(s, ()) if f.startswith(d)]
        collapsed.add((s, max(cands, key=len) if cands else f))
    return m.prf(gold, collapsed)


# ── Result adapters ───────────────────────────────────────────────────────────

def load_sota_doccode(approach, project, model=None):
    """SOTA doc-code links -> set[(sentence, normalized_path)], or None if absent."""
    name = f"{approach}-{project}" + (f"-{model}" if model else "") + ".csv"
    path = SOTA_DOCCODE / name
    if not path.exists():
        return None
    res = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            s = (r.get("sentence_id") or "").strip()
            t = r.get("target_id") or ""
            if s and t.strip():
                res.add((s, m.normalize_path(t.strip())))
    return res


def load_recovered_sam_code(project):
    """ARDoCo recovered sam-code -> {model_element_id: {normalized_file}}.

    NB: samCodeTlr_*.csv's first column is *labelled* ``sentenceID`` but actually
    holds the MODEL-ELEMENT id; the second column ``codeID`` is the file path.
    """
    by_comp = defaultdict(set)
    path = m.DEFAULT_RESULTS / project / "sam-code" / f"samCodeTlr_{project}.csv"
    with open(path) as f:
        for r in csv.DictReader(f):
            comp = (r.get("sentenceID") or r.get("modelElementID") or "").strip()
            code = r.get("codeID") or ""
            if comp and code.strip():
                by_comp[comp].add(m.normalize_path(code.strip()))
    return by_comp


def load_s20_doc_sam(project, run):
    """s20U doc->model links for one run -> set[(sentence, component_id)]."""
    path = S20U_SLOT / run / project / f"s_linker20_union_{project}_links.csv"
    links = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            s = (r.get("sentence") or "").strip()
            c = (r.get("component_id") or "").strip()
            if s and c:
                links.add((s, c))
    return links


def compose_s20_doc_code(project, run):
    """Chain s20U doc->model through recovered sam-code -> set[(sentence, file)]."""
    doc_sam = load_s20_doc_sam(project, run)
    sam_code = load_recovered_sam_code(project)
    return {(s, cf) for s, comp in doc_sam for cf in sam_code.get(comp, ())}


# ── Per-system scoring ────────────────────────────────────────────────────────

def score(gold_raw, code_files, res):
    """Return (noenroll_p, noenroll_r, noenroll_f1, enrolled_f1) for one result."""
    nep, ner, nef = noenroll_prf(gold_raw, res)
    enrolled_f1 = m.prf(m.enroll(gold_raw, code_files), res)[2]
    return nep, ner, nef, enrolled_f1


def score_s20(gold_raw, code_files, project, agg):
    """s20U over the 3 runs. agg='mean' averages per-run scores; 'union' scores
    the union of composed links once."""
    if agg == "union":
        res = set().union(*(compose_s20_doc_code(project, r) for r in S20U_RUNS))
        return score(gold_raw, code_files, res)
    per = [score(gold_raw, code_files, compose_s20_doc_code(project, r))
           for r in S20U_RUNS]
    return tuple(sum(col) / len(per) for col in zip(*per))


# ── CLI / output ──────────────────────────────────────────────────────────────

def macro(per_project):
    """Mean of each column over the projects a system covers."""
    vals = [v for v in per_project.values() if v is not None]
    if not vals:
        return None
    return tuple(sum(col) / len(vals) for col in zip(*vals))


def build_systems(lissa_model, s20_agg):
    """system_label -> callable(project, gold_raw, code_files) -> tuple|None."""
    return [
        ("artemis-gpt5.4", lambda p, g, c: _sota("artemis", p, g, c, "gpt-5.4")),
        ("transarc",       lambda p, g, c: _sota("transarc", p, g, c, None)),
        (f"lissa-{lissa_model}",
                           lambda p, g, c: _sota("lissa", p, g, c, lissa_model)),
        (f"s20U({s20_agg}3)",
                           lambda p, g, c: score_s20(g, c, p, s20_agg)),
    ]


def _sota(approach, project, gold_raw, code_files, model):
    res = load_sota_doccode(approach, project, model)
    return None if res is None else score(gold_raw, code_files, res)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lissa-model", default="gpt-5-mini",
                    choices=["gpt-5-mini", "gpt-4o-mini"],
                    help="LiSSA model variant (default pairs with Artemis gpt-5.4)")
    ap.add_argument("--s20-agg", default="mean", choices=["mean", "union"],
                    help="aggregate s20U's 3 runs by mean-of-metrics or link union")
    ap.add_argument("--csv", default=None, help="also write the macro panel to CSV")
    args = ap.parse_args()

    # gold (raw, pre-enrollment) + code-model files, per project.
    gold = {p: m.load_gs_sad_code_raw(p) for p in m.PROJECTS}
    cfiles = {p: m.load_code_model_files(p) for p in m.PROJECTS}

    systems = build_systems(args.lissa_model, args.s20_agg)
    # system -> {project: (neP, neR, neF1, enrF1) | None}
    table = {label: {p: fn(p, gold[p], cfiles[p]) for p in m.PROJECTS}
             for label, fn in systems}

    abbr = {"mediastore": "MS", "teastore": "TS", "teammates": "TM",
            "bigbluebutton": "BBB", "jabref": "JR"}

    print("NO-ENROLL doc-to-code (sad-code) — macro over covered projects")
    print(f"{'system':18}{'n':>3} │ {'ne_P':>7}{'ne_R':>7}{'ne_F1':>7} │"
          f" {'enr_F1':>7}{'Δinfl':>7}")
    print("-" * 64)
    macros = {}
    for label, per in table.items():
        mc = macro(per)
        macros[label] = mc
        n = sum(1 for v in per.values() if v is not None)
        if mc is None:
            print(f"{label:18}{n:>3} │  (no results)")
            continue
        neP, neR, neF, enrF = mc
        print(f"{label:18}{n:>3} │ {neP:7.3f}{neR:7.3f}{neF:7.3f} │"
              f" {enrF:7.3f}{enrF - neF:+7.3f}")

    print(f"\nPer-project no-enroll F1   ({'  '.join(abbr[p] for p in m.PROJECTS)})")
    print("-" * 64)
    for label, per in table.items():
        cells = []
        for p in m.PROJECTS:
            v = per[p]
            cells.append(f"{v[2]:5.3f}" if v is not None else "  -  ")
        print(f"{label:18} " + "  ".join(cells))

    print(f"\nProvenance: SOTA={SOTA_DOCCODE}  s20U={S20U_SLOT} (runs {','.join(S20U_RUNS)})")
    print("No-enroll = each gold package is one atomic target; a predicted file "
          "under it\n  satisfies it once. Δinfl = enrolled_F1 − no-enroll_F1 "
          "(enrollment inflation).")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["system", "n_projects", "noenroll_P", "noenroll_R",
                        "noenroll_F1", "enrolled_F1", "inflation_gap"])
            for label, per in table.items():
                mc = macros[label]
                if mc is None:
                    continue
                n = sum(1 for v in per.values() if v is not None)
                neP, neR, neF, enrF = mc
                w.writerow([label, n, f"{neP:.4f}", f"{neR:.4f}", f"{neF:.4f}",
                            f"{enrF:.4f}", f"{enrF - neF:.4f}"])
        print(f"\n[noenroll] wrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
