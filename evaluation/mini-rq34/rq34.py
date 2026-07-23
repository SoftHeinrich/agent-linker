#!/usr/bin/env python3
"""mini-rq34 — paper RQ3 (validator contribution) + RQ4 (per-module ablation)
metrics, computed from the agent-linker running results.

Self-contained, stdlib-only. Reads the canonical N=3 ``s_linker21`` sweep
(``v2.6.6_s21_gpt`` -> GPT-5.4 / paper main body; ``v2.6.6_s21_sonnet``
-> Claude / appendix mirror), reconstructs each validator's per-link decisions and
each linker's provenance from the ``phase_cache`` pickles (``layer3``,
``layer4``, ``final``), scores every link against the SAD-SAM gold standard, and
writes:

    reports/<backend>/<project>/rq3.csv         (4 variant rows)
    reports/<backend>/<project>/rq3_audit.csv   (2 validator rows)
    reports/<backend>/<project>/rq4.csv         (2 linker rows)
    reports/<backend>/<project>/rq4_upset.csv   (3 overlap-cell rows)
    reports/<backend>/runs_summary.csv          (all 3 runs, canonical marked)
    reports/rq3_validators.csv  reports/rq3_variants.csv   (run-aware aggregates, both backends)
    reports/rq4_linkers.csv     reports/rq4_variants.csv   (run-aware aggregates, both backends)

CSV only — no TeX, no markdown. Top-level aggregates include run1/run2/run3
and an average row; each run sums counts over the 5 projects (RQ4 also averages
its leave-one-out ΔF1).

Method (faithful to alinker-paper working/sections/results.tex):
  * RQ3 measures validator contribution from the full pipeline's *logged
    decisions*, not by re-running with a validator removed. A candidate link is
    a TP if it is in the gold standard, an FP otherwise. Per validator we report
    the TP/FP links it rejects and keeps, and — as the headline cost signal — the
    *unique rejected TP*: true links that validator rejects that the other
    validator does not (the analog of RQ4's unique_tps). The Full / No*Valid /
    NoValidator variant macro-F1 is still emitted as raw F1 (no per-validator
    ΔF1).
  * RQ4 decomposes each linker by *set overlap* (only_E / both / only_C). The
    leave-one-out delta-F1 is also emitted but is the contaminated comparison
    (the surviving linker recovers some removed hits), so overlap is headline.

Conventions (inherited from the mini-* studies):
  * stdlib only; no cross-module imports — the gold loader is inlined and the
    agent-linker dataclasses are *vendored* (see ``_alinker_types.py``), not
    imported from the approach package.
  * Roots derive from this file's location; override via ``$TRANSARC_BENCHMARK``,
    ``$RQ34_CLAUDE_SLOT``, ``$RQ34_OPENAI_SLOT``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.abc
import importlib.machinery
import json
import os
import statistics
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# --------------------------------------------------------------------------- #
# Roots (derived from file location; env-overridable).
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve().parent           # .../transarc-emp/mini-rq34
_ARDOCO_HOME = _HERE.parents[1]                    # .../ardoco-home

BENCHMARK = Path(os.environ.get(
    "TRANSARC_BENCHMARK",
    _ARDOCO_HOME / "ardoco/core/tests-base/src/main/resources/benchmark",
))

PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
RUNS = ["run1", "run2", "run3"]
# The phase_cache subdir name = the linker's _VARIANT_NAME. Defaults to the canonical
# s21 sweep; override via $RQ34_VARIANT to score the prior s_linker20_union canonical.
VARIANT = os.environ.get("RQ34_VARIANT", "s_linker21")

# backend -> (results slot, phase_cache backend subdir, paper role)
# Canonical = s21: GPT-5.4 (openai) = paper main body, Claude = appendix mirror.
SLOTS: Dict[str, Path] = {
    "claude": Path(os.environ.get(
        "RQ34_CLAUDE_SLOT", _ARDOCO_HOME / "agent-linker/results/v2.6.6_s21_sonnet")),
    "openai": Path(os.environ.get(
        "RQ34_OPENAI_SLOT", _ARDOCO_HOME / "agent-linker/results/v2.6.6_s21_gpt")),
}
PCACHE_BACKEND = {"claude": "claude", "openai": "openai"}

GS_SAD_SAM = {
    "mediastore":    "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    "teastore":      "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    "teammates":     "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "bigbluebutton": "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "jabref":        "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
}

LinkKey = Tuple[int, str]  # (sentence_number, component_id)


# --------------------------------------------------------------------------- #
# Unpickling support: register the vendored agent-linker dataclasses under their
# original module path so pickle finds them, with a permissive fallback for any
# other llm_sad_sam.* symbol (only the knowledge layer, which RQ3/RQ4 skip).
# --------------------------------------------------------------------------- #
import _alinker_types  # noqa: E402  (vendored copy; see module docstring)


class _Stub:
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        cls = type(name, (_Stub,), {"__module__": self.__name__})
        setattr(self, name, cls)
        return cls


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = _StubModule(spec.name)
        mod.__path__ = []
        return mod

    def exec_module(self, module):
        pass


class _StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "llm_sad_sam" or fullname.startswith("llm_sad_sam."):
            return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package=True)
        return None


def install_unpickler() -> None:
    # Real, vendored core types take precedence (this is what layer3/4/final use).
    sys.modules.setdefault("llm_sad_sam.core.data_types_v2", _alinker_types)
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _StubFinder())


# --------------------------------------------------------------------------- #
# Gold standard (inlined; SAD-SAM grain).
# --------------------------------------------------------------------------- #
def load_gold(project: str) -> Set[LinkKey]:
    """SAD-SAM gold as {(sentence_number:int, component_id:str)}."""
    gold: Set[LinkKey] = set()
    with (BENCHMARK / GS_SAD_SAM[project]).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gold.add((int(row["sentence"]), row["modelElementID"]))
    return gold


# --------------------------------------------------------------------------- #
# Metric primitives.
# --------------------------------------------------------------------------- #
import pickle  # noqa: E402  (after the unpickler classes are defined)


def prf(pred: Set[LinkKey], gold: Set[LinkKey]) -> Tuple[int, int, int, float]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return tp, fp, fn, f1


def prf3(pred: Set[LinkKey], gold: Set[LinkKey]) -> Tuple[float, float, float]:
    """(precision, recall, f1) for a predicted link set against the gold set."""
    tp, fp, fn, f1 = prf(pred, gold)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, f1


def _key(obj) -> LinkKey:
    return (int(obj.sentence_number), str(obj.component_id))


def _validated_sets(candidates: List, validated: List) -> Tuple[Set[LinkKey], Set[LinkKey]]:
    """(kept, rejected) for one linker. ``kept`` = the validator-approved output
    actually emitted (authoritative ``validated`` list); ``rejected`` = proposed
    candidates the validator rejected."""
    kept = {_key(x) for x in validated}
    rejected = {_key(x) for x in candidates} - kept
    return kept, rejected


# --------------------------------------------------------------------------- #
# Per-(backend, run, project) cell.
# --------------------------------------------------------------------------- #
class Cell:
    def __init__(self, project: str):
        self.project = project
        self.gold: Set[LinkKey] = set()
        self.final: Set[LinkKey] = set()
        self.ent_kept: Set[LinkKey] = set()
        self.ent_rejected: Set[LinkKey] = set()
        self.cor_kept: Set[LinkKey] = set()
        self.cor_rejected: Set[LinkKey] = set()
        self.warnings: List[str] = []


def _phase_dir(slot: Path, run: str, backend: str, project: str) -> Path:
    return slot / run / "phase_cache" / VARIANT / PCACHE_BACKEND[backend] / project


def compute_cell(slot: Path, run: str, backend: str, project: str) -> Cell:
    pdir = _phase_dir(slot, run, backend, project)
    with (pdir / "layer3.pkl").open("rb") as f:
        l3 = pickle.load(f)
    with (pdir / "layer4.pkl").open("rb") as f:
        l4 = pickle.load(f)
    with (pdir / "final.pkl").open("rb") as f:
        fin = pickle.load(f)

    cell = Cell(project)
    cell.gold = load_gold(project)
    cell.final = {_key(x) for x in fin["final"]}
    cell.ent_kept, cell.ent_rejected = _validated_sets(l3["candidates"], l3["validated"])
    cell.cor_kept, cell.cor_rejected = _validated_sets(l4["coref_raw"], l4["coref_validated"])

    union = cell.ent_kept | cell.cor_kept
    if union != cell.final:
        cell.warnings.append(
            f"final({len(cell.final)}) != entity_kept|coref_kept({len(union)}); "
            f"final-only={len(cell.final - union)} union-only={len(union - cell.final)}"
        )
    return cell


# --------------------------------------------------------------------------- #
# Derived rows.
# --------------------------------------------------------------------------- #
def rq3_variant_sets(cell: Cell) -> Dict[str, Set[LinkKey]]:
    return {
        "Full": cell.final,
        "NoEntityValid": cell.final | cell.ent_rejected,
        "NoCitation": cell.final | cell.cor_rejected,
        "NoValidator": cell.final | cell.ent_rejected | cell.cor_rejected,
    }


def rq3_audit(cell: Cell) -> Dict[str, Dict[str, int]]:
    # A candidate link is a TP if it is in the gold standard, an FP otherwise.
    # "rejected" = the validator dropped the link; "kept" = it survived to the output.
    # rejected_tp = true links wrongly dropped (raw); rejected_fp = false links
    # correctly dropped (benefit).
    #
    # unique_rejected_tp = true links this validator drops that are ABSENT from the
    # final output (recovered by neither linker) -- the validator's real recall cost,
    # equal to the single-ablation TP delta (e.g. NoCitation_tp - Full_tp). Earlier this
    # subtracted the OTHER validator's *rejections*, which badly overcounted: many
    # coref-rejected gold links are independently KEPT by the entity linker, so they are
    # still in `final` and were never lost. Subtracting `final` fixes that.
    def a(rejected, kept):
        rejected_tp = rejected & cell.gold
        return {
            "rejected_tp": len(rejected_tp),
            "unique_rejected_tp": len(rejected_tp - cell.final),
            "rejected_fp": len(rejected - cell.gold),
            "kept_tp": len(kept & cell.gold),
            "kept_fp": len(kept - cell.gold),
        }
    return {"entity": a(cell.ent_rejected, cell.ent_kept),
            "coref": a(cell.cor_rejected, cell.cor_kept)}


def rq3_combined_audit(cell: Cell) -> Dict[str, int]:
    """Unique-link audit matching the NoValidator set-union counterfactual."""
    rejected = cell.ent_rejected | cell.cor_rejected
    kept = cell.final
    return {
        "rejected_tp": len(rejected & cell.gold),
        "rejected_fp": len(rejected - cell.gold),
        "kept_tp": len(kept & cell.gold),
        "kept_fp": len(kept - cell.gold),
    }


def rq4_linkers(cell: Cell) -> Dict[str, Dict[str, float]]:
    E, C, G = cell.ent_kept, cell.cor_kept, cell.gold
    _, _, _, f1_full = prf(E | C, G)
    _, _, _, f1_c_only = prf(C, G)  # entity removed
    _, _, _, f1_e_only = prf(E, G)  # coref removed
    return {
        "Entity": {"tps_caught": len(E & G), "unique_tps": len((E & G) - C),
                   "fps": len(E - G), "delta_f1_if_removed": f1_full - f1_c_only},
        "Coref": {"tps_caught": len(C & G), "unique_tps": len((C & G) - E),
                  "fps": len(C - G), "delta_f1_if_removed": f1_full - f1_e_only},
    }


def rq4_upset(cell: Cell) -> Dict[str, int]:
    E, C, G = cell.ent_kept, cell.cor_kept, cell.gold
    return {"only_E": len((E & G) - C), "both": len(E & C & G), "only_C": len((C & G) - E)}


# --------------------------------------------------------------------------- #
# Run selection + I/O.
# --------------------------------------------------------------------------- #
def macro_f1(cells: Dict[str, Cell]) -> float:
    f1s = [prf(cells[p].final, cells[p].gold)[3] for p in PROJECTS if p in cells]
    return statistics.fmean(f1s) if f1s else 0.0


def pick_canonical(per_run: Dict[str, Dict[str, Cell]]) -> str:
    scored = sorted((macro_f1(cells), run) for run, cells in per_run.items())
    return scored[len(scored) // 2][1]


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_ablation_full(slot: Path, run: str, project: str) -> Optional[Dict]:
    files = sorted((slot / run / project).glob("ablation_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8")).get(project, {}).get(VARIANT)


def require_phase_files(slot: Path, run: str, backend: str, project: str) -> None:
    pdir = _phase_dir(slot, run, backend, project)
    missing = [name for name in ("layer3.pkl", "layer4.pkl", "final.pkl")
               if not (pdir / name).exists()]
    if missing:
        raise SystemExit(f"[{backend}] missing required phase cache for {run}/{project}: "
                         f"{', '.join(str(pdir / name) for name in missing)}")


# --------------------------------------------------------------------------- #
# Backend aggregate (over one run's 5 projects).
# --------------------------------------------------------------------------- #
class BackendAgg:
    def __init__(self, backend: str, run: str):
        self.backend = backend
        self.run = run
        self.macro_full = 0.0
        self.macro_no_entity = 0.0
        self.macro_no_coref = 0.0
        self.macro_no_all = 0.0
        self.macro_entity_only = 0.0
        self.macro_coref_only = 0.0
        self.audit = {v: {"rejected_tp": 0, "unique_rejected_tp": 0,
                          "rejected_fp": 0, "kept_tp": 0, "kept_fp": 0}
                      for v in ("entity", "coref")}
        self.combined_audit = {"rejected_tp": 0, "rejected_fp": 0, "kept_tp": 0, "kept_fp": 0}
        self.linkers = {l: {"tps_caught": 0, "unique_tps": 0, "fps": 0, "delta_f1_sum": 0.0, "n": 0}
                        for l in ("Entity", "Coref")}
        self.upset = {"only_E": 0, "both": 0, "only_C": 0}
        # per-project doc-to-model link P/R/F1, per single-linker set: {label: {project: (p, r, f1)}}
        self.dm_pp = {label: {} for label in ("entity_only", "coref_only", "full")}


def mean(vals):
    return statistics.fmean(vals) if vals else 0.0


def average_aggs(backend: str, aggs: List[BackendAgg]) -> BackendAgg:
    avg = BackendAgg(backend, "average")
    avg.macro_full = mean([a.macro_full for a in aggs])
    avg.macro_no_entity = mean([a.macro_no_entity for a in aggs])
    avg.macro_no_coref = mean([a.macro_no_coref for a in aggs])
    avg.macro_no_all = mean([a.macro_no_all for a in aggs])
    avg.macro_entity_only = mean([a.macro_entity_only for a in aggs])
    avg.macro_coref_only = mean([a.macro_coref_only for a in aggs])

    for v in ("entity", "coref"):
        for k in avg.audit[v]:
            avg.audit[v][k] = mean([a.audit[v][k] for a in aggs])
    for k in avg.combined_audit:
        avg.combined_audit[k] = mean([a.combined_audit[k] for a in aggs])
    for l in ("Entity", "Coref"):
        avg.linkers[l]["tps_caught"] = mean([a.linkers[l]["tps_caught"] for a in aggs])
        avg.linkers[l]["unique_tps"] = mean([a.linkers[l]["unique_tps"] for a in aggs])
        avg.linkers[l]["fps"] = mean([a.linkers[l]["fps"] for a in aggs])
        avg.linkers[l]["delta_f1_sum"] = sum(
            a.linkers[l]["delta_f1_sum"] / max(a.linkers[l]["n"], 1) for a in aggs)
        avg.linkers[l]["n"] = len(aggs)
    for c in avg.upset:
        avg.upset[c] = mean([a.upset[c] for a in aggs])
    for label in avg.dm_pp:
        projects = {p for a in aggs for p in a.dm_pp[label]}
        for p in projects:
            triples = [a.dm_pp[label][p] for a in aggs if p in a.dm_pp[label]]
            avg.dm_pp[label][p] = tuple(mean([t[i] for t in triples]) for i in range(3))
    return avg


def fmt_count(v):
    return f"{v:.2f}" if isinstance(v, float) and not v.is_integer() else str(int(v))


def process_backend(backend: str, csv_root: Path, run_override: Optional[str],
                    validate: bool) -> Tuple[List[BackendAgg], str, List[str], int, int, int]:
    slot = SLOTS[backend]
    per_run: Dict[str, Dict[str, Cell]] = {}
    for run in RUNS:
        cells = {}
        for project in PROJECTS:
            require_phase_files(slot, run, backend, project)
            cells[project] = compute_cell(slot, run, backend, project)
        per_run[run] = cells

    canonical = run_override or pick_canonical(per_run)

    # runs_summary.csv
    summary = []
    for run in RUNS:
        if run not in per_run:
            continue
        for project in PROJECTS:
            if project not in per_run[run]:
                continue
            tp, fp, fn, f1 = prf(per_run[run][project].final, per_run[run][project].gold)
            summary.append({"run": run, "project": project, "tp": tp, "fp": fp, "fn": fn,
                            "f1": f"{f1:.6f}", "canonical": "yes" if run == canonical else ""})
        summary.append({"run": run, "project": "MACRO", "tp": "", "fp": "", "fn": "",
                        "f1": f"{macro_f1(per_run[run]):.6f}",
                        "canonical": "yes" if run == canonical else ""})
    _write_csv(csv_root / backend / "runs_summary.csv",
               ["run", "project", "tp", "fp", "fn", "f1", "canonical"], summary)

    warns: List[str] = []
    mismatch = 0
    checked = 0
    skipped = 0

    def aggregate_run(run: str, write_drilldowns: bool) -> BackendAgg:
        nonlocal checked
        agg = BackendAgg(backend, run)
        f1_full_list, f1_ne_list, f1_nc_list, f1_na_list = [], [], [], []
        f1_e_only_list, f1_c_only_list = [], []

        for project in PROJECTS:
            cell = per_run[run][project]
            warns.extend(f"  [{backend}/{run}/{project}] {w}" for w in cell.warnings)

            variants = rq3_variant_sets(cell)
            v_f1 = {}
            rq3_rows = []
            for vname in ("Full", "NoEntityValid", "NoCitation", "NoValidator"):
                tp, fp, fn, f1 = prf(variants[vname], cell.gold)
                v_f1[vname] = f1
                rq3_rows.append({"variant": vname, "project": project, "tp": tp, "fp": fp,
                                 "fn": fn, "f1": f"{f1:.6f}"})

            audit = rq3_audit(cell)
            combined_audit = rq3_combined_audit(cell)
            linkers = rq4_linkers(cell)
            upset = rq4_upset(cell)

            if write_drilldowns:
                base = csv_root / backend / project
                _write_csv(base / "rq3.csv", ["variant", "project", "tp", "fp", "fn", "f1"], rq3_rows)
                _write_csv(base / "rq3_audit.csv",
                           ["validator", "rejected_tp", "unique_rejected_tp",
                            "rejected_fp", "kept_tp", "kept_fp"],
                           [{"validator": v, **audit[v]} for v in ("entity", "coref")])
                _write_csv(base / "rq4.csv",
                           ["linker", "tps_caught", "unique_tps", "fps", "delta_f1_if_removed"],
                           [{"linker": l, "tps_caught": linkers[l]["tps_caught"],
                             "unique_tps": linkers[l]["unique_tps"], "fps": linkers[l]["fps"],
                             "delta_f1_if_removed": f"{linkers[l]['delta_f1_if_removed']:.6f}"}
                            for l in ("Entity", "Coref")])
                _write_csv(base / "rq4_upset.csv", ["cell", "count"],
                           [{"cell": c, "count": upset[c]} for c in ("only_E", "both", "only_C")])

            f1_full_list.append(v_f1["Full"])
            f1_ne_list.append(v_f1["NoEntityValid"])
            f1_nc_list.append(v_f1["NoCitation"])
            f1_na_list.append(v_f1["NoValidator"])
            agg.dm_pp["full"][project] = prf3(cell.final, cell.gold)
            agg.dm_pp["entity_only"][project] = prf3(cell.ent_kept, cell.gold)
            agg.dm_pp["coref_only"][project] = prf3(cell.cor_kept, cell.gold)
            f1e = agg.dm_pp["entity_only"][project][2]
            f1c = agg.dm_pp["coref_only"][project][2]
            f1_e_only_list.append(f1e)
            f1_c_only_list.append(f1c)
            for v in ("entity", "coref"):
                for k in agg.audit[v]:
                    agg.audit[v][k] += audit[v][k]
            for k in agg.combined_audit:
                agg.combined_audit[k] += combined_audit[k]
            for l in ("Entity", "Coref"):
                agg.linkers[l]["tps_caught"] += linkers[l]["tps_caught"]
                agg.linkers[l]["unique_tps"] += linkers[l]["unique_tps"]
                agg.linkers[l]["fps"] += linkers[l]["fps"]
                agg.linkers[l]["delta_f1_sum"] += linkers[l]["delta_f1_if_removed"]
                agg.linkers[l]["n"] += 1
            for c in agg.upset:
                agg.upset[c] += upset[c]

            if validate:
                ref = read_ablation_full(slot, run, project)
                if ref:
                    checked += 1
                    tp, fp, fn, _ = prf(cell.final, cell.gold)
                    if (tp, fp, fn) != (int(ref["tp"]), int(ref["fp"]), int(ref["fn"])):
                        raise SystemExit(f"[{backend}/{run}/{project}] FULL MISMATCH vs "
                                         f"ablation.json: {tp}/{fp}/{fn} != "
                                         f"{ref['tp']}/{ref['fp']}/{ref['fn']}")
                else:
                    raise SystemExit(f"[{backend}/{run}/{project}] missing required "
                                     "ablation_*.json reference")

        agg.macro_full = statistics.fmean(f1_full_list)
        agg.macro_no_entity = statistics.fmean(f1_ne_list)
        agg.macro_no_coref = statistics.fmean(f1_nc_list)
        agg.macro_no_all = statistics.fmean(f1_na_list)
        agg.macro_entity_only = statistics.fmean(f1_e_only_list)
        agg.macro_coref_only = statistics.fmean(f1_c_only_list)
        return agg

    selected_runs = [run_override] if run_override else list(RUNS)
    aggs = [aggregate_run(run, write_drilldowns=(run == canonical)) for run in selected_runs]
    if len(aggs) > 1:
        aggs.append(average_aggs(backend, aggs))
    return aggs, canonical, warns, mismatch, checked, skipped


# --------------------------------------------------------------------------- #
# Aggregated report writers.
# --------------------------------------------------------------------------- #
def write_aggregates(csv_root: Path, aggs: Dict[str, List[BackendAgg]]) -> None:
    # rq3_validators.csv
    rows = []
    for backend, backend_aggs in aggs.items():
        for agg in backend_aggs:
            for v in ("entity", "coref"):
                a = agg.audit[v]
                rows.append({"backend": backend, "run": agg.run, "validator": v,
                             **{k: fmt_count(a[k]) for k in a}})
            # unique_rejected_tp is undefined for the union row (no "other" validator).
            rows.append({"backend": backend, "run": agg.run, "validator": "all_combined",
                         "unique_rejected_tp": "",
                         **{k: fmt_count(agg.combined_audit[k]) for k in agg.combined_audit}})
    _write_csv(csv_root / "rq3_validators.csv",
               ["backend", "run", "validator", "rejected_tp", "unique_rejected_tp",
                "rejected_fp", "kept_tp", "kept_fp"], rows)

    # rq4_linkers.csv
    rows = []
    for backend, backend_aggs in aggs.items():
        for agg in backend_aggs:
            for l in ("Entity", "Coref"):
                e = agg.linkers[l]
                rows.append({"backend": backend, "run": agg.run, "linker": l,
                             "tps_caught": fmt_count(e["tps_caught"]),
                             "unique_tps": fmt_count(e["unique_tps"]),
                             "fps": fmt_count(e["fps"]),
                             "delta_f1_if_removed": f"{e['delta_f1_sum'] / max(e['n'], 1):+.6f}"})
            rows.append({"backend": backend, "run": agg.run, "linker": "overlap(only_E/both/only_C)",
                         "tps_caught": fmt_count(agg.upset["only_E"]),
                         "unique_tps": fmt_count(agg.upset["both"]),
                         "fps": fmt_count(agg.upset["only_C"]), "delta_f1_if_removed": ""})
    _write_csv(csv_root / "rq4_linkers.csv",
               ["backend", "run", "linker", "tps_caught", "unique_tps", "fps",
                "delta_f1_if_removed"], rows)

    # rq3_variants.csv -- macro-F1 per RQ3 variant (the "validator removed" sets).
    rows = []
    for backend, backend_aggs in aggs.items():
        for agg in backend_aggs:
            for variant, macro in (("Full", agg.macro_full),
                                   ("NoEntityValid", agg.macro_no_entity),
                                   ("NoCitation", agg.macro_no_coref),
                                   ("NoValidator", agg.macro_no_all)):
                rows.append({"backend": backend, "run": agg.run, "variant": variant,
                             "macro_f1": f"{macro:.6f}"})
    _write_csv(csv_root / "rq3_variants.csv",
               ["backend", "run", "variant", "macro_f1"], rows)

    # rq4_variants.csv -- single-linker macro-F1 (entity-only / coref-only / full).
    rows = []
    for backend, backend_aggs in aggs.items():
        for agg in backend_aggs:
            for label, macro in (("entity_only", agg.macro_entity_only),
                                 ("coref_only", agg.macro_coref_only),
                                 ("full", agg.macro_full)):
                rows.append({"backend": backend, "run": agg.run, "linker_set": label,
                             "macro_f1": f"{macro:.6f}"})
    _write_csv(csv_root / "rq4_variants.csv",
               ["backend", "run", "linker_set", "macro_f1"], rows)

    # rq4_variants_perproject.csv -- per-project doc-to-model link P/R/F1 backing the
    # single-linker macro-F1 above (entity-only / coref-only / full).
    rows = []
    for backend, backend_aggs in aggs.items():
        for agg in backend_aggs:
            for label in ("entity_only", "coref_only", "full"):
                for project in PROJECTS:
                    if project not in agg.dm_pp[label]:
                        continue
                    p, r, f1 = agg.dm_pp[label][project]
                    rows.append({"backend": backend, "run": agg.run, "linker_set": label,
                                 "project": project, "doc_to_model_link_precision": f"{p:.6f}",
                                 "doc_to_model_link_recall": f"{r:.6f}",
                                 "doc_to_model_link_f1": f"{f1:.6f}"})
    _write_csv(csv_root / "rq4_variants_perproject.csv",
               ["backend", "run", "linker_set", "project", "doc_to_model_link_precision",
                "doc_to_model_link_recall", "doc_to_model_link_f1"], rows)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Compute RQ3/RQ4 paper metrics from running results.")
    ap.add_argument("--csv-root", type=Path, default=_HERE / "reports",
                    help="output root (default: mini-rq34/reports)")
    ap.add_argument("--backends", nargs="+", default=["claude", "openai"],
                    choices=["claude", "openai"])
    ap.add_argument("--run", default=None, choices=RUNS,
                    help="force a run instead of the median-macro run")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip cross-check of Full vs ablation_*.json")
    args = ap.parse_args()

    install_unpickler()
    print(f"[mini-rq34] benchmark = {BENCHMARK}")
    print(f"[mini-rq34] csv-root  = {args.csv_root}")

    aggs: Dict[str, List[BackendAgg]] = {}
    for backend in args.backends:
        backend_aggs, canonical, warns, mismatch, checked, skipped = process_backend(
            backend, args.csv_root, args.run, validate=not args.no_validate)
        aggs[backend] = backend_aggs
        if args.no_validate:
            flag = "skipped (--no-validate)"
        elif mismatch:
            flag = f"{mismatch} MISMATCH ({checked} checked, {skipped} no-ref)"
        else:
            flag = f"OK ({checked} checked, {skipped} no-ref)"
        run_bits = ", ".join(f"{agg.run}={agg.macro_full:.4f}" for agg in backend_aggs)
        print(f"[mini-rq34] {backend}: canonical={canonical} macro-F1[{run_bits}] "
              f"validate={flag}")
        for w in warns:
            print(w)

    write_aggregates(args.csv_root, aggs)
    print(f"[mini-rq34] wrote per-project CSVs + rq3_validators.csv + rq3_variants.csv + "
          f"rq4_linkers.csv + rq4_variants.csv under {args.csv_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
