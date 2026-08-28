#!/usr/bin/env python3
"""mini-rq34 — paper RQ3 (validator contribution) + RQ4 (per-module ablation)
metrics, computed from the agent-linker running results.

Self-contained, stdlib-only. Reads the canonical N=3 ``s_linker92a`` sweep
(``$RQ34_ARM=s92``, the default: terra -> paper body, luna -> mirror; set
``RQ34_ARM=s21`` for the retired GPT-5.4/Claude arm), reconstructs each judge's
per-link decisions and each linker's provenance from the run's phase state (see
the arm layout below), scores every link against the SAD-SAM gold standard, and
writes -- with one row per phase, so the row counts follow ``PHASES`` (3 on s92,
2 on s21):

    reports/<backend>/<project>/rq3.csv         (Full + one per judge + NoValidator)
    reports/<backend>/<project>/rq3_audit.csv   (one row per judge)
    reports/<backend>/<project>/rq4.csv         (one row per linker)
    reports/<backend>/<project>/rq4_upset.csv   (one row per linker + shared)
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
  * stdlib only; the benchmark layout, the gold loader and the F-measures come
    from the tree's shared core (``mini-src/metrics.py``) so RQ3/RQ4 score with
    the same arithmetic as RQ1/RQ2. The agent-linker dataclasses are still
    *vendored* (see ``_alinker_types.py``), never imported from the approach
    package.
  * Roots derive from this file's location; override via ``$TRANSARC_BENCHMARK``,
    ``$RQ34_CLAUDE_SLOT``, ``$RQ34_OPENAI_SLOT``.
"""

from __future__ import annotations

import argparse
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
_HERE = Path(__file__).resolve().parent           # .../evaluation/mini-rq34
_ARDOCO_HOME = _HERE.parents[1]                    # .../ardoco-home
sys.path.insert(0, str(_HERE.parent / "mini-src"))
import metrics as m  # noqa: E402  (shared core: benchmark layout, gold, F-measures)

# Re-exported under this module's names: rq34_rq2.py and rq_tables.py read them
# from here, and $TRANSARC_BENCHMARK still selects the benchmark root.
BENCHMARK = m.BENCHMARK
PROJECTS = m.PROJECTS
fbeta = m.fbeta                                   # recall-weighted F-beta (\ftwo)
RUNS = ["run1", "run2", "run3"]
# The phase subdir name = the linker's _VARIANT_NAME. Defaults to the arm the paper
# reports (s_linker92a); override via $RQ34_VARIANT.
DEFAULT_VARIANT = {"s21": "s_linker21", "s92": "s_linker92a"}

# ── Arm layout ───────────────────────────────────────────────────────────────
# A "phase" is one linker plus the judge that filters its candidates. The two arms
# record the same information in different shapes:
#
#   s21  two phases in ``<slot>/<run>/phase_cache/<variant>/<backend>/<project>/``:
#        layer3.pkl {candidates, validated} = the named-mention (entity) linker+judge,
#        layer4.pkl {coref_raw, coref_validated} = the coreference linker+judge.
#   s92  three phases in ``<rundir>/phase_states/<variant>/openai/<project>/``:
#        linker_{full_name,partial_name,coreference}.pkl, each {links, feedback}, where
#        ``links`` is the judged-and-kept output and ``feedback["judge_decisions"]``
#        carries one record per judged candidate with an ``approved`` flag. The s25
#        lineage split the s21 "entity" linker into full-name and partial-name, so this
#        arm has three judges, not two.
#
# Every downstream row is keyed by phase name, so the third phase adds a row/column
# instead of a special case. $RQ34_ARM selects the layout.
ARM = os.environ.get("RQ34_ARM", "s92")
VARIANT = os.environ.get("RQ34_VARIANT", DEFAULT_VARIANT[ARM])

PHASE_SETS = {
    "s21": [
        {"key": "entity", "linker": "Entity", "file": "layer3.pkl",
         "cand": "candidates", "kept": "validated", "variant": "NoEntityValid"},
        {"key": "coref", "linker": "Coref", "file": "layer4.pkl",
         "cand": "coref_raw", "kept": "coref_validated", "variant": "NoCitation"},
    ],
    "s92": [
        {"key": "full_name", "linker": "FullName", "file": "linker_full_name.pkl",
         "variant": "NoFullNameValid"},
        {"key": "partial_name", "linker": "PartialName", "file": "linker_partial_name.pkl",
         "variant": "NoPartialNameValid"},
        {"key": "coref", "linker": "Coref", "file": "linker_coreference.pkl",
         "variant": "NoCitation"},
    ],
}
PHASES = PHASE_SETS[ARM]
PHASE_KEYS = [ph["key"] for ph in PHASES]
LINKERS = [ph["linker"] for ph in PHASES]
KEY_OF_LINKER = {ph["linker"]: ph["key"] for ph in PHASES}

# backend -> results slot (s21) or the ordered per-run directories (s92).
# s21: GPT-5.4 (openai) = paper main body, Claude = appendix mirror.
# s92: terra = paper main body, luna = the mirror backend.
PCACHE_BACKEND = {"claude": "claude", "openai": "openai"}

# agent-linker's results root. Sibling layout puts it at <ardoco-home>/agent-linker/results;
# the nested layout (evaluation/ inside agent-linker) puts it at <ardoco-home>/results.
_RESULTS = Path(os.environ.get(
    "ALINKER_RESULTS",
    _ARDOCO_HOME / "results" if (_ARDOCO_HOME / "results").is_dir()
    else _ARDOCO_HOME / "agent-linker/results"))

# The s92 arm keeps each run in its own top-level directory rather than <slot>/run<i>/.
# $RQ34_S92_DIR_TMPL selects WHICH sweep to score -- the Full arm by default, or the
# no-knowledge sweep (RQ4's knowledge A/B) with
#   RQ34_S92_DIR_TMPL='regex_noknow_e2e_{model}_r{i}_20260826' RQ34_ABLATION_KEY=s_linker92a_noknow
# Note the split naming: the phase-state directory carries _VARIANT_NAME (s_linker92a for
# both arms), while the links CSV and the ablation JSON key carry the registry variant.
S92_DIR_TMPL = os.environ.get("RQ34_S92_DIR_TMPL", "regex_e2e_{model}_r{i}_20260822")
S92_RUN_DIRS: Dict[str, Dict[str, Path]] = {
    model: {run: _RESULTS / S92_DIR_TMPL.format(model=model, i=i)
            for i, run in enumerate(RUNS, 1)}
    for model in ("terra", "luna")
}
# Key under which the run's ablation JSON records the Full result (the tp/fp/fn oracle).
ABLATION_KEY = os.environ.get("RQ34_ABLATION_KEY", VARIANT)
BACKENDS = {"s21": ["claude", "openai"], "s92": ["terra", "luna"]}[ARM]

SLOTS: Dict[str, Path] = {
    "claude": Path(os.environ.get("RQ34_CLAUDE_SLOT", _RESULTS / "v2.6.6_s21_sonnet")),
    "openai": Path(os.environ.get("RQ34_OPENAI_SLOT", _RESULTS / "v2.6.6_s21_gpt")),
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
# Gold standard (shared reader; SAD-SAM grain).
# --------------------------------------------------------------------------- #
def load_gold(project: str) -> Set[LinkKey]:
    """SAD-SAM gold as {(sentence_number:int, component_id:str)}.

    ``metrics.load_gs_sad_sam`` reads the file; RQ3/RQ4 key their links the other
    way round and numerically (the phase caches carry int sentence numbers), so
    this only re-shapes its pairs.
    """
    return {(int(sentence), component)
            for component, sentence in m.load_gs_sad_sam(project)}


# --------------------------------------------------------------------------- #
# Metric primitives.
# --------------------------------------------------------------------------- #
import pickle  # noqa: E402  (after the unpickler classes are defined)


def prf(pred: Set[LinkKey], gold: Set[LinkKey]) -> Tuple[int, int, int, float]:
    """(tp, fp, fn, f1) -- the count-carrying view of ``metrics.prf_counts``.

    Mind the argument order: RQ3/RQ4 pass the prediction first, the shared core
    takes the gold first.
    """
    tp, fp, fn, _p, _r, f1 = m.prf_counts(gold, pred)
    return tp, fp, fn, f1


def prf3(pred: Set[LinkKey], gold: Set[LinkKey]) -> Tuple[float, float, float, float]:
    """(precision, recall, f1, f2) for a predicted link set against the gold set."""
    _tp, _fp, _fn, p, r, f1 = m.prf_counts(gold, pred)
    return p, r, f1, m.fbeta(p, r)


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
    """One (backend, run, project) cell: the gold, the pipeline output, and each
    phase's kept/rejected sets keyed by phase name (see PHASES)."""

    def __init__(self, project: str):
        self.project = project
        self.gold: Set[LinkKey] = set()
        self.final: Set[LinkKey] = set()
        self.kept: Dict[str, Set[LinkKey]] = {k: set() for k in PHASE_KEYS}
        self.rejected: Dict[str, Set[LinkKey]] = {k: set() for k in PHASE_KEYS}
        self.warnings: List[str] = []

    def others(self, key: str) -> Set[LinkKey]:
        """Everything the OTHER phases kept -- the baseline for `unique to this phase`."""
        return set().union(*(self.kept[k] for k in PHASE_KEYS if k != key)) \
            if len(PHASE_KEYS) > 1 else set()


def _phase_dir(slot: Path, run: str, backend: str, project: str) -> Path:
    if ARM == "s92":
        return S92_RUN_DIRS[backend][run] / "phase_states" / VARIANT / "openai" / project
    return slot / run / "phase_cache" / VARIANT / PCACHE_BACKEND[backend] / project


def _judged_sets(state: Dict) -> Tuple[Set[LinkKey], Set[LinkKey]]:
    """(kept, rejected) for one s92 linker phase.

    ``links`` is authoritative for kept -- it is what the phase emitted downstream.
    ``rejected`` comes from the judge's own decision log: every judged candidate
    marked ``approved=False``. Candidates carry a component *name* only, so the
    decision log (which carries ``component_id``) is the only record at link grain;
    a candidate that never reached a judge is therefore not counted as rejected.
    """
    kept = {_key(x) for x in state["links"]}
    decisions = state.get("feedback", {}).get("judge_decisions", []) or []
    rejected = {(int(d["sentence"]), str(d["component_id"]))
                for d in decisions if not d.get("approved")} - kept
    return kept, rejected


def compute_cell(slot: Path, run: str, backend: str, project: str) -> Cell:
    pdir = _phase_dir(slot, run, backend, project)
    cell = Cell(project)
    cell.gold = load_gold(project)
    with (pdir / "final.pkl").open("rb") as f:
        cell.final = {_key(x) for x in pickle.load(f)["final"]}

    for ph in PHASES:
        with (pdir / ph["file"]).open("rb") as f:
            state = pickle.load(f)
        if ARM == "s92":
            kept, rejected = _judged_sets(state)
        else:
            kept, rejected = _validated_sets(state[ph["cand"]], state[ph["kept"]])
        cell.kept[ph["key"]], cell.rejected[ph["key"]] = kept, rejected

    union = set().union(*cell.kept.values())
    if union != cell.final:
        cell.warnings.append(
            f"final({len(cell.final)}) != union of phase-kept({len(union)}); "
            f"final-only={len(cell.final - union)} union-only={len(union - cell.final)}"
        )
    return cell


# --------------------------------------------------------------------------- #
# Derived rows.
# --------------------------------------------------------------------------- #
def rq3_variant_sets(cell: Cell) -> Dict[str, Set[LinkKey]]:
    """Full plus one counterfactual per judge (its rejections put back), plus all."""
    sets = {"Full": cell.final}
    for ph in PHASES:
        sets[ph["variant"]] = cell.final | cell.rejected[ph["key"]]
    sets["NoValidator"] = cell.final | set().union(*cell.rejected.values())
    return sets


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
    return {k: a(cell.rejected[k], cell.kept[k]) for k in PHASE_KEYS}


def rq3_combined_audit(cell: Cell) -> Dict[str, int]:
    """Unique-link audit matching the NoValidator set-union counterfactual."""
    rejected = set().union(*cell.rejected.values())
    kept = cell.final
    return {
        "rejected_tp": len(rejected & cell.gold),
        # The outright cost of the whole stack: true links it rejected that NO linker
        # recovered. Strictly below the sum of the per-judge uniques, because two judges
        # can reject the same link.
        "unique_rejected_tp": len((rejected & cell.gold) - cell.final),
        "rejected_fp": len(rejected - cell.gold),
        "kept_tp": len(kept & cell.gold),
        "kept_fp": len(kept - cell.gold),
    }


def rq4_linkers(cell: Cell) -> Dict[str, Dict[str, float]]:
    """Per linker: TPs caught, TPs no other linker catches, FPs, and the F1 the
    pipeline loses if this linker is removed (the surviving linkers keep their hits)."""
    G = cell.gold
    _, _, _, f1_full = prf(set().union(*cell.kept.values()), G)
    out = {}
    for ph in PHASES:
        key, mine, rest = ph["key"], cell.kept[ph["key"]], cell.others(ph["key"])
        _, _, _, f1_without = prf(rest, G)
        out[ph["linker"]] = {"tps_caught": len(mine & G),
                             "unique_tps": len((mine & G) - rest),
                             "fps": len(mine - G),
                             "delta_f1_if_removed": f1_full - f1_without}
    return out


def rq4_upset(cell: Cell) -> Dict[str, int]:
    """TP overlap: exclusive TPs per linker plus the TPs at least two of them share."""
    G = cell.gold
    out = {f"only_{ph['key']}": len((cell.kept[ph["key"]] & G) - cell.others(ph["key"]))
           for ph in PHASES}
    caught = [cell.kept[k] & G for k in PHASE_KEYS]
    shared = {link for link in set().union(*caught)
              if sum(link in c for c in caught) > 1}
    out["shared"] = len(shared)
    return out


# --------------------------------------------------------------------------- #
# Run selection + I/O.
# --------------------------------------------------------------------------- #
def macro_f1(cells: Dict[str, Cell]) -> float:
    f1s = [prf(cells[p].final, cells[p].gold)[3] for p in PROJECTS if p in cells]
    return statistics.fmean(f1s) if f1s else 0.0


def pick_canonical(per_run: Dict[str, Dict[str, Cell]]) -> str:
    scored = sorted((macro_f1(cells), run) for run, cells in per_run.items())
    return scored[len(scored) // 2][1]


_write_csv = m.write_dict_csv   # the tree's one dict-row CSV writer


def read_ablation_full(slot: Path, run: str, project: str, backend: str = "") -> Optional[Dict]:
    """The run's own ablation JSON row for this (project, variant) -- the tp/fp/fn oracle.

    s21 writes one JSON per (run, project); s92 writes one per run directory, keyed by
    project. Both carry ``{project: {variant: {tp, fp, fn, ...}}}``."""
    root = S92_RUN_DIRS[backend][run] if ARM == "s92" else slot / run / project
    files = sorted(root.glob("ablation_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8")).get(project, {}).get(ABLATION_KEY)


def require_phase_files(slot: Path, run: str, backend: str, project: str) -> None:
    pdir = _phase_dir(slot, run, backend, project)
    missing = [name for name in [ph["file"] for ph in PHASES] + ["final.pkl"]
               if not (pdir / name).exists()]
    if missing:
        raise SystemExit(f"[{backend}] missing required phase cache for {run}/{project}: "
                         f"{', '.join(str(pdir / name) for name in missing)}")


# --------------------------------------------------------------------------- #
# Backend aggregate (over one run's 5 projects).
# --------------------------------------------------------------------------- #
# RQ3 variant names, in display order, and the single-linker RQ4 set labels.
RQ3_VARIANTS = ["Full"] + [ph["variant"] for ph in PHASES] + ["NoValidator"]
RQ4_SET_LABELS = [f"{ph['key']}_only" for ph in PHASES] + ["full"]
UPSET_CELLS = [f"only_{ph['key']}" for ph in PHASES] + ["shared"]


class BackendAgg:
    def __init__(self, backend: str, run: str):
        self.backend = backend
        self.run = run
        # macro F1 and F2 per RQ3 variant, and per RQ4 single-linker set (both phase-keyed).
        self.macro = {v: 0.0 for v in RQ3_VARIANTS}
        self.macro_f2 = {v: 0.0 for v in RQ3_VARIANTS}
        self.macro_only = {label: 0.0 for label in RQ4_SET_LABELS}
        self.macro_only_f2 = {label: 0.0 for label in RQ4_SET_LABELS}
        self.audit = {k: {"rejected_tp": 0, "unique_rejected_tp": 0,
                          "rejected_fp": 0, "kept_tp": 0, "kept_fp": 0}
                      for k in PHASE_KEYS}
        self.combined_audit = {"rejected_tp": 0, "unique_rejected_tp": 0,
                               "rejected_fp": 0, "kept_tp": 0, "kept_fp": 0}
        self.linkers = {l: {"tps_caught": 0, "unique_tps": 0, "fps": 0, "delta_f1_sum": 0.0, "n": 0}
                        for l in LINKERS}
        self.upset = {c: 0 for c in UPSET_CELLS}
        # per-project doc-to-model link P/R/F1, per single-linker set: {label: {project: (p, r, f1)}}
        self.dm_pp = {label: {} for label in RQ4_SET_LABELS}

    @property
    def macro_full(self) -> float:
        return self.macro["Full"]


def mean(vals):
    return statistics.fmean(vals) if vals else 0.0


def average_aggs(backend: str, aggs: List[BackendAgg]) -> BackendAgg:
    avg = BackendAgg(backend, "average")
    for v in RQ3_VARIANTS:
        avg.macro[v] = mean([a.macro[v] for a in aggs])
        avg.macro_f2[v] = mean([a.macro_f2[v] for a in aggs])
    for label in RQ4_SET_LABELS:
        avg.macro_only[label] = mean([a.macro_only[label] for a in aggs])
        avg.macro_only_f2[label] = mean([a.macro_only_f2[label] for a in aggs])

    for v in PHASE_KEYS:
        for k in avg.audit[v]:
            avg.audit[v][k] = mean([a.audit[v][k] for a in aggs])
    for k in avg.combined_audit:
        avg.combined_audit[k] = mean([a.combined_audit[k] for a in aggs])
    for l in LINKERS:
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
            vecs = [a.dm_pp[label][p] for a in aggs if p in a.dm_pp[label]]
            avg.dm_pp[label][p] = tuple(mean([v[i] for v in vecs]) for i in range(4))
    return avg


def fmt_count(v):
    return f"{v:.2f}" if isinstance(v, float) and not v.is_integer() else str(int(v))


def process_backend(backend: str, csv_root: Path, run_override: Optional[str],
                    validate: bool) -> Tuple[List[BackendAgg], str, List[str], int, int, int]:
    slot = SLOTS.get(backend, Path("/nonexistent"))   # s92 resolves per-run dirs instead
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
            cell = per_run[run][project]
            tp, fp, fn, f1 = prf(cell.final, cell.gold)
            _, _, _, f2 = prf3(cell.final, cell.gold)
            summary.append({"run": run, "project": project, "tp": tp, "fp": fp, "fn": fn,
                            "f1": f"{f1:.6f}", "f2": f"{f2:.6f}",
                            "canonical": "yes" if run == canonical else ""})
        macro_f2 = mean([prf3(per_run[run][pj].final, per_run[run][pj].gold)[3]
                         for pj in PROJECTS if pj in per_run[run]])
        summary.append({"run": run, "project": "MACRO", "tp": "", "fp": "", "fn": "",
                        "f1": f"{macro_f1(per_run[run]):.6f}", "f2": f"{macro_f2:.6f}",
                        "canonical": "yes" if run == canonical else ""})
    _write_csv(csv_root / backend / "runs_summary.csv",
               ["run", "project", "tp", "fp", "fn", "f1", "f2", "canonical"], summary)

    warns: List[str] = []
    mismatch = 0
    checked = 0
    skipped = 0

    def aggregate_run(run: str, write_drilldowns: bool) -> BackendAgg:
        nonlocal checked
        agg = BackendAgg(backend, run)
        variant_f1: Dict[str, List[float]] = {v: [] for v in RQ3_VARIANTS}
        variant_f2: Dict[str, List[float]] = {v: [] for v in RQ3_VARIANTS}
        only_f1: Dict[str, List[float]] = {label: [] for label in RQ4_SET_LABELS}
        only_f2: Dict[str, List[float]] = {label: [] for label in RQ4_SET_LABELS}

        for project in PROJECTS:
            cell = per_run[run][project]
            warns.extend(f"  [{backend}/{run}/{project}] {w}" for w in cell.warnings)

            variants = rq3_variant_sets(cell)
            v_f1 = {}
            rq3_rows = []
            v_f2 = {}
            for vname in RQ3_VARIANTS:
                tp, fp, fn, f1 = prf(variants[vname], cell.gold)
                f2 = prf3(variants[vname], cell.gold)[3]
                v_f1[vname], v_f2[vname] = f1, f2
                rq3_rows.append({"variant": vname, "project": project, "tp": tp, "fp": fp,
                                 "fn": fn, "f1": f"{f1:.6f}", "f2": f"{f2:.6f}"})

            audit = rq3_audit(cell)
            combined_audit = rq3_combined_audit(cell)
            linkers = rq4_linkers(cell)
            upset = rq4_upset(cell)

            if write_drilldowns:
                base = csv_root / backend / project
                _write_csv(base / "rq3.csv",
                           ["variant", "project", "tp", "fp", "fn", "f1", "f2"], rq3_rows)
                _write_csv(base / "rq3_audit.csv",
                           ["validator", "rejected_tp", "unique_rejected_tp",
                            "rejected_fp", "kept_tp", "kept_fp"],
                           [{"validator": v, **audit[v]} for v in PHASE_KEYS])
                _write_csv(base / "rq4.csv",
                           ["linker", "tps_caught", "unique_tps", "fps", "delta_f1_if_removed"],
                           [{"linker": l, "tps_caught": linkers[l]["tps_caught"],
                             "unique_tps": linkers[l]["unique_tps"], "fps": linkers[l]["fps"],
                             "delta_f1_if_removed": f"{linkers[l]['delta_f1_if_removed']:.6f}"}
                            for l in LINKERS])
                _write_csv(base / "rq4_upset.csv", ["cell", "count"],
                           [{"cell": c, "count": upset[c]} for c in UPSET_CELLS])

            for vname in RQ3_VARIANTS:
                variant_f1[vname].append(v_f1[vname])
                variant_f2[vname].append(v_f2[vname])
            agg.dm_pp["full"][project] = prf3(cell.final, cell.gold)
            only_f1["full"].append(agg.dm_pp["full"][project][2])
            only_f2["full"].append(agg.dm_pp["full"][project][3])
            for ph in PHASES:
                label = f"{ph['key']}_only"
                agg.dm_pp[label][project] = prf3(cell.kept[ph["key"]], cell.gold)
                only_f1[label].append(agg.dm_pp[label][project][2])
                only_f2[label].append(agg.dm_pp[label][project][3])
            for v in PHASE_KEYS:
                for k in agg.audit[v]:
                    agg.audit[v][k] += audit[v][k]
            for k in agg.combined_audit:
                agg.combined_audit[k] += combined_audit[k]
            for l in LINKERS:
                agg.linkers[l]["tps_caught"] += linkers[l]["tps_caught"]
                agg.linkers[l]["unique_tps"] += linkers[l]["unique_tps"]
                agg.linkers[l]["fps"] += linkers[l]["fps"]
                agg.linkers[l]["delta_f1_sum"] += linkers[l]["delta_f1_if_removed"]
                agg.linkers[l]["n"] += 1
            for c in agg.upset:
                agg.upset[c] += upset[c]

            if validate:
                ref = read_ablation_full(slot, run, project, backend)
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

        for vname in RQ3_VARIANTS:
            agg.macro[vname] = statistics.fmean(variant_f1[vname])
            agg.macro_f2[vname] = statistics.fmean(variant_f2[vname])
        for label in RQ4_SET_LABELS:
            agg.macro_only[label] = statistics.fmean(only_f1[label])
            agg.macro_only_f2[label] = statistics.fmean(only_f2[label])
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
            for v in PHASE_KEYS:
                a = agg.audit[v]
                rows.append({"backend": backend, "run": agg.run, "validator": v,
                             **{k: fmt_count(a[k]) for k in a}})
            rows.append({"backend": backend, "run": agg.run, "validator": "all_combined",
                         **{k: fmt_count(agg.combined_audit[k]) for k in agg.combined_audit}})
    _write_csv(csv_root / "rq3_validators.csv",
               ["backend", "run", "validator", "rejected_tp", "unique_rejected_tp",
                "rejected_fp", "kept_tp", "kept_fp"], rows)

    # rq4_linkers.csv
    rows = []
    for backend, backend_aggs in aggs.items():
        for agg in backend_aggs:
            for l in LINKERS:
                e = agg.linkers[l]
                rows.append({"backend": backend, "run": agg.run, "linker": l,
                             "tps_caught": fmt_count(e["tps_caught"]),
                             "unique_tps": fmt_count(e["unique_tps"]),
                             "fps": fmt_count(e["fps"]),
                             "delta_f1_if_removed": f"{e['delta_f1_sum'] / max(e['n'], 1):+.6f}"})
            for c in UPSET_CELLS:
                rows.append({"backend": backend, "run": agg.run, "linker": f"overlap:{c}",
                             "tps_caught": fmt_count(agg.upset[c]), "unique_tps": "",
                             "fps": "", "delta_f1_if_removed": ""})
    _write_csv(csv_root / "rq4_linkers.csv",
               ["backend", "run", "linker", "tps_caught", "unique_tps", "fps",
                "delta_f1_if_removed"], rows)

    # rq3_variants.csv -- macro-F1 per RQ3 variant (the "validator removed" sets).
    rows = []
    for backend, backend_aggs in aggs.items():
        for agg in backend_aggs:
            for variant in RQ3_VARIANTS:
                rows.append({"backend": backend, "run": agg.run, "variant": variant,
                             "macro_f1": f"{agg.macro[variant]:.6f}",
                             "macro_f2": f"{agg.macro_f2[variant]:.6f}"})
    _write_csv(csv_root / "rq3_variants.csv",
               ["backend", "run", "variant", "macro_f1", "macro_f2"], rows)

    # rq4_variants.csv -- single-linker macro-F1 (entity-only / coref-only / full).
    rows = []
    for backend, backend_aggs in aggs.items():
        for agg in backend_aggs:
            for label in RQ4_SET_LABELS:
                rows.append({"backend": backend, "run": agg.run, "linker_set": label,
                             "macro_f1": f"{agg.macro_only[label]:.6f}",
                             "macro_f2": f"{agg.macro_only_f2[label]:.6f}"})
    _write_csv(csv_root / "rq4_variants.csv",
               ["backend", "run", "linker_set", "macro_f1", "macro_f2"], rows)

    # rq4_variants_perproject.csv -- per-project doc-to-model link P/R/F1 backing the
    # single-linker macro-F1 above (entity-only / coref-only / full).
    rows = []
    for backend, backend_aggs in aggs.items():
        for agg in backend_aggs:
            for label in RQ4_SET_LABELS:
                for project in PROJECTS:
                    if project not in agg.dm_pp[label]:
                        continue
                    p, r, f1, f2 = agg.dm_pp[label][project]
                    rows.append({"backend": backend, "run": agg.run, "linker_set": label,
                                 "project": project, "doc_to_model_link_precision": f"{p:.6f}",
                                 "doc_to_model_link_recall": f"{r:.6f}",
                                 "doc_to_model_link_f1": f"{f1:.6f}",
                                 "doc_to_model_link_f2": f"{f2:.6f}"})
    _write_csv(csv_root / "rq4_variants_perproject.csv",
               ["backend", "run", "linker_set", "project", "doc_to_model_link_precision",
                "doc_to_model_link_recall", "doc_to_model_link_f1",
                "doc_to_model_link_f2"], rows)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Compute RQ3/RQ4 paper metrics from running results.")
    ap.add_argument("--csv-root", type=Path, default=_HERE / "reports",
                    help="output root (default: mini-rq34/reports)")
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=list(BACKENDS),
                    help=f"backends of the {ARM} arm (default: all)")
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
