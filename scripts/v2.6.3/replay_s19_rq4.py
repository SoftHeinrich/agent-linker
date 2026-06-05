#!/usr/bin/env python3
"""Phase 43 / v2.6.3 — RQ4 2-linker overlap emitter.

Per CONTEXT D-05 / D-06 the RQ4 question is reframed from "4 agents" to
"2 linkers": Entity vs Coref. Both linker outputs are computed *after* their
respective validators (Entity = layer3.validated, Coref = layer4.coref_validated).

Outputs per (backend, project):

  - rq4.csv        columns: linker, tps_caught, unique_tps, fps, delta_f1_if_removed
                   (2 data rows: Entity, Coref)
       - tps_caught          = |linker ∩ gold|
       - unique_tps          = |(linker - other_linker) ∩ gold|
       - fps                 = |linker - gold|
       - delta_f1_if_removed = f1(E ∪ C) - f1(other_linker alone)
         True linker-ablation: removing one linker from the pipeline leaves
         the *other linker's full prediction set* (not the literal set-diff
         E ∪ C \ this_linker). When the two linkers share TPs, the surviving
         linker still catches those shared TPs on its own.

  - rq4_upset.csv  columns: cell, count
                   (3 data rows: only_E, both, only_C — TP-share decomposition
                   per CONTEXT D-06 for the 3-cell UpSet figure)
       - only_E = |(E - C) ∩ G|
       - both   = |E ∩ C ∩ G|
       - only_C = |(C - E) ∩ G|

Zero LLM calls (assert_no_llm_env at entry).

Per CONTEXT D-01 / D-02 / D-05 / D-06 / D-14.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_common import (
    BACKENDS,
    PROJECTS,
    OUTPUT_ROOT,
    assert_no_llm_env,
    load_all_layers,
    load_gold_links,
)

LinkKey = Tuple[int, str]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _keys_of(items: Iterable) -> Set[LinkKey]:
    return {(it.sentence_number, it.component_id) for it in items}


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom > 0 else 0.0


def _f1(predicted: Set[LinkKey], gold: Set[LinkKey]) -> float:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    return _safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0


# ── Public computation API ───────────────────────────────────────────────────

def compute_overlap_decomposition(layers: Dict[str, dict], gold: Set[LinkKey]) \
        -> Tuple[Dict[str, Tuple[int, int, int, float]], Dict[str, int]]:
    """Compute the per-linker and 3-cell-UpSet decomposition for RQ4.

    Returns:
        (per_linker, upset)

        per_linker[name] = (tps_caught, unique_tps, fps, delta_f1_if_removed)
            for name in {"Entity", "Coref"}

        upset = {"only_E": int, "both": int, "only_C": int}
            (TP-share decomposition vs gold)
    """
    layer3 = layers["layer3"]
    layer4 = layers["layer4"]

    E = _keys_of(layer3["validated"])
    C = _keys_of(layer4["coref_validated"])
    G = gold

    # 3-cell TP-share decomposition (CONTEXT D-06).
    only_E = (E - C) & G
    both   = (E & C) & G
    only_C = (C - E) & G

    union = E | C
    f1_union = _f1(union, G)
    # True linker-ablation (CONTEXT D-05): removing one linker from the
    # pipeline leaves the *other linker's full predictions* — the surviving
    # linker still catches the TPs the two linkers share. Compare against
    # set-diff (union - linker), which would drop every shared TP and inflate
    # the delta.
    f1_only_C = _f1(C, G)  # Entity removed → only Coref survives
    f1_only_E = _f1(E, G)  # Coref  removed → only Entity survives

    per_linker = {
        "Entity": (
            len(E & G),                # tps_caught
            len((E - C) & G),          # unique_tps (TPs only Entity caught)
            len(E - G),                # fps
            round(f1_union - f1_only_C, 6),  # delta_f1_if_removed (true ablation)
        ),
        "Coref": (
            len(C & G),
            len((C - E) & G),
            len(C - G),
            round(f1_union - f1_only_E, 6),
        ),
    }
    upset = {
        "only_E": len(only_E),
        "both":   len(both),
        "only_C": len(only_C),
    }
    return per_linker, upset


# ── CSV writers ──────────────────────────────────────────────────────────────

def _write_rq4_csv(out_dir: Path, per_linker: Dict[str, Tuple[int, int, int, float]]) -> None:
    path = out_dir / "rq4.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["linker", "tps_caught", "unique_tps", "fps", "delta_f1_if_removed"])
        for name in ("Entity", "Coref"):
            tps, uniq, fps, dF1 = per_linker[name]
            w.writerow([name, tps, uniq, fps, f"{dF1:.6f}"])


def _write_rq4_upset_csv(out_dir: Path, upset: Dict[str, int]) -> None:
    path = out_dir / "rq4_upset.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "count"])
        for name in ("only_E", "both", "only_C"):
            w.writerow([name, upset[name]])


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="replay_s19_rq4.py",
        description="Replay s_linker19 phase cache → RQ4 2-linker overlap CSVs.",
    )
    p.add_argument("--backend", default="all",
                   choices=BACKENDS + ["all"],
                   help="Which backend(s) to replay (default: all).")
    p.add_argument("--project", default="all",
                   choices=PROJECTS + ["all"],
                   help="Which project(s) to replay (default: all).")
    p.add_argument("--out-root", default=str(OUTPUT_ROOT),
                   help=f"Output root directory (default: {OUTPUT_ROOT}).")
    return p


def main(argv=None) -> int:
    # LLM-call hard-guard at entry — see CONTEXT D-01 / D-14.
    assert_no_llm_env()

    parser = _build_parser()
    args = parser.parse_args(argv)

    backends = list(BACKENDS) if args.backend == "all" else [args.backend]
    projects = list(PROJECTS) if args.project == "all" else [args.project]

    out_root = Path(args.out_root)

    t0 = time.time()
    n_files = 0
    for backend in backends:
        for project in projects:
            try:
                layers = load_all_layers(backend, project)
                gold = load_gold_links(project)
            except FileNotFoundError as e:
                print(f"[replay-rq4] SKIP backend={backend} project={project}: {e}",
                      file=sys.stderr)
                continue
            per_linker, upset = compute_overlap_decomposition(layers, gold)

            out_dir = out_root / backend / project
            out_dir.mkdir(parents=True, exist_ok=True)
            _write_rq4_csv(out_dir, per_linker)
            _write_rq4_upset_csv(out_dir, upset)
            n_files += 1

            print(f"[replay-rq4] backend={backend} project={project} "
                  f"only_E={upset['only_E']} both={upset['both']} only_C={upset['only_C']}")

    dt = time.time() - t0
    print(f"[replay-rq4] wrote {n_files} rq4.csv + {n_files} rq4_upset.csv "
          f"in {dt:.1f}s (out_root={out_root})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
