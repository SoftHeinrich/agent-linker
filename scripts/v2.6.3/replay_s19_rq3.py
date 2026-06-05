#!/usr/bin/env python3
"""Phase 43 / v2.6.3 — RQ3 validator-counterfactual emitter.

Derives the four RQ3 variants from layer3.pkl and layer4.pkl without any new
LLM calls. Per CONTEXT D-08, the variants are:

  | Variant         | Definition                                          | Derivation                                       |
  | --------------- | --------------------------------------------------- | ------------------------------------------------ |
  | Full            | Entity-validator ON + Coref/Citation-validator ON   | layer3.validated ∪ layer4.coref_validated        |
  | NoEntityValid   | Skip layer3 entity validator                        | layer3.candidates ∪ layer4.coref_validated       |
  | NoCitation      | Skip layer4 coref/citation validator                | layer3.validated ∪ layer4.coref_raw              |
  | NoValidator     | Skip both LLM-call validators                       | layer3.candidates ∪ layer4.coref_raw             |

Outputs per (backend, project):
  - rq3.csv        columns: variant, tp, fp, fn, precision, recall, f1
                   (4 data rows: Full, NoEntityValid, NoCitation, NoValidator)
  - rq3_audit.csv  columns: validator, killed_gold, killed_spurious, kept_gold, kept_spurious
                   (2 data rows: entity, coref)

Zero LLM calls (assert_no_llm_env at entry).

Per CONTEXT D-01 / D-02 / D-08 / D-09 / D-14.
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

# Type alias for link-set keys.
LinkKey = Tuple[int, str]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _candidate_keys(items: Iterable) -> Set[LinkKey]:
    """Project a list of CandidateLink or SadSamLink objects onto their
    (sentence_number, component_id) keys.
    """
    keys: Set[LinkKey] = set()
    for it in items:
        keys.add((it.sentence_number, it.component_id))
    return keys


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom > 0 else 0.0


def _prf1(predicted: Set[LinkKey], gold: Set[LinkKey]) -> Tuple[int, int, int, float, float, float]:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0
    return tp, fp, fn, p, r, f1


# ── Public computation API ───────────────────────────────────────────────────

def compute_variant_metrics(layers: Dict[str, dict], gold: Set[LinkKey]) \
        -> Dict[str, Tuple[int, int, int, float, float, float]]:
    """Compute TP/FP/FN/P/R/F1 for the four RQ3 variants per CONTEXT D-08.

    Returns a dict variant_name -> (tp, fp, fn, p, r, f1).
    """
    layer3 = layers["layer3"]
    layer4 = layers["layer4"]

    candidates_keys = _candidate_keys(layer3["candidates"])
    validated_keys  = _candidate_keys(layer3["validated"])
    coref_raw_keys       = _candidate_keys(layer4["coref_raw"])
    coref_validated_keys = _candidate_keys(layer4["coref_validated"])

    variants = {
        "Full":          validated_keys  | coref_validated_keys,
        "NoEntityValid": candidates_keys | coref_validated_keys,
        "NoCitation":    validated_keys  | coref_raw_keys,
        "NoValidator":   candidates_keys | coref_raw_keys,
    }
    return {name: _prf1(pred, gold) for name, pred in variants.items()}


def compute_validator_audit_counts(layers: Dict[str, dict], gold: Set[LinkKey]) \
        -> Dict[str, Tuple[int, int, int, int]]:
    """Compute per-validator audit counts (killed_gold, killed_spurious,
    kept_gold, kept_spurious) from layer3 decisions (entity validator) and
    layer4 coref_decisions (coref validator).

    Returns {'entity': (kg, ks, kpg, kps), 'coref': (kg, ks, kpg, kps)}.
    Each bucket is keyed by approval state × gold membership of the
    candidate (s, c) key.
    """
    layer3 = layers["layer3"]
    layer4 = layers["layer4"]

    def _audit(cand_items, decisions: dict) -> Tuple[int, int, int, int]:
        killed_gold = killed_spurious = kept_gold = kept_spurious = 0
        for item in cand_items:
            key: LinkKey = (item.sentence_number, item.component_id)
            dec = decisions.get(key)
            approved = bool(dec and dec.get("approved", False))
            in_gold = key in gold
            if approved:
                if in_gold:
                    kept_gold += 1
                else:
                    kept_spurious += 1
            else:
                if in_gold:
                    killed_gold += 1
                else:
                    killed_spurious += 1
        return killed_gold, killed_spurious, kept_gold, kept_spurious

    entity = _audit(layer3["candidates"], layer3["decisions"])
    coref  = _audit(layer4["coref_raw"], layer4["coref_decisions"])
    return {"entity": entity, "coref": coref}


# ── CSV writers ──────────────────────────────────────────────────────────────

# Ordered variant list — header row order matches D-08 derivation table.
_VARIANT_ORDER = ["Full", "NoEntityValid", "NoCitation", "NoValidator"]


def _write_rq3_csv(out_dir: Path, metrics: Dict[str, Tuple[int, int, int, float, float, float]]) -> None:
    path = out_dir / "rq3.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "tp", "fp", "fn", "precision", "recall", "f1"])
        for name in _VARIANT_ORDER:
            tp, fp, fn, p, r, f1 = metrics[name]
            w.writerow([name, tp, fp, fn, f"{p:.6f}", f"{r:.6f}", f"{f1:.6f}"])


def _write_rq3_audit_csv(out_dir: Path, audit: Dict[str, Tuple[int, int, int, int]]) -> None:
    path = out_dir / "rq3_audit.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["validator", "killed_gold", "killed_spurious", "kept_gold", "kept_spurious"])
        for name in ("entity", "coref"):
            kg, ks, kpg, kps = audit[name]
            w.writerow([name, kg, ks, kpg, kps])


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="replay_s19_rq3.py",
        description="Replay s_linker19 phase cache → RQ3 validator-counterfactual CSVs.",
    )
    p.add_argument("--backend", default="all",
                   choices=BACKENDS + ["all"],
                   help="Which backend(s) to replay (default: all).")
    p.add_argument("--project", default="all",
                   choices=PROJECTS + ["all"],
                   help="Which project(s) to replay (default: all).")
    p.add_argument("--out-root", default=str(OUTPUT_ROOT),
                   help=f"Output root directory (default: {OUTPUT_ROOT}).")
    p.add_argument("--all", action="store_true",
                   help="Shorthand for --backend all --project all (also the default).")
    return p


def main(argv=None) -> int:
    # LLM-call hard-guard at entry — see CONTEXT D-01 / D-14.
    assert_no_llm_env()

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.all:
        backends = list(BACKENDS)
        projects = list(PROJECTS)
    else:
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
                print(f"[replay-rq3] SKIP backend={backend} project={project}: {e}",
                      file=sys.stderr)
                continue
            metrics = compute_variant_metrics(layers, gold)
            audit = compute_validator_audit_counts(layers, gold)

            out_dir = out_root / backend / project
            out_dir.mkdir(parents=True, exist_ok=True)
            _write_rq3_csv(out_dir, metrics)
            _write_rq3_audit_csv(out_dir, audit)
            n_files += 1

            full = metrics["Full"]
            print(f"[replay-rq3] backend={backend} project={project} "
                  f"Full tp={full[0]} fp={full[1]} fn={full[2]} "
                  f"f1={full[5]:.3f}")

    dt = time.time() - t0
    print(f"[replay-rq3] wrote {n_files} rq3.csv + {n_files} rq3_audit.csv "
          f"in {dt:.1f}s (out_root={out_root})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
