#!/usr/bin/env python3
"""Phase 43 / v2.6.3 — RQ1 replay emitter (sad-sam + sad-code).

Reads the existing s_linker19 phase cache for one or more (backend, project)
pairs and writes two flat CSVs per pair under
``results/v2.6.3/<backend>/<project>/``:

  - ``sad-sam.csv``  columns: modelElementID, sentence, source
    (schema satisfies transarc-emp's load_result_sad_sam_standalone, which
    ignores extra columns; ``source`` is a diagnostic = 'entity' | 'coreference')

  - ``sad-code.csv`` columns: sentence, codeID
    (built by composing each sad-sam link with the gold SAM->code map; schema
    satisfies transarc-emp's load_result_sad_code which keys on (modelElementID
    OR sentence, codeID); we emit (sentence, code_path) per CONTEXT D-01)

Zero LLM calls. Hard-fails on OPENAI_API_KEY / ANTHROPIC_API_KEY /
LLM_BACKEND set to anything other than checkpoint (see replay_common.assert_no_llm_env).

Per CONTEXT D-01 / D-02 / D-14.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Make sibling replay_common importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_common import (
    BACKENDS,
    PROJECTS,
    OUTPUT_ROOT,
    assert_no_llm_env,
    load_layer,
)

# transarc-emp's lib/ has the SAM->code expansion helpers. D-02 allows
# approach/ -> evaluation/ imports (the constraint forbids the reverse).
_TRANSARC_LIB = Path("/mnt/hostshare/ardoco-home/transarc-emp/src/lib")
sys.path.insert(0, str(_TRANSARC_LIB))
from transarc_error_analysis import (  # type: ignore
    load_code_model_files,
    load_gs_sam_code_maps,
)


# ── CSV emitters ─────────────────────────────────────────────────────────────

def _final_link_iter(final_pkl: dict):
    """Yield SadSamLink objects from a final.pkl dict.

    ``final.pkl`` is the dict s_linker19._save_phase emitted at the final
    phase; the link list lives under the 'final' key. Each item is a
    SadSamLink(sentence_number, component_id, component_name, confidence, source).
    """
    if "final" not in final_pkl:
        raise KeyError("final.pkl missing expected 'final' key; "
                       f"available keys={sorted(final_pkl.keys())}")
    return final_pkl["final"]


def replay_sad_sam_for_project(backend: str, project: str, out_root: Path) -> int:
    """Write ``sad-sam.csv`` for one (backend, project). Returns row count."""
    final_pkl = load_layer(backend, project, "final")
    out_dir = out_root / backend / project
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sad-sam.csv"

    # Deduplicate on (sentence_number, component_id) — final.pkl already
    # composes entity_links + coref_validated and deduplicates at compose time
    # in s_linker19.py, but we re-dedupe defensively here.
    seen = {}
    for link in _final_link_iter(final_pkl):
        key = (link.sentence_number, link.component_id)
        if key not in seen:
            seen[key] = link

    rows = sorted(
        seen.values(),
        key=lambda lk: (lk.sentence_number, lk.component_id),
    )
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["modelElementID", "sentence", "source"])
        for link in rows:
            w.writerow([link.component_id, str(link.sentence_number), link.source])
    return len(rows)


def replay_sad_code_for_project(backend: str, project: str, out_root: Path) -> int:
    """Write ``sad-code.csv`` for one (backend, project) by composing
    sad-sam(s, m) ⨝ gold-sam-code(m, c) -> (s, c).

    Returns row count.
    """
    final_pkl = load_layer(backend, project, "final")
    out_dir = out_root / backend / project
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sad-code.csv"

    code_files = load_code_model_files(project)
    model_to_code, _ = load_gs_sam_code_maps(project, code_files)

    pairs = set()
    for link in _final_link_iter(final_pkl):
        comp = link.component_id
        sent = str(link.sentence_number)
        if comp not in model_to_code:
            continue  # component has no SAM->code mapping; skip
        for code_path in model_to_code[comp]:
            pairs.add((sent, code_path))

    rows = sorted(pairs)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sentence", "codeID"])
        for sent, code in rows:
            w.writerow([sent, code])
    return len(rows)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="replay_s19_to_csv.py",
        description="Replay s_linker19 phase cache → sad-sam/sad-code CSVs (RQ1).",
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
    n_sad_sam = 0
    n_sad_code = 0
    for backend in backends:
        for project in projects:
            try:
                sam_rows = replay_sad_sam_for_project(backend, project, out_root)
                code_rows = replay_sad_code_for_project(backend, project, out_root)
            except FileNotFoundError as e:
                print(f"[replay-rq1] SKIP backend={backend} project={project}: {e}",
                      file=sys.stderr)
                continue
            n_sad_sam += 1
            n_sad_code += 1
            print(f"[replay-rq1] backend={backend} project={project} "
                  f"sad-sam={sam_rows} sad-code={code_rows}")

    dt = time.time() - t0
    print(f"[replay-rq1] wrote {n_sad_sam} sad-sam CSVs + "
          f"{n_sad_code} sad-code CSVs in {dt:.1f}s "
          f"(out_root={out_root})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
