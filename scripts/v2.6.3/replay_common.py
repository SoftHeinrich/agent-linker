#!/usr/bin/env python3
"""Shared replay helpers for Phase 43 (v2.6.3) paper-eval pipeline.

This module is the replay-stage common library used by:
  - replay_s19_to_csv.py  (RQ1 sad-sam + sad-code emitters)
  - replay_s19_rq3.py     (RQ3 validator-counterfactual emitter)
  - replay_s19_rq4.py     (RQ4 2-linker overlap emitter)

Per CONTEXT decisions:
  - D-01: replay-stage scripts live in approach/ because pickle deserialization of
    SadSamLink / CandidateLink dataclasses requires `src/llm_sad_sam/` on sys.path.
  - D-02: stdlib-only.
  - D-14: no algorithm changes; no LLM calls. `assert_no_llm_env()` enforces this
    invariant by hard-failing if any LLM-enabling environment variable is set.

Inputs (read-only):
  results/phase_cache/s_linker19/{backend}/{project}/{layer1..4,final}.pkl

Outputs:
  results/v2.6.3/{backend}/{project}/{sad-sam,sad-code,rq3,rq4,...}.csv

Gold-standard CSV path (read-only):
  /mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/
    benchmark/<project>/goldstandards/goldstandard_sad_<year>-sam_<year>.csv
"""

from __future__ import annotations

import csv
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

# ── sys.path setup ────────────────────────────────────────────────────────────
# Pickle deserialization of SadSamLink / CandidateLink requires `src/` on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
BACKENDS = ["claude", "openai"]
# D-03: display label for "openai" pickles is "GPT-5.4".
BACKEND_DISPLAY = {"claude": "Claude", "openai": "GPT-5.4"}

# Phase cache: respects $PHASE_CACHE_DIR (same env var s_linker19 uses) and
# defaults to <repo>/results/phase_cache.
_PHASE_CACHE_DIR_ENV = os.environ.get("PHASE_CACHE_DIR")
if _PHASE_CACHE_DIR_ENV:
    PHASE_CACHE_ROOT = Path(_PHASE_CACHE_DIR_ENV) / "s_linker19"
else:
    PHASE_CACHE_ROOT = _REPO_ROOT / "results" / "phase_cache" / "s_linker19"

# Default output root. Callers (CLI) typically override with --out-root.
OUTPUT_ROOT = _REPO_ROOT / "results" / "v2.6.3"

# Gold-standard CSV paths per project. Naming matches transarc-emp's
# transarc_error_analysis.GS_SAD_SAM table.
_BENCHMARK = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
GS_SAD_SAM_PATHS = {
    "mediastore":    _BENCHMARK / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    "teastore":      _BENCHMARK / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    "teammates":     _BENCHMARK / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "bigbluebutton": _BENCHMARK / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "jabref":        _BENCHMARK / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
}

# Layers the cache exposes.
_VALID_LAYERS = {"layer1", "layer2", "layer3", "layer4", "final"}


# ── LLM-guard ────────────────────────────────────────────────────────────────

def assert_no_llm_env() -> None:
    """Hard-fail if any environment variable that could trigger a live LLM call
    is set.

    Phase 43 forbids new LLM calls; see CONTEXT D-01 and D-14. The replay
    pipeline must be byte-deterministic from the pickle cache.

    Specifically refuses to run if:
      - OPENAI_API_KEY is set (non-empty)
      - ANTHROPIC_API_KEY is set (non-empty)
      - LLM_BACKEND is set to anything other than "" or "checkpoint"
    """
    bad = []
    if os.environ.get("OPENAI_API_KEY"):
        bad.append("OPENAI_API_KEY")
    if os.environ.get("ANTHROPIC_API_KEY"):
        bad.append("ANTHROPIC_API_KEY")
    llm_backend = os.environ.get("LLM_BACKEND", "")
    if llm_backend not in ("", "checkpoint"):
        bad.append(f"LLM_BACKEND={llm_backend!r}")
    if bad:
        raise RuntimeError(
            "Phase 43 forbids new LLM calls; see CONTEXT D-01 and D-14. "
            f"Detected forbidden env vars: {', '.join(bad)}. "
            "Unset them before running any v2.6.3 replay script."
        )


# ── Pickle loaders ───────────────────────────────────────────────────────────

def load_layer(backend: str, project: str, layer: str) -> dict:
    """Load one layer pickle for one (backend, project).

    `layer` must be one of layer1..4 or final. Returns the unpickled dict shape
    s_linker19._save_phase emitted at the corresponding phase.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    if project not in PROJECTS:
        raise ValueError(f"unknown project {project!r}; expected one of {PROJECTS}")
    if layer not in _VALID_LAYERS:
        raise ValueError(f"unknown layer {layer!r}; expected one of {sorted(_VALID_LAYERS)}")
    path = PHASE_CACHE_ROOT / backend / project / f"{layer}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"phase-cache pickle not found: {path.resolve()} "
            f"(backend={backend} project={project} layer={layer})"
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_all_layers(backend: str, project: str) -> Dict[str, dict]:
    """Load all five layer pickles for one (backend, project) and return a dict
    keyed by layer name.
    """
    return {layer: load_layer(backend, project, layer) for layer in
            ("layer1", "layer2", "layer3", "layer4", "final")}


# ── Gold loader ──────────────────────────────────────────────────────────────

def load_gold_links(project: str) -> Set[Tuple[int, str]]:
    """Return the project's SAD-SAM gold standard as a set of
    ``(sentence_number_int, component_id_str)`` tuples.

    Mirrors transarc_error_analysis.load_gs_sad_sam but normalizes the sentence
    column to int (the replay scripts key links by ``(sentence_number, component_id)``
    where ``sentence_number`` is an int from the SadSamLink dataclass).
    """
    if project not in PROJECTS:
        raise ValueError(f"unknown project {project!r}; expected one of {PROJECTS}")
    path = GS_SAD_SAM_PATHS[project]
    if not path.exists():
        raise FileNotFoundError(f"gold standard CSV not found: {path}")
    links: Set[Tuple[int, str]] = set()
    with open(path) as fh:
        for row in csv.DictReader(fh):
            sent = int(row["sentence"])
            comp = row["modelElementID"]
            links.add((sent, comp))
    return links


# ── Public exports ───────────────────────────────────────────────────────────

__all__ = [
    "PROJECTS",
    "BACKENDS",
    "BACKEND_DISPLAY",
    "PHASE_CACHE_ROOT",
    "OUTPUT_ROOT",
    "GS_SAD_SAM_PATHS",
    "assert_no_llm_env",
    "load_layer",
    "load_all_layers",
    "load_gold_links",
]
