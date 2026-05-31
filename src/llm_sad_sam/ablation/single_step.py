"""Single-step ablation engine for the v2.1 trim chain (Phase 12, PROMPT-02).

Loads upstream checkpoints from `results/phase_cache/<variant>/<dataset>/`,
re-executes ONE target phase on a given variant, propagates the result through
the downstream phases that depend on the modified phase's output, scores
against the gold standard, and writes a per-run results JSON with delta vs the
v2.0 baseline.

Why this exists: 12-CONTEXT.md "Execution Method — Checkpoint-Loaded
Single-Step Ablation (USER DIRECTIVE)" rules out full-pipeline sweeps per
trim. The harness reuses the cached output of upstream phases instead of
recomputing them, and only re-runs the modified phase plus its downstream
descendants per `DOWNSTREAM_DEPS`.

CRITICAL HARNESS CONTRACT (entity_candidates / entity_decisions reuse rule)
--------------------------------------------------------------------------
`layer2.pkl` is NOT a re-runnable phase — it is the synthesis pickle of the
`_run_parallel({seed_val, coref, entity})` block inside
`s_linker13_clean.link()`. When the requested phase is ``entity_candidates``
or ``entity_decisions``, the harness uses ``layer2.pkl`` purely as a CACHE
READ for the seed_val + coref tracks; the entity track is overridden
surgically. The harness MUST NOT make any live LLM calls to
``_run_seed_validation`` or ``_run_coreference`` in this mode — implemented
by monkey-patching both methods to raise ``AssertionError`` if invoked.

PHASE -> DOWNSTREAM RE-RUN TABLE (see 12-02-HARNESS-CONTRACT.md)
---------------------------------------------------------------
- layer1            -> layer2, entity_candidates, entity_decisions, final
- layer2            -> final
- entity_candidates -> entity_decisions, final
- entity_decisions  -> final
- final             -> (terminal, no downstream)

The harness coupling reaches into the semi-private methods
``_run_entity_pipeline``, ``_extract_entities_enriched``,
``_validate_with_evidence``, ``_run_seed_validation``,
``_run_coreference`` by name. They are stable in ``s_linker13_clean`` (Phase
10 promotion contract), but any future refactor MUST preserve them or
update this harness in lock-step. Tracked as technical debt in
12-02-SUMMARY.md.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import json
import os
import pickle
import sys
from pathlib import Path

# Ensure repository root + src are on sys.path so we can import run_ablation
# and llm_sad_sam.* regardless of how the harness was invoked.
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]  # src/llm_sad_sam/ablation/single_step.py -> repo
_SRC = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


#: Canonical pipeline order. Indices into this tuple drive the upstream
#: checkpoint requirements (every phase before X must exist before X can
#: re-run) and the downstream re-run set (every phase after X may need to
#: re-execute when X's output changes).
PHASE_ORDER = ("layer1", "layer2", "entity_candidates", "entity_decisions", "final")


#: Per-phase downstream re-run dependency. Maps a modified phase to the
#: tuple of phases that MUST be re-run when its output changes. Sourced
#: from 12-CONTEXT.md decisions ("the executor MUST either re-run downstream
#: phases or use existing checkpoint") and the s_linker13_clean.link()
#: DAG structure: seed_val + coref + entity all read layer1 state, and
#: dedup ("final") reads everything in layer2 + entity_decisions.
DOWNSTREAM_DEPS = {
    "layer1": ("layer2", "entity_candidates", "entity_decisions", "final"),
    "layer2": ("final",),
    "entity_candidates": ("entity_decisions", "final"),
    "entity_decisions": ("final",),
    "final": (),
}


_DEFAULT_PHASE_CACHE = "./results/phase_cache"
_BASELINE_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "v2_0_baseline.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_single_step(
    variant: str,
    dataset: str,
    phase: str,
    results_dir,
    backend: str = "claude",
    model: str | None = None,
    phase_cache_dir: str | None = None,
) -> dict:
    """Re-execute one target phase of a variant, score, write results JSON.

    Args:
        variant: Variant key in `VARIANT_SPECS` (e.g. "s_linker13_clean").
        dataset: Dataset key in `DATASETS` (e.g. "mediastore").
        phase: One of `PHASE_ORDER` ("layer1", "layer2", "entity_candidates",
            "entity_decisions", "final"). ValueError otherwise.
        results_dir: Where the per-run JSON is written
            (`<results_dir>/<variant>/<dataset>/<phase>.json`).
        backend: "claude" | "openai" | "checkpoint". Smoke tests use
            "checkpoint" so no live LLM call is made.
        model: Optional model override (passed to the variant constructor).
        phase_cache_dir: Optional override for the upstream checkpoint root.
            Defaults to env PHASE_CACHE_DIR or "./results/phase_cache".

    Returns:
        Result dict with keys:
            variant, dataset, phase, F1, P, R, fp, fn,
            baseline_F1, delta_F1,
            phase_cache_dir, timestamp.

    Raises:
        ValueError: phase not in PHASE_ORDER.
        KeyError: variant not in VARIANT_SPECS, dataset not in DATASETS.
        FileNotFoundError: a required upstream pickle is missing.
    """
    if phase not in PHASE_ORDER:
        raise ValueError(
            f"Unknown phase {phase!r}; must be one of {list(PHASE_ORDER)}"
        )

    # Lazy import to keep the contract layer free of heavy dependencies.
    import run_ablation as _ra
    from llm_sad_sam.llm_client import LLMBackend

    if variant not in _ra.VARIANT_SPECS:
        raise KeyError(
            f"Unknown variant {variant!r}; available: "
            f"{sorted(_ra.VARIANT_SPECS.keys())}"
        )
    if dataset not in _ra.DATASETS:
        raise KeyError(
            f"Unknown dataset {dataset!r}; available: {sorted(_ra.DATASETS.keys())}"
        )

    cache_root = Path(phase_cache_dir or os.environ.get(
        "PHASE_CACHE_DIR", _DEFAULT_PHASE_CACHE))
    baseline_cache = cache_root / variant / dataset

    upstream_needed = _required_upstream_pickles(phase)
    missing = [str(baseline_cache / f"{name}.pkl") for name in upstream_needed
               if not (baseline_cache / f"{name}.pkl").exists()]
    if missing:
        raise FileNotFoundError(
            "Required upstream checkpoint(s) missing for "
            f"phase={phase!r}, variant={variant!r}, dataset={dataset!r}: "
            f"{missing}"
        )

    # Instantiate the variant (canonical resolution via run_ablation).
    backend_enum = _coerce_backend(backend, LLMBackend)
    linker = _build_linker(variant, backend_enum, model, _ra)

    paths = _ra.DATASETS[dataset]
    text_path = str(paths["text"])
    model_path = str(paths["model"])

    # Redirect the variant's own _save_phase writes to a per-run tmp dir so
    # the canonical baseline cache stays untouched.
    out_results_dir = Path(results_dir) / variant / dataset
    out_results_dir.mkdir(parents=True, exist_ok=True)
    tmp_cache = out_results_dir / "_phase_cache_tmp"
    tmp_cache.mkdir(parents=True, exist_ok=True)
    saved_env = os.environ.get("PHASE_CACHE_DIR")
    os.environ["PHASE_CACHE_DIR"] = str(tmp_cache)

    try:
        final_links = _execute_phase(
            linker=linker,
            phase=phase,
            text_path=text_path,
            model_path=model_path,
            baseline_cache=baseline_cache,
        )
    finally:
        if saved_env is None:
            os.environ.pop("PHASE_CACHE_DIR", None)
        else:
            os.environ["PHASE_CACHE_DIR"] = saved_env

    # Score against gold.
    predicted = {(lk.sentence_number, lk.component_id) for lk in final_links}
    gold = _ra.load_gold_sam(str(paths["gold_sam"]))
    metrics = _ra.eval_metrics(predicted, gold)

    baseline_f1 = _lookup_baseline_f1(variant, dataset)
    delta_f1 = (metrics["F1"] - baseline_f1) if baseline_f1 is not None else None

    result = {
        "variant": variant,
        "dataset": dataset,
        "phase": phase,
        "F1": metrics["F1"],
        "P": metrics["P"],
        "R": metrics["R"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "baseline_F1": baseline_f1,
        "delta_F1": delta_f1,
        "phase_cache_dir": str(cache_root),
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    out_json = out_results_dir / f"{phase}.json"
    out_json.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _required_upstream_pickles(phase: str) -> tuple[str, ...]:
    """Return phases that MUST have a baseline pickle on disk for this run.

    Rule: every phase strictly before `phase` in PHASE_ORDER. With one
    refinement: `entity_decisions` additionally requires
    `entity_candidates.pkl`, which is already covered because
    `entity_candidates` precedes `entity_decisions` in PHASE_ORDER.
    """
    idx = PHASE_ORDER.index(phase)
    return PHASE_ORDER[:idx]


def _coerce_backend(backend, LLMBackend):
    name = (backend or "claude").strip().lower()
    table = {
        "claude": LLMBackend.CLAUDE,
        "openai": LLMBackend.OPENAI,
        "checkpoint": LLMBackend.CHECKPOINT,
        "codex": LLMBackend.CODEX,
    }
    if name not in table:
        raise ValueError(f"Unknown backend {backend!r}; expected one of {list(table)}")
    return table[name]


def _build_linker(variant: str, backend, model, _ra):
    """Instantiate the variant class (mirroring run_ablation.build_linker)."""
    canonical = _ra.canonical_variant(variant)
    if canonical == "i3":
        # i3 has no checkpoint phases; the harness only meaningfully supports
        # s_linker13_*-style variants. Use the adapter for completeness.
        return _ra.ILinker3Adapter(backend=backend)
    spec = _ra.VARIANT_SPECS[canonical]
    module = importlib.import_module(spec["module"])
    cls = getattr(module, spec["class_name"])
    kwargs = {"backend": backend}
    if model is not None:
        kwargs["model"] = model
    return cls(**kwargs)


def _lookup_baseline_f1(variant: str, dataset: str) -> float | None:
    """Look up the v2.0 baseline F1 for (variant, dataset) if pinned.

    Returns None if the variant is in the fixture's "missing" list (e.g.
    s_linker13_clean is post-v2.0). The harness uses the cached final.pkl-
    derived F1 as a fallback anchor so the no-op test can still assert
    equivalence.
    """
    if not _BASELINE_FIXTURE.exists():
        return None
    try:
        data = json.loads(_BASELINE_FIXTURE.read_text())
    except json.JSONDecodeError:
        return None
    variants = data.get("variants", {})
    if variant in variants:
        per = variants[variant].get("per_dataset", {})
        if dataset in per and "F1" in per[dataset]:
            return float(per[dataset]["F1"])
    # Fallback: anchor against the cached final.pkl-derived F1 so the
    # no-op equivalence test has a meaningful comparison point even for
    # post-v2.0 variants (like s_linker13_clean) that the fixture explicitly
    # marks as missing.
    return _baseline_from_cached_final(variant, dataset)


def _baseline_from_cached_final(variant: str, dataset: str) -> float | None:
    """Compute F1 from the canonical baseline final.pkl if available."""
    try:
        import run_ablation as _ra
    except Exception:
        return None
    cache_root = Path(os.environ.get("PHASE_CACHE_DIR", _DEFAULT_PHASE_CACHE))
    final_pkl = cache_root / variant / dataset / "final.pkl"
    if not final_pkl.exists():
        return None
    if dataset not in _ra.DATASETS:
        return None
    try:
        with open(final_pkl, "rb") as f:
            data = pickle.load(f)
        links = data.get("final", []) if isinstance(data, dict) else []
        predicted = {(lk.sentence_number, lk.component_id) for lk in links}
        gold = _ra.load_gold_sam(str(_ra.DATASETS[dataset]["gold_sam"]))
        return _ra.eval_metrics(predicted, gold)["F1"]
    except Exception:
        return None


def _load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _dedup_links(seed_links, validated, coref_links):
    """Reproduce s_linker13_clean.link() Tier 3 dedup exactly.

    Order: seed -> entity -> coref; first-seen wins on
    (sentence_number, component_id).
    """
    from llm_sad_sam.core.data_types_v2 import SadSamLink

    entity_links = [
        SadSamLink(c.sentence_number, c.component_id, c.component_name,
                   source=c.source)
        for c in validated
    ]
    all_links = list(seed_links) + entity_links + list(coref_links)
    seen = set()
    final = []
    for lk in all_links:
        key = (lk.sentence_number, lk.component_id)
        if key not in seen:
            seen.add(key)
            final.append(lk)
    return final


def _reconstruct_validated(entity_candidates, decisions):
    """Reconstruct the validated CandidateLink list from a decisions dict."""
    out = []
    for c in entity_candidates:
        key = (c.sentence_number, c.component_id)
        dec = decisions.get(key)
        if dec and dec.get("approved"):
            out.append(c)
    return out


def _patch_block_method(linker, name: str, message: str):
    """Replace `linker.<name>` with a raising stub to enforce zero-call contract.

    Used during phase=entity_candidates / entity_decisions to prove the
    seed_val and coref tracks are NOT invoked live.
    """
    def _stub(*args, **kwargs):
        raise AssertionError(
            f"{name} called during surgical re-run: {message}. "
            "This violates the harness CRITICAL CONTRACT — entity-track "
            "phases must reuse cached seed_val + coref from layer2.pkl."
        )
    setattr(linker, name, _stub)


def _execute_phase(linker, phase: str, text_path: str, model_path: str,
                   baseline_cache: Path):
    """Re-execute the target phase and return the final SadSamLink list.

    Dispatches per phase. For entity-track phases the seed_val and coref
    methods are blocked to enforce the cache-reuse contract.
    """
    # Common: load supporting data the variant needs but we don't want to
    # re-derive (PCM parse + sentence load are fast enough to redo).
    from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
    from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository

    if phase == "layer1":
        # Full pipeline re-run; the variant writes its own checkpoints into
        # the redirected tmp PHASE_CACHE_DIR.
        return linker.link(text_path=text_path, model_path=model_path)

    components = parse_pcm_repository(model_path)
    sentences = load_sentences(text_path)
    name_to_id = {c.name: c.id for c in components}
    sent_map = build_sent_map(sentences)

    # Restore Tier 1 state for the variant from the cached layer1.pkl.
    layer1 = _load_pkl(baseline_cache / "layer1.pkl")
    linker.model_knowledge = layer1["model_knowledge"]
    linker.doc_knowledge = layer1["doc_knowledge"]
    linker._current_text_path = text_path
    raw_seed_links = layer1["raw_seed_links"]

    if phase == "layer2":
        # Re-run seed_val + entity pipeline + coref; dedup.
        # NB: this DOES re-run the entity track AND seed_val + coref. Trim
        # plans 12-03/04/05 do not target this phase directly (Step 2 hits
        # entity_candidates / entity_decisions). Phase=layer2 here is the
        # composition shim used for completeness (e.g. when somebody trims
        # layer2 wholesale, which no current trim does).
        seed_links = linker._run_seed_validation(raw_seed_links, components, sent_map)
        validated = linker._run_entity_pipeline(sentences, components, name_to_id, sent_map)
        coref_links = linker._run_coreference(sentences, components, name_to_id, sent_map)
        return _dedup_links(seed_links, validated, coref_links)

    # Entity-track phases — reuse cached seed_val + coref under the
    # CRITICAL HARNESS CONTRACT.
    layer2 = _load_pkl(baseline_cache / "layer2.pkl")
    cached_seed_links = layer2["seed_links"]
    cached_coref_links = layer2["coref_links"]
    _patch_block_method(linker, "_run_seed_validation",
                        "seed_val should be reused from layer2.pkl")
    _patch_block_method(linker, "_run_coreference",
                        "coref should be reused from layer2.pkl")

    if phase == "entity_candidates":
        # Re-run entity extraction + validation only.
        validated = linker._run_entity_pipeline(sentences, components, name_to_id, sent_map)
        return _dedup_links(cached_seed_links, validated, cached_coref_links)

    if phase == "entity_decisions":
        # Re-run validation only; reuse cached entity_candidates + bundles.
        ec_pkl = _load_pkl(baseline_cache / "entity_candidates.pkl")
        candidates = ec_pkl["entity_candidates"]
        bundles = ec_pkl["bundles"]
        validated, _decisions = linker._validate_with_evidence(
            candidates, bundles, components, sent_map)
        return _dedup_links(cached_seed_links, validated, cached_coref_links)

    if phase == "final":
        # Pure dedup from cached upstream — no LLM calls at all.
        # Reuse the layer2 outputs + entity_decisions outputs as the variant
        # original linker.link() would have at Tier 3.
        ed_pkl = _load_pkl(baseline_cache / "entity_decisions.pkl")
        ec_pkl = _load_pkl(baseline_cache / "entity_candidates.pkl")
        decisions = ed_pkl["decisions"]
        candidates = ec_pkl["entity_candidates"]
        validated = _reconstruct_validated(candidates, decisions)
        return _dedup_links(cached_seed_links, validated, cached_coref_links)

    raise ValueError(f"Unhandled phase {phase!r}")  # pragma: no cover
