#!/usr/bin/env python3
"""EXT-01 D-02 offline diff stage.

For each of the 5 benchmark datasets and each candidate sub-variant
(s_linker13g_pre, s_linker13g_sem), compute:
  - regex baseline anchor set: S_regex[ds][comp] = set of snums where
    s_linker13._has_standalone_mention(comp_name, sent.text) is True
  - variant anchor set: S_v[ds][comp] = set of snums where the variant's
    _compute_standalone_mention_map returns True

Roll up per-(variant, ds):
  - min_jaccard_per_comp
  - mean_jaccard_weighted (weighted by |S_regex[comp]|)
  - count_components_with_J<0.5
  - max_symmetric_diff

Apply the catastrophic-diff drop rule (RESEARCH.md §"Empirical Matrix
Operationalization", lines 319-321):

  Drop variant if (on TM or BBB):
    - min_jaccard_per_comp < 0.3, OR
    - any (comp, ds) has max_symmetric_diff > 10, OR
    - count_components_with_J<0.5 > 25% of components.

Emits:
  - results/ablation_results/ablation_ext01_diff.json (machine-readable matrix)

Caching:
  - Each variant's _compute_standalone_mention_map already persists to
    PHASE_CACHE_DIR/<_VARIANT_NAME>/<ds>/standalone_map.pkl via _save_phase().
    On re-runs, this harness reloads the pickle and skips LLM calls.

This script does NOT invoke the full pipeline. It only runs Tier-1's
_compute_standalone_mention_map per variant + the regex baseline.

Usage:
  python scripts/ext01_diff_stage.py
  python scripts/ext01_diff_stage.py --variants s_linker13g_pre --datasets mediastore
"""

# ---------------------------------------------------------------------------
# Plan 06-07 extensions (in place; original two-variant behavior preserved):
# 1. Denominator-aware Jaccard skip — when |S_baseline[comp]| == 0 for a
#    (comp, ds) cell, J collapses to 0 mechanically. Per Plan 06-03 user
#    adjudication on BBB/`kurento` and BENCHMARK_TABOO.md §"Tailored Code
#    Anti-Patterns", we do NOT patch the baseline per component — we skip
#    the J check on that cell and rely on the symmetric-difference (D)
#    check alone. Encoded as a single module constant
#    DENOMINATOR_AWARE_J_SKIP (no per-call override surface).
# 2. Dual-baseline mode — the comparison anchor can be re-anchored against
#    the rejected pure-LLM baselines (s_linker13g_pre, s_linker13g_sem)
#    from Plan 06-04 via cached pickles, at zero LLM cost. This is the
#    empirical test for CONTEXT.md D-09 (do the alias-aware variants
#    actually deviate from the rejected baselines?). The regex baseline
#    remains the operative drop-decision-gating comparison.
# 3. Four new alias-aware variants from Plan 06-06 wired into
#    --variants choices and the variant_classes dict.
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Repo root inferred from script location.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_sad_sam.linkers.experimental.s_linker13 import SLinker13
from llm_sad_sam.linkers.experimental.s_linker13g_pre import SLinker13gPre
from llm_sad_sam.linkers.experimental.s_linker13g_sem import SLinker13gSem
from llm_sad_sam.linkers.experimental.s_linker13g_pre_alias import SLinker13gPreAlias
from llm_sad_sam.linkers.experimental.s_linker13g_sem_alias import SLinker13gSemAlias
from llm_sad_sam.linkers.experimental.s_linker13g_pre_full import SLinker13gPreFull
from llm_sad_sam.linkers.experimental.s_linker13g_sem_full import SLinker13gSemFull
from llm_sad_sam.core.document_loader_v2 import load_sentences
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository

# Sanity checks — class names must match variant registry names.
assert SLinker13._VARIANT_NAME == "s_linker13", f"unexpected {SLinker13._VARIANT_NAME!r}"
assert SLinker13gPre._VARIANT_NAME == "s_linker13g_pre", f"unexpected {SLinker13gPre._VARIANT_NAME!r}"
assert SLinker13gSem._VARIANT_NAME == "s_linker13g_sem", f"unexpected {SLinker13gSem._VARIANT_NAME!r}"
assert SLinker13gPreAlias._VARIANT_NAME == "s_linker13g_pre_alias", f"unexpected {SLinker13gPreAlias._VARIANT_NAME!r}"
assert SLinker13gSemAlias._VARIANT_NAME == "s_linker13g_sem_alias", f"unexpected {SLinker13gSemAlias._VARIANT_NAME!r}"
assert SLinker13gPreFull._VARIANT_NAME == "s_linker13g_pre_full", f"unexpected {SLinker13gPreFull._VARIANT_NAME!r}"
assert SLinker13gSemFull._VARIANT_NAME == "s_linker13g_sem_full", f"unexpected {SLinker13gSemFull._VARIANT_NAME!r}"


# Dataset registry — mirrors run_ablation.py:341-374.
BENCHMARK_BASE = (REPO_ROOT / "../ardoco/core/tests-base/src/main/resources/benchmark").resolve()

DATASETS: dict[str, dict[str, Path]] = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
        "model": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
    },
}

HARD_TIER = {"teammates", "bigbluebutton"}

# Drop-rule thresholds (RESEARCH.md §"Empirical Matrix Operationalization", lines 319-321).
# Encoded as module constants — no threshold args on apply_drop_rule (T-06-03-01 mitigation).
HARD_TIER_MIN_J = 0.3
MAX_SYM_DIFF = 10
HARD_TIER_PCT_LOW_J = 0.25

# Denominator-aware Jaccard skip (Plan 06-07).
# When |S_baseline[comp]| == 0 for a (comp, ds) cell, Jaccard collapses to 0 mechanically.
# Per Plan 06-03 user adjudication + BENCHMARK_TABOO Tailored Code Anti-Patterns, we
# do NOT patch baselines per-component — we skip the J check on the cell and rely on
# the symmetric-difference (D) check alone (D is well-defined when |S_baseline|=0).
DENOMINATOR_AWARE_J_SKIP = True

# Pure-LLM baseline mode — re-anchor against rejected-baseline variants (Plan 06-04).
PURE_LLM_BASELINE_VARIANTS = ("s_linker13g_pre", "s_linker13g_sem")


def compute_anchor_set(
    predicate: Callable[[str, str], bool],
    sentences: list,
    comp_names: list[str],
) -> dict[str, set[int]]:
    """Returns dict[comp_name -> set[snum]] of snums where predicate(cname, sent.text) is True."""
    out: dict[str, set[int]] = {c: set() for c in comp_names}
    for c in comp_names:
        for s in sentences:
            if predicate(c, s.text):
                out[c].add(s.number)
    return out


def regex_predicate(cname: str, text: str) -> bool:
    """The frozen regex baseline — calls s_linker13's static method directly."""
    return SLinker13._has_standalone_mention(cname, text)


def _variant_cache_path(variant_name: str, text_path: Path) -> Path:
    """Mirror SLinker13gPre._checkpoint_dir() — keeps harness cache compatible with
    the variant's own _save_phase mechanism so the standalone_map persists across runs.
    """
    cache_root = Path(os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache"))
    ds = text_path.stem
    return cache_root / variant_name / ds / "standalone_map.pkl"


def variant_anchor_set(
    variant_inst,
    sentences: list,
    components: list,
    text_path: Path,
    *,
    force_recompute: bool = False,
) -> tuple[dict[str, set[int]], bool, int]:
    """Invokes the variant's Tier-1 _compute_standalone_mention_map and projects to set form.

    Reuses the variant's pickled standalone_map if present (PHASE_CACHE_DIR/<v>/<ds>/standalone_map.pkl).
    Returns (anchor_set, from_cache, llm_pair_count) where llm_pair_count is the number of
    (cname, snum) pairs in the underlying map — a proxy for LLM cost (per Plan output spec).
    """
    cache_path = _variant_cache_path(variant_inst._VARIANT_NAME, text_path)
    smap: dict[tuple[str, int], bool] | None = None

    if cache_path.exists() and not force_recompute:
        try:
            with open(cache_path, "rb") as f:
                blob = pickle.load(f)
            smap = blob.get("standalone_map") if isinstance(blob, dict) else None
            if smap is not None:
                print(f"    [cache hit] {cache_path}: {len(smap)} (cname, snum) pairs")
        except Exception as exc:
            print(f"    [cache load failed: {exc}] recomputing")
            smap = None

    from_cache = smap is not None

    if smap is None:
        # Set _current_text_path so the variant's internal _save_phase calls use
        # the same dataset-namespaced cache dir.
        variant_inst._current_text_path = str(text_path)
        print(f"    [computing standalone_map] {variant_inst._VARIANT_NAME} on {text_path.stem}")
        smap = variant_inst._compute_standalone_mention_map(sentences, components)
        # Persist for re-runs — same envelope as the variant's _save_phase.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"standalone_map": smap}, f)
        print(f"    [cache write] {cache_path}: {len(smap)} pairs")

    comp_names = [c.name for c in components]
    out: dict[str, set[int]] = {c: set() for c in comp_names}
    for (cname, snum), is_standalone in smap.items():
        if is_standalone and cname in out:
            out[cname].add(snum)
    return out, from_cache, len(smap)


def load_pure_llm_baseline_anchor_set(
    baseline_variant: str,
    text_path: Path,
    comp_names: list[str],
) -> tuple[dict[str, set[int]], bool]:
    """Load a rejected-baseline (pure-LLM) anchor set from the cached pickle.

    Pickles live at PHASE_CACHE_DIR/<baseline_variant>/<ds>/standalone_map.pkl —
    populated by Plan 06-03 / 06-04. Loader is cache-only (no LLM cost). Errors
    fast if the pickle is missing (T-06-07-02 / T-06-07-05 mitigation).
    """
    if baseline_variant not in PURE_LLM_BASELINE_VARIANTS:
        raise ValueError(
            f"Pure-LLM baseline must be one of {PURE_LLM_BASELINE_VARIANTS}, got {baseline_variant!r}"
        )
    cache_path = _variant_cache_path(baseline_variant, text_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Pure-LLM baseline cache missing: {cache_path}. "
            f"Run Plan 06-03 / 06-04 first OR pass --baseline regex."
        )
    with open(cache_path, "rb") as f:
        blob = pickle.load(f)
    smap = blob["standalone_map"]
    out: dict[str, set[int]] = {c: set() for c in comp_names}
    for (cname, snum), is_standalone in smap.items():
        if is_standalone and cname in out:
            out[cname].add(snum)
    return out, True


def jaccard(s_v: set, s_r: set) -> float:
    """Standard Jaccard with vacuous-agreement convention (J=1.0 when both empty)."""
    union = s_v | s_r
    if not union:
        return 1.0
    return len(s_v & s_r) / len(union)


def symdiff(s_v: set, s_r: set) -> int:
    return len(s_v ^ s_r)


def rollup_dataset(
    per_comp_J: dict[str, float],
    per_comp_D: dict[str, int],
    per_comp_regex_size: dict[str, int],
    n_components_J_skipped: int = 0,
) -> dict[str, Any]:
    """Roll up per-component J/D into dataset-level stats.

    `per_comp_J` may exclude denominator-aware-skipped cells (Plan 06-07); their
    count is surfaced as `n_components_J_skipped` for the report. `per_comp_D`
    retains ALL components (D is never skipped — D is well-defined when
    |S_baseline|=0). `n_components` counts components present in per_comp_D so
    rollups across baselines remain comparable.
    """
    d_comps = list(per_comp_D.keys())
    j_comps = list(per_comp_J.keys())
    if not d_comps:
        return {
            "min_jaccard_per_comp": 1.0,
            "mean_jaccard_weighted": 1.0,
            "count_components_with_J<0.5": 0,
            "max_symmetric_diff": 0,
            "n_components": 0,
            "n_components_J_skipped": n_components_J_skipped,
        }
    if j_comps:
        weights = [max(1, per_comp_regex_size[c]) for c in j_comps]
        weight_sum = sum(weights)
        min_j = min(per_comp_J.values())
        mean_j = sum(per_comp_J[c] * w for c, w in zip(j_comps, weights)) / weight_sum
        count_low_j = sum(1 for j in per_comp_J.values() if j < 0.5)
    else:
        # All components had their J skipped — report vacuous-agreement values.
        min_j = 1.0
        mean_j = 1.0
        count_low_j = 0
    return {
        "min_jaccard_per_comp": min_j,
        "mean_jaccard_weighted": mean_j,
        "count_components_with_J<0.5": count_low_j,
        "max_symmetric_diff": max(per_comp_D.values()),
        "n_components": len(d_comps),
        "n_components_J_skipped": n_components_J_skipped,
    }


def apply_drop_rule(per_ds_rollup: dict[str, dict]) -> tuple[bool, list[str]]:
    """Mechanically applies the RESEARCH.md drop rule. Returns (drop?, reasons).

    Thresholds are module constants (HARD_TIER_MIN_J, MAX_SYM_DIFF, HARD_TIER_PCT_LOW_J)
    — no per-call overrides (T-06-03-01: prevents threshold drift).
    """
    reasons: list[str] = []
    for ds in HARD_TIER:
        r = per_ds_rollup.get(ds)
        if not r:
            continue
        if r["min_jaccard_per_comp"] < HARD_TIER_MIN_J:
            reasons.append(
                f"{ds}: min_jaccard_per_comp={r['min_jaccard_per_comp']:.3f} "
                f"< {HARD_TIER_MIN_J} (hard-tier Jaccard floor)"
            )
        if r["max_symmetric_diff"] > MAX_SYM_DIFF:
            reasons.append(
                f"{ds}: max_symmetric_diff={r['max_symmetric_diff']} "
                f"> {MAX_SYM_DIFF} (catastrophic per-comp divergence)"
            )
        pct = r["count_components_with_J<0.5"] / max(1, r["n_components"])
        if pct > HARD_TIER_PCT_LOW_J:
            reasons.append(
                f"{ds}: count_components_with_J<0.5={r['count_components_with_J<0.5']}/"
                f"{r['n_components']} = {pct:.1%} > {HARD_TIER_PCT_LOW_J:.0%} "
                f"(widespread low-Jaccard)"
            )
    # max_symmetric_diff is dataset-agnostic in the rule text ("any (comp, ds) has D > 10").
    # We check non-hard-tier datasets too:
    for ds, r in per_ds_rollup.items():
        if ds in HARD_TIER:
            continue
        if r["max_symmetric_diff"] > MAX_SYM_DIFF:
            reasons.append(
                f"{ds}: max_symmetric_diff={r['max_symmetric_diff']} "
                f"> {MAX_SYM_DIFF} (catastrophic per-comp divergence — any dataset)"
            )
    return (len(reasons) > 0, reasons)


def instantiate_variant(cls):
    """Instantiate a sibling linker with the project-default backend.

    The variants' __init__ accepts (backend, model, checkpoint_fallback, checkpoint_fallback_model);
    all defaults are LLMBackend.CLAUDE with the sonnet model — consistent with run_ablation.py.
    """
    return cls()


variant_classes = {
    "s_linker13g_pre": SLinker13gPre,
    "s_linker13g_sem": SLinker13gSem,
    "s_linker13g_pre_alias": SLinker13gPreAlias,
    "s_linker13g_sem_alias": SLinker13gSemAlias,
    "s_linker13g_pre_full": SLinker13gPreFull,
    "s_linker13g_sem_full": SLinker13gSemFull,
}


# Sentinel used to detect whether the user explicitly passed --output. When the
# user did NOT pass --output AND --baseline != "regex", we route the default
# output to the Plan 06-07 alias-aware path.
_OUTPUT_DEFAULT_SENTINEL = object()


def _resolve_baseline_anchor(
    baseline: str,
    sentences: list,
    comp_names: list[str],
    text_path: Path,
) -> tuple[dict[str, set[int]], str]:
    """Returns (S_baseline_by_comp, baseline_label).

    `baseline` is one of: "regex", "pure-llm-pre", "pure-llm-sem".
    """
    if baseline == "regex":
        return compute_anchor_set(regex_predicate, sentences, comp_names), "regex"
    if baseline == "pure-llm-pre":
        S, _ = load_pure_llm_baseline_anchor_set(
            "s_linker13g_pre", text_path, comp_names
        )
        return S, "pure-llm:s_linker13g_pre"
    if baseline == "pure-llm-sem":
        S, _ = load_pure_llm_baseline_anchor_set(
            "s_linker13g_sem", text_path, comp_names
        )
        return S, "pure-llm:s_linker13g_sem"
    raise ValueError(f"Unknown baseline: {baseline!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["s_linker13g_pre", "s_linker13g_sem"],
        choices=list(variant_classes.keys()),
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
    )
    ap.add_argument(
        "--output",
        default=_OUTPUT_DEFAULT_SENTINEL,
        help="Output JSON path. Defaults to ablation_ext01_diff.json for --baseline regex, "
             "ablation_ext01_diff_alias.json for pure-LLM baselines.",
    )
    ap.add_argument(
        "--baseline",
        choices=["regex", "pure-llm-pre", "pure-llm-sem"],
        default="regex",
        help="Comparison anchor. 'regex' is the Plan 06-03 protocol (drop-decision-gating). "
             "'pure-llm-*' re-anchors against the rejected-baseline cached pickles (Plan 06-04, D-09).",
    )
    ap.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore standalone_map.pkl caches and re-invoke the LLM.",
    )
    args = ap.parse_args()

    # Resolve --output default based on --baseline mode (Plan 06-07).
    if args.output is _OUTPUT_DEFAULT_SENTINEL:
        if args.baseline == "regex":
            args.output = str(REPO_ROOT / "results/ablation_results/ablation_ext01_diff.json")
        else:
            args.output = str(REPO_ROOT / "results/ablation_results/ablation_ext01_diff_alias.json")

    # Pure-LLM-baseline self-vs-self pairs are skipped (would always give J=1.0).
    baseline_self_variant = {
        "pure-llm-pre": "s_linker13g_pre",
        "pure-llm-sem": "s_linker13g_sem",
    }.get(args.baseline)

    matrix: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline": args.baseline,
        "denominator_aware_skip": DENOMINATOR_AWARE_J_SKIP,
        "thresholds": {
            "hard_tier_min_jaccard": HARD_TIER_MIN_J,
            "max_symmetric_diff": MAX_SYM_DIFF,
            "hard_tier_pct_low_jaccard": HARD_TIER_PCT_LOW_J,
            "hard_tier_datasets": sorted(HARD_TIER),
        },
        "variants": {},
    }

    for vname in args.variants:
        if vname == baseline_self_variant:
            print(f"  [skip self-vs-self] {vname} vs {args.baseline}")
            continue
        cls = variant_classes[vname]
        per_ds: dict[str, Any] = {}
        for ds in args.datasets:
            text_path = DATASETS[ds]["text"]
            repo_path = DATASETS[ds]["model"]
            if not text_path.exists() or not repo_path.exists():
                raise FileNotFoundError(
                    f"Dataset assets missing: {text_path} / {repo_path}. "
                    f"Verify BENCHMARK_BASE={BENCHMARK_BASE} resolves correctly."
                )

            sentences = load_sentences(str(text_path))
            components = parse_pcm_repository(str(repo_path))
            comp_names = [c.name for c in components]

            print(f"\n=== [{vname}] {ds} (baseline={args.baseline}) ===")
            print(f"    {len(components)} components, {len(sentences)} sentences")

            # Baseline (regex = free; pure-llm-* = cached pickle, also free)
            S_b_by_comp, baseline_label = _resolve_baseline_anchor(
                args.baseline, sentences, comp_names, text_path
            )
            b_total = sum(len(v) for v in S_b_by_comp.values())
            print(f"    baseline ({baseline_label}) anchor pairs: {b_total}")

            # Variant (LLM, cached)
            inst = instantiate_variant(cls)
            S_v_by_comp, from_cache, llm_map_size = variant_anchor_set(
                inst, sentences, components, text_path,
                force_recompute=args.force_recompute,
            )
            v_total = sum(len(v) for v in S_v_by_comp.values())
            print(f"    variant anchor pairs (true=standalone): {v_total} "
                  f"(map size {llm_map_size}, from_cache={from_cache})")

            # Denominator-aware skip set: components with |S_baseline[comp]| == 0.
            skipped_comps: set[str] = {
                c for c in comp_names if len(S_b_by_comp.get(c, set())) == 0
            }

            per_comp_J_full = {
                c: jaccard(S_v_by_comp.get(c, set()), S_b_by_comp.get(c, set()))
                for c in comp_names
            }
            per_comp_D = {
                c: symdiff(S_v_by_comp.get(c, set()), S_b_by_comp.get(c, set()))
                for c in comp_names
            }
            per_comp_baseline_size = {c: len(S_b_by_comp.get(c, set())) for c in comp_names}
            per_comp_variant_size = {c: len(S_v_by_comp.get(c, set())) for c in comp_names}

            # Build the J-dict used for rollup — excluding skipped cells when
            # DENOMINATOR_AWARE_J_SKIP=True. D is NEVER skipped.
            if DENOMINATOR_AWARE_J_SKIP:
                per_comp_J_for_rollup = {
                    c: per_comp_J_full[c] for c in comp_names if c not in skipped_comps
                }
                weights_for_rollup = {
                    c: per_comp_baseline_size[c]
                    for c in comp_names
                    if c not in skipped_comps
                }
                n_skipped = len(skipped_comps)
            else:
                per_comp_J_for_rollup = dict(per_comp_J_full)
                weights_for_rollup = dict(per_comp_baseline_size)
                n_skipped = 0

            per_ds[ds] = {
                "per_component": {
                    c: {
                        "J": per_comp_J_full[c],
                        "D": per_comp_D[c],
                        "baseline_size": per_comp_baseline_size[c],
                        "variant_size": per_comp_variant_size[c],
                        "J_skipped": (c in skipped_comps and DENOMINATOR_AWARE_J_SKIP),
                    }
                    for c in comp_names
                },
                "rollup": rollup_dataset(
                    per_comp_J_for_rollup,
                    per_comp_D,
                    weights_for_rollup,
                    n_components_J_skipped=n_skipped,
                ),
                "llm_pair_count": llm_map_size,
                "from_cache": from_cache,
                "baseline_label": baseline_label,
                "skipped_components": sorted(skipped_comps),
            }

        drop, reasons = apply_drop_rule({ds: per_ds[ds]["rollup"] for ds in per_ds})
        matrix["variants"][vname] = {
            "per_dataset": per_ds,
            "drop": drop,
            "drop_reasons": reasons,
        }
        print(f"\n[{vname}] drop={drop} reasons={reasons or '(none)'}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(matrix, f, indent=2, default=list)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
