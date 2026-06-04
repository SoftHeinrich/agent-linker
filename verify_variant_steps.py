"""Per-variant single-step verifier — each cleanup tested in isolation.

For each variant, only the step it modifies is re-executed on cached input.
No composition with other variants, no e2e re-runs. Uses LLM checkpoint
backend with openai fallback — cached prompts replay free, only new
prompts incur LLM cost.

Tests:
  18a : run SLinker18a._validate_with_evidence on cached union pool.
        Compare validated set vs 17f's validated. Report delta TP/FP.
        (Verifies cleanup F = drop generic-filter.)

  18b : run SLinker18b._validate_coref_links on cached coref_raw.
        Compare validated set vs 17f's coref_validated. Report delta TP/FP.
        (Verifies cleanup E = unify coref validation with entity twopass.)

  18c : compare cached layer3.validated_pre_4b to layer3.validated.
        No LLM calls. Report delta TP/FP from removing Phase 4b.
        (Verifies cleanup C = drop Phase 4b.)

  18d : apply alias-aware antecedent gate to cached coref_metadata.
        No LLM calls. Compare accepted set vs 17f's coref_raw.
        (Verifies cleanup B-refactor = alias-aware antecedent check.)

  18  : compare SLinker18._classify_mention output strings to
        SLinker17f._classify_mention on cached sentences.
        No LLM calls. Report any string differences.
        (Verifies cleanup A = enum refactor preserves behavior.)

Usage:
    cd approach/
    LLM_BACKEND=checkpoint CHECKPOINT_FALLBACK=openai OPENAI_API_KEY=$key \\
        OPENAI_MODEL_NAME=gpt-5.4 python verify_variant_steps.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

ROOT = Path(__file__).parent
BENCH = ROOT / "../ardoco/core/tests-base/src/main/resources/benchmark"
CACHE = ROOT / "results/phase_cache/s_linker17f/openai"

DATASETS = {
    "mediastore":    (BENCH/"mediastore/text_2016/mediastore.txt",
                      BENCH/"mediastore/model_2016/pcm/ms.repository",
                      BENCH/"mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"),
    "teastore":      (BENCH/"teastore/text_2020/teastore.txt",
                      BENCH/"teastore/model_2020/pcm/teastore.repository",
                      BENCH/"teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"),
    "teammates":     (BENCH/"teammates/text_2021/teammates.txt",
                      BENCH/"teammates/model_2021/pcm/teammates.repository",
                      BENCH/"teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "bigbluebutton": (BENCH/"bigbluebutton/text_2021/bigbluebutton.txt",
                      BENCH/"bigbluebutton/model_2021/pcm/bbb.repository",
                      BENCH/"bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "jabref":        (BENCH/"jabref/text_2021/jabref.txt",
                      BENCH/"jabref/model_2021/pcm/jabref.repository",
                      BENCH/"jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
}


def load_gold(p):
    g = set()
    for r in csv.DictReader(open(p)):
        cid = r.get("modelElementID", "").strip()
        sn = r.get("sentence", "").strip()
        if cid and sn:
            g.add((int(sn), cid))
    return g


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def _make_linker(VariantClass, backend, layer1):
    from llm_sad_sam.linkers.experimental.s_linker17f import SLinker17f as _

    linker = VariantClass(
        backend=backend,
        # If running checkpoint backend, fall back to openai for misses.
        checkpoint_fallback=os.environ.get("CHECKPOINT_FALLBACK", "openai"),
        checkpoint_fallback_model=os.environ.get("CHECKPOINT_FALLBACK_MODEL", "gpt-5.4"),
    )
    linker.model_knowledge = layer1["model_knowledge"]
    linker.doc_knowledge = layer1["doc_knowledge"]
    return linker


def _load_dataset_inputs(text_p, model_p):
    from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
    from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
    components = parse_pcm_repository(str(model_p))
    sentences = load_sentences(str(text_p))
    sent_map = build_sent_map(sentences)
    return components, sentences, sent_map


def keyset(items_or_decisions):
    """Accept a list of candidates/links OR a dict of decisions; return key set."""
    if isinstance(items_or_decisions, dict):
        return {k for k, v in items_or_decisions.items() if v.get("approved")}
    return {(x.sentence_number, x.component_id) for x in items_or_decisions}


def tp_fp(keys, gold):
    return len(keys & gold), len(keys - gold)


# ─────────────────────────────────────────────────────────────────────────────
# 18a — run _validate_with_evidence on cached union pool
# ─────────────────────────────────────────────────────────────────────────────
def verify_18a(ds, gold, layer1, layer3, components, sent_map, backend):
    from llm_sad_sam.linkers.experimental.s_linker18a import SLinker18a

    linker = _make_linker(SLinker18a, backend, layer1)
    candidates = layer3["candidates"]
    # Rebuild evidence bundles (17f did this in `link()` between Phase 3 and 4).
    bundles = {
        (c.sentence_number, c.component_id): linker._build_evidence_bundle(c, sent_map)
        for c in candidates
    }
    validated_18a, _ = linker._validate_with_evidence(candidates, bundles, components, sent_map)
    keys_18a = keyset(validated_18a)
    keys_17f = keyset(layer3["validated_pre_4b"])  # 17f's pre-Phase-4b set is the directly comparable validated set

    same = keys_18a == keys_17f
    only_18a = keys_18a - keys_17f
    only_17f = keys_17f - keys_18a
    tp_18a, fp_18a = tp_fp(keys_18a, gold)
    tp_17f, fp_17f = tp_fp(keys_17f, gold)

    return {
        "validated_17f_pre_4b": len(keys_17f),
        "validated_18a": len(keys_18a),
        "identical_keys": same,
        "only_in_18a": [list(k) for k in only_18a],
        "only_in_17f": [list(k) for k in only_17f],
        "tp_17f": tp_17f, "fp_17f": fp_17f,
        "tp_18a": tp_18a, "fp_18a": fp_18a,
        "delta_tp": tp_18a - tp_17f, "delta_fp": fp_18a - fp_17f,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 18b — run _validate_coref_links on cached coref_raw
# ─────────────────────────────────────────────────────────────────────────────
def verify_18b(ds, gold, layer1, layer4, components, sent_map, backend):
    from llm_sad_sam.linkers.experimental.s_linker18b import SLinker18b

    linker = _make_linker(SLinker18b, backend, layer1)
    coref_raw = layer4["coref_raw"]
    validated_18b, _ = linker._validate_coref_links(coref_raw, sent_map, components)
    keys_18b = keyset(validated_18b)
    keys_17f = keyset(layer4["coref_validated"])

    tp_18b, fp_18b = tp_fp(keys_18b, gold)
    tp_17f, fp_17f = tp_fp(keys_17f, gold)
    return {
        "coref_raw": len(coref_raw),
        "coref_validated_17f": len(keys_17f),
        "coref_validated_18b": len(keys_18b),
        "only_in_18b": [list(k) for k in (keys_18b - keys_17f)],
        "only_in_17f": [list(k) for k in (keys_17f - keys_18b)],
        "tp_17f": tp_17f, "fp_17f": fp_17f,
        "tp_18b": tp_18b, "fp_18b": fp_18b,
        "delta_tp": tp_18b - tp_17f, "delta_fp": fp_18b - fp_17f,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 18c — pure pkl comparison: drop Phase 4b means use validated_pre_4b
# ─────────────────────────────────────────────────────────────────────────────
def verify_18c(ds, gold, layer3):
    keys_post_4b = keyset(layer3["validated"])
    keys_pre_4b = keyset(layer3["validated_pre_4b"])
    p4b_dropped = keys_pre_4b - keys_post_4b
    tp_drop, fp_drop = tp_fp(p4b_dropped, gold)
    return {
        "validated_post_4b_17f": len(keys_post_4b),
        "validated_pre_4b_18c": len(keys_pre_4b),
        "phase_4b_dropped": [list(k) for k in p4b_dropped],
        "dropped_tp": tp_drop, "dropped_fp": fp_drop,
        # Delta vs 17f: 18c keeps what 4b dropped → +TP candidates, +FP candidates
        "delta_tp": tp_drop, "delta_fp": fp_drop,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 18d — apply alias-aware antecedent gate (pure structural, no LLM)
# ─────────────────────────────────────────────────────────────────────────────
def verify_18d(ds, gold, layer1, layer4):
    from llm_sad_sam.linkers.experimental.helper_v3 import has_standalone_mention

    doc_aliases = layer1["doc_knowledge"].aliases
    coref_raw = layer4["coref_raw"]
    coref_meta = layer4["coref_metadata"]
    comp_lookup = {(lk.sentence_number, lk.component_id): lk.component_name for lk in coref_raw}

    # 17f accepted = layer4.coref_raw (already filtered by 17f's via_alias-flag gate)
    keys_17f_accepted = set(comp_lookup.keys())

    # 18d's accepted = same candidates where structural gate ALSO accepts.
    # Since 17f already accepted, we want to verify 18d would too. For each
    # cached candidate: does antecedent_text contain canonical name OR any
    # alias of comp_name?
    keys_18d_accepted = set()
    failures = []
    for key, comp_name in comp_lookup.items():
        meta = coref_meta.get(key, {})
        ant_text = meta.get("antecedent_text", "")
        accept = False
        if has_standalone_mention(comp_name, ant_text):
            accept = True
        else:
            for alias, entry in doc_aliases.items():
                if entry.component != comp_name:
                    continue
                if has_standalone_mention(alias, ant_text):
                    accept = True; break
                if re.search(rf'\b{re.escape(alias)}\b', ant_text, re.IGNORECASE):
                    accept = True; break
        if accept:
            keys_18d_accepted.add(key)
        else:
            failures.append({
                "key": list(key), "component": comp_name,
                "antecedent_text": ant_text[:120],
                "antecedent_via_alias_17f_flag": meta.get("antecedent_via_alias"),
            })

    rejected_by_18d = keys_17f_accepted - keys_18d_accepted
    tp_rej, fp_rej = tp_fp(rejected_by_18d, gold)
    return {
        "coref_raw_17f": len(keys_17f_accepted),
        "coref_raw_18d_structural_accept": len(keys_18d_accepted),
        "rejected_by_18d": [list(k) for k in rejected_by_18d],
        "rejected_tp": tp_rej, "rejected_fp": fp_rej,
        "failures_detail": failures,
        # Negative delta = 18d's stricter check loses candidates 17f kept via LLM-flag bypass
        "delta_tp": -tp_rej, "delta_fp": -fp_rej,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 18 — verify _classify_mention enum refactor is behavior-preserving
# ─────────────────────────────────────────────────────────────────────────────
def verify_18(ds, layer1, layer3, components, sent_map):
    from llm_sad_sam.linkers.experimental.s_linker17f import SLinker17f
    from llm_sad_sam.linkers.experimental.s_linker18 import SLinker18

    # Construct two linkers with knowledge primed; we don't need LLM here.
    # _classify_mention only consults model_knowledge.ambiguous_names (none for
    # output of method itself) and doc_knowledge.aliases.
    l17 = SLinker17f.__new__(SLinker17f)
    l17.model_knowledge = layer1["model_knowledge"]
    l17.doc_knowledge = layer1["doc_knowledge"]
    l18 = SLinker18.__new__(SLinker18)
    l18.model_knowledge = layer1["model_knowledge"]
    l18.doc_knowledge = layer1["doc_knowledge"]

    mismatches = []
    for c in layer3["candidates"]:
        sent = sent_map.get(c.sentence_number)
        if not sent:
            continue
        s17 = l17._classify_mention(c.component_name, sent.text)
        s18 = l18._classify_mention(c.component_name, sent.text)
        if s17 != s18:
            mismatches.append({
                "key": [c.sentence_number, c.component_id],
                "component": c.component_name,
                "17f_classification": s17,
                "18_classification": s18,
            })
    return {
        "candidates_checked": len(layer3["candidates"]),
        "mismatches": mismatches,
        "delta_tp": 0, "delta_fp": 0,  # refactor only
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    ap.add_argument("--backend", default=os.environ.get("LLM_BACKEND", "checkpoint"))
    ap.add_argument("--variants", nargs="*",
                    default=["18a", "18b", "18c", "18d", "18"])
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    from llm_sad_sam.llm_client import LLMBackend
    backend_map = {
        "claude": LLMBackend.CLAUDE, "openai": LLMBackend.OPENAI,
        "checkpoint": LLMBackend.CHECKPOINT,
    }
    backend = backend_map.get(args.backend, LLMBackend.CHECKPOINT)
    os.environ["LLM_BACKEND"] = args.backend

    print(f"Per-step variant verifier | backend={backend.value} "
          f"| fallback={os.environ.get('CHECKPOINT_FALLBACK', 'openai')}\n")

    all_results = {}
    for ds in args.datasets:
        if ds not in DATASETS:
            continue
        text_p, model_p, gold_p = DATASETS[ds]
        if not (CACHE/ds/"layer1.pkl").exists():
            print(f"  skip {ds}: no cache at {CACHE/ds}")
            continue

        print(f"\n{'='*70}\n  {ds}\n{'='*70}")
        gold = load_gold(gold_p)
        layer1 = load_pkl(CACHE/ds/"layer1.pkl")
        layer3 = load_pkl(CACHE/ds/"layer3.pkl")
        layer4 = load_pkl(CACHE/ds/"layer4.pkl")
        components, sentences, sent_map = _load_dataset_inputs(text_p, model_p)
        ds_results = {"gold": len(gold)}

        if "18a" in args.variants:
            print(f"  [18a] run _validate_with_evidence on union pool ({len(layer3['candidates'])} cands)...")
            t0 = time.time()
            r = verify_18a(ds, gold, layer1, layer3, components, sent_map, backend)
            print(f"    17f→18a: validated {r['validated_17f_pre_4b']}→{r['validated_18a']}, "
                  f"ΔTP={r['delta_tp']:+d} ΔFP={r['delta_fp']:+d} ({round(time.time()-t0,1)}s)")
            ds_results["18a"] = r

        if "18b" in args.variants:
            print(f"  [18b] run _validate_coref_links on coref_raw ({len(layer4['coref_raw'])} cands)...")
            t0 = time.time()
            r = verify_18b(ds, gold, layer1, layer4, components, sent_map, backend)
            print(f"    17f→18b: coref-validated {r['coref_validated_17f']}→{r['coref_validated_18b']}, "
                  f"ΔTP={r['delta_tp']:+d} ΔFP={r['delta_fp']:+d} ({round(time.time()-t0,1)}s)")
            ds_results["18b"] = r

        if "18c" in args.variants:
            print(f"  [18c] compare validated_pre_4b vs validated (no LLM)...")
            r = verify_18c(ds, gold, layer3)
            print(f"    17f→18c: 4b dropped {len(r['phase_4b_dropped'])} candidates "
                  f"({r['dropped_tp']} TP, {r['dropped_fp']} FP). "
                  f"Δentity_TP={r['delta_tp']:+d} ΔFP={r['delta_fp']:+d}")
            ds_results["18c"] = r

        if "18d" in args.variants:
            print(f"  [18d] alias-aware antecedent gate on cached coref_raw...")
            r = verify_18d(ds, gold, layer1, layer4)
            print(f"    17f→18d: rejected {len(r['rejected_by_18d'])} of {r['coref_raw_17f']} "
                  f"({r['rejected_tp']} TP, {r['rejected_fp']} FP would be lost). "
                  f"Δcoref_TP={r['delta_tp']:+d} ΔFP={r['delta_fp']:+d}")
            ds_results["18d"] = r

        if "18" in args.variants:
            print(f"  [18] enum refactor — compare _classify_mention outputs (no LLM)...")
            r = verify_18(ds, layer1, layer3, components, sent_map)
            print(f"    classified {r['candidates_checked']} candidates, "
                  f"{len(r['mismatches'])} mismatches")
            ds_results["18"] = r

        all_results[ds] = ds_results

    # ── Aggregate per-variant deltas ───────────────────────────────────────
    print(f"\n\n{'='*100}")
    print("AGGREGATE per-variant deltas vs 17f (each cleanup tested in isolation)")
    print(f"{'='*100}")
    print(f"{'Cleanup':<8}" + "".join(f"{ds:>16}" for ds in all_results) + "    total ΔTP / ΔFP")
    print("-" * (8 + 16 * len(all_results) + 22))
    for v in ["18a", "18b", "18c", "18d", "18"]:
        if v not in args.variants:
            continue
        cells = []
        total_dtp = 0; total_dfp = 0
        for ds in all_results:
            r = all_results[ds].get(v, {})
            dtp = r.get("delta_tp", 0); dfp = r.get("delta_fp", 0)
            total_dtp += dtp; total_dfp += dfp
            cells.append(f"ΔTP={dtp:+d} ΔFP={dfp:+d}".rjust(16))
        print(f"{v:<8}" + "".join(cells) + f"    ΔTP={total_dtp:+d} ΔFP={total_dfp:+d}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults: {args.out_json}")


if __name__ == "__main__":
    main()
