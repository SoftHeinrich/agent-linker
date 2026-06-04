"""Per-variant F1 computer — uses cached 17f phase data + minimal fresh LLM calls.

Computes each variant's predicted F1 by composing:
  • cached layer pkls from 17f's openai run (unchanged phases)
  • parsed coref LLM responses from the 17f calls.json (for 18d's alias-aware gate)
  • ONE fresh LLM run per dataset: twopass-on-coref (for 18b's new validation)

Cleanly avoids re-running unchanged phases (Phase 1–3, framing extraction,
entity twopass) which would have replayed identical cached results anyway.

Total cost: ~$0.20 (10 datasets × 2 twopass batches ≈ 100 cached + 10 live calls).

Usage:
    cd approach/
    LLM_BACKEND=openai OPENAI_API_KEY=$key OPENAI_MODEL_NAME=gpt-5.4 \\
        OPENAI_SERVICE_TIER=flex python compute_variant_chain.py
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


def prf(keys, gold):
    tp = len(keys & gold)
    fp = len(keys - gold)
    p = tp / max(1, tp + fp)
    r = tp / max(1, len(gold))
    f1 = 2 * p * r / max(1e-9, p + r)
    return tp, fp, p, r, f1


def setup_linker(text_path, model_path, backend):
    from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
    from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
    from llm_sad_sam.linkers.experimental.s_linker18b import SLinker18b

    components = parse_pcm_repository(str(model_path))
    sentences = load_sentences(str(text_path))
    sent_map = build_sent_map(sentences)
    linker = SLinker18b(backend=backend)
    return linker, components, sentences, sent_map


def run_18b_coref_validation(linker, layer1, layer4, components, sent_map):
    """Fresh twopass on cached coref_raw — the only new LLM work needed."""
    linker.model_knowledge = layer1["model_knowledge"]
    linker.doc_knowledge = layer1["doc_knowledge"]
    validated, decisions = linker._validate_coref_links(
        layer4["coref_raw"], sent_map, components)
    return {(lk.sentence_number, lk.component_id) for lk in validated}, decisions


def apply_alias_aware_gate(layer4, doc_aliases):
    """Recompute coref_raw using 18d's structural alias-aware antecedent gate.

    For cached coref_metadata: each entry has antecedent_text and via_alias.
    18d's gate accepts iff antecedent_text contains either the canonical
    component name (standalone) OR any known alias of the component.

    Since 17f's gate ALREADY accepted these (they're in coref_metadata), and
    our new gate is at least as permissive for these candidates, all cached
    metadata entries are kept under 18d. 18d may also accept additional
    candidates the LLM had via_alias=False on but where an alias is actually
    present — but those weren't recorded in 17f's metadata (LLM rejected them
    via the standalone-mention check). We treat 18d_coref_raw ≈ 17f_coref_raw
    as a conservative approximation; the recall gain from 18d (if any) is
    bounded above by the bypass_only candidates feasibility B identified.
    """
    from llm_sad_sam.linkers.experimental.helper_v3 import has_standalone_mention
    coref_meta = layer4["coref_metadata"]
    kept = set()
    for key, meta in coref_meta.items():
        ant_text = meta.get("antecedent_text", "")
        # Resolve component name from the cached SadSamLink
        comp_name = None
        for lk in layer4["coref_raw"]:
            if (lk.sentence_number, lk.component_id) == key:
                comp_name = lk.component_name
                break
        if comp_name is None:
            continue
        # 18d gate: standalone OR any alias of comp_name in ant_text
        if has_standalone_mention(comp_name, ant_text):
            kept.add(key)
            continue
        for alias, entry in doc_aliases.items():
            if entry.component != comp_name:
                continue
            if has_standalone_mention(alias, ant_text):
                kept.add(key)
                break
            if re.search(rf'\b{re.escape(alias)}\b', ant_text, re.IGNORECASE):
                kept.add(key)
                break
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    ap.add_argument("--backend", default=os.environ.get("LLM_BACKEND", "openai"))
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    from llm_sad_sam.llm_client import LLMBackend
    backend_map = {
        "claude": LLMBackend.CLAUDE, "openai": LLMBackend.OPENAI,
        "checkpoint": LLMBackend.CHECKPOINT,
    }
    backend = backend_map.get(args.backend, LLMBackend.OPENAI)
    os.environ["LLM_BACKEND"] = args.backend

    print(f"Variant-chain computer | backend={backend.value}")
    print(f"Datasets: {args.datasets}\n")

    all_results = {}
    for ds in args.datasets:
        if ds not in DATASETS:
            print(f"  skip unknown dataset '{ds}'")
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
        final17f = load_pkl(CACHE/ds/"final.pkl")

        # Cached entity-validated keys (post-Phase 4) and post-4b
        entity_post_4b = {(c.sentence_number, c.component_id)
                          for c in layer3["validated"]}
        entity_pre_4b = {(c.sentence_number, c.component_id)
                         for c in layer3["validated_pre_4b"]}
        # Cached coref (single-pass validation, 17f)
        coref_17f_validated = {(lk.sentence_number, lk.component_id)
                                for lk in layer4["coref_validated"]}

        # ── 18b's coref via twopass: only fresh LLM work ─────────────────────
        print(f"  [18b coref twopass] running on {len(layer4['coref_raw'])} coref candidates...")
        linker, components, sentences, sent_map = setup_linker(text_p, model_p, backend)
        t0 = time.time()
        coref_18b_keys, coref_18b_decisions = run_18b_coref_validation(
            linker, layer1, layer4, components, sent_map)
        print(f"    18b coref validated: {len(coref_18b_keys)} / {len(layer4['coref_raw'])}"
              f"  ({round(time.time()-t0,1)}s)")

        # ── 18d's alias-aware gate (purely cached data) ──────────────────────
        doc_aliases = layer1["doc_knowledge"].aliases
        coref_18d_raw_keys = apply_alias_aware_gate(layer4, doc_aliases)
        # 18d uses twopass too (inherits from 18b), so validated set is the
        # twopass-approved subset of 18d's coref_raw. Approximation: intersect
        # 18b's twopass approvals with 18d's structural raw set.
        coref_18d_keys = coref_18b_keys & coref_18d_raw_keys

        # ── Compute final keys per variant ───────────────────────────────────
        variants = {
            "s_linker17f": entity_post_4b | coref_17f_validated,
            "s_linker18a": entity_post_4b | coref_17f_validated,            # F: no behavior change
            "s_linker18b": entity_post_4b | coref_18b_keys,                  # +E
            "s_linker18c": entity_pre_4b  | coref_18b_keys,                  # +C drops 4b
            "s_linker18d": entity_pre_4b  | coref_18d_keys,                  # +B-refactor
            "s_linker18":  entity_pre_4b  | coref_18d_keys,                  # +A enum (no change)
        }
        per_variant = {}
        for v, keys in variants.items():
            tp, fp, p, r, f1 = prf(keys, gold)
            per_variant[v] = {
                "tp": tp, "fp": fp, "p": round(p, 4),
                "r": round(r, 4), "f1": round(f1, 4),
                "n_keys": len(keys),
            }
            print(f"    {v:<14} TP={tp:>3} FP={fp:<3} P={p:.3f} R={r:.3f} F1={f1:.3f}")

        all_results[ds] = {
            "gold": len(gold),
            "entity_post_4b": len(entity_post_4b),
            "entity_pre_4b": len(entity_pre_4b),
            "coref_17f_validated": len(coref_17f_validated),
            "coref_18b_validated": len(coref_18b_keys),
            "coref_18d_validated": len(coref_18d_keys),
            "variants": per_variant,
        }

    # ── Aggregate ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print("AGGREGATE: macro F1 per variant")
    print(f"{'='*100}")
    variant_names = ["s_linker17f", "s_linker18a", "s_linker18b",
                     "s_linker18c", "s_linker18d", "s_linker18"]
    print(f"{'Variant':<14}" + "".join(f"{ds:>14}" for ds in all_results)
          + f"  {'macro F1':>10}  {'micro TP/FP':>12}")
    print("-" * (14 + 14 * len(all_results) + 26))
    for v in variant_names:
        f1s = []
        macro_tp = 0; macro_fp = 0
        cells = []
        for ds in all_results:
            r = all_results[ds]["variants"][v]
            cells.append(f"{r['tp']}/{r['fp']} {r['f1']:.3f}".rjust(14))
            f1s.append(r["f1"])
            macro_tp += r["tp"]; macro_fp += r["fp"]
        macro = sum(f1s) / max(1, len(f1s))
        print(f"{v:<14}" + "".join(cells)
              + f"  {macro:>10.4f}  {macro_tp}/{macro_fp:<8}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults: {args.out_json}")


if __name__ == "__main__":
    main()
