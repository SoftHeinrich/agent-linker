#!/usr/bin/env python3
"""Spike 004 — Stage 1/2: replay ONLY the validation gates on cached candidates.

Loads the frozen nothink cell (layer1/3/4), reconstructs evidence bundles, and re-runs
the entity twopass + coref validators with a chosen validator class at effort 0
(CLAUDE_DISABLE_THINKING=1). Everything upstream is cached — no re-extraction. Scores the
reassembled final links vs gold and logs token cost.

Usage (from repo root):
  python .planning/spikes/004-nogap-validator-ab/harness/replay.py \
      --run run1 --datasets teammates --label layered
  # validator class: --validator layered (default) | baseline (reuses cached prompt)
  # thinking:        --thinking on|off  (default off = effort 0)
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_ROOT_DEFAULT = os.path.join(
    REPO, ".planning", "spikes", "004-nogap-validator-ab", "results")


def reconstruct_bundles(layer3):
    from llm_sad_sam.linkers.experimental.s_linker20_union import EvidenceBundle
    out = {}
    for key, d in layer3["evidence_bundles"].items():
        if isinstance(d, dict):
            out[key] = EvidenceBundle(**d)
        else:
            out[key] = d  # already a dataclass
    return out


def replay_cell(linker, run, dataset, cache_root=None, cache_backend="claude"):
    import cache_io as C
    from llm_sad_sam.core.data_types_v2 import SadSamLink

    cell = C.load_cell(cache_root or C.NOTHINK_ROOT, run, dataset, cache_backend)
    bench = C.load_benchmark(dataset)
    components, sent_map, gold, id_to_name = (
        bench["components"], bench["sent_map"], bench["gold"], bench["id_to_name"])

    linker.model_knowledge = cell["layer1"]["model_knowledge"]
    linker.doc_knowledge = cell["layer1"]["doc_knowledge"]
    linker._llm_calls.clear()

    # --- entity twopass on cached candidates ---
    candidates = cell["layer3"]["candidates"]
    bundles = reconstruct_bundles(cell["layer3"])
    validated, entity_decisions = linker._validate_with_evidence(
        candidates, bundles, components, sent_map,
        p1_tag="phase_4_twopass_p1", p2_tag="phase_4_twopass_p2",
        stage_label="entity")

    # --- coref single-pass on cached raw coref ---
    coref_raw = cell["layer4"]["coref_raw"]
    coref_validated, coref_decisions = linker._validate_coref_links(
        coref_raw, sent_map, components)

    # --- Phase 6 dedup merge (entity-first) ---
    entity_links = [SadSamLink(c.sentence_number, c.component_id, c.component_name,
                               source="entity") for c in validated]
    all_links = entity_links + coref_validated
    seen, final = set(), []
    for lk in all_links:
        k = (lk.sentence_number, lk.component_id)
        if k not in seen:
            seen.add(k)
            final.append(lk)

    # --- score ---
    pred = {(lk.sentence_number, lk.component_id) for lk in final}
    by_key = {(lk.sentence_number, lk.component_id): lk for lk in final}
    m = C.score_pairs(pred, gold)

    fp_by_source = defaultdict(int)
    fp_details = []
    for s, c in sorted(pred - gold):
        lk = by_key[(s, c)]
        fp_by_source[lk.source] += 1
        sent = sent_map.get(s)
        fp_details.append({"sentence": s, "component": id_to_name.get(c, c),
                           "source": lk.source, "text": sent.text[:120] if sent else ""})
    fn_details = []
    for s, c in sorted(gold - pred):
        sent = sent_map.get(s)
        name = id_to_name.get(c, c)
        fn_details.append({"sentence": s, "component": name,
                           "name_in_text": (name.lower() in sent.text.lower()) if sent else False})

    # --- token cost from trace ---
    calls = list(linker._llm_calls)
    tok = defaultdict(int)
    for r in calls:
        u = r.get("token_usage") or {}
        tok["prompt"] += u.get("prompt_tokens", 0)
        tok["completion"] += u.get("completion_tokens", 0)
    n_entity_cand = len(candidates)
    n_coref_cand = len(coref_raw)

    return {
        "run": run, "dataset": dataset,
        "P": m["P"], "R": m["R"], "F1": m["F1"],
        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "n_links": len(final),
        "fp_by_source": dict(fp_by_source),
        "fp_details": fp_details, "fn_details": fn_details,
        "n_entity_candidates": n_entity_cand, "n_coref_candidates": n_coref_cand,
        "n_llm_calls": len(calls),
        "tokens": dict(tok),
        "entity_validated": len(validated), "coref_validated": len(coref_validated),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run1")
    ap.add_argument("--datasets", default="teammates")
    ap.add_argument("--validator", choices=["layered", "baseline"], default="layered")
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--rubric", default="v2", help="layered rubric version (see layered_validator.RUBRICS)")
    ap.add_argument("--coref-skeptic", action="store_true", help="Mode 4: skeptic pass on coref survivors")
    ap.add_argument("--backend", choices=["claude", "openai"], default="claude")
    ap.add_argument("--model", default="", help="openai model id override (e.g. gpt-5-mini, gpt-5-nano); default gpt-5.4")
    ap.add_argument("--effort", default="", help="openai reasoning effort (none|low|medium|high); empty = baseline no-reasoning (temperature path)")
    ap.add_argument("--cache-root", default=None, help="phase_cache sweep root (default: Sonnet nothink). For gpt use results/v2.6.5_s20union/gpt")
    ap.add_argument("--cache-backend", default=None, help="cache backend subdir (claude|openai); default follows --backend")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    args = ap.parse_args()

    from llm_sad_sam.llm_client import LLMBackend
    if args.backend == "openai":
        os.environ["LLM_BACKEND"] = "openai"
        os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.4")
        if args.effort:
            os.environ["OPENAI_REASONING_EFFORT"] = args.effort
        else:
            os.environ.pop("OPENAI_REASONING_EFFORT", None)  # baseline no-reasoning
        be = LLMBackend.OPENAI
        cache_root = args.cache_root or __import__("cache_io").GPT_ROOT
        cache_backend = args.cache_backend or "openai"
    else:
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        os.environ.setdefault("LLM_BACKEND", "claude")
        if args.thinking == "off":
            os.environ["CLAUDE_DISABLE_THINKING"] = "1"
        else:
            os.environ.pop("CLAUDE_DISABLE_THINKING", None)
        be = LLMBackend.CLAUDE
        cache_root = args.cache_root
        cache_backend = args.cache_backend or "claude"

    os.environ["SPIKE_RUBRIC"] = args.rubric
    if args.coref_skeptic:
        os.environ["SPIKE_CORE_SKEPTIC"] = "1"
    else:
        os.environ.pop("SPIKE_CORE_SKEPTIC", None)

    if args.validator == "layered":
        from layered_validator import LayeredValidator as VClass
    else:
        from llm_sad_sam.linkers.experimental.s_linker20_union import SLinker20Union as VClass

    rub = f"_{args.rubric}" if args.validator == "layered" else ""
    sk = "_sk" if args.coref_skeptic else ""
    cond = args.backend if args.backend == "openai" else args.thinking + "think"
    eff = f"_{args.effort}" if (args.backend == "openai" and args.effort) else ""
    label = args.label or f"{args.validator}{rub}{sk}_{cond}{eff}"
    datasets = [d for d in args.datasets.replace(",", " ").split() if d]

    linker = VClass(backend=be)
    out_dir = os.path.join(args.out_root, label, args.run)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== replay [{label}] run={args.run} datasets={datasets} "
          f"backend={args.backend} cache={cache_root or 'sonnet-nothink'} ===")
    for ds in datasets:
        t0 = time.time()
        res = replay_cell(linker, args.run, ds, cache_root=cache_root, cache_backend=cache_backend)
        res["elapsed_s"] = round(time.time() - t0, 1)
        with open(os.path.join(out_dir, f"{ds}.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"  {ds:14s} F1={res['F1']*100:5.1f}  P={res['P']*100:5.1f} R={res['R']*100:5.1f}"
              f"  TP={res['tp']} FP={res['fp']} (E{res['fp_by_source'].get('entity',0)}/"
              f"C{res['fp_by_source'].get('coreference',0)}) FN={res['fn']}"
              f"  calls={res['n_llm_calls']} out_tok={res['tokens'].get('completion',0)}"
              f"  {res['elapsed_s']}s")


if __name__ == "__main__":
    main()
