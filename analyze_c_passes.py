"""5-pass Framing-C experiment: which k-of-N voting threshold is best?

Runs Framing C N times per dataset (default 5), validates ONCE on the union
with Phase 4 + Phase 4b, then post-hoc evaluates every voting threshold
k ∈ {1..N} against gold. Cost: ~N+5 LLM calls per dataset.

When --with-coref is set, also runs Phase 5 (coref) + validates it, and reports
per-k F1 for (C-only ∪ coref). This directly answers "can A and B be removed?"
by comparing (C-union + coref) to the full 17f baseline (A+B+C-intersect+coref).

Usage:
    cd approach/
    LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.4 OPENAI_SERVICE_TIER=flex \\
        python analyze_c_passes.py --with-coref
    # Custom:
    python analyze_c_passes.py --passes 5 --datasets mediastore jabref
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

ROOT = Path(__file__).parent
BENCH = ROOT / "../ardoco/core/tests-base/src/main/resources/benchmark"
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


def run_dataset(name: str, paths, n_passes: int, linker, with_coref: bool = False) -> dict:
    text_path, model_path, gold_path = (str(x) for x in paths)
    gold = load_gold(gold_path)

    print(f"\n{'='*70}\n  {name}  (gold: {len(gold)}, passes: {n_passes})\n{'='*70}")
    t0 = time.time()

    from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
    from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
    from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names

    components = parse_pcm_repository(model_path)
    sentences = load_sentences(text_path)
    name_to_id = {c.name: c.id for c in components}
    sent_map = build_sent_map(sentences)
    comp_names = get_comp_names(components)

    # ── Phase 1: alias discovery (needed for C's mappings) ──────────────────
    # Reuse linker's existing phase 1 — runs model + doc knowledge in parallel
    print(f"  [Phase 1] knowledge acquisition...")
    knowledge = linker._run_parallel({
        "model": lambda: linker._analyze_model(components),
        "doc": lambda: linker._learn_document_knowledge_enriched(sentences, components),
    })
    linker.model_knowledge = knowledge["model"]
    linker.doc_knowledge = knowledge["doc"]
    print(f"    {len(linker.doc_knowledge.aliases)} aliases, "
          f"{len(linker.model_knowledge.ambiguous_names)} ambiguous")

    mappings = [
        f"{term}={entry.component}"
        for term, entry in linker.doc_knowledge.aliases.items()
        if entry.scope == "global"
    ]

    # ── N parallel passes of Framing C ─────────────────────────────────────
    print(f"  [Phase 2] running {n_passes} parallel C passes...")
    passes: list[dict] = [None] * n_passes
    with ThreadPoolExecutor(max_workers=n_passes) as pool:
        futures = {
            pool.submit(
                linker._run_extraction_pass,
                sentences, comp_names, mappings, name_to_id, sent_map,
                pass_label=f"[C{i+1}/{n_passes}] ",
                phase_tag=f"c5_pass{i+1}",
            ): i
            for i in range(n_passes)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            passes[i] = fut.result()
            print(f"    pass {i+1}: {len(passes[i])} candidates")

    # ── Union & vote counts ─────────────────────────────────────────────────
    all_keys: set = set()
    for p in passes:
        all_keys |= set(p.keys())
    vote_counts: dict = {k: 0 for k in all_keys}
    for p in passes:
        for k in p:
            vote_counts[k] += 1
    union_candidates = []
    seen = set()
    for p in passes:
        for k, cand in p.items():
            if k not in seen:
                seen.add(k)
                union_candidates.append(cand)
    print(f"  Union: {len(union_candidates)} unique candidates "
          f"(vote distribution: {sorted({v: sum(1 for x in vote_counts.values() if x == v) for v in range(1, n_passes+1)}.items())})")

    # ── Phase 4 + 4b on the union ───────────────────────────────────────────
    print(f"  [Phase 4] validating union of {len(union_candidates)} candidates...")
    bundles = {
        (c.sentence_number, c.component_id): linker._build_evidence_bundle(c, sent_map)
        for c in union_candidates
    }
    validated_pre_4b, p4_decisions = linker._validate_with_evidence(
        union_candidates, bundles, components, sent_map)
    print(f"    Phase 4 validated: {len(validated_pre_4b)} / {len(union_candidates)}")
    validated, p4b_decisions = linker._codepath_filter(
        validated_pre_4b, sent_map, components)
    print(f"    Phase 4b kept: {len(validated)} / {len(validated_pre_4b)}")
    validated_keys = {(c.sentence_number, c.component_id) for c in validated}

    # ── Optional: Phase 5 (coref) + Phase 5 validation ──────────────────────
    coref_keys: set = set()
    coref_validated_keys: set = set()
    if with_coref:
        print(f"  [Phase 5] running coreference...")
        coref_raw, coref_meta, anaph_snums, terminals = linker._run_coreference(
            sentences, components, name_to_id, sent_map)
        print(f"    coref raw: {len(coref_raw)}")
        coref_val_links, coref_decisions = linker._validate_coref_links(
            coref_raw, sent_map, components)
        coref_keys = {(lk.sentence_number, lk.component_id) for lk in coref_raw}
        coref_validated_keys = {(lk.sentence_number, lk.component_id) for lk in coref_val_links}
        print(f"    coref validated: {len(coref_validated_keys)} / {len(coref_raw)}")

    # ── Compute per-k metrics ───────────────────────────────────────────────
    # For each k threshold, report two regimes:
    #   C-only:  validated keys from the k-of-N C-union pool
    #   +coref:  C-only ∪ coref_validated_keys (deduped by key)
    results_per_k = {}
    for k in range(1, n_passes + 1):
        # Raw: any candidate with vote_count ≥ k
        raw_keys = {key for key, v in vote_counts.items() if v >= k}
        raw_tp = len(raw_keys & gold)
        raw_fp = len(raw_keys - gold)
        # Validated (C-only): raw ∩ validated_post_4b
        v_keys = raw_keys & validated_keys
        v_tp = len(v_keys & gold)
        v_fp = len(v_keys - gold)
        p = v_tp / max(1, v_tp + v_fp)
        r = v_tp / max(1, len(gold))
        f1 = 2 * p * r / max(1e-9, p + r)
        # Validated + coref union (deduped)
        if with_coref:
            combined = v_keys | coref_validated_keys
            c_tp = len(combined & gold)
            c_fp = len(combined - gold)
            cp = c_tp / max(1, c_tp + c_fp)
            cr = c_tp / max(1, len(gold))
            cf1 = 2 * cp * cr / max(1e-9, cp + cr)
        else:
            combined = v_keys
            c_tp, c_fp, cp, cr, cf1 = v_tp, v_fp, p, r, f1
        results_per_k[k] = {
            "raw_total": len(raw_keys), "raw_tp": raw_tp, "raw_fp": raw_fp,
            "validated_total": len(v_keys), "validated_tp": v_tp,
            "validated_fp": v_fp,
            "P": round(p, 4), "R": round(r, 4), "F1": round(f1, 4),
            "with_coref_total": len(combined), "with_coref_tp": c_tp,
            "with_coref_fp": c_fp,
            "with_coref_P": round(cp, 4), "with_coref_R": round(cr, 4),
            "with_coref_F1": round(cf1, 4),
        }

    elapsed = round(time.time() - t0, 1)
    print(f"\n  Per-k breakdown (after Phase 4 + 4b, vs gold |{len(gold)}|):")
    hdr = f"    {'k':<3} {'raw':>5} {'rawTP/FP':>10} {'val':>5} {'valTP/FP':>10} {'F1':>6}"
    if with_coref:
        hdr += f"  {'+coref':>8} {'cTP/FP':>10} {'cF1':>6}"
    print(hdr)
    for k in range(1, n_passes + 1):
        m = results_per_k[k]
        line = (f"    {k:<3} {m['raw_total']:>5} "
                f"{m['raw_tp']:>3}/{m['raw_fp']:<6} "
                f"{m['validated_total']:>5} "
                f"{m['validated_tp']:>3}/{m['validated_fp']:<6} "
                f"{m['F1']:>6.3f}")
        if with_coref:
            line += (f"  {m['with_coref_total']:>8} "
                     f"{m['with_coref_tp']:>3}/{m['with_coref_fp']:<6} "
                     f"{m['with_coref_F1']:>6.3f}")
        print(line)
    print(f"  Dataset wall time: {elapsed}s")

    return {
        "dataset": name,
        "gold": len(gold),
        "n_passes": n_passes,
        "pass_sizes": [len(p) for p in passes],
        "union_size": len(union_candidates),
        "vote_distribution": {
            v: sum(1 for x in vote_counts.values() if x == v)
            for v in range(1, n_passes + 1)
        },
        "validated_total": len(validated),
        "coref_raw": len(coref_keys),
        "coref_validated": len(coref_validated_keys),
        "elapsed_s": elapsed,
        "per_k": results_per_k,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--backend", default=os.environ.get("LLM_BACKEND", "openai"))
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--with-coref", action="store_true",
                    help="Also run coref and report C-union+coref F1 per k")
    args = ap.parse_args()

    os.environ["LLM_BACKEND"] = args.backend
    from llm_sad_sam.llm_client import LLMBackend
    from llm_sad_sam.linkers.experimental.s_linker17f import SLinker17f

    backend_map = {
        "claude": LLMBackend.CLAUDE, "openai": LLMBackend.OPENAI,
        "codex": LLMBackend.CODEX, "checkpoint": LLMBackend.CHECKPOINT,
    }
    backend = backend_map.get(args.backend, LLMBackend.OPENAI)
    print(f"\n5-pass Framing-C experiment | backend={backend.value} | "
          f"passes={args.passes} | datasets={args.datasets}\n")

    linker = SLinker17f(backend=backend)
    out = []
    for name in args.datasets:
        if name not in DATASETS:
            print(f"  unknown dataset '{name}', skipping")
            continue
        r = run_dataset(name, DATASETS[name], args.passes, linker, with_coref=args.with_coref)
        out.append(r)

    # ── Aggregate macro F1 across datasets per k ──────────────────────────
    print(f"\n\n{'='*90}")
    print("AGGREGATE: macro per-k summary")
    print(f"{'='*90}")
    print(f"\n{'Dataset':<14} | " + " | ".join(
        f"k={k}: TP/FP F1".rjust(15) for k in range(1, args.passes + 1)))
    print("-" * (14 + 18 * args.passes))
    macro_f1 = {k: 0.0 for k in range(1, args.passes + 1)}
    macro_tp = {k: 0 for k in range(1, args.passes + 1)}
    macro_fp = {k: 0 for k in range(1, args.passes + 1)}
    for r in out:
        cells = []
        for k in range(1, args.passes + 1):
            m = r["per_k"][k]
            cells.append(f"{m['validated_tp']}/{m['validated_fp']} {m['F1']:.3f}")
            macro_f1[k] += m["F1"]
            macro_tp[k] += m["validated_tp"]
            macro_fp[k] += m["validated_fp"]
        print(f"{r['dataset']:<14} | " + " | ".join(c.rjust(15) for c in cells))
    print("-" * (14 + 18 * args.passes))
    n = max(1, len(out))
    print(f"{'macro avg F1':<14} | " + " | ".join(
        f"{macro_f1[k]/n:.4f}".rjust(15) for k in range(1, args.passes + 1)))
    print(f"{'micro sum':<14} | " + " | ".join(
        f"{macro_tp[k]}/{macro_fp[k]}".rjust(15) for k in range(1, args.passes + 1)))

    if args.with_coref:
        print(f"\n{'='*90}")
        print("WITH COREF (C-union ∪ coref): per-k macro F1")
        print(f"{'='*90}")
        print(f"\n{'Dataset':<14} | " + " | ".join(
            f"k={k}: TP/FP F1".rjust(15) for k in range(1, args.passes + 1)))
        print("-" * (14 + 18 * args.passes))
        macro_cf1 = {k: 0.0 for k in range(1, args.passes + 1)}
        macro_ctp = {k: 0 for k in range(1, args.passes + 1)}
        macro_cfp = {k: 0 for k in range(1, args.passes + 1)}
        for r in out:
            cells = []
            for k in range(1, args.passes + 1):
                m = r["per_k"][k]
                cells.append(f"{m['with_coref_tp']}/{m['with_coref_fp']} {m['with_coref_F1']:.3f}")
                macro_cf1[k] += m["with_coref_F1"]
                macro_ctp[k] += m["with_coref_tp"]
                macro_cfp[k] += m["with_coref_fp"]
            print(f"{r['dataset']:<14} | " + " | ".join(c.rjust(15) for c in cells))
        print("-" * (14 + 18 * args.passes))
        print(f"{'macro avg F1':<14} | " + " | ".join(
            f"{macro_cf1[k]/n:.4f}".rjust(15) for k in range(1, args.passes + 1)))
        print(f"{'micro sum':<14} | " + " | ".join(
            f"{macro_ctp[k]}/{macro_cfp[k]}".rjust(15) for k in range(1, args.passes + 1)))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nResults saved to {args.out_json}")


if __name__ == "__main__":
    main()
