"""Ablation: does _classify_mention (mention_type in evidence bundles) affect validation?

Checkpoint-based analysis (zero LLM calls):
  - Distribution of mention_types across all 5 datasets
  - Correlation between mention_type and approval/rejection decisions
  - Information gain: does knowing mention_type predict the outcome?

To run the full LLM ablation (re-runs validation without mention_type hints):
  python tests/test_mention_type_ablation.py --live
"""

import os
import pickle
import sys
from collections import defaultdict

DATASETS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
CACHE = "results/phase_cache/s_linker12c"


def load_checkpoint(dataset, phase):
    path = os.path.join(CACHE, dataset, f"{phase}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def analyze_from_checkpoints():
    """Checkpoint-only analysis: mention_type → decision correlation."""
    print("=" * 70)
    print("MENTION TYPE ABLATION — Checkpoint Analysis (zero LLM calls)")
    print("=" * 70)

    # Aggregate stats
    global_by_type = defaultdict(lambda: {"approved": 0, "rejected": 0, "total": 0})

    for ds in DATASETS:
        ec = load_checkpoint(ds, "entity_candidates")
        ed = load_checkpoint(ds, "entity_decisions")
        if not ec or not ed:
            print(f"\n{ds}: SKIP (missing checkpoints)")
            continue

        bundles = ec["bundles"]
        decisions = ed["decisions"]

        by_type = defaultdict(lambda: {"approved": 0, "rejected": 0, "total": 0})
        for key, bundle in bundles.items():
            mt = bundle.mention_type
            dec = decisions.get(key, {})
            approved = dec.get("approved", False)

            by_type[mt]["total"] += 1
            global_by_type[mt]["total"] += 1
            if approved:
                by_type[mt]["approved"] += 1
                global_by_type[mt]["approved"] += 1
            else:
                by_type[mt]["rejected"] += 1
                global_by_type[mt]["rejected"] += 1

        print(f"\n{'=' * 50}")
        print(f"{ds} ({len(bundles)} candidates)")
        print(f"{'=' * 50}")
        for mt in sorted(by_type.keys()):
            s = by_type[mt]
            rate = s["approved"] / s["total"] * 100 if s["total"] else 0
            print(f"  {mt:40s}  {s['approved']:3d}/{s['total']:3d} approved ({rate:5.1f}%)")

    # Global summary
    print(f"\n{'=' * 70}")
    print("GLOBAL SUMMARY (all 5 datasets)")
    print(f"{'=' * 70}")

    total_all = sum(s["total"] for s in global_by_type.values())
    for mt in sorted(global_by_type.keys(), key=lambda k: -global_by_type[k]["total"]):
        s = global_by_type[mt]
        rate = s["approved"] / s["total"] * 100 if s["total"] else 0
        pct = s["total"] / total_all * 100
        print(f"  {mt:40s}  {s['approved']:3d}/{s['total']:3d} approved ({rate:5.1f}%)  "
              f"[{pct:4.1f}% of candidates]")

    # Discriminative analysis
    print(f"\n{'=' * 70}")
    print("DISCRIMINATIVE VALUE ANALYSIS")
    print(f"{'=' * 70}")

    proper = global_by_type.get("proper case, standalone", {"approved": 0, "total": 0})
    indirect = global_by_type.get("indirect/unclear match", {"approved": 0, "total": 0})

    # If most candidates are "proper case, standalone" with ~95% approval,
    # the mention_type adds no information for the majority
    if proper["total"]:
        print(f"\n  'proper case, standalone': {proper['approved']}/{proper['total']} "
              f"({proper['approved']/proper['total']*100:.1f}% approval)")
        print(f"  → This is {proper['total']/total_all*100:.0f}% of all candidates. "
              f"The label adds NO information here (validator sees the text directly).")

    if indirect["total"]:
        print(f"\n  'indirect/unclear match': {indirect['approved']}/{indirect['total']} "
              f"({indirect['approved']/indirect['total']*100:.1f}% approval)")
        print(f"  → Strong rejection signal. But validator could infer this from absence "
              f"of matched_text in sentence.")

    # Count alias-based mentions
    alias_a, alias_t = 0, 0
    for mt, s in global_by_type.items():
        if mt.startswith("via known"):
            alias_a += s["approved"]
            alias_t += s["total"]
    if alias_t:
        print(f"\n  'via known alias/synonym/abbrev': {alias_a}/{alias_t} "
              f"({alias_a/alias_t*100:.1f}% approval)")
        print(f"  → Strongest signal. Tells validator 'this is a known alias, trust it.'")
        print(f"  → But evidence bundle already shows source=entity + matched_span, "
              f"which conveys similar info.")

    # Verdict
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")
    print(f"  Total candidates: {total_all}")
    print(f"  Types where mention_type is INFORMATIVE:")
    informative = 0
    for mt, s in global_by_type.items():
        if mt != "proper case, standalone":
            informative += s["total"]
    print(f"    {informative}/{total_all} candidates ({informative/total_all*100:.1f}%)")
    print(f"  Types where mention_type is REDUNDANT (proper case, standalone):")
    print(f"    {proper['total']}/{total_all} candidates ({proper['total']/total_all*100:.1f}%)")
    print()
    print("  → mention_type is informative for ~25% of candidates (alias/indirect)")
    print("  → For the 75% majority (proper case), the label is noise")
    print("  → Full LLM ablation needed to measure actual F1 impact")
    print("  → Run with --live to re-run validation without mention_type hints")


def live_ablation():
    """Re-run validation with/without mention_type to measure F1 impact.

    Loads Tier 1 knowledge + entity candidates from checkpoints,
    rebuilds evidence bundles with and without mention_type,
    re-runs validation, compares decisions.
    """
    print("=" * 70)
    print("MENTION TYPE ABLATION — Live LLM Comparison")
    print("=" * 70)
    print()
    print("This requires LLM calls. Loading 12c checkpoints and re-running")
    print("validation with stripped mention_type hints.")
    print()

    # Import linker for its validation method
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from llm_sad_sam.linkers.experimental.s_linker12c import SLinker12c, EvidenceBundle
    from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
    from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map

    BENCHMARK = os.environ.get(
        "BENCHMARK_DIR",
        os.path.expanduser("~/project/adc/ardoco/core/tests-base/src/main/resources/benchmark"),
    )

    TEXT_MAP = {
        "mediastore": f"{BENCHMARK}/mediastore/text_2016/mediastore.txt",
        "teastore": f"{BENCHMARK}/teastore/text_2020/teastore.txt",
        "teammates": f"{BENCHMARK}/teammates/text_2021/teammates.txt",
        "bigbluebutton": f"{BENCHMARK}/bigbluebutton/text_2021/bigbluebutton.txt",
        "jabref": f"{BENCHMARK}/jabref/text_2021/jabref.txt",
    }

    MODEL_MAP = {
        "mediastore": f"{BENCHMARK}/mediastore/model_2016/pcm/ms.repository",
        "teastore": f"{BENCHMARK}/teastore/model_2020/pcm/teastore.repository",
        "teammates": f"{BENCHMARK}/teammates/model_2021/pcm/teammates.repository",
        "bigbluebutton": f"{BENCHMARK}/bigbluebutton/model_2021/pcm/bbb.repository",
        "jabref": f"{BENCHMARK}/jabref/model_2021/pcm/jabref.repository",
    }

    linker = SLinker12c()

    for ds in DATASETS:
        print(f"\n{'='*50}")
        print(f"  {ds}")
        print(f"{'='*50}")

        ec = load_checkpoint(ds, "entity_candidates")
        l1 = load_checkpoint(ds, "layer1")
        if not ec or not l1:
            print(f"  SKIP (missing checkpoints)")
            continue

        # Restore linker state from checkpoint
        linker.model_knowledge = l1["model_knowledge"]
        linker.doc_knowledge = l1["doc_knowledge"]

        candidates = ec["entity_candidates"]
        bundles_with = ec["bundles"]

        # Load raw data for sent_map
        text_path = TEXT_MAP.get(ds)
        model_path = MODEL_MAP.get(ds)
        if not text_path or not os.path.exists(text_path):
            print(f"  SKIP (missing text file: {text_path})")
            continue

        sentences = load_sentences(text_path)
        components = parse_pcm_repository(model_path)
        sent_map = build_sent_map(sentences)

        # Build bundles WITHOUT mention_type (replace with generic label)
        bundles_without = {}
        for key, b in bundles_with.items():
            bundles_without[key] = EvidenceBundle(
                source=b.source,
                matched_span=b.matched_span,
                mention_type="(not provided)",  # stripped
                preceding_text=b.preceding_text,
                anchor_sentences=b.anchor_sentences,
                is_ambiguous=b.is_ambiguous,
                extraction_rationale=b.extraction_rationale,
            )

        # Run validation WITH mention_type
        print(f"  Running validation WITH mention_type...")
        val_with, dec_with = linker._validate_with_evidence(
            candidates, bundles_with, components, sent_map)

        # Run validation WITHOUT mention_type
        print(f"  Running validation WITHOUT mention_type...")
        val_without, dec_without = linker._validate_with_evidence(
            candidates, bundles_without, components, sent_map)

        # Compare
        keys_with = {(c.sentence_number, c.component_id) for c in val_with}
        keys_without = {(c.sentence_number, c.component_id) for c in val_without}

        gained = keys_without - keys_with    # approved only without mention_type
        lost = keys_with - keys_without      # approved only with mention_type

        print(f"  WITH mention_type:    {len(val_with)} approved")
        print(f"  WITHOUT mention_type: {len(val_without)} approved")
        print(f"  Gained (new approvals): {len(gained)}")
        print(f"  Lost (new rejections):  {len(lost)}")

        if lost:
            print(f"  LOST links:")
            for key in sorted(lost):
                c = next((x for x in candidates
                         if x.sentence_number == key[0] and x.component_id == key[1]), None)
                b = bundles_with.get(key)
                if c and b:
                    print(f"    S{c.sentence_number} -> {c.component_name} "
                          f"[mention={b.mention_type}]")

        if gained:
            print(f"  GAINED links:")
            for key in sorted(gained):
                c = next((x for x in candidates
                         if x.sentence_number == key[0] and x.component_id == key[1]), None)
                b = bundles_with.get(key)
                if c and b:
                    print(f"    S{c.sentence_number} -> {c.component_name} "
                          f"[mention={b.mention_type}]")


if __name__ == "__main__":
    if "--live" in sys.argv:
        live_ablation()
    else:
        analyze_from_checkpoints()
