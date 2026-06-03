#!/usr/bin/env python3
"""Pareto ablation: test each V33 fix independently on V32 Sonnet checkpoints.

For each fix, loads V32 checkpoints up to the affected phase, applies ONLY that
fix, reruns from that phase onward, and compares to V32 baseline.

Fixes:
  Fix 1 (Phase 6): Relaxed _is_generic_mention — only dotted-path rejection
  Fix 2 (Phase 9): Judge Rule 4 exception for named components
  Fix 3 (Phase 6): Validation prompt architectural-context exception
  Fix 4 (Phase 7): Structural coref rules (SUBJECT CONTINUITY, PARAGRAPH TOPIC, RECENCY)

Usage:
    python test_v33_pareto.py
    python test_v33_pareto.py --datasets mediastore teammates
    python test_v33_pareto.py --fixes 1 2
"""

import csv
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.core.data_types import SadSamLink, CandidateLink
from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMBackend

os.environ["CLAUDE_MODEL"] = "sonnet"

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)

DATASETS = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
        "gold_sam": BENCHMARK_BASE / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
        "model": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
        "gold_sam": BENCHMARK_BASE / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
        "gold_sam": BENCHMARK_BASE / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
        "gold_sam": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
        "gold_sam": BENCHMARK_BASE / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
}

CACHE_DIR = Path("./results/phase_cache/v32")


def load_gold(gold_path):
    links = set()
    with open(gold_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def eval_metrics(predicted, gold):
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"tp": tp, "fp": fp, "fn": fn, "P": p, "R": r, "F1": f1}


def load_checkpoint(ds_name, phase_name):
    path = CACHE_DIR / ds_name / f"{phase_name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def get_v32_final(ds_name, gold):
    """Load V32 final results as baseline."""
    data = load_checkpoint(ds_name, "final")
    final = data["final"]
    pred = {(l.sentence_number, l.component_id) for l in final}
    return eval_metrics(pred, gold), final


# ── Fix 1: Relaxed generic mention filter ─────────────────────────────
# This is a deterministic code-level filter in Phase 6. We can test it
# offline by replaying Phase 5 candidates through the changed filter,
# then running remaining phases.

def test_fix1_offline(ds_name, components, sentences, sent_map, gold):
    """Fix 1: check how many extra candidates pass the relaxed filter.

    We don't need LLM — just count how many candidates the V32 filter
    rejects that the V33 filter would pass, and vice versa.
    """
    from llm_sad_sam.linkers.experimental.agent_linker_v26a import AgentLinkerV26a

    # Load Phase 5 candidates (before validation)
    p5 = load_checkpoint(ds_name, "phase5")
    candidates = p5["candidates"]

    # Instantiate V26a just to use the method
    linker = AgentLinkerV26a.__new__(AgentLinkerV26a)

    # V32 filter (original — inherited from V26a)
    def v32_is_generic(comp_name, sentence_text):
        if ' ' in comp_name or '-' in comp_name:
            return False
        if re.search(r'[a-z][A-Z]', comp_name):
            return False
        if comp_name.isupper():
            return False
        if comp_name[0].islower():
            return False
        if linker._has_standalone_mention(comp_name, sentence_text):
            return False
        word_lower = comp_name.lower()
        if re.search(rf'\b{re.escape(word_lower)}\b', sentence_text):
            return True
        return False

    # V33 filter (relaxed — only dotted-path)
    def v33_is_generic(comp_name, sentence_text):
        if ' ' in comp_name or '-' in comp_name:
            return False
        if re.search(r'[a-z][A-Z]', comp_name):
            return False
        if comp_name.isupper():
            return False
        if comp_name[0].islower():
            return False
        if linker._has_standalone_mention(comp_name, sentence_text):
            return False
        word_lower = comp_name.lower()
        if not re.search(rf'\b{re.escape(word_lower)}\b', sentence_text):
            return False
        if re.search(rf'\b{re.escape(word_lower)}\.\w', sentence_text):
            return True
        return False

    # The _is_generic_mention is called inside _validate_intersect before
    # sending candidates to LLM. Check each candidate.
    name_to_id = {c.name: c.id for c in components}

    v32_rejected = []
    v33_would_pass = []

    for c in candidates:
        sent = sent_map.get(c.sentence_number)
        if not sent:
            continue
        v32_gen = v32_is_generic(c.component_name, sent.text)
        v33_gen = v33_is_generic(c.component_name, sent.text)

        if v32_gen and not v33_gen:
            is_tp = (c.sentence_number, c.component_id) in gold
            v33_would_pass.append({
                "sent": c.sentence_number,
                "comp": c.component_name,
                "is_tp": is_tp,
                "text": sent.text[:100],
            })
        if v32_gen:
            is_tp = (c.sentence_number, c.component_id) in gold
            v32_rejected.append({
                "sent": c.sentence_number,
                "comp": c.component_name,
                "is_tp": is_tp,
            })

    return v32_rejected, v33_would_pass


# ── Fix 2: Judge Rule 4 exception ────────────────────────────────────
# This affects Phase 9. We can test by loading pre_judge checkpoint
# and re-running the judge with the new prompt.

def test_fix2(ds_name, components, sentences, sent_map, gold):
    """Fix 2: Re-run Phase 9 judge with Rule 4 exception on V32 pre_judge data."""
    from llm_sad_sam.linkers.experimental.ilinker2_v33 import ILinker2V33
    from llm_sad_sam.linkers.experimental.ilinker2_v32 import ILinker2V32

    pre = load_checkpoint(ds_name, "pre_judge")
    preliminary = pre["preliminary"]
    transarc_set = pre["transarc_set"]

    # Load model knowledge for judge
    p1 = load_checkpoint(ds_name, "phase1")
    p3 = load_checkpoint(ds_name, "phase3")

    # Run V32 judge (baseline)
    v32 = ILinker2V32(backend=LLMBackend.CLAUDE)
    v32.model_knowledge = p1["model_knowledge"]
    v32.doc_knowledge = p3["doc_knowledge"]
    v32._cached_components = components
    v32._cached_sent_map = sent_map

    v32_reviewed = v32._judge_review(preliminary, sentences, components, sent_map, transarc_set)
    v32_pred = {(l.sentence_number, l.component_id) for l in v32_reviewed}
    v32_m = eval_metrics(v32_pred, gold)

    # Run V33 judge (with Rule 4 exception) — only _build_judge_prompt differs
    v33 = ILinker2V33(backend=LLMBackend.CLAUDE)
    v33.model_knowledge = p1["model_knowledge"]
    v33.doc_knowledge = p3["doc_knowledge"]
    v33._cached_components = components
    v33._cached_sent_map = sent_map

    v33_reviewed = v33._judge_review(preliminary, sentences, components, sent_map, transarc_set)
    v33_pred = {(l.sentence_number, l.component_id) for l in v33_reviewed}
    v33_m = eval_metrics(v33_pred, gold)

    return v32_m, v33_m, v32_reviewed, v33_reviewed


# ── Fix 3: Validation prompt exception ────────────────────────────────
# This affects Phase 6. We need to re-run validation on Phase 5 candidates
# with the new prompt, then replay Phase 7-9.

def test_fix3(ds_name, components, sentences, sent_map, gold):
    """Fix 3: Re-run Phase 6 validation with architectural-context exception."""
    from llm_sad_sam.linkers.experimental.ilinker2_v33 import ILinker2V33
    from llm_sad_sam.linkers.experimental.ilinker2_v32 import ILinker2V32

    p5 = load_checkpoint(ds_name, "phase5")
    candidates = p5["candidates"]

    # V32 validation
    v32 = ILinker2V32(backend=LLMBackend.CLAUDE)
    v32.model_knowledge = load_checkpoint(ds_name, "phase1")["model_knowledge"]
    v32.doc_knowledge = load_checkpoint(ds_name, "phase3")["doc_knowledge"]
    v32._cached_components = components
    v32._cached_sent_map = sent_map
    v32_validated = v32._validate_intersect(candidates, components, sent_map)

    # V33 validation (only _qual_validation_pass differs)
    v33 = ILinker2V33(backend=LLMBackend.CLAUDE)
    v33.model_knowledge = load_checkpoint(ds_name, "phase1")["model_knowledge"]
    v33.doc_knowledge = load_checkpoint(ds_name, "phase3")["doc_knowledge"]
    v33._cached_components = components
    v33._cached_sent_map = sent_map
    v33_validated = v33._validate_intersect(candidates, components, sent_map)

    v32_set = {(c.sentence_number, c.component_id) for c in v32_validated}
    v33_set = {(c.sentence_number, c.component_id) for c in v33_validated}

    gained = v33_set - v32_set
    lost = v32_set - v33_set

    gained_tp = sum(1 for s, c in gained if (s, c) in gold)
    gained_fp = len(gained) - gained_tp
    lost_tp = sum(1 for s, c in lost if (s, c) in gold)
    lost_fp = len(lost) - lost_tp

    id_to_name = {c.id: c.name for c in components}

    return {
        "gained": len(gained), "gained_tp": gained_tp, "gained_fp": gained_fp,
        "lost": len(lost), "lost_tp": lost_tp, "lost_fp": lost_fp,
        "gained_details": [(s, id_to_name.get(c, c), (s, c) in gold) for s, c in sorted(gained)],
        "lost_details": [(s, id_to_name.get(c, c), (s, c) in gold) for s, c in sorted(lost)],
    }


def main():
    selected_datasets = list(DATASETS.keys())
    selected_fixes = [1, 2, 3, 4]

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--datasets":
            selected_datasets = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                selected_datasets.append(args[i])
                i += 1
        elif args[i] == "--fixes":
            selected_fixes = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                selected_fixes.append(int(args[i]))
                i += 1
        else:
            i += 1

    print("=" * 100)
    print("V33 PARETO ABLATION: Test each fix independently on V32 Sonnet checkpoints")
    print("=" * 100)

    for ds_name in selected_datasets:
        paths = DATASETS[ds_name]
        gold = load_gold(str(paths["gold_sam"]))
        components = parse_pcm_repository(str(paths["model"]))
        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        sent_map = {s.number: s for s in sentences}
        id_to_name = {c.id: c.name for c in components}

        # V32 baseline
        v32_m, v32_final = get_v32_final(ds_name, gold)

        print(f"\n{'='*100}")
        print(f"DATASET: {ds_name}")
        print(f"V32 baseline: P={v32_m['P']:.1%} R={v32_m['R']:.1%} F1={v32_m['F1']:.1%} "
              f"FP={v32_m['fp']} FN={v32_m['fn']}")
        print(f"{'='*100}")

        # ── Fix 1: Offline analysis (no LLM calls) ──
        if 1 in selected_fixes:
            print(f"\n--- Fix 1: Relaxed generic mention filter (OFFLINE) ---")
            v32_rej, v33_pass = test_fix1_offline(ds_name, components, sentences, sent_map, gold)
            print(f"  V32 rejects {len(v32_rej)} candidates as generic mention")
            for r in v32_rej:
                print(f"    {'TP' if r['is_tp'] else 'FP'}: S{r['sent']} → {r['comp']}")
            print(f"  V33 would PASS {len(v33_pass)} of those:")
            for p in v33_pass:
                print(f"    {'TP' if p['is_tp'] else 'FP'}: S{p['sent']} → {p['comp']} — {p['text']}")
            tp_gain = sum(1 for p in v33_pass if p["is_tp"])
            fp_gain = sum(1 for p in v33_pass if not p["is_tp"])
            print(f"  NET IMPACT: +{tp_gain} TP, +{fp_gain} FP (candidates only, before validation)")

        # ── Fix 2: Judge Rule 4 (needs LLM) ──
        if 2 in selected_fixes:
            print(f"\n--- Fix 2: Judge Rule 4 exception (LLM) ---")
            v32_jm, v33_jm, v32_rev, v33_rev = test_fix2(
                ds_name, components, sentences, sent_map, gold)
            v32_set = {(l.sentence_number, l.component_id) for l in v32_rev}
            v33_set = {(l.sentence_number, l.component_id) for l in v33_rev}
            gained = v33_set - v32_set
            lost = v32_set - v33_set
            print(f"  V32 judge: P={v32_jm['P']:.1%} R={v32_jm['R']:.1%} F1={v32_jm['F1']:.1%} "
                  f"FP={v32_jm['fp']} FN={v32_jm['fn']}")
            print(f"  V33 judge: P={v33_jm['P']:.1%} R={v33_jm['R']:.1%} F1={v33_jm['F1']:.1%} "
                  f"FP={v33_jm['fp']} FN={v33_jm['fn']}")
            delta = v33_jm['F1'] - v32_jm['F1']
            print(f"  Delta F1: {delta:+.1%}")
            for s, c in sorted(gained):
                is_tp = (s, c) in gold
                print(f"    GAINED {'TP' if is_tp else 'FP'}: S{s} → {id_to_name.get(c, c)}")
            for s, c in sorted(lost):
                is_tp = (s, c) in gold
                print(f"    LOST {'TP' if is_tp else 'FP'}: S{s} → {id_to_name.get(c, c)}")

        # ── Fix 3: Validation prompt (needs LLM) ──
        if 3 in selected_fixes:
            print(f"\n--- Fix 3: Validation prompt exception (LLM) ---")
            r3 = test_fix3(ds_name, components, sentences, sent_map, gold)
            print(f"  Gained: {r3['gained']} ({r3['gained_tp']} TP, {r3['gained_fp']} FP)")
            print(f"  Lost: {r3['lost']} ({r3['lost_tp']} TP, {r3['lost_fp']} FP)")
            for s, comp, is_tp in r3["gained_details"]:
                print(f"    GAINED {'TP' if is_tp else 'FP'}: S{s} → {comp}")
            for s, comp, is_tp in r3["lost_details"]:
                print(f"    LOST {'TP' if is_tp else 'FP'}: S{s} → {comp}")

        # ── Fix 4: Coref structural rules (needs LLM) ──
        if 4 in selected_fixes:
            print(f"\n--- Fix 4: Structural coref rules (LLM) ---")
            # Load V32 phase 6+7 checkpoints
            p6 = load_checkpoint(ds_name, "phase6")
            p7 = load_checkpoint(ds_name, "phase7")
            v32_coref = p7.get("coref_links", [])
            v32_coref_set = {(l.sentence_number, l.component_id) for l in v32_coref}

            # Run V33 coref
            from llm_sad_sam.linkers.experimental.ilinker2_v33 import ILinker2V33
            v33 = ILinker2V33(backend=LLMBackend.CLAUDE)
            p0 = load_checkpoint(ds_name, "phase0")
            p1 = load_checkpoint(ds_name, "phase1")
            p2 = load_checkpoint(ds_name, "phase2")
            p3 = load_checkpoint(ds_name, "phase3")
            v33.doc_profile = p0["doc_profile"]
            v33._is_complex = p0["is_complex"]
            v33.model_knowledge = p1["model_knowledge"]
            v33.learned_patterns = p2["learned_patterns"]
            v33.doc_knowledge = p3["doc_knowledge"]
            v33._cached_components = components
            v33._cached_sent_map = sent_map

            name_to_id = {c.name: c.id for c in components}

            if not v33._is_complex:
                discourse_model = v33._build_discourse_model(sentences, components, name_to_id)
                v33_coref = v33._coref_discourse(sentences, components, name_to_id, sent_map, discourse_model)
            else:
                v33_coref = v33._coref_debate(sentences, components, name_to_id, sent_map)

            v33_coref_set = {(l.sentence_number, l.component_id) for l in v33_coref}
            gained = v33_coref_set - v32_coref_set
            lost = v32_coref_set - v33_coref_set

            print(f"  V32 coref: {len(v32_coref)} links")
            print(f"  V33 coref: {len(v33_coref)} links")
            for s, c in sorted(gained):
                is_tp = (s, c) in gold
                print(f"    GAINED {'TP' if is_tp else 'FP'}: S{s} → {id_to_name.get(c, c)}")
            for s, c in sorted(lost):
                is_tp = (s, c) in gold
                print(f"    LOST {'TP' if is_tp else 'FP'}: S{s} → {id_to_name.get(c, c)}")


if __name__ == "__main__":
    main()
