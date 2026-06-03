#!/usr/bin/env python3
"""Test pure-LLM Fix 1: teach the validation prompt about package paths
instead of using a code-level _is_generic_mention filter.

Approach:
- _is_generic_mention always returns False (pass all candidates to LLM)
- Validation prompt gets a new rule about single-word component names:
  "lowercase uses in architectural context = component reference,
   lowercase uses in dotted-path context = sub-package, not component"

Loads V32 Phase 5 checkpoints, reruns Phase 6 validation with the new
prompt, and compares to V32 baseline.

Usage:
    python test_fix1_llm.py
    python test_fix1_llm.py --datasets mediastore teammates
"""

import csv
import os
import pickle
import sys
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
    data = load_checkpoint(ds_name, "final")
    final = data["final"]
    pred = {(l.sentence_number, l.component_id) for l in final}
    return eval_metrics(pred, gold), final


def test_fix1_llm(ds_name, components, sentences, sent_map, gold):
    """Pure-LLM Fix 1: disable _is_generic_mention, teach validation prompt."""
    from llm_sad_sam.linkers.experimental.ilinker2_v32 import ILinker2V32

    p5 = load_checkpoint(ds_name, "phase5")
    candidates = p5["candidates"]

    # ── V32 baseline validation ──
    v32 = ILinker2V32(backend=LLMBackend.CLAUDE)
    v32.model_knowledge = load_checkpoint(ds_name, "phase1")["model_knowledge"]
    v32.doc_knowledge = load_checkpoint(ds_name, "phase3")["doc_knowledge"]
    v32._cached_components = components
    v32._cached_sent_map = sent_map
    v32_validated = v32._validate_intersect(candidates, components, sent_map)

    # ── Pure-LLM Fix 1 validation ──
    # Subclass that disables the filter and teaches the prompt
    class Fix1LLM(ILinker2V32):
        def _is_generic_mention(self, comp_name, sentence_text):
            """Disabled — let LLM decide."""
            return False

        def _qual_validation_pass(self, comp_names, ctx, cases, focus):
            """V32 validation prompt + package-path awareness rule."""
            prompt = f"""Validate component references in a software architecture document. {focus}

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

DECISION RULES:
APPROVE when:
- The component is the grammatical actor or subject (the sentence is ABOUT the component)
- A section heading names the component (introduces that component's topic)
- The sentence describes what the component does, provides, or interacts with

REJECT when:
- The name is used as an ordinary English word, not as a proper name
  (Like "proxy" in "proxy pattern" is the design pattern concept, not the Proxy component — reject the component link)
- The name is a modifier inside a larger phrase, not a standalone reference
  (Like "observer" in "observer pattern" modifies pattern — reject if Observer is a component)
- The sentence is about a subprocess, algorithm, or implementation detail — not the component itself

SINGLE-WORD COMPONENT NAMES (important for names like single common English words):
When the system has a component with a single-word name that is also an ordinary English
word, apply these rules:
- APPROVE when the word is used as a NOUN referring to a part of the system in an
  architectural context (describing system behavior, interactions, responsibilities).
  Even if lowercase, the word refers to the component when the sentence discusses
  the system's architecture.
- REJECT when the word appears inside a DOTTED PACKAGE PATH or QUALIFIED NAME
  (e.g., "x.utils", "x.api", "x.datatransfer"). The dotted path refers to a
  sub-package inside the component, not to the component itself.
- REJECT when the word is used as a plain English adjective, verb, or in an idiom
  unrelated to the system (e.g., "common ground", "persistent effort").

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
            results = {}
            if data:
                for v in data.get("validations", []):
                    idx = v.get("case", 0) - 1
                    if 0 <= idx < len(cases):
                        results[idx] = v.get("approve", False)
            return results

    fix1 = Fix1LLM(backend=LLMBackend.CLAUDE)
    fix1.model_knowledge = load_checkpoint(ds_name, "phase1")["model_knowledge"]
    fix1.doc_knowledge = load_checkpoint(ds_name, "phase3")["doc_knowledge"]
    fix1._cached_components = components
    fix1._cached_sent_map = sent_map
    fix1_validated = fix1._validate_intersect(candidates, components, sent_map)

    v32_set = {(c.sentence_number, c.component_id) for c in v32_validated}
    fix1_set = {(c.sentence_number, c.component_id) for c in fix1_validated}

    gained = fix1_set - v32_set
    lost = v32_set - fix1_set

    id_to_name = {c.id: c.name for c in components}

    gained_tp = sum(1 for s, c in gained if (s, c) in gold)
    gained_fp = len(gained) - gained_tp
    lost_tp = sum(1 for s, c in lost if (s, c) in gold)
    lost_fp = len(lost) - lost_tp

    return {
        "v32_count": len(v32_validated),
        "fix1_count": len(fix1_validated),
        "gained": len(gained), "gained_tp": gained_tp, "gained_fp": gained_fp,
        "lost": len(lost), "lost_tp": lost_tp, "lost_fp": lost_fp,
        "gained_details": [(s, id_to_name.get(c, c), (s, c) in gold) for s, c in sorted(gained)],
        "lost_details": [(s, id_to_name.get(c, c), (s, c) in gold) for s, c in sorted(lost)],
    }


def main():
    selected_datasets = list(DATASETS.keys())

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--datasets":
            selected_datasets = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                selected_datasets.append(args[i])
                i += 1
        else:
            i += 1

    print("=" * 100)
    print("FIX 1 PURE-LLM: Disable _is_generic_mention, teach validation prompt")
    print("=" * 100)

    totals = {"gained_tp": 0, "gained_fp": 0, "lost_tp": 0, "lost_fp": 0}

    for ds_name in selected_datasets:
        paths = DATASETS[ds_name]
        gold = load_gold(str(paths["gold_sam"]))
        components = parse_pcm_repository(str(paths["model"]))
        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        sent_map = {s.number: s for s in sentences}

        v32_m, _ = get_v32_final(ds_name, gold)
        print(f"\n{'='*100}")
        print(f"DATASET: {ds_name}")
        print(f"V32 baseline: P={v32_m['P']:.1%} R={v32_m['R']:.1%} F1={v32_m['F1']:.1%} "
              f"FP={v32_m['fp']} FN={v32_m['fn']}")
        print(f"{'='*100}")

        r = test_fix1_llm(ds_name, components, sentences, sent_map, gold)
        print(f"\n  V32 validated: {r['v32_count']}, Fix1-LLM validated: {r['fix1_count']}")
        print(f"  Gained: {r['gained']} ({r['gained_tp']} TP, {r['gained_fp']} FP)")
        print(f"  Lost:   {r['lost']} ({r['lost_tp']} TP, {r['lost_fp']} FP)")

        for s, comp, is_tp in r["gained_details"]:
            print(f"    GAINED {'TP' if is_tp else 'FP'}: S{s} -> {comp}")
        for s, comp, is_tp in r["lost_details"]:
            print(f"    LOST {'TP' if is_tp else 'FP'}: S{s} -> {comp}")

        for k in totals:
            totals[k] += r[k]

    print(f"\n{'='*100}")
    print(f"TOTALS: gained {totals['gained_tp']} TP + {totals['gained_fp']} FP, "
          f"lost {totals['lost_tp']} TP + {totals['lost_fp']} FP")
    print(f"NET: {totals['gained_tp'] - totals['lost_tp']:+d} TP, "
          f"{totals['gained_fp'] - totals['lost_fp']:+d} FP")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
