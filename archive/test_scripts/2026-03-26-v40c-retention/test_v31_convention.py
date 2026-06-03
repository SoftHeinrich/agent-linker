#!/usr/bin/env python3
"""Unit test: V31 convention-aware boundary filter on V30c checkpoints.

Tests the LLM convention filter (3-step reasoning guide) as a replacement
for the regex-based _apply_boundary_filters. Loads V30c pre-P8c state
from checkpoints and runs the convention filter single-phase.
"""

import csv
import glob
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.core.data_types import SadSamLink
from llm_sad_sam.llm_client import LLMClient

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
DATASETS = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore" / "text_2016" / "mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore" / "model_2016" / "pcm" / "ms.repository",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore" / "text_2020" / "teastore.txt",
        "model": BENCHMARK_BASE / "teastore" / "model_2020" / "pcm" / "teastore.repository",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates" / "text_2021" / "teammates.txt",
        "model": BENCHMARK_BASE / "teammates" / "model_2021" / "pcm" / "teammates.repository",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton" / "text_2021" / "bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton" / "model_2021" / "pcm" / "bbb.repository",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref" / "text_2021" / "jabref.txt",
        "model": BENCHMARK_BASE / "jabref" / "model_2021" / "pcm" / "jabref.repository",
    },
}
CACHE_DIR = Path("./results/phase_cache/v30c")


def load_checkpoint(dataset, phase_name):
    path = CACHE_DIR / dataset / f"{phase_name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dataset(dataset):
    paths = DATASETS[dataset]
    components = parse_pcm_repository(str(paths["model"]))
    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    sent_map = DocumentLoader.build_sent_map(sentences)
    name_to_id = {c.name: c.id for c in components}
    return components, sentences, sent_map, name_to_id


def load_gold(dataset):
    gold_path = BENCHMARK_BASE / dataset
    pattern = str(gold_path / "**" / "goldstandard_sad_*-sam_*.csv")
    files = [f for f in glob.glob(pattern, recursive=True) if "UME" not in f and "code" not in f]
    gold = set()
    for f in files:
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                cid = row.get("modelElementID", "")
                sid = row.get("sentence", "")
                if sid and cid:
                    gold.add((int(sid), cid))
    return gold


def _has_clean_mention(term, text):
    pattern = rf'\b{re.escape(term)}\b'
    for m in re.finditer(pattern, text, re.IGNORECASE):
        s, e = m.start(), m.end()
        if s > 0 and text[s-1] == '.':
            continue
        if e < len(text) and text[e] == '.' and e + 1 < len(text) and text[e+1].isalpha():
            continue
        if (s > 0 and text[s-1] == '-') or (e < len(text) and text[e] == '-'):
            continue
        return True
    return False


def main():
    os.environ.setdefault("CLAUDE_MODEL", "sonnet")

    from llm_sad_sam.linkers.experimental.ilinker2_v31 import ILinker2V31

    print("=" * 90)
    print("  V31 Convention Filter Test (on V30c checkpoints)")
    print("=" * 90)

    totals = {"v30c_fp": 0, "v30c_tp_kill": 0, "v31_fp": 0, "v31_tp_kill": 0}

    for ds in DATASETS:
        data3 = load_checkpoint(ds, "phase3")
        data4 = load_checkpoint(ds, "phase4")
        data6 = load_checkpoint(ds, "phase6")
        data7 = load_checkpoint(ds, "phase7")
        if not all([data3, data4, data6, data7]):
            continue

        components, sentences, sent_map, name_to_id = load_dataset(ds)
        gold = load_gold(ds)

        transarc_set = data4["transarc_set"]
        transarc_links = data4.get("transarc_links", [])
        validated = data6.get("validated", [])
        coref_links = data7.get("coref_links", [])
        doc_knowledge = data3.get("doc_knowledge")

        # P8b partial injection
        partial_links = []
        if doc_knowledge and doc_knowledge.partial_references:
            existing = (transarc_set
                        | {(c.sentence_number, c.component_id) for c in validated}
                        | {(l.sentence_number, l.component_id) for l in coref_links})
            for partial, comp_name in doc_knowledge.partial_references.items():
                if comp_name not in name_to_id:
                    continue
                comp_id = name_to_id[comp_name]
                for sent in sentences:
                    key = (sent.number, comp_id)
                    if key in existing:
                        continue
                    if _has_clean_mention(partial, sent.text):
                        partial_links.append(SadSamLink(sent.number, comp_id, comp_name, 0.8, "partial_inject"))
                        existing.add(key)

        # Combine + deduplicate
        SOURCE_PRIORITY = {"transarc": 1, "entity": 2, "validated": 2, "coreference": 3,
                           "implicit": 4, "partial_inject": 5}
        entity_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
            for c in validated
        ]
        all_links = transarc_links + entity_links + coref_links + partial_links
        link_map = {}
        for lk in all_links:
            key = (lk.sentence_number, lk.component_id)
            if key not in link_map:
                link_map[key] = lk
            else:
                old_p = SOURCE_PRIORITY.get(link_map[key].source, 0)
                new_p = SOURCE_PRIORITY.get(lk.source, 0)
                if new_p > old_p:
                    link_map[key] = lk
        preliminary = list(link_map.values())

        print(f"\n{'─' * 70}")
        print(f"  {ds}: {len(preliminary)} pre-P8c links")
        print(f"{'─' * 70}")

        # ── V30c original (regex) ──
        from llm_sad_sam.linkers.experimental.agent_linker_v26a import AgentLinkerV26a

        class RegexTester(AgentLinkerV26a):
            def __init__(self):
                self.doc_knowledge = None
                self.model_knowledge = None
                self.GENERIC_COMPONENT_WORDS = set()
                self.GENERIC_PARTIALS = set()

        tester = RegexTester()
        v30c_kept, v30c_rejected = [], []
        for lk in preliminary:
            sent = sent_map.get(lk.sentence_number)
            if not sent:
                v30c_kept.append(lk)
                continue
            is_ta = (lk.sentence_number, lk.component_id) in transarc_set
            if is_ta:
                v30c_kept.append(lk)
                continue
            reason = None
            if lk.source in ("validated", "entity"):
                if tester._is_in_package_path(lk.component_name, sent.text):
                    reason = "package_path"
            if not reason and lk.source in ("validated", "entity", "coreference"):
                if tester._is_generic_word_usage(lk.component_name, sent.text):
                    reason = "generic_word"
            if not reason:
                if tester._is_weak_partial_match(lk, sent_map):
                    reason = "weak_partial"
            if reason:
                v30c_rejected.append((lk, reason))
            else:
                v30c_kept.append(lk)

        v30c_tp_kill = sum(1 for lk, _ in v30c_rejected if (lk.sentence_number, lk.component_id) in gold)
        v30c_fp_catch = len(v30c_rejected) - v30c_tp_kill

        print(f"  V30c regex: rejected {len(v30c_rejected)} ({v30c_tp_kill} TP killed, {v30c_fp_catch} FP caught)")
        for lk, reason in v30c_rejected:
            is_tp = (lk.sentence_number, lk.component_id) in gold
            label = "TP KILLED!" if is_tp else "FP caught"
            print(f"    [{reason}] [{label}] S{lk.sentence_number} → {lk.component_name} ({lk.source})")

        # ── V31 convention filter ──
        print(f"\n  V31 convention filter:")
        data1 = load_checkpoint(ds, "phase1")
        linker = ILinker2V31.__new__(ILinker2V31)
        linker.llm = LLMClient()
        linker._cached_components = components
        linker.doc_knowledge = doc_knowledge
        linker.model_knowledge = data1["model_knowledge"] if data1 else None

        v31_kept, v31_rejected = linker._apply_boundary_filters(preliminary, sent_map, transarc_set)
        v31_tp_kill = sum(1 for lk, _ in v31_rejected if (lk.sentence_number, lk.component_id) in gold)
        v31_fp_catch = len(v31_rejected) - v31_tp_kill

        print(f"\n  V31 convention: rejected {len(v31_rejected)} ({v31_tp_kill} TP killed, {v31_fp_catch} FP caught)")

        print(f"\n  SUMMARY:")
        print(f"    V30c regex:      -{v30c_fp_catch} FP, -{v30c_tp_kill} TP")
        print(f"    V31 convention:  -{v31_fp_catch} FP, -{v31_tp_kill} TP")

        totals["v30c_fp"] += v30c_fp_catch
        totals["v30c_tp_kill"] += v30c_tp_kill
        totals["v31_fp"] += v31_fp_catch
        totals["v31_tp_kill"] += v31_tp_kill

        v30c_rej_keys = {(lk.sentence_number, lk.component_id) for lk, _ in v30c_rejected}
        v31_rej_keys = {(lk.sentence_number, lk.component_id) for lk, _ in v31_rejected}
        only_regex = v30c_rej_keys - v31_rej_keys
        only_llm = v31_rej_keys - v30c_rej_keys

        if only_regex:
            print(f"    Regex-only ({len(only_regex)}):", end="")
            for key in sorted(only_regex):
                is_tp = key in gold
                lk = next(l for l, _ in v30c_rejected if (l.sentence_number, l.component_id) == key)
                print(f" S{key[0]}→{lk.component_name}[{'TP' if is_tp else 'FP'}]", end="")
            print()
        if only_llm:
            print(f"    LLM-only ({len(only_llm)}):", end="")
            for key in sorted(only_llm):
                is_tp = key in gold
                lk = next(l for l, _ in v31_rejected if (l.sentence_number, l.component_id) == key)
                print(f" S{key[0]}→{lk.component_name}[{'TP' if is_tp else 'FP'}]", end="")
            print()

    print(f"\n{'=' * 70}")
    print(f"  TOTALS:")
    print(f"    V30c regex:      -{totals['v30c_fp']} FP, -{totals['v30c_tp_kill']} TP killed")
    print(f"    V31 convention:  -{totals['v31_fp']} FP, -{totals['v31_tp_kill']} TP killed")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
