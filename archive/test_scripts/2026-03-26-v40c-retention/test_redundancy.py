#!/usr/bin/env python3
"""Test redundancy-based variance reduction: repeat high-precision steps, union results.

V33e — Phase 5: Run entity extraction twice, merge candidates, validate once
V33f — Phase 7: Run forward coref twice, union results

Both strategies use V33 checkpoints. Run each 3 times to measure variance reduction.
"""

import csv
import os
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

os.environ["CLAUDE_MODEL"] = "sonnet"

from llm_sad_sam.core.data_types import SadSamLink, CandidateLink, DiscourseContext
from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)

DATASETS = {
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore/text_2016/mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore/model_2016/pcm/ms.repository",
        "gold": BENCHMARK_BASE / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore/text_2020/teastore.txt",
        "model": BENCHMARK_BASE / "teastore/model_2020/pcm/teastore.repository",
        "gold": BENCHMARK_BASE / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    },
    "teammates": {
        "text": BENCHMARK_BASE / "teammates/text_2021/teammates.txt",
        "model": BENCHMARK_BASE / "teammates/model_2021/pcm/teammates.repository",
        "gold": BENCHMARK_BASE / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
        "gold": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref/text_2021/jabref.txt",
        "model": BENCHMARK_BASE / "jabref/model_2021/pcm/jabref.repository",
        "gold": BENCHMARK_BASE / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    },
}

N_RUNS = 3
PRONOUN_PATTERN = re.compile(
    r'\b(it|its|they|them|their|this|these|the component|the service|the module|the system)\b',
    re.IGNORECASE
)


def load_gold(gold_path):
    links = set()
    with open(gold_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                links.add((int(snum), cid))
    return links


def load_checkpoint(ds_name, phase):
    path = Path(f"./results/phase_cache/v33/{ds_name}/{phase}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def eval_links(link_set, gold):
    tp = len(link_set & gold)
    fp = len(link_set - gold)
    fn = len(gold - link_set)
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return tp, fp, fn, p, r, f1


# ═══════════════════════════════════════════════════════════════════════════════
# V33e: Phase 5 extraction ×2, union candidates, validate once
# ═══════════════════════════════════════════════════════════════════════════════

def run_extraction_once(llm, sentences, components, name_to_id, sent_map, doc_knowledge):
    """Single Phase 5 extraction pass. Returns set of (snum, comp_id) candidates."""
    comp_names = [c.name for c in components]
    comp_lower = {n.lower() for n in comp_names}

    mappings = []
    if doc_knowledge:
        mappings.extend([f"{a}={c}" for a, c in doc_knowledge.abbreviations.items()])
        mappings.extend([f"{s}={c}" for s, c in doc_knowledge.synonyms.items()])
        mappings.extend([f"{p}={c}" for p, c in doc_knowledge.partial_references.items()])

    batch_size = 50
    all_candidates = {}

    for batch_start in range(0, len(sentences), batch_size):
        batch = sentences[batch_start:batch_start + batch_size]

        prompt = f"""Extract ALL references to software architecture components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings[:20])}' if mappings else ''}

RULES — include a reference when:
1. The component name (or known alias) appears directly in the sentence
2. A space-separated form matches a compound name (e.g., "Memory Manager" → MemoryManager)
3. The sentence describes what a specific component does by name or role
4. A known synonym or partial reference is used
5. The component participates in an interaction described in the sentence (as sender, receiver, or target)
6. The component is mentioned in a passive or prepositional phrase

RULES — exclude when:
1. The name appears only inside a dotted path (e.g., com.example.name)
2. The name is used as an ordinary English word, not as a component reference

Favor inclusion over exclusion — later validation will filter borderline cases.

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence", "match_type": "exact|synonym|partial|functional"}}]}}
JSON only:"""

        for attempt in range(2):
            data = llm.extract_json(llm.query(prompt, timeout=240))
            if data and data.get("references"):
                break

        if not data:
            continue

        for ref in data.get("references", []):
            snum, cname = ref.get("sentence"), ref.get("component")
            if not (snum and cname and cname in name_to_id):
                continue
            if isinstance(snum, str):
                snum = snum.lstrip("S")
            try:
                snum = int(snum)
            except (ValueError, TypeError):
                continue
            sent = sent_map.get(snum)
            if not sent:
                continue

            matched = ref.get("matched_text", "")
            if matched and matched.lower() not in sent.text.lower():
                continue

            key = (snum, name_to_id[cname])
            if key not in all_candidates:
                all_candidates[key] = CandidateLink(
                    snum, sent.text, cname, name_to_id[cname],
                    matched, 0.85, "entity",
                    ref.get("match_type", "exact"), True)

    return all_candidates


def run_v33e_extraction(llm, ds_name, sent_map, sentences, components, name_to_id):
    """V33e: Run extraction twice, union candidates."""
    p3 = load_checkpoint(ds_name, "phase3")
    dk = p3["doc_knowledge"]

    pass1 = run_extraction_once(llm, sentences, components, name_to_id, sent_map, dk)
    print(f"    Pass 1: {len(pass1)} candidates")
    pass2 = run_extraction_once(llm, sentences, components, name_to_id, sent_map, dk)
    print(f"    Pass 2: {len(pass2)} candidates")

    # Union
    merged = dict(pass1)
    for k, v in pass2.items():
        if k not in merged:
            merged[k] = v

    only1 = set(pass1.keys()) - set(pass2.keys())
    only2 = set(pass2.keys()) - set(pass1.keys())
    both = set(pass1.keys()) & set(pass2.keys())
    print(f"    Union: {len(merged)} (both={len(both)}, pass1-only={len(only1)}, pass2-only={len(only2)})")

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# V33f: Phase 7 forward coref ×2, union results
# ═══════════════════════════════════════════════════════════════════════════════

def run_forward_coref_once(llm, sentences, components, name_to_id, sent_map, doc_knowledge, learned_patterns):
    """Single forward coref pass. Returns set of (snum, comp_id, comp_name)."""
    comp_names = [c.name for c in components]
    dk = doc_knowledge

    def has_mention(comp_name, text):
        text_lower = text.lower()
        if comp_name.lower() in text_lower:
            return True
        if dk:
            for a, cn in dk.abbreviations.items():
                if cn == comp_name and a.lower() in text_lower:
                    return True
            for s, cn in dk.synonyms.items():
                if cn == comp_name and s.lower() in text_lower:
                    return True
            for p, cn in dk.partial_references.items():
                if cn == comp_name and p.lower() in text_lower:
                    return True
        return False

    subprocess_terms = set()
    if learned_patterns and learned_patterns.subprocess_terms:
        subprocess_terms = learned_patterns.subprocess_terms

    def is_subprocess(text):
        text_lower = text.lower()
        return any(t.lower() in text_lower for t in subprocess_terms)

    pronoun_sents = [s for s in sentences if PRONOUN_PATTERN.search(s.text)]
    links = set()

    for batch_start in range(0, len(pronoun_sents), 12):
        batch = pronoun_sents[batch_start:batch_start + 12]

        prompt = f"""Resolve pronoun references to architecture components.

COMPONENTS: {', '.join(comp_names)}

"""
        for i, sent in enumerate(batch):
            prompt += f"--- Case {i+1}: S{sent.number} ---\n"
            prev = []
            for j in range(1, 4):
                p = sent_map.get(sent.number - j)
                if p:
                    prev.append(f"S{p.number}: {p.text}")
            if prev:
                prompt += "PREVIOUS:\n  " + "\n  ".join(reversed(prev)) + "\n"
            prompt += f">>> {sent.text}\n\n"

        prompt += """For each pronoun that refers to a component, provide:
- antecedent_sentence: the sentence number where the component was EXPLICITLY NAMED
- antecedent_text: the EXACT quote from that sentence containing the component name

RULES (all must hold):
1. The component name (or known alias) MUST appear verbatim in the antecedent sentence
2. The antecedent MUST be within the previous 3 sentences
3. The pronoun MUST grammatically refer back to that component as its subject
4. If the pronoun could refer to multiple things, DO NOT resolve it

Return JSON:
{"resolutions": [{"case": 1, "sentence": N_INTEGER, "pronoun": "it", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact text with component name"}]}

Only include resolutions you are CERTAIN about. JSON only:"""

        data = llm.extract_json(llm.query(prompt, timeout=150))
        if not data:
            continue

        for res in data.get("resolutions", []):
            comp = res.get("component")
            snum = res.get("sentence")
            if not (comp and snum and comp in name_to_id):
                continue
            if isinstance(snum, str):
                snum = snum.lstrip("S")
            try:
                snum = int(snum)
            except (ValueError, TypeError):
                continue

            ant_snum = res.get("antecedent_sentence")
            if ant_snum is not None:
                if isinstance(ant_snum, str):
                    ant_snum = ant_snum.lstrip("S")
                try:
                    ant_snum = int(ant_snum)
                except (ValueError, TypeError):
                    ant_snum = None

            if ant_snum is not None:
                ant_sent = sent_map.get(ant_snum)
                if not ant_sent:
                    continue
                if not has_mention(comp, ant_sent.text):
                    continue
                if abs(snum - ant_snum) > 3:
                    continue

            sent = sent_map.get(snum)
            if sent and is_subprocess(sent.text):
                continue
            links.add((snum, name_to_id[comp]))

    return links


def run_v33f_coref(llm, ds_name, sent_map, sentences, components, name_to_id):
    """V33f: Run forward coref twice, union."""
    p2 = load_checkpoint(ds_name, "phase2")
    p3 = load_checkpoint(ds_name, "phase3")

    pass1 = run_forward_coref_once(llm, sentences, components, name_to_id, sent_map,
                                    p3["doc_knowledge"], p2["learned_patterns"])
    print(f"    Pass 1: {len(pass1)} links")
    pass2 = run_forward_coref_once(llm, sentences, components, name_to_id, sent_map,
                                    p3["doc_knowledge"], p2["learned_patterns"])
    print(f"    Pass 2: {len(pass2)} links")

    union = pass1 | pass2
    only1 = pass1 - pass2
    only2 = pass2 - pass1
    both = pass1 & pass2
    print(f"    Union: {len(union)} (both={len(both)}, p1-only={len(only1)}, p2-only={len(only2)})")

    return union


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    selected = sys.argv[1:] if len(sys.argv) > 1 else ["teammates", "bigbluebutton"]
    variant = os.environ.get("VARIANT", "both")  # "v33e", "v33f", or "both"

    llm = LLMClient(backend=LLMBackend.CLAUDE)

    for ds_name in selected:
        paths = DATASETS[ds_name]
        gold = load_gold(str(paths["gold"]))
        components = parse_pcm_repository(str(paths["model"]))
        id_to_name = {c.id: c.name for c in components}
        name_to_id = {c.name: c.id for c in components}
        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        sent_map = DocumentLoader.build_sent_map(sentences)

        p4 = load_checkpoint(ds_name, "phase4")
        p5 = load_checkpoint(ds_name, "phase5")
        p7 = load_checkpoint(ds_name, "phase7")

        seed_set = {(l.sentence_number, l.component_id) for l in p4["transarc_links"]}
        baseline_candidates = {(c.sentence_number, c.component_id) for c in p5["candidates"]}
        baseline_coref = {(l.sentence_number, l.component_id) for l in p7["coref_links"]}

        print(f"\n{'='*100}")
        print(f"DATASET: {ds_name} (gold={len(gold)})")
        print(f"{'='*100}")

        bc_new = baseline_candidates - seed_set
        bc_tp, bc_fp, _, _, _, _ = eval_links(bc_new, gold)
        print(f"  Baseline P5 candidates (non-seed): {len(bc_new)} ({bc_tp} TP, {bc_fp} FP)")
        b7_tp, b7_fp, _, _, _, _ = eval_links(baseline_coref, gold)
        print(f"  Baseline P7 coref: {len(baseline_coref)} ({b7_tp} TP, {b7_fp} FP)")

        # ── V33e: extraction ×2 ──
        if variant in ("v33e", "both"):
            print(f"\n  ── V33e: Extraction ×2, union candidates (Phase 5) ──")
            runs = []
            for i in range(N_RUNS):
                print(f"\n  Run {i+1}/{N_RUNS}:")
                t0 = time.time()
                candidates = run_v33e_extraction(llm, ds_name, sent_map, sentences, components, name_to_id)
                elapsed = time.time() - t0
                cand_new = set(candidates.keys()) - seed_set
                tp, fp, _, _, _, _ = eval_links(cand_new, gold)
                print(f"    Non-seed candidates: {len(cand_new)} ({tp} TP, {fp} FP) ({elapsed:.0f}s)")
                runs.append(cand_new)

            # Variance
            all_u = set()
            all_i = runs[0].copy()
            for s in runs:
                all_u |= s
                all_i &= s
            var = all_u - all_i
            tp_s, fp_s, _, _, _, _ = eval_links(all_i, gold)
            tp_u, fp_u, _, _, _, _ = eval_links(all_u, gold)
            print(f"\n  V33e VARIANCE:")
            print(f"    Stable: {len(all_i)} ({tp_s}TP/{fp_s}FP)  Variant: {len(var)}  Union: {len(all_u)} ({tp_u}TP/{fp_u}FP)")
            print(f"    Baseline: {len(bc_new)} ({bc_tp}TP/{bc_fp}FP)")
            if var:
                counts = Counter()
                for s in runs:
                    for k in s:
                        counts[k] += 1
                for (snum, cid) in sorted(var):
                    is_tp = (snum, cid) in gold
                    c = counts[(snum, cid)]
                    print(f"      {'TP' if is_tp else 'FP'}: S{snum} -> {id_to_name.get(cid, cid)} ({c}/{N_RUNS})")

        # ── V33f: forward coref ×2 ──
        if variant in ("v33f", "both"):
            print(f"\n  ── V33f: Forward coref ×2, union (Phase 7) ──")
            runs = []
            for i in range(N_RUNS):
                print(f"\n  Run {i+1}/{N_RUNS}:")
                t0 = time.time()
                coref_set = run_v33f_coref(llm, ds_name, sent_map, sentences, components, name_to_id)
                elapsed = time.time() - t0
                tp, fp, _, _, _, _ = eval_links(coref_set, gold)
                print(f"    Coref: {len(coref_set)} ({tp} TP, {fp} FP) ({elapsed:.0f}s)")
                runs.append(coref_set)

            # Variance
            all_u = set()
            all_i = runs[0].copy()
            for s in runs:
                all_u |= s
                all_i &= s
            var = all_u - all_i
            tp_s, fp_s, _, _, _, _ = eval_links(all_i, gold)
            tp_u, fp_u, _, _, _, _ = eval_links(all_u, gold)
            print(f"\n  V33f VARIANCE:")
            print(f"    Stable: {len(all_i)} ({tp_s}TP/{fp_s}FP)  Variant: {len(var)}  Union: {len(all_u)} ({tp_u}TP/{fp_u}FP)")
            print(f"    Baseline: {len(baseline_coref)} ({b7_tp}TP/{b7_fp}FP)")
            if var:
                counts = Counter()
                for s in runs:
                    for k in s:
                        counts[k] += 1
                for (snum, cid) in sorted(var):
                    is_tp = (snum, cid) in gold
                    c = counts[(snum, cid)]
                    print(f"      {'TP' if is_tp else 'FP'}: S{snum} -> {id_to_name.get(cid, cid)} ({c}/{N_RUNS})")


if __name__ == "__main__":
    main()
