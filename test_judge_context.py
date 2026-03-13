#!/usr/bin/env python3
"""Unit test: Context-aware judge vs blind judge on BBB syn-safe links.

Loads V30c BBB pre_judge checkpoint, takes the 21 links that syn-safe
currently protects, and sends them to TWO judge variants:
  1. BLIND: current judge (no Phase 3 context) — should kill ~14 TPs
  2. CONTEXT-AWARE: judge with Phase 3 synonym/partial mappings — should keep TPs, kill FPs

This tests the hypothesis that the judge's flaw is lack of context,
not lack of capability.
"""
import csv
import glob
import os
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.llm_client import LLMClient, LLMBackend
from llm_sad_sam.core.document_loader import DocumentLoader

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
V30C_CACHE = Path("./results/phase_cache/v30c")

# Ensure sonnet
os.environ.setdefault("CLAUDE_MODEL", "sonnet")


def load_gold(dataset):
    gold_path = BENCHMARK_BASE / dataset
    pattern = str(gold_path / "**" / "goldstandard_sad_*-sam_*.csv")
    files = [f for f in glob.glob(pattern, recursive=True)
             if "UME" not in f and "code" not in f]
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


def get_syn_safe_links(prelim_links, doc_knowledge, sentences):
    """Identify which links would be syn-safe in the current pipeline."""
    sent_map = {s.number: s for s in sentences}
    syn_safe = []
    for l in prelim_links:
        sent = sent_map.get(l.sentence_number)
        if not sent:
            continue
        text_lower = sent.text.lower()
        is_syn = False
        # Check synonyms
        for syn, target in doc_knowledge.synonyms.items():
            if target == l.component_name:
                if re.search(rf'\b{re.escape(syn.lower())}\b', text_lower):
                    is_syn = True
                    break
        if not is_syn:
            # Check partials
            for partial, target in doc_knowledge.partial_references.items():
                if target == l.component_name:
                    if re.search(rf'\b{re.escape(partial.lower())}\b', text_lower):
                        is_syn = True
                        break
        if is_syn:
            syn_safe.append(l)
    return syn_safe


def build_context_string(doc_knowledge, comp_name):
    """Build Phase 3 context for a specific component."""
    parts = []
    for syn, target in doc_knowledge.synonyms.items():
        if target == comp_name:
            parts.append(f'"{syn}" is a known synonym for {comp_name}')
    for partial, target in doc_knowledge.partial_references.items():
        if target == comp_name:
            parts.append(f'"{partial}" is a known short name for {comp_name}')
    for abbr, target in doc_knowledge.abbreviations.items():
        if target == comp_name:
            parts.append(f'"{abbr}" is an abbreviation for {comp_name}')
    return "; ".join(parts) if parts else ""


def blind_judge(llm, snum, comp_name, sent_text, comp_names):
    """Current judge: no Phase 3 context."""
    prompt = f"""JUDGE: Should sentence S{snum} be linked to component "{comp_name}"?

SENTENCE: {sent_text}
ALL COMPONENTS: {', '.join(comp_names)}

APPROVAL CRITERIA — approve only if ALL four conditions are met:
1. EXPLICIT REFERENCE: The component name (or a direct reference) appears as a clear entity being discussed.
2. SYSTEM-LEVEL PERSPECTIVE: Describes component's role, responsibilities, or interactions within system architecture.
3. PRIMARY FOCUS: The component is the main subject, not incidental.
4. COMPONENT-SPECIFIC USAGE: Refers to the component as a named entity, not a generic concept sharing the name.

Return JSON: {{"approve": true/false, "reason": "brief explanation"}}
JSON only:"""
    data = llm.extract_json(llm.query(prompt, timeout=60))
    if data:
        return data.get("approve", True), data.get("reason", "")
    return True, "parse failure"


def context_judge(llm, snum, comp_name, sent_text, comp_names, context_str):
    """Context-aware judge v3: necessary-but-not-sufficient, with nuanced modifier handling."""
    prompt = f"""JUDGE: Should sentence S{snum} be linked to component "{comp_name}"?

DOCUMENT ANALYSIS CONTEXT:
{context_str}
These mappings were confirmed by prior analysis of this document's terminology.

SENTENCE: {sent_text}
ALL COMPONENTS: {', '.join(comp_names)}

IMPORTANT: A synonym/short name appearing in the sentence is NECESSARY but NOT SUFFICIENT.

Step 1 — REFERENCE CHECK: Does the component name or a confirmed synonym/short name appear
in the sentence as a reference to the component? REJECT if:
  - It is a hyphenated modifier with no standalone usage (e.g., "client-side communication")
  - It is part of a different proper name not in the synonym list
  - It is only in a transitional/navigational phrase (e.g., "below the X diagram")
  NOTE: "the client", "the server", "on the client side" DO count as references when the
  word is used as a noun referring to a system participant, even in a prepositional phrase.

Step 2 — ARCHITECTURAL RELEVANCE: Does the sentence describe what this component does,
how it interacts with other components, or its role in the system? A sentence that merely
mentions the component as a location or recipient without describing its behavior is still
relevant if the interaction itself is architectural.

Step 3 — NOT PURELY INCIDENTAL: The component should be a meaningful participant in what
the sentence describes, not just a passing mention in a list or aside.

Return JSON: {{"approve": true/false, "reason": "brief explanation"}}
JSON only:"""
    data = llm.extract_json(llm.query(prompt, timeout=60))
    if data:
        return data.get("approve", True), data.get("reason", "")
    return True, "parse failure"


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "bigbluebutton"
    backend_str = os.environ.get("LLM_BACKEND", "claude")
    backend = LLMBackend.OPENAI if backend_str == "openai" else LLMBackend.CLAUDE

    print(f"Dataset: {dataset}, Backend: {backend_str}")

    # Load checkpoints
    with open(V30C_CACHE / dataset / "pre_judge.pkl", "rb") as f:
        pj = pickle.load(f)
    with open(V30C_CACHE / dataset / "phase3.pkl", "rb") as f:
        p3 = pickle.load(f)
    with open(V30C_CACHE / dataset / "phase1.pkl", "rb") as f:
        p1 = pickle.load(f)

    dk = p3['doc_knowledge']
    prelim = pj['preliminary']
    gold = load_gold(dataset)

    # Load sentences
    # Find text file
    text_path = None
    for pattern in [f"text_*/{dataset}.txt", f"{dataset}_*.txt"]:
        hits = list((BENCHMARK_BASE / dataset).glob(pattern))
        if hits:
            text_path = str(hits[0])
            break
    if not text_path:
        print(f"ERROR: text file not found for {dataset}")
        sys.exit(1)
    loader = DocumentLoader()
    sentences = loader.load_sentences(text_path)
    sent_map = {s.number: s for s in sentences}

    # Get component names
    comp_names = sorted(set(l.component_name for l in prelim))

    # Find syn-safe links
    syn_safe = get_syn_safe_links(prelim, dk, sentences)
    print(f"\nSyn-safe links: {len(syn_safe)}")

    # Classify each as TP or FP
    tp_links = [l for l in syn_safe if (l.sentence_number, l.component_id) in gold]
    fp_links = [l for l in syn_safe if (l.sentence_number, l.component_id) not in gold]
    print(f"  TPs: {len(tp_links)}, FPs: {len(fp_links)}")

    # Show all syn-safe links
    print(f"\nAll syn-safe links:")
    for l in sorted(syn_safe, key=lambda x: x.sentence_number):
        is_tp = (l.sentence_number, l.component_id) in gold
        ctx = build_context_string(dk, l.component_name)
        sent = sent_map.get(l.sentence_number)
        label = "TP" if is_tp else "FP"
        print(f"  [{label}] S{l.sentence_number} -> {l.component_name} [{l.source}]")
        if sent:
            print(f"       Text: {sent.text[:100]}...")
        print(f"       Context: {ctx}")

    # Run both judges
    llm = LLMClient(backend=backend)

    print(f"\n{'='*80}")
    print(f"JUDGE COMPARISON (blind vs context-aware)")
    print(f"{'='*80}")

    blind_results = {"tp_kept": 0, "tp_killed": 0, "fp_kept": 0, "fp_killed": 0}
    ctx_results = {"tp_kept": 0, "tp_killed": 0, "fp_kept": 0, "fp_killed": 0}

    for l in sorted(syn_safe, key=lambda x: x.sentence_number):
        sent = sent_map.get(l.sentence_number)
        if not sent:
            continue
        is_tp = (l.sentence_number, l.component_id) in gold
        label = "TP" if is_tp else "FP"
        ctx_str = build_context_string(dk, l.component_name)

        # Run blind judge
        b_approve, b_reason = blind_judge(llm, l.sentence_number, l.component_name,
                                          sent.text, comp_names)
        # Run context judge
        c_approve, c_reason = context_judge(llm, l.sentence_number, l.component_name,
                                            sent.text, comp_names, ctx_str)

        # Track results
        if is_tp:
            if b_approve: blind_results["tp_kept"] += 1
            else: blind_results["tp_killed"] += 1
            if c_approve: ctx_results["tp_kept"] += 1
            else: ctx_results["tp_killed"] += 1
        else:
            if b_approve: blind_results["fp_kept"] += 1
            else: blind_results["fp_killed"] += 1
            if c_approve: ctx_results["fp_kept"] += 1
            else: ctx_results["fp_killed"] += 1

        # Show disagreements
        if b_approve != c_approve:
            print(f"\n  DISAGREE: [{label}] S{l.sentence_number} -> {l.component_name}")
            print(f"    Blind:   {'APPROVE' if b_approve else 'REJECT'} — {b_reason}")
            print(f"    Context: {'APPROVE' if c_approve else 'REJECT'} — {c_reason}")
        else:
            status = "APPROVE" if b_approve else "REJECT"
            outcome = "correct" if (is_tp == b_approve) else "WRONG"
            print(f"  AGREE {status}: [{label}] S{l.sentence_number} -> {l.component_name} ({outcome})")

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"  Total syn-safe links: {len(syn_safe)} ({len(tp_links)} TP, {len(fp_links)} FP)")
    print(f"")
    print(f"  BLIND JUDGE (no Phase 3 context):")
    print(f"    TPs kept:   {blind_results['tp_kept']}")
    print(f"    TPs killed: {blind_results['tp_killed']}  ← PROBLEM")
    print(f"    FPs killed: {blind_results['fp_killed']}")
    print(f"    FPs kept:   {blind_results['fp_kept']}")
    print(f"")
    print(f"  CONTEXT-AWARE JUDGE (with Phase 3 synonyms):")
    print(f"    TPs kept:   {ctx_results['tp_kept']}")
    print(f"    TPs killed: {ctx_results['tp_killed']}")
    print(f"    FPs killed: {ctx_results['fp_killed']}  ← GOAL")
    print(f"    FPs kept:   {ctx_results['fp_kept']}")
    print(f"")
    b_correct = blind_results['tp_kept'] + blind_results['fp_killed']
    c_correct = ctx_results['tp_kept'] + ctx_results['fp_killed']
    print(f"  Accuracy: blind={b_correct}/{len(syn_safe)}, context={c_correct}/{len(syn_safe)}")


if __name__ == "__main__":
    main()
