#!/usr/bin/env python3
"""Test two LLM-based variance reduction strategies using V33 checkpoints.

V33c — Phase 6: Universal Self-Consistency (3-pass majority vote)
  Current: 2 prompts, intersection (both must agree)
  New: 3 diverse prompts, majority vote (2/3 must agree)
  3rd prompt: ARCHITECTURAL CONTEXT angle

V33d — Phase 7: Bidirectional Coreference (forward + backward, union)
  Current: single forward pass (pronoun → component)
  New: add backward pass (component in S(n-1) → does S(n) continue?)
  Union of both passes

Both strategies use fixed checkpoints from V33 to isolate the effect.
Run each variant 3 times to measure if variance is reduced.
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
# V33c: Universal Self-Consistency for Phase 6
# ═══════════════════════════════════════════════════════════════════════════════

def validation_pass(llm, comp_names, ctx, cases, focus):
    """Single validation pass (replicates V33's _qual_validation_pass)."""
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

    data = llm.extract_json(llm.query(prompt, timeout=120))
    results = {}
    if data:
        for v in data.get("validations", []):
            idx = v.get("case", 0) - 1
            if 0 <= idx < len(cases):
                results[idx] = v.get("approve", False)
    return results


def run_v33c_validation(llm, ds_name, sent_map, components):
    """V33c: 3-pass majority vote for Phase 6 validation.

    Reloads Phase 5 candidates from checkpoint, re-runs validation with 3 diverse prompts.
    """
    p5 = load_checkpoint(ds_name, "phase5")
    candidates = p5["candidates"]

    p1 = load_checkpoint(ds_name, "phase1")
    p2 = load_checkpoint(ds_name, "phase2")

    comp_names = [c.name for c in components]

    # Separate needs_validation
    needs = [c for c in candidates if c.needs_validation]
    direct = [c for c in candidates if not c.needs_validation]

    # Code-first auto-approve (deterministic — same as V33)
    from llm_sad_sam.linkers.experimental.ilinker2_v33 import ILinker2V33
    linker = ILinker2V33(backend=LLMBackend.CLAUDE)
    linker.model_knowledge = p1["model_knowledge"]
    linker.GENERIC_COMPONENT_WORDS = p1["generic_component_words"]
    linker.GENERIC_PARTIALS = p1["generic_partials"]
    linker.learned_patterns = p2["learned_patterns"]

    p3 = load_checkpoint(ds_name, "phase3")
    linker.doc_knowledge = p3["doc_knowledge"]

    alias_map = {}
    for c in components:
        aliases = {c.name}
        if linker.doc_knowledge:
            for a, cn in linker.doc_knowledge.abbreviations.items():
                if cn == c.name:
                    aliases.add(a)
            for s, cn in linker.doc_knowledge.synonyms.items():
                if cn == c.name:
                    aliases.add(s)
            for p, cn in linker.doc_knowledge.partial_references.items():
                if cn == c.name:
                    aliases.add(p)
        alias_map[c.name] = aliases

    auto_approved = []
    llm_needed = []
    for c in needs:
        sent = sent_map.get(c.sentence_number)
        if not sent:
            continue
        matched = False
        for a in alias_map.get(c.component_name, set()):
            if len(a) >= 3:
                if a.lower() in sent.text.lower():
                    matched = True
                    break
            elif len(a) >= 2:
                if linker._word_boundary_match(a, sent.text):
                    matched = True
                    break
        if matched:
            c.confidence = 1.0
            c.source = "validated"
            auto_approved.append(c)
        else:
            llm_needed.append(c)

    print(f"    Code-first: {len(auto_approved)} auto, {len(llm_needed)} LLM-needed")

    if not llm_needed:
        return direct + auto_approved

    # Build context
    ctx = []
    if linker.learned_patterns:
        if linker.learned_patterns.action_indicators:
            ctx.append(f"ACTION: {', '.join(linker.learned_patterns.action_indicators[:4])}")
        if linker.learned_patterns.effect_indicators:
            ctx.append(f"EFFECT (reject): {', '.join(linker.learned_patterns.effect_indicators[:3])}")
        if linker.learned_patterns.subprocess_terms:
            ctx.append(f"Subprocess (reject): {', '.join(list(linker.learned_patterns.subprocess_terms)[:5])}")

    # 3 diverse prompts
    focus_prompts = [
        "Focus on ACTOR role: is the component performing an action or being described?",
        "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component, not a generic concept?",
        "Focus on ARCHITECTURAL CONTEXT: does this sentence contribute to understanding the component's role, responsibilities, or interactions within the system architecture?",
    ]

    # Run 3 passes
    twopass_approved = []
    generic_risk = set()
    if linker.model_knowledge and linker.model_knowledge.ambiguous_names:
        generic_risk |= linker.model_knowledge.ambiguous_names
    for c in components:
        if c.name.lower() in linker.GENERIC_COMPONENT_WORDS:
            generic_risk.add(c.name)

    for batch_start in range(0, len(llm_needed), 25):
        batch = llm_needed[batch_start:batch_start + 25]
        cases = []
        for i, c in enumerate(batch):
            prev = sent_map.get(c.sentence_number - 1)
            p = f"[prev: {prev.text[:35]}...] " if prev else ""
            cases.append(f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n  {p}"{c.sentence_text}"')

        # 3 passes
        results = []
        for focus in focus_prompts:
            r = validation_pass(llm, comp_names, ctx, cases, focus)
            results.append(r)

        # Majority vote: 2/3 must approve
        for i, c in enumerate(batch):
            votes = sum(1 for r in results if r.get(i, False))
            if votes >= 2:
                if c.component_name in generic_risk:
                    # Still need evidence for generic — skip for simplicity
                    pass
                else:
                    c.confidence = 1.0
                    c.source = "validated"
                    twopass_approved.append(c)

    print(f"    3-pass majority approved: {len(twopass_approved)}")
    return direct + auto_approved + twopass_approved


# ═══════════════════════════════════════════════════════════════════════════════
# V33d: Bidirectional Coreference for Phase 7
# ═══════════════════════════════════════════════════════════════════════════════

PRONOUN_PATTERN = re.compile(
    r'\b(it|its|they|them|their|this|these|the component|the service|the module|the system)\b',
    re.IGNORECASE
)


def run_v33d_coref(llm, ds_name, sent_map, sentences, components, name_to_id):
    """V33d: Bidirectional coreference — forward + backward, union."""
    p1 = load_checkpoint(ds_name, "phase1")
    p2 = load_checkpoint(ds_name, "phase2")
    p3 = load_checkpoint(ds_name, "phase3")

    comp_names = [c.name for c in components]

    # Build alias lookup for antecedent verification
    alias_set = set()
    for c in components:
        alias_set.add(c.name.lower())
    dk = p3["doc_knowledge"]
    if dk:
        for a in dk.abbreviations:
            alias_set.add(a.lower())
        for s in dk.synonyms:
            alias_set.add(s.lower())
        for p in dk.partial_references:
            alias_set.add(p.lower())

    def has_mention(comp_name, text):
        """Check if component name or alias appears in text."""
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

    def verify_antecedent(comp, snum, ant_snum):
        """Verify antecedent citation (same as V33)."""
        if ant_snum is None:
            return True  # no citation to verify
        ant_sent = sent_map.get(ant_snum)
        if not ant_sent:
            return False
        if not has_mention(comp, ant_sent.text):
            return False
        if abs(snum - ant_snum) > 3:
            return False
        return True

    subprocess_terms = set()
    lp = p2["learned_patterns"]
    if lp and lp.subprocess_terms:
        subprocess_terms = lp.subprocess_terms

    def is_subprocess(text):
        text_lower = text.lower()
        return any(t.lower() in text_lower for t in subprocess_terms)

    # ── Pass 1: Forward (pronoun → component) — original approach ──
    pronoun_sents = [s for s in sentences if PRONOUN_PATTERN.search(s.text)]
    forward_links = set()

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

            if not verify_antecedent(comp, snum, ant_snum):
                continue

            sent = sent_map.get(snum)
            if sent and is_subprocess(sent.text):
                continue
            forward_links.add((snum, name_to_id[comp], comp))

    print(f"    Forward pass: {len(forward_links)} coref links")

    # ── Pass 2: Backward (component in S(n-1) → does S(n) continue?) ──
    # Find sentences where a component was explicitly mentioned,
    # then check if the NEXT sentence continues discussing it via pronoun
    backward_links = set()

    # Build component-mention map: which sentences explicitly mention which components
    comp_mentions = {}  # snum -> list of component names
    for s in sentences:
        mentioned = []
        for c in components:
            if has_mention(c.name, s.text):
                mentioned.append(c.name)
        if mentioned:
            comp_mentions[s.number] = mentioned

    # Find continuation candidates: S(n) has pronoun, S(n-1) or S(n-2) mentions component
    continuation_cases = []
    for s in sentences:
        if not PRONOUN_PATTERN.search(s.text):
            continue
        # Check previous 1-2 sentences for component mentions
        for offset in [1, 2]:
            prev_snum = s.number - offset
            if prev_snum in comp_mentions:
                for comp_name in comp_mentions[prev_snum]:
                    continuation_cases.append((s, comp_name, prev_snum))

    if continuation_cases:
        # Batch and ask LLM
        for batch_start in range(0, len(continuation_cases), 15):
            batch = continuation_cases[batch_start:batch_start + 15]

            prompt = f"""Does a pronoun in the current sentence refer to the given architecture component?

COMPONENTS: {', '.join(comp_names)}

"""
            for i, (sent, comp_name, ant_snum) in enumerate(batch):
                ant_sent = sent_map.get(ant_snum)
                prompt += f"Case {i+1}:\n"
                prompt += f"  S{ant_snum}: {ant_sent.text}\n"
                if sent.number - ant_snum > 1:
                    mid = sent_map.get(ant_snum + 1)
                    if mid:
                        prompt += f"  S{mid.number}: {mid.text}\n"
                prompt += f"  S{sent.number}: {sent.text}\n"
                prompt += f"  Component: {comp_name}\n\n"

            prompt += """Return JSON:
{"continuations": [{"case": 1, "continues": true/false}]}
JSON only:"""

            data = llm.extract_json(llm.query(prompt, timeout=150))
            if not data:
                continue

            for res in data.get("continuations", []):
                idx = res.get("case", 0) - 1
                if idx < 0 or idx >= len(batch):
                    continue
                if res.get("continues", False):
                    sent, comp_name, ant_snum = batch[idx]
                    if is_subprocess(sent.text):
                        continue
                    backward_links.add((sent.number, name_to_id[comp_name], comp_name))

    print(f"    Backward pass: {len(backward_links)} coref links")

    # Report both strategies
    union = forward_links | backward_links
    intersect = forward_links & backward_links
    fwd_only = forward_links - backward_links
    bwd_only = backward_links - forward_links
    both = forward_links & backward_links
    print(f"    fwd-only={len(fwd_only)}, bwd-only={len(bwd_only)}, both={len(both)}")
    print(f"    Union: {len(union)}, Intersect: {len(intersect)}")

    return union, intersect, forward_links, backward_links


# ═══════════════════════════════════════════════════════════════════════════════
# Main test runner
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    selected = sys.argv[1:] if len(sys.argv) > 1 else ["teammates", "bigbluebutton"]
    variant = os.environ.get("VARIANT", "both")  # "v33c", "v33d", or "both"

    llm = LLMClient(backend=LLMBackend.CLAUDE)

    for ds_name in selected:
        paths = DATASETS[ds_name]
        gold = load_gold(str(paths["gold"]))
        components = parse_pcm_repository(str(paths["model"]))
        id_to_name = {c.id: c.name for c in components}
        name_to_id = {c.name: c.id for c in components}
        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        sent_map = DocumentLoader.build_sent_map(sentences)

        # Load baseline checkpoint data
        p4 = load_checkpoint(ds_name, "phase4")
        p6 = load_checkpoint(ds_name, "phase6")
        p7 = load_checkpoint(ds_name, "phase7")
        pf = load_checkpoint(ds_name, "final")

        seed_set = {(l.sentence_number, l.component_id) for l in p4["transarc_links"]}
        baseline_validated = {(l.sentence_number, l.component_id) for l in p6["validated"]}
        baseline_coref = {(l.sentence_number, l.component_id) for l in p7["coref_links"]}
        baseline_final = {(l.sentence_number, l.component_id) for l in pf["final"]}

        print(f"\n{'='*100}")
        print(f"DATASET: {ds_name} (gold={len(gold)} links)")
        print(f"{'='*100}")

        b_tp, b_fp, _, b_p, b_r, b_f1 = eval_links(baseline_final, gold)
        print(f"  Baseline final: {len(baseline_final)} links, {b_tp} TP, {b_fp} FP, F1={b_f1:.1%}")
        bv_tp, bv_fp, _, _, _, _ = eval_links(baseline_validated - seed_set, gold)
        print(f"  Baseline validated (non-seed): {len(baseline_validated - seed_set)} links ({bv_tp} TP, {bv_fp} FP)")
        bc_tp, bc_fp, _, _, _, _ = eval_links(baseline_coref, gold)
        print(f"  Baseline coref: {len(baseline_coref)} links ({bc_tp} TP, {bc_fp} FP)")

        # ── V33c: 3-pass majority validation ──
        if variant in ("v33c", "both"):
            print(f"\n  ── V33c: Universal Self-Consistency (Phase 6) ──")
            v33c_runs = []
            for i in range(N_RUNS):
                print(f"\n  Run {i+1}/{N_RUNS}:")
                t0 = time.time()
                validated = run_v33c_validation(llm, ds_name, sent_map, components)
                elapsed = time.time() - t0
                val_set = {(c.sentence_number, c.component_id) for c in validated}
                val_new = val_set - seed_set
                tp, fp, _, _, _, _ = eval_links(val_new, gold)
                print(f"    Validated (non-seed): {len(val_new)} links ({tp} TP, {fp} FP) ({elapsed:.0f}s)")
                v33c_runs.append(val_new)

            # Variance analysis
            union = set()
            inter = v33c_runs[0].copy()
            for s in v33c_runs:
                union |= s
                inter &= s
            variant_links = union - inter
            print(f"\n  V33c VARIANCE:")
            print(f"    Stable: {len(inter)}, Variant: {len(variant_links)}, Union: {len(union)}")
            print(f"    Baseline variance: compare with E2E test (validated had 9 variant links on teammates)")
            if variant_links:
                counts = Counter()
                for s in v33c_runs:
                    for k in s:
                        counts[k] += 1
                for (snum, cid) in sorted(variant_links):
                    is_tp = (snum, cid) in gold
                    c = counts[(snum, cid)]
                    print(f"      {'TP' if is_tp else 'FP'}: S{snum} -> {id_to_name.get(cid, cid)} ({c}/{N_RUNS})")

        # ── V33d: Bidirectional coreference ──
        if variant in ("v33d", "both"):
            print(f"\n  ── V33d: Bidirectional Coreference (Phase 7) ──")
            runs_union = []
            runs_intersect = []
            runs_fwd = []
            runs_bwd = []
            for i in range(N_RUNS):
                print(f"\n  Run {i+1}/{N_RUNS}:")
                t0 = time.time()
                union_set, intersect_set, fwd_set, bwd_set = run_v33d_coref(
                    llm, ds_name, sent_map, sentences, components, name_to_id)
                elapsed = time.time() - t0

                def to_set(s):
                    return {(snum, cid) for snum, cid, _ in s}

                u, i_s, f, b = to_set(union_set), to_set(intersect_set), to_set(fwd_set), to_set(bwd_set)
                for label, s in [("union", u), ("intersect", i_s), ("forward", f), ("backward", b)]:
                    tp, fp, _, _, _, _ = eval_links(s, gold)
                    print(f"    {label:10s}: {len(s):2d} links ({tp} TP, {fp} FP)")
                runs_union.append(u)
                runs_intersect.append(i_s)
                runs_fwd.append(f)
                runs_bwd.append(b)
                print(f"    ({elapsed:.0f}s)")

            # Variance analysis for each strategy
            print(f"\n  VARIANCE COMPARISON (baseline coref: {len(baseline_coref)} links, {bc_tp} TP, {bc_fp} FP):")
            for label, runs in [("forward", runs_fwd), ("backward", runs_bwd),
                                ("union", runs_union), ("intersect", runs_intersect)]:
                all_u = set()
                all_i = runs[0].copy()
                for s in runs:
                    all_u |= s
                    all_i &= s
                var = all_u - all_i
                # Quality of stable set
                tp_s, fp_s, _, _, _, _ = eval_links(all_i, gold)
                tp_u, fp_u, _, _, _, _ = eval_links(all_u, gold)
                print(f"    {label:10s}: stable={len(all_i)}({tp_s}TP/{fp_s}FP) variant={len(var)} union={len(all_u)}({tp_u}TP/{fp_u}FP)")


if __name__ == "__main__":
    main()
