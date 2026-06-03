#!/usr/bin/env python3
"""Test P8b replacement: LLM-aware partial injection vs blind substring matching.

Current P8b (`_inject_partial_references`) does blind substring matching:
  - Phase 3 discovers "Server" → "HTML5 Server" (partial)
  - P8b scans ALL sentences for "Server" → injects link to HTML5 Server
  - Problem: "BigBlueButton server" ≠ HTML5 Server, but P8b links it anyway
  - Result on BBB: 4 TP + 4 FP (net positive but noisy)

LLM replacement: Instead of blind injection, ask LLM to disambiguate each
candidate. Feed the partial reference, the sentence, and the full component
name. Let LLM decide if the mention refers to the specific component.

Test: Single-phase on BBB using V30c checkpoints. Compare TP/FP with original.
"""

import csv
import glob
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
    id_to_name = {c.id: c.name for c in components}
    return components, sentences, sent_map, name_to_id, id_to_name


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
    """Check if term appears cleanly (not in dotted path, not hyphenated)."""
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


def original_p8b(sentences, partials, name_to_id, existing):
    """Original blind substring matching."""
    injected = []
    for partial, comp_name in partials.items():
        if comp_name not in name_to_id:
            continue
        comp_id = name_to_id[comp_name]
        for sent in sentences:
            key = (sent.number, comp_id)
            if key in existing:
                continue
            if _has_clean_mention(partial, sent.text):
                injected.append(SadSamLink(sent.number, comp_id, comp_name, 0.8, "partial_inject"))
                existing.add(key)
    return injected


def _extract_json_array(text):
    """Extract a JSON array from LLM text that may have markdown fences."""
    import json
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    # Try to find array in text
    start = text.find("[")
    if start >= 0:
        depth = 0
        for j in range(start, len(text)):
            if text[j] == '[':
                depth += 1
            elif text[j] == ']':
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(text[start:j+1])
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        break
    return None


def llm_p8b(sentences, partials, name_to_id, existing, llm, components):
    """LLM-aware partial injection v1: no extra context."""
    return _llm_p8b_impl(sentences, partials, name_to_id, existing, llm, components,
                          use_context=False)


def llm_p8b_context(sentences, partials, name_to_id, existing, llm, components,
                     existing_links=None):
    """LLM-aware partial injection v2: with surrounding context + existing links."""
    return _llm_p8b_impl(sentences, partials, name_to_id, existing, llm, components,
                          use_context=True, existing_links=existing_links,
                          all_sentences=sentences)


def llm_p8b_aliases(sentences, partials, name_to_id, existing, llm, components,
                     existing_links=None, doc_knowledge=None):
    """LLM-aware partial injection v3: context + aliases + component descriptions."""
    return _llm_p8b_impl(sentences, partials, name_to_id, existing, llm, components,
                          use_context=True, existing_links=existing_links,
                          all_sentences=sentences, doc_knowledge=doc_knowledge)


def _llm_p8b_impl(sentences, partials, name_to_id, existing, llm, components,
                    use_context=False, existing_links=None, all_sentences=None,
                    doc_knowledge=None):
    """LLM-aware partial injection implementation."""
    # Build sentence lookup
    sent_by_num = {s.number: s for s in sentences}

    # Collect all candidates
    candidates = []
    for partial, comp_name in partials.items():
        if comp_name not in name_to_id:
            continue
        comp_id = name_to_id[comp_name]
        for sent in sentences:
            key = (sent.number, comp_id)
            if key in existing:
                continue
            if _has_clean_mention(partial, sent.text):
                candidates.append({
                    "sentence_number": sent.number,
                    "sentence_text": sent.text,
                    "component_id": comp_id,
                    "component_name": comp_name,
                    "partial": partial,
                })

    if not candidates:
        return []

    comp_names = [c.name for c in components]
    id_to_name = {c.id: c.name for c in components}

    # Build alias map per component
    alias_map = defaultdict(list)
    if doc_knowledge:
        for a, c in doc_knowledge.abbreviations.items():
            alias_map[c].append(f"{a} (abbrev)")
        for s, c in doc_knowledge.synonyms.items():
            alias_map[c].append(f"{s} (synonym)")
        for p, c in doc_knowledge.partial_references.items():
            alias_map[c].append(f"{p} (partial)")

    # Build existing link map for context
    existing_comp_by_sent = defaultdict(set)
    if existing_links and use_context:
        for lk in existing_links:
            existing_comp_by_sent[lk.sentence_number].add(lk.component_name)

    # Build component knowledge block
    alias_block = ""
    if doc_knowledge:
        alias_lines = []
        for comp_name in comp_names:
            aliases = alias_map.get(comp_name, [])
            if aliases:
                alias_lines.append(f"  {comp_name}: {', '.join(aliases)}")
        if alias_lines:
            alias_block = "\nKNOWN ALIASES (from document analysis):\n" + "\n".join(alias_lines) + "\n"

    # Build items with optional context
    items = []
    for i, c in enumerate(candidates):
        snum = c["sentence_number"]

        if use_context:
            # ±2 context window
            context_lines = []
            for offset in [-2, -1, 0, 1, 2]:
                s = sent_by_num.get(snum + offset)
                if s:
                    marker = ">>>" if offset == 0 else "   "
                    linked_comps = existing_comp_by_sent.get(s.number, set())
                    link_info = f" [linked to: {', '.join(sorted(linked_comps))}]" if linked_comps else ""
                    context_lines.append(f"  {marker} S{s.number}: {s.text}{link_info}")

            items.append(
                f'{i+1}. Partial "{c["partial"]}" → Component "{c["component_name"]}"\n'
                f'   Context:\n' + '\n'.join(context_lines)
            )
        else:
            items.append(
                f'{i+1}. S{snum}: "{c["sentence_text"]}"\n'
                f'   Partial name: "{c["partial"]}" → Full component: "{c["component_name"]}"'
            )

    context_instruction = ""
    if use_context:
        context_instruction = """
CONTEXT SIGNALS — use these to help disambiguate:
- The ">>>" marker shows the target sentence. Surrounding sentences provide section context.
- "[linked to: X]" shows components already linked to that sentence by earlier pipeline phases.
  If nearby sentences link to the target component, the partial name likely refers to it too.
- Section headers (short sentences, often just a component name) indicate which component
  is being discussed. Sentences following a header about component X likely discuss X.
- KNOWN ALIASES show all names discovered for each component in the document.
  Use these to understand which component owns which name.
"""

    prompt = f"""Disambiguate partial component name references in architecture documentation.

ARCHITECTURE COMPONENTS: {', '.join(comp_names)}
{alias_block}
For each item below, a partial name (short form) was found in the sentence.
The partial name maps to a specific multi-word component. Your task: does the
partial name IN THIS SENTENCE actually refer to that specific component?

The partial name is ambiguous (a common English word like "server", "client",
"conversion"). Determine whether the sentence is about the NAMED COMPONENT
or uses the word in a different sense.
{context_instruction}
NO_LINK — the word does NOT refer to the specific component:
- Hardware/infrastructure context: "a 32 CPU core server" → physical hardware
- Different named entity: "X Media Server is a media server" → refers to X, not target
- Generic process/activity: "SVG conversion", "the conversion flow" → a process, not the component
- Technology description: "X is a server that implements SFU" → describes X's capabilities

LINK — the word DOES refer to the specific component:
- The sentence is in a section about the component (nearby sentences discuss or link to it)
- Architectural role: "applications running on the server" in a section about the server component
- Inter-component interaction where context makes clear which component is meant
- The system name + partial: "the [System] server" when the component IS the system's server

IMPORTANT: In architecture documentation, when a system has a component called "X Server",
references to "the [System] server" in sections discussing that component typically mean
the X Server component. The system's server IS the component. Do not reject these as generic.

KEY TESTS (apply in order):
1. Does a DIFFERENT named entity in the sentence claim the word? (e.g., "Kurento server" → Kurento, not target) → NO_LINK
2. Hardware/quantity modifier? ("32-core server", "dedicated server") → NO_LINK
3. Generic process/activity? ("conversion flow", "data conversion") → NO_LINK
4. Section context or existing links suggest this is about the target component? → LINK
5. Default for architectural prose → LINK

{chr(10).join(items)}

Return JSON array:
[{{"id": 1, "verdict": "LINK" or "NO_LINK", "reason": "brief reason"}}]
JSON only:"""

    raw = llm.query(prompt, timeout=180)
    data = _extract_json_array(raw.text)
    if not data:
        print("    LLM returned no parseable array — falling back to original P8b")
        return original_p8b(sentences, partials, name_to_id, existing.copy())

    # Parse verdicts
    verdicts = {}
    if isinstance(data, list):
        for item in data:
            vid = item.get("id")
            verdict = item.get("verdict", "")
            reason = item.get("reason", "")
            if vid is not None:
                verdicts[vid] = verdict.upper().strip()
                print(f"    LLM #{vid}: {verdict} — {reason}")

    injected = []
    for i, c in enumerate(candidates):
        verdict = verdicts.get(i + 1, "LINK")  # default to LINK if missing
        key = (c["sentence_number"], c["component_id"])
        if "NO" in verdict:
            print(f"    LLM REJECT: S{c['sentence_number']}→{c['component_name']} "
                  f"(partial='{c['partial']}') reason={verdicts.get(i+1, '?')}")
        else:
            injected.append(SadSamLink(
                c["sentence_number"], c["component_id"], c["component_name"],
                0.8, "partial_inject"
            ))
            existing.add(key)

    return injected


def main():
    os.environ.setdefault("CLAUDE_MODEL", "sonnet")

    print("=" * 90)
    print("  P8b Test: Original (blind substring) vs LLM-aware partial injection")
    print("=" * 90)

    for ds in DATASETS:
        data3 = load_checkpoint(ds, "phase3")
        data4 = load_checkpoint(ds, "phase4")
        data6 = load_checkpoint(ds, "phase6")
        data7 = load_checkpoint(ds, "phase7")
        if not all([data3, data4, data6, data7]):
            continue

        doc_knowledge = data3["doc_knowledge"]
        partials = doc_knowledge.partial_references
        if not partials:
            print(f"\n  {ds}: No partials — P8b is no-op")
            continue

        components, sentences, sent_map, name_to_id, id_to_name = load_dataset(ds)
        gold = load_gold(ds)

        transarc_set = data4["transarc_set"]
        validated_set = {(c.sentence_number, c.component_id) for c in data6["validated"]}
        coref_set = {(l.sentence_number, l.component_id) for l in data7["coref_links"]}
        existing = transarc_set | validated_set | coref_set

        print(f"\n{'─' * 70}")
        print(f"  {ds}: Phase 3 partials = {dict(partials)}")
        print(f"{'─' * 70}")

        # ── Original P8b ──
        orig_links = original_p8b(sentences, partials, name_to_id, existing.copy())
        orig_tp = sum(1 for l in orig_links if (l.sentence_number, l.component_id) in gold)
        orig_fp = len(orig_links) - orig_tp

        print(f"\n  ORIGINAL P8b: {len(orig_links)} links ({orig_tp} TP, {orig_fp} FP)")
        for l in orig_links:
            is_tp = (l.sentence_number, l.component_id) in gold
            label = "TP" if is_tp else "FP"
            print(f"    [{label}] S{l.sentence_number} → {l.component_name}")

        # ── LLM P8b (no context) ──
        print(f"\n  LLM P8b (no context):")
        llm = LLMClient()
        llm_links = llm_p8b(sentences, partials, name_to_id, existing.copy(), llm, components)
        llm_tp = sum(1 for l in llm_links if (l.sentence_number, l.component_id) in gold)
        llm_fp = len(llm_links) - llm_tp

        print(f"\n  LLM P8b (no ctx): {len(llm_links)} links ({llm_tp} TP, {llm_fp} FP)")

        # ── LLM P8b (with context) ──
        existing_link_list = (data4.get("transarc_links", []) +
                              [SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
                               for c in data6["validated"]] +
                              data7["coref_links"])

        print(f"\n  LLM P8b (with context):")
        ctx_links = llm_p8b_context(
            sentences, partials, name_to_id, existing.copy(), llm, components,
            existing_links=existing_link_list
        )
        ctx_tp = sum(1 for l in ctx_links if (l.sentence_number, l.component_id) in gold)
        ctx_fp = len(ctx_links) - ctx_tp
        print(f"\n  LLM P8b (ctx): {len(ctx_links)} links ({ctx_tp} TP, {ctx_fp} FP)")

        # ── LLM P8b v3 (context + aliases) ──
        print(f"\n  LLM P8b (context + aliases):")
        doc_knowledge = data3["doc_knowledge"]
        alias_links = llm_p8b_aliases(
            sentences, partials, name_to_id, existing.copy(), llm, components,
            existing_links=existing_link_list, doc_knowledge=doc_knowledge
        )
        alias_tp = sum(1 for l in alias_links if (l.sentence_number, l.component_id) in gold)
        alias_fp = len(alias_links) - alias_tp
        print(f"\n  LLM P8b (aliases): {len(alias_links)} links ({alias_tp} TP, {alias_fp} FP)")
        for l in alias_links:
            is_tp = (l.sentence_number, l.component_id) in gold
            label = "TP" if is_tp else "FP"
            print(f"    [{label}] S{l.sentence_number} → {l.component_name}")

        # ── Comparison ──
        print(f"\n  COMPARISON:")
        print(f"    Original:        {orig_tp} TP, {orig_fp} FP")
        print(f"    LLM (no ctx):    {llm_tp} TP, {llm_fp} FP")
        print(f"    LLM (ctx):       {ctx_tp} TP, {ctx_fp} FP")
        print(f"    LLM (ctx+alias): {alias_tp} TP, {alias_fp} FP")
        for label, tp, fp in [("no ctx", llm_tp, llm_fp), ("ctx", ctx_tp, ctx_fp),
                               ("ctx+alias", alias_tp, alias_fp)]:
            delta_tp = tp - orig_tp
            delta_fp = fp - orig_fp
            verdict = ("BETTER" if delta_tp >= 0 and delta_fp < 0
                       else "WORSE" if delta_tp < 0
                       else "SAME" if delta_fp == 0 and delta_tp == 0
                       else "MIXED")
            print(f"    {label}: {delta_tp:+d} TP, {delta_fp:+d} FP → {verdict}")


if __name__ == "__main__":
    main()
