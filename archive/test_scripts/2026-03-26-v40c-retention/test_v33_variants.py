#!/usr/bin/env python3
"""Test V33 boundary filter variants on BBB (the bottleneck dataset).

Reconstructs V33 pre-boundary state from checkpoints, then applies
different boundary filter strategies. Only re-runs Phase 8c + Phase 9
(LLM calls for boundary filter + judge).

Variants:
  V33  (baseline): convention filter reviews ALL non-transarc (including partial_inject)
  V33a: context-aware softer prompt for partial_inject links (knows about partial mapping)
  V33b: multi-word trailing partial immunity (skip filter for partial→multi-word component)
  V33c: combined — V33a for non-immune partials, V33b immunity for trailing-word partials
"""

import csv
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

os.environ["CLAUDE_MODEL"] = "sonnet"

from llm_sad_sam.core.data_types import SadSamLink
from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMBackend, LLMClient

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
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


def eval_metrics(predicted, gold):
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"tp": tp, "fp": fp, "fn": fn, "P": p, "R": r, "F1": f1}


def extract_json_array(text):
    text = text.strip()
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


def reconstruct_pre_boundary(ds_name, sent_map, name_to_id, components):
    """Reconstruct the link state BEFORE boundary filter from V33 checkpoints."""
    cache = Path(f"./results/phase_cache/v33/{ds_name}")

    with open(cache / "phase4.pkl", "rb") as f:
        p4 = pickle.load(f)
    with open(cache / "phase6.pkl", "rb") as f:
        p6 = pickle.load(f)
    with open(cache / "phase7.pkl", "rb") as f:
        p7 = pickle.load(f)
    with open(cache / "phase3.pkl", "rb") as f:
        p3 = pickle.load(f)

    transarc_links = p4["transarc_links"]
    transarc_set = p4["transarc_set"]
    validated = p6["validated"]
    coref_links = p7["coref_links"]
    doc_knowledge = p3["doc_knowledge"]

    # Re-run Phase 8b partial injection (deterministic, no LLM)
    existing = (transarc_set
                | {(c.sentence_number, c.component_id) for c in validated}
                | {(l.sentence_number, l.component_id) for l in coref_links})

    partial_links = []
    if doc_knowledge and doc_knowledge.partial_references:
        from llm_sad_sam.linkers.experimental.agent_linker_v26a import AgentLinkerV26a
        linker = AgentLinkerV26a.__new__(AgentLinkerV26a)
        for partial, comp_name in doc_knowledge.partial_references.items():
            if comp_name not in name_to_id:
                continue
            comp_id = name_to_id[comp_name]
            for snum, sent in sent_map.items():
                key = (snum, comp_id)
                if key in existing:
                    continue
                # _has_clean_mention: check word boundary
                if re.search(rf'\b{re.escape(partial)}\b', sent.text, re.IGNORECASE):
                    partial_links.append(SadSamLink(
                        snum, comp_id, comp_name, 0.8, "partial_inject"
                    ))
                    existing.add(key)

    # Combine + deduplicate (same as V32 pipeline)
    entity_links = [
        SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
        for c in validated
    ]
    all_links = transarc_links + entity_links + coref_links + partial_links

    SOURCE_PRIORITY = {"transarc": 4, "validated": 3, "entity": 3, "coreference": 2, "partial_inject": 1}
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

    return preliminary, transarc_set, doc_knowledge


# ══════════════════════════════════════════════════════════════════════
# VARIANT BOUNDARY FILTERS
# ══════════════════════════════════════════════════════════════════════

# Import V32's CONVENTION_GUIDE
from llm_sad_sam.linkers.experimental.ilinker2_v32 import CONVENTION_GUIDE


def boundary_v33_baseline(links, sent_map, transarc_set, comp_names, doc_knowledge, llm):
    """V33 baseline: convention filter reviews ALL non-transarc."""
    safe = []
    to_review = []
    for lk in links:
        if (lk.sentence_number, lk.component_id) in transarc_set:
            safe.append(lk)
        else:
            to_review.append(lk)

    if not to_review:
        return safe, []

    items = []
    for i, lk in enumerate(to_review):
        sent = sent_map.get(lk.sentence_number)
        text = sent.text if sent else "(no text)"
        items.append(f'{i+1}. S{lk.sentence_number}: "{text}"\n   Component: "{lk.component_name}"')

    prompt = f"""Validate trace links between architecture documentation and components.

ARCHITECTURE COMPONENTS: {', '.join(comp_names)}

{CONVENTION_GUIDE}

---

For each sentence-component pair, apply the 3-step reasoning guide.
Decide LINK (keep the trace link) or NO_LINK (reject it).

{chr(10).join(items)}

Return JSON array:
[{{"id": N, "step": "1|2a|2b|3", "verdict": "LINK" or "NO_LINK", "reason": "brief"}}]
JSON only:"""

    raw = llm.query(prompt, timeout=180)
    data = extract_json_array(raw.text if hasattr(raw, 'text') else str(raw))
    verdicts = {}
    if data:
        for item in data:
            vid = item.get("id")
            verdict = item.get("verdict", "LINK").upper().strip()
            step = item.get("step", "3")
            reason = item.get("reason", "")
            if vid is not None:
                verdicts[vid] = (verdict, step, reason)

    kept = list(safe)
    rejected = []
    for i, lk in enumerate(to_review):
        verdict, step, reason = verdicts.get(i + 1, ("LINK", "3", "default"))
        if "NO" in verdict:
            rejected.append((lk, f"step{step}", reason))
        else:
            kept.append(lk)

    return kept, rejected


def boundary_v33a(links, sent_map, transarc_set, comp_names, doc_knowledge, llm):
    """V33a: context-aware softer prompt for partial_inject links."""
    safe = []
    standard_review = []
    partial_review = []

    # Build reverse partial map
    partial_map = {}
    if doc_knowledge and doc_knowledge.partial_references:
        for partial, comp in doc_knowledge.partial_references.items():
            partial_map[comp] = partial

    for lk in links:
        if (lk.sentence_number, lk.component_id) in transarc_set:
            safe.append(lk)
        elif lk.source == "partial_inject" and lk.component_name in partial_map:
            partial_review.append(lk)
        else:
            standard_review.append(lk)

    rejected = []

    # Standard links: full convention guide (same as baseline)
    if standard_review:
        items = []
        for i, lk in enumerate(standard_review):
            sent = sent_map.get(lk.sentence_number)
            text = sent.text if sent else "(no text)"
            items.append(f'{i+1}. S{lk.sentence_number}: "{text}"\n   Component: "{lk.component_name}"')

        prompt = f"""Validate trace links between architecture documentation and components.

ARCHITECTURE COMPONENTS: {', '.join(comp_names)}

{CONVENTION_GUIDE}

---

For each sentence-component pair, apply the 3-step reasoning guide.
Decide LINK (keep the trace link) or NO_LINK (reject it).

{chr(10).join(items)}

Return JSON array:
[{{"id": N, "step": "1|2a|2b|3", "verdict": "LINK" or "NO_LINK", "reason": "brief"}}]
JSON only:"""

        raw = llm.query(prompt, timeout=180)
        data = extract_json_array(raw.text if hasattr(raw, 'text') else str(raw))
        verdicts = {}
        if data:
            for item in data:
                vid = item.get("id")
                verdict = item.get("verdict", "LINK").upper().strip()
                step = item.get("step", "3")
                reason = item.get("reason", "")
                if vid is not None:
                    verdicts[vid] = (verdict, step, reason)

        for i, lk in enumerate(standard_review):
            verdict, step, reason = verdicts.get(i + 1, ("LINK", "3", "default"))
            if "NO" in verdict:
                rejected.append((lk, f"step{step}", reason))
            else:
                safe.append(lk)

    # Partial_inject links: softer prompt with partial mapping context
    if partial_review:
        items = []
        for i, lk in enumerate(partial_review):
            sent = sent_map.get(lk.sentence_number)
            text = sent.text if sent else "(no text)"
            short = partial_map.get(lk.component_name, "?")
            items.append(
                f'{i+1}. S{lk.sentence_number}: "{text}"\n'
                f'   Component: "{lk.component_name}" (matched via short name "{short}")'
            )

        prompt = f"""Validate partial-name trace links in a software architecture document.

These links were found because a SHORT NAME for a component appears in the sentence.
The short name was discovered from the document itself (e.g., "Server" is used as
shorthand for "HTML5 Server" throughout this document).

ARCHITECTURE COMPONENTS: {', '.join(comp_names)}

DECISION RULES:
APPROVE when the short name refers to the component in an ARCHITECTURAL context:
- The component participates in the interaction described (even if not the primary subject)
- The sentence describes behavior, communication, or structure involving the component
- "applications running on the X server" → APPROVE for the Server component
- "endpoint to control the X server" → APPROVE for the Server component

REJECT only when:
- The short name appears inside a DOTTED PACKAGE PATH (e.g., "server.utils.config")
- The short name is used in a clearly NON-ARCHITECTURAL sense (hardware spec, room name, etc.)
  Example: "a 32-core server" is about hardware capacity, not the Server component → REJECT
- The word is used as a generic English modifier, not as a component reference
  Example: "server-side rendering" as a general technique, not about the Server component → REJECT

IMPORTANT: The sentence does NOT need to be primarily about the component.
Secondary or incidental architectural mentions of the component are VALID links.

{chr(10).join(items)}

Return JSON array:
[{{"id": N, "verdict": "LINK" or "NO_LINK", "reason": "brief"}}]
JSON only:"""

        raw = llm.query(prompt, timeout=180)
        data = extract_json_array(raw.text if hasattr(raw, 'text') else str(raw))
        verdicts = {}
        if data:
            for item in data:
                vid = item.get("id")
                verdict = item.get("verdict", "LINK").upper().strip()
                reason = item.get("reason", "")
                if vid is not None:
                    verdicts[vid] = (verdict, reason)

        for i, lk in enumerate(partial_review):
            verdict, reason = verdicts.get(i + 1, ("LINK", "default"))
            if "NO" in verdict:
                rejected.append((lk, "partial-soft", reason))
            else:
                safe.append(lk)

    return safe, rejected


def boundary_v33b(links, sent_map, transarc_set, comp_names, doc_knowledge, llm):
    """V33b: multi-word trailing partial immunity — skip filter for partial→multi-word."""
    safe = []
    to_review = []

    for lk in links:
        if (lk.sentence_number, lk.component_id) in transarc_set:
            safe.append(lk)
        elif lk.source == "partial_inject" and ' ' in lk.component_name:
            # Multi-word component partial: immune (the partial mapping is already validated)
            safe.append(lk)
        else:
            to_review.append(lk)

    if not to_review:
        return safe, []

    # Standard convention filter for remaining
    items = []
    for i, lk in enumerate(to_review):
        sent = sent_map.get(lk.sentence_number)
        text = sent.text if sent else "(no text)"
        items.append(f'{i+1}. S{lk.sentence_number}: "{text}"\n   Component: "{lk.component_name}"')

    prompt = f"""Validate trace links between architecture documentation and components.

ARCHITECTURE COMPONENTS: {', '.join(comp_names)}

{CONVENTION_GUIDE}

---

For each sentence-component pair, apply the 3-step reasoning guide.
Decide LINK (keep the trace link) or NO_LINK (reject it).

{chr(10).join(items)}

Return JSON array:
[{{"id": N, "step": "1|2a|2b|3", "verdict": "LINK" or "NO_LINK", "reason": "brief"}}]
JSON only:"""

    raw = llm.query(prompt, timeout=180)
    data = extract_json_array(raw.text if hasattr(raw, 'text') else str(raw))
    verdicts = {}
    if data:
        for item in data:
            vid = item.get("id")
            verdict = item.get("verdict", "LINK").upper().strip()
            step = item.get("step", "3")
            reason = item.get("reason", "")
            if vid is not None:
                verdicts[vid] = (verdict, step, reason)

    rejected = []
    for i, lk in enumerate(to_review):
        verdict, step, reason = verdicts.get(i + 1, ("LINK", "3", "default"))
        if "NO" in verdict:
            rejected.append((lk, f"step{step}", reason))
        else:
            safe.append(lk)

    return safe, rejected


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    selected = sys.argv[1:] if len(sys.argv) > 1 else ["bigbluebutton"]

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

    llm = LLMClient(backend=LLMBackend.CLAUDE)

    variants = {
        "v33":  boundary_v33_baseline,
        "v33a": boundary_v33a,
        "v33b": boundary_v33b,
    }

    all_results = {v: {"tp": 0, "fp": 0, "fn": 0} for v in variants}

    for ds_name in selected:
        paths = DATASETS[ds_name]
        gold = load_gold(str(paths["gold"]))
        components = parse_pcm_repository(str(paths["model"]))
        sentences = DocumentLoader.load_sentences(str(paths["text"]))
        sent_map = {s.number: s for s in sentences}
        name_to_id = {c.name: c.id for c in components}
        id_to_name = {c.id: c.name for c in components}
        comp_names = [c.name for c in components]

        preliminary, transarc_set, doc_knowledge = reconstruct_pre_boundary(
            ds_name, sent_map, name_to_id, components
        )

        print(f"\n{'='*100}")
        print(f"DATASET: {ds_name} ({len(preliminary)} pre-boundary links)")
        print(f"{'='*100}")

        # Count partial_inject in preliminary
        pi_count = sum(1 for l in preliminary if l.source == "partial_inject")
        print(f"  partial_inject links: {pi_count}")
        for lk in sorted(preliminary, key=lambda x: x.sentence_number):
            if lk.source == "partial_inject":
                is_tp = (lk.sentence_number, lk.component_id) in gold
                print(f"    {'TP' if is_tp else 'FP'}: S{lk.sentence_number} -> {lk.component_name}")

        for vname, boundary_fn in variants.items():
            print(f"\n  --- {vname} ---")
            kept, rejected = boundary_fn(
                preliminary, sent_map, transarc_set, comp_names, doc_knowledge, llm
            )

            # Skip judge for speed — just evaluate post-boundary
            pred = {(l.sentence_number, l.component_id) for l in kept}
            m = eval_metrics(pred, gold)

            print(f"  Post-boundary: P={m['P']:.1%} R={m['R']:.1%} F1={m['F1']:.1%} "
                  f"TP={m['tp']} FP={m['fp']} FN={m['fn']}")

            if rejected:
                print(f"  Rejected ({len(rejected)}):")
                for lk, step, reason in rejected:
                    is_tp = (lk.sentence_number, lk.component_id) in gold
                    print(f"    {'TP!' if is_tp else 'FP '}: S{lk.sentence_number} -> "
                          f"{lk.component_name} [{lk.source}] ({step}) — {reason[:80]}")

            for k in ["tp", "fp", "fn"]:
                all_results[vname][k] += m[k]

    if len(selected) > 1:
        print(f"\n{'='*100}")
        print("TOTALS:")
        for vname in variants:
            r = all_results[vname]
            p = r["tp"] / (r["tp"] + r["fp"]) if (r["tp"] + r["fp"]) else 0
            rec = r["tp"] / (r["tp"] + r["fn"]) if (r["tp"] + r["fn"]) else 0
            f1 = 2 * p * rec / (p + rec) if (p + rec) else 0
            print(f"  {vname}: P={p:.1%} R={rec:.1%} F1={f1:.1%} TP={r['tp']} FP={r['fp']} FN={r['fn']}")


if __name__ == "__main__":
    main()
