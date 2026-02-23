#!/usr/bin/env python3
"""Test P8c replacement: annotation-convention-aware judge vs regex boundary filters.

Current P8c (`_apply_boundary_filters`) uses regex rules:
  - `_is_in_package_path`: detects dotted paths (logic.api, e2e.util)
  - `_is_generic_word_usage`: detects generic noun patterns
  - `_is_weak_partial_match`: detects low-confidence partial matches
  - `_abbreviation_guard_for_link`: checks abbreviation validity

The annotation convention findings (v14 reasoning guide) show:
  - Sub-package descriptions = 38% of all FPs (dominant pattern)
  - Dotted paths appear in 72% of non-gold mentions vs 6% of gold (11.9x)
  - The 3-step reasoning guide catches 38/40 FPs with 0 TPs killed

This test: Replace regex boundary filters with a single LLM call that uses
the 3-step reasoning guide from the annotation convention findings.
Test single-phase on TM (where P8c currently catches 5 package_path FPs)
and on all other datasets.

Two approaches tested:
  A) Enhanced Phase 9 judge: add convention reasoning to existing judge
  B) Standalone convention filter: separate LLM call before judge
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

# ── The v14 3-step reasoning guide (abstract, no benchmark leakage) ──

CONVENTION_GUIDE = """## Reasoning Guide for Trace Link Validation

A trace link connects a documentation sentence to an architectural component when
the sentence is RELEVANT to that component's role in the system architecture.
The bar for LINK is low — any architectural relevance counts. When in doubt, default to LINK.

### STEP 1 — Sub-package / internal structure description?

The most common reason for NO_LINK: the sentence describes what is INSIDE a component
(its sub-packages, internal classes, internal structure) rather than the component's
architectural role.

Recognize these patterns — all are NO_LINK for component X:
- "X.config loads environment variables" — dotted sub-package description
- "Package overview contains X.handlers, X.mappers, X.converters" — internal package listing
- "Classes in the X.internal package are not visible outside this module" — even with
  architectural language, if the subject is a sub-package → NO_LINK
- Bare name listed alongside dotted paths: "X, Y.adapters, Y.transformers follow a pipeline
  design" — when a bare name appears as a peer of qualified names, treat ALL as sub-package
  references → NO_LINK for both X and Y

EXCEPTION: If the sentence also explicitly names the target component AS A PROPER NOUN —
typically with the word "component" (e.g., "for the X component", "from the Y component") —
the explicit component reference overrides → LINK. But if the component name only appears
in lowercase or as a generic descriptor of an activity, the exception does NOT apply.

Cross-reference rule: A sub-package sentence that mentions a DIFFERENT component in an
architectural role is LINK for that other component.

### STEP 2 — Component name confused with a different entity?

**2a. Technology / methodology confusion:**
NO_LINK when the sentence:
- Describes what a technology IS — its definition, capabilities, or qualities
- Lists technologies as stack dependencies
- Names a COMPOUND ENTITY whose full name CONTAINS the component name
  ("X Protocol specification" → NO_LINK for "X")
- Uses the component name as part of a METHODOLOGY or PRACTICE name
  ("X testing in CI" → NO_LINK for "X")

LINK when users or other components INTERACT with or connect THROUGH the technology.

**2b. Generic word collision:**
NO_LINK — narrow, non-architectural sense:
- Process/activity modifier + word: "cascade X", "retry X", "validation X"
- Hardware/deployment context: "a dedicated hardware node"
- Possessive/personal attribute: "her settings", "their preferences"

LINK — system-level architectural sense:
- System name + word: "the [SystemName] gateway"
- Architectural role: "the orchestrator routes jobs to the gateway"

### STEP 3 — Default: LINK.

If neither Step 1 nor Step 2 applies, classify as LINK.

### Priority reminder:
Be AGGRESSIVE with NO_LINK on sub-package descriptions (Step 1). For Step 2, only
classify NO_LINK when confident. Always check the proper-noun exception before
finalizing NO_LINK."""


def _extract_json_array(text):
    """Extract a JSON array from LLM text that may have markdown fences."""
    import json
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


def original_p8c(links, sent_map, transarc_set):
    """Simulate original _apply_boundary_filters from V26a."""
    # Import the class to use its regex methods
    from llm_sad_sam.linkers.experimental.agent_linker_v26a import AgentLinkerV26a

    # Create minimal instance for method access
    class Tester(AgentLinkerV26a):
        def __init__(self):
            self.doc_knowledge = None
            self.model_knowledge = None
            self.GENERIC_COMPONENT_WORDS = set()
            self.GENERIC_PARTIALS = set()

    tester = Tester()

    kept = []
    rejected = []
    for lk in links:
        sent = sent_map.get(lk.sentence_number)
        if not sent:
            kept.append(lk)
            continue
        is_ta = (lk.sentence_number, lk.component_id) in transarc_set
        if is_ta:
            kept.append(lk)
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
            rejected.append((lk, reason))
        else:
            kept.append(lk)
    return kept, rejected


def llm_convention_filter(links, sent_map, transarc_set, llm, components):
    """LLM-based convention filter using 3-step reasoning guide.

    Replaces regex boundary filters with a single LLM call per batch.
    Only filters non-TransArc links (same as original P8c).
    """
    comp_names = [c.name for c in components]

    # Separate TransArc-immune links
    safe = []
    to_review = []
    for lk in links:
        is_ta = (lk.sentence_number, lk.component_id) in transarc_set
        if is_ta:
            safe.append(lk)
        else:
            to_review.append(lk)

    if not to_review:
        return safe, []

    # Batch LLM review with convention guide
    items = []
    for i, lk in enumerate(to_review):
        sent = sent_map.get(lk.sentence_number)
        text = sent.text if sent else "(no text)"
        items.append(
            f'{i+1}. S{lk.sentence_number}: "{text}"\n'
            f'   Component: "{lk.component_name}" (source: {lk.source})'
        )

    # Process in batches of 20
    batch_size = 20
    all_verdicts = {}

    for batch_start in range(0, len(items), batch_size):
        batch_items = items[batch_start:batch_start + batch_size]
        batch_indices = list(range(batch_start, min(batch_start + batch_size, len(items))))

        prompt = f"""You are validating trace links between architecture documentation and components.

ARCHITECTURE COMPONENTS: {', '.join(comp_names)}

{CONVENTION_GUIDE}

---

For each item below, apply the 3-step reasoning guide and decide LINK or NO_LINK.

{chr(10).join(batch_items)}

Return JSON array:
[{{"id": N, "step": "1|2a|2b|3", "verdict": "LINK" or "NO_LINK", "reason": "brief reason"}}]
JSON only:"""

        raw = llm.query(prompt, timeout=180)
        data = _extract_json_array(raw.text)
        if data:
            for item in data:
                vid = item.get("id")
                verdict = item.get("verdict", "LINK").upper().strip()
                step = item.get("step", "3")
                reason = item.get("reason", "")
                if vid is not None:
                    all_verdicts[vid] = (verdict, step, reason)

    kept = list(safe)
    rejected = []
    for i, lk in enumerate(to_review):
        verdict, step, reason = all_verdicts.get(i + 1, ("LINK", "3", "default"))
        if "NO" in verdict:
            rejected.append((lk, f"convention_step{step}"))
            print(f"    LLM REJECT [step {step}]: S{lk.sentence_number}→{lk.component_name} "
                  f"({lk.source}) — {reason}")
        else:
            kept.append(lk)

    return kept, rejected


def main():
    os.environ.setdefault("CLAUDE_MODEL", "sonnet")

    print("=" * 90)
    print("  P8c Test: Original (regex) vs LLM convention-aware boundary filter")
    print("=" * 90)

    for ds in DATASETS:
        pre_judge = load_checkpoint(ds, "pre_judge")
        data4 = load_checkpoint(ds, "phase4")
        if not all([pre_judge, data4]):
            continue

        components, sentences, sent_map, name_to_id, id_to_name = load_dataset(ds)
        gold = load_gold(ds)
        transarc_set = data4["transarc_set"]

        # The pre_judge checkpoint has links BEFORE boundary filters and judge.
        # We need the links AFTER Phase 8 combination but BEFORE P8c filtering.
        # The pre_judge checkpoint already has boundary filters applied.
        # We need to reconstruct pre-P8c state.
        # Actually, looking at V30c code: pre_judge is saved AFTER P8c but BEFORE Phase 9.
        # So pre_judge["preliminary"] already has P8c applied.
        # We need to use a different approach: load phase7 + reconstruct.

        data6 = load_checkpoint(ds, "phase6")
        data7 = load_checkpoint(ds, "phase7")
        data3 = load_checkpoint(ds, "phase3")
        if not all([data6, data7, data3]):
            continue

        # Reconstruct pre-P8c link set (same as V30c link() method)
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

        # Combine + deduplicate (same as V30c)
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

        # Parent-overlap guard (same as V30c)
        # Skip for simplicity — it only fires on TS (-1 link)

        print(f"\n{'─' * 70}")
        print(f"  {ds}: {len(preliminary)} pre-P8c links")
        print(f"{'─' * 70}")

        # ── Original P8c ──
        orig_kept, orig_rejected = original_p8c(preliminary, sent_map, transarc_set)
        orig_rej_tp = sum(1 for lk, r in orig_rejected if (lk.sentence_number, lk.component_id) in gold)
        orig_rej_fp = len(orig_rejected) - orig_rej_tp

        print(f"\n  ORIGINAL P8c: rejected {len(orig_rejected)} "
              f"({orig_rej_tp} TP killed, {orig_rej_fp} FP caught)")
        for lk, reason in orig_rejected:
            is_tp = (lk.sentence_number, lk.component_id) in gold
            label = "TP KILLED!" if is_tp else "FP caught"
            print(f"    [{reason}] [{label}] S{lk.sentence_number} → {lk.component_name} ({lk.source})")

        # ── LLM convention filter ──
        print(f"\n  LLM Convention Filter:")
        llm = LLMClient()
        llm_kept, llm_rejected = llm_convention_filter(
            preliminary, sent_map, transarc_set, llm, components
        )
        llm_rej_tp = sum(1 for lk, r in llm_rejected if (lk.sentence_number, lk.component_id) in gold)
        llm_rej_fp = len(llm_rejected) - llm_rej_tp

        print(f"\n  LLM Convention: rejected {len(llm_rejected)} "
              f"({llm_rej_tp} TP killed, {llm_rej_fp} FP caught)")

        # ── Comparison ──
        print(f"\n  COMPARISON:")
        print(f"    Original P8c: -{orig_rej_fp} FP, -{orig_rej_tp} TP")
        print(f"    LLM Conv:     -{llm_rej_fp} FP, -{llm_rej_tp} TP")

        # Check: does LLM catch what regex catches + more?
        orig_rej_keys = {(lk.sentence_number, lk.component_id) for lk, _ in orig_rejected}
        llm_rej_keys = {(lk.sentence_number, lk.component_id) for lk, _ in llm_rejected}

        only_regex = orig_rej_keys - llm_rej_keys
        only_llm = llm_rej_keys - orig_rej_keys
        both = orig_rej_keys & llm_rej_keys

        if only_regex:
            print(f"\n    Caught by REGEX only ({len(only_regex)}):")
            for key in sorted(only_regex):
                lk = next(l for l, _ in orig_rejected if (l.sentence_number, l.component_id) == key)
                is_tp = key in gold
                print(f"      S{key[0]}→{lk.component_name} [{'TP' if is_tp else 'FP'}]")

        if only_llm:
            print(f"\n    Caught by LLM only ({len(only_llm)}):")
            for key in sorted(only_llm):
                lk = next(l for l, _ in llm_rejected if (l.sentence_number, l.component_id) == key)
                is_tp = key in gold
                print(f"      S{key[0]}→{lk.component_name} [{'TP' if is_tp else 'FP'}]")

        if both:
            print(f"\n    Caught by BOTH ({len(both)})")

        # Net impact
        orig_fp_after = sum(1 for lk in orig_kept if (lk.sentence_number, lk.component_id) not in gold
                           and (lk.sentence_number, lk.component_id) not in transarc_set)
        llm_fp_after = sum(1 for lk in llm_kept if (lk.sentence_number, lk.component_id) not in gold
                          and (lk.sentence_number, lk.component_id) not in transarc_set)

        print(f"\n    Remaining non-TransArc FPs: regex={orig_fp_after}, LLM={llm_fp_after}")


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


if __name__ == "__main__":
    main()
