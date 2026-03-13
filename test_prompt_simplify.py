#!/usr/bin/env python3
"""Step-level tests for prompt simplification variants.

Loads V33 phase caches, replays Phase 8c (convention filter) and Phase 9 (judge)
with simplified prompt variants, and compares against gold standard.

Variants tested:
  A) SLIM_GUIDE:   CONVENTION_GUIDE with redundancy removed (~45 lines vs 78)
  B) REVERT_GENERIC: Restore V32's _is_generic_mention code heuristic
  C) SHORT_RULE4:  Compress Judge Rule 4 CRITICAL EXCEPTION (7→2 lines)
  D) ALL:          A + B + C combined

For each variant, we test:
  - Phase 8c alone (convention filter with different guides)
  - Phase 9 alone (judge with different prompts)
  - Phase 8c + Phase 9 combined (end-to-end from pre-8c links)

Usage:
    python test_prompt_simplify.py                     # all datasets, all variants
    python test_prompt_simplify.py --datasets teammates # single dataset
    python test_prompt_simplify.py --phase 8c          # only test Phase 8c
    python test_prompt_simplify.py --phase 9            # only test Phase 9
"""

import argparse
import csv
import glob
import json
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.core.data_types import SadSamLink
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient

os.environ.setdefault("CLAUDE_MODEL", "sonnet")

# ═══════════════════════════════════════════════════════════════════════════
# Dataset paths
# ═══════════════════════════════════════════════════════════════════════════

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
CACHE_DIR = Path("./results/phase_cache/v33")

SOURCE_PRIORITY = {"transarc": 1, "entity": 2, "validated": 2, "coreference": 3,
                   "implicit": 4, "partial_inject": 5}


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Variants
# ═══════════════════════════════════════════════════════════════════════════

# --- V33 ORIGINAL (baseline) ---
GUIDE_ORIGINAL = """### STEP 1 — Hierarchical name reference (not about the component itself)?

The most common reason for NO_LINK: the sentence mentions the component name only
as part of a HIERARCHICAL/QUALIFIED NAME (dotted path, namespace, module path) that
refers to a nested sub-unit, not the component's own architectural role.

Software documentation commonly uses hierarchical naming (e.g., "X.utils", "X/handlers",
"X::impl") to refer to parts inside a component. The component name appears only as
a prefix, not as the subject.

Recognize these patterns — all are NO_LINK for component X:
- "X.utils provides helper functions" — dotted sub-unit reference
- "X.handlers, X.mappers, X.adapters follow a pipeline" — listing sub-packages of X
- "Classes in the X.impl package are not exported" — even with
  architectural language, if the subject is X's sub-unit → NO_LINK
- Bare name mixed with qualified paths: "X, Y.adapters, Y.transformers follow
  a pipeline design" — treat ALL as hierarchical references → NO_LINK

KEY DISTINCTION: Sentences that describe what X DOES or HOW X INTERACTS with other
components are LINK, even if they mention implementation details (e.g., "X uses Y
technology for Z" → LINK for X, because it describes X's behavior).

EXCEPTION: If the sentence also explicitly names the target component AS A PROPER
NOUN with the word "component" (e.g., "for the X component") → LINK.

Cross-reference rule: A sub-unit sentence mentioning a DIFFERENT component
in an architectural role is LINK for that other component.

### STEP 2 — Component name confused with a different entity?

**2a. Technology / methodology confusion:**

CRITICAL RULE: If a component IS NAMED AFTER a technology (e.g., the architecture
has a component called "Kafka Broker" or "Nginx Proxy" or "Zookeeper"), then ANY sentence
describing that technology's capabilities, role, or behavior IS about the component → LINK.
This rule applies because architecture components are often named after the technology they wrap.

NO_LINK ONLY when:
- The sentence describes a technology that is NOT one of our components
- The name appears in a compound entity unrelated to the component
  ("X Protocol specification" → NO_LINK for "X" if X is not about that protocol)
- Uses the name as part of a METHODOLOGY ("X testing in CI" → NO_LINK for "X")

LINK when:
- The technology IS one of our architecture components (always LINK)
- Components INTERACT with or connect THROUGH the technology

**2b. Generic word collision:**
NO_LINK — narrow, non-architectural sense:
- Process/activity modifier: "throttle X", "batch X", "polling X"
- Hardware/deployment: "a physical rack-mounted appliance", "multi-socket machine"
- Possessive/personal: "her bookmarks", "their account"

LINK — system-level architectural sense:
- System name + word: "the [System] gateway"
- Architectural role: "the orchestrator routes jobs to the gateway"

### STEP 3 — Default: LINK.
If neither Step 1 nor Step 2 applies → LINK.

### IMPORTANT GUARDRAILS:
- Multi-word component names (e.g., "Kafka Broker", "Nginx Proxy") are NEVER generic words → LINK
- CamelCase identifiers are NEVER generic words → LINK
- Sentences describing how components interact, connect, or communicate → LINK for ALL components involved (not just the grammatical subject). "X connects to Y" is LINK for both X and Y.
- Sentences about what a component does, provides, or handles → LINK
- A component does NOT need to be the grammatical subject to be relevant. If a sentence says "X sends data to Y", both X and Y get LINK.
- Only use NO_LINK when you are CONFIDENT the name is NOT used as a component reference

### Priority:
Be AGGRESSIVE with NO_LINK on sub-package descriptions (Step 1).
For Step 2, only NO_LINK when confident. Default to LINK."""


# --- VARIANT A: SLIM GUIDE (redundancy removed, ~45 lines vs 78) ---
GUIDE_SLIM = """### STEP 1 — Hierarchical name reference?

NO_LINK when the component name appears only as part of a HIERARCHICAL/QUALIFIED NAME
(dotted path, namespace, module path) referring to a sub-unit, not the component itself.

Examples — all NO_LINK for component X:
- "X.utils provides helper functions" — dotted sub-unit
- "X.handlers, X.mappers, X.adapters follow a pipeline" — listing sub-packages
- "Classes in the X.impl package are not exported" — sub-unit even with architectural language
- Bare name mixed with qualified paths: "X, Y.adapters, Y.transformers" — all hierarchical

KEY DISTINCTION: Sentences describing what X DOES or HOW X INTERACTS → LINK,
even if they mention implementation details.

### STEP 2 — Name confused with a different entity?

**2a. Technology confusion:**
If a component IS NAMED AFTER a technology, sentences about that technology → LINK.
NO_LINK only when the technology is NOT one of our components, or the name appears
in a compound entity or methodology unrelated to the component.

**2b. Generic word collision:**
NO_LINK only for narrow non-architectural uses: process modifiers ("throttle X"),
hardware/deployment context, possessive/personal attributes.
LINK for system-level architectural references.

### STEP 3 — Default: LINK.
If neither Step 1 nor Step 2 applies → LINK.
Only use NO_LINK when CONFIDENT. Multi-word and CamelCase names are NEVER generic → always LINK.
A component does NOT need to be the grammatical subject to get LINK — "X sends data to Y" is LINK for both X and Y."""


# --- VARIANT A2: SLIM_V2 (keeps interaction guardrail from ORIGINAL) ---
GUIDE_SLIM_V2 = """### STEP 1 — Hierarchical name reference?

NO_LINK when the component name appears only as part of a HIERARCHICAL/QUALIFIED NAME
(dotted path, namespace, module path) referring to a sub-unit, not the component itself.

Examples — all NO_LINK for component X:
- "X.utils provides helper functions" — dotted sub-unit
- "X.handlers, X.mappers, X.adapters follow a pipeline" — listing sub-packages
- "Classes in the X.impl package are not exported" — sub-unit even with architectural language
- Bare name mixed with qualified paths: "X, Y.adapters, Y.transformers" — all hierarchical

KEY DISTINCTION: Sentences describing what X DOES or HOW X INTERACTS → LINK,
even if they mention implementation details.

### STEP 2 — Name confused with a different entity?

**2a. Technology confusion:**
If a component IS NAMED AFTER a technology, sentences about that technology → LINK.
NO_LINK only when the technology is NOT one of our components, or the name appears
in a compound entity or methodology unrelated to the component.

**2b. Generic word collision:**
NO_LINK only for narrow non-architectural uses: process modifiers ("throttle X"),
hardware/deployment context, possessive/personal attributes.
LINK for system-level architectural references.

### STEP 3 — Default: LINK.
If neither Step 1 nor Step 2 applies → LINK.

### GUARDRAILS:
- Multi-word component names (e.g., "Kafka Broker", "Nginx Proxy") are NEVER generic → LINK
- CamelCase identifiers are NEVER generic → LINK
- Sentences describing component interactions → LINK for ALL involved components
- Only use NO_LINK when CONFIDENT the name is NOT a component reference"""


# --- V33 ORIGINAL Judge Rule 4 ---
JUDGE_RULE4_ORIGINAL = """4. COMPONENT-SPECIFIC USAGE
   The reference is to the component as a named entity within the system architecture,
   not to a generic concept, pattern, or technology that happens to share a name.

   CRITICAL EXCEPTION: If the system has a component literally named "X" — even when
   X is a common word like "Scheduler", "Router", "Optimizer", "Parser", "Dispatcher",
   "Handler" — and the sentence describes what X does, how X interacts with other parts
   of the system, or X's responsibilities, then this IS component-specific usage.
   The existence of a named component means architectural references to that concept
   ARE about the component. Only reject Rule 4 when "X" is clearly used in a different
   domain (e.g., "scheduler" meaning "appointment planner" in a business context, not
   the software Scheduler component)."""

# --- VARIANT C: SHORT Rule 4 (compressed to 2 lines) ---
JUDGE_RULE4_SHORT = """4. COMPONENT-SPECIFIC USAGE
   The reference is to the component as a named entity within the system architecture,
   not to a generic concept or technology that happens to share a name.
   EXCEPTION: If the system literally has a component named "X", architectural references
   to X ARE component-specific. Only reject when "X" is clearly used in a different domain."""


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

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
            for row in csv.DictReader(fh):
                cid = row.get("modelElementID", "")
                sid = row.get("sentence", "")
                if sid and cid:
                    gold.add((int(sid), cid))
    return gold


def eval_metrics(predicted_pairs, gold):
    tp = len(predicted_pairs & gold)
    fp = len(predicted_pairs - gold)
    fn = len(gold - predicted_pairs)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"tp": tp, "fp": fp, "fn": fn, "P": p, "R": r, "F1": f1}


def links_to_pairs(links):
    return {(l.sentence_number, l.component_id) for l in links}


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


def has_clean_mention(name, text):
    """Check if name appears as standalone word (not in dotted path)."""
    pattern = rf'\b{re.escape(name)}\b'
    for m in re.finditer(pattern, text, re.IGNORECASE):
        s, e = m.start(), m.end()
        if s > 0 and text[s-1] == '.':
            continue
        if e < len(text) and text[e] == '.' and e + 1 < len(text) and text[e+1].isalpha():
            continue
        return True
    return False


def has_standalone_mention(comp_name, text):
    """V33's standalone mention check (case-sensitive for single words)."""
    is_single = ' ' not in comp_name
    if is_single:
        cap_name = comp_name[0].upper() + comp_name[1:]
        pattern = rf'\b{re.escape(cap_name)}\b'
        flags = 0
    else:
        pattern = rf'\b{re.escape(comp_name)}\b'
        flags = re.IGNORECASE
    for m in re.finditer(pattern, text, flags):
        s, e = m.start(), m.end()
        if s > 0 and text[s-1] == '.':
            continue
        if e < len(text) and text[e] == '.' and e + 1 < len(text) and text[e+1].isalpha():
            continue
        if s > 0 and text[s-1] == '-':
            continue
        if e < len(text) and text[e] == '-' and '-' not in comp_name:
            continue
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# V32 _is_generic_mention heuristic (Variant B restores this)
# ═══════════════════════════════════════════════════════════════════════════

def is_generic_mention_v32(comp_name, sentence_text):
    """V32's code heuristic: single-word capitalized names appearing only lowercase."""
    if ' ' in comp_name or '-' in comp_name:
        return False
    if re.search(r'[a-z][A-Z]', comp_name):
        return False
    if comp_name.isupper():
        return False
    if comp_name[0].islower():
        return False
    if has_standalone_mention(comp_name, sentence_text):
        return False
    word_lower = comp_name.lower()
    if re.search(rf'\b{re.escape(word_lower)}\b', sentence_text):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Reconstruct pre-Phase-8c links from caches
# ═══════════════════════════════════════════════════════════════════════════

def reconstruct_pre8c_links(dataset, sentences, name_to_id):
    """Reconstruct the combined link set before Phase 8c from cached phases."""
    data4 = load_checkpoint(dataset, "phase4")
    data6 = load_checkpoint(dataset, "phase6")
    data7 = load_checkpoint(dataset, "phase7")
    data3 = load_checkpoint(dataset, "phase3")
    if not all([data4, data6, data7]):
        return None, None

    transarc_links = data4.get("transarc_links", [])
    transarc_set = data4.get("transarc_set", set())
    validated = data6.get("validated", [])
    coref_links = data7.get("coref_links", [])
    doc_knowledge = data3.get("doc_knowledge") if data3 else None

    # Phase 8b: partial injection (deterministic, no LLM)
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
                if has_clean_mention(partial, sent.text):
                    partial_links.append(SadSamLink(sent.number, comp_id, comp_name, 0.8, "partial_inject"))
                    existing.add(key)

    # Combine + deduplicate
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

    return list(link_map.values()), transarc_set


# ═══════════════════════════════════════════════════════════════════════════
# Phase 8c: Convention filter with swappable guide
# ═══════════════════════════════════════════════════════════════════════════

def run_convention_filter(links, sent_map, transarc_set, llm, components, guide_text):
    """Run Phase 8c convention filter with the given CONVENTION_GUIDE text."""
    comp_names = [c.name for c in components]

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

    items = []
    for i, lk in enumerate(to_review):
        sent = sent_map.get(lk.sentence_number)
        text = sent.text if sent else "(no text)"
        items.append(
            f'{i+1}. S{lk.sentence_number}: "{text}"\n'
            f'   Component: "{lk.component_name}"'
        )

    batch_size = 25
    all_verdicts = {}

    for batch_start in range(0, len(items), batch_size):
        batch_items = items[batch_start:batch_start + batch_size]

        prompt = f"""Validate trace links between architecture documentation and components.

ARCHITECTURE COMPONENTS: {', '.join(comp_names)}

{guide_text}

---

For each sentence-component pair, apply the 3-step reasoning guide.
Decide LINK (keep the trace link) or NO_LINK (reject it).

{chr(10).join(batch_items)}

Return JSON array:
[{{"id": N, "step": "1|2a|2b|3", "verdict": "LINK" or "NO_LINK", "reason": "brief"}}]
JSON only:"""

        raw = llm.query(prompt, timeout=180)
        data = extract_json_array(raw.text if hasattr(raw, 'text') else str(raw))
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
            rejected.append((lk, f"step{step}: {reason}"))
        else:
            kept.append(lk)

    return kept, rejected


# ═══════════════════════════════════════════════════════════════════════════
# Phase 9: Judge with swappable Rule 4
# ═══════════════════════════════════════════════════════════════════════════

def build_judge_prompt(comp_names, cases, rule4_text):
    """Build the 4-rule judge prompt with swappable Rule 4."""
    return f"""JUDGE: Validate trace links between documentation and software architecture components.

APPROVAL CRITERIA:
A link S→C is valid when the sentence satisfies all four conditions:

1. EXPLICIT REFERENCE
   The component name (or a direct reference to it) appears in the sentence as a clear
   entity being discussed.

2. SYSTEM-LEVEL PERSPECTIVE
   The sentence describes the component's role, responsibilities, interfaces, or
   interactions within the overall system architecture.

3. PRIMARY FOCUS
   The component is the main subject of what the sentence conveys, not a secondary
   or incidental mention.

{rule4_text}

COMPONENTS: {', '.join(comp_names)}

LINKS:
{chr(10).join(cases)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief explanation"}}]}}
JSON only:"""


def run_judge_review(links, sentences, components, sent_map, transarc_set,
                     llm, doc_knowledge, rule4_text):
    """Run Phase 9 judge with the given Rule 4 text."""
    if len(links) < 5:
        return links

    comp_names = [c.name for c in components]

    # Triage (same as V33)
    safe, nomatch_links, ta_review = [], [], []
    syn_safe_count = 0
    for l in links:
        is_ta = (l.sentence_number, l.component_id) in transarc_set
        sent = sent_map.get(l.sentence_number)

        # Synonym-safe bypass
        if sent and _has_alias_mention(l.component_name, sent.text, doc_knowledge):
            safe.append(l)
            syn_safe_count += 1
            continue

        if is_ta:
            if _is_ambiguous_name(l.component_name):
                ta_review.append(l)
            else:
                safe.append(l)
            continue

        if not sent:
            nomatch_links.append(l)
            continue

        if has_standalone_mention(l.component_name, sent.text):
            safe.append(l)
        else:
            nomatch_links.append(l)

    print(f"    Triage: {len(safe)} safe ({syn_safe_count} syn-safe), "
          f"{len(ta_review)} ta-review, {len(nomatch_links)} no-match")

    # Deliberate TransArc (same as V33 — uses advocate-prosecutor, not affected by Rule 4)
    ta_approved = _deliberate_transarc_simple(ta_review, comp_names, sent_map, llm)

    # 4-rule judge for no-match links (THIS is what Rule 4 affects)
    nomatch_approved = _judge_nomatch(nomatch_links, comp_names, sent_map, llm, rule4_text)

    return safe + ta_approved + nomatch_approved


def _has_alias_mention(comp_name, sentence_text, doc_knowledge):
    if not doc_knowledge:
        return False
    text_lower = sentence_text.lower()
    for syn, target in doc_knowledge.synonyms.items():
        if target == comp_name:
            if re.search(rf'\b{re.escape(syn.lower())}\b', text_lower):
                return True
    for partial, target in doc_knowledge.partial_references.items():
        if target == comp_name:
            if re.search(rf'\b{re.escape(partial.lower())}\b', text_lower):
                return True
    return False


def _is_ambiguous_name(comp_name):
    if ' ' in comp_name or '-' in comp_name:
        return False
    if re.search(r'[a-z][A-Z]', comp_name):
        return False
    if comp_name.isupper():
        return False
    return True


def _deliberate_transarc_simple(links, comp_names, sent_map, llm):
    """Simplified advocate-prosecutor for TransArc links. Same prompts as V33."""
    if not links:
        return []
    approved = []
    comp_names_str = ', '.join(comp_names)
    for l in links:
        sent = sent_map.get(l.sentence_number)
        if not sent:
            approved.append(l)
            continue
        # Two independent passes with union voting
        verdicts = []
        for _ in range(2):
            adv_prompt = f"""You are the ADVOCATE for linking sentence S{l.sentence_number} to component "{l.component_name}".
The system has a component literally named "{l.component_name}". Your job is to defend valid links.
SENTENCE: {sent.text}
ALL COMPONENTS: {comp_names_str}
Find the STRONGEST evidence this sentence discusses "{l.component_name}" at the architectural level.
Return JSON: {{"argument": "your argument", "verdict": "APPROVE" or "REJECT"}}
JSON only:"""
            pros_prompt = f"""You are the PROSECUTOR arguing AGAINST linking sentence S{l.sentence_number} to component "{l.component_name}".
Only argue REJECT when there is CLEAR evidence the match is spurious.
SENTENCE: {sent.text}
ALL COMPONENTS: {comp_names_str}
Find CLEAR evidence this is a SPURIOUS match. Only reject for: modifier/adjective usage, incidental mention, technology not matching component, dotted path.
Return JSON: {{"argument": "your argument", "verdict": "APPROVE" or "REJECT"}}
JSON only:"""
            adv_data = llm.extract_json(llm.query(adv_prompt, timeout=60))
            pros_data = llm.extract_json(llm.query(pros_prompt, timeout=60))
            adv_arg = adv_data.get("argument", "") if adv_data else ""
            pros_arg = pros_data.get("argument", "") if pros_data else ""

            jury_prompt = f"""JURY: Decide if sentence S{l.sentence_number} should be linked to component "{l.component_name}".
The system has a component literally named "{l.component_name}". REJECT only with clear evidence — when in doubt, APPROVE.
SENTENCE: {sent.text}
ADVOCATE: {adv_arg}
PROSECUTOR: {pros_arg}
Return JSON: {{"verdict": "APPROVE" or "REJECT", "reason": "brief"}}
JSON only:"""
            jury_data = llm.extract_json(llm.query(jury_prompt, timeout=60))
            verdicts.append(jury_data.get("verdict", "APPROVE").upper() == "APPROVE" if jury_data else True)

        if verdicts[0] or verdicts[1]:
            approved.append(l)
        else:
            print(f"      Deliberation reject: S{l.sentence_number} -> {l.component_name}")
    return approved


def _judge_nomatch(nomatch_links, comp_names, sent_map, llm, rule4_text):
    """4-rule judge with union voting for no-match links."""
    if not nomatch_links:
        return []

    cases = []
    for i, l in enumerate(nomatch_links):
        sent = sent_map.get(l.sentence_number)
        text = sent.text if sent else "(no text)"
        cases.append(f'{i+1}. S{l.sentence_number} → "{l.component_name}": "{text}"')

    prompt = build_judge_prompt(comp_names, cases, rule4_text)
    n = min(30, len(nomatch_links))

    # Union voting
    data1 = llm.extract_json(llm.query(prompt, timeout=180))
    data2 = llm.extract_json(llm.query(prompt, timeout=180))

    rej1 = _parse_rejections(data1, n)
    rej2 = _parse_rejections(data2, n)
    rejected = rej1 & rej2

    result = []
    for i in range(n):
        if i not in rejected:
            result.append(nomatch_links[i])
        else:
            print(f"      4-rule reject: S{nomatch_links[i].sentence_number} -> {nomatch_links[i].component_name}")
    result.extend(nomatch_links[n:])
    return result


def _parse_rejections(data, n):
    rejected = set()
    if data:
        for j in data.get("judgments", []):
            idx = j.get("case", 0) - 1
            if 0 <= idx < n and not j.get("approve", False):
                rejected.add(idx)
    return rejected


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6 validation with Variant B (generic mention heuristic)
# ═══════════════════════════════════════════════════════════════════════════

def apply_generic_mention_filter(validated_links, sent_map):
    """Variant B: re-apply V32's _is_generic_mention heuristic post-validation.

    V33 disabled this heuristic and moved it to the LLM prompt. This filter
    simulates restoring it: any validated link where the component appears only
    in lowercase (generic mention) is downgraded — it would have gone through
    a stricter validation path in V32.

    Since we can't re-run Phase 6 LLM calls, we approximate by removing links
    that the V32 heuristic would have flagged AND that don't have a capitalized
    standalone mention in the sentence.
    """
    kept = []
    removed = []
    for lk in validated_links:
        sent = sent_map.get(lk.sentence_number)
        if not sent:
            kept.append(lk)
            continue
        if is_generic_mention_v32(lk.component_name, sent.text):
            removed.append(lk)
        else:
            kept.append(lk)
    return kept, removed


# ═══════════════════════════════════════════════════════════════════════════
# Main test runner
# ═══════════════════════════════════════════════════════════════════════════

def test_phase8c(datasets, llm):
    """Test Phase 8c convention filter with original vs slim guide."""
    print("\n" + "=" * 80)
    print("  PHASE 8c TEST: Convention Filter Prompt Variants")
    print("=" * 80)

    results = {}
    for ds in datasets:
        components, sentences, sent_map, name_to_id = load_dataset(ds)
        gold = load_gold(ds)

        preliminary, transarc_set = reconstruct_pre8c_links(ds, sentences, name_to_id)
        if preliminary is None:
            print(f"\n  {ds}: SKIP (missing cache)")
            continue

        pre8c_pairs = links_to_pairs(preliminary)
        pre8c_m = eval_metrics(pre8c_pairs, gold)

        print(f"\n{'─' * 70}")
        print(f"  {ds}: {len(preliminary)} pre-8c links "
              f"(P={pre8c_m['P']:.1%} R={pre8c_m['R']:.1%} F1={pre8c_m['F1']:.1%})")
        print(f"{'─' * 70}")

        ds_results = {}
        for guide_name, guide_text in [("ORIGINAL", GUIDE_ORIGINAL), ("SLIM", GUIDE_SLIM),
                                        ("SLIM_V2", GUIDE_SLIM_V2)]:
            t0 = time.time()
            kept, rejected = run_convention_filter(
                preliminary, sent_map, transarc_set, llm, components, guide_text
            )
            elapsed = time.time() - t0

            kept_pairs = links_to_pairs(kept)
            m = eval_metrics(kept_pairs, gold)
            rej_tp = sum(1 for lk, _ in rejected if (lk.sentence_number, lk.component_id) in gold)
            rej_fp = len(rejected) - rej_tp

            print(f"\n  [{guide_name}] {elapsed:.0f}s: {len(kept)} kept, {len(rejected)} rejected "
                  f"({rej_fp} FP caught, {rej_tp} TP killed)")
            print(f"    P={m['P']:.1%} R={m['R']:.1%} F1={m['F1']:.1%} "
                  f"(FP={m['fp']}, FN={m['fn']})")

            for lk, reason in rejected:
                is_tp = (lk.sentence_number, lk.component_id) in gold
                label = "TP!" if is_tp else "FP"
                print(f"    [{label}] S{lk.sentence_number}→{lk.component_name} ({lk.source}): {reason}")

            ds_results[guide_name] = m

        results[ds] = ds_results

    return results


def test_phase9(datasets, llm):
    """Test Phase 9 judge Rule 4 variants — 4-rule judge only (skip deliberation).

    Rule 4 only affects the 4-rule judge for non-TA, non-safe links.
    Deliberation uses advocate-prosecutor prompts (unaffected by Rule 4).
    To save LLM cost, we only test the 4-rule judge portion.
    """
    print("\n" + "=" * 80)
    print("  PHASE 9 TEST: 4-Rule Judge Only (Rule 4 Variants)")
    print("  (Deliberation skipped — it uses different prompts)")
    print("=" * 80)

    results = {}
    for ds in datasets:
        components, sentences, sent_map, name_to_id = load_dataset(ds)
        gold = load_gold(ds)

        pre_judge = load_checkpoint(ds, "pre_judge")
        data3 = load_checkpoint(ds, "phase3")
        if not pre_judge:
            print(f"\n  {ds}: SKIP (missing pre_judge cache)")
            continue

        preliminary = pre_judge["preliminary"]
        transarc_set = pre_judge["transarc_set"]
        doc_knowledge = data3["doc_knowledge"] if data3 else None
        comp_names = [c.name for c in components]

        # Triage (same as V33) to find the nomatch links that go through 4-rule judge
        safe, nomatch_links, ta_review = [], [], []
        for l in preliminary:
            is_ta = (l.sentence_number, l.component_id) in transarc_set
            sent = sent_map.get(l.sentence_number)
            if sent and _has_alias_mention(l.component_name, sent.text, doc_knowledge):
                safe.append(l)
                continue
            if is_ta:
                if _is_ambiguous_name(l.component_name):
                    ta_review.append(l)
                else:
                    safe.append(l)
                continue
            if not sent:
                nomatch_links.append(l)
                continue
            if has_standalone_mention(l.component_name, sent.text):
                safe.append(l)
            else:
                nomatch_links.append(l)

        print(f"\n{'─' * 70}")
        print(f"  {ds}: {len(preliminary)} total, {len(safe)} safe, "
              f"{len(ta_review)} ta-deliberation (SKIPPED), {len(nomatch_links)} 4-rule judge")
        print(f"{'─' * 70}")

        if not nomatch_links:
            print(f"  No links to test with 4-rule judge, skipping")
            continue

        # Show what nomatch links are
        for l in nomatch_links:
            is_tp = (l.sentence_number, l.component_id) in gold
            label = "TP" if is_tp else "FP"
            sent = sent_map.get(l.sentence_number)
            text = sent.text[:70] if sent else "?"
            print(f"    [{label}] S{l.sentence_number}→{l.component_name} ({l.source}): {text}...")

        ds_results = {}
        for rule4_name, rule4_text in [("ORIGINAL", JUDGE_RULE4_ORIGINAL),
                                        ("SHORT", JUDGE_RULE4_SHORT)]:
            t0 = time.time()
            approved = _judge_nomatch(nomatch_links, comp_names, sent_map, llm, rule4_text)
            elapsed = time.time() - t0

            rejected = [l for l in nomatch_links
                        if (l.sentence_number, l.component_id) not in
                           {(a.sentence_number, a.component_id) for a in approved}]
            rej_tp = sum(1 for l in rejected if (l.sentence_number, l.component_id) in gold)
            rej_fp = len(rejected) - rej_tp

            print(f"\n  [{rule4_name}] {elapsed:.0f}s: {len(approved)} approved, "
                  f"{len(rejected)} rejected ({rej_fp} FP caught, {rej_tp} TP killed)")
            for l in rejected:
                is_tp = (l.sentence_number, l.component_id) in gold
                label = "TP!" if is_tp else "FP"
                print(f"    [{label}] rejected S{l.sentence_number}→{l.component_name}")

            ds_results[rule4_name] = {"approved": len(approved), "rejected": len(rejected),
                                       "rej_tp": rej_tp, "rej_fp": rej_fp}

        results[ds] = ds_results

    return results


def test_variant_b(datasets):
    """Test Variant B: restore V32 _is_generic_mention (offline, no LLM)."""
    print("\n" + "=" * 80)
    print("  VARIANT B TEST: Restore V32 _is_generic_mention Heuristic")
    print("  (Offline — uses Phase 6 cache, no LLM calls)")
    print("=" * 80)

    results = {}
    for ds in datasets:
        components, sentences, sent_map, name_to_id = load_dataset(ds)
        gold = load_gold(ds)

        data6 = load_checkpoint(ds, "phase6")
        if not data6:
            print(f"\n  {ds}: SKIP (missing phase6 cache)")
            continue

        validated = data6["validated"]

        # Convert to SadSamLinks
        val_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
            for c in validated
        ]

        # Apply the V32 generic mention filter
        kept, removed = apply_generic_mention_filter(val_links, sent_map)

        print(f"\n{'─' * 70}")
        print(f"  {ds}: {len(val_links)} validated links")
        print(f"{'─' * 70}")

        removed_tp = sum(1 for lk in removed if (lk.sentence_number, lk.component_id) in gold)
        removed_fp = len(removed) - removed_tp

        print(f"  V32 heuristic would remove: {len(removed)} links "
              f"({removed_fp} FP, {removed_tp} TP)")
        for lk in removed:
            is_tp = (lk.sentence_number, lk.component_id) in gold
            label = "TP!" if is_tp else "FP"
            sent = sent_map.get(lk.sentence_number)
            text = sent.text[:80] if sent else "?"
            print(f"    [{label}] S{lk.sentence_number}→{lk.component_name} ({lk.source}): {text}...")

        results[ds] = {"removed": len(removed), "removed_tp": removed_tp, "removed_fp": removed_fp}

    return results


def test_e2e(datasets, llm):
    """End-to-end test: Phase 8c + Phase 9 combined for all variant combos."""
    print("\n" + "=" * 80)
    print("  E2E TEST: Phase 8c + Phase 9 Combined (4 variants)")
    print("=" * 80)

    # Variant B (restore _is_generic_mention) DROPPED: V33 Phase 6 produces
    # +32 TP, +7 FP vs V32 — disabling the heuristic is clearly better.
    VARIANTS = {
        "V33_ORIG":  {"guide": GUIDE_ORIGINAL, "rule4": JUDGE_RULE4_ORIGINAL, "generic_filter": False},
        "A_SLIM":    {"guide": GUIDE_SLIM,     "rule4": JUDGE_RULE4_ORIGINAL, "generic_filter": False},
        "A2_SLIMV2": {"guide": GUIDE_SLIM_V2,  "rule4": JUDGE_RULE4_ORIGINAL, "generic_filter": False},
        "C_SHORT4":  {"guide": GUIDE_ORIGINAL, "rule4": JUDGE_RULE4_SHORT,    "generic_filter": False},
        "D_BEST":    {"guide": GUIDE_SLIM_V2,  "rule4": JUDGE_RULE4_SHORT,    "generic_filter": False},
    }

    all_results = {}
    for ds in datasets:
        components, sentences, sent_map, name_to_id = load_dataset(ds)
        gold = load_gold(ds)

        preliminary, transarc_set = reconstruct_pre8c_links(ds, sentences, name_to_id)
        data3 = load_checkpoint(ds, "phase3")
        if preliminary is None:
            print(f"\n  {ds}: SKIP")
            continue

        doc_knowledge = data3["doc_knowledge"] if data3 else None

        print(f"\n{'━' * 70}")
        print(f"  {ds}: {len(preliminary)} pre-8c links, {len(gold)} gold")
        print(f"{'━' * 70}")

        ds_results = {}
        for vname, cfg in VARIANTS.items():
            t0 = time.time()

            # Optional: apply generic mention filter to remove V33's extra links
            input_links = list(preliminary)
            if cfg["generic_filter"]:
                kept_gen, removed_gen = apply_generic_mention_filter(input_links, sent_map)
                gen_removed = len(removed_gen)
                # Only remove non-transarc entity/validated links
                filtered = []
                for lk in input_links:
                    is_ta = (lk.sentence_number, lk.component_id) in transarc_set
                    if is_ta or lk.source in ("transarc", "coreference", "partial_inject"):
                        filtered.append(lk)
                    elif not is_generic_mention_v32(lk.component_name,
                                                    sent_map.get(lk.sentence_number, type('', (), {'text': ''})()).text
                                                    if sent_map.get(lk.sentence_number) else ""):
                        filtered.append(lk)
                    # else: removed by generic mention filter
                input_links = filtered
            else:
                gen_removed = 0

            # Phase 8c
            post8c, rejected_8c = run_convention_filter(
                input_links, sent_map, transarc_set, llm, components, cfg["guide"]
            )

            # Phase 9
            reviewed = run_judge_review(
                post8c, sentences, components, sent_map, transarc_set,
                llm, doc_knowledge, cfg["rule4"]
            )

            elapsed = time.time() - t0
            pairs = links_to_pairs(reviewed)
            m = eval_metrics(pairs, gold)

            rej8c_tp = sum(1 for lk, _ in rejected_8c if (lk.sentence_number, lk.component_id) in gold)

            print(f"\n  [{vname}] {elapsed:.0f}s")
            if gen_removed:
                print(f"    Generic filter: removed {gen_removed} links")
            print(f"    8c: rejected {len(rejected_8c)} ({len(rejected_8c) - rej8c_tp} FP, {rej8c_tp} TP)")
            print(f"    9:  {len(reviewed)}/{len(post8c)} approved")
            print(f"    => P={m['P']:.1%} R={m['R']:.1%} F1={m['F1']:.1%} "
                  f"(TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")

            ds_results[vname] = m
            all_results.setdefault(vname, {})[ds] = m

        # Per-dataset comparison table
        print(f"\n  {ds} comparison:")
        print(f"    {'Variant':<15} {'P':>6} {'R':>6} {'F1':>6} {'FP':>4} {'FN':>4}")
        for vname in VARIANTS:
            m = ds_results[vname]
            print(f"    {vname:<15} {m['P']:>5.1%} {m['R']:>5.1%} {m['F1']:>5.1%} "
                  f"{m['fp']:>4} {m['fn']:>4}")

    # Summary
    if all_results:
        print(f"\n{'━' * 70}")
        print(f"  MACRO AVERAGES")
        print(f"{'━' * 70}")
        print(f"  {'Variant':<15} {'Macro P':>8} {'Macro R':>8} {'Macro F1':>9} {'Tot FP':>7} {'Tot FN':>7}")
        for vname in VARIANTS:
            if vname not in all_results:
                continue
            ds_ms = all_results[vname]
            if not ds_ms:
                continue
            macro_p = sum(m["P"] for m in ds_ms.values()) / len(ds_ms)
            macro_r = sum(m["R"] for m in ds_ms.values()) / len(ds_ms)
            macro_f1 = sum(m["F1"] for m in ds_ms.values()) / len(ds_ms)
            tot_fp = sum(m["fp"] for m in ds_ms.values())
            tot_fn = sum(m["fn"] for m in ds_ms.values())
            print(f"  {vname:<15} {macro_p:>7.1%} {macro_r:>7.1%} {macro_f1:>8.1%} "
                  f"{tot_fp:>7} {tot_fn:>7}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Test prompt simplification variants")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    parser.add_argument("--phase", choices=["8c", "9", "b", "e2e", "all"], default="all",
                        help="Which test to run: 8c, 9, b (generic heuristic), e2e, or all")
    args = parser.parse_args()

    datasets = [d for d in args.datasets if d in DATASETS]
    if not datasets:
        print("No valid datasets specified")
        return

    print(f"Datasets: {', '.join(datasets)}")
    print(f"Phase: {args.phase}")

    llm = LLMClient()

    if args.phase in ("b", "all"):
        test_variant_b(datasets)

    if args.phase in ("8c", "all"):
        test_phase8c(datasets, llm)

    if args.phase in ("9", "all"):
        test_phase9(datasets, llm)

    if args.phase in ("e2e", "all"):
        test_e2e(datasets, llm)


if __name__ == "__main__":
    main()
