#!/usr/bin/env python3
"""Test: what happens when partial_inject links lose their immunity?

V31 protects partial_inject links from:
  1. Convention filter (P8c): partial_inject source → immune
  2. Phase 9 judge: syn-safe bypass (_has_alias_mention) → skips judge

The ASSUMPTION is that removing protection would kill TPs. But if the judge
works correctly, it should kill FPs and keep TPs. This test verifies that.

Approach:
  - Load V31 pre_judge checkpoints (links AFTER convention filter, BEFORE judge)
  - Extract partial_inject links
  - Feed them through: (A) convention filter, (B) judge, (C) both
  - Compare against gold standard
  - Report: TPs killed vs FPs caught for each scenario
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

from llm_sad_sam.core.data_types import SadSamLink
from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.llm_client import LLMClient
from llm_sad_sam.pcm_parser import parse_pcm_repository

BENCHMARK_BASE = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
DATASETS = {
    "teammates": {
        "text": BENCHMARK_BASE / "teammates" / "text_2021" / "teammates.txt",
        "model": BENCHMARK_BASE / "teammates" / "model_2021" / "pcm" / "teammates.repository",
    },
    "bigbluebutton": {
        "text": BENCHMARK_BASE / "bigbluebutton" / "text_2021" / "bigbluebutton.txt",
        "model": BENCHMARK_BASE / "bigbluebutton" / "model_2021" / "pcm" / "bbb.repository",
    },
    "mediastore": {
        "text": BENCHMARK_BASE / "mediastore" / "text_2016" / "mediastore.txt",
        "model": BENCHMARK_BASE / "mediastore" / "model_2016" / "pcm" / "ms.repository",
    },
    "teastore": {
        "text": BENCHMARK_BASE / "teastore" / "text_2020" / "teastore.txt",
        "model": BENCHMARK_BASE / "teastore" / "model_2020" / "pcm" / "teastore.repository",
    },
    "jabref": {
        "text": BENCHMARK_BASE / "jabref" / "text_2021" / "jabref.txt",
        "model": BENCHMARK_BASE / "jabref" / "model_2021" / "pcm" / "jabref.repository",
    },
}
V31_CACHE = Path("./results/phase_cache/v31")

# ── V31's convention guide (same as in ilinker2_v31.py) ──────────────
CONVENTION_GUIDE = """### STEP 1 — Hierarchical name reference (not about the component itself)?

The most common reason for NO_LINK: the sentence mentions the component name only
as part of a HIERARCHICAL/QUALIFIED NAME (dotted path, namespace, module path) that
refers to an internal sub-unit, not the component's own architectural role.

Software documentation commonly uses hierarchical naming (e.g., "X.config", "X/handlers",
"X::internal") to refer to parts inside a component. The component name appears only as
a prefix, not as the subject.

Recognize these patterns — all are NO_LINK for component X:
- "X.config loads environment variables" — dotted sub-unit reference
- "X.handlers, X.mappers, X.converters follow a pipeline" — listing internals of X
- "Classes in the X.internal module are not visible outside" — even with
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
NO_LINK when the sentence:
- Describes what a technology IS (definition, capabilities)
- Lists technologies as stack dependencies
- Names a COMPOUND ENTITY containing the component name
  ("X Protocol specification" → NO_LINK for "X")
- Uses the name as part of a METHODOLOGY ("X testing in CI" → NO_LINK for "X")

LINK when components INTERACT with or connect THROUGH the technology.

**2b. Generic word collision:**
NO_LINK — narrow, non-architectural sense:
- Process/activity modifier: "cascade X", "retry X", "validation X"
- Hardware/deployment: "a dedicated hardware node", "32-core server"
- Possessive/personal: "her settings", "their preferences"

LINK — system-level architectural sense:
- System name + word: "the [System] gateway"
- Architectural role: "the orchestrator routes jobs to the gateway"

### STEP 3 — Default: LINK.
If neither Step 1 nor Step 2 applies → LINK.

### Priority:
Be AGGRESSIVE with NO_LINK on sub-package descriptions (Step 1).
For Step 2, only NO_LINK when confident. Default to LINK."""


def load_checkpoint(dataset, phase_name):
    path = V31_CACHE / dataset / f"{phase_name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dataset(dataset):
    paths = DATASETS[dataset]
    components = parse_pcm_repository(str(paths["model"]))
    sentences = DocumentLoader.load_sentences(str(paths["text"]))
    sent_map = DocumentLoader.build_sent_map(sentences)
    return components, sentences, sent_map


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


def _extract_json_array(text):
    """Extract a JSON array from LLM text that may have markdown fences."""
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


# ── Test A: Convention filter on partial_inject links ─────────────────

def test_convention_filter(partial_links, sent_map, comp_names, llm):
    """Feed partial_inject links through the convention filter (removing immunity)."""
    if not partial_links:
        return [], []

    items = []
    for i, lk in enumerate(partial_links):
        sent = sent_map.get(lk.sentence_number)
        text = sent.text if sent else "(no text)"
        items.append(
            f'{i+1}. S{lk.sentence_number}: "{text}"\n'
            f'   Component: "{lk.component_name}"'
        )

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
    data = _extract_json_array(raw.text if hasattr(raw, 'text') else str(raw))

    kept, rejected = [], []
    verdicts = {}
    if data:
        for item in data:
            vid = item.get("id")
            verdict = item.get("verdict", "LINK").upper().strip()
            step = item.get("step", "3")
            reason = item.get("reason", "")
            if vid is not None:
                verdicts[vid] = (verdict, step, reason)

    for i, lk in enumerate(partial_links):
        verdict, step, reason = verdicts.get(i + 1, ("LINK", "3", "default"))
        if "NO" in verdict:
            rejected.append((lk, f"step{step}", reason))
        else:
            kept.append(lk)

    return kept, rejected


# ── Test B: Phase 9 judge on partial_inject links ────────────────────

def test_judge(partial_links, sent_map, comp_names, llm):
    """Feed partial_inject links through the Phase 9 judge (removing syn-safe bypass)."""
    if not partial_links:
        return [], []

    cases = []
    for i, lk in enumerate(partial_links):
        sent = sent_map.get(lk.sentence_number)
        text = sent.text if sent else "(no text)"
        cases.append(f'{i+1}. S{lk.sentence_number}: "{text}" → {lk.component_name}')

    # Use V31's reframed 4-rule judge prompt
    prompt = f"""JUDGE: Validate trace links between documentation and software architecture components.

APPROVAL CRITERIA:
A link S→C is valid when the sentence satisfies all four conditions:

1. EXPLICIT REFERENCE
   The component name (or a direct reference to it) appears in the sentence as a clear
   entity being discussed. This distinguishes component-specific statements from
   incidental mentions or generic discussions where the component name appears but is
   not the subject of the statement.

2. SYSTEM-LEVEL PERSPECTIVE
   The sentence describes the component's role, responsibilities, interfaces, or
   interactions within the overall system architecture. Reject statements focused on
   internal implementation details (data structures, algorithms, code-level concerns)
   that are invisible at the architectural abstraction level.

3. PRIMARY FOCUS
   The component is the main subject of what the sentence conveys, not a secondary
   or incidental mention. The sentence is fundamentally about what the component does
   or how it relates to other system elements.

4. COMPONENT-SPECIFIC USAGE
   The reference is to the component as a named entity within the system architecture,
   not to a generic concept, pattern, or technology that happens to share a name.
   This distinguishes component-specific statements from domain terminology or
   methodological discussions that use common words.

COMPONENTS: {', '.join(comp_names)}

LINKS:
{chr(10).join(cases)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief explanation"}}]}}
JSON only:"""

    raw = llm.query(prompt, timeout=180)
    raw_text = raw.text if hasattr(raw, 'text') else str(raw)

    # Parse judge response
    try:
        # Try direct JSON parse
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Try extracting JSON object
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw_text[start:end+1])
            except json.JSONDecodeError:
                data = None
        else:
            data = None

    judgments = {}
    if data and "judgments" in data:
        for j in data["judgments"]:
            case_num = j.get("case")
            approve = j.get("approve", True)
            reason = j.get("reason", "")
            if case_num is not None:
                judgments[case_num] = (approve, reason)

    kept, rejected = [], []
    for i, lk in enumerate(partial_links):
        approve, reason = judgments.get(i + 1, (True, "default"))
        if not approve:
            rejected.append((lk, "judge", reason))
        else:
            kept.append(lk)

    return kept, rejected


# ── Test D: Enriched judge (alias mappings + context window) ─────────

def test_enriched_judge(partial_links, sent_map, sentences, comp_names, doc_knowledge, llm):
    """Judge with alias context: tells the judge which partial names map to which components,
    and provides ±1 sentence context window for disambiguation."""
    if not partial_links:
        return [], []

    # Build alias info from doc_knowledge
    alias_info = {}
    if doc_knowledge:
        for partial, comp in doc_knowledge.partial_references.items():
            alias_info.setdefault(comp, []).append(f'"{partial}" (partial reference)')
        for syn, comp in doc_knowledge.synonyms.items():
            alias_info.setdefault(comp, []).append(f'"{syn}" (synonym)')
        for abbr, comp in doc_knowledge.abbreviations.items():
            alias_info.setdefault(comp, []).append(f'"{abbr}" (abbreviation)')

    alias_section = ""
    if alias_info:
        lines = []
        for comp, aliases in sorted(alias_info.items()):
            lines.append(f"  {comp}: {', '.join(aliases)}")
        alias_section = "KNOWN ALIASES (discovered from the document):\n" + "\n".join(lines)

    # Build cases with ±1 context window
    sent_list = sorted(sentences, key=lambda s: s.number)
    sent_by_num = {s.number: s for s in sent_list}

    cases = []
    for i, lk in enumerate(partial_links):
        sent = sent_map.get(lk.sentence_number)
        text = sent.text if sent else "(no text)"

        # ±1 context
        ctx_lines = []
        prev_sent = sent_by_num.get(lk.sentence_number - 1)
        next_sent = sent_by_num.get(lk.sentence_number + 1)
        if prev_sent:
            ctx_lines.append(f'    [S{prev_sent.number}]: "{prev_sent.text}"')
        ctx_lines.append(f'    [S{lk.sentence_number}]: "{text}"  ← TARGET')
        if next_sent:
            ctx_lines.append(f'    [S{next_sent.number}]: "{next_sent.text}"')
        ctx_block = "\n".join(ctx_lines)

        cases.append(
            f'{i+1}. Component: "{lk.component_name}"\n'
            f'   Matched via alias: the word in the sentence is a KNOWN PARTIAL REFERENCE for this component.\n'
            f'   Context:\n{ctx_block}'
        )

    prompt = f"""JUDGE: Validate trace links where a PARTIAL NAME of a component appears in a sentence.

IMPORTANT CONTEXT: These links were created because a KNOWN ALIAS (partial reference,
synonym, or abbreviation) for the component was found in the sentence text. The alias
mapping was discovered earlier from the document itself — it is not a guess.

Your task: determine whether the alias in THIS sentence refers to the component in an
ARCHITECTURAL sense, or is just a coincidental use of the same word.

{alias_section}

DECISION CRITERIA:

APPROVE (LINK) when:
- The alias word refers to the component in its architectural role, even if the component
  is not the primary subject. Any genuine reference counts.
- The sentence describes what the component does, how it connects to other components,
  or its responsibilities in the system.
- The surrounding context (±1 sentence) confirms the architectural discussion is about
  this component or its subsystem.

REJECT (NO_LINK) when:
- The alias word is used in its ordinary English meaning, completely unrelated to the
  component (e.g., "cascade logic" uses "logic" as a generic English word, not as the
  Logic component; "hardware server" uses "server" as generic hardware).
- The alias word appears only inside a dotted package path (e.g., "logic.core" — the
  word is a namespace prefix, not a component reference).
- The word clearly refers to a DIFFERENT entity (e.g., "Kurento Media Server" is about
  Kurento, not HTML5 Server).

KEY PRINCIPLE: When in doubt, APPROVE. The alias mapping was established from the document,
so there is a prior reason to believe the word refers to the component. Only reject when
the generic/unrelated usage is clear.

COMPONENTS: {', '.join(comp_names)}

LINKS TO JUDGE:
{chr(10).join(cases)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief explanation"}}]}}
JSON only:"""

    raw = llm.query(prompt, timeout=180)
    raw_text = raw.text if hasattr(raw, 'text') else str(raw)

    try:
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw_text[start:end+1])
            except json.JSONDecodeError:
                data = None
        else:
            data = None

    judgments = {}
    if data and "judgments" in data:
        for j in data["judgments"]:
            case_num = j.get("case")
            approve = j.get("approve", True)
            reason = j.get("reason", "")
            if case_num is not None:
                judgments[case_num] = (approve, reason)

    kept, rejected = [], []
    for i, lk in enumerate(partial_links):
        approve, reason = judgments.get(i + 1, (True, "default"))
        if not approve:
            rejected.append((lk, "enriched_judge", reason))
        else:
            kept.append(lk)

    return kept, rejected


# ── Main ──────────────────────────────────────────────────────────────

def main():
    os.environ.setdefault("CLAUDE_MODEL", "sonnet")
    llm = LLMClient()

    print("=" * 90)
    print("  TEST: Partial-inject links WITHOUT protection")
    print("  Question: Does removing immunity kill TPs, or just catch FPs?")
    print("=" * 90)

    total_stats = {
        "pi_tp": 0, "pi_fp": 0,
        "conv_tp_killed": 0, "conv_fp_caught": 0,
        "judge_tp_killed": 0, "judge_fp_caught": 0,
        "both_tp_killed": 0, "both_fp_caught": 0,
        "enriched_tp_killed": 0, "enriched_fp_caught": 0,
    }

    for ds_name in DATASETS:
        pre_judge = load_checkpoint(ds_name, "pre_judge")
        if not pre_judge:
            print(f"\n  {ds_name}: SKIP (no pre_judge checkpoint)")
            continue

        preliminary = pre_judge["preliminary"]
        transarc_set = pre_judge["transarc_set"]

        # Extract partial_inject links
        pi_links = [lk for lk in preliminary if lk.source == "partial_inject"]
        if not pi_links:
            print(f"\n  {ds_name}: 0 partial_inject links — nothing to test")
            continue

        components, sentences, sent_map = load_dataset(ds_name)
        gold = load_gold(ds_name)
        comp_names = [c.name for c in components]

        # Classify current PI links as TP/FP
        pi_tp = [lk for lk in pi_links if (lk.sentence_number, lk.component_id) in gold]
        pi_fp = [lk for lk in pi_links if (lk.sentence_number, lk.component_id) not in gold]

        print(f"\n{'━' * 90}")
        print(f"  {ds_name}: {len(pi_links)} partial_inject links ({len(pi_tp)} TP, {len(pi_fp)} FP)")
        print(f"{'━' * 90}")

        for lk in pi_links:
            is_tp = (lk.sentence_number, lk.component_id) in gold
            sent = sent_map.get(lk.sentence_number)
            text = sent.text[:80] if sent else "?"
            print(f"    {'TP' if is_tp else 'FP'} S{lk.sentence_number} → {lk.component_name}: \"{text}...\"")

        total_stats["pi_tp"] += len(pi_tp)
        total_stats["pi_fp"] += len(pi_fp)

        # ── Test A: Convention Filter ─────────────────────────────────
        print(f"\n  [A] Convention Filter (removing partial_inject immunity):")
        conv_kept, conv_rejected = test_convention_filter(pi_links, sent_map, comp_names, llm)

        conv_tp_killed = sum(1 for lk, _, _ in conv_rejected if (lk.sentence_number, lk.component_id) in gold)
        conv_fp_caught = sum(1 for lk, _, _ in conv_rejected if (lk.sentence_number, lk.component_id) not in gold)

        for lk, step, reason in conv_rejected:
            is_tp = (lk.sentence_number, lk.component_id) in gold
            label = "TP KILLED" if is_tp else "FP caught"
            print(f"    REJECT [{step}] [{label}] S{lk.sentence_number} → {lk.component_name}: {reason}")

        print(f"    Result: {conv_fp_caught} FP caught, {conv_tp_killed} TP killed "
              f"(of {len(pi_fp)} FP, {len(pi_tp)} TP)")

        total_stats["conv_tp_killed"] += conv_tp_killed
        total_stats["conv_fp_caught"] += conv_fp_caught

        # ── Test B: Phase 9 Judge ────────────────────────────────────
        print(f"\n  [B] Phase 9 Judge (removing syn-safe bypass):")
        judge_kept, judge_rejected = test_judge(pi_links, sent_map, comp_names, llm)

        judge_tp_killed = sum(1 for lk, _, _ in judge_rejected if (lk.sentence_number, lk.component_id) in gold)
        judge_fp_caught = sum(1 for lk, _, _ in judge_rejected if (lk.sentence_number, lk.component_id) not in gold)

        for lk, _, reason in judge_rejected:
            is_tp = (lk.sentence_number, lk.component_id) in gold
            label = "TP KILLED" if is_tp else "FP caught"
            print(f"    REJECT [judge] [{label}] S{lk.sentence_number} → {lk.component_name}: {reason}")

        print(f"    Result: {judge_fp_caught} FP caught, {judge_tp_killed} TP killed "
              f"(of {len(pi_fp)} FP, {len(pi_tp)} TP)")

        total_stats["judge_tp_killed"] += judge_tp_killed
        total_stats["judge_fp_caught"] += judge_fp_caught

        # ── Test C: Both (convention filter → judge on survivors) ────
        print(f"\n  [C] Both (convention filter THEN judge on survivors):")
        # Convention filter first
        conv_survivors = [lk for lk in conv_kept]  # TPs and FPs that passed convention filter
        if conv_survivors:
            both_kept, both_judge_rej = test_judge(conv_survivors, sent_map, comp_names, llm)
        else:
            both_kept, both_judge_rej = [], []

        # Total rejected = convention rejected + judge rejected (of survivors)
        both_all_rejected = list(conv_rejected) + both_judge_rej
        both_tp_killed = sum(1 for lk, _, _ in both_all_rejected if (lk.sentence_number, lk.component_id) in gold)
        both_fp_caught = sum(1 for lk, _, _ in both_all_rejected if (lk.sentence_number, lk.component_id) not in gold)

        for lk, step, reason in both_judge_rej:
            is_tp = (lk.sentence_number, lk.component_id) in gold
            label = "TP KILLED" if is_tp else "FP caught"
            print(f"    REJECT [judge-after-conv] [{label}] S{lk.sentence_number} → {lk.component_name}: {reason}")

        print(f"    Result: {both_fp_caught} FP caught, {both_tp_killed} TP killed "
              f"(of {len(pi_fp)} FP, {len(pi_tp)} TP)")

        total_stats["both_tp_killed"] += both_tp_killed
        total_stats["both_fp_caught"] += both_fp_caught

        # ── Test D: Enriched judge (alias mappings + context) ────────
        print(f"\n  [D] Enriched Judge (alias mappings + ±1 sentence context):")
        data3 = load_checkpoint(ds_name, "phase3")
        doc_knowledge = data3["doc_knowledge"] if data3 else None
        enr_kept, enr_rejected = test_enriched_judge(
            pi_links, sent_map, sentences, comp_names, doc_knowledge, llm
        )

        enr_tp_killed = sum(1 for lk, _, _ in enr_rejected if (lk.sentence_number, lk.component_id) in gold)
        enr_fp_caught = sum(1 for lk, _, _ in enr_rejected if (lk.sentence_number, lk.component_id) not in gold)

        for lk, _, reason in enr_rejected:
            is_tp = (lk.sentence_number, lk.component_id) in gold
            label = "TP KILLED" if is_tp else "FP caught"
            print(f"    REJECT [enriched] [{label}] S{lk.sentence_number} → {lk.component_name}: {reason}")

        if not enr_rejected:
            print(f"    No rejections — all links approved.")

        print(f"    Result: {enr_fp_caught} FP caught, {enr_tp_killed} TP killed "
              f"(of {len(pi_fp)} FP, {len(pi_tp)} TP)")

        total_stats["enriched_tp_killed"] += enr_tp_killed
        total_stats["enriched_fp_caught"] += enr_fp_caught

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'━' * 90}")
    print(f"  SUMMARY (all datasets)")
    print(f"{'━' * 90}")
    pi_total = total_stats["pi_tp"] + total_stats["pi_fp"]
    print(f"  Total partial_inject links: {pi_total} ({total_stats['pi_tp']} TP, {total_stats['pi_fp']} FP)")
    print()
    print(f"  {'Scenario':<35s} {'FP caught':>10s} {'TP killed':>10s} {'Net':>8s}  {'Safe?'}")
    print(f"  {'─'*35} {'─'*10} {'─'*10} {'─'*8}  {'─'*5}")

    for label, fp_key, tp_key in [
        ("A: Convention filter only", "conv_fp_caught", "conv_tp_killed"),
        ("B: Judge only (vanilla)", "judge_fp_caught", "judge_tp_killed"),
        ("C: Convention + Judge", "both_fp_caught", "both_tp_killed"),
        ("D: Enriched judge (aliases+ctx)", "enriched_fp_caught", "enriched_tp_killed"),
    ]:
        fp = total_stats[fp_key]
        tp = total_stats[tp_key]
        net = fp - tp
        safe = "YES" if tp == 0 else "RISKY" if tp <= 1 else "NO"
        print(f"  {label:<35s} {fp:>10d} {tp:>10d} {net:>+8d}  {safe}")

    print()
    if total_stats["both_tp_killed"] == 0:
        print("  CONCLUSION: Removing partial_inject immunity is SAFE.")
        print("  The judge/filter correctly distinguishes TPs from FPs.")
        print(f"  Potential improvement: -{total_stats['both_fp_caught']} FP with 0 TP lost.")
    elif total_stats["both_fp_caught"] > total_stats["both_tp_killed"]:
        print(f"  CONCLUSION: Mixed results — catches {total_stats['both_fp_caught']} FP "
              f"but kills {total_stats['both_tp_killed']} TP.")
        print("  Net gain but with TP damage. Consider selective approach.")
    else:
        print(f"  CONCLUSION: Removing immunity is HARMFUL — kills {total_stats['both_tp_killed']} TP "
              f"but only catches {total_stats['both_fp_caught']} FP.")
        print("  The protection is justified.")


if __name__ == "__main__":
    main()
