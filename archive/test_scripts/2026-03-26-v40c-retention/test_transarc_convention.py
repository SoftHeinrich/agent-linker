#!/usr/bin/env python3
"""
Empirical test: What happens if TransArc links go through the convention filter (Phase 8c)?

Currently TransArc links bypass the convention filter (line 698 in V39).
This script loads V39 checkpoints, reconstructs TransArc links, and runs them
through the convention filter LLM prompt to see which would be rejected.

Compares results against gold standard to measure:
- How many TransArc FPs the convention filter catches
- How many TransArc TPs the convention filter kills
- Net F1 impact per dataset

Uses LIVE LLM calls for the convention filter prompt.
"""

import csv
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load .env
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

os.environ.setdefault("CLAUDE_MODEL", "sonnet")

from llm_sad_sam.llm_client import LLMClient, LLMBackend
from llm_sad_sam.core.data_types import SadSamLink

# ── Paths ──────────────────────────────────────────────────────────────────

BENCHMARK_DIR = Path("/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark")
CHECKPOINT_DIR = Path("results/phase_cache/v39")

GOLD_STANDARD_FILES = {
    "mediastore": BENCHMARK_DIR / "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    "teastore": BENCHMARK_DIR / "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    "teammates": BENCHMARK_DIR / "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "bigbluebutton": BENCHMARK_DIR / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "jabref": BENCHMARK_DIR / "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
}

TEXT_FILES = {
    "mediastore": BENCHMARK_DIR / "mediastore/text_2016/mediastore.txt",
    "teastore": BENCHMARK_DIR / "teastore/text_2020/teastore.txt",
    "teammates": BENCHMARK_DIR / "teammates/text_2021/teammates.txt",
    "bigbluebutton": BENCHMARK_DIR / "bigbluebutton/text_2021/bigbluebutton.txt",
    "jabref": BENCHMARK_DIR / "jabref/text_2021/jabref.txt",
}

DATASETS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]

# ── Convention guide (same as V39) ─────────────────────────────────────────

CONVENTION_GUIDE = """### STEP 1 — Hierarchical name reference (not about the component itself)?

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


def load_gold(dataset: str) -> set[tuple[int, str]]:
    gold = set()
    with open(GOLD_STANDARD_FILES[dataset]) as f:
        for row in csv.DictReader(f):
            gold.add((int(row["sentence"]), row["modelElementID"]))
    return gold


def load_sentences(dataset: str) -> dict[int, str]:
    """Load sentences as {1-indexed number: text}."""
    text_path = TEXT_FILES[dataset]
    sent_map = {}
    with open(text_path) as f:
        for i, line in enumerate(f, 1):
            sent_map[i] = line.strip()
    return sent_map


def load_checkpoints(dataset: str) -> tuple[list, set, list, dict]:
    """Load V39 checkpoints, return (preliminary, transarc_set, transarc_links, phase3)."""
    base = CHECKPOINT_DIR / dataset

    with open(base / "pre_judge.pkl", "rb") as f:
        pre_judge = pickle.load(f)

    with open(base / "phase4.pkl", "rb") as f:
        phase4 = pickle.load(f)

    # Try to load phase3 for doc_knowledge context
    phase3 = {}
    p3_path = base / "phase3.pkl"
    if p3_path.exists():
        with open(p3_path, "rb") as f:
            phase3 = pickle.load(f)

    preliminary = pre_judge["preliminary"]
    transarc_set = pre_judge["transarc_set"]
    transarc_links = phase4["transarc_links"]

    return preliminary, transarc_set, transarc_links, phase3


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
    # Try to find array in text
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def run_convention_filter(llm, links, sent_map, comp_names, alias_context=""):
    """Run convention filter on a set of links. Returns (kept, rejected)."""
    items = []
    for i, lk in enumerate(links):
        text = sent_map.get(lk.sentence_number, "(no text)")
        items.append(
            f'{i+1}. S{lk.sentence_number}: "{text}"\n'
            f'   Component: "{lk.component_name}"'
        )

    if not items:
        return links, []

    batch_size = 25
    all_verdicts = {}

    for batch_start in range(0, len(items), batch_size):
        batch_items = items[batch_start:batch_start + batch_size]

        prompt = f"""Validate trace links between architecture documentation and components.

ARCHITECTURE COMPONENTS: {', '.join(comp_names)}

{alias_context}{CONVENTION_GUIDE}

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

    kept = []
    rejected = []
    for i, lk in enumerate(links):
        verdict, step, reason = all_verdicts.get(i + 1, ("LINK", "3", "default"))
        if "NO" in verdict:
            rejected.append((lk, step, reason))
        else:
            kept.append(lk)

    return kept, rejected


def analyze_dataset(dataset: str, llm: LLMClient):
    """Run convention filter on TransArc links for one dataset."""
    gold = load_gold(dataset)
    sent_map = load_sentences(dataset)
    preliminary, transarc_set, transarc_links_raw, phase3 = load_checkpoints(dataset)

    # Get TransArc links from preliminary (post-dedup, these are the ones that matter)
    transarc_links = [l for l in preliminary if l.source == "transarc"]
    non_transarc_links = [l for l in preliminary if l.source != "transarc"]

    # Get component names from all links
    comp_names = sorted(set(l.component_name for l in preliminary))

    # Build alias context from phase3 if available
    alias_context = ""
    if phase3:
        dk = phase3.get("doc_knowledge")
        if dk:
            alias_lines = []
            if hasattr(dk, 'partial_references'):
                for partial, comp in dk.partial_references.items():
                    alias_lines.append(f'  "{partial}" is a confirmed short name for {comp}')
            if hasattr(dk, 'synonyms'):
                for syn, comp in dk.synonyms.items():
                    alias_lines.append(f'  "{syn}" is a confirmed synonym for {comp}')
            if hasattr(dk, 'abbreviations'):
                for abbr, comp in dk.abbreviations.items():
                    alias_lines.append(f'  "{abbr}" is a confirmed abbreviation for {comp}')
            if alias_lines:
                alias_context = (
                    "CONFIRMED ALIASES (from document analysis):\n"
                    + "\n".join(alias_lines)
                    + "\n\nIMPORTANT: When a confirmed alias appears in a sentence, it IS a reference "
                    "to that component — even inside compound phrases.\n\n"
                )

    print(f"\n{'=' * 80}")
    print(f"  {dataset.upper()} — {len(transarc_links)} TransArc links to filter")
    print(f"{'=' * 80}")

    # Classify TransArc links as TP/FP
    ta_tp = [l for l in transarc_links if (l.sentence_number, l.component_id) in gold]
    ta_fp = [l for l in transarc_links if (l.sentence_number, l.component_id) not in gold]
    print(f"  TransArc: {len(ta_tp)} TP, {len(ta_fp)} FP")

    # Run convention filter on TransArc links
    kept, rejected = run_convention_filter(llm, transarc_links, sent_map, comp_names, alias_context)

    # Analyze rejected links
    rejected_tp = []
    rejected_fp = []
    for lk, step, reason in rejected:
        key = (lk.sentence_number, lk.component_id)
        if key in gold:
            rejected_tp.append((lk, step, reason))
        else:
            rejected_fp.append((lk, step, reason))

    print(f"\n  Convention filter results:")
    print(f"    Kept:     {len(kept)}")
    print(f"    Rejected: {len(rejected)}")
    print(f"      FPs caught: {len(rejected_fp)}")
    print(f"      TPs killed: {len(rejected_tp)}")

    if rejected_fp:
        print(f"\n    REJECTED FPs (good — these are correctly filtered):")
        for lk, step, reason in rejected_fp:
            print(f"      S{lk.sentence_number:3d} -> {lk.component_name:<30s} step={step} | {reason}")
            print(f"             \"{sent_map.get(lk.sentence_number, '?')}\"")

    if rejected_tp:
        print(f"\n    REJECTED TPs (bad — these are incorrectly filtered):")
        for lk, step, reason in rejected_tp:
            print(f"      S{lk.sentence_number:3d} -> {lk.component_name:<30s} step={step} | {reason}")
            print(f"             \"{sent_map.get(lk.sentence_number, '?')}\"")

    # Compute F1 impact
    # Current: all preliminary links (pre_judge checkpoint = post-8c)
    # The pre_judge already excluded convention-filtered non-TA links
    current_set = {(l.sentence_number, l.component_id) for l in preliminary}
    current_tp = len(current_set & gold)
    current_fp = len(current_set - gold)
    current_fn = len(gold - current_set)

    # New: remove convention-rejected TransArc links from preliminary
    rejected_keys = {(lk.sentence_number, lk.component_id) for lk, _, _ in rejected}
    new_links = [l for l in preliminary if (l.sentence_number, l.component_id) not in rejected_keys]
    new_set = {(l.sentence_number, l.component_id) for l in new_links}
    new_tp = len(new_set & gold)
    new_fp = len(new_set - gold)
    new_fn = len(gold - new_set)

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        return 2 * p * r / (p + r) if (p + r) else 0

    current_f1 = f1(current_tp, current_fp, current_fn)
    new_f1 = f1(new_tp, new_fp, new_fn)
    delta = new_f1 - current_f1

    print(f"\n  F1 impact (pre-judge, convention filter only):")
    print(f"    Current (TA immune):  TP={current_tp} FP={current_fp} FN={current_fn} F1={current_f1:.1%}")
    print(f"    New (TA filtered):    TP={new_tp} FP={new_fp} FN={new_fn} F1={new_f1:.1%}")
    print(f"    Delta:                {delta:+.1%} ({delta*100:+.2f}pp)")

    return {
        "dataset": dataset,
        "ta_total": len(transarc_links),
        "ta_tp": len(ta_tp),
        "ta_fp": len(ta_fp),
        "rejected_fp": len(rejected_fp),
        "rejected_tp": len(rejected_tp),
        "current_f1": current_f1,
        "new_f1": new_f1,
        "delta": delta,
        "current_tp": current_tp,
        "current_fp": current_fp,
        "new_tp": new_tp,
        "new_fp": new_fp,
    }


def main():
    _backend_env = os.environ.get("LLM_BACKEND", "claude")
    backend = LLMBackend.OPENAI if _backend_env == "openai" else LLMBackend.CLAUDE
    llm = LLMClient(backend=backend)

    print("=" * 80)
    print("EMPIRICAL TEST: TransArc links through convention filter (Phase 8c)")
    print(f"Backend: {backend.value}")
    print("=" * 80)

    results = []
    for dataset in DATASETS:
        checkpoint_path = CHECKPOINT_DIR / dataset / "pre_judge.pkl"
        if not checkpoint_path.exists():
            print(f"\n  SKIP {dataset}: no checkpoint")
            continue
        result = analyze_dataset(dataset, llm)
        results.append(result)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"  {'Dataset':<15s} {'TA':>4s} {'TP':>4s} {'FP':>4s} {'FP caught':>10s} {'TP killed':>10s} {'F1 curr':>8s} {'F1 new':>8s} {'Delta':>8s}")
    print(f"  {'─'*15} {'─'*4} {'─'*4} {'─'*4} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")

    total_caught = 0
    total_killed = 0
    macro_curr = 0
    macro_new = 0

    for r in results:
        print(f"  {r['dataset']:<15s} {r['ta_total']:>4d} {r['ta_tp']:>4d} {r['ta_fp']:>4d} "
              f"{r['rejected_fp']:>10d} {r['rejected_tp']:>10d} "
              f"{r['current_f1']:>7.1%} {r['new_f1']:>7.1%} {r['delta']:>+7.2%}")
        total_caught += r["rejected_fp"]
        total_killed += r["rejected_tp"]
        macro_curr += r["current_f1"]
        macro_new += r["new_f1"]

    n = len(results)
    if n:
        print(f"\n  Macro avg F1:  current={macro_curr/n:.1%}  new={macro_new/n:.1%}  delta={((macro_new-macro_curr)/n)*100:+.2f}pp")
        print(f"  Total FPs caught: {total_caught}  TPs killed: {total_killed}")

        if total_killed == 0 and total_caught > 0:
            print(f"\n  VERDICT: SAFE — convention filter catches {total_caught} TransArc FPs with 0 TP loss")
        elif total_killed > total_caught:
            print(f"\n  VERDICT: UNSAFE — kills more TPs ({total_killed}) than FPs caught ({total_caught})")
        else:
            print(f"\n  VERDICT: MIXED — catches {total_caught} FPs but kills {total_killed} TPs. Check per-dataset.")

    print()


if __name__ == "__main__":
    main()
