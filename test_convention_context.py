#!/usr/bin/env python3
"""Test: Convention filter with Phase 3 alias context.

Runs the convention filter on teastore S11/12/13/15→WebUI links
(which have "UI name" in text) to verify that providing Phase 3
context ("UI is a confirmed short name for WebUI") prevents the
filter from incorrectly killing them.

Also tests BBB Conversion links to ensure the filter still kills
true FPs where "Conversion" is used generically.
"""
import glob
import json
import os
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.stdout.reconfigure(line_buffering=True)

from llm_sad_sam.llm_client import LLMClient, LLMBackend
from llm_sad_sam.core.document_loader import DocumentLoader

os.environ.setdefault("CLAUDE_MODEL", "sonnet")

BENCHMARK = Path("/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark")

# Import CONVENTION_GUIDE from V39
from llm_sad_sam.linkers.experimental.ilinker2_v39 import CONVENTION_GUIDE


def run_convention_filter(llm, links, sent_map, comp_names, doc_knowledge,
                         activity_partials, label):
    """Run convention filter with Phase 3 alias context (NAME-type only)."""
    # Build alias context — exclude ORDINARY/activity-type partials
    alias_lines = []
    for partial, comp in doc_knowledge.partial_references.items():
        if partial not in activity_partials:
            alias_lines.append(f'  "{partial}" is a confirmed short name for {comp}')
        else:
            print(f"  (Excluding ORDINARY partial: \"{partial}\" → {comp})")
    for syn, comp in doc_knowledge.synonyms.items():
        alias_lines.append(f'  "{syn}" is a confirmed synonym for {comp}')
    for abbr, comp in doc_knowledge.abbreviations.items():
        alias_lines.append(f'  "{abbr}" is a confirmed abbreviation for {comp}')

    alias_context = ""
    if alias_lines:
        alias_context = (
            "CONFIRMED ALIASES (from document analysis):\n"
            + "\n".join(alias_lines)
            + "\n\nIMPORTANT: When a confirmed alias appears in a sentence, it IS a reference "
            "to that component — even inside compound phrases. For example, if \"UI\" is a "
            "confirmed short name for WebUI, then \"UI name\" in a sentence IS about WebUI.\n"
        )

    items = []
    for i, (snum, comp_name) in enumerate(links):
        sent = sent_map.get(snum)
        text = sent.text if sent else "(no text)"
        items.append(
            f'{i+1}. S{snum}: "{text}"\n'
            f'   Component: "{comp_name}"'
        )

    prompt = f"""Validate trace links between architecture documentation and components.

ARCHITECTURE COMPONENTS: {', '.join(comp_names)}

{alias_context}{CONVENTION_GUIDE}

---

For each sentence-component pair, apply the 3-step reasoning guide.
Decide LINK (keep the trace link) or NO_LINK (reject it).

{chr(10).join(items)}

Return JSON array:
[{{"id": N, "step": "1|2a|2b|3", "verdict": "LINK" or "NO_LINK", "reason": "brief"}}]
JSON only:"""

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"Alias context:\n{alias_context}")

    raw = llm.query(prompt, timeout=180)
    text = raw.text if hasattr(raw, 'text') else str(raw)
    print(f"\n  RAW LLM response (first 500 chars):\n  {text[:500]}")

    # Parse JSON array
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except:
        start = text.find("[")
        if start >= 0:
            depth = 0
            for j in range(start, len(text)):
                if text[j] == '[': depth += 1
                elif text[j] == ']': depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start:j+1])
                    except:
                        data = None
                    break
        else:
            data = None

    if data:
        for item in data:
            vid = item.get("id")
            verdict = item.get("verdict", "LINK").upper().strip()
            step = item.get("step", "3")
            reason = item.get("reason", "")
            if vid and vid <= len(links):
                snum, comp = links[vid - 1]
                status = "LINK" if "NO" not in verdict else "NO_LINK"
                print(f"  S{snum} -> {comp}: {status} [step {step}] — {reason}")


def main():
    llm = LLMClient(backend=LLMBackend.CLAUDE)
    loader = DocumentLoader()

    # ═══════════════════════════════════════════════════════════
    # TEST 1: TeaStore "UI name" links (should be LINK)
    # ═══════════════════════════════════════════════════════════
    with open("results/phase_cache/v39/teastore/phase3.pkl", "rb") as f:
        ts_p3 = pickle.load(f)

    ts_text = glob.glob(str(BENCHMARK / "teastore" / "**" / "teastore.txt"), recursive=True)
    ts_sents = loader.load_sentences(ts_text[0])
    ts_sent_map = {s.number: s for s in ts_sents}

    ts_dk = ts_p3['doc_knowledge']
    ts_comps = ['WebUI', 'Registry', 'Persistence', 'Recommender', 'Auth',
                'SlopeOneRecommender', 'OrderBasedRecommender', 'DummyRecommender',
                'PopularityBasedRecommender', 'ImageProvider', 'PreprocessedSlopeOneRecommender']

    ts_links = [(11, 'WebUI'), (12, 'WebUI'), (13, 'WebUI'), (15, 'WebUI')]

    # TeaStore has no activity partials (UI is NAME-type)
    run_convention_filter(llm, ts_links, ts_sent_map, ts_comps, ts_dk,
                         set(),  # no activity partials
                         "TEASTORE: 'UI name' → WebUI (expect LINK)")

    # ═══════════════════════════════════════════════════════════
    # TEST 2: BBB "conversion" links (FPs should be NO_LINK)
    # ═══════════════════════════════════════════════════════════
    with open("results/phase_cache/v39/bigbluebutton/phase3.pkl", "rb") as f:
        bbb_p3 = pickle.load(f)

    bbb_text = glob.glob(str(BENCHMARK / "bigbluebutton" / "**" / "bigbluebutton.txt"), recursive=True)
    bbb_sents = loader.load_sentences(bbb_text[0])
    bbb_sent_map = {s.number: s for s in bbb_sents}

    bbb_dk = bbb_p3['doc_knowledge']
    bbb_comps = ['HTML5 Client', 'HTML5 Server', 'Presentation Conversion',
                 'Apps', 'BBB web', 'FSESL', 'FreeSWITCH', 'Redis DB',
                 'Redis PubSub', 'Recording Service', 'WebRTC-SFU', 'kurento']

    # Mix of TPs and FPs
    bbb_links = [
        (76, 'Presentation Conversion'),  # FP: "conversion process"
        (79, 'Presentation Conversion'),  # FP: "conversion process"
        (80, 'Presentation Conversion'),  # TP: "Presentation conversion flow"
        (81, 'Presentation Conversion'),  # TP: "presentation conversion"
        (82, 'Presentation Conversion'),  # FP: "SWF, SVG and PNG conversion"
        (83, 'Presentation Conversion'),  # FP: "SVG conversion flow"
        (84, 'Presentation Conversion'),  # FP: "conversion fallback"
    ]

    # BBB: "Conversion" is ORDINARY (activity-type), excluded from alias context
    run_convention_filter(llm, bbb_links, bbb_sent_map, bbb_comps, bbb_dk,
                         {'Conversion'},  # activity partial — excluded from context
                         "BBB: 'Conversion' → Presentation Conversion (expect NO_LINK for FPs)")


if __name__ == "__main__":
    main()
