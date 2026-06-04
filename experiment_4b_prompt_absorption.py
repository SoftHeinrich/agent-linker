"""Phase 4b absorption via 3-word prompt modification.

The current twopass prompts (p1 = participation, p2 = specificity) approve
the 3 candidates Phase 4b later rejects on teammates. Can we kill those FPs
by adding ~3 words to ONE existing prompt — and not break legitimate refs?

Test set (cached from 17f openai run):
  Code-path FPs (kept by twopass, killed by Phase 4b, NOT in gold):
    S22  teammates "logic"            in "logic, ui.website, ui.controller represent..."
    S130 teammates "storage"          in "Package overview contains storage.api, storage.entity, ..."
    S159 teammates "common.datatransfer" in "common.datatransfer contains data transfer objects."
  Control TPs (kept by twopass, in gold — must NOT regress):
    S25  mediastore "Database component" → DB
    S31  mediastore "Database component" → DB
    S69  bigbluebutton "KMS"  → kurento
    S21  bigbluebutton "bbb-html5" → HTML5 Server

Prompt variants (3-word-ish modifications):
  baseline      : 17f prompts unchanged
  V1_validrules : add "appears only in a code path," to VALIDATION_RULES rejects
  V2_p2_focus   : add "or code-path token" to p2 focus
  V3_p1_focus   : add "and not just as a dotted-path identifier" to p1 focus
  V4_both_axes  : V1 + V2 combined (probe whether one axis suffices)

Reports:
  per-variant per-candidate: p1 / p2 / approved
  summary: which variants reject all 3 FPs while keeping all 4 TPs

Cost: 7 candidates × 5 variants × 2 passes = 70 LLM calls (cache-hit-friendly).
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

ROOT = Path(__file__).parent
BENCH = ROOT / "../ardoco/core/tests-base/src/main/resources/benchmark"
CACHE = ROOT / "results/phase_cache/s_linker17f/openai"

TEST_CASES = [
    # (dataset, snum, cid, expected, label)
    ("teammates", 22,  "_3LCnIKESEeu-mYqkDskRow", "reject", "FP code-path 'logic'"),
    ("teammates", 130, "_9JlCIKESEeu-mYqkDskRow", "reject", "FP code-path 'storage'"),
    ("teammates", 159, "_zUmhEKESEeu-mYqkDskRow", "reject", "FP code-path 'common.datatransfer'"),
    # Control TPs — must stay approved
    ("mediastore", 25, None, "approve", "TP DB via 'Database component'"),
    ("mediastore", 31, None, "approve", "TP DB via 'Database component'"),
    ("bigbluebutton", 69, None, "approve", "TP kurento via 'KMS'"),
    ("bigbluebutton", 21, None, "approve", "TP HTML5 Server via 'bbb-html5'"),
]

DATASETS_PATHS = {
    "mediastore":    (BENCH/"mediastore/text_2016/mediastore.txt",
                      BENCH/"mediastore/model_2016/pcm/ms.repository"),
    "teammates":     (BENCH/"teammates/text_2021/teammates.txt",
                      BENCH/"teammates/model_2021/pcm/teammates.repository"),
    "bigbluebutton": (BENCH/"bigbluebutton/text_2021/bigbluebutton.txt",
                      BENCH/"bigbluebutton/model_2021/pcm/bbb.repository"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt variants
# ─────────────────────────────────────────────────────────────────────────────
VALIDATION_RULES_BASELINE = (
    "Approve when the sentence treats the component as an architectural "
    "participant, including counterparts. Reject when the matching word is "
    "generic, names a different entity, or describes a technique that merely "
    "shares the component's name."
)
VALIDATION_RULES_V1 = (
    "Approve when the sentence treats the component as an architectural "
    "participant, including counterparts. Reject when the matching word is "
    "generic, names a different entity, appears only in a code path, or "
    "describes a technique that merely shares the component's name."
)
# +6 words: "appears only in a code path,"

P1_FOCUS_BASELINE = (
    "Check architectural participation: does the sentence name this component "
    "as an architectural participant — performing operations, providing "
    "services, or taking part in the described system behavior?"
)
P1_FOCUS_V3 = (
    "Check architectural participation: does the sentence name this component "
    "as an architectural participant — performing operations, providing "
    "services, or taking part in the described system behavior, and not just "
    "as a dotted-path identifier?"
)
# +7 words appended

P2_FOCUS_BASELINE = (
    "Check referential specificity: is the component name used to identify "
    "this specific architectural element, or does it serve as a generic "
    "technical term in this sentence?"
)
P2_FOCUS_V2 = (
    "Check referential specificity: is the component name used to identify "
    "this specific architectural element, or does it serve as a generic "
    "technical term or code-path token in this sentence?"
)
# +4 words: "or code-path token"

PROMPT_VARIANTS = {
    "baseline":       (VALIDATION_RULES_BASELINE, P1_FOCUS_BASELINE, P2_FOCUS_BASELINE),
    "V1_validrules":  (VALIDATION_RULES_V1,       P1_FOCUS_BASELINE, P2_FOCUS_BASELINE),
    "V2_p2_focus":    (VALIDATION_RULES_BASELINE, P1_FOCUS_BASELINE, P2_FOCUS_V2),
    "V3_p1_focus":    (VALIDATION_RULES_BASELINE, P1_FOCUS_V3,       P2_FOCUS_BASELINE),
    "V4_both_axes":   (VALIDATION_RULES_V1,       P1_FOCUS_BASELINE, P2_FOCUS_V2),
}


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def run_one_twopass(linker, case_text, comp_names, vrules, p1_focus, p2_focus, label):
    """Run twopass with custom prompts. Returns (p1, p2)."""
    # Build a one-shot validation prompt mirroring _run_validation_pass but
    # with overridable rules text.
    def query(focus):
        prompt = f"""Validate component references in a software architecture document. {focus}

COMPONENTS: {', '.join(comp_names)}

{vrules}

CASES:
{case_text}

Return JSON:
{{"validations": [{{"case": 1, "approve": true}}]}}
JSON only:"""
        data = None
        for attempt in range(2):
            data = linker.llm.extract_json(linker.llm.query(prompt, timeout=120))
            if data and data.get("validations"):
                break
        if not data:
            return False
        v = data["validations"][0] if data["validations"] else {}
        val = v.get("approve", False)
        return val is True or (isinstance(val, str) and val.lower() == "true")

    linker.llm.set_phase(f"prompt_4b_{label}_p1")
    p1 = query(p1_focus)
    linker.llm.set_phase(f"prompt_4b_{label}_p2")
    p2 = query(p2_focus)
    return p1, p2


def build_case_text(c, sent_map, linker):
    prev = sent_map.get(c.sentence_number - 1)
    p = f"[prev: {prev.text[:60]}] " if prev else ""
    bundle = linker._build_evidence_bundle(c, sent_map)
    return (
        f'Case 1: "{c.matched_text}" -> {c.component_name}\n'
        f'  {p}"{c.sentence_text}"\n'
        f'{linker._format_evidence(bundle)}'
    )


def main():
    os.environ.setdefault("LLM_BACKEND", "checkpoint")
    from llm_sad_sam.llm_client import LLMBackend
    from llm_sad_sam.linkers.experimental.s_linker17f import SLinker17f
    from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
    from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
    from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names

    backend_map = {"claude": LLMBackend.CLAUDE, "openai": LLMBackend.OPENAI,
                   "checkpoint": LLMBackend.CHECKPOINT}
    backend = backend_map.get(os.environ.get("LLM_BACKEND"), LLMBackend.CHECKPOINT)
    fallback = os.environ.get("CHECKPOINT_FALLBACK", "openai")

    print(f"Phase 4b prompt absorption experiment | backend={backend.value} "
          f"| fallback={fallback}\n")

    # Cache per-dataset linker + components + sent_map
    ds_state: dict = {}
    def get_ds(ds):
        if ds not in ds_state:
            text_p, model_p = DATASETS_PATHS[ds]
            components = parse_pcm_repository(str(model_p))
            sentences = load_sentences(str(text_p))
            sent_map = build_sent_map(sentences)
            linker = SLinker17f(
                backend=backend,
                checkpoint_fallback=fallback,
                checkpoint_fallback_model=os.environ.get("CHECKPOINT_FALLBACK_MODEL", "gpt-5.4"),
            )
            layer1 = load_pkl(CACHE/ds/"layer1.pkl")
            layer3 = load_pkl(CACHE/ds/"layer3.pkl")
            linker.model_knowledge = layer1["model_knowledge"]
            linker.doc_knowledge = layer1["doc_knowledge"]
            ds_state[ds] = (linker, components, sent_map, layer3)
        return ds_state[ds]

    # Resolve test cases to candidate objects
    resolved_cases = []
    for (ds, snum, cid, expected, label) in TEST_CASES:
        linker, components, sent_map, layer3 = get_ds(ds)
        # Find candidate matching (snum, cid) — cid may be None, then match snum + first
        cand = None
        for c in layer3["candidates"]:
            if c.sentence_number == snum and (cid is None or c.component_id == cid):
                cand = c; break
        if cand is None:
            print(f"  ! could not resolve {ds} S{snum} {cid} — skipping")
            continue
        resolved_cases.append((ds, cand, expected, label, linker, components, sent_map))

    # Run all variants on all candidates
    results: dict = {}
    for variant_name, (vrules, p1f, p2f) in PROMPT_VARIANTS.items():
        print(f"\n{'='*80}\n  variant: {variant_name}\n{'='*80}")
        results[variant_name] = []
        for (ds, cand, expected, label, linker, components, sent_map) in resolved_cases:
            comp_names = get_comp_names(components)
            case_text = build_case_text(cand, sent_map, linker)
            t0 = time.time()
            p1, p2 = run_one_twopass(linker, case_text, comp_names, vrules, p1f, p2f, variant_name)
            approved = p1 and p2
            verdict = "approve" if approved else "reject"
            match = "✓" if verdict == expected else "✗"
            print(f"  {match} [{ds:<13} {label:<40}] p1={p1} p2={p2} → {verdict}"
                  f" (expected {expected}) ({round(time.time()-t0,1)}s)")
            results[variant_name].append({
                "dataset": ds, "snum": cand.sentence_number,
                "component": cand.component_name, "matched_text": cand.matched_text,
                "label": label, "expected": expected,
                "p1": p1, "p2": p2, "approved": approved,
                "verdict": verdict, "correct": verdict == expected,
            })

    # ── Summary: which variants get the most correct? ──────────────────────
    print(f"\n\n{'='*100}")
    print("SUMMARY: per-variant correctness on 3 code-path FPs and 4 control TPs")
    print(f"{'='*100}")
    print(f"{'Variant':<18} {'Correct (of 7)':>15} {'FPs rejected':>14} {'TPs preserved':>15}")
    print("-" * 70)
    for v, recs in results.items():
        correct = sum(1 for r in recs if r["correct"])
        fp_rejected = sum(1 for r in recs if r["expected"] == "reject" and r["verdict"] == "reject")
        fp_total = sum(1 for r in recs if r["expected"] == "reject")
        tp_kept = sum(1 for r in recs if r["expected"] == "approve" and r["verdict"] == "approve")
        tp_total = sum(1 for r in recs if r["expected"] == "approve")
        print(f"{v:<18} {correct}/7{'':<13}  {fp_rejected}/{fp_total}{'':<12} {tp_kept}/{tp_total}{'':<13}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / f"results/4b_prompt_absorption_{ts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults: {out}")


if __name__ == "__main__":
    main()
