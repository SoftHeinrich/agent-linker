"""Dotted-path → SE-terminology rename, empirical tuning.

The s_linker19 paper variant carries one piece of ad-hoc vocabulary in its
prompts: the word "dotted-path". It appears in three load-bearing places:

  prompts_v5.P1_FOCUS                 — trailing clause "and not just as a
                                        dotted-path identifier?" (the +7-word
                                        V3 modification that absorbs Phase 4b)
  prompts_v5.ALIAS_SCOPE_RULES        — "Dotted-path fragments (tokens of the
                                        form X.Y or X.Y.Z) are NOT aliases"
  s_linker19.MentionType.CODE_TOKEN   — evidence-bundle label "lowercase,
                                        inside dotted path" (embedded into
                                        P1's prompt via _format_evidence)

The reviewer-facing concern: "dotted-path" is colloquial, not the textbook SE
term. The standard term is *qualified name* / *qualified identifier* — the
formal label used in the Java Language Specification, Python data model, and
most architecture textbooks for X.Y.Z-style package- or member-access paths.

The empirical concern: the dotted-path clause was tuned (4b absorption
experiment, V3: +7 words) and demonstrably catches 2/3 code-path FPs with 0
collateral damage. Swapping the noun phrase must NOT regress that probe.

This script holds prompts_v5.VALIDATION_RULES and P2_FOCUS fixed and varies
ONLY the trailing clause of P1_FOCUS and the matching CODE_TOKEN label. Both
strings flow into the same Phase-4 P1 prompt, so they are tested as a unit.

Test set (cached from 17f openai run — same as experiment_4b):
  Code-path FPs (must REJECT, not in gold):
    S22  teammates "logic"             in "logic, ui.website, ui.controller represent..."
    S130 teammates "storage"           in "Package overview contains storage.api, storage.entity, ..."
    S159 teammates "common.datatransfer" in "common.datatransfer contains data transfer objects."
  Control TPs (must APPROVE, in gold):
    S25  mediastore "Database component"   → DB
    S31  mediastore "Database component"   → DB
    S69  bigbluebutton "KMS"               → kurento
    S21  bigbluebutton "bbb-html5"         → HTML5 Server

Variants (clause appended to P1, paired with CODE_TOKEN label):
  V0_no_clause            : P1 baseline (no dotted clause) — sanity floor;
                            expected to leak 2-3 FPs like the pre-V3 17f did.
  V1_dotted_current       : production wording — "...and not just as a
                            dotted-path identifier?" + "lowercase, inside
                            dotted path". This is what s19 ships today.
  V2_qualified_full       : "...and not just as a qualified-name identifier
                            (e.g. a package- or member-access path X.Y.Z)?"
                            + "lowercase, inside qualified name".
  V3_qualified_short      : "...and not just as a qualified-name identifier?"
                            + "lowercase, inside qualified name".
  V4_qualified_hybrid     : keeps the cue "dotted" parenthetically — "...and
                            not just as a qualified-name (dotted) identifier?"
                            + "lowercase, inside qualified (dotted) name".
  V5_package_member_path  : "...and not just as a package- or member-access
                            path?" + "lowercase, inside member-access path".

Pass criterion (parity with V1):
  - rejects ≥ 2 of 3 code-path FPs    AND
  - approves all 4 control TPs

Cost: 7 cases × 6 variants × 1 P1 pass = 42 P1 calls. P2 is byte-identical
across variants (uses prompts_v5.P2_FOCUS unchanged), so its 7 calls are
cache-hit-friendly. With LLM_BACKEND=checkpoint + openai fallback the
baseline V1 and pre-V3 V0 hit cache from the 4b run; only V2–V5 incur new
LLM cost (~28 fresh calls on gpt-5.4).

Reports:
  per-variant per-candidate: p1 / p2 / approved / matches expected
  summary table: correctness count, FPs rejected, TPs preserved
  JSON dump under results/dotted_path_rename_YYYYMMDD_HHMMSS.json
"""
from __future__ import annotations

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
# Prompt-variant fixtures
#
# Each variant supplies (1) the trailing clause appended to P1_FOCUS and
# (2) the mention_type label used for CODE_TOKEN-classified candidates in
# the evidence bundle. Both end up in the same P1 prompt.
# ─────────────────────────────────────────────────────────────────────────────

P1_FOCUS_HEAD = (
    "Check architectural participation: does the sentence name this component "
    "as an architectural participant — performing operations, providing "
    "services, or taking part in the described system behavior"
)


def p1_focus(tail_clause: str) -> str:
    """Build P1_FOCUS with the given trailing clause (or none)."""
    if not tail_clause:
        return P1_FOCUS_HEAD + "?"
    return f"{P1_FOCUS_HEAD}, {tail_clause}?"


# (clause appended after "behavior, ", CODE_TOKEN label)
PROMPT_VARIANTS: dict[str, tuple[str, str]] = {
    "V0_no_clause": (
        "",
        "lowercase, inside dotted path",  # CODE_TOKEN label unchanged for sanity floor
    ),
    "V1_dotted_current": (
        "and not just as a dotted-path identifier",
        "lowercase, inside dotted path",
    ),
    "V2_qualified_full": (
        "and not just as a qualified-name identifier (e.g. a package- or member-access path X.Y.Z)",
        "lowercase, inside qualified name",
    ),
    "V3_qualified_short": (
        "and not just as a qualified-name identifier",
        "lowercase, inside qualified name",
    ),
    "V4_qualified_hybrid": (
        "and not just as a qualified-name (dotted) identifier",
        "lowercase, inside qualified (dotted) name",
    ),
    "V5_package_member_path": (
        "and not just as a package- or member-access path",
        "lowercase, inside member-access path",
    ),
}


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def run_one_pass(linker, case_text, comp_names, focus, validation_rules, phase_label):
    """One validation pass with overridable focus. Returns bool (approved)."""
    linker.llm.set_phase(phase_label)
    prompt = f"""Validate component references in a software architecture document. {focus}

COMPONENTS: {', '.join(comp_names)}

{validation_rules}

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


def build_case_text(c, sent_map, linker, code_token_label: str) -> str:
    """Build the Case-block string. If the candidate's mention_type would
    classify as CODE_TOKEN (the relevant branch for the dotted-path clause),
    override the bundle's mention_type with the variant's label so the
    variant's vocabulary lands in the prompt.
    """
    prev = sent_map.get(c.sentence_number - 1)
    p = f"[prev: {prev.text[:60]}] " if prev else ""
    bundle = linker._build_evidence_bundle(c, sent_map)
    # _build_evidence_bundle classified the mention; only override when the
    # original classification was CODE_TOKEN (i.e. inside a dotted path) —
    # otherwise the variant label would be inaccurate for the case.
    code_token_default = "lowercase, inside dotted path"
    if bundle.mention_type == code_token_default:
        bundle.mention_type = code_token_label
    return (
        f'Case 1: "{c.matched_text}" -> {c.component_name}\n'
        f'  {p}"{c.sentence_text}"\n'
        f'{linker._format_evidence(bundle)}'
    )


def main():
    os.environ.setdefault("LLM_BACKEND", "checkpoint")
    from llm_sad_sam.llm_client import LLMBackend
    from llm_sad_sam.linkers.experimental.s_linker19 import SLinker19
    from llm_sad_sam.linkers.experimental.prompts_v5 import (
        VALIDATION_RULES,
        P2_FOCUS,
    )
    from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
    from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
    from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names

    backend_map = {
        "claude": LLMBackend.CLAUDE,
        "openai": LLMBackend.OPENAI,
        "checkpoint": LLMBackend.CHECKPOINT,
    }
    backend = backend_map.get(os.environ.get("LLM_BACKEND"), LLMBackend.CHECKPOINT)
    fallback = os.environ.get("CHECKPOINT_FALLBACK", "openai")

    print(
        f"Dotted-path rename empirical tuning | backend={backend.value} "
        f"| fallback={fallback}"
    )
    print(f"Variants: {', '.join(PROMPT_VARIANTS)}\n")

    # Cache per-dataset linker + components + sent_map
    ds_state: dict = {}

    def get_ds(ds):
        if ds not in ds_state:
            text_p, model_p = DATASETS_PATHS[ds]
            components = parse_pcm_repository(str(model_p))
            sentences = load_sentences(str(text_p))
            sent_map = build_sent_map(sentences)
            linker = SLinker19(
                backend=backend,
                checkpoint_fallback=fallback,
                checkpoint_fallback_model=os.environ.get(
                    "CHECKPOINT_FALLBACK_MODEL", "gpt-5.4"
                ),
            )
            layer1 = load_pkl(CACHE / ds / "layer1.pkl")
            layer3 = load_pkl(CACHE / ds / "layer3.pkl")
            linker.model_knowledge = layer1["model_knowledge"]
            linker.doc_knowledge = layer1["doc_knowledge"]
            ds_state[ds] = (linker, components, sent_map, layer3)
        return ds_state[ds]

    # Resolve test cases to candidate objects
    resolved_cases = []
    for (ds, snum, cid, expected, label) in TEST_CASES:
        linker, components, sent_map, layer3 = get_ds(ds)
        cand = None
        for c in layer3["candidates"]:
            if c.sentence_number == snum and (cid is None or c.component_id == cid):
                cand = c
                break
        if cand is None:
            print(f"  ! could not resolve {ds} S{snum} {cid} — skipping")
            continue
        resolved_cases.append((ds, cand, expected, label, linker, components, sent_map))

    # Run all variants on all candidates.
    # P2 prompt is invariant across variants → run it once per case and cache.
    p2_cache: dict[tuple, bool] = {}
    results: dict = {}
    for variant_name, (tail_clause, code_token_label) in PROMPT_VARIANTS.items():
        print(f"\n{'='*80}\n  variant: {variant_name}\n{'='*80}")
        print(f"  P1 tail : {tail_clause or '(no tail clause — baseline)'}")
        print(f"  CODE_TOKEN: {code_token_label}")
        results[variant_name] = []
        focus_p1 = p1_focus(tail_clause)
        for (ds, cand, expected, label, linker, components, sent_map) in resolved_cases:
            comp_names = get_comp_names(components)
            case_text = build_case_text(cand, sent_map, linker, code_token_label)
            t0 = time.time()
            p1 = run_one_pass(
                linker, case_text, comp_names, focus_p1, VALIDATION_RULES,
                f"dotted_rename_{variant_name}_p1",
            )
            # P2 is invariant — reuse cached verdict if we've seen this case.
            cache_key = (ds, cand.sentence_number, cand.component_id)
            if cache_key in p2_cache:
                p2 = p2_cache[cache_key]
            else:
                p2 = run_one_pass(
                    linker, case_text, comp_names, P2_FOCUS, VALIDATION_RULES,
                    f"dotted_rename_p2_{ds}_s{cand.sentence_number}",
                )
                p2_cache[cache_key] = p2
            approved = p1 and p2
            verdict = "approve" if approved else "reject"
            match = "OK" if verdict == expected else "MISS"
            print(
                f"  [{match:<4}] {ds:<13} {label:<40} "
                f"p1={p1} p2={p2} → {verdict:<7} (expected {expected}) "
                f"({round(time.time() - t0, 1)}s)"
            )
            results[variant_name].append({
                "dataset": ds,
                "snum": cand.sentence_number,
                "component": cand.component_name,
                "matched_text": cand.matched_text,
                "label": label,
                "expected": expected,
                "p1": p1,
                "p2": p2,
                "approved": approved,
                "verdict": verdict,
                "correct": verdict == expected,
            })

    # ── Summary: which variants get the most correct? ──────────────────────
    print(f"\n\n{'='*100}")
    print("SUMMARY — dotted-path rename probe (V1 = current production)")
    print(f"{'='*100}")
    print(f"{'Variant':<26} {'Correct (of 7)':>15} {'FPs rejected':>14} {'TPs preserved':>15}")
    print("-" * 75)
    for v, recs in results.items():
        correct = sum(1 for r in recs if r["correct"])
        fp_rejected = sum(
            1 for r in recs if r["expected"] == "reject" and r["verdict"] == "reject"
        )
        fp_total = sum(1 for r in recs if r["expected"] == "reject")
        tp_kept = sum(
            1 for r in recs if r["expected"] == "approve" and r["verdict"] == "approve"
        )
        tp_total = sum(1 for r in recs if r["expected"] == "approve")
        print(
            f"{v:<26} {correct}/7{'':<13}  "
            f"{fp_rejected}/{fp_total}{'':<12} "
            f"{tp_kept}/{tp_total}{'':<13}"
        )

    # Pass-criterion summary: parity with V1 (current production).
    print(f"\n{'='*100}")
    print("Pass criterion: FPs rejected ≥ 2/3 AND TPs preserved = 4/4")
    print(f"{'='*100}")
    v1_recs = results.get("V1_dotted_current", [])
    v1_fp = sum(1 for r in v1_recs if r["expected"] == "reject" and r["verdict"] == "reject")
    v1_tp = sum(1 for r in v1_recs if r["expected"] == "approve" and r["verdict"] == "approve")
    print(f"V1 (production) reference: FP={v1_fp}/3, TP={v1_tp}/4")
    for v, recs in results.items():
        if v == "V1_dotted_current":
            continue
        fp = sum(1 for r in recs if r["expected"] == "reject" and r["verdict"] == "reject")
        tp = sum(1 for r in recs if r["expected"] == "approve" and r["verdict"] == "approve")
        passes = (fp >= 2) and (tp == 4)
        verdict = "PASS — parity with V1" if passes else "FAIL — regresses vs V1 floor"
        print(f"  {v:<26} FP={fp}/3 TP={tp}/4  → {verdict}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"dotted_path_rename_{ts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults: {out}")


if __name__ == "__main__":
    main()
