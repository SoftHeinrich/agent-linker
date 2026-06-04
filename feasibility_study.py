"""Feasibility study for unified s_linker18 architecture.

Tests each proposed cleanup EMPIRICALLY using existing 17f openai phase cache
+ minimal fresh LLM calls. No commitment to the rewrite until results say go.

Tests:
  B — antecedent_via_alias bypass:
      Count coref candidates that survive ONLY via the via_alias bypass.
      If those are mostly TPs, the bypass is doing real work and can't be
      dropped without an alternative. If mostly FPs / dropped at validation,
      it's a hot-fix.
      Cost: zero LLM calls (pure cache analysis).

  C — drop Phase 4b:
      Take candidates that Phase 4b killed (dotted-path FPs). Build new
      evidence bundles with mention_type='code_token' explicit. Re-run
      twopass (p1+p2). If twopass rejects them too, Phase 4b can be
      absorbed into twopass with a stronger signal.
      Cost: 2 LLM calls per killed candidate × ~3 candidates = ~6 calls.

  E — unify coref validation with entity twopass:
      Take coref_raw candidates. Build entity-style evidence bundle.
      Run entity twopass (p1+p2) on them. Compare approve/reject vs
      current single-pass coref validation. If outcomes agree, the
      asymmetric coref validator can be merged into twopass.
      Cost: 2 LLM calls per dataset × 5 = ~10 calls.

  F — drop generic-filter, route via twopass with is_ambiguous bundle field:
      Take candidates that went through generic-filter. Build evidence
      bundle with is_ambiguous=True explicit. Run twopass directly.
      Compare to generic-filter+twopass outcome.
      Cost: 2 LLM calls per dataset × 5 = ~10 calls.

Total expected cost: ~$0.50 on gpt-5.4 flex.

Usage:
    cd approach/
    LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.4 OPENAI_SERVICE_TIER=flex \\
        python feasibility_study.py
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

ROOT = Path(__file__).parent
BENCH = ROOT / "../ardoco/core/tests-base/src/main/resources/benchmark"
CACHE = ROOT / "results/phase_cache/s_linker17f/openai"

DATASETS = {
    "mediastore":    (BENCH/"mediastore/text_2016/mediastore.txt",
                      BENCH/"mediastore/model_2016/pcm/ms.repository",
                      BENCH/"mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"),
    "teastore":      (BENCH/"teastore/text_2020/teastore.txt",
                      BENCH/"teastore/model_2020/pcm/teastore.repository",
                      BENCH/"teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"),
    "teammates":     (BENCH/"teammates/text_2021/teammates.txt",
                      BENCH/"teammates/model_2021/pcm/teammates.repository",
                      BENCH/"teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "bigbluebutton": (BENCH/"bigbluebutton/text_2021/bigbluebutton.txt",
                      BENCH/"bigbluebutton/model_2021/pcm/bbb.repository",
                      BENCH/"bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "jabref":        (BENCH/"jabref/text_2021/jabref.txt",
                      BENCH/"jabref/model_2021/pcm/jabref.repository",
                      BENCH/"jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
}


def load_gold(p):
    g = set()
    for r in csv.DictReader(open(p)):
        cid = r.get("modelElementID", "").strip()
        sn = r.get("sentence", "").strip()
        if cid and sn:
            g.add((int(sn), cid))
    return g


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def setup_linker(text_path, model_path, backend):
    """Instantiate SLinker17f and prime its knowledge from cached layer1.

    Avoids re-running Phase 1 (Phase 1 LLM calls already cached in the run
    that produced phase_cache/s_linker17f/openai/).
    """
    from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
    from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
    from llm_sad_sam.linkers.experimental.s_linker17f import SLinker17f

    components = parse_pcm_repository(str(model_path))
    sentences = load_sentences(str(text_path))
    sent_map = build_sent_map(sentences)
    linker = SLinker17f(backend=backend)
    return linker, components, sentences, sent_map


def prime_knowledge(linker, layer1_pkl):
    linker.model_knowledge = layer1_pkl["model_knowledge"]
    linker.doc_knowledge = layer1_pkl["doc_knowledge"]


# ─────────────────────────────────────────────────────────────────────────────
# Experiment B — antecedent_via_alias bypass usage
# ─────────────────────────────────────────────────────────────────────────────

def exp_B(ds, gold, layer4, sent_map):
    """Count coref candidates surviving ONLY via the via_alias bypass.

    A coref link is kept iff (standalone_mention(comp, antecedent_text) OR
    antecedent_via_alias). The bypass is doing real work only when the first
    condition is False.
    """
    from llm_sad_sam.linkers.experimental.helper_v3 import has_standalone_mention

    coref_raw = layer4["coref_raw"]
    coref_meta = layer4["coref_metadata"]
    coref_decisions = layer4["coref_decisions"]

    bypass_total = 0
    bypass_validated = 0
    bypass_tp = 0
    bypass_fp = 0
    for lk in coref_raw:
        key = (lk.sentence_number, lk.component_id)
        meta = coref_meta.get(key, {})
        if not meta.get("antecedent_via_alias"):
            continue
        ant_text = meta.get("antecedent_text", "")
        if has_standalone_mention(lk.component_name, ant_text):
            # Bypass not needed — standalone match also fires.
            continue
        # Pure-bypass case
        bypass_total += 1
        if coref_decisions.get(key, {}).get("approved"):
            bypass_validated += 1
            if key in gold:
                bypass_tp += 1
            else:
                bypass_fp += 1

    return {
        "coref_raw": len(coref_raw),
        "bypass_only": bypass_total,
        "bypass_validated": bypass_validated,
        "bypass_validated_tp": bypass_tp,
        "bypass_validated_fp": bypass_fp,
        "verdict": "GO" if bypass_tp == 0 else ("CAUTION" if bypass_tp <= 1 else "NO-GO"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment C — drop Phase 4b: re-validate killed via twopass with code_token signal
# ─────────────────────────────────────────────────────────────────────────────

def exp_C(ds, linker, components, sent_map, layer3):
    """For each Phase-4b-killed candidate, build new bundle marking it as
    code_token and re-run twopass. If twopass rejects, Phase 4b is absorbable."""
    from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names
    from llm_sad_sam.linkers.experimental.s_linker17f import EvidenceBundle

    p4b = layer3["phase_4b_decisions"]
    killed = [(k, v) for k, v in p4b.items() if v.get("dropped")]
    if not killed:
        return {"killed_count": 0, "verdict": "GO (nothing to absorb)"}

    candidates = {(c.sentence_number, c.component_id): c
                  for c in layer3["candidates"]}
    comp_names = get_comp_names(components)

    rejected_by_twopass = 0
    approved_by_twopass = 0
    details = []
    for (key, dec) in killed:
        c = candidates.get(key)
        if not c:
            continue
        # Build evidence bundle the normal way, then override mention_type
        bundle = linker._build_evidence_bundle(
            c, sent_map, rationale="re-test for cleanup C: drop Phase 4b")
        # Strong signal: this candidate is a code token (Phase 4b's detection)
        bundle = EvidenceBundle(
            source=bundle.source,
            matched_span=bundle.matched_span,
            mention_type="code_token (component name appears only as dotted-path segment)",
            preceding_text=bundle.preceding_text,
            anchor_sentences=bundle.anchor_sentences,
            is_ambiguous=bundle.is_ambiguous,
            extraction_rationale=bundle.extraction_rationale,
        )
        prev = sent_map.get(c.sentence_number - 1)
        p_prev = f"[prev: {prev.text[:60]}] " if prev else ""
        case_text = (
            f'Case 1: "{c.matched_text}" -> {c.component_name}\n'
            f'  {p_prev}"{c.sentence_text}"\n'
            f'{linker._format_evidence(bundle)}'
        )
        r1 = linker._run_validation_pass(
            comp_names, [case_text],
            "Check architectural participation: does the sentence name this component as an architectural participant — performing operations, providing services, or taking part in the described system behavior?",
            phase_tag="feasibility_C_p1",
        )
        r2 = linker._run_validation_pass(
            comp_names, [case_text],
            "Check referential specificity: is the component name used to identify this specific architectural element, or does it serve as a generic technical term in this sentence?",
            phase_tag="feasibility_C_p2",
        )
        approved = r1.get(0, False) and r2.get(0, False)
        if approved:
            approved_by_twopass += 1
        else:
            rejected_by_twopass += 1
        details.append({
            "key": list(key), "component": c.component_name,
            "matched_text": c.matched_text,
            "twopass_p1": r1.get(0, False),
            "twopass_p2": r2.get(0, False),
            "approved": approved,
            "phase_4b_reason": dec.get("reason", "")[:80],
        })

    return {
        "killed_count": len(killed),
        "twopass_rejected": rejected_by_twopass,
        "twopass_approved": approved_by_twopass,
        "details": details,
        "verdict": "GO" if approved_by_twopass == 0 else
                   ("CAUTION" if approved_by_twopass <= rejected_by_twopass else "NO-GO"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment E — unify coref validation with entity twopass
# ─────────────────────────────────────────────────────────────────────────────

def exp_E(ds, linker, components, sent_map, layer4, gold):
    """Run entity twopass on coref_raw candidates. Compare vs single-pass."""
    from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names
    from llm_sad_sam.core.data_types_v2 import CandidateLink
    from llm_sad_sam.linkers.experimental.s_linker17f import EvidenceBundle

    coref_raw = layer4["coref_raw"]
    coref_meta = layer4["coref_metadata"]
    coref_decisions = layer4["coref_decisions"]
    if not coref_raw:
        return {"n_coref": 0, "verdict": "GO (no coref to compare)"}

    comp_names = get_comp_names(components)

    # Build a CandidateLink for each coref link so we can use _build_evidence_bundle
    candidates = []
    for lk in coref_raw:
        sent = sent_map.get(lk.sentence_number)
        if not sent:
            continue
        meta = coref_meta.get((lk.sentence_number, lk.component_id), {})
        ref_text = meta.get("reference", "")
        candidates.append(CandidateLink(
            lk.sentence_number, sent.text, lk.component_name, lk.component_id,
            ref_text or lk.component_name, source="coref_for_test",
            mention_type="anaphoric",
        ))

    # Build cases and call twopass
    cases = []
    for i, c in enumerate(candidates):
        bundle = linker._build_evidence_bundle(c, sent_map, rationale="coref reference")
        bundle = EvidenceBundle(
            source="coref",
            matched_span=c.matched_text,
            mention_type="anaphoric reference (pronoun or role-ref) — antecedent in prior context",
            preceding_text=bundle.preceding_text,
            anchor_sentences=bundle.anchor_sentences,
            is_ambiguous=bundle.is_ambiguous,
            extraction_rationale="coref resolution to a prior antecedent",
        )
        prev = sent_map.get(c.sentence_number - 1)
        p_prev = f"[prev: {prev.text[:60]}] " if prev else ""
        cases.append((
            f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n'
            f'  {p_prev}"{c.sentence_text}"\n'
            f'{linker._format_evidence(bundle)}',
            c,
        ))

    case_strings = [ct for ct, _ in cases]
    r1 = linker._run_validation_pass(
        comp_names, case_strings,
        "Check architectural participation: does the sentence name this component as an architectural participant — performing operations, providing services, or taking part in the described system behavior?",
        phase_tag="feasibility_E_p1",
    )
    r2 = linker._run_validation_pass(
        comp_names, case_strings,
        "Check referential specificity: is the component name used to identify this specific architectural element, or does it serve as a generic technical term in this sentence?",
        phase_tag="feasibility_E_p2",
    )

    agree = 0
    twopass_approves_extra = 0
    twopass_rejects_extra = 0
    twopass_tp = 0
    twopass_fp = 0
    single_tp = 0
    single_fp = 0
    for i, (_, c) in enumerate(cases):
        key = (c.sentence_number, c.component_id)
        tp_app = r1.get(i, False) and r2.get(i, False)
        sp_app = coref_decisions.get(key, {}).get("approved", False)
        if tp_app == sp_app:
            agree += 1
        elif tp_app and not sp_app:
            twopass_approves_extra += 1
        elif sp_app and not tp_app:
            twopass_rejects_extra += 1
        if tp_app:
            if key in gold: twopass_tp += 1
            else: twopass_fp += 1
        if sp_app:
            if key in gold: single_tp += 1
            else: single_fp += 1

    delta_tp = twopass_tp - single_tp
    delta_fp = twopass_fp - single_fp
    return {
        "n_coref": len(cases),
        "agreement": agree,
        "twopass_approves_extra": twopass_approves_extra,
        "twopass_rejects_extra": twopass_rejects_extra,
        "single_pass_tp_fp": [single_tp, single_fp],
        "twopass_tp_fp": [twopass_tp, twopass_fp],
        "delta_tp_fp": [delta_tp, delta_fp],
        "verdict": "GO" if (delta_tp >= 0 and delta_fp <= 0) else
                   ("CAUTION" if delta_tp >= -1 and delta_fp <= 1 else "NO-GO"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment F — drop generic-filter, route via twopass with is_ambiguous bundle
# ─────────────────────────────────────────────────────────────────────────────

def exp_F(ds, linker, components, sent_map, layer3, gold):
    """For candidates that went through generic-filter, re-run twopass directly
    (skipping generic-filter) with is_ambiguous=True signal in the bundle."""
    from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names
    from llm_sad_sam.linkers.experimental.s_linker17f import EvidenceBundle

    p4_generic = layer3["phase_4_generic_decisions"]
    if not p4_generic:
        return {"n_generic": 0, "verdict": "GO (nothing went through generic-filter)"}

    candidates = {(c.sentence_number, c.component_id): c
                  for c in layer3["candidates"]}
    comp_names = get_comp_names(components)

    cases = []
    for key in p4_generic:
        c = candidates.get(key)
        if not c:
            continue
        bundle = linker._build_evidence_bundle(c, sent_map, rationale="generic-filter test")
        bundle = EvidenceBundle(
            source=bundle.source,
            matched_span=bundle.matched_span,
            mention_type=bundle.mention_type,
            preceding_text=bundle.preceding_text,
            anchor_sentences=bundle.anchor_sentences,
            is_ambiguous=True,  # explicit signal
            extraction_rationale=bundle.extraction_rationale,
        )
        prev = sent_map.get(c.sentence_number - 1)
        p_prev = f"[prev: {prev.text[:60]}] " if prev else ""
        cases.append((
            f'Case {len(cases)+1}: "{c.matched_text}" -> {c.component_name}\n'
            f'  {p_prev}"{c.sentence_text}"\n'
            f'{linker._format_evidence(bundle)}',
            c, key,
        ))

    if not cases:
        return {"n_generic": 0, "verdict": "GO (no overlap with candidates)"}

    case_strings = [c for c, _, _ in cases]
    r1 = linker._run_validation_pass(
        comp_names, case_strings,
        "Check architectural participation: does the sentence name this component as an architectural participant — performing operations, providing services, or taking part in the described system behavior?",
        phase_tag="feasibility_F_p1",
    )
    r2 = linker._run_validation_pass(
        comp_names, case_strings,
        "Check referential specificity: is the component name used to identify this specific architectural element, or does it serve as a generic technical term in this sentence?",
        phase_tag="feasibility_F_p2",
    )

    agree = 0
    tp_extra_approves = 0
    tp_extra_rejects = 0
    new_tp = 0
    new_fp = 0
    cur_tp = 0
    cur_fp = 0
    for i, (_, c, key) in enumerate(cases):
        new_app = r1.get(i, False) and r2.get(i, False)
        cur_app = p4_generic.get(key, {}).get("approved", False)
        if new_app == cur_app:
            agree += 1
        elif new_app and not cur_app:
            tp_extra_approves += 1
        elif cur_app and not new_app:
            tp_extra_rejects += 1
        if new_app:
            if key in gold: new_tp += 1
            else: new_fp += 1
        if cur_app:
            if key in gold: cur_tp += 1
            else: cur_fp += 1

    return {
        "n_generic": len(cases),
        "agreement": agree,
        "twopass_approves_extra": tp_extra_approves,
        "twopass_rejects_extra": tp_extra_rejects,
        "current_tp_fp": [cur_tp, cur_fp],
        "twopass_only_tp_fp": [new_tp, new_fp],
        "delta_tp_fp": [new_tp - cur_tp, new_fp - cur_fp],
        "verdict": "GO" if (new_tp >= cur_tp and new_fp <= cur_fp) else
                   ("CAUTION" if abs(new_tp - cur_tp) <= 1 and abs(new_fp - cur_fp) <= 1 else "NO-GO"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from llm_sad_sam.llm_client import LLMBackend
    backend_map = {
        "claude": LLMBackend.CLAUDE, "openai": LLMBackend.OPENAI,
        "checkpoint": LLMBackend.CHECKPOINT,
    }
    backend = backend_map.get(os.environ.get("LLM_BACKEND", "openai"), LLMBackend.OPENAI)
    print(f"Feasibility study | backend={backend.value}\n")

    all_results = []
    for ds, (text_path, model_path, gold_path) in DATASETS.items():
        print(f"\n{'='*70}\n  {ds}\n{'='*70}")
        if not (CACHE/ds/"layer1.pkl").exists():
            print(f"  SKIP: no cache at {CACHE/ds}")
            continue

        gold = load_gold(gold_path)
        layer1 = load_pkl(CACHE/ds/"layer1.pkl")
        layer3 = load_pkl(CACHE/ds/"layer3.pkl")
        layer4 = load_pkl(CACHE/ds/"layer4.pkl")

        linker, components, sentences, sent_map = setup_linker(
            text_path, model_path, backend)
        prime_knowledge(linker, layer1)

        result = {"dataset": ds, "gold": len(gold)}

        print(f"\n  [B] antecedent_via_alias bypass usage")
        result["B"] = exp_B(ds, gold, layer4, sent_map)
        print(f"    {result['B']}")

        print(f"\n  [C] drop Phase 4b, re-validate via twopass with code_token signal")
        result["C"] = exp_C(ds, linker, components, sent_map, layer3)
        print(f"    {result['C']}")

        print(f"\n  [E] unify coref validation with entity twopass")
        result["E"] = exp_E(ds, linker, components, sent_map, layer4, gold)
        print(f"    {result['E']}")

        print(f"\n  [F] drop generic-filter, route via twopass with is_ambiguous bundle")
        result["F"] = exp_F(ds, linker, components, sent_map, layer3, gold)
        print(f"    {result['F']}")

        all_results.append(result)

    # ── Aggregate verdict per experiment ────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("AGGREGATE FEASIBILITY VERDICTS")
    print(f"{'='*90}")
    print(f"{'Dataset':<14} {'B (bypass)':>14} {'C (drop 4b)':>14} "
          f"{'E (unify coref)':>16} {'F (drop generic)':>17}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['dataset']:<14} {r['B']['verdict']:>14} {r['C']['verdict']:>14} "
              f"{r['E']['verdict']:>16} {r['F']['verdict']:>17}")
    print()

    # Save full results
    import time
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / f"results/feasibility_study_{ts}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Full results: {out}")


if __name__ == "__main__":
    main()
