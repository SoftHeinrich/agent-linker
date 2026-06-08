"""Build the REAL Phase 4 / Phase 5 validation prompts s_linker19 would send,
once with the current truncations and once with them removed, and report the
delta per batch / per pass / per project.

Uses the existing phase cache (results/phase_cache/s_linker19/<backend>/<proj>/)
so no LLM calls happen and no candidates are re-derived — the prompts that get
rebuilt here are byte-identical to what the linker actually sent.

Char-cap sites covered:
  L654 prev.text[:80]   inside _format_evidence (evidence stanza)
  L658 anchor[:100]     inside _format_evidence (anchor lines)
  L675 prev.text[:60]   _validate_with_evidence inline lead-in
  L862 prev.text[:60]   _validate_coref_links inline lead-in

Usage:
  python3 scripts/check_truncations.py                       # claude cache
  python3 scripts/check_truncations.py --backend openai
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map  # noqa: E402
from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names  # noqa: E402
from llm_sad_sam.linkers.experimental.prompts_v5 import (  # noqa: E402
    P1_FOCUS, P2_FOCUS, VALIDATION_RULES, COREF_VALIDATION_FOCUS,
)
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402

BENCHMARK_BASE = ROOT / "../ardoco/core/tests-base/src/main/resources/benchmark"
CACHE_BASE = ROOT / "results/phase_cache/s_linker19"

DATASETS = {
    "mediastore":    ("mediastore/text_2016/mediastore.txt",
                      "mediastore/model_2016/pcm/ms.repository"),
    "teastore":      ("teastore/text_2020/teastore.txt",
                      "teastore/model_2020/pcm/teastore.repository"),
    "teammates":     ("teammates/text_2021/teammates.txt",
                      "teammates/model_2021/pcm/teammates.repository"),
    "bigbluebutton": ("bigbluebutton/text_2021/bigbluebutton.txt",
                      "bigbluebutton/model_2021/pcm/bbb.repository"),
    "jabref":        ("jabref/text_2021/jabref.txt",
                      "jabref/model_2021/pcm/jabref.repository"),
}

BATCH_SIZE = 25  # matches s_linker19.py L670 / L851


# ─────────────────────────────────────────────────────────────────────────
# Verbatim reproductions of the s_linker19 prompt fragments
# (parameterized by a `caps` dict so we can flip 60→∞, 80→∞, 100→∞)
# ─────────────────────────────────────────────────────────────────────────

def format_evidence(bundle: dict, caps: dict) -> str:
    """Mirror s_linker19._format_evidence (L647-L659)."""
    lines = [
        f"  Evidence: source={bundle['source']}, "
        f"span=\"{bundle['matched_span']}\", "
        f"mention={bundle['mention_type']}, "
        f"ambiguous={bundle['is_ambiguous']}",
        f"  Rationale: {bundle['extraction_rationale']}",
    ]
    if bundle["preceding_text"]:
        lines.append(f"  [prev: \"{bundle['preceding_text'][:caps['L654']]}\"]")
    if bundle["anchor_sentences"]:
        lines.append("  Anchors (confirmed refs):")
        for a in bundle["anchor_sentences"][:3]:
            lines.append(f"    {a[:caps['L658']]}")
    return "\n".join(lines)


def build_entity_case(cand, bundle, sent_map, caps) -> str:
    """Mirror s_linker19._validate_with_evidence case loop (L673-L683)."""
    prev = sent_map.get(cand.sentence_number - 1)
    p = f"[prev: {prev.text[:caps['L675']]}] " if prev else ""
    evidence_block = format_evidence(bundle, caps) if bundle else ""
    return (
        f'Case {{i}}: "{cand.matched_text}" -> {cand.component_name}\n'
        f'  {p}"{cand.sentence_text}"\n'
        f'{evidence_block}'
    )


def build_coref_case(lk, sent_map, caps) -> str | None:
    """Mirror s_linker19._validate_coref_links case loop (L854-L867)."""
    sent = sent_map.get(lk.sentence_number)
    if not sent:
        return None
    prev = sent_map.get(lk.sentence_number - 1)
    p = f"[prev: {prev.text[:caps['L862']]}] " if prev else ""
    return (
        f'Case {{i}}: pronoun/role-ref -> {lk.component_name}\n'
        f'  {p}"{sent.text}"'
    )


def build_validation_prompt(comp_names: list[str], cases: list[str],
                            focus: str) -> str:
    """Mirror s_linker19._run_validation_pass (L704-L715)."""
    return (
        f"Validate component references in a software architecture document. "
        f"{focus}\n\n"
        f"COMPONENTS: {', '.join(comp_names)}\n\n"
        f"{VALIDATION_RULES}\n\n"
        f"CASES:\n"
        f"{chr(10).join(cases)}\n\n"
        f"Return JSON:\n"
        f"{{\"validations\": [{{\"case\": 1, \"approve\": true}}]}}\n"
        f"JSON only:"
    )


# ─────────────────────────────────────────────────────────────────────────

CAPS_REAL = {"L654": 80,           "L658": 100,           "L675": 60,           "L862": 60}
CAPS_OPEN = {"L654": 10**9,        "L658": 10**9,         "L675": 10**9,        "L862": 10**9}


def measure_project(project: str, backend: str) -> dict:
    text_rel, model_rel = DATASETS[project]
    text_path = BENCHMARK_BASE / text_rel
    model_path = BENCHMARK_BASE / model_rel
    cache_dir = CACHE_BASE / backend / project

    if not (cache_dir / "layer3.pkl").exists():
        return {"project": project, "missing": str(cache_dir)}

    sentences = load_sentences(str(text_path))
    sent_map = build_sent_map(sentences)
    components = parse_pcm_repository(str(model_path))
    comp_names = get_comp_names(components)

    with open(cache_dir / "layer3.pkl", "rb") as f:
        layer3 = pickle.load(f)
    candidates = layer3["candidates"]
    bundles = layer3["evidence_bundles"]

    coref_raw: list = []
    if (cache_dir / "layer4.pkl").exists():
        with open(cache_dir / "layer4.pkl", "rb") as f:
            coref_raw = pickle.load(f)["coref_raw"]

    # Phase 4 (entity twopass)
    p4_real = {"prompt_bytes": 0, "batches": 0, "per_batch_real": [],
               "per_batch_open": []}
    p4_open = {"prompt_bytes": 0}
    for batch_start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[batch_start:batch_start + BATCH_SIZE]
        cases_real, cases_open = [], []
        for i, c in enumerate(batch):
            bundle = bundles.get((c.sentence_number, c.component_id))
            cases_real.append(
                build_entity_case(c, bundle, sent_map, CAPS_REAL)
                .replace("{i}", str(i + 1)))
            cases_open.append(
                build_entity_case(c, bundle, sent_map, CAPS_OPEN)
                .replace("{i}", str(i + 1)))
        # P1 and P2 prompts (twopass = ×2)
        for focus in (P1_FOCUS, P2_FOCUS):
            real = build_validation_prompt(comp_names, cases_real, focus)
            opn = build_validation_prompt(comp_names, cases_open, focus)
            p4_real["prompt_bytes"] += len(real)
            p4_open["prompt_bytes"] += len(opn)
            p4_real["per_batch_real"].append(len(real))
            p4_real["per_batch_open"].append(len(opn))
        p4_real["batches"] += 1

    # Phase 5 (coref single-pass)
    p5_real = {"prompt_bytes": 0, "batches": 0}
    p5_open = {"prompt_bytes": 0}
    for batch_start in range(0, len(coref_raw), BATCH_SIZE):
        batch = coref_raw[batch_start:batch_start + BATCH_SIZE]
        cases_real, cases_open = [], []
        for i, lk in enumerate(batch):
            r = build_coref_case(lk, sent_map, CAPS_REAL)
            o = build_coref_case(lk, sent_map, CAPS_OPEN)
            if r is None or o is None:
                continue
            cases_real.append(r.replace("{i}", str(len(cases_real) + 1)))
            cases_open.append(o.replace("{i}", str(len(cases_open) + 1)))
        if not cases_real:
            continue
        real = build_validation_prompt(comp_names, cases_real,
                                       COREF_VALIDATION_FOCUS)
        opn = build_validation_prompt(comp_names, cases_open,
                                      COREF_VALIDATION_FOCUS)
        p5_real["prompt_bytes"] += len(real)
        p5_open["prompt_bytes"] += len(opn)
        p5_real["batches"] += 1

    return {
        "project": project,
        "n_candidates": len(candidates),
        "n_coref_raw": len(coref_raw),
        "p4_batches": p4_real["batches"],
        "p4_real_total_B": p4_real["prompt_bytes"],
        "p4_open_total_B": p4_open["prompt_bytes"],
        "p4_per_pass_real": p4_real["per_batch_real"],
        "p4_per_pass_open": p4_real["per_batch_open"],
        "p5_batches": p5_real["batches"],
        "p5_real_total_B": p5_real["prompt_bytes"],
        "p5_open_total_B": p5_open["prompt_bytes"],
    }


def fmt_kb(n: int) -> str:
    return f"{n/1024:7.1f} KB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="claude", choices=["claude", "openai"])
    args = ap.parse_args()

    results = []
    for project in DATASETS:
        r = measure_project(project, args.backend)
        results.append(r)

    print(f"\nbackend = {args.backend}\n")
    print(f"{'project':<14} {'cand':>5} {'p4batches':>10} "
          f"{'p4 real':>11} {'p4 uncap':>11} {'p4 Δ':>9} {'p4 Δ%':>7}  "
          f"{'p5 real':>10} {'p5 uncap':>10} {'p5 Δ':>9} {'p5 Δ%':>7}")

    tot_p4_real = tot_p4_open = tot_p5_real = tot_p5_open = 0
    for r in results:
        if "missing" in r:
            print(f"{r['project']:<14} -- no cache at {r['missing']}")
            continue
        p4d = r["p4_open_total_B"] - r["p4_real_total_B"]
        p5d = r["p5_open_total_B"] - r["p5_real_total_B"]
        p4pct = 100.0 * p4d / max(1, r["p4_real_total_B"])
        p5pct = 100.0 * p5d / max(1, r["p5_real_total_B"])
        print(f"{r['project']:<14} {r['n_candidates']:>5} {r['p4_batches']:>10} "
              f"{fmt_kb(r['p4_real_total_B']):>11} "
              f"{fmt_kb(r['p4_open_total_B']):>11} "
              f"{fmt_kb(p4d):>9} {p4pct:>6.1f}%  "
              f"{fmt_kb(r['p5_real_total_B']):>10} "
              f"{fmt_kb(r['p5_open_total_B']):>10} "
              f"{fmt_kb(p5d):>9} {p5pct:>6.1f}%")
        tot_p4_real += r["p4_real_total_B"]
        tot_p4_open += r["p4_open_total_B"]
        tot_p5_real += r["p5_real_total_B"]
        tot_p5_open += r["p5_open_total_B"]

    p4d = tot_p4_open - tot_p4_real
    p5d = tot_p5_open - tot_p5_real
    print("-" * 130)
    print(f"{'TOTAL':<14} {'':>5} {'':>10} "
          f"{fmt_kb(tot_p4_real):>11} {fmt_kb(tot_p4_open):>11} "
          f"{fmt_kb(p4d):>9} {100.0*p4d/max(1,tot_p4_real):>6.1f}%  "
          f"{fmt_kb(tot_p5_real):>10} {fmt_kb(tot_p5_open):>10} "
          f"{fmt_kb(p5d):>9} {100.0*p5d/max(1,tot_p5_real):>6.1f}%")

    # Per-batch stats — worst single prompt that would be sent
    print("\nPer-batch (single LLM call) sizes — Phase 4 only:")
    print(f"{'project':<14} {'max real':>10} {'max uncap':>10} "
          f"{'max Δ':>9} {'mean real':>10} {'mean uncap':>11}")
    for r in results:
        if "missing" in r:
            continue
        reals = r["p4_per_pass_real"]
        opens = r["p4_per_pass_open"]
        if not reals:
            continue
        max_r, max_o = max(reals), max(opens)
        mean_r = sum(reals) // len(reals)
        mean_o = sum(opens) // len(opens)
        print(f"{r['project']:<14} {fmt_kb(max_r):>10} {fmt_kb(max_o):>10} "
              f"{fmt_kb(max_o - max_r):>9} "
              f"{fmt_kb(mean_r):>10} {fmt_kb(mean_o):>11}")


if __name__ == "__main__":
    main()
