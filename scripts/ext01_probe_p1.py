#!/usr/bin/env python3
"""EXT-01 Plan 06-09 Probe P1 — Document-level full-context judge (BBB only).

Per CONTEXT.md D-12/D-13:
  Run ONE LLM call per BBB component. The call sees the full document
  (numbered sentences), the alias map for that component, and the
  remaining component table. It returns the list of sentence numbers
  in which the focal component is referenced as a standalone mention
  (i.e. a sentence the regex `_has_standalone_mention` would have
  approved IF it had project-aware knowledge of aliases).

The probe then monkeypatches `SLinker13._has_standalone_mention` with
a lookup against the precomputed (comp, snum) -> True map, runs the
full s_linker13 pipeline on BBB, and reports F1 + FN-recovery.

Budget: <=24 LLM calls (12 components x up to 2 attempts).

Per D-11 / BENCHMARK_TABOO.md: prompt examples are from safe SE textbook
domains only (compiler / OS / e-commerce). The alias map fed at runtime
is LLM-discovered project data, NOT hardcoded benchmark surface forms.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_dotenv() -> None:
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


load_dotenv()
os.environ.setdefault("CLAUDE_MODEL", "sonnet")

from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map  # noqa: E402
from llm_sad_sam.llm_client import LLMBackend, LLMClient  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker13 import SLinker13  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402


# Dataset registry — BBB only, paths inlined per plan (self-contained).
BENCHMARK_BASE = (REPO_ROOT / "../ardoco/core/tests-base/src/main/resources/benchmark").resolve()
BBB = {
    "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
    "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
    "gold_sam": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
}

PROBE_NAME = "ext01_probe_p1"
RESULTS_DIR = REPO_ROOT / "results/ablation_results/ext01_probes"
PHASE_CACHE_DIR = REPO_ROOT / "results/phase_cache" / PROBE_NAME
# Probe-private checkpoint area (T-06-09-02 mitigation).
# The patched run will write under PHASE_CACHE_DIR/s_linker13/<dataset>/.
os.environ["PHASE_CACHE_DIR"] = str(PHASE_CACHE_DIR)


PROMPT_TEMPLATE = """STANDALONE-MENTION JUDGE (document-level).

You are given a software architecture document and asked to identify
every sentence in which a FOCAL component is mentioned in a way that
the documentation treats as a direct, standalone reference to that
component.

A sentence MENTIONS the focal component when ANY of these hold:
- The sentence uses the focal component's canonical name verbatim
  (any case) AND that occurrence is not buried inside a longer
  qualified identifier (e.g. a dotted package path or a hyphenated
  sub-name of a different entity).
- The sentence uses one of the component's KNOWN ALIASES (provided
  below) to refer to it.
- The sentence uses a definite/bare phrasing whose immediate textual
  context unambiguously refers to this focal component (e.g. a
  paragraph already named it and the next sentence picks it up).

A sentence does NOT mention the focal component when:
- The component's name appears only as a fragment of a longer
  identifier that names a DIFFERENT entity.
- The component's name appears only as ordinary English vocabulary
  (a generic word), with no architectural intent.
- The sentence enumerates many other components and the focal one
  is only present as background.

Abstract patterns (safe SE textbook domains, not from any benchmark):
- Focal = "Parser"; sentence "The lexer feeds tokens to the parser."
  -> standalone mention (canonical name, architectural use).
- Focal = "Scheduler"; sentence "scheduler.queue.push(task) in the
  kernel module." -> NOT a mention (name is inside a dotted code path).
- Focal = "ShoppingCart"; sentence "Items added to the cart are
  persisted." with KNOWN ALIASES ["cart"] -> standalone mention
  (alias reference).
- Focal = "InvoiceHandler"; sentence "The system handles invoices in
  batches." -> NOT a mention ("invoices" is generic, not the named
  component).

FOCAL COMPONENT: "{focal}"

KNOWN ALIASES for the focal component (LLM-discovered from this
document at runtime; may be empty):
{aliases_block}

OTHER COMPONENTS in the system (do not return their sentences;
listed only so you can disambiguate):
{other_components}

DOCUMENT (sentences are 1-indexed; one sentence per line):
{document}

Return JSON with the sentence numbers that mention the FOCAL
component as defined above:

{{"standalone_sentences": [N_INTEGER, ...], "rationale": "brief"}}

Return only sentence numbers that appear in the document above
(1..{max_snum}).  JSON only:"""


def build_doc_knowledge(client: LLMClient, sentences, components):
    """Acquire BBB's doc_knowledge.aliases.

    Cleanest route: instantiate SLinker13 and call its existing
    `_learn_document_knowledge_enriched` method directly. This is
    Tier-1's doc-knowledge step in isolation, so it does not run
    seed / model-analysis / validation.
    """
    linker = SLinker13(backend=LLMBackend.CLAUDE)
    # Replace the live llm with the shared client so we count its calls.
    linker.llm = client
    print(f"  [P1] Acquiring doc_knowledge for BBB via "
          f"SLinker13._learn_document_knowledge_enriched ...")
    doc_knowledge = linker._learn_document_knowledge_enriched(sentences, components)
    print(f"  [P1] doc_knowledge.aliases: "
          f"{len(doc_knowledge.aliases)} entries")
    return doc_knowledge


def call_p1_judge(client: LLMClient, focal: str, aliases_for_focal,
                  other_comp_names, sentences):
    """One LLM call: judge which sentences mention `focal`.

    Returns (set_of_snums, parse_failed_bool, n_attempts).
    Approve-bias OFF: parse failure -> empty set (P1 is testing whether
    the LLM can pick the right sentences; failure is the conservative
    read).
    """
    if aliases_for_focal:
        aliases_block = "\n".join(
            f'  - "{a}" [scope={scope}]' for a, scope in aliases_for_focal
        )
    else:
        aliases_block = "  (none)"

    other_block = ", ".join(f'"{n}"' for n in sorted(other_comp_names))
    document_block = "\n".join(f"S{s.number}: {s.text}" for s in sentences)
    max_snum = max(s.number for s in sentences)

    prompt = PROMPT_TEMPLATE.format(
        focal=focal,
        aliases_block=aliases_block,
        other_components=other_block,
        document=document_block,
        max_snum=max_snum,
    )

    attempts = 0
    for attempt in range(2):
        attempts += 1
        data = client.extract_json(client.query(prompt, timeout=300))
        if data and "standalone_sentences" in data:
            raw = data.get("standalone_sentences") or []
            snums = set()
            for item in raw:
                # Accept "S42", "s42", "42", 42
                if isinstance(item, str):
                    item = item.lstrip("Ss")
                try:
                    snum = int(item)
                except (ValueError, TypeError):
                    continue
                if 1 <= snum <= max_snum:
                    snums.add(snum)
            return snums, False, attempts
        if attempt == 0:
            print(f"    [P1/{focal}] empty response, retrying ...")

    # Approve-bias OFF: parse failure -> empty (conservative)
    return set(), True, attempts


def load_gold_pairs(gold_path: str):
    import csv
    pairs = set()
    with open(gold_path) as f:
        for row in csv.DictReader(f):
            cid = row.get("modelElementID", "").strip()
            snum = row.get("sentence", "").strip()
            if cid and snum:
                pairs.add((int(snum), cid))
    return pairs


def eval_metrics(predicted, gold):
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "P": precision, "R": recall, "F1": f1}


def main() -> int:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Validate dataset paths.
    for k, p in BBB.items():
        if not Path(p).exists():
            raise FileNotFoundError(f"BBB {k} missing: {p}")

    print(f"[P1] Probe start. Phase cache: {PHASE_CACHE_DIR}")
    components = parse_pcm_repository(str(BBB["model"]))
    sentences = load_sentences(str(BBB["text"]))
    sent_map = build_sent_map(sentences)
    gold_pairs = load_gold_pairs(str(BBB["gold_sam"]))
    name_to_id = {c.name: c.id for c in components}
    print(f"[P1] components={len(components)} sentences={len(sentences)} "
          f"gold_pairs={len(gold_pairs)}")

    client = LLMClient(backend=LLMBackend.CLAUDE)

    # Acquire BBB doc_knowledge.aliases (one call internally; not counted
    # in the per-component budget below — this is the alias-discovery
    # baseline P1 prompts assume).
    doc_knowledge = build_doc_knowledge(client, sentences, components)
    aliases_by_component: dict[str, list[tuple[str, str]]] = {}
    for term, entry in doc_knowledge.aliases.items():
        aliases_by_component.setdefault(entry.component, []).append(
            (term, entry.scope)
        )

    # Per-component P1 call loop (the budgeted calls).
    standalone_map: dict[tuple[str, int], bool] = {}
    per_component_log = []
    judge_calls = 0
    parse_failures = 0
    judge_latency = 0.0

    for comp in components:
        other_names = [c.name for c in components if c.name != comp.name]
        a4c = aliases_by_component.get(comp.name, [])
        tc = time.time()
        snums, failed, attempts = call_p1_judge(
            client, comp.name, a4c, other_names, sentences
        )
        latency = time.time() - tc
        judge_latency += latency
        judge_calls += attempts
        if failed:
            parse_failures += 1
        for snum in snums:
            standalone_map[(comp.name, snum)] = True
        per_component_log.append({
            "component": comp.name,
            "aliases": [a for a, _ in a4c],
            "snums": sorted(snums),
            "n_snums": len(snums),
            "parse_failed": failed,
            "attempts": attempts,
            "latency_s": round(latency, 2),
        })
        print(f"    [P1/{comp.name}] snums={len(snums)} "
              f"attempts={attempts} latency={latency:.1f}s "
              f"failed={failed}")

    print(f"[P1] Judge calls done: {judge_calls} total, "
          f"{parse_failures} parse failures.")

    # Build text->snum reverse map for the monkeypatch.
    text_to_snum: dict[str, int] = {s.text: s.number for s in sentences}

    # Counter for fallback-regex usage.
    fallback_hits = {"n": 0}
    original_has_standalone = SLinker13._has_standalone_mention

    def patched_has_standalone_mention(comp_name, text):  # staticmethod shape
        snum = text_to_snum.get(text)
        if snum is None:
            fallback_hits["n"] += 1
            return original_has_standalone(comp_name, text)
        return standalone_map.get((comp_name, snum), False)

    SLinker13._has_standalone_mention = staticmethod(patched_has_standalone_mention)

    try:
        # Now run the full pipeline on BBB with the patched gate.
        print(f"[P1] Running patched SLinker13 sweep on BBB ...")
        linker = SLinker13(backend=LLMBackend.CLAUDE)
        t_sweep = time.time()
        predictions = linker.link(
            text_path=str(BBB["text"]),
            model_path=str(BBB["model"]),
        )
        sweep_time = time.time() - t_sweep
    finally:
        SLinker13._has_standalone_mention = staticmethod(original_has_standalone)

    predicted_pairs = {(p.sentence_number, p.component_id) for p in predictions}
    metrics = eval_metrics(predicted_pairs, gold_pairs)
    print(f"[P1] BBB sweep: P={metrics['P']:.4f} R={metrics['R']:.4f} "
          f"F1={metrics['F1']:.4f} TP={metrics['tp']} FP={metrics['fp']} "
          f"FN={metrics['fn']} time={sweep_time:.0f}s")

    # Reference baselines (constants from CONTEXT.md interfaces block).
    PARENT_F1 = 0.8990
    PURE_LLM_F1 = 0.8108

    # FN-recovery: compute against a baseline FN set we will compute or
    # reuse a sibling file from Tasks 2/3.
    baseline_fn_path = RESULTS_DIR / "baseline_fn_set.json"
    if baseline_fn_path.exists():
        baseline_blob = json.loads(baseline_fn_path.read_text())
        baseline_fns = {(tuple(it) if isinstance(it, list) else it)
                        for it in baseline_blob.get("fn_set", [])}
        baseline_fns = {(int(snum), cid) for snum, cid in baseline_fns}
        baseline_fn_count = baseline_blob.get("fn_count", len(baseline_fns))
        print(f"[P1] Loaded baseline FN set ({baseline_fn_count} entries) "
              f"from {baseline_fn_path}")
    else:
        # No sibling yet — we cannot compute FN-recovery here against an
        # absent baseline. Report -1 and let Task 4 reconcile via the
        # baseline file Task 3 writes.
        baseline_fns = None
        baseline_fn_count = -1
        print(f"[P1] No baseline_fn_set.json yet — fns_recovered will be -1 "
              f"(Task 4 will reconcile).")

    fn_pairs = gold_pairs - predicted_pairs
    if baseline_fns is not None:
        fns_recovered = len(baseline_fns - fn_pairs)
    else:
        fns_recovered = -1

    # New FPs introduced vs baseline (need baseline_fp_set or skip).
    new_fps = -1
    baseline_fp_path = RESULTS_DIR / "baseline_fp_set.json"
    if baseline_fp_path.exists():
        bblob = json.loads(baseline_fp_path.read_text())
        baseline_fps = {(int(s), c) for s, c in bblob.get("fp_set", [])}
        fp_pairs = predicted_pairs - gold_pairs
        new_fps = len(fp_pairs - baseline_fps)

    # Projected full-sweep call count (rough): 12 calls/BBB scaled by
    # the component count of each dataset (weighted total).
    # BBB has 12 components. Full sweep estimate uses mean components/dataset
    # times 5 datasets times ~1.2 retry overhead.
    projected_full_sweep_calls = int(round(
        # use the BBB count as a proxy; will be refined when other datasets sweep
        (judge_calls / max(1, len(components))) * 5 * len(components)
        * 1.2
    ))

    result_blob = {
        "probe": "P1",
        "dataset": "bigbluebutton",
        "bbb_f1": metrics["F1"],
        "bbb_precision": metrics["P"],
        "bbb_recall": metrics["R"],
        "bbb_tp": metrics["tp"],
        "bbb_fp": metrics["fp"],
        "bbb_fn": metrics["fn"],
        "delta_vs_parent_s_linker13": round(metrics["F1"] - PARENT_F1, 4),
        "delta_vs_pure_llm_floor": round(metrics["F1"] - PURE_LLM_F1, 4),
        "llm_calls": judge_calls,
        "total_latency_s": round(judge_latency + sweep_time, 1),
        "judge_latency_s": round(judge_latency, 1),
        "sweep_latency_s": round(sweep_time, 1),
        "projected_full_sweep_calls": projected_full_sweep_calls,
        "fns_recovered_of_17": fns_recovered,
        "baseline_fn_count": baseline_fn_count,
        "new_fps_introduced": new_fps,
        "fallback_regex_hits": fallback_hits["n"],
        "json_parse_failures": parse_failures,
        "generality_cost": (
            "Document-level LLM judge with no regex. Encodes the "
            "standalone-mention rule as natural-language prompt text "
            "(safe SE textbook examples only). Depends at runtime on "
            "LLM-discovered aliases (project-agnostic, computed by "
            "doc_knowledge phase)."
        ),
        "implementation_cost_estimate": (
            "Low–moderate: replace _has_standalone_mention with a "
            "pre-pipeline call to compute the standalone_map; six "
            "call sites become dict lookups. ~1–2 days to productionize "
            "incl. error handling + caching."
        ),
        "per_component_log": per_component_log,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_s": round(time.time() - t0, 1),
    }

    out_path = RESULTS_DIR / "p1_bbb.json"
    out_path.write_text(json.dumps(result_blob, indent=2))
    print(f"[P1] Wrote {out_path}")
    print(f"[P1] DONE. F1={metrics['F1']:.4f} parent_delta={result_blob['delta_vs_parent_s_linker13']:+.4f} "
          f"calls={judge_calls} fallback={fallback_hits['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
