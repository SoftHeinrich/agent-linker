#!/usr/bin/env python3
"""EXT-01 Plan 06-09 Probe P2 — Hybrid regex + LLM-rejection (BBB only).

Per CONTEXT.md D-13:
  Default = the existing `_has_standalone_mention` regex. After a True
  verdict, an LLM rejection prompt asks "given this sentence and
  ?1-sentence context, does the mention really refer to the focal
  component as a system entity (keep) or is it a generic / dotted /
  embedded use that the regex falsely accepted (drop)?"

Budget: warn but do not abort at 200 LLM calls; record final count.
Approve-bias: malformed -> "keep" (matches s_linker13 conventions).

Per D-11 / BENCHMARK_TABOO.md: prompt examples are from safe SE textbook
domains only.
"""

from __future__ import annotations

import json
import os
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


BENCHMARK_BASE = (REPO_ROOT / "../ardoco/core/tests-base/src/main/resources/benchmark").resolve()
BBB = {
    "text": BENCHMARK_BASE / "bigbluebutton/text_2021/bigbluebutton.txt",
    "model": BENCHMARK_BASE / "bigbluebutton/model_2021/pcm/bbb.repository",
    "gold_sam": BENCHMARK_BASE / "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
}

PROBE_NAME = "ext01_probe_p2"
RESULTS_DIR = REPO_ROOT / "results/ablation_results/ext01_probes"
PHASE_CACHE_DIR = REPO_ROOT / "results/phase_cache" / PROBE_NAME
os.environ["PHASE_CACHE_DIR"] = str(PHASE_CACHE_DIR)

P2_BUDGET_WARN = 200

PROMPT_TEMPLATE = """STANDALONE-MENTION RECHECK (single sentence).

A regex baseline has flagged the FOCAL component name as a standalone
mention in the TARGET sentence below. Your job is to RECHECK whether
the regex was right: does the TARGET really refer to FOCAL as a
specific architectural component, or did the regex mis-fire?

Approve as standalone (keep) when ALL hold:
- The word in the target sentence really names the focal component
  as a system entity (not as ordinary English, not as a fragment of
  a longer identifier that names something else).
- The mention is grammatically attributable to the focal component
  on its own (a stand-alone reference, not a buried sub-token).

Reject (drop) when ANY hold:
- The word appears inside a longer dotted/qualified identifier and
  the sentence is enumerating code-level structures.
- The word is used with its ordinary English meaning (generic verb
  / noun / adjective), no architectural intent.
- The word appears only as part of a longer proper name that names
  a different entity.

Abstract patterns (safe SE textbook domains, not from any benchmark):
- FOCAL=Parser, TARGET="The parser hands tokens to the AST builder."
  -> keep (architectural participant).
- FOCAL=Parser, TARGET="parser.tokens are stored in compiler/util.py"
  -> drop (inside dotted code identifier).
- FOCAL=Scheduler, TARGET="The system schedules tasks fairly."
  -> drop (generic English verb use, not the named entity).

FOCAL: "{focal}"

CONTEXT (the target sentence and immediate neighbours, target prefixed
">>>"):
{context_block}

Return JSON:
{{"verdict": "keep" or "drop", "reason": "brief"}}
JSON only:"""


def build_text_to_snum(sentences):
    return {s.text: s.number for s in sentences}


def call_p2_judge(client: LLMClient, focal: str, snum: int, sent_map, text_to_snum):
    """One LLM rejection call. Returns ("keep"|"drop", success_bool, attempts)."""
    parts = []
    for delta in (-1, 0, 1):
        s = sent_map.get(snum + delta)
        if not s:
            continue
        marker = ">>>" if delta == 0 else "   "
        parts.append(f"{marker} S{s.number}: {s.text}")
    context_block = "\n".join(parts)

    prompt = PROMPT_TEMPLATE.format(focal=focal, context_block=context_block)

    attempts = 0
    for attempt in range(2):
        attempts += 1
        data = client.extract_json(client.query(prompt, timeout=120))
        if data and "verdict" in data:
            verdict = str(data.get("verdict", "")).strip().lower()
            if verdict not in ("keep", "drop"):
                verdict = "keep"  # approve-bias
            return verdict, True, attempts
        if attempt == 0:
            print(f"    [P2/{focal}/S{snum}] empty response, retrying ...")
    # Approve-bias on parse failure.
    return "keep", False, attempts


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

    for k, p in BBB.items():
        if not Path(p).exists():
            raise FileNotFoundError(f"BBB {k} missing: {p}")

    print(f"[P2] Probe start. Phase cache: {PHASE_CACHE_DIR}")
    components = parse_pcm_repository(str(BBB["model"]))
    sentences = load_sentences(str(BBB["text"]))
    sent_map = build_sent_map(sentences)
    text_to_snum = build_text_to_snum(sentences)
    gold_pairs = load_gold_pairs(str(BBB["gold_sam"]))
    print(f"[P2] components={len(components)} sentences={len(sentences)} "
          f"gold_pairs={len(gold_pairs)}")

    client = LLMClient(backend=LLMBackend.CLAUDE)

    original_has_standalone = SLinker13._has_standalone_mention

    # Per-(comp, sentence_text) cache so the six call sites do not
    # multiply LLM cost during one sweep.
    rejection_cache: dict[tuple[str, str], bool] = {}
    counters = {
        "judge_calls": 0,
        "judge_attempts": 0,
        "judge_failures": 0,
        "regex_true_count": 0,
        "llm_drops": 0,
        "fallback_no_snum": 0,
    }
    judge_latency = {"s": 0.0}
    drop_log = []  # list of (comp, snum, verdict, reason)

    def patched_has_standalone_mention(comp_name, text):
        if not original_has_standalone(comp_name, text):
            return False
        counters["regex_true_count"] += 1
        key = (comp_name, text)
        if key in rejection_cache:
            return rejection_cache[key]
        snum = text_to_snum.get(text)
        if snum is None:
            counters["fallback_no_snum"] += 1
            rejection_cache[key] = True
            return True
        tc = time.time()
        verdict, success, attempts = call_p2_judge(
            client, comp_name, snum, sent_map, text_to_snum
        )
        judge_latency["s"] += time.time() - tc
        counters["judge_calls"] += 1
        counters["judge_attempts"] += attempts
        if not success:
            counters["judge_failures"] += 1
        if verdict == "drop":
            counters["llm_drops"] += 1
            drop_log.append({"comp": comp_name, "snum": snum})
            rejection_cache[key] = False
            return False
        rejection_cache[key] = True
        if counters["judge_calls"] >= P2_BUDGET_WARN \
                and counters["judge_calls"] % 25 == 0:
            print(f"    [P2] judge_calls={counters['judge_calls']} "
                  f"(over budget warn {P2_BUDGET_WARN}); continuing.")
        return True

    SLinker13._has_standalone_mention = staticmethod(patched_has_standalone_mention)

    try:
        print(f"[P2] Running patched SLinker13 sweep on BBB ...")
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
    print(f"[P2] BBB sweep: P={metrics['P']:.4f} R={metrics['R']:.4f} "
          f"F1={metrics['F1']:.4f} TP={metrics['tp']} FP={metrics['fp']} "
          f"FN={metrics['fn']} time={sweep_time:.0f}s "
          f"calls={counters['judge_calls']} drops={counters['llm_drops']}")

    PARENT_F1 = 0.8990
    PURE_LLM_F1 = 0.8108

    # FN-recovery + TP-lost / FP-killed breakdowns.
    name_to_id = {c.name: c.id for c in components}
    drop_pairs = {(d["snum"], name_to_id.get(d["comp"], d["comp"]))
                  for d in drop_log}
    tps_lost = len(drop_pairs & gold_pairs)
    fps_killed = len(drop_pairs - gold_pairs)

    baseline_fn_path = RESULTS_DIR / "baseline_fn_set.json"
    fns_recovered = -1
    baseline_fn_count = -1
    fn_pairs = gold_pairs - predicted_pairs
    if baseline_fn_path.exists():
        bblob = json.loads(baseline_fn_path.read_text())
        baseline_fns = {(int(s), c) for s, c in bblob.get("fn_set", [])}
        baseline_fn_count = bblob.get("fn_count", len(baseline_fns))
        fns_recovered = len(baseline_fns - fn_pairs)
        print(f"[P2] Loaded baseline FN set "
              f"({baseline_fn_count} entries)")
    else:
        print("[P2] No baseline_fn_set.json yet; Task 4 will reconcile.")

    baseline_fp_path = RESULTS_DIR / "baseline_fp_set.json"
    new_fps = -1
    if baseline_fp_path.exists():
        bblob = json.loads(baseline_fp_path.read_text())
        baseline_fps = {(int(s), c) for s, c in bblob.get("fp_set", [])}
        fp_pairs = predicted_pairs - gold_pairs
        new_fps = len(fp_pairs - baseline_fps)

    projected_full_sweep_calls = int(
        counters["judge_calls"] * 5 * 1.3
    )  # rough: BBB-call count -> 5 datasets x retry overhead

    result_blob = {
        "probe": "P2",
        "dataset": "bigbluebutton",
        "bbb_f1": metrics["F1"],
        "bbb_precision": metrics["P"],
        "bbb_recall": metrics["R"],
        "bbb_tp": metrics["tp"],
        "bbb_fp": metrics["fp"],
        "bbb_fn": metrics["fn"],
        "delta_vs_parent_s_linker13": round(metrics["F1"] - PARENT_F1, 4),
        "delta_vs_pure_llm_floor": round(metrics["F1"] - PURE_LLM_F1, 4),
        "llm_calls": counters["judge_calls"],
        "llm_attempts": counters["judge_attempts"],
        "json_parse_failures": counters["judge_failures"],
        "total_latency_s": round(judge_latency["s"] + sweep_time, 1),
        "judge_latency_s": round(judge_latency["s"], 1),
        "sweep_latency_s": round(sweep_time, 1),
        "projected_full_sweep_calls": projected_full_sweep_calls,
        "fns_recovered_of_17": fns_recovered,
        "baseline_fn_count": baseline_fn_count,
        "new_fps_introduced": new_fps,
        "fallback_no_snum_hits": counters["fallback_no_snum"],
        "regex_true_count": counters["regex_true_count"],
        "llm_drops": counters["llm_drops"],
        "tps_lost_to_llm_drop": tps_lost,
        "fps_killed_by_llm_drop": fps_killed,
        "generality_cost": (
            "KEEPS the regex baseline as a cheap pre-filter; LLM is "
            "only used to reject false-positives. From an EXT-01 "
            "purity standpoint, the structural rule is not removed — "
            "it is demoted to a pre-filter. Requires reframing EXT-01."
        ),
        "implementation_cost_estimate": (
            "Low: wrap existing _has_standalone_mention with cache + "
            "rejection call. ~0.5–1 day to productionize."
        ),
        "drop_log": drop_log,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_s": round(time.time() - t0, 1),
    }

    out_path = RESULTS_DIR / "p2_bbb.json"
    out_path.write_text(json.dumps(result_blob, indent=2))
    print(f"[P2] Wrote {out_path}")
    print(f"[P2] DONE. F1={metrics['F1']:.4f} TPs-lost={tps_lost} "
          f"FPs-killed={fps_killed} calls={counters['judge_calls']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
