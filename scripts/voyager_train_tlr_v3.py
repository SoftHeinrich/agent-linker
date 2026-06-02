"""Voyager-TLR pilot v3 — Claude Sonnet, 3 train/test splits, fresh-start,
full intermediate logging.

This is the v2.2-prep Claude Sonnet companion to ``scripts/voyager_train_tlr_v2.py``.
The v2 script (gpt-5.4) is preserved untouched. v3 is a near-identical sibling
that swaps backend defaults to Claude Sonnet so v2 (gpt-5.4) vs v3 (Claude)
become a direct, controlled comparison on the same 3 splits.

DIFFERENCES vs v2
-----------------
1. Default backend is Claude (``--backend claude --model sonnet``).
2. Output tree is ``results/voyager_pilot_v3_claude/`` (sibling of v2 tree).
3. Env-var prefix is ``VOYAGER3_*`` (independent budget knobs from v2).
4. Methodology (3 splits, fresh-start, intermediate dumps, GATE-06 regex,
   convergence threshold 0.90, dedupe floor 0.6) is BIT-EXACT identical to
   v2 so any delta is attributable to model, not loop changes.

CLI
---
    python scripts/voyager_train_tlr_v3.py train  --split-id 1 --fresh-start --save-intermediate
    python scripts/voyager_train_tlr_v3.py distill --split-id 1
    python scripts/voyager_train_tlr_v3.py test    --split-id 1
    python scripts/voyager_train_tlr_v3.py all     --split-id 1 --fresh-start --save-intermediate

After all three splits complete, run:
    python scripts/voyager_train_tlr_v3.py rollup

to generate ``results/voyager_pilot_v3_claude/crossplit_comparison.{json,pkl}``.

GATE-06
-------
Identical taboo regex to v1/v2 (mirrors BENCHMARK_TABOO.md). Every learned
pattern grepped before being committed to the bank; per-pattern critic call
during distillation.

BUDGET
------
$80 Claude Sonnet cap. Tracked via per-split LLM call count. Wallclock cap
4h. Env overrides: VOYAGER3_MAX_CALLS, VOYAGER3_WALL_BUDGET_S.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Repository roots
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from llm_sad_sam.llm_client import LLMBackend, LLMClient
from llm_sad_sam.linkers.experimental.s_linker13_skill_learned_clean import (
    SLinker13SkillLearned,
    PROMPT_CONSTANT_NAMES,
)
import run_ablation as _ra


# ───────────────────────────────────────────────────────────────────────────
# Split definitions (per v2.2-prep user directive)
# ───────────────────────────────────────────────────────────────────────────

SPLITS: dict[int, dict] = {
    1: {
        "name": "split1_replication",
        "train": ["mediastore", "teastore", "teammates"],
        "test": ["bigbluebutton", "jabref"],
        "description": "Replication of prior Phase 12 pilot split.",
    },
    2: {
        "name": "split2_bbb_in_train",
        "train": ["mediastore", "teastore", "bigbluebutton"],
        "test": ["teammates", "jabref"],
        "description": "BBB-in-train acid test — train on hardest dataset.",
    },
    3: {
        "name": "split3_rotated_holdout",
        "train": ["teastore", "teammates", "jabref"],
        "test": ["mediastore", "bigbluebutton"],
        "description": "Rotated hold-out — different mix to test split-stability.",
    },
}


# ───────────────────────────────────────────────────────────────────────────
# Tunables
# ───────────────────────────────────────────────────────────────────────────

MAX_OUTER_PASSES = int(os.environ.get("VOYAGER3_MAX_OUTER", "3"))
MAX_INNER_ITERS = int(os.environ.get("VOYAGER3_MAX_INNER", "3"))
CONVERGENCE_THRESHOLD = float(os.environ.get("VOYAGER3_CONV_THRESH", "0.90"))
PER_PROJECT_EARLY_STOP = 0.95
MAX_PATTERNS_PER_CALL = 3
DEDUP_SIM_FLOOR = 0.6
DISTILL_TARGET_COUNT = 8

MAX_CALLS = int(os.environ.get("VOYAGER3_MAX_CALLS", "200"))
WALL_BUDGET_SECONDS = int(os.environ.get("VOYAGER3_WALL_BUDGET_S", str(4 * 3600)))


# ───────────────────────────────────────────────────────────────────────────
# Output paths (per split)
# ───────────────────────────────────────────────────────────────────────────

ROOT_OUT_DIR = Path("./results/voyager_pilot_v3_claude")


def _split_dir(split_id: int) -> Path:
    return ROOT_OUT_DIR / SPLITS[split_id]["name"]


def _iter_dir(split_id: int) -> Path:
    return _split_dir(split_id) / "iter_states"


# ───────────────────────────────────────────────────────────────────────────
# GATE-06 taboo regex (mirrors BENCHMARK_TABOO.md scope + v1 regex)
# ───────────────────────────────────────────────────────────────────────────

TABOO_PATTERN = re.compile(
    r"(?i)\b("
    r"Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|"
    r"HTML5 Server|Nginx Proxy|Kafka Broker|Zookeeper|UserDBAdapter|"
    r"AudioWatermarking|MediaManagement|WebUI|Recommender|Persistence|"
    r"SlopeOneRecommender|ImageProvider|Datastore|JabRef|bibdatabase|bibentry|"
    r"mediastore|teastore|teammates|bigbluebutton|jabref|"
    r"PaymentSystem|UserDB|FrontEnd|Backend"
    r")\b"
)


# ───────────────────────────────────────────────────────────────────────────
# Budget tracking (per-process, reset per script invocation)
# ───────────────────────────────────────────────────────────────────────────

_call_count = 0
_start_time = time.time()


def _budget_ok() -> bool:
    return _call_count < MAX_CALLS and (time.time() - _start_time) < WALL_BUDGET_SECONDS


def _bump_call(reason: str = "") -> None:
    global _call_count
    _call_count += 1
    elapsed = time.time() - _start_time
    print(f"  [budget] calls={_call_count}/{MAX_CALLS} elapsed={elapsed:.0f}s {reason}")


# ───────────────────────────────────────────────────────────────────────────
# Dual-format serialization
# ───────────────────────────────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} not JSON serializable")


def _dump_pair(payload, base_path: Path) -> None:
    """Write payload to ``base_path.json`` AND ``base_path.pkl``.

    JSON is pretty-printed with custom defaults; pickle uses protocol 4
    for broad compatibility.
    """
    base_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = base_path.with_suffix(".json")
    pkl_path = base_path.with_suffix(".pkl")
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)


# ───────────────────────────────────────────────────────────────────────────
# Skill-bank IO (per-split)
# ───────────────────────────────────────────────────────────────────────────

def _bank_path(split_id: int) -> Path:
    return _split_dir(split_id) / "skill_bank.json"


def _distilled_path(split_id: int) -> Path:
    return _split_dir(split_id) / "distilled_skills.json"


def _load_skill_bank(split_id: int) -> list[dict[str, str]]:
    p = _bank_path(split_id)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, ValueError):
        return []
    return list(data.get("abstract_patterns", []))


def _save_skill_bank(split_id: int, patterns: list[dict[str, str]]) -> None:
    p = _bank_path(split_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"abstract_patterns": patterns}, indent=2))
    # Pickle counterpart
    with open(p.with_suffix(".pkl"), "wb") as f:
        pickle.dump({"abstract_patterns": patterns}, f, protocol=4)


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z]{3,}", s.lower())}


def _dedupe_patterns(
    existing: list[dict[str, str]],
    new: list[dict[str, str]],
) -> list[dict[str, str]]:
    kept: list[dict[str, str]] = []
    existing_tokens = [_tokens(p["pattern"]) for p in existing]
    for cand in new:
        ct = _tokens(cand["pattern"])
        if not ct:
            continue
        clash = False
        for et in existing_tokens + [_tokens(k["pattern"]) for k in kept]:
            inter = len(ct & et)
            union = len(ct | et) or 1
            if (inter / union) >= DEDUP_SIM_FLOOR:
                clash = True
                break
        if not clash:
            kept.append(cand)
    return kept


def _gate06_ok(pattern_text: str) -> tuple[bool, list[str]]:
    hits = TABOO_PATTERN.findall(pattern_text)
    return (len(hits) == 0, hits)


# ───────────────────────────────────────────────────────────────────────────
# Linker invocation + scoring
# ───────────────────────────────────────────────────────────────────────────

def _run_linker(
    project: str,
    backend: LLMBackend,
    skill_path: str,
    model: str | None = None,
) -> dict:
    paths = _ra.DATASETS[project]
    text_path = str(paths["text"])
    model_path = str(paths["model"])
    gold_path = str(paths["gold_sam"])

    linker = SLinker13SkillLearned(backend=backend, model=model, skill_path=skill_path)
    t0 = time.time()
    links = linker.link(text_path=text_path, model_path=model_path)
    elapsed = time.time() - t0
    _bump_call(f"LINKER run project={project}")

    predicted = {(lk.sentence_number, lk.component_id) for lk in links}
    gold = _ra.load_gold_sam(gold_path)
    metrics = _ra.eval_metrics(predicted, gold)

    id_to_name = {lk.component_id: lk.component_name for lk in links}
    from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
    components = parse_pcm_repository(model_path)
    for c in components:
        id_to_name.setdefault(c.id, c.name)

    fps = sorted(predicted - gold)
    fns = sorted(gold - predicted)

    return {
        "project": project,
        "F1": metrics["F1"],
        "P": metrics["P"],
        "R": metrics["R"],
        "fp_count": metrics["fp"],
        "fn_count": metrics["fn"],
        "elapsed_s": elapsed,
        "fps": [(s, c, id_to_name.get(c, "?")) for s, c in fps],
        "fns": [(s, c, id_to_name.get(c, "?")) for s, c in fns],
        "predicted_count": len(predicted),
        "gold_count": len(gold),
        "predicted_pairs": sorted(predicted),
        "gold_pairs": sorted(gold),
    }


def _list_components(project: str) -> list[str]:
    from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
    model_path = str(_ra.DATASETS[project]["model"])
    return [c.name for c in parse_pcm_repository(model_path)]


# ───────────────────────────────────────────────────────────────────────────
# Feedback LLM call
# ───────────────────────────────────────────────────────────────────────────

FEEDBACK_PROMPT = """You analyzed an alias / sentence-component judgment task on a software architecture document. We will use your feedback to improve future judgments on UNRELATED documents.

CURRENT SKILL BANK (already-derived abstract patterns; do not duplicate):
{current_skills_block}

PREDICTION ERRORS ON THIS PROJECT
=================================
FALSE POSITIVES (predicted a sentence-component pair, but gold says no): up to {fp_show} of {fp_total} shown.
{fp_block}

FALSE NEGATIVES (gold says yes, but we did not predict it): up to {fn_show} of {fn_total} shown.
{fn_block}

DOCUMENT SENTENCES referenced above are:
{sentence_block}

COMPONENTS (architectural elements of this project):
{component_list}

TASK
====
Identify 0-{max_patterns} ABSTRACT PATTERNS that would help future judgments avoid analogous errors on unrelated documents. For each pattern:
- Express it in textbook software-engineering terms ONLY (lexer, parser, scheduler, queue, broker, dispatcher, controller, etc.).
- DO NOT use any name, identifier, or term that came from the project document or component list above.
- Describe a category of error (e.g. "abbreviation introduced parenthetically without 'i.e.' should still be accepted"), not an individual case.
- Be specific about WHICH of these 9 prompts the pattern applies to (or empty for global):
{prompt_names_block}

Output JSON exactly:
{{"patterns": [
   {{"prompt": "<one of the 9 names above or empty>", "pattern": "<one-sentence abstract pattern>"}}
]}}

If no new generalizable pattern is suggested by the errors, output {{"patterns": []}}.
JSON only:"""


def _build_feedback_prompt(
    fp_records: list[tuple],
    fn_records: list[tuple],
    project: str,
    skill_bank: list[dict[str, str]],
) -> tuple[str, set[int]]:
    SHOW = 8
    fp_shown = fp_records[:SHOW]
    fn_shown = fn_records[:SHOW]
    referenced_sents: set[int] = set()
    for snum, _cid, _name in fp_shown + fn_shown:
        referenced_sents.add(snum)

    def _fmt(records):
        return "\n".join(
            f"  sentence {s}: predicted '{n}' (component_id={c})"
            for s, c, n in records
        ) or "  (none)"

    from llm_sad_sam.core.document_loader_v2 import load_sentences
    sents = load_sentences(str(_ra.DATASETS[project]["text"]))
    sent_block_lines = []
    for s in sorted(referenced_sents):
        match = [x for x in sents if x.number == s]
        if match:
            txt = match[0].text
            if len(txt) > 280:
                txt = txt[:277] + "..."
            sent_block_lines.append(f"  [S{s}] {txt}")
    sent_block = "\n".join(sent_block_lines) or "  (no referenced sentences)"

    components = _list_components(project)
    skills_str = "\n".join(
        f"  - [{p.get('prompt','')}] {p['pattern']}" for p in skill_bank
    ) or "  (empty — first iteration)"

    prompt = FEEDBACK_PROMPT.format(
        current_skills_block=skills_str,
        fp_show=len(fp_shown),
        fp_total=len(fp_records),
        fn_show=len(fn_shown),
        fn_total=len(fn_records),
        fp_block=_fmt(fp_shown),
        fn_block=_fmt(fn_shown),
        sentence_block=sent_block,
        component_list=", ".join(components),
        max_patterns=MAX_PATTERNS_PER_CALL,
        prompt_names_block="    " + ", ".join(PROMPT_CONSTANT_NAMES),
    )
    return prompt, referenced_sents


def _extract_patterns_capture(
    llm: LLMClient,
    feedback_prompt: str,
) -> dict:
    """Call LLM; return a full capture dict (prompt, raw, parsed, accepted, rejected)."""
    raw_response = llm.query(feedback_prompt, timeout=240)
    _bump_call("FEEDBACK pattern extraction")
    data = llm.extract_json(raw_response) or {}
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    patterns = data.get("patterns", []) if isinstance(data, dict) else []
    if not isinstance(patterns, list):
        patterns = []
    valid_scopes = set(PROMPT_CONSTANT_NAMES) | {""}
    raw_proposals = []
    for raw in patterns:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("pattern", "")).strip()
        scope = str(raw.get("prompt", "")).strip()
        if not text:
            continue
        raw_proposals.append({"prompt": scope, "pattern": text})
        if scope not in valid_scopes:
            scope = ""
        ok, hits = _gate06_ok(text)
        if not ok:
            rejected.append({
                "pattern": text,
                "prompt": scope,
                "reason": f"taboo hit: {hits}",
            })
            continue
        accepted.append({"prompt": scope, "pattern": text})
    return {
        "feedback_prompt": feedback_prompt,
        "raw_response": str(raw_response),
        "parsed_json": data,
        "raw_proposals": raw_proposals,
        "accepted_patterns": accepted,
        "rejected_patterns": rejected,
    }


# ───────────────────────────────────────────────────────────────────────────
# Training loop (per split)
# ───────────────────────────────────────────────────────────────────────────

def train(
    split_id: int,
    backend: LLMBackend,
    model: str,
    fresh_start: bool,
    save_intermediate: bool,
) -> dict:
    split = SPLITS[split_id]
    train_projects = split["train"]
    test_projects = split["test"]

    out_dir = _split_dir(split_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    _iter_dir(split_id).mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"V3 TRAINING — split {split_id}: {split['name']}")
    print(f"  Train: {train_projects}")
    print(f"  Test:  {test_projects}")
    print(f"  Backend: {backend.value}  model: {model}")
    print(f"  Fresh start: {fresh_start}  Save intermediate: {save_intermediate}")
    print(f"  Max outer: {MAX_OUTER_PASSES}  Max inner: {MAX_INNER_ITERS}")
    print(f"  Convergence threshold (macro): {CONVERGENCE_THRESHOLD}")
    print("=" * 78)

    split_config = {
        "split_id": split_id,
        "split_name": split["name"],
        "description": split["description"],
        "train_projects": train_projects,
        "test_projects": test_projects,
        "fresh_start": fresh_start,
        "save_intermediate": save_intermediate,
        "max_outer_passes": MAX_OUTER_PASSES,
        "max_inner_iters": MAX_INNER_ITERS,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "max_calls": MAX_CALLS,
        "wall_budget_s": WALL_BUDGET_SECONDS,
        "backend": backend.value,
        "model": model,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
    }
    _dump_pair(split_config, out_dir / "split_config")

    if fresh_start:
        _save_skill_bank(split_id, [])
        print("[fresh-start] skill_bank emptied")

    llm = LLMClient(backend=backend, model=model)
    skill_bank = _load_skill_bank(split_id)
    trajectory: list[dict] = []
    rejected_log: list[dict] = []

    iter_counter = 0  # global counter for intermediate file naming

    for outer in range(MAX_OUTER_PASSES):
        if not _budget_ok():
            print("[budget] stopping outer loop")
            break
        per_project_final_f1: dict[str, float] = {}
        ordered = list(train_projects)
        random.Random(42 + outer + split_id * 100).shuffle(ordered)
        for project in ordered:
            if not _budget_ok():
                break
            f1_history: list[float] = []
            for inner in range(MAX_INNER_ITERS):
                if not _budget_ok():
                    break
                print(f"\n--- split={split_id} outer={outer} inner={inner} "
                      f"project={project} skills={len(skill_bank)} ---")
                _save_skill_bank(split_id, skill_bank)
                run = _run_linker(project, backend, str(_bank_path(split_id)),
                                  model=model)
                f1_history.append(run["F1"])
                print(f"  F1={run['F1']:.4f} P={run['P']:.4f} R={run['R']:.4f} "
                      f"FP={run['fp_count']} FN={run['fn_count']}")

                traj_row = {
                    "iter_counter": iter_counter,
                    "split_id": split_id,
                    "outer": outer,
                    "inner": inner,
                    "project": project,
                    "F1": run["F1"],
                    "P": run["P"],
                    "R": run["R"],
                    "fp": run["fp_count"],
                    "fn": run["fn_count"],
                    "skills_before": len(skill_bank),
                    "elapsed_s": run["elapsed_s"],
                    "backend": backend.value,
                    "model": model,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                trajectory.append(traj_row)

                # Per-iter intermediate dump (state + preds + feedback)
                if save_intermediate:
                    state_payload = {
                        "iter_counter": iter_counter,
                        "trajectory_row": traj_row,
                        "skill_bank_snapshot": list(skill_bank),
                        "skill_bank_size_before_feedback": len(skill_bank),
                    }
                    _dump_pair(
                        state_payload,
                        _iter_dir(split_id) / f"iter_{iter_counter:03d}_state",
                    )
                    preds_payload = {
                        "iter_counter": iter_counter,
                        "project": project,
                        "fps": run["fps"],
                        "fns": run["fns"],
                        "predicted_pairs": run["predicted_pairs"],
                        "gold_pairs": run["gold_pairs"],
                        "predicted_count": run["predicted_count"],
                        "gold_count": run["gold_count"],
                    }
                    _dump_pair(
                        preds_payload,
                        _iter_dir(split_id) /
                        f"iter_{iter_counter:03d}_predictions_vs_gold",
                    )

                # Early-stop
                if run["F1"] >= PER_PROJECT_EARLY_STOP:
                    print(f"  [early-stop] project F1 >= {PER_PROJECT_EARLY_STOP}")
                    iter_counter += 1
                    break

                if run["fp_count"] == 0 and run["fn_count"] == 0:
                    print("  [skip feedback] no errors to learn from")
                    iter_counter += 1
                    break

                feedback_prompt, _ref = _build_feedback_prompt(
                    run["fps"], run["fns"], project, skill_bank
                )
                if not _budget_ok():
                    print("[budget] skip feedback call")
                    iter_counter += 1
                    break

                feedback_capture = _extract_patterns_capture(llm, feedback_prompt)
                feedback_capture["iter_counter"] = iter_counter
                feedback_capture["project"] = project
                feedback_capture["outer"] = outer
                feedback_capture["inner"] = inner
                feedback_capture["backend"] = backend.value
                feedback_capture["model"] = model
                feedback_capture["timestamp"] = datetime.now(timezone.utc).isoformat()

                if save_intermediate:
                    _dump_pair(
                        feedback_capture,
                        _iter_dir(split_id) /
                        f"iter_{iter_counter:03d}_feedback_call",
                    )

                accepted = feedback_capture["accepted_patterns"]
                rejected = feedback_capture["rejected_patterns"]
                for r in rejected:
                    rejected_log.append({
                        "iter_counter": iter_counter,
                        "outer": outer, "inner": inner, "project": project,
                        "backend": backend.value, "model": model,
                        **r,
                    })

                # Dedupe vs existing bank + accepted-this-call
                deduped = _dedupe_patterns(skill_bank, accepted)
                drop_count = len(accepted) - len(deduped)
                for cand in accepted:
                    if cand not in deduped:
                        rejected_log.append({
                            "iter_counter": iter_counter,
                            "outer": outer, "inner": inner, "project": project,
                            "backend": backend.value, "model": model,
                            "pattern": cand["pattern"],
                            "prompt": cand["prompt"],
                            "reason": "dedupe (jaccard >= floor)",
                        })
                skill_bank.extend(deduped)
                _save_skill_bank(split_id, skill_bank)
                print(f"  feedback: proposed={len(accepted) + len(rejected)} "
                      f"accepted={len(accepted)} taboo_rejected={len(rejected)} "
                      f"dedupe_dropped={drop_count} bank_size={len(skill_bank)}")

                iter_counter += 1

            if f1_history:
                per_project_final_f1[project] = f1_history[-1]
        macro = (sum(per_project_final_f1.values()) / len(per_project_final_f1)
                 if per_project_final_f1 else 0.0)
        print(f"\n=== split={split_id} outer={outer} done. "
              f"per-project final F1: {per_project_final_f1} macro={macro:.4f} ===")
        if macro >= CONVERGENCE_THRESHOLD:
            print(f"[converged] macro {macro:.4f} >= {CONVERGENCE_THRESHOLD}")
            break

    final_outer = locals().get("outer", -1)
    summary = {
        "split_id": split_id,
        "split_name": split["name"],
        "outer_passes_completed": final_outer + 1,
        "calls_used": _call_count,
        "elapsed_s": time.time() - _start_time,
        "final_skill_bank_size": len(skill_bank),
        "trajectory": trajectory,
        "rejected_patterns": rejected_log,
        "skill_bank": skill_bank,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    _dump_pair(summary, out_dir / "train_trajectory")
    _dump_pair({"rejected_patterns": rejected_log},
               out_dir / "rejected_patterns")
    return summary


# ───────────────────────────────────────────────────────────────────────────
# Distillation (per split)
# ───────────────────────────────────────────────────────────────────────────

DISTILL_PROMPT = """You are distilling a skill bank of learned abstract patterns down to at most {target} high-leverage patterns. Each kept pattern should:
- Be expressed in textbook software-engineering terms ONLY (lexer, parser, scheduler, broker, etc.).
- Be a single sentence stating an abstract rule, NOT a specific example or case.
- Be defensible to a reviewer as a UNIVERSAL property of alias / component-reference judgment, not tailored to one project.

CURRENT SKILL BANK ({n} patterns):
{bank_block}

Output JSON:
{{"distilled_skills": [
  {{"prompt": "<one of {prompt_names} or empty>", "pattern": "<one-sentence universal rule>"}}
]}}

JSON only:"""


def distill(
    split_id: int,
    backend: LLMBackend,
    model: str,
) -> dict:
    out_dir = _split_dir(split_id)
    print("\n" + "=" * 78)
    print(f"V3 DISTILLATION — split {split_id}: {SPLITS[split_id]['name']}")
    print(f"  Backend: {backend.value}  model: {model}")
    print("=" * 78)

    skill_bank = _load_skill_bank(split_id)
    distill_capture = {
        "split_id": split_id,
        "split_name": SPLITS[split_id]["name"],
        "backend": backend.value,
        "model": model,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "n_in": len(skill_bank),
    }

    if not skill_bank:
        print("[distill] empty skill bank — nothing to distill")
        distill_capture.update({
            "distill_prompt": None,
            "raw_response": None,
            "parsed_json": None,
            "raw_distilled": [],
            "accepted": [],
            "rejected": [],
        })
        _dump_pair(distill_capture, out_dir / "distill_call")
        out_dir.joinpath("distilled_skills.json").write_text(
            json.dumps({"distilled_skills": []}, indent=2)
        )
        with open(out_dir / "distilled_skills.pkl", "wb") as f:
            pickle.dump({"distilled_skills": []}, f, protocol=4)
        return {"distilled": [], "rejected": []}

    bank_block = "\n".join(
        f"  - [{p.get('prompt','')}] {p['pattern']}" for p in skill_bank
    )
    distill_prompt = DISTILL_PROMPT.format(
        target=DISTILL_TARGET_COUNT,
        n=len(skill_bank),
        bank_block=bank_block,
        prompt_names=PROMPT_CONSTANT_NAMES,
    )

    llm = LLMClient(backend=backend, model=model)
    raw_response = llm.query(distill_prompt, timeout=300)
    _bump_call("DISTILL")
    parsed = llm.extract_json(raw_response) or {}
    raw_distilled = parsed.get("distilled_skills", []) if isinstance(parsed, dict) else []

    accepted = []
    rejected = []
    for raw in raw_distilled:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("pattern", "")).strip()
        scope = str(raw.get("prompt", "")).strip()
        if not text:
            continue
        if scope not in (set(PROMPT_CONSTANT_NAMES) | {""}):
            scope = ""
        ok, hits = _gate06_ok(text)
        if ok:
            accepted.append({"prompt": scope, "pattern": text})
        else:
            rejected.append({"prompt": scope, "pattern": text,
                             "reason": f"taboo: {hits}"})

    distill_capture.update({
        "distill_prompt": distill_prompt,
        "raw_response": str(raw_response),
        "parsed_json": parsed,
        "raw_distilled": raw_distilled,
        "accepted_after_gate06": list(accepted),
        "rejected_at_gate06": list(rejected),
    })

    # Reviewer-defensibility critic
    reviewer_capture: dict = {
        "split_id": split_id,
        "backend": backend.value,
        "model": model,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "candidates": list(accepted),
    }
    if accepted:
        review_prompt = (
            "Read these candidate UNIVERSAL rules for judging whether alias mappings "
            "or sentence-component references are valid. For each rule, decide whether "
            "it could ONLY have been derived from a specific named project (project-derived) "
            "or whether it reads as a universal property of natural-language reference "
            "to architectural components (defensible).\n\n"
            "CANDIDATES:\n" +
            "\n".join(f"  {i}. {p['pattern']}" for i, p in enumerate(accepted)) +
            "\n\nOutput JSON: {\"verdicts\": [{\"index\": int, \"defensible\": bool, "
            "\"reason\": \"<short>\"}]}\nJSON only:"
        )
        review_raw = llm.query(review_prompt, timeout=180)
        _bump_call("DISTILL reviewer")
        review_data = llm.extract_json(review_raw) or {}
        verdicts = review_data.get("verdicts", []) if isinstance(review_data, dict) else []
        verdict_by_idx = {int(v["index"]): v for v in verdicts
                          if isinstance(v, dict) and "index" in v}
        kept = []
        flagged = []
        for i, p in enumerate(accepted):
            v = verdict_by_idx.get(i, {"defensible": True, "reason": "no verdict — kept"})
            if v.get("defensible", True):
                kept.append({**p, "review": v.get("reason", "")})
            else:
                flagged.append({**p, "review": v.get("reason", "flagged")})
                rejected.append({**p, "reason": f"reviewer: {v.get('reason', '')}"})
        accepted = kept

        reviewer_capture.update({
            "review_prompt": review_prompt,
            "raw_response": str(review_raw),
            "parsed_json": review_data,
            "verdicts": verdicts,
            "kept": kept,
            "flagged": flagged,
        })
    else:
        reviewer_capture.update({
            "review_prompt": None,
            "raw_response": None,
            "parsed_json": None,
            "verdicts": [],
            "kept": [],
            "flagged": [],
            "note": "no candidates to review (empty distillation)",
        })

    # Final distilled artefact
    distilled_payload = {"distilled_skills": accepted}
    out_dir.joinpath("distilled_skills.json").write_text(
        json.dumps(distilled_payload, indent=2))
    with open(out_dir / "distilled_skills.pkl", "wb") as f:
        pickle.dump(distilled_payload, f, protocol=4)

    distill_capture["n_out"] = len(accepted)
    distill_capture["rejected_total"] = rejected
    _dump_pair(distill_capture, out_dir / "distill_call")
    _dump_pair(reviewer_capture, out_dir / "reviewer_call")

    print(f"[distill] in={len(skill_bank)} out={len(accepted)} rejected={len(rejected)}")
    return {"n_in": len(skill_bank), "n_out": len(accepted),
            "rejected": rejected, "distilled": accepted}


# ───────────────────────────────────────────────────────────────────────────
# Test (per split)
# ───────────────────────────────────────────────────────────────────────────

def test(
    split_id: int,
    backend: LLMBackend,
    model: str,
) -> dict:
    split = SPLITS[split_id]
    out_dir = _split_dir(split_id)
    print("\n" + "=" * 78)
    print(f"V3 TEST — split {split_id}: {split['name']}")
    print(f"  Backend: {backend.value}  model: {model}")
    print(f"  Test:  {split['test']}")
    print(f"  Train sanity: {split['train']}")
    print("=" * 78)

    all_projects = list(split["test"]) + list(split["train"])

    results: dict[str, dict] = {}
    distilled_path = _distilled_path(split_id)

    for project in all_projects:
        if not _budget_ok():
            print(f"[budget] skip test on {project}")
            break
        print(f"\n--- TEST split={split_id} project={project} ---")

        empty_skill = out_dir / "_empty_skills.json"
        empty_skill.write_text('{"abstract_patterns": []}')
        axiom_run = _run_linker(project, backend, str(empty_skill), model=model)

        distilled_run = None
        if distilled_path.exists():
            distilled_run = _run_linker(project, backend, str(distilled_path),
                                        model=model)

        baseline_f1 = _baseline_from_cache(project, "s_linker13_clean")
        trim1_f1 = _baseline_from_cache(project, "s_linker13_trim1_judge_clean")

        results[project] = {
            "axiom_only_F1": axiom_run["F1"],
            "axiom_only_FP": axiom_run["fp_count"],
            "axiom_only_FN": axiom_run["fn_count"],
            "distilled_F1": distilled_run["F1"] if distilled_run else None,
            "distilled_FP": distilled_run["fp_count"] if distilled_run else None,
            "distilled_FN": distilled_run["fn_count"] if distilled_run else None,
            "s_linker13_clean_F1_cached": baseline_f1,
            "s_linker13_trim1_F1_cached": trim1_f1,
            "split": "test" if project in split["test"] else "train_sanity",
        }
        print(f"  axiom={axiom_run['F1']:.4f} "
              f"distilled={distilled_run['F1'] if distilled_run else 'NA'} "
              f"clean_baseline={baseline_f1} trim1={trim1_f1}")

    held_out = [r for k, r in results.items() if r["split"] == "test"]
    macros = {
        "axiom_only_macro_holdout": (
            sum(r["axiom_only_F1"] for r in held_out) / len(held_out)
            if held_out else None),
        "distilled_macro_holdout": (
            sum((r["distilled_F1"] or 0.0) for r in held_out) / len(held_out)
            if held_out and all(r["distilled_F1"] is not None for r in held_out)
            else None),
        "n_held_out": len(held_out),
    }
    summary = {
        "split_id": split_id,
        "split_name": split["name"],
        "backend": backend.value,
        "model": model,
        "per_project": results,
        "macros": macros,
        "calls_used": _call_count,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    _dump_pair(summary, out_dir / "test_results")
    print(f"\n[test] {summary['macros']}")
    return summary


def _baseline_from_cache(project: str, variant: str) -> float | None:
    cache_path = Path(f"./results/phase_cache/{variant}/{project}/final.pkl")
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        links = data.get("final", []) if isinstance(data, dict) else []
        predicted = {(lk.sentence_number, lk.component_id) for lk in links}
        gold = _ra.load_gold_sam(str(_ra.DATASETS[project]["gold_sam"]))
        return _ra.eval_metrics(predicted, gold)["F1"]
    except Exception as e:
        print(f"  [warn] baseline cache read failed for {variant}/{project}: {e}")
        return None


# ───────────────────────────────────────────────────────────────────────────
# Cross-split rollup
# ───────────────────────────────────────────────────────────────────────────

def rollup() -> dict:
    rows = []
    for sid in sorted(SPLITS):
        sd = _split_dir(sid)
        tr_path = sd / "test_results.json"
        bank_path = sd / "skill_bank.json"
        distilled_path = sd / "distilled_skills.json"
        if not tr_path.exists():
            print(f"[rollup] split{sid}: no test_results.json — skipping")
            continue
        tr = json.loads(tr_path.read_text())
        bank = json.loads(bank_path.read_text()) if bank_path.exists() else {}
        dist = json.loads(distilled_path.read_text()) if distilled_path.exists() else {}

        per_proj = tr.get("per_project", {})
        # Macro across all 5 projects (incl train sanity)
        all_distilled = [r.get("distilled_F1") for r in per_proj.values()
                         if r.get("distilled_F1") is not None]
        all_axiom = [r.get("axiom_only_F1") for r in per_proj.values()
                     if r.get("axiom_only_F1") is not None]
        all_trim1 = [r.get("s_linker13_trim1_F1_cached") for r in per_proj.values()
                     if r.get("s_linker13_trim1_F1_cached") is not None]

        macros = tr.get("macros", {})
        rows.append({
            "split_id": sid,
            "split_name": SPLITS[sid]["name"],
            "train_projects": SPLITS[sid]["train"],
            "test_projects": SPLITS[sid]["test"],
            "skill_bank_size": len(bank.get("abstract_patterns", [])),
            "distilled_skill_count": len(dist.get("distilled_skills", [])),
            "axiom_only_macro_holdout": macros.get("axiom_only_macro_holdout"),
            "distilled_macro_holdout": macros.get("distilled_macro_holdout"),
            "axiom_only_macro_all5": (sum(all_axiom) / len(all_axiom)) if all_axiom else None,
            "distilled_macro_all5": (sum(all_distilled) / len(all_distilled)) if all_distilled else None,
            "trim1_macro_all5": (sum(all_trim1) / len(all_trim1)) if all_trim1 else None,
            "delta_vs_axiom_holdout_pp": (
                (macros.get("distilled_macro_holdout") - macros.get("axiom_only_macro_holdout")) * 100
                if (macros.get("distilled_macro_holdout") is not None
                    and macros.get("axiom_only_macro_holdout") is not None)
                else None
            ),
            "per_project": per_proj,
        })

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "splits": rows,
    }
    _dump_pair(payload, ROOT_OUT_DIR / "crossplit_comparison")
    print(f"[rollup] wrote {ROOT_OUT_DIR / 'crossplit_comparison.json'}")
    return payload


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def _resolve_backend(name: str) -> LLMBackend:
    return {
        "claude": LLMBackend.CLAUDE,
        "openai": LLMBackend.OPENAI,
        "checkpoint": LLMBackend.CHECKPOINT,
    }[name]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["train", "distill", "test", "all", "rollup"])
    ap.add_argument("--split-id", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--backend", default="claude")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--fresh-start", action="store_true",
                    help="Empty the skill_bank before training (default per directive).")
    ap.add_argument("--save-intermediate", action="store_true",
                    help="Dump per-iter json+pkl (state, feedback, predictions).")
    args = ap.parse_args(argv)

    if args.phase == "rollup":
        rollup()
        return 0

    backend = _resolve_backend(args.backend)
    sid = args.split_id

    if args.phase in ("train", "all"):
        train(sid, backend, args.model, args.fresh_start, args.save_intermediate)
    if args.phase in ("distill", "all"):
        distill(sid, backend, args.model)
    if args.phase in ("test", "all"):
        test(sid, backend, args.model)

    print(f"\n[done] split={sid} total calls={_call_count} "
          f"elapsed={time.time() - _start_time:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
