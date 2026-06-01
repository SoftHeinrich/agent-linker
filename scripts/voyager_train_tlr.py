"""Voyager-style train/test pilot driver for the SAD-SAM TLR task.

Phase 1 — train on 3 random projects (MS, TS, TM), grow a skill bank from
supervised FP/FN feedback, save after each call.
Phase 2 — distill the skill bank to a frozen file.
Phase 3 — test on held-out 2 projects (BBB, JAB) with the frozen skills.
Phase 4 — compare against axiom-only floor + s_linker13 baseline.

USAGE
-----
    python scripts/voyager_train_tlr.py train
    python scripts/voyager_train_tlr.py distill
    python scripts/voyager_train_tlr.py test
    python scripts/voyager_train_tlr.py all

OUTPUTS (./results/voyager_pilot/)
----------------------------------
- skill_bank.json          (training, accumulates patterns)
- distilled_skills.json    (frozen for test)
- train_trajectory.json    (per-iter F1 per project)
- test_results.json        (held-out F1 vs baselines)
- run_log/<phase>/<project>/<iter>.json (per-run details)

GATE-06
-------
Every pattern written to skill_bank.json or distilled_skills.json passes
benchmark-taboo regex before persistence. Patterns that fail are dropped
and counted in the failure log.

BUDGET
------
Hard cap: 6 hour wallclock OR $50 estimated API spend (tracked via call count
at ~$0.10/Claude call rough estimate — we count calls, not dollars, since the
real meter is opaque). Cap configurable via env VOYAGER_MAX_CALLS.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
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
# Configuration
# ───────────────────────────────────────────────────────────────────────────

TRAIN_PROJECTS = ["mediastore", "teastore", "teammates"]
TEST_PROJECTS = ["bigbluebutton", "jabref"]
MAX_OUTER_PASSES = int(os.environ.get("VOYAGER_MAX_OUTER", "3"))
MAX_INNER_ITERS = int(os.environ.get("VOYAGER_MAX_INNER", "3"))
# Convergence threshold default 0.93 historically (Claude). gpt-5.4 baseline
# is ~0.91 so we relax to 0.90 under Scenario E. Overridable via env.
CONVERGENCE_THRESHOLD = float(os.environ.get("VOYAGER_CONV_THRESH", "0.93"))
PER_PROJECT_EARLY_STOP = 0.95  # if a project hits this, skip further inner iters
MAX_PATTERNS_PER_CALL = 3
DEDUP_SIM_FLOOR = 0.6           # crude jaccard token similarity
DISTILL_TARGET_COUNT = 8

OUT_DIR = Path("./results/voyager_pilot")
RUN_LOG_DIR = OUT_DIR / "run_log"
SKILL_BANK_PATH = OUT_DIR / "skill_bank.json"
DISTILLED_PATH = OUT_DIR / "distilled_skills.json"
TRAJECTORY_PATH = OUT_DIR / "train_trajectory.json"
TEST_RESULTS_PATH = OUT_DIR / "test_results.json"
DEFENSIBILITY_PATH = OUT_DIR / "defensibility_audit.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

# GATE-06 taboo regex (mirrors scripts/audit_12_05_revisit.py + adds
# project names to catch the obvious cases).
TABOO_PATTERN = re.compile(
    r"(?i)\b("
    r"Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|"
    r"HTML5 Server|Nginx Proxy|Kafka Broker|Zookeeper|UserDBAdapter|"
    r"AudioWatermarking|MediaManagement|WebUI|Recommender|Persistence|"
    r"SlopeOneRecommender|ImageProvider|Datastore|JabRef|bibdatabase|bibentry|"
    r"mediastore|teastore|teammates|bigbluebutton|jabref|"
    # universal taboo from BENCHMARK_TABOO.md
    r"PaymentSystem|UserDB|FrontEnd|Backend"
    r")\b"
)

# Per-call cap for budget protection
MAX_CALLS = int(os.environ.get("VOYAGER_MAX_CALLS", "120"))
WALL_BUDGET_SECONDS = int(os.environ.get("VOYAGER_WALL_BUDGET_S", str(6 * 3600)))

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
# Skill-bank IO
# ───────────────────────────────────────────────────────────────────────────

def _load_skill_bank() -> list[dict[str, str]]:
    if not SKILL_BANK_PATH.exists():
        return []
    try:
        data = json.loads(SKILL_BANK_PATH.read_text())
    except (json.JSONDecodeError, ValueError):
        return []
    return list(data.get("abstract_patterns", []))


def _save_skill_bank(patterns: list[dict[str, str]]) -> None:
    SKILL_BANK_PATH.write_text(
        json.dumps({"abstract_patterns": patterns}, indent=2)
    )


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z]{3,}", s.lower())}


def _dedupe_patterns(
    existing: list[dict[str, str]],
    new: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Crude jaccard-similarity dedupe so we don't accumulate near-clones."""
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
    """Run the skill-learned variant on one project; return F1 + FP/FN."""
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

    # Build a name lookup for human-readable FP/FN reports
    id_to_name = {lk.component_id: lk.component_name for lk in links}
    # Also enrich from components in the model — they may be in gold but not predicted
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
    }


def _read_doc(project: str) -> str:
    return Path(_ra.DATASETS[project]["text"]).read_text()


def _list_components(project: str) -> list[str]:
    from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
    model_path = str(_ra.DATASETS[project]["model"])
    return [c.name for c in parse_pcm_repository(model_path)]


# ───────────────────────────────────────────────────────────────────────────
# Feedback LLM call — derive abstract patterns from FP/FN
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
    """Construct feedback prompt + return the set of referenced sentence numbers."""
    SHOW = 8
    fp_shown = fp_records[:SHOW]
    fn_shown = fn_records[:SHOW]
    referenced_sents: set[int] = set()
    for snum, _cid, _name in fp_shown + fn_shown:
        referenced_sents.add(snum)

    # Build FP/FN block text
    def _fmt(records):
        return "\n".join(
            f"  sentence {s}: predicted '{n}' (component_id={c})"
            for s, c, n in records
        ) or "  (none)"

    # Load doc sentence text (1-indexed)
    from llm_sad_sam.core.document_loader_v2 import load_sentences
    sents = load_sentences(str(_ra.DATASETS[project]["text"]))
    sent_block_lines = []
    for s in sorted(referenced_sents):
        # sentence object indexed 1..N; load_sentences returns list
        # Sentence.number is 1-indexed (document_loader_v2)
        match = [x for x in sents if x.number == s]
        if match:
            txt = match[0].text
            # Truncate very long sentences
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


def _extract_patterns(
    llm: LLMClient,
    feedback_prompt: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Call LLM; return (accepted_patterns, rejected_patterns_with_reason)."""
    resp = llm.query(feedback_prompt, timeout=240)
    _bump_call("FEEDBACK pattern extraction")
    data = llm.extract_json(resp)
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    if not data:
        return accepted, rejected
    patterns = data.get("patterns", [])
    if not isinstance(patterns, list):
        return accepted, rejected
    valid_scopes = set(PROMPT_CONSTANT_NAMES) | {""}
    for raw in patterns:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("pattern", "")).strip()
        scope = str(raw.get("prompt", "")).strip()
        if not text:
            continue
        if scope not in valid_scopes:
            scope = ""  # demote to global
        # GATE-06 audit
        ok, hits = _gate06_ok(text)
        if not ok:
            rejected.append({
                "pattern": text,
                "prompt": scope,
                "reason": f"taboo hit: {hits}",
            })
            continue
        accepted.append({"prompt": scope, "pattern": text})
    return accepted, rejected


# ───────────────────────────────────────────────────────────────────────────
# Training loop
# ───────────────────────────────────────────────────────────────────────────

def train(backend: LLMBackend = LLMBackend.CLAUDE, model: str | None = None,
          resume: bool = False) -> dict:
    print("=" * 78)
    print("PHASE 1 — TRAINING")
    print(f"Train projects: {TRAIN_PROJECTS}")
    print(f"Backend: {backend.value} model: {model or '<default>'}")
    print(f"Max outer passes: {MAX_OUTER_PASSES}, max inner iters: {MAX_INNER_ITERS}")
    print(f"Convergence threshold (macro): {CONVERGENCE_THRESHOLD}")
    print(f"Resume mode: {resume}")
    print("=" * 78)

    llm = LLMClient(backend=backend, model=model)
    skill_bank = _load_skill_bank()
    trajectory: list[dict] = []
    rejected_log: list[dict] = []
    outer_offset = 0
    if resume:
        # Load prior trajectory + rejected_log so we APPEND rather than overwrite.
        prior = {}
        if TRAJECTORY_PATH.exists():
            try:
                prior = json.loads(TRAJECTORY_PATH.read_text())
            except (json.JSONDecodeError, ValueError):
                prior = {}
        trajectory = list(prior.get("trajectory", []))
        rejected_log = list(prior.get("rejected_patterns", []))
        # Offset outer index so new outer passes don't collide with prior ones.
        if trajectory:
            outer_offset = max((t.get("outer", 0) for t in trajectory)) + 1
        print(f"[resume] prior_trajectory_rows={len(trajectory)} "
              f"prior_rejected={len(rejected_log)} prior_skill_bank={len(skill_bank)} "
              f"outer_offset={outer_offset}")

    # Tag every new trajectory entry with backend/model so we can audit which
    # patterns came from which backend after the fact.
    _backend_tag = backend.value
    _model_tag = model or "<default>"

    for _outer in range(MAX_OUTER_PASSES):
        outer = _outer + outer_offset
        if not _budget_ok():
            print("[budget] stopping outer loop")
            break
        per_project_final_f1: dict[str, float] = {}
        # Shuffle order so we don't always hit projects in the same sequence
        ordered = list(TRAIN_PROJECTS)
        random.Random(42 + outer).shuffle(ordered)
        for project in ordered:
            if not _budget_ok():
                print("[budget] stopping inner loop")
                break
            f1_history: list[float] = []
            for inner in range(MAX_INNER_ITERS):
                if not _budget_ok():
                    print("[budget] stopping inner iter")
                    break
                print(f"\n--- outer={outer} inner={inner} project={project} skills={len(skill_bank)} ---")
                _save_skill_bank(skill_bank)  # ensure linker reads latest
                run = _run_linker(project, backend, str(SKILL_BANK_PATH), model=model)
                f1_history.append(run["F1"])
                # Log run
                (RUN_LOG_DIR / f"train_outer{outer}_{project}_iter{inner}.json").write_text(
                    json.dumps({"run": {k: v for k, v in run.items()
                                        if k not in ("fps", "fns")},
                                "fp_count": run["fp_count"],
                                "fn_count": run["fn_count"]}, indent=2)
                )
                print(f"  F1={run['F1']:.4f} P={run['P']:.4f} R={run['R']:.4f} "
                      f"FP={run['fp_count']} FN={run['fn_count']}")

                trajectory.append({
                    "outer": outer,
                    "inner": inner,
                    "project": project,
                    "F1": run["F1"],
                    "fp": run["fp_count"],
                    "fn": run["fn_count"],
                    "skills_before": len(skill_bank),
                    "elapsed_s": run["elapsed_s"],
                    "backend": _backend_tag,
                    "model": _model_tag,
                })
                # Persist trajectory after every iter so a budget/crash mid-loop
                # still preserves progress.
                TRAJECTORY_PATH.write_text(json.dumps({
                    "outer_passes_completed": outer + 1,
                    "calls_used": _call_count,
                    "elapsed_s": time.time() - _start_time,
                    "final_skill_bank_size": len(skill_bank),
                    "trajectory": trajectory,
                    "rejected_patterns": rejected_log,
                    "skill_bank": skill_bank,
                }, indent=2))

                if run["F1"] >= PER_PROJECT_EARLY_STOP:
                    print(f"  [early-stop] project F1 >= {PER_PROJECT_EARLY_STOP}")
                    break

                # Feedback call — only if errors exist
                if run["fp_count"] == 0 and run["fn_count"] == 0:
                    print("  [skip feedback] no errors to learn from")
                    break

                feedback_prompt, _ = _build_feedback_prompt(
                    run["fps"], run["fns"], project, skill_bank
                )
                if not _budget_ok():
                    print("[budget] skip feedback call")
                    break
                accepted, rejected = _extract_patterns(llm, feedback_prompt)
                for r in rejected:
                    rejected_log.append({
                        "outer": outer, "inner": inner, "project": project,
                        "backend": _backend_tag, "model": _model_tag,
                        **r,
                    })
                # Dedupe vs existing
                deduped = _dedupe_patterns(skill_bank, accepted)
                skill_bank.extend(deduped)
                _save_skill_bank(skill_bank)
                print(f"  feedback: proposed={len(accepted) + len(rejected)} "
                      f"accepted={len(accepted)} taboo_rejected={len(rejected)} "
                      f"after_dedupe={len(deduped)} bank_size={len(skill_bank)}")
            if f1_history:
                per_project_final_f1[project] = f1_history[-1]
        macro = (sum(per_project_final_f1.values()) / len(per_project_final_f1)
                 if per_project_final_f1 else 0.0)
        print(f"\n=== outer={outer} done. per-project final F1: {per_project_final_f1} "
              f"macro={macro:.4f} ===")
        if macro >= CONVERGENCE_THRESHOLD:
            print(f"[converged] macro {macro:.4f} >= {CONVERGENCE_THRESHOLD}")
            break

    # `outer` may be undefined if MAX_OUTER_PASSES == 0; guard.
    final_outer = locals().get("outer", outer_offset - 1)
    summary = {
        "outer_passes_completed": final_outer + 1,
        "calls_used": _call_count,
        "elapsed_s": time.time() - _start_time,
        "final_skill_bank_size": len(skill_bank),
        "trajectory": trajectory,
        "rejected_patterns": rejected_log,
        "skill_bank": skill_bank,
    }
    TRAJECTORY_PATH.write_text(json.dumps(summary, indent=2))
    return summary


# ───────────────────────────────────────────────────────────────────────────
# Distillation
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


def distill(backend: LLMBackend = LLMBackend.CLAUDE, model: str | None = None) -> dict:
    print("\n" + "=" * 78)
    print("PHASE 2 — DISTILLATION")
    print(f"Backend: {backend.value} model: {model or '<default>'}")
    print("=" * 78)

    skill_bank = _load_skill_bank()
    if not skill_bank:
        print("[distill] empty skill bank — nothing to distill")
        DISTILLED_PATH.write_text(json.dumps({"distilled_skills": []}, indent=2))
        return {"distilled": [], "rejected": []}

    bank_block = "\n".join(
        f"  - [{p.get('prompt','')}] {p['pattern']}" for p in skill_bank
    )
    prompt = DISTILL_PROMPT.format(
        target=DISTILL_TARGET_COUNT,
        n=len(skill_bank),
        bank_block=bank_block,
        prompt_names=PROMPT_CONSTANT_NAMES,
    )

    llm = LLMClient(backend=backend, model=model)
    resp = llm.query(prompt, timeout=300)
    _bump_call("DISTILL")
    data = llm.extract_json(resp)
    raw_distilled = data.get("distilled_skills", []) if data else []

    # Audit each distilled pattern
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
            rejected.append({"prompt": scope, "pattern": text, "reason": f"taboo: {hits}"})

    # Reviewer LLM call — defensibility check
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
        review_resp = llm.query(review_prompt, timeout=180)
        _bump_call("DISTILL reviewer")
        review_data = llm.extract_json(review_resp)
        verdicts = review_data.get("verdicts", []) if review_data else []
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
        defensibility_log = {"kept": kept, "flagged": flagged}
    else:
        defensibility_log = {"kept": [], "flagged": []}

    DISTILLED_PATH.write_text(json.dumps({"distilled_skills": accepted}, indent=2))
    DEFENSIBILITY_PATH.write_text(json.dumps(defensibility_log, indent=2))

    summary = {
        "n_in": len(skill_bank),
        "n_out": len(accepted),
        "rejected": rejected,
        "distilled": accepted,
    }
    print(f"[distill] in={summary['n_in']} out={summary['n_out']} "
          f"rejected={len(rejected)}")
    return summary


# ───────────────────────────────────────────────────────────────────────────
# Test phase
# ───────────────────────────────────────────────────────────────────────────

def test(backend: LLMBackend = LLMBackend.CLAUDE, model: str | None = None) -> dict:
    print("\n" + "=" * 78)
    print("PHASE 3 — TEST ON HELD-OUT")
    print(f"Backend: {backend.value} model: {model or '<default>'}")
    print(f"Test projects: {TEST_PROJECTS}")
    print("=" * 78)

    results: dict[str, dict] = {}
    # Re-run on training projects too (overfitting sanity check)
    all_projects = TEST_PROJECTS + TRAIN_PROJECTS
    for project in all_projects:
        if not _budget_ok():
            print(f"[budget] skipping test on {project}")
            break
        print(f"\n--- TEST project={project} ---")

        # Three variants per project:
        # 1) axiom-only floor (no skills)
        # 2) distilled skills (transfer)
        # 3) cached s_linker13_clean baseline (lookup from final.pkl if present)

        # Run 1 — axiom-only floor: point skill_path at a guaranteed-empty file
        empty_skill = OUT_DIR / "_empty_skills.json"
        empty_skill.write_text('{"abstract_patterns": []}')
        axiom_run = _run_linker(project, backend, str(empty_skill), model=model)
        (RUN_LOG_DIR / f"test_axiom_{project}.json").write_text(
            json.dumps({k: v for k, v in axiom_run.items()
                        if k not in ("fps", "fns")}, indent=2)
        )

        # Run 2 — distilled skills
        if DISTILLED_PATH.exists():
            distilled_run = _run_linker(project, backend, str(DISTILLED_PATH), model=model)
        else:
            distilled_run = None
        if distilled_run:
            (RUN_LOG_DIR / f"test_distilled_{project}.json").write_text(
                json.dumps({k: v for k, v in distilled_run.items()
                            if k not in ("fps", "fns")}, indent=2)
            )

        # Run 3 — baseline lookup from final.pkl in phase_cache
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
            "split": "test" if project in TEST_PROJECTS else "train_sanity",
        }
        print(f"  axiom={axiom_run['F1']:.4f} "
              f"distilled={distilled_run['F1'] if distilled_run else 'NA'} "
              f"clean_baseline={baseline_f1} trim1={trim1_f1}")

    # Macro summaries
    held_out = [r for k, r in results.items() if r["split"] == "test"]
    macro = {
        "axiom_only_macro_holdout": (
            sum(r["axiom_only_F1"] for r in held_out) / len(held_out)
            if held_out else None),
        "distilled_macro_holdout": (
            sum((r["distilled_F1"] or 0.0) for r in held_out) / len(held_out)
            if held_out and all(r["distilled_F1"] is not None for r in held_out) else None),
        "n_held_out": len(held_out),
    }
    summary = {"per_project": results, "macros": macro,
               "calls_used": _call_count}
    TEST_RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n[test] {summary['macros']}")
    return summary


def _baseline_from_cache(project: str, variant: str) -> float | None:
    """Read final.pkl from phase_cache and score against gold."""
    cache_path = Path(f"./results/phase_cache/{variant}/{project}/final.pkl")
    if not cache_path.exists():
        return None
    import pickle
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
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["train", "distill", "test", "all"])
    ap.add_argument("--backend", default="claude")
    ap.add_argument("--model", default=None,
                    help="Optional model override (e.g. 'gpt-5.4', 'sonnet').")
    ap.add_argument("--resume", action="store_true",
                    help="Resume training: keep existing skill_bank + trajectory, "
                         "offset outer index past prior passes.")
    args = ap.parse_args(argv)
    backend = {
        "claude": LLMBackend.CLAUDE,
        "openai": LLMBackend.OPENAI,
        "checkpoint": LLMBackend.CHECKPOINT,
    }[args.backend]

    if args.phase in ("train", "all"):
        train(backend, model=args.model, resume=args.resume)
    if args.phase in ("distill", "all"):
        distill(backend, model=args.model)
    if args.phase in ("test", "all"):
        test(backend, model=args.model)
    print(f"\n[done] total calls={_call_count} elapsed={time.time()-_start_time:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
