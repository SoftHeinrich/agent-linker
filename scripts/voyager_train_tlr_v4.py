"""Voyager-TLR v4 — multi-role training harness (5 roles R1-R5).

v2.2 PROBE WAVE / Phase 14 mechanism. Builds the 5-role harness from the
``voyager-v4-architecture-proposal.md`` spec:

  R1 — Linker (proposer)              [sees: doc + axioms + linker_skills]
  R2 — Validator (judge)              [sees: doc + axioms + validator_skills + R1]
  R3 — Skill Distillator              [sees: categorical signal + skill banks]
  R4 — Feedback Judge (oracle)        [sees: gold + R1 + R2 IDs only — NO doc]
  R5 — Abstraction Validator          [sees: pattern + textbook style library]

Design departures from voyager v2/v3 (single-role):
- Feedback channel is CATEGORICAL (no raw FP/FN sentences leak to R3).
- Skill banks are PER-ROLE (linker_skills.json + validator_skills.json).
- R5 evaluates every R3-proposed pattern against 5 architectural styles
  before it can enter a skill bank.

Probe scope per v2.2-prep user directive:
- ONE training outer pass on mediastore only.
- Backend: gpt-5.4 (Sonnet 4.6 mapping).
- Both skill banks start EMPTY.
- Measurements: did R5 accept any patterns? Do linker_skills vs
  validator_skills differ meaningfully? Did F1 lift over axiom-only?

CLI
---
    python scripts/voyager_train_tlr_v4.py probe \\
        --project mediastore \\
        --backend openai --model gpt-5.4

Outputs (under ``results/v2_2_probes/A_voyager_v4/probe_mediastore/``):
    iter0_axiom_only_results.json        — baseline (skill banks empty)
    r1_predictions.json                  — R1 linker output
    r2_verdicts.json                     — R2 validator output
    r4_categorical_signal.json           — R4 feedback (NO raw text)
    r3_proposed_patterns.json            — R3 distillator output (pre-R5)
    r5_abstraction_verdicts.json         — R5 per-pattern verdicts
    linker_skills.json                   — accepted linker patterns
    validator_skills.json                — accepted validator patterns
    iter1_after_skills_results.json      — F1 with skill banks populated
    probe_summary.json                   — top-level: F1 delta, R5 accept rate

Budget: ~5 LLM calls per outer iter × 1-3 iters ~= $10-15 envelope.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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


# ─────────────────────────────────────────────────────────────────────────────
# Output dirs
# ─────────────────────────────────────────────────────────────────────────────

OUT_ROOT = Path("results/v2_2_probes/A_voyager_v4")


def _project_dir(project: str) -> Path:
    return OUT_ROOT / f"probe_{project}"


def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# GATE-06 (mirror of v2/v3)
# ─────────────────────────────────────────────────────────────────────────────

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


def _gate06_ok(text: str) -> tuple[bool, list[str]]:
    hits = TABOO_PATTERN.findall(text or "")
    return (len(hits) == 0, hits)


# ─────────────────────────────────────────────────────────────────────────────
# Budget tracking
# ─────────────────────────────────────────────────────────────────────────────

_call_count = 0
_start_time = time.time()
MAX_CALLS = int(os.environ.get("VOYAGER4_MAX_CALLS", "30"))
WALL_BUDGET_SECONDS = int(os.environ.get("VOYAGER4_WALL_BUDGET_S", str(3 * 3600)))


def _budget_ok() -> bool:
    return _call_count < MAX_CALLS and (time.time() - _start_time) < WALL_BUDGET_SECONDS


def _bump(reason: str = "") -> None:
    global _call_count
    _call_count += 1
    print(f"  [budget] calls={_call_count}/{MAX_CALLS} {reason}")


# ─────────────────────────────────────────────────────────────────────────────
# R1 + R2 are the existing linker.
#
# We treat one SLinker13SkillLearned.link() call as one "R1+R2 unit": the
# linker internally runs proposer+judge stages. The split is logical (we
# split the SKILL BANKS into linker_skills + validator_skills) rather than
# physical (we don't re-architect the linker into two separate models).
#
# This is the pragmatic minimum that still tests the v4 hypothesis: if
# splitting skill banks per role + categorical R4 feedback + R5 abstraction
# validation produces different outcomes than v2/v3, the architecture-only
# change is informative even though R1 and R2 share a backend.
# ─────────────────────────────────────────────────────────────────────────────


def _run_linker(project: str, backend: LLMBackend, skill_path: str,
                model: str | None = None) -> dict:
    paths = _ra.DATASETS[project]
    linker = SLinker13SkillLearned(
        backend=backend, model=model, skill_path=skill_path)
    t0 = time.time()
    links = linker.link(
        text_path=str(paths["text"]),
        model_path=str(paths["model"]),
    )
    elapsed = time.time() - t0
    _bump(f"R1+R2 linker on {project}")

    predicted = {(lk.sentence_number, lk.component_id) for lk in links}
    gold = _ra.load_gold_sam(str(paths["gold_sam"]))
    metrics = _ra.eval_metrics(predicted, gold)

    from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
    id_to_name = {lk.component_id: lk.component_name for lk in links}
    for c in parse_pcm_repository(str(paths["model"])):
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
        "fps_id_only": [(s, c) for s, c in fps],
        "fns_id_only": [(s, c) for s, c in fns],
        # NOTE: we keep id_to_name accessible to R1 prompts but the
        # categorical R4 channel deliberately does NOT pass it on.
        "id_to_name": id_to_name,
        "predicted_count": len(predicted),
        "gold_count": len(gold),
    }


# ─────────────────────────────────────────────────────────────────────────────
# R4 — Feedback Judge (oracle role; ID-only view)
# ─────────────────────────────────────────────────────────────────────────────
#
# R4 sees: per-error tuples of (sentence_id, component_abstract_id, gold_yes/no,
# linker_predicted, validator_kept). It does NOT see component names or
# sentence text. It can only group errors into CATEGORIES based on the
# structural distribution.

R4_PROMPT = """You are the FEEDBACK JUDGE in a 5-role training loop for a software
architecture trace-link recovery pipeline. The linker (R1) and validator (R2)
have just produced predictions; the gold standard has been compared.

CRITICAL CONSTRAINT: you are an oracle role, but you have a RESTRICTED VIEW.
You CANNOT see the document text, the component names, or the sentence
strings. You see ONLY:
  - structural IDs (sentence id, component id mapped to abstract code)
  - aggregate error counts
  - the per-error category breakdown you yourself categorize

Your task: categorize the errors into ABSTRACT ERROR CLASSES. Use general
TLR error vocabulary, NOT project-specific labels.

Suggested abstract categories (you may invent more, but stay general):

LINKER ERROR CLASSES (false negatives, things linker missed):
- "abbrev_expansion_missed": linker missed an alias defined parenthetically
- "alias_synonym_missed": linker missed a multi-word descriptive synonym
- "containment_missed": linker missed that a sub-element should link to its parent
- "implicit_subject_missed": linker missed a sentence whose implicit subject is the component
- "passive_voice_missed": linker missed a sentence where component is the passive agent

VALIDATOR ERROR CLASSES (false positives, things validator over-approved):
- "tech_label_over_approved": validator approved a sentence where the name appears as a tech label
- "pattern_name_over_approved": validator approved when the name describes a design pattern, not the component
- "subprocess_over_approved": validator approved an algorithm/implementation that shares the component's name
- "ambiguous_over_approved": validator approved a generic single-word reference without sufficient evidence

INPUTS
======
project_id: {project_id}
linker_fp_total: {fp_total}
linker_fn_total: {fn_total}
fp_id_tuples: {fp_id_tuples}
fn_id_tuples: {fn_id_tuples}
total_components: {component_count}
total_sentences: {sentence_count}

For each error category that applies, report:
  - the abstract category name (snake_case)
  - the count
  - the abstract ROLE the category attributes to: "linker" or "validator"
  - one-sentence ABSTRACT advice (textbook SE vocabulary; no project terms)

Return JSON:
{{
  "categorical_signal": {{
     "linker_error_categories": [{{"category": "...", "count": N, "advice": "..."}}],
     "validator_error_categories": [{{"category": "...", "count": N, "advice": "..."}}]
  }},
  "summary_stats": {{
     "linker_fp_count": {fp_total}, "linker_fn_count": {fn_total}
  }}
}}
JSON only:"""


def _r4_feedback_judge(llm: LLMClient, project: str, run: dict) -> dict:
    """Categorize R1+R2 errors WITHOUT seeing document text or component names."""
    from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
    from llm_sad_sam.core.document_loader_v2 import load_sentences

    paths = _ra.DATASETS[project]
    components = parse_pcm_repository(str(paths["model"]))
    sentences = load_sentences(str(paths["text"]))

    # Build abstract component IDs (comp_0, comp_1, ...). Map real IDs
    # through the abstraction so R4 cannot reverse-engineer component names.
    id_to_abstract = {c.id: f"comp_{i}" for i, c in enumerate(components)}

    fp_tuples = [
        (s, id_to_abstract.get(c, "comp_?")) for s, c in run["fps_id_only"]
    ]
    fn_tuples = [
        (s, id_to_abstract.get(c, "comp_?")) for s, c in run["fns_id_only"]
    ]

    prompt = R4_PROMPT.format(
        project_id="project_under_test",
        fp_total=len(fp_tuples),
        fn_total=len(fn_tuples),
        fp_id_tuples=json.dumps(fp_tuples),
        fn_id_tuples=json.dumps(fn_tuples),
        component_count=len(components),
        sentence_count=len(sentences),
    )

    # Defensive GATE-06 check: the R4 prompt itself contains NO benchmark terms
    # by construction (we strip names before sending). Verify.
    ok, hits = _gate06_ok(prompt)
    if not ok:
        raise ValueError(
            f"R4 prompt contains taboo tokens {hits!r} — restricted-view "
            "contract violated; check abstraction logic."
        )

    response = llm.query(prompt, timeout=180)
    _bump("R4 feedback judge")
    data = llm.extract_json(response) or {}
    if not isinstance(data, dict):
        data = {}
    return data


# ─────────────────────────────────────────────────────────────────────────────
# R3 — Skill Distillator (categorical-only input)
# ─────────────────────────────────────────────────────────────────────────────

R3_PROMPT = """You are the SKILL DISTILLATOR in a 5-role TLR training loop.
You receive CATEGORICAL ERROR CLASSES from the feedback judge (R4). You DO
NOT see document text, component names, or raw error sentences. Your job:
turn each error class into ABSTRACT skill patterns for the linker (R1) and
the validator (R2) skill banks.

CONSTRAINTS:
- One pattern per category at most; skip if no actionable advice is possible.
- Use textbook SE vocabulary ONLY (lexer, parser, scheduler, broker,
  dispatcher, controller, queue, monitor, pipeline). NEVER use any name
  resembling a specific project component.
- Each pattern must be assignable to exactly one role: "linker" (advice on
  proposing more/fewer candidates) or "validator" (advice on approving/
  rejecting candidates).
- Each pattern should be 1 sentence, declarative, abstract.

CURRENT SKILL BANKS (already-derived; do not duplicate):
LINKER skills:
{linker_skills_block}
VALIDATOR skills:
{validator_skills_block}

CATEGORICAL SIGNAL FROM R4:
{categorical_signal}

Return JSON:
{{
  "proposed_patterns": [
    {{"role": "linker"|"validator", "category": "<from R4>", "pattern": "<abstract sentence>"}}
  ]
}}
JSON only:"""


def _r3_distillator(
    llm: LLMClient,
    categorical_signal: dict,
    linker_skills: list[dict],
    validator_skills: list[dict],
) -> list[dict]:
    def _fmt(skills):
        if not skills:
            return "  (empty)"
        return "\n".join(f"  - {p.get('pattern')}" for p in skills)

    prompt = R3_PROMPT.format(
        linker_skills_block=_fmt(linker_skills),
        validator_skills_block=_fmt(validator_skills),
        categorical_signal=json.dumps(categorical_signal, indent=2),
    )

    ok, hits = _gate06_ok(prompt)
    if not ok:
        raise ValueError(f"R3 prompt contains taboo tokens {hits!r}")

    data = llm.extract_json(llm.query(prompt, timeout=180)) or {}
    _bump("R3 distillator")
    patterns = data.get("proposed_patterns", []) if isinstance(data, dict) else []
    out = []
    for p in patterns:
        if not isinstance(p, dict):
            continue
        role = p.get("role")
        pattern = p.get("pattern")
        if role in ("linker", "validator") and isinstance(pattern, str) and pattern.strip():
            # Audit each proposed pattern individually.
            ok, hits = _gate06_ok(pattern)
            if not ok:
                print(f"  [R3] dropped pattern (GATE-06 hits {hits!r}): {pattern[:80]!r}")
                continue
            out.append({
                "role": role,
                "category": p.get("category", ""),
                "pattern": pattern,
            })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# R5 — Abstraction Validator
# ─────────────────────────────────────────────────────────────────────────────
# 5 textbook architectural styles. R5 asks: would this pattern give the SAME
# accept/reject decisions across all 5 styles? If style-neutral → ACCEPT.

R5_STYLE_LIBRARY = """ARCHITECTURAL STYLES TO TEST AGAINST:
  1. Microservice mesh — services communicate via async events; no shared DB.
  2. Event-sourced system — state derived from event stream; no current-state tables.
  3. Layered monolith — controller -> service -> repository -> DB.
  4. Pipe-and-filter — data transforms in series; no central state.
  5. Hexagonal / ports-and-adapters — domain core surrounded by adapters.
"""


R5_PROMPT = """You are the ABSTRACTION VALIDATOR in a 5-role TLR training loop.
A proposed skill pattern must be tested for STYLE NEUTRALITY before it can
enter a skill bank.

{style_library}

PROPOSED PATTERN ({role}):
"{pattern}"

For each style, would this pattern produce the SAME accept/reject decision?
- If yes for all 5 styles → "verdict": "ACCEPT" and "style_dependency": null.
- If the pattern depends on style-specific vocabulary or assumptions
  (e.g., requires there be a "queue" or a "controller") → "verdict": "REJECT"
  and "style_dependency": <the style that breaks neutrality>.

Return JSON:
{{
  "verdict": "ACCEPT" | "REJECT",
  "reason": "<one sentence>",
  "style_dependency": "<style name>" | null
}}
JSON only:"""


def _r5_abstraction_validator(llm: LLMClient, role: str, pattern: str) -> dict:
    prompt = R5_PROMPT.format(
        style_library=R5_STYLE_LIBRARY,
        role=role,
        pattern=pattern,
    )
    data = llm.extract_json(llm.query(prompt, timeout=120)) or {}
    _bump(f"R5 abstraction check on {role}")
    if not isinstance(data, dict):
        return {"verdict": "REJECT", "reason": "malformed R5 response", "style_dependency": None}
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Skill-bank IO (per-role)
# ─────────────────────────────────────────────────────────────────────────────


def _bank_path(out_dir: Path, role: str) -> Path:
    return out_dir / f"{role}_skills.json"


def _load_bank(out_dir: Path, role: str) -> list[dict]:
    p = _bank_path(out_dir, role)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, ValueError):
        return []
    return list(data.get("abstract_patterns", []))


def _save_bank(out_dir: Path, role: str, patterns: list[dict]) -> None:
    p = _bank_path(out_dir, role)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"abstract_patterns": patterns}, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# SLinker13SkillLearned reads ONE skill bank. To keep the probe simple we
# concatenate linker_skills + validator_skills into a unified skill bank for
# the inference-time linker. The PER-ROLE separation matters for TRAINING
# (R3 emits role-tagged patterns, R5 audits each); at inference the
# distinction is currently informational only. This is a v4 limitation we
# document — full per-role inference-time separation requires re-architecting
# the linker, which is out of probe scope.
# ─────────────────────────────────────────────────────────────────────────────


def _merge_banks_for_inference(out_dir: Path, linker_skills: list[dict],
                                validator_skills: list[dict]) -> Path:
    merged = []
    for p in linker_skills:
        merged.append({
            "prompt": p.get("prompt", ""),
            "pattern": f"[LINKER] {p.get('pattern')}",
        })
    for p in validator_skills:
        merged.append({
            "prompt": p.get("prompt", ""),
            "pattern": f"[VALIDATOR] {p.get('pattern')}",
        })
    merged_path = out_dir / "merged_skill_bank.json"
    merged_path.write_text(json.dumps({"abstract_patterns": merged}, indent=2))
    return merged_path


# ─────────────────────────────────────────────────────────────────────────────
# Outer training pass
# ─────────────────────────────────────────────────────────────────────────────


def run_probe(project: str, backend_name: str, model: str | None) -> dict:
    """Run ONE outer training pass on a single project + write all artifacts."""
    out_dir = _ensure(_project_dir(project))
    backend = LLMBackend.OPENAI if backend_name.lower() == "openai" else LLMBackend.CLAUDE

    # Reset skill banks to empty (Probe A directive: fresh start).
    _save_bank(out_dir, "linker", [])
    _save_bank(out_dir, "validator", [])
    empty_path = _merge_banks_for_inference(out_dir, [], [])

    print(f"\n[v4 PROBE] project={project} backend={backend_name} model={model}")

    # ─────── ITER 0: axiom-only baseline ───────
    print("\n--- iter 0: axiom-only baseline ---")
    iter0 = _run_linker(project, backend, str(empty_path), model=model)
    (out_dir / "iter0_axiom_only_results.json").write_text(json.dumps({
        "project": iter0["project"],
        "F1": iter0["F1"], "P": iter0["P"], "R": iter0["R"],
        "fp_count": iter0["fp_count"], "fn_count": iter0["fn_count"],
        "elapsed_s": iter0["elapsed_s"],
    }, indent=2))
    print(f"  iter0 F1={iter0['F1']:.4f} fp={iter0['fp_count']} fn={iter0['fn_count']}")

    # ─────── R4: categorical feedback (ID-only) ───────
    print("\n--- R4 feedback judge ---")
    llm = LLMClient(backend=backend, model=model)
    r4_signal = _r4_feedback_judge(llm, project, iter0)
    (out_dir / "r4_categorical_signal.json").write_text(json.dumps(r4_signal, indent=2))
    n_linker_classes = len(
        (r4_signal.get("categorical_signal", {}) or {}).get("linker_error_categories", []))
    n_validator_classes = len(
        (r4_signal.get("categorical_signal", {}) or {}).get("validator_error_categories", []))
    print(f"  R4 emitted {n_linker_classes} linker error classes, {n_validator_classes} validator error classes")

    # ─────── R3: distill patterns from categorical signal ───────
    print("\n--- R3 skill distillator ---")
    linker_skills = _load_bank(out_dir, "linker")
    validator_skills = _load_bank(out_dir, "validator")
    proposed = _r3_distillator(llm, r4_signal, linker_skills, validator_skills)
    (out_dir / "r3_proposed_patterns.json").write_text(
        json.dumps({"proposed_patterns": proposed}, indent=2)
    )
    print(f"  R3 proposed {len(proposed)} patterns")
    for p in proposed:
        print(f"    [{p['role']}] ({p['category']}) {p['pattern'][:120]}")

    # ─────── R5: per-pattern abstraction validation ───────
    print("\n--- R5 abstraction validator ---")
    r5_results = []
    accepted_linker = list(linker_skills)
    accepted_validator = list(validator_skills)
    for p in proposed:
        if not _budget_ok():
            print("  [budget] exhausted — skipping remaining R5 calls")
            break
        verdict = _r5_abstraction_validator(llm, p["role"], p["pattern"])
        r5_results.append({**p, "r5_verdict": verdict})
        if verdict.get("verdict") == "ACCEPT":
            target = accepted_linker if p["role"] == "linker" else accepted_validator
            target.append({"prompt": "", "pattern": p["pattern"], "category": p["category"]})
        else:
            print(f"  R5 REJECTED ({p['role']}): {verdict.get('reason', '?')[:120]}")
    (out_dir / "r5_abstraction_verdicts.json").write_text(
        json.dumps({"results": r5_results}, indent=2)
    )

    # Save updated per-role banks.
    _save_bank(out_dir, "linker", accepted_linker)
    _save_bank(out_dir, "validator", accepted_validator)
    merged_path = _merge_banks_for_inference(out_dir, accepted_linker, accepted_validator)

    n_r5_accept = sum(1 for r in r5_results if r["r5_verdict"].get("verdict") == "ACCEPT")
    n_r5_reject = sum(1 for r in r5_results if r["r5_verdict"].get("verdict") == "REJECT")
    r5_reject_rate = n_r5_reject / max(1, len(r5_results))
    print(f"  R5 verdict distribution: ACCEPT={n_r5_accept} REJECT={n_r5_reject} "
          f"reject_rate={r5_reject_rate:.2%}")

    # ─────── ITER 1: linker re-run with populated skill banks ───────
    if accepted_linker or accepted_validator:
        if _budget_ok():
            print("\n--- iter 1: linker re-run with v4 skill banks ---")
            iter1 = _run_linker(project, backend, str(merged_path), model=model)
            (out_dir / "iter1_after_skills_results.json").write_text(json.dumps({
                "project": iter1["project"],
                "F1": iter1["F1"], "P": iter1["P"], "R": iter1["R"],
                "fp_count": iter1["fp_count"], "fn_count": iter1["fn_count"],
                "elapsed_s": iter1["elapsed_s"],
            }, indent=2))
            delta = iter1["F1"] - iter0["F1"]
            print(f"  iter1 F1={iter1['F1']:.4f} delta={delta:+.4f}")
        else:
            iter1 = None
            print("\n  [budget] exhausted — skipping iter1 re-run")
    else:
        iter1 = None
        print("\n  R5 accepted no patterns; skipping iter1 re-run (no skill bank change)")

    # ─────── Probe summary ───────
    linker_vs_validator_meaningfully_different = (
        len(accepted_linker) > 0 and len(accepted_validator) > 0
        and {p["pattern"] for p in accepted_linker} !=
        {p["pattern"] for p in accepted_validator}
    )

    summary = {
        "project": project,
        "backend": backend_name,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "r5_accept_count": n_r5_accept,
        "r5_reject_count": n_r5_reject,
        "r5_reject_rate": r5_reject_rate,
        "iter0_F1": iter0["F1"],
        "iter1_F1": iter1["F1"] if iter1 else None,
        "delta_F1_axiom_to_v4": (iter1["F1"] - iter0["F1"]) if iter1 else None,
        "linker_skill_count": len(accepted_linker),
        "validator_skill_count": len(accepted_validator),
        "linker_vs_validator_different": linker_vs_validator_meaningfully_different,
        "total_llm_calls": _call_count,
        "verdict_gates": {
            "r5_rejected_100pct": (n_r5_reject > 0 and n_r5_accept == 0),
            "f1_lift_over_axiom_>=_0.5pp": (
                iter1 is not None and (iter1["F1"] - iter0["F1"]) >= 0.005
            ),
        },
    }
    (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2))

    if summary["verdict_gates"]["r5_rejected_100pct"]:
        print(f"\n  [VERDICT] PROBE FAIL — R5 rejected 100% of R3 proposals")
    elif summary["verdict_gates"]["f1_lift_over_axiom_>=_0.5pp"]:
        print(f"\n  [VERDICT] PROBE PASS — F1 lift >= 0.5pp over axiom-only")
    else:
        print(f"\n  [VERDICT] PROBE WEAK — F1 lift < 0.5pp; documented for rollup")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _load_dotenv() -> None:
    env_file = _ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_probe = sub.add_parser("probe", help="Run ONE outer training pass")
    p_probe.add_argument("--project", default="mediastore")
    p_probe.add_argument("--backend", default="openai", choices=["openai", "claude"])
    p_probe.add_argument("--model", default="gpt-5.4")
    args = parser.parse_args(argv)

    if args.cmd == "probe":
        run_probe(args.project, args.backend, args.model)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
