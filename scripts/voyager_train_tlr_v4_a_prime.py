"""Voyager-TLR v4 — Probe A' (A-prime) vocab-aligned R3 prompt retry.

v2.2 RANGE-D-PROBE-A-PRIME wave (per user directive 2026-06-01).

PROBLEM FIXED FROM PROBE A
==========================
The original Probe A R5 abstraction validator rejected 100% of R3's
proposed patterns. Root cause: R3 was instructed to use textbook SE role
vocabulary ("controller, dispatcher, broker, queue, monitor, scheduler,
parser, lexer, pipeline"), but R5 tests style-neutrality against 5
architectural styles where ANY such role term is style-dependent (e.g.,
"queue" is meaningful in microservice mesh but not in layered monolith).
The two prompts had inconsistent abstraction specs.

PROBE A' FIX
============
TIGHTEN R3 vocabulary to LINGUISTIC/DISCOURSE terms only:
- ALLOWED: subject, predicate, anaphora, antecedent, parenthetical,
  namespace prefix, section heading, sentence-position, qualifier clause,
  cross-reference, coordinated clause, possessive, definite article,
  apposition, head noun, modifier, etc.
- FORBIDDEN: role nouns (controller, scheduler, broker, queue, dispatcher,
  parser, lexer, monitor, pipeline, etc.) AND architectural style names
  (microservice, event-sourced, layered, hexagonal, pipe-and-filter).
- Examples make this EXPLICIT in the prompt.

Expected outcome: patterns now read like
  "Resolve a candidate only when the name appears as the subject of the
   sentence, not within a qualifier or apposition."
which is style-neutral by construction (discourse-layer rules apply to
text regardless of architectural style).

The Claude v3 finding (s_linker13_clean) noted Claude naturally produces
patterns like "subject-position naming overrides predicate keywords" —
these are discourse-level rules that pass R5's universal-transferability
test.

CLI
---
    python scripts/voyager_train_tlr_v4_a_prime.py probe \\
        --project mediastore \\
        --backend openai --model gpt-5.4

OUTPUTS (under ``results/v2_2_probes_a_prime/probe_mediastore/``):
    iter0_axiom_only_results.json
    r1_predictions.json, r2_verdicts.json
    r4_categorical_signal.json
    r3_proposed_patterns.json
    r5_abstraction_verdicts.json
    linker_skills.json, validator_skills.json
    iter1_after_skills_results.json
    probe_summary.json
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
# Output dirs (separate from Probe A!)
# ─────────────────────────────────────────────────────────────────────────────

OUT_ROOT = Path(os.environ.get("PROBE_A_PRIME_OUT_ROOT", "results/v2_2_probes_a_prime"))


def _project_dir(project: str) -> Path:
    return OUT_ROOT / f"probe_{project}"


def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# GATE-06
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
# R1 + R2 = SLinker13SkillLearned (one .link() call = one R1+R2 unit)
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
        "id_to_name": id_to_name,
        "predicted_count": len(predicted),
        "gold_count": len(gold),
    }


# ─────────────────────────────────────────────────────────────────────────────
# R4 — Feedback Judge (oracle; ID-only view) — IDENTICAL to Probe A
# ─────────────────────────────────────────────────────────────────────────────

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
  - one-sentence ABSTRACT advice (linguistic/discourse vocabulary preferred over role-noun vocabulary)

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
# R3 — Skill Distillator — VOCAB-ALIGNED (Probe A' fix)
# ─────────────────────────────────────────────────────────────────────────────
#
# Change from Probe A: ban role nouns + architectural style names; require
# LINGUISTIC/DISCOURSE vocabulary. Include explicit acceptable / unacceptable
# example wordings.

R3_PROMPT_VOCAB_ALIGNED = """You are the SKILL DISTILLATOR in a 5-role TLR training loop.
You receive CATEGORICAL ERROR CLASSES from the feedback judge (R4). You DO
NOT see document text, component names, or raw error sentences. Your job:
turn each error class into ABSTRACT skill patterns for the linker (R1) and
the validator (R2) skill banks.

────────────────────────────────────────────────────────────────────────
VOCABULARY CONSTRAINTS (CRITICAL — patterns are rejected if violated)
────────────────────────────────────────────────────────────────────────

ALLOWED vocabulary (linguistic / discourse terms only):
  subject, predicate, object, head noun, modifier, qualifier clause,
  apposition, parenthetical, antecedent, anaphora, coreference,
  pronoun, definite article, possessive, coordinated clause,
  subordinate clause, sentence-position, section heading,
  namespace prefix, cross-reference, dotted path, capitalization,
  multi-word phrase, single-word reference, hyphenation,
  introducing sentence, follow-up sentence, exact-string match,
  partial-string match, alias defined parenthetically.

FORBIDDEN vocabulary (will cause R5 rejection):
  - ROLE NOUNS: controller, scheduler, broker, queue, dispatcher, parser,
    lexer, monitor, pipeline, adapter, facade, mediator, observer,
    interpreter, proxy, gateway, router, worker, repository, service.
  - ARCHITECTURAL STYLE NAMES: microservice, event-sourced, layered,
    hexagonal, pipe-and-filter, monolith, ports-and-adapters, mesh, MVC.
  - DOMAIN NOUNS: payment, user, file, media, storage, cache, database,
    session, account, request, response, page, form, message.

The reason: such vocabulary is style-dependent (a "queue" is meaningful
in microservice mesh but absent in pipe-and-filter). Discourse-level
vocabulary is style-neutral — discourse rules apply to any text
regardless of architectural style.

────────────────────────────────────────────────────────────────────────
EXAMPLE PATTERNS
────────────────────────────────────────────────────────────────────────

ACCEPTABLE (style-neutral, discourse-based):
  - "A linker should treat a component name appearing in the SUBJECT
    position of a sentence as a stronger signal than the same name
    appearing inside a qualifier clause or parenthetical."
  - "A validator should reject a candidate when the component name
    appears only inside a definite article phrase that introduces a new
    topic at a section heading rather than continuing the antecedent."
  - "A linker should propose a candidate when a multi-word phrase
    coordinated with a prior name extends a single antecedent across
    a coordinated clause."

UNACCEPTABLE (style-dependent, will be R5-rejected):
  - "A linker should propose candidates when a controller, dispatcher,
    or broker is mentioned." (uses role nouns)
  - "A validator should reject candidates in an event-sourced context."
    (uses architectural style name)
  - "A linker should propose when a request is routed through a queue."
    (uses domain noun + role noun)

────────────────────────────────────────────────────────────────────────
GENERAL CONSTRAINTS
────────────────────────────────────────────────────────────────────────

- One pattern per category at most; skip if no actionable advice is possible.
- Each pattern must be assignable to exactly one role: "linker" (advice on
  proposing more/fewer candidates) or "validator" (advice on approving/
  rejecting candidates).
- Each pattern: 1 declarative sentence, abstract, LINGUISTIC vocabulary.
- Never use any name resembling a specific project component.
- If you cannot phrase a pattern without forbidden vocabulary, SKIP it.

────────────────────────────────────────────────────────────────────────
CURRENT SKILL BANKS (already-derived; do not duplicate):
────────────────────────────────────────────────────────────────────────
LINKER skills:
{linker_skills_block}
VALIDATOR skills:
{validator_skills_block}

CATEGORICAL SIGNAL FROM R4:
{categorical_signal}

Return JSON:
{{
  "proposed_patterns": [
    {{"role": "linker"|"validator", "category": "<from R4>", "pattern": "<discourse-level sentence>"}}
  ]
}}
JSON only:"""


# Lightweight client-side audit to catch role/style nouns before sending to R5.
# This is informational (logged), not enforced — R5 is still the validator.
_FORBIDDEN_VOCAB_PATTERN = re.compile(
    r"(?i)\b("
    r"controller|scheduler|broker|queue|dispatcher|parser|lexer|monitor|"
    r"pipeline|adapter|facade|mediator|observer|interpreter|proxy|gateway|"
    r"router|worker|repository|microservice|event-sourced|event sourced|"
    r"layered|hexagonal|pipe-and-filter|pipe and filter|monolith|"
    r"ports-and-adapters|mesh\b|MVC"
    r")\b"
)


def _forbidden_vocab_hits(text: str) -> list[str]:
    return _FORBIDDEN_VOCAB_PATTERN.findall(text or "")


def _r3_distillator(
    llm: LLMClient,
    categorical_signal: dict,
    linker_skills: list[dict],
    validator_skills: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Return (kept_patterns, forbidden_vocab_warnings).

    Forbidden-vocab warnings are tracked but NOT enforced client-side; R5
    is the canonical validator. The warning list is recorded in the probe
    summary for diagnostic purposes.
    """
    def _fmt(skills):
        if not skills:
            return "  (empty)"
        return "\n".join(f"  - {p.get('pattern')}" for p in skills)

    prompt = R3_PROMPT_VOCAB_ALIGNED.format(
        linker_skills_block=_fmt(linker_skills),
        validator_skills_block=_fmt(validator_skills),
        categorical_signal=json.dumps(categorical_signal, indent=2),
    )

    ok, hits = _gate06_ok(prompt)
    if not ok:
        raise ValueError(f"R3 prompt contains taboo tokens {hits!r}")

    data = llm.extract_json(llm.query(prompt, timeout=180)) or {}
    _bump("R3 distillator (vocab-aligned)")
    patterns = data.get("proposed_patterns", []) if isinstance(data, dict) else []
    out = []
    warnings = []
    for p in patterns:
        if not isinstance(p, dict):
            continue
        role = p.get("role")
        pattern = p.get("pattern")
        if role in ("linker", "validator") and isinstance(pattern, str) and pattern.strip():
            # GATE-06 (benchmark taboo)
            ok, hits = _gate06_ok(pattern)
            if not ok:
                print(f"  [R3] dropped pattern (GATE-06 hits {hits!r}): {pattern[:80]!r}")
                continue
            # Diagnostic vocab audit (do not drop — R5 is canonical validator).
            vocab_hits = _forbidden_vocab_hits(pattern)
            if vocab_hits:
                print(f"  [R3] forbidden-vocab warning {vocab_hits!r}: {pattern[:80]!r}")
                warnings.append({
                    "role": role,
                    "pattern": pattern,
                    "forbidden_vocab_hits": vocab_hits,
                })
            out.append({
                "role": role,
                "category": p.get("category", ""),
                "pattern": pattern,
            })
    return out, warnings


# ─────────────────────────────────────────────────────────────────────────────
# R5 — Abstraction Validator — IDENTICAL to Probe A (same 5-style library)
# ─────────────────────────────────────────────────────────────────────────────

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
    out_dir = _ensure(_project_dir(project))
    backend = LLMBackend.OPENAI if backend_name.lower() == "openai" else LLMBackend.CLAUDE

    _save_bank(out_dir, "linker", [])
    _save_bank(out_dir, "validator", [])
    empty_path = _merge_banks_for_inference(out_dir, [], [])

    print(f"\n[A' PROBE] project={project} backend={backend_name} model={model}")

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

    # ─────── R3: distill patterns from categorical signal (vocab-aligned!) ───────
    print("\n--- R3 skill distillator (vocab-aligned) ---")
    linker_skills = _load_bank(out_dir, "linker")
    validator_skills = _load_bank(out_dir, "validator")
    proposed, vocab_warnings = _r3_distillator(llm, r4_signal, linker_skills, validator_skills)
    (out_dir / "r3_proposed_patterns.json").write_text(
        json.dumps({"proposed_patterns": proposed,
                    "forbidden_vocab_warnings": vocab_warnings}, indent=2)
    )
    print(f"  R3 proposed {len(proposed)} patterns "
          f"({len(vocab_warnings)} with forbidden-vocab warnings)")
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

    _save_bank(out_dir, "linker", accepted_linker)
    _save_bank(out_dir, "validator", accepted_validator)
    merged_path = _merge_banks_for_inference(out_dir, accepted_linker, accepted_validator)

    n_r5_accept = sum(1 for r in r5_results if r["r5_verdict"].get("verdict") == "ACCEPT")
    n_r5_reject = sum(1 for r in r5_results if r["r5_verdict"].get("verdict") == "REJECT")
    r5_reject_rate = n_r5_reject / max(1, len(r5_results))
    print(f"  R5 verdict distribution: ACCEPT={n_r5_accept} REJECT={n_r5_reject} "
          f"reject_rate={r5_reject_rate:.2%}")

    # ─────── ITER 1: linker re-run with populated skill banks ───────
    iter1 = None
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
            print("\n  [budget] exhausted — skipping iter1 re-run")
    else:
        print("\n  R5 accepted no patterns; skipping iter1 re-run (no skill bank change)")

    # ─────── Probe summary ───────
    linker_vs_validator_meaningfully_different = (
        len(accepted_linker) > 0 and len(accepted_validator) > 0
        and {p["pattern"] for p in accepted_linker} !=
        {p["pattern"] for p in accepted_validator}
    )

    # Verdict gates per user directive:
    #   R5 reject < 30% AND F1 lift >= +0.5pp -> STRONG_PASS A'
    #   R5 reject < 30% AND F1 lift <  +0.5pp -> WEAK_PASS A'
    #   R5 reject >= 30% -> FAIL A'
    delta_f1 = (iter1["F1"] - iter0["F1"]) if iter1 else None
    if r5_reject_rate < 0.30 and delta_f1 is not None and delta_f1 >= 0.005:
        verdict = "STRONG_PASS"
    elif r5_reject_rate < 0.30:
        verdict = "WEAK_PASS"
    else:
        verdict = "FAIL"

    summary = {
        "probe": "A_prime",
        "project": project,
        "backend": backend_name,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "r5_accept_count": n_r5_accept,
        "r5_reject_count": n_r5_reject,
        "r5_reject_rate": r5_reject_rate,
        "iter0_F1": iter0["F1"],
        "iter1_F1": iter1["F1"] if iter1 else None,
        "delta_F1_axiom_to_v4": delta_f1,
        "linker_skill_count": len(accepted_linker),
        "validator_skill_count": len(accepted_validator),
        "linker_vs_validator_different": linker_vs_validator_meaningfully_different,
        "total_llm_calls": _call_count,
        "forbidden_vocab_warnings_count": len(vocab_warnings),
        "verdict": verdict,
        "verdict_gates": {
            "r5_reject_under_30pct": (r5_reject_rate < 0.30),
            "f1_lift_over_axiom_>=_0.5pp": (
                iter1 is not None and (iter1["F1"] - iter0["F1"]) >= 0.005
            ),
        },
    }
    (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  [VERDICT A'] {verdict}")
    print(f"    R5 reject rate: {r5_reject_rate:.2%}")
    if delta_f1 is not None:
        print(f"    F1 lift over axiom: {delta_f1:+.4f}")
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
        # Propagate env to subprocesses (LLMClient picks these up)
        os.environ["LLM_BACKEND"] = args.backend
        os.environ["OPENAI_MODEL_NAME"] = args.model
        if args.backend == "claude":
            os.environ["CLAUDE_MODEL"] = args.model
        run_probe(args.project, args.backend, args.model)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
