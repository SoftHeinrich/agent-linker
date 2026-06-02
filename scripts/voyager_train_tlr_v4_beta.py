"""Voyager-TLR v4 β — Multi-Role Training Harness.

Implements the β architecture (L + O + D-with-CoT-A + P) defined in
.planning/v2.3-prep/v2.3-ARCHITECTURE.md.

ROLES
-----
L  — Linker: runs s_linker14_voyager with current bank state.
O  — Oracle: text-aware error analysis → rich failure-mode JSON.
     Sees gold + L predictions + sentence text + current bank state.
     Does NOT produce bank entries directly.
D  — Distillator: text-blind, CoT-A inline → per-slot pattern proposals.
     Sees O's failure-mode JSON + current bank. Never sees raw text.
     Abstraction check (CoT-A) required before each proposed pattern.
Gate A — FM citation check: deterministic, $0. Rejects D proposals that do not
     cite at least one valid failure-mode ID from O's output for this pass.
Gate B — LLM dual-direction judge: ~$0.01/pass. Accepts only proposals where
     fixes_cited_fm=true AND causes_new_error=false AND confidence in {high, medium}.

ITERATION LOOP (per outer pass)
--------------------------------
  for project in training_set:
    L_run → O_json → D_proposals → GATE-06 filter → Gate A (FM citation)
                                                   → Gate B (LLM judge)
  commit if any accepted; else no-op
  convergence_check (FP+FN plateau: errors not improving vs prior pass, OR pass 5 cap)

DRY-RUN MODE
------------
  --dry-run: exercises L→O→D→P loop without any LLM calls.
  O returns mock failure modes; D returns mock pattern proposals.
  Bank state remains empty; P computes delta = 0.0 (no-op accept).
  Success criteria 1 (Phase 14): runs end-to-end with no errors.

CACHE
-----
Per-(text_stem, comp_hash, backend, model) on-disk cache for O and D
LLM outputs. Root: VOYAGER4B_CACHE_ROOT env var (default below).

GATE-06
-------
gate06_ok(text) → (bool, list[str]): benchmark-taboo grep.
reviewer_critic_stub(pattern, slot) → dict: Phase 14 stub (no LLM call).

CLI
---
  python scripts/voyager_train_tlr_v4_beta.py probe \\
      --projects mediastore --backend openai --model gpt-5.4

  python scripts/voyager_train_tlr_v4_beta.py range \\
      --projects mediastore,teastore,teammates \\
      --backend openai --model gpt-5.4

  python scripts/voyager_train_tlr_v4_beta.py probe \\
      --projects mediastore --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
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
import run_ablation as _ra


# ─────────────────────────────────────────────────────────────────────────────
# Paths + constants
# ─────────────────────────────────────────────────────────────────────────────

CACHE_ROOT = Path(os.environ.get("VOYAGER4B_CACHE_ROOT", "results/voyager_v4_beta/cache"))
OUT_ROOT = Path(os.environ.get("VOYAGER4B_OUT_ROOT", "results/voyager_v4_beta"))

MAINLINE_TRAIN = ["mediastore", "teastore", "teammates"]
MAINLINE_TEST = ["bigbluebutton", "jabref"]
MAX_OUTER_PASSES = 5
MIN_COMMIT_DELTA = 0.005  # REQ-V25-02: filters ±3-4pp BBB LLM run-to-run variance

SLOT_NAMES = (
    "AMBIGUITY_FEW_SHOT",
    "AMBIGUITY_RULES",
    "DOC_KNOWLEDGE_EXTRACTION_RULES",
    "DOC_KNOWLEDGE_JUDGE_EXAMPLES",
    "DOC_KNOWLEDGE_JUDGE_RULES",
    "ENTITY_EXTRACTION_RULES",
    "VALIDATION_RULES",
    "COREF_RULES",
    "SEED_DISAMBIGUATION_RULES",
    # 15-slot expansion (REQ-V25-07)
    "SEED_EXTRACTION_RULES",
    "SEED_ACTOR_RULES",
    "GENERIC_WORD_USAGE_RULES",
    "ALIAS_SCOPE_RULES",
    "ANTECEDENT_ALIAS_RULES",
    "COREF_TERMINAL_SPECIFICITY_RULES",
)


# ─────────────────────────────────────────────────────────────────────────────
# GATE-06 helpers (callable + unit-tested via tests/test_s_linker14_voyager_registration.py)
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


def gate06_ok(text: str) -> tuple[bool, list[str]]:
    """Return (passes, taboo_hits) for a text string.

    Used at bank-entry boundary to verify patterns contain no benchmark tokens.
    """
    hits = TABOO_PATTERN.findall(text or "")
    return (len(hits) == 0, hits)


def reviewer_critic_stub(pattern: str, slot: str) -> dict:
    """Phase 14 stub — returns advisory ACCEPT for all non-empty patterns.

    Real LLM-based critic activates in Phase 15+. The stub exercises the
    calling contract so tests can verify the helpers are callable.

    Returns:
        {"verdict": "ACCEPT" | "REJECT", "reason": str, "advisory": bool}
    """
    if not pattern.strip():
        return {"verdict": "REJECT", "reason": "empty pattern", "advisory": True}
    ok, hits = gate06_ok(pattern)
    if not ok:
        return {"verdict": "REJECT", "reason": f"taboo tokens {hits!r}", "advisory": True}
    return {"verdict": "ACCEPT", "reason": "stub advisory (Phase 14)", "advisory": True}


# ─────────────────────────────────────────────────────────────────────────────
# Cache adapter — per-(text_stem, comp_hash, backend, model)
# ─────────────────────────────────────────────────────────────────────────────

def _comp_hash(project: str) -> str:
    """Compute a hash of the component list for the project.

    Based on the PCM model path; stable across runs for the same model file.
    """
    paths = _ra.DATASETS[project]
    model_path = str(paths["model"])
    try:
        data = Path(model_path).read_bytes()
        return hashlib.sha256(data).hexdigest()[:16]
    except FileNotFoundError:
        return hashlib.sha256(model_path.encode()).hexdigest()[:16]


def _bank_content_hash(bank: dict) -> str:
    """Hash of bank slot_patterns content + axiom file — L cache key component.

    Includes axiom hash so cache invalidates when prompts_v3_axiom.py changes.
    """
    content = json.dumps(bank.get("slot_patterns", {}), sort_keys=True)
    bank_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    axiom_path = _ROOT / "src" / "llm_sad_sam" / "linkers" / "experimental" / "prompts_v3_axiom.py"
    try:
        axiom_hash = hashlib.md5(axiom_path.read_bytes()).hexdigest()[:6]
    except FileNotFoundError:
        axiom_hash = "noaxiom"
    return f"{bank_hash}_{axiom_hash}"


def _cache_key(text_path: str, project: str, backend: str, model: str, role: str) -> str:
    text_stem = Path(text_path).stem
    ch = _comp_hash(project)
    return f"{text_stem}_{ch}_{backend}_{model}_{role}"


def _cache_path(key: str) -> Path:
    root = Path(os.environ.get("VOYAGER4B_CACHE_ROOT", str(CACHE_ROOT)))
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{key}.json"


def _cache_read(key: str) -> dict | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, ValueError):
        return None


def _cache_write(key: str, data: dict) -> None:
    p = _cache_path(key)
    p.write_text(json.dumps(data, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Bank I/O
# ─────────────────────────────────────────────────────────────────────────────

def _bank_path(split_dir: Path, project: str) -> Path:
    return split_dir / f"{project}_bank.json"


def _load_bank(split_dir: Path, project: str) -> dict:
    """Load per-project bank. Returns empty slot-uniform bank on missing."""
    p = _bank_path(split_dir, project)
    if not p.exists():
        return {"version": "v4b", "project": project, "slot_patterns": {s: [] for s in SLOT_NAMES}}
    try:
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            sp = data.setdefault("slot_patterns", {})
            for s in SLOT_NAMES:
                sp.setdefault(s, [])
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return {"version": "v4b", "project": project, "slot_patterns": {s: [] for s in SLOT_NAMES}}


def _save_bank(split_dir: Path, project: str, bank: dict) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    _bank_path(split_dir, project).write_text(json.dumps(bank, indent=2))


def _total_patterns(bank: dict) -> int:
    return sum(len(v) for v in bank.get("slot_patterns", {}).values())


def _next_pattern_id(bank: dict) -> str:
    all_ids = [
        p.get("pattern_id", "p_000")
        for slot in bank.get("slot_patterns", {}).values()
        for p in slot
    ]
    nums = []
    for pid in all_ids:
        m = re.match(r"p_(\d+)", pid)
        if m:
            nums.append(int(m.group(1)))
    return f"p_{(max(nums) + 1 if nums else 1):03d}"


# ─────────────────────────────────────────────────────────────────────────────
# L role — run linker and compute metrics
# ─────────────────────────────────────────────────────────────────────────────

def _run_linker_l(project: str, backend: LLMBackend, model: str | None,
                  bank: dict, dry_run: bool = False) -> dict:
    """Run L on project with current bank state. Returns metrics dict.

    Cached per (project, bank_content_hash, backend, model) to avoid
    re-running L with identical bank state across passes or attempts.
    """
    from llm_sad_sam.linkers.experimental.s_linker14_voyager import SLinker14Voyager

    paths = _ra.DATASETS[project]
    import tempfile, json as _json

    if dry_run:
        gold = _ra.load_gold_sam(str(paths["gold_sam"]))
        return {
            "project": project, "F1": 0.50, "P": 0.60, "R": 0.45,
            "fp_count": 5, "fn_count": 8,
            "predicted": set(), "gold": gold,
            "fps": [], "fns": sorted(list(gold))[:5],
            "elapsed_s": 0.0, "dry_run": True,
        }

    backend_str_l = "openai" if backend == LLMBackend.OPENAI else "claude"
    model_str_l = model or "default"
    bch = _bank_content_hash(bank)
    ck_l = f"l_{project}_{bch}_{backend_str_l}_{model_str_l}"
    cached = _cache_read(ck_l)
    if cached:
        cached["fps"] = [tuple(x) for x in cached.get("fps", [])]
        cached["fns"] = [tuple(x) for x in cached.get("fns", [])]
        cached.setdefault("predicted", set())
        cached.setdefault("gold", set())
        print(f"  [L cache hit] {project} (bank={bch[:6]})")
        return cached

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
        _json.dump(bank, tf)
        tmp_bank_path = tf.name

    try:
        linker = SLinker14Voyager(backend=backend, model=model, bank_path=tmp_bank_path)
        t0 = time.time()
        links = linker.link(text_path=str(paths["text"]), model_path=str(paths["model"]))
        elapsed = time.time() - t0

        predicted = {(lk.sentence_number, lk.component_id) for lk in links}
        gold = _ra.load_gold_sam(str(paths["gold_sam"]))
        metrics = _ra.eval_metrics(predicted, gold)

        fps = sorted(predicted - gold)
        fns = sorted(gold - predicted)
        result = {
            "project": project,
            "F1": metrics["F1"], "P": metrics["P"], "R": metrics["R"],
            "fp_count": metrics["fp"], "fn_count": metrics["fn"],
            "predicted": predicted, "gold": gold,
            "fps": fps, "fns": fns,
            "elapsed_s": elapsed,
        }
        _cache_write(ck_l, {
            "project": result["project"],
            "F1": result["F1"], "P": result["P"], "R": result["R"],
            "fp_count": result["fp_count"], "fn_count": result["fn_count"],
            "fps": [list(x) for x in fps],
            "fns": [list(x) for x in fns],
            "elapsed_s": result["elapsed_s"],
        })
        return result
    finally:
        try:
            Path(tmp_bank_path).unlink(missing_ok=True)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# O role — Oracle (text-aware, failure-mode-centric)
# ─────────────────────────────────────────────────────────────────────────────

O_PROMPT = """You are the ORACLE in a multi-role training loop for a software architecture
trace-link recovery pipeline. Your role: analyze where the linker (L) failed and
WHY, given full access to ground truth and the document text.

VOCABULARY DISCIPLINE — CRITICAL
The failure-mode analysis you produce will be consumed by the Distillator (D),
which must generate patterns usable on ANY architecture documentation.
Use ONLY discourse-syntactic-functional vocabulary:
  ALLOWED: subject-position, predicate, anaphora, antecedent, parenthetical,
    namespace-prefix, section-heading, sentence-position, qualifier-clause,
    cross-reference, coordinated-clause, possessive, definite-article,
    apposition, head-noun, modifier, multi-word-phrase, exact-match,
    partial-match, over-approved, under-rejected, propagated, missed,
    alias-of, container-of, sub-element-of.
  FORBIDDEN: component names from the document, project names, technology names,
    domain nouns (payment, user, file, media), role nouns (controller, broker,
    queue, scheduler).

Abstract example pairs MUST be synthesized (constructed), NOT paraphrased from
the actual document text.

ITERATION CONTEXT
  iter: {iter_num}
  split: {split_name}
  L macro F1 (train projects): {macro_f1:.4f}
  delta from prior iter: {delta:+.4f}

LINKER RESULTS FOR THIS PROJECT
  project_id: {project_id}
  F1: {f1:.4f}  P: {p:.4f}  R: {r:.4f}
  FP count: {fp_count}  FN count: {fn_count}

FALSE POSITIVES (abstract component IDs, NOT real names):
{fp_abstract}

FALSE NEGATIVES (abstract component IDs, NOT real names):
{fn_abstract}

SAMPLE SENTENCES — FALSE NEGATIVES (context around FN cases, with +-1 sentence window):
{fn_context_sample}

SAMPLE SENTENCES — FALSE POSITIVES (context around FP cases, with +-1 sentence window):
{fp_context_sample}

CURRENT BANK STATE (per-slot pattern counts):
{bank_summary}

Identify the most impactful failure modes. For each:
- Title and affected slot
- Symptom (what L did wrong, abstract discourse vocabulary)
- Apparent cause (why current bank/axiom rules allow it)
- Suggested direction (what kind of pattern would address it)
- Evidence count
- Abstract example pair (SYNTHESIZED, not from doc)

Also identify patterns that may have introduced new errors this iter.

Return JSON (exact schema):
{{
  "iter": {iter_num},
  "split": "{split_name}",
  "L_predictions_summary": {{
    "macro_F1": {macro_f1:.4f},
    "delta_from_prior_iter": {delta},
    "per_dataset": {{{project_id_quoted}: {f1:.4f}}}
  }},
  "failure_modes": [
    {{
      "id": "FM-1",
      "title": "...",
      "affected_slot": "<one of {slot_list}>",
      "symptom": "...",
      "apparent_cause": "...",
      "suggested_direction": "...",
      "evidence_count": N,
      "abstract_example_pair": "TP: <synthesized>\\nFP: <synthesized>"
    }}
  ],
  "newly_introduced_errors": []
}}
JSON only:"""


def _run_oracle_o(llm: LLMClient, project: str, l_run: dict, bank: dict,
                  iter_num: int, split_name: str, macro_f1: float,
                  delta: float, backend_str: str, model_str: str,
                  dry_run: bool = False) -> dict:
    """Run Oracle O for the given project and L run result.

    text-aware: uses sentence text from the project file.
    Returns structured failure-mode JSON.
    """
    paths = _ra.DATASETS[project]

    if dry_run:
        return {
            "iter": iter_num, "split": split_name,
            "L_predictions_summary": {
                "macro_F1": macro_f1, "delta_from_prior_iter": delta,
                "per_dataset": {project: l_run["F1"]},
            },
            "failure_modes": [
                {
                    "id": "FM-1",
                    "title": "Mock failure mode (dry-run)",
                    "affected_slot": "AMBIGUITY_RULES",
                    "symptom": "dry-run placeholder",
                    "apparent_cause": "no LLM calls in dry-run mode",
                    "suggested_direction": "N/A",
                    "evidence_count": 0,
                    "abstract_example_pair": "TP: N/A\nFP: N/A",
                }
            ],
            "newly_introduced_errors": [],
            "dry_run": True,
        }

    # Cache lookup — key includes bank_content_hash to prevent cross-split oracle reuse (REQ-V25-01)
    text_path = str(paths["text"])
    bch = _bank_content_hash(bank)
    ck = f"{Path(text_path).stem}_{_comp_hash(project)}_{bch}_{backend_str}_{model_str}_oracle_iter{iter_num}"
    cached = _cache_read(ck)
    if cached:
        print(f"  [O cache hit] {project} iter{iter_num}")
        return cached

    # Load sentence text for context
    from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
    sentences = load_sentences(text_path)
    sent_map = build_sent_map(sentences)

    from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
    components = parse_pcm_repository(str(paths["model"]))
    id_to_abstract = {c.id: f"comp_{i}" for i, c in enumerate(components)}

    fps = l_run.get("fps", [])
    fns = l_run.get("fns", [])

    fp_abstract = "\n".join(
        f"  (S{s}, {id_to_abstract.get(c, 'comp_?')})" for s, c in fps[:20]
    ) or "  (none)"
    fn_abstract = "\n".join(
        f"  (S{s}, {id_to_abstract.get(c, 'comp_?')})" for s, c in fns[:20]
    ) or "  (none)"

    # Build sentence context for FN cases (abstract, no component names)
    fn_context_lines = []
    for s_num, c_id in fns[:10]:
        s = sent_map.get(s_num)
        prev = sent_map.get(s_num - 1)
        if s:
            ctx = f"  S{s_num}: {s.text}"
            if prev:
                ctx = f"  S{s_num-1}: {prev.text}\n" + ctx
            fn_context_lines.append(f"FN case ({id_to_abstract.get(c_id, 'comp_?')}):\n{ctx}")
    fn_context_sample = "\n".join(fn_context_lines[:5]) or "  (no FN context available)"

    # Build sentence context for FP cases — without FP text the Oracle can't identify
    # "effect-only sentence" or "algorithm-description" patterns from FPs.
    fp_context_lines = []
    for s_num, c_id in fps[:10]:
        s = sent_map.get(s_num)
        prev = sent_map.get(s_num - 1)
        nxt = sent_map.get(s_num + 1)
        if s:
            ctx = f"  S{s_num}: {s.text}"
            if prev:
                ctx = f"  S{s_num-1}: {prev.text}\n" + ctx
            if nxt:
                ctx += f"\n  S{s_num+1}: {nxt.text}"
            fp_context_lines.append(f"FP case ({id_to_abstract.get(c_id, 'comp_?')}):\n{ctx}")
    fp_context_sample = "\n".join(fp_context_lines[:5]) or "  (no FP context available)"

    bank_summary = "\n".join(
        f"  {slot}: {len(pats)} patterns"
        for slot, pats in bank.get("slot_patterns", {}).items()
    )

    slot_list = ", ".join(SLOT_NAMES)

    prompt = O_PROMPT.format(
        iter_num=iter_num,
        split_name=split_name,
        macro_f1=macro_f1,
        delta=delta,
        project_id=project,
        f1=l_run["F1"],
        p=l_run["P"],
        r=l_run["R"],
        fp_count=l_run["fp_count"],
        fn_count=l_run["fn_count"],
        fp_abstract=fp_abstract,
        fn_abstract=fn_abstract,
        fn_context_sample=fn_context_sample,
        fp_context_sample=fp_context_sample,
        bank_summary=bank_summary,
        slot_list=slot_list,
        project_id_quoted=f'"{project}"',
    )

    ok, hits = gate06_ok(prompt)
    if not ok:
        print(f"  [O] WARNING: prompt contains taboo tokens {hits!r} — check abstraction")

    data = llm.extract_json(llm.query(prompt, timeout=300)) or {}
    if not isinstance(data, dict):
        data = {}
    _cache_write(ck, data)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# D role — Distillator (text-blind, CoT-A inline)
# ─────────────────────────────────────────────────────────────────────────────

D_PROMPT = """You are the DISTILLATOR in a multi-role training loop for a software architecture
trace-link recovery pipeline. You receive failure-mode analysis from the Oracle (O)
and must propose new patterns for the linker's prompt slots.

CRITICAL CONSTRAINTS
────────────────────
1. TEXT-BLIND: You NEVER see raw document text or component names.
2. CoT-A REQUIRED: Before emitting each pattern, perform an explicit chain-of-thought
   abstraction check: test the pattern mentally against 5 architectural styles
   (microservice mesh, event-sourced, layered monolith, pipe-and-filter,
   hexagonal/ports-and-adapters). Only propose if STYLE-INVARIANT.
3. VOCABULARY: discourse/syntactic/functional terms ONLY (same rules as Oracle O).

ALLOWED vocabulary: subject-position, predicate, anaphora, antecedent, parenthetical,
  namespace-prefix, section-heading, sentence-position, qualifier-clause,
  cross-reference, coordinated-clause, possessive, definite-article, apposition,
  head-noun, modifier, multi-word-phrase, exact-match, partial-match,
  introducing-sentence, follow-up-sentence, alias-defined-parenthetically.
FORBIDDEN: role nouns (controller, broker, queue, scheduler, dispatcher, etc.),
  architectural style names, domain nouns (payment, user, file, media, etc.),
  any benchmark component names.

ORACLE FAILURE MODES (from O for this iter):
{oracle_json}

CURRENT BANK STATE (patterns already inserted; do not duplicate):
{bank_summary}

HIGH-PRIORITY SLOTS (zero patterns — propose for these first before adding to populated slots):
{underfilled_slots}

INSTRUCTIONS
────────────
For each failure mode, decide:
1. Can it be addressed by a new pattern in the affected slot?
2. If yes: write the pattern with EXPLICIT CoT-A inline (test against 5 styles).
3. Only propose patterns that PASS CoT-A (style-invariant = passes all 5 styles).
4. Synthesize example pairs (TP + FP), never paraphrase from Oracle context.
5. Also propose removals if existing patterns are causing new errors.
6. REQUIRED: Each proposed pattern MUST include `addresses_failure_modes` — a
   non-empty list of FM IDs (e.g. ["FM-1", "FM-2"]) from the Oracle output above
   that this pattern addresses. Proposals with empty or absent addresses_failure_modes
   are REJECTED by Gate A without review.

Return JSON (exact schema):
{{
  "iter": {iter_num},
  "patterns_proposed": [
    {{
      "slot": "<one of {slot_list}>",
      "rule_text": "<2-4 sentence abstract rule, discourse vocabulary>",
      "example_block": "TP: <synthesized correct example>\\nFP: <synthesized incorrect example>",
      "why_it_transfers": "<reasoning about style-invariance>",
      "abstraction_check_cot": "Tested against microservice/event-sourced/layered/pipe-filter/hexagonal: <verdict + reasoning>. Passes/Fails.",
      "addresses_failure_modes": ["FM-1"]
    }}
  ],
  "patterns_to_remove": [
    {{"pattern_id": "p_XXX", "reason": "<categorical reason>"}}
  ]
}}
JSON only:"""


def _run_distillator_d(llm: LLMClient, o_jsons: list[dict], bank: dict,
                        iter_num: int, backend_str: str, model_str: str,
                        dry_run: bool = False) -> dict:
    """Run Distillator D across all Oracle outputs for this outer pass.

    Receives a list of O outputs (one per training project), current bank.
    Returns proposed patterns and removals.
    """
    if dry_run:
        return {
            "iter": iter_num,
            "patterns_proposed": [
                {
                    "slot": "AMBIGUITY_RULES",
                    "rule_text": "Mock pattern (dry-run — no LLM call).",
                    "example_block": "TP: mock TP example.\nFP: mock FP example.",
                    "why_it_transfers": "dry-run placeholder",
                    "abstraction_check_cot": "dry-run: PASSES all styles (placeholder).",
                    "addresses_failure_modes": ["FM-1"],
                }
            ],
            "patterns_to_remove": [],
            "dry_run": True,
        }

    # Aggregate O outputs for the prompt
    oracle_summary = json.dumps(
        [{"project": o.get("L_predictions_summary", {}).get("per_dataset", {}),
          "failure_modes": o.get("failure_modes", [])}
         for o in o_jsons],
        indent=2
    )

    bank_summary = "\n".join(
        f"  {slot} ({len(pats)} patterns): "
        + (", ".join(f"{p.get('pattern_id', '?')}" for p in pats[:5]) if pats else "empty")
        for slot, pats in bank.get("slot_patterns", {}).items()
    )

    empty_slots = [s for s in SLOT_NAMES if not bank.get("slot_patterns", {}).get(s)]
    underfilled_slots = (
        ", ".join(empty_slots) if empty_slots else "(all slots have ≥1 pattern)"
    )

    slot_list = ", ".join(SLOT_NAMES)

    prompt = D_PROMPT.format(
        oracle_json=oracle_summary,
        bank_summary=bank_summary,
        underfilled_slots=underfilled_slots,
        iter_num=iter_num,
        slot_list=slot_list,
    )

    ok, hits = gate06_ok(prompt)
    if not ok:
        print(f"  [D] WARNING: prompt contains taboo tokens {hits!r}")

    # Full prompt hash ensures cache invalidates when D_PROMPT template changes
    ck = f"d_iter{iter_num}_{backend_str}_{model_str}_{hashlib.md5(prompt.encode()).hexdigest()[:12]}"
    cached = _cache_read(ck)
    if cached:
        print(f"  [D cache hit] iter{iter_num}")
        return cached

    data = llm.extract_json(llm.query(prompt, timeout=300)) or {}
    if not isinstance(data, dict):
        data = {}
    _cache_write(ck, data)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# GATE-06 + reviewer_critic filter for D proposals
# ─────────────────────────────────────────────────────────────────────────────

def _filter_proposals(proposals: list[dict]) -> tuple[list[dict], list[dict]]:
    """Apply GATE-06 grep and reviewer_critic_stub to D's proposed patterns.

    Returns (accepted, rejected). Rejected patterns are logged.
    """
    accepted = []
    rejected = []
    for p in proposals:
        if not isinstance(p, dict):
            continue
        slot = p.get("slot", "")
        rule_text = p.get("rule_text", "")
        example_block = p.get("example_block", "")
        full_text = f"{rule_text} {example_block}"

        # GATE-06 taboo grep
        ok, hits = gate06_ok(full_text)
        if not ok:
            print(f"  [GATE-06 REJECT] slot={slot}: taboo tokens {hits!r}")
            rejected.append({**p, "rejection_reason": f"taboo: {hits!r}"})
            continue

        if slot not in SLOT_NAMES:
            print(f"  [GATE-06 REJECT] unknown slot {slot!r}")
            rejected.append({**p, "rejection_reason": f"unknown slot: {slot!r}"})
            continue

        # Reviewer critic stub (Phase 14: advisory only)
        crit = reviewer_critic_stub(rule_text, slot)
        if crit.get("verdict") == "REJECT" and not crit.get("advisory"):
            print(f"  [CRITIC REJECT] slot={slot}: {crit['reason']}")
            rejected.append({**p, "rejection_reason": f"critic: {crit['reason']}"})
            continue

        if crit.get("verdict") == "REJECT":
            print(f"  [CRITIC advisory REJECT — kept] slot={slot}: {crit['reason']}")

        accepted.append(p)

    return accepted, rejected


# ─────────────────────────────────────────────────────────────────────────────
def _to_bool(v) -> bool:
    """Coerce LLM bool output (True, 'true', 'True') to Python bool."""
    return v is True or v == "true" or v == "True"


# ─────────────────────────────────────────────────────────────────────────────
# Gate A — FM citation check (deterministic, $0)
# ─────────────────────────────────────────────────────────────────────────────

def _gate_a_check(
    proposals: list[dict], o_jsons_map: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    """Gate A: FM citation check (deterministic, $0).

    Validates each proposal's cited FM IDs against its own project's O output only.
    Proposals must carry _project tag (set during D collection).
    """
    accepted = []
    rejected = []
    for p in proposals:
        slot = p.get("slot", "?")
        project = p.get("_project", "")
        if not project:
            print(f"  [Gate A REJECT] slot={slot}: missing _project tag")
            rejected.append({**p, "gate_a_rejection": "missing _project tag"})
            continue
        cited = p.get("addresses_failure_modes", [])
        if not isinstance(cited, list) or len(cited) == 0:
            print(f"  [Gate A REJECT] slot={slot}: addresses_failure_modes empty or missing")
            rejected.append({**p, "gate_a_rejection": "addresses_failure_modes empty"})
            continue
        o = o_jsons_map.get(project, {})
        valid_fm_ids = {fm.get("id", "") for fm in o.get("failure_modes", []) if fm.get("id")}
        unknown = [fid for fid in cited if fid not in valid_fm_ids]
        if unknown:
            print(f"  [Gate A REJECT] slot={slot}: unknown FM IDs {unknown!r} for project={project}")
            rejected.append({**p, "gate_a_rejection": f"unknown FM IDs: {unknown!r}"})
            continue
        accepted.append(p)

    return accepted, rejected


# ─────────────────────────────────────────────────────────────────────────────
# Gate B — LLM dual-direction judge (~$0.01/pass)
# ─────────────────────────────────────────────────────────────────────────────

GATE_B_PROMPT = """You are a quality judge for a software architecture trace-link pattern proposal.

PROPOSED PATTERN:
  Slot: {slot}
  Rule: {rule_text}
  Example: {example_block}
  Addresses failure modes: {cited_fm_ids}

CITED FAILURE MODES (from Oracle analysis):
{fm_details}

NEWLY INTRODUCED ERRORS (flagged by Oracle this pass):
{new_errors}

JUDGE TASK
──────────
1. Does this pattern genuinely address the cited failure mode(s)?
   Consider: does the rule_text target the symptom described in the FM?
2. Does this pattern risk introducing new errors?
   Consider: would applying this rule approve links that should not be approved,
   or reject links that should be approved?

Return JSON:
{{
  "fixes_cited_fm": true | false,
  "causes_new_error": true | false,
  "confidence": "high" | "medium" | "low",
  "rationale": "<one sentence>"
}}
JSON only:"""


def _gate_b_judge(
    llm, proposals: list[dict], o_jsons_map: dict[str, dict], dry_run: bool = False
) -> tuple[list[dict], list[dict]]:
    """Gate B: LLM dual-direction semantic judge (~$0.01/pass).

    Accept condition: fixes_cited_fm=true AND causes_new_error=false AND confidence in {high, medium}.
    Uses each proposal's _project to look up FM details from the correct O output only.
    In dry_run or with no LLM: accepts all proposals (structural test only).
    """
    if dry_run or llm is None:
        return proposals, []

    accepted = []
    rejected = []
    for prop in proposals:
        slot = prop.get("slot", "?")
        project = prop.get("_project", "")
        rule_text = prop.get("rule_text", "")
        example_block = prop.get("example_block", "")
        cited_ids = prop.get("addresses_failure_modes", [])

        o = o_jsons_map.get(project, {})
        fm_lookup = {fm.get("id", ""): fm for fm in o.get("failure_modes", []) if fm.get("id")}
        new_errors_list = [
            (e if isinstance(e, str) else e.get("description", str(e)))
            for e in o.get("newly_introduced_errors", [])
        ]
        new_errors_text = (
            "\n".join(f"  - {e}" for e in new_errors_list[:5]) if new_errors_list else "  (none)"
        )

        fm_details_lines = []
        for fid in cited_ids:
            fm = fm_lookup.get(fid)
            if fm:
                fm_details_lines.append(
                    f"  {fid}: {fm.get('title', '?')}\n"
                    f"    Symptom: {fm.get('symptom', '?')}\n"
                    f"    Direction: {fm.get('suggested_direction', '?')}"
                )
        fm_details = "\n".join(fm_details_lines) if fm_details_lines else "  (FM details not found)"

        prompt = GATE_B_PROMPT.format(
            slot=slot,
            rule_text=rule_text,
            example_block=example_block,
            cited_fm_ids=", ".join(cited_ids),
            fm_details=fm_details,
            new_errors=new_errors_text,
        )

        verdict = llm.extract_json(llm.query(prompt, timeout=300)) or {}
        fixes = verdict.get("fixes_cited_fm", False)
        causes = verdict.get("causes_new_error", True)
        confidence = verdict.get("confidence", "low")
        rationale = verdict.get("rationale", "")

        accept = _to_bool(fixes) and not _to_bool(causes) and confidence in ("high", "medium")
        if accept:
            print(f"  [Gate B ACCEPT] slot={slot}: {rationale[:80]}")
            accepted.append(prop)
        else:
            reason = (
                f"fixes_cited_fm={fixes}, causes_new_error={causes}, "
                f"confidence={confidence}: {rationale[:80]}"
            )
            print(f"  [Gate B REJECT] slot={slot}: {reason}")
            rejected.append({**prop, "gate_b_rejection": reason})

    return accepted, rejected


# ─────────────────────────────────────────────────────────────────────────────
# Bank mutation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_proposals(bank: dict, proposals: list[dict]) -> dict:
    """Insert accepted proposals into the bank, assigning new pattern IDs."""
    import copy
    bank = copy.deepcopy(bank)
    sp = bank.setdefault("slot_patterns", {})
    for slot in SLOT_NAMES:
        sp.setdefault(slot, [])

    for prop in proposals:
        slot = prop.get("slot")
        if slot not in SLOT_NAMES:
            continue
        pid = _next_pattern_id(bank)
        entry = {
            "pattern_id": pid,
            "rule_text": prop.get("rule_text", ""),
            "example_block": prop.get("example_block", ""),
            "why_it_transfers": prop.get("why_it_transfers", ""),
            "abstraction_check_cot": prop.get("abstraction_check_cot", ""),
            "addresses_failure_modes": prop.get("addresses_failure_modes", []),
        }
        sp[slot].append(entry)
    return bank


def _apply_removals(bank: dict, removals: list[dict]) -> dict:
    import copy
    bank = copy.deepcopy(bank)
    sp = bank.get("slot_patterns", {})
    remove_ids = {r.get("pattern_id") for r in removals if isinstance(r, dict)}
    for slot in SLOT_NAMES:
        sp[slot] = [p for p in sp.get(slot, []) if p.get("pattern_id") not in remove_ids]
    return bank


# ─────────────────────────────────────────────────────────────────────────────
# Outer training pass
# ─────────────────────────────────────────────────────────────────────────────

def run_outer_pass(
    pass_num: int,
    projects: list[str],
    split_dir: Path,
    backend: LLMBackend,
    model: str | None,
    backend_str: str,
    model_str: str,
    prior_f1s: dict[str, float],
    prior_errors: dict[str, int] | None = None,
    dry_run: bool = False,
    split_name: str = "mainline",
) -> dict:
    """Execute one outer pass: L → O → D → GATE-06 filter → P.

    Returns pass summary dict.
    """
    print(f"\n{'='*60}")
    print(f"OUTER PASS {pass_num} | projects={projects} | dry_run={dry_run}")
    print(f"{'='*60}")

    llm = LLMClient(backend=backend, model=model) if not dry_run else None

    # Load per-project banks
    project_banks = {p: _load_bank(split_dir, p) for p in projects}

    # Step 1: L — run linker on all projects
    print("\n[L] Running linker on all training projects...")
    l_runs: dict[str, dict] = {}
    for project in projects:
        print(f"\n  [L] project={project}")
        l_runs[project] = _run_linker_l(
            project, backend, model, project_banks[project], dry_run=dry_run
        )
        print(f"  [L] {project}: F1={l_runs[project]['F1']:.4f}")

    train_f1s = {p: r["F1"] for p, r in l_runs.items()}
    macro_f1 = sum(train_f1s.values()) / max(1, len(train_f1s))
    prior_macro = sum(prior_f1s.values()) / max(1, len(prior_f1s)) if prior_f1s else 0.0
    delta = macro_f1 - prior_macro

    train_errors = {p: l_runs[p]["fp_count"] + l_runs[p]["fn_count"] for p in projects}
    total_errors = sum(train_errors.values())
    prior_errors = prior_errors or {}
    prior_total_errors = sum(prior_errors.values()) if prior_errors else None
    delta_errors = (total_errors - prior_total_errors) if prior_total_errors is not None else None

    print(f"\n[L] Train macro F1: {macro_f1:.4f} (delta: {delta:+.4f})")
    print(f"[L] Total errors (FP+FN): {total_errors}"
          + (f" (delta: {delta_errors:+d})" if delta_errors is not None else ""))

    # Step 1.5: Min-commit delta gate (REQ-V25-02) — filters LLM run-to-run variance
    if not dry_run and delta < MIN_COMMIT_DELTA:
        print(f"\n[P] delta={delta:+.4f} < MIN_COMMIT_DELTA={MIN_COMMIT_DELTA} — skipping O+D (variance filter)")
        committed_f1s = prior_f1s if prior_f1s else train_f1s
        committed_macro = sum(committed_f1s.values()) / max(1, len(committed_f1s))
        summary = {
            "pass": pass_num,
            "split": split_name,
            "projects": projects,
            "dry_run": dry_run,
            "train_errors_before": prior_errors,
            "train_errors_after_l": train_errors,
            "total_errors_l": total_errors,
            "delta_errors_from_prior": delta_errors,
            "this_pass_errors": train_errors,
            "total_this_pass_errors": total_errors,
            "committed_errors": train_errors,
            "total_committed_errors": total_errors,
            "train_f1s_before": prior_f1s,
            "train_f1s_after_l": train_f1s,
            "macro_f1_l": macro_f1,
            "delta_f1_from_prior": delta,
            "committed_f1s": committed_f1s,
            "committed_macro_f1": committed_macro,
            "proposals_raw": 0,
            "proposals_gate06_accepted": 0,
            "proposals_gate06_rejected": 0,
            "proposals_gate_a_accepted": 0,
            "proposals_gate_a_rejected": 0,
            "proposals_gate_b_accepted": 0,
            "proposals_gate_b_rejected": 0,
            "removals": 0,
            "committed": False,
            "converged": False,
            "below_min_commit_delta": True,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        summary_path = split_dir / f"pass{pass_num}_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\n[Pass {pass_num}] below-threshold no-op: delta={delta:+.4f} committed_macro={committed_macro:.4f}")
        return summary

    # Step 2: O — Oracle (text-aware)
    print("\n[O] Running Oracle on all training projects...")
    o_jsons: list[dict] = []
    o_jsons_map: dict[str, dict] = {}
    for project in projects:
        print(f"\n  [O] project={project}")
        o_json = _run_oracle_o(
            llm=llm,
            project=project,
            l_run=l_runs[project],
            bank=project_banks[project],
            iter_num=pass_num,
            split_name=split_name,
            macro_f1=macro_f1,
            delta=delta,
            backend_str=backend_str,
            model_str=model_str,
            dry_run=dry_run,
        )
        o_jsons.append(o_json)
        o_jsons_map[project] = o_json
        n_fm = len(o_json.get("failure_modes", []))
        print(f"  [O] {project}: {n_fm} failure modes")
        # Save O output
        o_path = split_dir / f"pass{pass_num}_{project}_oracle.json"
        o_path.parent.mkdir(parents=True, exist_ok=True)
        o_path.write_text(json.dumps(o_json, indent=2))

    # Step 3: D — Distillator (text-blind, CoT-A inline), run per-project.
    # Per-project D prevents high-evidence projects from monopolising pattern proposals.
    # Ablation (probe pass 1): combined D → 3 slots; per-project D → 5 unique slots.
    print("\n[D] Running Distillator per-project (text-blind, CoT-A)...")
    proposals_raw: list[dict] = []
    project_removals: dict[str, list[dict]] = {p: [] for p in projects}
    seen_proposal_keys: set[str] = set()
    d_results: list[dict] = []
    for proj_idx, project in enumerate(projects):
        print(f"\n  [D] project={project}")
        proj_bank = project_banks[project]
        d_result = _run_distillator_d(
            llm=llm,
            o_jsons=[o_jsons[proj_idx]],
            bank=proj_bank,
            iter_num=pass_num,
            backend_str=backend_str,
            model_str=model_str,
            dry_run=dry_run,
        )
        d_results.append(d_result)
        for pat in d_result.get("patterns_proposed", []):
            pat = dict(pat)
            pat["_project"] = project  # tag for Gate A/B project-scoped FM lookup
            key = (pat.get("slot", "") + pat.get("rule_text", "")[:60]).lower()
            if key not in seen_proposal_keys:
                seen_proposal_keys.add(key)
                proposals_raw.append(pat)
        for rem in d_result.get("patterns_to_remove", []):
            rid = rem.get("pattern_id", str(rem))
            if all(r.get("pattern_id") != rid for r in project_removals[project]):
                project_removals[project].append(rem)
        n_proposed = len(d_result.get("patterns_proposed", []))
        n_removals = len(d_result.get("patterns_to_remove", []))
        print(f"  [D] {project}: proposed {n_proposed}, removals {n_removals}")

    total_removals = sum(len(v) for v in project_removals.values())
    print(f"  [D] total unique proposals: {len(proposals_raw)}, total removals: {total_removals}")

    # Save merged D output (all per-project results)
    d_path = split_dir / f"pass{pass_num}_distillator.json"
    d_path.write_text(json.dumps({"per_project": d_results, "merged_proposals": proposals_raw, "per_project_removals": project_removals}, indent=2))

    # Step 4: GATE-06 + reviewer_critic filter
    print("\n[GATE-06] Filtering D proposals...")
    accepted, rejected = _filter_proposals(proposals_raw)
    print(f"  [GATE-06] accepted={len(accepted)} rejected={len(rejected)}")

    # Step 5b: Gate A — FM citation check (deterministic, $0)
    print("\n[Gate A] FM citation check...")
    a_accepted, a_rejected = _gate_a_check(accepted, o_jsons_map)
    print(f"  [Gate A] accepted={len(a_accepted)} rejected={len(a_rejected)}")

    # Step 5c: Gate B — LLM dual-direction judge (~$0.01/pass)
    print("\n[Gate B] Semantic validity judge...")
    b_accepted, b_rejected = _gate_b_judge(llm, a_accepted, o_jsons_map, dry_run=dry_run)
    print(f"  [Gate B] accepted={len(b_accepted)} rejected={len(b_rejected)}")

    final_accepted = b_accepted

    # Step 6: Commit or no-op (no probation re-run)
    any_removals = any(project_removals[p] for p in projects)
    print(f"\n[Commit] {len(final_accepted)} patterns pass Gate A+B, {total_removals} removals")
    did_commit = bool(final_accepted or any_removals)
    if did_commit:
        committed_banks = {}
        for project in projects:
            cb = _apply_proposals(project_banks[project], final_accepted)
            cb = _apply_removals(cb, project_removals[project])
            _save_bank(split_dir, project, cb)
            print(f"  [Commit] bank saved: {project} ({_total_patterns(cb)} patterns)")
            committed_banks[project] = cb
        committed_f1s = train_f1s
        print(f"  [Commit] COMMITTED {len(final_accepted)} patterns + {total_removals} removals")
    else:
        print(f"  [Commit] no-op — no patterns passed Gate A+B and no removals")
        committed_banks = project_banks
        committed_f1s = prior_f1s if prior_f1s else train_f1s

    # this_pass_errors: L errors measured this pass with the bank from the *previous* commit.
    # Convergence fires when: (a) nothing was committed this pass (no-op = plateau), OR
    # (b) pass >= 2 AND errors not improving AND nothing new to commit.
    # We do NOT converge when patterns were committed — their effect is unmeasured until next pass.
    this_pass_errors = train_errors
    total_this_pass_errors = total_errors
    committed_macro = sum(committed_f1s.values()) / max(1, len(committed_f1s))

    converged = (
        not did_commit
        and pass_num >= 2
        and bool(prior_errors)
        and total_this_pass_errors >= sum(prior_errors.values())
    )

    summary = {
        "pass": pass_num,
        "split": split_name,
        "projects": projects,
        "dry_run": dry_run,
        "train_errors_before": prior_errors,
        "train_errors_after_l": train_errors,
        "total_errors_l": total_errors,
        "delta_errors_from_prior": delta_errors,
        "this_pass_errors": this_pass_errors,
        "total_this_pass_errors": total_this_pass_errors,
        # kept as alias for caller compatibility
        "committed_errors": this_pass_errors,
        "total_committed_errors": total_this_pass_errors,
        "train_f1s_before": prior_f1s,
        "train_f1s_after_l": train_f1s,
        "macro_f1_l": macro_f1,
        "delta_f1_from_prior": delta,
        "committed_f1s": committed_f1s,
        "committed_macro_f1": committed_macro,
        "proposals_raw": len(proposals_raw),
        "proposals_gate06_accepted": len(accepted),
        "proposals_gate06_rejected": len(rejected),
        "proposals_gate_a_accepted": len(a_accepted),
        "proposals_gate_a_rejected": len(a_rejected),
        "proposals_gate_b_accepted": len(b_accepted),
        "proposals_gate_b_rejected": len(b_rejected),
        "removals": total_removals,
        "committed": bool(final_accepted or any_removals),
        "converged": converged,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    summary_path = split_dir / f"pass{pass_num}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[Pass {pass_num}] errors={total_this_pass_errors} macro_f1={committed_macro:.4f} "
          f"converged={converged} did_commit={did_commit} "
          f"(gate_a={len(a_accepted)}, gate_b={len(b_accepted)}, removals={total_removals})")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Tier runners (probe / range)
# ─────────────────────────────────────────────────────────────────────────────

def run_probe(projects: list[str], backend: LLMBackend, model: str | None,
              dry_run: bool = False, split_name: str = "mainline") -> dict:
    """Run Probe tier: 1-2 outer passes. Returns probe summary."""
    backend_str = "openai" if backend == LLMBackend.OPENAI else "claude"
    model_str = model or "default"
    split_dir = OUT_ROOT / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[PROBE TIER] projects={projects} backend={backend_str} model={model_str} dry_run={dry_run}")

    prior_f1s: dict[str, float] = {}
    prior_errors: dict[str, int] = {}
    pass_summaries = []

    for pass_num in range(1, 3):  # 1-2 passes for probe
        summary = run_outer_pass(
            pass_num=pass_num,
            projects=projects,
            split_dir=split_dir,
            backend=backend,
            model=model,
            backend_str=backend_str,
            model_str=model_str,
            prior_f1s=prior_f1s,
            prior_errors=prior_errors,
            dry_run=dry_run,
            split_name=split_name,
        )
        pass_summaries.append(summary)
        prior_f1s = {p: summary["committed_f1s"].get(p, 0.0) for p in projects}
        prior_errors = {p: summary["committed_errors"].get(p, 0) for p in projects}

        if summary.get("converged"):
            print(f"\n[PROBE] Converged at pass {pass_num}")
            break

    final_errors = pass_summaries[-1]["total_committed_errors"]
    final_macro = pass_summaries[-1]["committed_macro_f1"]
    any_committed = any(s.get("committed") for s in pass_summaries)
    verdict = "CONTINUE" if any_committed else "MARGINAL"

    probe_summary = {
        "tier": "probe",
        "split": split_name,
        "projects": projects,
        "passes_run": len(pass_summaries),
        "final_train_macro_f1": final_macro,
        "final_total_errors": final_errors,
        "verdict": verdict,
        "pass_summaries": pass_summaries,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (split_dir / "probe_summary.json").write_text(json.dumps(probe_summary, indent=2))
    print(f"\n[PROBE] verdict={verdict} final_macro={final_macro:.4f}")
    return probe_summary


def run_range(projects: list[str], backend: LLMBackend, model: str | None,
              dry_run: bool = False, split_name: str = "mainline") -> dict:
    """Run Range tier: up to MAX_OUTER_PASSES passes, stop at convergence."""
    backend_str = "openai" if backend == LLMBackend.OPENAI else "claude"
    model_str = model or "default"
    split_dir = OUT_ROOT / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[RANGE TIER] projects={projects} backend={backend_str} model={model_str} dry_run={dry_run}")

    prior_f1s: dict[str, float] = {}
    prior_errors: dict[str, int] = {}
    pass_summaries = []

    for pass_num in range(1, MAX_OUTER_PASSES + 1):
        summary = run_outer_pass(
            pass_num=pass_num,
            projects=projects,
            split_dir=split_dir,
            backend=backend,
            model=model,
            backend_str=backend_str,
            model_str=model_str,
            prior_f1s=prior_f1s,
            prior_errors=prior_errors,
            dry_run=dry_run,
            split_name=split_name,
        )
        pass_summaries.append(summary)
        prior_f1s = {p: summary["committed_f1s"].get(p, 0.0) for p in projects}
        prior_errors = {p: summary["committed_errors"].get(p, 0) for p in projects}

        if summary.get("converged"):
            print(f"\n[RANGE] Converged at pass {pass_num} "
                  f"(errors={summary['total_committed_errors']}, macro={summary['committed_macro_f1']:.4f})")
            break

    final_errors = pass_summaries[-1]["total_committed_errors"]
    final_macro = pass_summaries[-1]["committed_macro_f1"]

    range_summary = {
        "tier": "range",
        "split": split_name,
        "projects": projects,
        "passes_run": len(pass_summaries),
        "final_train_macro_f1": final_macro,
        "final_total_errors": final_errors,
        "converged": pass_summaries[-1].get("converged", False),
        "pass_summaries": pass_summaries,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (split_dir / "range_summary.json").write_text(json.dumps(range_summary, indent=2))
    print(f"\n[RANGE] errors={final_errors} final_macro={final_macro:.4f} converged={range_summary['converged']}")
    return range_summary


# ─────────────────────────────────────────────────────────────────────────────
# .env loader
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for tier in ("probe", "range"):
        p = sub.add_parser(tier, help=f"Run {tier} tier")
        p.add_argument("--projects", default=",".join(MAINLINE_TRAIN),
                       help="Comma-separated training project list")
        p.add_argument("--backend", default="openai", choices=["openai", "claude"])
        p.add_argument("--model", default="gpt-5.4")
        p.add_argument("--dry-run", action="store_true",
                       help="No LLM calls — structural test only")
        p.add_argument("--split", default="mainline")

    args = parser.parse_args(argv)
    os.environ["LLM_BACKEND"] = args.backend
    if args.backend == "openai":
        os.environ["OPENAI_MODEL_NAME"] = args.model
    else:
        os.environ["CLAUDE_MODEL"] = args.model

    backend = LLMBackend.OPENAI if args.backend == "openai" else LLMBackend.CLAUDE
    projects = [p.strip() for p in args.projects.split(",") if p.strip()]

    if args.cmd == "probe":
        run_probe(projects, backend, args.model, dry_run=args.dry_run, split_name=args.split)
    elif args.cmd == "range":
        run_range(projects, backend, args.model, dry_run=args.dry_run, split_name=args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
