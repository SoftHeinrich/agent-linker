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
P  — Probation gate: mechanical F1 delta check after each outer iter.
     Batch rollback if probation delta < 0.

ITERATION LOOP (per outer pass)
--------------------------------
  for project in training_set:
    L_run → O_json → D_proposals → GATE-06 filter → candidate_bank
  probation_test(candidate_bank)
  if delta >= 0: commit; else: rollback
  convergence_check (macro_F1 >= 0.90 on train OR pass 5 cap)

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
CONVERGENCE_THRESHOLD = 0.90
CHEAP_KILL_THRESHOLD = 0.87

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
    """Run L on project with current bank state. Returns metrics dict."""
    from llm_sad_sam.linkers.experimental.s_linker14_voyager import SLinker14Voyager

    paths = _ra.DATASETS[project]
    import tempfile, json as _json

    if dry_run:
        # In dry-run: return mock metrics without any LLM calls
        gold = _ra.load_gold_sam(str(paths["gold_sam"]))
        return {
            "project": project, "F1": 0.50, "P": 0.60, "R": 0.45,
            "fp_count": 5, "fn_count": 8,
            "predicted": set(), "gold": gold,
            "fps": [], "fns": sorted(list(gold))[:5],
            "elapsed_s": 0.0, "dry_run": True,
        }

    # Write bank to a temp file for the linker constructor
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
        return {
            "project": project,
            "F1": metrics["F1"], "P": metrics["P"], "R": metrics["R"],
            "fp_count": metrics["fp"], "fn_count": metrics["fn"],
            "predicted": predicted, "gold": gold,
            "fps": fps, "fns": fns,
            "elapsed_s": elapsed,
        }
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

SAMPLE SENTENCES (context around FN cases, with +-1 sentence window):
{fn_context_sample}

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

    # Cache lookup
    text_path = str(paths["text"])
    ck = _cache_key(text_path, project, backend_str, model_str, f"oracle_iter{iter_num}")
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

    # Build sentence context sample for FN cases (abstract, no component names)
    fn_context_lines = []
    for s_num, c_id in fns[:10]:
        s = sent_map.get(s_num)
        prev = sent_map.get(s_num - 1)
        if s:
            ctx = f"  S{s_num}: {s.text}"
            if prev:
                ctx = f"  S{s_num-1}: {prev.text[:80]}\n" + ctx
            fn_context_lines.append(f"FN case ({id_to_abstract.get(c_id, 'comp_?')}):\n{ctx}")
    fn_context_sample = "\n".join(fn_context_lines[:5]) or "  (no FN context available)"

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

INSTRUCTIONS
────────────
For each failure mode, decide:
1. Can it be addressed by a new pattern in the affected slot?
2. If yes: write the pattern with EXPLICIT CoT-A inline (test against 5 styles).
3. Only propose patterns that PASS CoT-A (style-invariant = passes all 5 styles).
4. Synthesize example pairs (TP + FP), never paraphrase from Oracle context.
5. Also propose removals if existing patterns are causing new errors.

Return JSON (exact schema):
{{
  "iter": {iter_num},
  "patterns_proposed": [
    {{
      "slot": "<one of {slot_list}>",
      "rule_text": "<2-4 sentence abstract rule, discourse vocabulary>",
      "example_block": "TP: <synthesized correct example>\\nFP: <synthesized incorrect example>",
      "why_it_transfers": "<reasoning about style-invariance>",
      "abstraction_check_cot": "Tested against microservice/event-sourced/layered/pipe-filter/hexagonal: <verdict + reasoning>. Passes/Fails."
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
                }
            ],
            "patterns_to_remove": [],
            "dry_run": True,
        }

    # Aggregate O outputs for the prompt
    oracle_summary = json.dumps(
        [{"project": o.get("L_predictions_summary", {}).get("per_dataset", {}),
          "failure_modes": o.get("failure_modes", [])[:5]}
         for o in o_jsons],
        indent=2
    )

    bank_summary = "\n".join(
        f"  {slot} ({len(pats)} patterns): "
        + (", ".join(f"{p.get('pattern_id', '?')}" for p in pats[:5]) if pats else "empty")
        for slot, pats in bank.get("slot_patterns", {}).items()
    )

    slot_list = ", ".join(SLOT_NAMES)

    prompt = D_PROMPT.format(
        oracle_json=oracle_summary[:4000],  # truncate for budget
        bank_summary=bank_summary,
        iter_num=iter_num,
        slot_list=slot_list,
    )

    ok, hits = gate06_ok(prompt)
    if not ok:
        print(f"  [D] WARNING: prompt contains taboo tokens {hits!r}")

    # D is called once per outer pass (not per project) — no per-project cache key needed
    ck = f"d_iter{iter_num}_{backend_str}_{model_str}_{hashlib.md5(prompt[:200].encode()).hexdigest()[:8]}"
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
# P role — Probation gate (mechanical, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _probation_check(projects: list[str], candidate_bank: dict,
                      prior_f1s: dict[str, float], backend: LLMBackend,
                      model: str | None, dry_run: bool = False) -> tuple[float, dict[str, float]]:
    """Run L with candidate_bank on probation projects, compute macro F1 delta.

    Returns (delta, new_f1s). Uses cached L runs where available.
    """
    if dry_run:
        return 0.0, {p: prior_f1s.get(p, 0.5) for p in projects}

    new_f1s = {}
    for project in projects:
        run = _run_linker_l(project, backend, model, candidate_bank, dry_run=False)
        new_f1s[project] = run["F1"]

    prior_macro = sum(prior_f1s.values()) / max(1, len(prior_f1s))
    new_macro = sum(new_f1s.values()) / max(1, len(new_f1s))
    delta = new_macro - prior_macro

    print(f"  [P] probation macro: {new_macro:.4f} (prior: {prior_macro:.4f}, delta: {delta:+.4f})")
    return delta, new_f1s


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
    print(f"\n[L] Train macro F1: {macro_f1:.4f} (delta: {delta:+.4f})")

    # Step 2: O — Oracle (text-aware)
    print("\n[O] Running Oracle on all training projects...")
    o_jsons: list[dict] = []
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
        n_fm = len(o_json.get("failure_modes", []))
        print(f"  [O] {project}: {n_fm} failure modes")
        # Save O output
        o_path = split_dir / f"pass{pass_num}_{project}_oracle.json"
        o_path.parent.mkdir(parents=True, exist_ok=True)
        o_path.write_text(json.dumps(o_json, indent=2))

    # Step 3: D — Distillator (text-blind, CoT-A inline)
    # D runs once per outer pass across all O outputs
    print("\n[D] Running Distillator (text-blind, CoT-A)...")
    # Use a representative bank for D (merge slot_patterns from all projects)
    merged_bank = {"version": "v4b", "slot_patterns": {s: [] for s in SLOT_NAMES}}
    seen_ids: set[str] = set()
    for p in projects:
        for slot, pats in project_banks[p].get("slot_patterns", {}).items():
            for pat in pats:
                pid = pat.get("pattern_id", "")
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    merged_bank["slot_patterns"][slot].append(pat)

    d_result = _run_distillator_d(
        llm=llm,
        o_jsons=o_jsons,
        bank=merged_bank,
        iter_num=pass_num,
        backend_str=backend_str,
        model_str=model_str,
        dry_run=dry_run,
    )
    proposals_raw = d_result.get("patterns_proposed", [])
    removals = d_result.get("patterns_to_remove", [])
    print(f"  [D] proposed {len(proposals_raw)} patterns, {len(removals)} removals")

    # Save D output
    d_path = split_dir / f"pass{pass_num}_distillator.json"
    d_path.write_text(json.dumps(d_result, indent=2))

    # Step 4: GATE-06 + reviewer_critic filter
    print("\n[GATE-06] Filtering D proposals...")
    accepted, rejected = _filter_proposals(proposals_raw)
    print(f"  [GATE-06] accepted={len(accepted)} rejected={len(rejected)}")

    # Step 5: Build candidate bank by applying proposals to each project bank
    candidate_banks = {}
    for project in projects:
        cb = _apply_proposals(project_banks[project], accepted)
        cb = _apply_removals(cb, removals)
        candidate_banks[project] = cb

    # Step 6: P — Probation gate
    print("\n[P] Probation gate...")
    prob_delta, new_f1s = _probation_check(
        projects=projects,
        candidate_bank=candidate_banks[projects[0]],  # representative
        prior_f1s=train_f1s,
        backend=backend,
        model=model,
        dry_run=dry_run,
    )

    if prob_delta >= 0:
        print(f"  [P] COMMIT (delta={prob_delta:+.4f} >= 0)")
        for project in projects:
            _save_bank(split_dir, project, candidate_banks[project])
            print(f"  [P] bank saved: {project} ({_total_patterns(candidate_banks[project])} patterns)")
        committed_banks = candidate_banks
        committed_f1s = new_f1s
    else:
        print(f"  [P] ROLLBACK (delta={prob_delta:+.4f} < 0) — discarding all {len(accepted)} patterns")
        for project in projects:
            _save_bank(split_dir, project, project_banks[project])
        committed_banks = project_banks
        committed_f1s = train_f1s

    # Recompute macro for committed state
    committed_macro = sum(committed_f1s.values()) / max(1, len(committed_f1s))

    summary = {
        "pass": pass_num,
        "split": split_name,
        "projects": projects,
        "dry_run": dry_run,
        "train_f1s_before": prior_f1s,
        "train_f1s_after_l": train_f1s,
        "macro_f1_l": macro_f1,
        "delta_from_prior": delta,
        "proposals_raw": len(proposals_raw),
        "proposals_accepted": len(accepted),
        "proposals_rejected": len(rejected),
        "removals": len(removals),
        "probation_delta": prob_delta,
        "committed": prob_delta >= 0,
        "committed_macro_f1": committed_macro,
        "converged": committed_macro >= CONVERGENCE_THRESHOLD,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    summary_path = split_dir / f"pass{pass_num}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[Pass {pass_num}] committed_macro={committed_macro:.4f} converged={summary['converged']}")
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
            dry_run=dry_run,
            split_name=split_name,
        )
        pass_summaries.append(summary)
        prior_f1s = {p: summary["train_f1s_after_l"].get(p, 0.0) for p in projects}

        # Cheap-kill gate after pass 2
        if pass_num == 2:
            macro = summary["committed_macro_f1"]
            if macro < CHEAP_KILL_THRESHOLD:
                print(f"\n[PROBE] CHEAP-KILL: macro F1 {macro:.4f} < {CHEAP_KILL_THRESHOLD} after pass 2")
                print("[PROBE] v4 KILLED — Phase 18 Compact-B should activate")
                break

        if summary.get("converged"):
            print(f"\n[PROBE] Converged at pass {pass_num}")
            break

    final_macro = pass_summaries[-1]["committed_macro_f1"]
    verdict = "CONTINUE" if final_macro >= CHEAP_KILL_THRESHOLD else "KILL"

    probe_summary = {
        "tier": "probe",
        "split": split_name,
        "projects": projects,
        "passes_run": len(pass_summaries),
        "final_train_macro_f1": final_macro,
        "verdict": verdict,
        "cheap_kill_threshold": CHEAP_KILL_THRESHOLD,
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
            dry_run=dry_run,
            split_name=split_name,
        )
        pass_summaries.append(summary)
        prior_f1s = {p: summary["train_f1s_after_l"].get(p, 0.0) for p in projects}

        if summary.get("converged"):
            print(f"\n[RANGE] Converged at pass {pass_num} (macro={summary['committed_macro_f1']:.4f})")
            break

    final_macro = pass_summaries[-1]["committed_macro_f1"]

    range_summary = {
        "tier": "range",
        "split": split_name,
        "projects": projects,
        "passes_run": len(pass_summaries),
        "final_train_macro_f1": final_macro,
        "converged": pass_summaries[-1].get("converged", False),
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "pass_summaries": pass_summaries,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (split_dir / "range_summary.json").write_text(json.dumps(range_summary, indent=2))
    print(f"\n[RANGE] final_macro={final_macro:.4f} converged={range_summary['converged']}")
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
