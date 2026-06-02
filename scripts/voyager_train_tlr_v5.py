"""Voyager-TLR v5 — LLM-Driven Training Harness (Phase 32).

Replaces v4_beta (Gate A + Gate B F1-delta loop) with:

  OD role  — Oracle + Distillator merged into one text-aware call per project.
             Receives: current bank, FP/FN sentences, gold context.
             Emits: failure modes WITH labels AND proposed patterns in one response.

  Assessor — Replaces Gate A + Gate B entirely.
             Receives: proposed pattern, full bank, FP/FN sentence list with context.
             Emits: accept | reject | revise + rationale citing specific sentences.
             Max 1 revision cycle per proposal.

  [TRAIN]/[TEST] separation — every pass logs both:
             [TRAIN] {project}: F1=x  macro=x
             [TEST]  {project}: F1=x  macro=x
             Commit-decision delta uses [TRAIN] macro only.
             [TEST] is diagnostic only.

  Per-split independent training — each split trains on its own train projects,
  evaluates on its own held-out test projects vs an axiom-only baseline.
  No cross-split aggregation during training.

CHANGES FROM v4_beta
---------------------
  Removed: O role, D role, Gate A, Gate B, reviewer_critic_stub.
  Added:   OD role, Assessor role (LLM), test-project eval per pass.
  Kept:    GATE-06 grep, L caching, bank I/O helpers, convergence logic,
           dry-run mode, SLOT_NAMES, TABOO_PATTERN.

CLI
---
  python scripts/voyager_train_tlr_v5.py probe \\
      --projects mediastore --backend openai --model gpt-5.4

  python scripts/voyager_train_tlr_v5.py range \\
      --projects mediastore,teastore,teammates \\
      --test-projects bigbluebutton,jabref \\
      --backend openai --model gpt-5.4

  python scripts/voyager_train_tlr_v5.py confirmation \\
      --split split1_replication \\
      --train-projects mediastore,teastore,teammates \\
      --test-projects bigbluebutton,jabref \\
      --backend openai --model gpt-5.4
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

CACHE_ROOT = Path(os.environ.get("VOYAGER5_CACHE_ROOT", "results/voyager_v5/cache"))
OUT_ROOT = Path(os.environ.get("VOYAGER5_OUT_ROOT", "results/voyager_v5"))

MAINLINE_TRAIN = ["mediastore", "teastore", "teammates"]
MAINLINE_TEST = ["bigbluebutton", "jabref"]
MAX_OUTER_PASSES = 6
PROBE_PASSES = 3

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
    "SEED_EXTRACTION_RULES",
    "SEED_ACTOR_RULES",
    "GENERIC_WORD_USAGE_RULES",
    "ALIAS_SCOPE_RULES",
    "ANTECEDENT_ALIAS_RULES",
    "COREF_TERMINAL_SPECIFICITY_RULES",
)


# ─────────────────────────────────────────────────────────────────────────────
# GATE-06 helpers
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
    hits = TABOO_PATTERN.findall(text or "")
    return (len(hits) == 0, hits)


# ─────────────────────────────────────────────────────────────────────────────
# Cache adapter
# ─────────────────────────────────────────────────────────────────────────────

def _comp_hash(project: str) -> str:
    paths = _ra.DATASETS[project]
    model_path = str(paths["model"])
    try:
        data = Path(model_path).read_bytes()
        return hashlib.sha256(data).hexdigest()[:16]
    except FileNotFoundError:
        return hashlib.sha256(model_path.encode()).hexdigest()[:16]


def _bank_content_hash(bank: dict) -> str:
    content = json.dumps(bank.get("slot_patterns", {}), sort_keys=True)
    bank_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    axiom_path = _ROOT / "src" / "llm_sad_sam" / "linkers" / "experimental" / "prompts_v4_axiom.py"
    try:
        axiom_hash = hashlib.md5(axiom_path.read_bytes()).hexdigest()[:6]
    except FileNotFoundError:
        axiom_hash = "noaxiom"
    return f"{bank_hash}_{axiom_hash}"


def _cache_path(key: str) -> Path:
    root = Path(os.environ.get("VOYAGER5_CACHE_ROOT", str(CACHE_ROOT)))
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
    _cache_path(key).write_text(json.dumps(data, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Bank I/O
# ─────────────────────────────────────────────────────────────────────────────

def _bank_path_file(split_dir: Path, project: str) -> Path:
    return split_dir / f"{project}_bank.json"


def _load_bank(split_dir: Path, project: str) -> dict:
    p = _bank_path_file(split_dir, project)
    if not p.exists():
        return {"version": "v5", "project": project, "slot_patterns": {s: [] for s in SLOT_NAMES}}
    try:
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            sp = data.setdefault("slot_patterns", {})
            for s in SLOT_NAMES:
                sp.setdefault(s, [])
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return {"version": "v5", "project": project, "slot_patterns": {s: [] for s in SLOT_NAMES}}


def _save_bank(split_dir: Path, project: str, bank: dict) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    _bank_path_file(split_dir, project).write_text(json.dumps(bank, indent=2))


def _snapshot_banks(split_dir: Path, pass_num: int, projects: list[str]) -> None:
    snap_dir = split_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for project in projects:
        src = _bank_path_file(split_dir, project)
        if src.exists():
            (snap_dir / f"pass{pass_num}_{project}_bank.json").write_text(src.read_text())


def _restore_best_banks(split_dir: Path, best_pass: int, projects: list[str]) -> None:
    snap_dir = split_dir / "snapshots"
    for project in projects:
        snap = snap_dir / f"pass{best_pass}_{project}_bank.json"
        if snap.exists():
            _bank_path_file(split_dir, project).write_text(snap.read_text())


def _total_patterns(bank: dict) -> int:
    return sum(len(v) for v in bank.get("slot_patterns", {}).values())


def _next_pattern_id(bank: dict) -> str:
    nums = []
    for slot in bank.get("slot_patterns", {}).values():
        for p in slot:
            m = re.match(r"p_(\d+)", p.get("pattern_id", ""))
            if m:
                nums.append(int(m.group(1)))
    return f"p_{(max(nums) + 1 if nums else 1):03d}"


# ─────────────────────────────────────────────────────────────────────────────
# L role — run linker and compute metrics
# ─────────────────────────────────────────────────────────────────────────────

def _run_linker_l(project: str, backend: LLMBackend, model: str | None,
                  bank: dict, dry_run: bool = False) -> dict:
    """Run L on project with current bank. Cached per (project, bank_content_hash, backend, model)."""
    from llm_sad_sam.linkers.experimental.s_linker14_voyager import SLinker14Voyager
    import tempfile

    paths = _ra.DATASETS[project]

    if dry_run:
        gold = _ra.load_gold_sam(str(paths["gold_sam"]))
        return {
            "project": project, "F1": 0.50, "P": 0.60, "R": 0.45,
            "fp_count": 5, "fn_count": 8,
            "predicted": set(), "gold": gold,
            "fps": [], "fns": sorted(list(gold))[:5],
            "elapsed_s": 0.0, "dry_run": True,
        }

    backend_str = "openai" if backend == LLMBackend.OPENAI else "claude"
    model_str = model or "default"
    bch = _bank_content_hash(bank)
    ck = f"l_{project}_{bch}_{backend_str}_{model_str}"
    total_pats = _total_patterns(bank)
    filled_slots = sum(1 for v in bank.get("slot_patterns", {}).values() if v)
    total_slots = len(SLOT_NAMES)
    bank_label = f"bank={bch[:8]}, {total_pats}p/{filled_slots}/{total_slots}slots"
    cached = _cache_read(ck)
    if cached:
        cached["fps"] = [tuple(x) for x in cached.get("fps", [])]
        cached["fns"] = [tuple(x) for x in cached.get("fns", [])]
        cached.setdefault("predicted", set())
        cached.setdefault("gold", set())
        print(f"  [L cache hit] {project} ({bank_label})")
        return cached
    print(f"  [L run fresh] {project} ({bank_label})")

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf:
        json.dump(bank, tf)
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
        _cache_write(ck, {
            "project": project,
            "F1": result["F1"], "P": result["P"], "R": result["R"],
            "fp_count": result["fp_count"], "fn_count": result["fn_count"],
            "fps": [list(x) for x in fps],
            "fns": [list(x) for x in fns],
            "elapsed_s": elapsed,
        })
        return result
    finally:
        try:
            Path(tmp_bank_path).unlink(missing_ok=True)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Axiom-only baseline (empty bank)
# ─────────────────────────────────────────────────────────────────────────────

def _run_axiom_only_baseline(projects: list[str], backend: LLMBackend,
                              model: str | None, dry_run: bool = False) -> dict[str, float]:
    """Evaluate projects with empty bank (axiom-only floor). Returns {project: F1}."""
    print("\n[Axiom-only baseline] Running with empty bank...")
    empty_bank = {"version": "v5", "slot_patterns": {s: [] for s in SLOT_NAMES}}
    results: dict[str, float] = {}
    for project in projects:
        print(f"  [axiom] {project}...", end=" ", flush=True)
        r = _run_linker_l(project, backend, model, empty_bank, dry_run=dry_run)
        results[project] = r["F1"]
        print(f"F1={r['F1']:.4f}")
    macro = sum(results.values()) / max(1, len(results))
    print(f"  [axiom] macro F1 = {macro:.4f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# OD role — Oracle + Distillator merged (REQ-V26-03)
# ─────────────────────────────────────────────────────────────────────────────

OD_PROMPT = """You are the OD ANALYST in a multi-role training loop for a software
architecture trace-link recovery pipeline. You perform four reasoning steps in
one response: CITE → DERIVE → WRITE → SELF-CHECK.

═══════════════════════════════════════════════════════════════
VOCABULARY DISCIPLINE
═══════════════════════════════════════════════════════════════
ALLOWED:  subject-position, predicate, anaphora, antecedent, parenthetical,
  namespace-prefix, section-heading, sentence-position, qualifier-clause,
  cross-reference, coordinated-clause, possessive, definite-article,
  apposition, head-noun, modifier, multi-word-phrase, exact-match,
  partial-match, over-approved, under-rejected, propagated, missed,
  alias-of, container-of, sub-element-of, introducing-sentence,
  follow-up-sentence, alias-defined-parenthetically, role-referential-NP,
  responsibility-list, section-context-name, gerund-fragment, antecedent-via-alias.

FORBIDDEN:  any component names from the document, project names, technology names,
  domain nouns (payment, user, file, media), role nouns (controller, broker,
  queue, scheduler, dispatcher). Synthesize all examples — never quote from doc.

═══════════════════════════════════════════════════════════════
TRAINING CONTEXT
═══════════════════════════════════════════════════════════════
  pass:  {pass_num}    split: {split_name}    project: {project_id}
  [TRAIN] F1: {f1:.4f}  P: {p:.4f}  R: {r:.4f}
  FP count: {fp_count}  FN count: {fn_count}
  macro F1 this pass: {macro_f1:.4f}  (delta from prior: {delta:+.4f})

FAILURE MODE TITLES ALREADY COVERED IN PRIOR PASSES (DO NOT REPEAT — propose orthogonal modes):
{covered_fms}

CURRENT BANK (slot → pattern count):
{bank_summary}

UNDER-FILLED SLOTS (zero patterns): {underfilled_slots}
QUOTA: if any under-filled slot exists, at least one proposal MUST target an under-filled slot.

═══════════════════════════════════════════════════════════════
EVIDENCE
═══════════════════════════════════════════════════════════════
FALSE POSITIVES (abstract comp IDs):
{fp_abstract}

FP SENTENCES (context ±1):
{fp_context}

FALSE NEGATIVES (abstract comp IDs):
{fn_abstract}

FN SENTENCES (context ±1):
{fn_context}

═══════════════════════════════════════════════════════════════
STEP 1 — CITE EVIDENCE
═══════════════════════════════════════════════════════════════
For each failure mode, list the exact FP/FN sentence IDs that exhibit it and
the discourse token (using ALLOWED vocabulary) that caused the error.

═══════════════════════════════════════════════════════════════
STEP 2 — DERIVE MECHANISM
═══════════════════════════════════════════════════════════════
For each failure mode, name the discourse mechanism (in one phrase) responsible
for the cited evidence. Must use ALLOWED vocabulary.

═══════════════════════════════════════════════════════════════
STEP 3 — WRITE RULE
═══════════════════════════════════════════════════════════════
Choose a slot from {slot_list}. Write a 2-4 sentence abstract rule that references
the discourse mechanism from STEP 2. Synthesize one TP and one FP example pair —
never quote from the doc. Each proposal MUST reference at least one FM ID from STEP 1.

═══════════════════════════════════════════════════════════════
STEP 4 — SELF-CHECK
═══════════════════════════════════════════════════════════════
For each proposal, apply the rule mentally to the cited sentences:
  - Does it fire on the FP sentence (preventing the over-approval)?
  - Does it preserve TPs (not rejecting valid links of similar shape)?
If the answer is no, revise the rule before emitting.

Test against 5 architectural styles (microservice, event-sourced, layered,
pipe-filter, hexagonal). Emit only if STYLE-INVARIANT.

Also propose removals if any existing patterns appear to cause the new errors.

Return JSON:
{{
  "failure_modes": [
    {{
      "id": "FM-1",
      "title": "<must not duplicate covered_fms>",
      "affected_slot": "<slot>",
      "cited_fp_sentences": [N, ...],
      "cited_fn_sentences": [N, ...],
      "discourse_mechanism": "<phrase from STEP 2>",
      "symptom": "...",
      "apparent_cause": "...",
      "evidence_count": N
    }}
  ],
  "patterns_proposed": [
    {{
      "slot": "<slot>",
      "rule_text": "<2-4 sentence abstract rule referencing the discourse mechanism>",
      "example_block": "TP: <synthesized>\\nFP: <synthesized>",
      "why_it_transfers": "<style-invariance reasoning>",
      "abstraction_check_cot": "Tested microservice/event-sourced/layered/pipe-filter/hexagonal: PASSES/FAILS.",
      "self_check": "Applied to FP S<N>: FIRES. Applied to TP shape: PRESERVED.",
      "addresses_failure_modes": ["FM-1"]
    }}
  ],
  "patterns_to_remove": [
    {{"pattern_id": "p_XXX", "reason": "<categorical reason>"}}
  ]
}}
JSON only:"""


def _run_od(llm: LLMClient, project: str, l_run: dict, bank: dict,
            pass_num: int, split_name: str, macro_f1: float, delta: float,
            backend_str: str, model_str: str, dry_run: bool = False,
            covered_fm_titles: list[str] | None = None) -> dict:
    """Run OD analyst for one project. Returns failure modes + proposals."""
    paths = _ra.DATASETS[project]

    if dry_run:
        return {"failure_modes": [], "patterns_proposed": [],
                "patterns_to_remove": [], "dry_run": True}

    text_path = str(paths["text"])
    bch = _bank_content_hash(bank)
    ck = (f"od_{Path(text_path).stem}_{_comp_hash(project)}_{bch}"
          f"_{backend_str}_{model_str}_pass{pass_num}")
    cached = _cache_read(ck)
    if cached:
        print(f"  [OD cache hit] {project} pass{pass_num}")
        return cached

    from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
    from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository

    sentences = load_sentences(text_path)
    sent_map = build_sent_map(sentences)
    components = parse_pcm_repository(str(paths["model"]))
    id_to_abstract = {c.id: f"comp_{i}" for i, c in enumerate(components)}

    fps = l_run.get("fps", [])
    fns = l_run.get("fns", [])

    def _context_block(pairs, label, n=8) -> str:
        lines = []
        for s_num, c_id in pairs[:n]:
            s = sent_map.get(s_num)
            prev = sent_map.get(s_num - 1)
            nxt = sent_map.get(s_num + 1)
            if s:
                block = []
                if prev:
                    block.append(f"  S{s_num-1}: {prev.text}")
                block.append(f"  S{s_num}: {s.text}")
                if nxt:
                    block.append(f"  S{s_num+1}: {nxt.text}")
                lines.append(f"{label} ({id_to_abstract.get(c_id, 'comp_?')}):\n" + "\n".join(block))
        return "\n".join(lines) or "  (none)"

    fp_abstract = "\n".join(
        f"  (S{s}, {id_to_abstract.get(c, 'comp_?')})" for s, c in fps[:20]
    ) or "  (none)"
    fn_abstract = "\n".join(
        f"  (S{s}, {id_to_abstract.get(c, 'comp_?')})" for s, c in fns[:20]
    ) or "  (none)"

    bank_summary = "\n".join(
        f"  {slot}: {len(pats)} patterns"
        for slot, pats in bank.get("slot_patterns", {}).items()
    )
    empty_slots = [s for s in SLOT_NAMES if not bank.get("slot_patterns", {}).get(s)]
    underfilled = ", ".join(empty_slots) if empty_slots else "(all slots populated)"
    covered = covered_fm_titles or []
    covered_str = "\n".join(f"  - {t}" for t in covered) if covered else "  (none — first pass)"

    prompt = OD_PROMPT.format(
        pass_num=pass_num,
        split_name=split_name,
        project_id=project,
        f1=l_run["F1"], p=l_run["P"], r=l_run["R"],
        fp_count=l_run["fp_count"], fn_count=l_run["fn_count"],
        macro_f1=macro_f1, delta=delta,
        fp_abstract=fp_abstract,
        fp_context=_context_block(fps, "FP"),
        fn_abstract=fn_abstract,
        fn_context=_context_block(fns, "FN"),
        bank_summary=bank_summary,
        underfilled_slots=underfilled,
        covered_fms=covered_str,
        slot_list=", ".join(SLOT_NAMES),
    )

    ok, hits = gate06_ok(prompt)
    if not ok:
        print(f"  [OD] WARNING: prompt contains taboo tokens {hits!r}")

    data = llm.extract_json(llm.query(prompt, timeout=300)) or {}
    if not isinstance(data, dict):
        data = {}
    _cache_write(ck, data)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Assessor role — replaces Gate A + Gate B (REQ-V26-04)
# ─────────────────────────────────────────────────────────────────────────────

ASSESSOR_PROMPT = """You are the ASSESSOR in a training loop for a software architecture
trace-link recovery pipeline. Review a proposed bank pattern and decide whether to
accept it, reject it, or request a revision. You may also request removal of any
existing pattern in this slot that appears to be causing the current FP/FN.

PROPOSED PATTERN:
  Slot:      {slot}
  Rule:      {rule_text}
  Example:   {example_block}
  Addresses: {fm_ids}

FAILURE MODES THIS PATTERN CLAIMS TO ADDRESS:
{fm_details}

CURRENT BANK (existing patterns in this slot, with their original worked evidence):
{slot_patterns}

REMAINING FP SENTENCES (top 5, with ±1 context):
{fp_sample}

REMAINING FN SENTENCES (top 5, with ±1 context):
{fn_sample}

ASSESSMENT TASK
───────────────
1. Does the rule_text target the symptom(s) described in the failure mode(s)?
   Cite at least one specific FP or FN sentence.
2. Would applying this rule risk approving links that should be rejected,
   or rejecting links that should be approved?
3. Is the rule abstract enough for any architecture document (style-invariant)?
4. Does it duplicate an existing bank pattern?
5. Does any existing pattern in this slot now appear to be over-rejecting valid
   links (its original worked_fn_evidence no longer matches the current FN pattern)
   or over-approving invalid links (its worked_fp_evidence no longer prevents the
   current FP pattern)? If so, list its pattern_id under removal_targets.

Return one of:
  "accept"  — pattern is correct, targeted, and style-invariant.
  "reject"  — fundamental problem; do not revise (state why).
  "revise"  — good intent but needs adjustment; provide revised_rule_text.

Return JSON:
{{
  "verdict": "accept" | "reject" | "revise",
  "rationale": "<one sentence citing specific FP/FN evidence>",
  "fp_evidence": "S<N>: <paraphrase of evidence sentence>" | null,
  "fn_evidence": "S<N>: <paraphrase of evidence sentence>" | null,
  "revised_rule_text": "<revised rule, if verdict=revise>" | null,
  "removal_targets": [{{"pattern_id": "p_XXX", "reason": "<one sentence>"}}]
}}
JSON only:"""


def _run_assessor(llm: LLMClient, proposal: dict, bank: dict,
                  fp_sentences: list, fn_sentences: list, od_fms: list,
                  sent_map: dict, id_to_abstract: dict,
                  backend_str: str, model_str: str,
                  dry_run: bool = False) -> dict:
    """Run Assessor on one proposal. Returns verdict dict.

    If verdict=revise: re-runs once with revised_rule_text (max 1 revision cycle).
    """
    if dry_run or llm is None:
        return {"verdict": "reject", "rationale": "dry-run: no assessor call", "revised_rule_text": None}

    def _run_once(rule_text: str, is_revision: bool = False) -> dict:
        slot = proposal.get("slot", "?")
        fm_ids = proposal.get("addresses_failure_modes", [])
        example_block = proposal.get("example_block", "")

        fm_lookup = {fm.get("id", ""): fm for fm in od_fms if fm.get("id")}
        fm_details = "\n".join(
            f"  {fid}: {fm_lookup[fid].get('title', '?')} — {fm_lookup[fid].get('symptom', '?')}"
            for fid in fm_ids if fid in fm_lookup
        ) or "  (no matching FM details)"

        slot_pats = bank.get("slot_patterns", {}).get(slot, [])
        def _fmt_existing(p):
            line = f"  {p.get('pattern_id', '?')}: {p.get('rule_text', '')[:120]}"
            wfp = p.get("worked_fp_evidence")
            wfn = p.get("worked_fn_evidence")
            if wfp:
                line += f"\n      (orig FP: {str(wfp)[:120]})"
            if wfn:
                line += f"\n      (orig FN: {str(wfn)[:120]})"
            return line
        existing = "\n".join(_fmt_existing(p) for p in slot_pats[:5]) or "  (empty)"

        def _sent_sample(pairs, n=5) -> str:
            lines = []
            for s_num, c_id in pairs[:n]:
                s = sent_map.get(s_num)
                prev = sent_map.get(s_num - 1)
                if s:
                    ctx = f"  S{s_num}: {s.text}"
                    if prev:
                        ctx = f"  S{s_num-1}: {prev.text}\n" + ctx
                    lines.append(ctx)
            return "\n".join(lines) or "  (none)"

        prompt = ASSESSOR_PROMPT.format(
            slot=slot,
            rule_text=rule_text,
            example_block=example_block,
            fm_ids=", ".join(fm_ids),
            fm_details=fm_details,
            slot_patterns=existing,
            fp_sample=_sent_sample(fp_sentences),
            fn_sample=_sent_sample(fn_sentences),
        )

        ok, hits = gate06_ok(prompt)
        if not ok:
            print(f"  [Assessor] WARNING: prompt taboo tokens {hits!r}")

        data = llm.extract_json(llm.query(prompt, timeout=120)) or {}
        return data if isinstance(data, dict) else {}

    result = _run_once(proposal.get("rule_text", ""))
    verdict = result.get("verdict", "reject").lower()

    if verdict == "revise" and result.get("revised_rule_text"):
        revised = result["revised_rule_text"]
        print(f"  [Assessor] revise → re-running with revised rule...")
        result2 = _run_once(revised, is_revision=True)
        verdict2 = result2.get("verdict", "reject").lower()
        result2["original_rule_text"] = proposal.get("rule_text", "")
        result2["revised_rule_text"] = revised
        result2["revision_cycle"] = 1
        if verdict2 == "accept":
            result2["_accepted_rule_text"] = revised
        return result2

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Bank mutation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_proposals(bank: dict, proposals: list[dict]) -> dict:
    import copy
    bank = copy.deepcopy(bank)
    sp = bank.setdefault("slot_patterns", {})
    for slot in SLOT_NAMES:
        sp.setdefault(slot, [])
    for prop in proposals:
        slot = prop.get("slot")
        if slot not in SLOT_NAMES:
            continue
        rule_text = prop.get("_accepted_rule_text") or prop.get("rule_text", "")
        pid = _next_pattern_id(bank)
        sp[slot].append({
            "pattern_id": pid,
            "rule_text": rule_text,
            "example_block": prop.get("example_block", ""),
            "why_it_transfers": prop.get("why_it_transfers", ""),
            "abstraction_check_cot": prop.get("abstraction_check_cot", ""),
            "addresses_failure_modes": prop.get("addresses_failure_modes", []),
            "worked_fp_evidence": prop.get("_worked_fp_evidence"),
            "worked_fn_evidence": prop.get("_worked_fn_evidence"),
        })
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
# Outer training pass (v5)
# ─────────────────────────────────────────────────────────────────────────────

def run_outer_pass(
    pass_num: int,
    train_projects: list[str],
    test_projects: list[str],
    split_dir: Path,
    backend: LLMBackend,
    model: str | None,
    backend_str: str,
    model_str: str,
    prior_train_f1s: dict[str, float],
    prior_train_errors: dict[str, int] | None = None,
    dry_run: bool = False,
    split_name: str = "mainline",
    covered_fm_titles: list[str] | None = None,
) -> dict:
    """One outer pass: L(train) → OD → Assessor → Commit. Then L(test) for [TEST] log.

    Returns pass summary dict.
    """
    print(f"\n{'='*60}")
    print(f"OUTER PASS {pass_num} | train={train_projects} | test={test_projects}"
          f" | dry_run={dry_run}")
    print(f"{'='*60}")

    llm = LLMClient(backend=backend, model=model) if not dry_run else None
    project_banks = {p: _load_bank(split_dir, p) for p in train_projects}

    # ── Step 1: L(train) ────────────────────────────────────────────────────
    print("\n[L-TRAIN] Running linker on training projects...")
    l_train: dict[str, dict] = {}
    for project in train_projects:
        print(f"\n  [L-TRAIN] {project}...", end=" ", flush=True)
        r = _run_linker_l(project, backend, model, project_banks[project], dry_run=dry_run)
        l_train[project] = r
        print(f"F1={r['F1']:.4f}")

    train_f1s = {p: l_train[p]["F1"] for p in train_projects}
    macro_train = sum(train_f1s.values()) / max(1, len(train_f1s))
    prior_macro = sum(prior_train_f1s.values()) / max(1, len(prior_train_f1s)) if prior_train_f1s else 0.0
    delta = macro_train - prior_macro

    train_errors = {p: l_train[p]["fp_count"] + l_train[p]["fn_count"] for p in train_projects}
    total_train_errors = sum(train_errors.values())
    prior_errors = prior_train_errors or {}
    prior_total_errors = sum(prior_errors.values()) if prior_errors else None
    delta_errors = (total_train_errors - prior_total_errors) if prior_total_errors is not None else None

    print(f"\n[TRAIN] macro F1: {macro_train:.4f} (delta: {delta:+.4f})")
    for p in train_projects:
        print(f"  [TRAIN] {p}: F1={train_f1s[p]:.4f}")
    print(f"[TRAIN] total errors: {total_train_errors}"
          + (f" (delta: {delta_errors:+d})" if delta_errors is not None else ""))

    # ── Step 2: OD per training project ────────────────────────────────────
    print("\n[OD] Running OD Analyst per training project...")
    od_results: dict[str, dict] = {}
    seen_fm_titles: list[str] = list(covered_fm_titles or [])
    for project in train_projects:
        print(f"\n  [OD] {project}...")
        od = _run_od(llm=llm, project=project, l_run=l_train[project],
                     bank=project_banks[project], pass_num=pass_num,
                     split_name=split_name, macro_f1=macro_train, delta=delta,
                     backend_str=backend_str, model_str=model_str, dry_run=dry_run,
                     covered_fm_titles=seen_fm_titles)
        od_results[project] = od
        n_fm = len(od.get("failure_modes", []))
        n_prop = len(od.get("patterns_proposed", []))
        print(f"  [OD] {project}: {n_fm} failure modes, {n_prop} proposals")
        for fm in od.get("failure_modes", []):
            t = (fm.get("title") or "").strip()
            if t and t not in seen_fm_titles:
                seen_fm_titles.append(t)
        od_path = split_dir / f"pass{pass_num}_{project}_od.json"
        od_path.parent.mkdir(parents=True, exist_ok=True)
        od_path.write_text(json.dumps(od, indent=2))

    # ── Step 3: Collect proposals, run Assessor ─────────────────────────────
    print("\n[Assessor] Running Assessor on proposals...")
    all_accepted: list[dict] = []
    all_rejected: list[dict] = []
    all_removals: list[dict] = []
    assessor_decisions: list[dict] = []
    seen_keys: set[str] = set()

    for project in train_projects:
        od = od_results[project]
        fps = l_train[project].get("fps", [])
        fns = l_train[project].get("fns", [])
        od_fms = od.get("failure_modes", [])

        # Load sentence map for assessor context
        from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
        from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
        paths = _ra.DATASETS[project]
        sentences = load_sentences(str(paths["text"]))
        sent_map = build_sent_map(sentences)
        components = parse_pcm_repository(str(paths["model"]))
        id_to_abstract = {c.id: f"comp_{i}" for i, c in enumerate(components)}

        for prop in od.get("patterns_proposed", []):
            if not isinstance(prop, dict):
                continue
            slot = prop.get("slot", "")
            rule_text = prop.get("rule_text", "")

            # GATE-06 taboo check
            ok06, hits = gate06_ok(f"{rule_text} {prop.get('example_block', '')}")
            if not ok06:
                print(f"  [GATE-06 REJECT] slot={slot}: {hits!r}")
                all_rejected.append({**prop, "rejection_reason": f"gate06: {hits!r}",
                                      "_project": project})
                continue
            if slot not in SLOT_NAMES:
                print(f"  [GATE-06 REJECT] unknown slot {slot!r}")
                all_rejected.append({**prop, "rejection_reason": f"unknown slot: {slot!r}",
                                      "_project": project})
                continue

            # Dedup
            key = (slot + rule_text[:60]).lower()
            if key in seen_keys:
                continue
            seen_keys.add(key)

            prop = {**prop, "_project": project}
            verdict = _run_assessor(
                llm=llm, proposal=prop, bank=project_banks[project],
                fp_sentences=fps, fn_sentences=fns, od_fms=od_fms,
                sent_map=sent_map, id_to_abstract=id_to_abstract,
                backend_str=backend_str, model_str=model_str, dry_run=dry_run,
            )
            v = verdict.get("verdict", "reject").lower()
            assessor_decisions.append({
                "project": project, "slot": slot,
                "verdict": v,
                "rationale": verdict.get("rationale", ""),
                "fp_evidence": verdict.get("fp_evidence"),
                "fn_evidence": verdict.get("fn_evidence"),
                "removal_targets": verdict.get("removal_targets", []),
            })
            print(f"  [Assessor] {project} slot={slot}: {v} — {verdict.get('rationale', '')[:80]}")
            enriched = {**prop,
                        "_worked_fp_evidence": verdict.get("fp_evidence"),
                        "_worked_fn_evidence": verdict.get("fn_evidence")}
            if v == "accept":
                all_accepted.append(enriched)
            elif v == "revise" and verdict.get("_accepted_rule_text"):
                all_accepted.append({**enriched, "_accepted_rule_text": verdict["_accepted_rule_text"]})
            else:
                all_rejected.append({**prop, "assessor_rejection": verdict.get("rationale", "")})

            for rt in verdict.get("removal_targets", []) or []:
                if isinstance(rt, dict):
                    rid = rt.get("pattern_id", "")
                    if rid and all(r.get("pattern_id") != rid for r in all_removals):
                        all_removals.append({**rt, "_source": "assessor"})

        for rem in od.get("patterns_to_remove", []):
            rid = rem.get("pattern_id", "")
            if rid and all(r.get("pattern_id") != rid for r in all_removals):
                all_removals.append(rem)

    print(f"\n[Assessor] accepted={len(all_accepted)} rejected={len(all_rejected)} removals={len(all_removals)}")

    # Save assessor log
    assessor_log_path = split_dir / f"pass{pass_num}_assessor.json"
    assessor_log_path.write_text(json.dumps({
        "decisions": assessor_decisions,
        "accepted": len(all_accepted),
        "rejected": len(all_rejected),
        "removals": len(all_removals),
    }, indent=2))

    # ── Step 4: Commit ──────────────────────────────────────────────────────
    did_commit = bool(all_accepted or all_removals)
    if did_commit:
        for project in train_projects:
            cb = _apply_proposals(project_banks[project], all_accepted)
            cb = _apply_removals(cb, all_removals)
            if not dry_run:
                _save_bank(split_dir, project, cb)
                print(f"  [Commit] bank saved: {project} ({_total_patterns(cb)} patterns)")
            project_banks[project] = cb
        committed_f1s = train_f1s
        print(f"  [Commit] COMMITTED {len(all_accepted)} patterns + {len(all_removals)} removals")
    if not dry_run:
        _snapshot_banks(split_dir, pass_num, train_projects)
    if not did_commit:
        print("  [Commit] no-op — no patterns accepted, no removals")
        committed_f1s = prior_train_f1s if prior_train_f1s else train_f1s

    committed_macro = sum(committed_f1s.values()) / max(1, len(committed_f1s))

    # ── Step 5: L(test) — [TEST] eval ───────────────────────────────────────
    test_f1s = _eval_test_projects(test_projects, backend, model, project_banks, dry_run)

    # ── Convergence check ────────────────────────────────────────────────────
    converged = (
        not did_commit
        and pass_num >= 2
        and bool(prior_errors)
        and total_train_errors >= sum(prior_errors.values())
    )

    summary = _make_pass_summary(
        pass_num, split_name, train_projects, test_projects, dry_run,
        prior_errors, train_errors, total_train_errors, delta_errors,
        train_f1s, macro_train, delta, committed_f1s, committed_macro,
        test_f1s, len(all_accepted), len(all_rejected), len(all_removals),
        len(assessor_decisions), did_commit, converged,
        split_dir=split_dir,
    )
    summary["fm_titles_seen"] = seen_fm_titles
    (split_dir / f"pass{pass_num}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _eval_test_projects(test_projects: list[str], backend: LLMBackend,
                         model: str | None, project_banks: dict,
                         dry_run: bool) -> dict[str, float]:
    """Evaluate test projects with current committed bank. Returns {project: F1}."""
    if not test_projects:
        return {}
    print("\n[L-TEST] Evaluating test projects...")
    # Use the first train project's bank (shared across training projects)
    # or empty bank if no train banks. The committed bank is the union of training.
    # For test: use first available bank (they're equivalent in single-bank mode).
    test_bank = next(iter(project_banks.values())) if project_banks else {
        "version": "v5", "slot_patterns": {s: [] for s in SLOT_NAMES}
    }
    test_f1s: dict[str, float] = {}
    for project in test_projects:
        print(f"  [L-TEST] {project}...", end=" ", flush=True)
        r = _run_linker_l(project, backend, model, test_bank, dry_run=dry_run)
        test_f1s[project] = r["F1"]
        print(f"F1={r['F1']:.4f}")
    macro_test = sum(test_f1s.values()) / max(1, len(test_f1s)) if test_f1s else 0.0
    print(f"[TEST] macro F1: {macro_test:.4f}")
    for p in test_projects:
        print(f"  [TEST] {p}: F1={test_f1s[p]:.4f}")
    return test_f1s


def _make_pass_summary(
    pass_num, split_name, train_projects, test_projects, dry_run,
    prior_errors, train_errors, total_train_errors, delta_errors,
    train_f1s, macro_train, delta, committed_f1s, committed_macro,
    test_f1s, n_accepted, n_rejected, n_removals, n_assessor_decisions,
    did_commit, converged, split_dir: Path,
) -> dict:
    macro_test = sum(test_f1s.values()) / max(1, len(test_f1s)) if test_f1s else 0.0
    summary = {
        "pass": pass_num,
        "split": split_name,
        "train_projects": train_projects,
        "test_projects": test_projects,
        "dry_run": dry_run,
        "train_errors_before": prior_errors,
        "train_errors_after_l": train_errors,
        "total_train_errors": total_train_errors,
        "delta_errors_from_prior": delta_errors,
        "train_f1s": train_f1s,
        "macro_train_f1": macro_train,
        "delta_train_f1_from_prior": delta,
        "committed_f1s": committed_f1s,
        "committed_macro_f1": committed_macro,
        "test_f1s": test_f1s,
        "macro_test_f1": macro_test,
        "assessor_decisions": n_assessor_decisions,
        "proposals_accepted": n_accepted,
        "proposals_rejected": n_rejected,
        "removals": n_removals,
        "committed": did_commit,
        "converged": converged,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    path = split_dir / f"pass{pass_num}_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    print(f"\n[Pass {pass_num}] [TRAIN] macro={macro_train:.4f}  [TEST] macro={macro_test:.4f}"
          f"  committed={did_commit}  converged={converged}"
          f"  assessor_decisions={n_assessor_decisions}  accepted={n_accepted}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Tier runners
# ─────────────────────────────────────────────────────────────────────────────

def run_probe(train_projects: list[str], test_projects: list[str],
              backend: LLMBackend, model: str | None,
              dry_run: bool = False, split_name: str = "mainline") -> dict:
    """Probe tier: PROBE_PASSES passes. Returns probe summary."""
    backend_str = "openai" if backend == LLMBackend.OPENAI else "claude"
    model_str = model or "default"
    split_dir = OUT_ROOT / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[PROBE TIER] train={train_projects} test={test_projects}"
          f" backend={backend_str} model={model_str}")

    # Axiom-only baseline for dynamic kill threshold (per-project, no bank)
    print("\n[PROBE] Computing axiom-only baseline for test projects...")
    axiom_test_f1s = _run_axiom_only_baseline(test_projects, backend, model, dry_run=dry_run)
    axiom_test_macro = sum(axiom_test_f1s.values()) / max(1, len(axiom_test_f1s))
    print(f"  [axiom] test macro={axiom_test_macro:.4f}  "
          + "  ".join(f"{p}={axiom_test_f1s[p]:.4f}" for p in test_projects))

    prior_f1s: dict[str, float] = {}
    prior_errors: dict[str, int] = {}
    covered_fms: list[str] = []
    pass_summaries = []

    for pass_num in range(1, PROBE_PASSES + 1):
        summary = run_outer_pass(
            pass_num=pass_num,
            train_projects=train_projects,
            test_projects=test_projects,
            split_dir=split_dir,
            backend=backend, model=model,
            backend_str=backend_str, model_str=model_str,
            prior_train_f1s=prior_f1s,
            prior_train_errors=prior_errors,
            dry_run=dry_run,
            split_name=split_name,
            covered_fm_titles=covered_fms,
        )
        pass_summaries.append(summary)
        prior_f1s = {p: summary["committed_f1s"].get(p, 0.0) for p in train_projects}
        prior_errors = {p: summary["train_errors_after_l"].get(p, 0) for p in train_projects}
        covered_fms = list(summary.get("fm_titles_seen", covered_fms))
        if summary.get("converged"):
            print(f"\n[PROBE] Converged at pass {pass_num}")
            break

    best_idx = max(range(len(pass_summaries)),
                   key=lambda i: pass_summaries[i].get("committed_macro_f1", 0.0))
    best_pass = pass_summaries[best_idx]["pass"]
    if not dry_run:
        _restore_best_banks(split_dir, best_pass, train_projects)
        print(f"\n[PROBE] best-bank checkpoint restored: pass {best_pass} "
              f"(macro={pass_summaries[best_idx]['committed_macro_f1']:.4f})")

    final_train = pass_summaries[best_idx]["committed_macro_f1"]
    final_test = pass_summaries[best_idx]["macro_test_f1"]
    best_test_f1s = pass_summaries[best_idx].get("test_f1s", {})
    any_assessor_active = any(s.get("assessor_decisions", 0) > 0 for s in pass_summaries)

    # Dynamic kill: mean(bank_F1[p] / axiom_F1[p]) < 0.95 across test projects.
    # Harder projects (low axiom floor) are judged relative to their own baseline,
    # so a single weak dataset cannot anchor-kill the whole probe.
    per_project_ratios = {
        p: best_test_f1s.get(p, 0.0) / max(axiom_test_f1s.get(p, 1e-6), 1e-6)
        for p in test_projects
    }
    mean_ratio = sum(per_project_ratios.values()) / max(1, len(per_project_ratios))
    kill = mean_ratio < 0.95
    ratio_str = "  ".join(f"{p}={per_project_ratios[p]:.3f}" for p in test_projects)
    print(f"\n[PROBE] kill-check: mean_ratio={mean_ratio:.3f} (threshold=0.95)  {ratio_str}")
    verdict = "KILL" if kill else ("CONTINUE" if any(s.get("committed") for s in pass_summaries) else "MARGINAL")

    probe_summary = {
        "tier": "probe",
        "split": split_name,
        "train_projects": train_projects,
        "test_projects": test_projects,
        "passes_run": len(pass_summaries),
        "final_train_macro_f1": final_train,
        "final_test_macro_f1": final_test,
        "axiom_test_f1s": axiom_test_f1s,
        "per_project_kill_ratios": per_project_ratios,
        "mean_kill_ratio": mean_ratio,
        "kill_threshold": 0.95,
        "assessor_active": any_assessor_active,
        "verdict": verdict,
        "pass_summaries": pass_summaries,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (split_dir / "probe_summary.json").write_text(json.dumps(probe_summary, indent=2))
    print(f"\n[PROBE] verdict={verdict}  [TRAIN] macro={final_train:.4f}  [TEST] macro={final_test:.4f}")
    return probe_summary


def run_range(train_projects: list[str], test_projects: list[str],
              backend: LLMBackend, model: str | None,
              dry_run: bool = False, split_name: str = "mainline") -> dict:
    """Range tier: up to MAX_OUTER_PASSES, stop at convergence."""
    backend_str = "openai" if backend == LLMBackend.OPENAI else "claude"
    model_str = model or "default"
    split_dir = OUT_ROOT / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[RANGE TIER] train={train_projects} test={test_projects}"
          f" backend={backend_str} model={model_str}")

    prior_f1s: dict[str, float] = {}
    prior_errors: dict[str, int] = {}
    covered_fms: list[str] = []
    pass_summaries = []

    for pass_num in range(1, MAX_OUTER_PASSES + 1):
        summary = run_outer_pass(
            pass_num=pass_num,
            train_projects=train_projects,
            test_projects=test_projects,
            split_dir=split_dir,
            backend=backend, model=model,
            backend_str=backend_str, model_str=model_str,
            prior_train_f1s=prior_f1s,
            prior_train_errors=prior_errors,
            dry_run=dry_run,
            split_name=split_name,
            covered_fm_titles=covered_fms,
        )
        pass_summaries.append(summary)
        prior_f1s = {p: summary["committed_f1s"].get(p, 0.0) for p in train_projects}
        prior_errors = {p: summary["train_errors_after_l"].get(p, 0) for p in train_projects}
        covered_fms = list(summary.get("fm_titles_seen", covered_fms))

        if summary.get("converged"):
            print(f"\n[RANGE] Stopped at pass {pass_num} (converged)"
                  f" macro_train={summary['committed_macro_f1']:.4f}")
            break

    best_idx = max(range(len(pass_summaries)),
                   key=lambda i: pass_summaries[i].get("committed_macro_f1", 0.0))
    best_pass = pass_summaries[best_idx]["pass"]
    if not dry_run:
        _restore_best_banks(split_dir, best_pass, train_projects)
        print(f"\n[RANGE] best-bank checkpoint restored: pass {best_pass} "
              f"(macro={pass_summaries[best_idx]['committed_macro_f1']:.4f})")

    final_train = pass_summaries[best_idx]["committed_macro_f1"]
    final_test = pass_summaries[best_idx]["macro_test_f1"]
    any_revise = any(
        any(d.get("verdict") == "revise" for d in s.get("assessor_decisions_list", []))
        for s in pass_summaries
    )

    tier_verdict = "STRONG" if final_test >= 0.9173 else ("WEAK" if final_test >= 0.87 else "FAIL")
    range_summary = {
        "tier": "range",
        "split": split_name,
        "train_projects": train_projects,
        "test_projects": test_projects,
        "passes_run": len(pass_summaries),
        "final_train_macro_f1": final_train,
        "final_test_macro_f1": final_test,
        "tier_verdict": tier_verdict,
        "converged": pass_summaries[-1].get("converged", False),
        "pass_summaries": pass_summaries,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (split_dir / "range_summary.json").write_text(json.dumps(range_summary, indent=2))
    print(f"\n[RANGE] verdict={tier_verdict}  [TRAIN] macro={final_train:.4f}"
          f"  [TEST] macro={final_test:.4f}")
    return range_summary


def run_confirmation(split_name: str, train_projects: list[str],
                     test_projects: list[str], backend: LLMBackend,
                     model: str | None, dry_run: bool = False) -> dict:
    """Confirmation tier: independent per-split training with axiom-only baseline.

    SC5 (REQ-V26-05): trains on train_projects, evaluates on test_projects vs
    axiom-only baseline. Verdict per split: does training improve test F1 beyond axiom-only?
    """
    backend_str = "openai" if backend == LLMBackend.OPENAI else "claude"
    model_str = model or "default"
    split_dir = OUT_ROOT / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[CONFIRMATION TIER] split={split_name}")
    print(f"  train={train_projects} test={test_projects}")

    # Axiom-only baseline on test projects (computed once before training)
    print("\n[Confirmation] Computing axiom-only baseline on test projects...")
    axiom_f1s = _run_axiom_only_baseline(test_projects, backend, model, dry_run=dry_run)
    axiom_macro = sum(axiom_f1s.values()) / max(1, len(axiom_f1s)) if axiom_f1s else 0.0
    print(f"  Axiom-only baseline: macro={axiom_macro:.4f}")

    # Train on train_projects to convergence
    prior_f1s: dict[str, float] = {}
    prior_errors: dict[str, int] = {}
    covered_fms: list[str] = []
    pass_summaries = []

    for pass_num in range(1, MAX_OUTER_PASSES + 1):
        summary = run_outer_pass(
            pass_num=pass_num,
            train_projects=train_projects,
            test_projects=test_projects,
            split_dir=split_dir,
            backend=backend, model=model,
            backend_str=backend_str, model_str=model_str,
            prior_train_f1s=prior_f1s,
            prior_train_errors=prior_errors,
            dry_run=dry_run,
            split_name=split_name,
            covered_fm_titles=covered_fms,
        )
        pass_summaries.append(summary)
        prior_f1s = {p: summary["committed_f1s"].get(p, 0.0) for p in train_projects}
        prior_errors = {p: summary["train_errors_after_l"].get(p, 0) for p in train_projects}
        covered_fms = list(summary.get("fm_titles_seen", covered_fms))

        if summary.get("converged"):
            print(f"\n[Confirmation] Stopped at pass {pass_num} (converged)")
            break

    best_idx = max(range(len(pass_summaries)),
                   key=lambda i: pass_summaries[i].get("committed_macro_f1", 0.0))
    best_pass = pass_summaries[best_idx]["pass"]
    if not dry_run:
        _restore_best_banks(split_dir, best_pass, train_projects)
        print(f"\n[Confirmation] best-bank checkpoint restored: pass {best_pass} "
              f"(macro={pass_summaries[best_idx]['committed_macro_f1']:.4f})")

    final_test = pass_summaries[best_idx]["macro_test_f1"]
    final_test_f1s = pass_summaries[best_idx]["test_f1s"]
    final_train = pass_summaries[best_idx]["committed_macro_f1"]
    lift = final_test - axiom_macro

    split_verdict = "IMPROVE" if lift > 0.005 else ("NEUTRAL" if lift > -0.01 else "REGRESS")
    conf_summary = {
        "tier": "confirmation",
        "split": split_name,
        "train_projects": train_projects,
        "test_projects": test_projects,
        "passes_run": len(pass_summaries),
        "axiom_only_baseline": {"macro": axiom_macro, "per_project": axiom_f1s},
        "trained_test_macro_f1": final_test,
        "trained_test_f1s": final_test_f1s,
        "trained_train_macro_f1": final_train,
        "lift_vs_axiom_pp": round(lift * 100, 2),
        "split_verdict": split_verdict,
        "pass_summaries": pass_summaries,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (split_dir / "confirmation_summary.json").write_text(json.dumps(conf_summary, indent=2))
    print(f"\n[Confirmation] split={split_name}  verdict={split_verdict}")
    print(f"  axiom-only={axiom_macro:.4f} → trained={final_test:.4f}"
          f"  lift={lift*100:+.2f}pp")
    return conf_summary


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

    # probe / range share most args
    for tier in ("probe", "range"):
        p = sub.add_parser(tier, help=f"Run {tier} tier")
        p.add_argument("--projects", default=",".join(MAINLINE_TRAIN))
        p.add_argument("--test-projects", default=",".join(MAINLINE_TEST))
        p.add_argument("--backend", default="openai", choices=["openai", "claude"])
        p.add_argument("--model", default="gpt-5.4")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--split", default="mainline")

    # confirmation tier
    c = sub.add_parser("confirmation", help="Run confirmation tier (single split)")
    c.add_argument("--split", required=True)
    c.add_argument("--train-projects", required=True)
    c.add_argument("--test-projects", required=True)
    c.add_argument("--backend", default="openai", choices=["openai", "claude"])
    c.add_argument("--model", default="gpt-5.4")
    c.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    os.environ["LLM_BACKEND"] = args.backend
    if args.backend == "openai":
        os.environ["OPENAI_MODEL_NAME"] = args.model
    else:
        os.environ["CLAUDE_MODEL"] = args.model

    backend = LLMBackend.OPENAI if args.backend == "openai" else LLMBackend.CLAUDE

    if args.cmd in ("probe", "range"):
        train_projects = [p.strip() for p in args.projects.split(",") if p.strip()]
        test_projects = [p.strip() for p in args.test_projects.split(",") if p.strip()]
        if args.cmd == "probe":
            run_probe(train_projects, test_projects, backend, args.model,
                      dry_run=args.dry_run, split_name=args.split)
        else:
            run_range(train_projects, test_projects, backend, args.model,
                      dry_run=args.dry_run, split_name=args.split)
    elif args.cmd == "confirmation":
        train_projects = [p.strip() for p in args.train_projects.split(",") if p.strip()]
        test_projects = [p.strip() for p in args.test_projects.split(",") if p.strip()]
        run_confirmation(args.split, train_projects, test_projects, backend, args.model,
                         dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
