# Phase 16: Range Tier — Context

**Gathered:** 2026-06-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Run β training to convergence on the mainline split (MS+TS+TM train, BBB+JAB test) using the Range tier of `scripts/voyager_train_tlr_v4_beta.py`. Convergence = D proposes zero net patterns (accepted+removals == 0), OR pass 5 cap. After convergence: aggregate per-project banks into `final_bank.json`, evaluate `s_linker14_voyager` on all 5 datasets (gpt-5.4), compute 3-tier verdict (STRONG ≥ 0.9173 / WEAK [0.87, 0.9173) / FAIL < 0.87), and compare vs `prompts_v3_axiom` (axiom-only floor) per REQ-V23-15.

Budget cap: $15–25 gpt-5.4 total for this phase.

**Key continuity from Phase 15:**
- Probe converged at pass 1 (training macro F1 = 0.9152 > CONVERGENCE_THRESHOLD 0.90).
- Per-project probe banks already exist at `results/voyager_v4_beta/mainline/{mediastore,teastore,teammates}_bank.json`.
- `run_range` loads existing banks from disk at start of pass 1 — automatic warm-start, no code change needed.
- 3 real LLM-generated patterns in probe banks: p_001 DOC_KNOWLEDGE_EXTRACTION_RULES, p_002 DOC_KNOWLEDGE_JUDGE_RULES, p_003 VALIDATION_RULES (from mediastore; cross-project replicated to TS+TM).
- mediastore bank has 3 dry-run placeholder patterns (p_004/p_005/p_006 in AMBIGUITY_RULES) — range D role will see these; if unhelpful, D may propose removals.

**Convergence behavior for range:**
- Range convergence = `len(accepted) == 0 and len(removals) == 0` (D has nothing to propose/remove).
- This is NOT the F1 threshold used in probe (probe had an early-exit shortcut; range does not).
- Expect 1–3 passes: pass 1 will run L (high starting F1 from probe banks) → O (fresh failure modes on same data) → D (may propose additional patterns or remove dry-run placeholders) → GATE-06 → P.
- Dry-run placeholder patterns (p_004–p_006) in mediastore bank may be removed by D in pass 1.

**No new code required** — all machinery shipped in Phase 14, operational in Phase 15. Phase 16 is:
1. Run range tier (LLM calls, real cost)
2. Aggregate `final_bank.json` (inline Python or short script)
3. Evaluate all 5 datasets via `run_ablation.py`
4. Document verdict
</domain>

<decisions>
## Implementation Decisions

### Range Execution
- Run `python scripts/voyager_train_tlr_v4_beta.py range --projects mediastore,teastore,teammates --backend openai --model gpt-5.4`
- Log to `logs/voyager_v4_beta/range.log` (consistent with `logs/voyager_v4_beta/probe.log` pattern)
- Range reads existing banks from `results/voyager_v4_beta/mainline/` — warm-start is automatic
- `range_summary.json` written to `results/voyager_v4_beta/mainline/range_summary.json` by harness

### final_bank.json Aggregation
- Union of all patterns from per-project banks, slot-grouped, deduped by pattern_id
- Written to `results/voyager_v4_beta/mainline/final_bank.json`
- Must contain non-empty slot entries for all 9 axiom slots OR mark slots as `[]` if no patterns trained
- This is the bank `s_linker14_voyager` reads by default (`DEFAULT_BANK_PATH` in linker)

### 5-Dataset Evaluation
- Use `run_ablation.py --variants s_linker14_voyager --datasets mediastore teastore teammates bigbluebutton jabref`
- `final_bank.json` at default path → no env override needed
- Axiom-only comparison (REQ-V23-15): run same command with `VOYAGER4B_BANK_PATH=/dev/null` or empty bank override OR use the trained bank for the primary and separately note the probe pass-0 (empty bank) F1 from Phase 14 dry-run (no-LLM) as the axiom floor
- GATE-01 regression: verify `s_linker13_min` is unaffected (run GATE-01 test or note it was verified in Phase 14/15)

### Plan Structure
- 2 plans: (1) range run + final_bank + 5-dataset eval; (2) verdict document + state update
- Verdict document: `.planning/phases/16-range-tier/16-RANGE-VERDICT.md`
- Log gpt-5.4 cost; if `LLMClient.get_session_usage()` unavailable, estimate from pass count × 10 LLM calls/pass

### Claude's Discretion
- Exact aggregation strategy for final_bank.json (slot-level union vs. project-weighted merge)
- Whether to retain or strip dry-run placeholder patterns if D doesn't remove them
- How to handle axiom-only baseline (use Phase 14 dry-run F1 if available, or run empty-bank mode)
- Exact CLI invocation for run_ablation.py (--variants, --datasets flags)
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/voyager_train_tlr_v4_beta.py range` — range CLI mode, warm-start from existing banks
- `src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py` — reads `final_bank.json` by default (`DEFAULT_BANK_PATH = "results/voyager_v4_beta/mainline/final_bank.json"`)
- `run_ablation.py` — runs variants on datasets; `s_linker14_voyager` registered (`experimental=True`)
- Existing probe banks: `results/voyager_v4_beta/mainline/{mediastore,teastore,teammates}_bank.json`
- Existing probe Oracle + Distillator JSONs at `results/voyager_v4_beta/mainline/pass1_*.json`

### Bank Schema (from Phase 14)
- Slot-uniform 9 slots; each slot is a list of pattern dicts `{pattern_id, title, content, created_pass, ...}`
- 9 axiom slots: DOC_KNOWLEDGE_EXTRACTION_RULES, DOC_KNOWLEDGE_JUDGE_RULES, AMBIGUITY_RULES, VALIDATION_RULES, SYNONYM_INJECTION_RULES, COREF_RULES, BOUNDARY_FILTER_RULES, PARTIAL_MATCH_RULES, LINK_JUDGE_RULES
- Aggregation: union by pattern_id across per-project banks (same pattern_id = same pattern; keep once)

### Integration Points
- `s_linker14_voyager` bank_path constructor arg or `VOYAGER4B_BANK_PATH` env override
- `run_ablation.py` reads `DATASETS` dict for project paths (benchmark base)
- GATE-06: taboo-grep + advisory critic — already active in range tier (same as probe)
- s_linker13_min GATE-01: `canonical=True`, Claude macro 0.9506, gpt-5.4 macro 0.9069 — must be verified unchanged

### Key Convergence Invariant
- `run_range` stops when `summary["converged"] == True` (i.e., D proposed 0 patterns and 0 removals)
- F1 threshold (CONVERGENCE_THRESHOLD=0.90) does NOT gate range exit — only probe
- Range may run 1–5 passes; probe banks give high starting F1 so delta pressure on D is low
</code_context>

<specifics>
## Specific Ideas

- If dry-run placeholder patterns (p_004–p_006 in mediastore AMBIGUITY_RULES) cause FPs, D may propose removals in range pass 1. This is expected and correct behavior.
- Axiom-only floor for REQ-V23-15: simplest approach is to record Phase 14 dry-run F1 (0.0 or near-random from mock predictions) as "not usable" and instead run a real axiom-only evaluation using `s_linker14_voyager` with `VOYAGER4B_BANK_PATH` pointing to an empty bank file. Cost: same as the 5-dataset evaluation (~$5-10 more). Alternative: use Phase 14 test results if any real F1 was recorded.
- Cost estimate: 3 projects × (L + O + D + P) per pass × ~$0.50-0.70/call × 1-3 passes = $4-10 for range. 5 datasets × eval run = $5-10 for evaluation. Total: $9-20, within $25 cap.
- GATE-06 advisory warnings for project names expected (same as probe.log lines 235-247). Non-blocking.
</specifics>

<deferred>
## Deferred Ideas

- Phase 17 Confirmation Tier (3-split sweep) — deferred until Phase 16 verdict determines path
- Compact-B (Phase 18) — deferred unless Phase 16 returns FAIL (< 0.87)
- Cross-model Claude evaluation of voyager bank — explicitly out of scope (gpt-5.4 only per backend policy)
- Reviewer_critic upgrade to blocking mode — advisory-only for probe+range per Phase 15 decision
</deferred>
