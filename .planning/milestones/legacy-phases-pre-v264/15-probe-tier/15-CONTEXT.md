# Phase 15: Probe Tier - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Run β training on the mainline split (MS+TS+TM train, BBB+JAB test) for 1–2 outer passes using gpt-5.4, within the $5–10 budget. Output: a binary CONTINUE / KILL verdict with numeric evidence (per-project F1 + macro F1 after each pass). If KILL: Phase 18 Compact-B is flagged as next action. If CONTINUE: Phase 16 Range Tier proceeds.

This phase consumes no new code — all machinery was built in Phase 14 (`scripts/voyager_train_tlr_v4_beta.py`, `src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py`). Phase 15 is purely operational: invoke the harness, observe results, document verdict.

</domain>

<decisions>
## Implementation Decisions

### Probe Tier Execution Scope
- Run all 3 mainline train projects (MS+TS+TM) in pass 1 — matches REQ-V23-07, cheap-kill logic requires complete macro across train set
- Log to `logs/voyager_v4_beta/probe.log` — consistent with existing `logs/voyager_*/` pattern
- Pass 2 runs if pass-1 training-project macro F1 ≥ 0.80 (not yet killed), per v2.3-ROADMAP Phase 15 SC#3
- Verdict documented in `.planning/phases/15-probe-tier/15-PROBE-VERDICT.md` + STATE.md update

### Plan Structure
- Plan Phase 16/17 AFTER Phase 15 verdict (no wasted planning if KILL path taken)
- 2 plans for Phase 15: (1) run probe harness, (2) document verdict + update state
- Log gpt-5.4 cost in probe.log via LLM client token tracking; summarize in verdict file

### Claude's Discretion
- Exact CLI invocation arguments (--projects order, --passes flag if added, output dirs)
- Whether to run pass 2 immediately after pass 1 in the same script invocation or in a second call
- How to handle partial failures (single project failing mid-run)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/voyager_train_tlr_v4_beta.py` — full Phase 14 harness (L+O+D+P), probe/range CLI modes, dry-run verified
- `src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py` — linker consumer, axiom-only/empty-bank mode
- `results/voyager_v4_beta/` — output root (VOYAGER4B_OUT_ROOT), cache at `results/voyager_v4_beta/cache/`
- Existing `logs/voyager_*/` pattern for log files

### Established Patterns
- Training runs log to `logs/` directory; per-run logs named `<variant>_<tier>.log` (e.g., `voyager_gpt54/distill.log`)
- Phase 15 verdict format mirrors prior probe SUMMARY pattern (`.planning/v2.2-prep/probe-*-SUMMARY.md`)
- Cost tracking: LLM client has token-level tracking; prior probes estimated ~$2-3 per project per pass on gpt-5.4

### Integration Points
- `run_ablation.py` CANONICAL_VARIANTS + VARIANT_SPECS — `s_linker14_voyager` already registered (Phase 14)
- GATE-06 helpers (`gate06_ok`, `reviewer_critic_stub`) callable from within harness — Phase 15 activates REAL reviewer_critic LLM call (replaces stub)
- Per-(text_stem, comp_hash, backend, model) cache at VOYAGER4B_CACHE_ROOT — O and D outputs cached

</code_context>

<specifics>
## Specific Ideas

- GATE-06 reviewer_critic activates real LLM in Phase 15 (stub was Phase 14 only). Check if `reviewer_critic_stub` in the harness needs to be promoted to a real LLM call or if it can be left as-is for the probe tier (advisory, doesn't block).
- Per-project `_bank.json` at end of each pass must be persisted (SC#5). Verify harness writes these — confirmed in Phase 14 dry-run.
- Budget: prior v2.2 probes ran ~$1-2 per project per pass on gpt-5.4. 3 projects × 2 passes × ~$1.50 ≈ $9 — within $10 cap.

</specifics>

<deferred>
## Deferred Ideas

- Phase 16 Range Tier planning — deferred until Phase 15 verdict (CONTINUE or KILL decision)
- Phase 17 Confirmation Tier — deferred until Phase 16 verdict
- Compact-B (Phase 18) implementation — deferred unless Phase 15 returns KILL
- Claude cross-model re-test of voyager bank — explicitly out of scope per backend policy

</deferred>
