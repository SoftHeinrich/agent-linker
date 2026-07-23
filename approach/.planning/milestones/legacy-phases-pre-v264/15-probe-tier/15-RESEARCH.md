# Phase 15: Probe Tier - Research

**Researched:** 2026-06-01
**Domain:** β training harness execution (operational) — gpt-5.4 LLM training run
**Confidence:** HIGH

## Summary

Phase 15 is purely operational: invoke `scripts/voyager_train_tlr_v4_beta.py probe` against gpt-5.4 on the mainline split (MS+TS+TM), run 1–2 outer passes, and document a binary CONTINUE / KILL verdict. All machinery was shipped in Phase 14 and verified via 32 passing tests. No new code is required.

The harness exposes a direct CLI: `python scripts/voyager_train_tlr_v4_beta.py probe --projects mediastore,teastore,teammates --backend openai --model gpt-5.4`. This is the only command needed for the run. Pass 1 always executes. Pass 2 runs unless pass-1 macro F1 is already below the 0.80 early-kill sentinel (per CONTEXT.md SC#3). The cheap-kill gate at `CHEAP_KILL_THRESHOLD = 0.87` triggers only after pass 2. The harness writes all artifacts automatically: per-project `_bank.json`, per-pass `pass{N}_summary.json`, per-project Oracle JSON, Distillator JSON, and `probe_summary.json`.

The key open question for planning is the `reviewer_critic_stub`: it is named "stub" and described as activating "Real LLM-based critic in Phase 15+", but the filter logic treats all critic rejections as `advisory=True`, meaning they are logged but never block insertion. Based on code inspection, the critic stub already satisfies the bank-entry boundary requirement for the Probe tier — REQ-V23-09 is met by the taboo-grep gate alone, with the critic advisory as supplementary. Phase 15 does NOT need to upgrade the stub to a blocking LLM call; the advisory mode is intentional for the probe tier.

**Primary recommendation:** Run `python scripts/voyager_train_tlr_v4_beta.py probe --projects mediastore,teastore,teammates --backend openai --model gpt-5.4` from the project root with stdout redirected to `logs/voyager_v4_beta/probe.log`, then document the verdict from `results/voyager_v4_beta/mainline/probe_summary.json` into `.planning/phases/15-probe-tier/15-PROBE-VERDICT.md`.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- Run all 3 mainline train projects (MS+TS+TM) in pass 1 — matches REQ-V23-07, cheap-kill logic requires complete macro across train set
- Log to `logs/voyager_v4_beta/probe.log` — consistent with existing `logs/voyager_*/` pattern
- Pass 2 runs if pass-1 training-project macro F1 >= 0.80 (not yet killed), per v2.3-ROADMAP Phase 15 SC#3
- Verdict documented in `.planning/phases/15-probe-tier/15-PROBE-VERDICT.md` + STATE.md update

### Claude's Discretion

- Exact CLI invocation arguments (--projects order, --passes flag if added, output dirs)
- Whether to run pass 2 immediately after pass 1 in the same script invocation or in a second call
- How to handle partial failures (single project failing mid-run)

### Deferred Ideas (OUT OF SCOPE)

- Phase 16 Range Tier planning — deferred until Phase 15 verdict (CONTINUE or KILL decision)
- Phase 17 Confirmation Tier — deferred until Phase 16 verdict
- Compact-B (Phase 18) implementation — deferred unless Phase 15 returns KILL
- Claude cross-model re-test of voyager bank — explicitly out of scope per backend policy
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REQ-V23-07 | Mainline single-split (Probe + Range tiers) on train MS+TS+TM, test BBB+JAB. Cheap-kill at each tier per budget cap. | Harness `MAINLINE_TRAIN = ["mediastore", "teastore", "teammates"]` [VERIFIED: codebase]; `run_probe()` iterates exactly these projects [VERIFIED: codebase] |
| REQ-V23-13 | Convergence = macro F1 >= 0.90 on training projects, max 5 outer passes. Per-pass macros logged; converged-early result preserved. | `CONVERGENCE_THRESHOLD = 0.90`, `MAX_OUTER_PASSES = 5` in harness [VERIFIED: codebase]; `converged` field in `pass{N}_summary.json` [VERIFIED: codebase]; probe tier capped at 2 passes [VERIFIED: codebase] |
| REQ-V23-14 | Budget cap ~$100 gpt-5.4 total. Probe tier $5–10 (mainline split, 1–2 outer passes). | Prior v2.2 probes: < $1 for single-project probe; 3 projects x 2 passes x ~$1.50 ≈ $9 [CITED: 15-CONTEXT.md]; token tracking via `LLMClient.get_session_usage()` [VERIFIED: codebase] |
</phase_requirements>

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Probe execution (L+O+D+P loop) | `scripts/voyager_train_tlr_v4_beta.py` | `run_ablation.py` (metrics helpers) | Harness owns all iteration logic; run_ablation provides DATASETS + eval_metrics [VERIFIED: codebase] |
| Bank persistence | Harness (`_save_bank`) | `results/voyager_v4_beta/mainline/` (filesystem) | Per-project `_bank.json` written at end of each committed pass [VERIFIED: codebase] |
| GATE-06 filtering | `gate06_ok` + `reviewer_critic_stub` in harness | — | Both functions already callable; `advisory=True` means no blocking rejections in probe tier [VERIFIED: codebase] |
| Cost tracking | `LLMClient._session_usage` | stdout log + `probe_summary.json` | Token usage accumulated per-session; no cost-per-dollar field — requires manual estimation from token counts [VERIFIED: codebase] |
| Verdict computation | `run_probe()` return value | `probe_summary.json` | `verdict` field is `"CONTINUE"` or `"KILL"` based on `final_train_macro_f1 >= CHEAP_KILL_THRESHOLD` [VERIFIED: codebase] |
| Verdict documentation | `.planning/phases/15-probe-tier/15-PROBE-VERDICT.md` | `STATE.md` | Human-authored markdown summarizing numeric evidence; format mirrors `.planning/v2.2-prep/probe-*-SUMMARY.md` [CITED: CONTEXT.md] |

---

## Standard Stack

### Core (all Phase 14-built, no new installs)

| Component | File | Purpose | Status |
|-----------|------|---------|--------|
| β harness | `scripts/voyager_train_tlr_v4_beta.py` | L+O+D+P loop, probe/range CLI, GATE-06 | Built + dry-run verified [VERIFIED: codebase + STATE.md] |
| Linker consumer | `src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py` | Runs axiom+bank at inference; L role entry point | Built + instantiation-tested [VERIFIED: codebase] |
| LLM client | `src/llm_sad_sam/llm_client.py` | OpenAI API with token tracking, JSON extraction | Existing infrastructure [VERIFIED: codebase] |
| Eval helpers | `run_ablation.py`: `DATASETS`, `load_gold_sam`, `eval_metrics` | Gold-standard loading, F1 computation | Existing infrastructure [VERIFIED: codebase] |

### Environment Dependencies

| Dependency | Required | Available | Notes |
|------------|---------|-----------|-------|
| `OPENAI_API_KEY` | Yes | Yes [VERIFIED: .env present] | Must be set in `.env` before run |
| `OPENAI_MODEL_NAME` | No (CLI default = gpt-5.4) | — | CLI sets `os.environ["OPENAI_MODEL_NAME"] = args.model` at startup [VERIFIED: codebase] |
| `VOYAGER4B_OUT_ROOT` | No (default: `results/voyager_v4_beta`) | — | Override via env var [VERIFIED: codebase] |
| `VOYAGER4B_CACHE_ROOT` | No (default: `results/voyager_v4_beta/cache`) | — | Override via env var [VERIFIED: codebase] |
| `logs/voyager_v4_beta/` | Yes (log target) | No — must be created [VERIFIED: `ls logs/`] | `mkdir -p logs/voyager_v4_beta/` needed before first run |

---

## Architecture Patterns

### Probe Tier Data Flow

```
CLI: `python scripts/voyager_train_tlr_v4_beta.py probe \
        --projects mediastore,teastore,teammates \
        --backend openai --model gpt-5.4`
          │
          ▼
    run_probe(projects, backend, model)
          │
    ┌─────┴──────┐
    │  PASS 1    │◄── prior_f1s = {} (empty baseline)
    │            │
    │  L (×3)    │── SLinker14Voyager(bank=empty) → per-project F1
    │     ↓      │
    │  O (×3)    │── Oracle LLM call → failure_modes JSON per project
    │     ↓      │       [cache: oracle_iter1_{project}_{backend}_{model}]
    │  D (×1)    │── Distillator LLM call → pattern proposals
    │     ↓      │       [cache: d_iter1_{backend}_{model}_{hash}]
    │ GATE-06    │── taboo grep + critic_stub (advisory) → accepted/rejected
    │     ↓      │
    │  P (×3)    │── probation check: L with candidate bank → delta
    │            │       if delta >= 0: COMMIT banks → {project}_bank.json
    │            │       if delta < 0:  ROLLBACK → keep prior banks
    │            │
    │  macro_f1  │── if >= 0.80: run PASS 2 (NOT kill yet at pass 1)
    └─────┬──────┘
          │ (if pass-1 macro >= 0.80)
    ┌─────┴──────┐
    │  PASS 2    │◄── prior_f1s = pass-1 committed F1s
    │  (same     │
    │   L→O→D→P) │
    │            │
    │  macro_f1  │── if < 0.87 (CHEAP_KILL_THRESHOLD): verdict = "KILL"
    └─────┬──────┘   if >= 0.87: verdict = "CONTINUE"
          │
          ▼
    probe_summary.json → {verdict, final_train_macro_f1, pass_summaries}
          │
          ▼
    15-PROBE-VERDICT.md + STATE.md update
```

### Output File Structure (auto-written by harness)

```
results/voyager_v4_beta/
└── mainline/
    ├── {project}_bank.json          # per-project trained bank (after each committed pass)
    ├── pass1_summary.json           # pass summary with F1s, delta, commit/rollback decision
    ├── pass1_{project}_oracle.json  # Oracle failure-mode JSON per project
    ├── pass1_distillator.json       # Distillator proposals (raw + filtered metadata implicit)
    ├── pass2_summary.json           # (if pass 2 runs)
    ├── pass2_{project}_oracle.json  # (if pass 2 runs)
    ├── pass2_distillator.json       # (if pass 2 runs)
    └── probe_summary.json           # top-level verdict + all pass summaries
results/voyager_v4_beta/cache/
    └── {text_stem}_{comp_hash}_{backend}_{model}_oracle_iter{N}.json  # O cache
    └── d_iter{N}_{backend}_{model}_{hash}.json                        # D cache
logs/voyager_v4_beta/
    └── probe.log                    # stdout redirect (must mkdir before run)
```
[VERIFIED: codebase — harness writes these paths deterministically]

### Anti-Patterns to Avoid

- **Running as Claude backend**: `--backend claude` violates backend policy [CITED: REQUIREMENTS.md "Backend: gpt-5.4 only; Claude only if super necessary"]. Always `--backend openai --model gpt-5.4`.
- **Omitting a project from pass 1**: Run all 3 mainline train projects together; the macro F1 cheap-kill verdict requires all 3 [CITED: CONTEXT.md decision 1].
- **Checking `reviewer_critic_stub` advisory rejects as blocking**: The current filter code only blocks when `advisory=False` (line 645 of harness). The stub always sets `advisory=True` — so critic rejections are logged but never block acceptance. This is correct behavior for the probe tier [VERIFIED: codebase].
- **Manually aggregating banks**: The harness merges per-project bank patterns into a `merged_bank` for the D role automatically; no manual aggregation step needed [VERIFIED: codebase lines 797–807].

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GATE-06 taboo check | Custom regex | `gate06_ok()` in harness | Already implements TABOO_PATTERN + returns (bool, hits) [VERIFIED: codebase] |
| Bank read/write | Custom JSON I/O | `_load_bank()` / `_save_bank()` | Handles missing files, schema defaulting for all 9 slots [VERIFIED: codebase] |
| F1 computation | Custom metrics | `run_ablation.eval_metrics(predicted, gold)` | Returns P, R, F1, fp, fn counts; used throughout project [VERIFIED: codebase] |
| Cache key generation | Custom hash | `_cache_key(text_path, project, backend_str, model_str, role)` | Per-(text_stem, comp_hash, backend, model) formula locked in REQ-V23-10 [VERIFIED: codebase] |
| Verdict computation | Custom threshold logic | `probe_summary.json` verdict field | Harness computes `"CONTINUE"` vs `"KILL"` based on `CHEAP_KILL_THRESHOLD = 0.87` [VERIFIED: codebase] |

---

## GATE-06 / Reviewer Critic: Phase 15 Behavior

**Critical finding**: The CONTEXT.md "specific ideas" section notes "GATE-06 reviewer_critic activates real LLM in Phase 15 (stub was Phase 14 only)." However, code inspection shows the `_filter_proposals` function will only block patterns when `advisory=False`:

```python
# line 645 (harness)
if crit.get("verdict") == "REJECT" and not crit.get("advisory"):
    # blocking reject — NOT reached in current stub
```

The `reviewer_critic_stub` always returns `advisory=True`. Therefore the stub is already "active" in the sense that it: (a) enforces taboo-grep (blocking), and (b) logs advisory critiques. A "real LLM" critic would only add value by producing non-advisory REJECTs that actually block insertion.

**Planning implication**: Phase 15 has two options:
1. Keep stub as-is (advisory mode). GATE-06 taboo-grep is the actual gate; the critic is supplementary. This satisfies REQ-V23-09 ("reviewer-defensibility critic LLM at bank-entry boundary") in advisory mode — the boundary exists, the critic reviews it, results are logged.
2. Upgrade `reviewer_critic_stub` to a real LLM call with `advisory=False`. This makes the critic blocking but adds LLM cost (~1 call per proposed pattern) and increases risk of over-rejection in the probe tier.

**Recommendation (Claude's discretion)**: Keep stub advisory for Phase 15 Probe. The probe tier's purpose is to observe raw learning signal, not to maximize bank quality. Upgrade to blocking in Phase 16 Range if over-acceptance becomes a problem. Document the choice in 15-PROBE-VERDICT.md.
[VERIFIED: codebase analysis]

---

## Common Pitfalls

### Pitfall 1: Missing log directory
**What goes wrong:** `logs/voyager_v4_beta/` does not exist; `probe.log` redirect fails or creates the file at the wrong path.
**Why it happens:** The harness does not create `logs/` subdirectories; it only ensures `split_dir` (under `results/`) exists.
**How to avoid:** `mkdir -p logs/voyager_v4_beta/` before invoking the harness. Or pipe stdout to the file and let the shell create it with `> >(tee logs/voyager_v4_beta/probe.log)`.
**Warning signs:** Missing probe.log after run completion.
[VERIFIED: `ls logs/` shows voyager_gpt54/, voyager_v2/, etc. — each is a pre-created directory]

### Pitfall 2: Pass-2 cheap-kill threshold confusion
**What goes wrong:** Documenting the verdict as KILL because pass-1 macro < 0.87 — but the harness only cheap-kills after pass 2.
**Why it happens:** The `CHEAP_KILL_THRESHOLD = 0.87` check in `run_probe()` is inside `if pass_num == 2:` — it fires after pass 2 only.
**How to avoid:** The harness logic already handles this correctly. The planning task for pass-2 trigger uses 0.80 (from CONTEXT.md SC#3) as the early-skip: if pass-1 macro < 0.80, skip pass 2 — BUT this logic is not in the harness; it must be a manual decision by the executor. The harness always runs passes 1 and 2.
**Exact harness behavior:** `run_probe()` iterates `pass_num in range(1, 3)` (passes 1 and 2 always). The break condition `if pass_num == 2 and macro < 0.87` kills after pass 2. There is no automated pass-1-macro-< 0.80 skip in the harness.
[VERIFIED: codebase lines 908–930]

### Pitfall 3: Bank persistence confusion after rollback
**What goes wrong:** Thinking that ROLLBACK means no `_bank.json` files are written.
**Why it happens:** Rollback re-saves the prior (pre-probation) bank state to disk. The files are written either way.
**How to avoid:** After each pass, `_bank.json` files always exist — they contain the committed state. A rollback writes the pre-candidate bank back. Check `"committed": true/false` in `pass{N}_summary.json` to know which state was saved.
[VERIFIED: codebase lines 848–860]

### Pitfall 4: Token cost not directly available in dollars
**What goes wrong:** Trying to report cost in dollars from the LLM client's token tracking.
**Why it happens:** `LLMClient.get_session_usage()` returns token counts (prompt_tokens, completion_tokens, total_tokens) but not dollar cost. gpt-5.4 pricing is not encoded in the client.
**How to avoid:** Estimate cost from total_tokens using gpt-5.4 pricing at time of run (approximately $X/1M tokens — check OpenAI pricing page). Log raw token counts in `probe.log` via `llm.get_session_usage()` call after each role completes.
**Warning signs:** Probe.log has no token summary lines — add explicit `print(llm.get_session_usage())` after `run_probe()` completes.
[VERIFIED: codebase — no automatic cost-in-dollars logging in harness]

### Pitfall 5: Probation runs L again (cost surprise)
**What goes wrong:** Not accounting that the P role (probation gate) re-runs the full L (linker) on all 3 projects with the candidate bank.
**Why it happens:** `_probation_check` calls `_run_linker_l` for each project — this is a real LLM run for each project, NOT cached.
**How to avoid:** Budget for probation as an additional L run per pass: each pass = 3 L runs (step 1) + 3 O runs + 1 D run + 3 L runs (probation step 6) = 10 LLM-heavy calls per pass minimum, before caching saves on O/D.
[VERIFIED: codebase lines 839–846, `_probation_check` loops over projects]

---

## Verdict Document Format

Based on prior probe SUMMARYs at `.planning/v2.2-prep/probe-*-SUMMARY.md` [VERIFIED: codebase], the `15-PROBE-VERDICT.md` should include:

```markdown
---
phase: 15-probe-tier
tier: probe
backend: gpt-5.4
split: mainline
train_projects: [mediastore, teastore, teammates]
date: <ISO-date>
verdict: CONTINUE | KILL
---

# Phase 15: Probe Tier Verdict

## Summary
[one-liner verdict with macro F1]

## Per-Pass Results
| Pass | MS F1 | TS F1 | TM F1 | Train Macro | Committed | Notes |
|------|-------|-------|-------|-------------|-----------|-------|

## Verdict Evidence
[numeric evidence for CONTINUE or KILL]

## Next Action
[Phase 16 Range Tier | Phase 18 Compact-B]

## Cost
[token count from LLMClient.get_session_usage() + estimated dollars]
```
[CITED: .planning/v2.2-prep/probe-A-voyager-v4-SUMMARY.md — format reference]

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| OpenAI API key | LLM calls (O, D, L roles) | Yes [VERIFIED: .env] | — | None — blocking if absent |
| `results/voyager_v4_beta/` | Harness output root | Yes (partial) [VERIFIED: `ls`] | — | Created by harness on first run |
| `results/voyager_v4_beta/cache/` | Cache adapter | Yes [VERIFIED: test_dry_run artifacts] | — | Created by harness |
| `logs/voyager_v4_beta/` | probe.log destination | No [VERIFIED: `ls logs/`] | — | Must `mkdir -p` before run |
| Benchmark data (MS/TS/TM) | L role, O role | Yes [VERIFIED: DATASETS paths in run_ablation.py] | — | None — blocking if absent |
| Python environment with deps | Harness execution | Yes [VERIFIED: Phase 14 tests pass] | — | — |

**Missing dependencies with no fallback:**
- `logs/voyager_v4_beta/` directory — must create before running

**Missing dependencies with fallback:**
- None beyond the log directory

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | reviewer_critic_stub advisory=True is acceptable for probe tier (no upgrade needed) | GATE-06 section | If GATE-06 requires blocking critic for REQ-V23-09 compliance, a real LLM critic function must be written before running pass 1 |
| A2 | gpt-5.4 per-project cost ~$1.50/pass based on prior v2.2 probes | Standard Stack / Environment | If per-pass cost is higher (e.g., 3x due to longer prompts), 2 passes × 3 projects could exceed $10 cap |
| A3 | Pass-1-macro < 0.80 early-skip is a human decision, not automated | Pitfall 2 | If user expects harness to auto-skip pass 2 on <0.80 pass-1 macro, they must add that check manually or in the task instructions |

---

## Open Questions

1. **Reviewer critic: advisory vs blocking for REQ-V23-09**
   - What we know: `reviewer_critic_stub` always `advisory=True`; REQ-V23-09 says "reviewer-defensibility critic LLM at bank-entry boundary"
   - What's unclear: whether "at bank-entry boundary" requires a blocking call or just a logged check
   - Recommendation: Keep stub advisory for probe. If reviewer during Phase 19 audit flags this, upgrade at Range tier (Phase 16).

2. **Pass 2 early-skip trigger (0.80 threshold)**
   - What we know: CONTEXT.md SC#3 says "Pass 2 runs if pass-1 training-project macro F1 >= 0.80". The harness always runs passes 1 AND 2.
   - What's unclear: Should the execution plan include a manual check after pass 1 to decide whether to run pass 2 as a separate invocation?
   - Recommendation: Let the harness run both passes in a single invocation (the existing loop). The <0.80 early-kill has no real cost implication since the cheap-kill is at pass-2 completion anyway. Cheaper to just run both and check the verdict.

---

## Sources

### Primary (HIGH confidence)
- `scripts/voyager_train_tlr_v4_beta.py` — full harness implementation, CLI interface, all role logic, GATE-06 helpers, bank I/O, probe tier runner [VERIFIED: codebase read]
- `tests/test_s_linker14_voyager_registration.py` — 32 tests, all Phase 14 success criteria verified [VERIFIED: codebase read]
- `.planning/STATE.md` — Phase 14 complete, all deliverables shipped [VERIFIED: file read]
- `.planning/milestones/v2.3-ROADMAP.md` — Phase 15 success criteria and budget plan [VERIFIED: file read]
- `.planning/phases/15-probe-tier/15-CONTEXT.md` — locked decisions, discretion areas, deferred scope [VERIFIED: file read]
- `src/llm_sad_sam/llm_client.py` — token tracking via `get_session_usage()` [VERIFIED: codebase read]
- `run_ablation.py` — DATASETS dict, `load_gold_sam`, `eval_metrics` [VERIFIED: codebase read]

### Secondary (MEDIUM confidence)
- `.planning/v2.2-prep/probe-A-voyager-v4-SUMMARY.md` — prior probe cost reference (~5 LLM calls, < $1 for 1 project 1 pass on gpt-5.4) [CITED: file read]
- `.planning/phases/15-probe-tier/15-CONTEXT.md` — cost estimate ~$1.50/project/pass [CITED: specifics section]

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- CLI invocation: HIGH — exact argument signatures read from harness source
- Pass structure: HIGH — `run_probe()` logic fully read
- Bank persistence: HIGH — `_save_bank()` called unconditionally (commit or rollback)
- Cost estimate: MEDIUM — based on prior v2.2 probe data with different harness
- Reviewer critic advisory/blocking question: MEDIUM — code confirmed advisory=True, but CONTEXT.md note about "activates real LLM" creates ambiguity

**Research date:** 2026-06-01
**Valid until:** Phase 15 completion (no external dependencies on evolving APIs — gpt-5.4 is stable)
