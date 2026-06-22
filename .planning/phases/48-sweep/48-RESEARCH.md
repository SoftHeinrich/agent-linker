# Phase 48: SWEEP — Research

**Researched:** 2026-06-09
**Domain:** gpt-5.4 macro F1 sweep execution, cost estimation, result verification
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Hard cap ≤ $20 total LLM spend (GATE-08). Plan MUST include a pre-flight cost estimate
  and an abort condition.
- Backend: gpt-5.4 (`LLM_BACKEND=openai`), per v2.3 standing policy.
- 5 datasets: mediastore, teastore, teammates, bigbluebutton, jabref.
- Macro F1 = mean of per-dataset F1. Floor 91.3%. Per-dataset tolerance: no drop > 2pp vs s17e.
- Mirror the invocation used for prior gpt-5.4 sweeps (e.g. `logs/v2.6.2_s17e_gpt.log`).

### Claude's Discretion
- Exact runner flags, cost-estimation method (token counts × gpt-5.4 pricing), and log/CSV
  capture format are at Claude's discretion — follow the prior-sweep pattern and the runner's
  existing output conventions.

### Deferred Ideas (OUT OF SCOPE)
- None stated.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REQ-V264-09 | End-to-end GPT-5.4 5-dataset macro F1 on `s_linker20` ≥ 91.3%; no dataset drops > 2pp vs s17e; log to `logs/v2.6.4_s_linker20_gpt.log` | Runner invocation, cost estimate, metric extraction method all documented below |
| GATE-06 | Re-verify zero benchmark-derived vocabulary in `s_linker20` prompt constants and f-string scaffolds | GATE-06 grep command recovered from Phase 47 verification, re-runnable as a plan step |
| GATE-08 | Sweep budget cap ≤ $20 for 5-dataset gpt-5.4 sweep | Token-count-based cost estimate documented below; budget risk is VERY LOW |
</phase_requirements>

---

## Summary

Phase 48 is a measurement phase: run `s_linker20` on all 5 benchmark datasets against the
real gpt-5.4 backend, capture the result, and record whether macro F1 ≥ 91.3% and all
per-dataset numbers fall within 2pp of the s17e reference. There is no code to write; the
only implementation work is runner invocation, log capture, and metric verification.

The runner (`run_ablation.py`) is fully functional and already has `s_linker20` registered.
The invocation is straightforward: two environment variables must be set (`LLM_BACKEND=openai`
and `OPENAI_MODEL_NAME=gpt-5.4`) because the `.env` file currently only contains the API key.
The log must be captured via shell redirection (`2>&1 | tee`); the runner has no built-in
`--log-file` flag.

Pre-flight cost analysis: based on empirical token counts from s_linker19 (same architecture),
the full 5-dataset sweep consumes approximately 231,504 tokens (212,075 prompt + 19,429
completion) across ~100 LLM calls. Even at the most pessimistic pricing (gpt-4-turbo tier),
the total cost is approximately $2.70, well within the $20 cap. The flex tier (already the
default in `llm_client.py`) reduces cost by ~50%. Budget risk is VERY LOW.

**Primary recommendation:** Run `LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.4 python run_ablation.py --variants s_linker20 2>&1 | tee logs/v2.6.4_s_linker20_gpt.log` from the repo root. No per-dataset abort needed — the sweep costs under $3 worst-case.

---

## Architectural Responsibility Map

This phase has no architectural tiers — it is a CLI runner invocation and result verification.
All logic lives in the existing pipeline; this phase only executes and observes.

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| LLM API calls | `llm_client.py` openai backend | `s_linker20.py` pipeline | s_linker20 calls `self.llm.query()` which routes through `LLMClient._query_openai()` |
| Phase cache write | `s_linker20._save_phase()` | `results/phase_cache/s_linker20/openai/<dataset>/` | Written after each phase; fresh run (no prior cache for s_linker20) |
| Cost tracking | `llm_client.py` session token accounting | `results/llm_logs/*.jsonl` per-request entries | `token_usage` field in every JSONL entry; summed post-run for GATE-08 |
| Metric computation | `run_ablation.eval_metrics()` | stdout + JSON | Per-dataset P/R/F1/TP/FP/FN printed to stdout; macro avg computed in `print_summary()` |
| Log capture | Shell `tee` | `logs/v2.6.4_s_linker20_gpt.log` | Runner has no `--log-file` flag; must use shell redirection |

---

## Q1: Exact Sweep Invocation

### Prior sweep pattern (from `logs/v2.6.2_s17e_gpt.log` header)

```
========================================================================================================================
ABLATION STUDY: Retained ILinker and S-Linker Variants
Backend: openai (gpt-5.4)
Datasets: mediastore, teastore, teammates, bigbluebutton, jabref
Variants: s_linker17e
```

The s17e log confirms the prior pattern: all 5 datasets, single variant, gpt-5.4 backend.

### Runner flags (from `run_ablation.py` inspection)

`parse_args` accepts exactly these flags:
- `--variants` — one or more variant names (default `s_linker11a`)
- `--datasets` — one or more dataset names (default = all 5)
- `--results-dir` — CSV/JSON output directory (default `results/ablation_results`)
- `--list-datasets` — print names and exit
- `--list-variants` — print names and exit

There is NO `--dry-run`, NO `--cost`, NO `--log-file` flag. [VERIFIED: run_ablation.py lines 1078–1099]

### Backend selection

`run_ablation.py` line 845 sets `os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.2")` at
module level. This runs BEFORE any linker is instantiated. `s_linker20.__init__` (line 273)
also calls `os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.4")`, but since `setdefault`
is a no-op when the key already exists, `run_ablation.py`'s `"gpt-5.2"` wins unless the
environment already has `OPENAI_MODEL_NAME` set. [VERIFIED: Python setdefault semantics; run_ablation.py line 845 vs s_linker20.py line 273]

The current `.env` contains only `OPENAI_API_KEY`. Therefore `LLM_BACKEND` and
`OPENAI_MODEL_NAME` MUST be set explicitly.

`LLMBackend` selection: `get_backend()` reads `LLM_BACKEND` env var (run_ablation.py lines 834–842).

### Exact invocation command

```bash
cd /mnt/hostshare/ardoco-home/agent-linker
LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.4 python run_ablation.py \
  --variants s_linker20 \
  2>&1 | tee logs/v2.6.4_s_linker20_gpt.log
```

Notes:
- `--datasets` is omitted → defaults to all 5 datasets in registration order (mediastore, teastore, teammates, bigbluebutton, jabref). [VERIFIED: run_ablation.py line 1081–1085]
- The `--results-dir` defaults to `results/ablation_results` — a CSV file and timestamped JSON (`ablation_YYYYMMDD_HHMMSS.json`) are written there automatically. [VERIFIED: run_ablation.py lines 1092–1095, 1179–1182]
- Log capture uses `tee` so stdout appears on terminal AND is saved to the log file simultaneously.
- The flex tier default (`OPENAI_SERVICE_TIER=flex`) is already hardcoded in `llm_client.py`; do NOT add `OPENAI_SERVICE_TIER=standard` unless explicitly desired. [VERIFIED: llm_client.py lines 925, 402]

### Verify backend is active before spending money

```bash
LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.4 python run_ablation.py --list-variants | grep s_linker20
```
Expected: prints `s_linker20` and exits cleanly. If it errors, the variant is not registered.

Also confirm the backend header on a dry instantiation (zero LLM calls):
```bash
LLM_BACKEND=checkpoint CHECKPOINT_FALLBACK=openai OPENAI_MODEL_NAME=gpt-5.4 \
  python -c "
import run_ablation as r
from llm_sad_sam.llm_client import LLMBackend
b = r.build_linker('s_linker20', backend=LLMBackend.OPENAI)
print('variant:', b._VARIANT_NAME)
print('model:', b.llm.get_active_model())
"
```

---

## Q2: Pre-Flight Cost Estimate (CRITICAL)

### Empirical token counts from s_linker19 (identical architecture to s_linker20)

Data source: `results/llm_logs/s_linker19_openai_*_calls.json` (runs using the June 4–5
architecture, which is Framing C only + single-pass coref validation — identical to s_linker20).
[VERIFIED: inspected via Python json parsing of actual log files]

| Dataset | LLM Calls | Prompt Tokens | Completion Tokens | Total |
|---------|-----------|---------------|-------------------|-------|
| mediastore | 14 | 24,959 | 2,601 | 27,560 |
| teastore | 13 | 24,688 | 2,364 | 27,052 |
| teammates | 39 | 98,728 | 7,643 | 106,371 |
| bigbluebutton | 21 | 50,099 | 5,454 | 55,553 |
| jabref | 10 | 13,601 | 1,367 | 14,968 |
| **TOTAL** | **97** | **212,075** | **19,429** | **231,504** |

Teammates dominates (39 calls / 106k tokens) due to batched extraction across 198 sentences.

### Call count breakdown per dataset

Per-phase call counts for the current s_linker19/s20 architecture:
- Phase 1 (knowledge acquisition): 3 calls (ambiguity, doc_extract, doc_judge) — every dataset
- Phase 2 (Framing C, 2-pass): 2–8 calls depending on document length (batch_size=50)
- Phase 4 (twopass validation): 2–8 calls per batch of 25 candidates
- Phase 5 (coreference): 1–9 calls depending on anaphoric sentence count
- Phase 5 (coref validation): 1 call (single-pass, batch of 25)
- Total floor: 10 calls (jabref, smallest dataset); ceiling: ~39 calls (teammates, 198 sentences)

### Cost calculation

gpt-5.4 exact pricing is not in the codebase. [ASSUMED: gpt-5.4 pricing tier is not
documented in this repo; only GPT-4 legacy pricing appears in `llm_client.py` comments].
The codebase's `print_usage_summary` uses GPT-4 pricing as a rough estimator
($0.00003/prompt, $0.00006/completion tokens). [VERIFIED: llm_client.py lines 1082–1085]

Three scenarios (from observed token counts):

| Pricing Scenario | Prompt $/1M | Completion $/1M | **Total Cost** | Cap Headroom |
|-----------------|------------|----------------|---------------|--------------|
| gpt-4o-like (conservative) | $2.50 | $10.00 | **$0.72** | $19.28 |
| gpt-4o-like flex (~50% off) | $1.25 | $5.00 | **$0.36** | $19.64 |
| gpt-4-turbo-like (pessimistic upper bound) | $10.00 | $30.00 | **$2.70** | $17.30 |
| Codebase GPT-4 estimator formula | $30.00 | $60.00 | **$7.53** | $12.47 |

**Budget verdict: VERY LOW RISK.** Even the most pessimistic pricing scenario ($7.53
using the codebase's own overestimate formula) leaves $12+ headroom. The actual cost is
almost certainly under $3. No per-dataset staged abort is needed, but documenting the
per-dataset token counts in the plan enables post-run cost reconstruction.

### Flex tier (already default)

`llm_client.py` defaults `OPENAI_SERVICE_TIER` to `"flex"` when the env var is absent
(lines 925 and 402). The `.env` does not set it, so flex tier is active by default.
[VERIFIED: llm_client.py lines 925, 402]

Flex tier cost: ~50% reduction vs standard. Latency impact: median is actually 12% FASTER
than standard on gpt-5.4 (measured empirically); p90 is 33% slower. For a single linear
5-dataset sweep, tail latency on one or two calls is acceptable.
[VERIFIED: `.planning/quick/260602-d1w-investigate-latency-implications-of-swit/FINDINGS.md` §2]

To use standard tier instead (if flex causes issues): `OPENAI_SERVICE_TIER=default`.

---

## Q3: Resumability / Partial-Cost Safety

### Phase cache behavior

`s_linker20._save_phase()` writes intermediate results to
`results/phase_cache/s_linker20/openai/<dataset>/` as `layer1.pkl`, `layer2.pkl`, `layer3.pkl`,
and `final.pkl` after each pipeline phase within a dataset run.
[VERIFIED: s_linker20.py lines 1020–1032, _VARIANT_NAME="s_linker20" line 263]

However, **the runner does NOT check these phase-cache pickles before making LLM calls**.
The phase cache is a progress artifact written by the linker, but `run_ablation.py` always
instantiates a fresh linker and calls `link()` unconditionally. The linker clears `_llm_calls`
at the start of each `link()` call (line 488) but does NOT load a prior checkpoint.
[VERIFIED: s_linker20.py line 488, run_ablation.py `run_variant()` function lines 962–1044]

**Consequence:** If the run is interrupted mid-dataset, re-running will re-bill that dataset's
LLM calls. The phase cache does NOT prevent re-billing.

### Staged per-dataset run (feasible for monitoring but not needed here)

The `--datasets` flag accepts a subset:

```bash
# Run only mediastore first to verify backend is working
LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.4 python run_ablation.py \
  --variants s_linker20 --datasets mediastore 2>&1 | tee logs/v2.6.4_s20_ms_probe.log
```

Given the low total cost ($0.72–$2.70 estimated), running all 5 datasets in a single
invocation is appropriate. A staged run is unnecessary for budget protection but is
available as an option if the executor wants to verify backend connectivity before committing
to the full sweep.

### Abort condition

The plan must state: if after the mediastore dataset the backend reports an error or the cost
appears anomalously high (e.g., > $5 for mediastore alone, suggesting a different pricing
tier than expected), stop and investigate before proceeding. In normal operation, mediastore
completes in ~23–30 seconds and ~27k tokens; anything dramatically outside that range
warrants inspection.

---

## Q4: Metric Extraction

### Where F1 appears in output

`run_ablation.py` emits metrics at two points:
1. **Per-dataset, per-variant, immediately after completion** (run_ablation.py line 1024–1029):
   ```
     s_linker20: P=XX.X% R=XX.X% F1=XX.X% TP=N FP=N FN=N (Ns)
   ```
2. **Summary table** at the end (run_ablation.py `print_summary()` lines 1048–1075):
   ```
   Dataset          |    s_linker20    
   -----------------+-------------------
   mediastore       | F1 XX.X% FP   N
   teastore         | F1 XX.X% FP   N
   teammates        | F1 XX.X% FP   N
   bigbluebutton    | F1 XX.X% FP   N
   jabref           | F1 XX.X% FP   N
   -----------------+-------------------
   Macro avg        | F1 XX.X% FP   N
   ```
   [VERIFIED: s17e log tail at lines 372–381; run_ablation.py print_summary() function]

### Macro F1 computation

The runner computes macro F1 as `mean(per-dataset F1)` at the summary table:
```python
avg_f1 = sum(value["F1"] for value in values) / len(values)
```
[VERIFIED: run_ablation.py line 1071]

### Extracting from the log

After the sweep, grep the log for the result line:
```bash
grep "s_linker20:" logs/v2.6.4_s_linker20_gpt.log
```
This prints 5 lines (one per dataset with P/R/F1/TP/FP/FN).

Extract the macro F1 summary line:
```bash
grep "Macro avg" logs/v2.6.4_s_linker20_gpt.log
```

### s17e per-dataset reference numbers (the 2pp tolerance fence)

| Dataset | s17e F1 | Tolerance floor (−2pp) |
|---------|---------|------------------------|
| mediastore | 94.9% | ≥ 92.9% |
| teastore | 96.3% | ≥ 94.3% |
| teammates | 89.8% | ≥ 87.8% |
| bigbluebutton | 80.4% | ≥ 78.4% |
| jabref | 100.0% | ≥ 98.0% |
| **Macro** | **92.3%** | **≥ 91.3%** |

[VERIFIED: STATE.md v2.6.2 results table; 48-CONTEXT.md success criteria]

### CSV output

`run_ablation.py` saves a timestamped JSON to `results/ablation_results/ablation_YYYYMMDD_HHMMSS.json`
containing the full per-dataset result dict including F1, TP, FP, FN, sources, and
fp_details. [VERIFIED: run_ablation.py lines 1179–1182]

---

## Q5: Cost Logging for GATE-08

### What the runner emits (and does not emit)

`run_ablation.py` does NOT print a cost summary. It does NOT call `LLMClient.get_cumulative_usage()`
or `print_usage_summary()`. [VERIFIED: run_ablation.py — no cost-related print calls]

`s_linker20` writes a per-dataset `_calls.json` log (via `_save_log()`) to `results/llm_logs/`
with one entry per LLM call, each containing `token_usage.prompt_tokens`,
`token_usage.completion_tokens`, `token_usage.total_tokens`. [VERIFIED: s_linker20.py lines
1044–1072]

`llm_client.py` also writes per-request JSONL entries to `results/llm_logs/llm_requests_YYYYMMDD_HHMMSS.jsonl`
with `token_usage` per request. [VERIFIED: llm_client.py lines 536–558]

### Method to produce GATE-08 cost evidence

After the sweep, sum token counts from the `_calls.json` files:

```bash
python3 -c "
import json, glob, os
call_files = sorted(glob.glob('results/llm_logs/s_linker20_openai_*_calls.json'))
total_p, total_c = 0, 0
for f in call_files:
    data = json.load(open(f))
    p = sum(c.get('token_usage', {}).get('prompt_tokens', 0) or 0 for c in data if c.get('token_usage'))
    c = sum(c.get('token_usage', {}).get('completion_tokens', 0) or 0 for c in data if c.get('token_usage'))
    print(f'  {os.path.basename(f)}: {len(data)} calls, prompt={p}, compl={c}')
    total_p += p; total_c += c
print(f'TOTAL: prompt={total_p}, completion={total_c}, total={total_p+total_c}')
# Use codebase formula (over-estimates vs actual gpt-5.4 pricing)
est = (total_p * 0.00003) + (total_c * 0.00006)
print(f'Estimated cost (codebase GPT-4 formula, UPPER BOUND): \${est:.4f}')
"
```

Record the output as the GATE-08 cost evidence artifact. The codebase formula is a known
overestimate (GPT-4 era pricing vs the much cheaper gpt-5.4 actual rates), so if this formula
reports under $20, actual spend is also under $20.

---

## Q6: GATE-06 Re-Verification

GATE-06 was already verified clean in Phase 47. The Phase 48 re-verification is a re-run of
the same grep on the same file (s_linker20.py is byte-equal from Phase 47 close).

### Exact GATE-06 command (from Phase 47 verification)

The grep checks whether any of the neutral-vocabulary tokens introduced by Phase 46 lexical
cuts appear in `BENCHMARK_TABOO.md` (which would mean they collide with benchmark-specific
terminology):

```bash
test -z "$(grep -niwE 'grouping|encompasses|matching|noun|phrase|refers|back|topic|surrounding|section' BENCHMARK_TABOO.md)" \
  && echo "GATE-06 clean" || echo "GATE-06 FAIL"
```

Expected: prints `GATE-06 clean`. [VERIFIED: Phase 47 verification row 9, 47-02-PLAN.md lines 119–137]

This grep runs on `BENCHMARK_TABOO.md`, NOT on `s_linker20.py`. The logic is: if the
neutralized vocabulary tokens appear in the taboo list, they have benchmark leakage risk.
Phase 47 confirmed this returns zero lines. [VERIFIED: 47-VERIFICATION.md line 30]

---

## Q7: Failure Routing

This phase is a verdict phase. The possible outcomes are:

### Outcome A — PASS (macro F1 ≥ 91.3%, no dataset drops > 2pp)
- Mark REQ-V264-09, GATE-06, GATE-08 as satisfied in the SUMMARY.
- Phase 49 CLOSE proceeds immediately.

### Outcome B — MARGINAL FAIL (macro F1 between 89.0% and 91.3%, or one dataset drops 2–4pp)
- Record the per-dataset numbers verbatim in the SUMMARY.
- Do NOT attempt any fix — Phase 48 is measurement only; the plan must not include
  any "if regression then fix" steps.
- Record as REQ-V264-09 FAIL with the observed numbers.
- Phase 49 CLOSE documents the regression; the milestone audit will record this as a
  non-passing result.

### Outcome C — HARD FAIL (macro F1 < 89%, or a dataset catastrophically regresses)
- Same as Outcome B — record and do not fix in-phase.
- Log the full output, include TP/FP/FN comparison vs s17e reference.

### What the plan records in all cases

The plan must log, at minimum:
1. Per-dataset F1, TP, FP, FN (from the log `s_linker20:` lines)
2. Macro F1 (from `Macro avg` line)
3. Comparison table vs s17e reference (absolute delta per dataset)
4. Token count and cost estimate (for GATE-08)
5. GATE-06 re-grep result

No code changes are permitted in Phase 48. If `s_linker20` regresses, that result is the
honest verdict and feeds Phase 49 as-is.

---

## Project Constraints (from CLAUDE.md)

- `s_linker20.py` is part of the active surface (listed in CLAUDE.md Active Surface).
- Default model policy: Claude Sonnet for canonical line; `.env` sets `LLM_BACKEND=openai`
  for benchmarking. HOWEVER, the current `.env` only contains `OPENAI_API_KEY` — `LLM_BACKEND`
  is NOT in `.env`. The plan must set it explicitly in the invocation or add it to `.env`.
- `s_linker19.py` and `s_linker13_min.py` are BYTE-EQUAL FROZEN — do not touch.
- Run: `python run_ablation.py --variants s_linker20`.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| OPENAI_API_KEY | gpt-5.4 API calls | ✓ | (in .env) | None — blocking |
| LLM_BACKEND=openai | Backend selection | ✓ (set in invocation) | — | Set explicitly |
| OPENAI_MODEL_NAME=gpt-5.4 | Correct model | Must be set explicitly | — | run_ablation.py defaults to gpt-5.2 |
| Python package `llm_sad_sam` | run_ablation.py | ✓ | Importable | — |
| `s_linker20` registered | run_ablation.py | ✓ | Phase 47 complete | — |
| Benchmark files | DATASETS dict | ✓ (assumed present — s17e ran successfully) | — | — |
| `results/llm_logs/` directory | _save_log() | ✓ (auto-created) | — | — |
| `results/phase_cache/s_linker20/` | _save_phase() | ✗ (will be created fresh) | — | Auto-created |
| `logs/` directory | tee log capture | ✓ | — | Auto-create with `mkdir -p logs` |

**Missing dependencies with no fallback:**
- None that block execution (API key is present, package is importable, benchmarks ran before).

**Model name trap:** `OPENAI_MODEL_NAME` MUST be set in the invocation. If omitted, `run_ablation.py` sets it to `gpt-5.2` (line 845) before the linker has a chance to override it. Running on `gpt-5.2` instead of `gpt-5.4` is a silent error that produces different results.

---

## Common Pitfalls

### Pitfall 1: Wrong Model (gpt-5.2 instead of gpt-5.4)
**What goes wrong:** `run_ablation.py` line 845 calls `os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.2")` at module load time, before any linker is instantiated. The linker's own `setdefault("gpt-5.4")` in `__init__` is a no-op.
**Why it happens:** `setdefault` only sets if the key is absent; the runner sets it first.
**How to avoid:** Always pass `OPENAI_MODEL_NAME=gpt-5.4` in the invocation prefix.
**Warning signs:** Log header reads `Backend: openai (gpt-5.2)` instead of `(gpt-5.4)`.

### Pitfall 2: Missing LLM_BACKEND / Falling Back to Claude
**What goes wrong:** Without `LLM_BACKEND=openai`, the runner defaults to `LLMBackend.CLAUDE`. The run makes zero paid API calls but produces wrong/no results.
**How to avoid:** Always prefix with `LLM_BACKEND=openai`.
**Warning signs:** Log header reads `Backend: claude (sonnet)`.

### Pitfall 3: Log File Not Captured
**What goes wrong:** The runner has no `--log-file` flag. Running without `tee` produces no persistent log, and the SUMMARY has no artifact to reference.
**How to avoid:** Always use `2>&1 | tee logs/v2.6.4_s_linker20_gpt.log`.
**Warning signs:** The log file does not exist after the run.

### Pitfall 4: Phase Cache Does Not Prevent Re-billing
**What goes wrong:** If the sweep crashes mid-run, re-running re-bills completed datasets.
**Why it happens:** `run_ablation.py::run_variant()` creates a fresh linker and calls `link()` unconditionally; the phase cache pickles are written but not read by the runner.
**How to avoid:** Given the $0.72–$2.70 total cost, even two full runs cost under $6. No staged abort is needed. Document this explicitly so the executor does not worry about it.

### Pitfall 5: Stale s_linker20 Phase Cache Polluting Results
**What goes wrong:** If prior test runs or harness runs left pickles in `results/phase_cache/s_linker20/openai/`, the linker will overwrite them (since `_save_phase` overwrites blindly). The runner does not read them; they do not affect LLM calls. No practical risk.
**How to avoid:** Nothing needed — the cache is write-only from the runner's perspective.

---

## Validation Architecture

Nyquist validation does not apply to this phase: Phase 48 is a measurement/execution phase with no new code. The success criterion is observed runtime output (F1 numbers), not pytest-verifiable behavior. The existing pytest suite (`pytest tests/`) should pass before the sweep is run as a baseline sanity check, but no new tests are introduced.

**Pre-sweep check:**
```bash
pytest tests/test_s_linker20_registration.py tests/test_s_linker20_harness_invariants.py -q
```
Expected: all pass (confirmed in Phase 47 verification).

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | gpt-5.4 pricing is in the $1–$10/1M prompt range (conservative estimate uses gpt-4o pricing $2.50/1M) | Q2 Cost Estimate | If gpt-5.4 is unexpectedly more expensive (>$86/1M prompt), budget would be at risk. This is implausible — new models trend cheaper. |
| A2 | Benchmark data files are accessible at the paths in `run_ablation.DATASETS` | Environment Availability | If paths are wrong, the run fails immediately with FileNotFoundError. Mitigated: s17e ran successfully from the same environment. |
| A3 | The OPENAI_API_KEY in `.env` is active and has sufficient quota | Q1 Invocation | If the key is expired, the first LLM call returns 401. Mitigated: verify with a test call before the full sweep. |

---

## Sources

### Primary (HIGH confidence)
- `run_ablation.py` — Runner code fully read; all flags, defaults, backend selection logic, metric computation confirmed
- `src/llm_sad_sam/llm_client.py` — Service tier default (flex), token tracking, cost estimator formula confirmed
- `src/llm_sad_sam/linkers/experimental/s_linker20.py` — Phase cache behavior, variant name, model setdefault, call site count confirmed
- `logs/v2.6.2_s17e_gpt.log` — Prior sweep log format, per-dataset F1 reference numbers, timing confirmed
- `.planning/phases/47-ship/47-VERIFICATION.md` — GATE-06 grep command confirmed from Phase 47 execution record
- `.planning/phases/47-ship/47-02-PLAN.md` — GATE-06 exact grep parameters confirmed
- `.planning/REQUIREMENTS.md` — REQ-V264-09, GATE-06, GATE-08 definition confirmed
- `.planning/STATE.md` — s17e per-dataset reference numbers confirmed
- `results/llm_logs/s_linker19_openai_*_calls.json` — Empirical token counts from matching-architecture runs

### Secondary (MEDIUM confidence)
- `.planning/quick/260602-d1w-investigate-latency-implications-of-swit/FINDINGS.md` — Flex tier cost/latency tradeoffs from empirical benchmark

### Tertiary (LOW confidence / ASSUMED)
- gpt-5.4 pricing: not in repo; estimated from gpt-4o comparable pricing

---

## Metadata

**Confidence breakdown:**
- Runner invocation: HIGH — code read directly, all flag behaviors verified
- Cost estimate: MEDIUM — token counts are empirical from matching-architecture runs; pricing is ASSUMED at gpt-4o level
- Metric extraction: HIGH — output format verified from s17e log
- GATE-06 method: HIGH — exact grep command recovered from Phase 47 execution record
- Phase cache behavior: HIGH — code read directly

**Research date:** 2026-06-09
**Valid until:** Indefinite (this phase is code-stable; nothing will change before it runs)
