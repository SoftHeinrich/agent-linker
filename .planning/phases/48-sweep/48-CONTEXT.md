# Phase 48: SWEEP - Context

**Gathered:** 2026-06-09
**Status:** Ready for planning
**Mode:** Auto-generated (measurement/benchmark phase — discuss skipped; methodology fixed by ROADMAP gates)

<domain>
## Phase Boundary

Validate `s_linker20` (shipped in Phase 47) at **gpt-5.4 macro F1 ≥ 91.3%** across all 5
datasets within the **≤ $20 budget cap** (GATE-08), confirming the Pareto-minimized prompts do
not regress the 17e-line breakthrough floor.

Delivers (success criteria from ROADMAP):
1. `logs/v2.6.4_s_linker20_gpt.log` exists and records a completed 5-dataset gpt-5.4 sweep on
   `s_linker20`.
2. Macro F1 ≥ 91.3% (= s17e 92.3% − T 1.0pp).
3. No individual dataset drops more than 2pp vs s17e per-dataset numbers
   (MediaStore 94.9%, TeaStore 96.3%, TeaMmates 89.8%, BigBlueButton 80.4%, JabRef 100.0%).
4. GATE-06 re-verified on `s_linker20`: zero benchmark-derived vocabulary in any inlined constant
   or f-string scaffold.
5. Total API cost for this sweep ≤ $20 (GATE-08); cost logged or estimated from token counts.
</domain>

<decisions>
## Implementation Decisions

### Budget guardrail (NON-NEGOTIABLE)
- Hard cap ≤ $20 total LLM spend (GATE-08). The plan MUST estimate per-dataset / total cost
  BEFORE the full run and include an abort/stop condition if projected cost would exceed $20.
- Backend: gpt-5.4 (`LLM_BACKEND=openai`), per v2.3 standing policy and `.env`.
- User has explicitly approved spending up to $20 for this sweep (2026-06-09).

### Methodology (fixed — not a grey area)
- 5 datasets: MediaStore, TeaStore, TeaMmates, BigBlueButton, JabRef.
- Macro F1 = mean of per-dataset F1. Floor 91.3%. Per-dataset tolerance: no drop > 2pp vs the
  s17e reference numbers above.
- Mirror the invocation used for prior gpt-5.4 sweeps (e.g. the s17e run: `logs/v2.6.2_s17e_gpt.log`).
  `run_ablation.py --variants s_linker20` with the openai backend.

### Claude's Discretion
- Exact runner flags, cost-estimation method (token counts × gpt-5.4 pricing), and log/CSV
  capture format are at Claude's discretion — follow the prior-sweep pattern and the runner's
  existing output conventions.
</decisions>

<code_context>
## Existing Code Insights

### Runner + variant
- `s_linker20` registered in `run_ablation.py` (Phase 47): VARIANT_SPECS entry,
  module=`llm_sad_sam.linkers.experimental.s_linker20`, class=`SLinker20`, experimental=True.
- Run: `python run_ablation.py --variants s_linker20` (openai backend via `.env` / LLM_BACKEND=openai).
- Phase-cache pickles live under `results/phase_cache/openai/<project>/`. A real sweep makes
  fresh LLM calls (no cache for s_linker20 yet) — this is the budgeted work.

### Prior sweep references (mirror these)
- `logs/v2.6.2_s17e_gpt.log` — the s17e 5-dataset gpt-5.4 sweep (92.3% macro, the reference floor).
- Other gpt-5.4 sweep logs in `logs/` (17f flex, c5pass, dotted_rename) show the invocation +
  flex-tier latency option investigated in quick task 260602-d1w.

### Gates
- GATE-06: re-grep inlined after-text against `BENCHMARK_TABOO.md` (already clean per Phase 47).
- GATE-08: ≤ $20 cost cap — NEW active gate for this phase.
- GATE-01 unaffected (s_linker19/s_linker13_min untouched by a sweep).
</code_context>

<specifics>
## Specific Ideas

Capture the per-dataset F1 table + macro + FP count in the log so Phase 49 can record the verdict
directly. Compare against the s17e per-dataset reference inline.
</specifics>

<deferred>
## Deferred Ideas

None — Phase 49 (MILESTONE CLOSE) consumes this sweep's result.
</deferred>
