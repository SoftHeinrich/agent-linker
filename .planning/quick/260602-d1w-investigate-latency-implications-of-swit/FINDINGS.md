# Flex Tier Investigation: Latency Implications for Voyager Training Loop

**Investigation date:** 2026-06-02
**Quick task:** 260602-d1w
**STATE.md item:** C-6 (Flex tier cost optimization)
**Verdict:** DO NOT SWITCH — defer indefinitely for the synchronous training loop.

---

## 1. What is Flex Tier?

OpenAI Flex Processing is an asynchronous batch API that offers up to 50% cost reduction
on supported models (including gpt-4.1, o3, o4-mini, and similar tiers as of mid-2025).

Key characteristics:
- Requests are queued and processed within a **24-hour window** with no per-request latency SLA.
- The API is invoked via a distinct endpoint (`POST /v1/batches`), not the standard
  `/v1/chat/completions` endpoint. Callers submit a JSONL batch file, receive a batch ID,
  then poll for status and retrieve results after completion.
- **NOT a drop-in replacement** for synchronous `LLMClient.query()` calls. It requires
  a fundamentally different client architecture: submit -> poll (minimum 10-minute interval
  per OpenAI recommendation) -> retrieve.
- Flex tier applies at the batch level, not the individual call level. If you submit a batch
  of 50 calls, all 50 complete together (turnaround: 1-24h), not individually.
- Available since mid-2025 as an extension of the existing Batch API (`POST /v1/batches`
  with `completion_window: "24h"` and priority/flex pricing).

---

## 2. Expected Latency: Flex vs Standard

Measured from `results/llm_logs/llm_requests_202603*.jsonl` (271 gpt-5.4 calls):

| Metric          | Standard tier (gpt-5.4, measured) | Flex tier (OpenAI docs, estimated) |
|-----------------|-----------------------------------|------------------------------------|
| Per-call min    | 0.7 s                             | N/A (batch, not per-call)          |
| Per-call median | 1.5 s                             | N/A                                |
| Per-call p75    | 1.9 s                             | N/A                                |
| Per-call max    | 11.5 s                            | N/A                                |
| Batch turnaround| N/A                               | 1-24 h per batch                   |
| Cost multiplier | 1.0x                              | ~0.5x                              |

For the Voyager training loop specifically, per-project L-role elapsed times from
`logs/voyager_v4_beta/range.log` (Pass 1, fresh cache, gpt-5.4):

| Project       | L-role elapsed (standard) |
|---------------|---------------------------|
| mediastore    | 30 s                      |
| teastore      | 32 s                      |
| teammates     | 113 s                     |
| bigbluebutton | 38 s                      |
| jabref        | 45 s                      |

Each L-role run makes approximately 9-11 sequential LLM calls internally
(s_linker14_voyager.py has 9 `llm.query()` call sites; ILinker4 adds 1-2 more for
Pass A / Pass B). The 30-113 s wall-clock times reflect all of these calls running
in the pipeline's mixed parallel/serial DAG (Tier 1 and Tier 2 use ThreadPool
parallelism; Tier 3 is serial).

---

## 3. Training Loop Compatibility Analysis

### Call topology and sequencing

The `voyager_train_tlr_v5.py` outer pass loop (`run_outer_pass`) has the following
sequential dependency chain:

```
For each outer pass (up to MAX_OUTER_PASSES = 5):
    Step 1: L(train projects) -- synchronous per-project loop
            _run_linker_l(project, ...) waits for linker.link() to return
            FP/FN sets are computed from returned results
            -> Must complete for all train projects before proceeding

    Step 2: OD(train projects) -- sequential per-project loop
            _run_od(llm, project, l_run, ...) makes 1 LLM call per project
            Consumes FP/FN from L result -> Cannot start until Step 1 done

    Step 3: Assessor -- sequential per-proposal loop
            _run_assessor(...) makes 1-2 LLM calls per proposal (1 revision max)
            Consumes OD proposals -> Cannot start until Step 2 done

    Step 4: Commit -- bank update
            Must complete before next outer pass (bank_content_hash changes)

    Step 5: L(test projects) -- sequential per-project eval
            Uses committed bank from Step 4
```

Confirmed: `grep "async|await|ThreadPool|asyncio" voyager_train_tlr_v5.py` returns
**zero hits**. The outer training loop is entirely synchronous. The `ThreadPoolExecutor`
usage is internal to `SLinker14Voyager._run_parallel()` (within a single L-role call),
not at the outer pass level.

### What Flex tier would require

To use Flex tier, `LLMClient.query()` cannot be used synchronously. The architecture
would need:

1. **New backend enum:** `LLMBackend.OPENAI_FLEX` in `llm_client.py`.
2. **Batch submission:** Pre-generate all prompts for each role, submit as JSONL to
   `POST /v1/batches`, receive a batch ID.
3. **Polling loop:** Poll `GET /v1/batches/{batch_id}` at minimum 10-minute intervals
   until status is `completed`.
4. **Result retrieval:** Download and parse the output JSONL file.
5. **Dependency re-architecture:** Because OD consumes FP/FN from L, and Assessor
   consumes OD proposals, roles cannot be batched together. Each outer pass requires
   at least 3 sequential batch submissions:
   - Batch 1: All L calls (submit -> wait 1-24h -> retrieve FP/FN)
   - Batch 2: All OD calls (submit -> wait 1-24h -> retrieve proposals)
   - Batch 3: All Assessor calls (submit -> wait 1-24h -> commit/reject)

### Wall-clock impact

| Tier         | Current (standard, estimated)        | With Flex (worst case)               |
|--------------|--------------------------------------|--------------------------------------|
| Probe        | ~15-40 min (2 passes x 5 projects)   | 6 batch windows x 1-24h = 6h-6 days |
| Range        | ~1-2 h (up to 5 passes x 5 projects) | 15 batch windows = 15h-15 days       |
| Confirmation | ~2-4 h (up to 5 passes x 5 projects) | 15 batch windows = 15h-15 days       |

Timing basis: 3 train projects x ~58 s avg L-role + 2 test projects x ~42 s avg = ~290 s
L per pass. OD: 3 x ~5 s = ~15 s. Assessor: ~6 calls x ~3 s = ~18 s. Total per pass: ~5-10 min.
Full probe: ~15-40 min (standard tier).

With Flex: each batch submission introduces a minimum ~1-hour wait (typical) to 24-hour
wait (worst case). Three sequential batches per outer pass x up to 5 passes = 15 batch
windows. At 1h minimum each: ~15 h minimum; at 24h each: ~15 days. Interactive development
becomes infeasible.

---

## 4. Cost Impact

Estimated costs at gpt-5.4 pricing (~$10/M input, ~$30/M output):

| Tier         | Standard (est.) | Flex (est. 0.5x) | Saving    |
|--------------|-----------------|------------------|-----------|
| Probe        | $1.25-$2.50     | $0.63-$1.25      | ~$1       |
| Range        | $5-$10          | $2.50-$5         | ~$2.50-$5 |
| Confirmation | $15-$30         | $7.50-$15        | ~$7.50-$15|
| v2.6 total   | $21-$43         | $10-$21          | ~$10-$22  |

The $80 budget cap (Phases 34-36) has substantial headroom at standard rates. The v2.5
milestone total was ~$62 across all phases including non-training work. Expected training-only
spend for v2.6 is $21-$43. The L-role cache (already implemented) eliminates the majority
of redundant LLM calls in passes 2-5, which is why estimates are well below the cap.

The absolute saving from Flex tier over the entire v2.6 training run is estimated at
**$10-$22**. This is the engineering budget available to implement the retrofit: clearly
insufficient given the scope of changes required (new backend, batch assembly, polling
infrastructure, result reassembly, error recovery for partial batches).

---

## 5. Recommendation

**DO NOT switch to Flex tier for v2.6 or any future synchronous training loop.**

Rationale:

1. **Incompatible execution model.** The training loop is a hard sequential dependency
   chain: L -> OD -> Assessor -> Commit -> next pass. Flex tier requires async batch
   submit/poll semantics. Retrofitting this requires redesigning the entire outer pass
   structure, not just swapping a backend parameter.

2. **Latency penalty eliminates usefulness.** Flex tier increases wall-clock time per
   outer pass from ~5-10 minutes to at minimum 3 hours (3 batch windows at 1h typical).
   A 5-pass range run takes 15+ hours instead of ~2 hours. The training loop's purpose
   is iterative convergence — multi-day turnaround defeats this entirely.

3. **Small absolute savings.** The saving over all of v2.6 training is ~$10-$22. The
   engineering cost of the retrofit exceeds this by a wide margin. The $80 budget cap
   has adequate headroom at standard rates.

4. **Better optimization already deployed.** The L-role cache (`_run_linker_l` with
   `_bank_content_hash` keying) already eliminates redundant LLM calls. This provides
   0-latency cache hits on unchanged inputs — far more effective than Flex tier.

**The only appropriate use case for Flex tier in this project is a future offline bulk
benchmark sweep** (e.g., evaluating all 5 datasets against N bank variants in one overnight
run, where each evaluation is independent and latency is irrelevant). This is not the
current training loop use case.

**C-6 disposition:** Close C-6 as "investigated — not viable for synchronous training
loop (latency multiplication by 100-1000x in exchange for ~50% cost reduction; absolute
saving ~$10-$22 over v2.6 budget, insufficient to justify retrofit). If a bulk offline
benchmark sweep use case is added in v2.8+, re-open as a new task scoped to that context.
Do not apply to the synchronous L->OD->Assessor->Commit training loop."
