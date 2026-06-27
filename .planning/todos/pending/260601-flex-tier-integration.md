---
id: 260601-flex-tier-integration
created: 2026-06-01
priority: medium
blocks: nothing (cost optimization only)
---

# OpenAI Flex Tier Integration

50% cost reduction (same as Batch API pricing). Requires infra work before enabling.

## What's already done

- `OPENAI_SERVICE_TIER` env var already wired in `llm_client.py` lines 410-411, 942-943.
  Setting `OPENAI_SERVICE_TIER=flex` in `.env` is all that's needed at the API level.

## What needs doing first

### 1. Timeout strategy
Flex tier latency: seconds to ~15 min depending on queue depth. Current per-call
timeouts (60-300s) will cause false timeouts + retries, burning retries on queued requests.

Options:
- Bump all LLM call timeouts to 900s (15 min) when Flex is active
- Or: read `OPENAI_SERVICE_TIER` in `_run_oracle_o` / `_run_distillator_d` and pass
  `timeout = 900 if flex else 300`
- Or: add `LLM_TIMEOUT_OVERRIDE` env var read by `llm_client.py`

### 2. 429 "Resource Unavailable" handling
Flex returns `429 Resource Unavailable` (not `429 Rate Limit`) when capacity exhausted.
Current retry logic in `llm_client.py` catches `"429"` string — verify it retries on
this variant, or add `"resource_unavailable"` / `"capacity"` to the retry keyword list.

### 3. Gate B timeout
Gate B currently `timeout=300` (bumped from 60 in Phase 20 audit fix). Flex queue
could stall Gate B for the full 15 min * 13 proposals = hours per pass. Consider
whether Gate B should use Flex or stay on default tier (it's cheap, ~$0.01/pass).

### 4. Per-role tier control
Best UX: `OPENAI_SERVICE_TIER=flex` enables Flex for O+D+L (expensive roles) but
Gate B stays `default` (cheap, latency-sensitive for inner loop).
Implement via `service_tier` override param on `llm_client.query()`.

## Estimated savings
Phase 21 probe (2 passes × 3 projects × O+D): ~$2-4 → ~$1-2 with Flex.
Full range run (5 passes × 5 projects): ~$15-20 → ~$7-10.
