# Fail-closed LLM failure verification

Date: 2026-07-24

## Regression reproduced

The BBB portion of
`s24_anchored_validator_gpt54_openai_flex_noreasoning_20260723` logged 11
exhausted Flex `429 flex_unavailable` responses. They were converted into empty
JSON results by the old `_ask()` path, so required extraction, validation, and
coreference batches were silently omitted. That run's BBB full score is invalid.

## Guard verification

An isolated client test monkeypatched OpenAI to return an exhausted
`429 flex_unavailable resource_unavailable` response with `max_retries=1`.

Expected and observed result:

```text
PASS fail-closed Flex exhaustion: raises and records FATAL phase trace
```

The guard treats Flex capacity errors as retryable. If all retries are exhausted,
the tracing wrapper records a `success: false`, `error: FATAL: ...` call record
and re-raises, aborting the affected project rather than emitting partial CSV
predictions.
