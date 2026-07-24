# S24 discourse-scope replacement pilot

Date: 2026-07-24

Configuration:

- backend: `openai`
- model: `gpt-5.6-terra`
- reasoning effort: `none`
- credential mapping: process-local `OPENAI_API_KEY="$OAI_KEY"`; no value
  persisted
- dataset: `bigbluebutton`
- comparison floor:
  `results/s24_lexical_entity_e2e_v1_20260724`

Deterministic verification:

```text
$ ../.venv/bin/python pilot/test_s24_discourse_scope.py
PASS: S24 discourse-scope contracts
```

Pilot command pattern:

```bash
OPENAI_API_KEY="$OAI_KEY" \
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_REASONING_EFFORT=none \
LLM_LOG_DIR=../results/s24_discourse_scope_pilot_vN_20260724/llm_logs \
  ../.venv/bin/python pilot/s24_discourse_scope_pilot.py \
  --datasets bigbluebutton \
  --baseline-dir ../results/s24_lexical_entity_e2e_v1_20260724 \
  --results-dir ../results/s24_discourse_scope_pilot_vN_20260724
```

Text results:

| Run | Role TP / FP | Final TP / FP / FN | F1 | F2 | Exit |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 | 4 / 2 | 53 / 6 / 9 | 0.8760 | 0.8632 | 1 |
| v2 | 6 / 2 | 55 / 6 / 7 | 0.8943 | 0.8900 | 1 |
| v3 | 6 / 0 | 55 / 4 / 7 | 0.9091 | 0.8958 | 1 |

All exits are expected gate failures. The best result missed the predeclared
requirement of at least seven role true positives by one, so the experimental
path was not promoted and a five-project E2E was not authorized by the staged
protocol. Full structured decisions and model traces are preserved in each
run's `pilot_results.json`; replacement links are preserved in the adjacent
CSV.
