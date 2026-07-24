# S24 lexical-entity focused pilot

Fresh OpenAI `gpt-5.6-terra`, reasoning effort `none`, intact BigBlueButton
document. Gold was loaded only after inference.

## Command

```bash
OPENAI_API_KEY="$OAI_KEY" \
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_REASONING_EFFORT=none \
PHASE_CACHE_DIR=../results/s24_lexical_entity_pilot_openai_v1_20260724/phase_states \
LLM_LOG_DIR=../results/s24_lexical_entity_pilot_openai_v1_20260724/llm_logs \
  ../.venv/bin/python pilot/s24_lexical_entity_pilot.py \
  --backend openai \
  --datasets bigbluebutton \
  --results-dir ../results/s24_lexical_entity_pilot_openai_v1_20260724
```

## Result

```text
lexical_tp: 2
lexical_fp: 0
pass_gate: true
```

Both accepted additions map `bbb-web` to `BBB web` at sentences 30 and 78.
The workflow was entity → coreference → relation-role → finalize and used 26
LLM calls.
