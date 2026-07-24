# Compact S24 production checkpoint v1 — failed

Configuration: OpenAI `gpt-5.6-terra`, reasoning effort `none`;
BigBlueButton and TeamMates; fresh phase cache.

Command:

```bash
OPENAI_API_KEY="$OAI_KEY" LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra OPENAI_REASONING_EFFORT=none \
PHASE_CACHE_DIR=../results/s24_simple_production_checkpoint_v1_20260725/phase_states \
LLM_LOG_DIR=../results/s24_simple_production_checkpoint_v1_20260725/llm_logs \
../.venv/bin/python run_ablation.py \
  --variants s_linker24_role_orchestrator \
  --datasets bigbluebutton teammates \
  --results-dir ../results/s24_simple_production_checkpoint_v1_20260725
```

Result: BigBlueButton 46 TP / 3 FP / 16 FN; TeamMates 54 TP / 4 FP /
3 FN. The participant judge returned exact claims wrapped in quotation marks;
the strict substring validator rejected every BigBlueButton keep judgment.
This run failed the participant-recovery gate and was not promoted.
