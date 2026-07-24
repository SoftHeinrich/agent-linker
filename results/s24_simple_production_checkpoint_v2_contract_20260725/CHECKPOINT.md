# Compact S24 production checkpoint v2 — passed

Configuration: OpenAI `gpt-5.6-terra`, reasoning effort `none`;
BigBlueButton and TeamMates; fresh phase cache.

Command:

```bash
OPENAI_API_KEY="$OAI_KEY" LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra OPENAI_REASONING_EFFORT=none \
PHASE_CACHE_DIR=../results/s24_simple_production_checkpoint_v2_contract_20260725/phase_states \
LLM_LOG_DIR=../results/s24_simple_production_checkpoint_v2_contract_20260725/llm_logs \
../.venv/bin/python run_ablation.py \
  --variants s_linker24_role_orchestrator \
  --datasets bigbluebutton teammates \
  --results-dir ../results/s24_simple_production_checkpoint_v2_contract_20260725
```

Result:

| Project | TP | FP | FN | F1 | F2 | Participant TP/FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BigBlueButton | 56 | 2 | 6 | 93.3 | 91.5 | 9 / 0 |
| TeamMates | 54 | 4 | 3 | 93.9 | 94.4 | 3 / 0 |

Focused controller-plus-participant prompts used 6,632 input tokens:
4,372 on BigBlueButton and 2,260 on TeamMates. The spike-014 surface used
49,284 on the same projects, so the reduction is 86.5%.
