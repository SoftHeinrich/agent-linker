# Compact S24 fresh paired five-project E2E

Configuration: OpenAI `gpt-5.6-terra`, reasoning effort `none`; fresh phase
cache for both variants.

Command:

```bash
OPENAI_API_KEY="$OAI_KEY" LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra OPENAI_REASONING_EFFORT=none \
PHASE_CACHE_DIR=../results/s24_simple_e2e_v1_contract_20260725/phase_states \
LLM_LOG_DIR=../results/s24_simple_e2e_v1_contract_20260725/llm_logs \
../.venv/bin/python run_ablation.py \
  --variants s_linker21 s_linker24_role_orchestrator \
  --datasets mediastore teammates teastore bigbluebutton jabref \
  --results-dir ../results/s24_simple_e2e_v1_contract_20260725
```

| Variant | TP | FP | FN | Macro F1 | Macro F2 | Pooled F1 | Pooled F2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S21 | 175 | 16 | 20 | 93.71 | 93.29 | 90.67 | 90.11 |
| compact S24 | 185 | 4 | 10 | 97.13 | 96.47 | 96.35 | 95.46 |

Compact S24 improves every aggregate. Per-project S24 F1/F2:

- MediaStore: 95.1 / 94.2
- TeamMates: 95.6 / 95.1
- TeaStore: 100.0 / 100.0
- BigBlueButton: 95.0 / 93.1
- JabRef: 100.0 / 100.0

The participant source contributes 13 TP / 0 FP: ten on BigBlueButton and
three on TeamMates.
