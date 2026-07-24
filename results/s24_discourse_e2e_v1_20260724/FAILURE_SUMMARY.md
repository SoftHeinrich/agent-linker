# S24 discourse E2E v1 — superseded safety failure

Date: 2026-07-24

Configuration: OpenAI `gpt-5.6-terra`, reasoning effort `none`, fresh phase
state, process-local `OPENAI_API_KEY="$OAI_KEY"`.

Command:

```bash
OPENAI_API_KEY="$OAI_KEY" \
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_REASONING_EFFORT=none \
PHASE_CACHE_DIR=../results/s24_discourse_e2e_v1_20260724/phase_states \
LLM_LOG_DIR=../results/s24_discourse_e2e_v1_20260724/llm_logs \
  ../.venv/bin/python run_ablation.py \
  --variants s_linker21 s_linker24_role_orchestrator \
  --datasets mediastore teastore teammates bigbluebutton jabref \
  --results-dir ../results/s24_discourse_e2e_v1_20260724
```

Text result:

```text
S21:  macro F1/F2 89.9/91.1, pooled F1/F2 83.7/86.3
S24:  macro F1/F2 96.4/95.4, pooled F1/F2 95.0/93.7
```

This run is not promotion evidence. Fresh alias induction did not assign
`conversion process` to the entity path, allowing the role path to emit two
process-nominal false positives. The event-nominal ownership gate was added
and the complete E2E was rerun from fresh state in
`s24_discourse_e2e_v2_eventnominal_20260724`.
