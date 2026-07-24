# S24 discourse-scope participant resolution — final E2E

Date: 2026-07-24

Configuration:

- backend: `openai`
- model: `gpt-5.6-terra`
- reasoning effort: `none`
- credential: process-local `OPENAI_API_KEY="$OAI_KEY"` mapping; no value
  persisted
- fresh phase-state and LLM-log directories

Command:

```bash
OPENAI_API_KEY="$OAI_KEY" \
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_REASONING_EFFORT=none \
PHASE_CACHE_DIR=../results/s24_discourse_e2e_v2_eventnominal_20260724/phase_states \
LLM_LOG_DIR=../results/s24_discourse_e2e_v2_eventnominal_20260724/llm_logs \
  ../.venv/bin/python run_ablation.py \
  --variants s_linker21 s_linker24_role_orchestrator \
  --datasets mediastore teastore teammates bigbluebutton jabref \
  --results-dir ../results/s24_discourse_e2e_v2_eventnominal_20260724
```

Results:

| Variant | TP / FP / FN | Macro F1 | Pooled F1 | Macro F2 | Pooled F2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S21 | 174 / 16 / 21 | 93.33% | 90.39% | 92.69% | 89.69% |
| S24 discourse | 180 / 5 / 15 | 96.21% | 94.74% | 95.09% | 93.26% |
| Delta | +6 / -11 / -6 | +2.88 pp | +4.35 pp | +2.40 pp | +3.57 pp |

Per-project text output:

```text
mediastore    S21 F1/F2 98.4/97.4  S24 96.7/94.8
teastore      S21 F1/F2 98.1/97.0  S24 100.0/100.0
teammates     S21 F1/F2 86.0/89.0  S24 92.9/91.9
bigbluebutton S21 F1/F2 84.2/80.0  S24 91.5/88.8
jabref        S21 F1/F2 100.0/100.0 S24 100.0/100.0
```

Source audit:

```text
relation_role_resolution: 10 TP / 0 FP
  TeamMates: 3 TP / 0 FP (sentences 122, 138, 141)
  BigBlueButton: 7 TP / 0 FP (sentences 9, 10, 12, 13, 19, 76, 79)
```

The preceding unsafeguarded E2E admitted two process-nominal role false
positives when alias induction varied. The final implementation excludes
event-nominal terminal handles generically and the fresh production checkpoint
and final E2E both report zero role-source false positives.
