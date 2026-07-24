# S24 lexical-entity fresh paired E2E

All predictions were obtained from fresh OpenAI API calls using
`gpt-5.6-terra` with reasoning effort `none`. The host credential was mapped
process-locally from `OAI_KEY` to `OPENAI_API_KEY`; no credential value was
written to disk. Gold was consulted only after inference for evaluation.

## Commands

```bash
../.venv/bin/python pilot/test_s24_lexical_entity.py
../.venv/bin/python pilot/test_s24_orchestrator.py

OPENAI_API_KEY="$OAI_KEY" \
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_REASONING_EFFORT=none \
PHASE_CACHE_DIR=../results/s24_lexical_entity_e2e_v1_20260724/phase_states \
LLM_LOG_DIR=../results/s24_lexical_entity_e2e_v1_20260724/llm_logs \
  ../.venv/bin/python run_ablation.py \
  --variants s_linker21 s_linker24_role_orchestrator \
  --datasets mediastore teastore teammates bigbluebutton jabref \
  --results-dir ../results/s24_lexical_entity_e2e_v1_20260724
```

## Deterministic verification

```text
PASS: S24 lexical entity contracts
PASS: SLinker24RoleOrchestrator contracts
```

## Aggregate result

| Variant | TP / FP / FN | Macro F1 | Pooled F1 | Macro F2 | Pooled F2 | Calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S21 | 174 / 26 / 21 | 90.95% | 88.10% | 91.93% | 88.78% | 100 |
| S24 lexical entity | 180 / 11 / 15 | 95.29% | 93.26% | 94.75% | 92.69% | 112 |
| Delta | +6 / -15 / -6 | +4.34 pp | +5.16 pp | +2.82 pp | +3.91 pp | +12 |

## Per-project S24 result

| Project | TP / FP / FN | F1 | F2 | Lexical TP / FP |
| --- | --- | ---: | ---: | ---: |
| MediaStore | 29 / 0 / 2 | 96.67% | 94.77% | 0 / 0 |
| TeaStore | 27 / 0 / 0 | 100.00% | 100.00% | 0 / 0 |
| TEAMMATES | 53 / 5 / 4 | 92.17% | 92.66% | 0 / 0 |
| BigBlueButton | 53 / 6 / 9 | 87.60% | 86.32% | 2 / 0 |
| JabRef | 18 / 0 / 0 | 100.00% | 100.00% | 0 / 0 |

The lexical entity path recovered BigBlueButton sentences 30 and 78 for
`BBB web` from the surface form `bbb-web`. It is not a controller tool and has
no dedicated prompt; both candidates were validated in the existing entity
batch.

Raw runner result:
`ablation_20260724_224207.json`.
